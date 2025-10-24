# src/evaluator.py
#
# By Ian Drumm, The University of Salford, UK.
#
import json
import os
import random
import numpy as np
from typing import List, Dict, Optional, Union
from tqdm import tqdm
from src.rag_chatbot import RAGCommentSimulator
from src.score_and_categorise_gpus import CommentScorer
import pandas as pd
from datetime import datetime
from langchain.schema import Document
from src.debug_log import DebugLogger
import traceback
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
import re
import warnings
import logging

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required, will use system environment variables

# Suppress specific transformers warnings
warnings.filterwarnings("ignore", message=".*Some weights of.*were not initialized.*")
warnings.filterwarnings("ignore", message=".*You should probably TRAIN this model.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Initialize global debug logger
_global_debug_logger = DebugLogger()

# Additional imports for new metrics
try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    print("Warning: bert_score not available. Install with: pip install bert_score")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence_transformers not available. Install with: pip install sentence-transformers")

try:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers torch")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("Warning: vaderSentiment not available. Install with: pip install vaderSentiment")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("Warning: textblob not available. Install with: pip install textblob")

from sklearn.metrics.pairwise import cosine_similarity


def create_llm(
    model_name: str,
    backend: Optional[str] = None,
    temperature: float = 0.1,
    timeout: int = 600,
    api_key: Optional[str] = None,
    **kwargs
):
    """
    Create an LLM instance based on backend and model name.
    
    Args:
        model_name: Name of the model (e.g., "gpt-4", "llama3.1:8b", "gemini-pro")
        backend: LLM backend ("openai", "gemini", "anthropic", "ollama", or None for auto-detect)
        temperature: Temperature for generation
        timeout: Request timeout in seconds
        api_key: API key for the backend (if None, uses environment variable)
        **kwargs: Additional backend-specific arguments
    
    Returns:
        LangChain LLM instance
    """
    # Auto-detect backend if not specified
    if backend is None:
        model_lower = model_name.lower()
        if model_lower.startswith("gpt-") or "gpt" in model_lower:
            backend = "openai"
        elif "gemini" in model_lower or "palm" in model_lower:
            backend = "gemini"
        elif "claude" in model_lower:
            backend = "anthropic"
        else:
            backend = "ollama"  # Default to Ollama for local models
    
    backend = backend.lower()
    
    # Prepare common arguments
    common_args = {
        "temperature": temperature,
        **kwargs
    }
    
    if backend == "openai":
        if api_key:
            common_args["openai_api_key"] = api_key
        elif "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        return ChatOpenAI(
            model=model_name,
            timeout=timeout,
            **common_args
        )
    
    elif backend == "gemini":
        if api_key:
            common_args["google_api_key"] = api_key
        elif "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        return ChatGoogleGenerativeAI(
            model=model_name,
            timeout=timeout,
            **common_args
        )
    
    elif backend == "anthropic":
        if api_key:
            common_args["anthropic_api_key"] = api_key
        elif "ANTHROPIC_API_KEY" not in os.environ:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
        
        return ChatAnthropic(
            model=model_name,
            timeout=timeout,
            **common_args
        )
    
    elif backend == "ollama":
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            timeout=timeout
        )
    
    else:
        raise ValueError(f"Unsupported backend: {backend}. Supported: openai, gemini, anthropic, ollama")


class AdvancedMetricsCalculator:
    """Helper class to calculate advanced similarity and quality metrics."""
    
    def __init__(self):
        self.sentence_model = None
        self.perplexity_model = None
        self.perplexity_tokenizer = None
        self.sentiment_analyzer = None
        self._WORD_RE = re.compile(r"[A-Za-z']+")
        self._SENT_RE = re.compile(r"[.!?]+")
        self._VOWELS = set("aeiouy")
        self._detox_model = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Loaded sentence transformer model for embedding similarity.")
            except Exception as e:
                print(f"Failed to load sentence transformer: {e}")
                self.sentence_model = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.perplexity_model = GPT2LMHeadModel.from_pretrained('gpt2')
                self.perplexity_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
                self.perplexity_model.eval()
                print("Loaded GPT-2 model for perplexity calculation.")
            except Exception as e:
                print(f"Failed to load GPT-2 for perplexity: {e}")
                self.perplexity_model = None
        
        if VADER_AVAILABLE:
            try:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
                print("Loaded VADER sentiment analyzer for emotional similarity.")
            except Exception as e:
                print(f"Failed to load VADER sentiment analyzer: {e}")
                self.sentiment_analyzer = None
    
    def calculate_bert_score(self, generated_text: str, reference_text: str) -> Dict[str, float]:
        """Calculate BERTScore between generated and reference text."""
        if not BERT_SCORE_AVAILABLE or not generated_text.strip() or not reference_text.strip():
            return {"bert_score_f1": None, "bert_score_precision": None, "bert_score_recall": None}
        
        try:
            P, R, F1 = bert_score([generated_text], [reference_text], model_type="distilbert-base-uncased", lang="en", verbose=False)
            return {
                "bert_score_f1": F1.item(),
                "bert_score_precision": P.item(),
                "bert_score_recall": R.item()
            }
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
            return {"bert_score_f1": None, "bert_score_precision": None, "bert_score_recall": None}
    
    def calculate_embedding_similarity(self, generated_text: str, reference_text: str) -> float:
        """Calculate cosine similarity between sentence embeddings."""
        if not self.sentence_model or not generated_text.strip() or not reference_text.strip():
            return None
        
        try:
            embeddings = self.sentence_model.encode([generated_text, reference_text])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating embedding similarity: {e}")
            return None
    
    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity of text using GPT-2."""
        if not self.perplexity_model or not self.perplexity_tokenizer or not text.strip():
            return None
        
        try:
            inputs = self.perplexity_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            with torch.no_grad():
                outputs = self.perplexity_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                return perplexity
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            return None
    
    def calculate_emotional_similarity(self, generated_text: str, reference_text: str) -> Dict[str, float]:
        """Calculate emotional similarity between texts using sentiment analysis."""
        if not self.sentiment_analyzer or not generated_text.strip() or not reference_text.strip():
            return {"emotional_similarity": None, "sentiment_difference": None}
        
        try:
            gen_sentiment = self.sentiment_analyzer.polarity_scores(generated_text)
            ref_sentiment = self.sentiment_analyzer.polarity_scores(reference_text)
            
            emotions = ['neg', 'neu', 'pos', 'compound']
            similarities = []
            
            for emotion in emotions:
                gen_score = gen_sentiment[emotion]
                ref_score = ref_sentiment[emotion]
                diff = abs(gen_score - ref_score)
                similarity = 1 - diff
                similarities.append(similarity)
            
            avg_emotional_similarity = np.mean(similarities)
            sentiment_difference = abs(gen_sentiment['compound'] - ref_sentiment['compound'])
            
            return {
                "emotional_similarity": float(avg_emotional_similarity),
                "sentiment_difference": float(sentiment_difference)
            }
        except Exception as e:
            print(f"Error calculating emotional similarity: {e}")
            return {"emotional_similarity": None, "sentiment_difference": None}

    def _words(self, text: str) -> List[str]:
        return self._WORD_RE.findall(text.lower())

    def _sentences(self, text: str) -> List[str]:
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sents if s]

    def _count_syllables_in_word(self, word: str) -> int:
        """Heuristic syllable counter."""
        w = re.sub(r"[^a-z]", "", word.lower())
        if not w:
            return 0
        if w.endswith("e"):
            w = w[:-1] or w
        count = 0
        prev_vowel = False
        for ch in w:
            is_vowel = ch in self._VOWELS
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        return max(count, 1)

    def _count_syllables(self, words: List[str]) -> int:
        return sum(self._count_syllables_in_word(w) for w in words)

    def calculate_toxicity_scores(self, text: str) -> Dict[str, float]:
        """Returns toxicity-related scores in [0,1]."""
        try:
            if self._detox_model is None:
                from detoxify import Detoxify
                self._detox_model = Detoxify('original')
            preds = self._detox_model.predict(text or "")
            out = {}
            mapping = {
                'toxicity': 'toxicity',
                'severe_toxicity': 'severe_toxicity',
                'insult': 'insult',
                'threat': 'threat',
                'obscene': 'obscene',
                'identity_attack': 'identity_attack',
                'identity_hate': 'identity_attack'
            }
            for k, v in preds.items():
                if k in mapping:
                    out[mapping[k]] = float(v)
            if 'toxicity' not in out:
                out['toxicity'] = float(preds.get('toxicity', 0.0))
            return out
        except Exception:
            lexicon = {
                "idiot","stupid","moron","dumb","shut up","trash","garbage","hate","loser",
                "suck","crap","bastard","hell","damn","fool","jerk","pathetic","disgusting"
            }
            text_l = (text or "").lower()
            toks = self._words(text_l)
            if not toks:
                return {"toxicity": 0.0}
            hits = 0
            for w in toks:
                if w in lexicon:
                    hits += 1
            for phrase in ["shut up"]:
                if phrase in text_l:
                    hits += 1
            proxy = min(hits / max(len(toks), 1), 1.0)
            return {"toxicity": proxy}

    def calculate_readability_metrics(self, text: str) -> Dict[str, float]:
        """Returns readability metrics."""
        words = self._words(text or "")
        sents = self._sentences(text or "")
        n_words = len(words)
        n_sents = max(len(sents), 1)
        n_syll = self._count_syllables(words)

        avg_sentence_length = n_words / n_sents
        avg_word_length = sum(len(w) for w in words) / max(n_words, 1)
        words_per_sentence = n_words / n_sents
        syllables_per_word = n_syll / max(n_words, 1)

        flesch_reading_ease = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
        flesch_kincaid_grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59

        return {
            "flesch_reading_ease": float(flesch_reading_ease),
            "flesch_kincaid_grade": float(flesch_kincaid_grade),
            "avg_sentence_length": float(avg_sentence_length),
            "avg_word_length": float(avg_word_length),
        }

    def _ttr(self, tokens: List[str]) -> float:
        n = len(tokens)
        if n == 0:
            return 0.0
        return len(set(tokens)) / n

    def _mtld(self, tokens: List[str], ttr_threshold: float = 0.72, min_seg: int = 10) -> float:
        """Measure of Textual Lexical Diversity (MTLD)."""
        if len(tokens) < min_seg:
            return float('nan')

        def _mtld_calc(seq: List[str]) -> float:
            factors = 0
            types = set()
            tokens_count = 0
            for tok in seq:
                tokens_count += 1
                types.add(tok)
                if (len(types) / tokens_count) <= ttr_threshold and tokens_count >= min_seg:
                    factors += 1
                    types = set()
                    tokens_count = 0
            remainder = tokens_count / max(len(types), 1) if tokens_count > 0 else 0
            return len(seq) / (factors + remainder if (factors + remainder) > 0 else float('inf'))

        forward = _mtld_calc(tokens)
        backward = _mtld_calc(list(reversed(tokens)))
        return (forward + backward) / 2.0

    def calculate_lexical_diversity(self, text: str) -> Dict[str, float]:
        """Returns lexical diversity metrics."""
        toks = self._words(text or "")
        if not toks:
            return {"ttr": 0.0, "mtld": float('nan'), "hapax_ratio": 0.0}
        ttr = self._ttr(toks)
        mtld = self._mtld(toks)
        from collections import Counter
        freqs = Counter(toks)
        hapax = sum(1 for _, c in freqs.items() if c == 1)
        hapax_ratio = hapax / len(toks)
        return {"ttr": float(ttr), "mtld": float(mtld), "hapax_ratio": float(hapax_ratio)}

    def calculate_all_metrics(self, generated_text: str, reference_text: str) -> Dict[str, float]:
        """Calculate all advanced metrics for a pair of texts."""
        metrics = {}
        
        bert_scores = self.calculate_bert_score(generated_text, reference_text)
        metrics.update(bert_scores)
        metrics["embedding_similarity"] = self.calculate_embedding_similarity(generated_text, reference_text)
        metrics["generated_perplexity"] = self.calculate_perplexity(generated_text)
        metrics["reference_perplexity"] = self.calculate_perplexity(reference_text)
        
        emotional_metrics = self.calculate_emotional_similarity(generated_text, reference_text)
        metrics.update(emotional_metrics)

        gen_tox = self.calculate_toxicity_scores(generated_text)
        ref_tox = self.calculate_toxicity_scores(reference_text)

        print("\n\nEmotional Similarity Metrics:")
        print(emotional_metrics)
        print("Generated Toxicity Scores:")
        print(gen_tox)

        metrics.update({f"real_{k}": v for k, v in gen_tox.items()})
        metrics.update({f"fake_{k}": v for k, v in ref_tox.items()})

        gen_read = self.calculate_readability_metrics(generated_text)
        ref_read = self.calculate_readability_metrics(reference_text)
        metrics.update({f"real_{k}": v for k, v in gen_read.items()})
        metrics.update({f"fake_{k}": v for k, v in ref_read.items()})

        gen_lex = self.calculate_lexical_diversity(generated_text)
        ref_lex = self.calculate_lexical_diversity(reference_text)
        metrics.update({f"real_{k}": v for k, v in gen_lex.items()})
        metrics.update({f"fake_{k}": v for k, v in ref_lex.items()})

        return metrics


class EnhancedCommentScorer(CommentScorer):
    """Enhanced CommentScorer with alignment scoring and categorization capabilities."""
    
    def __init__(self, 
        features_path: str, 
        model: str,
        backend="ollama_async",
        scorer_backend: Optional[str] = None, 
        api_key: Optional[str] = None, 
        openai_api_key: Optional[str] = None,
        **kwargs):
        """
        Initialize EnhancedCommentScorer with LLM support.
        
        Args:
            features_path: Path to features JSON file
            model: Model name for scoring
            backend: Backend string for parent class (e.g., "ollama_async")
            scorer_backend: LLM backend (openai, gemini, anthropic, ollama, or None for auto-detect)
            api_key: API key for the backend
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(features_path=features_path, model=model, backend=backend, openai_api_key=openai_api_key, **kwargs)

        # Ensure self.llm is set - create it if parent class didn't
        if not hasattr(self, 'llm') or self.llm is None:
            print(f"Initializing LLM for EnhancedCommentScorer: model={model}, backend={scorer_backend}")
            self.llm = create_llm(
                model_name=model,
                backend=scorer_backend,
                temperature=0.1,
                timeout=600,
                api_key=api_key
            )
            print(f"LLM initialized successfully: {type(self.llm).__name__}")
        else:
            print(f"Using existing LLM from parent class: {type(self.llm).__name__}")
    
    def score_batch(self, items: List[Dict], include_justifications: bool = True, 
                    features_filter: Optional[List[str]] = None) -> List[Dict]:
        """Score a batch of items, optionally filtering to specific features."""
        original_features = None

        if features_filter:
            original_features = self.features.copy()
            filtered_features = []
            for f in self.features:
                if hasattr(f, 'feature'):
                    feature_name = f.feature
                elif isinstance(f, dict):
                    feature_name = f.get("feature")
                else:
                    continue
                filtered_features.append(f)
            self.features = filtered_features

        _global_debug_logger.write_file("Features",[(f.key, f.type) for f in self.features])
        _global_debug_logger.write_file("Items in evaluator score batch", items)

        try:
            return super().score_batch(items, include_justifications, features_filter=features_filter)
        finally:
            if original_features is not None:
                self.features = original_features
    
    def _extract_category_from_response(self, response_text: str, valid_options: List[str]) -> Dict:
        """
        Extract category and justification from LLM response.
        
        Args:
            response_text: Raw LLM response
            valid_options: List of valid category options
            
        Returns:
            Dict with 'category' and optionally 'justification'
        """
        result = {}
        
        # Try to parse as JSON first
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if "category" in parsed:
                    result["category"] = parsed["category"]
                if "justification" in parsed:
                    result["justification"] = parsed["justification"]
                return result
            except json.JSONDecodeError:
                pass
        
        # Fallback: Look for category mentions in text
        response_lower = response_text.lower()
        for option in valid_options:
            option_lower = option.lower()
            # Look for explicit category statements
            patterns = [
                f'"category":\\s*"{option}"',
                f'category:\\s*{option}',
                f'classified as:?\\s*{option_lower}',
                f'category is:?\\s*{option_lower}',
                f'\\b{option_lower}\\b'
            ]
            
            for pattern in patterns:
                if re.search(pattern, response_lower):
                    result["category"] = option
                    break
            
            if "category" in result:
                break
        
        # Extract justification if present
        justification_patterns = [
            r'"justification":\s*"([^"]+)"',
            r'justification:\s*([^\n]+)',
            r'because\s+([^\n]+)',
            r'explanation:\s*([^\n]+)'
        ]
        
        for pattern in justification_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                result["justification"] = match.group(1).strip()
                break
        
        # If no category found, return first option as default
        if "category" not in result:
            result["category"] = valid_options[0] if valid_options else "Unknown"
            result["justification"] = "Could not determine category from response"
        
        return result

    def score_alignment_comparison(self, comparison_item: Dict, 
                                 features_filter: Optional[List[str]] = None) -> Dict:
        """Score alignment features by comparing generated comment to original comment."""
        post = comparison_item["post"]
        original_comment = comparison_item["original_comment"]
        generated_comment = comparison_item["generated_comment"]
        
        alignment_features = []
        for f in self.features:
            feature_type = None
            feature_name = None
            
            if hasattr(f, 'type') and hasattr(f, 'feature'):
                feature_type = f.type
                feature_name = f.feature
            elif isinstance(f, dict):
                feature_type = f.get("type")
                feature_name = f.get("feature")
            else:
                continue
            
            if (feature_type == "alignment_score" and 
                (features_filter is None or feature_name in features_filter)):
                alignment_features.append(f)
        
        if not alignment_features:
            return {"scores": {}}
        
        prompt_parts = [
            f"Post: {post}",
            f"Original Comment: {original_comment}",
            f"Generated Comment: {generated_comment}",
            "",
            "Please evaluate how well the Generated Comment aligns with the Original Comment in the context of the given Post.",
            "Rate each feature on the specified scale:",
            ""
        ]
        
        for feature in alignment_features:
            if hasattr(feature, 'feature') and hasattr(feature, 'rubric'):
                feature_name = feature.feature
                rubric = feature.rubric
            elif isinstance(feature, dict):
                feature_name = feature.get("feature", "unknown")
                rubric = feature.get("rubric", "")
            else:
                continue
            prompt_parts.append(f"{feature_name}: {rubric}")
        
        prompt_parts.extend([
            "",
            "Provide your scores in the following JSON format:",
            "{"
        ])
        
        for i, feature in enumerate(alignment_features):
            if hasattr(feature, 'feature'):
                feature_name = feature.feature
            elif isinstance(feature, dict):
                feature_name = feature.get("feature", "unknown")
            else:
                continue
            feature_key = feature_name.lower().replace(" ", "_")
            comma = "," if i < len(alignment_features) - 1 else ""
            prompt_parts.append(f'  "{feature_key}": <score>{comma}')
        
        prompt_parts.append("}")
        full_prompt = "\n".join(prompt_parts)
        
        try:
            response = self.llm.invoke(full_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            scores = self._extract_json_from_response(response_text)
            return {"scores": scores}
        except Exception as e:
            print(f"Error in alignment scoring: {e}")
            return {"scores": {}}

    def _extract_json_from_response(self, response_text: str) -> Dict:
        """Extract JSON scores from LLM response."""
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                return parsed
            except json.JSONDecodeError as e:
                print(f"DEBUG: JSON parsing failed: {e}")
        
        scores = {}
        lines = response_text.split('\n')
        
        for line in lines:
            match = re.search(r'"([^"]+)":\s*(\d+)', line)
            if match:
                feature_name = match.group(1)
                score = int(match.group(2))
                scores[feature_name] = score
                continue
            
            match = re.search(r'([a-zA-Z_][a-zA-Z0-9_\s]*?):\s*(\d+)', line)
            if match:
                feature_name = match.group(1).strip()
                score = int(match.group(2))
                scores[feature_name] = score
                continue
        
        return scores


class ChatbotEvaluator:
    def __init__(
        self,
        vector_db_path: str,
        evaluation_prompts_path: str,
        chat_model: str = "llama3.1:8b",
        scorer_model: str = "llama3.1:8b",
        backend: str = "ollama_async",
        embed_model: str = "nomic-embed-text:latest",
        chat_backend: Optional[str] = None,
        scorer_backend: Optional[str] = None,
        embed_backend: Optional[str] = None,
        chat_api_key: Optional[str] = None,
        scorer_api_key: Optional[str] = None,
        embed_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the ChatbotEvaluator with support for multiple LLM backends.
        
        Args:
            vector_db_path: Path to the vector database
            evaluation_prompts_path: Path to evaluation prompts JSON
            chat_model: Model name for chat generation (default: "llama3.1:8b")
            scorer_model: Model name for scoring (default: "llama3.1:8b")
            backend: Backend string for parent classes (e.g., "ollama_async") - DEPRECATED, use specific backends
            embed_model: Embedding model name (default: "nomic-embed-text:latest")
            chat_backend: Backend for chat model ("openai", "gemini", "anthropic", "ollama", or None for auto-detect)
            scorer_backend: Backend for scorer model (same options as chat_backend)
            embed_backend: Backend for embedding model ("openai", "gemini", "ollama", or None for auto-detect)
            chat_api_key: API key for chat model backend
            scorer_api_key: API key for scorer model backend
            embed_api_key: API key for embedding model backend
            openai_api_key: DEPRECATED - use chat_api_key, scorer_api_key, or embed_api_key instead
        """
        # Initialize RAGCommentSimulator with full backend support
        self.rag_chatbot = RAGCommentSimulator(
            embed_model=embed_model,
            chat_model=chat_model,
            persist_directory=vector_db_path,
            backend=chat_backend,
            chat_api_key=chat_api_key,
            embed_backend=embed_backend,
            embed_api_key=embed_api_key
        )
        
        # Determine the backend to use for scorer
        # Priority: scorer_backend > backend (for backward compatibility)
        effective_scorer_backend = scorer_backend if scorer_backend is not None else backend
        
        # Initialize scorer with backend support
        self.eval_scorer = EnhancedCommentScorer(
            features_path=evaluation_prompts_path,
            model=scorer_model,
            backend=effective_scorer_backend,
            scorer_backend=scorer_backend,
            api_key=scorer_api_key,
            openai_api_key=openai_api_key  # Keep for backward compatibility
        )
        
        self.vector_db_path = vector_db_path
        self.evaluation_prompts_path = evaluation_prompts_path
        self.metrics_calculator = AdvancedMetricsCalculator()
        
        # Store backend information for later use
        self.chat_model = chat_model
        self.scorer_model = scorer_model
        self.embed_model = embed_model
        self.chat_backend = chat_backend
        self.scorer_backend = scorer_backend
        self.embed_backend = embed_backend
        self.chat_api_key = chat_api_key
        self.scorer_api_key = scorer_api_key
        self.embed_api_key = embed_api_key
        
        print(f"Initialized ChatbotEvaluator:")
        print(f"  - Embedding: {embed_model} (backend: {embed_backend or 'auto-detect'})")
        print(f"  - Chat: {chat_model} (backend: {chat_backend or 'auto-detect'})")
        print(f"  - Scorer: {scorer_model} (backend: {scorer_backend or 'auto-detect'})")

    def _load_comment_ids_from_csv(self, csv_path: str) -> List[str]:
        """
        Load comment IDs from a CSV file.
        
        Args:
            csv_path: Path to CSV file containing 'original_comment_id' column
            
        Returns:
            List of comment IDs as strings
        """
        try:
            df = pd.read_csv(csv_path)
            
            if 'original_comment_id' not in df.columns:
                raise ValueError(
                    f"CSV file must contain 'original_comment_id' column. "
                    f"Found columns: {', '.join(df.columns)}"
                )
            
            # Extract comment IDs and remove any NaN values
            comment_ids = df['original_comment_id'].dropna().astype(str).tolist()
            
            print(f"Loaded {len(comment_ids)} comment IDs from {csv_path}")
            return comment_ids
            
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_path}")
            return []
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return []

    def _load_data_from_db(
        self,
        sample_size: int,
        target_cluster_id: Optional[int] = None,
        target_category: Optional[str] = None,
        target_category_framework: Optional[str] = None,
        search_query: Optional[str] = None,
        source: str = "json",
        comment_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Loads and samples real post-comment pairs with optional filtering.
        
        Args:
            sample_size: Number of samples to return (ignored if comment_ids provided)
            target_cluster_id: Filter by cluster ID
            target_category: Filter by category
            target_category_framework: Framework for category filtering
            search_query: Filter by search query
            source: Data source ("json" or "vector_db")
            comment_ids: List of specific comment IDs to load (overrides sampling)
        """
        full_data = []

        if source == "json":
            try:
                with open("data/clustered_data.json", "r") as f:
                    full_data = json.load(f)
            except FileNotFoundError:
                print("Error: data/clustered_data.json not found.")
                return []
                
        elif source == "vector_db":
            print(f"Searching vector database {self.vector_db_path}")

            where_conditions = []
            
            # If specific comment IDs provided, filter by those
            if comment_ids:
                # ChromaDB uses $in operator for list matching
                where_conditions.append({"comment_id": {"$in": comment_ids}})
                print(f"Filtering to {len(comment_ids)} specific comment IDs")
            
            if target_cluster_id is not None:
                where_conditions.append({"cluster_id": {"$eq": target_cluster_id}})
            if search_query is not None:
                where_conditions.append({"search_query": {"$eq": search_query}})
            
            if target_category_framework is not None and target_category is not None:
                category_key = f"category_{target_category_framework}"
                where_conditions.append({category_key: {"$eq": target_category}})
            elif target_category is not None:
                where_conditions.append({"category": {"$eq": target_category}})

            where_filter = None
            if len(where_conditions) == 1:
                where_filter = where_conditions[0]
            elif len(where_conditions) > 1:
                where_filter = {"$and": where_conditions}

            try:
                # Increase limit if we're looking for specific IDs
                fetch_limit = len(comment_ids) if comment_ids else sample_size
                
                results = self.rag_chatbot.vector_store._collection.get(
                    where=where_filter,
                    limit=fetch_limit
                )
                
                if not results['ids']:
                    print(f"Warning: No documents found matching filters: {where_filter}")
                    return []

                for i, doc_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i]
                    document_text = results['documents'][i]
                    try:
                        post_text = document_text.split("Post: ", 1)[1].split("\n\nComment:", 1)[0].strip()
                        comment_text = document_text.split("Comment: ", 1)[1].strip()
                    except IndexError:
                        continue
                    reconstructed_item = {
                        "post": post_text,
                        "comment": comment_text,
                        "post_metadata": {"search_query": metadata.get("search_query")},
                        "comment_metadata": {"id": doc_id},
                        "cluster_id": metadata.get("cluster_id"),
                        "category": metadata.get("category"),
                        "scores_json": metadata.get("scores_json", {}),
                        "categories_json": metadata.get("categories_json", {}),
                        "justifications_json": metadata.get("justifications_json", {})
                    }
                    full_data.append(reconstructed_item)

                print(f"Found {len(full_data)} items in vector database")

            except Exception as e:
                print(f"Error accessing vector database: {e}")
                return []
        else:
            print(f"Error: Invalid source '{source}'")
            return []

        if not full_data:
            print("No data available after initial loading/filtering.")
            return []
        
        # If specific comment IDs provided, filter to only those IDs
        if comment_ids and source == "json":
            initial_count = len(full_data)
            comment_id_set = set(comment_ids)
            full_data = [
                item for item in full_data 
                if str(item.get("comment_metadata", {}).get("id")) in comment_id_set
            ]
            print(f"Filtered to {len(full_data)} items matching provided comment IDs (from {initial_count} total)")
            
            # If we have specific IDs, don't apply other filters or sampling
            if full_data:
                return full_data
        
        # Apply other filters only if we're not using specific comment IDs
        if not comment_ids:
            if target_cluster_id is not None and source == "json":
                initial_count = len(full_data)
                full_data = [item for item in full_data if item.get("cluster_id") == target_cluster_id]
                print(f"Filtered to cluster {target_cluster_id}: {len(full_data)} items from {initial_count}")

            if target_category is not None and source == "json":
                initial_count = len(full_data)
                filtered_data = []
                
                for item in full_data:
                    categories = item.get("categories_json", {})
                    if isinstance(categories, str):
                        try:
                            categories = json.loads(categories)
                        except json.JSONDecodeError:
                            categories = {}
                    
                    if isinstance(categories, dict):
                        if target_category_framework is not None:
                            framework_value = categories.get(target_category_framework)
                            if framework_value is not None and str(framework_value).lower() == target_category.lower():
                                filtered_data.append(item)
                        else:
                            if any(str(cat_value).lower() == target_category.lower() 
                                   for cat_value in categories.values()):
                                filtered_data.append(item)
                
                full_data = filtered_data
                
                if target_category_framework:
                    print(f"Filtered to category '{target_category}' in framework '{target_category_framework}': {len(full_data)} from {initial_count}")
                else:
                    print(f"Filtered to category '{target_category}': {len(full_data)} from {initial_count}")

            if search_query is not None and source == "json":
                initial_count = len(full_data)
                full_data = [item for item in full_data if item.get("post_metadata", {}).get("search_query") == search_query]
                print(f"Filtered to search query '{search_query}': {len(full_data)} from {initial_count}")

            # Apply sampling only if not using specific comment IDs
            if len(full_data) < sample_size:
                print(f"Warning: Sample size ({sample_size}) > available data ({len(full_data)}). Using all data.")
                return full_data
            
            return random.sample(full_data, sample_size)
        
        return full_data

    def score_alignment_direct(self, comparison_item: Dict, alignment_features: List[Dict]) -> Dict:
        """Score alignment features directly using the configured LLM."""
        if not alignment_features:
            return {"scores": {}}
        
        post = comparison_item["post"]
        original_comment = comparison_item["original_comment"]
        generated_comment = comparison_item["generated_comment"]
        
        prompt_parts = [
            f"Post: {post}",
            f"Original Comment: {original_comment}",
            f"Generated Comment: {generated_comment}",
            "",
            "Please evaluate how well the Generated Comment aligns with the Original Comment.",
            "Rate each feature on the specified scale:",
            ""
        ]
        
        for feature in alignment_features:
            feature_name = feature.get("feature", "unknown")
            rubric = feature.get("rubric", "")
            prompt_parts.append(f"{feature_name}: {rubric}")
        
        prompt_parts.extend(["", "Provide your scores in the following JSON format:", "{"])
        
        for i, feature in enumerate(alignment_features):
            feature_name = feature.get("feature", "unknown")
            feature_key = feature_name.lower().replace(" ", "_").replace(" with ", "_")
            comma = "," if i < len(alignment_features) - 1 else ""
            prompt_parts.append(f'  "{feature_key}": <score>{comma}')
        
        prompt_parts.append("}")
        full_prompt = "\n".join(prompt_parts)
        
        try:
            # Create LLM instance with backend support
            temp_llm = create_llm(
                model_name=self.scorer_model,
                backend=self.scorer_backend,
                temperature=0.1,
                timeout=600,
                api_key=self.scorer_api_key
            )
            
            response = temp_llm.invoke(full_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            scores = self._extract_json_from_response_direct(response_text)
            return {"scores": scores}
        except Exception as e:
            print(f"ERROR: Exception during alignment scoring: {e}")
            traceback.print_exc()
            return {"scores": {}}

    def _extract_json_from_response_direct(self, response_text: str) -> Dict:
        """Extract JSON scores from LLM response."""
        scores = {}
        
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            try:
                scores = json.loads(json_match.group())
                if scores:
                    return scores
            except json.JSONDecodeError as e:
                print(f"ERROR: JSON parsing failed: {e}")
        
        lines = response_text.split('\n')
        for line in lines:
            match = re.search(r'"([^"]+)":\s*(\d+)', line)
            if match:
                feature_name = match.group(1)
                score = int(match.group(2))
                scores[feature_name] = score
                continue
            
            match = re.search(r'([a-zA-Z_][a-zA-Z0-9_\s]*?):\s*(\d+)', line)
            if match:
                feature_name = match.group(1).strip()
                score = int(match.group(2))
                scores[feature_name] = score
                continue
        
        return scores

    def calculate_robust_statistics(self, results: List[Dict]) -> Dict:
        """Calculate median and IQR for perplexity metrics to handle outliers."""
        real_perps = []
        bot_perps = []
        
        for res in results:
            adv_metrics = res.get("advanced_metrics", {})
            if adv_metrics.get("reference_perplexity") is not None:
                real_perps.append(adv_metrics["reference_perplexity"])
            if adv_metrics.get("generated_perplexity") is not None:
                bot_perps.append(adv_metrics["generated_perplexity"])
        
        stats = {}
        
        if real_perps:
            stats["real_perplexity_median"] = float(np.median(real_perps))
            stats["real_perplexity_q25"] = float(np.percentile(real_perps, 25))
            stats["real_perplexity_q75"] = float(np.percentile(real_perps, 75))
            stats["real_perplexity_mean"] = float(np.mean(real_perps))
            stats["real_perplexity_std"] = float(np.std(real_perps))
            stats["real_perplexity_max"] = float(np.max(real_perps))
            stats["real_perplexity_n"] = len(real_perps)
        
        if bot_perps:
            stats["bot_perplexity_median"] = float(np.median(bot_perps))
            stats["bot_perplexity_q25"] = float(np.percentile(bot_perps, 25))
            stats["bot_perplexity_q75"] = float(np.percentile(bot_perps, 75))
            stats["bot_perplexity_mean"] = float(np.mean(bot_perps))
            stats["bot_perplexity_std"] = float(np.std(bot_perps))
            stats["bot_perplexity_max"] = float(np.max(bot_perps))
            stats["bot_perplexity_n"] = len(bot_perps)
        
        return stats

    def _log_first_prompt(self, prompt: str) -> None:
        """Log first prompt to file"""
        import datetime
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"eval_first_prompt_{timestamp}.log")
        
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"FIRST CHATBOT PROMPT\n")
            f.write("=" * 70 + "\n")
            f.write(f"Chat Model: {self.chat_model}\n")
            f.write(f"Chat Backend: {self.chat_backend or 'auto-detect'}\n")
            f.write(f"Embed Model: {self.embed_model}\n")
            f.write(f"Embed Backend: {self.embed_backend or 'auto-detect'}\n")
            f.write(f"PROMPT:\n{prompt}\n")

    def format_docs(self, docs):
        formatted = []
        for doc in docs:
            meta = doc.metadata
            content = doc.page_content.strip().replace("\n", " ")
            formatted.append(
                f"- ID: {doc.id}\n"
                f"  Subreddit: {meta.get('subreddit')}\n"
                f"  Date: {meta.get('created_utc')}\n"
                f"  Category: {meta.get('category_climate_attitude_uk')}\n"
                f"  Comment: {content}\n"
            )
        return "\n".join(formatted)

    def evaluate(
        self, 
        sample_size: int = 100, 
        cluster_id: Optional[int] = None,
        category: Optional[str] = None,
        category_framework: Optional[str] = None,
        search_query: Optional[str] = None, 
        source_data: str = "json", 
        batch_size: int = 50, 
        include_justifications: bool = False,
        categorize_comments: bool = False,
        category_features_path: Optional[str] = None,
        comment_ids_csv: Optional[str] = None
    ) -> List[Dict]:
        """
        Evaluate the chatbot by generating comments for sampled post/comment pairs and scoring them.
        
        Args:
            sample_size: Number of samples to evaluate (ignored if comment_ids_csv provided)
            cluster_id: Optional cluster ID to filter by
            category: Optional category to filter by
            category_framework: Optional framework for category filtering
            search_query: Optional search query to filter by
            source_data: Source of data ("json" or "vector_db")
            batch_size: Batch size for scoring
            include_justifications: Whether to include justifications
            categorize_comments: Whether to categorize generated comments
            category_features_path: Path to category features JSON
            comment_ids_csv: Path to CSV file with 'original_comment_id' column
        
        Returns:
            List of evaluation results
        """
        
        first_flag = True

        if category_framework is not None and category is None:
            print("Warning: category_framework specified without category. Framework filter will be ignored.")
            print("Please specify both --target_category and --target_category_framework together.")
            category_framework = None
        
        # Load comment IDs from CSV if provided
        specific_comment_ids = None
        if comment_ids_csv:
            specific_comment_ids = self._load_comment_ids_from_csv(comment_ids_csv)
            if not specific_comment_ids:
                print("No valid comment IDs found in CSV. Exiting.")
                return []
            print(f"Will evaluate {len(specific_comment_ids)} specific comments from CSV")
        
        sampled_data = self._load_data_from_db(
            sample_size, 
            target_cluster_id=cluster_id,
            target_category=category,
            target_category_framework=category_framework,
            search_query=search_query, 
            source=source_data,
            comment_ids=specific_comment_ids
        )
        
        if not sampled_data:
            if comment_ids_csv:
                print("No data found for the provided comment IDs.")
            return []

        print(f"Loaded {len(sampled_data)} comments for evaluation")
        if comment_ids_csv:
            print(f"Using comments specified in: {comment_ids_csv}")
        if category_framework and category:
            print(f"Filtered to category '{category}' in framework '{category_framework}'")
        elif category:
            print(f"Filtered to category '{category}' (all frameworks)")
        if include_justifications:
            print("Will include justifications with scoring")
        if categorize_comments:
            print("Will categorize generated comments")
        
        processed_items = []
        for item in tqdm(sampled_data, desc="Generating fake comments"):
            original_post = item.get("post", "")
            original_comment = item.get("comment", "")
            
            if not original_post or not original_comment:
                print(f"Skipping item due to missing post or comment: {item.get('comment_metadata', {}).get('id', 'N/A')}")
                continue
            
            item_id_for_exclusion = item.get("comment_metadata", {}).get("id")
            item_cluster_id = item.get("cluster_id")
            
            rag_categories = None
            if category_framework and category:
                rag_categories = {category_framework: category}
                print(f"Generating with RAG filter: {rag_categories}")
            
            generated_response = self.rag_chatbot.generate_reply(
                original_post,
                cluster_id=item_cluster_id,
                exclude_doc_id=item_id_for_exclusion,
                categories=rag_categories
            )

            if first_flag==True:
                prompt_text = generated_response.get("full_prompt", "")
                retrieved_docs = generated_response.get("source_documents", [])
                self._log_first_prompt(prompt_text + "\n\n---\n\nRetrieved docs:\n" + self.format_docs(retrieved_docs))
                first_flag=False

            fake_comment = generated_response.get("result", "")
            mmr_params = generated_response.get("mmr_params", {})
            
            prefixes_to_remove = [
                "Comment:\n\n", "Here's a new comment:\n\n", "Comment: ",
                "Here's a new comment: ", "New comment: "
            ]
            for prefix in prefixes_to_remove:
                if fake_comment.startswith(prefix):
                    fake_comment = fake_comment[len(prefix):]
                    break
            fake_comment = fake_comment.strip()
            
            # Extract IDs from metadata
            post_id = item.get("post_metadata", {}).get("id") or item.get("post_metadata", {}).get("post_id")
            comment_id = item.get("comment_metadata", {}).get("id") or item.get("comment_metadata", {}).get("comment_id")

            processed_items.append({
                "original_post": original_post,
                "original_comment": original_comment,
                "fake_comment": fake_comment,
                "original_post_id": post_id,
                "original_comment_id": comment_id,
                "evaluated_cluster_id": item_cluster_id,
                "rag_categories_used": rag_categories,
                "rag_filters_applied": mmr_params.get("filters_applied"),
                "rag_category_filters": mmr_params.get("category_filters"),
                "rag_cluster_ids_used": mmr_params.get("cluster_ids_used"),
                "rag_final_doc_count": mmr_params.get("final_doc_count"),
                "item": item,
                "original_scores": item.get("scores_json", {}),
                "original_categories": item.get("categories_json", {}),
                "original_justifications": item.get("justifications_json", {})
            })

        print(f"Generated fake comments for {len(processed_items)} items")
        
        if not processed_items:
            return []

        print("Load evaluation features")

        try:
            with open(self.evaluation_prompts_path, "r") as f:
                evaluation_features = json.load(f)
        except FileNotFoundError:
            print(f"Error: Evaluation prompts file not found at {self.evaluation_prompts_path}")
            return []

        score_features = [f for f in evaluation_features if f.get("type") == "score"]
        alignment_features = [f for f in evaluation_features if f.get("type") == "alignment_score"]
        category_features = [f for f in evaluation_features if f.get("type") == "category"]

        print(f"Found {len(score_features)} score features and {len(alignment_features)} alignment features")
        
        if alignment_features:
            print("Alignment features found:")
            for f in alignment_features:
                print(f"  - {f.get('feature', 'unknown')}")
        else:
            print("Warning: No alignment features found in evaluation_features")
            feature_types = set(f.get("type") for f in evaluation_features)
            print(f"Available feature types: {feature_types}")
        
        print(f"Scoring {len(processed_items)} comment pairs in batches of {batch_size}...")
        
        all_results = []
        
        for i in range(0, len(processed_items), batch_size):
            batch_items = processed_items[i:i + batch_size]
            batch_results = []
            
            real_scores_batch = []
            fake_scores_batch = []
            alignment_scores_batch = []
            real_categories_batch = []
            fake_categories_batch = []
            
            if score_features or category_features:
                print(f"Processing batch {i//batch_size + 1}: Scoring individual comments...")
                
                real_comment_batch = []
                fake_comment_batch = []
                
                for proc_item in batch_items:
                    real_comment_batch.append({
                        "post": proc_item["original_post"],
                        "original_comment": proc_item["original_comment"],
                        "comment": proc_item["original_comment"],
                        "post_metadata": proc_item["item"].get("post_metadata", {}),
                        "comment_metadata": proc_item["item"].get("comment_metadata", {})
                    })
                    
                    fake_comment_batch.append({
                        "post": proc_item["original_post"],
                        "original_comment": proc_item["original_comment"],
                        "comment": proc_item["fake_comment"],
                        "post_metadata": proc_item["item"].get("post_metadata", {}),
                        "comment_metadata": {"type": "generated"}
                    })
                
                if score_features:
                    real_scores_batch = self.eval_scorer.score_batch(
                        real_comment_batch,
                        include_justifications=include_justifications,
                        features_filter=[f["feature"] for f in score_features]
                    )
                    
                    fake_scores_batch = self.eval_scorer.score_batch(
                        fake_comment_batch,
                        include_justifications=include_justifications,
                        features_filter=[f["feature"] for f in score_features]
                    )
                else:
                    real_scores_batch = [{"scores": {}} for _ in batch_items]
                    fake_scores_batch = [{"scores": {}} for _ in batch_items]

                if alignment_features:
                    alignment_scores_batch = self.eval_scorer.score_batch(
                        fake_comment_batch,
                        include_justifications=include_justifications,
                        features_filter=[f["feature"] for f in alignment_features]
                    )
                else:
                    alignment_scores_batch = [{"scores": {}} for _ in batch_items]

                if category_features:
                    real_categories_batch = self.eval_scorer.score_batch(
                        real_comment_batch,
                        include_justifications=include_justifications,
                        features_filter=[f["feature"] for f in category_features]
                    )

                    fake_categories_batch = self.eval_scorer.score_batch(
                        fake_comment_batch,
                        include_justifications=include_justifications,
                        features_filter=[f["feature"] for f in category_features]
                    )
                else:
                    real_categories_batch = [{"categories": {}} for _ in batch_items]
                    fake_categories_batch = [{"categories": {}} for _ in batch_items]

            else:
                real_scores_batch = [{"scores": {}} for _ in batch_items]
                fake_scores_batch = [{"scores": {}} for _ in batch_items]
                alignment_scores_batch = [{"scores": {}} for _ in batch_items]
                real_categories_batch = [{"categories": {}} for _ in batch_items]
                fake_categories_batch = [{"categories": {}} for _ in batch_items]
            
            for j, proc_item in enumerate(batch_items):
                real_scores = real_scores_batch[j].get("scores", {}) if j < len(real_scores_batch) else {}
                fake_scores = fake_scores_batch[j].get("scores", {}) if j < len(fake_scores_batch) else {}
                alignment_scores = alignment_scores_batch[j].get("scores", {}) if j < len(alignment_scores_batch) else {}
                
                real_scores_justifications = real_scores_batch[j].get("justifications", {}) if j < len(real_scores_batch) else {}
                fake_scores_justifications = fake_scores_batch[j].get("justifications", {}) if j < len(fake_scores_batch) else {}
                alignment_scores_justifications = alignment_scores_batch[j].get("justifications", {}) if j < len(alignment_scores_batch) else {}

                real_categories = real_categories_batch[j].get("categories", {}) if j < len(real_categories_batch) else {}
                fake_categories = fake_categories_batch[j].get("categories", {}) if j < len(fake_categories_batch) else {}
                real_category_justifications = real_categories_batch[j].get("justifications", {}) if j < len(real_categories_batch) else {}
                fake_category_justifications = fake_categories_batch[j].get("justifications", {}) if j < len(fake_categories_batch) else {}

                advanced_metrics = self.metrics_calculator.calculate_all_metrics(
                    proc_item["fake_comment"], proc_item["original_comment"]
                )

                batch_results.append({
                    "original_post": proc_item["original_post"],
                    "original_comment": proc_item["original_comment"],
                    "fake_comment": proc_item["fake_comment"],
                    "original_post_id": proc_item["original_post_id"],
                    "original_comment_id": proc_item["original_comment_id"],
                    "evaluated_cluster_id": proc_item["evaluated_cluster_id"],
                    "rag_filters_applied": proc_item.get("rag_filters_applied"),
                    "rag_category_filters": proc_item.get("rag_category_filters"),
                    "rag_cluster_ids_used": proc_item.get("rag_cluster_ids_used"),
                    "rag_final_doc_count": proc_item.get("rag_final_doc_count"),
                    "real_comment_scores": real_scores,
                    "fake_comment_scores": fake_scores,
                    "alignment_scores": alignment_scores,
                    "real_comment_categories": real_categories,
                    "fake_comment_categories": fake_categories,
                    "advanced_metrics": advanced_metrics,
                    "real_comment_scores_justifications": real_scores_justifications,
                    "fake_comment_scores_justifications": fake_scores_justifications,
                    "alignment_scores_justifications": alignment_scores_justifications,
                    "real_comment_category_justifications": real_category_justifications,
                    "fake_comment_category_justifications": fake_category_justifications,
                    "original_scores": proc_item["original_scores"],
                    "original_categories": proc_item["original_categories"],
                    "original_justifications": proc_item["original_justifications"]
                })
            
            all_results.extend(batch_results)
            first_flag=True

        return all_results

    def report_results(self, results: List[Dict], output_filename_base: str = "results/evaluation_results", output_format: str = "json"):
        """Generate and save evaluation report."""
        if not results:
            print("No evaluation results to report.")
            return

        print("\n--- Evaluation Summary ---")
        
        _global_debug_logger.write_file("Evaluation Results", results)

        try:
            with open(self.evaluation_prompts_path, "r") as f:
                evaluation_features = json.load(f)
        except FileNotFoundError:
            print(f"Error: Evaluation prompts file not found at {self.evaluation_prompts_path}")
            return
        
        score_features = [f for f in evaluation_features if f.get("type") == "score"]
        alignment_features = [f for f in evaluation_features if f.get("type") == "alignment_score"]
        all_score_features = [f for f in evaluation_features if f.get("type") in ["score", "alignment_score"]]

        aggregated_real_scores = {f["feature"].lower().replace(" ", "_"): [] for f in all_score_features}
        aggregated_fake_scores = {f["feature"].lower().replace(" ", "_"): [] for f in all_score_features}

        for res in results:
            real_scores = res.get("real_comment_scores", {})
            fake_scores = res.get("fake_comment_scores", {})

            for metric, score in real_scores.items():
                if isinstance(score, (int, float)) and metric in aggregated_real_scores:
                    aggregated_real_scores[metric].append(score)
            
            for metric, score in fake_scores.items():
                if isinstance(score, (int, float)) and metric in aggregated_fake_scores:
                    aggregated_fake_scores[metric].append(score)

        if all_score_features:
            print("\nIndividual Comment Scores (scored separately):")
            for f in all_score_features:
                metric = f["feature"].lower().replace(" ", "_")
                
                if metric in aggregated_real_scores and aggregated_real_scores[metric]:
                    avg_real = f"{sum(aggregated_real_scores[metric]) / len(aggregated_real_scores[metric]):.2f}"
                else:
                    avg_real = "N/A"

                if metric in aggregated_fake_scores and aggregated_fake_scores[metric]:
                    avg_fake = f"{sum(aggregated_fake_scores[metric]) / len(aggregated_fake_scores[metric]):.2f}"
                else:
                    avg_fake = "N/A"

                print(f"  {metric}: Real = {avg_real}, Fake = {avg_fake}")

        # Report category distributions if categorization was performed
        has_categories = any(res.get("real_comment_categories") or res.get("fake_comment_categories") for res in results)
        if has_categories:
            print("\n--- Category Distributions ---")
            
            # Collect all category feature names
            category_feature_names = set()
            for res in results:
                if res.get("real_comment_categories"):
                    category_feature_names.update(res["real_comment_categories"].keys())
                if res.get("fake_comment_categories"):
                    category_feature_names.update(res["fake_comment_categories"].keys())
            
            for feature_name in sorted(category_feature_names):
                print(f"\n{feature_name}:")
                
                # Count real comment categories
                real_category_counts = {}
                fake_category_counts = {}
                
                for res in results:
                    real_cat = res.get("real_comment_categories", {}).get(feature_name)
                    if real_cat:
                        real_category_counts[real_cat] = real_category_counts.get(real_cat, 0) + 1
                    
                    fake_cat = res.get("fake_comment_categories", {}).get(feature_name)
                    if fake_cat:
                        fake_category_counts[fake_cat] = fake_category_counts.get(fake_cat, 0) + 1
                
                # Get all unique categories
                all_categories = set(real_category_counts.keys()) | set(fake_category_counts.keys())
                
                print("  Category Distribution:")
                print(f"    {'Category':<20} {'Real':<10} {'Fake':<10}")
                print(f"    {'-'*20} {'-'*10} {'-'*10}")
                
                for cat in sorted(all_categories):
                    real_count = real_category_counts.get(cat, 0)
                    fake_count = fake_category_counts.get(cat, 0)
                    real_pct = f"({real_count/len(results)*100:.1f}%)" if results else ""
                    fake_pct = f"({fake_count/len(results)*100:.1f}%)" if results else ""
                    print(f"    {cat:<20} {real_count:<4}{real_pct:<6} {fake_count:<4}{fake_pct:<6}")
                
                # Calculate agreement
                agreement_count = sum(
                    1 for res in results 
                    if res.get("real_comment_categories", {}).get(feature_name) == 
                       res.get("fake_comment_categories", {}).get(feature_name)
                )
                if results:
                    agreement_pct = agreement_count / len(results) * 100
                    print(f"\n  Category Agreement: {agreement_count}/{len(results)} ({agreement_pct:.1f}%)")

        print("\nAdvanced Similarity Metrics (Generated vs Original):")
        
        advanced_metric_names = set()
        for res in results:
            advanced_metric_names.update(res.get("advanced_metrics", {}).keys())
        
        aggregated_advanced_metrics = {metric: [] for metric in advanced_metric_names}
        
        for res in results:
            advanced_metrics = res.get("advanced_metrics", {})
            for metric, score in advanced_metrics.items():
                if score is not None:
                    aggregated_advanced_metrics[metric].append(score)
        
        for metric in sorted(advanced_metric_names):
            values = aggregated_advanced_metrics[metric]
            if values:
                avg_score = sum(values) / len(values)
                print(f"  {metric}: {avg_score:.4f} (n={len(values)})")
            else:
                print(f"  {metric}: N/A (no valid scores)")
        
        print("\n--- Perplexity Statistics (Robust Measures) ---")
        perplexity_stats = self.calculate_robust_statistics(results)
        
        if "real_perplexity_median" in perplexity_stats:
            print(f"\nReal Comment Perplexity:")
            print(f"  Median: {perplexity_stats['real_perplexity_median']:.2f}")
            print(f"  IQR: [{perplexity_stats['real_perplexity_q25']:.2f}, {perplexity_stats['real_perplexity_q75']:.2f}]")
            print(f"  Mean ± SD: {perplexity_stats['real_perplexity_mean']:.2f} ± {perplexity_stats['real_perplexity_std']:.2f}")
            print(f"  Max: {perplexity_stats['real_perplexity_max']:.2f}")
            print(f"  n: {perplexity_stats['real_perplexity_n']}")
        
        if "bot_perplexity_median" in perplexity_stats:
            print(f"\nBot Comment Perplexity:")
            print(f"  Median: {perplexity_stats['bot_perplexity_median']:.2f}")
            print(f"  IQR: [{perplexity_stats['bot_perplexity_q25']:.2f}, {perplexity_stats['bot_perplexity_q75']:.2f}]")
            print(f"  Mean ± SD: {perplexity_stats['bot_perplexity_mean']:.2f} ± {perplexity_stats['bot_perplexity_std']:.2f}")
            print(f"  Max: {perplexity_stats['bot_perplexity_max']:.2f}")
            print(f"  n: {perplexity_stats['bot_perplexity_n']}")
        
        if "real_perplexity_median" in perplexity_stats and "bot_perplexity_median" in perplexity_stats:
            print(f"\nComparison:")
            print(f"  Median difference: {perplexity_stats['real_perplexity_median'] - perplexity_stats['bot_perplexity_median']:.2f}")
            if perplexity_stats['bot_perplexity_median'] > 0:
                print(f"  Real comments are {perplexity_stats['real_perplexity_median'] / perplexity_stats['bot_perplexity_median']:.2f}x more perplexing than bot comments (median ratio)")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format == "json":
            output_filename = f"{output_filename_base}_{timestamp}.json"
            output_dir = os.path.dirname(output_filename)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_filename, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nDetailed evaluation results saved to {output_filename}")

        elif output_format == "csv":
            flattened_results = []
            for res in results:
                flat_row = {
                    "original_post_id": res.get("original_post_id", ""),
                    "original_comment_id": res.get("original_comment_id", ""),
                    "original_post": res.get("original_post", ""),
                    "original_comment": res.get("original_comment", ""),
                    "fake_comment": res.get("fake_comment", ""),
                    "evaluated_cluster_id": res.get("evaluated_cluster_id", None),
                    "rag_filters_applied": json.dumps(res.get("rag_filters_applied")) if res.get("rag_filters_applied") else None,
                    "rag_category_filters": json.dumps(res.get("rag_category_filters")) if res.get("rag_category_filters") else None,
                    "rag_cluster_ids_used": json.dumps(res.get("rag_cluster_ids_used")) if res.get("rag_cluster_ids_used") else None,
                    "rag_final_doc_count": res.get("rag_final_doc_count")
                }
                
                # Add scores
                for metric, score in res.get("real_comment_scores", {}).items():
                    if isinstance(score, (int, float)):
                        flat_row[f"real_{metric}_score"] = round(score, 4)
                    else:
                        flat_row[f"real_{metric}_score"] = score
                
                for metric, score in res.get("fake_comment_scores", {}).items():
                    if isinstance(score, (int, float)):
                        flat_row[f"fake_{metric}_score"] = round(score, 4)
                    else:
                        flat_row[f"fake_{metric}_score"] = score
                
                for metric, score in res.get("alignment_scores", {}).items():
                    if isinstance(score, (int, float)):
                        flat_row[f"alignment_{metric}_score"] = round(score, 4)
                    else:
                        flat_row[f"alignment_{metric}_score"] = score
                
                # Add categories
                for feature_name, category in res.get("real_comment_categories", {}).items():
                    flat_row[f"real_{feature_name}_category"] = category
                
                for feature_name, category in res.get("fake_comment_categories", {}).items():
                    flat_row[f"fake_{feature_name}_category"] = category

                # Add original data
                original_scores = res.get("original_scores", {})
                parsed_original_scores = {}
                if isinstance(original_scores, str):
                    try:
                        parsed_original_scores = json.loads(original_scores)
                    except json.JSONDecodeError:
                        parsed_original_scores = {}
                elif isinstance(original_scores, dict):
                    parsed_original_scores = original_scores
                
                for metric, score in parsed_original_scores.items():
                    if isinstance(score, (int, float)):
                        flat_row[f"original_{metric}_score"] = round(score, 4)
                    else:
                        flat_row[f"original_{metric}_score"] = score

                original_categories = res.get("original_categories", [])
                parsed_original_categories = []
                if isinstance(original_categories, str):
                    try:
                        parsed_original_categories = json.loads(original_categories)
                    except json.JSONDecodeError:
                        flat_row["original_categories_raw"] = str(original_categories)
                        parsed_original_categories = []
                else:
                    parsed_original_categories = original_categories
                
                if isinstance(parsed_original_categories, list):
                    for category_dict in parsed_original_categories:
                        if isinstance(category_dict, dict):
                            for category_key, category_value in category_dict.items():
                                flat_row[f"original_category_{category_key}"] = str(category_value)
                elif isinstance(parsed_original_categories, dict):
                    for category_key, category_value in parsed_original_categories.items():
                        flat_row[f"original_category_{category_key}"] = str(category_value)

                # Add advanced metrics
                for metric, score in res.get("advanced_metrics", {}).items():
                    if score is not None:
                        if isinstance(score, (int, float)):
                            if abs(score) < 0.001:
                                flat_row[f"{metric}"] = f"{score:.6f}"
                            else:
                                flat_row[f"{metric}"] = round(score, 4)
                        else:
                            flat_row[f"{metric}"] = score
                    else:
                        flat_row[f"{metric}"] = None
                
                # Add justifications if present
                for metric, justification in res.get("real_comment_scores_justifications", {}).items():
                    flat_row[f"real_{metric}_score_justification"] = justification

                for metric, justification in res.get("fake_comment_scores_justifications", {}).items():
                    flat_row[f"fake_{metric}_score_justification"] = justification

                for metric, justification in res.get("alignment_scores_justifications", {}).items():
                    flat_row[f"alignment_{metric}_score_justification"] = justification
                
                for feature_name, justification in res.get("real_comment_category_justifications", {}).items():
                    flat_row[f"real_{feature_name}_category_justification"] = justification
                
                for feature_name, justification in res.get("fake_comment_category_justifications", {}).items():
                    flat_row[f"fake_{feature_name}_category_justification"] = justification

                original_justifications = res.get("original_justifications", {})
                parsed_original_justifications = {}
                if isinstance(original_justifications, str):
                    try:
                        parsed_original_justifications = json.loads(original_justifications)
                    except json.JSONDecodeError:
                        parsed_original_justifications = {}
                elif isinstance(original_justifications, dict):
                    parsed_original_justifications = original_justifications
                
                for metric, justification in parsed_original_justifications.items():
                    flat_row[f"original_{metric}_justification"] = justification
            
                flattened_results.append(flat_row)
            
            df = pd.DataFrame(flattened_results)
            output_filename = f"{output_filename_base}_{timestamp}.csv"
            output_dir = os.path.dirname(output_filename)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            df.to_csv(output_filename, index=False)
            print(f"\nDetailed evaluation results saved to {output_filename}")
        
        else:
            print(f"Unsupported output format: {output_format}. Results not saved to file.")

        print("\n--- End of Evaluation ---")