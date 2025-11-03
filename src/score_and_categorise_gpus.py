#
# By Ian Drumm, The University of Salford, UK.
# Modified with improved timeout handling and retry logic
#
import json
import os
import re
import sys
import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from matplotlib.pylab import record
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
from jsonschema import Draft202012Validator, exceptions as js_exceptions
from pydantic import BaseModel, Field, ValidationError
import threading

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file from current directory
except ImportError:
    # python-dotenv not installed, will fall back to system environment variables
    pass

try:
    import aiohttp
except ImportError:
    print("Please install aiohttp: pip install aiohttp")
    sys.exit(1)

# Import API clients - made optional
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Existing imports from original code
from src.debug_log import DebugLogger
_global_debug_logger = DebugLogger()


# ============================================================================
# Abstract LLM Client Interface
# ============================================================================

class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    async def chat(self, model: str, messages: List[Dict[str, str]], 
                   format: str = "json", options: Optional[Dict] = None) -> Dict[str, Any]:
        """Send a chat completion request"""
        pass
    
    @abstractmethod
    async def __aenter__(self):
        """Async context manager entry"""
        pass
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        pass


# ============================================================================
# OpenAI Client Implementation
# ============================================================================

class OpenAIClient(LLMClient):
    """OpenAI API client with async support"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, timeout: int = 300):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter")
        
        self.base_url = base_url
        self.timeout = timeout
        self.client = None
    
    async def __aenter__(self):
        # Initialize async OpenAI client
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.close()
    
    async def chat(self, model: str, messages: List[Dict[str, str]], 
                   format: str = "json", options: Optional[Dict] = None) -> Dict[str, Any]:
        """Make OpenAI chat completion request"""
        options = options or {}
        temperature = options.get("temperature", 0.0)
        
        # Build request parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        # Add JSON mode if requested
        if format == "json":
            params["response_format"] = {"type": "json_object"}
        
        # Add other options
        if "max_tokens" in options:
            params["max_tokens"] = options["max_tokens"]
        
        try:
            response = await self.client.chat.completions.create(**params)
            
            # Format response to match expected structure
            return {
                "message": {
                    "content": response.choices[0].message.content
                }
            }
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")


# ============================================================================
# Gemini Client Implementation
# ============================================================================

class GeminiClient(LLMClient):
    """Google Gemini API client with async support"""
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 300):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI library not installed. Run: pip install google-generativeai")
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable or pass api_key parameter")
        
        genai.configure(api_key=self.api_key)
        self.timeout = timeout
        self.model_instance = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def chat(self, model: str, messages: List[Dict[str, str]], 
                   format: str = "json", options: Optional[Dict] = None) -> Dict[str, Any]:
        """Make Gemini chat completion request"""
        options = options or {}
        temperature = options.get("temperature", 0.0)
        
        # Initialize model with generation config
        generation_config = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": options.get("max_tokens", 8192),
        }
        
        # Add JSON response format if requested
        if format == "json":
            generation_config["response_mime_type"] = "application/json"
        
        model_instance = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config
        )
        
        # Convert messages to Gemini format (combine system + user messages)
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"{content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        full_prompt = "\n\n".join(prompt_parts)
        
        try:
            # Run in executor since Gemini SDK doesn't have native async
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: model_instance.generate_content(full_prompt)
                ),
                timeout=self.timeout
            )
            
            # Format response to match expected structure
            return {
                "message": {
                    "content": response.text
                }
            }
        except asyncio.TimeoutError:
            raise Exception(f"Gemini API timeout after {self.timeout} seconds")
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")


# ============================================================================
# Ollama Async Client Implementation
# ============================================================================

class OllamaAsyncClient(LLMClient):
    """Ollama API client with improved timeout and error handling"""
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = None
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def chat(self, model: str, messages: List[Dict[str, str]], 
                   format: str = "json", options: Optional[Dict] = None) -> Dict[str, Any]:
        """Make Ollama chat completion request with timeout"""
        options = options or {}
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": format,
            "options": options
        }
        
        try:
            async with self.session.post(f"{self.base_url}/api/chat", json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"Ollama API error {resp.status}: {text}")
                
                data = await resp.json()
                return data
                
        except asyncio.TimeoutError:
            raise Exception(f"Ollama API timeout after {self.timeout} seconds")
        except aiohttp.ClientError as e:
            raise Exception(f"Ollama network error: {str(e)}")


# ============================================================================
# Ollama Synchronous Client (Multi-GPU)
# ============================================================================

class OllamaClient(LLMClient):
    """Synchronous Ollama client for multi-GPU setup"""
    
    def __init__(self, ports: List[int] = None, timeout: int = 300):
        self.ports = ports or [11434]
        self.timeout = timeout
        self.current_index = 0
        self.lock = threading.Lock()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def _get_next_port(self) -> int:
        """Round-robin port selection"""
        with self.lock:
            port = self.ports[self.current_index % len(self.ports)]
            self.current_index += 1
            return port
    
    async def chat(self, model: str, messages: List[Dict[str, str]], 
                   format: str = "json", options: Optional[Dict] = None) -> Dict[str, Any]:
        """Make Ollama request with round-robin port selection"""
        import requests
        
        options = options or {}
        port = self._get_next_port()
        url = f"http://localhost:{port}/api/chat"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": format,
            "options": options
        }
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(url, json=payload, timeout=self.timeout)
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error {response.status_code}: {response.text}")
            
            return response.json()
            
        except requests.Timeout:
            raise Exception(f"Ollama API timeout after {self.timeout} seconds on port {port}")
        except Exception as e:
            raise Exception(f"Ollama error on port {port}: {str(e)}")


# ============================================================================
# Client Factory
# ============================================================================

def create_llm_client(backend: str, **kwargs) -> LLMClient:
    """Factory function to create appropriate LLM client"""
    backend = backend.lower()
    
    if backend == "ollama_async":
        base_url = kwargs.get("base_url", "http://localhost:11434")
        timeout = kwargs.get("timeout", 300)
        return OllamaAsyncClient(base_url=base_url, timeout=timeout)
    
    elif backend == "ollama":
        ports = kwargs.get("ports", [11434])
        timeout = kwargs.get("timeout", 300)
        return OllamaClient(ports=ports, timeout=timeout)
    
    elif backend == "openai":
        api_key = kwargs.get("api_key")
        base_url = kwargs.get("base_url")
        timeout = kwargs.get("timeout", 300)
        return OpenAIClient(api_key=api_key, base_url=base_url, timeout=timeout)
    
    elif backend == "gemini":
        api_key = kwargs.get("api_key")
        timeout = kwargs.get("timeout", 300)
        return GeminiClient(api_key=api_key, timeout=timeout)
    
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose from: ollama_async, ollama, openai, gemini")


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_content(resp: Dict[str, Any]) -> Optional[str]:
    """Extract content from LLM response"""
    msg = resp.get("message", {})
    return msg.get("content")


def _build_feature_schema(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build JSON schema for features"""
    properties = {}
    required = []
    
    for feat in features:
        name = feat["name"]
        required.append(name)
        
        if feat["type"] == "score":
            properties[name] = {
                "type": "object",
                "properties": {
                    "score": {"type": "integer"},
                    "justification": {"type": "string"}
                },
                "required": ["score", "justification"]
            }
        elif feat["type"] == "category":
            properties[name] = {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "justification": {"type": "string"}
                },
                "required": ["category", "justification"]
            }
    
    schema = {
        "type": "object",
        "properties": properties,
        "required": required
    }
    
    return schema


# ============================================================================
# Main Scorer Class
# ============================================================================

class MultiGPUCommentScorer:
    """Comment scorer with multiple backend support and improved timeout handling"""
    
    def __init__(
        self,
        features_path: Optional[str] = None,
        prompts_path: Optional[str] = None,
        model: str = "llama3:8b",
        temperature: float = 0.0,
        max_concurrent: int = 2,  # Changed default from 8 to 2 for single GPU
        backend: str = "ollama_async",
        ollama_ports: List[int] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        timeout: int = 300,  # 5 minute default timeout
        max_retries: int = 3,  # Number of retries on failure
    ):
        """
        Initialize scorer with timeout and retry capabilities.
        
        Args:
            features_path: Path to features JSON (old format)
            prompts_path: Path to prompts JSON (new format)
            model: Model name
            temperature: Temperature for generation
            max_concurrent: Maximum concurrent requests (use 1-2 for single GPU)
            backend: LLM backend ('ollama_async', 'ollama', 'openai', 'gemini')
            ollama_ports: List of Ollama ports for multi-GPU
            openai_api_key: OpenAI API key
            openai_base_url: OpenAI base URL
            gemini_api_key: Gemini API key
            timeout: Request timeout in seconds (default: 300)
            max_retries: Number of retry attempts on failure (default: 3)
        """
        self.model = model
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        self.backend = backend
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Load features/prompts
        if prompts_path:
            self._load_prompts_format(prompts_path)
        elif features_path:
            self._load_features_format(features_path)
        else:
            raise ValueError("Either features_path or prompts_path must be provided")
        
        # Setup client configuration
        self.client_config = {"timeout": timeout}
        
        if backend == "ollama":
            print("Ollama ports ", ollama_ports)
            self.client_config["ports"] = ollama_ports or [11434]
        elif backend == "openai":
            self.client_config["api_key"] = openai_api_key
            self.client_config["base_url"] = openai_base_url
        elif backend == "gemini":
            self.client_config["api_key"] = gemini_api_key
        
        self.first_prompt_logged = False
        
        print(f"Initialized scorer: backend={backend}, model={model}, max_concurrent={max_concurrent}, timeout={timeout}s")

    def _load_features_format(self, features_path: str):
        """Load old features format - handles both dict and list formats"""
        with open(features_path, "r") as f:
            data = json.load(f)
        
        # Check if it's a list (alternative format with "feature" and "rubric" keys)
        if isinstance(data, list):
            print(f"Detected list-based rubrics format with {len(data)} features, converting...")
            
            # Convert list format to expected structure
            self.features = []
            for item in data:
                feature = {
                    "name": item.get("feature", item.get("name", "")),
                    "type": item.get("type", "score"),
                    "description": item.get("rubric", item.get("description", ""))
                }
                self.features.append(feature)
            
            # Create default prompts that work well with the rubrics
            self.system_prompt = (
                "You are an expert analyst evaluating Reddit comments. "
                "Analyze each comment carefully according to the provided rubrics. "
                "Pay special attention to sarcasm, negation, and quoted text as outlined in the feature descriptions."
            )
            
            self.user_prompt_template = """Analyze the following Reddit post and comment:

Post: {post}

Comment: {comment}

Evaluate this comment using the features below. For each feature, provide both a score/category (following the rubric exactly) and a brief justification.

Features to evaluate:
{features}

Return your analysis as a JSON object where each key is the exact feature name, and each value is an object containing:
- "score" or "category": the rating according to the rubric
- "justification": a brief explanation of your decision

Important: Watch for sarcasm, negation, and quoted text as described in the rubrics."""
            
            print(f"✓ Loaded {len(self.features)} features from list-based format")
        
        # Standard dictionary format
        elif isinstance(data, dict):
            self.system_prompt = data.get("system_prompt", "")
            self.user_prompt_template = data.get("user_prompt_template", "")
            self.features = data.get("features", [])
            print(f"✓ Loaded {len(self.features)} features from dict format")
        
        else:
            raise ValueError(f"Unexpected features file format: {type(data)}. Expected dict or list.")
        
        # Build key map and schema (same for both formats)
        self.key_map = {f["name"]: f["name"] for f in self.features}
        self.schema = _build_feature_schema(self.features)

    def _load_prompts_format(self, prompts_path: str):
        """Load prompts format - handles both dict and list formats"""
        with open(prompts_path, "r") as f:
            data = json.load(f)
        
        # Check if it's a list (alternative format with "feature" and "rubric" keys)
        if isinstance(data, list):
            print(f"Detected list-based rubrics format with {len(data)} features, converting...")
            
            # Convert list format to expected structure
            self.features = []
            for item in data:
                feature = {
                    "name": item.get("feature", item.get("name", "")),
                    "type": item.get("type", "score"),
                    "description": item.get("rubric", item.get("description", ""))
                }
                self.features.append(feature)
            
            # Create default prompts that work well with the rubrics
            self.system_prompt = (
                "You are an expert analyst evaluating Reddit comments. "
                "Analyze each comment carefully according to the provided rubrics. "
                "Pay special attention to sarcasm, negation, and quoted text as outlined in the feature descriptions."
            )
            
            self.user_prompt_template = """Analyze the following Reddit post and comment:

Post: {post}

Comment: {comment}

Evaluate this comment using the features below. For each feature, provide both a score/category (following the rubric exactly) and a brief justification.

Features to evaluate:
{features}

Return your analysis as a JSON object where each key is the exact feature name, and each value is an object containing:
- "score" or "category": the rating according to the rubric
- "justification": a brief explanation of your decision

Important: Watch for sarcasm, negation, and quoted text as described in the rubrics."""
            
            print(f"✓ Loaded {len(self.features)} features from list-based format")
        
        # Standard dictionary format
        elif isinstance(data, dict):
            self.system_prompt = data.get("system_prompt", "")
            self.user_prompt_template = data.get("user_prompt_template", "")
            self.features = data.get("features", [])
            print(f"✓ Loaded {len(self.features)} features from dict format")
        
        else:
            raise ValueError(f"Unexpected prompts file format: {type(data)}. Expected dict or list.")
        
        # Build key map and schema (same for both formats)
        self.key_map = {f["name"]: f["name"] for f in self.features}
        self.schema = _build_feature_schema(self.features)

    async def _call_llm_async(self, post: str, comment: str, client: LLMClient, 
                             original_comment: Optional[str] = None,
                             features_filter: Optional[List[str]] = None) -> Optional[str]:
        """Call LLM with timeout handling"""
        
        # Build prompt
        prompt_vars = {
            "post": post,
            "comment": comment,
            "original_comment": original_comment or comment
        }
        
        # Filter features if specified
        if features_filter:
            features_to_use = [f for f in self.features if f["name"] in features_filter]
        else:
            features_to_use = self.features
        
        prompt_vars["features"] = json.dumps(features_to_use, indent=2)
        prompt = self.user_prompt_template.format(**prompt_vars)
        
        # Log first prompt
        if not self.first_prompt_logged:
            self._log_first_prompt(prompt, post, comment, original_comment)
            self.first_prompt_logged = True
        
        try:
            resp = await client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return ONLY valid JSON that matches the user's requested schema. No prose."},
                    {"role": "user", "content": prompt},
                ],
                format="json",
                options={"temperature": self.temperature}
            )
            
            return _extract_content(resp)
        except Exception as e:
            err = f"LLM call failed ({self.backend}): {e}"
            _global_debug_logger.write_file("ERROR", err)
            print(err)
            return None

    def _log_first_prompt(self, prompt: str, post: str, comment: str, original_comment: Optional[str]) -> None:
        """Log first prompt to file"""
        import datetime
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"first_prompt_{timestamp}.log")
        
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"FIRST PROMPT - Backend: {self.backend}\n")
            f.write("=" * 70 + "\n")
            f.write(f"Model: {self.model}\n")
            f.write(f"Temperature: {self.temperature}\n")
            f.write(f"Max Concurrent: {self.max_concurrent}\n")
            f.write(f"Timeout: {self.timeout}s\n\n")
            f.write(f"POST:\n{post}\n\n")
            f.write(f"COMMENT:\n{comment}\n\n")
            if original_comment:
                f.write(f"ORIGINAL:\n{original_comment}\n\n")
            f.write(f"PROMPT:\n{prompt}\n")

    async def _process_item_async(self, item: Dict[str, Any], client: LLMClient, 
                                  include_justifications: bool, 
                                  features_filter: Optional[List[str]]) -> Optional[Dict[str, Any]]:
        """Process single item with retry logic"""
        post = item.get("post", "")
        comment = item.get("comment", "")
        original_comment = item.get("original_comment")
        
        for attempt in range(self.max_retries):
            try:
                raw = await self._call_llm_async(
                    post=post, 
                    comment=comment, 
                    client=client, 
                    original_comment=original_comment, 
                    features_filter=features_filter
                )
                
                if raw is None:
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        print(f"  Retry {attempt + 1}/{self.max_retries} after {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    return None

                try:
                    model_obj = json.loads(raw)
                except json.JSONDecodeError as e:
                    if attempt < self.max_retries - 1:
                        print(f"  JSON decode error, retrying: {e}")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    print(f"  JSON decode error after {self.max_retries} attempts: {e}")
                    return None
                
                return self._format_pipeline_record(item, model_obj, include_justifications)
                
            except (asyncio.TimeoutError, Exception) as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  Error on attempt {attempt + 1}/{self.max_retries}: {e}")
                    print(f"  Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"  Failed after {self.max_retries} attempts: {e}")
                    return None
        
        return None

    def _format_pipeline_record(self, item: Dict[str, Any], model_obj: Dict[str, Any], 
                                include_justifications: bool) -> Dict[str, Any]:
        """Format output record"""
        scores: Dict[str, int] = {}
        categories: Dict[str, str] = {}
        justifications: Dict[str, str] = {}

        for k, v in model_obj.items():
            if k in ("evidence_spans", "meta"):
                continue
            ext = self.key_map.get(k, k)
            if isinstance(v, dict):
                if "score" in v:
                    scores[ext] = int(v.get("score", 0))
                elif "category" in v:
                    categories[ext] = v["category"]
                if include_justifications and "justification" in v:
                    justifications[ext] = v["justification"]

        record = {
            "post": item.get("post", ""),
            "comment": item.get("comment", ""),
            "scores": scores,
            "categories": categories,
            "backend": self.backend,
            "model": self.model,
        }
        if include_justifications:
            record["justifications"] = justifications
        if "post_metadata" in item:
            record["post_metadata"] = item["post_metadata"]
        if "comment_metadata" in item:
            record["comment_metadata"] = item["comment_metadata"]
        return record

    async def score_batch_async(self, pairs: List[Dict[str, Any]], 
                               include_justifications: bool = True, 
                               features_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Score batch using selected backend"""
        
        # Create appropriate client
        client = create_llm_client(self.backend, **self.client_config)
        
        async with client:
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def bounded_process(item):
                async with semaphore:
                    return await self._process_item_async(item, client, include_justifications, features_filter)
            
            print(f"Processing {len(pairs)} items with {self.backend} backend")
            print(f"Model: {self.model}, Max concurrent: {self.max_concurrent}, Timeout: {self.timeout}s")
            
            tasks = [bounded_process(item) for item in pairs]
            results = await async_tqdm.gather(*tasks, desc=f"Scoring with {self.model}")
            
            successful = [r for r in results if r is not None]
            failed = len(results) - len(successful)
            
            if failed > 0:
                print(f"Warning: {failed} items failed to process")
            
            return successful

    def score_batch(self, pairs: List[Dict[str, Any]], include_justifications: bool = True, 
                   features_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Synchronous entry point"""
        return asyncio.run(self.score_batch_async(pairs, include_justifications, features_filter))

    def score_one(self, post: str, comment: str, post_metadata: Optional[Dict[str, Any]] = None,
                 comment_metadata: Optional[Dict[str, Any]] = None, include_justifications: bool = False,
                 original_comment: Optional[str] = None, features_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """Score single item"""
        item = {
            "post": post,
            "comment": comment,
            "post_metadata": post_metadata or {},
            "comment_metadata": comment_metadata or {},
            "original_comment": original_comment
        }
        results = self.score_batch([item], include_justifications, features_filter)
        return results[0] if results else {}


# Backward compatibility
CommentScorer = MultiGPUCommentScorer


# ============================================================================
# CLI (Updated with new backends)
# ============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Score comments with multiple LLM backends")
    p.add_argument("--features", required=False, help="Path to features JSON")
    p.add_argument("--model", default="llama3:8b", help="Model name")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--input", help="Input JSONL file")
    p.add_argument("--out", help="Output JSONL file")
    p.add_argument("--include-justifications", action="store_true")
    p.add_argument("--max-concurrent", type=int, default=2, 
                   help="Max concurrent requests (use 1-2 for single GPU, default: 2)")
    p.add_argument("--timeout", type=int, default=300,
                   help="Request timeout in seconds (default: 300)")
    p.add_argument("--max-retries", type=int, default=3,
                   help="Number of retry attempts on failure (default: 3)")
    
    # Backend selection
    p.add_argument("--backend", default="ollama_async", 
                   choices=["ollama_async", "ollama", "openai", "gemini"],
                   help="LLM backend to use")
    
    # Ollama-specific options
    p.add_argument("--ports", nargs="+", type=int, default=[11434],
                   help="Ollama ports (for ollama backend)")
    
    # OpenAI-specific options
    p.add_argument("--openai-api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    p.add_argument("--openai-base-url", help="OpenAI base URL (for custom endpoints)")
    
    # Gemini-specific options
    p.add_argument("--gemini-api-key", help="Gemini API key (or set GEMINI_API_KEY env var)")

    args = p.parse_args(argv)

    if not args.features:
        p.error("--features is required")

    scorer = MultiGPUCommentScorer(
        features_path=args.features,
        model=args.model,
        temperature=args.temperature,
        max_concurrent=args.max_concurrent,
        backend=args.backend,
        ollama_ports=args.ports,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        gemini_api_key=args.gemini_api_key,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    if not args.input:
        # Demo
        demo_item = {
            "post": "The administration announced new tariffs.",
            "comment": "About time—we should protect local jobs.",
        }
        results = scorer.score_batch([demo_item], args.include_justifications)
        print(json.dumps(results, indent=2))
    else:
        # Load and process input file
        with open(args.input, "r") as f:
            pairs = [json.loads(line) for line in f if line.strip()]
        
        results = scorer.score_batch(pairs, args.include_justifications)
        
        if args.out:
            with open(args.out, "w") as w:
                for r in results:
                    w.write(json.dumps(r) + "\n")
        else:
            print(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())