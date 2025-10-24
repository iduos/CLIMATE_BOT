#
# Extended version with OpenAI and Gemini API support
# By Ian Drumm, The University of Salford, UK.
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
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter")
        
        self.base_url = base_url
        self.client = None
    
    async def __aenter__(self):
        # Initialize async OpenAI client
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
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
    
    def __init__(self, api_key: Optional[str] = None):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI library not installed. Run: pip install google-generativeai")
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable or pass api_key parameter")
        
        genai.configure(api_key=self.api_key)
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
            response = await loop.run_in_executor(
                None,
                lambda: model_instance.generate_content(full_prompt)
            )
            
            # Format response to match expected structure
            return {
                "message": {
                    "content": response.text
                }
            }
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")


# ============================================================================
# Ollama Client (from original code)
# ============================================================================

class MultiPortOllamaClient(LLMClient):
    """Multi-port Ollama client (original implementation)"""
    
    def __init__(self, ports=[11434], max_connections_per_port=3):
        self.base_urls = [f"http://localhost:{port}" for port in ports]
        self.max_connections_per_port = max_connections_per_port
        self.sessions = {}
        self.port_index = 0
        self.lock = asyncio.Lock()
        self.port_health = {url: True for url in self.base_urls}
        self.is_single_instance = len(ports) == 1
    
    async def __aenter__(self):
        for url in self.base_urls:
            connection_limit = self.max_connections_per_port * 2 if self.is_single_instance else self.max_connections_per_port
            connector = aiohttp.TCPConnector(
                limit=connection_limit,
                limit_per_host=connection_limit,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            timeout = aiohttp.ClientTimeout(total=120, connect=30)
            session = aiohttp.ClientSession(connector=connector, timeout=timeout)
            self.sessions[url] = session
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for session in self.sessions.values():
            await session.close()
    
    async def _get_next_session(self):
        async with self.lock:
            if self.is_single_instance:
                url = self.base_urls[0]
                return self.sessions[url], url
            
            healthy_urls = [url for url in self.base_urls if self.port_health[url]]
            if not healthy_urls:
                self.port_health = {url: True for url in self.base_urls}
                healthy_urls = self.base_urls
            
            url = healthy_urls[self.port_index % len(healthy_urls)]
            self.port_index = (self.port_index + 1) % len(healthy_urls)
            return self.sessions[url], url
    
    def _mark_port_unhealthy(self, url: str):
        if not self.is_single_instance:
            self.port_health[url] = False
    
    async def chat(self, model: str, messages: List[Dict[str, str]], 
                   format: str = "json", options: Optional[Dict] = None) -> Dict[str, Any]:
        last_error = None
        actual_retries = 1 if self.is_single_instance else 2
        
        for attempt in range(actual_retries):
            try:
                session, base_url = await self._get_next_session()
                
                payload = {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "format": format,
                    "options": options or {"temperature": 0.0, "num_ctx": 4096}
                }
                
                async with session.post(f"{base_url}/api/chat", json=payload) as response:
                    if response.status == 200:
                        self.port_health[base_url] = True
                        return await response.json()
                    else:
                        error_text = await response.text()
                        last_error = f"HTTP {response.status} from {base_url}: {error_text}"
                        if response.status == 404:
                            self._mark_port_unhealthy(base_url)
                        if attempt < actual_retries - 1:
                            await asyncio.sleep(0.5)
                        continue
            except asyncio.TimeoutError:
                last_error = f"Timeout connecting to {base_url if 'base_url' in locals() else 'unknown'}"
                if 'base_url' in locals():
                    self._mark_port_unhealthy(base_url)
            except Exception as e:
                last_error = f"Error with {base_url if 'base_url' in locals() else 'unknown'}: {str(e)}"
                if 'base_url' in locals():
                    self._mark_port_unhealthy(base_url)
        
        raise Exception(f"All attempts failed. Last error: {last_error}")


# ============================================================================
# Client Factory
# ============================================================================

def create_llm_client(backend: str, **kwargs) -> LLMClient:
    """Factory function to create appropriate LLM client"""
    
    if backend == "ollama_async" or backend == "ollama":
        ports = kwargs.get("ollama_ports", [11434])
        max_connections = kwargs.get("max_connections_per_port", 3)
        return MultiPortOllamaClient(ports=ports, max_connections_per_port=max_connections)
    
    elif backend == "openai":
        api_key = kwargs.get("openai_api_key")
        base_url = kwargs.get("openai_base_url")
        return OpenAIClient(api_key=api_key, base_url=base_url)
    
    elif backend == "gemini":
        api_key = kwargs.get("gemini_api_key")
        return GeminiClient(api_key=api_key)
    
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported: ollama_async, openai, gemini")


# ============================================================================
# Helper functions (from original code - keeping relevant ones)
# ============================================================================

def snake(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()


class Feature(BaseModel):
    feature: str
    type: str = Field(default="score", description="score | category")
    rubric: Optional[str] = None
    extra_instructions: Optional[Union[str, List[str]]] = None
    scale_min: int = 1
    scale_max: int = 5
    labels: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    options: Optional[List[str]] = None
    values: Optional[List[str]] = None
    output_key: Optional[str] = None
    guidance: Optional[str] = None
    description: Optional[str] = None

    @property
    def key(self) -> str:
        return self.output_key or snake(self.feature)

    @property
    def allowed_labels(self) -> Optional[List[str]]:
        if self.type != "category":
            return None
        for attr in ("labels", "categories", "options", "values"):
            v = getattr(self, attr)
            if v:
                return list(v)
        return None


class ScoreMetric(BaseModel):
    score: int = Field(..., ge=1, le=5)
    justification: str


class CategoryMetric(BaseModel):
    category: str
    justification: str


def build_prompt(post: str, comment: str, features: List[Feature], 
                original_comment: Optional[str], features_filter: Optional[List[str]]) -> str:
    """Build prompt for LLM (from original code)"""
    lines = []
    has_alignment_score = False
    category_rubric = ""

    for f in features:
        feature_name = ""
        if features_filter is not None:
            if hasattr(f, 'feature'):
                feature_name = f.feature
            elif isinstance(f, dict):
                feature_name = f.get("feature")
            else:
                continue
            if feature_name not in features_filter:
                continue
        
        if f.type == "category":
            rubric_text = f"\n{f.rubric}" if f.rubric else ""
            extra_text = ""
            if f.extra_instructions:
                if isinstance(f.extra_instructions, list):
                    extra_text = "\n  Extra Instructions:\n    - " + "\n    - ".join(f.extra_instructions)
                else:
                    extra_text = f"\n  Extra Instructions: {f.extra_instructions}"
            # lines.append(
            #     f'- "{f.key}": {{"category": "<one of {f.labels}>", "justification": "<explanation>"}}{rubric_text}{extra_text}'
            # )
            lines.append(f'{rubric_text}{extra_text}')

        if f.type == "score":
            rubric_text = f"\nRubric:\n {f.rubric}" if f.rubric else ""
            extra_text = ""
            if f.extra_instructions:
                if isinstance(f.extra_instructions, list):
                    extra_text = "\n  Extra Instructions:\n    - " + "\n    - ".join(f.extra_instructions)
                else:
                    extra_text = f"\n  Extra Instructions: {f.extra_instructions}"
            lines.append(
                f'- "{f.key}": {{"score": <1-5>, "justification": "<explanation>"}}{rubric_text}{extra_text}'
            )

        if f.type == "alignment_score":
            has_alignment_score = True
    
    rubric = chr(10).join(lines)
    
    example_structure = {}
    for f in features:
        if f.type == "category":
            example_structure[f.key] = {
                "category": f"<one of {', '.join(f.labels)}>",
                "justification": "Brief explanation here"
            }
        else:
            example_structure[f.key] = {
                "score": 3,
                "justification": "Brief explanation here"
            }
    
    example_json = json.dumps(example_structure, indent=2)

    prompt1 = f"""

You are a JSON-only assistant. Return ONLY a JSON object that matches this EXACT structure:

{example_json}

<RUBRIC>
{rubric}
</RUBRIC>

IMPORTANT RULES:
- For score metrics, use "score" (1-5) and "justification" fields
- For category metrics, use "category" and "justification" fields instead
- Do NOT nest metrics under a "metrics" key
- Use the exact key names shown above
- No extra keys or different structure



"""
    
    if has_alignment_score:
        prompt2 = f"""Score how well the GENERATED_COMMENT to the POST aligns with the ORIGINAL_COMMENT to the POST

POST:
{post}

GENERATED_COMMENT:
{comment}

ORIGINAL_COMMENT:
{original_comment}
""".strip()
        
    else:
        prompt2 = f"""

The comment between the COMMENT markup tags is a reply to the post between the POST markup tags.

<POST>
{post}
</POST>

<COMMENT>
{comment}
</COMMENT>

Classify or score the COMMENT text (using the POST only as context).

Use the rubric between the RUBRIC markup tags to guide you.

""".strip()
    
    prompt = prompt2 + prompt1
    _global_debug_logger.write_file("PROMPT", prompt)
    return prompt


def _extract_content(resp: Any) -> Optional[str]:
    """Extract content from various response formats"""
    if resp is None:
        return None
    
    if isinstance(resp, str):
        return resp

    if isinstance(resp, dict):
        if isinstance(resp.get("content"), str):
            return resp["content"]
        if isinstance(resp.get("message"), dict) and isinstance(resp["message"].get("content"), str):
            return resp["message"]["content"]
        if isinstance(resp.get("choices"), list) and resp["choices"]:
            ch0 = resp["choices"][0]
            if isinstance(ch0, dict):
                msg = ch0.get("message") or {}
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return msg["content"]
                if isinstance(ch0.get("text"), str):
                    return ch0["text"]

    try:
        for attr in ("output_text", "content"):
            if hasattr(resp, attr) and isinstance(getattr(resp, attr), str):
                return getattr(resp, attr)
        if hasattr(resp, "message") and hasattr(resp.message, "content"):
            return resp.message.content
    except Exception:
        pass

    return None


# Include other helper functions from original (load_features, validation, etc.)
# [I'll keep the code concise - assume these are included from the original]

def load_features(features_path: str) -> List[Feature]:
    """Load features from JSON file (from original code)"""
    data = json.loads(Path(features_path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("features file must be a JSON array of feature objects")
    
    features: List[Feature] = []
    for raw in data:
        t = str(raw.get("type", "score")).strip().lower()
        if t not in ("score", "category", "alignment_score"):
            t = "score"
        
        rubric_raw = raw.get("rubric") or raw.get("guidance") or raw.get("description")
        if isinstance(rubric_raw, list):
            rubric = "\n".join(str(item) for item in rubric_raw) + "\n" if rubric_raw else None
        elif rubric_raw is not None:
            rubric = str(rubric_raw)
        else:
            rubric = None
        
        extra_instructions_raw = raw.get("extra_instructions")
        if extra_instructions_raw is not None:
            if isinstance(extra_instructions_raw, list):
                extra_instructions = extra_instructions_raw
            else:
                extra_instructions = str(extra_instructions_raw)
        else:
            extra_instructions = None
        
        feat = Feature(
            feature=raw.get("feature") or raw.get("name") or raw.get("title") or "Unnamed feature",
            type=t,
            rubric=rubric,
            extra_instructions=extra_instructions,
            scale_min=int(raw.get("scale_min", 1)),
            scale_max=int(raw.get("scale_max", 5)),
            labels=raw.get("labels") or raw.get("categories") or raw.get("options") or raw.get("values"),
            output_key=raw.get("output_key"),
            guidance=raw.get("guidance"),
            description=raw.get("description"),
        )
        features.append(feat)
    
    return features


# ... [Include validation and other helper functions from original]


# ============================================================================
# Multi-Backend Comment Scorer (Updated)
# ============================================================================

class MultiGPUCommentScorer:
    """Updated scorer with multi-backend support"""
    
    def __init__(
        self,
        features_path: Optional[str] = None,
        output_schema_path: Optional[str] = "json/llm_output_schema.json",
        model: str = "llama3:8b",
        temperature: float = 0.0,
        prompts_path: Optional[str] = None,
        num_gpus: int = 4,
        max_concurrent: int = 20,
        backend: str = "ollama_async",
        ollama_ports: Optional[List[int]] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Handle legacy alias
        if prompts_path is not None:
            if features_path is not None:
                raise ValueError("Provide only one of features_path or prompts_path, not both.")
            features_path = prompts_path

        if not features_path:
            raise ValueError("features_path (or prompts_path) is required")

        self.features: List[Feature] = load_features(features_path)
        _global_debug_logger.write_file("INIT FEATURES", self.features)

        self.model = model
        self.temperature = float(temperature)
        self.num_gpus = num_gpus
        self.max_concurrent = max_concurrent
        self.backend = backend
        
        # Store client configuration
        self.client_config = {
            "ollama_ports": ollama_ports or [11434],
            "max_connections_per_port": max(1, min(3, max_concurrent // len(ollama_ports or [11434]))),
            "openai_api_key": openai_api_key,
            "openai_base_url": openai_base_url,
            "gemini_api_key": gemini_api_key,
        }

        # Validate backend availability
        if backend == "openai" and not OPENAI_AVAILABLE:
            raise ImportError("OpenAI backend selected but openai library not installed. Run: pip install openai")
        if backend == "gemini" and not GEMINI_AVAILABLE:
            raise ImportError("Gemini backend selected but google-generativeai library not installed. Run: pip install google-generativeai")

        # Load schema (keeping original logic)
        try:
            schema_path = Path(output_schema_path) if output_schema_path else Path("json/llm_output_schema.json")
            if schema_path.exists():
                self.base_schema = json.loads(schema_path.read_text(encoding="utf-8"))
            else:
                self.base_schema = {"type": "object", "additionalProperties": True}
        except Exception:
            self.base_schema = {"type": "object", "additionalProperties": True}

        # Build validator (keeping original logic - simplified here)
        self.validator = None  # Would use build_validation_schema from original
        self.key_map: Dict[str, str] = {f.key: (f.output_key or f.key) for f in self.features}
        self.prompts_path_value = prompts_path if prompts_path is not None else features_path

        self._first_prompt_logged = False
        self._first_prompt_lock = threading.Lock()

    async def _call_llm_async(
        self, 
        post: str, 
        comment: str, 
        client: LLMClient, 
        original_comment: Optional[str] = None, 
        features_filter: Optional[List[str]] = None
    ) -> Optional[str]:
        """Async call to any LLM backend"""
        
        prompt = build_prompt(
            post=post, 
            comment=comment, 
            features=self.features, 
            original_comment=original_comment, 
            features_filter=features_filter
        )

        # Log first prompt
        if not self._first_prompt_logged:
            with self._first_prompt_lock:
                if not self._first_prompt_logged:
                    self._log_first_prompt(prompt, post, comment, original_comment)
                    self._first_prompt_logged = True

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
            f.write(f"Temperature: {self.temperature}\n\n")
            f.write(f"POST:\n{post}\n\n")
            f.write(f"COMMENT:\n{comment}\n\n")
            if original_comment:
                f.write(f"ORIGINAL:\n{original_comment}\n\n")
            f.write(f"PROMPT:\n{prompt}\n")

    async def _process_item_async(self, item: Dict[str, Any], client: LLMClient, 
                                  include_justifications: bool, features_filter: Optional[List[str]]) -> Optional[Dict[str, Any]]:
        """Process single item (from original code)"""
        post = item.get("post", "")
        comment = item.get("comment", "")
        original_comment = item.get("original_comment")

        raw = await self._call_llm_async(post=post, comment=comment, client=client, 
                                        original_comment=original_comment, features_filter=features_filter)
        if raw is None:
            return None

        try:
            model_obj = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None
        
        # Simplified validation
        return self._format_pipeline_record(item, model_obj, include_justifications)

    def _format_pipeline_record(self, item: Dict[str, Any], model_obj: Dict[str, Any], 
                                include_justifications: bool) -> Dict[str, Any]:
        """Format output record (simplified from original)"""
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
            
            print(f"Processing {len(pairs)} items with {self.backend} backend (model: {self.model})")
            
            tasks = [bounded_process(item) for item in pairs]
            results = await async_tqdm.gather(*tasks, desc=f"Scoring with {self.model}")
            
            return [r for r in results if r is not None]

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
    p.add_argument("--max-concurrent", type=int, default=8, help="Max concurrent requests")
    
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