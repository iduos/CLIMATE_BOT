# src/rag_chatbot.py
#
# By Ian Drumm, The University of Salford, UK.
#
import re
import json
import os
from typing import Optional, List, Dict, Union
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
#from langchain.schema import Document
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from src.debug_log import DebugLogger

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required, will use system environment variables

_global_debug_logger = DebugLogger()


def create_embeddings(
    model_name: str,
    backend: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
):
    """
    Create an embeddings instance based on backend and model name.
    
    Args:
        model_name: Name of the model (e.g., "text-embedding-3-small", "nomic-embed-text")
        backend: Embeddings backend ("openai", "gemini", "ollama", or None for auto-detect)
        api_key: API key for the backend (if None, uses environment variable)
        **kwargs: Additional backend-specific arguments
    
    Returns:
        LangChain Embeddings instance
    """
    # Auto-detect backend if not specified
    if backend is None:
        model_lower = model_name.lower()
        if "text-embedding" in model_lower or model_lower.startswith("embed-"):
            backend = "openai"
        elif "embedding" in model_lower and ("gemini" in model_lower or "gecko" in model_lower):
            backend = "gemini"
        else:
            backend = "ollama"  # Default to Ollama for local models
    
    backend = backend.lower()
    
    if backend == "openai":
        from langchain_openai import OpenAIEmbeddings
        
        if api_key:
            kwargs["openai_api_key"] = api_key
        elif "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        return OpenAIEmbeddings(
            model=model_name,
            **kwargs
        )
    
    elif backend == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        
        if api_key:
            kwargs["google_api_key"] = api_key
        elif "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            **kwargs
        )
    
    elif backend == "ollama":
        
        return OllamaEmbeddings(
            model=model_name,
            **kwargs
        )

    else:
        raise ValueError(f"Unsupported embedding backend: {backend}. Supported: openai, gemini, ollama")


def create_llm(
    model_name: str,
    backend: Optional[str] = None,
    temperature: float = 0.7,
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


class RAGCommentSimulator:
    def __init__(self, 
                 embed_model: str = "nomic-embed-text:latest",
                 chat_model: str = "llama3.1:8b", 
                 persist_directory: str = "chroma_db",
                 temperature: float = 0.7,
                 backend: Optional[str] = None,
                 chat_api_key: Optional[str] = None,
                 embed_backend: Optional[str] = None,
                 embed_api_key: Optional[str] = None):
        """
        Initialize RAGCommentSimulator with multi-backend support.
        
        Args:
            embed_model: Embedding model name
            chat_model: Chat model name
            persist_directory: Path to Chroma database
            temperature: Temperature for chat generation
            backend: Backend for chat model ("openai", "gemini", "anthropic", "ollama", or None for auto-detect)
            chat_api_key: API key for chat backend
            embed_backend: Backend for embedding model ("openai", "gemini", "ollama", or None for auto-detect)
            embed_api_key: API key for embedding backend
        """
        # Create embeddings with backend support
        embedder = create_embeddings(
            model_name=embed_model,
            backend=embed_backend,
            api_key=embed_api_key
        )
        
        self.vector_store = Chroma(embedding_function=embedder, persist_directory=persist_directory)
        
        # Create chat model with backend support
        self.chat_model = create_llm(
            model_name=chat_model,
            backend=backend,
            temperature=temperature,
            api_key=chat_api_key
        )
        
        print(f"Initialized RAG chatbot:")
        print(f"  - Embeddings: {type(embedder).__name__} using model: {embed_model}")
        print(f"  - Chat: {type(self.chat_model).__name__} using model: {chat_model}")

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an original Reddit comment generator. Your response MUST ONLY be the comment itself, with no additional text or explanations."),
                ("user", "Generate a relevant and original Reddit comment to the Reddit post given in the POST markup tags.\n"
                 "Use the given context in the CONTEXT markups tags to infer the dominant viewpoint and tone of the example comments (e.g., sarcastic, cynical, dismissive, etc.), and match that tone.\n"
                 "Emulate the prevailing viewpoint, style and sentiment of comments given in the context.\n\n"
                         "<POST>{post}</POST>\n\n"
                         "<CONTEXT>{context}</CONTEXT>\n")
            ]
        )

    def _parse_categories_json(self, categories_json_str: str) -> Dict[str, str]:
        """Parse categories_json string into a dictionary."""
        try:
            if isinstance(categories_json_str, str):
                parsed = json.loads(categories_json_str)
            elif isinstance(categories_json_str, dict):
                return categories_json_str
            else:
                return {}
            
            # Handle list format: [{"political_ideology": "Far Right"}, {"climate_attitude": "Disengaged"}]
            if isinstance(parsed, list):
                result = {}
                for item in parsed:
                    if isinstance(item, dict):
                        result.update(item)
                return result
            # Handle dict format: {"political_ideology": "Far Right", "climate_attitude": "Disengaged"}
            elif isinstance(parsed, dict):
                return parsed
            
            return {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def _check_enhanced_metadata(self) -> bool:
        """Check if the vector store has enhanced metadata with category_ fields."""
        try:
            # Get a small sample to check metadata structure
            results = self.vector_store.get(limit=1, include=["metadatas"])
            if results.get('metadatas') and len(results['metadatas']) > 0:
                metadata = results['metadatas'][0]
                # Check if any field starts with 'category_'
                return any(key.startswith('category_') for key in metadata.keys())
            return False
        except Exception:
            return False

    def _normalize_cluster_ids(self, cluster_ids: Union[int, List[int], None]) -> Optional[List[int]]:
        """Normalize cluster_ids input to a list of integers or None."""
        if cluster_ids is None:
            return None
        elif isinstance(cluster_ids, int):
            return [cluster_ids]
        elif isinstance(cluster_ids, list):
            return cluster_ids
        else:
            # Try to convert to int if it's a string or other type
            try:
                return [int(cluster_ids)]
            except (ValueError, TypeError):
                return None

    def _format_prompt_as_string(self, messages) -> str:
        """Convert formatted messages to a readable string for display."""
        prompt_parts = []
        for msg in messages:
            role = msg.type if hasattr(msg, 'type') else 'unknown'
            content = msg.content if hasattr(msg, 'content') else str(msg)
            prompt_parts.append(f"[{role.upper()}]\n{content}\n")
        return "\n".join(prompt_parts)

    def generate_reply(self, post_content: str, 
                    cluster_id: Optional[int] = None,
                    cluster_ids: Union[int, List[int], None] = None,
                    exclude_doc_id: Optional[str] = None, 
                    search_query: Optional[str] = None,
                    categories: Optional[Dict[str, str]] = None,
                    k: int = 6, lambda_mult: float = 0.5, fetch_k: int = 20) -> Dict:
        """
        Generates a simulated Reddit comment using MMR retrieval and optional metadata filtering.
        
        Args:
            post_content: The post content to generate a reply for
            cluster_id: Optional single cluster ID to filter documents (for backward compatibility)
            cluster_ids: Optional cluster ID(s) to filter documents - can be int, list of ints, or None
            exclude_doc_id: Optional document ID to exclude from context
            search_query: Optional search query to filter documents by metadata
            categories: Optional dictionary of categories to filter documents by (e.g., {"climate_attitude_six_americas": "Disengaged"})
            k: Number of documents to return after MMR re-ranking
            lambda_mult: Balance between relevance (1.0) and diversity (0.0) in MMR
            fetch_k: Number of initial documents to fetch before MMR re-ranking
        """
        
        # Handle backward compatibility - if both cluster_id and cluster_ids are provided, use cluster_ids
        if cluster_ids is not None:
            normalized_cluster_ids = self._normalize_cluster_ids(cluster_ids)
        elif cluster_id is not None:
            normalized_cluster_ids = self._normalize_cluster_ids(cluster_id)
        else:
            normalized_cluster_ids = None

        # Build ChromaDB metadata filter using where clause format
        where_conditions = []
        
        # Handle cluster filtering for multiple clusters
        if normalized_cluster_ids is not None:
            if len(normalized_cluster_ids) == 1:
                where_conditions.append({"cluster_id": {"$eq": normalized_cluster_ids[0]}})
            else:
                # ChromaDB may not support $in operator, so we'll use $or with multiple $eq conditions
                cluster_conditions = []
                for cid in normalized_cluster_ids:
                    cluster_conditions.append({"cluster_id": {"$eq": cid}})
                where_conditions.append({"$or": cluster_conditions})
        
        if search_query is not None:
            where_conditions.append({"search_query": {"$eq": search_query}})
        
        # OPTIMIZED: Add category filters directly to ChromaDB where clause
        # This uses the pre-indexed category_FRAMEWORK metadata fields
        category_filters = categories if categories else {}
        if category_filters:
            for cat_key, cat_value in category_filters.items():
                # Add filter using the category_ prefix that was set during add_documents
                where_conditions.append({f"category_{cat_key}": {"$eq": cat_value}})
                print(f"Adding category filter to ChromaDB: category_{cat_key} = {cat_value}")

        # Combine conditions with $and if multiple filters
        where_filter = None
        if len(where_conditions) == 1:
            where_filter = where_conditions[0]
        elif len(where_conditions) > 1:
            where_filter = {"$and": where_conditions}

        # For exclusion, increase fetch_k to account for documents we'll filter out
        adjusted_fetch_k = fetch_k
        if exclude_doc_id is not None:
            # Only need small increase now since category filtering happens at DB level
            adjusted_fetch_k = max(fetch_k + 5, int(fetch_k * 1.3))
        
        # Configure MMR search kwargs for ChromaDB
        search_kwargs = {
            "k": adjusted_fetch_k,  # Fetch more initially since we'll filter afterward
            "lambda_mult": lambda_mult,
            "fetch_k": min(adjusted_fetch_k * 2, 100)  # Fetch even more for MMR pool
        }
        
        # Add ChromaDB where filter if we have any filters to apply
        if where_filter is not None:
            search_kwargs["filter"] = where_filter
        
        _global_debug_logger.write_file("WHERE_FILTER", where_filter)
        _global_debug_logger.print("WHERE_FILTER", where_filter)
        _global_debug_logger.print("NORMALIZED_CLUSTER_IDS", normalized_cluster_ids)
        _global_debug_logger.print("CATEGORY_FILTERS", category_filters)

        try:
            # Use MMR search with ChromaDB
            retrieved_docs = self.vector_store.max_marginal_relevance_search(
                query=post_content,
                **search_kwargs
            )

            _global_debug_logger.write_file("RETRIEVED_DOCS_BEFORE_FILTERING", retrieved_docs)
            _global_debug_logger.print("RETRIEVED_DOCS_COUNT", len(retrieved_docs))

            # Category filtering now happens at ChromaDB level, so this is just a safety check
            # This fallback is only needed if ChromaDB filtering didn't work as expected
            if category_filters:
                initial_count = len(retrieved_docs)
                filtered_docs = []
                for doc in retrieved_docs:
                    # Verify the category filter was applied correctly
                    matches_all_filters = True
                    for cat_key, cat_value in category_filters.items():
                        db_value = doc.metadata.get(f"category_{cat_key}")
                        if db_value != cat_value:
                            # This shouldn't happen if DB filtering worked, but keep as safety
                            matches_all_filters = False
                            break
                    
                    if matches_all_filters:
                        filtered_docs.append(doc)
                
                if len(filtered_docs) < initial_count:
                    print(f"Safety filter removed {initial_count - len(filtered_docs)} documents (DB filter may have failed)")
                    retrieved_docs = filtered_docs

            # Handle exclude_doc_id after retrieval
            if exclude_doc_id:
                initial_count = len(retrieved_docs)
                retrieved_docs = [doc for doc in retrieved_docs 
                                if doc.metadata.get("comment_id") != exclude_doc_id]
                excluded_count = initial_count - len(retrieved_docs)
                if excluded_count > 0:
                    print(f"Excluded {excluded_count} document(s) by comment_id: {exclude_doc_id}")
                    _global_debug_logger.print("EXCLUDED_DOC_ID", exclude_doc_id)

            # Limit to requested k documents
            retrieved_docs = retrieved_docs[:k]
            
            # Check if we have any documents left
            if not retrieved_docs:
                return {
                    "result": "No relevant context found with the specified filters. Unable to generate a meaningful reply.",
                    "source_documents": [],
                    "full_prompt": "No documents found - no prompt was generated.",
                    "mmr_params": {
                        "k": k,
                        "lambda_mult": lambda_mult,
                        "fetch_k": adjusted_fetch_k,
                        "filters_applied": where_filter,
                        "category_filters": category_filters,
                        "excluded_doc_id": exclude_doc_id,
                        "cluster_ids_used": normalized_cluster_ids,
                        "final_doc_count": 0
                    }
                }
            
            # Format retrieved documents for context
            context_str = "\n".join([doc.page_content for doc in retrieved_docs])
            
            # Prepare input for the chat model
            inputs = {"post": post_content, "context": context_str}

            _global_debug_logger.write_file("INPUTS", inputs)

            # Format the prompt messages
            formatted_messages = self.prompt_template.format_messages(**inputs)
            
            # Convert formatted messages to a readable string for display
            full_prompt_string = self._format_prompt_as_string(formatted_messages)

            # Generate the response
            generated_comment = self.chat_model.invoke(formatted_messages)
            
            # Use a regular expression to find and remove the <think> tag and its contents
            clean_comment = re.sub(r'<think>.*?</think>', '', generated_comment.content, flags=re.DOTALL).strip()

            return {
                "result": clean_comment, 
                "source_documents": retrieved_docs,
                "full_prompt": full_prompt_string,
                "mmr_params": {
                    "k": k,
                    "lambda_mult": lambda_mult,
                    "fetch_k": adjusted_fetch_k,
                    "filters_applied": where_filter,
                    "category_filters": category_filters,
                    "excluded_doc_id": exclude_doc_id,
                    "final_doc_count": len(retrieved_docs),
                    "cluster_ids_used": normalized_cluster_ids
                }
            }
            
        except Exception as e:
            print(f"Error during ChromaDB MMR retrieval or generation: {e}")
            _global_debug_logger.write_file("ERROR", str(e))
            _global_debug_logger.print("ERROR_DETAILS", {
                "error": str(e),
                "where_filter": where_filter,
                "cluster_ids": normalized_cluster_ids,
                "category_filters": category_filters,
                "search_kwargs": search_kwargs
            })
            
            # Try fallback approach if the filter failed
            if where_filter is not None:
                print("Attempting fallback without metadata filter...")
                try:
                    fallback_docs = self.vector_store.max_marginal_relevance_search(
                        query=post_content,
                        k=adjusted_fetch_k,
                        lambda_mult=lambda_mult,
                        fetch_k=min(adjusted_fetch_k * 2, 100)
                    )
                    
                    # Apply filtering manually after retrieval
                    if normalized_cluster_ids is not None:
                        fallback_docs = [doc for doc in fallback_docs 
                                       if doc.metadata.get("cluster_id") in normalized_cluster_ids]
                    
                    if search_query is not None:
                        fallback_docs = [doc for doc in fallback_docs 
                                       if doc.metadata.get("search_query") == search_query]
                    
                    # Apply category filters manually
                    if category_filters:
                        filtered_fallback = []
                        for doc in fallback_docs:
                            matches_all_filters = True
                            for cat_key, cat_value in category_filters.items():
                                db_value = doc.metadata.get(f"category_{cat_key}")
                                if db_value != cat_value:
                                    matches_all_filters = False
                                    break
                            
                            if matches_all_filters:
                                filtered_fallback.append(doc)
                        
                        print(f"Category filtering (fallback): {len(fallback_docs)} -> {len(filtered_fallback)} documents")
                        fallback_docs = filtered_fallback
                    
                    if fallback_docs:
                        print(f"Fallback successful: found {len(fallback_docs)} documents")
                        retrieved_docs = fallback_docs

                        # Handle exclude_doc_id
                        if exclude_doc_id:
                            initial_count = len(retrieved_docs)
                            retrieved_docs = [doc for doc in retrieved_docs 
                                            if doc.metadata.get("comment_id") != exclude_doc_id]
                            excluded_count = initial_count - len(retrieved_docs)
                            if excluded_count > 0:
                                _global_debug_logger.print("EXCLUDED_DOC_ID", exclude_doc_id)

                        # Limit to requested k documents
                        retrieved_docs = retrieved_docs[:k]
                        
                        if retrieved_docs:
                            # Format retrieved documents for context
                            context_str = "\n".join([doc.page_content for doc in retrieved_docs])
                            
                            # Prepare input for the chat model
                            inputs = {"post": post_content, "context": context_str}
                            _global_debug_logger.write_file("INPUTS", inputs)

                            # Format the prompt messages
                            formatted_messages = self.prompt_template.format_messages(**inputs)
                            
                            # Convert formatted messages to a readable string for display
                            full_prompt_string = self._format_prompt_as_string(formatted_messages)

                            # Generate the response
                            generated_comment = self.chat_model.invoke(formatted_messages)
                            
                            # Use a regular expression to find and remove the <think> tag and its contents
                            clean_comment = re.sub(r'<think>.*?</think>', '', generated_comment.content, flags=re.DOTALL).strip()

                            return {
                                "result": clean_comment, 
                                "source_documents": retrieved_docs,
                                "full_prompt": full_prompt_string,
                                "mmr_params": {
                                    "k": k,
                                    "lambda_mult": lambda_mult,
                                    "fetch_k": adjusted_fetch_k,
                                    "filters_applied": where_filter,
                                    "category_filters": category_filters,
                                    "excluded_doc_id": exclude_doc_id,
                                    "final_doc_count": len(retrieved_docs),
                                    "cluster_ids_used": normalized_cluster_ids,
                                    "fallback_used": True
                                }
                            }
                    
                except Exception as fallback_error:
                    print(f"Fallback also failed: {fallback_error}")
                    _global_debug_logger.write_file("FALLBACK_ERROR", str(fallback_error))
            
            return {
                "result": "Error generating comment with MMR retrieval.",
                "source_documents": [],
                "full_prompt": f"Error occurred: {str(e)}",
                "mmr_params": {
                    "error": str(e),
                    "k": k,
                    "lambda_mult": lambda_mult,
                    "fetch_k": adjusted_fetch_k,
                    "cluster_ids_used": normalized_cluster_ids,
                    "category_filters": category_filters
                }
            }

    def add_documents(self, documents: List[Document]):
        """Adds documents to the vector store with parsed categories."""
        processed_documents = []
        
        for doc in documents:
            # Create a copy of the document with parsed categories
            new_metadata = doc.metadata.copy()
            
            # Parse categories_json and add individual category fields
            categories_json_str = new_metadata.get("categories_json", "{}")
            parsed_categories = self._parse_categories_json(categories_json_str)
            
            # Add each category as a separate metadata field for easier filtering
            for category_key, category_value in parsed_categories.items():
                new_metadata[f"category_{category_key}"] = category_value
            
            # Create new document with updated metadata
            processed_doc = Document(
                page_content=doc.page_content,
                metadata=new_metadata
            )
            processed_documents.append(processed_doc)
        
        self.vector_store.add_documents(processed_documents)

    def retrieve_context(self, query: str, cluster_id: Optional[int] = None) -> List[Document]:
        """Retrieves relevant context documents from the vector store."""
        retriever_kwargs = {}
        if cluster_id is not None:
            retriever_kwargs['filter'] = {"cluster_id": cluster_id}

        retriever = self.vector_store.as_retriever(search_kwargs=retriever_kwargs)
        return retriever.invoke(query)