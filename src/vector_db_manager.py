# src/vector_db_manager.py
#
# By Ian Drumm, The University of Salford, UK.
#
import json
from typing import List, Dict, Optional
from langchain_chroma import Chroma
from langchain.schema import Document
import os

class VectorDBManager:
    def __init__(self, persist_directory: str = "chroma_db", 
                 embed_model: str = "nomic-embed-text:latest",
                 embed_backend: Optional[str] = None,
                 embed_api_key: Optional[str] = None):
        """
        Initialize VectorDBManager with multi-backend embedding support.
        
        Args:
            persist_directory: Path to Chroma database
            embed_model: Embedding model name
            embed_backend: Backend for embedding model ("openai", "gemini", "ollama", or None for auto-detect)
            embed_api_key: API key for embedding backend
        """
        from src.rag_chatbot import create_embeddings
        
        self.persist_directory = persist_directory
        self.embedder = create_embeddings(
            model_name=embed_model,
            backend=embed_backend,
            api_key=embed_api_key
        )
        self.db = self._initialize_vectorstore()
        
        print(f"Initialized VectorDBManager with {type(self.embedder).__name__} using model: {embed_model}")

    def _initialize_vectorstore(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        # Initialize with the embedding function; it will load existing data if present
        return Chroma(embedding_function=self.embedder, persist_directory=self.persist_directory)

    @staticmethod
    def _clean_text(text):
        """
        Clean text by normalizing problematic Unicode characters.
        
        Args:
            text: String to clean, or any other type (returned as-is)
            
        Returns:
            Cleaned string or original value if not a string
        """
        if not isinstance(text, str):
            return text
        
        # Replace smart quotes and other problematic Unicode characters
        replacements = {
            '\u2018': "'",  # Left single quote
            '\u2019': "'",  # Right single quote/apostrophe
            '\u201c': '"',  # Left double quote
            '\u201d': '"',  # Right double quote
            '\u2013': '-',  # En dash
            '\u2014': '-',  # Em dash
            '\u2026': '...',  # Ellipsis
            '\u00a0': ' ',  # Non-breaking space
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text

    def _parse_categories(self, categories_data) -> Dict[str, str]:
        """
        Parse categories from different input formats into a consistent dictionary.
        
        Args:
            categories_data: Can be a list, dict, or string
            
        Returns:
            Dict[str, str]: Parsed categories as key-value pairs
        """
        try:
            if isinstance(categories_data, str):
                # If it's a JSON string, parse it
                parsed = json.loads(categories_data)
            elif isinstance(categories_data, (list, dict)):
                parsed = categories_data
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
            
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            print(f"Warning: Failed to parse categories: {categories_data}, Error: {e}")
            return {}

    def add_documents_from_json(self, file_path: str):
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)

        new_docs = []
        doc_ids = []

        for item in data:
            # Clean text content at ingestion
            post_content = self._clean_text(item.get("post", ""))
            comment_content = self._clean_text(item.get("comment", ""))
            doc_content = f"Post: {post_content}\nComment: {comment_content}"

            comment_id = item.get("comment_metadata", {}).get("id")
            
            if not comment_id:
                print(f"Warning: Skipping document due to missing comment_id in item: {item.get('comment_metadata')}")
                continue
            
            raw_scores = item.get("scores", {})
            scores_as_string = json.dumps(raw_scores)

            raw_categories = item.get("categories", [])
            categories_as_string = json.dumps(raw_categories)

            raw_justifications = item.get("justifications", [])
            justifications_as_string = json.dumps(raw_justifications)

            # Parse categories for individual field storage
            parsed_categories = self._parse_categories(raw_categories)

            # Build base metadata with text cleaning
            metadata = {
                "scores_json": scores_as_string,
                "comment_id": self._clean_text(str(comment_id)),
                "cluster_id": item.get("cluster_id"),
                "subreddit": self._clean_text(item.get("post_metadata", {}).get("subreddit", "")),
                "created_utc": item.get("comment_metadata", {}).get("created_utc"),
                "search_query": self._clean_text(item.get("post_metadata", {}).get("search_query", "")),
                "prompts_path": self._clean_text(item.get("prompts_path", "")),
                "categories_json": categories_as_string,
                "justifications_json": justifications_as_string
            }
            
            # Add individual category fields with prefix for easy filtering
            for category_key, category_value in parsed_categories.items():
                # Clean up the key name (remove special characters, make it ChromaDB-friendly)
                clean_key = category_key.replace(" ", "_").replace("-", "_").lower()
                # Clean the category value text
                metadata[f"category_{clean_key}"] = self._clean_text(str(category_value))
            
            # Check if UMAP coordinates are present and add them to metadata
            if "umap_x" in item and "umap_y" in item:
                metadata["umap_x"] = item["umap_x"]
                metadata["umap_y"] = item["umap_y"]
            
            new_docs.append(Document(page_content=doc_content, metadata=metadata))
            doc_ids.append(comment_id)

        if new_docs:
            print(f"Attempting to add/update {len(new_docs)} documents to the vector database...")
            # Show example of enhanced metadata for first document
            if new_docs:
                print("Example enhanced metadata structure:")
                example_metadata = new_docs[0].metadata
                for key, value in example_metadata.items():
                    if key.startswith('category_'):
                        print(f"  {key}: {value}")
            
            self.db.add_documents(new_docs, ids=doc_ids)
            print("Documents added/updated successfully with enhanced category metadata.")
        else:
            print("No new documents to add or all documents skipped due to missing IDs.")

    def get_unique_metadata_values(self, metadata_field: str) -> List[str]:
        """Get unique values for a metadata field."""
        try:
            results = self.db.get(include=["metadatas"])
            all_values = [
                md.get(metadata_field)
                for md in results['metadatas']
                if md and md.get(metadata_field) is not None
            ]
            unique_values = sorted(list(set(all_values)))
            return unique_values
        except Exception as e:
            print(f"Error retrieving unique metadata for field '{metadata_field}': {e}")
            return []

    def get_unique_categories(self) -> Dict[str, List[str]]:
        """
        Get all unique category types and their possible values.
        
        Returns:
            Dict where keys are category types and values are lists of possible values
        """
        try:
            results = self.db.get(include=["metadatas"])
            categories = {}
            
            for metadata in results.get('metadatas', []):
                if not metadata:
                    continue
                    
                # Look for all fields that start with 'category_'
                for key, value in metadata.items():
                    if key.startswith('category_'):
                        category_type = key[9:]  # Remove 'category_' prefix
                        if category_type not in categories:
                            categories[category_type] = set()
                        categories[category_type].add(str(value))
            
            # Convert sets to sorted lists
            return {k: sorted(list(v)) for k, v in categories.items()}
            
        except Exception as e:
            print(f"Error retrieving unique categories: {e}")
            return {}

    def get_category_distribution(self) -> Dict[str, Dict[str, int]]:
        """
        Get the distribution of values for each category type.
        
        Returns:
            Dict where keys are category types and values are dicts of value->count
        """
        try:
            results = self.db.get(include=["metadatas"])
            distribution = {}
            
            for metadata in results.get('metadatas', []):
                if not metadata:
                    continue
                    
                for key, value in metadata.items():
                    if key.startswith('category_'):
                        category_type = key[9:]  # Remove 'category_' prefix
                        if category_type not in distribution:
                            distribution[category_type] = {}
                        
                        value_str = str(value)
                        distribution[category_type][value_str] = distribution[category_type].get(value_str, 0) + 1
            
            return distribution
            
        except Exception as e:
            print(f"Error retrieving category distribution: {e}")
            return {}

    def get_retriever(self, search_type: str = "mmr", filter_cluster_id: Optional[int] = None,
                     category_filters: Optional[Dict[str, str]] = None):
        """
        Get a retriever with optional filtering.
        
        Args:
            search_type: Type of search ("mmr", "similarity", etc.)
            filter_cluster_id: Optional cluster ID to filter by
            category_filters: Optional dict of category filters (e.g., {"political_ideology": "Far Right"})
        """
        search_kwargs = {}
        
        # Build filter conditions
        where_conditions = []
        
        if filter_cluster_id is not None:
            where_conditions.append({"cluster_id": {"$eq": filter_cluster_id}})
        
        if category_filters:
            for category_key, category_value in category_filters.items():
                clean_key = category_key.replace(" ", "_").replace("-", "_").lower()
                where_conditions.append({f"category_{clean_key}": {"$eq": str(category_value)}})
        
        # Combine conditions
        if len(where_conditions) == 1:
            search_kwargs["filter"] = where_conditions[0]
        elif len(where_conditions) > 1:
            search_kwargs["filter"] = {"$and": where_conditions}
        
        return self.db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    
    def count_documents(self):
        try:
            return self.db._collection.count()
        except Exception as e:
            print(f"Error counting documents: {e}")
            return 0

    def inspect_sample_metadata(self, limit: int = 3):
        """
        Print sample metadata to understand the structure.
        """
        try:
            results = self.db.get(limit=limit, include=["metadatas"])
            print(f"Sample metadata from {len(results.get('metadatas', []))} documents:")
            
            for i, metadata in enumerate(results.get('metadatas', [])[:limit]):
                print(f"\n--- Document {i+1} ---")
                for key, value in sorted(metadata.items()):
                    if key.startswith('category_'):
                        print(f"  {key}: {value}")
                    elif key in ['comment_id', 'cluster_id', 'subreddit']:
                        print(f"  {key}: {value}")
                        
        except Exception as e:
            print(f"Error inspecting metadata: {e}")

    def update_document_clusters(self, doc_updates: List[Dict]):
        """
        Update the cluster_id, umap_x, and umap_y metadata fields for a list of documents.

        Args:
            doc_updates: A list of dictionaries, each with 'id', 'cluster_id',
                         'umap_x', and 'umap_y' keys for the document to update.
        """
        # Create lists for the update call
        ids = [doc['id'] for doc in doc_updates]
        metadatas = []
        for doc in doc_updates:
            metadata = {
                "cluster_id": doc["cluster_id"],
                "umap_x": doc["umap_x"],
                "umap_y": doc["umap_y"]
            }
            metadatas.append(metadata)

        # Call the update method on the underlying ChromaDB collection
        try:
            # CORRECTED LINE: Use self.db._collection.update()
            self.db._collection.update(
                ids=ids,
                metadatas=metadatas
            )
            print(f"Successfully updated cluster information for {len(ids)} documents.")
        except Exception as e:
            print(f"Error updating documents: {e}")