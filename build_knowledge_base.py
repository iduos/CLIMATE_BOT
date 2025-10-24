# build_knowledge_base.py
#
# By Ian Drumm, The University of Salford, UK.
#
# Commands for building and managing the Reddit comment knowledge base:
# - scrape: Scrape Reddit comments
# - score: Score comments using LLM prompts
# - cluster: Perform clustering on scored comments
# - json-to-db: Add clustered data to vector database
# - rescore: Re-score existing data with updated prompts
# - process-all: Run the full pipeline (scrape, score, cluster, add to DB)
#
import argparse
import json
import csv
import datetime
import os
import random
from typing import List, Dict, Any, Optional

from src.scraper import RedditScraper
from src.scraper import format_search_query
from src.data_cleaning import clean_reddit_data
from src.score_and_categorise_gpus import CommentScorer
from src.cluster import ClusterVisualizer
from src.vector_db_manager import VectorDBManager


def score_comments_data(input_data, prompts_path: str, model: str = "llama3.1:8B", backend: str = "ollama", include_justifications: bool=False, openai_api_key: Optional[str] = None):
    print(f"Scoring with {model}")
    scorer = CommentScorer(prompts_path=prompts_path, model=model,backend=backend)
    scored = scorer.score_batch(input_data, include_justifications=include_justifications)
    return scored


def cluster_comments_data(input_data, n_clusters: int, start_date: str, end_date: str, clustering_method: str = "gower_kmedoids"):
    # Parse date strings into datetime.date objects for ClusterVisualizer
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    clusterer = ClusterVisualizer(n_clusters=n_clusters)
    print(f"Performing {clustering_method} clustering on {len(input_data)} scored comments (clusters: {n_clusters})...")
    clustered = clusterer.run_clustering(input_data, date_range=(start, end), clustering_method=clustering_method)
    if not clustered:
        print(f"No data matched the date range ({start_date} to {end_date}) for clustering. Clustered output will be empty.")
    return clustered


def write_log_entry(log_data: Dict[str, Any], log_dir: str = "logs") -> None:
    """
    Write a log entry with pipeline execution details.
    
    Args:
        log_data: Dictionary containing log information
        log_dir: Directory to save log files
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"reddit_analysis_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # Write log entry
    with open(log_path, "w") as f:
        f.write("REDDIT ANALYSIS PIPELINE LOG\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\nEXECUTION DETAILS:\n")
        f.write("-" * 20 + "\n")
        
        for key, value in log_data.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for sub_key, sub_value in value.items():
                    f.write(f"  {sub_key}: {sub_value}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"\nLog file saved to: {log_path}")


def score_comments(input_path: str, output_path: str, prompts_path: str, model: str = "llama3.1:8B", include_justifications: bool=False):
    with open(input_path, "r") as f:
        data = json.load(f)
    scored = score_comments_data(data, prompts_path, model, include_justifications)
    with open(output_path, "w") as f:
        json.dump(scored, f, indent=2)
    print(f"Scored data saved to {output_path}")


def cluster_comments(input_path: str, output_path: str, n_clusters: int, start_date: str, end_date: str, clustering_method: str = "gower_kmedoids"):
    with open(input_path, "r") as f:
        data = json.load(f)
    clustered = cluster_comments_data(data, n_clusters, start_date, end_date, clustering_method=clustering_method)
    with open(output_path, "w") as f:
        json.dump(clustered, f, indent=2)
    print("Clustering complete. Clustered data saved to " + output_path)


def add_to_vector_db(clustered_data_path: str, vector_db_path: str, 
                     embed_model: str = "nomic-embed-text:latest",
                     embed_backend: Optional[str] = None,
                     embed_api_key: Optional[str] = None):
    """
    Add clustered data to a vector database.
    
    Args:
        clustered_data_path: Path to the clustered JSON data file
        vector_db_path: Path to the vector database directory
        embed_model: Embedding model to use for the vector database
        embed_backend: Embedding backend (openai, gemini, ollama)
        embed_api_key: API key for embedding backend
    """
    print(f"Step 4/4: Adding clustered data to vector database at {vector_db_path}...")
    
    try:
        # Initialize the vector database manager
        vector_db_manager = VectorDBManager(
            persist_directory=vector_db_path,
            embed_model=embed_model,
            embed_backend=embed_backend,
            embed_api_key=embed_api_key
        )
        
        # Get document count before adding
        initial_count = vector_db_manager.count_documents()
        print(f"Vector database currently contains {initial_count} documents")
        
        # Add documents from the clustered JSON file
        vector_db_manager.add_documents_from_json(clustered_data_path)
        
        # Get document count after adding
        final_count = vector_db_manager.count_documents()
        print(f"Vector database now contains {final_count} documents ({final_count - initial_count} added/updated)")
        
        return final_count - initial_count
        
    except Exception as e:
        print(f"Error adding data to vector database: {e}")
        raise


def rescore_database(
    input_source: str,
    vector_db_path: str,
    scoring_prompts: str,
    output_json: Optional[str] = None,
    scorer_model: str = "llama3.1:8B",
    scorer_backend: str = "ollama",
    include_justifications: bool = False,
    sample_size: Optional[int] = None,
    filter_cluster_id: Optional[int] = None,
    embed_model: str = "nomic-embed-text:latest",
    embed_backend: Optional[str] = None,
    embed_api_key: Optional[str] = None,
):
    """
    Re-score existing data with updated prompts/rubrics without re-scraping Reddit.
    
    Args:
        input_source: Either 'vector_db' to read from ChromaDB, or path to JSON file
        vector_db_path: Path to vector database directory
        scoring_prompts: Path to new/updated prompts JSON file
        output_json: Optional path to save re-scored JSON data before updating DB
        scorer_model: Model to use for scoring
        scorer_backend: Backend to use (ollama, openai, gemini, anthropic)
        include_justifications: Whether to include justifications
        sample_size: Optional limit on number of items to rescore
        filter_cluster_id: Optional filter to only rescore specific cluster
        embed_model: Embedding model for vector database
        embed_backend: Embedding backend (openai, gemini, ollama)
        embed_api_key: API key for embedding backend
    """
    print("="*60)
    print("RE-SCORING DATABASE WITH UPDATED PROMPTS")
    print("="*60)
    
    # Step 1: Load data from source
    print(f"\nStep 1/3: Loading data from {input_source}...")
    
    items_to_score = []
    
    if input_source.lower() == 'vector_db':
        # Load from vector database
        vector_db_manager = VectorDBManager(
            persist_directory=vector_db_path,
            embed_model=embed_model,
            embed_backend=embed_backend,
            embed_api_key=embed_api_key
        )
        
        # Get all documents
        print(f"Reading from vector database at {vector_db_path}...")
        results = vector_db_manager.db.get(include=["documents", "metadatas"])
        
        documents = results.get('documents', [])
        metadatas = results.get('metadatas', [])
        ids = results.get('ids', [])
        
        print(f"Found {len(documents)} documents in vector database")
        
        # Parse documents back into post/comment pairs
        for doc_content, metadata, doc_id in zip(documents, metadatas, ids):
            # Parse "Post: {post}\nComment: {comment}" format
            if doc_content.startswith("Post: ") and "\nComment: " in doc_content:
                parts = doc_content.split("\nComment: ", 1)
                post = parts[0].replace("Post: ", "", 1)
                comment = parts[1] if len(parts) > 1 else ""
                
                # Filter by cluster if specified
                if filter_cluster_id is not None:
                    if metadata.get("cluster_id") != filter_cluster_id:
                        continue
                
                # Reconstruct item with all metadata
                item = {
                    "post": post,
                    "comment": comment,
                    "comment_metadata": {
                        "id": metadata.get("comment_id"),
                        "created_utc": metadata.get("created_utc"),
                    },
                    "post_metadata": {
                        "subreddit": metadata.get("subreddit"),
                        "search_query": metadata.get("search_query"),
                    },
                    "cluster_id": metadata.get("cluster_id"),
                    "umap_x": metadata.get("umap_x"),
                    "umap_y": metadata.get("umap_y"),
                    "_original_id": doc_id,
                }
                items_to_score.append(item)
    else:
        # Load from JSON file
        print(f"Reading from JSON file: {input_source}")
        with open(input_source, "r") as f:
            items_to_score = json.load(f)
    
    if not items_to_score:
        print("No items found to rescore. Exiting.")
        return
    
    # Apply sample size limit if specified
    if sample_size and sample_size < len(items_to_score):
        import random
        items_to_score = random.sample(items_to_score, sample_size)
        print(f"Sampling {sample_size} items for rescoring")
    
    print(f"Prepared {len(items_to_score)} items for rescoring")
    
    # Step 2: Re-score with new prompts
    print(f"\nStep 2/3: Re-scoring with updated prompts from {scoring_prompts}...")
    print(f"Using model: {scorer_model} (backend: {scorer_backend})")
    
    rescored_data = score_comments_data(
        items_to_score,
        prompts_path=scoring_prompts,
        model=scorer_model,
        backend=scorer_backend,
        include_justifications=include_justifications
    )
    
    print(f"Successfully re-scored {len(rescored_data)} items")
    
    # Preserve original metadata (cluster_id, UMAP coordinates, etc.)
    for i, (original_item, scored_item) in enumerate(zip(items_to_score, rescored_data)):
        if "cluster_id" in original_item:
            scored_item["cluster_id"] = original_item["cluster_id"]
        if "umap_x" in original_item:
            scored_item["umap_x"] = original_item["umap_x"]
        if "umap_y" in original_item:
            scored_item["umap_y"] = original_item["umap_y"]
        if "_original_id" in original_item:
            scored_item["_original_id"] = original_item["_original_id"]
        
        # Update prompts_path to track which rubric was used
        if "post_metadata" not in scored_item:
            scored_item["post_metadata"] = {}
        scored_item["prompts_path"] = scoring_prompts
    
    # Save to JSON if requested
    if output_json:
        with open(output_json, "w") as f:
            json.dump(rescored_data, f, indent=2)
        print(f"\nRe-scored data saved to {output_json}")
    
    # Step 3: Update vector database
    print(f"\nStep 3/3: Updating vector database at {vector_db_path}...")
    
    vector_db_manager = VectorDBManager(
        persist_directory=vector_db_path,
        embed_model=embed_model,
        embed_backend=embed_backend,
        embed_api_key=embed_api_key
    )
    
    # Update documents with new scores and categories
    updated_count = 0
    for item in rescored_data:
        comment_id = item.get("comment_metadata", {}).get("id")
        if not comment_id:
            print(f"Warning: Skipping item without comment_id")
            continue
        
        # Prepare updated metadata
        raw_scores = item.get("scores", {})
        scores_as_string = json.dumps(raw_scores)
        
        raw_categories = item.get("categories", {})
        categories_as_string = json.dumps(raw_categories)
        
        raw_justifications = item.get("justifications", {})
        justifications_as_string = json.dumps(raw_justifications)
        
        # Parse categories for individual field storage
        parsed_categories = vector_db_manager._parse_categories(raw_categories)
        
        # Build updated metadata
        updated_metadata = {
            "scores_json": scores_as_string,
            "categories_json": categories_as_string,
            "justifications_json": justifications_as_string,
            "prompts_path": scoring_prompts,
        }
        
        # Add individual category fields
        for category_key, category_value in parsed_categories.items():
            clean_key = category_key.replace(" ", "_").replace("-", "_").lower()
            updated_metadata[f"category_{clean_key}"] = vector_db_manager._clean_text(str(category_value))
        
        # Preserve cluster and UMAP data if present
        if "cluster_id" in item:
            updated_metadata["cluster_id"] = item["cluster_id"]
        if "umap_x" in item:
            updated_metadata["umap_x"] = item["umap_x"]
        if "umap_y" in item:
            updated_metadata["umap_y"] = item["umap_y"]
        
        # Update the document
        try:
            vector_db_manager.db._collection.update(
                ids=[str(comment_id)],
                metadatas=[updated_metadata]
            )
            updated_count += 1
        except Exception as e:
            print(f"Error updating document {comment_id}: {e}")
    
    print(f"Successfully updated {updated_count} documents in vector database")
    print("\n" + "="*60)
    print("RE-SCORING COMPLETE")
    print("="*60)
    print(f"Items re-scored: {len(rescored_data)}")
    print(f"Database documents updated: {updated_count}")
    print(f"New prompts file: {scoring_prompts}")


def export_scores_to_csv(scored_data, csv_path: str, include_justifications: bool = False) -> None:
    """
    Extracts post, comment, score, category, and (optionally) justification fields from scored JSON data 
    and writes them to a CSV in a defined column order.
    
    Args:
        scored_data: The scored data (list of dictionaries).
        csv_path: Output path for the CSV file.
        include_justifications: Whether to include justifications in the CSV output.
        
    Raises:
        FileNotFoundError: If the JSON file doesn't exist.
        json.JSONDecodeError: If the JSON file is malformed.
        PermissionError: If unable to write to the CSV path.
    """
    if not scored_data:
        print("No scored data found. CSV not created.")
        return

    # Validate that scored_data is a list
    if not isinstance(scored_data, list):
        raise ValueError("Expected JSON data to be a list of items")

    flattened_rows = []
    score_keys = set()
    category_keys = set()
    justification_keys = set()

    for i, item in enumerate(scored_data):
        if not isinstance(item, dict):
            print(f"Warning: Item {i} is not a dictionary, skipping...")
            continue
            
        row: Dict[str, Any] = {
            "post": item.get("post", ""),
            "comment": item.get("comment", "")
        }

        # Handle scores
        scores = item.get("scores", {})
        if isinstance(scores, dict):
            # Add scores with original keys
            for key, value in scores.items():
                row[key] = value
                score_keys.add(key)
        elif scores is not None:
            print(f"Warning: Item {i} has non-dict scores field, skipping scores...")

        # Handle categories - FIXED: Add categories as columns with their values
        categories = item.get("categories", {})
        if isinstance(categories, dict):
            # Add categories with original keys and their values
            for key, value in categories.items():
                row[key] = value
                category_keys.add(key)  # Collect the key names, not the values
        elif categories is not None:
            print(f"Warning: Item {i} has non-dict categories field, skipping categories...")

        # Handle justifications
        if include_justifications:
            justifications = item.get("justifications", {})
            if isinstance(justifications, dict):
                # Add justifications with "_justification" suffix to avoid key conflicts
                for key, value in justifications.items():
                    justification_key = f"{key}_justification"
                    row[justification_key] = value
                    justification_keys.add(justification_key)
            elif justifications is not None:
                print(f"Warning: Item {i} has non-dict justifications field, skipping justifications...")

        flattened_rows.append(row)

    if not flattened_rows:
        print("No valid data rows found. CSV not created.")
        return

    # Define column order: post, comment, score columns, category columns, then justification columns
    fieldnames = (
        ["post", "comment"] +
        sorted(score_keys) +
        sorted(category_keys) +  # FIXED: Now includes category columns
        (sorted(justification_keys) if include_justifications else [])
    )

    # Create directory if it doesn't exist
    csv_dir = os.path.dirname(csv_path)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)

    try:
        with open(csv_path, "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened_rows)
    except PermissionError:
        raise PermissionError(f"Permission denied writing to {csv_path}")
    except Exception as e:
        raise Exception(f"Error writing CSV file {csv_path}: {e}")

    print(f"Scored / Categorised data CSV saved to {csv_path} ({len(flattened_rows)} rows)")


def process_all_comments_pipeline(
    subreddits: List[str],
    query: str,
    scrape_limit: int,
    scoring_prompts: str,
    n_clusters: int,
    start_date: str,
    end_date: str,
    bin_by_period: Optional[str],  # 'day', 'week', 'month', or None
    output: str,
    auto_format_query: bool = True,
    scorer_model: str = "llama3.1:8B",
    scorer_backend: str = "ollama",
    sample_size_for_umap: int = 1000,
    include_justifications: bool = False,
    vector_db_path: Optional[str] = None,
    embed_model: str = "nomic-embed-text:latest",
    embed_backend: Optional[str] = None,
    embed_api_key: Optional[str] = None,
    save_intermediate_files: bool = False,
    clustering_method: str = "gower_kmedoids",
    enable_logging: bool = False,
):
    # Initialize log data
    log_data = {
        "search_query": query,
        "subreddits": subreddits,
        "scrape_limit": scrape_limit,
        "scoring_prompts": scoring_prompts,
        "n_clusters": n_clusters,
        "date_range": {"start": start_date, "end": end_date},
        "scorer_model": scorer_model,
        "scorer_backend": scorer_backend,
        "embed_model": embed_model,
        "embed_backend": embed_backend or "auto-detect",
        "sample_size": sample_size_for_umap,
        "vector_db_path": vector_db_path,
        "output_path": output
    }
    
    # Track overall timing
    pipeline_start_time = datetime.datetime.now()
    
    # Handle comma-separated subreddits string
    if len(subreddits) == 1 and ',' in subreddits[0]:
        subreddits = [sub.strip() for sub in subreddits[0].split(',')]
    
    # Auto-format the search query
    if auto_format_query:
        query = format_search_query(query)
    
    # Define paths for intermediate files
    RAW_DATA_PATH = output.replace('.json', '_raw.json')
    CLEANED_DATA_PATH = output.replace('.json', '_cleaned.json')
    SAMPLED_DATA_PATH = output.replace('.json', '_sampled.json')
    SCORED_DATA_PATH = output.replace('.json', '_scored.json')

    total_steps = 5 if vector_db_path else 4

    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output), exist_ok=True)
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)

    # Parse date strings into datetime objects
    parsed_start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    parsed_end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    # Step 1: Scrape
    print(f"Step 1/{total_steps}: Scraping Reddit comments...")
    print(f"Target subreddits: {subreddits}")
    print(f"Final search query: {query}")
    
    scrape_start_time = datetime.datetime.now()
    scraper = RedditScraper()
    raw_data = scraper.scrape_multiple(
        subreddits=subreddits,
        query=query,
        limit=scrape_limit,
        start_date=parsed_start_date,
        end_date=parsed_end_date,
        bin_by=bin_by_period # 'day', 'week', 'month', or None
    )
    scrape_end_time = datetime.datetime.now()
    
    log_data["scraping"] = {
        "raw_items_pulled": len(raw_data),
        "duration_seconds": (scrape_end_time - scrape_start_time).total_seconds(),
        "start_time": scrape_start_time.strftime('%Y-%m-%d %H:%M:%S'),
        "end_time": scrape_end_time.strftime('%Y-%m-%d %H:%M:%S')
    }

    if not raw_data:
        print("No data scraped within the specified parameters or date range. Exiting pipeline.")
        with open(output, "w") as f:
            json.dump([], f, indent=2)
        
        if enable_logging:
            write_log_entry(log_data)
        return

    if save_intermediate_files:
        with open(RAW_DATA_PATH, "w") as f:
            json.dump(raw_data, f, indent=2)
            print(f"Scraped {len(raw_data)} comments. Raw data saved to {RAW_DATA_PATH}")

    # Step 2: Clean data of items with unusable comments
    print(f"Step 2/{total_steps}: Cleaning data of unusable comments...")
    clean_start_time = datetime.datetime.now()
    cleaned_data = clean_reddit_data(raw_data)
    clean_end_time = datetime.datetime.now()
    
    log_data["cleaning"] = {
        "items_before_cleaning": len(raw_data),
        "items_after_cleaning": len(cleaned_data),
        "items_removed": len(raw_data) - len(cleaned_data),
        "duration_seconds": (clean_end_time - clean_start_time).total_seconds()
    }

    if not cleaned_data:
        print("No usable comments found after cleaning. Exiting pipeline.")
        with open(output, "w") as f:
            json.dump([], f, indent=2)
        
        if enable_logging:
            write_log_entry(log_data)
        return

    if save_intermediate_files:
        with open(CLEANED_DATA_PATH, "w") as f:
            json.dump(cleaned_data, f, indent=2)
        print(f"Cleaned data saved to {CLEANED_DATA_PATH}. Proceeding with {len(cleaned_data)} comments.")

    if sample_size_for_umap > 0 and sample_size_for_umap < len(raw_data):
        sampled_data = random.sample(cleaned_data, sample_size_for_umap)
        print(f"Sampled {sample_size_for_umap} comments for processing.")
    else:
        sampled_data = cleaned_data
        print(f"No sampling applied. Using all {len(cleaned_data)} comments.")

    if save_intermediate_files:
        with open(SAMPLED_DATA_PATH, "w") as f:
            json.dump(sampled_data, f, indent=2)

    # Step 3: Score
    print(f"Step 3/{total_steps}: Scoring comments using {scoring_prompts} with {scorer_model} ({scorer_backend})...")
    scoring_start_time = datetime.datetime.now()
    
    scored_data = score_comments_data(
        sampled_data, 
        prompts_path=scoring_prompts, 
        model=scorer_model, 
        backend=scorer_backend, 
        include_justifications=include_justifications
    )

    scoring_end_time = datetime.datetime.now()
    scoring_duration = scoring_end_time - scoring_start_time
    
    log_data["scoring"] = {
        "items_scored": len(scored_data),
        "duration_seconds": scoring_duration.total_seconds(),
        "start_time": scoring_start_time.strftime('%Y-%m-%d %H:%M:%S'),
        "end_time": scoring_end_time.strftime('%Y-%m-%d %H:%M:%S'),
        "comments_per_second": len(scored_data) / scoring_duration.total_seconds() if scoring_duration.total_seconds() > 0 else 0
    }
    
    print(f"Scoring completed at: {scoring_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Scoring duration: {scoring_duration}")
    print(f"Comments scored: {len(scored_data)}")
    
    if scoring_duration.total_seconds() > 0:
        comments_per_second = len(scored_data) / scoring_duration.total_seconds()
        print(f"Scoring rate: {comments_per_second:.2f} comments/second")
    
    if save_intermediate_files:
        with open(SCORED_DATA_PATH, "w") as f:
            json.dump(scored_data, f, indent=2)

    # Export scores to CSV just for analysis
    export_scores_to_csv(scored_data, csv_path=output.replace('.json', '.csv'), include_justifications=include_justifications)

    # Step 4: Cluster
    print(f"Step 4/{total_steps}: Performing clustering on {len(sampled_data)} scored comments (clusters: {n_clusters})...")
    cluster_start_time = datetime.datetime.now()
    clustered = cluster_comments_data(scored_data, n_clusters, start_date, end_date, clustering_method=clustering_method)

    cluster_end_time = datetime.datetime.now()
        
    log_data["clustering"] = {
        "items_clustered": len(clustered),
        "n_clusters": n_clusters,
        "duration_seconds": (cluster_end_time - cluster_start_time).total_seconds(),
        "clustering_method": clustering_method
    }

    with open(output, "w") as f:
        json.dump(clustered, f, indent=2)

    print("Clustering complete. Clustered data saved to " + output)

    # Step 5: Add to Vector Database (optional)
    vector_db_items_added = 0
    if vector_db_path:
        print(f"Step 5/{total_steps}: Add to Vector Database {len(clustered)} post/comment pairs with scores, and cluster id of clusters: {n_clusters})...")
 
        try:
            vector_db_start_time = datetime.datetime.now()
            vector_db_items_added = add_to_vector_db(
                output, 
                vector_db_path, 
                embed_model,
                embed_backend,
                embed_api_key
            )
            vector_db_end_time = datetime.datetime.now()
            
            log_data["vector_db"] = {
                "items_added": vector_db_items_added,
                "vector_db_path": vector_db_path,
                "duration_seconds": (vector_db_end_time - vector_db_start_time).total_seconds()
            }
            
        except Exception as e:
            print(f"Warning: Failed to add data to vector database: {e}")
            print("Pipeline completed successfully, but vector database step failed.")
            log_data["vector_db"] = {
                "error": str(e),
                "items_added": 0
            }
    else:
        print("Vector database path not provided, skipping vector database step.")
        log_data["vector_db"] = {
            "skipped": True
        }
    
    # Final summary and logging
    pipeline_end_time = datetime.datetime.now()
    total_duration = pipeline_end_time - pipeline_start_time
    
    log_data["pipeline_summary"] = {
        "total_duration_seconds": total_duration.total_seconds(),
        "pipeline_start": pipeline_start_time.strftime('%Y-%m-%d %H:%M:%S'),
        "pipeline_end": pipeline_end_time.strftime('%Y-%m-%d %H:%M:%S'),
        "final_output_items": len(clustered),
        "final_output_path": output
    }
    
    # Final summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"Scoring model: {scorer_model}")
    print(f"Scoring backend: {scorer_backend}")
    print(f"Embedding model: {embed_model}")
    print(f"Embedding backend: {embed_backend or 'auto-detect'}")
    print(f"Scoring start time: {log_data['scoring']['start_time']}")
    print(f"Scoring end time: {log_data['scoring']['end_time']}")
    print(f"Scoring duration: {datetime.timedelta(seconds=log_data['scoring']['duration_seconds'])}")
    if log_data['scoring']['duration_seconds'] > 0:
        print(f"Scoring rate: {log_data['scoring']['comments_per_second']:.2f} comments/second")
    print(f"Comments processed: {log_data['scoring']['items_scored']}")
    print(f"Total pipeline duration: {datetime.timedelta(seconds=total_duration.total_seconds())}")
    print("="*60)
    
    # Write log if enabled
    if enable_logging:
        write_log_entry(log_data)


def main():
    parser = argparse.ArgumentParser(description="Reddit Comment Knowledge Base Builder CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Scrape Reddit comments")
    scrape_parser.add_argument("--subreddits", type=str, nargs='+', required=True, help="List of subreddits to scrape from")
    scrape_parser.add_argument("--query", type=str, required=True, help="Search query")
    scrape_parser.add_argument("--limit", type=int, default=50, help="Max posts to scrape per subreddit")
    scrape_parser.add_argument("--output", type=str, default="data/raw_data.json", help="Path to output raw JSON data")
    scrape_parser.set_defaults(func=lambda args: RedditScraper().scrape_multiple(
        args.subreddits, format_search_query(args.query), args.limit,
        start_date=None,
        end_date=None
    ))

    # Score command
    score_parser = subparsers.add_parser("score", help="Score comments using LLM prompts")
    score_parser.add_argument("--input", type=str, default="data/raw_data.json", help="Path to input raw JSON data")
    score_parser.add_argument("--output", type=str, default="data/scored_data.json", help="Path to output scored JSON data")
    score_parser.add_argument("--prompts", type=str, default="prompts/prompts.json", help="Path to JSON file with LLM scoring prompts")
    score_parser.set_defaults(func=lambda args: score_comments(args.input, args.output, args.prompts))

    # Cluster command
    cluster_parser = subparsers.add_parser("cluster", help="Perform UMAP and KMeans clustering on scored comments")
    cluster_parser.add_argument("--input", type=str, default="data/scored_data.json", help="Path to input scored JSON data")
    cluster_parser.add_argument("--output", type=str, default="data/clustered_data.json", help="Path to output clustered JSON data")
    cluster_parser.add_argument("--n_clusters", type=int, default=5, help="Number of clusters for KMeans")
    cluster_parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD) for filtering comments")
    cluster_parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD) for filtering comments")
    cluster_parser.add_argument(
        "--clustering_method",
        type=str,
        default="gower_kmedoids",
        help="Clustering method used gower_kmedoids, hdbscan, umap_kmeans (default: 'gower_kmedoids')"
    )
    cluster_parser.set_defaults(func=lambda args: cluster_comments(
        args.input, args.output, args.n_clusters, args.start_date, args.end_date, args.clustering_method
    ))

    # JSON to Vector Database command
    json_to_db_parser = subparsers.add_parser("json-to-db", help="Add clustered JSON data to vector database")
    json_to_db_parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the clustered JSON data file to add to vector database"
    )
    json_to_db_parser.add_argument(
        "--vector_db_path", 
        type=str, 
        required=True, 
        help="Path to the vector database directory (ChromaDB persistence directory)"
    )
    json_to_db_parser.add_argument(
        "--embed_model", 
        type=str, 
        default="nemotron:latest", 
        help="Embedding model to use for the vector database (default: 'nemotron:latest')"
    )
    json_to_db_parser.add_argument(
        "--embed_backend",
        type=str,
        default=None,
        help="Embedding backend: openai, gemini, ollama (auto-detect if not specified)"
    )
    json_to_db_parser.add_argument(
        "--embed_api_key",
        type=str,
        default=None,
        help="API key for embedding backend (or set via environment variable)"
    )
    json_to_db_parser.set_defaults(func=lambda args: add_to_vector_db(
        args.input, 
        args.vector_db_path, 
        args.embed_model,
        args.embed_backend,
        args.embed_api_key
    ))

    # Rescore Database command
    rescore_parser = subparsers.add_parser(
        "rescore", 
        help="Re-score existing data with updated prompts without re-scraping Reddit"
    )
    rescore_parser.add_argument(
        "--input",
        type=str,
        default="vector_db",
        help="Source of data: 'vector_db' to read from ChromaDB, or path to JSON file"
    )
    rescore_parser.add_argument(
        "--vector_db_path",
        type=str,
        required=True,
        help="Path to the vector database directory"
    )
    rescore_parser.add_argument(
        "--scoring_prompts",
        type=str,
        required=True,
        help="Path to new/updated prompts JSON file with scoring rubrics"
    )
    rescore_parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional: Save re-scored data to JSON file before updating database"
    )
    rescore_parser.add_argument(
        "--scorer_model",
        type=str,
        default="llama3.1:8B",
        help="LLM model for re-scoring (e.g., 'llama3.1:8B', 'gemini-2.5-flash', 'gpt-4')"
    )
    rescore_parser.add_argument(
        "--scorer_backend",
        type=str,
        default="ollama",
        help="Backend to use for scoring LLM (e.g., 'ollama', 'gemini', 'openai', 'anthropic')"
    )
    rescore_parser.add_argument(
        "--include_justifications",
        action="store_true",
        default=False,
        help="Include justifications in the output"
    )
    rescore_parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Optional: Limit number of items to rescore (useful for testing)"
    )
    rescore_parser.add_argument(
        "--filter_cluster_id",
        type=int,
        default=None,
        help="Optional: Only rescore items from a specific cluster"
    )
    rescore_parser.add_argument(
        "--embed_model",
        type=str,
        default="nomic-embed-text:latest",
        help="Embedding model for vector database (default: 'nomic-embed-text:latest')"
    )
    rescore_parser.add_argument(
        "--embed_backend",
        type=str,
        default=None,
        help="Embedding backend: openai, gemini, ollama (auto-detect if not specified)"
    )
    rescore_parser.add_argument(
        "--embed_api_key",
        type=str,
        default=None,
        help="API key for embedding backend (or set via environment variable)"
    )
    rescore_parser.set_defaults(
        func=lambda args: rescore_database(
            input_source=args.input,
            vector_db_path=args.vector_db_path,
            scoring_prompts=args.scoring_prompts,
            output_json=args.output_json,
            scorer_model=args.scorer_model,
            scorer_backend=args.scorer_backend,
            include_justifications=args.include_justifications,
            sample_size=args.sample_size,
            filter_cluster_id=args.filter_cluster_id,
            embed_model=args.embed_model,
            embed_backend=args.embed_backend,
            embed_api_key=args.embed_api_key
        )
    )

    # Process All (pipeline) command
    process_all_parser = subparsers.add_parser("process-all", help="Run the full pipeline: scrape, score, cluster, and optionally add to vector database")
    process_all_parser.add_argument(
        "--subreddits",
        type=str,
        nargs='+',
        required=True,
        help="List of subreddits to scrape from (e.g., 'politics worldnews' or 'politics,worldnews,science')"
    )
    process_all_parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query - will be auto-formatted for Reddit search (e.g., 'AI regulation OR AI governance OR AI policy')"
    )
    process_all_parser.add_argument(
        "--scrape_limit",
        type=int,
        default=50,
        help="Max posts to scrape per subreddit (total comments may exceed this)"
    )
    process_all_parser.add_argument(
        "--scoring_prompts",
        type=str,
        default="prompts/prompts.json",
        help="Path to JSON file with LLM scoring prompts (for authenticity, ideology, etc.)"
    )
    process_all_parser.add_argument(
        "--n_clusters",
        type=int,
        default=5,
        help="Number of clusters for KMeans"
    )
    process_all_parser.add_argument(
        "--start_date",
        type=str,
        default="2000-01-01",
        help="Start date (YYYY-MM-DD) for filtering comments for clustering"
    )
    process_all_parser.add_argument(
        "--end_date",
        type=str,
        default="2030-01-01",
        help="End date (YYYY-MM-DD) for filtering comments for clustering"
    )
    process_all_parser.add_argument(
        "--output",
        type=str,
        default="data/clustered_data_pipeline_output.json",
        help="Path to output the final clustered data JSON file"
    )
    process_all_parser.add_argument(
        "--scorer_model",
        type=str,
        default="llama3.1:8B",
        help="LLM model for scoring comments (e.g., 'llama3.1:8B', 'gemini-2.5-flash', 'gpt-4')"
    )
    process_all_parser.add_argument(
        "--scorer_backend",
        type=str,
        default="ollama",
        help="Backend to use for scoring LLM (e.g., 'ollama', 'gemini', 'openai', 'anthropic')"
    )
    process_all_parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="sample size for scoring and clustering (default: 1000, or -1 for no sampling)"
    )
    process_all_parser.add_argument(
        "--include_justifications",
        action="store_true",
        default=False,
        help="Whether to include justifications in the output"
    )
    process_all_parser.add_argument(
        "--save_intermediate_files",
        action="store_true",
        default=False,
        help="Save files with intermediate steps (raw, cleaned, sampled, scored)"
    )
    process_all_parser.add_argument(
        "--no-auto-format",
        action="store_true",
        help="Disable automatic query formatting (use raw query as provided)"
    )
    process_all_parser.add_argument(
        "--vector_db_path",
        type=str,
        default=None,
        help="Optional: Path to vector database directory. If provided, clustered data will be added to the vector database as a final step"
    )
    process_all_parser.add_argument(
        "--embed_model",
        type=str,
        default="nomic-embed-text:latest",
        help="Embedding model to use for vector database (default: 'nomic-embed-text:latest')"
    )
    process_all_parser.add_argument(
        "--embed_backend",
        type=str,
        default=None,
        help="Embedding backend: openai, gemini, ollama (auto-detect if not specified)"
    )
    process_all_parser.add_argument(
        "--embed_api_key",
        type=str,
        default=None,
        help="API key for embedding backend (or set via environment variable)"
    )
    process_all_parser.add_argument(
        "--clustering_method",
        type=str,
        default="gower_kmedoids",
        help="Clustering method used gower_kmedoids, hdbscan, umap_kmeans (default: 'gower_kmedoids')"
    )
    process_all_parser.add_argument(
        "--bin_by_period",
        type=str,
        default=None,
        choices=["day", "week", "month"],
        help="Period to bin comments by 'day', 'week', 'month' (default: None, no binning)"
    )
    process_all_parser.add_argument(
        "--log",
        action="store_true",
        help="Enable logging to file with detailed execution metrics"
    )
    process_all_parser.set_defaults(
        func=lambda args: process_all_comments_pipeline(
            subreddits=args.subreddits,
            query=args.query,
            scrape_limit=args.scrape_limit,
            scoring_prompts=args.scoring_prompts,
            n_clusters=args.n_clusters,
            start_date=args.start_date,
            end_date=args.end_date,
            bin_by_period=args.bin_by_period,
            output=args.output,
            auto_format_query=not args.no_auto_format,
            scorer_model=args.scorer_model,
            scorer_backend=args.scorer_backend,
            sample_size_for_umap=args.sample_size,
            include_justifications=args.include_justifications,
            vector_db_path=args.vector_db_path,
            embed_model=args.embed_model,
            embed_backend=args.embed_backend,
            embed_api_key=args.embed_api_key,
            save_intermediate_files=args.save_intermediate_files,
            clustering_method=args.clustering_method,
            enable_logging=args.log
        )
    )

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()