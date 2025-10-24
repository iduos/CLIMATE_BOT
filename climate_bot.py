# bot_cat.py
#
# By Ian Drumm, The University of Salford, UK.
#
import streamlit as st
import argparse
import sys
import json
import re
import plotly.express as px
import pandas as pd
import os
from src.vector_db_manager import VectorDBManager
from src.rag_chatbot import RAGCommentSimulator

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file from current directory
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"Could not load .env file: {e}")

# Parse command line arguments
if len(sys.argv) > 1:
    if '--' in sys.argv:
        separator_index = sys.argv.index('--')
        script_args = sys.argv[separator_index + 1:]
    else:
        script_args = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Social Media RAG Comment Simulator")
    parser.add_argument("--vector_db_path", default="vdbs/augdb", help="Path to the Chroma database")
    parser.add_argument("--embed_model", default="nemotron:latest", help="Embedding model to use")
    parser.add_argument("--embed_backend", default=None, help="Embedding backend: openai, gemini, ollama (auto-detect if not specified)")
    parser.add_argument("--embed_api_key", default=None, help="API key for embedding backend (or set via environment variable)")
    parser.add_argument("--chat_model", default="llama3.1:8b", help="Chat model to use")
    parser.add_argument("--chat_backend", default=None, help="Chat backend: openai, gemini, anthropic, ollama (auto-detect if not specified)")
    parser.add_argument("--chat_api_key", default=None, help="API key for chat backend (or set via environment variable)")
    
    args = parser.parse_args(script_args)
else:
    class Args:
        def __init__(self):
            self.vector_db_path = "../OS_BOT/vdbs/vdb1"
            self.embed_model = "nemotron:latest"
            self.embed_backend = None
            self.embed_api_key = None
            self.chat_model = "llama3.1:8b"
            self.chat_backend = None
            self.chat_api_key = None
    args = Args()

# Page config for better styling
st.set_page_config(page_title="Social Media Comment Simulator", page_icon="💬", layout="wide")

# Custom CSS for colorful UI
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .filter-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
    }
    .post-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
    }
    .reply-section {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        color: white;
    }
    .stButton>button {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #38ef7d 0%, #11998e 100%);
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>💬 Social Media Comment Simulator</h1><p>Generate authentic social media comments based on user archetypes</p></div>', unsafe_allow_html=True)

db_manager = VectorDBManager(
    persist_directory=args.vector_db_path, 
    embed_model=args.embed_model,
    embed_backend=args.embed_backend,
    embed_api_key=args.embed_api_key
)
db_name = args.vector_db_path.split("/")[-1]
record_count = db_manager.count_documents()

with st.expander("📊 Database and Model Information", expanded=False):
    st.subheader(f"Database: {db_name}")
    st.write(f"Total records: {record_count}")
    
    st.subheader("Models")
    st.write(f"**Embedding Model:** {args.embed_model}")
    if args.embed_backend:
        st.write(f"**Embedding Backend:** {args.embed_backend}")
    else:
        st.write(f"**Embedding Backend:** Auto-detect")
    
    st.write(f"**Chat Model:** {args.chat_model}")
    
    # Display backend information
    backend_info = args.chat_backend if args.chat_backend else "Auto-detect"
    st.write(f"**Chat Backend:** {backend_info}")
    
    # Show API key status (without revealing the key)
    if args.embed_api_key:
        st.write(f"**Embedding API Key:** Provided via command line")
    else:
        # Check environment variables for embedding
        embed_env_keys = {
            "openai": "OPENAI_API_KEY",
            "gemini": "GOOGLE_API_KEY"
        }
        
        if args.embed_backend and args.embed_backend.lower() in embed_env_keys:
            env_var = embed_env_keys[args.embed_backend.lower()]
            if os.environ.get(env_var):
                st.write(f"**Embedding API Key:** Using {env_var} from environment")
            else:
                st.warning(f"**Embedding API Key:** {env_var} not found in environment")
        elif args.embed_backend and args.embed_backend.lower() == "ollama":
            st.write("**Embedding API Key:** Not required (local Ollama)")
        else:
            st.write("**Embedding API Key:** Will check environment variables based on detected backend")
    
    if args.chat_api_key:
        st.write(f"**Chat API Key:** Provided via command line")
    else:
        # Check environment variables for chat
        chat_env_keys = {
            "openai": "OPENAI_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY"
        }
        
        if args.chat_backend and args.chat_backend.lower() in chat_env_keys:
            env_var = chat_env_keys[args.chat_backend.lower()]
            if os.environ.get(env_var):
                st.write(f"**Chat API Key:** Using {env_var} from environment")
            else:
                st.warning(f"**Chat API Key:** {env_var} not found in environment")
        elif args.chat_backend and args.chat_backend.lower() == "ollama":
            st.write("**Chat API Key:** Not required (local Ollama)")
        else:
            st.write("**Chat API Key:** Will check environment variables based on detected backend")
    
    # Category Distribution Chart
    if st.checkbox("Show category distribution"):
        category_distribution = db_manager.get_category_distribution()
        
        if category_distribution:
            # Prepare data for plotting
            chart_data = []
            for category_type, value_counts in category_distribution.items():
                for value, count in value_counts.items():
                    chart_data.append({
                        'Category': category_type.replace('_', ' ').title(),
                        'Value': value,
                        'Count': count,
                        'Category_Value': f"{category_type}: {value}"
                    })
            
            if chart_data:
                df = pd.DataFrame(chart_data)
                
                # Create a horizontal bar chart with different colors for each category
                fig = px.bar(
                    df, 
                    x='Count', 
                    y='Category_Value',
                    color='Category',
                    title='Distribution of Category Values in Database',
                    labels={'Category_Value': 'Category: Value', 'Count': 'Number of Documents'},
                    height=max(400, len(chart_data) * 25),  # Dynamic height based on number of bars
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                
                # Improve layout
                fig.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show summary statistics
                st.subheader("Distribution Summary")
                for category_type, value_counts in category_distribution.items():
                    display_name = category_type.replace('_', ' ').title()
                    total_docs = sum(value_counts.values())
                    st.write(f"**{display_name}**: {len(value_counts)} unique values, {total_docs} total documents")
            else:
                st.info("No category distribution data available.")
        else:
            st.info("No categories found in the database.")
    
    if st.checkbox("Show raw metadata preview"):
        results = db_manager.db.get(include=["metadatas"])
        metadatas = results.get("metadatas", []) or []
        st.write(f"Total metadatas: {len(metadatas)}")
        for i, m in enumerate(metadatas[:5]):
            st.markdown(f"**Metadata #{i} types**")
            st.json({k: type(v).__name__ for k, v in m.items()})
            st.markdown("**Raw**")
            st.json(m)
            
    # Export functionality
    st.subheader("Export Database Contents")
    
    @st.cache_data
    def export_database_to_csv():
        """Export all database contents to CSV format, sorted by query and cluster ID."""
        try:
            # Get all documents with metadata and content
            results = db_manager.db.get(include=["metadatas", "documents"])
            
            if not results or not results.get("ids"):
                return None, "No data found in database"
            
            # Prepare data for CSV export
            export_data = []
            
            for i, doc_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                document = results["documents"][i]
                
                def parse_post_and_comment(document_text):
                    """
                    Parse the document text to extract post and comment separately.
                    Expected format: "Post: {post_content}\nComment: {comment_content}"
                    """
                    try:
                        if not document_text or not isinstance(document_text, str):
                            return "", ""
                        
                        # Split by newline and look for "Post:" and "Comment:" prefixes
                        lines = document_text.split('\n')
                        post_content = ""
                        comment_content = ""
                        
                        current_section = None
                        for line in lines:
                            if line.startswith("Post: "):
                                current_section = "post"
                                post_content = line[6:]  # Remove "Post: " prefix
                            elif line.startswith("Comment: "):
                                current_section = "comment"
                                comment_content = line[9:]  # Remove "Comment: " prefix
                            else:
                                # Continue adding to the current section
                                if current_section == "post":
                                    post_content += "\n" + line if post_content else line
                                elif current_section == "comment":
                                    comment_content += "\n" + line if comment_content else line
                        
                        return post_content.strip(), comment_content.strip()
                    
                    except Exception as e:
                        print(f"Error parsing document: {e}")
                        return "", ""

                # Parse post and comment from the document field
                post_content, comment_content = parse_post_and_comment(document)

                # Extract basic metadata
                row = {
                    "id": doc_id,
                    "document": document,  # Keep original combined document
                    "post": post_content,  # New: separated post content
                    "comment": comment_content,  # New: separated comment content
                    "search_query": metadata.get("search_query", ""),
                    "cluster_id": metadata.get("cluster_id", ""),
                    "subreddit": metadata.get("subreddit", ""),
                    "created_utc": metadata.get("created_utc", ""),
                    "prompts_path": metadata.get("prompts_path", ""),
                    "umap_x": metadata.get("umap_x", ""),
                    "umap_y": metadata.get("umap_y", "")
                }

                # Extract scores from scores_json
                scores_json = metadata.get("scores_json", "{}")
                try:
                    scores = json.loads(scores_json)
                    for score_name, score_value in scores.items():
                        row[f"score_{score_name}"] = score_value
                except json.JSONDecodeError:
                    pass
                
                # Extract categories from categories_json
                categories_json = metadata.get("categories_json", "[]")
                category_names = []
                try:
                    categories = json.loads(categories_json)
                    if isinstance(categories, list):
                        # Handle list format: [{"political_ideology": "Far Right"}, ...]
                        for cat_dict in categories:
                            if isinstance(cat_dict, dict):
                                for cat_name, cat_value in cat_dict.items():
                                    row[f"category_{cat_name}"] = cat_value
                                    category_names.append(cat_name)
                    elif isinstance(categories, dict):
                        # Handle dict format: {"political_ideology": "Far Right", ...}
                        for cat_name, cat_value in categories.items():
                            row[f"category_{cat_name}"] = cat_value
                            category_names.append(cat_name)
                except json.JSONDecodeError:
                    pass
                
                # Extract justifications from justifications_json
                # Match each justification to its corresponding category
                justifications_json = metadata.get("justifications_json", "[]")
                try:
                    justifications = json.loads(justifications_json)
                    if isinstance(justifications, list):
                        # Match justifications with categories by index
                        for idx, justification in enumerate(justifications):
                            if idx < len(category_names):
                                cat_name = category_names[idx]
                                row[f"justification_{cat_name}"] = str(justification)
                            else:
                                # If there are more justifications than categories
                                row[f"justification_{idx}"] = str(justification)
                    elif isinstance(justifications, dict):
                        # If justifications is already a dict with category names as keys
                        for cat_name, justification in justifications.items():
                            row[f"justification_{cat_name}"] = str(justification)
                except json.JSONDecodeError:
                    pass
                
                export_data.append(row)
            
            # Convert to DataFrame and sort
            df = pd.DataFrame(export_data)
            
            # Sort by search_query first, then by cluster_id
            # Handle potential None or non-numeric cluster_id values
            def safe_cluster_sort(x):
                try:
                    return (0, int(x)) if x != "" and x is not None else (1, str(x))
                except (ValueError, TypeError):
                    return (1, str(x))
            
            df["cluster_sort_key"] = df["cluster_id"].apply(safe_cluster_sort)
            df = df.sort_values(["search_query", "cluster_sort_key"])
            df = df.drop("cluster_sort_key", axis=1)
            
            # Convert DataFrame to CSV
            csv_data = df.to_csv(index=False)
            
            return csv_data, None
            
        except Exception as e:
            return None, f"Error exporting data: {str(e)}"
    
    # Export button and download
    if st.button("Export to CSV", help="Export all database contents sorted by search query and cluster ID"):
        with st.spinner("Exporting database contents..."):
            csv_data, error = export_database_to_csv()
            
            if error:
                st.error(error)
            elif csv_data:
                # Generate filename with timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{db_name}_export_{timestamp}.csv"
                
                # Provide download button
                st.download_button(
                    label="Download CSV File",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    help=f"Download the exported database contents as {filename}"
                )
            else:
                st.warning("No data available for export")

db_unique_categories = db_manager.get_unique_categories()

# Colorful filter section
st.markdown("---")
st.markdown("## 🎯 Select User Archetype")

# Create category filters from database with colored containers
selected_categories_dict = {}

# Use columns for better layout
num_categories = len(db_unique_categories)
if num_categories > 0:
    cols = st.columns(min(3, num_categories))
    
    for idx, (category_type, category_options) in enumerate(db_unique_categories.items()):
        with cols[idx % 3]:
            # Convert category_type back to display format
            display_name = category_type.replace('_', ' ').title()
            
            # Create a safe key for the selectbox
            safe_key = f"selectbox_{category_type}"
            
            selected_option = st.selectbox(
                f"🏷️ {display_name}",
                ["All"] + sorted(category_options),
                key=safe_key
            )
            
            # Add to dictionary if not "All"
            if selected_option != "All":
                selected_categories_dict[category_type] = selected_option

st.markdown("---")

# Colorful post input section
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("## 📝 Create Your Post")
    post_content = st.text_area(
        "Enter a social media post",
        height=150,
        placeholder="What's on your mind? Type your post here...",
        key="textarea_post",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("## ")
    st.markdown("## ")
    generate_button = st.button("✨ Generate Comment", use_container_width=True)

if 'generated_reply' not in st.session_state:
    st.session_state.generated_reply = None
if 'source_documents' not in st.session_state:
    st.session_state.source_documents = []
if 'full_prompt' not in st.session_state:
    st.session_state.full_prompt = None

if generate_button and post_content:
    with st.spinner("🤖 Generating comment..."):
        try:
            # Initialize RAG simulator with backend support
            rag_simulator = RAGCommentSimulator(
                persist_directory=args.vector_db_path,
                embed_model=args.embed_model,
                embed_backend=args.embed_backend,
                embed_api_key=args.embed_api_key,
                chat_model=args.chat_model,
                backend=args.chat_backend,
                chat_api_key=args.chat_api_key
            )
            
            # No search query filter - always use "All"
            filter_search_query = None

            # Generate reply with the combined filter
            result = rag_simulator.generate_reply(
                post_content,
                search_query=filter_search_query,
                categories=selected_categories_dict
            )
            
            st.session_state.generated_reply = result["result"]
            st.session_state.source_documents = result["source_documents"]
            st.session_state.full_prompt = result.get("full_prompt", None)
            
        except ValueError as e:
            # Handle API key errors gracefully
            st.error(f"⚠️ Configuration Error: {str(e)}")
            st.info("Please ensure you have set the appropriate API key via command line argument or environment variable.")
        except Exception as e:
            st.error(f"❌ Error generating comment: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

if st.session_state.generated_reply:
    st.markdown("---")
    st.markdown("## 💬 Generated Comment")
    
    # Display reply in a colorful container
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; 
                    border-radius: 15px; 
                    color: white; 
                    font-size: 1.1rem;
                    line-height: 1.6;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            {st.session_state.generated_reply}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if st.session_state.source_documents:
        with st.expander("🔍 Source Documents & Prompt Details", expanded=False):
            # Display the full prompt sent to LLM if available
            if st.session_state.full_prompt:
                st.markdown("### 📝 Full Prompt Sent to LLM")
                st.text_area(
                    "Prompt",
                    value=st.session_state.full_prompt,
                    height=400,
                    key="full_prompt_display",
                    disabled=True
                )
                st.markdown("---")
            else:
                st.warning("⚠️ Full prompt not available.")
            
            # Display source documents
            st.markdown("### 📚 Source Documents Used")
            for i, doc in enumerate(st.session_state.source_documents):
                with st.container():
                    st.markdown(f"#### 📄 Document {i+1}")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"**🔎 Search Query:** {doc.metadata.get('search_query', 'N/A')}")
                        
                        categories_json_str = doc.metadata.get('categories_json', '{}')
                        try:
                            categories_json = json.loads(categories_json_str)
                            if isinstance(categories_json, list):
                                # Handle list format
                                for cat_dict in categories_json:
                                    if isinstance(cat_dict, dict):
                                        for key, value in cat_dict.items():
                                            display_key = key.replace('_', ' ').title()
                                            st.markdown(f"**🏷️ {display_key}:** {value}")
                            elif isinstance(categories_json, dict):
                                # Handle dict format
                                for key, value in categories_json.items():
                                    display_key = key.replace('_', ' ').title()
                                    st.markdown(f"**🏷️ {display_key}:** {value}")
                        except json.JSONDecodeError:
                            st.markdown(f"**Categories:** N/A (invalid JSON)")

                        st.markdown(f"**⭐ Scores:** {doc.metadata.get('scores', 'N/A')}")
                    
                    with col2:
                        st.markdown(f"**📄 Content:**")
                        st.info(doc.page_content)
                    
                    st.markdown("---")
    else:
        st.info("ℹ️ No source documents were used for this reply.")