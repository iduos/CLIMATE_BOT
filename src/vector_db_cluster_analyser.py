# Enhanced Pattern-Aware LLM Cluster Analyzer
# This replaces/enhances the existing vector_db_cluster_analyser.py approach
# Enhanced Pattern-Aware LLM Cluster Analyzer
# This replaces/enhances the existing vector_db_cluster_analyser.py approach

import json
import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler

class EnhancedVectorClusterAnalyzer:
    """
    Enhanced cluster analyzer that understands Gower distance clustering patterns
    rather than relying solely on simple statistical means.
    """
    
    def __init__(self, persist_directory: str, 
                 embed_model: str = "mxbai-embed-large", 
                 chat_model: str = "llama3.1:8b",
                 embed_backend: Optional[str] = None,
                 embed_api_key: Optional[str] = None,
                 chat_backend: Optional[str] = None,
                 chat_api_key: Optional[str] = None):
        """
        Initialize the enhanced analyzer
        
        Args:
            persist_directory: Path to vector database
            embed_model: Embedding model name
            chat_model: Chat model name
            embed_backend: Embedding backend (openai, gemini, ollama, or None for auto-detect)
            embed_api_key: API key for embedding backend
            chat_backend: Chat backend (openai, gemini, anthropic, ollama, or None for auto-detect)
            chat_api_key: API key for chat backend
        """
        from src.vector_db_manager import VectorDBManager
        from src.rag_chatbot import RAGCommentSimulator
        
        self.db_manager = VectorDBManager(
            persist_directory=persist_directory, 
            embed_model=embed_model,
            embed_backend=embed_backend,
            embed_api_key=embed_api_key
        )
        
        self.rag_chatbot = RAGCommentSimulator(
            embed_model=embed_model,
            chat_model=chat_model,
            persist_directory=persist_directory,
            embed_backend=embed_backend,
            embed_api_key=embed_api_key,
            backend=chat_backend,
            chat_api_key=chat_api_key
        )
        
        print(f"Initialized EnhancedVectorClusterAnalyzer:")
        print(f"  - Embedding model: {embed_model} (backend: {embed_backend or 'auto-detect'})")
        print(f"  - Chat model: {chat_model} (backend: {chat_backend or 'auto-detect'})")
        
        # Enhanced prompt template that focuses on patterns and requests structured output
        self.analysis_prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert data scientist specializing in pattern-based cluster analysis. "
                      "Your task is to interpret clustering results by identifying meaningful patterns "
                      "rather than just statistical averages. Focus on behavioral patterns, extreme values, "
                      "consistency patterns, and multi-dimensional relationships."),
            ("user", "Analyze the following cluster pattern analysis and provide insights:\n\n{analysis_context}\n\n"
                     "Please provide your analysis structured into the following sections:\n"
                     "1. **Executive Summary**: A brief, high-level summary of the overall clustering result and what the key finding is.\n"
                     "2. **Cluster Archetypes**: For each cluster, describe the coherent user archetype or comment style it represents. Synthesize the detected patterns and text analysis to paint a clear picture. Discuss the relative size of each archetype.\n"
                     "3. **Interpretation of UMAP Dimensions**: Based on the provided feature correlations, explain what the UMAP X and Y axes represent. For example, 'The X-axis appears to represent a spectrum from X to Y'.\n"
                     "4. **Driving Features**: Identify the features that were most instrumental in creating the distinctions between the clusters. Explain why they were important.\n\n"
                     "Note that cluster '-1' represents noise/outliers and should be mentioned but not characterized as a main archetype.")
        ])
        
        self.data = None
        self.df = None
    
    def load_data_from_vectordb(self, search_query_filter: Optional[str] = None) -> List[Dict]:
        """Load data from vector database with optional filtering"""
        try:
            results = self.db_manager.db.get(include=["documents", "metadatas"])
            documents = [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(results['documents'], results['metadatas'])
            ]
            
            if search_query_filter:
                documents = [
                    doc for doc in documents 
                    if doc.metadata.get('search_query') == search_query_filter
                ]
                print(f"Filtered to {len(documents)} documents with search_query='{search_query_filter}'")
            
            converted_data = []
            for doc in documents:
                try:
                    content = doc.page_content
                    if "Post:" in content and "Comment:" in content:
                        parts = content.split("Comment:", 1)
                        post_content = parts[0].replace("Post:", "").strip()
                        comment_content = parts[1].strip()
                    else:
                        post_content = doc.metadata.get('post', '')
                        comment_content = content
                    
                    scores = {}
                    if 'scores_json' in doc.metadata:
                        try:
                            scores = json.loads(doc.metadata['scores_json'])
                        except (json.JSONDecodeError, TypeError):
                            scores = {}
                    
                    data_item = {
                        'post': post_content,
                        'comment': comment_content,
                        'scores': scores,
                        'cluster_id': doc.metadata.get('cluster_id'),
                        'umap_x': doc.metadata.get('umap_x'),
                        'umap_y': doc.metadata.get('umap_y'),
                        'comment_metadata': {
                            'id': doc.metadata.get('comment_id'),
                            'created_utc': doc.metadata.get('created_utc')
                        },
                        'post_metadata': {
                            'subreddit': doc.metadata.get('subreddit'),
                            'title': doc.metadata.get('title'),
                            'search_query': doc.metadata.get('search_query')
                        }
                    }
                    
                    if (data_item['cluster_id'] is not None and 
                        data_item['umap_x'] is not None and 
                        data_item['umap_y'] is not None):
                        converted_data.append(data_item)
                        
                except Exception as e:
                    print(f"Warning: Skipping document due to parsing error: {e}")
                    continue
            
            print(f"Loaded {len(converted_data)} documents from vector database")
            self.data = converted_data
            self.df = pd.DataFrame(converted_data)
            return converted_data
            
        except Exception as e:
            print(f"Error loading data from vector database: {e}")
            return []
    
    def detect_cluster_patterns(self) -> Dict[str, Any]:
        """
        Detect behavioral and statistical patterns that distinguish clusters,
        going beyond simple means to understand clustering logic.
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data_from_vectordb() first.")
        
        # Get all score features
        all_score_keys = set()
        for item in self.data:
            if 'scores' in item and isinstance(item['scores'], dict):
                all_score_keys.update(item['scores'].keys())
        
        all_score_keys = list(all_score_keys)
        
        # Separate noise from real clusters
        real_clusters = [c for c in self.df['cluster_id'].unique() if c != -1]
        noise_count = (self.df['cluster_id'] == -1).sum()
        
        cluster_patterns = {}
        
        for cluster_id in real_clusters:
            cluster_data = self.df[self.df['cluster_id'] == cluster_id]
            
            # Extract scores for this cluster
            cluster_scores = {}
            for score_key in all_score_keys:
                scores = []
                for _, row in cluster_data.iterrows():
                    if isinstance(row['scores'], dict) and score_key in row['scores']:
                        try:
                            scores.append(float(row['scores'][score_key]))
                        except (ValueError, TypeError):
                            continue
                cluster_scores[score_key] = scores
            
            # Pattern detection for this cluster
            patterns = self._detect_feature_patterns(cluster_scores, all_score_keys)
            
            # Add cluster metadata
            cluster_patterns[int(cluster_id)] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(self.df) * 100,
                'patterns': patterns,
                'sample_comments': cluster_data['comment'].head(5).tolist(),
                'text_analysis': self._analyze_cluster_text(cluster_data)
            }
        
        return {
            'cluster_patterns': cluster_patterns,
            'noise_count': int(noise_count),
            'noise_percentage': float(noise_count / len(self.df) * 100),
            'feature_importance': self._calculate_pattern_based_importance(all_score_keys, real_clusters),
            'global_stats': self._calculate_global_feature_stats(all_score_keys),
            'umap_correlations': self._calculate_umap_correlations(all_score_keys)
        }
    
    def _detect_feature_patterns(self, cluster_scores: Dict[str, List], all_score_keys: List[str]) -> Dict[str, Any]:
        """
        Detect various types of patterns in cluster feature distributions
        """
        patterns = {}
        
        # Get global distributions for comparison
        global_scores = {}
        for score_key in all_score_keys:
            global_vals = []
            for item in self.data:
                if isinstance(item['scores'], dict) and score_key in item['scores']:
                    try:
                        global_vals.append(float(item['scores'][score_key]))
                    except (ValueError, TypeError):
                        continue
            global_scores[score_key] = global_vals
        
        for score_key, cluster_vals in cluster_scores.items():
            if not cluster_vals or score_key not in global_scores:
                continue
                
            global_vals = global_scores[score_key]
            if not global_vals:
                continue
            
            cluster_array = np.array(cluster_vals)
            global_array = np.array(global_vals)
            
            feature_patterns = []
            
            # 1. Central tendency patterns
            cluster_median = np.median(cluster_array)
            global_median = np.median(global_array)
            global_q25, global_q75 = np.percentile(global_array, [25, 75])
            
            if cluster_median > global_q75:
                feature_patterns.append({
                    'type': 'consistently_high',
                    'strength': (cluster_median - global_median) / (global_q75 - global_median) if (global_q75 - global_median) > 0 else 1,
                    'description': f"consistently high {score_key} scores"
                })
            elif cluster_median < global_q25:
                feature_patterns.append({
                    'type': 'consistently_low',
                    'strength': (global_median - cluster_median) / (global_median - global_q25) if (global_median - global_q25) > 0 else 1,
                    'description': f"consistently low {score_key} scores"
                })
            
            # 2. Consistency patterns (low variance within cluster)
            if len(cluster_vals) > 1:
                cluster_cv = np.std(cluster_array) / abs(np.mean(cluster_array)) if np.mean(cluster_array) != 0 else 0
                global_cv = np.std(global_array) / abs(np.mean(global_array)) if np.mean(global_array) != 0 else 0
                
                if cluster_cv < global_cv * 0.6 and cluster_cv < 0.3:
                    feature_patterns.append({
                        'type': 'highly_consistent',
                        'strength': 1 - (cluster_cv / global_cv) if global_cv > 0 else 1,
                        'description': f"very consistent {score_key} scores (low variation within cluster)"
                    })
            
            # 3. Extreme value patterns
            global_q10, global_q90 = np.percentile(global_array, [10, 90])
            extreme_count = np.sum((cluster_array < global_q10) | (cluster_array > global_q90))
            extreme_percentage = extreme_count / len(cluster_array)
            
            if extreme_percentage > 0.5:
                feature_patterns.append({
                    'type': 'extreme_values',
                    'strength': extreme_percentage,
                    'description': f"contains many extreme {score_key} values ({extreme_percentage:.1%} in top/bottom 10%)"
                })
            
            # 4. Bimodal or unusual distribution patterns
            if len(cluster_vals) > 10:
                # Simple bimodality check using histogram
                hist, _ = np.histogram(cluster_array, bins=5)
                # Look for valleys in the distribution
                if np.min(hist[1:-1]) < np.max(hist) * 0.3:
                    feature_patterns.append({
                        'type': 'bimodal_distribution',
                        'strength': 1 - (np.min(hist[1:-1]) / np.max(hist)),
                        'description': f"shows bimodal distribution in {score_key} (two distinct sub-groups)"
                    })
            
            # 5. Range compression/expansion patterns
            cluster_range = np.max(cluster_array) - np.min(cluster_array)
            global_range = np.max(global_array) - np.min(global_array)
            
            if global_range > 0:
                range_ratio = cluster_range / global_range
                if range_ratio < 0.3:
                    feature_patterns.append({
                        'type': 'compressed_range',
                        'strength': 1 - range_ratio,
                        'description': f"narrow range of {score_key} values (compressed distribution)"
                    })
                elif range_ratio > 1.5:
                    feature_patterns.append({
                        'type': 'expanded_range',
                        'strength': range_ratio - 1,
                        'description': f"wide range of {score_key} values (diverse within cluster)"
                    })
            
            patterns[score_key] = feature_patterns
        
        return patterns
    
    def _analyze_cluster_text(self, cluster_data: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced text analysis focusing on distinctive patterns"""
        comments = cluster_data['comment'].dropna().astype(str).tolist()
        if not comments:
            return {}
        
        # Basic text statistics
        text_analysis = {
            'avg_length': np.mean([len(comment.split()) for comment in comments]),
            'median_length': np.median([len(comment.split()) for comment in comments]),
            'length_variation': np.std([len(comment.split()) for comment in comments]),
        }
        
        # Sentiment and style patterns (simple heuristics)
        question_count = sum(1 for comment in comments if '?' in comment)
        exclamation_count = sum(1 for comment in comments if '!' in comment)
        caps_count = sum(1 for comment in comments if any(word.isupper() and len(word) > 2 for word in comment.split()))
        
        text_analysis.update({
            'question_percentage': question_count / len(comments),
            'exclamation_percentage': exclamation_count / len(comments),
            'caps_percentage': caps_count / len(comments),
        })
        
        # Extract distinctive terms (simple frequency-based)
        all_text = ' '.join(comments).lower()
        words = re.findall(r'\b\w+\b', all_text)
        word_freq = Counter(words)
        
        # Filter out common words and short words
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'a', 'an'}
        
        distinctive_words = [(word, count) for word, count in word_freq.most_common(20) 
                           if len(word) > 2 and word not in common_words]
        
        text_analysis['distinctive_terms'] = distinctive_words[:10]
        
        return text_analysis
    
    def _calculate_pattern_based_importance(self, all_score_keys: List[str], real_clusters: List) -> Dict[str, float]:
        """
        Calculate feature importance based on pattern detection rather than simple variance
        """
        feature_importance = {}
        
        for score_key in all_score_keys:
            importance_score = 0
            pattern_count = 0
            
            # Count how many clusters show strong patterns for this feature
            for cluster_id in real_clusters:
                cluster_data = self.df[self.df['cluster_id'] == cluster_id]
                cluster_vals = []
                
                for _, row in cluster_data.iterrows():
                    if isinstance(row['scores'], dict) and score_key in row['scores']:
                        try:
                            cluster_vals.append(float(row['scores'][score_key]))
                        except (ValueError, TypeError):
                            continue
                
                if len(cluster_vals) > 0:
                    # Check for patterns in this cluster for this feature
                    global_vals = []
                    for item in self.data:
                        if isinstance(item['scores'], dict) and score_key in item['scores']:
                            try:
                                global_vals.append(float(item['scores'][score_key]))
                            except (ValueError, TypeError):
                                continue
                    
                    if global_vals:
                        cluster_median = np.median(cluster_vals)
                        global_q25, global_q75 = np.percentile(global_vals, [25, 75])
                        
                        # Strong pattern detection
                        if cluster_median > global_q75 or cluster_median < global_q25:
                            importance_score += 1
                            pattern_count += 1
                        
                        # Consistency pattern
                        if len(cluster_vals) > 1:
                            cluster_cv = np.std(cluster_vals) / abs(np.mean(cluster_vals)) if np.mean(cluster_vals) != 0 else 0
                            global_cv = np.std(global_vals) / abs(np.mean(global_vals)) if np.mean(global_vals) != 0 else 0
                            
                            if cluster_cv < global_cv * 0.6:
                                importance_score += 0.5
                                pattern_count += 0.5
            
            # Normalize by number of clusters
            if len(real_clusters) > 0:
                feature_importance[score_key] = importance_score / len(real_clusters)
            else:
                feature_importance[score_key] = 0
        
        return feature_importance

    def _calculate_umap_correlations(self, all_score_keys: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate Pearson correlation between each score feature and the UMAP axes.
        """
        umap_correlations = {}
        
        if not all_score_keys or self.df.empty:
            return umap_correlations
        
        # Prepare a clean DataFrame for correlation
        corr_df = self.df[self.df['cluster_id'] != -1].copy()
        
        # Add score columns to the dataframe
        for score_key in all_score_keys:
            corr_df[score_key] = corr_df['scores'].apply(
                lambda x: x.get(score_key, np.nan) if isinstance(x, dict) else np.nan
            ).astype(float)
        
        # Drop rows with NaN in UMAP coordinates
        corr_df = corr_df.dropna(subset=['umap_x', 'umap_y'])
        
        if len(corr_df) < 2:
            return umap_correlations

        for score_key in all_score_keys:
            try:
                corr_x = corr_df['umap_x'].corr(corr_df[score_key])
                corr_y = corr_df['umap_y'].corr(corr_df[score_key])
                umap_correlations[score_key] = {
                    'umap_x_corr': float(corr_x) if not pd.isna(corr_x) else 0.0,
                    'umap_y_corr': float(corr_y) if not pd.isna(corr_y) else 0.0
                }
            except Exception as e:
                print(f"Warning: Could not calculate correlation for {score_key}: {e}")
                continue

        return umap_correlations
    
    def _calculate_global_feature_stats(self, all_score_keys: List[str]) -> Dict[str, Any]:
        """Calculate global statistics for all features"""
        global_stats = {}
        
        for score_key in all_score_keys:
            values = []
            for item in self.data:
                if isinstance(item['scores'], dict) and score_key in item['scores']:
                    try:
                        values.append(float(item['scores'][score_key]))
                    except (ValueError, TypeError):
                        continue
            
            if values:
                values_array = np.array(values)
                global_stats[score_key] = {
                    'mean': float(np.mean(values_array)),
                    'median': float(np.median(values_array)),
                    'std': float(np.std(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                    'q25': float(np.percentile(values_array, 25)),
                    'q75': float(np.percentile(values_array, 75)),
                    'entropy': float(entropy(np.histogram(values_array, bins=10)[0] + 1e-10))  # Add small constant to avoid log(0)
                }
        
        return global_stats
    
    def create_enhanced_analysis_context(self, pattern_analysis: Dict[str, Any], 
                                       search_query_filter: Optional[str] = None) -> str:
        """
        Create analysis context focusing on patterns rather than simple statistics
        """
        cluster_patterns = pattern_analysis['cluster_patterns']
        feature_importance = pattern_analysis['feature_importance']
        umap_correlations = pattern_analysis['umap_correlations']
        noise_info = f"Noise/outlier points: {pattern_analysis['noise_count']} ({pattern_analysis['noise_percentage']:.1f}%)"
        
        context = f"""ENHANCED PATTERN-BASED CLUSTER ANALYSIS
        
Dataset Overview:
- Total samples: {len(self.data)}
- Meaningful clusters: {len(cluster_patterns)}
- {noise_info}
- Search query filter: {search_query_filter or 'None (all data)'}

FEATURE IMPORTANCE (Pattern-Based):
The following features show the strongest clustering patterns:"""
        
        # Sort features by pattern-based importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:5], 1):
            context += f"\n  {i}. {feature}: {importance:.2f} (higher = more distinctive patterns across clusters)"

        context += "\n\nUMAP DIMENSION ANALYSIS:\n"
        if umap_correlations:
            # Sort features by absolute correlation with UMAP_x
            sorted_x_corr = sorted(umap_correlations.items(), key=lambda x: abs(x[1]['umap_x_corr']), reverse=True)
            context += f"Features most correlated with UMAP X-axis (horizontal separation):\n"
            for feature, corr in sorted_x_corr[:3]:
                sign = "positively" if corr['umap_x_corr'] > 0 else "negatively"
                context += f"  • {feature}: {sign} correlated (r={corr['umap_x_corr']:.2f})\n"

            # Sort features by absolute correlation with UMAP_y
            sorted_y_corr = sorted(umap_correlations.items(), key=lambda x: abs(x[1]['umap_y_corr']), reverse=True)
            context += f"Features most correlated with UMAP Y-axis (vertical separation):\n"
            for feature, corr in sorted_y_corr[:3]:
                sign = "positively" if corr['umap_y_corr'] > 0 else "negatively"
                context += f"  • {feature}: {sign} correlated (r={corr['umap_y_corr']:.2f})\n"
        else:
            context += "No UMAP correlations could be calculated.\n"
        
        context += "\n\nCLUSTER PATTERN ANALYSIS:\n"
        
        # Sort clusters by size for consistent ordering
        sorted_clusters = sorted(cluster_patterns.items(), 
                               key=lambda x: x[1]['size'], reverse=True)
        
        for cluster_id, cluster_info in sorted_clusters:
            context += f"\n=== CLUSTER {cluster_id} ===\n"
            context += f"Size: {cluster_info['size']} comments ({cluster_info['percentage']:.1f}% of dataset)\n"
            
            # Detected patterns
            patterns_found = []
            for feature, feature_patterns in cluster_info['patterns'].items():
                for pattern in feature_patterns:
                    if pattern['strength'] > 0.3:  # Only include strong patterns
                        patterns_found.append({
                            'feature': feature,
                            'description': pattern['description'],
                            'strength': pattern['strength'],
                            'type': pattern['type']
                        })
            
            if patterns_found:
                # Sort by strength
                patterns_found.sort(key=lambda x: x['strength'], reverse=True)
                context += "Behavioral Patterns Detected:\n"
                
                for pattern in patterns_found[:5]:  # Top 5 patterns
                    strength_desc = "very strong" if pattern['strength'] > 0.8 else "strong" if pattern['strength'] > 0.5 else "moderate"
                    context += f"  • {pattern['description']} ({strength_desc} pattern)\n"
            else:
                context += "No strong distinctive patterns detected.\n"
            
            # Text characteristics
            text_analysis = cluster_info.get('text_analysis', {})
            if text_analysis:
                avg_length = text_analysis.get('avg_length', 0)
                length_desc = "very long" if avg_length > 100 else "long" if avg_length > 50 else "medium" if avg_length > 20 else "short"
                context += f"Comment Style: {length_desc} comments (avg: {avg_length:.1f} words)\n"
                
                # Communication patterns
                comm_patterns = []
                if text_analysis.get('question_percentage', 0) > 0.3:
                    comm_patterns.append(f"frequently asks questions ({text_analysis['question_percentage']:.1%})")
                if text_analysis.get('exclamation_percentage', 0) > 0.2:
                    comm_patterns.append(f"uses exclamations often ({text_analysis['exclamation_percentage']:.1%})")
                if text_analysis.get('caps_percentage', 0) > 0.1:
                    comm_patterns.append(f"uses caps for emphasis ({text_analysis['caps_percentage']:.1%})")
                
                if comm_patterns:
                    context += f"Communication Style: {', '.join(comm_patterns)}\n"
                
                # Distinctive terms
                if text_analysis.get('distinctive_terms'):
                    top_terms = [term for term, _ in text_analysis['distinctive_terms'][:8]]
                    context += f"Key Terms: {', '.join(top_terms)}\n"
            
            # Sample comments (shortened for context)
            if cluster_info.get('sample_comments'):
                context += "Example Comments:\n"
                for i, comment in enumerate(cluster_info['sample_comments'][:3], 1):
                    excerpt = comment[:100].strip()
                    if len(comment) > 100:
                        excerpt += "..."
                    context += f"  {i}. \"{excerpt}\"\n"
        
        context += f"\n\nPATTERN-BASED INSIGHTS:\n"
        
        # Generate insights based on pattern analysis
        insights = []
        
        # Feature insights
        if sorted_features:
            top_feature = sorted_features[0]
            if top_feature[1] > 0.5:
                insights.append(f"'{top_feature[0]}' shows the strongest clustering patterns, driving most cluster distinctions")
        
        # Cluster quality insights
        strong_pattern_clusters = sum(1 for _, info in cluster_patterns.items() 
                                    if any(any(p['strength'] > 0.5 for p in patterns) 
                                          for patterns in info['patterns'].values()))
        
        if strong_pattern_clusters >= len(cluster_patterns) * 0.7:
            insights.append("Most clusters show strong behavioral patterns, indicating good clustering quality")
        elif strong_pattern_clusters < len(cluster_patterns) * 0.3:
            insights.append("Few clusters show clear patterns - consider adjusting cluster count or feature selection")
        
        # Size distribution insights
        if cluster_patterns:
            sizes = [info['size'] for info in cluster_patterns.values()]
            size_ratio = max(sizes) / min(sizes) if min(sizes) > 0 else 0
            if size_ratio > 5:
                insights.append("Highly uneven cluster sizes suggest some behavioral patterns are much more common")
        
        # Add insights to context
        for insight in insights:
            context += f"• {insight}\n"
        
        context += f"\nMETHODOLOGY NOTE:\n"
        context += "This analysis focuses on detecting behavioral and statistical patterns that distinguish clusters, "
        context += "rather than just comparing means. Patterns include consistency, extreme values, distribution shapes, "
        context += "and communication styles. This approach better reflects how Gower distance clustering actually works."
        
        return context
    
    def generate_enhanced_interpretation(self, search_query_filter: Optional[str] = None, 
                                       prompts_file_used_for_scoring: Optional[str] = None) -> Tuple[str, str]:
        """
        Generate enhanced interpretation focusing on patterns rather than simple statistics
        """
        print("Loading data from vector database...")
        if search_query_filter:
            print(f"Filtering by search_query metadata: '{search_query_filter}'")
        
        data = self.load_data_from_vectordb(search_query_filter)
        
        if not data:
            if search_query_filter:
                return f"No data found in vector database with search_query='{search_query_filter}'.", ""
            else:
                return "No data found in vector database.", ""
        
        print("Detecting cluster patterns...")
        pattern_analysis = self.detect_cluster_patterns()
        
        print("Creating enhanced analysis context...")
        analysis_context = self.create_enhanced_analysis_context(pattern_analysis, search_query_filter)
        
        # Load score interpretations if provided
        legend_text = ""
        if prompts_file_used_for_scoring:
            try:
                # Try to load score interpretations from the prompts file
                with open(prompts_file_used_for_scoring, 'r') as f:
                    prompts_data = json.load(f)
                    if 'score_interpretations' in prompts_data:
                        legend_text = json.dumps(prompts_data['score_interpretations'], indent=2)
                    elif 'prompts' in prompts_data:
                        # Extract score information from prompts
                        legend_parts = []
                        for prompt_item in prompts_data['prompts']:
                            if 'name' in prompt_item and 'scale' in prompt_item:
                                legend_parts.append(f"{prompt_item['name']}: {prompt_item.get('scale', 'N/A')}")
                        if legend_parts:
                            legend_text = "\n".join(legend_parts)
            except Exception as e:
                print(f"Warning: Could not load score interpretations from {prompts_file_used_for_scoring}: {e}")
                legend_text = ""
            
            if legend_text:
                analysis_context = "\n\nSCORE INTERPRETATION LEGEND:\n" + legend_text + "\n\n" + analysis_context
        
        print("Generating enhanced LLM interpretation...")
        try:
            messages = self.analysis_prompt_template.format_messages(analysis_context=analysis_context)
            interpretation = self.rag_chatbot.chat_model.invoke(messages)
            return interpretation.content, legend_text
        except Exception as e:
            print(f"Error generating interpretation: {e}")
            return f"Error generating interpretation: {e}", legend_text


def analyze_vector_clusters_enhanced(
    persist_directory: str, 
    search_query_filter: Optional[str] = None,
    embed_model: str = "nomic-embed-text:latest", 
    chat_model: str = "llama3.1:8b",
    embed_backend: Optional[str] = None,
    embed_api_key: Optional[str] = None,
    chat_backend: Optional[str] = None,
    chat_api_key: Optional[str] = None,
    prompts_file_used_for_scoring: Optional[str] = None
) -> Tuple[str, str]:
    """
    Enhanced convenience function for pattern-aware cluster analysis
    
    Args:
        persist_directory: Path to vector database
        search_query_filter: Optional filter for search query
        embed_model: Embedding model name
        chat_model: Chat model name
        embed_backend: Embedding backend (openai, gemini, ollama, or None for auto-detect)
        embed_api_key: API key for embedding backend
        chat_backend: Chat backend (openai, gemini, anthropic, ollama, or None for auto-detect)
        chat_api_key: API key for chat backend
        prompts_file_used_for_scoring: Optional path to prompts file for score interpretations
    
    Returns:
        Tuple of (interpretation text, legend text)
    """
    analyzer = EnhancedVectorClusterAnalyzer(
        persist_directory=persist_directory,
        embed_model=embed_model,
        chat_model=chat_model,
        embed_backend=embed_backend,
        embed_api_key=embed_api_key,
        chat_backend=chat_backend,
        chat_api_key=chat_api_key
    )
    
    return analyzer.generate_enhanced_interpretation(
        search_query_filter=search_query_filter,
        prompts_file_used_for_scoring=prompts_file_used_for_scoring
    )


# Usage example:
if __name__ == "__main__":
    persist_directory = "chroma_db"
    search_query = "politics"

    # Example 1: Using Ollama (local models)
    print("Example 1: Using Ollama (local models)")
    print("=" * 60)
    try:
        interpretation, scoring_info = analyze_vector_clusters_enhanced(
            persist_directory=persist_directory,
            search_query_filter=search_query,
            embed_model="nomic-embed-text:latest",
            chat_model="llama3.1:8b",
            embed_backend="ollama",
            chat_backend="ollama",
            prompts_file_used_for_scoring="json/tariffs.json"
        )
        
        print("Enhanced Cluster Analysis Interpretation:")
        print("=" * 50)
        print(interpretation)
        if scoring_info:
            print("\nLegend:\n", scoring_info)
        
    except Exception as e:
        print(f"Error analyzing clusters: {e}")

    # Example 2: Using OpenAI embeddings and chat
    print("\n\nExample 2: Using OpenAI embeddings and chat")
    print("=" * 60)
    try:
        interpretation, scoring_info = analyze_vector_clusters_enhanced(
            persist_directory=persist_directory,
            search_query_filter=search_query,
            embed_model="text-embedding-3-small",
            chat_model="gpt-4",
            embed_backend="openai",
            chat_backend="openai",
            # API keys will be read from environment variables
            prompts_file_used_for_scoring="json/tariffs.json"
        )
        
        print("Enhanced Cluster Analysis Interpretation:")
        print("=" * 50)
        print(interpretation)
        if scoring_info:
            print("\nLegend:\n", scoring_info)
        
    except Exception as e:
        print(f"Error analyzing clusters: {e}")

    # Example 3: Mixed - Ollama embeddings + OpenAI chat
    print("\n\nExample 3: Mixed - Ollama embeddings + OpenAI chat")
    print("=" * 60)
    try:
        interpretation, scoring_info = analyze_vector_clusters_enhanced(
            persist_directory=persist_directory,
            search_query_filter=search_query,
            embed_model="nomic-embed-text:latest",
            chat_model="gpt-4",
            embed_backend="ollama",
            chat_backend="openai",
            prompts_file_used_for_scoring="json/tariffs.json"
        )
        
        print("Enhanced Cluster Analysis Interpretation:")
        print("=" * 50)
        print(interpretation)
        if scoring_info:
            print("\nLegend:\n", scoring_info)
        
    except Exception as e:
        print(f"Error analyzing clusters: {e}")