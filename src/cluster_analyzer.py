# src/cluster_analyzer.py
#
# By Ian Drumm, The Univesity of Salford, UK.
#
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any, Optional
from langchain_community.chat_models import ChatOllama # For LLM summarization
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

class ClusterAnalyzer:
    def __init__(self, clustered_data: List[Dict]):
        self.data = clustered_data
        self.df = pd.DataFrame(self.data)
        # Ensure 'comment' and 'cluster_id' columns exist
        if 'comment' not in self.df.columns or 'cluster_id' not in self.df.columns:
            raise ValueError("Clustered data must contain 'comment' and 'cluster_id' columns.")

        # Initialize LLM for summarization
        # Make sure your Ollama server is running with 'llama3.1:8b' or your chosen model
        self.chat_model = ChatOllama(model="llama3.1:8B") 
        self.summary_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert in social media discourse analysis. Summarize the key themes and perspectives present in the following collection of Reddit comments."),
                ("user", "Summarize these comments:\n\n{comments}")
            ]
        )

    def get_top_keywords_per_cluster(self, top_n: int = 10) -> Dict[int, List[str]]:
        """
        Extracts top keywords for each cluster.
        This is a placeholder; you'd typically use TF-IDF, N-grams, or other text analysis methods.
        """
        cluster_keywords = defaultdict(lambda: defaultdict(int))
        
        # A very basic tokenization and counting for demonstration
        for _, row in self.df.iterrows():
            cluster_id = row['cluster_id']
            comment = str(row['comment']).lower() # Ensure comment is string
            # Simple tokenization: split by space, remove punctuation, filter short words
            words = [word.strip(".,!?\"'()[]{}/\\-_") for word in comment.split() if len(word) > 2] 
            for word in words:
                cluster_keywords[cluster_id][word] += 1
        
        top_keywords = {}
        for cluster_id, word_counts in cluster_keywords.items():
            sorted_words = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
            top_keywords[cluster_id] = [word for word, count in sorted_words[:top_n]]
            
        return dict(top_keywords)

    def get_cluster_summaries(self) -> Dict[int, str]:
        """
        Generates LLM-based summaries for each cluster.
        **Requires LLM setup and an active LLM instance.**
        """
        summaries = {}
        unique_cluster_ids = self.df['cluster_id'].unique()

        for cluster_id in unique_cluster_ids:
            # Filter comments for the current cluster, ensuring 'comment' column exists and is not None
            cluster_comments = self.df[self.df['cluster_id'] == cluster_id]['comment'].dropna().tolist()
            if not cluster_comments:
                summaries[cluster_id] = "No comments available for summarization in this cluster."
                continue

            # Take a sample of comments if there are too many for LLM context window
            # Adjust this number based on your LLM's context window limit. 
            # 50 comments might still be too many if comments are long. Consider fewer.
            sample_comments = cluster_comments[:20] # Limiting to 20 comments for summarization
            comments_str = "\n---\n".join(sample_comments)

            try:
                # LLM call to summarize
                inputs = {"comments": comments_str}
                summary_chain = self.summary_prompt_template | self.chat_model | StrOutputParser()
                llm_summary = summary_chain.invoke(inputs)
                summaries[cluster_id] = llm_summary
            except Exception as e:
                summaries[cluster_id] = f"Error generating summary for cluster {cluster_id}: {e}"
        return summaries

    def get_average_scores_per_cluster(self) -> Dict[int, Dict[str, float]]:
        """
        Calculates the average of scalar scores (ideology, agreement, etc.) per cluster.
        Assumes 'scores' is a dictionary within each item.
        """
        if 'scores' not in self.df.columns:
            print("Warning: 'scores' column not found in data. Cannot calculate average scores.")
            return {}

        # Initialize a dictionary to hold sums and counts for each score metric per cluster
        cluster_score_sums = defaultdict(lambda: defaultdict(float))
        cluster_score_counts = defaultdict(lambda: defaultdict(int))

        for _, row in self.df.iterrows():
            cluster_id = row['cluster_id']
            scores = row.get('scores') # 'scores' column should already be dict from JSON load by previous steps

            if isinstance(scores, dict): # Ensure 'scores' is a dictionary
                for metric, value in scores.items():
                    # Ensure the score value is numeric and within an expected range (e.g., 1-5)
                    if isinstance(value, (int, float)) and 1 <= value <= 5: 
                        cluster_score_sums[cluster_id][metric] += value
                        cluster_score_counts[cluster_id][metric] += 1

        # Calculate averages
        average_scores = {}
        for cluster_id, metric_sums in cluster_score_sums.items():
            average_scores[cluster_id] = {}
            for metric, total_sum in metric_sums.items():
                count = cluster_score_counts[cluster_id][metric]
                if count > 0:
                    average_scores[cluster_id][metric] = round(total_sum / count, 2) # Round for readability
                else:
                    average_scores[cluster_id][metric] = 0.0 # No data for this metric in cluster
        return average_scores