# Enhanced ClusterVisualizer with consistent orientation
#
# By Ian Drumm, The Univesity of Salford, UK.
#
import numpy as np
from datetime import datetime, date
from typing import List, Dict, Tuple, Literal, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import umap
from kmedoids import KMedoids
from sklearn.metrics.pairwise import pairwise_distances
import hdbscan
from scipy.spatial import procrustes

class ClusterVisualizer:
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.reference_embedding = None
        self.reference_method = None
        
    def create_base_features(self, data: List[Dict], date_range=None):
        """Extract and preprocess features consistently across all methods"""
        # Filter by date
        filtered = []

        for item in data:
            try:
                created_str = item["comment_metadata"]["created_utc"]

                

                if created_str.endswith('Z'):
                    created_str = created_str[:-1] + '+00:00'
                created_dt = datetime.fromisoformat(created_str).date()
                created_dt = min(created_dt, datetime.today().date())
                if date_range is None or (date_range[0] <= created_dt <= date_range[1]):
                    filtered.append(item)
            except Exception:
                continue
        
        # Fixed: Return 3 values to match unpacking expectation
        if not filtered:
            return np.array([]), [], []
        
        # Extract features
        features = []
        valid_items = []
        category_only_items = []
        expected_score_len = None
        
        for item in filtered:
            has_valid_scores = False
            try:
                if "scores" in item and isinstance(item["scores"], dict):
                    sorted_keys = sorted(item["scores"].keys())
                    score_vals = [float(item["scores"][k]) for k in sorted_keys 
                                if item["scores"][k] is not None]
                    
                    if expected_score_len is None:
                        expected_score_len = len(score_vals)
                    
                    if len(score_vals) == expected_score_len and len(score_vals) > 0:
                        features.append(score_vals)
                        valid_items.append(item)
                        has_valid_scores = True
            except (KeyError, ValueError, TypeError):
                pass
            
            if not has_valid_scores:
                category_only_items.append(item)
        
        return np.array(features) if features else np.array([]), valid_items, category_only_items
    
    def align_to_reference(self, embedding, method_name):
        """Align embedding to reference using Procrustes analysis"""
        if self.reference_embedding is None:
            self.reference_embedding = embedding.copy()
            self.reference_method = method_name
            return embedding
        
        try:
            _, aligned_embedding, _ = procrustes(self.reference_embedding, embedding)
            return aligned_embedding
        except:
            return self._simple_alignment(self.reference_embedding, embedding)
    
    def _simple_alignment(self, reference, embedding):
        """Simple alignment using transformations"""
        best_embedding = embedding.copy()
        best_score = np.inf
        
        transformations = [
            lambda x: x,
            lambda x: x * [-1, 1],
            lambda x: x * [1, -1],
            lambda x: x * [-1, -1],
        ]
        
        for transform in transformations:
            transformed = transform(embedding)
            score = np.sum((reference - transformed) ** 2)
            if score < best_score:
                best_score = score
                best_embedding = transformed
        
        return best_embedding
    
    def run_clustering(
        self,
        data: List[Dict],
        date_range: Optional[Tuple[date, date]] = None,
        clustering_method: Literal["gower_kmedoids", "hdbscan", "umap_kmeans"] = "gower_kmedoids",
        hdbscan_min_cluster_size: Optional[int] = None,
        hdbscan_noise_tolerance: float = 0.2,
        hdbscan_target_low: Optional[int] = None,
        hdbscan_target_high: Optional[int] = None,
        align_embeddings: bool = True
    ) -> List[Dict]:
        """
        Enhanced clustering with consistent orientation
        """
        features_np, valid_items, category_only_items = self.create_base_features(data, date_range)
        
        # If no items have numerical scores, assign all to cluster -1
        if len(valid_items) == 0:
            print("No numerical score features found - assigning all items cluster_id = -1")
            for item in category_only_items:
                item["umap_x"] = 0.0
                item["umap_y"] = 0.0
                item["cluster_id"] = -1
            return category_only_items
        
        # Handle minimal data case
        if len(valid_items) < 2:
            for item in valid_items + category_only_items:
                item["umap_x"] = 0.0
                item["umap_y"] = 0.0
                item["cluster_id"] = 0 if item in valid_items else -1
            return valid_items + category_only_items
        
        # Consistent base preprocessing for all methods
        base_scaler = MinMaxScaler()
        features_normalized = base_scaler.fit_transform(features_np)
        
        embedding = None
        labels = None
        
        try:
            if clustering_method == "umap_kmeans":
                distance_matrix = pairwise_distances(features_normalized, metric='manhattan')
                
                if distance_matrix.max() > 0:
                    distance_matrix = distance_matrix / distance_matrix.max()
                
                n_neighbors = min(15, len(features_normalized) - 1)
                reducer = umap.UMAP(
                    n_neighbors=n_neighbors, 
                    min_dist=0.1, 
                    random_state=42,
                    metric='precomputed'
                )
                embedding = reducer.fit_transform(distance_matrix)
                
                actual_n_clusters = min(self.n_clusters, len(features_normalized))
                kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init='auto')
                labels = kmeans.fit_predict(embedding)
            
            elif clustering_method in ["gower_kmedoids", "hdbscan"]:
                distance_matrix = pairwise_distances(features_normalized, metric='manhattan')
                
                if distance_matrix.max() > 0:
                    distance_matrix = distance_matrix / distance_matrix.max()
                
                n_neighbors = min(15, len(features_normalized) - 1)
                reducer = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    random_state=42,
                    metric='precomputed'
                )
                embedding = reducer.fit_transform(distance_matrix)
                
                if clustering_method == "gower_kmedoids":
                    actual_n_clusters = min(self.n_clusters, len(features_normalized))
                    kmedoids = KMedoids(n_clusters=actual_n_clusters, metric='precomputed')
                    labels = kmedoids.fit_predict(distance_matrix)
                
                else:  # hdbscan
                    N = len(valid_items)
                    
                    if hdbscan_target_low is None or hdbscan_target_high is None:
                        hdbscan_target_low = max(1, self.n_clusters - 1)
                        hdbscan_target_high = self.n_clusters + 1
                    
                    def run_once(mcs, ms, eps):
                        mcs = min(mcs, N - 1)
                        ms = min(ms, mcs)
                        
                        if mcs <= 1:
                            return np.zeros(N, dtype=int), 1, 0.0
                        
                        clusterer = hdbscan.HDBSCAN(
                            metric='precomputed',
                            min_cluster_size=mcs,
                            min_samples=ms,
                            cluster_selection_method='eom',
                            cluster_selection_epsilon=eps,
                            prediction_data=False
                        )
                        lbls = clusterer.fit_predict(distance_matrix)
                        k = len(set(lbls)) - (1 if -1 in lbls else 0)
                        noise_frac = (lbls == -1).mean() if hasattr(lbls, "mean") else (list(lbls).count(-1) / len(lbls))
                        return lbls, k, noise_frac
                    
                    target_low = hdbscan_target_low
                    target_high = hdbscan_target_high
                    
                    if hdbscan_min_cluster_size is not None:
                        mcs = min(int(hdbscan_min_cluster_size), N - 1)
                        ms = max(1, min(int(hdbscan_noise_tolerance * mcs), mcs))
                        eps = 0.05
                        labels, k, _ = run_once(mcs, ms, eps)
                        if k > target_high:
                            labels, k, _ = run_once(mcs, ms, 0.10)
                    else:
                        low = max(2, min(int(0.05 * N), N // 4))
                        high = max(low + 1, min(int(0.35 * N), N - 1))
                        mcs = max(low, min(int(0.10 * N), N // 3))
                        eps = 0.05
                        
                        for _ in range(6):
                            ms = max(1, min(int(hdbscan_noise_tolerance * mcs), mcs))
                            labels, k, noise = run_once(mcs, ms, eps)
                            
                            if k > target_high:
                                if k <= target_high + 1 and eps < 0.10:
                                    eps = 0.10
                                else:
                                    mcs = min(high, max(mcs + 1, int(mcs * 1.25)))
                                    eps = 0.05
                            elif k < target_low:
                                mcs = max(low, int(mcs * 0.85))
                                eps = 0.05
                            else:
                                break
                        
                        if k > target_high:
                            labels, k, _ = run_once(mcs, ms, 0.10)
                    
                    def _assign_noise_to_nearest(labels, D, quantile=0.95):
                        labels = labels.copy()
                        clusters = [c for c in np.unique(labels) if c != -1]
                        if not clusters or not np.any(labels == -1):
                            return labels
                        
                        reps = []
                        for c in clusters:
                            idx = np.where(labels == c)[0]
                            if len(idx) == 0:
                                continue
                            subD = D[np.ix_(idx, idx)]
                            reps.append(idx[np.argmin(subD.sum(axis=1))])
                        
                        radii = []
                        for c, r in zip(clusters, reps):
                            idx = np.where(labels == c)[0]
                            if len(idx) > 0:
                                radii.append(np.quantile(D[idx, r], quantile))
                            else:
                                radii.append(0)
                        
                        for i in np.where(labels == -1)[0]:
                            if len(reps) == 0:
                                break
                            d_to_reps = D[i, reps]
                            j = int(np.argmin(d_to_reps))
                            if j < len(radii) and d_to_reps[j] <= radii[j]:
                                labels[i] = clusters[j]
                        return labels
                    
                    labels = _assign_noise_to_nearest(labels, distance_matrix, quantile=0.95)
        
        except Exception as e:
            labels = np.zeros(len(valid_items), dtype=int)
            if embedding is None:
                embedding = np.column_stack([
                    np.arange(len(valid_items), dtype=float),
                    np.zeros(len(valid_items), dtype=float)
                ])
        
        # Apply alignment if requested
        if align_embeddings and embedding is not None:
            embedding = self.align_to_reference(embedding, clustering_method)
        
        # Assign results
        for i, item in enumerate(valid_items):
            if embedding is not None and i < len(embedding):
                item["umap_x"] = float(embedding[i][0])
                item["umap_y"] = float(embedding[i][1])
            else:
                item["umap_x"] = float(i)
                item["umap_y"] = 0.0
            
            if labels is not None and i < len(labels):
                item["cluster_id"] = int(labels[i])
            else:
                item["cluster_id"] = 0
        
        # Add category-only items
        for item in category_only_items:
            item["umap_x"] = 0.0
            item["umap_y"] = 0.0
            item["cluster_id"] = -1
        
        return valid_items + category_only_items
    
    def reset_reference(self):
        """Reset reference embedding (call when switching datasets)"""
        self.reference_embedding = None
        self.reference_method = None

# Usage in your Streamlit app:
# Replace your existing ClusterVisualizer instantiation with:
# clusterer = EnhancedClusterVisualizer(n_clusters=n_clusters if n_clusters else 5)

# Add this checkbox in your re-clustering section:
# align_visualizations = st.checkbox("Keep consistent orientation across methods", value=True)

# Then use align_embeddings parameter:
# clustered_data = clusterer.run_clustering(
#     data=data_to_cluster,
#     clustering_method=clustering_method,
#     align_embeddings=align_visualizations
# )