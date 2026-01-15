"""
K-means Clustering Model for Dengue Risk Zone Classification
Clusters time periods and regions into risk zones based on outbreak patterns
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle


class DengueClusteringModel:
    """
    K-means clustering model for identifying dengue risk zones
    """
    
    def __init__(self, n_clusters=3, random_state=42):
        """
        Initialize clustering model
        
        Args:
            n_clusters: Number of risk zones (default: 3 for Low/Medium/High)
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.is_trained = False
        self.cluster_centers = None
        self.cluster_labels_map = {}
        
    def train(self, X_train, y_train=None):
        """
        Train the K-means clustering model
        
        Args:
            X_train: Training features
            y_train: Optional - dengue cases for labeling clusters
            
        Returns:
            dict: Training results
        """
        print("\n" + "="*60)
        print("TRAINING K-MEANS CLUSTERING MODEL")
        print("="*60 + "\n")
        
        # Fit the model
        self.model.fit(X_train)
        self.is_trained = True
        
        # Get cluster centers
        self.cluster_centers = self.model.cluster_centers_
        
        # Get cluster assignments
        cluster_labels = self.model.labels_
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_train, cluster_labels)
        
        print(f"✓ Model trained with {self.n_clusters} clusters")
        print(f"✓ Silhouette Score: {silhouette_avg:.4f}")
        
        # Label clusters based on average dengue cases
        if y_train is not None:
            self._label_clusters(cluster_labels, y_train)
        
        results = {
            'n_clusters': self.n_clusters,
            'silhouette_score': silhouette_avg,
            'inertia': self.model.inertia_
        }
        
        return results
    
    def _label_clusters(self, cluster_labels, dengue_cases):
        """
        Assign risk level names to clusters based on average dengue cases
        
        Args:
            cluster_labels: Cluster assignments
            dengue_cases: Actual dengue case counts
        """
        # Calculate average cases per cluster
        cluster_avg_cases = {}
        
        for cluster_id in range(self.n_clusters):
            mask = cluster_labels == cluster_id
            avg_cases = dengue_cases[mask].mean()
            cluster_avg_cases[cluster_id] = avg_cases
        
        # Sort clusters by average cases
        sorted_clusters = sorted(cluster_avg_cases.items(), key=lambda x: x[1])
        
        # Assign labels based on case counts
        if self.n_clusters == 3:
            risk_levels = ['Low Risk', 'Medium Risk', 'High Risk']
        else:
            risk_levels = [f'Risk Zone {i+1}' for i in range(self.n_clusters)]
        
        for idx, (cluster_id, avg_cases) in enumerate(sorted_clusters):
            self.cluster_labels_map[cluster_id] = {
                'label': risk_levels[idx] if idx < len(risk_levels) else f'Zone {idx+1}',
                'avg_cases': avg_cases
            }
        
        print("\nCluster Risk Levels:")
        for cluster_id, info in self.cluster_labels_map.items():
            print(f"  Cluster {cluster_id}: {info['label']} (Avg: {info['avg_cases']:.1f} cases)")
    
    def predict(self, X):
        """
        Predict cluster assignments for new data
        
        Args:
            X: Feature data
            
        Returns:
            array: Cluster assignments
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def get_cluster_labels(self, X):
        """
        Get risk level labels for data points
        
        Args:
            X: Feature data
            
        Returns:
            list: Risk level labels
        """
        cluster_ids = self.predict(X)
        
        labels = []
        for cluster_id in cluster_ids:
            if cluster_id in self.cluster_labels_map:
                labels.append(self.cluster_labels_map[cluster_id]['label'])
            else:
                labels.append(f'Cluster {cluster_id}')
        
        return labels
    
    def analyze_clusters(self, X, feature_names=None):
        """
        Analyze cluster characteristics
        
        Args:
            X: Feature data
            feature_names: Names of features
            
        Returns:
            DataFrame: Cluster analysis
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        print("\n" + "="*60)
        print("CLUSTER ANALYSIS")
        print("="*60 + "\n")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        # Create DataFrame with cluster centers
        centers_df = pd.DataFrame(
            self.cluster_centers,
            columns=feature_names
        )
        centers_df.insert(0, 'Cluster', range(self.n_clusters))
        
        # Add risk labels
        if self.cluster_labels_map:
            centers_df['Risk_Level'] = centers_df['Cluster'].map(
                lambda x: self.cluster_labels_map.get(x, {}).get('label', f'Cluster {x}')
            )
        
        print("Cluster Centers (Feature Averages):")
        print(centers_df)
        
        # Get cluster assignments
        cluster_assignments = self.predict(X)
        
        # Count samples per cluster
        unique, counts = np.unique(cluster_assignments, return_counts=True)
        print("\nCluster Distribution:")
        for cluster_id, count in zip(unique, counts):
            percentage = (count / len(cluster_assignments)) * 100
            risk_label = self.cluster_labels_map.get(cluster_id, {}).get('label', f'Cluster {cluster_id}')
            print(f"  {risk_label}: {count} samples ({percentage:.1f}%)")
        
        return centers_df
    
    def find_optimal_clusters(self, X, max_clusters=10):
        """
        Find optimal number of clusters using elbow method
        
        Args:
            X: Feature data
            max_clusters: Maximum number of clusters to test
            
        Returns:
            dict: Scores for different cluster counts
        """
        print("\n" + "="*60)
        print("FINDING OPTIMAL NUMBER OF CLUSTERS")
        print("="*60 + "\n")
        
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
            
            print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.4f}")
        
        return {
            'K': list(K_range),
            'inertia': inertias,
            'silhouette': silhouette_scores
        }
    
    def predict_risk_zone(self, features_dict):
        """
        Predict risk zone for a single observation
        
        Args:
            features_dict: Dictionary of feature values
            
        Returns:
            dict: Risk prediction
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Convert dict to array
        features_array = np.array(list(features_dict.values())).reshape(1, -1)
        
        # Get cluster assignment
        cluster_id = self.predict(features_array)[0]
        
        # Get risk information
        risk_info = self.cluster_labels_map.get(cluster_id, {
            'label': f'Cluster {cluster_id}',
            'avg_cases': 'Unknown'
        })
        
        return {
            'cluster_id': int(cluster_id),
            'risk_level': risk_info['label'],
            'expected_avg_cases': risk_info.get('avg_cases', 'Unknown')
        }
    
    def get_cluster_summary(self):
        """
        Get summary of all clusters
        
        Returns:
            dict: Cluster summary information
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        summary = {
            'n_clusters': self.n_clusters,
            'cluster_labels': self.cluster_labels_map
        }
        
        return summary
    
    def save_model(self, filepath):
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save model file
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'n_clusters': self.n_clusters,
            'cluster_labels_map': self.cluster_labels_map,
            'cluster_centers': self.cluster_centers
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Clustering model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model from disk
        
        Args:
            filepath: Path to model file
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.n_clusters = model_data['n_clusters']
        self.cluster_labels_map = model_data['cluster_labels_map']
        self.cluster_centers = model_data['cluster_centers']
        self.is_trained = True
        
        print(f"✓ Clustering model loaded from {filepath}")


def cluster_time_periods(df, date_column, cluster_column):
    """
    Analyze which time periods belong to which clusters
    
    Args:
        df: DataFrame with date and cluster assignments
        date_column: Name of date column
        cluster_column: Name of cluster column
        
    Returns:
        DataFrame: Time period analysis
    """
    df['Month'] = pd.to_datetime(df[date_column]).dt.month
    df['Year'] = pd.to_datetime(df[date_column]).dt.year
    
    analysis = df.groupby([cluster_column, 'Month']).size().reset_index(name='Count')
    
    return analysis


if __name__ == "__main__":
    # Example usage
    print("K-means Clustering Model for Dengue Risk Classification")
    print("This module should be used with preprocessed data")
    print("\nExample:")
    print("  model = DengueClusteringModel(n_clusters=3)")
    print("  model.train(X_train, y_train)")
    print("  clusters = model.predict(X_test)")
    print("  risk_levels = model.get_cluster_labels(X_test)")
