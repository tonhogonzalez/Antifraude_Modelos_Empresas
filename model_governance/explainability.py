"""
Global Explainability Module

Provides feature importance and global model interpretation capabilities.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class GlobalExplainer:
    """Provides global model explanations and feature importance."""
    
    def __init__(self, model: Optional[IsolationForest] = None):
        """
        Initialize the explainer.
        
        Args:
            model: Trained Isolation Forest model (optional)
        """
        self.model = model
        self.feature_names = None
        self.feature_importance = None
    
    def calculate_feature_importance(
        self,
        X: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate feature importance using permutation-based approach.
        
        For Isolation Forest, we use the path length variance as proxy for importance.
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if feature_names is None:
            feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
        
        self.feature_names = feature_names
        
        if self.model is None:
            # Return dummy importance for demo
            importance = np.random.rand(len(feature_names))
            importance = importance / importance.sum()  # Normalize
        else:
            # Use feature importance from trees if available
            try:
                importance = self.model.feature_importances_
            except AttributeError:
                # Fallback: Calculate based on feature variance impact on scores
                base_scores = -self.model.score_samples(X)
                importance = []
                
                for i in range(X.shape[1]):
                    X_permuted = X.copy()
                    if isinstance(X, pd.DataFrame):
                        X_permuted.iloc[:, i] = np.random.permutation(X.iloc[:, i])
                    else:
                        X_permuted[:, i] = np.random.permutation(X[:, i])
                    
                    permuted_scores = -self.model.score_samples(X_permuted)
                    importance.append(np.abs(permuted_scores - base_scores).mean())
                
                importance = np.array(importance)
                importance = importance / importance.sum()  # Normalize
        
        self.feature_importance = dict(zip(feature_names, importance))
        return self.feature_importance
    
    def get_top_features(self, n: int = 15) -> List[Tuple[str, float]]:
        """
        Get top N most important features.
        
        Args:
            n: Number of top features to return
            
        Returns:
            List of (feature_name, importance) tuples, sorted by importance
        """
        if self.feature_importance is None:
            raise ValueError("Must call calculate_feature_importance first")
        
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:n]
    
    def sector_analysis(
        self,
        df: pd.DataFrame,
        sector_column: str = 'cnae_division'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze fraud patterns by sector.
        
        Args:
            df: DataFrame with company data
            sector_column: Name of sector column
            
        Returns:
            Dictionary with sector-level statistics
        """
        sector_stats = {}
        
        # Check if sector column exists, if not try common alternatives
        if sector_column not in df.columns:
            # Try common sector column names
            possible_columns = ['sector', 'cnae', 'cnae_code', 'industry', 'sector_code']
            sector_column = next((col for col in possible_columns if col in df.columns), None)
            
            # If still no sector column found, return empty dict
            if sector_column is None:
                return {}
        
        for sector in df[sector_column].unique():
            sector_df = df[df[sector_column] == sector]
            
            sector_stats[sector] = {
                "count": len(sector_df),
                "avg_fraud_score": sector_df['fraud_score'].mean() if 'fraud_score' in sector_df.columns else 0,
                "anomaly_rate": (sector_df['anomaly_label'] == -1).mean() if 'anomaly_label' in sector_df.columns else 0,
                "avg_revenue": sector_df['ventas_netas'].mean() if 'ventas_netas' in sector_df.columns else 0
            }
        
        return sector_stats
    
    def get_pattern_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of detected patterns.
        
        Args:
            df: DataFrame with analysis results
            
        Returns:
            Dictionary with pattern insights
        """
        patterns = {
            "total_companies": len(df),
            "anomalies_detected": (df['anomaly_label'] == -1).sum() if 'anomaly_label' in df.columns else 0,
            "anomaly_rate": (df['anomaly_label'] == -1).mean() if 'anomaly_label' in df.columns else 0,
            "avg_fraud_score": df['fraud_score'].mean() if 'fraud_score' in df.columns else 0,
            "high_risk_companies": (df['fraud_score'] > 0.7).sum() if 'fraud_score' in df.columns else 0
        }
        
        # Add flag-based patterns if available
        flag_columns = [col for col in df.columns if col.startswith('flag_')]
        if flag_columns:
            patterns["most_common_flags"] = {
                flag: df[flag].sum() for flag in flag_columns[:5]
            }
        
        return patterns
