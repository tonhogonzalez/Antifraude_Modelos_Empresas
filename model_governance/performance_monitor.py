"""
Performance Monitor

Tracks model performance metrics, detects drift, and generates alerts.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from scipy.stats import entropy
from .storage import save_metrics, load_metrics


class PerformanceMonitor:
    """Monitors model performance and detects degradation/drift."""
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.metrics_history = load_metrics()
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores (optional)
            
        Returns:
            Dictionary of metrics
        """
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "fpr": fpr,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn)
        }
        
        return metrics
    
    def detect_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        threshold: float = 0.1
    ) -> Tuple[bool, float, str]:
        """
        Detect distribution drift using KL divergence.
        
        Args:
            reference_data: Training/reference data distribution
            current_data: Current production data distribution
            threshold: KL divergence threshold for alert
            
        Returns:
            Tuple of (has_drift, kl_divergence, severity)
        """
        # Create histograms (normalized to probability distributions)
        bins = 20
        ref_hist, bin_edges = np.histogram(reference_data, bins=bins, density=True)
        curr_hist, _ = np.histogram(current_data, bins=bin_edges, density=True)
        
        # Normalize to probabilities (add small epsilon to avoid log(0))
        epsilon = 1e-10
        ref_dist = ref_hist / (ref_hist.sum() + epsilon) + epsilon
        curr_dist = curr_hist / (curr_hist.sum() + epsilon) + epsilon
        
        # Calculate KL divergence
        kl_div = entropy(curr_dist, ref_dist)
        
        # Determine severity
        if kl_div < threshold:
            severity = "none"
            has_drift = False
        elif kl_div < threshold * 2:
            severity = "low"
            has_drift = True
        elif kl_div < threshold * 4:
            severity = "medium"
            has_drift = True
        else:
            severity = "high"
            has_drift = True
        
        return has_drift, float(kl_div), severity
    
    def save_current_metrics(
        self,
        metrics: Dict[str, Any],
        additional_info: Optional[Dict] = None
    ) -> bool:
        """
        Save current metrics to history.
        
        Args:
            metrics: Performance metrics dictionary
            additional_info: Additional metadata
            
        Returns:
            True if saved successfully
        """
        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        if additional_info:
            metrics_data.update(additional_info)
        
        if save_metrics(metrics_data):
            self.metrics_history = load_metrics()
            return True
        return False
    
    def get_metrics_history(
        self,
        days: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get metrics history.
        
        Args:
            days: Number of days to look back (None = all history)
            
        Returns:
            List of metrics dictionaries
        """
        if not days:
            return self.metrics_history
        
        cutoff = datetime.now() - timedelta(days=days)
        filtered = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m['timestamp']) > cutoff
        ]
        return filtered
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """
        Generate alerts based on current metrics and trends.
        
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        if len(self.metrics_history) < 2:
            return alerts
        
        latest = self.metrics_history[-1]
        previous = self.metrics_history[-2]
        
        # Check for FPR increase
        if 'metrics' in latest and 'metrics' in previous:
            latest_fpr = latest['metrics'].get('fpr', 0)
            prev_fpr = previous['metrics'].get('fpr', 0)
            
            if latest_fpr > prev_fpr * 1.2:  # 20% increase
                alerts.append({
                    "type": "fpr_increase",
                    "severity": "warning",
                    "message": f"FPR aumentó de {prev_fpr:.2%} a {latest_fpr:.2%}",
                    "timestamp": latest['timestamp']
                })
            
            # Check for precision drop
            latest_prec = latest['metrics'].get('precision', 0)
            prev_prec = previous['metrics'].get('precision', 0)
            
            if latest_prec < prev_prec * 0.9:  # 10% drop
                alerts.append({
                    "type": "precision_drop",
                    "severity": "critical",
                    "message": f"Precisión cayó de {prev_prec:.2%} a {latest_prec:.2%}",
                    "timestamp": latest['timestamp']
                })
        
        return alerts
