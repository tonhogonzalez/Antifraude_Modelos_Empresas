"""
Model Version Manager

Handles versioning, changelog tracking, and model comparison for governance.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from .storage import save_version, load_history, get_latest_version, get_version_by_number


class ModelVersionManager:
    """Manages model versions and tracks changes over time."""
    
    def __init__(self):
        """Initialize the version manager."""
        self.history = load_history()
    
    def create_version(
        self,
        version: str,
        author: str,
        changes: Dict[str, Any],
        metrics: Dict[str, float],
        description: str = "",
        version_type: str = "manual"
    ) -> bool:
        """
        Create a new model version.
        
        Args:
            version: Version number (e.g., "1.1.0")
            author: Who created this version
            changes: Dictionary of parameter changes
            metrics: Performance metrics for this version
            description: Optional description of changes
            version_type: "manual", "automatic", or "scheduled"
            
        Returns:
            True if version was created successfully
        """
        version_data = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "author": author,
            "type": version_type,
            "description": description,
            "changes": changes,
            "metrics": metrics
        }
        
        if save_version(version_data):
            self.history = load_history()  # Reload history
            return True
        return False
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get version history.
        
        Args:
            limit: Maximum number of versions to return (most recent first)
            
        Returns:
            List of version dictionaries
        """
        history = list(reversed(self.history))  # Most recent first
        if limit:
            return history[:limit]
        return history
    
    def compare_versions(
        self,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare two versions and return the differences.
        
        Args:
            version1: First version number
            version2: Second version number
            
        Returns:
            Dictionary with changes and metric deltas
        """
        v1 = get_version_by_number(version1)
        v2 = get_version_by_number(version2)
        
        if not v1 or not v2:
            return {"error": "Version not found"}
        
        # Calculate parameter changes
        param_changes = {}
        all_params = set(v1['changes'].keys()) | set(v2['changes'].keys())
        
        for param in all_params:
            old_val = v1['changes'].get(param)
            new_val = v2['changes'].get(param)
            if old_val != new_val:
                param_changes[param] = {
                    "old": old_val,
                    "new": new_val,
                    "changed": True
                }
        
        # Calculate metric deltas
        metric_deltas = {}
        all_metrics = set(v1['metrics'].keys()) | set(v2['metrics'].keys())
        
        for metric in all_metrics:
            old_val = v1['metrics'].get(metric, 0)
            new_val = v2['metrics'].get(metric, 0)
            delta = new_val - old_val
            metric_deltas[metric] = {
                "old": old_val,
                "new": new_val,
                "delta": delta,
                "percent_change": (delta / old_val * 100) if old_val != 0 else 0
            }
        
        return {
            "version1": v1,
            "version2": v2,
            "parameter_changes": param_changes,
            "metric_deltas": metric_deltas
        }
    
    def get_champion_challenger(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Get the current champion (production) and challenger (latest) models.
        
        Returns:
            Tuple of (champion, challenger) version dictionaries
        """
        if len(self.history) < 2:
            champion = self.history[-1] if self.history else None
            return (champion, None)
        
        # Champion is second-to-last (current production)
        # Challenger is latest (candidate for promotion)
        return (self.history[-2], self.history[-1])
    
    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Get the most recent version."""
        return get_latest_version()
