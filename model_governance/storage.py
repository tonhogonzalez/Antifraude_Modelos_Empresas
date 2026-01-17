"""
Storage utilities for model governance data persistence.
Uses JSON files for demo purposes.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Base directory for governance data
DATA_DIR = Path(__file__).parent / "data"
VERSION_FILE = DATA_DIR / "version_history.json"
METRICS_FILE = DATA_DIR / "performance_metrics.json"

def ensure_data_dir():
    """Ensure data directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def save_version(version_data: Dict[str, Any]) -> bool:
    """
    Save a new model version to history.
    
    Args:
        version_data: Dictionary containing version information
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_data_dir()
        
        # Load existing history
        history = load_history()
        
        # Add new version
        history.append(version_data)
        
        # Save updated history
        with open(VERSION_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error saving version: {e}")
        return False

def load_history() -> List[Dict[str, Any]]:
    """
    Load version history.
    
    Returns:
        List of version dictionaries
    """
    try:
        ensure_data_dir()
        
        if VERSION_FILE.exists():
            with open(VERSION_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Initialize with a default version if no history exists
            default_version = {
                "version": "1.0.0",
                "timestamp": "2026-01-15T10:00:00",
                "author": "System",
                "type": "initial",
                "description": "Versión inicial del modelo FraudHunter",
                "changes": {
                    "contamination": 0.05,
                    "n_estimators": 100,
                    "max_samples": "auto"
                },
                "metrics": {
                    "precision": 0.87,
                    "recall": 0.82,
                    "f1_score": 0.845,
                    "fpr": 0.13
                }
            }
            save_version(default_version)
            return [default_version]
    except Exception as e:
        print(f"Error loading history: {e}")
        return []

def save_metrics(metrics_data: Dict[str, Any]) -> bool:
    """
    Save performance metrics.
    
    Args:
        metrics_data: Dictionary containing metrics
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_data_dir()
        
        # Load existing metrics
        all_metrics = load_metrics()
        
        # Add timestamp if not present
        if 'timestamp' not in metrics_data:
            metrics_data['timestamp'] = datetime.now().isoformat()
        
        # Add new metrics
        all_metrics.append(metrics_data)
        
        # Save updated metrics
        with open(METRICS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error saving metrics: {e}")
        return False

def load_metrics() -> List[Dict[str, Any]]:
    """
    Load performance metrics history.
    
    Returns:
        List of metrics dictionaries
    """
    try:
        ensure_data_dir()
        
        if METRICS_FILE.exists():
            with open(METRICS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return []
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return []

def get_latest_version() -> Optional[Dict[str, Any]]:
    """Get the most recent version from history."""
    history = load_history()
    return history[-1] if history else None

def get_version_by_number(version_number: str) -> Optional[Dict[str, Any]]:
    """Get a specific version by its version number."""
    history = load_history()
    for version in history:
        if version.get('version') == version_number:
            return version
    return None
