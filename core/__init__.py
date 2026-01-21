"""
FraudHunter Enterprise OS - Core Module
========================================
Sistema operativo de prevención de fraude para banca Tier-1.
"""

from .loaders import M200WideLoader, M347Loader, M349Loader
from .profilers import TaxProfiler, NetworkProfiler
from .features import TaxFeatureEngine, NetworkFeatureEngine
from .models import SectoralBenchmarker, ScoreComposer

__version__ = "2.0.0"
__all__ = [
    "M200WideLoader",
    "M347Loader", 
    "M349Loader",
    "TaxProfiler",
    "NetworkProfiler",
    "TaxFeatureEngine",
    "NetworkFeatureEngine",
    "SectoralBenchmarker",
    "ScoreComposer",
]
