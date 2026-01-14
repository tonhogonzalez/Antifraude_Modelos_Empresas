# =============================================================================
# Continuous Learning Module for FraudHunter
# =============================================================================
"""
Módulo de Aprendizaje Continuo que permite al sistema de detección de fraude
aprender de las decisiones de los analistas.

Componentes:
- FeedbackStorePandas: Gestión de feedback de analistas (versión local)
- FeedbackStoreSpark: Gestión de feedback de analistas (versión Databricks)
- HybridFraudReRanker: Re-ranking supervisado de alertas
- AdaptiveThresholdManager: Ajuste dinámico de umbrales
"""

# Import feature flags first (no dependencies)
from .feature_flags import (
    ContinuousLearningConfig, 
    FeatureFlags,
    AuditLogger,
    get_config,
    get_flags,
    get_audit_logger
)

# Import modules with lazy loading to avoid circular imports
from .feedback_store import (
    FeedbackStorePandas,
    FeedbackRecord,
    get_feedback_store
)

from .hybrid_reranker import (
    HybridFraudReRanker,
    ReRankerMetrics,
    get_reranker,
    run_shadow_mode_analysis
)

from .adaptive_thresholds import (
    AdaptiveThresholdManager,
    ThresholdRule,
    ThresholdSuggestion,
    get_threshold_manager,
    generate_threshold_report
)

__version__ = "1.0.0"
__all__ = [
    # Config
    "ContinuousLearningConfig",
    "FeatureFlags",
    "AuditLogger",
    "get_config",
    "get_flags",
    "get_audit_logger",
    # Feedback
    "FeedbackStorePandas",
    "FeedbackRecord",
    "get_feedback_store",
    # ReRanker
    "HybridFraudReRanker",
    "ReRankerMetrics",
    "get_reranker",
    "run_shadow_mode_analysis",
    # Thresholds
    "AdaptiveThresholdManager",
    "ThresholdRule",
    "ThresholdSuggestion",
    "get_threshold_manager",
    "generate_threshold_report"
]
