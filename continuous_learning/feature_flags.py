# =============================================================================
# Feature Flags & Configuration for Continuous Learning
# =============================================================================
"""
Configuración centralizada y feature flags para el módulo de Continuous Learning.
Permite activar/desactivar funcionalidades de forma segura sin cambiar código.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class ContinuousLearningConfig:
    """
    Configuración principal del módulo de Continuous Learning.
    
    GUARDRAILS DE SEGURIDAD:
    - Todos los valores tienen límites mínimos/máximos
    - Cambios quedan registrados para auditoría
    """
    
    # ==========================================================================
    # FEEDBACK STORE
    # ==========================================================================
    feedback_table_path: str = "delta/gold_fraud_feedback"
    feedback_retention_days: int = 365  # Retención de datos de feedback
    
    # ==========================================================================
    # COLD START THRESHOLDS
    # ==========================================================================
    min_samples_for_training: int = 100  # Mínimo feedback total
    min_samples_per_class: int = 30  # Mínimo por clase (FP y TP)
    
    # ==========================================================================
    # RERANKER CONFIGURATION
    # ==========================================================================
    reranker_confidence_threshold: float = 0.95  # P(rechazo) para suprimir
    max_suppression_rate: float = 0.20  # Máximo % de alertas suprimibles
    shadow_mode_days: int = 14  # Días obligatorios en shadow mode
    
    # ==========================================================================
    # ADAPTIVE THRESHOLDS
    # ==========================================================================
    fp_rate_alert_threshold: float = 0.80  # Tasa FP para sugerir cambio
    max_threshold_relaxation: float = 0.50  # Máximo relajar umbral (50%)
    threshold_change_cooldown_days: int = 7  # Días entre cambios de umbral
    
    # ==========================================================================
    # MLFLOW TRACKING
    # ==========================================================================
    mlflow_experiment_name: str = "FraudHunter_ContinuousLearning"
    model_registry_name: str = "FraudReRanker"
    
    # ==========================================================================
    # GUARDRAILS (LÍMITES DUROS - NO MODIFICAR SIN AUDITORÍA)
    # ==========================================================================
    HARD_MIN_SAMPLES: int = field(default=50, init=False)  # Absoluto mínimo
    HARD_MAX_SUPPRESSION: float = field(default=0.30, init=False)  # Absoluto máximo
    HARD_MIN_SHADOW_DAYS: int = field(default=7, init=False)  # Mínimo shadow
    
    def __post_init__(self):
        """Valida que la configuración respete los guardrails."""
        if self.min_samples_for_training < self.HARD_MIN_SAMPLES:
            raise ValueError(
                f"min_samples_for_training ({self.min_samples_for_training}) "
                f"no puede ser menor que HARD_MIN_SAMPLES ({self.HARD_MIN_SAMPLES})"
            )
        
        if self.max_suppression_rate > self.HARD_MAX_SUPPRESSION:
            raise ValueError(
                f"max_suppression_rate ({self.max_suppression_rate}) "
                f"no puede exceder HARD_MAX_SUPPRESSION ({self.HARD_MAX_SUPPRESSION})"
            )
        
        if self.shadow_mode_days < self.HARD_MIN_SHADOW_DAYS:
            raise ValueError(
                f"shadow_mode_days ({self.shadow_mode_days}) "
                f"no puede ser menor que HARD_MIN_SHADOW_DAYS ({self.HARD_MIN_SHADOW_DAYS})"
            )
    
    def to_dict(self) -> Dict:
        """Exporta configuración como diccionario para logging."""
        return {
            "feedback_table_path": self.feedback_table_path,
            "min_samples_for_training": self.min_samples_for_training,
            "min_samples_per_class": self.min_samples_per_class,
            "reranker_confidence_threshold": self.reranker_confidence_threshold,
            "max_suppression_rate": self.max_suppression_rate,
            "shadow_mode_days": self.shadow_mode_days,
            "fp_rate_alert_threshold": self.fp_rate_alert_threshold,
            "max_threshold_relaxation": self.max_threshold_relaxation
        }


@dataclass
class FeatureFlags:
    """
    Feature flags para activación gradual de funcionalidades.
    
    IMPORTANTE: Cambiar estos flags solo después de validación en shadow mode.
    """
    
    # ==========================================================================
    # MÓDULO FEEDBACK STORE
    # ==========================================================================
    enable_feedback_collection: bool = True  # Siempre activo en Fase 1
    
    # ==========================================================================
    # MÓDULO RERANKER
    # ==========================================================================
    enable_reranker_shadow_mode: bool = False  # Calcula pero no aplica
    enable_reranker_production: bool = False  # Aplica re-ranking
    reranker_traffic_percentage: float = 0.0  # % tráfico afectado (0-100)
    
    # ==========================================================================
    # MÓDULO ADAPTIVE THRESHOLDS
    # ==========================================================================
    enable_threshold_suggestions: bool = False  # Solo sugerencias
    enable_auto_threshold_adjustment: bool = False  # Ajuste automático
    
    # ==========================================================================
    # AUDITORÍA
    # ==========================================================================
    enable_detailed_logging: bool = True
    log_all_reranking_decisions: bool = True
    
    # ==========================================================================
    # METADATOS
    # ==========================================================================
    shadow_mode_start_date: Optional[str] = None
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_by: str = "system"
    
    def is_reranker_ready(self, config: ContinuousLearningConfig) -> bool:
        """
        Verifica si el ReRanker puede activarse en producción.
        
        Condiciones:
        1. Shadow mode estuvo activo el tiempo requerido
        2. Flag de producción está habilitado
        """
        if not self.enable_reranker_production:
            return False
        
        if not self.shadow_mode_start_date:
            logger.warning("ReRanker: No hay fecha de inicio de shadow mode registrada")
            return False
        
        shadow_start = datetime.fromisoformat(self.shadow_mode_start_date)
        days_in_shadow = (datetime.now() - shadow_start).days
        
        if days_in_shadow < config.shadow_mode_days:
            logger.warning(
                f"ReRanker: Solo {days_in_shadow} días en shadow mode "
                f"(mínimo requerido: {config.shadow_mode_days})"
            )
            return False
        
        return True
    
    def activate_shadow_mode(self) -> None:
        """Activa el shadow mode y registra la fecha de inicio."""
        if self.shadow_mode_start_date is None:
            self.shadow_mode_start_date = datetime.now().isoformat()
            self.enable_reranker_shadow_mode = True
            self.last_updated = datetime.now().isoformat()
            logger.info(f"Shadow mode activado: {self.shadow_mode_start_date}")
    
    def promote_to_production(self, config: ContinuousLearningConfig) -> bool:
        """
        Intenta promocionar ReRanker a producción.
        
        Returns:
            True si la promoción fue exitosa, False si no cumple requisitos.
        """
        if not self.is_reranker_ready(config):
            return False
        
        self.enable_reranker_production = True
        self.reranker_traffic_percentage = 10.0  # Empezar con 10%
        self.last_updated = datetime.now().isoformat()
        logger.info("ReRanker promocionado a producción (10% tráfico)")
        return True
    
    def to_dict(self) -> Dict:
        """Exporta flags como diccionario."""
        return {
            "enable_feedback_collection": self.enable_feedback_collection,
            "enable_reranker_shadow_mode": self.enable_reranker_shadow_mode,
            "enable_reranker_production": self.enable_reranker_production,
            "reranker_traffic_percentage": self.reranker_traffic_percentage,
            "enable_threshold_suggestions": self.enable_threshold_suggestions,
            "enable_auto_threshold_adjustment": self.enable_auto_threshold_adjustment,
            "shadow_mode_start_date": self.shadow_mode_start_date,
            "last_updated": self.last_updated,
            "updated_by": self.updated_by
        }
    
    def save(self, path: str = "continuous_learning/feature_flags.json") -> None:
        """Persiste los flags a disco."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Feature flags guardados en {path}")
    
    @classmethod
    def load(cls, path: str = "continuous_learning/feature_flags.json") -> "FeatureFlags":
        """Carga flags desde disco."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        return cls()


# =============================================================================
# AUDIT LOG
# =============================================================================

class AuditLogger:
    """
    Logger de auditoría para cambios en configuración y decisiones del sistema.
    
    CRÍTICO: Todos los cambios de configuración y decisiones de supresión
    deben quedar registrados para cumplimiento regulatorio.
    """
    
    def __init__(self, log_path: str = "continuous_learning/audit_log.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    def log_event(
        self,
        event_type: str,
        details: Dict,
        user: str = "system"
    ) -> None:
        """
        Registra un evento de auditoría.
        
        Args:
            event_type: Tipo de evento (config_change, suppression, threshold_change, etc.)
            details: Diccionario con detalles del evento
            user: Usuario que realizó la acción
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user": user,
            "details": details
        }
        
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(event) + "\n")
        
        logger.info(f"Audit event: {event_type} by {user}")
    
    def log_suppression_decision(
        self,
        nif: str,
        original_score: float,
        adjusted_score: float,
        reranker_proba: float,
        was_suppressed: bool
    ) -> None:
        """Registra una decisión de supresión de alerta."""
        self.log_event(
            event_type="suppression_decision",
            details={
                "nif": nif,
                "original_score": original_score,
                "adjusted_score": adjusted_score,
                "reranker_proba": reranker_proba,
                "was_suppressed": was_suppressed
            }
        )
    
    def log_threshold_change(
        self,
        rule_name: str,
        old_value: float,
        new_value: float,
        reason: str,
        sector: Optional[str] = None
    ) -> None:
        """Registra un cambio de umbral."""
        self.log_event(
            event_type="threshold_change",
            details={
                "rule_name": rule_name,
                "old_value": old_value,
                "new_value": new_value,
                "reason": reason,
                "sector": sector
            }
        )
    
    def log_model_deployment(
        self,
        model_version: str,
        metrics: Dict,
        promoted_to_production: bool
    ) -> None:
        """Registra un despliegue de modelo."""
        self.log_event(
            event_type="model_deployment",
            details={
                "model_version": model_version,
                "metrics": metrics,
                "promoted_to_production": promoted_to_production
            }
        )


# Instancias globales (singleton pattern)
_config: Optional[ContinuousLearningConfig] = None
_flags: Optional[FeatureFlags] = None
_audit_logger: Optional[AuditLogger] = None


def get_config() -> ContinuousLearningConfig:
    """Obtiene la configuración global (singleton)."""
    global _config
    if _config is None:
        _config = ContinuousLearningConfig()
    return _config


def get_flags() -> FeatureFlags:
    """Obtiene los feature flags globales (singleton)."""
    global _flags
    if _flags is None:
        _flags = FeatureFlags.load()
    return _flags


def get_audit_logger() -> AuditLogger:
    """Obtiene el logger de auditoría global (singleton)."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
