# =============================================================================
# AdaptiveThresholdManager - Optimizador Dinámico de Umbrales
# =============================================================================
"""
Sistema para ajustar automáticamente los umbrales de las reglas de detección
basándose en la tasa de falsos positivos histórica.

VALOR: Permite que las reglas se adapten automáticamente a los cambios en
patrones de negocio y reduce la carga de trabajo de los analistas.

SEGURIDAD:
- Los umbrales tienen límites mínimos/máximos (guardrails)
- Cambios automáticos son opcionales (modo sugerencia por defecto)
- Todos los cambios quedan auditados
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
import os

import pandas as pd
import numpy as np

from .feature_flags import (
    get_config, get_flags, get_audit_logger,
    ContinuousLearningConfig, FeatureFlags
)
from .feedback_store import FeedbackStorePandas, get_feedback_store

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ThresholdRule:
    """
    Definición de una regla con su umbral configurable.
    """
    name: str  # Nombre del flag (ej: flag_incoherencia_logistica)
    description: str
    current_threshold: float
    default_threshold: float
    min_threshold: float  # Guardrail: Nunca bajar de aquí
    max_threshold: float  # Guardrail: Nunca subir de aquí
    direction: str = "upper"  # "upper" = flag si > umbral, "lower" = flag si < umbral
    
    def can_relax(self, amount: float) -> bool:
        """Verifica si el umbral puede relajarse en la cantidad indicada."""
        if self.direction == "upper":
            new_value = self.current_threshold * (1 + amount)
            return new_value <= self.max_threshold
        else:
            new_value = self.current_threshold * (1 - amount)
            return new_value >= self.min_threshold
    
    def relaxed_value(self, amount: float) -> float:
        """Calcula el nuevo valor relajado."""
        if self.direction == "upper":
            return min(self.current_threshold * (1 + amount), self.max_threshold)
        else:
            return max(self.current_threshold * (1 - amount), self.min_threshold)


@dataclass
class ThresholdSuggestion:
    """
    Sugerencia de cambio de umbral generada por el sistema.
    """
    rule_name: str
    current_value: float
    suggested_value: float
    reason: str
    sector: Optional[str] = None  # Si aplica solo a un sector
    fp_rate: float = 0.0
    sample_size: int = 0
    confidence: str = "medium"  # low, medium, high
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    applied: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "rule_name": self.rule_name,
            "current_value": self.current_value,
            "suggested_value": self.suggested_value,
            "change_pct": (self.suggested_value - self.current_value) / self.current_value * 100,
            "reason": self.reason,
            "sector": self.sector,
            "fp_rate": self.fp_rate,
            "sample_size": self.sample_size,
            "confidence": self.confidence,
            "created_at": self.created_at
        }


# =============================================================================
# DEFAULT RULE DEFINITIONS
# =============================================================================

DEFAULT_THRESHOLD_RULES = {
    "flag_incoherencia_logistica": ThresholdRule(
        name="flag_incoherencia_logistica",
        description="Operaciones UE sin gastos de transporte proporcionales",
        current_threshold=0.01,  # transporte/entregas_ue < 1%
        default_threshold=0.01,
        min_threshold=0.001,  # Nunca menos del 0.1%
        max_threshold=0.10,   # Nunca más del 10%
        direction="lower"
    ),
    "flag_empresa_pantalla": ThresholdRule(
        name="flag_empresa_pantalla",
        description="Ventas altas con gastos de personal muy bajos",
        current_threshold=50.0,  # productividad > 50x
        default_threshold=50.0,
        min_threshold=20.0,
        max_threshold=100.0,
        direction="upper"
    ),
    "flag_hidden_debt": ThresholdRule(
        name="flag_hidden_debt",
        description="Tasa de interés implícita muy alta (deuda oculta)",
        current_threshold=2.0,  # tasa > 2x mercado
        default_threshold=2.0,
        min_threshold=1.5,
        max_threshold=4.0,
        direction="upper"
    ),
    "flag_numeros_redondos": ThresholdRule(
        name="flag_numeros_redondos",
        description="Porcentaje alto de facturas con números redondos",
        current_threshold=0.30,  # > 30% redondos
        default_threshold=0.30,
        min_threshold=0.15,
        max_threshold=0.50,
        direction="upper"
    ),
    "flag_benford_anomaly": ThresholdRule(
        name="flag_benford_anomaly",
        description="Divergencia alta respecto a distribución de Benford",
        current_threshold=0.15,  # KL divergence > 0.15
        default_threshold=0.15,
        min_threshold=0.08,
        max_threshold=0.30,
        direction="upper"
    ),
    "flag_sectoral_outlier": ThresholdRule(
        name="flag_sectoral_outlier",
        description="Distancia Mahalanobis alta respecto al sector",
        current_threshold=3.0,  # > 3 desviaciones
        default_threshold=3.0,
        min_threshold=2.0,
        max_threshold=5.0,
        direction="upper"
    )
}


# =============================================================================
# ADAPTIVE THRESHOLD MANAGER
# =============================================================================

class AdaptiveThresholdManager:
    """
    Gestor de umbrales adaptativos que analiza el feedback de los analistas
    y sugiere (o aplica) ajustes a los umbrales de las reglas.
    
    FLUJO:
    1. Analiza estadísticas de feedback por regla y sector
    2. Si FP rate > 80% para una regla/sector → sugiere relajar umbral
    3. Genera sugerencias que el operador puede aprobar
    4. Opcionalmente, aplica cambios automáticamente (con guardrails)
    
    GUARDRAILS:
    - Cada regla tiene min/max threshold
    - Máximo relajación del 50% respecto al valor original
    - Cooldown de 7 días entre cambios para la misma regla
    """
    
    def __init__(
        self,
        config: ContinuousLearningConfig = None,
        flags: FeatureFlags = None,
        rules: Dict[str, ThresholdRule] = None,
        history_path: str = "continuous_learning/threshold_history.json"
    ):
        self.config = config or get_config()
        self.flags = flags or get_flags()
        self.audit_logger = get_audit_logger()
        self.history_path = history_path
        
        # Cargar reglas (custom o default)
        self.rules = rules or DEFAULT_THRESHOLD_RULES.copy()
        
        # Historial de cambios
        self.change_history: List[Dict] = []
        self._load_history()
    
    def _load_history(self) -> None:
        """Carga historial de cambios desde disco."""
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r') as f:
                    self.change_history = json.load(f)
            except Exception as e:
                logger.warning(f"No se pudo cargar historial: {e}")
                self.change_history = []
    
    def _save_history(self) -> None:
        """Persiste historial de cambios."""
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        with open(self.history_path, 'w') as f:
            json.dump(self.change_history, f, indent=2, default=str)
    
    def analyze_rule_performance(
        self,
        rule_name: str,
        feedback_df: pd.DataFrame,
        window_days: int = 7
    ) -> Dict:
        """
        Analiza el rendimiento de una regla específica.
        
        Args:
            rule_name: Nombre del flag (ej: flag_incoherencia_logistica)
            feedback_df: DataFrame con feedback que incluye 'flags_active'
            window_days: Ventana de tiempo para análisis
            
        Returns:
            Diccionario con estadísticas de rendimiento
        """
        if rule_name not in self.rules:
            return {"error": f"Regla desconocida: {rule_name}"}
        
        # Filtrar registros donde este flag estaba activo
        mask = feedback_df['flags_active'].apply(
            lambda x: rule_name in x if isinstance(x, list) else False
        )
        
        rule_df = feedback_df[mask]
        
        if len(rule_df) == 0:
            return {
                "rule_name": rule_name,
                "total_alerts": 0,
                "fp_rate": 0.0,
                "sample_size": 0,
                "by_sector": {}
            }
        
        # Estadísticas globales
        total = len(rule_df)
        fp_count = (rule_df['analyst_verdict'] == 0).sum()
        fp_rate = fp_count / total
        
        # Estadísticas por sector
        by_sector = {}
        if 'cnae_sector' in rule_df.columns:
            for sector in rule_df['cnae_sector'].dropna().unique():
                sector_df = rule_df[rule_df['cnae_sector'] == sector]
                if len(sector_df) >= 5:  # Mínimo 5 muestras por sector
                    sector_fp = (sector_df['analyst_verdict'] == 0).sum()
                    by_sector[sector] = {
                        "total": len(sector_df),
                        "fp_rate": sector_fp / len(sector_df),
                        "fp_count": int(sector_fp)
                    }
        
        return {
            "rule_name": rule_name,
            "total_alerts": total,
            "false_positives": int(fp_count),
            "confirmed_fraud": int(total - fp_count),
            "fp_rate": fp_rate,
            "sample_size": total,
            "by_sector": by_sector,
            "current_threshold": self.rules[rule_name].current_threshold
        }
    
    def suggest_threshold_changes(
        self,
        feedback_df: pd.DataFrame = None,
        window_days: int = 7
    ) -> List[ThresholdSuggestion]:
        """
        Genera sugerencias de cambio de umbral basándose en el feedback.
        
        Args:
            feedback_df: DataFrame con feedback (si None, carga desde FeedbackStore)
            window_days: Ventana de tiempo para análisis
            
        Returns:
            Lista de sugerencias de cambio
        """
        suggestions = []
        
        # Cargar feedback si no se proporciona
        if feedback_df is None:
            store = get_feedback_store()
            min_date = datetime.now() - timedelta(days=window_days)
            feedback_df = store.get_training_data(min_date=min_date)
        
        if len(feedback_df) == 0:
            logger.info("No hay feedback suficiente para generar sugerencias")
            return suggestions
        
        # Analizar cada regla
        for rule_name, rule in self.rules.items():
            stats = self.analyze_rule_performance(rule_name, feedback_df, window_days)
            
            if stats.get("sample_size", 0) < 10:
                continue  # No suficientes muestras
            
            # Verificar cooldown
            if not self._check_cooldown(rule_name):
                logger.info(f"Regla {rule_name} en cooldown. Saltando.")
                continue
            
            # Analizar FP rate global
            if stats["fp_rate"] > self.config.fp_rate_alert_threshold:
                # Sugerir relajar umbral
                relax_amount = 0.10  # 10% por defecto
                
                if rule.can_relax(relax_amount):
                    suggestions.append(ThresholdSuggestion(
                        rule_name=rule_name,
                        current_value=rule.current_threshold,
                        suggested_value=rule.relaxed_value(relax_amount),
                        reason=f"FP rate global ({stats['fp_rate']:.1%}) > umbral ({self.config.fp_rate_alert_threshold:.1%})",
                        fp_rate=stats["fp_rate"],
                        sample_size=stats["sample_size"],
                        confidence="high" if stats["sample_size"] > 50 else "medium"
                    ))
            
            # Analizar FP rate por sector
            for sector, sector_stats in stats.get("by_sector", {}).items():
                if sector_stats["fp_rate"] > self.config.fp_rate_alert_threshold:
                    # Sugerencia específica por sector
                    suggestions.append(ThresholdSuggestion(
                        rule_name=rule_name,
                        current_value=rule.current_threshold,
                        suggested_value=rule.relaxed_value(0.15),  # 15% para sector específico
                        reason=f"FP rate en sector {sector} ({sector_stats['fp_rate']:.1%}) muy alto",
                        sector=sector,
                        fp_rate=sector_stats["fp_rate"],
                        sample_size=sector_stats["total"],
                        confidence="medium" if sector_stats["total"] > 20 else "low"
                    ))
        
        logger.info(f"Generadas {len(suggestions)} sugerencias de cambio de umbral")
        return suggestions
    
    def _check_cooldown(self, rule_name: str) -> bool:
        """Verifica si la regla puede ser modificada (respecto al cooldown)."""
        cooldown_days = self.config.threshold_change_cooldown_days
        cutoff_date = datetime.now() - timedelta(days=cooldown_days)
        
        for change in reversed(self.change_history):
            if change.get("rule_name") == rule_name:
                change_date = datetime.fromisoformat(change.get("timestamp", "1970-01-01"))
                if change_date > cutoff_date:
                    return False
        
        return True
    
    def apply_suggestion(
        self,
        suggestion: ThresholdSuggestion,
        approved_by: str = "system"
    ) -> bool:
        """
        Aplica una sugerencia de cambio de umbral.
        
        Args:
            suggestion: Sugerencia a aplicar
            approved_by: ID del aprobador
            
        Returns:
            True si se aplicó correctamente
        """
        if suggestion.rule_name not in self.rules:
            logger.error(f"Regla desconocida: {suggestion.rule_name}")
            return False
        
        rule = self.rules[suggestion.rule_name]
        old_value = rule.current_threshold
        new_value = suggestion.suggested_value
        
        # Verificar guardrails
        if new_value < rule.min_threshold or new_value > rule.max_threshold:
            logger.error(
                f"Nuevo valor {new_value} fuera de rango "
                f"[{rule.min_threshold}, {rule.max_threshold}]"
            )
            return False
        
        # Verificar máxima relajación
        original = rule.default_threshold
        max_relaxation = self.config.max_threshold_relaxation
        
        if rule.direction == "upper":
            max_allowed = original * (1 + max_relaxation)
            if new_value > max_allowed:
                logger.error(f"Excede máxima relajación permitida: {new_value} > {max_allowed}")
                return False
        else:
            min_allowed = original * (1 - max_relaxation)
            if new_value < min_allowed:
                logger.error(f"Excede máxima relajación permitida: {new_value} < {min_allowed}")
                return False
        
        # Aplicar cambio
        rule.current_threshold = new_value
        suggestion.applied = True
        
        # Registrar en historial
        change_record = {
            "rule_name": rule.name,
            "old_value": old_value,
            "new_value": new_value,
            "reason": suggestion.reason,
            "sector": suggestion.sector,
            "approved_by": approved_by,
            "timestamp": datetime.now().isoformat()
        }
        self.change_history.append(change_record)
        self._save_history()
        
        # Audit log
        self.audit_logger.log_threshold_change(
            rule_name=rule.name,
            old_value=old_value,
            new_value=new_value,
            reason=suggestion.reason,
            sector=suggestion.sector
        )
        
        logger.info(
            f"Umbral actualizado: {rule.name} = {old_value} → {new_value} "
            f"(aprobado por {approved_by})"
        )
        
        return True
    
    def apply_changes_batch(
        self,
        suggestions: List[ThresholdSuggestion],
        approved_by: str = "system",
        auto_approve: bool = False
    ) -> Dict[str, bool]:
        """
        Aplica múltiples sugerencias en batch.
        
        Args:
            suggestions: Lista de sugerencias
            approved_by: ID del aprobador
            auto_approve: Si True, aplica sin verificación adicional
            
        Returns:
            Diccionario {rule_name: success}
        """
        if not auto_approve and not self.flags.enable_auto_threshold_adjustment:
            logger.warning(
                "Ajuste automático deshabilitado. "
                "Habilitar enable_auto_threshold_adjustment o usar apply_suggestion manualmente."
            )
            return {s.rule_name: False for s in suggestions}
        
        results = {}
        for suggestion in suggestions:
            results[suggestion.rule_name] = self.apply_suggestion(suggestion, approved_by)
        
        return results
    
    def get_current_thresholds(self) -> Dict[str, Dict]:
        """Retorna los umbrales actuales de todas las reglas."""
        return {
            name: {
                "current": rule.current_threshold,
                "default": rule.default_threshold,
                "min": rule.min_threshold,
                "max": rule.max_threshold,
                "description": rule.description
            }
            for name, rule in self.rules.items()
        }
    
    def reset_to_defaults(self) -> None:
        """Resetea todos los umbrales a sus valores por defecto."""
        for rule in self.rules.values():
            old_value = rule.current_threshold
            rule.current_threshold = rule.default_threshold
            
            self.audit_logger.log_threshold_change(
                rule_name=rule.name,
                old_value=old_value,
                new_value=rule.default_threshold,
                reason="Reset to defaults"
            )
        
        logger.info("Todos los umbrales reseteados a valores por defecto")
    
    def export_config(self) -> Dict:
        """Exporta configuración actual para uso en FraudHunterConfig."""
        return {
            "benford_threshold": self.rules.get("flag_benford_anomaly", ThresholdRule(
                name="", description="", current_threshold=0.15,
                default_threshold=0.15, min_threshold=0.08, max_threshold=0.30
            )).current_threshold,
            "round_number_threshold": self.rules.get("flag_numeros_redondos", ThresholdRule(
                name="", description="", current_threshold=0.30,
                default_threshold=0.30, min_threshold=0.15, max_threshold=0.50
            )).current_threshold,
            # Añadir otros según sea necesario
        }


# =============================================================================
# FACTORY / HELPER FUNCTIONS
# =============================================================================

def get_threshold_manager() -> AdaptiveThresholdManager:
    """Obtiene instancia del ThresholdManager (singleton pattern)."""
    return AdaptiveThresholdManager()


def generate_threshold_report(
    feedback_df: pd.DataFrame = None,
    window_days: int = 7
) -> Dict:
    """
    Genera un informe completo del estado de los umbrales.
    
    Returns:
        Diccionario con estado actual, sugerencias, y estadísticas
    """
    manager = get_threshold_manager()
    
    # Cargar feedback
    if feedback_df is None:
        store = get_feedback_store()
        min_date = datetime.now() - timedelta(days=window_days)
        feedback_df = store.get_training_data(min_date=min_date)
    
    # Análisis por regla
    rule_stats = {}
    for rule_name in manager.rules:
        rule_stats[rule_name] = manager.analyze_rule_performance(
            rule_name, feedback_df, window_days
        )
    
    # Sugerencias
    suggestions = manager.suggest_threshold_changes(feedback_df, window_days)
    
    return {
        "report_date": datetime.now().isoformat(),
        "window_days": window_days,
        "total_feedback": len(feedback_df),
        "current_thresholds": manager.get_current_thresholds(),
        "rule_statistics": rule_stats,
        "suggestions": [s.to_dict() for s in suggestions],
        "suggestions_count": len(suggestions),
        "auto_adjustment_enabled": manager.flags.enable_auto_threshold_adjustment
    }
