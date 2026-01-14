# =============================================================================
# Retrain Pipeline - Orquestador de Reentrenamiento
# =============================================================================
"""
Pipeline de reentrenamiento del HybridFraudReRanker.

FLUJO:
1. Cargar feedback acumulado desde FeedbackStore
2. Verificar cold start (mínimo 100 muestras, 30 por clase)
3. Entrenar modelo candidato
4. Validar métricas en holdout
5. Registrar en MLflow
6. Promocionar a producción si mejora

EJECUCIÓN:
- Databricks: Programar como Job semanal
- Local: python pipelines/retrain_pipeline.py

USO:
    from pipelines.retrain_pipeline import RetrainPipeline
    
    pipeline = RetrainPipeline()
    result = pipeline.run()
"""

import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

# Añadir path del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from continuous_learning.feature_flags import (
    get_config, get_flags, get_audit_logger,
    ContinuousLearningConfig, FeatureFlags
)
from continuous_learning.feedback_store import (
    FeedbackStorePandas, get_feedback_store
)
from continuous_learning.hybrid_reranker import (
    HybridFraudReRanker, ReRankerMetrics, get_reranker
)
from continuous_learning.adaptive_thresholds import (
    AdaptiveThresholdManager, get_threshold_manager,
    generate_threshold_report
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PIPELINE RESULT
# =============================================================================

class PipelineResult:
    """Resultado de la ejecución del pipeline."""
    
    def __init__(self):
        self.success: bool = False
        self.stage: str = "not_started"
        self.message: str = ""
        self.metrics: Optional[ReRankerMetrics] = None
        self.promoted_to_production: bool = False
        self.threshold_report: Optional[Dict] = None
        self.execution_time_seconds: float = 0.0
        self.timestamp: str = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "stage": self.stage,
            "message": self.message,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "promoted_to_production": self.promoted_to_production,
            "threshold_suggestions_count": len(
                self.threshold_report.get("suggestions", [])
            ) if self.threshold_report else 0,
            "execution_time_seconds": self.execution_time_seconds,
            "timestamp": self.timestamp
        }
    
    def __str__(self) -> str:
        status = "✅" if self.success else "❌"
        return f"{status} Pipeline {self.stage}: {self.message}"


# =============================================================================
# RETRAIN PIPELINE
# =============================================================================

class RetrainPipeline:
    """
    Pipeline de reentrenamiento del sistema de Continuous Learning.
    
    Ejecuta las siguientes etapas:
    1. LOAD: Cargar feedback desde FeedbackStore
    2. VALIDATE: Verificar requisitos de cold start
    3. TRAIN: Entrenar modelo candidato
    4. EVALUATE: Validar métricas
    5. REGISTER: Registrar en MLflow (si disponible)
    6. PROMOTE: Promocionar si mejora
    7. THRESHOLDS: Analizar y sugerir ajustes de umbrales
    """
    
    def __init__(
        self,
        config: ContinuousLearningConfig = None,
        flags: FeatureFlags = None,
        min_precision_for_promotion: float = 0.80,
        auto_promote: bool = False
    ):
        self.config = config or get_config()
        self.flags = flags or get_flags()
        self.audit_logger = get_audit_logger()
        self.min_precision = min_precision_for_promotion
        self.auto_promote = auto_promote
        
        self.feedback_store = get_feedback_store()
        self.reranker = get_reranker()
        self.threshold_manager = get_threshold_manager()
    
    def run(self, dry_run: bool = False) -> PipelineResult:
        """
        Ejecuta el pipeline completo.
        
        Args:
            dry_run: Si True, no aplica cambios (solo validación)
            
        Returns:
            PipelineResult con el resultado de la ejecución
        """
        result = PipelineResult()
        start_time = datetime.now()
        
        try:
            # =================================================================
            # STAGE 1: LOAD FEEDBACK
            # =================================================================
            result.stage = "load_feedback"
            logger.info("=" * 60)
            logger.info("STAGE 1: Cargando feedback...")
            
            feedback_df = self.feedback_store.get_training_data()
            
            if len(feedback_df) == 0:
                result.message = "No hay feedback disponible"
                logger.warning(result.message)
                return result
            
            logger.info(f"Feedback cargado: {len(feedback_df)} registros")
            
            # =================================================================
            # STAGE 2: VALIDATE COLD START
            # =================================================================
            result.stage = "validate_cold_start"
            logger.info("=" * 60)
            logger.info("STAGE 2: Validando requisitos de cold start...")
            
            is_ready, reason = self.feedback_store.is_ready_for_training()
            
            if not is_ready:
                result.message = f"Cold start: {reason}"
                logger.warning(result.message)
                
                # Aunque no podamos entrenar, generar reporte de umbrales
                result.stage = "cold_start_fallback"
                result.threshold_report = generate_threshold_report(
                    feedback_df, window_days=30
                )
                return result
            
            logger.info(f"Validación OK: {reason}")
            
            # =================================================================
            # STAGE 3: PREPARE TRAINING DATA
            # =================================================================
            result.stage = "prepare_data"
            logger.info("=" * 60)
            logger.info("STAGE 3: Preparando datos de entrenamiento...")
            
            # Enriquecer con features si es necesario
            training_df = self._prepare_training_data(feedback_df)
            
            logger.info(f"Datos preparados: {len(training_df)} registros, {len(training_df.columns)} columnas")
            
            # =================================================================
            # STAGE 4: TRAIN MODEL
            # =================================================================
            result.stage = "train_model"
            logger.info("=" * 60)
            logger.info("STAGE 4: Entrenando modelo ReRanker...")
            
            if dry_run:
                logger.info("[DRY RUN] Saltando entrenamiento real")
                result.message = "Dry run completado"
                result.success = True
                return result
            
            metrics = self.reranker.fit(training_df)
            result.metrics = metrics
            
            logger.info(f"Entrenamiento completado:")
            logger.info(f"  - Precision: {metrics.precision:.3f}")
            logger.info(f"  - Recall: {metrics.recall:.3f}")
            logger.info(f"  - F1: {metrics.f1:.3f}")
            logger.info(f"  - AUC-ROC: {metrics.auc_roc:.3f}")
            
            # =================================================================
            # STAGE 5: EVALUATE
            # =================================================================
            result.stage = "evaluate"
            logger.info("=" * 60)
            logger.info("STAGE 5: Evaluando métricas...")
            
            if not metrics.is_acceptable(self.min_precision):
                result.message = (
                    f"Métricas insuficientes: precision={metrics.precision:.3f} "
                    f"< min={self.min_precision:.3f}"
                )
                logger.warning(result.message)
                # Modelo guardado pero no promocionado
                result.success = True
                return result
            
            logger.info("Métricas aceptables para producción")
            
            # =================================================================
            # STAGE 6: REGISTER TO MLFLOW
            # =================================================================
            result.stage = "register_mlflow"
            logger.info("=" * 60)
            logger.info("STAGE 6: Registrando en MLflow...")
            
            model_uri = self.reranker.register_to_mlflow()
            
            if model_uri:
                logger.info(f"Modelo registrado: {model_uri}")
            else:
                logger.info("MLflow no disponible, modelo guardado localmente")
            
            # =================================================================
            # STAGE 7: PROMOTE TO PRODUCTION
            # =================================================================
            result.stage = "promote"
            logger.info("=" * 60)
            logger.info("STAGE 7: Evaluando promoción a producción...")
            
            if self.auto_promote:
                # Activar shadow mode si no está activo
                if not self.flags.enable_reranker_shadow_mode:
                    self.flags.activate_shadow_mode()
                    logger.info("Shadow mode activado")
                
                # Intentar promocionar si cumple requisitos
                if self.flags.is_reranker_ready(self.config):
                    if self.flags.promote_to_production(self.config):
                        result.promoted_to_production = True
                        logger.info("🚀 Modelo PROMOCIONADO a producción")
                    else:
                        logger.info("Promoción no aplicada (requisitos no cumplidos)")
                else:
                    logger.info("Modelo en shadow mode (esperando período de validación)")
            else:
                logger.info("Auto-promote deshabilitado. Revisar manualmente.")
            
            # =================================================================
            # STAGE 8: THRESHOLD ANALYSIS
            # =================================================================
            result.stage = "threshold_analysis"
            logger.info("=" * 60)
            logger.info("STAGE 8: Analizando umbrales adaptativos...")
            
            result.threshold_report = generate_threshold_report(
                feedback_df, window_days=7
            )
            
            n_suggestions = len(result.threshold_report.get("suggestions", []))
            logger.info(f"Sugerencias de umbral generadas: {n_suggestions}")
            
            for suggestion in result.threshold_report.get("suggestions", []):
                logger.info(
                    f"  - {suggestion['rule_name']}: "
                    f"{suggestion['current_value']:.3f} → {suggestion['suggested_value']:.3f} "
                    f"(FP rate: {suggestion['fp_rate']:.1%})"
                )
            
            # =================================================================
            # COMPLETADO
            # =================================================================
            result.stage = "completed"
            result.success = True
            result.message = "Pipeline completado exitosamente"
            
            logger.info("=" * 60)
            logger.info("🎉 PIPELINE COMPLETADO EXITOSAMENTE")
            
        except Exception as e:
            result.message = f"Error en stage {result.stage}: {str(e)}"
            logger.exception(result.message)
        
        finally:
            result.execution_time_seconds = (datetime.now() - start_time).total_seconds()
            
            # Audit log
            self.audit_logger.log_event(
                event_type="pipeline_execution",
                details=result.to_dict()
            )
        
        return result
    
    def _prepare_training_data(self, feedback_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara datos de entrenamiento enriqueciendo con features simuladas
        si no están disponibles.
        
        En producción, estos datos vendrían del snapshot de features
        guardado en FeedbackStore.
        """
        df = feedback_df.copy()
        
        # Lista de features esperadas
        expected_features = [
            'fraud_score_normalized', 'cobertura_ventas', 'accruals_ratio',
            'tasa_interes_implicita', 'productividad_laboral', 
            'pct_numeros_redondos', 'benford_divergence', 'mahalanobis_distance',
            'flag_incoherencia_logistica', 'flag_hidden_debt',
            'flag_empresa_pantalla', 'flag_numeros_redondos',
            'flag_benford_anomaly', 'flag_sectoral_outlier'
        ]
        
        # Si faltan features, simular (solo para demo/testing)
        for feature in expected_features:
            if feature not in df.columns:
                if feature.startswith('flag_'):
                    # Binary flags
                    df[feature] = np.random.binomial(1, 0.3, len(df))
                elif feature == 'fraud_score_normalized':
                    # Score normalizado según veredicto (con ruido)
                    df[feature] = df['analyst_verdict'] * 0.8 + np.random.uniform(0, 0.2, len(df))
                else:
                    # Features numéricas continuas
                    df[feature] = np.random.uniform(0, 1, len(df))
        
        return df


# =============================================================================
# CLI
# =============================================================================

def main():
    """Punto de entrada para ejecución desde línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Pipeline de reentrenamiento del sistema de Continuous Learning"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ejecutar sin aplicar cambios"
    )
    
    parser.add_argument(
        "--auto-promote",
        action="store_true",
        help="Promocionar automáticamente si métricas son aceptables"
    )
    
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.80,
        help="Precisión mínima para promoción (default: 0.80)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("RETRAIN PIPELINE - CONTINUOUS LEARNING")
    logger.info("=" * 60)
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Auto promote: {args.auto_promote}")
    logger.info(f"Min precision: {args.min_precision}")
    
    pipeline = RetrainPipeline(
        min_precision_for_promotion=args.min_precision,
        auto_promote=args.auto_promote
    )
    
    result = pipeline.run(dry_run=args.dry_run)
    
    print("\n" + "=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)
    print(result)
    
    if result.metrics:
        print(f"\nMétricas:")
        print(f"  Precision: {result.metrics.precision:.3f}")
        print(f"  Recall: {result.metrics.recall:.3f}")
        print(f"  F1: {result.metrics.f1:.3f}")
        print(f"  AUC-ROC: {result.metrics.auc_roc:.3f}")
    
    if result.threshold_report:
        n_sugg = len(result.threshold_report.get("suggestions", []))
        print(f"\nSugerencias de umbral: {n_sugg}")
    
    print(f"\nTiempo de ejecución: {result.execution_time_seconds:.1f}s")
    
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
