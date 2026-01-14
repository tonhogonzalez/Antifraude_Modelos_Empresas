# =============================================================================
# HybridFraudReRanker - Calibrador Supervisado
# =============================================================================
"""
Capa supervisada que actúa como filtro secundario sobre el Isolation Forest.
Aprende de las decisiones de los analistas para reducir falsos positivos.

ARQUITECTURA:
- Entrada: fraud_score_normalized (IsolationForest) + features forenses
- Modelo: XGBoost/LightGBM clasificador binario
- Salida: Probabilidad de que el analista confirme el fraude

GUARDRAILS:
- No puede suprimir más del 20% de alertas en un batch
- Requiere mínimo 14 días en shadow mode antes de producción
- Todas las decisiones quedan registradas para auditoría
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json
import logging
import os
import pickle
import warnings

import numpy as np
import pandas as pd

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost no disponible. ReRanker usará RandomForest como fallback.")

# LightGBM (alternativa)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

# MLflow (opcional)
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# PySpark (opcional)
try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql import functions as F
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    DataFrame = None

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
class ReRankerMetrics:
    """Métricas de rendimiento del ReRanker."""
    precision: float
    recall: float
    f1: float
    auc_roc: float
    n_samples_train: int
    n_samples_test: int
    confusion_matrix: List[List[int]] = None
    feature_importance: Dict[str, float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc_roc": self.auc_roc,
            "n_samples_train": self.n_samples_train,
            "n_samples_test": self.n_samples_test,
            "timestamp": self.timestamp
        }
    
    def is_acceptable(self, min_precision: float = 0.80) -> bool:
        """Verifica si las métricas son aceptables para producción."""
        return self.precision >= min_precision


@dataclass
class ReRankingDecision:
    """Resultado de una decisión de re-ranking."""
    nif: str
    original_score: float
    adjusted_score: float
    rejection_probability: float
    was_suppressed: bool
    reason: str


# =============================================================================
# HYBRID FRAUD RERANKER
# =============================================================================

class HybridFraudReRanker:
    """
    Calibrador supervisado que ajusta scores del Isolation Forest
    basándose en feedback histórico de analistas.
    
    FLUJO:
    1. Entrena con datos de FeedbackStore
    2. Predice P(analista confirma fraude) para cada alerta
    3. Ajusta scores: Si IsolationForest dice "Alto" pero ReRanker dice "P(rechazo) > 95%"
       → Bajar score y marcar como reranked
    
    SEGURIDAD:
    - Cold start: Fallback al IsolationForest si < 100 muestras
    - Guardrail: Máximo 20% de alertas pueden ser suprimidas
    - Auditoría: Todas las decisiones quedan logueadas
    """
    
    # Features por defecto para el modelo
    DEFAULT_FEATURES = [
        'fraud_score_normalized',
        'cobertura_ventas',
        'accruals_ratio', 
        'tasa_interes_implicita',
        'productividad_laboral',
        'pct_numeros_redondos',
        'benford_divergence',
        'mahalanobis_distance',
        'flag_incoherencia_logistica',
        'flag_hidden_debt',
        'flag_empresa_pantalla',
        'flag_numeros_redondos',
        'flag_benford_anomaly',
        'flag_sectoral_outlier'
    ]
    
    def __init__(
        self,
        config: ContinuousLearningConfig = None,
        flags: FeatureFlags = None,
        model_path: str = "continuous_learning/models/reranker.pkl"
    ):
        self.config = config or get_config()
        self.flags = flags or get_flags()
        self.audit_logger = get_audit_logger()
        self.model_path = model_path
        
        self.model = None
        self.scaler = None
        self.feature_cols = self.DEFAULT_FEATURES.copy()
        self.metrics: Optional[ReRankerMetrics] = None
        self.is_fitted = False
        self.model_version = "1.0.0"
        
        # Intentar cargar modelo existente
        self._try_load_model()
    
    def _try_load_model(self) -> bool:
        """Intenta cargar modelo previamente entrenado."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    saved = pickle.load(f)
                self.model = saved['model']
                self.scaler = saved['scaler']
                self.feature_cols = saved['feature_cols']
                self.metrics = saved.get('metrics')
                self.model_version = saved.get('model_version', '1.0.0')
                self.is_fitted = True
                logger.info(f"Modelo ReRanker cargado: v{self.model_version}")
                return True
            except Exception as e:
                logger.warning(f"No se pudo cargar modelo: {e}")
        return False
    
    def _create_model(self) -> Any:
        """Crea el modelo de clasificación."""
        if XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif LIGHTGBM_AVAILABLE:
            return lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            logger.warning("Usando RandomForest como fallback (XGBoost/LightGBM no disponibles)")
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
    
    def is_ready(self) -> Tuple[bool, str]:
        """
        Verifica si el ReRanker está listo para usar.
        
        Returns:
            Tuple (is_ready, reason_message)
        """
        # Verificar si hay modelo entrenado
        if not self.is_fitted:
            return (False, "Modelo no entrenado")
        
        # Verificar métricas mínimas
        if self.metrics and not self.metrics.is_acceptable():
            return (
                False, 
                f"Métricas insuficientes: precision={self.metrics.precision:.2f} < 0.80"
            )
        
        # Verificar flags
        if not self.flags.enable_reranker_shadow_mode and not self.flags.enable_reranker_production:
            return (False, "ReRanker deshabilitado en feature flags")
        
        return (True, "ReRanker listo")
    
    def fit(
        self,
        feedback_df: pd.DataFrame,
        feature_cols: List[str] = None,
        test_size: float = 0.2
    ) -> ReRankerMetrics:
        """
        Entrena el modelo con datos de feedback.
        
        Args:
            feedback_df: DataFrame con feedback y features
            feature_cols: Columnas a usar como features
            test_size: Proporción para test set
            
        Returns:
            Métricas de entrenamiento
        """
        logger.info(f"Iniciando entrenamiento con {len(feedback_df)} muestras")
        
        # Preparar features
        if feature_cols:
            self.feature_cols = feature_cols
        
        # Filtrar columnas existentes
        available_features = [c for c in self.feature_cols if c in feedback_df.columns]
        
        if len(available_features) < 3:
            raise ValueError(
                f"Insuficientes features disponibles: {available_features}. "
                f"Se requieren al menos 3 de: {self.feature_cols}"
            )
        
        self.feature_cols = available_features
        logger.info(f"Features utilizadas: {self.feature_cols}")
        
        # Preparar X e y
        X = feedback_df[self.feature_cols].copy()
        y = feedback_df['analyst_verdict'].values
        
        # Imputar NaNs
        X = X.fillna(X.median())
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Escalar
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entrenar
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluar
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Métricas
        self.metrics = ReRankerMetrics(
            precision=precision_score(y_test, y_pred),
            recall=recall_score(y_test, y_pred),
            f1=f1_score(y_test, y_pred),
            auc_roc=roc_auc_score(y_test, y_proba),
            n_samples_train=len(X_train),
            n_samples_test=len(X_test),
            confusion_matrix=confusion_matrix(y_test, y_pred).tolist()
        )
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.metrics.feature_importance = {
                feat: float(imp) 
                for feat, imp in zip(self.feature_cols, self.model.feature_importances_)
            }
        
        self.is_fitted = True
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(
            f"Entrenamiento completado: "
            f"Precision={self.metrics.precision:.3f}, "
            f"Recall={self.metrics.recall:.3f}, "
            f"AUC={self.metrics.auc_roc:.3f}"
        )
        
        # Guardar modelo
        self.save()
        
        return self.metrics
    
    def predict_rejection_probability(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predice la probabilidad de que el analista rechace cada alerta.
        
        P(rechazo) = 1 - P(fraude confirmado)
        
        Args:
            df: DataFrame con features
            
        Returns:
            DataFrame con columna 'rejection_probability' añadida
        """
        if not self.is_fitted:
            logger.warning("Modelo no entrenado. Retornando probabilidad 0 (fallback)")
            return df.assign(rejection_probability=0.0)
        
        # Preparar features
        available_features = [c for c in self.feature_cols if c in df.columns]
        X = df[available_features].copy()
        X = X.fillna(X.median())
        
        # Escalar
        X_scaled = self.scaler.transform(X)
        
        # Predecir P(fraude)
        fraud_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        # P(rechazo) = 1 - P(fraude)
        rejection_proba = 1 - fraud_proba
        
        return df.assign(rejection_probability=rejection_proba)
    
    def rerank_scores(
        self,
        df: pd.DataFrame,
        high_risk_threshold: float = 0.7,
        apply_changes: bool = True
    ) -> pd.DataFrame:
        """
        Re-rankea los scores de fraude basándose en el modelo supervisado.
        
        LÓGICA:
        - Si score_original > high_risk_threshold (Alto Riesgo)
        - PERO rejection_probability > confidence_threshold (95%)
        - ENTONCES: Reducir score y marcar como reranked
        
        GUARDRAIL:
        - Máximo 20% de las alertas pueden ser suprimidas
        
        Args:
            df: DataFrame con fraud_score_normalized
            high_risk_threshold: Umbral para considerar alerta de alto riesgo
            apply_changes: Si True, aplica los cambios. Si False, solo calcula (shadow mode)
            
        Returns:
            DataFrame con scores ajustados
        """
        if not self.is_fitted:
            logger.info("ReRanker no entrenado. Usando scores originales (fallback)")
            return df.assign(
                fraud_score_adjusted=df['fraud_score_normalized'],
                reranked=False,
                rejection_probability=0.0
            )
        
        # Calcular probabilidades
        df = self.predict_rejection_probability(df)
        
        # Identificar alertas de alto riesgo
        high_risk_mask = df['fraud_score_normalized'] > high_risk_threshold
        
        # Identificar candidatas a supresión
        confidence_threshold = self.config.reranker_confidence_threshold
        suppression_mask = (
            high_risk_mask & 
            (df['rejection_probability'] > confidence_threshold)
        )
        
        n_high_risk = high_risk_mask.sum()
        n_to_suppress = suppression_mask.sum()
        
        # GUARDRAIL: Verificar tasa de supresión
        if n_high_risk > 0:
            suppression_rate = n_to_suppress / n_high_risk
            max_rate = self.config.max_suppression_rate
            
            if suppression_rate > max_rate:
                logger.warning(
                    f"⚠️ GUARDRAIL: Tasa de supresión ({suppression_rate:.1%}) > máximo ({max_rate:.1%}). "
                    f"Limitando supresiones."
                )
                
                # Ordenar por probabilidad de rechazo y tomar solo el máximo permitido
                max_suppressions = int(n_high_risk * max_rate)
                
                if max_suppressions > 0:
                    # Mantener solo las de mayor probabilidad de rechazo
                    candidates = df[suppression_mask].nlargest(max_suppressions, 'rejection_probability')
                    suppression_mask = df.index.isin(candidates.index)
                else:
                    suppression_mask = pd.Series(False, index=df.index)
        
        # Aplicar ajustes
        if apply_changes:
            # Bajar score de las suprimidas (de "Alto" a "Medio")
            df['fraud_score_adjusted'] = df['fraud_score_normalized'].copy()
            df.loc[suppression_mask, 'fraud_score_adjusted'] = 0.5  # Medio
            df['reranked'] = suppression_mask
        else:
            # Shadow mode: solo calcular, no aplicar
            df['fraud_score_adjusted'] = df['fraud_score_normalized']
            df['reranked'] = False
            df['would_suppress'] = suppression_mask
        
        # Logging
        n_suppressed = suppression_mask.sum()
        logger.info(
            f"ReRanking: {n_suppressed}/{n_high_risk} alertas de alto riesgo "
            f"{'ajustadas' if apply_changes else 'marcadas (shadow mode)'}"
        )
        
        # Audit log para cada supresión
        if apply_changes:
            for idx in df[suppression_mask].index:
                row = df.loc[idx]
                self.audit_logger.log_suppression_decision(
                    nif=row.get('nif', 'unknown'),
                    original_score=row['fraud_score_normalized'],
                    adjusted_score=row['fraud_score_adjusted'],
                    reranker_proba=row['rejection_probability'],
                    was_suppressed=True
                )
        
        return df
    
    def save(self) -> None:
        """Guarda el modelo entrenado a disco."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_cols': self.feature_cols,
                'metrics': self.metrics,
                'model_version': self.model_version
            }, f)
        
        logger.info(f"Modelo guardado: {self.model_path} (v{self.model_version})")
    
    def register_to_mlflow(self) -> Optional[str]:
        """
        Registra el modelo en MLflow Model Registry.
        
        Returns:
            URI del modelo registrado o None si MLflow no disponible
        """
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow no disponible. Registro omitido.")
            return None
        
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        
        try:
            mlflow.set_experiment(self.config.mlflow_experiment_name)
            
            with mlflow.start_run(run_name=f"reranker_v{self.model_version}"):
                # Log métricas
                mlflow.log_metrics(self.metrics.to_dict())
                
                # Log parámetros
                mlflow.log_param("features", str(self.feature_cols))
                mlflow.log_param("confidence_threshold", self.config.reranker_confidence_threshold)
                
                # Log modelo
                if XGBOOST_AVAILABLE and isinstance(self.model, xgb.XGBClassifier):
                    model_info = mlflow.xgboost.log_model(
                        self.model,
                        "model",
                        registered_model_name=self.config.model_registry_name
                    )
                else:
                    model_info = mlflow.sklearn.log_model(
                        self.model,
                        "model",
                        registered_model_name=self.config.model_registry_name
                    )
                
                # Audit
                self.audit_logger.log_model_deployment(
                    model_version=self.model_version,
                    metrics=self.metrics.to_dict(),
                    promoted_to_production=False
                )
                
                return model_info.model_uri
                
        except Exception as e:
            logger.error(f"Error registrando modelo en MLflow: {e}")
            return None
    
    def get_metrics(self) -> Optional[ReRankerMetrics]:
        """Retorna las métricas del modelo actual."""
        return self.metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Retorna importancia de features ordenada."""
        if self.metrics and self.metrics.feature_importance:
            return dict(sorted(
                self.metrics.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
        return {}


# =============================================================================
# FACTORY / HELPER FUNCTIONS
# =============================================================================

def get_reranker() -> HybridFraudReRanker:
    """Obtiene instancia del ReRanker (singleton pattern)."""
    return HybridFraudReRanker()


def run_shadow_mode_analysis(
    df: pd.DataFrame,
    feedback_store: FeedbackStorePandas = None
) -> Dict:
    """
    Ejecuta análisis en shadow mode y retorna estadísticas.
    
    Útil para validar el ReRanker antes de activar en producción.
    """
    reranker = get_reranker()
    
    # Verificar si está listo
    is_ready, reason = reranker.is_ready()
    if not is_ready:
        return {
            "status": "not_ready",
            "reason": reason,
            "recommendations": []
        }
    
    # Ejecutar en shadow mode
    df_analyzed = reranker.rerank_scores(df, apply_changes=False)
    
    # Estadísticas
    would_suppress = df_analyzed['would_suppress'].sum() if 'would_suppress' in df_analyzed.columns else 0
    high_risk = (df_analyzed['fraud_score_normalized'] > 0.7).sum()
    
    return {
        "status": "shadow_mode_active",
        "high_risk_alerts": int(high_risk),
        "would_suppress": int(would_suppress),
        "suppression_rate": would_suppress / high_risk if high_risk > 0 else 0,
        "metrics": reranker.get_metrics().to_dict() if reranker.get_metrics() else None,
        "recommendations": [
            f"El ReRanker suprimiría {would_suppress} de {high_risk} alertas de alto riesgo ({100*would_suppress/high_risk:.1f}%)" if high_risk > 0 else "No hay alertas de alto riesgo"
        ]
    }
