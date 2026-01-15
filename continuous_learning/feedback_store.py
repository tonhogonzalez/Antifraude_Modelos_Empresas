# =============================================================================
# FeedbackStore - Gestión de Feedback de Analistas
# =============================================================================
"""
Módulo para almacenar y gestionar las decisiones de los analistas sobre alertas.
Utiliza Delta Lake para versionado temporal y auditoría.

VALOR: Permite al sistema aprender de las decisiones humanas para reducir
falsos positivos progresivamente.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import hashlib
import json
import logging
import uuid
import os

import pandas as pd
import numpy as np

# PySpark imports (con fallback para testing local)
try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField, StringType, IntegerType, 
        DoubleType, TimestampType, ArrayType
    )
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    DataFrame = None

# Delta Lake (opcional)
try:
    from delta.tables import DeltaTable
    DELTA_AVAILABLE = True
except ImportError:
    DELTA_AVAILABLE = False

from .feature_flags import get_config, get_audit_logger, ContinuousLearningConfig

logger = logging.getLogger(__name__)


# =============================================================================
# SCHEMAS
# =============================================================================

# Definir schema solo si Spark está disponible
if SPARK_AVAILABLE:
    FEEDBACK_SCHEMA = StructType([
        StructField("feedback_id", StringType(), False),
        StructField("nif", StringType(), False),
        StructField("fecha_alerta", TimestampType(), False),
        StructField("fecha_analisis", TimestampType(), False),
        StructField("model_version", StringType(), False),
        StructField("analyst_id", StringType(), True),
        StructField("analyst_verdict", IntegerType(), False),  # 0=FP, 1=Fraude, 2=Watchlist
        StructField("reason_code", StringType(), True),
        StructField("rejection_reason_code", StringType(), True),  # Causa raíz de FP
        StructField("fraud_typology_code", StringType(), True),    # Tipo de fraude
        StructField("analyst_confidence", IntegerType(), True),    # 1-5 confianza
        StructField("fraud_score_original", DoubleType(), False),
        StructField("feature_vector_hash", StringType(), False),
        StructField("cnae_sector", StringType(), True),
        StructField("ventas_netas", DoubleType(), True),
        StructField("flags_active", ArrayType(StringType()), True),
        StructField("created_at", TimestampType(), False),
    ])
else:
    FEEDBACK_SCHEMA = None  # Fallback para entornos sin Spark


# =============================================================================
# CONSTANTES DE CÓDIGOS
# =============================================================================

# Códigos de rechazo para Falsos Positivos
REJECTION_REASON_CODES = {
    'SECTOR_NORMAL': 'Comportamiento normal para el sector',
    'DATA_ERROR': 'Error en los datos de origen',
    'LEGITIMATE_BUSINESS': 'Operación legítima explicada',
    'SEASONAL': 'Patrón estacional (Navidad, verano, etc.)',
    'ONE_TIME': 'Evento puntual no repetible',
    'OTHER': 'Otra razón (ver comentarios)'
}

# Tipologías de fraude
FRAUD_TYPOLOGY_CODES = {
    'CARRUSEL': 'Fraude carrusel IVA',
    'PANTALLA': 'Empresa pantalla / Instrumental',
    'FACTURAS_FALSAS': 'Facturación ficticia',
    'CONTABILIDAD': 'Manipulación contable',
    'DEUDA_OCULTA': 'Ocultación de pasivos',
    'INSOLVENCIA_PUNIBLE': 'Insolvencia punible',
    'OTHER': 'Otro tipo de fraude'
}

# Valores de veredicto
VERDICT_FALSE_POSITIVE = 0
VERDICT_FRAUD = 1
VERDICT_WATCHLIST = 2  # Zona gris


@dataclass
class FeedbackRecord:
    """
    Registro de feedback de un analista sobre una alerta.
    
    analyst_verdict values:
        0 = Falso Positivo
        1 = Fraude Confirmado  
        2 = Watchlist (Zona Gris)
    """
    nif: str
    analyst_verdict: int  # 0=FP, 1=Fraude, 2=Watchlist
    fraud_score_original: float
    feature_vector: Dict[str, float]  # Features al momento de la alerta
    
    # Opcionales
    reason_code: str = None
    rejection_reason_code: str = None  # Obligatorio si verdict=0
    fraud_typology_code: str = None    # Obligatorio si verdict=1
    analyst_confidence: int = 3        # 1-5 (default: medio)
    analyst_id: str = None
    cnae_sector: str = None
    ventas_netas: float = None
    flags_active: List[str] = None
    model_version: str = "1.0.0"
    fecha_alerta: datetime = None
    
    def __post_init__(self):
        if self.fecha_alerta is None:
            self.fecha_alerta = datetime.now()
        if self.flags_active is None:
            self.flags_active = []
        # Validar confianza
        if self.analyst_confidence is not None:
            self.analyst_confidence = max(1, min(5, self.analyst_confidence))
    
    def to_dict(self) -> Dict:
        """Convierte el registro a diccionario para inserción."""
        return {
            "feedback_id": str(uuid.uuid4()),
            "nif": self.nif,
            "fecha_alerta": self.fecha_alerta,
            "fecha_analisis": datetime.now(),
            "model_version": self.model_version,
            "analyst_id": self.analyst_id or "anonymous",
            "analyst_verdict": self.analyst_verdict,
            "reason_code": self.reason_code,
            "rejection_reason_code": self.rejection_reason_code,
            "fraud_typology_code": self.fraud_typology_code,
            "analyst_confidence": self.analyst_confidence,
            "fraud_score_original": self.fraud_score_original,
            "feature_vector_hash": self._hash_features(),
            "cnae_sector": self.cnae_sector,
            "ventas_netas": self.ventas_netas,
            "flags_active": self.flags_active,
            "created_at": datetime.now()
        }
    
    def _hash_features(self) -> str:
        """Genera hash SHA256 del vector de features."""
        feature_str = json.dumps(self.feature_vector, sort_keys=True)
        return hashlib.sha256(feature_str.encode()).hexdigest()


# =============================================================================
# FEEDBACK STORE (PANDAS VERSION - Para uso local/Streamlit)
# =============================================================================

class FeedbackStorePandas:
    """
    Implementación de FeedbackStore usando Pandas y Parquet.
    Para uso en entornos sin Spark (Streamlit local, testing).
    
    Persistencia: Usa ruta absoluta para garantizar que los datos
    se mantengan entre sesiones de Streamlit.
    """
    
    # Columnas del schema
    SCHEMA_COLUMNS = [
        "feedback_id", "nif", "fecha_alerta", "fecha_analisis",
        "model_version", "analyst_id", "analyst_verdict", "reason_code",
        "rejection_reason_code", "fraud_typology_code", "analyst_confidence",
        "fraud_score_original", "feature_vector_hash", "cnae_sector",
        "ventas_netas", "flags_active", "created_at"
    ]
    
    def __init__(
        self,
        storage_path: str = None,
        config: ContinuousLearningConfig = None
    ):
        # Usar ruta absoluta relativa al módulo para persistencia
        if storage_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(module_dir)
            storage_path = os.path.join(project_dir, "data", "feedback_store.parquet")
        
        self.storage_path = storage_path
        self.config = config or get_config()
        self.audit_logger = get_audit_logger()
        self._ensure_storage_exists()
    
    def _ensure_storage_exists(self) -> None:
        """Crea el archivo de storage si no existe."""
        storage_dir = os.path.dirname(self.storage_path)
        if storage_dir:
            os.makedirs(storage_dir, exist_ok=True)
        
        if not os.path.exists(self.storage_path):
            # Crear DataFrame vacío con el schema completo
            empty_df = pd.DataFrame(columns=self.SCHEMA_COLUMNS)
            empty_df.to_parquet(self.storage_path, index=False)
            logger.info(f"Creado FeedbackStore vacío en {self.storage_path}")
        else:
            # Verificar y migrar schema si faltan columnas
            self._migrate_schema_if_needed()
    
    def _migrate_schema_if_needed(self) -> None:
        """Migra el schema añadiendo columnas nuevas si faltan."""
        try:
            df = pd.read_parquet(self.storage_path)
            missing_cols = set(self.SCHEMA_COLUMNS) - set(df.columns)
            
            if missing_cols:
                logger.info(f"Migrando schema: añadiendo columnas {missing_cols}")
                for col in missing_cols:
                    df[col] = None
                df.to_parquet(self.storage_path, index=False)
                logger.info("Migración de schema completada")
        except Exception as e:
            logger.warning(f"No se pudo migrar schema: {e}")
    
    def log_feedback(self, feedback: FeedbackRecord) -> str:
        """
        Registra el feedback de un analista.
        
        Args:
            feedback: Registro de feedback
            
        Returns:
            ID del feedback registrado
        """
        # Validar
        if not self._validate_feedback(feedback):
            raise ValueError("Feedback inválido")
        
        # Cargar datos existentes
        df = pd.read_parquet(self.storage_path)
        
        # Convertir a dict y añadir
        record = feedback.to_dict()
        new_row = pd.DataFrame([record])
        
        # Append
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Guardar
        df.to_parquet(self.storage_path, index=False)
        
        logger.info(f"Feedback registrado: {record['feedback_id']} para NIF {feedback.nif}")
        
        # Audit log
        verdict_labels = {0: "Falso Positivo", 1: "Fraude", 2: "Watchlist"}
        self.audit_logger.log_event(
            event_type="feedback_logged",
            details={
                "feedback_id": record["feedback_id"],
                "nif": feedback.nif,
                "verdict": verdict_labels.get(feedback.analyst_verdict, "Desconocido"),
                "rejection_reason": feedback.rejection_reason_code,
                "fraud_typology": feedback.fraud_typology_code,
                "confidence": feedback.analyst_confidence
            }
        )
        
        return record["feedback_id"]

    def get_last_feedback(self, nif: str) -> Optional[Dict]:
        """
        Obtiene el último feedback registrado para un NIF.
        Retorna None si no hay feedback previo.
        """
        try:
            # Optimización: Leer solo si el archivo existe
            if not os.path.exists(self.storage_path):
                return None
                
            # Leer dataframe (potencial mejora: cachear lectura si es frecuente)
            df = pd.read_parquet(self.storage_path)
            
            if len(df) == 0:
                return None
                
            # Filtrar por NIF
            nif_feedback = df[df['nif'] == nif]
            
            if len(nif_feedback) == 0:
                return None
                
            # Ordenar por fecha (más reciente primero) y tomar el primero
            if 'created_at' in nif_feedback.columns:
                latest = nif_feedback.sort_values('created_at', ascending=False).iloc[0]
            else:
                # Fallback si no hay created_at (usar orden de inserción)
                latest = nif_feedback.iloc[-1]
            
            return latest.to_dict()
            
        except Exception as e:
            logger.error(f"Error recuperando feedback para {nif}: {e}")
            return None
    
    def _validate_feedback(self, feedback: FeedbackRecord) -> bool:
        """Valida la integridad del feedback."""
        if not feedback.nif or len(feedback.nif) < 8:
            logger.error(f"NIF inválido: {feedback.nif}")
            return False
        
        # Aceptar 0=FP, 1=Fraude, 2=Watchlist
        if feedback.analyst_verdict not in [VERDICT_FALSE_POSITIVE, VERDICT_FRAUD, VERDICT_WATCHLIST]:
            logger.error(f"Veredicto inválido: {feedback.analyst_verdict}")
            return False
        
        if not isinstance(feedback.fraud_score_original, (int, float)):
            logger.error("fraud_score_original debe ser numérico")
            return False
        
        # Validar campos condicionales
        if feedback.analyst_verdict == VERDICT_FALSE_POSITIVE:
            if not feedback.rejection_reason_code:
                logger.warning("FP sin rejection_reason_code - recomendado")
        
        if feedback.analyst_verdict == VERDICT_FRAUD:
            if not feedback.fraud_typology_code:
                logger.warning("Fraude sin fraud_typology_code - recomendado")
        
        return True
    
    def get_training_data(
        self,
        min_date: datetime = None,
        max_date: datetime = None
    ) -> pd.DataFrame:
        """
        Obtiene datos para entrenamiento del ReRanker.
        
        Args:
            min_date: Fecha mínima de feedback
            max_date: Fecha máxima de feedback
            
        Returns:
            DataFrame con feedback filtrado
        """
        df = pd.read_parquet(self.storage_path)
        
        if len(df) == 0:
            return df
        
        # Convertir columnas de fecha
        df["fecha_analisis"] = pd.to_datetime(df["fecha_analisis"])
        
        if min_date:
            df = df[df["fecha_analisis"] >= min_date]
        
        if max_date:
            df = df[df["fecha_analisis"] <= max_date]
        
        return df
    
    def get_feedback_stats(self, window_days: int = 7) -> Dict:
        """
        Calcula estadísticas de feedback reciente.
        
        Args:
            window_days: Número de días hacia atrás
            
        Returns:
            Diccionario con estadísticas
        """
        min_date = datetime.now() - timedelta(days=window_days)
        df = self.get_training_data(min_date=min_date)
        
        if len(df) == 0:
            return {
                "total_feedback": 0,
                "false_positives": 0,
                "confirmed_fraud": 0,
                "fp_rate": 0.0,
                "by_sector": {},
                "by_rule": {}
            }
        
        total = len(df)
        fp_count = (df["analyst_verdict"] == 0).sum()
        fraud_count = (df["analyst_verdict"] == 1).sum()
        
        # Stats por sector
        by_sector = {}
        if "cnae_sector" in df.columns:
            for sector in df["cnae_sector"].dropna().unique():
                sector_df = df[df["cnae_sector"] == sector]
                sector_fp = (sector_df["analyst_verdict"] == 0).sum()
                by_sector[sector] = {
                    "total": len(sector_df),
                    "fp_rate": sector_fp / len(sector_df) if len(sector_df) > 0 else 0
                }
        
        # Stats por regla (basado en flags_active)
        by_rule = {}
        if "flags_active" in df.columns:
            all_flags = []
            for flags_list in df["flags_active"].dropna():
                if isinstance(flags_list, list):
                    all_flags.extend(flags_list)
            
            unique_flags = set(all_flags)
            for flag in unique_flags:
                # Contar FP rate para empresas con este flag
                flag_df = df[df["flags_active"].apply(
                    lambda x: flag in x if isinstance(x, list) else False
                )]
                if len(flag_df) > 0:
                    flag_fp = (flag_df["analyst_verdict"] == 0).sum()
                    by_rule[flag] = {
                        "total": len(flag_df),
                        "fp_rate": flag_fp / len(flag_df)
                    }
        
        return {
            "total_feedback": total,
            "false_positives": int(fp_count),
            "confirmed_fraud": int(fraud_count),
            "fp_rate": fp_count / total if total > 0 else 0.0,
            "by_sector": by_sector,
            "by_rule": by_rule,
            "window_days": window_days,
            "min_date": min_date.isoformat()
        }
    
    def is_ready_for_training(self) -> Tuple[bool, str]:
        """
        Verifica si hay suficiente feedback para entrenar.
        
        Returns:
            Tuple (is_ready, reason_message)
        """
        df = self.get_training_data()
        
        total = len(df)
        if total < self.config.min_samples_for_training:
            return (
                False, 
                f"Insuficientes muestras: {total} < {self.config.min_samples_for_training}"
            )
        
        fp_count = (df["analyst_verdict"] == 0).sum()
        fraud_count = (df["analyst_verdict"] == 1).sum()
        
        if fp_count < self.config.min_samples_per_class:
            return (
                False,
                f"Insuficientes Falsos Positivos: {fp_count} < {self.config.min_samples_per_class}"
            )
        
        if fraud_count < self.config.min_samples_per_class:
            return (
                False,
                f"Insuficientes Fraudes Confirmados: {fraud_count} < {self.config.min_samples_per_class}"
            )
        
        return (True, f"Listo para entrenamiento: {total} muestras ({fp_count} FP, {fraud_count} TP)")
    
    def get_sample_count(self) -> Dict[str, int]:
        """Retorna conteo de muestras por clase incluyendo Watchlist."""
        df = self.get_training_data()
        if len(df) == 0:
            return {
                "total": 0,
                "false_positives": 0,
                "confirmed_fraud": 0,
                "watchlist": 0
            }
        return {
            "total": len(df),
            "false_positives": int((df["analyst_verdict"] == VERDICT_FALSE_POSITIVE).sum()),
            "confirmed_fraud": int((df["analyst_verdict"] == VERDICT_FRAUD).sum()),
            "watchlist": int((df["analyst_verdict"] == VERDICT_WATCHLIST).sum())
        }
    
    def get_watchlist_nifs(self, max_age_days: int = 180) -> List[str]:
        """
        Obtiene lista de NIFs en Watchlist (zona gris).
        
        Args:
            max_age_days: Antigüedad máxima del feedback para considerar
            
        Returns:
            Lista de NIFs que están en watchlist
        """
        min_date = datetime.now() - timedelta(days=max_age_days)
        df = self.get_training_data(min_date=min_date)
        
        if len(df) == 0:
            return []
        
        # Filtrar solo watchlist (verdict = 2)
        watchlist_df = df[df["analyst_verdict"] == VERDICT_WATCHLIST]
        
        return watchlist_df["nif"].unique().tolist()
    
    def analyze_rejection_reasons(self, window_days: int = 30) -> Dict:
        """
        Analiza las causas de rechazo (falsos positivos) para sugerir
        ajustes de umbrales.
        
        Returns:
            Diccionario con estadísticas por razón de rechazo y sector
        """
        min_date = datetime.now() - timedelta(days=window_days)
        df = self.get_training_data(min_date=min_date)
        
        if len(df) == 0:
            return {"total_rejections": 0, "by_reason": {}, "by_sector": {}}
        
        # Solo falsos positivos
        fp_df = df[df["analyst_verdict"] == VERDICT_FALSE_POSITIVE]
        
        if len(fp_df) == 0:
            return {"total_rejections": 0, "by_reason": {}, "by_sector": {}}
        
        # Estadísticas por razón
        by_reason = {}
        if "rejection_reason_code" in fp_df.columns:
            reason_counts = fp_df["rejection_reason_code"].value_counts()
            total_fp = len(fp_df)
            for reason, count in reason_counts.items():
                if reason and reason != "":
                    by_reason[reason] = {
                        "count": int(count),
                        "percentage": count / total_fp * 100
                    }
        
        # Estadísticas por sector + razón
        by_sector = {}
        if "cnae_sector" in fp_df.columns and "rejection_reason_code" in fp_df.columns:
            for sector in fp_df["cnae_sector"].dropna().unique():
                sector_df = fp_df[fp_df["cnae_sector"] == sector]
                sector_reasons = sector_df["rejection_reason_code"].value_counts()
                sector_total = len(sector_df)
                
                # Si >50% son SECTOR_NORMAL, sugerir relajar umbral
                sector_normal_pct = sector_reasons.get("SECTOR_NORMAL", 0) / sector_total if sector_total > 0 else 0
                
                by_sector[sector] = {
                    "total_fp": int(sector_total),
                    "sector_normal_pct": sector_normal_pct * 100,
                    "suggest_relax_threshold": sector_normal_pct > 0.5,
                    "reasons": {k: int(v) for k, v in sector_reasons.items() if k}
                }
        
        return {
            "total_rejections": len(fp_df),
            "by_reason": by_reason,
            "by_sector": by_sector,
            "window_days": window_days
        }
    
    def get_fraud_typology_stats(self, window_days: int = 90) -> Dict:
        """
        Estadísticas de tipologías de fraude confirmadas.
        
        Returns:
            Diccionario con conteo por tipología
        """
        min_date = datetime.now() - timedelta(days=window_days)
        df = self.get_training_data(min_date=min_date)
        
        if len(df) == 0:
            return {"total_frauds": 0, "by_typology": {}}
        
        # Solo fraudes confirmados
        fraud_df = df[df["analyst_verdict"] == VERDICT_FRAUD]
        
        if len(fraud_df) == 0:
            return {"total_frauds": 0, "by_typology": {}}
        
        by_typology = {}
        if "fraud_typology_code" in fraud_df.columns:
            for typology, count in fraud_df["fraud_typology_code"].value_counts().items():
                if typology:
                    by_typology[typology] = {
                        "count": int(count),
                        "description": FRAUD_TYPOLOGY_CODES.get(typology, typology)
                    }
        
        return {
            "total_frauds": len(fraud_df),
            "by_typology": by_typology,
            "window_days": window_days
        }


# =============================================================================
# FEEDBACK STORE (SPARK VERSION - Para Databricks/producción)
# =============================================================================

class FeedbackStoreSpark:
    """
    Implementación de FeedbackStore usando PySpark y Delta Lake.
    Para uso en entornos Databricks/producción.
    """
    
    def __init__(
        self,
        spark: "SparkSession",
        table_path: str = None,
        config: ContinuousLearningConfig = None
    ):
        if not SPARK_AVAILABLE:
            raise RuntimeError("PySpark no disponible. Usar FeedbackStorePandas.")
        
        self.spark = spark
        self.config = config or get_config()
        self.table_path = table_path or self.config.feedback_table_path
        self.audit_logger = get_audit_logger()
        self._ensure_table_exists()
    
    def _ensure_table_exists(self) -> None:
        """Crea la tabla Delta si no existe."""
        if DELTA_AVAILABLE:
            try:
                DeltaTable.forPath(self.spark, self.table_path)
                logger.info(f"Tabla Delta existente: {self.table_path}")
            except Exception:
                # Crear tabla vacía
                empty_df = self.spark.createDataFrame([], FEEDBACK_SCHEMA)
                empty_df.write.format("delta").mode("overwrite").save(self.table_path)
                logger.info(f"Tabla Delta creada: {self.table_path}")
        else:
            logger.warning("Delta Lake no disponible. Usando Parquet simple.")
    
    def log_feedback(self, feedback: FeedbackRecord) -> str:
        """
        Registra el feedback de un analista usando upsert Delta.
        """
        record = feedback.to_dict()
        
        # Crear DataFrame de una fila
        record_df = self.spark.createDataFrame([record], FEEDBACK_SCHEMA)
        
        if DELTA_AVAILABLE:
            # Upsert usando Delta merge
            delta_table = DeltaTable.forPath(self.spark, self.table_path)
            
            delta_table.alias("target").merge(
                record_df.alias("source"),
                "target.nif = source.nif AND target.fecha_alerta = source.fecha_alerta"
            ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
        else:
            # Append simple para Parquet
            record_df.write.mode("append").parquet(self.table_path)
        
        logger.info(f"Feedback registrado (Spark): {record['feedback_id']}")
        
        return record["feedback_id"]
    
    def get_training_data(
        self,
        min_date: datetime = None,
        max_date: datetime = None
    ) -> "DataFrame":
        """Obtiene datos para entrenamiento (Spark DataFrame)."""
        if DELTA_AVAILABLE:
            df = self.spark.read.format("delta").load(self.table_path)
        else:
            df = self.spark.read.parquet(self.table_path)
        
        if min_date:
            df = df.filter(F.col("fecha_analisis") >= min_date)
        
        if max_date:
            df = df.filter(F.col("fecha_analisis") <= max_date)
        
        return df
    
    def get_feedback_stats(self, window_days: int = 7) -> Dict:
        """Calcula estadísticas (versión Spark)."""
        min_date = datetime.now() - timedelta(days=window_days)
        df = self.get_training_data(min_date=min_date)
        
        # Convertir a Pandas para estadísticas simples
        pdf = df.toPandas()
        
        # Reusar lógica de versión Pandas
        pandas_store = FeedbackStorePandas.__new__(FeedbackStorePandas)
        pandas_store.config = self.config
        return pandas_store.get_feedback_stats.__func__(
            type('obj', (object,), {'get_training_data': lambda *args, **kwargs: pdf, 'config': self.config})(),
            window_days
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_feedback_store(
    spark: "SparkSession" = None,
    use_spark: bool = None
) -> "FeedbackStorePandas | FeedbackStoreSpark":
    """
    Factory para obtener la implementación correcta de FeedbackStore.
    
    Args:
        spark: SparkSession (opcional)
        use_spark: Forzar uso de Spark (default: auto-detectar)
        
    Returns:
        FeedbackStorePandas o FeedbackStoreSpark según disponibilidad
    """
    if use_spark is None:
        use_spark = spark is not None and SPARK_AVAILABLE
    
    if use_spark:
        if spark is None:
            raise ValueError("SparkSession requerida para FeedbackStoreSpark")
        return FeedbackStoreSpark(spark)
    else:
        return FeedbackStorePandas()
