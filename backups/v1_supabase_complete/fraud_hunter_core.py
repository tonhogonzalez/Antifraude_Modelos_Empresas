"""
FraudHunter Core - Sistema de Detección de Fraude Empresarial
==============================================================
Aplicación MLOps para Databricks que analiza datos tributarios españoles
(Modelos 200, 347, 349) mediante ML no supervisado y análisis forense.

Author: Lead MLOps Engineer
Version: 1.0.0
Runtime: Databricks ML (PySpark, Delta Lake, MLflow)
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

# PySpark imports
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    IntegerType, DateType, BooleanType
)
from pyspark.sql.window import Window

# ML imports
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from scipy import stats
from scipy.spatial.distance import mahalanobis

# PySpark ML imports for distributed operations
try:
    from pyspark.ml.stat import Summarizer
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql.functions import pandas_udf
    PYSPARK_ML_AVAILABLE = True
except ImportError:
    PYSPARK_ML_AVAILABLE = False
    warnings.warn("PySpark ML no disponible. Algunas optimizaciones serán deshabilitadas.")

# MLflow (optional for local testing)
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow no disponible. El registro de modelos será omitido.")

# GraphFrames (optional)
try:
    from graphframes import GraphFrame
    GRAPHFRAMES_AVAILABLE = True
except ImportError:
    GRAPHFRAMES_AVAILABLE = False
    warnings.warn("GraphFrames no disponible. Análisis de grafos será omitido.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FraudHunter")


# =============================================================================
# CONFIGURACIÓN Y CONSTANTES
# =============================================================================

# Importar configuración centralizada de casillas AEAT
from tax_mapping_config import TAX_MAPPING_CONFIG

@dataclass
class FraudHunterConfig:
    """Configuración global del sistema."""
    contamination: float = 0.05  # % esperado de fraude
    benford_threshold: float = 0.15  # Umbral divergencia Benford
    round_number_threshold: float = 0.30  # % números redondos sospechoso
    interest_rate_market: float = 0.05  # Tasa interés mercado referencia
    min_transactions: int = 10  # Mínimo transacciones para análisis
    
    # Mapeo de casillas AEAT - Centralizado desde tax_mapping_config.py
    TAX_MAPPING: Dict = None
    
    def __post_init__(self):
        """Inicializa el mapeo de casillas desde configuración centralizada."""
        if self.TAX_MAPPING is None:
            self.TAX_MAPPING = {
                "M200": {
                    var: codes for var, codes in TAX_MAPPING_CONFIG["M200"].items()
                    if isinstance(codes, list)  # Solo mapeos de casillas
                },
                # M349: extraer solo source_to_canonical si existe
                "M349": TAX_MAPPING_CONFIG.get("M349", {}).get("source_to_canonical", {}),
                # M347: extraer solo source_to_canonical si existe
                "M347": TAX_MAPPING_CONFIG.get("M347", {}).get("source_to_canonical", {}),
            }


# =============================================================================
# CLASE 1: TaxDataLoader (ETL: Vertical -> Horizontal)
# =============================================================================

class TaxDataLoader:
    """
    Capa de Ingesta y Transformación (Bronze -> Silver).
    Transforma datos verticales (Key-Value) a tabla analítica horizontal.
    
    VALOR FORENSE: El pivoteo inteligente filtra solo las casillas relevantes,
    optimizando memoria y velocidad en datasets grandes (100K+ empresas).
    """
    
    def __init__(self, spark: SparkSession, config: FraudHunterConfig = None):
        self.spark = spark
        self.config = config or FraudHunterConfig()
        self.logger = logging.getLogger(f"{__name__}.TaxDataLoader")

    @staticmethod
    def validate_nif(nif_col: str) -> F.Column:
        """
        Valida formato NIF español (8 dígitos + letra o letra + 7 dígitos + letra).
        
        Detección de fraude: NIFs inválidos pueden indicar empresas fantasma
        o errores deliberados para dificultar el cruce de datos.
        """
        pattern_persona = r'^[0-9]{8}[A-Z]$'
        pattern_empresa = r'^[A-Z][0-9]{7}[A-Z0-9]$'
        return (
            F.col(nif_col).rlike(pattern_persona) | 
            F.col(nif_col).rlike(pattern_empresa)
        )

    def load_and_pivot_data(self, paths_config: Dict[str, str]) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Carga tablas verticales, filtra casillas inútiles y pivota.
        
        Args:
            paths_config: Diccionario con rutas {'m200': path, 'm347': path, 'm349': path}
            
        Input esperado: Tablas con schema [nif, ejercicio, casilla, valor]
        
        Returns:
            Tupla (df_m200_pivoted, df_m347, df_m349_pivoted)
        """
        self.logger.info("Iniciando carga y transformación de datos tributarios...")
        
        # 1. Cargar Datos Crudos (Verticales)
        df_raw_m200 = self._load_raw(paths_config['m200'], "M200")
        df_m347 = self._load_raw(paths_config['m347'], "M347")  # M347 es transaccional
        df_raw_m349 = self._load_raw(paths_config['m349'], "M349")

        # 2. Transformar M200 (Pivot)
        self.logger.info("Transformando Modelo 200 (Pivoting)...")
        df_m200_clean = self._pivot_tax_model(df_raw_m200, "M200")
        
        # 3. Transformar M349 (Pivot / Agregación)
        self.logger.info("Transformando Modelo 349...")
        df_m349_clean = self._pivot_tax_model(df_raw_m349, "M349")

        self.logger.info(
            f"Datos transformados: M200={df_m200_clean.count()}, "
            f"M347={df_m347.count()}, M349={df_m349_clean.count()}"
        )

        return df_m200_clean, df_m347, df_m349_clean

    def _load_raw(self, path: str, name: str) -> DataFrame:
        """Carga genérica (Parquet/Delta) con validación de NIFs."""
        try:
            try:
                df = self.spark.read.format("delta").load(path)
            except Exception:
                df = self.spark.read.parquet(path)
            
            # Validar NIFs si la columna existe
            nif_col = 'nif' if 'nif' in df.columns else 'nif_declarante'
            if nif_col in df.columns:
                df = df.filter(self.validate_nif(nif_col))
            
            self.logger.info(f"{name} cargado: {df.count()} registros")
            return df
            
        except Exception as e:
            self.logger.error(f"Error cargando {name}: {str(e)}")
            raise

    def _pivot_tax_model(self, df: DataFrame, model_type: str) -> DataFrame:
        """
        Realiza el pivoteo optimizado:
        1. Filtra solo las casillas definidas en el MAPEO (ahorra memoria).
        2. Normaliza códigos de casilla.
        3. Pivota a columnas de negocio.
        
        VALOR FORENSE: El filtrado temprano reduce dramáticamente el volumen
        de datos procesados (típicamente 70-80% menos filas).
        """
        mapping = self.config.TAX_MAPPING.get(model_type, {})
        
        if not mapping:
            self.logger.warning(f"No hay mapeo definido para {model_type}")
            return df
        
        # Verificar que el DataFrame tiene las columnas necesarias
        required_cols = ['nif', 'ejercicio', 'casilla', 'valor']
        if not all(col in df.columns for col in required_cols):
            self.logger.warning(f"Esquema incorrecto para pivot en {model_type}")
            return df
        
        # Aplanar lista de códigos relevantes para filtrado (Push-down predicate)
        relevant_codes = [code for codes in mapping.values() for code in codes]
        
        # Filtrar DF para trabajar solo con lo necesario
        df_filtered = df.filter(F.col("casilla").isin(relevant_codes))
        
        self.logger.info(
            f"   {model_type}: Filtradas {df.count()} -> {df_filtered.count()} filas "
            f"(casillas relevantes: {len(relevant_codes)})"
        )

        # Crear expresión CASE WHEN para renombrar códigos a variables legibles
        # Ejemplo: WHEN casilla IN ('00255', 'C00255') THEN 'cifra_negocios'
        case_expr = "CASE "
        for business_var, tech_codes in mapping.items():
            codes_list = ",".join([f"'{c}'" for c in tech_codes])
            case_expr += f" WHEN casilla IN ({codes_list}) THEN '{business_var}'"
        case_expr += " ELSE NULL END"

        df_mapped = df_filtered.select(
            "nif", "ejercicio", "valor",
            F.expr(case_expr).alias("kpi_name")
        ).filter(F.col("kpi_name").isNotNull())

        # PIVOT FINAL
        # GroupBy NIF+Ejercicio -> Pivot KPI -> Sum Valor
        df_pivoted = df_mapped.groupBy("nif", "ejercicio") \
            .pivot("kpi_name") \
            .sum("valor") \
            .fillna(0)  # Rellenar nulos con 0 es seguro en datos tributarios

        self.logger.info(f"   {model_type}: Pivotado a {len(df_pivoted.columns)} columnas")
        
        return df_pivoted

    # Método legacy para compatibilidad con código existente
    def load_and_clean_data(
        self, 
        paths_config: Dict[str, str]
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Método de compatibilidad - redirige a load_and_pivot_data.
        
        NOTA: Este método se mantiene para no romper código existente.
        Para nuevas implementaciones, usar load_and_pivot_data().
        """
        self.logger.warning(
            "load_and_clean_data() está deprecado. "
            "Usar load_and_pivot_data() para mejor rendimiento."
        )
        return self.load_and_pivot_data(paths_config)


# =============================================================================
# CLASE 2: ForensicFeatureEngineer (Silver -> Gold)
# =============================================================================

class ForensicFeatureEngineer:
    """
    Motor de ingeniería de características forenses (Silver → Gold).
    
    Implementa métricas basadas en:
    - Hard Rules: Cruces entre modelos tributarios
    - Financial Forensics: Ratios de manipulación contable
    - Statistical Patterns: Ley de Benford y números redondos
    """
    
    def __init__(self, spark: SparkSession, config: FraudHunterConfig = None):
        self.spark = spark
        self.config = config or FraudHunterConfig()
        self.logger = logging.getLogger(f"{__name__}.ForensicFeatureEngineer")
    
    def build_abt(
        self, 
        df_m200: DataFrame, 
        df_m347: DataFrame, 
        df_m349: DataFrame
    ) -> DataFrame:
        """
        Construye la Analytical Base Table (ABT) con todas las features.
        
        Args:
            df_m200: Datos del Modelo 200 (ya pivotados con nombres de TAX_MAPPING)
            df_m347: Datos del Modelo 347
            df_m349: Datos del Modelo 349
        
        Returns:
            DataFrame con todas las features forenses calculadas
        """
        self.logger.info("Construyendo ABT con features forenses...")
        
        # Agregar M347 por declarante
        df_m347_agg = df_m347.groupBy("nif_declarante", "ejercicio").agg(
            F.sum("importe_total").alias("total_m347"),
            F.count("*").alias("num_contrapartes_347"),
            F.collect_list("importe_total").alias("importes_347")
        )
        
        # Agregar M349 por declarante
        df_m349_agg = df_m349.groupBy("nif_declarante", "ejercicio").agg(
            F.sum("importe").alias("total_m349"),
            F.count("*").alias("num_operaciones_349")
        )
        
        # JOIN base M200 con agregados
        abt = df_m200.alias("m200") \
            .join(
                df_m347_agg.alias("m347"),
                (F.col("m200.nif") == F.col("m347.nif_declarante")) &
                (F.col("m200.ejercicio") == F.col("m347.ejercicio")),
                "left"
            ) \
            .join(
                df_m349_agg.alias("m349"),
                (F.col("m200.nif") == F.col("m349.nif_declarante")) &
                (F.col("m200.ejercicio") == F.col("m349.ejercicio")),
                "left"
            ) \
            .select(
                F.col("m200.*"),
                F.coalesce(F.col("m347.total_m347"), F.lit(0.0)).alias("total_m347"),
                F.coalesce(F.col("m347.num_contrapartes_347"), F.lit(0)).alias("num_contrapartes_347"),
                F.col("m347.importes_347"),
                F.coalesce(F.col("m349.total_m349"), F.lit(0.0)).alias("total_m349"),
                F.coalesce(F.col("m349.num_operaciones_349"), F.lit(0)).alias("num_operaciones_349")
            )
        
        # =====================================================================
        # PRE-CÁLCULO DE VARIABLES DERIVADAS
        # =====================================================================
        
        # Deuda Bancaria Total = Deuda LP + Deuda CP (si existen las columnas)
        if "deuda_bancaria_lp" in abt.columns and "deuda_bancaria_cp" in abt.columns:
            abt = abt.withColumn(
                "deuda_bancaria", 
                F.coalesce(F.col("deuda_bancaria_lp"), F.lit(0.0)) + 
                F.coalesce(F.col("deuda_bancaria_cp"), F.lit(0.0))
            )
        elif "deuda_bancaria_lp" in abt.columns:
            # Solo LP disponible
            abt = abt.withColumn("deuda_bancaria", F.col("deuda_bancaria_lp"))
        
        # Proxy de Flujo de Caja = Resultado + Amortizaciones
        # (Mejor que usar solo resultado, ya que amortización no es salida de caja)
        if "resultado_ejercicio" in abt.columns and "amortizaciones" in abt.columns:
            abt = abt.withColumn(
                "flujo_caja_operativo",
                F.coalesce(F.col("resultado_ejercicio"), F.lit(0.0)) + 
                F.coalesce(F.col("amortizaciones"), F.lit(0.0))
            )
        
        # Intereses pagados = gastos_financieros (alias para compatibilidad)
        if "gastos_financieros" in abt.columns and "intereses_pagados" not in abt.columns:
            abt = abt.withColumn("intereses_pagados", F.col("gastos_financieros"))
        
        # =====================================================================
        # RENOMBRAR COLUMNAS PARA COMPATIBILIDAD CON MÉTODOS EXISTENTES
        # =====================================================================
        column_renames = {
            "cifra_negocios": "ventas_netas",
            "resultado_ejercicio": "resultado_neto",
            "total_activo": "activo_total",
            "otros_gastos_expl": "gastos_transporte",  # Proxy para gastos logísticos
            "efectivo_tesoreria": "caja"
        }
        
        for old_name, new_name in column_renames.items():
            if old_name in abt.columns:
                abt = abt.withColumnRenamed(old_name, new_name)
        
        self.logger.info(f"   Variables derivadas calculadas y columnas renombradas")
        
        # =====================================================================
        # CALCULAR FEATURES FORENSES
        # =====================================================================
        abt = self.calc_cobertura_ventas(abt)
        abt = self.calc_incoherencia_logistica(abt)
        abt = self.calc_accruals_ratio(abt)
        abt = self.calc_hidden_debt(abt)
        abt = self.calc_labor_productivity(abt)
        abt = self.calc_round_numbers(abt)
        
        # Benford requiere UDF (calculado aparte)
        abt = self.calc_benford_divergence(abt)
        
        self.logger.info(f"ABT construida: {abt.count()} empresas, {len(abt.columns)} features")
        return abt
    
    def calc_cobertura_ventas(self, df: DataFrame) -> DataFrame:
        """
        Calcula: Suma M347 / Ventas M200.
        
        VALOR FORENSE: Una ratio muy baja (<0.5) indica que la empresa declara
        ventas que no aparecen en el cruce con terceros, sugiriendo:
        - Ventas ficticias para inflar facturación
        - Contrapartes que no declaran (empresas fantasma)
        Una ratio muy alta (>1.2) puede indicar ocultación de ventas.
        """
        return df.withColumn(
            "cobertura_ventas",
            F.when(
                F.col("ventas_netas") > 0,
                F.col("total_m347") / F.col("ventas_netas")
            ).otherwise(F.lit(None))
        )
    
    def calc_incoherencia_logistica(self, df: DataFrame) -> DataFrame:
        """
        Flag: Operaciones UE (M349) > 0 AND Gastos Transporte ≈ 0.
        
        VALOR FORENSE: Una empresa que declara operaciones intracomunitarias
        (importaciones/exportaciones) pero no tiene gastos de transporte es
        altamente sospechosa. Patrón típico de:
        - Fraude carrusel IVA
        - Empresas pantalla que solo facturan sin actividad real
        - Missing trader fraud
        """
        umbral_transporte = 1000  # Mínimo razonable de gastos transporte
        
        return df.withColumn(
            "flag_incoherencia_logistica",
            F.when(
                (F.col("total_m349") > 0) & 
                (F.col("gastos_transporte") < umbral_transporte),
                F.lit(1)
            ).otherwise(F.lit(0))
        )
    
    def calc_accruals_ratio(self, df: DataFrame) -> DataFrame:
        """
        Calcula: (Resultado Neto - Flujo Caja Operativo) / Activo Total.
        
        VALOR FORENSE: Ratio alta indica "earnings management" - manipulación
        contable donde los beneficios no se traducen en caja real. Señales:
        - Ingresos ficticios registrados pero no cobrados
        - Capitalización agresiva de gastos
        - Técnica documentada en fraudes como Enron, WorldCom
        Valores > 0.10 requieren investigación.
        """
        return df.withColumn(
            "accruals_ratio",
            F.when(
                F.col("activo_total") > 0,
                (F.col("resultado_neto") - F.col("flujo_caja_operativo")) / 
                F.col("activo_total")
            ).otherwise(F.lit(None))
        )
    
    def calc_hidden_debt(self, df: DataFrame) -> DataFrame:
        """
        Calcula: Tasa implícita = Intereses / Deuda Bancaria.
        
        VALOR FORENSE: Una tasa de interés muy superior al mercado (>8-10%)
        puede indicar:
        - Financiación de fuentes no bancarias (usura, blanqueo)
        - Intereses ficticios para reducir base imponible
        - Préstamos con partes vinculadas a precios de transferencia
        Flag si tasa > 2x mercado.
        """
        return df.withColumn(
            "tasa_interes_implicita",
            F.when(
                F.col("deuda_bancaria") > 0,
                F.col("intereses_pagados") / F.col("deuda_bancaria")
            ).otherwise(F.lit(0.0))
        ).withColumn(
            "flag_hidden_debt",
            F.when(
                F.col("tasa_interes_implicita") > (self.config.interest_rate_market * 2),
                F.lit(1)
            ).otherwise(F.lit(0))
        )
    
    def calc_labor_productivity(self, df: DataFrame) -> DataFrame:
        """
        Calcula: Ventas / Gastos de Personal.
        
        VALOR FORENSE: Productividad extremadamente alta (>50x) sugiere:
        - Empresa pantalla con facturación sin empleados reales
        - Estructura para facturación falsa
        - Personal externalizado de forma irregular
        Productividad muy baja (<2x) puede indicar:
        - Gastos de personal inflados para reducir beneficios
        - Nóminas ficticias
        """
        return df.withColumn(
            "productividad_laboral",
            F.when(
                F.col("gastos_personal") > 0,
                F.col("ventas_netas") / F.col("gastos_personal")
            ).otherwise(F.lit(None))
        ).withColumn(
            "flag_empresa_pantalla",
            F.when(
                (F.col("ventas_netas") > 100000) & 
                (F.col("gastos_personal") < 10000),
                F.lit(1)
            ).otherwise(F.lit(0))
        )
    
    def calc_round_numbers(self, df: DataFrame) -> DataFrame:
        """
        Calcula: % de importes M347 terminados en '00' o '000'.
        
        VALOR FORENSE: Las facturas reales rara vez tienen importes redondos.
        Un porcentaje alto (>30%) de números terminados en ceros sugiere:
        - Facturas fabricadas manualmente
        - Importes inventados (no calculados de unidades x precio)
        - Patrón clásico de fraude documental
        
        OPTIMIZACIÓN: Usa Pandas UDF vectorizada para ~10x mejor rendimiento.
        """
        # Usar Pandas UDF vectorizada si está disponible (más eficiente)
        if PYSPARK_ML_AVAILABLE:
            @pandas_udf(DoubleType())
            def calc_round_pct_vectorized(importes_series: pd.Series) -> pd.Series:
                """Pandas UDF vectorizada para calcular % números redondos."""
                def calc_pct(importes):
                    if importes is None or len(importes) == 0:
                        return None
                    round_count = sum(
                        1 for imp in importes 
                        if imp is not None and int(imp) % 100 == 0
                    )
                    return round_count / len(importes)
                return importes_series.apply(calc_pct)
            
            calc_round_pct = calc_round_pct_vectorized
        else:
            # Fallback a UDF regular si PySpark ML no está disponible
            @F.udf(DoubleType())
            def calc_round_pct(importes):
                if not importes or len(importes) == 0:
                    return None
                round_count = sum(
                    1 for imp in importes 
                    if imp and (int(imp) % 100 == 0)
                )
                return round_count / len(importes)
        
        return df.withColumn(
            "pct_numeros_redondos",
            calc_round_pct(F.col("importes_347"))
        ).withColumn(
            "flag_numeros_redondos",
            F.when(
                F.col("pct_numeros_redondos") > self.config.round_number_threshold,
                F.lit(1)
            ).otherwise(F.lit(0))
        )
    
    def calc_benford_divergence(self, df: DataFrame) -> DataFrame:
        """
        Calcula: Divergencia KL del primer dígito vs distribución Benford.
        
        VALOR FORENSE: La Ley de Benford establece que en datos naturales,
        el primer dígito sigue una distribución logarítmica (1 aparece ~30%,
        9 aparece ~5%). Datos manipulados violan esta ley porque:
        - Humanos tienden a inventar números "uniformes"
        - Duplicación de facturas altera la distribución
        - Técnica usada para detectar fraude electoral y contable
        KL divergence > 0.15 es altamente sospechoso.
        """
        # Distribución esperada de Benford
        benford_expected = {
            1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097,
            5: 0.079, 6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046
        }
        
        @F.udf(DoubleType())
        def calc_kl_divergence(importes):
            if not importes or len(importes) < 10:
                return None
            
            # Extraer primeros dígitos
            first_digits = []
            for imp in importes:
                if imp and imp > 0:
                    first_digit = int(str(int(abs(imp)))[0])
                    if 1 <= first_digit <= 9:
                        first_digits.append(first_digit)
            
            if len(first_digits) < 10:
                return None
            
            # Calcular distribución observada
            observed = {i: 0 for i in range(1, 10)}
            for d in first_digits:
                observed[d] += 1
            total = len(first_digits)
            
            # KL Divergence: sum(P(x) * log(P(x)/Q(x)))
            kl_div = 0.0
            for digit in range(1, 10):
                p = observed[digit] / total
                q = benford_expected[digit]
                if p > 0:
                    kl_div += p * np.log(p / q)
            
            return float(kl_div)
        
        return df.withColumn(
            "benford_divergence",
            calc_kl_divergence(F.col("importes_347"))
        ).withColumn(
            "flag_benford_anomaly",
            F.when(
                F.col("benford_divergence") > self.config.benford_threshold,
                F.lit(1)
            ).otherwise(F.lit(0))
        )


# =============================================================================
# CLASE 3: GraphNetworkAnalyzer (GraphFrames)
# =============================================================================

class GraphNetworkAnalyzer:
    """
    Analizador de redes de transacciones basado en GraphFrames.
    
    VALOR FORENSE: Las tramas de fraude rara vez operan de forma aislada.
    Identificar patrones de red (hubs, comunidades cerradas, ciclos) permite
    detectar estructuras de blanqueo y fraude carrusel.
    
    Nota: Esta clase requiere GraphFrames. Si no está disponible, los métodos
    devolverán DataFrames vacíos con advertencias (graceful degradation).
    """
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.graph = None
        self.logger = logging.getLogger(f"{__name__}.GraphNetworkAnalyzer")
        
        if not GRAPHFRAMES_AVAILABLE:
            self.logger.warning(
                "GraphFrames no disponible. Análisis de grafos deshabilitado."
            )
    
    def build_graph(self, df_edges: DataFrame) -> Optional['GraphFrame']:
        """
        Construye el grafo de relaciones desde M347.
        
        Args:
            df_edges: DataFrame con columnas (nif_declarante, nif_contraparte, importe_total)
        
        Returns:
            GraphFrame o None si GraphFrames no disponible
            
        VALOR FORENSE: El grafo representa el flujo de dinero entre empresas.
        Estructuras anómalas (estrellas, ciclos, cliques) indican patrones
        sospechosos que merecen investigación.
        """
        if not GRAPHFRAMES_AVAILABLE:
            self.logger.warning("Saltando build_graph: GraphFrames no disponible")
            return None
        
        try:
            # Preparar vértices (NIFs únicos)
            vertices_src = df_edges.select(
                F.col("nif_declarante").alias("id")
            ).distinct()
            vertices_dst = df_edges.select(
                F.col("nif_contraparte").alias("id")
            ).distinct()
            vertices = vertices_src.union(vertices_dst).distinct()
            
            # Preparar aristas
            edges = df_edges.select(
                F.col("nif_declarante").alias("src"),
                F.col("nif_contraparte").alias("dst"),
                F.col("importe_total").alias("weight")
            )
            
            self.graph = GraphFrame(vertices, edges)
            self.logger.info(
                f"Grafo construido: {vertices.count()} nodos, {edges.count()} aristas"
            )
            return self.graph
            
        except Exception as e:
            self.logger.error(f"Error construyendo grafo: {str(e)}")
            return None
    
    def calc_pagerank(self, reset_probability: float = 0.15, max_iter: int = 10) -> DataFrame:
        """
        Calcula PageRank para identificar nodos centrales.
        
        VALOR FORENSE: Empresas con alto PageRank son "hubs" en la red de
        transacciones. Una empresa con PageRank anormalmente alto pero
        baja facturación real puede ser:
        - Sociedad instrumental para canalizar fondos
        - Hub de blanqueo de dinero
        - Distribuidora ficticia en fraude carrusel
        
        Args:
            reset_probability: Probabilidad de salto aleatorio (default 0.15)
            max_iter: Iteraciones máximas
        
        Returns:
            DataFrame con (id, pagerank)
        """
        if not GRAPHFRAMES_AVAILABLE or self.graph is None:
            self.logger.warning("PageRank no calculado: grafo no disponible")
            return self.spark.createDataFrame([], "id STRING, pagerank DOUBLE")
        
        try:
            results = self.graph.pageRank(
                resetProbability=reset_probability,
                maxIter=max_iter
            )
            return results.vertices.select("id", "pagerank")
            
        except Exception as e:
            self.logger.error(f"Error en PageRank: {str(e)}")
            return self.spark.createDataFrame([], "id STRING, pagerank DOUBLE")
    
    def calc_connected_components(self) -> DataFrame:
        """
        Detecta componentes conexos (comunidades) en el grafo.
        
        VALOR FORENSE: Componentes pequeños y cerrados (10-50 empresas)
        que operan mayoritariamente entre sí pueden indicar:
        - Fraude carrusel IVA (circuito cerrado de facturas)
        - Red de empresas vinculadas para fragmentar operaciones
        - Grupo de blanqueo con estructura "lavadero"
        
        Returns:
            DataFrame con (id, component)
        """
        if not GRAPHFRAMES_AVAILABLE or self.graph is None:
            self.logger.warning("Componentes no calculados: grafo no disponible")
            return self.spark.createDataFrame([], "id STRING, component LONG")
        
        try:
            # Nota: connectedComponents requiere checkpoint
            self.spark.sparkContext.setCheckpointDir("/tmp/graphframes_checkpoint")
            results = self.graph.connectedComponents()
            return results.select("id", "component")
            
        except Exception as e:
            self.logger.error(f"Error en componentes conexos: {str(e)}")
            return self.spark.createDataFrame([], "id STRING, component LONG")
    
    def detect_circularity(self, max_path_length: int = 3) -> DataFrame:
        """
        Detecta patrones de facturación circular (A->B->...->A).
        
        VALOR FORENSE: La facturación recíproca o circular es un indicador
        fuerte de fraude:
        - A->B->A: Facturación cruzada para generar IVA ficticio
        - A->B->C->A: Fraude carrusel clásico
        - Ciclos con importes similares: Operaciones sin sustancia económica
        
        Args:
            max_path_length: Longitud máxima del ciclo a buscar
        
        Returns:
            DataFrame con ciclos detectados (src, path, dst)
        """
        if not GRAPHFRAMES_AVAILABLE or self.graph is None:
            self.logger.warning("Circularidad no detectada: grafo no disponible")
            return self.spark.createDataFrame(
                [], "src STRING, via STRING, dst STRING, is_circular BOOLEAN"
            )
        
        try:
            # Buscar patrones A->B->A (longitud 2)
            motif_2 = self.graph.find("(a)-[e1]->(b); (b)-[e2]->(a)")
            circular_2 = motif_2.filter(F.col("a.id") != F.col("b.id"))
            
            results = circular_2.select(
                F.col("a.id").alias("src"),
                F.col("b.id").alias("via"),
                F.col("a.id").alias("dst"),
                F.lit(True).alias("is_circular")
            ).distinct()
            
            # Buscar patrones A->B->C->A (longitud 3) si se requiere
            if max_path_length >= 3:
                motif_3 = self.graph.find(
                    "(a)-[e1]->(b); (b)-[e2]->(c); (c)-[e3]->(a)"
                )
                circular_3 = motif_3.filter(
                    (F.col("a.id") != F.col("b.id")) &
                    (F.col("b.id") != F.col("c.id")) &
                    (F.col("a.id") != F.col("c.id"))
                )
                
                results_3 = circular_3.select(
                    F.col("a.id").alias("src"),
                    F.concat_ws("->", F.col("b.id"), F.col("c.id")).alias("via"),
                    F.col("a.id").alias("dst"),
                    F.lit(True).alias("is_circular")
                ).distinct()
                
                results = results.union(results_3)
            
            self.logger.info(f"Ciclos detectados: {results.count()}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error detectando circularidad: {str(e)}")
            return self.spark.createDataFrame(
                [], "src STRING, via STRING, dst STRING, is_circular BOOLEAN"
            )


# =============================================================================
# CLASE 4: SectoralBenchmarker (Comparativa Sectorial)
# =============================================================================

class SectoralBenchmarker:
    """
    Comparador sectorial basado en CNAE.
    
    VALOR FORENSE: Una empresa que se desvía significativamente de sus pares
    sectoriales merece investigación. La distancia de Mahalanobis captura
    desviaciones multidimensionales que serían invisibles analizando
    cada ratio por separado.
    """
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = logging.getLogger(f"{__name__}.SectoralBenchmarker")
    
    def calc_mahalanobis_distance(
        self, 
        df: DataFrame,
        feature_cols: List[str] = None
    ) -> DataFrame:
        """
        Calcula la distancia de Mahalanobis de cada empresa vs su sector CNAE.
        
        VALOR FORENSE: La distancia de Mahalanobis mide cuán "rara" es una
        empresa comparada con sus pares, considerando las correlaciones
        entre variables. Detecta:
        - Empresas con combinaciones inusuales de ratios (ej: alto margen
          pero baja rotación en sector que típicamente tiene lo opuesto)
        - Outliers multidimensionales invisibles en análisis univariante
        - Posibles empresas con contabilidad manipulada para parecer "normales"
          en cada métrica individual pero anormales en conjunto
        
        Args:
            df: DataFrame con datos de empresas
            feature_cols: Columnas a usar (default: margen, rotación, endeudamiento)
        
        Returns:
            DataFrame con columna 'mahalanobis_distance' añadida
        """
        if feature_cols is None:
            feature_cols = ['margen_neto', 'rotacion_activos', 'ratio_endeudamiento']
        
        # Calcular ratios si no existen
        df_enriched = self._ensure_ratios(df)
        
        # Convertir a Pandas para cálculo de Mahalanobis
        # (En Databricks usaríamos Spark UDF con broadcast de covarianza)
        try:
            pdf = df_enriched.select(
                'nif', 'cnae', *feature_cols
            ).toPandas()
            
            # Calcular distancia por sector
            results = []
            for cnae in pdf['cnae'].unique():
                if pd.isna(cnae):
                    continue
                    
                sector_data = pdf[pdf['cnae'] == cnae][feature_cols].dropna()
                
                if len(sector_data) < 3:  # Mínimo para matriz de covarianza
                    continue
                
                # Centroide y covarianza del sector
                centroid = sector_data.mean().values
                cov_matrix = sector_data.cov().values
                
                # Regularización para evitar matriz singular
                cov_matrix += np.eye(len(feature_cols)) * 1e-6
                
                try:
                    cov_inv = np.linalg.inv(cov_matrix)
                except np.linalg.LinAlgError:
                    continue
                
                # Calcular distancia para cada empresa del sector
                sector_nifs = pdf[pdf['cnae'] == cnae]['nif'].tolist()
                for nif in sector_nifs:
                    row = pdf[(pdf['nif'] == nif) & (pdf['cnae'] == cnae)]
                    if row[feature_cols].isna().any().any():
                        distance = None
                    else:
                        point = row[feature_cols].values.flatten()
                        distance = float(mahalanobis(point, centroid, cov_inv))
                    results.append({'nif': nif, 'mahalanobis_distance': distance})
            
            # Convertir resultados a DataFrame Spark
            if results:
                df_distances = self.spark.createDataFrame(pd.DataFrame(results))
                df_result = df_enriched.join(
                    df_distances,
                    on='nif',
                    how='left'
                )
            else:
                df_result = df_enriched.withColumn(
                    'mahalanobis_distance', 
                    F.lit(None).cast(DoubleType())
                )
            
            # Flag para empresas muy desviadas (>3 desviaciones típicas)
            df_result = df_result.withColumn(
                'flag_sectoral_outlier',
                F.when(
                    F.col('mahalanobis_distance') > 3.0,
                    F.lit(1)
                ).otherwise(F.lit(0))
            )
            
            self.logger.info("Distancias Mahalanobis calculadas por sector")
            return df_result
            
        except Exception as e:
            self.logger.error(f"Error en cálculo Mahalanobis: {str(e)}")
            return df.withColumn(
                'mahalanobis_distance', 
                F.lit(None).cast(DoubleType())
            ).withColumn(
                'flag_sectoral_outlier',
                F.lit(0)
            )
    
    def _ensure_ratios(self, df: DataFrame) -> DataFrame:
        """Calcula ratios financieros si no existen."""
        # Margen neto
        if 'margen_neto' not in df.columns:
            df = df.withColumn(
                'margen_neto',
                F.when(
                    F.col('ventas_netas') > 0,
                    F.col('resultado_neto') / F.col('ventas_netas')
                ).otherwise(F.lit(None))
            )
        
        # Rotación de activos
        if 'rotacion_activos' not in df.columns:
            df = df.withColumn(
                'rotacion_activos',
                F.when(
                    F.col('activo_total') > 0,
                    F.col('ventas_netas') / F.col('activo_total')
                ).otherwise(F.lit(None))
            )
        
        # Ratio de endeudamiento
        if 'ratio_endeudamiento' not in df.columns:
            df = df.withColumn(
                'ratio_endeudamiento',
                F.when(
                    F.col('activo_total') > 0,
                    F.col('deuda_bancaria') / F.col('activo_total')
                ).otherwise(F.lit(None))
            )
        
        return df

    def calc_mahalanobis_distributed(
        self, 
        df: DataFrame,
        feature_cols: List[str] = None
    ) -> DataFrame:
        """
        Calcula la distancia de Mahalanobis de forma DISTRIBUIDA usando PySpark ML.
        
        VALOR FORENSE: Igual que calc_mahalanobis_distance pero escala a millones
        de registros sin traer datos al driver (evita OOM).
        
        OPTIMIZACIÓN: Usa VectorAssembler + Summarizer para calcular estadísticas
        por sector de forma distribuida, luego aplica Pandas UDF grupada.
        
        Args:
            df: DataFrame con datos de empresas (debe incluir columna 'cnae')
            feature_cols: Columnas a usar (default: margen, rotación, endeudamiento)
        
        Returns:
            DataFrame con columna 'mahalanobis_distance' añadida
        """
        if not PYSPARK_ML_AVAILABLE:
            self.logger.warning(
                "PySpark ML no disponible. Usando método tradicional (toPandas)."
            )
            return self.calc_mahalanobis_distance(df, feature_cols)
        
        if feature_cols is None:
            feature_cols = ['margen_neto', 'rotacion_activos', 'ratio_endeudamiento']
        
        # Calcular ratios si no existen
        df_enriched = self._ensure_ratios(df)
        
        try:
            # 1. Vectorizar features con VectorAssembler
            assembler = VectorAssembler(
                inputCols=feature_cols, 
                outputCol="features", 
                handleInvalid="skip"
            )
            df_vec = assembler.transform(df_enriched)
            
            # 2. Calcular Media y Covarianza por Sector (CNAE) - DISTRIBUIDO
            stats = df_vec.groupBy("cnae").agg(
                Summarizer.mean(F.col("features")).alias("sector_mean"),
                Summarizer.covariance(F.col("features")).alias("sector_cov")
            )
            
            # 3. Join con estadísticas del sector
            df_with_stats = df_vec.join(
                F.broadcast(stats),  # Broadcast para eficiencia
                on="cnae", 
                how="left"
            )
            
            # 4. Calcular distancia con Pandas UDF
            n_features = len(feature_cols)
            
            @pandas_udf(DoubleType())
            def mahalanobis_udf(
                features: pd.Series, 
                mean: pd.Series, 
                cov: pd.Series
            ) -> pd.Series:
                """Calcula distancia Mahalanobis vectorizadamente."""
                results = []
                for f, m, c in zip(features, mean, cov):
                    if f is None or m is None or c is None:
                        results.append(None)
                        continue
                    try:
                        # Convertir vectores de Spark a arrays numpy
                        point = np.array(f.toArray())
                        centroid = np.array(m.toArray())
                        cov_matrix = np.array(c.toArray()).reshape(n_features, n_features)
                        
                        # Regularización para evitar matriz singular
                        cov_matrix += np.eye(n_features) * 1e-6
                        cov_inv = np.linalg.inv(cov_matrix)
                        
                        # Distancia Mahalanobis: sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))
                        diff = point - centroid
                        dist = float(np.sqrt(diff @ cov_inv @ diff))
                        results.append(dist)
                    except Exception:
                        results.append(None)
                return pd.Series(results)
            
            # Aplicar UDF
            df_result = df_with_stats.withColumn(
                "mahalanobis_distance",
                mahalanobis_udf(
                    F.col("features"), 
                    F.col("sector_mean"), 
                    F.col("sector_cov")
                )
            )
            
            # Limpiar columnas temporales
            df_result = df_result.drop("features", "sector_mean", "sector_cov")
            
            # Flag para empresas muy desviadas (>3 desviaciones típicas)
            df_result = df_result.withColumn(
                'flag_sectoral_outlier',
                F.when(
                    F.col('mahalanobis_distance') > 3.0,
                    F.lit(1)
                ).otherwise(F.lit(0))
            )
            
            self.logger.info("Distancias Mahalanobis calculadas (modo distribuido)")
            return df_result
            
        except Exception as e:
            self.logger.error(f"Error en cálculo Mahalanobis distribuido: {str(e)}")
            self.logger.info("Fallback a método tradicional...")
            return self.calc_mahalanobis_distance(df, feature_cols)



# =============================================================================
# CLASE 5: FraudDetectorModel (MLflow)
# =============================================================================

class FraudDetectorModel:
    """
    Motor de detección de anomalías basado en Isolation Forest.
    
    VALOR FORENSE: El aprendizaje no supervisado permite detectar patrones
    de fraude sin necesidad de casos etiquetados previamente. Isolation Forest
    es especialmente efectivo porque:
    - Aísla anomalías en pocas particiones (empresas fraudulentas son "raras")
    - Robusto ante alta dimensionalidad
    - No asume distribución normal de los datos
    """
    
    def __init__(self, spark: SparkSession, config: FraudHunterConfig = None):
        self.spark = spark
        self.config = config or FraudHunterConfig()
        self.model = None
        self.scaler = None
        self.logger = logging.getLogger(f"{__name__}.FraudDetectorModel")
    
    def train_predict(
        self, 
        features_df: DataFrame,
        feature_cols: List[str] = None
    ) -> DataFrame:
        """
        Entrena Isolation Forest y genera predicciones.
        
        VALOR FORENSE: El modelo asigna un score de anomalía a cada empresa.
        Scores más negativos (cercanos a -1) indican empresas más anómalas.
        El ensemble de árboles "aísla" observaciones que requieren menos
        particiones para separarse, identificándolas como outliers.
        
        Args:
            features_df: DataFrame con features calculadas
            feature_cols: Columnas a usar como features (default: auto-detectadas)
        
        Returns:
            DataFrame con columnas 'fraud_score' y 'anomaly_label' añadidas
        """
        if feature_cols is None:
            # Usar todas las features numéricas calculadas
            feature_cols = [
                'cobertura_ventas', 'accruals_ratio', 'tasa_interes_implicita',
                'productividad_laboral', 'pct_numeros_redondos', 'benford_divergence',
                'mahalanobis_distance', 'flag_incoherencia_logistica',
                'flag_hidden_debt', 'flag_empresa_pantalla', 'flag_numeros_redondos',
                'flag_benford_anomaly', 'flag_sectoral_outlier'
            ]
        
        # Filtrar solo columnas que existen
        existing_cols = [c for c in feature_cols if c in features_df.columns]
        
        if len(existing_cols) < 3:
            self.logger.error("Insuficientes features para entrenar modelo")
            return features_df.withColumn(
                'fraud_score', F.lit(None).cast(DoubleType())
            ).withColumn(
                'anomaly_label', F.lit(None).cast(IntegerType())
            )
        
        try:
            # Convertir a Pandas para scikit-learn
            pdf = features_df.select('nif', *existing_cols).toPandas()
            
            # Preparar matriz de features
            X = pdf[existing_cols].copy()
            X = X.fillna(X.median())  # Imputar NaNs con mediana
            
            # Escalar con RobustScaler (resistente a outliers extremos)
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Entrenar Isolation Forest
            self.model = IsolationForest(
                contamination='auto',  # Estima automáticamente la proporción
                n_estimators=100,
                max_samples='auto',
                random_state=42,
                n_jobs=-1
            )
            
            # Fit y predict
            pdf['anomaly_label'] = self.model.fit_predict(X_scaled)
            pdf['fraud_score'] = self.model.decision_function(X_scaled)
            
            # Normalizar score a [0, 1] donde 1 = más sospechoso
            min_score = pdf['fraud_score'].min()
            max_score = pdf['fraud_score'].max()
            if max_score != min_score:
                pdf['fraud_score_normalized'] = 1 - (
                    (pdf['fraud_score'] - min_score) / (max_score - min_score)
                )
            else:
                pdf['fraud_score_normalized'] = 0.5
            
            # Registrar en MLflow si disponible
            if MLFLOW_AVAILABLE:
                self._log_to_mlflow(X_scaled, pdf)
            
            # Convertir resultado a Spark DataFrame
            result_pdf = pdf[['nif', 'fraud_score', 'fraud_score_normalized', 'anomaly_label']]
            result_df = self.spark.createDataFrame(result_pdf)
            
            # Join con DataFrame original
            final_df = features_df.join(result_df, on='nif', how='left')
            
            # Estadísticas
            n_anomalies = pdf[pdf['anomaly_label'] == -1].shape[0]
            self.logger.info(
                f"Modelo entrenado. Anomalías detectadas: {n_anomalies} "
                f"({100*n_anomalies/len(pdf):.2f}%)"
            )
            
            return final_df
            
        except Exception as e:
            self.logger.error(f"Error en entrenamiento: {str(e)}")
            return features_df.withColumn(
                'fraud_score', F.lit(None).cast(DoubleType())
            ).withColumn(
                'anomaly_label', F.lit(None).cast(IntegerType())
            )
    
    def _log_to_mlflow(self, X_scaled: np.ndarray, pdf: pd.DataFrame) -> None:
        """Registra métricas y modelo en MLflow."""
        try:
            with mlflow.start_run(run_name="fraud_hunter_isolation_forest"):
                # Log parámetros
                mlflow.log_param("contamination", "auto")
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("n_features", X_scaled.shape[1])
                mlflow.log_param("n_samples", X_scaled.shape[0])
                
                # Log métricas
                n_anomalies = (pdf['anomaly_label'] == -1).sum()
                mlflow.log_metric("n_anomalies", n_anomalies)
                mlflow.log_metric("anomaly_rate", n_anomalies / len(pdf))
                mlflow.log_metric("mean_fraud_score", pdf['fraud_score'].mean())
                mlflow.log_metric("std_fraud_score", pdf['fraud_score'].std())
                
                # Log modelo
                mlflow.sklearn.log_model(self.model, "isolation_forest_model")
                
                self.logger.info("Modelo registrado en MLflow")
                
        except Exception as e:
            self.logger.warning(f"Error registrando en MLflow: {str(e)}")


# =============================================================================
# GENERADOR DE DATOS DUMMY (Para demostración)
# =============================================================================

class DummyDataGenerator:
    """Genera datos de prueba para demostración."""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
    
    def generate_m200(self, n_companies: int = 100) -> DataFrame:
        """Genera datos dummy del Modelo 200."""
        np.random.seed(42)
        cnaes = ['4511', '4719', '6201', '6831', '4941']
        
        data = []
        for i in range(n_companies):
            nif = f"B{str(i).zfill(7)}A"
            is_suspicious = np.random.random() < 0.1  # 10% sospechosos
            
            ventas = np.random.uniform(100000, 10000000)
            if is_suspicious:
                # Empresas sospechosas: alto margen pero poco flujo caja
                resultado = ventas * np.random.uniform(0.15, 0.25)
                flujo_caja = resultado * np.random.uniform(0.1, 0.3)
                gastos_personal = ventas * np.random.uniform(0.001, 0.01)
            else:
                resultado = ventas * np.random.uniform(0.02, 0.08)
                flujo_caja = resultado * np.random.uniform(0.7, 1.1)
                gastos_personal = ventas * np.random.uniform(0.1, 0.3)
            
            data.append({
                'nif': nif,
                'ejercicio': 2024,
                'cnae': np.random.choice(cnaes),
                'ventas_netas': ventas,
                'resultado_neto': resultado,
                'activo_total': ventas * np.random.uniform(0.5, 1.5),
                'patrimonio_neto': ventas * np.random.uniform(0.2, 0.5),
                'deuda_bancaria': ventas * np.random.uniform(0.1, 0.4),
                'gastos_personal': gastos_personal,
                'gastos_transporte': 0 if is_suspicious else ventas * 0.02,
                'intereses_pagados': ventas * np.random.uniform(0.01, 0.05),
                'flujo_caja_operativo': flujo_caja,
                'amortizaciones': ventas * np.random.uniform(0.02, 0.05),
            })
        
        return self.spark.createDataFrame(pd.DataFrame(data))
    
    def generate_m347(self, n_companies: int = 100) -> DataFrame:
        """Genera datos dummy del Modelo 347."""
        np.random.seed(43)
        
        data = []
        for i in range(n_companies):
            nif_declarante = f"B{str(i).zfill(7)}A"
            n_contrapartes = np.random.randint(5, 30)
            is_suspicious = np.random.random() < 0.1
            
            for j in range(n_contrapartes):
                nif_contraparte = f"B{str(np.random.randint(0, n_companies)).zfill(7)}A"
                
                if is_suspicious:
                    # Importes redondos para empresas sospechosas
                    importe = np.random.choice([10000, 50000, 100000, 200000])
                else:
                    importe = np.random.uniform(5000, 500000)
                
                data.append({
                    'nif_declarante': nif_declarante,
                    'ejercicio': 2024,
                    'nif_contraparte': nif_contraparte,
                    'importe_total': float(importe),
                    'clave_operacion': np.random.choice(['A', 'B']),
                })
        
        return self.spark.createDataFrame(pd.DataFrame(data))
    
    def generate_m349(self, n_companies: int = 100) -> DataFrame:
        """Genera datos dummy del Modelo 349."""
        np.random.seed(44)
        paises = ['DE', 'FR', 'IT', 'PT', 'NL']
        
        data = []
        # Solo 30% de empresas tienen operaciones UE
        for i in range(int(n_companies * 0.3)):
            nif_declarante = f"B{str(i).zfill(7)}A"
            n_operaciones = np.random.randint(1, 10)
            
            for j in range(n_operaciones):
                data.append({
                    'nif_declarante': nif_declarante,
                    'ejercicio': 2024,
                    'nif_operador_ue': f"{np.random.choice(paises)}{str(np.random.randint(100000, 999999))}",
                    'pais_ue': np.random.choice(paises),
                    'importe': np.random.uniform(10000, 1000000),
                    'clave_operacion': np.random.choice(['E', 'A']),
                })
        
        return self.spark.createDataFrame(pd.DataFrame(data))


# =============================================================================
# ORQUESTADOR PRINCIPAL
# =============================================================================

def run_fraud_hunter_pipeline(
    spark: SparkSession,
    paths_config: Dict[str, str] = None,
    use_dummy_data: bool = True
) -> DataFrame:
    """
    Ejecuta el pipeline completo de FraudHunter.
    
    Args:
        spark: SparkSession activa
        paths_config: Rutas a los datos (None para usar dummy)
        use_dummy_data: Si True, genera datos de demostración
    
    Returns:
        DataFrame con scores de fraude para cada empresa
    """
    logger.info("=" * 60)
    logger.info("FRAUDHUNTER CORE - Iniciando pipeline de detección")
    logger.info("=" * 60)
    
    config = FraudHunterConfig()
    
    # Paso 1: Cargar datos
    if use_dummy_data:
        logger.info("Generando datos dummy para demostración...")
        dummy_gen = DummyDataGenerator(spark)
        df_m200 = dummy_gen.generate_m200(100)
        df_m347 = dummy_gen.generate_m347(100)
        df_m349 = dummy_gen.generate_m349(100)
    else:
        loader = TaxDataLoader(spark)
        df_m200, df_m347, df_m349 = loader.load_and_clean_data(paths_config)
    
    # Paso 2: Feature Engineering
    logger.info("Construyendo features forenses...")
    feature_engineer = ForensicFeatureEngineer(spark, config)
    df_features = feature_engineer.build_abt(df_m200, df_m347, df_m349)
    
    # Paso 3: Análisis de grafos (si disponible)
    logger.info("Analizando red de transacciones...")
    graph_analyzer = GraphNetworkAnalyzer(spark)
    
    if GRAPHFRAMES_AVAILABLE:
        graph_analyzer.build_graph(df_m347)
        df_pagerank = graph_analyzer.calc_pagerank()
        df_circularity = graph_analyzer.detect_circularity()
        
        # Unir métricas de grafo
        if df_pagerank.count() > 0:
            df_features = df_features.join(
                df_pagerank.withColumnRenamed('id', 'nif'),
                on='nif',
                how='left'
            )
    else:
        logger.warning("Análisis de grafos omitido (GraphFrames no disponible)")
    
    # Paso 4: Benchmarking sectorial
    logger.info("Calculando distancias sectoriales...")
    benchmarker = SectoralBenchmarker(spark)
    df_features = benchmarker.calc_mahalanobis_distance(df_features)
    
    # Paso 5: Modelo de detección
    logger.info("Entrenando modelo de detección de anomalías...")
    detector = FraudDetectorModel(spark, config)
    df_final = detector.train_predict(df_features)
    
    # Resumen final
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETADO")
    logger.info(f"Total empresas analizadas: {df_final.count()}")
    
    anomalies = df_final.filter(F.col('anomaly_label') == -1).count()
    logger.info(f"Anomalías detectadas: {anomalies}")
    logger.info("=" * 60)
    
    return df_final


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    """
    Demostración del pipeline FraudHunter con datos dummy.
    
    En Databricks, este bloque se reemplazaría por un notebook que:
    1. Configure las rutas a Delta Tables reales
    2. Ejecute el pipeline con use_dummy_data=False
    3. Guarde resultados en Delta Lake
    4. Configure alertas basadas en fraud_score
    """
    
    # Crear SparkSession local para demo
    spark = SparkSession.builder \
        .appName("FraudHunter-Demo") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    try:
        # Ejecutar pipeline con datos dummy
        df_results = run_fraud_hunter_pipeline(
            spark=spark,
            use_dummy_data=True
        )
        
        # Mostrar top 10 empresas más sospechosas
        print("\n" + "=" * 60)
        print("TOP 10 EMPRESAS MÁS SOSPECHOSAS")
        print("=" * 60)
        
        df_results.select(
            'nif', 'cnae', 'ventas_netas', 'fraud_score_normalized',
            'anomaly_label', 'flag_incoherencia_logistica',
            'flag_empresa_pantalla', 'flag_benford_anomaly'
        ).filter(
            F.col('anomaly_label') == -1
        ).orderBy(
            F.col('fraud_score_normalized').desc()
        ).show(10, truncate=False)
        
        # Estadísticas de flags
        print("\n" + "=" * 60)
        print("RESUMEN DE FLAGS DE RIESGO")
        print("=" * 60)
        
        flag_cols = [c for c in df_results.columns if c.startswith('flag_')]
        for flag in flag_cols:
            count = df_results.filter(F.col(flag) == 1).count()
            print(f"  {flag}: {count} empresas")
        
    except Exception as e:
        logger.error(f"Error en ejecución: {str(e)}")
        raise
    finally:
        spark.stop()
