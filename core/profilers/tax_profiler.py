"""
Tax Profiler
============
Analiza densidad y calidad de datos fiscales antes de ejecutar algoritmos.

Por qué es crítico:
- Evita ejecutar algoritmos costosos en datos escasos
- Identifica empresas con datos incompletos (incertidumbre)
- Optimiza costes de cómputo
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class TaxProfileStats:
    """Estadísticas del perfil fiscal."""
    nif: str
    rows_m200: int
    rows_m347: int
    rows_m349: int
    years_available: int
    data_completeness: float  # 0-1
    is_sufficient: bool
    missing_models: list


class TaxProfiler:
    """
    Perfila la densidad y calidad de datos fiscales por empresa.
    
    Uso:
        profiler = TaxProfiler(min_years=2, min_m347_rows=5)
        stats = profiler.profile(nif='B12345678', df_m200=..., df_m347=...)
    """
    
    def __init__(
        self,
        min_years: int = 2,
        min_m347_rows: int = 5,
        min_m349_rows: int = 1
    ):
        """
        Args:
            min_years: Mínimo de años de M200 para análisis completo
            min_m347_rows: Mínimo de operaciones M347
            min_m349_rows: Mínimo de operaciones M349
        """
        self.min_years = min_years
        self.min_m347_rows = min_m347_rows
        self.min_m349_rows = min_m349_rows
    
    def profile(
        self,
        nif: str,
        df_m200: Optional[pd.DataFrame] = None,
        df_m347: Optional[pd.DataFrame] = None,
        df_m349: Optional[pd.DataFrame] = None
    ) -> TaxProfileStats:
        """
        Genera perfil de densidad de datos para una empresa.
        
        Returns:
            TaxProfileStats con métricas de calidad
        """
        rows_m200 = 0
        rows_m347 = 0
        rows_m349 = 0
        years = set()
        missing = []
        
        # M200
        if df_m200 is not None and not df_m200.empty:
            m200_company = df_m200[df_m200['nif'] == nif]
            rows_m200 = len(m200_company)
            if 'ejercicio' in m200_company.columns:
                years.update(m200_company['ejercicio'].unique())
        else:
            missing.append('M200')
        
        # M347
        if df_m347 is not None and not df_m347.empty:
            m347_company = df_m347[df_m347['nif_declarante'] == nif]
            rows_m347 = len(m347_company)
        else:
            missing.append('M347')
        
        # M349
        if df_m349 is not None and not df_m349.empty:
            m349_company = df_m349[df_m349['nif_declarante'] == nif]
            rows_m349 = len(m349_company)
        else:
            missing.append('M349')
        
        # Calcular completitud
        completeness = self._calculate_completeness(
            rows_m200, rows_m347, rows_m349, len(years)
        )
        
        # Determinar si es suficiente para análisis completo
        is_sufficient = (
            len(years) >= self.min_years and
            rows_m347 >= self.min_m347_rows
        )
        
        return TaxProfileStats(
            nif=nif,
            rows_m200=rows_m200,
            rows_m347=rows_m347,
            rows_m349=rows_m349,
            years_available=len(years),
            data_completeness=completeness,
            is_sufficient=is_sufficient,
            missing_models=missing
        )
    
    def profile_all(
        self,
        nifs: list,
        df_m200: Optional[pd.DataFrame] = None,
        df_m347: Optional[pd.DataFrame] = None,
        df_m349: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Perfila múltiples empresas a la vez.
        
        Returns:
            DataFrame con perfil de cada empresa
        """
        records = []
        
        for nif in nifs:
            stats = self.profile(nif, df_m200, df_m347, df_m349)
            records.append({
                'nif': stats.nif,
                'rows_m200': stats.rows_m200,
                'rows_m347': stats.rows_m347,
                'rows_m349': stats.rows_m349,
                'years_available': stats.years_available,
                'data_completeness': stats.data_completeness,
                'is_sufficient': stats.is_sufficient
            })
        
        return pd.DataFrame(records)
    
    def _calculate_completeness(
        self,
        rows_m200: int,
        rows_m347: int,
        rows_m349: int,
        years: int
    ) -> float:
        """
        Calcula score de completitud de datos 0-1.
        
        Pesos:
        - M200 (años): 40%
        - M347 (operaciones): 40%
        - M349 (intracomunitarias): 20%
        """
        score = 0.0
        
        # M200 years score
        years_score = min(years / self.min_years, 1.0)
        score += years_score * 0.4
        
        # M347 rows score
        m347_score = min(rows_m347 / self.min_m347_rows, 1.0)
        score += m347_score * 0.4
        
        # M349 rows score
        m349_score = min(rows_m349 / max(self.min_m349_rows, 1), 1.0)
        score += m349_score * 0.2
        
        return round(score, 2)
