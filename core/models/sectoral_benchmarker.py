"""
Sectoral Benchmarker
====================
Comparación de empresas con su sector usando Mediana/MAD.

Por qué Mediana/MAD en lugar de Media/StdDev:
- Los datos financieros tienen colas largas (outliers extremos)
- La media es muy sensible a outliers (Elon Musk en un barrio)
- MAD (Median Absolute Deviation) es robusto a outliers
- El Z-score con MAD detecta anomalías reales, no ruido
"""

from typing import List, Optional, Dict, Tuple
import pandas as pd
import numpy as np
from scipy import stats


class SectoralBenchmarker:
    """
    Compara empresas con sus pares sectoriales usando estadísticas robustas.
    
    Uso:
        benchmarker = SectoralBenchmarker()
        df = benchmarker.fit_transform(df, sector_col='cnae', metrics=['margen_neto', 'leverage'])
    """
    
    def __init__(self, min_sector_size: int = 5):
        """
        Args:
            min_sector_size: Mínimo de empresas por sector para calcular benchmark
        """
        self.min_sector_size = min_sector_size
        self.sector_stats: Dict[str, Dict] = {}
    
    def fit(
        self, 
        df: pd.DataFrame, 
        sector_col: str,
        metrics: List[str]
    ) -> 'SectoralBenchmarker':
        """
        Calcula estadísticas por sector.
        
        Args:
            df: DataFrame con empresas
            sector_col: Columna de sector (CNAE, industria, etc.)
            metrics: Lista de métricas a comparar
        """
        self.sector_stats = {}
        
        for sector in df[sector_col].unique():
            sector_data = df[df[sector_col] == sector]
            
            if len(sector_data) < self.min_sector_size:
                continue
            
            self.sector_stats[sector] = {}
            
            for metric in metrics:
                if metric not in sector_data.columns:
                    continue
                
                values = sector_data[metric].dropna()
                
                if len(values) < self.min_sector_size:
                    continue
                
                # Mediana (robusto al centro)
                median = values.median()
                
                # MAD = Median Absolute Deviation (robusto a dispersión)
                # Por qué 1.4826: Factor de corrección para equivalencia con StdDev en distribución normal
                mad = np.median(np.abs(values - median)) * 1.4826
                
                # Evitar MAD = 0 (cuando todos los valores son iguales)
                mad = max(mad, 1e-6)
                
                # Percentiles para contexto
                p25 = values.quantile(0.25)
                p75 = values.quantile(0.75)
                
                self.sector_stats[sector][metric] = {
                    'median': median,
                    'mad': mad,
                    'p25': p25,
                    'p75': p75,
                    'n': len(values)
                }
        
        return self
    
    def transform(
        self, 
        df: pd.DataFrame, 
        sector_col: str,
        metrics: List[str]
    ) -> pd.DataFrame:
        """
        Aplica benchmarks calculados a un DataFrame.
        
        Genera columnas:
        - {metric}_sector_median
        - {metric}_sector_mad
        - {metric}_z_score: Z-score robusto (cuántas MADs del centro)
        - {metric}_flag_outlier: True si |z_score| > 3
        """
        if not self.sector_stats:
            raise ValueError("Debes llamar fit() primero")
        
        df = df.copy()
        
        for metric in metrics:
            # Inicializar columnas
            df[f'{metric}_sector_median'] = np.nan
            df[f'{metric}_sector_mad'] = np.nan
            df[f'{metric}_z_score'] = np.nan
            df[f'{metric}_flag_outlier'] = False
            
            for sector, stats in self.sector_stats.items():
                if metric not in stats:
                    continue
                
                sector_mask = df[sector_col] == sector
                median = stats[metric]['median']
                mad = stats[metric]['mad']
                
                df.loc[sector_mask, f'{metric}_sector_median'] = median
                df.loc[sector_mask, f'{metric}_sector_mad'] = mad
                
                # Z-score robusto
                # Interpretación: Cuántas desviaciones (MAD) está del centro (mediana)
                z_score = (df.loc[sector_mask, metric] - median) / mad
                df.loc[sector_mask, f'{metric}_z_score'] = z_score
                
                # Outlier si |z_score| > 3 (equivalente a 3 sigma en distribución normal)
                df.loc[sector_mask, f'{metric}_flag_outlier'] = np.abs(z_score) > 3.0
        
        return df
    
    def fit_transform(
        self, 
        df: pd.DataFrame, 
        sector_col: str,
        metrics: List[str]
    ) -> pd.DataFrame:
        """Fit y transform en un solo paso."""
        self.fit(df, sector_col, metrics)
        return self.transform(df, sector_col, metrics)
    
    def get_sector_summary(self) -> pd.DataFrame:
        """Retorna resumen de estadísticas por sector."""
        records = []
        
        for sector, metrics in self.sector_stats.items():
            for metric, stats in metrics.items():
                records.append({
                    'sector': sector,
                    'metric': metric,
                    'median': stats['median'],
                    'mad': stats['mad'],
                    'p25': stats['p25'],
                    'p75': stats['p75'],
                    'n_companies': stats['n']
                })
        
        return pd.DataFrame(records)
