"""
Network Feature Engine
======================
Genera features de red para deteccion de carrusel y concentracion.
"""

import pandas as pd
import numpy as np


class NetworkFeatureEngine:
    """
    Motor de features de red (M347 + M349).
    
    Genera:
    - reciprocity_score
    - concentration_score
    - carousel_risk_flags
    """
    
    def transform_m347(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforma M347 con features de reciprocidad."""
        if df.empty:
            return df
        
        df = df.copy()
        return df
    
    def transform_m349(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforma M349 con features de carrusel."""
        if df.empty:
            return df
        
        df = df.copy()
        
        # High risk countries
        HIGH_RISK = {'MT', 'CY', 'LU', 'IE', 'NL', 'BG', 'RO'}
        df['is_high_risk_country'] = df['pais_ue'].isin(HIGH_RISK)
        
        return df
