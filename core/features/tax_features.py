"""
Tax Feature Engine
==================
Genera features de solvencia fiscal para el ScoreComposer.
"""

import pandas as pd
import numpy as np


class TaxFeatureEngine:
    """
    Motor de features fiscales (M200 + M347 agregado).
    
    Genera:
    - interest_burden, leverage, cash_buffer
    - Flags de estres financiero
    """
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma DataFrame con features fiscales.
        """
        df = df.copy()
        
        # Asegurar columnas numericas
        for col in ['cifra_negocios', 'gastos_financieros', 'total_activo',
                    'deuda_bancaria_lp', 'deuda_bancaria_cp', 'efectivo_tesoreria',
                    'resultado_ejercicio']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Interest Burden = Intereses / Ventas
        df['interest_burden'] = np.where(
            df['cifra_negocios'] > 0,
            df.get('gastos_financieros', 0) / df['cifra_negocios'],
            0
        )
        
        # Deuda total
        df['deuda_bancaria_total'] = (
            df.get('deuda_bancaria_lp', 0) + 
            df.get('deuda_bancaria_cp', 0)
        )
        
        # Leverage = Deuda / Activo
        df['leverage'] = np.where(
            df['total_activo'] > 0,
            df['deuda_bancaria_total'] / df['total_activo'],
            0
        )
        
        # Cash Buffer = Caja / (Intereses + 1)
        df['cash_buffer'] = np.where(
            df.get('gastos_financieros', 0) > 0,
            df.get('efectivo_tesoreria', 0) / (df['gastos_financieros'] + 1),
            df.get('efectivo_tesoreria', 0)
        )
        
        # Margen neto
        df['margen_neto'] = np.where(
            df['cifra_negocios'] > 0,
            df.get('resultado_ejercicio', 0) / df['cifra_negocios'],
            0
        )
        
        # Rotacion activos
        df['rotacion_activos'] = np.where(
            df['total_activo'] > 0,
            df['cifra_negocios'] / df['total_activo'],
            0
        )
        
        # Flags
        df['flag_stress_liquidez'] = (
            (df['cash_buffer'] < 1.0) & 
            (df['leverage'] > 0.7)
        )
        
        return df
