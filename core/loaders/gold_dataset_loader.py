"""
FraudHunter Enterprise OS - Gold Dataset Loader
==============================================
Cargador para los resultados finales del análisis (Gold Layer).
Funde datos de scores, flags y auditoría en una vista unificada para el Cockpit.
"""

import pandas as pd
import numpy as np
import streamlit as st
from .base_loader import BaseLoader

class GoldDatasetLoader(BaseLoader):
    def __init__(self, supabase_client=None):
        super().__init__(supabase_client)
        
    @property
    def required_columns(self):
        return ['nif', 'final_score', 'benford_kl_divergence']
        
    @property
    def table_name(self):
        return "gold_analysis"
        
    def validate(self, df):
        return df.dropna(subset=['nif'])

    def load(self):
        """
        Carga el dataset Gold. 
        En producción: Query a Supabase 'gold_fraud_analysis' view.
        En dev: Genera datos sintéticos con la nueva estructura FraudHunter OS.
        """
        # Intentar cargar desde session_state si ya existe un análisis previo
        if 'df_results' in st.session_state:
            return st.session_state.df_results
            
        # Generar datos Mock para demostración del Cockpit
        return self._generate_mock_gold()

    def _generate_mock_gold(self):
        """Genera un dataset Gold sintético con 100 empresas para el Cockpit."""
        np.random.seed(42)
        n = 100
        
        nifs = [f"{np.random.choice(['A', 'B', 'W'])}{np.random.randint(10000000, 99999999)}" for _ in range(n)]
        nombres = [f"EMPRESA FORENSIC {i:03d} SL" for i in range(n)]
        
        df = pd.DataFrame({
            'nif': nifs,
            'razon_social': nombres,
            'cnae': np.random.choice(['4110', '4673', '6201', '5510', '4321'], n),
            'final_score': np.random.uniform(0.1, 0.95, n),
            'tax_score': np.random.uniform(0.1, 0.9, n),
            'network_score': np.random.uniform(0.1, 0.7, n),
            'benford_kl_divergence': np.random.exponential(0.05, n),
            'cifra_negocios': np.random.lognormal(12, 1.5, n),
            'efectivo_tesoreria': np.random.lognormal(10, 1.2, n),
            'stage': np.random.choice([1, 2, 3], n, p=[0.8, 0.15, 0.05]),
            'leverage': np.random.uniform(0, 0.8, n)
        })
        
        # Clasificación de riesgo
        df['risk_level'] = df['final_score'].apply(lambda x: 'HIGH' if x > 0.7 else 'MEDIUM' if x > 0.4 else 'LOW')
        
        # Sort by score
        df = df.sort_values('final_score', ascending=False)
        return df
