"""
M349 Loader
===========
Carga el Modelo 349 (Operaciones Intracomunitarias).

Por qué crítico:
- Detección de fraude carrusel (MTIC - Missing Trader Intra-Community)
- Países de alto riesgo fiscal
- Patrones de triangulación
"""

import time
from typing import List, Optional
import pandas as pd
import numpy as np

from .base_loader import BaseLoader, LoaderStats


# Países de alto riesgo para fraude carrusel
HIGH_RISK_COUNTRIES = {'MT', 'CY', 'LU', 'IE', 'NL', 'BG', 'RO', 'LV', 'LT', 'EE'}


class M349Loader(BaseLoader):
    """
    Loader para Modelo 349 (Operaciones Intracomunitarias).
    
    Uso:
        loader = M349Loader(supabase_client)
        df = loader.load(ejercicio=2023)
        df_risk = loader.calculate_carousel_indicators(df)
    """
    
    @property
    def required_columns(self) -> List[str]:
        return [
            'nif_declarante',
            'nif_operador_ue',
            'pais_ue',
            'importe',
            'clave_operacion',
        ]
    
    @property
    def table_name(self) -> str:
        return "modelo_349"
    
    def load(
        self,
        ejercicio: Optional[int] = None,
        pais_ue: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Carga datos del Modelo 349.
        
        Args:
            ejercicio: Filtrar por año
            pais_ue: Filtrar por país UE
            limit: Limitar registros
        """
        start_time = time.time()
        
        if self.client is None:
            raise ValueError("Supabase client not configured")
        
        query = self.client.table(self.table_name).select("*")
        
        if ejercicio:
            query = query.eq("ejercicio", ejercicio)
        
        if pais_ue:
            query = query.eq("pais_ue", pais_ue)
        
        if limit:
            query = query.limit(limit)
        
        response = query.execute()
        
        if not response.data:
            return pd.DataFrame()
        
        df = pd.DataFrame(response.data)
        df = self.validate(df)
        
        load_time = time.time() - start_time
        found, missing = self._validate_required_columns(df)
        
        self._stats = LoaderStats(
            rows_loaded=len(response.data),
            rows_valid=len(df),
            rows_invalid=len(response.data) - len(df),
            load_time_seconds=load_time,
            source=f"supabase:{self.table_name}",
            columns_found=found,
            columns_missing=missing
        )
        
        return df
    
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida datos del M349."""
        if df.empty:
            return df
        
        df = df.copy()
        
        # NIFs y país no vacíos
        df = df.dropna(subset=['nif_declarante', 'pais_ue'])
        
        # Importe numérico
        if 'importe' in df.columns:
            df['importe'] = pd.to_numeric(df['importe'], errors='coerce').fillna(0)
        
        # País UE válido (2 caracteres)
        df = df[df['pais_ue'].str.len() == 2]
        
        # Marcar países de alto riesgo
        df['is_high_risk_country'] = df['pais_ue'].isin(HIGH_RISK_COUNTRIES)
        
        return df
    
    def aggregate_by_country_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega operaciones por país y período.
        
        Por qué:
        - Prepara datos para Gold 3 (company_ue_period)
        - Permite detectar patrones temporales
        """
        if df.empty:
            return df
        
        # Clasificar por tipo de operación
        df = df.copy()
        
        # E, M, H = Entregas; A = Adquisiciones; S = Servicios prestados; I = Servicios recibidos
        df['is_entrega'] = df['clave_operacion'].isin(['E', 'M', 'H', 'T'])
        df['is_adquisicion'] = df['clave_operacion'] == 'A'
        df['is_servicio_prestado'] = df['clave_operacion'] == 'S'
        df['is_servicio_recibido'] = df['clave_operacion'] == 'I'
        df['is_triangular'] = df['clave_operacion'] == 'T'
        
        # Agregar
        agg = df.groupby(['nif_declarante', 'pais_ue', 'ejercicio', 'periodo']).agg({
            'importe': 'sum',
            'is_entrega': lambda x: (x * df.loc[x.index, 'importe']).sum(),
            'is_adquisicion': lambda x: (x * df.loc[x.index, 'importe']).sum(),
            'is_triangular': 'any',
            'is_high_risk_country': 'first',
        }).reset_index()
        
        agg.columns = [
            'nif_declarante', 'pais_ue', 'ejercicio', 'periodo',
            'importe_total', 'total_entregas', 'total_adquisiciones',
            'is_triangular_pattern', 'is_high_risk_country'
        ]
        
        # Balance neto
        agg['balance_neto'] = agg['total_entregas'] - agg['total_adquisiciones']
        
        return agg
    
    def calculate_carousel_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula indicadores de fraude carrusel (MTIC).
        
        Patrones sospechosos:
        1. Alto volumen con países de alto riesgo
        2. Operaciones triangulares (clave T)
        3. Balance muy desequilibrado (solo entregas o solo adquisiciones)
        4. Cambios bruscos entre períodos
        """
        if df.empty:
            return df
        
        agg = self.aggregate_by_country_period(df)
        
        # Flag: Riesgo carrusel
        agg['flag_carousel_risk'] = (
            (agg['is_high_risk_country']) & 
            (agg['importe_total'] > 100000) &
            (
                (agg['is_triangular_pattern']) |
                (agg['balance_neto'].abs() > agg['importe_total'] * 0.8)
            )
        )
        
        # Flag: Anomalía de balance (solo entregas sin adquisiciones o viceversa)
        agg['flag_balance_anomaly'] = (
            (agg['total_entregas'] > 0) & (agg['total_adquisiciones'] == 0) |
            (agg['total_adquisiciones'] > 0) & (agg['total_entregas'] == 0)
        ) & (agg['importe_total'] > 50000)
        
        return agg
