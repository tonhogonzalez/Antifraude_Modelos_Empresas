"""
M347 Loader
===========
Carga el Modelo 347 (Operaciones con terceros).

Por qué granular:
- Permite detectar patrones de concentración
- Habilita análisis de reciprocidad (A→B y B→A)
- Base para detección de fraude carrusel
"""

import time
from typing import List, Optional
import pandas as pd
import numpy as np

from .base_loader import BaseLoader, LoaderStats


class M347Loader(BaseLoader):
    """
    Loader para Modelo 347 granular (operación por operación).
    
    Uso:
        loader = M347Loader(supabase_client)
        df = loader.load(ejercicio=2023)
        df_agg = loader.aggregate_by_counterparty(df)
    """
    
    @property
    def required_columns(self) -> List[str]:
        return [
            'nif_declarante',
            'nif_contraparte',
            'importe_total',
            'ejercicio',
        ]
    
    @property
    def table_name(self) -> str:
        return "modelo_347"
    
    def load(
        self,
        ejercicio: Optional[int] = None,
        nif_declarante: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Carga datos del Modelo 347.
        
        Args:
            ejercicio: Filtrar por año
            nif_declarante: Filtrar por NIF declarante específico
            limit: Limitar registros
        
        Returns:
            DataFrame con operaciones M347
        """
        start_time = time.time()
        
        if self.client is None:
            raise ValueError("Supabase client not configured")
        
        query = self.client.table(self.table_name).select("*")
        
        if ejercicio:
            query = query.eq("ejercicio", ejercicio)
        
        if nif_declarante:
            query = query.eq("nif_declarante", nif_declarante)
        
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
        """Valida datos del M347."""
        if df.empty:
            return df
        
        df = df.copy()
        
        # NIFs no vacíos
        df = df.dropna(subset=['nif_declarante', 'nif_contraparte'])
        
        # Importes numéricos
        if 'importe_total' in df.columns:
            df['importe_total'] = pd.to_numeric(df['importe_total'], errors='coerce').fillna(0)
        
        # Filtrar importes válidos (> 3.005,06€ es el umbral legal)
        df = df[df['importe_total'].abs() > 0]
        
        return df
    
    def aggregate_by_counterparty(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega operaciones por contraparte.
        
        Por qué:
        - Calcula métricas de concentración
        - Prepara datos para Gold 2 (company_counterparty_year)
        """
        if df.empty:
            return df
        
        agg = df.groupby(['nif_declarante', 'nif_contraparte', 'ejercicio']).agg({
            'importe_total': ['sum', 'count', 'mean'],
        }).reset_index()
        
        # Flatten column names
        agg.columns = [
            'nif_declarante', 'nif_contraparte', 'ejercicio',
            'importe_total', 'num_operaciones', 'importe_promedio'
        ]
        
        return agg
    
    def calculate_reciprocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula métricas de reciprocidad (A→B vs B→A).
        
        Por qué:
        - La reciprocidad alta es indicador de fraude carrusel
        - Empresas normales no tienen flujos bidireccionales similares
        """
        if df.empty:
            return df
        
        # Agregar por contraparte primero
        agg = self.aggregate_by_counterparty(df)
        
        # Self-join para encontrar flujos inversos
        agg_inverse = agg.rename(columns={
            'nif_declarante': 'nif_contraparte_temp',
            'nif_contraparte': 'nif_declarante',
            'importe_total': 'importe_inverso'
        })
        agg_inverse['nif_contraparte'] = agg_inverse['nif_contraparte_temp']
        agg_inverse = agg_inverse[['nif_declarante', 'nif_contraparte', 'ejercicio', 'importe_inverso']]
        
        # Merge
        result = agg.merge(
            agg_inverse,
            on=['nif_declarante', 'nif_contraparte', 'ejercicio'],
            how='left'
        )
        
        result['importe_inverso'] = result['importe_inverso'].fillna(0)
        
        # Reciprocity ratio = min(A→B, B→A) / max(A→B, B→A)
        # 1.0 = perfectamente recíproco (sospechoso)
        # 0.0 = unidireccional (normal)
        max_importe = result[['importe_total', 'importe_inverso']].max(axis=1)
        min_importe = result[['importe_total', 'importe_inverso']].min(axis=1)
        
        result['reciprocity_ratio'] = np.where(
            max_importe > 0,
            min_importe / max_importe,
            0
        )
        
        # Normalize to 0-1 score (higher = more suspicious)
        result['reciprocity_score'] = result['reciprocity_ratio']
        
        # Flag
        result['flag_reciprocidad_sospechosa'] = result['reciprocity_score'] > 0.7
        
        return result
