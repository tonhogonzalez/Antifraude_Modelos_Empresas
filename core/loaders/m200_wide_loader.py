"""
M200 Wide Loader
================
Carga el Modelo 200 (Impuesto Sociedades) en formato ancho.

Por qué Wide vs EAV:
- El formato Wide (columnas por casilla) es más eficiente para análisis
- Evita pivots costosos en runtime
- Facilita el cálculo de ratios financieros
"""

import time
from typing import List, Optional
import pandas as pd
import numpy as np

from .base_loader import BaseLoader, LoaderStats


class M200WideLoader(BaseLoader):
    """
    Loader para Modelo 200 en formato ancho.
    
    Uso:
        loader = M200WideLoader(supabase_client)
        df = loader.load(ejercicio=2023)
    """
    
    @property
    def required_columns(self) -> List[str]:
        return [
            'nif',
            'ejercicio',
            'cifra_negocios',
            'resultado_ejercicio',
            'total_activo',
        ]
    
    @property
    def table_name(self) -> str:
        return "modelo_200"
    
    def load(
        self, 
        ejercicio: Optional[int] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Carga datos del Modelo 200 desde Supabase.
        
        Args:
            ejercicio: Filtrar por año fiscal
            limit: Limitar número de registros
        
        Returns:
            DataFrame con datos del M200
        """
        start_time = time.time()
        
        if self.client is None:
            raise ValueError("Supabase client not configured")
        
        # Build query
        query = self.client.table(self.table_name).select("*")
        
        if ejercicio:
            query = query.eq("ejercicio", ejercicio)
        
        if limit:
            query = query.limit(limit)
        
        # Execute
        response = query.execute()
        
        if not response.data:
            return pd.DataFrame()
        
        df = pd.DataFrame(response.data)
        
        # Validate
        df = self.validate(df)
        
        # Calculate stats
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
        """
        Valida y limpia datos del M200.
        
        Reglas:
        - NIF no puede estar vacío
        - cifra_negocios debe ser numérica
        - total_activo debe ser > 0 (empresas activas)
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Remove rows without NIF
        df = df.dropna(subset=['nif'])
        
        # Ensure numeric columns
        numeric_cols = [
            'cifra_negocios', 'resultado_ejercicio', 'total_activo',
            'patrimonio_neto', 'gastos_personal', 'efectivo_tesoreria',
            'deuda_bancaria_lp', 'deuda_bancaria_cp', 'gastos_financieros',
            'amortizaciones', 'aprovisionamientos', 'otros_gastos_expl'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Filter active companies (activo > 0)
        if 'total_activo' in df.columns:
            df = df[df['total_activo'] > 0]
        
        return df
    
    def calculate_enterprise_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula los nuevos ratios financieros del Enterprise OS.
        
        Por qué estos ratios:
        - interest_burden: El fraude en financiación se detecta por estrés de liquidez
        - leverage: Apalancamiento excesivo indica riesgo
        - cash_buffer: Capacidad de pago inmediata
        """
        df = df.copy()
        
        # Interest Burden = Intereses / Ventas
        # Why: Mide el peso de los intereses sobre la actividad
        df['interest_burden'] = np.where(
            df['cifra_negocios'] > 0,
            df['gastos_financieros'] / df['cifra_negocios'],
            0
        )
        
        # Leverage = Deuda Bancaria Total / Activo
        # Why: Apalancamiento excesivo es señal de riesgo
        df['deuda_bancaria_total'] = (
            df.get('deuda_bancaria_lp', 0) + 
            df.get('deuda_bancaria_cp', 0)
        )
        df['leverage'] = np.where(
            df['total_activo'] > 0,
            df['deuda_bancaria_total'] / df['total_activo'],
            0
        )
        
        # Cash Buffer = Caja / (Intereses + 1)
        # Why: Capacidad de pago inmediata sin recurrir a financiación
        df['cash_buffer'] = np.where(
            df['gastos_financieros'] > 0,
            df['efectivo_tesoreria'] / (df['gastos_financieros'] + 1),
            df['efectivo_tesoreria']  # Si no hay intereses, solo caja
        )
        
        # Ratios tradicionales mejorados
        df['margen_neto'] = np.where(
            df['cifra_negocios'] > 0,
            df['resultado_ejercicio'] / df['cifra_negocios'],
            0
        )
        
        df['rotacion_activos'] = np.where(
            df['total_activo'] > 0,
            df['cifra_negocios'] / df['total_activo'],
            0
        )
        
        df['ratio_endeudamiento'] = df['leverage']  # Alias
        
        return df
