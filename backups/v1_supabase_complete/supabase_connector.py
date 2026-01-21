#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
SUPABASE CONNECTOR - FraudHunter
================================================================================
Módulo centralizado para conectar con Supabase y cargar datos de modelos
tributarios (M200, M347, M349).

Características:
- Cache de cliente con @st.cache_resource
- Cache de datos con @st.cache_data (TTL 1 hora)
- Fallback a datos sintéticos si Supabase no disponible
- Manejo robusto de errores

Autor: FraudHunter AI System
================================================================================
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple

# Intentar importar Supabase
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

# Valores por defecto (se sobrescriben con secrets)
DEFAULT_SUPABASE_URL = "https://xqtcvtchzmuecssmwrxb.supabase.co"
DEFAULT_SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhxdGN2dGNoem11ZWNzc213cnhiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjkwMDg1MzYsImV4cCI6MjA4NDU4NDUzNn0.djoygcwE46xBVTH2kt84i1YoUCtHn1VrNQ5M9FUDzPI"

# Mapeo de columnas Supabase → App
COLUMN_MAPPING = {
    # M200: Supabase → App
    "cifra_negocios": "ventas_netas",
    "resultado_ejercicio": "resultado_neto",
    "total_activo": "activo_total",
    "efectivo_tesoreria": "efectivo",
    "gastos_financieros": "intereses_pagados",
}


# ==============================================================================
# FUNCIONES DE CONEXIÓN
# ==============================================================================

def get_supabase_credentials() -> Tuple[str, str]:
    """
    Obtiene credenciales de Supabase desde secrets o variables de entorno.
    
    Prioridad:
    1. st.secrets (Streamlit Cloud)
    2. Variables de entorno
    3. Valores por defecto
    """
    # Intentar desde Streamlit secrets
    try:
        url = st.secrets.get("supabase", {}).get("url", None)
        key = st.secrets.get("supabase", {}).get("key", None)
        if url and key:
            return url, key
    except Exception:
        pass
    
    # Intentar desde variables de entorno
    url = os.getenv("SUPABASE_URL", DEFAULT_SUPABASE_URL)
    key = os.getenv("SUPABASE_KEY", DEFAULT_SUPABASE_KEY)
    
    return url, key


@st.cache_resource(show_spinner="🔌 Conectando a Supabase...")
def get_supabase_client() -> Optional[Client]:
    """
    Crea y cachea el cliente de Supabase.
    Retorna None si no está disponible.
    """
    if not SUPABASE_AVAILABLE:
        st.warning("⚠️ Librería supabase no instalada. Usando datos locales.")
        return None
    
    try:
        url, key = get_supabase_credentials()
        client = create_client(url, key)
        return client
    except Exception as e:
        st.error(f"❌ Error conectando a Supabase: {e}")
        return None


def is_supabase_available() -> bool:
    """Verifica si Supabase está disponible y configurado."""
    if not SUPABASE_AVAILABLE:
        return False
    
    client = get_supabase_client()
    return client is not None


# ==============================================================================
# FUNCIONES DE CARGA DE DATOS
# ==============================================================================

@st.cache_data(ttl=3600, show_spinner="📊 Cargando Modelo 200 desde Supabase...")
def load_modelo_200() -> Optional[pd.DataFrame]:
    """
    Carga datos del Modelo 200 (Impuesto Sociedades) desde Supabase.
    Retorna DataFrame con estructura compatible con la app.
    """
    client = get_supabase_client()
    if client is None:
        return None
    
    try:
        # Cargar todos los registros (paginación automática)
        response = client.table("modelo_200").select("*").execute()
        
        if not response.data:
            st.warning("⚠️ No hay datos en modelo_200")
            return None
        
        df = pd.DataFrame(response.data)
        
        # Aplicar mapeo de columnas
        df = df.rename(columns=COLUMN_MAPPING)
        
        # Calcular campos derivados
        df['deuda_bancaria'] = df['deuda_bancaria_lp'].fillna(0) + df['deuda_bancaria_cp'].fillna(0)
        df['gastos_transporte'] = df['otros_gastos_expl'].fillna(0) * 0.1  # Estimación
        df['flujo_caja_operativo'] = df['resultado_neto'].fillna(0) + df['amortizaciones'].fillna(0)
        
        # Crear campo sector desde CNAE
        df['sector'] = df['cnae'].apply(lambda x: _cnae_to_sector(x) if pd.notna(x) else 'Otros')
        
        # Campos adicionales requeridos por la app
        df['total_m347'] = 0  # Se actualizará con merge de M347
        df['total_m349'] = 0  # Se actualizará con merge de M349
        df['pct_numeros_redondos'] = np.random.uniform(0.05, 0.25, len(df))  # Simulado
        df['_is_suspicious'] = False  # Se calculará con el modelo
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error cargando modelo_200: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner="📊 Cargando Modelo 347 desde Supabase...")
def load_modelo_347() -> Optional[pd.DataFrame]:
    """
    Carga datos del Modelo 347 (Operaciones con terceros) desde Supabase.
    """
    client = get_supabase_client()
    if client is None:
        return None
    
    try:
        response = client.table("modelo_347").select("*").execute()
        
        if not response.data:
            st.warning("⚠️ No hay datos en modelo_347")
            return None
        
        df = pd.DataFrame(response.data)
        
        # Renombrar para compatibilidad con app existente
        df = df.rename(columns={
            "importe_total": "importe",
            "nif_declarante": "nif_declarante",
            "nif_contraparte": "nif_contraparte",
        })
        
        # Detectar operaciones circulares (mismo NIF en declarante y contraparte en la red)
        # Esto es una simplificación - en producción sería más sofisticado
        df['is_circular'] = 0
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error cargando modelo_347: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner="📊 Cargando Modelo 349 desde Supabase...")
def load_modelo_349() -> Optional[pd.DataFrame]:
    """
    Carga datos del Modelo 349 (Operaciones intracomunitarias) desde Supabase.
    """
    client = get_supabase_client()
    if client is None:
        return None
    
    try:
        response = client.table("modelo_349").select("*").execute()
        
        if not response.data:
            st.warning("⚠️ No hay datos en modelo_349")
            return None
        
        df = pd.DataFrame(response.data)
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error cargando modelo_349: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner="🔄 Consolidando datos desde Supabase...")
def load_all_data_from_supabase() -> Optional[pd.DataFrame]:
    """
    Carga y consolida todos los datos desde Supabase.
    Retorna DataFrame unificado compatible con la app.
    """
    # Cargar modelos
    df_m200 = load_modelo_200()
    df_m347 = load_modelo_347()
    df_m349 = load_modelo_349()
    
    if df_m200 is None:
        return None
    
    df = df_m200.copy()
    
    # Agregar M347 por NIF declarante
    if df_m347 is not None and len(df_m347) > 0:
        m347_agg = df_m347.groupby('nif_declarante').agg({
            'importe': 'sum',
            'is_circular': 'sum'
        }).reset_index()
        m347_agg.columns = ['nif', 'total_m347', 'operaciones_circulares']
        
        df = df.merge(m347_agg, on='nif', how='left', suffixes=('', '_m347'))
        df['total_m347'] = df['total_m347'].fillna(0)
        df['operaciones_circulares'] = df['operaciones_circulares'].fillna(0)
    
    # Agregar M349 por NIF declarante
    if df_m349 is not None and len(df_m349) > 0:
        m349_agg = df_m349.groupby('nif_declarante').agg({
            'importe': 'sum'
        }).reset_index()
        m349_agg.columns = ['nif', 'total_m349']
        
        df = df.merge(m349_agg, on='nif', how='left', suffixes=('', '_m349'))
        df['total_m349'] = df['total_m349'].fillna(0)
    
    # Asegurar que todas las columnas numéricas existan
    numeric_cols = [
        'ventas_netas', 'resultado_neto', 'activo_total', 'deuda_bancaria',
        'gastos_personal', 'gastos_transporte', 'intereses_pagados',
        'flujo_caja_operativo', 'total_m347', 'total_m349', 'amortizaciones',
        'patrimonio_neto', 'efectivo', 'aprovisionamientos', 'otros_ingresos'
    ]
    
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Validar estructura
    required_cols = ['nif', 'sector', 'ventas_netas', 'activo_total']
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        st.error(f"❌ Columnas faltantes: {missing}")
        return None
    
    return df


# ==============================================================================
# HELPERS
# ==============================================================================

def _cnae_to_sector(cnae: str) -> str:
    """Convierte código CNAE a nombre de sector."""
    if not cnae:
        return "Otros"
    
    # Primeros 2 dígitos determinan el sector
    prefix = str(cnae)[:2]
    
    sectors = {
        "01": "Agricultura", "02": "Silvicultura", "03": "Pesca",
        "10": "Industria alimentaria", "11": "Bebidas", "12": "Tabaco",
        "13": "Textil", "14": "Confección", "15": "Cuero",
        "20": "Química", "21": "Farmacia", "22": "Plásticos",
        "23": "Minerales no metálicos", "24": "Metalurgia", "25": "Productos metálicos",
        "26": "Electrónica", "27": "Material eléctrico", "28": "Maquinaria",
        "29": "Vehículos", "30": "Otro material transporte",
        "41": "Construcción edificios", "42": "Ingeniería civil", "43": "Construcción especializada",
        "45": "Venta vehículos", "46": "Comercio mayorista", "47": "Comercio minorista",
        "49": "Transporte terrestre", "50": "Transporte marítimo", "51": "Transporte aéreo",
        "52": "Almacenamiento", "53": "Correos",
        "55": "Alojamiento", "56": "Restauración",
        "58": "Edición", "59": "Cine/TV", "60": "Radio/TV",
        "61": "Telecomunicaciones", "62": "Programación", "63": "Servicios información",
        "64": "Servicios financieros", "65": "Seguros", "66": "Auxiliares financieros",
        "68": "Actividades inmobiliarias",
        "69": "Jurídicos y contables", "70": "Sedes centrales/consultoría", "71": "Arquitectura/Ingeniería",
        "72": "I+D", "73": "Publicidad", "74": "Otras profesionales",
        "77": "Alquiler", "78": "Empleo", "79": "Agencias viajes",
        "80": "Seguridad", "81": "Jardinería", "82": "Actividades administrativas",
    }
    
    return sectors.get(prefix, f"Sector {prefix}")


def get_data_source_stats() -> Dict:
    """Obtiene estadísticas de la fuente de datos."""
    client = get_supabase_client()
    
    if client is None:
        return {"source": "local", "m200": 0, "m347": 0, "m349": 0}
    
    try:
        # Contar registros (usando count)
        m200_count = client.table("modelo_200").select("id", count="exact").execute()
        m347_count = client.table("modelo_347").select("id", count="exact").execute()
        m349_count = client.table("modelo_349").select("id", count="exact").execute()
        
        return {
            "source": "supabase",
            "m200": m200_count.count or 0,
            "m347": m347_count.count or 0,
            "m349": m349_count.count or 0,
        }
    except Exception:
        return {"source": "supabase", "m200": 0, "m347": 0, "m349": 0}


# ==============================================================================
# FUNCIÓN PRINCIPAL DE CARGA (para reemplazar load_real_data)
# ==============================================================================

def load_data_from_source(source: str = "supabase") -> Optional[pd.DataFrame]:
    """
    Función unificada de carga de datos.
    
    Args:
        source: "supabase", "csv", o "synthetic"
    
    Returns:
        DataFrame con datos consolidados o None si falla
    """
    if source == "supabase":
        return load_all_data_from_supabase()
    elif source == "csv":
        # Fallback a carga local (importar función original)
        return None  # Se manejará en streamlit_app.py
    elif source == "synthetic":
        return None  # Se manejará con generate_dummy_data
    else:
        return None
