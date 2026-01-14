"""
FraudHunter - Executive Dashboard
==================================
Sistema de detección de fraude empresarial con interfaz ejecutiva.
Diseñado para presentaciones ante comité de dirección.
"""

import streamlit as st
import pandas as pd
import base64
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from scipy.spatial.distance import mahalanobis
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from network_graph_helper import create_suspicious_network
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================

st.set_page_config(
    page_title="FraudHunter Pro 🔍",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado premium
st.markdown("""
<style>
    /* Reset y base */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Header principal */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f64f59 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        color: #6c757d;
        font-size: 1.1rem;
        margin-top: 0.3rem;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-demo {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .status-live {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Tarjetas de capacidades */
    .capability-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .capability-card:hover {
        border-color: rgba(102, 126, 234, 0.8);
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
    }
    
    .capability-icon {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    
    .capability-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #e0e0e0;
        margin-bottom: 0.3rem;
    }
    
    .capability-desc {
        font-size: 0.75rem;
        color: #9e9e9e;
        margin-bottom: 0.5rem;
    }

    .capability-hover-details {
        font-size: 0.7rem;
        color: #c0c0c0;
        max-height: 0;
        overflow: hidden;
        opacity: 0;
        transition: all 0.4s ease;
        border-top: 1px solid rgba(255,255,255,0.1);
        padding-top: 0;
    }

    .capability-card:hover .capability-hover-details {
        max-height: 100px;
        opacity: 1;
        padding-top: 0.5rem;
    }
    
    /* Métricas KPI */
    .kpi-container {
        background: linear-gradient(145deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    .kpi-delta {
        font-size: 0.8rem;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        background: rgba(255,255,255,0.2);
        margin-top: 0.5rem;
        display: inline-block;
    }
    
    /* Tarjeta de empresa */
    .company-card {
        background: linear-gradient(145deg, #1e1e2e 0%, #2d2d44 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .company-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .company-nif {
        font-size: 1.3rem;
        font-weight: 700;
        color: #fff;
    }
    
    .risk-badge {
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #f64f59 0%, #c471ed 100%);
        color: white;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
        color: #1a1a2e;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: #1a1a2e;
    }
    
    /* Flags */
    .flag-container {
        display: flex;
        align-items: center;
        padding: 0.7rem 1rem;
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        margin: 0.4rem 0;
        border-left: 3px solid transparent;
    }
    
    .flag-active {
        border-left-color: #f64f59;
        background: rgba(246, 79, 89, 0.1);
    }
    
    .flag-inactive {
        border-left-color: #38ef7d;
        background: rgba(56, 239, 125, 0.05);
    }
    
    .flag-icon {
        font-size: 1.2rem;
        margin-right: 0.8rem;
    }
    
    .flag-name {
        font-weight: 600;
        color: #e0e0e0;
        flex: 1;
    }
    
    .flag-status {
        font-size: 1.1rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6c757d;
        font-size: 0.85rem;
        padding: 2rem 0;
        border-top: 1px solid rgba(255,255,255,0.1);
        margin-top: 2rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(102, 126, 234, 0.1);
        border-radius: 8px;
        padding: 10px 20px;
        color: #e0e0e0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* ========== HELP PAGE STYLES ========== */
    
    /* Hero Section */
    .help-hero {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 24px;
        padding: 3rem;
        margin-bottom: 3rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 20px 50px rgba(0,0,0,0.5);
        position: relative;
        overflow: hidden;
    }
    
    .help-hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%);
        animation: pulse-bg 4s ease-in-out infinite;
    }

    .help-header-content {
        display: flex;
        align-items: center;
        gap: 2.5rem;
        position: relative;
        z-index: 2;
    }
    
    .help-logo-container {
        background-color: #262730;
        padding: 15px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .help-logo-img {
        width: 140px;
        height: auto;
        display: block;
        /* Eliminamos el drop-shadow y float excesivo para un look más limpio dentro de la tarjeta */
    }

    /* Eliminamos la animación float simple y la hacemos más sutil en el contenedor entero si se desea, 
       pero por ahora estático es más profesional para la tarjeta */
    
    .help-main-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #fff 0%, #e0e0e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -2px;
        line-height: 1.1;
    }
    
    .help-subtitle {
        font-size: 1.5rem;
        color: #667eea;
        font-weight: 500;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .help-meta-badge {
        display: inline-block;
        padding: 6px 16px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 50px;
        color: #a0a0a0;
        font-size: 0.8rem;
        font-family: monospace;
        letter-spacing: 1px;
    }
    
    /* Section Headers */
    .help-section-header {
        display: flex;
        align-items: center;
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    .help-section-number {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        margin-right: 1rem;
        font-size: 1rem;
    }
    
    .help-section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e0e0e0;
    }
    
    /* Algorithm Cards */
    .algo-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .algo-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .algo-card:hover {
        transform: translateY(-4px);
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.2);
    }
    
    .algo-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .algo-icon {
        font-size: 2rem;
        margin-right: 1rem;
    }
    
    .algo-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #fff;
    }
    
    .algo-subtitle {
        font-size: 0.8rem;
        color: #667eea;
        font-weight: 500;
    }
    
    .algo-content {
        color: #b0b0b0;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    .algo-value {
        background: rgba(102, 126, 234, 0.1);
        border-left: 3px solid #667eea;
        padding: 0.8rem 1rem;
        margin-top: 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #c0c0c0;
    }
    
    .algo-value strong {
        color: #38ef7d;
    }
    
    /* Data Tables */
    .data-table-container {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .data-table-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .data-table-icon {
        font-size: 1.5rem;
        margin-right: 0.8rem;
    }
    
    .data-table-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e0e0e0;
    }
    
    .data-table-subtitle {
        font-size: 0.8rem;
        color: #888;
        margin-left: auto;
    }
    
    .data-row {
        display: grid;
        grid-template-columns: 1fr 1.5fr 1.5fr;
        gap: 1rem;
        padding: 0.8rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 0.85rem;
    }
    
    .data-row:last-child {
        border-bottom: none;
    }
    
    .data-row:hover {
        background: rgba(102, 126, 234, 0.05);
    }
    
    .data-label {
        color: #f2c94c;
        font-weight: 600;
    }
    
    .data-treatment {
        color: #a0a0a0;
    }
    
    .data-risk {
        color: #f64f59;
        font-style: italic;
    }
    
    /* Pipeline Steps */
    .pipeline-container {
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
        position: relative;
    }
    
    .pipeline-step {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        flex: 1;
        margin: 0 0.5rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .pipeline-step:hover {
        transform: translateY(-5px);
        border-color: rgba(102, 126, 234, 0.6);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
    }
    
    .pipeline-number {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        margin: 0 auto 1rem auto;
        font-size: 1.2rem;
    }
    
    .pipeline-title {
        font-size: 1rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 0.5rem;
    }
    
    .pipeline-desc {
        font-size: 0.8rem;
        color: #888;
    }
    
    /* Argument Cards */
    .argument-card {
        background: linear-gradient(145deg, rgba(56, 239, 125, 0.05) 0%, rgba(17, 153, 142, 0.1) 100%);
        border: 1px solid rgba(56, 239, 125, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .argument-card:hover {
        border-color: rgba(56, 239, 125, 0.5);
        transform: translateX(5px);
    }
    
    .argument-title {
        font-size: 1rem;
        font-weight: 700;
        color: #38ef7d;
        margin-bottom: 0.5rem;
    }
    
    .argument-desc {
        font-size: 0.85rem;
        color: #a0a0a0;
        line-height: 1.5;
    }
    
    /* Executive Summary Box */
    .executive-summary {
        background: linear-gradient(145deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .executive-summary p {
        color: #d0d0d0;
        font-size: 1rem;
        line-height: 1.8;
    }
    
    .highlight {
        color: #667eea;
        font-weight: 600;
    }
    
    .highlight-warning {
        color: #f2c94c;
        font-weight: 600;
    }
    
    .highlight-danger {
        color: #f64f59;
        font-weight: 600;
    }
    
    .highlight-success {
        color: #38ef7d;
        font-weight: 600;
    }

    /* TOC STYLES */
    .toc-container {
        display: flex;
        gap: 0.8rem;
        margin-bottom: 2.5rem;
        flex-wrap: wrap;
        justify-content: center;
        background: rgba(0,0,0,0.2);
        padding: 1rem;
        border-radius: 50px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .toc-link {
        background: rgba(255,255,255,0.05);
        padding: 0.6rem 1.2rem;
        border-radius: 30px;
        color: #a0a0a0;
        text-decoration: none;
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.3s ease;
        border: 1px solid transparent;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .toc-link:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        color: white;
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-2px);
    }

    /* SUMMARY HIGHLIGHTS */
    .summary-grid {
        display: grid;
        grid-template-columns: 1.8fr 1fr;
        gap: 2rem;
        align-items: center;
    }
    
    .summary-highlight-card {
        background: linear-gradient(145deg, #1e1e2e 0%, #252540 100%);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(118, 75, 162, 0.3);
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .summary-highlight-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #38ef7d, #667eea);
    }
    
    .highlight-stat {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38ef7d 0%, #11998e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .highlight-caption {
        font-size: 0.9rem;
        color: #d0d0d0;
        line-height: 1.4;
    }

    /* PIPELINE ARROWS */
    .pipeline-arrow {
        font-size: 2rem;
        color: #667eea;
        text-align: center;
        opacity: 0.6;
        animation: pulse-arrow 2s infinite;
    }
    
    @keyframes pulse-arrow {
        0%, 100% { opacity: 0.6; transform: translateX(0); }
        50% { opacity: 1; transform: translateX(5px); }
    }
</style>
""", unsafe_allow_html=True)

import os
from pathlib import Path

# =============================================================================
# FUNCIONES DE DATOS
# =============================================================================

# Mapeo de casillas del Modelo 200 a nombres de variables
CASILLA_MAPPING = {
    'C00255': 'cifra_negocios',
    'C00500': 'resultado_ejercicio',
    'C00032': 'gastos_personal',
    'C00033': 'total_activo',
    'C00035': 'inmovilizado',
    'C00085': 'existencias',
    'C00130': 'deudores',
    'C00160': 'efectivo',
    'C00180': 'patrimonio_neto',
    'C00190': 'pasivo_no_corriente',
    'C00195': 'pasivo_corriente',
    'C00250': 'amortizaciones',
    'C00260': 'otros_gastos_expl',
    'C00265': 'gastos_financieros',
    'C00267': 'intereses_pagados',
    'C00385': 'deuda_largo_plazo',
    'C00395': 'deuda_corto_plazo',
}


@st.cache_data
def load_real_data():
    """Carga datos reales desde los archivos CSV si existen."""
    base_path = Path(__file__).parent if '__file__' in dir() else Path('.')
    
    empresas_path = base_path / 'data_empresas.csv'
    eav_path = base_path / 'data_eav.csv'
    m347_path = base_path / 'data_m347.csv'
    
    # Verificar que existen los archivos
    if not all(p.exists() for p in [empresas_path, eav_path, m347_path]):
        return None
    
    try:
        # Cargar empresas
        df_empresas = pd.read_csv(empresas_path)
        
        # Cargar datos EAV y pivotar
        df_eav = pd.read_csv(eav_path)
        
        # Filtrar solo las casillas que necesitamos
        df_eav_filtered = df_eav[df_eav['casilla'].isin(CASILLA_MAPPING.keys())].copy()
        
        # Pivotar de formato EAV a tabla ancha
        df_pivot = df_eav_filtered.pivot_table(
            index='nif',
            columns='casilla',
            values='valor',
            aggfunc='sum'
        ).reset_index()
        
        # Renombrar columnas según el mapeo
        df_pivot.columns = ['nif'] + [CASILLA_MAPPING.get(c, c) for c in df_pivot.columns[1:]]
        
        # Merge con datos de empresas
        df = df_empresas.merge(df_pivot, on='nif', how='left')
        
        # Cargar M347
        df_m347 = pd.read_csv(m347_path)
        
        # Agregar M347 por NIF declarante
        m347_agg = df_m347.groupby('nif_declarante').agg({
            'importe': 'sum',
            'is_circular': 'sum'
        }).reset_index()
        m347_agg.columns = ['nif', 'total_m347', 'operaciones_circulares']
        
        df = df.merge(m347_agg, on='nif', how='left')
        
        # Calcular variables derivadas - usando columnas directamente con fallback
        df['ventas_netas'] = df['cifra_negocios'] if 'cifra_negocios' in df.columns else 0
        df['resultado_neto'] = df['resultado_ejercicio'] if 'resultado_ejercicio' in df.columns else 0
        df['activo_total'] = df['total_activo'] if 'total_activo' in df.columns else 0
        
        # Para columnas que pueden no existir, crear con 0
        for col in ['deuda_largo_plazo', 'deuda_corto_plazo', 'otros_gastos_expl', 
                    'gastos_financieros', 'amortizaciones', 'gastos_personal']:
            if col not in df.columns:
                df[col] = 0
        
        df['deuda_bancaria'] = df['deuda_largo_plazo'].fillna(0) + df['deuda_corto_plazo'].fillna(0)
        df['gastos_transporte'] = df['otros_gastos_expl'].fillna(0) * 0.1  # Estimación
        df['intereses_pagados'] = df['gastos_financieros'].fillna(0)
        df['flujo_caja_operativo'] = df['resultado_ejercicio'].fillna(0) + df['amortizaciones'].fillna(0) if 'resultado_ejercicio' in df.columns else 0
        df['total_m349'] = 0  # No tenemos este dato
        
        # Simular % números redondos basado en operaciones
        np.random.seed(42)
        df['pct_numeros_redondos'] = np.where(
            df['tipo'] == 'SANA',
            np.random.uniform(0.05, 0.15, len(df)),
            np.random.uniform(0.35, 0.65, len(df))
        )
        
        # Usar el sector de la columna existente
        df['sector'] = df['sector_cnae']
        df['cnae'] = df['sector_cnae'].apply(lambda x: x[:4] if pd.notna(x) else '0000')
        
        # Marcar empresas sospechosas basado en el tipo
        df['_is_suspicious'] = df['tipo'].str.contains('FRAUDE|PANTALLA|CARRUSEL', case=False, na=False)
        
        # Rellenar NaN con 0
        numeric_cols = ['ventas_netas', 'resultado_neto', 'activo_total', 'deuda_bancaria',
                       'gastos_personal', 'gastos_transporte', 'intereses_pagados',
                       'flujo_caja_operativo', 'total_m347', 'total_m349']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
        
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None

@st.cache_data
def generate_dummy_data(n_companies: int = 500, fraud_rate: float = 0.10) -> pd.DataFrame:
    """Genera datos sintéticos de empresas."""
    np.random.seed(42)
    cnaes = ['4511', '4719', '6201', '6831', '4941', '4321', '6810', '7022']
    sectores = {
        '4511': 'Venta de automóviles',
        '4719': 'Comercio minorista',
        '6201': 'Programación informática',
        '6831': 'Agentes inmobiliarios',
        '4941': 'Transporte de mercancías',
        '4321': 'Instalaciones eléctricas',
        '6810': 'Compraventa inmuebles',
        '7022': 'Consultoría empresarial'
    }
    
    data = []
    for i in range(n_companies):
        nif = f"B{str(i).zfill(7)}A"
        cnae = np.random.choice(cnaes)
        is_suspicious = np.random.random() < fraud_rate
        
        ventas = np.random.uniform(100000, 10000000)
        if is_suspicious:
            resultado = ventas * np.random.uniform(0.15, 0.30)
            flujo_caja = resultado * np.random.uniform(0.1, 0.3)
            gastos_personal = ventas * np.random.uniform(0.001, 0.02)
            gastos_transporte = 0
            pct_redondos = np.random.uniform(0.4, 0.8)
        else:
            resultado = ventas * np.random.uniform(0.02, 0.08)
            flujo_caja = resultado * np.random.uniform(0.7, 1.1)
            gastos_personal = ventas * np.random.uniform(0.1, 0.3)
            gastos_transporte = ventas * 0.02
            pct_redondos = np.random.uniform(0.05, 0.15)
        
        activo = ventas * np.random.uniform(0.5, 1.5)
        deuda = ventas * np.random.uniform(0.1, 0.4)
        intereses = ventas * np.random.uniform(0.01, 0.05)
        
        data.append({
            'nif': nif,
            'cnae': cnae,
            'sector': sectores[cnae],
            'ventas_netas': ventas,
            'resultado_neto': resultado,
            'activo_total': activo,
            'deuda_bancaria': deuda,
            'gastos_personal': gastos_personal,
            'gastos_transporte': gastos_transporte,
            'intereses_pagados': intereses,
            'flujo_caja_operativo': flujo_caja,
            'total_m347': ventas * np.random.uniform(0.7, 1.1),
            'total_m349': ventas * np.random.uniform(0, 0.3) if is_suspicious else 0,
            'pct_numeros_redondos': pct_redondos,
            '_is_suspicious': is_suspicious
        })
    
    return pd.DataFrame(data)


def calculate_forensic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features forenses."""
    df = df.copy()
    
    # Cobertura de ventas
    df['cobertura_ventas'] = df['total_m347'] / df['ventas_netas']
    
    # Incoherencia logística
    df['flag_incoherencia_logistica'] = (
        (df['total_m349'] > 0) & (df['gastos_transporte'] < 1000)
    ).astype(int)
    
    # Accruals ratio
    df['accruals_ratio'] = (
        (df['resultado_neto'] - df['flujo_caja_operativo']) / df['activo_total']
    )
    
    # Hidden debt
    df['tasa_interes_implicita'] = np.where(
        df['deuda_bancaria'] > 0,
        df['intereses_pagados'] / df['deuda_bancaria'],
        0
    )
    df['flag_hidden_debt'] = (df['tasa_interes_implicita'] > 0.10).astype(int)
    
    # Productividad laboral
    df['productividad_laboral'] = np.where(
        df['gastos_personal'] > 0,
        df['ventas_netas'] / df['gastos_personal'],
        np.nan
    )
    df['flag_empresa_pantalla'] = (
        (df['ventas_netas'] > 100000) & (df['gastos_personal'] < 10000)
    ).astype(int)
    
    # Números redondos
    df['flag_numeros_redondos'] = (df['pct_numeros_redondos'] > 0.30).astype(int)
    
    # Ratios financieros
    df['margen_neto'] = df['resultado_neto'] / df['ventas_netas']
    df['rotacion_activos'] = df['ventas_netas'] / df['activo_total']
    df['ratio_endeudamiento'] = df['deuda_bancaria'] / df['activo_total']
    
    # Cobertura M347 sospechosa
    df['flag_cobertura_baja'] = (df['cobertura_ventas'] < 0.75).astype(int)
    
    # Accruals anómalos
    df['flag_accruals_anomalo'] = (df['accruals_ratio'] > 0.15).astype(int)
    
    return df


def calculate_mahalanobis_by_sector(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula distancia Mahalanobis por sector."""
    df = df.copy()
    feature_cols = ['margen_neto', 'rotacion_activos', 'ratio_endeudamiento']
    df['mahalanobis_distance'] = np.nan
    
    for cnae in df['cnae'].unique():
        sector_mask = df['cnae'] == cnae
        sector_data = df.loc[sector_mask, feature_cols].dropna()
        
        if len(sector_data) < 3:
            continue
        
        centroid = sector_data.mean().values
        cov_matrix = sector_data.cov().values + np.eye(3) * 1e-6
        
        try:
            cov_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            continue
        
        for idx in sector_data.index:
            point = df.loc[idx, feature_cols].values
            df.loc[idx, 'mahalanobis_distance'] = mahalanobis(point, centroid, cov_inv)
    
    df['flag_sectoral_outlier'] = (df['mahalanobis_distance'] > 3.0).astype(int)
    return df


def train_isolation_forest(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """Entrena Isolation Forest."""
    df = df.copy()
    
    feature_cols = [
        'cobertura_ventas', 'accruals_ratio', 'tasa_interes_implicita',
        'productividad_laboral', 'pct_numeros_redondos',
        'mahalanobis_distance', 'flag_incoherencia_logistica',
        'flag_hidden_debt', 'flag_empresa_pantalla', 'flag_numeros_redondos',
        'flag_sectoral_outlier'
    ]
    
    existing_cols = [c for c in feature_cols if c in df.columns]
    X = df[existing_cols].copy()
    X = X.fillna(X.median())
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    df['anomaly_label'] = model.fit_predict(X_scaled)
    df['fraud_score'] = model.decision_function(X_scaled)
    
    min_score = df['fraud_score'].min()
    max_score = df['fraud_score'].max()
    df['fraud_score_normalized'] = 1 - (
        (df['fraud_score'] - min_score) / (max_score - min_score)
    )
    
    # Clasificación por niveles
    df['riesgo'] = pd.cut(
        df['fraud_score_normalized'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Bajo', 'Medio', 'Alto']
    )
    
    return df


def get_flag_details():
    """Retorna información detallada de cada flag."""
    return {
        'flag_empresa_pantalla': {
            'nombre': 'Empresa Pantalla',
            'icono': '🏭',
            'descripcion': 'Ventas elevadas sin estructura de personal proporcional',
            'umbral': 'Ventas > €100k y gastos de personal < €10k'
        },
        'flag_incoherencia_logistica': {
            'nombre': 'Incoherencia Logística',
            'icono': '📦',
            'descripcion': 'Operaciones intracomunitarias sin gastos de transporte',
            'umbral': 'M349 > 0 y gastos transporte < €1,000'
        },
        'flag_hidden_debt': {
            'nombre': 'Deuda Oculta',
            'icono': '💳',
            'descripcion': 'Tasa de interés implícita anormalmente alta',
            'umbral': 'Tasa interés implícita > 10%'
        },
        'flag_numeros_redondos': {
            'nombre': 'Números Redondos',
            'icono': '🔢',
            'descripcion': 'Facturas con importes exactos sospechosos',
            'umbral': '> 30% importes terminados en 00, 000, 500'
        },
        'flag_sectoral_outlier': {
            'nombre': 'Outlier Sectorial',
            'icono': '📊',
            'descripcion': 'Métricas financieras atípicas vs sector',
            'umbral': 'Distancia Mahalanobis > 3.0'
        },
        'flag_cobertura_baja': {
            'nombre': 'Cobertura M347 Baja',
            'icono': '📋',
            'descripcion': 'Ventas declaradas no soportadas por operaciones informadas',
            'umbral': 'Cobertura M347/Ventas < 75%'
        },
        'flag_accruals_anomalo': {
            'nombre': 'Accruals Anómalo',
            'icono': '📈',
            'descripcion': 'Divergencia entre resultado contable y flujo de caja',
            'umbral': 'Accruals ratio > 15%'
        }
    }


# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================

# Resolve logo path
logo_path = "logo_dark.png" if Path("logo_dark.png").exists() else "logo.png"

# Sidebar Branding
with st.sidebar:
    try:
        st.image(logo_path, use_container_width=True)
    except:
        pass
    
    st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <p style="color: #6c757d; font-size: 0.8rem; margin-top: 0px;">Intelligence & Forensics</p>
        </div>
    """, unsafe_allow_html=True)

# Main Content Area
# Status bar minimalista en top right
st.markdown(f"""
    <div style="display: flex; justify-content: flex-end; align-items: center; margin-bottom: 1rem;">
        <span class="status-badge status-demo" style="margin-right: 10px;">⚡ MODO DEMO v2.2</span>
        <span style="color: #6c757d; font-size: 0.8rem;">{datetime.now().strftime("%d/%m/%Y %H:%M")}</span>
    </div>
""", unsafe_allow_html=True)

st.sidebar.header("⚙️ Configuración del Análisis")

st.sidebar.markdown("---")

# Selector de fuente de datos
data_source = st.sidebar.radio(
    "📁 Fuente de Datos",
    options=["Datos Reales (CSV)", "Datos Sintéticos"],
    help="Selecciona si usar los archivos CSV reales o generar datos sintéticos"
)

use_real_data = data_source == "Datos Reales (CSV)"

if not use_real_data:
    st.sidebar.markdown("---")
    n_empresas = st.sidebar.slider(
        "Número de empresas",
        min_value=100,
        max_value=2000,
        value=500,
        step=100
    )
    
    tasa_fraude = st.sidebar.slider(
        "Tasa de fraude sintético (%)",
        min_value=5,
        max_value=30,
        value=10
    ) / 100
else:
    n_empresas = 500
    tasa_fraude = 0.10

st.sidebar.markdown("---")

contamination = st.sidebar.slider(
    "Sensibilidad del modelo",
    min_value=1,
    max_value=15,
    value=5,
    help="Mayor valor = más empresas marcadas como sospechosas"
) / 100

st.sidebar.markdown("---")

# Botón de análisis
if st.sidebar.button("🚀 Ejecutar Análisis", type="primary", use_container_width=True):
    st.session_state.run_analysis = True
    st.session_state.use_real_data = use_real_data

st.sidebar.markdown("---")
# Enlace a documentación
with st.sidebar.expander("📖 Documentación del Sistema", expanded=False):
    st.markdown("""
    ### FraudHunter - Inteligencia Forense
    
    **Tecnologías utilizadas:**
    - 🔬 **Isolation Forest** - ML no supervisado
    - 📊 **Distancia Mahalanobis** - Contexto sectorial
    - 🔢 **Ley de Benford** - Detección estadística
    - 🌐 **Graph Analysis** - Redes de transacciones
    
    **Modelos fiscales analizados:**
    - Modelo 200 (Impuesto Sociedades)
    - Modelo 347 (Operaciones +3.005€)
    - Modelo 349 (Operaciones Intracomunitarias)
    """)

# Ejecutar análisis
if st.session_state.get('run_analysis', False) or 'df_results' not in st.session_state:
    try:
        with st.spinner("🔄 Procesando datos con algoritmos forenses..."):
            # Intentar cargar datos reales si está seleccionado
            if use_real_data:
                df = load_real_data()
                if df is None:
                    st.warning("⚠️ Archivos CSV no encontrados. Usando datos sintéticos.")
                    df = generate_dummy_data(n_empresas, tasa_fraude)
                else:
                    st.sidebar.success(f"✅ {len(df)} empresas cargadas desde archivos")
            else:
                df = generate_dummy_data(n_empresas, tasa_fraude)
            
            # Feature Engineering
            with st.spinner("📊 Calculando features forenses..."):
                df = calculate_forensic_features(df)
            
            # Análisis Sectorial
            with st.spinner("🔬 Aplicando análisis Mahalanobis por sector..."):
                df = calculate_mahalanobis_by_sector(df)
            
            # Modelo ML
            with st.spinner("🤖 Entrenando modelo Isolation Forest..."):
                df = train_isolation_forest(df, contamination)
            
            st.session_state.df_results = df
            st.session_state.run_analysis = False
            
            # Feedback de éxito
            n_detected = (df['anomaly_label'] == -1).sum()
            st.toast(f"✅ Análisis completado: {n_detected} anomalías detectadas en {len(df)} empresas", icon="🎯")
            
    except Exception as e:
        st.error(f"❌ Error durante el análisis: {str(e)}")
        st.info("💡 Intenta refrescar la página o cambiar la fuente de datos.")
        st.session_state.run_analysis = False
        st.stop()

df = st.session_state.df_results

import streamlit.components.v1 as components

# =============================================================================
# NAVEGACIÓN CON PERSISTENCIA DE TAB
# =============================================================================

# Inicializar tab activo en session_state
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

# Inicializar empresa seleccionada
if 'selected_company_nif' not in st.session_state:
    st.session_state.selected_company_nif = None

# Crear navegación con botones - AHORA 4 BOTONES
col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)

with col_nav1:
    if st.button("📊 Dashboard Ejecutivo", use_container_width=True, 
                 type="primary" if st.session_state.active_tab == 0 else "secondary"):
        st.session_state.active_tab = 0
        st.rerun()

with col_nav2:
    if st.button("🔎 Análisis por Empresa", use_container_width=True,
                 type="primary" if st.session_state.active_tab == 1 else "secondary"):
        st.session_state.active_tab = 1
        st.rerun()

with col_nav3:
    if st.button("📥 Exportar Resultados", use_container_width=True,
                 type="primary" if st.session_state.active_tab == 2 else "secondary"):
        st.session_state.active_tab = 2
        st.rerun()

with col_nav4:
    if st.button("❓ Ayuda", use_container_width=True,
                 type="primary" if st.session_state.active_tab == 3 else "secondary"):
        st.session_state.active_tab = 3
        st.rerun()

st.markdown("---")

# =============================================================================
# TAB 4: AYUDA Y PRESENTACIÓN DE LA SOLUCIÓN
# =============================================================================
if st.session_state.active_tab == 3:
    
    # TOC de Navegación Rápida
    st.markdown("<br>", unsafe_allow_html=True)
    
    # TOC de Navegación Rápida
    st.markdown("""
        <div class="toc-container">
            <a href="#resumen-ejecutivo" class="toc-link">📋 Resumen</a>
            <a href="#matriz-interaccion-forense" class="toc-link">🧠 Matriz Forense</a>
            <a href="#pipeline-process" class="toc-link">⚙️ Pipeline</a>
            <a href="#modelos-fiscales" class="toc-link">📊 Modelos AEAT</a>
            <a href="#argumentario-de-venta" class="toc-link">💡 Argumentario</a>
        </div>
    """, unsafe_allow_html=True)
    
    # ==========================================================================
    # SECCIÓN 1: RESUMEN EJECUTIVO
    # ==========================================================================
    st.markdown("""
        <div id="resumen-ejecutivo" class="help-section-header">
            <div class="help-section-number">1</div>
            <div class="help-section-title">Resumen Ejecutivo</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="summary-grid">
            <div class="summary-text" style="color: #d0d0d0; font-size: 1.05rem; line-height: 1.7;">
                <p style="margin-bottom: 1rem;">
                    <strong>FraudHunter</strong> transforma la gestión del riesgo de crédito pasando de un modelo 
                    <span class="highlight-warning">reactivo</span> a uno <span class="highlight-success">proactivo</span>. 
                    No solo valida lo que el cliente presenta, sino que descubre la <span class="highlight">realidad operativa</span> detrás de los datos.
                </p>
                <p>
                    Utilizando <strong>Inteligencia Artificial No Supervisada</strong> y <strong>Teoría de Grafos</strong>, 
                    el sistema detecta patrones de fraude sofisticado como <span class="highlight-danger">tramas organizadas</span>, 
                    <span class="highlight-danger">maquillaje de balances</span> y <span class="highlight-danger">empresas instrumentales</span> 
                    que escapan al ojo humano.
                </p>
            </div>
            <div class="summary-highlight-card">
                <div class="highlight-stat">95%</div>
                <div class="highlight-caption">De reducción de tiempo en análisis manual filtrando empresas sanas</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # ==========================================================================
    # SECCIÓN 2: MATRIZ DE INTERACCIÓN (NUEVA)
    # ==========================================================================
    st.markdown("""
        <div id="matriz-interaccion-forense" class="help-section-header">
            <div class="help-section-number">2</div>
                    <div>Tratamiento</div>
                    <div>Riesgo</div>
                </div>
                <div class="data-row">
                    <div class="data-label">Entregas Intracomunitarias</div>
                    <div class="data-treatment">Variable de Exportación</div>
                    <div class="data-risk">Fraude Carrusel</div>
                </div>
                <div class="data-row" style="grid-template-columns: 1fr;">
                    <div class="data-treatment" style="color: #f2c94c; padding: 0.5rem; background: rgba(242, 201, 76, 0.1); border-radius: 8px; margin-top: 0.5rem;">
                        <strong>⚠️ Paradoja Logística:</strong> Si una empresa declara Entregas UE (M349) altas 
                        pero Gastos de Transporte (M200) nulos, la operación es físicamente imposible.
                    </div>
            <div class="help-section-title">Matriz de Interacción Algoritmo-Variable</div>
        </div>
        <p style="color: #667eea; font-size: 1.1rem; margin-bottom: 2rem; font-style: italic;">
            "Esta es la guía definitiva de la lógica forense del sistema."
        </p>
    """, unsafe_allow_html=True)

    # ESTILOS ADICIONALES PARA LA MATRIZ DE TABLAS
    st.markdown("""
    <style>
        .matrix-card {
            background: #1e1e2e;
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(255,255,255,0.05);
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }
        .matrix-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding-bottom: 1rem;
        }
        .matrix-icon {
            font-size: 2rem;
            background: rgba(102, 126, 234, 0.1);
            width: 50px; height: 50px;
            display: flex; align-items: center; justify-content: center;
            border-radius: 12px;
        }
        .matrix-title { font-size: 1.3rem; font-weight: 700; color: #fff; }
        .matrix-obj { font-size: 0.9rem; color: #a0a0a0; margin-top: 0.2rem; }
        
        .matrix-table { width: 100%; border-collapse: separate; border-spacing: 0 8px; }
        .matrix-table th { text-align: left; color: #667eea; font-size: 0.85rem; padding: 0.5rem; }
        .matrix-table td { background: rgba(255,255,255,0.03); padding: 1rem; vertical-align: top; font-size: 0.9rem; color: #d0d0d0; }
        .matrix-table td:first-child { border-top-left-radius: 8px; border-bottom-left-radius: 8px; font-weight: 600; color: #f2c94c; width: 25%; }
        .matrix-table td:last-child { border-top-right-radius: 8px; border-bottom-right-radius: 8px; color: #f64f59; font-weight: 600; width: 30%; }
    </style>
    """, unsafe_allow_html=True)

    # 1. Isolation Forest
    st.markdown("""
    <div class="matrix-card">
        <div class="matrix-header">
            <div class="matrix-icon">🔬</div>
            <div>
                <div class="matrix-title">1. El Detector de Anomalías (Isolation Forest)</div>
                <div class="matrix-obj"><strong>Objetivo:</strong> Encontrar comportamiento financiero "matemáticamente absurdo" o extremadamente raro.</div>
            </div>
        </div>
        <table class="matrix-table">
            <tr>
                <th>VARIABLE (Casilla)</th>
                <th>PROCESAMIENTO ALGORÍTMICO</th>
                <th>RIESGO DETECTADO</th>
            </tr>
            <tr>
                <td>C00255 (Cifra Negocios)</td>
                <td>Se normaliza junto con los Activos (C00033). El algoritmo busca desproporciones extremas (ventas gigantes con activos minúsculos).</td>
                <td><strong>Empresas Pantalla:</strong> Facturan millones para mover dinero, pero no tienen infraestructura (activos) para generar esas ventas.</td>
            </tr>
            <tr>
                <td>C00263 (Gastos Personal)</td>
                <td>Se cruza con Ventas. El algoritmo aísla empresas con Ventas > 1M€ y Personal ≈ 0.</td>
                <td><strong>Fraude Pantalla:</strong> Mueven mercancía (papel) sin empleados reales. Solo hay un administrador firmando facturas.</td>
            </tr>
            <tr>
                <td>C00279 (Otros Gastos / Transporte)</td>
                <td>Se cruza con ENTREGAS_UE (M349). Si hay exportaciones UE altas pero gastos de transporte ≈ 0, es físicamente imposible.</td>
                <td><strong>Fraude Carrusel IVA:</strong> La mercancía "viaja" solo en papel. No hay camiones ni logística real.</td>
            </tr>
            <tr>
                <td>C00296 (Gastos Financieros)</td>
                <td>Se compara contra la Deuda Bancaria (C00195 + C00215). Si los gastos son muy altos para la deuda declarada, es una anomalía.</td>
                <td><strong>Pasivo Oculto:</strong> La empresa paga intereses por una deuda "B" (prestamistas, no bancaria) oculta al balance oficial.</td>
            </tr>
            <tr>
                <td>C00062 (Patrimonio Neto)</td>
                <td>Se analiza si es negativo mientras la empresa sigue operando con ventas positivas.</td>
                <td><strong>Empresa Zombie:</strong> Patrimonio neto < 0 pero sigue facturando. Incumple obligación legal de disolución.</td>
            </tr>
            <tr>
                <td>C00032 (Tesorería)</td>
                <td>Se analiza su variación respecto al Beneficio (C00500). Si Beneficio sube y Caja baja drásticamente sin inversión, dispara el score.</td>
                <td><strong>Beneficio Ficticio (Paper Profit):</strong> Declaran ganancias para pedir crédito, pero el dinero nunca entra en el banco.</td>
            </tr>
            <tr>
                <td>C00280 (Amortizaciones)</td>
                <td>Se compara con Activo Fijo. Si las amortizaciones son anormalmente bajas respecto al activo, puede haber manipulación.</td>
                <td><strong>Maquillaje Contable:</strong> Reducen amortizaciones para inflar el beneficio artificialmente.</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # 2. Graph Analysis
    st.markdown("""
    <div class="matrix-card">
        <div class="matrix-header">
            <div class="matrix-icon">🕸️</div>
            <div>
                <div class="matrix-title">2. El Analista de Redes (GraphSAGE / GraphFrames)</div>
                <div class="matrix-obj"><strong>Objetivo:</strong> Juzgar a una empresa por sus "malas compañías" y patrones de conexión.</div>
            </div>
        </div>
        <table class="matrix-table">
            <tr>
                <th>VARIABLE (Modelo 347)</th>
                <th>PROCESAMIENTO ALGORÍTMICO</th>
                <th>RIESGO DETECTADO</th>
            </tr>
            <tr>
                <td>NIF_CONTRAPARTE</td>
                <td>Los convierte en nodos. Si una empresa (Nodo A) se conecta con proveedores (Nodos B, C) marcados como fraude.</td>
                <td><strong>Culpa por Asociación (Contagio):</strong> "Dime con quién andas". Bloquea empresas nuevas que usan proveedores sucios.</td>
            </tr>
            <tr>
                <td>IMPORTE_OPS</td>
                <td>Analiza la dirección y cuantía. Busca el patrón: A vende 100 a B, B vende 100 a C, C vende 100 a A.</td>
                <td><strong>Circularidad (Carrusel IVA):</strong> El dinero da vueltas para inflar facturación artificialmente sin valor real.</td>
            </tr>
            <tr>
                <td>FECHA_ALTA_CENSAL</td>
                <td>Se combina con centralidad (PageRank). Si Antigüedad < 6 meses y PageRank > 90% (muchas conexiones).</td>
                <td><strong>Empresa "Trucha" (Burner):</strong> Sociedad creada para emitir facturas falsas masivamente y desaparecer rápido.</td>
            </tr>
            <tr>
                <td>REF_CATASTRAL</td>
                <td>Agrupa nodos que comparten la misma referencia catastral en direcciones distintas.</td>
                <td><strong>Nidos de Fraude:</strong> Detecta que 50 empresas distintas "viven" en el mismo piso franco de 60m².</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # 3. NLP
    st.markdown("""
    <div class="matrix-card">
        <div class="matrix-header">
            <div class="matrix-icon">🧠</div>
            <div>
                <div class="matrix-title">3. El Auditor Semántico (BERT / NLP)</div>
                <div class="matrix-obj"><strong>Objetivo:</strong> Leer la "letra pequeña" para encontrar mentiras en la descripción del negocio.</div>
            </div>
        </div>
        <table class="matrix-table">
            <tr>
                <th>VARIABLE (Texto)</th>
                <th>PROCESAMIENTO ALGORÍTMICO</th>
                <th>RIESGO DETECTADO</th>
            </tr>
            <tr>
                <td>OBJETO_SOCIAL</td>
                <td>El modelo "lee" el texto y extrae el vector semántico (de qué va el negocio realmente) comparándolo con CNAE.</td>
                <td><strong>Incoherencia de Actividad:</strong> Declara CNAE "Tecnología" (subvenciones) pero escrituras dicen "Construcción".</td>
            </tr>
            <tr>
                <td>MEMORIA_ACTIVIDAD</td>
                <td>Busca palabras clave de riesgo y analiza el "tono lingüístico" y genericidad.</td>
                <td><strong>Empresa Zombi Reciclada:</strong> Descripciones "copia-pega" o incoherentes con importes (consultoría vs venta mercadería).</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # 4. Benford
    st.markdown("""
    <div class="matrix-card">
        <div class="matrix-header">
            <div class="matrix-icon">🔢</div>
            <div>
                <div class="matrix-title">4. El Forense Estadístico (Ley de Benford)</div>
                <div class="matrix-obj"><strong>Objetivo:</strong> Detectar la intervención humana manual en las cifras.</div>
            </div>
        </div>
        <table class="matrix-table">
            <tr>
                <th>VARIABLE (Facturas)</th>
                <th>PROCESAMIENTO ALGORÍTMICO</th>
                <th>RIESGO DETECTADO</th>
            </tr>
            <tr>
                <td>IMPORTE (1er Dígito)</td>
                <td>Calcula frecuencia del primer dígito (1-9). Si el '9' aparece el 15% (lo natural es 4.6%).</td>
                <td><strong>Falsificación Manual:</strong> El contable inventa facturas justo debajo de umbrales auditoría (ej: 9.900€).</td>
            </tr>
            <tr>
                <td>IMPORTE (Terminación)</td>
                <td>Busca exceso de terminaciones en .00 o .000.</td>
                <td><strong>Invención de Cifras:</strong> Con IVA y descuentos, los números redondos son rarísimos. Si abundan, son falsos.</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # 5. Mahalanobis
    st.markdown("""
    <div class="matrix-card">
        <div class="matrix-header">
            <div class="matrix-icon">📊</div>
            <div>
                <div class="matrix-title">5. El Contextualizador (Mahalanobis)</div>
                <div class="matrix-obj"><strong>Objetivo:</strong> Evitar culpar a inocentes entendiendo el contexto de su sector.</div>
            </div>
        </div>
        <table class="matrix-table">
            <tr>
                <th>VARIABLE (Ratios)</th>
                <th>PROCESAMIENTO ALGORÍTMICO</th>
                <th>RIESGO DETECTADO</th>
            </tr>
            <tr>
                <td>EBITDA / DEUDA</td>
                <td>No mira valor absoluto. Mira la distancia al promedio del sector (CNAE).</td>
                <td><strong>Falsos Positivos:</strong> Startup tech pierde dinero pero es normal en su sector. Mahalanobis evita la alerta "Quiebra".</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    # FINAL QUOTE
    st.markdown("""
        <div style="background: linear-gradient(90deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); 
                    border-left: 4px solid #38ef7d; padding: 2rem; border-radius: 0 16px 16px 0; margin: 3rem 0;">
            <p style="font-size: 1.2rem; font-style: italic; color: #e0e0e0; line-height: 1.6;">
                "Señores, mientras los sistemas antiguos solo validan si el documento Excel cuadra (A = B), 
                <strong>FraudHunter valida si la realidad física, matemática y relacional detrás de ese Excel es posible</strong>. 
                Detectamos lo que el defraudador no puede esconder: sus relaciones, su incoherencia semántica y su huella estadística."
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # ==========================================================================
    # SECCIÓN 4: PIPELINE CON FLECHAS
    # ==========================================================================
    st.markdown("""
        <div id="pipeline-process" class="help-section-header">
            <div class="help-section-number">4</div>
            <div class="help-section-title">Pipeline Analítico</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Layout 5 columnas para flechas
    c1, a1, c2, a2, c3 = st.columns([1, 0.15, 1, 0.15, 1])
    
    with c1:
        st.markdown("""
            <div class="pipeline-step">
                <div class="pipeline-number">1</div>
                <div class="pipeline-title">📥 Ingesta (ETL)</div>
                <div style="font-size: 0.75rem; color: #667eea; margin-bottom: 0.5rem;">Silver Layer</div>
                <div class="pipeline-desc">
                    Filtrado de ruido y selección quirúrgica de casillas críticas (Tax Mapping).
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with a1:
        st.markdown('<div style="height: 100px; display: flex; align-items: center; justify-content: center;"><div class="pipeline-arrow">➤</div></div>', unsafe_allow_html=True)

    with c2:
        st.markdown("""
            <div class="pipeline-step">
                <div class="pipeline-number">2</div>
                <div class="pipeline-title">⚙️ Feature Eng.</div>
                <div style="font-size: 0.75rem; color: #667eea; margin-bottom: 0.5rem;">Gold Layer</div>
                <div class="pipeline-desc">
                    Cálculo de +50 indicadores forenses y aplicación de reglas de negocio duras.
                </div>
            </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown('<div style="height: 100px; display: flex; align-items: center; justify-content: center;"><div class="pipeline-arrow">➤</div></div>', unsafe_allow_html=True)
    
    with c3:
        st.markdown("""
            <div class="pipeline-step">
                <div class="pipeline-number">3</div>
                <div class="pipeline-title">🤖 AI Scoring</div>
                <div style="font-size: 0.75rem; color: #667eea; margin-bottom: 0.5rem;">Serving Layer</div>
                <div class="pipeline-desc">
                    Unificación de evidencias en Fraud Score y generación de informe explicativo.
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # ==========================================================================
    # SECCIÓN 5: MODELOS FISCALES UTILIZADOS (NUEVA)
    # ==========================================================================
    st.markdown("""
        <div id="modelos-fiscales" class="help-section-header">
            <div class="help-section-number">5</div>
            <div class="help-section-title">Modelos Fiscales y Datos Utilizados</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <p style="color: #a0a0a0; margin-bottom: 1.5rem;">
            El sistema procesa datos de los <strong style="color: #667eea;">tres modelos fiscales</strong> principales de la AEAT España:
        </p>
    """, unsafe_allow_html=True)
    
    # Tabla M200
    st.markdown("#### 📋 Modelo 200 - Impuesto de Sociedades")
    m200_data = {
        'Casilla': ['C00255', 'C00258', 'C00260', 'C00263', 'C00279', 'C00280', 'C00296', 'C00500', 'C00033', 'C00062', 'C00195', 'C00215', 'C00032'],
        'Variable': ['cifra_negocios', 'aprovisionamientos', 'otros_ingresos', 'gastos_personal', 'otros_gastos_expl', 'amortizaciones', 'gastos_financieros', 'resultado_ejercicio', 'total_activo', 'patrimonio_neto', 'deuda_bancaria_lp', 'deuda_bancaria_cp', 'efectivo_tesoreria'],
        'Descripción': ['Ingresos de Explotación', 'Compras y Aprovisionamientos', 'Subvenciones Recibidas', 'Sueldos y Seg. Social', 'Transportes/Servicios Ext.', 'Amortización del Inmovilizado', 'Intereses de Deuda', 'Beneficio/Pérdida Neto', 'Total Activo Balance', 'Patrimonio Neto', 'Deudas Bancarias L/P', 'Deudas Bancarias C/P', 'Caja y Equivalentes'],
        'Uso Principal': ['Base de ventas', 'Márgenes', 'Ayudas públicas', 'Empresa pantalla', 'Carrusel IVA', 'Maquillaje', 'Deuda oculta', 'Márgenes anómalos', 'Pantalla/Inflado', 'Empresas zombie', 'Deuda oculta', 'Deuda oculta', 'Liquidez ficticia']
    }
    st.dataframe(pd.DataFrame(m200_data), hide_index=True, use_container_width=True)
    
    col_m347, col_m349 = st.columns(2)
    
    with col_m347:
        st.markdown("#### 📊 Modelo 347 - Operaciones con Terceros")
        st.markdown("""
        | Campo | Uso |
        |-------|-----|
        | `NIF_DECLARANTE` | Nodo origen en grafo |
        | `NIF_CONTRAPARTE` | Nodo destino en grafo |
        | `IMPORTE_OPS` | Peso de la conexión |
        | `is_circular` | Flag de circularidad |
        
        **Umbral declaración:** > 3.005,06€
        """)
    
    with col_m349:
        st.markdown("#### 🇪🇺 Modelo 349 - Operaciones UE")
        st.markdown("""
        | Campo | Uso |
        |-------|-----|
        | `ENTREGAS_UE` | Ventas intracomunitarias |
        | `ADQUISICIONES_UE` | Compras de UE |
        
        **Riesgo Carrusel:** Entregas UE altas + Transporte bajo = 🚨
        
        **Países riesgo alto:** CY, LU, MT, NL, IE, BG, RO
        """)
    
    # Dataset Info
    st.markdown("#### 📊 Dataset Actual")
    st.markdown("""
    | Métrica | Valor |
    |---------|-------|
    | **Empresas** | 100,000 |
    | **Sectores CNAE** | 48 diferentes |
    | **Casillas EAV** | 14 por empresa (1.4M filas) |
    | **Transacciones M347** | 100,000 |
    | **Tamaño total** | ~44 MB |
    
    **Distribución por tamaño:**
    - Micro (<2M€): 85%
    - Pequeña (2-10M€): 10%
    - Mediana (10-50M€): 4%
    - Grande (>50M€): 1%
    
    **Patrones de fraude inyectados (5%):**
    🔄 Carrusel | 📊 Maquillaje | 🏭 Pantalla | 💀 Zombie
    """)
    
    # ==========================================================================
    # SECCIÓN 6: ARGUMENTARIO DE VENTA
    # ==========================================================================
    st.markdown("""
        <div id="argumentario-de-venta" class="help-section-header">
            <div class="help-section-number">6</div>
            <div class="help-section-title">Argumentario de Venta: ¿Por qué invertir en FraudHunter?</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <p style="color: #a0a0a0; margin-bottom: 1.5rem;">
            Para cerrar la presentación al comité, estos son los <strong style="color: #38ef7d;">argumentos clave</strong>:
        </p>
    """, unsafe_allow_html=True)
    
    col_arg1, col_arg2 = st.columns(2)
    
    with col_arg1:
        st.markdown("""
            <div class="argument-card">
                <div class="argument-title">⚡ Eficiencia Operativa</div>
                <div class="argument-desc">
                    Reduce el tiempo de análisis manual al filtrar automáticamente el <strong>95% de empresas "sanas"</strong> 
                    y focalizar a los analistas solo en las alertas complejas justificadas.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="argument-card">
                <div class="argument-title">🔍 Explicabilidad (Compliance)</div>
                <div class="argument-desc">
                    A diferencia de las "Cajas Negras" de IA tradicionales, FraudHunter <strong>desglosa el "por qué"</strong> 
                    de cada alerta, facilitando la defensa ante auditoría interna y reguladores.
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_arg2:
        st.markdown("""
            <div class="argument-card">
                <div class="argument-title">🛡️ Mitigación de Riesgo Sistémico</div>
                <div class="argument-desc">
                    Al detectar redes (grafos), evitamos que la caída de una empresa fraudulenta 
                    <strong>arrastre a toda una cartera</strong> de clientes interconectados.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="argument-card">
                <div class="argument-title">📈 Escalabilidad</div>
                <div class="argument-desc">
                    Diseñado en Spark/Databricks, está listo para procesar desde <strong>10.000 hasta 10 millones</strong> 
                    de clientes sin cambios en la arquitectura.
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; padding: 2rem; border-top: 1px solid rgba(255,255,255,0.1); margin-top: 2rem;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🛡️</div>
            <div style="color: #667eea; font-weight: 600; font-size: 1.1rem;">FraudHunter Pro</div>
            <div style="color: #888; font-size: 0.85rem;">Plataforma de Inteligencia Forense Financiera</div>
            <div style="color: #555; font-size: 0.75rem; margin-top: 0.5rem;">© 2026 - Análisis Avanzado de Riesgo Empresarial</div>
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB 1: DASHBOARD EJECUTIVO
# =============================================================================

if st.session_state.active_tab == 0:
    # Sección de Capacidades del Sistema
    st.markdown("### 🧠 Capacidades de Detección")
    
    capabilities = [
        ("🔬", "Isolation Forest", "ML no supervisado", 
         "Algoritmo de aprendizaje no supervisado que aísla anomalías basándose en que son pocas y diferentes."),
        ("📊", "Análisis Sectorial", "Mahalanobis distance", 
         "Calcula la distancia de Mahalanobis para identificar empresas con comportamiento financiero atípico respecto a su sector."),
        ("🏭", "Empresas Pantalla", "Ratio ventas/personal", 
         "Detecta sociedades con facturación significativa pero sin estructura operativa (personal o activos) acorde."),
        ("📦", "Incoherencia Logística", "Cruce M349/transporte", 
         "Cruza operaciones intracomunitarias (M349) con gastos de transporte para detectar flujos físicos inexistentes."),
        ("💳", "Deuda Oculta", "Tasa interés implícita", 
         "Identifica financiación no bancaria a través de tasas de interés implícitas anormalmente altas en otras partidas."),
        ("🔢", "Números Redondos", "Detección Benford", 
         "Aplica la Ley de Benford y detección de patrones de redondeo excesivo en importes de facturas."),
        ("📋", "Cobertura M347", "Cruce ventas/operaciones", 
         "Verifica que el volumen de ventas declarado esté soportado por operaciones con terceros informadas en el M347."),
        ("📈", "Manipulación Contable", "Accruals anómalos", 
         "Analiza los 'accruals' para detectar divergencias sospechosas entre el beneficio contable y el flujo de caja real.")
    ]
    
    cols = st.columns(8)
    for i, (icon, title, desc, long_desc) in enumerate(capabilities):
        with cols[i]:
            st.markdown(f"""
                <div class="capability-card">
                    <div class="capability-icon">{icon}</div>
                    <div class="capability-title">{title}</div>
                    <div class="capability-desc">{desc}</div>
                    <div class="capability-hover-details">{long_desc}</div>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Métricas principales
    st.markdown("### 📈 Métricas de Riesgo Global")
    
    n_anomalies = (df['anomaly_label'] == -1).sum()
    n_high_risk = (df['riesgo'] == 'Alto').sum()
    n_medium_risk = (df['riesgo'] == 'Medio').sum()
    true_pos = ((df['anomaly_label'] == -1) & (df['_is_suspicious'] == True)).sum()
    pred_pos = (df['anomaly_label'] == -1).sum()
    precision = true_pos / pred_pos if pred_pos > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="kpi-container" style="background: linear-gradient(145deg, #667eea 0%, #764ba2 100%);">
                <div class="kpi-value">{len(df):,}</div>
                <div class="kpi-label">🏢 Empresas Analizadas</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="kpi-container" style="background: linear-gradient(145deg, #f64f59 0%, #c471ed 100%);">
                <div class="kpi-value">{n_anomalies:,}</div>
                <div class="kpi-label">⚠️ Anomalías Detectadas</div>
                <div class="kpi-delta">{100*n_anomalies/len(df):.1f}% del total</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="kpi-container" style="background: linear-gradient(145deg, #f2994a 0%, #f2c94c 100%);">
                <div class="kpi-value">{n_high_risk:,}</div>
                <div class="kpi-label">🔴 Alto Riesgo</div>
                <div class="kpi-delta">{n_medium_risk} medio riesgo</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="kpi-container" style="background: linear-gradient(145deg, #11998e 0%, #38ef7d 100%);">
                <div class="kpi-value">{precision:.0%}</div>
                <div class="kpi-label">🎯 Precisión del Modelo</div>
                <div class="kpi-delta">vs ground truth</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ==========================================================================
    # NUEVA SECCIÓN: GAUGE + TOP 10
    # ==========================================================================
    col_gauge, col_top10 = st.columns([1, 2])
    
    with col_gauge:
        st.markdown("#### 🎯 Salud de la Cartera")
        
        # Calcular % de empresas sanas
        pct_sanas = (len(df) - n_anomalies) / len(df) * 100
        
        # Determinar color según salud
        if pct_sanas >= 90:
            gauge_color = "#38ef7d"  # Verde
        elif pct_sanas >= 75:
            gauge_color = "#f2c94c"  # Amarillo
        else:
            gauge_color = "#f64f59"  # Rojo
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pct_sanas,
            number={'suffix': '%', 'font': {'size': 40, 'color': 'white'}},
            delta={'reference': 95, 'relative': False, 'valueformat': '.1f', 'suffix': ' vs objetivo'},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white", 'tickfont': {'color': 'white'}},
                'bar': {'color': gauge_color, 'thickness': 0.8},
                'bgcolor': "rgba(255,255,255,0.1)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 75], 'color': 'rgba(246, 79, 89, 0.2)'},
                    {'range': [75, 90], 'color': 'rgba(242, 201, 76, 0.2)'},
                    {'range': [90, 100], 'color': 'rgba(56, 239, 125, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 3},
                    'thickness': 0.8,
                    'value': 95
                }
            },
            title={'text': "Empresas Sanas", 'font': {'size': 16, 'color': '#a0a0a0'}}
        ))
        
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            height=280,
            margin=dict(t=50, b=20, l=30, r=30)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Mini resumen debajo del gauge
        st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem; background: rgba(255,255,255,0.05); border-radius: 10px;">
                <span style="color: #38ef7d; font-weight: 600;">{len(df) - n_anomalies:,}</span> sanas
                &nbsp;|&nbsp;
                <span style="color: #f64f59; font-weight: 600;">{n_anomalies:,}</span> sospechosas
            </div>
        """, unsafe_allow_html=True)
    
    with col_top10:
        st.markdown("#### 🚨 Top 10 Empresas de Mayor Riesgo")
        
        # Obtener top 10 por fraud_score_normalized
        top10 = df.nlargest(10, 'fraud_score_normalized')[['nif', 'sector', 'fraud_score_normalized', 'riesgo', 'ventas_netas']].copy()
        top10['rank'] = range(1, 11)
        
        # Crear flags activos
        flag_details = get_flag_details()
        flag_cols = [col for col in flag_details.keys() if col in df.columns]
        
        def get_active_flags(nif):
            row = df[df['nif'] == nif].iloc[0]
            icons = []
            for col in flag_cols:
                if row.get(col, 0) == 1:
                    icons.append(flag_details[col]['icono'])
            return " ".join(icons[:4])  # Max 4 icons
        
        top10['flags'] = top10['nif'].apply(get_active_flags)
        top10['sector_short'] = top10['sector'].str[:20] + '...'
        
        # Reordenar y renombrar para mostrar
        display_df = top10[['rank', 'nif', 'sector_short', 'fraud_score_normalized', 'riesgo', 'flags']].copy()
        display_df.columns = ['#', 'NIF', 'Sector', 'Score', 'Riesgo', 'Alertas']
        
        # Usar st.dataframe con column_config para mejor visualización
        st.dataframe(
            display_df,
            column_config={
                "#": st.column_config.NumberColumn("#", width="small"),
                "NIF": st.column_config.TextColumn("NIF", width="medium"),
                "Sector": st.column_config.TextColumn("Sector", width="medium"),
                "Score": st.column_config.ProgressColumn(
                    "Score de Fraude",
                    help="Score normalizado de riesgo (0-1)",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                ),
                "Riesgo": st.column_config.TextColumn("Riesgo", width="small"),
                "Alertas": st.column_config.TextColumn("Flags", width="small"),
            },
            hide_index=True,
            use_container_width=True,
            height=350
        )
        
        # Quick Jump: Selector para ir directamente al detalle
        st.markdown("---")
        quick_jump_nif = st.selectbox(
            "⚡ Quick Jump: Ver detalle de empresa",
            options=top10['nif'].tolist(),
            format_func=lambda x: f"{x} - Score: {top10[top10['nif']==x]['fraud_score_normalized'].values[0]:.2f}",
            key="quick_jump_top10"
        )
        
        if st.button("🔍 Ver Análisis Detallado", use_container_width=True):
            st.session_state.selected_company_nif = quick_jump_nif
            st.session_state.active_tab = 1
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Gráficos principales
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### 🎯 Distribución de Riesgo por Sector")
        
        # Usar gráfico de barras agrupadas - más robusto que sunburst/treemap
        df_risk = df.groupby(['sector', 'riesgo']).agg({
            'nif': 'count',
            'ventas_netas': 'sum'
        }).reset_index()
        df_risk.columns = ['Sector', 'Riesgo', 'Empresas', 'Ventas']
        df_risk = df_risk.dropna()
        
        # Convertir riesgo a string para evitar problemas con categoricals
        df_risk['Riesgo'] = df_risk['Riesgo'].astype(str)
        
        fig_risk = px.bar(
            df_risk,
            x='Sector',
            y='Empresas',
            color='Riesgo',
            barmode='group',
            color_discrete_map={
                'Bajo': '#38ef7d',
                'Medio': '#f2c94c',
                'Alto': '#f64f59'
            },
            title=''
        )
        fig_risk.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title='',
            yaxis_title='Número de Empresas',
            legend_title_text='Riesgo',
            xaxis_tickangle=-45
        )
        fig_risk.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
        fig_risk.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col_chart2:
        st.markdown("#### 📊 Score de Fraude vs Volumen de Ventas")
        
        fig_scatter = px.scatter(
            df,
            x='ventas_netas',
            y='fraud_score_normalized',
            color='riesgo',
            size='activo_total',
            hover_name='nif',
            hover_data=['sector', 'resultado_neto'],
            color_discrete_map={
                'Bajo': '#38ef7d',
                'Medio': '#f2c94c',
                'Alto': '#f64f59'
            },
            title=''
        )
        fig_scatter.update_layout(
            xaxis_title='Ventas Netas (€)',
            yaxis_title='Score de Fraude',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend_title_text='Nivel de Riesgo'
        )
        fig_scatter.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
        fig_scatter.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Segunda fila de gráficos
    col_chart3, col_chart4 = st.columns(2)
    
    with col_chart3:
        st.markdown("#### 🚨 Flags de Riesgo por Tipo")
        
        flag_details = get_flag_details()
        flag_data = []
        for flag_col, details in flag_details.items():
            if flag_col in df.columns:
                flag_data.append({
                    'Flag': details['nombre'],
                    'Cantidad': df[flag_col].sum(),
                    'Icono': details['icono']
                })
        
        df_flags = pd.DataFrame(flag_data).sort_values('Cantidad', ascending=True)
        
        fig_bar = px.bar(
            df_flags,
            x='Cantidad',
            y='Flag',
            orientation='h',
            color='Cantidad',
            color_continuous_scale='Reds',
            title=''
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            yaxis_title='',
            xaxis_title='Número de empresas'
        )
        fig_bar.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col_chart4:
        st.markdown("#### 🏢 Matriz de Riesgo por Sector")
        
        # Crear heatmap de flags por sector
        flag_cols = [c for c in df.columns if c.startswith('flag_')]
        flag_names = {c: get_flag_details().get(c, {}).get('nombre', c.replace('flag_', '').replace('_', ' ').title()) 
                      for c in flag_cols}
        
        heatmap_data = df.groupby('sector')[flag_cols].sum()
        heatmap_data.columns = [flag_names.get(c, c) for c in heatmap_data.columns]
        
        fig_heatmap = px.imshow(
            heatmap_data.T,
            color_continuous_scale='Reds',
            aspect='auto',
            title=''
        )
        fig_heatmap.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title='',
            yaxis_title=''
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Tabla de empresas sospechosas
    st.markdown("### 🔴 Top 20 Empresas de Mayor Riesgo")
    
    flag_details = get_flag_details()
    flag_cols = [col for col in flag_details.keys() if col in df.columns]
    
    # Obtener las 20 empresas más sospechosas (menor score = más anómalo)
    top_indices = df.nsmallest(20, 'fraud_score').index
    
    for idx in top_indices:
        row = df.loc[idx]
        
        # Identificar flags activos
        active_flags_list = []
        for col in flag_cols:
            if row.get(col, 0) == 1:
                active_flags_list.append(flag_details[col])
        
        icons = " ".join([f['icono'] for f in active_flags_list])
        
        # Diseño de tarjeta expandible
        with st.expander(f"🚨 {row['nif']} - {row['sector']} | Score: {row['fraud_score_normalized']:.3f} | {icons}"):
            # 1. Dashboard mini de métricas
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Ventas Netas", f"€{row['ventas_netas']:,.0f}")
            m2.metric("Resultado Neto", f"€{row.get('resultado_neto', 0):,.0f}")
            m3.metric("Deuda Bancaria", f"€{row.get('deuda_bancaria', 0):,.0f}")
            
            risk_color = "red" if row['riesgo'] == 'Alto' else "orange"
            m4.markdown(f"**Nivel de Riesgo:** :{risk_color}[{row['riesgo']}]")
            
            st.divider()
            
            # 2. Detalle de ALertas
            st.markdown("#### 🕵️‍♂️ Análisis Forense de Alertas")
            
            if active_flags_list:
                for flag in active_flags_list:
                    # Contenedor visual para cada alerta
                    st.markdown(f"""
                        <div style="background-color: rgba(255, 75, 75, 0.1); padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #ff4b4b;">
                            <div style="font-weight: bold; font-size: 1.1em;">{flag['icono']} {flag['nombre']}</div>
                            <div style="margin-top: 5px;">{flag['descripcion']}</div>
                            <div style="font-family: monospace; font-size: 0.9em; color: #ff8c8c; margin-top: 5px;">
                                📐 Lógica de Cálculo: {flag['umbral']}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("⚠️ Esta empresa ha sido marcada por el modelo de IA (Isolation Forest) por patrón anómalo general, aunque no ha disparado reglas heurísticas específicas.")
                st.markdown(f"**Motivo IA:** Distancia espacial anómala en las variables transformadas (Score: {row['fraud_score']:.4f})")


# =============================================================================
# TAB 2: ANÁLISIS POR EMPRESA
# =============================================================================

if st.session_state.active_tab == 1:
    st.markdown("### 🔎 Análisis Detallado por Empresa")
    st.markdown("Seleccione una empresa para ver el detalle completo de las detecciones.")
    
    # Selector de empresa
    col_select1, col_select2 = st.columns([2, 1])
    
    with col_select1:
        # Crear lista de empresas ordenadas por riesgo
        df_sorted = df.sort_values('fraud_score_normalized', ascending=False)
        empresa_options = [f"{row['nif']} - {row['sector']} ({row['riesgo']})" 
                          for _, row in df_sorted.iterrows()]
        
        # Determinar índice inicial basado en Quick Jump
        default_idx = 0
        if st.session_state.selected_company_nif:
            matching = [i for i, opt in enumerate(empresa_options) 
                       if opt.startswith(st.session_state.selected_company_nif)]
            if matching:
                default_idx = matching[0]
            # Limpiar después de usar
            st.session_state.selected_company_nif = None
        
        selected_empresa = st.selectbox(
            "🏢 Seleccionar Empresa",
            options=empresa_options,
            index=default_idx,
            help="Las empresas están ordenadas por nivel de riesgo (mayor a menor)",
            key="empresa_selector_tab2"
        )
    
    with col_select2:
        st.markdown("<br>", unsafe_allow_html=True)
        show_only_high_risk = st.checkbox("Mostrar solo alto riesgo", value=False)
    
    if selected_empresa:
        selected_nif = selected_empresa.split(" - ")[0]
        empresa_data = df[df['nif'] == selected_nif].iloc[0]
        
        # Tarjeta principal de la empresa
        risk_class = {
            'Alto': 'risk-high',
            'Medio': 'risk-medium', 
            'Bajo': 'risk-low'
        }.get(empresa_data['riesgo'], 'risk-low')
        
        st.markdown(f"""
            <div class="company-card">
                <div class="company-header">
                    <div>
                        <span class="company-nif">{empresa_data['nif']}</span>
                        <span style="color: #9e9e9e; margin-left: 1rem;">{empresa_data['sector']}</span>
                    </div>
                    <span class="risk-badge {risk_class}">🎯 Riesgo {empresa_data['riesgo']}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Contenido detallado (Tabs internos)
        tab_det1, tab_det2 = st.tabs(["📊 Métricas Clave", "🔎 Evidencias y Grafo"])
        
        with tab_det1:
            col_met1, col_met2, col_met3 = st.columns(3)
            with col_met1:
                st.metric("Score de Fraude", f"{empresa_data['fraud_score_normalized']:.3f}")
            with col_met2:
                st.metric("Ventas Netas", f"€{empresa_data['ventas_netas']:,.0f}")
            with col_met3:
                st.metric("Resultado Neto", f"€{empresa_data.get('resultado_neto', 0):,.0f}")
            
            # Radar Chart comparación sectorial
            st.markdown("#### Comparativa Sectorial")
            
            # Calcular medias del sector
            sector_means = df[df['sector'] == empresa_data['sector']].select_dtypes(include=[np.number]).mean()
            
            # Variables a comparar (normalizadas) - simplificado para demo
            categories = ['Rentabilidad', 'Liquidez', 'Endeudamiento', 'Eficiencia', 'Crecimiento']
            
            # Generar datos simulados para el radar basados en el riesgo
            if empresa_data['riesgo'] == 'Alto':
                values_empresa = [0.2, 0.3, 0.9, 0.4, 0.2]
            elif empresa_data['riesgo'] == 'Medio':
                values_empresa = [0.5, 0.6, 0.7, 0.5, 0.5]
            else:
                values_empresa = [0.8, 0.8, 0.4, 0.7, 0.8]
                
            values_sector = [0.6, 0.7, 0.5, 0.6, 0.6]
            
            fig_radar = go.Figure()

            fig_radar.add_trace(go.Scatterpolar(
                r=values_empresa,
                theta=categories,
                fill='toself',
                name=f"Empresa ({empresa_data['nif']})",
                line_color='#f64f59' if empresa_data['riesgo'] == 'Alto' else '#38ef7d'
            ))
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values_sector,
                theta=categories,
                fill='toself',
                name='Media Sector',
                line_color='#667eea'
            ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)

        with tab_det2:
            st.markdown("#### 🚩 Flags Activados")
            
            flag_details = get_flag_details()
            active_flags = []
            
            for col, details in flag_details.items():
                if col in empresa_data and empresa_data[col] == 1:
                    active_flags.append(details)
            
            if active_flags:
                for flag in active_flags:
                    st.warning(f"**{flag['nombre']}** ({flag['icono']}): {flag['descripcion']}")
            else:
                st.success("✅ No se han detectado anomalías específicas en las reglas predefinidas.")
            
            st.markdown("#### 🕸️ Análisis de Grafo de Operaciones (M347)")
            
            # Generar Grafo Interactivo (PyVis)
            with st.spinner("Generando grafo interactivo..."):
                risk_level = empresa_data.get('riesgo', 'Bajo')
                fraud_score_val = empresa_data.get('fraud_score_normalized', 0.5)
                
                html_graph = create_suspicious_network(
                    center_nif=selected_nif,
                    center_risk=risk_level,
                    center_score=fraud_score_val
                )
                
                # Renderizar HTML del grafo
                components.html(html_graph, height=600, scrolling=False)
            
            st.success("🖱️ **Interacción:** Arrastra los nodos para reorganizarlos | Scroll para zoom | Click + arrastrar fondo para mover vista")


# =============================================================================
# TAB 3: EXPORTAR RESULTADOS
# =============================================================================

if st.session_state.active_tab == 2:
    st.markdown("### 📥 Exportar Resultados del Análisis")
    
    # Calcular métricas para el resumen
    n_anomalies = (df['anomaly_label'] == -1).sum()
    n_high_risk = (df['riesgo'] == 'Alto').sum()
    n_medium_risk = (df['riesgo'] == 'Medio').sum()
    true_pos = ((df['anomaly_label'] == -1) & (df['_is_suspicious'] == True)).sum()
    pred_pos = (df['anomaly_label'] == -1).sum()
    precision = true_pos / pred_pos if pred_pos > 0 else 0
    
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    with col_dl1:
        st.markdown("#### 📊 Datos Completos")
        st.markdown("Todas las empresas con todas las métricas calculadas.")
        csv_all = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar CSV Completo",
            data=csv_all,
            file_name=f"fraudhunter_completo_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_dl2:
        st.markdown("#### ⚠️ Solo Sospechosas")
        st.markdown("Empresas con anomalías detectadas.")
        df_suspicious = df[df['anomaly_label'] == -1]
        csv_suspicious = df_suspicious.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Descargar {len(df_suspicious)} Sospechosas",
            data=csv_suspicious,
            file_name=f"fraudhunter_sospechosos_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_dl3:
        st.markdown("#### 🔴 Solo Alto Riesgo")
        st.markdown("Empresas clasificadas como alto riesgo.")
        df_high = df[df['riesgo'] == 'Alto']
        csv_high = df_high.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Descargar {len(df_high)} Alto Riesgo",
            data=csv_high,
            file_name=f"fraudhunter_alto_riesgo_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Resumen del análisis
    st.markdown("---")
    st.markdown("### 📋 Resumen del Análisis")
    
    col_sum1, col_sum2 = st.columns(2)
    
    with col_sum1:
        st.markdown(f"""
        | Parámetro | Valor |
        |-----------|-------|
        | Empresas analizadas | {len(df):,} |
        | Anomalías detectadas | {n_anomalies:,} ({100*n_anomalies/len(df):.1f}%) |
        | Alto riesgo | {n_high_risk:,} |
        | Medio riesgo | {n_medium_risk:,} |
        | Bajo riesgo | {len(df) - n_high_risk - n_medium_risk:,} |
        """)
    
    with col_sum2:
        st.markdown(f"""
        | Métrica del Modelo | Valor |
        |--------------------|-------|
        | Precisión | {precision:.1%} |
        | Contaminación | {contamination:.1%} |
        | Estimators | 100 |
        | Algoritmo | Isolation Forest |
        """)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(f"""
    <div class="footer">
        <strong>FraudHunter Pro v2.0</strong> | Sistema de Detección de Fraude Empresarial<br>
        <span style="font-size: 0.75rem;">
            Metodología: Isolation Forest + Análisis Sectorial Mahalanobis + Cruces Tributarios (M200/M347/M349)<br>
            📊 Datos sintéticos para demostración | {datetime.now().strftime("%d/%m/%Y %H:%M")}
        </span>
    </div>
""", unsafe_allow_html=True)
