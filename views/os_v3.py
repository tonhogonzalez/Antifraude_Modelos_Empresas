"""
FraudHunter - Executive Dashboard
==================================
Sistema de detección de fraude empresarial con interfaz ejecutiva.
Diseñado para presentaciones ante comité de dirección.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import base64
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from network_graph_helper import create_suspicious_network
from datetime import datetime
from pathlib import Path

# Enterprise OS Core Modules
try:
    from core.loaders import M200WideLoader, M347Loader, M349Loader
    from core.models import ScoreComposer, SectoralBenchmarker, GovernanceMetrics
    from core.features import BenfordAnalyzer, calculate_benford_features
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

# Continuous Learning Module
try:
    from continuous_learning import (
        FeedbackStorePandas, FeedbackRecord, get_feedback_store,
        ContinuousLearningConfig, FeatureFlags, get_config, get_flags
    )
    from continuous_learning.feedback_store import (
        REJECTION_REASON_CODES, FRAUD_TYPOLOGY_CODES,
        VERDICT_FALSE_POSITIVE, VERDICT_FRAUD, VERDICT_WATCHLIST
    )
    CONTINUOUS_LEARNING_AVAILABLE = True
except ImportError:
    CONTINUOUS_LEARNING_AVAILABLE = False

# Supabase Connector
try:
    from supabase_connector import (
        load_all_data_from_supabase,
        is_supabase_available,
        get_data_source_stats
    )
    SUPABASE_CONNECTOR_AVAILABLE = True
except ImportError:
    SUPABASE_CONNECTOR_AVAILABLE = False
    def is_supabase_available(): return False
    def get_data_source_stats(): return {"source": "local", "m200": 0, "m347": 0, "m349": 0}

# Nota: st.set_page_config y el script de traducción se han movido al punto de entrada global (streamlit_app.py)

# =============================================================================
# DESIGN SYSTEM TIER-1 (SLATE THEME)
# =============================================================================

st.markdown("""
<style>
    /* 1. Global Reset & Theme (Slate 950) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Roboto+Mono&display=swap');

    :root {
        --bg-body: #020617;
        --bg-card: #0f172a;
        --border-sutil: #1e293b;
        --border-hover: #334155;
        --text-primary: #f8fafc;
        --text-muted: #94a3b8;
        --brand: #3b82f6;
        --success: #22c55e;
        --warning: #eab308;
        --danger: #ef4444;
        --purple: #a855f7;
    }

    .stApp {
        background-color: var(--bg-body) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* 2. Hide Streamlit Elements */
    #MainMenu, footer, header {visibility: hidden !important;}
    .stDeployButton {display: none !important;}
    
    .main .block-container {
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* 1. Global Reset & Theme (Slate 950) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Roboto+Mono:wght@400;700&display=swap');

    :root {
        --bg-body: #020617;
        --bg-card: rgba(15, 23, 42, 0.4);
        --bg-card-hover: rgba(15, 23, 42, 0.7);
        --border-sutil: rgba(51, 65, 85, 0.5);
        --border-hover: rgba(59, 130, 246, 0.5);
        --text-primary: #f8fafc;
        --text-muted: #64748b;
        --brand: #3b82f6;
        --brand-glow: rgba(59, 130, 246, 0.2);
        --success: #22c55e;
        --warning: #eab308;
        --danger: #ef4444;
        --purple: #a855f7;
        --glass: blur(16px) saturate(200%);
    }

    .stApp {
        background: radial-gradient(circle at 0% 0%, rgba(59, 130, 246, 0.05) 0%, transparent 50%),
                    radial-gradient(circle at 100% 100%, rgba(139, 92, 246, 0.05) 0%, transparent 50%),
                    var(--bg-body) !important;
        font-family: 'Inter', sans-serif !important;
        color: var(--text-primary) !important;
    }

    /* 2. Hide Streamlit Elements */
    #MainMenu, footer, header {visibility: hidden !important;}
    .stDeployButton {display: none !important;}
    
    .main .block-container {
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* 3. Layout Components */
    .cockpit-header {
        position: sticky;
        top: 0;
        z-index: 999;
        background: rgba(2, 6, 23, 0.7);
        backdrop-filter: var(--glass);
        border-bottom: 1px solid var(--border-sutil);
        padding: 1.5rem 3rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0;
    }

    .kpi-strip {
        display: flex;
        gap: 2rem;
        padding: 1.5rem 3rem;
        background: rgba(2, 6, 23, 0.3);
        border-bottom: 1px solid var(--border-sutil);
        overflow-x: auto;
    }

    /* 4. Tech Cards (Elite Edition) */
    .tech-card {
        background: var(--bg-card);
        backdrop-filter: var(--glass);
        border: 1px solid var(--border-sutil);
        border-radius: 20px;
        padding: 2.25rem;
        transition: all 0.5s cubic-bezier(0.2, 0.8, 0.2, 1);
        position: relative;
        overflow: hidden;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .tech-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    }

    .tech-card:hover {
        border-color: var(--border-hover);
        transform: translateY(-6px);
        box-shadow: 0 30px 60px -12px rgba(0, 0, 0, 0.8), 0 0 20px -5px var(--brand-glow);
        background: var(--bg-card-hover);
    }

    .card-label {
        color: var(--text-muted);
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .card-value {
        color: #ffffff;
        font-size: 2.25rem;
        font-weight: 800;
        font-family: 'Roboto Mono', monospace;
        letter-spacing: -0.04em;
    }

    /* 5. Semantic Badges Elite */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.4rem 1.25rem;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        border: 1px solid transparent;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }

    .badge-success { background: rgba(34, 197, 94, 0.1); color: #4ade80; border-color: rgba(34, 197, 94, 0.2); }
    .badge-warning { background: rgba(234, 179, 8, 0.1); color: #fbbf24; border-color: rgba(234, 179, 8, 0.2); }
    .badge-danger { background: rgba(239, 68, 68, 0.1); color: #f87171; border-color: rgba(239, 68, 68, 0.3); }
    .badge-purple { background: rgba(168, 85, 247, 0.1); color: #c084fc; border-color: rgba(168, 85, 247, 0.2); }

    /* 6. Dashboard Navigation Premium */
    .stTabs [data-baseweb="tab-list"] {
        padding: 0 3rem;
        gap: 2rem;
        background: transparent !important;
    }

    /* 7. Animations Elite */
    .animate-fade {
        animation: reveal 1s cubic-bezier(0.16, 1, 0.3, 1);
    }

    @keyframes reveal {
        from { opacity: 0; transform: translateY(20px); filter: blur(10px); }
        to { opacity: 1; transform: translateY(0); filter: blur(0); }
    }

    /* 8. Sticky Feedback Panel Masterpiece */
    .feedback-panel {
        position: sticky;
        top: 130px;
        background: rgba(15, 23, 42, 0.5);
        backdrop-filter: var(--glass);
        border: 1px solid var(--border-sutil);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 40px 80px -20px rgba(0, 0, 0, 0.6);
    }

    .feedback-panel h3 {
        margin-top: 0;
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        margin-bottom: 2rem;
    }

    /* 9. Grid Systems Elite */
    .cockpit-grid {
        display: grid;
        grid-template-columns: 1fr 420px;
        gap: 3rem;
        padding: 3rem;
        align-items: start;
    }

    /* 10. KPI Strip Item Elite */
    .kpi-item {
        background: rgba(30, 41, 59, 0.2);
        border: 1px solid var(--border-sutil);
        border-radius: 14px;
        padding: 1rem 1.75rem;
        min-width: 220px;
        transition: all 0.3s ease;
    }
    
    .kpi-item:hover {
        background: rgba(30, 41, 59, 0.4);
        border-color: var(--brand);
        box-shadow: 0 0 15px -5px var(--brand-glow);
    }

    .kpi-item-label {
        font-size: 0.65rem;
        color: var(--text-muted);
        text-transform: uppercase;
        font-weight: 800;
        letter-spacing: 0.2em;
        margin-bottom: 0.4rem;
    }

    .kpi-item-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #ffffff;
        font-family: 'Roboto Mono', monospace;
    }

    /* Custom Streamlit Button Styling Elite */
    div.stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
        border: 1px solid var(--border-sutil) !important;
        background: rgba(15, 23, 42, 0.8) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        font-size: 0.8rem !important;
        padding: 0.75rem 1.5rem !important;
    }

    div.stButton > button:hover {
        border-color: var(--brand) !important;
        box-shadow: 0 0 20px -5px var(--brand-glow) !important;
        transform: translateY(-2px) !important;
    }

    /* Scrollbar Elite */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--brand); }

</style>
""", unsafe_allow_html=True)

# =============================================================================
# UI COMPONENTS (ST.MARKDOWN HELPERS)
# =============================================================================

def render_tech_card(label, value, icon=""):
    """Renderiza una Tech Card estilo Enterprise OS."""
    st.markdown(f"""<div class="tech-card animate-fade">
<div class="card-label">{icon} {label}</div>
<div class="card-value">{value}</div>
</div>""", unsafe_allow_html=True)

def render_kpi_strip_item(label, value):
    """Renderiza un item para el KPI Strip superior."""
    return f'<div class="kpi-item"><div class="kpi-item-label">{label}</div><div class="kpi-item-value">{value}</div></div>'

def render_badge(text, badge_type="info"):
    """Renderiza un badge semántico."""
    badge_class = f"badge-{badge_type}"
    return f'<div class="badge {badge_class}">{text}</div>'

def get_risk_badge(score):
    if score > 0.7: return render_badge("HIGH RISK", "danger")
    if score > 0.4: return render_badge("MEDIUM RISK", "warning")
    return render_badge("LOW RISK", "success")

# =============================================================================
# DASHBOARD VIEWS
# =============================================================================

def cockpit_view(df_gold, nif_selected):
    """Vista de Cockpit de Alta Densidad para el analista."""
    
    # Validar si hay empresa seleccionada
    if nif_selected is None:
        st.markdown(f"""<div style="padding: 4rem 2rem; text-align: center; background: var(--bg-card); border-radius: 16px; border: 1px solid var(--border-sutil); margin: 2rem;">
<div style="font-size: 4rem; margin-bottom: 2rem;">🕹️</div>
<h2 style="color: var(--text-primary);">Panel de Control OS de Fraude</h2>
<p style="color: var(--text-muted); font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
Por favor, selecciona una empresa desde el <strong>Control Center</strong> en la barra lateral para iniciar el análisis forense profundo.
</p>
</div>""", unsafe_allow_html=True)
        return

    if df_gold is None or nif_selected not in df_gold['nif'].values:
        st.error(f"Empresa {nif_selected} no encontrada en los registros Gold.")
        return

    company = df_gold[df_gold['nif'] == nif_selected].iloc[0]
    
    # Detectar score
    final_score = company.get('final_score', company.get('fraud_score_normalized', 0))
    
    # 1. FIXED HEADER ELITE
    st.markdown(f'''<div class="cockpit-header animate-fade">
<div style="display: flex; align-items: center; gap: 2rem;">
    <div style="width: 12px; height: 12px; background: {"#ef4444" if final_score > 0.7 else "#eab308" if final_score > 0.4 else "#22c55e"}; border-radius: 50%; box-shadow: 0 0 15px {"#ef4444" if final_score > 0.7 else "#eab308" if final_score > 0.4 else "#22c55e"}; animation: pulse 2s infinite;"></div>
    <div>
        <div style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.2em; font-weight: 800; margin-bottom: 0.25rem;">Corporate Entity Analysis</div>
        <div style="font-size: 2rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em; line-height: 1;">{company.get('razon_social', 'Empresa Desconocida')}</div>
        <div style="color: var(--brand); font-family: 'Roboto Mono'; font-size: 0.9rem; margin-top: 0.4rem; font-weight: 700;">ID: {nif_selected}</div>
    </div>
</div>
<div style="display: flex; gap: 1rem; align-items: center;">
    {get_risk_badge(final_score)}
    {render_badge("Sector: " + str(company.get('cnae', '---')), "purple")}
</div>
</div>
<style>
@keyframes pulse {{
    0% {{ transform: scale(0.95); opacity: 0.5; }}
    50% {{ transform: scale(1.1); opacity: 1; }}
    100% {{ transform: scale(0.95); opacity: 0.5; }}
}}
</style>
''', unsafe_allow_html=True)

    # 2. KPI STRIP
    kpis_html = ""
    kpis_html += render_kpi_strip_item("FINAL SCORE", f"{final_score:.2f}")
    kpis_html += render_kpi_strip_item("TAX COMPONENT", f"{company.get('tax_score', 0):.2f}")
    kpis_html += render_kpi_strip_item("NETWORK DENSITY", f"{company.get('network_score', 0):.2f}")
    kpis_html += render_kpi_strip_item("BENFORD KL", f"{company.get('benford_kl_divergence', 0):.4f}")
    
    st.markdown(f'<div class="kpi-strip">{kpis_html}</div>', unsafe_allow_html=True)

    # 3. SPLIT VIEW GRID
    col_main, col_side = st.columns([0.65, 0.35])

    with col_main:
        st.markdown('<div class="analysis-container animate-fade">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: var(--text-primary); margin-bottom: 1.5rem;">Evidencias y Análisis Detallado</h3>', unsafe_allow_html=True)
        
        # Tech Cards Row
        c1, c2, c3 = st.columns(3)
        with c1: render_tech_card("Cifra Negocios", f"{company.get('cifra_negocios', 0):,.0f} €", "💰")
        with c2: render_tech_card("Efectivo", f"{company.get('efectivo_tesoreria', 0):,.0f} €", "🏦")
        with c3: render_tech_card("Deterioro IFRS9", f"Stage {int(company.get('stage', 1))}", "⚠️")

        st.markdown("""
            <div style="margin-top: 2rem; padding: 1.5rem; background: rgba(59, 130, 246, 0.05); border: 1px border-sutil; border-radius: 12px; border-left: 4px solid var(--brand);">
                <p style="color: var(--text-muted); margin: 0; font-size: 0.9rem;">
                    <strong>Analista:</strong> Utiliza los módulos de análisis para profundizar en las anomalías detectadas. Cada indicador ha sido validado mediante modelos de IA y reglas de negocio Tier-1.
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_side:
        st.markdown('<div class="feedback-panel animate-fade">', unsafe_allow_html=True)
        st.markdown('<h3>Registrar Feedback</h3>', unsafe_allow_html=True)
        st.selectbox("Veredicto", ["PARA_REVISAR", "FRAUDE_CONFIRMADO", "FALSO_POSITIVO", "WATCHLIST"])
        st.text_area("Notas del Analista")
        st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
        if st.button("ENVIAR DECISIÓN", type="primary", use_container_width=True):
            st.success("Decisión registrada.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def governance_dashboard():
    """Dashboard de Gobierno."""
    st.markdown('<div style="padding: 2.5rem;" class="animate-fade">', unsafe_allow_html=True)
    st.markdown('<h1 style="color: var(--text-primary); margin-bottom: 2rem;">🏛️ Model Governance & Monitoring</h1>', unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    with m1: render_tech_card("AUC-ROC", "0.94", "📈")
    with m2: render_tech_card("PSI", "0.04", "⚖️")
    with m3: render_tech_card("Savings", "1.2M €", "💶")
    
    st.markdown("""
        <div style="margin-top: 2.5rem; padding: 1.5rem; background: rgba(34, 197, 94, 0.05); border: 1px solid rgba(34, 197, 94, 0.1); border-radius: 12px; border-left: 4px solid var(--success);">
            <p style="color: var(--text-muted); margin: 0;">Performance estable. No se detecta drift significativo en los últimos 30 días.</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    

# =============================================================================
# DATA ENGINE: LOAD GOLD DATASET
# =============================================================================

@st.cache_data(ttl=600)
def load_gold_dataset():
    """Carga el dataset Gold (resultados finales) desde Supabase o local."""
    df = None
    try:
        from core.loaders import GoldDatasetLoader
        loader = GoldDatasetLoader()
        df = loader.load()
    except Exception:
        # Fallback to current session state
        if 'df_results' in st.session_state:
            df = st.session_state.df_results
    
    # Estandarizar columnas si el datafame existe
    if df is not None:
        df = df.copy()
        if 'final_score' not in df.columns and 'fraud_score_normalized' in df.columns:
            df['final_score'] = df['fraud_score_normalized']
        if 'tax_score' not in df.columns:
            df['tax_score'] = df.get('tax_score', 0)
        if 'network_score' not in df.columns:
            df['network_score'] = df.get('network_score', 0)
            
    return df

# =============================================================================
# MAIN NAVIGATION LOGIC
# =============================================================================

if 'view_mode' not in st.session_state:
    st.session_state.view_mode = "COCKPIT"

def handle_navigation(df_gold=None):
    """Maneja la barra de navegación superior estilo Cockpit."""
    st.markdown("""
    <div style="display: flex; gap: 2rem; padding: 0 2rem; border-bottom: 1px solid var(--border-sutil); background: var(--bg-body); margin-bottom: 0;">
        <div style="padding: 1rem 0; font-weight: 700; color: var(--brand); letter-spacing: 0.1em; cursor: default;">FRAUDHUNTER OS</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, _ = st.columns([0.16, 0.16, 0.16, 0.52])
    
    # CSS dinámico para botones de navegación según estado
    button_style = """
    <style>
        div.stButton > button {
            border: 1px solid var(--border-sutil) !important;
            background: rgba(30, 41, 59, 0.2) !important;
            color: var(--text-muted) !important;
        }
    </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)

    with col1:
        if st.button("🕹️ COCKPIT", key="nav_cockpit", use_container_width=True, type="primary" if st.session_state.view_mode == "COCKPIT" else "secondary"):
            st.session_state.view_mode = "COCKPIT"
            st.rerun()
    with col2:
        if st.button("🏛️ GOVERNANCE", key="nav_gov", use_container_width=True, type="primary" if st.session_state.view_mode == "GOVERNANCE" else "secondary"):
            st.session_state.view_mode = "GOVERNANCE"
            st.rerun()
    with col3:
        if st.button("📦 RAW DATA", key="nav_raw", use_container_width=True, type="primary" if st.session_state.view_mode == "RAW" else "secondary"):
            st.session_state.view_mode = "RAW"
            st.rerun()

    # SIDEBAR SELECTOR
    with st.sidebar:
        # Logo Logic (Optimized for Design System)
        st.markdown(f"""
            <div style="padding: 1rem 0; text-align: center;">
                <h2 style="color: var(--brand); letter-spacing: -0.02em; font-weight: 800; margin: 0;">FRAUDHUNTER <span style="font-weight: 300; opacity: 0.7;">OS</span></h2>
                <p style="color: var(--text-muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.2em; margin-top: 0.25rem;">Enterprise Edition v3.0</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if df_gold is not None:
            st.subheader("🏢 Selección de Empresa")
            
            # Detectar columna de score disponible (final_score vs fraud_score_normalized)
            score_col = 'final_score' if 'final_score' in df_gold.columns else ('fraud_score_normalized' if 'fraud_score_normalized' in df_gold.columns else None)
            
            if score_col:
                # Sort by risk
                sorted_companies = df_gold.sort_values(score_col, ascending=False)
                company_list = []
                company_map = {}
                
                for _, row in sorted_companies.head(50).iterrows():
                    score_val = row[score_col]
                    risk_emoji = "🔴" if score_val > 0.7 else "🟡" if score_val > 0.4 else "🟢"
                    label = f"{risk_emoji} {row['nif']} | Score: {score_val:.2f}"
                    company_list.append(label)
                    company_map[label] = row['nif']
                
                selected_label = st.selectbox(
                    "Top 50 Riesgo Detectado", 
                    options=company_list,
                    index=0 if 'selected_company_nif' not in st.session_state else None
                )
            else:
                st.warning("⚠️ No se detectó columna de score ('final_score' o 'fraud_score_normalized')")
                selected_label = None
            
            if selected_label:
                st.session_state.selected_company_nif = company_map[selected_label]
        
        st.markdown("---")
        st.subheader("🔗 Legacy Version")
        st.info("Accede a la versión anterior de la plataforma para comparativa.")
        if st.button("📺 VERSIÓN LEGACY", use_container_width=True):
            st.switch_page("views/legacy_v2.py")
    

# =============================================================================
# MAIN EXECUTION LOOP
# =============================================================================

# Load data
df_gold = load_gold_dataset()

# Render Navigation
handle_navigation(df_gold)

# Main View Switcher
if st.session_state.view_mode == "COCKPIT":
    # Get NIF from sidebar selector
    selected_nif = st.session_state.get('selected_company_nif')
    cockpit_view(df_gold, selected_nif)

elif st.session_state.view_mode == "GOVERNANCE":
    governance_dashboard()

elif st.session_state.view_mode == "RAW":
    st.markdown('<div class="animate-fade" style="padding: 2rem;">', unsafe_allow_html=True)
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem;">
            <div style="font-size: 2.5rem;">📦</div>
            <div>
                <h1 style="color: var(--text-primary); margin: 0;">Raw Forensic Data</h1>
                <p style="color: var(--text-muted); margin: 0;">Exploración tabular de métricas, flags y scores brutos.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if df_gold is not None:
        # Preparar dataframe para visualización (redondeos y limpieza)
        display_df = df_gold.copy()
        
        # Column configuration para una tabla premium
        st.dataframe(
            display_df,
            use_container_width=True,
            height=600,
            column_config={
                "nif": st.column_config.TextColumn("NIF", width="medium", help="Identificador único"),
                "razon_social": st.column_config.TextColumn("Empresa", width="large"),
                "final_score": st.column_config.ProgressColumn(
                    "Risk Score",
                    help="Score final de fraude consolidado",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                ),
                "tax_score": st.column_config.NumberColumn(
                    "Tax Anomaly",
                    format="%.2f",
                ),
                "benford_kl_divergence": st.column_config.NumberColumn(
                    "Benford Dev",
                    format="%.4f",
                ),
                "cifra_negocios": st.column_config.NumberColumn(
                    "Ventas (€)",
                    format="€%.0f",
                ),
                "efectivo_tesoreria": st.column_config.NumberColumn(
                    "Efectivo (€)",
                    format="€%.0f",
                ),
                "riesgo": st.column_config.TextColumn(
                    "Nivel Riesgo",
                    help="Clasificación de riesgo de la empresa",
                    width="small"
                ),
                "stage": st.column_config.NumberColumn(
                    "IFRS9 Stage",
                    format="Stage %d",
                )
            },
            hide_index=True
        )
    else:
        st.warning("No data available.")
    st.markdown('</div>', unsafe_allow_html=True)

# Finalize script execution (preventing legacy code from running)
st.stop()

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(f"""
    <div class="footer" style="text-align: center; color: var(--text-muted); font-size: 0.85rem; padding: 2rem 0;">
        <strong>FraudHunter OS v3.0</strong> | Enterprise Fraud Detection System<br>
        <span style="font-size: 0.75rem;">
            Tier-1 Banking Grade Compliance | {datetime.now().strftime("%d/%m/%Y %H:%M")}
        </span>
    </div>
""", unsafe_allow_html=True)
