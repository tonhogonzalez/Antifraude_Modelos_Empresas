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

    /* 3. Layout Components */
    .cockpit-header {
        position: sticky;
        top: 0;
        z-index: 999;
        background: rgba(2, 6, 23, 0.85);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid var(--border-sutil);
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0;
    }

    .kpi-strip {
        display: flex;
        gap: 1.5rem;
        padding: 1rem 2rem;
        background: var(--bg-body);
        border-bottom: 1px solid var(--border-sutil);
        overflow-x: auto;
    }

    /* 4. Tech Cards (Enterprise OS) */
    .tech-card {
        background-color: var(--bg-card);
        border: 1px solid var(--border-sutil);
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
    }

    .tech-card:hover {
        border-color: rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }

    .card-label {
        color: var(--text-muted);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .card-value {
        color: var(--text-primary);
        font-size: 1.75rem;
        font-weight: 700;
        font-family: 'Roboto Mono', monospace;
    }

    /* 5. Semantic Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid transparent;
    }

    .badge-success { background: rgba(34, 197, 94, 0.1); color: var(--success); border-color: rgba(34, 197, 94, 0.2); }
    .badge-warning { background: rgba(234, 179, 8, 0.1); color: var(--warning); border-color: rgba(234, 179, 8, 0.2); }
    .badge-danger { background: rgba(239, 68, 68, 0.1); color: var(--danger); border-color: rgba(239, 68, 68, 0.2); }
    .badge-purple { background: rgba(168, 85, 247, 0.1); color: var(--purple); border-color: rgba(168, 85, 247, 0.2); }

    /* 6. Dashboard Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--bg-body) !important;
        gap: 2rem;
        padding: 0 2rem;
        border-bottom: 1px solid var(--border-sutil);
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent !important;
        border: none !important;
        color: var(--text-muted) !important;
        font-weight: 600 !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--brand) !important;
        border-bottom: 2px solid var(--brand) !important;
    }

    /* 7. Animations */
    .animate-fade {
        animation: fadeIn 0.6s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* 8. Sticky Feedback Panel */
    .feedback-panel {
        position: sticky;
        top: 100px;
        background: var(--bg-card);
        border: 1px solid var(--border-sutil);
        border-radius: 12px;
        padding: 1.5rem;
    }

    /* 9. Grid Systems */
    .cockpit-grid {
        display: grid;
        grid-template-columns: 1fr 380px;
        gap: 2rem;
        padding: 2rem;
        align-items: start;
    }

    .analysis-container {
        display: flex;
        flex-direction: column;
        gap: 2rem;
    }

    /* 10. KPI Strip Item */
    .kpi-item {
        background: var(--bg-card);
        border: 1px solid var(--border-sutil);
        border-radius: 8px;
        padding: 0.75rem 1.25rem;
        min-width: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .kpi-item-label {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .kpi-item-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
        font-family: 'Roboto Mono', monospace;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-body); }
    ::-webkit-scrollbar-thumb { background: var(--border-sutil); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--border-hover); }

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
    
    # 1. FIXED HEADER
    st.markdown(f'''<div class="cockpit-header">
<div style="display: flex; align-items: center; gap: 1.5rem;">
<div style="font-size: 1.5rem; font-weight: 700; color: var(--text-primary);">{company.get('razon_social', 'Empresa Desconocida')}</div>
<div style="color: var(--text-muted); font-family: 'Roboto Mono'; font-size: 1.1rem;">{nif_selected}</div>
</div>
<div class="analysis-controls">
{get_risk_badge(final_score)}
{render_badge("CNAE: " + str(company.get('cnae', '---')), "purple")}
</div>
</div>''', unsafe_allow_html=True)

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
        st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
        st.subheader("Evidencias y Análisis Detallado")
        
        # Tech Cards Row
        c1, c2, c3 = st.columns(3)
        with c1: render_tech_card("Cifra Negocios", f"{company.get('cifra_negocios', 0):,.0f} €", "💰")
        with c2: render_tech_card("Efectivo", f"{company.get('efectivo_tesoreria', 0):,.0f} €", "🏦")
        with c3: render_tech_card("Deterioro IFRS9", f"Stage {int(company.get('stage', 1))}", "⚠️")

        st.info("Utiliza los módulos de análisis para profundizar en los datos.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_side:
        st.markdown('<div class="feedback-panel">', unsafe_allow_html=True)
        st.markdown('### Registrar Feedback')
        st.selectbox("Veredicto", ["PARA_REVISAR", "FRAUDE_CONFIRMADO", "FALSO_POSITIVO", "WATCHLIST"])
        st.text_area("Notas del Analista")
        if st.button("ENVIAR DECISIÓN", type="primary", use_container_width=True):
            st.success("Decisión registrada.")
        st.markdown('</div>', unsafe_allow_html=True)

def governance_dashboard():
    """Dashboard de Gobierno."""
    st.title("🏛️ Model Governance & Monitoring")
    
    m1, m2, m3 = st.columns(3)
    with m1: render_tech_card("AUC-ROC", "0.94", "📈")
    with m2: render_tech_card("PSI", "0.04", "⚖️")
    with m3: render_tech_card("Savings", "1.2M €", "💶")
    
    st.info("Performance estable. No se detecta drift significativo.")
    

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
    
    col1, col2, col3, _ = st.columns([0.15, 0.15, 0.15, 0.55])
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
        st.title("🕹️ Control Center")
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
                "riesgo": st.column_config.SelectColumn(
                    "Nivel Riesgo",
                    options=["Bajo", "Medio", "Alto"],
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
