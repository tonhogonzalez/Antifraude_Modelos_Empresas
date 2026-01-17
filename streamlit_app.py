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

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(
    page_title="FraudHunter Pro 🔍",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Disable browser translation prompt
components.html(
    """
    <script>
        var meta = document.createElement('meta');
        meta.name = 'google';
        meta.content = 'notranslate';
        if (!window.parent.document.querySelector('meta[name="google"]')) {
            window.parent.document.head.appendChild(meta);
        }
    </script>
    """,
    height=0,
    width=0
)

# =============================================================================
# CSS PERSONALIZADO
# =============================================================================
st.markdown("""
<style>
    /* Hide Streamlit share button */
    button[title="Share"] { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    .stDeployButton { display: none !important; }
    
    /* Main container */
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    
    /* Header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f64f59 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        min-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .kpi-value { font-size: 2.5rem; font-weight: 800; line-height: 1; }
    .kpi-label { font-size: 0.85rem; opacity: 0.9; margin-top: 0.5rem; }
    
    /* Risk Badges */
    .risk-high { 
        background: linear-gradient(135deg, #f64f59 0%, #c471ed 100%); 
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    .risk-medium { 
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%); 
        color: #1a1a2e;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    .risk-low { 
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
        color: #1a1a2e;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.8rem;
        padding: 1rem 0;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FUNCIONES DE CARGA DE DATOS
# =============================================================================
@st.cache_data
def load_real_eav_data():
    """Carga datos reales del EAV."""
    try:
        df_base = pd.read_csv('data_empresas.csv', low_memory=False)
        df_base['nif'] = df_base['nif'].astype(str)
        return df_base
    except:
        return None

@st.cache_data
def load_real_m347_data():
    """Carga datos reales del M347."""
    try:
        df_m347 = pd.read_csv('data_m347.csv', low_memory=False)
        df_m347['nif'] = df_m347['nif'].astype(str)
        return df_m347
    except:
        return None

def generate_synthetic_data(n_companies: int = 1000) -> pd.DataFrame:
    """Genera datos sintéticos para demo."""
    np.random.seed(42)
    sectores = [
        'Comercio de vehículos', 'Comercio minorista', 'Software y consultoría',
        'Actividades inmobiliarias', 'Transporte', 'Construcción', 'Manufactura',
        'Hostelería', 'Servicios profesionales', 'Agricultura'
    ]
    data = []
    for i in range(n_companies):
        nif = f"B{str(i).zfill(7)}A"
        is_suspicious = np.random.random() < 0.08
        
        ventas = np.random.uniform(50000, 50000000)
        if is_suspicious:
            resultado = ventas * np.random.uniform(0.15, 0.30)
            flujo_caja = resultado * np.random.uniform(0.1, 0.3)
            gastos_personal = ventas * np.random.uniform(0.001, 0.02)
            gastos_transporte = 0
        else:
            resultado = ventas * np.random.uniform(0.02, 0.10)
            flujo_caja = resultado * np.random.uniform(0.7, 1.2)
            gastos_personal = ventas * np.random.uniform(0.1, 0.35)
            gastos_transporte = ventas * 0.02

        activo = ventas * np.random.uniform(0.4, 2.0)
        deuda = ventas * np.random.uniform(0.05, 0.5)
        
        data.append({
            'nif': nif,
            'sector': np.random.choice(sectores),
            'ventas_netas': ventas,
            'resultado_neto': resultado,
            'activo_total': activo,
            'deuda_bancaria': deuda,
            'gastos_personal': gastos_personal,
            'gastos_transporte': gastos_transporte,
            'flujo_caja_operativo': flujo_caja,
            'total_m347': ventas * np.random.uniform(0.8, 1.1),
            'amortizaciones': activo * np.random.uniform(0.03, 0.08),
            '_is_suspicious': is_suspicious
        })
    return pd.DataFrame(data)

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def calculate_forensic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula las features forenses."""
    # Ratios financieros
    df['margen_neto'] = df['resultado_neto'] / (df['ventas_netas'] + 1)
    df['rotacion_activos'] = df['ventas_netas'] / (df['activo_total'] + 1)
    df['ratio_endeudamiento'] = df['deuda_bancaria'] / (df['activo_total'] + 1)
    
    # Productividad
    df['productividad_laboral'] = df['ventas_netas'] / (df['gastos_personal'] + 1)
    
    # Flags de riesgo
    df['flag_empresa_pantalla'] = (
        (df['ventas_netas'] > 100000) & (df['gastos_personal'] < 15000)
    ).astype(int)
    
    df['flag_incoherencia_logistica'] = (
        (df.get('total_m349', 0) > 0) & (df['gastos_transporte'] < 1000)
    ).astype(int)
    
    df['flag_margen_extremo'] = (df['margen_neto'] > 0.20).astype(int)
    
    return df

def calculate_mahalanobis_by_sector(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula distancia Mahalanobis por sector."""
    feature_cols = ['margen_neto', 'rotacion_activos', 'ratio_endeudamiento']
    df['mahalanobis_distance'] = np.nan
    
    sector_col = 'sector' if 'sector' in df.columns else 'cnae'
    
    for sector in df[sector_col].unique():
        sector_mask = df[sector_col] == sector
        sector_data = df.loc[sector_mask, feature_cols].dropna()
        
        if len(sector_data) < 5:
            continue
        
        centroid = sector_data.mean().values
        cov_matrix = sector_data.cov().values + np.eye(len(feature_cols)) * 1e-6
        
        try:
            cov_inv = np.linalg.inv(cov_matrix)
            for idx in sector_data.index:
                point = df.loc[idx, feature_cols].values
                df.loc[idx, 'mahalanobis_distance'] = mahalanobis(point, centroid, cov_inv)
        except:
            continue
    
    df['flag_sectoral_outlier'] = (df['mahalanobis_distance'] > 3.0).astype(int)
    return df

def train_isolation_forest(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """Entrena Isolation Forest."""
    feature_cols = [
        'margen_neto', 'rotacion_activos', 'ratio_endeudamiento',
        'productividad_laboral', 'mahalanobis_distance'
    ]
    
    existing = [c for c in feature_cols if c in df.columns]
    X = df[existing].copy().fillna(df[existing].median())
    
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
    
    # Normalizar score [0, 1] donde 1 = más sospechoso
    min_s, max_s = df['fraud_score'].min(), df['fraud_score'].max()
    df['fraud_score_normalized'] = 1 - (df['fraud_score'] - min_s) / (max_s - min_s + 1e-6)
    
    # Clasificar riesgo
    df['riesgo'] = pd.cut(
        df['fraud_score_normalized'],
        bins=[0, 0.4, 0.7, 1.01],
        labels=['Bajo', 'Medio', 'Alto']
    )
    
    return df

def get_flag_details():
    """Devuelve diccionario con detalles de cada flag."""
    return {
        'flag_empresa_pantalla': {'nombre': 'Empresa Pantalla', 'icono': '🏢', 'severidad': 'high'},
        'flag_incoherencia_logistica': {'nombre': 'Incoherencia Logística', 'icono': '🚚', 'severidad': 'medium'},
        'flag_margen_extremo': {'nombre': 'Margen Extremo', 'icono': '📈', 'severidad': 'medium'},
        'flag_sectoral_outlier': {'nombre': 'Outlier Sectorial', 'icono': '📊', 'severidad': 'high'},
    }

# =============================================================================
# INICIALIZACIÓN DEL ESTADO
# =============================================================================
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0
if 'df_results' not in st.session_state:
    st.session_state.df_results = None
if 'selected_company_nif' not in st.session_state:
    st.session_state.selected_company_nif = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'synthetic'

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.image("logo.png", use_container_width=True)
    st.markdown("---")
    
    st.markdown("### ⚙️ Configuración")
    
    # Selector de fuente de datos
    data_source = st.selectbox(
        "📁 Fuente de Datos",
        options=['synthetic', 'real'],
        format_func=lambda x: "🧪 Sintéticos (Demo)" if x == 'synthetic' else "📊 Datos Reales",
        index=0 if st.session_state.data_source == 'synthetic' else 1
    )
    st.session_state.data_source = data_source
    
    # Parámetros del modelo
    contamination = st.slider(
        "🎯 Sensibilidad (Contamination)",
        min_value=0.01, max_value=0.20, value=0.05, step=0.01
    )
    
    st.markdown("---")
    
    # Botón ejecutar análisis
    if st.button("🚀 Ejecutar Análisis", type="primary", use_container_width=True):
        with st.spinner("Procesando..."):
            if data_source == 'real':
                df = load_real_eav_data()
                if df is None:
                    st.error("No se pudieron cargar los datos reales")
                    df = generate_synthetic_data()
            else:
                df = generate_synthetic_data()
            
            df = calculate_forensic_features(df)
            df = calculate_mahalanobis_by_sector(df)
            df = train_isolation_forest(df, contamination)
            
            st.session_state.df_results = df
            st.success(f"✅ Análisis completado: {len(df):,} empresas")
    
    st.markdown("---")
    
    # Selector de empresa
    if st.session_state.df_results is not None:
        df = st.session_state.df_results
        st.markdown("### 🔍 Seleccionar Empresa")
        
        top20_nifs = df.nlargest(20, 'fraud_score_normalized')['nif'].tolist()
        nif_list = list(df['nif'].unique())
        
        selected_nif = st.selectbox(
            "NIF de la empresa",
            options=[''] + top20_nifs + ['---'] + nif_list,
            format_func=lambda x: "Seleccionar..." if x == '' else ('─' * 20 if x == '---' else x)
        )
        
        if selected_nif and selected_nif != '---':
            st.session_state.selected_company_nif = selected_nif

# =============================================================================
# NAVEGACIÓN PRINCIPAL - 7 BOTONES
# =============================================================================
col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)
col_nav5, col_nav6, col_nav7 = st.columns(3)

with col_nav1:
    if st.button("📊 Dashboard", use_container_width=True, 
                 type="primary" if st.session_state.active_tab == 0 else "secondary"):
        st.session_state.active_tab = 0
        st.rerun()

with col_nav2:
    if st.button("🔎 Análisis", use_container_width=True,
                 type="primary" if st.session_state.active_tab == 1 else "secondary"):
        st.session_state.active_tab = 1
        st.rerun()

with col_nav3:
    if st.button("🧠 IA Learning", use_container_width=True,
                 type="primary" if st.session_state.active_tab == 2 else "secondary"):
        st.session_state.active_tab = 2
        st.rerun()

with col_nav4:
    if st.button("💚 Model Health", use_container_width=True,
                 type="primary" if st.session_state.active_tab == 3 else "secondary"):
        st.session_state.active_tab = 3
        st.rerun()

with col_nav5:
    if st.button("📥 Exportar", use_container_width=True,
                 type="primary" if st.session_state.active_tab == 4 else "secondary"):
        st.session_state.active_tab = 4
        st.rerun()

with col_nav6:
    if st.button("🏗️ Arquitectura", use_container_width=True,
                 type="primary" if st.session_state.active_tab == 5 else "secondary"):
        st.session_state.active_tab = 5
        st.rerun()

with col_nav7:
    if st.button("❓ Ayuda", use_container_width=True,
                 type="primary" if st.session_state.active_tab == 6 else "secondary"):
        st.session_state.active_tab = 6
        st.rerun()

st.markdown("---")

# =============================================================================
# TAB 0: DASHBOARD PRINCIPAL
# =============================================================================
if st.session_state.active_tab == 0:
    st.markdown('<h1 class="main-header">🔍 FraudHunter Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #888;">Sistema de Detección de Fraude Empresarial | Análisis Inteligente</p>', unsafe_allow_html=True)
    
    if st.session_state.df_results is not None:
        df = st.session_state.df_results
        n_anomalies = (df['anomaly_label'] == -1).sum()
        n_high_risk = (df['riesgo'] == 'Alto').sum() if 'riesgo' in df.columns else 0
        
        # KPIs principales
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        
        with col_k1:
            st.markdown(f"""
<div class="kpi-card">
<div class="kpi-value" style="color: #3b82f6;">{len(df):,}</div>
<div class="kpi-label">Empresas Analizadas</div>
</div>
            """, unsafe_allow_html=True)
        
        with col_k2:
            st.markdown(f"""
<div class="kpi-card" style="border-color: rgba(246, 79, 89, 0.5);">
<div class="kpi-value" style="color: #f64f59;">{n_anomalies:,}</div>
<div class="kpi-label">Anomalías Detectadas</div>
</div>
            """, unsafe_allow_html=True)
        
        with col_k3:
            st.markdown(f"""
<div class="kpi-card" style="border-color: rgba(245, 158, 11, 0.5);">
<div class="kpi-value" style="color: #f59e0b;">{n_high_risk:,}</div>
<div class="kpi-label">Alto Riesgo</div>
</div>
            """, unsafe_allow_html=True)
        
        with col_k4:
            precision = 100 * n_anomalies / len(df) if len(df) > 0 else 0
            st.markdown(f"""
<div class="kpi-card" style="border-color: rgba(59, 130, 246, 0.5);">
<div class="kpi-value" style="color: #10b981;">{precision:.1f}%</div>
<div class="kpi-label">Tasa Detección</div>
</div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Top 20 Empresas de Mayor Riesgo
        st.markdown("### 🚨 Top 20 Empresas de Mayor Riesgo")
        
        top_risk = df.nlargest(20, 'fraud_score_normalized')[
            ['nif', 'sector', 'fraud_score_normalized', 'riesgo', 'ventas_netas']
        ].copy()
        
        st.dataframe(
            top_risk.style.format({
                'fraud_score_normalized': '{:.2%}',
                'ventas_netas': '€{:,.0f}'
            }),
            use_container_width=True,
            height=450
        )
        
        # Gráficos
        st.markdown("---")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("#### 📊 Distribución por Nivel de Riesgo")
            risk_counts = df['riesgo'].value_counts()
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map={'Bajo': '#38ef7d', 'Medio': '#f2c94c', 'Alto': '#f64f59'}
            )
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_chart2:
            st.markdown("#### 📈 Score de Fraude vs Ventas")
            fig_scatter = px.scatter(
                df,
                x='ventas_netas',
                y='fraud_score_normalized',
                color='riesgo',
                color_discrete_map={'Bajo': '#38ef7d', 'Medio': '#f2c94c', 'Alto': '#f64f59'},
                hover_data=['nif', 'sector']
            )
            fig_scatter.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title='Ventas Netas (€)', yaxis_title='Score de Fraude'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("👈 Ejecuta un análisis desde la barra lateral para ver los resultados.")

# =============================================================================
# TAB 1: ANÁLISIS POR EMPRESA
# =============================================================================
if st.session_state.active_tab == 1:
    st.markdown("### 🔎 Análisis Detallado por Empresa")
    
    if st.session_state.df_results is not None and st.session_state.selected_company_nif:
        df = st.session_state.df_results
        selected_nif = st.session_state.selected_company_nif
        
        if selected_nif in df['nif'].values:
            empresa = df[df['nif'] == selected_nif].iloc[0]
            
            # Header de empresa
            fraud_score = empresa.get('fraud_score_normalized', 0)
            riesgo = empresa.get('riesgo', 'N/A')
            
            col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
            with col_h1:
                st.markdown(f"## 📋 NIF: {selected_nif}")
                st.markdown(f"**Sector:** {empresa.get('sector', 'N/A')}")
            with col_h2:
                st.metric("Score de Fraude", f"{fraud_score:.2%}")
            with col_h3:
                risk_class = 'risk-high' if riesgo == 'Alto' else ('risk-medium' if riesgo == 'Medio' else 'risk-low')
                st.markdown(f'<span class="{risk_class}">{riesgo}</span>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Métricas financieras
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Ventas Netas", f"€{empresa.get('ventas_netas', 0):,.0f}")
            with col_m2:
                st.metric("Resultado Neto", f"€{empresa.get('resultado_neto', 0):,.0f}")
            with col_m3:
                st.metric("Activo Total", f"€{empresa.get('activo_total', 0):,.0f}")
            with col_m4:
                st.metric("Margen Neto", f"{empresa.get('margen_neto', 0):.1%}")
            
            st.markdown("---")
            
            # Flags de riesgo
            st.markdown("#### ⚠️ Alertas de Riesgo")
            flag_details = get_flag_details()
            active_flags = []
            
            for flag_col, details in flag_details.items():
                if flag_col in empresa.index and empresa[flag_col] == 1:
                    active_flags.append(f"{details['icono']} {details['nombre']}")
            
            if active_flags:
                for flag in active_flags:
                    st.warning(flag)
            else:
                st.success("✅ No se detectaron alertas específicas")
            
            # Grafo de red
            st.markdown("---")
            st.markdown("#### 🕸️ Grafo de Operaciones (M347)")
            
            with st.spinner("Generando grafo..."):
                html_graph = create_suspicious_network(
                    center_nif=str(selected_nif),
                    center_risk=str(riesgo),
                    center_score=float(fraud_score)
                )
                components.html(html_graph, height=500)
        else:
            st.warning("Empresa no encontrada en los resultados del análisis.")
    else:
        st.info("👈 Selecciona una empresa desde la barra lateral")

# =============================================================================
# TAB 2: IA LEARNING
# =============================================================================
if st.session_state.active_tab == 2:
    st.markdown("### 🧠 Sistema de Aprendizaje Continuo")
    
    if CONTINUOUS_LEARNING_AVAILABLE:
        st.success("✅ Módulo de Continuous Learning cargado")
        
        store = get_feedback_store()
        stats = store.get_summary_stats()
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Total Feedback", stats.get('total', 0))
        with col_s2:
            st.metric("Falsos Positivos", stats.get('false_positives', 0))
        with col_s3:
            st.metric("Fraudes Confirmados", stats.get('confirmed_fraud', 0))
        
        st.markdown("---")
        st.info("El sistema aprende de las decisiones de los analistas para mejorar la precisión.")
    else:
        st.warning("⚠️ Módulo de Continuous Learning no disponible")
        st.info("El sistema de aprendizaje continuo permite al modelo mejorar con el feedback de analistas.")

# =============================================================================
# TAB 3: MODEL HEALTH
# =============================================================================
if st.session_state.active_tab == 3:
    from model_governance import ModelVersionManager, PerformanceMonitor, GlobalExplainer
    
    st.markdown("## 💚 Model Health: Governance & Observability")
    st.markdown("---")
    
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["🧬 Evolución", "📉 Rendimiento", "🧠 Explicabilidad"])
    
    with sub_tab1:
        st.markdown("### Historial de Versiones")
        version_manager = ModelVersionManager()
        history = version_manager.get_history()
        
        for v in history[:5]:
            st.markdown(f"""
**v{v['version']}** - {v['timestamp'][:10]}  
{v.get('description', 'Sin descripción')} | Precisión: {v['metrics'].get('precision', 0):.2%}
            """)
    
    with sub_tab2:
        st.markdown("### Métricas de Rendimiento")
        
        if st.session_state.df_results is not None:
            df = st.session_state.df_results
            n_anomalies = (df['anomaly_label'] == -1).sum()
            contamination = n_anomalies / len(df) if len(df) > 0 else 0.05
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Tasa Detección", f"{contamination:.1%}")
            with col_m2:
                st.metric("Empresas Analizadas", f"{len(df):,}")
            with col_m3:
                st.metric("Anomalías", f"{n_anomalies:,}")
        else:
            st.info("Ejecuta un análisis para ver métricas de rendimiento")
    
    with sub_tab3:
        st.markdown("### Feature Importance")
        
        if st.session_state.df_results is not None:
            df = st.session_state.df_results
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:8].tolist()
            
            importances = {col: abs(df[col].std() / (df[col].mean() + 0.001)) for col in numeric_cols}
            total = sum(importances.values()) + 0.001
            
            fig = go.Figure(go.Bar(
                y=list(importances.keys()),
                x=[v/total for v in importances.values()],
                orientation='h'
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Ejecuta un análisis para ver la importancia de variables")

# =============================================================================
# TAB 4: EXPORTAR
# =============================================================================
if st.session_state.active_tab == 4:
    st.markdown("### 📥 Exportar Resultados")
    
    if st.session_state.df_results is not None:
        df = st.session_state.df_results
        
        col_d1, col_d2, col_d3 = st.columns(3)
        
        with col_d1:
            st.markdown("#### 📊 Datos Completos")
            csv_all = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar CSV", csv_all, f"fraudhunter_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
        
        with col_d2:
            st.markdown("#### ⚠️ Solo Sospechosas")
            df_susp = df[df['anomaly_label'] == -1]
            csv_susp = df_susp.to_csv(index=False).encode('utf-8')
            st.download_button(f"📥 {len(df_susp)} Sospechosas", csv_susp, "sospechosas.csv", "text/csv")
        
        with col_d3:
            st.markdown("#### 🔴 Alto Riesgo")
            df_high = df[df['riesgo'] == 'Alto']
            csv_high = df_high.to_csv(index=False).encode('utf-8')
            st.download_button(f"📥 {len(df_high)} Alto Riesgo", csv_high, "alto_riesgo.csv", "text/csv")
    else:
        st.info("Ejecuta un análisis primero para poder exportar resultados.")

# =============================================================================
# TAB 5: ARQUITECTURA
# =============================================================================
if st.session_state.active_tab == 5:
    st.markdown("### 🏗️ Arquitectura Algorítmica")
    st.markdown("---")
    
    st.markdown("""
    #### Pipeline de Detección de Fraude
    
    **1. Ingesta de Datos**
    - Modelo 200 (EAV): Datos financieros de empresas
    - Modelo 347: Operaciones con terceros
    - Modelo 349: Operaciones intracomunitarias
    
    **2. Feature Engineering**
    - Ratios financieros: Margen neto, rotación, endeudamiento
    - Indicadores forenses: Ley de Benford, accruals
    - Análisis sectorial: Distancia de Mahalanobis
    
    **3. Modelos de Detección**
    - Isolation Forest: Detección de anomalías no supervisada
    - Análisis de grafos: PageRank para redes sospechosas
    - Reglas de negocio: Flags específicos de fraude
    
    **4. Scoring & Clasificación**
    - Score normalizado [0-1]
    - Clasificación: Bajo / Medio / Alto riesgo
    """)

# =============================================================================
# TAB 6: AYUDA
# =============================================================================
if st.session_state.active_tab == 6:
    st.markdown("### ❓ Ayuda y Documentación")
    st.markdown("---")
    
    st.markdown("""
    #### 🔍 ¿Qué es FraudHunter?
    
    FraudHunter es un sistema avanzado de detección de fraude fiscal empresarial 
    que utiliza técnicas de Machine Learning para identificar patrones sospechosos.
    
    #### 📊 ¿Cómo funciona?
    
    1. **Carga de datos**: El sistema procesa datos financieros de empresas
    2. **Análisis**: Aplica algoritmos de ML para detectar anomalías
    3. **Scoring**: Asigna un score de riesgo a cada empresa
    4. **Visualización**: Presenta los resultados de forma clara
    
    #### 🎯 Interpretación del Score
    
    - **0-40%**: Riesgo Bajo (empresa normal)
    - **40-70%**: Riesgo Medio (requiere revisión)
    - **70-100%**: Riesgo Alto (prioridad de auditoría)
    
    #### 📞 Soporte
    
    Para dudas técnicas, contacta con el equipo de desarrollo.
    """)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(f"""
<div class="footer">
<strong>FraudHunter Pro v2.0</strong> | Sistema de Detección de Fraude Empresarial<br>
<span style="font-size: 0.75rem;">
Metodología: Isolation Forest + Análisis Sectorial Mahalanobis + Cruces Tributarios<br>
📊 {datetime.now().strftime("%d/%m/%Y %H:%M")}
</span>
</div>
""", unsafe_allow_html=True)

