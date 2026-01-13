"""
FraudHunter - Executive Dashboard
==================================
Sistema de detección de fraude empresarial con interfaz ejecutiva.
Diseñado para presentaciones ante comité de dirección.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from scipy.spatial.distance import mahalanobis
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================

st.set_page_config(
    page_title="FraudHunter Pro 🔍",
    page_icon="🔍",
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

# Header
# Header
col_logo, col_title, col_status = st.columns([1, 5, 2])

with col_logo:
    try:
        st.image("logo.png", width=100)
    except:
        st.write("🔍")

with col_title:
    st.markdown('<h1 class="main-header" style="margin-top: 0;">FraudHunter Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Sistema Avanzado de Detección de Fraude Empresarial con Machine Learning</p>', unsafe_allow_html=True)

with col_status:
    st.markdown(f"""
        <div style="text-align: right; padding-top: 1rem;">
            <span class="status-badge status-demo">⚡ MODO DEMO</span>
            <p style="color: #6c757d; font-size: 0.8rem; margin-top: 0.5rem;">
                {datetime.now().strftime("%d/%m/%Y %H:%M")}
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    try:
        st.image("logo.png", use_container_width=True)
    except:
        pass
st.sidebar.header("⚙️ Configuración del Análisis")

# Enlace a documentación
st.sidebar.markdown("---")
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
    
    [📑 Ver Presentación Técnica Completa](https://github.com/tonhogonzalez/Antifraude_Modelos_Empresas/blob/main/Presentaci%C3%B3n_solucion.html)
    """)
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

# Ejecutar análisis
if st.session_state.get('run_analysis', False) or 'df_results' not in st.session_state:
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
        
        df = calculate_forensic_features(df)
        df = calculate_mahalanobis_by_sector(df)
        df = train_isolation_forest(df, contamination)
        st.session_state.df_results = df
        st.session_state.run_analysis = False

df = st.session_state.df_results

import streamlit.components.v1 as components

# =============================================================================
# NAVEGACIÓN CON PERSISTENCIA DE TAB
# =============================================================================

# Inicializar tab activo en session_state
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

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
    if st.button("📑 Presentación", use_container_width=True,
                 type="primary" if st.session_state.active_tab == 3 else "secondary"):
        st.session_state.active_tab = 3
        st.rerun()

st.markdown("---")

# =============================================================================
# TAB 4: PRESENTACIÓN (Renderizado de HTML)
# =============================================================================
if st.session_state.active_tab == 3:
    try:
        # Cargar el archivo HTML
        base_path = Path(__file__).parent if '__file__' in dir() else Path('.')
        html_path = base_path / 'Presentación_solucion.html'
        
        if html_path.exists():
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Renderizar HTML
            st.markdown("### 📑 Presentación del Proyecto")
            components.html(html_content, height=800, scrolling=True)
        else:
            st.error("Archivo de presentación no encontrado.")
    except Exception as e:
        st.error(f"Error al cargar la presentación: {e}")

# =============================================================================
# TAB 1: DASHBOARD EJECUTIVO
# =============================================================================

if st.session_state.active_tab == 0:
    # Sección de Capacidades del Sistema
    st.markdown("### 🧠 Capacidades de Detección")
    
    capabilities = [
        ("🔬", "Isolation Forest", "ML no supervisado"),
        ("📊", "Análisis Sectorial", "Mahalanobis distance"),
        ("🏭", "Empresas Pantalla", "Ratio ventas/personal"),
        ("📦", "Incoherencia Logística", "Cruce M349/transporte"),
        ("💳", "Deuda Oculta", "Tasa interés implícita"),
        ("🔢", "Números Redondos", "Detección Benford"),
        ("📋", "Cobertura M347", "Cruce ventas/operaciones"),
        ("📈", "Manipulación Contable", "Accruals anómalos")
    ]
    
    cols = st.columns(8)
    for i, (icon, title, desc) in enumerate(capabilities):
        with cols[i]:
            st.markdown(f"""
                <div class="capability-card">
                    <div class="capability-icon">{icon}</div>
                    <div class="capability-title">{title}</div>
                    <div class="capability-desc">{desc}</div>
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
    
    # Obtener los flag details para crear la columna de alertas
    flag_details = get_flag_details()
    flag_cols = [col for col in flag_details.keys() if col in df.columns]
    
    top_suspicious = df.nsmallest(20, 'fraud_score')[
        ['nif', 'sector', 'ventas_netas', 'resultado_neto', 
         'fraud_score_normalized', 'riesgo'] + flag_cols
    ].copy()
    
    # Crear columna de alertas con iconos
    def build_alerts(row):
        alerts = []
        for flag_col in flag_cols:
            if row.get(flag_col, 0) == 1:
                alerts.append(flag_details[flag_col]['icono'])
        return ' '.join(alerts) if alerts else '✅'
    
    top_suspicious['Alertas'] = top_suspicious.apply(build_alerts, axis=1)
    
    top_suspicious['ventas_netas'] = top_suspicious['ventas_netas'].apply(lambda x: f"€{x:,.0f}")
    top_suspicious['resultado_neto'] = top_suspicious['resultado_neto'].apply(lambda x: f"€{x:,.0f}")
    top_suspicious['Score'] = top_suspicious['fraud_score_normalized'].apply(lambda x: f"{x:.3f}")
    
    display_df = top_suspicious[['nif', 'sector', 'Alertas', 'ventas_netas', 'resultado_neto', 'Score', 'riesgo']].copy()
    display_df.columns = ['NIF', 'Sector', '🚨 Alertas', 'Ventas', 'Resultado', 'Score', 'Riesgo']
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=450,
        hide_index=True
    )


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
        
        selected_empresa = st.selectbox(
            "🏢 Seleccionar Empresa",
            options=empresa_options,
            help="Las empresas están ordenadas por nivel de riesgo (mayor a menor)",
            key="empresa_selector_tab2"
        )
    
    with col_select2:
        st.markdown("<br>", unsafe_allow_html=True)
        show_only_high_risk = st.checkbox("Mostrar solo alto riesgo", value=False)
    
    if selected_empresa:
        nif_selected = selected_empresa.split(" - ")[0]
        empresa_data = df[df['nif'] == nif_selected].iloc[0]
        
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
            
            st.markdown("#### 🕸️ Análisis de Grafo (Simulado)")
            st.info("Visualización de relaciones transaccionales (Modelo 347). En la versión completa, aquí se muestra el grafo interactivo de relaciones con empresas pantalla.")


# =============================================================================
# TAB 3: EXPORTAR RESULTADOS
# =============================================================================

if st.session_state.active_tab == 2:
    st.markdown("### 📥 Exportar Resultados del Análisis")
    
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
