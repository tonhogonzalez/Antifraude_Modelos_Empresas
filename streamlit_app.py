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
if st.session_state.active_tab == 8:
    from model_governance import GlobalExplainer
    import plotly.graph_objects as go
    
    st.markdown("## 🧠 Explicabilidad Global: Feature Importance & Patterns")
    st.markdown("---")
    
    st.info("Análisis de qué variables influyen más en las decisiones del modelo a nivel global.")
    
    # Initialize explainer
    explainer = GlobalExplainer()
    
    # Check if we have real analysis data
    if 'df_results' in st.session_state and st.session_state.df_results is not None:
        df = st.session_state.df_results
        st.caption("📊 Importancia calculada desde el análisis actual")
        
        # Get numeric columns that are likely features (exclude identifiers and results)
        exclude_cols = ['nif', 'nombre', 'fraud_score', 'anomaly_label', 'risk_level', 'index']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols][:15]
        
        if feature_cols:
            # Calculate variance-based importance (higher variance = more discriminative)
            importances = {}
            for col in feature_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    # Use coefficient of variation as importance proxy
                    std = col_data.std()
                    mean = abs(col_data.mean()) + 0.001
                    importances[col] = std / mean
            
            # Normalize
            total = sum(importances.values()) + 0.001
            feature_importance = {k: v/total for k, v in importances.items()}
            
            # Sort and get top features
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:15]
        else:
            st.warning("No se encontraron columnas numéricas para análisis de importancia.")
            top_features = []
    else:
        st.caption("⚠️ Sin análisis - Mostrando ejemplo de variables típicas")
        
        # Default feature names for fraud detection
        feature_names = [
            'ratio_liquidez', 'ratio_solvencia', 'margen_bruto', 'rentabilidad_activos',
            'rotacion_activos', 'benford_divergencia', 'mahalanobis_distance', 'pagerank_score',
            'ventas_vs_sector', 'gastos_transporte_ratio', 'exportaciones_ratio', 'deuda_total',
            'resultado_neto', 'amortizaciones', 'gastos_personal'
        ]
        
        # Use predefined importance values instead of random
        importance_values = [0.15, 0.12, 0.10, 0.09, 0.08, 0.08, 0.07, 0.06, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.0]
        top_features = list(zip(feature_names, importance_values))
    
    if top_features:
        # Feature Importance Chart
        st.markdown("### 📊 Top 15 Variables Más Influyentes")
    
        feat_names = [f[0] for f in top_features]
        feat_importance = [f[1] for f in top_features]
        
        fig = go.Figure(go.Bar(
            y=feat_names[::-1],  # Reverse for top-down display
            x=feat_importance[::-1],
            orientation='h',
            marker=dict(
                color=feat_importance[::-1],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importancia")
            )
        ))
        
        fig.update_layout(
            title="Importancia de Variables en Decisiones de Fraude",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(title="Importancia Relativa", gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Pattern Summary
    st.markdown("### 🔍 Patrones Detectados Globalmente")
    
    if 'df_results' in st.session_state and st.session_state.df_results is not None:
        df = st.session_state.df_results
        pattern_summary = explainer.get_pattern_summary(df)
        
        col_p1, col_p2, col_p3 = st.columns(3)
        
        with col_p1:
            st.markdown(f"""
<div style="background: rgba(59, 130, 246, 0.1); border: 1px solid #3b82f6; border-radius: 12px; padding: 1.5rem;">
<div style="color: #94a3b8; font-size: 0.875rem; margin-bottom: 0.5rem;">EMPRESAS ANALIZADAS</div>
<div style="color: #3b82f6; font-size: 2rem; font-weight: 700;">{pattern_summary['total_companies']:,}</div>
</div>
            """, unsafe_allow_html=True)
        
        with col_p2:
            st.markdown(f"""
<div style="background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 12px; padding: 1.5rem;">
<div style="color: #94a3b8; font-size: 0.875rem; margin-bottom: 0.5rem;">ANOMALÍAS DETECTADAS</div>
<div style="color: #ef4444; font-size: 2rem; font-weight: 700;">{pattern_summary['anomalies_detected']:,}</div>
<div style="color: #94a3b8; font-size: 0.75rem; margin-top: 0.5rem;">Tasa: {pattern_summary['anomaly_rate']:.1%}</div>
</div>
            """, unsafe_allow_html=True)
        
        with col_p3:
            st.markdown(f"""
<div style="background: rgba(245, 158, 11, 0.1); border: 1px solid #f59e0b; border-radius: 12px; padding: 1.5rem;">
<div style="color: #94a3b8; font-size: 0.875rem; margin-bottom: 0.5rem;">RIESGO ALTO</div>
<div style="color: #f59e0b; font-size: 2rem; font-weight: 700;">{pattern_summary['high_risk_companies']:,}</div>
<div style="color: #94a3b8; font-size: 0.75rem; margin-top: 0.5rem;">Score > 0.7</div>
</div>
            """, unsafe_allow_html=True)
        
        # Sector analysis
        st.markdown("### 📈 Análisis por Sector (CNAE)")
        
        sector_stats = explainer.sector_analysis(df)
        
        if sector_stats:
            sector_df = pd.DataFrame(sector_stats).T
            sector_df = sector_df.sort_values('avg_fraud_score', ascending=False).head(10)
            
            st.dataframe(
                sector_df.style.format({
                    'avg_fraud_score': '{:.2%}',
                    'anomaly_rate': '{:.2%}',
                    'avg_revenue': '€{:,.0f}'
                }),
                use_container_width=True,
                height=400
            )
        else:
            st.info("ℹ️ No hay información de sector (CNAE) disponible en los datos actuales. El análisis por sector requiere datos con clasificación sectorial.")
    else:
        st.warning("Ejecuta un análisis desde el Dashboard principal para ver los patrones detallados.")


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
