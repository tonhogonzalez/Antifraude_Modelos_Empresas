"""
FraudHunter Core - Demo Interactiva con Reporte HTML
=====================================================
Genera un reporte visual HTML con los resultados del análisis.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from scipy.spatial.distance import mahalanobis
import webbrowser
import os

warnings.filterwarnings('ignore')

# =============================================================================
# GENERADOR DE DATOS
# =============================================================================

def generate_dummy_data(n_companies: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    cnaes = ['4511', '4719', '6201', '6831', '4941']
    cnae_names = {
        '4511': 'Venta vehiculos',
        '4719': 'Comercio minorista',
        '6201': 'Programacion',
        '6831': 'Agencia inmobiliaria',
        '4941': 'Transporte mercancias'
    }
    
    data = []
    for i in range(n_companies):
        nif = f"B{str(i).zfill(7)}A"
        is_suspicious = np.random.random() < 0.1
        cnae = np.random.choice(cnaes)
        
        ventas = np.random.uniform(100000, 10000000)
        if is_suspicious:
            resultado = ventas * np.random.uniform(0.15, 0.25)
            flujo_caja = resultado * np.random.uniform(0.1, 0.3)
            gastos_personal = ventas * np.random.uniform(0.001, 0.01)
            gastos_transporte = 0
            pct_redondos = np.random.uniform(0.4, 0.7)
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
            'sector': cnae_names[cnae],
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


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df['cobertura_ventas'] = df['total_m347'] / df['ventas_netas']
    df['flag_incoherencia_logistica'] = ((df['total_m349'] > 0) & (df['gastos_transporte'] < 1000)).astype(int)
    df['accruals_ratio'] = (df['resultado_neto'] - df['flujo_caja_operativo']) / df['activo_total']
    df['tasa_interes_implicita'] = np.where(df['deuda_bancaria'] > 0, df['intereses_pagados'] / df['deuda_bancaria'], 0)
    df['flag_hidden_debt'] = (df['tasa_interes_implicita'] > 0.10).astype(int)
    df['productividad_laboral'] = np.where(df['gastos_personal'] > 0, df['ventas_netas'] / df['gastos_personal'], np.nan)
    df['flag_empresa_pantalla'] = ((df['ventas_netas'] > 100000) & (df['gastos_personal'] < 10000)).astype(int)
    df['flag_numeros_redondos'] = (df['pct_numeros_redondos'] > 0.30).astype(int)
    df['margen_neto'] = df['resultado_neto'] / df['ventas_netas']
    df['rotacion_activos'] = df['ventas_netas'] / df['activo_total']
    df['ratio_endeudamiento'] = df['deuda_bancaria'] / df['activo_total']
    
    # Mahalanobis
    feature_cols = ['margen_neto', 'rotacion_activos', 'ratio_endeudamiento']
    df['mahalanobis_distance'] = np.nan
    for cnae in df['cnae'].unique():
        sector_data = df.loc[df['cnae'] == cnae, feature_cols].dropna()
        if len(sector_data) < 3:
            continue
        centroid = sector_data.mean().values
        cov_matrix = sector_data.cov().values + np.eye(3) * 1e-6
        try:
            cov_inv = np.linalg.inv(cov_matrix)
            for idx in sector_data.index:
                point = df.loc[idx, feature_cols].values
                df.loc[idx, 'mahalanobis_distance'] = mahalanobis(point, centroid, cov_inv)
        except:
            continue
    df['flag_sectoral_outlier'] = (df['mahalanobis_distance'] > 3.0).astype(int)
    
    return df


def train_model(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        'cobertura_ventas', 'accruals_ratio', 'tasa_interes_implicita',
        'productividad_laboral', 'pct_numeros_redondos', 'mahalanobis_distance',
        'flag_incoherencia_logistica', 'flag_hidden_debt', 'flag_empresa_pantalla',
        'flag_numeros_redondos', 'flag_sectoral_outlier'
    ]
    existing_cols = [c for c in feature_cols if c in df.columns]
    X = df[existing_cols].fillna(df[existing_cols].median())
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = IsolationForest(contamination='auto', n_estimators=100, random_state=42)
    df['anomaly_label'] = model.fit_predict(X_scaled)
    df['fraud_score'] = model.decision_function(X_scaled)
    
    min_s, max_s = df['fraud_score'].min(), df['fraud_score'].max()
    df['fraud_score_normalized'] = 1 - ((df['fraud_score'] - min_s) / (max_s - min_s))
    
    return df


def generate_html_report(df: pd.DataFrame) -> str:
    # Stats
    n_total = len(df)
    n_anomalies = (df['anomaly_label'] == -1).sum()
    true_pos = ((df['anomaly_label'] == -1) & (df['_is_suspicious'])).sum()
    true_neg = ((df['anomaly_label'] == 1) & (~df['_is_suspicious'])).sum()
    false_pos = ((df['anomaly_label'] == -1) & (~df['_is_suspicious'])).sum()
    false_neg = ((df['anomaly_label'] == 1) & (df['_is_suspicious'])).sum()
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    
    # Top suspicious
    top = df.nsmallest(15, 'fraud_score')[['nif', 'sector', 'ventas_netas', 'fraud_score_normalized', 'anomaly_label', '_is_suspicious']].copy()
    
    top_rows = ""
    for _, row in top.iterrows():
        risk_class = "high-risk" if row['anomaly_label'] == -1 else "normal"
        ground_truth = "SOSPECHOSA (Ground Truth)" if row['_is_suspicious'] else "Normal"
        gt_class = "suspicious" if row['_is_suspicious'] else ""
        score_bar = f"<div class='score-bar'><div class='score-fill' style='width: {row['fraud_score_normalized']*100:.0f}%'></div></div>"
        top_rows += f"""
        <tr class="{risk_class}">
            <td><code>{row['nif']}</code></td>
            <td>{row['sector']}</td>
            <td>EUR {row['ventas_netas']:,.0f}</td>
            <td>{score_bar} <span class='score-value'>{row['fraud_score_normalized']:.3f}</span></td>
            <td class="{gt_class}">{ground_truth}</td>
        </tr>"""
    
    # Flags summary
    flag_cols = [c for c in df.columns if c.startswith('flag_')]
    flags_html = ""
    for flag in flag_cols:
        count = int(df[flag].sum())
        pct = 100 * count / n_total
        flag_name = flag.replace('flag_', '').replace('_', ' ').title()
        flags_html += f"""
        <div class="flag-item">
            <div class="flag-name">{flag_name}</div>
            <div class="flag-bar"><div class="flag-fill" style="width: {pct}%"></div></div>
            <div class="flag-count">{count} ({pct:.1f}%)</div>
        </div>"""
    
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FraudHunter Core - Resultados</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ 
            font-family: 'Segoe UI', system-ui, sans-serif; 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 40px 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ 
            font-size: 2.5rem; 
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        .subtitle {{ color: #888; margin-bottom: 40px; font-size: 1.1rem; }}
        
        .stats-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 20px; 
            margin-bottom: 40px;
        }}
        .stat-card {{ 
            background: rgba(255,255,255,0.05); 
            border-radius: 16px; 
            padding: 24px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .stat-value {{ font-size: 2.5rem; font-weight: bold; color: #00d9ff; }}
        .stat-label {{ color: #888; margin-top: 8px; }}
        .stat-card.danger .stat-value {{ color: #ff4757; }}
        .stat-card.success .stat-value {{ color: #2ed573; }}
        
        .section {{ 
            background: rgba(255,255,255,0.03); 
            border-radius: 16px; 
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.08);
        }}
        .section-title {{ font-size: 1.4rem; margin-bottom: 20px; color: #fff; }}
        
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ text-align: left; padding: 12px; color: #888; border-bottom: 1px solid rgba(255,255,255,0.1); }}
        td {{ padding: 14px 12px; border-bottom: 1px solid rgba(255,255,255,0.05); }}
        tr.high-risk {{ background: rgba(255,71,87,0.1); }}
        tr.high-risk td:first-child {{ border-left: 3px solid #ff4757; }}
        .suspicious {{ color: #ff4757; font-weight: bold; }}
        code {{ background: rgba(0,217,255,0.1); padding: 4px 8px; border-radius: 4px; color: #00d9ff; }}
        
        .score-bar {{ 
            width: 100px; height: 8px; background: rgba(255,255,255,0.1); 
            border-radius: 4px; display: inline-block; vertical-align: middle; margin-right: 10px;
        }}
        .score-fill {{ height: 100%; background: linear-gradient(90deg, #2ed573, #ffa502, #ff4757); border-radius: 4px; }}
        .score-value {{ color: #fff; font-weight: bold; }}
        
        .flag-item {{ display: flex; align-items: center; padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }}
        .flag-name {{ width: 200px; color: #fff; }}
        .flag-bar {{ flex: 1; height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; margin: 0 20px; }}
        .flag-fill {{ height: 100%; background: #ff4757; border-radius: 4px; }}
        .flag-count {{ width: 100px; text-align: right; color: #888; }}
        
        .metrics-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
        .metric {{ padding: 20px; background: rgba(255,255,255,0.03); border-radius: 12px; }}
        .metric-name {{ color: #888; margin-bottom: 8px; }}
        .metric-value {{ font-size: 1.8rem; font-weight: bold; }}
        .metric-value.good {{ color: #2ed573; }}
        .metric-value.warning {{ color: #ffa502; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>FraudHunter Core</h1>
        <p class="subtitle">Sistema de Deteccion de Fraude Empresarial - Resultados del Analisis</p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{n_total}</div>
                <div class="stat-label">Empresas Analizadas</div>
            </div>
            <div class="stat-card danger">
                <div class="stat-value">{n_anomalies}</div>
                <div class="stat-label">Anomalias Detectadas</div>
            </div>
            <div class="stat-card success">
                <div class="stat-value">{precision:.0%}</div>
                <div class="stat-label">Precision del Modelo</div>
            </div>
            <div class="stat-card success">
                <div class="stat-value">{recall:.0%}</div>
                <div class="stat-label">Recall del Modelo</div>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">Top 15 Empresas Mas Sospechosas</h2>
            <table>
                <thead>
                    <tr>
                        <th>NIF</th>
                        <th>Sector</th>
                        <th>Ventas</th>
                        <th>Score de Fraude</th>
                        <th>Ground Truth</th>
                    </tr>
                </thead>
                <tbody>
                    {top_rows}
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2 class="section-title">Resumen de Flags de Riesgo</h2>
            {flags_html}
        </div>
        
        <div class="section">
            <h2 class="section-title">Metricas de Validacion</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-name">Verdaderos Positivos</div>
                    <div class="metric-value good">{true_pos}</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Verdaderos Negativos</div>
                    <div class="metric-value good">{true_neg}</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Falsos Positivos</div>
                    <div class="metric-value warning">{false_pos}</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Falsos Negativos</div>
                    <div class="metric-value warning">{false_neg}</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""
    return html


if __name__ == "__main__":
    print("Generando datos...")
    df = generate_dummy_data(100)
    
    print("Calculando features forenses...")
    df = calculate_features(df)
    
    print("Entrenando modelo Isolation Forest...")
    df = train_model(df)
    
    print("Generando reporte HTML...")
    html = generate_html_report(df)
    
    report_path = os.path.join(os.path.dirname(__file__), "fraud_report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Reporte guardado en: {report_path}")
    print("Abriendo en navegador...")
    webbrowser.open(f'file://{report_path}')
