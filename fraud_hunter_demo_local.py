"""
FraudHunter Core - Demo Local (Sin PySpark)
============================================
Versión de demostración que usa Pandas en lugar de PySpark
para probar la lógica de detección de fraude localmente.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from scipy.spatial.distance import mahalanobis
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Suppress warnings for cleaner output  
warnings.filterwarnings('ignore')

print("=" * 70)
print("FRAUDHUNTER CORE - Demo Local (Pandas)")
print("=" * 70)


# =============================================================================
# GENERADOR DE DATOS DUMMY
# =============================================================================

def generate_dummy_data(n_companies: int = 100) -> pd.DataFrame:
    """Genera datos dummy con empresas normales y sospechosas."""
    np.random.seed(42)
    cnaes = ['4511', '4719', '6201', '6831', '4941']
    
    data = []
    for i in range(n_companies):
        nif = f"B{str(i).zfill(7)}A"
        is_suspicious = np.random.random() < 0.1  # 10% sospechosos
        
        ventas = np.random.uniform(100000, 10000000)
        if is_suspicious:
            # Empresas sospechosas: alto margen pero poco flujo caja
            resultado = ventas * np.random.uniform(0.15, 0.25)
            flujo_caja = resultado * np.random.uniform(0.1, 0.3)
            gastos_personal = ventas * np.random.uniform(0.001, 0.01)
            gastos_transporte = 0
            pct_redondos = np.random.uniform(0.4, 0.7)  # Muchos números redondos
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
            'cnae': np.random.choice(cnaes),
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
            '_is_suspicious': is_suspicious  # Ground truth for validation
        })
    
    return pd.DataFrame(data)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def calculate_forensic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula las features forenses."""
    print("\n📊 Calculando features forenses...")
    
    # Cobertura de ventas (M347 / Ventas)
    df['cobertura_ventas'] = df['total_m347'] / df['ventas_netas']
    
    # Incoherencia logística (M349 > 0 pero sin gastos transporte)
    df['flag_incoherencia_logistica'] = (
        (df['total_m349'] > 0) & (df['gastos_transporte'] < 1000)
    ).astype(int)
    
    # Accruals ratio (manipulación contable)
    df['accruals_ratio'] = (
        (df['resultado_neto'] - df['flujo_caja_operativo']) / df['activo_total']
    )
    
    # Hidden debt (tasa interés implícita)
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
    
    # Números redondos (ya calculados en dummy data)
    df['flag_numeros_redondos'] = (df['pct_numeros_redondos'] > 0.30).astype(int)
    
    # Ratios financieros
    df['margen_neto'] = df['resultado_neto'] / df['ventas_netas']
    df['rotacion_activos'] = df['ventas_netas'] / df['activo_total']
    df['ratio_endeudamiento'] = df['deuda_bancaria'] / df['activo_total']
    
    print(f"   ✓ {len(df.columns)} columnas calculadas")
    return df


# =============================================================================
# DISTANCIA MAHALANOBIS POR SECTOR
# =============================================================================

def calculate_mahalanobis_by_sector(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula distancia Mahalanobis por sector CNAE."""
    print("\n📐 Calculando distancias sectoriales (Mahalanobis)...")
    
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
    
    print(f"   ✓ Distancias calculadas para {len(df['cnae'].unique())} sectores")
    return df


# =============================================================================
# MODELO ISOLATION FOREST
# =============================================================================

def train_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    """Entrena Isolation Forest y genera predicciones."""
    print("\n🤖 Entrenando modelo Isolation Forest...")
    
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
    
    # RobustScaler para manejar outliers
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Entrenar modelo
    model = IsolationForest(
        contamination='auto',
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    df['anomaly_label'] = model.fit_predict(X_scaled)
    df['fraud_score'] = model.decision_function(X_scaled)
    
    # Normalizar score [0, 1] donde 1 = más sospechoso
    min_score = df['fraud_score'].min()
    max_score = df['fraud_score'].max()
    df['fraud_score_normalized'] = 1 - (
        (df['fraud_score'] - min_score) / (max_score - min_score)
    )
    
    n_anomalies = (df['anomaly_label'] == -1).sum()
    print(f"   ✓ Modelo entrenado")
    print(f"   ✓ Anomalías detectadas: {n_anomalies} ({100*n_anomalies/len(df):.1f}%)")
    
    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Paso 1: Generar datos
    print("\n📁 Generando datos dummy (100 empresas)...")
    df = generate_dummy_data(100)
    print(f"   ✓ {len(df)} empresas generadas (10% sospechosas)")
    
    # Paso 2: Feature Engineering
    df = calculate_forensic_features(df)
    
    # Paso 3: Benchmarking sectorial
    df = calculate_mahalanobis_by_sector(df)
    
    # Paso 4: Detección de anomalías
    df = train_isolation_forest(df)
    
    # ==========================================================================
    # RESULTADOS
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("🔍 TOP 10 EMPRESAS MÁS SOSPECHOSAS")
    print("=" * 70)
    
    top_suspicious = df.nsmallest(10, 'fraud_score')[
        ['nif', 'cnae', 'ventas_netas', 'fraud_score_normalized', 
         'anomaly_label', '_is_suspicious']
    ].copy()
    
    top_suspicious['ventas_netas'] = top_suspicious['ventas_netas'].apply(
        lambda x: f"€{x:,.0f}"
    )
    top_suspicious['fraud_score_normalized'] = top_suspicious['fraud_score_normalized'].apply(
        lambda x: f"{x:.3f}"
    )
    top_suspicious['Ground Truth'] = top_suspicious['_is_suspicious'].apply(
        lambda x: '⚠️ SOSPECHOSA' if x else '✅ Normal'
    )
    top_suspicious = top_suspicious.drop('_is_suspicious', axis=1)
    
    print(top_suspicious.to_string(index=False))
    
    # Resumen de flags
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE FLAGS DE RIESGO")
    print("=" * 70)
    
    flag_cols = [c for c in df.columns if c.startswith('flag_')]
    for flag in flag_cols:
        count = df[flag].sum()
        pct = 100 * count / len(df)
        print(f"   {flag}: {count} empresas ({pct:.1f}%)")
    
    # Validación del modelo
    print("\n" + "=" * 70)
    print("✅ VALIDACIÓN DEL MODELO")
    print("=" * 70)
    
    # Confusion matrix simplificada
    true_pos = ((df['anomaly_label'] == -1) & (df['_is_suspicious'] == True)).sum()
    true_neg = ((df['anomaly_label'] == 1) & (df['_is_suspicious'] == False)).sum()
    false_pos = ((df['anomaly_label'] == -1) & (df['_is_suspicious'] == False)).sum()
    false_neg = ((df['anomaly_label'] == 1) & (df['_is_suspicious'] == True)).sum()
    
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    
    print(f"   Verdaderos Positivos: {true_pos}")
    print(f"   Verdaderos Negativos: {true_neg}")
    print(f"   Falsos Positivos: {false_pos}")
    print(f"   Falsos Negativos: {false_neg}")
    print(f"   Precision: {precision:.2%}")
    print(f"   Recall: {recall:.2%}")
    
    print("\n" + "=" * 70)
    print("🎉 DEMO COMPLETADA EXITOSAMENTE")
    print("=" * 70)
