import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json

# ==========================================
# 1. GENERADOR DE DATOS "VERTICALES" (EAV)
# ==========================================
# Simulamos cómo te llegan los datos de la base de datos corporativa
# Esquema: [NIF, MODELO, EJERCICIO, CODIGO_CASILLA, VALOR]

def generar_datos_prueba():
    data = []
    
    # --- ESCENARIO A: EMPRESA SANA (SANA_SA) ---
    # Coherente: Vende 1M, Gana 100k, Tiene Caja, Paga Transporte.
    nif = "A_SANA"
    data.extend([
        # M200 (Sociedades)
        {'nif': nif, 'mod': '200', 'cod': 'C00255', 'val': 1000000}, # Cifra Negocio
        {'nif': nif, 'mod': '200', 'cod': 'C00500', 'val': 100000},  # Resultado
        {'nif': nif, 'mod': '200', 'cod': 'C00033', 'val': 500000},  # Activo
        {'nif': nif, 'mod': '200', 'cod': 'C00280', 'val': 20000},   # Amortización
        {'nif': nif, 'mod': '200', 'cod': 'C00032', 'val': 80000},   # Caja (Efectivo)
        {'nif': nif, 'mod': '200', 'cod': 'C00296', 'val': 5000},    # Gastos Financieros
        {'nif': nif, 'mod': '200', 'cod': 'C00195', 'val': 100000},  # Deuda
        {'nif': nif, 'mod': '200', 'cod': 'GASTOS_TRANSP', 'val': 25000}, # Transporte OK
        # M349 (Intracomunitario)
        {'nif': nif, 'mod': '349', 'cod': 'ENTREGAS_UE', 'val': 50000}, # Exporta poco
    ])

    # --- ESCENARIO B: EL FANTASMA (GHOST_SL) ---
    # Fraude Carrusel: Declara exportar 2M€ pero NO tiene gastos de transporte ni personal.
    nif = "B_GHOST"
    data.extend([
        # M200
        {'nif': nif, 'mod': '200', 'cod': 'C00255', 'val': 2000000},
        {'nif': nif, 'mod': '200', 'cod': 'C00500', 'val': 10000},   # Beneficio ridículo
        {'nif': nif, 'mod': '200', 'cod': 'C00033', 'val': 50000},   # Activo mínimo
        {'nif': nif, 'mod': '200', 'cod': 'C00280', 'val': 0},
        {'nif': nif, 'mod': '200', 'cod': 'C00032', 'val': 1000},
        {'nif': nif, 'mod': '200', 'cod': 'GASTOS_TRANSP', 'val': 0}, # <--- ALERTA ROJA (Exporta sin transporte)
        # M349
        {'nif': nif, 'mod': '349', 'cod': 'ENTREGAS_UE', 'val': 1800000}, # Dice que exportó 1.8M
    ])

    # --- ESCENARIO C: EL MAQUILLADOR (COOKING_SA) ---
    # Fraude Contable: Beneficio enorme, Caja vacía, Pasivo Oculto.
    nif = "A_COOKING"
    data.extend([
        # M200
        {'nif': nif, 'mod': '200', 'cod': 'C00255', 'val': 5000000},
        {'nif': nif, 'mod': '200', 'cod': 'C00500', 'val': 1200000}, # ¡Gran Beneficio!
        {'nif': nif, 'mod': '200', 'cod': 'C00033', 'val': 4000000},
        {'nif': nif, 'mod': '200', 'cod': 'C00280', 'val': 100000},
        {'nif': nif, 'mod': '200', 'cod': 'C00032', 'val': 5000},    # <--- ALERTA: Caja vacía pese a beneficio
        {'nif': nif, 'mod': '200', 'cod': 'C00296', 'val': 150000},  # Paga muchos intereses
        {'nif': nif, 'mod': '200', 'cod': 'C00195', 'val': 300000},  # Deuda declarada baja (50% interés implícito)
        {'nif': nif, 'mod': '200', 'cod': 'GASTOS_TRANSP', 'val': 100000},
        # M349
        {'nif': nif, 'mod': '349', 'cod': 'ENTREGAS_UE', 'val': 0},
    ])

    return pd.DataFrame(data)

# ==========================================
# 2. TRANSFORMADOR (ETL / PIVOT)
# ==========================================
def transformar_datos(df_raw):
    # Pivotamos: De Vertical a Horizontal
    # Index: NIF | Columns: CODIGO_CASILLA | Values: VALOR
    df_pivot = df_raw.pivot_table(index='nif', columns='cod', values='val', aggfunc='sum').fillna(0)
    return df_pivot

# ==========================================
# 3. MOTOR FORENSE (FEATURE ENGINEERING)
# ==========================================
def calcular_metricas(df):
    results = df.copy()
    
    # Métrica 1: Paradoja Logística (Exportaciones vs Transporte)
    # Si exportas mucho (M349) pero no gastas en transporte, es sospechoso.
    results['logistics_gap'] = results['ENTREGAS_UE'] - (results['GASTOS_TRANSP'] * 100) 
    # (Multiplicamos transp * 100 asumiendo coste logístico del 1%. Si gap es positivo alto = malo)

    # Métrica 2: Accruals Ratio (Calidad del Beneficio)
    # (Beneficio - (Caja + Amortización)) / Activo. 
    # Si es muy alto, el beneficio es "de papel".
    flujo_caja_aprox = results['C00032'] + results['C00280'] # Simplificación para POC
    results['accruals_ratio'] = (results['C00500'] - flujo_caja_aprox) / (results['C00033'] + 1)

    # Métrica 3: Interés Implícito (Deuda Oculta)
    # Gastos Financieros / Deuda Bancaria
    results['implied_interest'] = results['C00296'] / (results['C00195'] + 1)

    return results

# ==========================================
# 4. DETECCIÓN (AI MODEL)
# ==========================================
def ejecutar_modelo(df_features):
    # Seleccionamos solo las métricas calculadas para el modelo
    features = ['logistics_gap', 'accruals_ratio', 'implied_interest']
    X = df_features[features]
    
    # Normalizamos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest (Detecta lo raro)
    model = IsolationForest(contamination='auto', random_state=42) # 'auto' para datasets pequeños
    df_features['anomaly_score'] = model.fit_predict(X_scaled)
    df_features['decision_function'] = model.decision_function(X_scaled)
    
    return df_features

# ==========================================
# EJECUCIÓN
# ==========================================
print("--- 1. Generando Datos Simulados (Verticales) ---")
df_raw = generar_datos_prueba()
print(df_raw.head())

print("\n--- 2. Transformando a Tabla Analítica (Horizontal) ---")
df_wide = transformar_datos(df_raw)

print("\n--- 3. Calculando Métricas Forenses ---")
df_enriched = calcular_metricas(df_wide)

print("\n--- 4. Ejecutando IA (Isolation Forest) ---")
df_final = ejecutar_modelo(df_enriched)

# MOSTRAR RESULTADOS
display_cols = ['logistics_gap', 'accruals_ratio', 'implied_interest', 'anomaly_score', 'decision_function']
print("\n RESULTADOS DEL ANÁLISIS:")
print(df_final[display_cols])

# Interpretación rápida
print("\n--- INTERPRETACIÓN ---")
for nif, row in df_final.iterrows():
    status = "FRAUDE" if row['anomaly_score'] == -1 else "OK"
    motivo = ""
    if row['logistics_gap'] > 500000: motivo += "[Exportacion Falsa] "
    if row['accruals_ratio'] > 0.1: motivo += "[Beneficio Ficticio] "
    if row['implied_interest'] > 0.2: motivo += "[Deuda Oculta] "
    
    print(f"Empresa {nif}: {status} -> {motivo if motivo else 'Sin alertas'}")
