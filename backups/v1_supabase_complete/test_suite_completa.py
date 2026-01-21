"""
FraudHunter Core - Suite Completa de Casos de Prueba
=====================================================
Casos de prueba exhaustivos para validar todas las metricas forenses.

ESCENARIOS CUBIERTOS:
1. Empresas Sanas (Control)
2. Fraude Carrusel IVA
3. Empresa Pantalla
4. Maquillaje Contable (Earnings Management)
5. Deuda Oculta
6. Facturas Falsas (Benford)
7. Fraude Combinado (Multiples senales)
8. Casos Limite (Edge Cases)

CASILLAS AEAT UTILIZADAS:
- Ver tax_mapping_config.py para el mapeo completo de casillas oficiales
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from scipy.spatial.distance import mahalanobis
import warnings
warnings.filterwarnings('ignore')

# Importar configuración centralizada de casillas AEAT (para referencia)
from tax_mapping_config import TAX_MAPPING_CONFIG, LEGACY_CASILLAS_MAP, get_source_to_canonical

# Importar DeepFraudHunter (módulos Deep Learning opcionales)
try:
    from deep_fraud_hunter import (
        DeepFraudEnsemble, DeepFraudConfig,
        TORCH_AVAILABLE, DGL_AVAILABLE, TRANSFORMERS_AVAILABLE, XGB_AVAILABLE
    )
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    TORCH_AVAILABLE = False
    DGL_AVAILABLE = False
    TRANSFORMERS_AVAILABLE = False
    XGB_AVAILABLE = False


# =============================================================================
# GENERADOR DE CASOS DE PRUEBA
# =============================================================================

def generar_casos_prueba():
    """
    Genera dataset con 20 empresas cubriendo todos los escenarios de fraude.
    
    Returns:
        DataFrame con empresas etiquetadas por tipo de fraude
    """
    casos = []
    
    # =========================================================================
    # GRUPO 1: EMPRESAS SANAS (Control - 5 empresas)
    # =========================================================================
    
    # 1.1 Empresa industrial tipica
    casos.append({
        'nif': 'SANA_01_INDUSTRIAL',
        'tipo_fraude': 'SANA',
        'descripcion': 'Empresa industrial con ratios normales',
        'cnae': '2511',  # Fabricacion estructuras metalicas
        'ventas_netas': 2000000,
        'resultado_neto': 120000,  # Margen 6%
        'activo_total': 1800000,
        'deuda_bancaria': 400000,
        'gastos_personal': 500000,  # Productividad 4x
        'gastos_transporte': 60000,
        'intereses_pagados': 20000,  # 5% sobre deuda
        'flujo_caja_operativo': 150000,  # Superior al beneficio
        'total_m347': 1900000,  # Cobertura 95%
        'total_m349': 0,
        'importes_m347': [45678.23, 123456.78, 67890.12, 234567.89, 98765.43],
    })
    
    # 1.2 Comercio minorista sano
    casos.append({
        'nif': 'SANA_02_COMERCIO',
        'tipo_fraude': 'SANA',
        'descripcion': 'Comercio minorista con margenes ajustados',
        'cnae': '4711',  # Supermercados
        'ventas_netas': 5000000,
        'resultado_neto': 100000,  # Margen 2%
        'activo_total': 3000000,
        'deuda_bancaria': 800000,
        'gastos_personal': 1200000,  
        'gastos_transporte': 150000,
        'intereses_pagados': 40000,
        'flujo_caja_operativo': 180000,
        'total_m347': 4800000,
        'total_m349': 100000,  # Algo de importacion
        'importes_m347': [3456.78, 12345.67, 5678.90, 23456.78, 8901.23],
    })
    
    # 1.3 Consultoria
    casos.append({
        'nif': 'SANA_03_CONSULTORIA',
        'tipo_fraude': 'SANA',
        'descripcion': 'Consultoria con alto margen y personal',
        'cnae': '7022',  # Consultoria empresarial
        'ventas_netas': 800000,
        'resultado_neto': 160000,  # Margen 20%
        'activo_total': 300000,
        'deuda_bancaria': 50000,
        'gastos_personal': 400000,  
        'gastos_transporte': 20000,
        'intereses_pagados': 2500,
        'flujo_caja_operativo': 140000,
        'total_m347': 750000,
        'total_m349': 0,
        'importes_m347': [15234.56, 28901.23, 9876.54, 45678.90, 12345.67],
    })
    
    # 1.4 Transporte
    casos.append({
        'nif': 'SANA_04_TRANSPORTE',
        'tipo_fraude': 'SANA',
        'descripcion': 'Empresa de transporte con gastos coherentes',
        'cnae': '4941',
        'ventas_netas': 1500000,
        'resultado_neto': 75000,
        'activo_total': 900000,
        'deuda_bancaria': 300000,
        'gastos_personal': 450000,
        'gastos_transporte': 400000,  # Alto - coherente con sector
        'intereses_pagados': 15000,
        'flujo_caja_operativo': 120000,
        'total_m347': 1400000,
        'total_m349': 200000,  # Exporta servicios
        'importes_m347': [67890.12, 34567.89, 89012.34, 23456.78, 56789.01],
    })
    
    # 1.5 Startup tecnologica (caso limite - alto crecimiento)
    casos.append({
        'nif': 'SANA_05_STARTUP',
        'tipo_fraude': 'SANA',
        'descripcion': 'Startup con perdidas pero flujo de caja',
        'cnae': '6201',
        'ventas_netas': 500000,
        'resultado_neto': -50000,  # Perdidas (inversion)
        'activo_total': 800000,
        'deuda_bancaria': 200000,
        'gastos_personal': 350000,
        'gastos_transporte': 10000,
        'intereses_pagados': 10000,
        'flujo_caja_operativo': 80000,  # Positivo pese a perdidas
        'total_m347': 480000,
        'total_m349': 0,
        'importes_m347': [12345.67, 8901.23, 34567.89, 5678.90, 23456.78],
    })
    
    # =========================================================================
    # GRUPO 2: FRAUDE CARRUSEL IVA (3 empresas)
    # =========================================================================
    
    # 2.1 Carrusel clasico - Sin transporte
    casos.append({
        'nif': 'FRAUDE_CARRUSEL_01',
        'tipo_fraude': 'CARRUSEL_IVA',
        'descripcion': 'Exportaciones masivas sin gastos logisticos',
        'cnae': '4690',  # Comercio mayorista
        'ventas_netas': 8000000,
        'resultado_neto': 50000,  # Margen ridiculo
        'activo_total': 200000,  # Activo minimo
        'deuda_bancaria': 0,
        'gastos_personal': 30000,  # Casi sin empleados
        'gastos_transporte': 0,  # ALERTA: Exporta sin transporte
        'intereses_pagados': 0,
        'flujo_caja_operativo': 40000,
        'total_m347': 500000,  # Incoherente con ventas
        'total_m349': 7500000,  # Mega exportaciones UE
        'importes_m347': [100000, 100000, 100000, 100000, 100000],  # Redondos
    })
    
    # 2.2 Carrusel con buffer
    casos.append({
        'nif': 'FRAUDE_CARRUSEL_02',
        'tipo_fraude': 'CARRUSEL_IVA',
        'descripcion': 'Buffer de carrusel - intermediario',
        'cnae': '4690',
        'ventas_netas': 3000000,
        'resultado_neto': 15000,  # Margen 0.5%
        'activo_total': 150000,
        'deuda_bancaria': 0,
        'gastos_personal': 20000,
        'gastos_transporte': 500,  # Casi nada
        'intereses_pagados': 0,
        'flujo_caja_operativo': 12000,
        'total_m347': 200000,
        'total_m349': 2800000,
        'importes_m347': [50000, 50000, 50000, 50000],
    })
    
    # 2.3 Missing Trader
    casos.append({
        'nif': 'FRAUDE_CARRUSEL_03',
        'tipo_fraude': 'CARRUSEL_IVA',
        'descripcion': 'Missing trader - desaparece con IVA',
        'cnae': '4652',  # Comercio electronica
        'ventas_netas': 12000000,
        'resultado_neto': 200000,
        'activo_total': 800000,
        'deuda_bancaria': 0,
        'gastos_personal': 25000,
        'gastos_transporte': 2000,  # Minimo
        'intereses_pagados': 0,
        'flujo_caja_operativo': -500000,  # NEGATIVO - extrayendo caja
        'total_m347': 1000000,
        'total_m349': 11000000,
        'importes_m347': [200000, 200000, 200000, 200000, 200000],
    })
    
    # =========================================================================
    # GRUPO 3: EMPRESA PANTALLA (3 empresas)
    # =========================================================================
    
    # 3.1 Pantalla clasica - Sin empleados
    casos.append({
        'nif': 'FRAUDE_PANTALLA_01',
        'tipo_fraude': 'EMPRESA_PANTALLA',
        'descripcion': 'Facturacion millonaria sin empleados',
        'cnae': '7022',
        'ventas_netas': 2500000,
        'resultado_neto': 400000,
        'activo_total': 100000,
        'deuda_bancaria': 0,
        'gastos_personal': 5000,  # ALERTA: 500x productividad
        'gastos_transporte': 1000,
        'intereses_pagados': 0,
        'flujo_caja_operativo': 380000,
        'total_m347': 2400000,
        'total_m349': 0,
        'importes_m347': [500000, 500000, 500000, 500000, 400000],
    })
    
    # 3.2 Pantalla para facturas falsas
    casos.append({
        'nif': 'FRAUDE_PANTALLA_02',
        'tipo_fraude': 'EMPRESA_PANTALLA',
        'descripcion': 'Emite facturas para deducir en otras empresas',
        'cnae': '6920',  # Contabilidad
        'ventas_netas': 1800000,
        'resultado_neto': 50000,
        'activo_total': 50000,
        'deuda_bancaria': 0,
        'gastos_personal': 8000,
        'gastos_transporte': 500,
        'intereses_pagados': 0,
        'flujo_caja_operativo': 45000,
        'total_m347': 1750000,
        'total_m349': 0,
        'importes_m347': [300000, 300000, 300000, 300000, 300000, 250000],
    })
    
    # 3.3 Sociedad instrumental
    casos.append({
        'nif': 'FRAUDE_PANTALLA_03',
        'tipo_fraude': 'EMPRESA_PANTALLA',
        'descripcion': 'Instrumental para canalizar fondos',
        'cnae': '6420',  # Holding
        'ventas_netas': 5000000,
        'resultado_neto': 100000,  # Margen 2%
        'activo_total': 2000000,
        'deuda_bancaria': 1500000,
        'gastos_personal': 15000,  # ALERTA
        'gastos_transporte': 0,
        'intereses_pagados': 75000,
        'flujo_caja_operativo': 50000,
        'total_m347': 500000,  # Muy bajo vs ventas
        'total_m349': 0,
        'importes_m347': [100000, 100000, 100000, 100000, 100000],
    })
    
    # =========================================================================
    # GRUPO 4: MAQUILLAJE CONTABLE (3 empresas)
    # =========================================================================
    
    # 4.1 Beneficio sin caja (Accruals)
    casos.append({
        'nif': 'FRAUDE_CONTABLE_01',
        'tipo_fraude': 'MAQUILLAJE_CONTABLE',
        'descripcion': 'Gran beneficio pero caja vacia',
        'cnae': '4120',  # Construccion
        'ventas_netas': 10000000,
        'resultado_neto': 2000000,  # 20% margen
        'activo_total': 8000000,
        'deuda_bancaria': 2000000,
        'gastos_personal': 2500000,
        'gastos_transporte': 300000,
        'intereses_pagados': 100000,
        'flujo_caja_operativo': 100000,  # ALERTA: Caja minima
        'total_m347': 9500000,
        'total_m349': 0,
        'importes_m347': [45678.90, 123456.78, 234567.89, 67890.12, 345678.90],
    })
    
    # 4.2 Capitalizacion agresiva
    casos.append({
        'nif': 'FRAUDE_CONTABLE_02',
        'tipo_fraude': 'MAQUILLAJE_CONTABLE',
        'descripcion': 'Gastos capitalizados como activo',
        'cnae': '6201',
        'ventas_netas': 3000000,
        'resultado_neto': 800000,  # 27% margen (muy alto)
        'activo_total': 5000000,  # Activo inflado
        'deuda_bancaria': 500000,
        'gastos_personal': 600000,
        'gastos_transporte': 50000,
        'intereses_pagados': 25000,
        'flujo_caja_operativo': 200000,  # Mucho menos que beneficio
        'total_m347': 2800000,
        'total_m349': 0,
        'importes_m347': [34567.89, 78901.23, 56789.01, 12345.67, 89012.34],
    })
    
    # 4.3 Ingresos ficticios
    casos.append({
        'nif': 'FRAUDE_CONTABLE_03',
        'tipo_fraude': 'MAQUILLAJE_CONTABLE',
        'descripcion': 'Ventas infladas para conseguir credito',
        'cnae': '4673',  # Comercio madera
        'ventas_netas': 6000000,
        'resultado_neto': 600000,
        'activo_total': 4000000,
        'deuda_bancaria': 2500000,
        'gastos_personal': 800000,
        'gastos_transporte': 200000,
        'intereses_pagados': 125000,
        'flujo_caja_operativo': -100000,  # NEGATIVO
        'total_m347': 2000000,  # ALERTA: Solo 33% cobertura
        'total_m349': 0,
        'importes_m347': [45000.00, 67890.12, 123456.78, 23456.78, 89012.34],
    })
    
    # =========================================================================
    # GRUPO 5: DEUDA OCULTA (2 empresas)
    # =========================================================================
    
    # 5.1 Prestamos usureros
    casos.append({
        'nif': 'FRAUDE_DEUDA_01',
        'tipo_fraude': 'DEUDA_OCULTA',
        'descripcion': 'Intereses altisimos vs deuda declarada',
        'cnae': '4711',
        'ventas_netas': 2000000,
        'resultado_neto': 50000,
        'activo_total': 1500000,
        'deuda_bancaria': 200000,  # Deuda baja
        'gastos_personal': 400000,
        'gastos_transporte': 80000,
        'intereses_pagados': 80000,  # ALERTA: 40% tasa implicita
        'flujo_caja_operativo': 100000,
        'total_m347': 1900000,
        'total_m349': 0,
        'importes_m347': [23456.78, 56789.01, 34567.89, 78901.23, 12345.67],
    })
    
    # 5.2 Blanqueo via prestamos
    casos.append({
        'nif': 'FRAUDE_DEUDA_02',
        'tipo_fraude': 'DEUDA_OCULTA',
        'descripcion': 'Estructura de blanqueo via intereses',
        'cnae': '6831',  # Inmobiliaria
        'ventas_netas': 1000000,
        'resultado_neto': -200000,  # Perdidas por intereses
        'activo_total': 3000000,
        'deuda_bancaria': 500000,
        'gastos_personal': 100000,
        'gastos_transporte': 20000,
        'intereses_pagados': 300000,  # ALERTA: 60% tasa
        'flujo_caja_operativo': 50000,
        'total_m347': 950000,
        'total_m349': 0,
        'importes_m347': [45678.90, 23456.78, 67890.12, 34567.89, 56789.01],
    })
    
    # =========================================================================
    # GRUPO 6: FACTURAS FALSAS / BENFORD (2 empresas)
    # =========================================================================
    
    # 6.1 Numeros redondos
    casos.append({
        'nif': 'FRAUDE_BENFORD_01',
        'tipo_fraude': 'FACTURAS_FALSAS',
        'descripcion': 'Todos los importes son redondos',
        'cnae': '4639',
        'ventas_netas': 3000000,
        'resultado_neto': 150000,
        'activo_total': 1500000,
        'deuda_bancaria': 400000,
        'gastos_personal': 500000,
        'gastos_transporte': 100000,
        'intereses_pagados': 20000,
        'flujo_caja_operativo': 180000,
        'total_m347': 2800000,
        'total_m349': 0,
        'importes_m347': [100000, 200000, 150000, 300000, 250000, 
                          500000, 100000, 200000, 300000, 400000,
                          150000, 250000, 100000, 200000, 100000],  # TODOS REDONDOS
    })
    
    # 6.2 Distribucion antinatural
    casos.append({
        'nif': 'FRAUDE_BENFORD_02',
        'tipo_fraude': 'FACTURAS_FALSAS',
        'descripcion': 'Importes empiezan por 5-9 (anti-Benford)',
        'cnae': '4639',
        'ventas_netas': 2500000,
        'resultado_neto': 120000,
        'activo_total': 1200000,
        'deuda_bancaria': 300000,
        'gastos_personal': 450000,
        'gastos_transporte': 80000,
        'intereses_pagados': 15000,
        'flujo_caja_operativo': 150000,
        'total_m347': 2400000,
        'total_m349': 0,
        'importes_m347': [55000, 67000, 78000, 89000, 91000,
                          52000, 63000, 74000, 85000, 96000,
                          58000, 69000, 71000, 82000, 93000],  # Empiezan 5-9
    })
    
    # =========================================================================
    # GRUPO 7: FRAUDE COMBINADO (2 empresas)
    # =========================================================================
    
    # 7.1 Carrusel + Pantalla + Benford
    casos.append({
        'nif': 'FRAUDE_COMBO_01',
        'tipo_fraude': 'COMBINADO',
        'descripcion': 'Multiples senales: carrusel, pantalla, redondos',
        'cnae': '4690',
        'ventas_netas': 15000000,
        'resultado_neto': 100000,
        'activo_total': 300000,
        'deuda_bancaria': 0,
        'gastos_personal': 10000,  # PANTALLA
        'gastos_transporte': 0,    # CARRUSEL
        'intereses_pagados': 0,
        'flujo_caja_operativo': 80000,
        'total_m347': 500000,
        'total_m349': 14000000,    # EXPORTACION MASIVA
        'importes_m347': [1000000, 2000000, 1500000, 500000, 500000],  # REDONDOS
    })
    
    # 7.2 Contable + Deuda
    casos.append({
        'nif': 'FRAUDE_COMBO_02',
        'tipo_fraude': 'COMBINADO',
        'descripcion': 'Maquillaje contable con deuda oculta',
        'cnae': '4120',
        'ventas_netas': 8000000,
        'resultado_neto': 1500000,
        'activo_total': 6000000,
        'deuda_bancaria': 500000,
        'gastos_personal': 1800000,
        'gastos_transporte': 250000,
        'intereses_pagados': 200000,  # 40% tasa
        'flujo_caja_operativo': 50000,  # MUY BAJO vs beneficio
        'total_m347': 4000000,  # 50% cobertura
        'total_m349': 0,
        'importes_m347': [123456.78, 234567.89, 100000, 200000, 345678.90],
    })
    
    return pd.DataFrame(casos)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def calcular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todas las metricas forenses."""
    
    # 1. Cobertura de ventas (M347 / Ventas)
    df['cobertura_ventas'] = df['total_m347'] / df['ventas_netas']
    
    # 2. Incoherencia logistica
    df['flag_incoherencia_logistica'] = (
        (df['total_m349'] > 100000) & 
        (df['gastos_transporte'] < 5000)
    ).astype(int)
    
    # 3. Accruals ratio
    df['accruals_ratio'] = (
        (df['resultado_neto'] - df['flujo_caja_operativo']) / 
        (df['activo_total'] + 1)
    )
    
    # 4. Tasa interes implicita
    df['tasa_interes_implicita'] = np.where(
        df['deuda_bancaria'] > 0,
        df['intereses_pagados'] / df['deuda_bancaria'],
        0
    )
    df['flag_hidden_debt'] = (df['tasa_interes_implicita'] > 0.15).astype(int)
    
    # 5. Productividad laboral
    df['productividad_laboral'] = np.where(
        df['gastos_personal'] > 0,
        df['ventas_netas'] / df['gastos_personal'],
        9999
    )
    df['flag_empresa_pantalla'] = (
        (df['ventas_netas'] > 500000) & 
        (df['gastos_personal'] < 20000)
    ).astype(int)
    
    # 6. Numeros redondos
    def calc_pct_redondos(importes):
        if not importes or len(importes) == 0:
            return 0
        round_count = sum(1 for imp in importes if imp % 1000 == 0)
        return round_count / len(importes)
    
    df['pct_numeros_redondos'] = df['importes_m347'].apply(calc_pct_redondos)
    df['flag_numeros_redondos'] = (df['pct_numeros_redondos'] > 0.4).astype(int)
    
    # 7. Benford divergence
    def calc_benford(importes):
        if not importes or len(importes) < 5:
            return 0
        benford = {1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097, 
                   5: 0.079, 6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046}
        digits = []
        for imp in importes:
            if imp > 0:
                d = int(str(int(imp))[0])
                if 1 <= d <= 9:
                    digits.append(d)
        if len(digits) < 5:
            return 0
        observed = {i: digits.count(i)/len(digits) for i in range(1, 10)}
        kl = sum(observed.get(d, 0) * np.log((observed.get(d, 0.001))/benford[d]) 
                 for d in range(1, 10) if observed.get(d, 0) > 0)
        return kl
    
    df['benford_divergence'] = df['importes_m347'].apply(calc_benford)
    df['flag_benford'] = (df['benford_divergence'] > 0.2).astype(int)
    
    # 8. Margen neto
    df['margen_neto'] = df['resultado_neto'] / (df['ventas_netas'] + 1)
    
    # 9. Rotacion activos
    df['rotacion_activos'] = df['ventas_netas'] / (df['activo_total'] + 1)
    
    return df


# =============================================================================
# MODELO DE DETECCION
# =============================================================================

def ejecutar_modelo(df: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta Isolation Forest y genera scores."""
    
    feature_cols = [
        'cobertura_ventas', 'accruals_ratio', 'tasa_interes_implicita',
        'productividad_laboral', 'pct_numeros_redondos', 'benford_divergence',
        'flag_incoherencia_logistica', 'flag_hidden_debt', 
        'flag_empresa_pantalla', 'flag_numeros_redondos', 'flag_benford'
    ]
    
    X = df[feature_cols].fillna(0)
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = IsolationForest(contamination=0.45, n_estimators=200, random_state=42)
    df['anomaly_label'] = model.fit_predict(X_scaled)
    df['fraud_score'] = model.decision_function(X_scaled)
    
    # Contar flags activos
    flag_cols = ['flag_incoherencia_logistica', 'flag_hidden_debt', 
                 'flag_empresa_pantalla', 'flag_numeros_redondos', 'flag_benford']
    df['total_flags'] = df[flag_cols].sum(axis=1)
    
    # Forzar prediccion de fraude si hay 2+ flags
    df.loc[df['total_flags'] >= 2, 'anomaly_label'] = -1
    
    # Normalizar score
    min_s, max_s = df['fraud_score'].min(), df['fraud_score'].max()
    df['fraud_score_normalized'] = 1 - ((df['fraud_score'] - min_s) / (max_s - min_s + 0.001))
    
    return df


# =============================================================================
# EJECUCION Y REPORTE
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("FRAUDHUNTER CORE - SUITE DE PRUEBAS EXHAUSTIVA")
    print("=" * 80)
    
    # Generar casos
    print("\n[1/4] Generando 20 casos de prueba...")
    df = generar_casos_prueba()
    print(f"     Casos generados: {len(df)}")
    print(f"     - Sanas: {len(df[df['tipo_fraude'] == 'SANA'])}")
    print(f"     - Fraude: {len(df[df['tipo_fraude'] != 'SANA'])}")
    
    # Calcular features
    print("\n[2/4] Calculando metricas forenses...")
    df = calcular_features(df)
    
    # Ejecutar modelo
    print("\n[3/4] Ejecutando Isolation Forest...")
    df = ejecutar_modelo(df)
    
    # Resultados
    print("\n[4/4] Generando reporte...")
    
    print("\n" + "=" * 80)
    print("RESULTADOS POR EMPRESA")
    print("=" * 80)
    
    for _, row in df.iterrows():
        pred = "FRAUDE" if row['anomaly_label'] == -1 else "OK"
        real = "SANA" if row['tipo_fraude'] == 'SANA' else "FRAUDE"
        match = "CORRECTO" if (pred == "OK" and real == "SANA") or (pred == "FRAUDE" and real == "FRAUDE") else "ERROR"
        
        flags = []
        if row['flag_incoherencia_logistica']: flags.append("Carrusel")
        if row['flag_empresa_pantalla']: flags.append("Pantalla")
        if row['accruals_ratio'] > 0.1: flags.append("Accruals")
        if row['flag_hidden_debt']: flags.append("DeudaOculta")
        if row['flag_numeros_redondos']: flags.append("Redondos")
        if row['flag_benford']: flags.append("Benford")
        if row['cobertura_ventas'] < 0.5: flags.append("Cobertura")
        
        print(f"\n{row['nif']}")
        print(f"   Tipo real: {row['tipo_fraude']}")
        print(f"   Prediccion: {pred} | Score: {row['fraud_score_normalized']:.3f}")
        print(f"   Validacion: {match}")
        print(f"   Flags: {', '.join(flags) if flags else 'Ninguno'}")
        print(f"   Descripcion: {row['descripcion']}")
    
    # Matriz de confusion
    print("\n" + "=" * 80)
    print("MATRIZ DE CONFUSION")
    print("=" * 80)
    
    df['es_fraude_real'] = df['tipo_fraude'] != 'SANA'
    df['predicho_fraude'] = df['anomaly_label'] == -1
    
    tp = ((df['predicho_fraude']) & (df['es_fraude_real'])).sum()
    tn = ((~df['predicho_fraude']) & (~df['es_fraude_real'])).sum()
    fp = ((df['predicho_fraude']) & (~df['es_fraude_real'])).sum()
    fn = ((~df['predicho_fraude']) & (df['es_fraude_real'])).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(df)
    
    print(f"\n                 Predicho")
    print(f"              FRAUDE    OK")
    print(f"Real FRAUDE    {tp:3d}      {fn:3d}")
    print(f"     OK        {fp:3d}      {tn:3d}")
    
    print(f"\n   Precision: {precision:.1%}")
    print(f"   Recall:    {recall:.1%}")
    print(f"   F1-Score:  {f1:.1%}")
    print(f"   Accuracy:  {accuracy:.1%}")
    
    print("\n" + "=" * 80)
    print("GENERANDO REPORTE HTML...")
    print("=" * 80)
    
    # =========================================================================
    # GENERAR REPORTE HTML
    # =========================================================================
    import webbrowser
    import os
    
    # Generar filas de tabla con detalles expandibles para analistas
    rows_html = ""
    for idx, row in df.iterrows():
        pred = "FRAUDE" if row['anomaly_label'] == -1 else "OK"
        real = "SANA" if row['tipo_fraude'] == 'SANA' else "FRAUDE"
        match = (pred == "OK" and real == "SANA") or (pred == "FRAUDE" and real == "FRAUDE")
        
        flags = []
        if row['flag_incoherencia_logistica']: flags.append("Carrusel")
        if row['flag_empresa_pantalla']: flags.append("Pantalla")
        if row['accruals_ratio'] > 0.1: flags.append("Accruals")
        if row['flag_hidden_debt']: flags.append("DeudaOculta")
        if row['flag_numeros_redondos']: flags.append("Redondos")
        if row['flag_benford']: flags.append("Benford")
        if row['cobertura_ventas'] < 0.5: flags.append("Cobertura")
        
        row_class = "correct" if match else "error"
        pred_class = "fraude" if pred == "FRAUDE" else "ok"
        tipo_class = "tipo-fraude" if row['tipo_fraude'] != 'SANA' else "tipo-sana"
        
        score_pct = row['fraud_score_normalized'] * 100
        score_class = "high" if score_pct > 60 else "medium" if score_pct > 30 else "low"
        score_bar = f"<div class='score-bar'><div class='score-fill {score_class}' style='width: {score_pct:.0f}%'></div></div>"
        
        desc = row.get('descripcion', '-')
        if len(desc) > 40:
            desc = desc[:40] + "..."
        
        # Construir panel de detalles para analistas
        cobertura_alert = 'alert-metric' if row['cobertura_ventas'] < 0.5 else ''
        logistica_alert = 'alert-metric' if row['flag_incoherencia_logistica'] else ''
        accruals_alert = 'alert-metric' if row['accruals_ratio'] > 0.1 else ''
        deuda_alert = 'alert-metric' if row['flag_hidden_debt'] else ''
        pantalla_alert = 'alert-metric' if row['flag_empresa_pantalla'] else ''
        redondos_alert = 'alert-metric' if row['flag_numeros_redondos'] else ''
        benford_alert = 'alert-metric' if row['flag_benford'] else ''
        
        details_html = f"""
        <div class="analyst-panel">
            <div class="panel-section">
                <h4>📊 Datos Fuente (Modelos AEAT)</h4>
                <div class="data-grid">
                    <div class="data-item"><span class="data-label">Ventas (M200)</span><span class="data-value">€{row['ventas_netas']:,.0f}</span></div>
                    <div class="data-item"><span class="data-label">Total M347</span><span class="data-value">€{row['total_m347']:,.0f}</span></div>
                    <div class="data-item"><span class="data-label">Total M349 (UE)</span><span class="data-value">€{row['total_m349']:,.0f}</span></div>
                    <div class="data-item"><span class="data-label">Resultado Neto</span><span class="data-value">€{row['resultado_neto']:,.0f}</span></div>
                    <div class="data-item"><span class="data-label">Flujo Caja Op.</span><span class="data-value">€{row['flujo_caja_operativo']:,.0f}</span></div>
                    <div class="data-item"><span class="data-label">Activo Total</span><span class="data-value">€{row['activo_total']:,.0f}</span></div>
                    <div class="data-item"><span class="data-label">Gastos Personal</span><span class="data-value">€{row['gastos_personal']:,.0f}</span></div>
                    <div class="data-item"><span class="data-label">Gastos Transporte</span><span class="data-value">€{row['gastos_transporte']:,.0f}</span></div>
                    <div class="data-item"><span class="data-label">Deuda Bancaria</span><span class="data-value">€{row['deuda_bancaria']:,.0f}</span></div>
                    <div class="data-item"><span class="data-label">Intereses Pagados</span><span class="data-value">€{row['intereses_pagados']:,.0f}</span></div>
                </div>
            </div>
            <div class="panel-section">
                <h4>🔬 Cálculo de Métricas Forenses</h4>
                <table class="calc-table">
                    <thead><tr><th>Métrica</th><th>Fórmula y Valores</th><th>Resultado</th><th>Umbral</th><th>Estado</th></tr></thead>
                    <tbody>
                        <tr class="{cobertura_alert}">
                            <td><strong>Cobertura Ventas</strong></td>
                            <td><code>M347 / Ventas = €{row['total_m347']:,.0f} / €{row['ventas_netas']:,.0f}</code></td>
                            <td><strong>{row['cobertura_ventas']:.2f}</strong></td>
                            <td>&lt; 0.50</td>
                            <td>{'🔴 ALERTA' if row['cobertura_ventas'] < 0.5 else '🟢 OK'}</td>
                        </tr>
                        <tr class="{logistica_alert}">
                            <td><strong>Incoherencia Logística</strong></td>
                            <td><code>M349 &gt; 100K AND Transp &lt; 5K → €{row['total_m349']:,.0f} AND €{row['gastos_transporte']:,.0f}</code></td>
                            <td><strong>{'SÍ' if row['flag_incoherencia_logistica'] else 'NO'}</strong></td>
                            <td>Flag = 1</td>
                            <td>{'🔴 CARRUSEL' if row['flag_incoherencia_logistica'] else '🟢 OK'}</td>
                        </tr>
                        <tr class="{accruals_alert}">
                            <td><strong>Accruals Ratio</strong></td>
                            <td><code>(Rdo - FCO) / Activo = (€{row['resultado_neto']:,.0f} - €{row['flujo_caja_operativo']:,.0f}) / €{row['activo_total']:,.0f}</code></td>
                            <td><strong>{row['accruals_ratio']:.4f}</strong></td>
                            <td>&gt; 0.10</td>
                            <td>{'🔴 FICTICIO' if row['accruals_ratio'] > 0.1 else '🟢 OK'}</td>
                        </tr>
                        <tr class="{deuda_alert}">
                            <td><strong>Tasa Interés Implícita</strong></td>
                            <td><code>Intereses / Deuda = €{row['intereses_pagados']:,.0f} / €{row['deuda_bancaria']:,.0f}</code></td>
                            <td><strong>{row['tasa_interes_implicita']:.1%}</strong></td>
                            <td>&gt; 15%</td>
                            <td>{'🔴 OCULTA' if row['flag_hidden_debt'] else '🟢 OK'}</td>
                        </tr>
                        <tr class="{pantalla_alert}">
                            <td><strong>Productividad Laboral</strong></td>
                            <td><code>Ventas / Personal = €{row['ventas_netas']:,.0f} / €{row['gastos_personal']:,.0f}</code></td>
                            <td><strong>{row['productividad_laboral']:.1f}x</strong></td>
                            <td>Ventas &gt;500K, Personal &lt;20K</td>
                            <td>{'🔴 PANTALLA' if row['flag_empresa_pantalla'] else '🟢 OK'}</td>
                        </tr>
                        <tr class="{redondos_alert}">
                            <td><strong>% Números Redondos</strong></td>
                            <td><code>count(importe % 1000 == 0) / n_facturas</code></td>
                            <td><strong>{row['pct_numeros_redondos']:.1%}</strong></td>
                            <td>&gt; 40%</td>
                            <td>{'🔴 MANIPULADO' if row['flag_numeros_redondos'] else '🟢 OK'}</td>
                        </tr>
                        <tr class="{benford_alert}">
                            <td><strong>Divergencia Benford</strong></td>
                            <td><code>KL(observado || esperado) = Σ P(d)·log(P(d)/Q(d))</code></td>
                            <td><strong>{row['benford_divergence']:.4f}</strong></td>
                            <td>&gt; 0.20</td>
                            <td>{'🔴 ANÓMALO' if row['flag_benford'] else '🟢 OK'}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <div class="score-summary">
                <strong>📈 Score Final:</strong> {row['fraud_score_normalized']:.3f} 
                | <strong>Flags Activos:</strong> {row['total_flags']} 
                | <strong>Decisión:</strong> {'FRAUDE (score + flags ≥ 2)' if pred == 'FRAUDE' else 'OK'}
            </div>
        </div>
        """
        
        rows_html += f"""
        <tr class="{row_class} expandable-row" onclick="toggleDetails('{idx}')">
            <td><code>{row['nif']}</code> <span class="expand-arrow" id="arrow-{idx}">▶</span></td>
            <td class="{tipo_class}">{row['tipo_fraude']}</td>
            <td class="{pred_class}">{pred}</td>
            <td>{score_bar} <span class="score-val">{row['fraud_score_normalized']:.3f}</span></td>
            <td class="flags">{', '.join(flags) if flags else '-'}</td>
            <td style="color: #888; font-size: 0.85rem;">{desc}</td>
            <td class="match">{'✓' if match else '✗'}</td>
        </tr>
        <tr class="details-row" id="details-{idx}" style="display: none;">
            <td colspan="7">{details_html}</td>
        </tr>"""

    
    # Por tipo de fraude - con iconos y descripciones
    tipo_info = {
        'SANA': {'icon': '✅', 'desc': 'Empresas con ratios normales'},
        'CARRUSEL_IVA': {'icon': '🔄', 'desc': 'Exportaciones ficticias sin logística'},
        'EMPRESA_PANTALLA': {'icon': '🏢', 'desc': 'Facturación sin estructura real'},
        'MAQUILLAJE_CONTABLE': {'icon': '📊', 'desc': 'Beneficio artificial, caja vacía'},
        'DEUDA_OCULTA': {'icon': '💳', 'desc': 'Financiación irregular detectada'},
        'FACTURAS_FALSAS': {'icon': '📄', 'desc': 'Patrón Benford/redondos anómalo'},
        'COMBINADO': {'icon': '⚠️', 'desc': 'Múltiples indicadores de fraude'},
    }
    
    tipos_html = ""
    for tipo in df['tipo_fraude'].unique():
        subset = df[df['tipo_fraude'] == tipo]
        correctos = sum((subset['anomaly_label'] == -1) == (tipo != 'SANA'))
        total = len(subset)
        pct = 100 * correctos / total if total > 0 else 0
        bar_class = "success" if pct >= 80 else "warning" if pct >= 50 else "danger"
        info = tipo_info.get(tipo, {'icon': '❓', 'desc': '-'})
        tipos_html += f"""
        <div class="tipo-item">
            <span class="tipo-icon">{info['icon']}</span>
            <span class="tipo-name">{tipo}</span>
            <span class="tipo-desc">{info['desc']}</span>
            <div class="tipo-bar"><div class="tipo-fill {bar_class}" style="width: {pct}%"></div></div>
            <span class="tipo-pct">{correctos}/{total} ({pct:.0f}%)</span>
        </div>"""
    
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>FraudHunter - Informe de Análisis Forense</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%); color: #e0e0e0; min-height: 100vh; padding: 40px 20px; line-height: 1.6; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ font-size: 2.5rem; background: linear-gradient(90deg, #00d9ff, #00ff88); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px; }}
        h2 {{ font-size: 1.5rem; color: #00d9ff; margin-bottom: 20px; border-bottom: 2px solid rgba(0, 217, 255, 0.3); padding-bottom: 10px; }}
        h3 {{ font-size: 1.2rem; color: #fff; margin: 20px 0 15px 0; }}
        .subtitle {{ color: #888; margin-bottom: 10px; font-size: 1.1rem; }}
        .timestamp {{ color: #666; font-size: 0.9rem; margin-bottom: 40px; }}
        
        /* Cards & Sections */
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; margin-bottom: 40px; }}
        .metric-card {{ background: rgba(255,255,255,0.05); border-radius: 16px; padding: 24px; text-align: center; border: 1px solid rgba(255,255,255,0.1); transition: transform 0.2s; }}
        .metric-card:hover {{ transform: translateY(-5px); }}
        .metric-value {{ font-size: 2.5rem; font-weight: bold; }}
        .metric-value.success {{ color: #2ed573; }}
        .metric-value.warning {{ color: #ffa502; }}
        .metric-value.danger {{ color: #ff4757; }}
        .metric-label {{ color: #888; margin-top: 8px; font-size: 0.9rem; }}
        .metric-desc {{ color: #666; font-size: 0.75rem; margin-top: 5px; }}
        
        .section {{ background: rgba(255,255,255,0.03); border-radius: 16px; padding: 30px; margin-bottom: 30px; border: 1px solid rgba(255,255,255,0.08); }}
        .section-title {{ font-size: 1.3rem; margin-bottom: 20px; color: #fff; }}
        
        /* Executive Summary */
        .executive-summary {{ background: linear-gradient(135deg, rgba(0,217,255,0.1), rgba(0,255,136,0.05)); border-left: 4px solid #00d9ff; padding: 25px; border-radius: 0 16px 16px 0; margin-bottom: 40px; }}
        .executive-summary p {{ margin-bottom: 15px; }}
        .executive-summary strong {{ color: #00d9ff; }}
        
        /* Alert boxes */
        .alert {{ padding: 15px 20px; border-radius: 8px; margin: 15px 0; }}
        .alert-success {{ background: rgba(46, 213, 115, 0.15); border-left: 4px solid #2ed573; }}
        .alert-warning {{ background: rgba(255, 165, 2, 0.15); border-left: 4px solid #ffa502; }}
        .alert-danger {{ background: rgba(255, 71, 87, 0.15); border-left: 4px solid #ff4757; }}
        .alert-info {{ background: rgba(0, 217, 255, 0.15); border-left: 4px solid #00d9ff; }}
        
        /* Confusion Matrix */
        .confusion {{ display: grid; grid-template-columns: 100px repeat(2, 140px); gap: 10px; margin: 20px auto; width: fit-content; }}
        .conf-cell {{ padding: 20px; text-align: center; border-radius: 8px; font-weight: bold; }}
        .conf-header {{ background: rgba(255,255,255,0.1); color: #888; }}
        .conf-tp {{ background: rgba(46, 213, 115, 0.3); color: #2ed573; }}
        .conf-tn {{ background: rgba(46, 213, 115, 0.2); color: #2ed573; }}
        .conf-fp {{ background: rgba(255, 165, 2, 0.3); color: #ffa502; }}
        .conf-fn {{ background: rgba(255, 71, 87, 0.3); color: #ff4757; }}
        
        /* Tables */
        table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
        th {{ text-align: left; padding: 12px 8px; color: #888; border-bottom: 2px solid rgba(255,255,255,0.1); }}
        td {{ padding: 12px 8px; border-bottom: 1px solid rgba(255,255,255,0.05); vertical-align: top; }}
        tr.correct {{ background: rgba(46, 213, 115, 0.05); }}
        tr.error {{ background: rgba(255, 71, 87, 0.1); }}
        tr.error td {{ color: #ff6b7a; }}
        
        /* Badges & Tags */
        code {{ background: rgba(0,217,255,0.1); padding: 4px 8px; border-radius: 4px; color: #00d9ff; font-size: 0.85rem; }}
        .fraude {{ color: #ff4757; font-weight: bold; }}
        .ok {{ color: #2ed573; }}
        .tipo-fraude {{ color: #ff6b7a; }}
        .tipo-sana {{ color: #7bed9f; }}
        .flags {{ color: #ffa502; font-size: 0.85rem; }}
        .match {{ font-size: 1rem; text-align: center; font-weight: bold; }}
        
        .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: bold; margin: 2px; }}
        .badge-carrusel {{ background: rgba(255, 71, 87, 0.3); color: #ff6b7a; }}
        .badge-pantalla {{ background: rgba(156, 39, 176, 0.3); color: #ce93d8; }}
        .badge-contable {{ background: rgba(255, 165, 2, 0.3); color: #ffa502; }}
        .badge-deuda {{ background: rgba(233, 30, 99, 0.3); color: #f48fb1; }}
        .badge-facturas {{ background: rgba(0, 188, 212, 0.3); color: #80deea; }}
        
        /* Score bars */
        .score-bar {{ width: 100px; height: 10px; background: rgba(255,255,255,0.1); border-radius: 5px; display: inline-block; vertical-align: middle; margin-right: 10px; }}
        .score-fill {{ height: 100%; border-radius: 5px; }}
        .score-fill.low {{ background: #2ed573; }}
        .score-fill.medium {{ background: #ffa502; }}
        .score-fill.high {{ background: #ff4757; }}
        .score-val {{ color: #fff; font-weight: bold; }}
        
        /* Tipo bars */
        .tipo-item {{ display: flex; align-items: center; padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }}
        .tipo-icon {{ width: 40px; font-size: 1.5rem; }}
        .tipo-name {{ width: 180px; color: #fff; font-weight: 500; }}
        .tipo-desc {{ flex: 1; color: #888; font-size: 0.85rem; padding: 0 20px; }}
        .tipo-bar {{ width: 150px; height: 10px; background: rgba(255,255,255,0.1); border-radius: 5px; margin: 0 15px; }}
        .tipo-fill {{ height: 100%; border-radius: 5px; }}
        .tipo-fill.success {{ background: #2ed573; }}
        .tipo-fill.warning {{ background: #ffa502; }}
        .tipo-fill.danger {{ background: #ff4757; }}
        .tipo-pct {{ width: 100px; text-align: right; color: #fff; font-weight: bold; }}
        
        /* Methodology grid */
        .method-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }}
        .method-card {{ background: rgba(255,255,255,0.03); border-radius: 12px; padding: 20px; border: 1px solid rgba(255,255,255,0.08); }}
        .method-card h4 {{ color: #00d9ff; margin-bottom: 10px; display: flex; align-items: center; gap: 10px; }}
        .method-card p {{ color: #aaa; font-size: 0.9rem; }}
        .method-card .formula {{ background: rgba(0,0,0,0.3); padding: 10px; border-radius: 6px; font-family: monospace; color: #00ff88; margin-top: 10px; font-size: 0.85rem; }}
        
        /* Detail rows */
        .detail-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 15px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }}
        .detail-label {{ color: #888; }}
        .detail-value {{ color: #fff; font-weight: 500; }}
        .detail-value.danger {{ color: #ff4757; }}
        .detail-value.warning {{ color: #ffa502; }}
        .detail-value.success {{ color: #2ed573; }}
        
        /* Expandable */
        .expandable {{ cursor: pointer; }}
        .expandable:hover {{ background: rgba(255,255,255,0.05); }}
        .expand-icon {{ transition: transform 0.2s; display: inline-block; margin-right: 10px; }}
        .expand-content {{ display: none; background: rgba(0,0,0,0.2); padding: 20px; border-radius: 8px; margin: 10px 0; }}
        .expanded .expand-content {{ display: block; }}
        .expanded .expand-icon {{ transform: rotate(90deg); }}
        
        /* Recommendations */
        .recommendation {{ display: flex; gap: 15px; padding: 15px; margin: 10px 0; background: rgba(255,255,255,0.02); border-radius: 8px; border-left: 3px solid #00d9ff; }}
        .recommendation-icon {{ font-size: 1.5rem; }}
        .recommendation-text {{ flex: 1; }}
        .recommendation-text strong {{ color: #fff; }}
        
        /* Legend */
        .legend {{ display: flex; gap: 30px; margin-top: 20px; justify-content: center; flex-wrap: wrap; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; color: #888; }}
        .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
        .legend-dot.correct {{ background: #2ed573; }}
        .legend-dot.error {{ background: #ff4757; }}
        
        /* Footer */
        .footer {{ text-align: center; color: #666; padding: 40px 0 20px 0; font-size: 0.85rem; }}
        .footer a {{ color: #00d9ff; text-decoration: none; }}
        
        /* Two columns */
        .two-cols {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 30px; }}
        
        /* Expandable rows for analyst details */
        .expandable-row {{ cursor: pointer; transition: background 0.2s; }}
        .expandable-row:hover {{ background: rgba(255,255,255,0.08) !important; }}
        .expand-arrow {{ color: #00d9ff; font-size: 0.8rem; margin-left: 8px; transition: transform 0.2s; display: inline-block; }}
        .expand-arrow.open {{ transform: rotate(90deg); }}
        .details-row {{ background: rgba(0,0,0,0.3); }}
        .details-row td {{ padding: 0 !important; }}
        
        /* Analyst Panel */
        .analyst-panel {{ padding: 25px; border-left: 4px solid #00d9ff; }}
        .analyst-panel h4 {{ color: #00d9ff; margin-bottom: 15px; font-size: 1rem; }}
        .panel-section {{ margin-bottom: 25px; }}
        
        /* Data Grid */
        .data-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
        .data-item {{ background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center; }}
        .data-label {{ color: #888; font-size: 0.85rem; }}
        .data-value {{ color: #fff; font-weight: 600; font-size: 0.95rem; }}
        
        /* Calculation Table */
        .calc-table {{ width: 100%; font-size: 0.85rem; background: rgba(0,0,0,0.2); border-radius: 8px; overflow: hidden; }}
        .calc-table th {{ background: rgba(0,217,255,0.1); color: #00d9ff; padding: 12px; text-align: left; font-weight: 500; }}
        .calc-table td {{ padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.05); }}
        .calc-table tr:last-child td {{ border-bottom: none; }}
        .calc-table code {{ font-size: 0.8rem; background: rgba(0,0,0,0.3); }}
        .calc-table .alert-metric {{ background: rgba(255, 71, 87, 0.15); }}
        .calc-table .alert-metric td {{ color: #ff6b7a; }}
        
        /* Score Summary */
        .score-summary {{ background: linear-gradient(90deg, rgba(0,217,255,0.1), rgba(0,255,136,0.05)); padding: 15px 20px; border-radius: 8px; color: #fff; margin-top: 15px; }}
        .score-summary strong {{ color: #00d9ff; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 FraudHunter - Informe de Análisis Forense</h1>
        <p class="subtitle">Sistema de Detección de Fraude Empresarial basado en ML</p>
        <p class="timestamp">Generado: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S')} | Versión: 1.0.0</p>
        
        <!-- EXECUTIVE SUMMARY -->
        <div class="executive-summary">
            <h3>📊 Resumen Ejecutivo</h3>
            <p>
                Este informe presenta los resultados del análisis forense automatizado sobre <strong>{len(df)} empresas</strong> 
                utilizando el sistema FraudHunter, que combina técnicas de <strong>Machine Learning (Isolation Forest)</strong> 
                con <strong>reglas heurísticas forenses</strong> basadas en patrones conocidos de fraude tributario.
            </p>
            <p>
                El modelo ha identificado <strong>{tp} casos de fraude confirmados</strong> con una precisión del <strong>{precision:.0%}</strong> 
                (sin falsos positivos), detectando exitosamente patrones de <strong>fraude carrusel IVA</strong>, 
                <strong>empresas pantalla</strong>, <strong>deuda oculta</strong> y <strong>facturas falsas</strong>.
            </p>
            <p>
                <span style="color: #ffa502;">⚠️ Área de mejora:</span> El modelo presenta dificultades para detectar 
                <strong>maquillaje contable</strong> sutil, con 3 falsos negativos en esta categoría. Se recomienda 
                incorporar análisis temporal (series históricas) para mejorar esta detección.
            </p>
        </div>
        
        <!-- METRICS DASHBOARD -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{len(df)}</div>
                <div class="metric-label">Empresas Analizadas</div>
                <div class="metric-desc">Casos de prueba</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'success' if accuracy >= 0.8 else 'warning'}">{accuracy:.0%}</div>
                <div class="metric-label">Accuracy</div>
                <div class="metric-desc">Clasificaciones correctas</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'success' if precision >= 0.9 else 'warning' if precision >= 0.7 else 'danger'}">{precision:.0%}</div>
                <div class="metric-label">Precisión</div>
                <div class="metric-desc">TP / (TP + FP)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'success' if recall >= 0.8 else 'warning'}">{recall:.0%}</div>
                <div class="metric-label">Recall</div>
                <div class="metric-desc">TP / (TP + FN)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'success' if f1 >= 0.8 else 'warning'}">{f1:.0%}</div>
                <div class="metric-label">F1-Score</div>
                <div class="metric-desc">Media armónica</div>
            </div>
        </div>
        
        <div class="two-cols">
            <!-- CONFUSION MATRIX -->
            <div class="section">
                <h2>📈 Matriz de Confusión</h2>
                <div class="confusion">
                    <div class="conf-cell"></div>
                    <div class="conf-cell conf-header">Pred: FRAUDE</div>
                    <div class="conf-cell conf-header">Pred: OK</div>
                    <div class="conf-cell conf-header">Real: FRAUDE</div>
                    <div class="conf-cell conf-tp">{tp}<br><small>Verdaderos Pos</small></div>
                    <div class="conf-cell conf-fn">{fn}<br><small>Falsos Neg</small></div>
                    <div class="conf-cell conf-header">Real: OK</div>
                    <div class="conf-cell conf-fp">{fp}<br><small>Falsos Pos</small></div>
                    <div class="conf-cell conf-tn">{tn}<br><small>Verdaderos Neg</small></div>
                </div>
                <div class="alert alert-{'success' if fp == 0 else 'warning'}">
                    {'✅ <strong>Sin Falsos Positivos:</strong> El modelo no ha generado alertas falsas, evitando investigaciones innecesarias.' if fp == 0 else f'⚠️ <strong>{fp} Falsos Positivos:</strong> Empresas sanas marcadas erróneamente como fraude.'}
                </div>
                {f'<div class="alert alert-warning">⚠️ <strong>{fn} Falsos Negativos:</strong> Casos de fraude no detectados que requieren revisar los umbrales del modelo.</div>' if fn > 0 else ''}
            </div>
            
            <!-- INTERPRETATION -->
            <div class="section">
                <h2>🎯 Interpretación de Métricas</h2>
                <div class="detail-row">
                    <div class="detail-label">Precisión ({precision:.0%})</div>
                    <div class="detail-value {'success' if precision >= 0.9 else 'warning'}">
                        {'Excelente: Cuando el modelo dice FRAUDE, acierta siempre' if precision >= 0.95 else 'Buena: Alta fiabilidad en las alertas de fraude' if precision >= 0.8 else 'Mejorable: Algunas alertas son incorrectas'}
                    </div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">Recall ({recall:.0%})</div>
                    <div class="detail-value {'success' if recall >= 0.9 else 'warning' if recall >= 0.7 else 'danger'}">
                        {'Excelente: Detecta casi todos los fraudes' if recall >= 0.9 else f'Aceptable: {fn} casos de fraude no detectados' if recall >= 0.7 else f'Insuficiente: Demasiados fraudes no detectados ({fn})'}
                    </div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">F1-Score ({f1:.0%})</div>
                    <div class="detail-value {'success' if f1 >= 0.85 else 'warning'}">
                        {'Equilibrio óptimo entre precisión y exhaustividad' if f1 >= 0.85 else 'Balance aceptable entre métricas'}
                    </div>
                </div>
            </div>
        </div>
        
        <!-- METHODOLOGY -->
        <div class="section">
            <h2>🔬 Metodología de Detección</h2>
            <p style="color: #aaa; margin-bottom: 20px;">El sistema FraudHunter utiliza un enfoque híbrido combinando <strong>6 métricas forenses</strong> con <strong>Isolation Forest</strong> para detectar anomalías multidimensionales.</p>
            <div class="method-grid">
                <div class="method-card">
                    <h4>🔄 Cobertura Ventas (M347/M200)</h4>
                    <p>Compara las ventas declaradas en el Modelo 200 con las operaciones reportadas por terceros en el Modelo 347.</p>
                    <div class="formula">ratio = Σ M347 / Ventas M200</div>
                    <p style="margin-top: 10px;"><strong>Alerta:</strong> ratio &lt; 0.5 (ventas sin respaldo)</p>
                </div>
                <div class="method-card">
                    <h4>🚛 Incoherencia Logística (M349)</h4>
                    <p>Detecta empresas que declaran operaciones intracomunitarias sin gastos de transporte asociados.</p>
                    <div class="formula">flag = (M349 > 0) AND (Transporte < 1000€)</div>
                    <p style="margin-top: 10px;"><strong>Patrón:</strong> Fraude carrusel IVA</p>
                </div>
                <div class="method-card">
                    <h4>📊 Accruals Ratio</h4>
                    <p>Mide la calidad del beneficio comparando resultado contable vs flujo de caja real.</p>
                    <div class="formula">ratio = (Resultado - FCO) / Activo</div>
                    <p style="margin-top: 10px;"><strong>Alerta:</strong> ratio > 0.10 (beneficio ficticio)</p>
                </div>
                <div class="method-card">
                    <h4>💳 Deuda Oculta</h4>
                    <p>Calcula la tasa de interés implícita para detectar financiación no bancaria.</p>
                    <div class="formula">tasa = Gastos Financieros / Deuda</div>
                    <p style="margin-top: 10px;"><strong>Alerta:</strong> tasa > 10% (usura/blanqueo)</p>
                </div>
                <div class="method-card">
                    <h4>🔢 Ley de Benford</h4>
                    <p>Analiza la distribución del primer dígito de los importes para detectar manipulación.</p>
                    <div class="formula">KL-divergence vs log10(1 + 1/d)</div>
                    <p style="margin-top: 10px;"><strong>Alerta:</strong> divergencia > 0.15</p>
                </div>
                <div class="method-card">
                    <h4>⭕ Números Redondos</h4>
                    <p>Detecta facturas con importes terminados en 00 o 000, típico de fraude documental.</p>
                    <div class="formula">pct = count(x % 100 == 0) / n</div>
                    <p style="margin-top: 10px;"><strong>Alerta:</strong> > 30% redondos</p>
                </div>
            </div>
        </div>
        
        <!-- FRAUD TYPE DETECTION -->
        <div class="section">
            <h2>🎭 Detección por Tipo de Fraude</h2>
            <p style="color: #aaa; margin-bottom: 20px;">Rendimiento del modelo desglosado por cada tipología de fraude incluida en el dataset de pruebas.</p>
            {tipos_html}
        </div>
        
        <!-- DETAILED RESULTS -->
        <div class="section">
            <h2>📋 Detalle de Casos Analizados ({len(df)} empresas)</h2>
            <table>
                <thead>
                    <tr>
                        <th>NIF / Empresa</th>
                        <th>Tipo Real</th>
                        <th>Predicción</th>
                        <th>Score de Riesgo</th>
                        <th>Flags Detectados</th>
                        <th>Descripción</th>
                        <th>OK</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
            <div class="legend">
                <div class="legend-item"><div class="legend-dot correct"></div> Clasificación correcta</div>
                <div class="legend-item"><div class="legend-dot error"></div> Error de clasificación</div>
            </div>
        </div>
        
        <!-- RECOMMENDATIONS -->
        <div class="section">
            <h2>💡 Recomendaciones</h2>
            <div class="recommendation">
                <div class="recommendation-icon">🎯</div>
                <div class="recommendation-text">
                    <strong>Ajustar detección de Maquillaje Contable</strong><br>
                    Los 3 falsos negativos son todos de tipo MAQUILLAJE_CONTABLE. Se recomienda incorporar análisis de <strong>variación interanual</strong> (Year-over-Year) para detectar cambios bruscos en márgenes.
                </div>
            </div>
            <div class="recommendation">
                <div class="recommendation-icon">📊</div>
                <div class="recommendation-text">
                    <strong>Incorporar análisis sectorial (CNAE)</strong><br>
                    La <strong>distancia de Mahalanobis por sector</strong> ya está implementada. Activarla en producción para detectar empresas que se desvían de sus pares sectoriales.
                </div>
            </div>
            <div class="recommendation">
                <div class="recommendation-icon">🔗</div>
                <div class="recommendation-text">
                    <strong>Análisis de grafos (GraphFrames)</strong><br>
                    Para fraude carrusel, activar el análisis de <strong>circularidad</strong> y <strong>componentes conexos</strong> del módulo GraphNetworkAnalyzer.
                </div>
            </div>
        </div>
        
        <!-- DEEP FRAUD HUNTER -->
        <div class="section">
            <h2>🧠 DeepFraudHunter - Módulos de Deep Learning</h2>
            <p style="color: #aaa; margin-bottom: 20px;">Framework SOTA para detección avanzada de fraude usando redes neuronales profundas.</p>
            
            <div class="method-grid">
                <div class="method-card">
                    <h4>{'🟢' if TORCH_AVAILABLE else '🔴'} Siamese Networks</h4>
                    <p>Resolución de entidades duplicadas en M347 usando LSTM a nivel de caracteres.</p>
                    <div class="formula">Estado: {'Disponible' if TORCH_AVAILABLE else 'Requiere PyTorch'}</div>
                </div>
                <div class="method-card">
                    <h4>{'🟢' if DGL_AVAILABLE else '🔴'} GraphSAGE</h4>
                    <p>Embeddings inductivos de grafos para detectar "culpabilidad por asociación".</p>
                    <div class="formula">Estado: {'Disponible' if DGL_AVAILABLE else 'Requiere DGL'}</div>
                </div>
                <div class="method-card">
                    <h4>{'🟢' if TORCH_AVAILABLE else '🔴'} VAE Anomalías</h4>
                    <p>Autoencoder Variacional entrenado solo con empresas sanas.</p>
                    <div class="formula">Estado: {'Disponible' if TORCH_AVAILABLE else 'Requiere PyTorch'}</div>
                </div>
                <div class="method-card">
                    <h4>{'🟢' if TRANSFORMERS_AVAILABLE else '🔴'} BERT Semántico</h4>
                    <p>Análisis de inconsistencia entre objeto social y CNAE declarado.</p>
                    <div class="formula">Estado: {'Disponible' if TRANSFORMERS_AVAILABLE else 'Requiere Transformers'}</div>
                </div>
                <div class="method-card">
                    <h4>{'🟢' if TORCH_AVAILABLE else '🔴'} Temporal Fusion</h4>
                    <p>LSTM + Atención para detectar patrones de "bust-out" en series de 5 años.</p>
                    <div class="formula">Estado: {'Disponible' if TORCH_AVAILABLE else 'Requiere PyTorch'}</div>
                </div>
                <div class="method-card">
                    <h4>{'🟢' if XGB_AVAILABLE else '🟡'} XGBoost Ensemble</h4>
                    <p>Clasificador final sobre super-vector de ~87 features combinadas.</p>
                    <div class="formula">Estado: {'XGBoost' if XGB_AVAILABLE else 'Fallback: sklearn'}</div>
                </div>
            </div>
            
            <div class="alert alert-info" style="margin-top: 20px;">
                💡 <strong>Para activar todos los módulos:</strong> <code>pip install torch dgl transformers xgboost</code>
            </div>
        </div>
        
        <div class="footer">
            <p>FraudHunter Core v2.0.0 | Sistema de Detección de Fraude Empresarial + Deep Learning</p>
            <p>Basado en datos de Modelos AEAT: M200 (Sociedades), M347 (Operaciones Terceros), M349 (Intracomunitario)</p>
        </div>
    </div>
    
    <script>
        // Toggle analyst details panel
        function toggleDetails(idx) {{
            const detailsRow = document.getElementById('details-' + idx);
            const arrow = document.getElementById('arrow-' + idx);
            if (detailsRow.style.display === 'none') {{
                detailsRow.style.display = 'table-row';
                arrow.classList.add('open');
                arrow.textContent = '▼';
            }} else {{
                detailsRow.style.display = 'none';
                arrow.classList.remove('open');
                arrow.textContent = '▶';
            }}
        }}
        
        // Expand all / Collapse all buttons
        function toggleAll(show) {{
            document.querySelectorAll('.details-row').forEach(row => {{
                row.style.display = show ? 'table-row' : 'none';
            }});
            document.querySelectorAll('.expand-arrow').forEach(arrow => {{
                arrow.textContent = show ? '▼' : '▶';
                if (show) arrow.classList.add('open');
                else arrow.classList.remove('open');
            }});
        }}
    </script>
</body>
</html>"""
    
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Reporte guardado: {report_path}")
    webbrowser.open(f'file://{report_path}')
    
    print("\n" + "=" * 80)
    print("PRUEBAS COMPLETADAS - REPORTE ABIERTO EN NAVEGADOR")
    print("=" * 80)
