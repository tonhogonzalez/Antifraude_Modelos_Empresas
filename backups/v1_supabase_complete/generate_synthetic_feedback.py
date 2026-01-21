"""
Script para generar dataset sintético de feedback de analistas.
Genera +150 registros con variedad de veredictos, razones y tipologías
para demostrar las capacidades del sistema de aprendizaje continuo.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Añadir path del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from continuous_learning import (
    FeedbackStorePandas, FeedbackRecord,
    REJECTION_REASON_CODES, FRAUD_TYPOLOGY_CODES,
    VERDICT_FALSE_POSITIVE, VERDICT_FRAUD, VERDICT_WATCHLIST
)


def generate_synthetic_feedback(n_records: int = 150):
    """
    Genera feedback sintético realista.
    
    Distribución objetivo:
    - 40% Fraudes confirmados
    - 45% Falsos Positivos  
    - 15% Watchlist
    """
    
    # Sectores y sus características
    SECTORES = {
        '4511': {'nombre': 'Venta de automóviles', 'fp_rate': 0.6},  # Alto FP
        '4719': {'nombre': 'Comercio minorista', 'fp_rate': 0.5},
        '6201': {'nombre': 'Programación informática', 'fp_rate': 0.4},
        '6831': {'nombre': 'Agentes inmobiliarios', 'fp_rate': 0.3},  # Bajo FP (más fraude)
        '4941': {'nombre': 'Transporte de mercancías', 'fp_rate': 0.35},
        '4321': {'nombre': 'Instalaciones eléctricas', 'fp_rate': 0.45},
        '6810': {'nombre': 'Compraventa inmuebles', 'fp_rate': 0.25},  # Muy bajo FP
        '7022': {'nombre': 'Consultoría empresarial', 'fp_rate': 0.55}
    }
    
    # Analistas simulados
    ANALYSTS = ['analyst_01', 'analyst_02', 'analyst_03', 'inspector_jefe', 'auditor_senior']
    
    # Flags activos típicos
    FLAGS = [
        'flag_empresa_pantalla', 'flag_incoherencia_logistica', 'flag_hidden_debt',
        'flag_numeros_redondos', 'flag_sectoral_outlier', 'flag_cobertura_baja',
        'flag_accruals_anomalo', 'flag_benford_anomaly'
    ]
    
    records = []
    
    # Generar fechas distribuidas en los últimos 90 días
    base_date = datetime.now()
    
    for i in range(n_records):
        # Seleccionar sector
        cnae = random.choice(list(SECTORES.keys()))
        sector_info = SECTORES[cnae]
        
        # Determinar veredicto basado en características del sector
        rand = random.random()
        if rand < 0.15:  # 15% Watchlist
            verdict = VERDICT_WATCHLIST
            rejection_reason = None
            fraud_typology = None
        elif rand < 0.15 + sector_info['fp_rate'] * 0.6:  # Variable FP según sector
            verdict = VERDICT_FALSE_POSITIVE
            # Elegir razón de rechazo
            reject_weights = {
                'SECTOR_NORMAL': 0.35 if sector_info['fp_rate'] > 0.4 else 0.15,
                'DATA_ERROR': 0.15,
                'LEGITIMATE_BUSINESS': 0.25,
                'SEASONAL': 0.10,
                'ONE_TIME': 0.10,
                'OTHER': 0.05
            }
            rejection_reason = random.choices(
                list(reject_weights.keys()),
                weights=list(reject_weights.values())
            )[0]
            fraud_typology = None
        else:  # Fraude
            verdict = VERDICT_FRAUD
            rejection_reason = None
            # Elegir tipología según sector
            if cnae in ['6810', '6831']:  # Inmobiliario
                typology_weights = {'PANTALLA': 0.4, 'FACTURAS_FALSAS': 0.3, 'CONTABILIDAD': 0.2, 'DEUDA_OCULTA': 0.1}
            elif cnae in ['4941', '4511']:  # Transporte/Autos
                typology_weights = {'CARRUSEL': 0.5, 'FACTURAS_FALSAS': 0.3, 'PANTALLA': 0.2}
            else:
                typology_weights = {'FACTURAS_FALSAS': 0.3, 'CONTABILIDAD': 0.25, 'PANTALLA': 0.2, 'CARRUSEL': 0.15, 'DEUDA_OCULTA': 0.1}
            
            fraud_typology = random.choices(
                list(typology_weights.keys()),
                weights=list(typology_weights.values())
            )[0]
        
        # Generar NIF
        nif = f"B{str(random.randint(10000000, 99999999))}A"
        
        # Score original (más alto para fraudes confirmados)
        if verdict == VERDICT_FRAUD:
            score = random.uniform(0.65, 0.95)
        elif verdict == VERDICT_WATCHLIST:
            score = random.uniform(0.50, 0.75)
        else:
            score = random.uniform(0.55, 0.85)
        
        # Confianza del analista (fraudes tienen mayor confianza)
        if verdict == VERDICT_FRAUD:
            confidence = random.choices([3, 4, 5], weights=[0.2, 0.4, 0.4])[0]
        elif verdict == VERDICT_WATCHLIST:
            confidence = random.choices([2, 3], weights=[0.6, 0.4])[0]
        else:
            confidence = random.choices([3, 4, 5], weights=[0.3, 0.4, 0.3])[0]
        
        # Flags activos
        n_flags = random.randint(1, 4)
        active_flags = random.sample(FLAGS, n_flags)
        
        # Fecha aleatoria en los últimos 90 días
        days_ago = random.randint(0, 90)
        fecha_alerta = base_date - timedelta(days=days_ago, hours=random.randint(0, 23))
        
        # Ventas
        ventas = random.uniform(100000, 5000000)
        
        # Crear feature vector simulado
        feature_vector = {
            'fraud_score_normalized': score,
            'cobertura_ventas': random.uniform(0.6, 1.2),
            'accruals_ratio': random.uniform(-0.1, 0.25),
            'productividad_laboral': random.uniform(50, 500),
            'pct_numeros_redondos': random.uniform(0.05, 0.6)
        }
        
        record = FeedbackRecord(
            nif=nif,
            analyst_verdict=verdict,
            fraud_score_original=score,
            feature_vector=feature_vector,
            rejection_reason_code=rejection_reason,
            fraud_typology_code=fraud_typology,
            analyst_confidence=confidence,
            analyst_id=random.choice(ANALYSTS),
            cnae_sector=sector_info['nombre'],
            ventas_netas=ventas,
            flags_active=active_flags,
            fecha_alerta=fecha_alerta
        )
        records.append(record)
    
    return records


def main():
    print("=" * 60)
    print("🔧 Generador de Feedback Sintético")
    print("=" * 60)
    
    # Generar registros
    n_records = 150
    print(f"\n📊 Generando {n_records} registros de feedback...")
    records = generate_synthetic_feedback(n_records)
    
    # Obtener store (con ruta absoluta)
    store = FeedbackStorePandas()
    print(f"📁 Guardando en: {store.storage_path}")
    
    # Guardar cada registro
    for i, record in enumerate(records):
        feedback_id = store.log_feedback(record)
        if (i + 1) % 30 == 0:
            print(f"   ✓ {i + 1}/{n_records} registros guardados...")
    
    print(f"\n✅ {n_records} registros guardados correctamente!")
    
    # Mostrar estadísticas
    print("\n" + "=" * 60)
    print("📈 ESTADÍSTICAS DEL DATASET GENERADO")
    print("=" * 60)
    
    counts = store.get_sample_count()
    print(f"\n📊 Conteo por veredicto:")
    print(f"   • Falsos Positivos: {counts['false_positives']}")
    print(f"   • Fraudes:          {counts['confirmed_fraud']}")
    print(f"   • Watchlist:        {counts['watchlist']}")
    print(f"   • TOTAL:            {counts['total']}")
    
    # Estadísticas de rechazo
    rejection_stats = store.analyze_rejection_reasons(window_days=90)
    print(f"\n🔍 Análisis de Causas de Rechazo:")
    print(f"   Total FPs analizados: {rejection_stats['total_rejections']}")
    if rejection_stats['by_reason']:
        for reason, stats in rejection_stats['by_reason'].items():
            print(f"   • {reason}: {stats['count']} ({stats['percentage']:.1f}%)")
    
    # Sectores con sugerencia de relajar umbral
    print(f"\n⚙️ Sectores para considerar ajuste de umbral:")
    for sector, stats in rejection_stats.get('by_sector', {}).items():
        if stats.get('suggest_relax_threshold'):
            print(f"   ⚠️ {sector}: {stats['sector_normal_pct']:.0f}% FPs son 'comportamiento normal'")
    
    # Tipologías de fraude
    typology_stats = store.get_fraud_typology_stats(window_days=90)
    print(f"\n🎯 Tipologías de Fraude Detectadas:")
    for typology, stats in typology_stats.get('by_typology', {}).items():
        print(f"   • {typology}: {stats['count']} casos - {stats['description']}")
    
    # Watchlist
    watchlist = store.get_watchlist_nifs()
    print(f"\n👁️ NIFs en Watchlist (Zona Gris): {len(watchlist)}")
    if watchlist[:3]:
        print(f"   Ejemplos: {', '.join(watchlist[:3])}...")
    
    # Estado del sistema
    is_ready, reason = store.is_ready_for_training()
    print(f"\n🤖 Estado del Sistema:")
    print(f"   {'✅' if is_ready else '⏳'} {reason}")
    
    print("\n" + "=" * 60)
    print("🎉 ¡Dataset listo para demostración!")
    print("=" * 60)


if __name__ == "__main__":
    main()
