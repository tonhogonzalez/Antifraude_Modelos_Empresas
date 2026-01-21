"""
Generate 100K Company Dataset - Optimized for GitHub (<100MB)
==============================================================
Genera dataset de 100.000 empresas optimizado para no superar 100MB.
"""
import pandas as pd
import numpy as np
import random
from pathlib import Path

def generate_100k_dataset():
    N_COMPANIES = 100_000
    print(f"🚀 Generando dataset de {N_COMPANIES:,} empresas...")
    
    np.random.seed(42)
    random.seed(42)
    
    # =========================================================================
    # 1. DATOS BASE DE EMPRESAS
    # =========================================================================
    print("   [1/4] Generando datos base...")
    
    # NIFs únicos con todas las letras válidas para CIF
    letras_cif = 'ABCDEFGHJNPQRSUVW'  # 17 letras válidas para CIF
    nifs = [f"{random.choice(letras_cif)}{str(i).zfill(7)}{random.choice('ABCDEFGHJKLMNPQRSTVWXYZ')}" 
            for i in range(N_COMPANIES)]
    
    # CNAE completos - Todos los sectores principales de la economía española
    cnaes = {
        # SECTOR PRIMARIO
        '0111': ('Cultivo de cereales', 'Agricultura'),
        '0311': ('Pesca marina', 'Pesca'),
        '0510': ('Extracción de antracita y hulla', 'Minería'),
        # INDUSTRIA
        '1011': ('Procesado de carne', 'Alimentación'),
        '1102': ('Elaboración de vinos', 'Bebidas'),
        '1412': ('Confección de ropa de trabajo', 'Textil'),
        '2010': ('Fabricación de productos químicos', 'Química'),
        '2410': ('Fabricación de productos siderúrgicos', 'Metalurgia'),
        '2511': ('Fabricación de estructuras metálicas', 'Metal'),
        '2611': ('Fabricación de componentes electrónicos', 'Electrónica'),
        '2910': ('Fabricación de vehículos de motor', 'Automoción'),
        # CONSTRUCCIÓN
        '4110': ('Promoción inmobiliaria', 'Inmobiliaria'),
        '4121': ('Construcción de edificios residenciales', 'Construcción'),
        '4221': ('Construcción de redes de suministros', 'Infraestructuras'),
        '4321': ('Instalaciones eléctricas', 'Instalaciones'),
        # COMERCIO
        '4511': ('Venta de automóviles', 'Automoción'),
        '4719': ('Comercio minorista', 'Retail'),
        '4711': ('Supermercados e hipermercados', 'Alimentación'),
        '4730': ('Comercio de combustible', 'Energía'),
        '4641': ('Comercio de textiles', 'Textil'),
        # TRANSPORTE Y LOGÍSTICA
        '4941': ('Transporte de mercancías por carretera', 'Transporte'),
        '5210': ('Depósito y almacenamiento', 'Logística'),
        '5229': ('Actividades anexas al transporte', 'Logística'),
        # HOSTELERÍA
        '5510': ('Hoteles y alojamientos similares', 'Hostelería'),
        '5610': ('Restaurantes y puestos de comidas', 'Restauración'),
        # TECNOLOGÍA Y COMUNICACIONES
        '6110': ('Telecomunicaciones por cable', 'Telecomunicaciones'),
        '6201': ('Actividades de programación informática', 'Tecnología'),
        '6311': ('Procesamiento de datos, hosting', 'Tecnología'),
        # FINANZAS Y SEGUROS
        '6419': ('Otra intermediación monetaria', 'Banca'),
        '6512': ('Seguros distintos de vida', 'Seguros'),
        '6630': ('Actividades de gestión de fondos', 'Inversión'),
        # INMOBILIARIO
        '6810': ('Compraventa de bienes inmobiliarios', 'Inmobiliaria'),
        '6820': ('Alquiler de bienes inmobiliarios', 'Inmobiliaria'),
        '6831': ('Agentes de la propiedad inmobiliaria', 'Inmobiliaria'),
        # SERVICIOS PROFESIONALES
        '6910': ('Actividades jurídicas', 'Legal'),
        '6920': ('Actividades de contabilidad', 'Consultoría'),
        '7010': ('Actividades de sedes centrales', 'Corporativo'),
        '7022': ('Otras actividades de consultoría', 'Consultoría'),
        '7111': ('Servicios técnicos de arquitectura', 'Ingeniería'),
        '7112': ('Servicios técnicos de ingeniería', 'Ingeniería'),
        '7311': ('Agencias de publicidad', 'Marketing'),
        # SANIDAD Y EDUCACIÓN
        '8510': ('Educación preprimaria', 'Educación'),
        '8520': ('Educación primaria', 'Educación'),
        '8610': ('Actividades hospitalarias', 'Sanidad'),
        '8621': ('Actividades de medicina general', 'Sanidad'),
        # OTROS SERVICIOS
        '9001': ('Artes escénicas', 'Entretenimiento'),
        '9311': ('Gestión de instalaciones deportivas', 'Deporte'),
        '9602': ('Peluquería y otros tratamientos de belleza', 'Servicios'),
    }
    
    cnae_codes = list(cnaes.keys())
    cnae_weights = [1/len(cnae_codes)] * len(cnae_codes)  # Distribución uniforme
    
    sector_ids = np.random.choice(cnae_codes, N_COMPANIES, p=cnae_weights)
    sector_names = [cnaes[s][1] for s in sector_ids]
    sector_descriptions = [cnaes[s][0] for s in sector_ids]
    
    # TAMAÑO DE EMPRESAS (según criterios UE)
    # Micro: <10 empleados, <2M facturación
    # Pequeña: <50 empleados, <10M facturación
    # Mediana: <250 empleados, <50M facturación
    # Grande: >=250 empleados o >=50M facturación
    
    tamaños = ['Micro', 'Pequeña', 'Mediana', 'Grande']
    tamaño_weights = [0.85, 0.10, 0.04, 0.01]  # Distribución realista España
    tamaño_arr = np.random.choice(tamaños, N_COMPANIES, p=tamaño_weights)
    
    # Facturación base según tamaño
    ventas = np.zeros(N_COMPANIES, dtype=np.int64)
    ventas[tamaño_arr == 'Micro'] = np.random.randint(10000, 2000000, size=(tamaño_arr == 'Micro').sum())
    ventas[tamaño_arr == 'Pequeña'] = np.random.randint(2000000, 10000000, size=(tamaño_arr == 'Pequeña').sum())
    ventas[tamaño_arr == 'Mediana'] = np.random.randint(10000000, 50000000, size=(tamaño_arr == 'Mediana').sum())
    ventas[tamaño_arr == 'Grande'] = np.random.randint(50000000, 500000000, size=(tamaño_arr == 'Grande').sum())
    
    # Tipos de empresa (5% fraude)
    tipos = np.array(['SANA'] * N_COMPANIES)
    n_fraude = int(N_COMPANIES * 0.05)  # 5000 empresas fraude
    fraud_idx = np.random.choice(N_COMPANIES, n_fraude, replace=False)
    fraud_types = ['FRAUDE_CARRUSEL', 'FRAUDE_MAQUILLAJE', 'FRAUDE_PANTALLA', 'FRAUDE_ZOMBIE']
    tipos[fraud_idx] = np.random.choice(fraud_types, n_fraude)
    
    df_empresas = pd.DataFrame({
        'id': range(N_COMPANIES),
        'nif': nifs,
        'nombre': [f"{cnaes[s][0][:20]}_{i}" for i, s in zip(range(N_COMPANIES), sector_ids)],
        'tipo': tipos,
        'sector_cnae': sector_names,
        'cnae_code': sector_ids,
        'tamaño': tamaño_arr,
        'ventas_base': ventas
    })
    
    print(f"      ✓ Empresas: {len(df_empresas):,}")
    
    # =========================================================================
    # 2. DATOS EAV (Entity-Attribute-Value) - COMPLETO
    # =========================================================================
    print("   [2/4] Generando EAV con todas las casillas fiscales...")
    
    # TODAS las casillas del M200 necesarias para los algoritmos
    # Mapeo: variable -> casilla AEAT
    casillas_m200 = {
        'C00255': 'cifra_negocios',      # Ingresos de Explotación
        'C00258': 'aprovisionamientos',  # Compras
        'C00260': 'otros_ingresos',      # Subvenciones
        'C00263': 'gastos_personal',     # Sueldos + Seg. Social
        'C00279': 'otros_gastos_expl',   # Transportes/Servicios Ext.
        'C00280': 'amortizaciones',      # Amortizaciones
        'C00296': 'gastos_financieros',  # Clave para deuda oculta
        'C00500': 'resultado_ejercicio', # Beneficio Neto
        'C00033': 'total_activo',        # Total Activo
        'C00062': 'patrimonio_neto',     # Patrimonio Neto
        'C00195': 'deuda_bancaria_lp',   # Deudas L/P
        'C00215': 'deuda_bancaria_cp',   # Deudas C/P
        'C00032': 'efectivo_tesoreria',  # Caja
        'C00257': 'trabajos_realizados_activo',  # Capitalización de gastos
    }
    
    eav_data = []
    
    for i, nif in enumerate(nifs):
        venta = int(ventas[i])
        tipo = tipos[i]
        is_carrusel = tipo == 'FRAUDE_CARRUSEL'
        is_maquillaje = tipo == 'FRAUDE_MAQUILLAJE'
        is_pantalla = tipo == 'FRAUDE_PANTALLA'
        is_zombie = tipo == 'FRAUDE_ZOMBIE'
        is_fraud = tipo.startswith('FRAUDE')
        
        # Calcular métricas financieras realistas según tipo
        aprovisionamientos = int(venta * np.random.uniform(0.5, 0.7))
        gastos_personal = int(venta * np.random.uniform(0.15, 0.25))
        otros_gastos = int(venta * np.random.uniform(0.02, 0.05))  # Transporte
        amortizaciones = int(venta * np.random.uniform(0.02, 0.05))
        otros_ingresos = int(venta * np.random.uniform(0, 0.02))
        
        # Activo y pasivo
        total_activo = int(venta * np.random.uniform(0.8, 1.5))
        patrimonio_neto = int(total_activo * np.random.uniform(0.3, 0.6))
        deuda_lp = int(total_activo * np.random.uniform(0.1, 0.3))
        deuda_cp = int(total_activo * np.random.uniform(0.05, 0.15))
        efectivo = int(total_activo * np.random.uniform(0.05, 0.15))
        gastos_financieros = int((deuda_lp + deuda_cp) * 0.05)  # 5% interés
        
        # Resultado según margen típico
        resultado = int(venta * np.random.uniform(0.03, 0.12))
        
        # APLICAR PATRONES DE FRAUDE
        if is_carrusel:
            # Carrusel: Mucha venta, poco transporte, alto M349
            otros_gastos = int(venta * 0.001)  # Transporte irrisorio
            resultado = int(venta * 0.005)  # Margen mínimo
            entregas_ue = int(venta * 0.9)  # 90% ventas a UE
        elif is_maquillaje:
            # Maquillaje: Cifras perfectamente redondas
            resultado = int(round(venta * 0.10, -3))  # Exactamente 10% redondeado
            gastos_financieros = int(round(gastos_financieros, -3))
            entregas_ue = 0
        elif is_pantalla:
            # Pantalla: Sin personal, activo inflado
            gastos_personal = int(venta * 0.01)  # Casi sin empleados
            total_activo = int(venta * 5)  # Activo muy alto
            otros_gastos = 0
            entregas_ue = 0
        elif is_zombie:
            # Zombie: Patrimonio negativo, sigue operando
            patrimonio_neto = -abs(patrimonio_neto)
            resultado = -int(venta * 0.1)  # Pérdidas
            efectivo = 1000  # Caja vacía
            entregas_ue = 0
        else:
            entregas_ue = int(venta * np.random.uniform(0, 0.1))  # Sanas: 0-10% UE
        
        # Añadir todas las casillas
        eav_data.append({'nif': nif, 'casilla': 'C00255', 'valor': venta})
        eav_data.append({'nif': nif, 'casilla': 'C00258', 'valor': aprovisionamientos})
        eav_data.append({'nif': nif, 'casilla': 'C00260', 'valor': otros_ingresos})
        eav_data.append({'nif': nif, 'casilla': 'C00263', 'valor': gastos_personal})
        eav_data.append({'nif': nif, 'casilla': 'C00279', 'valor': otros_gastos})
        eav_data.append({'nif': nif, 'casilla': 'C00280', 'valor': amortizaciones})
        eav_data.append({'nif': nif, 'casilla': 'C00296', 'valor': gastos_financieros})
        eav_data.append({'nif': nif, 'casilla': 'C00500', 'valor': resultado})
        eav_data.append({'nif': nif, 'casilla': 'C00033', 'valor': total_activo})
        eav_data.append({'nif': nif, 'casilla': 'C00062', 'valor': patrimonio_neto})
        eav_data.append({'nif': nif, 'casilla': 'C00195', 'valor': deuda_lp})
        eav_data.append({'nif': nif, 'casilla': 'C00215', 'valor': deuda_cp})
        eav_data.append({'nif': nif, 'casilla': 'C00032', 'valor': efectivo})
        eav_data.append({'nif': nif, 'casilla': 'ENTREGAS_UE', 'valor': entregas_ue})  # M349
    
    df_eav = pd.DataFrame(eav_data)
    print(f"      ✓ EAV Rows: {len(df_eav):,} ({len(casillas_m200)+1} casillas por empresa)")
    
    # =========================================================================
    # 3. M347 (Transacciones) - SPARSE
    # =========================================================================
    print("   [3/4] Generando transacciones M347...")
    
    # Solo 1 transacción por empresa (promedio) para mantener tamaño pequeño
    n_trans = N_COMPANIES
    
    declarantes = np.random.choice(nifs, n_trans)
    contrapartes = np.random.choice(nifs, n_trans)
    importes = np.random.lognormal(10, 1, n_trans).astype(np.int32)
    
    df_m347 = pd.DataFrame({
        'nif_declarante': declarantes,
        'nif_contraparte': contrapartes,
        'importe': importes,
        'is_circular': False
    })
    
    # Añadir circularidad para fraudes carrusel
    carrusel_nifs = df_empresas[df_empresas['tipo'] == 'FRAUDE_CARRUSEL']['nif'].values
    if len(carrusel_nifs) > 10:
        n_circ = len(carrusel_nifs)
        df_circ = pd.DataFrame({
            'nif_declarante': carrusel_nifs,
            'nif_contraparte': np.roll(carrusel_nifs, 1),
            'importe': np.random.randint(100000, 500000, n_circ),
            'is_circular': True
        })
        df_m347 = pd.concat([df_m347, df_circ], ignore_index=True)
    
    print(f"      ✓ M347 Rows: {len(df_m347):,}")
    
    # =========================================================================
    # 4. GUARDAR CSVs
    # =========================================================================
    print("   [4/4] Guardando CSVs...")
    
    base_path = Path(__file__).parent
    
    # Guardar con compresión de tipos
    df_empresas.to_csv(base_path / 'data_empresas.csv', index=False)
    df_eav.to_csv(base_path / 'data_eav.csv', index=False)
    df_m347.to_csv(base_path / 'data_m347.csv', index=False)
    
    # Calcular tamaños
    size_emp = (base_path / 'data_empresas.csv').stat().st_size / 1024 / 1024
    size_eav = (base_path / 'data_eav.csv').stat().st_size / 1024 / 1024
    size_m347 = (base_path / 'data_m347.csv').stat().st_size / 1024 / 1024
    total_size = size_emp + size_eav + size_m347
    
    print(f"\n✅ DATASET GENERADO EXITOSAMENTE")
    print(f"   📊 Empresas: {len(df_empresas):,}")
    print(f"   📊 EAV Rows: {len(df_eav):,}")
    print(f"   📊 M347 Rows: {len(df_m347):,}")
    print(f"\n   💾 Tamaños de archivo:")
    print(f"      - data_empresas.csv: {size_emp:.2f} MB")
    print(f"      - data_eav.csv: {size_eav:.2f} MB")
    print(f"      - data_m347.csv: {size_m347:.2f} MB")
    print(f"      - TOTAL: {total_size:.2f} MB")
    
    if total_size > 95:
        print("⚠️ ADVERTENCIA: Tamaño cercano al límite de 100MB")
    else:
        print(f"✅ OK: {100 - total_size:.1f} MB de margen bajo límite GitHub")
    
    return df_empresas, df_eav, df_m347

if __name__ == "__main__":
    generate_100k_dataset()
