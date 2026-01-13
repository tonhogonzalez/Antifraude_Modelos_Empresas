import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_massive_dataset(n_companies=500000):
    print(f"🚀 Iniciando generación masiva de {n_companies:,} empresas...")
    
    # 1. GENERACIÓN DE NIFS Y DATOS BÁSICOS (Vectorizado)
    # -------------------------------------------------------------------------
    print("   [1/4] Generando datos base...")
    ids = np.arange(n_companies)
    nifs = [f"A{str(i).zfill(8)}" for i in ids]
    
    # Sectores
    sectores = ['Comercio', 'Construccion', 'Tecnologia', 'Transporte', 'Hosteleria', 'Inmobiliaria', 'Industria']
    w_sectores = [0.25, 0.20, 0.10, 0.10, 0.15, 0.10, 0.10]
    sector_arr = np.random.choice(sectores, n_companies, p=w_sectores)
    
    # Tamaños (Micro vs Pyme vs Large) - Log-normal para facturación base
    # Media ~1M, pero con cola larga hasta 100M+
    mu, sigma = 13, 1.5  # Log-normal distribution parameters
    base_revenue = np.random.lognormal(mu, sigma, n_companies)
    base_revenue = np.clip(base_revenue, 10000, 500000000) # Clip entre 10k y 500M
    
    df_empresas = pd.DataFrame({
        'nif': nifs,
        'nombre': [f"Empresa_{i}_{s}" for i, s in zip(ids, sector_arr)],
        'sector_cnae': sector_arr,
        'cifra_negocio_base': base_revenue,
        'tipo': 'SANA' # Default
    })
    
    # 2. INYECCIÓN DE PATRONES DE FRAUDE (Vectorizado con máscaras)
    # -------------------------------------------------------------------------
    print("   [2/4] Inyectando patrones de fraude complejos...")
    
    # IDs aleatorios para fraudes
    n_fraude = int(n_companies * 0.05) # 5% fraude total ~25k empresas
    fraud_indices = np.random.choice(df_empresas.index, n_fraude, replace=False)
    
    # Tipos de fraude
    fraud_types = ['FRAUDE_CARRUSEL', 'FRAUDE_ZOMBIE', 'FRAUDE_FANTASMA', 'FRAUDE_BUSTOUT', 'FRAUDE_MAQUILLAJE']
    fraud_labels = np.random.choice(fraud_types, n_fraude)
    
    df_empresas.loc[fraud_indices, 'tipo'] = fraud_labels
    
    # -- Lógica Financiera Diferencial por Tipo --
    
    # Inicializar columnas financieras
    rev = df_empresas['cifra_negocio_base'].values
    res = rev * np.random.uniform(0.02, 0.15, n_companies) # Margen sano 2-15%
    assets = rev * np.random.uniform(0.5, 2.0, n_companies)
    cash = assets * np.random.uniform(0.05, 0.20, n_companies)
    debt = assets * np.random.uniform(0.10, 0.60, n_companies)
    m347_in = rev * np.random.uniform(0.4, 0.8, n_companies) # Compras declaradas
    m349_out = np.zeros(n_companies)
    gastos_transporte = rev * np.random.uniform(0.01, 0.05, n_companies)
    
    # Mascaras booleanas para aplicar lógica específica
    is_carrusel = df_empresas['tipo'] == 'FRAUDE_CARRUSEL'
    is_zombie = df_empresas['tipo'] == 'FRAUDE_ZOMBIE'
    is_fantasma = df_empresas['tipo'] == 'FRAUDE_FANTASMA'
    is_bustout = df_empresas['tipo'] == 'FRAUDE_BUSTOUT'
    is_maquillaje = df_empresas['tipo'] == 'FRAUDE_MAQUILLAJE'
    
    # APLICAR PATRONES
    
    # 1. CARRUSEL: Mucha venta UE (M349), poco transporte, compras locales altas
    m349_out[is_carrusel] = rev[is_carrusel] * 0.9 # 90% ventas a UE
    gastos_transporte[is_carrusel] = 1000 # Transporte irrisorio
    res[is_carrusel] = rev[is_carrusel] * 0.005 # Margen mínimo para no pagar impuesto
    
    # 2. ZOMBIE: Patrimonio neto negativo (Deuda > Activo), pero sigue operando
    debt[is_zombie] = assets[is_zombie] * 1.5 
    res[is_zombie] = -1 * (rev[is_zombie] * 0.1) # Pérdidas continuas
    cash[is_zombie] = 1000 # Caja vacía
    
    # 3. FANTASMA (Shell): Sin estructura, alto activo ficticio, sin personal (implícito)
    assets[is_fantasma] = rev[is_fantasma] * 10 # Activos inflados
    gastos_transporte[is_fantasma] = 0
    m347_in[is_fantasma] = 0 # No compra insumos reales
    
    # 4. BUST-OUT: Sube crédito rápido, saquea caja
    debt[is_bustout] = rev[is_bustout] * 2.0
    cash[is_bustout] = rev[is_bustout] * 0.8 # Tiene la caja del préstamo lista para fugar
    res[is_bustout] = -rev[is_bustout] # Pérdida masiva final
    
    # 5. MAQUILLAJE: Cifras redondas, márgenes perfectos
    res[is_maquillaje] = rev[is_maquillaje] * 0.1000000001 # Exactamente 10%
    
    # Añadir ruido a sanas
    noise = np.random.normal(1.0, 0.05, n_companies)
    
    # Asignar valores finales al DF
    df_empresas['cifra_negocio'] = rev
    df_empresas['resultado'] = res
    df_empresas['activo'] = assets
    df_empresas['caja'] = cash
    df_empresas['deuda_bancaria'] = debt
    df_empresas['total_m347'] = m347_in
    df_empresas['exportaciones_ue'] = m349_out
    df_empresas['gastos_transporte'] = gastos_transporte
    df_empresas['gastos_financieros'] = debt * 0.05 # 5% interest default
    
    # Ajustes finos post-vectorización
    # Ocultar deuda en zombies/bustout (Interés alto pero poca deuda balance)
    mask_hidden = (is_zombie | is_bustout) & (np.random.rand(n_companies) > 0.5)
    # Simulamos deuda oculta poniendo deuda baja en balance pero gastos fin altos
    real_debt = df_empresas.loc[mask_hidden, 'deuda_bancaria']
    df_empresas.loc[mask_hidden, 'deuda_bancaria'] = real_debt * 0.1 
    df_empresas.loc[mask_hidden, 'gastos_financieros'] = real_debt * 0.15 # 15% real rate implied
    
    # 3. GENERAR TRANSACCIONES M347 (Relaciones)
    # -------------------------------------------------------------------------
    print("   [3/4] Generando grafo de transacciones M347...")
    # Creamos un edgelist simplificado. 
    # Cada empresa compra a X proveedores.
    # Fraudes Carrusel compran a Truchas (Fantasmas)
    
    # Generar proveedores aleatorios para todos
    n_transactions = n_companies * 2 # Media de 2 conexiones conocidas (sparse para ser rápido)
    
    declarantes = np.random.choice(nifs, n_transactions)
    contrapartes = np.random.choice(nifs, n_transactions)
    importes = np.random.lognormal(10, 1, n_transactions)
    
    df_m347 = pd.DataFrame({
        'nif_declarante': declarantes,
        'nif_contraparte': contrapartes,
        'importe': importes,
        'is_circular': False
    })
    
    # Inyectar circularidad en Carruseles
    carrusel_nifs = df_empresas[is_carrusel]['nif'].values
    if len(carrusel_nifs) > 10:
        # Cadena A -> B -> C -> A
        n_circ = len(carrusel_nifs)
        # Shift
        compradores = carrusel_nifs
        vendedores = np.roll(carrusel_nifs, 1) # Cada uno compra al anterior
        
        df_circular = pd.DataFrame({
            'nif_declarante': compradores,
            'nif_contraparte': vendedores,
            'importe': np.random.uniform(500000, 2000000, n_circ),
            'is_circular': True
        })
        df_m347 = pd.concat([df_m347, df_circular], ignore_index=True)
    
    # 4. CONVERTIR A FORMATO EAV (Entity-Attribute-Value) para compatibilidad
    # -------------------------------------------------------------------------
    print("   [4/4] Serializando a formato EAV...")
    
    # Mapeo CASILLAS AEAT
    eav_rows = []
    
    # Función auxiliar vectorizada es dificil para melt masivo, 
    # mejor exportamos CSV directo de Wide Format y adaptamos el script de carga
    # para leer Wide Format si detecta el archivo 'optimized', 
    # PERO para mantener compatibilidad 100% con `run_full_analysis` que espera EAV:
    
    # Vamos a guardar df_empresas como 'data_empresas_500k.csv'
    # y df_m347 como 'data_m347_500k.csv'
    
    # Para EAV, seleccionamos solo un subset representativo o generamos un EAV "dummy" 
    # si el script de análisis puede leer formato wide.
    # Revisando `run_full_analysis.py`, este hace Pivot del EAV.
    # Podemos hackear `run_full_analysis` para que acepte formato Wide directo si existe.
    
    return df_empresas, df_m347

if __name__ == "__main__":
    df_emp, df_rel = generate_massive_dataset(500000)
    
    print("💾 Guardando CSVs...")
    df_emp.to_csv('data_empresas_500k.csv', index=False)
    df_rel.to_csv('data_m347_500k.csv', index=False)
    
    # Generar fake EAV solo para que no falle la carga si se requiere, 
    # pero modificaremos run_analysis para usar data_empresas directo si queremos velocidad.
    # O mejor, generamos el EAV real via melt, pandas es eficiente.
    
    print("🔄 Generando EAV (Melt)...")
    # Mapping var -> casilla
    cols_to_melt = {
        'cifra_negocio': 'C00255',
        'resultado': 'C00500',
        'activo': 'C00033',
        'caja': 'C00032',
        'deuda_bancaria': 'C00195',
        'gastos_financieros': 'C00296',
        'gastos_transporte': 'C00279',
        'exportaciones_ue': 'EIC_UE'
    }
    
    # Melt
    df_melt = df_emp.rename(columns=cols_to_melt)
    df_eav = df_melt.melt(
        id_vars=['nif'], 
        value_vars=list(cols_to_melt.values()),
        var_name='casilla', 
        value_name='valor'
    )
    df_eav.to_csv('data_eav_500k.csv', index=False)
    
    print("✅ DATASET GENERADO EXITOSAMENTE")
    print(f"   Empresas: {len(df_emp):,}")
    print(f"   EAV Rows: {len(df_eav):,}")
    print(f"   M347 Rows: {len(df_rel):,}")
