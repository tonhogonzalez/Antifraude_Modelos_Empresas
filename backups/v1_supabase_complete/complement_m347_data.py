"""
Script para complementar data_m347.csv con datos realistas de relaciones empresariales.
Genera transacciones que crean un grafo visible con patrones de fraude detectables.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import random

# Configuración
np.random.seed(42)
random.seed(42)

# Cargar empresas existentes
base_path = Path(__file__).parent
df_empresas = pd.read_csv(base_path / 'data_empresas.csv')

print(f"📊 Empresas cargadas: {len(df_empresas)}")
print(f"   - SANA: {(df_empresas['tipo'] == 'SANA').sum()}")
print(f"   - FRAUDE_CARRUSEL: {(df_empresas['tipo'] == 'FRAUDE_CARRUSEL').sum()}")
print(f"   - FRAUDE_MAQUILLAJE: {(df_empresas['tipo'] == 'FRAUDE_MAQUILLAJE').sum()}")

# Separar empresas por tipo
empresas_sanas = df_empresas[df_empresas['tipo'] == 'SANA']['nif'].tolist()
empresas_carrusel = df_empresas[df_empresas['tipo'] == 'FRAUDE_CARRUSEL']['nif'].tolist()
empresas_maquillaje = df_empresas[df_empresas['tipo'] == 'FRAUDE_MAQUILLAJE']['nif'].tolist()
todas_empresas = df_empresas['nif'].tolist()

# Mapeo NIF -> Nombre para razon_social
nif_to_nombre = dict(zip(df_empresas['nif'], df_empresas['nombre']))

# Lista para almacenar nuevas transacciones
nuevas_transacciones = []

# =============================================================================
# 1. GENERAR TRANSACCIONES CIRCULARES (FRAUDE CARRUSEL)
# =============================================================================
print("\n🔄 Generando patrones circulares (Fraude Carrusel)...")

# Crear clusters circulares de 3-5 empresas
num_clusters = min(10, len(empresas_carrusel) // 3)
random.shuffle(empresas_carrusel)

for cluster_id in range(num_clusters):
    # Tamaño del cluster: 3-5 empresas
    cluster_size = random.randint(3, min(5, len(empresas_carrusel) - cluster_id * 5))
    start_idx = cluster_id * 5
    cluster_nifs = empresas_carrusel[start_idx:start_idx + cluster_size]
    
    if len(cluster_nifs) < 3:
        continue
    
    # Importe base para el círculo (casi idéntico entre transacciones)
    importe_base = random.uniform(50000, 200000)
    
    # Crear el círculo: A→B→C→...→A
    for i in range(len(cluster_nifs)):
        nif_origen = cluster_nifs[i]
        nif_destino = cluster_nifs[(i + 1) % len(cluster_nifs)]
        
        # Variación pequeña (<5%) para simular el "carrusel"
        importe = importe_base * random.uniform(0.97, 1.03)
        
        nuevas_transacciones.append({
            'nif_declarante': nif_origen,
            'nif_contraparte': nif_destino,
            'importe': round(importe, 2),
            'is_circular': True
        })
    
    # Añadir algunas transacciones adicionales dentro del cluster
    for _ in range(random.randint(2, 5)):
        nif_a = random.choice(cluster_nifs)
        nif_b = random.choice(cluster_nifs)
        if nif_a != nif_b:
            nuevas_transacciones.append({
                'nif_declarante': nif_a,
                'nif_contraparte': nif_b,
                'importe': round(importe_base * random.uniform(0.8, 1.2), 2),
                'is_circular': True
            })

print(f"   ✅ {num_clusters} clusters circulares creados")

# =============================================================================
# 2. GENERAR TRANSACCIONES CONCENTRADAS (EMPRESAS PANTALLA)
# =============================================================================
print("\n🏭 Generando patrones de concentración (Empresas Pantalla)...")

for nif_pantalla in empresas_maquillaje[:30]:  # Limitar a 30 empresas
    # Una empresa pantalla tiene muchas operaciones con pocos proveedores
    num_proveedores = random.randint(1, 3)
    proveedores = random.sample(empresas_sanas, min(num_proveedores, len(empresas_sanas)))
    
    # Generar 10-20 operaciones concentradas
    for _ in range(random.randint(10, 20)):
        proveedor = random.choice(proveedores)
        
        # Importes redondos (señal de fraude)
        base = random.choice([10000, 25000, 50000, 100000])
        importe = base + random.choice([0, 500, 1000])
        
        nuevas_transacciones.append({
            'nif_declarante': nif_pantalla,
            'nif_contraparte': proveedor,
            'importe': float(importe),
            'is_circular': False
        })

print(f"   ✅ Transacciones concentradas para {min(30, len(empresas_maquillaje))} empresas")

# =============================================================================
# 3. GENERAR TRANSACCIONES NORMALES (EMPRESAS SANAS)
# =============================================================================
print("\n✅ Generando transacciones normales (Empresas Sanas)...")

# Muestrear 500 empresas sanas para no sobrecargar
muestra_sanas = random.sample(empresas_sanas, min(500, len(empresas_sanas)))

for nif_sana in muestra_sanas:
    # 3-8 operaciones por empresa
    num_ops = random.randint(3, 8)
    contrapartes = random.sample(todas_empresas, min(num_ops, len(todas_empresas)))
    
    for contraparte in contrapartes:
        if nif_sana != contraparte:
            # Importes log-normal (distribución realista)
            importe = np.random.lognormal(mean=10, sigma=1)  # ~22k promedio
            importe = max(3006, min(importe, 500000))  # Clamp entre 3006 y 500k
            
            nuevas_transacciones.append({
                'nif_declarante': nif_sana,
                'nif_contraparte': contraparte,
                'importe': round(importe, 2),
                'is_circular': False
            })

print(f"   ✅ Transacciones normales para {len(muestra_sanas)} empresas")

# =============================================================================
# 4. CREAR RED DE HUBS (EMPRESAS CENTRALES)
# =============================================================================
print("\n🌐 Creando hubs de conexión...")

# Seleccionar 5 empresas como "hubs" muy conectados
hubs = random.sample(todas_empresas, 5)

for hub in hubs:
    # Conectar con 20-50 empresas
    conexiones = random.sample(todas_empresas, random.randint(20, 50))
    for conexion in conexiones:
        if hub != conexion:
            # El hub actúa como proveedor o cliente
            if random.random() > 0.5:
                origen, destino = hub, conexion
            else:
                origen, destino = conexion, hub
            
            nuevas_transacciones.append({
                'nif_declarante': origen,
                'nif_contraparte': destino,
                'importe': round(random.uniform(10000, 150000), 2),
                'is_circular': False
            })

print(f"   ✅ {len(hubs)} hubs creados con múltiples conexiones")

# =============================================================================
# 5. CONSOLIDAR Y GUARDAR
# =============================================================================
print("\n💾 Consolidando datos...")

# Crear DataFrame con nuevas transacciones
df_nuevo = pd.DataFrame(nuevas_transacciones)

# Cargar M347 existente
df_m347_existente = pd.read_csv(base_path / 'data_m347.csv')
print(f"   - Transacciones existentes: {len(df_m347_existente)}")
print(f"   - Nuevas transacciones: {len(df_nuevo)}")

# Combinar (primero las nuevas para priorizarlas en el grafo)
df_combinado = pd.concat([df_nuevo, df_m347_existente], ignore_index=True)

# Eliminar duplicados exactos
df_combinado = df_combinado.drop_duplicates(subset=['nif_declarante', 'nif_contraparte', 'importe'])

# Guardar
output_path = base_path / 'data_m347.csv'
df_combinado.to_csv(output_path, index=False)

print(f"\n✅ Archivo guardado: {output_path}")
print(f"   - Total transacciones finales: {len(df_combinado)}")
print(f"   - Transacciones circulares: {df_combinado['is_circular'].sum()}")

# Estadísticas del grafo
nifs_unicos = set(df_combinado['nif_declarante'].unique()) | set(df_combinado['nif_contraparte'].unique())
nifs_del_dataset = set(todas_empresas)
cobertura = len(nifs_unicos & nifs_del_dataset) / len(nifs_del_dataset) * 100

print(f"\n📈 Estadísticas del Grafo:")
print(f"   - NIFs únicos en transacciones: {len(nifs_unicos)}")
print(f"   - Cobertura de empresas del dataset: {cobertura:.1f}%")
