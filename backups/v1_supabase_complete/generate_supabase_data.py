#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
GENERADOR DE DATOS TRIBUTARIOS SIMULADOS PARA SUPABASE
================================================================================
Genera y sube datos para los modelos M200, M347 y M349 directamente a Supabase.

Volúmenes:
- M200: 5.000 empresas únicas
- M347: 50.000 operaciones con terceros
- M349: 20.000 operaciones intracomunitarias

Autor: FraudHunter AI System
================================================================================
"""

import os
import random
import string
from datetime import datetime
from typing import List, Dict, Any
import time

# Supabase client
from supabase import create_client, Client

# ==============================================================================
# CONFIGURACIÓN DE SUPABASE
# ==============================================================================

SUPABASE_URL = "https://xqtcvtchzmuecssmwrxb.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhxdGN2dGNoem11ZWNzc213cnhiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjkwMDg1MzYsImV4cCI6MjA4NDU4NDUzNn0.djoygcwE46xBVTH2kt84i1YoUCtHn1VrNQ5M9FUDzPI")

# Volúmenes objetivo
N_EMPRESAS = 5_000
N_M347 = 50_000
N_M349 = 20_000
BATCH_SIZE = 500  # Registros por lote para inserción

# ==============================================================================
# DATOS DE REFERENCIA PARA SIMULACIÓN REALISTA
# ==============================================================================

# CNAEs representativos por sector
CNAES = {
    "retail": ["4711", "4719", "4721", "4751", "4759"],
    "construccion": ["4110", "4120", "4211", "4221", "4299"],
    "servicios": ["6201", "6202", "6311", "6920", "7010"],
    "industria": ["1011", "1020", "2511", "2562", "2899"],
    "transporte": ["4941", "4942", "5210", "5229", "5320"],
    "hosteleria": ["5510", "5520", "5610", "5621", "5630"],
    "inmobiliario": ["6810", "6820", "6831", "6832"],
    "financiero": ["6419", "6492", "6499", "6612"],
}

# Provincias españolas (código + nombre)
PROVINCIAS = [
    "01-Álava", "02-Albacete", "03-Alicante", "04-Almería", "05-Ávila",
    "06-Badajoz", "07-Baleares", "08-Barcelona", "09-Burgos", "10-Cáceres",
    "11-Cádiz", "12-Castellón", "13-Ciudad Real", "14-Córdoba", "15-A Coruña",
    "16-Cuenca", "17-Girona", "18-Granada", "19-Guadalajara", "20-Guipúzcoa",
    "21-Huelva", "22-Huesca", "23-Jaén", "24-León", "25-Lleida",
    "26-La Rioja", "27-Lugo", "28-Madrid", "29-Málaga", "30-Murcia",
    "31-Navarra", "32-Ourense", "33-Asturias", "34-Palencia", "35-Las Palmas",
    "36-Pontevedra", "37-Salamanca", "38-S.C. Tenerife", "39-Cantabria", "40-Segovia",
    "41-Sevilla", "42-Soria", "43-Tarragona", "44-Teruel", "45-Toledo",
    "46-Valencia", "47-Valladolid", "48-Vizcaya", "49-Zamora", "50-Zaragoza",
]

# Países UE para M349
PAISES_UE = ["DE", "FR", "IT", "PT", "NL", "BE", "AT", "PL", "CZ", "RO", "BG", "HU", "SE", "DK", "FI", "IE", "GR", "SK", "HR", "SI", "EE", "LV", "LT", "CY", "LU", "MT"]
PAISES_RIESGO = ["CY", "LU", "MT", "NL", "IE", "BG", "RO"]  # Mayor probabilidad de fraude carrusel

# Claves de operación
CLAVES_M347 = ["A", "B", "C", "D", "E", "F", "G", "I"]
CLAVES_M349 = ["E", "M", "H", "T", "A", "S", "I"]

# Nombres de empresas (prefijos y sufijos)
PREFIJOS_EMPRESA = ["Grupo", "Corporación", "Inversiones", "Servicios", "Comercial", "Industrial", "Logistics", "Tech", "Digital", "Smart", "Global", "Euro", "Iberia", "Norte", "Sur", "Centro", "Capital", "Premium", "Pro", "Max"]
SUFIJOS_EMPRESA = ["S.L.", "S.A.", "S.L.U.", "S.A.U.", "S.Coop.", "AIE"]


# ==============================================================================
# FUNCIONES DE GENERACIÓN
# ==============================================================================

def generate_nif() -> str:
    """Genera un NIF español válido (formato: letra + 8 dígitos)."""
    letters = "ABCDEFGHJKLMNPQRSUVW"
    letter = random.choice(letters)
    digits = "".join(random.choices(string.digits, k=8))
    return f"{letter}{digits}"


def generate_nif_ue(pais: str) -> str:
    """Genera un NIF intracomunitario para un país UE."""
    if pais == "DE":
        return f"DE{random.randint(100000000, 999999999)}"
    elif pais == "FR":
        return f"FR{random.choice(string.ascii_uppercase)}{random.choice(string.ascii_uppercase)}{random.randint(100000000, 999999999)}"
    elif pais in ["IT", "PT", "PL", "RO", "BG"]:
        return f"{pais}{random.randint(10000000000, 99999999999)}"
    else:
        return f"{pais}{random.randint(100000000, 999999999)}"


def generate_empresa_name() -> str:
    """Genera un nombre de empresa realista."""
    prefijo = random.choice(PREFIJOS_EMPRESA)
    # Añadir un identificador único
    id_part = "".join(random.choices(string.ascii_uppercase, k=2)) + str(random.randint(10, 99))
    sufijo = random.choice(SUFIJOS_EMPRESA)
    return f"{prefijo} {id_part} {sufijo}"


def generate_m200_record(nif: str, ejercicio: int, size_category: str) -> Dict[str, Any]:
    """
    Genera un registro M200 (Impuesto Sociedades) realista.
    size_category: 'micro', 'pyme', 'grande'
    """
    # Parámetros según tamaño
    if size_category == "micro":
        cifra_base = random.uniform(50_000, 500_000)
        margen = random.uniform(0.02, 0.15)
    elif size_category == "pyme":
        cifra_base = random.uniform(500_000, 10_000_000)
        margen = random.uniform(0.03, 0.12)
    else:  # grande
        cifra_base = random.uniform(10_000_000, 500_000_000)
        margen = random.uniform(0.04, 0.10)

    cifra_negocios = round(cifra_base, 2)
    aprovisionamientos = round(cifra_negocios * random.uniform(0.40, 0.70), 2)
    otros_ingresos = round(cifra_negocios * random.uniform(0, 0.05), 2)
    gastos_personal = round(cifra_negocios * random.uniform(0.10, 0.35), 2)
    otros_gastos_expl = round(cifra_negocios * random.uniform(0.05, 0.15), 2)
    amortizaciones = round(cifra_negocios * random.uniform(0.02, 0.08), 2)
    
    # Calcular resultado antes de gastos financieros
    ebitda = cifra_negocios + otros_ingresos - aprovisionamientos - gastos_personal - otros_gastos_expl
    ebit = ebitda - amortizaciones
    
    # Gastos financieros proporcionales a deuda
    deuda_bancaria_lp = round(cifra_negocios * random.uniform(0.05, 0.40), 2)
    deuda_bancaria_cp = round(cifra_negocios * random.uniform(0.02, 0.15), 2)
    gastos_financieros = round((deuda_bancaria_lp + deuda_bancaria_cp) * random.uniform(0.03, 0.08), 2)
    
    resultado_ejercicio = round(ebit - gastos_financieros, 2)
    
    # Balance
    total_activo = round(cifra_negocios * random.uniform(0.8, 2.5), 2)
    patrimonio_neto = round(total_activo * random.uniform(0.20, 0.60), 2)
    efectivo_tesoreria = round(total_activo * random.uniform(0.02, 0.15), 2)

    # Seleccionar CNAE del sector
    sector = random.choice(list(CNAES.keys()))
    cnae = random.choice(CNAES[sector])

    return {
        "nif": nif,
        "ejercicio": ejercicio,
        "cnae": cnae,
        "nombre_empresa": generate_empresa_name(),
        "cifra_negocios": cifra_negocios,
        "aprovisionamientos": aprovisionamientos,
        "otros_ingresos": otros_ingresos,
        "gastos_personal": gastos_personal,
        "otros_gastos_expl": otros_gastos_expl,
        "amortizaciones": amortizaciones,
        "gastos_financieros": gastos_financieros,
        "resultado_ejercicio": resultado_ejercicio,
        "total_activo": total_activo,
        "patrimonio_neto": patrimonio_neto,
        "deuda_bancaria_lp": deuda_bancaria_lp,
        "deuda_bancaria_cp": deuda_bancaria_cp,
        "efectivo_tesoreria": efectivo_tesoreria,
    }


def generate_m347_record(nif_declarante: str, nif_contraparte: str, ejercicio: int) -> Dict[str, Any]:
    """Genera un registro M347 (Operaciones con terceros)."""
    clave = random.choices(CLAVES_M347, weights=[30, 35, 5, 3, 2, 5, 5, 15])[0]  # B y A más comunes
    
    # Importe total (mínimo umbral declaración: 3005.06€)
    importe_total = round(random.uniform(3100, 500_000), 2)
    
    # Distribuir por trimestres
    pesos_trim = [random.random() for _ in range(4)]
    suma_pesos = sum(pesos_trim)
    importes_trim = [round(importe_total * p / suma_pesos, 2) for p in pesos_trim]
    
    # Ajustar para que sumen exactamente importe_total
    diff = importe_total - sum(importes_trim)
    importes_trim[0] += diff
    
    # Metálico solo en algunos casos
    importe_metalico = round(random.uniform(6000, 20000), 2) if random.random() < 0.05 else 0
    
    # Inmuebles/arrendamiento solo para ciertas claves
    importe_inmuebles = round(random.uniform(50_000, 500_000), 2) if clave in ["B", "A"] and random.random() < 0.02 else 0
    importe_arrendamiento = round(random.uniform(5_000, 50_000), 2) if clave in ["B"] and random.random() < 0.08 else 0

    return {
        "nif_declarante": nif_declarante,
        "nif_contraparte": nif_contraparte,
        "nombre_contraparte": generate_empresa_name(),
        "ejercicio": ejercicio,
        "importe_total": importe_total,
        "importe_metalico": importe_metalico,
        "importe_transmision_inmuebles": importe_inmuebles,
        "importe_arrendamiento": importe_arrendamiento,
        "importe_1t": importes_trim[0],
        "importe_2t": importes_trim[1],
        "importe_3t": importes_trim[2],
        "importe_4t": importes_trim[3],
        "clave_operacion": clave,
        "provincia": random.choice(PROVINCIAS).split("-")[0],
        "pais": "ES",
    }


def generate_m349_record(nif_declarante: str, ejercicio: int) -> Dict[str, Any]:
    """Genera un registro M349 (Operaciones intracomunitarias)."""
    # Mayor probabilidad para países de riesgo (simular fraude)
    if random.random() < 0.15:
        pais_ue = random.choice(PAISES_RIESGO)
    else:
        pais_ue = random.choice(PAISES_UE)
    
    clave = random.choices(CLAVES_M349, weights=[40, 5, 3, 7, 25, 12, 8])[0]
    
    importe = round(random.uniform(5_000, 2_000_000), 2)
    base_rectificada = round(importe * random.uniform(-0.05, 0.05), 2) if random.random() < 0.03 else 0
    
    # Periodo: mensual o trimestral
    periodo = random.choice(["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "1T", "2T", "3T", "4T"])

    return {
        "nif_declarante": nif_declarante,
        "nif_operador_ue": generate_nif_ue(pais_ue),
        "nombre_operador": f"Company {pais_ue} " + "".join(random.choices(string.ascii_uppercase, k=4)),
        "pais_ue": pais_ue,
        "ejercicio": ejercicio,
        "periodo": periodo,
        "importe": importe,
        "base_rectificada": base_rectificada,
        "clave_operacion": clave,
    }


# ==============================================================================
# FUNCIONES DE SUBIDA A SUPABASE
# ==============================================================================

def upload_batch(supabase: Client, table: str, records: List[Dict], batch_num: int, total_batches: int) -> bool:
    """Sube un lote de registros a Supabase."""
    try:
        supabase.table(table).insert(records).execute()
        print(f"  ✓ Batch {batch_num}/{total_batches} subido ({len(records)} registros)")
        return True
    except Exception as e:
        print(f"  ✗ Error en batch {batch_num}: {e}")
        return False


def generate_and_upload_m200(supabase: Client, nifs: List[str], ejercicio: int = 2024):
    """Genera y sube todos los registros M200."""
    print(f"\n{'='*60}")
    print(f"📊 GENERANDO M200 - {len(nifs)} empresas")
    print(f"{'='*60}")
    
    records = []
    size_weights = [0.70, 0.25, 0.05]  # 70% micro, 25% pyme, 5% grande
    sizes = ["micro", "pyme", "grande"]
    
    for i, nif in enumerate(nifs):
        size = random.choices(sizes, weights=size_weights)[0]
        record = generate_m200_record(nif, ejercicio, size)
        records.append(record)
        
        if (i + 1) % 1000 == 0:
            print(f"  Generados {i + 1}/{len(nifs)} registros...")
    
    # Subir en lotes
    total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n🚀 Subiendo {len(records)} registros en {total_batches} lotes...")
    
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        upload_batch(supabase, "modelo_200", batch, batch_num, total_batches)
        time.sleep(0.1)  # Rate limiting suave
    
    print(f"✅ M200 completado: {len(records)} empresas")


def generate_and_upload_m347(supabase: Client, nifs: List[str], ejercicio: int = 2024):
    """Genera y sube todos los registros M347."""
    print(f"\n{'='*60}")
    print(f"📊 GENERANDO M347 - {N_M347} operaciones")
    print(f"{'='*60}")
    
    records = []
    for i in range(N_M347):
        nif_declarante = random.choice(nifs)
        nif_contraparte = random.choice(nifs)
        # Evitar auto-operaciones
        while nif_contraparte == nif_declarante:
            nif_contraparte = random.choice(nifs)
        
        record = generate_m347_record(nif_declarante, nif_contraparte, ejercicio)
        records.append(record)
        
        if (i + 1) % 5000 == 0:
            print(f"  Generados {i + 1}/{N_M347} registros...")
    
    # Subir en lotes
    total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n🚀 Subiendo {len(records)} registros en {total_batches} lotes...")
    
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        upload_batch(supabase, "modelo_347", batch, batch_num, total_batches)
        time.sleep(0.1)
    
    print(f"✅ M347 completado: {len(records)} operaciones")


def generate_and_upload_m349(supabase: Client, nifs: List[str], ejercicio: int = 2024):
    """Genera y sube todos los registros M349."""
    print(f"\n{'='*60}")
    print(f"📊 GENERANDO M349 - {N_M349} operaciones")
    print(f"{'='*60}")
    
    # Solo un subconjunto de empresas tienen operaciones UE
    nifs_ue = random.sample(nifs, min(len(nifs) // 3, N_M349 // 5))
    
    records = []
    for i in range(N_M349):
        nif_declarante = random.choice(nifs_ue)
        record = generate_m349_record(nif_declarante, ejercicio)
        records.append(record)
        
        if (i + 1) % 2000 == 0:
            print(f"  Generados {i + 1}/{N_M349} registros...")
    
    # Subir en lotes
    total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n🚀 Subiendo {len(records)} registros en {total_batches} lotes...")
    
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        upload_batch(supabase, "modelo_349", batch, batch_num, total_batches)
        time.sleep(0.1)
    
    print(f"✅ M349 completado: {len(records)} operaciones")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║     🏛️ GENERADOR DE DATOS TRIBUTARIOS PARA SUPABASE 🏛️          ║
╠══════════════════════════════════════════════════════════════════╣
║  M200: 5.000 empresas (Balance + PyG)                            ║
║  M347: 50.000 operaciones con terceros                           ║
║  M349: 20.000 operaciones intracomunitarias                      ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Conectar a Supabase
    print("🔌 Conectando a Supabase...")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Conexión establecida")
    
    # Generar NIFs únicos para todas las empresas
    print(f"\n🏢 Generando {N_EMPRESAS} NIFs únicos...")
    nifs = list(set(generate_nif() for _ in range(N_EMPRESAS + 1000)))[:N_EMPRESAS]
    print(f"✅ {len(nifs)} NIFs generados")
    
    start_time = time.time()
    
    # Generar y subir datos
    generate_and_upload_m200(supabase, nifs, ejercicio=2024)
    generate_and_upload_m347(supabase, nifs, ejercicio=2024)
    generate_and_upload_m349(supabase, nifs, ejercicio=2024)
    
    elapsed = time.time() - start_time
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║     ✅ PROCESO COMPLETADO                                        ║
╠══════════════════════════════════════════════════════════════════╣
║  M200: {N_EMPRESAS:,} empresas                                          ║
║  M347: {N_M347:,} operaciones                                        ║
║  M349: {N_M349:,} operaciones                                        ║
║  Tiempo total: {elapsed:.1f} segundos                                    ║
╚══════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
