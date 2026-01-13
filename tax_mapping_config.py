# ==============================================================================
# CONFIGURACIÓN MAESTRA DE MAPEO DE MODELOS TRIBUTARIOS (AEAT ESPAÑA)
# ==============================================================================
"""
Este módulo centraliza el mapeo entre variables de negocio y Casillas Oficiales AEAT.
Úsalo en todos los módulos de FraudHunter para garantizar consistencia.

Author: FraudHunter Team
Version: 1.0.0
"""

from typing import List, Optional, Dict

# ==============================================================================
# CONFIGURACIÓN PRINCIPAL
# ==============================================================================

TAX_MAPPING_CONFIG = {
    # --- MODELO 200 (IMPUESTO DE SOCIEDADES) ---
    "M200": {
        # CUENTA DE PÉRDIDAS Y GANANCIAS
        "cifra_negocios":       ["00255", "C00255"],  # Ingresos de Explotación
        "aprovisionamientos":   ["00258", "C00258"],  # Compras
        "otros_ingresos":       ["00260", "C00260"],  # Subvenciones
        "gastos_personal":      ["00263", "C00263"],  # Sueldos + Seg. Social
        "otros_gastos_expl":    ["00279", "C00279"],  # Proxy para Transportes/Servicios Ext.
        "amortizaciones":       ["00280", "C00280"], 
        "gastos_financieros":   ["00296", "C00296"],  # Clave para deuda oculta
        "resultado_ejercicio":  ["00500", "C00500"],  # Beneficio Neto

        # BALANCE DE SITUACIÓN
        "total_activo":         ["00033", "C00033"],
        "patrimonio_neto":      ["00062", "C00062"],
        "deuda_bancaria_lp":    ["00195", "C00195"],  # Deudas entidades crédito L/P
        "deuda_bancaria_cp":    ["00215", "C00215"],  # Deudas entidades crédito C/P
        "efectivo_tesoreria":   ["00032", "C00032"],  # Caja y equivalentes
        
        # EXTRAS ESPECÍFICOS
        "trabajos_realizados_activo": ["00257", "C00257"]  # Alerta de capitalización de gastos
    },

    # --- MODELO 347 (OPERACIONES CON TERCEROS - ANUAL) ---
    # Declaración informativa de operaciones con terceras personas > 3.005,06€
    "M347": {
        # =====================================================================
        # CAMPOS ESTRUCTURALES (Identificación)
        # =====================================================================
        "id_declarante": "NIF_DECLARANTE",
        "id_contraparte": "NIF_CONTRAPARTE",          # NIF del cliente/proveedor
        "nombre_contraparte": "NOMBRE_RAZON_SOCIAL",  # Nombre del tercero
        "provincia": "CODIGO_PROVINCIA",              # 2 primeros dígitos CP
        "pais": "CODIGO_PAIS",                        # Código ISO país
        "ejercicio": "EJERCICIO",
        
        # =====================================================================
        # IMPORTES (Valores monetarios - incluyen IVA)
        # =====================================================================
        "importe_anual": "IMPORTE_OPERACIONES",       # Importe total anual
        "importe_metalico": "IMPORTE_METALICO",       # Pagos en efectivo >6.000€
        "importe_transmision_inmuebles": "IMPORTE_INMUEBLES",  # Operaciones inmobiliarias
        "importe_arrendamiento": "IMPORTE_ARRENDAMIENTO",      # Alquileres locales
        
        # Importes trimestrales (desglose)
        "importe_1t": "IMPORTE_1T",
        "importe_2t": "IMPORTE_2T", 
        "importe_3t": "IMPORTE_3T",
        "importe_4t": "IMPORTE_4T",
        
        # =====================================================================
        # CLAVES DE OPERACIÓN (Tipo de transacción)
        # =====================================================================
        "clave_operacion": "CLAVE_OPERACION",
        
        # Catálogo completo de claves AEAT
        "claves": {
            "A": "Compras/Adquisiciones de bienes y servicios",
            "B": "Ventas/Entregas de bienes y servicios",
            "C": "Cobros por cuenta de terceros (>300,51€)",
            "D": "Adquisiciones Entidades Públicas al margen actividad",
            "E": "Subvenciones y ayudas (emisor: Administración Pública)",
            "F": "Ventas agencias de viaje en nombre propio",
            "G": "Compras agencias de viaje en nombre propio",
            "I": "Operaciones con inversión del sujeto pasivo (ISP)",
        },
        
        # =====================================================================
        # FILTROS PARA ETL (Agrupación de operaciones)
        # =====================================================================
        "filtro_ventas": ["B", "F"],               # Claves que suman a Ventas
        "filtro_compras": ["A", "G"],              # Claves que suman a Compras
        "filtro_cobros_terceros": ["C"],           # Intermediación
        "filtro_inversion_iva": ["I"],             # Riesgo Carrusel IVA
        "filtro_subvenciones": ["E"],              # Ayudas públicas
        "filtro_sector_publico": ["D"],            # Entidades públicas
        
        # =====================================================================
        # DATOS INMUEBLES (Arrendamientos)
        # =====================================================================
        "inmueble_referencia_catastral": "REF_CATASTRAL",
        "inmueble_direccion": "DIRECCION_INMUEBLE",
        "inmueble_tipo_via": "TIPO_VIA",
        "inmueble_cp": "CODIGO_POSTAL",
        
        # =====================================================================
        # FLAGS FORENSES (Indicadores de riesgo)
        # =====================================================================
        "umbral_declaracion": 3005.06,             # Umbral legal declaración
        "umbral_metalico": 6000.00,                # Umbral pago efectivo
        "umbral_cobros_terceros": 300.51,          # Umbral cobros cuenta ajena
    },

    # --- MODELO 349 (OPERACIONES INTRACOMUNITARIAS) ---
    # Declaración recapitulativa de operaciones con operadores UE
    "M349": {
        # =====================================================================
        # CAMPOS ESTRUCTURALES (Identificación)
        # =====================================================================
        "id_declarante": "NIF_DECLARANTE",
        "id_operador_ue": "NIF_OPERADOR_UE",       # NIF-IVA intracomunitario
        "nombre_operador": "NOMBRE_RAZON_SOCIAL",
        "codigo_pais": "CODIGO_PAIS_UE",           # Código ISO país UE (DE, FR, IT...)
        "ejercicio": "EJERCICIO",
        "periodo": "PERIODO",                      # Mes (01-12) o Trimestre (1T-4T)
        
        # =====================================================================
        # IMPORTES (Base imponible - SIN IVA)
        # =====================================================================
        "base_imponible": "BASE_IMPONIBLE",        # Importe operación sin IVA
        "base_rectificada": "BASE_RECTIFICADA",    # Rectificaciones declaraciones anteriores
        
        # =====================================================================
        # CLAVES DE OPERACIÓN (Tipo de transacción intracomunitaria)
        # =====================================================================
        "clave_operacion": "CLAVE_OPERACION",
        
        # Catálogo completo de claves AEAT para M349
        "claves": {
            # Entregas (Exportaciones a UE)
            "E": "Entregas intracomunitarias de bienes",
            "M": "Entregas ICom posteriores a importación exenta",
            "H": "Entregas ICom por representante fiscal tras importación exenta",
            "T": "Ventas triangulares (venta posterior a adquisición exenta)",
            
            # Adquisiciones (Importaciones de UE)
            "A": "Adquisiciones intracomunitarias de bienes",
            
            # Servicios
            "S": "Prestaciones de servicios intracomunitarios",
            "I": "Adquisiciones intracomunitarias de servicios",
        },
        
        # =====================================================================
        # FILTROS PARA ETL (Agrupación de operaciones)
        # =====================================================================
        "filtro_entregas_bienes": ["E", "M", "H", "T"],  # Todas las entregas/ventas UE
        "filtro_adquisiciones_bienes": ["A"],            # Compras bienes UE
        "filtro_servicios_prestados": ["S"],             # Servicios vendidos a UE
        "filtro_servicios_recibidos": ["I"],             # Servicios comprados de UE
        
        # Agregaciones para ABT
        "filtro_exportaciones_ue": ["E", "M", "H", "T", "S"],  # Todo lo vendido a UE
        "filtro_importaciones_ue": ["A", "I"],                  # Todo lo comprado de UE
        
        # =====================================================================
        # FLAGS FORENSES (Indicadores de riesgo)
        # =====================================================================
        # Operaciones triangulares (T) son riesgo alto de carrusel IVA
        "filtro_riesgo_carrusel": ["T", "M", "H"],
        
        # Países con mayor riesgo de fraude carrusel (histórico)
        "paises_riesgo_alto": ["CY", "LU", "MT", "NL", "IE", "BG", "RO"],
    }
}


# ==============================================================================
# FUNCIONES HELPER
# ==============================================================================

def get_casillas(modelo: str, variable: str) -> List[str]:
    """
    Obtiene la lista de casillas posibles para una variable.
    
    Args:
        modelo: Identificador del modelo ('M200', 'M347', 'M349')
        variable: Nombre de la variable de negocio
        
    Returns:
        Lista de casillas válidas (ej: ['00255', 'C00255'])
        
    Example:
        >>> get_casillas('M200', 'cifra_negocios')
        ['00255', 'C00255']
    """
    if modelo not in TAX_MAPPING_CONFIG:
        raise ValueError(f"Modelo desconocido: {modelo}")
    
    modelo_config = TAX_MAPPING_CONFIG[modelo]
    if variable not in modelo_config:
        raise ValueError(f"Variable '{variable}' no encontrada en {modelo}")
    
    return modelo_config[variable]


def get_primary_casilla(modelo: str, variable: str) -> str:
    """Obtiene la casilla principal (sin prefijo C) para una variable."""
    casillas = get_casillas(modelo, variable)
    # Preferir la versión sin prefijo
    for c in casillas:
        if not c.startswith('C'):
            return c
    return casillas[0]


def get_prefixed_casilla(modelo: str, variable: str) -> str:
    """Obtiene la casilla con prefijo C para una variable."""
    casillas = get_casillas(modelo, variable)
    for c in casillas:
        if c.startswith('C'):
            return c
    return f"C{casillas[0]}"


def get_variable_name(casilla: str, modelo: str = "M200") -> Optional[str]:
    """
    Obtiene el nombre de variable para una casilla dada.
    
    Args:
        casilla: Código de casilla (ej: 'C00255' o '00255')
        modelo: Modelo a buscar
        
    Returns:
        Nombre de la variable o None si no se encuentra
    """
    if modelo not in TAX_MAPPING_CONFIG:
        return None
    
    # Normalizar casilla (quitar prefijo C si existe)
    casilla_norm = casilla.lstrip('C') if casilla.startswith('C') else casilla
    
    for var_name, casillas in TAX_MAPPING_CONFIG[modelo].items():
        if isinstance(casillas, list):
            casillas_norm = [c.lstrip('C') if c.startswith('C') else c for c in casillas]
            if casilla_norm in casillas_norm:
                return var_name
    
    return None


def build_pivot_rename_map() -> Dict[str, str]:
    """
    Construye un diccionario para renombrar columnas después del pivot EAV.
    
    Maneja ambos formatos de casilla (con y sin prefijo C).
    
    Returns:
        Dict mapeando casilla -> nombre_variable
    """
    rename_map = {}
    
    for var_name, casillas in TAX_MAPPING_CONFIG["M200"].items():
        if isinstance(casillas, list):
            for casilla in casillas:
                rename_map[casilla] = var_name
    
    # Agregar mapeo para M349 (exportaciones/importaciones UE)
    # Estas se usan como casillas especiales en el formato EAV
    rename_map["ENTREGAS_UE"] = "entregas_ue"
    rename_map["ADQUISICIONES_UE"] = "adquisiciones_ue"
    
    return rename_map


def get_all_m200_casillas() -> List[str]:
    """Obtiene todas las casillas del M200 (formato con prefijo C)."""
    casillas = []
    for var_name, vals in TAX_MAPPING_CONFIG["M200"].items():
        if isinstance(vals, list):
            casillas.extend([c for c in vals if c.startswith('C')])
    return casillas


# ==============================================================================
# CONSTANTES DERIVADAS
# ==============================================================================

# Casillas críticas para detección de fraude
CASILLAS_FRAUDE_CARRUSEL = [
    get_prefixed_casilla("M200", "otros_gastos_expl"),  # Gastos transporte
    get_prefixed_casilla("M200", "cifra_negocios"),
]

CASILLAS_FRAUDE_MAQUILLAJE = [
    get_prefixed_casilla("M200", "resultado_ejercicio"),
    get_prefixed_casilla("M200", "efectivo_tesoreria"),
    get_prefixed_casilla("M200", "gastos_financieros"),
    get_prefixed_casilla("M200", "deuda_bancaria_lp"),
    get_prefixed_casilla("M200", "deuda_bancaria_cp"),
]

CASILLAS_EMPRESA_PANTALLA = [
    get_prefixed_casilla("M200", "cifra_negocios"),
    get_prefixed_casilla("M200", "gastos_personal"),
]


# ==============================================================================
# MAPEO LEGACY (Para compatibilidad con código existente)
# ==============================================================================
# Este mapeo permite una migración gradual sin romper el código existente

LEGACY_CASILLAS_MAP = {
    'cifra_negocio': get_prefixed_casilla("M200", "cifra_negocios"),
    'resultado': get_prefixed_casilla("M200", "resultado_ejercicio"),
    'activo': get_prefixed_casilla("M200", "total_activo"),
    'caja': get_prefixed_casilla("M200", "efectivo_tesoreria"),
    'deuda_bancaria': get_prefixed_casilla("M200", "deuda_bancaria_lp"),
    'gastos_financieros': get_prefixed_casilla("M200", "gastos_financieros"),
    'gastos_transporte': get_prefixed_casilla("M200", "otros_gastos_expl"),
    'entregas_ue': 'ENTREGAS_UE',  # El M349 se procesa diferente
    'gastos_personal': get_prefixed_casilla("M200", "gastos_personal"),
    'aprovisionamientos': get_prefixed_casilla("M200", "aprovisionamientos"),
    'amortizaciones': get_prefixed_casilla("M200", "amortizaciones"),
    'patrimonio_neto': get_prefixed_casilla("M200", "patrimonio_neto"),
}


if __name__ == "__main__":
    # Demo de uso
    print("=" * 60)
    print("TAX MAPPING CONFIG - AEAT España")
    print("=" * 60)
    
    print("\n[M200] Casillas principales:")
    for var in TAX_MAPPING_CONFIG["M200"]:
        casillas = TAX_MAPPING_CONFIG["M200"][var]
        if isinstance(casillas, list):
            print(f"  {var:25} -> {casillas}")
    
    print("\n[M347] Configuración:")
    for k, v in TAX_MAPPING_CONFIG["M347"].items():
        print(f"  {k:25} -> {v}")
    
    print("\n[M349] Operaciones UE:")
    for k, v in TAX_MAPPING_CONFIG["M349"].items():
        print(f"  {k:25} -> {v}")
    
    print("\n[Helper] Ejemplo get_casillas('M200', 'cifra_negocios'):")
    print(f"  {get_casillas('M200', 'cifra_negocios')}")
    
    print("\n[Helper] Pivot rename map (primeras 5):")
    rename_map = build_pivot_rename_map()
    for i, (k, v) in enumerate(rename_map.items()):
        if i >= 5:
            break
        print(f"  {k} -> {v}")
