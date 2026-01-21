# ==============================================================================
# CONFIGURACIÓN MAESTRA DE MAPEO DE MODELOS TRIBUTARIOS (AEAT ESPAÑA) - ADAPTADA
# Formato M200: columnas tipo `m200_p7_00255` (wide)
# Formato M347/M349: mapeo source->canonical para normalización
# ==============================================================================

from typing import List, Optional, Dict

# ------------------------------------------------------------------------------
# Helpers internos
# ------------------------------------------------------------------------------

def _mk_m200_col(page: int, code: str) -> str:
    """
    Construye el nombre de columna M200 en formato wide: m200_p{page}_{code}
    code debe venir ya con el padding que uses (ej: '00255', '00032', '0062').
    """
    return f"m200_p{page}_{code}"

def _variants(*cols: str) -> List[str]:
    """Devuelve lista de variantes de columna (para soportar padding distinto)."""
    return list(dict.fromkeys([c for c in cols if c]))


# ------------------------------------------------------------------------------
# CONFIGURACIÓN PRINCIPAL
# ------------------------------------------------------------------------------

TAX_MAPPING_CONFIG: Dict = {
    # --- MODELO 200 (IS) ---
    # En tu lake llega como tabla horizontal: columnas m200_pX_YYYYY
    "M200": {
        # =========================
        # PYG (en tu ejemplo: p7)
        # =========================
        "cifra_negocios":       _variants(_mk_m200_col(7, "00255")),
        "aprovisionamientos":   _variants(_mk_m200_col(7, "00258")),  # OJO: ajusta si tu fuente usa 00260
        "otros_ingresos":       _variants(_mk_m200_col(7, "00260")),  # OJO: ajusta si tu fuente usa 00265
        "gastos_personal":      _variants(_mk_m200_col(7, "00263")),
        "otros_gastos_expl":    _variants(_mk_m200_col(7, "00279")),
        "amortizaciones":       _variants(_mk_m200_col(7, "00280")),
        "gastos_financieros":   _variants(_mk_m200_col(7, "00296")),
        "resultado_ejercicio":  _variants(_mk_m200_col(7, "00500")),

        # =========================
        # BALANCE (según tus ejemplos)
        # =========================
        # total_activo: no me diste el ejemplo, pero en tu mapping original era 00033
        "total_activo":         _variants(_mk_m200_col(1, "00033"), _mk_m200_col(1, "0033")),
        # patrimonio_neto: en tu ejemplo aparece m200_p2_0062
        "patrimonio_neto":      _variants(_mk_m200_col(2, "0062"), _mk_m200_col(2, "00062")),
        # deuda bancaria LP/CP: tus códigos originales eran 00195/00215 (mantengo formato wide por coherencia)
        "deuda_bancaria_lp":    _variants(_mk_m200_col(2, "0195"), _mk_m200_col(2, "00195")),
        "deuda_bancaria_cp":    _variants(_mk_m200_col(2, "0215"), _mk_m200_col(2, "00215")),
        # efectivo: en tu ejemplo aparece m200_p1_00032
        "efectivo_tesoreria":   _variants(_mk_m200_col(1, "00032"), _mk_m200_col(1, "0032")),

        # =========================
        # EXTRAS
        # =========================
        "trabajos_realizados_activo": _variants(_mk_m200_col(7, "00257")),
        # Si existe CNAE en tu M200 wide, añádelo aquí como columna "directa"
        # "cnae": _variants("cnae", "m200_cnae", "m200_pX_?????"),
    },

    # --- MODELO 347 (Operaciones con terceros) ---
    # Corregido: define mapeo source->canonical para que el pipeline siempre use:
    # nif_declarante, nif_contraparte, ejercicio, importe_total, clave_operacion
    "M347": {
        "source_to_canonical": {
            # Identificación
            "NIF_DECLARANTE": "nif_declarante",
            "NIF_CONTRAPARTE": "nif_contraparte",
            "NOMBRE_RAZON_SOCIAL": "nombre_contraparte",
            "EJERCICIO": "ejercicio",

            # Importes
            "IMPORTE_OPERACIONES": "importe_total",
            "IMPORTE_METALICO": "importe_metalico",
            "IMPORTE_INMUEBLES": "importe_transmision_inmuebles",
            "IMPORTE_ARRENDAMIENTO": "importe_arrendamiento",
            "IMPORTE_1T": "importe_1t",
            "IMPORTE_2T": "importe_2t",
            "IMPORTE_3T": "importe_3t",
            "IMPORTE_4T": "importe_4t",

            # Clave
            "CLAVE_OPERACION": "clave_operacion",

            # Geo (opcionales)
            "CODIGO_PROVINCIA": "provincia",
            "CODIGO_PAIS": "pais",
        },

        # Catálogo + filtros (se mantienen como "config", NO como casillas)
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

        "filtros": {
            "filtro_ventas": ["B", "F"],
            "filtro_compras": ["A", "G"],
            "filtro_cobros_terceros": ["C"],
            "filtro_inversion_iva": ["I"],
            "filtro_subvenciones": ["E"],
            "filtro_sector_publico": ["D"],
        },

        "umbrales": {
            "umbral_declaracion": 3005.06,
            "umbral_metalico": 6000.00,
            "umbral_cobros_terceros": 300.51,
        },

        "inmuebles": {
            "REF_CATASTRAL": "inmueble_referencia_catastral",
            "DIRECCION_INMUEBLE": "inmueble_direccion",
            "TIPO_VIA": "inmueble_tipo_via",
            "CODIGO_POSTAL": "inmueble_cp",
        },
    },

    # --- MODELO 349 (Operaciones intracomunitarias) ---
    # Corregido: normaliza a nombres canónicos esperables por el pipeline:
    # nif_declarante, nif_operador_ue, pais_ue, ejercicio, periodo, importe (base imponible)
    "M349": {
        "source_to_canonical": {
            "NIF_DECLARANTE": "nif_declarante",
            "NIF_OPERADOR_UE": "nif_operador_ue",
            "CODIGO_PAIS_UE": "pais_ue",
            "EJERCICIO": "ejercicio",
            "PERIODO": "periodo",

            # Importes: el pipeline actual usa "importe"; lo normalizamos desde BASE_IMPONIBLE
            "BASE_IMPONIBLE": "importe",
            "BASE_RECTIFICADA": "base_rectificada",

            "CLAVE_OPERACION": "clave_operacion",
            "NOMBRE_RAZON_SOCIAL": "nombre_operador",
        },

        "claves": {
            "E": "Entregas intracomunitarias de bienes",
            "M": "Entregas ICom posteriores a importación exenta",
            "H": "Entregas ICom por representante fiscal tras importación exenta",
            "T": "Ventas triangulares (venta posterior a adquisición exenta)",
            "A": "Adquisiciones intracomunitarias de bienes",
            "S": "Prestaciones de servicios intracomunitarios",
            "I": "Adquisiciones intracomunitarias de servicios",
        },

        "filtros": {
            "filtro_entregas_bienes": ["E", "M", "H", "T"],
            "filtro_adquisiciones_bienes": ["A"],
            "filtro_servicios_prestados": ["S"],
            "filtro_servicios_recibidos": ["I"],
            "filtro_exportaciones_ue": ["E", "M", "H", "T", "S"],
            "filtro_importaciones_ue": ["A", "I"],
            "filtro_riesgo_carrusel": ["T", "M", "H"],
            "paises_riesgo_alto": ["CY", "LU", "MT", "NL", "IE", "BG", "RO"],
        },
    },
}

# ------------------------------------------------------------------------------
# Helpers públicos
# ------------------------------------------------------------------------------

def build_m200_column_rename_map() -> Dict[str, str]:
    """
    Devuelve dict {col_m200_wide: variable_negocio} para renombrar rápidamente.
    Soporta variantes de padding (ej: m200_p2_0062 y m200_p2_00062).
    """
    out: Dict[str, str] = {}
    for var_name, cols in TAX_MAPPING_CONFIG["M200"].items():
        if isinstance(cols, list):
            for c in cols:
                out[c] = var_name
    return out

def get_m200_columns() -> List[str]:
    """Lista completa de columnas M200 wide que se usarán."""
    cols: List[str] = []
    for v in TAX_MAPPING_CONFIG["M200"].values():
        if isinstance(v, list):
            cols.extend(v)
    return list(dict.fromkeys(cols))

def get_source_to_canonical(modelo: str) -> Dict[str, str]:
    """Devuelve el diccionario source->canonical para M347/M349."""
    cfg = TAX_MAPPING_CONFIG.get(modelo, {})
    return cfg.get("source_to_canonical", {})


# ------------------------------------------------------------------------------
# LEGACY: Compatibilidad hacia atrás con código existente
# ------------------------------------------------------------------------------

# Mapeo legacy de casillas AEAT para código que aún usa el formato antiguo
LEGACY_CASILLAS_MAP: Dict[str, str] = {
    # M200 - Impuesto Sociedades (casillas oficiales)
    "cifra_negocio": "C00255",
    "resultado": "C00500",
    "activo": "C00033",
    "caja": "C00032",
    "deuda_bancaria": "C00195",
    "gastos_financieros": "C00296",
    "gastos_transporte": "C00279",
    # M349 - Operaciones Intracomunitarias
    "entregas_ue": "EIC_UE",
    "adquisiciones_ue": "AIC_UE",
}


def get_prefixed_casilla(casilla_code: str, prefix: str = "C") -> str:
    """
    Legacy: Añade prefijo a código de casilla si no lo tiene.
    Ejemplo: get_prefixed_casilla("00255", "C") -> "C00255"
    """
    if casilla_code.startswith(prefix):
        return casilla_code
    return f"{prefix}{casilla_code}"


def build_pivot_rename_map() -> Dict[str, str]:
    """
    Legacy: Alias de build_m200_column_rename_map para compatibilidad.
    Devuelve dict {col_m200_wide: variable_negocio}.
    """
    return build_m200_column_rename_map()
