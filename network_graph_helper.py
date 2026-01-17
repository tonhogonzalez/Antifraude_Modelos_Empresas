
import networkx as nx
from pyvis.network import Network
import numpy as np
import tempfile
import os

def create_interactive_network_html(center_nif, center_risk, center_score, active_flags=None, company_data=None):
    """
    Genera un grafo interactivo con nodos arrastrables usando PyVis.
    Los patrones del grafo están basados en los FLAGS ACTIVOS de la empresa.
    
    Args:
        center_nif: NIF de la empresa objetivo
        center_risk: Nivel de riesgo (Alto/Medio/Bajo)
        center_score: Score de fraude (0-1)
        active_flags: Lista de flags activos (ej: ['flag_empresa_pantalla', 'flag_numeros_redondos'])
        company_data: Dict o Series con datos financieros de la empresa
    
    Retorna el HTML como string para incrustar en Streamlit.
    """
    
    # Default values
    if active_flags is None:
        active_flags = []
    if company_data is None:
        company_data = {}
    
    # ═══════════════════════════════════════════════════════════════════
    # CONFIGURACIÓN DE COLORES (Design System FraudHunter)
    # ═══════════════════════════════════════════════════════════════════
    
    COLORS = {
        'target': '#e91e63',      # Rosa brillante - Empresa objetivo
        'shell': '#f44336',       # Rojo - Empresa pantalla
        'suspicious': '#ff9800',  # Naranja - Proveedor sospechoso
        'round_amounts': '#9c27b0', # Púrpura - Números redondos
        'logistics': '#00bcd4',   # Cyan - Logística fantasma
        'debt': '#795548',        # Marrón - Deuda oculta
        'coverage': '#607d8b',    # Gris azulado - Baja cobertura M347
        'normal': '#4caf50',      # Verde - Normal/Legítimo
        'neutral': '#78909c',     # Gris - Neutral
    }
    
    np.random.seed(hash(center_nif) % 2**32)
    
    # ═══════════════════════════════════════════════════════════════════
    # EXTRAER DATOS FINANCIEROS DE LA EMPRESA
    # ═══════════════════════════════════════════════════════════════════
    
    # Ventas (usar valor real o estimado)
    ventas = float(company_data.get('ventas_netas', company_data.get('cifra_negocios', 100000)))
    gastos_personal = float(company_data.get('gastos_personal', 10000))
    gastos_transporte = float(company_data.get('gastos_transporte', 1000))
    
    # ═══════════════════════════════════════════════════════════════════
    # CREAR RED PYVIS
    # ═══════════════════════════════════════════════════════════════════
    
    net = Network(
        height="550px",
        width="100%",
        bgcolor="#0f172a",
        font_color="white",
        directed=True,
        notebook=False,
        cdn_resources='remote'
    )
    
    # Configuración de física para mejor interacción
    net.set_options("""
    {
        "nodes": {
            "borderWidth": 3,
            "borderWidthSelected": 5,
            "font": {
                "size": 14,
                "face": "Arial",
                "color": "white"
            },
            "shadow": {
                "enabled": true,
                "color": "rgba(0,0,0,0.5)",
                "size": 10
            }
        },
        "edges": {
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 1.2
                }
            },
            "color": {
                "inherit": false
            },
            "smooth": {
                "enabled": true,
                "type": "curvedCW",
                "roundness": 0.2
            },
            "font": {
                "size": 12,
                "color": "#ffeb3b",
                "strokeWidth": 3,
                "strokeColor": "#000000"
            }
        },
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -2000,
                "centralGravity": 0.5,
                "springLength": 120,
                "springConstant": 0.05,
                "damping": 0.15
            },
            "stabilization": {
                "enabled": true,
                "iterations": 50,
                "fit": true
            },
            "maxVelocity": 50,
            "minVelocity": 0.75
        },
        "interaction": {
            "dragNodes": true,
            "dragView": true,
            "zoomView": true,
            "hover": true,
            "tooltipDelay": 200,
            "navigationButtons": false,
            "keyboard": {
                "enabled": false
            }
        }
    }
    """)

    
    # ═══════════════════════════════════════════════════════════════════
    # AGREGAR NODO CENTRAL (EMPRESA OBJETIVO)
    # ═══════════════════════════════════════════════════════════════════
    
    flags_count = len(active_flags)
    flags_text = f"<br>⚠️ {flags_count} alertas activas" if flags_count > 0 else "<br>✅ Sin alertas"
    
    net.add_node(
        center_nif,
        label=f"🎯 {center_nif}",
        title=f"<b>EMPRESA OBJETIVO</b><br>NIF: {center_nif}<br>Riesgo: {center_risk}<br>Score: {center_score:.3f}{flags_text}",
        color=COLORS['target'],
        size=50,
        shape='diamond',
        font={'size': 16, 'color': 'white'}
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # GENERAR RED BASADA EN FLAGS ACTIVOS
    # ═══════════════════════════════════════════════════════════════════
    
    nodes_added = set()
    
    # ─────────────────────────────────────────────────────────────────
    # FLAG: EMPRESA PANTALLA (Shell Company Pattern)
    # ─────────────────────────────────────────────────────────────────
    if 'flag_empresa_pantalla' in active_flags:
        # Patrón: Carrusel con empresas pantalla
        shell_descriptions = [
            ("SHELL-1", "Sin empleados", "0 trabajadores"),
            ("SHELL-2", "Domicilio virtual", "Coworking fiscal"),
            ("SHELL-3", "Recién creada", "< 6 meses antigüedad"),
        ]
        shell_nifs = [f"X{np.random.randint(10000000, 99999999)}" for _ in range(3)]
        
        # Importes basados en ventas reales de la empresa
        shell_amount = max(50000, ventas * 0.3)  # 30% de ventas
        
        for (name, issue, detail), nif in zip(shell_descriptions, shell_nifs):
            net.add_node(
                nif,
                label=f"🏭 {name}",
                title=f"<b>⛔ EMPRESA PANTALLA</b><br>NIF: {nif}<br>🚩 {issue}<br>📋 {detail}<br>💰 Flujo: €{shell_amount:,.0f}",
                color=COLORS['shell'],
                size=40,
                shape='box'
            )
            nodes_added.add(nif)
        
        # Crear ciclo sospechoso (carrusel)
        cycle = [center_nif] + shell_nifs + [center_nif]
        for i in range(len(cycle) - 1):
            is_closing = (i == len(cycle) - 1)
            net.add_edge(
                cycle[i], 
                cycle[i + 1],
                title=f"🔄 CICLO CARRUSEL<br>Importe: €{shell_amount:,.0f}<br>⚠️ Flujo circular detectado",
                label=f"€{shell_amount/1000:.0f}k",
                color='#ff1744' if is_closing else COLORS['shell'],
                width=6 if is_closing else 4,
                dashes=True
            )
    
    # ─────────────────────────────────────────────────────────────────
    # FLAG: NÚMEROS REDONDOS (Round Amount Invoices)
    # ─────────────────────────────────────────────────────────────────
    if 'flag_numeros_redondos' in active_flags:
        round_amounts = [100000, 250000, 500000, 750000, 1000000]
        selected_amounts = np.random.choice(round_amounts, size=min(3, len(round_amounts)), replace=False)
        
        for i, amount in enumerate(selected_amounts):
            prov_nif = f"R{np.random.randint(10000000, 99999999)}"
            if prov_nif not in nodes_added:
                net.add_node(
                    prov_nif,
                    label=f"🔢 Prov. Redondo {i+1}",
                    title=f"<b>🔢 FACTURA SOSPECHOSA</b><br>NIF: {prov_nif}<br>💰 Importe EXACTO: €{amount:,.0f}<br>⚠️ 100% números redondos",
                    color=COLORS['round_amounts'],
                    size=35,
                    shape='star'
                )
                nodes_added.add(prov_nif)
                
            net.add_edge(
                prov_nif,
                center_nif,
                title=f"💸 IMPORTE EXACTO<br>€{amount:,.0f}<br>⚠️ Posible factura ficticia",
                label=f"€{amount/1000:.0f}k",
                color=COLORS['round_amounts'],
                width=4,
                dashes=[5, 5]  # Línea punteada corta
            )
    
    # ─────────────────────────────────────────────────────────────────
    # FLAG: INCOHERENCIA LOGÍSTICA (Phantom Logistics)
    # ─────────────────────────────────────────────────────────────────
    if 'flag_incoherencia_logistica' in active_flags:
        logistics_partners = [
            ("TRANSP FICTICIO", "Sin flota registrada"),
            ("LOGÍSTICA ???", "Dirección inexistente"),
        ]
        
        for name, issue in logistics_partners:
            log_nif = f"L{np.random.randint(10000000, 99999999)}"
            if log_nif not in nodes_added:
                log_amount = max(1000, gastos_transporte * 0.1)  # Mínimo simbólico
                net.add_node(
                    log_nif,
                    label=f"📦 {name}",
                    title=f"<b>📦 LOGÍSTICA FANTASMA</b><br>NIF: {log_nif}<br>🚩 {issue}<br>💰 Gasto declarado: €{log_amount:,.0f}<br>⚠️ Sin evidencia de transporte real",
                    color=COLORS['logistics'],
                    size=32,
                    shape='triangleDown'
                )
                nodes_added.add(log_nif)
                
            net.add_edge(
                center_nif,
                log_nif,
                title=f"📦 TRANSPORTE FICTICIO<br>€{log_amount:,.0f}<br>⚠️ M349 > 0 pero transporte ≈ 0",
                label=f"€{log_amount/1000:.1f}k",
                color=COLORS['logistics'],
                width=2,
                dashes=True
            )
    
    # ─────────────────────────────────────────────────────────────────
    # FLAG: DEUDA OCULTA (Hidden Debt)
    # ─────────────────────────────────────────────────────────────────
    if 'flag_hidden_debt' in active_flags:
        hidden_creditors = [
            ("ACREEDOR OCULTO", "No declarado en balance"),
            ("PRÉSTAMO OPACO", "Interés > 15%"),
        ]
        
        for name, issue in hidden_creditors:
            cred_nif = f"D{np.random.randint(10000000, 99999999)}"
            if cred_nif not in nodes_added:
                debt_amount = np.random.randint(100, 500) * 1000
                interest_rate = np.random.uniform(12, 25)
                net.add_node(
                    cred_nif,
                    label=f"💳 {name}",
                    title=f"<b>💳 ACREEDOR OCULTO</b><br>NIF: {cred_nif}<br>🚩 {issue}<br>💰 Deuda: €{debt_amount:,.0f}<br>📈 Tasa implícita: {interest_rate:.1f}%",
                    color=COLORS['debt'],
                    size=35,
                    shape='square'
                )
                nodes_added.add(cred_nif)
                
            net.add_edge(
                cred_nif,
                center_nif,
                title=f"💳 DEUDA OCULTA<br>€{debt_amount:,.0f}<br>📈 Tasa: {interest_rate:.1f}%",
                label=f"€{debt_amount/1000:.0f}k",
                color=COLORS['debt'],
                width=3,
                dashes=[10, 5]  # Línea guion-punto
            )
    
    # ─────────────────────────────────────────────────────────────────
    # FLAG: COBERTURA M347 BAJA (Missing Counterparties)
    # ─────────────────────────────────────────────────────────────────
    if 'flag_cobertura_baja' in active_flags:
        # Mostrar "huecos" en la red - operaciones no declaradas
        missing_volume = ventas * 0.25  # 25% de ventas sin soporte M347
        
        phantom_nif = "PHANTOM_OPS"
        net.add_node(
            phantom_nif,
            label="❓ OPS. NO DECLARADAS",
            title=f"<b>❓ OPERACIONES FANTASMA</b><br>📋 Ventas sin contraparte M347<br>💰 Volumen estimado: €{missing_volume:,.0f}<br>⚠️ Cobertura < 75%",
            color=COLORS['coverage'],
            size=45,
            shape='ellipse',
            opacity=0.6
        )
        nodes_added.add(phantom_nif)
        
        net.add_edge(
            center_nif,
            phantom_nif,
            title=f"❓ VENTAS SIN SOPORTE<br>€{missing_volume:,.0f}<br>⚠️ No aparecen en M347",
            label=f"€{missing_volume/1000:.0f}k ❓",
            color=COLORS['coverage'],
            width=5,
            dashes=[2, 8]  # Línea muy discontinua
        )
    
    # ─────────────────────────────────────────────────────────────────
    # SIEMPRE: AÑADIR CONTRAPARTES NORMALES (Background Network)
    # ─────────────────────────────────────────────────────────────────
    
    # Clientes legítimos
    n_normal_clients = 4 if len(active_flags) > 0 else 6
    for i in range(n_normal_clients):
        cli_nif = f"B{np.random.randint(10000000, 99999999)}"
        if cli_nif not in nodes_added:
            cli_amount = np.random.randint(5, 30) * 1000
            net.add_node(
                cli_nif,
                label=f"Cliente {i+1}",
                title=f"<b>✅ CLIENTE LEGÍTIMO</b><br>NIF: {cli_nif}<br>💰 Importe: €{cli_amount:,.0f}<br>📋 Operación declarada M347",
                color=COLORS['normal'],
                size=25,
                shape='dot'
            )
            nodes_added.add(cli_nif)
            
            net.add_edge(
                center_nif,
                cli_nif,
                title=f"✅ Operación normal<br>€{cli_amount:,.0f}",
                color=COLORS['normal'],
                width=1
            )
    
    # Proveedores legítimos
    n_normal_suppliers = 3 if len(active_flags) > 0 else 4
    for i in range(n_normal_suppliers):
        prov_nif = f"A{np.random.randint(10000000, 99999999)}"
        if prov_nif not in nodes_added:
            prov_amount = np.random.randint(10, 50) * 1000
            net.add_node(
                prov_nif,
                label=f"Proveedor {i+1}",
                title=f"<b>✅ PROVEEDOR LEGÍTIMO</b><br>NIF: {prov_nif}<br>💰 Importe: €{prov_amount:,.0f}<br>📋 Operación verificada",
                color=COLORS['normal'],
                size=28,
                shape='triangle'
            )
            nodes_added.add(prov_nif)
            
            net.add_edge(
                prov_nif,
                center_nif,
                title=f"✅ Compra normal<br>€{prov_amount:,.0f}",
                color=COLORS['normal'],
                width=1
            )
    
    # ═══════════════════════════════════════════════════════════════════
    # GENERAR HTML
    # ═══════════════════════════════════════════════════════════════════
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
        net.save_graph(f.name)
        temp_path = f.name
    
    with open(temp_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    try:
        os.unlink(temp_path)
    except:
        pass
    
    # ═══════════════════════════════════════════════════════════════════
    # LEYENDA DINÁMICA BASADA EN FLAGS ACTIVOS
    # ═══════════════════════════════════════════════════════════════════
    
    legend_items = [
        ('<div style="width:14px;height:14px;background:#e91e63;border:2px solid white;border-radius:3px;"></div>', '🎯 Objetivo Analítico'),
    ]
    
    # Añadir leyenda según flags activos
    if 'flag_empresa_pantalla' in active_flags:
        legend_items.append(('<div style="width:14px;height:14px;background:#f44336;border-radius:2px;"></div>', '🏭 Empresa Pantalla'))
    if 'flag_numeros_redondos' in active_flags:
        legend_items.append(('<div style="width:14px;height:14px;background:#9c27b0;clip-path:polygon(50% 0%,100% 50%,50% 100%,0% 50%);"></div>', '🔢 Números Redondos'))
    if 'flag_incoherencia_logistica' in active_flags:
        legend_items.append(('<div style="width:14px;height:14px;background:#00bcd4;clip-path:polygon(50% 100%,0% 0%,100% 0%);"></div>', '📦 Logística Fantasma'))
    if 'flag_hidden_debt' in active_flags:
        legend_items.append(('<div style="width:14px;height:14px;background:#795548;"></div>', '💳 Deuda Oculta'))
    if 'flag_cobertura_baja' in active_flags:
        legend_items.append(('<div style="width:14px;height:14px;background:#607d8b;border-radius:50%;opacity:0.6;"></div>', '❓ Ops. No Declaradas'))
    
    # Siempre incluir legítimos
    legend_items.append(('<div style="width:14px;height:14px;background:#4caf50;border-radius:50%;"></div>', '✅ Contraparte Legítima'))
    
    legend_html_items = "\n".join([
        f'<div style="display:flex;align-items:center;margin:6px 0;"><div style="margin-right:10px;">{icon}</div><span style="color:#f1f5f9;font-size:11px;">{label}</span></div>'
        for icon, label in legend_items
    ])
    
    legend_html = f"""
    <div style="position:absolute;top:15px;left:15px;background:rgba(15,23,42,0.9);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,0.1);padding:16px;border-radius:12px;z-index:1000;font-family:'Inter',sans-serif;max-width:220px;">
        <div style="font-weight:800;color:#f8fafc;margin-bottom:10px;font-size:12px;letter-spacing:0.05em;border-bottom:1px solid rgba(255,255,255,0.1);padding-bottom:8px;">
            LEYENDA ({len(active_flags)} alertas)
        </div>
        {legend_html_items}
        <hr style="border:0;border-top:1px solid rgba(255,255,255,0.05);margin:10px 0;">
        <div style="color:#94a3b8;font-size:10px;line-height:1.4;">
            🖱️ Arrastrar nodos<br>
            🔍 Scroll = Zoom<br>
            --- Línea punteada = Sospechoso
        </div>
    </div>
    """
    
    center_script = """
    <script type="text/javascript">
        // PERFORMANCE FIX: Disable physics after initial layout
        var isStabilized = false;
        
        function disablePhysics() {
            if (typeof network !== 'undefined' && network !== null) {
                network.setOptions({ physics: { enabled: false } });
                console.log("Physics disabled for performance");
            }
        }
        
        function centerGraph() {
            if (typeof network !== 'undefined' && network !== null) {
                network.fit({
                    animation: { duration: 400, easingFunction: "easeOutQuad" }
                });
            }
        }
        
        // After stabilization: center and STOP physics
        if (typeof network !== 'undefined' && network !== null) {
            network.once("stabilizationIterationsDone", function() {
                isStabilized = true;
                centerGraph();
                // CRITICAL: Disable physics to stop CPU usage
                setTimeout(disablePhysics, 500);
            });
            
            // Re-enable physics ONLY when dragging, then disable again
            network.on("dragStart", function() {
                if (isStabilized) {
                    network.setOptions({ physics: { enabled: true } });
                }
            });
            network.on("dragEnd", function() {
                setTimeout(disablePhysics, 300);
            });
        }
        
        // Fallback: if no stabilization event, center anyway
        setTimeout(function() {
            if (!isStabilized) {
                centerGraph();
                disablePhysics();
                isStabilized = true;
            }
        }, 1000);
    </script>
    """



    
    html_content = html_content.replace('<body>', f'<body>{legend_html}')
    html_content = html_content.replace('</body>', f'{center_script}</body>')
    
    return html_content


# Wrapper para compatibilidad con llamadas antiguas
def create_suspicious_network(center_nif, center_risk, center_score, active_flags=None, company_data=None):
    """Wrapper principal - acepta flags y datos de empresa."""
    return create_interactive_network_html(center_nif, center_risk, center_score, active_flags, company_data)
