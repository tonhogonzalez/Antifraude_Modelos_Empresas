
import networkx as nx
from pyvis.network import Network
import numpy as np
import tempfile
import os

def create_interactive_network_html(center_nif, center_risk, center_score):
    """
    Genera un grafo interactivo con nodos arrastrables usando PyVis.
    Retorna el HTML como string para incrustar en Streamlit.
    """
    
    # ═══════════════════════════════════════════════════════════════════
    # CONFIGURACIÓN DE COLORES
    # ═══════════════════════════════════════════════════════════════════
    
    COLORS = {
        'target': '#e91e63',      # Rosa brillante
        'shell': '#f44336',       # Rojo
        'suspicious': '#ff9800',  # Naranja
        'normal': '#4caf50',      # Verde
        'neutral': '#78909c',     # Gris
    }
    
    np.random.seed(hash(center_nif) % 2**32)
    
    # ═══════════════════════════════════════════════════════════════════
    # CREAR RED PYVIS
    # ═══════════════════════════════════════════════════════════════════
    
    net = Network(
        height="550px",
        width="100%",
        bgcolor="#0e1117",
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
                "gravitationalConstant": -3000,
                "centralGravity": 0.3,
                "springLength": 150,
                "springConstant": 0.04,
                "damping": 0.09
            },
            "stabilization": {
                "enabled": true,
                "iterations": 100
            }
        },
        "interaction": {
            "dragNodes": true,
            "dragView": true,
            "zoomView": true,
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": {
                "enabled": true
            }
        }
    }
    """)
    
    # ═══════════════════════════════════════════════════════════════════
    # AGREGAR NODO CENTRAL
    # ═══════════════════════════════════════════════════════════════════
    
    net.add_node(
        center_nif,
        label=f"🎯 {center_nif}",
        title=f"<b>EMPRESA OBJETIVO</b><br>NIF: {center_nif}<br>Riesgo: {center_risk}<br>Score: {center_score:.3f}",
        color=COLORS['target'],
        size=50,
        shape='diamond',
        font={'size': 16, 'color': 'white'}
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # GENERAR RED SEGÚN RIESGO
    # ═══════════════════════════════════════════════════════════════════
    
    if center_risk == 'Alto':
        # ─────────────────────────────────────────────────────────────
        # PATRÓN CARRUSEL
        # ─────────────────────────────────────────────────────────────
        shell_names = ["SHELL A", "SHELL B", "SHELL C"]
        shell_nifs = [f"X{np.random.randint(10000000, 99999999)}" for _ in range(3)]
        amount = np.random.choice([500, 750, 1000, 1250]) * 1000
        
        for name, nif in zip(shell_names, shell_nifs):
            net.add_node(
                nif,
                label=f"🏭 {name}",
                title=f"<b>⛔ EMPRESA PANTALLA</b><br>NIF: {nif}<br>Tipo: {name}<br>⚠️ Sin actividad real",
                color=COLORS['shell'],
                size=40,
                shape='box'
            )
        
        # Crear ciclo
        cycle = [center_nif] + shell_nifs + [center_nif]
        for i in range(len(cycle) - 1):
            is_closing = (i == len(cycle) - 1)
            net.add_edge(
                cycle[i], 
                cycle[i + 1],
                title=f"€{amount:,.0f}",
                label=f"€{amount/1000:.0f}k",
                color=COLORS['shell'] if not is_closing else '#ff1744',
                width=4 if not is_closing else 6
            )
        
        # Clientes normales
        for i in range(5):
            cli_nif = f"B{np.random.randint(10000000, 99999999)}"
            cli_amount = np.random.randint(5, 20) * 1000
            net.add_node(
                cli_nif,
                label=f"Cliente {i+1}",
                title=f"<b>Cliente Normal</b><br>NIF: {cli_nif}<br>Importe: €{cli_amount:,.0f}",
                color=COLORS['neutral'],
                size=25,
                shape='dot'
            )
            net.add_edge(
                center_nif,
                cli_nif,
                title=f"€{cli_amount:,.0f}",
                color=COLORS['neutral'],
                width=1
            )
            
    elif center_risk == 'Medio':
        # ─────────────────────────────────────────────────────────────
        # PATRÓN HUB
        # ─────────────────────────────────────────────────────────────
        for i in range(3):
            prov_nif = f"Y{np.random.randint(10000000, 99999999)}"
            prov_amount = np.random.randint(50, 150) * 1000
            net.add_node(
                prov_nif,
                label=f"⚠️ Prov. {i+1}",
                title=f"<b>⚠️ PROVEEDOR SOSPECHOSO</b><br>NIF: {prov_nif}<br>Importe: €{prov_amount:,.0f}",
                color=COLORS['suspicious'],
                size=35,
                shape='triangle'
            )
            net.add_edge(
                prov_nif,
                center_nif,
                title=f"€{prov_amount:,.0f}",
                label=f"€{prov_amount/1000:.0f}k",
                color=COLORS['suspicious'],
                width=3
            )
        
        for i in range(6):
            cli_nif = f"B{np.random.randint(10000000, 99999999)}"
            cli_amount = np.random.randint(8, 40) * 1000
            net.add_node(
                cli_nif,
                label=f"Cliente {i+1}",
                title=f"<b>Cliente</b><br>NIF: {cli_nif}<br>Importe: €{cli_amount:,.0f}",
                color=COLORS['normal'],
                size=28,
                shape='dot'
            )
            net.add_edge(
                center_nif,
                cli_nif,
                title=f"€{cli_amount:,.0f}",
                color=COLORS['normal'],
                width=1
            )
    else:
        # ─────────────────────────────────────────────────────────────
        # PATRÓN NORMAL
        # ─────────────────────────────────────────────────────────────
        for i in range(4):
            prov_nif = f"A{np.random.randint(10000000, 99999999)}"
            prov_amount = np.random.randint(10, 60) * 1000
            net.add_node(
                prov_nif,
                label=f"Proveedor {i+1}",
                title=f"<b>Proveedor</b><br>NIF: {prov_nif}<br>Importe: €{prov_amount:,.0f}",
                color=COLORS['normal'],
                size=30,
                shape='triangle'
            )
            net.add_edge(
                prov_nif,
                center_nif,
                title=f"€{prov_amount:,.0f}",
                color=COLORS['normal'],
                width=2
            )
        
        for i in range(6):
            cli_nif = f"B{np.random.randint(10000000, 99999999)}"
            cli_amount = np.random.randint(3, 20) * 1000
            net.add_node(
                cli_nif,
                label=f"Cliente {i+1}",
                title=f"<b>Cliente</b><br>NIF: {cli_nif}<br>Importe: €{cli_amount:,.0f}",
                color=COLORS['normal'],
                size=25,
                shape='dot'
            )
            net.add_edge(
                center_nif,
                cli_nif,
                title=f"€{cli_amount:,.0f}",
                color=COLORS['normal'],
                width=1
            )

    # ═══════════════════════════════════════════════════════════════════
    # GENERAR HTML
    # ═══════════════════════════════════════════════════════════════════
    
    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
        net.save_graph(f.name)
        temp_path = f.name
    
    # Leer el HTML generado
    with open(temp_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Limpiar archivo temporal
    try:
        os.unlink(temp_path)
    except:
        pass
    
    # Añadir estilos adicionales y leyenda
    legend_html = """
    <div style="position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.8); padding: 15px; border-radius: 8px; z-index: 1000;">
        <div style="font-weight: bold; color: white; margin-bottom: 10px; font-size: 14px;">LEYENDA</div>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 15px; height: 15px; background: #e91e63; border-radius: 3px; margin-right: 8px;"></div>
            <span style="color: white; font-size: 12px;">🎯 Empresa Objetivo</span>
        </div>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 15px; height: 15px; background: #f44336; border-radius: 3px; margin-right: 8px;"></div>
            <span style="color: white; font-size: 12px;">🏭 Empresa Pantalla</span>
        </div>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 15px; height: 15px; background: #ff9800; border-radius: 3px; margin-right: 8px;"></div>
            <span style="color: white; font-size: 12px;">⚠️ Sospechoso</span>
        </div>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 15px; height: 15px; background: #4caf50; border-radius: 3px; margin-right: 8px;"></div>
            <span style="color: white; font-size: 12px;">✅ Normal</span>
        </div>
        <hr style="border-color: #444; margin: 10px 0;">
        <div style="color: #aaa; font-size: 11px;">
            🖱️ Arrastra nodos para moverlos<br>
            🔍 Scroll para zoom<br>
            ✋ Click + arrastrar fondo para mover
        </div>
    </div>
    """
    
    # Script para forzar centrado inicial (Versión Robusta)
    center_script = """
    <script type="text/javascript">
        // Función de centrado seguro
        function forceFit() {
            if (typeof network !== 'undefined') {
                network.fit({
                    animation: {
                        duration: 1000,
                        easingFunction: "easeInOutQuad"
                    }
                });
                console.log("Network forcing fit...");
            }
        }

        // 1. Intentar inmediatamente por si ya está listo
        setTimeout(forceFit, 100);

        // 2. Intentar después de estabilización
        if (typeof network !== 'undefined') {
            network.once("stabilizationIterationsDone", function() {
                console.log("Stabilization done");
                forceFit();
            });
            
            // 3. Intentar en al primer dibujo (fallback)
            network.once("afterDrawing", function() {
                console.log("First drawing done");
                forceFit();
            });
        }
        
        // 4. Último recurso: timer largo
        setTimeout(forceFit, 2000);
    </script>
    """
    
    # Insertar leyenda después del body y script antes del cierre
    html_content = html_content.replace('<body>', f'<body>{legend_html}')
    html_content = html_content.replace('</body>', f'{center_script}</body>')
    
    return html_content


# Mantener compatibilidad con el nombre anterior
def create_suspicious_network(center_nif, center_risk, center_score):
    """Wrapper para compatibilidad."""
    return create_interactive_network_html(center_nif, center_risk, center_score)
