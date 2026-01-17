
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
        bgcolor="#0f172a", # matched to slate-900
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
            "navigationButtons": false,
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
                title=f"🔄 CICLO SOSPECHOSO<br>Importe: €{amount:,.0f}",
                label=f"€{amount/1000:.0f}k",
                color=COLORS['shell'] if not is_closing else '#ff1744',
                width=4 if not is_closing else 6,
                dashes=True # Contextual indicator for suspicious flow
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
    <div style="position: absolute; top: 15px; left: 15px; background: rgba(15, 23, 42, 0.85); 
                backdrop-filter: blur(8px); border: 1px solid rgba(255, 255, 255, 0.1); 
                padding: 18px; border-radius: 12px; z-index: 1000; font-family: 'Inter', sans-serif;">
        <div style="font-weight: 800; color: #f8fafc; margin-bottom: 12px; font-size: 13px; 
                    letter-spacing: 0.05em; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px;">
            SISTEMA DE LEYENDA
        </div>
        <div style="display: flex; align-items: center; margin: 8px 0;">
            <div style="width: 14px; height: 14px; background: #e91e63; border: 2px solid white; border-radius: 3px; margin-right: 10px;"></div>
            <span style="color: #f1f5f9; font-size: 12px; font-weight: 600;">🎯 Objetivo Analítico</span>
        </div>
        <div style="display: flex; align-items: center; margin: 8px 0;">
            <div style="width: 14px; height: 14px; background: #f44336; border-radius: 2px; margin-right: 10px;"></div>
            <span style="color: #f1f5f9; font-size: 12px;">🏭 Empresa Pantalla</span>
        </div>
        <div style="display: flex; align-items: center; margin: 8px 0;">
            <div style="width: 14px; height: 14px; border: 2px solid #ff9800; transform: rotate(45deg); margin-right: 10px;"></div>
            <span style="color: #f1f5f9; font-size: 12px;">⚠️ Proveedor Inusual</span>
        </div>
        <div style="display: flex; align-items: center; margin: 8px 0;">
            <div style="width: 14px; height: 14px; background: #4caf50; border-radius: 50%; margin-right: 10px;"></div>
            <span style="color: #f1f5f9; font-size: 12px;">✅ Cliente Legítimo</span>
        </div>
        <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.05); margin: 12px 0;">
        <div style="color: #94a3b8; font-size: 11px; line-height: 1.5;">
            🖱️ <b>Arrastrar:</b> Mover nodos<br>
            🔍 <b>Scroll:</b> Zoom dinámico<br>
            ✋ <b>Click fondo:</b> Panorámica
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
