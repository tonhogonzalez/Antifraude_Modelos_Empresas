
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import math

def create_suspicious_network(center_nif, center_risk, center_score):
    """
    Genera un grafo de red transaccional profesional y usable.
    Optimizado para claridad visual y facilidad de uso.
    """
    
    # ═══════════════════════════════════════════════════════════════════
    # CONFIGURACIÓN
    # ═══════════════════════════════════════════════════════════════════
    
    COLORS = {
        'bg': '#0e1117',
        'target': '#e91e63',      # Rosa/Magenta brillante
        'shell': '#f44336',       # Rojo
        'suspicious': '#ff9800',  # Naranja
        'normal': '#4caf50',      # Verde
        'neutral': '#607d8b',     # Gris azulado
    }
    
    np.random.seed(hash(center_nif) % 2**32)
    
    # ═══════════════════════════════════════════════════════════════════
    # CONSTRUIR DATOS DEL GRAFO
    # ═══════════════════════════════════════════════════════════════════
    
    nodes = []
    edges = []
    
    # Nodo central
    nodes.append({
        'id': center_nif,
        'label': f"🎯 {center_nif}",
        'size': 70,
        'color': COLORS['target'],
        'type': 'target',
        'hover': f"<b>EMPRESA OBJETIVO</b><br>NIF: {center_nif}<br>Riesgo: {center_risk}<br>Score: {center_score:.3f}"
    })
    
    if center_risk == 'Alto':
        # ─────────────────────────────────────────────────────────────
        # PATRÓN CARRUSEL: Empresas pantalla formando ciclo
        # ─────────────────────────────────────────────────────────────
        shell_names = ["SHELL A", "SHELL B", "SHELL C"]
        shell_nifs = [f"X{np.random.randint(10000000, 99999999)}" for _ in range(3)]
        amount = np.random.choice([500, 750, 1000, 1250]) * 1000
        
        for i, (name, nif) in enumerate(zip(shell_names, shell_nifs)):
            nodes.append({
                'id': nif,
                'label': f"🏭 {name}",
                'size': 50,
                'color': COLORS['shell'],
                'type': 'shell',
                'hover': f"<b>⛔ EMPRESA PANTALLA</b><br>NIF: {nif}<br>Tipo: {name}<br>⚠️ Sin actividad real"
            })
        
        # Crear ciclo: Target → A → B → C → Target
        cycle = [center_nif] + shell_nifs + [center_nif]
        for i in range(len(cycle) - 1):
            edges.append({
                'source': cycle[i],
                'target': cycle[i + 1],
                'amount': amount,
                'color': COLORS['shell'],
                'type': 'fraud'
            })
        
        # Clientes normales
        for i in range(5):
            cli_nif = f"B{np.random.randint(10000000, 99999999)}"
            cli_amount = np.random.randint(5, 20) * 1000
            nodes.append({
                'id': cli_nif,
                'label': f"Cliente {i+1}",
                'size': 25,
                'color': COLORS['neutral'],
                'type': 'normal',
                'hover': f"<b>Cliente</b><br>NIF: {cli_nif}<br>Importe: €{cli_amount:,.0f}"
            })
            edges.append({
                'source': center_nif,
                'target': cli_nif,
                'amount': cli_amount,
                'color': COLORS['neutral'],
                'type': 'normal'
            })
            
    elif center_risk == 'Medio':
        # ─────────────────────────────────────────────────────────────
        # PATRÓN HUB: Proveedores sospechosos
        # ─────────────────────────────────────────────────────────────
        for i in range(3):
            prov_nif = f"Y{np.random.randint(10000000, 99999999)}"
            prov_amount = np.random.randint(50, 150) * 1000
            nodes.append({
                'id': prov_nif,
                'label': f"⚠️ Prov. {i+1}",
                'size': 40,
                'color': COLORS['suspicious'],
                'type': 'suspicious',
                'hover': f"<b>⚠️ PROVEEDOR SOSPECHOSO</b><br>NIF: {prov_nif}<br>Importe: €{prov_amount:,.0f}"
            })
            edges.append({
                'source': prov_nif,
                'target': center_nif,
                'amount': prov_amount,
                'color': COLORS['suspicious'],
                'type': 'warning'
            })
        
        # Clientes normales
        for i in range(6):
            cli_nif = f"B{np.random.randint(10000000, 99999999)}"
            cli_amount = np.random.randint(8, 40) * 1000
            nodes.append({
                'id': cli_nif,
                'label': f"Cliente {i+1}",
                'size': 28,
                'color': COLORS['normal'],
                'type': 'normal',
                'hover': f"<b>Cliente</b><br>NIF: {cli_nif}<br>Importe: €{cli_amount:,.0f}"
            })
            edges.append({
                'source': center_nif,
                'target': cli_nif,
                'amount': cli_amount,
                'color': COLORS['normal'],
                'type': 'normal'
            })
    else:
        # ─────────────────────────────────────────────────────────────
        # PATRÓN NORMAL
        # ─────────────────────────────────────────────────────────────
        for i in range(4):
            prov_nif = f"A{np.random.randint(10000000, 99999999)}"
            prov_amount = np.random.randint(10, 60) * 1000
            nodes.append({
                'id': prov_nif,
                'label': f"Proveedor {i+1}",
                'size': 32,
                'color': COLORS['normal'],
                'type': 'normal',
                'hover': f"<b>Proveedor</b><br>NIF: {prov_nif}<br>Importe: €{prov_amount:,.0f}"
            })
            edges.append({
                'source': prov_nif,
                'target': center_nif,
                'amount': prov_amount,
                'color': COLORS['normal'],
                'type': 'normal'
            })
        
        for i in range(6):
            cli_nif = f"B{np.random.randint(10000000, 99999999)}"
            cli_amount = np.random.randint(3, 20) * 1000
            nodes.append({
                'id': cli_nif,
                'label': f"Cliente {i+1}",
                'size': 28,
                'color': COLORS['normal'],
                'type': 'normal',
                'hover': f"<b>Cliente</b><br>NIF: {cli_nif}<br>Importe: €{cli_amount:,.0f}"
            })
            edges.append({
                'source': center_nif,
                'target': cli_nif,
                'amount': cli_amount,
                'color': COLORS['normal'],
                'type': 'normal'
            })

    # ═══════════════════════════════════════════════════════════════════
    # CALCULAR POSICIONES (Layout circular mejorado)
    # ═══════════════════════════════════════════════════════════════════
    
    positions = {}
    
    # Centro
    positions[center_nif] = (0, 0)
    
    # Separar nodos por tipo
    shells = [n for n in nodes if n['type'] == 'shell']
    suspicious = [n for n in nodes if n['type'] == 'suspicious']
    normals = [n for n in nodes if n['type'] == 'normal']
    
    # Posicionar shells en círculo interior
    if shells:
        radius = 1.5
        for i, node in enumerate(shells):
            angle = 2 * math.pi * i / len(shells) - math.pi/2
            positions[node['id']] = (radius * math.cos(angle), radius * math.sin(angle))
    
    # Posicionar sospechosos
    if suspicious:
        radius = 2.0
        for i, node in enumerate(suspicious):
            angle = 2 * math.pi * i / len(suspicious) + math.pi/4
            positions[node['id']] = (radius * math.cos(angle), radius * math.sin(angle))
    
    # Posicionar normales en círculo exterior
    if normals:
        radius = 3.0
        for i, node in enumerate(normals):
            angle = 2 * math.pi * i / len(normals)
            positions[node['id']] = (radius * math.cos(angle), radius * math.sin(angle))

    # ═══════════════════════════════════════════════════════════════════
    # CREAR TRAZAS PLOTLY
    # ═══════════════════════════════════════════════════════════════════
    
    traces = []
    annotations = []
    
    # ─────────────────────────────────────────────────────────────────
    # ARISTAS (con curvas Bezier para mejor visualización)
    # ─────────────────────────────────────────────────────────────────
    
    for edge in edges:
        x0, y0 = positions[edge['source']]
        x1, y1 = positions[edge['target']]
        
        # Grosor basado en importe
        width = max(2, min(8, edge['amount'] / 100000))
        
        # Crear línea
        edge_trace = go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode='lines',
            line=dict(
                width=width,
                color=edge['color'],
            ),
            hoverinfo='text',
            hovertext=f"€{edge['amount']:,.0f}",
            opacity=0.7
        )
        traces.append(edge_trace)
        
        # Añadir flecha (triángulo al final)
        dx = x1 - x0
        dy = y1 - y0
        length = math.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            # Punto cerca del destino
            arrow_x = x0 + dx * 0.7
            arrow_y = y0 + dy * 0.7
            angle = math.degrees(math.atan2(dy, dx))
            
            arrow_trace = go.Scatter(
                x=[arrow_x],
                y=[arrow_y],
                mode='markers',
                marker=dict(
                    symbol='triangle-right',
                    size=width * 2 + 4,
                    color=edge['color'],
                    angle=angle
                ),
                hoverinfo='skip'
            )
            traces.append(arrow_trace)
        
        # Etiqueta de importe para transacciones importantes
        if edge['type'] in ['fraud', 'warning']:
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            annotations.append(dict(
                x=mid_x,
                y=mid_y + 0.25,
                text=f"<b>€{edge['amount']/1000:.0f}k</b>",
                showarrow=False,
                font=dict(size=12, color='#ffeb3b'),
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor=edge['color'],
                borderwidth=2,
                borderpad=4
            ))

    # ─────────────────────────────────────────────────────────────────
    # NODOS
    # ─────────────────────────────────────────────────────────────────
    
    for node in nodes:
        x, y = positions[node['id']]
        
        node_trace = go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            marker=dict(
                size=node['size'],
                color=node['color'],
                line=dict(width=3, color='white'),
                opacity=1
            ),
            text=node['label'],
            textposition='bottom center',
            textfont=dict(
                size=12 if node['type'] == 'target' else 10,
                color='white',
                family='Arial'
            ),
            hoverinfo='text',
            hovertext=node['hover'],
            hoverlabel=dict(
                bgcolor='rgba(20,20,30,0.95)',
                bordercolor='white',
                font=dict(size=13, color='white')
            )
        )
        traces.append(node_trace)

    # ═══════════════════════════════════════════════════════════════════
    # FIGURA FINAL
    # ═══════════════════════════════════════════════════════════════════
    
    fig = go.Figure(data=traces)
    
    # Leyenda como shapes/annotations
    legend_items = [
        ("🎯 Empresa Objetivo", COLORS['target']),
        ("🏭 Empresa Pantalla", COLORS['shell']),
        ("⚠️ Sospechoso", COLORS['suspicious']),
        ("✅ Normal", COLORS['normal']),
    ]
    
    for i, (text, color) in enumerate(legend_items):
        # Círculo de color
        fig.add_shape(
            type="circle",
            x0=-4.8, y0=3.5 - i*0.6 - 0.15,
            x1=-4.5, y1=3.5 - i*0.6 + 0.15,
            fillcolor=color,
            line=dict(color='white', width=1)
        )
        # Texto
        annotations.append(dict(
            x=-4.3,
            y=3.5 - i*0.6,
            text=text,
            showarrow=False,
            font=dict(size=11, color='white'),
            xanchor='left'
        ))
    
    # Título y configuración
    fig.update_layout(
        title=dict(
            text=f"<b>Red de Operaciones M347</b> | {center_nif} | <span style='color:{COLORS['target']}'>Riesgo {center_risk}</span>",
            font=dict(size=16, color='white'),
            x=0.5,
            y=0.98
        ),
        showlegend=False,
        hovermode='closest',
        plot_bgcolor=COLORS['bg'],
        paper_bgcolor=COLORS['bg'],
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-5.5, 5.5],
            scaleanchor='y'
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-4.5, 4.5]
        ),
        height=600,
        annotations=annotations,
        dragmode='pan'
    )
    
    # Configuración de interactividad
    fig.update_layout(
        modebar=dict(
            bgcolor='rgba(30,30,40,0.8)',
            color='white',
            activecolor='#00bcd4',
            orientation='h'
        )
    )
    
    return fig
