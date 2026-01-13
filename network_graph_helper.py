
import networkx as nx
import plotly.graph_objects as go
import numpy as np

def create_suspicious_network(center_nif, center_risk, center_score):
    """
    Genera un subgrafo transaccional simulado alrededor de una empresa.
    Si la empresa es de alto riesgo, genera patrones de fraude (circularidad, hubs).
    """
    G = nx.DiGraph()
    
    # Colores según riesgo
    color_map = {
        'Alto': '#ff4b4b',      # Rojo brillante
        'Medio': '#ffa726',     # Naranja
        'Bajo': '#66bb6a'       # Verde
    }
    center_color = color_map.get(center_risk, '#66bb6a')
    
    # Nodo central (TARGET)
    G.add_node(center_nif, 
               size=50, 
               color=center_color, 
               label=f"🎯 {center_nif}",
               hover=f"<b>EMPRESA TARGET</b><br>NIF: {center_nif}<br>Riesgo: {center_risk}<br>Score: {center_score:.3f}",
               type='target')
    
    # Semilla para reproducibilidad
    np.random.seed(hash(center_nif) % 2**32)
    
    if center_risk == 'Alto':
        # ═══════════════════════════════════════════════════════════════
        # PATRÓN FRAUDE CARRUSEL: A → B → C → A (ciclo cerrado)
        # ═══════════════════════════════════════════════════════════════
        shell_companies = []
        for i in range(3):
            nif = f"X{np.random.randint(10000000, 99999999)}"
            shell_companies.append(nif)
            G.add_node(nif, 
                       size=35, 
                       color='#e53935',  # Rojo intenso
                       label=f"🏭 Shell Co. {i+1}",
                       hover=f"<b>⚠️ EMPRESA PANTALLA</b><br>NIF: {nif}<br>Sin actividad real<br>Solo facturación circular",
                       type='shell')
        
        # Crear ciclo fraudulento con importes elevados
        main_amount = np.random.randint(800000, 2500000)
        
        # Target → Shell1
        G.add_edge(center_nif, shell_companies[0], 
                   weight=main_amount, 
                   color='#ff5252',
                   label=f"€{main_amount/1000:.0f}k",
                   hover=f"<b>TRANSACCIÓN SOSPECHOSA</b><br>Importe: €{main_amount:,.0f}<br>⚠️ Importe redondo")
        
        # Shell1 → Shell2
        G.add_edge(shell_companies[0], shell_companies[1], 
                   weight=main_amount, 
                   color='#ff5252',
                   label=f"€{main_amount/1000:.0f}k",
                   hover=f"<b>TRANSACCIÓN CIRCULAR</b><br>Importe: €{main_amount:,.0f}<br>⚠️ Mismo importe exacto")
        
        # Shell2 → Shell3
        G.add_edge(shell_companies[1], shell_companies[2], 
                   weight=main_amount,
                   color='#ff5252',
                   label=f"€{main_amount/1000:.0f}k",
                   hover=f"<b>TRANSACCIÓN CIRCULAR</b><br>Importe: €{main_amount:,.0f}<br>⚠️ Flujo sin justificación")
        
        # Shell3 → Target (CIERRE DEL CÍRCULO)
        G.add_edge(shell_companies[2], center_nif, 
                   weight=main_amount,
                   color='#ff1744',  # Rojo más intenso
                   label=f"€{main_amount/1000:.0f}k ⚠️",
                   hover=f"<b>🔴 CIERRE CARRUSEL</b><br>Importe: €{main_amount:,.0f}<br>⛔ CIRCULARIDAD DETECTADA")
        
        # Añadir algunos clientes normales para "disimular"
        for i in range(4):
            client_nif = f"B{np.random.randint(10000000, 99999999)}"
            client_amount = np.random.randint(5000, 25000)
            G.add_node(client_nif, 
                       size=20, 
                       color='#78909c',  # Gris
                       label=f"Cliente {i+1}",
                       hover=f"<b>Cliente Normal</b><br>NIF: {client_nif}<br>Operación: €{client_amount:,.0f}",
                       type='normal')
            G.add_edge(center_nif, client_nif, weight=client_amount, color='#546e7a')

    elif center_risk == 'Medio':
        # ═══════════════════════════════════════════════════════════════
        # PATRÓN HUB: Concentración anómala de proveedores sospechosos
        # ═══════════════════════════════════════════════════════════════
        
        # Proveedores sospechosos
        for i in range(3):
            prov_nif = f"Y{np.random.randint(10000000, 99999999)}"
            prov_amount = np.random.randint(50000, 150000)
            G.add_node(prov_nif, 
                       size=30, 
                       color='#ffb74d',  # Naranja
                       label=f"⚠️ Prov. {i+1}",
                       hover=f"<b>Proveedor Sospechoso</b><br>NIF: {prov_nif}<br>Facturación Alta: €{prov_amount:,.0f}<br>⚠️ Sin histórico previo",
                       type='warning')
            G.add_edge(prov_nif, center_nif, weight=prov_amount, color='#ff9800',
                      hover=f"<b>Compra a Proveedor</b><br>Importe: €{prov_amount:,.0f}")
        
        # Clientes normales
        for i in range(6):
            cli_nif = f"B{np.random.randint(10000000, 99999999)}"
            cli_amount = np.random.randint(8000, 40000)
            G.add_node(cli_nif, 
                       size=22, 
                       color='#78909c',
                       label=f"Cliente {i+1}",
                       hover=f"<b>Cliente</b><br>NIF: {cli_nif}<br>Venta: €{cli_amount:,.0f}",
                       type='normal')
            G.add_edge(center_nif, cli_nif, weight=cli_amount, color='#546e7a')

    else:
        # ═══════════════════════════════════════════════════════════════
        # PATRÓN NORMAL: Red de negocio estándar
        # ═══════════════════════════════════════════════════════════════
        
        # Proveedores legítimos
        for i in range(np.random.randint(3, 5)):
            prov_nif = f"A{np.random.randint(10000000, 99999999)}"
            prov_amount = np.random.randint(10000, 80000)
            G.add_node(prov_nif, 
                       size=25, 
                       color='#4caf50',  # Verde
                       label=f"Prov. {i+1}",
                       hover=f"<b>Proveedor Verificado</b><br>NIF: {prov_nif}<br>Compra: €{prov_amount:,.0f}",
                       type='normal')
            G.add_edge(prov_nif, center_nif, weight=prov_amount, color='#66bb6a')
        
        # Clientes legítimos
        for i in range(np.random.randint(5, 8)):
            cli_nif = f"B{np.random.randint(10000000, 99999999)}"
            cli_amount = np.random.randint(3000, 20000)
            G.add_node(cli_nif, 
                       size=22, 
                       color='#4caf50',
                       label=f"Cliente {i+1}",
                       hover=f"<b>Cliente</b><br>NIF: {cli_nif}<br>Venta: €{cli_amount:,.0f}",
                       type='normal')
            G.add_edge(center_nif, cli_nif, weight=cli_amount, color='#66bb6a')

    # ═══════════════════════════════════════════════════════════════════
    # VISUALIZACIÓN PLOTLY MEJORADA
    # ═══════════════════════════════════════════════════════════════════
    
    # Layout con mejor distribución
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # --- EDGES ---
    edge_traces = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Calcular punto medio para la flecha
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        
        edge_color = edge[2].get('color', '#555')
        edge_weight = edge[2].get('weight', 1000)
        
        # Grosor basado en el importe
        line_width = max(1, min(8, edge_weight / 100000))
        
        # Trace para la línea
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=line_width, color=edge_color),
            hoverinfo='skip',
            showlegend=False
        )
        edge_traces.append(edge_trace)
        
        # Añadir marcador de flecha en el punto medio
        # Calcular dirección
        dx = x1 - x0
        dy = y1 - y0
        
        arrow_trace = go.Scatter(
            x=[mid_x],
            y=[mid_y],
            mode='markers',
            marker=dict(
                symbol='triangle-right',
                size=12,
                color=edge_color,
                angle=np.degrees(np.arctan2(dy, dx))
            ),
            hoverinfo='text',
            hovertext=edge[2].get('hover', f"€{edge_weight:,.0f}"),
            showlegend=False
        )
        edge_traces.append(arrow_trace)

    # --- NODES ---
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_texts = []
    node_hovers = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(G.nodes[node].get('color', '#888'))
        node_sizes.append(G.nodes[node].get('size', 20))
        node_texts.append(G.nodes[node].get('label', node))
        node_hovers.append(G.nodes[node].get('hover', node))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white'),
            opacity=0.95
        ),
        text=node_texts,
        textposition='top center',
        textfont=dict(size=10, color='white'),
        hoverinfo='text',
        hovertext=node_hovers,
        showlegend=False
    )

    # --- FIGURE ---
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        title=dict(
            text=f"🕸️ Red Transaccional M347 - {center_nif}",
            font=dict(size=18, color='white'),
            x=0.5
        ),
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        height=500
    )
    
    # Añadir anotaciones para los importes de las transacciones principales
    annotations = []
    for edge in G.edges(data=True):
        if 'label' in edge[2]:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            annotations.append(
                dict(
                    x=(x0 + x1) / 2,
                    y=(y0 + y1) / 2 + 0.08,
                    text=edge[2]['label'],
                    showarrow=False,
                    font=dict(size=9, color='#ddd'),
                    bgcolor='rgba(0,0,0,0.5)',
                    borderpad=2
                )
            )
    
    fig.update_layout(annotations=annotations)
    
    return fig
