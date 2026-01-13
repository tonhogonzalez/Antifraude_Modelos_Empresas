
import networkx as nx
import plotly.graph_objects as go
import numpy as np

def create_suspicious_network(center_nif, center_risk, center_score):
    """
    Genera un subgrafo transaccional simulado alrededor de una empresa.
    Visualización profesional con zoom, pan e interactividad completa.
    """
    G = nx.DiGraph()
    
    # ═══════════════════════════════════════════════════════════════════
    # CONFIGURACIÓN DE COLORES Y ESTILOS
    # ═══════════════════════════════════════════════════════════════════
    COLORS = {
        'target_high': '#ff1744',     # Rojo brillante
        'target_medium': '#ff9100',   # Naranja
        'target_low': '#00e676',      # Verde neón
        'shell': '#d50000',           # Rojo oscuro
        'suspicious': '#ff6d00',      # Naranja oscuro
        'normal': '#455a64',          # Gris azulado
        'edge_fraud': '#ff5252',      # Rojo para transacciones fraudulentas
        'edge_warning': '#ffab40',    # Naranja para alertas
        'edge_normal': '#78909c',     # Gris para normales
        'bg': '#0e1117',              # Fondo oscuro
    }
    
    target_color = {
        'Alto': COLORS['target_high'],
        'Medio': COLORS['target_medium'],
        'Bajo': COLORS['target_low']
    }.get(center_risk, COLORS['target_low'])
    
    # ═══════════════════════════════════════════════════════════════════
    # NODO CENTRAL (TARGET)
    # ═══════════════════════════════════════════════════════════════════
    G.add_node(center_nif, 
               size=60, 
               color=target_color,
               symbol='diamond',
               label=f"🎯 {center_nif}",
               hover=f"<b>═══ EMPRESA OBJETIVO ═══</b><br><br>"
                     f"<b>NIF:</b> {center_nif}<br>"
                     f"<b>Nivel de Riesgo:</b> {center_risk}<br>"
                     f"<b>Score de Fraude:</b> {center_score:.3f}<br><br>"
                     f"<i>Empresa bajo análisis forense</i>",
               category='target')

    np.random.seed(hash(center_nif) % 2**32)
    
    # ═══════════════════════════════════════════════════════════════════
    # GENERACIÓN DE RED SEGÚN NIVEL DE RIESGO
    # ═══════════════════════════════════════════════════════════════════
    
    if center_risk == 'Alto':
        # ──────────────────────────────────────────────────────────────
        # PATRÓN: CARRUSEL DE IVA (Fraude Fiscal Grave)
        # Estructura: Target ↔ Shell1 ↔ Shell2 ↔ Shell3 → Target
        # ──────────────────────────────────────────────────────────────
        
        # Crear empresas pantalla
        shell_companies = []
        for i in range(4):
            nif = f"X{np.random.randint(10000000, 99999999)}"
            shell_companies.append(nif)
            
            company_types = ["HOLDING OFFSHORE", "IMPORT/EXPORT S.L.", "TRADING CO.", "INVESTMENTS LTD"]
            G.add_node(nif, 
                       size=45, 
                       color=COLORS['shell'],
                       symbol='square',
                       label=f"🏭 {company_types[i]}",
                       hover=f"<b>⛔ EMPRESA PANTALLA DETECTADA</b><br><br>"
                             f"<b>NIF:</b> {nif}<br>"
                             f"<b>Tipo:</b> {company_types[i]}<br>"
                             f"<b>Personal:</b> 0-1 empleados<br>"
                             f"<b>Sede:</b> Domicilio fiscal ficticio<br><br>"
                             f"<span style='color:#ff5252'>⚠️ Sin actividad económica real</span>",
                       category='shell')
        
        # Transacciones del carrusel (importes elevados y redondos)
        carousel_amount = np.random.choice([500000, 750000, 1000000, 1250000, 1500000])
        
        # Ciclo fraudulento
        edges_carousel = [
            (center_nif, shell_companies[0]),
            (shell_companies[0], shell_companies[1]),
            (shell_companies[1], shell_companies[2]),
            (shell_companies[2], shell_companies[3]),
            (shell_companies[3], center_nif),  # Cierre del ciclo
        ]
        
        for i, (src, dst) in enumerate(edges_carousel):
            is_closing = (i == len(edges_carousel) - 1)
            G.add_edge(src, dst, 
                       weight=carousel_amount,
                       color=COLORS['edge_fraud'],
                       width=6 if is_closing else 4,
                       dash='solid',
                       label=f"€{carousel_amount/1000:.0f}k",
                       hover=f"<b>{'🔴 CIERRE CARRUSEL' if is_closing else '⚠️ TRANSACCIÓN CIRCULAR'}</b><br><br>"
                             f"<b>Importe:</b> €{carousel_amount:,.0f}<br>"
                             f"<b>Origen:</b> {src[:12]}...<br>"
                             f"<b>Destino:</b> {dst[:12]}...<br><br>"
                             f"<span style='color:#ff5252'>⛔ Patrón de circularidad detectado</span>",
                       category='fraud')
        
        # Añadir clientes reales (para camuflaje)
        for i in range(6):
            cli_nif = f"B{np.random.randint(10000000, 99999999)}"
            cli_amount = np.random.randint(3000, 15000)
            G.add_node(cli_nif, 
                       size=20, 
                       color=COLORS['normal'],
                       symbol='circle',
                       label=f"Cliente",
                       hover=f"<b>Cliente Regular</b><br><br>"
                             f"<b>NIF:</b> {cli_nif}<br>"
                             f"<b>Facturación:</b> €{cli_amount:,.0f}<br>"
                             f"<b>Estado:</b> ✅ Sin alertas",
                       category='normal')
            G.add_edge(center_nif, cli_nif, 
                       weight=cli_amount, 
                       color=COLORS['edge_normal'],
                       width=1,
                       category='normal')
        
        # Añadir proveedores sospechosos adicionales
        for i in range(2):
            prov_nif = f"Z{np.random.randint(10000000, 99999999)}"
            prov_amount = np.random.randint(100000, 300000)
            G.add_node(prov_nif, 
                       size=35, 
                       color=COLORS['suspicious'],
                       symbol='triangle-up',
                       label=f"⚠️ Prov. Susp.",
                       hover=f"<b>⚠️ PROVEEDOR SOSPECHOSO</b><br><br>"
                             f"<b>NIF:</b> {prov_nif}<br>"
                             f"<b>Facturación:</b> €{prov_amount:,.0f}<br>"
                             f"<b>Alerta:</b> Alta concentración de facturación<br>"
                             f"<b>Histórico:</b> Sin relación previa",
                       category='suspicious')
            G.add_edge(prov_nif, center_nif, 
                       weight=prov_amount, 
                       color=COLORS['edge_warning'],
                       width=3,
                       category='warning')

    elif center_risk == 'Medio':
        # ──────────────────────────────────────────────────────────────
        # PATRÓN: HUB ANÓMALO (Concentración sospechosa)
        # ──────────────────────────────────────────────────────────────
        
        # Proveedores con alertas
        for i in range(4):
            prov_nif = f"Y{np.random.randint(10000000, 99999999)}"
            prov_amount = np.random.randint(40000, 120000)
            G.add_node(prov_nif, 
                       size=32, 
                       color=COLORS['suspicious'],
                       symbol='triangle-up',
                       label=f"⚠️ Proveedor {i+1}",
                       hover=f"<b>⚠️ PROVEEDOR CON ALERTAS</b><br><br>"
                             f"<b>NIF:</b> {prov_nif}<br>"
                             f"<b>Facturación:</b> €{prov_amount:,.0f}<br>"
                             f"<b>Alerta:</b> Números redondos detectados<br>"
                             f"<b>Sector:</b> No coincide con actividad declarada",
                       category='suspicious')
            G.add_edge(prov_nif, center_nif, 
                       weight=prov_amount, 
                       color=COLORS['edge_warning'],
                       width=2,
                       category='warning')
        
        # Clientes normales
        for i in range(8):
            cli_nif = f"B{np.random.randint(10000000, 99999999)}"
            cli_amount = np.random.randint(5000, 35000)
            G.add_node(cli_nif, 
                       size=22, 
                       color=COLORS['normal'],
                       symbol='circle',
                       label=f"Cliente {i+1}",
                       hover=f"<b>Cliente</b><br><br>"
                             f"<b>NIF:</b> {cli_nif}<br>"
                             f"<b>Venta:</b> €{cli_amount:,.0f}<br>"
                             f"<b>Estado:</b> ✅ Normal",
                       category='normal')
            G.add_edge(center_nif, cli_nif, 
                       weight=cli_amount, 
                       color=COLORS['edge_normal'],
                       width=1,
                       category='normal')

    else:
        # ──────────────────────────────────────────────────────────────
        # PATRÓN: RED COMERCIAL ESTÁNDAR
        # ──────────────────────────────────────────────────────────────
        
        # Proveedores verificados
        for i in range(np.random.randint(4, 7)):
            prov_nif = f"A{np.random.randint(10000000, 99999999)}"
            prov_amount = np.random.randint(8000, 60000)
            G.add_node(prov_nif, 
                       size=28, 
                       color='#43a047',
                       symbol='triangle-up',
                       label=f"✅ Prov. {i+1}",
                       hover=f"<b>✅ Proveedor Verificado</b><br><br>"
                             f"<b>NIF:</b> {prov_nif}<br>"
                             f"<b>Compra:</b> €{prov_amount:,.0f}<br>"
                             f"<b>Antigüedad:</b> >3 años<br>"
                             f"<b>Estado:</b> Sin alertas",
                       category='normal')
            G.add_edge(prov_nif, center_nif, 
                       weight=prov_amount, 
                       color='#66bb6a',
                       width=2,
                       category='normal')
        
        # Clientes verificados
        for i in range(np.random.randint(6, 10)):
            cli_nif = f"B{np.random.randint(10000000, 99999999)}"
            cli_amount = np.random.randint(2000, 25000)
            G.add_node(cli_nif, 
                       size=24, 
                       color='#43a047',
                       symbol='circle',
                       label=f"✅ Cliente {i+1}",
                       hover=f"<b>✅ Cliente Verificado</b><br><br>"
                             f"<b>NIF:</b> {cli_nif}<br>"
                             f"<b>Venta:</b> €{cli_amount:,.0f}<br>"
                             f"<b>Estado:</b> Relación comercial estable",
                       category='normal')
            G.add_edge(center_nif, cli_nif, 
                       weight=cli_amount, 
                       color='#66bb6a',
                       width=1,
                       category='normal')

    # ═══════════════════════════════════════════════════════════════════
    # LAYOUT OPTIMIZADO PARA VISUALIZACIÓN
    # ═══════════════════════════════════════════════════════════════════
    
    pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42, scale=2)
    
    # ═══════════════════════════════════════════════════════════════════
    # CONSTRUCCIÓN DE TRAZAS PLOTLY
    # ═══════════════════════════════════════════════════════════════════
    
    edge_traces = []
    
    # Procesar aristas
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_color = edge[2].get('color', COLORS['edge_normal'])
        edge_width = edge[2].get('width', 1)
        
        # Línea principal
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=edge_width, color=edge_color),
            hoverinfo='skip',
            showlegend=False
        )
        edge_traces.append(edge_trace)
        
        # Flecha en el medio
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        dx = x1 - x0
        dy = y1 - y0
        angle = np.degrees(np.arctan2(dy, dx))
        
        arrow_trace = go.Scatter(
            x=[mid_x],
            y=[mid_y],
            mode='markers',
            marker=dict(
                symbol='triangle-right',
                size=edge_width * 3 + 6,
                color=edge_color,
                angle=angle,
                line=dict(width=1, color='white')
            ),
            hoverinfo='text',
            hovertext=edge[2].get('hover', ''),
            showlegend=False
        )
        edge_traces.append(arrow_trace)

    # Procesar nodos
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_symbols = []
    node_texts = []
    node_hovers = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(G.nodes[node].get('color', COLORS['normal']))
        node_sizes.append(G.nodes[node].get('size', 20))
        node_symbols.append(G.nodes[node].get('symbol', 'circle'))
        node_texts.append(G.nodes[node].get('label', ''))
        node_hovers.append(G.nodes[node].get('hover', node))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            symbol=node_symbols,
            line=dict(width=2, color='rgba(255,255,255,0.8)'),
            opacity=0.95
        ),
        text=node_texts,
        textposition='top center',
        textfont=dict(size=11, color='white', family='Arial Black'),
        hoverinfo='text',
        hovertext=node_hovers,
        hoverlabel=dict(
            bgcolor='rgba(0,0,0,0.85)',
            bordercolor='white',
            font=dict(size=12, color='white')
        ),
        showlegend=False
    )

    # ═══════════════════════════════════════════════════════════════════
    # CONFIGURACIÓN DEL GRÁFICO
    # ═══════════════════════════════════════════════════════════════════
    
    fig = go.Figure(data=edge_traces + [node_trace])
    
    # Configuración de layout con interactividad completa
    fig.update_layout(
        title=dict(
            text=f"🕸️ Red de Operaciones M347 | <b>{center_nif}</b> | Riesgo: <b>{center_risk}</b>",
            font=dict(size=16, color='white', family='Arial'),
            x=0.5,
            xanchor='center'
        ),
        showlegend=False,
        hovermode='closest',
        plot_bgcolor=COLORS['bg'],
        paper_bgcolor=COLORS['bg'],
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False, 
            visible=False,
            scaleanchor='y',
            scaleratio=1
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False, 
            visible=False
        ),
        height=550,
        # ═════ INTERACTIVIDAD ═════
        dragmode='pan',  # Permite arrastrar para mover el grafo
    )
    
    # Añadir botones de zoom y controles
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.1,
                y=1.12,
                showactive=True,
                buttons=[
                    dict(
                        label="🔍 Zoom +",
                        method="relayout",
                        args=[{"xaxis.range": [-1.5, 1.5], "yaxis.range": [-1.5, 1.5]}]
                    ),
                    dict(
                        label="🔎 Zoom -",
                        method="relayout",
                        args=[{"xaxis.range": [-3, 3], "yaxis.range": [-3, 3]}]
                    ),
                    dict(
                        label="↺ Reset",
                        method="relayout",
                        args=[{"xaxis.autorange": True, "yaxis.autorange": True}]
                    ),
                ],
                bgcolor='rgba(30,30,40,0.8)',
                bordercolor='#555',
                font=dict(color='white', size=11)
            )
        ]
    )
    
    # Configuración del modo de interacción
    fig.update_layout(
        modebar=dict(
            bgcolor='rgba(0,0,0,0.5)',
            color='white',
            activecolor='#00e5ff'
        ),
        modebar_add=[
            'zoom2d', 'pan2d', 'select2d', 'lasso2d', 
            'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'
        ]
    )
    
    # Anotaciones de importes sobre las aristas principales
    annotations = []
    for edge in G.edges(data=True):
        if edge[2].get('category') in ['fraud', 'warning'] and 'label' in edge[2]:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            annotations.append(
                dict(
                    x=(x0 + x1) / 2,
                    y=(y0 + y1) / 2 + 0.15,
                    text=f"<b>{edge[2]['label']}</b>",
                    showarrow=False,
                    font=dict(size=10, color='#ffeb3b', family='Arial Black'),
                    bgcolor='rgba(0,0,0,0.7)',
                    bordercolor=edge[2].get('color', '#fff'),
                    borderwidth=1,
                    borderpad=3
                )
            )
    
    # Leyenda manual
    legend_y = 0.95
    legend_items = [
        ('🎯 Empresa Objetivo', target_color),
        ('🏭 Empresa Pantalla', COLORS['shell']),
        ('⚠️ Entidad Sospechosa', COLORS['suspicious']),
        ('✅ Entidad Normal', COLORS['normal']),
    ]
    
    for text, color in legend_items:
        annotations.append(
            dict(
                x=1.02,
                y=legend_y,
                xref='paper',
                yref='paper',
                text=f"<b>{text}</b>",
                showarrow=False,
                font=dict(size=10, color=color),
                xanchor='left'
            )
        )
        legend_y -= 0.07
    
    fig.update_layout(annotations=annotations)
    
    return fig
