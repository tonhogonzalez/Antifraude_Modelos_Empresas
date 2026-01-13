
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import random

def create_suspicious_network(center_nif, center_risk, center_score):
    """
    Genera un subgrafo transaccional simulado alrededor de una empresa.
    Si la empresa es de alto riesgo, genera patrones de fraude (circularidad, hubs).
    """
    G = nx.DiGraph()
    
    # Nodo central
    color_map = {'Alto': '#ff4b4b', 'Medio': '#ffa500', 'Bajo': '#00cc96'}
    center_color = color_map.get(center_risk, '#00cc96')
    
    G.add_node(center_nif, size=25, color=center_color, label=f"TARGET\n{center_nif}", type='target')
    
    # Determinar patrón basado en riesgo
    np.random.seed(hash(center_nif) % 2**32)
    
    neighbors = []
    
    if center_risk == 'Alto':
        # PATRÓN 1: CARRUSEL (Circularidad A->B->C->A)
        # Crear 3 nodos testaferros
        n1 = f"T{np.random.randint(1000,9999)}"
        n2 = f"T{np.random.randint(1000,9999)}"
        n3 = f"T{np.random.randint(1000,9999)}"
        
        G.add_node(n1, size=15, color='#ff4b4b', label="Shell Co.", type='fraud')
        G.add_node(n2, size=15, color='#ff4b4b', label="Shell Co.", type='fraud')
        G.add_node(n3, size=15, color='#ff4b4b', label="Shell Co.", type='fraud')
        
        # Conexiones circulares fuertes
        amount = np.random.randint(500000, 2000000)
        G.add_edge(center_nif, n1, weight=amount, label=f"€{amount/1000:.0f}k")
        G.add_edge(n1, n2, weight=amount, label=f"€{amount/1000:.0f}k")
        G.add_edge(n2, n3, weight=amount, label=f"€{amount/1000:.0f}k")
        G.add_edge(n3, center_nif, weight=amount, label=f"€{amount/1000:.0f}k")
        
        neighbors.extend([n1, n2, n3])
        
        # Ruido adicional (clientes reales para disimular)
        for i in range(3):
            client = f"C{np.random.randint(10000,99999)}"
            G.add_node(client, size=10, color='#888', label="Cliente", type='normal')
            G.add_edge(center_nif, client, weight=np.random.randint(1000, 5000))

    elif center_risk == 'Medio':
        # PATRÓN 2: HUB (Muchas conexiones pequeñas, algunas sospechosas)
        # Proveedores sospechosos
        for i in range(2):
            prov = f"P{np.random.randint(1000,9999)}"
            G.add_node(prov, size=12, color='#ffa500', label="Prov. Susp.", type='warning')
            G.add_edge(prov, center_nif, weight=np.random.randint(10000, 50000))
        
        # Clientes normales
        for i in range(5):
            cli = f"C{np.random.randint(1000,9999)}"
            G.add_node(cli, size=10, color='#888', label="Cliente", type='normal')
            G.add_edge(center_nif, cli, weight=np.random.randint(5000, 20000))
            
    else:
        # PATRÓN 3: ESTÁNDAR (Relaciones normales)
        # Proveedores
        for i in range(np.random.randint(3, 6)):
            prov = f"P_REAL_{i}"
            G.add_node(prov, size=10, color='#00cc96', label="Proveedor", type='normal')
            G.add_edge(prov, center_nif, weight=np.random.randint(5000, 50000))
            
        # Clientes
        for i in range(np.random.randint(5, 10)):
            cli = f"C_REAL_{i}"
            G.add_node(cli, size=10, color='#00cc96', label="Cliente", type='normal')
            G.add_edge(center_nif, cli, weight=np.random.randint(2000, 15000))

    # --- PLOTLY VISUALIZATION ---
    pos = nx.spring_layout(G, seed=42)
    
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        # Flecha indicativa simple (solo línea por ahora)
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#555'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(G.nodes[node].get('label', node))
        node_color.append(G.nodes[node].get('color', '#888'))
        node_size.append(G.nodes[node].get('size', 10))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[G.nodes[n].get('label', '') for n in G.nodes()],
        textposition="top center",
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=dict(
                        text=f'Grafo de Relaciones M347 - {center_nif}',
                        font=dict(size=16)
                    ),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    
    # Anotaciones para edges (importes)
    annotations = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        if 'label' in edge[2]:
            annotations.append(
                dict(
                    x=(x0+x1)/2,
                    y=(y0+y1)/2,
                    xref='x',
                    yref='y',
                    text=edge[2]['label'],
                    showarrow=False,
                    font=dict(size=8, color="#aaa"),
                    bgcolor="#0e1117"
                )
            )
    fig.update_layout(annotations=annotations)
    
    return fig
