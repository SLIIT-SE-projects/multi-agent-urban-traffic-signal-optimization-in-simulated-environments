import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData

def plot_graph_topology(data: HeteroData, show_lanes=False):

    plt.figure(figsize=(12, 12))
    
    # 1. Create a NetworkX Graph for Intersections
    G = nx.Graph()
    
    # Get positions from the data object
    pos_dict = {}
    num_intersections = data['intersection'].num_nodes
    
    for i in range(num_intersections):
        x, y = data['intersection'].pos[i].tolist()
        pos_dict[i] = (x, y)
        G.add_node(i)

    # Add Edges ('intersection', 'adjacent_to', 'intersection')
    edge_index = data['intersection', 'adjacent_to', 'intersection'].edge_index
    edges = edge_index.t().tolist() # Convert to list of [src, dst]
    G.add_edges_from(edges)

    # 2. Draw the Topology
    print("Drawing Network Topology...")
    
    # Draw Edges (Roads connecting intersections)
    nx.draw_networkx_edges(G, pos_dict, alpha=0.5, edge_color='gray')
    
    # Draw Nodes (Intersections)
    nx.draw_networkx_nodes(G, pos_dict, node_size=50, node_color='red', label='Intersections')

    # Optional: Draw Lanes as smaller dots
    if show_lanes:
        # This part would require passing lane positions too
        pass 

    plt.title(f"Traffic Network Topology\n({num_intersections} Intersections)")
    plt.axis('off') # Hide axis ticks
    plt.tight_layout()
    plt.show()

# view function version 2

def plot_graph_topology_ver_2(data: HeteroData):

    plt.figure(figsize=(14, 14), facecolor='white')
    ax = plt.gca()
    
    # --- 1. Setup Positions ---
    pos = {}
    
    # Intersections (Indices 0 to N-1)
    num_inter = data['intersection'].num_nodes
    for i in range(num_inter):
        pos[f"i_{i}"] = data['intersection'].pos[i].tolist()
        
    # Lanes (Indices 0 to M-1)
    num_lanes = data['lane'].num_nodes
    for i in range(num_lanes):

        pos[f"l_{i}"] = data['lane'].pos[i].tolist()

    # --- 2. Draw Edges (Layers) ---

    # Layer 1: Intersection -> Intersection (Adjacency) Orange
    if ('intersection', 'adjacent_to', 'intersection') in data.edge_index_dict:
        edge_index = data['intersection', 'adjacent_to', 'intersection'].edge_index
        edges = [(f"i_{u}", f"i_{v}") for u, v in edge_index.t().tolist()]
        
        nx.draw_networkx_edges(
            nx.Graph(), pos, edgelist=edges, ax=ax,
            edge_color='orange', width=2.0, alpha=0.6, 
            arrows=False, style='dashed'
        )

    # Layer 2: Lane -> Lane (Flow) Purple
    if ('lane', 'feeds_into', 'lane') in data.edge_index_dict:
        edge_index = data['lane', 'feeds_into', 'lane'].edge_index
        edges = [(f"l_{u}", f"l_{v}") for u, v in edge_index.t().tolist()]
        
        # Only draw a subset if there are too many (optimization for rendering)
        if len(edges) > 5000: edges = edges[::2] 
        
        nx.draw_networkx_edges(
            nx.Graph(), pos, edgelist=edges, ax=ax,
            edge_color='purple', width=0.5, alpha=0.3, arrows=True
        )

    # Layer 3: Lane -> Intersection (Part Of) Green
    if ('lane', 'part_of', 'intersection') in data.edge_index_dict:
        edge_index = data['lane', 'part_of', 'intersection'].edge_index
        edges = [(f"l_{u}", f"i_{v}") for u, v in edge_index.t().tolist()]
        
        nx.draw_networkx_edges(
            nx.Graph(), pos, edgelist=edges, ax=ax,
            edge_color='green', width=1.0, alpha=0.4
        )

    # --- 3. Draw Nodes (Layers) ---

    # Layer 4: Lane Nodes Blue
    lane_nodes = [f"l_{i}" for i in range(num_lanes)]
    nx.draw_networkx_nodes(
        nx.Graph(), pos, nodelist=lane_nodes, ax=ax,
        node_size=10, node_color='blue', alpha=0.6, label='Lanes'
    )

    # Layer 5: Intersection Nodes Red
    inter_nodes = [f"i_{i}" for i in range(num_inter)]
    nx.draw_networkx_nodes(
        nx.Graph(), pos, nodelist=inter_nodes, ax=ax,
        node_size=100, node_color='red', edgecolors='black', label='Intersections'
    )

    # --- 4. Legend & formatting ---
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Intersections'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Lanes'),
        Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Adj (Int->Int)'),
        Line2D([0], [0], color='purple', lw=1, label='Flow (Lane->Lane)'),
        Line2D([0], [0], color='green', lw=1, label='Part Of (Lane->Int)'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper right')
    plt.title(f"Traffic Graph Structure\n{num_inter} Intersections | {num_lanes} Lanes")
    plt.axis('off')
    plt.tight_layout()
    plt.show()