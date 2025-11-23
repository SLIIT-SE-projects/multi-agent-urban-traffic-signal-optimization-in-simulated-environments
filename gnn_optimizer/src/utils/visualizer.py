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