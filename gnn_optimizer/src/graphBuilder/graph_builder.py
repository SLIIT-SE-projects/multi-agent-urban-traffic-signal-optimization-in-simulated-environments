import torch
from torch_geometric.data import HeteroData
import sumolib
import numpy as np

class TrafficGraphBuilder:
    def __init__(self, net_file_path):
        
        print(f"Loading Network: {net_file_path}")
        # withNodeNeighbors=True ensures we can traverse the graph topology
        self.net = sumolib.net.readNet(net_file_path, withNodeNeighbors=True)
        
        # --- 1. Identify Nodes (Intersections & Lanes) ---
        
        # Filter for nodes that are actually Traffic Lights
        self.tls_nodes = [n for n in self.net.getNodes() if n.getType() == "traffic_light"]
        
        # FIX: Get all lanes by iterating through all edges first
        self.all_lanes = []
        for edge in self.net.getEdges():
            # Filter out internal function lanes (usually start with ':')
            # if edge.getFunction() == 'internal': continue
            self.all_lanes.extend(edge.getLanes())

        # Create Mappings: String ID -> Integer Index
        self.tls_map = {node.getID(): i for i, node in enumerate(self.tls_nodes)}
        self.lane_map = {lane.getID(): i for i, lane in enumerate(self.all_lanes)}
        
        self.num_intersections = len(self.tls_nodes)
        self.num_lanes = len(self.all_lanes)
        
        print(f"Graph Initialized: {self.num_intersections} Intersections, {self.num_lanes} Lanes.")

        # --- 2. Build Static Edges ---
        self.static_edges = self._build_static_topology()

    def _build_static_topology(self):

        src_part_of, dst_part_of = [], []
        src_adj, dst_adj = [], []
        src_feed, dst_feed = [], []

        # A. Lane -> Intersection Connectivity ('part_of')
        for lane in self.all_lanes:
            lane_id = lane.getID()
            edge = lane.getEdge()
            # The edge points TO a node. That node is the intersection this lane feeds.
            target_node = edge.getToNode()
            
            if target_node.getType() == "traffic_light" and target_node.getID() in self.tls_map:
                l_idx = self.lane_map[lane_id]
                i_idx = self.tls_map[target_node.getID()]
                
                src_part_of.append(l_idx)
                dst_part_of.append(i_idx)

        # B. Intersection -> Intersection Topology ('adjacent_to')
        for node in self.tls_nodes:
            u_idx = self.tls_map[node.getID()]
            # Check outgoing edges to find neighbor intersections
            for outgoing_edge in node.getOutgoing():
                neighbor_node = outgoing_edge.getToNode()
                if neighbor_node.getType() == "traffic_light" and neighbor_node.getID() in self.tls_map:
                    v_idx = self.tls_map[neighbor_node.getID()]
                    src_adj.append(u_idx)
                    dst_adj.append(v_idx)
        
        # C. Lane -> Lane Flow ('feeds_into')
        for lane in self.all_lanes:
            if lane.getID() not in self.lane_map: continue
            l_from_idx = self.lane_map[lane.getID()]
            
            # getOutgoing returns a list of Connection objects
            for conn in lane.getOutgoing():
                to_lane = conn.getToLane()
                if to_lane.getID() in self.lane_map:
                    l_to_idx = self.lane_map[to_lane.getID()]
                    src_feed.append(l_from_idx)
                    dst_feed.append(l_to_idx)

        return {
            'part_of': torch.tensor([src_part_of, dst_part_of], dtype=torch.long),
            'adjacent': torch.tensor([src_adj, dst_adj], dtype=torch.long),
            'feeds': torch.tensor([src_feed, dst_feed], dtype=torch.long)
        }