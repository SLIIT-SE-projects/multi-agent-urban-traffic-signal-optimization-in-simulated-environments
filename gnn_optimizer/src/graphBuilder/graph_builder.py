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

    def create_hetero_data(self, snapshot):
        
        data = HeteroData()
        
        # --- 1. Node Features (Dynamic) ---
        
        # A. Intersection Features & Positions
        x_inter = torch.zeros((self.num_intersections, 5), dtype=torch.float)
        pos_inter = [[0.0, 0.0] for _ in range(self.num_intersections)]
        
        for tls_id, info in snapshot['intersections'].items():
            if tls_id in self.tls_map:
                idx = self.tls_map[tls_id]
                # Features
                p_idx = int(info['phase_index']) % 4
                x_inter[idx, p_idx] = 1.0 
                x_inter[idx, 4] = float(info['time_to_switch'])
                # Position
                sumo_node = self.net.getNode(tls_id)
                if sumo_node:
                    x, y = sumo_node.getCoord()
                    pos_inter[idx] = [x, y]

        data['intersection'].x = x_inter
        data['intersection'].pos = torch.tensor(pos_inter, dtype=torch.float)

        # B. Lane Features & Positions
        x_lane = torch.zeros((self.num_lanes, 2), dtype=torch.float)
        pos_lane = [[0.0, 0.0] for _ in range(self.num_lanes)]

        for lane_id, info in snapshot['lanes'].items():
            if lane_id in self.lane_map:
                idx = self.lane_map[lane_id]
                # Features
                x_lane[idx, 0] = float(info['queue_length'])
                x_lane[idx, 1] = float(info['avg_speed'])
                
                # Position (Calculate Center of Lane)
                try:
                    lane_shape = self.net.getLane(lane_id).getShape()
                    if lane_shape:
                        # Calculate simple centroid (average of all shape points)
                        avg_x = sum(p[0] for p in lane_shape) / len(lane_shape)
                        avg_y = sum(p[1] for p in lane_shape) / len(lane_shape)
                        pos_lane[idx] = [avg_x, avg_y]
                except:
                    pass # Keep default 0.0 if shape retrieval fails

        data['lane'].x = x_lane
        data['lane'].pos = torch.tensor(pos_lane, dtype=torch.float)

        # --- 2. Edges (Static) ---
        if self.static_edges['part_of'].numel() > 0:
            data['lane', 'part_of', 'intersection'].edge_index = self.static_edges['part_of']
        
        if self.static_edges['adjacent'].numel() > 0:
            data['intersection', 'adjacent_to', 'intersection'].edge_index = self.static_edges['adjacent']
            
        if self.static_edges['feeds'].numel() > 0:
            data['lane', 'feeds_into', 'lane'].edge_index = self.static_edges['feeds']

        return data