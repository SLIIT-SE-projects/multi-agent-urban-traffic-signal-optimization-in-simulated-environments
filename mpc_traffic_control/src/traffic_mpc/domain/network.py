"""
Domain Entity: TrafficNetwork
Represents the physical road topology using a NetworkX graph.
"""
import networkx as nx
from pydantic import BaseModel, Field, PositiveFloat
from typing import List, Dict, Tuple

class LinkAttribute(BaseModel):
    """
    Physical attributes of a road link (edge).
    Validated using Pydantic V2.
    """
    id: str
    length_meters: PositiveFloat = Field(..., description="Physical length")
    capacity_vph: PositiveFloat = Field(..., description="Saturation flow (veh/hour)")
    free_flow_speed: PositiveFloat = Field(..., description="Speed limit (m/s)")
    num_lanes: int = Field(default=1, ge=1)

class TrafficNetwork:
    """
    Wrapper around NetworkX DiGraph to enforce type safety.
    """
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_intersection(self, node_id: str, pos: Tuple[float, float]):
        """Adds a node (intersection) with geospatial coordinates."""
        self.graph.add_node(node_id, pos=pos)

    def add_link(self, u: str, v: str, attr: LinkAttribute):
        """Adds a directed edge (road) with validated attributes."""
        # Store the Pydantic model directly in the edge data
        self.graph.add_edge(u, v, object=attr)

    def get_link_data(self, u: str, v: str) -> LinkAttribute:
        """Retrieves the LinkAttribute object safely."""
        return self.graph[u][v]['object']

    @property
    def number_of_nodes(self):
        return self.graph.number_of_nodes()

    def validate_topology(self):
        """
        Sanity check: Ensure graph is connected and valid.
        """
        if not nx.is_weakly_connected(self.graph):
            raise ValueError("Traffic network is disjoint (not connected)!")