"""Core topology data structures and models."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field

from braintop.core.frames import RigidFrame3D, Matrix3, Vector3


class LayerType(str, Enum):
    """Types of topology layers."""
    CONCEPTUAL = "conceptual"
    RIEMANNIAN = "riemannian"
    HYPERBOLIC = "hyperbolic"
    EUCLIDEAN = "euclidean"
    CUSTOM = "custom"


class NodeModel(BaseModel):
    """Model for a node in the topology."""
    id: str = Field(..., description="Unique identifier for the node")
    position: Tuple[float, float, float] = Field(..., description="3D position (x, y, z)")
    label: str = Field(default="", description="Human-readable label")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    layer_id: str = Field(..., description="ID of the layer this node belongs to")


class EdgeModel(BaseModel):
    """Model for an edge in the topology."""
    id: str = Field(..., description="Unique identifier for the edge")
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    weight: float = Field(default=1.0, description="Edge weight")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class LayerModel(BaseModel):
    """Model for a topology layer."""
    id: str = Field(..., description="Unique identifier for the layer")
    name: str = Field(..., description="Human-readable name")
    layer_type: LayerType = Field(..., description="Type of layer")
    nodes: List[NodeModel] = Field(default_factory=list, description="Nodes in this layer")
    config: Dict[str, Any] = Field(default_factory=dict, description="Layer-specific configuration")
    frame: RigidFrame3D = Field(default_factory=RigidFrame3D, description="Local-to-world display-coordinate frame")


@dataclass
class Topology:
    """Main topology structure containing layers and their relationships.
    
    This class represents a complete 3D topology that can be visualized.
    It supports multiple layers of different types and inter-layer connections.
    """
    
    id: str
    name: str
    layers: List[LayerModel] = field(default_factory=list)
    inter_layer_edges: List[EdgeModel] = field(default_factory=list)
    intra_layer_edges: List[EdgeModel] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_layer(self, layer: LayerModel) -> None:
        """Add a layer, rejecting ambiguous layer and node identifiers."""
        if self.get_layer(layer.id) is not None:
            raise ValueError(f"Duplicate layer ID: {layer.id}")
        existing = {node.id for node in self.get_all_nodes()}
        incoming = [node.id for node in layer.nodes]
        if len(incoming) != len(set(incoming)) or existing.intersection(incoming):
            raise ValueError("Node IDs must be unique across the whole topology")
        if any(node.layer_id != layer.id for node in layer.nodes):
            raise ValueError("Each node's layer_id must match its containing layer")
        self.layers.append(layer)

    def _edge_endpoint_layers(self, edge: EdgeModel) -> Tuple[str, str]:
        node_to_layer = {
            node.id: node.layer_id for node in self.get_all_nodes()
        }
        if edge.source not in node_to_layer or edge.target not in node_to_layer:
            raise ValueError(
                f"Edge endpoints must refer to existing nodes: {edge.source}, {edge.target}"
            )
        if any(
            existing_edge.id == edge.id for existing_edge in self.get_all_edges()
        ):
            raise ValueError(f"Duplicate edge ID: {edge.id}")
        return node_to_layer[edge.source], node_to_layer[edge.target]

    def add_inter_layer_edge(self, edge: EdgeModel) -> None:
        """Add an edge joining nodes in *different* layers."""
        source_layer, target_layer = self._edge_endpoint_layers(edge)
        if source_layer == target_layer:
            raise ValueError("Inter-layer edge must connect distinct layers")
        self.inter_layer_edges.append(edge)

    def add_intra_layer_edge(self, edge: EdgeModel) -> None:
        """Add an edge joining nodes in the *same* layer."""
        source_layer, target_layer = self._edge_endpoint_layers(edge)
        if source_layer != target_layer:
            raise ValueError("Intra-layer edge endpoints must share a layer")
        self.intra_layer_edges.append(edge)

    def world_position(self, node: NodeModel) -> Vector3:
        """Ambient 3D position using the node's validated layer frame."""
        layer = self.get_layer(node.layer_id)
        if layer is None:
            raise ValueError(f"Unknown layer for node {node.id}: {node.layer_id}")
        return layer.frame.to_world(node.position)

    def reexpress_layer(
        self, layer_id: str, rotation: Matrix3, translation: Vector3
    ) -> None:
        """Perform a *passive* change of local coordinates y=Qx+s.

        Neither world positions nor stored graph edges change. To physically
        move a layer instead, update layer.frame (and rebuild nearest edges
        if the application needs neighbours recomputed).
        """
        layer = self.get_layer(layer_id)
        if layer is None:
            raise ValueError(f"Unknown layer: {layer_id}")
        frame = layer.frame.reparameterized(rotation, translation)
        q = np.asarray(rotation, dtype=float)
        s = np.asarray(translation, dtype=float)
        new_positions = [
            tuple(float(v) for v in q @ np.asarray(node.position) + s)
            for node in layer.nodes
        ]
        for node, position in zip(layer.nodes, new_positions):
            node.position = position
        layer.frame = frame
    
    def get_layer(self, layer_id: str) -> Optional[LayerModel]:
        """Get a layer by its ID."""
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        return None
    
    def get_all_nodes(self) -> List[NodeModel]:
        """Get all nodes from all layers."""
        nodes = []
        for layer in self.layers:
            nodes.extend(layer.nodes)
        return nodes
    
    def get_all_edges(self) -> List[EdgeModel]:
        """Get all edges including inter-layer edges."""
        return self.inter_layer_edges.copy() + self.intra_layer_edges.copy()
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute topology statistics."""
        total_nodes = sum(len(layer.nodes) for layer in self.layers)
        total_edges = len(self.get_all_edges())
        
        return {
            "num_layers": len(self.layers),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "inter_layer_edges": len(self.inter_layer_edges),
            "intra_layer_edges": len(self.intra_layer_edges),
            "layer_types": [layer.layer_type for layer in self.layers],
        }

    def compute_layer_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Compute statistics for each layer."""
        node_layer_map = {
            node.id: node.layer_id for layer in self.layers for node in layer.nodes
        }
        edge_counts: Dict[str, int] = {layer.id: 0 for layer in self.layers}
        for edge in self.get_all_edges():
            for node_id in (edge.source, edge.target):
                layer_id = node_layer_map.get(node_id)
                if layer_id:
                    edge_counts[layer_id] += 1

        stats: Dict[str, Dict[str, Any]] = {}
        for layer in self.layers:
            node_count = len(layer.nodes)
            edge_count = edge_counts.get(layer.id, 0)
            avg_degree = edge_count / node_count if node_count else 0.0
            stats[layer.id] = {
                "layer_name": layer.name,
                "layer_type": layer.layer_type.value,
                "node_count": node_count,
                "edge_count": edge_count,
                "avg_degree": round(avg_degree, 3),
            }

        return stats
