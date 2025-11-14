"""Core topology data structures and models."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field


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
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_layer(self, layer: LayerModel) -> None:
        """Add a layer to the topology."""
        self.layers.append(layer)
    
    def add_inter_layer_edge(self, edge: EdgeModel) -> None:
        """Add an edge connecting nodes across different layers."""
        self.inter_layer_edges.append(edge)
    
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
        return self.inter_layer_edges.copy()
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute topology statistics."""
        total_nodes = sum(len(layer.nodes) for layer in self.layers)
        total_edges = len(self.inter_layer_edges)
        
        return {
            "num_layers": len(self.layers),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "layer_types": [layer.layer_type for layer in self.layers],
        }
