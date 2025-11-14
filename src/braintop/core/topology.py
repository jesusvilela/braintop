"""
Core topology abstractions and base classes.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import networkx as nx


class TopologyLayer(ABC):
    """
    Abstract base class for all topology layers.
    
    Each layer represents a different mathematical or conceptual
    perspective on the topology structure.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize a topology layer.
        
        Args:
            name: Unique identifier for the layer
            **kwargs: Additional layer-specific parameters
        """
        self.name = name
        self.config = kwargs
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._edges: List[Tuple[str, str, Dict[str, Any]]] = []
        
    @abstractmethod
    def compute_positions(self) -> Dict[str, np.ndarray]:
        """
        Compute 3D positions for all nodes in this layer.
        
        Returns:
            Dictionary mapping node IDs to 3D position arrays
        """
        pass
    
    @abstractmethod
    def compute_geometry(self) -> Dict[str, Any]:
        """
        Compute layer-specific geometric properties.
        
        Returns:
            Dictionary containing geometric metadata
        """
        pass
    
    def add_node(self, node_id: str, **attributes):
        """
        Add a node to the topology layer.
        
        Args:
            node_id: Unique identifier for the node
            **attributes: Node attributes (e.g., label, weight, type)
        """
        self._nodes[node_id] = attributes
        
    def add_edge(self, source: str, target: str, **attributes):
        """
        Add an edge between two nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            **attributes: Edge attributes (e.g., weight, type)
        """
        self._edges.append((source, target, attributes))
    
    def get_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Get all nodes in the layer."""
        return self._nodes.copy()
    
    def get_edges(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get all edges in the layer."""
        return self._edges.copy()
    
    def clear(self):
        """Clear all nodes and edges from the layer."""
        self._nodes.clear()
        self._edges.clear()


class TopologyGraph:
    """
    Container for multiple topology layers, representing different
    perspectives on the same underlying structure.
    """
    
    def __init__(self):
        """Initialize an empty topology graph."""
        self._layers: Dict[str, TopologyLayer] = {}
        self._metadata: Dict[str, Any] = {}
        
    def add_layer(self, layer: TopologyLayer):
        """
        Add a topology layer to the graph.
        
        Args:
            layer: The topology layer to add
        """
        self._layers[layer.name] = layer
        
    def get_layer(self, name: str) -> Optional[TopologyLayer]:
        """
        Retrieve a layer by name.
        
        Args:
            name: Layer identifier
            
        Returns:
            The requested layer or None if not found
        """
        return self._layers.get(name)
    
    def get_all_layers(self) -> Dict[str, TopologyLayer]:
        """Get all layers in the graph."""
        return self._layers.copy()
    
    def remove_layer(self, name: str):
        """
        Remove a layer from the graph.
        
        Args:
            name: Layer identifier to remove
        """
        if name in self._layers:
            del self._layers[name]
    
    def set_metadata(self, key: str, value: Any):
        """Set metadata for the topology graph."""
        self._metadata[key] = value
    
    def get_metadata(self, key: str) -> Optional[Any]:
        """Get metadata from the topology graph."""
        return self._metadata.get(key)
    
    def to_networkx(self, layer_name: Optional[str] = None) -> nx.Graph:
        """
        Convert a layer to a NetworkX graph for analysis.
        
        Args:
            layer_name: Name of the layer to convert (uses first layer if None)
            
        Returns:
            NetworkX graph representation
        """
        if layer_name is None:
            if not self._layers:
                return nx.Graph()
            layer_name = list(self._layers.keys())[0]
        
        layer = self._layers.get(layer_name)
        if layer is None:
            return nx.Graph()
        
        G = nx.Graph()
        
        # Add nodes with attributes
        for node_id, attrs in layer.get_nodes().items():
            G.add_node(node_id, **attrs)
        
        # Add edges with attributes
        for source, target, attrs in layer.get_edges():
            G.add_edge(source, target, **attrs)
        
        return G
