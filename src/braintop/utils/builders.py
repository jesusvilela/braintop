"""Builder utilities for creating topologies."""

from typing import List, Optional, Dict, Any
import numpy as np

from braintop.core.topology import Topology, LayerModel, EdgeModel
from braintop.layers.conceptual import ConceptualLayer
from braintop.layers.riemannian import RiemannianLayer
from braintop.integrations.llm_connector import LLMConnector


class TopologyBuilder:
    """Builder class for constructing complex topologies.
    
    This class provides a fluent interface for creating multi-layer
    topologies with various configurations.
    """
    
    def __init__(self, topology_id: str, name: str):
        """Initialize the builder.
        
        Args:
            topology_id: Unique identifier for the topology
            name: Human-readable name
        """
        self.topology = Topology(
            id=topology_id,
            name=name,
            layers=[],
            inter_layer_edges=[],
            metadata={}
        )
        self._edge_counter = 0
    
    def add_conceptual_layer(
        self,
        layer_id: str,
        num_nodes: int,
        concepts: Optional[List[str]] = None,
        embeddings: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> "TopologyBuilder":
        """Add a conceptual layer to the topology.
        
        Args:
            layer_id: Unique identifier for the layer
            num_nodes: Number of nodes in the layer
            concepts: Optional concept labels
            embeddings: Optional embeddings
            **kwargs: Additional configuration
            
        Returns:
            Self for chaining
        """
        layer = ConceptualLayer(layer_id)
        
        # Apply configuration
        if kwargs:
            layer.set_config(**kwargs)
        
        # Create layer with nodes
        layer_model = layer.create_layer_with_nodes(num_nodes, concepts, embeddings)
        self.topology.add_layer(layer_model)
        
        return self
    
    def add_riemannian_layer(
        self,
        layer_id: str,
        num_nodes: int,
        manifold_type: str = "sphere",
        labels: Optional[List[str]] = None,
        **kwargs: Any
    ) -> "TopologyBuilder":
        """Add a Riemannian layer to the topology.
        
        Args:
            layer_id: Unique identifier for the layer
            num_nodes: Number of nodes in the layer
            manifold_type: Type of manifold ('sphere', 'torus', 'hyperbolic')
            labels: Optional node labels
            **kwargs: Additional configuration
            
        Returns:
            Self for chaining
        """
        layer = RiemannianLayer(layer_id)
        layer.set_config(manifold_type=manifold_type, **kwargs)
        
        # Create layer with nodes
        layer_model = layer.create_layer_with_nodes(num_nodes, labels)
        self.topology.add_layer(layer_model)
        
        return self
    
    def add_layer_from_llm(
        self,
        layer_id: str,
        llm_connector: LLMConnector,
        texts: List[str],
        **kwargs: Any
    ) -> "TopologyBuilder":
        """Add a layer created from LLM embeddings.
        
        Args:
            layer_id: Unique identifier for the layer
            llm_connector: Connected LLM connector
            texts: Input texts to visualize
            **kwargs: Additional configuration
            
        Returns:
            Self for chaining
        """
        # Get topology data from LLM
        llm_data = llm_connector.create_topology_from_texts(texts, layer_id)
        
        # Create conceptual layer with LLM embeddings
        layer = ConceptualLayer(layer_id)
        if kwargs:
            layer.set_config(**kwargs)
        
        layer_model = layer.create_layer_with_nodes(
            num_nodes=llm_data["num_nodes"],
            concepts=llm_data["labels"],
            embeddings=llm_data["embeddings"]
        )
        
        self.topology.add_layer(layer_model)
        
        # Add edges from LLM relationships
        for edge_data in llm_data["edges"]:
            source_id = f"{layer_id}_node_{edge_data['source']}"
            target_id = f"{layer_id}_node_{edge_data['target']}"
            
            edge = EdgeModel(
                id=f"edge_{self._edge_counter}",
                source=source_id,
                target=target_id,
                weight=edge_data["weight"]
            )
            self.topology.add_inter_layer_edge(edge)
            self._edge_counter += 1
        
        return self
    
    def connect_layers(
        self,
        source_layer_id: str,
        target_layer_id: str,
        connection_type: str = "nearest",
        num_connections: int = 5
    ) -> "TopologyBuilder":
        """Create connections between two layers.
        
        Args:
            source_layer_id: Source layer ID
            target_layer_id: Target layer ID
            connection_type: Type of connection ('nearest', 'random', 'all')
            num_connections: Number of connections per node
            
        Returns:
            Self for chaining
        """
        source_layer = self.topology.get_layer(source_layer_id)
        target_layer = self.topology.get_layer(target_layer_id)
        
        if not source_layer or not target_layer:
            raise ValueError("Invalid layer IDs")
        
        if connection_type == "nearest":
            self._connect_nearest(source_layer, target_layer, num_connections)
        elif connection_type == "random":
            self._connect_random(source_layer, target_layer, num_connections)
        elif connection_type == "all":
            self._connect_all(source_layer, target_layer)
        
        return self
    
    def _connect_nearest(
        self,
        source_layer: LayerModel,
        target_layer: LayerModel,
        k: int
    ) -> None:
        """Connect each source node to k nearest target nodes."""
        for source_node in source_layer.nodes:
            # Compute distances to all target nodes
            distances = []
            for target_node in target_layer.nodes:
                dist = self._euclidean_distance(
                    source_node.position,
                    target_node.position
                )
                distances.append((dist, target_node))
            
            # Sort by distance and connect to k nearest
            distances.sort(key=lambda x: x[0])
            for dist, target_node in distances[:k]:
                edge = EdgeModel(
                    id=f"edge_{self._edge_counter}",
                    source=source_node.id,
                    target=target_node.id,
                    weight=1.0 / (1.0 + dist)  # Inverse distance weight
                )
                self.topology.add_inter_layer_edge(edge)
                self._edge_counter += 1
    
    def _connect_random(
        self,
        source_layer: LayerModel,
        target_layer: LayerModel,
        num_connections: int
    ) -> None:
        """Connect each source node to random target nodes."""
        for source_node in source_layer.nodes:
            # Randomly select target nodes
            target_indices = np.random.choice(
                len(target_layer.nodes),
                size=min(num_connections, len(target_layer.nodes)),
                replace=False
            )
            
            for idx in target_indices:
                target_node = target_layer.nodes[idx]
                edge = EdgeModel(
                    id=f"edge_{self._edge_counter}",
                    source=source_node.id,
                    target=target_node.id,
                    weight=1.0
                )
                self.topology.add_inter_layer_edge(edge)
                self._edge_counter += 1
    
    def _connect_all(
        self,
        source_layer: LayerModel,
        target_layer: LayerModel
    ) -> None:
        """Connect all source nodes to all target nodes."""
        for source_node in source_layer.nodes:
            for target_node in target_layer.nodes:
                edge = EdgeModel(
                    id=f"edge_{self._edge_counter}",
                    source=source_node.id,
                    target=target_node.id,
                    weight=1.0
                )
                self.topology.add_inter_layer_edge(edge)
                self._edge_counter += 1
    
    @staticmethod
    def _euclidean_distance(
        pos1: tuple,
        pos2: tuple
    ) -> float:
        """Compute Euclidean distance between two positions."""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
    
    def set_metadata(self, **kwargs: Any) -> "TopologyBuilder":
        """Set metadata for the topology.
        
        Args:
            **kwargs: Metadata key-value pairs
            
        Returns:
            Self for chaining
        """
        self.topology.metadata.update(kwargs)
        return self
    
    def build(self) -> Topology:
        """Build and return the topology.
        
        Returns:
            Constructed Topology object
        """
        return self.topology
