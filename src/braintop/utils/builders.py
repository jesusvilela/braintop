"""Builder utilities for creating topologies."""

from typing import List, Optional, Dict, Any
import numpy as np

from braintop.core.topology import Topology, LayerModel, EdgeModel
from braintop.core.intrinsic import nearest_intrinsic_pairs
from braintop.core.correspondence import RadialCapMap, compare_radial_cap
from braintop.core.frames import RigidFrame3D, Matrix3, Vector3
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
        parameters: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> "TopologyBuilder":
        """Add a Riemannian layer to the topology.
        
        Args:
            layer_id: Unique identifier for the layer
            num_nodes: Number of nodes in the layer
            manifold_type: Type of manifold ('sphere', 'torus', 'hyperbolic')
            labels: Optional node labels
            parameters: Explicit two-column geometric parameters, if available
            **kwargs: Additional configuration
            
        Returns:
            Self for chaining
        """
        layer = RiemannianLayer(layer_id)
        layer.set_config(manifold_type=manifold_type, **kwargs)
        
        # Create layer with nodes
        layer_model = layer.create_layer_with_nodes(num_nodes, labels, parameters)
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
            self.topology.add_intra_layer_edge(edge)
            self._edge_counter += 1

        return self

    def set_layer_frame(
        self, layer_id: str, rotation: Matrix3 = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
        translation: Vector3 = (0., 0., 0.)
    ) -> "TopologyBuilder":
        """Set physical layer placement in a shared ambient 3D display frame.

        Does not infer intrinsic manifold correspondence. Call before
        connect_layers; existing edges are a snapshot and are not recomputed.
        """
        layer = self.topology.get_layer(layer_id)
        if layer is None:
            raise ValueError(f"Unknown layer: {layer_id}")
        layer.frame = RigidFrame3D(rotation=rotation, translation=translation)
        return self

    def reexpress_layer(
        self, layer_id: str, rotation: Matrix3, translation: Vector3
    ) -> "TopologyBuilder":
        """Passive coordinate change, preserving world positions and edges."""
        self.topology.reexpress_layer(layer_id, rotation, translation)
        return self

    def connect_intrinsic_neighbors(
        self, layer_id: str, k: int = 1
    ) -> "TopologyBuilder":
        """Join each point's k nearest neighbours using intrinsic geodesics.

        Only supported for a single sphere or hyperbolic layer with preserved
        polar chart coordinates. Does NOT infer cross-manifold transport.
        Edges are undirected, with explicit positive affinity 1/(1+distance).
        Repeated calls are idempotent for the same node pairs.
        """
        layer = self.topology.get_layer(layer_id)
        if layer is None:
            raise ValueError(f"Unknown layer: {layer_id}")
        pairs = nearest_intrinsic_pairs(layer, k=k)
        member_ids = {node.id for node in layer.nodes}
        occupied = {
            tuple(sorted((edge.source, edge.target)))
            for edge in self.topology.intra_layer_edges
            if edge.source in member_ids and edge.target in member_ids
        }
        existing_ids = {edge.id for edge in self.topology.get_all_edges()}
        metric = (
            "sphere_geodesic"
            if layer.config.get("manifold_type") == "sphere"
            else "hyperbolic_geodesic"
        )

        for source_id, target_id, distance in pairs:
            if (source_id, target_id) in occupied:
                continue
            while f"edge_{self._edge_counter}" in existing_ids:
                self._edge_counter += 1
            edge_id = f"edge_{self._edge_counter}"
            edge = EdgeModel(
                id=edge_id,
                source=source_id,
                target=target_id,
                weight=1.0 / (1.0 + distance),
                metadata={
                    "metric": metric,
                    "intrinsic_distance": distance,
                    "units": "layer_geodesic_length",
                    "weight_transform": "1/(1+intrinsic_distance)",
                },
            )
            self.topology.add_intra_layer_edge(edge)
            occupied.add((source_id, target_id))
            existing_ids.add(edge_id)
            self._edge_counter += 1
        return self

    def connect_declared_correspondence(
        self,
        spec: RadialCapMap,
        pairs: List[tuple],
        max_target_error: float = 1e-9,
    ) -> "TopologyBuilder":
        """Connect explicitly paired sphere/hyperbolic nodes after metric audit.

        This is a chosen mapping and finite-sample consistency check, NOT
        canonical inter-manifold transport. No nearest-neighbour guessing and
        no implicit unit conversion. Edges are deliberately unweighted links.
        """
        source_layer = self.topology.get_layer(spec.source_layer_id)
        target_layer = self.topology.get_layer(spec.target_layer_id)
        if source_layer is None or target_layer is None:
            raise ValueError("Declared correspondence refers to unknown layers")

        # Evaluate *all* matches before adding *any* edge.
        report = compare_radial_cap(
            source_layer, target_layer, spec, pairs,
            max_target_error=max_target_error,
        )
        occupied = {
            (edge.source, edge.target): edge
            for edge in self.topology.inter_layer_edges
        }
        new_matches = []
        for match in report.matches:
            existing = occupied.get((match.source_id, match.target_id))
            if existing is not None:
                meta = existing.metadata
                identical = (
                    meta.get("relation") == "declared_radial_cap"
                    and meta.get("provenance") == spec.provenance
                    and meta.get("radial_scale") == spec.radial_scale
                    and meta.get("cap_theta") == spec.cap_theta
                    and meta.get("angle_offset") == spec.angle_offset
                    and meta.get("source_length_unit") == spec.source_length_unit
                    and meta.get("target_length_unit") == spec.target_length_unit
                )
                if not identical:
                    raise ValueError(
                        "An incompatible edge already occupies a declared node pair"
                    )
                continue
            new_matches.append(match)

        existing_ids = {edge.id for edge in self.topology.get_all_edges()}
        for match in new_matches:
            while f"edge_{self._edge_counter}" in existing_ids:
                self._edge_counter += 1
            edge_id = f"edge_{self._edge_counter}"
            edge = EdgeModel(
                id=edge_id,
                source=match.source_id,
                target=match.target_id,
                weight=1.0,
                metadata={
                    "relation": "declared_radial_cap",
                    "provenance": spec.provenance,
                    "radial_scale": spec.radial_scale,
                    "cap_theta": spec.cap_theta,
                    "angle_offset": spec.angle_offset,
                    "source_length_unit": spec.source_length_unit,
                    "target_length_unit": spec.target_length_unit,
                    "target_mismatch": match.mismatch,
                    "max_target_error": max_target_error,
                    "weight_semantics": "unweighted_declared_correspondence",
                },
            )
            self.topology.add_inter_layer_edge(edge)
            existing_ids.add(edge_id)
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
        if source_layer_id == target_layer_id:
            raise ValueError(
                "connect_layers requires distinct layers; use connect_intrinsic_neighbors "
                "for a supported within-layer geometric relation"
            )
        
        if connection_type not in ("nearest", "random", "all"):
            raise ValueError(
                f"Unsupported connection type '{connection_type}'. "
                "Use 'nearest', 'random', or 'all'."
            )
        if connection_type != "all" and num_connections < 0:
            raise ValueError("num_connections must be non-negative")

        if connection_type == "nearest":
            self._connect_nearest(source_layer, target_layer, num_connections)
        elif connection_type == "random":
            self._connect_random(source_layer, target_layer, num_connections)
        else:
            self._connect_all(source_layer, target_layer)
        
        return self
    
    def _connect_nearest(
        self,
        source_layer: LayerModel,
        target_layer: LayerModel,
        k: int
    ) -> None:
        """Connect by ambient Euclidean world coordinates, NOT manifold geodesics."""
        for source_node in source_layer.nodes:
            # Compute distances to all target nodes
            distances = []
            for target_node in target_layer.nodes:
                dist = self._euclidean_distance(
                    self.topology.world_position(source_node),
                    self.topology.world_position(target_node)
                )
                distances.append((dist, target_node))
            
            # Sort by distance and connect to k nearest
            distances.sort(key=lambda x: (x[0], x[1].id))
            for dist, target_node in distances[:k]:
                edge = EdgeModel(
                    id=f"edge_{self._edge_counter}",
                    source=source_node.id,
                    target=target_node.id,
                    weight=1.0 / (1.0 + dist),  # Inverse AMBIENT distance
                    metadata={"metric": "ambient_euclidean_3d", "distance": float(dist)},
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
