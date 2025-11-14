"""
Neural Layer - Neural network topology visualization.

This layer represents neural network architectures and their
connection patterns, useful for visualizing LLM structures.
"""

import numpy as np
from typing import Dict, Any
from ..core.topology import TopologyLayer


class NeuralLayer(TopologyLayer):
    """
    Topology layer for neural network architectures.
    
    Visualizes layers, neurons, and connections in a neural network
    structure, particularly useful for transformer architectures.
    """
    
    def __init__(self, name: str = "neural", **kwargs):
        """
        Initialize the neural layer.
        
        Args:
            name: Layer identifier
            **kwargs: Additional configuration
                - layer_spacing: Vertical spacing between network layers
                - neuron_spacing: Horizontal spacing between neurons
        """
        super().__init__(name, **kwargs)
        self.layer_spacing = kwargs.get("layer_spacing", 3.0)
        self.neuron_spacing = kwargs.get("neuron_spacing", 1.5)
        
    def compute_positions(self) -> Dict[str, np.ndarray]:
        """
        Compute 3D positions for neural network nodes.
        
        Arranges nodes in layers with neurons spread out spatially.
        
        Returns:
            Dictionary mapping node IDs to 3D position arrays
        """
        positions = {}
        nodes = self.get_nodes()
        
        if not nodes:
            return positions
        
        # Group nodes by layer
        layers = self._group_by_layer()
        
        # Position each layer
        for layer_idx, layer_nodes in enumerate(layers):
            num_neurons = len(layer_nodes)
            
            for neuron_idx, node_id in enumerate(layer_nodes):
                # Calculate position
                # Layers are stacked along Z axis
                z = layer_idx * self.layer_spacing
                
                # Neurons arranged in a grid pattern within the layer
                grid_size = int(np.ceil(np.sqrt(num_neurons)))
                row = neuron_idx // grid_size
                col = neuron_idx % grid_size
                
                # Center the grid
                x = (col - grid_size / 2) * self.neuron_spacing
                y = (row - grid_size / 2) * self.neuron_spacing
                
                # Add small random perturbation for visual interest
                attrs = nodes[node_id]
                activation = attrs.get("activation", 0.5)
                x += np.random.randn() * 0.1 * activation
                y += np.random.randn() * 0.1 * activation
                
                positions[node_id] = np.array([x, y, z])
        
        return positions
    
    def compute_geometry(self) -> Dict[str, Any]:
        """
        Compute neural layer geometry metadata.
        
        Returns:
            Dictionary with geometric properties
        """
        nodes = self.get_nodes()
        edges = self.get_edges()
        layers = self._group_by_layer()
        
        # Compute connectivity statistics
        in_degrees = {}
        out_degrees = {}
        for node_id in nodes:
            in_degrees[node_id] = 0
            out_degrees[node_id] = 0
        
        for source, target, _ in edges:
            if source in out_degrees:
                out_degrees[source] += 1
            if target in in_degrees:
                in_degrees[target] += 1
        
        avg_in_degree = np.mean(list(in_degrees.values())) if in_degrees else 0
        avg_out_degree = np.mean(list(out_degrees.values())) if out_degrees else 0
        
        return {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "num_layers": len(layers),
            "avg_neurons_per_layer": len(nodes) / max(len(layers), 1),
            "avg_in_degree": float(avg_in_degree),
            "avg_out_degree": float(avg_out_degree),
            "type": "neural",
        }
    
    def _group_by_layer(self) -> list:
        """
        Group nodes by their network layer.
        
        Returns:
            List of lists, where each inner list contains node IDs in that layer
        """
        nodes = self.get_nodes()
        edges = self.get_edges()
        
        # Build layer assignments
        layer_map = {}
        for node_id, attrs in nodes.items():
            # Check if layer is explicitly specified
            if "layer" in attrs:
                layer_map[node_id] = attrs["layer"]
            else:
                # Infer layer from node ID if it follows a pattern
                # e.g., "layer_0_neuron_5" -> layer 0
                if "_" in node_id:
                    parts = node_id.split("_")
                    for i, part in enumerate(parts):
                        if part == "layer" and i + 1 < len(parts):
                            try:
                                layer_map[node_id] = int(parts[i + 1])
                                break
                            except ValueError:
                                pass
        
        # Assign layer 0 to nodes without explicit layer
        for node_id in nodes:
            if node_id not in layer_map:
                layer_map[node_id] = 0
        
        # Group nodes by layer
        max_layer = max(layer_map.values()) if layer_map else 0
        layers = [[] for _ in range(max_layer + 1)]
        
        for node_id, layer_idx in layer_map.items():
            layers[layer_idx].append(node_id)
        
        return layers
    
    def set_layer_structure(self, layer_sizes: list):
        """
        Create a standard feed-forward network structure.
        
        Args:
            layer_sizes: List of integers specifying neurons per layer
        """
        self.clear()
        
        # Create nodes for each layer
        for layer_idx, size in enumerate(layer_sizes):
            for neuron_idx in range(size):
                node_id = f"layer_{layer_idx}_neuron_{neuron_idx}"
                self.add_node(
                    node_id,
                    layer=layer_idx,
                    neuron_index=neuron_idx,
                    activation=np.random.rand(),
                )
        
        # Connect adjacent layers (fully connected)
        for layer_idx in range(len(layer_sizes) - 1):
            for i in range(layer_sizes[layer_idx]):
                for j in range(layer_sizes[layer_idx + 1]):
                    source = f"layer_{layer_idx}_neuron_{i}"
                    target = f"layer_{layer_idx + 1}_neuron_{j}"
                    weight = np.random.randn() * 0.5
                    self.add_edge(source, target, weight=weight, type="forward")
