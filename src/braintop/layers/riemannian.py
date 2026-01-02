"""Riemannian layer for curved geometry topology."""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from braintop.core.topology import LayerModel, NodeModel, LayerType
from braintop.layers.base import BaseLayer


class RiemannianLayer(BaseLayer):
    """Layer for representing topology on Riemannian manifolds.
    
    This layer visualizes data on curved surfaces (spheres, tori, hyperbolic
    spaces) using Riemannian geometry. This is useful for representing
    hierarchical relationships and continuous transformations in LLMs.
    """
    
    def __init__(self, layer_id: str, name: str = "Riemannian Layer"):
        """Initialize the Riemannian layer.
        
        Args:
            layer_id: Unique identifier for the layer
            name: Human-readable name
        """
        super().__init__(layer_id, name, LayerType.RIEMANNIAN)
        self.set_config(
            manifold_type="sphere",  # sphere, torus, hyperbolic
            radius=1.0,
            curvature=1.0
        )
    
    def generate_nodes(
        self,
        num_nodes: int,
        labels: Optional[List[str]] = None,
        parameters: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> List[NodeModel]:
        """Generate nodes on a Riemannian manifold.
        
        Args:
            num_nodes: Number of nodes to generate
            labels: Optional node labels
            parameters: Manifold parameters for each node
            **kwargs: Additional parameters
            
        Returns:
            List of generated nodes
        """
        if labels is None:
            labels = [f"Node_{i}" for i in range(num_nodes)]
        
        # Generate or use provided parameters
        if parameters is None:
            # Create parameters for uniform distribution on manifold
            parameters = self._generate_uniform_parameters(num_nodes)
        
        # Compute 3D positions on manifold
        positions = self.compute_positions(parameters, **kwargs)
        radii = np.array([np.linalg.norm(position) for position in positions])
        radius_min = float(np.min(radii))
        radius_max = float(np.max(radii))
        radius_range = radius_max - radius_min if radius_max != radius_min else 1.0
        
        # Create nodes
        nodes = []
        for i in range(num_nodes):
            node = NodeModel(
                id=f"{self.layer_id}_node_{i}",
                position=positions[i],
                label=labels[i] if i < len(labels) else f"Node_{i}",
                layer_id=self.layer_id,
                metadata={
                    "manifold_type": self.config.get("manifold_type"),
                    "curvature": self.config.get("curvature"),
                    "radial_distance": float(radii[i]),
                    "normalized_radius": float((radii[i] - radius_min) / radius_range),
                }
            )
            nodes.append(node)
        
        return nodes
    
    def compute_positions(
        self,
        data: np.ndarray,
        **kwargs: Any
    ) -> List[Tuple[float, float, float]]:
        """Compute 3D positions on Riemannian manifold.
        
        Args:
            data: Manifold parameters (n_samples, n_params)
            **kwargs: Additional parameters
            
        Returns:
            List of (x, y, z) positions
        """
        manifold_type = self.config.get("manifold_type", "sphere")
        
        if manifold_type == "sphere":
            return self._sphere_embedding(data)
        elif manifold_type == "torus":
            return self._torus_embedding(data)
        elif manifold_type == "hyperbolic":
            return self._hyperbolic_embedding(data)
        else:
            # Default to sphere
            return self._sphere_embedding(data)
    
    def _generate_uniform_parameters(self, num_nodes: int) -> np.ndarray:
        """Generate uniform parameters for the manifold.
        
        Args:
            num_nodes: Number of parameter sets to generate
            
        Returns:
            Parameter array
        """
        manifold_type = self.config.get("manifold_type", "sphere")
        
        if manifold_type == "sphere":
            # Generate uniform points on sphere using spherical coordinates
            # theta: [0, pi], phi: [0, 2*pi]
            theta = np.arccos(2 * np.random.random(num_nodes) - 1)
            phi = 2 * np.pi * np.random.random(num_nodes)
            return np.column_stack([theta, phi])
        
        elif manifold_type == "torus":
            # Generate uniform points on torus
            # u, v: [0, 2*pi]
            u = 2 * np.pi * np.random.random(num_nodes)
            v = 2 * np.pi * np.random.random(num_nodes)
            return np.column_stack([u, v])
        
        else:
            # Default to 2D parameters
            return np.random.random((num_nodes, 2))
    
    def _sphere_embedding(self, params: np.ndarray) -> List[Tuple[float, float, float]]:
        """Embed points on a sphere.
        
        Args:
            params: Spherical coordinates (theta, phi) for each point
            
        Returns:
            List of 3D Cartesian coordinates
        """
        radius = self.config.get("radius", 1.0)
        
        theta = params[:, 0]
        phi = params[:, 1]
        
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        
        positions = [(float(xi), float(yi), float(zi)) 
                    for xi, yi, zi in zip(x, y, z)]
        
        return positions
    
    def _torus_embedding(self, params: np.ndarray) -> List[Tuple[float, float, float]]:
        """Embed points on a torus.
        
        Args:
            params: Toroidal coordinates (u, v) for each point
            
        Returns:
            List of 3D Cartesian coordinates
        """
        R = self.config.get("radius", 1.0)  # Major radius
        r = R * 0.4  # Minor radius
        
        u = params[:, 0]
        v = params[:, 1]
        
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        
        positions = [(float(xi), float(yi), float(zi)) 
                    for xi, yi, zi in zip(x, y, z)]
        
        return positions
    
    def _hyperbolic_embedding(self, params: np.ndarray) -> List[Tuple[float, float, float]]:
        """Embed points in hyperbolic space (Poincaré disk model).
        
        Args:
            params: Hyperbolic coordinates for each point
            
        Returns:
            List of 3D Cartesian coordinates (with z representing depth)
        """
        # Map to Poincaré disk
        r = np.tanh(params[:, 0] * 0.5)
        theta = params[:, 1] * 2 * np.pi
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = -np.log(1 - r**2) * 0.5  # Depth encoding
        
        positions = [(float(xi), float(yi), float(zi)) 
                    for xi, yi, zi in zip(x, y, z)]
        
        return positions
    
    def create_layer_with_nodes(
        self,
        num_nodes: int,
        labels: Optional[List[str]] = None,
        parameters: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> LayerModel:
        """Create a complete layer model with generated nodes.
        
        Args:
            num_nodes: Number of nodes to generate
            labels: Optional node labels
            parameters: Optional manifold parameters
            **kwargs: Additional parameters
            
        Returns:
            Complete LayerModel
        """
        layer_model = self.to_layer_model()
        layer_model.nodes = self.generate_nodes(num_nodes, labels, parameters, **kwargs)
        return layer_model
