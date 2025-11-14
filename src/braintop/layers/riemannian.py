"""
Riemannian Layer - Geometric manifold topology visualization.

This layer represents topology using Riemannian geometry concepts,
useful for visualizing latent spaces and embedding manifolds.
"""

import numpy as np
from typing import Dict, Any
from scipy.spatial.distance import pdist, squareform
from ..core.topology import TopologyLayer


class RiemannianLayer(TopologyLayer):
    """
    Topology layer based on Riemannian geometry.
    
    Positions nodes on a curved manifold surface, representing
    the intrinsic geometry of the data space.
    """
    
    def __init__(self, name: str = "riemannian", **kwargs):
        """
        Initialize the Riemannian layer.
        
        Args:
            name: Layer identifier
            **kwargs: Additional configuration
                - curvature: Manifold curvature (-1 to 1)
                - manifold_type: Type of manifold ('sphere', 'hyperbolic', 'torus')
        """
        super().__init__(name, **kwargs)
        self.curvature = kwargs.get("curvature", 0.5)
        self.manifold_type = kwargs.get("manifold_type", "sphere")
        
    def compute_positions(self) -> Dict[str, np.ndarray]:
        """
        Compute 3D positions on a Riemannian manifold.
        
        Maps nodes to points on a curved surface based on their
        intrinsic relationships.
        
        Returns:
            Dictionary mapping node IDs to 3D position arrays
        """
        positions = {}
        nodes = self.get_nodes()
        
        if not nodes:
            return positions
        
        # Get node embeddings or create them
        embeddings = self._get_or_create_embeddings()
        
        # Map to manifold based on type
        if self.manifold_type == "sphere":
            positions = self._map_to_sphere(embeddings)
        elif self.manifold_type == "hyperbolic":
            positions = self._map_to_hyperbolic(embeddings)
        elif self.manifold_type == "torus":
            positions = self._map_to_torus(embeddings)
        else:
            # Default to sphere
            positions = self._map_to_sphere(embeddings)
        
        return positions
    
    def compute_geometry(self) -> Dict[str, Any]:
        """
        Compute Riemannian geometry metadata.
        
        Returns:
            Dictionary with geometric properties
        """
        nodes = self.get_nodes()
        positions = self.compute_positions()
        
        # Compute geodesic properties
        if len(positions) > 1:
            pos_array = np.array(list(positions.values()))
            distances = pdist(pos_array)
            avg_distance = np.mean(distances)
            std_distance = np.std(distances)
        else:
            avg_distance = 0.0
            std_distance = 0.0
        
        return {
            "num_nodes": len(nodes),
            "manifold_type": self.manifold_type,
            "curvature": self.curvature,
            "avg_geodesic_distance": float(avg_distance),
            "std_geodesic_distance": float(std_distance),
            "type": "riemannian",
        }
    
    def _get_or_create_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Get or create 2D embeddings for nodes.
        
        Returns:
            Dictionary mapping node IDs to 2D embedding arrays
        """
        embeddings = {}
        nodes = self.get_nodes()
        
        for i, (node_id, attrs) in enumerate(nodes.items()):
            # Check if node has explicit embedding
            if "embedding" in attrs:
                embeddings[node_id] = np.array(attrs["embedding"][:2])
            else:
                # Create a default embedding using node position
                angle = 2 * np.pi * i / len(nodes)
                embeddings[node_id] = np.array([np.cos(angle), np.sin(angle)])
        
        return embeddings
    
    def _map_to_sphere(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Map 2D embeddings to a spherical surface.
        
        Args:
            embeddings: 2D embeddings for each node
            
        Returns:
            3D positions on sphere
        """
        positions = {}
        radius = 3.0 + self.curvature * 2.0
        
        for node_id, emb in embeddings.items():
            # Normalize and scale to sphere
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            
            # Map to sphere using spherical coordinates
            theta = np.arctan2(emb[1], emb[0])  # Azimuthal angle
            phi = np.pi / 2 * (1 - norm)  # Polar angle (0 to pi/2)
            
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            
            positions[node_id] = np.array([x, y, z])
        
        return positions
    
    def _map_to_hyperbolic(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Map 2D embeddings to hyperbolic space (Poincaré disk model).
        
        Args:
            embeddings: 2D embeddings for each node
            
        Returns:
            3D positions in hyperbolic space
        """
        positions = {}
        scale = 3.0
        
        for node_id, emb in embeddings.items():
            # Ensure embeddings are within unit disk
            norm = np.linalg.norm(emb)
            if norm >= 1:
                emb = emb / (norm + 0.1)
            
            # Map to upper half of hyperboloid
            x = scale * emb[0]
            y = scale * emb[1]
            z = scale * np.sqrt(1 + np.linalg.norm(emb)**2) * self.curvature
            
            positions[node_id] = np.array([x, y, z])
        
        return positions
    
    def _map_to_torus(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Map 2D embeddings to a torus surface.
        
        Args:
            embeddings: 2D embeddings for each node
            
        Returns:
            3D positions on torus
        """
        positions = {}
        R = 3.0  # Major radius
        r = 1.0  # Minor radius
        
        for node_id, emb in embeddings.items():
            # Map to torus using parametric equations
            u = np.pi * emb[0]  # Angle around major circle
            v = np.pi * emb[1]  # Angle around minor circle
            
            x = (R + r * np.cos(v)) * np.cos(u)
            y = (R + r * np.cos(v)) * np.sin(u)
            z = r * np.sin(v)
            
            positions[node_id] = np.array([x, y, z])
        
        return positions
