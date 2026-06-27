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
            curvature=None,
            max_radius=4.0,
        )

    def generate_nodes(
        self,
        num_nodes: int,
        labels: Optional[List[str]] = None,
        parameters: Optional[np.ndarray] = None,
        **kwargs: Any,
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
        if num_nodes < 0:
            raise ValueError("num_nodes must be non-negative")
        if num_nodes == 0:
            return []
        if labels is None:
            labels = [f"Node_{i}" for i in range(num_nodes)]

        # Generate or use provided parameters
        if parameters is None:
            # Create parameters for uniform distribution on manifold
            parameters = self._generate_uniform_parameters(num_nodes)
        else:
            parameters = self._validate_parameter_array(parameters, num_nodes)

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
                },
            )
            nodes.append(node)

        return nodes

    def compute_positions(
        self, data: np.ndarray, **kwargs: Any
    ) -> List[Tuple[float, float, float]]:
        """Compute 3D positions on Riemannian manifold.

        Args:
            data: Manifold parameters (n_samples, n_params)
            **kwargs: Additional parameters

        Returns:
            List of (x, y, z) positions
        """
        data = self._validate_parameter_array(data)
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

    @staticmethod
    def _validate_parameter_array(
        data: np.ndarray,
        expected_rows: Optional[int] = None,
    ) -> np.ndarray:
        """Validate and coerce manifold parameter arrays."""
        array = np.asarray(data, dtype=float)
        if array.ndim != 2:
            raise ValueError("Manifold parameters must be a 2D array")
        if array.shape[0] == 0:
            raise ValueError("Manifold parameters must contain at least one row")
        if expected_rows is not None and array.shape[0] != expected_rows:
            raise ValueError("Parameter row count must match num_nodes")
        if array.shape[1] < 2:
            raise ValueError("Manifold parameters must include at least two columns")
        if not np.all(np.isfinite(array)):
            raise ValueError("Manifold parameters must contain only finite values")
        return array

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
            # Generate points uniformly with respect to torus surface area.
            # The area element is proportional to R + r*cos(v), so v cannot be
            # sampled uniformly without biasing points toward the inner rim.
            u = 2 * np.pi * np.random.random(num_nodes)
            v = self._sample_torus_minor_angles(num_nodes)
            return np.column_stack([u, v])

        elif manifold_type == "hyperbolic":
            # Sample uniformly by hyperbolic area on a finite geodesic disk.
            # For curvature K < 0, area density is proportional to
            # sinh(sqrt(|K|) * rho), so the inverse CDF is analytic.
            curvature = self._hyperbolic_curvature()
            curvature_scale = np.sqrt(abs(curvature))
            rho_max = float(self.config.get("max_radius", 4.0))
            if rho_max <= 0 or not np.isfinite(rho_max):
                raise ValueError("Hyperbolic max_radius must be positive and finite")

            u = np.random.random(num_nodes)
            cdf_scale = np.cosh(curvature_scale * rho_max) - 1.0
            rho = np.arccosh(1.0 + u * cdf_scale) / curvature_scale
            theta = 2 * np.pi * np.random.random(num_nodes)
            return np.column_stack([rho, theta])

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
        radius = float(self.config.get("radius", 1.0))
        if radius <= 0:
            raise ValueError("Sphere radius must be positive")

        theta = params[:, 0]
        phi = params[:, 1]

        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)

        positions = [(float(xi), float(yi), float(zi)) for xi, yi, zi in zip(x, y, z)]

        return positions

    def _torus_embedding(self, params: np.ndarray) -> List[Tuple[float, float, float]]:
        """Embed points on a torus.

        Args:
            params: Toroidal coordinates (u, v) for each point

        Returns:
            List of 3D Cartesian coordinates
        """
        R = float(self.config.get("radius", 1.0))  # Major radius
        r = float(self.config.get("minor_radius", R * 0.4))
        if R <= 0 or r <= 0:
            raise ValueError("Torus radii must be positive")
        if r >= R:
            raise ValueError("Torus minor_radius must be smaller than radius")

        u = params[:, 0]
        v = params[:, 1]

        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)

        positions = [(float(xi), float(yi), float(zi)) for xi, yi, zi in zip(x, y, z)]

        return positions

    def _sample_torus_minor_angles(self, num_nodes: int) -> np.ndarray:
        """Sample torus minor angles uniformly by surface area.

        Returns angles v with density proportional to R + r*cos(v), using
        rejection sampling from a uniform proposal on [0, 2*pi].
        """
        R = float(self.config.get("radius", 1.0))
        r = float(self.config.get("minor_radius", R * 0.4))
        if R <= 0 or r <= 0:
            raise ValueError("Torus radii must be positive")
        if r >= R:
            raise ValueError("Torus minor_radius must be smaller than radius")

        samples = []
        max_density = R + r
        while len(samples) < num_nodes:
            batch_size = max(num_nodes - len(samples), 16) * 2
            candidates = 2 * np.pi * np.random.random(batch_size)
            accept_prob = (R + r * np.cos(candidates)) / max_density
            accepted = candidates[np.random.random(batch_size) <= accept_prob]
            samples.extend(accepted.tolist())

        return np.array(samples[:num_nodes])

    def _hyperbolic_curvature(self) -> float:
        """Return validated negative curvature for hyperbolic geometry."""
        configured_curvature = self.config.get("curvature")
        curvature = -1.0 if configured_curvature is None else float(configured_curvature)
        if curvature >= 0 or not np.isfinite(curvature):
            raise ValueError("Hyperbolic curvature must be negative and finite")
        return curvature

    def _hyperbolic_embedding(self, params: np.ndarray) -> List[Tuple[float, float, float]]:
        """Embed points in hyperbolic space (Poincaré disk model).

        Args:
            params: Hyperbolic coordinates for each point

        Returns:
            List of 3D Cartesian coordinates (with z representing depth)
        """
        curvature = self._hyperbolic_curvature()
        curvature_scale = 1.0 / np.sqrt(abs(curvature))
        rho = np.maximum(params[:, 0], 0.0)
        theta = params[:, 1]

        # Poincare disk radial coordinate for geodesic radius rho on curvature K:
        # disk_r = tanh(rho / (2 / sqrt(|K|))). The visible z coordinate is not an
        # isometric embedding; it encodes hyperbolic depth/geodesic radius.
        disk_r = np.tanh(rho / (2.0 * curvature_scale))

        x = disk_r * np.cos(theta)
        y = disk_r * np.sin(theta)
        z = rho * curvature_scale

        positions = [(float(xi), float(yi), float(zi)) for xi, yi, zi in zip(x, y, z)]

        return positions

    def create_layer_with_nodes(
        self,
        num_nodes: int,
        labels: Optional[List[str]] = None,
        parameters: Optional[np.ndarray] = None,
        **kwargs: Any,
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
