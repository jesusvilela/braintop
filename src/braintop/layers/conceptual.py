"""Conceptual layer for semantic/concept-based topology."""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from braintop.core.topology import LayerModel, NodeModel, LayerType
from braintop.layers.base import BaseLayer


class ConceptualLayer(BaseLayer):
    """Layer for representing conceptual relationships in semantic space.
    
    This layer visualizes concepts and their relationships in a 3D space,
    using techniques like dimensionality reduction (PCA, t-SNE, UMAP) to
    project high-dimensional semantic vectors into 3D space.
    """
    
    def __init__(self, layer_id: str, name: str = "Conceptual Layer"):
        """Initialize the conceptual layer.
        
        Args:
            layer_id: Unique identifier for the layer
            name: Human-readable name
        """
        super().__init__(layer_id, name, LayerType.CONCEPTUAL)
        self.set_config(
            projection_method="pca",
            cluster_threshold=0.5,
            spacing_factor=1.0
        )
    
    def generate_nodes(
        self,
        num_nodes: int,
        concepts: Optional[List[str]] = None,
        embeddings: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> List[NodeModel]:
        """Generate nodes representing concepts.
        
        Args:
            num_nodes: Number of nodes to generate
            concepts: List of concept labels
            embeddings: Optional high-dimensional embeddings to project
            **kwargs: Additional parameters
            
        Returns:
            List of generated nodes
        """
        if concepts is None:
            concepts = [f"Concept_{i}" for i in range(num_nodes)]
        
        # Generate or use provided embeddings
        if embeddings is None:
            # Create random high-dimensional embeddings for demonstration
            embeddings = np.random.randn(num_nodes, 128)
        
        # Compute 3D positions
        positions = self.compute_positions(embeddings, **kwargs)
        embedding_norms = np.linalg.norm(embeddings, axis=1)
        norm_min = float(np.min(embedding_norms))
        norm_max = float(np.max(embedding_norms))
        norm_range = norm_max - norm_min if norm_max != norm_min else 1.0
        
        # Create nodes
        nodes = []
        for i in range(num_nodes):
            node = NodeModel(
                id=f"{self.layer_id}_node_{i}",
                position=positions[i],
                label=concepts[i] if i < len(concepts) else f"Concept_{i}",
                layer_id=self.layer_id,
                metadata={
                    "concept": concepts[i] if i < len(concepts) else f"Concept_{i}",
                    "embedding_dim": embeddings.shape[1] if embeddings is not None else 0,
                    "embedding_norm": float(embedding_norms[i]),
                    "salience": float((embedding_norms[i] - norm_min) / norm_range),
                }
            )
            nodes.append(node)
        
        return nodes
    
    def compute_positions(
        self,
        data: np.ndarray,
        method: Optional[str] = None,
        **kwargs: Any
    ) -> List[Tuple[float, float, float]]:
        """Compute 3D positions from high-dimensional embeddings.
        
        Args:
            data: High-dimensional embeddings (n_samples, n_features)
            method: Projection method ('pca', 'random', or None to use config)
            **kwargs: Additional parameters
            
        Returns:
            List of (x, y, z) positions
        """
        method = method or self.config.get("projection_method", "pca")
        spacing = self.config.get("spacing_factor", 1.0)
        
        if method == "pca":
            # Simple PCA projection
            positions = self._pca_projection(data)
        else:
            # Random projection as fallback
            positions = self._random_projection(data)
        
        # Apply spacing factor
        positions = [(x * spacing, y * spacing, z * spacing) for x, y, z in positions]
        
        return positions
    
    def _pca_projection(self, data: np.ndarray) -> List[Tuple[float, float, float]]:
        """Project data to 3D using PCA.
        
        Args:
            data: Input data (n_samples, n_features)
            
        Returns:
            List of 3D positions
        """
        if data.ndim != 2:
            raise ValueError("PCA projection expects a 2D array")

        # Center the data. SVD on the centered design matrix is numerically
        # stable for PCA and handles n_samples == 1 without np.cov producing
        # a scalar/NaN covariance matrix.
        data_centered = data - np.mean(data, axis=0, keepdims=True)

        # Vt rows are principal axes ordered by decreasing singular value.
        _, _, vt = np.linalg.svd(data_centered, full_matrices=False)
        n_components = min(3, vt.shape[0])
        projected = data_centered @ vt[:n_components].T

        # Always return exactly three coordinates. If the rank/feature count is
        # lower than three, missing principal components have zero variance.
        if n_components < 3:
            projected = np.pad(
                projected,
                ((0, 0), (0, 3 - n_components)),
                mode="constant",
            )

        positions = [(float(x), float(y), float(z)) for x, y, z in projected]

        return positions
    
    def _random_projection(self, data: np.ndarray) -> List[Tuple[float, float, float]]:
        """Project data to 3D using random projection.
        
        Args:
            data: Input data (n_samples, n_features)
            
        Returns:
            List of 3D positions
        """
        n_features = data.shape[1]
        
        # Create random projection matrix
        projection_matrix = np.random.randn(n_features, 3) / np.sqrt(n_features)
        
        # Project data
        projected = data @ projection_matrix
        
        # Convert to list of tuples
        positions = [(float(x), float(y), float(z)) 
                    for x, y, z in projected]
        
        return positions
    
    def create_layer_with_nodes(
        self,
        num_nodes: int,
        concepts: Optional[List[str]] = None,
        embeddings: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> LayerModel:
        """Create a complete layer model with generated nodes.
        
        Args:
            num_nodes: Number of nodes to generate
            concepts: List of concept labels
            embeddings: Optional high-dimensional embeddings
            **kwargs: Additional parameters
            
        Returns:
            Complete LayerModel
        """
        layer_model = self.to_layer_model()
        layer_model.nodes = self.generate_nodes(num_nodes, concepts, embeddings, **kwargs)
        return layer_model
