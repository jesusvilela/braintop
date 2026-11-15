"""Base layer class for topology layers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

from braintop.core.topology import LayerModel, NodeModel, LayerType


class BaseLayer(ABC):
    """Abstract base class for all topology layers.
    
    This class defines the interface that all layer implementations must follow.
    Layers are responsible for generating node positions and configurations
    based on their specific geometric properties.
    """
    
    def __init__(self, layer_id: str, name: str, layer_type: LayerType):
        """Initialize the layer.
        
        Args:
            layer_id: Unique identifier for the layer
            name: Human-readable name
            layer_type: Type of the layer
        """
        self.layer_id = layer_id
        self.name = name
        self.layer_type = layer_type
        self.config: Dict[str, Any] = {}
    
    @abstractmethod
    def generate_nodes(
        self,
        num_nodes: int,
        **kwargs: Any
    ) -> List[NodeModel]:
        """Generate nodes for this layer.
        
        Args:
            num_nodes: Number of nodes to generate
            **kwargs: Additional layer-specific parameters
            
        Returns:
            List of generated nodes
        """
        pass
    
    @abstractmethod
    def compute_positions(
        self,
        data: Any,
        **kwargs: Any
    ) -> List[Tuple[float, float, float]]:
        """Compute 3D positions for nodes based on input data.
        
        Args:
            data: Input data (format depends on layer type)
            **kwargs: Additional parameters
            
        Returns:
            List of (x, y, z) positions
        """
        pass
    
    def to_layer_model(self) -> LayerModel:
        """Convert to LayerModel for use in Topology.
        
        Returns:
            LayerModel instance
        """
        return LayerModel(
            id=self.layer_id,
            name=self.name,
            layer_type=self.layer_type,
            nodes=[],
            config=self.config
        )
    
    def set_config(self, **kwargs: Any) -> None:
        """Set configuration parameters for the layer.
        
        Args:
            **kwargs: Configuration key-value pairs
        """
        self.config.update(kwargs)
