# BrainTop API Documentation

## Core Module (`braintop.core`)

### TopologyLayer

Abstract base class for all topology layers.

```python
class TopologyLayer(ABC):
    def __init__(self, name: str, **kwargs)
    
    @abstractmethod
    def compute_positions(self) -> Dict[str, np.ndarray]
    
    @abstractmethod
    def compute_geometry(self) -> Dict[str, Any]
    
    def add_node(self, node_id: str, **attributes)
    def add_edge(self, source: str, target: str, **attributes)
    def get_nodes(self) -> Dict[str, Dict[str, Any]]
    def get_edges(self) -> List[Tuple[str, str, Dict[str, Any]]]
    def clear(self)
```

**Methods:**
- `compute_positions()`: Compute 3D positions for all nodes
- `compute_geometry()`: Compute layer-specific geometric properties
- `add_node(node_id, **attributes)`: Add a node with attributes
- `add_edge(source, target, **attributes)`: Add an edge between nodes
- `get_nodes()`: Get all nodes
- `get_edges()`: Get all edges
- `clear()`: Clear all data

### TopologyGraph

Container for multiple topology layers.

```python
class TopologyGraph:
    def __init__(self)
    
    def add_layer(self, layer: TopologyLayer)
    def get_layer(self, name: str) -> Optional[TopologyLayer]
    def get_all_layers(self) -> Dict[str, TopologyLayer]
    def remove_layer(self, name: str)
    def set_metadata(self, key: str, value: Any)
    def get_metadata(self, key: str) -> Optional[Any]
    def to_networkx(self, layer_name: Optional[str] = None) -> nx.Graph
```

**Methods:**
- `add_layer(layer)`: Add a topology layer
- `get_layer(name)`: Retrieve a layer by name
- `get_all_layers()`: Get all layers
- `remove_layer(name)`: Remove a layer
- `set_metadata(key, value)`: Set graph metadata
- `get_metadata(key)`: Get graph metadata
- `to_networkx(layer_name)`: Convert to NetworkX graph

## Layers Module (`braintop.layers`)

### ConceptualLayer

Hierarchical concept topology layer.

```python
class ConceptualLayer(TopologyLayer):
    def __init__(
        self,
        name: str = "conceptual",
        hierarchy_strength: float = 0.7,
        spread_factor: float = 2.0,
        **kwargs
    )
```

**Parameters:**
- `name`: Layer identifier
- `hierarchy_strength`: How strongly to enforce hierarchy (0-1)
- `spread_factor`: Controls node spacing

**Features:**
- Hierarchical positioning based on parent-child relationships
- Circular arrangement within hierarchy levels
- Semantic clustering with noise

**Edge Types:**
- `parent_of`: Parent-child relationship
- `contains`: Container relationship
- `child_of`: Inverse parent relationship
- `related`: General semantic relationship

### RiemannianLayer

Geometric manifold topology layer.

```python
class RiemannianLayer(TopologyLayer):
    def __init__(
        self,
        name: str = "riemannian",
        curvature: float = 0.5,
        manifold_type: str = "sphere",
        **kwargs
    )
```

**Parameters:**
- `name`: Layer identifier
- `curvature`: Manifold curvature (-1 to 1)
- `manifold_type`: Type of manifold

**Manifold Types:**
- `sphere`: Spherical surface
- `hyperbolic`: Hyperbolic space (Poincaré disk model)
- `torus`: Torus surface

**Node Attributes:**
- `embedding`: 2D embedding array [x, y]
- `weight`: Node importance

**Geometry Metrics:**
- `avg_geodesic_distance`: Average distance between nodes
- `std_geodesic_distance`: Standard deviation of distances

### NeuralLayer

Neural network architecture layer.

```python
class NeuralLayer(TopologyLayer):
    def __init__(
        self,
        name: str = "neural",
        layer_spacing: float = 3.0,
        neuron_spacing: float = 1.5,
        **kwargs
    )
    
    def set_layer_structure(self, layer_sizes: List[int])
```

**Parameters:**
- `name`: Layer identifier
- `layer_spacing`: Vertical spacing between network layers
- `neuron_spacing`: Horizontal spacing between neurons

**Methods:**
- `set_layer_structure(layer_sizes)`: Create feed-forward network

**Node Attributes:**
- `layer`: Network layer index
- `neuron_index`: Index within layer
- `activation`: Activation value

**Edge Attributes:**
- `weight`: Connection weight
- `type`: Connection type (e.g., "forward", "attention")

## Visualization Module (`braintop.visualization`)

### Renderer3D

3D visualization renderer.

```python
class Renderer3D:
    def __init__(
        self,
        title: str = "BrainTop 3D Topology Visualizer",
        width: int = 1200,
        height: int = 800,
        theme: str = "dark",
        **kwargs
    )
    
    def render_layer(
        self,
        layer: TopologyLayer,
        show_edges: bool = True,
        node_size: float = 8,
        edge_width: float = 2
    ) -> go.Figure
    
    def render_graph(
        self,
        graph: TopologyGraph,
        layers: Optional[List[str]] = None,
        **kwargs
    ) -> go.Figure
    
    def save(self, fig: go.Figure, filename: str, format: str = "html")
```

**Parameters:**
- `title`: Visualization title
- `width`: Figure width in pixels
- `height`: Figure height in pixels
- `theme`: Color theme

**Themes:**
- `dark`: Dark background with bright nodes
- `light`: Light background with dark nodes
- `scientific`: Scientific color scheme

**Methods:**
- `render_layer(layer, ...)`: Render a single layer
- `render_graph(graph, layers, ...)`: Render multiple layers
- `save(fig, filename, format)`: Save to file

**Output Formats:**
- `html`: Interactive HTML file
- `png`: PNG image
- `jpg`: JPEG image
- `svg`: SVG vector image
- `pdf`: PDF document

## Integration Module (`braintop.integration`)

### LLMConnector

Abstract base class for LLM connectors.

```python
class LLMConnector(ABC):
    def __init__(self, name: str)
    
    @abstractmethod
    def extract_topology(self) -> TopologyGraph
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]
```

### AgentConnector

Connector for LLM agents.

```python
class AgentConnector(LLMConnector):
    def __init__(self, name: str = "agent", **kwargs)
    
    def add_agent_node(
        self,
        graph: TopologyGraph,
        layer_name: str,
        agent_id: str,
        **attributes
    )
```

### OrchestratorConnector

Connector for LLM orchestrators.

```python
class OrchestratorConnector(LLMConnector):
    def __init__(self, name: str = "orchestrator", **kwargs)
    
    def add_task_node(
        self,
        graph: TopologyGraph,
        layer_name: str,
        task_id: str,
        **attributes
    )
```

### TransformerConnector

Connector for transformer models.

```python
class TransformerConnector(LLMConnector):
    def __init__(self, name: str = "transformer", **kwargs)
```

### DataTransformer

Utility for transforming data formats.

```python
class DataTransformer:
    @staticmethod
    def from_dict(
        data: Dict[str, Any],
        layer_type: str = "conceptual"
    ) -> TopologyGraph
    
    @staticmethod
    def from_networkx(
        nx_graph,
        layer_type: str = "conceptual"
    ) -> TopologyGraph
```

**Methods:**
- `from_dict(data, layer_type)`: Create from dictionary
- `from_networkx(nx_graph, layer_type)`: Create from NetworkX

**Dictionary Format:**
```python
{
    "nodes": [
        {"id": "node1", "weight": 1.0, ...},
        {"id": "node2", "weight": 0.5, ...}
    ],
    "edges": [
        {"source": "node1", "target": "node2", "weight": 0.8, ...}
    ]
}
```

## Extension Guide

### Creating Custom Layers

Subclass `TopologyLayer` and implement required methods:

```python
from braintop.core import TopologyLayer
import numpy as np

class CustomLayer(TopologyLayer):
    def compute_positions(self):
        positions = {}
        for node_id in self.get_nodes():
            # Your positioning logic
            positions[node_id] = np.array([x, y, z])
        return positions
    
    def compute_geometry(self):
        return {
            "num_nodes": len(self.get_nodes()),
            "type": "custom",
            # Add custom metrics
        }
```

### Creating Custom Connectors

Subclass connector base classes:

```python
from braintop.integration import LLMConnector
from braintop import TopologyGraph, ConceptualLayer

class MyConnector(LLMConnector):
    def extract_topology(self):
        graph = TopologyGraph()
        layer = ConceptualLayer()
        
        # Extract topology from your system
        # Add nodes and edges
        
        graph.add_layer(layer)
        return graph
    
    def get_metadata(self):
        return {
            "connector_type": "custom",
            "name": self.name,
            # Add metadata
        }
```
