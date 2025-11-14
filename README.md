# 🧠 BrainTop

**A beautiful rich 3D topology visualizer for Large Language Models**

BrainTop provides a modular, extensible framework for visualizing complex topology structures of LLMs, agents, and orchestrators. With support for multiple mathematical perspectives including conceptual hierarchies, Riemannian manifolds, and neural architectures, BrainTop makes it easy to understand and explore the structure of AI systems.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- 🎨 **Beautiful 3D Visualizations** - Interactive, high-quality visualizations using Plotly
- 🧩 **Modular Architecture** - Loosely coupled design with pluggable topology layers
- 📐 **Multiple Perspectives** - View topology through different mathematical lenses:
  - **Conceptual Layer** - Hierarchical concept relationships
  - **Riemannian Layer** - Geometric manifold embeddings (sphere, hyperbolic, torus)
  - **Neural Layer** - Network architecture visualization
- 🔌 **LLM Integration** - Easy-to-use interfaces for connecting to LLMs, agents, and orchestrators
- 🐍 **Python Foundation** - Pure Python implementation with minimal dependencies
- 📊 **Rich Metadata** - Comprehensive geometric and topological statistics

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/jesusvilela/braintop.git
cd braintop

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from braintop import TopologyGraph, ConceptualLayer, Renderer3D

# Create a topology graph
graph = TopologyGraph()

# Create a conceptual layer
layer = ConceptualLayer(name="concepts")

# Add nodes and edges
layer.add_node("AI", weight=1.0, type="root")
layer.add_node("ML", weight=0.9, type="concept")
layer.add_edge("AI", "ML", type="parent_of")

# Add to graph
graph.add_layer(layer)

# Visualize
renderer = Renderer3D(theme="dark")
fig = renderer.render_layer(layer)
renderer.save(fig, "topology.html")
```

## 📚 Documentation

### Topology Layers

#### Conceptual Layer
Represents abstract concepts and their hierarchical relationships. Perfect for visualizing semantic spaces and knowledge graphs.

```python
from braintop import ConceptualLayer

layer = ConceptualLayer(
    name="concepts",
    hierarchy_strength=0.7,  # How strongly to enforce hierarchy
    spread_factor=2.0         # Controls node spacing
)

layer.add_node("concept_id", weight=1.0, type="category")
layer.add_edge("parent", "child", type="parent_of")
```

#### Riemannian Layer
Maps data onto curved manifolds, useful for visualizing embedding spaces and latent representations.

```python
from braintop import RiemannianLayer

layer = RiemannianLayer(
    name="embeddings",
    manifold_type="sphere",  # Options: sphere, hyperbolic, torus
    curvature=0.5            # Manifold curvature (-1 to 1)
)

layer.add_node("word", embedding=[0.5, 0.3], weight=1.0)
```

#### Neural Layer
Visualizes neural network architectures, particularly useful for transformer models.

```python
from braintop import NeuralLayer

layer = NeuralLayer(
    name="network",
    layer_spacing=3.0,      # Vertical spacing between layers
    neuron_spacing=1.5      # Horizontal spacing between neurons
)

# Automatically create a network structure
layer.set_layer_structure([512, 256, 128, 256, 512])
```

### Rendering

```python
from braintop import Renderer3D

renderer = Renderer3D(
    title="My Visualization",
    width=1200,
    height=800,
    theme="dark"  # Options: dark, light, scientific
)

# Render a single layer
fig = renderer.render_layer(layer, show_edges=True, node_size=8)

# Render multiple layers
fig = renderer.render_graph(graph, layers=["layer1", "layer2"])

# Save to file
renderer.save(fig, "output.html")  # HTML format
renderer.save(fig, "output.png", format="png")  # Image format
```

### LLM Integration

BrainTop provides connectors for integrating with various LLM systems:

```python
from braintop.integration import (
    AgentConnector,
    OrchestratorConnector,
    TransformerConnector,
    DataTransformer
)

# Create custom connectors by subclassing
class MyAgentConnector(AgentConnector):
    def extract_topology(self):
        graph = TopologyGraph()
        # Your custom extraction logic
        return graph

# Transform existing data structures
graph = DataTransformer.from_dict({
    "nodes": [{"id": "node1", "weight": 1.0}],
    "edges": [{"source": "node1", "target": "node2"}]
})

# Convert from NetworkX
import networkx as nx
nx_graph = nx.karate_club_graph()
graph = DataTransformer.from_networkx(nx_graph, layer_type="conceptual")
```

## 🎯 Examples

The `examples/` directory contains several complete examples:

- **basic_conceptual.py** - Simple conceptual hierarchy visualization
- **riemannian_manifolds.py** - Word embeddings on curved manifolds
- **neural_network.py** - Transformer architecture visualization
- **advanced_multi_layer.py** - Multi-layer LLM agent system

Run any example:

```bash
cd examples
python basic_conceptual.py
```

## 🏗️ Architecture

BrainTop follows a modular, loosely coupled architecture:

```
braintop/
├── core/           # Core abstractions (TopologyLayer, TopologyGraph)
├── layers/         # Specific layer implementations
├── visualization/  # 3D rendering engine
├── integration/    # LLM/agent connectors
└── utils/          # Utility functions
```

### Extending BrainTop

Create custom topology layers by subclassing `TopologyLayer`:

```python
from braintop.core import TopologyLayer
import numpy as np

class CustomLayer(TopologyLayer):
    def compute_positions(self):
        positions = {}
        for node_id in self.get_nodes():
            # Your custom positioning logic
            positions[node_id] = np.array([x, y, z])
        return positions
    
    def compute_geometry(self):
        return {
            "num_nodes": len(self.get_nodes()),
            "type": "custom",
            # Your custom metrics
        }
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Plotly](https://plotly.com/) for beautiful 3D visualizations
- Geometric algorithms inspired by NetworkX and SciPy
- Designed for the modern LLM ecosystem

## 📧 Contact

For questions and support, please open an issue on GitHub.

---

**Made with ❤️ for the AI/ML community**
