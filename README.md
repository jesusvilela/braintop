# Braintop 🧠

A beautiful rich 3D topology visualizer for Large Language Models (LLMs) with modular architecture supporting multiple geometric layer types.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- **🎨 Beautiful 3D Visualizations**: Interactive 3D plots using Plotly with rich customization
- **🏗️ Modular Architecture**: Loosely coupled, extensible design with plugin support
- **📊 Multiple Layer Types**:
  - **Conceptual Layers**: Semantic/concept-based topology with PCA projection
  - **Riemannian Layers**: Curved manifolds (sphere, torus, hyperbolic space)
  - **Custom Layers**: Extensible base class for custom geometries
- **🤖 LLM Integration**: Plugin architecture for connecting to various LLM providers
- **🔗 Agent & Orchestrator Ready**: Designed to integrate with AI agents and orchestration systems
- **📈 Network Analysis**: Export to NetworkX for advanced graph analysis
- **🐍 Pure Python**: Built on numpy, plotly, and scipy

## 🚀 Quick Start

### Installation

```bash
# Install from source
git clone https://github.com/jesusvilela/braintop.git
cd braintop
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
from braintop.utils.builders import TopologyBuilder
from braintop.core.visualizer import TopologyVisualizer

# Create a topology
builder = TopologyBuilder("my_topology", "My First Topology")

# Add a conceptual layer
concepts = ["Attention", "Embedding", "Transformer", "Decoder", "Encoder"]
builder.add_conceptual_layer("concepts", num_nodes=5, concepts=concepts)

# Add a Riemannian sphere layer
builder.add_riemannian_layer("sphere", num_nodes=8, manifold_type="sphere")

# Connect layers
builder.connect_layers("concepts", "sphere", "nearest", num_connections=3)

# Build and visualize
topology = builder.build()
visualizer = TopologyVisualizer(topology)
visualizer.show()
```

### CLI Usage

```bash
# Show information
braintop info

# Create and visualize a demo topology
braintop visualize

# Save to file
braintop visualize -o my_topology.html
```

## 📚 Documentation

### Core Concepts

#### Topology
The main data structure representing a complete 3D topology with multiple layers and inter-layer connections.

```python
from braintop.core.topology import Topology, LayerModel, NodeModel, EdgeModel

topology = Topology(
    id="my_topology",
    name="My Topology",
    layers=[],
    inter_layer_edges=[],
    metadata={}
)
```

#### Layers

**Conceptual Layer**: Represents semantic relationships in high-dimensional space, projected to 3D.

```python
from braintop.layers.conceptual import ConceptualLayer

layer = ConceptualLayer("concepts_layer")
layer.set_config(projection_method="pca", spacing_factor=1.5)
layer_model = layer.create_layer_with_nodes(
    num_nodes=10,
    concepts=["Concept A", "Concept B", ...],
    embeddings=my_embeddings  # Optional numpy array
)
```

**Riemannian Layer**: Represents data on curved manifolds.

```python
from braintop.layers.riemannian import RiemannianLayer

# Sphere
sphere_layer = RiemannianLayer("sphere")
sphere_layer.set_config(manifold_type="sphere", radius=2.0)

# Torus
torus_layer = RiemannianLayer("torus")
torus_layer.set_config(manifold_type="torus", radius=1.5)

# Hyperbolic (Poincaré disk)
hyperbolic_layer = RiemannianLayer("hyperbolic")
hyperbolic_layer.set_config(manifold_type="hyperbolic", curvature=-1.0)

layer_model = sphere_layer.create_layer_with_nodes(num_nodes=15)
```

#### TopologyBuilder

Fluent interface for constructing complex topologies:

```python
from braintop.utils.builders import TopologyBuilder

topology = (TopologyBuilder("topology_id", "Topology Name")
    .add_conceptual_layer("layer1", num_nodes=10, concepts=concepts)
    .add_riemannian_layer("layer2", num_nodes=15, manifold_type="sphere")
    .connect_layers("layer1", "layer2", "nearest", num_connections=5)
    .set_metadata(description="My topology", version="1.0")
    .build())
```

#### Visualizer

Create and display 3D visualizations:

```python
from braintop.core.visualizer import TopologyVisualizer

visualizer = TopologyVisualizer(topology)
fig = visualizer.create_figure()

# Show in browser
visualizer.show()

# Save to file
visualizer.save("topology.html", format="html")

# Export to NetworkX for analysis
graph = visualizer.get_network_graph()
```

### LLM Integration

Connect to LLMs to extract embeddings and relationships:

```python
from braintop.integrations.llm_connector import LLMConnector
from braintop.utils.builders import TopologyBuilder

# Connect to LLM
llm = LLMConnector(provider="mock", model_name="gpt-4")
llm.connect()

# Create topology from texts
texts = [
    "Neural networks learn patterns",
    "Transformers use attention",
    "Embeddings represent meaning"
]

builder = TopologyBuilder("llm_topology", "LLM Topology")
builder.add_layer_from_llm("concepts", llm, texts)
topology = builder.build()
```

## 🎯 Use Cases

- **LLM Analysis**: Visualize attention patterns, embedding spaces, and layer relationships
- **Concept Mapping**: Create 3D maps of semantic relationships
- **Neural Architecture Visualization**: Display multi-layer neural network topologies
- **Knowledge Graphs**: Visualize hierarchical and networked knowledge structures
- **Agent Coordination**: Display relationships between AI agents in orchestration systems

## 🏗️ Architecture

```
braintop/
├── core/              # Core data structures and visualization
│   ├── topology.py    # Topology, Node, Edge, Layer models
│   └── visualizer.py  # 3D visualization engine
├── layers/            # Layer implementations
│   ├── base.py        # Abstract base layer
│   ├── conceptual.py  # Conceptual/semantic layers
│   └── riemannian.py  # Riemannian manifold layers
├── integrations/      # External system integrations
│   ├── base.py        # Integration interface
│   └── llm_connector.py  # LLM connector
├── utils/             # Utilities
│   └── builders.py    # Topology builder
└── cli.py             # Command-line interface
```

## 📦 Examples

See the `examples/` directory for complete examples:

- `basic_example.py`: Simple topology creation and visualization
- `llm_integration_example.py`: Using LLM embeddings
- `multi_layer_example.py`: Complex multi-layer topology
- `riemannian_example.py`: Different manifold types

## 🔧 Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built with:
- [Plotly](https://plotly.com/) - Interactive 3D visualization
- [NumPy](https://numpy.org/) - Numerical computing
- [NetworkX](https://networkx.org/) - Network analysis
- [Pydantic](https://pydantic.dev/) - Data validation

## 🔮 Future Roadmap

- [ ] Real LLM provider integrations (OpenAI, Anthropic, HuggingFace)
- [ ] Interactive web dashboard with Dash
- [ ] Animation support for temporal evolution
- [ ] Custom shader support for advanced rendering
- [ ] Export to other 3D formats (glTF, OBJ)
- [ ] Real-time streaming updates
- [ ] Collaborative multi-user viewing
