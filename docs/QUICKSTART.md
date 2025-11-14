# Braintop Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/jesusvilela/braintop.git
cd braintop

# Install in development mode
pip install -e .

# Or install with dependencies
pip install -r requirements.txt
pip install -e .
```

## Basic Usage

### 1. Create a Simple Topology

```python
from braintop.utils.builders import TopologyBuilder
from braintop.core.visualizer import TopologyVisualizer

# Build topology
builder = TopologyBuilder("my_topology", "My First Topology")

# Add a conceptual layer
concepts = ["Neural Network", "Deep Learning", "Transformer"]
builder.add_conceptual_layer("concepts", num_nodes=3, concepts=concepts)

# Add a sphere layer
builder.add_riemannian_layer("sphere", num_nodes=5, manifold_type="sphere")

# Connect them
builder.connect_layers("concepts", "sphere", "nearest", num_connections=2)

# Build and visualize
topology = builder.build()
visualizer = TopologyVisualizer(topology)
visualizer.show()  # Opens in browser
```

### 2. Use LLM Integration

```python
from braintop.integrations.llm_connector import LLMConnector
from braintop.utils.builders import TopologyBuilder

# Connect to LLM (mock for demo)
llm = LLMConnector(provider="mock")
llm.connect()

# Create topology from texts
texts = [
    "Neural networks learn patterns",
    "Transformers use attention",
    "Embeddings capture meaning"
]

builder = TopologyBuilder("llm_topology", "LLM Topology")
builder.add_layer_from_llm("concepts", llm, texts)
topology = builder.build()

# Visualize
from braintop.core.visualizer import TopologyVisualizer
visualizer = TopologyVisualizer(topology)
visualizer.save("llm_topology.html")
```

### 3. Multi-Layer Topology

```python
from braintop.utils.builders import TopologyBuilder

builder = TopologyBuilder("complex", "Complex Topology")

# Input layer
builder.add_conceptual_layer("input", num_nodes=5, 
                             concepts=["Input 1", "Input 2", "Input 3", "Input 4", "Input 5"])

# Processing layer (sphere)
builder.add_riemannian_layer("processing", num_nodes=10, 
                             manifold_type="sphere", radius=2.0)

# Hidden layer
builder.add_conceptual_layer("hidden", num_nodes=8)

# Output layer (torus)
builder.add_riemannian_layer("output", num_nodes=6, 
                             manifold_type="torus", radius=1.5)

# Connect layers
builder.connect_layers("input", "processing", "nearest", num_connections=3)
builder.connect_layers("processing", "hidden", "random", num_connections=2)
builder.connect_layers("hidden", "output", "all")

topology = builder.build()
```

### 4. Using the CLI

```bash
# Show information
python -m braintop.cli info

# Generate demo visualization
python -m braintop.cli visualize

# Save to file
python -m braintop.cli visualize -o my_topology.html
```

## Layer Types

### Conceptual Layer
Projects high-dimensional data (like embeddings) to 3D using PCA.

```python
builder.add_conceptual_layer(
    "layer_id",
    num_nodes=10,
    concepts=["Label 1", "Label 2", ...],
    embeddings=numpy_array,  # Optional
    projection_method="pca"
)
```

### Riemannian Sphere
Points distributed on a sphere surface.

```python
builder.add_riemannian_layer(
    "layer_id",
    num_nodes=15,
    manifold_type="sphere",
    radius=2.0
)
```

### Riemannian Torus
Points distributed on a torus (donut shape).

```python
builder.add_riemannian_layer(
    "layer_id",
    num_nodes=20,
    manifold_type="torus",
    radius=1.5
)
```

### Riemannian Hyperbolic
Points in hyperbolic space (Poincaré disk model).

```python
builder.add_riemannian_layer(
    "layer_id",
    num_nodes=12,
    manifold_type="hyperbolic",
    curvature=-1.0
)
```

## Connection Types

- **nearest**: Connect each node to k nearest nodes in target layer
- **random**: Connect each node to k random nodes in target layer
- **all**: Connect every source node to every target node

```python
builder.connect_layers("source_layer", "target_layer", "nearest", num_connections=5)
```

## Visualization Options

```python
visualizer = TopologyVisualizer(topology)

# Show in browser
visualizer.show()

# Save to file
visualizer.save("output.html", format="html")

# Get NetworkX graph for analysis
graph = visualizer.get_network_graph()
print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")
```

## Next Steps

- Check the `examples/` directory for complete examples
- Read `docs/ARCHITECTURE.md` for architecture details
- See `CONTRIBUTING.md` to contribute
- Explore the API documentation
