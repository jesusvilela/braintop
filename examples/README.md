# BrainTop Examples

This directory contains example scripts demonstrating the various features of BrainTop.

## Running Examples

Make sure you have installed BrainTop and its dependencies:

```bash
cd ..
pip install -e .
```

Then run any example:

```bash
python basic_conceptual.py
python riemannian_manifolds.py
python neural_network.py
python advanced_multi_layer.py
```

## Examples Overview

### 1. basic_conceptual.py
**Simple conceptual hierarchy visualization**

Creates a hierarchy of AI/ML concepts and visualizes them using the ConceptualLayer. This demonstrates:
- Creating hierarchical relationships
- Parent-child concept edges
- Cross-connections between concepts

**Output:** `conceptual_topology.html`

### 2. riemannian_manifolds.py
**Word embeddings on curved manifolds**

Visualizes word embeddings on three different Riemannian manifolds (sphere, hyperbolic, torus). This demonstrates:
- Mapping 2D embeddings to 3D curved surfaces
- Different manifold types
- Semantic relationships between words

**Output:** `riemannian_topology.html`

### 3. neural_network.py
**Transformer architecture visualization**

Visualizes neural network architectures including a transformer model and attention patterns. This demonstrates:
- Creating feed-forward network structures
- Multi-head attention visualization
- Layer-by-layer neural topology

**Output:** `neural_topology.html`, `transformer_architecture.html`

### 4. advanced_multi_layer.py
**Complete LLM agent system**

Advanced example showing multiple topology layers working together to visualize an LLM agent orchestrator system. This demonstrates:
- Multi-layer visualization
- Agent hierarchy (conceptual)
- Semantic embedding space (Riemannian)
- Underlying neural architecture (neural)
- Data transformation utilities

**Output:** `multi_layer_topology.html`, `simple_topology.html`

## Viewing the Visualizations

All examples generate interactive HTML files that you can open in any web browser. The visualizations support:

- **3D rotation** - Click and drag to rotate
- **Zoom** - Scroll to zoom in/out
- **Pan** - Right-click and drag to pan
- **Hover** - Hover over nodes to see details
- **Theme** - Visualizations use different color themes

## Customization

Each example can be modified to:
- Change the number of nodes and edges
- Adjust visualization parameters (node size, edge width, colors)
- Switch between different layer types
- Modify topology structure

Feel free to experiment with the code to create your own custom visualizations!
