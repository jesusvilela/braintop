# Examples

This directory contains example scripts demonstrating how to use Braintop.

## Running Examples

Make sure you have installed braintop first:

```bash
pip install -e .
```

Then run any example:

```bash
python examples/basic_example.py
python examples/llm_integration_example.py
python examples/multi_layer_example.py
python examples/riemannian_example.py
```

## Examples Overview

### basic_example.py
Demonstrates basic topology creation with conceptual and Riemannian layers.

### llm_integration_example.py
Shows how to integrate with LLMs to create topologies from text embeddings.

### multi_layer_example.py
Creates a complex multi-layer topology with various layer types connected together.

### riemannian_example.py
Showcases different Riemannian manifold types (sphere, torus, hyperbolic).

## Output

Each example generates an HTML file that can be opened in a browser to view the interactive 3D visualization.


### showcase.py
Builds two deterministic, publication-friendly views from Braintop's own topology and
visualizer APIs: a side-by-side geometry atlas (sphere, torus, hyperbolic disk) and a
synthetic semantic topology connected to a spherical scaffold.

For interactive HTML only:

```bash
python examples/showcase.py --output-dir build/showcase
```

For HTML and static PNG exports:

```bash
pip install -e ".[static]"
python examples/showcase.py --output-dir build/showcase --require-static
```

The semantic vectors are synthetic demo data, not embeddings from a trained language model.
