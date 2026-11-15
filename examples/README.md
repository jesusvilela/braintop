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

### solution_plan_implementation.py
Runs the plan in `SOLUTION_PLAN.md` and asserts each of its constraint checks.

### create_demo_gguf.py and visualize_gguf.py
Create a tiny GGUF file and draw its tensor layout. See
`docs/GGUF_VISUALIZATION.md`.

## Output

Each example generates an HTML file that can be opened in a browser to view the interactive 3D visualization.
