<div align="center">

# braintop 🧠

### a small observatory for concepts, vectors & geometric relations

*Place a structure. Choose a geometry. Turn it slowly. See what becomes visible.*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-8A9A5B.svg)](LICENSE)

</div>

---

> **間 — ma.** The space between things is not necessarily empty. Sometimes it is where the relation appears.

Braintop is a Python workbench for building and exploring multi-layer 3D topologies. It brings together conceptual vectors, graph relationships and selected geometric spaces in one inspectable representation.

Not another claim that a picture explains a mind. Not a black box with a prettier dashboard.

Just points, layers, edges, geometry and the freedom to ask a better question.

## the shape of the thing

A topology is a container. Inside it live layers; inside each layer, nodes with positions, labels and metadata. Edges connect nodes, including nodes that belong to different layers.

~~~mermaid
flowchart TD
    T["Topology"] --> L1["Conceptual layer"]
    T --> L2["Riemannian layer"]
    T --> E["Edges + metadata"]
    L1 --> P["Vectors projected to 3D"]
    L2 --> G["Sphere · Torus · Hyperbolic"]
    P --> V["Interactive 3D view"]
    G --> V
    E --> V
    V --> N["NetworkX graph"]
~~~

The architecture stays deliberately modular:

~~~mermaid
flowchart LR
    I["Texts / vectors / labels"] --> C["LLMConnector"]
    C --> B["TopologyBuilder"]
    A["ConceptualLayer"] --> B
    R["RiemannianLayer"] --> B
    B --> T["Topology"]
    T --> P["Plotly visualizer"]
    T --> X["NetworkX"]
    BL["BaseLayer"] -. "extend" .-> A
    BL -. "extend" .-> R
~~~

A layer is a choice of representation, not a claim that every kind of data naturally belongs to the same space.

## geometry, not decoration

| Layer | What it does | Current representation |
|---|---|---|
| **Conceptual** | Places labels and supplied high-dimensional vectors in a shared view | Centred SVD/PCA projection to three coordinates; random projection is the fallback |
| **Sphere** | Samples points on a positively curved surface | Spherical coordinates mapped to 3D Cartesian coordinates |
| **Torus** | Gives periodic structure a place to live | Toroidal coordinates with configurable major and minor radii |
| **Hyperbolic** | Explores negatively curved, tree-friendly geometry | Finite geodesic disk in the Poincaré disk model, with a depth coordinate for display |
| **Custom** | Makes room for another geometry | Extend the BaseLayer interface |

A few implementation details matter:

- Sphere samples are generated uniformly over the surface.
- Torus sampling accounts for the surface-area element rather than treating both angles as uniformly interchangeable.
- Hyperbolic samples use an area-based distribution over a finite geodesic disk, with negative curvature and a configurable maximum radius.

The point is not to flatten every geometry into a single story. It is to let different structures sit beside one another, while keeping their construction visible.

## from vectors to relations

Braintop can work with embeddings that you already have, or receive them through the LLM connector.

The connector provides a small, provider-neutral contract:

- embeddings as finite two-dimensional arrays, optionally L2-normalized;
- cosine-similarity relationships, with a threshold and optional top-k selection;
- normalized inference responses;
- optional attention-like matrices, validated as square, finite, non-negative and row-stochastic;
- deterministic mock behaviour for offline examples and tests;
- packaged adapters for OpenAI-compatible clients, SentenceTransformers and llama.cpp;
- caller-supplied functions or client objects for other systems.

Install only the adapter you intend to use:

```bash
pip install -e ".[openai]"
pip install -e ".[sentence-transformers]"
pip install -e ".[llama-cpp]"
```

The mock provider is intentionally modest: it builds deterministic hash-based vectors from tokens and adjacent token pairs. It is useful for exercising the pipeline without credentials or network access. It is **not** a language model and should not be mistaken for semantic model embeddings.

Custom functions can connect your own embedding or inference stack without making Braintop depend on a particular vendor SDK.

## a first orbit

Install from source:

~~~bash
git clone https://github.com/jesusvilela/braintop.git
cd braintop
pip install -e .
~~~

For development:

~~~bash
pip install -e ".[dev]"
~~~

Build a topology from supplied vectors, add a curved layer, connect them and save the view:

~~~python
import numpy as np

from braintop.core.visualizer import TopologyVisualizer
from braintop.utils.builders import TopologyBuilder

concepts = [
    "attention",
    "embedding",
    "transformer",
    "decoder",
    "encoder",
]

# Replace these with embeddings from your own model or data pipeline.
vectors = np.random.default_rng(7).normal(size=(len(concepts), 64))

builder = TopologyBuilder("first_orbit", "A first orbit")

builder.add_conceptual_layer(
    "concepts",
    num_nodes=len(concepts),
    concepts=concepts,
    embeddings=vectors,
    projection_method="pca",
)

builder.add_riemannian_layer(
    "sphere",
    num_nodes=12,
    manifold_type="sphere",
    radius=1.5,
)

builder.add_riemannian_layer(
    "hyperbolic",
    num_nodes=16,
    manifold_type="hyperbolic",
    curvature=-1.0,
    max_radius=3.0,
)

builder.connect_layers("concepts", "sphere", "nearest", num_connections=3)
builder.connect_layers("sphere", "hyperbolic", "random", num_connections=2)

topology = builder.build()

visualizer = TopologyVisualizer(topology)
visualizer.create_figure(color_by="degree", size_by="betweenness")
visualizer.save("first_orbit.html")

graph = visualizer.get_network_graph()
print(graph.number_of_nodes(), graph.number_of_edges())
~~~

Open first_orbit.html in a browser. Hover over nodes to inspect labels, layer information, metadata and graph metrics.

## bring your own model

The mock bridge is enough to explore the shape of the pipeline:

~~~python
from braintop.integrations.llm_connector import LLMConnector
from braintop.utils.builders import TopologyBuilder
from braintop.core.visualizer import TopologyVisualizer

texts = [
    "Transformers use attention mechanisms",
    "Embeddings represent semantic meaning",
    "Neural networks learn patterns from data",
    "Language models generate text",
]

connector = LLMConnector(provider="mock", model_name="offline-demo")
connector.connect()

builder = TopologyBuilder("text_topology", "Text relationships")
builder.add_layer_from_llm("text", connector, texts)
topology = builder.build()

visualizer = TopologyVisualizer(topology)
visualizer.create_figure()
visualizer.save("text_topology.html")

connector.disconnect()
~~~

For real data, pass your own embedding callable or a client that exposes the methods you need. The connector does not magically inspect arbitrary model internals: access to embeddings, inference or attention depends on what your model client actually exposes.

A custom embedding function can be as small as:

~~~python
def embedding_fn(texts):
    # Return one finite vector per text.
    # Connect your preferred local model, service or embedding pipeline here.
    return my_embedding_model.encode(texts)

connector = LLMConnector(
    provider="custom",
    model_name="my-embedder",
    config={"embedding_fn": embedding_fn},
)
connector.connect()
~~~

See [the inference bridge specification](docs/LLM_INFERENCE_BRIDGE_SPEC.md) for the data contract and validation rules.

## look closer

The Plotly visualizer can:

- render interactive 3D nodes, labels and weighted edges;
- colour or size nodes by layer, degree, betweenness, closeness, salience or normalized radius;
- create coordinate-based sectional studies along the x, y or z axis;
- generate ablation views with edges removed, individual layers isolated, or a layer omitted;
- export the topology as a NetworkX graph for further analysis;
- save interactive HTML; static PNG export is available through the optional `static` extra.

To generate two deterministic showcase views from Braintop's own topology and visualizer
APIs — a geometry atlas and a synthetic semantic topology — run:

```bash
pip install -e ".[static]"
python examples/showcase.py --output-dir build/showcase --require-static
```

The HTML files are interactive; the PNG files are convenient for documentation and papers.
The semantic example uses synthetic vectors, not embeddings from a trained model.

The command line is intentionally small:

~~~bash
braintop info
braintop visualize
braintop visualize -o topology.html
braintop visualize -o topology.html --sectional --section-axis z --sections 4
braintop visualize -o topology.html --ablations
~~~

## a note on interpretation

A rendered coordinate is a view of a structure, not the structure in its entirety.

- PCA is a dimensionality-reduction view. It can discard information; distances and neighbourhoods in three dimensions are not guaranteed to preserve those in the original embedding space.
- The hyperbolic display uses the Poincaré disk radial coordinate in x/y and encodes geodesic depth in z. That 3D display is **not an isometric embedding** of the hyperbolic plane.
- The builder's nearest connection rule compares the stored 3D coordinates with Euclidean distance. It is a practical wiring rule, not a geodesic-distance calculation across different manifolds.
- Mock vectors are deterministic test data, not evidence about language or model semantics.

These distinctions are not footnotes to hide. They are part of the instrument.

Use Braintop to inspect, compare and formulate hypotheses. If a visual pattern matters, test it in the underlying representation with the metric and assumptions that actually apply.

## the small map

~~~text
src/braintop/
├── core/
│   ├── topology.py       # topology, layer, node and edge models
│   └── visualizer.py     # Plotly views, sections, ablations, NetworkX export
├── layers/
│   ├── base.py           # extension interface
│   ├── conceptual.py    # vector projection
│   └── riemannian.py    # sphere, torus and hyperbolic layers
├── integrations/
│   ├── base.py
│   ├── adapters.py
│   └── llm_connector.py
├── utils/
│   └── builders.py      # fluent topology construction
└── cli.py
~~~

## examples & notes

Start with the runnable scripts:

- [Basic topology](examples/basic_example.py)
- [LLM connector](examples/llm_integration_example.py)
- [Multi-layer topology](examples/multi_layer_example.py)
- [Riemannian geometries](examples/riemannian_example.py)

Further reading:

- [Quick start](docs/QUICKSTART.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Inference bridge contract](docs/LLM_INFERENCE_BRIDGE_SPEC.md)
- [Contributing](CONTRIBUTING.md)

## development

~~~bash
pip install -e ".[dev]"
pytest
black src/ tests/
flake8 src/
mypy src/
~~~

Braintop is an evolving project. The current implementation is a compact geometric and graph-visualization workbench; it is not a general-purpose model-inspection framework, and its API may change as the experiments mature.

## license

MIT. See [LICENSE](LICENSE).

---

<div align="center">

*Less spectacle. More structure. Keep turning the object until the question changes.*

</div>
