# Braintop Solution Plan

A step-by-step plan, expressed with the actions and constraints of
[PROBLEM_MODEL.md](PROBLEM_MODEL.md), for building a three-layer topology of
transformer concepts and exporting it.

The plan is executable: `examples/solution_plan_implementation.py` carries it
out and asserts every check listed below. All figures in this document are
the output of that script (seed 7, mock LLM provider); the test suite runs it
on every change.

```bash
python examples/solution_plan_implementation.py -o transformer_topology.html
```

## Goal

- 3 layers: one conceptual layer built from text embeddings, one sphere, one
  hyperbolic disk
- Edges within the conceptual layer from embedding similarity, and edges
  between layers
- An interactive HTML figure and a NetworkX graph

The mock provider produces deterministic, hash-based embeddings. They stand
in for a real model so the plan can run offline; the similarities they give
carry no semantic meaning.

## Steps

| # | Action | Preconditions | Effect (measured) |
|---|---|---|---|
| 1 | `CREATE_TOPOLOGY("transformer_topology", ...)` | id and name are non-empty | empty topology |
| 2 | `CONNECT_LLM(provider="mock")` | none | connector is connected |
| 3 | `EXTRACT_EMBEDDINGS(10 texts)` | connected; texts non-blank | 10 × 768 matrix, finite, unit-length rows |
| 4 | `ADD_LAYER_FROM_LLM("concepts_layer", spacing_factor=2.0)` | connected; layer id unused | 10 nodes placed by PCA; 4 similarity edges (cosine ≥ 0.2) |
| 5 | `ADD_RIEMANNIAN_LAYER("sphere_layer", 8, sphere, radius=3)` | layer id unused; 8 > 0 | 8 nodes on the sphere |
| 6 | `ADD_RIEMANNIAN_LAYER("hyperbolic_layer", 12, hyperbolic, curvature=-1)` | layer id unused; curvature < 0 | 12 nodes in the Poincaré disk |
| 7 | `CONNECT_LAYERS(concepts → sphere, nearest, k=3)` | both layers exist and differ | 10 × 3 = 30 edges, weight 1/(1+distance) |
| 8 | `CONNECT_LAYERS(sphere → hyperbolic, nearest, k=4)` | as above | 8 × 4 = 32 edges |
| 9 | `CONNECT_LAYERS(concepts → hyperbolic, random, k=2)` | as above | 10 × 2 = 20 edges, weight 1 |
| 10 | Validate constraints (table below) | topology built | all pass |
| 11 | `CREATE_VISUALIZATION` | at least one non-empty layer | one node trace per layer, 30 nodes drawn |
| 12 | `EXPORT_VISUALIZATION(html)` | figure exists; path writable | 4.9 MB HTML file |
| 13 | `EXPORT_TO_NETWORKX` | topology built | graph with 30 nodes, 86 edges |

## Final state

| | |
|---|---|
| Layers | `concepts_layer` (10 nodes), `sphere_layer` (8), `hyperbolic_layer` (12) |
| Nodes | 30 |
| Edges | 86: 4 similarity, 30 concepts→sphere, 32 sphere→hyperbolic, 20 concepts→hyperbolic |
| Embedding dimension | 768 |
| Variance kept by the 3-D PCA projection | 43.2% |

## Constraint checks

Numbers refer to PROBLEM_MODEL.md section 4. Each row is an assertion in the
script.

| Constraint | Check | Result |
|---|---|---|
| 1 Unique identifiers | layer, node and edge ids contain no duplicates | pass |
| 2 Reference integrity | every node's `layer_id` and every edge endpoint exists | pass |
| 3 Cardinality | no layer is empty | pass |
| 4, 9 Types and numerics | positions are finite 3-tuples; weights finite and ≥ 0 | pass |
| 5 Sphere | every node at distance 3 from the origin | pass, max error 4.4e-16 |
| 5 Hyperbolic | every node has x² + y² < 1 | pass, max radius 0.962 |
| 12 Edge consistency | no self-loops, no repeated (source, target) pair | pass |
| 13 Metadata | topology metadata serializes to JSON | pass |

Not checked, because they are not properties the script can test:

- Constraint 7 (distance preservation) is a soft goal. The measured figure is
  that three principal components keep 43.2% of the variance of these
  embeddings, so the 3-D positions are a lossy view of the 768-D space.
- Constraints 8, 15 and 17 (scalability, visual distinguishability,
  interactivity) depend on the browser and the viewer.

## Known gaps between the model and the code

- Similarity edges from step 4 connect nodes of the same layer but are stored
  in `Topology.inter_layer_edges`; the library has no separate intra-layer
  edge list.
- Sphere and hyperbolic node positions are random samples. The script fixes
  the seed to make the run reproducible; the library itself does not.
- The NetworkX export carries node metadata and edge weights, not node
  positions.
