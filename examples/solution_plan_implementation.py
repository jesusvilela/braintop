#!/usr/bin/env python3
"""Executable version of SOLUTION_PLAN.md.

Builds a three-layer topology from ten transformer concepts using the mock
LLM provider, and checks every constraint from PROBLEM_MODEL.md that the
library can actually violate. Each check is a real assertion: the script
exits with an error if one fails, and every number it prints is measured
from the objects it built.

The run is seeded, so the output is reproducible.

Usage:
    python examples/solution_plan_implementation.py [-o transformer_topology.html]
"""

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

from braintop.core.topology import Topology
from braintop.core.visualizer import TopologyVisualizer
from braintop.integrations.llm_connector import LLMConnector
from braintop.utils.builders import TopologyBuilder

SEED = 7
SPHERE_RADIUS = 3.0
SIMILARITY_THRESHOLD = 0.2
TEXTS = [
    "Self-attention mechanism",
    "Multi-head attention",
    "Positional encoding",
    "Feed-forward network",
    "Layer normalization",
    "Residual connections",
    "Token embeddings",
    "Transformer encoder",
    "Transformer decoder",
    "Attention weights",
]


def check(condition: bool, description: str) -> None:
    """Assert a condition and report it."""
    if not condition:
        raise AssertionError(f"FAILED: {description}")
    print(f"  ok  {description}")


def explained_variance_3d(embeddings: np.ndarray) -> float:
    """Fraction of variance kept by the top three principal components."""
    centered = embeddings - embeddings.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    variance = singular_values ** 2
    return float(variance[:3].sum() / variance.sum())


def build(texts: List[str]) -> Dict[str, object]:
    """Phases 1-4: create the topology, its layers and its connections."""
    print("Phase 1-2: topology and LLM connection")
    builder = TopologyBuilder("transformer_topology", "Transformer Architecture Visualization")
    connector = LLMConnector(
        provider="mock",
        model_name="mock-embedder",
        config={"similarity_threshold": SIMILARITY_THRESHOLD},
    )
    check(connector.connect(), "mock LLM connector connects")
    embeddings = connector.get_embeddings(texts)
    check(embeddings.shape[0] == len(texts), f"one embedding per text, shape {embeddings.shape}")
    check(bool(np.isfinite(embeddings).all()), "embeddings are finite")
    check(
        bool(np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)),
        "embeddings are unit length",
    )

    print("Phase 3: layers")
    builder.add_layer_from_llm("concepts_layer", connector, texts, spacing_factor=2.0)
    llm_edges = len(builder.topology.inter_layer_edges)
    builder.add_riemannian_layer(
        "sphere_layer", num_nodes=8, manifold_type="sphere", radius=SPHERE_RADIUS
    )
    builder.add_riemannian_layer(
        "hyperbolic_layer", num_nodes=12, manifold_type="hyperbolic", curvature=-1.0
    )
    check(len(builder.topology.layers) == 3, "three layers created")

    print("Phase 4: inter-layer connections")
    counts = {"llm_similarity": llm_edges}
    for name, source, target, kind, k in (
        ("concepts_to_sphere", "concepts_layer", "sphere_layer", "nearest", 3),
        ("sphere_to_hyperbolic", "sphere_layer", "hyperbolic_layer", "nearest", 4),
        ("concepts_to_hyperbolic", "concepts_layer", "hyperbolic_layer", "random", 2),
    ):
        before = len(builder.topology.inter_layer_edges)
        builder.connect_layers(source, target, kind, num_connections=k)
        counts[name] = len(builder.topology.inter_layer_edges) - before
        expected = len(builder.topology.get_layer(source).nodes) * k
        check(counts[name] == expected, f"{name}: {counts[name]} edges ({kind}, k={k})")

    return {
        "topology": builder.build(),
        "edge_counts": counts,
        "embedding_dim": int(embeddings.shape[1]),
        "variance_kept": explained_variance_3d(embeddings),
    }


def validate(topology: Topology) -> None:
    """Check the constraints from PROBLEM_MODEL.md section 4."""
    print("Phase 5a: constraint checks")
    nodes = topology.get_all_nodes()
    edges = topology.get_all_edges()
    node_layer = {node.id: node.layer_id for node in nodes}
    layer_ids = [layer.id for layer in topology.layers]

    check(len(set(layer_ids)) == len(layer_ids), "C1 layer ids are unique")
    check(len(node_layer) == len(nodes), "C1 node ids are unique")
    check(len({e.id for e in edges}) == len(edges), "C1 edge ids are unique")
    check(all(n.layer_id in layer_ids for n in nodes), "C2 every node references its layer")
    check(
        all(e.source in node_layer and e.target in node_layer for e in edges),
        "C2 every edge references existing nodes",
    )
    check(all(len(layer.nodes) >= 1 for layer in topology.layers), "C3 no empty layers")
    check(
        all(len(n.position) == 3 and all(math.isfinite(c) for c in n.position) for n in nodes),
        "C4/C9 positions are finite 3-tuples",
    )
    check(
        all(math.isfinite(e.weight) and e.weight >= 0 for e in edges),
        "C4/C9 edge weights are finite and non-negative",
    )
    sphere_error = max(
        abs(math.sqrt(sum(c * c for c in n.position)) - SPHERE_RADIUS)
        for n in topology.get_layer("sphere_layer").nodes
    )
    check(sphere_error < 1e-9, f"C5 sphere nodes lie on radius {SPHERE_RADIUS} "
          f"(max error {sphere_error:.1e})")
    disk_max = max(
        math.hypot(n.position[0], n.position[1])
        for n in topology.get_layer("hyperbolic_layer").nodes
    )
    check(disk_max < 1.0, f"C5 hyperbolic nodes lie inside the unit disk (max r {disk_max:.3f})")
    check(all(e.source != e.target for e in edges), "C12 no self-loops")
    pairs = collections.Counter((e.source, e.target) for e in edges)
    check(max(pairs.values()) == 1, "C12 no duplicate edges")
    json.dumps(topology.metadata)
    check(True, "C13 topology metadata is JSON-serializable")


def export(topology: Topology, output: Path) -> None:
    """Phase 5b: build the figure, save it and export to NetworkX."""
    print("Phase 5b: visualization and export")
    visualizer = TopologyVisualizer(topology)
    figure = visualizer.create_figure()
    node_traces = [t for t in figure.data if t.mode and "markers" in t.mode]
    check(len(node_traces) == len(topology.layers), "one node trace per layer")
    check(
        sum(len(t.x) for t in node_traces) == len(topology.get_all_nodes()),
        "every node is drawn",
    )
    visualizer.save(str(output))
    check(output.stat().st_size > 0, f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")
    graph = visualizer.get_network_graph()
    check(
        graph.number_of_nodes() == len(topology.get_all_nodes()),
        f"NetworkX export has {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} undirected edges",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", default="transformer_topology.html")
    args = parser.parse_args()

    np.random.seed(SEED)
    result = build(TEXTS)
    topology = result["topology"]
    assert isinstance(topology, Topology)
    validate(topology)
    export(topology, Path(args.output))

    stats = topology.compute_statistics()
    print("\nResult")
    print(f"  layers: {[(layer.id, len(layer.nodes)) for layer in topology.layers]}")
    print(f"  nodes: {stats['total_nodes']}, edges: {stats['total_edges']}")
    print(f"  edges by origin: {result['edge_counts']}")
    print(f"  embedding dimension: {result['embedding_dim']}")
    print(f"  variance kept by 3 principal components: {result['variance_kept']:.1%}")


if __name__ == "__main__":
    main()
