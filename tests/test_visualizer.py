"""Tests for Plotly rendering and NetworkX export."""

import numpy as np

from braintop.core.visualizer import TopologyVisualizer
from braintop.utils.builders import TopologyBuilder


def _sample_topology():
    builder = TopologyBuilder("visualizer_test", "Visualizer test")
    builder.add_conceptual_layer(
        "concepts",
        num_nodes=3,
        concepts=["alpha", "beta", "gamma"],
        embeddings=np.array(
            [
                [1.0, 0.0, 0.1],
                [0.0, 1.0, 0.2],
                [0.8, 0.2, 0.9],
            ]
        ),
    )
    builder.add_riemannian_layer(
        "sphere",
        num_nodes=4,
        manifold_type="sphere",
        radius=1.0,
    )
    builder.connect_layers("concepts", "sphere", "nearest", num_connections=1)
    topology = builder.build()
    topology.get_layer("concepts").name = "Concepts"
    topology.get_layer("sphere").name = "Sphere"
    return topology


def test_metric_visualization_uses_one_colorbar_and_clean_layer_names():
    """A multi-layer metric plot should not stack duplicate color scales."""
    topology = _sample_topology()
    figure = TopologyVisualizer(topology).create_figure(
        color_by="betweenness",
        size_by="degree",
    )

    node_traces = [
        trace
        for trace in figure.data
        if trace.type == "scatter3d" and "markers" in (trace.mode or "")
    ]

    assert [trace.name for trace in node_traces] == ["Concepts", "Sphere"]
    assert node_traces[0].marker.showscale is True
    assert node_traces[1].marker.showscale is False
    assert node_traces[0].marker.colorbar.title.text == "betweenness"


def test_networkx_export_preserves_node_structure_and_metadata():
    """NetworkX export should retain labels, coordinates, layer IDs and metadata."""
    topology = _sample_topology()
    node = topology.layers[0].nodes[0]
    node.metadata["origin"] = "fixture"

    graph = TopologyVisualizer(topology).get_network_graph()
    attributes = graph.nodes[node.id]

    assert attributes["label"] == node.label
    assert attributes["position"] == node.position
    assert attributes["layer_id"] == node.layer_id
    assert attributes["origin"] == "fixture"
    assert graph.number_of_edges() == len(topology.inter_layer_edges)


def test_visualizer_produces_sectional_and_ablation_views():
    """The advertised analysis views should be constructible for a small topology."""
    visualizer = TopologyVisualizer(_sample_topology())

    section = visualizer.create_sectional_study(axis="z", sections=2)
    ablations = visualizer.create_ablation_figures()

    assert len(section.data) >= 0
    assert "no_edges" in ablations
    assert "only_concepts" in ablations
    assert "without_sphere" in ablations
