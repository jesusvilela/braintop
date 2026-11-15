"""Known-answer and metamorphic controls for Braintop's ambient frame contract."""
import numpy as np
import pytest

from braintop.core.frames import RigidFrame3D
from braintop.core.topology import Topology, LayerModel, LayerType, NodeModel, EdgeModel
from braintop.core.visualizer import TopologyVisualizer
from braintop.utils.builders import TopologyBuilder


Q90 = ((0., -1., 0.), (1., 0., 0.), (0., 0., 1.))
IDENTITY = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))


def two_layer_builder():
    """Fixed points, no stochastic sampler or projection."""
    builder = TopologyBuilder("frames", "Frame controls")
    builder.topology.add_layer(LayerModel(
        id="source", name="Source", layer_type=LayerType.CONCEPTUAL,
        nodes=[
            NodeModel(id="s0", layer_id="source", position=(0., 0., 0.)),
            NodeModel(id="s1", layer_id="source", position=(4., 0., 0.)),
        ],
    ))
    builder.topology.add_layer(LayerModel(
        id="target", name="Target", layer_type=LayerType.RIEMANNIAN,
        nodes=[
            NodeModel(id="t0", layer_id="target", position=(0.1, 0., 0.)),
            NodeModel(id="t1", layer_id="target", position=(-2., 0., 0.)),
            NodeModel(id="t2", layer_id="target", position=(4.1, 0., 0.)),
        ],
    ))
    return builder


def pairs(topology):
    return [(e.source, e.target) for e in topology.inter_layer_edges]


def test_known_world_transform_and_validation():
    frame = RigidFrame3D(rotation=Q90, translation=(3., 4., 5.))
    assert np.allclose(frame.to_world((2., 1., 0.)), (2., 6., 5.))
    with pytest.raises(ValueError, match="determinant"):
        RigidFrame3D(rotation=((-1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    with pytest.raises(ValueError, match="orthonormal"):
        RigidFrame3D(rotation=((2., 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    with pytest.raises(ValueError, match="finite"):
        RigidFrame3D(translation=(float("nan"), 0., 0.))


def test_passive_coordinate_change_preserves_world_and_connections():
    a = two_layer_builder()
    a.connect_layers("source", "target", "nearest", num_connections=1)
    original = pairs(a.topology)
    original_world = [a.topology.world_position(n) for n in a.topology.get_all_nodes()]
    # New coordinates y = Qx + s; compensate physical frame exactly.
    b = two_layer_builder()
    b.reexpress_layer("target", Q90, (2., -3., 4.))
    changed_world = [b.topology.world_position(n) for n in b.topology.get_all_nodes()]
    assert np.allclose(changed_world, original_world, rtol=0, atol=1e-12)
    b.connect_layers("source", "target", "nearest", num_connections=1)
    assert pairs(b.topology) == original
    assert [e.weight for e in a.topology.inter_layer_edges] == pytest.approx(
        [e.weight for e in b.topology.inter_layer_edges], abs=1e-12
    )


def test_global_rigid_motion_preserves_nearest_assignments():
    a = two_layer_builder()
    a.connect_layers("source", "target", "nearest", num_connections=1)
    b = two_layer_builder()
    b.set_layer_frame("source", Q90, (20., -3., 1.))
    b.set_layer_frame("target", Q90, (20., -3., 1.))
    b.connect_layers("source", "target", "nearest", num_connections=1)
    assert pairs(a.topology) == pairs(b.topology)


def test_relative_movement_can_change_correspondence():
    a = two_layer_builder()
    a.connect_layers("source", "target", "nearest", num_connections=1)
    b = two_layer_builder()
    b.set_layer_frame("target", IDENTITY, (3., 0., 0.))
    b.connect_layers("source", "target", "nearest", num_connections=1)
    assert pairs(a.topology)[0] == ("s0", "t0")
    assert pairs(b.topology)[0] == ("s0", "t1")


def test_renderer_and_networkx_respect_world_frames():
    b = two_layer_builder()
    b.set_layer_frame("target", Q90, (3., 4., 5.))
    b.connect_layers("source", "target", "nearest", num_connections=1)
    topology = b.build()
    visualizer = TopologyVisualizer(topology)
    figure = visualizer.create_figure()
    plotted = next(t for t in figure.data if t.name == "Target")
    expected = [topology.world_position(n) for n in topology.get_layer("target").nodes]
    assert np.allclose(np.stack([plotted.x, plotted.y, plotted.z], axis=1), expected)
    graph = visualizer.get_network_graph()
    assert graph.nodes["t0"]["position"] == (0.1, 0., 0.)
    assert np.allclose(graph.nodes["t0"]["world_position"], expected[0])
    section = visualizer.create_sectional_study(axis="z", sections=2)
    assert section.data


def test_semantic_edges_are_intralayer_but_are_returned_in_full_graph():
    topology = Topology("semantic", "Semantic")
    topology.add_layer(LayerModel(
        id="concepts", name="Concepts", layer_type=LayerType.CONCEPTUAL,
        nodes=[NodeModel(id=x, layer_id="concepts", position=(0., 0., 0.))
               for x in ("a", "b")],
    ))
    topology.add_intra_layer_edge(EdgeModel(id="e0", source="a", target="b", weight=0.7))
    assert topology.compute_statistics()["intra_layer_edges"] == 1
    assert len(topology.inter_layer_edges) == 0
    assert len(topology.get_all_edges()) == 1
    graph = TopologyVisualizer(topology).get_network_graph()
    assert graph.edges["a", "b"]["role"] == "intra"
    assert graph.edges["a", "b"]["weight"] == pytest.approx(.7)


def test_weighted_centrality_requires_explicit_cost_conversion():
    topology = Topology("weighted", "Weighted")
    topology.add_layer(LayerModel(
        id="l", name="L", layer_type=LayerType.CONCEPTUAL,
        nodes=[NodeModel(id=x, layer_id="l", position=(float(i), 0., 0.))
               for i, x in enumerate("abc")],
    ))
    for i, (a, b, strength) in enumerate(
        [("a", "b", .9), ("b", "c", .9), ("a", "c", .1)]
    ):
        topology.add_intra_layer_edge(
            EdgeModel(id=f"e{i}", source=a, target=b, weight=strength)
        )
    plain = TopologyVisualizer(topology)
    hops = plain._compute_node_metrics(topology.get_all_nodes(), topology.get_all_edges())
    weighted = TopologyVisualizer(topology, config={"centrality_distance": "inverse_strength"})
    inverse = weighted._compute_node_metrics(topology.get_all_nodes(), topology.get_all_edges())
    assert hops["b"]["betweenness"] == pytest.approx(0.)
    assert inverse["b"]["betweenness"] == pytest.approx(1.)
    with pytest.raises(ValueError, match="positive"):
        topology.intra_layer_edges[0].weight = 0.
        weighted.create_figure()


def test_global_color_range_and_batched_edge_traces():
    b = two_layer_builder()
    b.connect_layers("source", "target", "all")
    figure = TopologyVisualizer(b.build()).create_figure(color_by="degree")
    nodes = [t for t in figure.data if "markers" in (t.mode or "")]
    assert len(nodes) == 2
    assert nodes[0].marker.cmin == nodes[1].marker.cmin
    assert nodes[0].marker.cmax == nodes[1].marker.cmax
    assert sum(t.marker.showscale is True for t in nodes) == 1
    lines = [t for t in figure.data if t.mode == "lines"]
    assert len(lines) <= 6
    # Batched traces still contain three entries per edge: start, end, separator.
    assert sum(len(t.x) for t in lines) == 3 * len(b.topology.inter_layer_edges)


def test_topology_rejects_mislabeled_edges_and_duplicate_ids():
    b = two_layer_builder()
    topo = b.topology
    with pytest.raises(ValueError, match="Duplicate layer"):
        topo.add_layer(LayerModel(
            id="source", name="duplicate", layer_type=LayerType.CONCEPTUAL
        ))
    with pytest.raises(ValueError, match="Node IDs must be unique"):
        topo.add_layer(LayerModel(
            id="other", name="other", layer_type=LayerType.CONCEPTUAL,
            nodes=[NodeModel(id="s0", layer_id="other", position=(0., 0., 0.))],
        ))
    with pytest.raises(ValueError, match="distinct layers"):
        topo.add_inter_layer_edge(EdgeModel(id="wrong", source="s0", target="s1"))
    with pytest.raises(ValueError, match="share a layer"):
        topo.add_intra_layer_edge(EdgeModel(id="wrong", source="s0", target="t0"))
    with pytest.raises(ValueError, match="existing nodes"):
        topo.add_intra_layer_edge(EdgeModel(id="wrong", source="s0", target="missing"))

    topo.add_intra_layer_edge(EdgeModel(
        id="same", source="s0", target="s1", metadata={"weight": 123.0, "role": "fake"},
        weight=0.75
    ))
    with pytest.raises(ValueError, match="Duplicate edge"):
        topo.add_intra_layer_edge(EdgeModel(id="same", source="s0", target="s1"))

    graph = TopologyVisualizer(topo).get_network_graph()
    assert graph.edges["s0", "s1"]["role"] == "intra"
    assert graph.edges["s0", "s1"]["weight"] == pytest.approx(0.75)


def test_llm_generated_similarity_relations_are_intralayer():
    from braintop.integrations.llm_connector import LLMConnector

    connector = LLMConnector(
        provider="mock",
        config={"similarity_threshold": 0.9, "top_k": 1},
    )
    connector.connect()
    builder = TopologyBuilder("semantic-links", "Semantic links")
    builder.add_layer_from_llm("texts", connector, ["identical", "identical"])
    topology = builder.build()

    assert len(topology.intra_layer_edges) == 1
    assert len(topology.inter_layer_edges) == 0
    assert topology.compute_statistics()["total_edges"] == 1
    assert TopologyVisualizer(topology).get_network_graph().number_of_edges() == 1


def test_legacy_positional_topology_constructor_preserves_metadata_slot():
    """Do not mistake the historical fifth positional argument for intra-layer edges."""
    topology = Topology("old", "Legacy", [], [], {"source": "legacy"})
    assert topology.metadata == {"source": "legacy"}
    assert topology.intra_layer_edges == []
