"""Analytic, independent-reference, and moving-frame tests for intrinsic metrics.

All test vectors are explicit polar coordinates, not sampled illustrations.
"""
import math

import numpy as np
import pytest

from braintop.core.intrinsic import intrinsic_distance, nearest_intrinsic_pairs
from braintop.core.topology import LayerModel, LayerType, NodeModel
from braintop.utils.builders import TopologyBuilder


def layer_from(kind, params, **config):
    builder = TopologyBuilder("geodesic_lab", "Intrinsic geodesic lab")
    builder.add_riemannian_layer(
        "m",
        num_nodes=len(params),
        manifold_type=kind,
        parameters=np.asarray(params, dtype=float),
        **config
    )
    return builder, builder.topology.get_layer("m")


def test_sphere_known_answers_and_chart_provenance():
    r = 2.5
    _, layer = layer_from("sphere", [
        (0.0, 0.0),
        (math.pi / 2.0, 0.0),
        (math.pi, 0.0),
    ], radius=r)
    assert intrinsic_distance(layer, "m_node_0", "m_node_0") == pytest.approx(0.0, abs=1e-14)
    assert intrinsic_distance(layer, "m_node_0", "m_node_1") == pytest.approx(r * math.pi / 2)
    assert intrinsic_distance(layer, "m_node_0", "m_node_2") == pytest.approx(r * math.pi)

    assert layer.nodes[1].metadata["intrinsic_chart"] == "sphere_polar_v1"
    assert layer.nodes[1].metadata["intrinsic_parameters"] == (math.pi / 2.0, 0.0)
    assert layer.nodes[1].metadata["radial_distance"] == pytest.approx(r)


def test_sphere_nearest_order_matches_chords_for_fixed_radius():
    # On one sphere, d_chord = 2*r*sin(d_geodesic/(2*r)), a monotone map.
    _, layer = layer_from("sphere", [
        (math.pi / 2, 0.0),
        (math.pi / 2, 0.4),
        (math.pi / 2, 1.1),
        (math.pi / 2, 2.2),
    ], radius=1.7)
    source = layer.nodes[0]
    by_intrinsic = sorted(
        layer.nodes[1:],
        key=lambda target: intrinsic_distance(layer, source, target),
    )
    by_chord = sorted(
        layer.nodes[1:],
        key=lambda target: np.linalg.norm(
            np.asarray(source.position) - np.asarray(target.position)
        ),
    )
    assert [n.id for n in by_intrinsic] == [n.id for n in by_chord]


def _disk_reference(a, b, curvature):
    """Separate Poincare disk oracle; unlike the production polar cosines."""
    s = math.sqrt(-curvature)
    def disk(rho, theta):
        t = math.tanh(s * rho / 2.0)
        return np.array([t * math.cos(theta), t * math.sin(theta)])
    u = disk(*a)
    v = disk(*b)
    denom = math.sqrt((1 - np.dot(u, u)) * (1 - np.dot(v, v)))
    return 2.0 * math.asinh(np.linalg.norm(u - v) / denom) / s


@pytest.mark.parametrize("curvature", [-0.25, -1.0, -4.0])
def test_hyperbolic_known_answers_and_independent_disk_oracle(curvature):
    params = [
        (0.0, 0.0), (1.0, 0.0), (1.0, math.pi),
        (2.1, 1.4), (0.5, 0.7),
    ]
    _, layer = layer_from("hyperbolic", params, curvature=curvature, max_radius=3.0)
    assert intrinsic_distance(layer, "m_node_0", "m_node_1") == pytest.approx(1.0)
    assert intrinsic_distance(layer, "m_node_1", "m_node_2") == pytest.approx(2.0)
    assert intrinsic_distance(layer, "m_node_1", "m_node_3") == pytest.approx(
        _disk_reference(params[1], params[3], curvature), rel=1e-12
    )
    assert intrinsic_distance(layer, "m_node_3", "m_node_4") == pytest.approx(
        _disk_reference(params[3], params[4], curvature), rel=1e-12
    )


def test_hyperbolic_display_nearest_disagrees_with_intrinsic_nearest():
    # From rho=4: angular displacement 0.4 looks close in the 3D display,
    # while radial displacement 1 is intrinsically much closer.
    builder, layer = layer_from("hyperbolic", [
        (4.0, 0.0),   # source
        (4.0, 0.4),   # display-near, intrinsic-far
        (3.0, 0.0),   # intrinsically near, display-far
    ], curvature=-1.0, max_radius=5.0)

    source, angular, radial = layer.nodes
    d_angular = intrinsic_distance(layer, source, angular)
    d_radial = intrinsic_distance(layer, source, radial)
    assert d_radial == pytest.approx(1.0)
    assert d_angular > 4.0
    assert d_radial < d_angular

    chord_angular = np.linalg.norm(
        np.asarray(source.position) - np.asarray(angular.position)
    )
    chord_radial = np.linalg.norm(
        np.asarray(source.position) - np.asarray(radial.position)
    )
    assert chord_angular < chord_radial

    neighbors = nearest_intrinsic_pairs(layer, k=1)
    chosen_for_source = [
        (a, b) for a, b, _ in neighbors if "m_node_0" in (a, b)
    ]
    assert ("m_node_0", "m_node_2") in chosen_for_source
    builder.connect_intrinsic_neighbors("m", k=1)
    geo_edges = builder.topology.intra_layer_edges
    assert geo_edges
    assert all(edge.metadata["metric"] == "hyperbolic_geodesic" for edge in geo_edges)


def test_neighbours_invariant_under_passive_frame_change_and_permutation():
    builder, layer = layer_from("hyperbolic", [
        (1.0, 0.0),
        (2.1, 0.3),
        (0.5, 0.7),
        (1.9, 2.5),
    ], curvature=-1.3)
    before = nearest_intrinsic_pairs(layer, k=2)

    rotation = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    builder.reexpress_layer("m", rotation, (9.0, -2.0, 4.0))
    layer.nodes.reverse()  # ordering also changes, geometric metadata does not.
    after = nearest_intrinsic_pairs(layer, k=2)
    assert before == pytest.approx(after) if False else before == after

    before_edges = len(builder.topology.intra_layer_edges)
    builder.connect_intrinsic_neighbors("m", k=2)
    expected_edges = len(nearest_intrinsic_pairs(layer, k=2))
    assert len(builder.topology.intra_layer_edges) == expected_edges
    builder.connect_intrinsic_neighbors("m", k=2)
    assert len(builder.topology.intra_layer_edges) == expected_edges
    assert before_edges == 0


def test_reject_unsupported_geometry_and_missing_provenance():
    _, torus = layer_from("torus", [(0.0, 0.0), (1.0, 1.0)])
    with pytest.raises(NotImplementedError, match="torus"):
        nearest_intrinsic_pairs(torus, k=1)

    _, sphere = layer_from("sphere", [(0.0, 0.0), (1.0, 0.2)])
    sphere.nodes[0].metadata.pop("intrinsic_chart")
    with pytest.raises(ValueError, match="intrinsic_chart"):
        intrinsic_distance(sphere, "m_node_0", "m_node_1")
    with pytest.raises(ValueError, match="intrinsic_chart"):
        nearest_intrinsic_pairs(sphere)


def test_invalid_parameters_and_cross_layer_usage_fail_closed():
    with pytest.raises(ValueError, match="Sphere polar theta"):
        layer_from("sphere", [(-0.1, 0.0)])
    with pytest.raises(ValueError, match="Hyperbolic geodesic radius"):
        layer_from("hyperbolic", [(-0.1, 0.0)])
    with pytest.raises(ValueError, match="positive and finite"):
        layer_from("sphere", [(0.2, 0.5)], radius=float("inf"))

    _, layer = layer_from("sphere", [(0.0, 0.0), (1.0, 0.5)])
    with pytest.raises(ValueError, match="registered nodes"):
        intrinsic_distance(layer, "m_node_0", "other_node")
    with pytest.raises(ValueError, match="non-negative integer"):
        nearest_intrinsic_pairs(layer, k=-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        nearest_intrinsic_pairs(layer, k=1.25)

    builder = TopologyBuilder("same", "same")
    builder.add_riemannian_layer("sphere", 4)
    with pytest.raises(ValueError, match="distinct layers"):
        builder.connect_layers("sphere", "sphere", "nearest", num_connections=1)


def test_no_metric_implicitly_assigned_to_conceptual_layer():
    abstract = LayerModel(
        id="abstract",
        name="Conceptual",
        layer_type=LayerType.CONCEPTUAL,
        nodes=[
            NodeModel(id="a", layer_id="abstract", position=(0.0, 0.0, 0.0)),
            NodeModel(id="b", layer_id="abstract", position=(1.0, 0.0, 0.0)),
        ],
    )
    with pytest.raises(ValueError, match="Riemannian"):
        nearest_intrinsic_pairs(abstract)
