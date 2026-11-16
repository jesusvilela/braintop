"""Independent geometry controls for a declared sphere-cap -> hyperbolic map.

The target parameters are constructed independently of the production map;
the reference hyperbolic distances use the Poincare disk, rather than the
polar hyperbolic cosine law used by the production implementation.
"""
import math

import numpy as np
import pytest

from braintop.core.correspondence import RadialCapMap, compare_radial_cap
from braintop.core.intrinsic import intrinsic_distance
from braintop.utils.builders import TopologyBuilder


def fixture_layers(shift=0.0):
    radius, radial_scale, curvature = 2.0, 0.5, -1.0
    offset = 0.3
    sphere_params = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, math.pi / 2),
        (1.0, 0.0),
    ]
    # Deliberately not obtained from RadialCapMap.map_parameters.
    hyperbolic_params = [
        (radial_scale * radius * theta, (phi + offset) % (2 * math.pi))
        for theta, phi in sphere_params
    ]
    hyperbolic_params[2] = (
        hyperbolic_params[2][0],
        hyperbolic_params[2][1] + shift,
    )
    builder = TopologyBuilder("declared", "Geometric comparison")
    builder.add_riemannian_layer(
        "S", len(sphere_params), "sphere",
        radius=radius, length_unit="source-length",
        parameters=np.asarray(sphere_params),
    )
    builder.add_riemannian_layer(
        "H", len(hyperbolic_params), "hyperbolic",
        curvature=curvature, length_unit="target-length",
        parameters=np.asarray(hyperbolic_params),
    )
    spec = RadialCapMap(
        source_layer_id="S",
        target_layer_id="H",
        radial_scale=radial_scale,
        cap_theta=1.2,
        angle_offset=offset,
        source_length_unit="source-length",
        target_length_unit="target-length",
        provenance="explicit-test-map",
    )
    pairs = [("S_node_" + str(i), "H_node_" + str(i))
             for i in range(len(sphere_params))]
    return builder, spec, pairs


def disk_reference(a, b, curvature=-1.0):
    """Independent Poincare disk oracle for hyperbolic geodesic distances."""
    s = math.sqrt(-curvature)

    def embed(rho, theta):
        radius = math.tanh(s * rho / 2)
        return np.array([radius * math.cos(theta), radius * math.sin(theta)])

    x, y = embed(*a), embed(*b)
    return 2 * math.asinh(
        np.linalg.norm(x-y) /
        math.sqrt((1-np.dot(x,x)) * (1-np.dot(y,y)))
    ) / s


def evaluate(builder, spec, pairs, tolerance=1e-9):
    return compare_radial_cap(
        builder.topology.get_layer("S"),
        builder.topology.get_layer("H"),
        spec, pairs,
        max_target_error=tolerance,
    )


def test_radial_control_zero_remainder_and_nonisometric_angular_pair():
    builder, spec, pairs = fixture_layers()
    report = evaluate(builder, spec, pairs)

    assert len(report.matches) == 4
    assert len(report.pair_remainders) == 6
    assert report.max_target_mismatch == pytest.approx(0.0, abs=1e-12)

    radial = next(p for p in report.pair_remainders
                  if {p.source_a, p.source_b} == {"S_node_1", "S_node_3"})
    angular = next(p for p in report.pair_remainders
                   if {p.source_a, p.source_b} == {"S_node_1", "S_node_2"})
    assert radial.ideal_signed_remainder == pytest.approx(0.0, abs=1e-12)
    assert angular.ideal_signed_remainder > 0.0  # different curvatures
    assert report.max_abs_ideal_remainder >= angular.ideal_signed_remainder
    assert report.max_abs_observed_remainder == pytest.approx(
        report.max_abs_ideal_remainder, abs=1e-12
    )

    source = builder.topology.get_layer("S")
    target = builder.topology.get_layer("H")
    assert angular.source_distance == pytest.approx(
        intrinsic_distance(source, "S_node_1", "S_node_2")
    )
    ta = target.nodes[1].metadata["intrinsic_parameters"]
    tb = target.nodes[2].metadata["intrinsic_parameters"]
    assert angular.mapped_target_distance == pytest.approx(
        disk_reference(ta, tb), rel=1e-12
    )


def test_mismatch_and_triangle_inequality_remainder_envelope():
    builder, spec, pairs = fixture_layers(shift=0.015)
    report = evaluate(builder, spec, pairs, tolerance=0.02)
    assert report.max_target_mismatch > 0.005
    for p in report.pair_remainders:
        assert abs(
            p.observed_signed_remainder - p.ideal_signed_remainder
        ) <= p.observed_error_bound + 1e-9

    with pytest.raises(ValueError, match="too far"):
        evaluate(builder, spec, pairs, tolerance=1e-4)

    with pytest.raises(ValueError, match="too far"):
        builder.connect_declared_correspondence(spec, pairs, max_target_error=1e-4)
    assert builder.topology.inter_layer_edges == []  # transactional failure


def test_full_correspondence_is_invariant_to_ambient_frames_and_node_order():
    builder, spec, pairs = fixture_layers()
    baseline = evaluate(builder, spec, pairs)
    quarter_turn = ((0.0,-1.0,0.0),(1.0,0.0,0.0),(0.0,0.0,1.0))
    builder.reexpress_layer("S", quarter_turn, (7.0,1.0,-3.0))
    builder.reexpress_layer("H", quarter_turn, (-4.0,5.0,2.0))
    builder.topology.get_layer("S").nodes.reverse()
    builder.topology.get_layer("H").nodes.reverse()
    after = evaluate(builder, spec, pairs)
    assert after == baseline


def test_audit_precedes_cross_layer_edge_commit_and_is_idempotent():
    builder, spec, pairs = fixture_layers()
    builder.connect_declared_correspondence(spec, pairs)
    edges = builder.topology.inter_layer_edges
    assert len(edges) == len(pairs)
    assert all(e.metadata["relation"] == "declared_radial_cap" for e in edges)
    assert all(e.metadata["weight_semantics"] == "unweighted_declared_correspondence" for e in edges)
    assert all(e.weight == 1.0 for e in edges)
    builder.connect_declared_correspondence(spec, pairs)
    assert len(edges) == len(pairs)


def test_existing_incompatible_edge_causes_no_partial_commit():
    builder, spec, pairs = fixture_layers()
    builder.connect_layers("S", "H", "all")
    original = len(builder.topology.inter_layer_edges)
    with pytest.raises(ValueError, match="incompatible edge"):
        builder.connect_declared_correspondence(spec, pairs)
    assert len(builder.topology.inter_layer_edges) == original


@pytest.mark.parametrize("field,value", [
    ("source_length_unit", ""),
    ("target_length_unit", ""),
    ("provenance", ""),
    ("radial_scale", 0),
    ("radial_scale", float("inf")),
    ("cap_theta", math.pi),
    ("cap_theta", float("nan")),
    ("angle_offset", float("inf")),
])
def test_explicit_map_rejects_missing_or_invalid_contract_fields(field, value):
    _, spec, _ = fixture_layers()
    vals = dict(spec.__dict__)
    vals[field] = value
    with pytest.raises(ValueError):
        RadialCapMap(**vals)


def test_fail_closed_for_bad_pairs_and_cap_domain():
    builder, spec, pairs = fixture_layers()
    src = builder.topology.get_layer("S")
    tgt = builder.topology.get_layer("H")

    with pytest.raises(ValueError, match="one-to-one"):
        evaluate(builder, spec, [pairs[0], (pairs[0][0], pairs[1][1])])
    with pytest.raises(ValueError, match="registered"):
        evaluate(builder, spec, [("missing", pairs[0][1])])
    with pytest.raises(ValueError, match="at least one"):
        evaluate(builder, spec, [])
    with pytest.raises(ValueError, match="non-negative"):
        evaluate(builder, spec, pairs, tolerance=-1.0)

    src.nodes[3].metadata["intrinsic_parameters"] = (1.5, 0.0)
    with pytest.raises(ValueError, match="outside the declared cap"):
        compare_radial_cap(src, tgt, spec, pairs)


def test_no_implicit_mixed_manifold_transport():
    builder, spec, pairs = fixture_layers()
    builder.add_riemannian_layer(
        "torus", 2, "torus",
        parameters=np.array([[0.0,0.0],[1.0,1.0]])
    )
    with pytest.raises(ValueError, match="Only explicit sphere-cap"):
        compare_radial_cap(builder.topology.get_layer("torus"),
                           builder.topology.get_layer("H"),
                           RadialCapMap(
                               "torus", "H", 1.0, 1.0, "a", "b", "declared"
                           ),
                           [("torus_node_0","H_node_0")])
    # General ambient cross-layer links still exist, but are a different API.
    builder.connect_layers("S", "H", "nearest", num_connections=1)
    assert all(e.metadata["metric"] == "ambient_euclidean_3d"
               for e in builder.topology.inter_layer_edges)
    assert not any("relation" in e.metadata
                   for e in builder.topology.inter_layer_edges)
