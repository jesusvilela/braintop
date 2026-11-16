"""Known-answer and representation controls for convex-cap global bounds.

A cap-radius sweep checks the *analytic envelope* against independently
computed Poincare-disk distances, not against the same production formula.
"""
import math

import numpy as np
import pytest

from braintop.core.correspondence import (
    RadialCapMap, compare_radial_cap, convex_cap_distortion_bound,
)
from braintop.utils.builders import TopologyBuilder


def make_cap(theta, *, lam=0.6, radius=2.0, curvature=-0.64,
             angle_offset=0.3, perturb_target=0.0):
    # A radial pair, a small-angle boundary pair, and two interior controls.
    source = np.asarray([
        [0.0, 0.0],
        [theta, 0.0],
        [theta, 1e-4],
        [0.5 * theta, 1.0],
        [0.25 * theta, 2.3],
    ])
    target = np.asarray([
        [lam * radius * t, (p + angle_offset) % (2 * math.pi)]
        for t, p in source
    ])
    target[2, 1] += perturb_target

    builder = TopologyBuilder("cap", "Convex cap")
    builder.add_riemannian_layer(
        "sphere", len(source), manifold_type="sphere", radius=radius,
        length_unit="source-length", parameters=source,
    )
    builder.add_riemannian_layer(
        "hyperbolic", len(target), manifold_type="hyperbolic",
        curvature=curvature, length_unit="target-length",
        parameters=target,
    )
    spec = RadialCapMap(
        "sphere", "hyperbolic", lam, theta,
        "source-length", "target-length", "controlled-fixture",
        angle_offset=angle_offset,
    )
    pairs = [(f"sphere_node_{i}", f"hyperbolic_node_{i}")
             for i in range(len(source))]
    return builder, spec, pairs


def cap_bound(builder, spec):
    return convex_cap_distortion_bound(
        builder.topology.get_layer("sphere"),
        builder.topology.get_layer("hyperbolic"), spec,
    )


def cap_report(builder, spec, pairs, max_target_error=1e-9):
    return compare_radial_cap(
        builder.topology.get_layer("sphere"),
        builder.topology.get_layer("hyperbolic"), spec, pairs,
        max_target_error=max_target_error,
    )


def disk_reference(source_a, source_b, lam, radius, a, offset):
    """Poincare disk reference, independent of production polar cosines."""
    def image(theta, phi):
        rho = lam * radius * theta
        angle = phi + offset
        r = math.tanh(a * rho / 2)
        return np.array([r * math.cos(angle), r * math.sin(angle)])
    u, v = image(*source_a), image(*source_b)
    return 2 * math.asinh(np.linalg.norm(u-v) /
        math.sqrt((1-np.dot(u, u))*(1-np.dot(v, v)))) / a


@pytest.mark.parametrize("cap", [0.02, 0.1, 0.3, 0.55, 0.9, 1.2, 1.5])
def test_cap_sweep_matches_mathematical_envelope_and_disk_oracle(cap):
    builder, spec, pairs = make_cap(cap)
    bound = cap_bound(builder, spec)
    report = cap_report(builder, spec, pairs)

    assert bound.radial_stretch == 1.0
    assert bound.angular_stretch_max >= 1.0
    assert len(report.pair_remainders) == 10

    source_params = {
        f"sphere_node_{i}": builder.topology.get_layer("sphere").nodes[i]
        .metadata["intrinsic_parameters"] for i in range(5)
    }
    for p in report.pair_remainders:
        if p.source_distance < 1e-12:
            continue
        ratio = p.mapped_target_distance / (spec.radial_scale * p.source_distance)
        assert 1.0 - 1e-10 <= ratio <= bound.angular_stretch_max + 1e-10
        expected = disk_reference(
            source_params[p.source_a], source_params[p.source_b],
            spec.radial_scale, 2.0, 0.8, spec.angle_offset,
        )
        assert p.mapped_target_distance == pytest.approx(
            expected, rel=1e-10, abs=1e-11
        )

    # Infinitesimal angular tangent at cap boundary approaches the sharp bound.
    boundary_pair = next(p for p in report.pair_remainders
                         if {p.source_a, p.source_b}
                         == {"sphere_node_1", "sphere_node_2"})
    ratio = (boundary_pair.mapped_target_distance /
             (spec.radial_scale * boundary_pair.source_distance))
    assert ratio == pytest.approx(bound.angular_stretch_max, rel=2e-6)


def test_upper_bound_monotone_with_cap_radius_and_radial_control():
    radii = [0.02, 0.1, 0.3, 0.55, 0.9, 1.2, 1.5]
    bounds = []
    for cap in radii:
        builder, spec, pairs = make_cap(cap)
        b = cap_bound(builder, spec)
        bounds.append(b.angular_stretch_max)
        radial = next(p for p in cap_report(builder, spec, pairs).pair_remainders
                      if {p.source_a, p.source_b}
                      == {"sphere_node_0", "sphere_node_1"})
        assert radial.mapped_target_distance == pytest.approx(
            spec.radial_scale * radial.source_distance, abs=1e-10
        )
    assert all(a < b for a, b in zip(bounds, bounds[1:]))


def test_equivalent_frames_do_not_change_bound_or_metric_report():
    builder, spec, pairs = make_cap(1.3)
    before_bound = cap_bound(builder, spec)
    before_report = cap_report(builder, spec, pairs)
    rotation = ((0., -1., 0.), (1., 0., 0.), (0., 0., 1.))
    builder.reexpress_layer("sphere", rotation, (7., 2., 3.))
    builder.reexpress_layer("hyperbolic", rotation, (-4., 9., 1.))
    builder.topology.get_layer("sphere").nodes.reverse()
    builder.topology.get_layer("hyperbolic").nodes.reverse()
    assert cap_bound(builder, spec) == before_bound
    assert cap_report(builder, spec, pairs) == before_report


def test_larger_than_hemisphere_refuses_global_bound_without_disabling_local_map():
    builder, spec, pairs = make_cap(2.0)
    report = cap_report(builder, spec, pairs)
    assert len(report.matches) == 5
    with pytest.raises(ValueError, match="strictly convex"):
        cap_bound(builder, spec)


def test_unit_labels_must_bind_to_layer_configuration():
    builder, spec, pairs = make_cap(0.8)
    source = builder.topology.get_layer("sphere")
    source.config["length_unit"] = "other"
    with pytest.raises(ValueError, match="source layer length_unit"):
        cap_report(builder, spec, pairs)
    with pytest.raises(ValueError, match="source layer length_unit"):
        cap_bound(builder, spec)
    source.config.pop("length_unit")
    with pytest.raises(ValueError, match="must declare"):
        cap_report(builder, spec, pairs)
    assert not builder.topology.inter_layer_edges


def test_target_units_and_strict_zero_mismatch_budget_fail_closed():
    builder, spec, pairs = make_cap(1.0, perturb_target=1e-7)
    target = builder.topology.get_layer("hyperbolic")
    target.config["length_unit"] = "wrong"
    with pytest.raises(ValueError, match="target layer length_unit"):
        cap_report(builder, spec, pairs)
    target.config["length_unit"] = "target-length"
    with pytest.raises(ValueError, match="too far"):
        cap_report(builder, spec, pairs, max_target_error=0.0)
    with pytest.raises(ValueError, match="too far"):
        builder.connect_declared_correspondence(spec, pairs, max_target_error=0.0)
    assert len(builder.topology.inter_layer_edges) == 0


def test_large_product_fails_instead_of_overflowing():
    builder, spec, pairs = make_cap(0.5, lam=2.0)
    huge = RadialCapMap(
        spec.source_layer_id, spec.target_layer_id, 2e300,
        spec.cap_theta, spec.source_length_unit, spec.target_length_unit,
        spec.provenance, spec.angle_offset,
    )
    with pytest.raises(ValueError, match="finite numeric range"):
        cap_bound(builder, huge)


def test_near_north_pole_cap_bound_remains_finite():
    builder, spec, _ = make_cap(1e-12)
    bound = cap_bound(builder, spec)
    assert math.isfinite(bound.angular_stretch_max)
    assert bound.angular_stretch_max == pytest.approx(1.0, abs=1e-12)


def test_large_finite_hyperbolic_radius_flows_through_report_and_bound():
    """The analytic envelope can be finite even when naive sinh products overflow."""
    source = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.2, 0.1]])
    target = np.array([[0.0, 0.0], [250.0, 0.0], [500.0, 0.0], [500.0, 0.1]])
    builder = TopologyBuilder("large", "Large finite radius")
    builder.add_riemannian_layer(
        "sphere", 4, "sphere", radius=1.0,
        length_unit="source-length", parameters=source,
    )
    builder.add_riemannian_layer(
        "hyperbolic", 4, "hyperbolic", curvature=-1.0,
        length_unit="target-length", parameters=target,
    )
    spec = RadialCapMap(
        "sphere", "hyperbolic", 2500.0, 0.2,
        "source-length", "target-length", "large-radius-control",
    )
    pairs = [(f"sphere_node_{i}", f"hyperbolic_node_{i}") for i in range(4)]
    bound = cap_bound(builder, spec)
    report = cap_report(builder, spec, pairs, max_target_error=0.0)
    assert math.isfinite(bound.angular_stretch_max)
    assert math.isfinite(report.max_abs_ideal_remainder)
    assert report.max_target_mismatch == 0.0
    assert all(math.isfinite(p.mapped_target_distance) for p in report.pair_remainders)
