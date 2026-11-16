"""Finite controls for a global lower metric bound on nearly-full spherical caps.

An analytic path argument—not these tests—supports the theorem. The code
checks independent coordinate formulas, nonconvex source caps, paired-target
error budgets, and passive display-frame invariance. Sampling is not proof.
"""
import math

import numpy as np
import pytest

from braintop.core.correspondence import (
    RadialCapMap,
    compare_radial_cap,
    convex_cap_distortion_bound,
    radial_cap_global_lower_bound,
)
from braintop.utils.builders import TopologyBuilder


def cap_fixture(cap, *, radial_scale=0.7, radius=1.3,
                curvature=-1.1, perturb=None):
    source = np.array([
        [0.0, 0.0],            # polar origin
        [cap, 0.0],            # boundary ray
        [cap, math.pi],        # opposite boundary ray
        [0.65 * cap, 0.0],     # radial control
        [0.8 * cap, 1.2],      # angular control
    ], dtype=float)
    offset = 0.2
    target = np.array([
        [radial_scale * radius * theta, (angle + offset) % (2 * math.pi)]
        for theta, angle in source
    ], dtype=float)
    if perturb is not None:
        target[2, 1] += perturb

    builder = TopologyBuilder("lower", "Global lower bound fixture")
    builder.add_riemannian_layer(
        "sphere", 5, manifold_type="sphere", radius=radius,
        length_unit="sphere-unit", parameters=source,
    )
    builder.add_riemannian_layer(
        "hyperbolic", 5, manifold_type="hyperbolic",
        curvature=curvature, length_unit="hyperbolic-unit", parameters=target,
    )
    spec = RadialCapMap(
        source_layer_id="sphere",
        target_layer_id="hyperbolic",
        radial_scale=radial_scale,
        cap_theta=cap,
        source_length_unit="sphere-unit",
        target_length_unit="hyperbolic-unit",
        provenance="nonconvex-global-lower-control",
        angle_offset=offset,
    )
    pairs = [(f"sphere_node_{i}", f"hyperbolic_node_{i}") for i in range(5)]
    return builder, spec, pairs, source, target


def haversine_sphere(a, b, radius):
    """Independent spherical haversine, not production atan2(cross,dot)."""
    th1, ph1 = a
    th2, ph2 = b
    h = math.sin((th1 - th2) / 2) ** 2
    h += math.sin(th1) * math.sin(th2) * math.sin((ph1 - ph2) / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(max(0.0, min(1.0, h))))


def disk_distance(a, b, curvature):
    """Independent hyperbolic reference in Poincare disk coordinates."""
    k = math.sqrt(-curvature)
    def z(p):
        rho, phi = p
        t = math.tanh(k * rho / 2)
        return np.array([t * math.cos(phi), t * math.sin(phi)])
    u, v = z(a), z(b)
    denom = math.sqrt((1 - np.dot(u, u)) * (1 - np.dot(v, v)))
    return 2 * math.asinh(float(np.linalg.norm(u-v)) / denom) / k


@pytest.mark.parametrize("theta_max", [
    0.3, 1.4, math.pi / 2 + 0.1, 2.0, 2.7, math.pi - 1e-6
])
def test_global_lower_bound_extends_past_hemisphere(theta_max):
    builder, spec, pairs, source, target = cap_fixture(theta_max)
    sl, hl = builder.topology.get_layer("sphere"), builder.topology.get_layer("hyperbolic")

    bound = radial_cap_global_lower_bound(sl, hl, spec)
    assert bound.lower_stretch == 1.0
    assert bound.cap_theta == theta_max
    report = compare_radial_cap(sl, hl, spec, pairs)
    assert len(report.pair_remainders) == 10
    for pair in report.pair_remainders:
        assert pair.ideal_signed_remainder >= -1e-9
        i = int(pair.source_a.rsplit("_", 1)[1])
        j = int(pair.source_b.rsplit("_", 1)[1])
        ds = haversine_sphere(source[i], source[j], 1.3)
        dh = disk_distance(target[i], target[j], -1.1)
        assert ds == pytest.approx(pair.source_distance, rel=1e-8, abs=1e-9)
        assert dh == pytest.approx(pair.mapped_target_distance, rel=1e-8, abs=1e-9)
        assert dh + 1e-9 >= spec.radial_scale * ds

    if theta_max >= math.pi / 2:
        with pytest.raises(ValueError, match="strictly convex"):
            convex_cap_distortion_bound(sl, hl, spec)


@pytest.mark.parametrize("epsilon", [0.4, 0.1, 0.01, 1e-3, 1e-5])
def test_south_pole_opposite_ray_controls_and_no_uniform_upper(epsilon):
    theta = math.pi - epsilon
    builder, spec, pairs, source, target = cap_fixture(theta)
    report = compare_radial_cap(
        builder.topology.get_layer("sphere"),
        builder.topology.get_layer("hyperbolic"), spec, pairs
    )
    boundary_pair = next(p for p in report.pair_remainders
                         if {p.source_a, p.source_b}
                         == {"sphere_node_1", "sphere_node_2"})
    # For theta > pi/2, the *ambient sphere* shortest arc runs across
    # the south pole, OUTSIDE this cap. The H geodesic passes via origin.
    expected_sphere = 2 * 1.3 * epsilon
    expected_hyperbolic = 2 * spec.radial_scale * 1.3 * theta
    assert boundary_pair.source_distance == pytest.approx(
        expected_sphere, rel=1e-9, abs=1e-9
    )
    assert boundary_pair.mapped_target_distance == pytest.approx(
        expected_hyperbolic, rel=1e-9, abs=1e-9
    )
    ratio = boundary_pair.mapped_target_distance / (
        spec.radial_scale * boundary_pair.source_distance
    )
    assert ratio == pytest.approx(theta / epsilon, rel=1e-9)
    assert ratio >= 1.0
    # Uniform-in-cap-size upper stretch is impossible as epsilon tends to zero.
    if epsilon <= 1e-3:
        assert ratio > 1000.0


def test_nonconvex_observed_lower_allows_declared_target_mismatch():
    builder, spec, pairs, _, _ = cap_fixture(2.4, perturb=0.012)
    sl, hl = builder.topology.get_layer("sphere"), builder.topology.get_layer("hyperbolic")
    report = compare_radial_cap(sl, hl, spec, pairs, max_target_error=0.1)
    bound = radial_cap_global_lower_bound(sl, hl, spec)
    assert bound.lower_stretch == 1.0
    assert report.max_target_mismatch > 0
    for pair in report.pair_remainders:
        assert pair.observed_target_distance + pair.observed_error_bound + 1e-9 >= (
            spec.radial_scale * pair.source_distance
        )


def test_display_frame_changes_do_not_change_global_lower_bound():
    builder, spec, pairs, _, _ = cap_fixture(math.pi - 1e-4)
    sl, hl = builder.topology.get_layer("sphere"), builder.topology.get_layer("hyperbolic")
    before = radial_cap_global_lower_bound(sl, hl, spec)
    original = compare_radial_cap(sl, hl, spec, pairs)
    Q = ((0., -1., 0.), (1., 0., 0.), (0., 0., 1.))
    builder.reexpress_layer("sphere", Q, (1., 7., -3.))
    builder.reexpress_layer("hyperbolic", Q, (-8., 0., 4.))
    sl.nodes.reverse()
    hl.nodes.reverse()
    assert radial_cap_global_lower_bound(sl, hl, spec) == before
    assert compare_radial_cap(sl, hl, spec, pairs) == original


def test_explicit_contract_rejects_invalid_units_and_geometry():
    builder, spec, pairs, _, _ = cap_fixture(2.0)
    sl, hl = builder.topology.get_layer("sphere"), builder.topology.get_layer("hyperbolic")
    sl.config["length_unit"] = "wrong"
    with pytest.raises(ValueError, match="source layer length_unit"):
        radial_cap_global_lower_bound(sl, hl, spec)
    sl.config["length_unit"] = "sphere-unit"
    hl.config["manifold_type"] = "torus"
    with pytest.raises(ValueError, match="sphere and hyperbolic"):
        radial_cap_global_lower_bound(sl, hl, spec)


def test_cap_at_south_pole_is_invalid_not_a_degenerate_transport():
    builder, spec, pairs, _, _ = cap_fixture(math.pi - 1e-5)
    with pytest.raises(ValueError, match="strictly between 0 and pi"):
        RadialCapMap(
            "sphere", "hyperbolic", 0.7, math.pi,
            "sphere-unit", "hyperbolic-unit", "singular-south-pole"
        )
