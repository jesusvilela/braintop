"""Finite and analytic controls for a cap-dependent, nonconvex upper bound.

These controls do not constitute a proof. The proof is a boundary-detour
construction in docs/CORRESPONDENCE.md. The numerical oracles here use
spherical haversine and Poincare disk coordinates rather than the
production intrinsic polar distance implementation.
"""
import math

import numpy as np
import pytest

from braintop.core.correspondence import (
    RadialCapMap,
    convex_cap_distortion_bound,
    radial_cap_global_envelope,
    radial_cap_global_lower_bound,
)
from braintop.utils.builders import TopologyBuilder


LAMBDA = 0.65
RADIUS = 1.7
CURVATURE = -0.9
OFFSET = 0.2


def make_layers(theta, *, lam=LAMBDA, radius=RADIUS, curvature=CURVATURE):
    builder = TopologyBuilder("detour", "Nonconvex cap upper controls")
    source = np.array([[0.0, 0.0], [theta, 0.0], [theta, math.pi]])
    target = np.array([
        [lam * radius * polar, (azimuth + OFFSET) % (2 * math.pi)]
        for polar, azimuth in source
    ])
    builder.add_riemannian_layer(
        "S", len(source), "sphere", radius=radius,
        length_unit="source-u", parameters=source,
    )
    builder.add_riemannian_layer(
        "H", len(target), "hyperbolic", curvature=curvature,
        length_unit="target-u", parameters=target,
    )
    spec = RadialCapMap(
        "S", "H", lam, theta, "source-u", "target-u", "detour-controls",
        angle_offset=OFFSET,
    )
    return builder, spec


def sphere_haversine(a, b):
    x, px = a
    y, py = b
    h = (math.sin((x-y)/2)**2
         + math.sin(x)*math.sin(y)*math.sin((px-py)/2)**2)
    return 2 * RADIUS * math.asin(math.sqrt(min(1.0, max(0.0, h))))


def hyperbolic_disk(a, b):
    """Independent Poincare disk formula at moderate mapped radii."""
    k = math.sqrt(-CURVATURE)
    def disk(p):
        theta, phi = p
        radius = math.tanh(k * LAMBDA * RADIUS * theta / 2)
        angle = phi + OFFSET
        return np.array([radius*math.cos(angle), radius*math.sin(angle)])
    u, v = disk(a), disk(b)
    return (2/k)*math.asinh(
        float(np.linalg.norm(u-v))
        / math.sqrt((1-float(u@u))*(1-float(v@v)))
    )


@pytest.mark.parametrize("theta", [
    0.1, 0.9, 1.4, math.pi/2, 1.8, 2.3, 2.8, math.pi-0.01
])
def test_global_upper_and_lower_both_hold_against_disk_oracle(theta):
    b, spec = make_layers(theta)
    bound = radial_cap_global_envelope(
        b.topology.get_layer("S"), b.topology.get_layer("H"), spec
    )
    assert bound.lower_stretch == 1.0
    assert bound.upper_stretch >= bound.angular_stretch_max >= 1.0
    assert 1.0 <= bound.detour_factor <= math.pi/2 + 1e-12

    rng = np.random.default_rng(314159)
    params = [
        np.array([0., 0.]),
        np.array([theta, 0.]),
        np.array([theta, math.pi]),
        np.array([theta, 1e-6]),
    ]
    for _ in range(40):
        params.append(np.array([theta*rng.uniform(), rng.uniform(-math.pi, math.pi)]))

    for i, a in enumerate(params):
        for b_ in params[i+1:]:
            d_source = sphere_haversine(a, b_)
            if d_source < 1e-12:
                continue
            ratio = hyperbolic_disk(a, b_) / (LAMBDA*d_source)
            assert ratio >= bound.lower_stretch - 3e-9
            assert ratio <= bound.upper_stretch + 3e-9

    if theta <= math.pi/2:
        assert bound.detour_factor == 1.0
    else:
        assert bound.detour_factor > 1.0


@pytest.mark.parametrize("theta", [0.3, 0.8, 1.2, 1.5])
def test_convex_cap_envelope_matches_prior_sharp_angular_bound(theta):
    b, spec = make_layers(theta)
    sl, hl = b.topology.get_layer("S"), b.topology.get_layer("H")
    bound = radial_cap_global_envelope(sl, hl, spec)
    old = convex_cap_distortion_bound(sl, hl, spec)
    assert bound.detour_factor == 1.0
    assert bound.upper_stretch == pytest.approx(
        old.angular_stretch_max, rel=1e-13
    )


@pytest.mark.parametrize("theta", [1.7, 2.1, 2.8, 3.1])
def test_excluded_south_disk_boundary_detour_is_bounded(theta):
    """A closed-form short-arc control proves the detour's geometric factor."""
    b, spec = make_layers(theta)
    bound = radial_cap_global_envelope(
        b.topology.get_layer("S"), b.topology.get_layer("H"), spec
    )
    alpha = math.pi - theta
    expected_factor = math.pi * math.sin(alpha)/(2*alpha)
    assert bound.detour_factor == pytest.approx(expected_factor, rel=1e-13)

    for gap in [0.01, 0.3, 0.8, 1.3, 2.4, math.pi]:
        boundary_length = RADIUS * math.sin(alpha) * gap
        ambient_short_arc = 2*RADIUS*math.asin(
            math.sin(alpha)*math.sin(gap/2)
        )
        assert boundary_length <= (
            bound.detour_factor * ambient_short_arc + 2e-13
        )
    gap = math.pi
    equality_distance = 2*RADIUS*alpha
    equality_latitude = math.pi*RADIUS*math.sin(alpha)
    assert equality_latitude/equality_distance == pytest.approx(
        bound.detour_factor, rel=1e-13
    )


def test_global_upper_is_cap_dependent_and_not_claimed_sharp():
    caps = [math.pi-0.5, math.pi-0.1, math.pi-0.01, math.pi-1e-4]
    upper = []
    opposite = []
    detours = []
    for theta in caps:
        b, spec = make_layers(theta)
        bound = radial_cap_global_envelope(
            b.topology.get_layer("S"), b.topology.get_layer("H"), spec
        )
        upper.append(bound.upper_stretch)
        detours.append(bound.detour_factor)
        opposite.append(theta/(math.pi-theta))
        assert bound.upper_stretch >= opposite[-1] - 1e-7
    assert all(a < b for a,b in zip(upper, upper[1:]))
    assert all(a < b for a,b in zip(detours, detours[1:]))
    assert upper[-1] > 10000.0


def test_frame_change_does_not_affect_bound_and_units_still_checked():
    b, spec = make_layers(2.4)
    sl, hl = b.topology.get_layer("S"), b.topology.get_layer("H")
    initial = radial_cap_global_envelope(sl, hl, spec)
    turn = ((0., -1., 0.), (1., 0., 0.), (0., 0., 1.))
    b.reexpress_layer("S", turn, (2., 3., 4.))
    b.reexpress_layer("H", turn, (-5., 1., -2.))
    sl.nodes.reverse()
    hl.nodes.reverse()
    assert radial_cap_global_envelope(sl, hl, spec) == initial
    assert radial_cap_global_lower_bound(sl, hl, spec).lower_stretch == 1.0
    hl.config["length_unit"] = "wrong-unit"
    with pytest.raises(ValueError, match="target layer length_unit"):
        radial_cap_global_envelope(sl, hl, spec)


def test_nonfinite_geometric_product_rejected_and_south_pole_excluded():
    b, spec = make_layers(2.1)
    sl, hl = b.topology.get_layer("S"), b.topology.get_layer("H")
    huge = RadialCapMap(
        "S", "H", 2e300, 2.1, "source-u", "target-u", "overflow-control"
    )
    with pytest.raises(ValueError, match="finite numeric range"):
        radial_cap_global_envelope(sl, hl, huge)
    with pytest.raises(ValueError, match="strictly between 0 and pi"):
        RadialCapMap(
            "S", "H", 0.65, math.pi,
            "source-u", "target-u", "excluded-pole"
        )
