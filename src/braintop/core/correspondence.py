"""An explicit, non-canonical correspondence between spherical caps and H^2.

This is a finite-sample comparison of two specified Riemannian metrics.
It is NOT parallel transport, a connection, a canonical equivalence, or an
isometry between spaces of different Gaussian curvature.

A declared radial map sends sphere polar coordinates (theta, phi) to
hyperbolic geodesic polar coordinates (lambda*r*theta, phi+offset).
Only explicitly paired, registered source and target nodes are compared.
"""

from dataclasses import dataclass
from math import isfinite, pi, sin, sinh
from typing import List, Sequence, Tuple

import numpy as np

from braintop.core.intrinsic import _distance, _metric, _parameters
from braintop.core.topology import LayerModel, LayerType


@dataclass(frozen=True)
class RadialCapMap:
    """A user-chosen smooth radial map on a spherical cap, not an isometry.

    radial_scale is numerically in *target length units per source length unit*.
    Units and provenance are mandatory; no implicit physical unit conversion.
    cap_theta must remain strictly below the south pole, where polar angle
    coordinates become singular.
    """

    source_layer_id: str
    target_layer_id: str
    radial_scale: float
    cap_theta: float
    source_length_unit: str
    target_length_unit: str
    provenance: str
    angle_offset: float = 0.0

    def __post_init__(self) -> None:
        for field in (
            "source_layer_id", "target_layer_id",
            "source_length_unit", "target_length_unit", "provenance"
        ):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise ValueError(f"{field} must be a nonempty explicit string")
        if self.source_layer_id == self.target_layer_id:
            raise ValueError("A cross-manifold map requires distinct layer IDs")
        if not isfinite(self.radial_scale) or self.radial_scale <= 0:
            raise ValueError("radial_scale must be positive and finite")
        if not isfinite(self.cap_theta) or not 0 < self.cap_theta < pi:
            raise ValueError("cap_theta must lie strictly between 0 and pi")
        if not isfinite(self.angle_offset):
            raise ValueError("angle_offset must be finite")

    def map_parameters(self, source_layer: LayerModel, theta: float, phi: float) -> Tuple[float, float]:
        """Map explicitly supplied sphere polar coordinates into H^2 polar coordinates."""
        if source_layer.id != self.source_layer_id:
            raise ValueError("Source layer ID does not match correspondence")
        kind, radius = _metric(source_layer)
        if kind != "sphere":
            raise ValueError("RadialCapMap requires a sphere source layer")
        if not np.isfinite([theta, phi]).all() or theta < 0 or theta > self.cap_theta:
            raise ValueError("Source polar coordinates are outside the declared cap")
        rho = self.radial_scale * radius * theta
        psi = (phi + self.angle_offset) % (2 * pi)
        if not isfinite(rho) or not isfinite(psi):
            raise ValueError("Mapped coordinates are outside the finite numeric domain")
        return float(rho), float(psi)


@dataclass(frozen=True)
class NodeMatch:
    source_id: str
    target_id: str
    mismatch: float  # target geodesic units


@dataclass(frozen=True)
class PairRemainder:
    source_a: str
    source_b: str
    source_distance: float  # source units
    mapped_target_distance: float  # target units
    observed_target_distance: float  # target units
    ideal_signed_remainder: float  # mapped distance - scale*source distance
    observed_signed_remainder: float  # observed distance - scale*source distance
    observed_error_bound: float  # mismatch(a)+mismatch(b), target units


@dataclass(frozen=True)
class CorrespondenceReport:
    spec: RadialCapMap
    matches: Tuple[NodeMatch, ...]
    pair_remainders: Tuple[PairRemainder, ...]
    max_target_mismatch: float
    max_abs_ideal_remainder: float
    max_abs_observed_remainder: float



def _check_length_units(
    source_layer: LayerModel, target_layer: LayerModel, spec: RadialCapMap
) -> None:
    """Bind unit declarations to actual layers; do not silently convert units.

    This is a consistency check on declared labels, not dimensional analysis.
    radial_scale remains an explicit user-supplied target/source conversion.
    """
    for role, layer, expected in (
        ("source", source_layer, spec.source_length_unit),
        ("target", target_layer, spec.target_length_unit),
    ):
        actual = layer.config.get("length_unit")
        if not isinstance(actual, str) or not actual.strip():
            raise ValueError(
                f"{role} layer must declare config['length_unit'] to audit correspondence"
            )
        if actual != expected:
            raise ValueError(
                f"{role} layer length_unit {actual!r} does not match "
                f"correspondence {expected!r}"
            )


@dataclass(frozen=True)
class RadialCapGlobalLowerBound:
    """Co-Lipschitz distance bound for the declared radial map on any cap <pi.

    For ALL source points p,q in the cap, irrespective of whether the
    spherical minimizing geodesic lies inside it:

        d_H(F(p), F(q)) >= radial_scale * d_S(p,q).

    The target disk's geodesic convexity, not source-cap convexity,
    establishes the inequality. The output is not an upper bound.
    """

    source_layer_id: str
    target_layer_id: str
    cap_theta: float
    lower_stretch: float
    source_length_unit: str
    target_length_unit: str


def radial_cap_global_lower_bound(
    source_layer: LayerModel,
    target_layer: LayerModel,
    spec: RadialCapMap,
) -> RadialCapGlobalLowerBound:
    """Certify the conditional geometric lower factor, without sampling.

    On the spherical cap theta<=cap_theta<pi, F is invertible onto a
    hyperbolic geodesic disk. F* g_H >= lambda^2 g_S pointwise. The
    hyperbolic minimizing geodesic stays in that disk, so its inverse
    path has sphere length <= d_H/lambda. The ambient spherical distance
    is <= this inverse-path length. Source-cap convexity is unnecessary.

    The factor one is sharp: points on the same radial meridian reach
    equality. The theorem refers to exact geometry, not to finite-precision
    point comparisons or to arbitrary user-supplied target samples.
    """
    if (source_layer.id != spec.source_layer_id
            or target_layer.id != spec.target_layer_id):
        raise ValueError("Correspondence layer IDs do not match the provided layers")
    if (
        source_layer.layer_type != LayerType.RIEMANNIAN
        or target_layer.layer_type != LayerType.RIEMANNIAN
        or source_layer.config.get("manifold_type") != "sphere"
        or target_layer.config.get("manifold_type") != "hyperbolic"
    ):
        raise ValueError("Global lower bound requires sphere and hyperbolic layers")
    _check_length_units(source_layer, target_layer, spec)
    _, radius = _metric(source_layer)
    _, scale = _metric(target_layer)
    target_radius = spec.radial_scale * radius * spec.cap_theta
    if not isfinite(target_radius) or target_radius <= 0:
        raise ValueError("Mapped cap radius must be positive and finite")
    if not isfinite(scale * target_radius):
        raise ValueError("Mapped hyperbolic cap radius exceeds numeric range")
    return RadialCapGlobalLowerBound(
        source_layer_id=source_layer.id,
        target_layer_id=target_layer.id,
        cap_theta=spec.cap_theta,
        lower_stretch=1.0,
        source_length_unit=spec.source_length_unit,
        target_length_unit=spec.target_length_unit,
    )


@dataclass(frozen=True)
class RadialCapGlobalEnvelope:
    """Conservative two-sided distance envelope for every declared cap <pi.

    For distinct source points in this cap:
        1 <= d_H(F(p), F(q)) / (lambda*d_S(p,q)) <= upper_stretch.

    The lower factor one is sharp. On caps above a hemisphere the *upper*
    factor is conservative: maximal local angular stretch multiplied by
    a justified detour factor. It is NOT claimed to be sharp.
    """

    source_layer_id: str
    target_layer_id: str
    cap_theta: float
    lower_stretch: float
    upper_stretch: float
    angular_stretch_max: float
    detour_factor: float
    source_length_unit: str
    target_length_unit: str


def radial_cap_global_envelope(
    source_layer: LayerModel,
    target_layer: LayerModel,
    spec: RadialCapMap,
) -> RadialCapGlobalEnvelope:
    """Provable cap-dependent upper and sharp lower factors for theta_max<pi.

    Let Theta=spec.cap_theta, c=a*lambda*R, and A(Theta) be the largest
    tangent stretch relative to lambda times the spherical metric:
        A(Theta) = sinh(c*Theta)/(c*sin(Theta)).

    For Theta<=pi/2 the spherical source cap is geodesically convex, and
    d_H <= lambda*A(Theta)*d_S follows by mapping a minimizing geodesic.

    For pi/2<Theta<pi, let alpha=pi-Theta be the radius of the excluded
    (convex) south-polar disk. Replace any segment of a minimizing
    spherical geodesic passing through this excluded disk by the shorter
    boundary-latitude arc. If its endpoints have azimuth gap delta<=pi:
        boundary length = R*sin(alpha)*delta
        spherical distance = 2*R*asin(sin(alpha)*sin(delta/2)).
    Concavity of u -> asin(sin(alpha)*sin(u)) on [0, pi/2] implies
        boundary length <= B(alpha) * spherical distance,
    with B(alpha) = pi*sin(alpha)/(2*alpha) <= pi/2.
    Thus the detour lies in the original cap and has length at most
    B(alpha)*d_S(p,q). Mapping it yields
        d_H(F(p),F(q)) <= lambda*A(Theta)*B(alpha)*d_S(p,q).
    The bound is explicit but generally not optimal.

    The lower factor comes from radial_cap_global_lower_bound and
    holds even for nonconvex caps by convexity of the target disk.
    """
    lower = radial_cap_global_lower_bound(source_layer, target_layer, spec)
    _, radius = _metric(source_layer)
    _, a = _metric(target_layer)
    theta = spec.cap_theta
    c = a * spec.radial_scale * radius
    x = c * theta

    if not isfinite(c) or not isfinite(x) or x > 700.0:
        raise ValueError("Cap stretch exceeds finite numeric range")
    if x < 1e-6:
        x2 = x * x
        sinh_over_x = 1.0 + x2 / 6.0 + x2 * x2 / 120.0
    else:
        sinh_over_x = sinh(x) / x

    angular = sinh_over_x * (theta / sin(theta))
    if theta <= pi / 2:
        detour = 1.0
    else:
        alpha = pi - theta
        # sin(alpha)/alpha is smooth at alpha=0; no cancellation for
        # the small but positive alpha values allowed by the spec.
        if alpha < 1e-5:
            a2 = alpha * alpha
            sin_over_alpha = 1.0 - a2 / 6.0 + (a2 * a2) / 120.0
        else:
            sin_over_alpha = sin(alpha) / alpha
        detour = (pi / 2.0) * sin_over_alpha

    upper = angular * detour
    if (not isfinite(upper) or not isfinite(angular) or
            detour < 1.0 - 1e-12 or upper < 1.0):
        raise ValueError("Global cap envelope exceeds finite numeric range")
    return RadialCapGlobalEnvelope(
        source_layer_id=spec.source_layer_id,
        target_layer_id=spec.target_layer_id,
        cap_theta=theta,
        lower_stretch=lower.lower_stretch,
        upper_stretch=upper,
        angular_stretch_max=angular,
        detour_factor=detour,
        source_length_unit=spec.source_length_unit,
        target_length_unit=spec.target_length_unit,
    )


@dataclass(frozen=True)
class ConvexCapDistortionBound:
    """Sharp tangent-range bound and valid global *pairwise* distance bounds.

    This applies ONLY to a spherical cap with angular radius strictly less
    than pi/2 (geodesically convex) mapped to a hyperbolic geodesic disk.
    For distinct source points:
        1 <= d_H(F(p),F(q)) / (lambda*d_S(p,q)) <= angular_stretch_max
    """

    source_layer_id: str
    target_layer_id: str
    cap_theta: float
    radial_stretch: float
    angular_stretch_max: float
    source_length_unit: str
    target_length_unit: str


def convex_cap_distortion_bound(
    source_layer: LayerModel,
    target_layer: LayerModel,
    spec: RadialCapMap,
) -> ConvexCapDistortionBound:
    """Return a geometrically justified finite upper bound, not a fitted score.

    Sphere metric: g_S = R^2(dtheta^2 + sin^2(theta) dphi^2).
    Hyperbolic metric: g_H = drho^2 + (sinh(a*rho)/a)^2 dpsi^2.
    F(theta,phi)=(lambda*R*theta,phi+omega).

    Relative to lambda^2*g_S, radial tangent stretch is 1 and angular
    stretch A(theta)=sinh(a*lambda*R*theta)/(a*lambda*R*sin(theta)).
    A(theta) increases on (0,pi). The source cap is geodesically convex
    only in the declared regime theta_max < pi/2; its image is a convex
    hyperbolic disk. Curve-length comparison then gives the global bound.
    For larger caps, no such global bound is returned.
    """
    if (
        source_layer.id != spec.source_layer_id
        or target_layer.id != spec.target_layer_id
    ):
        raise ValueError("Correspondence layer IDs do not match the provided layers")
    if (
        source_layer.layer_type != LayerType.RIEMANNIAN
        or target_layer.layer_type != LayerType.RIEMANNIAN
        or source_layer.config.get("manifold_type") != "sphere"
        or target_layer.config.get("manifold_type") != "hyperbolic"
    ):
        raise ValueError("Convex cap bound requires a sphere source and hyperbolic target")

    _check_length_units(source_layer, target_layer, spec)
    _, radius = _metric(source_layer)
    _, a = _metric(target_layer)
    theta = spec.cap_theta
    if theta >= pi / 2:
        raise ValueError(
            "Global bound requires a strictly convex spherical cap: "
            "cap_theta must be less than pi/2"
        )
    c = a * spec.radial_scale * radius
    x = c * theta
    if not isfinite(c) or not isfinite(x) or x > 700.0:
        raise ValueError("Cap stretch exceeds finite numeric range")
    if x < 1e-6:
        x2 = x * x
        sinh_over_x = 1.0 + x2 / 6.0 + x2 * x2 / 120.0
    else:
        sinh_over_x = sinh(x) / x
    upper = sinh_over_x * (theta / sin(theta))
    if not isfinite(upper) or upper < 1.0:
        raise ValueError("Cap distortion bound is outside finite valid range")
    return ConvexCapDistortionBound(
        source_layer_id=source_layer.id,
        target_layer_id=target_layer.id,
        cap_theta=theta,
        radial_stretch=1.0,
        angular_stretch_max=upper,
        source_length_unit=spec.source_length_unit,
        target_length_unit=spec.target_length_unit,
    )

def compare_radial_cap(
    source_layer: LayerModel,
    target_layer: LayerModel,
    spec: RadialCapMap,
    pairs: Sequence[Tuple[str, str]],
    max_target_error: float = 1e-9,
) -> CorrespondenceReport:
    """Validate explicit pairs and evaluate map distortion with typed remainder.

    No pair is inferred from display coordinates. Each declared target must be
    within max_target_error (target geodesic units) of the predicted image.
    For every pair of matched nodes, the triangle inequality gives the bound

      |(d_target_observed - scale*d_source) -
        (d_target_predicted - scale*d_source)| <= e_i + e_j.

    The report keeps predicted and observed remainders separate so sampling
    error cannot be silently promoted to metric distortion.
    """
    if source_layer.id != spec.source_layer_id or target_layer.id != spec.target_layer_id:
        raise ValueError("Correspondence layer IDs do not match the provided layers")
    if (
        source_layer.layer_type != LayerType.RIEMANNIAN
        or target_layer.layer_type != LayerType.RIEMANNIAN
        or source_layer.config.get("manifold_type") != "sphere"
        or target_layer.config.get("manifold_type") != "hyperbolic"
    ):
        raise ValueError("Only explicit sphere-cap to hyperbolic correspondence is implemented")
    _check_length_units(source_layer, target_layer, spec)
    src_kind, src_radius = _metric(source_layer)
    tgt_kind, tgt_scale = _metric(target_layer)
    if not isfinite(max_target_error) or max_target_error < 0:
        raise ValueError("max_target_error must be finite and non-negative")
    if not pairs:
        raise ValueError("Provide at least one explicit source/target node pair")

    src = {node.id: node for node in source_layer.nodes}
    tgt = {node.id: node for node in target_layer.nodes}
    if len(src) != len(source_layer.nodes) or len(tgt) != len(target_layer.nodes):
        raise ValueError("Node IDs must be unique within each layer")
    if len({a for a, _ in pairs}) != len(pairs) or len({b for _, b in pairs}) != len(pairs):
        raise ValueError("Explicit correspondence pairs must be one-to-one")

    matches: List[NodeMatch] = []
    source_params = []
    mapped_params = []
    target_params = []
    for source_id, target_id in pairs:
        if source_id not in src or target_id not in tgt:
            raise ValueError("Correspondence pairs must refer to registered layer nodes")
        a = _parameters(src[source_id], source_layer, src_kind)
        b = _parameters(tgt[target_id], target_layer, tgt_kind)
        mapped = spec.map_parameters(source_layer, float(a[0]), float(a[1]))
        predicted = np.asarray(mapped)
        e = _distance("hyperbolic", tgt_scale, predicted, b)
        if not isfinite(e):
            raise ValueError("Target mismatch is not finite")
        if e > max_target_error:
            raise ValueError(
                f"Declared target {target_id} is too far from mapped source {source_id}: "
                f"mismatch {e} > {max_target_error} {spec.target_length_unit}"
            )
        matches.append(NodeMatch(source_id, target_id, e))
        source_params.append(a)
        mapped_params.append(predicted)
        target_params.append(b)

    remainders: List[PairRemainder] = []
    for i in range(len(matches)):
        for j in range(i + 1, len(matches)):
            source_distance = _distance("sphere", src_radius, source_params[i], source_params[j])
            ideal_target = _distance(
                "hyperbolic", tgt_scale, mapped_params[i], mapped_params[j]
            )
            observed_target = _distance(
                "hyperbolic", tgt_scale, target_params[i], target_params[j]
            )
            expected = spec.radial_scale * source_distance
            ideal_remainder = ideal_target - expected
            observed_remainder = observed_target - expected
            bound = matches[i].mismatch + matches[j].mismatch
            # A numerical assertion on the triangle inequality, not a
            # mechanism that clips or modifies the reported residual.
            if abs(observed_remainder - ideal_remainder) > bound + 1e-9:
                raise ArithmeticError("Metric remainder violates the mismatch bound")
            remainders.append(PairRemainder(
                matches[i].source_id,
                matches[j].source_id,
                source_distance,
                ideal_target,
                observed_target,
                ideal_remainder,
                observed_remainder,
                bound,
            ))

    return CorrespondenceReport(
        spec=spec,
        matches=tuple(matches),
        pair_remainders=tuple(remainders),
        max_target_mismatch=max(match.mismatch for match in matches),
        max_abs_ideal_remainder=max(
            (abs(p.ideal_signed_remainder) for p in remainders), default=0.0
        ),
        max_abs_observed_remainder=max(
            (abs(p.observed_signed_remainder) for p in remainders), default=0.0
        ),
    )
