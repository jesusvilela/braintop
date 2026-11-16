"""An explicit, non-canonical correspondence between spherical caps and H^2.

This is a finite-sample comparison of two specified Riemannian metrics.
It is NOT parallel transport, a connection, a canonical equivalence, or an
isometry between spaces of different Gaussian curvature.

A declared radial map sends sphere polar coordinates (theta, phi) to
hyperbolic geodesic polar coordinates (lambda*r*theta, phi+offset).
Only explicitly paired, registered source and target nodes are compared.
"""

from dataclasses import dataclass
from math import isfinite, pi
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
        if e > max_target_error + 1e-12:
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
