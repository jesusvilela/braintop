"""Intrinsic distances on Braintop's sphere and hyperbolic disk layers.

The observable is a geodesic distance **within the same configured layer**,
calculated from preserved manifold parameters, not its 3D display positions.

No cross-manifold distance, torus geodesic solver, transport, or parallel
transport is implemented here.
"""

from typing import Dict, List, Tuple, Union

import math

import numpy as np

from braintop.core.topology import LayerModel, LayerType, NodeModel


_CHARTS = {
    "sphere": "sphere_polar_v1",
    "torus": "torus_angles_v1",
    "hyperbolic": "hyperbolic_geodesic_polar_v1",
}


def _metric(layer: LayerModel) -> Tuple[str, float]:
    if layer.layer_type != LayerType.RIEMANNIAN:
        raise ValueError("Intrinsic geometry requires a Riemannian layer")
    kind = layer.config.get("manifold_type")
    if kind == "torus":
        raise NotImplementedError(
            "Intrinsic torus geodesics are not implemented; "
            "ambient chord distances are not a substitute"
        )
    if kind == "sphere":
        radius = float(layer.config.get("radius", 1.0))
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Sphere radius must be positive and finite")
        return kind, radius
    if kind == "hyperbolic":
        configured = layer.config.get("curvature")
        curvature = -1.0 if configured is None else float(configured)
        if not np.isfinite(curvature) or curvature >= 0:
            raise ValueError("Hyperbolic curvature must be negative and finite")
        return kind, float(np.sqrt(-curvature))
    raise ValueError(
        f"Unsupported intrinsic manifold type {kind!r}: "
        "only sphere and hyperbolic have implemented distances"
    )


def _parameters(node: NodeModel, layer: LayerModel, kind: str) -> np.ndarray:
    if node.layer_id != layer.id:
        raise ValueError("Intrinsic distances require nodes in the same layer")

    expected_chart = _CHARTS[kind]
    if node.metadata.get("intrinsic_chart") != expected_chart:
        raise ValueError(
            f"Node {node.id} needs intrinsic_chart={expected_chart}; "
            "regenerate it from explicit manifold parameters"
        )
    if node.metadata.get("manifold_type") != kind:
        raise ValueError("Node's intrinsic manifold type disagrees with its layer")

    values = np.asarray(node.metadata.get("intrinsic_parameters"), dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("Intrinsic parameters must be two finite coordinates")
    if kind == "sphere" and not 0 <= values[0] <= np.pi:
        raise ValueError("Sphere polar theta must lie in [0, pi]")
    if kind == "hyperbolic" and values[0] < 0:
        raise ValueError("Hyperbolic geodesic radius must be non-negative")
    return values



def _log_sinh_positive(x: float) -> float:
    """Log(sinh(x)) for x>=0 without overflowing at large x."""
    if x == 0.0:
        return -math.inf
    if x < 0.0 or not math.isfinite(x):
        raise ValueError("log-sinh input must be finite and non-negative")
    if x < 1e-4:
        # sinh(x)/x = 1 + x^2/6 + x^4/120 + O(x^6).
        sq = x * x
        return math.log(x) + math.log1p(sq / 6.0 + sq * sq / 120.0)
    if x < 30.0:
        return math.log(math.sinh(x))
    return x - math.log(2.0) + math.log1p(-math.exp(-2.0 * x))

def _distance(kind: str, scale: float, a: np.ndarray, b: np.ndarray) -> float:
    """Numerically stable known-formula geodesic distance in layer units."""
    if kind == "sphere":
        theta1, phi1 = a
        theta2, phi2 = b
        u = np.array([
            np.sin(theta1) * np.cos(phi1),
            np.sin(theta1) * np.sin(phi1),
            np.cos(theta1),
        ])
        v = np.array([
            np.sin(theta2) * np.cos(phi2),
            np.sin(theta2) * np.sin(phi2),
            np.cos(theta2),
        ])
        # atan2 remains well-conditioned near antipodal positions.
        angle = np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v))
        return float(scale * angle)

    # Curvature K=-scale**2. Geodesic polar hyperbolic law of cosines:
    # sinh^2(scale*d/2) = sinh^2(scale*(rho1-rho2)/2)
    #   + sinh(scale*rho1)*sinh(scale*rho2)*sin^2((theta1-theta2)/2).
    #
    # Directly multiplying sinh(scale*rho1)*sinh(scale*rho2) can overflow
    # even for small finite distances (notably when theta1 == theta2).
    # Evaluate both positive summands in logarithmic form. This preserves
    # moderate-distance behavior while extending the finite numeric range.
    rho1, angle1 = float(a[0]), float(a[1])
    rho2, angle2 = float(b[0]), float(b[1])
    if rho1 < 0 or rho2 < 0 or not all(
        math.isfinite(value) for value in (scale, rho1, rho2, angle1, angle2)
    ) or scale <= 0:
        raise ValueError("Hyperbolic scale, radii and angles must be finite and valid")

    if rho1 == 0.0:
        return rho2
    if rho2 == 0.0:
        return rho1

    # Reduce each angle before subtraction, avoiding overflow of their
    # difference for large but finite coordinate representations.
    delta = math.remainder(
        math.remainder(angle1, 2.0 * math.pi)
        - math.remainder(angle2, 2.0 * math.pi),
        2.0 * math.pi,
    )
    sine_half = abs(math.sin(delta / 2.0))
    if sine_half == 0.0:
        return abs(rho1 - rho2)

    x = scale * rho1
    y = scale * rho2
    half_delta = 0.5 * scale * abs(rho1 - rho2)
    if not all(math.isfinite(v) for v in (x, y, half_delta)):
        raise ValueError("Hyperbolic parameters exceed the finite numeric domain")

    log_radial = _log_sinh_positive(half_delta)
    log_angular = (
        0.5 * _log_sinh_positive(x)
        + 0.5 * _log_sinh_positive(y)
        + math.log(sine_half)
    )
    high = max(log_radial, log_angular)
    low = min(log_radial, log_angular)
    log_norm = high + 0.5 * math.log1p(math.exp(2.0 * (low - high)))

    # asinh(exp(z)) = z + log(2) + O(exp(-2z)) for large z.
    # This avoids constructing exp(z) when its value would overflow.
    if log_norm > 36.0:
        half_distance = log_norm + math.log(2.0)
    else:
        half_distance = math.asinh(math.exp(log_norm))
    distance = (2.0 / scale) * half_distance
    if not math.isfinite(distance):
        raise ValueError("Hyperbolic distance exceeds the finite numeric domain")
    return distance


def intrinsic_distance(
    layer: LayerModel,
    source: Union[str, NodeModel],
    target: Union[str, NodeModel],
) -> float:
    """Geodesic distance between two registered nodes of one supported layer.

    Resolve IDs against the layer so stale detached node objects cannot silently
    introduce geometric parameters inconsistent with the layer's current state.
    """
    kind, scale = _metric(layer)
    members = {node.id: node for node in layer.nodes}
    if len(members) != len(layer.nodes):
        raise ValueError("Duplicate node IDs prevent intrinsic comparisons")
    if isinstance(source, NodeModel) and source.layer_id != layer.id:
        raise ValueError("Source node belongs to a different layer")
    if isinstance(target, NodeModel) and target.layer_id != layer.id:
        raise ValueError("Target node belongs to a different layer")
    a_id = source.id if isinstance(source, NodeModel) else source
    b_id = target.id if isinstance(target, NodeModel) else target
    if a_id not in members or b_id not in members:
        raise ValueError("Intrinsic distances require registered nodes in one layer")
    a = _parameters(members[a_id], layer, kind)
    b = _parameters(members[b_id], layer, kind)
    return _distance(kind, scale, a, b)


def nearest_intrinsic_pairs(
    layer: LayerModel, k: int = 1,
) -> List[Tuple[str, str, float]]:
    """Deterministic undirected union of each node's k intrinsic neighbours.

    Each distance is evaluated once per unordered pair, with ID tie-breaking.
    This is O(n^2) and intentionally makes no inter-layer correspondence.
    """
    kind, scale = _metric(layer)
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")
    nodes = sorted(layer.nodes, key=lambda node: node.id)
    if len({node.id for node in nodes}) != len(nodes):
        raise ValueError("Duplicate node IDs prevent intrinsic comparisons")
    params: Dict[str, np.ndarray] = {
        node.id: _parameters(node, layer, kind) for node in nodes
    }
    if k == 0 or len(nodes) < 2:
        return []

    candidates: Dict[str, List[Tuple[float, str]]] = {
        node.id: [] for node in nodes
    }
    for i, source in enumerate(nodes):
        for target in nodes[i + 1:]:
            d = _distance(kind, scale, params[source.id], params[target.id])
            candidates[source.id].append((d, target.id))
            candidates[target.id].append((d, source.id))

    picked: Dict[Tuple[str, str], float] = {}
    for node in nodes:
        for distance, target in sorted(candidates[node.id])[:k]:
            key = tuple(sorted((node.id, target)))
            picked[key] = distance

    return [(a, b, picked[a, b]) for a, b in sorted(picked)]
