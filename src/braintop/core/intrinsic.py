"""Intrinsic distances on Braintop's sphere and hyperbolic disk layers.

The observable is a geodesic distance **within the same configured layer**,
calculated from preserved manifold parameters, not its 3D display positions.

No cross-manifold distance, torus geodesic solver, transport, or parallel
transport is implemented here.
"""

from typing import Dict, List, Tuple, Union

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
    # sinh^2(scale*d/2) =
    #   sinh^2(scale*(rho1-rho2)/2)
    #   + sinh(scale*rho1)*sinh(scale*rho2)*sin^2((theta1-theta2)/2)
    r1, theta1 = a
    r2, theta2 = b
    with np.errstate(over="raise", invalid="raise"):
        try:
            radial = np.sinh(scale * (r1 - r2) / 2.0)
            angular = (
                np.sqrt(np.sinh(scale * r1) * np.sinh(scale * r2))
                * np.sin((theta1 - theta2) / 2.0)
            )
            distance = 2.0 * np.arcsinh(np.hypot(radial, angular)) / scale
        except FloatingPointError as exc:
            raise ValueError("Hyperbolic parameters overflow the numeric domain") from exc
    if not np.isfinite(distance):
        raise ValueError("Hyperbolic distance is non-finite")
    return float(distance)


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
