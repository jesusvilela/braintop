# Intrinsic geodesics · a separate geometry from the picture

Braintop now supports **within-layer** intrinsic distance on a sampled sphere or
hyperbolic plane. This is a deliberately small mathematical primitive, not a
general-purpose manifold solver.

The 3D Plotly point is a **display coordinate**; the original
`intrinsic_parameters` and their `intrinsic_chart` are preserved in each newly
generated Riemannian node's metadata. A passive SE(3) change of *display*
coordinates never rewrites those parameters.

## Three distances that must not be conflated

| Object | Observable | Status |
|---|---|---|
| Ambient 3D distance | Euclidean length between world/display points | Existing `connect_layers(..., "nearest")`; cross-layer graphical wiring |
| Sphere intrinsic distance | Shortest great-circle arc on a radius-$r$ sphere | Implemented within a single sphere layer |
| Hyperbolic intrinsic distance | Riemannian geodesic length in constant curvature $K<0$ | Implemented within a single hyperbolic layer |

A torus has a known induced Riemannian metric, but **its geodesic distance
solver is not implemented** here. An attempt to use it raises
`NotImplementedError`; Braintop will not silently substitute a chord.

### Sphere

Given polar parameters $(\theta,\phi)$, define the unit 3-vector

$
u=(\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ \cos\theta).
$

For two points on the *same sphere of radius* $r$,

$
d_{S^2_r}(u,v)
 = r\,\operatorname{atan2}\bigl(\|u\times v\|,\ u\cdot v\bigr).
$

The great-circle distance and Euclidean chord length are related by

$
d_{\mathrm{chord}}=2r\sin\!\left(\frac{d_{S^2_r}}{2r}\right).
$

On this single fixed-radius sphere, the relation is monotone: **nearest
neighbours cannot change simply by switching between geodesic and chord
distance**, aside from floating-point or exact-tie effects. The distances
themselves differ.

### Hyperbolic geodesic polar chart

For curvature $K=-a^2<0$, polar coordinates $(\rho,\theta)$ measure
radial geodesic distance $\rho$ from the origin and angle $\theta$. The
implementation uses the positive form of the hyperbolic cosine law:

$
\sinh^2\!\left(\frac{a d}{2}\right)
=
\sinh^2\!\left(\frac{a(\rho_1-\rho_2)}{2}\right)
+
\sinh(a\rho_1)\sinh(a\rho_2)
\sin^2\!\left(\frac{\theta_1-\theta_2}{2}\right).
$

For an independent oracle, write the corresponding Poincaré disk points as

$
z_i=\tanh(a\rho_i/2)(\cos\theta_i,\sin\theta_i).
$

Then

$
d_H(z_1,z_2)=\frac{2}{a}
\operatorname{asinh}
\left(
\frac{\|z_1-z_2\|}
{\sqrt{(1-\|z_1\|^2)(1-\|z_2\|^2)}}
\right).
$

The production hyperbolic-distance implementation evaluates the geodesic-polar
formula using log-sinh arithmetic. This avoids an intermediate overflow from
products such as `sinh(500) * sinh(490)` when the actual distance is finite.
It still rejects non-finite scales and radii whose scaled products cannot be
represented.

The Poincaré-disk formula is an alternative coordinate reference at moderate
radii, but its floating-point disk coordinates saturate close to the boundary
for large geodesic radii. For those cases, the tests instead use analytical
radial and opposite-ray distances, plus a large-radius angular asymptotic.

**Angular precision matters:** at large hyperbolic radius even a rounding-scale
difference between the floating-point angles `0.3` and `0.3 + 2*pi` can
have a very large geometric effect. The implementation does not silently snap
nearby represented angles together. Exact periodic representatives such as
`0` and the same floating-point `2*pi` constant are checked separately.

The third display coordinate `z` used by Braintop's visualizer is *not* an
isometric hyperbolic embedding.

## First experiment

```python
import numpy as np
from braintop.core.intrinsic import intrinsic_distance
from braintop.utils.builders import TopologyBuilder

params = np.array([
    [4.0, 0.0],  # source
    [4.0, 0.4],  # nearby in the 3D view, far by hyperbolic geodesics
    [3.0, 0.0],  # further in the 3D view, near intrinsically
])

builder = TopologyBuilder("hyperbolic", "Intrinsic neighbours")
builder.add_riemannian_layer(
    "disk",
    num_nodes=3,
    manifold_type="hyperbolic",
    curvature=-1.0,
    max_radius=5.0,
    parameters=params,
)

layer = builder.topology.get_layer("disk")
print(intrinsic_distance(layer, "disk_node_0", "disk_node_1"))
print(intrinsic_distance(layer, "disk_node_0", "disk_node_2"))

# Creates undirected within-layer edges; never cross-manifold transports.
builder.connect_intrinsic_neighbors("disk", k=1)
topology = builder.build()
```

The `connect_intrinsic_neighbors` operation takes the union of each node's
`k` nearest neighbours, with a deterministic node-ID tie break. Edges store
their **intrinsic distance, named metric and units** in metadata and a
positive *affinity* `weight = 1 / (1 + distance)`. This transform is a choice
for visualization, **not an intrinsic normalization across manifolds or units**.

The `intrinsic_distance` function operates on registered nodes in **one**
layer. It rejects missing polar parameters, unrecognized charts, unsupported
manifold types, invalid curvature/radius, and incompatible layer identities.

Newly created Riemannian nodes store their parameters. Existing serialized
nodes without this geometric provenance cannot be used for intrinsic geodesics
without regeneration or a validated explicit migration of their original chart.
There is no silent inversion from mutable 3D display coordinates.

## Verification

`tests/test_intrinsic_geodesics.py` contains known-answer cases (antipodal
sphere points, hyperbolic radial lines and opposite angles), an independently
derived disk-coordinate reference, ordering controls, parameter rejection, and
passive moving-frame equivalence.

The central deliberately **negative** example is curvature $-1$ with
source $(\rho,\theta)=(4,0)$: the point $(4,0.4)$ is closer in Braintop's
3D display than $(3,0)$, while the intrinsic geodesic chooses $(3,0)$.
This falsifies the interpretation that the current 3D hyperbolic display is
itself an intrinsic-distance oracle.

**Evidence status:** analytical formulas under stated constant-curvature
models (P); tests and numerical comparisons (M, limited to tested inputs);
cross-manifold transport (H, not implemented). See also the
[moving-frame contract](MOVING_FRAMES.md).
