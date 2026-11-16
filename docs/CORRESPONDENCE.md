# Correspondence, with the remainder left visible

A sphere and a hyperbolic plane are not interchangeable because they can both be drawn in three dimensions. This bridge is an **explicit, restricted map** chosen by the caller, with assumptions and metric mismatch retained as data.

## Finite experiment

Take a sphere $S^2_R$, $R>0$, and a hyperbolic plane $H^2_{-a^2}$, with curvature $-a^2<0$. On a north-polar cap $0\leq\theta\leq\theta_{\max}<\pi$, declare:

$
F(\theta,\phi) = (\rho=\lambda R\theta,\quad\psi=\phi+\omega\pmod {2\pi}).
$

$\lambda>0$ is an explicitly declared target-length/source-length factor, **not an inferred natural isometry**. $\omega$ fixes a chosen angular alignment. At the north pole, all polar angles represent the same point and map to the hyperbolic origin.

For source samples $p_i,p_j$, the **ideal signed remainder** in target length units is

$
\Delta_{ij}=d_H(F(p_i),F(p_j))-\lambda d_S(p_i,p_j).
$

For explicitly paired target samples $q_i,q_j$, keep the **observed remainder** distinct:

$
\widehat{\Delta}_{ij}=d_H(q_i,q_j)-\lambda d_S(p_i,p_j).
$

With pointwise mismatches $e_i=d_H(F(p_i),q_i)$, the triangle inequality supplies a **non-fitted bound**:

$
|\widehat{\Delta}_{ij}-\Delta_{ij}|\leq e_i+e_j.
$

## The local obstruction, not hidden behind a graph

The sphere and hyperbolic plane have different polar metrics:

$
g_S = R^2(d\theta^2+\sin^2\theta\,d\phi^2),\qquad
g_H = d\rho^2+\frac{\sinh^2(a\rho)}{a^2}\,d\psi^2.
$

Pulling back the hyperbolic metric through the **declared** radial map gives

$
F^*g_H=(\lambda R)^2d\theta^2+
\frac{\sinh^2(a\lambda R\theta)}{a^2}\,d\phi^2.
$

Relative to the scaled source metric $\lambda^2g_S$, radial tangent lengths
have factor **1**, whereas angular tangent lengths have the factor

$
L_{\mathrm{angular}}(\theta)=
\frac{\sinh(a\lambda R\theta)}
     {a\lambda R\sin\theta}>1
\quad\text{for }0<\theta<\theta_{\max}<\pi.
$

This follows directly from $\sinh x>x$ and $\sin\theta<\theta$ for the
stated positive arguments. The ratio tends to **1** at the north pole.
The map is smooth on the cap but cannot erase the metric's angular
distortion. This first observation is **local**. A separate global
bound requires a geodesically convex source domain.

All raw distances and signed remainders remain visible. Radial pairs along one meridian have $\Delta=0$ under this particular map; nondegenerate angular pairs usually have $\Delta>0$. A smooth map is not an isometry between these different curvature geometries.

## A global bound — but only on convex caps

For the explicit map above, take $0<\theta_{\max}<\pi/2$, so the **spherical
source cap is geodesically convex**. Its image is a hyperbolic geodesic disk,
which is also convex. Put $c=a\lambda R$.

The pointwise angular stretch
$A(\theta)=\sinh(c\theta)/(c\sin\theta)$ increases with $\theta$, and the
radial stretch is exactly 1, relative to $\lambda^2g_S$. Therefore the
length of every curve within the source cap obeys the corresponding
differential inequalities. Applying the upper inequality to a
minimizing spherical geodesic (which remains inside the cap), and the
lower inequality to the inverse of a minimizing hyperbolic geodesic
(which remains in the target disk), proves for **distinct** cap points:

$
1\ \leq\
\frac{d_H(F(p),F(q))}{\lambda\,d_S(p,q)}
\ \leq\
A(\theta_{\max})
=
\frac{\sinh(c\theta_{\max})}{c\sin\theta_{\max}}.
$

The bounds are sharp as tangent-direction limits: radial direction gives 1,
while infinitesimal angular separation at the cap boundary approaches
$A(\theta_{\max})$. The formula is an *analytic envelope*, not an empirical
maximum found by sampling. For actual declared target samples, the mismatch
allowances $e_i,e_j$ give the unnormalized target-distance inequalities:

$
\lambda d_S(p_i,p_j)-(e_i+e_j)
\ \leq\ d_H(q_i,q_j)\
\leq\ \lambda A(\theta_{\max})d_S(p_i,p_j)+(e_i+e_j).
$

The lower bound can be negative; that is a mathematically valid weak lower
bound, not a negative measured distance. **No measurements are clipped.**

The public function \`convex_cap_distortion_bound\` returns the analytic
bounds only within this strictly convex-cap regime. For a larger cap
$\theta_{\max}\geq\pi/2$, Braintop retains the original finite-pair
remainder calculation but refuses to promote the local stretch formula
to a global theorem using this argument.

### Unit provenance is checked

The \`RadialCapMap\` unit names must agree with each layer's
\`config["length_unit"]\`. Names are compared literally; this does not
provide automated dimensional conversions. The numeric scale $\lambda$
is still the user's explicit target-length/source-length choice.
A missing or mismatched length-unit declaration fails before
comparison or edge creation.


## Minimal use

~~~python
import numpy as np
from braintop.core.correspondence import (
    RadialCapMap, compare_radial_cap, convex_cap_distortion_bound,
)
from braintop.utils.builders import TopologyBuilder

source = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 1.57]])
target = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 1.57]])

builder = TopologyBuilder("radial-cap", "Declared geometry bridge")
builder.add_riemannian_layer("sphere", 3, manifold_type="sphere",
                            radius=1.0, length_unit="declared-source-length", parameters=source)
builder.add_riemannian_layer("hyperbolic", 3, manifold_type="hyperbolic",
                            curvature=-1.0, length_unit="declared-target-length", parameters=target)

spec = RadialCapMap(
    source_layer_id="sphere",
    target_layer_id="hyperbolic",
    radial_scale=1.0,
    cap_theta=1.0,
    source_length_unit="declared-source-length",
    target_length_unit="declared-target-length",
    provenance="radial-cap-example-v1",
)
pairs = [
    ("sphere_node_0", "hyperbolic_node_0"),
    ("sphere_node_1", "hyperbolic_node_1"),
    ("sphere_node_2", "hyperbolic_node_2"),
]

report = compare_radial_cap(
    builder.topology.get_layer("sphere"),
    builder.topology.get_layer("hyperbolic"),
    spec, pairs, max_target_error=1e-8,
)
bound = convex_cap_distortion_bound(
    builder.topology.get_layer("sphere"),
    builder.topology.get_layer("hyperbolic"), spec,
)
print(bound.angular_stretch_max)
print(report.max_abs_ideal_remainder)
print(report.pair_remainders)

# Optional: materialize declared links only after the check passes.
builder.connect_declared_correspondence(spec, pairs, max_target_error=1e-8)
~~~

No pairs are inferred. All declared matches must satisfy the target error limit before *any* edges are created. Edges have weight 1.0 and **weight_semantics = unweighted_declared_correspondence**. Their existence is a declared relation, not a metric closeness score. The separate report carries the metric evidence.

## Controls and limits

| Evidence | Control | Interpretation |
|---|---|---|
| Analytic | Same-meridian radial pair | Ideal remainder zero |
| Negative | Equal-radius source points separated angularly | Nonzero ideal remainder |
| Independent coordinate expression | Poincaré-disk geodesic formula | Cross-check target distances |
| Metamorphic | Passive SE(3) display-frame change | Report unchanged |
| Matched error control | Perturb an explicitly paired target | Observed/ideal difference bounded by pointwise mismatches |
| Fail closed | Unsupported manifold, units, pairs, mismatch | No partial edge commit |

Passing tests under this contract does **not** establish torus geodesics, a canonical sphere-to-hyperbolic functor, parallel transport, sheaf connections, or geometry inferred from 3D appearance.

**Kill condition:** a confirmed finite counterexample violates the mismatch inequality or passive display-frame invariance beyond documented numerical tolerance. Withdraw the implementation claim rather than inventing an untested ontology.

See also [intrinsic geodesics](INTRINSIC_GEODESICS.md) and [rigid display frames](MOVING_FRAMES.md).
