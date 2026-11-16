# Correspondence, with the remainder left visible

A sphere and a hyperbolic plane are not interchangeable because they can both be drawn in three dimensions. This bridge is an **explicit, restricted map** chosen by the caller, with assumptions and metric mismatch retained as data.

## Finite experiment

Take a sphere $S^2_R$, $R>0$, and a hyperbolic plane $H^2_{-a^2}$, with curvature $-a^2<0$. On a north-polar cap $0\leq\theta\leq\theta_{\max}<\pi$, declare:

$$
F(\theta,\phi) = (\rho=\lambda R\theta,\quad\psi=\phi+\omega\pmod {2\pi}).
$$

$\lambda>0$ is an explicitly declared target-length/source-length factor, **not an inferred natural isometry**. $\omega$ fixes a chosen angular alignment. At the north pole, all polar angles represent the same point and map to the hyperbolic origin.

For source samples $p_i,p_j$, the **ideal signed remainder** in target length units is

$$
\Delta_{ij}=d_H(F(p_i),F(p_j))-\lambda d_S(p_i,p_j).
$$

For explicitly paired target samples $q_i,q_j$, keep the **observed remainder** distinct:

$$
\widehat{\Delta}_{ij}=d_H(q_i,q_j)-\lambda d_S(p_i,p_j).
$$

With pointwise mismatches $e_i=d_H(F(p_i),q_i)$, the triangle inequality supplies a **non-fitted bound**:

$$
|\widehat{\Delta}_{ij}-\Delta_{ij}|\leq e_i+e_j.
$$

## The local obstruction, not hidden behind a graph

The sphere and hyperbolic plane have different polar metrics:

$$
g_S = R^2(d\theta^2+\sin^2\theta\,d\phi^2),\qquad
g_H = d\rho^2+\frac{\sinh^2(a\rho)}{a^2}\,d\psi^2.
$$

Pulling back the hyperbolic metric through the **declared** radial map gives

$$
F^*g_H=(\lambda R)^2d\theta^2+
\frac{\sinh^2(a\lambda R\theta)}{a^2}\,d\phi^2.
$$

Relative to the scaled source metric $\lambda^2g_S$, radial tangent lengths
have factor **1**, whereas angular tangent lengths have the factor

$$
L_{\mathrm{angular}}(\theta)=
\frac{\sinh(a\lambda R\theta)}
     {a\lambda R\sin\theta}>1
\quad\text{for }0<\theta<\theta_{\max}<\pi.
$$

This follows directly from $\sinh x>x$ and $\sin\theta<\theta$ for the
stated positive arguments. The ratio tends to **1** at the north pole.
The map is smooth on every cap with $\theta_{\max}<\pi$, but cannot
erase the metric's angular distortion. The local expansion alone does not
give a global lower distance bound in an arbitrary target space: here the
**convex hyperbolic image disk** supplies the missing global argument.
Source-cap convexity is needed for our existing *upper-bound* proof, not
for the *lower* bound.

All raw distances and signed remainders remain visible. Radial pairs along one meridian have $\Delta=0$ under this particular map; nondegenerate angular pairs usually have $\Delta>0$. A smooth map is not an isometry between these different curvature geometries.

## A global lower bound even beyond a hemisphere

For **every** declared cap with $0<\theta_{\max}<\pi$ (including nonconvex
spherical caps), the chosen radial map satisfies:

$
d_H(F(p),F(q))\geq \lambda\,d_S(p,q)
\qquad\text{for all }p,q\text{ in the cap}.
$

**Proof.** Let $D_H$ be the hyperbolic geodesic disk of radius
$\lambda R\theta_{\max}$. The map $F$ is a smooth bijection with smooth
inverse between the interior of the source cap and the interior of $D_H$
(including the north pole, expressed using normal coordinates). The boundary
statement follows by continuity; the south pole is excluded.

The pointwise pullback inequality

$
F^*g_H\succeq\lambda^2g_S
$

gives $L_H(F\circ\gamma)\geq\lambda L_S(\gamma)$ for any piecewise
smooth source path. Given $F(p),F(q)$, choose a minimizing hyperbolic
geodesic $\eta$. **Hyperbolic disks are geodesically convex**, so $\eta$
stays in $D_H$ and $F^{-1}\circ\eta$ remains in the spherical cap.
Therefore

$
d_H(F(p),F(q))
=L_H(\eta)
\geq\lambda L_S(F^{-1}\circ\eta)
\geq\lambda\,d_S(p,q).
$

Notice that the last $d_S$ is the **ambient spherical geodesic distance**,
not the constrained intrinsic path metric inside the cap. No assumption
that the shortest spherical path stays inside the source cap is needed.
Equality occurs for points on the same meridian connected by a radial
geodesic, so the factor 1 relative to $\lambda d_S$ is sharp.

The public `radial_cap_global_lower_bound` returns this factor with its
source/target unit declarations for the *exact chosen map*. For observed
target samples $q_i$ with geodesic discrepancies
$e_i=d_H(F(p_i),q_i)$, one obtains the explicitly weakened bound

$
d_H(q_i,q_j)\geq\lambda d_S(p_i,p_j)-(e_i+e_j).
$

This is an inequality, **not** a claim that observed target samples lie
exactly on the map, nor a lower-bound estimate fitted to test data.

### Why an upper bound cannot be cap-size independent

Choose two cap-boundary points at equal polar angle
$\theta=\pi-\varepsilon$, opposite in azimuth ($0$ and $\pi$), with
$0<\varepsilon<\pi/2$. Their ambient spherical shortest arc goes over
the south pole, but their hyperbolic images are on opposite rays:

$
d_S(p,q)=2R\varepsilon,\qquad
d_H(F(p),F(q))=2\lambda R(\pi-\varepsilon),
$

and hence

$
\frac{d_H(F(p),F(q))}{\lambda d_S(p,q)}
=\frac{\pi-\varepsilon}{\varepsilon}\longrightarrow\infty
\quad (\varepsilon\downarrow 0).
$

Thus no **uniform finite upper constant independent of cap size** holds
as the cap approaches the excluded south pole. This does **not** rule
out the constructive cap-dependent bound derived below.

## A conservative upper bound for every fixed cap below the south pole

A cap larger than a hemisphere is not geodesically convex: an ambient
shortest spherical arc can pass through the *excluded* south-polar disk.
We can nevertheless construct a controlled path **inside** the cap.

Let $\Theta=\theta_{\max}$ and $c=a\lambda R$.
The local differential stretching of the declared radial map is at most

$
A(\Theta)=\frac{\sinh(c\Theta)}{c\sin\Theta}.
$

For $\Theta\leq\pi/2$ the cap is geodesically convex, and a shortest
source geodesic stays inside it. We may take a detour factor $B(\Theta)=1$.

For $\pi/2<\Theta<\pi$, the excluded disk about the south pole has
angular radius $\alpha=\pi-\Theta<\pi/2$ and is itself geodesically
convex. A minimizing spherical arc between cap points intersects this
excluded disk in at most one contiguous segment. Replace that segment
by the shorter *latitude-boundary* arc between its entry and exit points.
The replacement stays inside the original cap.

For boundary points separated in longitude by $\delta\in[0,\pi]$,

$
L_{\partial}=R\sin\alpha\,\delta,\qquad
d_S(x,y)=2R\arcsin\!\left(\sin\alpha\sin\frac{\delta}{2}\right).
$

The function $u\mapsto\arcsin(\sin\alpha\sin u)$ is concave on
$[0,\pi/2]$, vanishes at zero, and equals $\alpha$ at $\pi/2$.
Its graph lies above the secant line, giving

$
d_S(x,y)\geq\frac{2R\alpha}{\pi}\,\delta,
\qquad
L_{\partial}\leq
\underbrace{\frac{\pi\sin\alpha}{2\alpha}}_{B(\Theta)}
\,d_S(x,y).
$

The discarded spherical segment is at least $d_S(x,y)$ long.
Thus replacing it changes the total path length by **at most**
the factor $B(\Theta)$. Mapping this explicitly constructed path into
the hyperbolic disk and applying the local stretch ceiling gives

$
1\leq
\frac{d_H(F(p),F(q))}{\lambda\,d_S(p,q)}
\leq C(\Theta),\qquad
C(\Theta)=A(\Theta)B(\Theta),
$

where

$
B(\Theta)=
\begin{cases}
1,&0<\Theta\leq\pi/2,\\[2pt]
\dfrac{\pi\sin(\pi-\Theta)}{2(\pi-\Theta)},
&\pi/2<\Theta<\pi.
\end{cases}
$

For nonconvex caps, $1<B(\Theta)<\pi/2$. The upper bound is
**conservative**: $B$ is a geometric detour penalty, not the exact
optimal Lipschitz constant of $F$. The lower bound remains sharp,
using the separate target-convexity proof above. At $\Theta=\pi/2$,
the two formulas join continuously. For every fixed $\Theta<\pi$,
$C(\Theta)$ is finite under the usual finite-parameter conditions.

For registered *observed* target samples $q_i$ with mismatches $e_i$,
the same argument yields a bound that preserves its remainder:

$
\lambda d_S(p_i,p_j)-(e_i+e_j)
\leq d_H(q_i,q_j)
\leq\lambda C(\Theta)d_S(p_i,p_j)+(e_i+e_j).
$

The API \`radial_cap_global_envelope\` returns the sharp lower factor,
the maximal local angular stretch $A$, the detour factor $B$, and the
conservative upper factor $C$ separately. The legacy
\`convex_cap_distortion_bound\` remains restricted to $\Theta<\pi/2$
and retains its sharper convex-domain interpretation. No bound here
asserts an isometry or canonical manifold transport.

### Falsification and numerical boundaries

The shortest spherical arc and detour argument are analytic.
The deterministic numerical sweep uses finite samples and a separately
implemented Poincaré-disk distance expression to check that sampled
ratios stay in the declared envelope. A confirmed violation from a
valid finite case would kill the implementation claim; passing
samples does not replace the geometric proof.

There can be no upper factor **independent of cap size**: the opposite
south-boundary pair described above has ratio
$(\pi-\varepsilon)/\varepsilon\to\infty$. The cap-dependent factor
$C(\Theta)$ diverges as $\Theta\to\pi$ in agreement with this
counterexample. Near the south pole, floating-point angle precision
and finite numeric range must be tracked separately from this exact
geometric statement.

## A two-sided global bound on convex caps

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

$$
1\ \leq\
\frac{d_H(F(p),F(q))}{\lambda\,d_S(p,q)}
\ \leq\
A(\theta_{\max})
=
\frac{\sinh(c\theta_{\max})}{c\sin\theta_{\max}}.
$$

The bounds are sharp as tangent-direction limits: radial direction gives 1,
while infinitesimal angular separation at the cap boundary approaches
$A(\theta_{\max})$. The formula is an *analytic envelope*, not an empirical
maximum found by sampling. For actual declared target samples, the mismatch
allowances $e_i,e_j$ give the unnormalized target-distance inequalities:

$$
\lambda d_S(p_i,p_j)-(e_i+e_j)
\ \leq\ d_H(q_i,q_j)\
\leq\ \lambda A(\theta_{\max})d_S(p_i,p_j)+(e_i+e_j).
$$

The lower bound can be negative; that is a mathematically valid weak lower
bound, not a negative measured distance. **No measurements are clipped.**

The public function `convex_cap_distortion_bound` returns this
**two-sided envelope** only on strictly convex caps. For larger caps
$\theta_{\max}\geq\pi/2$, the global lower bound remains valid through
`radial_cap_global_lower_bound`, but this convexity-based **upper-bound**
argument is not promoted beyond its proven domain. The finite-pair
remainder calculation remains available in both regimes.

### Unit provenance is checked

The `RadialCapMap` unit names must agree with each layer's
`config["length_unit"]`. Names are compared literally; this does not
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
