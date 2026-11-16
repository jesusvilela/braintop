# Changelog

Notable changes to Braintop are recorded here.

## Unreleased

- Add a sharp global lower distance factor for the declared sphere-cap to hyperbolic radial map on every cap of angular radius below pi, including nonconvex caps.
- Preserve the convex-cap restriction on the existing upper envelope; add near-south-pole controls showing the absence of a cap-size-independent upper bound.


- Bind declared correspondence units to the source/target layer configurations; reject mismatches and enforce strict target-error budgets.
- Add an analytic convex-cap distance-distortion envelope for spherical caps with radius strictly below π/2.
- Generate a deterministic distortion-versus-cap-radius plot and CSV in showcase CI.
- Repair GitHub display-math delimiters and clarify global versus local validity.


- Introduce a declared sphere-cap to hyperbolic radial map with explicit units and provenance.
- Report ideal versus observed geodesic remainders and triangle-inequality mismatch bounds; materialize only audited explicit cross-layer pairs.


- Preserve manifold chart coordinates independently of 3D display positions.
- Implement within-layer sphere and hyperbolic geodesic metrics and intrinsic nearest-neighbour edges.
- Add known-answer and representation-invariance controls; reject unsupported torus geodesics and cross-manifold interpretations.


- Route named LLM providers through explicit adapters.
- Fail closed when a non-mock provider lacks an embedding or inference capability.
- Validate projection methods, manifold types, connection strategies and visualization inputs.
- Improve multi-layer Plotly color scales and NetworkX node metadata.
- Add deterministic geometry and semantic-topology showcase generators.
- Expand CI across supported Python versions and validate built distributions.
