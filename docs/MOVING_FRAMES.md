# Moving frames, graph semantics, and what Braintop does not claim

Braintop operates on **three-dimensional display-coordinate point clouds**. A point
from a curved manifold is *displayed* as a 3-vector, but that vector does not
necessarily preserve intrinsic distances or define a canonical transport to
another manifold.

## Display-frame contract

Each `LayerModel` has a validated `frame: RigidFrame3D`.

```text
local node position p ---- R,t ----> world/display position Rp+t
```

`R` must be a finite proper orthogonal 3×3 rotation with determinant +1;
`t` is a finite translation. The default frame is the identity, so existing
topologies retain their former coordinates.

```python
from braintop.utils.builders import TopologyBuilder

builder = TopologyBuilder("frames", "An ambient-frame experiment")
builder.add_conceptual_layer("concepts", num_nodes=4)
builder.add_riemannian_layer("sphere", num_nodes=12, manifold_type="sphere")

# Position the display of the sphere relative to the concept layer.
builder.set_layer_frame("sphere", translation=(3.0, 0.0, 0.0))
builder.connect_layers("concepts", "sphere", "nearest", num_connections=1)
topology = builder.build()
```

The `nearest` builder strategy uses **ambient Euclidean distance between the
world/display coordinates**. It is not geodesic distance and does not claim
intrinsic correspondence between manifolds.

### Passive coordinate changes are different from moving a layer

For a local coordinate relabelling $y=Qx+s$, the frame must become

$
R' = RQ^{\mathsf T}, \qquad t' = t - RQ^{\mathsf T}s.
$

Then `R' @ y + t' == R @ x + t` up to floating point error. You can apply that
relabelling atomically:

```python
quarter_turn = (
    (0., -1., 0.),
    (1.,  0., 0.),
    (0.,  0., 1.),
)
topology.reexpress_layer("sphere", quarter_turn, (2., -1., 0.))
```

The layer's local node coordinates **and** its local-to-world frame are updated.
The existing edges remain valid, because the world coordinates did not move.

By contrast, `builder.set_layer_frame(...)` specifies physical placement in the
ambient display. Move only one layer, and nearest neighbours may change.
**Connections are construction-time snapshots**. Changing a frame *after* calling
`connect_layers` does not dynamically recompute existing edges. If that is
required, construct a new topology (or explicitly remove/rebuild those edges).

### Plotting and graph export

The following use world coordinates: 3D node rendering, edge endpoints,
coordinate sections and cross-layer `nearest` connections.

NetworkX export retains both `position` (layer-local) and `world_position`,
along with `layer_id`, `weight` and a relation `role` of `intra` or
`inter`.

Edges created from text similarities are now stored in
`Topology.intra_layer_edges`, while explicit cross-layer links remain in
`Topology.inter_layer_edges`. `Topology.get_all_edges()` gives both.

The graph exporter currently returns an ordinary undirected NetworkX
`Graph`; parallel edges with the same endpoints collapse into one edge. If
you need multigraph identity, retain `Topology.get_all_edges()` as the
authoritative edge collection.

## Weighted centrality: explicit semantics

The default centrality is **unweighted** hop betweenness and hop closeness.
Changing the line width or visual color does not alter these values.

To interpret positive edge weights as **affinities** and convert them to
shortest-path costs $c_{ij}=1/w_{ij}$, explicitly opt in:

```python
from braintop.core.visualizer import TopologyVisualizer

visualizer = TopologyVisualizer(
    topology,
    config={"centrality_distance": "inverse_strength"},
)
figure = visualizer.create_figure(
    color_by="betweenness",
    size_by="degree",
)
```

Zero and negative weights are rejected in this opt-in mode. Unweighted degree
still counts neighbours; closeness/betweenness use the derived positive costs.
This choice is an **analysis convention**, not a measured geometry.

All layers in a figure share the same numeric color scale when colored by a
numeric metric. For performance, edges are batched into at most six 3D line
traces by `intra/inter` role and one of three coarse weight bands. These
width bands are qualitative. Raw weights remain in the topology and graph.

## Verification and open remainder

`tests/test_moving_frames.py` includes:

- a known-answer quarter-turn + translation;
- invalid rotation, reflection and non-finite coordinate rejection;
- a passive frame relabelling that must leave world positions and neighbours
  invariant;
- simultaneous global rigid motion as a matched control;
- relative motion of one layer as a **negative control** that legitimately
  changes a nearest-neighbour relation;
- world-coordinate rendering, sectional views, edge typing, global color-scale
  consistency, batched rendering and opt-in weighted centrality.

Passing these tests validates a narrow SE(3) **display-frame** contract. It
does **not** validate intrinsic sheaf transport, connections between distinct
manifolds, parallel transport, holonomy, or moving moving frames in the
differential-geometric sense.

The next research step requires a typed correspondence
`source-manifold × target-manifold -> admissible mapping + metric + remainder`,
including domain, units, uncertainty and a reference oracle. Do not infer
such a map from 3D drawing proximity.
