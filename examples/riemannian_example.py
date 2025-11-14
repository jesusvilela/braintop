"""Example showing different Riemannian manifold types."""

from braintop.utils.builders import TopologyBuilder
from braintop.core.visualizer import TopologyVisualizer


def main():
    """Demonstrate different Riemannian manifold types."""
    
    print("Creating Riemannian manifolds topology...")
    builder = TopologyBuilder("manifolds_topology", "Riemannian Manifolds Showcase")
    
    # Sphere manifold
    print("Adding sphere manifold...")
    builder.add_riemannian_layer(
        "sphere",
        num_nodes=15,
        manifold_type="sphere",
        radius=2.0
    )
    
    # Torus manifold
    print("Adding torus manifold...")
    builder.add_riemannian_layer(
        "torus",
        num_nodes=20,
        manifold_type="torus",
        radius=1.5
    )
    
    # Hyperbolic manifold (Poincaré disk)
    print("Adding hyperbolic manifold...")
    builder.add_riemannian_layer(
        "hyperbolic",
        num_nodes=12,
        manifold_type="hyperbolic",
        curvature=-1.0
    )
    
    # Connect manifolds
    print("Creating inter-manifold connections...")
    builder.connect_layers("sphere", "torus", "nearest", num_connections=3)
    builder.connect_layers("torus", "hyperbolic", "nearest", num_connections=2)
    
    # Add metadata
    builder.set_metadata(
        description="Showcase of different Riemannian manifolds",
        manifolds=["sphere", "torus", "hyperbolic"]
    )
    
    # Build
    topology = builder.build()
    
    print(f"\nTopology created: {topology.name}")
    stats = topology.compute_statistics()
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total edges: {stats['total_edges']}")
    print(f"Layer types: {stats['layer_types']}")
    
    # Visualize
    print("\nCreating visualization...")
    visualizer = TopologyVisualizer(topology)
    fig = visualizer.create_figure()
    
    # Customize title
    fig.update_layout(
        title="Riemannian Manifolds: Sphere, Torus, and Hyperbolic Space"
    )
    
    # Save
    output_file = "riemannian_manifolds.html"
    visualizer.save(output_file)
    print(f"Saved visualization to {output_file}")
    
    # Uncomment to open in browser
    # visualizer.show()


if __name__ == "__main__":
    main()
