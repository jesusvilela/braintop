"""Basic example: Creating and visualizing a simple topology."""

from braintop import Topology, TopologyVisualizer, ConceptualLayer, RiemannianLayer
from braintop.utils.builders import TopologyBuilder


def main():
    """Create and visualize a basic topology."""
    
    # Method 1: Using the builder
    print("Creating topology with builder...")
    builder = TopologyBuilder("example_topology", "Basic Example Topology")
    
    # Add a conceptual layer
    concepts = ["Attention", "Embedding", "Transformer", "Decoder", "Encoder"]
    builder.add_conceptual_layer(
        "concepts",
        num_nodes=5,
        concepts=concepts
    )
    
    # Add a Riemannian sphere layer
    builder.add_riemannian_layer(
        "sphere",
        num_nodes=8,
        manifold_type="sphere",
        radius=1.5
    )
    
    # Connect the layers
    builder.connect_layers("concepts", "sphere", "nearest", num_connections=3)
    
    # Build the topology
    topology = builder.build()
    
    print(f"Created topology: {topology.name}")
    print(f"Layers: {len(topology.layers)}")
    print(f"Statistics: {topology.compute_statistics()}")
    
    # Visualize
    print("\nCreating visualization...")
    visualizer = TopologyVisualizer(topology)
    visualizer.create_figure()
    
    # Save to file
    output_file = "basic_topology.html"
    visualizer.save(output_file)
    print(f"Saved visualization to {output_file}")
    
    # Uncomment to open in browser
    # visualizer.show()


if __name__ == "__main__":
    main()
