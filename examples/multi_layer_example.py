"""Multi-layer example: Creating complex topologies with multiple layer types."""

import numpy as np
from braintop.utils.builders import TopologyBuilder
from braintop.core.visualizer import TopologyVisualizer


def main():
    """Create a complex multi-layer topology."""
    
    print("Creating multi-layer topology...")
    builder = TopologyBuilder("complex_topology", "Multi-Layer Neural Topology")
    
    # Layer 1: Input concepts (conceptual layer)
    input_concepts = [
        "Text Input", "Image Input", "Audio Input",
        "Video Input", "Multimodal Input"
    ]
    builder.add_conceptual_layer(
        "input_layer",
        num_nodes=5,
        concepts=input_concepts
    )
    
    # Layer 2: Processing sphere (Riemannian sphere)
    builder.add_riemannian_layer(
        "processing_sphere",
        num_nodes=12,
        manifold_type="sphere",
        radius=2.0
    )
    
    # Layer 3: Hidden representations (conceptual layer)
    hidden_concepts = [
        "Feature 1", "Feature 2", "Feature 3", "Feature 4",
        "Feature 5", "Feature 6", "Feature 7", "Feature 8"
    ]
    # Create synthetic embeddings
    hidden_embeddings = np.random.randn(8, 256)
    builder.add_conceptual_layer(
        "hidden_layer",
        num_nodes=8,
        concepts=hidden_concepts,
        embeddings=hidden_embeddings,
        projection_method="pca"
    )
    
    # Layer 4: Output torus (Riemannian torus)
    builder.add_riemannian_layer(
        "output_torus",
        num_nodes=10,
        manifold_type="torus",
        radius=1.5
    )
    
    # Layer 5: Decision concepts (conceptual layer)
    output_concepts = ["Classification", "Generation", "Prediction"]
    builder.add_conceptual_layer(
        "decision_layer",
        num_nodes=3,
        concepts=output_concepts
    )
    
    # Connect layers in sequence
    print("Connecting layers...")
    builder.connect_layers("input_layer", "processing_sphere", "nearest", num_connections=4)
    builder.connect_layers("processing_sphere", "hidden_layer", "nearest", num_connections=3)
    builder.connect_layers("hidden_layer", "output_torus", "nearest", num_connections=3)
    builder.connect_layers("output_torus", "decision_layer", "all")
    
    # Add metadata
    builder.set_metadata(
        description="Complex multi-layer topology demonstrating various layer types",
        architecture="Sequential multi-layer",
        layer_count=5
    )
    
    # Build
    topology = builder.build()
    
    print(f"\nTopology: {topology.name}")
    print(f"Statistics: {topology.compute_statistics()}")
    print(f"Metadata: {topology.metadata}")
    
    # Visualize
    print("\nCreating visualization...")
    visualizer = TopologyVisualizer(topology)
    visualizer.create_figure()
    
    # Save
    output_file = "complex_topology.html"
    visualizer.save(output_file)
    print(f"Saved visualization to {output_file}")
    
    # Get network graph for analysis
    graph = visualizer.get_network_graph()
    print("\nNetwork statistics:")
    print(f"  Nodes: {graph.number_of_nodes()}")
    print(f"  Edges: {graph.number_of_edges()}")
    
    # Uncomment to open in browser
    # visualizer.show()


if __name__ == "__main__":
    main()
