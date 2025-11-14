"""LLM integration example: Using LLMConnector to create topology from text."""

from braintop.integrations.llm_connector import LLMConnector
from braintop.utils.builders import TopologyBuilder
from braintop.core.visualizer import TopologyVisualizer


def main():
    """Create topology from LLM embeddings."""
    
    # Initialize LLM connector (using mock for demo)
    print("Connecting to LLM...")
    llm = LLMConnector(provider="mock", model_name="demo-model")
    llm.connect()
    
    print(f"Connected to: {llm.get_model_info()}")
    
    # Define concepts to visualize
    texts = [
        "Neural networks learn patterns from data",
        "Transformers use attention mechanisms",
        "Deep learning models have many layers",
        "Embeddings represent semantic meaning",
        "Language models generate text",
        "Attention helps focus on relevant information",
        "Training requires large datasets",
        "Fine-tuning adapts models to specific tasks"
    ]
    
    # Create topology using LLM embeddings
    print("\nCreating topology from LLM embeddings...")
    builder = TopologyBuilder("llm_topology", "LLM Embedding Topology")
    
    # Add layer from LLM
    builder.add_layer_from_llm(
        "llm_concepts",
        llm,
        texts,
        projection_method="pca"
    )
    
    # Build topology
    topology = builder.build()
    
    print(f"Created topology with {len(topology.layers)} layer(s)")
    print(f"Total nodes: {topology.compute_statistics()['total_nodes']}")
    print(f"Total edges: {topology.compute_statistics()['total_edges']}")
    
    # Visualize
    print("\nCreating visualization...")
    visualizer = TopologyVisualizer(topology)
    visualizer.create_figure()
    
    # Save
    output_file = "llm_topology.html"
    visualizer.save(output_file)
    print(f"Saved visualization to {output_file}")
    
    # Disconnect
    llm.disconnect()
    
    # Uncomment to open in browser
    # visualizer.show()


if __name__ == "__main__":
    main()
