"""
Neural Network Example: Visualizing Transformer Architecture

This example demonstrates the Neural layer for visualizing
neural network architectures, particularly transformers.
"""

import sys
sys.path.insert(0, '../src')

from braintop import TopologyGraph, NeuralLayer, Renderer3D


def main():
    # Create a topology graph
    graph = TopologyGraph()
    
    # Create a neural layer for a simple transformer-like architecture
    transformer_layer = NeuralLayer(
        name="transformer",
        layer_spacing=3.5,
        neuron_spacing=1.2
    )
    
    # Define architecture: [embedding, attention1, ffn1, attention2, ffn2, output]
    layer_structure = [8, 6, 8, 6, 8, 4]
    
    # Create the network structure
    transformer_layer.set_layer_structure(layer_structure)
    
    # Add the layer to the graph
    graph.add_layer(transformer_layer)
    
    # Print statistics
    geometry = transformer_layer.compute_geometry()
    print("Transformer Architecture Statistics:")
    print(f"  Total Neurons: {geometry['num_nodes']}")
    print(f"  Total Connections: {geometry['num_edges']}")
    print(f"  Number of Layers: {geometry['num_layers']}")
    print(f"  Avg Neurons per Layer: {geometry['avg_neurons_per_layer']:.1f}")
    print(f"  Avg In-Degree: {geometry['avg_in_degree']:.2f}")
    print(f"  Avg Out-Degree: {geometry['avg_out_degree']:.2f}")
    
    # Create a second layer showing attention patterns
    attention_layer = NeuralLayer(
        name="attention_heads",
        layer_spacing=2.0,
        neuron_spacing=2.0
    )
    
    # Simulate multi-head attention (4 heads, 8 queries each)
    num_heads = 4
    queries_per_head = 8
    
    for head in range(num_heads):
        for query in range(queries_per_head):
            node_id = f"head_{head}_query_{query}"
            attention_layer.add_node(
                node_id,
                layer=head,
                head=head,
                query=query,
                activation=0.5 + 0.5 * (query / queries_per_head)
            )
    
    # Add attention connections (each head attends to previous head)
    for head in range(1, num_heads):
        for q1 in range(queries_per_head):
            for q2 in range(queries_per_head):
                # Simulate attention weights
                if abs(q1 - q2) <= 2:  # Local attention
                    source = f"head_{head-1}_query_{q2}"
                    target = f"head_{head}_query_{q1}"
                    weight = 1.0 / (1.0 + abs(q1 - q2))
                    attention_layer.add_edge(
                        source, target,
                        weight=weight,
                        type="attention"
                    )
    
    graph.add_layer(attention_layer)
    
    # Create separate renderers for each visualization
    renderer = Renderer3D(
        title="Neural Network Topology Visualizations",
        theme="dark",
        width=1400,
        height=900
    )
    
    # Render both layers together
    fig = renderer.render_graph(
        graph,
        layers=["transformer", "attention_heads"],
        show_edges=True,
        node_size=6,
        edge_width=1
    )
    
    # Save to HTML file
    renderer.save(fig, "neural_topology.html")
    print("\nVisualization saved to: neural_topology.html")
    print("Open this file in a web browser to view the interactive 3D visualization!")
    
    # Also create a single-layer view of just the transformer
    fig_transformer = renderer.render_layer(
        transformer_layer,
        show_edges=True,
        node_size=8,
        edge_width=1.5
    )
    renderer.save(fig_transformer, "transformer_architecture.html")
    print("Transformer-only view saved to: transformer_architecture.html")


if __name__ == "__main__":
    main()
