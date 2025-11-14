"""
Basic Example: Visualizing a Simple Conceptual Topology

This example demonstrates the basic usage of BrainTop to create
and visualize a conceptual topology layer.
"""

import sys
sys.path.insert(0, '../src')

from braintop import TopologyGraph, ConceptualLayer, Renderer3D


def main():
    # Create a topology graph
    graph = TopologyGraph()
    
    # Create a conceptual layer
    layer = ConceptualLayer(name="concepts", hierarchy_strength=0.8)
    
    # Add some conceptual nodes
    layer.add_node("AI", weight=1.0, type="root")
    layer.add_node("ML", weight=0.9, type="concept")
    layer.add_node("DL", weight=0.8, type="concept")
    layer.add_node("NLP", weight=0.8, type="concept")
    layer.add_node("CV", weight=0.8, type="concept")
    layer.add_node("Transformers", weight=0.7, type="technique")
    layer.add_node("CNNs", weight=0.7, type="technique")
    layer.add_node("RNNs", weight=0.7, type="technique")
    layer.add_node("GPT", weight=0.6, type="model")
    layer.add_node("BERT", weight=0.6, type="model")
    layer.add_node("ResNet", weight=0.6, type="model")
    
    # Add hierarchical relationships
    layer.add_edge("AI", "ML", type="parent_of", strength=1.0)
    layer.add_edge("ML", "DL", type="parent_of", strength=0.9)
    layer.add_edge("DL", "NLP", type="parent_of", strength=0.8)
    layer.add_edge("DL", "CV", type="parent_of", strength=0.8)
    layer.add_edge("NLP", "Transformers", type="parent_of", strength=0.7)
    layer.add_edge("CV", "CNNs", type="parent_of", strength=0.7)
    layer.add_edge("NLP", "RNNs", type="parent_of", strength=0.7)
    layer.add_edge("Transformers", "GPT", type="parent_of", strength=0.6)
    layer.add_edge("Transformers", "BERT", type="parent_of", strength=0.6)
    layer.add_edge("CNNs", "ResNet", type="parent_of", strength=0.6)
    
    # Add some cross-connections
    layer.add_edge("Transformers", "CNNs", type="related", strength=0.3)
    layer.add_edge("GPT", "BERT", type="related", strength=0.5)
    
    # Add layer to graph
    graph.add_layer(layer)
    
    # Print some statistics
    geometry = layer.compute_geometry()
    print("Conceptual Layer Statistics:")
    print(f"  Nodes: {geometry['num_nodes']}")
    print(f"  Edges: {geometry['num_edges']}")
    print(f"  Hierarchy Levels: {geometry['num_levels']}")
    print(f"  Max Depth: {geometry['max_depth']}")
    print(f"  Density: {geometry['density']:.3f}")
    
    # Create a renderer
    renderer = Renderer3D(
        title="AI Concepts Topology",
        theme="dark",
        width=1200,
        height=800
    )
    
    # Render the layer
    fig = renderer.render_layer(layer, show_edges=True, node_size=10, edge_width=2)
    
    # Save to HTML file
    renderer.save(fig, "conceptual_topology.html")
    print("\nVisualization saved to: conceptual_topology.html")
    print("Open this file in a web browser to view the interactive 3D visualization!")


if __name__ == "__main__":
    main()
