"""
Quick Demo - Showcase all BrainTop features in one script

This script demonstrates the key capabilities of BrainTop by creating
a small example of each layer type and rendering them together.
"""

import sys
sys.path.insert(0, '../src')

from braintop import (
    TopologyGraph,
    ConceptualLayer,
    RiemannianLayer,
    NeuralLayer,
    Renderer3D,
)


def create_demo_graph():
    """Create a demonstration topology graph with all layer types."""
    graph = TopologyGraph()
    graph.set_metadata("demo", "BrainTop Feature Showcase")
    
    # 1. Conceptual Layer - Simple hierarchy
    conceptual = ConceptualLayer("concepts")
    conceptual.add_node("Root", weight=1.0, type="root")
    conceptual.add_node("Branch_A", weight=0.8, type="branch")
    conceptual.add_node("Branch_B", weight=0.8, type="branch")
    conceptual.add_node("Leaf_1", weight=0.6, type="leaf")
    conceptual.add_node("Leaf_2", weight=0.6, type="leaf")
    
    conceptual.add_edge("Root", "Branch_A", type="parent_of")
    conceptual.add_edge("Root", "Branch_B", type="parent_of")
    conceptual.add_edge("Branch_A", "Leaf_1", type="parent_of")
    conceptual.add_edge("Branch_B", "Leaf_2", type="parent_of")
    
    graph.add_layer(conceptual)
    
    # 2. Riemannian Layer - Sphere manifold
    riemannian = RiemannianLayer("manifold", manifold_type="sphere")
    
    import numpy as np
    for i in range(8):
        angle = 2 * np.pi * i / 8
        riemannian.add_node(
            f"point_{i}",
            embedding=[np.cos(angle) * 0.7, np.sin(angle) * 0.7],
            weight=0.5 + 0.5 * i / 8
        )
    
    # Add some connections
    for i in range(7):
        riemannian.add_edge(f"point_{i}", f"point_{i+1}", type="flow")
    riemannian.add_edge("point_7", "point_0", type="flow")
    
    graph.add_layer(riemannian)
    
    # 3. Neural Layer - Small network
    neural = NeuralLayer("network")
    neural.set_layer_structure([4, 3, 2])
    
    graph.add_layer(neural)
    
    return graph


def main():
    """Run the quick demo."""
    print("=" * 60)
    print("BrainTop Quick Demo - Feature Showcase")
    print("=" * 60)
    
    # Create demonstration graph
    print("\n📊 Creating demonstration topology graph...")
    graph = create_demo_graph()
    
    # Print statistics for each layer
    print("\n📈 Layer Statistics:")
    print("-" * 60)
    
    for layer_name, layer in graph.get_all_layers().items():
        geometry = layer.compute_geometry()
        print(f"\n{layer_name.upper()}:")
        print(f"  Type: {geometry['type']}")
        print(f"  Nodes: {geometry['num_nodes']}")
        print(f"  Edges: {geometry.get('num_edges', 0)}")
        
        if 'num_levels' in geometry:
            print(f"  Hierarchy Levels: {geometry['num_levels']}")
        if 'manifold_type' in geometry:
            print(f"  Manifold: {geometry['manifold_type']}")
        if 'num_layers' in geometry:
            print(f"  Network Layers: {geometry['num_layers']}")
    
    # Create visualizations with different themes
    print("\n🎨 Generating visualizations...")
    
    themes = ["dark", "light", "scientific"]
    
    for theme in themes:
        renderer = Renderer3D(
            title=f"BrainTop Demo - {theme.capitalize()} Theme",
            theme=theme,
            width=1400,
            height=900
        )
        
        # Render all layers
        fig = renderer.render_graph(
            graph,
            layers=["concepts", "manifold", "network"],
            show_edges=True,
            node_size=7,
            edge_width=1.5
        )
        
        filename = f"demo_{theme}_theme.html"
        renderer.save(fig, filename)
        print(f"  ✓ {filename}")
    
    # Also create individual layer renders
    print("\n🎯 Generating individual layer views...")
    
    renderer = Renderer3D(theme="dark", width=1200, height=800)
    
    for layer_name, layer in graph.get_all_layers().items():
        fig = renderer.render_layer(
            layer,
            show_edges=True,
            node_size=10,
            edge_width=2
        )
        filename = f"demo_{layer_name}_only.html"
        renderer.save(fig, filename)
        print(f"  ✓ {filename}")
    
    print("\n" + "=" * 60)
    print("✨ Demo Complete!")
    print("=" * 60)
    print("\nGenerated Files:")
    print("  • demo_dark_theme.html")
    print("  • demo_light_theme.html")
    print("  • demo_scientific_theme.html")
    print("  • demo_concepts_only.html")
    print("  • demo_manifold_only.html")
    print("  • demo_network_only.html")
    print("\nOpen any HTML file in a web browser to explore the")
    print("interactive 3D visualizations!")
    print("\n🚀 BrainTop v0.1.0 - Ready for LLM Topology Visualization")


if __name__ == "__main__":
    main()
