"""
Riemannian Manifold Example: Visualizing Latent Space Topology

This example demonstrates the Riemannian layer for visualizing
embeddings on curved manifolds.
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
from braintop import TopologyGraph, RiemannianLayer, Renderer3D


def main():
    # Create a topology graph
    graph = TopologyGraph()
    
    # Create Riemannian layers with different manifold types
    sphere_layer = RiemannianLayer(
        name="sphere",
        manifold_type="sphere",
        curvature=0.8
    )
    
    hyperbolic_layer = RiemannianLayer(
        name="hyperbolic",
        manifold_type="hyperbolic",
        curvature=0.5
    )
    
    torus_layer = RiemannianLayer(
        name="torus",
        manifold_type="torus",
        curvature=0.3
    )
    
    # Generate some random embeddings (simulating word embeddings)
    words = [
        "king", "queen", "man", "woman",
        "cat", "dog", "pet", "animal",
        "car", "vehicle", "transport", "train",
        "happy", "joy", "sad", "emotion",
    ]
    
    # Add nodes to each layer with 2D embeddings
    for i, word in enumerate(words):
        # Create a simple 2D embedding
        angle = 2 * np.pi * i / len(words)
        radius = 0.5 + 0.3 * np.sin(2 * angle)
        
        embedding = [radius * np.cos(angle), radius * np.sin(angle)]
        
        # Add to all layers
        for layer in [sphere_layer, hyperbolic_layer, torus_layer]:
            layer.add_node(
                word,
                embedding=embedding,
                category=i // 4,  # Group into categories
                weight=np.random.rand()
            )
    
    # Add some semantic edges
    semantic_pairs = [
        ("king", "queen"), ("man", "woman"),
        ("cat", "dog"), ("cat", "pet"), ("dog", "pet"),
        ("pet", "animal"), ("car", "vehicle"),
        ("vehicle", "transport"), ("train", "transport"),
        ("happy", "joy"), ("sad", "emotion"),
    ]
    
    for word1, word2 in semantic_pairs:
        for layer in [sphere_layer, hyperbolic_layer, torus_layer]:
            layer.add_edge(word1, word2, type="semantic", strength=0.8)
    
    # Add layers to graph
    graph.add_layer(sphere_layer)
    graph.add_layer(hyperbolic_layer)
    graph.add_layer(torus_layer)
    
    # Print statistics for each layer
    print("Riemannian Layer Statistics:\n")
    for layer_name, layer in graph.get_all_layers().items():
        geometry = layer.compute_geometry()
        print(f"{layer_name.capitalize()} Layer:")
        print(f"  Manifold Type: {geometry['manifold_type']}")
        print(f"  Curvature: {geometry['curvature']:.2f}")
        print(f"  Avg Geodesic Distance: {geometry['avg_geodesic_distance']:.3f}")
        print(f"  Nodes: {geometry['num_nodes']}")
        print()
    
    # Create a renderer
    renderer = Renderer3D(
        title="Word Embeddings on Riemannian Manifolds",
        theme="scientific",
        width=1400,
        height=900
    )
    
    # Render all layers in one visualization
    fig = renderer.render_graph(
        graph,
        layers=["sphere", "hyperbolic", "torus"],
        show_edges=True,
        node_size=8,
        edge_width=1.5
    )
    
    # Save to HTML file
    renderer.save(fig, "riemannian_topology.html")
    print("Visualization saved to: riemannian_topology.html")
    print("Open this file in a web browser to view the interactive 3D visualization!")


if __name__ == "__main__":
    main()
