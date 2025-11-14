"""
Advanced Example: Multi-Layer Visualization with Integration

This example demonstrates advanced usage including multiple layers,
LLM integration, and data transformation.
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
from braintop import (
    TopologyGraph,
    ConceptualLayer,
    RiemannianLayer,
    NeuralLayer,
    Renderer3D,
)
from braintop.integration import DataTransformer


def create_llm_agent_topology():
    """Create a topology representing an LLM agent system."""
    graph = TopologyGraph()
    graph.set_metadata("system", "LLM Agent Orchestrator")
    graph.set_metadata("timestamp", "2024-11-14")
    
    # Conceptual layer: Agent relationships
    conceptual = ConceptualLayer(name="agent_hierarchy")
    
    # Add agents
    conceptual.add_node("orchestrator", weight=1.0, type="coordinator")
    conceptual.add_node("planner", weight=0.9, type="agent")
    conceptual.add_node("executor", weight=0.9, type="agent")
    conceptual.add_node("critic", weight=0.9, type="agent")
    
    # Add tools
    conceptual.add_node("code_tool", weight=0.7, type="tool")
    conceptual.add_node("search_tool", weight=0.7, type="tool")
    conceptual.add_node("memory_tool", weight=0.7, type="tool")
    
    # Add edges
    conceptual.add_edge("orchestrator", "planner", type="parent_of")
    conceptual.add_edge("orchestrator", "executor", type="parent_of")
    conceptual.add_edge("orchestrator", "critic", type="parent_of")
    conceptual.add_edge("executor", "code_tool", type="uses")
    conceptual.add_edge("planner", "search_tool", type="uses")
    conceptual.add_edge("critic", "memory_tool", type="uses")
    
    graph.add_layer(conceptual)
    
    # Riemannian layer: Semantic embedding space
    riemannian = RiemannianLayer(
        name="semantic_space",
        manifold_type="sphere",
        curvature=0.6
    )
    
    # Add same nodes with embeddings
    nodes_with_embeddings = {
        "orchestrator": [0.0, 0.0],
        "planner": [0.8, 0.2],
        "executor": [0.2, 0.8],
        "critic": [-0.8, 0.3],
        "code_tool": [0.3, 0.9],
        "search_tool": [0.9, 0.1],
        "memory_tool": [-0.9, 0.4],
    }
    
    for node_id, embedding in nodes_with_embeddings.items():
        riemannian.add_node(
            node_id,
            embedding=embedding,
            weight=np.random.rand()
        )
    
    # Add semantic relationships
    riemannian.add_edge("planner", "search_tool", type="semantic")
    riemannian.add_edge("executor", "code_tool", type="semantic")
    riemannian.add_edge("critic", "memory_tool", type="semantic")
    
    graph.add_layer(riemannian)
    
    # Neural layer: Underlying LLM architecture
    neural = NeuralLayer(name="llm_backbone")
    neural.set_layer_structure([512, 256, 128, 256, 512])  # Encoder-decoder style
    
    graph.add_layer(neural)
    
    return graph


def main():
    print("Creating LLM Agent Topology...")
    graph = create_llm_agent_topology()
    
    # Print metadata
    print(f"\nSystem: {graph.get_metadata('system')}")
    print(f"Timestamp: {graph.get_metadata('timestamp')}")
    
    # Print statistics for each layer
    print("\nLayer Statistics:")
    print("-" * 60)
    
    for layer_name, layer in graph.get_all_layers().items():
        geometry = layer.compute_geometry()
        print(f"\n{layer_name.upper()} Layer:")
        print(f"  Type: {geometry['type']}")
        print(f"  Nodes: {geometry['num_nodes']}")
        
        if 'num_levels' in geometry:
            print(f"  Hierarchy Levels: {geometry['num_levels']}")
        if 'manifold_type' in geometry:
            print(f"  Manifold: {geometry['manifold_type']}")
        if 'num_layers' in geometry:
            print(f"  Network Layers: {geometry['num_layers']}")
    
    # Create renderer
    renderer = Renderer3D(
        title="LLM Agent System - Multi-Layer Topology",
        theme="scientific",
        width=1600,
        height=1000
    )
    
    # Render all layers
    fig = renderer.render_graph(
        graph,
        layers=["agent_hierarchy", "semantic_space", "llm_backbone"],
        show_edges=True,
        node_size=8,
        edge_width=2
    )
    
    # Save visualization
    renderer.save(fig, "multi_layer_topology.html")
    print("\n" + "=" * 60)
    print("Visualization saved to: multi_layer_topology.html")
    print("=" * 60)
    
    # Demonstrate data transformation
    print("\nDemonstrating DataTransformer...")
    
    # Create a simple dictionary structure
    dict_data = {
        "nodes": [
            {"id": "input", "weight": 1.0},
            {"id": "hidden", "weight": 0.8},
            {"id": "output", "weight": 0.6},
        ],
        "edges": [
            {"source": "input", "target": "hidden", "weight": 0.9},
            {"source": "hidden", "target": "output", "weight": 0.7},
        ],
    }
    
    # Transform to topology graph
    simple_graph = DataTransformer.from_dict(dict_data, layer_type="conceptual")
    print(f"Created graph from dictionary with {len(simple_graph.get_all_layers())} layer(s)")
    
    # Render simple graph
    simple_fig = renderer.render_graph(simple_graph, show_edges=True)
    renderer.save(simple_fig, "simple_topology.html")
    print("Simple topology saved to: simple_topology.html")
    
    print("\n✨ All visualizations generated successfully!")
    print("Open the HTML files in a web browser to explore the interactive 3D visualizations.")


if __name__ == "__main__":
    main()
