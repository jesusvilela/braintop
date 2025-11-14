"""Command-line interface for braintop."""

import argparse
import sys
from typing import Optional

from braintop import __version__
from braintop.core.topology import Topology
from braintop.core.visualizer import TopologyVisualizer
from braintop.utils.builders import TopologyBuilder


def create_demo_topology() -> Topology:
    """Create a demo topology for visualization."""
    print("Creating demo topology...")
    
    builder = TopologyBuilder("demo_topology", "Demo Multi-Layer Topology")
    
    # Add conceptual layer
    concepts = [
        "Neural Network", "Deep Learning", "Transformer",
        "Attention", "Embedding", "Token"
    ]
    builder.add_conceptual_layer(
        "conceptual_layer",
        num_nodes=6,
        concepts=concepts,
        projection_method="pca"
    )
    
    # Add Riemannian layer (sphere)
    builder.add_riemannian_layer(
        "sphere_layer",
        num_nodes=8,
        manifold_type="sphere",
        radius=2.0
    )
    
    # Add Riemannian layer (torus)
    builder.add_riemannian_layer(
        "torus_layer",
        num_nodes=10,
        manifold_type="torus",
        radius=1.5
    )
    
    # Connect layers
    builder.connect_layers("conceptual_layer", "sphere_layer", "nearest", num_connections=3)
    builder.connect_layers("sphere_layer", "torus_layer", "random", num_connections=2)
    
    # Set metadata
    builder.set_metadata(
        description="Demo topology showing multiple layer types",
        created_by="braintop CLI"
    )
    
    topology = builder.build()
    print(f"Created topology with {len(topology.layers)} layers")
    print(f"Statistics: {topology.compute_statistics()}")
    
    return topology


def visualize_command(args: argparse.Namespace) -> int:
    """Handle the visualize command."""
    print(f"Braintop v{__version__} - 3D Topology Visualizer")
    print("=" * 50)
    
    # Create demo topology
    topology = create_demo_topology()
    
    # Create visualizer
    print("\nCreating visualization...")
    visualizer = TopologyVisualizer(topology)
    fig = visualizer.create_figure()
    
    # Save or show
    if args.output:
        print(f"Saving visualization to {args.output}...")
        visualizer.save(args.output, format=args.format)
        print("Done!")
    else:
        print("Opening visualization in browser...")
        visualizer.show()
    
    return 0


def info_command(args: argparse.Namespace) -> int:
    """Handle the info command."""
    print(f"Braintop v{__version__}")
    print("A beautiful rich 3D topology visualizer for LLMs")
    print()
    print("Features:")
    print("  - Modular architecture with multiple layer types")
    print("  - Conceptual layers for semantic relationships")
    print("  - Riemannian layers for curved geometry (sphere, torus, hyperbolic)")
    print("  - LLM integration for extracting embeddings and relationships")
    print("  - Interactive 3D visualization with Plotly")
    print("  - Plugin system for custom layers and integrations")
    print()
    print("Supported layer types:")
    print("  - Conceptual: Semantic/concept-based topology")
    print("  - Riemannian: Curved manifolds (sphere, torus, hyperbolic)")
    print("  - Custom: Extensible for custom geometries")
    
    return 0


def main(argv: Optional[list] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Braintop - 3D Topology Visualizer for LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"braintop {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Visualize command
    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Create and visualize a demo topology"
    )
    visualize_parser.add_argument(
        "-o", "--output",
        help="Output file path (default: show in browser)"
    )
    visualize_parser.add_argument(
        "-f", "--format",
        choices=["html", "png", "jpg", "svg"],
        default="html",
        help="Output format (default: html)"
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about braintop"
    )
    
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == "visualize":
        return visualize_command(args)
    elif args.command == "info":
        return info_command(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
