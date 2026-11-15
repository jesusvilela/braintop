"""Command-line interface for braintop."""

import argparse
import os
import sys
from typing import Optional, Any

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
    stats = topology.compute_statistics()
    layer_stats = topology.compute_layer_statistics()
    topology.metadata["layer_statistics"] = layer_stats
    print(f"Created topology with {len(topology.layers)} layers")
    print(f"Statistics: {stats}")
    print(f"Layer statistics: {layer_stats}")
    
    return topology


def visualize_command(args: argparse.Namespace) -> int:
    """Handle the visualize command."""
    print(f"Braintop v{__version__} - 3D Topology Visualizer")
    print("=" * 50)

    def save_figure(figure: Any, path: str) -> None:
        if args.format == "html":
            figure.write_html(path)
        else:
            figure.write_image(path, format=args.format)
    
    # Create demo topology
    topology = create_demo_topology()
    
    # Create visualizer
    print("\nCreating visualization...")
    visualizer = TopologyVisualizer(topology)
    fig = visualizer.create_figure(
        color_by=args.color_by,
        size_by=args.size_by,
    )
    
    # Save or show
    if args.output:
        print(f"Saving visualization to {args.output}...")
        visualizer.save(args.output, format=args.format)
        base, _ = os.path.splitext(args.output)
        if args.sectional:
            sectional_path = f"{base}_sectional.{args.format}"
            print(f"Saving sectional study to {sectional_path}...")
            sectional_fig = visualizer.create_sectional_study(
                axis=args.section_axis,
                sections=args.sections,
            )
            save_figure(sectional_fig, sectional_path)
        if args.ablations:
            ablation_figs = visualizer.create_ablation_figures()
            for name, ablation_fig in ablation_figs.items():
                ablation_path = f"{base}_{name}.{args.format}"
                print(f"Saving ablation study to {ablation_path}...")
                save_figure(ablation_fig, ablation_path)
        print("Done!")
    else:
        print("Opening visualization in browser...")
        visualizer.show()
        if args.sectional:
            visualizer.create_sectional_study(
                axis=args.section_axis,
                sections=args.sections,
            ).show()
        if args.ablations:
            for ablation_fig in visualizer.create_ablation_figures().values():
                ablation_fig.show()
    
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
    visualize_parser.add_argument(
        "--color-by",
        choices=["layer", "degree", "betweenness", "closeness", "salience", "normalized_radius"],
        default="layer",
        help="Color nodes by metric (default: layer)"
    )
    visualize_parser.add_argument(
        "--size-by",
        choices=["degree", "betweenness", "closeness", "salience", "normalized_radius"],
        default="degree",
        help="Size nodes by metric (default: degree)"
    )
    visualize_parser.add_argument(
        "--sectional",
        action="store_true",
        help="Generate sectional studies (2D slices)"
    )
    visualize_parser.add_argument(
        "--section-axis",
        choices=["x", "y", "z"],
        default="z",
        help="Axis to slice for sectional studies (default: z)"
    )
    visualize_parser.add_argument(
        "--sections",
        type=int,
        default=3,
        help="Number of sections for sectional studies (default: 3)"
    )
    visualize_parser.add_argument(
        "--ablations",
        action="store_true",
        help="Generate ablation studies (layer removal/isolations)"
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
