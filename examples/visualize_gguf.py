#!/usr/bin/env python3
"""
Generalized GGUF Model Visualizer

This script can load any GGUF file and create an interactive 3D topology
visualization of the model architecture, showing layers, attention mechanisms,
and feed-forward networks.

Usage:
    python visualize_gguf.py <path_to_gguf_file> [options]

Examples:
    # Basic visualization
    python visualize_gguf.py model.gguf

    # With custom output filename
    python visualize_gguf.py model.gguf --output model_viz.html

    # Show detailed layer info
    python visualize_gguf.py model.gguf --verbose
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from braintop.integrations.gguf_loader import GGUFLoader
from braintop.utils.builders import TopologyBuilder
from braintop.core.visualizer import TopologyVisualizer


def create_gguf_topology(gguf_file: str, verbose: bool = False) -> TopologyBuilder:
    """Create a topology from a GGUF file.

    Args:
        gguf_file: Path to the GGUF file
        verbose: Whether to print detailed information

    Returns:
        TopologyBuilder with the model architecture
    """
    print("="*80)
    print("GGUF MODEL TOPOLOGY VISUALIZER")
    print("="*80)

    # Step 1: Load GGUF file
    print("\n[STEP 1] Loading GGUF file...")
    loader = GGUFLoader()
    loader.load_file(gguf_file)

    # Print summary
    if verbose:
        loader.print_summary()
    else:
        arch = loader.architecture
        print(f"\nModel: {arch.get('name', 'Unknown')}")
        print(f"Architecture: {arch.get('type', 'Unknown')}")
        if 'num_layers' in arch:
            print(f"Layers: {arch['num_layers']}")
        if 'embedding_dim' in arch:
            print(f"Embedding Dimension: {arch['embedding_dim']}")
        total_params = sum(t['elements'] for t in loader.tensor_info)
        print(f"Total Parameters: {total_params:,} ({total_params / 1e9:.2f}B)")

    # Step 2: Extract topology data
    print("\n[STEP 2] Extracting model architecture...")
    topology_data = loader.create_topology_data()

    print(f"Identified {len(topology_data['layers'])} topology layers:")
    for layer in topology_data['layers']:
        print(f"  - {layer['name']}: {layer['num_nodes']} nodes ({layer['type']} geometry)")

    # Step 3: Build topology
    print("\n[STEP 3] Building 3D topology...")

    model_name = loader.architecture.get('name', Path(gguf_file).stem)
    builder = TopologyBuilder(
        topology_id=f"gguf_{Path(gguf_file).stem}",
        name=f"{model_name} Architecture"
    )

    # Add metadata
    builder.set_metadata(
        source_file=str(gguf_file),
        architecture=loader.architecture,
        total_params=topology_data['total_params'],
        file_size_gb=topology_data['file_info']['size_gb']
    )

    # Create layers
    layer_configs = {
        'conceptual': {'projection_method': 'pca', 'spacing_factor': 2.0},
        'sphere': {'radius': 3.0},
        'hyperbolic': {'curvature': -1.0}
    }

    created_layers = []

    for layer_data in topology_data['layers']:
        layer_id = layer_data['id']
        layer_type = layer_data['type']
        num_nodes = layer_data['num_nodes']
        labels = layer_data.get('node_labels', [])

        # Ensure we have enough labels
        while len(labels) < num_nodes:
            labels.append(f"node_{len(labels)}")

        if layer_type == 'conceptual':
            builder.add_conceptual_layer(
                layer_id,
                num_nodes=num_nodes,
                concepts=labels[:num_nodes],
                **layer_configs['conceptual']
            )
        elif layer_type == 'sphere':
            builder.add_riemannian_layer(
                layer_id,
                num_nodes=num_nodes,
                manifold_type='sphere',
                labels=labels[:num_nodes],
                **layer_configs['sphere']
            )
        elif layer_type == 'hyperbolic':
            builder.add_riemannian_layer(
                layer_id,
                num_nodes=num_nodes,
                manifold_type='hyperbolic',
                labels=labels[:num_nodes],
                **layer_configs['hyperbolic']
            )

        created_layers.append(layer_id)
        print(f"  ✓ Created layer: {layer_data['name']} ({num_nodes} nodes)")

    # Step 4: Connect layers
    print("\n[STEP 4] Connecting layers...")

    # Connect sequential layers
    for i in range(len(created_layers) - 1):
        source_id = created_layers[i]
        target_id = created_layers[i + 1]

        # Determine connection strategy based on layer types
        source_layer = next(l for l in topology_data['layers'] if l['id'] == source_id)
        target_layer = next(l for l in topology_data['layers'] if l['id'] == target_id)

        # Use fewer connections for large layers
        source_nodes = source_layer['num_nodes']
        target_nodes = target_layer['num_nodes']
        max_connections = min(5, min(source_nodes, target_nodes))

        builder.connect_layers(
            source_id,
            target_id,
            connection_type='nearest',
            num_connections=max_connections
        )
        print(f"  ✓ Connected {source_layer['name']} → {target_layer['name']} "
              f"({max_connections} connections per node)")

    # Optional: Connect attention and FFN layers within same block
    attn_ffn_pairs = []
    for i, layer_data in enumerate(topology_data['layers']):
        if 'attn' in layer_data['id']:
            # Look for corresponding FFN layer
            layer_num = layer_data['metadata'].get('layer_num')
            if layer_num is not None:
                ffn_id = f'layer_{layer_num}_ffn'
                if ffn_id in created_layers:
                    attn_ffn_pairs.append((layer_data['id'], ffn_id))

    if attn_ffn_pairs:
        print("\n  Connecting attention-FFN pairs:")
        for attn_id, ffn_id in attn_ffn_pairs:
            builder.connect_layers(
                attn_id,
                ffn_id,
                connection_type='random',
                num_connections=2
            )
            print(f"  ✓ Connected {attn_id} ↔ {ffn_id}")

    return builder


def visualize_gguf(
    gguf_file: str,
    output_file: Optional[str] = None,
    show_browser: bool = False,
    verbose: bool = False
) -> str:
    """Create and save a visualization of a GGUF model.

    Args:
        gguf_file: Path to the GGUF file
        output_file: Optional output HTML filename
        show_browser: Whether to open in browser
        verbose: Whether to print detailed information

    Returns:
        Path to the output HTML file
    """
    # Create topology
    builder = create_gguf_topology(gguf_file, verbose=verbose)
    topology = builder.build()

    # Create visualization
    print("\n[STEP 5] Creating 3D visualization...")
    visualizer = TopologyVisualizer(topology)
    figure = visualizer.create_figure()

    stats = topology.compute_statistics()
    print(f"  ✓ Visualization created")
    print(f"    - Total nodes: {stats['total_nodes']}")
    print(f"    - Total edges: {stats['total_edges']}")
    print(f"    - Layers: {stats['num_layers']}")

    # Save to HTML
    if output_file is None:
        output_file = f"{Path(gguf_file).stem}_topology.html"

    print(f"\n[STEP 6] Exporting to HTML...")
    visualizer.save(output_file, format='html')

    file_size = Path(output_file).stat().st_size / (1024**2)
    print(f"  ✓ Saved to: {output_file} ({file_size:.1f} MB)")

    # Optionally show in browser
    if show_browser:
        print(f"\n[STEP 7] Opening in browser...")
        visualizer.show()

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nTo view: open {output_file} in your browser")
    print("\nInteractive features:")
    print("  - Rotate: Click and drag")
    print("  - Zoom: Scroll wheel")
    print("  - Pan: Right-click and drag")
    print("  - Hover: View node details")

    return output_file


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Visualize GGUF model architecture in 3D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python visualize_gguf.py model.gguf

  # Custom output file
  python visualize_gguf.py model.gguf -o my_model.html

  # Show in browser immediately
  python visualize_gguf.py model.gguf --show

  # Verbose mode with detailed layer info
  python visualize_gguf.py model.gguf -v
        """
    )

    parser.add_argument(
        'gguf_file',
        help='Path to the GGUF model file'
    )

    parser.add_argument(
        '-o', '--output',
        help='Output HTML filename (default: <model>_topology.html)',
        default=None
    )

    parser.add_argument(
        '--show',
        action='store_true',
        help='Open visualization in browser immediately'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed information about the model'
    )

    args = parser.parse_args()

    # Check file exists
    if not Path(args.gguf_file).exists():
        print(f"Error: File not found: {args.gguf_file}")
        sys.exit(1)

    try:
        output_file = visualize_gguf(
            args.gguf_file,
            output_file=args.output,
            show_browser=args.show,
            verbose=args.verbose
        )
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
