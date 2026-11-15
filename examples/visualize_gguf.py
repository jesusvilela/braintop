#!/usr/bin/env python3
"""Visualize the tensor layout of a GGUF model file.

Equivalent to ``braintop gguf MODEL.gguf -o OUTPUT.html``; shown here as a
script to illustrate the Python API.

Usage:
    python examples/visualize_gguf.py MODEL.gguf [-o OUTPUT.html] [--max-blocks N]
"""

import argparse
from pathlib import Path

from braintop.core.visualizer import TopologyVisualizer
from braintop.integrations import GGUFLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a GGUF model file")
    parser.add_argument("gguf_file")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--max-blocks", type=int, default=None)
    args = parser.parse_args()

    loader = GGUFLoader()
    loader.load_file(args.gguf_file)
    print(loader.format_summary())

    topology = loader.build_topology(max_blocks=args.max_blocks)
    visualizer = TopologyVisualizer(topology)
    # Node size reflects each tensor's share of the largest tensor's parameters.
    visualizer.create_figure(color_by="layer", size_by="param_share")

    output = args.output or f"{Path(args.gguf_file).stem}_topology.html"
    visualizer.save(output)
    print(f"Saved visualization to {output}")


if __name__ == "__main__":
    main()
