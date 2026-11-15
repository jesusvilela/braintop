#!/usr/bin/env python3
"""Create a tiny, valid GGUF file for trying the GGUF visualizer.

The file uses llama.cpp's tensor naming and holds zero-filled F32 tensors, so
it is a few hundred kilobytes and can be read by any GGUF reader. It is not a
usable model.

Usage:
    python examples/create_demo_gguf.py [-o demo_model.gguf] [--blocks 4]
"""

import argparse

from braintop.integrations.gguf_loader import write_demo_gguf


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a demo GGUF file")
    parser.add_argument("-o", "--output", default="demo_model.gguf")
    parser.add_argument("--blocks", type=int, default=4, help="Number of blocks")
    args = parser.parse_args()

    path = write_demo_gguf(args.output, n_blocks=args.blocks)
    print(f"Created {path} ({path.stat().st_size / 1024:.1f} KiB)")
    print(f"Visualize it with: braintop gguf {path} -o demo_model.html")


if __name__ == "__main__":
    main()
