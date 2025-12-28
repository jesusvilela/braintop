#!/usr/bin/env python3
"""
Create a minimal demo GGUF file for testing visualization.

This creates a small, valid GGUF file that represents a tiny transformer model,
useful for testing the GGUF visualization without needing a large model file.
"""

import struct
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from braintop.integrations.gguf_loader import (
    GGUF_MAGIC, GGUF_VERSION,
    GGUF_TYPE_STRING, GGUF_TYPE_UINT32
)


def write_string(f, s: str):
    """Write a GGUF string."""
    encoded = s.encode('utf-8')
    f.write(struct.pack('<Q', len(encoded)))
    f.write(encoded)


def write_metadata_entry(f, key: str, value_type: int, value):
    """Write a metadata key-value pair."""
    write_string(f, key)
    f.write(struct.pack('<I', value_type))

    if value_type == GGUF_TYPE_STRING:
        write_string(f, value)
    elif value_type == GGUF_TYPE_UINT32:
        f.write(struct.pack('<I', value))


def create_demo_gguf(output_file: str = "demo_model.gguf"):
    """Create a minimal demo GGUF file.

    Args:
        output_file: Path to output GGUF file
    """
    print(f"Creating demo GGUF file: {output_file}")

    with open(output_file, 'wb') as f:
        # Write header
        f.write(struct.pack('<I', GGUF_MAGIC))  # Magic number
        f.write(struct.pack('<I', GGUF_VERSION))  # Version

        # Define metadata
        metadata = [
            ("general.architecture", GGUF_TYPE_STRING, "llama"),
            ("general.name", GGUF_TYPE_STRING, "DemoModel-7B"),
            ("llama.block_count", GGUF_TYPE_UINT32, 4),
            ("llama.embedding_length", GGUF_TYPE_UINT32, 768),
            ("llama.head_count", GGUF_TYPE_UINT32, 12),
            ("llama.head_count_kv", GGUF_TYPE_UINT32, 12),
            ("llama.feed_forward_length", GGUF_TYPE_UINT32, 3072),
            ("llama.context_length", GGUF_TYPE_UINT32, 2048),
            ("llama.vocab_size", GGUF_TYPE_UINT32, 32000),
        ]

        # Define tensors (minimal set representing a 4-layer transformer)
        tensors = []

        # Embedding layer
        tensors.append(("token_embd.weight", [32000, 768]))

        # 4 transformer blocks
        for i in range(4):
            # Attention
            tensors.append((f"blk.{i}.attn_q.weight", [768, 768]))
            tensors.append((f"blk.{i}.attn_k.weight", [768, 768]))
            tensors.append((f"blk.{i}.attn_v.weight", [768, 768]))
            tensors.append((f"blk.{i}.attn_output.weight", [768, 768]))

            # FFN
            tensors.append((f"blk.{i}.ffn_gate.weight", [3072, 768]))
            tensors.append((f"blk.{i}.ffn_up.weight", [3072, 768]))
            tensors.append((f"blk.{i}.ffn_down.weight", [768, 3072]))

            # Normalization
            tensors.append((f"blk.{i}.attn_norm.weight", [768]))
            tensors.append((f"blk.{i}.ffn_norm.weight", [768]))

        # Output layer
        tensors.append(("output.weight", [32000, 768]))

        # Write tensor count and metadata count
        f.write(struct.pack('<Q', len(tensors)))
        f.write(struct.pack('<Q', len(metadata)))

        # Write metadata
        for key, value_type, value in metadata:
            write_metadata_entry(f, key, value_type, value)

        # Write tensor info
        for name, dims in tensors:
            write_string(f, name)
            f.write(struct.pack('<I', len(dims)))  # n_dims

            for dim in dims:
                f.write(struct.pack('<Q', dim))

            f.write(struct.pack('<I', 0))  # tensor type (F32)
            f.write(struct.pack('<Q', 0))  # offset (dummy)

    file_size = Path(output_file).stat().st_size
    print(f"✓ Created {output_file} ({file_size / 1024:.1f} KB)")
    print(f"\nDemo model specs:")
    print(f"  - Architecture: LLaMA")
    print(f"  - Layers: 4")
    print(f"  - Embedding dim: 768")
    print(f"  - Attention heads: 12")
    print(f"  - FFN dim: 3072")
    print(f"  - Vocab size: 32,000")
    print(f"  - Total tensors: {len(tensors)}")

    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create a demo GGUF file for testing")
    parser.add_argument(
        '-o', '--output',
        default='demo_model.gguf',
        help='Output filename (default: demo_model.gguf)'
    )

    args = parser.parse_args()

    try:
        output_file = create_demo_gguf(args.output)
        print(f"\nTo visualize:")
        print(f"  python examples/visualize_gguf.py {output_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
