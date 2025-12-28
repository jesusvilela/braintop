"""GGUF file loader for visualizing LLM model architectures.

This module provides functionality to load and parse GGUF files,
extract model architecture information, and create topology visualizations
from the model structure.
"""

from typing import Any, Dict, List, Optional, Tuple
import struct
import numpy as np
from pathlib import Path

from braintop.integrations.base import BaseIntegration


# GGUF format constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

# GGUF metadata value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


class GGUFLoader(BaseIntegration):
    """Loader for GGUF model files.

    This class can parse GGUF files, extract model architecture metadata,
    and prepare data for topology visualization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the GGUF loader.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.file_path: Optional[Path] = None
        self.metadata: Dict[str, Any] = {}
        self.architecture: Dict[str, Any] = {}
        self.tensor_info: List[Dict[str, Any]] = []

    def connect(self) -> bool:
        """Establish connection (validate file exists).

        Returns:
            True if file is accessible
        """
        if self.file_path and self.file_path.exists():
            self.is_connected = True
            return True
        return False

    def disconnect(self) -> None:
        """Close connection."""
        self.is_connected = False
        self.metadata = {}
        self.architecture = {}
        self.tensor_info = []

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from the model.

        Note: This returns synthetic embeddings based on model architecture.
        For real embeddings, you would need to run inference with the model.

        Args:
            texts: List of text strings

        Returns:
            Array of synthetic embeddings
        """
        embedding_dim = self.architecture.get('embedding_dim', 768)
        embeddings = np.random.randn(len(texts), embedding_dim)
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            'file_path': str(self.file_path) if self.file_path else None,
            'architecture': self.architecture,
            'total_tensors': len(self.tensor_info),
            'total_parameters': sum(t['elements'] for t in self.tensor_info),
            'is_connected': self.is_connected,
        }

    def load_file(self, file_path: str) -> bool:
        """Load a GGUF file and extract metadata.

        Args:
            file_path: Path to the GGUF file

        Returns:
            True if loading successful
        """
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {file_path}")

        print(f"Loading GGUF file: {file_path}")
        print(f"File size: {self.file_path.stat().st_size / (1024**3):.2f} GB")

        try:
            with open(self.file_path, 'rb') as f:
                # Read header
                magic = struct.unpack('<I', f.read(4))[0]
                if magic != GGUF_MAGIC:
                    raise ValueError(f"Invalid GGUF file: magic number mismatch (got {magic:#x})")

                version = struct.unpack('<I', f.read(4))[0]
                print(f"GGUF version: {version}")

                # Read tensor count and metadata count
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                metadata_count = struct.unpack('<Q', f.read(8))[0]

                print(f"Metadata entries: {metadata_count}")
                print(f"Tensors: {tensor_count}")

                # Read metadata
                self.metadata = self._read_metadata(f, metadata_count)

                # Extract architecture info
                self._extract_architecture_info()

                # Read tensor info (without loading actual tensor data)
                self.tensor_info = self._read_tensor_info(f, tensor_count)

                self.is_connected = True
                return True

        except Exception as e:
            print(f"Error loading GGUF file: {e}")
            raise

    def _read_string(self, f) -> str:
        """Read a GGUF string from file."""
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8')

    def _read_metadata_value(self, f, value_type: int) -> Any:
        """Read a metadata value based on its type."""
        if value_type == GGUF_TYPE_UINT8:
            return struct.unpack('<B', f.read(1))[0]
        elif value_type == GGUF_TYPE_INT8:
            return struct.unpack('<b', f.read(1))[0]
        elif value_type == GGUF_TYPE_UINT16:
            return struct.unpack('<H', f.read(2))[0]
        elif value_type == GGUF_TYPE_INT16:
            return struct.unpack('<h', f.read(2))[0]
        elif value_type == GGUF_TYPE_UINT32:
            return struct.unpack('<I', f.read(4))[0]
        elif value_type == GGUF_TYPE_INT32:
            return struct.unpack('<i', f.read(4))[0]
        elif value_type == GGUF_TYPE_FLOAT32:
            return struct.unpack('<f', f.read(4))[0]
        elif value_type == GGUF_TYPE_UINT64:
            return struct.unpack('<Q', f.read(8))[0]
        elif value_type == GGUF_TYPE_INT64:
            return struct.unpack('<q', f.read(8))[0]
        elif value_type == GGUF_TYPE_FLOAT64:
            return struct.unpack('<d', f.read(8))[0]
        elif value_type == GGUF_TYPE_BOOL:
            return struct.unpack('<?', f.read(1))[0]
        elif value_type == GGUF_TYPE_STRING:
            return self._read_string(f)
        elif value_type == GGUF_TYPE_ARRAY:
            array_type = struct.unpack('<I', f.read(4))[0]
            array_len = struct.unpack('<Q', f.read(8))[0]
            return [self._read_metadata_value(f, array_type) for _ in range(array_len)]
        else:
            raise ValueError(f"Unknown metadata type: {value_type}")

    def _read_metadata(self, f, count: int) -> Dict[str, Any]:
        """Read all metadata key-value pairs."""
        metadata = {}

        for _ in range(count):
            # Read key
            key = self._read_string(f)

            # Read value type
            value_type = struct.unpack('<I', f.read(4))[0]

            # Read value
            value = self._read_metadata_value(f, value_type)

            metadata[key] = value

        return metadata

    def _read_tensor_info(self, f, count: int) -> List[Dict[str, Any]]:
        """Read tensor information without loading tensor data."""
        tensors = []

        for i in range(count):
            # Read tensor name
            name = self._read_string(f)

            # Read number of dimensions
            n_dims = struct.unpack('<I', f.read(4))[0]

            # Read dimension sizes
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]

            # Read tensor type
            tensor_type = struct.unpack('<I', f.read(4))[0]

            # Read tensor offset
            offset = struct.unpack('<Q', f.read(8))[0]

            # Calculate tensor size
            total_elements = np.prod(dims) if dims else 0

            tensors.append({
                'name': name,
                'dims': dims,
                'type': tensor_type,
                'offset': offset,
                'elements': total_elements,
                'shape_str': 'x'.join(map(str, dims))
            })

        return tensors

    def _extract_architecture_info(self) -> None:
        """Extract architecture information from metadata."""
        arch = {}

        # Common architecture keys
        for key, value in self.metadata.items():
            # General architecture info
            if 'general.architecture' in key:
                arch['type'] = value
            elif 'general.name' in key:
                arch['name'] = value
            elif 'general.file_type' in key:
                arch['file_type'] = value

            # Model-specific architecture
            elif '.block_count' in key:
                arch['num_layers'] = value
            elif '.embedding_length' in key or '.embed_length' in key:
                arch['embedding_dim'] = value
            elif '.head_count' in key:
                arch['num_heads'] = value
            elif '.head_count_kv' in key:
                arch['num_kv_heads'] = value
            elif '.feed_forward_length' in key:
                arch['ffn_dim'] = value
            elif '.context_length' in key:
                arch['context_length'] = value
            elif '.rope.dimension_count' in key:
                arch['rope_dim'] = value
            elif '.attention.layer_norm_rms_epsilon' in key:
                arch['layer_norm_eps'] = value
            elif 'vocab_size' in key:
                arch['vocab_size'] = value

        self.architecture = arch

    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get a summary of the model architecture.

        Returns:
            Dictionary with architecture details
        """
        return {
            'architecture': self.architecture,
            'total_tensors': len(self.tensor_info),
            'total_parameters': sum(t['elements'] for t in self.tensor_info),
            'metadata_keys': len(self.metadata),
        }

    def get_layer_info(self) -> List[Dict[str, Any]]:
        """Extract layer-wise information from tensors.

        Returns:
            List of layer information dictionaries
        """
        layers = {}

        # Group tensors by layer
        for tensor in self.tensor_info:
            name = tensor['name']

            # Parse layer information from tensor name
            # Common patterns: "blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight"
            layer_num = None
            layer_type = 'unknown'

            if 'blk.' in name:
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part == 'blk' and i + 1 < len(parts):
                        try:
                            layer_num = int(parts[i + 1])
                            break
                        except ValueError:
                            pass
            elif 'layers.' in name:
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        try:
                            layer_num = int(parts[i + 1])
                            break
                        except ValueError:
                            pass

            # Determine layer type
            if 'attn' in name or 'attention' in name:
                layer_type = 'attention'
            elif 'ffn' in name or 'mlp' in name or 'feed_forward' in name:
                layer_type = 'feedforward'
            elif 'norm' in name:
                layer_type = 'normalization'
            elif 'embed' in name or 'token' in name:
                layer_type = 'embedding'
            elif 'output' in name or 'lm_head' in name:
                layer_type = 'output'

            # Store layer info
            if layer_num is not None:
                if layer_num not in layers:
                    layers[layer_num] = {
                        'layer_num': layer_num,
                        'tensors': [],
                        'types': set(),
                        'total_params': 0
                    }

                layers[layer_num]['tensors'].append(tensor)
                layers[layer_num]['types'].add(layer_type)
                layers[layer_num]['total_params'] += tensor['elements']

        # Convert to sorted list
        return [
            {
                'layer_num': num,
                'types': list(info['types']),
                'num_tensors': len(info['tensors']),
                'total_params': info['total_params'],
                'tensors': info['tensors']
            }
            for num, info in sorted(layers.items())
        ]

    def create_topology_data(self) -> Dict[str, Any]:
        """Create topology visualization data from model architecture.

        Returns:
            Dictionary with topology data for visualization
        """
        if not self.is_connected:
            raise RuntimeError("GGUF file not loaded. Call load_file() first.")

        layer_info = self.get_layer_info()
        arch = self.architecture

        # Create layer groups for visualization
        topology_layers = []

        # Group 1: Embedding layer
        embedding_tensors = [t for t in self.tensor_info if 'embed' in t['name'].lower()]
        if embedding_tensors:
            topology_layers.append({
                'id': 'embeddings',
                'name': 'Token Embeddings',
                'type': 'conceptual',
                'num_nodes': len(embedding_tensors),
                'node_labels': [t['name'].split('.')[-2] if '.' in t['name'] else t['name']
                               for t in embedding_tensors],
                'metadata': {
                    'total_params': sum(t['elements'] for t in embedding_tensors),
                    'layer_type': 'embedding'
                }
            })

        # Group 2: Transformer blocks (attention + FFN)
        num_layers = arch.get('num_layers', len(layer_info))
        if num_layers > 0:
            # Sample layers for visualization (avoid too many)
            sample_layers = self._sample_layers(num_layers)

            for layer_idx in sample_layers:
                layer_data = next((l for l in layer_info if l['layer_num'] == layer_idx), None)

                if layer_data:
                    # Create attention layer
                    attn_tensors = [t for t in layer_data['tensors']
                                   if 'attn' in t['name'].lower()]
                    if attn_tensors:
                        topology_layers.append({
                            'id': f'layer_{layer_idx}_attn',
                            'name': f'Layer {layer_idx} Attention',
                            'type': 'sphere',
                            'num_nodes': len(attn_tensors),
                            'node_labels': [self._simplify_tensor_name(t['name'])
                                          for t in attn_tensors],
                            'metadata': {
                                'layer_num': layer_idx,
                                'total_params': sum(t['elements'] for t in attn_tensors),
                                'layer_type': 'attention'
                            }
                        })

                    # Create FFN layer
                    ffn_tensors = [t for t in layer_data['tensors']
                                  if any(k in t['name'].lower() for k in ['ffn', 'mlp', 'feed_forward'])]
                    if ffn_tensors:
                        topology_layers.append({
                            'id': f'layer_{layer_idx}_ffn',
                            'name': f'Layer {layer_idx} FFN',
                            'type': 'hyperbolic',
                            'num_nodes': len(ffn_tensors),
                            'node_labels': [self._simplify_tensor_name(t['name'])
                                          for t in ffn_tensors],
                            'metadata': {
                                'layer_num': layer_idx,
                                'total_params': sum(t['elements'] for t in ffn_tensors),
                                'layer_type': 'feedforward'
                            }
                        })

        # Group 3: Output layer
        output_tensors = [t for t in self.tensor_info
                         if any(k in t['name'].lower() for k in ['output', 'lm_head', 'head'])]
        if output_tensors:
            topology_layers.append({
                'id': 'output',
                'name': 'Output/LM Head',
                'type': 'conceptual',
                'num_nodes': len(output_tensors),
                'node_labels': [self._simplify_tensor_name(t['name']) for t in output_tensors],
                'metadata': {
                    'total_params': sum(t['elements'] for t in output_tensors),
                    'layer_type': 'output'
                }
            })

        return {
            'layers': topology_layers,
            'architecture': arch,
            'total_params': sum(t['elements'] for t in self.tensor_info),
            'file_info': {
                'path': str(self.file_path),
                'size_gb': self.file_path.stat().st_size / (1024**3)
            }
        }

    def _sample_layers(self, num_layers: int, max_layers: int = 6) -> List[int]:
        """Sample layer indices for visualization.

        Args:
            num_layers: Total number of layers
            max_layers: Maximum layers to show

        Returns:
            List of layer indices to visualize
        """
        if num_layers <= max_layers:
            return list(range(num_layers))

        # Sample first, middle, and last layers
        indices = [0, 1]  # First two

        # Middle layers
        middle = num_layers // 2
        indices.extend([middle - 1, middle, middle + 1])

        # Last layer
        indices.append(num_layers - 1)

        return sorted(set(i for i in indices if 0 <= i < num_layers))

    def _simplify_tensor_name(self, name: str) -> str:
        """Simplify tensor name for display.

        Args:
            name: Full tensor name

        Returns:
            Simplified name
        """
        # Remove common prefixes
        name = name.replace('model.', '').replace('transformer.', '')

        # Get last meaningful parts
        parts = name.split('.')
        if len(parts) > 2:
            # Keep layer info and component name
            return '.'.join(parts[-2:])

        return name

    def print_summary(self) -> None:
        """Print a summary of the loaded model."""
        if not self.is_connected:
            print("No GGUF file loaded")
            return

        print("\n" + "="*80)
        print("GGUF MODEL SUMMARY")
        print("="*80)

        print(f"\nFile: {self.file_path.name}")
        print(f"Size: {self.file_path.stat().st_size / (1024**3):.2f} GB")

        print("\nArchitecture:")
        for key, value in self.architecture.items():
            print(f"  {key}: {value}")

        total_params = sum(t['elements'] for t in self.tensor_info)
        print(f"\nTotal Parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
        print(f"Total Tensors: {len(self.tensor_info)}")

        layer_info = self.get_layer_info()
        if layer_info:
            print(f"\nTransformer Layers: {len(layer_info)}")
            print("\nSample Layer Info:")
            for layer in layer_info[:3]:  # Show first 3 layers
                print(f"  Layer {layer['layer_num']}: {layer['num_tensors']} tensors, "
                      f"{layer['total_params']:,} params, types: {layer['types']}")

        print("\n" + "="*80)
