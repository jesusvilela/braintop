# GGUF Model Visualization Guide

Braintop now supports visualizing **any** GGUF model file as an interactive 3D topology. This feature allows you to explore the architecture of large language models stored in the GGUF format.

## What is GGUF?

GGUF (GPT-Generated Unified Format) is a file format for storing large language models, commonly used by llama.cpp and similar inference engines. It contains model weights, architecture metadata, and tokenizer information in a single file.

## Features

- **Automatic Architecture Detection**: Extracts model structure from GGUF metadata
- **Layer Visualization**: Shows transformer blocks, attention mechanisms, and feed-forward networks
- **3D Geometric Mapping**: Different layer types use different geometries:
  - **Attention layers**: Spherical geometry (hierarchical relationships)
  - **Feed-Forward (FFN) layers**: Hyperbolic geometry (tree-like structures)
  - **Embedding/Output layers**: Conceptual geometry (semantic space)
- **Interactive Exploration**: Rotate, zoom, and inspect individual tensors
- **Model Statistics**: View parameter counts, layer information, and architecture details

## Quick Start

### 1. Basic Usage

```bash
python examples/visualize_gguf.py path/to/model.gguf
```

This will create an HTML visualization file: `model_topology.html`

### 2. Custom Output Filename

```bash
python examples/visualize_gguf.py model.gguf --output my_viz.html
```

### 3. Verbose Mode (Detailed Information)

```bash
python examples/visualize_gguf.py model.gguf --verbose
```

Shows:
- Complete metadata
- All layer details
- Tensor information
- Architecture summary

### 4. Open in Browser Immediately

```bash
python examples/visualize_gguf.py model.gguf --show
```

## Creating a Demo Model

If you don't have a GGUF file, you can create a small demo model:

```bash
# Create a minimal demo GGUF file
python examples/create_demo_gguf.py

# Visualize it
python examples/visualize_gguf.py demo_model.gguf
```

The demo model represents a 4-layer transformer with:
- 4 transformer blocks
- 768-dimensional embeddings
- 12 attention heads
- ~87M parameters

## Command-Line Options

```
python examples/visualize_gguf.py <gguf_file> [options]

Positional Arguments:
  gguf_file           Path to the GGUF model file

Optional Arguments:
  -o, --output FILE   Output HTML filename (default: <model>_topology.html)
  --show              Open visualization in browser immediately
  -v, --verbose       Print detailed model information
  -h, --help          Show help message
```

## Output Information

When you run the visualizer, you'll see:

### Step 1: Loading GGUF File
- File size
- GGUF version
- Number of tensors and metadata entries

### Step 2: Architecture Extraction
- Model type (LLaMA, Mistral, etc.)
- Number of layers identified
- Geometry types for each layer

### Step 3: Building Topology
- Layer creation progress
- Node counts

### Step 4: Connecting Layers
- Sequential connections (layer-to-layer)
- Attention-FFN connections (within blocks)

### Step 5: Visualization
- Total nodes and edges
- Layer count

### Step 6: Export
- Output file location and size

## Understanding the Visualization

### Color Coding

Different layers are color-coded in the 3D visualization:
- **Green**: Conceptual layers (embeddings, output)
- **Red**: Spherical layers (attention mechanisms)
- **Blue**: Hyperbolic layers (feed-forward networks)

### Node Labels

Hover over nodes to see:
- Tensor name
- Layer number
- Parameter count
- Layer type

### Connections

Edges represent:
- **Sequential flow**: Information flow through the network
- **Attention-FFN pairs**: Connections within transformer blocks

### Layer Sampling

For models with many layers (>6), the visualizer samples key layers:
- First two layers (0, 1)
- Middle layers
- Last layer

This prevents overcrowding while showing the overall architecture.

## Supported Model Types

The visualizer works with any GGUF file, but has been optimized for:

- **LLaMA** and variants (Alpaca, Vicuna, etc.)
- **Mistral**
- **Mixtral**
- **GPT-J**
- **MPT**
- **Falcon**

## Technical Details

### Architecture Detection

The loader extracts architecture from GGUF metadata keys:
- `general.architecture`: Model type
- `*.block_count`: Number of transformer layers
- `*.embedding_length`: Embedding dimensions
- `*.head_count`: Number of attention heads
- `*.feed_forward_length`: FFN dimensions
- `*.context_length`: Maximum context

### Tensor Parsing

Tensors are grouped by:
- **Layer number**: Extracted from tensor names (`blk.0`, `layers.0`, etc.)
- **Component type**: Attention, feed-forward, normalization, etc.

### Visualization Mapping

| Model Component | Topology Layer | Geometry | Rationale |
|----------------|----------------|----------|-----------|
| Token Embeddings | Conceptual | PCA projection | Semantic space |
| Attention | Sphere | Uniform sphere | Hierarchical attention |
| Feed-Forward | Hyperbolic | Poincaré disk | Tree-like expansion |
| Output/LM Head | Conceptual | PCA projection | Vocabulary space |

## Examples

### Example 1: Visualize a 7B Model

```bash
# Download a GGUF model (example)
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf

# Visualize
python examples/visualize_gguf.py llama-2-7b.Q4_K_M.gguf --verbose

# Output: llama-2-7b.Q4_K_M_topology.html
```

### Example 2: Compare Different Models

```bash
# Visualize multiple models
python examples/visualize_gguf.py model_a.gguf -o model_a.html
python examples/visualize_gguf.py model_b.gguf -o model_b.html

# Open both in browser to compare architectures
```

### Example 3: Programmatic Usage

```python
from braintop.integrations.gguf_loader import GGUFLoader
from braintop.utils.builders import TopologyBuilder
from braintop.core.visualizer import TopologyVisualizer

# Load GGUF file
loader = GGUFLoader()
loader.load_file("model.gguf")

# Get topology data
topology_data = loader.create_topology_data()

# Build and visualize
builder = TopologyBuilder("my_model", "My Model")
# ... (add layers from topology_data)

visualizer = TopologyVisualizer(builder.build())
visualizer.save("output.html")
```

## Limitations

### Current Limitations

1. **No Weight Visualization**: Only architecture is shown, not actual weight values
2. **No Inference**: Cannot run the model, only visualize structure
3. **Large Models**: Very large models (>100 layers) may produce crowded visualizations
4. **Memory**: Loading very large GGUF files requires sufficient RAM

### Future Enhancements

- Weight distribution visualization
- Attention pattern display
- Layer activation visualization
- Real-time inference integration
- Support for model comparison views

## Troubleshooting

### File Not Found

```
Error: File not found: model.gguf
```

**Solution**: Check the file path is correct.

### Invalid GGUF File

```
ValueError: Invalid GGUF file: magic number mismatch
```

**Solution**: Ensure the file is a valid GGUF file, not corrupted.

### Memory Error

```
MemoryError: Unable to allocate array
```

**Solution**: For very large models, the loader only reads metadata, not weights. If you still get memory errors, increase system RAM or use a smaller model.

### No Layers Detected

```
Identified 0 topology layers
```

**Solution**: The model may use an unsupported naming convention. Try verbose mode (`-v`) to see what tensors were found.

## Integration with Solution Plan

This GGUF visualization feature follows the same problem model and solution plan defined in:
- `PROBLEM_MODEL.md`: Formal definition of entities, actions, and constraints
- `SOLUTION_PLAN.md`: Step-by-step workflow for topology creation

The GGUF loader extracts real model data instead of synthetic data, but follows the same visualization pipeline.

## API Reference

### GGUFLoader Class

```python
class GGUFLoader(BaseIntegration):
    """Loader for GGUF model files."""

    def load_file(file_path: str) -> bool:
        """Load a GGUF file and extract metadata."""

    def get_architecture_summary() -> Dict[str, Any]:
        """Get model architecture summary."""

    def get_layer_info() -> List[Dict[str, Any]]:
        """Get layer-wise information."""

    def create_topology_data() -> Dict[str, Any]:
        """Create topology visualization data."""

    def print_summary() -> None:
        """Print model summary."""
```

### Visualization Functions

```python
def create_gguf_topology(gguf_file: str, verbose: bool = False) -> TopologyBuilder:
    """Create a topology from a GGUF file."""

def visualize_gguf(
    gguf_file: str,
    output_file: Optional[str] = None,
    show_browser: bool = False,
    verbose: bool = False
) -> str:
    """Create and save a visualization of a GGUF model."""
```

## Contributing

Found a model that doesn't visualize correctly? Please open an issue with:
- Model name and source
- GGUF file metadata (from `--verbose` mode)
- Expected vs actual layer detection

## References

- [GGUF Format Specification](https://github.com/ggerganov/llama.cpp/blob/master/docs/gguf.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [GGUF Models on Hugging Face](https://huggingface.co/models?library=gguf)
