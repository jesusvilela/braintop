# GGUF Model Visualization

Braintop can read a GGUF model file (the format used by llama.cpp) and draw
its tensor layout as an interactive 3D figure.

## What it shows, and what it does not

The loader reads the file's header, metadata and tensor directory. It never
reads tensor data, so a multi-gigabyte model loads in well under a second.

The figure is an inventory of the file's tensors:

- One stage per part of the model: the embedding, each block, and the output.
- Stages are stacked along the vertical axis in depth order. Height is the
  block index, so order and depth can be read directly.
- Each stage has a hub node at its centre and one node per tensor on a ring
  around it, grouped by role (normalization, attention, feed-forward).
- Node size is the tensor's parameter count relative to the largest tensor.
- Only stage names are drawn as text. Hovering a tensor shows its full name,
  role, shape, storage type and parameter count.
- The layout is deterministic: the same file always gives the same figure.

It does not show weight values, activations, attention patterns or anything
else that requires running the model. Edges mean "this stage contains this
tensor" and "this stage follows that one"; they are not a computation graph.

## Usage

```bash
# Print a summary only
braintop gguf model.gguf --summary

# Write an interactive HTML figure
braintop gguf model.gguf -o model.html

# Draw at most 8 evenly spaced blocks of a deep model
braintop gguf model.gguf -o model.html --max-blocks 8
```

Without `-o` the figure opens in a browser. `-f png|jpg|svg` writes a static
image and needs the optional `kaleido` package.

With `--max-blocks`, the first and last blocks are always kept and the rest
are evenly spaced. Blocks keep their true height, so the gaps are visible,
and each bridging edge records how many blocks it skips (`omitted_blocks`).

### Trying it without a model

```bash
python examples/create_demo_gguf.py -o demo_model.gguf
braintop gguf demo_model.gguf -o demo_model.html
```

The demo file is a valid GGUF file with llama.cpp tensor names and
zero-filled weights. It is not a usable model.

### Python API

```python
from braintop.core.visualizer import TopologyVisualizer
from braintop.integrations import GGUFLoader

loader = GGUFLoader()
loader.load_file("model.gguf")

print(loader.format_summary())
loader.architecture        # {'type': 'llama', 'num_layers': ..., 'num_heads': ..., ...}
loader.metadata            # every metadata key/value in the file
loader.tensor_info         # name, dims, dtype, elements, block, component per tensor
loader.get_layer_info()    # tensors and parameter counts grouped by block

topology = loader.build_topology(max_blocks=None)
visualizer = TopologyVisualizer(topology)
visualizer.create_figure(color_by="layer", size_by="param_share")
visualizer.save("model.html")
```

`loader.get_embeddings()` raises `NotImplementedError`: the loader does not
run the model. Use `LlamaCppAdapter` for embeddings from a GGUF model.

## How tensors are classified

Roles are inferred from tensor names, following the names llama.cpp writes:

| Name pattern | Role |
|---|---|
| `blk.N.` prefix (also `layers.N.`, `h.N.`, `block.N.`) | belongs to block N |
| contains `norm` | normalization |
| contains `attn` or `attention` | attention |
| contains `ffn`, `mlp` or `feed_forward` | feed-forward |
| `token_embd`, `position_embd`, `token_types` (outside a block) | embedding |
| `output.*`, `lm_head.*` (outside a block) | output |
| anything else | other |

Inside a block, the norm rule is applied first, so `blk.0.attn_norm.weight`
is normalization and `blk.0.attn_output.weight` is attention. Tensors that
match no rule are still drawn, under "other"; nothing is dropped.

Architecture fields are read by exact key, `<architecture>.<field>`, for
example `llama.attention.head_count` and `llama.attention.head_count_kv`.

## Verification status

- The parser is cross-checked in the test suite against the official `gguf`
  Python reader (tensor names, dimensions and data offsets).
- One complete real model, `stories260K.gguf` from `ggml-org/models` (llama,
  5 blocks), was compared with the official reader: tensor count, parameter
  count and offsets matched.
- The headers of six further published models were read and every tensor
  received a role (none fell into "other"): GPT-2, BERT (`bge-small-en-v1.5`),
  Qwen2.5-0.5B, Phi-3-mini, Gemma-2-2B and Flan-T5-small (encoder and decoder
  blocks, drawn encoder first).
- The figure was rendered to an image and inspected for the llama and Qwen
  models.
- Architectures outside that list have not been tested. Models whose tensor
  names do not follow the patterns above will parse correctly but show
  tensors under "other".

## Limitations

- GGUF versions 2 and 3, little-endian only. Version 1 and big-endian files
  are rejected with `GGUFFormatError`.
- Parameter counts are element counts from the tensor directory; they do not
  account for quantization.
- Each edge is a separate Plotly trace, so a model with hundreds of tensors
  produces a heavy figure. Use `--max-blocks` for deep models.
- Every HTML export embeds the Plotly library and is about 5 MB.

## Reference

- [GGUF format specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
