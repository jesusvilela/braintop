"""GGUF file loader for visualizing LLM model architectures.

Reads the header, metadata and tensor directory of a GGUF file (the model
format used by llama.cpp) without loading any weights, classifies each tensor
by the role its name implies, and builds a :class:`~braintop.core.topology.Topology`
whose layout is deterministic and ordered by depth.

What the resulting picture shows is the *tensor inventory* of the file: which
tensors exist, which block they belong to, their shapes, types and sizes. It
does not show weight values, activations or attention patterns.

Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

import math
import re
import struct
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from braintop.core.topology import EdgeModel, LayerModel, LayerType, NodeModel, Topology
from braintop.integrations.base import BaseIntegration

GGUF_MAGIC = 0x46554747  # b"GGUF" read as a little-endian uint32
GGUF_VERSION = 3
SUPPORTED_VERSIONS = (2, 3)
DEFAULT_ALIGNMENT = 32

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

_SCALAR_FORMATS: Dict[int, str] = {
    GGUF_TYPE_UINT8: "B",
    GGUF_TYPE_INT8: "b",
    GGUF_TYPE_UINT16: "H",
    GGUF_TYPE_INT16: "h",
    GGUF_TYPE_UINT32: "I",
    GGUF_TYPE_INT32: "i",
    GGUF_TYPE_FLOAT32: "f",
    GGUF_TYPE_BOOL: "?",
    GGUF_TYPE_UINT64: "Q",
    GGUF_TYPE_INT64: "q",
    GGUF_TYPE_FLOAT64: "d",
}

# ggml tensor types, as numbered in the GGUF specification.
GGML_TYPE_NAMES: Dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
}

# Tensor roles
EMBEDDING = "embedding"
ATTENTION = "attention"
FEEDFORWARD = "feedforward"
NORMALIZATION = "normalization"
OUTPUT = "output"
OTHER = "other"
COMPONENT_ORDER = (EMBEDDING, NORMALIZATION, ATTENTION, FEEDFORWARD, OUTPUT, OTHER)

_BLOCK_RE = re.compile(r"^(?:(?P<prefix>.*?)\.)?(?:blk|layers?|h|block)\.(?P<index>\d+)\.(?P<rest>.+)$")
_NORM_RE = re.compile(r"norm|(?:^|[._])ln(?:[._\d]|$)")
_ATTN_RE = re.compile(r"attn|attention")
_FFN_RE = re.compile(r"ffn|mlp|feed_forward|(?:^|[._])fc(?:[._\d]|$)")
_EMBED_RE = re.compile(r"embd|embed|(?:^|\.)wte(?:\.|$)|(?:^|\.)wpe(?:\.|$)|token_types")
_OUTPUT_RE = re.compile(r"^(?:output|lm_head|cls)(?:\.|$)")


class GGUFFormatError(ValueError):
    """Raised when a file is not a GGUF file this loader can read."""


def classify_tensor(name: str) -> Tuple[Optional[str], Optional[int], str]:
    """Classify a tensor by its name.

    Args:
        name: Tensor name as stored in the GGUF file, e.g. ``blk.3.attn_q.weight``.

    Returns:
        ``(prefix, block_index, component)``. ``block_index`` is ``None`` for
        tensors outside any repeated block. ``prefix`` is the text before the
        block marker (``"enc"`` in ``enc.blk.0...``) or ``None``. ``component``
        is one of the module-level role constants.
    """
    match = _BLOCK_RE.match(name)
    if match:
        rest = match.group("rest")
        if _NORM_RE.search(rest):
            component = NORMALIZATION
        elif _ATTN_RE.search(rest):
            component = ATTENTION
        elif _FFN_RE.search(rest):
            component = FEEDFORWARD
        else:
            component = OTHER
        return match.group("prefix"), int(match.group("index")), component

    if _NORM_RE.search(name):
        return None, None, NORMALIZATION
    if _EMBED_RE.search(name):
        return None, None, EMBEDDING
    if _OUTPUT_RE.search(name):
        return None, None, OUTPUT
    return None, None, OTHER


class GGUFLoader(BaseIntegration):
    """Loader for GGUF model files.

    Parses metadata and the tensor directory only; tensor data is never read,
    so loading a multi-gigabyte model takes a fraction of a second.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the GGUF loader.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.file_path: Optional[Path] = None
        self.version: Optional[int] = None
        self.metadata: Dict[str, Any] = {}
        self.architecture: Dict[str, Any] = {}
        self.tensor_info: List[Dict[str, Any]] = []
        self.data_offset: Optional[int] = None

    # ------------------------------------------------------------------
    # BaseIntegration interface
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Report whether a file has been loaded successfully."""
        return self.is_connected

    def disconnect(self) -> None:
        """Forget the loaded file."""
        self.is_connected = False
        self.version = None
        self.metadata = {}
        self.architecture = {}
        self.tensor_info = []
        self.data_offset = None

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Not supported: this loader reads file structure, it does not run the model.

        Use :class:`braintop.integrations.LlamaCppAdapter` to obtain embeddings
        from a GGUF model.
        """
        raise NotImplementedError(
            "GGUFLoader does not run inference. Use LlamaCppAdapter to compute "
            "embeddings from a GGUF model."
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "file_path": str(self.file_path) if self.file_path else None,
            "gguf_version": self.version,
            "architecture": dict(self.architecture),
            "total_tensors": len(self.tensor_info),
            "total_parameters": self.total_parameters,
            "is_connected": self.is_connected,
        }

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def load_file(self, file_path: Union[str, Path]) -> bool:
        """Load a GGUF file's metadata and tensor directory.

        Args:
            file_path: Path to the GGUF file

        Returns:
            True on success

        Raises:
            FileNotFoundError: If the file does not exist
            GGUFFormatError: If the file is not a readable GGUF file
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"GGUF file not found: {file_path}")

        self.disconnect()
        self.file_path = path
        self._file_size = path.stat().st_size

        try:
            with open(path, "rb") as f:
                (magic,) = self._unpack(f, "I")
                if magic != GGUF_MAGIC:
                    raise GGUFFormatError(
                        f"Not a GGUF file: magic is {magic:#010x}, expected {GGUF_MAGIC:#010x}"
                    )
                (version,) = self._unpack(f, "I")
                if version not in SUPPORTED_VERSIONS:
                    if version & 0xFFFF == 0:
                        raise GGUFFormatError(
                            "Big-endian GGUF files are not supported"
                        )
                    raise GGUFFormatError(
                        f"Unsupported GGUF version {version}; "
                        f"supported versions are {SUPPORTED_VERSIONS}"
                    )
                tensor_count, metadata_count = self._unpack(f, "QQ")
                # Every entry needs at least a few bytes; reject absurd counts early.
                if tensor_count + metadata_count > self._file_size:
                    raise GGUFFormatError("Corrupt GGUF header: entry counts exceed file size")

                metadata = {}
                for _ in range(metadata_count):
                    key = self._read_string(f)
                    (value_type,) = self._unpack(f, "I")
                    metadata[key] = self._read_value(f, value_type)

                tensors = [self._read_tensor_entry(f) for _ in range(tensor_count)]

                alignment = int(metadata.get("general.alignment", DEFAULT_ALIGNMENT))
                position = f.tell()
                data_offset = position + (-position % alignment) if alignment > 0 else position
        except struct.error as exc:
            raise GGUFFormatError(f"Truncated or corrupt GGUF file: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise GGUFFormatError(f"Corrupt GGUF string: {exc}") from exc

        self.version = version
        self.metadata = metadata
        self.tensor_info = tensors
        self.data_offset = data_offset
        self.architecture = self._extract_architecture()
        self.is_connected = True
        return True

    @staticmethod
    def _unpack(f: BinaryIO, fmt: str) -> Tuple[Any, ...]:
        fmt = "<" + fmt
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

    def _read_string(self, f: BinaryIO) -> str:
        (length,) = self._unpack(f, "Q")
        if length > self._file_size:
            raise GGUFFormatError("Corrupt GGUF file: string length exceeds file size")
        data = f.read(length)
        if len(data) != length:
            raise GGUFFormatError("Truncated GGUF file: unexpected end of string")
        return data.decode("utf-8")

    def _read_value(self, f: BinaryIO, value_type: int) -> Any:
        fmt = _SCALAR_FORMATS.get(value_type)
        if fmt is not None:
            return self._unpack(f, fmt)[0]
        if value_type == GGUF_TYPE_STRING:
            return self._read_string(f)
        if value_type == GGUF_TYPE_ARRAY:
            (item_type,) = self._unpack(f, "I")
            (length,) = self._unpack(f, "Q")
            if length > self._file_size:
                raise GGUFFormatError("Corrupt GGUF file: array length exceeds file size")
            item_fmt = _SCALAR_FORMATS.get(item_type)
            if item_fmt is not None:
                return list(self._unpack(f, f"{length}{item_fmt}"))
            return [self._read_value(f, item_type) for _ in range(length)]
        raise GGUFFormatError(f"Unknown GGUF metadata value type: {value_type}")

    def _read_tensor_entry(self, f: BinaryIO) -> Dict[str, Any]:
        name = self._read_string(f)
        (n_dims,) = self._unpack(f, "I")
        if n_dims > 8:
            raise GGUFFormatError(f"Corrupt GGUF file: tensor {name!r} has {n_dims} dimensions")
        dims = [int(d) for d in self._unpack(f, f"{n_dims}Q")]
        tensor_type, offset = self._unpack(f, "IQ")
        prefix, block, component = classify_tensor(name)
        return {
            "name": name,
            # Dimensions are kept in the order the file stores them (ggml order,
            # i.e. reversed relative to the usual row-major convention).
            "dims": dims,
            "type": int(tensor_type),
            "dtype": GGML_TYPE_NAMES.get(int(tensor_type), f"type_{tensor_type}"),
            "offset": int(offset),
            "elements": math.prod(dims) if dims else 0,
            "shape_str": "x".join(map(str, dims)),
            "prefix": prefix,
            "block": block,
            "component": component,
        }

    def _extract_architecture(self) -> Dict[str, Any]:
        """Collect well-known architecture fields from metadata by exact key."""
        meta = self.metadata
        arch: Dict[str, Any] = {}
        arch_name = meta.get("general.architecture")
        if arch_name is not None:
            arch["type"] = arch_name
        for field, key in (("name", "general.name"), ("file_type", "general.file_type")):
            if key in meta:
                arch[field] = meta[key]

        if isinstance(arch_name, str):
            fields = (
                ("num_layers", "block_count"),
                ("embedding_dim", "embedding_length"),
                ("context_length", "context_length"),
                ("ffn_dim", "feed_forward_length"),
                ("num_heads", "attention.head_count"),
                ("num_kv_heads", "attention.head_count_kv"),
                ("rope_dim", "rope.dimension_count"),
                ("layer_norm_eps", "attention.layer_norm_epsilon"),
                ("layer_norm_rms_eps", "attention.layer_norm_rms_epsilon"),
                ("vocab_size", "vocab_size"),
            )
            for field, suffix in fields:
                key = f"{arch_name}.{suffix}"
                if key in meta:
                    arch[field] = meta[key]

        tokens = meta.get("tokenizer.ggml.tokens")
        if "vocab_size" not in arch and isinstance(tokens, list):
            arch["vocab_size"] = len(tokens)
        return arch

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    @property
    def total_parameters(self) -> int:
        """Total number of tensor elements in the file."""
        return sum(t["elements"] for t in self.tensor_info)

    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get a summary of the model architecture."""
        return {
            "architecture": dict(self.architecture),
            "total_tensors": len(self.tensor_info),
            "total_parameters": self.total_parameters,
            "metadata_keys": len(self.metadata),
        }

    def get_layer_info(self) -> List[Dict[str, Any]]:
        """Group block tensors by block.

        Returns:
            One entry per block, in file order of block prefix and then by block
            index, each with the block's tensors and per-component parameter counts.
        """
        blocks: Dict[Tuple[str, int], Dict[str, Any]] = {}
        prefix_order: Dict[str, int] = {}
        for tensor in self.tensor_info:
            if tensor["block"] is None:
                continue
            prefix = tensor["prefix"] or ""
            prefix_order.setdefault(prefix, len(prefix_order))
            entry = blocks.setdefault(
                (prefix, tensor["block"]),
                {
                    "prefix": prefix,
                    "layer_num": tensor["block"],
                    "tensors": [],
                    "total_params": 0,
                    "params_by_component": {},
                },
            )
            entry["tensors"].append(tensor)
            entry["total_params"] += tensor["elements"]
            by_component = entry["params_by_component"]
            by_component[tensor["component"]] = (
                by_component.get(tensor["component"], 0) + tensor["elements"]
            )

        ordered = sorted(blocks.items(), key=lambda kv: (prefix_order[kv[0][0]], kv[0][1]))
        result = []
        for _, entry in ordered:
            entry["num_tensors"] = len(entry["tensors"])
            entry["types"] = [c for c in COMPONENT_ORDER if c in entry["params_by_component"]]
            result.append(entry)
        return result

    def format_summary(self) -> str:
        """Return a human-readable summary of the loaded model."""
        if not self.is_connected or self.file_path is None:
            return "No GGUF file loaded"

        lines = [
            f"File: {self.file_path.name} ({self._file_size / 1024 ** 2:.2f} MiB, "
            f"GGUF v{self.version})",
            "Architecture:",
        ]
        lines.extend(f"  {key}: {value}" for key, value in self.architecture.items())
        lines.append(f"Tensors: {len(self.tensor_info)}")
        lines.append(f"Parameters: {self.total_parameters:,}")

        totals: Dict[str, int] = {}
        for tensor in self.tensor_info:
            totals[tensor["component"]] = totals.get(tensor["component"], 0) + tensor["elements"]
        lines.append("Parameters by role:")
        lines.extend(f"  {c}: {totals[c]:,}" for c in COMPONENT_ORDER if c in totals)

        blocks = self.get_layer_info()
        lines.append(f"Blocks found in tensor names: {len(blocks)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    def build_topology(
        self,
        max_blocks: Optional[int] = None,
        ring_radius: float = 2.0,
        block_spacing: float = 1.5,
    ) -> Topology:
        """Build a deterministic, depth-ordered topology of the file's tensors.

        Each stage of the model (embedding, every block, output) becomes one
        topology layer placed at a height proportional to its depth. A stage
        has a hub node at the centre and one node per tensor on a ring around
        it, grouped by role. Hubs of consecutive stages are joined by an edge.

        Args:
            max_blocks: If set and the model has more blocks, only this many
                evenly spaced blocks (always including the first and last) are
                drawn. Heights still reflect the true block index, and the edge
                bridging a gap records how many blocks were omitted.
            ring_radius: Radius of the tensor ring around each hub
            block_spacing: Vertical distance between consecutive stages

        Returns:
            The constructed topology
        """
        if not self.is_connected or self.file_path is None:
            raise RuntimeError("GGUF file not loaded. Call load_file() first.")
        if max_blocks is not None and max_blocks < 2:
            raise ValueError("max_blocks must be at least 2")

        blocks = self.get_layer_info()
        shown = _sample_indices(len(blocks), max_blocks)
        globals_by_role: Dict[str, List[Dict[str, Any]]] = {}
        for tensor in self.tensor_info:
            if tensor["block"] is None:
                globals_by_role.setdefault(tensor["component"], []).append(tensor)

        # Stages in depth order: (layer id, display name, depth, tensors)
        stages: List[Tuple[str, str, float, List[Dict[str, Any]]]] = []
        if EMBEDDING in globals_by_role:
            stages.append(("embedding", "Embedding", -1.0, globals_by_role[EMBEDDING]))
        for position in shown:
            block = blocks[position]
            label = f"{block['prefix']}.blk.{block['layer_num']}" if block["prefix"] else (
                f"Block {block['layer_num']}"
            )
            stages.append((f"block_{position}", label, float(position), block["tensors"]))
        tail = globals_by_role.get(NORMALIZATION, []) + globals_by_role.get(OUTPUT, [])
        if tail:
            stages.append(("output", "Output", float(len(blocks)), tail))
        if OTHER in globals_by_role:
            stages.append(("other", "Other tensors", float(len(blocks)) + 1.0,
                           globals_by_role[OTHER]))

        model_name = str(self.architecture.get("name") or self.file_path.stem)
        topology = Topology(id=f"gguf_{self.file_path.stem}", name=f"{model_name} (GGUF)")
        largest = max((t["elements"] for t in self.tensor_info), default=0) or 1
        edge_count = 0
        previous_hub: Optional[str] = None
        previous_depth: Optional[float] = None

        for layer_id, name, depth, tensors in stages:
            z = depth * block_spacing
            stage_params = sum(t["elements"] for t in tensors)
            hub = NodeModel(
                id=f"{layer_id}_hub",
                position=(0.0, 0.0, z),
                label=name,
                layer_id=layer_id,
                metadata={
                    "role": "stage",
                    "tensors": len(tensors),
                    "parameters": stage_params,
                    "param_share": 0.0,
                },
            )
            nodes = [hub]
            ordered = sorted(
                tensors, key=lambda t: (COMPONENT_ORDER.index(t["component"]), t["name"])
            )
            for index, tensor in enumerate(ordered):
                angle = 2.0 * math.pi * index / len(ordered)
                node = NodeModel(
                    id=f"{layer_id}_t{index}",
                    position=(
                        ring_radius * math.cos(angle),
                        ring_radius * math.sin(angle),
                        z,
                    ),
                    label=_short_name(tensor["name"]),
                    layer_id=layer_id,
                    metadata={
                        "tensor": tensor["name"],
                        "role": tensor["component"],
                        "shape": tensor["shape_str"],
                        "dtype": tensor["dtype"],
                        "parameters": tensor["elements"],
                        "param_share": tensor["elements"] / largest,
                    },
                )
                nodes.append(node)
                topology.add_inter_layer_edge(EdgeModel(
                    id=f"edge_{edge_count}",
                    source=hub.id,
                    target=node.id,
                    weight=tensor["elements"] / largest,
                    metadata={"kind": "contains"},
                ))
                edge_count += 1

            topology.add_layer(LayerModel(
                id=layer_id,
                name=name,
                layer_type=LayerType.CUSTOM,
                nodes=nodes,
                config={"depth": depth, "parameters": stage_params},
            ))

            if previous_hub is not None and previous_depth is not None and layer_id != "other":
                omitted = 0
                if layer_id.startswith("block_") or layer_id == "output":
                    omitted = max(int(depth - previous_depth) - 1, 0)
                topology.add_inter_layer_edge(EdgeModel(
                    id=f"edge_{edge_count}",
                    source=previous_hub,
                    target=hub.id,
                    weight=1.0,
                    metadata={"kind": "sequence", "omitted_blocks": omitted},
                ))
                edge_count += 1
            if layer_id != "other":
                previous_hub, previous_depth = hub.id, depth

        topology.metadata.update({
            "source_file": str(self.file_path),
            "gguf_version": self.version,
            "architecture": {k: _plain(v) for k, v in self.architecture.items()},
            "total_parameters": self.total_parameters,
            "total_tensors": len(self.tensor_info),
            "blocks_in_file": len(blocks),
            "blocks_shown": len(shown),
        })
        return topology


def _sample_indices(count: int, limit: Optional[int]) -> List[int]:
    """Pick up to ``limit`` evenly spaced indices from ``range(count)``."""
    if limit is None or count <= limit:
        return list(range(count))
    return sorted({round(i * (count - 1) / (limit - 1)) for i in range(limit)})


def _short_name(name: str) -> str:
    """Drop the block prefix from a tensor name for display."""
    match = _BLOCK_RE.match(name)
    return match.group("rest") if match else name


def _plain(value: Any) -> Any:
    """Convert numpy scalars to plain Python values for JSON export."""
    return value.item() if isinstance(value, np.generic) else value


# ----------------------------------------------------------------------
# Minimal writer, used for demos and tests
# ----------------------------------------------------------------------

def _pack_string(text: str) -> bytes:
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _pack_value(value: Any) -> bytes:
    if isinstance(value, bool):
        return struct.pack("<I?", GGUF_TYPE_BOOL, value)
    if isinstance(value, int):
        return struct.pack("<II", GGUF_TYPE_UINT32, value)
    if isinstance(value, float):
        return struct.pack("<If", GGUF_TYPE_FLOAT32, value)
    if isinstance(value, str):
        return struct.pack("<I", GGUF_TYPE_STRING) + _pack_string(value)
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        body = b"".join(_pack_string(v) for v in value)
        return struct.pack("<IIQ", GGUF_TYPE_ARRAY, GGUF_TYPE_STRING, len(value)) + body
    raise TypeError(f"Unsupported metadata value: {value!r}")


def write_gguf(
    path: Union[str, Path],
    metadata: Dict[str, Any],
    tensors: Sequence[Tuple[str, Sequence[int]]],
) -> Path:
    """Write a small, valid GGUF v3 file with zero-filled F32 tensors.

    Intended for demos and tests; it supports only bool, uint32, float32,
    string and string-array metadata.

    Args:
        path: Output path
        metadata: Metadata key/value pairs
        tensors: ``(name, dims)`` pairs, dims in the order GGUF stores them

    Returns:
        The output path
    """
    path = Path(path)
    alignment = DEFAULT_ALIGNMENT
    header = struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(tensors), len(metadata))
    body = b"".join(_pack_string(key) + _pack_value(value) for key, value in metadata.items())

    directory = b""
    offset = 0
    sizes = []
    for name, dims in tensors:
        size = 4 * math.prod(dims)
        directory += _pack_string(name) + struct.pack("<I", len(dims))
        directory += struct.pack(f"<{len(dims)}Q", *dims)
        directory += struct.pack("<IQ", 0, offset)
        sizes.append(size)
        offset += size + (-size % alignment)

    prefix = header + body + directory
    with open(path, "wb") as f:
        f.write(prefix)
        f.write(b"\x00" * (-len(prefix) % alignment))
        for size in sizes:
            f.write(b"\x00" * (size + (-size % alignment)))
    return path


def write_demo_gguf(
    path: Union[str, Path],
    n_blocks: int = 4,
    embedding_dim: int = 16,
    ffn_dim: int = 32,
    vocab_size: int = 64,
    n_heads: int = 4,
    n_kv_heads: int = 2,
) -> Path:
    """Write a tiny llama-style GGUF file using llama.cpp's tensor naming."""
    metadata = {
        "general.architecture": "llama",
        "general.name": "braintop-demo",
        "llama.block_count": n_blocks,
        "llama.embedding_length": embedding_dim,
        "llama.feed_forward_length": ffn_dim,
        "llama.context_length": 128,
        "llama.attention.head_count": n_heads,
        "llama.attention.head_count_kv": n_kv_heads,
        "llama.attention.layer_norm_rms_epsilon": 1e-5,
        "tokenizer.ggml.model": "llama",
        "tokenizer.ggml.tokens": [f"<tok{i}>" for i in range(vocab_size)],
    }
    d, h = embedding_dim, ffn_dim
    tensors: List[Tuple[str, Sequence[int]]] = [("token_embd.weight", [d, vocab_size])]
    for i in range(n_blocks):
        tensors += [
            (f"blk.{i}.attn_norm.weight", [d]),
            (f"blk.{i}.attn_q.weight", [d, d]),
            (f"blk.{i}.attn_k.weight", [d, d]),
            (f"blk.{i}.attn_v.weight", [d, d]),
            (f"blk.{i}.attn_output.weight", [d, d]),
            (f"blk.{i}.ffn_norm.weight", [d]),
            (f"blk.{i}.ffn_gate.weight", [d, h]),
            (f"blk.{i}.ffn_up.weight", [d, h]),
            (f"blk.{i}.ffn_down.weight", [h, d]),
        ]
    tensors += [("output_norm.weight", [d]), ("output.weight", [d, vocab_size])]
    return write_gguf(path, metadata, tensors)
