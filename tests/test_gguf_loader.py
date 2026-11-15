"""Tests for the GGUF loader."""

import collections
import struct

import pytest

from braintop.cli import main as cli_main
from braintop.integrations import GGUFFormatError, GGUFLoader
from braintop.integrations.gguf_loader import (
    ATTENTION,
    EMBEDDING,
    FEEDFORWARD,
    NORMALIZATION,
    OTHER,
    OUTPUT,
    classify_tensor,
    write_demo_gguf,
    write_gguf,
)

N_BLOCKS, DIM, FFN, VOCAB = 3, 8, 16, 32


@pytest.fixture()
def demo(tmp_path):
    path = write_demo_gguf(
        tmp_path / "demo.gguf",
        n_blocks=N_BLOCKS, embedding_dim=DIM, ffn_dim=FFN, vocab_size=VOCAB,
        n_heads=8, n_kv_heads=4,
    )
    loader = GGUFLoader()
    assert loader.load_file(path) is True
    return loader


@pytest.mark.parametrize("name,expected", [
    ("token_embd.weight", (None, None, EMBEDDING)),
    ("position_embd.weight", (None, None, EMBEDDING)),
    ("blk.0.attn_q.weight", (None, 0, ATTENTION)),
    ("blk.11.attn_output.weight", (None, 11, ATTENTION)),
    ("blk.2.attn_norm.weight", (None, 2, NORMALIZATION)),
    ("blk.2.ffn_norm.weight", (None, 2, NORMALIZATION)),
    ("blk.2.ffn_gate.weight", (None, 2, FEEDFORWARD)),
    ("blk.2.ffn_down.bias", (None, 2, FEEDFORWARD)),
    ("enc.blk.4.attn_k.weight", ("enc", 4, ATTENTION)),
    ("model.layers.3.mlp.fc1.weight", ("model", 3, FEEDFORWARD)),
    ("output_norm.weight", (None, None, NORMALIZATION)),
    ("output.weight", (None, None, OUTPUT)),
    ("rope_freqs.weight", (None, None, OTHER)),
])
def test_classify_tensor(name, expected):
    assert classify_tensor(name) == expected


def test_architecture_fields_use_exact_keys(demo):
    arch = demo.architecture
    assert arch["type"] == "llama"
    assert arch["name"] == "braintop-demo"
    assert arch["num_layers"] == N_BLOCKS
    assert arch["embedding_dim"] == DIM
    assert arch["ffn_dim"] == FFN
    # head_count must not be overwritten by head_count_kv
    assert arch["num_heads"] == 8
    assert arch["num_kv_heads"] == 4
    assert arch["vocab_size"] == VOCAB
    assert arch["layer_norm_rms_eps"] == pytest.approx(1e-5)


def test_tensor_directory_and_parameter_count(demo):
    assert len(demo.tensor_info) == 1 + 9 * N_BLOCKS + 2
    per_block = 2 * DIM + 4 * DIM * DIM + 3 * DIM * FFN
    expected = 2 * DIM * VOCAB + DIM + N_BLOCKS * per_block
    assert demo.total_parameters == expected
    assert all(isinstance(t["elements"], int) for t in demo.tensor_info)
    assert {t["dtype"] for t in demo.tensor_info} == {"F32"}


def test_layer_info_groups_by_block(demo):
    blocks = demo.get_layer_info()
    assert [b["layer_num"] for b in blocks] == list(range(N_BLOCKS))
    for block in blocks:
        assert block["num_tensors"] == 9
        assert block["types"] == [NORMALIZATION, ATTENTION, FEEDFORWARD]
        assert block["params_by_component"][NORMALIZATION] == 2 * DIM
        assert block["params_by_component"][ATTENTION] == 4 * DIM * DIM


def test_matches_official_reader(demo):
    gguf = pytest.importorskip("gguf")
    reader = gguf.GGUFReader(str(demo.file_path))
    assert [t.name for t in reader.tensors] == [t["name"] for t in demo.tensor_info]
    assert [[int(d) for d in t.shape] for t in reader.tensors] == [
        t["dims"] for t in demo.tensor_info
    ]
    assert [int(t.data_offset) for t in reader.tensors] == [
        demo.data_offset + t["offset"] for t in demo.tensor_info
    ]


def test_topology_structure(demo):
    topology = demo.build_topology()
    assert [layer.name for layer in topology.layers] == [
        "Embedding", "Block 0", "Block 1", "Block 2", "Output",
    ]
    nodes = topology.get_all_nodes()
    # One hub per stage plus one node per tensor, each tensor exactly once.
    assert len(nodes) == len(topology.layers) + len(demo.tensor_info)
    tensor_names = [n.metadata["tensor"] for n in nodes if "tensor" in n.metadata]
    assert sorted(tensor_names) == sorted(t["name"] for t in demo.tensor_info)

    output = topology.get_layer("output")
    assert sorted(n.metadata["tensor"] for n in output.nodes if "tensor" in n.metadata) == [
        "output.weight", "output_norm.weight",
    ]

    # Only stage hubs carry a visible label; tensors are identified on hover.
    assert [n.label for n in nodes if n.label] == [layer.name for layer in topology.layers]
    assert all(n.metadata["short_name"] for n in nodes if "tensor" in n.metadata)

    node_ids = {n.id for n in nodes}
    assert len(node_ids) == len(nodes)
    pairs = collections.Counter((e.source, e.target) for e in topology.inter_layer_edges)
    assert max(pairs.values()) == 1
    assert all(e.source in node_ids and e.target in node_ids
               for e in topology.inter_layer_edges)
    assert len(topology.inter_layer_edges) == len(demo.tensor_info) + len(topology.layers) - 1


def test_topology_is_deterministic_and_depth_ordered(demo):
    first, second = demo.build_topology(), demo.build_topology()
    assert [n.position for n in first.get_all_nodes()] == [
        n.position for n in second.get_all_nodes()
    ]
    heights = [layer.nodes[0].position[2] for layer in first.layers]
    assert heights == sorted(heights)
    assert len(set(heights)) == len(heights)
    for layer in first.layers:
        assert len({n.position[2] for n in layer.nodes}) == 1
    positions = [n.position for n in first.get_all_nodes()]
    assert len(set(positions)) == len(positions)


def test_max_blocks_samples_and_records_gaps(tmp_path):
    path = write_demo_gguf(tmp_path / "deep.gguf", n_blocks=10, embedding_dim=4,
                           ffn_dim=8, vocab_size=8)
    loader = GGUFLoader()
    loader.load_file(path)
    topology = loader.build_topology(max_blocks=4)

    names = [layer.name for layer in topology.layers]
    assert names == ["Embedding", "Block 0", "Block 3", "Block 6", "Block 9", "Output"]
    assert topology.metadata["blocks_in_file"] == 10
    assert topology.metadata["blocks_shown"] == 4
    # Heights follow the true block index, so the gaps stay visible.
    assert [layer.config["depth"] for layer in topology.layers] == [-1, 0, 3, 6, 9, 10]
    gaps = [e.metadata["omitted_blocks"] for e in topology.inter_layer_edges
            if e.metadata["kind"] == "sequence"]
    assert gaps == [0, 2, 2, 2, 0]

    with pytest.raises(ValueError):
        loader.build_topology(max_blocks=1)


def test_rejects_non_gguf_and_unsupported_files(tmp_path):
    loader = GGUFLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_file(tmp_path / "missing.gguf")

    bad_magic = tmp_path / "bad.gguf"
    bad_magic.write_bytes(b"NOPE" + b"\x00" * 32)
    with pytest.raises(GGUFFormatError, match="Not a GGUF file"):
        loader.load_file(bad_magic)

    old = tmp_path / "v1.gguf"
    old.write_bytes(struct.pack("<IIQQ", 0x46554747, 1, 0, 0))
    with pytest.raises(GGUFFormatError, match="Unsupported GGUF version 1"):
        loader.load_file(old)

    big_endian = tmp_path / "be.gguf"
    big_endian.write_bytes(b"GGUF" + struct.pack(">IQQ", 3, 0, 0))
    with pytest.raises(GGUFFormatError, match="Big-endian"):
        loader.load_file(big_endian)

    assert loader.is_connected is False


def test_rejects_truncated_file(tmp_path):
    path = write_demo_gguf(tmp_path / "ok.gguf", n_blocks=1, embedding_dim=4,
                           ffn_dim=8, vocab_size=8)
    truncated = tmp_path / "cut.gguf"
    truncated.write_bytes(path.read_bytes()[:200])
    with pytest.raises(GGUFFormatError):
        GGUFLoader().load_file(truncated)


def test_requires_loaded_file_and_has_no_embeddings(demo):
    with pytest.raises(RuntimeError):
        GGUFLoader().build_topology()
    with pytest.raises(NotImplementedError):
        demo.get_embeddings(["text"])
    info = demo.get_model_info()
    assert info["total_tensors"] == len(demo.tensor_info)
    assert "Parameters by role" in demo.format_summary()


def test_metadata_value_types_round_trip(tmp_path):
    path = write_gguf(
        tmp_path / "meta.gguf",
        {"general.architecture": "x", "a.flag": True, "a.count": 7, "a.ratio": 0.5,
         "a.names": ["p", "q"]},
        [("w", [2, 3])],
    )
    loader = GGUFLoader()
    loader.load_file(path)
    assert loader.metadata["a.flag"] is True
    assert loader.metadata["a.count"] == 7
    assert loader.metadata["a.ratio"] == 0.5
    assert loader.metadata["a.names"] == ["p", "q"]
    assert loader.tensor_info[0]["elements"] == 6


def test_cli_gguf_command(demo, tmp_path, capsys):
    output = tmp_path / "out.html"
    assert cli_main(["gguf", str(demo.file_path), "-o", str(output)]) == 0
    assert output.stat().st_size > 0
    assert "braintop-demo" in capsys.readouterr().out

    assert cli_main(["gguf", str(demo.file_path), "--summary"]) == 0
    assert cli_main(["gguf", str(tmp_path / "missing.gguf")]) == 1


def test_encoder_blocks_precede_decoder_blocks(tmp_path):
    tensors = [
        ("token_embd.weight", [4, 8]),
        ("dec.blk.0.attn_q.weight", [4, 4]),
        ("dec.blk.0.cross_attn_q.weight", [4, 4]),
        ("dec.blk.1.ffn_up.weight", [4, 8]),
        ("enc.blk.0.attn_q.weight", [4, 4]),
        ("enc.blk.1.ffn_up.weight", [4, 8]),
        ("enc.output_norm.weight", [4]),
    ]
    path = write_gguf(tmp_path / "t5.gguf", {"general.architecture": "t5"}, tensors)
    loader = GGUFLoader()
    loader.load_file(path)
    assert [(b["prefix"], b["layer_num"]) for b in loader.get_layer_info()] == [
        ("enc", 0), ("enc", 1), ("dec", 0), ("dec", 1),
    ]
    names = [layer.name for layer in loader.build_topology().layers]
    assert names == ["Embedding", "enc.blk.0", "enc.blk.1", "dec.blk.0", "dec.blk.1", "Output"]
