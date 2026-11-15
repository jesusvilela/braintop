"""Tests for LLM integration."""

import pytest
import numpy as np

from braintop.integrations.llm_connector import LLMConnector


def test_llm_connector_creation():
    """Test creating LLMConnector."""
    connector = LLMConnector(provider="mock", model_name="test-model")

    assert connector.provider == "mock"
    assert connector.model_name == "test-model"
    assert not connector.is_connected


def test_llm_connector_connect():
    """Test connecting to LLM."""
    connector = LLMConnector(provider="mock")

    result = connector.connect()

    assert result is True
    assert connector.is_connected


def test_llm_connector_get_embeddings():
    """Test getting embeddings from LLM."""
    connector = LLMConnector(provider="mock")
    connector.connect()

    texts = ["Text 1", "Text 2", "Text 3"]
    embeddings = connector.get_embeddings(texts)

    assert embeddings.shape[0] == 3
    assert embeddings.shape[1] == 768  # Default embedding dimension


def test_llm_connector_get_embeddings_not_connected():
    """Test that getting embeddings fails when not connected."""
    connector = LLMConnector(provider="mock")

    with pytest.raises(RuntimeError):
        connector.get_embeddings(["text"])


def test_llm_connector_get_model_info():
    """Test getting model information."""
    connector = LLMConnector(provider="mock", model_name="test-model")
    connector.connect()

    info = connector.get_model_info()

    assert info["provider"] == "mock"
    assert info["model_name"] == "test-model"
    assert info["is_connected"] is True
    assert "embedding_dim" in info


def test_llm_connector_extract_topology_data():
    """Test extracting topology data."""
    connector = LLMConnector(provider="mock")
    connector.connect()

    texts = ["Text A", "Text B", "Text C"]
    data = connector.extract_topology_data(texts)

    assert "embeddings" in data
    assert "similarities" in data
    assert "edges" in data
    assert "texts" in data
    assert len(data["texts"]) == 3


def test_llm_connector_create_topology_from_texts():
    """Test creating topology from texts."""
    connector = LLMConnector(provider="mock")
    connector.connect()

    texts = ["Neural networks", "Deep learning", "Transformers"]
    topology_data = connector.create_topology_from_texts(texts, "test_layer")

    assert topology_data["layer_id"] == "test_layer"
    assert topology_data["num_nodes"] == 3
    assert "embeddings" in topology_data
    assert "labels" in topology_data
    assert topology_data["labels"] == texts


def test_llm_connector_disconnect():
    """Test disconnecting from LLM."""
    connector = LLMConnector(provider="mock")
    connector.connect()

    assert connector.is_connected

    connector.disconnect()

    assert not connector.is_connected


def test_mock_embeddings_are_deterministic_across_instances():
    """Eval deterministic offline embedding bridge behavior."""
    texts = ["attention routes context", "embeddings encode meaning"]
    first = LLMConnector(provider="mock", config={"embedding_dim": 32})
    second = LLMConnector(provider="mock", config={"embedding_dim": 32})
    first.connect()
    second.connect()

    assert np.allclose(first.get_embeddings(texts), second.get_embeddings(texts))


def test_custom_embedding_callable_is_used_and_normalized():
    """Eval custom client embedding bridge and normalization."""
    calls = []

    def embedding_fn(texts):
        calls.append(list(texts))
        return np.array([[3.0, 4.0], [0.0, 2.0]])

    connector = LLMConnector(
        provider="custom",
        model_name="custom-embedder",
        config={"embedding_fn": embedding_fn},
    )
    connector.connect()

    embeddings = connector.get_embeddings(["alpha", "beta"])

    assert calls == [["alpha", "beta"]]
    assert embeddings.shape == (2, 2)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)


def test_inference_bridge_normalizes_mock_and_custom_responses():
    """Eval normalized inference response schema."""
    mock = LLMConnector(provider="mock", model_name="mock-chat")
    mock.connect()

    mock_response = mock.run_inference("Summarize topology", context={"nodes": 3})

    assert mock_response["provider"] == "mock"
    assert mock_response["model"] == "mock-chat"
    assert "Summarize topology" in mock_response["text"]
    assert mock_response["metadata"]["context"] == {"nodes": 3}

    custom = LLMConnector(
        provider="custom",
        model_name="custom-chat",
        config={
            "inference_fn": lambda prompt, context=None: {"text": prompt.upper(), "score": 0.9}
        },
    )
    custom.connect()

    custom_response = custom.run_inference("bridge ready")

    assert custom_response["text"] == "BRIDGE READY"
    assert custom_response["metadata"]["raw"]["score"] == 0.9


def test_topology_extraction_applies_top_k_threshold_and_ordering():
    """Eval deterministic relationship extraction semantics."""
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ]
    )

    connector = LLMConnector(
        provider="custom",
        config={
            "embedding_fn": lambda texts: vectors,
            "similarity_threshold": 0.8,
            "top_k": 1,
        },
    )
    connector.connect()

    data = connector.extract_topology_data(["a", "b", "c", "d"])

    assert data["edges"] == [
        {"source": 0, "target": 1, "weight": pytest.approx(0.9938837346736189)},
        {"source": 2, "target": 3, "weight": pytest.approx(0.9938837346736189)},
    ]
    assert all(edge["source"] != edge["target"] for edge in data["edges"])


def test_attention_bridge_validates_and_row_normalizes():
    """Eval attention matrix production invariants."""
    connector = LLMConnector(
        provider="custom",
        config={"attention_fn": lambda text, layer=None: np.array([[2.0, 2.0], [0.0, 4.0]])},
    )
    connector.connect()

    attention = connector.get_attention_weights("hello world")

    assert attention.shape == (2, 2)
    assert np.allclose(attention.sum(axis=1), 1.0)
    assert np.all(attention >= 0)


def test_connector_rejects_invalid_text_batches():
    """Eval input validation for production bridge calls."""
    connector = LLMConnector(provider="mock")
    connector.connect()

    with pytest.raises(ValueError, match="at least one"):
        connector.get_embeddings([])

    with pytest.raises(ValueError, match="blank"):
        connector.get_embeddings(["valid", "  "])


def test_mock_embeddings_remain_normalizable_for_tiny_dimensions():
    """Review regression: hash cancellations must not create zero mock vectors."""
    connector = LLMConnector(provider="mock", config={"embedding_dim": 1})
    connector.connect()

    embeddings = connector.get_embeddings(["a a"])

    assert embeddings.shape == (1, 1)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)


def test_topology_extraction_rejects_nonfinite_threshold():
    """Review regression: non-finite thresholds are not mathematically meaningful."""
    connector = LLMConnector(
        provider="custom",
        config={
            "embedding_fn": lambda texts: np.eye(len(texts)),
            "similarity_threshold": np.inf,
        },
    )
    connector.connect()

    with pytest.raises(ValueError, match="similarity_threshold"):
        connector.extract_topology_data(["a", "b"])
