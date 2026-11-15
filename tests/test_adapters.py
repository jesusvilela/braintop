"""Tests for packaged LLM provider adapters."""

from types import SimpleNamespace

import numpy as np
import pytest

from braintop.integrations import (
    CallableAdapter,
    LLMConnector,
    LlamaCppAdapter,
    OpenAIAdapter,
    SentenceTransformerAdapter,
)


class FakeSentenceTransformer:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.array([[1.0, 0.0], [0.0, 2.0]])


class FakeOpenAIEmbeddings:
    def create(self, model, input):
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[1.0, 0.0, 0.0]),
                SimpleNamespace(embedding=[0.0, 2.0, 0.0]),
            ]
        )


class FakeOpenAICompletions:
    def create(self, model, messages, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=messages[-1]["content"].upper()))
            ]
        )


class FakeOpenAIClient:
    def __init__(self):
        self.embeddings = FakeOpenAIEmbeddings()
        self.chat = SimpleNamespace(completions=FakeOpenAICompletions())


class FakeLlamaClient:
    def create_embedding(self, input):
        return {
            "data": [
                {"embedding": [1.0, 1.0]},
                {"embedding": [0.0, 2.0]},
            ]
        }

    def create_chat_completion(self, messages, **kwargs):
        return {"choices": [{"message": {"content": messages[-1]["content"][::-1]}}]}


def test_callable_adapter_wraps_packaged_callables():
    """CallableAdapter exposes embedding, inference, and attention hooks."""
    adapter = CallableAdapter(
        embedding_fn=lambda texts: np.ones((len(texts), 2)),
        inference_fn=lambda prompt, context=None: {"text": prompt.upper()},
        attention_fn=lambda text, layer=None: np.eye(len(text.split())),
        model_name="callable-model",
    )
    adapter.connect()

    assert adapter.get_embeddings(["a", "b"]).shape == (2, 2)
    assert adapter.run_inference("ok") == {"text": "OK"}
    assert adapter.get_attention_weights("a b").shape == (2, 2)
    assert adapter.get_model_info()["adapter"] == "callable"


def test_sentence_transformer_adapter_uses_prebuilt_client():
    """SentenceTransformerAdapter supports dependency-injected clients."""
    client = FakeSentenceTransformer()
    adapter = SentenceTransformerAdapter(
        model_name="mini",
        client=client,
        config={"encode_kwargs": {"batch_size": 8}},
    )
    adapter.connect()

    embeddings = adapter.get_embeddings(["alpha", "beta"])

    assert embeddings.shape == (2, 2)
    assert client.calls[0][0] == ["alpha", "beta"]
    assert client.calls[0][1]["convert_to_numpy"] is True
    assert client.calls[0][1]["batch_size"] == 8


def test_openai_adapter_supports_embeddings_and_inference_with_fake_client():
    """OpenAIAdapter parses OpenAI-compatible embedding and chat responses."""
    adapter = OpenAIAdapter(model_name="text-model", client=FakeOpenAIClient())
    adapter.connect()

    embeddings = adapter.get_embeddings(["alpha", "beta"])
    response = adapter.run_inference("hello")

    assert embeddings.shape == (2, 3)
    assert response == "HELLO"
    assert adapter.get_model_info()["adapter"] == "openai"


def test_llama_cpp_adapter_supports_embeddings_and_inference_with_fake_client():
    """LlamaCppAdapter parses llama.cpp-style dict responses."""
    adapter = LlamaCppAdapter(model_name="model.gguf", client=FakeLlamaClient())
    adapter.connect()

    embeddings = adapter.get_embeddings(["alpha", "beta"])
    response = adapter.run_inference("hello")

    assert embeddings.shape == (2, 2)
    assert response == "olleh"
    assert adapter.get_model_info()["adapter"] == "llama-cpp"


def test_llm_connector_auto_wraps_packaged_openai_adapter():
    """LLMConnector provider='openai' wraps prebuilt clients with OpenAIAdapter."""
    connector = LLMConnector(
        provider="openai",
        model_name="text-model",
        config={"client": FakeOpenAIClient(), "embedding_model": "embed-model"},
    )
    connector.connect()

    embeddings = connector.get_embeddings(["alpha", "beta"])
    inference = connector.run_inference("bridge")
    info = connector.get_model_info()

    assert embeddings.shape == (2, 3)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)
    assert inference["text"] == "BRIDGE"
    assert info["adapter"] == "openai"
    assert info["supports_inference"] is True



def test_non_mock_provider_never_falls_back_to_mock_embeddings():
    """A provider without embeddings must fail instead of returning synthetic vectors."""
    connector = LLMConnector(
        provider="custom",
        config={"inference_fn": lambda prompt, context=None: prompt},
    )
    connector.connect()

    with pytest.raises(NotImplementedError, match="embedding_fn configured"):
        connector.get_embeddings(["real input"])


def test_unsupported_provider_fails_closed_without_client():
    """Unknown providers must not report a successful mock-backed connection."""
    connector = LLMConnector(provider="anthropic")

    with pytest.raises(ValueError, match="Unsupported provider"):
        connector.connect()


def test_embedding_only_adapter_does_not_fake_inference():
    """An embedding-only adapter must not silently return mock inference text."""
    connector = LLMConnector(
        provider="sentence-transformers",
        model_name="test-encoder",
        config={"client": FakeSentenceTransformer()},
    )
    connector.connect()

    with pytest.raises(NotImplementedError, match="inference capability"):
        connector.run_inference("This should not be synthesized")


def test_embedding_only_callable_reports_its_actual_capabilities():
    """Do not confuse stub adapter methods with operational capabilities."""
    connector = LLMConnector(
        provider="custom",
        model_name="embedding-only",
        config={"embedding_fn": lambda texts: np.ones((len(texts), 3))},
    )
    connector.connect()
    info = connector.get_model_info()

    assert info["supports_embeddings"] is True
    assert info["supports_inference"] is False
    assert info["supports_attention"] is False
    assert info["embedding_dim"] is None

    with pytest.raises(NotImplementedError, match="inference_fn"):
        connector.run_inference("not configured")
    with pytest.raises(NotImplementedError, match="attention_fn"):
        connector.get_attention_weights("not configured")


def test_embedding_only_sentence_transformer_does_not_claim_attention():
    connector = LLMConnector(
        provider="sentence-transformers",
        model_name="tiny",
        config={"client": FakeSentenceTransformer()},
    )
    connector.connect()
    info = connector.get_model_info()
    assert info["supports_embeddings"] is True
    assert info["supports_inference"] is False
    assert info["supports_attention"] is False
    with pytest.raises(NotImplementedError, match="attention capability"):
        connector.get_attention_weights("hello there")
