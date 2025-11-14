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
