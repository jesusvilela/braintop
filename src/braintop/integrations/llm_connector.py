"""LLM connector for integrating with language models."""

from typing import Any, Dict, List, Optional
import numpy as np

from braintop.integrations.base import BaseIntegration


class LLMConnector(BaseIntegration):
    """Connector for integrating with Large Language Models.
    
    This class provides a flexible interface to connect braintop to various
    LLM providers (OpenAI, Anthropic, local models, etc.) and extract
    embeddings, attention patterns, and other topology-relevant data.
    """
    
    def __init__(
        self,
        provider: str = "mock",
        model_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the LLM connector.
        
        Args:
            provider: LLM provider name ('openai', 'anthropic', 'huggingface', 'mock')
            model_name: Specific model to use
            config: Additional configuration
        """
        super().__init__(config)
        self.provider = provider
        self.model_name = model_name or "default"
        self.client = None
    
    def connect(self) -> bool:
        """Establish connection to the LLM provider.
        
        Returns:
            True if connection successful
        """
        if self.provider == "mock":
            # Mock connection for demonstration
            self.is_connected = True
            return True
        
        # In a real implementation, this would initialize API clients
        # For now, we'll use a mock
        self.is_connected = True
        return True
    
    def disconnect(self) -> None:
        """Close connection to the LLM provider."""
        self.client = None
        self.is_connected = False
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from the LLM.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to LLM. Call connect() first.")
        
        # Mock implementation - returns random embeddings
        # In a real implementation, this would call the LLM API
        embedding_dim = self.config.get("embedding_dim", 768)
        embeddings = np.random.randn(len(texts), embedding_dim)
        
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the connected model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "is_connected": self.is_connected,
            "embedding_dim": self.config.get("embedding_dim", 768),
        }
    
    def get_attention_weights(
        self,
        text: str,
        layer: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """Get attention weights from the model.
        
        Args:
            text: Input text
            layer: Specific layer to extract attention from (None for all)
            
        Returns:
            Attention weight matrix or None if not available
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to LLM. Call connect() first.")
        
        # Mock implementation
        # In a real implementation, this would extract actual attention weights
        num_tokens = len(text.split())
        attention = np.random.random((num_tokens, num_tokens))
        
        # Make symmetric and normalize
        attention = (attention + attention.T) / 2
        attention = attention / attention.sum(axis=1, keepdims=True)
        
        return attention
    
    def extract_topology_data(self, texts: List[str]) -> Dict[str, Any]:
        """Extract topology-relevant data from the LLM.
        
        Args:
            texts: List of input texts
            
        Returns:
            Dictionary containing embeddings, relationships, etc.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to LLM. Call connect() first.")
        
        embeddings = self.get_embeddings(texts)
        
        # Compute pairwise similarities
        similarities = embeddings @ embeddings.T
        
        # Create relationship graph (edges for high similarity)
        threshold = self.config.get("similarity_threshold", 0.7)
        edges = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if similarities[i, j] > threshold:
                    edges.append({
                        "source": i,
                        "target": j,
                        "weight": float(similarities[i, j])
                    })
        
        return {
            "embeddings": embeddings,
            "similarities": similarities,
            "edges": edges,
            "texts": texts,
        }
    
    def create_topology_from_texts(
        self,
        texts: List[str],
        layer_id: str = "llm_layer"
    ) -> Dict[str, Any]:
        """Create a topology structure from input texts.
        
        Args:
            texts: List of input texts to visualize
            layer_id: ID for the created layer
            
        Returns:
            Dictionary with topology data ready for visualization
        """
        topology_data = self.extract_topology_data(texts)
        
        return {
            "layer_id": layer_id,
            "num_nodes": len(texts),
            "embeddings": topology_data["embeddings"],
            "labels": texts,
            "edges": topology_data["edges"],
            "metadata": {
                "provider": self.provider,
                "model": self.model_name,
            }
        }
