"""LLM connector for integrating with language models."""

from typing import Any, Callable, Dict, List, Optional
import hashlib
import re

import numpy as np

from braintop.integrations.base import BaseIntegration


class LLMConnector(BaseIntegration):
    """Connector for integrating with Large Language Models.

    This class provides a flexible interface to connect braintop to various
    LLM providers (OpenAI, Anthropic, local models, etc.) and extract
    embeddings, inference responses, attention patterns, and other
    topology-relevant data. The production bridge intentionally avoids direct
    vendor SDK imports; callers can pass provider clients or callables in
    ``config`` while tests and examples use deterministic ``mock`` behavior.
    """

    def __init__(
        self,
        provider: str = "mock",
        model_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the LLM connector.

        Args:
            provider: LLM provider name ('custom', 'openai', 'anthropic',
                'huggingface', 'mock', etc.)
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
        self.client = self.config.get("client")
        if self.provider == "custom" and self.client is None:
            self.client = {
                "embedding_fn": self.config.get("embedding_fn"),
                "inference_fn": self.config.get("inference_fn"),
                "attention_fn": self.config.get("attention_fn"),
            }

        # Vendor SDK construction is intentionally delegated to callers so the
        # core package remains lightweight and import-safe in offline contexts.
        self.is_connected = True
        return True

    def disconnect(self) -> None:
        """Close connection to the provider."""
        self.client = None
        self.is_connected = False

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get normalized embeddings from the provider.

        Args:
            texts: List of text strings to embed

        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to LLM. Call connect() first.")

        validated_texts = self._validate_texts(texts)
        embedding_fn = self._get_callable("embedding_fn")

        if embedding_fn is not None:
            raw_embeddings = embedding_fn(validated_texts)
        elif self.client is not None and hasattr(self.client, "get_embeddings"):
            raw_embeddings = self.client.get_embeddings(validated_texts)
        else:
            raw_embeddings = self._mock_embeddings(validated_texts)

        embeddings = self._coerce_embedding_matrix(raw_embeddings, len(validated_texts))
        if self.config.get("normalize_embeddings", True):
            embeddings = self._normalize_rows(embeddings)

        return embeddings

    def run_inference(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run an inference/completion request and normalize the response.

        Args:
            prompt: Prompt to send to the provider
            context: Optional structured context for the provider

        Returns:
            Normalized inference response dictionary
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to LLM. Call connect() first.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-blank string")

        inference_fn = self._get_callable("inference_fn")
        if inference_fn is not None:
            raw_response = inference_fn(prompt, context=context)
        elif self.client is not None and hasattr(self.client, "run_inference"):
            raw_response = self.client.run_inference(prompt, context=context)
        else:
            raw_response = self._mock_inference(prompt, context)

        return self._normalize_inference_response(raw_response, context)

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
            "supports_inference": self._supports_callable("inference_fn")
            or self.provider == "mock",
            "supports_attention": self._supports_callable("attention_fn")
            or self.provider == "mock",
        }

    def get_attention_weights(
        self,
        text: str,
        layer: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Get a validated row-stochastic attention matrix.

        Args:
            text: Input text
            layer: Specific layer to extract attention from (None for all)

        Returns:
            Attention weight matrix or None if not available
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to LLM. Call connect() first.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-blank string")

        attention_fn = self._get_callable("attention_fn")
        if attention_fn is not None:
            raw_attention = attention_fn(text, layer=layer)
        elif self.client is not None and hasattr(self.client, "get_attention_weights"):
            raw_attention = self.client.get_attention_weights(text, layer=layer)
        elif self.provider == "mock":
            raw_attention = self._mock_attention(text)
        else:
            return None

        return self._validate_attention_matrix(raw_attention)

    def extract_topology_data(self, texts: List[str]) -> Dict[str, Any]:
        """Extract topology-relevant data from provider embeddings.

        Args:
            texts: List of input texts

        Returns:
            Dictionary containing embeddings, relationships, etc.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to LLM. Call connect() first.")

        validated_texts = self._validate_texts(texts)
        embeddings = self.get_embeddings(validated_texts)
        normalized_embeddings = self._normalize_rows(embeddings)

        similarities = normalized_embeddings @ normalized_embeddings.T
        np.fill_diagonal(similarities, 1.0)

        threshold = float(self.config.get("similarity_threshold", 0.7))
        top_k = self.config.get("top_k")
        edges = self._build_similarity_edges(similarities, threshold, top_k)

        return {
            "embeddings": embeddings,
            "similarities": similarities,
            "edges": edges,
            "texts": validated_texts,
        }

    def create_topology_from_texts(
        self,
        texts: List[str],
        layer_id: str = "llm_layer",
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a topology structure from input texts.

        Args:
            texts: List of input texts to visualize
            layer_id: ID for the created layer
            prompt: Optional inference prompt to attach to topology metadata

        Returns:
            Dictionary with topology data ready for visualization
        """
        topology_data = self.extract_topology_data(texts)
        inference = self.run_inference(prompt, context={"texts": texts}) if prompt else None
        embedding_dim = int(topology_data["embeddings"].shape[1])

        result = {
            "layer_id": layer_id,
            "num_nodes": len(texts),
            "embeddings": topology_data["embeddings"],
            "labels": topology_data["texts"],
            "edges": topology_data["edges"],
            "metadata": {
                "provider": self.provider,
                "model": self.model_name,
                "embedding_dim": embedding_dim,
                "similarity_threshold": float(self.config.get("similarity_threshold", 0.7)),
                "top_k": self.config.get("top_k"),
            },
        }
        }
        if inference is not None:
            result["inference"] = inference

        return result

    def _get_callable(self, name: str) -> Optional[Callable[..., Any]]:
        candidate = self.config.get(name)
        if callable(candidate):
            return candidate
        if isinstance(self.client, dict) and callable(self.client.get(name)):
            return self.client[name]
        if self.client is not None and callable(getattr(self.client, name, None)):
            return getattr(self.client, name)
        return None

    def _supports_callable(self, name: str) -> bool:
        return self._get_callable(name) is not None

    @staticmethod
    def _validate_texts(texts: List[str]) -> List[str]:
        if not texts:
            raise ValueError("Text batch must contain at least one text")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Text batch cannot contain blank text entries")
        return list(texts)

    def _mock_embeddings(self, texts: List[str]) -> np.ndarray:
        embedding_dim = int(self.config.get("embedding_dim", 768))
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        vectors = np.zeros((len(texts), embedding_dim), dtype=float)
        for row, text in enumerate(texts):
            features = self._text_features(text)
            for feature in features:
                digest = hashlib.sha256(feature.encode("utf-8")).digest()
                index = int.from_bytes(digest[:8], "big") % embedding_dim
                sign = 1.0 if digest[8] % 2 == 0 else -1.0
                vectors[row, index] += sign

            # Extremely small embedding dimensions can make signed feature hashes
            # cancel exactly. Add a deterministic fallback coordinate so mock
            # embeddings remain normalizable for every valid text/configuration.
            if not np.any(vectors[row]):
                digest = hashlib.sha256(text.strip().encode("utf-8")).digest()
                vectors[row, int.from_bytes(digest[:8], "big") % embedding_dim] = 1.0
        return vectors

    @staticmethod
    def _text_features(text: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        features = list(tokens)
        features.extend(" ".join(pair) for pair in zip(tokens, tokens[1:]))
        if not features:
            features = [text.strip().lower()]
        return features

    @staticmethod
    def _coerce_embedding_matrix(raw_embeddings: Any, expected_rows: int) -> np.ndarray:
        embeddings = np.asarray(raw_embeddings, dtype=float)
        if embeddings.ndim != 2:
            raise ValueError("Embedding provider must return a 2D array")
        if embeddings.shape[0] != expected_rows:
            raise ValueError("Embedding row count must match number of texts")
        if embeddings.shape[1] == 0:
            raise ValueError("Embedding dimension must be positive")
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("Embeddings must contain only finite values")
        return embeddings

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Cannot normalize zero-vector embeddings")
        return matrix / norms

    def _mock_inference(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "text": f"Mock inference for '{prompt.strip()}' using {self.model_name}.",
            "context": context or {},
        }

    def _normalize_inference_response(
        self,
        raw_response: Any,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if isinstance(raw_response, dict):
            if "text" not in raw_response:
                raise ValueError("Inference response dictionary must include a 'text' field")
            text = str(raw_response["text"])
            metadata = {"raw": dict(raw_response), "context": context or {}}
        else:
            text = str(raw_response)
            metadata = {"raw": raw_response, "context": context or {}}

        if not text.strip():
            raise ValueError("Inference response text cannot be blank")

        return {
            "text": text,
            "provider": self.provider,
            "model": self.model_name,
            "metadata": metadata,
        }
        }
        if inference is not None:
            result["inference"] = inference

        return result

    def _get_callable(self, name: str) -> Optional[Callable[..., Any]]:
        candidate = self.config.get(name)
        if callable(candidate):
            return candidate
        if isinstance(self.client, dict) and callable(self.client.get(name)):
            return self.client[name]
        if self.client is not None and callable(getattr(self.client, name, None)):
            return getattr(self.client, name)
        return None

    def _supports_callable(self, name: str) -> bool:
        return self._get_callable(name) is not None

    @staticmethod
    def _validate_texts(texts: List[str]) -> List[str]:
        if not texts:
            raise ValueError("Text batch must contain at least one text")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Text batch cannot contain blank text entries")
        return list(texts)

    def _mock_embeddings(self, texts: List[str]) -> np.ndarray:
        embedding_dim = int(self.config.get("embedding_dim", 768))
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        vectors = np.zeros((len(texts), embedding_dim), dtype=float)
        for row, text in enumerate(texts):
            features = self._text_features(text)
            for feature in features:
                digest = hashlib.sha256(feature.encode("utf-8")).digest()
                index = int.from_bytes(digest[:8], "big") % embedding_dim
                sign = 1.0 if digest[8] % 2 == 0 else -1.0
                vectors[row, index] += sign

            # Extremely small embedding dimensions can make signed feature hashes
            # cancel exactly. Add a deterministic fallback coordinate so mock
            # embeddings remain normalizable for every valid text/configuration.
            if not np.any(vectors[row]):
                digest = hashlib.sha256(text.strip().encode("utf-8")).digest()
                vectors[row, int.from_bytes(digest[:8], "big") % embedding_dim] = 1.0
        return vectors

    @staticmethod
    def _text_features(text: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        features = list(tokens)
        features.extend(" ".join(pair) for pair in zip(tokens, tokens[1:]))
        if not features:
            features = [text.strip().lower()]
        return features

    @staticmethod
    def _coerce_embedding_matrix(raw_embeddings: Any, expected_rows: int) -> np.ndarray:
        embeddings = np.asarray(raw_embeddings, dtype=float)
        if embeddings.ndim != 2:
            raise ValueError("Embedding provider must return a 2D array")
        if embeddings.shape[0] != expected_rows:
            raise ValueError("Embedding row count must match number of texts")
        if embeddings.shape[1] == 0:
            raise ValueError("Embedding dimension must be positive")
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("Embeddings must contain only finite values")
        return embeddings

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Cannot normalize zero-vector embeddings")
        return matrix / norms

    def _mock_inference(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "text": f"Mock inference for '{prompt.strip()}' using {self.model_name}.",
            "context": context or {},
        }

    def _normalize_inference_response(
        self,
        raw_response: Any,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if isinstance(raw_response, dict):
            if "text" not in raw_response:
                raise ValueError("Inference response dictionary must include a 'text' field")
            text = str(raw_response["text"])
            metadata = {"raw": dict(raw_response), "context": context or {}}
        else:
            text = str(raw_response)
            metadata = {"raw": raw_response, "context": context or {}}

        if not text.strip():
            raise ValueError("Inference response text cannot be blank")

        return {
            "text": text,
            "provider": self.provider,
            "model": self.model_name,
            "metadata": metadata,
        }

    @staticmethod
    def _validate_attention_matrix(raw_attention: Any) -> np.ndarray:
        attention = np.asarray(raw_attention, dtype=float)
        if attention.ndim != 2 or attention.shape[0] != attention.shape[1]:
            raise ValueError("Attention weights must be a square 2D matrix")
        if not np.all(np.isfinite(attention)):
            raise ValueError("Attention weights must contain only finite values")
        if np.any(attention < 0):
            raise ValueError("Attention weights must be non-negative")

        row_sums = attention.sum(axis=1, keepdims=True)
        if np.any(row_sums == 0):
            raise ValueError("Attention rows must have positive mass")
        return attention / row_sums

    def _mock_attention(self, text: str) -> np.ndarray:
        tokens = text.split()
        num_tokens = max(len(tokens), 1)
        attention = np.eye(num_tokens, dtype=float)
        for i in range(num_tokens):
            for j in range(num_tokens):
                if i != j:
                    distance = abs(i - j)
                    attention[i, j] = 1.0 / (1.0 + distance)
        return attention

    @staticmethod
    def _build_similarity_edges(
        similarities: np.ndarray,
        threshold: float,
        top_k: Optional[int],
    ) -> List[Dict[str, Any]]:
        if not np.isfinite(threshold):
            raise ValueError("similarity_threshold must be finite")

        num_texts = similarities.shape[0]
        edge_map: Dict[tuple, float] = {}

        for source in range(num_texts):
            candidates = [
                (target, float(similarities[source, target]))
                for target in range(num_texts)
                if target != source
            ]
            candidates.sort(key=lambda item: (-item[1], item[0]))
            if top_k is not None:
                k = int(top_k)
                if k < 0:
                    raise ValueError("top_k must be non-negative")
                candidates = candidates[:k]

            for target, weight in candidates:
                if weight >= threshold:
                    key = (min(source, target), max(source, target))
                    edge_map[key] = max(edge_map.get(key, float("-inf")), weight)

        return [
            {"source": source, "target": target, "weight": weight}
            for (source, target), weight in sorted(
                edge_map.items(), key=lambda item: (item[0][0], item[0][1])
            )
        ]
