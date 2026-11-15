"""Packaged provider adapters for :class:`braintop.integrations.LLMConnector`.

Adapters keep optional vendor SDK imports out of module import time. Each adapter
accepts a prebuilt client for tests/production dependency injection, or imports
its SDK lazily from ``connect()`` when a client is not supplied.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np


class CallableAdapter:
    """Adapter around user-supplied callables."""

    def __init__(
        self,
        embedding_fn: Optional[Callable[[List[str]], Any]] = None,
        inference_fn: Optional[Callable[..., Any]] = None,
        attention_fn: Optional[Callable[..., Any]] = None,
        model_name: str = "custom",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.embedding_fn = embedding_fn
        self.inference_fn = inference_fn
        self.attention_fn = attention_fn
        self.model_name = model_name
        self.config = config or {}
        self.is_connected = False

    def connect(self) -> bool:
        self.is_connected = True
        return True

    def disconnect(self) -> None:
        self.is_connected = False

    def get_embeddings(self, texts: List[str]) -> Any:
        if self.embedding_fn is None:
            raise NotImplementedError("No embedding_fn configured")
        return self.embedding_fn(texts)

    def run_inference(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Any:
        if self.inference_fn is None:
            raise NotImplementedError("No inference_fn configured")
        return self.inference_fn(prompt, context=context)

    def get_attention_weights(self, text: str, layer: Optional[int] = None) -> Any:
        if self.attention_fn is None:
            raise NotImplementedError("No attention_fn configured")
        return self.attention_fn(text, layer=layer)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "adapter": "callable",
            "model_name": self.model_name,
            "supports_embeddings": self.embedding_fn is not None,
            "supports_inference": self.inference_fn is not None,
            "supports_attention": self.attention_fn is not None,
        }


class SentenceTransformerAdapter:
    """Adapter for ``sentence-transformers`` embedding models."""

    def __init__(
        self,
        model_name: str,
        client: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.client = client
        self.config = config or {}
        self.is_connected = False

    def connect(self) -> bool:
        if self.client is None:
            from sentence_transformers import SentenceTransformer

            model_kwargs = self.config.get("model_kwargs", {})
            self.client = SentenceTransformer(self.model_name, **model_kwargs)

        self.is_connected = True
        return True

    def disconnect(self) -> None:
        self.is_connected = False

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        encode_kwargs = dict(self.config.get("encode_kwargs", {}))
        encode_kwargs.setdefault("convert_to_numpy", True)
        return np.asarray(self.client.encode(texts, **encode_kwargs), dtype=float)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "adapter": "sentence-transformers",
            "model_name": self.model_name,
            "supports_embeddings": True,
            "supports_inference": False,
            "supports_attention": False,
        }


class OpenAIAdapter:
    """Adapter for OpenAI-compatible clients.

    The adapter expects either a prebuilt ``client`` with ``embeddings`` and
    optionally ``chat.completions`` resources, or the ``openai`` package
    available at runtime.
    """

    def __init__(
        self,
        model_name: str,
        client: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.client = client
        self.config = config or {}
        self.is_connected = False

    def connect(self) -> bool:
        if self.client is None:
            from openai import OpenAI

            client_kwargs = dict(self.config.get("client_kwargs", {}))
            for key in ("api_key", "base_url", "organization", "project"):
                if key in self.config:
                    client_kwargs[key] = self.config[key]
            self.client = OpenAI(**client_kwargs)

        self.is_connected = True
        return True

    def disconnect(self) -> None:
        self.is_connected = False

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        embedding_model = self.config.get("embedding_model", self.model_name)
        response = self.client.embeddings.create(model=embedding_model, input=texts)
        return np.asarray([item.embedding for item in response.data], dtype=float)

    def run_inference(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        chat_model = self.config.get("chat_model", self.model_name)
        system_prompt = self.config.get("system_prompt")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if context:
            messages.append({"role": "system", "content": f"Context: {context}"})
        messages.append({"role": "user", "content": prompt})

        completion_kwargs = dict(self.config.get("completion_kwargs", {}))
        response = self.client.chat.completions.create(
            model=chat_model,
            messages=messages,
            **completion_kwargs,
        )
        return response.choices[0].message.content

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "adapter": "openai",
            "model_name": self.model_name,
            "embedding_model": self.config.get("embedding_model", self.model_name),
            "chat_model": self.config.get("chat_model", self.model_name),
            "supports_embeddings": True,
            "supports_inference": True,
            "supports_attention": False,
        }


class LlamaCppAdapter:
    """Adapter for ``llama-cpp-python`` clients."""

    def __init__(
        self,
        model_name: str,
        client: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.client = client
        self.config = config or {}
        self.is_connected = False

    def connect(self) -> bool:
        if self.client is None:
            model_path = self.config.get("model_path") or self.model_name
            if not model_path:
                raise ValueError("model_path or model_name is required for LlamaCppAdapter")
            from llama_cpp import Llama

            llama_kwargs = dict(self.config.get("llama_kwargs", {}))
            self.client = Llama(model_path=model_path, embedding=True, **llama_kwargs)

        self.is_connected = True
        return True

    def disconnect(self) -> None:
        self.is_connected = False

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        response = self.client.create_embedding(input=texts)
        data = response["data"] if isinstance(response, dict) else response.data
        embeddings = [
            item["embedding"] if isinstance(item, dict) else item.embedding for item in data
        ]
        return np.asarray(embeddings, dtype=float)

    def run_inference(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        inference_kwargs = dict(self.config.get("inference_kwargs", {}))
        if hasattr(self.client, "create_chat_completion"):
            messages = []
            if context:
                messages.append({"role": "system", "content": f"Context: {context}"})
            messages.append({"role": "user", "content": prompt})
            response = self.client.create_chat_completion(messages=messages, **inference_kwargs)
            choice = response["choices"][0]
            return choice.get("message", {}).get("content", choice.get("text", ""))

        response = self.client(prompt, **inference_kwargs)
        if isinstance(response, dict):
            return response.get("choices", [{}])[0].get("text", "")
        return str(response)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "adapter": "llama-cpp",
            "model_name": self.model_name,
            "supports_embeddings": True,
            "supports_inference": True,
            "supports_attention": False,
        }


__all__ = [
    "CallableAdapter",
    "SentenceTransformerAdapter",
    "OpenAIAdapter",
    "LlamaCppAdapter",
]
