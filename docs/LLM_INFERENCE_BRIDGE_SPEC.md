# LLM Inference Bridge Production Spec

## Goals

The LLM inference bridge makes `braintop` useful with either real model clients or deterministic offline execution. It must expose one stable interface for embeddings, inference text, attention-like matrices, and topology-ready relationship extraction.

## Provider Contract

`LLMConnector` supports these provider modes:

- `mock`: deterministic local behavior for tests, demos, and offline examples.
- `custom`: caller-supplied callables in `config`:
  - `embedding_fn(texts) -> array-like[n_texts, embedding_dim]`
  - `inference_fn(prompt, context=None) -> str | dict`
  - `attention_fn(text, layer=None) -> array-like[n_tokens, n_tokens]`
- `openai`: packaged adapter for OpenAI-compatible clients.
- `sentence-transformers`: packaged embedding adapter.
- `llama-cpp`: packaged adapter for `llama-cpp-python`.
- Other provider names are accepted only when the supplied client implements at least one of
  `get_embeddings`, `run_inference`, or `get_attention_weights`.

Adapters import optional vendor SDKs lazily. Install the matching extra (for example,
`pip install -e ".[openai]"`) or inject a prebuilt client through `config["client"]`.

A missing capability is an explicit `NotImplementedError`; an unknown provider without a
compatible client is a `ValueError`. Neither case silently falls back to mock output.
The `mock` provider is the only provider that synthesizes deterministic demo data.

## Reliability Requirements

1. Embeddings are always a 2D `numpy.ndarray` with one row per input text.
2. Embeddings are L2-normalized unless `normalize_embeddings=False` is configured.
3. Empty text batches and blank text entries are rejected with `ValueError`.
4. Mock embeddings are deterministic across connector instances and process runs.
5. Non-mock providers never synthesize mock embeddings or inference responses when a
   capability is absent.
6. Topology extraction computes cosine similarities from normalized embeddings, excludes self-loops, and emits deterministic edge ordering.
7. `top_k` relationship extraction keeps at most `top_k` outgoing candidates per source node before thresholding.
8. Inference responses are normalized into dictionaries with `text`, `provider`, `model`, and `metadata` fields.
9. Attention matrices are square, finite, non-negative, and row-stochastic.

## Topology Output Schema

`create_topology_from_texts(texts, layer_id)` returns:

```python
{
    "layer_id": str,
    "num_nodes": int,
    "embeddings": np.ndarray,
    "labels": list[str],
    "edges": list[{"source": int, "target": int, "weight": float}],
    "inference": optional dict,
    "metadata": {
        "provider": str,
        "model": str,
        "embedding_dim": int,
        "similarity_threshold": float,
        "top_k": int | None,
    },
}
```

## Acceptance Evals

- Deterministic mock embeddings: same inputs and config produce identical vectors.
- Custom embedding bridge: supplied embedding callable is used, normalized, and validated.
- Inference bridge: mock and custom inference responses are normalized into the production schema.
- Relationship extraction: `top_k` and threshold produce deterministic, self-loop-free edges.
- Attention bridge: custom or mock attention matrices are validated and row-normalized.
