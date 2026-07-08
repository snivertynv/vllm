# Gemma 4 MTP Training Prototype

This standalone project consumes the training-state format produced by
`vllm-project/speculators` PR #674 without modifying either upstream repository.

It currently proves:

- strict validation of the connector's token, hidden-state, and four K/V tensors;
- conversion from connector K/V layout to the Hugging Face assistant layout;
- one-position-at-a-time teacher forcing without future-token K/V leakage;
- left-padded batching of causal training anchors;
- local K/V slicing to the configured sliding window;
- frozen verifier input embeddings and trainable assistant weights; and
- synthetic optimization and checkpoint round-trip tests needing no model access.

## Why training iterates positions

`Gemma4AssistantForCausalLM` is an inference-time, single-position model. Its
attention mask is bidirectional over the supplied verifier cache. Passing a full
training sequence and the full verifier cache would expose future tokens.

For anchor position `t`, this prototype supplies:

- verifier embedding and final hidden state at `t`;
- global K/V from positions `0..t`;
- local K/V from `max(0, t-window+1)..t`; and
- token `t+1` as the target.

The position loop is a correctness reference. A production trainer can batch
anchors with padded K/V prefixes after matching its loss numerically.

## Run the synthetic tests

From this directory, using an environment created with `uv`:

```bash
uv pip install -e '.[test]'
../.venv/bin/python -m pytest -q
```

## Real-checkpoint integration still required

The next stage will load `google/gemma-4-31B-it-assistant` through Transformers
and the verifier's frozen embedding table. `load_pretrained_trainer()` reads
only the verifier safetensor shard containing `embed_tokens`, avoiding loading
the 31B verifier stack into the training process. Real-logit comparison still
requires accepted Gemma access and suitable hardware.

Before using long examples, PR #674 also needs a numerical test showing that
sliding-window K/V rows older than the live vLLM window were preserved rather
than reconstructed from null/freed blocks. Shape and finiteness checks alone do
not establish that alignment.
