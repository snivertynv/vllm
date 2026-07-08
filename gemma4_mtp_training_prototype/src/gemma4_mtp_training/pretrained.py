"""Load real assistant and verifier-embedding weights without the verifier stack."""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoModelForCausalLM

from .model import Gemma4AssistantTrainer

_VERIFIER_EMBEDDING_KEYS = (
    "language_model.model.embed_tokens.weight",
    "model.embed_tokens.weight",
)


def _find_safetensor_for_key(checkpoint_dir: Path, key: str) -> Path | None:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open(encoding="utf-8") as index_file:
            weight_map = json.load(index_file)["weight_map"]
        filename = weight_map.get(key)
        return checkpoint_dir / filename if filename is not None else None

    single_file = checkpoint_dir / "model.safetensors"
    if not single_file.exists():
        return None
    with safe_open(single_file, framework="pt", device="cpu") as checkpoint:
        return single_file if key in checkpoint.keys() else None


def load_verifier_embeddings(checkpoint_dir: str | Path) -> nn.Embedding:
    """Load only Gemma 4's frozen token-embedding table.

    The caller must first download the verifier checkpoint to a local directory.
    Loading one safetensor shard avoids materializing the 31B transformer stack.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise ValueError(f"Verifier checkpoint is not a directory: {checkpoint_dir}")

    for key in _VERIFIER_EMBEDDING_KEYS:
        shard = _find_safetensor_for_key(checkpoint_dir, key)
        if shard is None:
            continue
        with safe_open(shard, framework="pt", device="cpu") as checkpoint:
            weight = checkpoint.get_tensor(key)
        return nn.Embedding.from_pretrained(weight, freeze=True)

    raise ValueError(
        "No Gemma 4 verifier embedding was found. Tried keys "
        f"{_VERIFIER_EMBEDDING_KEYS} in {checkpoint_dir}."
    )


def load_pretrained_trainer(
    *,
    assistant_path: str | Path,
    verifier_path: str | Path,
    sliding_window: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
) -> Gemma4AssistantTrainer:
    """Load a real Gemma 4 assistant and its frozen verifier embeddings."""
    assistant = AutoModelForCausalLM.from_pretrained(
        assistant_path,
        dtype=dtype,
    )
    model_type = getattr(assistant.config, "model_type", None)
    if model_type not in {"gemma4_assistant", "gemma4_unified_assistant"}:
        raise ValueError(f"Expected a Gemma 4 assistant, got model_type={model_type!r}")

    verifier_embeddings = load_verifier_embeddings(verifier_path)
    backbone_hidden_size = getattr(assistant.config, "backbone_hidden_size", None)
    if verifier_embeddings.embedding_dim != backbone_hidden_size:
        raise ValueError(
            "Verifier embedding width does not match assistant backbone width: "
            f"{verifier_embeddings.embedding_dim} != {backbone_hidden_size}"
        )

    trainer = Gemma4AssistantTrainer(
        assistant,
        verifier_embeddings,
        sliding_window=sliding_window,
    )
    return trainer.to(device)
