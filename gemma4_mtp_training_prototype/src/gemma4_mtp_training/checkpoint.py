"""Checkpoint helpers for the standalone assistant prototype."""

from pathlib import Path

from safetensors.torch import load_model, save_model
from torch import nn


def save_assistant_weights(model: nn.Module, output_dir: str | Path) -> Path:
    """Save assistant weights, preserving tied tensor aliases."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.safetensors"
    save_model(model, str(output_path))
    return output_path


def load_assistant_weights(model: nn.Module, checkpoint: str | Path) -> None:
    """Load weights saved by :func:`save_assistant_weights`."""
    missing, unexpected = load_model(model, str(checkpoint), strict=True)
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )


def save_pretrained_assistant(model: nn.Module, output_dir: str | Path) -> Path:
    """Save a Transformers assistant as a vLLM-loadable checkpoint."""
    save_pretrained = getattr(model, "save_pretrained", None)
    if save_pretrained is None:
        raise TypeError("Assistant does not implement save_pretrained()")
    output_dir = Path(output_dir)
    save_pretrained(output_dir, safe_serialization=True)
    return output_dir
