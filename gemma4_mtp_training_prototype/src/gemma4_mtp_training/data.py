"""Load and validate tensors exported by Speculators PR #674."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import load_file

CONNECTOR_KV_KEYS = (
    "kv_last_local_k",
    "kv_last_local_v",
    "kv_last_global_k",
    "kv_last_global_v",
)
_REQUIRED_KEYS = ("token_ids", "hidden_states", *CONNECTOR_KV_KEYS)


@dataclass(frozen=True)
class Gemma4TrainingSample:
    """One token-aligned verifier trace.

    K/V tensors use the connector layout ``[sequence, heads, head_dim]``.
    ``hidden_states`` may contain one or more captured verifier layers and uses
    ``[sequence, layers, hidden]``.
    """

    token_ids: torch.Tensor
    hidden_states: torch.Tensor
    local_key: torch.Tensor
    local_value: torch.Tensor
    global_key: torch.Tensor
    global_value: torch.Tensor
    loss_mask: torch.Tensor

    @property
    def sequence_length(self) -> int:
        return int(self.token_ids.shape[0])

    @property
    def verifier_last_hidden_states(self) -> torch.Tensor:
        return self.hidden_states[:, -1]

    def to(self, device: torch.device | str) -> "Gemma4TrainingSample":
        """Move every tensor to ``device``."""
        return Gemma4TrainingSample(
            token_ids=self.token_ids.to(device),
            hidden_states=self.hidden_states.to(device),
            local_key=self.local_key.to(device),
            local_value=self.local_value.to(device),
            global_key=self.global_key.to(device),
            global_value=self.global_value.to(device),
            loss_mask=self.loss_mask.to(device),
        )


@dataclass(frozen=True)
class Gemma4AnchorExample:
    """One causally sliced next-token training example."""

    token_id: torch.Tensor
    verifier_hidden_state: torch.Tensor
    local_key: torch.Tensor
    local_value: torch.Tensor
    global_key: torch.Tensor
    global_value: torch.Tensor
    position_id: torch.Tensor
    target_id: torch.Tensor


@dataclass(frozen=True)
class Gemma4AnchorBatch:
    """Left-padded training anchors ready for the assistant wrapper."""

    token_ids: torch.Tensor
    verifier_hidden_states: torch.Tensor
    local_key: torch.Tensor
    local_value: torch.Tensor
    global_key: torch.Tensor
    global_value: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    target_ids: torch.Tensor

    def to(self, device: torch.device | str) -> "Gemma4AnchorBatch":
        """Move every tensor to ``device``."""
        values = {
            field: getattr(self, field).to(device)
            for field in self.__dataclass_fields__
        }
        return Gemma4AnchorBatch(**values)


def iter_anchor_examples(
    sample: Gemma4TrainingSample,
    *,
    sliding_window: int,
):
    """Yield causally sliced anchors selected by the sample loss mask."""
    if sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}")
    for anchor in range(sample.sequence_length - 1):
        target_index = anchor + 1
        if not bool(sample.loss_mask[target_index]):
            continue
        local_start = max(0, anchor + 1 - sliding_window)
        yield Gemma4AnchorExample(
            token_id=sample.token_ids[anchor],
            verifier_hidden_state=sample.verifier_last_hidden_states[anchor],
            local_key=sample.local_key[local_start : anchor + 1],
            local_value=sample.local_value[local_start : anchor + 1],
            global_key=sample.global_key[: anchor + 1],
            global_value=sample.global_value[: anchor + 1],
            position_id=torch.tensor(
                anchor, dtype=torch.long, device=sample.token_ids.device
            ),
            target_id=sample.token_ids[target_index],
        )


def _left_pad(tensors: list[torch.Tensor]) -> torch.Tensor:
    max_length = max(tensor.shape[0] for tensor in tensors)
    output = tensors[0].new_zeros((len(tensors), max_length, *tensors[0].shape[1:]))
    for row, tensor in enumerate(tensors):
        output[row, -tensor.shape[0] :] = tensor
    return output


def collate_anchor_examples(examples: list[Gemma4AnchorExample]) -> Gemma4AnchorBatch:
    """Left-pad causal K/V prefixes and collate training anchors."""
    if not examples:
        raise ValueError("Cannot collate an empty list of Gemma 4 anchors")
    global_lengths = [example.global_key.shape[0] for example in examples]
    max_global_length = max(global_lengths)
    attention_mask = torch.zeros(
        (len(examples), max_global_length),
        dtype=torch.bool,
        device=examples[0].token_id.device,
    )
    for row, length in enumerate(global_lengths):
        attention_mask[row, -length:] = True

    return Gemma4AnchorBatch(
        token_ids=torch.stack([example.token_id for example in examples]),
        verifier_hidden_states=torch.stack(
            [example.verifier_hidden_state for example in examples]
        ),
        local_key=_left_pad([example.local_key for example in examples]),
        local_value=_left_pad([example.local_value for example in examples]),
        global_key=_left_pad([example.global_key for example in examples]),
        global_value=_left_pad([example.global_value for example in examples]),
        attention_mask=attention_mask,
        position_ids=torch.stack([example.position_id for example in examples]),
        target_ids=torch.stack([example.target_id for example in examples]),
    )


def _require_shape(name: str, tensor: torch.Tensor, ndim: int) -> None:
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got {tuple(tensor.shape)}")


def sample_from_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    loss_mask: torch.Tensor | None = None,
) -> Gemma4TrainingSample:
    """Validate connector tensors and construct a training sample."""
    missing = [key for key in _REQUIRED_KEYS if key not in tensors]
    if missing:
        raise ValueError(f"Connector sample is missing required keys: {missing}")

    token_ids = tensors["token_ids"].long()
    hidden_states = tensors["hidden_states"]
    local_key = tensors["kv_last_local_k"]
    local_value = tensors["kv_last_local_v"]
    global_key = tensors["kv_last_global_k"]
    global_value = tensors["kv_last_global_v"]

    _require_shape("token_ids", token_ids, 1)
    _require_shape("hidden_states", hidden_states, 3)
    for name, tensor in (
        ("kv_last_local_k", local_key),
        ("kv_last_local_v", local_value),
        ("kv_last_global_k", global_key),
        ("kv_last_global_v", global_value),
    ):
        _require_shape(name, tensor, 3)

    sequence_length = token_ids.shape[0]
    aligned = {
        "hidden_states": hidden_states.shape[0],
        "kv_last_local_k": local_key.shape[0],
        "kv_last_local_v": local_value.shape[0],
        "kv_last_global_k": global_key.shape[0],
        "kv_last_global_v": global_value.shape[0],
    }
    mismatched = {
        name: size for name, size in aligned.items() if size != sequence_length
    }
    if mismatched:
        raise ValueError(
            "Connector tensors are not token-aligned: "
            f"token_ids={sequence_length}, mismatched={mismatched}"
        )
    if local_key.shape != local_value.shape:
        raise ValueError(
            "Local K/V shapes differ: "
            f"{tuple(local_key.shape)} != {tuple(local_value.shape)}"
        )
    if global_key.shape != global_value.shape:
        raise ValueError(
            "Global K/V shapes differ: "
            f"{tuple(global_key.shape)} != {tuple(global_value.shape)}"
        )

    if loss_mask is None:
        loss_mask = torch.ones(sequence_length, dtype=torch.bool)
    else:
        loss_mask = loss_mask.bool()
        _require_shape("loss_mask", loss_mask, 1)
        if loss_mask.shape[0] != sequence_length:
            raise ValueError(
                "loss_mask is not token-aligned: "
                f"{loss_mask.shape[0]} != {sequence_length}"
            )

    return Gemma4TrainingSample(
        token_ids=token_ids,
        hidden_states=hidden_states,
        local_key=local_key,
        local_value=local_value,
        global_key=global_key,
        global_value=global_value,
        loss_mask=loss_mask,
    )


def load_connector_sample(
    path: str | Path,
    *,
    loss_mask: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> Gemma4TrainingSample:
    """Load a PR #674 ``.safetensors`` artifact."""
    tensors = load_file(str(path), device=str(device))
    return sample_from_tensors(tensors, loss_mask=loss_mask)
