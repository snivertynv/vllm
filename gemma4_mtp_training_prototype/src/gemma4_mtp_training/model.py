"""Causally correct training wrapper for Gemma 4 assistant checkpoints."""

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn
from torch.nn import functional as F

from .data import (
    Gemma4AnchorBatch,
    Gemma4TrainingSample,
    collate_anchor_examples,
    iter_anchor_examples,
)


class AssistantOutput(Protocol):
    """Subset of the Transformers assistant output used by training."""

    logits: torch.Tensor


class Gemma4Assistant(Protocol):
    """Forward contract implemented by Gemma4AssistantForCausalLM."""

    def __call__(
        self,
        *,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> AssistantOutput: ...


@dataclass
class Gemma4TrainingOutput:
    """Loss and metrics for one verifier trace."""

    loss: torch.Tensor
    loss_sum: torch.Tensor
    token_count: torch.Tensor
    accuracy: torch.Tensor


class Gemma4AssistantTrainer(nn.Module):
    """Train an assistant from token-aligned verifier hidden and K/V states.

    The production assistant forward is a single-position operation. This
    wrapper preserves that contract: for anchor ``t`` it predicts token
    ``t + 1`` using verifier state at ``t``, global K/V through ``t``, and the
    active local K/V window through ``t``. Iterating anchors is intentionally
    simple and serves as a correctness reference for a later packed kernel.
    """

    def __init__(
        self,
        assistant: nn.Module,
        verifier_embeddings: nn.Embedding,
        *,
        sliding_window: int,
    ) -> None:
        super().__init__()
        if sliding_window <= 0:
            raise ValueError(f"sliding_window must be positive, got {sliding_window}")
        self.assistant = assistant
        self.verifier_embeddings = verifier_embeddings
        self.verifier_embeddings.requires_grad_(False)
        self.sliding_window = sliding_window

    @staticmethod
    def _as_attention_kv(tensor: torch.Tensor) -> torch.Tensor:
        """Convert connector K/V layout to assistant K/V layout."""
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor.permute(0, 2, 1, 3).contiguous()

    def forward_batch(self, batch: Gemma4AnchorBatch) -> Gemma4TrainingOutput:
        """Compute loss for a left-padded batch of causal anchors."""
        token_embeddings = self.verifier_embeddings(batch.token_ids).unsqueeze(1)
        verifier_hidden = batch.verifier_hidden_states.unsqueeze(1)
        inputs_embeds = torch.cat((token_embeddings, verifier_hidden), dim=-1)
        output = self.assistant(
            inputs_embeds=inputs_embeds,
            position_ids=batch.position_ids.unsqueeze(1),
            attention_mask=batch.attention_mask,
            shared_kv_states={
                "sliding_attention": (
                    self._as_attention_kv(batch.local_key),
                    self._as_attention_kv(batch.local_value),
                ),
                "full_attention": (
                    self._as_attention_kv(batch.global_key),
                    self._as_attention_kv(batch.global_value),
                ),
            },
        )
        logits = output.logits[:, -1].float()
        loss_sum = F.cross_entropy(logits, batch.target_ids, reduction="sum")
        token_count = torch.tensor(
            batch.target_ids.shape[0], dtype=torch.float32, device=logits.device
        )
        accuracy = (logits.argmax(dim=-1) == batch.target_ids).float().mean()
        return Gemma4TrainingOutput(
            loss=loss_sum / token_count,
            loss_sum=loss_sum,
            token_count=token_count,
            accuracy=accuracy,
        )

    def _forward_anchor(
        self,
        sample: Gemma4TrainingSample,
        anchor: int,
    ) -> torch.Tensor:
        token = sample.token_ids[anchor].reshape(1, 1)
        token_embedding = self.verifier_embeddings(token)
        verifier_hidden = sample.verifier_last_hidden_states[anchor].reshape(1, 1, -1)
        inputs_embeds = torch.cat((token_embedding, verifier_hidden), dim=-1)

        local_start = max(0, anchor + 1 - self.sliding_window)
        local_key = sample.local_key[local_start : anchor + 1]
        local_value = sample.local_value[local_start : anchor + 1]
        global_key = sample.global_key[: anchor + 1]
        global_value = sample.global_value[: anchor + 1]
        shared_kv_states = {
            "sliding_attention": (
                self._as_attention_kv(local_key),
                self._as_attention_kv(local_value),
            ),
            "full_attention": (
                self._as_attention_kv(global_key),
                self._as_attention_kv(global_value),
            ),
        }

        attention_mask = torch.ones(
            (1, anchor + 1), dtype=torch.bool, device=token.device
        )
        position_ids = torch.full(
            (1, 1), anchor, dtype=torch.long, device=token.device
        )
        output = self.assistant(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,
        )
        return output.logits[:, -1]

    def forward(self, sample: Gemma4TrainingSample) -> Gemma4TrainingOutput:
        """Compute next-token loss over all enabled anchors in ``sample``."""
        if sample.sequence_length < 2:
            raise ValueError("Gemma 4 assistant training needs at least two tokens")
        examples = list(
            iter_anchor_examples(sample, sliding_window=self.sliding_window)
        )
        if not examples:
            raise ValueError("loss_mask selects no next-token targets")
        return self.forward_batch(collate_anchor_examples(examples))
