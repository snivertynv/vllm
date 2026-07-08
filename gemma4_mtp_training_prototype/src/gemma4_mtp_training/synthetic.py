"""Tiny assistant components used to validate training without model access."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SyntheticAssistantOutput:
    """Output compatible with the training wrapper."""

    logits: torch.Tensor


class SyntheticGemma4Assistant(nn.Module):
    """Small differentiable stand-in for the Transformers assistant.

    This is not an architectural replica. It verifies the external training
    contract: fused input state, two differently shaped K/V sources, and a
    trainable next-token objective.
    """

    def __init__(self, input_size: int, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size, bias=False)
        self.local_projection = nn.LazyLinear(hidden_size, bias=False)
        self.global_projection = nn.LazyLinear(hidden_size, bias=False)
        self.output_projection = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> SyntheticAssistantOutput:
        del position_ids
        local_key, local_value = shared_kv_states["sliding_attention"]
        global_key, global_value = shared_kv_states["full_attention"]
        local_mask = attention_mask[:, -local_key.shape[2] :]
        local_context = self._masked_context(local_key, local_value, local_mask)
        global_context = self._masked_context(
            global_key, global_value, attention_mask
        )
        hidden = self.input_projection(inputs_embeds)
        hidden = hidden + self.local_projection(local_context).unsqueeze(1)
        hidden = hidden + self.global_projection(global_context).unsqueeze(1)
        return SyntheticAssistantOutput(
            logits=self.output_projection(torch.tanh(hidden))
        )

    @staticmethod
    def _masked_context(
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        states = torch.cat((key, value), dim=-1)
        weights = mask[:, None, :, None].to(states.dtype)
        summed = (states * weights).sum(dim=(1, 2))
        denominator = weights.sum(dim=(1, 2)).clamp(min=1) * states.shape[1]
        return summed / denominator
