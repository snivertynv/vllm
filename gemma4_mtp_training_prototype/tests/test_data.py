import pytest
import torch

from gemma4_mtp_training.data import (
    collate_anchor_examples,
    iter_anchor_examples,
    sample_from_tensors,
)


def make_tensors(sequence_length: int = 5) -> dict[str, torch.Tensor]:
    return {
        "token_ids": torch.arange(sequence_length),
        "hidden_states": torch.randn(sequence_length, 2, 8),
        "kv_last_local_k": torch.randn(sequence_length, 2, 4),
        "kv_last_local_v": torch.randn(sequence_length, 2, 4),
        "kv_last_global_k": torch.randn(sequence_length, 1, 6),
        "kv_last_global_v": torch.randn(sequence_length, 1, 6),
    }


def test_sample_accepts_pr674_layout() -> None:
    sample = sample_from_tensors(make_tensors())
    assert sample.sequence_length == 5
    assert sample.verifier_last_hidden_states.shape == (5, 8)
    assert sample.loss_mask.all()


def test_sample_rejects_misaligned_kv() -> None:
    tensors = make_tensors()
    tensors["kv_last_local_k"] = tensors["kv_last_local_k"][:-1]
    with pytest.raises(ValueError, match="not token-aligned"):
        sample_from_tensors(tensors)


def test_collator_left_pads_causal_prefixes() -> None:
    sample = sample_from_tensors(make_tensors())
    examples = list(iter_anchor_examples(sample, sliding_window=2))
    batch = collate_anchor_examples([examples[0], examples[-1]])
    assert batch.global_key.shape[:2] == (2, 4)
    assert batch.local_key.shape[:2] == (2, 2)
    assert batch.attention_mask.tolist() == [
        [False, False, False, True],
        [True, True, True, True],
    ]
    assert torch.equal(batch.target_ids, torch.tensor([1, 4]))
