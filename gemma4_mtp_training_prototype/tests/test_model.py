import torch
from torch import nn
from torch.nn import functional as F

from gemma4_mtp_training.checkpoint import (
    load_assistant_weights,
    save_assistant_weights,
)
from gemma4_mtp_training.data import sample_from_tensors
from gemma4_mtp_training.model import Gemma4AssistantTrainer
from gemma4_mtp_training.synthetic import SyntheticGemma4Assistant


def make_sample(sequence_length: int = 6):
    tensors = {
        "token_ids": torch.arange(sequence_length) % 16,
        "hidden_states": torch.randn(sequence_length, 1, 8),
        "kv_last_local_k": torch.randn(sequence_length, 2, 4),
        "kv_last_local_v": torch.randn(sequence_length, 2, 4),
        "kv_last_global_k": torch.randn(sequence_length, 1, 6),
        "kv_last_global_v": torch.randn(sequence_length, 1, 6),
    }
    return sample_from_tensors(tensors)


def test_one_optimization_step_updates_assistant_only() -> None:
    torch.manual_seed(1)
    embeddings = nn.Embedding(16, 8)
    assistant = SyntheticGemma4Assistant(input_size=16, hidden_size=12, vocab_size=16)
    trainer = Gemma4AssistantTrainer(assistant, embeddings, sliding_window=3)
    optimizer = torch.optim.AdamW(assistant.parameters(), lr=1e-2)

    before_embeddings = embeddings.weight.detach().clone()
    output = trainer(make_sample())
    assert output.loss.isfinite()
    assert output.token_count.item() == 5
    output.loss.backward()
    optimizer.step()

    assert torch.equal(embeddings.weight, before_embeddings)
    assert any(parameter.grad is not None for parameter in assistant.parameters())


def test_loss_mask_selects_next_token_targets() -> None:
    sample = make_sample()
    masked = sample_from_tensors(
        {
            "token_ids": sample.token_ids,
            "hidden_states": sample.hidden_states,
            "kv_last_local_k": sample.local_key,
            "kv_last_local_v": sample.local_value,
            "kv_last_global_k": sample.global_key,
            "kv_last_global_v": sample.global_value,
        },
        loss_mask=torch.tensor([0, 0, 1, 0, 1, 0]),
    )
    embeddings = nn.Embedding(16, 8)
    assistant = SyntheticGemma4Assistant(input_size=16, hidden_size=12, vocab_size=16)
    output = Gemma4AssistantTrainer(assistant, embeddings, sliding_window=3)(masked)
    assert output.token_count.item() == 2


def test_batched_loss_matches_single_anchor_reference() -> None:
    torch.manual_seed(2)
    sample = make_sample()
    embeddings = nn.Embedding(16, 8)
    assistant = SyntheticGemma4Assistant(input_size=16, hidden_size=12, vocab_size=16)
    trainer = Gemma4AssistantTrainer(assistant, embeddings, sliding_window=3)

    batched = trainer(sample)
    reference = torch.zeros(())
    for anchor in range(sample.sequence_length - 1):
        logits = trainer._forward_anchor(sample, anchor)
        reference = reference + F.cross_entropy(
            logits.float(), sample.token_ids[anchor + 1].reshape(1)
        )
    assert torch.allclose(batched.loss_sum, reference, atol=1e-6)


def test_synthetic_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(3)
    sample = make_sample()
    embeddings = nn.Embedding(16, 8)
    source = SyntheticGemma4Assistant(input_size=16, hidden_size=12, vocab_size=16)
    source_trainer = Gemma4AssistantTrainer(source, embeddings, sliding_window=3)
    expected = source_trainer(sample).loss.detach()

    checkpoint = save_assistant_weights(source, tmp_path)
    restored = SyntheticGemma4Assistant(input_size=16, hidden_size=12, vocab_size=16)
    restored_trainer = Gemma4AssistantTrainer(restored, embeddings, sliding_window=3)
    restored_trainer(sample)
    load_assistant_weights(restored, checkpoint)
    actual = restored_trainer(sample).loss.detach()
    assert torch.equal(actual, expected)
