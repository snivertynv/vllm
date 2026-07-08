import json

import torch
from safetensors.torch import save_file

from gemma4_mtp_training.pretrained import load_verifier_embeddings


def test_load_verifier_embeddings_from_sharded_index(tmp_path) -> None:
    weight = torch.randn(17, 8)
    shard_name = "model-00001-of-00001.safetensors"
    key = "language_model.model.embed_tokens.weight"
    save_file({key: weight}, tmp_path / shard_name)
    with (tmp_path / "model.safetensors.index.json").open("w") as index_file:
        json.dump({"weight_map": {key: shard_name}}, index_file)

    embedding = load_verifier_embeddings(tmp_path)
    assert not embedding.weight.requires_grad
    assert torch.equal(embedding.weight, weight)
