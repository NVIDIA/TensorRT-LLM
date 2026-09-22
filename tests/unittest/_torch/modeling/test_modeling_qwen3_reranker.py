# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from tensorrt_llm._torch.models.modeling_qwen3 import Qwen3ForTextReranking
from tensorrt_llm.serve.reranker import rerank_probability

_YES_TOKEN_ID = 9693
_NO_TOKEN_ID = 2152


def _vocab_weight(hidden_size=8):
    return torch.randn(_YES_TOKEN_ID + 1, hidden_size)


def test_qwen3_reranker_is_registered_and_non_generation():
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING

    assert "Qwen3ForTextReranking" in MODEL_CLASS_MAPPING
    assert ModelConfig.is_generation_model(["Qwen3ForTextReranking"]) is False
    assert ModelConfig.is_generation_model(["Qwen3ForCausalLM"]) is True


def test_derived_head_matches_yes_minus_no_logits():
    lm_head = _vocab_weight()
    source_name, score_weight = Qwen3ForTextReranking._derive_score_weight(
        {"lm_head.weight": lm_head}
    )
    hidden = torch.randn(3, lm_head.shape[1])

    assert source_name == "lm_head.weight"
    expected = hidden @ lm_head[_YES_TOKEN_ID] - hidden @ lm_head[_NO_TOKEN_ID]
    actual = (hidden @ score_weight.T).flatten()
    torch.testing.assert_close(actual, expected)


def test_derived_head_uses_tied_embedding_weight():
    embedding = _vocab_weight()
    source_name, score_weight = Qwen3ForTextReranking._derive_score_weight(
        {"model.embed_tokens.weight": embedding}
    )

    assert source_name == "model.embed_tokens.weight"
    torch.testing.assert_close(
        score_weight,
        embedding[_YES_TOKEN_ID : _YES_TOKEN_ID + 1] - embedding[_NO_TOKEN_ID : _NO_TOKEN_ID + 1],
    )


def test_derived_head_supports_lazy_slices():
    class LazySlice:
        def __init__(self, tensor):
            self.tensor = tensor
            self.requests = []

        def __getitem__(self, index):
            self.requests.append(index)
            return self.tensor[index]

    lazy = LazySlice(_vocab_weight())
    _, score_weight = Qwen3ForTextReranking._derive_score_weight({"lm_head.weight": lazy})

    assert len(lazy.requests) == 2
    assert score_weight.shape == (1, lazy.tensor.shape[1])


def test_derived_head_rejects_checkpoint_without_lm_head_or_embeddings():
    with pytest.raises(ValueError, match="must contain lm_head.weight"):
        Qwen3ForTextReranking._derive_score_weight({})


def test_sigmoid_matches_official_two_token_softmax():
    no_logit = torch.tensor(-1.25)
    yes_logit = torch.tensor(2.5)
    expected = torch.softmax(torch.stack([no_logit, yes_logit]), dim=0)[1]

    actual = rerank_probability(float(yes_logit - no_logit))
    assert actual == pytest.approx(float(expected))
    assert rerank_probability(1000.0) == 1.0
    assert rerank_probability(-1000.0) == 0.0
