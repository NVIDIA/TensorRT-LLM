# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm.inputs.multimodal import MultimodalParams


def make_embedding(
    num_embeddings: int = 100, hidden_size: int = 16, device: str = "cpu"
) -> Embedding:
    torch.manual_seed(0)
    emb = Embedding(num_embeddings=num_embeddings, embedding_dim=hidden_size)
    emb.weight.data.normal_(mean=0.0, std=0.02)
    return emb.to(device)


class DummyMultimodalModel(MultimodalModelMixin):
    def __init__(self, embedding: Embedding, mm_token_ids: torch.Tensor):
        self.embedding = embedding
        self._mm_token_ids = mm_token_ids

    @property
    def multimodal_token_ids(self) -> torch.Tensor:
        return self._mm_token_ids

    @property
    def text_embedding_layer(self) -> Embedding:
        return self.embedding

    def encode_multimodal_inputs(self, multimodal_params, **encoder_kwargs):
        raise AssertionError("Tests use cached multimodal embeddings and should not encode.")


def make_cached_multimodal_param(mm_embeds: torch.Tensor) -> MultimodalParams:
    return MultimodalParams(multimodal_data={"multimodal_embedding": mm_embeds})


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_prepare_multimodal_inputs_forwards_precomputed_indices(device):
    hidden = 8
    mm_token_id = 7
    emb = make_embedding(num_embeddings=40, hidden_size=hidden, device=device)
    model = DummyMultimodalModel(
        emb,
        torch.tensor([mm_token_id], dtype=torch.long, device=device),
    )

    input_ids = torch.tensor([0, mm_token_id, 1, mm_token_id, 2], dtype=torch.long, device=device)
    text_idx = torch.tensor([0, 2, 3, 4], dtype=torch.long, device=device)
    mm_idx = torch.tensor([1], dtype=torch.long, device=device)
    mm_emb = torch.randn(mm_idx.shape[0], hidden, device=device)

    out = model.prepare_multimodal_inputs(
        input_ids=input_ids,
        positions=None,
        multimodal_params=[make_cached_multimodal_param(mm_emb)],
        num_context_requests=1,
        text_token_indices=text_idx,
        mm_token_indices=mm_idx,
    )

    assert out.input_ids is None
    assert out.inputs_embeds is not None
    assert out.inputs_embeds.shape == (input_ids.numel(), hidden)
    torch.testing.assert_close(
        out.inputs_embeds[mm_idx],
        mm_emb.to(dtype=out.inputs_embeds.dtype, device=out.inputs_embeds.device),
    )
    torch.testing.assert_close(out.inputs_embeds[text_idx], emb(input_ids[text_idx]))
