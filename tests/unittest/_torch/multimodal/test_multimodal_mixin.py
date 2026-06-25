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

from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm.inputs.multimodal import MultimodalInput, MultimodalParams, MultimodalRuntimeData
from tensorrt_llm.llmapi.llm_args import MultimodalConfig


def make_embedding(
    num_embeddings: int = 100, hidden_size: int = 16, device: str = "cpu"
) -> Embedding:
    torch.manual_seed(0)
    emb = Embedding(num_embeddings=num_embeddings, embedding_dim=hidden_size)
    emb.weight.data.normal_(mean=0.0, std=0.02)
    return emb.to(device)


class DummyMultimodalModel(MultimodalModelMixin):
    def __init__(self, embedding: Embedding, mm_token_ids: torch.Tensor):
        self.model_config = ModelConfig()
        self.embedding = embedding
        self._mm_token_ids = mm_token_ids

    @property
    def multimodal_token_ids(self) -> torch.Tensor:
        return self._mm_token_ids

    @property
    def text_embedding_layer(self) -> Embedding:
        return self.embedding

    @property
    def embedding_dim(self) -> int:
        return self.embedding.embedding_dim

    @property
    def embedding_dtype(self) -> torch.dtype:
        return self.embedding.weight.dtype

    def encode_multimodal_inputs(self, multimodal_params):
        raise AssertionError("Tests use cached multimodal embeddings and should not encode.")


class TensorEncoderMultimodalModel(DummyMultimodalModel):
    def __init__(
        self,
        embedding: Embedding,
        mm_token_ids: torch.Tensor,
        mm_embeds: torch.Tensor,
    ):
        super().__init__(embedding, mm_token_ids)
        self.mm_embeds = mm_embeds

    def encode_multimodal_inputs(self, multimodal_params, **encoder_kwargs) -> torch.Tensor:
        return self.mm_embeds


class NoEmbeddingMetadataMultimodalModel(DummyMultimodalModel):
    @property
    def embedding_dim(self) -> int:
        raise NotImplementedError

    @property
    def embedding_dtype(self) -> torch.dtype:
        raise NotImplementedError


class CountingEncoderMultimodalModel(DummyMultimodalModel):
    def __init__(
        self,
        embedding: Embedding,
        mm_token_ids: torch.Tensor,
        *,
        encoder_cache_max_bytes: int = 0,
    ):
        super().__init__(embedding, mm_token_ids)
        self.model_config = ModelConfig(
            multimodal_config=MultimodalConfig(encoder_cache_max_bytes=encoder_cache_max_bytes)
        )
        self.encode_calls = 0

    def encode_multimodal_inputs(self, multimodal_params, **encoder_kwargs) -> torch.Tensor:
        self.encode_calls += 1
        total_rows = sum(
            param.multimodal_runtime.total_embeds_in_request for param in multimodal_params
        )
        return torch.full(
            (total_rows, self.embedding.embedding_dim),
            float(self.encode_calls),
            dtype=torch.float32,
        )


def make_cached_multimodal_param(mm_embeds: torch.Tensor) -> MultimodalParams:
    return MultimodalParams(multimodal_data={"multimodal_embedding": mm_embeds})


def make_raw_multimodal_param() -> MultimodalParams:
    return MultimodalParams(multimodal_data={"image": {"pixel_values": torch.empty(1)}})


def make_runtime(total_embeds: int) -> MultimodalRuntimeData:
    return MultimodalRuntimeData(
        embed_mask_cumsum=torch.arange(1, total_embeds + 1, dtype=torch.int64),
        past_seen_token_num=0,
        chunk_end_pos=total_embeds,
    )


def make_keyed_multimodal_param(
    *,
    item_hashes: list[list[int]] | None = None,
    embedding_lengths: list[int] | None = None,
    kwargs_hash: str | None = "kwargs-a",
    local_embedding: torch.Tensor | None = None,
) -> MultimodalParams:
    if item_hashes is None:
        item_hashes = [[1, 2, 3, 4, 5, 6, 7, 8]]
    if embedding_lengths is None:
        embedding_lengths = [2]

    mm_data = {
        "image": {"pixel_values": torch.empty(1)},
        "multimodal_embedding_lengths": embedding_lengths,
        "mm_processor_kwargs_hash": kwargs_hash,
    }
    if local_embedding is not None:
        mm_data["multimodal_embedding"] = local_embedding

    return MultimodalParams(
        multimodal_input=MultimodalInput(
            multimodal_hashes=item_hashes,
            multimodal_positions=[0] * len(item_hashes),
            multimodal_lengths=embedding_lengths,
        ),
        multimodal_data=mm_data,
        multimodal_runtime=make_runtime(sum(embedding_lengths)),
    )


def test_cast_multimodal_encoder_dtype_keeps_meta_tensors_meta():
    module = torch.nn.Linear(4, 4, device="meta")

    MultimodalModelMixin._cast_multimodal_encoder_dtype(module, torch.float16)

    assert module.weight.device.type == "meta"
    assert module.weight.dtype == torch.float16
    assert module.bias.device.type == "meta"
    assert module.bias.dtype == torch.float16


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


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_prepare_multimodal_inputs_accepts_tensor_encoder_output(device):
    hidden = 8
    mm_token_id = 7
    emb = make_embedding(num_embeddings=40, hidden_size=hidden, device=device)

    input_ids = torch.tensor([0, mm_token_id, 1], dtype=torch.long, device=device)
    text_idx = torch.tensor([0, 2], dtype=torch.long, device=device)
    mm_idx = torch.tensor([1], dtype=torch.long, device=device)
    mm_emb = torch.randn(mm_idx.shape[0], hidden, device=device)
    model = TensorEncoderMultimodalModel(
        emb,
        torch.tensor([mm_token_id], dtype=torch.long, device=device),
        mm_emb,
    )

    out = model.prepare_multimodal_inputs(
        input_ids=input_ids,
        positions=None,
        multimodal_params=[make_raw_multimodal_param()],
        num_context_requests=1,
        text_token_indices=text_idx,
        mm_token_indices=mm_idx,
    )

    assert out.input_ids is None
    assert out.inputs_embeds is not None
    torch.testing.assert_close(
        out.inputs_embeds[mm_idx],
        mm_emb.to(dtype=out.inputs_embeds.dtype, device=out.inputs_embeds.device),
    )
    torch.testing.assert_close(out.inputs_embeds[text_idx], emb(input_ids[text_idx]))


def test_encoder_cache_first_request_writes_per_item_entries():
    model = CountingEncoderMultimodalModel(
        make_embedding(hidden_size=4),
        torch.tensor([7]),
        encoder_cache_max_bytes=4096,
    )
    param = make_keyed_multimodal_param(
        item_hashes=[[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]],
        embedding_lengths=[2, 1],
    )

    embeddings = model._get_or_encode_multimodal_embeddings([param])

    assert model.encode_calls == 1
    assert embeddings.shape == (3, 4)
    assert len(model._multimodal_encoder_cache) == 2


def test_encoder_cache_creation_logs_embedding_row_capacity():
    model = CountingEncoderMultimodalModel(
        make_embedding(hidden_size=4),
        torch.tensor([7]),
        encoder_cache_max_bytes=4096,
    )

    with patch("tensorrt_llm._torch.models.modeling_multimodal_mixin.logger.info") as info:
        model._get_multimodal_encoder_cache()

    messages = [" ".join(map(str, call.args)) for call in info.call_args_list]
    assert any(
        "mm_encoder_cache: created with max_bytes=4096, max_embedding_rows=256, "
        "embedding_dim=4, embedding_dtype=torch.float32, clone_on_insert=True" in message
        for message in messages
    )


def test_encoder_cache_creation_logs_byte_capacity_without_embedding_metadata():
    model = NoEmbeddingMetadataMultimodalModel(make_embedding(), torch.tensor([7]))
    model.model_config = ModelConfig(
        multimodal_config=MultimodalConfig(encoder_cache_max_bytes=4096)
    )

    with patch("tensorrt_llm._torch.models.modeling_multimodal_mixin.logger.info") as info:
        model._get_multimodal_encoder_cache()

    messages = [" ".join(map(str, call.args)) for call in info.call_args_list]
    assert any(
        "mm_encoder_cache: created with max_bytes=4096, clone_on_insert=True; "
        "embedding row capacity unavailable because the model does not implement "
        "embedding_dim and embedding_dtype." in message
        for message in messages
    )


def test_encoder_cache_repeated_item_across_requests_skips_encoder():
    model = CountingEncoderMultimodalModel(
        make_embedding(hidden_size=4),
        torch.tensor([7]),
        encoder_cache_max_bytes=4096,
    )
    first = make_keyed_multimodal_param()
    second = make_keyed_multimodal_param()

    first_embeddings = model._get_or_encode_multimodal_embeddings([first])
    second_embeddings = model._get_or_encode_multimodal_embeddings([second])

    assert model.encode_calls == 1
    torch.testing.assert_close(second_embeddings, first_embeddings)


def test_encoder_cache_partial_hit_logs_and_uses_encoder():
    model = CountingEncoderMultimodalModel(
        make_embedding(hidden_size=4),
        torch.tensor([7]),
        encoder_cache_max_bytes=4096,
    )
    first = make_keyed_multimodal_param()
    partial = make_keyed_multimodal_param(
        item_hashes=[
            [1, 2, 3, 4, 5, 6, 7, 8],
            [9, 9, 9, 9, 9, 9, 9, 9],
        ],
        embedding_lengths=[2, 2],
    )

    model._get_or_encode_multimodal_embeddings([first])
    with patch("tensorrt_llm._torch.models.modeling_multimodal_mixin.logger.debug") as debug:
        embeddings = model._get_or_encode_multimodal_embeddings([partial])

    messages = [" ".join(map(str, call.args)) for call in debug.call_args_list]
    assert model.encode_calls == 2
    assert embeddings.shape == (4, 4)
    assert any(
        "mm_encoder_cache: cache miss; hit_items=1, total_items=2" in msg for msg in messages
    )


def test_encoder_cache_logs_rejected_oversized_write():
    model = CountingEncoderMultimodalModel(
        make_embedding(hidden_size=4),
        torch.tensor([7]),
        encoder_cache_max_bytes=16,
    )
    param = make_keyed_multimodal_param(embedding_lengths=[2])

    with patch("tensorrt_llm._torch.models.modeling_multimodal_mixin.logger.debug") as debug:
        model._get_or_encode_multimodal_embeddings([param])

    cache = model._multimodal_encoder_cache
    assert cache is not None
    assert len(cache) == 0
    assert cache.stats().rejected_insertions == 1
    messages = [" ".join(map(str, call.args)) for call in debug.call_args_list]
    assert any("mm_encoder_cache: wrote 0 item entries, rejected=1" in msg for msg in messages)


def test_encoder_cache_mm_processor_kwargs_do_not_collide():
    model = CountingEncoderMultimodalModel(
        make_embedding(hidden_size=4),
        torch.tensor([7]),
        encoder_cache_max_bytes=4096,
    )
    first = make_keyed_multimodal_param(kwargs_hash="kwargs-a")
    second = make_keyed_multimodal_param(kwargs_hash="kwargs-b")

    first_embeddings = model._get_or_encode_multimodal_embeddings([first])
    second_embeddings = model._get_or_encode_multimodal_embeddings([second])

    assert model.encode_calls == 2
    assert not torch.equal(first_embeddings, second_embeddings)


def test_disabled_encoder_cache_preserves_current_behavior():
    model = CountingEncoderMultimodalModel(
        make_embedding(hidden_size=4),
        torch.tensor([7]),
        encoder_cache_max_bytes=0,
    )

    model._get_or_encode_multimodal_embeddings([make_keyed_multimodal_param()])
    model._get_or_encode_multimodal_embeddings([make_keyed_multimodal_param()])

    assert model.encode_calls == 2
    assert model._multimodal_encoder_cache is None


@pytest.mark.parametrize(
    "param",
    [
        MultimodalParams(
            multimodal_data={
                "image": {"pixel_values": torch.empty(1)},
                "multimodal_embedding_lengths": [2],
                "mm_processor_kwargs_hash": "kwargs-a",
            },
            multimodal_runtime=make_runtime(2),
        ),
        make_keyed_multimodal_param(kwargs_hash=None),
    ],
    ids=["missing_hashes", "unserializable_kwargs"],
)
def test_unkeyable_requests_skip_persistent_encoder_cache(param):
    model = CountingEncoderMultimodalModel(
        make_embedding(hidden_size=4),
        torch.tensor([7]),
        encoder_cache_max_bytes=4096,
    )

    model._get_or_encode_multimodal_embeddings([param])

    assert model.encode_calls == 1
    assert len(model._multimodal_encoder_cache) == 0


def test_request_local_multimodal_embedding_wins_over_encoder_cache():
    model = CountingEncoderMultimodalModel(
        make_embedding(hidden_size=4),
        torch.tensor([7]),
        encoder_cache_max_bytes=4096,
    )
    model._get_or_encode_multimodal_embeddings([make_keyed_multimodal_param()])
    local_embedding = torch.full((2, 4), 99.0)
    chunk_param = make_keyed_multimodal_param(local_embedding=local_embedding)

    embeddings = model._get_or_encode_multimodal_embeddings([chunk_param])

    assert model.encode_calls == 1
    torch.testing.assert_close(embeddings, local_embedding)
