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
from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    MultimodalModelMixin,
    _assemble_multimodal_encoder_embeddings,
    make_multimodal_encoder_model_config,
    reorder_multimodal_embeddings_by_modality,
)
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalInput,
    MultimodalParams,
    MultimodalRuntimeData,
)
from tensorrt_llm.inputs.registry import MultimodalEncoderItemMetadata
from tensorrt_llm.llmapi.llm_args import MultimodalConfig
from tensorrt_llm.mapping import Mapping


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
    supports_encoder_cache = True

    @property
    def embedding_dim(self) -> int:
        raise NotImplementedError

    @property
    def embedding_dtype(self) -> torch.dtype:
        raise NotImplementedError


class CountingEncoderMultimodalModel(DummyMultimodalModel):
    supports_encoder_cache = True

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
        total_rows = 0
        for param in multimodal_params:
            # Residuals built by the partial-cache path carry `multimodal_embedding_lengths`
            # but no `multimodal_runtime`; fall through to the metadata in that case.
            if param.multimodal_runtime is not None:
                total_rows += param.multimodal_runtime.total_embeds_in_request
            else:
                total_rows += sum(param.multimodal_data["multimodal_embedding_lengths"])
        return torch.full(
            (total_rows, self.embedding.embedding_dim),
            float(self.encode_calls),
            dtype=torch.float32,
        )


class DataParallelEncoderMultimodalModel(DummyMultimodalModel):
    supports_encoder_data_parallel = True

    def __init__(self, rank: int, *, enable_attention_dp: bool = False):
        embedding = make_embedding(num_embeddings=8, hidden_size=2)
        super().__init__(embedding, torch.tensor([7]))
        self.model_config = ModelConfig(
            mapping=Mapping(
                world_size=2,
                rank=rank,
                tp_size=2,
                enable_attention_dp=enable_attention_dp,
            ),
            multimodal_config=MultimodalConfig(
                encoder_data_parallel_size=1 if enable_attention_dp else 2,
                encoder_cache_max_bytes=0,
            ),
        )
        self.peer_output = None
        self.encoded_item_ids = []

    def encode_multimodal_inputs(self, multimodal_params, **encoder_kwargs) -> torch.Tensor:
        rows = []
        for param in multimodal_params:
            item_ids = param.multimodal_data["image"]["pixel_values"][:, 0, 0, 0]
            lengths = param.multimodal_data["multimodal_embedding_lengths"]
            self.encoded_item_ids.extend(int(item_id) for item_id in item_ids)
            for item_id, length in zip(item_ids, lengths, strict=True):
                rows.append(torch.full((length, 2), float(item_id)))
        return torch.cat(rows, dim=0)

    def _allreduce_encoder_dp_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype == torch.int32:
            return tensor
        assert self.peer_output is not None
        return tensor + self.peer_output


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
    n_items = len(embedding_lengths)

    # Pattern-A image data so the mixin's default `build_multimodal_encoder_input` can
    # slice this param (dim-0 `pixel_values [B, C, H, W]` parallel to a per-item
    # `image_sizes` list).
    mm_data = {
        "image": {
            "pixel_values": torch.arange(n_items * 3 * 2 * 2, dtype=torch.float32).reshape(
                n_items, 3, 2, 2
            ),
            "image_sizes": [[2, 2]] * n_items,
        },
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


def make_encoder_dp_param() -> MultimodalParams:
    return MultimodalParams(
        multimodal_data={
            "image": {
                "pixel_values": torch.tensor([10.0, 20.0, 30.0]).reshape(3, 1, 1, 1),
                "image_sizes": [[1, 1], [1, 1], [1, 1]],
            },
            "multimodal_embedding_lengths": [4, 1, 3],
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalEncoderItemMetadata(
                item_refs=[("image", 0), ("image", 1), ("image", 2)],
                encoder_token_lengths=[4, 1, 3],
                output_embedding_lengths=[4, 1, 3],
            ),
        }
    )


def make_encoder_dp_param_with_input_costs() -> MultimodalParams:
    param = make_encoder_dp_param()
    param.multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = MultimodalEncoderItemMetadata(
        item_refs=[("image", 0), ("image", 1), ("image", 2)],
        encoder_token_lengths=[1, 100, 1],
        output_embedding_lengths=[4, 1, 3],
    )
    return param


@pytest.mark.parametrize("world_size,rank", [(2, 1), (4, 3)])
def test_make_multimodal_encoder_model_config_replicates_encoder_mapping(world_size, rank):
    model_config = ModelConfig(
        mapping=Mapping(world_size=world_size, rank=rank, tp_size=world_size),
        multimodal_config=MultimodalConfig(encoder_data_parallel_size=world_size),
    )

    encoder_config = make_multimodal_encoder_model_config(model_config)

    assert model_config.mapping.tp_size == world_size
    assert encoder_config.mapping.tp_size == 1
    assert encoder_config.mapping.pp_size == world_size
    assert encoder_config.mapping.rank == rank
    assert encoder_config.mapping.local_rank == model_config.mapping.local_rank


def test_attention_dp_replicates_encoder_mapping_without_explicit_encoder_dp():
    model_config = ModelConfig(
        mapping=Mapping(
            world_size=2,
            rank=0,
            tp_size=2,
            enable_attention_dp=True,
        ),
        multimodal_config=MultimodalConfig(),
    )

    encoder_config = make_multimodal_encoder_model_config(model_config)

    assert model_config.mapping.enable_attention_dp
    assert encoder_config.mapping.tp_size == 1
    assert not encoder_config.mapping.enable_attention_dp


@pytest.mark.parametrize(
    "mapping,error_match",
    [
        (
            Mapping(world_size=2, rank=0, tp_size=2, enable_attention_dp=True),
            "cannot be combined with attention data parallelism",
        ),
        (
            Mapping(world_size=4, rank=0, tp_size=2, pp_size=2),
            "requires pipeline parallel size 1 and context parallel size 1",
        ),
        (
            Mapping(world_size=4, rank=0, tp_size=2, cp_size=2),
            "requires pipeline parallel size 1 and context parallel size 1",
        ),
        (
            Mapping(world_size=4, rank=0, tp_size=4),
            "must equal the tensor parallel size",
        ),
    ],
    ids=["attention_dp", "pipeline_parallel", "context_parallel", "partial_tp_group"],
)
def test_encoder_data_parallel_rejects_unsupported_parallelism(mapping, error_match):
    model_config = ModelConfig(
        mapping=mapping,
        multimodal_config=MultimodalConfig(encoder_data_parallel_size=2),
    )

    with pytest.raises((NotImplementedError, ValueError), match=error_match):
        make_multimodal_encoder_model_config(model_config)


def test_encoder_data_parallel_partitions_items_and_restores_order():
    param = make_encoder_dp_param()
    rank_zero_model = DataParallelEncoderMultimodalModel(rank=0)
    rank_zero_model.peer_output = (
        torch.tensor([0.0] * 4 + [20.0] + [30.0] * 3).unsqueeze(1).repeat(1, 2)
    )

    output = rank_zero_model._run_multimodal_encoder([param])

    assert rank_zero_model.encoded_item_ids == [10]
    expected = torch.tensor([10.0] * 4 + [20.0] + [30.0] * 3).unsqueeze(1).repeat(1, 2)
    torch.testing.assert_close(output, expected)


def test_item_scheduled_encoder_uses_data_parallel_execution():
    param = make_encoder_dp_param()
    model = DataParallelEncoderMultimodalModel(rank=0)
    model.peer_output = torch.tensor([0.0] * 4 + [20.0] + [30.0] * 3).unsqueeze(1).repeat(1, 2)

    encoder_inputs = model.prepare_multimodal_encoder_inputs([(param, 0), (param, 1), (param, 2)])
    outputs = model.forward_multimodal_encoder_items(encoder_inputs)

    assert model.encoded_item_ids == [10]
    item_metadata = encoder_inputs[0][0].multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY]
    assert item_metadata.encoder_token_lengths == [4, 1, 3]
    assert [output.shape[0] for output in outputs] == [4, 1, 3]
    expected = torch.tensor([10.0] * 4 + [20.0] + [30.0] * 3).unsqueeze(1).repeat(1, 2)
    torch.testing.assert_close(torch.cat(outputs), expected)


def test_encoder_data_parallel_uses_physical_input_token_costs():
    param = make_encoder_dp_param_with_input_costs()
    model = DataParallelEncoderMultimodalModel(rank=0)
    model.peer_output = torch.tensor([10.0] * 4 + [0.0] + [30.0] * 3).unsqueeze(1).repeat(1, 2)

    output = model._run_multimodal_encoder([param])

    assert model.encoded_item_ids == [20]
    expected = torch.tensor([10.0] * 4 + [20.0] + [30.0] * 3).unsqueeze(1).repeat(1, 2)
    torch.testing.assert_close(output, expected)


def test_attention_data_parallel_encodes_only_local_requests_without_collective():
    model = DataParallelEncoderMultimodalModel(rank=1, enable_attention_dp=True)
    param = make_encoder_dp_param()

    output = model._run_multimodal_encoder([param])

    assert model.encoded_item_ids == [10, 20, 30]
    assert output.shape == (8, 2)


def test_encoder_data_parallel_requires_model_opt_in():
    model = DummyMultimodalModel(make_embedding(), torch.tensor([7]))
    model.model_config = ModelConfig(
        mapping=Mapping(world_size=2, rank=0, tp_size=2),
        multimodal_config=MultimodalConfig(encoder_data_parallel_size=2),
    )

    with pytest.raises(NotImplementedError, match="does not support"):
        model._run_multimodal_encoder([make_encoder_dp_param()])


def test_reorder_modality_grouped_embeddings_restores_prompt_order():
    params = [
        MultimodalParams(
            multimodal_data={
                "image": {"pixel_values": torch.empty(2, 1)},
                "video": {"pixel_values_videos": torch.empty(3, 1)},
                "multimodal_embedding_lengths": [3, 2],
            },
            mm_item_order=[
                {"modality": "video", "index": 0},
                {"modality": "image", "index": 0},
            ],
        )
    ]
    image_embeddings = torch.full((2, 1), 10.0)
    video_embeddings = torch.full((3, 1), 20.0)

    output = reorder_multimodal_embeddings_by_modality(
        params,
        ("image", "video"),
        (image_embeddings, video_embeddings),
    )

    expected = torch.tensor([[20.0], [20.0], [20.0], [10.0], [10.0]])
    torch.testing.assert_close(output, expected)


@pytest.mark.cpu_only
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


def _test_prepare_multimodal_inputs_accepts_tensor_encoder_output(device):
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


@pytest.mark.cpu_only
def test_prepare_multimodal_inputs_accepts_tensor_encoder_output_cpu():
    _test_prepare_multimodal_inputs_accepts_tensor_encoder_output("cpu")


def test_prepare_multimodal_inputs_accepts_tensor_encoder_output_cuda():
    _test_prepare_multimodal_inputs_accepts_tensor_encoder_output("cuda")


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


def test_encoder_cache_requires_model_opt_in():
    model = DummyMultimodalModel(make_embedding(hidden_size=4), torch.tensor([7]))
    model.model_config = ModelConfig(
        multimodal_config=MultimodalConfig(encoder_cache_max_bytes=4096)
    )

    assert not model.encoder_cache_active


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
        "embedding_dim=4, embedding_dtype=torch.float32" in message
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
        "mm_encoder_cache: created with max_bytes=4096, embedding row capacity unavailable "
        "because the model does not implement "
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


def test_encoder_cache_full_hit_does_not_rewrite_entries():
    model = CountingEncoderMultimodalModel(
        make_embedding(hidden_size=4),
        torch.tensor([7]),
        encoder_cache_max_bytes=4096,
    )
    model._get_or_encode_multimodal_embeddings([make_keyed_multimodal_param()])
    cache = model._multimodal_encoder_cache
    assert cache is not None

    with patch.object(cache, "put", wraps=cache.put) as put:
        model._get_or_encode_multimodal_embeddings([make_keyed_multimodal_param()])

    assert model.encode_calls == 1
    put.assert_not_called()


def test_encoder_cache_repeated_chunk_does_not_rewrite_entries():
    model = CountingEncoderMultimodalModel(
        make_embedding(hidden_size=4),
        torch.tensor([7]),
        encoder_cache_max_bytes=4096,
    )
    param = make_keyed_multimodal_param()
    first_embeddings = model._get_or_encode_multimodal_embeddings([param])
    cache = model._multimodal_encoder_cache
    assert cache is not None
    assert model.encode_calls == 1

    with patch.object(cache, "put", wraps=cache.put) as put:
        second_embeddings = model._get_or_encode_multimodal_embeddings([param])

    # We should not have called the multimodal encoder forward a second time.
    assert model.encode_calls == 1
    torch.testing.assert_close(second_embeddings, first_embeddings)
    put.assert_not_called()
    assert cache.stats().replacements == 0


def test_encoder_cache_mixed_attached_and_uncached_requests():
    model = CountingEncoderMultimodalModel(
        make_embedding(hidden_size=4),
        torch.tensor([7]),
        encoder_cache_max_bytes=4096,
    )
    local_embedding = torch.full((2, 4), 99.0)
    attached = make_keyed_multimodal_param(local_embedding=local_embedding)
    uncached = make_keyed_multimodal_param(
        item_hashes=[[9] * 8],
    )

    embeddings = model._get_or_encode_multimodal_embeddings([attached, uncached])

    assert model.encode_calls == 1
    # Since we set `local_embedding` to be `99.0` above, if we had actually called
    # `model.encode_multimodal_inputs` on it, we would have had `1.0` as the value instead for the
    # first request's embeddings.
    torch.testing.assert_close(embeddings[:2], local_embedding)
    # For the 2nd request, it should be equal to the `model.encode_calls` above.
    torch.testing.assert_close(embeddings[2:], torch.ones((2, 4)))
    cache = model._multimodal_encoder_cache
    assert cache is not None
    assert len(cache) == 1


def test_encoder_cache_partial_hit_encodes_miss_and_interleaves():
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

    # Encoder ran twice: once for the initial miss item, once for the residual containing
    # only the second request's novel item. Assembled tensor puts the cached hit before
    # the freshly encoded miss in item-index order.
    assert model.encode_calls == 2
    torch.testing.assert_close(embeddings[:2], torch.full((2, 4), 1.0))
    torch.testing.assert_close(embeddings[2:], torch.full((2, 4), 2.0))
    assert len(model._multimodal_encoder_cache) == 2
    messages = [" ".join(map(str, call.args)) for call in debug.call_args_list]
    assert any(
        "mm_encoder_cache: partial-hit encode total_items=2 hit_items=1 encoded_items=1" in msg
        for msg in messages
    )


def test_encoder_cache_partial_hit_batches_encoder_across_partial_params():
    # Two partial-hit params in the same batch must share a single encoder call so
    # launch overhead scales with iterations, not with partial-hit count.
    model = CountingEncoderMultimodalModel(
        make_embedding(hidden_size=4),
        torch.tensor([7]),
        encoder_cache_max_bytes=4096,
    )
    shared = [1, 1, 1, 1, 1, 1, 1, 1]
    seed = make_keyed_multimodal_param(item_hashes=[shared, [2] * 8], embedding_lengths=[2, 2])
    partial_a = make_keyed_multimodal_param(item_hashes=[shared, [3] * 8], embedding_lengths=[2, 2])
    partial_b = make_keyed_multimodal_param(
        item_hashes=[[2] * 8, [4] * 8], embedding_lengths=[2, 2]
    )

    model._get_or_encode_multimodal_embeddings([seed])
    encode_calls_before = model.encode_calls
    model._get_or_encode_multimodal_embeddings([partial_a, partial_b])

    # Single batched encoder call for both partial residuals.
    assert model.encode_calls == encode_calls_before + 1
    # Both new miss items (`[3]*8` and `[4]*8`) written to cache alongside the two
    # already-cached seed items.
    assert len(model._multimodal_encoder_cache) == 4


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
    cache = model._multimodal_encoder_cache
    assert cache is not None

    with patch.object(cache, "put", wraps=cache.put) as put:
        embeddings = model._get_or_encode_multimodal_embeddings([chunk_param])

    assert model.encode_calls == 1
    torch.testing.assert_close(embeddings, local_embedding)
    put.assert_not_called()
    assert cache.stats().replacements == 0


def test_partition_encoder_cache_dispatches_by_hit_outcome():
    model = CountingEncoderMultimodalModel(
        make_embedding(hidden_size=4),
        torch.tensor([7]),
        encoder_cache_max_bytes=4096,
    )
    seeded = make_keyed_multimodal_param(item_hashes=[[1] * 8, [2] * 8], embedding_lengths=[2, 2])
    model._get_or_encode_multimodal_embeddings([seeded])
    cache = model._multimodal_encoder_cache

    full_hit = make_keyed_multimodal_param(item_hashes=[[1] * 8, [2] * 8], embedding_lengths=[2, 2])
    part = model.partition_encoder_cache(full_hit, cache)
    assert part.is_full_hit and not part.is_full_miss and part.miss_indices == []

    full_miss = make_keyed_multimodal_param(
        item_hashes=[[8] * 8, [9] * 8], embedding_lengths=[2, 2]
    )
    part = model.partition_encoder_cache(full_miss, cache)
    assert part.is_full_miss and not part.is_full_hit and part.hits == {}

    partial = make_keyed_multimodal_param(item_hashes=[[1] * 8, [3] * 8], embedding_lengths=[2, 2])
    part = model.partition_encoder_cache(partial, cache)
    assert not part.is_full_hit and not part.is_full_miss
    assert list(part.hits) == [0] and part.miss_indices == [1]


def test_assemble_full_embedding_preserves_item_order():
    per_item = {
        0: torch.tensor([[0.0]]),
        1: torch.tensor([[1.0], [1.5]]),
        2: torch.tensor([[2.0]]),
    }
    torch.testing.assert_close(
        _assemble_multimodal_encoder_embeddings(per_item, 3),
        torch.tensor([[0.0], [1.0], [1.5], [2.0]]),
    )
    # Even a single item is copied into request-owned storage. The sources here
    # are cache entries, which `TensorLRUCache.get` returns as aliases of
    # cache-owned tensors; handing one straight to a request would leave the two
    # sharing storage and the cache's byte accounting short by an entry it can
    # no longer actually free.
    single = per_item[1]
    assembled = _assemble_multimodal_encoder_embeddings({0: single}, 1)
    assert assembled is not single
    torch.testing.assert_close(assembled, single)


def test_assemble_full_embedding_rejects_incompatible_items():
    with pytest.raises(ValueError, match="matching output shape, dtype, and device"):
        _assemble_multimodal_encoder_embeddings({0: torch.ones(1, 2), 1: torch.ones(1, 3)}, 2)


def test_build_multimodal_encoder_input_slices_packed_grid_thw():
    # Qwen3-VL-style layout: `pixel_values` is a single packed tensor sized by the
    # cumulative patch counts declared in `image_grid_thw`. `second_per_grid_ts`
    # stands in for any per-item sibling field that must stay in sync with the
    # sliced items; `per_request_scalar` stands in for non-per-item siblings that
    # must pass through unchanged.
    grids = torch.tensor([[1, 1, 2], [1, 1, 3], [1, 1, 1]])  # 2 + 3 + 1 patches
    pixels = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    per_item_meta = torch.tensor([0.1, 0.2, 0.3])
    param = MultimodalParams(
        multimodal_input=MultimodalInput(
            multimodal_hashes=[[i] * 8 for i in range(3)],
            multimodal_positions=[0, 0, 0],
            multimodal_lengths=[2, 3, 1],
        ),
        multimodal_data={
            "image": {
                "pixel_values": pixels,
                "image_grid_thw": grids,
                "second_per_grid_ts": per_item_meta,
                "per_item_list": ["a", "b", "c"],
                "per_request_scalar": torch.tensor(42.0),
            },
            "multimodal_embedding_lengths": [2, 3, 1],
            "mm_processor_kwargs_hash": "kw",
        },
    )
    model = DummyMultimodalModel(make_embedding(hidden_size=1), torch.tensor([0]))

    residual = model.build_multimodal_encoder_input(param, [2, 0])

    # Item 2 spans rows [5], item 0 spans rows [0, 1]; residual concatenates them in
    # the requested item order and slices every parallel sibling the same way.
    residual_image = residual.multimodal_data["image"]
    torch.testing.assert_close(
        residual_image["pixel_values"], torch.cat([pixels[5:6], pixels[0:2]], dim=0)
    )
    torch.testing.assert_close(residual_image["image_grid_thw"], grids[[2, 0]])
    torch.testing.assert_close(residual_image["second_per_grid_ts"], per_item_meta[[2, 0]])
    assert residual_image["per_item_list"] == ["c", "a"]
    torch.testing.assert_close(residual_image["per_request_scalar"], torch.tensor(42.0))


def test_build_multimodal_encoder_input_stacked_crops_padding_to_miss_max_size():
    # `pixel_values` is padded to request-wide 5x5 but item 0's true size is (3, 4)
    # and item 1's is (5, 5). Slicing to just item 0 must crop `pixel_values` down to
    # (3, 4) so Mistral 3's `batch_pixel_values` (which re-pads to
    # `max(image_sizes)`) doesn't apply a negative pad amount.
    pixels = torch.arange(2 * 3 * 5 * 5, dtype=torch.float32).reshape(2, 3, 5, 5)
    param = MultimodalParams(
        multimodal_input=MultimodalInput(
            multimodal_hashes=[[i] * 8 for i in range(2)],
            multimodal_positions=[0, 0],
            multimodal_lengths=[1, 1],
        ),
        multimodal_data={
            "image": {
                "pixel_values": pixels,
                "image_sizes": [[3, 4], [5, 5]],
            },
            "multimodal_embedding_lengths": [1, 1],
            "mm_processor_kwargs_hash": "kw",
        },
    )
    model = DummyMultimodalModel(make_embedding(hidden_size=1), torch.tensor([0]))

    residual = model.build_multimodal_encoder_input(param, [0])

    residual_image = residual.multimodal_data["image"]
    assert residual_image["image_sizes"] == [[3, 4]]
    assert residual_image["pixel_values"].shape == (1, 3, 3, 4)
    # Cropped tensor preserves item 0's top-left (H=0..3, W=0..4) window.
    torch.testing.assert_close(residual_image["pixel_values"], pixels[0:1, :, :3, :4])


def test_build_multimodal_encoder_input_slices_audio_input_features():
    # Whisper / Gemma4-audio layout: `input_features [B, mel, T]` stacked on
    # dim 0, with an optional per-item mask that sibling-slices automatically.
    # Two clips: item 0 and item 1 -- slice to [1, 0] to also confirm item order
    # is preserved.
    features = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    param = MultimodalParams(
        multimodal_input=MultimodalInput(
            multimodal_hashes=[[i] * 8 for i in range(2)],
            multimodal_positions=[0, 0],
            multimodal_lengths=[1, 1],
        ),
        multimodal_data={
            "audio": {
                "input_features": features,
                "input_features_mask": mask,
            },
            "multimodal_embedding_lengths": [1, 1],
            "mm_processor_kwargs_hash": "kw",
        },
    )
    model = DummyMultimodalModel(make_embedding(hidden_size=1), torch.tensor([0]))

    residual = model.build_multimodal_encoder_input(param, [1, 0])

    residual_audio = residual.multimodal_data["audio"]
    torch.testing.assert_close(residual_audio["input_features"], features[[1, 0]])
    # Per-item mask is caught by the generic sibling-slice pass.
    torch.testing.assert_close(residual_audio["input_features_mask"], mask[[1, 0]])


@pytest.mark.parametrize(
    "mm_data, expected_match",
    [
        # `_encoder_cache_modality` returns None -> single-modality guard fires.
        ({}, "cannot infer the modality"),
        # Modality present but layout is neither pattern A (image_sizes) nor pattern B
        # (grid_thw); default has nothing to dispatch on.
        ({"image": {"pixel_values": torch.zeros(2)}}, "cannot slice image layout"),
        # Audio modality but no `input_features`; default falls through.
        ({"audio": {"nonsense": torch.zeros(2)}}, "cannot slice audio layout"),
    ],
    ids=["no_modality", "unhandled_image_layout", "unhandled_audio_layout"],
)
def test_build_multimodal_encoder_input_unhandled_layout_raises(mm_data, expected_match):
    param = MultimodalParams(multimodal_data=mm_data)
    model = DummyMultimodalModel(make_embedding(hidden_size=1), torch.tensor([0]))
    with pytest.raises(NotImplementedError, match=expected_match):
        model.build_multimodal_encoder_input(param, [0])
