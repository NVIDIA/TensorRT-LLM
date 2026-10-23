# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle
import sys
from contextlib import nullcontext
from functools import partial
from unittest.mock import patch

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.inputs.registry import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalEncoderItemMetadata,
)
from tensorrt_llm.llmapi.llm_args import MultimodalConfig
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

pytestmark = pytest.mark.threadleak(enabled=False)


class _EncoderDpModel(torch.nn.Module, MultimodalModelMixin):
    def __init__(self, rank: int, *, enable_attention_dp: bool) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            mapping=Mapping(
                world_size=2,
                rank=rank,
                tp_size=2,
                enable_attention_dp=enable_attention_dp,
            ),
            multimodal_config=MultimodalConfig(
                encoder_data_parallel_size=2,
                encoder_cache_max_bytes=0,
            ),
        )
        self.embedding = torch.nn.Embedding(
            num_embeddings=8,
            embedding_dim=2,
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.encoded_item_ids: list[int] = []

    @property
    def text_embedding_layer(self) -> torch.nn.Embedding:
        return self.embedding

    @property
    def embedding_dim(self) -> int:
        return self.embedding.embedding_dim

    @property
    def embedding_dtype(self) -> torch.dtype:
        return self.embedding.weight.dtype

    def encode_multimodal_inputs(
        self,
        multimodal_params: list[MultimodalParams],
        **encoder_kwargs,
    ) -> torch.Tensor:
        rows = []
        for param in multimodal_params:
            item_ids = param.multimodal_data["image"]["pixel_values"][:, 0, 0, 0]
            lengths = param.multimodal_data["multimodal_embedding_lengths"]
            self.encoded_item_ids.extend(int(item_id) for item_id in item_ids)
            for item_id, length in zip(item_ids, lengths, strict=True):
                rows.append(
                    torch.full(
                        (length, self.embedding_dim),
                        float(item_id),
                        dtype=self.embedding_dtype,
                        device="cuda",
                    )
                )
        return torch.cat(rows, dim=0)


def _make_param(item_ids: list[int], lengths: list[int]) -> MultimodalParams:
    return MultimodalParams(
        multimodal_data={
            "image": {
                "pixel_values": torch.tensor(item_ids, dtype=torch.float32).reshape(-1, 1, 1, 1),
                "image_sizes": [[1, 1] for _ in item_ids],
            },
            "multimodal_embedding_lengths": lengths,
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalEncoderItemMetadata(
                item_refs=[("image", i) for i in range(len(item_ids))],
                encoder_token_lengths=lengths,
                output_embedding_lengths=lengths,
            ),
        }
    )


@torch.inference_mode()
def _run_encoder_dp_rank(enable_attention_dp: bool, lengths: list[int] | None = None):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    model = _EncoderDpModel(rank, enable_attention_dp=enable_attention_dp)

    if enable_attention_dp:
        param = _make_param([10 + rank], [rank + 1])
    else:
        lengths = [4, 1, 3] if lengths is None else lengths
        param = _make_param([10, 20, 30][: len(lengths)], lengths)

    output = model._run_multimodal_encoder([param])
    model.encoded_item_ids.clear()
    previous_sync_debug_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        output = model._run_multimodal_encoder([param])
    finally:
        torch.cuda.set_sync_debug_mode(previous_sync_debug_mode)
    torch.cuda.synchronize()
    return rank, model.encoded_item_ids, output.cpu()


@torch.inference_mode()
def _run_encoder_dp_failure_rank(_):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    model = _EncoderDpModel(rank, enable_attention_dp=False)
    param = _make_param([10, 20, 30], [4, 1, 3])
    failure = (
        patch.object(model, "encode_multimodal_inputs", side_effect=ValueError("encoder failed"))
        if rank == 0
        else nullcontext()
    )
    with failure, pytest.raises(RuntimeError) as error:
        model._run_multimodal_encoder([param])
    message = str(error.value)
    cause = type(error.value.__cause__).__name__

    output = model._run_multimodal_encoder([param])
    return rank, message, cause, output.cpu()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_encoder_dp_propagates_rank_failure_and_recovers(mpi_pool_executor):
    results = sorted(mpi_pool_executor.map(_run_encoder_dp_failure_rank, [None, None]))

    assert "rank failed" in results[0][1]
    assert results[0][2] == "ValueError"
    assert "peer rank failed" in results[1][1]
    torch.testing.assert_close(results[0][3], results[1][3])
    assert results[0][3].shape == (8, 2)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize(
    "lengths,expected_item_ids",
    [([4, 1, 3], [[10], [20, 30]]), ([2, 4, 1], [[20], [10, 30]]), ([4], [[10], []])],
    ids=["equal_rows", "unequal_rows_reordered", "empty_rank"],
)
def test_encoder_dp_partitions_and_reconstructs_with_nccl(
    mpi_pool_executor, lengths, expected_item_ids
):
    results = mpi_pool_executor.map(partial(_run_encoder_dp_rank, lengths=lengths), [False, False])
    results = sorted(results)

    assert [result[1] for result in results] == expected_item_ids
    expected = (
        torch.tensor(
            [
                item_id
                for item_id, length in zip([10.0, 20.0, 30.0], lengths)
                for _ in range(length)
            ],
            dtype=torch.bfloat16,
        )
        .unsqueeze(1)
        .repeat(1, 2)
    )
    for _, _, output in results:
        torch.testing.assert_close(output, expected)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_attention_dp_encodes_rank_local_requests(mpi_pool_executor):
    results = mpi_pool_executor.map(_run_encoder_dp_rank, [True, True])
    results = sorted(results)

    assert results[0][1] == [10]
    assert results[1][1] == [11]
    torch.testing.assert_close(
        results[0][2],
        torch.full((1, 2), 10.0, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        results[1][2],
        torch.full((2, 2), 11.0, dtype=torch.bfloat16),
    )


class _QwenVisionEncoderModel(_EncoderDpModel):
    """Use the production Qwen2.5-VL tower and weight loader, with small dimensions."""

    multimodal_data_device_paths = ["image.pixel_values"]

    def __init__(self, rank, attention_dp, encoder_dp, pretrained_config, weights):
        from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
            make_multimodal_encoder_model_config,
        )
        from tensorrt_llm._torch.models.modeling_qwen2vl import (
            Qwen2_5_VisionModel,
            Qwen2VisionModelBase,
        )

        super().__init__(rank, enable_attention_dp=attention_dp)
        self.model_config = ModelConfig(
            pretrained_config=pretrained_config,
            mapping=Mapping(world_size=2, rank=rank, tp_size=2, enable_attention_dp=attention_dp),
            multimodal_config=MultimodalConfig(encoder_data_parallel_size=encoder_dp),
            attn_backend="TRTLLM",
            max_num_tokens=256,
        )
        self.embedding = torch.nn.Embedding(8, 64, dtype=torch.bfloat16, device="cuda")
        self.tower = Qwen2VisionModelBase(
            make_multimodal_encoder_model_config(self.model_config), Qwen2_5_VisionModel
        ).to("cuda")
        self.tower.load_weights(weights)
        self.tower.visual.setup_attn_metadata(max_num_tokens=256)

    def encode_multimodal_inputs(self, multimodal_params, **kwargs):
        return torch.cat(self.tower(multimodal_params), dim=0)


def _make_qwen_vision_param(rank):
    grids = [[1, 4, 4], [1, 4, 8]] if rank == 0 else [[1, 8, 4]]
    lengths = [t * h * w // 4 for t, h, w in grids]
    return MultimodalParams(
        multimodal_data={
            "image": {
                "pixel_values": torch.randn(
                    sum(lengths) * 4, 24, generator=torch.Generator().manual_seed(100 + rank)
                ).cuda(),
                "image_grid_thw": torch.tensor(grids),
            },
            "multimodal_embedding_lengths": lengths,
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalEncoderItemMetadata(
                item_refs=[("image", i) for i in range(len(grids))],
                encoder_token_lengths=[length * 4 for length in lengths],
                output_embedding_lengths=lengths,
            ),
        }
    )


@torch.inference_mode()
def _run_qwen_vision_parallel_rank(config):
    from transformers import Qwen2_5_VLConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel,
    )

    attention_dp, encoder_dp = config
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    torch.manual_seed(17)
    pretrained_config = Qwen2_5_VLConfig(
        vision_config=dict(
            depth=2,
            hidden_size=128,
            intermediate_size=256,
            num_heads=4,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=64,
            window_size=8,
            fullatt_block_indexes=[1],
            dtype="bfloat16",
        ),
        text_config=dict(max_position_embeddings=2048, rms_norm_eps=1e-6),
        dtype="bfloat16",
    )
    hf_tower = Qwen2_5_VisionTransformerPretrainedModel(pretrained_config.vision_config)
    weights = {"visual." + name: value for name, value in hf_tower.state_dict().items()}
    reference = _QwenVisionEncoderModel(rank, False, 2, pretrained_config, weights)
    model = _QwenVisionEncoderModel(rank, attention_dp, encoder_dp, pretrained_config, weights)
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.tower.parameters())

    # Different ADP owners have different image counts and token counts. TP
    # ranks instead receive the same requests. Then test an empty/cache-hit
    # owner and an iteration where every owner has no encoder work.
    param = _make_qwen_vision_param(rank if attention_dp else 0)
    lengths = param.multimodal_data["multimodal_embedding_lengths"]
    expected = reference.encode_multimodal_inputs([param])
    prepared = model.prepare_multimodal_encoder_inputs([(param, i) for i in range(len(lengths))])
    actual = torch.cat(model.forward_multimodal_encoder_items(prepared), dim=0)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    if attention_dp:
        outputs = model.forward_multimodal_encoder_items(prepared if rank == 0 else [])
        if rank == 0:
            torch.testing.assert_close(torch.cat(outputs), expected, atol=2e-2, rtol=2e-2)
        else:
            assert outputs == []
        assert model.forward_multimodal_encoder_items([]) == []
    # Exercise the production scheduler/cache publication boundary as well as
    # the direct model API. ADP peers with no producers still participate.
    from tensorrt_llm._torch.pyexecutor.engine.multimodal import MultimodalItemScheduler
    from tensorrt_llm._torch.pyexecutor.llm_request import (
        LlmRequest,
        initialize_multimodal_encoder_request,
        make_mm_encoder_transient_cache_key,
    )
    from tensorrt_llm._torch.tensor_lru_cache import TensorLRUCache
    from tensorrt_llm.bindings import SamplingConfig

    model._multimodal_encoder_cache = TensorLRUCache(1 << 20)
    scheduler = MultimodalItemScheduler(
        model=model,
        bytes_per_embedding=model.embedding_dim * model.embedding_dtype.itemsize,
    )
    request = LlmRequest(
        request_id=rank + 1,
        max_new_tokens=1,
        input_tokens=[1, 2, 3],
        sampling_config=SamplingConfig(),
        is_streaming=False,
        py_multimodal_data=param.multimodal_data,
    )
    initialize_multimodal_encoder_request(request, max_num_tokens=256)
    state = request.py_mm_encoder_state
    for item_idx, rows in enumerate(lengths):
        key = make_mm_encoder_transient_cache_key(request.request_id, item_idx)
        assert model._multimodal_encoder_cache.acquire(key, rows * 128, retain_after_release=False)
        state.set_item_cache_key(item_idx, key, ready=False)
    scheduled = {request.request_id: list(range(len(lengths)))}
    if attention_dp and rank == 1:
        # A rank with no scheduled work participates without publishing output.
        scheduler.forward_items([], {})
    else:
        scheduler.forward_items([request], scheduled)
        cached = torch.cat(
            [model._multimodal_encoder_cache.get(key) for key in state.item_cache_keys]
        )
        torch.testing.assert_close(cached, expected, atol=2e-2, rtol=2e-2)
        assert all(state.item_ready)
    if attention_dp:
        scheduler.forward_items([], {})
    return parameter_bytes


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("attention_dp", [False, True], ids=["llm_tp", "llm_adp"])
@pytest.mark.parametrize("encoder_dp", [1, 2], ids=["encoder_tp", "encoder_dp"])
def test_qwen_vision_parallel_combinations(mpi_pool_executor, attention_dp, encoder_dp):
    results = list(
        mpi_pool_executor.map(_run_qwen_vision_parallel_rank, [(attention_dp, encoder_dp)] * 2)
    )
    assert results[0] == results[1] > 0
