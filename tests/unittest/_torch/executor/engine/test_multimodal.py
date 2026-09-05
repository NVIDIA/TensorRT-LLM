# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from _torch.executor.multimodal_utils import (
    bare_mm_item_scheduler,
    make_llm_request,
    make_mm_request,
)

from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    MultimodalEncoderContractError,
    MultimodalModelMixin,
)
from tensorrt_llm._torch.pyexecutor.engine.multimodal import (
    MultimodalItemScheduler,
    resolve_bytes_per_mm_encoder_embedding,
    resolve_mm_encoder_output_budget,
    validate_mm_encoder_scheduling_compatibility,
)
from tensorrt_llm._torch.pyexecutor.llm_request import (
    MultimodalEncoderRequestError,
    initialize_multimodal_encoder_request,
    is_multimodal_encoder_ready,
    make_mm_encoder_transient_cache_key,
)
from tensorrt_llm.inputs.multimodal import MULTIMODAL_ENCODER_ITEM_METADATA_KEY, MultimodalParams
from tensorrt_llm.inputs.registry import MultimodalEncoderItemMetadata
from tensorrt_llm.llmapi.llm_args import MultimodalEncoderSchedulingPolicy

# The item-scheduling surface is pure logic: no kernels, no device transfers.
pytestmark = pytest.mark.cpu_only


def _bind_items(mm_item_scheduler: MultimodalItemScheduler, request, *, row_bytes: int = 8) -> None:
    state = request.py_mm_encoder_state
    for item_idx, rows in enumerate(state.embedding_lengths):
        cache_key = make_mm_encoder_transient_cache_key(request.request_id, item_idx)
        assert mm_item_scheduler.encoder_cache.acquire(
            cache_key, rows * row_bytes, retain_after_release=False
        )
        state.set_item_cache_key(item_idx, cache_key, ready=False)


def test_qwen3_output_budget_uses_post_merge_embedding_capacity() -> None:
    from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLInputProcessorBase

    processor = object.__new__(Qwen3VLInputProcessorBase)
    processor._config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
    model = SimpleNamespace(embedding_dim=16384, embedding_dtype=torch.float16)

    budget, bytes_per_embedding = resolve_mm_encoder_output_budget(processor, 65536, model)

    assert bytes_per_embedding == 32768
    assert budget == 512 * 1024**2


def test_output_row_bytes_use_config_dtype_without_embedding_weight() -> None:
    model = SimpleNamespace(
        embedding_dim=16384,
        model_config=SimpleNamespace(torch_dtype=torch.bfloat16),
    )

    assert resolve_bytes_per_mm_encoder_embedding(model) == 32768


def test_output_budget_requires_processor_embedding_capacity() -> None:
    processor = SimpleNamespace(get_max_mm_encoder_output_embeddings=lambda *_: None)

    with pytest.raises(ValueError, match="get_max_mm_encoder_output_embeddings"):
        resolve_mm_encoder_output_budget(processor, 65536, None)


def test_eager_compatibility_is_checked_only_for_item_scheduled_models() -> None:
    args = SimpleNamespace(
        multimodal_config=SimpleNamespace(
            encoder_scheduling_policy=MultimodalEncoderSchedulingPolicy.EAGER,
            encoder_side_stream_max_ahead=0,
        ),
        pipeline_parallel_size=1,
        enable_attention_dp=True,
        cache_transceiver_config=SimpleNamespace(backend="NIXL"),
    )

    validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=False)

    with pytest.raises(ValueError, match="attention DP"):
        validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=True)

    args.enable_attention_dp = False
    with pytest.raises(ValueError, match="disaggregated"):
        validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=True)


def test_side_stream_compatibility_is_checked_only_for_item_scheduled_models() -> None:
    args = SimpleNamespace(
        multimodal_config=SimpleNamespace(
            encoder_scheduling_policy=MultimodalEncoderSchedulingPolicy.DEFAULT,
            encoder_side_stream_max_ahead=1,
        ),
        pipeline_parallel_size=1,
        enable_attention_dp=False,
        cache_transceiver_config=None,
    )

    validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=False)

    with pytest.raises(ValueError, match="side-stream prefetch"):
        validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=True)


def test_pipeline_parallel_compatibility_is_checked_only_for_item_scheduled_models() -> None:
    args = SimpleNamespace(
        multimodal_config=SimpleNamespace(
            encoder_scheduling_policy=MultimodalEncoderSchedulingPolicy.DEFAULT,
            encoder_side_stream_max_ahead=0,
        ),
        pipeline_parallel_size=2,
        enable_attention_dp=False,
        cache_transceiver_config=None,
    )

    validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=False)

    with pytest.raises(ValueError, match="pipeline parallelism"):
        validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=True)


def test_item_encoder_classifies_request_state_contract_errors() -> None:
    mm_item_scheduler = bare_mm_item_scheduler(MultimodalModelMixin())
    request = make_llm_request(1)

    with pytest.raises(MultimodalEncoderRequestError, match="no longer active"):
        mm_item_scheduler.forward_items([], {request.request_id: [0]})

    with pytest.raises(MultimodalEncoderRequestError, match="no encoder item state"):
        mm_item_scheduler.forward_items([request], {request.request_id: [0]})


def test_item_encoder_classifies_output_count_contract_error() -> None:
    class _Model(MultimodalModelMixin):
        def prepare_multimodal_encoder_inputs(self, _):
            encoder_input = SimpleNamespace(to_device=lambda *_args, **_kwargs: None)
            return [(encoder_input, [1], "image")]

        def forward_multimodal_encoder_items(self, _):
            return []

    mm_item_scheduler = bare_mm_item_scheduler(_Model())
    request = make_mm_request(1, [4])
    _bind_items(mm_item_scheduler, request)

    with pytest.raises(MultimodalEncoderRequestError, match="one output per item"):
        mm_item_scheduler.forward_items([request], {request.request_id: [0]})


@pytest.mark.parametrize("failure_stage", ["prepare", "forward"])
def test_item_encoder_translates_model_contract_errors(failure_stage: str) -> None:
    class _Model(MultimodalModelMixin):
        def prepare_multimodal_encoder_inputs(self, _):
            if failure_stage == "prepare":
                raise MultimodalEncoderContractError("bad request metadata")
            encoder_input = SimpleNamespace(to_device=lambda *_args, **_kwargs: None)
            return [(encoder_input, [1], "image")]

        def forward_multimodal_encoder_items(self, _):
            raise MultimodalEncoderContractError("bad encoder output rows")

    mm_item_scheduler = bare_mm_item_scheduler(_Model())
    request = make_mm_request(1, [4])
    _bind_items(mm_item_scheduler, request)

    expected = "bad request metadata" if failure_stage == "prepare" else "bad encoder output rows"
    with pytest.raises(MultimodalEncoderRequestError, match=expected):
        mm_item_scheduler.forward_items([request], {request.request_id: [0]})


@pytest.mark.parametrize("failure_stage", ["prepare", "forward"])
def test_item_encoder_does_not_translate_system_errors(failure_stage: str) -> None:
    class _Model(MultimodalModelMixin):
        def prepare_multimodal_encoder_inputs(self, _):
            if failure_stage == "prepare":
                raise torch.cuda.OutOfMemoryError("encoder OOM")
            encoder_input = SimpleNamespace(to_device=lambda *_args, **_kwargs: None)
            return [(encoder_input, [1], "image")]

        def forward_multimodal_encoder_items(self, _):
            raise torch.cuda.OutOfMemoryError("encoder OOM")

    mm_item_scheduler = bare_mm_item_scheduler(_Model())
    request = make_mm_request(1, [4])
    _bind_items(mm_item_scheduler, request)

    with pytest.raises(torch.cuda.OutOfMemoryError, match="encoder OOM"):
        mm_item_scheduler.forward_items([request], {request.request_id: [0]})


def test_item_outputs_commit_to_prompt_ordered_cache_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Model(MultimodalModelMixin):
        def forward_multimodal_encoder_items(self, encoder_inputs):
            return [
                torch.full((embedding_length, 2), float(embedding_length))
                for _, embedding_lengths, _ in encoder_inputs
                for embedding_length in embedding_lengths
            ]

    monkeypatch.setattr(MultimodalParams, "to_device", lambda self, *args, **kwargs: self)
    mm_item_scheduler = bare_mm_item_scheduler(_Model())
    request = make_llm_request(
        1,
        multimodal_data={
            "image": {
                "pixel_values": torch.arange(5).unsqueeze(1),
                "image_grid_thw": torch.tensor([[1, 1, 2], [1, 1, 3]]),
            },
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalEncoderItemMetadata(
                item_refs=[("image", 0), ("image", 1)],
                encoder_token_lengths=[2, 3],
                output_embedding_lengths=[2, 3],
            ),
            "multimodal_embedding_lengths": [2, 3],
        },
    )
    initialize_multimodal_encoder_request(request, max_num_tokens=8)
    _bind_items(mm_item_scheduler, request)
    state = request.py_mm_encoder_state

    mm_item_scheduler.forward_items([request], {request.request_id: [0]})

    assert state.item_ready == [True, False]
    first = mm_item_scheduler.encoder_cache.get(state.item_cache_keys[0])
    torch.testing.assert_close(first, torch.full((2, 2), 2.0))
    assert "image" in request.py_multimodal_data

    mm_item_scheduler.forward_items([request], {request.request_id: [1]})

    second = mm_item_scheduler.encoder_cache.get(state.item_cache_keys[1])
    torch.testing.assert_close(second, torch.full((3, 2), 3.0))
    assert "multimodal_embedding" not in request.py_multimodal_data
    assert "image" not in request.py_multimodal_data
    assert is_multimodal_encoder_ready(request)

    multimodal_data = mm_item_scheduler.build_multimodal_data_for_llm(request)
    torch.testing.assert_close(
        multimodal_data["multimodal_embedding"],
        torch.cat([torch.full((2, 2), 2.0), torch.full((3, 2), 3.0)]),
    )
