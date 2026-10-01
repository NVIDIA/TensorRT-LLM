# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

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
    setup_mm_encoder_attn_metadata,
    validate_mm_encoder_scheduling_compatibility,
)
from tensorrt_llm._torch.pyexecutor.llm_request import (
    MultimodalEncoderRequestError,
    initialize_multimodal_encoder_request,
    is_multimodal_encoder_ready,
    make_mm_encoder_transient_cache_key,
)
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalParams,
    MultimodalRuntimeData,
)
from tensorrt_llm.inputs.registry import (
    BaseMultimodalDummyInputsBuilder,
    MultimodalEncoderItemMetadata,
)
from tensorrt_llm.llmapi.llm_args import MultimodalEncoderSchedulingPolicy

# The item-scheduling surface is pure logic: no kernels, no device transfers.
pytestmark = pytest.mark.cpu_only


def test_disabled_encoder_does_not_initialize_processor():
    processor = Mock(spec=BaseMultimodalDummyInputsBuilder)
    processor.get_mm_max_tokens_per_item.side_effect = AssertionError("encoder is disabled")
    model = torch.nn.Module()
    model.mm_encoder = None
    setup_mm_encoder_attn_metadata(model, processor, 1024, None)
    processor.get_mm_max_tokens_per_item.assert_not_called()


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
    # A follower of the scheduled entry keeps its reference to produce it later.
    follower = make_mm_request(2, [4])
    shared_key = request.py_mm_encoder_state.item_cache_keys[0]
    assert mm_item_scheduler.encoder_cache.acquire(shared_key, 8, retain_after_release=False)
    follower.py_mm_encoder_state.set_item_cache_key(0, shared_key, ready=False)

    expected = "bad request metadata" if failure_stage == "prepare" else "bad encoder output rows"
    with pytest.raises(MultimodalEncoderRequestError, match=expected) as error:
        mm_item_scheduler.forward_items([request, follower], {request.request_id: [0]})

    assert error.value.request_ids == {request.request_id}


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


@pytest.mark.parametrize("invalid_output", ["rows", "width", "dtype", "rank", "type"])
def test_item_encoder_validates_all_outputs_before_commit(invalid_output: str) -> None:
    class _Model(MultimodalModelMixin):
        embedding_dtype = torch.bfloat16

        def prepare_multimodal_encoder_inputs(self, _):
            return []

        def forward_multimodal_encoder_items(self, _):
            return outputs

    outputs = [torch.ones(1, 4, dtype=torch.bfloat16) for _ in range(3)]
    outputs[1] = {
        "rows": torch.ones(2, 4, dtype=torch.bfloat16),
        "width": torch.ones(1, 3, dtype=torch.bfloat16),
        # Same byte size as bfloat16: byte accounting alone cannot catch this.
        "dtype": torch.ones(1, 4, dtype=torch.float16),
        "rank": torch.ones(4, dtype=torch.bfloat16),
        "type": None,
    }[invalid_output]
    actual = {
        "rows": "a torch.bfloat16 tensor with shape (2, 4)",
        "width": "a torch.bfloat16 tensor with shape (1, 3)",
        "dtype": "a torch.float16 tensor with shape (1, 4)",
        "rank": "a torch.bfloat16 tensor with shape (4,)",
        "type": "NoneType",
    }[invalid_output]
    scheduler = bare_mm_item_scheduler(_Model())
    scheduler.bytes_per_embedding = 8
    producers = [make_mm_request(request_id, [4]) for request_id in (1, 2, 3)]
    for request in producers:
        _bind_items(scheduler, request)
    follower = make_mm_request(4, [4])
    shared_key = producers[1].py_mm_encoder_state.item_cache_keys[0]
    assert scheduler.encoder_cache.acquire(shared_key, 8, retain_after_release=False)
    follower.py_mm_encoder_state.set_item_cache_key(0, shared_key, ready=False)
    requests = [*producers, follower]
    selected = {request.request_id: [0] for request in producers}

    with pytest.raises(MultimodalEncoderRequestError, match="must produce") as error:
        scheduler.forward_items(requests, selected)

    assert str(error.value).endswith(f"got {actual}")
    assert error.value.request_ids == {2, 4}
    assert scheduler.encoder_cache.current_bytes == 0
    for request in requests:
        state = request.py_mm_encoder_state
        assert state.item_ready == [False]
        assert state.item_cache_keys[0] is not None
        assert "image" in request.py_multimodal_data

    outputs[1] = torch.ones(1, 4, dtype=torch.bfloat16)
    scheduler.forward_items(requests, selected)
    assert scheduler.encoder_cache.current_bytes == 24
    assert all(is_multimodal_encoder_ready(request) for request in requests)
    assert all("image" not in request.py_multimodal_data for request in requests)


@pytest.mark.parametrize("window", [None, (0, 3), (2, 7), (6, 8), (3, 5)])
def test_item_outputs_commit_to_prompt_ordered_cache_keys(
    monkeypatch: pytest.MonkeyPatch,
    window: tuple[int, int] | None,
) -> None:
    class _Model(MultimodalModelMixin):
        embedding_dtype = torch.float32

        def forward_multimodal_encoder_items(self, encoder_inputs):
            return [
                torch.full((embedding_length, 2), float(embedding_length))
                for _, embedding_lengths, _ in encoder_inputs
                for embedding_length in embedding_lengths
            ]

    monkeypatch.setattr(MultimodalParams, "to_device", lambda self, *args, **kwargs: self)
    mm_item_scheduler = bare_mm_item_scheduler(_Model())
    mm_item_scheduler.bytes_per_embedding = 8
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

    runtime = None
    if window is not None:
        # Two image-A rows, a text-only gap, then three image-B rows.
        runtime = MultimodalRuntimeData(
            embed_mask_cumsum=torch.tensor([0, 1, 2, 2, 2, 3, 4, 5]),
            past_seen_token_num=window[0],
            chunk_end_pos=window[1],
        )
    start = runtime.num_cached_mm_tokens if runtime is not None else 0
    end = start + runtime.num_mm_tokens_in_chunk if runtime is not None else 5
    expected = torch.cat([first, second])[start:end]
    with patch("torch.cat", wraps=torch.cat) as join:
        multimodal_data = mm_item_scheduler.build_multimodal_data_for_llm(request, runtime)
    output = multimodal_data["multimodal_embedding"]
    torch.testing.assert_close(
        output,
        expected,
    )
    assert join.call_count == int(start < 2 < end)
    if start < end and join.call_count == 0:
        source = first if end <= 2 else second
        assert output.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    assert multimodal_data["multimodal_embedding_is_chunk"]
    assert "multimodal_embedding_is_chunk" not in request.py_multimodal_data
    assert "multimodal_embedding" not in request.py_multimodal_data
    assert mm_item_scheduler.encoder_cache.current_bytes == 40


def test_cache_hit_only_request_omits_raw_inputs_from_llm_payload() -> None:
    class _Model(MultimodalModelMixin):
        embedding_dtype = torch.float32

    mm_item_scheduler = bare_mm_item_scheduler(_Model())
    mm_item_scheduler.bytes_per_embedding = 8
    request = make_llm_request(
        1,
        multimodal_data={
            "image": {
                "pixel_values": torch.arange(2).unsqueeze(1),
                "image_grid_thw": torch.tensor([[1, 1, 2]]),
            },
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalEncoderItemMetadata(
                item_refs=[("image", 0)],
                encoder_token_lengths=[2],
                output_embedding_lengths=[2],
            ),
            "multimodal_embedding_lengths": [2],
        },
    )
    initialize_multimodal_encoder_request(request, max_num_tokens=8)
    # Bind the item to an already committed entry, as a scheduler cache hit
    # does, without running `forward_items`.
    encoder_cache = mm_item_scheduler.encoder_cache
    cache_key = make_mm_encoder_transient_cache_key(request.request_id, 0)
    assert encoder_cache.acquire(cache_key, 16, retain_after_release=False)
    encoder_cache.commit(cache_key, torch.ones(2, 2))
    request.py_mm_encoder_state.set_item_cache_key(0, cache_key, ready=True)
    assert is_multimodal_encoder_ready(request)

    multimodal_data = mm_item_scheduler.build_multimodal_data_for_llm(request)

    assert "image" not in multimodal_data
    torch.testing.assert_close(multimodal_data["multimodal_embedding"], torch.ones(2, 2))
    # Only the per-forward payload drops the raw inputs.
    assert "image" in request.py_multimodal_data
