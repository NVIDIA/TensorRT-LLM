# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_mistral import Mistral3InputProcessor
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm._torch.models.modeling_qwen2vl import Qwen2VLInputProcessorBase
from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.llm_request import (
    MultimodalEncoderProgress,
    get_multimodal_encoder_progress,
    get_multimodal_encoder_token_lengths,
    initialize_multimodal_encoder_request,
)
from tensorrt_llm._torch.pyexecutor.model_engine import (
    PyTorchModelEngine,
    _resolve_mm_encoder_token_budget,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import (
    MultimodalEagerEncoderScheduler,
    MultimodalScheduler,
)
from tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue import FCFSWaitingQueue
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalParams,
    strip_mm_encoder_inputs,
)
from tensorrt_llm.inputs.registry import MultimodalEncoderItemMetadata


class _CapacityScheduler:
    def schedule_request(self, requests):
        return list(requests), [], []


class _RejectMultimodalCapacityScheduler:
    def schedule_request(self, requests):
        fitting = [request for request in requests if not request.py_is_multimodal_encoder_request]
        return fitting, [], []


class _MicroBatchScheduler:
    def schedule(self, requests, inflight_request_ids):
        del inflight_request_ids
        return [], list(requests), []


class _BaseScheduler:
    def __init__(self):
        self.capacity_scheduler = _CapacityScheduler()
        self.micro_batch_scheduler = _MicroBatchScheduler()

    def can_schedule(self, requests):
        return bool(requests)


def _request(request_id, costs, *, ready=()):
    outputs = [None] * len(costs)
    for item_idx in ready:
        outputs[item_idx] = torch.empty(1)
    return SimpleNamespace(
        request_id=request_id,
        py_request_id=request_id,
        py_is_multimodal_encoder_request=True,
        py_multimodal_data={
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalEncoderItemMetadata(
                item_refs=[("image", item_idx) for item_idx in range(len(costs))],
                encoder_token_lengths=costs,
                output_embedding_lengths=[1] * len(costs),
            ),
            "multimodal_embedding_lengths": [1] * len(costs),
        },
        py_mm_encoder_outputs=outputs,
        multimodal_lengths=None,
        is_context_init_state=False,
    )


def test_mm_encoder_token_lengths_distinguishes_missing_and_invalid_data():
    request = SimpleNamespace(py_multimodal_data=None)

    assert get_multimodal_encoder_token_lengths(request) is None

    request.py_multimodal_data = []
    with pytest.raises(TypeError, match="py_multimodal_data must be a dict"):
        get_multimodal_encoder_token_lengths(request)


def test_mm_encoder_progress_is_derived_from_request_local_outputs():
    request = _request(1, [4, 4])
    assert get_multimodal_encoder_progress(request) is MultimodalEncoderProgress.PENDING

    request.py_mm_encoder_outputs[0] = torch.empty(1)
    assert get_multimodal_encoder_progress(request) is MultimodalEncoderProgress.PARTIAL

    request.py_mm_encoder_outputs[1] = torch.empty(1)
    assert get_multimodal_encoder_progress(request) is MultimodalEncoderProgress.READY

    request.py_mm_encoder_outputs = [None, None]
    request.py_multimodal_data["multimodal_embedding"] = torch.empty(2, 1)
    assert get_multimodal_encoder_progress(request) is MultimodalEncoderProgress.READY


def test_mm_encoder_item_metadata_survives_python_object_broadcast_serialization():
    metadata = MultimodalEncoderItemMetadata(
        item_refs=[("image", 0), ("video", 0)],
        encoder_token_lengths=[16, 32],
        output_embedding_lengths=[4, 8],
    )

    restored = pickle.loads(pickle.dumps({MULTIMODAL_ENCODER_ITEM_METADATA_KEY: metadata}))[
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY
    ]

    assert isinstance(restored, MultimodalEncoderItemMetadata)
    assert restored == metadata


def test_multimodal_scheduler_keeps_items_atomic_and_backfills_requests():
    scheduler = MultimodalScheduler(_BaseScheduler(), max_num_items=2, max_num_tokens=10)
    first = _request(1, [7, 7])
    second = _request(2, [3])

    output = scheduler.schedule_request([first, second], set())

    assert output.scheduled_mm_encoder_items == {1: [0], 2: [0]}
    assert output.context_requests == [second]


def test_multimodal_scheduler_schedules_full_request_batch_when_all_items_fit():
    scheduler = MultimodalScheduler(_BaseScheduler(), max_num_items=2, max_num_tokens=10)
    request = _request(1, [6, 4])

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items is None
    assert output.context_requests == [request]


def test_multimodal_scheduler_uses_item_path_only_for_overflow():
    scheduler = MultimodalScheduler(_BaseScheduler(), max_num_items=3, max_num_tokens=10)
    request = _request(1, [6, 4, 1])

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items == {1: [0, 1]}
    assert output.context_requests == []


def test_multimodal_scheduler_preserves_non_multimodal_requests():
    scheduler = MultimodalScheduler(_BaseScheduler(), max_num_items=1, max_num_tokens=1)
    request = SimpleNamespace(
        request_id=1,
        py_request_id=1,
        py_is_multimodal_encoder_request=False,
        is_context_init_state=False,
    )

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items is None
    assert output.context_requests == [request]


def test_encoder_token_budget_auto_raises_for_atomic_item():
    assert _resolve_mm_encoder_token_budget(8192, 65536) == 65536


def test_request_rejects_item_above_effective_startup_maximum():
    request = _request(1, [9])
    request.py_multimodal_data["image"] = {"pixel_values": torch.empty(1)}

    with pytest.raises(ValueError, match="exceeding the effective startup maximum 8"):
        initialize_multimodal_encoder_request(request, max_num_tokens=8)


def test_eager_scheduler_encodes_request_rejected_by_llm_capacity():
    base_scheduler = _BaseScheduler()
    base_scheduler.capacity_scheduler = _RejectMultimodalCapacityScheduler()
    scheduler = MultimodalEagerEncoderScheduler(base_scheduler, max_num_items=1, max_num_tokens=8)
    multimodal_request = _request(1, [8])
    text_request = SimpleNamespace(
        request_id=2,
        py_request_id=2,
        py_is_multimodal_encoder_request=False,
        is_context_init_state=False,
    )

    output = scheduler.schedule_request([multimodal_request, text_request], set())

    assert output.scheduled_mm_encoder_items == {1: [0]}
    assert output.context_requests == [text_request]


def test_forward_multimodal_encoder_step_delegates_to_model_engine():
    calls = []
    executor = object.__new__(PyExecutor)
    executor.active_requests = [SimpleNamespace(request_id=1)]
    executor.model_engine = SimpleNamespace(
        forward_multimodal_encoder_items=lambda requests, items: calls.append((requests, items))
    )
    scheduled_items = {1: [0]}
    scheduled_requests = SimpleNamespace(scheduled_mm_encoder_items=scheduled_items)

    executor._forward_multimodal_encoder_step(scheduled_requests)

    assert calls == [(executor.active_requests, scheduled_items)]


def _executor_for_mm_admission(active_requests, *, max_num_tokens=8):
    executor = object.__new__(PyExecutor)
    executor.enable_attention_dp = False
    executor.dist = SimpleNamespace(tp_size=1)
    executor.max_num_active_requests = 8
    executor.is_benchmark_disagg = False
    executor.is_multimodal_model = True
    executor.model_engine = SimpleNamespace(
        encoder_max_num_items=8,
        encoder_max_num_tokens=max_num_tokens,
    )
    executor.active_requests = active_requests
    return executor


def _waiting_item(request_id, costs=None):
    multimodal_data = None
    if costs is not None:
        multimodal_data = {
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalEncoderItemMetadata(
                item_refs=[("image", item_idx) for item_idx in range(len(costs))],
                encoder_token_lengths=costs,
                output_embedding_lengths=[1] * len(costs),
            ),
            "multimodal_embedding_lengths": [1] * len(costs),
        }
    return RequestQueueItem(
        request_id,
        SimpleNamespace(py_multimodal_data=multimodal_data),
    )


def test_mm_admission_does_not_charge_ready_active_request():
    active = _request(1, [8])
    active.py_multimodal_data["multimodal_embedding"] = torch.empty(1, 1)
    waiting = FCFSWaitingQueue([_waiting_item(2, [8])])
    executor = _executor_for_mm_admission([active])

    admitted = executor._pop_from_waiting_queue(waiting, 1)

    assert [item.id for item in admitted] == [2]
    assert not waiting


def test_mm_admission_passes_oversized_request_to_validation():
    waiting = FCFSWaitingQueue([_waiting_item(1, [9]), _waiting_item(2, None)])
    executor = _executor_for_mm_admission([], max_num_tokens=8)

    admitted = executor._pop_from_waiting_queue(waiting, 0)

    assert [item.id for item in admitted] == [1, 2]
    assert not waiting


def test_item_encoder_slices_and_restores_selected_item_order():
    class _Model(MultimodalModelMixin):
        def encode_multimodal_inputs(self, multimodal_params):
            return torch.cat(
                [param.multimodal_data["image"]["pixel_values"] for param in multimodal_params]
            )

    multimodal_param = MultimodalParams(
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
        }
    )

    model = _Model()
    encoder_inputs = model.prepare_multimodal_encoder_inputs(
        [(multimodal_param, 1), (multimodal_param, 0)]
    )
    outputs = model.forward_multimodal_encoder_items(encoder_inputs)

    assert [output.squeeze(1).tolist() for output in outputs] == [
        [2, 3, 4],
        [0, 1],
    ]


def test_prepare_multimodal_encoder_inputs_slices_before_device_transfer():
    multimodal_param = MultimodalParams(
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
        }
    )

    encoder_inputs = MultimodalModelMixin.prepare_multimodal_encoder_inputs(
        MultimodalModelMixin(), [(multimodal_param, 1)]
    )

    item_param, embedding_length, modality = encoder_inputs[0]
    assert modality == "image"
    assert embedding_length == 3
    assert item_param.multimodal_data["image"]["pixel_values"].squeeze(1).tolist() == [2, 3, 4]
    assert multimodal_param.multimodal_data["image"]["pixel_values"].shape[0] == 5


def test_prepare_multimodal_encoder_inputs_rejects_invalid_metadata_types():
    multimodal_param = MultimodalParams(
        multimodal_data={
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: ("image", 0),
            "multimodal_embedding_lengths": [1],
        }
    )

    with pytest.raises(TypeError, match="must be a MultimodalEncoderItemMetadata"):
        MultimodalModelMixin().prepare_multimodal_encoder_inputs([(multimodal_param, 0)])


def test_strip_mm_encoder_inputs_preserves_embedding_and_runtime_metadata():
    embedding = torch.empty(3, 4)
    mm_data = {
        "image": {"pixel_values": torch.empty(2, 3)},
        "video": {"pixel_values_videos": torch.empty(2, 3)},
        "multimodal_embedding": embedding,
        "multimodal_embed_mask_cumsum": torch.tensor([0, 1]),
    }

    strip_mm_encoder_inputs(mm_data)

    assert "image" not in mm_data
    assert "video" not in mm_data
    assert mm_data["multimodal_embedding"] is embedding
    assert "multimodal_embed_mask_cumsum" in mm_data


def test_item_outputs_fill_one_contiguous_buffer_and_release_raw_data(monkeypatch):
    class _Model(MultimodalModelMixin):
        def forward_multimodal_encoder_items(self, encoder_inputs):
            return [
                torch.full((embedding_length, 2), float(embedding_length))
                for _, embedding_length, _ in encoder_inputs
            ]

    monkeypatch.setattr(MultimodalParams, "to_device", lambda self, *args, **kwargs: self)
    engine = object.__new__(PyTorchModelEngine)
    engine.model = _Model()
    request = SimpleNamespace(
        request_id=1,
        py_request_id=1,
        py_multimodal_data={
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
        py_mm_encoder_outputs=[None, None],
        py_mm_encoder_output_buffer=None,
        py_mm_encoder_output_offsets=[0, 2, 5],
        multimodal_lengths=None,
    )

    engine.forward_multimodal_encoder_items([request], {1: [0]})

    output_buffer = request.py_mm_encoder_output_buffer
    assert output_buffer.shape == (5, 2)
    assert request.py_mm_encoder_outputs[0].data_ptr() == output_buffer.data_ptr()
    assert request.py_mm_encoder_outputs[1] is None
    assert "image" in request.py_multimodal_data

    engine.forward_multimodal_encoder_items([request], {1: [1]})

    assert request.py_multimodal_data["multimodal_embedding"] is output_buffer
    assert (
        request.py_mm_encoder_outputs[1].untyped_storage().data_ptr()
        == output_buffer.untyped_storage().data_ptr()
    )
    assert output_buffer.tolist() == [[2.0, 2.0], [2.0, 2.0], [3.0, 3.0], [3.0, 3.0], [3.0, 3.0]]
    assert "image" not in request.py_multimodal_data


def test_qwen_item_metadata_uses_prompt_order_and_pre_merger_costs():
    processor = object.__new__(Qwen2VLInputProcessorBase)
    processor._config = SimpleNamespace(
        image_token_id=11,
        video_token_id=12,
        vision_start_token_id=10,
        vision_end_token_id=13,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    prompt_token_ids = [10, 12, 12, 13, 1, 10, 11, 11, 13]
    multimodal_data = {
        "image": {"image_grid_thw": torch.tensor([[1, 4, 4]])},
        "video": {"video_grid_thw": torch.tensor([[2, 4, 4]])},
    }

    metadata = processor.get_mm_encoder_item_metadata(prompt_token_ids, multimodal_data)

    assert isinstance(metadata, MultimodalEncoderItemMetadata)
    assert metadata.item_refs == [("video", 0), ("image", 0)]
    assert metadata.encoder_token_lengths == [32, 16]
    assert metadata.output_embedding_lengths == [8, 4]


def test_qwen_item_metadata_collapses_frame_spans_into_original_video():
    processor = object.__new__(Qwen2VLInputProcessorBase)
    processor._config = SimpleNamespace(
        image_token_id=11,
        video_token_id=12,
        vision_start_token_id=10,
        vision_end_token_id=13,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    prompt_token_ids = [10, 12, 13, 100, 10, 12, 13]
    multimodal_data = {
        "video": {"video_grid_thw": torch.tensor([[2, 4, 4]])},
    }

    metadata = processor.get_mm_encoder_item_metadata(prompt_token_ids, multimodal_data)

    assert metadata.item_refs == [("video", 0)]
    assert metadata.encoder_token_lengths == [32]
    assert metadata.output_embedding_lengths == [8]


def test_mistral_item_metadata_separates_patch_and_embedding_units():
    processor = object.__new__(Mistral3InputProcessor)
    processor._vision_geometry = lambda: (14, 2, 3, 1024)

    metadata = processor.get_mm_encoder_item_metadata(
        [], {"image": {"image_sizes": [[28, 56], [56, 56]]}}
    )

    assert metadata.item_refs == [("image", 0), ("image", 1)]
    assert metadata.encoder_token_lengths == [8, 16]
    assert metadata.output_embedding_lengths == [2, 4]
