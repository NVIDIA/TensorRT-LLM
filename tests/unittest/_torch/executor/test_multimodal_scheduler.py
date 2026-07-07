# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.models.modeling_mistral import Mistral3InputProcessor
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm._torch.models.modeling_qwen2vl import Qwen2VLInputProcessorBase
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import MultimodalScheduler
from tensorrt_llm.inputs.multimodal import MultimodalParams


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
            "multimodal_encoder_token_lengths": costs,
            "multimodal_embedding_lengths": [1] * len(costs),
        },
        py_mm_encoder_outputs=outputs,
        py_mm_encoder_inflight_items=set(),
        multimodal_lengths=None,
        is_context_init_state=False,
    )


def test_multimodal_scheduler_keeps_items_atomic_and_backfills_requests():
    scheduler = MultimodalScheduler(_BaseScheduler(), max_batch_size=2, max_num_tokens=10)
    first = _request(1, [7, 7])
    second = _request(2, [3])

    output = scheduler.schedule_request([first, second], set())

    assert output.scheduled_mm_encoder_items == {1: [0], 2: [0]}
    assert output.context_requests == [second]
    assert first.py_mm_encoder_inflight_items == {0}
    assert second.py_mm_encoder_inflight_items == {0}


def test_multimodal_scheduler_uses_legacy_path_when_all_items_fit():
    scheduler = MultimodalScheduler(_BaseScheduler(), max_batch_size=2, max_num_tokens=10)
    request = _request(1, [6, 4])

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items is None
    assert output.context_requests == [request]
    assert request.py_mm_encoder_inflight_items == set()


def test_multimodal_scheduler_uses_item_path_only_for_overflow():
    scheduler = MultimodalScheduler(_BaseScheduler(), max_batch_size=3, max_num_tokens=10)
    request = _request(1, [6, 4, 1])

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items == {1: [0, 1]}
    assert output.context_requests == []
    assert request.py_mm_encoder_inflight_items == {0, 1}


def test_multimodal_scheduler_preserves_non_multimodal_requests():
    scheduler = MultimodalScheduler(_BaseScheduler(), max_batch_size=1, max_num_tokens=1)
    request = SimpleNamespace(
        request_id=1,
        py_request_id=1,
        py_is_multimodal_encoder_request=False,
        is_context_init_state=False,
    )

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items is None
    assert output.context_requests == [request]


def test_independent_mode_encodes_request_rejected_by_llm_capacity():
    base_scheduler = _BaseScheduler()
    base_scheduler.capacity_scheduler = _RejectMultimodalCapacityScheduler()
    scheduler = MultimodalScheduler(
        base_scheduler, max_batch_size=1, max_num_tokens=8, independent=True
    )
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
            "multimodal_item_refs": [("image", 0), ("image", 1)],
            "multimodal_embedding_lengths": [2, 3],
        }
    )

    outputs = _Model().encode_multimodal_items(
        [
            (multimodal_param, 1),
            (multimodal_param, 0),
        ]
    )

    assert [output.squeeze(1).tolist() for output in outputs] == [
        [2, 3, 4],
        [0, 1],
    ]


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

    refs, costs, embedding_lengths = processor.get_mm_encoder_item_metadata(
        prompt_token_ids, multimodal_data
    )

    assert refs == [("video", 0), ("image", 0)]
    assert costs == [32, 16]
    assert embedding_lengths == [8, 4]


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

    refs, costs, embedding_lengths = processor.get_mm_encoder_item_metadata(
        prompt_token_ids, multimodal_data
    )

    assert refs == [("video", 0)]
    assert costs == [32]
    assert embedding_lengths == [8]


def test_mistral_item_metadata_separates_patch_and_embedding_units():
    processor = object.__new__(Mistral3InputProcessor)
    processor._vision_geometry = lambda: (14, 2, 3, 1024)

    refs, costs, embedding_lengths = processor.get_mm_encoder_item_metadata(
        [], {"image": {"image_sizes": [[28, 56], [56, 56]]}}
    )

    assert refs == [("image", 0), ("image", 1)]
    assert costs == [8, 16]
    assert embedding_lengths == [2, 4]
