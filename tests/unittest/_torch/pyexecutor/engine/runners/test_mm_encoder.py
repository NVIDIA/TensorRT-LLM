# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.engine.runners import mm_encoder as mm_encoder_module
from tensorrt_llm._torch.pyexecutor.engine.runners.interface import ScheduledInputs
from tensorrt_llm._torch.pyexecutor.engine.runners.mm_encoder import MultimodalEncoderRunner
from tensorrt_llm._torch.pyexecutor.llm_request import PyResult
from tensorrt_llm._torch.pyexecutor.sampler import EarlyStopWithMMResult
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm._torch.utils import get_model_extra_attrs
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalItemMetadata,
    MultimodalParams,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def unpinned_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mm_encoder_module, "prefer_pinned", lambda: False)


def test_mm_encoder_runner_forward_returns_empty_result_without_multimodal_params() -> None:
    runner = MultimodalEncoderRunner(
        SimpleNamespace(extra_attrs={}), encoder_cache_enabled=False, moe_load_balancer=None
    )

    result = runner.forward(
        ScheduledInputs(batch=ScheduledRequests()),
        resource_manager=SimpleNamespace(get_resource_manager=Mock(return_value=None)),
    )

    assert result == {
        "mm_embeddings": [],
        "mm_embedding_request_indices": [],
        "mm_embedding_lengths": [],
    }


def test_mm_encoder_runner_forward_skips_metadata_only_and_missing_length_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = [
        SimpleNamespace(py_multimodal_data=None),
        SimpleNamespace(py_multimodal_data={"mrope_config": {}}),
        SimpleNamespace(py_multimodal_data={"image": object()}),
        SimpleNamespace(
            py_multimodal_data={
                "image": object(),
                MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                    [("image", 0)], output_embedding_lengths=[2]
                ),
            },
            multimodal_lengths=[2],
        ),
    ]
    for index, request in enumerate(requests):
        request.get_num_tokens = Mock(return_value=index + 1)
        request.get_tokens = Mock(side_effect=AssertionError("token lists must not be read"))
        request.py_multimodal_params = MultimodalParams(multimodal_data=request.py_multimodal_data)
    multimodal_input = object()
    build_input = Mock(return_value=multimodal_input)
    monkeypatch.setattr(mm_encoder_module, "_build_request_multimodal_input", build_input)
    model = SimpleNamespace(
        extra_attrs={}, forward=Mock(return_value=[torch.arange(4).reshape(2, 2)])
    )
    runner = MultimodalEncoderRunner(model, encoder_cache_enabled=True, moe_load_balancer=None)
    monkeypatch.setattr(mm_encoder_module, "prefer_pinned", lambda: True)

    with patch.object(
        MultimodalParams, "to_device", autospec=True, side_effect=MultimodalParams.to_device
    ) as to_device:
        result = runner.forward(
            ScheduledInputs(batch=SimpleNamespace(context_requests=requests)),
            resource_manager=SimpleNamespace(get_resource_manager=Mock(return_value=None)),
        )

    build_input.assert_called_once_with(requests[3], True)
    model.forward.assert_called_once()
    params = model.forward.call_args.args[0]
    assert len(params) == 1
    assert params[0].multimodal_input is multimodal_input
    assert params[0].multimodal_data == requests[3].py_multimodal_data
    assert params[0].input_ids_start_offset == 6
    # An encoder without device paths moves every tensor per request, as before.
    to_device.assert_called_once_with(
        params[0], "multimodal_data", "cuda", pin_memory=True, target_keywords=None
    )
    for request in requests:
        request.get_tokens.assert_not_called()
    assert result["logits"] is None
    assert result["mm_embedding_request_indices"] == [3]
    assert result["mm_embedding_lengths"] == [[2]]
    torch.testing.assert_close(result["mm_embeddings"][0], torch.arange(4).reshape(2, 2))


def test_mm_encoder_runner_leaves_raw_media_to_the_encoder_call() -> None:
    """A default-builder encoder moves only MRoPE per request; its forward moves the media."""
    from tensorrt_llm._torch.models.modeling_multimodal_mixin import EncoderGroup

    model = SimpleNamespace(
        extra_attrs={},
        mm_encoder_groups=(EncoderGroup(("image", "video"), lambda **kwargs: None),),
        multimodal_data_device_paths=[
            "image.pixel_values",
            "video.pixel_values_videos",
            "mrope_config.mrope_position_ids",
        ],
    )
    runner = MultimodalEncoderRunner(model, encoder_cache_enabled=False, moe_load_balancer=None)
    assert runner._device_paths == ["mrope_config.mrope_position_ids"]


def test_mm_encoder_runner_splits_embeddings_and_returns_mrope_metadata() -> None:
    requests = [
        SimpleNamespace(
            py_multimodal_data={
                "image": object(),
                MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                    [("image", 0)], output_embedding_lengths=[1]
                ),
                "mrope_config": {
                    "mrope_position_ids": "first-ids",
                    "mrope_position_deltas": "first-deltas",
                },
            },
            multimodal_lengths=[1],
        ),
        SimpleNamespace(
            py_multimodal_data={
                "image": object(),
                MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                    [("image", 0)], output_embedding_lengths=[2]
                ),
                "mrope_config": {
                    "mrope_position_ids": "second-ids",
                    "mrope_position_deltas": "second-deltas",
                },
            },
            multimodal_lengths=[2],
        ),
    ]
    embeddings = torch.arange(6).reshape(3, 2)
    model = SimpleNamespace(extra_attrs={}, forward=Mock(return_value=[embeddings]))
    runner = MultimodalEncoderRunner(model, encoder_cache_enabled=False, moe_load_balancer=None)

    for index, request in enumerate(requests):
        request.py_multimodal_params = MultimodalParams(multimodal_data=request.py_multimodal_data)
        request.get_num_tokens = Mock(return_value=index + 1)
    result = runner.forward(
        ScheduledInputs(batch=SimpleNamespace(context_requests=requests)),
        resource_manager=SimpleNamespace(get_resource_manager=Mock(return_value=None)),
    )

    model.forward.assert_called_once()
    params = model.forward.call_args.args[0]
    assert [param.input_ids_start_offset for param in params] == [0, 1]
    assert result["logits"] is None
    assert result["mm_embedding_request_indices"] == [0, 1]
    assert result["mm_embedding_lengths"] == [[1], [2]]
    torch.testing.assert_close(result["mm_embeddings"][0], embeddings[:1])
    torch.testing.assert_close(result["mm_embeddings"][1], embeddings[1:])
    assert result["mrope_position_ids"] == ["first-ids", "second-ids"]
    assert result["mrope_position_deltas"] == ["first-deltas", "second-deltas"]


def test_mm_encoder_handoff_preserves_per_item_metadata_in_sparse_batch() -> None:
    requests = [
        SimpleNamespace(
            py_multimodal_data={
                "image": object(),
                MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                    [("image", 0)], output_embedding_lengths=[1]
                ),
            },
            multimodal_lengths=[1],
        ),
        SimpleNamespace(py_multimodal_data=None),
        SimpleNamespace(
            py_multimodal_data={
                "video": object(),
                MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                    [("video", 0), ("video", 1)], output_embedding_lengths=[3, 2]
                ),
            },
            multimodal_lengths=[7, 6],
        ),
    ]
    for request in requests:
        request.py_result = PyResult(prompt_len=1, max_new_tokens=1)
        request.set_finished_reason = Mock()
        request.py_multimodal_params = MultimodalParams(multimodal_data=request.py_multimodal_data)
        request.get_num_tokens = Mock(return_value=1)
    scheduled_requests = SimpleNamespace(
        context_requests=requests, generation_requests=[], num_context_requests=len(requests)
    )
    embeddings = torch.arange(12, dtype=torch.float32).reshape(6, 2)

    def encode(batch_params: list[MultimodalParams]) -> list[torch.Tensor]:
        assert torch.is_inference_mode_enabled()
        assert get_model_extra_attrs() is model.extra_attrs
        batch_params[1].multimodal_data["multimodal_embedding_metadata"] = [
            {"retained_token_counts": [2, 1]},
            {"retained_token_counts": [2, 0]},
        ]
        return [embeddings]

    model = SimpleNamespace(extra_attrs={}, forward=encode)
    runner = MultimodalEncoderRunner(model, encoder_cache_enabled=False, moe_load_balancer=None)
    result = runner.forward(
        ScheduledInputs(batch=scheduled_requests),
        resource_manager=SimpleNamespace(get_resource_manager=Mock(return_value=None)),
    )
    assert get_model_extra_attrs() is None
    sampler = EarlyStopWithMMResult()
    state = sampler.sample_async(scheduled_requests, result, [])
    sampler.update_requests(state)

    assert requests[1].py_result.mm_embedding_handles is None
    image_handles = json.loads(json.dumps(requests[0].py_result.mm_embedding_handles))
    assert len(image_handles) == 1
    assert "metadata" not in image_handles[0]
    torch.testing.assert_close(
        SharedTensorContainer.from_dict(image_handles[0]).get_local_view(), embeddings[:1]
    )
    video_handles = json.loads(json.dumps(requests[2].py_result.mm_embedding_handles))
    assert [handle["metadata"]["retained_token_counts"] for handle in video_handles] == [
        [2, 1],
        [2, 0],
    ]
    restored = [
        SharedTensorContainer.from_dict(handle).get_local_view() for handle in video_handles
    ]
    assert [len(embedding) for embedding in restored] == [3, 2]
    torch.testing.assert_close(torch.cat(restored), embeddings[1:])


def test_mm_encoder_runner_rejects_kv_manager_before_preparation() -> None:
    runner = MultimodalEncoderRunner(
        SimpleNamespace(extra_attrs={}), encoder_cache_enabled=False, moe_load_balancer=None
    )
    runner._prepare_inputs = Mock()
    resource_manager = Mock()
    resource_manager.get_resource_manager.return_value = object()

    with pytest.raises(
        AssertionError,
        match="no-KV-cache runner was initialized, but a KV cache manager was allocated",
    ):
        runner.forward(
            ScheduledInputs(batch=ScheduledRequests()), resource_manager=resource_manager
        )

    runner._prepare_inputs.assert_not_called()
