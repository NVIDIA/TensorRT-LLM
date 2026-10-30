# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from _torch.pyexecutor.engine.runners.utils import _make_runner

from tensorrt_llm._torch.pyexecutor.engine.runners.mm_encoder import MultimodalEncoderRunner
from tensorrt_llm._torch.pyexecutor.llm_request import PyResult
from tensorrt_llm._torch.pyexecutor.sampler import EarlyStopWithMMResult
from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalItemMetadata,
)

pytestmark = pytest.mark.cpu_only


def test_mm_encoder_runner_forward_step_returns_empty_result_without_multimodal_params() -> None:
    runner = _make_runner(MultimodalEncoderRunner, SimpleNamespace())

    result = runner._forward_step({}, SimpleNamespace(context_requests=[]))

    assert result == {
        "mm_embeddings": [],
        "mm_embedding_request_indices": [],
        "mm_embedding_lengths": [],
    }


def test_mm_encoder_runner_forward_step_rejects_payload_request_count_mismatch() -> None:
    request = SimpleNamespace(py_multimodal_data={"image": object()})
    runner = _make_runner(MultimodalEncoderRunner, SimpleNamespace())

    with pytest.raises(ValueError, match="one multimodal payload per context"):
        runner._forward_step(
            {"multimodal_params": [SimpleNamespace(), SimpleNamespace()]},
            SimpleNamespace(context_requests=[request]),
        )


def test_mm_encoder_runner_forward_step_skips_metadata_only_and_missing_length_requests() -> None:
    requests = [
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
    params = [
        SimpleNamespace(multimodal_data={}),
        SimpleNamespace(multimodal_data={}),
        SimpleNamespace(multimodal_data={}),
    ]
    model = SimpleNamespace(forward=Mock(return_value=[torch.arange(4).reshape(2, 2)]))
    runner = _make_runner(MultimodalEncoderRunner, model)

    result = runner._forward_step(
        {"multimodal_params": params},
        SimpleNamespace(context_requests=requests),
    )

    model.forward.assert_called_once_with([params[2]])
    assert result["logits"] is None
    assert result["mm_embedding_request_indices"] == [2]
    assert result["mm_embedding_lengths"] == [[2]]
    torch.testing.assert_close(result["mm_embeddings"][0], torch.arange(4).reshape(2, 2))


def test_mm_encoder_runner_splits_embeddings_and_returns_mrope_metadata() -> None:
    first_param = SimpleNamespace(
        multimodal_data={
            "image": object(),
            "mrope_config": {
                "mrope_position_ids": "first-ids",
                "mrope_position_deltas": "first-deltas",
            },
        }
    )
    second_param = SimpleNamespace(
        multimodal_data={
            "image": object(),
            "mrope_config": {
                "mrope_position_ids": "second-ids",
                "mrope_position_deltas": "second-deltas",
            },
        }
    )
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
    embeddings = torch.arange(6).reshape(3, 2)
    model = SimpleNamespace(forward=Mock(return_value=[embeddings]))
    runner = _make_runner(MultimodalEncoderRunner, model)

    result = runner._forward_step(
        {"multimodal_params": [first_param, second_param]},
        SimpleNamespace(context_requests=requests),
    )

    model.forward.assert_called_once_with([first_param, second_param])
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
    params = [
        SimpleNamespace(multimodal_data=request.py_multimodal_data)
        for request in requests
        if request.py_multimodal_data is not None
    ]
    scheduled_requests = SimpleNamespace(
        context_requests=requests, generation_requests=[], num_context_requests=len(requests)
    )
    embeddings = torch.arange(12, dtype=torch.float32).reshape(6, 2)

    def encode(batch_params: list[SimpleNamespace]) -> list[torch.Tensor]:
        batch_params[1].multimodal_data["multimodal_embedding_metadata"] = [
            {"retained_token_counts": [2, 1]},
            {"retained_token_counts": [2, 0]},
        ]
        return [embeddings]

    runner = _make_runner(MultimodalEncoderRunner, SimpleNamespace(forward=encode))
    result = runner._forward_step(
        {"multimodal_params": params},
        scheduled_requests,
    )
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
