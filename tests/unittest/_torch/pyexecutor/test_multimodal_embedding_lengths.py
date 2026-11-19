# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import PyResult, get_multimodal_embedding_lengths
from tensorrt_llm._torch.pyexecutor.scheduled_encoder_executor import MultimodalEmbeddingCompletion
from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalItemMetadata,
    MultimodalParams,
)


@pytest.mark.parametrize(
    "req,expected",
    [
        (
            SimpleNamespace(multimodal_lengths=[3, 5], py_multimodal_data=None),
            None,
        ),
        (
            SimpleNamespace(
                multimodal_lengths=[6, 5],
                py_multimodal_data={
                    MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                        [("image", 0), ("image", 1)], output_embedding_lengths=[5, 3]
                    )
                },
            ),
            (5, 3),
        ),
        (
            SimpleNamespace(
                multimodal_lengths=[6, 5],
                py_multimodal_data={
                    "layout_metadata": {
                        MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                            [("image", 0), ("image", 1)], output_embedding_lengths=[6, 4]
                        ),
                    },
                },
            ),
            None,
        ),
    ],
)
@pytest.mark.cpu_only
def test_multimodal_embedding_lengths_returns_top_level_metadata(req, expected):
    """Getter reads top-level lengths and ignores layout metadata."""
    assert get_multimodal_embedding_lengths(req) == expected


@pytest.mark.cpu_only
def test_multimodal_embedding_lengths_uses_item_metadata():
    """Encoder output export uses the same rows as model input assembly."""
    metadata = MultimodalItemMetadata([("image", 0)], output_embedding_lengths=[2])
    data = {MULTIMODAL_ENCODER_ITEM_METADATA_KEY: metadata}
    params = MultimodalParams(multimodal_data=data)
    req = SimpleNamespace(multimodal_lengths=[4], py_multimodal_data=params.multimodal_data)

    assert get_multimodal_embedding_lengths(req) is params.get_item_layout()[1]


@pytest.mark.parametrize(
    "req,exception,match",
    [
        (
            SimpleNamespace(
                multimodal_lengths=[4],
                py_multimodal_data={
                    MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                        [("image", 0), ("image", 1)], output_embedding_lengths=[3, 1]
                    )
                },
            ),
            ValueError,
            "length must match",
        ),
        (
            SimpleNamespace(
                multimodal_lengths=[4],
                py_multimodal_data={
                    MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                        [("image", 0)], output_embedding_lengths=[5]
                    )
                },
            ),
            ValueError,
            "exceeds",
        ),
        (
            SimpleNamespace(
                multimodal_lengths=[4],
                py_multimodal_data={
                    MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                        [("image", 0)], output_embedding_lengths=[-1]
                    )
                },
            ),
            ValueError,
            "non-negative",
        ),
        (
            SimpleNamespace(
                multimodal_lengths=[4],
                py_multimodal_data={
                    MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                        [("image", i) for i in range(len(torch.tensor([4])))],
                        output_embedding_lengths=torch.tensor([4]),
                    ),
                },
            ),
            TypeError,
            "must contain only integers",
        ),
        (
            SimpleNamespace(
                multimodal_lengths=[4],
                py_multimodal_data={
                    MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                        [("image", 0)], output_embedding_lengths=(4.0,)
                    )
                },
            ),
            TypeError,
            "integers",
        ),
        (
            SimpleNamespace(
                multimodal_lengths=[4],
                py_multimodal_data=["multimodal_embedding_lengths", [4]],
            ),
            TypeError,
            "multimodal_data must be a dict",
        ),
    ],
)
@pytest.mark.cpu_only
def test_multimodal_embedding_lengths_rejects_invalid_metadata(req, exception, match):
    """Bad length metadata is rejected by the getter."""
    with pytest.raises(exception, match=match):
        get_multimodal_embedding_lengths(req)


class _FakePyResult:
    def __init__(self):
        self.mm_embeddings = []
        self.mrope_position = None

    def append_mm_embeddings(self, mm_embedding, mm_embedding_lengths, mm_embedding_metadata=None):
        self.mm_embeddings.append((mm_embedding, mm_embedding_lengths))

    def set_mrope_position(self, position_ids, position_deltas):
        self.mrope_position = (position_ids, position_deltas)


class _FakeRequest:
    def __init__(self, multimodal_lengths=None):
        self.multimodal_lengths = multimodal_lengths
        self.py_result = _FakePyResult()


@pytest.mark.parametrize("exports", [True, False], ids=["response_rank", "follower_rank"])
@pytest.mark.cpu_only
def test_mm_embedding_completion_attaches_results_by_request_index(exports):
    """A sparse batch's results reach their own requests, only on the rank that exports them."""
    text_request = _FakeRequest()
    mm_request = _FakeRequest(multimodal_lengths=[4])

    MultimodalEmbeddingCompletion(exports=exports).store(
        [text_request, mm_request],
        {
            "mm_embeddings": [torch.ones(4, 2)],
            "mm_embedding_request_indices": [1],
            "mm_embedding_lengths": [[4]],
            "mrope_position_ids": ["mm-pos"],
            "mrope_position_deltas": ["mm-delta"],
        },
    )

    assert text_request.py_result.mm_embeddings == []
    assert text_request.py_result.mrope_position is None
    if exports:
        [(mm_embedding, mm_embedding_lengths)] = mm_request.py_result.mm_embeddings
        assert mm_embedding.shape == (4, 2)
        assert mm_embedding_lengths == [4]
        assert mm_request.py_result.mrope_position == ("mm-pos", "mm-delta")
    else:
        assert mm_request.py_result.mm_embeddings == []
        assert mm_request.py_result.mrope_position is None


@pytest.mark.cpu_only
def test_py_result_mm_embedding_handles_use_shared_tensor_handles():
    """MM encoder result handles should preserve the producer tensor device."""
    result = PyResult(prompt_len=1, max_new_tokens=1)
    source = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    result.append_mm_embeddings(source, [1, 3])

    handles = result.mm_embedding_handles
    assert handles is not None
    assert [handle["method_key"] for handle in handles] == [2, 2]
    restored = [SharedTensorContainer.from_dict(handle).get_local_view() for handle in handles]
    assert torch.equal(torch.cat(restored, dim=0), source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_py_result_cuda_mm_embedding_handles_stay_cuda_backed():
    """CUDA MM encoder outputs should not move to CPU for handoff handles."""
    result = PyResult(prompt_len=1, max_new_tokens=1)
    source = torch.arange(8, dtype=torch.float32, device="cuda").reshape(4, 2)

    result.append_mm_embeddings(source, [2, 2])

    handles = result.mm_embedding_handles
    assert handles is not None
    assert [handle["method_key"] for handle in handles] == [1, 1]
