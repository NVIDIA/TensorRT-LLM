# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import PyResult, get_multimodal_embedding_lengths
from tensorrt_llm._torch.pyexecutor.sampler import EarlyStopWithMMResult, MultimodalResult
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
    def __init__(self) -> None:
        self.outputs = []

    def set_multimodal_result(self, result: MultimodalParams) -> None:
        self.outputs.append(result)


class _FakeRequest:
    def __init__(self) -> None:
        self.py_result = _FakePyResult()
        self.state = None

    def set_finished_reason(self, reason: object, index: int) -> None:
        self.finished_reason = (reason, index)


def _encoded(
    rows: int = 4, lengths: list[int] | None = None, device: str = "cpu"
) -> MultimodalParams:
    lengths = [rows] if lengths is None else lengths
    return MultimodalParams(
        multimodal_data={
            "multimodal_embedding": torch.arange(
                rows * 2, dtype=torch.float32, device=device
            ).reshape(rows, 2),
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                [("image", index) for index in range(len(lengths))],
                output_embedding_lengths=lengths,
            ),
        }
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("exports_results", [False, True])
def test_mm_sampler_maps_encoded_views_to_requests_and_controls_export(
    exports_results: bool,
) -> None:
    sampler = EarlyStopWithMMResult(return_mm_results=exports_results)
    text_request, mm_request = _FakeRequest(), _FakeRequest()
    result = _encoded()
    state = sampler.SampleState(
        requests=[text_request, mm_request],
        data=MultimodalResult(
            mm_encoder_outputs=[result], mm_embedding_request_indices=[1], num_context_requests=2
        ),
    )
    sampler.update_requests(state)
    assert not text_request.py_result.outputs
    assert mm_request.py_result.outputs == ([result] if exports_results else [])
    assert text_request.state == mm_request.state
    assert text_request.finished_reason == mm_request.finished_reason


@pytest.mark.cpu_only
def test_py_result_cpu_item_handles_roundtrip() -> None:
    from tensorrt_llm.disaggregated_params import DisaggregatedParams

    result = PyResult(prompt_len=1, max_new_tokens=1)
    source = _encoded(lengths=[1, 3])
    result.set_multimodal_result(source)
    handles = DisaggregatedParams(multimodal=result.multimodal).get_multimodal_embedding_handles()
    assert [handle["method_key"] for handle in handles] == [2, 2]
    restored = [SharedTensorContainer.from_dict(handle).get_local_view() for handle in handles]
    torch.testing.assert_close(torch.cat(restored), source.multimodal_data["multimodal_embedding"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_py_result_cuda_item_handles_stay_cuda_backed() -> None:
    from tensorrt_llm.disaggregated_params import DisaggregatedParams

    result = PyResult(prompt_len=1, max_new_tokens=1)
    source = _encoded(lengths=[2, 2], device="cuda")
    result.set_multimodal_result(source)
    handles = DisaggregatedParams(multimodal=result.multimodal).get_multimodal_embedding_handles()
    assert [handle["method_key"] for handle in handles] == [1, 1]


class _FakeScheduledRequests:
    def __init__(self, num_context_requests):
        self.generation_requests = []
        self.context_requests = [_FakeRequest() for _ in range(num_context_requests)]

    @property
    def num_context_requests(self):
        return len(self.context_requests)


@pytest.mark.cpu_only
def test_mm_encoder_sampler_preserves_the_result_view_and_request_correspondence() -> None:
    result = _encoded()
    model_outputs = {"mm_encoder_outputs": [result], "mm_embedding_request_indices": [1]}
    state = EarlyStopWithMMResult().sample_async(_FakeScheduledRequests(2), model_outputs, [])
    assert state.data.mm_encoder_outputs[0] is result
    assert state.data.mm_embedding_request_indices == [1]
    assert model_outputs["mm_encoder_outputs"][0] is result


@pytest.mark.cpu_only
@pytest.mark.parametrize("indices,expected_error", [([], "align"), ([1], "request index")])
def test_mm_encoder_sampler_rejects_invalid_request_correspondence(
    indices: list[int], expected_error: str
) -> None:
    with pytest.raises(ValueError, match=expected_error):
        EarlyStopWithMMResult().sample_async(
            _FakeScheduledRequests(1),
            {"mm_encoder_outputs": [_encoded()], "mm_embedding_request_indices": indices},
            [],
        )


@pytest.mark.cpu_only
def test_multimodal_result_rejects_embedding_shape_mismatch():
    with pytest.raises(ValueError, match="output rows"):
        MultimodalResult(
            mm_encoder_outputs=[_encoded(lengths=[3])],
            mm_embedding_request_indices=[0],
            num_context_requests=1,
        )


@pytest.mark.cpu_only
def test_prefill_decode_handoff_owns_only_declared_generation_state() -> None:
    from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLInputProcessorBase

    delta = torch.tensor([2])
    source = MultimodalParams(
        multimodal_data={
            "mrope_config": {
                "mrope_position_ids": torch.arange(12).reshape(3, 1, 4),
                "mrope_position_deltas": delta,
            },
            "multimodal_embedding": torch.zeros(3, 4),
            "image": {"pixel_values": torch.zeros(1, 3)},
        }
    )
    processor = object.__new__(Qwen3VLInputProcessorBase)
    retained = processor.prepare_multimodal_decode_inputs(source)
    result = PyResult(prompt_len=4, max_new_tokens=1)
    result.set_multimodal_decode_inputs(
        retained,
        request_layout={
            "prompt_token_ids": [1, 2, 3, 4],
            "model_data": {"encoder_state": delta},
        },
    )
    wire = result.multimodal
    assert wire["items"] == []
    assert wire["request_layout"] == {"prompt_token_ids": [1, 2, 3, 4]}
    assert set(wire["decode_inputs"]) == {"mrope_config"}
    assert set(wire["decode_inputs"]["mrope_config"]) == {"mrope_position_deltas"}
    imported = SharedTensorContainer.from_dict(
        wire["decode_inputs"]["mrope_config"]["mrope_position_deltas"]
    ).get_local_view()
    torch.testing.assert_close(imported, delta)
    assert imported.untyped_storage() is not delta.untyped_storage()
