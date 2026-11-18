# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.llmapi.llm import BaseLLM
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


class Processor:
    config = SimpleNamespace(text_config=SimpleNamespace(rope_scaling={"mrope_section": [1, 1, 1]}))

    def get_mrope_config(self):
        raise AssertionError("mRoPE must be supplied by context")

    def get_mm_token_ids(self):
        return torch.tensor([91, 92])

    def __call__(self, *args, **kwargs):
        raise AssertionError("input processor must not run")


class PlainRopeProcessor(Processor):
    config = SimpleNamespace(text_config=SimpleNamespace(rope_scaling={"rope_type": "llama3"}))


class VocabProcessor(Processor):
    """Reports no media token IDs; media tokens are at or above the vocabulary size."""

    def get_mm_token_ids(self):
        return None

    def get_vocab_size(self):
        return 100


@pytest.fixture
def llm():
    llm = BaseLLM.__new__(BaseLLM)
    llm.input_processor = Processor()
    llm.args = SimpleNamespace()
    llm._is_encoder_decoder_model = lambda: False
    return llm


def params(prompt_len=3):
    return DisaggregatedParams(
        request_type="generation_only",
        mrope_position_ids_handle=SharedTensorContainer.from_tensor_bytes(
            torch.arange(3 * prompt_len).reshape(3, 1, prompt_len)
        ).dump_to_dict(),
        mrope_position_deltas_handle=SharedTensorContainer.from_tensor_bytes(
            torch.tensor([-128], dtype=torch.int32)
        ).dump_to_dict(),
    )


def test_expanded_tokens_skip_input_processor(llm):
    ids, prompt, mm, _ = llm._preprocess(
        {"prompt_token_ids": [1, 91, 2], "multi_modal_data": None},
        SamplingParams(max_tokens=4),
        params(),
    )
    assert ids == [1, 91, 2]
    assert prompt is None
    mm.to_tensor("multimodal_data")
    assert mm.multimodal_data["mrope_config"]["mrope_position_deltas"].item() == -128


@pytest.mark.parametrize("missing", ["mrope_position_ids_handle", "mrope_position_deltas_handle"])
def test_missing_mrope_fails_loudly(llm, missing):
    disagg = params()
    setattr(disagg, missing, None)
    with pytest.raises(ValueError, match="requires portable mRoPE"):
        llm._preprocess(
            {"prompt_token_ids": [1, 91, 2], "multi_modal_data": None},
            SamplingParams(max_tokens=4),
            disagg,
        )


@pytest.mark.parametrize("processor", [Processor, PlainRopeProcessor])
def test_cuda_ipc_mrope_is_rejected(llm, processor):
    llm.input_processor = processor()
    disagg = params()
    disagg.mrope_position_ids_handle = {"method_key": 1}
    with pytest.raises(ValueError, match="requires portable mRoPE"):
        llm._preprocess({"prompt_token_ids": [1, 91, 2]}, SamplingParams(max_tokens=4), disagg)


@pytest.mark.parametrize("key", ["tensor_size", "dtype", "data"])
def test_incomplete_byte_mrope_is_rejected(llm, key):
    disagg = params()
    del disagg.mrope_position_deltas_handle[key]
    with pytest.raises(ValueError, match="requires portable mRoPE"):
        llm._preprocess({"prompt_token_ids": [1, 91, 2]}, SamplingParams(max_tokens=4), disagg)


@pytest.mark.parametrize(
    "prompt, name, mutate, error",
    [
        ([1, 91, 2], "mrope_position_ids_handle", {"tensor_size": [3, 1, 4]}, "shape"),
        ([1, 91, 2], "mrope_position_deltas_handle", {"tensor_size": [2]}, "shape"),
        ([1, 91, 2], "mrope_position_ids_handle", {"dtype": "torch.float32"}, "integers"),
        ([1, 91, 2], "mrope_position_ids_handle", {"dtype": "torch.int32"}, "of 36 bytes"),
        ([1, 91, 2], "mrope_position_ids_handle", lambda data: data + "AAAA", "of 72 bytes"),
        ([1, 91, 2], "mrope_position_ids_handle", lambda data: "*" + data[1:], "of 72 bytes"),
        ([1, 2, 3], "mrope_position_ids_handle", lambda data: data + "AAAA", "of 72 bytes"),
    ],
    ids=["ids-shape", "delta-shape", "float", "dtype-size", "too-long", "not-base64", "text-only"],
)
def test_malformed_byte_mrope_is_rejected_before_decoding(llm, prompt, name, mutate, error):
    disagg = params()
    handle = getattr(disagg, name)
    if callable(mutate):
        handle["data"] = mutate(handle["data"])
    else:
        handle.update(mutate)
    with pytest.raises(ValueError, match=f"{name}.*{error}"):
        llm._preprocess({"prompt_token_ids": prompt}, SamplingParams(max_tokens=4), disagg)


def test_media_tokens_above_vocab_are_detected(llm):
    llm.input_processor = VocabProcessor()
    disagg = DisaggregatedParams(request_type="generation_only")
    with pytest.raises(ValueError, match="requires portable mRoPE"):
        llm._preprocess({"prompt_token_ids": [1, 101, 2]}, SamplingParams(max_tokens=4), disagg)
    _, _, mm, _ = llm._preprocess(
        {"prompt_token_ids": [1, 2, 99]}, SamplingParams(max_tokens=4), disagg
    )
    assert mm is None


def test_text_only_does_not_require_mrope(llm):
    ids, _, mm, _ = llm._preprocess(
        {"prompt_token_ids": [1, 2, 3]},
        SamplingParams(max_tokens=4),
        DisaggregatedParams(request_type="generation_only"),
    )
    assert ids == [1, 2, 3]
    assert mm is None


def test_context_exports_portable_mrope_before_executor_submission(llm, monkeypatch):
    positions = torch.arange(12).reshape(3, 1, 4)
    deltas = torch.tensor([-32], dtype=torch.int32)

    def processor(*args, **kwargs):
        return [1, 91, 91, 2], {
            "multimodal_data": {
                "mrope_config": {"mrope_position_ids": positions, "mrope_position_deltas": deltas}
            }
        }

    monkeypatch.setattr(
        "tensorrt_llm.llmapi.llm.create_input_processor_with_hash",
        lambda *args, **kwargs: processor,
    )
    llm.args = SimpleNamespace(multimodal_config=None)
    disagg = DisaggregatedParams(request_type="context_only")
    _, _, mm, _ = llm._preprocess(
        {"prompt_token_ids": [1, 91, 2], "multi_modal_data": {"image": [object()]}},
        SamplingParams(max_tokens=4),
        disagg,
    )
    assert disagg.mrope_position_ids_handle["method_key"] == 4
    assert disagg.mrope_position_deltas_handle["method_key"] == 4
    deltas.zero_()
    restored = SharedTensorContainer.from_dict(disagg.mrope_position_deltas_handle).get_local_view()
    assert restored.item() == -32
    mm.to_tensor("multimodal_data")


def test_generation_only_preserves_multimodal_cache_identity(llm):
    from dataclasses import asdict

    from tensorrt_llm.inputs.multimodal import MultimodalInput

    identity = MultimodalInput.from_components(
        [[1, 2, 3, 4, 5, 6, 7, 8]], [1], [2], ["image-id"], [0, 2], [1, 3], [1, 1]
    )
    disagg = params(4)
    disagg.multimodal_input = asdict(identity)
    ids, _, mm, _ = llm._preprocess(
        {"prompt_token_ids": [1, 91, 2, 91], "multi_modal_data": None},
        SamplingParams(max_tokens=4),
        disagg,
    )
    assert ids == [1, 91, 2, 91]
    assert asdict(mm.multimodal_input) == asdict(identity)


@pytest.mark.parametrize("reuse", [False, True])
def test_plain_rope_model_does_not_require_mrope(llm, reuse):
    from dataclasses import asdict

    from tensorrt_llm.inputs.multimodal import MultimodalInput

    identity = MultimodalInput.from_components(
        [[1, 2, 3, 4, 5, 6, 7, 8]], [1], [2], ["image-id"], [0, 2], [1, 3], [1, 1]
    )
    llm.input_processor = PlainRopeProcessor()
    llm.args = SimpleNamespace(kv_cache_config=SimpleNamespace(enable_block_reuse=reuse))
    disagg = DisaggregatedParams(request_type="generation_only", multimodal_input=asdict(identity))
    ids, _, mm, _ = llm._preprocess(
        {"prompt_token_ids": [1, 91, 2, 91], "multi_modal_data": None},
        SamplingParams(max_tokens=4),
        disagg,
    )
    assert ids == [1, 91, 2, 91]
    assert asdict(mm.multimodal_input) == asdict(identity)
    assert "mrope_config" not in mm.multimodal_data


def test_context_result_carries_cache_identity():
    from dataclasses import asdict

    from tensorrt_llm.executor import GenerationRequest, GenerationResult
    from tensorrt_llm.inputs.multimodal import MultimodalInput, MultimodalParams

    identity = MultimodalInput.from_components(
        [[1, 2, 3, 4, 5, 6, 7, 8]], [1], [2], ["image-id"], [0, 2], [1, 3], [1, 1]
    )
    request = GenerationRequest(
        prompt_token_ids=[1, 91, 2, 91],
        sampling_params=SamplingParams(max_tokens=1),
        multimodal_params=MultimodalParams(multimodal_input=identity),
    )
    request.set_id(1)
    original = DisaggregatedParams(request_type="context_only")
    result = GenerationResult(request, disaggregated_params=original)
    assert original.multimodal_input is None
    assert result.disaggregated_params.multimodal_input == asdict(identity)
    identity.multimodal_hashes[0][0] = 99
    assert result.disaggregated_params.multimodal_input["multimodal_hashes"][0][0] == 1


def test_generation_only_cache_reuse_requires_multimodal_identity(llm):
    llm.args = SimpleNamespace(kv_cache_config=SimpleNamespace(enable_block_reuse=True))
    with pytest.raises(ValueError, match="KV reuse requires cache identity") as error:
        llm._preprocess(
            {"prompt_token_ids": [1, 91, 2], "multi_modal_data": None},
            SamplingParams(max_tokens=4),
            params(),
        )
    # The error names the setting that serves the request without an identity.
    assert "kv_cache_config.enable_block_reuse=False" in str(error.value)
    llm.args = SimpleNamespace(kv_cache_config=SimpleNamespace(enable_block_reuse=False))
    ids, _, mm, _ = llm._preprocess(
        {"prompt_token_ids": [1, 91, 2], "multi_modal_data": None},
        SamplingParams(max_tokens=4),
        params(),
    )
    assert ids == [1, 91, 2]
    assert mm.multimodal_input is None


@pytest.mark.parametrize(
    "update, error",
    [
        ({"multimodal_hashes": [[1] * 7]}, "eight int32"),
        ({"multimodal_hashes": [[2**31] + [1] * 7]}, "eight int32"),
        ({"multimodal_positions": [3]}, "exceed the prompt"),
        ({"multimodal_run_positions": [1, 4]}, "exceed the prompt"),
    ],
    ids=["hash-width", "hash-int32", "span", "run"],
)
def test_invalid_multimodal_identity_is_rejected(llm, update, error):
    from dataclasses import asdict

    from tensorrt_llm.inputs.multimodal import MultimodalInput

    identity = MultimodalInput.from_components(
        [[1, 2, 3, 4, 5, 6, 7, 8]], [1], [2], ["image-id"], [0, 2], [1, 3], [1, 1]
    )
    disagg = params(4)
    disagg.multimodal_input = {**asdict(identity), **update}
    with pytest.raises(ValueError, match=error):
        llm._preprocess({"prompt_token_ids": [1, 91, 2, 91]}, SamplingParams(max_tokens=4), disagg)
