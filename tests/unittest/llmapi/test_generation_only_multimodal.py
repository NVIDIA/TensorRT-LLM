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
    def get_mrope_config(self):
        raise AssertionError("mRoPE must be supplied by context")

    def get_mm_token_ids(self):
        return torch.tensor([91, 92])

    def __call__(self, *args, **kwargs):
        raise AssertionError("input processor must not run")


@pytest.fixture
def llm():
    llm = BaseLLM.__new__(BaseLLM)
    llm.input_processor = Processor()
    llm.args = SimpleNamespace()
    llm._is_encoder_decoder_model = lambda: False
    return llm


def params():
    return DisaggregatedParams(
        request_type="generation_only",
        mrope_position_ids_handle=SharedTensorContainer.from_tensor_bytes(
            torch.arange(9).reshape(3, 1, 3)
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


def test_cuda_ipc_mrope_is_rejected(llm):
    disagg = params()
    disagg.mrope_position_ids_handle = {"method_key": 1}
    with pytest.raises(ValueError, match="requires portable mRoPE"):
        llm._preprocess({"prompt_token_ids": [1, 91, 2]}, SamplingParams(max_tokens=4), disagg)


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
    disagg = params()
    disagg.multimodal_input = asdict(identity)
    ids, _, mm, _ = llm._preprocess(
        {"prompt_token_ids": [1, 91, 2, 91], "multi_modal_data": None},
        SamplingParams(max_tokens=4),
        disagg,
    )
    assert ids == [1, 91, 2, 91]
    assert asdict(mm.multimodal_input) == asdict(identity)


def test_context_captures_cache_identity_before_submission(llm, monkeypatch):
    from dataclasses import asdict

    from tensorrt_llm.inputs.multimodal import MultimodalInput

    identity = MultimodalInput.from_components(
        [[1, 2, 3, 4, 5, 6, 7, 8]], [1], [2], ["image-id"], [0, 2], [1, 3], [1, 1]
    )

    def processor(*args, **kwargs):
        return [1, 91, 2, 91], {"multimodal_input": identity}

    monkeypatch.setattr(
        "tensorrt_llm.llmapi.llm.create_input_processor_with_hash",
        lambda *args, **kwargs: processor,
    )
    llm.args = SimpleNamespace(multimodal_config=None)
    disagg = DisaggregatedParams(request_type="context_only")
    llm._preprocess(
        {"prompt_token_ids": [1, 91, 2, 91], "multi_modal_data": {"image": [object()]}},
        SamplingParams(max_tokens=4),
        disagg,
    )
    assert disagg.multimodal_input == asdict(identity)
    identity.multimodal_hashes[0][0] = 99
    assert disagg.multimodal_input["multimodal_hashes"][0][0] == 1


def test_embedding_handoff_captures_cache_identity(llm, monkeypatch):
    from dataclasses import asdict

    from tensorrt_llm.inputs.multimodal import DisaggPrefillMultimodalInputs

    layout = DisaggPrefillMultimodalInputs(
        prompt_token_ids=[1, 91, 2, 91],
        multimodal_lengths=[2],
        multimodal_positions=[1],
        multimodal_embedding_lengths=[2],
        multimodal_item_run_cu_offsets=[0, 2],
        multimodal_run_positions=[1, 3],
        multimodal_run_lengths=[1, 1],
    )
    llm.input_processor.support_mm_disagg = True
    llm.input_processor.build_disagg_prefill_multimodal_inputs = lambda *args: layout
    monkeypatch.setattr("tensorrt_llm.llmapi.llm.maybe_compute_mm_embed_cumsum", lambda *args: None)
    disagg = DisaggregatedParams(
        request_type="context_only",
        multimodal_embedding_handles=[{}],
        multimodal_hashes=[[1, 2, 3, 4, 5, 6, 7, 8]],
    )
    _, _, mm, _ = llm._preprocess(
        {"prompt_token_ids": [1, 91, 2, 91]}, SamplingParams(max_tokens=4), disagg
    )
    assert disagg.multimodal_input == asdict(mm.multimodal_input)
    assert disagg.multimodal_input["multimodal_run_positions"] == [1, 3]


def test_generation_only_cache_reuse_requires_multimodal_identity(llm):
    llm.args = SimpleNamespace(kv_cache_config=SimpleNamespace(enable_block_reuse=True))
    with pytest.raises(ValueError, match="KV reuse requires cache identity"):
        llm._preprocess(
            {"prompt_token_ids": [1, 91, 2], "multi_modal_data": None},
            SamplingParams(max_tokens=4),
            params(),
        )
