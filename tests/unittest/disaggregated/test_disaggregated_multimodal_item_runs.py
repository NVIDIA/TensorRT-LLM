# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.executor.utils import to_binding_multimodal_input
from tensorrt_llm.inputs.multimodal import MultimodalInput, MultimodalParams
from tensorrt_llm.llmapi.llm import BaseLLM
from tensorrt_llm.sampling_params import SamplingParams

MM_HASHES = [[1, 2, 3, 4, 5, 6, 7, 8]]
MM_ITEM_RUNS = [[(1, 1, []), (3, 1, [])]]
MM_HANDLES = [{"tensor_size": [2, 8], "ipc_handle": "handle"}]


def _runs_to_tuples(item_runs):
    return [[_run_to_tuple(run) for run in runs] for runs in item_runs]


def _run_to_tuple(run):
    if hasattr(run, "prompt_start"):
        return run.prompt_start, run.run_length, list(run.non_embed_offsets)
    start, length, non_embed_offsets = run
    return start, length, list(non_embed_offsets)


class _FakeDisaggInputProcessor:
    support_mm_disagg = True

    def __init__(self, prompt_token_ids, *, fail_on_get_prompt_token_ids=False):
        self.prompt_token_ids = prompt_token_ids
        self.fail_on_get_prompt_token_ids = fail_on_get_prompt_token_ids
        self.seen_mm_item_runs = None

    def get_prompt_token_ids(self, inputs, mm_handles, mm_item_runs=None):
        if self.fail_on_get_prompt_token_ids:
            raise AssertionError("get_prompt_token_ids should not be called")
        assert mm_handles == MM_HANDLES
        self.seen_mm_item_runs = mm_item_runs
        return self.prompt_token_ids

    def get_vocab_size(self):
        return None

    def get_mm_token_ids(self):
        return torch.tensor([999])

    def get_mm_special_token_ids(self):
        return None


class _FakeTokenizer:
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(self, text, return_tensors=None):
        assert text
        assert return_tensors == "pt"
        return SimpleNamespace(input_ids=torch.tensor([self.token_ids]))


def _make_llm(prompt_token_ids, *, fail_on_get_prompt_token_ids=False):
    llm = BaseLLM.__new__(BaseLLM)
    llm.input_processor = _FakeDisaggInputProcessor(
        prompt_token_ids, fail_on_get_prompt_token_ids=fail_on_get_prompt_token_ids
    )
    return llm


def _make_qwen_input_processor(processor_cls, *, token_ids, hidden_size=8):
    processor = processor_cls.__new__(processor_cls)
    processor._tokenizer = _FakeTokenizer(token_ids)
    processor._config = SimpleNamespace(
        image_token_id=101,
        video_token_id=102,
        text_config=SimpleNamespace(hidden_size=hidden_size),
        vision_config=SimpleNamespace(deepstack_visual_indexes=[]),
    )
    processor.tllm_multimodal_token_id = 999
    return processor


def _response(
    token_id: int,
    *,
    finished: bool,
    context_phase_params=None,
    mm_embedding_handles=None,
):
    finish_reason = tllm.FinishReason.END_ID if finished else tllm.FinishReason.NOT_FINISHED
    response_result = SimpleNamespace(
        is_final=finished,
        context_phase_params=context_phase_params,
        decoding_iter=0,
        cached_tokens=0,
        avg_decoded_tokens_per_iter=None,
        finish_reasons=[finish_reason],
        output_token_ids=[[token_id]],
        sequence_index=0,
        cum_log_probs=None,
        log_probs=None,
        generation_logits=None,
        additional_context_outputs=None,
        additional_generation_outputs=None,
        request_perf_metrics=None,
        time_breakdown_metrics=None,
        context_logits=None,
        mm_embedding_handles=mm_embedding_handles,
    )
    return SimpleNamespace(has_error=lambda: False, result=response_result)


def test_generation_result_preserves_item_runs_through_epd_handoff():
    multimodal_input = MultimodalInput.from_components(MM_HASHES, MM_ITEM_RUNS)
    request = GenerationRequest(
        prompt_token_ids=[11, 999, 77, 999, 12],
        sampling_params=SamplingParams(max_tokens=4),
        multimodal_params=MultimodalParams(multimodal_input=multimodal_input),
    )
    result = GenerationResult(request)

    result._handle_response(_response(10, finished=False, mm_embedding_handles=MM_HANDLES))
    assert result.disaggregated_params.multimodal_hashes == MM_HASHES
    assert _runs_to_tuples(result.disaggregated_params.multimodal_item_runs) == MM_ITEM_RUNS

    context_phase_params = SimpleNamespace(
        first_gen_tokens=[10],
        req_id=123,
        opaque_state=b"state",
        draft_tokens=None,
        ctx_dp_rank=0,
        disagg_info_endpoint="tcp://127.0.0.1:9000",
    )
    result._handle_response(_response(11, finished=True, context_phase_params=context_phase_params))

    assert result.disaggregated_params.multimodal_embedding_handles is None
    assert result.disaggregated_params.multimodal_hashes == MM_HASHES
    assert _runs_to_tuples(result.disaggregated_params.multimodal_item_runs) == MM_ITEM_RUNS


def test_generation_result_accepts_embedding_handles_without_hashes():
    request = GenerationRequest(
        prompt_token_ids=[11],
        sampling_params=SamplingParams(max_tokens=4),
    )
    result = GenerationResult(request)

    result._handle_response(_response(10, finished=False, mm_embedding_handles=MM_HANDLES))

    assert result.disaggregated_params.multimodal_embedding_handles == MM_HANDLES
    assert result.disaggregated_params.multimodal_hashes is None
    assert result.disaggregated_params.multimodal_item_runs is None


def test_to_binding_multimodal_input_uses_binding_constructor_contract():
    multimodal_input = MultimodalInput.from_components(
        MM_HASHES, MM_ITEM_RUNS, mm_uuids=["video-uuid"]
    )

    binding_input = to_binding_multimodal_input(multimodal_input)

    assert binding_input.multimodal_hashes == MM_HASHES
    assert _runs_to_tuples(binding_input.multimodal_item_runs) == MM_ITEM_RUNS
    assert binding_input.multimodal_uuids == ["video-uuid"]


def test_llm_preprocess_reconstructs_multimodal_input_with_item_runs():
    llm = _make_llm([11, 999, 77, 999, 12])
    disaggregated_params = DisaggregatedParams(
        multimodal_embedding_handles=MM_HANDLES,
        multimodal_hashes=MM_HASHES,
        multimodal_item_runs=MM_ITEM_RUNS,
    )

    prompt_token_ids, prompt, query_token_ids, multimodal_params = BaseLLM._preprocess(
        llm, {"prompt": "describe image"}, SamplingParams(max_tokens=1), disaggregated_params
    )

    assert prompt_token_ids == [11, 999, 77, 999, 12]
    assert _runs_to_tuples(llm.input_processor.seen_mm_item_runs) == MM_ITEM_RUNS
    assert prompt == "describe image"
    assert query_token_ids is None
    assert multimodal_params.multimodal_data["multimodal_embedding"] == MM_HANDLES
    assert multimodal_params.multimodal_input.multimodal_hashes == MM_HASHES
    assert _runs_to_tuples(multimodal_params.multimodal_input.multimodal_item_runs) == MM_ITEM_RUNS


def test_llm_preprocess_uses_disaggregated_prompt_token_ids_when_provided():
    prompt_token_ids = [11, 999, 999, 77, 999, 999, 12]
    llm = _make_llm([], fail_on_get_prompt_token_ids=True)
    disaggregated_params = DisaggregatedParams(
        multimodal_embedding_handles=MM_HANDLES,
        multimodal_hashes=MM_HASHES,
        multimodal_item_runs=MM_ITEM_RUNS,
    )

    processed_ids, prompt, query_token_ids, multimodal_params = BaseLLM._preprocess(
        llm,
        {"prompt": "describe video", "prompt_token_ids": prompt_token_ids},
        SamplingParams(max_tokens=1),
        disaggregated_params,
    )

    assert processed_ids == prompt_token_ids
    assert llm.input_processor.seen_mm_item_runs is None
    assert prompt == "describe video"
    assert query_token_ids is None
    assert multimodal_params.multimodal_data["multimodal_embedding"] == MM_HANDLES
    assert multimodal_params.multimodal_input.multimodal_hashes == MM_HASHES
    assert _runs_to_tuples(multimodal_params.multimodal_input.multimodal_item_runs) == MM_ITEM_RUNS


def test_llm_preprocess_rejects_out_of_prompt_disaggregated_item_runs():
    llm = _make_llm([11, 999, 77, 999, 12])
    disaggregated_params = DisaggregatedParams(
        multimodal_embedding_handles=MM_HANDLES,
        multimodal_hashes=MM_HASHES,
        multimodal_item_runs=[[(99, 1, [])]],
    )

    with pytest.raises(ValueError, match="exceeding input sequence length"):
        BaseLLM._preprocess(
            llm, {"prompt": "describe image"}, SamplingParams(max_tokens=1), disaggregated_params
        )


def test_llm_preprocess_accepts_embedding_handles_without_hashes():
    llm = _make_llm([11, 999, 12])
    disaggregated_params = DisaggregatedParams(
        multimodal_embedding_handles=MM_HANDLES,
    )

    prompt_token_ids, prompt, query_token_ids, multimodal_params = BaseLLM._preprocess(
        llm, {"prompt": "describe image"}, SamplingParams(max_tokens=1), disaggregated_params
    )

    assert prompt_token_ids == [11, 999, 12]
    assert llm.input_processor.seen_mm_item_runs is None
    assert prompt == "describe image"
    assert query_token_ids is None
    assert multimodal_params.multimodal_data["multimodal_embedding"] == MM_HANDLES
    assert multimodal_params.multimodal_input is None


def test_qwen_vl_disaggregated_prompt_expansion_rejects_marker_per_run_layout():
    from tensorrt_llm._torch.models.modeling_multimodal_utils import (
        expand_mm_prompt_token_ids_for_disagg,
    )

    input_ids = torch.tensor([11, 102, 77, 102, 12], dtype=torch.int64)

    with pytest.raises(ValueError, match="multimodal items"):
        expand_mm_prompt_token_ids_for_disagg(
            input_ids,
            multimodal_token_ids=[101, 102],
            mm_handles=[
                {
                    "tensor_size": [4, 8],
                    "ipc_handle": "video-handle",
                }
            ],
            placeholder_id=999,
            mm_item_runs=[[(1, 2, []), (4, 2, [])]],
        )


def test_qwen_vl_disaggregated_prompt_expansion_rejects_non_cpu_input_ids():
    from tensorrt_llm._torch.models.modeling_multimodal_utils import (
        expand_mm_prompt_token_ids_for_disagg,
    )

    input_ids = torch.empty(3, dtype=torch.int64, device="meta")

    with pytest.raises(ValueError, match="CPU-resident"):
        expand_mm_prompt_token_ids_for_disagg(
            input_ids,
            multimodal_token_ids=[101, 102],
            mm_handles=[
                {
                    "tensor_size": [4, 8],
                    "ipc_handle": "video-handle",
                }
            ],
            placeholder_id=999,
            mm_item_runs=[[(1, 2, []), (3, 2, [])]],
        )


def test_qwen_vl_disaggregated_prompt_expansion_accepts_one_marker_per_item():
    from tensorrt_llm._torch.models.modeling_multimodal_utils import (
        expand_mm_prompt_token_ids_for_disagg,
    )

    input_ids = torch.tensor([11, 102, 12], dtype=torch.int64)

    expanded_ids = expand_mm_prompt_token_ids_for_disagg(
        input_ids,
        multimodal_token_ids=[101, 102],
        mm_handles=[
            {
                "tensor_size": [4, 8],
                "ipc_handle": "video-handle",
            }
        ],
        placeholder_id=999,
        mm_item_runs=[[(1, 2, []), (3, 2, [])]],
    )

    assert expanded_ids == [11, 999, 999, 999, 999, 12]


def test_qwen_vl_disaggregated_prompt_expansion_accepts_video_without_item_runs():
    from tensorrt_llm._torch.models.modeling_multimodal_utils import (
        expand_mm_prompt_token_ids_for_disagg,
    )

    input_ids = torch.tensor([11, 102, 12], dtype=torch.int64)

    expanded_ids = expand_mm_prompt_token_ids_for_disagg(
        input_ids,
        multimodal_token_ids=[101, 102],
        mm_handles=[
            {
                "tensor_size": [2, 8],
                "ipc_handle": "video-handle",
            }
        ],
        placeholder_id=999,
    )

    assert expanded_ids == [11, 999, 999, 12]


@pytest.mark.parametrize(
    "processor_import_path, processor_name",
    [
        ("tensorrt_llm._torch.models.modeling_qwen2vl", "Qwen2_5VLInputProcessorBase"),
        ("tensorrt_llm._torch.models.modeling_qwen3vl", "Qwen3VLInputProcessorBase"),
    ],
)
def test_qwen_vl_disagg_input_processors_expand_video_item_runs(
    processor_import_path, processor_name
):
    module = __import__(processor_import_path, fromlist=[processor_name])
    processor_cls = getattr(module, processor_name)
    processor = _make_qwen_input_processor(
        processor_cls,
        token_ids=[11, 102, 12],
    )

    expanded_ids = processor.get_prompt_token_ids(
        {"prompt": "describe video"},
        [
            {
                "tensor_size": [4, 8],
                "ipc_handle": "video-handle",
            }
        ],
        mm_item_runs=[[(1, 2, []), (3, 2, [])]],
    )

    assert expanded_ids == [11, 999, 999, 999, 999, 12]


def test_qwen3_vl_moe_registers_shared_item_run_aware_processor():
    from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLInputProcessorBase
    from tensorrt_llm._torch.models.modeling_qwen3vl_moe import Qwen3MoeVLModel
    from tensorrt_llm.inputs.registry import INPUT_PROCESSOR_REGISTRY

    assert Qwen3MoeVLModel.support_mm_disagg is True
    assert (
        INPUT_PROCESSOR_REGISTRY._input_processors_cls_by_model_type[Qwen3MoeVLModel]
        is Qwen3VLInputProcessorBase
    )
