# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Beam-search response handling must materialize C++ binding properties once.

``LlmResult.output_token_ids`` is not in ``py_result_properties``, so it falls
through to the C++ binding for ``Result``, which rebuilds the whole (beams, tokens)
nested list on every access. ``_handle_response`` calls ``_handle_sequence``
once per beam, so reading it there costs N full rebuilds per response -- and
because each rebuild is itself O(N), the waste is quadratic in beam width.
"""

import pytest

import tensorrt_llm.bindings.executor as tllm
from tensorrt_llm._torch.pyexecutor.llm_request import LlmResult, PyResult
from tensorrt_llm.executor.result import CompletionOutput, GenerationResultBase
from tensorrt_llm.sampling_params import SamplingParams

BEAMS = 8
TOKENS = 3


class CountingResult:
    """Forwards to a real ``LlmResult``, counting reads of one property."""

    def __init__(self, inner: LlmResult, name: str):
        self._inner = inner
        self._name = name
        self.reads = 0

    def __getattr__(self, item):
        if item == object.__getattribute__(self, "_name"):
            self.reads += 1
        return getattr(object.__getattribute__(self, "_inner"), item)


class StubResponse:
    """Duck-typed response; ``is_llm_response`` only checks for ``has_error``."""

    def __init__(self, result):
        self._result = result

    def has_error(self) -> bool:
        return False

    @property
    def result(self):
        return self._result

    @property
    def should_abort(self) -> bool:
        return False


class StubGenerationResult(GenerationResultBase):
    """A real ``GenerationResultBase`` with the side-effecting hooks stubbed."""

    def __init__(self, sampling_params: SamplingParams, n_outputs: int):
        super().__init__(id=0, sampling_params=sampling_params)
        # Seed minimal response state and bypass tracing/speculative metrics
        # below to isolate token materialization without a full executor.
        self._outputs = [CompletionOutput(index=i) for i in range(n_outputs)]
        self._streaming = False
        self._multimodal_hashes = None
        self._context_logits = None

    def do_tracing(self, *args, **kwargs) -> None:
        pass

    def _maybe_fill_spec_dec_perf_metrics(self, *args, **kwargs) -> None:
        pass


def make_counting_result(num_rows: int) -> CountingResult:
    raw = tllm.Result()
    raw.output_token_ids = [[1000 + b + t for t in range(TOKENS)] for b in range(num_rows)]
    raw.finish_reasons = [tllm.FinishReason.END_ID] * num_rows
    raw.is_final = True
    raw.sequence_index = 0
    py_result = PyResult(prompt_len=8, max_new_tokens=TOKENS, return_log_probs=False)
    llm_result = LlmResult(result=raw, py_result=py_result, is_final=True)
    return CountingResult(llm_result, "output_token_ids")


@pytest.mark.cpu_only
def test_beam_response_materializes_output_token_ids_once():
    sampling_params = SamplingParams(
        n=BEAMS, best_of=BEAMS, use_beam_search=True, max_tokens=TOKENS, temperature=0.0
    )
    counted = make_counting_result(BEAMS)
    result = StubGenerationResult(sampling_params, BEAMS)

    result._handle_response(StubResponse(counted))

    assert counted.reads == 1, (
        f"output_token_ids was materialized {counted.reads} times for "
        f"{BEAMS} beams; each access rebuilds the full (beams, tokens) list "
        f"across the C++ binding boundary, so it must be hoisted out of the loop"
    )
    for beam in range(BEAMS):
        expected = [1000 + beam + t for t in range(TOKENS)]
        assert result._outputs[beam].token_ids == expected


@pytest.mark.cpu_only
def test_non_beam_response_materializes_output_token_ids_once():
    sampling_params = SamplingParams(n=1, max_tokens=TOKENS, temperature=0.0)
    counted = make_counting_result(1)
    result = StubGenerationResult(sampling_params, 1)

    result._handle_response(StubResponse(counted))

    assert counted.reads == 1
    assert result._outputs[0].token_ids == [1000 + t for t in range(TOKENS)]
