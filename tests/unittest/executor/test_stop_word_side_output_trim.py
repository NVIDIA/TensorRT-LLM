# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Stop-word trimming must also trim the per-token side outputs.

When a sequence finishes on a stop word and ``include_stop_str_in_output`` is
False (the OpenAI server default), ``_handle_sequence`` removes the stop tokens
from ``token_ids``.  ``logprobs``, ``generation_logits`` and every value of
``additional_generation_outputs`` are indexed along their first axis by the same
positions, so leaving them untrimmed makes them longer than ``token_ids`` and
any consumer that zips the two together fails -- e.g. the OpenAI server's
``create_logprobs``, which turned the mismatch into an HTTP 400
"Postprocessing error: token_ids and logprobs have different lengths".

This bites models whose ``generation_config.eos_token_id`` is a list: every
entry other than the tokenizer's ``eos_token_id`` becomes a ``stop_token_ids``
entry, so which EOS variant the model happens to emit decides whether the
request finishes via END_ID (fine) or STOP_WORDS (trimmed).  CPU-only; no
engine needed.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.executor.result import GenerationResultBase, Logprob
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only

# Nemotron-3.5 as trtllm-serve configures it: end_id is the tokenizer's
# eos_token_id (11, <|im_end|>) and the other generation_config eos entry
# (2, </s>) is injected into stop_token_ids by SamplingParams._setup.
END_ID = 11
STOP_TOKEN_ID = 2
VOCAB = 32
HIDDEN = 8


def _handle(
    finish_reason, generated, sampling_params, with_logits=False, with_additional_outputs=False
):
    """Run one finished response through the client-side result handling."""
    result = GenerationResultBase(id=0, sampling_params=sampling_params)
    logits = torch.zeros(1, len(generated), VOCAB) if with_logits else None
    # HandleAdditionalOutputs concatenates one [1, beam_width, hidden] slice
    # per generated token, so axis 0 is the token axis.
    additional_generation_outputs = (
        {"generation_output": torch.zeros(len(generated), 1, HIDDEN)}
        if with_additional_outputs
        else None
    )
    response = SimpleNamespace(
        output_token_ids=[list(generated)],
        cum_log_probs=None,
        log_probs=[[{token: Logprob(logprob=-0.5, rank=1)} for token in generated]],
        generation_logits=logits,
        request_perf_metrics=None,
        additional_context_outputs=None,
        additional_generation_outputs=additional_generation_outputs,
    )
    result._done = True
    result._handle_sequence([finish_reason], response, 0)
    return result._outputs[0]


def test_stop_token_trims_logprobs():
    output = _handle(
        tllm.FinishReason.STOP_WORDS,
        [100, 101, 102, STOP_TOKEN_ID],
        SamplingParams(end_id=END_ID, stop_token_ids=[STOP_TOKEN_ID], logprobs=5),
    )
    assert output.stop_reason == STOP_TOKEN_ID
    assert output.token_ids == [100, 101, 102]
    assert len(output.logprobs) == len(output.token_ids)


def test_multi_token_stop_string_trims_logprobs():
    """A stop *string* spans several tokens; all of them must be trimmed."""
    sampling_params = SamplingParams(end_id=END_ID, stop="XY", logprobs=5)
    # Stands in for SamplingParams._setup tokenizing "XY" into two tokens.
    sampling_params._stop_word_ids = [[102, 103]]

    output = _handle(tllm.FinishReason.STOP_WORDS, [100, 101, 102, 103], sampling_params)

    assert output.stop_reason == "XY"
    assert output.token_ids == [100, 101]
    assert len(output.logprobs) == len(output.token_ids)


def test_stop_token_trims_generation_logits():
    output = _handle(
        tllm.FinishReason.STOP_WORDS,
        [100, 101, 102, STOP_TOKEN_ID],
        SamplingParams(
            end_id=END_ID, stop_token_ids=[STOP_TOKEN_ID], logprobs=5, return_generation_logits=True
        ),
        with_logits=True,
    )
    assert output.generation_logits.shape[0] == len(output.token_ids)


def test_stop_token_trims_additional_generation_outputs():
    output = _handle(
        tllm.FinishReason.STOP_WORDS,
        [100, 101, 102, STOP_TOKEN_ID],
        SamplingParams(end_id=END_ID, stop_token_ids=[STOP_TOKEN_ID], logprobs=5),
        with_additional_outputs=True,
    )
    trimmed = output.additional_generation_outputs["generation_output"]
    assert trimmed.shape[0] == len(output.token_ids)
    # The beam and hidden axes must survive untouched.
    assert trimmed.shape[1:] == (1, HIDDEN)


def test_include_stop_str_in_output_keeps_everything():
    output = _handle(
        tllm.FinishReason.STOP_WORDS,
        [100, 101, 102, STOP_TOKEN_ID],
        SamplingParams(
            end_id=END_ID,
            stop_token_ids=[STOP_TOKEN_ID],
            logprobs=5,
            include_stop_str_in_output=True,
        ),
    )
    assert output.token_ids == [100, 101, 102, STOP_TOKEN_ID]
    assert len(output.logprobs) == len(output.token_ids)


def test_end_id_finish_is_untouched():
    """The END_ID path never trimmed, and must stay that way."""
    output = _handle(
        tllm.FinishReason.END_ID,
        [100, 101, 102, END_ID],
        SamplingParams(end_id=END_ID, stop_token_ids=[STOP_TOKEN_ID], logprobs=5),
    )
    assert output.stop_reason is None
    assert len(output.logprobs) == len(output.token_ids) == 4


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("beam_search", [False, True])
def test_sampler_logprobs_survive_prompt_response_wrapper(streaming, beam_search):
    from tensorrt_llm.executor.result import LogProbsResult

    params = SamplingParams(n=2 if beam_search else 1, use_beam_search=beam_search, logprobs=1)
    result = GenerationResultBase(id=0, sampling_params=params)
    result._streaming = streaming
    prompt_logprobs = [{1: Logprob(logprob=-0.25, rank=1)}]
    wrapped = LogProbsResult(prompt=prompt_logprobs)
    sequences = [[10, 11], [20, 21]] if beam_search else [[10, 11]]
    sampler_logprobs = [
        [{token: Logprob(logprob=-0.5, rank=1)} for token in tokens] for tokens in sequences
    ]

    # Multi-beam responses carry cumulative tokens and logprobs; sampling
    # responses carry only the newly generated tokens and logprobs.
    for step in range(2):
        response = SimpleNamespace(
            output_token_ids=[
                tokens[: step + 1] if beam_search else tokens[step : step + 1]
                for tokens in sequences
            ],
            cum_log_probs=None,
            log_probs=[
                values[: step + 1] if beam_search else values[step : step + 1]
                for values in sampler_logprobs
            ],
            generation_logits=None,
            request_perf_metrics=None,
        )
        for beam in range(len(sequences)):
            result._handle_sequence(
                [tllm.FinishReason.NOT_FINISHED] * len(sequences),
                response,
                beam,
                logprobs_result=wrapped,
            )

    for beam, output in enumerate(result._outputs):
        assert output.token_ids == sequences[beam]
        assert output.logprobs == sampler_logprobs[beam]
        assert output.prompt_logprobs == prompt_logprobs


@pytest.mark.parametrize("first_token_logprob_transferred", [False, True])
def test_generation_only_response_preserves_sampler_logprobs(first_token_logprob_transferred):
    result = GenerationResultBase(id=0, sampling_params=SamplingParams(logprobs=1))
    result._disaggregated_params = SimpleNamespace(request_type="generation_only")
    tokens = [10, 11]
    returned_tokens = tokens if first_token_logprob_transferred else tokens[1:]
    sampler_logprobs = [{token: Logprob(logprob=-0.5, rank=1)} for token in returned_tokens]
    response = SimpleNamespace(
        output_token_ids=[tokens],
        cum_log_probs=None,
        log_probs=[sampler_logprobs],
        generation_logits=None,
        request_perf_metrics=None,
    )

    result._handle_sequence([tllm.FinishReason.NOT_FINISHED], response, 0)

    output = result._outputs[0]
    assert output.token_ids == tokens
    assert output.logprobs == sampler_logprobs
    assert output.disaggregated_params is result.disaggregated_params
