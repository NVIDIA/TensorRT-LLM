# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib as _pl
from typing import Any

import pytest
import torch
from test_beam_search_util import (BeamSearchTestOutput, DummyConfigLoader,
                                   DummyWeightLoader, get_expected_outputs)
from utils.llm_data import llm_models_root
from utils.util import assert_no_cuda_sync, force_ampere

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.models.checkpoints import HfCheckpointLoader
from tensorrt_llm._torch.pyexecutor.llm_request import (LlmRequest,
                                                        SamplingConfig)
from tensorrt_llm._torch.pyexecutor.sampler import BeamHistory, TorchSampler
from tensorrt_llm._torch.pyexecutor.sampling_utils import (
    BEAM_SEARCH_PAD_TOKEN, BeamSearchMetadata, FinishReason,
    beam_search_sampling_batch)
from tensorrt_llm.executor import RequestError
from tensorrt_llm.executor.result import CompletionOutput, GenerationResult
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig


@pytest.fixture(scope="module")
def input_prompts():
    return [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


@pytest.fixture(scope="module")
def fixed_params():
    return {"max_tokens": 8, "max_beam_width": 2}


@pytest.fixture(scope="module",
                params=[("TRTLLMSampler", False), ("TorchSampler", False),
                        ("TorchSampler", True)])
def sampling_information(request):
    return request.param


@pytest.fixture(scope="module")
def model_kwargs(fixed_params, sampling_information) -> dict[str, Any]:

    assert fixed_params[
        "max_beam_width"] == 2, "This test only works for a beam width of 2"
    return dict(
        model=_pl.Path("dummy_path"),
        checkpoint_loader=HfCheckpointLoader(
            weight_loader=DummyWeightLoader(),
            config_loader=DummyConfigLoader(),
        ),
        sampler_type=sampling_information[0],
        disable_flashinfer_sampling=sampling_information[1],
    )


def _build_llm(fixed_params, input_prompts, model_kwargs):
    return LLM(
        **model_kwargs,
        kv_cache_config=KvCacheConfig(max_tokens=10000),
        max_batch_size=fixed_params["max_beam_width"] * len(
            input_prompts
        ),  # use small batch size to prevent large buffers from possibly hiding wrong data accesses.
        max_seq_len=32,
        max_beam_width=fixed_params["max_beam_width"],
        disable_overlap_scheduler=True,
        cuda_graph_config=None,
    )


@pytest.fixture(scope="module")
def llm(fixed_params, input_prompts, model_kwargs):
    llm = _build_llm(fixed_params, input_prompts, model_kwargs)
    yield llm
    llm.shutdown()


@pytest.fixture(scope="module")
def llm_cuda_graph(fixed_params, input_prompts, model_kwargs):
    llm = LLM(
        **model_kwargs,
        kv_cache_config=KvCacheConfig(max_tokens=10000),
        max_batch_size=fixed_params["max_beam_width"] * len(
            input_prompts
        ),  # use small batch size to prevent large buffers from possibly hiding wrong data accesses.
        max_seq_len=32,
        max_beam_width=fixed_params["max_beam_width"],
        disable_overlap_scheduler=False,
        cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 4, 8],
                                          enable_padding=True),
    )
    yield llm
    llm.shutdown()


def check_generation_logits(beam: CompletionOutput,
                            sampling_params: SamplingParams,
                            valid_tokens: int | None) -> None:
    """Check if the generation logits have the correct shape"""
    if sampling_params.return_generation_logits:
        gen_logits = beam.generation_logits
        generated_tokens = valid_tokens if valid_tokens is not None else sampling_params.max_tokens
        assert gen_logits is not None, "generation logits should not be None"
        assert gen_logits.ndim == 2, f"generation logits should have 2 dimensions, but got {gen_logits.ndim}"
        assert gen_logits.shape[
            0] == generated_tokens, f"expected {generated_tokens} generation logits, but got {gen_logits.shape[0]}"
    else:
        assert beam.generation_logits is None, "generation logits should be None"


def check_logprobs(beam: CompletionOutput, sampling_params: SamplingParams,
                   valid_tokens: int | None) -> None:
    """Check if the logprobs have the correct shape"""
    if sampling_params.logprobs is not None:
        generated_tokens = valid_tokens if valid_tokens is not None else sampling_params.max_tokens
        assert len(
            beam.logprobs
        ) == generated_tokens, f"expected {generated_tokens} logprobs, but got {len(beam.logprobs)}"
        log_sum = 0.0
        for logprob_dict in (beam.logprobs):
            for logprob_value in logprob_dict.values():
                log_sum += logprob_value.logprob
        assert log_sum == beam.cumulative_logprob, f"expected {beam.cumulative_logprob} logprob, but got {log_sum}"
    else:
        assert len(beam.logprobs) == 0, "logprobs should be empty"


def check_cache_indirection(beam: CompletionOutput,
                            sampling_params: SamplingParams,
                            reference_cache_indirection: torch.Tensor,
                            prompt_length: int, beam_idx: int,
                            valid_tokens: int | None) -> None:
    """Check if the cache indirection seen by the model is the same as the expected cache indirection"""
    cache_indirection = beam.additional_generation_outputs["cache_indirection"]
    assert cache_indirection is not None, "cache indirection should not be None"
    assert cache_indirection.shape[
        1] == sampling_params.best_of, f"expected {sampling_params.best_of} entries in dim 1 of cache indirection, but got {cache_indirection.shape[1]}"

    num_generated_tokens = valid_tokens if valid_tokens is not None else sampling_params.max_tokens
    # We return the cache indirection before the sampling step, therefore cache indirection does not reflect changes during the sampling of the last token
    num_valid_cache_indirection = num_generated_tokens - 1

    # check if the cache indirection is correct for the given deterministic input prompt
    # Check only the last cache indirection
    last_cache_indirection = cache_indirection[num_valid_cache_indirection,
                                               beam_idx]

    assert all(last_cache_indirection[:prompt_length] ==
               0), "prompt tokens should have a cache indirection of 0"
    # remove the prompt tokens from the cache indirection and check if the remaining cache indirection is correct
    valid_cache_indirection = last_cache_indirection[
        prompt_length:prompt_length + num_valid_cache_indirection]
    assert all(
        valid_cache_indirection == reference_cache_indirection[
            beam_idx, :num_valid_cache_indirection]
    ), f"expected {reference_cache_indirection[beam_idx, :num_valid_cache_indirection].tolist()} cache indirection, but got {valid_cache_indirection.tolist()}"


def validate_output_beam(beam: CompletionOutput,
                         expected_outputs: BeamSearchTestOutput,
                         sampling_params: SamplingParams, prompt_length: int,
                         beam_idx: int) -> None:
    """Perform several checks on the output of a single beam"""

    valid_tokens = None
    if sampling_params.stop_token_ids is not None:
        if sampling_params.stop_token_ids[0] in expected_outputs.outputs[
                beam_idx].tolist():
            valid_tokens = expected_outputs.outputs[beam_idx].tolist().index(
                sampling_params.stop_token_ids[0]) + 1

    check_generation_logits(beam, sampling_params, valid_tokens)
    check_logprobs(beam, sampling_params, valid_tokens)
    check_cache_indirection(beam, sampling_params,
                            expected_outputs.cache_indirection, prompt_length,
                            beam_idx, valid_tokens)
    # Check output similarity

    assert valid_tokens is None or valid_tokens > 0
    assert beam.token_ids == expected_outputs.outputs[
        beam_idx, :valid_tokens].tolist(
        ), f"expected {expected_outputs.outputs[beam_idx, :valid_tokens].tolist()} token ids, but got {beam.token_ids}"


def check_context_logits(output: GenerationResult,
                         sampling_params: SamplingParams):
    """Check if the context logits have the correct shape"""
    if sampling_params.return_context_logits:
        assert output.context_logits is not None, "context logits should not be None"
        assert len(output.prompt_token_ids) == output.context_logits.shape[
            0], f"expected {len(output.prompt_token_ids)} context logits, but got {output.context_logits.shape[0]}"
    else:
        assert output.context_logits is None, "context logits should be None"


def validate_output(output: GenerationResult, input_prompt: list[int],
                    sampling_params: SamplingParams) -> None:
    """Perform several checks on the output of a single prompt"""
    check_context_logits(output, sampling_params)

    # validate number of outputs equals beam width
    num_output_beams = sampling_params.n
    assert len(
        output.outputs
    ) == num_output_beams, f"expected {num_output_beams} outputs, but got {len(output.outputs)}"
    # check each beam
    expected_outputs = get_expected_outputs(
        input_prompt[-1], num_iterations=sampling_params.max_tokens)
    for beam_idx, beam in enumerate(output.outputs):
        validate_output_beam(beam, expected_outputs, sampling_params,
                             len(input_prompt), beam_idx)


def validate_outputs(llm: LLM, input_prompts: list[list[int]],
                     sampling_params: SamplingParams) -> None:
    """Generate outputs for a list of prompts and validate the outputs"""
    outputs = llm.generate(input_prompts, sampling_params=sampling_params)
    num_prompts = len(input_prompts)

    assert len(
        outputs
    ) == num_prompts, f"expected {num_prompts} outputs, but got {len(outputs)}"
    for output_idx, output in enumerate(outputs):
        validate_output(output, input_prompts[output_idx], sampling_params)


###########################################################################
# End to end tests
###########################################################################


@pytest.mark.parametrize("return_log_probs", [True, False])
@pytest.mark.parametrize("gather_generation_logits", [True, False])
@pytest.mark.parametrize("gather_context_logits", [True, False])
@pytest.mark.parametrize("num_output_beams", [1, 2])
@pytest.mark.parametrize("num_prompts", [1, 2])
@pytest.mark.threadleak(enabled=False)
def test_beam_search_e2e(
    gather_context_logits: bool,
    gather_generation_logits: bool,
    return_log_probs: bool,
    num_output_beams: int,
    num_prompts: int,
    llm,
    fixed_params,
    input_prompts,
) -> None:
    if return_log_probs and num_prompts > 1 and llm.args.sampler_type == "TRTLLMSampler":
        pytest.skip(
            "Beam search currently does not support return_log_probs with multiple prompts"
        )
    if return_log_probs and llm.args.sampler_type == "TRTLLMSampler":
        pytest.skip(
            "Beam search on TRTLLMSampler does not correctly handle log_probs if called multiple times"
        )

    # create sampling parameters
    # additional_model_outputs is used to gather the cache indirection from the model.
    sampling_params = SamplingParams(
        max_tokens=fixed_params["max_tokens"],
        n=num_output_beams,
        best_of=fixed_params["max_beam_width"],
        use_beam_search=True,
        return_context_logits=gather_context_logits,
        return_generation_logits=gather_generation_logits,
        logprobs=return_log_probs,
        end_id=-1,
        additional_model_outputs=["cache_indirection"],
    )
    validate_outputs(llm, input_prompts[:num_prompts], sampling_params)


@pytest.mark.parametrize("return_log_probs", [True, False])
@pytest.mark.parametrize("gather_generation_logits", [True, False])
@pytest.mark.parametrize("gather_context_logits", [True, False])
@pytest.mark.parametrize("num_output_beams", [1, 2])
@pytest.mark.parametrize("num_prompts", [1, 2, 3])
@pytest.mark.parametrize("stop_token_ids", [[15], None])
@pytest.mark.threadleak(enabled=False)
def test_beam_search_e2e_cuda_graph_and_overlap(
    gather_context_logits: bool,
    gather_generation_logits: bool,
    return_log_probs: bool,
    num_output_beams: int,
    num_prompts: int,
    stop_token_ids: list[int] | None,
    llm_cuda_graph,
    fixed_params,
    input_prompts,
) -> None:
    if return_log_probs and num_prompts > 1 and llm_cuda_graph.args.sampler_type == "TRTLLMSampler":
        pytest.skip(
            "Beam search currently does not support return_log_probs with multiple prompts"
        )
    if return_log_probs and llm_cuda_graph.args.sampler_type == "TRTLLMSampler":
        pytest.skip(
            "Beam search on TRTLLMSampler does not correctly handle log_probs if called multiple times"
        )
    if stop_token_ids is not None and llm_cuda_graph.args.sampler_type == "TRTLLMSampler":
        pytest.skip(
            "Beam search on TRTLLMSampler does not correctly handle stop_token_ids"
        )
    # create sampling parameters
    # additional_model_outputs is used to gather the cache indirection from the model.
    sampling_params = SamplingParams(
        max_tokens=fixed_params["max_tokens"],
        n=num_output_beams,
        best_of=fixed_params["max_beam_width"],
        use_beam_search=True,
        return_context_logits=gather_context_logits,
        return_generation_logits=gather_generation_logits,
        logprobs=return_log_probs,
        end_id=-1,
        stop_token_ids=stop_token_ids,
        additional_model_outputs=["cache_indirection"],
    )
    validate_outputs(llm_cuda_graph, input_prompts[:num_prompts],
                     sampling_params)


###########################################################################
# Unit tests
###########################################################################
class GeneralTestParams:
    # Test Parameters for the update_beam_history and finish_beams tests
    beam_width = 3
    max_beam_width = 4
    max_batch_size = 5
    max_seq_len = 123
    input_tokens = [20, 21, 22, 23, 24]
    prompt_len = len(input_tokens)
    num_generated_tokens = 5
    seq_len = prompt_len + num_generated_tokens
    num_logprobs = 0
    seq_slot = 4
    end_id = 99
    batch_size = 2
    vocab_size = 100


def test_beam_search_sampling_batch_basic():
    """Test basic beam search sampling functionality."""

    test_params = GeneralTestParams()
    batch_size = test_params.batch_size
    beam_width = test_params.beam_width
    vocab_size = test_params.vocab_size
    max_batch_size = test_params.max_batch_size
    seq_len = test_params.seq_len
    temperature = 1.0

    # Create logits: [batch_size * beam_width, vocab_size]
    torch.manual_seed(42)
    logits = torch.randn((batch_size * beam_width, vocab_size),
                         dtype=torch.float32)
    for entry in range(batch_size * beam_width):
        assert (logits[entry] != logits[entry, 0]).sum(
        ) > 0, "Logits of a sequence must not only contain the same value. Otherwise change the seed."

    # create a randomly filled cache indirection
    cache_indirection = torch.randint(
        0,
        beam_width,
        (
            max_batch_size, beam_width, seq_len + 1
        ),  # +1 as we should not be calling sampling, when seq_len is already at the max
        dtype=torch.int32)
    assert cache_indirection.sum(
    ) > 0, "Cache indirection must not only contain zeros. Otherwise change the seed."
    # create a result tensor for the cache indirection that will be updated by the beam search sampling
    cache_indirection_result = cache_indirection.clone()
    # Fill this buffer with invalid values
    cache_indirection_buffer = torch.full(
        (max_batch_size, beam_width, seq_len + 1), -1, dtype=torch.int32)

    # create a zero filled cumulative log probs
    cum_log_probs = torch.zeros((max_batch_size, beam_width),
                                dtype=torch.float32)
    # create a result tensor for the cumulative log probs that will be updated by the beam search sampling
    cum_log_probs_result = cum_log_probs.clone()

    # add an offset, so that seq slots is not just the first few entries
    seq_slots = torch.arange(
        batch_size, dtype=torch.int64) + (max_batch_size - batch_size) // 2
    seq_lens = torch.full((batch_size, ), seq_len, dtype=torch.int32)

    # we will finish the last beam of the first request  (Note: This also enforces a beam swap in the first request)
    finished_beams = torch.zeros((max_batch_size, beam_width),
                                 dtype=torch.int32)
    finished_beams[seq_slots[0], beam_width - 1] = FinishReason.STOP_WORDS.value
    finished_beams_result = finished_beams.clone()

    # define our end ids for each request
    end_ids = torch.tensor([vocab_size - 1] * batch_size, dtype=torch.int32)

    new_log_probs = torch.zeros((max_batch_size, beam_width),
                                dtype=torch.float32)
    predecessor_beams = torch.zeros((max_batch_size, beam_width),
                                    dtype=torch.int32)

    new_log_probs_result = new_log_probs.clone()
    predecessor_beams_result = predecessor_beams.clone()

    # Create BeamSearchMetadata
    beam_search_args = BeamSearchMetadata(
        cache_indirection=cache_indirection_result,
        cache_indirection_buffer=cache_indirection_buffer,
        cum_log_probs=cum_log_probs_result,
        seq_slots=seq_slots,
        seq_lens=seq_lens,
        finished_beams=finished_beams_result,
        new_log_probs=new_log_probs_result,
        predecessor_beams=predecessor_beams_result,
        end_ids=end_ids,
    )

    # Run beam search sampling
    with assert_no_cuda_sync():
        next_tokens, softmax = beam_search_sampling_batch(
            logits=logits,
            beam_width_in=beam_width,
            beam_width_out=beam_width,
            beam_search_args=beam_search_args,
            temperature=temperature,
            generator=None,
            return_probs=True,
        )

    # Validate output shapes
    expected_tokens_shape = (batch_size, beam_width)
    assert next_tokens.shape == expected_tokens_shape, (
        f"Expected shape {expected_tokens_shape}, got {next_tokens.shape}")
    expected_softmax_shape = (batch_size, beam_width, vocab_size)
    assert softmax.shape == expected_softmax_shape, (
        f"Expected shape {expected_softmax_shape}, got {softmax.shape}")

    # Validate tokens are within vocab range
    assert torch.all(next_tokens[1:] >= 0), "Tokens out of vocab range"
    # First request has finished beams. Some beams may have BEAM_SEARCH_PAD_TOKEN (-1) as a token
    assert torch.all(
        next_tokens[0] >= BEAM_SEARCH_PAD_TOKEN), "Tokens out of vocab range"
    assert torch.all(next_tokens < vocab_size), "Tokens out of vocab range"

    # Validate softmax probabilities sum to 1
    torch.testing.assert_close(softmax.sum(dim=-1),
                               torch.ones(batch_size, beam_width))

    # Setup our expected values:

    # get the top tokens and beams for each request
    logprobs = torch.log_softmax(logits, dim=-1)
    # adjust our expected log probs accordingly
    logprobs[beam_width - 1] = float('-inf')
    logprobs[beam_width - 1, end_ids[0]] = 0
    logprobs = logprobs.view(batch_size, beam_width * vocab_size)

    _, top_indices = torch.topk(logprobs, k=beam_width, dim=-1, sorted=True)

    top_tokens = top_indices % vocab_size
    top_beams = top_indices // vocab_size

    torch.cuda.synchronize()

    # Validate cache indirection was updated
    for req_idx, seq_slot in enumerate(seq_slots):
        for beam_idx in range(beam_width):
            ideal_beam = top_beams[req_idx][beam_idx]
            torch.testing.assert_close(
                cache_indirection_result[seq_slot, beam_idx, :seq_len],
                cache_indirection[seq_slot, ideal_beam, :seq_len])
            assert cache_indirection_result[
                seq_slot, beam_idx,
                seq_len] == beam_idx, "The last slot of the cache indirection should equal beam_idx"
    # Validate cache indirection buffer was updated
    for req_idx, seq_slot in enumerate(seq_slots):
        for beam_idx in range(beam_width):
            torch.testing.assert_close(
                cache_indirection_buffer[seq_slot, beam_idx, :],
                cache_indirection[seq_slot, beam_idx, :])
    # Validate cumulative log probs were updated
    for req_idx, seq_slot in enumerate(seq_slots):
        for beam_idx in range(beam_width):
            predecessor_beam = top_beams[req_idx][beam_idx]
            old_scores = cum_log_probs[seq_slot, predecessor_beam]
            new_scores = cum_log_probs_result[seq_slot, beam_idx]
            if finished_beams_result[
                    seq_slot, beam_idx] != FinishReason.NOT_FINISHED.value:
                # finished beams do not calculate the logprobs but set the end tokens logprobs to 0
                torch.testing.assert_close(new_scores, old_scores + 0.0)
            else:
                torch.testing.assert_close(
                    new_scores, old_scores + torch.log_softmax(
                        logits[req_idx * beam_width + predecessor_beam],
                        dim=-1)[top_tokens[req_idx][beam_idx]])
    # Validate finished beams were updated: TODO -- This test currently always passes, as finished beams is always 0.
    for req_idx, seq_slot in enumerate(seq_slots):
        for beam_idx in range(beam_width):
            predecessor_beam = top_beams[req_idx][beam_idx]
            torch.testing.assert_close(
                finished_beams_result[seq_slot, beam_idx],
                finished_beams[seq_slot, predecessor_beam])
    # test the new log probs
    for req_idx, seq_slot in enumerate(seq_slots):
        for beam_idx in range(beam_width):
            predecessor_beam = top_beams[req_idx][beam_idx]
            torch.testing.assert_close(
                cum_log_probs_result[seq_slot, beam_idx] -
                new_log_probs_result[seq_slot, beam_idx],
                cum_log_probs[seq_slot, predecessor_beam])
    # test the predecessor beams
    for req_idx, seq_slot in enumerate(seq_slots):
        for beam_idx in range(beam_width):
            predecessor_beam = top_beams[req_idx][beam_idx]
            torch.testing.assert_close(
                predecessor_beams_result[seq_slot, beam_idx],
                torch.tensor(predecessor_beam, dtype=torch.int32))


def create_default_request(test_params: GeneralTestParams) -> LlmRequest:
    sampling_params = SamplingParams(n=test_params.beam_width,
                                     best_of=test_params.beam_width,
                                     use_beam_search=True)
    return LlmRequest(request_id=0,
                      seq_slot=test_params.seq_slot,
                      max_new_tokens=test_params.num_generated_tokens,
                      input_tokens=test_params.input_tokens,
                      end_id=test_params.end_id,
                      sampling_config=SamplingConfig(
                          sampling_params._get_sampling_config()),
                      return_log_probs=test_params.num_logprobs >= 0,
                      num_logprobs=test_params.num_logprobs,
                      is_streaming=False)


def create_default_sampler(test_params: GeneralTestParams) -> TorchSampler:
    sampler = TorchSampler(
        TorchSampler.Args(
            max_seq_len=test_params.max_seq_len,
            max_draft_len=0,
            max_num_sequences=test_params.max_batch_size,
            max_beam_width=test_params.max_beam_width,
            max_total_draft_tokens=0,
            disable_overlap_scheduler=True,
        ))
    max_beam_width = sampler.max_beam_width
    max_seq_len = sampler.max_seq_len
    max_batch_size = sampler.max_num_sequences

    # perform assertion tests for the selected parameter
    assert max_beam_width > test_params.beam_width, "Max beam width must be greater than beam width"
    assert max_seq_len > test_params.seq_len, "Max sequence length must be greater than sequence length"
    assert max_batch_size > test_params.batch_size, "Max batch size must be greater than batch size"
    assert max_batch_size > test_params.seq_slot, "Max batch size must be greater than sequence slot"
    assert sampler.store.cache_indirection.shape == (
        max_batch_size, max_beam_width,
        max_seq_len), "Cache indirection shape mismatch"
    assert sampler.store.original_tokens.shape == (
        max_batch_size, max_beam_width,
        max_seq_len), "Original tokens shape mismatch"
    return sampler


def test_create_beam_history():
    """Test TorchSampler._create_beam_history method.

    This test verifies that beam history is correctly reconstructed by following
    the cache_indirection backwards to obtain the correct token sequence.
    """
    test_params = GeneralTestParams()
    request = create_default_request(test_params)
    sampler = create_default_sampler(test_params)

    # Extract parameters from the test parameters
    beam_width = test_params.beam_width
    prompt_len = test_params.prompt_len
    num_generated_tokens = test_params.num_generated_tokens
    seq_slot = test_params.seq_slot
    vocab_size = test_params.vocab_size
    num_logprobs = test_params.num_logprobs + 1
    cache_indirection = sampler.store.cache_indirection
    original_tokens = sampler.store.original_tokens
    original_logprobs = torch.zeros(
        (beam_width, num_generated_tokens, num_logprobs),
        dtype=torch.float32,
        device=original_tokens.device)
    original_logprob_indices = torch.zeros(
        (beam_width, num_generated_tokens, num_logprobs),
        dtype=torch.int32,
        device=original_tokens.device)
    original_cum_logprobs = sampler.store.cum_log_probs

    # Fill the request with some random tokens that will be overwritten by the beam search sampling
    # Beam history is created before add_token is called
    request.set_generated_tokens(
        torch.randint(0,
                      vocab_size, (beam_width, num_generated_tokens - 1),
                      dtype=torch.int32).tolist())
    # random fill
    torch.manual_seed(42)
    original_tokens[seq_slot, :beam_width, prompt_len:prompt_len +
                    num_generated_tokens] = torch.randint(
                        0,
                        beam_width, (beam_width, num_generated_tokens),
                        dtype=torch.int32)
    assert original_tokens.sum(
    ) > 0, "Original tokens must not only contain zeros. Otherwise change the seed."

    original_logprobs[:beam_width] = torch.randn(
        (beam_width, num_generated_tokens, original_logprobs.shape[-1]),
        dtype=torch.float32)
    original_logprob_indices[:beam_width] = torch.randint(
        0,
        vocab_size,
        (beam_width, num_generated_tokens, original_logprobs.shape[-1]),
        dtype=torch.int32)
    assert (original_logprobs != 0).sum(
    ) > 0, "Original log probs must not only contain zeros. Otherwise change the seed."
    assert (original_logprob_indices).sum(
    ) > 0, "Original log prob indices must not only contain zeros. Otherwise change the seed."

    # set the logprobs in the request:
    token_logprobs = sampler._convert_logprobs_tensor_to_list(
        original_logprob_indices[:beam_width, :num_generated_tokens - 1],
        original_logprobs[:beam_width, :num_generated_tokens - 1],
        None,
        None,
        None,
    )
    request.py_result.set_log_probs(
        token_logprobs,
        cum_log_probs=torch.zeros_like(
            original_cum_logprobs[seq_slot, :beam_width]).tolist())

    original_cum_logprobs[seq_slot, :beam_width] = torch.randn(
        (beam_width, ), dtype=torch.float32)
    assert (original_cum_logprobs != 0).sum(
    ) > 0, "Original cumulative log probs must not only contain zeros. Otherwise change the seed."

    cache_indirection[seq_slot, :beam_width, prompt_len:prompt_len +
                      num_generated_tokens] = torch.randint(
                          0,
                          beam_width, (beam_width, num_generated_tokens),
                          dtype=torch.int32)
    assert cache_indirection[
        seq_slot, :beam_width,
        prompt_len:prompt_len + num_generated_tokens].sum(
        ) > 0, "Deterministic offsets must not only contain zeros. Otherwise change the seed."

    # set the new log probs and tokens for the beam search sampling
    sampler.store.sampled_log_probs[
        seq_slot, :beam_width] = original_logprobs[:beam_width,
                                                   num_generated_tokens - 1,
                                                   0:1]
    sampler.store.new_tokens[
        0,
        seq_slot, :beam_width] = original_logprob_indices[:beam_width,
                                                          num_generated_tokens -
                                                          1, 0]
    # test
    beam_history = sampler._create_beam_history(request)

    # expected selection:
    # Currently beam history only contains the generated tokens, not the prompt tokens.
    expected_tokens = torch.zeros(
        (sampler.max_beam_width, num_generated_tokens),
        dtype=torch.int32,
        device=original_tokens.device)
    expected_logprobs = torch.zeros(
        (beam_width, num_generated_tokens, original_logprobs.shape[-1]),
        dtype=torch.float32,
        device=original_logprobs.device)
    for gen_idx in range(num_generated_tokens):
        token_idx = prompt_len + gen_idx
        expected_tokens[:, gen_idx] = original_tokens[
            seq_slot, cache_indirection[seq_slot, :, token_idx], token_idx]
        expected_logprobs[:, gen_idx] = original_logprobs[cache_indirection[
            seq_slot, :beam_width, token_idx], gen_idx]

    torch.testing.assert_close(beam_history.tokens[:beam_width],
                               expected_tokens[:beam_width])
    # test logprobs as well
    torch.testing.assert_close(beam_history.logprobs[:beam_width],
                               expected_logprobs[:beam_width])
    torch.testing.assert_close(beam_history.cum_logprobs[:beam_width],
                               original_cum_logprobs[seq_slot, :beam_width])

    return


def test_finish_beams():
    """Test TorchSampler._finish_beams method.

    This test verifies that beams are correctly finalized.
    """

    test_params = GeneralTestParams()
    beam_width = test_params.beam_width
    num_generated_tokens = test_params.num_generated_tokens
    test_params.seq_len
    end_id = test_params.end_id
    batch_size = test_params.batch_size
    vocab_size = test_params.vocab_size
    num_logprobs = 1
    request = create_default_request(test_params)
    sampler = create_default_sampler(test_params)
    store_device = sampler.store.cache_indirection.device

    request.set_generated_tokens(
        torch.randint(0,
                      vocab_size, (beam_width, num_generated_tokens),
                      dtype=torch.int32).tolist())

    torch.manual_seed(42)
    # Do not keep end_id tokens in the tensor. This would interfere with the test.
    tokens = torch.randint(
        0,
        end_id, (batch_size, sampler.max_beam_width, num_generated_tokens),
        dtype=torch.int32,
        device=store_device)
    logprobs = torch.randn((batch_size, sampler.max_beam_width,
                            num_generated_tokens, num_logprobs),
                           dtype=torch.float32,
                           device=store_device)
    cum_logprobs = logprobs[..., 0].sum(dim=-1)

    # assert that the  buffers are different from zero. Otherwise the test may pass if the function does not work.
    assert tokens.sum(
    ) > 0, "Tokens must not only contain zeros. Otherwise change the seed."
    assert torch.any(logprobs != 0) and torch.any(
        cum_logprobs != 0
    ), "Log probs and cumulative log probs must not only contain zeros. Otherwise change the seed."

    tokens[batch_size - 1, 0, num_generated_tokens //
           2:] = BEAM_SEARCH_PAD_TOKEN  # simulate early finished beam

    for batch_idx in range(batch_size):
        beam_history = BeamHistory(
            tokens=tokens[batch_idx, :beam_width],
            logprobs=logprobs[batch_idx, :beam_width],
            cum_logprobs=cum_logprobs[batch_idx, :beam_width])
        request.py_return_log_probs = False
        prompt_len = request.py_prompt_len

        if batch_idx < batch_size - 1:
            # requests are not finished yet
            sampler._finalize_beam(request, beam_history)
            final_tokens = torch.tensor(request.get_tokens(),
                                        device=store_device,
                                        dtype=torch.int32)[:, prompt_len:]
            torch.testing.assert_close(final_tokens,
                                       tokens[batch_idx, :beam_width])
        # Test the case where end_ids are present in the output
        else:
            sampler._finalize_beam(request, beam_history)

            # Given input for beam 0: [ token, token, ..., token, BEAM_SEARCH_PAD_TOKEN, BEAM_SEARCH_PAD_TOKEN, ..., BEAM_SEARCH_PAD_TOKEN]
            # Expected output for beam 0: [ token, token, ..., token]
            final_tokens_1p = torch.tensor(request.get_tokens()[1:],
                                           device=store_device,
                                           dtype=torch.int32)[:, prompt_len:]
            final_tokens_0 = torch.tensor(request.get_tokens()[0],
                                          device=store_device,
                                          dtype=torch.int32)[prompt_len:]
            torch.testing.assert_close(final_tokens_1p, tokens[batch_idx,
                                                               1:beam_width])
            torch.testing.assert_close(final_tokens_0.shape[0],
                                       num_generated_tokens // 2)
            torch.testing.assert_close(
                final_tokens_0, tokens[batch_idx,
                                       0, :num_generated_tokens // 2])


@force_ampere  # Save H100 resource
class TestParameterValidation:
    """Ensure that unsupported request parameters do not crash/hang the engine."""

    @pytest.fixture(scope="module")
    @staticmethod
    def fixed_params():
        return {"max_tokens": 8, "max_beam_width": 4}

    @pytest.fixture(scope="module")
    @staticmethod
    def model_kwargs() -> dict[str, Any]:
        root = llm_models_root()
        assert root is not None
        return dict(model=root / "llama-models-v2" /
                    "TinyLlama-1.1B-Chat-v1.0", )

    # NB: Class-level fixture overrides do not work without this
    @pytest.fixture(scope="module")
    @staticmethod
    def llm(fixed_params, input_prompts, model_kwargs):
        return _build_llm(fixed_params, input_prompts, model_kwargs)

    def _check_engine_responds(self, llm: LLM, input_prompts: list[str],
                               fixed_params: dict):
        _ = llm.generate(input_prompts,
                         sampling_params=SamplingParams(
                             max_tokens=fixed_params["max_tokens"],
                             n=1,
                             best_of=fixed_params["max_beam_width"],
                             use_beam_search=True,
                             end_id=-1,
                         ))

    @pytest.mark.timeout(120)
    @pytest.mark.threadleak(enabled=False)
    def test_use_beam_search_false(
        self,
        llm: LLM,
        input_prompts: list[str],
        fixed_params: dict,
    ):
        assert fixed_params["max_beam_width"] > 2
        with pytest.raises(
                ValueError,
                match=
                ".*Greedy decoding in the LLM API does not allow multiple returns.*"
        ):
            _ = llm.generate(input_prompts,
                             sampling_params=SamplingParams(
                                 max_tokens=fixed_params["max_tokens"],
                                 n=1,
                                 best_of=fixed_params["max_beam_width"],
                                 use_beam_search=False,
                                 end_id=-1,
                             ))
        self._check_engine_responds(llm, input_prompts, fixed_params)

    @pytest.mark.timeout(120)
    @pytest.mark.threadleak(enabled=False)
    def test_use_beam_search_ommitted(
        self,
        llm: LLM,
        input_prompts: list[str],
        fixed_params: dict,
    ):
        assert fixed_params["max_beam_width"] > 2
        with pytest.raises(
                ValueError,
                match=
                ".*Greedy decoding in the LLM API does not allow multiple returns.*"
        ):
            _ = llm.generate(input_prompts,
                             sampling_params=SamplingParams(
                                 max_tokens=fixed_params["max_tokens"],
                                 n=1,
                                 best_of=fixed_params["max_beam_width"],
                                 end_id=-1,
                             ))
        self._check_engine_responds(llm, input_prompts, fixed_params)

    @pytest.mark.timeout(120)
    @pytest.mark.threadleak(enabled=False)
    def test_smaller_beam_width(
        self,
        llm: LLM,
        input_prompts: list[str],
        fixed_params: dict,
    ):
        assert fixed_params["max_beam_width"] > 2
        with pytest.raises(
                RequestError,
                match=".*Request beam width 2 is not equal to max_beam_width 4*"
        ):
            _ = llm.generate(input_prompts,
                             sampling_params=SamplingParams(
                                 max_tokens=fixed_params["max_tokens"],
                                 n=1,
                                 best_of=2,
                                 use_beam_search=True,
                                 end_id=-1,
                             ))
        self._check_engine_responds(llm, input_prompts, fixed_params)


if __name__ == "__main__":
    pytest.main([__file__])
