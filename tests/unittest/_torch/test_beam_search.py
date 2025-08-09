import os

import pytest
from utils.llm_data import llm_models_root
from utils.util import force_ampere, similar

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig


@pytest.fixture(scope="module")
def input_prompts():
    return [
        "Born in north-east France, Soyer trained as a",
        "The future of AI is",
    ]


@pytest.fixture(scope="module")
def expected_outputs():
    return {
        "Born in north-east France, Soyer trained as a": [
            "painter in Paris before moving to London in",
            "painter and sculptor in Paris before moving"
        ],
        "The future of AI is":
        ["bright, but it's not without", "bright, but it's not going"],
    }


@pytest.fixture(scope="module")
def fixed_params():
    return {"max_tokens": 8, "max_beam_width": 2}


@pytest.fixture(scope="module")
def llm(fixed_params, input_prompts):
    return LLM(
        model=os.path.join(llm_models_root(), "llama-models-v2",
                           "TinyLlama-1.1B-Chat-v1.0"),
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
def llm_cuda_graph(fixed_params, input_prompts):
    return LLM(
        model=os.path.join(llm_models_root(), "llama-models-v2",
                           "TinyLlama-1.1B-Chat-v1.0"),
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


@force_ampere  # Save H100 resource
@pytest.mark.parametrize("return_log_probs", [True, False])
@pytest.mark.parametrize("gather_generation_logits", [True, False])
@pytest.mark.parametrize("gather_context_logits", [True, False])
@pytest.mark.parametrize("num_output_beams", [1, 2])
@pytest.mark.parametrize("num_prompts", [1, 2])
@pytest.mark.threadleak(enabled=False)
def test_beam_search_output_shapes(gather_context_logits: bool,
                                   gather_generation_logits: bool,
                                   return_log_probs: bool,
                                   num_output_beams: int, num_prompts: int, llm,
                                   fixed_params, input_prompts,
                                   expected_outputs):
    if return_log_probs and num_prompts > 1:
        pytest.skip(
            "Beam search currently does not support return_log_probs with multiple prompts"
        )
    sampling_params = SamplingParams(
        max_tokens=fixed_params["max_tokens"],
        n=num_output_beams,
        best_of=fixed_params["max_beam_width"],
        use_beam_search=True,
        return_context_logits=gather_context_logits,
        return_generation_logits=gather_generation_logits,
        logprobs=return_log_probs,
    )
    outputs = llm.generate(input_prompts[:num_prompts],
                           sampling_params=sampling_params)
    assert len(outputs) == num_prompts
    for output_idx, output in enumerate(outputs):
        if gather_context_logits:
            assert output.context_logits is not None
            assert len(
                output.prompt_token_ids) == output.context_logits.shape[0]
        else:
            assert output.context_logits is None
        assert len(output.outputs) == num_output_beams
        for beam_idx, beam in enumerate(output.outputs):
            if gather_generation_logits:
                gen_logits = beam.generation_logits
                assert gen_logits is not None
                assert gen_logits.ndim == 2
                assert gen_logits.shape[0] == sampling_params.max_tokens
            else:
                assert beam.generation_logits is None

            if return_log_probs:
                assert len(beam.logprobs) == sampling_params.max_tokens
            else:
                assert len(beam.logprobs) == 0
            # Check output similarity
            assert similar(
                beam.text,
                expected_outputs[input_prompts[output_idx]][beam_idx])


@force_ampere  # Save H100 resource
@pytest.mark.parametrize("return_log_probs", [True, False])
@pytest.mark.parametrize("gather_generation_logits", [True, False])
@pytest.mark.parametrize("gather_context_logits", [True, False])
@pytest.mark.parametrize("num_output_beams", [1, 2])
@pytest.mark.parametrize("num_prompts", [1, 2, 3])
@pytest.mark.threadleak(enabled=False)
def test_beam_search_output_shapes_cuda_graph_and_overlap(
        gather_context_logits: bool, gather_generation_logits: bool,
        return_log_probs: bool, num_output_beams: int, num_prompts: int,
        llm_cuda_graph, fixed_params, input_prompts, expected_outputs):
    if return_log_probs and num_prompts > 1:
        pytest.skip(
            "Beam search currently does not support return_log_probs with multiple prompts"
        )
    sampling_params = SamplingParams(
        max_tokens=fixed_params["max_tokens"],
        n=num_output_beams,
        best_of=fixed_params["max_beam_width"],
        use_beam_search=True,
        return_context_logits=gather_context_logits,
        return_generation_logits=gather_generation_logits,
        logprobs=return_log_probs,
    )
    # test padding of cuda graph with 3 prompts
    # replicate the prompts to have more than 2 prompts available
    if (num_prompts == 3 and len(input_prompts) == 2):
        input_prompts = [input_prompts[0]] * 3
    outputs = llm_cuda_graph.generate(input_prompts[:num_prompts],
                                      sampling_params=sampling_params)
    assert len(outputs) == num_prompts
    for output_idx, output in enumerate(outputs):
        if gather_context_logits:
            assert output.context_logits is not None
            assert len(
                output.prompt_token_ids) == output.context_logits.shape[0]
        else:
            assert output.context_logits is None
        assert len(output.outputs) == num_output_beams
        for beam_idx, beam in enumerate(output.outputs):
            if gather_generation_logits:
                gen_logits = beam.generation_logits
                assert gen_logits is not None
                assert gen_logits.ndim == 2
                assert gen_logits.shape[0] == sampling_params.max_tokens
            else:
                assert beam.generation_logits is None

            if return_log_probs:
                assert len(beam.logprobs) == sampling_params.max_tokens
            else:
                assert len(beam.logprobs) == 0
            # Check output similarity
            assert similar(
                beam.text,
                expected_outputs[input_prompts[output_idx]][beam_idx])
