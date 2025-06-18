import os

import pytest
from utils.llm_data import llm_models_root
from utils.util import force_ampere, similar

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm.llmapi.llm_utils import KvCacheConfig

prompts = [
    "Born in north-east France, Soyer trained as a",
    "The future of AI is",
]
expected_outputs = {
    "Born in north-east France, Soyer trained as a": [
        "painter in Paris before moving to London in",
        "painter and sculptor in Paris before moving"
    ],
    "The future of AI is":
    ["bright, but it's not without", "bright, but it's not going"],
}

global_kvcache_config = KvCacheConfig(max_tokens=10000)


@force_ampere  # Save H100 resource
@pytest.mark.parametrize("return_log_probs", [True, False])
@pytest.mark.parametrize("gather_generation_logits", [True, False])
@pytest.mark.parametrize("gather_context_logits", [True, False])
@pytest.mark.parametrize("enable_trtllm_sampler", [True])
@pytest.mark.parametrize("disable_overlap_scheduler", [True, False])
@pytest.mark.parametrize("max_beam_width", [2])
@pytest.mark.parametrize("max_tokens", [8])
@pytest.mark.parametrize("num_prompts", [1, 2])
def test_beam_search_output_shapes(disable_overlap_scheduler: bool,
                                   enable_trtllm_sampler: bool,
                                   gather_context_logits: bool,
                                   gather_generation_logits: bool,
                                   return_log_probs: bool, max_beam_width: int,
                                   max_tokens: int, num_prompts: int):
    if not enable_trtllm_sampler:
        pytest.skip(
            "Beam search currently is only supported with TRTLLMSampler")
    if return_log_probs and num_prompts > 1:
        pytest.skip(
            "Beam search currently does not support return_log_probs with multiple prompts"
        )
    llm = LLM(
        model=os.path.join(llm_models_root(), "llama-models-v2",
                           "TinyLlama-1.1B-Chat-v1.0"),
        kv_cache_config=global_kvcache_config,
        gather_generation_logits=gather_generation_logits,
        max_batch_size=
        128,  # reduce buffer sizes, specially for generation logits
        max_seq_len=128,
        enable_trtllm_sampler=enable_trtllm_sampler,
        disable_overlap_scheduler=disable_overlap_scheduler,
        max_beam_width=max_beam_width,
    )
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        top_k=None,
        top_p=None,
        n=max_beam_width,
        use_beam_search=max_beam_width > 1,
        return_context_logits=gather_context_logits,
        return_generation_logits=gather_generation_logits,
        logprobs=return_log_probs,
    )
    with llm:
        for output_idx, output in enumerate(
                llm.generate(prompts[:num_prompts],
                             sampling_params=sampling_params)):
            if gather_context_logits:
                assert output.context_logits is not None
                assert len(
                    output.prompt_token_ids) == output.context_logits.shape[0]
            else:
                assert output.context_logits is None
            assert len(output.outputs) == max_beam_width
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
                assert similar(beam.text,
                               expected_outputs[prompts[output_idx]][beam_idx])
