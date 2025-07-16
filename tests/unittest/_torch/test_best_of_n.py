import os

import pytest
from utils.llm_data import llm_models_root
from utils.util import force_ampere, similar

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi.llm_utils import KvCacheConfig

prompts = [
    "Born in north-east France, Soyer trained as a",
    "The future of AI is",
]
expected_outputs = {
    "Born in north-east France, Soyer trained as a": [
        "lawyer and was a member of the French Resistance",
        "cook before turning to painting."
    ],
    "The future of AI is": [
        "all about human-machine collaboration.",
        "more promising than you think."
    ],
}

global_kvcache_config = KvCacheConfig(max_tokens=10000)


@force_ampere  # Save H100 resource
@pytest.mark.parametrize("n", [2])
@pytest.mark.parametrize("best_of", [None, 3])
def test_n_outputs(n: int, best_of: int):
    llm = LLM(model=os.path.join(llm_models_root(), "llama-models-v2",
                                 "TinyLlama-1.1B-Chat-v1.0"),
              kv_cache_config=global_kvcache_config,
              max_batch_size=128,
              max_seq_len=128,
              enable_trtllm_sampler=True)
    sampling_params = SamplingParams(
        n=n,
        best_of=best_of,
        temperature=0.8,  # ensure different outputs
        top_p=0.95,  # ensure different outputs
        use_beam_search=False)
    with llm:
        for output_idx, output in enumerate(
                llm.generate(prompts, sampling_params=sampling_params)):

            assert len(output.outputs) == n

            for idx, sequence in enumerate(output.outputs):
                if n == best_of:
                    assert similar(sequence.text,
                                   expected_outputs[prompts[output_idx]][idx])
