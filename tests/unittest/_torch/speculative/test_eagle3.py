import os
import sys
import unittest

import pytest
import torch

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.llmapi import EagleDecodingConfig, KvCacheConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../integration'))
from defs.common import similar

# We have the reference answers hard coded here for the following reasons:
# 1. Note having run a reference LLM makes this test faster
# 2. For the relaxed decoding, we can't really run a reference model. By definition,
# the text you get from the model with relaxed decoding on will be different.
# To avoid making this test brittle, we test a similarity score with a high threshold.
# This gives us some wiggle room in case the kernels change, for example.
_PROMPTS = ["The capital of France is", "The president of the United States is"]

_TEXT_REF_EXACT = [
    " a city of romance, art, fashion, and cuisine. Paris is a must-visit destination for anyone who loves history, architecture, and culture. From the",
    " the head of state and head of government of the United States. The president serves a four-year term and is limited to two terms. The president is elected through",
]

_TEXT_REF_RELAXED = [
    " a not-to-be-missed destination for any traveler. Paris, the City of love, art, fashion, and cuisine, is a must-visit destination",
    " the head of state and head of the United States. The president serves a four-year term and is elected through the Electoral College system. The president is responsible for",
]


@pytest.mark.parametrize("use_cuda_graph", [True, False],
                         ids=["enable_graphs", "disable_graphs"])
@pytest.mark.parametrize("use_relaxed_decoding", [True, False],
                         ids=["use_relaxed_decoding", "use_greedy_decoding"])
def test_llama_eagle3(use_cuda_graph: bool, use_relaxed_decoding: bool):
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 35:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()

    pytorch_config = PyTorchConfig(
        enable_overlap_scheduler=False,
        use_cuda_graph=use_cuda_graph,
        # Only create a single CUDA graph to prevent OOM in CI
        cuda_graph_batch_sizes=[1],
    )

    kv_cache_config = KvCacheConfig(enable_block_reuse=False, )

    eagle_model_dir = f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B"
    target_model_dir = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"

    draft_len = 4
    spec_config = EagleDecodingConfig(
        max_draft_len=draft_len,
        pytorch_eagle_weights_path=eagle_model_dir,
        greedy_sampling=not use_relaxed_decoding)

    llm_spec = LLM(model=target_model_dir,
                   pytorch_backend_config=pytorch_config,
                   kv_cache_config=kv_cache_config,
                   speculative_config=spec_config)

    sampling_params = SamplingParams(
        max_tokens=32,
        temperature=0,
    )

    # First make sure the acceptance rate is reasonable.
    tok_ids = llm_spec.tokenizer.encode("The future of AI is")
    num_tokens = 0

    num_drafted = 0
    num_accepted = 0

    for output in llm_spec.generate_async(tok_ids,
                                          SamplingParams(max_tokens=128,
                                                         temperature=0),
                                          streaming=True):
        beam = output.outputs[0]
        new_tokens = beam.token_ids

        num_drafted += draft_len
        num_accepted += len(new_tokens) - num_tokens - 1

        num_tokens = len(new_tokens)

    accept_rate = num_accepted / num_drafted
    assert accept_rate > 0.20

    results_spec = llm_spec.generate(_PROMPTS, sampling_params)
    generated_text_spec = [result.outputs[0].text for result in results_spec]
    llm_spec.shutdown()

    generated_text_ref = _TEXT_REF_EXACT if not use_relaxed_decoding else _TEXT_REF_RELAXED

    for text_spec in generated_text_spec:
        assert any(
            similar(text_spec, text_ref, threshold=0.9)
            for text_ref in generated_text_ref)


if __name__ == "__main__":
    unittest.main()
