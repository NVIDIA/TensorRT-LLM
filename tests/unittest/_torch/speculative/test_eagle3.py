import os
import sys
import unittest

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.llmapi import EagleDecodingConfig, KvCacheConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root


def test_llama_eagle3():
    models_path = llm_models_root()

    pytorch_config = PyTorchConfig(
        enable_overlap_scheduler=False,
        use_cuda_graph=False,
    )

    kv_cache_config = KvCacheConfig(enable_block_reuse=False, )

    eagle_model_dir = f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B"
    target_model_dir = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"

    draft_len = 4
    spec_config = EagleDecodingConfig(
        max_draft_len=draft_len, pytorch_eagle_weights_path=eagle_model_dir)

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
    assert accept_rate > 0.25

    prompts = [
        "The capital of France is", "The president of the United States is"
    ]
    results_spec = llm_spec.generate(prompts, sampling_params)
    generated_text_spec = [result.outputs[0].text for result in results_spec]
    llm_spec.shutdown()

    llm_ref = LLM(model=target_model_dir,
                  pytorch_backend_config=pytorch_config,
                  kv_cache_config=kv_cache_config)

    results_ref = llm_ref.generate(prompts, sampling_params)
    generated_text_ref = [result.outputs[0].text for result in results_ref]
    llm_ref.shutdown()

    for text_spec, text_ref in zip(generated_text_spec, generated_text_ref):
        # The spec decode algorithm currently guarantees identical results
        assert text_spec == text_ref


if __name__ == "__main__":
    unittest.main()
