import unittest

from parameterized import parameterized

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.sampling_params import SamplingParams

# isort: off
from utils.util import unittest_name_func, similar
from utils.llm_data import llm_models_root
# isort: on


class TestOutOfTree(unittest.TestCase):

    @parameterized.expand([False, True], name_func=unittest_name_func)
    def test_llm_api(self, import_oot_code: bool):
        if import_oot_code:
            # Import out-of-tree modeling code for OPTForCausalLM
            import os
            import sys
            sys.path.append(
                os.path.join(
                    os.path.dirname(__file__),
                    '../../../../examples/llm-api/out_of_tree_example'))
            import modeling_opt  # noqa

        model_dir = str(llm_models_root() / "opt-125m")
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)

        if not import_oot_code:
            with self.assertRaises(RuntimeError):
                # estimate_max_kv_cache_tokens will create a request of max_num_tokens for forward.
                # Default 8192 will exceed the max length of absolute positional embedding in OPT, leading to out of range indexing.
                llm = LLM(model=model_dir,
                          kv_cache_config=kv_cache_config,
                          max_num_tokens=2048)
            return

        llm = LLM(model=model_dir,
                  kv_cache_config=kv_cache_config,
                  max_num_tokens=2048,
                  disable_overlap_scheduler=True)

        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]

        references = [
            " J.C. and I am a student at",
            " not a racist. He is a racist.\n",
            " the capital of the French Republic.\n\nThe",
            " in the hands of the people.\n\nThe",
        ]

        sampling_params = SamplingParams(max_tokens=10)
        with llm:
            outputs = llm.generate(prompts, sampling_params=sampling_params)

        for output, ref in zip(outputs, references):
            assert similar(output.outputs[0].text, ref)
