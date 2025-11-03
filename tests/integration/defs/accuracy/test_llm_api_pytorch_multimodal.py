import pytest

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams

from ..conftest import llm_models_root
from .accuracy_core import MMMU, LlmapiAccuracyTestHarness


class TestQwen2_VL_7B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/Qwen2-VL-7B-Instruct"
    MAX_NUM_TOKENS = 16384
    # NOTE: MMMU adds <|endoftext|> to the stop token.
    sampling_params = SamplingParams(
        max_tokens=MMMU.MAX_OUTPUT_LEN,
        truncate_prompt_tokens=MMMU.MAX_INPUT_LEN,
        stop="<|endoftext|>",
    )

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6)

    @pytest.mark.skip(reason="https://nvbugs/5601909")
    def test_auto_dtype(self):
        with LLM(
            self.MODEL_PATH,
            max_num_tokens=self.MAX_NUM_TOKENS,
            kv_cache_config=self.kv_cache_config,
        ) as llm:
            task = MMMU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=self.sampling_params)


class TestQwen2_5_VL_7B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/Qwen2.5-VL-7B-Instruct"
    MAX_NUM_TOKENS = 16384

    # NOTE: MMMU adds <|endoftext|> to the stop token.
    sampling_params = SamplingParams(
        max_tokens=MMMU.MAX_OUTPUT_LEN,
        truncate_prompt_tokens=MMMU.MAX_INPUT_LEN,
        stop="<|endoftext|>",
    )

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6)

    def test_auto_dtype(self):
        with LLM(
            self.MODEL_PATH,
            max_num_tokens=self.MAX_NUM_TOKENS,
            kv_cache_config=self.kv_cache_config,
        ) as llm:
            task = MMMU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=self.sampling_params)


class TestNano_V2_VLM(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/Nano-v2-VLM"
    MODEL_PATH = f"{llm_models_root()}/Nano-v2-VLM"
    MAX_NUM_TOKENS = 25600

    # NOTE: MMMU adds <|endoftext|> to the stop token.
    sampling_params = SamplingParams(
        max_tokens=MAX_NUM_TOKENS, truncate_prompt_tokens=MMMU.MAX_INPUT_LEN, stop="<|endoftext|>"
    )

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8, enable_block_reuse=False)

    @pytest.mark.skip(reason="Nano V2 VLM ckpt is not released yet.")
    def test_auto_dtype(self):
        with LLM(
            self.MODEL_PATH,
            max_batch_size=128,
            max_num_tokens=self.MAX_NUM_TOKENS,
            kv_cache_config=self.kv_cache_config,
        ) as llm:
            task = MMMU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=self.sampling_params)


class TestLlava_V1_6_Mistral_7B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b"
    MODEL_PATH = f"{llm_models_root()}/llava-v1.6-mistral-7b"
    MAX_NUM_TOKENS = 16384

    # NOTE: MMMU adds <|endoftext|> to the stop token.
    sampling_params = SamplingParams(
        max_tokens=MMMU.MAX_OUTPUT_LEN,
        truncate_prompt_tokens=MMMU.MAX_INPUT_LEN,
        stop="<|endoftext|>",
    )

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6)

    def test_auto_dtype(self):
        with LLM(
            self.MODEL_PATH,
            max_num_tokens=self.MAX_NUM_TOKENS,
            kv_cache_config=self.kv_cache_config,
        ) as llm:
            task = MMMU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=self.sampling_params)


class TestNVILA_8B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Efficient-Large-Model/NVILA-8B"
    MODEL_PATH = f"{llm_models_root()}/vila/NVILA-8B"
    MAX_NUM_TOKENS = 16384

    # NOTE: MMMU adds <|endoftext|> to the stop token.
    sampling_params = SamplingParams(
        max_tokens=MMMU.MAX_OUTPUT_LEN,
        truncate_prompt_tokens=MMMU.MAX_INPUT_LEN,
        stop="<|endoftext|>",
    )

    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.6,
        # NOTE: VILA models do not support block reuse.
        enable_block_reuse=False,
    )

    def test_auto_dtype(self):
        with LLM(
            self.MODEL_PATH,
            max_num_tokens=self.MAX_NUM_TOKENS,
            kv_cache_config=self.kv_cache_config,
        ) as llm:
            task = MMMU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=self.sampling_params)


class TestVILA1_5_3B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Efficient-Large-Model/VILA1.5-3b"
    MODEL_PATH = f"{llm_models_root()}/vila/VILA1.5-3b"
    MAX_NUM_TOKENS = 16384

    # NOTE: MMMU adds <|endoftext|> to the stop token.
    sampling_params = SamplingParams(
        max_tokens=MMMU.MAX_OUTPUT_LEN,
        truncate_prompt_tokens=MMMU.MAX_INPUT_LEN,
        stop="<|endoftext|>",
    )

    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.6,
        # NOTE: VILA models do not support block reuse.
        enable_block_reuse=False,
    )

    def test_auto_dtype(self):
        with LLM(
            self.MODEL_PATH,
            max_num_tokens=self.MAX_NUM_TOKENS,
            kv_cache_config=self.kv_cache_config,
        ) as llm:
            task = MMMU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=self.sampling_params)
