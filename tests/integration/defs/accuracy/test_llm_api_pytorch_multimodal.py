import pytest

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig, SamplingParams

from ..conftest import llm_models_root, skip_post_blackwell, skip_pre_blackwell, skip_pre_hopper
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
    MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"
    MODEL_PATH = f"{llm_models_root()}/llava-v1.6-mistral-7b-hf"
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


class TestNemotron_Nano_12B_V2_VL(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
    MODEL_PATH = f"{llm_models_root()}/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
    MAX_NUM_TOKENS = 25600
    EXTRA_EVALUATOR_KWARGS = dict(
        apply_chat_template=True,
        system_prompt="/no_think",
    )

    # NOTE: MMMU adds <|endoftext|> to the stop token.
    sampling_params = SamplingParams(
        max_tokens=MAX_NUM_TOKENS,
        truncate_prompt_tokens=MMMU.MAX_INPUT_LEN,
        temperature=0.0,
        top_k=1,
        stop="<|endoftext|>",
    )

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8, enable_block_reuse=False)

    def test_auto_dtype(self):
        with LLM(
            self.MODEL_PATH,
            max_batch_size=128,
            max_num_tokens=self.MAX_NUM_TOKENS,
            kv_cache_config=self.kv_cache_config,
        ) as llm:
            task = MMMU(self.MODEL_NAME)
            task.evaluate(
                llm,
                sampling_params=self.sampling_params,
                extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS,
            )


class TestPhi4MMFusedVisionLora(LlmapiAccuracyTestHarness):
    MODEL_NAME = "microsoft/Phi-4-multimodal-instruct"
    MODEL_PATH = f"{llm_models_root()}/multimodals/Phi-4-multimodal-instruct-fuse-vision-lora"
    MAX_NUM_TOKENS = 25600

    sampling_params = SamplingParams(
        max_tokens=MAX_NUM_TOKENS, truncate_prompt_tokens=MMMU.MAX_INPUT_LEN, stop="<|USER|>"
    )

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.7)

    def test_auto_dtype(self):
        with LLM(
            self.MODEL_PATH,
            max_batch_size=32,
            max_num_tokens=self.MAX_NUM_TOKENS,
            kv_cache_config=self.kv_cache_config,
        ) as llm:
            task = MMMU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=self.sampling_params)


@skip_pre_hopper
@skip_post_blackwell
class TestGemma3_27BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "google/gemma-3-27b-it"
    # Note: This has only the LLM part quantized. Vision part is in bfloat16.
    MODEL_PATH = f"{llm_models_root()}/gemma/gemma-3-27b-it-fp8/"
    MAX_NUM_TOKENS = 12800

    sampling_params = SamplingParams(
        max_tokens=MAX_NUM_TOKENS, truncate_prompt_tokens=MMMU.MAX_INPUT_LEN, stop="<end_of_turn>"
    )

    # Gemma3 VLM needs KV cache reuse disabled for custom mask support.
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,
        enable_partial_reuse=False,
        free_gpu_memory_fraction=0.4,
        dtype="fp8",
    )

    def test_fp8_prequantized(self):
        # Gemma3 VLM needs FlashInfer attention backend for custom mask support.
        with LLM(
            self.MODEL_PATH,
            max_batch_size=16,
            max_num_tokens=self.MAX_NUM_TOKENS,
            max_seq_len=8704,  # 8192 + 512.
            kv_cache_config=self.kv_cache_config,
            attn_backend="FLASHINFER",
            enable_chunked_prefill=False,
        ) as llm:
            task = MMMU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=self.sampling_params)


class TestQwen3VL_MOE(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/Qwen3/Qwen3-VL-30B-A3B-Instruct"
    MAX_NUM_TOKENS = 16384

    sampling_params = SamplingParams(
        max_tokens=MAX_NUM_TOKENS, truncate_prompt_tokens=MMMU.MAX_INPUT_LEN, stop="<|endoftext|>"
    )

    @pytest.mark.skip_less_device_memory(140000)
    def test_auto_dtype(self):
        with LLM(
            self.MODEL_PATH,
            max_num_tokens=self.MAX_NUM_TOKENS,
        ) as llm:
            task = MMMU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=self.sampling_params)


class TestMistralLarge3_675B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "mistral/Mistral-Large-3-675B"
    MODEL_PATH = (
        f"{llm_models_root()}/Mistral-Large-3-675B/Mistral-Large-3-675B-Instruct-2512-NVFP4/"
    )
    MAX_NUM_TOKENS = 16384

    sampling_params = SamplingParams(
        max_tokens=MAX_NUM_TOKENS, truncate_prompt_tokens=MMMU.MAX_INPUT_LEN, stop="<|endoftext|>"
    )

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(4)
    @pytest.mark.skip_less_device_memory(183000)
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler,moe_backend",
        [
            (4, 1, 4, False, True, True, "TRTLLM"),
        ],
        ids=[
            "latency_moe_trtllm",
        ],
    )
    def test_nvfp4_4gpus(
        self,
        tp_size,
        pp_size,
        ep_size,
        attention_dp,
        cuda_graph,
        overlap_scheduler,
        moe_backend,
        mocker,
    ):
        mocker.patch.dict(
            MMMU.EVALUATE_KWARGS, {"model_type": "mistral_large_3", "is_force_single_image": True}
        )
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            moe_config=MoeConfig(backend=moe_backend),
        )

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)

        with LLM(
            self.MODEL_PATH,
            max_num_tokens=self.MAX_NUM_TOKENS,
            checkpoint_format="mistral",
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            moe_expert_parallel_size=ep_size,
            **pytorch_config,
            enable_attention_dp=attention_dp,
            kv_cache_config=kv_cache_config,
        ) as llm:
            task = MMMU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=self.sampling_params)


class TestQwen3VL(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/Qwen3/Qwen3-VL-8B-Instruct"
    MAX_NUM_TOKENS = 16384

    sampling_params = SamplingParams(
        max_tokens=MAX_NUM_TOKENS, truncate_prompt_tokens=MMMU.MAX_INPUT_LEN, stop="<|endoftext|>"
    )

    def test_auto_dtype(self):
        with LLM(
            self.MODEL_PATH,
            max_num_tokens=self.MAX_NUM_TOKENS,
        ) as llm:
            task = MMMU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=self.sampling_params)


class TestMistralSmall24B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    MODEL_PATH = f"{llm_models_root()}/Mistral-Small-3.1-24B-Instruct-2503"
    MAX_NUM_TOKENS = 16384

    # NOTE: MMMU adds <|endoftext|> to the stop token.
    sampling_params = SamplingParams(
        max_tokens=MMMU.MAX_OUTPUT_LEN,
        truncate_prompt_tokens=MMMU.MAX_INPUT_LEN,
        stop="<|endoftext|>",
    )

    @pytest.mark.skip_less_device_memory(80000)
    def test_auto_dtype(self):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        with LLM(
            self.MODEL_PATH,
            kv_cache_config=kv_cache_config,
            enable_chunked_prefill=True,
            max_num_tokens=self.MAX_NUM_TOKENS,
        ) as llm:
            task = MMMU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=self.sampling_params)
