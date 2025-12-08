import os
import sys
import time

import pytest
from test_base_worker import create_fake_executor_config

from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer
from tensorrt_llm.llmapi.utils import logger_debug
from tensorrt_llm.sampling_params import SamplingParams

# isort: off
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from utils.llm_data import llm_models_root
from utils.util import similar, skip_single_gpu
# isort: on

model_path = llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


class TestRpcProxy:

    def create_proxy(self, tp_size: int):
        # Create executor config with the correct tp_size
        llm_args, executor_config = create_fake_executor_config(model_path,
                                                                tp_size=tp_size)

        # Enable KV cache events
        llm_args.kv_cache_config = KvCacheConfig(
            event_buffer_max_size=1000,  # Enable event buffer
            enable_block_reuse=True,  # Required for KV cache events
            free_gpu_memory_fraction=0.6,
        )

        proxy = GenerationExecutorRpcProxy(
            worker_kwargs={
                "engine": model_path,
                "executor_config": None,
                "llm_args": llm_args,
                "model_world_size": tp_size,
                "hf_model_dir": model_path,
            },
            model_world_size=tp_size,
            is_llm_executor=True,  # Enable stats collection
        )

        # Add additional wait for PyTorch backend with multi-rank setup
        if tp_size > 1:
            print(f"[Test] Waiting for {tp_size} ranks to initialize...")
            time.sleep(
                5)  # Give more time for multi-rank PyTorch initialization

        return proxy

    @pytest.mark.parametrize("num_reqs", [1, 5, 10])
    def test_tp1(self, num_reqs):
        tokenizer = TransformersTokenizer.from_pretrained(model_path)
        prompt = "A B C D"
        prompt_token_ids = tokenizer.encode(prompt)
        max_tokens = 8

        with self.create_proxy(tp_size=1) as proxy:
            logger_debug(f"[Test] Proxy created", color="green")
            sampling_params = SamplingParams(max_tokens=max_tokens)
            for _ in range(num_reqs):
                logger_debug(f"[Test] Generating {_}th", color="green")
                result = proxy.generate(prompt_token_ids, sampling_params)
                assert similar(tokenizer.decode(result.outputs[0].token_ids),
                               'E F G H I J K L')
                logger_debug(f"req {_} get result: {result}", color="green")

            #stats = proxy.get_stats(timeout=2)
            #assert stats

            #kv_cache_events = proxy.get_kv_events(timeout=2)
            # KV cache events may be empty if no cache operations occurred
            #assert isinstance(kv_cache_events, list)

    @pytest.mark.parametrize("num_reqs", [1, 10])
    @skip_single_gpu
    @pytest.mark.gpu2
    def test_tp2(self, num_reqs):
        tokenizer = TransformersTokenizer.from_pretrained(model_path)
        prompt = "A B C D"
        prompt_token_ids = tokenizer.encode(prompt)
        max_tokens = 8

        with self.create_proxy(tp_size=2) as proxy:
            sampling_params = SamplingParams(max_tokens=max_tokens)
            for _ in range(num_reqs):
                result = proxy.generate(prompt_token_ids, sampling_params)
                print(f"get result: {result}")
                assert similar(tokenizer.decode(result.outputs[0].token_ids),
                               'E F G H I J K L')

    def test_hmac_key_generation(self):
        """Test that HMAC key is automatically generated and properly propagated."""
        tokenizer = TransformersTokenizer.from_pretrained(model_path)
        prompt = "A B C D"
        prompt_token_ids = tokenizer.encode(prompt)
        max_tokens = 8

        with self.create_proxy(tp_size=1) as proxy:
            assert proxy.hmac_key is not None, "HMAC key should be generated"
            assert len(
                proxy.hmac_key
            ) == 32, f"HMAC key should be 32 bytes, got {len(proxy.hmac_key)}"

            # Verify key is properly stored in worker_kwargs
            assert 'hmac_key' in proxy.worker_kwargs, "HMAC key should be in worker_kwargs"
            assert proxy.worker_kwargs[
                'hmac_key'] is not None, "HMAC key in worker_kwargs should not be None"

            # Verify both references point to the same key object
            assert proxy.hmac_key is proxy.worker_kwargs['hmac_key'], \
                "HMAC key should be the same object in both locations"

            logger_debug(
                f"[Test] HMAC key verified: length={len(proxy.hmac_key)} bytes",
                color="green")

            # Verify RPC communication works with the generated key
            sampling_params = SamplingParams(max_tokens=max_tokens)
            result = proxy.generate(prompt_token_ids, sampling_params)
            assert similar(
                tokenizer.decode(result.outputs[0].token_ids), 'E F G H I J K L'
            ), "Generation should work with auto-generated HMAC key"

            logger_debug(
                f"[Test] HMAC key test passed: RPC communication successful",
                color="green")


if __name__ == "__main__":
    TestRpcProxy().test_tp1(20)
