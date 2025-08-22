import os
import sys
import time

import pytest
from test_worker_base import create_fake_executor_config

from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy
from tensorrt_llm.llmapi.mpi_session import MpiPoolSession
from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer
from tensorrt_llm.sampling_params import SamplingParams

# isort: off
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from utils.llm_data import llm_models_root
from utils.util import similar
# isort: on

model_path = llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


class TestRpcProxy:

    def create_proxy(self, tp_size: int):
        # Create executor config with the correct tp_size
        executor_config = create_fake_executor_config(model_path,
                                                      tp_size=tp_size)

        mpi_session = MpiPoolSession(n_workers=tp_size)
        proxy = GenerationExecutorRpcProxy(
            worker_kwargs={
                "engine": model_path,
                "executor_config": executor_config,
                "model_world_size": tp_size,
            },
            model_world_size=tp_size,
            mpi_session=mpi_session,
        )

        # Add additional wait for PyTorch backend with multi-rank setup
        if tp_size > 1:
            print(f"[Test] Waiting for {tp_size} ranks to initialize...")
            time.sleep(
                5)  # Give more time for multi-rank PyTorch initialization

        return proxy

    @pytest.mark.parametrize("num_reqs", [1, 10])
    def test_tp1(self, num_reqs):
        tokenizer = TransformersTokenizer.from_pretrained(model_path)
        prompt = "A B C D"
        prompt_token_ids = tokenizer.encode(prompt)
        max_tokens = 8

        with self.create_proxy(tp_size=1) as proxy:
            sampling_params = SamplingParams(max_tokens=max_tokens)
            for _ in range(num_reqs):
                result = proxy.generate(prompt_token_ids, sampling_params)
                print(f"get result: {result}")
                assert similar(tokenizer.decode(result.outputs[0].token_ids),
                               'E F G H I J K L')

    @pytest.mark.parametrize("num_reqs", [1, 10])
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


if __name__ == "__main__":
    TestRpcProxyTp1().test_tp1()
