import os
import sys

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


class TestRpcProxyTp1:

    def setup_method(self):
        self.executor_config = create_fake_executor_config(model_path)

    def create_proxy(self, tp_size: int):
        mpi_session = MpiPoolSession(n_workers=tp_size)
        proxy = GenerationExecutorRpcProxy(
            worker_kwargs={
                "engine": model_path,
                "executor_config": self.executor_config,
                "model_world_size": tp_size,
            },
            mpi_session=mpi_session,
        )
        return proxy

    def test_tp1(self):
        tokenizer = TransformersTokenizer.from_pretrained(model_path)
        prompt = "A B C D"
        prompt_token_ids = tokenizer.encode(prompt)
        max_tokens = 8

        with self.create_proxy(tp_size=1) as proxy:
            sampling_params = SamplingParams(max_tokens=max_tokens)
            result = proxy.generate(prompt_token_ids, sampling_params)
            print(f"get result: {result}")
            assert similar(tokenizer.decode(result.outputs[0].token_ids),
                           'E F G H I J K L')


if __name__ == "__main__":
    TestRpcProxyTp1().test_tp1()
