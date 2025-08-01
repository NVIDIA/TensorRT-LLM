import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.rpc import RPCClient
from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy
from tensorrt_llm.executor.rpc_worker import RpcWorker
from tensorrt_llm.sampling_params import SamplingParams

# isort: off
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from utils.llm_data import llm_models_root
# isort: on

model_path = llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


class TestRpcWorker:

    def create_tp1_worker_process(self):
        addr = GenerationExecutorRpcProxy.gen_uniq_rpc_addr()
        # Use spawn method instead of fork
        mp_context = multiprocessing.get_context('spawn')
        pool = ProcessPoolExecutor(max_workers=1, mp_context=mp_context)
        pool.submit(RpcWorker.main_task, engine=model_path, rpc_addr=addr)
        return pool, addr

    def create_rpc_client(self, addr: str):
        client = RPCClient(addr)
        return client

    def test_main(self):
        pool, addr = self.create_tp1_worker_process()
        client = self.create_rpc_client(addr)
        client.setup_engine(engine=model_path)
        time.sleep(1)
        client.submit(
            GenerationRequest(prompt_token_ids=[3, 4, 5],
                              sampling_params=SamplingParams(max_tokens=10)))
        responses = client.fetch_responses()
        assert responses

        client.shutdown()
        pool.shutdown()


if __name__ == '__main__':
    worker = TestRpcWorker()
    worker.test_main()
