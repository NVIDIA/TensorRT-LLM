import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from test_worker_base import create_fake_executor_config

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

    def setup_method(self):
        self.executor_config = create_fake_executor_config(model_path)

    def create_tp1_worker_process(self):
        addr = GenerationExecutorRpcProxy.gen_uniq_rpc_addr()
        # Use spawn method instead of fork
        mp_context = multiprocessing.get_context('spawn')
        pool = ProcessPoolExecutor(max_workers=1, mp_context=mp_context)
        pool.submit(RpcWorker.main_task,
                    engine=model_path,
                    rpc_addr=addr,
                    executor_config=self.executor_config)
        return pool, addr

    def create_rpc_client(self, addr: str):
        client = RPCClient(addr)
        return client

    def test_main_loop(self):
        pool, addr = self.create_tp1_worker_process()
        client = self.create_rpc_client(addr)
        client.setup_engine(__rpc_timeout=120)
        time.sleep(1)

        def process_request():
            ret = client.submit(GenerationRequest(
                prompt_token_ids=[3, 4, 5],
                sampling_params=SamplingParams(max_tokens=10)),
                                __rpc_need_response=False)
            assert ret is None  # need_response = False

            print(f"submit result: {ret}")
            print("call fetch_responses")
            # NOTE: known issue, the responses should be fetched before shutdown,
            # or the shutdown will hang.
            results = []
            time.sleep(8)  # wait for PyExecutor to finish the generation
            results.extend(
                client.fetch_responses())  # fetch_responses will block
            print(f"fetch_responses result: {results}")
            assert len(results) == 1  # one request, one response

        def process_request_streaming():
            ret = client.submit(GenerationRequest(
                prompt_token_ids=[3, 4, 5],
                sampling_params=SamplingParams(max_tokens=10),
                streaming=True),
                                __rpc_need_response=False)
            assert ret is None
            print("submit result: ", ret)

            # NOTE: known issue, the responses should be fetched before shutdown,
            # or the shutdown will hang.
            results = []
            time.sleep(8)

            while not results:
                time.sleep(1)
                results.extend(client.fetch_responses(__rpc_timeout=10))
                print(f"try fetch_responses result: {results}")
            print(f"fetch_responses result: {results}")
            assert results

        for i in range(5):
            process_request()
        process_request_streaming()

        print("call shutdown")
        client.shutdown(__rpc_timeout=10)
        pool.shutdown()
        client.close()


if __name__ == '__main__':
    worker = TestRpcWorker()
    worker.test_main_loop()
