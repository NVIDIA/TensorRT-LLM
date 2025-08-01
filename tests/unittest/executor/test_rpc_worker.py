import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from test_worker_base import TestWorkerBase

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

    def __init__(self):
        self.executor_config = TestWorkerBase.create_fake_executor_config(
            model_path)

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

    def test_main(self):
        pool, addr = self.create_tp1_worker_process()
        client = self.create_rpc_client(addr)
        print("call setup_engine")
        client.setup_engine(engine=model_path,
                            executor_config=self.executor_config,
                            __rpc_timeout=120)
        print("call submit")
        time.sleep(1)

        def process_request():
            ret = client.submit(GenerationRequest(
                prompt_token_ids=[3, 4, 5],
                sampling_params=SamplingParams(max_tokens=10)),
                                __rpc_need_response=False)
            assert ret is None

            print(f"submit result: {ret}")
            print("call fetch_responses")
            # NOTE: known issue, the responses should be fetched before shutdown,
            # or the shutdown will hang.
            results = []
            for i in range(3):
                time.sleep(3)
                results.extend(client.fetch_responses())
                print(f"fetch_responses result: {results}")
            assert len(results) == 1

        def process_request_streaming():
            ret = client.submit(prompt_token_ids=[3, 4, 5],
                                sampling_params=SamplingParams(max_tokens=10),
                                streaming=True,
                                __rpc_need_response=False)
            assert ret is None

            print("call fetch_responses")
            # NOTE: known issue, the responses should be fetched before shutdown,
            # or the shutdown will hang.
            results = []
            for i in range(3):
                time.sleep(3)
                results.extend(client.fetch_responses())
                print(f"fetch_responses result: {results}")
            print(f"generate_async result: {results}")

        process_request()
        process_request_streaming()

        print("call shutdown")
        client.shutdown(__rpc_timeout=10)
        pool.shutdown()


if __name__ == '__main__':
    worker = TestRpcWorker()
    worker.test_main()
