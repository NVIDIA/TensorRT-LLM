import asyncio
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pytest
from test_worker_base import create_fake_executor_config

from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.rpc import RPCClient, RPCParams
from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy
from tensorrt_llm.executor.rpc_worker import RpcWorker
from tensorrt_llm.llmapi.mpi_session import MpiPoolSession
from tensorrt_llm.sampling_params import SamplingParams

# isort: off
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from utils.llm_data import llm_models_root
# isort: on

model_path = llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


class TestRpcWorkerTP1:

    def setup_method(self):
        self.executor_config = create_fake_executor_config(model_path)
        self.pool, self.addr = self.create_worker_pool()
        self.client = self.create_rpc_client(self.addr)
        self.client.setup_engine()
        time.sleep(10)

    def teardown_method(self):
        self.client.shutdown()
        self.pool.shutdown()
        self.client.close()

    def create_worker_pool(self):
        addr = GenerationExecutorRpcProxy.gen_uniq_rpc_addr()
        mp_context = multiprocessing.get_context(
            'spawn')  # spawn for CUDA context
        pool = ProcessPoolExecutor(max_workers=1, mp_context=mp_context)
        pool.submit(RpcWorker.main_task,
                    engine=model_path,
                    rpc_addr=addr,
                    executor_config=self.executor_config)
        return pool, addr

    def create_rpc_client(self, addr: str):
        client = RPCClient(addr)
        return client

    def test_create_shutdown(self):
        pass

    def test_fetch_responses_sync(self):
        self.client.submit(GenerationRequest(
            prompt_token_ids=[3, 4, 5],
            sampling_params=SamplingParams(max_tokens=5)),
                           __rpc_params=RPCParams(need_response=False))
        results = self.client.fetch_responses()
        assert len(results) == 1

    def test_fetch_responses_streaming_sync(self):
        self.client.submit(GenerationRequest(
            prompt_token_ids=[3, 4, 5],
            sampling_params=SamplingParams(max_tokens=5),
            streaming=True),
                           __rpc_params=RPCParams(need_response=False))

        results = []
        for i in range(10):
            res = self.client.fetch_responses()
            results.extend(res)
            print(f"fetch_responses {i} result: {results}")
        assert 0 < len(results) <= 5

        time.sleep(5)

    @pytest.mark.asyncio
    async def test_fetch_responses_streaming_async(self):
        self.client.submit(GenerationRequest(
            prompt_token_ids=[3, 4, 5],
            sampling_params=SamplingParams(max_tokens=5),
            streaming=True),
                           __rpc_params=RPCParams(need_response=False))

        results = []
        # Must fetch all the responses, or the PyExecutor will hang
        for i in range(10):
            res = await self.client.fetch_responses_async.call_async()
            results.extend(res)
            print(f"fetch_responses_async {i} result: {results}")
        assert 0 < len(results) <= 5

    @pytest.mark.asyncio
    @pytest.mark.parametrize("req_count", [10])
    async def test_main_loop_async(self, req_count: int):
        await asyncio.sleep(1)

        async def process_request_streaming():
            for i in range(req_count):
                ret = self.client.submit(
                    GenerationRequest(
                        prompt_token_ids=[3, 4, 5],
                        sampling_params=SamplingParams(max_tokens=5),
                        streaming=True),
                    __rpc_params=RPCParams(need_response=False))
                assert ret is None
                print("submit result: ", ret)

            # NOTE: known issue, the responses should be fetched before shutdown,
            # or the shutdown will hang.
            results = []

            print(f"start to fetch_responses_async")
            no = 0
            async for result in self.client.fetch_responses_loop_async.call_streaming(
            ):
                print(f"fetch_responses_async {no} result: {result}")
                results.extend(result)  # result is a list of responses
                no += 1
                if no >= req_count * 5:  # Break after receiving 5 batches
                    print(f"break after receiving {no} batches")
                    break
            print(f"Received {no} batches of streaming responses")
            print(f"fetch_responses result: {results}")
            assert results

        await process_request_streaming()

    def test_main_loop(self):
        time.sleep(1)

        def process_request():
            ret = self.client.submit(
                GenerationRequest(
                    prompt_token_ids=[3, 4, 5],
                    sampling_params=SamplingParams(max_tokens=10)),
                __rpc_params=RPCParams(need_response=False))
            assert ret is None  # need_response = False

            print(f"submit result: {ret}")
            print("call fetch_responses")
            # NOTE: known issue, the responses should be fetched before shutdown,
            # or the shutdown will hang.
            results = []
            time.sleep(8)  # wait for PyExecutor to finish the generation
            results.extend(
                self.client.fetch_responses())  # fetch_responses will block
            print(f"fetch_responses result: {results}")
            assert len(results) == 1  # one request, one response

        def process_request_streaming():
            ret = self.client.submit(
                GenerationRequest(prompt_token_ids=[3, 4, 5],
                                  sampling_params=SamplingParams(max_tokens=10),
                                  streaming=True),
                __rpc_params=RPCParams(need_response=False))
            assert ret is None
            print("submit result: ", ret)

            # NOTE: known issue, the responses should be fetched before shutdown,
            # or the shutdown will hang.
            results = []
            time.sleep(8)

            while not results:
                time.sleep(1)
                results.extend(
                    self.client.fetch_responses(__rpc_params=RPCParams(
                        timeout=10)))
                print(f"try fetch_responses result: {results}")
            print(f"fetch_responses result: {results}")
            assert results

        for i in range(5):
            process_request()
        process_request_streaming()


class TestRpcWorkerTP2:

    def setup_method(self):
        self.executor_config = create_fake_executor_config(model_path,
                                                           tp_size=2)
        self.session, self.addr, self.futures = self.create_worker_session()
        self.client = self.create_rpc_client(self.addr)
        self.client.setup_engine()
        time.sleep(10)

    def teardown_method(self):
        self.client.shutdown()
        self.session.shutdown()
        self.client.close()

    def create_worker_session(self):
        session = MpiPoolSession(n_workers=2)
        addr = GenerationExecutorRpcProxy.gen_uniq_rpc_addr()
        futures = session.submit(RpcWorker.main_task,
                                 engine=model_path,
                                 rpc_addr=addr,
                                 executor_config=self.executor_config,
                                 model_world_size=2)
        return session, addr, futures

    def create_rpc_client(self, addr: str):
        return RPCClient(addr)

    def test_create_shutdown(self):
        # Invoke setup_engine in rank 0, and that will unblock all the ranks to
        # invoke setup_engine simultaneously.
        pass

    def test_fetch_responses_sync(self):
        self.client.submit(GenerationRequest(
            prompt_token_ids=[3, 4, 5],
            sampling_params=SamplingParams(max_tokens=5)),
                           __rpc_params=RPCParams(need_response=False))
        results = self.client.fetch_responses()
        assert len(results) == 1
