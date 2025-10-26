import asyncio
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pytest
from test_base_worker import create_fake_executor_config

from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.rpc import RPCClient
from tensorrt_llm.executor.rpc.rpc_common import get_unique_ipc_addr
from tensorrt_llm.executor.rpc_worker import RpcWorker
from tensorrt_llm.llmapi.mpi_session import MpiPoolSession
from tensorrt_llm.sampling_params import SamplingParams

# isort: off
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from utils.llm_data import llm_models_root
from utils.util import skip_single_gpu
# isort: on

model_path = llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
assert model_path.exists()


class TestRpcWorkerTP1:

    def setup_method(self):
        self.llm_args, self.executor_config = create_fake_executor_config(
            model_path)
        self.pool, self.addr = self.create_worker_pool()
        self.client = self.create_rpc_client(self.addr)
        self.client.setup_engine().remote()
        print(f"Worker setup engine done")
        time.sleep(10)

    def teardown_method(self):
        self.client.shutdown().remote()
        self.pool.shutdown()
        self.client.close()

    def create_worker_pool(self):
        addr = get_unique_ipc_addr()
        mp_context = multiprocessing.get_context(
            'spawn')  # spawn for CUDA context
        pool = ProcessPoolExecutor(max_workers=1, mp_context=mp_context)
        pool.submit(
            RpcWorker.main_task,
            engine=model_path,
            rpc_addr=addr,
            executor_config=self.executor_config,
            llm_args=self.llm_args,
            hf_model_dir=model_path,
        )
        return pool, addr

    def create_rpc_client(self, addr: str):
        client = RPCClient(addr)
        return client

    def test_create_shutdown(self):
        pass

    def test_fetch_responses_sync(self):
        # Wait a bit to ensure engine is ready
        time.sleep(1)

        print(f"start to submit")
        self.client.submit(
            GenerationRequest(prompt_token_ids=[3, 4, 5],
                              sampling_params=SamplingParams(
                                  max_tokens=5)), ).remote(need_response=False)
        print(f"submit done")

        time.sleep(3)

        results = []
        # Fetch responses
        results.extend(self.client.fetch_responses().remote())
        assert len(results) == 1

    @pytest.mark.skip(reason="https://nvbugs/5583261")
    def test_fetch_responses_streaming_sync(self):
        self.client.submit(
            GenerationRequest(prompt_token_ids=[3, 4, 5],
                              sampling_params=SamplingParams(max_tokens=5),
                              streaming=True), ).remote(need_response=False)

        results = []
        for i in range(10):
            res = self.client.fetch_responses().remote(timeout=1.0)
            results.extend(res)
            print(f"fetch_responses {i} result: {results}")
            # If we've received enough results, break early
            if len(results) >= 5:
                break
        assert 0 < len(results) <= 5

    @pytest.mark.skip(reason="https://nvbugs/5583261")
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
                        streaming=True), ).remote(need_response=False)
                assert ret is None
                print("submit result: ", ret)

            # NOTE: known issue, the responses should be fetched before shutdown,
            # or the shutdown will hang.
            results = []
            responses_per_client = {}
            expected_responses_per_client = 5  # max_tokens=5

            print(f"start to fetch_responses_async")
            no = 0
            async for result in self.client.fetch_responses_loop_async(
            ).remote_streaming():
                if result:  # result is already a list of lists
                    print(
                        f"fetch_responses_async batch {no}, received {len(result)} sub-batches"
                    )
                    for batch in result:
                        if isinstance(batch, list):
                            print(f"  Sub-batch has {len(batch)} responses")
                            results.extend(batch)
                            # Track responses per client
                            for response in batch:
                                client_id = response.client_id
                                if client_id not in responses_per_client:
                                    responses_per_client[client_id] = 0
                                responses_per_client[client_id] += 1
                        else:
                            # Single response
                            results.append(batch)
                            client_id = batch.client_id
                            if client_id not in responses_per_client:
                                responses_per_client[client_id] = 0
                            responses_per_client[client_id] += 1

                no += 1

                # Check if all clients have received their expected responses
                completed_clients = sum(
                    1 for count in responses_per_client.values()
                    if count >= expected_responses_per_client)

                print(f"Responses per client: {responses_per_client}")
                print(f"Completed clients: {completed_clients}/{req_count}")

                # Break when we've received all expected responses
                if completed_clients >= req_count:
                    print(
                        f"All {completed_clients} clients completed after {no} batches"
                    )
                    break

                # Safety break to prevent infinite loop
                if no >= req_count * 20:  # Much higher limit as safety
                    print(f"Safety break after {no} batches")
                    break

            print(f"Received {no} batches of streaming responses")
            print(f"Total responses received: {len(results)}")
            print(f"Final responses per client: {responses_per_client}")
            assert results
            assert len(responses_per_client) >= req_count

        await process_request_streaming()

    @pytest.mark.skip(reason="https://nvbugs/5583261")
    @pytest.mark.asyncio
    async def test_fetch_stats_loop_async(self):
        await asyncio.sleep(1)
        results = []
        max_batches = 5

        async def consume_stats():
            async for stats in self.client.fetch_stats_loop_async(
            ).remote_streaming():
                results.append(stats)
                assert not stats  # empty stats
                if len(results) >= max_batches:
                    break

        await asyncio.wait_for(consume_stats(), timeout=5)

        assert len(results) == max_batches
        assert all(not stats for stats in results)


class TestRpcWorkerTP2:

    def setup_method(self):
        self.llm_args, self.executor_config = create_fake_executor_config(
            model_path, tp_size=2)
        self.session, self.addr, self.futures = self.create_worker_session()
        self.client = self.create_rpc_client(self.addr)
        self.client.setup_engine().remote()
        time.sleep(10)

    def teardown_method(self):
        self.client.shutdown().remote()
        self.session.shutdown()
        self.client.close()

    def create_worker_session(self):
        session = MpiPoolSession(n_workers=2)
        addr = get_unique_ipc_addr()
        futures = session.submit(RpcWorker.main_task,
                                 engine=model_path,
                                 rpc_addr=addr,
                                 executor_config=self.executor_config,
                                 llm_args=self.llm_args,
                                 hf_model_dir=model_path,
                                 model_world_size=2)
        return session, addr, futures

    def create_rpc_client(self, addr: str):
        return RPCClient(addr)

    @skip_single_gpu
    @pytest.mark.gpu2
    @pytest.mark.skip(reason="https://nvbugs/5583261")
    def test_create_shutdown(self):
        # Invoke setup_engine in rank 0, and that will unblock all the ranks to
        # invoke setup_engine simultaneously.
        pass

    @skip_single_gpu
    @pytest.mark.gpu2
    @pytest.mark.skip(reason="https://nvbugs/5583261")
    def test_fetch_responses_sync(self):
        # Wait a bit to ensure engine is ready
        time.sleep(1)

        self.client.submit(
            GenerationRequest(prompt_token_ids=[3, 4, 5],
                              sampling_params=SamplingParams(
                                  max_tokens=5)), ).remote(need_response=False)

        # Wait for generation to complete
        time.sleep(3)

        results = []
        # Fetch responses with timeout
        results.extend(self.client.fetch_responses().remote(timeout=5))
        assert len(results) == 1
