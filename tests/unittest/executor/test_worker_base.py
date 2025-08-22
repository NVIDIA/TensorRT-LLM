import os
import sys
import time

import pytest
import torch

from tensorrt_llm._utils import mpi_comm, mpi_rank, mpi_world_size
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.llmapi.mpi_session import MpiPoolSession, set_mpi_session_cpp

# isort: off
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from utils.llm_data import llm_models_root
# isort: on

from tensorrt_llm._torch.pyexecutor.config import update_executor_config
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.worker_base import WorkerBase
from tensorrt_llm.llmapi.llm_args import LlmArgs
from tensorrt_llm.sampling_params import SamplingParams

default_model_name = "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
model_path = llm_models_root() / default_model_name


class TestWorkerBase:

    class FakeWorker(WorkerBase):

        def __init__(self, engine: str):
            super().__init__(engine=engine)
            executor_config = create_fake_executor_config(engine)
            # Pass config in constructor and finalize with parameterless setup
            self._executor_config = executor_config
            self.setup_engine()

    def test_create_engine(self):
        with self.FakeWorker(engine=model_path) as worker:
            print(f"Created engine: {worker.engine}")

    def test_submit_request(self):
        sampling_params = SamplingParams(max_tokens=10)
        request = GenerationRequest(prompt_token_ids=[3, 4, 5],
                                    sampling_params=sampling_params)
        with self.FakeWorker(engine=model_path) as worker:
            print(f"Created engine: {worker.engine}")
            worker.submit(request)
            for i in range(10):
                time.sleep(0.5)
                worker.await_responses()
            print(f"Submitted request: {request}")
            time.sleep(6)

    def test_fetch_stats(self):
        request = GenerationRequest(
            prompt_token_ids=[3, 4, 5],
            sampling_params=SamplingParams(max_tokens=10))
        with self.FakeWorker(engine=model_path) as worker:
            worker.submit(request)
            time.sleep(1)
            worker.await_responses()
            stats = worker.fetch_stats()
            assert len(stats) > 0

    def test_dispatch_stats_task(self):
        request = GenerationRequest(
            prompt_token_ids=[3, 4, 5],
            sampling_params=SamplingParams(max_tokens=10))
        with self.FakeWorker(engine=model_path) as worker:
            worker.submit(request)
            worker.await_responses()
            time.sleep(10)
            stats = worker.fetch_stats()
            assert len(stats) == 1

    @pytest.mark.parametrize("timeout", [0.1, 0.2, 1])
    def test_fetch_responses_timeout(self, timeout: float):
        with self.FakeWorker(engine=model_path) as worker:
            # Not submit any request, and let the await_responses timeout.
            start_time = time.time()
            results = worker.await_responses(timeout=timeout)
            elapsed = time.time() - start_time
            print(f"await_responses latency: {elapsed:.3f} seconds")
            assert timeout / 2 <= elapsed <= timeout * 2, f"Latency out of expected range: {elapsed}"
            assert results is None


def create_fake_executor_config(model_path, tp_size=1):
    llm_args = LlmArgs(model=model_path, cuda_graph_config=None)

    executor_config = tllm.ExecutorConfig(1)
    executor_config.max_batch_size = 1
    executor_config.model_world_size = tp_size

    # For PyTorch backend with TP > 1, we need proper parallel config
    if tp_size > 1:
        llm_args.parallel_config.tp_size = tp_size

    update_executor_config(
        executor_config,
        backend="pytorch",
        pytorch_backend_config=llm_args.get_pytorch_backend_config(),
        mapping=llm_args.parallel_config.to_mapping(),
        speculative_config=llm_args.speculative_config,
        hf_model_dir=model_path,
        max_input_len=20,
        max_seq_len=40,
        checkpoint_format=llm_args.checkpoint_format,
        checkpoint_loader=llm_args.checkpoint_loader,
    )

    return executor_config


class TestRpcWorkerBaseTP2:

    def setup_method(self):
        self.executor_config = create_fake_executor_config(model_path,
                                                           tp_size=2)
        self.session = self.create_worker_session()
        # No need to sleep here - the session is ready immediately

    def create_worker_session(self):
        session = MpiPoolSession(n_workers=2)
        return session

    def test_create_executor(self):
        futures = self.session.submit(TestRpcWorkerBaseTP2.create_executor,
                                      engine=model_path,
                                      executor_config=self.executor_config)
        # Wait for completion
        for future in futures:
            future.result()

        self.session.shutdown()

    @staticmethod
    def create_executor(engine, executor_config):
        # Set MPI session for C++ backend
        set_mpi_session_cpp(mpi_comm())

        # Set CUDA device for this rank
        rank = mpi_rank()
        world_size = mpi_world_size()
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)

        # Don't set CUDA_VISIBLE_DEVICES as it interferes with MPI multi-GPU setup

        print(f"[Test] Rank {rank}/{world_size} using device {device_id}")

        # Synchronize all workers before creating executor
        mpi_comm().barrier()

        try:
            print(f"[Test] Rank {rank} creating WorkerBase...")
            executor = WorkerBase(engine=engine,
                                  executor_config=executor_config)

            # For PyTorch backend, all ranks need to participate in setup
            print(f"[Test] Rank {rank} calling setup_engine...")

            # Setup the engine which contains another barrier
            executor.setup_engine()

            print(f"[Test] Rank {rank} setup_engine completed successfully")

            executor.shutdown()

        except Exception as e:
            print(f"[Test] Rank {rank} failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise

        return None  # executor cannot be picked and returned


if __name__ == "__main__":
    test_worker_base = TestWorkerBase()
    test_worker_base.test_fetch_stats()
