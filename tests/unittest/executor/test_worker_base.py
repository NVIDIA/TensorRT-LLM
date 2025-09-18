import os
import sys
import time

import pytest
import torch

from tensorrt_llm._utils import mpi_comm, mpi_rank, mpi_world_size
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

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


class FakeWorker(WorkerBase):

    def __init__(self, engine: str, tp_size: int = 1):
        llm_args, executor_config = create_fake_executor_config(engine, tp_size)
        super().__init__(
            engine=engine,
            llm_args=llm_args,
            hf_model_dir=engine,
        )
        # Pass config in constructor and finalize with parameterless setup
        self._executor_config = executor_config
        self.llm_args = llm_args
        self.setup_engine()

    def start(self):
        pass

    def shutdown(self):
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None


class TestWorkerBase:

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
            print(stats)

    @pytest.mark.parametrize("timeout", [0.1, 0.2, 1])
    def test_fetch_responses_timeout(self, timeout: float):
        with self.FakeWorker(engine=model_path) as worker:
            # Not submit any request, and let the await_responses timeout.
            start_time = time.time()
            results = worker.await_responses(timeout=timeout)
            elapsed = time.time() - start_time
            print(f"await_responses latency: {elapsed:.3f} seconds")
            assert timeout / 2 <= elapsed <= timeout * 2, f"Latency out of expected range: {elapsed}"


def create_fake_executor_config(model_path, tp_size=1):
    llm_args = LlmArgs(model=model_path,
                       cuda_graph_config=None,
                       tensor_parallel_size=tp_size)

    executor_config = tllm.ExecutorConfig(1)
    executor_config.max_batch_size = 1
    executor_config.model_world_size = tp_size

    update_executor_config(
        executor_config,
        pytorch_backend_config=llm_args.get_pytorch_backend_config(),
        mapping=llm_args.parallel_config.to_mapping(),
        speculative_config=llm_args.speculative_config,
        hf_model_dir=model_path,
        max_input_len=20,
        max_seq_len=40,
        checkpoint_format=llm_args.checkpoint_format,
        checkpoint_loader=llm_args.checkpoint_loader,
    )

    return llm_args, executor_config


class TestRpcWorkerBaseTP2:

    def setup_method(self):
        self.llm_args = LlmArgs(model=model_path, tensor_parallel_size=2)
        self.session = self.create_worker_session()

    def create_worker_session(self):
        session = MpiPoolSession(n_workers=2)
        return session

    def test_create_executor(self):
        futures = self.session.submit(
            TestRpcWorkerBaseTP2.create_executor,
            engine=model_path,
            llm_args=self.llm_args,
        )
        # Wait for completion
        for future in futures:
            future.result()

        self.session.shutdown()

    @staticmethod
    def create_executor(engine, llm_args):
        rank = mpi_rank()
        world_size = mpi_world_size()
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)

        print(f"[Test] Rank {rank}/{world_size} using device {device_id}")

        # Synchronize all workers before creating executor
        mpi_comm().barrier()

        print(f"[Test] Rank {rank} creating WorkerBase...")
        executor = FakeWorker(engine=engine, tp_size=2)

        # For PyTorch backend, all ranks need to participate in setup
        print(f"[Test] Rank {rank} calling setup_engine...")

        # Setup the engine which contains another barrier
        executor.setup_engine()

        print(f"[Test] Rank {rank} setup_engine completed successfully")

        executor.shutdown()


if __name__ == "__main__":
    test_worker_base = TestWorkerBase()
    test_worker_base.test_fetch_stats()
