import os
import sys
from queue import Queue

# isort: off
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from utils.llm_data import llm_models_root
from tensorrt_llm.bindings import executor as tllm
# isort: on

from tensorrt_llm._torch.pyexecutor.config import update_executor_config
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.worker_base import WorkerBase
from tensorrt_llm.llmapi.llm_args import LlmArgs
from tensorrt_llm.sampling_params import SamplingParams

default_model_name = "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
model_path = llm_models_root() / default_model_name


class TestWorkerBase:

    def test_create_engine(self):
        with WorkerBase(engine=model_path) as worker:
            pass

    def test_submit_request(self):
        sampling_params = SamplingParams(max_tokens=10)
        request = GenerationRequest(prompt_token_ids=[3, 4, 5],
                                    sampling_params=sampling_params)
        with WorkerBase(engine=model_path) as worker:
            worker.submit(request)

    def test_await_responses(self):
        sampling_params = SamplingParams(max_tokens=10)
        request = GenerationRequest(prompt_token_ids=[3, 4, 5],
                                    sampling_params=sampling_params)
        with WorkerBase(engine=model_path) as worker:
            result_queue = Queue()
            worker.set_result_queue(result_queue)

            worker.submit(request)
            for i in range(10):
                worker.await_responses()

            assert result_queue.qsize() > 0

    def _create_executor_config(self):
        llm_args = LlmArgs(model=model_path, cuda_graph_config=None)

        executor_config = tllm.ExecutorConfig(1)
        executor_config.max_batch_size = 1

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


if __name__ == "__main__":
    test_worker_base = TestWorkerBase()
    test_worker_base.test_create_engine()
