import asyncio
import threading
import time

import pytest
import torch

from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.llmapi.utils import AsyncQueue

# isort: off
from .test_llm import llama_model_path
# isort: on


@pytest.mark.cpu_only
def test_LlmArgs_default_gpus_per_node():
    # default
    llm_args = TorchLlmArgs(model=llama_model_path)
    assert llm_args.gpus_per_node == torch.cuda.device_count()

    # set explicitly
    llm_args = TorchLlmArgs(model=llama_model_path, gpus_per_node=6)
    assert llm_args.gpus_per_node == 6


@pytest.mark.cpu_only
def test_AsyncQueue():
    queue = AsyncQueue()

    # put data to queue sync in a thread
    # async get data from queue in the current event loop
    # NOTE: the event loop in the two threads are different

    def put_data_to_queue():
        for i in range(10):
            time.sleep(0.1)
            queue.put(i)

    async def get_data_from_queue():
        for i in range(10):
            print(f"get: {queue.get()}")

    thread = threading.Thread(target=put_data_to_queue)
    thread.start()
    asyncio.run(get_data_from_queue())
    thread.join()
