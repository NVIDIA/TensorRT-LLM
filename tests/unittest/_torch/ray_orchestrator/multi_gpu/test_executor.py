import os

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch.utils import get_device_uuid


class DummyWorkerExtension:

    def additional_method(self):
        return "SUCCESS"


def test_worker_extension():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
              ray_worker_extension_cls="test_executor.DummyWorkerExtension",
              orchestrator_type="ray")
    result = llm._collective_rpc("additional_method")
    assert result[0] == "SUCCESS"


@pytest.mark.gpu2
def test_cuda_visible_device():
    """Placement via cuda_visible_device"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
              orchestrator_type="ray")

    infer_actor_uuids = llm._collective_rpc("report_device_id")

    del os.environ["CUDA_VISIBLE_DEVICES"]
    assert infer_actor_uuids[0] == get_device_uuid(1)
    print(f"{infer_actor_uuids=}")
