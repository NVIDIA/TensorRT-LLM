import os
import tempfile

import torch

from tensorrt_llm import serialization
from tensorrt_llm.auto_parallel.config import AutoParallelConfig
from tensorrt_llm.auto_parallel.parallelization import ParallelConfig
from tensorrt_llm.auto_parallel.simplifier import GraphConfig, StageType


class TestClass:

    def __init__(self, name: str):
        self.name = name


def test_serialization_allowed_class():
    obj = TestClass("test")
    serialization.register_approved_class(TestClass)
    module = TestClass.__module__
    assert module in serialization.BASE_EXAMPLE_CLASSES
    assert "TestClass" in serialization.BASE_EXAMPLE_CLASSES[module]
    a = serialization.dumps(obj)
    b = serialization.loads(a,
                            approved_imports=serialization.BASE_EXAMPLE_CLASSES)
    assert type(obj) == type(b) and obj.name == b.name


def test_serialization_disallowed_class():
    obj = TestClass("test")
    a = serialization.dumps(obj)
    excep = None
    try:
        serialization.loads(a, approved_imports={})
    except Exception as e:
        excep = e
        print(excep)
    assert isinstance(excep, ValueError) and str(
        excep) == "Import llmapi.test_serialization | TestClass is not allowed"


def test_serialization_basic_object():
    obj = {"test": "test"}
    a = serialization.dumps(obj)
    b = serialization.loads(a,
                            approved_imports=serialization.BASE_EXAMPLE_CLASSES)
    assert obj == b


def test_serialization_complex_object_allowed_class():
    obj = torch.tensor([1, 2, 3])
    a = serialization.dumps(obj)
    b = serialization.loads(a,
                            approved_imports=serialization.BASE_EXAMPLE_CLASSES)
    assert torch.all(obj == b)


def test_serialization_complex_object_partially_allowed_class():
    obj = torch.tensor([1, 2, 3])
    a = serialization.dumps(obj)
    excep = None
    try:
        b = serialization.loads(a,
                                approved_imports={
                                    'torch._utils': ['_rebuild_tensor_v2'],
                                })
    except Exception as e:
        excep = e
    assert isinstance(excep, ValueError) and str(
        excep) == "Import torch.storage | _load_from_bytes is not allowed"


def test_serialization_complex_object_disallowed_class():
    obj = torch.tensor([1, 2, 3])
    a = serialization.dumps(obj)
    excep = None
    try:
        serialization.loads(a)
    except Exception as e:
        excep = e
    assert isinstance(excep, ValueError) and str(
        excep) == "Import torch._utils | _rebuild_tensor_v2 is not allowed"


def test_parallel_config_serialization():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a ParallelConfig instance with some test data
        config = ParallelConfig()
        config.version = "test_version"
        config.network_hash = "test_hash"
        config.auto_parallel_config = AutoParallelConfig(
            world_size=2, gpus_per_node=2, cluster_key="test_cluster")
        config.graph_config = GraphConfig(num_micro_batches=2,
                                          num_blocks=3,
                                          num_stages=2)
        config.cost = 1.5
        config.stage_type = StageType.START

        config_path = os.path.join(tmpdir, "parallel_config.pkl")
        config.save(config_path)

        loaded_config = ParallelConfig.from_file(config_path)

        # Verify the loaded config matches the original
        assert loaded_config.version == config.version
        assert loaded_config.network_hash == config.network_hash
        assert loaded_config.auto_parallel_config.world_size == config.auto_parallel_config.world_size
        assert loaded_config.auto_parallel_config.gpus_per_node == config.auto_parallel_config.gpus_per_node
        assert loaded_config.auto_parallel_config.cluster_key == config.auto_parallel_config.cluster_key
        assert loaded_config.graph_config.num_micro_batches == config.graph_config.num_micro_batches
        assert loaded_config.graph_config.num_blocks == config.graph_config.num_blocks
        assert loaded_config.graph_config.num_stages == config.graph_config.num_stages
        assert loaded_config.cost == config.cost
        assert loaded_config.stage_type == config.stage_type


if __name__ == "__main__":
    test_serialization_allowed_class()
    test_parallel_config_serialization()
