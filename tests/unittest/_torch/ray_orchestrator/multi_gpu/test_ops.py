import os
from operator import attrgetter
from typing import Optional

import pytest
import torch

try:
    import ray
except ModuleNotFoundError:
    from tensorrt_llm import ray_stub as ray

from tensorrt_llm._torch.distributed.communicator import TorchDist
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy
from tensorrt_llm.mapping import Mapping


@ray.remote(num_gpus=1)
class PgOpTest:

    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.master_address = os.environ["MASTER_ADDR"]

        assert len(ray.get_gpu_ids()) == 1
        self.gpu = int(ray.get_gpu_ids()[0])
        from tensorrt_llm.executor.ray_gpu_worker import RayWorkerWrapper
        local_gpu = RayWorkerWrapper.physical_to_local_id(self.gpu)
        torch.cuda.set_device(local_gpu)

    def _create_tcp_store(self,
                          port: Optional[int] = None
                          ) -> torch.distributed.TCPStore:
        actual_port = port if port is not None else 0
        return torch.distributed.TCPStore(host_name=self.master_address,
                                          port=actual_port,
                                          world_size=self.world_size,
                                          is_master=(self.rank == 0),
                                          wait_for_workers=False)

    def setup_tcp_store(self):
        if self.rank != 0:
            raise RuntimeError("Only the master worker can setup TCP store")
        self.store = self._create_tcp_store()
        return self.store.port

    def setup_distributed_env(self, port: int):
        if self.rank != 0:
            self.store = self._create_tcp_store(port)

        torch.distributed.init_process_group(backend="cuda:nccl,cpu:gloo",
                                             store=self.store,
                                             world_size=self.world_size,
                                             rank=self.rank)
        self.mapping = Mapping(world_size=self.world_size,
                               gpus_per_node=self.world_size,
                               tp_size=self.world_size,
                               rank=self.rank)
        TorchDist(self.mapping)

    def run(self, pg_op_name: str, test_tensor: torch.Tensor,
            expected_result: torch.Tensor, **additional_kwargs):
        additional_kwargs.update({
            "group": self.mapping.tp_group,
        })
        if pg_op_name == "allreduce_pg":
            additional_kwargs.update({"pg": self.mapping.tp_group_pg.boxed()})
        else:
            additional_kwargs.update(
                {"process_group": self.mapping.tp_group_pg.boxed()})
        test_tensor = test_tensor.cuda()
        expected_result = expected_result.cuda()
        pg_op_func = attrgetter(f'ops.trtllm.{pg_op_name}')(torch)
        output = pg_op_func(test_tensor, **additional_kwargs)
        if isinstance(output, (list, tuple)):
            result_tensor = output[0]
        else:
            result_tensor = output
        rtol, atol = 0.05, 0.15
        torch.testing.assert_close(result_tensor,
                                   expected_result,
                                   rtol=rtol,
                                   atol=atol)

        return True


@pytest.mark.gpu2
@pytest.mark.parametrize("hidden_size", [128, 1024],
                         ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize("seq_len", [16, 64], ids=lambda x: f"seqlen:{x}")
@pytest.mark.parametrize("var_len", [True, False], ids=lambda x: f"var_len:{x}")
def test_allgather_pg_op(setup_ray_cluster, seq_len, hidden_size, var_len):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    world_size = 2
    if var_len:
        test_tensor_list = [
            torch.randn((seq_len * (i + 1), hidden_size), dtype=dtype)
            for i in range(world_size)
        ]
        expected_result = torch.cat(test_tensor_list, dim=0)
        sizes = [seq_len * (i + 1) for i in range(world_size)]
    else:
        test_tensor = torch.randn((seq_len, hidden_size), dtype=dtype)
        expected_result = test_tensor.repeat(world_size, 1)
        sizes = None

    remotePGTests = []
    runtime_env = ray.runtime_env.RuntimeEnv()
    runtime_env["env_vars"] = os.environ.copy()
    runtime_env["env_vars"].update({
        "TLLM_DISABLE_MPI": "1",
        "MASTER_ADDR": "127.0.0.1",
    })

    for rank in range(world_size):
        remotePGTests.append(
            PgOpTest.options(runtime_env=runtime_env).remote(rank, world_size))

    ray.get(
        [remotePGTest.__ray_ready__.remote() for remotePGTest in remotePGTests])

    port = ray.get(remotePGTests[0].setup_tcp_store.remote())
    ray.get([
        remotePGTest.setup_distributed_env.remote(port)
        for remotePGTest in remotePGTests
    ])

    if var_len:
        results = ray.get([
            remotePGTest.run.remote("allgather_pg",
                                    test_tensor,
                                    expected_result,
                                    sizes=sizes) for remotePGTest, test_tensor
            in zip(remotePGTests, test_tensor_list)
        ])
    else:
        results = ray.get([
            remotePGTest.run.remote("allgather_pg",
                                    test_tensor,
                                    expected_result,
                                    sizes=sizes)
            for remotePGTest in remotePGTests
        ])
    for r in results:
        assert r is True


@pytest.mark.gpu2
@pytest.mark.parametrize("hidden_size", [128, 1024],
                         ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize("seq_len", [16, 64], ids=lambda x: f"seqlen:{x}")
@pytest.mark.parametrize("var_len", [True, False], ids=lambda x: f"var_len:{x}")
def test_reducescatter_pg_op(setup_ray_cluster, seq_len, hidden_size, var_len):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    world_size = 2
    if var_len:
        total_seq_len = sum([seq_len * (i + 1) for i in range(world_size)])
        test_tensor = torch.randn((total_seq_len, hidden_size), dtype=dtype)
        expected_result_list = []
        offset = 0
        for i in range(world_size):
            expected_result_list.append(test_tensor[offset:offset + seq_len *
                                                    (i + 1)] * world_size)
            offset += seq_len * (i + 1)
        sizes = [seq_len * (i + 1) for i in range(world_size)]
    else:
        test_tensor = torch.randn((seq_len * world_size, hidden_size),
                                  dtype=dtype)
        expected_result_list = [
            test_tensor[i * seq_len:(i + 1) * seq_len] * world_size
            for i in range(world_size)
        ]
        sizes = None

    runtime_env = ray.runtime_env.RuntimeEnv()
    runtime_env["env_vars"] = os.environ.copy()
    runtime_env["env_vars"].update({
        "TLLM_DISABLE_MPI": "1",
        "MASTER_ADDR": "127.0.0.1",
    })

    remotePGTests = []
    for rank in range(world_size):
        remotePGTests.append(
            PgOpTest.options(runtime_env=runtime_env).remote(rank, world_size))

    ray.get(
        [remotePGTest.__ray_ready__.remote() for remotePGTest in remotePGTests])

    port = ray.get(remotePGTests[0].setup_tcp_store.remote())
    ray.get([
        remotePGTest.setup_distributed_env.remote(port)
        for remotePGTest in remotePGTests
    ])

    results = ray.get([
        remotePGTest.run.remote("reducescatter_pg",
                                test_tensor,
                                expected_result,
                                sizes=sizes) for remotePGTest, expected_result
        in zip(remotePGTests, expected_result_list)
    ])
    for r in results:
        assert r is True


@pytest.mark.gpu2
@pytest.mark.parametrize("hidden_size", [128, 1024],
                         ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize("seq_len", [16, 64], ids=lambda x: f"seqlen:{x}")
def test_allreduce_pg_op(setup_ray_cluster, seq_len, hidden_size):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    world_size = 2
    test_tensor = torch.randn((seq_len, hidden_size), dtype=dtype)
    expected_result = test_tensor * world_size

    runtime_env = ray.runtime_env.RuntimeEnv()
    runtime_env["env_vars"] = os.environ.copy()
    runtime_env["env_vars"].update({
        "TLLM_DISABLE_MPI": "1",
        "MASTER_ADDR": "127.0.0.1",
    })

    remotePGTests = []
    for rank in range(world_size):
        remotePGTests.append(
            PgOpTest.options(runtime_env=runtime_env).remote(rank, world_size))

    ray.get(
        [remotePGTest.__ray_ready__.remote() for remotePGTest in remotePGTests])

    port = ray.get(remotePGTests[0].setup_tcp_store.remote())
    ray.get([
        remotePGTest.setup_distributed_env.remote(port)
        for remotePGTest in remotePGTests
    ])

    results = ray.get([
        remotePGTest.run.remote("allreduce_pg",
                                test_tensor,
                                expected_result,
                                residual=None,
                                norm_weight=None,
                                scale=None,
                                bias=None,
                                workspace=None,
                                strategy=AllReduceStrategy.NCCL,
                                op=AllReduceFusionOp.NONE,
                                eps=1e-5,
                                trigger_completion_at_end=True,
                                rank=i)
        for i, remotePGTest in enumerate(remotePGTests)
    ])
    for r in results:
        assert r is True


@ray.remote(num_gpus=1)
class CpBroadcastTest:
    """Test worker for cp_broadcast operations with context parallelism."""

    def __init__(self, rank, world_size, tp_size, cp_size):
        self.rank = rank
        self.world_size = world_size
        self.tp_size = tp_size
        self.cp_size = cp_size
        self.master_address = os.environ["MASTER_ADDR"]

        assert len(ray.get_gpu_ids()) == 1
        self.gpu = int(ray.get_gpu_ids()[0])
        from tensorrt_llm.executor.ray_gpu_worker import RayWorkerWrapper
        local_gpu = RayWorkerWrapper.physical_to_local_id(self.gpu)
        torch.cuda.set_device(local_gpu)

    def _create_tcp_store(self,
                          port: Optional[int] = None
                          ) -> torch.distributed.TCPStore:
        actual_port = port if port is not None else 0
        return torch.distributed.TCPStore(host_name=self.master_address,
                                          port=actual_port,
                                          world_size=self.world_size,
                                          is_master=(self.rank == 0),
                                          wait_for_workers=False)

    def setup_tcp_store(self):
        if self.rank != 0:
            raise RuntimeError("Only the master worker can setup TCP store")
        self.store = self._create_tcp_store()
        return self.store.port

    def setup_distributed_env(self, port: int):
        if self.rank != 0:
            self.store = self._create_tcp_store(port)

        torch.distributed.init_process_group(backend="cuda:nccl,cpu:gloo",
                                             store=self.store,
                                             world_size=self.world_size,
                                             rank=self.rank)
        self.mapping = Mapping(world_size=self.world_size,
                               gpus_per_node=self.world_size,
                               tp_size=self.tp_size,
                               cp_size=self.cp_size,
                               rank=self.rank)
        self.dist = TorchDist(self.mapping)

    def run_tensor_broadcast(self, root_tensor: torch.Tensor, root: int = 0):
        """Test broadcasting a tensor via cp_broadcast."""
        cp_rank = self.mapping.cp_rank
        if cp_rank == root:
            # Root rank has the tensor to broadcast.
            tensor = root_tensor.cuda()
        else:
            # Non-root ranks start with zeros.
            tensor = torch.zeros_like(root_tensor).cuda()

        result = self.dist.cp_broadcast(tensor, root=root)

        # After broadcast, all CP ranks should have the same tensor.
        expected = root_tensor.cuda()
        return torch.allclose(result, expected)

    def run_object_broadcast(self, root_obj, root: int = 0):
        """Test broadcasting a non-tensor object via cp_broadcast."""
        cp_rank = self.mapping.cp_rank
        if cp_rank == root:
            obj = root_obj
        else:
            obj = None

        result = self.dist.cp_broadcast(obj, root=root)

        # After broadcast, all CP ranks should have the same object.
        return result == root_obj

    def run_tp_cp_broadcast(self, root_obj, root: int = 0):
        """Test broadcasting an object via tp_cp_broadcast."""
        # For tp_cp_broadcast, only rank 0 in both TP and CP should have the object.
        tp_rank = self.mapping.tp_rank
        cp_rank = self.mapping.cp_rank
        if tp_rank == root and cp_rank == root:
            obj = root_obj
        else:
            obj = None

        result = self.dist.tp_cp_broadcast(obj, root=root)

        # After broadcast, all TP and CP ranks should have the same object.
        return result == root_obj


@pytest.mark.gpu2
@pytest.mark.parametrize("hidden_size", [128, 512], ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize("seq_len", [16, 32], ids=lambda x: f"seqlen:{x}")
def test_cp_broadcast_tensor(setup_ray_cluster, seq_len, hidden_size):
    """Test TorchDist.cp_broadcast with tensor data."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    world_size = 2
    tp_size = 1
    cp_size = 2  # Enable context parallelism.

    # Create tensor to broadcast from root.
    root_tensor = torch.randn((seq_len, hidden_size), dtype=dtype)

    runtime_env = ray.runtime_env.RuntimeEnv()
    runtime_env["env_vars"] = os.environ.copy()
    runtime_env["env_vars"].update({
        "TLLM_DISABLE_MPI": "1",
        "MASTER_ADDR": "127.0.0.1",
    })

    remote_tests = []
    for rank in range(world_size):
        remote_tests.append(
            CpBroadcastTest.options(runtime_env=runtime_env).remote(
                rank, world_size, tp_size, cp_size))

    ray.get([test.__ray_ready__.remote() for test in remote_tests])

    port = ray.get(remote_tests[0].setup_tcp_store.remote())
    ray.get([test.setup_distributed_env.remote(port) for test in remote_tests])

    # Test broadcasting from root=0.
    results = ray.get([
        test.run_tensor_broadcast.remote(root_tensor, root=0)
        for test in remote_tests
    ])
    for r in results:
        assert r is True, "Tensor broadcast from root=0 failed"


@pytest.mark.gpu2
@pytest.mark.parametrize("test_object", [
    {
        "key1": "value1",
        "key2": [1, 2, 3]
    },
    ["item1", "item2", {
        "nested": True
    }],
    "simple_string",
],
                         ids=["dict", "list", "string"])
@pytest.mark.parametrize("broadcast_method", [
    "run_object_broadcast",
    "run_tp_cp_broadcast",
],
                         ids=["cp_broadcast", "tp_cp_broadcast"])
def test_cp_tp_broadcast_object(setup_ray_cluster, test_object,
                                broadcast_method):
    """Test TorchDist.cp_broadcast and tp_cp_broadcast with non-tensor objects.

    This tests both cp_broadcast (for context parallelism only) and tp_cp_broadcast
    (for combined TP+CP broadcast used in helix parallelism).
    """
    world_size = 2
    tp_size = 1
    cp_size = 2  # Enable context parallelism.

    runtime_env = ray.runtime_env.RuntimeEnv()
    runtime_env["env_vars"] = os.environ.copy()
    runtime_env["env_vars"].update({
        "TLLM_DISABLE_MPI": "1",
        "MASTER_ADDR": "127.0.0.1",
    })

    remote_tests = []
    for rank in range(world_size):
        remote_tests.append(
            CpBroadcastTest.options(runtime_env=runtime_env).remote(
                rank, world_size, tp_size, cp_size))

    ray.get([test.__ray_ready__.remote() for test in remote_tests])

    port = ray.get(remote_tests[0].setup_tcp_store.remote())
    ray.get([test.setup_distributed_env.remote(port) for test in remote_tests])

    # Test broadcasting object from root=0 using the specified method.
    results = ray.get([
        getattr(test, broadcast_method).remote(test_object, root=0)
        for test in remote_tests
    ])
    for r in results:
        assert r is True, f"{broadcast_method} from root=0 failed for {type(test_object)}"
