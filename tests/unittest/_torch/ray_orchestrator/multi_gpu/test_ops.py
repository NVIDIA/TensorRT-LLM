import os
from operator import attrgetter

import pytest
import torch

try:
    import ray
except ModuleNotFoundError:
    from tensorrt_llm import ray_stub as ray

from tensorrt_llm._torch.distributed.communicator import TorchDist
from tensorrt_llm._utils import get_free_port
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy
from tensorrt_llm.mapping import Mapping


@ray.remote(num_gpus=1)
class AllgatherPGTest:

    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.master_address = os.environ["MASTER_ADDR"]
        self.master_port = os.environ["MASTER_PORT"]
        assert len(ray.get_gpu_ids()) == 1
        self.gpu = int(ray.get_gpu_ids()[0])
        from tensorrt_llm.executor.ray_gpu_worker import RayWorkerWrapper
        self.local_gpu = RayWorkerWrapper.physical_to_local_id(self.gpu)

        torch.cuda.set_device(self.local_gpu)
        self.local_device = torch.device(f"cuda:{self.local_gpu}")

        torch.distributed.init_process_group(
            backend="cuda:nccl,cpu:gloo",
            init_method=f"tcp://{self.master_address}:{self.master_port}",
            world_size=world_size,
            rank=rank)

    @torch.inference_mode()
    def run_allgather_pg_op(self, test_tensor, expected_result, sizes):
        test_tensor = test_tensor.to(self.local_device)
        expected_result = expected_result.to(self.local_device)

        mapping = Mapping(world_size=self.world_size,
                          gpus_per_node=self.world_size,
                          tp_size=self.world_size,
                          rank=self.rank)
        if torch.distributed.is_initialized():
            TorchDist(mapping)
        else:
            raise RuntimeError("torch.distributed is not initialized")

        allgather_pg_op = attrgetter('ops.trtllm.allgather_pg')(torch)
        output = allgather_pg_op(test_tensor, sizes, mapping.tp_group,
                                 mapping.tp_group_pg.boxed())

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


@ray.remote(num_gpus=1)
class ReducescatterPGTest:

    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.master_address = os.environ["MASTER_ADDR"]
        self.master_port = os.environ["MASTER_PORT"]
        assert len(ray.get_gpu_ids()) == 1
        self.gpu = int(ray.get_gpu_ids()[0])
        from tensorrt_llm.executor.ray_gpu_worker import RayWorkerWrapper
        self.local_gpu = RayWorkerWrapper.physical_to_local_id(self.gpu)

        torch.cuda.set_device(self.local_gpu)
        self.local_device = torch.device(f"cuda:{self.local_gpu}")

        torch.distributed.init_process_group(
            backend="cuda:nccl,cpu:gloo",
            init_method=f"tcp://{self.master_address}:{self.master_port}",
            world_size=world_size,
            rank=rank)

    @torch.inference_mode()
    def run_reducescatter_pg_op(self, test_tensor, expected_result, sizes):
        test_tensor = test_tensor.to(self.local_device)
        expected_result = expected_result.to(self.local_device)

        mapping = Mapping(world_size=self.world_size,
                          gpus_per_node=self.world_size,
                          tp_size=self.world_size,
                          rank=self.rank)
        if torch.distributed.is_initialized():
            TorchDist(mapping)
        else:
            raise RuntimeError("torch.distributed is not initialized")

        reducescatter_pg_op = attrgetter('ops.trtllm.reducescatter_pg')(torch)
        output = reducescatter_pg_op(test_tensor, sizes, mapping.tp_group,
                                     mapping.tp_group_pg.boxed())

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


@ray.remote(num_gpus=1)
class AllreducePGTest:

    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.master_address = os.environ["MASTER_ADDR"]
        self.master_port = os.environ["MASTER_PORT"]
        assert len(ray.get_gpu_ids()) == 1
        self.gpu = int(ray.get_gpu_ids()[0])
        from tensorrt_llm.executor.ray_gpu_worker import RayWorkerWrapper
        self.local_gpu = RayWorkerWrapper.physical_to_local_id(self.gpu)

        torch.cuda.set_device(self.local_gpu)
        self.local_device = torch.device(f"cuda:{self.local_gpu}")

        torch.distributed.init_process_group(
            backend="cuda:nccl,cpu:gloo",
            init_method=f"tcp://{self.master_address}:{self.master_port}",
            world_size=world_size,
            rank=rank)

    @torch.inference_mode()
    def run_allreduce_pg_op(self, test_tensor, expected_result):
        test_tensor = test_tensor.to(self.local_device)
        expected_result = expected_result.to(self.local_device)

        mapping = Mapping(world_size=self.world_size,
                          gpus_per_node=self.world_size,
                          tp_size=self.world_size,
                          rank=self.rank)
        if torch.distributed.is_initialized():
            TorchDist(mapping)
        else:
            raise RuntimeError("torch.distributed is not initialized")

        allreduce_pg_op = attrgetter('ops.trtllm.allreduce_pg')(torch)
        output = allreduce_pg_op(
            input=test_tensor,
            residual=None,
            norm_weight=None,
            scale=None,
            bias=None,
            workspace=None,
            group=mapping.tp_group,
            strategy=AllReduceStrategy.NCCL,
            op=AllReduceFusionOp.NONE,  # Pure allreduce, no fusion
            eps=1e-5,
            trigger_completion_at_end=True,
            rank=self.rank,
            pg=mapping.tp_group_pg.boxed())

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
def test_allgather_pg_op(seq_len, hidden_size, var_len):
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

    runtime_env = {
        "env_vars": {
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"
        }
    }
    ray_init_args = {
        "include_dashboard": False,
        "namespace": "test_allgather_pg_op",
        "ignore_reinit_error": True,
        "runtime_env": runtime_env
    }

    try:
        ray.init(address="local", **ray_init_args)

        master_port = get_free_port()
        runtime_env = ray.runtime_env.RuntimeEnv()
        runtime_env["env_vars"] = os.environ.copy()
        runtime_env["env_vars"].update({
            "TLLM_DISABLE_MPI": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port)
        })

        remotePGTests = []
        for rank in range(world_size):
            remotePGTests.append(
                AllgatherPGTest.options(runtime_env=runtime_env).remote(
                    rank, world_size))

        ray.get([
            remotePGTest.__ray_ready__.remote()
            for remotePGTest in remotePGTests
        ])

        if var_len:
            results = ray.get([
                remotePGTest.run_allgather_pg_op.remote(test_tensor,
                                                        expected_result, sizes)
                for remotePGTest, test_tensor in zip(remotePGTests,
                                                     test_tensor_list)
            ])
        else:
            results = ray.get([
                remotePGTest.run_allgather_pg_op.remote(test_tensor,
                                                        expected_result, sizes)
                for remotePGTest in remotePGTests
            ])
        for r in results:
            assert r is True
    finally:
        if ray.is_initialized():
            ray.shutdown()


@pytest.mark.gpu2
@pytest.mark.parametrize("hidden_size", [128, 1024],
                         ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize("seq_len", [16, 64], ids=lambda x: f"seqlen:{x}")
@pytest.mark.parametrize("var_len", [True, False], ids=lambda x: f"var_len:{x}")
def test_reducescatter_pg_op(seq_len, hidden_size, var_len):
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

    runtime_env = {
        "env_vars": {
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"
        }
    }
    ray_init_args = {
        "include_dashboard": False,
        "namespace": "test_reducescatter_pg_op",
        "ignore_reinit_error": True,
        "runtime_env": runtime_env
    }

    try:
        ray.init(address="local", **ray_init_args)

        master_port = get_free_port()
        runtime_env = ray.runtime_env.RuntimeEnv()
        runtime_env["env_vars"] = os.environ.copy()
        runtime_env["env_vars"].update({
            "TLLM_DISABLE_MPI": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port)
        })

        remotePGTests = []
        for rank in range(world_size):
            remotePGTests.append(
                ReducescatterPGTest.options(runtime_env=runtime_env).remote(
                    rank, world_size))

        ray.get([
            remotePGTest.__ray_ready__.remote()
            for remotePGTest in remotePGTests
        ])

        if var_len:
            results = ray.get([
                remotePGTest.run_reducescatter_pg_op.remote(
                    test_tensor, expected_result, sizes)
                for remotePGTest, expected_result in zip(
                    remotePGTests, expected_result_list)
            ])
        else:
            results = ray.get([
                remotePGTest.run_reducescatter_pg_op.remote(
                    test_tensor, expected_result, sizes)
                for remotePGTest, expected_result in zip(
                    remotePGTests, expected_result_list)
            ])
        for r in results:
            assert r is True
    finally:
        if ray.is_initialized():
            ray.shutdown()


@pytest.mark.gpu2
@pytest.mark.parametrize("hidden_size", [128, 1024],
                         ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize("seq_len", [16, 64], ids=lambda x: f"seqlen:{x}")
def test_allreduce_pg_op(seq_len, hidden_size):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    world_size = 2
    test_tensor = torch.randn((seq_len, hidden_size), dtype=dtype)
    expected_result = test_tensor * world_size

    runtime_env = {
        "env_vars": {
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"
        }
    }
    ray_init_args = {
        "include_dashboard": False,
        "namespace": "test_allreduce_pg_op",
        "ignore_reinit_error": True,
        "runtime_env": runtime_env
    }

    try:
        ray.init(address="local", **ray_init_args)

        master_port = get_free_port()
        runtime_env = ray.runtime_env.RuntimeEnv()
        runtime_env["env_vars"] = os.environ.copy()
        runtime_env["env_vars"].update({
            "TLLM_DISABLE_MPI": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port)
        })

        remotePGTests = []
        for rank in range(world_size):
            remotePGTests.append(
                AllreducePGTest.options(runtime_env=runtime_env).remote(
                    rank, world_size))

        ray.get([
            remotePGTest.__ray_ready__.remote()
            for remotePGTest in remotePGTests
        ])

        results = ray.get([
            remotePGTest.run_allreduce_pg_op.remote(test_tensor,
                                                    expected_result)
            for remotePGTest in remotePGTests
        ])
        for r in results:
            assert r is True
    finally:
        if ray.is_initialized():
            ray.shutdown()
