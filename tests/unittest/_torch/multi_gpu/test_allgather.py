import os

import pytest
import ray
import torch

from tensorrt_llm._torch.distributed.communicator import TorchDist
from tensorrt_llm._utils import get_free_port
from tensorrt_llm.mapping import Mapping


@ray.remote(num_gpus=1)
class AllgatherPGTest:

    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.master_address = os.environ["MASTER_ADDR"]
        self.master_port = os.environ["MASTER_PORT"]
        assert len(ray.get_gpu_ids()) == 1

        torch.distributed.init_process_group(
            backend="cuda:nccl,cpu:gloo",
            init_method=f"tcp://{self.master_address}:{self.master_port}",
            world_size=world_size,
            rank=rank)

    @torch.inference_mode()
    def run_allgather_pg_op(self, test_tensor, expected_result, sizes):
        torch.cuda.set_device(0)
        test_tensor = test_tensor.cuda(0)
        expected_result = expected_result.cuda(0)

        mapping = Mapping(world_size=self.world_size,
                          gpus_per_node=self.world_size,
                          tp_size=self.world_size,
                          rank=self.rank)
        if torch.distributed.is_initialized():
            TorchDist(mapping)
        else:
            raise RuntimeError("torch.distributed is not initialized")

        module = torch
        op_path = ['ops', 'trtllm', 'allgather_pg']
        for attr in op_path:
            module = getattr(module, attr)
        allgather_pg_op = module
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


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Requires at least 2 GPUs for this test")
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
    ray_init_args = {
        "include_dashboard": False,
        "namespace": "test_allgather_pg_op",
        "ignore_reinit_error": True
    }

    try:
        ray.init(**ray_init_args)

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
    except Exception as e:
        if ray.is_initialized():
            ray.shutdown()
        raise e

    ray.shutdown()
    for r in results:
        assert r is True
