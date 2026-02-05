import pytest
import torch
from _dist_test_utils import get_device_counts
from _model_test_utils import MLP

import tensorrt_llm._torch.auto_deploy.distributed.common as dist
from tensorrt_llm._torch.auto_deploy.utils.benchmark import benchmark


def run_benchmark(rank, world_size):
    input_dim, hidden_dim, output_dim = 128, 4096, 128

    mod = MLP(input_dim, hidden_dim, output_dim).to("cuda").to(torch.float16)
    state_dict = [mod.state_dict()]
    dist.broadcast_object_list(state_dict, src=0)
    mod.load_state_dict(state_dict[0])
    x = torch.randn(2048, 128).to("cuda").half()

    benchmark(
        lambda: mod(x),
        num_runs=10,
    )


@pytest.mark.parametrize("device_count", get_device_counts())
def test_benchmark_torch_mod(device_count):
    dist.spawn_multiprocess_job(job=run_benchmark, size=device_count)
