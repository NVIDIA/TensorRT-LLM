"""Unit tests for custom dist ops."""

import pytest
import torch
from _dist_test_utils import get_device_counts

from tensorrt_llm._torch.auto_deploy.distributed.common import spawn_multiprocess_job


def _run_all_reduce_test(rank, world_size):
    x = torch.ones(10, 10).to("cuda")
    y = torch.ops.auto_deploy.torch_dist_all_reduce(x)

    assert torch.equal(x * world_size, y)


def _run_all_gather_test(rank, world_size):
    x = torch.ones(10, 10).to("cuda")
    y = torch.ops.auto_deploy.torch_dist_all_gather(x)

    assert torch.sum(y) == world_size * torch.sum(x)
    assert y.shape == (world_size * x.shape[0], *x.shape[1:])


@pytest.mark.parametrize("device_count", get_device_counts())
def test_all_reduce(device_count):
    spawn_multiprocess_job(job=_run_all_reduce_test, size=device_count)


@pytest.mark.parametrize("device_count", get_device_counts())
def test_all_gather(device_count):
    spawn_multiprocess_job(job=_run_all_gather_test, size=device_count)
