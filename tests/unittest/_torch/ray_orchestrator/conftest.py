import os
import sys

import pytest
import ray

from tensorrt_llm._utils import mpi_disabled


def pytest_configure(config):
    if config.getoption("--run-ray"):
        os.environ["TLLM_DISABLE_MPI"] = "1"
        os.environ["TLLM_RAY_FORCE_LOCAL_CLUSTER"] = "1"


run_ray_flag = "--run-ray" in sys.argv
if run_ray_flag:
    os.environ["TLLM_DISABLE_MPI"] = "1"
    os.environ["TLLM_RAY_FORCE_LOCAL_CLUSTER"] = "1"

if not mpi_disabled():
    pytest.skip(
        "Ray tests are only tested in Ray CI stage or with --run-ray flag",
        allow_module_level=True)


@pytest.fixture(scope="function")
def setup_ray_cluster():
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
        yield
    finally:
        if ray.is_initialized():
            ray.shutdown()
