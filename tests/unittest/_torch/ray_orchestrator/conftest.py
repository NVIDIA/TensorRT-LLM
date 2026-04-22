import os
import sys

import pytest

from tensorrt_llm._utils import mpi_disabled


def pytest_configure(config):
    if config.getoption("--run-ray"):
        os.environ["TLLM_DISABLE_MPI"] = "1"
        os.environ["TLLM_RAY_FORCE_LOCAL_CLUSTER"] = "1"
        os.environ["RAY_raylet_start_wait_time_s"] = "120"


run_ray_flag = "--run-ray" in sys.argv
if run_ray_flag:
    os.environ["TLLM_DISABLE_MPI"] = "1"
    os.environ["TLLM_RAY_FORCE_LOCAL_CLUSTER"] = "1"
    os.environ["RAY_raylet_start_wait_time_s"] = "120"


def pytest_collection_modifyitems(config, items):
    """Skip ray_orchestrator tests when MPI is not disabled.

    Uses hook instead of module-level pytest.skip() which is incompatible
    with conftest loading in pytest 8+.
    """
    if not mpi_disabled():
        skip_ray = pytest.mark.skip(
            reason=
            "Ray tests are only tested in Ray CI stage or with --run-ray flag")
        for item in items:
            if "ray_orchestrator" in item.nodeid:
                item.add_marker(skip_ray)
