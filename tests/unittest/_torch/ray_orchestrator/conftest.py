import os
import sys
from pathlib import Path

import pytest
from utils.cpp_paths import llm_root  # noqa: F401

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
def add_worker_extension_path(llm_root: Path):
    worker_extension_path = str(llm_root / "examples" / "llm-api" / "rlhf")
    original_python_path = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = os.pathsep.join(
        filter(None, [worker_extension_path, original_python_path]))
    yield
    os.environ['PYTHONPATH'] = original_python_path
