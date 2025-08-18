# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# # Force resource release after test

import os
import traceback
from typing import Any

import pytest
import torch
import tqdm
from mpi4py.futures import MPIPoolExecutor


def pytest_configure(config):
    # avoid thread leak of tqdm's TMonitor
    tqdm.tqdm.monitor_interval = 0


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    yield

    import sys
    for m in sys.modules:
        if m == 'torch' or m.startswith('torch.'):
            import gc
            import os

            import torch
            worker_count = int(os.environ.get('PYTEST_XDIST_WORKER_COUNT', 1))

            if (torch.cuda.memory_reserved(0) + torch.cuda.memory_allocated(0)
                ) >= (torch.cuda.get_device_properties(0).total_memory //
                      worker_count) * 0.9:
                gc.collect()
                print("torch.cuda.memory_allocated: %fGB" %
                      (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
                print("torch.cuda.memory_reserved: %fGB" %
                      (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
                print("torch.cuda.max_memory_reserved: %fGB" %
                      (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

                torch.cuda.empty_cache()
            break


# Logging exceptions to stdout to prevent them from being masked by
# pytest-threadleak complaints.
@pytest.hookimpl(wrapper=True)
def pytest_pyfunc_call(pyfuncitem) -> Any:
    try:
        return (yield)
    # NB: _pytest.outcomes.OutcomeException subclasses BaseException
    except BaseException as e:
        print(f"TEST RAISED ERROR: {e}")
        traceback.print_exception(e)
        raise


def pytest_addoption(parser):
    parser.addoption(
        "--test-prefix",
        "-P",
        action="store",
        default=None,
        help=
        "Prepend a prefix to the test names. Useful for distinguishing different test runs in a test report."
    )
    parser.addoption(
        "--run-ray",
        action="store_true",
        default=False,
        help="Run Ray-marked tests (by default they are skipped).",
    )


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_collection_modifyitems(session, config, items):
    test_prefix = config.getoption("--test-prefix")

    yield

    if test_prefix:
        # Override the internal nodeid of each item to contain the correct test prefix.
        # This is needed for reporting to correctly process the test name in order to bucket
        # it into the appropriate test suite.
        for item in items:
            item._nodeid = f"{test_prefix}/{item._nodeid}"

    # Ray tests are disabled by default
    run_ray = config.getoption("--run-ray") or os.environ.get(
        "TLLM_RUN_RAY_TESTS") == "1"
    if not run_ray:
        skip_marker = pytest.mark.skip(
            reason=
            "Ray tests skipped; pass --run-ray or set TLLM_RUN_RAY_TESTS=1")
        for item in items:
            if "ray" in item.keywords:
                item.add_marker(skip_marker)


def pytest_sessionstart(session):
    # To counter TransformerEngine v2.3's lazy_compile deferral,
    # which will cause Pytest thinks there's a thread leakage.
    import torch._inductor.async_compile  # noqa: F401


@pytest.fixture(autouse=True)
def torch_empty_cache() -> None:
    """
    Automatically empty the torch CUDA cache before each test, to reduce risk of OOM errors.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="module", params=[2, 4, 8])
def mpi_pool_executor(request):
    """
    Start an MPIPoolExecutor with `request.param` workers.
    """
    num_workers = request.param
    with MPIPoolExecutor(num_workers) as executor:
        # make the number of workers visible to tests
        setattr(executor, "num_workers", num_workers)
        yield executor


def pytest_generate_tests(metafunc: pytest.Metafunc):
    if metafunc.definition.get_closest_marker('mpi_ray_parity'):
        run_ray = metafunc.config.getoption("--run-ray") or os.environ.get(
            "TLLM_RUN_RAY_TESTS") == "1"
        if metafunc.definition.get_closest_marker('mpi_ray_parity') and run_ray:
            metafunc.parametrize(
                'ray_mode',
                [
                    pytest.param('ray', id='ray', marks=pytest.mark.ray),
                ],
                indirect=True,
            )


@pytest.fixture
def ray_mode(request):
    return getattr(request, 'param', 'mpi')


@pytest.fixture(autouse=True)
def _maybe_force_ray(request, monkeypatch, ray_mode):
    """
    Patch the LLM class (torch only) to use Ray executor.
    """
    if 'mpi_ray_parity' not in request.node.keywords or ray_mode != 'ray':
        return

    def wrap_llm(cls):

        class LLMProxy(cls):

            def __init__(self, *args, **kwargs):
                kwargs["executor_type"] = "ray"
                super().__init__(*args, **kwargs)

        return LLMProxy

    test_mod = request.node.module

    # Only patch the torch LLM class
    if hasattr(test_mod, 'LLM'):
        try:
            from tensorrt_llm._tensorrt_engine import LLM as LLM_legacy
            is_trtllm_backend = (test_mod.LLM is LLM_legacy)
        except Exception:
            is_trtllm_backend = False
        if not is_trtllm_backend:
            monkeypatch.setattr(test_mod,
                                'LLM',
                                wrap_llm(test_mod.LLM),
                                raising=False)
    if hasattr(test_mod, 'LLM_torch'):
        monkeypatch.setattr(test_mod,
                            'LLM_torch',
                            wrap_llm(test_mod.LLM_torch),
                            raising=False)

    try:
        import tensorrt_llm.llmapi.llm as llm_mod
        monkeypatch.setattr(llm_mod,
                            'LLM',
                            wrap_llm(llm_mod.LLM),
                            raising=False)
    except Exception:
        pass
