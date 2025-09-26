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
import sys
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


def pytest_sessionstart(session):
    # To counter TransformerEngine v2.3's lazy_compile deferral,
    # which will cause Pytest thinks there's a thread leakage.
    import torch._inductor.async_compile  # noqa: F401


@pytest.fixture(autouse=True)
def cuda_error_early_quit(capfd):
    """
    Fixture to handle CUDA error.

    CUDA error are usually persistent that requires restart process to recover.
    Immediately stop the current worker when CUDA error occurred.
    It will then be restarted by the master process.
    """
    if torch.cuda.is_available() and os.environ.get("PYTEST_XDIST_WORKER",
                                                    None):
        try:
            yield
            torch.cuda.synchronize()
        except RuntimeError as e:
            msg = str(e)
            if 'CUDA error:' in msg:
                with capfd.disabled():
                    traceback.print_exception(e, file=sys.stderr)
                    print("CUDA Error occurred, worker must quit now",
                          file=sys.stderr)
                os._exit(1)
            raise
    else:
        yield


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
