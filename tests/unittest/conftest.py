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
from functools import partial
from typing import Any

import _pytest.outcomes
import pytest
import torch
import tqdm
from mpi4py.futures import MPIPoolExecutor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integration.defs import test_list_parser


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
        if not isinstance(e, _pytest.outcomes.Skipped):
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
    parser.addoption(
        "--waives-file",
        "-S",
        action="store",
        default=None,
        help=
        "Specify a file containing a list of waives, one per line. After filtering collected tests, Pytest will "
        "apply the waive state specified by this file to the set of tests to be run.",
    )


def apply_waives_ut(waives_file, items: list[pytest.Item], config):
    """Apply waives based on the waive state specified by the given waives_file."""

    # Corrections don't make sense for the waives file as it specifies global negative
    # filters that may or may not be applicable to the current platform (i.e., the test names
    # being waived may not be generated on the current platform).
    try:
        parse_test_list_lines_bak = test_list_parser.parse_test_list_lines
        test_list_parser.parse_test_list_lines = partial(
            test_list_parser.parse_test_list_lines, convert_unittest=False)
        ret = test_list_parser.parse_and_validate_test_list(
            waives_file,
            config,
            items,
            check_for_corrections=False,
        )
    finally:
        test_list_parser.parse_test_list_lines = parse_test_list_lines_bak
    if not ret:
        return
    _, test_name_to_marker_dict = ret

    filtered_dict = {}
    for waiver in test_name_to_marker_dict.keys():
        if "unittest/" not in waiver:
            continue
        elif "unittest/unittest/" in waiver:
            filtered_dict[waiver.replace(
                "unittest/unittest/",
                "unittest/")] = test_name_to_marker_dict[waiver]
        else:
            filtered_dict[waiver] = test_name_to_marker_dict[waiver]

    # Fuzzy match is supported in the following order:
    # 1. exact match
    # 2. remove parameterization part in square brackets and try again (function level)
    # 3. remove the last part after '::' and try again, until no '::' left (file level)
    # Note: directory level match is not supported.
    def match_waiver(id: str):
        if id in filtered_dict:
            return filtered_dict[id]
        if id.endswith("]"):
            id = id.split("[")[0]
            if id in filtered_dict:
                return filtered_dict[id]
        while "::" in id:
            id = id.rsplit("::", 1)[0]
            if id in filtered_dict:
                return filtered_dict[id]

        return None

    # For each item in the list, apply waives if a waive entry exists
    for item in items:
        waiver = match_waiver(item.nodeid)
        if waiver:
            marker, reason, _ = waiver
            if marker:
                mark_func = getattr(pytest.mark, marker.lower())
                mark = mark_func(reason=reason)
                item.add_marker(mark)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_collection_modifyitems(session, config, items):
    test_prefix = config.getoption("--test-prefix")
    waives_file = config.getoption("--waives-file")

    yield

    if test_prefix:
        # Override the internal nodeid of each item to contain the correct test prefix.
        # This is needed for reporting to correctly process the test name in order to bucket
        # it into the appropriate test suite.
        for item in items:
            item._nodeid = f"{test_prefix}/{item._nodeid}"

    if waives_file:
        apply_waives_ut(waives_file, items, config)
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
    if session.config.getoption("--run-ray"):
        os.environ["TLLM_DISABLE_MPI"] = "1"
        os.environ["TLLM_RAY_FORCE_LOCAL_CLUSTER"] = "1"

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


def pytest_generate_tests(metafunc: pytest.Metafunc):
    if metafunc.definition.get_closest_marker('mpi_ray_parity'):
        run_ray = metafunc.config.getoption("--run-ray") or os.environ.get(
            "TLLM_RUN_RAY_TESTS") == "1"
        if run_ray:
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
                kwargs["orchestrator_type"] = "ray"
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
