import os
import threading

import pytest

from tensorrt_llm.llmapi import LlmArgs
from tensorrt_llm.llmapi import utils as llmapi_utils
from tensorrt_llm.llmapi.utils import (ApiStatusRegistry,
                                       _set_affinity_all_threads,
                                       generate_api_docs_as_docstring,
                                       get_executor_loop_cpus)

pytestmark = pytest.mark.cpu_only


def test_api_status_registry():

    @ApiStatusRegistry.set_api_status("beta")
    def _my_method(self, *args, **kwargs):
        pass

    assert ApiStatusRegistry.get_api_status(_my_method) == "beta"

    @ApiStatusRegistry.set_api_status("prototype")
    def _my_method(self, *args, **kwargs):
        pass

    # will always keep the first status, and the behaviour will be unknown if
    # one method is registered with a different status in different files.
    assert ApiStatusRegistry.get_api_status(_my_method) == "beta"

    class App:

        @ApiStatusRegistry.set_api_status("beta")
        def _my_method(self, *args, **kwargs):
            pass

    assert ApiStatusRegistry.get_api_status(App._my_method) == "beta"


def test_generate_api_docs_as_docstring():
    doc = generate_api_docs_as_docstring(LlmArgs)
    assert ":tag:`beta`" in doc, "the label is not generated"
    print(doc)


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="Linux only")
def test_set_affinity_all_threads_binds_existing_threads():
    all_cpus = sorted(os.sched_getaffinity(0))
    if len(all_cpus) < 2:
        pytest.skip("needs at least two CPUs")
    ready = threading.Event()
    release = threading.Event()
    seen = []

    def worker():
        ready.set()
        release.wait()
        seen.append(sorted(os.sched_getaffinity(0)))

    thread = threading.Thread(target=worker)
    thread.start()
    ready.wait()
    try:
        subset = all_cpus[:1]
        assert _set_affinity_all_threads(subset) >= 2
        release.set()
        thread.join()
        assert seen == [subset]
        assert sorted(os.sched_getaffinity(0)) == subset
    finally:
        release.set()
        _set_affinity_all_threads(all_cpus)


def test_get_executor_loop_cpus(monkeypatch):
    numa_cpus = {
        0: list(range(0, 32)),
        1: list(range(0, 32)),
        2: list(range(32, 64)),
        3: list(range(32, 64)),
    }
    monkeypatch.setattr(llmapi_utils, "get_numa_aware_cpu_affinity",
                        numa_cpus.__getitem__)
    monkeypatch.delenv("TLLM_EXECUTOR_LOOP_PIN", raising=False)
    assert get_executor_loop_cpus(0) == []

    monkeypatch.setenv("TLLM_EXECUTOR_LOOP_PIN", "1")
    assert get_executor_loop_cpus(0) == [16, 17]
    assert get_executor_loop_cpus(1) == [18, 19]
    assert get_executor_loop_cpus(2) == [48, 49]
    assert get_executor_loop_cpus(3) == [50, 51]

    monkeypatch.setenv("TLLM_EXECUTOR_LOOP_PIN_OFFSET", "4")
    monkeypatch.setenv("TLLM_EXECUTOR_LOOP_PIN_NCORES", "3")
    assert get_executor_loop_cpus(1) == [7, 8, 9]

    monkeypatch.setenv("TLLM_EXECUTOR_LOOP_PIN_OFFSET", "40")
    assert get_executor_loop_cpus(0) == []


class DelayedAssert:

    def __init__(self, store_stack: bool = False):
        self.assertions = []
        self.store_stack = store_stack

    def add(self, result: bool, msg: str):
        import traceback
        self.assertions.append(
            (bool(result), str(msg), traceback.format_stack()))

    def get_msg(self):
        ret = ['Some assertions failed:']
        for result, msg, stack in self.assertions:
            ret.append('\n'.join([
                f'Assert result: {result}', msg,
                ''.join(stack) if self.store_stack else ''
            ]))
        ret = '\n-----------------------------------------\n'.join(ret)
        ret = 'Some assertions failed:\n' + ret
        return ret

    def clear(self):
        self.assertions.clear()

    def assert_all(self):
        assert all(ret[0] for ret in self.assertions), self.get_msg()
        self.clear()
