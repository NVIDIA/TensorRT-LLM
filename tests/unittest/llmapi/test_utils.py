# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import errno
import os
import subprocess
import sys
import types

import pytest

import tensorrt_llm.llmapi.utils as llmapi_utils
from tensorrt_llm.llmapi import LlmArgs
from tensorrt_llm.llmapi.utils import (ApiStatusRegistry,
                                       _set_affinity_all_threads,
                                       configure_cpu_affinity,
                                       generate_api_docs_as_docstring)

_TASK_DIR = "/proc/self/task"

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


class _RecordingLogger:
    """Stands in for the module logger so tests can assert on what it emits."""

    def __init__(self):
        self.warnings = []
        self.infos = []

    def warning(self, msg):
        self.warnings.append(str(msg))

    def info(self, msg):
        self.infos.append(str(msg))


@pytest.fixture
def rec_logger(monkeypatch):
    log = _RecordingLogger()
    monkeypatch.setattr(llmapi_utils, "logger", log)
    return log


def _fake_procfs(monkeypatch, tids, setaffinity=None, listdir_error=None):
    """Make the helper walk `tids` without touching this process's affinity."""
    real_isdir, real_listdir = os.path.isdir, os.listdir

    def isdir(path):
        return True if path == _TASK_DIR else real_isdir(path)

    def listdir(path):
        if path != _TASK_DIR:
            return real_listdir(path)
        if listdir_error is not None:
            raise listdir_error
        return [str(tid) for tid in tids]

    monkeypatch.setattr(os.path, "isdir", isdir)
    monkeypatch.setattr(os, "listdir", listdir)
    monkeypatch.setattr(os,
                        "sched_setaffinity",
                        setaffinity or (lambda tid, cpus: None),
                        raising=False)


def _fake_psutil(monkeypatch, current, ncpus):
    """Stub psutil so `configure_cpu_affinity` sees `current` as the mask."""

    class _Proc:

        def __init__(self, *args):
            pass

        def cpu_affinity(self, cpus=None):
            return list(current)

    monkeypatch.setattr(
        llmapi_utils, "psutil",
        types.SimpleNamespace(Process=_Proc, cpu_count=lambda: ncpus))


def test_set_affinity_rejects_an_empty_cpu_list(rec_logger):
    # An empty mask is EINVAL for every thread.
    assert _set_affinity_all_threads([]) == (0, 0)
    assert len(rec_logger.warnings) == 1
    assert "empty" in rec_logger.warnings[0]


def test_set_affinity_binds_every_listed_thread(monkeypatch, rec_logger):
    bound_to = {}
    _fake_procfs(monkeypatch, [1, 2, 3],
                 lambda tid, cpus: bound_to.__setitem__(tid, list(cpus)))

    assert _set_affinity_all_threads([0, 1]) == (3, 3)
    assert bound_to == {1: [0, 1], 2: [0, 1], 3: [0, 1]}
    assert rec_logger.warnings == []


def test_set_affinity_ignores_threads_that_exited(monkeypatch, rec_logger):

    def setaffinity(tid, cpus):
        if tid == 2:
            raise ProcessLookupError(errno.ESRCH, "No such process")

    _fake_procfs(monkeypatch, [1, 2, 3], setaffinity)

    # Exiting mid-walk is expected: neither counted nor reported.
    assert _set_affinity_all_threads([0]) == (2, 2)
    assert rec_logger.warnings == []


def test_set_affinity_reports_partial_failure(monkeypatch, rec_logger):
    injected = {2: errno.EPERM, 3: errno.EINVAL}

    def setaffinity(tid, cpus):
        if tid in injected:
            raise OSError(injected[tid], "injected")

    _fake_procfs(monkeypatch, [1, 2, 3], setaffinity)

    assert _set_affinity_all_threads([0]) == (1, 3)
    assert len(rec_logger.warnings) == 1
    assert "2 of 3" in rec_logger.warnings[0]
    assert "EPERM" in rec_logger.warnings[0]
    assert "EINVAL" in rec_logger.warnings[0]


def test_set_affinity_survives_an_unreadable_task_dir(monkeypatch, rec_logger):
    _fake_procfs(monkeypatch, [1],
                 listdir_error=PermissionError(errno.EACCES, "Denied"))

    # A failure here must not abort worker start-up.
    assert _set_affinity_all_threads([0]) == (0, 0)
    assert len(rec_logger.warnings) == 1
    assert _TASK_DIR in rec_logger.warnings[0]


def test_configure_does_not_claim_success_when_nothing_was_bound(
        monkeypatch, rec_logger):
    _fake_psutil(monkeypatch, current=[0, 1, 2, 3], ncpus=4)
    monkeypatch.setattr(llmapi_utils, "get_numa_aware_cpu_affinity",
                        lambda device_id: [0, 1])
    monkeypatch.setattr(llmapi_utils, "_set_affinity_all_threads", lambda cpus:
                        (0, 4))
    monkeypatch.delenv("TLLM_NUMA_AWARE_WORKER_AFFINITY", raising=False)

    configure_cpu_affinity(0)

    # The mask is read back from the main thread only, so a total failure
    # must not be logged as success.
    assert rec_logger.infos == []
    assert any("could not set" in msg.lower() for msg in rec_logger.warnings)


_REAL_THREAD_PROBE = """
import os
import sys
import threading

from tensorrt_llm.llmapi.utils import _set_affinity_all_threads

all_cpus = sorted(os.sched_getaffinity(0))
if len(all_cpus) < 2:
    print("SKIP")
    sys.exit(0)

ready, release = threading.Event(), threading.Event()
seen = []


def worker():
    ready.set()
    release.wait(60)
    seen.append(sorted(os.sched_getaffinity(0)))


thread = threading.Thread(target=worker, daemon=True)
thread.start()
try:
    ready.wait(60)
    subset = all_cpus[:1]
    bound, attempted = _set_affinity_all_threads(subset)
finally:
    release.set()
thread.join(60)

assert bound == attempted, (bound, attempted)
assert bound >= 2, (bound, attempted)
assert seen == [subset], seen
assert sorted(os.sched_getaffinity(0)) == subset
print("OK")
"""


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="Linux only")
def test_set_affinity_binds_pre_existing_threads_for_real():
    # In a child: the helper would otherwise rebind the pytest process.
    probe = subprocess.run([sys.executable, "-c", _REAL_THREAD_PROBE],
                           capture_output=True,
                           text=True,
                           timeout=600)
    assert probe.returncode == 0, probe.stderr
    assert "OK" in probe.stdout or "SKIP" in probe.stdout, (probe.stdout,
                                                            probe.stderr)


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
