import pytest

from tensorrt_llm._torch import virtual_memory
from tensorrt_llm._torch.pyexecutor import py_executor_creator


class _DummyKvCacheCreator:

    def __init__(self, *, fail_once: bool):
        self.fail_once = fail_once
        self.calls: list[tuple[dict, bool]] = []

    def build_managers(self, resources, estimating_kv_cache: bool):
        self.calls.append((resources, estimating_kv_cache))
        if self.fail_once and len(self.calls) == 1:
            raise RuntimeError("out of memory")


def test_build_kv_cache_managers_retries_final_sleep_path(monkeypatch):
    creator = _DummyKvCacheCreator(fail_once=True)
    resources = {"kv": "cache"}
    sleeps: list[float] = []

    monkeypatch.setattr(virtual_memory.torch.cuda, "synchronize",
                        lambda: None)
    monkeypatch.setattr(virtual_memory.torch.cuda, "empty_cache",
                        lambda: None)
    monkeypatch.setattr(virtual_memory.gc, "collect", lambda: 0)
    monkeypatch.setattr(virtual_memory.time, "sleep", sleeps.append)

    py_executor_creator._build_kv_cache_managers(
        creator,
        resources,
        estimating_kv_cache=False,
        enable_sleep=True,
    )

    assert creator.calls == [(resources, False), (resources, False)]
    assert sleeps == [1.0]


def test_build_kv_cache_managers_skips_retry_for_estimation(monkeypatch):
    creator = _DummyKvCacheCreator(fail_once=True)
    resources = {"kv": "cache"}
    monkeypatch.setattr(
        py_executor_creator,
        "run_with_oom_retry",
        lambda *args, **kwargs: pytest.fail("estimation path should not retry"),
    )

    with pytest.raises(RuntimeError, match="out of memory"):
        py_executor_creator._build_kv_cache_managers(
            creator,
            resources,
            estimating_kv_cache=True,
            enable_sleep=True,
        )

    assert creator.calls == [(resources, True)]
