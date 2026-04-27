import pytest

from tensorrt_llm._torch import virtual_memory


class _RetryingManager:

    def __init__(self):
        self.calls = 0

    def materialize_with_tag(self, tag: str) -> int:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError(f"{tag} out of memory")
        return 1


def test_materialize_with_tag_retries_oom(monkeypatch):
    manager = _RetryingManager()
    sleeps: list[float] = []
    sync_calls: list[None] = []
    empty_cache_calls: list[None] = []
    gc_calls: list[int] = []

    monkeypatch.setattr(virtual_memory, "get_virtual_memory_manager",
                        lambda: manager)
    monkeypatch.setattr(virtual_memory.torch.cuda, "synchronize",
                        lambda: sync_calls.append(None))
    monkeypatch.setattr(virtual_memory.torch.cuda, "empty_cache",
                        lambda: empty_cache_calls.append(None))
    monkeypatch.setattr(virtual_memory.gc, "collect",
                        lambda: gc_calls.append(1) or 0)
    monkeypatch.setattr(virtual_memory.time, "sleep", sleeps.append)

    assert virtual_memory.materialize_with_tag("kv_cache") == 1
    assert manager.calls == 2
    assert sleeps == [1.0]
    assert len(sync_calls) == 1
    assert len(empty_cache_calls) == 1
    assert gc_calls == [1]


def test_materialize_with_tag_does_not_retry_non_oom(monkeypatch):
    class _FailingManager:

        def materialize_with_tag(self, tag: str) -> int:
            raise RuntimeError(f"{tag} permission denied")

    sleeps: list[float] = []
    monkeypatch.setattr(virtual_memory, "get_virtual_memory_manager",
                        lambda: _FailingManager())
    monkeypatch.setattr(virtual_memory.time, "sleep", sleeps.append)

    with pytest.raises(RuntimeError, match="permission denied"):
        virtual_memory.materialize_with_tag("kv_cache")

    assert not sleeps
