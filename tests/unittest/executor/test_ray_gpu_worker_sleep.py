from contextlib import contextmanager

import tensorrt_llm.executor.ray_gpu_worker as ray_gpu_worker
from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType


class _DummyTorchLlmArgs:

    def __init__(self):
        self.sleep_config = object()


class _DummyEngine:

    def __init__(self):
        self.reset_calls = 0

    @contextmanager
    def control_action(self):
        yield

    def reset_prefix_cache(self):
        self.reset_calls += 1


def _make_worker(monkeypatch):
    monkeypatch.setattr(ray_gpu_worker, "TorchLlmArgs", _DummyTorchLlmArgs)
    worker = object.__new__(ray_gpu_worker.RayGPUWorker)
    worker.engine = _DummyEngine()
    worker.llm_args = _DummyTorchLlmArgs()
    return worker


def test_wakeup_resets_prefix_cache_for_kv_cache(monkeypatch):
    worker = _make_worker(monkeypatch)
    materialized: list[tuple[ExecutorMemoryType, ...]] = []

    monkeypatch.setattr(ray_gpu_worker.torch.cuda, "synchronize",
                        lambda: None)
    monkeypatch.setattr(ray_gpu_worker, "materialize_with_tag",
                        lambda *tags: materialized.append(tags) or 1)

    ray_gpu_worker.RayGPUWorker.wakeup(worker,
                                       [ExecutorMemoryType.KV_CACHE.value])

    assert materialized == [(ExecutorMemoryType.KV_CACHE,)]
    assert worker.engine.reset_calls == 1


def test_wakeup_skips_prefix_cache_reset_without_kv_cache(monkeypatch):
    worker = _make_worker(monkeypatch)

    monkeypatch.setattr(ray_gpu_worker.torch.cuda, "synchronize",
                        lambda: None)
    monkeypatch.setattr(ray_gpu_worker, "materialize_with_tag",
                        lambda *tags: 1)

    ray_gpu_worker.RayGPUWorker.wakeup(
        worker, [ExecutorMemoryType.MODEL_WEIGHTS_MAIN.value])

    assert worker.engine.reset_calls == 0
