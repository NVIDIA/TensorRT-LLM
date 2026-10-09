# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.visual_gen import sleep as sleep_module
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm._torch.visual_gen.sleep import PipelineSleepManager

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def allocation_backend(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    backend = SimpleNamespace(
        pool=object(),
        release=MagicMock(return_value=2),
        restore=MagicMock(return_value=2),
        synchronize=MagicMock(),
        empty_cache=MagicMock(),
    )
    backend.scope = MagicMock(side_effect=lambda *_args: nullcontext(backend.pool))
    monkeypatch.setattr(sleep_module.virtual_memory, "scope", backend.scope)
    monkeypatch.setattr(sleep_module.virtual_memory, "release_with_tag", backend.release)
    monkeypatch.setattr(sleep_module.virtual_memory, "materialize_with_tag", backend.restore)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", backend.synchronize)
    monkeypatch.setattr(torch.cuda, "empty_cache", backend.empty_cache)
    return backend


@pytest.mark.parametrize("mode", ["CPU", "PINNED"])
def test_sleep_cycle_and_idempotence(allocation_backend: SimpleNamespace, mode: str) -> None:
    manager = PipelineSleepManager(mode, torch.device("cuda:0"))
    with manager.loading():
        pass
    manager.ensure_awake()
    assert manager._pool is allocation_backend.pool
    tag, restore_mode = allocation_backend.scope.call_args.args
    assert restore_mode == sleep_module.virtual_memory.RestoreMode[mode]

    for cycle in range(3):
        manager.sleep()
        manager.sleep()
        assert manager.is_sleeping
        with pytest.raises(RuntimeError, match="asleep"):
            manager.ensure_awake()
        assert allocation_backend.release.call_count == cycle + 1
        allocation_backend.release.assert_called_with(tag)
        manager.wake_up()
        manager.wake_up()
        manager.ensure_awake()
        assert not manager.is_sleeping
        assert allocation_backend.restore.call_count == cycle + 1
        allocation_backend.restore.assert_called_with(tag)


def test_pipeline_tags_are_isolated(allocation_backend: SimpleNamespace) -> None:
    managers = [PipelineSleepManager("CPU", torch.device("cuda:0")) for _ in range(2)]
    for manager in managers:
        with manager.loading():
            pass
        manager.sleep()
    tags = [call.args[0] for call in allocation_backend.release.call_args_list]
    assert tags[0] != tags[1]


@pytest.mark.parametrize("mode", ["NONE", "MEMSET", "pinned", ""])
def test_rejects_destructive_backing_modes(mode: str) -> None:
    with pytest.raises(ValueError, match="CPU or PINNED"):
        PipelineSleepManager(mode, torch.device("cuda:0"))


def test_load_failure_is_not_usable(allocation_backend: SimpleNamespace) -> None:
    manager = PipelineSleepManager("CPU", torch.device("cuda:0"))
    with pytest.raises(ValueError, match="load failed"):
        with manager.loading():
            raise ValueError("load failed")
    with pytest.raises(RuntimeError, match="failed"):
        manager.ensure_awake()
    with pytest.raises(RuntimeError, match="failed"):
        manager.wake_up()
    allocation_backend.restore.assert_not_called()


@pytest.mark.parametrize("operation", ["sleep", "wake_up"])
@pytest.mark.parametrize("failure", [RuntimeError("allocator failed"), KeyboardInterrupt()])
def test_interrupted_transition_fails_closed(
    allocation_backend: SimpleNamespace, operation: str, failure: BaseException
) -> None:
    manager = PipelineSleepManager("PINNED", torch.device("cuda:0"))
    with manager.loading():
        pass
    if operation == "wake_up":
        manager.sleep()
        allocation_backend.restore.side_effect = failure
    else:
        allocation_backend.release.side_effect = failure
    with pytest.raises(type(failure)):
        getattr(manager, operation)()
    with pytest.raises(RuntimeError, match="failed"):
        manager.ensure_awake()
    with pytest.raises(RuntimeError, match="failed"):
        manager.wake_up()


@pytest.mark.parametrize("operation", ["sleep", "wake_up"])
def test_empty_capture_is_not_reported_as_success(
    allocation_backend: SimpleNamespace, operation: str
) -> None:
    manager = PipelineSleepManager("CPU", torch.device("cuda:0"))
    with manager.loading():
        pass
    if operation == "wake_up":
        manager.sleep()
        allocation_backend.restore.return_value = 0
    else:
        allocation_backend.release.return_value = 0
    with pytest.raises(RuntimeError, match="sleep-managed"):
        getattr(manager, operation)()
    with pytest.raises(RuntimeError, match="failed"):
        manager.ensure_awake()


def test_partial_restore_is_not_reported_as_success(allocation_backend: SimpleNamespace) -> None:
    manager = PipelineSleepManager("CPU", torch.device("cuda:0"))
    with manager.loading():
        pass
    manager.sleep()
    allocation_backend.restore.return_value = 1
    with pytest.raises(RuntimeError, match="expected 2"):
        manager.wake_up()
    with pytest.raises(RuntimeError, match="failed"):
        manager.ensure_awake()


@pytest.mark.parametrize("failure", [None, ValueError, KeyboardInterrupt])
def test_sleep_drains_generation_and_closes_admission(
    allocation_backend: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, failure: type | None
) -> None:
    manager = PipelineSleepManager("PINNED", torch.device("cuda:0"))
    with manager.loading():
        pass
    entered, finish, draining = Event(), Event(), Event()
    wait_for = manager._condition.wait_for

    def wait_for_generation(predicate: Callable[[], bool]) -> bool:
        draining.set()
        assert wait_for(predicate, timeout=10)
        return True

    monkeypatch.setattr(manager._condition, "wait_for", wait_for_generation)

    def generate() -> None:
        with manager.generation():
            entered.set()
            assert finish.wait(10)
            if failure is not None:
                raise failure("generation interrupted")

    with ThreadPoolExecutor(max_workers=2) as executor:
        generation = executor.submit(generate)
        try:
            assert entered.wait(10)
            sleeping = executor.submit(manager.sleep)
            assert draining.wait(10)
            assert not sleeping.done()
            allocation_backend.synchronize.assert_not_called()
            allocation_backend.release.assert_not_called()
            with pytest.raises(RuntimeError, match="sleeping"):
                with manager.generation():
                    pytest.fail("Generation admitted while sleep was pending")
        finally:
            finish.set()
        if failure is None:
            generation.result(timeout=10)
        else:
            with pytest.raises(failure, match="generation interrupted"):
                generation.result(timeout=10)
        sleeping.result(timeout=10)
    assert manager.is_sleeping
    allocation_backend.release.assert_called_once()
    manager.wake_up()
    with manager.generation():
        pass


@pytest.mark.parametrize("operation", ["sleep", "wake_up"])
def test_generation_callback_cannot_deadlock_transition(
    allocation_backend: SimpleNamespace, operation: str
) -> None:
    manager = PipelineSleepManager("CPU", torch.device("cuda:0"))
    with manager.loading():
        pass
    with manager.generation():
        with pytest.raises(RuntimeError, match="inside an active generation"):
            getattr(manager, operation)()
    manager.ensure_awake()
    allocation_backend.release.assert_not_called()
    allocation_backend.restore.assert_not_called()


def test_overlapping_generation_rejected_and_managers_independent(
    allocation_backend: SimpleNamespace,
) -> None:
    first = PipelineSleepManager("CPU", torch.device("cuda:0"))
    second = PipelineSleepManager("CPU", torch.device("cuda:0"))
    for manager in (first, second):
        with manager.loading():
            pass
    with first.generation():
        with ThreadPoolExecutor(max_workers=1) as executor:

            def generate() -> None:
                with first.generation():
                    pytest.fail("Overlapping generation admitted")

            with pytest.raises(RuntimeError, match="Concurrent generation"):
                executor.submit(generate).result(timeout=10)
        with second.generation():
            pass
        second.sleep()
    with first.generation():
        pass


@pytest.mark.parametrize(
    "first_operation, second_operation",
    [
        ("sleep", "sleep"),
        ("sleep", "wake_up"),
        ("wake_up", "wake_up"),
        ("wake_up", "sleep"),
    ],
)
def test_transitions_are_serialized(
    allocation_backend: SimpleNamespace, first_operation: str, second_operation: str
) -> None:
    manager = PipelineSleepManager("CPU", torch.device("cuda:0"))
    with manager.loading():
        pass
    if first_operation == "wake_up":
        manager.sleep()
    allocation_backend.release.reset_mock()
    allocation_backend.restore.reset_mock()
    entered, finish, second_entered = Event(), Event(), Event()
    backend = (
        allocation_backend.release if first_operation == "sleep" else allocation_backend.restore
    )

    def transition(_tag: str) -> int:
        entered.set()
        assert finish.wait(10)
        return 2

    def second_transition() -> None:
        second_entered.set()
        getattr(manager, second_operation)()

    backend.side_effect = transition
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(getattr(manager, first_operation))
        try:
            assert entered.wait(10)
            second = executor.submit(second_transition)
            assert second_entered.wait(10)
            assert not first.done()
            assert not second.done()
            with pytest.raises(RuntimeError, match="sleeping|waking"):
                with manager.generation():
                    pytest.fail("Generation admitted during transition")
            assert (
                allocation_backend.release.call_count + allocation_backend.restore.call_count == 1
            )
        finally:
            finish.set()
        first.result(timeout=10)
        second.result(timeout=10)
    assert manager.is_sleeping == (second_operation == "sleep")
    assert allocation_backend.release.call_count + allocation_backend.restore.call_count == (
        1 if first_operation == second_operation else 2
    )


def test_interrupted_drain_fails_closed(
    allocation_backend: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = PipelineSleepManager("CPU", torch.device("cuda:0"))
    with manager.loading():
        pass
    monkeypatch.setattr(manager._condition, "wait_for", MagicMock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        manager.sleep()
    with pytest.raises(RuntimeError, match="failed"):
        with manager.generation():
            pytest.fail("Generation admitted after interrupted drain")
    allocation_backend.release.assert_not_called()


def test_loader_default_path_does_not_capture(allocation_backend: SimpleNamespace) -> None:
    loader = PipelineLoader.__new__(PipelineLoader)
    pipeline = object()
    loader._load = MagicMock(return_value=pipeline)
    assert loader.load("checkpoint", skip_warmup=True) is pipeline
    loader._load.assert_called_once_with("checkpoint", True, None)
    allocation_backend.scope.assert_not_called()


def test_loader_warmup_is_outside_persistent_pool(allocation_backend: SimpleNamespace) -> None:
    loader = PipelineLoader.__new__(PipelineLoader)
    loader.args = SimpleNamespace(
        parallel_config=SimpleNamespace(n_workers=1),
        cpu_offload_config=SimpleNamespace(enable=False),
        cuda_graph_config=SimpleNamespace(enable=False),
        cache_config=None,
        runtime_lora_config=None,
    )
    loader.device = torch.device("cuda:0")
    pipeline = SimpleNamespace(warmup=MagicMock())
    loader._load = MagicMock(return_value=pipeline)
    events = []

    class AllocationScope:
        def __enter__(self) -> object:
            events.append("enter")
            return allocation_backend.pool

        def __exit__(self, *_args) -> None:
            events.append("exit")

    allocation_backend.scope.side_effect = lambda *_args: AllocationScope()
    pipeline.warmup.side_effect = lambda: events.append("warmup")
    assert loader.load("checkpoint", sleep_restore_mode="PINNED") is pipeline
    loader._load.assert_called_once_with("checkpoint", True, None, sleep_enabled=True)
    assert events == ["enter", "exit", "warmup"]
    pipeline._sleep_manager.ensure_awake()


@pytest.mark.parametrize("feature", ["multi_gpu", "cpu_offload", "cuda_graph", "cache", "lora"])
def test_loader_rejects_unsupported_combinations_before_capture(
    allocation_backend: SimpleNamespace, feature: str
) -> None:
    loader = PipelineLoader.__new__(PipelineLoader)
    loader.args = SimpleNamespace(
        parallel_config=SimpleNamespace(n_workers=2 if feature == "multi_gpu" else 1),
        cpu_offload_config=SimpleNamespace(enable=feature == "cpu_offload"),
        cuda_graph_config=SimpleNamespace(enable=feature == "cuda_graph"),
        cache_config=object() if feature == "cache" else None,
        runtime_lora_config=object() if feature == "lora" else None,
    )
    loader._load = MagicMock()
    with pytest.raises(ValueError):
        loader.load("checkpoint", sleep_restore_mode="PINNED")
    loader._load.assert_not_called()
    allocation_backend.scope.assert_not_called()


def test_loader_rejects_other_models_before_materializing(
    allocation_backend: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensorrt_llm._torch.visual_gen import pipeline_loader as loader_module
    from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    config = DiffusionPipelineConfig()
    loader = PipelineLoader(VisualGenArgs(model="checkpoint"))
    monkeypatch.setattr(loader, "_resolve_checkpoint_dir", lambda path: path)
    monkeypatch.setattr(loader, "_resolve_pipeline_config", lambda path: {})
    monkeypatch.setattr(loader, "_setup_visual_gen_mapping", lambda config: None)
    monkeypatch.setattr(DiffusionPipelineConfig, "from_pretrained", lambda *args, **kwargs: config)
    monkeypatch.setattr(
        loader_module.AutoPipeline,
        "from_config",
        lambda *args: SimpleNamespace(),
    )
    materialize = MagicMock()
    monkeypatch.setattr(loader, "_materialize_meta_tensors", materialize)
    with pytest.raises(ValueError, match="only for MiniMax-H3"):
        loader.load(sleep_restore_mode="CPU")
    materialize.assert_not_called()


def test_sleeping_h3_rejects_forward_before_cuda_work() -> None:
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3 import (
        MiniMaxH3Pipeline,
    )

    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline._sleep_manager = MagicMock()
    pipeline._sleep_manager.generation.side_effect = RuntimeError("Pipeline is asleep")
    with pytest.raises(RuntimeError, match="asleep"):
        pipeline.forward(
            prompt="A spacecraft above a moon.",
            seed=42,
            height=544,
            width=960,
            num_frames=124,
            frame_rate=24,
            num_inference_steps=28,
        )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_h3_forward_holds_generation_scope(
    allocation_backend: SimpleNamespace, enabled: bool, raises: bool
) -> None:
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3 import (
        MiniMaxH3Pipeline,
    )

    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    manager = PipelineSleepManager("CPU", torch.device("cuda:0"))
    with manager.loading():
        pass
    pipeline._sleep_manager = manager if enabled else None
    request = dict(
        prompt="A spacecraft.",
        seed=42,
        height=128,
        width=128,
        num_frames=124,
        frame_rate=24.0,
        num_inference_steps=4,
    )
    output = object()

    def forward(**kwargs) -> object:
        assert torch.is_inference_mode_enabled()
        assert (manager._generation_thread is not None) == enabled
        assert kwargs == dict(request, keyframes=None, keyframe_anchors=None)
        if raises:
            raise ValueError("generation failed")
        return output

    pipeline._forward = forward
    if raises:
        with pytest.raises(ValueError, match="generation failed"):
            pipeline.forward(**request)
    else:
        assert pipeline.forward(**request) is output
    assert manager._generation_thread is None
    manager.sleep()


def test_sleep_api_belongs_only_to_h3() -> None:
    from tensorrt_llm._torch.visual_gen.models.flux import FluxPipeline
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3 import (
        MiniMaxH3Pipeline,
    )
    from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline
    from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline

    for name in ("sleep", "wake_up", "is_sleeping", "_sleep_manager"):
        for pipeline_cls in (BasePipeline, WanPipeline, FluxPipeline):
            assert not hasattr(pipeline_cls, name)
    for name in ("sleep", "wake_up", "is_sleeping"):
        assert name in MiniMaxH3Pipeline.__dict__


@pytest.mark.parametrize("operation", ["sleep", "wake_up"])
@pytest.mark.parametrize("world_size", [1, 2])
def test_h3_sleep_controls_require_single_rank(operation: str, world_size: int) -> None:
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3 import (
        MiniMaxH3Pipeline,
    )

    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.mapping = SimpleNamespace(world_size=world_size)
    pipeline._sleep_manager = MagicMock()
    if world_size == 1:
        getattr(pipeline, operation)()
        getattr(pipeline._sleep_manager, operation).assert_called_once_with()
    else:
        with pytest.raises(ValueError, match="requires world_size=1"):
            getattr(pipeline, operation)()
        pipeline._sleep_manager.sleep.assert_not_called()
        pipeline._sleep_manager.wake_up.assert_not_called()


@pytest.mark.parametrize("operation", ["sleep", "wake_up"])
def test_h3_without_opt_in_rejects_sleep_controls(operation: str) -> None:
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3 import (
        MiniMaxH3Pipeline,
    )

    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline._sleep_manager = None
    assert not pipeline.is_sleeping
    with pytest.raises(RuntimeError, match="not enabled during loading"):
        getattr(pipeline, operation)()
