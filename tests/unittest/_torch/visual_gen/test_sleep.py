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

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
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


@pytest.mark.parametrize("operation", ["sleep", "wake_up"])
def test_generation_callback_rejected_while_sleep_drains(
    allocation_backend: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    manager = PipelineSleepManager("CPU", torch.device("cuda:0"))
    with manager.loading():
        pass
    entered, allow_callback, draining = Event(), Event(), Event()
    wait_for = manager._condition.wait_for

    def wait_for_generation(predicate: Callable[[], bool]) -> bool:
        draining.set()
        assert wait_for(predicate, timeout=10), "Sleep could not drain callback generation"
        return True

    monkeypatch.setattr(manager._condition, "wait_for", wait_for_generation)

    def generate() -> None:
        with manager.generation():
            entered.set()
            assert allow_callback.wait(10)
            with pytest.raises(RuntimeError, match="inside an active generation"):
                getattr(manager, operation)()

    with ThreadPoolExecutor(max_workers=2) as executor:
        generation = executor.submit(generate)
        try:
            assert entered.wait(10)
            sleeping = executor.submit(manager.sleep)
            assert draining.wait(10)
            assert not sleeping.done()
            allocation_backend.release.assert_not_called()
        finally:
            allow_callback.set()
        generation.result(timeout=15)
        sleeping.result(timeout=15)
    assert manager.is_sleeping
    allocation_backend.release.assert_called_once()
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

    transition_lock = manager._transition_lock

    class ObservedTransitionLock:
        def __enter__(self) -> None:
            if entered.is_set():
                second_entered.set()
            transition_lock.acquire()

        def __exit__(self, *_args) -> None:
            transition_lock.release()

    manager._transition_lock = ObservedTransitionLock()
    backend.side_effect = transition
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(getattr(manager, first_operation))
        try:
            assert entered.wait(10)
            second = executor.submit(getattr(manager, second_operation))
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


@pytest.fixture
def h3_loader(monkeypatch: pytest.MonkeyPatch) -> tuple:
    from tensorrt_llm._torch.visual_gen import pipeline_loader as loader_module
    from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3 import (
        MiniMaxH3Pipeline,
    )
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    config = DiffusionPipelineConfig(torch_compile={"enable": False})
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.pipeline_config = config
    pipeline.mapping = config.mapping
    pipeline._device = torch.device("cuda:0")
    pipeline._sleep_manager = None
    pipeline.transformer = None
    for name in (
        "load_weights",
        "load_standard_components",
        "post_load_weights",
        "_setup_runtime_lora",
        "initialize_offload_pipeline",
        "warmup",
    ):
        setattr(pipeline, name, MagicMock())
    pipeline.load_transformer_weights = MagicMock(return_value={})
    loader = PipelineLoader(VisualGenArgs(model="checkpoint"), device="cuda:0")
    monkeypatch.setattr(loader, "_resolve_checkpoint_dir", lambda path: path)
    monkeypatch.setattr(
        loader_module.AutoPipeline,
        "resolve",
        lambda path, options: SimpleNamespace(path=path, pipeline_config={}),
    )
    monkeypatch.setattr(loader, "_setup_visual_gen_mapping", lambda config: None)
    monkeypatch.setattr(DiffusionPipelineConfig, "from_pretrained", lambda *args, **kwargs: config)
    monkeypatch.setattr(loader_module.AutoPipeline, "from_config", lambda *args: pipeline)
    monkeypatch.setattr(loader, "_get_load_time_cpu_offload_modules", lambda pipeline: [])
    monkeypatch.setattr(loader, "_materialize_meta_tensors", MagicMock())
    return loader, pipeline


@pytest.mark.parametrize("skip_warmup", [False, True])
def test_loader_warmup_is_outside_persistent_pool(
    allocation_backend: SimpleNamespace, h3_loader: tuple, skip_warmup: bool
) -> None:
    loader, pipeline = h3_loader
    events = []

    class AllocationScope:
        def __enter__(self) -> object:
            events.append("enter")
            return allocation_backend.pool

        def __exit__(self, *_args) -> None:
            events.append("exit")

    allocation_backend.scope.side_effect = lambda *_args: AllocationScope()

    def loaded_stage(name: str) -> Callable:
        def record(*_args, **_kwargs) -> None:
            assert events[0] == "enter" and "exit" not in events
            events.append(name)

        return record

    loader._materialize_meta_tensors.side_effect = loaded_stage("materialize")
    pipeline.load_weights.side_effect = loaded_stage("weights")
    pipeline.load_standard_components.side_effect = loaded_stage("components")
    pipeline.post_load_weights.side_effect = loaded_stage("post_load")

    def warmup() -> None:
        pipeline._sleep_manager.ensure_awake()
        events.append("warmup")

    pipeline.warmup.side_effect = warmup
    assert (
        loader.load("checkpoint", skip_warmup=skip_warmup, sleep_restore_mode="PINNED") is pipeline
    )
    assert events == ["enter", "materialize", "weights", "components", "post_load", "exit"] + (
        [] if skip_warmup else ["warmup"]
    )
    pipeline._sleep_manager.ensure_awake()


@pytest.mark.parametrize("failure_stage", [None, "construction", "components"])
def test_sleep_loading_selects_loader_device_and_restores_previous_device(
    allocation_backend: SimpleNamespace,
    h3_loader: tuple,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str | None,
) -> None:
    from tensorrt_llm._torch.visual_gen import pipeline_loader as loader_module

    loader, pipeline = h3_loader
    loader.device = torch.device("cuda:3")
    pipeline._device = torch.device("cuda")
    current_device = 0
    events = []

    @contextmanager
    def device_context(device: torch.device) -> Iterator[None]:
        nonlocal current_device
        previous_device = current_device
        current_device = device.index
        try:
            yield
        finally:
            current_device = previous_device

    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current_device)

    def construct(*_args) -> object:
        assert torch.cuda.current_device() == 3
        allocation_backend.scope.assert_not_called()
        events.append("construction")
        if failure_stage == "construction":
            raise ValueError("construction failed")
        return pipeline

    def components(*_args, **_kwargs) -> None:
        assert torch.cuda.current_device() == 3
        events.append("components")
        if failure_stage == "components":
            raise ValueError("components failed")

    def warmup() -> None:
        assert torch.cuda.current_device() == 0
        pipeline._sleep_manager.ensure_awake()
        events.append("warmup")

    monkeypatch.setattr(loader_module.AutoPipeline, "from_config", construct)
    pipeline.load_standard_components.side_effect = components
    pipeline.warmup.side_effect = warmup
    if failure_stage is None:
        assert loader.load(sleep_restore_mode="CPU") is pipeline
        assert events == ["construction", "components", "warmup"]
        allocation_backend.scope.assert_called_once()
    else:
        with pytest.raises(ValueError, match=f"{failure_stage} failed"):
            loader.load(sleep_restore_mode="CPU")
        assert pipeline._sleep_manager is None
        pipeline.warmup.assert_not_called()
    assert torch.cuda.current_device() == 0


@pytest.mark.parametrize(
    "feature, message",
    [
        ("multi_gpu", "requires world_size=1"),
        ("parallel_config", "requires world_size=1"),
        ("visual_mapping", "requires world_size=1"),
        ("ulysses_config", "requires world_size=1"),
        ("cpu_offload", "cannot be combined with CPU offloading or CUDA graphs"),
        ("cuda_graph", "cannot be combined with CPU offloading or CUDA graphs"),
        ("cache", "does not yet support cache acceleration or runtime LoRA"),
        ("lora", "does not yet support cache acceleration or runtime LoRA"),
    ],
)
def test_h3_rejects_unsupported_combinations_before_capture(
    allocation_backend: SimpleNamespace, h3_loader: tuple, feature: str, message: str
) -> None:
    loader, pipeline = h3_loader
    config = pipeline.pipeline_config
    if feature == "multi_gpu":
        pipeline.mapping = SimpleNamespace(world_size=2)
    elif feature == "parallel_config":
        config.parallel.tp_size = 2
    elif feature == "visual_mapping":
        config.visual_gen_mapping = SimpleNamespace(world_size=2)
    elif feature == "ulysses_config":
        config.parallel.ulysses_size = 2
    elif feature == "cpu_offload":
        config.cpu_offload_config.enable = True
    elif feature == "cuda_graph":
        config.cuda_graph.enable = True
    elif feature == "cache":
        config.cache = object()
    else:
        config.runtime_lora = object()
    with pytest.raises(ValueError, match=message):
        loader.load("checkpoint", sleep_restore_mode="PINNED")
    loader._materialize_meta_tensors.assert_not_called()
    pipeline.load_transformer_weights.assert_not_called()
    allocation_backend.scope.assert_not_called()


def test_loader_rejects_other_models_before_materializing(
    allocation_backend: SimpleNamespace, h3_loader: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensorrt_llm._torch.visual_gen import pipeline_loader as loader_module
    from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline

    loader, _ = h3_loader
    pipeline = BasePipeline.__new__(BasePipeline)
    torch.nn.Module.__init__(pipeline)
    monkeypatch.setattr(
        loader_module.AutoPipeline,
        "from_config",
        lambda *args: pipeline,
    )
    with pytest.raises(
        ValueError, match="Pipeline sleep is currently not supported for this model"
    ):
        loader.load(sleep_restore_mode="CPU")
    loader._materialize_meta_tensors.assert_not_called()
    allocation_backend.scope.assert_not_called()


def test_loader_uses_model_hook_without_h3_dependency(
    allocation_backend: SimpleNamespace, h3_loader: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensorrt_llm._torch.visual_gen import pipeline_loader as loader_module
    from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline

    loader, original = h3_loader
    events = []

    class OtherSleepPipeline(BasePipeline):
        @contextmanager
        def _sleep_loading_context(self, restore_mode, *, device):
            assert restore_mode == "CPU"
            assert device == loader.device
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")

    pipeline = OtherSleepPipeline.__new__(OtherSleepPipeline)
    torch.nn.Module.__init__(pipeline)
    for name in (
        "load_transformer_weights",
        "load_weights",
        "load_standard_components",
        "post_load_weights",
        "_setup_runtime_lora",
        "initialize_offload_pipeline",
        "warmup",
    ):
        setattr(pipeline, name, getattr(original, name))
    pipeline.transformer = None
    monkeypatch.setattr(loader_module.AutoPipeline, "from_config", lambda *args: pipeline)
    loader._materialize_meta_tensors.side_effect = lambda *_args: events.append("materialize")
    pipeline.warmup.side_effect = lambda: events.append("warmup")
    assert loader.load(sleep_restore_mode="CPU") is pipeline
    assert events == ["enter", "materialize", "exit", "warmup"]
    allocation_backend.scope.assert_not_called()


@pytest.mark.parametrize("explicit_none", [False, True])
def test_default_loading_bypasses_model_sleep_hook(
    allocation_backend: SimpleNamespace,
    h3_loader: tuple,
    explicit_none: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader, pipeline = h3_loader
    device_context = MagicMock(side_effect=AssertionError("Sleep device context entered"))
    monkeypatch.setattr(torch.cuda, "device", device_context)
    pipeline._sleep_loading_context = MagicMock(side_effect=AssertionError("Sleep hook called"))
    options = {"sleep_restore_mode": None} if explicit_none else {}
    assert loader.load(skip_warmup=True, **options) is pipeline
    pipeline._sleep_loading_context.assert_not_called()
    loader._materialize_meta_tensors.assert_called_once()
    pipeline.load_standard_components.assert_called_once()
    assert pipeline._sleep_manager is None
    allocation_backend.scope.assert_not_called()
    device_context.assert_not_called()


def test_loader_failure_exits_capture_without_enabling_sleep(
    allocation_backend: SimpleNamespace, h3_loader: tuple
) -> None:
    loader, pipeline = h3_loader
    exited = []

    @contextmanager
    def scope(*_args):
        try:
            yield allocation_backend.pool
        finally:
            exited.append(True)

    allocation_backend.scope.side_effect = scope
    pipeline.load_standard_components.side_effect = ValueError("component load failed")
    with pytest.raises(ValueError, match="component load failed"):
        loader.load(sleep_restore_mode="PINNED")
    assert exited == [True]
    assert pipeline._sleep_manager is None
    pipeline.warmup.assert_not_called()


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
@pytest.mark.parametrize("references", [None, [object()]])
def test_h3_forward_holds_generation_scope(
    allocation_backend: SimpleNamespace, enabled: bool, raises: bool, references: list | None
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
        assert kwargs == dict(request, keyframes=None, keyframe_anchors=None, references=references)
        assert kwargs["references"] is references
        if raises:
            raise ValueError("generation failed")
        return output

    pipeline._forward = forward
    if raises:
        with pytest.raises(ValueError, match="generation failed"):
            pipeline.forward(**request, references=references)
    else:
        assert pipeline.forward(**request, references=references) is output
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
@pytest.mark.parametrize("mapping_kind", ["llm", "visual_gen", "parallel"])
def test_h3_sleep_controls_require_single_rank(
    operation: str, world_size: int, mapping_kind: str
) -> None:
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3 import (
        MiniMaxH3Pipeline,
    )

    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.mapping = SimpleNamespace(world_size=world_size if mapping_kind == "llm" else 1)
    pipeline.pipeline_config = SimpleNamespace(
        visual_gen_mapping=SimpleNamespace(
            world_size=world_size if mapping_kind == "visual_gen" else 1
        ),
        parallel=SimpleNamespace(n_workers=world_size if mapping_kind == "parallel" else 1),
    )
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
