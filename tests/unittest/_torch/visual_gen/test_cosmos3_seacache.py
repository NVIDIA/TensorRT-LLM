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
"""SeaCache on the real Cosmos3-Nano pipeline: caching, request isolation and fallbacks."""

import gc
from collections.abc import Generator

import PIL.Image
import pytest
import torch
from utils.llm_data import get_checkpoint

from tensorrt_llm._torch.visual_gen.models.wan.vae_loader import TRTLLM_USE_DIFFUSER_VAE_ENV
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.visual_gen.args import (
    CompilationConfig,
    SeaCacheConfig,
    TorchCompileConfig,
    VisualGenArgs,
)

pytestmark = [
    pytest.mark.cosmos3,
    pytest.mark.integration,
    pytest.mark.high_cuda_memory,
    pytest.mark.usefixtures("disable_cosmos3_guardrails"),
]

PROMPT = "A serene mountain lake at sunrise with mist rising from the water."
NUM_STEPS = 8
SEED = 42
HEIGHT = 720
WIDTH = 1280
NUM_FRAMES = 9
GUIDANCE_SCALE = 6.0
# A threshold no step reaches makes the skipped steps depend only on the cap and the
# warm-up, so the pattern is fixed: steps 1, 2, 4, 5 cached; 0, 3, 6 and the last step
# computed.
SEACACHE = SeaCacheConfig(threshold=1e6, max_warmup_steps=0, max_continuous_cached_steps=2)
EXPECTED_CACHED_STEPS = [1, 2, 4, 5]
# With the default config the skipped steps depend on the SEA filter and the threshold on
# real latents. Cached steps of the default-config request below, recorded on a B200.
DEFAULT_CONFIG_STEPS = 35
DEFAULT_CONFIG_CACHED_STEPS = 13
DEFAULT_CONFIG_TOLERANCE = 1


class _InjectedFailure(RuntimeError):
    pass


def _load_pipeline(compile_enabled: bool, cache_config: SeaCacheConfig = SEACACHE) -> Generator:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    checkpoint = get_checkpoint("Cosmos3-Nano")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.delenv(TRTLLM_USE_DIFFUSER_VAE_ENV, raising=False)
        args = VisualGenArgs(
            model=checkpoint,
            torch_compile_config=TorchCompileConfig(enable=compile_enabled),
            compilation_config=CompilationConfig(
                resolutions=[(HEIGHT, WIDTH)], num_frames=[NUM_FRAMES]
            ),
            cache_config=cache_config,
        )
        pipeline = PipelineLoader(args).load(skip_warmup=not compile_enabled)
        yield pipeline
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()


# Class-scoped so the eager and compiled pipelines are never resident at the same time.
@pytest.fixture(scope="class")
def seacache_pipeline() -> Generator:
    yield from _load_pipeline(compile_enabled=False)


@pytest.fixture(scope="class")
def compiled_seacache_pipeline() -> Generator:
    yield from _load_pipeline(compile_enabled=True)


@pytest.fixture(scope="class")
def default_seacache_pipeline() -> Generator:
    yield from _load_pipeline(compile_enabled=False, cache_config=SeaCacheConfig())


def _generate(pipeline, *, image=None, num_frames=NUM_FRAMES, num_inference_steps=NUM_STEPS):
    return pipeline.forward(
        prompt=PROMPT,
        image=image,
        height=HEIGHT,
        width=WIDTH,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=GUIDANCE_SCALE,
        seed=SEED,
        frame_rate=24.0,
        use_guardrails=False,
    )


def _cached_steps(pipeline) -> list[int]:
    return pipeline.cache_accelerator.get_stats()["cached_step_indices"]


def _assert_video(video: torch.Tensor, num_frames: int = NUM_FRAMES) -> None:
    assert video.dtype == torch.uint8
    assert video.shape == (1, num_frames, HEIGHT, WIDTH, 3)
    assert video.float().std() > 0


def _assert_tensors_released(pipeline) -> None:
    state = pipeline.cache_accelerator._controller._state
    assert state.history == []
    assert state.previous_indicator is None


def _check_failed_request_recovers(pipeline, monkeypatch) -> None:
    """A request that fails part-way through denoising leaves no cache state behind, and
    the next request reproduces a successful reference exactly."""
    reference = _generate(pipeline)
    accelerator = pipeline.cache_accelerator
    controller = accelerator._controller
    original_step = controller.step
    attempted = []

    def failing_step(step, *args, **kwargs):
        attempted.append(step)
        if len(attempted) == 4:
            raise _InjectedFailure("injected failure")
        return original_step(step, *args, **kwargs)

    monkeypatch.setattr(controller, "step", failing_step)
    with pytest.raises(_InjectedFailure):
        _generate(pipeline)
    monkeypatch.undo()

    # Checked before any further request, which would overwrite this state anyway.
    assert attempted == [0, 1, 2, 3]
    assert accelerator._step is None
    assert accelerator._last_run_step is None
    assert accelerator._call_latents is None
    _assert_tensors_released(pipeline)

    after = _generate(pipeline)
    assert _cached_steps(pipeline) == EXPECTED_CACHED_STEPS
    assert torch.equal(reference.video, after.video)


class TestEager:
    def test_requests_are_isolated(self, seacache_pipeline) -> None:
        """Each request starts from a clean cache: a T2V request repeated after an I2V
        request and an uncached single-frame request reproduces the first one exactly
        and skips the same steps."""
        first = _generate(seacache_pipeline)
        first_cached = _cached_steps(seacache_pipeline)
        image = PIL.Image.new("RGB", (WIDTH, HEIGHT), color=(64, 128, 192))
        i2v = _generate(seacache_pipeline, image=image)
        i2v_cached = _cached_steps(seacache_pipeline)
        single_frame = _generate(seacache_pipeline, num_frames=1)
        single_frame_cached = _cached_steps(seacache_pipeline)
        again = _generate(seacache_pipeline)

        _assert_video(first.video)
        _assert_video(i2v.video)
        _assert_tensors_released(seacache_pipeline)
        assert first_cached == EXPECTED_CACHED_STEPS
        assert i2v_cached == EXPECTED_CACHED_STEPS
        assert single_frame.video.shape[1] == 1
        assert single_frame_cached == []
        assert _cached_steps(seacache_pipeline) == EXPECTED_CACHED_STEPS
        assert torch.equal(first.video, again.video)

    def test_failed_request_does_not_leak_state(self, seacache_pipeline, monkeypatch) -> None:
        _check_failed_request_recovers(seacache_pipeline, monkeypatch)


class TestCompiled:
    def test_failed_request_does_not_leak_state(
        self, compiled_seacache_pipeline, monkeypatch
    ) -> None:
        """Same recovery check with torch.compile on: the computed steps run the
        compiled generator blocks that SeaCache wraps."""
        blocks = compiled_seacache_pipeline.transformer.gen_layers
        assert all(isinstance(b, torch._dynamo.eval_frame.OptimizedModule) for b in blocks)
        # Compiled code is cached per code object, and only the generator stack uses
        # this block class, so this observes the warm-up compiling gen_layers.
        block_forward = type(blocks[0]._orig_mod).forward.__code__
        assert torch._dynamo.eval_frame._debug_get_cache_entry_list(block_forward), (
            "warm-up did not compile the generator blocks"
        )
        _check_failed_request_recovers(compiled_seacache_pipeline, monkeypatch)


class TestDefaultConfig:
    def test_cached_step_count_matches_the_recorded_value(self, default_seacache_pipeline) -> None:
        """At the default config, the number of steps skipped on real latents stays within
        a tolerance of the recorded value, so a change to the SEA filter, the threshold or
        the warm-up shows up here."""
        output = _generate(default_seacache_pipeline, num_inference_steps=DEFAULT_CONFIG_STEPS)
        _assert_video(output.video)
        stats = default_seacache_pipeline.cache_accelerator.get_stats()
        warmup = max(SeaCacheConfig().max_warmup_steps, 1)
        assert stats["total"] == DEFAULT_CONFIG_STEPS
        assert all(
            warmup <= step < DEFAULT_CONFIG_STEPS - 1 for step in stats["cached_step_indices"]
        )
        assert abs(stats["cached"] - DEFAULT_CONFIG_CACHED_STEPS) <= DEFAULT_CONFIG_TOLERANCE, stats
