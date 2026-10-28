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
from tensorrt_llm.visual_gen.args import SeaCacheConfig, TorchCompileConfig, VisualGenArgs

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
# A threshold no step reaches makes the skipped steps depend only on the cap, so
# the pattern is fixed: steps 1, 2, 4, 5 cached; 0, 3, 6 and the last step computed.
SEACACHE = SeaCacheConfig(threshold=1e6, max_consecutive_cached_steps=2)
EXPECTED_CACHED_STEPS = [1, 2, 4, 5]


class _InjectedFailure(RuntimeError):
    pass


@pytest.fixture(scope="module")
def seacache_pipeline() -> Generator:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    checkpoint = get_checkpoint("Cosmos3-Nano")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.delenv(TRTLLM_USE_DIFFUSER_VAE_ENV, raising=False)
        args = VisualGenArgs(
            model=checkpoint,
            torch_compile_config=TorchCompileConfig(enable=False),
            cache_config=SEACACHE,
        )
        pipeline = PipelineLoader(args).load(skip_warmup=True)
        yield pipeline
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()


def _generate(pipeline, *, image=None, num_frames=NUM_FRAMES):
    return pipeline.forward(
        prompt=PROMPT,
        image=image,
        height=HEIGHT,
        width=WIDTH,
        num_frames=num_frames,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        seed=SEED,
        frame_rate=24.0,
        use_guardrails=False,
    )


def _cached_steps(pipeline) -> list[int]:
    return pipeline.cache_accelerator.get_stats()["cached_step_indices"]


def test_requests_are_isolated(seacache_pipeline) -> None:
    """Each request starts from a clean cache: a T2V request repeated after an I2V
    request and an uncached single-frame request reproduces the first one exactly
    and skips the same steps."""
    first = _generate(seacache_pipeline)
    first_cached = _cached_steps(seacache_pipeline)
    image = PIL.Image.new("RGB", (WIDTH, HEIGHT), color=(64, 128, 192))
    _generate(seacache_pipeline, image=image)
    i2v_cached = _cached_steps(seacache_pipeline)
    single_frame = _generate(seacache_pipeline, num_frames=1)
    single_frame_cached = _cached_steps(seacache_pipeline)
    again = _generate(seacache_pipeline)

    assert first_cached == EXPECTED_CACHED_STEPS
    assert i2v_cached == EXPECTED_CACHED_STEPS
    assert single_frame.video.shape[1] == 1
    assert single_frame_cached == []
    assert _cached_steps(seacache_pipeline) == EXPECTED_CACHED_STEPS
    assert torch.equal(first.video, again.video)


def test_failed_request_does_not_leak_state(seacache_pipeline, monkeypatch) -> None:
    """A request that fails part-way through denoising leaves no cache state behind."""
    reference = _generate(seacache_pipeline)
    accelerator = seacache_pipeline.cache_accelerator
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
        _generate(seacache_pipeline)
    monkeypatch.undo()

    # Checked before any further request, which would overwrite this state anyway.
    assert attempted == [0, 1, 2, 3]
    assert accelerator._step is None
    assert accelerator._last_run_step is None
    assert accelerator._call_latents is None

    after = _generate(seacache_pipeline)
    assert _cached_steps(seacache_pipeline) == EXPECTED_CACHED_STEPS
    assert torch.equal(reference.video, after.video)
