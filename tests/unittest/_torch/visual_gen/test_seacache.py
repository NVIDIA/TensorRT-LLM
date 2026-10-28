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
"""Unit tests for SeaCache (CPU-only, no model weights needed)."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tensorrt_llm._torch.visual_gen.cache import SeaCacheAccelerator, seacache
from tensorrt_llm._torch.visual_gen.cache.seacache import (
    SeaCacheController,
    apply_sea_filter,
    predict_stack_output,
)
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm.visual_gen.args import SeaCacheConfig

pytestmark = pytest.mark.cpu_only

LATENT_SHAPE = (2, 4, 3, 6, 8)  # [B, C, T, H, W]
STACK_SHAPE = (2, 10, 16)  # [B, S, D]


def _reference_sea_filter(latent: torch.Tensor, sigma: float, power_exp: float = 3.0):
    """Dense (non-separable) evaluation of the SEA gain on a [T, H, W, C] latent."""
    latent = latent.double()
    grids = torch.meshgrid(
        *[torch.fft.fftfreq(n, dtype=torch.float64) for n in latent.shape[:3]], indexing="ij"
    )
    s, n = 1.0 - sigma, sigma
    gain = torch.ones(latent.shape[:3], dtype=torch.float64)
    for freqs in grids:
        power = 1.0 / (freqs.abs() ** power_exp + 1e-16)
        gain = gain * s * power / (s**2 * power + n**2 + 1e-16)
    gain = gain / gain.mean()
    spectrum = torch.fft.fftn(latent, dim=(0, 1, 2))
    return torch.fft.ifftn(spectrum * gain[..., None], dim=(0, 1, 2)).real


@pytest.mark.parametrize("sigma", [0.9, 0.5, 0.1])
def test_sea_filter_matches_dense_gain(sigma):
    latent = torch.randn(3, 6, 8, 4, generator=torch.Generator().manual_seed(0))
    out = apply_sea_filter(latent, sigma)
    torch.testing.assert_close(
        out.double(), _reference_sea_filter(latent, sigma), rtol=1e-4, atol=1e-4
    )


def test_sea_filter_keeps_shape_and_dtype():
    latent = torch.randn(3, 6, 8, 4).to(torch.bfloat16)
    out = apply_sea_filter(latent, 0.5)
    assert out.shape == latent.shape and out.dtype == torch.bfloat16


def test_predicted_residual_is_linear_in_step():
    history = [(2, torch.full((3,), 1.0)), (4, torch.full((3,), 2.0))]
    stack_input = torch.full((3,), 10.0)
    torch.testing.assert_close(
        predict_stack_output(history, 5, stack_input), torch.full((3,), 12.5)
    )
    torch.testing.assert_close(
        predict_stack_output(history[-1:], 5, stack_input), torch.full((3,), 12.0)
    )


def _slow_latents(step: int) -> torch.Tensor:
    """Latents that change by ~1e-4 relative per step, far below any threshold used here."""
    base = torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(1))
    return base * (1.0 + 1e-4 * step)


def _run_trajectory(controller, num_steps, latents_fn=_slow_latents, residual_fn=None):
    """Drive the controller like a denoising loop; returns (outputs, computed steps, true outputs).

    The noise level is held fixed so the indicator distance reflects only the latent change.
    """
    residual_fn = residual_fn or (lambda step: torch.full(STACK_SHAPE, float(step)))
    stack_input = torch.randn(STACK_SHAPE, generator=torch.Generator().manual_seed(2))
    controller.reset(num_steps)
    computed, outputs, true_outputs = [], [], []
    for step in range(num_steps):

        def run_stack(x, step=step):
            computed.append(step)
            return x + residual_fn(step)

        outputs.append(controller.step(step, 0.5, latents_fn(step), stack_input, run_stack))
        true_outputs.append(stack_input + residual_fn(step))
    return outputs, computed, true_outputs


def test_first_and_last_steps_always_compute_and_cap_limits_consecutive_skips():
    controller = SeaCacheController(threshold=0.25, max_consecutive_cached_steps=2)
    _, computed, _ = _run_trajectory(controller, num_steps=8)
    assert computed == [0, 3, 6, 7]
    stats = controller.get_stats()
    assert stats["cached_step_indices"] == [1, 2, 4, 5]
    assert stats["total"] == 8 and stats["cached"] == 4
    assert stats["hit_rate"] == pytest.approx(0.5)


def test_zero_cap_means_unlimited_consecutive_skips():
    controller = SeaCacheController(threshold=0.25, max_consecutive_cached_steps=0)
    _, computed, _ = _run_trajectory(controller, num_steps=8)
    assert computed == [0, 7]


def test_large_indicator_change_computes_every_step():
    controller = SeaCacheController(threshold=0.25, max_consecutive_cached_steps=2)
    _, computed, _ = _run_trajectory(
        controller,
        num_steps=6,
        latents_fn=lambda step: torch.randn(
            LATENT_SHAPE, generator=torch.Generator().manual_seed(100 + step)
        ),
    )
    assert computed == list(range(6))


def test_accumulated_distance_forces_compute():
    # Each step changes the latents by ~10% relative: the second skip candidate
    # pushes the accumulated distance past the threshold.
    controller = SeaCacheController(threshold=0.15, max_consecutive_cached_steps=0)
    base = torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(3))
    _, computed, _ = _run_trajectory(
        controller, num_steps=6, latents_fn=lambda step: base * (1.1**step)
    )
    assert computed == [0, 2, 4, 5]


def test_skipped_steps_extrapolate_linear_residuals_exactly():
    controller = SeaCacheController(threshold=0.25, max_consecutive_cached_steps=2)
    outputs, computed, true_outputs = _run_trajectory(controller, num_steps=8)
    assert computed == [0, 3, 6, 7]
    # Steps 1-2 only have one computed residual (reused as is); from step 4 on the
    # two-point linear prediction reproduces a residual that is linear in the step.
    torch.testing.assert_close(outputs[1], true_outputs[0])
    for step in (4, 5):
        torch.testing.assert_close(outputs[step], true_outputs[step])


def test_indicator_reference_advances_on_skipped_steps():
    # The latents jump by 10% once, then stay constant. Distances are measured
    # against the previous step (cached or not), so only the jump accumulates and
    # every middle step is cached. Measuring against the last computed step
    # instead would accumulate 0.1 per step and force a compute at step 3.
    controller = SeaCacheController(threshold=0.25, max_consecutive_cached_steps=0)
    base = torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(4))
    _, computed, _ = _run_trajectory(
        controller, num_steps=6, latents_fn=lambda step: base if step == 0 else base * 1.1
    )
    assert computed == [0, 5]


def test_non_adjacent_step_restarts_trajectory():
    controller = SeaCacheController(threshold=0.25, max_consecutive_cached_steps=2)
    controller.reset(10)
    stack_input = torch.zeros(STACK_SHAPE)
    computed = []

    def run_stack(x):
        computed.append(True)
        return x + 1.0

    for step in (0, 1, 5, 6):
        controller.step(step, 0.5, _slow_latents(step), stack_input, run_stack)
    # Step 1 reuses step 0; step 5 is not adjacent, so it starts a new trajectory.
    assert controller.get_stats()["cached_step_indices"] == [1, 6]
    assert len(computed) == 2


def test_reset_clears_state_between_generations():
    controller = SeaCacheController(threshold=0.25, max_consecutive_cached_steps=2)
    _run_trajectory(controller, num_steps=8)
    _, computed, _ = _run_trajectory(controller, num_steps=4)
    assert computed == [0, 3]
    assert controller.get_stats()["total"] == 4


# ---------------------------------------------------------------------------
# SeaCacheAccelerator wiring
# ---------------------------------------------------------------------------


class _FakeTransformer(nn.Module):
    """Mimics the Cosmos3 transformer's generator-stack seam."""

    def __init__(self):
        super().__init__()
        self.stack_calls = 0
        self.step_precision_controller = None
        self.denoising_step = None

    def set_denoising_step(self, step_index, num_steps):
        self.denoising_step = step_index

    def reset_denoising_step(self):
        self.denoising_step = None

    def _run_gen_stack(self, hidden_gen, freqs_gen, *, timestep, layer_text_lens, offload_context):
        self.stack_calls += 1
        return hidden_gen + 1.0

    def forward(self, hidden_states, timestep=None, audio_latents=None):
        hidden_gen = hidden_states.flatten(2).transpose(1, 2)
        return self._run_gen_stack(
            hidden_gen,
            (None, None),
            timestep=timestep,
            layer_text_lens=None,
            offload_context=lambda name: nullcontext(),
        )


def _fake_pipeline(num_steps=8, **overrides):
    pipeline = SimpleNamespace(
        transformer=_FakeTransformer(),
        scheduler=SimpleNamespace(sigmas=torch.full((num_steps + 1,), 0.5)),
        sampling=SimpleNamespace(is_distilled=False),
        world_size=1,
        pipeline_config=SimpleNamespace(cuda_graph=SimpleNamespace(enable=False)),
        offloader=SimpleNamespace(stages=lambda: []),
    )
    for key, value in overrides.items():
        setattr(pipeline, key, value)
    return pipeline


def _denoise(pipeline, accelerator, num_steps, latents_fn=_slow_latents, **forward_kwargs):
    """Mimic BasePipeline.denoise + the Cosmos3 forward_fn step selection."""
    accelerator.refresh(num_steps)
    try:
        for step in range(num_steps):
            pipeline.transformer.set_denoising_step(step_index=step, num_steps=num_steps)
            pipeline.transformer(hidden_states=latents_fn(step), **forward_kwargs)
    finally:
        pipeline.transformer.reset_denoising_step()


def test_accelerator_skips_generator_stack_on_cached_steps():
    pipeline = _fake_pipeline()
    accelerator = SeaCacheAccelerator(pipeline, SeaCacheConfig())
    accelerator.wrap()
    assert accelerator.is_enabled()
    _denoise(pipeline, accelerator, 8)
    assert pipeline.transformer.stack_calls == 4
    assert accelerator.get_stats()["cached_step_indices"] == [1, 2, 4, 5]


def test_accelerator_reads_sigma_of_the_current_step(monkeypatch):
    pipeline = _fake_pipeline()
    pipeline.scheduler.sigmas = torch.linspace(0.9, 0.0, 9)
    observed = []
    original = seacache.build_indicator

    def recording_build_indicator(latents, sigma):
        observed.append(sigma)
        return original(latents, sigma)

    monkeypatch.setattr(seacache, "build_indicator", recording_build_indicator)
    accelerator = SeaCacheAccelerator(pipeline, SeaCacheConfig())
    accelerator.wrap()
    _denoise(pipeline, accelerator, 8)
    assert observed == pytest.approx(pipeline.scheduler.sigmas[:8].tolist())


def test_accelerator_forwards_step_hooks():
    pipeline = _fake_pipeline()
    accelerator = SeaCacheAccelerator(pipeline, SeaCacheConfig())
    accelerator.wrap()
    pipeline.transformer.set_denoising_step(step_index=3, num_steps=8)
    assert vars(pipeline.transformer)["denoising_step"] == 3
    pipeline.transformer.reset_denoising_step()
    assert vars(pipeline.transformer)["denoising_step"] is None


def test_accelerator_runs_calls_outside_a_denoising_loop_uncached():
    # E.g. the Cosmos3 transfer loop: no refresh() and no set_denoising_step(),
    # even right after a cached request.
    pipeline = _fake_pipeline()
    accelerator = SeaCacheAccelerator(pipeline, SeaCacheConfig())
    accelerator.wrap()
    _denoise(pipeline, accelerator, 8)
    calls_after_first_request = pipeline.transformer.stack_calls
    for step in range(8):
        pipeline.transformer(hidden_states=_slow_latents(step))
    assert pipeline.transformer.stack_calls == calls_after_first_request + 8


def test_accelerator_runs_single_frame_latents_uncached():
    pipeline = _fake_pipeline()
    accelerator = SeaCacheAccelerator(pipeline, SeaCacheConfig())
    accelerator.wrap()
    _denoise(pipeline, accelerator, 8, latents_fn=lambda step: _slow_latents(step)[:, :, :1])
    assert pipeline.transformer.stack_calls == 8


def test_accelerator_runs_unsupported_streams_uncached():
    pipeline = _fake_pipeline()
    accelerator = SeaCacheAccelerator(pipeline, SeaCacheConfig())
    accelerator.wrap()
    _denoise(pipeline, accelerator, 8, audio_latents=torch.zeros(1))
    assert pipeline.transformer.stack_calls == 8
    assert accelerator.get_stats()["total"] == 0


def test_accelerator_unwrap_restores_original_stack():
    pipeline = _fake_pipeline()
    accelerator = SeaCacheAccelerator(pipeline, SeaCacheConfig())
    accelerator.wrap()
    accelerator.unwrap()
    assert not accelerator.is_enabled()
    for attr in ("_run_gen_stack", "set_denoising_step", "reset_denoising_step"):
        assert attr not in vars(pipeline.transformer)
    _denoise(pipeline, accelerator, 8)
    assert pipeline.transformer.stack_calls == 8


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"transformer": nn.Linear(2, 2)}, "not supported"),
        ({"sampling": SimpleNamespace(is_distilled=True)}, "distilled"),
        ({"scheduler": SimpleNamespace(timesteps=None)}, "sigmas"),
        ({"world_size": 2}, "single-GPU"),
        (
            {"pipeline_config": SimpleNamespace(cuda_graph=SimpleNamespace(enable=True))},
            "CUDA graphs",
        ),
        ({"offloader": SimpleNamespace(stages=lambda: ["generator"])}, "offloading"),
    ],
)
def test_accelerator_rejects_unsupported_setups(overrides, match):
    accelerator = SeaCacheAccelerator(_fake_pipeline(**overrides), SeaCacheConfig())
    with pytest.raises(NotImplementedError, match=match):
        accelerator.wrap()
    assert not accelerator.is_enabled()


def test_accelerator_rejects_step_precision_policy():
    pipeline = _fake_pipeline()
    pipeline.transformer.step_precision_controller = object()
    with pytest.raises(NotImplementedError, match="precision policy"):
        SeaCacheAccelerator(pipeline, SeaCacheConfig()).wrap()


def test_pipeline_setup_installs_seacache_accelerator():
    pipeline = _fake_pipeline()
    pipeline.cache_accelerator = None
    pipeline.pipeline_config.cache_backend = "seacache"
    pipeline.pipeline_config.seacache = SeaCacheConfig(threshold=0.1)
    BasePipeline._setup_cache_acceleration(pipeline)
    assert isinstance(pipeline.cache_accelerator, SeaCacheAccelerator)
    assert pipeline.cache_accelerator.is_enabled()
