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
"""Heun sampling for MiniMax-H3 and the shared ``extra_params["sampler"]`` contract.

Scheduler math is checked against diffusers' Euler ``MiniMaxH3Scheduler``, an
independent Heun reference, and the closed-form probability-flow solution for
Gaussian data. No checkpoint, CPU only.

Run:
    pytest tests/unittest/_torch/visual_gen/test_minimax_h3_heun_scheduler.py -v
"""

import os
from types import SimpleNamespace

os.environ["TLLM_DISABLE_MPI"] = "1"

import pytest
import torch
from diffusers import MiniMaxH3Scheduler

from tensorrt_llm._torch.visual_gen.models.minimax_h3.heun_scheduler import MiniMaxH3HeunScheduler
from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.samplers import heun_timesteps
from tensorrt_llm.visual_gen.params import VisualGenParams, validate_visual_gen_params


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# MiniMaxH3Pipeline's video and audio shifts.
_SHIFTS = (MiniMaxH3Pipeline.VIDEO_SCHEDULER_SHIFT, MiniMaxH3Pipeline.AUDIO_SCHEDULER_SHIFT)

# Gaussian data N(mu, std^2): the probability-flow ODE of x_sigma = (1 - sigma) * x0 + sigma * noise
# maps x(1) to x(sigma) = (1 - sigma) * mu + x(1) * sqrt((1 - sigma)^2 * std^2 + sigma^2).
_MU, _STD = 0.7, 0.5


def _gaussian_velocity(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Exact data-ward velocity ``v = -dx/dsigma`` (MiniMax-H3 convention) for Gaussian data."""
    sigma = sigma.to(x.dtype)
    scale = torch.sqrt((1 - sigma) ** 2 * _STD**2 + sigma**2)
    dscale = (sigma - (1 - sigma) * _STD**2) / scale
    z = (x - (1 - sigma) * _MU) / scale
    return _MU - z * dscale


def _run(scheduler: MiniMaxH3Scheduler, sample: torch.Tensor) -> torch.Tensor:
    """Evaluate the model, then step, once per timestep (``BasePipeline.denoise`` order)."""
    for timestep in scheduler.timesteps:
        velocity = _gaussian_velocity(sample, 1 - timestep)
        (sample,) = scheduler.step(velocity, timestep, sample, return_dict=False)
    return sample


def _reference_heun(sigmas: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    """k-diffusion ``sample_heun`` (no churn) with MiniMax-H3's ``x0 = x + (1 - t) * v``."""
    timesteps = 1.0 - sigmas

    def denoise(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Like the scheduler: the model sees 1 - t in t's dtype, x0 uses 1 - t in x's dtype.
        return x + (1 - t.to(x.dtype)) * _gaussian_velocity(x, 1 - t)

    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i].double(), sigmas[i + 1].double()
        derivative = (sample - denoise(sample, timesteps[i])) / sigma
        proposal = sample + (sigma_next - sigma) * derivative
        if sigma_next == 0:
            sample = proposal
            continue
        next_derivative = (proposal - denoise(proposal, timesteps[i + 1])) / sigma_next
        sample = sample + (sigma_next - sigma) * (derivative + next_derivative) / 2
    return sample


def _schedulers(shift: float, num_inference_steps: int):
    euler = MiniMaxH3Scheduler(shift=shift)
    euler.set_timesteps(num_inference_steps)
    heun = MiniMaxH3HeunScheduler(shift=shift)
    heun.set_timesteps(num_inference_steps)
    return euler, heun


def _initial_noise(dtype=torch.float64) -> torch.Tensor:
    return torch.randn(4, 16, generator=torch.Generator().manual_seed(0), dtype=dtype)


@pytest.mark.parametrize("shift", _SHIFTS)
@pytest.mark.parametrize("num_inference_steps", [2, 3, 8, 28])
def test_heun_schedule_takes_2s_minus_1_evaluations(shift, num_inference_steps):
    euler, heun = _schedulers(shift, num_inference_steps)
    steps = len(euler.timesteps)
    assert len(heun.timesteps) == 2 * steps - 1
    assert torch.equal(heun.timesteps, heun_timesteps(euler.timesteps))
    assert heun.num_inference_steps == euler.num_inference_steps == steps
    assert torch.equal(heun.sigmas, euler.sigmas)


def test_video_and_audio_heun_schedules_stay_in_lockstep():
    video = MiniMaxH3HeunScheduler(shift=_SHIFTS[0])
    audio = MiniMaxH3HeunScheduler(shift=_SHIFTS[1])
    video.set_timesteps(28)
    audio.set_timesteps(28)
    assert len(video.timesteps) == len(audio.timesteps) == 53


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_heun_proposals_and_final_step_match_euler_bit_for_bit(dtype):
    """Every first evaluation of a step is the parent's Euler step, exactly."""
    _, heun = _schedulers(_SHIFTS[0], 8)
    steps = len(heun.sigmas) - 1
    sample = _initial_noise(dtype)
    for index, timestep in enumerate(heun.timesteps):
        velocity = _gaussian_velocity(sample.float(), 1 - timestep).to(dtype)
        (out,) = heun.step(velocity, timestep, sample, return_dict=False)
        if index % 2 == 0:
            euler = MiniMaxH3Scheduler(shift=_SHIFTS[0])
            euler.set_timesteps(8)
            euler.set_begin_index(index // 2)
            (expected,) = euler.step(velocity, timestep, sample, return_dict=False)
            assert torch.equal(out, expected), f"step {index // 2} of {steps}"
        sample = out


def test_single_step_heun_is_euler():
    euler, heun = _schedulers(_SHIFTS[0], 2)
    sample = _initial_noise()
    assert torch.equal(_run(heun, sample), _run(euler, sample))


@pytest.mark.parametrize("shift", _SHIFTS)
def test_heun_matches_reference_implementation(shift):
    _, heun = _schedulers(shift, 28)
    sample = _initial_noise()
    torch.testing.assert_close(
        _run(heun, sample), _reference_heun(heun.sigmas, sample), rtol=1e-10, atol=1e-10
    )


def test_heun_converges_at_second_order():
    """Halving the step size cuts the error ~4x for Heun and ~2x for Euler."""
    sample = _initial_noise()
    exact = _MU + sample * _STD
    errors = {"euler": [], "heun": []}
    # shift=1 keeps the grid uniform; 9/17/33 grid points are 8/16/32 steps.
    for num_inference_steps in (9, 17, 33):
        euler, heun = _schedulers(1.0, num_inference_steps)
        errors["euler"].append((_run(euler, sample) - exact).abs().max().item())
        errors["heun"].append((_run(heun, sample) - exact).abs().max().item())
    for coarse, fine in zip(errors["heun"], errors["heun"][1:]):
        assert coarse / fine > 3.5, errors
    for coarse, fine in zip(errors["euler"], errors["euler"][1:]):
        assert coarse / fine < 2.5, errors
    assert errors["heun"][-1] < errors["euler"][-1] / 10, errors


def test_heun_is_unaffected_by_callers_writing_latents_in_place():
    """Diffusers' MiniMax-H3 loop writes each step's result back into the sample buffer."""
    _, heun = _schedulers(_SHIFTS[0], 8)
    expected = _run(heun, _initial_noise())

    heun.set_timesteps(8)
    sample = _initial_noise()
    for timestep in heun.timesteps:
        velocity = _gaussian_velocity(sample, 1 - timestep)
        sample[:] = heun.step(velocity, timestep, sample, return_dict=False)[0]
    assert torch.equal(sample, expected)


def test_heun_state_resets_and_rejects_steps_past_the_schedule():
    _, heun = _schedulers(_SHIFTS[0], 8)
    sample = _initial_noise()
    first = _run(heun, sample)

    # A schedule abandoned between a step's two evaluations leaves no state behind.
    heun.set_timesteps(8)
    timestep = heun.timesteps[0]
    heun.step(_gaussian_velocity(sample, 1 - timestep), timestep, sample)
    heun.set_timesteps(8)
    assert torch.equal(_run(heun, sample), first)

    with pytest.raises(ValueError, match="past the end of the schedule"):
        heun.step(_gaussian_velocity(sample, 1 - timestep), timestep, sample)


def _bare_pipeline(workflow: str = "fl2va") -> MiniMaxH3Pipeline:
    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline.workflow = workflow
    pipeline.scheduler = MiniMaxH3Scheduler(shift=_SHIFTS[0])
    pipeline.audio_scheduler = MiniMaxH3Scheduler(shift=_SHIFTS[1])
    return pipeline


@pytest.mark.parametrize("workflow", ["fl2va", "ref2va"])
def test_minimax_h3_declares_sampler_extra_param(workflow):
    specs = _bare_pipeline(workflow).extra_param_specs
    assert specs["sampler"].type == "Literal['euler', 'heun']"
    assert specs["sampler"].default == "euler"
    assert ("reference_order" in specs) == (workflow == "ref2va")

    validate_visual_gen_params(
        VisualGenParams(extra_params={"sampler": "heun"}),
        declared_defaults=None,
        extra_param_specs=specs,
    )
    with pytest.raises(ValueError, match=r"extra_params\['sampler'\] expected one of"):
        validate_visual_gen_params(
            VisualGenParams(extra_params={"sampler": "dpm"}),
            declared_defaults=None,
            extra_param_specs=specs,
        )


def test_pipelines_without_sampler_choice_reject_the_key():
    assert BasePipeline.supported_samplers.fget(None) == ()
    specs = BasePipeline.sampler_param_specs(SimpleNamespace(supported_samplers=()))
    assert specs == {}
    with pytest.raises(ValueError, match=r"Unknown extra_params \['sampler'\]"):
        validate_visual_gen_params(
            VisualGenParams(extra_params={"sampler": "heun"}),
            declared_defaults=None,
            extra_param_specs=specs,
        )


def test_minimax_h3_request_schedulers():
    pipeline = _bare_pipeline()
    assert pipeline._request_schedulers("euler") == (pipeline.scheduler, pipeline.audio_scheduler)

    video, audio = pipeline._request_schedulers("heun")
    assert isinstance(video, MiniMaxH3HeunScheduler) and isinstance(audio, MiniMaxH3HeunScheduler)
    assert (video.shift, audio.shift) == _SHIFTS
    assert pipeline._request_schedulers("heun")[0] is not video

    with pytest.raises(ValueError, match="Unsupported MiniMax-H3 sampler"):
        pipeline._request_schedulers("dpm")


def test_minimax_h3_infer_forwards_requested_sampler():
    pipeline = _bare_pipeline()
    pipeline.forward = lambda **kwargs: kwargs
    params = VisualGenParams(
        seed=0,
        height=128,
        width=128,
        num_frames=124,
        frame_rate=24.0,
        num_inference_steps=28,
        extra_params={"sampler": "heun"},
    )
    request = SimpleNamespace(
        prompt=["a fox"],
        params=params,
        prepared_inputs={"keyframes": [], "keyframe_anchors": ()},
    )
    assert pipeline.infer(request)["sampler"] == "heun"
    params.extra_params = {"sampler": None}
    assert pipeline.infer(request)["sampler"] == "euler"
