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
"""DMD denoising loop math tests for WanDMDPipeline.

Tests _denoise in isolation: no checkpoint, no text encoder, no VAE.
A stub pipeline (object.__new__) with a fake transformer supplies controlled
inputs so the DMD formulas can be verified numerically.

GPU required (tensor ops + nvtx_range decorator).

Run:
    pytest tests/unittest/_torch/visual_gen/test_fastwan_dmd_math.py -v -s
"""

import os
import unittest.mock as mock

os.environ["TLLM_DISABLE_MPI"] = "1"

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.wan.pipeline_fastwan import WanDMDPipeline


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PATCH_SIZE = (1, 2, 2)

# Small latent (B=1, C=16, T=3, H=4, W=4)
# nf=3, nh=4//2=2, nw=4//2=2  →  12 patches per sample
_LATENT_SHAPE = (1, 16, 3, 4, 4)
_NUM_PATCHES = 3 * 2 * 2  # 12

# With pred_noise=0 and noise=0, re-noising reduces to:
#   latents_{i+1} = (1 - sigma_next) * latents_i
# So final output = (1-sigma_1)*(1-sigma_2)*initial  =  0.243*0.478*initial  ≈  0.1162
_SIGMA_1 = WanDMDPipeline.DMD_TIMESTEPS[1] / WanDMDPipeline.NUM_TRAIN_TIMESTEPS  # 0.757
_SIGMA_2 = WanDMDPipeline.DMD_TIMESTEPS[2] / WanDMDPipeline.NUM_TRAIN_TIMESTEPS  # 0.522
_ZERO_NOISE_EXPECTED_SCALE = (1.0 - _SIGMA_1) * (1.0 - _SIGMA_2)  # ≈ 0.1162

_RANDN_PATH = "tensorrt_llm._torch.visual_gen.models.wan.pipeline_fastwan.randn_tensor"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingTransformer:
    """Fake transformer: records every timestep call, returns zeros.

    parameters() yields a single bfloat16 tensor so BasePipeline.dtype
    returns torch.bfloat16 without needing a real nn.Module.
    """

    def __init__(self):
        self.captured_timesteps = []
        self._dummy_param = torch.zeros(1, dtype=torch.bfloat16, device="cuda")

        class _Config:
            patch_size = _PATCH_SIZE

        self.config = _Config()

    def parameters(self):
        yield self._dummy_param

    def __call__(self, hidden_states, timestep, encoder_hidden_states):
        self.captured_timesteps.append(timestep.detach().clone())
        return torch.zeros_like(hidden_states)

    @property
    def call_count(self):
        return len(self.captured_timesteps)


def _make_stub(transformer):
    """WanDMDPipeline shell — no weights, no __init__.

    We bypass nn.Module.__setattr__ by writing directly into __dict__.
    - transformer: accessed via __dict__, so no nn.Module submodule registration
    - pipeline_config: WanPipeline.dtype reads self.pipeline_config.torch_dtype
    - rank: read-only property on BasePipeline — returns 0 when MPI is disabled
    """
    pipe = object.__new__(WanDMDPipeline)
    pipe.__dict__["transformer"] = transformer
    pipe.__dict__["pipeline_config"] = type("_Config", (), {"torch_dtype": torch.bfloat16})()
    return pipe


def _run(pipe, latents, zero_noise=False):
    gen = torch.Generator(device=latents.device).manual_seed(42)
    embeds = torch.zeros(1, 8, 4096, device=latents.device, dtype=torch.bfloat16)
    if zero_noise:
        zeros = torch.zeros_like(latents)
        with mock.patch(_RANDN_PATH, return_value=zeros):
            return pipe._denoise(latents, embeds, gen)
    return pipe._denoise(latents, embeds, gen)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestDMDTimesteps:
    """sigma = t / 1000 for every step; tensor shape and call count correct."""

    def test_sigma_values(self):
        recorder = _RecordingTransformer()
        _run(_make_stub(recorder), torch.ones(_LATENT_SHAPE, device="cuda", dtype=torch.bfloat16))

        expected = [t / 1000.0 for t in WanDMDPipeline.DMD_TIMESTEPS]
        for i, (captured, exp) in enumerate(zip(recorder.captured_timesteps, expected)):
            assert torch.allclose(captured, torch.full_like(captured, exp)), (
                f"Step {i}: expected sigma {exp}, got {captured[0, 0].item():.6f}"
            )

    def test_timestep_tensor_shape(self):
        recorder = _RecordingTransformer()
        _run(_make_stub(recorder), torch.ones(_LATENT_SHAPE, device="cuda", dtype=torch.bfloat16))

        for i, captured in enumerate(recorder.captured_timesteps):
            assert captured.shape == (1, _NUM_PATCHES), (
                f"Step {i}: expected (1, {_NUM_PATCHES}), got {captured.shape}"
            )

    def test_transformer_called_three_times(self):
        """One forward per step — CFG-free means no double-pass."""
        recorder = _RecordingTransformer()
        _run(_make_stub(recorder), torch.ones(_LATENT_SHAPE, device="cuda", dtype=torch.bfloat16))
        assert recorder.call_count == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestDMDMath:
    def test_output_dtype_is_bfloat16(self):
        output = _run(
            _make_stub(_RecordingTransformer()),
            torch.ones(_LATENT_SHAPE, device="cuda", dtype=torch.bfloat16),
        )
        assert output.dtype == torch.bfloat16

    def test_exact_output_with_zero_noise(self):
        """pred_noise=0 and randn=0 → output = (1-sigma_1)*(1-sigma_2)*initial ≈ 0.1162*initial."""
        initial = torch.ones(_LATENT_SHAPE, device="cuda", dtype=torch.bfloat16)
        output = _run(_make_stub(_RecordingTransformer()), initial, zero_noise=True)

        expected = torch.full(
            _LATENT_SHAPE, _ZERO_NOISE_EXPECTED_SCALE, dtype=torch.float32, device="cuda"
        ).to(torch.bfloat16)
        assert torch.allclose(output.float(), expected.float(), atol=5e-3), (
            f"Expected ≈{_ZERO_NOISE_EXPECTED_SCALE:.4f}, got {output.float().mean().item():.4f}"
        )

    def test_final_step_does_not_call_randn_tensor(self):
        """randn_tensor must be called for steps 0 and 1 only — never on the final step."""
        from diffusers.utils.torch_utils import randn_tensor as _orig

        call_count = [0]

        def _counting(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 3:
                raise AssertionError("randn_tensor called on the final step — must not happen")
            return _orig(*args, **kwargs)

        pipe = _make_stub(_RecordingTransformer())
        gen = torch.Generator(device="cuda").manual_seed(42)
        embeds = torch.zeros(1, 8, 4096, device="cuda", dtype=torch.bfloat16)
        latents = torch.ones(_LATENT_SHAPE, device="cuda", dtype=torch.bfloat16)

        with mock.patch(_RANDN_PATH, side_effect=_counting):
            pipe._denoise(latents, embeds, gen)

        assert call_count[0] == 2, f"Expected 2 randn_tensor calls, got {call_count[0]}"
