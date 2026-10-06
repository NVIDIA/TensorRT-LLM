# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the TensorRT-LLM project

import math

import numpy as np
import torch

from tensorrt_llm._torch.visual_gen.models.qwen_image_21 import (
    QwenImage21FlowMatchEulerScheduler,
    QwenImage21Pipeline,
    append_target_slots,
    calculate_dimensions,
    calculate_shift,
    pack_latents,
    unpack_latents,
)


def test_qwen_image_21_shift_matches_reference_formula():
    image_seq_len = 4096
    expected = 0.5 + (0.9 - 0.5) * ((image_seq_len - 256) / (8192 - 256))
    assert math.isclose(calculate_shift(image_seq_len, max_seq_len=8192, max_shift=0.9), expected)


def test_qwen_image_21_dimension_rounding_matches_32_pixel_grid():
    width, height, sentinel = calculate_dimensions(1024 * 1024, 16 / 9)
    assert sentinel is None
    assert width % 32 == 0
    assert height % 32 == 0
    assert width >= height


def test_qwen_image_21_pack_unpack_roundtrip_shape():
    latents = torch.arange(2 * 1 * 64 * 4 * 6, dtype=torch.float32).reshape(2, 1, 64, 4, 6)
    packed = pack_latents(latents, batch_size=2, num_channels_latents=64, height=4, width=6)
    assert packed.shape == (2, 24, 64)
    unpacked = unpack_latents(packed, height=64, width=96, vae_scale_factor=16)
    assert unpacked.shape == (2, 64, 1, 4, 6)
    assert torch.equal(unpacked, latents.transpose(1, 2))


def test_qwen_image_21_appends_one_mask_slot_per_2x2_latent_group():
    mask = torch.tensor([[True, False, True]])
    out = append_target_slots(mask, 16)
    assert out.shape == (1, 7)
    assert out[0].tolist() == [True, False, True, True, True, True, True]


def test_qwen_image_21_native_flowmatch_scheduler_matches_reference_equations():
    scheduler = QwenImage21FlowMatchEulerScheduler(
        base_image_seq_len=256,
        base_shift=0.5,
        max_image_seq_len=8192,
        max_shift=0.9,
        shift_terminal=0.02,
        use_dynamic_shifting=True,
        time_shift_type="exponential",
    )
    sigmas = np.linspace(1.0, 0.25, 4, dtype=np.float32)
    mu = calculate_shift(4096, max_seq_len=8192, max_shift=0.9)
    scheduler.set_timesteps(sigmas=sigmas, mu=mu, device="cpu")

    shifted = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
    one_minus_z = 1 - shifted
    scale_factor = one_minus_z[-1] / (1 - 0.02)
    expected_sigmas = 1 - (one_minus_z / scale_factor)
    expected = torch.tensor(expected_sigmas, dtype=torch.float32)

    assert torch.allclose(scheduler.timesteps, expected * 1000, atol=1e-5, rtol=1e-5)
    assert torch.allclose(scheduler.sigmas[:-1], expected, atol=1e-6, rtol=1e-6)
    assert scheduler.sigmas[-1].item() == 0.0

    sample = torch.tensor([[1.0, -2.0]], dtype=torch.float16)
    model_output = torch.tensor([[0.25, -0.5]], dtype=torch.float16)
    out = scheduler.step(model_output, scheduler.timesteps[0], sample, return_dict=False)[0]
    dt = scheduler.sigmas[1] - scheduler.sigmas[0]
    expected_step = (sample.float() + dt * model_output.float()).half()
    assert torch.equal(out, expected_step)


def test_qwen_image_21_pipeline_runtime_metadata():
    assert QwenImage21Pipeline.DEFAULT_GENERATION_PARAMS["num_inference_steps"] == 40
    assert QwenImage21Pipeline.DEFAULT_GENERATION_PARAMS["guidance_scale"] == 1.0
    assert QwenImage21Pipeline.latent_channels == 64
    assert QwenImage21Pipeline.scheduler_class is QwenImage21FlowMatchEulerScheduler
