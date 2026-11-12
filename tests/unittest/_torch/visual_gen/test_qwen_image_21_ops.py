# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the TensorRT-LLM project

import json
import math

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.embeddings import TimestepEmbedding

from tensorrt_llm._torch.visual_gen.models.qwen_image_21 import (
    QwenImage21FlowMatchEulerScheduler,
    QwenImage21Pipeline,
    append_target_slots,
    calculate_dimensions,
    calculate_shift,
    pack_latents,
    unpack_latents,
)
from tensorrt_llm._torch.visual_gen.models.qwen_image_21.transformer_qwen_image_21 import (
    QwenImage21Attention,
    QwenImage21RMSNorm,
    QwenImage21TimestepProjEmbeddings,
    QwenImage21ZeroCenterRMSNorm,
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


def test_qwen_image_21_flowmatch_scheduler_matches_reference_equations():
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


@pytest.mark.parametrize("filename", ["scheduler_config.json", "config.json"])
def test_qwen_image_21_scheduler_loads_local_config(tmp_path, filename):
    config_dir = tmp_path / "scheduler"
    config_dir.mkdir()
    (config_dir / filename).write_text(json.dumps({"shift_terminal": 0.05, "max_shift": 1.1}))
    scheduler = QwenImage21FlowMatchEulerScheduler.from_pretrained(str(tmp_path))
    assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
    assert scheduler.config.shift_terminal == 0.05
    assert scheduler.config.max_shift == 1.1
    assert scheduler.config.use_dynamic_shifting is True
    assert scheduler.config.max_image_seq_len == 8192


@pytest.mark.parametrize("sample_dtype", [torch.float32, torch.bfloat16])
def test_qwen_image_21_scheduler_matches_diffusers_across_steps(sample_dtype):
    scheduler = QwenImage21FlowMatchEulerScheduler()
    reference = FlowMatchEulerDiscreteScheduler.from_config(scheduler.config)
    for instance in (scheduler, reference):
        instance.set_timesteps(sigmas=[1.0, 0.75, 0.5, 0.25], mu=0.7, device="cpu")
        instance.set_begin_index(0)
    sample = torch.tensor([[1.1, -2.3]], dtype=sample_dtype)
    expected = sample.clone()
    model_output = torch.tensor([[0.27, -0.53]], dtype=torch.bfloat16)
    for t, reference_t in zip(scheduler.timesteps, reference.timesteps):
        sample = scheduler.step(model_output, t, sample, return_dict=False)[0].to(sample_dtype)
        expected = reference.step(model_output, reference_t, expected, return_dict=False)[0].to(
            sample_dtype
        )
        torch.testing.assert_close(sample, expected, rtol=0, atol=0)


def test_qwen_image_21_timestep_embedding_preserves_checkpoint_layout():
    embedding = QwenImage21TimestepProjEmbeddings(16).timestep_embedder
    assert isinstance(embedding, TimestepEmbedding)
    assert set(embedding.state_dict()) == {"linear_1.weight", "linear_2.weight"}
    x = torch.linspace(-1, 1, 256).unsqueeze(0)
    expected = torch.nn.functional.linear(x, embedding.linear_1.weight)
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(expected), embedding.linear_2.weight
    )
    torch.testing.assert_close(embedding(x), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qwen_image_21_timesteps_preserve_scaling_and_frequency_order(dtype):
    embedding = QwenImage21TimestepProjEmbeddings(16)
    timestep = torch.tensor([0.0, 0.25, 0.75, 1.0], dtype=dtype)
    hidden_states = torch.zeros(4, 16)
    freqs = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32) / 128)
    phases = (timestep.float() * 1000.0)[:, None] * freqs[None]
    expected = embedding.timestep_embedder(torch.cat([phases.cos(), phases.sin()], dim=-1))
    torch.testing.assert_close(embedding(timestep, hidden_states), expected, atol=1e-5, rtol=1e-4)


def test_qwen_image_21_attention_qk_norm_uses_regular_rmsnorm():
    attn = QwenImage21Attention(dim=16, heads=2, dim_head=4)

    assert isinstance(attn.norm_q, QwenImage21RMSNorm)
    assert isinstance(attn.norm_k, QwenImage21RMSNorm)
    assert not isinstance(attn.norm_q, QwenImage21ZeroCenterRMSNorm)
    assert not isinstance(attn.norm_k, QwenImage21ZeroCenterRMSNorm)
    assert torch.equal(attn.norm_q.weight, torch.ones_like(attn.norm_q.weight))
    assert torch.equal(attn.norm_k.weight, torch.ones_like(attn.norm_k.weight))


def test_qwen_image_21_attention_rmsnorm_is_not_zero_centered():
    x = torch.tensor(
        [
            [[[1.0, -2.0, 3.0, -4.0], [0.25, -0.5, 0.75, -1.0]]],
            [[[-1.5, 2.5, -3.5, 4.5], [1.25, 0.5, -0.75, -1.5]]],
        ],
        dtype=torch.float32,
    )
    weight = torch.tensor([0.5, 0.75, 1.25, 1.5], dtype=torch.float32)
    regular = QwenImage21RMSNorm(4, eps=1e-6)
    zero_center = QwenImage21ZeroCenterRMSNorm(4, eps=1e-6)
    with torch.no_grad():
        regular.weight.copy_(weight)
        zero_center.weight.copy_(weight)

    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * weight
    assert torch.allclose(regular(x), expected, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(zero_center(x), expected, atol=1e-6, rtol=1e-6)


def test_qwen_image_21_pipeline_runtime_metadata():
    assert QwenImage21Pipeline.DEFAULT_GENERATION_PARAMS["num_inference_steps"] == 40
    assert QwenImage21Pipeline.DEFAULT_GENERATION_PARAMS["guidance_scale"] == 1.0
    assert QwenImage21Pipeline.latent_channels == 64
    assert QwenImage21Pipeline.scheduler_class is QwenImage21FlowMatchEulerScheduler
