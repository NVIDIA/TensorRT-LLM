# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""IC-LoRA conditioning and exact stage-weight semantics."""

import pytest
import torch
from safetensors.torch import save_file

from tensorrt_llm._torch.visual_gen.models.ltx2_5.conditioning import (
    STAGE_1_SIGMAS,
    STAGE_2_SIGMAS,
    add_noise,
    append_reference,
    euler_step,
    video_state,
)
from tensorrt_llm._torch.visual_gen.models.ltx2_5.config import LTX25Config
from tensorrt_llm._torch.visual_gen.models.ltx2_5.lora import StageLoRAWeights

pytestmark = pytest.mark.cpu_only


def test_reference_noise_consumption_and_clean_tokens() -> None:
    target = video_state(torch.zeros(1, 128, 2, 2, 2, dtype=torch.bfloat16), 24)
    reference = torch.ones(1, 128, 2, 1, 1, dtype=torch.bfloat16)
    state = append_reference(target, reference, 24, spatial_scale=2)
    generator = torch.Generator().manual_seed(42)
    actual = add_noise(state, generator, 1.0)
    control = torch.Generator().manual_seed(42)
    noise = torch.randn(state.latent.shape, generator=control, dtype=torch.bfloat16)
    torch.testing.assert_close(actual.latent[:, :8], noise[:, :8], rtol=0, atol=0)
    torch.testing.assert_close(actual.latent[:, 8:], state.clean[:, 8:], rtol=0, atol=0)
    assert torch.equal(generator.get_state(), control.get_state())
    assert not state.keyframes_mask[:, 8:].any()


def test_euler_preserves_clean_reference_and_bf16_rounding() -> None:
    state = video_state(torch.randn(1, 128, 1, 2, 2, dtype=torch.bfloat16), 24)
    state = append_reference(state, torch.ones(1, 128, 1, 1, 1, dtype=torch.bfloat16), 24)
    state = add_noise(state, torch.Generator().manual_seed(3), 1.0)
    velocity = torch.randn_like(state.latent)
    sigma, next_sigma = torch.tensor(0.725), torch.tensor(0.421875)
    # Independent X0 -> clean conditioning blend -> Euler reference expression.
    x0 = (state.latent.float() - velocity.float() * (state.mask * sigma)).bfloat16()
    x0 = (x0 * state.mask + state.clean.float() * (1 - state.mask)).bfloat16()
    derivative = ((state.latent.float() - x0.float()) / sigma.item()).bfloat16()
    expected = (state.latent.float() + derivative.float() * (next_sigma - sigma)).bfloat16()
    actual = euler_step(state, velocity, sigma, next_sigma)
    torch.testing.assert_close(actual.latent, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.latent[:, -1], state.clean[:, -1], rtol=0, atol=0)


def test_distilled_schedules() -> None:
    assert len(STAGE_1_SIGMAS) == 9 and len(STAGE_2_SIGMAS) == 4
    assert STAGE_2_SIGMAS == STAGE_1_SIGMAS[-4:]


@pytest.mark.parametrize("strength", [-0.1, 1.1, float("nan")])
def test_reference_rejects_invalid_strength(strength: float) -> None:
    latent = torch.zeros(1, 128, 1, 1, 1)
    with pytest.raises(ValueError, match="strength"):
        append_reference(video_state(latent, 24), latent, 24, strength=strength)


def test_stage_weights_match_official_merge_and_restore_exactly(tmp_path) -> None:
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16))
    original = model[0].weight.detach().clone()
    a, b = torch.randn(2, 4).bfloat16(), torch.randn(3, 2).bfloat16()
    path = tmp_path / "adapter.safetensors"
    save_file({"0.lora_A.weight": a, "0.lora_B.weight": b}, str(path))
    stages = StageLoRAWeights(model, str(path), 0.7)
    expected = ((b * 0.7) @ a).add_(original)
    for _ in range(2):
        stages.select(True)
        torch.testing.assert_close(model[0].weight, expected, rtol=0, atol=0)
        stages.select(False)
        torch.testing.assert_close(model[0].weight, original, rtol=0, atol=0)


def test_invalid_lora_does_not_mutate_weights(tmp_path) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16))
    original = model[0].weight.detach().clone()
    path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "0.lora_A.weight": torch.ones(2, 4),
            "0.lora_B.weight": torch.ones(3, 2),
            "missing.lora_A.weight": torch.ones(2, 4),
            "missing.lora_B.weight": torch.ones(3, 2),
        },
        str(path),
    )
    with pytest.raises(ValueError, match="unmatched"):
        StageLoRAWeights(model, str(path), 1.0)
    torch.testing.assert_close(model[0].weight, original, rtol=0, atol=0)


def test_pipeline_options_reject_unknown_and_nonfinite() -> None:
    with pytest.raises(ValueError):
        LTX25Config(skip_stage_2=True)
    with pytest.raises(ValueError):
        LTX25Config(iclora_scale=float("nan"))


def test_stage_lora_fused_qkv_slices(tmp_path):
    from safetensors.torch import save_file

    from tensorrt_llm._torch.visual_gen.models.ltx2_5.lora import StageLoRAWeights

    model = torch.nn.Module()
    model.attn = torch.nn.Module()
    model.attn.q_dim = 4
    model.attn.kv_dim = 4
    model.attn.qkv_proj = torch.nn.Linear(4, 12, bias=False, dtype=torch.bfloat16)
    base = model.attn.qkv_proj.weight.detach().clone()
    a = torch.ones(2, 4, dtype=torch.bfloat16)
    b = torch.ones(4, 2, dtype=torch.bfloat16)
    path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "diffusion_model.attn.to_k.lora_A.weight": a,
            "diffusion_model.attn.to_k.lora_B.weight": b,
        },
        str(path),
    )
    stages = StageLoRAWeights(model, str(path), 0.5)
    stages.select(True)
    expected = base.clone()
    expected[4:8] += (b * 0.5) @ a
    torch.testing.assert_close(model.attn.qkv_proj.weight, expected, rtol=0, atol=0)
    stages.select(False)
    torch.testing.assert_close(model.attn.qkv_proj.weight, base, rtol=0, atol=0)


def test_incomplete_transformer_checkpoint_fails_before_loading():
    from tensorrt_llm._torch.visual_gen.models.ltx2_5.loading import validate_transformer_weights

    model = torch.nn.Linear(4, 3, dtype=torch.bfloat16)
    original = model.weight.detach().clone()
    with pytest.raises(ValueError, match="missing=.*bias"):
        validate_transformer_weights(model, {"weight": torch.zeros_like(model.weight)})
    torch.testing.assert_close(model.weight, original, rtol=0, atol=0)


def test_iclora_scale_does_not_use_peft_alpha(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16))
    original = model[0].weight.detach().clone()
    path = tmp_path / "adapter.safetensors"
    a, b = torch.ones(2, 4), torch.ones(3, 2)
    save_file(
        {"0.lora_A.weight": a, "0.lora_B.weight": b, "0.alpha": torch.tensor(16.0)}, str(path)
    )
    (tmp_path / "adapter_config.json").write_text('{"lora_alpha": 32}')
    stages = StageLoRAWeights(model, str(path), 0.5)
    stages.select(True)
    expected = original + (b.bfloat16() * 0.5) @ a.bfloat16()
    torch.testing.assert_close(model[0].weight, expected, rtol=0, atol=0)
