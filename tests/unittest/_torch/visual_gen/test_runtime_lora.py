# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from tensorrt_llm._torch.visual_gen.runtime_lora import RuntimeLoRALinear, apply_runtime_lora
from tensorrt_llm.visual_gen.args import RuntimeLoRAConfig


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 3, bias=False)


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv_proj = nn.Linear(2, 9, bias=False)


def test_runtime_lora_adds_forward_delta_without_mutating_base_weight(tmp_path):
    model = TinyTransformer()
    with torch.no_grad():
        model.proj.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    original_weight = model.proj.weight.detach().clone()

    lora_path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "proj.lora_A.weight": torch.tensor([[1.0, 2.0]]),
            "proj.lora_B.weight": torch.tensor([[0.5], [1.0], [1.5]]),
            "proj.alpha": torch.tensor(1.0),
        },
        str(lora_path),
    )

    report = apply_runtime_lora(model, RuntimeLoRAConfig(path=str(lora_path)))

    assert report.applied_modules == ("proj",)
    assert isinstance(model.proj, RuntimeLoRALinear)
    torch.testing.assert_close(model.proj.base_layer.weight, original_weight)

    x = torch.tensor([[3.0, 4.0]])
    base = x @ original_weight.T
    lora_inner = x @ torch.tensor([[1.0], [2.0]])
    expected = base + lora_inner @ torch.tensor([[0.5, 1.0, 1.5]])
    torch.testing.assert_close(model.proj(x), expected)


def test_runtime_lora_applies_scale_and_alpha(tmp_path):
    model = TinyTransformer()
    with torch.no_grad():
        model.proj.weight.zero_()

    lora_path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "proj.lora_down.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "proj.lora_up.weight": torch.tensor([[1.0, 1.0], [2.0, 0.0], [0.0, 2.0]]),
            "proj.alpha": torch.tensor(4.0),
        },
        str(lora_path),
    )

    apply_runtime_lora(model, RuntimeLoRAConfig(path=str(lora_path), scale=0.5))

    x = torch.tensor([[2.0, 3.0]])
    expected = torch.tensor([[5.0, 4.0, 6.0]])
    torch.testing.assert_close(model.proj(x), expected)


def test_runtime_lora_strips_prefix_and_uses_key_map(tmp_path):
    model = TinyTransformer()
    with torch.no_grad():
        model.proj.weight.zero_()

    lora_path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "wrapped.old_proj.lora_A.weight": torch.tensor([[1.0, 0.0]]),
            "wrapped.old_proj.lora_B.weight": torch.ones(3, 1),
        },
        str(lora_path),
    )

    config = RuntimeLoRAConfig(
        path=str(lora_path),
        strip_prefixes=["wrapped."],
        key_map={"old_proj": "proj"},
    )
    report = apply_runtime_lora(model, config)

    assert report.applied_modules == ("proj",)
    torch.testing.assert_close(model.proj(torch.tensor([[2.0, 4.0]])), torch.full((1, 3), 2.0))


def test_runtime_lora_fuses_qkv_segments(tmp_path):
    model = TinyAttention()
    with torch.no_grad():
        model.attn.qkv_proj.weight.zero_()

    lora_path = tmp_path / "adapter.safetensors"
    tensors = {}
    for suffix, value in (("to_q", 1.0), ("to_k", 2.0), ("to_v", 3.0)):
        tensors[f"attn.{suffix}.lora_A.weight"] = torch.tensor([[1.0, 0.0]])
        tensors[f"attn.{suffix}.lora_B.weight"] = torch.full((3, 1), value)
    save_file(tensors, str(lora_path))

    report = apply_runtime_lora(model, RuntimeLoRAConfig(path=str(lora_path)))

    assert report.applied_modules == ("attn.qkv_proj",)
    out = model.attn.qkv_proj(torch.tensor([[5.0, 7.0]]))
    expected = torch.tensor([[5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 15.0, 15.0, 15.0]])
    torch.testing.assert_close(out, expected)


def test_runtime_lora_raises_when_no_modules_match(tmp_path):
    model = TinyTransformer()
    lora_path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "missing.lora_A.weight": torch.ones(1, 2),
            "missing.lora_B.weight": torch.ones(3, 1),
        },
        str(lora_path),
    )

    with pytest.raises(ValueError, match="No Runtime LoRA modules"):
        apply_runtime_lora(model, RuntimeLoRAConfig(path=str(lora_path)))


def test_runtime_lora_raises_on_shape_mismatch(tmp_path):
    model = TinyTransformer()
    lora_path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "proj.lora_A.weight": torch.ones(1, 4),
            "proj.lora_B.weight": torch.ones(3, 1),
        },
        str(lora_path),
    )

    with pytest.raises(ValueError, match="input mismatch"):
        apply_runtime_lora(model, RuntimeLoRAConfig(path=str(lora_path)))
