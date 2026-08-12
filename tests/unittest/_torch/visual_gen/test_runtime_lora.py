# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.runtime_lora import apply_runtime_lora
from tensorrt_llm.visual_gen.args import RuntimeLoRAConfig


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 3, bias=False)


class TinyWanFFNTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Module()
        self.block.ffn = nn.Module()
        self.block.ffn.up_proj = nn.Linear(2, 3, bias=False)
        self.block.ffn.down_proj = nn.Linear(2, 3, bias=False)


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv_proj = nn.Linear(2, 9, bias=False)


class TinyTritonLikeLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 2
        self.out_features = 3
        self.weight = nn.Parameter(torch.zeros(1, 2, 3), requires_grad=False)

    def forward(self, x):
        return x @ self.weight.squeeze(0)


class TinyTritonTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = TinyTritonLikeLinear()


class TinyPipeline(BasePipeline):
    def __init__(self, runtime_lora_config):
        nn.Module.__init__(self)
        self._runtime_lora_applications = []
        self.pipeline_config = MagicMock()
        self.pipeline_config.runtime_lora = runtime_lora_config
        self.transformer = nn.Linear(2, 3, bias=False)
        self._profiler = object()

    @property
    def transformer_components(self):
        return ["transformer", "_profiler"]

    def forward(self, *args, **kwargs):
        pass

    def _init_transformer(self):
        pass

    def infer(self, req):
        pass


class TinyPipelineWithTransformerOnly(TinyPipeline):
    @property
    def transformer_components(self):
        return ["transformer"]


def test_runtime_lora_fuses_delta_into_weight(tmp_path):
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
    expected_weight = original_weight + torch.tensor([[0.5, 1.0], [1.0, 2.0], [1.5, 3.0]])
    torch.testing.assert_close(model.proj.weight, expected_weight)

    x = torch.tensor([[3.0, 4.0]])
    expected = x @ expected_weight.T
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


def test_runtime_lora_maps_wan_ffn_names(tmp_path):
    model = TinyWanFFNTransformer()
    with torch.no_grad():
        model.block.ffn.up_proj.weight.zero_()
        model.block.ffn.down_proj.weight.zero_()

    lora_path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "block.ffn.net.0.proj.lora_A.weight": torch.tensor([[1.0, 0.0]]),
            "block.ffn.net.0.proj.lora_B.weight": torch.ones(3, 1),
            "block.ffn.net.2.lora_A.weight": torch.tensor([[0.0, 1.0]]),
            "block.ffn.net.2.lora_B.weight": torch.full((3, 1), 2.0),
        },
        str(lora_path),
    )

    report = apply_runtime_lora(model, RuntimeLoRAConfig(path=str(lora_path)))

    assert report.applied_modules == ("block.ffn.up_proj", "block.ffn.down_proj")
    torch.testing.assert_close(
        model.block.ffn.up_proj(torch.tensor([[5.0, 7.0]])),
        torch.full((1, 3), 5.0),
    )
    torch.testing.assert_close(
        model.block.ffn.down_proj(torch.tensor([[5.0, 7.0]])),
        torch.full((1, 3), 14.0),
    )


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
    torch.testing.assert_close(
        model.attn.qkv_proj.weight,
        torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [2.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [3.0, 0.0],
                [3.0, 0.0],
            ]
        ),
    )
    out = model.attn.qkv_proj(torch.tensor([[5.0, 7.0]]))
    expected = torch.tensor([[5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 15.0, 15.0, 15.0]])
    torch.testing.assert_close(out, expected)


def test_runtime_lora_rejects_mismatched_qkv_total_span(tmp_path):
    model = TinyAttention()
    with torch.no_grad():
        model.attn.qkv_proj.weight.fill_(2.0)
    original_weight = model.attn.qkv_proj.weight.detach().clone()

    lora_path = tmp_path / "adapter.safetensors"
    tensors = {}
    for suffix, rows in (("to_q", 3), ("to_k", 2), ("to_v", 3)):
        tensors[f"attn.{suffix}.lora_A.weight"] = torch.ones(1, 2)
        tensors[f"attn.{suffix}.lora_B.weight"] = torch.ones(rows, 1)
    save_file(tensors, str(lora_path))

    with pytest.raises(ValueError, match="fused-QKV spans"):
        apply_runtime_lora(model, RuntimeLoRAConfig(path=str(lora_path)))
    torch.testing.assert_close(model.attn.qkv_proj.weight, original_weight)


def test_runtime_lora_rejects_incomplete_qkv_in_strict_mode(tmp_path):
    model = TinyAttention()
    lora_path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "attn.to_q.lora_A.weight": torch.tensor([[1.0, 0.0]]),
            "attn.to_q.lora_B.weight": torch.ones(3, 1),
            "attn.to_k.lora_A.weight": torch.tensor([[1.0, 0.0]]),
            "attn.to_k.lora_B.weight": torch.ones(3, 1),
        },
        str(lora_path),
    )

    with pytest.raises(ValueError, match="incomplete fused-QKV"):
        apply_runtime_lora(model, RuntimeLoRAConfig(path=str(lora_path)))


def test_runtime_lora_allows_incomplete_qkv_when_not_strict(tmp_path):
    model = TinyAttention()
    lora_path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "attn.to_q.lora_A.weight": torch.tensor([[1.0, 0.0]]),
            "attn.to_q.lora_B.weight": torch.ones(3, 1),
        },
        str(lora_path),
    )

    report = apply_runtime_lora(
        model,
        RuntimeLoRAConfig(path=str(lora_path), strict=False),
        raise_on_no_matches=False,
    )

    assert report.applied_modules == ()
    assert report.skipped_incomplete == 1


def test_runtime_lora_fuses_triton_weight_layout(tmp_path):
    model = TinyTritonTransformer()
    lora_path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "proj.lora_A.weight": torch.tensor([[1.0, 2.0]]),
            "proj.lora_B.weight": torch.tensor([[0.5], [1.0], [1.5]]),
        },
        str(lora_path),
    )

    report = apply_runtime_lora(model, RuntimeLoRAConfig(path=str(lora_path)))

    assert report.applied_modules == ("proj",)
    expected_weight = torch.tensor([[[0.5, 1.0, 1.5], [1.0, 2.0, 3.0]]])
    torch.testing.assert_close(model.proj.weight, expected_weight)
    torch.testing.assert_close(
        model.proj(torch.tensor([[3.0, 4.0]])),
        torch.tensor([[5.5, 11.0, 16.5]]),
    )


def test_runtime_lora_rejects_double_fusion(tmp_path):
    model = TinyTransformer()
    with torch.no_grad():
        model.proj.weight.zero_()

    lora_path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "proj.lora_A.weight": torch.tensor([[1.0, 2.0]]),
            "proj.lora_B.weight": torch.tensor([[0.5], [1.0], [1.5]]),
        },
        str(lora_path),
    )

    apply_runtime_lora(model, RuntimeLoRAConfig(path=str(lora_path)))
    fused_weight = model.proj.weight.detach().clone()

    with pytest.raises(ValueError, match="already fused"):
        apply_runtime_lora(model, RuntimeLoRAConfig(path=str(lora_path)))
    torch.testing.assert_close(model.proj.weight, fused_weight)


def test_runtime_lora_shape_failure_does_not_mutate_weight(tmp_path):
    model = TinyAttention()
    with torch.no_grad():
        model.attn.qkv_proj.weight.fill_(2.0)
    original_weight = model.attn.qkv_proj.weight.detach().clone()

    lora_path = tmp_path / "adapter.safetensors"
    tensors = {}
    for suffix, value in (("to_q", 1.0), ("to_k", 2.0), ("to_v", 3.0)):
        tensors[f"attn.{suffix}.lora_A.weight"] = torch.ones(1, 2)
        rows = 4 if suffix == "to_v" else 3
        tensors[f"attn.{suffix}.lora_B.weight"] = torch.full((rows, 1), value)
    save_file(tensors, str(lora_path))

    with pytest.raises(ValueError, match="fused-QKV spans"):
        apply_runtime_lora(model, RuntimeLoRAConfig(path=str(lora_path)))
    torch.testing.assert_close(model.attn.qkv_proj.weight, original_weight)


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


def test_pipeline_runtime_lora_rejects_non_module_target_component():
    pipe = TinyPipeline(
        RuntimeLoRAConfig(
            path="/tmp/lora.safetensors",
            target_components=["_profiler"],
        )
    )

    with pytest.raises(ValueError, match="non-module component"):
        pipe._setup_runtime_lora()


def test_pipeline_runtime_lora_skips_non_module_target_component_when_not_strict():
    pipe = TinyPipeline(
        RuntimeLoRAConfig(
            path="/tmp/lora.safetensors",
            target_components=["_profiler"],
            strict=False,
        )
    )

    pipe._setup_runtime_lora()

    assert pipe._runtime_lora_applications == []


def test_pipeline_runtime_lora_rejects_component_outside_transformer_surface():
    pipe = TinyPipelineWithTransformerOnly(
        RuntimeLoRAConfig(
            path="/tmp/lora.safetensors",
            target_components=["_profiler"],
        )
    )

    with pytest.raises(ValueError, match="not in transformer_components"):
        pipe._setup_runtime_lora()
