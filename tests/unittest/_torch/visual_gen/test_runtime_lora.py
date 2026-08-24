# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.runtime_lora import apply_runtime_lora
from tensorrt_llm.visual_gen.args import RuntimeLoRAConfig


class TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 3, bias=False)


class TinyQwenMLPTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Module()
        self.block.img_mlp = nn.Module()
        self.block.img_mlp.up_proj = nn.Linear(2, 3, bias=False)
        self.block.img_mlp.down_proj = nn.Linear(2, 3, bias=False)
        self.block.txt_mlp = nn.Module()
        self.block.txt_mlp.up_proj = nn.Linear(2, 3, bias=False)
        self.block.txt_mlp.down_proj = nn.Linear(2, 3, bias=False)


class TinyTwoLinearTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(2, 3, bias=False)
        self.second = nn.Linear(2, 3, bias=False)


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv_proj = nn.Linear(2, 9, bias=False)


class TinyUnsupportedTPLinear(nn.Linear):
    def __init__(self) -> None:
        super().__init__(2, 3, bias=False)
        self.tp_size = 2
        self.tp_mode = "row"


class TinyTPTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = TinyUnsupportedTPLinear()


class TinyPipeline(BasePipeline):
    def __init__(self, runtime_lora_config: RuntimeLoRAConfig) -> None:
        nn.Module.__init__(self)
        self._runtime_lora_applications = []
        self.pipeline_config = MagicMock()
        self.pipeline_config.runtime_lora = runtime_lora_config
        self.transformer = TinyTransformer()
        self._profiler = object()

    @property
    def transformer_components(self) -> list[str]:
        return ["transformer", "_profiler"]

    def forward(self, *args: object, **kwargs: object) -> None:
        pass

    def _init_transformer(self) -> None:
        pass

    def infer(self, req: object) -> None:
        pass


class TinyPipelineWithTwoTransformers(TinyPipeline):
    def __init__(self, runtime_lora_config: RuntimeLoRAConfig) -> None:
        super().__init__(runtime_lora_config)
        self.transformer = TinyTransformer()
        self.transformer_2 = TinyTransformer()

    @property
    def transformer_components(self) -> list[str]:
        return ["transformer", "transformer_2"]


def test_runtime_lora_fuses_comfy_delta_with_alpha_into_weight(tmp_path: Path) -> None:
    model = TinyTransformer()
    with torch.no_grad():
        model.proj.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    original_weight = model.proj.weight.detach().clone()

    lora_path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "proj.lora_down.weight": torch.tensor([[1.0, 2.0]]),
            "proj.lora_up.weight": torch.tensor([[0.5], [1.0], [1.5]]),
            "proj.alpha": torch.tensor(2.0),
        },
        str(lora_path),
    )

    report = apply_runtime_lora(model, RuntimeLoRAConfig(path=str(lora_path)))

    assert report.applied_modules == ("proj",)
    expected_weight = original_weight + torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    torch.testing.assert_close(model.proj.weight, expected_weight)


def test_runtime_lora_uses_peft_adapter_config_alpha(tmp_path: Path) -> None:
    model = TinyTransformer()
    with torch.no_grad():
        model.proj.weight.zero_()

    lora_dir = tmp_path / "adapter"
    lora_dir.mkdir()
    save_file(
        {
            "proj.lora_A.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "proj.lora_B.weight": torch.tensor([[1.0, 1.0], [2.0, 0.0], [0.0, 2.0]]),
        },
        str(lora_dir / "adapter_model.safetensors"),
    )
    (lora_dir / "adapter_config.json").write_text('{"lora_alpha": 4}', encoding="utf-8")

    apply_runtime_lora(model, RuntimeLoRAConfig(path=str(lora_dir), scale=0.5))

    torch.testing.assert_close(
        model.proj(torch.tensor([[2.0, 3.0]])),
        torch.tensor([[5.0, 4.0, 6.0]]),
    )


def test_runtime_lora_maps_qwen_mlp_names(tmp_path: Path) -> None:
    model = TinyQwenMLPTransformer()
    with torch.no_grad():
        model.block.img_mlp.up_proj.weight.zero_()
        model.block.img_mlp.down_proj.weight.zero_()
        model.block.txt_mlp.up_proj.weight.zero_()
        model.block.txt_mlp.down_proj.weight.zero_()

    lora_path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "block.img_mlp.net.0.proj.lora_A.weight": torch.tensor([[1.0, 0.0]]),
            "block.img_mlp.net.0.proj.lora_B.weight": torch.ones(3, 1),
            "block.img_mlp.net.2.lora_A.weight": torch.tensor([[0.0, 1.0]]),
            "block.img_mlp.net.2.lora_B.weight": torch.full((3, 1), 2.0),
            "block.txt_mlp.net.0.proj.lora_A.weight": torch.tensor([[3.0, 0.0]]),
            "block.txt_mlp.net.0.proj.lora_B.weight": torch.ones(3, 1),
            "block.txt_mlp.net.2.lora_A.weight": torch.tensor([[0.0, 3.0]]),
            "block.txt_mlp.net.2.lora_B.weight": torch.full((3, 1), 2.0),
        },
        str(lora_path),
    )

    report = apply_runtime_lora(model, RuntimeLoRAConfig(path=str(lora_path)))

    assert set(report.applied_modules) == {
        "block.img_mlp.up_proj",
        "block.img_mlp.down_proj",
        "block.txt_mlp.up_proj",
        "block.txt_mlp.down_proj",
    }
    assert report.skipped_non_targets == 0


def test_runtime_lora_fuses_qkv_segments(tmp_path: Path) -> None:
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


def test_runtime_lora_rejects_mismatched_qkv_total_span(tmp_path: Path) -> None:
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


def test_runtime_lora_rejects_partial_unmatched_targets_before_mutating(
    tmp_path: Path,
) -> None:
    model = TinyTransformer()
    with torch.no_grad():
        model.proj.weight.zero_()

    lora_path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "proj.lora_A.weight": torch.tensor([[1.0, 0.0]]),
            "proj.lora_B.weight": torch.ones(3, 1),
            "missing.lora_A.weight": torch.ones(1, 2),
            "missing.lora_B.weight": torch.ones(3, 1),
        },
        str(lora_path),
    )

    with pytest.raises(ValueError, match="skipped 1 adapter target"):
        apply_runtime_lora(model, RuntimeLoRAConfig(path=str(lora_path)))

    torch.testing.assert_close(model.proj.weight, torch.zeros_like(model.proj.weight))


def test_runtime_lora_strict_shape_failure_does_not_mutate_prior_targets(
    tmp_path: Path,
) -> None:
    model = TinyTwoLinearTransformer()
    with torch.no_grad():
        model.first.weight.zero_()
        model.second.weight.zero_()

    lora_path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "first.lora_A.weight": torch.tensor([[1.0, 0.0]]),
            "first.lora_B.weight": torch.ones(3, 1),
            "second.lora_A.weight": torch.ones(1, 4),
            "second.lora_B.weight": torch.ones(3, 1),
        },
        str(lora_path),
    )

    with pytest.raises(ValueError, match="input mismatch"):
        apply_runtime_lora(model, RuntimeLoRAConfig(path=str(lora_path)))

    torch.testing.assert_close(model.first.weight, torch.zeros_like(model.first.weight))
    torch.testing.assert_close(model.second.weight, torch.zeros_like(model.second.weight))
    assert not getattr(model.first, "_trtllm_runtime_lora_fused", False)
    assert not getattr(model.second, "_trtllm_runtime_lora_fused", False)


def test_runtime_lora_rejects_duplicate_keys_across_safetensors(tmp_path: Path) -> None:
    lora_dir = tmp_path / "adapter"
    lora_dir.mkdir()
    tensors = {
        "proj.lora_A.weight": torch.ones(1, 2),
        "proj.lora_B.weight": torch.ones(3, 1),
    }
    save_file(tensors, str(lora_dir / "adapter_1.safetensors"))
    save_file(tensors, str(lora_dir / "adapter_2.safetensors"))

    with pytest.raises(ValueError, match="duplicate tensor key"):
        apply_runtime_lora(TinyTransformer(), RuntimeLoRAConfig(path=str(lora_dir)))


def test_runtime_lora_non_strict_skips_unsupported_tp_target(tmp_path: Path) -> None:
    model = TinyTPTransformer()
    lora_path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "proj.lora_A.weight": torch.ones(1, 2),
            "proj.lora_B.weight": torch.ones(3, 1),
        },
        str(lora_path),
    )

    report = apply_runtime_lora(model, RuntimeLoRAConfig(path=str(lora_path), strict=False))

    assert report.applied_modules == ()
    assert report.skipped_non_targets == 1


def test_pipeline_runtime_lora_requires_target_components_for_multi_transformer() -> None:
    pipe = TinyPipelineWithTwoTransformers(RuntimeLoRAConfig(path="/tmp/lora.safetensors"))

    with pytest.raises(ValueError, match="target_components must be set"):
        pipe._setup_runtime_lora()


def test_pipeline_runtime_lora_rejects_duplicate_target_components() -> None:
    pipe = TinyPipelineWithTwoTransformers(
        RuntimeLoRAConfig(
            path="/tmp/lora.safetensors",
            target_components=["transformer", "transformer"],
        )
    )

    with pytest.raises(ValueError, match="target_components contains duplicates"):
        pipe._setup_runtime_lora()


def test_pipeline_runtime_lora_strict_failure_does_not_mutate_components(
    tmp_path: Path,
) -> None:
    lora_path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "proj.lora_A.weight": torch.tensor([[1.0, 0.0]]),
            "proj.lora_B.weight": torch.ones(3, 1),
        },
        str(lora_path),
    )
    pipe = TinyPipelineWithTwoTransformers(
        RuntimeLoRAConfig(
            path=str(lora_path),
            target_components=["transformer", "transformer_2"],
        )
    )
    pipe.transformer_2.proj = nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        pipe.transformer.proj.weight.zero_()
        pipe.transformer_2.proj.weight.zero_()

    with pytest.raises(ValueError, match="input mismatch"):
        pipe._setup_runtime_lora()

    torch.testing.assert_close(
        pipe.transformer.proj.weight,
        torch.zeros_like(pipe.transformer.proj.weight),
    )
    torch.testing.assert_close(
        pipe.transformer_2.proj.weight,
        torch.zeros_like(pipe.transformer_2.proj.weight),
    )
    assert not getattr(pipe.transformer.proj, "_trtllm_runtime_lora_fused", False)
    assert not getattr(pipe.transformer_2.proj, "_trtllm_runtime_lora_fused", False)
    assert pipe._runtime_lora_applications == []


def test_pipeline_runtime_lora_strict_partial_component_match_does_not_mutate(
    tmp_path: Path,
) -> None:
    lora_path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "model.transformer.proj.lora_A.weight": torch.tensor([[1.0, 0.0]]),
            "model.transformer.proj.lora_B.weight": torch.ones(3, 1),
        },
        str(lora_path),
    )
    pipe = TinyPipelineWithTwoTransformers(
        RuntimeLoRAConfig(
            path=str(lora_path),
            target_components=["transformer", "transformer_2"],
        )
    )
    with torch.no_grad():
        pipe.transformer.proj.weight.zero_()
        pipe.transformer_2.proj.weight.zero_()

    with pytest.raises(ValueError, match="selected component 'transformer_2'"):
        pipe._setup_runtime_lora()

    torch.testing.assert_close(
        pipe.transformer.proj.weight,
        torch.zeros_like(pipe.transformer.proj.weight),
    )
    torch.testing.assert_close(
        pipe.transformer_2.proj.weight,
        torch.zeros_like(pipe.transformer_2.proj.weight),
    )
    assert not getattr(pipe.transformer.proj, "_trtllm_runtime_lora_fused", False)
    assert not getattr(pipe.transformer_2.proj, "_trtllm_runtime_lora_fused", False)
    assert pipe._runtime_lora_applications == []
