# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_lora import apply_flux3_runtime_lora
from tensorrt_llm.visual_gen.args import RuntimeLoRAConfig


class _GateUpLinear(nn.Linear):
    """load_shard() identity passthrough, matching real Linear at tp_size==1."""

    def load_shard(self, tensor, name=None, device=None):
        return tensor.to(device=device) if device is not None else tensor


class _TinyFlux3Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.to_q = nn.Linear(4, 4, bias=False)
        self.attn.to_k = nn.Linear(4, 4, bias=False)
        self.attn.to_v = nn.Linear(4, 4, bias=False)
        self.attn_out = nn.Linear(4, 4, bias=False)
        self.ff = nn.Module()
        self.ff.gate_up_proj = _GateUpLinear(4, 8, bias=False)
        self.ff.down_proj = nn.Linear(4, 4, bias=False)
        for module in (
            self.attn.to_q,
            self.attn.to_k,
            self.attn.to_v,
            self.attn_out,
            self.ff.gate_up_proj,
            self.ff.down_proj,
        ):
            module.weight.data.zero_()


class TinyFlux3Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_TinyFlux3Block()])


class _TinyFlux3FusedQKVBlock(nn.Module):
    """tp_size==1 topology: one fused qkv_proj instead of to_q/to_k/to_v."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv_proj = nn.Linear(4, 12, bias=False)
        self.attn_out = nn.Linear(4, 4, bias=False)
        self.ff = nn.Module()
        self.ff.gate_up_proj = _GateUpLinear(4, 8, bias=False)
        self.ff.down_proj = nn.Linear(4, 4, bias=False)
        for module in (self.attn.qkv_proj, self.attn_out, self.ff.gate_up_proj, self.ff.down_proj):
            module.weight.data.zero_()


class TinyFlux3FusedQKVTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_TinyFlux3FusedQKVBlock()])


class _TPFusedQKVLinear(nn.Linear):
    """Fused qkv_proj that also reports itself as TP-sharded (still unsupported)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=False)
        self.tp_size = 2
        self.tp_mode = "column"


class TinyFlux3FusedQKVUnderTPTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        block = nn.Module()
        block.attn = nn.Module()
        block.attn.qkv_proj = _TPFusedQKVLinear(4, 12)
        block.attn_out = nn.Linear(4, 4, bias=False)
        block.ff = nn.Module()
        block.ff.gate_up_proj = _GateUpLinear(4, 8, bias=False)
        block.ff.down_proj = nn.Linear(4, 4, bias=False)
        for module in (
            block.attn.qkv_proj,
            block.attn_out,
            block.ff.gate_up_proj,
            block.ff.down_proj,
        ):
            module.weight.data.zero_()
        self.blocks = nn.ModuleList([block])


class _TPColumnLinear(nn.Linear):
    """Column-parallel stand-in: load_shard() returns this rank's (0) local half."""

    def __init__(self, in_features: int, out_features_local: int) -> None:
        super().__init__(in_features, out_features_local, bias=False)
        self.tp_size = 2
        self.tp_mode = "column"

    def load_shard(self, tensor, name=None, device=None):
        half = tensor.shape[0] // 2
        shard = tensor[:half, :]
        return shard.to(device=device) if device is not None else shard


def _save_qkv_lora(
    path: Path,
    out_features=(4, 4, 4),
    down_suffix=".lora_A.weight",
    up_suffix=".lora_B.weight",
    extra: dict | None = None,
) -> None:
    torch.manual_seed(0)
    tensors = dict(extra or {})
    for name, out_dim in zip(("q_proj", "k_proj", "v_proj"), out_features):
        tensors[f"dit.blocks.0.{name}{down_suffix}"] = torch.randn(1, 4)
        tensors[f"dit.blocks.0.{name}{up_suffix}"] = torch.randn(out_dim, 1)
    save_file(tensors, str(path))


def _save_bfl_native_lora(path: Path, prefix: str) -> None:
    torch.manual_seed(0)
    tensors = {}
    for name, out_features in (
        ("q_proj", 4),
        ("k_proj", 4),
        ("v_proj", 4),
        ("attn_out", 4),
        ("mlp_out", 4),
    ):
        tensors[f"{prefix}blocks.0.{name}.lora_A.weight"] = torch.randn(1, 4)
        tensors[f"{prefix}blocks.0.{name}.lora_B.weight"] = torch.randn(out_features, 1)
    tensors[f"{prefix}blocks.0.mlp_in.lora_A.weight"] = torch.randn(1, 4)
    tensors[f"{prefix}blocks.0.mlp_in.lora_B.weight"] = torch.randn(8, 1)
    save_file(tensors, str(path))


@pytest.mark.parametrize(
    "prefix", ["dit.", "diffusion_model.", "transformer.", "model.transformer."]
)
def test_flux3_lora_fuses_all_targets(tmp_path: Path, prefix: str) -> None:
    lora_dir = tmp_path / "adapter"
    lora_dir.mkdir()
    _save_bfl_native_lora(lora_dir / "adapter.safetensors", prefix)

    transformer = TinyFlux3Transformer()
    application = apply_flux3_runtime_lora(transformer, RuntimeLoRAConfig(path=str(lora_dir)))

    block = transformer.blocks[0]
    assert set(application.applied_modules) == {
        "blocks.0.attn.to_q",
        "blocks.0.attn.to_k",
        "blocks.0.attn.to_v",
        "blocks.0.attn_out",
        "blocks.0.ff.gate_up_proj",
        "blocks.0.ff.down_proj",
    }
    assert not torch.allclose(block.attn.to_q.weight, torch.zeros(4, 4))
    assert not torch.allclose(block.ff.gate_up_proj.weight, torch.zeros(8, 4))


def test_flux3_lora_gate_up_split_matches_expected_delta(tmp_path: Path) -> None:
    lora_dir = tmp_path / "adapter"
    lora_dir.mkdir()
    a = torch.randn(1, 4)
    b = torch.randn(8, 1)
    tensors = {
        "dit.blocks.0.mlp_in.lora_A.weight": a,
        "dit.blocks.0.mlp_in.lora_B.weight": b,
    }
    save_file(tensors, str(lora_dir / "adapter.safetensors"))

    transformer = TinyFlux3Transformer()
    apply_flux3_runtime_lora(transformer, RuntimeLoRAConfig(path=str(lora_dir)))

    expected = (b @ a).to(torch.float32)
    torch.testing.assert_close(transformer.blocks[0].ff.gate_up_proj.weight.data, expected)


def test_flux3_lora_strict_rejects_unmatched_target(tmp_path: Path) -> None:
    lora_dir = tmp_path / "adapter"
    lora_dir.mkdir()
    tensors = {
        "dit.blocks.0.unknown_proj.lora_A.weight": torch.randn(1, 4),
        "dit.blocks.0.unknown_proj.lora_B.weight": torch.randn(4, 1),
    }
    save_file(tensors, str(lora_dir / "adapter.safetensors"))

    transformer = TinyFlux3Transformer()
    with pytest.raises(ValueError, match="not found in transformer"):
        apply_flux3_runtime_lora(transformer, RuntimeLoRAConfig(path=str(lora_dir)))


def test_flux3_lora_non_strict_skips_unmatched_target(tmp_path: Path) -> None:
    lora_dir = tmp_path / "adapter"
    lora_dir.mkdir()
    tensors = {
        "dit.blocks.0.unknown_proj.lora_A.weight": torch.randn(1, 4),
        "dit.blocks.0.unknown_proj.lora_B.weight": torch.randn(4, 1),
    }
    save_file(tensors, str(lora_dir / "adapter.safetensors"))

    transformer = TinyFlux3Transformer()
    application = apply_flux3_runtime_lora(
        transformer, RuntimeLoRAConfig(path=str(lora_dir), strict=False)
    )
    assert application.applied_modules == ()
    assert application.skipped_non_targets == 1


def test_flux3_lora_fuses_qkv_segments_into_fused_qkv_proj(tmp_path: Path) -> None:
    lora_dir = tmp_path / "adapter"
    lora_dir.mkdir()
    _save_qkv_lora(lora_dir / "adapter.safetensors")

    transformer = TinyFlux3FusedQKVTransformer()
    application = apply_flux3_runtime_lora(transformer, RuntimeLoRAConfig(path=str(lora_dir)))

    assert "blocks.0.attn.qkv_proj" in application.applied_modules
    assert application.skipped_non_targets == 0
    assert application.skipped_incomplete == 0

    qkv = transformer.blocks[0].attn.qkv_proj.weight.data
    assert not torch.allclose(qkv, torch.zeros_like(qkv))
    assert not torch.allclose(qkv[0:4], qkv[4:8])
    assert not torch.allclose(qkv[4:8], qkv[8:12])


def test_flux3_lora_fused_qkv_under_tp_strict_raises(tmp_path: Path) -> None:
    lora_dir = tmp_path / "adapter"
    lora_dir.mkdir()
    _save_qkv_lora(lora_dir / "adapter.safetensors")

    transformer = TinyFlux3FusedQKVUnderTPTransformer()
    with pytest.raises(ValueError, match="fused QKV output spans with TP"):
        apply_flux3_runtime_lora(transformer, RuntimeLoRAConfig(path=str(lora_dir)))


def test_flux3_lora_fused_qkv_under_tp_non_strict_skips(tmp_path: Path) -> None:
    lora_dir = tmp_path / "adapter"
    lora_dir.mkdir()
    _save_qkv_lora(lora_dir / "adapter.safetensors")

    transformer = TinyFlux3FusedQKVUnderTPTransformer()
    application = apply_flux3_runtime_lora(
        transformer, RuntimeLoRAConfig(path=str(lora_dir), strict=False)
    )
    assert "blocks.0.attn.qkv_proj" not in application.applied_modules
    assert application.skipped_non_targets == 3  # q, k, v each counted
    qkv = transformer.blocks[0].attn.qkv_proj.weight.data
    assert torch.allclose(qkv, torch.zeros_like(qkv))  # left untouched


def test_flux3_lora_applies_ordinary_tp_sharded_target(tmp_path: Path) -> None:
    lora_dir = tmp_path / "adapter"
    lora_dir.mkdir()
    a = torch.randn(1, 4)
    b = torch.randn(4, 1)  # global (unsharded) out_features for to_q
    tensors = {
        "dit.blocks.0.q_proj.lora_A.weight": a,
        "dit.blocks.0.q_proj.lora_B.weight": b,
    }
    save_file(tensors, str(lora_dir / "adapter.safetensors"))

    transformer = TinyFlux3Transformer()
    transformer.blocks[0].attn.to_q = _TPColumnLinear(4, 2)  # this rank's local half (out=2)
    transformer.blocks[0].attn.to_q.weight.data.zero_()

    apply_flux3_runtime_lora(transformer, RuntimeLoRAConfig(path=str(lora_dir), strict=False))

    b_local = b[:2, :]
    expected = (b_local @ a).to(torch.float32)
    torch.testing.assert_close(transformer.blocks[0].attn.to_q.weight.data, expected)


def test_key_map_applies_to_standard_and_gate_up_targets(tmp_path):
    from safetensors.torch import load_file

    original = tmp_path / "original.safetensors"
    renamed = tmp_path / "renamed.safetensors"
    _save_bfl_native_lora(original, "dit.")
    save_file({k.replace("blocks.", "custom."): v for k, v in load_file(original).items()}, renamed)
    expected, actual = TinyFlux3Transformer(), TinyFlux3Transformer()
    apply_flux3_runtime_lora(expected, RuntimeLoRAConfig(path=str(original)))
    applied = apply_flux3_runtime_lora(
        actual, RuntimeLoRAConfig(path=str(renamed), key_map={"custom": "blocks"})
    )
    assert len(applied.applied_modules) == 6
    for key, weight in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], weight)


def test_fuse_qkv_false_is_honored(tmp_path):
    path = tmp_path / "adapter.safetensors"
    _save_qkv_lora(path)
    transformer = TinyFlux3FusedQKVTransformer()
    with pytest.raises(ValueError, match="not found"):
        apply_flux3_runtime_lora(transformer, RuntimeLoRAConfig(path=str(path), fuse_qkv=False))
    assert torch.count_nonzero(transformer.blocks[0].attn.qkv_proj.weight) == 0


def test_lora_validation_failure_does_not_partially_merge(tmp_path):
    path = tmp_path / "adapter.safetensors"
    _save_qkv_lora(
        path,
        extra={
            "dit.blocks.0.mlp_in.lora_A.weight": torch.ones(2, 5),
            "dit.blocks.0.mlp_in.lora_B.weight": torch.ones(8, 2),
        },
    )
    transformer = TinyFlux3Transformer()
    before = {k: v.clone() for k, v in transformer.state_dict().items()}
    with pytest.raises(ValueError, match="gate/up input mismatch"):
        apply_flux3_runtime_lora(transformer, RuntimeLoRAConfig(path=str(path)))
    for key, value in transformer.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert not any(getattr(m, "_trtllm_runtime_lora_fused", False) for m in transformer.modules())


@pytest.mark.parametrize(
    "components,strict", [(["vae"], True), (["vae"], False), (["transformer", "transformer"], True)]
)
def test_pipeline_lora_component_selection(components, strict):
    from types import SimpleNamespace

    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import Flux3Pipeline

    pipeline = Flux3Pipeline.__new__(Flux3Pipeline)
    nn.Module.__init__(pipeline)
    pipeline._runtime_lora_applications = []
    pipeline.pipeline_config = SimpleNamespace(
        runtime_lora=RuntimeLoRAConfig(
            path="must-not-be-read.safetensors", target_components=components, strict=strict
        )
    )
    pipeline.transformer = TinyFlux3Transformer()
    if strict:
        with pytest.raises(ValueError, match="components"):
            pipeline._setup_runtime_lora()
    else:
        pipeline._setup_runtime_lora()
    assert pipeline._runtime_lora_applications == []


def test_lora_rejects_odd_packed_gate_up_rows(tmp_path):
    path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "dit.blocks.0.mlp_in.lora_A.weight": torch.ones(1, 4),
            "dit.blocks.0.mlp_in.lora_B.weight": torch.ones(9, 1),
        },
        path,
    )
    transformer = TinyFlux3Transformer()
    with pytest.raises(ValueError, match="packed gate/up shape"):
        apply_flux3_runtime_lora(transformer, RuntimeLoRAConfig(path=str(path)))
    assert torch.count_nonzero(transformer.blocks[0].ff.gate_up_proj.weight) == 0


def test_diffusers_lora_names_match_bfl_delta(tmp_path):
    """Both supported LoRA naming schemes resolve to identical single-block updates."""
    from safetensors.torch import load_file

    bfl = tmp_path / "bfl.safetensors"
    diffusers = tmp_path / "diffusers.safetensors"
    _save_bfl_native_lora(bfl, "dit.")
    tensors = {
        key.replace("blocks.", "single_blocks."): value for key, value in load_file(bfl).items()
    }
    save_file(tensors, bfl)
    names = {
        "q_proj": "attn.to_q",
        "k_proj": "attn.to_k",
        "v_proj": "attn.to_v",
        "attn_out": "attn.to_out.0",
        "mlp_in": "ff.linear_in",
        "mlp_out": "ff.linear_out",
    }
    renamed = {}
    for key, value in tensors.items():
        key = key.replace("dit.single_blocks.", "transformer.single_transformer_blocks.")
        for source, target in names.items():
            key = key.replace(f".0.{source}.", f".0.{target}.")
        renamed[key] = value
    save_file(renamed, diffusers)
    expected, actual = TinyFlux3Transformer(), TinyFlux3Transformer()
    for model in (expected, actual):
        model.single_blocks = model.blocks
        del model.blocks
    apply_flux3_runtime_lora(expected, RuntimeLoRAConfig(path=str(bfl)))
    application = apply_flux3_runtime_lora(actual, RuntimeLoRAConfig(path=str(diffusers)))
    assert len(application.applied_modules) == 6
    for key, weight in expected.state_dict().items():
        assert weight.count_nonzero() > 0, key
        torch.testing.assert_close(actual.state_dict()[key], weight)
