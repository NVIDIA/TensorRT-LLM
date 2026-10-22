# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wan FP8 paths: TRTLLM per-tensor FP8 attention on the trtllm-gen dense FP8 library (SM107),
static-FP8 WanBlock fusions against the unfused block, and the enable_fp8_block_fusions knob (CPU).
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.fmha.trtllm_gen_dense_fp8 import TrtllmGenDenseFp8Fmha
from tensorrt_llm._torch.modules.linear import Linear

# Importing models applies the pipeline registration decorators.
from tensorrt_llm._torch.visual_gen import models  # noqa: F401
from tensorrt_llm._torch.visual_gen.attention.attention import Attention, QKVMode
from tensorrt_llm._torch.visual_gen.attention.backends.cudnn import CuDNNAttention
from tensorrt_llm._torch.visual_gen.attention.backends.utils import create_attention
from tensorrt_llm._torch.visual_gen.attention.backends.vanilla import VanillaAttention
from tensorrt_llm._torch.visual_gen.config import (
    DiffusionModelConfig,
    DiffusionPipelineConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import (
    WanBlock,
    WanTransformer3DModel,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig, VisualGenArgs

_FP8_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 89, reason="static FP8 GEMMs need SM89+"
)

E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0
HEADS, HEAD_DIM, FFN = 24, 128, 8192
HIDDEN = HEADS * HEAD_DIM
INPUT_SCALE = 8.0 / E4M3_MAX
WEIGHT_SCALE = 0.1 / E4M3_MAX


_TRTLLM_FP8 = QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8")
_SM107_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="TRTLLM dense FP8 attention runs on SM107",
)


def _cosine(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _config(backend, quant_attention, fp8=True):
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(
            num_attention_heads=HEADS,
            attention_head_dim=HEAD_DIM,
            ffn_dim=FFN,
            eps=1e-6,
            cross_attn_norm=True,
        ),
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8) if fp8 else QuantConfig(),
        attention=AttentionConfig(backend=backend, quant_attention_config=quant_attention),
    )
    config.attention_metadata_state = create_attention_metadata_state()
    return config


def _block(config):
    """A WanBlock with seeded weights: blocks built from equal configs are identical."""
    block = WanBlock(config, _layer_idx=0).cuda()
    g = torch.Generator(device="cuda").manual_seed(0)

    def randn(shape, scale):
        return torch.randn(shape, device="cuda", generator=g) * scale

    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, Linear):
                w = randn(module.weight.shape, 0.02)
                if module.weight.dtype == E4M3:
                    module.weight.copy_((w / WEIGHT_SCALE).to(E4M3))
                    module.weight_scale.fill_(WEIGHT_SCALE)
                    module.input_scale.fill_(INPUT_SCALE)
                    module.inv_input_scale.fill_(1.0 / INPUT_SCALE)
                else:
                    module.weight.copy_(w)
                if module.bias is not None:
                    module.bias.copy_(randn(module.bias.shape, 0.02))
        for name, p in block.named_parameters():
            if "norm" in name and p.dtype != E4M3:
                p.copy_(
                    1.0 + randn(p.shape, 0.1) if name.endswith("weight") else randn(p.shape, 0.1)
                )
        block.scale_shift_table.copy_(randn(block.scale_shift_table.shape, 0.1))
    for module in block.modules():
        if isinstance(module, Linear):
            module.post_load_weights()
    return block


def _inputs(batch, seq, text_len=64):
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(batch, seq, HIDDEN, device="cuda", generator=g).bfloat16()
    text = torch.randn(batch, text_len, HIDDEN, device="cuda", generator=g).bfloat16()
    temb = (torch.randn(batch, 6, HIDDEN, device="cuda", generator=g) * 0.1).bfloat16()
    pos = torch.arange(seq, dtype=torch.float64)
    inv = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM))
    angles = torch.outer(pos, inv)
    cos = angles.cos().repeat_interleave(2, dim=-1).float().view(1, seq, 1, HEAD_DIM).cuda()
    sin = angles.sin().repeat_interleave(2, dim=-1).float().view(1, seq, 1, HEAD_DIM).cuda()
    return x, text, temb, cos, sin


@_FP8_GPU
@pytest.mark.parametrize(
    "backend,quant_attention",
    [
        ("VANILLA", None),
        pytest.param(
            "TRTLLM",
            _TRTLLM_FP8,
            marks=_SM107_ONLY,
        ),
    ],
)
def test_fused_block_matches_unfused(backend, quant_attention):
    block = _block(_config(backend, quant_attention))
    unfused = _block(_config(backend, quant_attention))
    assert block.plan_fp8_fusions()
    assert block._fp8_block and block.ffn._use_fp8_gelu_fusion
    fp8_attention_out = block.attn1._fp8_qkv_producer
    assert (block.attn1._fp8_out_scale is not None) == fp8_attention_out

    # Batch 2 gives each batch its own modulation rows; the compile test covers batch 1.
    x, text, temb, cos, sin = _inputs(2, 1560)
    with torch.no_grad():
        fused = block(x, text, temb, cos, sin)
        reference = unfused(x, text, temb, cos, sin)
    assert fused.shape == reference.shape and fused.dtype == torch.bfloat16
    assert _cosine(fused, reference) > 0.999


@_FP8_GPU
@pytest.mark.parametrize(
    "backend,quant_attention,fullgraph",
    [
        ("VANILLA", None, True),
        # The TRTLLM FP8 recipe runs cross-attention on cuDNN FP8, whose backend keeps its
        # fused-QKV check outside the graph; the fused block itself traces without breaks.
        pytest.param(
            "TRTLLM",
            _TRTLLM_FP8,
            False,
            marks=_SM107_ONLY,
        ),
    ],
)
def test_fused_block_compiles(backend, quant_attention, fullgraph):
    block = _block(_config(backend, quant_attention))
    assert block.plan_fp8_fusions()
    x, text, temb, cos, sin = _inputs(1, 1560)
    with torch.no_grad():
        eager = block(x, text, temb, cos, sin)
        compiled = torch.compile(block, fullgraph=fullgraph)(x, text, temb, cos, sin)
    assert compiled.shape == eager.shape
    assert _cosine(compiled, eager) > 0.999


@_FP8_GPU
def test_per_token_temb_takes_the_unfused_path():
    block = _block(_config("VANILLA", None))
    unfused = _block(_config("VANILLA", None))
    assert block.plan_fp8_fusions()
    x, text, temb, cos, sin = _inputs(1, 256)
    temb_per_token = temb[:, None].expand(1, 256, 6, HIDDEN).contiguous()
    with torch.no_grad():
        fused_plan = block(x, text, temb_per_token, cos, sin)
        reference = unfused(x, text, temb_per_token, cos, sin)
    assert _cosine(fused_plan, reference) > 0.999


@_FP8_GPU
def test_bf16_block_is_not_planned():
    block = _block(_config("VANILLA", None, fp8=False))
    assert not block.plan_fp8_fusions(log=True)
    assert not block._fp8_block


# ---------------------------------------------------------------------------
# TRTLLM per-tensor FP8 attention on the trtllm-gen dense FP8 library.
# ---------------------------------------------------------------------------


def _sdpa(q, k, v):
    """[B, S, H, D] reference in FP32."""
    out = F.scaled_dot_product_attention(*(t.float().transpose(1, 2) for t in (q, k, v)))
    return out.transpose(1, 2)


def _qkv(batch, seq, heads, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [
        torch.randn(batch, seq, heads, HEAD_DIM, device="cuda", generator=g).bfloat16()
        for _ in range(3)
    ]


def _backend(backend, heads, quant=_TRTLLM_FP8):
    return create_attention(
        backend=backend,
        layer_idx=0,
        num_heads=heads,
        head_dim=HEAD_DIM,
        dtype=torch.bfloat16,
        attention_config=AttentionConfig(backend=backend, quant_attention_config=quant),
        attention_metadata_state=create_attention_metadata_state(),
    )


def _model_config(backend, quant=_TRTLLM_FP8, heads=12):
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(
            hidden_size=heads * HEAD_DIM,
            num_attention_heads=heads,
            attention_head_dim=HEAD_DIM,
            eps=1e-6,
        ),
        attention=AttentionConfig(backend=backend, quant_attention_config=quant),
    )
    config.attention_metadata_state = create_attention_metadata_state()
    return config


def _wan_self_attention(backend, quant=_TRTLLM_FP8, heads=12, **kwargs):
    attn = Attention(
        hidden_size=heads * HEAD_DIM,
        num_attention_heads=heads,
        qkv_mode=QKVMode.FUSE_QKV,
        qk_norm=True,
        qk_norm_mode="full",
        fuse_qk_norm_rope=True,
        config=_model_config(backend, quant, heads),
        enable_sequence_parallel=False,
        **kwargs,
    ).cuda()
    g = torch.Generator(device="cuda").manual_seed(7)
    with torch.no_grad():
        for name, p in attn.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, device="cuda", generator=g))
            else:
                p.copy_(0.02 * torch.randn(p.shape, device="cuda", generator=g))
    return attn


def _wan_rope(seq):
    """Wan-style [1, S, 1, D] cos/sin tables with interleaved pairs."""
    pos = torch.arange(seq, dtype=torch.float64)
    inv = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM))
    angles = torch.outer(pos, inv)
    cos = angles.cos().repeat_interleave(2, dim=-1).float()
    sin = angles.sin().repeat_interleave(2, dim=-1).float()
    return cos.view(1, seq, 1, HEAD_DIM).cuda(), sin.view(1, seq, 1, HEAD_DIM).cuda()


@_SM107_ONLY
@pytest.mark.parametrize("batch,seq,heads", [(1, 1560, 10), (2, 1560, 40)])
def test_bf16_input_matches_sdpa_and_cudnn(batch, seq, heads):
    trtllm = _backend("TRTLLM", heads)
    assert any(isinstance(f, TrtllmGenDenseFp8Fmha) for f in trtllm._fmha_manager.fmha_libs)
    q, k, v = _qkv(batch, seq, heads)
    out = trtllm.forward(q=q, k=k, v=v, batch_size=batch, seq_len=seq)
    assert out.shape == (batch, seq, heads * HEAD_DIM) and out.dtype == torch.bfloat16
    ref = _sdpa(q, k, v).reshape(batch, seq, -1)
    assert _cosine(out, ref) > 0.99

    cudnn = _backend("CUDNN", heads)
    out_cudnn = cudnn.forward(q=q, k=k, v=v, batch_size=batch, seq_len=seq)
    assert _cosine(out, out_cudnn.reshape(batch, seq, -1)) > 0.998


@_SM107_ONLY
@pytest.mark.parametrize("out_fp8", [False, True])
def test_prequantized_matches_sdpa(out_fp8):
    batch, seq, heads = 2, 1560, 40
    trtllm = _backend("TRTLLM", heads)
    assert trtllm.accepts_fp8_qkv()
    q, k, v = _qkv(batch, seq, heads, seed=1)
    qk_mul = math.sqrt(math.log2(math.e) / math.sqrt(HEAD_DIM))
    v_descale = (v.float().abs().amax() / E4M3_MAX).reshape(1)
    q8, k8 = ((t.float() * qk_mul).to(E4M3) for t in (q, k))
    v8 = (v.float() / v_descale).to(E4M3)
    ref = _sdpa(q, k, v)
    out_scale = (ref.abs().amax() / E4M3_MAX).reshape(1) if out_fp8 else None

    out = trtllm.forward_fp8(q8, k8, v8, v_descale, batch, seq, out_scale=out_scale)
    assert out.shape == (batch, seq, heads, HEAD_DIM)
    assert out.dtype == (E4M3 if out_fp8 else torch.bfloat16)
    dequant = out.float() * out_scale if out_fp8 else out.float()
    assert _cosine(dequant, ref) > 0.99


@_SM107_ONLY
def test_attention_module_uses_fp8_producer():
    batch, seq, heads = 2, 1560, 12
    attn = _wan_self_attention("TRTLLM", heads=heads)
    assert attn._fp8_qkv_producer
    hidden = torch.randn(batch, seq, heads * HEAD_DIM, device="cuda").bfloat16()
    freqs = _wan_rope(seq)
    with torch.no_grad():
        out = attn(hidden, freqs=freqs)
        # Same module through the BF16 QK norm + RoPE and the backend's own quantization.
        attn._fp8_qkv_producer = False
        out_backend_quant = attn(hidden, freqs=freqs)
        vanilla = _wan_self_attention("VANILLA", quant=None, heads=heads)
        vanilla.load_state_dict(attn.state_dict())
        out_bf16 = vanilla(hidden, freqs=freqs)
    assert out.shape == out_bf16.shape
    assert _cosine(out, out_backend_quant) > 0.995
    assert _cosine(out, out_bf16) > 0.99


@_SM107_ONLY
def test_attention_module_fp8_producer_under_compile():
    batch, seq, heads = 1, 1560, 12
    attn = _wan_self_attention("TRTLLM", heads=heads)
    hidden = torch.randn(batch, seq, heads * HEAD_DIM, device="cuda").bfloat16()
    freqs = _wan_rope(seq)
    with torch.no_grad():
        eager = attn(hidden, freqs=freqs)
        compiled = torch.compile(attn, fullgraph=True)(hidden, freqs=freqs)
    torch.testing.assert_close(compiled, eager, rtol=0.02, atol=0.02)
    assert _cosine(compiled, eager) > 0.9999


@_SM107_ONLY
@pytest.mark.parametrize(
    "backend,kwargs",
    [
        ("TRTLLM", {"qk_norm_mode": "per_head"}),
        ("TRTLLM", {"fuse_qk_norm_rope": False}),
        ("CUDNN", {}),
    ],
)
def test_fp8_producer_off(backend, kwargs):
    defaults = dict(
        qkv_mode=QKVMode.FUSE_QKV, qk_norm=True, qk_norm_mode="full", fuse_qk_norm_rope=True
    )
    defaults.update(kwargs)
    attn = Attention(
        hidden_size=12 * HEAD_DIM,
        num_attention_heads=12,
        config=_model_config(backend),
        enable_sequence_parallel=False,
        **defaults,
    )
    assert not attn._fp8_qkv_producer


@_SM107_ONLY
def test_cross_attention_runs_on_cudnn_fp8(monkeypatch):
    def cross_attention():
        return Attention(
            hidden_size=12 * HEAD_DIM,
            num_attention_heads=12,
            qkv_mode=QKVMode.SEPARATE_QKV,
            qk_norm=True,
            config=_model_config("TRTLLM"),
            enable_sequence_parallel=False,
        )

    assert isinstance(cross_attention().attn, CuDNNAttention)
    # PrimTS serves the recipe when enabled; cross-attention keeps the VANILLA fallback.
    monkeypatch.setenv("TLLM_FMHA_LIBS", "+prims_ts")
    assert isinstance(cross_attention().attn, VanillaAttention)


@_SM107_ONLY
def test_dense_fp8_library_needs_the_recipe():
    trtllm = _backend("TRTLLM", 10, quant=None)
    assert not any(isinstance(f, TrtllmGenDenseFp8Fmha) for f in trtllm._fmha_manager.fmha_libs)
    assert not trtllm.accepts_fp8_qkv()


# ---------------------------------------------------------------------------
# Wan pipeline_config key enable_fp8_block_fusions.
# ---------------------------------------------------------------------------


@pytest.fixture
def wan_checkpoint(tmp_path: Path) -> str:
    """Only the metadata that pipeline detection and configuration read."""
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "WanPipeline",
                "transformer": ["diffusers", "WanTransformer3DModel"],
                "vae": ["diffusers", "AutoencoderKLWan"],
            }
        ),
        encoding="utf-8",
    )
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(
        json.dumps({"_class_name": "WanTransformer3DModel"}), encoding="utf-8"
    )
    return str(tmp_path)


@pytest.mark.cpu_only
def test_block_fusions_default_off_and_override_reaches_the_model_config(
    wan_checkpoint: str,
) -> None:
    default_args = VisualGenArgs(model=wan_checkpoint)
    loader = PipelineLoader(default_args, device="cpu")
    assert loader._resolve_pipeline_config(wan_checkpoint) == {"enable_fp8_block_fusions": False}

    args = VisualGenArgs(model=wan_checkpoint, pipeline_config={"enable_fp8_block_fusions": True})
    resolved = PipelineLoader(args, device="cpu")._resolve_pipeline_config(wan_checkpoint)
    assert resolved == {"enable_fp8_block_fusions": True}

    config = DiffusionPipelineConfig.from_pretrained(
        wan_checkpoint, args=args, pipeline_config=resolved, device="cpu"
    )
    assert config.extra_attrs["enable_fp8_block_fusions"] is True


@pytest.mark.cpu_only
def test_unknown_pipeline_config_key_is_rejected(wan_checkpoint: str) -> None:
    args = VisualGenArgs(model=wan_checkpoint, pipeline_config={"enable_fp8_block_fusion": True})
    with pytest.raises(ValueError, match="Unknown pipeline_config keys for WanPipeline"):
        PipelineLoader(args, device="cpu")._resolve_pipeline_config(wan_checkpoint)


@pytest.mark.cpu_only
@pytest.mark.parametrize("enabled", [False, True])
def test_post_load_weights_plans_block_fusions_only_when_enabled(monkeypatch, enabled) -> None:
    planned = []
    monkeypatch.setattr(
        WanBlock, "plan_fp8_fusions", lambda self, log=False: planned.append(self) or False
    )
    heads, head_dim = 2, 128
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(
            attention_head_dim=head_dim,
            cross_attn_norm=True,
            eps=1e-6,
            ffn_dim=512,
            freq_dim=256,
            in_channels=16,
            num_attention_heads=heads,
            num_layers=2,
            out_channels=16,
            patch_size=[1, 2, 2],
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=1024,
            text_dim=256,
            hidden_size=heads * head_dim,
            torch_dtype="bfloat16",
        ),
        quant_config=QuantConfig(),
        skip_create_weights_in_init=False,
    )
    config.extra_attrs["enable_fp8_block_fusions"] = enabled
    model = WanTransformer3DModel(model_config=config)
    model.post_load_weights()
    assert len(planned) == (2 if enabled else 0)
