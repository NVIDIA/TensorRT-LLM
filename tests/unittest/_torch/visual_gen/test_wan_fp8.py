# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wan FP8 paths: TRTLLM per-tensor FP8 attention on the trtllm-gen dense FP8 library (SM107),
static-FP8 WanBlock fusions against the unfused block (with the FFN residual handoff, the
static-FP8 cross-attention and the output-head modulation), and the enable_fp8_block_fusions knob
and its load-time plans (CPU).
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.fmha.trtllm_gen_dense_fp8 import (
    PREQUANTIZED_SKIP_CORRECTION_LOG2,
    TrtllmGenDenseFp8Fmha,
)
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode

# Importing models applies the pipeline registration decorators.
from tensorrt_llm._torch.visual_gen import models  # noqa: F401
from tensorrt_llm._torch.visual_gen.attention.attention import (
    Attention,
    QKVMode,
    _needs_cudnn_fp8_fallback,
)
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
    WanFfnResidual,
    WanTransformer3DModel,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger as trtllm_logger
from tensorrt_llm.mapping import Mapping
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
# Gates near 1, so an error in any branch shows in the block's update instead of being damped.
TEMB_STD = 1.0


_TRTLLM_FP8 = QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8")
_SM107_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="TRTLLM dense FP8 attention runs on SM107",
)


def _cosine(a, b):
    """Cosine similarity of two tensors in fp32."""
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _rel_l2(a, b):
    """Relative L2 error of a against b in fp32; unlike cosine it sees a uniform scale error."""
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def _config(backend, quant_attention, fp8=True):
    """Model config for a small Wan block."""
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


def _block(config, seed=0):
    """A WanBlock with seeded weights: blocks built from equal configs and seeds are identical."""
    block = WanBlock(config, _layer_idx=0).cuda()
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(shape, scale):
        """Seeded normal tensor on the GPU."""
        return torch.randn(shape, device="cuda", generator=g) * scale

    with torch.no_grad():
        linears = [module for module in block.modules() if isinstance(module, Linear)]
        for idx, module in enumerate(linears):
            w = randn(module.weight.shape, 0.02)
            if module.weight.dtype == E4M3:
                # A distinct input scale per Linear, so using another layer's scale shows up.
                input_scale = INPUT_SCALE * (1.0 + 0.125 * idx)
                module.weight.copy_((w / WEIGHT_SCALE).to(E4M3))
                module.weight_scale.fill_(WEIGHT_SCALE)
                module.input_scale.fill_(input_scale)
                module.inv_input_scale.fill_(1.0 / input_scale)
            else:
                module.weight.copy_(w)
            if module.bias is not None:
                # Biases large enough that dropping, doubling or swapping one is visible.
                module.bias.copy_(randn(module.bias.shape, 0.5))
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


def _inputs(batch, seq, text_len=64, temb_std=TEMB_STD):
    """Seeded hidden states, text, timestep embedding and RoPE tables."""
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(batch, seq, HIDDEN, device="cuda", generator=g).bfloat16()
    text = torch.randn(batch, text_len, HIDDEN, device="cuda", generator=g).bfloat16()
    temb = (torch.randn(batch, 6, HIDDEN, device="cuda", generator=g) * temb_std).bfloat16()
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
    """The fused FP8 block matches the unfused block."""
    block = _block(_config(backend, quant_attention))
    unfused = _block(_config(backend, quant_attention))
    state_keys = set(block.state_dict())
    assert block.plan_fp8_fusions()
    assert block._fp8_block and block.ffn._use_fp8_gelu_fusion
    assert block.attn1._fp8_output == block.attn1._fp8_qkv_producer
    # The lossy deferred attention rescale is off by default and only the planner turns it on.
    lib = getattr(block.attn1.attn, "_prequantized_fp8_lib", None)
    if lib is not None:
        assert lib._prequantized_skip_correction == PREQUANTIZED_SKIP_CORRECTION_LOG2
        assert unfused.attn1.attn._prequantized_fp8_lib._prequantized_skip_correction == 0.0
    # Static-FP8 cross-attention where the cross-attention backend has static FP8 I/O (cuDNN FP8).
    assert block.attn2.static_fp8_cross_attention == block.attn2.attn.accepts_static_fp8_io()
    # Planning registers no new parameters or buffers (the to_out scales stay to_out's own).
    assert set(block.state_dict()) == state_keys

    # Batch 2 gives each batch its own modulation rows; the compile test covers batch 1.
    x, text, temb, cos, sin = _inputs(2, 1560)
    with torch.no_grad():
        fused = block(x, text, temb, cos, sin)
        reference = unfused(x, text, temb, cos, sin)
    assert fused.shape == reference.shape and fused.dtype == torch.bfloat16
    assert _cosine(fused, reference) > 0.999
    # The residual stream dominates the output; compare the block's update itself. With
    # gates near 1, a row kernel or epilogue reading another Linear's scale moves the update
    # well past this bound; only a wrong attn2.to_q scale does not, as norm_q normalizes it.
    update, reference_update = fused.float() - x.float(), reference.float() - x.float()
    assert _cosine(update, reference_update) > 0.99
    assert _rel_l2(update, reference_update) < 0.06


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
    """The fused FP8 block under torch.compile matches eager."""
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
    """A per-token timestep embedding falls back to the unfused path."""
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
@pytest.mark.parametrize(
    "case,blocker",
    [("bf16", "not static FP8"), ("expand_timesteps", "expand_timesteps")],
)
def test_block_is_not_planned(case, blocker):
    """Blocks the fused path does not serve stay unfused and name the reason."""
    config = _config("VANILLA", None, fp8=case != "bf16")
    if case == "expand_timesteps":
        config.pretrained_config.expand_timesteps = True
    block = _block(config)
    assert not block.plan_fp8_fusions(log=True)
    assert not block._fp8_block
    assert any(blocker in reason for reason in block.fp8_fusion_blockers)


@pytest.mark.cpu_only
def test_row_parallel_linear_without_reduction_produces_partial_output(monkeypatch):
    """A ROW tensor-parallel Linear whose reduction a wrapper runs later (token-sharded TP) still
    reports partial outputs, and forward_without_bias rejects it."""
    # A plain MPI-style Mapping: multi-GPU test modules collected in the same session set
    # TLLM_DISABLE_MPI at import, which would make Mapping build a DeviceMesh here.
    monkeypatch.delenv("TLLM_DISABLE_MPI", raising=False)
    linear = Linear(
        256,
        128,
        bias=True,
        dtype=torch.bfloat16,
        mapping=Mapping(world_size=2, tp_size=2, rank=0),
        tensor_parallel_mode=TensorParallelMode.ROW,
        reduce_output=False,
    )
    assert linear.produces_partial_output()
    with pytest.raises(ValueError, match="tensor-parallel"):
        linear.forward_without_bias(torch.zeros(4, 128, dtype=torch.bfloat16))


def _bitwise(a, b):
    assert a.dtype == b.dtype and torch.equal(a, b)


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
@pytest.mark.parametrize("batch", [1, 2])
def test_ffn_residual_handoff_is_bitwise(backend, quant_attention, batch):
    """The FFN residual applied in the next block's AdaLN1 pass equals its own residual pass."""
    config = _config(backend, quant_attention)
    # Different weights, so a residual applied with the wrong block's bias or gate shows up.
    first, second = _block(config, seed=1), _block(config, seed=2)
    assert first.plan_fp8_fusions() and second.plan_fp8_fusions()
    x, text, temb, cos, sin = _inputs(batch, 1560)
    with torch.no_grad():
        reference = second(first(x, text, temb, cos, sin), text, temb, cos, sin)
        x_pre, residual = first(x, text, temb, cos, sin, defer_ffn_residual=True)
        assert isinstance(residual, WanFfnResidual)
        handed_off = second(x_pre, text, temb, cos, sin, ffn_residual=residual)
    _bitwise(handed_off, reference)


@_FP8_GPU
def test_ffn_residual_reaches_a_block_off_the_fp8_path():
    """A block that does not take the static-FP8 path (per-token temb) applies the residual first."""
    config = _config("VANILLA", None)
    # Different weights, so a residual applied with the wrong block's bias or gate shows up.
    first, second = _block(config, seed=1), _block(config, seed=2)
    assert first.plan_fp8_fusions() and second.plan_fp8_fusions()
    x, text, temb, cos, sin = _inputs(1, 256)
    temb_per_token = temb[:, None].expand(1, 256, 6, HIDDEN).contiguous()
    assert not second.takes_fp8_block_path(temb_per_token, x.shape[0])
    with torch.no_grad():
        x_pre, residual = first(x, text, temb, cos, sin, defer_ffn_residual=True)
        x_full = first(x, text, temb, cos, sin)
        handed_off = second(x_pre, text, temb_per_token, cos, sin, ffn_residual=residual)
        reference = second(x_full, text, temb_per_token, cos, sin)
    _bitwise(handed_off, reference)


@_FP8_GPU
def test_ffn_residual_handoff_compiles():
    """The handoff traces under torch.compile(fullgraph=True) and stays close to eager (the
    compiled graph may round the eager-only ops differently)."""
    config = _config("VANILLA", None)
    # Different weights, so a residual applied with the wrong block's bias or gate shows up.
    first, second = _block(config, seed=1), _block(config, seed=2)
    assert first.plan_fp8_fusions() and second.plan_fp8_fusions()
    x, text, temb, cos, sin = _inputs(1, 1560)

    def two_blocks(first_block, second_block, x):
        x_pre, residual = first_block(x, text, temb, cos, sin, defer_ffn_residual=True)
        return second_block(x_pre, text, temb, cos, sin, ffn_residual=residual)

    with torch.no_grad():
        eager = two_blocks(first, second, x)
        compiled = two_blocks(
            torch.compile(first, fullgraph=True), torch.compile(second, fullgraph=True), x
        )
    assert _cosine(compiled, eager) > 0.999


@_FP8_GPU
def test_unplanned_block_does_not_defer():
    block = _block(_config("VANILLA", None, fp8=False))
    assert not block.plan_fp8_fusions()
    x, text, temb, cos, sin = _inputs(1, 256)
    with torch.no_grad():
        out = block(x, text, temb, cos, sin, defer_ffn_residual=True)
    assert isinstance(out, torch.Tensor)


@_FP8_GPU
@pytest.mark.parametrize("batch", [1, 2])
def test_output_modulation_kernel_matches_eager(batch):
    """The output head's one-pass modulation on the model's chunked [B, 1, D] scale and shift."""
    model = WanTransformer3DModel.__new__(WanTransformer3DModel)
    g = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(batch, 3120, HIDDEN, device="cuda", generator=g).bfloat16()
    table = torch.randn(1, 2, HIDDEN, device="cuda", generator=g) * 0.1
    temb = (torch.randn(batch, HIDDEN, device="cuda", generator=g) * 0.1).bfloat16()
    shift, scale = (table + temb.unsqueeze(1)).chunk(2, dim=1)
    assert not scale.reshape(batch, -1).is_contiguous() or batch == 1
    assert model._modulation_kernel_applies(x, scale, shift, torch.bfloat16)
    # Per-token rows and non-fp32 modulation keep the eager ops.
    per_token = torch.zeros(batch, x.shape[1], HIDDEN, device="cuda")
    assert not model._modulation_kernel_applies(x, per_token, per_token, torch.bfloat16)
    assert not model._modulation_kernel_applies(x, scale.bfloat16(), shift, torch.bfloat16)
    fused = model._modulate_output(x, scale, shift, torch.bfloat16)
    _bitwise(fused, (x * (1 + scale) + shift).to(torch.bfloat16))


# ---------------------------------------------------------------------------
# TRTLLM per-tensor FP8 attention on the trtllm-gen dense FP8 library.
# ---------------------------------------------------------------------------


def _sdpa(q, k, v):
    """[B, S, H, D] reference in FP32."""
    out = F.scaled_dot_product_attention(*(t.float().transpose(1, 2) for t in (q, k, v)))
    return out.transpose(1, 2)


def _qkv(batch, seq, heads, seed=0):
    """Seeded BF16 [B, S, H, D] Q, K and V."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [
        torch.randn(batch, seq, heads, HEAD_DIM, device="cuda", generator=g).bfloat16()
        for _ in range(3)
    ]


def _backend(backend, heads, quant=_TRTLLM_FP8):
    """Attention backend with the given FP8 recipe."""
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
    """Model config for a Wan attention layer."""
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
    """Wan self-attention with fused QKV, QK norm and RoPE and seeded weights."""
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
    """BF16 inputs on the dense FP8 library match SDPA and cuDNN FP8."""
    trtllm = _backend("TRTLLM", heads)
    assert any(isinstance(f, TrtllmGenDenseFp8Fmha) for f in trtllm._fmha_manager.fmha_libs)
    q, k, v = _qkv(batch, seq, heads)
    out = trtllm.forward(q=q, k=k, v=v, batch_size=batch, seq_len=seq)
    assert out.shape == (batch, seq, heads * HEAD_DIM) and out.dtype == torch.bfloat16
    ref = _sdpa(q, k, v).reshape(batch, seq, -1)
    assert _cosine(out, ref) > 0.99
    assert _rel_l2(out, ref) < 0.15

    cudnn = _backend("CUDNN", heads)
    out_cudnn = cudnn.forward(q=q, k=k, v=v, batch_size=batch, seq_len=seq)
    assert _cosine(out, out_cudnn.reshape(batch, seq, -1)) > 0.998


@_SM107_ONLY
@pytest.mark.parametrize("out_fp8", [False, True])
def test_prequantized_matches_sdpa(out_fp8):
    """Prequantized Q/K/V match SDPA with BF16 and e4m3 output."""
    batch, seq, heads = 2, 1560, 40
    trtllm = _backend("TRTLLM", heads)
    assert trtllm.accepts_fp8_qkv()
    # SM107 ships the e4m3-output kernel for head_dim 128.
    assert trtllm.accepts_fp8_qkv(fp8_output=True)
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
    assert _rel_l2(dequant, ref) < 0.15


@_SM107_ONLY
def test_attention_module_uses_fp8_producer():
    """The FP8 QKV producer matches the backend's own quantization and BF16."""
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
    """The FP8 QKV producer under torch.compile matches eager."""
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
    """Per-head QK norm, unfused QK norm and cuDNN do not use the FP8 producer."""
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
    """Cross-attention with the FP8 recipe runs on cuDNN FP8."""

    def cross_attention():
        """Wan cross-attention layer on the TRTLLM backend."""
        return Attention(
            hidden_size=12 * HEAD_DIM,
            num_attention_heads=12,
            qkv_mode=QKVMode.SEPARATE_QKV,
            qk_norm=True,
            config=_model_config("TRTLLM"),
            enable_sequence_parallel=False,
        )

    attn = cross_attention()
    assert isinstance(attn.attn, CuDNNAttention)
    # cuDNN FP8 does not take key_padding_mask; masked calls go to VANILLA.
    assert isinstance(attn._masked_attn, VanillaAttention)
    # PrimTS on or off, the routing follows the dense FP8 library that runs the FP8 producer.
    monkeypatch.setenv("TLLM_FMHA_LIBS", "+prims_ts")
    assert isinstance(cross_attention().attn, CuDNNAttention)
    monkeypatch.setenv("TLLM_FMHA_LIBS", "-prims_ts")
    assert isinstance(cross_attention().attn, CuDNNAttention)
    # Without the dense FP8 library, cross-attention takes the VANILLA fallback as on Blackwell.
    monkeypatch.setenv("TLLM_FMHA_LIBS", "-trtllm_gen_dense_fp8")
    assert isinstance(cross_attention().attn, VanillaAttention)


@_SM107_ONLY
def test_masked_cross_attention_matches_vanilla():
    """Masked cross-attention under the FP8 recipe runs on VANILLA and matches it."""

    def cross_attention(backend, quant):
        """Wan cross-attention layer with seeded weights."""
        attn = Attention(
            hidden_size=12 * HEAD_DIM,
            num_attention_heads=12,
            qkv_mode=QKVMode.SEPARATE_QKV,
            qk_norm=True,
            config=_model_config(backend, quant),
            enable_sequence_parallel=False,
        ).cuda()
        g = torch.Generator(device="cuda").manual_seed(3)
        with torch.no_grad():
            for name, p in attn.named_parameters():
                scale = 0.1 if "norm" in name else 0.02
                base = 1.0 if "norm" in name else 0.0
                p.copy_(base + scale * torch.randn(p.shape, device="cuda", generator=g))
        return attn

    fp8, vanilla = cross_attention("TRTLLM", _TRTLLM_FP8), cross_attention("VANILLA", None)
    hidden = torch.randn(1, 256, 12 * HEAD_DIM, device="cuda").bfloat16()
    text = torch.randn(1, 64, 12 * HEAD_DIM, device="cuda").bfloat16()
    mask = torch.ones(1, 64, dtype=torch.bool, device="cuda")
    mask[:, 48:] = False
    with torch.no_grad():
        out = fp8(hidden, encoder_hidden_states=text, key_padding_mask=mask)
        expected = vanilla(hidden, encoder_hidden_states=text, key_padding_mask=mask)
    torch.testing.assert_close(out, expected)


@_SM107_ONLY
def test_dense_fp8_library_needs_the_recipe():
    """Without the FP8 recipe the dense FP8 library is not registered."""
    trtllm = _backend("TRTLLM", 10, quant=None)
    assert not any(isinstance(f, TrtllmGenDenseFp8Fmha) for f in trtllm._fmha_manager.fmha_libs)
    assert not trtllm.accepts_fp8_qkv()


@pytest.mark.cpu_only
@pytest.mark.parametrize("sm,sparse", [(100, False), (103, False), (107, True)])
def test_dense_fp8_library_unavailable_off_sm107_or_on_sparse_layers(monkeypatch, sm, sparse):
    """The dense FP8 library serves only SM107 and never a sparse (e.g. SkipSoftmax) layer."""
    from tensorrt_llm._torch.attention.backends.fmha import trtllm_gen_dense_fp8

    monkeypatch.setattr(trtllm_gen_dense_fp8, "get_sm_version", lambda: sm)
    attn = SimpleNamespace(
        quant_attention_config=_TRTLLM_FP8,
        sparse_params=object() if sparse else None,
        is_mla_enable=False,
        num_heads=12,
        num_kv_heads=12,
        head_dim=HEAD_DIM,
    )
    assert not TrtllmGenDenseFp8Fmha._is_available(attn)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "sm,fmha_libs,sparse,expected",
    [
        (107, None, False, True),
        (100, None, False, False),
        (103, None, False, False),
        (107, "-trtllm_gen_dense_fp8", False, False),
        # PrimTS on or off: the FP8 producer runs on the dense FP8 library either way.
        (107, "+prims_ts", False, True),
        (107, "-prims_ts", False, True),
        (107, None, True, False),
    ],
)
def test_dense_fp8_routing_follows_the_fmha_library(monkeypatch, sm, fmha_libs, sparse, expected):
    """Cross-attention and context-parallel routing follow the dense FP8 library's selection."""
    monkeypatch.setattr("tensorrt_llm._utils.get_sm_version", lambda: sm)
    if fmha_libs is None:
        monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)
    else:
        monkeypatch.setenv("TLLM_FMHA_LIBS", fmha_libs)
    config = SimpleNamespace(
        quant_attention_config=_TRTLLM_FP8,
        sparse_attention_config=object() if sparse else None,
    )
    assert _needs_cudnn_fp8_fallback(config) is expected


@_SM107_ONLY
def test_prequantized_path_defers_the_output_rescale_only_when_asked():
    """The lossy deferred output rescale is off by default and follows defer_fp8_rescale()."""
    trtllm = _backend("TRTLLM", 40)
    assert trtllm.accepts_fp8_qkv()
    lib = trtllm._prequantized_fp8_lib
    assert lib._prequantized_skip_correction == 0.0
    assert trtllm.defer_fp8_rescale(True)
    assert lib._prequantized_skip_correction == PREQUANTIZED_SKIP_CORRECTION_LOG2
    assert not trtllm.defer_fp8_rescale(False)
    assert lib._prequantized_skip_correction == 0.0


def _wan_cross_attention(backend, heads=12, quant=_TRTLLM_FP8):
    attn = Attention(
        hidden_size=heads * HEAD_DIM,
        num_attention_heads=heads,
        qkv_mode=QKVMode.SEPARATE_QKV,
        qk_norm=True,
        qk_norm_mode="full",
        config=_model_config(backend, quant, heads),
        enable_sequence_parallel=False,
    ).cuda()
    g = torch.Generator(device="cuda").manual_seed(9)
    with torch.no_grad():
        for name, p in attn.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, device="cuda", generator=g))
            else:
                p.copy_(0.02 * torch.randn(p.shape, device="cuda", generator=g))
    return attn


@_SM107_ONLY
@pytest.mark.parametrize("heads", [24, 40])
@pytest.mark.parametrize("calibrated", [False, True])
def test_static_fp8_cross_attention(heads, calibrated):
    """norm_q + static-scale e4m3 Q in one kernel feeding cuDNN FP8, against the per-call scale,
    with the bound-derived scale or a calibrated maximum."""
    attn = _wan_cross_attention("TRTLLM", heads)
    assert isinstance(attn.attn, CuDNNAttention)
    # Gaussian rows keep |norm_q(q)| near 8; a calibrated maximum of 16 leaves headroom.
    q_amax = 16.0 if calibrated else None
    state_keys = set(attn.state_dict())
    assert attn.enable_fp8_cross_attention(q_amax)
    # Enabling registers no new parameters or buffers (to_out's scale stays to_out's own).
    assert set(attn.state_dict()) == state_keys
    width = heads * HEAD_DIM
    bound = width**0.5 * attn.norm_q.weight.float().abs().max() * (1.0 + 2.0**-8) ** 2
    expected = min(bound.item(), q_amax) if calibrated else bound.item()
    torch.testing.assert_close(attn._fp8_q_descale.item(), expected / E4M3_MAX)

    g = torch.Generator(device="cuda").manual_seed(4)
    x = torch.randn(2, 1560, width, device="cuda", generator=g).bfloat16()
    text = torch.randn(2, 512, width, device="cuda", generator=g).bfloat16()
    with torch.no_grad():
        q, k, v = attn.get_qkv(x, text)
        static = attn.fp8_cross_attention(q, k, v)
        qn, kn = attn.apply_qk_norm(q, k)
        dynamic = attn._attn_impl(qn, kn, v)
    assert static.shape == dynamic.shape
    assert (static.dtype == E4M3) == attn._fp8_cross_output
    out = static.float() * attn.to_out[0].input_scale if static.dtype == E4M3 else static
    assert _cosine(out, dynamic) > 0.995


@_FP8_GPU
def test_static_fp8_cross_attention_needs_static_fp8_io():
    # VANILLA (BF16) cross-attention has no static FP8 inputs or outputs.
    attn = _wan_cross_attention("VANILLA", quant=None)
    assert not attn.enable_fp8_cross_attention()
    assert attn._fp8_q_descale is None


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
    """enable_fp8_block_fusions defaults off and an override reaches the model config."""
    default_args = VisualGenArgs(model=wan_checkpoint)
    loader = PipelineLoader(default_args, device="cpu")
    assert loader._resolve_pipeline_config(wan_checkpoint) == {
        "enable_fp8_block_fusions": False,
        "fp8_cross_attention_q_amax": "",
    }

    args = VisualGenArgs(model=wan_checkpoint, pipeline_config={"enable_fp8_block_fusions": True})
    resolved = PipelineLoader(args, device="cpu")._resolve_pipeline_config(wan_checkpoint)
    assert resolved == {"enable_fp8_block_fusions": True, "fp8_cross_attention_q_amax": ""}

    config = DiffusionPipelineConfig.from_pretrained(
        wan_checkpoint, args=args, pipeline_config=resolved, device="cpu"
    )
    assert config.extra_attrs["enable_fp8_block_fusions"] is True


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "key,value",
    [
        # A quoted "false" is rejected instead of being read as true.
        ("enable_fp8_block_fusions", "false"),
        ("enable_fp8_block_fusions", 1),
        # A non-string Q-amax path is rejected instead of being opened as a file descriptor.
        ("fp8_cross_attention_q_amax", True),
    ],
)
def test_pipeline_config_keys_take_their_own_type(wan_checkpoint: str, key, value) -> None:
    """bool and str pipeline_config keys reject values of another type at load."""
    args = VisualGenArgs(model=wan_checkpoint, pipeline_config={key: value})
    with pytest.raises(ValueError, match=f"wrong type.*{key}"):
        PipelineLoader(args, device="cpu")._resolve_pipeline_config(wan_checkpoint)


def _tiny_model(enabled: bool, num_layers: int = 2) -> WanTransformer3DModel:
    """A small Wan transformer on CPU with the block-fusion key set."""
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
            num_layers=num_layers,
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
    return WanTransformer3DModel(model_config=config)


@pytest.mark.cpu_only
@pytest.mark.parametrize("enabled", [False, True])
def test_post_load_weights_plans_block_fusions_only_when_enabled(monkeypatch, enabled) -> None:
    """post_load_weights plans block fusions only when enabled."""
    planned = []
    monkeypatch.setattr(
        WanBlock,
        "plan_fp8_fusions",
        lambda self, log=False, cross_attention_q_amax=None: planned.append(self) or False,
    )
    model = _tiny_model(enabled)
    model.post_load_weights()
    assert len(planned) == (2 if enabled else 0)


def _plan(monkeypatch, model, planned_blocks):
    def plan(self, log=False, cross_attention_q_amax=None):
        self._fp8_block = self in planned_blocks
        return self._fp8_block

    monkeypatch.setattr(WanBlock, "plan_fp8_fusions", plan)
    model.post_load_weights()


@pytest.mark.cpu_only
def test_ffn_residual_handoff_between_consecutive_planned_blocks(monkeypatch) -> None:
    model = _tiny_model(True, num_layers=4)
    blocks = list(model.blocks)
    _plan(monkeypatch, model, {blocks[0], blocks[1], blocks[3]})

    hidden = model.config.hidden_size
    per_batch_temb = torch.zeros(1, 6, hidden)
    per_token_temb = torch.zeros(1, 8, 6, hidden)
    assert model._ffn_residual_handoff(per_batch_temb, 1) == [True, False, False, False]
    assert model._ffn_residual_handoff(per_token_temb, 1) == [False] * 4
    # A temb with another batch than x keeps every block on its own residual pass.
    assert model._ffn_residual_handoff(per_batch_temb, 2) == [False] * 4


@pytest.mark.cpu_only
def test_ffn_residual_handoff_follows_compiled_blocks_only(monkeypatch) -> None:
    model = _tiny_model(True)
    _plan(monkeypatch, model, set(model.blocks))
    temb = torch.zeros(1, 6, model.config.hidden_size)

    class Compiled(torch.nn.Module):
        def __init__(self, block):
            super().__init__()
            self._orig_mod = block

    class Wrapper(torch.nn.Module):
        pass

    blocks = list(model.blocks)
    model.blocks = torch.nn.ModuleList([Compiled(b) for b in blocks])
    assert model._ffn_residual_handoff(temb, 1) == [True, False]
    model.blocks = torch.nn.ModuleList([Wrapper(), Compiled(blocks[1])])
    assert model._ffn_residual_handoff(temb, 1) == [False, False]


@pytest.mark.cpu_only
def test_output_modulation_kernel_rule() -> None:
    """CPU tensors keep the eager output-head modulation."""
    model = _tiny_model(False)
    hidden = model.config.hidden_size
    x = torch.zeros(2, 8, hidden, dtype=torch.bfloat16)
    per_batch = torch.zeros(2, 1, hidden)
    # CPU tensors and per-token modulation keep the eager ops.
    assert not model._modulation_kernel_applies(x, per_batch, per_batch, torch.bfloat16)


@pytest.mark.cpu_only
def test_cross_attention_q_amax_file(tmp_path, monkeypatch) -> None:
    """The calibrated Q maxima (times their margin) reach each planned block by module name; a
    missing layer and an empty path leave the bound-derived scale."""
    from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import (
        load_cross_attention_q_amax,
    )

    assert load_cross_attention_q_amax("") == {}
    path = tmp_path / "q_amax.json"
    path.write_text(json.dumps({"margin": 1.5, "amax": {"blocks.0.attn2": 4.0}}))
    assert load_cross_attention_q_amax(str(path)) == {"blocks.0.attn2": 6.0}

    seen = {}

    def plan(self, log=False, cross_attention_q_amax=None):
        seen[self.attn2.module_name] = cross_attention_q_amax
        return False

    monkeypatch.setattr(WanBlock, "plan_fp8_fusions", plan)
    model = _tiny_model(True)
    model.model_config.extra_attrs["fp8_cross_attention_q_amax"] = str(path)
    model.post_load_weights()
    assert seen == {"blocks.0.attn2": 6.0, "blocks.1.attn2": None}

    path.write_text(json.dumps({"margin": 0.5, "amax": {}}))
    with pytest.raises(ValueError, match="margin"):
        load_cross_attention_q_amax(str(path))
    path.write_text(json.dumps({"margin": 1.5, "blocks.0.attn2": 4.0}))
    with pytest.raises(ValueError, match="'amax' object"):
        load_cross_attention_q_amax(str(path))
    with pytest.raises(FileNotFoundError):
        load_cross_attention_q_amax(str(tmp_path / "missing.json"))

    # Names of another transformer match no layer here: the bound scale everywhere, with a warning.
    path.write_text(json.dumps({"amax": {"transformer_2.blocks.0.attn2": 4.0}}))
    seen.clear()
    with patch.object(trtllm_logger, "warning") as warning:
        model.post_load_weights()
    assert seen == {"blocks.0.attn2": None, "blocks.1.attn2": None}
    assert any("match" in str(call.args[0]) for call in warning.call_args_list)
