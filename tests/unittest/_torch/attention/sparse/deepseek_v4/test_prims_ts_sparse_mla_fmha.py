# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the PrimTS native sparse MLA FMHA library (DeepSeek-V4, SM100)."""

import gc
import math
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha import registry
from tensorrt_llm._torch.attention.backends.fmha.prims_ts_sparse_mla import (
    PrimsTSSparseMlaFmha,
    get_prims_ts_toolchain_unsupported_reason,
)
from tensorrt_llm._utils import get_sm_version

from ..test_sparse_mla_forward import HAS_FLASH_MLA
from ..test_sparse_mla_forward import (
    test_forward_sparse_mla_unified as _run_unified_sparse_mla_forward,
)

LIB_NAME = "prims_ts_sparse_mla"
LATENT = 512
ROPE = 64
BLOCK = 128

pytestmark = pytest.mark.skipif(
    get_sm_version() not in (100, 103), reason="PrimTS sparse MLA requires SM100 or SM103"
)


def _skip_unless_toolchain() -> None:
    """Skip the tests that need the library to actually serve a batch."""
    if get_sm_version() != 100:
        pytest.skip("PrimTS sparse MLA serves SM100 only (SM103 is not yet qualified)")
    reason = get_prims_ts_toolchain_unsupported_reason()
    if reason is not None:
        pytest.skip(reason)


def test_registry_entry_is_default_before_fallback(monkeypatch) -> None:
    """The library ships enabled and ahead of the TRTLLM-Gen fallback.

    Its ``is_available`` restricts it to DeepSeek-V4 on SM100, so being
    in the default set changes nothing for other models or architectures.
    """
    monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)
    assert registry.FMHA_LIBS[LIB_NAME] is PrimsTSSparseMlaFmha
    assert LIB_NAME in registry.DEFAULT_FMHA_LIBS
    names = tuple(registry.FMHA_LIBS)
    assert names.index(LIB_NAME) < names.index("flashinfer_trtllm_gen")
    assert names.index(LIB_NAME) < names.index("fallback")
    assert PrimsTSSparseMlaFmha in registry.get_enabled_fmha_lib_classes()
    monkeypatch.setenv("TLLM_FMHA_LIBS", f"-{LIB_NAME}")
    assert PrimsTSSparseMlaFmha not in registry.get_enabled_fmha_lib_classes()


def test_sm103_stays_on_the_fallback_until_qualified(monkeypatch) -> None:
    """B300/GB300 report an SM reason; SM100 proceeds to the shape and toolchain checks.

    The vendored kernels admit SM103 through their architecture guard, but the
    flashinfer signoff is SM100 only and L0 #64329 hit an illegal memory access
    in the GB300 DeepSeek-V4-Pro warmup, so the adapter keeps SM103 on TRTLLM-Gen.
    """
    import tensorrt_llm._torch.attention.backends.fmha.prims_ts_sparse_mla as mod

    attn = SimpleNamespace(
        use_fp8_ds_mla=False,
        kv_cache_dtype="auto",
        _uses_nvfp4_compress=False,
        kv_lora_rank=448,
        qk_rope_head_dim=64,
        head_dim=512,
        num_heads=16,
    )
    monkeypatch.setattr(mod, "get_sm_version", lambda: 103)
    assert "SM103" in PrimsTSSparseMlaFmha._unavailable_reason(attn)
    monkeypatch.setattr(mod, "get_sm_version", lambda: 90)
    assert "requires SM100" in PrimsTSSparseMlaFmha._unavailable_reason(attn)
    monkeypatch.setattr(mod, "get_sm_version", lambda: 100)
    reason = PrimsTSSparseMlaFmha._unavailable_reason(attn)
    assert reason is None or reason == get_prims_ts_toolchain_unsupported_reason()


def test_softmax_scale_follows_qk_head_dim() -> None:
    """The scale uses ``qk_nope + qk_rope`` like attentionOp.cpp, not the 512 latent.

    DeepSeek-V4 checkpoints have 448 + 64 = 512, so only a narrower nope width
    (the 128 + 64 scenario of ``test_deepseek_v4_sparse_mla``) tells the two apart;
    L0 #64329 caught the latent-width version as a ~0.3 BF16 output mismatch.
    """
    attn = SimpleNamespace(qk_nope_head_dim=128, qk_rope_head_dim=64, q_scaling=0.5336)
    expected = 1.0 / (0.5336 * math.sqrt(192.0))
    assert PrimsTSSparseMlaFmha._softmax_scale(attn) == pytest.approx(expected, rel=1e-12)
    attn = SimpleNamespace(qk_nope_head_dim=448, qk_rope_head_dim=64, q_scaling=1.0)
    assert PrimsTSSparseMlaFmha._softmax_scale(attn) == pytest.approx(LATENT**-0.5, rel=1e-12)


def test_pool_view_cache_follows_the_owning_manager() -> None:
    """A new cache manager gets fresh pool views, even at the old manager's address.

    The views wrap the raw pool base pointer, so a view cached for a freed
    manager would alias released memory. The cache is owned through a weak
    reference and dropped whenever the owner changes.
    """
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType,
    )

    class _Owner:  # weak-referenceable stand-ins for TrtllmAttention / the manager
        pass

    def make_manager(rows: int) -> _Owner:
        manager = _Owner()
        manager.pool = torch.empty(rows, 1, LATENT, dtype=torch.bfloat16, device="cuda")
        manager.swa_pool_ptr = manager.pool.data_ptr()
        manager.compress_pool_ptrs = {}
        manager.pp_layers = [0]
        manager._compress_ratios = {0: 4}
        manager.dtype = torch.bfloat16
        manager.get_buffers = lambda layer, attn_type: manager.pool
        return manager

    attn = _Owner()
    fmha = PrimsTSSparseMlaFmha(attn)
    first = make_manager(8)
    view = fmha._pool_view(first, DeepseekV4AttentionType.SWA, 1)
    assert view.data_ptr() == first.swa_pool_ptr
    assert tuple(view.shape) == (8, 1, LATENT)
    assert fmha._pool_view(first, DeepseekV4AttentionType.SWA, 1) is view

    second = make_manager(16)
    other = fmha._pool_view(second, DeepseekV4AttentionType.SWA, 1)
    assert other is not view
    assert other.data_ptr() == second.swa_pool_ptr
    assert tuple(other.shape) == (16, 1, LATENT)
    assert fmha._pool_views_owner() is second

    del second
    gc.collect()
    assert fmha._pool_views_owner() is None
    third = make_manager(4)
    assert tuple(fmha._pool_view(third, DeepseekV4AttentionType.SWA, 1).shape) == (4, 1, LATENT)


def test_plans_are_shared_through_the_metadata_cache(monkeypatch) -> None:
    """Layers with one static profile share one plan via ``metadata.fmha_plan_caches``.

    The plan owns the kernel workspace, so caching it per layer would keep one
    workspace per layer per CUDA-graph geometry; the metadata cache (the
    convention of the block-sparse and FlashInfer wrappers) keeps one per
    profile, and layers sharing a metadata object run serially.
    """
    import tensorrt_llm._torch.attention.backends.prims_ts.mla_decode as mla_decode

    planned = []

    class _FakeWrapper:
        def plan(self, device, rows, heads, **kwargs):
            planned.append((device, rows, heads, kwargs))

    monkeypatch.setattr(mla_decode, "BatchSparseMLADecodePagedTSWrapper", _FakeWrapper)

    class _Attn:  # weak-referenceable stand-in for TrtllmAttention
        num_heads = 16
        sparse_attention_config = SimpleNamespace(window_size=128)

    attn_a, attn_b = _Attn(), _Attn()
    layer_a, layer_b = PrimsTSSparseMlaFmha(attn_a), PrimsTSSparseMlaFmha(attn_b)
    metadata = SimpleNamespace(fmha_plan_caches={})
    profile = dict(
        metadata=metadata,
        device=torch.device("cuda"),
        rows=4,
        extra=256,
        q_dtype=torch.bfloat16,
        has_sinks=False,
        fused_groups=None,
    )
    plan_a = layer_a._get_plan(**profile)
    plan_b = layer_b._get_plan(**profile)
    assert plan_a is plan_b
    assert len(planned) == 1
    assert planned[0][3]["max_topk"] == 128 and planned[0][3]["max_extra_topk"] == 256
    cache = metadata.fmha_plan_caches[PrimsTSSparseMlaFmha.PLAN_CACHE_KEY]
    assert list(cache.values()) == [plan_a]

    other = layer_a._get_plan(**{**profile, "extra": 0})
    assert other is not plan_a
    assert other.extra_lengths is None
    assert len(planned) == 2
    assert len(cache) == 2


def _rope_table(max_positions: int, device: torch.device) -> torch.Tensor:
    """Forward [cos(32), sin(32)] rows, the layout both kernels consume."""
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, ROPE, 2, dtype=torch.float32, device=device) / ROPE)
    )
    angles = torch.outer(torch.arange(max_positions, dtype=torch.float32, device=device), inv_freq)
    return torch.cat([angles.cos(), angles.sin()], dim=-1).contiguous()


def _random_indices(rows: int, capacity: int, pool_rows: int, device: torch.device):
    lengths = torch.randint(1, capacity + 1, (rows,), dtype=torch.int32, device=device)
    indices = torch.randint(0, pool_rows, (rows, capacity), dtype=torch.int32, device=device)
    inactive = torch.arange(capacity, device=device)[None, :] >= lengths[:, None]
    indices[inactive] = -1
    return indices, lengths


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"])
@pytest.mark.parametrize(
    "rows,heads,groups,topk,extra_topk",
    [
        (1, 128, 16, 128, 0),
        (5, 128, 16, 128, 512),
        (64, 128, 16, 128, 512),
        (16, 64, 8, 128, 256),
        (3, 64, 8, 128, 0),
    ],
)
def test_fused_output_matches_trtllm_inverse_rope_quant(
    dtype, rows, heads, groups, topk, extra_topk
):
    """The kernel's fused epilogue must reproduce TRT-LLM's inverse-RoPE + FP8 quant.

    Reference: the same attention with a BF16 output, followed by
    ``trtllm.fused_inv_rope_fp8_quant_vllm_port`` (the op the unfused DSv4
    path runs). The fused kernel quantizes from FP32 accumulators instead of
    the BF16 output, so compare dequantized values up to FP8 rounding.
    """
    _skip_unless_toolchain()
    from tensorrt_llm._torch.attention.backends.prims_ts.mla_decode import (
        BatchSparseMLADecodePagedTSWrapper,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    pool_rows, extra_rows, max_positions = 4096, 2048, 4096

    def make_pool(n):
        return (torch.randn(n, 1, LATENT, dtype=torch.bfloat16, device=device)).to(dtype)

    pool = make_pool(pool_rows)
    extra_pool = make_pool(extra_rows) if extra_topk else None
    indices, lengths = _random_indices(rows, topk, pool_rows, device)
    extra_indices, extra_lengths = (
        (None, None) if extra_topk == 0 else _random_indices(rows, extra_topk, extra_rows, device)
    )
    query = (torch.randn(rows, 1, heads, LATENT, dtype=torch.bfloat16, device=device) * 0.5).to(
        dtype
    )
    positions = torch.randint(0, max_positions, (rows,), dtype=torch.int32, device=device)
    cos_sin = _rope_table(max_positions, device)

    softmax_scale = LATENT**-0.5
    descale = 0.75 if dtype == torch.float8_e4m3fn else 1.0
    # The kernel takes the combined scales (attention x Q descale x KV descale
    # on the softmax, the KV descale on the output), as the adapter passes them.
    run_scales = dict(softmax_scale=softmax_scale * descale * descale, output_scale=descale)
    sources = dict(extra_indices=extra_indices, extra_lengths=extra_lengths) if extra_topk else {}
    plan_args = dict(max_topk=topk, max_extra_topk=extra_topk, max_seq_len_q=1, q_data_type=dtype)

    reference = BatchSparseMLADecodePagedTSWrapper()
    reference.plan(device, rows, heads, **plan_args)
    out_bf16 = reference.run(query, pool, indices, lengths, extra_pool, **sources, **run_scales)
    heads_per_group = heads // groups
    ref_fp8, ref_scale = torch.ops.trtllm.fused_inv_rope_fp8_quant_vllm_port(
        out_bf16.view(rows, heads, LATENT),
        positions,
        cos_sin,
        groups,
        heads_per_group,
        LATENT - ROPE,
        ROPE,
        BLOCK,
        False,
    )

    fused = BatchSparseMLADecodePagedTSWrapper()
    fused.plan(
        device,
        rows,
        heads,
        output_quant_block_size=BLOCK,
        out_scale_format="fp32",
        num_output_groups=groups,
        **plan_args,
    )
    channels = heads_per_group * LATENT
    padded_rows = (rows + 3) // 4 * 4
    # The exact buffers TRT-LLM's fused DSv4 epilogue allocates.
    out_buf = torch.empty((groups, rows, channels), dtype=torch.float8_e4m3fn, device=device)
    scale_buf = torch.empty(
        (groups, channels // BLOCK, padded_rows), dtype=torch.float32, device=device
    )
    fused_out, fused_scale = fused.run(
        query,
        pool,
        indices,
        lengths,
        extra_pool,
        **sources,
        out=out_buf.transpose(0, 1),
        out_scale=scale_buf.permute(2, 0, 1)[:rows],
        token_positions=positions.view(rows, 1),
        cos_sin_cache=cos_sin,
        **run_scales,
    )
    assert fused_out.data_ptr() == out_buf.data_ptr()
    assert fused_scale.data_ptr() == scale_buf.data_ptr()
    assert ref_fp8.shape == out_buf.shape and ref_scale.shape == scale_buf.shape

    ref_scales = ref_scale[..., :rows].permute(0, 2, 1)  # [G, T, blocks]
    got_scales = scale_buf[..., :rows].permute(0, 2, 1)
    torch.testing.assert_close(got_scales, ref_scales, rtol=2e-2, atol=1e-6)

    def dequant(values, scales):
        blocks = values.float().view(groups, rows, channels // BLOCK, BLOCK)
        return (blocks * scales[..., None]).flatten(-2)

    ref = dequant(ref_fp8, ref_scales)
    got = dequant(out_buf, got_scales)
    # One E4M3 ulp is 1/8 of the magnitude. The reference rotates the
    # BF16-rounded output, so near-cancelling RoPE channels (D[448:512]) carry
    # an absolute error of a few BF16 ulps of their O(1) inputs, which the
    # kernel (rotating FP32 accumulators) does not have; that shows up as up to
    # about one quantization step of the block. Allow two steps there and half
    # a step elsewhere, on top of two relative ulps.
    block_scales = ref_scales.repeat_interleave(BLOCK, dim=-1)
    is_rope = torch.zeros(channels, dtype=torch.bool, device=device)
    is_rope.view(-1, LATENT)[:, LATENT - ROPE :] = True
    step_allowance = torch.where(is_rope, 2.0, 0.5) * block_scales
    ulp = ref.abs() * 0.125 + block_scales * 2.0**-6
    diff = (got - ref).abs()
    assert (diff <= 2.0 * ulp + step_allowance + 1e-6).all(), (
        f"max fused-vs-reference gap {diff.max().item()}"
    )
    assert diff.mean().item() <= 0.02 * ref.abs().mean().item() + 1e-6


@pytest.mark.skipif(not HAS_FLASH_MLA, reason="FlashMLA not available")
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize(
    "batch_name",
    [
        "small_decode",
        "medium_decode",
        "single_decode",
        "small_mixed",
        "mixed_heavy_decode",
    ],
)
def test_forward_sparse_mla_unified_prims_ts(monkeypatch, batch_name, kv_cache_dtype):
    """Run the DeepSeek-V4 module test on the default library set.

    No ``TLLM_FMHA_LIBS`` override: the PrimTS library is in the default set and
    must win the generation phase on its own.
    """
    _skip_unless_toolchain()
    monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)
    calls = []
    original_forward = PrimsTSSparseMlaFmha.forward

    def counting_forward(self, *args, **kwargs):
        calls.append(1)
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(PrimsTSSparseMlaFmha, "forward", counting_forward)
    _run_unified_sparse_mla_forward(batch_name, kv_cache_dtype, "deepseek_v4")
    assert calls, "the PrimTS sparse MLA FMHA did not serve the generation phase"


def _dequant_fused(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize a fused DSv4 epilogue buffer.

    ``values`` is ``[G, T, C]`` E4M3 and ``scales`` is ``[G, C // 128, pad4(T)]``
    FP32, exactly as ``_create_dsv4_epilogue_buffers`` allocates them.
    """
    groups, rows, channels = values.shape
    sc = scales[..., :rows].permute(0, 2, 1)  # [G, T, blocks]
    blocks = values.float().view(groups, rows, channels // BLOCK, BLOCK)
    return (blocks * sc[..., None]).flatten(-2), sc


def _assert_fused_close(got, got_sc, ref, ref_sc, channels, device):
    """Same tolerance as the kernel-level fused test: two relative E4M3 ulps plus
    half a quantization step, two steps on the RoPE channels D[448:512]."""
    torch.testing.assert_close(got_sc, ref_sc, rtol=2e-2, atol=1e-6)
    block_scales = ref_sc.repeat_interleave(BLOCK, dim=-1)
    is_rope = torch.zeros(channels, dtype=torch.bool, device=device)
    is_rope.view(-1, LATENT)[:, LATENT - ROPE :] = True
    step_allowance = torch.where(is_rope, 2.0, 0.5) * block_scales
    diff = (got - ref).abs()
    ulp = ref.abs() / 8.0
    assert (diff <= 2.0 * ulp + step_allowance + 1e-6).all(), (
        f"max diff {diff.max().item():.4g} vs allowance "
        f"{(2.0 * ulp + step_allowance).max().item():.4g}"
    )
    assert diff.mean().item() <= 0.02 * ref.abs().mean().item() + 1e-6


@pytest.mark.skipif(not HAS_FLASH_MLA, reason="FlashMLA not available")
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("batch_name", ["small_decode", "medium_decode", "single_decode"])
def test_forward_sparse_mla_unified_fused_epilogue(monkeypatch, batch_name, kv_cache_dtype):
    """Module-level coverage of the fused DSv4 epilogue (production ADP path).

    ``prepare_sparse_attn_outputs`` only hands out the 3D fused buffer under
    attention-DP with 128 heads, an FP8 cache and an FP8 ``o_a_proj``, which the
    module test harness does not configure. The generation entry point
    ``forward_generation_sparse_attn`` takes ``enable_dsv4_epilogue_fusion``
    directly, so wrap the harness's call to it and, on the same prepared cache,
    metadata and inputs, additionally run the fused mode on the default library
    set: PrimTS must serve it with ``forward_args.enable_dsv4_epilogue_fusion``
    set, and its fused output must match the unfused BF16 output pushed through
    ``trtllm.fused_inv_rope_fp8_quant_vllm_port`` (the path the module takes when
    fusion is off). The harness's own reference check still runs on the unfused
    call.

    The TRTLLM-Gen fallback is not run in fused mode here: the module only
    enables the fused epilogue for it at 128 heads, and at the harness's 64
    heads its C++ op aborts the process rather than raising.
    """
    _skip_unless_toolchain()
    monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)

    from .. import test_sparse_mla_forward as harness

    original_gen = harness.forward_generation_deepseek_v4_mla
    original_forward = PrimsTSSparseMlaFmha.forward
    seen_fused_flags: list[bool] = []
    results: dict[str, object] = {}

    def spy_forward(self, q, k, v, metadata, forward_args):
        seen_fused_flags.append(bool(forward_args.enable_dsv4_epilogue_fusion))
        return original_forward(self, q, k, v, metadata, forward_args)

    _INPLACE_INPUTS = ("q", "compressed_kv", "k_pe", "latent_cache")

    def run_fused(mla, kwargs, pristine):
        # The RoPE/KV-write prologue rotates ``q`` and ``latent_cache`` in
        # place, so every call needs the inputs as they were before the first.
        fused_kwargs = dict(kwargs)
        fused_kwargs.update({k: v.clone() for k, v in pristine.items()})
        fused_kwargs["output"] = None
        fused_kwargs["enable_dsv4_epilogue_fusion"] = True
        return original_gen(mla, **fused_kwargs)

    def wrapped_gen(mla, **kwargs):
        pristine = {k: kwargs[k].clone() for k in _INPLACE_INPUTS if kwargs.get(k) is not None}
        # 1. The harness's unfused call, untouched (it validates against the
        #    Python reference afterwards).
        out = original_gen(mla, **kwargs)
        output = kwargs["output"]
        rows = output.shape[0]
        heads = mla.num_heads_tp
        groups = mla.n_local_groups
        positions = kwargs["position_ids"].view(-1).to(torch.int32)
        cos_sin = mla.inverse_rotary_emb.rotary_cos_sin
        torch.cuda.synchronize()

        # Path B: the standalone inverse-RoPE + FP8 quant op on the BF16 output.
        ref_fp8, ref_scale = torch.ops.trtllm.fused_inv_rope_fp8_quant_vllm_port(
            output.view(rows, heads, LATENT).clone(),
            positions,
            cos_sin,
            groups,
            heads // groups,
            LATENT - ROPE,
            ROPE,
            BLOCK,
            False,
        )

        # 2. Path A on the default library set: PrimTS serves the fused epilogue.
        #    (The module hands out the 3D buffer only under ADP + 128 heads + FP8,
        #    which is what production throughput configs run.)
        before = len(seen_fused_flags)
        prims_fp8, prims_scale = run_fused(mla, kwargs, pristine)
        torch.cuda.synchronize()
        assert seen_fused_flags[before:] == [True], (
            "the PrimTS sparse MLA FMHA did not serve the fused generation call"
        )

        results.update(
            rows=rows,
            heads=heads,
            groups=groups,
            ref=(ref_fp8.clone(), ref_scale.clone()),
            prims=(prims_fp8.clone(), prims_scale.clone()),
        )
        return out

    monkeypatch.setattr(PrimsTSSparseMlaFmha, "forward", spy_forward)
    monkeypatch.setattr(harness, "forward_generation_deepseek_v4_mla", wrapped_gen)
    _run_unified_sparse_mla_forward(batch_name, kv_cache_dtype, "deepseek_v4")

    assert results, "the generation phase never ran"
    assert seen_fused_flags.count(False) >= 1, "the unfused call did not reach PrimTS"
    device = results["prims"][0].device
    channels = results["heads"] // results["groups"] * LATENT

    ref, ref_sc = _dequant_fused(*results["ref"])
    got, got_sc = _dequant_fused(*results["prims"])
    assert got.shape == ref.shape
    _assert_fused_close(got, got_sc, ref, ref_sc, channels, device)
