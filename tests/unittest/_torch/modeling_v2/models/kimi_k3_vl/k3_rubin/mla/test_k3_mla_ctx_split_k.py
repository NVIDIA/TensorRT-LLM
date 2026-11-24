# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 MLA cached-prefix context on the Rubin FP8 split-K context FMHA
(k3_rubin/mla/ctx_split_k.py).

CPU: which layers and calls take the split-K path and when kv_b_proj takes the MXFP8 cache
gather. GPU (SM89+): the partition schedule handed to the FMHA. SM107: the MXFP8 gather against
the dense gather plus the C++ FP8 quantizers, and the whole chunked prefill on a real FP4 cache
against an FP32 reference, with and without TRTLLM_K3_MLA_CTX_TRITON_QUANT."""

import math
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.kernels import (
    k3_context_fmha_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.mla import ctx_split_k
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.mla.ctx_split_k import (  # noqa: E501
    KimiK3MlaContextSplitKMixin,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention

H, NOPE, ROPE, V, LORA = 96, 128, 64, 128, 512
N_KVB = H * (NOPE + V)
PAGE = 128

requires_fp8_cvt = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="E4M3 conversion instructions require SM89+",
)
requires_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="The K3 split-K context FMHA, the MXFP8 gather and the MXFP8 GEMM require SM107",
)


# ---------------------------------------------------------------------------------------------
# CPU: selection
# ---------------------------------------------------------------------------------------------


def _backend(heads=H, kv_heads=None, nope=NOPE, rope=ROPE, v=V):
    return SimpleNamespace(
        num_heads=heads,
        num_kv_heads=heads if kv_heads is None else kv_heads,
        qk_nope_head_dim=nope,
        qk_rope_head_dim=rope,
        v_head_dim=v,
    )


@pytest.fixture
def info_once(monkeypatch):
    calls = []
    monkeypatch.setattr(ctx_split_k.logger, "info_once", lambda msg, key: calls.append((msg, key)))
    return calls


@pytest.mark.cpu_only
def test_split_k_support_is_shape_gated(monkeypatch, info_once):
    monkeypatch.setattr(ctx_split_k, "get_sm_version", lambda: 107)
    monkeypatch.setattr(k3_context_fmha_op, "IS_K3_CONTEXT_FMHA_AVAILABLE", True)
    monkeypatch.setattr(k3_context_fmha_op, "K3_CONTEXT_FMHA_IMPORT_ERROR", None)

    for heads in (96, 48, 24, 12, 6, 3):
        assert ctx_split_k.supports_k3_fp8_split_k_context(_backend(heads))
    assert not info_once

    cases = [
        (_backend(12, 6), "matching positive Q/KV head counts per rank, got 12/6"),
        (_backend(0), "matching positive Q/KV head counts per rank, got 0/0"),
        (_backend(nope=64), "requires head dimensions"),
        (_backend(rope=128), "requires head dimensions"),
    ]
    for attn, reason in cases:
        assert not ctx_split_k.supports_k3_fp8_split_k_context(attn)
        assert reason in info_once[-1][0]
    # One line per reason.
    assert ctx_split_k.split_k_context_unavailable_reason(_backend(12, 6)) in info_once[0][1]

    monkeypatch.setattr(ctx_split_k, "get_sm_version", lambda: 103)
    assert not ctx_split_k.supports_k3_fp8_split_k_context(_backend())
    assert "requires SM107, got SM103" in info_once[-1][0]

    monkeypatch.setattr(ctx_split_k, "get_sm_version", lambda: 107)
    monkeypatch.setattr(k3_context_fmha_op, "IS_K3_CONTEXT_FMHA_AVAILABLE", False)
    assert not ctx_split_k.supports_k3_fp8_split_k_context(_backend())
    assert "lacks split-K epilogue merge support" in info_once[-1][0]
    monkeypatch.setattr(
        k3_context_fmha_op, "K3_CONTEXT_FMHA_IMPORT_ERROR", ImportError("missing CuTe runtime")
    )
    assert not ctx_split_k.supports_k3_fp8_split_k_context(_backend())
    assert "import failed" in info_once[-1][0] and "missing CuTe runtime" in info_once[-1][0]
    assert "keep the FP8 context MLA FMHA partition loop" in info_once[-1][0]


def _trtllm_backend(*, fp4=True, sparse=None):
    attention = object.__new__(TrtllmAttention)
    attention.is_mla_enable = True
    attention.has_fp4_kv_cache = fp4
    attention.has_fp8_kv_cache = not fp4
    attention.sparse_params = sparse
    return attention


class _BaseLoop:
    def forward_context_with_chunked_prefill_fp4(self, *args):
        return "base"


class _Layer(KimiK3MlaContextSplitKMixin, _BaseLoop):
    def __init__(self, mha, kv_b_proj=None):
        self.mha = mha
        self.kv_b_proj = kv_b_proj
        self.out_scale = None
        self.kv_lora_rank, self.qk_rope_head_dim = LORA, ROPE
        self.qk_nope_head_dim, self.v_head_dim = NOPE, V
        self.num_heads_tp = H

    def _forward_context_split_k_fp4(self, *args):
        return "split_k"


@pytest.mark.cpu_only
def test_split_k_replaces_the_partition_loop_only_where_it_applies(monkeypatch):
    supported = {"value": True}
    monkeypatch.setattr(
        ctx_split_k, "supports_k3_fp8_split_k_context", lambda attn: supported["value"]
    )
    bf16 = torch.empty(1, dtype=torch.bfloat16)

    def run(layer, q=bf16, output=bf16):
        return layer.forward_context_with_chunked_prefill_fp4(q, None, None, None, output)

    layer = _Layer(_trtllm_backend())
    assert run(layer) == "split_k"
    assert run(layer, q=bf16.half()) == "base"
    assert run(layer, output=bf16.float()) == "base"
    layer.out_scale = torch.ones(1)
    assert run(layer) == "base"
    layer.out_scale = None
    # Only NVFP4 latent caches (not FP8 / sparse layers) and only a TrtllmAttention backend.
    assert run(_Layer(_trtllm_backend(fp4=False))) == "base"
    assert run(_Layer(_trtllm_backend(sparse=object()))) == "base"
    assert run(_Layer(SimpleNamespace(uses_fp4_mla_attention=True))) == "base"
    supported["value"] = False
    assert run(layer) == "base"


def _kv_b_proj(**overrides):
    fields = dict(
        weight=torch.empty(2, 2, dtype=torch.float8_e4m3fn),
        weight_scale=torch.empty(2, dtype=torch.uint8),
        supports_prequantized_input=True,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.cpu_only
def test_kv_b_takes_mxfp8_where_the_weight_is_in_the_cute_mxfp8_layout(monkeypatch):
    monkeypatch.setattr(ctx_split_k, "get_sm_version", lambda: 107)
    monkeypatch.setattr(ctx_split_k, "IS_CUTLASS_DSL_RUBIN_AVAILABLE", True)
    mha = _trtllm_backend()

    def takes(**overrides):
        return _Layer(mha, _kv_b_proj(**overrides))._kv_b_takes_mxfp8()

    assert takes()
    block_scales = SimpleNamespace(quant_mode=SimpleNamespace(has_fp8_block_scales=lambda: True))
    for gemm in ("use_cute_dsl_blockscaling_mm", "disable_deep_gemm"):
        assert takes(supports_prequantized_input=False, quant_config=block_scales, **{gemm: True})
    assert not takes(supports_prequantized_input=False, quant_config=block_scales)
    assert not takes(supports_prequantized_input=False)
    assert not takes(bias=torch.zeros(2))
    assert not takes(weight=torch.empty(2, 2, dtype=torch.bfloat16))
    assert not takes(weight_scale=torch.empty(2, dtype=torch.int32))
    assert not takes(weight_scale=None)
    layer = _Layer(mha, _kv_b_proj())
    layer.kv_lora_rank = 256
    assert not layer._kv_b_takes_mxfp8()
    monkeypatch.setattr(ctx_split_k, "IS_CUTLASS_DSL_RUBIN_AVAILABLE", False)
    assert not takes()
    monkeypatch.setattr(ctx_split_k, "IS_CUTLASS_DSL_RUBIN_AVAILABLE", True)
    monkeypatch.setattr(ctx_split_k, "get_sm_version", lambda: 103)
    assert not takes()


@pytest.mark.cpu_only
def test_k3_mla_attention_takes_the_split_k_override():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.mla import (
        KimiK3MLAAttention,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.mla.ctx_fp8 import (  # noqa: E501
        KimiK3MlaContextFp8Mixin,
    )

    mro = KimiK3MLAAttention.__mro__
    assert mro.index(KimiK3MlaContextSplitKMixin) < mro.index(KimiK3MlaContextFp8Mixin)
    assert (
        KimiK3MLAAttention.forward_context_with_chunked_prefill_fp4
        is KimiK3MlaContextSplitKMixin.forward_context_with_chunked_prefill_fp4
    )


# ---------------------------------------------------------------------------------------------
# Shared GPU fixtures: a real dense FP4 MLA cache and chunked-prefill metadata
# ---------------------------------------------------------------------------------------------


def _e4m3_byte(value):
    return int(torch.tensor([value]).to(torch.float8_e4m3fn).view(torch.uint8).item())


def _build_prefix_case(prefix, new, chunk_size, buffer_batch, seed=0, global_scale=1.0):
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

    nctx = len(prefix)
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    pages = [math.ceil((p + q) / PAGE) for p, q in zip(prefix, new)]
    stride = max(pages)
    total_pages = sum(pages) + 2
    kv = torch.randint(
        0, 256, (total_pages, 1, PAGE, 1, 320), dtype=torch.uint8, device="cuda", generator=g
    )
    sf = torch.randint(
        _e4m3_byte(0.0625),
        _e4m3_byte(0.5),
        (total_pages, PAGE, 40),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    perm = torch.randperm(total_pages, generator=torch.Generator().manual_seed(seed)).tolist()
    rows, base = [], 0
    for p in pages:
        rows += perm[base : base + p] + [0] * (stride - p)
        base += p
    loops = math.ceil(sum(prefix) / (chunk_size * buffer_batch))
    shim = SimpleNamespace(
        num_contexts=nctx,
        runtime_features=SimpleNamespace(
            chunk_size=chunk_size, chunked_prefill_buffer_batch_size=buffer_batch
        ),
        kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=list(prefix)),
    )
    seq = torch.zeros((loops, nctx), dtype=torch.int)
    cu = torch.zeros((loops, nctx + 1), dtype=torch.int64)
    offsets = torch.zeros((loops + 1, nctx), dtype=torch.int64)
    merge = torch.empty((loops + 1, nctx), dtype=torch.int64)
    max_len = []
    TrtllmAttentionMetadata.pre_process_for_chunked_prefill(
        shim,
        chunked_seq_len=seq,
        chunked_global_offset=offsets,
        cu_chunked_seq_len=cu,
        merge_op_tensor=merge,
        max_chunk_len_per_loop=max_len,
        chunked_loop_num=loops,
    )
    T = sum(new)
    cu_q = torch.tensor(
        [0] + torch.tensor(new).cumsum(0).tolist(), dtype=torch.int32, device="cuda"
    )
    meta = SimpleNamespace(
        num_contexts=nctx,
        num_ctx_tokens=T,
        chunked_loop_num=loops,
        chunked_seq_len=seq.cuda(),
        host_chunked_seq_len=seq,
        cu_chunked_seq_len=cu.cuda(),
        host_cu_chunked_seq_len=cu,
        chunked_global_offset=offsets.cuda(),
        max_chunk_len_per_loop=max_len,
        merge_op_tensor=merge.cuda(),
        max_ctx_seq_len=max(new),
        mla_prepare_ctx_cu_seqlens=lambda: cu_q,
        fp4_mla_state=SimpleNamespace(
            device_page_table=True,
            device_page_table_valid=True,
            page_table_stride=stride,
            _paged_kv_indices=torch.tensor(rows, dtype=torch.int32, device="cuda"),
            kv_global_scale=torch.full((1,), global_scale, dtype=torch.float32, device="cuda"),
        ),
        page_size=PAGE,
        kv_cache_manager=SimpleNamespace(get_fp4_mla_cache_buffers=lambda _layer: (kv, sf)),
    )
    q = torch.randn(T, H * (NOPE + ROPE), generator=g, device="cuda").to(torch.bfloat16)
    latent = (torch.randn(T, LORA + ROPE, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    return SimpleNamespace(
        prefix=prefix,
        new=new,
        meta=meta,
        q=q,
        latent=latent,
        rows=rows,
        stride=stride,
        T=T,
        loops=loops,
        cu_q=cu_q,
    )


def _dense_gather(meta, *, tokens, cu, offsets, max_len):
    from tensorrt_llm._torch.attention.backends.fp4_mla import load_fp4_mla_chunked_kv_cache

    return load_fp4_mla_chunked_kv_cache(
        meta,
        0,
        num_ctx_cached_tokens=tokens,
        cu_chunked_seq_len=cu,
        chunked_global_offset=offsets,
        chunked_max_seq_len=max_len,
        out_dtype=torch.bfloat16,
        kv_lora_rank=LORA,
        qk_rope_head_dim=ROPE,
    )


class _KvBProj:
    """kv_b_proj of the reference: a BF16 GEMM and, with ``mxfp8`` (SM100 family), its 1x128
    MXFP8 quantization in the layout ``cute_dsl_mxfp8_gemm_rubin`` reads."""

    bias = None

    def __init__(self, mxfp8=False, seed=1):
        g = torch.Generator(device="cuda")
        g.manual_seed(seed)
        self.weight_bf16 = (torch.randn(N_KVB, LORA, generator=g, device="cuda") / 16).to(
            torch.bfloat16
        )
        self.weight = self.weight_scale = None
        if mxfp8:
            self.weight, self.weight_scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(
                self.weight_bf16
            )

    def __call__(self, x):
        return x @ self.weight_bf16.T


class _Mha:
    """The attention backend calls of the split-K path, over the case's FP4 cache."""

    layer_idx = 0
    num_heads = num_kv_heads = H
    qk_nope_head_dim, qk_rope_head_dim, v_head_dim = NOPE, ROPE, V
    q_scaling = 1.0

    def __init__(self, loader=None):
        self.kv_cache_scaling_factor = torch.ones(1, dtype=torch.float32, device="cuda")
        self._loader = loader

    def append_fp4_mla_context_cache(self, q, latent_cache, metadata):
        pass

    def load_chunked_kv_cache_for_mla(
        self,
        metadata,
        num_ctx_cached_tokens,
        cu_chunked_seq_len,
        chunked_global_offset,
        chunked_max_seq_len,
        out_dtype,
    ):
        assert out_dtype == torch.bfloat16
        if self._loader is not None:
            return self._loader(num_ctx_cached_tokens)
        return _dense_gather(
            metadata,
            tokens=num_ctx_cached_tokens,
            cu=cu_chunked_seq_len,
            offsets=chunked_global_offset,
            max_len=chunked_max_seq_len,
        )


class _SplitKLayer(KimiK3MlaContextSplitKMixin):
    def __init__(self, mha, kv_b_proj, mxfp8):
        self.mha = mha
        self.kv_b_proj = kv_b_proj
        self.out_scale = None
        self.kv_lora_rank, self.qk_rope_head_dim = LORA, ROPE
        self.qk_nope_head_dim, self.v_head_dim = NOPE, V
        self.num_heads_tp = H
        self._mxfp8 = mxfp8

    def _kv_b_takes_mxfp8(self):
        return self._mxfp8

    def forward(self, case):
        out = torch.empty(case.T, H * V, dtype=torch.bfloat16, device="cuda")
        self._forward_context_split_k_fp4(
            case.q, case.latent[:, :LORA].contiguous(), case.latent, case.meta, out
        )
        torch.cuda.synchronize()
        return out


def _unit_quant(x):
    unit = torch.ones(1, dtype=torch.float32, device=x.device)
    return torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x.contiguous(), unit)[0]


_CASES = [
    ([5000], [300], 1024, 2),  # slices 2048, 2048, 904 of one request
    ([1500, 3000, 700], [128, 64, 200], 1024, 2),  # slices span requests
    ([66000], [256], 16384, 4),  # production slice size: 65536 + 464
]


# ---------------------------------------------------------------------------------------------
# GPU: the partition schedule
# ---------------------------------------------------------------------------------------------


@requires_fp8_cvt
@pytest.mark.parametrize("triton_quant", ["0", "1"])
def test_split_k_partitions_follow_the_chunk_schedule(monkeypatch, k3_features, triton_quant):
    """Every cached-prefix slice and then the current chunk reach the FMHA once, with their KV
    lengths, merge ops and mask, unconcatenated FP8 K-nope / K-pe / V, and one FP8 Q."""
    import tensorrt_llm  # noqa: F401

    k3_features(mla_ctx_triton_quant=triton_quant == "1")
    case = _build_prefix_case([1500, 3000, 700], [128, 64, 200], 1024, 2)
    loaded = []

    def loader(tokens):
        latent = (torch.randn(tokens, LORA + ROPE, device="cuda") * 0.5).to(torch.bfloat16)
        loaded.append(latent)
        return latent[:, :LORA].contiguous(), latent[:, LORA:].contiguous()

    calls = []

    def record(attn, q, k_nope, k_pe, v, output, stats, merge_op, cu_q, cu_kv, **kwargs):
        calls.append(
            dict(
                q=q,
                k_nope=k_nope,
                k_pe=k_pe,
                v=v,
                output=output,
                stats=stats,
                merge_op=merge_op,
                cu_q=cu_q,
                cu_kv=cu_kv,
                **kwargs,
            )
        )

    monkeypatch.setattr(ctx_split_k, "run_k3_split_k_context_partition", record)
    kv_b_proj = _KvBProj()
    layer = _SplitKLayer(_Mha(loader), kv_b_proj, mxfp8=False)
    out = layer.forward(case)

    meta = case.meta
    assert len(calls) == case.loops + 1 == len(loaded) + 1
    q_fp8 = calls[0]["q"]
    assert q_fp8.dtype == torch.float8_e4m3fn
    assert torch.equal(q_fp8.view(torch.uint8), _unit_quant(case.q).view(torch.uint8))
    stats = layer.softmax_stats_tensor
    assert stats.shape == (case.T, H, 2) and stats.dtype == torch.float32
    for loop_idx, call in enumerate(calls):
        last = loop_idx == case.loops
        latent = case.latent if last else loaded[loop_idx]
        assert call["q"] is q_fp8 and call["stats"] is stats and call["cu_q"] is case.cu_q
        assert call["output"].data_ptr() == out.data_ptr()
        assert call["is_causal"] == last
        assert call["max_q_len"] == meta.max_ctx_seq_len
        assert call["merge_op"].data_ptr() == meta.merge_op_tensor[loop_idx].data_ptr()
        if last:
            assert call["cu_kv"] is case.cu_q and call["max_kv_len"] == meta.max_ctx_seq_len
        else:
            assert call["cu_kv"].data_ptr() == meta.cu_chunked_seq_len[loop_idx].data_ptr()
            assert call["max_kv_len"] == meta.max_chunk_len_per_loop[loop_idx]
        kv = _unit_quant(kv_b_proj(latent[:, :LORA].contiguous()))
        assert torch.equal(call["k_nope"].view(torch.uint8), kv[:, : H * NOPE].view(torch.uint8))
        assert torch.equal(call["v"].view(torch.uint8), kv[:, H * NOPE :].view(torch.uint8))
        assert call["k_nope"].stride(-1) == 1 and call["v"].stride(-1) == 1
        assert call["k_pe"].shape == (latent.shape[0], ROPE)
        assert torch.equal(
            call["k_pe"].view(torch.uint8), _unit_quant(latent[:, LORA:]).view(torch.uint8)
        )


# ---------------------------------------------------------------------------------------------
# SM107: the MXFP8 gather and the whole chunked prefill
# ---------------------------------------------------------------------------------------------


@requires_sm107
@pytest.mark.parametrize("global_scale", [1.0, 32.0])
@pytest.mark.parametrize("prefix,new,chunk_size,buffer_batch", _CASES[:2])
def test_mxfp8_gather_matches_the_quantized_dense_gather(
    prefix, new, chunk_size, buffer_batch, global_scale
):
    """Per partition, the MXFP8 latent (every scale byte, padding included) and the FP8 K-pe are
    the C++ MXFP8 / unit FP8 quantization of the dense BF16 gather (power-of-two global scales
    keep the division exact in both)."""
    import tensorrt_llm  # noqa: F401

    case = _build_prefix_case(prefix, new, chunk_size, buffer_batch, global_scale=global_scale)
    meta = case.meta
    for loop_idx in range(case.loops):
        tokens = int(meta.host_cu_chunked_seq_len[loop_idx, meta.num_contexts])
        args = dict(
            cu=meta.cu_chunked_seq_len[loop_idx],
            offsets=meta.chunked_global_offset[loop_idx],
            max_len=int(meta.max_chunk_len_per_loop[loop_idx]),
        )
        (latent, latent_sf), k_pe = ctx_split_k.load_fp4_mla_chunked_kv_cache_mxfp8(
            meta,
            0,
            num_ctx_cached_tokens=tokens,
            cu_chunked_seq_len=args["cu"],
            chunked_global_offset=args["offsets"],
            chunked_max_seq_len=args["max_len"],
            kv_lora_rank=LORA,
            qk_rope_head_dim=ROPE,
        )
        dense, dense_pe = _dense_gather(meta, tokens=tokens, **args)
        expected, expected_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(dense)
        assert torch.equal(latent.view(torch.uint8), expected.view(torch.uint8))
        assert torch.equal(latent_sf.view(torch.uint8), expected_sf.view(torch.uint8))
        assert torch.equal(k_pe.view(torch.uint8), _unit_quant(dense_pe).view(torch.uint8))


@requires_sm107
def test_mxfp8_gather_validates_its_inputs():
    import tensorrt_llm  # noqa: F401

    case = _build_prefix_case([300], [10], 1024, 1)
    meta = case.meta
    kwargs = dict(
        num_ctx_cached_tokens=300,
        cu_chunked_seq_len=meta.cu_chunked_seq_len[0],
        chunked_global_offset=meta.chunked_global_offset[0],
        chunked_max_seq_len=300,
        kv_lora_rank=LORA,
        qk_rope_head_dim=ROPE,
    )
    with pytest.raises(ValueError, match="kv_lora_rank=512"):
        ctx_split_k.load_fp4_mla_chunked_kv_cache_mxfp8(meta, 0, **{**kwargs, "kv_lora_rank": 256})
    with pytest.raises(ValueError, match="int64 cumulative-length"):
        ctx_split_k.load_fp4_mla_chunked_kv_cache_mxfp8(
            meta, 0, **{**kwargs, "cu_chunked_seq_len": meta.cu_chunked_seq_len[0].int()}
        )
    (latent, latent_sf), k_pe = ctx_split_k.load_fp4_mla_chunked_kv_cache_mxfp8(
        meta, 0, **{**kwargs, "num_ctx_cached_tokens": 0}
    )
    assert latent.shape == (0, LORA) and latent_sf.numel() == 0 and k_pe.shape == (0, ROPE)


def _reference(case, kv_b_proj, mxfp8):
    """FP32 attention over the FP8 Q / K / V the FMHA consumes, the whole prefix in one piece."""
    q8 = _unit_quant(case.q).float().view(case.T, H, NOPE + ROPE)

    def expand(latent):
        if mxfp8:
            a, a_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(latent)
            return torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(
                a,
                kv_b_proj.weight,
                a_sf,
                kv_b_proj.weight_scale,
                output_dtype=torch.float8_e4m3fn,
            ).float()
        return _unit_quant(kv_b_proj(latent)).float()

    cur_kv = expand(case.latent[:, :LORA].contiguous())
    cur_pe = _unit_quant(case.latent[:, LORA:]).float()
    out = torch.empty(case.T, H, V, device="cuda")
    q0 = 0
    for r, (p, n) in enumerate(zip(case.prefix, case.new)):
        meta = SimpleNamespace(**vars(case.meta))
        meta.num_contexts = 1
        meta.fp4_mla_state = SimpleNamespace(**vars(case.meta.fp4_mla_state))
        meta.fp4_mla_state._paged_kv_indices = torch.tensor(
            case.rows[r * case.stride : (r + 1) * case.stride], dtype=torch.int32, device="cuda"
        )
        dense, dense_pe = _dense_gather(
            meta,
            tokens=p,
            cu=torch.tensor([0, p], dtype=torch.int64, device="cuda"),
            offsets=torch.zeros(1, dtype=torch.int64, device="cuda"),
            max_len=p,
        )
        kv = torch.cat((expand(dense), cur_kv[q0 : q0 + n]))
        pe = torch.cat((_unit_quant(dense_pe).float(), cur_pe[q0 : q0 + n]))
        k_nope = kv[:, : H * NOPE].view(-1, H, NOPE)
        v = kv[:, H * NOPE :].view(-1, H, V)
        causal = torch.arange(n, device="cuda")[None, :] > torch.arange(n, device="cuda")[:, None]
        for h0 in range(0, H, 16):
            heads = slice(h0, h0 + 16)
            scores = torch.einsum(
                "qhd,khd->hqk", q8[q0 : q0 + n, heads, :NOPE], k_nope[:, heads]
            ) + torch.einsum("qhd,kd->hqk", q8[q0 : q0 + n, heads, NOPE:], pe)
            scores /= math.sqrt(NOPE + ROPE)
            scores[:, :, p:].masked_fill_(causal[None], -torch.inf)
            out[q0 : q0 + n, heads] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v[:, heads])
        q0 += n
    return out.view(case.T, H * V)


@requires_sm107
@pytest.mark.parametrize("mxfp8", [True, False], ids=["mxfp8_gather", "bf16_gather"])
@pytest.mark.parametrize("prefix,new,chunk_size,buffer_batch", _CASES)
def test_split_k_chunked_prefill_matches_reference(
    k3_features, prefix, new, chunk_size, buffer_batch, mxfp8
):
    """Output against an FP32 reference; features.mla_ctx_triton_quant does not change a byte."""
    import tensorrt_llm  # noqa: F401

    reason = ctx_split_k.split_k_context_unavailable_reason(_Mha())
    if reason is not None:
        pytest.skip(f"K3 split-K context FMHA unavailable: {reason}")
    k3_features(mla_ctx_triton_quant=False)
    kv_b_proj = _KvBProj(mxfp8)
    layer = _SplitKLayer(_Mha(), kv_b_proj, mxfp8)
    case = _build_prefix_case(prefix, new, chunk_size, buffer_batch)
    out = layer.forward(case)
    k3_features(mla_ctx_triton_quant=True)
    assert torch.equal(layer.forward(case).view(torch.int16), out.view(torch.int16))
    torch.testing.assert_close(
        out.float(), _reference(case, kv_b_proj, mxfp8), atol=5e-2, rtol=5e-2
    )
