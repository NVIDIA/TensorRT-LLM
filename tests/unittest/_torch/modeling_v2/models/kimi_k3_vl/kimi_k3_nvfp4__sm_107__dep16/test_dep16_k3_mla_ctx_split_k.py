# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 dep16 MLA cached-prefix context on the Rubin FP8 split-K context FMHA
(``K3MLAAttention`` of kimi_k3_nvfp4__sm_107__dep16/modeling.py).

CPU: which layers and calls take the split-K path and when kv_b_proj takes the MXFP8 cache
gather. GPU (SM89+): the partition schedule handed to the FMHA. SM107: the MXFP8 gather against
the dense gather plus the C++ FP8 quantizers, and the whole chunked prefill on a real FP4 cache
against an FP32 reference."""

import math
from types import SimpleNamespace

import pytest
import torch

__extra_import_path__ = ["../../.."]

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_mla_ctx import (
    k3_context_fmha_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__dep16 import (
    modeling as context,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__dep16 import (
    modeling as ctx_split_k,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention.k3_mla_state import (
    K3MlaAttnState,
)

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
    assert "cannot attend a cached prefix" in info_once[-1][0]


def _mla_state(*, fp4=True):
    state = object.__new__(K3MlaAttnState)
    state.has_fp4_kv_cache = fp4
    state.has_fp8_kv_cache = not fp4
    return state


def _layer(mha, kv_b_proj=None):
    """The attributes of the MLA layer the split-K context reads."""
    return SimpleNamespace(
        mha=mha,
        kv_b_proj=kv_b_proj,
        out_scale=None,
        kv_lora_rank=LORA,
        qk_rope_head_dim=ROPE,
        qk_nope_head_dim=NOPE,
        v_head_dim=V,
        num_heads_tp=H,
    )


@pytest.mark.cpu_only
def test_split_k_serves_bf16_fp4_cache_layers_only(monkeypatch):
    supported = {"value": True}
    monkeypatch.setattr(
        ctx_split_k, "supports_k3_fp8_split_k_context", lambda attn: supported["value"]
    )
    bf16 = torch.empty(1, dtype=torch.bfloat16)

    def uses(layer, q=bf16, output=bf16):
        return ctx_split_k.uses_k3_split_k_context(layer, q, output)

    layer = _layer(_mla_state())
    assert uses(layer)
    assert not uses(layer, q=bf16.half())
    assert not uses(layer, output=bf16.float())
    layer.out_scale = torch.ones(1)
    assert not uses(layer)
    layer.out_scale = None
    # Only NVFP4 latent caches (not FP8 layers) and only the family MLA state.
    assert not uses(_layer(_mla_state(fp4=False)))
    assert not uses(_layer(SimpleNamespace(uses_fp4_mla_attention=True)))
    supported["value"] = False
    assert not uses(layer)


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
    mha = _mla_state()

    def takes(**overrides):
        return ctx_split_k.kv_b_takes_mxfp8(_layer(mha, _kv_b_proj(**overrides)))

    assert takes()
    assert not takes(supports_prequantized_input=False)
    assert not takes(bias=torch.zeros(2))
    assert not takes(weight=torch.empty(2, 2, dtype=torch.bfloat16))
    assert not takes(weight_scale=torch.empty(2, dtype=torch.int32))
    assert not takes(weight_scale=None)
    layer = _layer(mha, _kv_b_proj())
    layer.kv_lora_rank = 256
    assert not ctx_split_k.kv_b_takes_mxfp8(layer)
    monkeypatch.setattr(ctx_split_k, "IS_CUTLASS_DSL_RUBIN_AVAILABLE", False)
    assert not takes()
    monkeypatch.setattr(ctx_split_k, "IS_CUTLASS_DSL_RUBIN_AVAILABLE", True)
    monkeypatch.setattr(ctx_split_k, "get_sm_version", lambda: 103)
    assert not takes()


@pytest.mark.cpu_only
def test_fp4_cache_chunked_prefill_runs_split_k_or_raises(monkeypatch):
    """The context dispatch hands an FP4-cache chunked prefill to the split-K context FMHA, and
    raises where it does not apply (there is no other FP4 partition loop)."""
    monkeypatch.setattr(context, "get_sm_version", lambda: 107)
    applies = {"value": True}
    monkeypatch.setattr(
        ctx_split_k, "uses_k3_split_k_context", lambda layer, q, output: applies["value"]
    )
    monkeypatch.setattr(
        ctx_split_k, "forward_context_split_k_fp4", lambda layer, q, ckv, latent, md, out: "split_k"
    )
    mha = SimpleNamespace(
        uses_fp4_mla_attention=True,
        has_cached_kv_for_mla_context_warmup=lambda md: False,
        is_chunked_prefill_for_mla_context=lambda md: True,
        has_cached_kv_for_mla_context=lambda md: True,
    )
    layer = _layer(mha)
    md = SimpleNamespace()
    run = lambda: context.forward_context(layer, None, None, md, None, None)  # noqa: E731
    assert run() == "split_k"
    applies["value"] = False
    with pytest.raises(NotImplementedError, match="split-K context FMHA"):
        run()
    mha.is_chunked_prefill_for_mla_context = lambda md: False
    with pytest.raises(RuntimeError, match="enable chunked prefill"):
        run()


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
    """The MLA state and the family cache ops of the split-K path, over the case's FP4 cache
    (:func:`_split_k_forward` routes ``mla_cache``'s calls here)."""

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


def _split_k_forward(monkeypatch, mha, kv_b_proj, mxfp8, case):
    """``ctx_split_k.forward_context_split_k_fp4`` of one case, with kv_b on the MXFP8 gather or
    not."""
    monkeypatch.setattr(ctx_split_k, "kv_b_takes_mxfp8", lambda layer: mxfp8)
    monkeypatch.setattr(
        ctx_split_k,
        "mla_cache",
        SimpleNamespace(
            append_fp4_mla_context_cache=lambda state, *args: state.append_fp4_mla_context_cache(
                *args
            ),
            load_chunked_kv_cache_for_mla=lambda state, metadata, **kwargs: (
                state.load_chunked_kv_cache_for_mla(metadata, **kwargs)
            ),
        ),
    )
    out = torch.empty(case.T, H * V, dtype=torch.bfloat16, device="cuda")
    ctx_split_k.forward_context_split_k_fp4(
        _layer(mha, kv_b_proj),
        case.q,
        case.latent[:, :LORA].contiguous(),
        case.latent,
        case.meta,
        out,
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
def test_split_k_partitions_follow_the_chunk_schedule(monkeypatch):
    """Every cached-prefix slice and then the current chunk reach the FMHA once, with their KV
    lengths, merge ops and mask, unconcatenated FP8 K-nope / K-pe / V, and one FP8 Q."""
    import tensorrt_llm  # noqa: F401

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
    out = _split_k_forward(monkeypatch, _Mha(loader), kv_b_proj, False, case)

    meta = case.meta
    assert len(calls) == case.loops + 1 == len(loaded) + 1
    q_fp8 = calls[0]["q"]
    assert q_fp8.dtype == torch.float8_e4m3fn
    assert torch.equal(q_fp8.view(torch.uint8), _unit_quant(case.q).view(torch.uint8))
    stats = calls[0]["stats"]
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
            return _unit_quant(
                torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(
                    a, kv_b_proj.weight, a_sf, kv_b_proj.weight_scale
                )
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
    prefix, new, chunk_size, buffer_batch, mxfp8, monkeypatch
):
    """Output against an FP32 reference over the C++ FP8 quantization of Q / K / V (the Triton
    quantizers' bytes are those, see ``test_k3_mla_ctx_fp8.py``)."""
    import tensorrt_llm  # noqa: F401

    reason = ctx_split_k.split_k_context_unavailable_reason(_Mha())
    if reason is not None:
        pytest.skip(f"K3 split-K context FMHA unavailable: {reason}")
    kv_b_proj = _KvBProj(mxfp8)
    case = _build_prefix_case(prefix, new, chunk_size, buffer_batch)
    out = _split_k_forward(monkeypatch, _Mha(), kv_b_proj, mxfp8, case)
    torch.testing.assert_close(
        out.float(), _reference(case, kv_b_proj, mxfp8), atol=5e-2, rtol=5e-2
    )


# ---------------------------------------------------------------------------------------------
# SM107: the fp4_mla_gather_mxfp8 entry on its cells
# ---------------------------------------------------------------------------------------------


def _gather_entry():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention.fp4_mla_gather_mxfp8 import (  # noqa: E501
        fp4_mla_gather_mxfp8,
    )

    return fp4_mla_gather_mxfp8


@requires_sm107
@pytest.mark.parametrize(
    "cell", _gather_entry().CELLS, ids=[c.why[:48] for c in _gather_entry().CELLS]
)
def test_fp4_mla_gather_mxfp8_cells(cell):
    """Every partition of the cell's chunk schedule under ``validating``: K-pe and the MXFP8
    latent within one E4M3 step of the reference (the kernel's approximate division by the
    global scale and reciprocal of 448 can move a BF16 value by an ulp, or a block's exponent by
    one at a power-of-two boundary)."""
    from _validating import validating

    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.attention.backends.fp4_mla import FP4_MLA_KV_GLOBAL_SCALE

    op = _gather_entry()
    spec = cell.spec
    case = _build_prefix_case(
        spec["prefix"],
        spec["new"],
        spec["chunk_size"],
        spec["buffer_batch"],
        global_scale=FP4_MLA_KV_GLOBAL_SCALE,
    )
    meta = case.meta
    kv, sf = meta.kv_cache_manager.get_fp4_mla_cache_buffers(0)
    state = meta.fp4_mla_state
    for loop_idx in range(case.loops):
        tokens = int(meta.host_cu_chunked_seq_len[loop_idx, meta.num_contexts])
        a = torch.empty((tokens, LORA), dtype=torch.float8_e4m3fn, device="cuda")
        asc = torch.full(((tokens + 127) // 128 * 128 * 16,), 0xFF, dtype=torch.uint8,
                         device="cuda")  # fmt: skip
        pe = torch.empty((tokens, ROPE), dtype=torch.float8_e4m3fn, device="cuda")
        args = (a, asc, pe, kv, sf.view(torch.float8_e4m3fn), state._paged_kv_indices,
                meta.cu_chunked_seq_len[loop_idx], meta.chunked_global_offset[loop_idx],
                state.kv_global_scale, int(meta.max_chunk_len_per_loop[loop_idx]), PAGE,
                state.page_table_stride, meta.num_contexts)  # fmt: skip
        ref_a, ref_asc, ref_pe = op.reference(*args)
        with validating(op):
            op(*args)
        torch.cuda.synchronize()
        # The kernel divides by the global scale with div.full (2 ulp); a BF16 rounding can then
        # move a K-pe value by one E4M3 step (2^-3 relative), nothing more.
        pe_err = (pe.float() - ref_pe.float()).abs()
        assert bool((pe_err <= ref_pe.float().abs() * 2.0**-3 + 2.0**-9).all()), cell.why
        assert int((asc.int() - ref_asc.int()).abs().max()) <= 1, cell.why

        def dequant(codes, scales):
            from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention.k3_mla_q_fused import (  # noqa: E501
                swizzled_sf_offset,
            )

            rows = torch.arange(tokens, device="cuda")[:, None]
            offsets = swizzled_sf_offset(rows, torch.arange(16, device="cuda")[None, :], 16)
            exponent = scales[offsets].float().repeat_interleave(32, -1)
            return codes.float() * torch.exp2(exponent - 127)

        got, ref = dequant(a, asc), dequant(ref_a, ref_asc)
        step = ref.abs().view(tokens, 4, 128).amax(-1, keepdim=True) * 2.0**-3
        assert bool(((got - ref).abs().view(tokens, 4, 128) <= step).all()), cell.why
