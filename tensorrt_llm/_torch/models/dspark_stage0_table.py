# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSpark stage-0 token table: the attention-TP draft's first-stage inputs, which depend only on the
token id up to the RoPE; filled by the live kernels, used only if bit-identical to them."""

from __future__ import annotations

import functools
from typing import Callable, NamedTuple, Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm.logger import logger

# Rows of the attention's draft-block operand (``[G, 8, head_dim]``, zero padded).
DRAFT_BLOCK_STORAGE = 8
# Draft blocks per fill launch, at most ``_FILL_MAX_ROWS`` rows.
_FILL_BLOCKS = 12
# Row cap of the small-M FP8 kernels the fill runs.
_FILL_MAX_ROWS = 64
# Bonus tokens the check runs through the live chain.
_CHECK_TOKENS = 64
# Contractions of ``a * b -/+ c * d``: unfused (0), first product fused (1), second fused (2).
_ROPE_MODES = (1, 0, 2)


class Stage0Weights(NamedTuple):
    """The live stage-0 operands of one attention-TP rank."""

    embed: torch.Tensor  # [V, hidden] BF16
    hc_mult: int
    entry_norm_w: torch.Tensor  # [hidden] (the stage's input_layernorm)
    entry_norm_eps: float
    qkv_weight: torch.Tensor  # [q_rank + head_dim, hidden] E4M3 ([wq_a; wkv])
    qkv_scale: torch.Tensor  # [q_rank + head_dim, hidden / 32] UE8M0, linear
    q_rank: int
    q_norm_w: torch.Tensor  # [q_rank] BF16
    kv_norm_w: torch.Tensor  # [head_dim] BF16
    eps: float
    wq_b_weight: torch.Tensor  # [H * head_dim, q_rank] E4M3
    wq_b_scale: torch.Tensor  # [H * head_dim, q_rank / 32] UE8M0, linear
    heads: int
    head_dim: int
    rope_dim: int
    # Entry ``lagged_coefficients``: ``(pre, post, comb)`` of a ``[M, hc, hidden]`` residual.
    coefficients: Optional[Callable] = None


class Stage0Table:
    """Per-token stage-0 values of this rank's query heads (row ``vocab``: the zero embedding): BF16
    non-RoPE columns, FP32 RoPE pairs and, with ``hc_mult``, FP32 entry coefficients."""

    def __init__(
        self, vocab: int, heads: int, head_dim: int, rope_dim: int, device, hc_mult: int = 0
    ):
        self.vocab, self.heads, self.head_dim, self.rope_dim = vocab, heads, head_dim, rope_dim
        rows, nope = vocab + 1, head_dim - rope_dim
        self.q_nope = torch.empty((rows, heads, nope), dtype=torch.bfloat16, device=device)
        self.q_rope = torch.empty((rows, heads, rope_dim), dtype=torch.float32, device=device)
        self.kv_nope = torch.empty((rows, nope), dtype=torch.bfloat16, device=device)
        self.kv_rope = torch.empty((rows, rope_dim), dtype=torch.float32, device=device)
        self.hc_mult = hc_mult
        self.coeff = (
            torch.empty((rows, (2 + hc_mult) * hc_mult), dtype=torch.float32, device=device)
            if hc_mult
            else None
        )
        # How the query / K/V kernels contract their RoPE products (set by the check).
        self.modes: Optional[dict] = None
        # Whether the gathered entry coefficients equal the live kernels' (set by the check).
        self.coeff_ready = False

    @property
    def nbytes(self) -> int:
        tensors = (self.q_nope, self.q_rope, self.kv_nope, self.kv_rope, self.coeff)
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)


# The live chain: the entry collapse and ``dspark_attention_forward_tp``'s pre-attention calls.


def _residual(w: Stage0Weights, tokens: torch.Tensor) -> torch.Tensor:
    """``[M, hc, hidden]`` residual of ``tokens`` (ids outside the vocabulary embed as zeros)."""
    vocab, hidden = w.embed.shape
    valid = (tokens >= 0) & (tokens < vocab)
    emb = w.embed[torch.where(valid, tokens, torch.zeros_like(tokens))]
    emb = torch.where(valid.unsqueeze(1), emb, torch.zeros_like(emb))
    return emb.unsqueeze(1).expand(tokens.shape[0], w.hc_mult, hidden).contiguous()


def _prefix(w: Stage0Weights, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q_a [M, q_rank], kv [M, head_dim])`` of ``tokens``: the entry collapse and ``[wq_a; wkv]``
    as the live chain runs them."""
    from ..modules.mhc.mhc_lowm import mhc_post_collapse_norm
    from .modeling_dspark import _dspark_tp_qkv

    pre = torch.zeros((tokens.shape[0], w.hc_mult), dtype=torch.float32, device=tokens.device)
    pre[:, 0] = 1.0
    _, layer_input = mhc_post_collapse_norm(
        None, _residual(w, tokens), None, None, pre, w.entry_norm_w, w.entry_norm_eps
    )
    return _dspark_tp_qkv(layer_input, (w.qkv_weight, w.qkv_scale, w.q_rank))


def live_stage0(
    w: Stage0Weights,
    tokens: torch.Tensor,
    block_rope_freqs: torch.Tensor,
    rope_table: torch.Tensor,
    *,
    block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q [M, H * D], draft_block [M / block, 8, D])`` BF16 of ``tokens [M]`` by the live chain;
    ``block_rope_freqs [M, R / 2, 2]`` and ``rope_table [M, 2, R / 2]``: the step's RoPE inputs."""
    from ..custom_ops.dspark_rmsnorm_rope_custom_op import cute_dsl_dspark_rmsnorm_rope_draft_block
    from .modeling_dspark import _dspark_tp_query

    rows = tokens.shape[0]
    q_a, kv = _prefix(w, tokens)
    positions = torch.arange(rows, dtype=torch.int32, device=tokens.device)
    q = _dspark_tp_query(
        q_a,
        w.q_norm_w,
        w.eps,
        w.wq_b_weight,
        w.wq_b_scale,
        torch.view_as_complex(block_rope_freqs).view(rows // block, block, -1),
        rope=(positions, rope_table, w.head_dim),
    )
    draft_block = cute_dsl_dspark_rmsnorm_rope_draft_block(
        kv.view(rows // block, block, -1), w.kv_norm_w, block_rope_freqs, w.eps
    )
    return q, draft_block


def _wq_b_fp32(
    w: Stage0Weights, q_in: torch.Tensor, rms_norm: Optional[tuple[torch.Tensor, float]]
) -> torch.Tensor:
    """``wq_b``'s unrounded FP32 outputs for ``_dspark_tp_query_input``'s ``(q_in, rms_norm)``: the
    values its RoPE epilogue rotates."""
    import cuda.bindings.driver as cuda

    from ..cute_dsl_utils import graph_safe_from_dlpack
    from ..flashinfer_utils import get_env_enable_pdl
    from ..modules.low_m_gemm import _compiled_mma, _no_gamma, _rope_placeholders, lowm_mma_options

    m, k = q_in.shape
    n = w.wq_b_weight.shape[0]
    bf16_dequant, swap_ab = lowm_mma_options()
    out = torch.empty((m, n), dtype=torch.float32, device=q_in.device)
    fn = _compiled_mma(
        m,
        n,
        k,
        get_env_enable_pdl(),
        0,
        out_dtype=torch.float32,
        a_rms_eps=None if rms_norm is None else float(rms_norm[1]),
        bf16_dequant=bf16_dequant,
        swap_ab=swap_ab,
    )
    positions, cos_sin = _rope_placeholders(m, q_in.device)
    gamma = _no_gamma(q_in.device) if rms_norm is None else rms_norm[0].view(1, k)
    fn(
        graph_safe_from_dlpack(q_in),
        graph_safe_from_dlpack(w.wq_b_weight.view(torch.uint8)),
        graph_safe_from_dlpack(w.wq_b_scale),
        graph_safe_from_dlpack(out),
        graph_safe_from_dlpack(out),
        graph_safe_from_dlpack(positions),
        graph_safe_from_dlpack(cos_sin),
        graph_safe_from_dlpack(gamma),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    return out


@functools.lru_cache(maxsize=None)
def _compile_kv_norm_fp32(head_dim: int, eps: float):
    """``DSparkRMSNormRoPEKernel`` without RoPE and with unrounded FP32 outputs."""
    import cutlass
    import cutlass.cute as cute

    from ..attention.kernels.blackwell.dspark.rmsnorm_rope import DSparkRMSNormRoPEKernel

    rows = cute.sym_int()

    def fake(dtype, shape, order):
        return cute.runtime.make_fake_compact_tensor(dtype, shape, stride_order=order)

    return cute.compile(
        DSparkRMSNormRoPEKernel(head_dim, 0, 1, eps, True, True, False),
        fake(cutlass.BFloat16, (rows, head_dim), (1, 0)),
        fake(cutlass.BFloat16, (head_dim,), (0,)),
        fake(cutlass.Float32, (cute.sym_int(), cute.sym_int(), 2), (2, 1, 0)),
        fake(cutlass.Float32, (rows, head_dim), (1, 0)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )


def _kv_norm_fp32(kv: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The draft-block kernel's weighted RMSNorm of ``kv`` rows in FP32, before its RoPE."""
    rows, dim = kv.shape
    out = torch.empty((rows, dim), dtype=torch.float32, device=kv.device)
    freqs = torch.zeros((rows, 1, 2), dtype=torch.float32, device=kv.device)
    _compile_kv_norm_fp32(dim, float(eps))(kv.contiguous(), weight.contiguous(), freqs, out)
    return out


def fill_stage0_table(table: Stage0Table, w: Stage0Weights, block: int) -> None:
    """Fill ``table`` from the live kernels over every token id (eager, outside graph capture)."""
    from ..custom_ops.dspark_rmsnorm_rope_custom_op import cute_dsl_dspark_rmsnorm_rope_draft_block
    from .modeling_dspark import _dspark_tp_query, _dspark_tp_query_input

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("the DSpark stage-0 table is filled outside CUDA-graph capture")
    blocks = max(1, min(_FILL_BLOCKS, _FILL_MAX_ROWS // block))
    chunk = blocks * block
    device = w.embed.device
    total = table.vocab + 1
    nope = table.head_dim - table.rope_dim
    half = table.rope_dim // 2
    # Identity rotation: only the non-RoPE columns of these launches are kept.
    block_freqs = torch.zeros((chunk, half, 2), dtype=torch.float32, device=device)
    block_freqs[..., 0] = 1.0
    query_freqs = torch.view_as_complex(block_freqs).view(blocks, block, half)
    with torch.inference_mode():
        for start in range(0, total, chunk):
            n = min(chunk, total - start)
            # Ids from the vocabulary up embed as zeros (row ``vocab``).
            tokens = torch.arange(start, start + chunk, dtype=torch.int64, device=device)
            q_a, kv = _prefix(w, tokens)
            q_bf16 = _dspark_tp_query(
                q_a, w.q_norm_w, w.eps, w.wq_b_weight, w.wq_b_scale, query_freqs
            )
            q_f32 = _wq_b_fp32(w, *_dspark_tp_query_input(q_a, w.q_norm_w, w.eps, query_freqs))
            blk = cute_dsl_dspark_rmsnorm_rope_draft_block(
                kv.view(blocks, block, -1), w.kv_norm_w, block_freqs, w.eps
            )[:, :block].reshape(chunk, table.head_dim)
            kv_f32 = _kv_norm_fp32(kv, w.kv_norm_w, w.eps)
            heads = (chunk, table.heads, table.head_dim)
            table.q_nope[start : start + n].copy_(q_bf16.view(heads)[:n, :, :nope])
            table.q_rope[start : start + n].copy_(q_f32.view(heads)[:n, :, nope:])
            table.kv_nope[start : start + n].copy_(blk[:n, :nope])
            table.kv_rope[start : start + n].copy_(kv_f32[:n, nope:])
        if table.coeff is not None and w.coefficients is not None:
            _fill_coefficients(table, w, block)


def _fill_coefficients(table: Stage0Table, w: Stage0Weights, block: int) -> None:
    """Every token id's entry coefficients, a draft block per launch as in a decode step."""
    hc = table.hc_mult
    total = table.vocab + 1
    device = table.coeff.device
    ids = torch.arange(block, dtype=torch.int64, device=device)
    tokens = ids.clone()
    out = torch.empty((block, (2 + hc) * hc), dtype=torch.float32, device=device)

    def run() -> torch.dtype:
        pre, post, comb = w.coefficients(_residual(w, tokens))
        out[:, :hc].copy_(pre.reshape(block, hc))
        out[:, hc : 2 * hc].copy_(post.reshape(block, hc))
        out[:, 2 * hc :].copy_(comb.reshape(block, hc * hc))
        return pre.dtype

    if run() != torch.float32:
        table.coeff = None
        return
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for start in range(0, total, block):
        n = min(block, total - start)
        torch.add(ids, start, out=tokens)
        graph.replay()
        table.coeff[start : start + n].copy_(out[:n])


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _fma_rn(a, b, c):
    return tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", [a, b, c], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _rope_re(re, im, cos, sin, MODE: tl.constexpr):
    if MODE == 0:
        out = _add_rn(_mul_rn(re, cos), -_mul_rn(im, sin))
    elif MODE == 1:
        out = _fma_rn(re, cos, -_mul_rn(im, sin))
    else:
        out = _fma_rn(-im, sin, _mul_rn(re, cos))
    return out


@triton.jit
def _rope_im(re, im, cos, sin, MODE: tl.constexpr):
    if MODE == 0:
        out = _add_rn(_mul_rn(im, cos), _mul_rn(re, sin))
    elif MODE == 1:
        out = _fma_rn(im, cos, _mul_rn(re, sin))
    else:
        out = _fma_rn(re, sin, _mul_rn(im, cos))
    return out


@triton.jit
def _stage0_gather_kernel(
    bonus_ptr,
    freqs_ptr,
    q_nope_ptr,
    q_rope_ptr,
    kv_nope_ptr,
    kv_rope_ptr,
    coeff_ptr,
    q_out_ptr,
    block_out_ptr,
    table_out_ptr,
    pre_out_ptr,
    post_out_ptr,
    comb_out_ptr,
    noise_id,
    vocab,
    HEADS: tl.constexpr,
    NOPE: tl.constexpr,
    NOPE_P2: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK: tl.constexpr,
    STORAGE: tl.constexpr,
    Q_RE: tl.constexpr,
    Q_IM: tl.constexpr,
    KV_RE: tl.constexpr,
    KV_IM: tl.constexpr,
    HC: tl.constexpr,
    COEFF_P2: tl.constexpr,
    WAIT: tl.constexpr,
):
    """Program ``(row, head)``: the row's query head with RoPE; head ``HEADS``: its draft-block K/V,
    rotation-table row, entry coefficients (``HC > 0``) and (a request's row 0) the zero padding."""
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    request = row // BLOCK
    j = row - request * BLOCK
    if WAIT:
        # Every read below may be of a predecessor's output.
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    bonus = tl.load(bonus_ptr + request)
    raw = tl.where(j == 0, bonus, noise_id.to(tl.int64))
    token = tl.where((raw >= 0) & (raw < vocab), raw, vocab)
    pairs = tl.arange(0, HALF)
    cos = tl.load(freqs_ptr + row * (2 * HALF) + 2 * pairs)
    sin = tl.load(freqs_ptr + row * (2 * HALF) + 2 * pairs + 1)
    dim: tl.constexpr = NOPE + 2 * HALF
    cols = tl.arange(0, NOPE_P2)
    cmask = cols < NOPE
    if head < HEADS:
        src = token * HEADS + head
        dst = q_out_ptr + (row * HEADS + head) * dim
        tl.store(dst + cols, tl.load(q_nope_ptr + src * NOPE + cols, mask=cmask), mask=cmask)
        re = tl.load(q_rope_ptr + src * (2 * HALF) + 2 * pairs)
        im = tl.load(q_rope_ptr + src * (2 * HALF) + 2 * pairs + 1)
        tl.store(dst + NOPE + 2 * pairs, _rope_re(re, im, cos, sin, Q_RE).to(tl.bfloat16))
        tl.store(dst + NOPE + 2 * pairs + 1, _rope_im(re, im, cos, sin, Q_IM).to(tl.bfloat16))
    else:
        dst = block_out_ptr + (request * STORAGE + j) * dim
        tl.store(dst + cols, tl.load(kv_nope_ptr + token * NOPE + cols, mask=cmask), mask=cmask)
        re = tl.load(kv_rope_ptr + token * (2 * HALF) + 2 * pairs)
        im = tl.load(kv_rope_ptr + token * (2 * HALF) + 2 * pairs + 1)
        tl.store(dst + NOPE + 2 * pairs, _rope_re(re, im, cos, sin, KV_RE).to(tl.bfloat16))
        tl.store(dst + NOPE + 2 * pairs + 1, _rope_im(re, im, cos, sin, KV_IM).to(tl.bfloat16))
        tl.store(table_out_ptr + row * (2 * HALF) + pairs, cos)
        tl.store(table_out_ptr + row * (2 * HALF) + HALF + pairs, sin)
        if HC > 0:
            width: tl.constexpr = (2 + HC) * HC
            idx = tl.arange(0, COEFF_P2)
            values = tl.load(coeff_ptr + token * width + idx, mask=idx < width)
            tl.store(pre_out_ptr + row * HC + idx, values, mask=idx < HC)
            tl.store(
                post_out_ptr + row * HC + (idx - HC), values, mask=(idx >= HC) & (idx < 2 * HC)
            )
            tl.store(
                comb_out_ptr + row * (HC * HC) + (idx - 2 * HC),
                values,
                mask=(idx >= 2 * HC) & (idx < width),
            )
        if j == 0:
            zeros = tl.zeros((NOPE_P2,), dtype=tl.bfloat16)
            for z in tl.static_range(BLOCK, STORAGE):
                zdst = block_out_ptr + (request * STORAGE + z) * dim
                tl.store(zdst + cols, zeros, mask=cols < dim)
                if NOPE_P2 < dim:
                    tl.store(zdst + NOPE_P2 + cols, zeros, mask=NOPE_P2 + cols < dim)


def stage0_gather(
    table: Stage0Table,
    bonus: torch.Tensor,
    block_rope_freqs: torch.Tensor,
    noise_id: int,
    block: int,
    modes: Optional[dict] = None,
    coefficients: Optional[bool] = None,
) -> tuple:
    """``(q, draft_block, rope_table, coeffs)`` of int64 ``bonus [G]`` in one launch, as the live
    chain holds them; FP32 ``block_rope_freqs [G * block, R / 2, 2]``; ``coeffs`` may be None."""
    from ..flashinfer_utils import get_env_enable_pdl

    modes = modes or table.modes
    if coefficients is None:
        coefficients = table.coeff_ready
    hc = table.hc_mult if coefficients and table.coeff is not None else 0
    g = bonus.shape[0]
    rows = g * block
    half = table.rope_dim // 2
    nope = table.head_dim - table.rope_dim
    device = bonus.device
    q = torch.empty((rows, table.heads * table.head_dim), dtype=torch.bfloat16, device=device)
    draft_block = torch.empty(
        (g, DRAFT_BLOCK_STORAGE, table.head_dim), dtype=torch.bfloat16, device=device
    )
    rope_table = torch.empty((rows, 2, half), dtype=torch.float32, device=device)
    coeffs = None
    if hc:
        coeffs = (
            torch.empty((rows, hc, 1), dtype=torch.float32, device=device),
            torch.empty((rows, hc, 1), dtype=torch.float32, device=device),
            torch.empty((rows, hc, hc), dtype=torch.float32, device=device),
        )
    pre_out, post_out, comb_out = coeffs or (rope_table,) * 3
    use_pdl = get_env_enable_pdl()
    _stage0_gather_kernel[(rows, table.heads + 1)](
        bonus,
        block_rope_freqs,
        table.q_nope,
        table.q_rope,
        table.kv_nope,
        table.kv_rope,
        table.coeff if hc else rope_table,
        q,
        draft_block,
        rope_table,
        pre_out,
        post_out,
        comb_out,
        int(noise_id),
        table.vocab,
        HEADS=table.heads,
        NOPE=nope,
        NOPE_P2=triton.next_power_of_2(nope),
        HALF=half,
        BLOCK=block,
        STORAGE=DRAFT_BLOCK_STORAGE,
        Q_RE=modes["q_re"],
        Q_IM=modes["q_im"],
        KV_RE=modes["kv_re"],
        KV_IM=modes["kv_im"],
        HC=hc,
        COEFF_P2=triton.next_power_of_2(max((2 + hc) * hc, 1)),
        WAIT=use_pdl,
        num_warps=2,
        launch_pdl=use_pdl,
    )
    return q, draft_block, rope_table, coeffs


def _mismatches(table, w, bonus, block_rope_freqs, noise_id, block, modes) -> dict:
    """Mismatching elements of the gather vs the live chain per column class."""
    g = bonus.shape[0]
    tokens = torch.full((g, block), int(noise_id), dtype=torch.int64, device=bonus.device)
    tokens[:, 0] = bonus
    rope_table = block_rope_freqs.transpose(1, 2).contiguous()
    live_q, live_blk = live_stage0(w, tokens.reshape(-1), block_rope_freqs, rope_table, block=block)
    q, blk, table_rows, _ = stage0_gather(
        table, bonus, block_rope_freqs, noise_id, block, modes, coefficients=False
    )
    nope = table.head_dim - table.rope_dim
    lq = live_q.view(g * block, table.heads, table.head_dim).view(torch.int16)
    gq = q.view(g * block, table.heads, table.head_dim).view(torch.int16)
    lb, gb = live_blk.view(torch.int16), blk.view(torch.int16)
    return {
        "q_nope": int((lq[..., :nope] != gq[..., :nope]).sum()),
        "q_re": int((lq[..., nope::2] != gq[..., nope::2]).sum()),
        "q_im": int((lq[..., nope + 1 :: 2] != gq[..., nope + 1 :: 2]).sum()),
        "kv_nope": int((lb[:, :block, :nope] != gb[:, :block, :nope]).sum()),
        "kv_re": int((lb[:, :block, nope::2] != gb[:, :block, nope::2]).sum()),
        "kv_im": int((lb[:, :block, nope + 1 :: 2] != gb[:, :block, nope + 1 :: 2]).sum()),
        "pad": int((lb[:, block:] != gb[:, block:]).sum()),
        "table": int((table_rows != rope_table).sum()),
    }


def _coeff_mismatches(table, w, bonus, block_rope_freqs, noise_id, block) -> int:
    """Mismatching entry-coefficient elements of the gather vs the live kernels."""
    g = bonus.shape[0]
    tokens = torch.full((g, block), int(noise_id), dtype=torch.int64, device=bonus.device)
    tokens[:, 0] = bonus
    live = w.coefficients(_residual(w, tokens.reshape(-1)))
    gathered = stage0_gather(
        table, bonus, block_rope_freqs, noise_id, block, dict.fromkeys(_RMODE_KEYS, 1), True
    )[3]
    return sum(
        int((a.reshape(-1).view(torch.int32) != b.reshape(-1).view(torch.int32)).sum())
        for a, b in zip(live, gathered)
    )


_RMODE_KEYS = ("q_re", "q_im", "kv_re", "kv_im")


def check_stage0_table(
    table: Stage0Table,
    w: Stage0Weights,
    freqs_cis: torch.Tensor,
    noise_id: int,
    block: int,
) -> bool:
    """Compare the gather with the live chain over sample bonus tokens, picking each RoPE class's
    contraction; ``modes`` / ``coeff_ready`` set on exact matches only."""
    device = w.embed.device
    gen = torch.Generator(device="cpu").manual_seed(0)
    fixed = [int(noise_id), table.vocab, -1, 0, table.vocab - 1]
    rand = torch.randint(0, table.vocab, (_CHECK_TOKENS - len(fixed),), generator=gen).tolist()
    tokens = fixed + rand
    max_pos = freqs_cis.shape[0] - block - 2
    freqs = torch.view_as_real(freqs_cis)
    batches = []
    for token in tokens:
        start = torch.randint(0, max_pos, (1,), generator=gen)
        pos = (start + 1 + torch.arange(block)).to(device)
        batches.append(
            (torch.tensor([token], dtype=torch.int64, device=device), freqs[pos].contiguous())
        )
    keys = _RMODE_KEYS
    chosen = {}
    with torch.inference_mode():
        if table.coeff is not None and w.coefficients is not None:
            wrong = sum(
                _coeff_mismatches(table, w, bonus, block_freqs, noise_id, block)
                for bonus, block_freqs in batches
            )
            table.coeff_ready = wrong == 0
            if wrong:
                logger.warning(f"[DSpark] stage-0 table: {wrong} entry coefficients differ")
        for mode in _ROPE_MODES:
            modes = dict.fromkeys(keys, mode)
            totals = dict.fromkeys(keys, 0)
            other = 0
            for bonus, block_freqs in batches:
                diffs = _mismatches(table, w, bonus, block_freqs, noise_id, block, modes)
                for key in keys:
                    totals[key] += diffs.pop(key)
                other += sum(diffs.values())
            if other:
                logger.warning(f"[DSpark] stage-0 table: {other} non-RoPE elements differ")
                return False
            for key in keys:
                if key not in chosen and totals[key] == 0:
                    chosen[key] = mode
            if len(chosen) == len(keys):
                table.modes = chosen
                return True
    logger.warning(
        f"[DSpark] stage-0 table: no RoPE contraction matches for {set(keys) - set(chosen)}"
    )
    return False
