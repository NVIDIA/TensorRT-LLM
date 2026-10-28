# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3.5 MTP greedy draft head (``TRTLLM_Q35_DRAFT_FP8_HEAD=1``): MXFP8 copy of the LM-head shard.

``Q35DraftHead.local_keys`` streams the MXFP8 shard (E4M3, one UE8M0 scale per 32 K-elements) and folds
the per-row argmax into one int64 key ``(orderable_int32(logit) << 32) | (0xFFFFFFFF - vocab_id)``, so the
max over TP ranks is the global argmax with ties to the smallest id. ``exchange_argmax_keys`` (LL exchange)
or NCCL all-gather + ``decode_keys`` turn the keys into token ids and re-arm the key buffer to INT64_MIN.

Only the draft logits change (the target head stays BF16), which can change the acceptance length.

The GEMV keeps the activation in registers, so it takes at most ``MAX_ROWS`` (16) rows.
``TRTLLM_Q35_DRAFT_HEAD_MX_ROWS=<n>`` (default 0 = off; MXFP8 copy only) runs draft passes of 17..n rows
(n <= 128) as a cuBLAS block-scaled MXFP8 GEMM (``torch._scaled_mm``) on the same head copy, with the
activation quantized to MXFP8 exactly as ``quantize_mxfp8_rows`` quantizes the weights (one Triton
launch, scales written in cuBLAS's tiled layout); the FP32 logits go through the one-launch argmax + key
exchange (``TPKeyExchange.argmax_logits``). Unlike the GEMV, the activation is quantized, which changes
the draft logits of those passes - gate with the unforced acceptance-length check.
"""

from __future__ import annotations

import functools
import os
from typing import Optional, Tuple

import torch

from tensorrt_llm.logger import logger

_ENV_ENABLE = "TRTLLM_Q35_DRAFT_FP8_HEAD"
# TRTLLM_Q35_DRAFT_HEAD_MX_ROWS=<n>: passes of MAX_ROWS + 1 .. n rows on the cuBLAS MXFP8 GEMM (module docstring).
_ENV_MX_ROWS = "TRTLLM_Q35_DRAFT_HEAD_MX_ROWS"
# Rows of the head quantized per step of the one-time conversion (bounds its FP32 scratch).
_QUANT_ROWS = 4096
MAX_ROWS = 16
MX_MAX_ROWS = 128  # one 128-row tile of activation block scales
INT64_MIN = -(1 << 63)


def draft_fp8_head_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "0") == "1"


@functools.lru_cache(maxsize=None)
def _device_supported() -> bool:
    try:
        from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
    except ImportError:
        return False
    return (
        IS_CUTLASS_DSL_AVAILABLE
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() in ((10, 0), (10, 3))
    )


def _use_pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def default_slice_blocks(k: int) -> int:
    """Scale blocks of K per warp of the GEMV for a head of hidden size ``k``."""
    # About 16-20 warps per CTA (GB300; K = 4096: 16 warps of 8 blocks).
    for slice_blocks in (8, 16, 4, 2):
        if k % (32 * slice_blocks) == 0 and 4 <= k // (32 * slice_blocks) <= 32:
            return slice_blocks
    raise ValueError(f"unsupported K={k} for the Qwen3.5 draft head GEMV")


def quantize_mxfp8_rows(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``weight`` [N, K] as E4M3 values with one UE8M0 scale per 32 K-elements: each block is
    divided by the power of two that brings its absolute max to at most 448."""
    rows, k = weight.shape
    assert k % 32 == 0, k
    values = torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=weight.device)
    scales = torch.empty((rows, k // 32), dtype=torch.uint8, device=weight.device)
    for start in range(0, rows, _QUANT_ROWS):
        end = min(start + _QUANT_ROWS, rows)
        blocks = weight[start:end].float().view(end - start, k // 32, 32)
        amax = blocks.abs().amax(-1).clamp_min(2.0**-126)
        exponent = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
        values[start:end] = (
            (blocks / torch.exp2(exponent)[..., None]).view(end - start, k).to(torch.float8_e4m3fn)
        )
        scales[start:end] = (exponent + 127).to(torch.uint8)
    return values, scales


def dequantize_mxfp8_rows(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """FP32 ``[N, K]`` from ``quantize_mxfp8_rows`` output (reference / tests)."""
    rows, k = values.shape
    scale = torch.exp2(scales.float() - 127.0)
    return (values.float().view(rows, k // 32, 32) * scale[..., None]).view(rows, k)


def blocked_scales(scales: torch.Tensor) -> torch.Tensor:
    """``scales`` [rows, cols] (UE8M0 bytes) in the 128 x 4 tiled layout cuBLAS block scaling reads, flattened
    (rows padded to a multiple of 128, cols to a multiple of 4, padding = 2^0): entry (r, c) lands at
    ``((r // 128) * ceil(cols / 4) + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4``."""
    rows, cols = scales.shape
    nr, nc = -(-rows // 128), -(-cols // 4)
    padded = torch.full((nr * 128, nc * 4), 127, dtype=torch.uint8, device=scales.device)
    padded[:rows, :cols] = scales
    tiles = padded.view(nr, 128, nc, 4).permute(0, 2, 1, 3)
    return tiles.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).contiguous()


class _GraphSafeDlpack:
    """DLPack export that does not synchronize streams (CUDA-graph capture safe)."""

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor.detach()

    def __dlpack__(self, stream=None):
        return self._tensor.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self._tensor.__dlpack_device__()


def _cute(t: torch.Tensor, align: int = 16):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(_GraphSafeDlpack(t), assumed_align=align)


@functools.lru_cache(maxsize=None)
def _compiled(
    m: int,
    n: int,
    k: int,
    slice_blocks: int,
    num_ctas: int,
    use_pdl: bool,
    argmax: bool,
    out_dtype: torch.dtype,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ..kernels.blackwell.q35_draft_head_gemv import Q35DraftHeadGemvKernel

    kernel = Q35DraftHeadGemvKernel(
        num_rows=m,
        k_extent=k,
        num_ctas=num_ctas,
        slice_blocks=slice_blocks,
        use_pdl=use_pdl,
        argmax=argmax,
    )
    dev = torch.device("cuda")
    a = torch.empty((m, k), dtype=torch.bfloat16, device=dev)
    b = torch.empty((n, k), dtype=torch.uint8, device=dev)
    s = torch.empty((n, k // 32), dtype=torch.uint8, device=dev)
    if argmax:
        out = torch.empty((m, 2), dtype=torch.int32, device=dev)
    else:
        out = torch.empty((m, n), dtype=out_dtype, device=dev)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(
        kernel,
        _cute(a),
        _cute(b),
        _cute(s),
        _cute(out, 8),
        cutlass.Int32(0),
        stream,
    )


def mxfp8_head_gemv(
    x: torch.Tensor,
    values: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor,
    *,
    argmax: bool,
    vocab_offset: int = 0,
    slice_blocks: Optional[int] = None,
) -> torch.Tensor:
    """``argmax=False``: ``out[M, N] = x @ dequant(values, scales).T`` (FP32 or BF16 ``out``).
    ``argmax=True``: ``out`` is ``[M, 2]`` int32 (the words of one int64 key per row, which must hold
    ``INT64_MIN`` before the call); the per-row max key of this shard is folded into it.

    x: [M, K] bf16 contiguous, M <= 16; values: [N, K] e4m3 (or uint8) contiguous; scales: [N, K/32]
    uint8 contiguous.
    """
    import cuda.bindings.driver as cuda
    import cutlass

    assert x.dim() == 2 and x.is_contiguous() and x.dtype == torch.bfloat16
    m, k = x.shape
    n = values.shape[0]
    assert values.shape == (n, k) and values.is_contiguous()
    assert scales.shape == (n, k // 32) and scales.is_contiguous()
    assert scales.dtype == torch.uint8
    if not 0 < m <= MAX_ROWS or k % 512:
        raise ValueError(f"unsupported shape M={m}, K={k} (M <= {MAX_ROWS}, K % 512 == 0)")
    if argmax:
        assert out.shape == (m, 2) and out.dtype == torch.int32 and out.is_contiguous()
    else:
        assert out.shape == (m, n) and out.is_contiguous()
    device = x.device.index if x.device.index is not None else torch.cuda.current_device()
    num_ctas = min(_num_sms(device), -(-n // 8))
    fn = _compiled(
        m,
        n,
        k,
        slice_blocks or default_slice_blocks(k),
        num_ctas,
        _use_pdl(),
        argmax,
        torch.int32 if argmax else out.dtype,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(
        _cute(x),
        _cute(values.view(torch.uint8)),
        _cute(scales),
        _cute(out, 8),
        cutlass.Int32(vocab_offset),
        stream,
    )
    return out


# ----------------------------------------------------------------------------------------------
# Key exchange and decode
# ----------------------------------------------------------------------------------------------
def exchange_argmax_keys(
    keys: torch.Tensor, mapping, out: Optional[torch.Tensor] = None
) -> Optional[torch.Tensor]:
    """Global greedy token ids (int32 [M]) from every TP rank's keys [M] int64 via the LL exchange,
    which also re-arms ``keys``; None when that exchange is unavailable (caller falls back to NCCL)."""
    from ..distributed import ll_exchange

    if not ll_exchange.ll_argmax_exchange_enabled():
        return None
    exchange = ll_exchange.get_tp_key_exchange(mapping)
    if exchange is None or keys.numel() > exchange.MAX_ROWS:
        return None
    tokens = (
        out
        if out is not None
        else torch.empty((keys.numel(),), dtype=torch.int32, device=keys.device)
    )
    exchange.exchange(keys, tokens_out=tokens, reset_keys=True, channel=0)
    return tokens


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _q35_decode_keys_kernel(
        gathered_ptr, keys_ptr, tokens_ptr, M, RANKS: tl.constexpr, BLOCK: tl.constexpr
    ):
        offs = tl.arange(0, BLOCK)
        mask = offs < M
        lowest = tl.full([BLOCK], 1, tl.int64) << 63
        best = lowest
        for r in tl.static_range(RANKS):
            key = tl.load(gathered_ptr + r * M + offs, mask=mask, other=0)
            best = tl.maximum(best, key)
        token = (~best) & 0xFFFFFFFF
        token = tl.where(best == lowest, 0, token)
        tl.store(tokens_ptr + offs, token.to(tl.int32), mask=mask)
        # Re-arm this rank's key buffer for the next head launch.
        tl.store(keys_ptr + offs, lowest, mask=mask)

    @triton.jit
    def _q35_mx_quant_kernel(
        x_ptr, stride_x, q_ptr, s_ptr, K: tl.constexpr, NC: tl.constexpr, BLOCK: tl.constexpr
    ):
        """Row ``program_id(0)``, elements ``program_id(1) * BLOCK ..`` of the activation as MXFP8, bit for bit
        ``quantize_mxfp8_rows``: per 32 elements the exponent e = ceil(log2(amax / 448)), taken from the bits of
        amax (its exponent - 8, + 1 when its significand exceeds 1.75), values x * 2^-e rounded to E4M3, and the
        scale byte e + 127 stored at its place in the 128 x 4 tiled layout (``blocked_scales``; ``NC`` = tiles
        across the K / 32 scale columns)."""
        row = tl.program_id(0)
        part = tl.program_id(1)
        offs = part * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + row.to(tl.int64) * stride_x + offs).to(tl.float32)
        xb = tl.reshape(x, (BLOCK // 32, 32))
        amax = tl.maximum(tl.max(tl.abs(xb), axis=1), 1.1754943508222875e-38)  # clamp_min(2^-126)
        bits = amax.to(tl.int32, bitcast=True)
        e = ((bits >> 23) & 0xFF) - 135 + ((bits & 0x7FFFFF) > 0x600000).to(tl.int32)
        # BF16 activations keep e <= 120; 126 keeps 2^-e a normal FP32 power of two.
        e = tl.minimum(tl.maximum(e, -127), 126)
        inv = ((127 - e) << 23).to(tl.float32, bitcast=True)
        q = (xb * inv[:, None]).to(tl.float8e4nv)
        tl.store(q_ptr + row.to(tl.int64) * K + offs, tl.reshape(q, (BLOCK,)))
        col = part * (BLOCK // 32) + tl.arange(0, BLOCK // 32)
        soff = (
            ((row // 128) * NC + col // 4) * 512
            + (row % 32) * 16
            + ((row % 128) // 32) * 4
            + col % 4
        )
        tl.store(s_ptr + soff, (e + 127).to(tl.uint8))

    _TRITON_OK = True
except ImportError:  # pragma: no cover
    _TRITON_OK = False


def decode_keys(
    gathered: torch.Tensor, ranks: int, keys: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """int32 token ids ``[M]`` (into ``out`` if given) = low words of the max key over ``ranks``; resets ``keys``."""
    m = keys.shape[0]
    tokens = out if out is not None else torch.empty((m,), dtype=torch.int32, device=keys.device)
    if _TRITON_OK:
        _q35_decode_keys_kernel[(1,)](
            gathered, keys, tokens, m, RANKS=ranks, BLOCK=max(16, triton.next_power_of_2(m))
        )
    else:
        best = gathered.view(ranks, m).max(dim=0).values
        tokens.copy_(
            torch.where(best == INT64_MIN, torch.zeros_like(best), (~best) & 0xFFFFFFFF).to(
                torch.int32
            )
        )
        keys.fill_(INT64_MIN)
    return tokens


def mxfp8_quantize_rows_blocked(
    x: torch.Tensor, values: torch.Tensor, blocked: torch.Tensor
) -> None:
    """``x`` [M, K] (BF16, unit column stride, M <= MX_MAX_ROWS) as MXFP8, bit for bit ``quantize_mxfp8_rows``:
    E4M3 values into rows 0..M-1 of ``values`` ([>= M, K], contiguous) and the UE8M0 block scales into ``blocked``
    (uint8, one 128-row band of the ``blocked_scales`` layout: >= 128 * ceil(K / 128) * 4 bytes)."""
    m, k = x.shape
    assert x.stride(1) == 1 and 0 < m <= MX_MAX_ROWS and k % 512 == 0, (x.shape, x.stride())
    assert values.is_contiguous() and values.shape[1] == k and values.shape[0] >= m
    nc = -(-(k // 32) // 4)
    assert blocked.dtype == torch.uint8 and blocked.numel() >= 128 * nc * 4
    _q35_mx_quant_kernel[(m, k // 512)](
        x, x.stride(0), values, blocked, K=k, NC=nc, BLOCK=512, num_warps=4
    )


# ----------------------------------------------------------------------------------------------
# Draft head
# ----------------------------------------------------------------------------------------------
class Q35DraftHead:
    """Draft-only MXFP8 copy of a column-parallel (vocabulary-sharded) BF16 LM head plus the fused
    argmax path. Built once after weight loading (``build_draft_head_if_enabled``)."""

    def __init__(self, lm_head, mapping) -> None:
        weight = lm_head.weight
        assert weight.dim() == 2 and weight.dtype == torch.bfloat16, (weight.shape, weight.dtype)
        local_vocab = lm_head.out_features
        tp_rank = getattr(lm_head, "tp_rank", 0) if lm_head.tp_size > 1 else 0
        self.vocab_offset = tp_rank * local_vocab
        # Rows past the real vocabulary (shard padding of the last rank) never win.
        valid = max(0, min(local_vocab, lm_head.num_embeddings - self.vocab_offset))
        self.values, self.scales = quantize_mxfp8_rows(weight.detach()[:valid].contiguous())
        self.k = weight.shape[1]
        self.mapping = mapping
        self.keys = torch.full((MAX_ROWS,), INT64_MIN, dtype=torch.int64, device=weight.device)
        self.slice_blocks = default_slice_blocks(self.k)
        logger.info(
            f"[Q35DraftHead] MXFP8 draft head: rows {valid} of {local_vocab} (offset "
            f"{self.vocab_offset}), K {self.k}, {self.values.numel() / 2**20:.1f} MiB values + "
            f"{self.scales.numel() / 2**20:.1f} MiB scales"
        )
        # TRTLLM_Q35_DRAFT_HEAD_MX_ROWS: MAX_ROWS + 1 .. mx_rows rows on the cuBLAS MXFP8 GEMM.
        self.mx_rows = 0
        mx_rows = int(os.environ.get(_ENV_MX_ROWS, "0") or "0")
        if mx_rows > MAX_ROWS:
            if not _TRITON_OK or not hasattr(torch, "float8_e8m0fnu") or self.k % 512:
                logger.warning(
                    f"[Q35DraftHead] {_ENV_MX_ROWS}={mx_rows} ignored: needs Triton, "
                    f"torch.float8_e8m0fnu and K % 512 == 0"
                )
            else:
                self.mx_rows = min(mx_rows, MX_MAX_ROWS)
                self.mx_w_scales = blocked_scales(self.scales)
                self.mx_x = torch.empty(
                    (self.mx_rows, self.k), dtype=torch.float8_e4m3fn, device=weight.device
                )
                self.mx_x_scales = torch.full(
                    (128 * -(-(self.k // 32) // 4) * 4,),
                    127,
                    dtype=torch.uint8,
                    device=weight.device,
                )
                logger.info(
                    f"[Q35DraftHead] {_ENV_MX_ROWS}={self.mx_rows}: draft passes of {MAX_ROWS + 1}.."
                    f"{self.mx_rows} rows on the cuBLAS MXFP8 GEMM (activation quantized to MXFP8)"
                )

    def supports(self, rows: int) -> bool:
        return self.k % 512 == 0 and (
            0 < rows <= MAX_ROWS or MAX_ROWS < rows <= getattr(self, "mx_rows", 0)
        )

    def local_keys(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's packed argmax keys ``[M]`` int64 (a view of the persistent key buffer)."""
        m = x.shape[0]
        keys = self.keys[:m]
        mxfp8_head_gemv(
            x.contiguous(),
            self.values,
            self.scales,
            keys.view(torch.int32).view(m, 2),
            argmax=True,
            vocab_offset=self.vocab_offset,
            slice_blocks=self.slice_blocks,
        )
        return keys

    def mx_logits(self, x: torch.Tensor) -> torch.Tensor:
        """FP32 logits ``[M, rows of this shard]`` of ``MAX_ROWS < M <= mx_rows`` rows: the activation quantized to
        MXFP8 (one launch) and the cuBLAS block-scaled MXFP8 GEMM on the head copy."""
        m = x.shape[0]
        q = self.mx_x[:m]
        mxfp8_quantize_rows_blocked(x if x.stride(1) == 1 else x.contiguous(), q, self.mx_x_scales)
        return torch._scaled_mm(
            q,
            self.values.t(),
            scale_a=self.mx_x_scales.view(torch.float8_e8m0fnu),
            scale_b=self.mx_w_scales.view(torch.float8_e8m0fnu),
            out_dtype=torch.float32,
        )

    def _sample_mx(self, x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``sample`` for ``MAX_ROWS < M <= mx_rows``: ``mx_logits``, then the argmax + key exchange in one launch
        (``TPKeyExchange.argmax_logits``; without it: row maxima, keys, NCCL all-gather + ``decode_keys``)."""
        logits = self.mx_logits(x)
        tp_size = getattr(self.mapping, "tp_size", 1) if self.mapping is not None else 1
        if tp_size > 1:
            from ..distributed import ll_exchange

            if ll_exchange.ll_argmax_exchange_enabled():
                exchange = ll_exchange.get_tp_key_exchange(self.mapping)
                if exchange is not None and logits.shape[0] <= exchange.MAX_ROWS:
                    return exchange.argmax_logits(
                        logits, self.vocab_offset, tokens_out=out, channel=0
                    )
        from ..distributed.ll_exchange import pack_argmax_keys

        vmax, imax = torch.max(logits, dim=-1)
        keys = pack_argmax_keys(vmax, imax, self.vocab_offset)
        if tp_size > 1:
            from ..distributed import allgather

            return decode_keys(allgather(keys, self.mapping, dim=0), tp_size, keys, out)
        return decode_keys(keys, 1, keys, out)

    def sample(self, x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Greedy draft token ids ``[M]`` int32 (global vocabulary), identical on every TP rank; written into
        ``out`` (contiguous int32 ``[M]``) when given."""
        if x.shape[0] > MAX_ROWS:
            return self._sample_mx(x, out)
        keys = self.local_keys(x)
        tp_size = getattr(self.mapping, "tp_size", 1) if self.mapping is not None else 1
        if tp_size > 1:
            tokens = exchange_argmax_keys(keys, self.mapping, out)
            if tokens is not None:
                return tokens
            from ..distributed import allgather

            return decode_keys(allgather(keys, self.mapping, dim=0), tp_size, keys, out)
        return decode_keys(keys, 1, keys, out)


def build_draft_head_if_enabled(lm_head, mapping) -> Optional[Q35DraftHead]:
    """``Q35DraftHead`` for ``lm_head`` when ``TRTLLM_Q35_DRAFT_FP8_HEAD=1`` and the head is a BF16,
    unquantized, column-parallel (or unsharded) head without attention DP; otherwise None."""
    if not draft_fp8_head_enabled():
        return None
    from .linear import TensorParallelMode

    reason = None
    if not _device_supported():
        reason = "needs SM100/SM103 and the CuTe DSL"
    elif getattr(lm_head, "has_any_quant", False) or lm_head.weight.dtype != torch.bfloat16:
        reason = "the LM head is not BF16"
    elif mapping is not None and mapping.enable_attention_dp:
        reason = "attention DP"
    elif lm_head.tp_size > 1 and lm_head.tp_mode != TensorParallelMode.COLUMN:
        reason = "the LM head is not vocabulary-sharded"
    if reason is not None:
        logger.warning(f"[Q35DraftHead] {_ENV_ENABLE}=1 ignored: {reason}")
        return None
    return Q35DraftHead(lm_head, mapping)


__all__ = [
    "Q35DraftHead",
    "blocked_scales",
    "build_draft_head_if_enabled",
    "decode_keys",
    "draft_fp8_head_enabled",
    "dequantize_mxfp8_rows",
    "exchange_argmax_keys",
    "mxfp8_head_gemv",
    "mxfp8_quantize_rows_blocked",
    "quantize_mxfp8_rows",
]
