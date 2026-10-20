# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 MTP draft head: MXFP8 head copy, fused shard argmax, cross-rank key exchange.

``TRTLLM_Q35_DRAFT_FP8_HEAD=1`` (default off) replaces, in every greedy draft step of the one-model MTP
loop (``Eagle3OneModelWorker._forward_linear_draft_loop``), the BF16 vocabulary-parallel LM head
(cuBLAS split-K GEMM + reduce), the local ``torch.max``, the (index, value) packing and the compiled
pick with:

1. ``Q35DraftHead.local_keys``: one CuTe DSL launch (``cute_dsl_kernels/blackwell/q35_draft_head_gemv``)
   that streams a draft-only MXFP8 copy of this rank's LM-head shard (E4M3 values, one UE8M0 scale per
   32 K-elements; built once at load time) at decode M, and folds the shard's per-row argmax into one
   packed int64 key per row:
   ``key = (orderable_int32(logit_fp32) << 32) | (0xFFFFFFFF - vocab_id)``.
2. ``exchange_argmax_keys``: the one-launch LL exchange (``distributed/ll_exchange.py``,
   ``TRTLLM_Q35_LL_ARGMAX_EXCHANGE=1``): max of the keys over the TP ranks (the global argmax; equal logits
   resolve to the smallest vocabulary id, as before) -> int32 token ids, and the local key buffer re-armed
   to ``INT64_MIN`` for the next launch. Without it: NCCL all-gather + ``decode_keys`` (one Triton launch:
   max over the gathered keys -> token ids, plus the re-arm).

The target LM head is untouched (BF16). Only the draft logits change (FP8 weights, exact products, FP32
accumulation), which can change the draft proposals - gate with the unforced acceptance-length check.
"""

from __future__ import annotations

import functools
import os
from typing import Optional, Tuple

import torch

from tensorrt_llm.logger import logger

_ENV_ENABLE = "TRTLLM_Q35_DRAFT_FP8_HEAD"
# TRTLLM_Q35_DRAFT_HEAD_NVFP4=1 (with TRTLLM_Q35_DRAFT_FP8_HEAD=1; default off): the draft head copy in NVFP4
# (E2M1 values, one E4M3 scale per 16 K-elements, one FP32 scale per shard) instead of MXFP8: half the bytes.
_ENV_NVFP4 = "TRTLLM_Q35_DRAFT_HEAD_NVFP4"
_ENV_EARLY_TRIGGER = "TRTLLM_Q35_DRAFT_HEAD_EARLY_TRIGGER"
_ENV_SLICE_BLOCKS = "TRTLLM_Q35_DRAFT_HEAD_SLICE_BLOCKS"
# Rows of the head quantized per step of the one-time conversion (bounds its FP32 scratch).
_QUANT_ROWS = 4096
MAX_ROWS = 16
INT64_MIN = -(1 << 63)


def draft_fp8_head_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "0") == "1"


@functools.lru_cache(maxsize=None)
def _device_supported() -> bool:
    try:
        from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
    except ImportError:
        return False
    return (IS_CUTLASS_DSL_AVAILABLE and torch.cuda.is_available()
            and torch.cuda.get_device_capability() in ((10, 0), (10, 3)))


def _use_pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


def _early_trigger() -> bool:
    return os.environ.get(_ENV_EARLY_TRIGGER, "1") == "1"


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def default_slice_blocks(k: int) -> int:
    env = os.environ.get(_ENV_SLICE_BLOCKS)
    if env:
        return int(env)
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
        values[start:end] = (blocks / torch.exp2(exponent)[..., None]).view(
            end - start, k).to(torch.float8_e4m3fn)
        scales[start:end] = (exponent + 127).to(torch.uint8)
    return values, scales


_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_E2M1_TIE_UP = (0.75, 1.75, 3.5)  # midpoints whose upper neighbour has the even code
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def draft_head_nvfp4_enabled() -> bool:
    return os.environ.get(_ENV_NVFP4, "0") == "1"


def quantize_nvfp4_rows(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """``weight`` [N, K] as NVFP4: packed E2M1 ``[N, K/2]`` uint8 (element ``2j`` in the low nibble of byte
    ``j``), E4M3 block scales ``[N, K/16]`` (as uint8) and the FP32 per-tensor scale ``g`` with
    ``w ~ e2m1 * e4m3 * g`` (``g = amax / (6 * 448)``; values rounded to nearest, ties to the even code)."""
    rows, k = weight.shape
    assert k % 16 == 0, k
    amax = float(weight.abs().max().float()) if weight.numel() else 0.0
    g = amax / (6.0 * 448.0) if amax > 0 else 1.0
    packed = torch.empty((rows, k // 2), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((rows, k // 16), dtype=torch.uint8, device=weight.device)
    mids = torch.tensor(_E2M1_MIDPOINTS, device=weight.device)
    tie_up = torch.tensor(_E2M1_TIE_UP, device=weight.device)
    for start in range(0, rows, _QUANT_ROWS):
        end = min(start + _QUANT_ROWS, rows)
        blocks = weight[start:end].float().view(end - start, k // 16, 16)
        s_q = (blocks.abs().amax(-1) / 6.0 / g).clamp(max=448.0).to(torch.float8_e4m3fn)
        scales[start:end] = s_q.view(torch.uint8)
        denom = (s_q.float() * g)[..., None]
        x = torch.where(denom > 0, blocks / torch.where(denom > 0, denom, 1.0), 0.0)
        mag = x.abs().clamp(max=6.0)
        code = torch.bucketize(mag, mids, right=False)  # ties -> lower code
        code = code + torch.isin(mag, tie_up).to(code.dtype)  # ... except where the upper code is even
        code = code | ((x < 0) & (code > 0)).to(code.dtype) << 3
        code = code.to(torch.uint8).view(end - start, k)
        packed[start:end] = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed, scales, g


def dequantize_nvfp4_rows(packed: torch.Tensor, scales: torch.Tensor, global_scale: float) -> torch.Tensor:
    """FP32 ``[N, K]`` from ``quantize_nvfp4_rows`` output (reference / tests)."""
    rows, half = packed.shape
    k = 2 * half
    code = torch.stack([packed & 0x0F, packed >> 4], dim=-1).view(rows, k).long()
    grid = torch.tensor(_E2M1_VALUES, device=packed.device)
    val = grid[code & 7] * torch.where(code >= 8, -1.0, 1.0)
    s = scales.view(torch.float8_e4m3fn).float()
    return (val.view(rows, k // 16, 16) * s[..., None]).view(rows, k) * global_scale


def dequantize_mxfp8_rows(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """FP32 ``[N, K]`` from ``quantize_mxfp8_rows`` output (reference / tests)."""
    rows, k = values.shape
    scale = torch.exp2(scales.float() - 127.0)
    return (values.float().view(rows, k // 32, 32) * scale[..., None]).view(rows, k)


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
def _compiled(m: int, n: int, k: int, slice_blocks: int, num_ctas: int,
              use_pdl: bool, early_trigger: bool, argmax: bool,
              out_dtype: torch.dtype, debug_skip_wait: bool = False,
              weight_format: str = "mxfp8"):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ..cute_dsl_kernels.blackwell.q35_draft_head_gemv import \
        Q35DraftHeadGemvKernel

    kernel = Q35DraftHeadGemvKernel(num_rows=m,
                                    k_extent=k,
                                    num_ctas=num_ctas,
                                    slice_blocks=slice_blocks,
                                    use_pdl=use_pdl,
                                    early_trigger=early_trigger,
                                    argmax=argmax,
                                    debug_skip_wait=debug_skip_wait,
                                    weight_format=weight_format)
    dev = torch.device("cuda")
    nvfp4 = weight_format == "nvfp4"
    a = torch.empty((m, k), dtype=torch.bfloat16, device=dev)
    b = torch.empty((n, k // 2 if nvfp4 else k), dtype=torch.uint8, device=dev)
    s = torch.empty((n, k // 16 if nvfp4 else k // 32), dtype=torch.uint8, device=dev)
    if argmax:
        out = torch.empty((m, 2), dtype=torch.int32, device=dev)
    else:
        out = torch.empty((m, n), dtype=out_dtype, device=dev)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(kernel, _cute(a), _cute(b), _cute(s), _cute(out, 8),
                        cutlass.Int32(0), cutlass.Float32(1.0), stream)


def mxfp8_head_gemv(x: torch.Tensor,
                    values: torch.Tensor,
                    scales: torch.Tensor,
                    out: torch.Tensor,
                    *,
                    argmax: bool,
                    vocab_offset: int = 0,
                    slice_blocks: Optional[int] = None,
                    _debug_skip_wait: bool = False) -> torch.Tensor:
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
    fn = _compiled(m, n, k, slice_blocks or default_slice_blocks(k), num_ctas,
                   _use_pdl(), _early_trigger(), argmax,
                   torch.int32 if argmax else out.dtype, _debug_skip_wait)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(_cute(x), _cute(values.view(torch.uint8)), _cute(scales), _cute(out, 8),
       cutlass.Int32(vocab_offset), cutlass.Float32(1.0), stream)
    return out


def nvfp4_head_gemv(x: torch.Tensor,
                    packed: torch.Tensor,
                    scales: torch.Tensor,
                    global_scale: float,
                    out: torch.Tensor,
                    *,
                    argmax: bool,
                    vocab_offset: int = 0,
                    slice_blocks: Optional[int] = None,
                    _debug_skip_wait: bool = False) -> torch.Tensor:
    """``mxfp8_head_gemv`` for NVFP4 weights: ``packed`` [N, K/2] uint8, ``scales`` [N, K/16] E4M3 (uint8),
    ``global_scale`` the per-tensor FP32 scale (``quantize_nvfp4_rows``)."""
    import cuda.bindings.driver as cuda
    import cutlass

    assert x.dim() == 2 and x.is_contiguous() and x.dtype == torch.bfloat16
    m, k = x.shape
    n = packed.shape[0]
    assert packed.shape == (n, k // 2) and packed.is_contiguous() and packed.dtype == torch.uint8
    assert scales.shape == (n, k // 16) and scales.is_contiguous()
    if not 0 < m <= MAX_ROWS or k % 512:
        raise ValueError(f"unsupported shape M={m}, K={k} (M <= {MAX_ROWS}, K % 512 == 0)")
    if argmax:
        assert out.shape == (m, 2) and out.dtype == torch.int32 and out.is_contiguous()
    else:
        assert out.shape == (m, n) and out.is_contiguous()
    device = x.device.index if x.device.index is not None else torch.cuda.current_device()
    num_ctas = min(_num_sms(device), -(-n // 8))
    fn = _compiled(m, n, k, slice_blocks or default_slice_blocks(k), num_ctas,
                   _use_pdl(), _early_trigger(), argmax,
                   torch.int32 if argmax else out.dtype, _debug_skip_wait, "nvfp4")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(_cute(x), _cute(packed), _cute(scales.view(torch.uint8)), _cute(out, 8),
       cutlass.Int32(vocab_offset), cutlass.Float32(global_scale), stream)
    return out


# ----------------------------------------------------------------------------------------------
# Key exchange and decode
# ----------------------------------------------------------------------------------------------
def exchange_argmax_keys(keys: torch.Tensor, mapping,
                         out: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    """Global greedy token ids (int32 [M]) from every TP rank's packed keys ``keys`` [M] int64 with the
    one-launch LL exchange (``TRTLLM_Q35_LL_ARGMAX_EXCHANGE=1``; channel 0), which also
    re-arms ``keys`` (INT64_MIN); None when that exchange is unavailable (caller: NCCL fallback)."""
    from ..distributed import ll_exchange
    if not ll_exchange.ll_argmax_exchange_enabled():
        return None
    exchange = ll_exchange.get_tp_key_exchange(mapping)
    if exchange is None or keys.numel() > exchange.MAX_ROWS:
        return None
    tokens = out if out is not None else torch.empty((keys.numel(), ), dtype=torch.int32, device=keys.device)
    exchange.exchange(keys, tokens_out=tokens, reset_keys=True, channel=0)
    return tokens


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _q35_decode_keys_kernel(gathered_ptr, keys_ptr, tokens_ptr, M,
                                RANKS: tl.constexpr, BLOCK: tl.constexpr):
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

    _TRITON_OK = True
except ImportError:  # pragma: no cover
    _TRITON_OK = False


def decode_keys(gathered: torch.Tensor, ranks: int, keys: torch.Tensor,
                out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """int32 token ids ``[M]`` (into ``out`` if given) = low words of the max key over ``ranks``; resets ``keys``."""
    m = keys.shape[0]
    tokens = out if out is not None else torch.empty((m, ), dtype=torch.int32, device=keys.device)
    if _TRITON_OK:
        _q35_decode_keys_kernel[(1, )](gathered, keys, tokens, m, RANKS=ranks,
                                       BLOCK=max(16, triton.next_power_of_2(m)))
    else:
        best = gathered.view(ranks, m).max(dim=0).values
        tokens.copy_(torch.where(best == INT64_MIN, torch.zeros_like(best),
                                 (~best) & 0xFFFFFFFF).to(torch.int32))
        keys.fill_(INT64_MIN)
    return tokens


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
        self.nvfp4 = draft_head_nvfp4_enabled()
        if self.nvfp4:
            self.values, self.scales, self.global_scale = quantize_nvfp4_rows(
                weight.detach()[:valid].contiguous())
        else:
            self.values, self.scales = quantize_mxfp8_rows(weight.detach()[:valid].contiguous())
            self.global_scale = 1.0
        self.k = weight.shape[1]
        self.mapping = mapping
        self.keys = torch.full((MAX_ROWS, ), INT64_MIN, dtype=torch.int64, device=weight.device)
        self.slice_blocks = default_slice_blocks(self.k)
        logger.info(
            f"[Q35DraftHead] {'NVFP4' if self.nvfp4 else 'MXFP8'} draft head: rows {valid} of {local_vocab} (offset "
            f"{self.vocab_offset}), K {self.k}, {self.values.numel() / 2**20:.1f} MiB values + "
            f"{self.scales.numel() / 2**20:.1f} MiB scales")

    def supports(self, rows: int) -> bool:
        return 0 < rows <= MAX_ROWS and self.k % 512 == 0

    def local_keys(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's packed argmax keys ``[M]`` int64 (a view of the persistent key buffer)."""
        m = x.shape[0]
        keys = self.keys[:m]
        if getattr(self, "nvfp4", False):  # (getattr: subclasses / stand-ins that skip __init__)
            nvfp4_head_gemv(x.contiguous(), self.values, self.scales, self.global_scale,
                            keys.view(torch.int32).view(m, 2), argmax=True,
                            vocab_offset=self.vocab_offset, slice_blocks=self.slice_blocks)
        else:
            mxfp8_head_gemv(x.contiguous(), self.values, self.scales,
                            keys.view(torch.int32).view(m, 2), argmax=True,
                            vocab_offset=self.vocab_offset, slice_blocks=self.slice_blocks)
        return keys

    def sample(self, x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Greedy draft token ids ``[M]`` int32 (global vocabulary), identical on every TP rank; written into
        ``out`` (contiguous int32 ``[M]``) when given."""
        keys = self.local_keys(x)
        tp_size = getattr(self.mapping, "tp_size", 1) if self.mapping is not None else 1
        if tp_size > 1:
            # One launch: exchange + token ids + key re-arm (comm LL exchange).
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
    "Q35DraftHead", "build_draft_head_if_enabled", "decode_keys", "draft_fp8_head_enabled",
    "draft_head_nvfp4_enabled", "dequantize_mxfp8_rows", "dequantize_nvfp4_rows", "exchange_argmax_keys",
    "mxfp8_head_gemv", "nvfp4_head_gemv", "quantize_mxfp8_rows", "quantize_nvfp4_rows"
]
