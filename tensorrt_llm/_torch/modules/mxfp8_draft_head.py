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
"""Greedy MTP draft head on an MXFP8 copy of the LM-head shard (``MTPDecodingConfig.draft_lm_head_dtype``).

``MXFP8DraftHead.local_keys`` streams the MXFP8 shard (E4M3, one UE8M0 scale per 32 K-elements) and folds
the per-row argmax into one int64 key ``(orderable_int32(logit) << 32) | (0xFFFFFFFF - vocab_id)``, so the
max over TP ranks is the global argmax with ties to the smallest id. ``exchange_argmax_keys`` (LL exchange)
or NCCL all-gather + ``decode_keys`` turn the keys into token ids and re-arm the key buffer to INT64_MIN.

Only the draft logits change (the target head stays BF16), which can change the acceptance length.
The GEMV keeps the activation in registers, so it takes at most ``MAX_ROWS`` (16) rows.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from ..flashinfer_utils import get_env_enable_pdl

# Rows of the head quantized per step of the one-time conversion (bounds its FP32 scratch).
_QUANT_ROWS = 4096
MAX_ROWS = 16
INT64_MIN = -(1 << 63)


def is_supported() -> bool:
    """Whether the GEMV runs on this device: SM 100 / SM 103 with the CuTe DSL."""
    try:
        from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
    except ImportError:
        return False
    return IS_CUTLASS_DSL_AVAILABLE and torch.cuda.is_available() and get_sm_version() in (100, 103)


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def default_slice_blocks(k: int) -> int:
    """Scale blocks of K per warp of the GEMV for a head of hidden size ``k`` (about 16-20 warps per CTA)."""
    for slice_blocks in (8, 16, 4, 2):
        if k % (32 * slice_blocks) == 0 and 4 <= k // (32 * slice_blocks) <= 32:
            return slice_blocks
    raise ValueError(f"unsupported K={k} for the MXFP8 draft head GEMV")


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


def _cute(t: torch.Tensor, align: int = 16):
    from ..cute_dsl_utils import graph_safe_cute_tensor

    return graph_safe_cute_tensor(t, assumed_align=align)


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

    from ..kernels.blackwell.mxfp8_argmax_gemv import MXFP8ArgmaxGemvKernel

    kernel = MXFP8ArgmaxGemvKernel(
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
    use_pdl: bool = True,
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
        use_pdl,
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
    keys: torch.Tensor, exchange, out: Optional[torch.Tensor] = None
) -> Optional[torch.Tensor]:
    """Global greedy token ids (int32 [M]) from every TP rank's keys [M] int64 via the LL exchange
    ``exchange`` (``ll_exchange.TPKeyExchange``), which also re-arms ``keys``; None without it or above its
    rows (caller falls back to NCCL)."""
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
    def _decode_keys_kernel(
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
        _decode_keys_kernel[(1,)](
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


# ----------------------------------------------------------------------------------------------
# Draft head
# ----------------------------------------------------------------------------------------------
class MXFP8DraftHead:
    """Draft-only MXFP8 copy of a column-parallel (vocabulary-sharded) BF16 LM head plus the fused
    argmax path. Built once after weight loading (``build_mxfp8_draft_head``)."""

    def __init__(self, lm_head, mapping, key_exchange=None) -> None:
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
        self.key_exchange = key_exchange
        self.keys = torch.full((MAX_ROWS,), INT64_MIN, dtype=torch.int64, device=weight.device)
        self.slice_blocks = default_slice_blocks(self.k)
        self.use_pdl = get_env_enable_pdl()
        logger.info(
            f"MXFP8 draft head: rows {valid} of {local_vocab} (offset {self.vocab_offset}), K {self.k}, "
            f"{self.values.numel() / 2**20:.1f} MiB values + {self.scales.numel() / 2**20:.1f} MiB scales"
        )

    def supports(self, rows: int) -> bool:
        return self.k % 512 == 0 and 0 < rows <= MAX_ROWS

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
            use_pdl=self.use_pdl,
        )
        return keys

    def sample(self, x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Greedy draft token ids ``[M]`` int32 (global vocabulary), identical on every TP rank; written into
        ``out`` (contiguous int32 ``[M]``) when given."""
        keys = self.local_keys(x)
        tp_size = getattr(self.mapping, "tp_size", 1) if self.mapping is not None else 1
        if tp_size > 1:
            tokens = exchange_argmax_keys(keys, self.key_exchange, out)
            if tokens is not None:
                return tokens
            from ..distributed import allgather

            return decode_keys(allgather(keys, self.mapping, dim=0), tp_size, keys, out)
        return decode_keys(keys, 1, keys, out)


def build_mxfp8_draft_head(lm_head, mapping, key_exchange=None) -> Optional[MXFP8DraftHead]:
    """``MXFP8DraftHead`` for ``lm_head`` when the device supports it and the head is a BF16, unquantized,
    column-parallel (or unsharded) head without attention DP; otherwise None (with a warning).
    ``key_exchange``: the TP key exchange of the greedy tokens, or None (NCCL all-gather)."""
    from .linear import TensorParallelMode

    reason = None
    if not is_supported():
        reason = "needs SM 100 / SM 103 and the CuTe DSL"
    elif getattr(lm_head, "has_any_quant", False) or lm_head.weight.dtype != torch.bfloat16:
        reason = "the LM head is not BF16"
    elif mapping is not None and mapping.enable_attention_dp:
        reason = "attention DP"
    elif lm_head.tp_size > 1 and lm_head.tp_mode != TensorParallelMode.COLUMN:
        reason = "the LM head is not vocabulary-sharded"
    if reason is not None:
        logger.warning(f"draft_lm_head_dtype=mxfp8 ignored: {reason}")
        return None
    return MXFP8DraftHead(lm_head, mapping, key_exchange)


__all__ = [
    "MXFP8DraftHead",
    "build_mxfp8_draft_head",
    "decode_keys",
    "dequantize_mxfp8_rows",
    "exchange_argmax_keys",
    "is_supported",
    "mxfp8_head_gemv",
    "quantize_mxfp8_rows",
]
