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
# =============================================================================
# DSpark vanilla-Markov head: the biased block logits' glue around the bias GEMM, in two launches
# =============================================================================
#
# It replaces, bit for bit, the elementwise lines of the batched biased-logits form (the tp targets' S4-logits path)
#   prevs  = cat([first_prev.view(B, 1), tokens[:, :K - 1].long()], 1).reshape(-1)
#   emb    = markov_prev_embeddings(prevs, w1)     (get_masked_input_and_mask, F.embedding, masked_fill: an anchor
#                                                   outside [0, V) gives a zero row)
#   bias   = F.linear(emb, w2_shard)                (unchanged: the caller's GEMM)
#   out    = shard_logits.float() + bias.view(B, K, N).to(float32)
# with ``prev_embeddings`` (one launch for the 11 index / mask / gather / fill launches) and ``widen_add`` (one launch
# for the two widening copies and the add). The gather copies w1's rows bit for bit (zero rows are +0.0, as
# masked_fill(0)); the add is one fp32 IEEE add of two exactly widened bf16 values, as torch's.
#
# DSPARK_MARKOV_BIAS_CPP (no Triton in served paths): both launches are CUDA C++ torch ops
# (family_ops().minimax_m31_dspark_prev_embeddings / minimax_m31_dspark_widen_add,
# the family's csrc/m31/dsparkMarkovBias.cu), bit for bit the Triton kernels they replace. ``supports`` is False when
# the ops are not registered or TRTLLM_M31_DISABLE_DSPARK_MARKOV_CPP=1 (the kill switch); the caller then runs the
# in-tree lines.


import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.m31_switches import (
    m31_disabled,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops

_CPP_OFF = m31_disabled("DSPARK_MARKOV_CPP")
_PREV_OP = "minimax_m31_dspark_prev_embeddings"
_WIDEN_OP = "minimax_m31_dspark_widen_add"
_logged: set = set()


def _cpp_op(name: str):
    """The C++ op, or None (killed or not registered), with the engagement line logged once per (op, outcome)."""
    op = None if _CPP_OFF else getattr(family_ops(), name, None)
    why = (
        None
        if op is not None
        else (
            "killed (TRTLLM_M31_DISABLE_DSPARK_MARKOV_CPP=1)"
            if _CPP_OFF
            else "the op is not registered"
        )
    )
    if (name, why) not in _logged:
        _logged.add((name, why))
        from tensorrt_llm.logger import logger

        if why is None:
            logger.info(f"[M3 dispatch] DSPARK_MARKOV_BIAS_CPP taken: {name}")
        else:
            logger.info(
                f"[M3 dispatch] DSPARK_MARKOV_BIAS_CPP not taken: {name}: {why}; in-tree lines"
            )
    return op


_INT = (torch.int32, torch.int64)
# The targets that armed strided logits rows (``strided_by``), each named once in the log.
_STRIDED_LOGGED: set = set()


def _logits_row_stride(shard_logits: torch.Tensor, strided_by):
    """widen_add's logits row stride: a contiguous tensor's (N) unless ``strided_by`` (the arming target) lets the
    rows be evenly spaced in a padded buffer; None when the layout is not taken."""
    if strided_by is None:
        return shard_logits.shape[-1] if shard_logits.is_contiguous() else None
    return _row_stride(shard_logits)


def _row_stride(shard_logits: torch.Tensor):
    """The row stride of a [B, K, N] logits tensor whose B * K rows are evenly spaced with unit column stride (a
    contiguous tensor, or the first N columns of a padded [B * K, rows] buffer), else None."""
    B, K, N = shard_logits.shape
    s0, s1, s2 = shard_logits.stride()
    if (s2 == 1 or N == 1) and s1 >= N and (s0 == K * s1 or B == 1):
        return s1
    return None


def prev_embeddings_available() -> bool:
    """Whether ``prev_embeddings`` can run (its C++ op registered and not killed); callers that skip ``supports``
    check this first and otherwise run the in-tree lines."""
    return _cpp_op(_PREV_OP) is not None


def supports(
    shard_logits: torch.Tensor,
    first_prev: torch.Tensor,
    tokens: torch.Tensor,
    w1: torch.Tensor,
    strided_by: str | None = None,
) -> bool:
    """Whether both launches take these operands; a caller that gets False runs the in-tree lines. The logits are
    contiguous, or, with ``strided_by`` (the name of the target that arms it), the first N columns of a padded row
    buffer."""
    if shard_logits.dim() != 3 or tokens.dim() != 2 or first_prev.dim() != 1 or w1.dim() != 2:
        return False
    if _cpp_op(_PREV_OP) is None or _cpp_op(_WIDEN_OP) is None:
        return False
    B, K, _ = shard_logits.shape
    R = w1.shape[1]
    return (
        shard_logits.is_cuda
        and shard_logits.dtype == torch.bfloat16
        and _logits_row_stride(shard_logits, strided_by) is not None
        and w1.dtype == torch.bfloat16
        and w1.is_contiguous()
        and w1.device == shard_logits.device
        and R >= 16
        and (R & (R - 1)) == 0
        and R <= 1024
        and first_prev.dtype in _INT
        and tokens.dtype in _INT
        and first_prev.is_contiguous()
        and tokens.is_contiguous()
        and first_prev.device == shard_logits.device
        and tokens.device == shard_logits.device
        and first_prev.numel() == B
        and tuple(tokens.shape) == (B, K)
        and B > 0
        and K > 0
    )


def prev_embeddings(
    first_prev: torch.Tensor, tokens: torch.Tensor, w1: torch.Tensor
) -> torch.Tensor:
    """[B * K, R] bf16: row b * K + k is w1[prev] with prev = first_prev[b] (k = 0) or tokens[b, k - 1], and a zero
    row for a prev outside [0, V) -- bitwise markov_prev_embeddings of the in-tree prevs."""
    B, K = tokens.shape
    R = w1.shape[1]
    out = torch.empty(B * K, R, dtype=w1.dtype, device=w1.device)
    op = _cpp_op(_PREV_OP)
    if op is None:
        raise RuntimeError(
            f"prev_embeddings: torch.ops.trtllm.{_PREV_OP} is unavailable (check supports())"
        )
    op(first_prev, tokens, w1, out)
    return out


def widen_add(
    shard_logits: torch.Tensor, bias: torch.Tensor, strided_by: str | None = None
) -> torch.Tensor:
    """fp32 ``shard_logits.float() + bias.to(float32)`` ([B, K, N] bf16 each, the bias contiguous), bitwise torch's.
    The logits are contiguous, or, with ``strided_by`` (the arming target, logged once), the first N columns of a
    padded row buffer."""
    B, K, N = shard_logits.shape
    if bias.dtype != torch.bfloat16 or not bias.is_contiguous() or bias.numel() != B * K * N:
        raise ValueError("widen_add: bias must be contiguous bf16 with the logits' element count")
    row_stride = _logits_row_stride(shard_logits, strided_by)
    if row_stride is None:
        raise ValueError(
            "widen_add: the logits must be contiguous (or, armed by a target, evenly spaced rows)"
        )
    if (
        strided_by is not None
        and not shard_logits.is_contiguous()
        and strided_by not in _STRIDED_LOGGED
    ):
        from tensorrt_llm.logger import logger

        _STRIDED_LOGGED.add(strided_by)
        logger.info(
            f"[catalog] dspark_markov_bias.widen_add: padded logits rows (row stride {row_stride}) armed by "
            f"{strided_by}"
        )
    op = _cpp_op(_WIDEN_OP)
    if op is None:
        raise RuntimeError(
            f"widen_add: torch.ops.trtllm.{_WIDEN_OP} is unavailable (check supports())"
        )
    out = torch.empty(B, K, N, dtype=torch.float32, device=shard_logits.device)
    op(shard_logits, row_stride, bias, out)
    return out
