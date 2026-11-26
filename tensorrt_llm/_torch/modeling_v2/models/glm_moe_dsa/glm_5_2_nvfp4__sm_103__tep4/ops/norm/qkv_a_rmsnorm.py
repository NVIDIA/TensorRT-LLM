# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`qkv_a_rmsnorm`: the q_a and kv norms of the fused qkva projection's output, on the kernel
that also writes q_a's MXF8 operand for its reader where one is asked for and the kernel
takes the call, else on the plain norm (warp-row or one CTA per row by the row count).

Two readers take that operand: the decode q_b MXF8 GEMM (`qb_pre`, the layer's load-time
handoff: the norm writes into the GEMM's own operand buffers and releases it at entry, so
q_b runs without its activation quant), and the fp8 context attention's q_b (`q8_rows`, a
step's plan: the norm writes the first `q8_rows` rows into a new operand it returns).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import OpWrapper

from .._build import Count
from ..gemm.projection import quant_reads
from . import cute_dsl_qkv_a_rmsnorm as _qkv_a_norm_op
from . import cute_dsl_qkv_a_rmsnorm_q8 as _qkv_a_norm8_op
from .cute_dsl_qkv_a_rmsnorm import cute_dsl_qkv_a_rmsnorm
from .cute_dsl_qkv_a_rmsnorm_q8 import cute_dsl_qkv_a_rmsnorm_q8


@dataclass(frozen=True)
class QkvANormHandle:
    """What `qkv_a_rmsnorm` launches: `plain` the norm's two kernels, `q8` the norm writing
    q_a's MXF8 operand (None: never written), `qb_operand` the decode q_b MXF8 GEMM's own
    operand buffers `(e4m3, scales)` (None without that GEMM), `eps`."""

    plain: Any
    q8: Any
    qb_operand: tuple | None
    eps: float
    count: Count


def build(plain, q8, qb_runner, eps: float, count: Count) -> QkvANormHandle:
    """The handle over the kernels built at load: `qb_runner` is the decode q_b MXF8 GEMM's
    runner (or None), whose operand buffers the norm writes."""
    return QkvANormHandle(
        plain=plain,
        q8=q8,
        qb_operand=None if qb_runner is None else qb_runner.operand,
        eps=eps,
        count=count,
    )


class QkvARmsnorm(OpWrapper):
    """The norms of `qkva` (`[rows, q_lora + kv_lora + rope]`): returns `(q_a, latent, q8,
    qb_pre)`. The kv norm writes in place, so `latent` (`[ckv | k_pe]`) is a strided view of
    `qkva`'s tail. With `q8_rows` (eager context rows the q8 kernel takes) the norm also
    writes the first `q8_rows` q_a rows as `q8`, the fp8 context attention's q_b operand
    (counted `qkv_a_q8`), skipping the bf16 q rows below `q_store_from` (no reader;
    `qkv_a_q8_skip`). Else with `qb_pre` on decode rows (up to 64) the norm writes q_a's
    bytes into the decode q_b GEMM's operand: `qb_pre` comes back True where q_a is an
    operand that GEMM reads, and q_b takes it as `dec_pre`. Else the plain norm (counted
    `qkv_a_warp_row` / `qkv_a_cta`, `qkv_a_direct` on the in-place layout)."""

    def __call__(
        self,
        qkva: torch.Tensor,
        q_weight: torch.Tensor,
        kv_weight: torch.Tensor,
        handle: QkvANormHandle,
        qb_pre: bool = False,
        q8_rows: int = 0,
        q_store_from: int = 0,
    ):
        q_lora, kv_lora = int(q_weight.shape[0]), int(kv_weight.shape[0])
        rope = int(qkva.shape[1]) - q_lora - kv_lora
        q_raw, ckv_raw, _ = torch.split(qkva, [q_lora, kv_lora, rope], -1)
        latent = torch.split(qkva, [q_lora, kv_lora + rope], -1)[1]
        rows = int(q_raw.shape[0])
        h = handle
        if (
            q8_rows
            and h.q8 is not None
            and rows >= _qkv_a_norm_op.WARP_ROW_MIN_ROWS
            and not torch.cuda.is_current_stream_capturing()
            and _qkv_a_norm8_op.takes(q_raw, q_weight, ckv_raw, kv_weight)
        ):
            q8 = _qkv_a_norm8_op.operand(q8_rows, qkva.device)
            q_a = cute_dsl_qkv_a_rmsnorm_q8(
                q_raw, q_weight, ckv_raw, kv_weight, h.eps, *q8, h.q8, q_store_from=q_store_from
            )
            h.count("qkv_a_q8", True)
            if q_store_from:
                h.count("qkv_a_q8_skip", True)
                if h.q8.ilp:
                    h.count("norm_ilp", True)
            return q_a, latent, q8, False
        if (
            qb_pre
            and rows <= _qkv_a_norm8_op.DECODE_MAX_ROWS
            and h.q8 is not None
            and h.qb_operand is not None
            and _qkv_a_norm8_op.takes(q_raw, q_weight, ckv_raw, kv_weight)
        ):
            xq, sf = h.qb_operand
            q_a = cute_dsl_qkv_a_rmsnorm_q8(
                q_raw, q_weight, ckv_raw, kv_weight, h.eps, xq[:rows], sf, h.q8
            )
            return q_a, latent, None, quant_reads(q_a)
        # Both kernels are this op's; the counter shows which one a step's row count took.
        wr = _qkv_a_norm_op.takes_warp_row(q_raw, q_weight, ckv_raw, kv_weight)
        h.count("qkv_a_warp_row" if wr else "qkv_a_cta", True)
        # Eager calls on the in-place layout skip the kernel module's per-call checks.
        if _qkv_a_norm_op.takes_direct(q_raw, q_weight, ckv_raw, kv_weight):
            h.count("qkv_a_direct", True)
        q_a, _ = cute_dsl_qkv_a_rmsnorm(q_raw, q_weight, ckv_raw, kv_weight, h.eps, h.plain)
        return q_a, latent, None, False

    def reference(
        self,
        qkva: torch.Tensor,
        q_weight: torch.Tensor,
        kv_weight: torch.Tensor,
        handle: QkvANormHandle,
        qb_pre: bool = False,
        q8_rows: int = 0,
        q_store_from: int = 0,
    ):
        """`(q_a, latent)`: the two RMS norms in fp32, rounded once to `qkva`'s dtype; the
        latent's k_pe columns pass through. `qkva` is not written."""
        q_lora, kv_lora = int(q_weight.shape[0]), int(kv_weight.shape[0])
        x = qkva.float()

        def rms(t, w):
            return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + handle.eps) * w.float()

        q_a = rms(x[:, :q_lora], q_weight).to(qkva.dtype)
        ckv = rms(x[:, q_lora : q_lora + kv_lora], kv_weight)
        latent = torch.cat([ckv, x[:, q_lora + kv_lora :]], -1).to(qkva.dtype)
        return q_a, latent


qkv_a_rmsnorm = QkvARmsnorm()
