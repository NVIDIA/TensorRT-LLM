# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish BF16 rows into a Lamport mailbox slot (Kimi K3 o_proj, ``TRTLLM_K3_OPROJ_LAMPORT``).

The o_proj GEMM publishes its partial from its epilogue (``lamport_epilogue.py``) when its tactic
runs the base kernel. Any other tactic (split-K, mixed-cluster) runs as the stock GEMM, and this
kernel then copies the stock output into the slot, so the published values are the stock
GEMM's whatever the tactic. One thread per 16-byte fragment: after the PDL wait it loads the
fragment, turns BF16 ``-0`` into ``+0`` (so no word is the mailbox sentinel) and stores it with
``st.relaxed.sys`` to the slot's multicast address, the publish of the one-shot all-reduce in
``k3_lamport_allreduce.py``. Its dependents (the consumer) may launch at its entry.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from ..blackwell.utils import TRTLLM_ENABLE_PDL
from .k3_lamport import (
    FRAGMENT_BYTES,
    FRAGMENT_ELEMS,
    PUBLISH_ST,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    load_v4,
    publish_v4,
    sanitize_v4,
)

#: CTA sizes to try, largest first (a multiple of 32 dividing the row's fragments).
_CTA_THREADS = (128, 64, 32)


class K3LamportPublishKernel:
    """``slot row i = x row i`` (BF16, ``-0`` stored as ``+0``) for ``m`` rows of ``hidden``."""

    def __init__(self, hidden: int, threads: int, use_pdl: bool):
        frags = hidden // FRAGMENT_ELEMS
        if hidden % FRAGMENT_ELEMS or threads % 32 or frags % threads:
            raise ValueError(f"hidden {hidden} / threads {threads}: no whole fragments per CTA")
        self.row_bytes = hidden * 2
        self.threads = threads
        self.cpr = frags // threads
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        x_addr: Int64,
        slot_addr: Int64,
        slot_row_bytes: Int64,
        m: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(x_addr, slot_addr, slot_row_bytes).launch(
            grid=(m * self.cpr, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(self, x_addr: Int64, slot_addr: Int64, slot_row_bytes: Int64):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx // self.cpr
        col = Int64((bidx - row * self.cpr) * self.threads + tidx) * FRAGMENT_BYTES
        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_launch_dependents()
            # The mailbox contract: publish only after the PDL wait (x is the predecessor's).
            griddepcontrol_wait()
        words = sanitize_v4(load_v4(x_addr + Int64(row) * self.row_bytes + col))
        publish_v4(slot_addr + Int64(row) * slot_row_bytes + col, words, PUBLISH_ST)


@functools.cache
def _compiled(hidden: int, threads: int, use_pdl: bool):
    kernel = K3LamportPublishKernel(hidden, threads, use_pdl)
    addr = Int64(1 << 20)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(kernel, addr, addr, Int64(hidden * 2), Int32(1), stream)


def _cta_threads(hidden: int) -> int:
    frags = hidden // FRAGMENT_ELEMS
    for threads in _CTA_THREADS:
        if frags % threads == 0:
            return threads
    raise ValueError(f"hidden {hidden}: no CTA size of {_CTA_THREADS} divides {frags} fragments")


def run_k3_lamport_publish(
    x: torch.Tensor,
    slot_address: int,
    leading_dim: int,
    use_pdl: bool = TRTLLM_ENABLE_PDL,
) -> None:
    """Publish ``x`` (BF16 ``[M, H]``, contiguous) into the mailbox slot at ``slot_address``
    (rows ``leading_dim`` elements apart; a multicast address reaches every rank's copy), on the
    current stream after its PDL wait. The caller owns the mailbox use (see
    ``tensorrt_llm/_torch/distributed/k3_lamport_mailbox.py``)."""
    if x.dtype != torch.bfloat16 or x.dim() != 2 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("x must be a contiguous 2-D bf16 CUDA tensor")
    m, hidden = x.shape
    if m == 0 or hidden % FRAGMENT_ELEMS or x.data_ptr() % FRAGMENT_BYTES:
        raise ValueError(f"x [{m}, {hidden}] must have rows of whole, 16-byte aligned fragments")
    if slot_address % FRAGMENT_BYTES or leading_dim < hidden or leading_dim % FRAGMENT_ELEMS:
        raise ValueError(
            f"Lamport slot {slot_address:#x} / leading dim {leading_dim} must be 16-byte "
            f"aligned and hold a row of {hidden}"
        )
    compiled = _compiled(hidden, _cta_threads(hidden), use_pdl)
    compiled(
        Int64(x.data_ptr()),
        Int64(slot_address),
        Int64(leading_dim * 2),
        Int32(m),
        cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
    )


__all__ = ["K3LamportPublishKernel", "run_k3_lamport_publish"]
