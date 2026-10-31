# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-connection post-block injection: a per-stream gated accumulate of one
block output into the multi-stream residual.

Native torch (sourcing tier 4): one kernel-launching call (``torch.addcmul``)
plus view glue. No ``torch.ops.trtllm.*`` op computes this accumulate from the
gates with an *identity* cross-stream mix: ``mhc_post_mapping`` applies a
Sinkhorn doubly-stochastic mix across streams, fixes the stream count at 4 and
instantiates its MMA paths for hidden sizes this checkpoint does not use; the
built-in Qwen4-Exp's ``trtllm::qwen4_exp_hc_combine``
(``_torch/modules/qwen4_exp/hyper_connection_kernels.py``) takes the injection
*logits* and applies ``2 * sigmoid(logits / hc_count)`` itself, a different
fusion boundary from the gates ``qwen4_exp_gated_residual`` hands this op.
"""

from __future__ import annotations

import torch

from .._op import Arch, Cell, OpWrapper

#: This checkpoint's hyper-connection geometry: 4 streams of 2560, packed
#: stream-major into a 10240-wide residual.
CKPT_HC_COUNT = 4
CKPT_HIDDEN = 2560


class Qwen4ExpPostInject(OpWrapper):
    """`out[..., s * H + i] = hyper_input[..., s * H + i] + block_out[..., i] *
    inject[..., s]`, with `H = block_out.shape[-1]` and `hc_count =
    inject.shape[-1]`: one block output accumulated into **every** stream, each
    scaled by its own gate. Returns a fresh contiguous tensor of `hyper_input`'s
    shape; no input is mutated.

    `hyper_input` is the **pre-norm** residual, packed **stream-major** (last
    dim is `(hc_count, H)`, not `(H, hc_count)` -- both have the same shape, so
    a stream-minor caller is silently wrong). The cross-stream mix is the
    identity: no stream's value enters another.

    Fusion boundary: the accumulate only. The grouped norm, the input mixer
    that produced the block's input and `inject` (`qwen4_exp_gated_residual`),
    and the block itself are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: leading (token dims), hc_count, hidden, dtype; layout
    # "contiguous" (default) | "col_slice" (`buf[:, :width]`, last-dim stride 1).
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"checkpoint 4x2560 bf16, {why}",
            spec=dict(leading=(t,), hc_count=CKPT_HC_COUNT, hidden=CKPT_HIDDEN),
        )
        for t, why in (
            (64, "decode batch / MTP rows, 64 tokens"),
            (8192, "prefill 8192 tokens, largest measured"),
        )
    ) + (
        Cell(
            why="hyper_input a column slice buf[:, :width]",
            spec=dict(leading=(8,), hc_count=CKPT_HC_COUNT, hidden=CKPT_HIDDEN, layout="col_slice"),
        ),
    )
    # The unified tp4/tep4/dep4 target's sites, each with `hyper_input` a column
    # slice of a wider row buffer (row stride twice the packed width). The op
    # sees a rank's full 4 x 2560 residual at every member; only the row count
    # moves: one decode request, max_batch_size, a verify step's trunk and a
    # max_num_tokens prefill chunk.
    CELLS += tuple(
        Cell(
            why=f"col_slice {t} rows: {why}",
            spec=dict(leading=(t,), hc_count=CKPT_HC_COUNT, hidden=CKPT_HIDDEN, layout="col_slice"),
        )
        for t, why in (
            (1, "one decode request; the slice reads as contiguous at one row"),
            (128, "a max_batch_size decode step"),
            (1024, "a verify step's trunk, 128 requests x verify width 8"),
            (8192, "a max_num_tokens prefill chunk, the envelope's largest row"),
        )
    )

    note = """
    Metadata consumed: none. Call sites: twice per decoder layer (after the
    attention/GDN block and after the MoE) and twice in the MTP layer, with
    `inject` the `[T, 4]` weights `qwen4_exp_gated_residual` returned for that
    block -- in `[0, 2]`, because they are `2 * sigmoid(.)`.

    Preconditions, all loud except the guarded one: the packed width must equal
    `inject.shape[-1] * block_out.shape[-1]` (the unflatten raises); all three
    on one CUDA device (raises). The leading dims must be **equal**, not merely
    broadcastable -- a 1-D `inject` or `block_out` was measured to be accepted
    and to broadcast one row over every token (`is_valid`).

    Mixed dtypes are not rejected, they promote: an fp32 `inject` beside a
    bf16 stream returns an fp32 result -- correct values, twice the bytes.
    Recorded, not guarded.

    Rounding: `addcmul` forms product and sum in fp32 opmath and rounds once
    on the store. Measured bit-exact against an fp32 reference rounded once at
    1..8192 tokens; the bf16-multiply-then-bf16-add form differs by 0.0625.
    Wrong operators at `[64, 10240]` bf16, max deviation from the bit-exact
    result: stream-minor packing 10.34, no gate 3.77, gate on the residual 5.89.

    CUDA-graph capture certified; the output is a fresh allocation, so under
    capture it comes from the graph's private pool. Cost, not gated: 17 us at
    64 tokens, 53 us at 2048, 213 us at 8192 (~1.78 TB/s) -- the stride-0
    broadcast operands take a non-vectorized path. An in-place form was not
    measured. Column-slice `hyper_input` on GB300, per call replayed in a CUDA
    graph (operands L2-warm): 2.3 us at 1 row, 4.8 at 128, 25.4 at 1024, 213 at
    8192. Stock `trtllm::qwen4_exp_hc_combine` (Triton, logits in) on the same
    shapes: 1.6, 2.1, 5.0 and 63.7 us -- faster at every row count, behind a
    different fusion boundary (see the module docstring).

    Certified: bf16 2-D operands at the checkpoint's 4 x 2560, 64 and 8192
    rows contiguous; `hyper_input` a column slice (`buf[:, :width]` of a
    2 x width row buffer) at 1, 8, 128, 1024 and 8192 rows; graph replay. At
    one row the slice is contiguous by torch's definition (a size-1 dim's
    stride is ignored), so that cell is the call site's layout, not a strided
    read.

    Not certified: other row counts, a row-slice `hyper_input`, hc_count other
    than 4, hidden other than 2560, float16, float32 and float8 operands,
    leading dims beyond one. Hard limits: not probed. The largest measured
    `hyper_input` is 8192 x 10240 = 2^26.3 elements (its column-slice buffer
    2^27.3); 2^31 elements of the packed residual is 209,715 rows, past every
    row count the target's envelope (max_num_tokens 8192) issues.
    """

    def __call__(
        self,
        hyper_input: torch.Tensor,
        block_out: torch.Tensor,
        inject: torch.Tensor,
    ) -> torch.Tensor:
        streams = hyper_input.unflatten(-1, (inject.shape[-1], block_out.shape[-1]))
        return torch.addcmul(streams, block_out.unsqueeze(-2), inject.unsqueeze(-1)).flatten(-2)

    def reference(
        self,
        hyper_input: torch.Tensor,
        block_out: torch.Tensor,
        inject: torch.Tensor,
    ) -> torch.Tensor:
        """fp32 product and sum, rounded once to the promoted dtype."""
        hc_count, hidden = inject.shape[-1], block_out.shape[-1]
        streams = hyper_input.float().unflatten(-1, (hc_count, hidden))
        gated = block_out.float().unsqueeze(-2) * inject.float().unsqueeze(-1)
        out_dtype = torch.promote_types(
            torch.promote_types(hyper_input.dtype, block_out.dtype), inject.dtype
        )
        return (streams + gated).flatten(-2).to(out_dtype)

    def is_valid(
        self,
        hyper_input: torch.Tensor,
        block_out: torch.Tensor,
        inject: torch.Tensor,
    ) -> None:
        # Measured: a 1-D `inject` or `block_out` is accepted and returns a full
        # `[8, 10240]` result computed from one row shared by every token.
        assert hyper_input.shape[:-1] == block_out.shape[:-1] == inject.shape[:-1], (
            f"leading dims must match exactly; got hyper_input {tuple(hyper_input.shape)}, "
            f"block_out {tuple(block_out.shape)}, inject {tuple(inject.shape)}"
        )


qwen4_exp_post_inject = Qwen4ExpPostInject()
