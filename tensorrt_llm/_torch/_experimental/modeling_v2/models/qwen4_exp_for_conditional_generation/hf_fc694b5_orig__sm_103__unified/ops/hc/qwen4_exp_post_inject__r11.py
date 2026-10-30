# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-connection post-block injection: a per-stream gated accumulate of one
block output into the multi-stream residual.

Tier-3 composed from native torch: one kernel-launching call
(``torch.addcmul``) plus view glue. No ``torch.ops.trtllm.*`` op computes this
accumulate with an *identity* cross-stream mix: ``mhc_post_mapping`` applies a
Sinkhorn doubly-stochastic mix across streams, fixes the stream count at 4 and
instantiates its MMA paths for hidden sizes this checkpoint does not use. The
built-in Qwen4-Exp implementation has its own fused op for it, not wrapped here.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

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
    measured.

    Certified: bf16 2-D operands at the checkpoint's 4 x 2560, 64 and 8192
    rows contiguous, 8 rows with `hyper_input` a column slice; graph replay.

    Not certified: other row counts, a row-slice `hyper_input`, hc_count other
    than 4, hidden other than 2560, float16, float32 and float8 operands,
    leading dims beyond one. Hard limits: not
    probed (the largest measured operand is 8192 x 10240 = 2^26.3 elements).
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
