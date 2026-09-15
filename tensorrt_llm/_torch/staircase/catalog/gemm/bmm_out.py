# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched matmul into a caller-provided output buffer via the trtllm bmm_out op."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def bmm_out(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """Compute `out[i] = a[i] @ b[i]` for every batch index, in one bmm_out call."""
    # ONE guard, and it is the only domain here that fails silently. Everything
    # else this wrapper used to check is rejected loudly by the op itself and is
    # asserted as such in `test_op_rejects_loudly`: a 2-D operand
    # (`batch1 must be a 3D tensor`), a batch or K mismatch, an `out` dtype that
    # differs from the inputs, fp8 operands, and all twelve mixed-dtype
    # combinations over {bf16, fp16, fp32}. Repeating a check the op already
    # makes would only change which error the caller sees.
    #
    # A wrong-shaped `out` is the exception: the op resizes it with nothing but
    # a deprecation warning. Measured on sm_103 / 1.3.0rc26 / torch 2.12, the
    # damage depends on where the buffer came from and neither form is
    # catchable -- a standalone tensor is REALLOCATED (its `data_ptr` moves, so
    # anything that aliased the old storage is silently stale), while a view
    # into a larger arena is resized IN PLACE and overwrites the arena's
    # surrounding layout: 991 of 1024 elements of the arena's original window
    # then disagree with the product, and a sibling view keeps its own stale
    # shape over the rewritten bytes. A 2-D or 4-D `out` is silently reshaped
    # to 3-D by the same path, which this one assert also covers.
    #
    # THE RANK CHECK GUARDS THE GUARD. `b.shape[2]` on a 2-D `b` raises
    # `IndexError: tuple index out of range` from inside the wrapper, replacing
    # the op's own loud, documented `batch1 must be a 3D tensor` with a worse
    # error that names none of the operands. The expression is only well formed
    # once `a` and `b` are 3-D, so it is only evaluated then; every other rank
    # falls through to the op and gets the message the contract quotes.
    if a.dim() == 3 and b.dim() == 3:
        assert out.shape == (a.shape[0], a.shape[1], b.shape[2]), (
            f"out must be [B, M, N] = {(a.shape[0], a.shape[1], b.shape[2])} to match a [B, M, K] "
            f"and b [B, K, N], but is {tuple(out.shape)}; the op would silently resize it"
        )
    torch.ops.trtllm.bmm_out(a, b, out)
