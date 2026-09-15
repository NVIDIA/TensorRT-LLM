# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP8 x MXFP8 dense GEMM in nn.Linear layout via the trtllm mxfp8_mxfp8_gemm op."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def _swizzled_scale_bytes(rows: int, k: int) -> int:
    """Length of the 128x4-swizzled UE8M0 scale buffer for a `[rows, k]` operand."""
    return -(-rows // 128) * 128 * (-(-(k // 32) // 4) * 4)


def mxfp8_mxfp8_gemm(
    act: torch.Tensor,
    act_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return `global_scale * (act @ weight.T)` over block-scaled MXFP8 operands, one op call."""
    # A guard here is earned only by a violation that goes wrong SILENTLY. The op
    # rejects thirteen input errors itself, `test_op_rejects_loudly` asserts all
    # thirteen, and none is repeated here: operand and scale dtypes, contiguity
    # of both operands, device, rank, the shared K, K % 32, N % 32 and out_dtype.
    # A zero-row call is in that same loud set -- it raises `Error Internal` from
    # the CUTLASS runner rather than returning an empty result -- so `M >= 1` is
    # a documented caller precondition, not a wrapper assert. Repeating a check
    # the op already makes would only change which error the caller sees.
    #
    # The two below are different: neither is validated by the op at all, so
    # violating one gives the caller nothing to catch.
    assert global_scale.numel() == 1, (
        "global_scale must hold exactly one element; extra elements are "
        "silently ignored (element 0 is used as alpha)"
    )
    # Each scale buffer is sized from *its own* operand. Sizing both from `act`
    # would mislabel a K mismatch as a short buffer and preempt the op's own
    # check, which is the better error for that case.
    for name, buf, rows, k in (
        ("act_scale", act_scale, act.shape[0], act.shape[-1]),
        ("weight_scale", weight_scale, weight.shape[0], weight.shape[-1]),
    ):
        need = _swizzled_scale_bytes(rows, k)
        assert buf.numel() >= need, (
            f"{name} holds {buf.numel()} bytes but the 128x4 swizzled layout for "
            f"{rows} rows x {k // 32} blocks needs {need}; the op does not check "
            f"this length at all and reads by computed offset regardless. There "
            f"is no deterministic failure to catch: one byte short returned a "
            f"silently wrong result (15.0 away on this checkpoint's dense "
            f"surfaces), and a tenth of the size has been observed returning "
            f"garbage, returning inf/NaN, AND faulting asynchronously with "
            f"`CUDA error: an illegal memory access was encountered`, which ends "
            f"the CUDA context for the whole process"
        )
    return torch.ops.trtllm.mxfp8_mxfp8_gemm(
        act,
        act_scale,
        weight,
        weight_scale,
        global_scale,
        out_dtype,
    )
