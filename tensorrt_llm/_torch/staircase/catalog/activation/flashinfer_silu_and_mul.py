# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SiLU-gated multiply (SwiGLU activation) via the flashinfer silu_and_mul kernel."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def flashinfer_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Return `silu(x[..., :d]) * x[..., d:]` with `d = x.shape[-1] // 2` as a new tensor."""
    # The op's own check is on the *row*, `x.shape[-1] * itemsize % 16 == 0`,
    # which a final dimension of 24 passes. The kernel vectorizes over the two
    # halves, so what has to be 16-byte aligned is the *half*: at 24 the
    # second half starts mid-vector and the launch dies with `CUDA misaligned
    # address`, poisoning the context rather than raising. This is the
    # stricter precondition the contract states.
    width = x.shape[-1]
    assert width % 2 == 0, f"x.shape[-1] must be even to split in half; got {width}"
    half_bytes = (width // 2) * x.element_size()
    assert half_bytes % 16 == 0, (
        f"x.shape[-1] // 2 must be a whole number of 16-byte vectors; "
        f"{width} halves to {half_bytes} bytes, which is not a multiple of 16 "
        f"-- the kernel would fault on the misaligned second half"
    )
    assert half_bytes >= 16, (
        f"x.shape[-1] // 2 must hold at least one 16-byte vector; {width} "
        f"gives {half_bytes} bytes, for which the computed block size is 0"
    )
    return torch.ops.trtllm.flashinfer_silu_and_mul(x)
