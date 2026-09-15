# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic MXFP8 quantization: bf16/fp16 -> e4m3 data + per-32-element UE8M0 block scales."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def mxfp8_quantize(
    input: torch.Tensor,
    swizzled_layout: bool = True,
    alignment: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to MXFP8: returns (e4m3 data [..., pad_up(K, alignment)], uint8 UE8M0 scales, 1D).

    One scale per 32 contiguous elements along the last dim; `swizzled_layout`
    selects the 128x4 swizzled scale order (True) or the row-major linear
    order (False).
    """
    # ONE guard, on the only domain here that fails silently. The op checks
    # `alignment % 32 == 0` and raises, but `0 % 32 == 0` passes that check and
    # then reaches `padded_k = ((k + alignment - 1) / alignment) * alignment`
    # (`cpp/tensorrt_llm/thop/mxFp8Quantize.cpp:60`) with no zero guard.
    #
    # What that does depends on the HOST ISA, and on this one it is the
    # dangerous direction: aarch64's SDIV returns 0 for a zero divisor instead
    # of trapping the way x86's DIV does, so on this GB300 the call RETURNS an
    # empty `[M, 0]` result and an empty scale buffer rather than dying. A
    # negative multiple of 32 is quiet for a different reason -- it passes the
    # modulus check and truncates, returning fewer columns than K with every
    # surviving column bit-exact, so the result looks entirely well-formed with
    # the tail of every row simply gone.
    #
    # Everything else the op rejects itself, loudly, and `test_op_rejects_*`
    # asserts each message: a non-multiple-of-32 alignment, K % 32, an fp32 or
    # 1-D or non-contiguous input, and a CPU tensor.
    assert alignment > 0, (
        f"alignment must be positive, got {alignment}; the op does not check the sign. "
        f"0 passes its `% 32 == 0` check and then divides by zero -- which on aarch64 "
        f"returns an empty [M, 0] result instead of trapping -- and a negative multiple "
        f"of 32 silently truncates K instead of padding it"
    )
    return torch.ops.trtllm.mxfp8_quantize(input, swizzled_layout, alignment)
