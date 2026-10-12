# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic per-tensor E4M3 quantization with a power-of-two scale, via the trtllm Triton op.

The op is `torch.ops.trtllm.e4m3_quantize_pow2_dynamic`, registered in
`tensorrt_llm/_torch/custom_ops/e4m3_quantize_pow2_dynamic.py`.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops.e4m3_quantize_pow2_dynamic  # noqa: F401 — registers the op
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_E4M3_MAX = 448.0
_AMAX_FLOOR = 1e-4

#: Row counts the DeepSeek-V4.1 engine hands the attention query at: decode at
#: 1..32 requests x next_n 1 or 6 (DSpark verify), and context chunks up to 8192.
_QUERY_ROWS = (1, 2, 6, 8, 32, 48, 192, 300, 2048, 4097, 8192)


class E4m3QuantizePow2Dynamic(OpWrapper):
    """Quantize a whole tensor to E4M3 with one power-of-two scale from its amax.

    One call, three launches, no host synchronization::

        amax         = max(max|x|, 1e-4)                      # fp32, whole tensor
        s            = 2^ceil(log2(amax * fp32(1/448)))
        out          = e4m3_rne(clamp(x * (1/s), -448, 448))  # exact scaling
        scale_out[0] = s * scale_mult                         # one fp32 multiply

    Fusion boundary: the amax reduction, the scale and the quantization are
    inside the call; the consumer of `scale_out` (an attention's `bmm1_scale`,
    with `scale_mult = softmax_scale * kv_scale`) is not. Writes `out` and
    element 0 of `scale_out` in place and returns `(out, scale_out)`.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"query T={t} H={h}: [T, H, 512] bf16 at an engine row count",
                spec=dict(
                    kind="query", T=t, H=h, dtype=torch.bfloat16, scale_mult=512**-0.5 * 2.0**-7
                ),
            )
            for h in (16, 64)
            for t in _QUERY_ROWS
        )
        + tuple(
            Cell(
                why=f"input dtype {str(dt).removeprefix('torch.')}: [33, 16, 512]",
                spec=dict(kind="query", T=33, H=16, dtype=dt, scale_mult=1.0),
            )
            for dt in (torch.bfloat16, torch.float16, torch.float32)
        )
        + tuple(
            Cell(
                why=f"scale edge amax={amax!r}: {reason}",
                spec=dict(kind="edge", amax=amax, dtype=torch.float32, scale_mult=1.0),
            )
            for amax, reason in (
                (448.0, "fp32(1/448) rounds 448 * 2^k up to 2^(k+1)"),
                (448.0 * 2**-9, "the same rounding nine binades down"),
                (449.0, "just above 448, scale 2"),
                (3.0e4, "far above 448"),
                (1e-6, "below the 1e-4 amax floor"),
                (0.0, "an all-zero tensor, the floor alone sets the scale"),
            )
        )
    )

    note = """
    H is 16 for tep4/tp4 and 64 for dep4. Per-(token, head) magnitudes in the
    cells span 2^-6 .. 2^6, the range the golden queries reach.

    `amax * fp32(1/448)` is formed in fp32 like DeepSeek-V4.1's MXFP8
    activation scale; because `fp32(1/448)` is slightly above `1/448`, an amax
    of exactly `448 * 2^k` gives `s = 2^(k+1)`. No value saturates:
    `max|x| / s` lies in `[224, 448]`, and `out * s` is `x` rounded to E4M3's
    3-bit mantissa (relative round-trip error <= 2^-4 in E4M3's normal range).

    Metadata consumed: none.

    Preconditions: `x` contiguous bf16/fp16/fp32 holding finite values (a NaN
    makes the amax NaN); `out` contiguous float8_e4m3fn with `x.numel()`
    elements; `scale_out` float32 with at least one element. The op raises
    ValueError on a wrong dtype, device, contiguity or numel. The one scale
    spans the whole tensor, so an outlier coarsens every other value's
    subnormal threshold.

    Allocates one float of scratch per call; inside CUDA-graph capture it comes
    from the graph's pool, and a replay recomputes the scale from the data.

    Not certified: tensors above 8192 x 64 x 512 elements.
    """

    def raw_call(
        self,
        x: torch.Tensor,
        out: torch.Tensor,
        scale_out: torch.Tensor,
        scale_mult: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torch.ops.trtllm.e4m3_quantize_pow2_dynamic(x, out, scale_out, scale_mult)
        return out, scale_out

    def reference(
        self,
        x: torch.Tensor,
        out: torch.Tensor,
        scale_out: torch.Tensor,
        scale_mult: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Codes and the scale output, derived independently of the kernel.

        The scale exponent comes from `torch.frexp` in fp64 rather than from
        the kernel's IEEE-bit derivation; the codes are torch's RNE cast of
        `x * 2^-e`, an exact scaling. `out` and `scale_out` are named to mirror
        `raw_call` and are not mutated: the expected `out` is returned as a new
        tensor of its shape, and the expected `scale_out` is a copy with
        element 0 replaced.
        """
        amax = max(x.float().abs().max().item(), _AMAX_FLOOR) if x.numel() else _AMAX_FLOOR
        scaled = torch.tensor(amax, dtype=torch.float32) * torch.tensor(
            1.0 / _E4M3_MAX, dtype=torch.float32
        )
        mantissa, exponent = torch.frexp(scaled.double())
        e = int(exponent.item()) - (1 if float(mantissa.item()) == 0.5 else 0)
        codes = (x.float() * (2.0**-e)).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
        expected_scale = scale_out.clone()
        expected_scale[0] = (
            torch.tensor(2.0**e, dtype=torch.float32)
            * torch.tensor(scale_mult, dtype=torch.float32)
        ).item()
        return codes.reshape(out.shape), expected_scale

    def compare(
        self,
        out: tuple[torch.Tensor, torch.Tensor],
        ref: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Bitwise on the codes and on `scale_out[0]`.

        Kernel and reference make the same exact power-of-two scaling and the
        same RNE cast, so there is nothing to round differently: any code that
        differs is a wrong scale. Controls that use a non-power-of-two scale or
        round the scale down change more than 30% of the codes.
        """
        (codes, scale_out), (ref_codes, ref_scale) = out, ref
        assert codes.dtype == ref_codes.dtype == torch.float8_e4m3fn, (codes.dtype, ref_codes.dtype)
        assert codes.shape == ref_codes.shape, (codes.shape, ref_codes.shape)
        bad = (codes.view(torch.uint8) != ref_codes.view(torch.uint8)).sum().item()
        assert bad == 0, f"{bad} of {codes.numel()} codes differ"
        assert scale_out[0].item() == ref_scale[0].item(), (
            scale_out[0].item(),
            ref_scale[0].item(),
        )


e4m3_quantize_pow2_dynamic = E4m3QuantizePow2Dynamic()
