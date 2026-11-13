# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated RMS normalization with the gate read from a token-major 3-D view,
with optional fused static-scale FP8 output quantization."""

from __future__ import annotations

from typing import Optional

import torch
import triton

# Registration is lazy: the op is defined beside the mamba/GDN mixer, and
# `import tensorrt_llm._torch.custom_ops` alone does not make
# torch.ops.trtllm.rms_norm_gated_token_major resolve.
import tensorrt_llm._torch.modules.mamba.layernorm_gated as _layernorm_gated
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The checkpoint's GDN epilogue: 48 value heads of 128, `output_gate_type: sigmoid`.
_CKPT = dict(heads=48, width=128, dtype=torch.bfloat16, eps=1e-6, gate_activation="sigmoid")
#: One tp4/tep4 rank's share: 12 of the 48 value heads, `z` the column view of that rank's
#: 4120-wide input projection `[q k v | z | a | b]` = 2560 | 1536 | 12 | 12.
_TP4 = dict(_CKPT, heads=12, z="in_proj", z_row_stride=4120)
#: A dep4 rank: all 48 heads, `z` in the full 16480-wide projection.
_DEP4 = dict(_CKPT, z="in_proj", z_row_stride=16480)
#: Prefill-sized launches (at least this many rows, i.e. 683 tokens of 48 heads) run
#: the multi-row kernel at `_PREFILL_ROWS` rows a program.
_PREFILL_MIN_ROWS = 32768  # capacity: fixed: a dispatch threshold, not a buffer size
_PREFILL_ROWS = 8


def _multirow(x: torch.Tensor, z: torch.Tensor, gate_activation: str) -> bool:
    """The registered op's multi-row branch (its `eligible`, the gates it knows). A `z`
    that does not cover `x` goes to the registered op, which rejects it."""
    n = x.shape[-1]
    return (
        x.shape[0] == z.shape[0] * z.shape[1]
        and z.shape[2] == n
        and x.stride(-1) == 1
        and z.stride(2) == 1
        and z.stride(1) == n
        and n <= 256
        and n & (n - 1) == 0
        and gate_activation in ("silu", "sigmoid")
    )


def _prefill_multirow(x: torch.Tensor, z: torch.Tensor, gate_activation: str) -> bool:
    return x.shape[0] >= _PREFILL_MIN_ROWS and _multirow(x, z, gate_activation)


def _direct_launch(x, z, weight, eps, gate_activation):
    """The registered op's own multi-row launch (same kernel, 4 rows a program, PDL by the
    op's policy), skipping only the custom-op dispatch."""
    rows, n = x.shape
    out = torch.empty_like(x)
    grid = triton.cdiv(rows, _layernorm_gated._MULTIROW_ROWS)
    device = x.device.index
    pdl = _layernorm_gated._multirow_pdl(grid, device)
    with torch.cuda.device(device):
        _layernorm_gated._rms_norm_gated_fwd_multirow_kernel[(grid,)](
            x, out, weight, z, None, None, x.stride(0), out.stride(0), z.stride(0), rows, eps,
            N=n, ROWS=_layernorm_gated._MULTIROW_ROWS, HEADS_PER_TOK=z.shape[1],
            GATE_SIGMOID=gate_activation == "sigmoid", WEIGHT_IS_DELTA=False,
            LAUNCH_WITH_PDL=pdl, num_warps=_layernorm_gated._MULTIROW_NUM_WARPS,
            EARLY_TRIGGER=_layernorm_gated._pdl_device_policy(device)[0], launch_pdl=pdl,
        )  # fmt: skip
    return out


def _prefill_launch(x, z, weight, eps, gate_activation):
    """The multi-row kernel at `_PREFILL_ROWS` rows a program (no PDL: the grid covers every SM)."""
    rows, n = x.shape
    out = torch.empty_like(x)
    _layernorm_gated._rms_norm_gated_fwd_multirow_kernel[(triton.cdiv(rows, _PREFILL_ROWS),)](
        x, out, weight, z, None, None, x.stride(0), out.stride(0), z.stride(0), rows, eps,
        N=n, ROWS=_PREFILL_ROWS, HEADS_PER_TOK=z.shape[1], GATE_SIGMOID=gate_activation == "sigmoid",
        WEIGHT_IS_DELTA=False, LAUNCH_WITH_PDL=False, num_warps=4,
    )  # fmt: skip
    return out


class RmsNormGatedTokenMajor(OpWrapper):
    """`rmsnorm(x) * weight * act(z)`, new tensor out, row `r` of `x` gated by `z[r // heads, r % heads]`.

    `x` is `[T * heads, N]`; `z` is `[T, heads, N]` (heads = `z.shape[1]`),
    its `(heads, N)` block contiguous per token and its token stride free, so a
    column slice of a wider projection is read in place. `act(v)` is
    `v * sigmoid(v)` for `"silu"` and `sigmoid(v)` for `"sigmoid"`. All fp32,
    rounded once. Norm before gate, plain `w` (not `1 + w`).

    With `fp8_scale` (one fp32 element, the downstream linear's static input
    scale) the output is `float8_e4m3fn`: the gated value is rounded to
    `x.dtype`, multiplied by `1 / fp8_scale[0]` and saturated at +-448 --
    byte-identical to the unfused norm -> static-quantize pair.

    Call sites (`PrefillTarget._gdn_epilogue`, and `DecodeTarget._gdn_epilogue`
    at an MTP verify): `z` is the column view of the GDN layer's one
    input-projection output, `[T, 6144]` of `[T, 16480]` reshaped `[T, 48, 128]`
    on a dep4 rank, `[T, 1536]` of `[T, 4120]` reshaped `[T, 12, 128]` on a
    tp4/tep4 rank, which holds 12 of the 48 value heads.

    Fusion boundary: norm, weight, gate and the optional quantize. The caller
    owns both projections producing `x` and `z`, and every reshape.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: z the in_proj column view (token stride 16480), 128 tokens",
            spec=dict(tokens=128, z="in_proj", **_CKPT),
        ),
        Cell(
            why="prefill: z the in_proj column view, 16384 tokens at 8 rows a program",
            spec=dict(tokens=16384, z="in_proj", **_CKPT),
        ),
        Cell(
            why="683 tokens: the first launch at 8 rows a program",
            spec=dict(tokens=683, z="in_proj", **_CKPT),
        ),
        Cell(
            why="tp4/tep4, one token: 12 heads, z at token stride 4120; 4-row kernel, PDL",
            spec=dict(tokens=1, **_TP4),
        ),
        Cell(
            why="tp4/tep4, 4 tokens (an MTP-3 verify of one request): 4-row kernel, PDL",
            spec=dict(tokens=4, **_TP4),
        ),
        Cell(
            why="tp4/tep4, 128 tokens (the con-128 decode batch): grid past the SM count",
            spec=dict(tokens=128, **_TP4),
        ),
        Cell(
            why="tp4/tep4, 1024 tokens: the MTP-7 verify of 128 requests x 8 rows",
            spec=dict(tokens=1024, **_TP4),
        ),
        Cell(
            why="tp4/tep4 prefill at max_num_tokens 8192: 98304 rows on the 8-row kernel",
            spec=dict(tokens=8192, **_TP4),
        ),
        Cell(
            why="dep4 decode: 48 heads at token stride 16480, a rank's 32 tokens",
            spec=dict(tokens=32, **_DEP4),
        ),
    )

    note = """
    Metadata consumed: none. Two Triton kernels: a multi-row kernel (4 rows per
    program; 8 from 32768 rows, without fp8 output) when `x.stride(-1) == 1`,
    `z.stride(2) == 1`, `z.stride(1) == N` and N is a power of two <= 256 --
    the checkpoint's N = 128 -- else the generic single-row kernel on a packed
    copy of `z`. PDL on the 4-row path only when the grid is smaller than the
    SM count and `TRTLLM_ENABLE_PDL != 0`; results are bitwise identical with
    it off. The 8-row programs reduce a row's mean square in another order
    than the 4-row ones (one-ulp differences on some outputs). With
    `direct_launch=True` (the FP8 target) below the 8-row threshold, the 4-row
    launch is made here (`_direct_launch`, a copy of the registered op's own),
    skipping only the custom-op dispatch: 24 instead of 39 us of host time a
    call on GB300. The test holds it bitwise equal to the op at the
    direct-launch cells and more shapes, so a change to the op's launch fails
    there.

    Loud (measured), so not guarded: 3-D `x` (ValueError), a `z` not covering `x`
    (AssertionError "does not match x shape"; 2-D `z` ValueError), an unknown
    `gate_activation` (ValueError before launch), and on the fallback a
    non-unit `x.stride(-1)` (bare AssertionError). `is_valid` covers what the
    multi-row kernel accepts silently: a weight longer than N (truncated to
    `weight[:N]`) or shorter (read out of bounds), and an `fp8_scale` with
    extra elements (only `[0]` read) or not fp32 (widened silently).

    `weight` and `z` are widened to fp32 and need not match `x.dtype`. The
    output is freshly allocated and contiguous; no input is mutated.

    Both kernels are Triton, JIT-compiled at first call for the running GPU;
    the cells ran that JIT path on sm_103 (GB300, 152 SMs).

    Certified: bf16 `x` with the sigmoid gate over the in_proj column view of
    `z`, N = 128.
      48 heads, token stride 16480 (tp1 and dep4): 128 and 32 tokens on the
        4-row kernel without PDL; 683 tokens (the first 8-row launch) and
        16384 tokens on the 8-row kernel.
      12 heads, token stride 4120 (a tp4/tep4 rank): 1 and 4 tokens on the
        4-row kernel with PDL (grids of 3 and 12 programs); 128 and 1024 tokens
        on it without; 8192 tokens (98304 rows) on the 8-row kernel.
      24 heads, token stride 8240 (the FP8 target's tp2/tep2 rank, with
        `direct_launch`, cells in the test): 1 and 683 tokens on the 4-row
        kernel, 8192 tokens on the 8-row kernel.
    A one-token `z` view reports the dense token stride (1536 at 12 heads);
    only token 0 is read, so it computes the same.
    Not certified: fp8 output, the silu gate, the generic fallback kernel,
    fp16/fp32 operands, a strided `x`. At 12 heads, the 8-row kernel's first
    launch (2731 tokens) and its masked last program, which an odd token count
    from 2731 up leaves half full (no 48-head row count does, and the 4-row
    kernel never has one at either width): a prompt's last prefill chunk.

    Numerics: bf16 inside the default tolerance against the fp32 reference.
    Measured on sm_103: at most one bf16 ulp from the bf16-rounded reference,
    on about 1 in 10^5 elements or fewer (none at 1, 4 and 32 tokens).
    Against the unrounded fp32 value the output's relative L2 (1.56e-3 to
    1.66e-3) equals the rounded reference's, and two calls are bitwise equal.
    fp8 output byte-identical (measured). Nothing loosened.

    Hard limits
      N * itemsize  fallback holds a row in 64 KB  boundary 65536 B
      probed at bf16 N = 32768 (returns) and 40960 (RuntimeError)
      offsets       the multi-row kernel (4 and 8 rows a program) indexes x, z
                    and out in int64; the widest call the envelope makes (8192
                    tokens x 48 heads x 128, z at stride 16480) stays below
                    2^31 elements on every operand anyway
    """

    def __call__(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        fp8_scale: Optional[torch.Tensor] = None,
        gate_activation: str = "silu",
        direct_launch: bool = False,
    ) -> torch.Tensor:
        if fp8_scale is None and z.dim() == 3 and _prefill_multirow(x, z, gate_activation):
            return _prefill_launch(x, z, weight, eps, gate_activation)
        if (
            direct_launch
            and fp8_scale is None
            and x.dim() == 2
            and z.dim() == 3
            and weight.is_contiguous()
            and _multirow(x, z, gate_activation)
        ):
            return _direct_launch(x, z, weight, eps, gate_activation)
        return torch.ops.trtllm.rms_norm_gated_token_major(
            x, z, weight, eps, fp8_scale, gate_activation
        )

    def reference(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        fp8_scale: Optional[torch.Tensor] = None,
        gate_activation: str = "silu",
        direct_launch: bool = False,
    ) -> torch.Tensor:
        """fp32 throughout; row `t * heads + h` pairs with `z[t, h]`. `direct_launch` does
        not change the result.

        With `fp8_scale` the gated value is rounded to `x.dtype` first, then
        multiplied by the fp32 reciprocal of the scale, as the kernel does.
        """
        num_rows, width = x.shape
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps) * weight.float()
        zf = z.reshape(num_rows, width).float()
        gate = torch.sigmoid(zf)
        if gate_activation != "sigmoid":
            gate = gate * zf
        y = y * gate
        if fp8_scale is None:
            return y.to(x.dtype)
        return (y.to(x.dtype).float() * (1.0 / fp8_scale.float())).to(torch.float8_e4m3fn)

    def is_valid(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        fp8_scale: Optional[torch.Tensor] = None,
        gate_activation: str = "silu",
        direct_launch: bool = False,
    ) -> None:
        assert weight.shape == (x.shape[-1],), (
            f"weight must be 1-D of length x.shape[-1]={x.shape[-1]}; got "
            f"{tuple(weight.shape)} -- the multi-row kernel reads weight[0:N] "
            "unmasked, so a longer weight is silently truncated and a shorter "
            "one is read out of bounds"
        )
        if fp8_scale is not None:
            assert fp8_scale.numel() == 1, (
                f"fp8_scale must hold exactly one element; got {fp8_scale.numel()} "
                "-- the multi-row kernel loads fp8_scale[0] and ignores the rest"
            )
            assert fp8_scale.dtype == torch.float32, (
                f"fp8_scale must be float32; got {fp8_scale.dtype} -- the "
                "multi-row kernel widens it and quantizes against the "
                "lower-precision scale"
            )


rms_norm_gated_token_major = RmsNormGatedTokenMajor()
