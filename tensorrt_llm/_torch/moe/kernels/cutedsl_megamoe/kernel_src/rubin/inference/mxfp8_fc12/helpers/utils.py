# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Device helpers of the Rubin MXFP8 FC12 kernel.

TensorRT-LLM-local file, not part of the upstream cutedsl_megamoe export; it serves the
kernel adapted from the upstream training forward GLU kernel
(``sources/kernel_src/rubin/training/mega/fwd_glu/``). A re-export must carry it forward.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import math as _math
from cutlass.cutlass_dsl import Float32

from ......helpers.constants import Fp8E4M3RcpLimit, Fp8E5M2RcpLimit, Fp32Max, Log2E
from ......helpers.ptx_helpers import (
    cvt_f32_to_fp8_to_f32,
    cvt_f32_to_ue8m0_raw_i32,
    cvt_f32x4_to_f8x4_pack_i32,
    cvt_f32x4_to_f8x4_scaled_pack_i32,
)


@cute.jit
def swiglu_act(
    t_swiglu: cute.Tensor,
    t_up: cute.Tensor,
    t_gate: cute.Tensor,
    prob: Optional[Float32] = None,
    gate_up_clamp: Optional[Float32] = None,
) -> None:
    """SwiGLU with optional gate-upper/up-symmetric clamp."""
    for i in cutlass.range_constexpr(0, cute.size(t_swiglu), 2):
        gate = (t_gate[i], t_gate[i + 1])
        up = (t_up[i], t_up[i + 1])
        if cutlass.const_expr(gate_up_clamp is not None):
            gate = (
                cute.arch.fmin(gate[0], gate_up_clamp),
                cute.arch.fmin(gate[1], gate_up_clamp),
            )
            up = (
                cute.arch.fmax(cute.arch.fmin(up[0], gate_up_clamp), -gate_up_clamp),
                cute.arch.fmax(cute.arch.fmin(up[1], gate_up_clamp), -gate_up_clamp),
            )
        up_gate = cute.arch.mul_packed_f32x2(
            up,
            gate,
            rnd="rn",
            ftz=False,
        )
        gate_log2e = cute.arch.mul_packed_f32x2(
            gate,
            (-Log2E, -Log2E),
            rnd="rn",
            ftz=False,
        )
        one_plus_exp = cute.arch.add_packed_f32x2(
            (
                cute.math.exp2(gate_log2e[0], fastmath=True),
                cute.math.exp2(gate_log2e[1], fastmath=True),
            ),
            (1.0, 1.0),
        )
        sigmoid = (cute.arch.rcp_approx(one_plus_exp[0]), cute.arch.rcp_approx(one_plus_exp[1]))
        (t_swiglu[i], t_swiglu[i + 1]) = cute.arch.mul_packed_f32x2(
            up_gate,
            sigmoid,
            rnd="rn",
            ftz=False,
        )
        if cutlass.const_expr(prob is not None):
            (t_swiglu[i], t_swiglu[i + 1]) = cute.arch.mul_packed_f32x2(
                (t_swiglu[i], t_swiglu[i + 1]),
                (prob, prob),
                rnd="rn",
                ftz=False,
            )


@cute.jit
def quant_sfd_row(
    src: cute.Tensor,
    dst: cute.Tensor,
    norm_const,
    sf_vec_size,
    sf_dtype,
    d_dtype,
    use_scaled_cvt: bool = False,
):
    """Quantize the ``sf_vec_size`` values in ``src`` to ``d_dtype`` with one block scale."""
    rcp_limit = Fp8E4M3RcpLimit if d_dtype == cutlass.Float8E4M3FN else Fp8E5M2RcpLimit
    acc_frg = src.load()
    abs_acc_frg_ir = _math.absf(acc_frg.ir_value())
    abs_acc_frg = type(acc_frg)(abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype)
    # Fuse the two loop-invariant constants into one multiply
    rcp_limit_norm = rcp_limit * norm_const
    avg_fp32 = abs_acc_frg.reduce(cute.ReductionOp.MAX, Float32(0.0), 0) * rcp_limit_norm

    if cutlass.const_expr(use_scaled_cvt):
        if cutlass.const_expr(sf_dtype is not cutlass.Float8E8M0FNU):
            raise ValueError("Scaled FP8 conversion requires UE8M0 scale factors.")

        qpvscale_raw = cvt_f32_to_ue8m0_raw_i32(avg_fp32)

        # Preserve the general St contract. Production MXFP8 passes 1.0, so
        # this multiply constant-folds away in the intended path.
        for ei in cutlass.range_constexpr(0, sf_vec_size, 2):
            src[ei], src[ei + 1] = cute.arch.mul_packed_f32x2(
                (src[ei], src[ei + 1]),
                (norm_const, norm_const),
                rnd="rn",
                ftz=False,
            )

        dst_i32 = cute.recast_tensor(dst, cutlass.Int32)
        acc_vec = src.load()
        for ei in cutlass.range_constexpr(0, sf_vec_size, 4):
            fp32x4 = cute.make_rmem_tensor(4, Float32)
            fp32x4[0] = acc_vec[ei + 0]
            fp32x4[1] = acc_vec[ei + 1]
            fp32x4[2] = acc_vec[ei + 2]
            fp32x4[3] = acc_vec[ei + 3]
            dst_i32[ei // 4] = cvt_f32x4_to_f8x4_scaled_pack_i32(fp32x4, d_dtype, qpvscale_raw)

        # The caller stores this to E8M0; the pre-round value produces the
        # same byte without a UE8M0 -> FP32 round trip.
        return avg_fp32

    qpvscale_up = cvt_f32_to_fp8_to_f32(avg_fp32, sf_dtype)
    acc_scale = norm_const * cute.arch.rcp_approx(qpvscale_up)
    acc_scale = cute.arch.fmin(acc_scale, Fp32Max, nan=True)
    for ei in cutlass.range_constexpr(0, sf_vec_size, 2):
        src[ei], src[ei + 1] = cute.arch.mul_packed_f32x2(
            (src[ei], src[ei + 1]),
            (acc_scale, acc_scale),
            rnd="rn",
            ftz=False,
        )
    dst_i32 = cute.recast_tensor(dst, cutlass.Int32)
    for ei in cutlass.range_constexpr(0, sf_vec_size, 4):
        fp32x4 = cute.make_rmem_tensor(4, Float32)
        fp32x4[0] = src[ei + 0]
        fp32x4[1] = src[ei + 1]
        fp32x4[2] = src[ei + 2]
        fp32x4[3] = src[ei + 3]
        fp8x4_i32 = cvt_f32x4_to_f8x4_pack_i32(fp32x4, d_dtype)
        dst_i32[ei // 4] = cutlass.Int32(fp8x4_i32)
    return qpvscale_up


__all__ = ["quant_sfd_row", "swiglu_act"]
