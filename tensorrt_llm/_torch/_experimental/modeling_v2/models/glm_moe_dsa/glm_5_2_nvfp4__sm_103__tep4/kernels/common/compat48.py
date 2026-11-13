# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cluster primitives the GLM kernels use that cutlass-dsl 4.8 does not wrap."""

import cutlass
import cutlass.experimental.primitives.descriptors as _d
from cutlass._mlir.dialects import nvvm as _nvvm_raw
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental.primitives.nvvm_wrapper import _to_ir


@dsl_user_op
def store_async(addr, value, mbarrier, *, loc=None, ip=None) -> None:
    """``st.async.shared::cluster.mbarrier::complete_tx::bytes`` to a peer CTA."""
    _nvvm_raw.store_async(
        _to_ir(addr, loc, ip),
        _to_ir(value, loc, ip),
        _to_ir(mbarrier, loc, ip),
        loc=loc,
        ip=ip,
    ).operation.verify()


@dsl_user_op
def tcgen05_smem_desc(
    start_address,
    leading_byte_offset: int = 0,
    stride_byte_offset: int = 0,
    version: int = 1,
    base_offset: int = 0,
    layout=0,
    leading_dim_mode: int = 0,
    k_segment_offset: int = 0,
    *,
    loc=None,
    ip=None,
) -> _d.Tcgen05SmemDesc:
    """A ``tcgen05.mma`` SMEM descriptor carrying the descriptor-version and
    K-segment fields, built on the ``tcgen05_mma_smem_desc_v2`` op.

    Byte offsets as in ``Tcgen05SmemDesc.build``: pass raw byte addresses and
    offsets; the 16-byte granule shift happens here.
    """
    if not isinstance(layout, (int, _d.Numeric)) and hasattr(layout, "to"):
        layout = layout.to(_d.Tcgen05SmemSwizzle)
    swizzle = cutlass.Int8(layout) if isinstance(layout, _d.Numeric) else cutlass.Int8(int(layout))
    if isinstance(start_address, (_d.Array, cutlass.Pointer)):
        addr = cutlass.Int32(_d.llvm.ptrtoint(cutlass.Int32.mlir_type, start_address.ir_value()))
    else:
        addr = cutlass.Int32(start_address)
    lbo = cutlass.Int32(_d._drop_low_bits(leading_byte_offset, 4, "leading_byte_offset"))
    sbo = cutlass.Int32(_d._drop_low_bits(stride_byte_offset, 4, "stride_byte_offset"))
    return _d.Tcgen05SmemDesc(
        cutlass.Int64(
            _d._nvvm.tcgen05_mma_smem_desc_v2(
                addr >> 4,
                lbo,
                sbo,
                cutlass.Int8(version),
                cutlass.Int8(base_offset),
                cutlass.Boolean(leading_dim_mode),
                cutlass.Boolean(k_segment_offset),
                swizzle,
                loc=loc,
                ip=ip,
            )
        )
    )
