# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Architecture-aware tcgen05 shared-memory descriptor construction."""

from typing import Any

import cutlass
from cutlass.experimental import primitives as prims
from cutlass.experimental.primitives.nvvm_wrapper import (
    _tcgen05_mma_smem_desc_v2,
)

from .batched_gemm_config import BatchedGemmConfig


def _offset_in_16b_units(value: int, name: str) -> int:
    if value % 16 != 0:
        raise ValueError(f"{name} must be a multiple of 16 bytes, got {value}")
    return value >> 4


@cutlass.dsl_user_op
def build_tcgen05_smem_desc(
    cfg: BatchedGemmConfig,
    start_address: Any,
    *,
    leading_byte_offset: int = 0,
    stride_byte_offset: int = 0,
    base_offset: int = 0,
    layout: int = 0,
    leading_dim_mode: int = 0,
    loc=None,
    ip=None,
):
    """Build the legacy SMEM descriptor or Rubin's extended v2 descriptor.

    Rubin exposes a fifteenth encoded start-address bit.  The v2 intrinsic is
    required to preserve that bit and descriptor version 1 when an operand is
    placed at or above the 256 KiB boundary.
    """
    if not cfg.is_rubin:
        return prims.Tcgen05SmemDesc.build(
            start_address,
            leading_byte_offset=leading_byte_offset,
            stride_byte_offset=stride_byte_offset,
            base_offset=base_offset,
            layout=layout,
            leading_dim_mode=leading_dim_mode,
        )

    address = start_address
    if hasattr(address, "data_ptr"):
        address = address.data_ptr()
    if hasattr(address, "toint"):
        address = address.toint()
    address = cutlass.Int32(address)

    return prims.Tcgen05SmemDesc(
        _tcgen05_mma_smem_desc_v2(
            address >> 4,
            _offset_in_16b_units(leading_byte_offset, "leading_byte_offset"),
            _offset_in_16b_units(stride_byte_offset, "stride_byte_offset"),
            1,  # SM107 descriptor version
            base_offset,
            leading_dim_mode,
            0,  # first K segment
            layout,
            loc=loc,
            ip=ip,
        )
    )


__all__ = ["build_tcgen05_smem_desc"]
