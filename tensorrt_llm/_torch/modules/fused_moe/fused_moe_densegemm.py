# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selector for DenseGEMM MoE SM-partition strategy.

Controls which parallel strategy is used when ``TRTLLM_MOE_FUSED_FC2_ALPHA=0``
(i.e. the non-fused path where FC1 and Router GEMM run concurrently):

- ``gc``         (default) – Green Context: hardware-level SM pool isolation via
  ``torch.cuda.GreenContext``.  Tuned by ``TRTLLM_MOE_FC1_SM_NUNBER`` (default 0.5).
- ``smp``        – SM Partition: soft SM limit via the ``sm_budget`` kernel launch
  parameter.  Tuned by ``TRTLLM_MOE_FC1_SM_NUNBER`` (default 0.5).
- ``no_overlap`` – Single stream, no SM partitioning.  All kernels use all available
  SMs and run sequentially.  Useful as a performance baseline.

Set ``TRTLLM_MOE_SM_SPLIT_MODE`` to one of the above values to select the backend.
"""

import os

_mode = os.environ.get("TRTLLM_MOE_SM_SPLIT_MODE", "gc").strip().lower()

if _mode == "smp":
    from .fused_moe_densegemm_smp import DenseGEMMFusedMoE
elif _mode == "no_overlap":
    from .fused_moe_densegemm_no_overlap import NoOverlapDenseGEMMFusedMoE as DenseGEMMFusedMoE
else:
    if _mode != "gc":
        import warnings

        warnings.warn(
            f"Unknown TRTLLM_MOE_SM_SPLIT_MODE='{_mode}', falling back to 'gc'.",
            stacklevel=1,
        )
    from .fused_moe_densegemm_gc import DenseGEMMFusedMoE

__all__ = ["DenseGEMMFusedMoE"]
