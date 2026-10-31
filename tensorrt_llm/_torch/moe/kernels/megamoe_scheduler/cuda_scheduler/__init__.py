# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-CUDA physical-slot scheduler.

HALO-M hierarchical placement followed by HALO-Q quota repair, behind one
four-tensor placement ABI. Copy routing belongs to the in-switch copy layer.  The CPU oracle that states the same placement lives in
``research/halo_q_oracle.py`` and is not imported from here.
"""

from .runtime import (
    CudaPhysicalSlotScheduler,
    CudaSchedulerConfig,
    CudaSchedulerOutputs,
    recommend_cuda_scheduler_ctas,
)

__all__ = [
    "CudaPhysicalSlotScheduler",
    "CudaSchedulerConfig",
    "CudaSchedulerOutputs",
    "recommend_cuda_scheduler_ctas",
]
