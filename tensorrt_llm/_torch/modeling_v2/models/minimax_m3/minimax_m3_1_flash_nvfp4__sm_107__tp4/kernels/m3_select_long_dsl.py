# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp4 target's long-row top-16 select: the family ``kernels/m3_select_long_dsl`` traced with this target's
switches.

SELECT_W512 (opt-in, TRTLLM_M31_ENABLE_SELECT_W512=1): 512 threads per row and finer keys-per-thread classes.
SELECT_K13 (opt-in, TRTLLM_M31_ENABLE_SELECT_K13=1; SELECT_W512 wins if both are set): the deployed 384 threads and
classes plus a 13-key class. Read at import; with both off this is the family module itself.
"""

from ...kernels.m3_select_long_dsl import kernel_module
from ..m31_switches import m31_enabled

_KERNEL = kernel_module(
    select_w512=m31_enabled("SELECT_W512"), select_k13=m31_enabled("SELECT_K13")
)

SELECT_W512 = _KERNEL.SELECT_W512
SELECT_K13 = _KERNEL.SELECT_K13
SELECT_THREADS = _KERNEL.SELECT_THREADS
ITEM_CLASSES = _KERNEL.ITEM_CLASSES
MAX_BLOCKS = _KERNEL.MAX_BLOCKS
TOPK = _KERNEL.TOPK
kernel_cache = _KERNEL.kernel_cache
kernel_key = _KERNEL.kernel_key
prebuild = _KERNEL.prebuild
select_long_dsl = _KERNEL.select_long_dsl

__all__ = [
    "ITEM_CLASSES",
    "MAX_BLOCKS",
    "SELECT_K13",
    "SELECT_THREADS",
    "SELECT_W512",
    "TOPK",
    "kernel_cache",
    "kernel_key",
    "prebuild",
    "select_long_dsl",
]
