# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Stream dependencies for kernels that share candidate and recovery storage."""
import cutlass
import cutlass.cute as cute


@cute.jit
def pdl_prologue(enabled: cutlass.Constexpr):
    if cutlass.const_expr(enabled):
        # Every thread waits before global reads, writes or arena reuse.
        cute.arch.griddepcontrol_wait()
        # This permits launch overlap, not early access to this kernel's data.
        # The dependent kernel must still wait for completion and visibility.
        cute.arch.griddepcontrol_launch_dependents()
