# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restore the exact compiled argument names for loaded MegaMoE functions."""

from inspect import signature

_SCALE_ARGUMENTS = frozenset(("fc1_alpha", "fc2_alpha", "fc1_norm_const"))
_READY_ARGUMENTS = frozenset(
    ("hot_expert_weight_ready_flags", "hot_expert_weight_ready_generation")
)


def megamoe_aot_argument_names(kernel):
    """Return the runtime arguments present in this kernel variant AOT ABI."""
    uses_global_scale = kernel.quant_kind.uses_global_scale
    helper_enabled = kernel.helper_expert_count > 0
    return [
        name
        for name in signature(kernel.__call__).parameters
        if (uses_global_scale or name not in _SCALE_ARGUMENTS)
        and (helper_enabled or name not in _READY_ARGUMENTS)
    ]


def make_megamoe_aot_callable(function, kernel):
    """Use TVM-FFI's native adapter without attaching scheduler metadata."""
    from tvm_ffi.utils.kwargs_wrapper import make_kwargs_wrapper

    return make_kwargs_wrapper(function, megamoe_aot_argument_names(kernel))
