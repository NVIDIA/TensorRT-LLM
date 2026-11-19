# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The experimental primitives of the CuTe DSL that the kernels call (cutlass.experimental.primitives). This module
alone imports them, each by name, and the kernels import them from here: the experimental interface may change
between releases of the DSL, and a release without one of the names fails here, on import, with the installed
version and the validated ones."""

import cutlass

# the releases of nvidia-cutlass-dsl the kernels are validated with
VALIDATED_DSL_VERSIONS = ("4.8.0", "4.8.1")

try:
    from cutlass.experimental.primitives import (
        BarrierRedux,
        CTAGroup,
        FPRoundingMode,
        LoadCacheModifier,
        MMALayout,
        ReductionKind,
        SaturationMode,
        Tcgen05CpShape,
        Tcgen05Fence,
        Tcgen05LdStShape,
        Tcgen05Wait,
        barrier_cta_red,
        barrier_cta_sync,
        convert,
        convert_and_pack_integer,
        cp_async_shared_global,
        cvt_f32x2_to_f8x2,
        ldmatrix,
        make_tmem_ptr,
        nanosleep,
        tcgen05_cp,
        tcgen05_fence,
        tcgen05_ld,
        tcgen05_ld_red,
        tcgen05_st,
        tcgen05_wait,
    )
except ImportError as error:
    raise ImportError(
        f"deco_attention needs a primitive that the installed nvidia-cutlass-dsl "
        f"{getattr(cutlass, '__version__', '(unknown version)')} does not provide ({error}); "
        f"the validated versions are {' and '.join(VALIDATED_DSL_VERSIONS)}"
    ) from error

__all__ = [
    "BarrierRedux",
    "CTAGroup",
    "FPRoundingMode",
    "LoadCacheModifier",
    "MMALayout",
    "ReductionKind",
    "SaturationMode",
    "Tcgen05CpShape",
    "Tcgen05Fence",
    "Tcgen05LdStShape",
    "Tcgen05Wait",
    "barrier_cta_red",
    "barrier_cta_sync",
    "convert",
    "convert_and_pack_integer",
    "cp_async_shared_global",
    "cvt_f32x2_to_f8x2",
    "ldmatrix",
    "make_tmem_ptr",
    "nanosleep",
    "tcgen05_cp",
    "tcgen05_fence",
    "tcgen05_ld",
    "tcgen05_ld_red",
    "tcgen05_st",
    "tcgen05_wait",
]
