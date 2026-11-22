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
"""Compile targets and the hardware capabilities the kernels branch on at trace time.

The target is the architecture the running compilation produces code for: CUTE_DSL_ARCH when set (offline builds),
else that of CUDA device 0. Kernels never test architecture names directly; they ask for a capability, so a further
target is added by its row in the tables below and an implementation of each capability that differs. sm_107a runs
the same kernels with the capabilities of the tables below. sm_120a (RTX PRO 6000) kernels use warp-level MMAs and
form their own family (deco_attention/sm120), which kernel_family() names. The tables name the architecture-specific
targets (sm_100a, sm_103a, sm_107a, sm_120a), those the kernels are built and validated for.
"""

import functools

from cutlass import Arch
from cutlass.cutlass_dsl import BaseDSL


def _archs(*names):
    """The architectures of these names that the installed CuTe DSL defines."""
    return frozenset(getattr(Arch, n) for n in names if hasattr(Arch, n))


# kind::mxf4nvf4 MMA with K = 96
_K96_FP4_MMA = _archs("sm_103a")
# kind::mxf4nvf4 MMA with K = 128 (SM107MmaMXF4NVF4Op): the 128 head channels of the QK in one instruction
_K128_FP4_MMA = _archs("sm_107a")
# kind::f8f6f4 MMA with K = 64 (SM107MmaFP8Op)
_K64_FP8_MMA = _archs("sm_107a")
# tcgen05.ld.red: a TMEM load that also returns the max (or max |x|) of the loaded values
_TMEM_LD_RED = _archs("sm_103a", "sm_107a")
# redux.sync on f32: the maximum of a value over the lanes of a warp in one instruction
_REDUX_F32 = _archs("sm_100a", "sm_103a", "sm_107a")
# the tcgen05 targets -> the family name of CuTe's TMEM helpers
_NAMES = {a: family for family in ("sm_100", "sm_103", "sm_107") for a in _archs(family + "a")}
# the targets with DeCo kernels -> their kernel family (deco_attention/<family>): the tcgen05 targets, and the
# warp-level MMA targets (mma.sync, block-scaled FP4 and FP8)
_FAMILY = {**{a: "tcgen05" for a in _NAMES}, **{a: "sm120" for a in _archs("sm_120a")}}


@functools.cache
def arch():
    """The architecture enum of the current compilation (NotImplementedError for a target without DeCo kernels: the
    one error a caller may answer with another implementation); cached, as the target is fixed per process."""
    a = BaseDSL._get_dsl().get_arch_enum()
    if a not in _FAMILY:
        raise NotImplementedError(f"no DeCo attention kernels for {a}")
    return a


@functools.cache
def kernel_family():
    """The kernel family of the current target, deco_attention/<family>: "tcgen05" (B200, GB300, Rubin) or "sm120"
    (warp-level MMAs: RTX PRO 6000); cached, as the target is fixed per process."""
    return _FAMILY[arch()]


def arch_name():
    """The family name CuTe's TMEM helpers take ("sm_100", "sm_103", "sm_107"); tcgen05 targets only."""
    return _NAMES[arch()]


def has_k96_fp4_mma():
    return arch() in _K96_FP4_MMA


def has_k128_fp4_mma():
    return arch() in _K128_FP4_MMA


def has_k64_fp8_mma():
    return arch() in _K64_FP8_MMA


def has_tmem_ld_red():
    return arch() in _TMEM_LD_RED


def has_redux_f32():
    return arch() in _REDUX_F32
