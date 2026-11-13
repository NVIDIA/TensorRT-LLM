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

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Shared CuTe DSL helpers for Blackwell prefill MQA logits."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from cutlass import Int32, testing
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op


@dataclass(frozen=True)
class KernelLaunch:
    """A prepared kernel launch reusable for dispatch and timing."""

    compiled_fn: Callable[..., object]
    arguments: testing.JitArguments
    logits: torch.Tensor

    def run(self) -> None:
        """Dispatch the prepared kernel once."""
        self.compiled_fn(*self.arguments.args, **self.arguments.kwargs)


# SMEM b32 load/store wrappers — `cute.arch.ld_shared / st_shared` are not
# exposed in the DSL surface, so we wrap raw PTX.


@dsl_user_op
def ld_shared_b32(smem_ptr, *, loc=None, ip=None) -> Int32:
    # `_Pointer.toint()` returns Int32 (SMEM addr fits in 32 bits).
    smem_addr = smem_ptr.toint(loc=loc, ip=ip)
    return Int32(
        llvm.inline_asm(
            Int32.mlir_type,
            [smem_addr.ir_value(loc=loc, ip=ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_shared_b32(smem_ptr, val: Int32, *, loc=None, ip=None) -> None:
    smem_addr = smem_ptr.toint(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [smem_addr.ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


# Packed f16x2 and bf16x2 arithmetic helpers shared by the epilogues.


def ceil_div(x: int, y: int) -> int:
    """Return ``ceil(x / y)`` for integers."""
    return (x + y - 1) // y


def align(x: int, alignment: int) -> int:
    """Round ``x`` up to a multiple of ``alignment``."""
    return ceil_div(x, alignment) * alignment


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """Return the symmetric normalized difference between two tensors."""
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denominator)
