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
