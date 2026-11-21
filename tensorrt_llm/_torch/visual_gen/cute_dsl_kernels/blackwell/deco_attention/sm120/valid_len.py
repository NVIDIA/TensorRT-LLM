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
"""Per-sample valid prefix lengths: the kernels of the family read the caller's int32 lengths [B] in place."""

import cutlass
from cutlass import Int32, Int64

from ..common.mem import load_invariant


def valid_len(lens_p, b, s):
    """The valid prefix length of batch element b, clamped to [0, s]: the int32 word b at the global address
    lens_p."""
    n = load_invariant(Int32, Int64(lens_p) + Int64(b) * Int64(4))
    return cutlass.max(cutlass.min(n, Int32(s)), Int32(0))
