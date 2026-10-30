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
"""``CutlassFusedMoE``, the CUTLASS family name: :class:`.cutlass.CutlassFusedMoEBase`.

The implementation lives in the :mod:`.cutlass` subpackage. The base is
abstract and is the parent of every CUTLASS leaf (and of the B12x leaves, which
reuse its construction), so gate on it with ``issubclass`` / ``isinstance``,
never ``type(x) is CutlassFusedMoE``.
"""

from .cutlass import CutlassFusedMoEBase, raise_moe_lora_multichunk_unsupported

CutlassFusedMoE = CutlassFusedMoEBase

__all__ = [
    "CutlassFusedMoE",
    "raise_moe_lora_multichunk_unsupported",
]
