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
"""The ``PrimsTsFusedMoE`` family name; the implementation is the :mod:`.prims_ts` package.

Every gate against this name has to be ``issubclass`` / ``isinstance`` and not
an equality check: the leaves are what resolution hands over, so
``type(x) is PrimsTsFusedMoE`` matches nothing.
"""

from .prims_ts import PrimsTsFusedMoEBase, find_prims_ts_leaf, prims_ts_leaf

# An alias, not a base class, so there is no second class to keep in step.
PrimsTsFusedMoE = PrimsTsFusedMoEBase

__all__ = ["PrimsTsFusedMoE", "find_prims_ts_leaf", "prims_ts_leaf"]
