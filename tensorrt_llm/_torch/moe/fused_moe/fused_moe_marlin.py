# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""The ``MarlinFusedMoE`` family name, kept importable from its own path.

The implementation lives in the :mod:`.marlin` subpackage; the lookup helpers
are forwarded so that this path resolves everything the pre-split one did.

Every gate against this name has to be ``issubclass`` / ``isinstance`` and not
an equality check: the leaves are what resolution hands over, so
``type(x) is MarlinFusedMoE`` matches nothing.
"""

from .marlin import MarlinFusedMoEBase, find_marlin_leaf, marlin_leaf

# An alias, not a base class, so there is no second class to keep in step.
MarlinFusedMoE = MarlinFusedMoEBase

__all__ = ["MarlinFusedMoE", "marlin_leaf", "find_marlin_leaf"]
