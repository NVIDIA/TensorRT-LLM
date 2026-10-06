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
"""The ``TRTLLMGenFusedMoE`` family name, kept importable from its own path.

The implementation lives in the :mod:`.trtllm_gen` subpackage; the lookup
helpers are forwarded so that this path resolves everything the pre-split one
did.

Every gate against this name has to be ``issubclass`` / ``isinstance`` and not
an equality check: the leaves are what resolution hands over, so
``type(x) is TRTLLMGenFusedMoE`` matches nothing.
"""

from .trtllm_gen import (TrtllmGenFusedMoEBase, find_trtllm_gen_leaf,
                         trtllm_gen_leaf)

# An alias, not a base class, so there is no second class to keep in step.
TRTLLMGenFusedMoE = TrtllmGenFusedMoEBase

__all__ = ["TRTLLMGenFusedMoE", "trtllm_gen_leaf", "find_trtllm_gen_leaf"]
