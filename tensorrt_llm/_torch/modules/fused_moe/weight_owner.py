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
"""The single gate that decides whether a module owns MoE expert weights."""

import torch.nn as nn

from .impl_base import MoEImplBase
from .interface import MoE

# Modules that own MoE expert weights and therefore need MoE-specific renaming
# (HF ``gate_proj``/``up_proj``/``down_proj`` -> ``w1``/``w3``/``w2``, plus the
# Transformers 5.x unfuse) and path rewriting (dropping the trailing
# ``.backend``, which checkpoint keys do not carry) during weight load.
#
# Both entries are required because the two roles are converging, not merged:
# ``MoE`` is a complete layer that also owns weights, ``MoEImplBase`` is an
# execution unit that owns weights and nothing else. Backends are moving from
# the first to the second one class at a time, and a gate that names only the
# class a backend used to have would stop recognising it *silently* -- the
# weights would simply not be loaded, with no exception and a model that still
# constructs and still runs on random expert weights.
_MOE_WEIGHT_OWNER_TYPES = (MoE, MoEImplBase)


def is_moe_weight_owner(module: nn.Module) -> bool:
    """Whether ``module`` holds the expert weights of a MoE layer.

    Deliberately a type test rather than an attribute probe: a
    ``hasattr(module, "w3_w1_weight")`` gate trades coupling to a class for
    coupling to a field name and fails just as silently when the weight layout
    changes (``modules/dwdp/setup.py`` has one such probe today).
    """
    return isinstance(module, _MOE_WEIGHT_OWNER_TYPES)
