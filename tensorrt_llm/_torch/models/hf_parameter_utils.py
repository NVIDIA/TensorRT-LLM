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

"""``get_parameter_device`` / ``get_parameter_dtype`` helpers.

Transformers v5 no longer exports these from ``transformers.modeling_utils``; these
replacements match ``ModuleUtilsMixin`` behavior for plain ``nn.Module`` stacks.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def get_parameter_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def get_parameter_dtype(module: nn.Module) -> torch.dtype:
    return next(param.dtype for param in module.parameters() if param.is_floating_point())
