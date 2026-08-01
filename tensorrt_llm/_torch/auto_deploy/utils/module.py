# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Common utils for nn.Module."""

from typing import Tuple

import torch.nn as nn


def get_submodule_of_param(gm: nn.Module, param_name: str) -> Tuple[nn.Module, str, str]:
    # Returns (module, module_path, attr_name)
    if "." not in param_name:
        # param on the root
        return gm, "", param_name
    mod_path, _, attr = param_name.rpartition(".")
    return gm.get_submodule(mod_path), mod_path, attr
