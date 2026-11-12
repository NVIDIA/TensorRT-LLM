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
"""Exact sample/filter/recover MQA top-k for SM100."""
import importlib

try:
    memory = importlib.import_module("cutlass.memory")
except ModuleNotFoundError as error:
    if error.name != "cutlass.memory":
        raise
    # CuTe DSL 4.7 exposes these allocators under utils.
    from cutlass import utils as memory
try:
    tensor_utils = importlib.import_module("cutlass.tensor_utils")
except ModuleNotFoundError as error:
    if error.name != "cutlass.tensor_utils":
        raise
    from cutlass import utils as tensor_utils
