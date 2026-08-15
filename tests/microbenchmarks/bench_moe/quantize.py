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

"""Quantization helper loader for ``bench_moe``.

The benchmark reuses unittest quantization fixtures for MoE weight generation,
but those fixtures import ``moe_test_utils`` only for a memory skip helper. That
top-level import eagerly loads every backend and can fail when optional CUTEDSL
dependencies are unavailable. This loader provides the one skip helper while
importing the fixture module, then restores the normal module table.
"""

from __future__ import annotations

import importlib
import sys
import types

from .backend import ensure_cute_dsl_importable_for_benchmark


def _make_moe_test_utils_stub() -> types.ModuleType:
    stub = types.ModuleType("_torch.modules.moe.moe_test_utils")

    def skip_if_insufficient_gpu_memory(*_args, **_kwargs) -> None:
        return None

    stub.skip_if_insufficient_gpu_memory = skip_if_insufficient_gpu_memory
    return stub


def _load_get_test_quant_params():
    ensure_cute_dsl_importable_for_benchmark()
    dependency_name = "_torch.modules.moe.moe_test_utils"
    original_dependency = sys.modules.get(dependency_name)
    inserted_stub = original_dependency is None
    if inserted_stub:
        sys.modules[dependency_name] = _make_moe_test_utils_stub()
    try:
        quantize_utils = importlib.import_module("_torch.modules.moe.quantize_utils")
    finally:
        if inserted_stub:
            sys.modules.pop(dependency_name, None)
        elif original_dependency is not None:
            sys.modules[dependency_name] = original_dependency
    return quantize_utils.get_test_quant_params


get_test_quant_params = _load_get_test_quant_params()
