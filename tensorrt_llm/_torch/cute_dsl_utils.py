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

import platform
import sys
import types

from ..logger import logger


def _skip_legacy_cutlass_mlir_helpers() -> None:
    """Keep Cutlass version discovery from importing its legacy helper tree."""
    legacy_name = "cutlass.base_dsl._mlir_helpers"
    if legacy_name in sys.modules:
        return

    # Cutlass uses pkgutil.walk_packages to hash its sources. The internal
    # package also ships this unused legacy helper tree alongside the canonical
    # cutlass._mlir_helpers package. Prevent pkgutil from descending into the
    # legacy tree, which would register the same MLIR value casters twice.
    legacy_module = types.ModuleType(legacy_name)
    legacy_module.__path__ = []
    sys.modules[legacy_name] = legacy_module


IS_CUTLASS_DSL_AVAILABLE = False
# TODO: Remove IS_CUTLASS_DSL_INTERNAL_AVAILABLE once rubin_helpers is available
# in the base nvidia-cutlass-dsl package (currently only in nvidia-cutlass-dsl-internal)
IS_CUTLASS_DSL_INTERNAL_AVAILABLE = False

if platform.system() != "Windows":
    try:
        from cutlass import cute  # noqa
        _skip_legacy_cutlass_mlir_helpers()
        logger.info(f"cutlass dsl is available")
        IS_CUTLASS_DSL_AVAILABLE = True

        # Check for internal cutlass DSL package (has rubin_helpers for SM107)
        try:
            import cutlass.utils.rubin_helpers  # noqa
            logger.info(f"cutlass dsl internal (rubin) is available")
            IS_CUTLASS_DSL_INTERNAL_AVAILABLE = True
        except ImportError:
            pass
    except ImportError:
        pass
