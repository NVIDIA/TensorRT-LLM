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

"""Custom ops and make sure they are all registered."""

import importlib
import pkgutil

from ..utils.logger import ad_logger

__all__ = []

from .._compat import TRTLLM_AVAILABLE

# Optional packages whose absence should NOT crash auto_deploy import.
# tensorrt_llm: core optional — standalone mode runs without it.
# triton_kernels: internal NVIDIA package for specialized MoE kernels.
_OPTIONAL_PACKAGES = frozenset({"tensorrt_llm", "triton_kernels"})


def _is_trtllm_import_error(exc: BaseException) -> bool:
    """Check if an exception is caused by a missing/broken tensorrt_llm dependency.

    Covers two cases:
    - ModuleNotFoundError where the missing module is tensorrt_llm (or a submodule)
    - ImportError from C++ binding load failures (e.g., missing libmpi.so) when
      tensorrt_llm is partially installed but its native extensions can't load
    """
    if isinstance(exc, ModuleNotFoundError):
        missing_top = exc.name.split(".")[0] if exc.name else ""
        return missing_top in _OPTIONAL_PACKAGES
    if isinstance(exc, ImportError) and not TRTLLM_AVAILABLE:
        # In standalone mode, any ImportError from a tensorrt_llm module is expected
        return True
    return False


# Recursively import subpackages and modules so their side-effects (e.g.,
# op registrations) are applied even when nested in subdirectories.
for _, full_name, _ in pkgutil.walk_packages(__path__, prefix=f"{__name__}."):
    try:
        importlib.import_module(full_name)
        __all__.append(full_name)
    except (ModuleNotFoundError, ImportError) as e:
        if _is_trtllm_import_error(e):
            ad_logger.debug(
                "Skipping %s (tensorrt_llm not fully available): %s",
                full_name,
                e,
            )
        else:
            raise  # Required package missing or real bug — propagate
