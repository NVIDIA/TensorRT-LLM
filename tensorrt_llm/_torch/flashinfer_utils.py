# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import platform
import traceback

from .._utils import get_sm_version
from ..logger import logger

IS_FLASHINFER_AVAILABLE = False


def is_pdl_enabled() -> bool:
    """Return whether PDL is requested and supported by the current GPU."""
    env_value = os.environ.get("TRTLLM_ENABLE_PDL")
    if env_value not in (None, "1"):
        return False

    sm_version = get_sm_version()
    enabled = sm_version >= 90
    if not enabled and env_value == "1":
        detected_device = "no CUDA GPU" if sm_version < 0 else f"SM{sm_version}"
        raise ValueError(
            "TRTLLM_ENABLE_PDL=1 requires SM90 or newer, "
            f"but detected {detected_device}. Unset TRTLLM_ENABLE_PDL to use "
            "the architecture-aware default, or set it to 0.")

    if not getattr(is_pdl_enabled, "_printed", False):
        if enabled:
            logger.info("PDL enabled")
        elif sm_version < 0:
            logger.info("PDL disabled: no CUDA GPU is available")
        else:
            logger.info(
                f"PDL disabled on SM{sm_version}: requires SM90 or newer")
        setattr(is_pdl_enabled, "_printed", True)
    return enabled


if platform.system() != "Windows":
    try:
        import flashinfer
        logger.info(f"flashinfer is available: {flashinfer.__version__}")
        IS_FLASHINFER_AVAILABLE = True
    except ImportError:
        traceback.print_exc()
        print(
            "flashinfer is not installed properly, please try pip install or building from source codes"
        )
