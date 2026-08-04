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

import os
import tempfile
from pathlib import Path

_FLASHINFER_WORKSPACE_ENV = "FLASHINFER_WORKSPACE_BASE"
_FLASHINFER_WORKSPACE_ISOLATION_ENV = "TRTLLM_FLASHINFER_WORKSPACE_PER_PROCESS"


def _configure_flashinfer_workspace() -> None:
    """Give an opted-in process a private FlashInfer JIT workspace.

    FlashInfer reads ``FLASHINFER_WORKSPACE_BASE`` during import. This helper
    therefore runs at the start of the TensorRT-LLM package import, before any
    module can import FlashInfer. An explicit workspace always takes precedence.
    """
    if os.environ.get(_FLASHINFER_WORKSPACE_ISOLATION_ENV) != "1":
        return
    if _FLASHINFER_WORKSPACE_ENV in os.environ:
        return

    get_user_id = getattr(os, "getuid", lambda: 0)
    workspace = Path(tempfile.gettempdir()) / (f"trtllm-flashinfer-{get_user_id()}-{os.getpid()}")
    os.environ[_FLASHINFER_WORKSPACE_ENV] = str(workspace)
