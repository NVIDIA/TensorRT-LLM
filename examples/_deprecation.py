# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Legacy-workflow warnings for the TensorRT engine-build path.

The TensorRT engine-build workflow (convert_checkpoint.py -> trtllm-build ->
run.py) is a legacy path. The PyTorch backend (trtllm-serve / LLM API) is the
recommended approach for new projects.

This module provides a shared warning function that can be called from legacy
scripts to inform users about the recommended migration path.
"""

import warnings

_DEPRECATION_DOCS_URL = "https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html"


def emit_engine_arch_deprecation(script_name: str) -> None:
    """Emit a FutureWarning for legacy engine-architecture scripts.

    Args:
        script_name: The name of the legacy script (e.g., "convert_checkpoint.py").
    """
    warnings.warn(
        f"\n{'=' * 70}\n"
        f"LEGACY WARNING: {script_name}\n"
        f"{'=' * 70}\n"
        f"This script is part of the legacy TensorRT engine-build workflow.\n"
        f"New projects should use the PyTorch backend instead.\n\n"
        f"For new projects, use the PyTorch backend instead:\n\n"
        f"  # Serve a model (recommended):\n"
        f"  trtllm-serve <model_name_or_path>\n\n"
        f"  # Python API:\n"
        f"  from tensorrt_llm import LLM\n"
        f"  llm = LLM(model='<model_name_or_path>')\n"
        f"  output = llm.generate(['Hello, how are you?'])\n\n"
        f"Documentation: {_DEPRECATION_DOCS_URL}\n"
        f"{'=' * 70}\n",
        FutureWarning,
        stacklevel=2,
    )
