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
"""Backend selection without importing native extensions or initializing a GPU."""

import os
from typing import Literal


def select_backend(hip_version: str | None) -> Literal["cuda", "rocm"]:
    """Select ROCm for a HIP PyTorch build, or honor an explicit backend choice.

    ``TRTLLM_BACKEND=rocm`` also permits importing the portable API with a CPU
    PyTorch build. Execution on CPU still requires explicitly passing ``device='cpu'``.
    """
    choice = os.environ.get("TRTLLM_BACKEND", "auto").lower()
    if choice not in ("auto", "cuda", "rocm"):
        raise ValueError("TRTLLM_BACKEND must be auto, cuda, or rocm")
    if choice == "cuda" and hip_version:
        raise ValueError("A HIP PyTorch build cannot load the NVIDIA CUDA backend")
    return "rocm" if choice == "rocm" or (choice == "auto" and hip_version) else "cuda"


__all__ = ["select_backend"]
