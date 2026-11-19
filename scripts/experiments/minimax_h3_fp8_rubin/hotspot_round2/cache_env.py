# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Writable task-local caches for the host UID inside the container."""

import os
from pathlib import Path

for key, value in {
    "XDG_CACHE_HOME": "/tmp/h3-cache",
    "HF_HOME": "/tmp/h3-cache/huggingface",
    "FLASHINFER_WORKSPACE_BASE": "/tmp/h3-cache/flashinfer",
    "TORCHINDUCTOR_CACHE_DIR": "/tmp/h3-cache/inductor",
    "TORCH_EXTENSIONS_DIR": "/tmp/h3-cache/extensions",
    "TRITON_CACHE_DIR": "/tmp/h3-cache/triton",
    "TORCH_HOME": "/tmp/h3-cache/torch",
    "CUDA_CACHE_PATH": "/tmp/h3-cache/cuda",
}.items():
    os.environ.setdefault(key, value)
    Path(value).mkdir(parents=True, exist_ok=True)
