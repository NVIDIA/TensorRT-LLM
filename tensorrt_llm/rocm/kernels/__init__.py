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
"""Lazy, explicit loading of native RDNA4 wave32 HIP kernels.

No compiler is invoked at package import or with kernels='torch'. An explicit
HIP kernel request builds on the user's ROCm host, not during wheel packaging.
"""

import threading
from pathlib import Path

import torch

from ..runtime import resolve_device

_lock = threading.Lock()
_loaded = False


def load_kernels(device: torch.device, verbose: bool = False) -> None:
    """Build/load the registered gfx1200/gfx1201 operators once per process."""
    global _loaded
    if resolve_device(device).type != "cuda":
        raise ValueError("Native HIP kernels require an RDNA4 GPU")
    with _lock:
        if _loaded:
            return
        if hasattr(torch.ops.trtllm_rdna4, "rms_norm"):
            _loaded = True
            return
        from torch.utils.cpp_extension import ROCM_HOME, load

        if ROCM_HOME is None:
            raise RuntimeError(
                "The ROCm SDK/hipcc is required for kernels='hip'; set ROCM_HOME or use kernels='torch'"
            )
        source = Path(__file__).parent / "csrc"
        load(
            name="trtllm_rdna4_ops_v1",
            sources=[str(source / "register.cpp"), str(source / "rdna4Ops.hip")],
            extra_include_paths=[str(source)],
            extra_cflags=["-O2", "-std=c++17"],
            extra_cuda_cflags=[
                "-O2",
                "-std=c++17",
                "-fno-fast-math",
                "-mwavefrontsize32",
                "--offload-arch=gfx1200",
                "--offload-arch=gfx1201",
            ],
            with_cuda=True,
            is_python_module=False,
            verbose=verbose,
        )
        _loaded = True


__all__ = ["load_kernels"]
