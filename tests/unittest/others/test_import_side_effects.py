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
"""Guards against CUDA side effects of ``import tensorrt_llm``.

Importing tensorrt_llm must not create a CUDA context: every process pays the
context + module memory (~0.5-1.2 GiB depending on arch) on its default device,
including processes that never launch a kernel (e.g. the trtllm-bench parent)
and MPI workers that have not yet called ``torch.cuda.set_device``. Those
contexts are resident when the KV cache pool is sized from free GPU memory,
silently shrinking it (nvbug 6419139: ~10% throughput regression on RTX 6000D
between 1.3.0rc19 and 1.3.0rc20, caused by an import-time deep_gemm.set_pdl()
call instantiating DeepGEMM's DeviceRuntime).
"""

import os
import subprocess
import sys

import pytest
import torch

_PDL_FLAG_SCRIPT = r"""
import sys

import tensorrt_llm  # noqa: F401  (the import under test)
from tensorrt_llm._torch.pyexecutor import model_engine

if model_engine._DEEP_GEMM_PDL_CONFIGURED:
    sys.exit("DeepGEMM PDL was configured at import time; it must stay lazy "
             "(nvbug 6419139): deep_gemm.set_pdl() instantiates DeepGEMM's "
             "DeviceRuntime and creates a CUDA context")
"""

_NO_CONTEXT_SCRIPT = r"""
import os
import sys

import pynvml

pynvml.nvmlInit()
# NVML indices are physical and ignore CUDA_VISIBLE_DEVICES; the test runner
# pins CUDA_VISIBLE_DEVICES to a single physical index, so use that one.
physical = int(os.environ["CUDA_VISIBLE_DEVICES"])
handle = pynvml.nvmlDeviceGetHandleByIndex(physical)

import tensorrt_llm  # noqa: F401  (the import under test)

procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
mine = [p for p in procs if p.pid == os.getpid()]
if mine:
    used = (mine[0].usedGpuMemory or 0) >> 20
    sys.exit(f"import tensorrt_llm created a CUDA context using {used} MiB")
"""


def _run_in_subprocess(script: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    # Pin to one physical device so the pynvml handle and the CUDA default
    # device agree regardless of the outer environment.
    visible = env.get("CUDA_VISIBLE_DEVICES", "").split(",")[0] or "0"
    env["CUDA_VISIBLE_DEVICES"] = visible
    return subprocess.run([sys.executable, "-c", script],
                          env=env,
                          capture_output=True,
                          text=True,
                          timeout=300)


def test_deep_gemm_pdl_configuration_is_lazy():
    """DeepGEMM PDL setup must not run at import (nvbug 6419139)."""
    result = _run_in_subprocess(_PDL_FLAG_SCRIPT)
    assert result.returncode == 0, (result.stdout + result.stderr)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="requires a CUDA device")
def test_import_creates_no_cuda_context():
    result = _run_in_subprocess(_NO_CONTEXT_SCRIPT)
    assert result.returncode == 0, (result.stdout + result.stderr)
