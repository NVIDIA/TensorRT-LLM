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
"""Guard that ``import tensorrt_llm`` stays GPU-free under ``CUDA_VISIBLE_DEVICES=""``.

A pure-client process -- most notably the ``benchmark_serving`` load generator --
runs with ``CUDA_VISIBLE_DEVICES=""`` (no visible GPU) yet still does
``import tensorrt_llm`` to reuse the tokenizer / API helpers. Several modules
probe the CUDA device at *import* time (device capability / properties,
``deep_gemm.set_pdl()``), which used to either raise or silently force a CUDA
context just from the import. Those probes are now guarded by
``torch.cuda.is_available()``; this test pins that contract -- dropping a guard
(or adding a new unguarded import-time probe) makes the subprocess crash or
report an initialized CUDA context.

The check runs in a *fresh* subprocess on purpose: ``CUDA_VISIBLE_DEVICES`` is
read once when the CUDA runtime first initializes, so it must be set before
``torch`` is imported -- which already happened in the pytest process.
"""

import os
import subprocess
import sys
import textwrap

_SENTINEL = "GPU-FREE-OK"


def _run_gpu_free(body: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run ``body`` in a fresh interpreter that sees no CUDA device.

    The wrapper asserts the GPU is hidden before ``body`` runs, asserts no CUDA
    context was created afterwards, and prints ``_SENTINEL`` on success so the
    caller can tell a clean run apart from one that exited early.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    precondition = textwrap.dedent("""\
        import torch

        assert not torch.cuda.is_available(), (
            "precondition failed: CUDA_VISIBLE_DEVICES='' did not hide the GPU"
        )
        """)
    postcondition = textwrap.dedent(f"""\

        assert not torch.cuda.is_initialized(), (
            "import forced a CUDA context with CUDA_VISIBLE_DEVICES='' "
            "(the GPU-free import contract was broken)"
        )
        print({_SENTINEL!r})
        """)
    script = precondition + textwrap.dedent(body) + postcondition

    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def test_import_tensorrt_llm_is_gpu_free():
    """``import tensorrt_llm`` must succeed without a visible GPU.

    This is the headline contract for pure-client processes (e.g. the
    ``benchmark_serving`` load generator): the import must neither require a GPU
    nor initialize a CUDA context.
    """
    result = _run_gpu_free("import tensorrt_llm\n")
    assert result.returncode == 0, (
        "`import tensorrt_llm` failed with CUDA_VISIBLE_DEVICES=''\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert _SENTINEL in result.stdout
