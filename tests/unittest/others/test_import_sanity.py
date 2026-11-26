# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cold imports of runtime entry points in the full package-test environment.

Each target gets a fresh interpreter so imports already performed by pytest,
its conftest, or another case cannot hide a broken import graph. The targets
exercise CLI registration, eager annotations, custom-op registration, config
schemas and backend imports. This does not validate kernel execution, lazy
operator lookup, cache invalidation or distributed startup.

The file is registered in CPU and package-sanity inventories. These checks
require the real native dependencies, but no visible GPU, model weights or
inference.
"""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.cpu_only

_MODULES = (
    "tensorrt_llm",
    "tensorrt_llm._bootstrap",
    "tensorrt_llm.commands.serve",
    "tensorrt_llm._torch.pyexecutor._util",
    "tensorrt_llm._torch.pyexecutor.guided_decoder",
    "tensorrt_llm._torch.pyexecutor.py_executor",
    "tensorrt_llm._torch.pyexecutor.py_executor_creator",
    "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs",
    "tensorrt_llm._torch.pyexecutor.request_utils",
    "tensorrt_llm._torch.custom_ops.torch_custom_ops",
    "tensorrt_llm.llmapi.llm_args",
    "tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen",
    "tensorrt_llm._torch.moe.fused_moe.mega_moe.mega_moe_deepgemm",
    "tensorrt_llm._torch.moe.fused_moe.mega_moe.mega_moe_cute_dsl",
    "tensorrt_llm._torch.models.checkpoints.hf.weight_loader",
)

_SENTINEL = "IMPORT-SANITY-OK:"
_PROBE = """
import importlib
import sys

importlib.import_module(sys.argv[1])
print('IMPORT-SANITY-OK:' + sys.argv[1])
"""


def _check_cold_import(module_name: str, timeout: int = 300) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, module_name],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        f"Cold import failed: {module_name}\n"
        f"--- stdout ---\n{result.stdout[-8000:]}\n"
        f"--- stderr ---\n{result.stderr[-8000:]}"
    )
    # An imported module calling sys.exit(0) is not a successful import.
    assert _SENTINEL + module_name in result.stdout.splitlines(), (
        f"Cold import exited before completion: {module_name}\n"
        f"--- stdout ---\n{result.stdout[-8000:]}\n"
        f"--- stderr ---\n{result.stderr[-8000:]}"
    )


@pytest.mark.parametrize("module_name", _MODULES)
def test_runtime_module_imports_in_fresh_interpreter(module_name: str) -> None:
    _check_cold_import(module_name)
