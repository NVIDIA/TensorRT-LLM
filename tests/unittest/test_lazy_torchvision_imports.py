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
"""Modules that touch torchvision import cleanly without torchvision.

The audited modules below moved their ``import torchvision`` (or
``from torchvision...``) statements from module level into the function
that uses them.  This test pins that contract: any future regression
that re-introduces a top-level torchvision import will fail loudly here
rather than silently making torchvision a hard dependency of TRT-LLM.

Each subtest spawns a Python subprocess that installs a meta-path
finder hiding ``torchvision*`` from ``importlib`` and then imports one
of the audited modules.  A subprocess is required because once any
torchvision module is loaded into the parent pytest process, neither
``sys.modules`` manipulation nor a meta-path finder can un-load it.
"""

import subprocess
import sys
import textwrap

import pytest

# Modules whose torchvision imports must be lazy.  Add to this list
# whenever a new module touches torchvision -- and put the
# ``import torchvision`` (or ``from torchvision...``) inside the
# function that uses it, not at module top.
MODULES_REQUIRING_LAZY_TORCHVISION = [
    "tensorrt_llm._torch.models.modeling_mistral",
    "tensorrt_llm._torch.models.modeling_multimodal_utils",
    "tensorrt_llm._torch.models.modeling_phi4mm",
    "tensorrt_llm.inputs.utils",
]

# The probe runs in a fresh Python process and:
#   1) marks ``torchvision`` (and any cached submodules) as ``None`` in
#      ``sys.modules`` -- Python's documented contract makes this raise
#      ``ModuleNotFoundError`` on any subsequent ``import torchvision``
#      while still letting ``importlib.util.find_spec("torchvision")``
#      return ``None`` (i.e. "not installed") rather than raising,
#   2) imports the target module and returns success.
#
# We deliberately do NOT use a meta-path finder that raises on
# ``find_spec``: third-party libraries (notably ``transformers``) probe
# torchvision availability with ``find_spec`` and will surface the
# raised exception, which would mask whether the *trt-llm* module under
# test actually requires torchvision at import time.
_PROBE = textwrap.dedent("""\
    import sys, importlib

    sys.modules["torchvision"] = None
    for k in [k for k in list(sys.modules) if k.startswith("torchvision.")]:
        sys.modules[k] = None
    importlib.invalidate_caches()

    importlib.import_module({module!r})
    print("OK")
""")


@pytest.mark.parametrize("module", MODULES_REQUIRING_LAZY_TORCHVISION)
def test_module_imports_without_torchvision(module: str) -> None:
    """The module must load cleanly when torchvision is unavailable."""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"{module} failed to import without torchvision installed.\n"
        f"This means {module} (or one of its module-level imports) is "
        f"calling ``import torchvision`` / ``from torchvision...`` at "
        f"import time.  Move that import inside the function that uses "
        f"it so torchvision stays an optional runtime dependency.\n\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
