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
"""The disaggregated serving modules load without the Responses implementation.

``tensorrt_llm/serve/responses_utils.py`` is the Responses API implementation; it
pulls in the tool parsers, ``web_search`` and the harmony adapter. The
disaggregated client, service, coordinator and router need only the service
types and the shared Responses stream module in ``tensorrt_llm/serve/_base/``,
so importing them must not load it. The
layering contract in ``test_layering.py`` checks direct imports; this test
checks what a fresh interpreter actually loads.
"""

import json
import subprocess
import sys
import textwrap

import pytest

# The L0 CPU stage invokes registered files with `-m cpu_only`; without this
# marker every test is deselected and pytest exits with code 5.
pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

RESPONSES_IMPLEMENTATION = "tensorrt_llm.serve.responses_utils"

DISAGGREGATED_MODULES = (
    "tensorrt_llm.serve.openai_client",
    "tensorrt_llm.serve.openai_service",
    "tensorrt_llm.serve.openai_disagg_service",
    "tensorrt_llm.serve.disagg_coordinator",
    "tensorrt_llm.serve.router",
)


def _loads_responses_implementation(module: str) -> bool:
    """Import ``module`` in a new interpreter; whether the Responses implementation loaded."""
    code = textwrap.dedent(
        f"""
        import importlib, json, sys
        importlib.import_module({module!r})
        print(json.dumps({{"loaded": {RESPONSES_IMPLEMENTATION!r} in sys.modules}}))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        print(result.stderr[-2000:], file=sys.stderr)  # shown in the failing test's captured output
    result.check_returncode()  # CalledProcessError: an import crash is not the expected failure
    return json.loads(result.stdout.strip().splitlines()[-1])["loaded"]


@pytest.mark.parametrize("module", DISAGGREGATED_MODULES)
def test_disaggregated_modules_do_not_load_responses_utils(module: str) -> None:
    assert not _loads_responses_implementation(module), (
        f"importing {module} loads {RESPONSES_IMPLEMENTATION} (the whole Responses "
        "implementation, its tool parsers and web_search). Import the disaggregated service "
        "types from tensorrt_llm.serve._base.service_types and the clock from "
        "tensorrt_llm._utils; shared Responses stream code belongs in "
        "tensorrt_llm/serve/_base/responses_stream.py."
    )


def test_probe_sees_the_responses_implementation_when_it_is_imported() -> None:
    # Positive control for the probe above: importing the implementation itself
    # must be reported as loaded, or a "not loaded" result proves nothing.
    assert _loads_responses_implementation(RESPONSES_IMPLEMENTATION)
