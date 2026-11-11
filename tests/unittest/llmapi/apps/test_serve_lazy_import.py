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
"""``tensorrt_llm.serve`` loads its two servers lazily."""

import json
import subprocess
import sys
import textwrap

import pytest

# The L0 CPU stage invokes registered files with `-m cpu_only`; without this
# marker every test is deselected and pytest exits with code 5.
pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]


def _run_in_fresh_interpreter(code: str) -> dict:
    """Run ``code`` in a new interpreter and return the JSON it prints last."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_importing_the_package_does_not_load_the_servers() -> None:
    out = _run_in_fresh_interpreter(
        """
        import json, sys
        import tensorrt_llm.serve  # noqa: F401
        print(json.dumps({
            "openai_server": "tensorrt_llm.serve.openai_server" in sys.modules,
            "openai_disagg_server": "tensorrt_llm.serve.openai_disagg_server" in sys.modules,
        }))
        """
    )
    assert out == {"openai_server": False, "openai_disagg_server": False}


def test_servers_load_on_first_access_and_are_the_real_classes() -> None:
    out = _run_in_fresh_interpreter(
        """
        import json, sys
        import tensorrt_llm.serve as serve
        before = "tensorrt_llm.serve.openai_server" in sys.modules
        server_cls = serve.OpenAIServer
        disagg_cls = serve.OpenAIDisaggServer
        from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
        from tensorrt_llm.serve.openai_server import OpenAIServer
        print(json.dumps({
            "loaded_before_access": before,
            "server_is_real": server_cls is OpenAIServer,
            "disagg_is_real": disagg_cls is OpenAIDisaggServer,
            "cached": serve.__dict__.get("OpenAIServer") is OpenAIServer,
        }))
        """
    )
    assert out == {
        "loaded_before_access": False,
        "server_is_real": True,
        "disagg_is_real": True,
        "cached": True,
    }


def test_from_import_of_the_servers_and_of_submodules_still_works() -> None:
    out = _run_in_fresh_interpreter(
        """
        import json
        from tensorrt_llm.serve import OpenAIDisaggServer, OpenAIServer
        from tensorrt_llm.serve import chat_utils
        print(json.dumps({
            "server": OpenAIServer.__name__,
            "disagg": OpenAIDisaggServer.__name__,
            "submodule": chat_utils.__name__,
        }))
        """
    )
    assert out == {
        "server": "OpenAIServer",
        "disagg": "OpenAIDisaggServer",
        "submodule": "tensorrt_llm.serve.chat_utils",
    }


def test_unknown_attribute_raises_attribute_error() -> None:
    import tensorrt_llm.serve as serve

    with pytest.raises(AttributeError):
        serve.definitely_not_a_serve_attribute  # noqa: B018


def test_dir_lists_the_lazy_names() -> None:
    import tensorrt_llm.serve as serve

    names = dir(serve)
    assert "OpenAIServer" in names
    assert "OpenAIDisaggServer" in names
