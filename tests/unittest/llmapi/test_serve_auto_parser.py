# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""How trtllm-serve reacts to `--reasoning_parser auto`.

Auto-detection answers "no parser" for a model that emits no reasoning block
and "I do not know this model" for an unrecognized one. The parser modules
report both as None, so only the CLI shows that the first starts the server
and the second is rejected.
"""

import json

import pytest
from click.testing import CliRunner

from tensorrt_llm.commands.serve import serve

pytestmark = pytest.mark.cpu_only


class _ReachedLaunch(Exception):
    """Raised in place of building the engine, once the CLI accepts the flags."""


def _write_model(tmp_path, name: str, model_type: str, chat_template: str):
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": model_type}))
    (model_dir / "chat_template.jinja").write_text(chat_template)
    return str(model_dir)


def test_auto_serves_a_model_that_needs_no_reasoning_parser(tmp_path, monkeypatch):
    """An Instruct checkpoint has no reasoning block, which is not an error."""
    captured = {}

    def _fake_get_llm_args(**kwargs):
        captured.update(kwargs)
        raise _ReachedLaunch

    monkeypatch.setattr("tensorrt_llm.commands.serve.get_llm_args", _fake_get_llm_args)
    model = _write_model(
        tmp_path,
        "Qwen3-30B-A3B-Instruct-2507",
        "qwen3_moe",
        "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}",
    )

    result = CliRunner().invoke(serve, [model, "--reasoning_parser", "auto"])

    assert isinstance(result.exception, _ReachedLaunch), result.output
    assert captured["reasoning_parser"] is None


def test_auto_rejects_an_unrecognized_model(tmp_path):
    model = _write_model(tmp_path, "SomeUnknownModel", "unknown_type", "")

    result = CliRunner().invoke(serve, [model, "--reasoning_parser", "auto"])

    assert result.exit_code != 0
    # The message is built from the mapping, so it cannot drift out of date.
    assert "Cannot auto-detect reasoning parser" in result.output
    assert "qwen3_5" in result.output
