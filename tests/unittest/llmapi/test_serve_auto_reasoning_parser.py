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
"""Unit coverage for how trtllm-serve acts on `--reasoning_parser auto`.

The serve command is driven through click up to `get_llm_args`, the first call
that receives the resolved parser, which is replaced by a recorder that stops
the command there.
"""

import json
from pathlib import Path
from typing import Optional

import pytest
from click.testing import CliRunner

import tensorrt_llm.commands.serve as serve_module

pytestmark = pytest.mark.cpu_only

_HYBRID_TEMPLATE = (
    "{%- if enable_thinking is defined and enable_thinking is false %}"
    "{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}"
)
_FORCED_NON_THINKING_TEMPLATE = (
    "{%- for message in messages %}{{- '<|im_start|>' + message.role }}{%- endfor %}"
)


class _ReachedGetLlmArgs(Exception):
    """Raised by the recorder once serve has resolved its arguments."""


def _write_checkpoint(model_dir: Path, model_type: str, jinja: Optional[str] = None) -> str:
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": model_type}))
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer"})
    )
    if jinja is not None:
        (model_dir / "chat_template.jinja").write_text(jinja)
    return str(model_dir)


def _invoke_serve_auto(monkeypatch: pytest.MonkeyPatch, model: str):
    recorded = {}

    def record_get_llm_args(*args, **kwargs):
        recorded["reasoning_parser"] = kwargs["reasoning_parser"]
        raise _ReachedGetLlmArgs()

    monkeypatch.setattr(serve_module, "get_llm_args", record_get_llm_args)
    result = CliRunner().invoke(serve_module.serve, [model, "--reasoning_parser", "auto"])
    return result, recorded


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        pytest.param(_HYBRID_TEMPLATE, "qwen3", id="hybrid"),
        pytest.param(_FORCED_NON_THINKING_TEMPLATE, None, id="forced_non_thinking"),
    ],
)
def test_serve_auto_reasoning_parser_starts_for_qwen3_jinja_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, template: str, expected: Optional[str]
) -> None:
    """A Qwen3 template read from chat_template.jinja decides the parser.

    A template that never opens a reasoning block starts serve without one.
    """
    model = _write_checkpoint(tmp_path / "Qwen3-jinja", "qwen3_moe", jinja=template)

    result, recorded = _invoke_serve_auto(monkeypatch, model)

    assert isinstance(result.exception, _ReachedGetLlmArgs), result.output
    assert recorded["reasoning_parser"] == expected


@pytest.mark.parametrize(
    ("name", "model_type"),
    [
        ("DeepSeek-V3.1", "deepseek_v3"),
        ("SomeUnknownModel", "some_unknown_type"),
    ],
)
def test_serve_auto_reasoning_parser_rejects_undetectable_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, model_type: str
) -> None:
    """Without template evidence, a None result is still a usage error."""
    model = _write_checkpoint(tmp_path / name, model_type, jinja=_FORCED_NON_THINKING_TEMPLATE)

    result, recorded = _invoke_serve_auto(monkeypatch, model)

    assert result.exit_code == 2, result.output
    assert "Cannot auto-detect reasoning parser" in result.output
    assert "deepseek_v3 (R1 only)" in result.output
    assert recorded == {}
