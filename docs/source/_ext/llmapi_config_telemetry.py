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

from __future__ import annotations

import json
from pathlib import Path

_GOLDEN_REL = "tensorrt_llm/usage/llm_args_golden_manifest.json"

_REFERENCE_PREAMBLE = """\
# Telemetry

This page documents TensorRT-LLM usage telemetry. It is generated during the
Sphinx docs build by rendering the committed telemetry manifest
(`tensorrt_llm/usage/llm_args_golden_manifest.json`).

Start with the
[Telemetry Data Collection section in the root README](source:README.md#telemetry-data-collection)
for the user-facing collection and opt-out overview, and the
[telemetry schema reference](source:tensorrt_llm/usage/schemas/README.md)
for the wire schema.

**No PII or free-form fields are captured.** LLM API configuration capture is
automatic for `bool`, `int`, finite `float`, `Literal`, `Enum`, supported unions,
and homogeneous sequences. Unsafe scalar `str`, `Any`, and `object` branches
require `TelemetryField.categorical(...)`; paths, mappings, callables, and
unsupported structures always fail closed. Use `telemetry=False` to exclude a
field. The runtime can capture nothing absent from the list below.

`capture_policy` branches separated by `|` are tried independently; `enum[X]`
requires the exact enum type `X`. The categorical domain lists tokens from
`Literal`/`Enum` annotations or explicit `allowed_values`; it does not restrict
`bool`, `int`, or `float` branches.

If the manifest check fails, run `python3 scripts/generate_llm_args_golden_manifest.py`, then commit
`tensorrt_llm/usage/llm_args_golden_manifest.json`; new fields require telemetry/privacy CODEOWNER approval.

## LLM API Configuration Fields

A field can still be absent from a specific payload when its parent config is
unset or when the safety sanitizer rejects the runtime value.
"""


def _escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _format_values(values: list[object]) -> str:
    def format_value(value: object) -> str:
        text = value if isinstance(value, str) else json.dumps(value)
        return f"`{_escape(text)}`"

    return ", ".join(format_value(value) for value in values)


def _table(rows: list[dict]) -> str:
    lines = [
        "| Captured key | Capture policy | Kind | Categorical domain |",
        "|--------------|----------------|------|--------------------|",
    ]
    for row in rows:
        lines.append(
            f"| `{_escape(row['path'])}` | `{_escape(row['capture_policy'])}` | "
            f"`{_escape(row['kind'])}` | "
            f"{_format_values(row.get('allowed_values', []))} |"
        )
    return "\n".join(lines)


def generate_telemetry_reference(repo_root: Path | str, output_path: Path | str) -> None:
    repo_root = Path(repo_root)
    golden = json.loads((repo_root / _GOLDEN_REL).read_text())
    rows = golden.get("TorchLlmArgs", [])
    content = [
        _REFERENCE_PREAMBLE,
        "### `TorchLlmArgs`",
        "",
        f"{len(rows)} captured fields.",
        "",
        _table(rows),
        "",
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(content))


def _on_builder_inited(app) -> None:
    docs_source = Path(app.confdir)
    repo_root = docs_source.parents[1]
    generate_telemetry_reference(repo_root, docs_source / "developer-guide/telemetry.md")


def setup(app) -> dict[str, object]:
    app.connect("builder-inited", _on_builder_inited)
    return {"version": "0.2", "parallel_read_safe": True, "parallel_write_safe": True}
