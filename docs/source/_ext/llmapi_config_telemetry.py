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
*type-driven*: fields whose type is categorical (`Literal`/`Enum`/`bool`) or
numeric (`int`/`float`), plus safe collections of those, are captured
automatically. Free-form `str`/`Any`/`Path`/`dict`/`Callable` are never captured
unless a field carries an explicit allowlist (`TelemetryField.categorical(...)`),
and any field may opt out with `telemetry=False`. Every captured field is listed
below; the runtime can capture nothing absent from this list.

If the manifest check fails, run `python3 scripts/generate_llm_args_golden_manifest.py`, then commit
`tensorrt_llm/usage/llm_args_golden_manifest.json`; new fields require telemetry/privacy CODEOWNER approval.

## LLM API Configuration Fields

A field can still be absent from a specific payload when its parent config is
unset or when the safety sanitizer rejects the runtime value.
"""


def _escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _format_values(values: list[str]) -> str:
    return ", ".join(f"`{_escape(v)}`" for v in values) if values else ""


def _table(rows: list[dict]) -> str:
    lines = [
        "| Captured key | Annotation | Kind | Converter | Allowed values |",
        "|--------------|------------|------|-----------|----------------|",
    ]
    for row in rows:
        lines.append(
            f"| `{_escape(row['path'])}` | `{_escape(row['annotation'])}` | "
            f"`{_escape(row['kind'])}` | {_escape(row['converter'])} | "
            f"{_format_values(row['allowed_values'])} |"
        )
    return "\n".join(lines)


def generate_telemetry_reference(repo_root: Path | str, output_path: Path | str) -> None:
    repo_root = Path(repo_root)
    golden = json.loads((repo_root / _GOLDEN_REL).read_text())
    content = [_REFERENCE_PREAMBLE]
    for args_class in ("TorchLlmArgs", "TrtLlmArgs"):
        rows = golden.get(args_class, [])
        content.extend(
            [
                f"### `{args_class}`",
                "",
                f"{len(rows)} captured fields.",
                "",
                _table(rows),
                "",
            ]
        )
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
