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
"""The serve agent docs name only paths and commands that exist.

Rules written for agents rot silently when the file or command they name moves;
this test keeps every inline-code path and command in the serve ``AGENTS.md``
files real, and bans line-number anchors, which drift without failing anything.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[3]

DOCS = ("tensorrt_llm/serve/AGENTS.md", "tensorrt_llm/commands/AGENTS.md")

PATH_ROOTS = (
    "tensorrt_llm/",
    "serve/",
    "commands/",
    "tests/",
    "scripts/",
    "jenkins/",
    ".github/",
    "docs/",
    "examples/",
)
ROOT_FILES = ("AGENTS.md", "CLAUDE.md", "legacy-files.txt", "CODING_GUIDELINES.md")
COMMAND_STARTS = ("pytest", "python", "python3", "PYTHONPATH=", "LLM_MODELS_ROOT=")

_INLINE_CODE = re.compile(r"`([^`]+)`")
_LINE_ANCHOR = re.compile(r"\.(py|md|ya?ml|groovy|txt)\s*:\s*\d")
_BARE_ANCHOR_SPAN = re.compile(r"`:\d")


def _spans(doc: Path):
    return _INLINE_CODE.findall(doc.read_text())


def _resolve(path_text: str) -> Path:
    if path_text.startswith(("serve/", "commands/")):
        return REPO_ROOT / "tensorrt_llm" / path_text
    return REPO_ROOT / path_text


def _is_template(token: str) -> bool:
    return any(symbol in token for symbol in "<>*")


def _missing(doc: str, token: str) -> str:
    return (
        f"{doc} names `{token}`, which does not exist. Update the rule in the PR that "
        "moved or deleted it."
    )


def _check_command(doc: str, span: str, problems: list) -> None:
    tokens = span.split(" ")
    pythonpath_roots = []
    for token in tokens:
        if token.startswith("PYTHONPATH="):
            pythonpath_roots = token.split("=", 1)[1].split(":")
    for index, token in enumerate(tokens):
        if _is_template(token):
            continue
        if index > 0 and tokens[index - 1] == "-m" and "/" not in token and tokens[0] != "pytest":
            # `python3 -m <module>`: resolve to a file under the PYTHONPATH=
            # roots or the repository root.
            candidates = []
            for root in [*pythonpath_roots, "."]:
                base = REPO_ROOT / root / Path(*token.split("."))
                candidates.extend((base.with_suffix(".py"), base / "__init__.py"))
            if not any(candidate.exists() for candidate in candidates):
                problems.append(_missing(doc, token))
            continue
        if token.startswith(PATH_ROOTS):
            target = token.split("::")[0]
            if not _resolve(target).exists():
                problems.append(_missing(doc, token))


@pytest.mark.parametrize("doc", DOCS)
def test_paths_and_commands_are_real(doc):
    doc_path = REPO_ROOT / doc
    problems = []
    for span in _spans(doc_path):
        if " " not in span and (span.startswith(PATH_ROOTS) or span in ROOT_FILES):
            if _is_template(span):
                continue
            target = _resolve(span.split("::")[0])
            if not target.exists():
                problems.append(_missing(doc, span))
        elif span.startswith(COMMAND_STARTS):
            _check_command(doc, span, problems)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("doc", DOCS)
def test_no_line_number_anchors(doc):
    text = (REPO_ROOT / doc).read_text()
    assert not _LINE_ANCHOR.search(text) and not _BARE_ANCHOR_SPAN.search(text), (
        f"{doc} contains a line-number anchor. Name files and symbols, never line numbers: "
        "anchors drift silently when the file changes."
    )


@pytest.mark.parametrize("doc", DOCS)
def test_sibling_claude_md(doc):
    doc_path = REPO_ROOT / doc
    sibling = doc_path.with_name("CLAUDE.md")
    assert sibling.exists(), f"{doc} needs a sibling CLAUDE.md pointer file"
    lines = [line.strip() for line in sibling.read_text().splitlines() if line.strip()]
    assert lines == ["# In ./CLAUDE.md", "@AGENTS.md"], (
        f"{sibling.relative_to(REPO_ROOT)} must hold exactly the repo's two-line pointer "
        "form: `# In ./CLAUDE.md` and `@AGENTS.md`."
    )
