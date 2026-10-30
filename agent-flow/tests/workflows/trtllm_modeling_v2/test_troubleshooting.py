# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The troubleshooting doc, kept honest.

Infrastructure knowledge rots the same way engine knowledge does, and in the
same direction: the symptom stays true while everything around it moves. So
this doc is keyed on **verbatim output** — the string you actually saw — and
these checks hold it to that.

Two failure modes are worth spending a test on. A doc that names a check
which no longer exists sends the reader looking for code that is not there;
and a doc nobody is pointed at when they need it is a doc nobody reads, which
is the more common way a troubleshooting file dies.
"""

from __future__ import annotations

import re
from pathlib import Path

import agent_flow.workflows.trtllm_modeling_v2 as pkg
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TROUBLESHOOTING_HINT

PACKAGE = Path(pkg.__file__).parent
DOC = PACKAGE / "TROUBLESHOOTING.md"


def _text() -> str:
    return DOC.read_text(encoding="utf-8")


def test_the_doc_is_packaged_next_to_the_workflow():
    """Not in a docs tree somewhere: the failure message names this path."""
    assert DOC.is_file()


def test_the_failure_message_points_at_a_file_that_exists():
    """The hint is built from a path; a typo there is silent until it fires."""
    quoted = re.search(r"`([^`]+TROUBLESHOOTING\.md)`", TROUBLESHOOTING_HINT)
    assert quoted, TROUBLESHOOTING_HINT
    assert Path(quoted.group(1)).is_file()


def test_the_readme_sends_the_reader_here():
    assert "TROUBLESHOOTING.md" in (PACKAGE / "README.md").read_text(encoding="utf-8")


def test_every_check_the_doc_credits_actually_exists():
    """A `*Caught by:* `x`` line is a claim about code, so resolve it.

    A doc that credits a check which has been renamed or deleted is worse
    than one that credits nothing: the reader concludes the case is handled
    and stops looking.
    """
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in PACKAGE.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    # The test suite is the other legitimate home for a named check.
    sources += "\n".join(
        path.read_text(encoding="utf-8") for path in (Path(__file__).parent).glob("test_*.py")
    )

    # Only the *first* backticked token names the check. What follows is
    # prose that legitimately quotes config keys and env vars — `runtime`,
    # `TMPDIR` — and resolving those as symbols is how this check would
    # become noisy enough to switch off.
    missing: list[str] = []
    for line in _text().splitlines():
        if not line.startswith("*Caught by:*"):
            continue
        for name in re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)\(?\)?`", line)[:1]:
            if re.search(
                rf"\b(def|class)\s+{re.escape(name)}\b|^{re.escape(name)}\s*=", sources, re.M
            ):
                continue
            missing.append(f"{name} (in: {line.strip()})")
    assert not missing, "TROUBLESHOOTING.md credits checks that do not exist:\n  " + "\n  ".join(
        missing
    )


def test_the_preflight_table_matches_the_checks_preflight_runs():
    """The table is a promise about what runs before an agent exists."""
    from agent_flow.workflows.trtllm_modeling_v2.run import preflight

    text = _text()
    for name in (
        "check_checkpoint",
        "check_tmpdir",
        "claim_devices",
        "check_multirank_substrate",
        "substrate_gate",
    ):
        assert hasattr(preflight, name), name
        assert f"`{name}`" in text, f"{name} is a preflight check the doc does not list"


def test_the_doc_obeys_the_citation_rule_it_relies_on():
    """It is keyed on verbatim output, which is the durable half.

    A line number would undercut the whole premise: the reader greps for the
    symptom, and the symptom is what survives. Quoted diagnostics keep their
    line numbers — those are part of the message the engine prints.
    """
    from .test_skills import (
        BARE_LINE_REFERENCE,
        LINE_CITATION,
        _is_shown_as_an_example,
        _quotes_a_diagnostic,
    )

    lines = _text().splitlines()
    offences: list[str] = []
    for index, line in enumerate(lines):
        if BARE_LINE_REFERENCE.search(line):
            offences.append(f"{index + 1}: bare `:NNN`")
        hit = LINE_CITATION.search(line)
        if hit is None or _quotes_a_diagnostic(lines, index):
            continue
        if _is_shown_as_an_example(line, hit.start(), hit.end()):
            continue
        offences.append(f"{index + 1}: {hit.group(0)}")
    assert not offences, "TROUBLESHOOTING.md cites code by line:\n  " + "\n  ".join(offences)


def test_the_open_problem_is_still_labelled_open():
    """`get_stats()` is why three features are unverified. Do not lose that."""
    text = _text()
    assert "get_stats()" in text
    assert "*Open.*" in text


def test_the_doc_is_declared_as_package_data():
    """Otherwise a non-editable install has no file behind the hint.

    This is the failure that cannot be caught by running from a source tree,
    which is the only way these tests ever run — so it is asserted against
    the packaging declaration instead.
    """
    import tomllib

    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    declared = data["tool"]["setuptools"]["package-data"]["agent_flow.workflows.trtllm_modeling_v2"]
    assert "TROUBLESHOOTING.md" in declared
