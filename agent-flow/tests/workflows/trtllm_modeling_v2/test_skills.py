# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the packaged agent skills.

The skills are thousands of lines of prose ported from a repository with a
different layout. Prose has no compiler, so the failure mode is a skill that
confidently tells an agent to write a file into a directory that no longer
exists — and the agent then spends a turn discovering that. These checks are
the compiler that prose does not have.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

import agent_flow.workflows.trtllm_modeling_v2 as pkg
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import FEATURE_SKILLS

SKILLS_DIR = Path(pkg.__file__).parent / "agents" / "skills"
REPO_ROOT = Path(__file__).resolve().parents[4]

# Retired by the port. A skill naming one is telling an agent to use a
# product that no longer exists.
RETIRED = (
    "uv run",
    "ty check",
    "1.3.0rc25",
    ".allow-trtllm",
    "bench/accuracy.py",
    "bench/perf.py",
)

# Narrow, per-file exceptions to the retired-product check. The op inventory
# is *generated*, and its provenance line records the engine version the
# snapshot was taken at — that is a stamp on an observation, not a pin the
# targets assert against, and the file's header says so. Deleting it would
# make the snapshot undateable.
RETIRED_EXCEPTIONS = {
    ("trtllm-modeling-v2-kernel-feasibility/references/trtllm-ops.md", "1.3.0rc25"),
}

# Repo-relative path prefixes worth resolving. Anything quoted in backticks
# that starts with one of these should exist in the checkout.
CHECKED_PREFIXES = ("tensorrt_llm/", "tests/", "examples/", "jenkins/")

# Paths a skill may name that do not exist yet, each with a reason. Keeping
# this list short is the point: it is the inventory of what the port still
# owes, not a place to silence the check.
KNOWN_ABSENT = {
    # Ported separately; referenced by the kernel skills.
    "agent_flow/workflows/trtllm_modeling_v2/tools/list_trtllm_ops.py",
}


def _skill_dirs() -> list[Path]:
    """Every packaged skill, and nothing else that happens to be a directory.

    Dot-directories are excluded because tooling leaves them here. A Claude
    Code session writes an empty `.claude/.cc-writes/` into whatever tree it
    touches, and without this every check below reported the packaged skills
    as malformed — in a working tree, on a file git does not even track,
    which is the most confusing possible way for a suite to go red.
    """
    if not SKILLS_DIR.is_dir():
        return []
    return sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir() and not p.name.startswith("."))


def _markdown_files() -> list[Path]:
    return sorted(p for skill in _skill_dirs() for p in skill.rglob("*.md"))


pytestmark = pytest.mark.skipif(
    not _skill_dirs(), reason="no skills packaged yet (the knowledge port has not landed)"
)


def _front_matter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, f"{path} has no YAML front matter"
    return yaml.safe_load(match.group(1)) or {}


def test_every_skill_directory_has_a_skill_md():
    for skill in _skill_dirs():
        assert (skill / "SKILL.md").is_file(), skill.name


def test_every_skill_is_prefixed():
    """The prefix is what keeps these from colliding with the neighbouring roots."""
    for skill in _skill_dirs():
        assert skill.name.startswith("trtllm-modeling-v2-"), skill.name


def test_front_matter_name_matches_the_directory():
    for skill in _skill_dirs():
        data = _front_matter(skill / "SKILL.md")
        assert data.get("name") == skill.name, skill.name


def test_every_skill_has_a_description():
    """The description is what the model matches on when deciding to load it."""
    for skill in _skill_dirs():
        description = _front_matter(skill / "SKILL.md").get("description", "")
        assert len(description) > 60, skill.name


def test_no_skill_names_a_retired_product():
    offences = []
    for path in _markdown_files():
        relative = str(path.relative_to(SKILLS_DIR))
        text = path.read_text(encoding="utf-8")
        for retired in RETIRED:
            if (relative, retired) in RETIRED_EXCEPTIONS:
                continue
            # Word-bounded on the left, or "ty check" matches "identity
            # check" and a porting agent spends a turn rewriting a sentence
            # that was already correct.
            pattern = re.compile(rf"(?<![\w.]){re.escape(retired)}")
            for line_no, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    offences.append(f"{relative}:{line_no}: {retired}")
    assert not offences, "retired products still named:\n  " + "\n  ".join(offences)


#: Skill subtrees whose documents describe work that has NOT been done yet, so
#: the paths in them are proposals rather than pointers.
#:
#: `references/decompositions/` is the only one. A MACRO decomposition report
#: exists precisely to say "here is the entry that should exist and the test
#: that should certify it" -- naming a path that does not resolve is its whole
#: job. Checking it as a pointer turned a correct report into a suite failure,
#: which is the check working against the thing it is meant to protect.
PROPOSAL_SUBTREES: tuple[str, ...] = ("references/decompositions/",)


def test_every_repo_path_a_skill_names_exists():
    """The mechanical half of the repath edit, checked rather than trusted."""
    missing: list[str] = []
    for path in _markdown_files():
        relative = str(path.relative_to(SKILLS_DIR))
        if any(subtree in relative for subtree in PROPOSAL_SUBTREES):
            continue
        text = path.read_text(encoding="utf-8")
        for token in re.findall(r"`([^`\n]+)`", text):
            token = token.strip().rstrip(".,;:").split("::", 1)[0]
            if not token.startswith(CHECKED_PREFIXES):
                continue
            if token in KNOWN_ABSENT:
                continue
            # Skip globs and placeholders — they name a shape, not a file.
            if any(ch in token for ch in "*<>{}") or token.endswith("/"):
                continue
            candidates = [REPO_ROOT / token]
            if not Path(token).suffix:
                # A catalog entry is habitually named without an extension
                # (`catalog/comm/allgather`), because the entry is the
                # contract *and* the wrapper *and* the test. Resolve either.
                candidates += [REPO_ROOT / f"{token}.py", REPO_ROOT / f"{token}.md"]
            if not any(candidate.exists() for candidate in candidates):
                missing.append(f"{path.relative_to(SKILLS_DIR)}: {token}")
    assert not missing, "skills name paths that do not exist:\n  " + "\n  ".join(
        sorted(set(missing))
    )


#: A citation of one symbol in one file: `<repo path>.py::<name>`. The path
#: check above resolves the file; this one also requires the name to appear in
#: it, which is what catches a function that was renamed or removed while its
#: file stayed.
FILE_SYMBOL = re.compile(r"`((?:" + "|".join(CHECKED_PREFIXES) + r")[\w./-]+\.py)::(\w+)`")


def _unresolved_symbols(text: str) -> list[str]:
    unresolved = []
    for path, symbol in FILE_SYMBOL.findall(text):
        source = REPO_ROOT / path
        if not source.is_file():
            unresolved.append(f"{path}::{symbol} (no such file)")
        elif not re.search(rf"\b{re.escape(symbol)}\b", source.read_text(encoding="utf-8")):
            unresolved.append(f"{path}::{symbol} (no such name in the file)")
    return unresolved


def test_every_file_symbol_citation_resolves():
    unresolved = [
        f"{path.relative_to(SKILLS_DIR)}: {item}"
        for path in _markdown_files()
        for item in _unresolved_symbols(path.read_text(encoding="utf-8"))
    ]
    assert not unresolved, "skills cite symbols the tree does not have:\n  " + "\n  ".join(
        sorted(set(unresolved))
    )


def test_the_symbol_citation_rule_actually_bites():
    assert _unresolved_symbols("`tensorrt_llm/llmapi/llm_args.py::TorchLlmArgs`") == []
    assert _unresolved_symbols("`tensorrt_llm/llmapi/llm_args.py::NoSuchArgsClassAnywhere`")
    assert _unresolved_symbols("`tensorrt_llm/no/such/module.py::anything`")


def test_cross_skill_references_resolve():
    """A skill pointing at a sibling must point at one that is packaged."""
    names = {skill.name for skill in _skill_dirs()}
    dangling: list[str] = []
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8")
        for referenced in re.findall(r"trtllm-modeling-v2-[a-z0-9-]+", text):
            if referenced not in names:
                dangling.append(f"{path.relative_to(SKILLS_DIR)}: {referenced}")
    assert not dangling, "dangling skill references:\n  " + "\n  ".join(sorted(set(dangling)))


def test_every_feature_has_the_skill_the_schema_promises():
    """`features: [mtp]` must resolve to a skill the assembler can actually load."""
    names = {skill.name for skill in _skill_dirs()}
    for feature, skill in FEATURE_SKILLS.items():
        assert skill in names, f"feature {feature!r} maps to {skill!r}, which is not packaged"


def test_the_assembler_always_loads_modeling_core():
    assert (SKILLS_DIR / "trtllm-modeling-v2-core" / "SKILL.md").is_file()


def test_the_two_smith_modes_are_packaged():
    for name in ("trtllm-modeling-v2-kernel-onboard", "trtllm-modeling-v2-kernel-verify"):
        assert (SKILLS_DIR / name / "SKILL.md").is_file(), name


def test_port_markers_are_visible_rather_than_silent():
    """`<!-- PORT: verify -->` is a deliberate to-do, and it should be findable.

    This does not fail on them — leaving one is the correct action when a fact
    could not be confirmed. It asserts they carry a note, so the next pass
    knows what was being questioned.
    """
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"<!--\s*PORT:\s*verify(.*?)-->", text, re.DOTALL):
            assert match.group(1).strip(), f"{path.relative_to(SKILLS_DIR)}: bare PORT marker"


# --------------------------------------------- how engine knowledge is written

#: A skill's job is to tell an agent what the engine does and where to look —
#: not to reproduce what the engine's code says. Reproducing it is the trap:
#: engine code moves every day, prose does not, and prose that has quietly
#: stopped being true is worse than prose that was never written, because the
#: agent believes it and spends a turn on it.
#:
#: So: **describe the behaviour, name the symbol, let the agent read the
#: source.** A file path and a function, class or field name survive a
#: refactor. A line number does not — measured, not assumed: the sibling
#: catalog held 7 line citations and all 7 pointed at the wrong line within
#: days of being written, while every symbol and path beside them still
#: resolved. `tests/unittest/_torch/modeling_v2/test_modeling_v2_no_stale_claims.py`
#: enforces the same rule on that tree; this is the skills half.
LINE_CITATION = re.compile(r"\b[A-Za-z0-9_./-]+\.(?:py|h|hpp|cpp|cu|cuh):\d+(?:[-,]\d+)*")
BARE_LINE_REFERENCE = re.compile(r"`:\d+`")

#: The one exception, and not a loophole: a line number inside quoted engine
#: output is part of the message the engine prints, so it is what an agent
#: greps for on hitting it rather than a direction to navigate.
DIAGNOSTIC_MARKERS = ("Assertion failed", "RuntimeError", "Error:", "error:", "[ERROR]")
DIAGNOSTIC_WINDOW = 3


#: A citation inside a ``double-backtick`` span is being *shown*, not made --
#: markdown's idiom for displaying a literal. The rule's own documentation has
#: to be able to print the bad form next to the good one, and a rule that
#: cannot state its own counter-example is a rule nobody can teach.
_SHOWN_LITERAL = re.compile(r"``.*?``")


def _is_shown_as_an_example(line: str, start: int, end: int) -> bool:
    return any(m.start() <= start and end <= m.end() for m in _SHOWN_LITERAL.finditer(line))


def _quotes_a_diagnostic(lines: list[str], index: int) -> bool:
    lo = max(0, index - DIAGNOSTIC_WINDOW)
    hi = min(len(lines), index + DIAGNOSTIC_WINDOW + 1)
    return any(marker in line for line in lines[lo:hi] for marker in DIAGNOSTIC_MARKERS)


def test_no_skill_cites_engine_code_by_line_number():
    """Knowledge with a shelf life, kept out of the knowledge layer."""
    offences: list[str] = []
    for path in _markdown_files():
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            bare = BARE_LINE_REFERENCE.search(line)
            if bare and not _is_shown_as_an_example(line, bare.start(), bare.end()):
                offences.append(f"{path.relative_to(SKILLS_DIR)}:{index + 1}: bare `:NNN`")
            hit = LINE_CITATION.search(line)
            if (
                hit is None
                or _quotes_a_diagnostic(lines, index)
                or _is_shown_as_an_example(line, hit.start(), hit.end())
            ):
                continue
            offences.append(f"{path.relative_to(SKILLS_DIR)}:{index + 1}: {hit.group(0)}")
    assert not offences, (
        "skills must name the file and the symbol, never the line — a line number "
        "is wrong by the next merge and sends the agent to whatever now occupies "
        "it:\n  " + "\n  ".join(sorted(set(offences)))
    )


def test_the_line_citation_rule_actually_bites():
    """A guard nobody has seen fail is a guard nobody knows works.

    Both halves matter: it must flag navigation, and it must leave a quoted
    diagnostic alone. The second is the one that would quietly over-fire and
    get the whole rule switched off.
    """
    navigation = ["warmup is at `_torch/pyexecutor/model_engine.py:1103`"]
    assert LINE_CITATION.search(navigation[0]) and not _quotes_a_diagnostic(navigation, 0)

    symptom = ["raises `RuntimeError: bad dtype (../tensorrt_llm/runtime/torchUtils.h:113)`"]
    assert LINE_CITATION.search(symptom[0]) and _quotes_a_diagnostic(symptom, 0)

    # A pasted traceback wraps, putting the marker a line above the citation.
    wrapped = [
        "```",
        "RuntimeError: [TensorRT-LLM][ERROR] Assertion failed:",
        "Failed: MPI error",
        "(../tensorrt_llm/runtime/utils/mpiUtils.cpp:277)",
    ]
    assert _quotes_a_diagnostic(wrapped, 3)

    good = ["warmup calls `_run_autotuner_warmup` (`_torch/pyexecutor/model_engine.py`)"]
    assert LINE_CITATION.search(good[0]) is None


def test_the_entry_spec_separates_pool_geometry_from_batch_geometry():
    """The one-line omission that cost nine entries.

    Eight different smiths, over two days, each sized a state pool to their
    test's batch -- the natural reading of "test it at the shape it is called
    with", and the one arrangement the runtime never produces. An audit found
    all nine outside the engine's real width of 461 at once.

    The spec has to say this explicitly because the correct instinct produces
    the wrong answer here: the batch comes from the executor, the pool's slot
    count from the cache manager's capacity planning, and the two are
    unrelated numbers.
    """
    spec = (
        SKILLS_DIR / "trtllm-modeling-v2-kernel-onboard" / "references" / "entry-spec.md"
    ).read_text(encoding="utf-8")
    assert "never from your batch" in spec
    # and the reason, not just the rule -- a rule with no failure attached is
    # the first thing a hurried reader skips
    assert "pool_slots == batch_size" in spec
    # and where the slot count is recorded: a cell axis, not a free one
    assert "Make the slot count a `spec` axis" in spec
