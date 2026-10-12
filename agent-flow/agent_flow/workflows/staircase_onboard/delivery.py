# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase D — verify the agents' claims against disk, before anything merges.

Staircase's rule: **agent reports are claims.** Verify against disk before
every transition — the gate numbers against the runs, the products against
the directory. Most of that verification is mechanical, so most of it is
here rather than in an agent turn. What is left over — comparing the
runtime-value dump against contracts written in prose — is the auditor's,
and it is the only part of delivery that needs a language model.

In tree a finished onboarding touches **six** trees, three of them outside
the package the CI selector claims. Every one has a way of being silently
half-done, so every one gets a check:

1. the catalog entries the target imports
2. their GPU tests, in the tests tree
3. the target itself
4. the routing entry that makes it reachable
5. the accuracy anchor
6. the test-list registration that makes CI run the gate

Everything here reads files. Nothing imports the target: importing it needs
torch, a built extension and — for a multi-rank target — a live world, none
of which the orchestrator has or should need in order to notice that a file
is missing. The checks that genuinely require a running engine already ran:
they are the gates.
"""

from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml

MODELING_V2_PACKAGE = "tensorrt_llm/_torch/_experimental/modeling_v2"
CATALOG_DIR = f"{MODELING_V2_PACKAGE}/catalog"
CATALOG_TESTS_DIR = "tests/unittest/_torch/modeling_v2"
ACCURACY_REFERENCES_DIR = "tests/integration/defs/accuracy/references"
ACCURACY_TESTS_DIR = "tests/integration/defs/accuracy"
TEST_DB_DIR = "tests/integration/test_lists/test-db"

# ``catalog/torch/`` holds thin mirrors of torch callables. Upstream owns
# their correctness, so by design they carry no contract, no test and no
# receipts — absence there is not a gap.
MIRROR_CATEGORY = "torch"

_CATALOG_IMPORT = re.compile(
    r"tensorrt_llm\._torch\._experimental\.modeling_v2\.catalog\.([a-z_]+)\.([a-z0-9_]+)"
)


@dataclass(frozen=True)
class Check:
    """One verification, and what it found."""

    name: str
    passed: bool
    detail: str

    def render(self) -> str:
        return f"[{'ok' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


@dataclass(frozen=True)
class DeliveryReport:
    checks: tuple[Check, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def render(self) -> str:
        return "\n".join(check.render() for check in self.checks)


def imported_catalog_entries(modeling_py: Path) -> list[tuple[str, str]]:
    """Return ``(category, name)`` for every catalog entry a target imports.

    Parsed from the AST rather than by importing: a target's import pulls in
    torch and a built extension, and this has to work on a login node.
    """
    tree = ast.parse(modeling_py.read_text(encoding="utf-8"))
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        module = None
        if isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                match = _CATALOG_IMPORT.search(alias.name)
                if match:
                    found.append((match.group(1), match.group(2)))
            continue
        if module:
            match = _CATALOG_IMPORT.search(module)
            if match:
                found.append((match.group(1), match.group(2)))
    # Stable order, de-duplicated: a target may import two symbols from one
    # entry, which is one entry to verify, not two.
    return sorted(set(found))


def declared_required_ops(modeling_py: Path) -> list[str]:
    """Read ``REQUIRED_TRTLLM_OPS`` statically."""
    tree = ast.parse(modeling_py.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "REQUIRED_TRTLLM_OPS":
                try:
                    value = ast.literal_eval(node.value)
                except ValueError:
                    return []
                if isinstance(value, (list, tuple)):
                    return [str(v) for v in value]
    return []


def catalog_index_paths(repo_root: Path) -> set[str]:
    """Every entry path recorded in ``catalog/index.yaml``."""
    index = repo_root / CATALOG_DIR / "index.yaml"
    if not index.is_file():
        return set()
    data = yaml.safe_load(index.read_text(encoding="utf-8")) or {}
    return {
        str(entry.get("path", "")).strip()
        for entry in (data.get("entries") or [])
        if isinstance(entry, dict)
    }


def check_catalog_entries(repo_root: Path, entries: Iterable[tuple[str, str]]) -> list[Check]:
    """Every imported entry exists, is indexed, and (unless a mirror) is contracted."""
    checks: list[Check] = []
    indexed = catalog_index_paths(repo_root)
    for category, name in entries:
        rel = f"{category}/{name}.py"
        wrapper = repo_root / CATALOG_DIR / rel
        contract = wrapper.with_suffix(".md")
        problems: list[str] = []
        if not wrapper.is_file():
            problems.append(f"wrapper missing: {wrapper}")
        if category != MIRROR_CATEGORY and not contract.is_file():
            problems.append(f"contract missing: {contract}")
        if rel not in indexed:
            problems.append(f"not listed in catalog/index.yaml (looked for {rel!r})")
        checks.append(
            Check(
                name=f"catalog:{rel}",
                passed=not problems,
                detail="; ".join(problems) or "wrapper, contract and index entry present",
            )
        )
    return checks


def check_catalog_tests(repo_root: Path, entries: Iterable[tuple[str, str]]) -> list[Check]:
    """Every non-mirror entry has its GPU test, under the name CI matches.

    The name is not a preference: every file in the subtree is called
    ``test_modeling_v2_*`` precisely so it cannot collide with upstream's
    test of the same op, and so the CI selector's pattern matches it exactly.
    A correctly written test under any other name is a test on no list.

    A multi-rank entry's test is an op matrix over world sizes and is named
    ``test_modeling_v2_<name>_op_matrix.py``, as the in-tree ``comm`` tests
    are; either name satisfies the check.
    """
    checks: list[Check] = []
    for category, name in entries:
        if category == MIRROR_CATEGORY:
            continue
        candidates = [
            f"{CATALOG_TESTS_DIR}/{category}/test_modeling_v2_{name}{suffix}.py"
            for suffix in ("", "_op_matrix")
        ]
        found = next((rel for rel in candidates if (repo_root / rel).is_file()), None)
        checks.append(
            Check(
                name=f"test:{category}/{name}",
                passed=found is not None,
                detail=str(repo_root / found) if found else f"missing: {candidates[0]}",
            )
        )
    return checks


def check_target(repo_root: Path, target_rel: str) -> list[Check]:
    """The target's two products exist and it declares the surface it touches."""
    target = repo_root / target_rel
    checks = []
    for product in ("modeling.py", "weights.py"):
        path = target / product
        checks.append(
            Check(
                name=f"target:{product}",
                passed=path.is_file(),
                detail=str(path) if path.is_file() else f"missing: {path}",
            )
        )
    modeling = target / "modeling.py"
    if modeling.is_file():
        declared = declared_required_ops(modeling)
        checks.append(
            Check(
                name="target:REQUIRED_TRTLLM_OPS",
                passed=bool(declared),
                detail=(
                    f"declares {len(declared)} op(s): {', '.join(declared)}"
                    if declared
                    else "declares no REQUIRED_TRTLLM_OPS — nothing would catch a "
                    "renamed op until the forward hit it"
                ),
            )
        )
    return checks


def check_routing(repo_root: Path, family: str, target_rel: str) -> list[Check]:
    """The target is reachable: routing names it, and the claims test exists.

    A target that exists but is not routed is invisible — and worse than
    invisible, because under ``TRTLLM_MODELING_V2=auto`` the built-in
    implementation answers instead and everything looks like it worked.
    """
    routing = repo_root / MODELING_V2_PACKAGE / "models" / family / "routing.py"
    if not routing.is_file():
        return [Check(name="routing:module", passed=False, detail=f"missing: {routing}")]

    text = routing.read_text(encoding="utf-8")
    # The dotted form routing uses, e.g.
    # ``models.gpt_oss.gpt_oss_120b__sm_103__tp1.modeling``.
    dotted = target_rel.split(f"{MODELING_V2_PACKAGE}/", 1)[-1].replace("/", ".").rstrip(".")
    referenced = dotted in text
    claims_test = repo_root / CATALOG_TESTS_DIR / "test_modeling_v2_claims.py"
    return [
        Check(
            name="routing:target",
            passed=referenced,
            detail=(
                f"{routing.name} references {dotted}"
                if referenced
                else f"{routing.name} does not reference {dotted}; the target would "
                f"never be selected, and under `auto` the built-in implementation "
                f"would answer instead"
            ),
        ),
        Check(
            name="routing:claims-test",
            passed=claims_test.is_file(),
            detail=str(claims_test) if claims_test.is_file() else f"missing: {claims_test}",
        ),
    ]


def check_accuracy_reference(repo_root: Path, reference_yaml: str, model_key: str) -> list[Check]:
    """The anchor this run gated against is written down, keyed by the model."""
    path = Path(reference_yaml)
    if not path.is_absolute():
        path = repo_root / path
    if not path.is_file():
        return [Check(name="accuracy:reference", passed=False, detail=f"missing: {path}")]
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    present = model_key in data
    return [
        Check(
            name="accuracy:reference",
            passed=present,
            detail=(
                f"{path.name} carries an entry for {model_key}"
                if present
                else f"{path.name} has no entry for {model_key}; the gate ran against "
                f"an anchor that is not recorded anywhere"
            ),
        )
    ]


def check_test_registration(repo_root: Path, accuracy_test: str) -> list[Check]:
    """The accuracy gate is on a list CI actually reads.

    A gate that passes on a branch and is on no list passes exactly once.
    """
    test_path = repo_root / ACCURACY_TESTS_DIR / accuracy_test
    checks = [
        Check(
            name="tests:accuracy-file",
            passed=test_path.is_file(),
            detail=str(test_path) if test_path.is_file() else f"missing: {test_path}",
        )
    ]
    db = repo_root / TEST_DB_DIR
    stem = Path(accuracy_test).stem
    listed: list[str] = []
    if db.is_dir():
        for yml in sorted(db.glob("*.yml")):
            if stem in yml.read_text(encoding="utf-8"):
                listed.append(yml.name)
    checks.append(
        Check(
            name="tests:test-db",
            passed=bool(listed),
            detail=(
                f"{stem} is listed in {', '.join(listed)}"
                if listed
                else f"{stem} appears in no {TEST_DB_DIR}/*.yml; CI would never run it"
            ),
        )
    )
    return checks


def check_tree_committed(repo_root: Path, paths: Sequence[str]) -> list[Check]:
    """Whether the delivered trees are in git, not just on somebody's disk.

    **Every other check in this module reads files.** That is the right thing
    for "does this entry have a test", and the wrong thing for "is this target
    delivered": a file answers both identically whether or not it was ever
    committed.

    Measured, once: a run passed ``routing:target`` — see ``check_routing`` —
    and reported ``mmmu 58.33`` while ``routing.py``, the target's
    ``modeling.py`` (984 changed lines) and its ``weights.py`` sat uncommitted
    in the working tree, and the catalog commits certifying those very lines
    were already in history. It would have closed ``delivered`` and shipped a
    ``routing.py`` with no multimodal unit. The auditor was the only thing
    that looked.

    The same measurement runs the other way, and that is why this check reads
    *dirty*, not *absent*. A turn can do real work and never hand back — the
    work sitting in ``HEAD`` while the orchestrator waits on a decision file
    that was never written — and, conversely, a gate can measure a tree
    dirtied by somebody else: one run had an agent editing ``modeling.py``
    while three accuracy gates were queued, where a pass would have
    attributed a number to a commit it was never measured on. A delivery that
    checks "the file exists" cannot tell those two apart.

    Returns **no check at all** outside a git work tree, rather than a failing
    one. Where there is no git there is no claim to make, and a fixture in a
    temporary directory is not a delivery.
    """
    if not (repo_root / ".git").exists():
        return []
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", *paths],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [
            Check(
                name="git:target-committed",
                passed=False,
                detail=f"could not run git status: {type(exc).__name__}: {exc}",
            )
        ]
    if result.returncode != 0:
        return [
            Check(
                name="git:target-committed",
                passed=False,
                detail=f"git status exited {result.returncode}: {result.stderr.strip()[:200]}",
            )
        ]
    dirty = [line for line in result.stdout.splitlines() if line.strip()]
    return [
        Check(
            name="git:target-committed",
            passed=not dirty,
            detail=(
                "the delivered trees are clean at HEAD"
                if not dirty
                else "uncommitted, so a clone would not have them:\n  " + "\n  ".join(dirty[:12])
            ),
        )
    ]


def verify_delivery(
    *,
    repo_root: Path | str,
    target_rel: str,
    family: str,
    reference_yaml: str,
    model_key: str,
    accuracy_test: str,
    extra_entries: Sequence[tuple[str, str]] = (),
) -> DeliveryReport:
    """Run every mechanical delivery check and return the report.

    ``extra_entries`` lets the caller add catalog entries the target does not
    import directly — an entry a feature path reaches through another module,
    say — so the check set matches what the run actually produced.
    """
    root = Path(repo_root)
    checks: list[Check] = []

    modeling = root / target_rel / "modeling.py"
    entries: list[tuple[str, str]] = list(extra_entries)
    if modeling.is_file():
        entries = sorted(set(entries) | set(imported_catalog_entries(modeling)))
    else:
        checks.append(
            Check(
                name="target:modeling.py",
                passed=False,
                detail=f"missing: {modeling} — nothing else can be checked without it",
            )
        )

    checks.extend(check_target(root, target_rel))
    checks.extend(check_catalog_entries(root, entries))
    checks.extend(check_catalog_tests(root, entries))
    checks.extend(check_routing(root, family, target_rel))
    checks.extend(check_accuracy_reference(root, reference_yaml, model_key))
    checks.extend(check_test_registration(root, accuracy_test))
    checks.extend(check_tree_committed(root, (target_rel, MODELING_V2_PACKAGE, CATALOG_TESTS_DIR)))
    return DeliveryReport(checks=tuple(checks))


def summarize_for_auditor(report: DeliveryReport, dump_excerpt: str = "") -> str:
    """Render what the auditor is handed: the mechanical results, plus the dump.

    The auditor is told what has already been checked so it does not spend
    its one turn re-deriving any of it — its job is the single check that
    reading a file cannot settle.
    """
    lines = ["## Mechanical delivery checks (already run)", "", report.render(), ""]
    if dump_excerpt:
        lines += ["## Runtime-value dump", "", "```", dump_excerpt.rstrip(), "```", ""]
    return "\n".join(lines)


__all__ = [
    "ACCURACY_REFERENCES_DIR",
    "CATALOG_DIR",
    "CATALOG_TESTS_DIR",
    "Check",
    "DeliveryReport",
    "MIRROR_CATEGORY",
    "MODELING_V2_PACKAGE",
    "TEST_DB_DIR",
    "catalog_index_paths",
    "check_accuracy_reference",
    "check_catalog_entries",
    "check_catalog_tests",
    "check_routing",
    "check_target",
    "check_test_registration",
    "check_tree_committed",
    "declared_required_ops",
    "imported_catalog_entries",
    "summarize_for_auditor",
    "verify_delivery",
]
