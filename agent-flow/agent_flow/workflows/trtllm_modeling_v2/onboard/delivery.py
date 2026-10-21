# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delivery — verify the agents' claims against disk, before anything merges.

Everything here reads files. Nothing imports the target: importing it needs
torch, a built extension and — for a multi-rank target — a live world, none
of which the orchestrator has or should need in order to notice that a file
is missing.
"""

from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from ..infra.committer import (
    DEFAULT_BASE_REFS,
    CommitPolicy,
    CommitPolicyError,
    audit_commits,
    resolve_base,
)

MODELING_V2_PACKAGE = "tensorrt_llm/_torch/_experimental/modeling_v2"
CATALOG_DIR = f"{MODELING_V2_PACKAGE}/catalog"
CATALOG_TESTS_DIR = "tests/unittest/_torch/modeling_v2"
ACCURACY_REFERENCES_DIR = "tests/integration/defs/accuracy/references"
ACCURACY_TESTS_DIR = "tests/integration/defs/accuracy"
TEST_DB_DIR = "tests/integration/test_lists/test-db"

#: A target's own entries -- ops only it calls -- sit in this directory of the
#: target, one ``<category>/<name>.py`` each, and their tests mirror that path
#: under ``CATALOG_TESTS_DIR/models/<family>/<target>/``.
MODEL_OPS_DIRNAME = "ops"

# What a target is made of: the core dispatches each step to one of these two,
# chosen by its ``_select_target`` (decode when ``num_contexts == 0``).
TARGET_CLASSES = ("PrefillTarget", "DecodeTarget")
# The fields the core routes on; a decode target states them, never reads them.
PHASE_FIELDS = ("num_contexts", "num_ctx_tokens")
# Module-level names of a manifest written beside the weight table by hand.
HAND_MANIFEST_NAMES = ("MANIFEST", "_manifest")

_CATALOG_IMPORT = re.compile(
    r"tensorrt_llm\._torch\._experimental\.modeling_v2\.catalog\.([a-z_]+)\.([a-z0-9_]+)"
)
_MODEL_OPS_IMPORT = re.compile(
    r"tensorrt_llm\._torch\._experimental\.modeling_v2\.models\.[a-z0-9_]+\."
    r"([a-z0-9_]+)\.ops\.([a-z_]+)\.([a-z0-9_]+)"
)
_RELATIVE_OPS_IMPORT = re.compile(r"^ops\.([a-z_]+)\.([a-z0-9_]+)")


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
    """Return ``(category, name)`` for every catalog entry a target imports."""
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
    return sorted(set(found))


def imported_model_ops(modeling_py: Path) -> list[tuple[str, str, str]]:
    """Return ``(target_dir, category, name)`` for every ``ops/`` entry a target imports.

    A target reaches its own ``ops/`` relatively (``from .ops.moe.x import X``)
    or by absolute path; either way ``target_dir`` is the directory the entry
    lives in, so an import of *another* target's op is visible as one.
    """
    tree = ast.parse(modeling_py.read_text(encoding="utf-8"))
    own = modeling_py.parent.name
    found: set[tuple[str, str, str]] = set()
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.level == 1:
                match = _RELATIVE_OPS_IMPORT.match(node.module)
                if match:
                    found.add((own, match.group(1), match.group(2)))
                continue
            modules = [node.module]
        elif isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        for module in modules:
            match = _MODEL_OPS_IMPORT.search(module)
            if match:
                found.add((match.group(1), match.group(2), match.group(3)))
    return sorted(found)


@dataclass(frozen=True)
class EntryContract:
    """What an entry's source states about itself, read without importing it.

    An entry is an ``OpWrapper`` subclass: ``__call__`` is the op call,
    ``reference`` what it should compute, ``CELLS`` the inputs it is certified
    over and ``ARCHS`` the architectures those cells were driven on.
    """

    classes: tuple[str, ...] = ()
    archs: tuple[str, ...] = ()
    cells: int = 0
    problems: tuple[str, ...] = ()


def _is_opwrapper_base(base: ast.expr) -> bool:
    return (isinstance(base, ast.Name) and base.id == "OpWrapper") or (
        isinstance(base, ast.Attribute) and base.attr == "OpWrapper"
    )


def _class_assignment(cls: ast.ClassDef, name: str) -> ast.expr | None:
    for node in cls.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return node.value
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            return node.value
    return None


def _arch_names(value: ast.expr | None) -> list[str]:
    """``frozenset({Arch.SM_103})`` -> ``["sm_103"]``; anything else -> ``[]``."""
    archs = []
    for node in ast.walk(value) if value is not None else ():
        if isinstance(node, ast.Attribute) and node.attr.startswith("SM_"):
            archs.append(node.attr.lower())
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("sm_"):
                archs.append(node.value)
    return sorted(set(archs))


def _cell_count(value: ast.expr | None) -> int:
    if value is None:
        return 0
    return sum(
        1
        for node in ast.walk(value)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "Cell")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "Cell")
        )
    )


def entry_contract(path: Path) -> EntryContract:
    """Read an entry's executable contract statically. See :class:`EntryContract`."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        return EntryContract(problems=(f"unreadable: {type(exc).__name__}: {exc}",))
    classes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and any(_is_opwrapper_base(b) for b in node.bases)
    ]
    if not classes:
        return EntryContract(
            problems=(
                "no OpWrapper subclass: an entry is its executable contract "
                "(__call__, reference, CELLS, ARCHS), not a bare function",
            )
        )
    problems: list[str] = []
    archs: set[str] = set()
    cells = 0
    for cls in classes:
        methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
        for required in ("__call__", "reference"):
            if required not in methods:
                problems.append(f"{cls.name} defines no {required}")
        cls_cells = _cell_count(_class_assignment(cls, "CELLS"))
        cls_archs = _arch_names(_class_assignment(cls, "ARCHS"))
        if not cls_cells:
            problems.append(f"{cls.name} declares no CELLS: nothing is certified")
        if not cls_archs:
            problems.append(
                f"{cls.name} declares no ARCHS: its cells were driven on no architecture"
            )
        cells += cls_cells
        archs.update(cls_archs)
    return EntryContract(
        classes=tuple(c.name for c in classes),
        archs=tuple(sorted(archs)),
        cells=cells,
        problems=tuple(problems),
    )


def _contract_detail(contract: EntryContract) -> str:
    return f"{'/'.join(contract.classes)}: {contract.cells} cell(s) on {', '.join(contract.archs)}"


def _entry_problems(wrapper: Path) -> tuple[list[str], EntryContract | None]:
    if not wrapper.is_file():
        return [f"entry missing: {wrapper}"], None
    contract = entry_contract(wrapper)
    return list(contract.problems), contract


def check_catalog_entries(repo_root: Path, entries: Iterable[tuple[str, str]]) -> list[Check]:
    """Every imported shared entry exists and states a certified contract."""
    checks: list[Check] = []
    for category, name in entries:
        rel = f"{category}/{name}.py"
        problems, contract = _entry_problems(repo_root / CATALOG_DIR / rel)
        checks.append(
            Check(
                name=f"catalog:{rel}",
                passed=not problems,
                detail="; ".join(problems) or _contract_detail(contract),
            )
        )
    return checks


def model_ops_tests_dir(target_rel: str) -> str:
    """``.../modeling_v2/models/<family>/<target>`` -> its tests' directory."""
    tail = target_rel.rstrip("/").split(f"{MODELING_V2_PACKAGE}/", 1)[-1]
    return f"{CATALOG_TESTS_DIR}/{tail}"


def check_model_ops(
    repo_root: Path, target_rel: str, ops: Iterable[tuple[str, str, str]]
) -> list[Check]:
    """Every ``ops/`` entry the target imports is its own, and states a certified contract."""
    own = Path(target_rel.rstrip("/")).name
    checks: list[Check] = []
    for target_dir, category, name in ops:
        rel = f"{MODEL_OPS_DIRNAME}/{category}/{name}.py"
        if target_dir != own:
            checks.append(
                Check(
                    name=f"ops:{rel}",
                    passed=False,
                    detail=f"imported from another target's ops/ ({target_dir}); targets share "
                    f"the catalog and nothing else -- promote the entry to "
                    f"{CATALOG_DIR}/{category}/{name}.py",
                )
            )
            continue
        problems, contract = _entry_problems(repo_root / target_rel / rel)
        checks.append(
            Check(
                name=f"ops:{rel}",
                passed=not problems,
                detail="; ".join(problems) or _contract_detail(contract),
            )
        )
    return checks


def _test_check(repo_root: Path, tests_dir: str, label: str, category: str, name: str) -> Check:
    candidates = [
        f"{tests_dir}/{category}/test_modeling_v2_{name}{suffix}.py"
        for suffix in ("", "_op_matrix")
    ]
    found = next((rel for rel in candidates if (repo_root / rel).is_file()), None)
    if found is None:
        return Check(name=label, passed=False, detail=f"missing: {candidates[0]}")
    test = repo_root / found
    # A multi-rank entry's collected file only launches its rank body, which
    # sits beside it as `_<name>_op_matrix.py`; the cells are driven there.
    sources = [test, test.parent / f"_{name}_op_matrix.py"]
    drives = any(p.is_file() and ".CELLS" in p.read_text(encoding="utf-8") for p in sources)
    return Check(
        name=label,
        passed=drives,
        detail=str(test)
        if drives
        else f"{test} never reads the entry's .CELLS: parametrize it off them, or the "
        f"cells it claims were never driven",
    )


def check_catalog_tests(repo_root: Path, entries: Iterable[tuple[str, str]]) -> list[Check]:
    """Every shared entry has its GPU test, under the name CI matches, driving its cells.

    A multi-rank entry's test is an op matrix over world sizes and is named
    ``test_modeling_v2_<name>_op_matrix.py``, as the in-tree ``comm`` tests
    are; either name satisfies the check.
    """
    return [
        _test_check(repo_root, CATALOG_TESTS_DIR, f"test:{category}/{name}", category, name)
        for category, name in entries
    ]


def check_model_op_tests(
    repo_root: Path, target_rel: str, ops: Iterable[tuple[str, str, str]]
) -> list[Check]:
    """Every own ``ops/`` entry has its GPU test under the target's mirror of the tests tree."""
    own = Path(target_rel.rstrip("/")).name
    tests_dir = model_ops_tests_dir(target_rel)
    return [
        _test_check(repo_root, tests_dir, f"test:ops/{category}/{name}", category, name)
        for target_dir, category, name in ops
        if target_dir == own
    ]


def _class_nodes(tree: ast.AST) -> dict[str, ast.ClassDef]:
    return {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def _check_phase_targets(tree: ast.AST) -> list[Check]:
    """Both targets exist, each answers for its step, and decode never reads the phase back.

    Decode is routed to only when ``num_contexts == 0``, so a read of
    ``num_contexts`` / ``num_ctx_tokens`` inside it is a branch on a
    per-capture constant -- frozen at whatever the capturing batch carried.
    """
    classes = _class_nodes(tree)
    problems: list[str] = []
    for name in TARGET_CLASSES:
        node = classes.get(name)
        if node is None:
            problems.append(f"no {name}")
            continue
        methods = {
            item.name
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        missing = [m for m in ("step_args", "forward") if m not in methods]
        if missing:
            problems.append(f"{name} defines no {', '.join(missing)}")
    checks = [
        Check(
            name="target:prefill-decode",
            passed=not problems,
            detail="; ".join(problems)
            or "PrefillTarget and DecodeTarget define step_args and forward",
        )
    ]
    decode = classes.get("DecodeTarget")
    if decode is not None:
        reads = sorted(
            {
                f"line {inner.lineno} .{inner.attr}"
                for inner in ast.walk(decode)
                if isinstance(inner, ast.Attribute) and inner.attr in PHASE_FIELDS
            }
        )
        checks.append(
            Check(
                name="target:decode-reads-no-phase",
                passed=not reads,
                detail=(
                    "DecodeTarget states the context counts instead of reading them"
                    if not reads
                    else "DecodeTarget reads the phase it was routed on ("
                    + ", ".join(reads)
                    + "); state num_contexts/num_ctx_tokens as 0 in its step_args"
                ),
            )
        )
    return checks


def _check_weight_table(weights_py: Path) -> Check:
    """``weights.py`` states each weight once, in a ``WEIGHTS`` table, exposed as ``MODEL_WEIGHTS``."""
    tree = ast.parse(weights_py.read_text(encoding="utf-8"))
    tables = [
        node
        for node in _class_nodes(tree).values()
        if any(
            (isinstance(item, ast.Assign) and any(_is_name(t, "WEIGHTS") for t in item.targets))
            or (isinstance(item, ast.AnnAssign) and _is_name(item.target, "WEIGHTS"))
            for item in node.body
        )
    ]
    hand_manifest = sorted(
        {
            target.id
            for node in tree.body
            for target in (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, ast.AnnAssign)
                else []
            )
            if isinstance(target, ast.Name) and target.id in HAND_MANIFEST_NAMES
        }
        | {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in HAND_MANIFEST_NAMES
        }
    )
    exposed = any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            _is_name(t, "MODEL_WEIGHTS")
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
        for node in tree.body
    )
    problems = []
    if not tables:
        problems.append("no class with a WEIGHTS table")
    if not exposed:
        problems.append("no module-level MODEL_WEIGHTS instance of the table")
    if hand_manifest:
        problems.append(
            f"hand-written {', '.join(hand_manifest)} beside the table; "
            "the table's manifest() generates it from WEIGHTS"
        )
    return Check(
        name="target:weight-table",
        passed=not problems,
        detail="; ".join(problems)
        or f"{tables[0].name} declares WEIGHTS; MODEL_WEIGHTS exposes it",
    )


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def check_target(repo_root: Path, target_rel: str) -> list[Check]:
    """The target's two products exist and have the shape the framework expects.

    A core that dispatches to a prefill and a decode target, and a weight
    table the manifest is generated from.
    """
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
        checks.extend(_check_phase_targets(ast.parse(modeling.read_text(encoding="utf-8"))))
    weights = target / "weights.py"
    if weights.is_file():
        checks.append(_check_weight_table(weights))
    return checks


def check_routing(repo_root: Path, family: str, target_rel: str) -> list[Check]:
    """The target is reachable: routing names it, and the claims test exists."""
    routing = repo_root / MODELING_V2_PACKAGE / "models" / family / "routing.py"
    if not routing.is_file():
        return [Check(name="routing:module", passed=False, detail=f"missing: {routing}")]

    text = routing.read_text(encoding="utf-8")
    # Routing imports the target where it returns it, relative to the family
    # package (``from .<target>.modeling import ...``); an
    # absolute ``models.<family>.<target>`` form names it too.
    dotted = target_rel.split(f"{MODELING_V2_PACKAGE}/", 1)[-1].replace("/", ".").rstrip(".")
    relative = f"from .{Path(target_rel.rstrip('/')).name}."
    referenced = dotted in text or relative in text
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


def check_accuracy_reference(
    repo_root: Path, reference_yaml: str, model_key: str, *, pending: bool = False
) -> list[Check]:
    """The anchor this run gated against is written down, keyed by the model.

    ``pending``: the spec says no written anchor exists yet (``accuracy.anchor``).
    The check then fails whatever the file holds: the row is the user's to record
    from a written source, never this run's to write from its own measurement.
    """
    if pending:
        return [
            Check(
                name="accuracy:reference",
                passed=False,
                detail=f"anchor pending: no written accuracy anchor for {model_key} yet; "
                "record the row from a written source (never from this run's own score)",
            )
        ]
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


def accuracy_test_path(repo_root: Path | str, family: str) -> str:
    """A target's CI accuracy test under ``ACCURACY_TESTS_DIR``, in the tree's own layout.

    Newer trees keep the modeling_v2 accuracy tests in a package,
    ``modeling_v2/test_<family>.py``; older ones beside the rest, as
    ``test_modeling_v2_<family>.py``. The package's presence says which.
    """
    if (Path(repo_root) / ACCURACY_TESTS_DIR / "modeling_v2").is_dir():
        return f"modeling_v2/test_{family}.py"
    return f"test_modeling_v2_{family}.py"


def check_test_registration(repo_root: Path, accuracy_test: str) -> list[Check]:
    """The accuracy gate is on a list CI actually reads."""
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


def check_commit_hygiene(repo_root: Path, policy: CommitPolicy) -> list[Check]:
    """The run's commits meet the commit rules ``commit.sh`` enforces one at a time.

    The audit runs over every commit since the merge base that touches the
    paths the run may commit (``commit_policy.allowed_paths``). Returns no
    check outside a git work tree, or when no base resolves.
    """
    if not (repo_root / ".git").exists():
        return []
    base = resolve_base(repo_root, [policy.base_ref] if policy.base_ref else DEFAULT_BASE_REFS)
    if not base:
        return []
    try:
        findings = audit_commits(
            repo_root,
            f"{base}..HEAD",
            allow_co_authors=policy.allow_co_authors,
            paths=policy.allowed_paths,
        )
    except CommitPolicyError as exc:
        return [Check(name="git:commit-hygiene", passed=False, detail=str(exc))]
    return [
        Check(
            name="git:commit-hygiene",
            passed=not findings,
            detail=(
                f"every commit since {base[:10]} under {', '.join(policy.allowed_paths) or 'the repo'} "
                f"is signed off by its author, with no co-author, symlink or placeholder"
                if not findings
                else f"{len(findings)} finding(s):\n  "
                + "\n  ".join(f.render() for f in findings[:12])
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
    commit_policy: CommitPolicy | None = None,
    anchor_pending: bool = False,
) -> DeliveryReport:
    """Run every mechanical delivery check and return the report."""
    root = Path(repo_root)
    checks: list[Check] = []

    modeling = root / target_rel / "modeling.py"
    entries: list[tuple[str, str]] = []
    model_ops: list[tuple[str, str, str]] = []
    if modeling.is_file():
        entries = imported_catalog_entries(modeling)
        model_ops = imported_model_ops(modeling)
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
    checks.extend(check_model_ops(root, target_rel, model_ops))
    checks.extend(check_model_op_tests(root, target_rel, model_ops))
    checks.extend(check_routing(root, family, target_rel))
    checks.extend(check_accuracy_reference(root, reference_yaml, model_key, pending=anchor_pending))
    checks.extend(check_test_registration(root, accuracy_test))
    checks.extend(check_tree_committed(root, (target_rel, MODELING_V2_PACKAGE, CATALOG_TESTS_DIR)))
    if commit_policy is not None:
        checks.extend(check_commit_hygiene(root, commit_policy))
    return DeliveryReport(checks=tuple(checks))


__all__ = [
    "ACCURACY_REFERENCES_DIR",
    "accuracy_test_path",
    "CATALOG_DIR",
    "CATALOG_TESTS_DIR",
    "Check",
    "DeliveryReport",
    "EntryContract",
    "MODELING_V2_PACKAGE",
    "MODEL_OPS_DIRNAME",
    "TEST_DB_DIR",
    "check_accuracy_reference",
    "check_catalog_entries",
    "check_catalog_tests",
    "check_commit_hygiene",
    "check_model_op_tests",
    "check_model_ops",
    "check_routing",
    "check_target",
    "check_test_registration",
    "check_tree_committed",
    "entry_contract",
    "imported_catalog_entries",
    "imported_model_ops",
    "model_ops_tests_dir",
    "verify_delivery",
]
