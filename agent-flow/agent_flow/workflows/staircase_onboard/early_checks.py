# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delivery checks that need no finished target, run early, as warnings.

A run whose every unit is green can still fail delivery on things no unit gate
looks at: a catalog test under a name the CI selector does not match, no
integration accuracy test for the model, no test-db line that runs it. Found
at delivery, each costs a turn and -- on the run this was written for -- hours
of an idle run. None of them needs the target to be finished, so this module
runs them at the end of the plan phase and after every unit, and reports them
as **warnings**: a run is never stopped here, it is told early.

It also answers a question delivery never asked: **will CI actually run the
accuracy test, or skip it?** The test is read statically for what it needs --
a checkpoint under ``LLM_MODELS_ROOT``, environment variables, a GPU count, an
architecture -- and compared with the test-db stages that list it (pre- or
post-merge, which GPUs, how many). What it needs is emitted as the
**required CI assets** list, because a test that skips on every CI machine
reads green forever.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .delivery import (
    ACCURACY_TESTS_DIR,
    CATALOG_DIR,
    CATALOG_TESTS_DIR,
    MODEL_OPS_DIRNAME,
    TEST_DB_DIR,
)

LEVEL_OK = "ok"
LEVEL_WARN = "warn"
LEVEL_INFO = "info"

MODELS_ROOT_ENV = "LLM_MODELS_ROOT"
_MODELS_ROOT_CALL = "llm_models_root"

#: Which GPU names in a test-db wildcard serve an architecture. Only used to
#: say "this stage can never satisfy the test's architecture skip".
ARCH_GPUS: Mapping[str, tuple[str, ...]] = {
    "sm_80": ("a100", "a30", "a10"),
    "sm_89": ("l40", "l4"),
    "sm_90": ("h100", "h200", "gh200", "h20"),
    "sm_100": ("b200", "gb200"),
    "sm_103": ("b300", "gb300"),
    "sm_120": ("rtx", "gb202", "gb203"),
    "sm_121": ("gb10",),
}
_ARCH_FROM_SKIP = {"blackwell": "sm_100", "hopper": "sm_90", "ada": "sm_89"}


@dataclass(frozen=True)
class EarlyFinding:
    name: str
    level: str
    detail: str

    def render(self) -> str:
        return f"[{self.level}] {self.name}: {self.detail}"


@dataclass(frozen=True)
class CiAsset:
    """One thing CI must provide, or the accuracy test skips."""

    kind: str
    what: str
    source: str
    note: str = ""

    def render(self) -> str:
        note = f" -- {self.note}" if self.note else ""
        return f"- **{self.kind}**: {self.what} (`{self.source}`){note}"


@dataclass(frozen=True)
class CiListing:
    """One test-db block that lists the accuracy test."""

    file: str
    stage: str
    gpus: tuple[str, ...]
    gpu_count: tuple[int | None, int | None]


@dataclass(frozen=True)
class EarlyReport:
    stage: str
    findings: tuple[EarlyFinding, ...] = ()
    assets: tuple[CiAsset, ...] = ()
    listings: tuple[CiListing, ...] = field(default_factory=tuple)

    @property
    def warnings(self) -> tuple[EarlyFinding, ...]:
        return tuple(f for f in self.findings if f.level == LEVEL_WARN)

    def render(self) -> str:
        lines = [
            f"# Early delivery checks ({self.stage})",
            "",
            "Nothing here stops the run. A `warn` line is something delivery fails on, or "
            "something CI will skip without saying so; fix it now rather than at the end.",
            "",
            *(f"- {finding.render()}" for finding in self.findings),
            "",
            "## Required CI assets",
            "",
            "What CI must provide for the accuracy test to RUN rather than skip:",
            "",
        ]
        lines += [asset.render() for asset in self.assets] or ["- (no accuracy test to read yet)"]
        if self.listings:
            lines += ["", "## Where CI runs it", ""]
            for listing in self.listings:
                low, high = listing.gpu_count
                lines.append(
                    f"- `{listing.file}`: stage {listing.stage or '?'}, GPUs "
                    f"{', '.join(listing.gpus) or '*'}, count {low or '?'}..{high or '?'}"
                )
        return "\n".join(lines) + "\n"


# ------------------------------------------------------------ catalog tests


def vocabulary_entries(vocabulary_path: Path | str, catalog_root: Path) -> list[tuple[str, str]]:
    """``(category, name)`` for every shared entry the plan names, resolved against the catalog.

    A target's own entries (``ops/<category>/<name>``) are left out: their tests
    sit under the target's path in the tests tree, which delivery checks once
    the target exists.
    """
    path = Path(vocabulary_path)
    if not path.is_file():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return []
    if not isinstance(data, dict):
        return []
    named: list[str] = []
    for step in data.get("steps") or ():
        if isinstance(step, dict) and step.get("entry"):
            named.append(str(step["entry"]))
    for row in data.get("queue") or ():
        if isinstance(row, dict) and row.get("target"):
            named.append(str(row["target"]))
    entries: set[tuple[str, str]] = set()
    for raw in named:
        text = raw.strip().removesuffix(".py")
        if text.startswith(f"{MODEL_OPS_DIRNAME}/") or f"/{MODEL_OPS_DIRNAME}/" in text:
            continue
        if "catalog/" in text:
            text = text.split("catalog/", 1)[1]
        if "/" in text:
            category, name = text.rsplit("/", 1)
            entries.add((category.rsplit("/", 1)[-1], name))
            continue
        # A bare name: find its category in the catalog, if it exists yet.
        matches = sorted(catalog_root.glob(f"*/{text}.py")) if catalog_root.is_dir() else []
        if len(matches) == 1:
            entries.add((matches[0].parent.name, text))
    return sorted(entries)


def check_test_naming(repo_root: Path, entries: Iterable[tuple[str, str]]) -> list[EarlyFinding]:
    """Every planned entry's GPU test is (or will be) under the name CI matches."""
    findings: list[EarlyFinding] = []
    tests_root = repo_root / CATALOG_TESTS_DIR
    for category, name in entries:
        expected = tests_root / category / f"test_modeling_v2_{name}.py"
        if expected.is_file():
            continue
        others = (
            sorted(
                p.name
                for p in (tests_root / category).glob(f"test_*{name}*.py")
                if p.name != expected.name
            )
            if (tests_root / category).is_dir()
            else []
        )
        if others:
            findings.append(
                EarlyFinding(
                    name=f"test-name:{category}/{name}",
                    level=LEVEL_WARN,
                    detail=(
                        f"tested as {', '.join(others)}; delivery and the CI selector expect "
                        f"{CATALOG_TESTS_DIR}/{category}/{expected.name}"
                    ),
                )
            )
        elif (repo_root / CATALOG_DIR / category / f"{name}.py").is_file():
            findings.append(
                EarlyFinding(
                    name=f"test-name:{category}/{name}",
                    level=LEVEL_WARN,
                    detail=f"the entry exists and has no test at {CATALOG_TESTS_DIR}/{category}/{expected.name}",
                )
            )
    return findings


# ------------------------------------------------------------ accuracy test


def accuracy_test_path(repo_root: Path, family: str) -> Path:
    return repo_root / ACCURACY_TESTS_DIR / f"test_modeling_v2_{family}.py"


def find_test_db_listings(repo_root: Path, stem: str) -> list[CiListing]:
    """Every test-db block whose tests name ``stem``."""
    listings: list[CiListing] = []
    db = repo_root / TEST_DB_DIR
    if not db.is_dir():
        return listings
    for path in sorted(db.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        if stem not in text:
            continue
        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError:
            listings.append(
                CiListing(file=path.name, stage="unparsable", gpus=(), gpu_count=(None, None))
            )
            continue
        # `{l0_x: [block, ...]}` is the house shape; a bare list of blocks is
        # read the same way.
        groups = list(data.values()) if isinstance(data, dict) else [data]
        for blocks in groups:
            for block in blocks if isinstance(blocks, list) else ():
                if not isinstance(block, dict):
                    continue
                tests = [str(t) for t in block.get("tests") or ()]
                if not any(stem in t for t in tests):
                    continue
                condition = block.get("condition") or {}
                terms = condition.get("terms") or {}
                wildcards = condition.get("wildcards") or {}
                count = (condition.get("ranges") or {}).get("system_gpu_count") or {}
                listings.append(
                    CiListing(
                        file=path.name,
                        stage=str(terms.get("stage", "")),
                        gpus=tuple(str(g) for g in wildcards.get("gpu") or ()),
                        gpu_count=(count.get("gte"), count.get("lte")),
                    )
                )
    return listings


def _models_root_suffix(node: ast.AST) -> str | None:
    """The path a checkpoint expression appends to ``llm_models_root()``, if it is one."""

    def is_root(expr: ast.AST) -> bool:
        return isinstance(expr, ast.Call) and _call_name(expr.func) == _MODELS_ROOT_CALL

    if isinstance(node, ast.JoinedStr):
        parts = node.values
        for index, part in enumerate(parts):
            if isinstance(part, ast.FormattedValue) and is_root(part.value):
                rest = "".join(p.value for p in parts[index + 1 :] if isinstance(p, ast.Constant))
                return rest.lstrip("/")
    if isinstance(node, ast.Call) and _call_name(node.func) == "join" and node.args:
        if is_root(node.args[0]):
            return "/".join(str(a.value) for a in node.args[1:] if isinstance(a, ast.Constant))
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add)):
        left = node.left
        if isinstance(left, ast.Call) and _call_name(left.func) in ("Path",) and left.args:
            left = left.args[0]
        if is_root(left) and isinstance(node.right, ast.Constant):
            return str(node.right.value).lstrip("/")
    return None


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _sm_requirement(node: ast.AST) -> str | None:
    """``get_sm_version() != 103`` (a skip condition) -> ``sm_103``."""
    for child in ast.walk(node):
        if not isinstance(child, ast.Compare) or len(child.comparators) != 1:
            continue
        left, right = child.left, child.comparators[0]
        if isinstance(left, ast.Call) and _call_name(left.func) == "get_sm_version":
            if isinstance(right, ast.Constant) and isinstance(right.value, int):
                if isinstance(child.ops[0], ast.NotEq):
                    return f"sm_{right.value}"
                if isinstance(child.ops[0], ast.Lt):
                    return f"sm_{right.value}+"
    return None


def analyze_skips(test_file: Path) -> list[CiAsset]:
    """What the accuracy test needs from CI, read without importing it."""
    rel = str(test_file)
    try:
        source = test_file.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError) as exc:
        return [CiAsset(kind="unreadable", what=str(exc), source=rel)]
    assets: list[CiAsset] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, what: str, line: int, note: str = "") -> None:
        if (kind, what) in seen:
            return
        seen.add((kind, what))
        assets.append(CiAsset(kind=kind, what=what, source=f"{test_file.name}:{line}", note=note))

    for node in ast.walk(tree):
        suffix = (
            _models_root_suffix(node)
            if isinstance(node, (ast.JoinedStr, ast.Call, ast.BinOp))
            else None
        )
        if suffix is not None:
            add(
                "checkpoint",
                f"${MODELS_ROOT_ENV}/{suffix}",
                node.lineno,
                "the test's model path; absent on a CI machine, the load fails or the case skips",
            )
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name == "skipif" and node.args:
                reason = next(
                    (
                        kw.value.value
                        for kw in node.keywords
                        if kw.arg == "reason" and isinstance(kw.value, ast.Constant)
                    ),
                    "",
                )
                arch = _sm_requirement(node.args[0])
                if arch:
                    add("arch", arch, node.lineno, str(reason))
                else:
                    add("skip", ast.unparse(node.args[0]), node.lineno, str(reason))
            elif name in ("skip_less_device", "skip_less_mpi_world_size") and node.args:
                count = node.args[0]
                if isinstance(count, ast.Constant):
                    add("gpus", f">= {count.value}", node.lineno)
            elif name == "skip_less_device_memory" and node.args:
                memory = node.args[0]
                if isinstance(memory, ast.Constant):
                    add("gpu memory", f">= {memory.value} MiB", node.lineno)
            elif name == "skip" and isinstance(node.func, ast.Attribute):
                reason = (
                    node.args[0].value
                    if node.args and isinstance(node.args[0], ast.Constant)
                    else ""
                )
                add("skip", "pytest.skip(...) at runtime", node.lineno, str(reason))
            elif name in ("getenv", "get") and node.args and isinstance(node.args[0], ast.Constant):
                owner = node.func.value if isinstance(node.func, ast.Attribute) else None
                if name == "getenv" or (
                    isinstance(owner, ast.Attribute) and owner.attr == "environ"
                ):
                    add("env", str(node.args[0].value), node.lineno)
        elif isinstance(node, ast.Subscript):
            if (
                isinstance(node.value, ast.Attribute)
                and node.value.attr == "environ"
                and isinstance(node.slice, ast.Constant)
            ):
                add("env", str(node.slice.value), node.lineno)
        elif isinstance(node, ast.Name) and node.id.startswith(
            ("skip_pre_", "skip_no_", "skip_post_")
        ):
            if isinstance(node.ctx, ast.Load):
                generation = node.id.rsplit("_", 1)[-1]
                arch = _ARCH_FROM_SKIP.get(generation)
                if node.id.startswith("skip_pre_") and arch:
                    add("arch", f"{arch}+", node.lineno, node.id)
                else:
                    add("skip", node.id, node.lineno)
    return assets


def _arch_served(arch: str, gpus: Sequence[str]) -> bool | None:
    """Whether a listing's GPU wildcards can satisfy an arch requirement; None = unknown."""
    base = arch.rstrip("+")
    names = ARCH_GPUS.get(base)
    if not names or not gpus:
        return None
    if arch.endswith("+"):
        # At least this arch: anything listed at or above it.
        newer = [a for a in ARCH_GPUS if _arch_number(a) >= _arch_number(base)]
        names = tuple(n for a in newer for n in ARCH_GPUS[a])
    return any(any(name in g.lower() for name in names) for g in gpus)


def _arch_number(arch: str) -> int:
    digits = re.sub(r"\D", "", arch)
    return int(digits) if digits else 0


def check_ci_skip(
    repo_root: Path,
    family: str,
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[list[EarlyFinding], list[CiAsset], list[CiListing]]:
    """Will CI run the accuracy test? Findings, the required assets, the listings."""
    test = accuracy_test_path(repo_root, family)
    findings: list[EarlyFinding] = []
    if not test.is_file():
        findings.append(
            EarlyFinding(
                name="tests:accuracy-file",
                level=LEVEL_WARN,
                detail=(
                    f"no {ACCURACY_TESTS_DIR}/{test.name}: delivery requires the model's "
                    f"integration accuracy test; write it with the unit, not after the last one"
                ),
            )
        )
        return findings, [], []
    findings.append(EarlyFinding(name="tests:accuracy-file", level=LEVEL_OK, detail=str(test)))

    listings = find_test_db_listings(repo_root, test.stem)
    if not listings:
        findings.append(
            EarlyFinding(
                name="tests:test-db",
                level=LEVEL_WARN,
                detail=f"{test.stem} appears in no {TEST_DB_DIR}/*.yml; CI would never run it",
            )
        )
    else:
        stages = {listing.stage for listing in listings}
        level = LEVEL_OK
        detail = ", ".join(sorted({f"{item.file} ({item.stage or '?'})" for item in listings}))
        if stages and stages <= {"post_merge"}:
            level = LEVEL_WARN
            detail += "; post-merge only, so no pre-merge run ever executes it"
        findings.append(EarlyFinding(name="tests:test-db", level=level, detail=detail))

    assets = analyze_skips(test)
    for listing in listings:
        assets.append(
            CiAsset(
                kind="stage",
                what=f"{listing.file}: {listing.stage or '?'}",
                source=f"{TEST_DB_DIR}/{listing.file}",
                note=f"GPUs {', '.join(listing.gpus) or '*'}",
            )
        )

    environment = dict(os.environ)
    environment.update({str(k): str(v) for k, v in (env or {}).items()})
    models_root = environment.get(MODELS_ROOT_ENV)
    for asset in assets:
        if asset.kind == "checkpoint":
            if models_root:
                local = Path(models_root) / asset.what.split("/", 1)[1]
                level = LEVEL_OK if local.exists() else LEVEL_WARN
                detail = (
                    f"{local} exists here"
                    if local.exists()
                    else f"{local} does not exist under this host's {MODELS_ROOT_ENV}; "
                    f"CI's model store must hold it or every case skips or fails"
                )
            else:
                level = LEVEL_INFO
                detail = (
                    f"{MODELS_ROOT_ENV} is not set here, so whether CI's model store holds "
                    f"{asset.what} cannot be checked -- confirm it before merge"
                )
            findings.append(EarlyFinding(name="ci:checkpoint", level=level, detail=detail))
        elif asset.kind == "arch":
            for listing in listings:
                served = _arch_served(asset.what, listing.gpus)
                if served is False:
                    findings.append(
                        EarlyFinding(
                            name="ci:arch",
                            level=LEVEL_WARN,
                            detail=(
                                f"the test needs {asset.what} ({asset.source}) and "
                                f"{listing.file} lists GPUs {', '.join(listing.gpus)}: it "
                                f"always skips there"
                            ),
                        )
                    )
        elif asset.kind == "gpus":
            wanted = int(re.sub(r"\D", "", asset.what) or 0)
            for listing in listings:
                high = listing.gpu_count[1]
                if isinstance(high, int) and high < wanted:
                    findings.append(
                        EarlyFinding(
                            name="ci:gpus",
                            level=LEVEL_WARN,
                            detail=(
                                f"the test needs {asset.what} GPUs and {listing.file} runs "
                                f"on at most {high}: it always skips there"
                            ),
                        )
                    )
        elif asset.kind == "env" and asset.what not in environment:
            findings.append(
                EarlyFinding(
                    name=f"ci:env:{asset.what}",
                    level=LEVEL_WARN,
                    detail=(
                        f"the test reads ${asset.what} ({asset.source}); it is unset here, "
                        f"and CI must provide it"
                    ),
                )
            )
    return findings, assets, listings


# -------------------------------------------------------------------- run


def run_early_checks(
    *,
    repo_root: Path | str,
    family: str,
    stage: str,
    vocabulary_path: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    capacity: Any = None,
    certification: Any = None,
    workspace: Path | str | None = None,
) -> EarlyReport:
    """Every delivery check that needs no finished target, as warnings.

    ``capacity`` is a :class:`~.capacity_audit.CapacityAudit` and
    ``certification`` a :class:`~.envelope.CertificationPlan`; when given,
    their delivery failures are reported here early as well.
    """
    root = Path(repo_root)
    findings: list[EarlyFinding] = []
    entries = vocabulary_entries(vocabulary_path, root / CATALOG_DIR) if vocabulary_path else []
    findings.extend(check_test_naming(root, entries))

    ci_findings, assets, listings = check_ci_skip(root, family, env=env)
    findings.extend(ci_findings)

    if capacity is not None:
        for finding in capacity.failures:
            findings.append(
                EarlyFinding(
                    name=f"capacity:{finding.name}",
                    level=LEVEL_WARN,
                    detail=f"{finding.file}:{finding.line}: {finding.detail}",
                )
            )
    if certification is not None and workspace is not None:
        from .envelope import check_external_evidence

        for check in check_external_evidence(certification, workspace):
            if not check.passed:
                findings.append(
                    EarlyFinding(name=check.name, level=LEVEL_WARN, detail=check.detail)
                )
    if vocabulary_path:
        from .smith_brief import load_steps, parse_call_sites

        for step in load_steps(vocabulary_path):
            for error in parse_call_sites(step)[1]:
                findings.append(
                    EarlyFinding(name="vocabulary:call_sites", level=LEVEL_WARN, detail=error)
                )
    return EarlyReport(
        stage=stage,
        findings=tuple(findings),
        assets=tuple(assets),
        listings=tuple(listings),
    )


__all__ = [
    "ARCH_GPUS",
    "CiAsset",
    "EarlyFinding",
    "EarlyReport",
    "LEVEL_INFO",
    "LEVEL_OK",
    "LEVEL_WARN",
    "MODELS_ROOT_ENV",
    "CiListing",
    "accuracy_test_path",
    "analyze_skips",
    "check_ci_skip",
    "check_test_naming",
    "run_early_checks",
    "find_test_db_listings",
    "vocabulary_entries",
]
