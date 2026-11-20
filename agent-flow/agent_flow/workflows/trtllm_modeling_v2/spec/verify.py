# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The spec's own tests (``verify``): what else the target must pass, by feature, at a stage.

Every check of the contract runs at one of two stages:

- ``landing``: when a landing names it (the agent loop), and once more after the
  work if it has not passed;
- ``delivery``: after the work is done and every other required check has passed.

The flow's own checks take theirs from the contract (``run/validation/contract.py``).
``verify.tests`` adds tests of the tree, by the feature each one proves::

    verify:
      tests:
        core:
          e2e:
            purpose: GSM8K, a long-context subset and the reuse panel on one engine
            pytest: tests/integration/defs/accuracy/.../test_accuracy.py::TestX::test_e2e
            members: [tep4, dep4]   # a whole-model test: once per member, with its configuration
            stage: delivery
          moe_routing:
            purpose: the router picks the experts the HF reference picks
            pytest: tests/unittest/_torch/modeling_v2/test_moe_routing.py
            devices: 1              # a module test: once, on this many devices

A test is a pytest node of the tree, run from its pytest root as CI runs it, and
judged by its JUnit report: every selected test passed, at least one was
selected; a skipped one fails it. With ``members`` the test runs once per member
and is handed that member's configuration of its feature (:data:`HANDOVER_ENV`);
with ``devices`` (the default, one device) it runs once and is handed nothing.

A test is *defined* when the tree's sources hold its file, class and function
(:func:`defined`); nothing is imported, collected or run to say so.

This module reads the spec alone: the schema calls :func:`validate` with what it
knows of the spec's units and members.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

VERIFY_FIELD = "verify"
VERIFY_KEYS: tuple[str, ...] = ("tests",)

STAGE_LANDING = "landing"
STAGE_DELIVERY = "delivery"
STAGES: tuple[str, ...] = (STAGE_LANDING, STAGE_DELIVERY)

TEST_KEYS: tuple[str, ...] = (
    "purpose",
    "pytest",
    "members",
    "devices",
    "stage",
    "timeout",
    "env",
)
#: A test's name: it names the test's checks (``core:test[e2e]@tep4``).
TEST_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
#: ``members: all``: every member the feature is gated at.
ALL_MEMBERS = "all"
#: Seconds one pytest node may take when the test states none.
DEFAULT_TIMEOUT_S = 3600
#: In a node id, the member and the feature of the check that runs it.
MEMBER_PLACEHOLDER = "{member}"
FEATURE_PLACEHOLDER = "{feature}"

#: What a whole-model test is handed, as environment variables of its pytest process.
FEATURE_ENV = "MV2FLOW_FEATURE"
MEMBER_ENV = "MV2FLOW_MEMBER"
MODEL_ENV = "MV2FLOW_MODEL"
LLM_OPTIONS_ENV = "MV2FLOW_LLM_OPTIONS"
HANDOVER_ENV: tuple[str, ...] = (FEATURE_ENV, MEMBER_ENV, MODEL_ENV, LLM_OPTIONS_ENV)

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_GATE = re.compile(r"^test\[([A-Za-z0-9_.-]+)\]$")


@dataclass(frozen=True)
class UserTest:
    """One test of ``verify.tests``, as the spec declares it."""

    name: str
    #: The unit or variant it proves, and whose target and settings it runs with.
    feature: str
    purpose: str
    #: Repository-relative pytest node ids; each may name ``{member}`` and ``{feature}``.
    nodes: tuple[str, ...]
    stage: str = STAGE_LANDING
    #: Where a whole-model test runs: members, or :data:`ALL_MEMBERS`; ``()`` for a module test.
    members: tuple[str, ...] = ()
    #: The devices a module test runs on; ``0`` for a whole-model test (its member's width).
    devices: int = 0
    timeout: int = DEFAULT_TIMEOUT_S
    env: Mapping[str, str] | None = None

    @property
    def whole_model(self) -> bool:
        """Whether it runs per member with that member's configuration (``members``)."""
        return bool(self.members)

    @property
    def gate(self) -> str:
        return gate_name(self.name)

    def node_ids(self, member: str | None) -> tuple[str, ...]:
        """Its node ids at ``member``, the placeholders filled in."""
        return tuple(
            node.replace(MEMBER_PLACEHOLDER, str(member or "")).replace(
                FEATURE_PLACEHOLDER, self.feature
            )
            for node in self.nodes
        )


def gate_name(name: str) -> str:
    """The gate of test ``name``: ``test[<name>]``."""
    return f"test[{name}]"


def test_of_gate(gate: str) -> str | None:
    """The name of the test whose gate is ``gate``; ``None`` for any other gate."""
    match = _GATE.match(str(gate))
    return match.group(1) if match else None


def module_member(devices: int) -> str:
    """The member a module test's check is recorded at: ``tp<N>``, its devices."""
    return f"tp{int(devices)}"


def _nodes(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(v) for v in value)
    return ()


def _members(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(v) for v in value)
    return ()


def user_tests(data: Mapping[str, Any]) -> list[UserTest]:
    """The tests ``verify.tests`` declares, feature by feature, in declaration order.

    Reads a validated spec: an entry the schema would refuse is skipped.
    """
    block = data.get(VERIFY_FIELD)
    by_feature = block.get("tests") if isinstance(block, Mapping) else None
    if not isinstance(by_feature, Mapping):
        return []
    out: list[UserTest] = []
    for feature, entries in by_feature.items():
        if not isinstance(entries, Mapping):
            continue
        for name, entry in entries.items():
            if not isinstance(entry, Mapping) or not _nodes(entry.get("pytest")):
                continue
            members = _members(entry.get("members"))
            devices = 0 if members else int(entry.get("devices") or 1)
            out.append(
                UserTest(
                    name=str(name),
                    feature=str(feature),
                    purpose=" ".join(str(entry.get("purpose") or "").split()),
                    nodes=_nodes(entry.get("pytest")),
                    stage=str(entry.get("stage") or STAGE_LANDING),
                    members=members,
                    devices=devices,
                    timeout=int(entry.get("timeout") or DEFAULT_TIMEOUT_S),
                    env={str(k): str(v) for k, v in (entry.get("env") or {}).items()} or None,
                )
            )
    return out


def find(data: Mapping[str, Any], name: str) -> UserTest | None:
    """Test ``name`` of ``verify.tests``; ``None`` when the spec declares none of that name."""
    return next((t for t in user_tests(data) if t.name == name), None)


def validate(
    data: Mapping[str, Any],
    errors: list[str],
    *,
    features: set[str],
    member_error: Callable[[str], str | None],
) -> None:
    """``verify``: every test names a feature of the spec, a node of the tree and where it runs.

    ``features`` are the unit and variant names a test may prove; ``member_error``
    says what is wrong with a member's spelling, ``None`` when nothing is.
    """
    block = data.get(VERIFY_FIELD)
    if block is None:
        return
    if not isinstance(block, Mapping):
        errors.append(f"'{VERIFY_FIELD}' must be a mapping with the key 'tests'")
        return
    for key in block:
        if key not in VERIFY_KEYS:
            errors.append(
                f"'{VERIFY_FIELD}.{key}' is not a key of '{VERIFY_FIELD}' ({list(VERIFY_KEYS)})"
            )
    by_feature = block.get("tests")
    if by_feature is None:
        return
    if not isinstance(by_feature, Mapping):
        errors.append(
            f"'{VERIFY_FIELD}.tests' must map a feature (core, a feature or a variant) to its tests"
        )
        return
    seen: dict[str, str] = {}
    for feature, entries in by_feature.items():
        where = f"{VERIFY_FIELD}.tests.{feature}"
        if str(feature) not in features:
            errors.append(f"'{where}' names no feature of this spec ({sorted(features)})")
        if not isinstance(entries, Mapping) or not entries:
            errors.append(f"'{where}' must map a test's name to its entry")
            continue
        for name, entry in entries.items():
            _validate_test(str(feature), str(name), entry, errors, seen, member_error)


def _validate_test(
    feature: str,
    name: str,
    entry: Any,
    errors: list[str],
    seen: dict[str, str],
    member_error: Callable[[str], str | None],
) -> None:
    where = f"{VERIFY_FIELD}.tests.{feature}.{name}"
    if not TEST_NAME.match(name):
        errors.append(f"'{where}': a test's name is letters, digits, '_', '.' and '-'")
    elif name in seen:
        errors.append(
            f"'{where}': test `{name}` is also declared under `{seen[name]}`; a name is one test"
        )
    else:
        seen[name] = feature
    if not isinstance(entry, Mapping):
        errors.append(f"'{where}' must be a mapping with keys {list(TEST_KEYS)}")
        return
    for key in entry:
        if key not in TEST_KEYS:
            errors.append(f"'{where}.{key}' is not a key of a test ({list(TEST_KEYS)})")
    purpose = entry.get("purpose")
    if not isinstance(purpose, str) or not purpose.strip():
        errors.append(f"'{where}.purpose' must say what the test proves and what passing means")
    nodes = entry.get("pytest")
    listed = [nodes] if isinstance(nodes, str) else nodes
    if not isinstance(listed, list) or not listed:
        errors.append(f"'{where}.pytest' must be a pytest node id, or a nonempty list of them")
    else:
        for node in listed:
            if (
                not isinstance(node, str)
                or not node.strip()
                or Path(node.partition("::")[0]).is_absolute()
                or not node.partition("::")[0].endswith(".py")
            ):
                errors.append(
                    f"'{where}.pytest' must name repository-relative test files "
                    f"(`tests/.../test_x.py` or `tests/.../test_x.py::TestY::test_z`), got {node!r}"
                )
    members = entry.get("members")
    devices = entry.get("devices")
    if members is not None and devices is not None:
        errors.append(
            f"'{where}' sets both 'members' (a whole-model test, once per member) and "
            f"'devices' (a module test, once); say one"
        )
    if members is not None:
        listed_members = [members] if isinstance(members, str) else members
        if not isinstance(listed_members, list) or not listed_members:
            errors.append(f"'{where}.members' must be a member, a list of them, or '{ALL_MEMBERS}'")
        elif listed_members != [ALL_MEMBERS]:
            for member in listed_members:
                said = member_error(str(member)) if isinstance(member, str) else "not a string"
                if said:
                    errors.append(f"'{where}.members': {member!r}: {said}")
    if devices is not None and (
        isinstance(devices, bool) or not isinstance(devices, int) or devices < 1
    ):
        errors.append(f"'{where}.devices' must be an integer >= 1")
    if members is None and isinstance(listed, list):
        for node in listed:
            if isinstance(node, str) and MEMBER_PLACEHOLDER in node:
                errors.append(
                    f"'{where}.pytest' names {MEMBER_PLACEHOLDER}, which only a test with "
                    f"'members' has"
                )
    stage = entry.get("stage")
    if stage is not None and stage not in STAGES:
        errors.append(f"'{where}.stage' must be one of {list(STAGES)}, got {stage!r}")
    timeout = entry.get("timeout")
    if timeout is not None and (
        isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0
    ):
        errors.append(f"'{where}.timeout' must be a positive number of seconds")
    env = entry.get("env")
    if env is not None:
        if not isinstance(env, Mapping):
            errors.append(f"'{where}.env' must be a mapping from variable to value")
        else:
            for var, value in env.items():
                if not isinstance(var, str) or not _ENV_NAME.fullmatch(var):
                    errors.append(
                        f"'{where}.env' names {var!r}, which is not a shell variable name"
                    )
                elif var in HANDOVER_ENV:
                    errors.append(f"'{where}.env.{var}' is set by the flow for a whole-model test")
                if not isinstance(value, (str, int, float, bool)):
                    errors.append(f"'{where}.env.{var}' must be a scalar the shell can carry")


# ----------------------------------------------------------------------------- definition


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return None


def _imported(tree: ast.Module, path: Path, repo: Path, name: str) -> tuple[Path, str] | None:
    """Where module ``path`` imports ``name`` from: the file, and its name there; else ``None``."""
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        for alias in node.names:
            if (alias.asname or alias.name) != name:
                continue
            base = repo
            if node.level:
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
            target = base.joinpath(*node.module.split("."))
            for candidate in (target.with_suffix(".py"), target / "__init__.py"):
                if candidate.is_file():
                    return candidate, alias.name
    return None


def _binds(tree: ast.Module, name: str) -> bool:
    """Whether ``tree`` binds ``name`` at module level by an import or an assignment."""
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if any((alias.asname or alias.name.partition(".")[0]) == name for alias in node.names):
                return True
        elif isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return True
    return False


def _has_method(path: Path, repo: Path, cls: str, method: str, depth: int = 4) -> bool | None:
    """Whether class ``cls`` of module ``path`` has ``method``, its own or a base's.

    Bases are followed through the tree's sources: one the module defines, or
    imports from a module of the tree. ``None`` when one cannot be followed, so
    the method may be inherited from it.
    """
    tree = _parse(path)
    if tree is None or depth < 0:
        return None
    node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls), None)
    if node is None:
        source = _imported(tree, path, repo, cls)
        return _has_method(source[0], repo, source[1], method, depth - 1) if source else None
    if any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method
        for n in node.body
    ):
        return True
    unknown = False
    for base in node.bases:
        if not isinstance(base, ast.Name):
            unknown = True
            continue
        found = _has_method(path, repo, base.id, method, depth - 1)
        if found:
            return True
        unknown = unknown or found is None
    return None if unknown else False


def defined(repo: str | Path, node: str) -> tuple[bool | None, str]:
    """Whether the tree at ``repo`` defines pytest node ``node``, and what says so.

    Defined when its file holds its class and the class its function, its own or
    one a base of the tree gives it. A parametrization (``[tep4]``) is not looked
    at, and a base the sources do not show counts as giving it. ``None`` when the
    tree cannot be read for it.
    """
    file, _, rest = node.partition("::")
    path = Path(repo) / file
    if not file or not path.is_file():
        return False, f"{file or '(no file)'} is not in the tree"
    names = [part.partition("[")[0] for part in rest.split("::") if part]
    if not names:
        return True, file
    tree = _parse(path)
    if tree is None:
        return None, f"{file} could not be read as Python"
    first = names[0]
    local = any(
        isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == first
        for n in tree.body
    )
    if not local and not _binds(tree, first):
        return False, f"{file} defines no `{first}`"
    if len(names) > 1 and _has_method(path, Path(repo), first, names[1]) is False:
        return False, f"{file} defines no `{first}::{names[1]}`"
    return True, node


__all__ = [
    "ALL_MEMBERS",
    "DEFAULT_TIMEOUT_S",
    "FEATURE_ENV",
    "FEATURE_PLACEHOLDER",
    "HANDOVER_ENV",
    "LLM_OPTIONS_ENV",
    "MEMBER_ENV",
    "MEMBER_PLACEHOLDER",
    "MODEL_ENV",
    "STAGES",
    "STAGE_DELIVERY",
    "STAGE_LANDING",
    "TEST_KEYS",
    "TEST_NAME",
    "UserTest",
    "VERIFY_FIELD",
    "VERIFY_KEYS",
    "defined",
    "find",
    "gate_name",
    "module_member",
    "test_of_gate",
    "user_tests",
    "validate",
]
