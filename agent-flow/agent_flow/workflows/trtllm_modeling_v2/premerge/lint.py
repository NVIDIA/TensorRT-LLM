# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The lint pass: the repository's pre-commit hooks over what the branch changed.

Two passes over the same file set, the set CI's release check gives
pre-commit (``git diff --name-only --diff-filter=d <base> HEAD``):

1. **Pass 1** lets the formatters rewrite files in place.
2. **Pass 2** runs the same hooks again. Anything still failing is a real
   error no formatter can fix, and nothing is committed: the tree is
   restored and the formatting diff is kept as a patch.

When pass 2 is clean, every rewritten Python file is put through an **AST
proof** — parse it before and after and compare — so a "formatting" commit
cannot carry a semantic change unnoticed. Differences a formatter
legitimately makes are classified rather than waved through: import order,
an unused import removed, whitespace inside a docstring. Anything else is
``different`` and refuses the commit unless explicitly allowed. The proof's
counts go into the commit message.

The commit is staged by explicit path, signed off, and authored as the
repository's user, then audited with the same rules ``commit.sh`` enforces.

Environment it copes with, all measured:

- ``~/.cache/pre-commit`` may be read-only (an agent sandbox). The hook
  environments can stay there; only the store's database and lock need
  writing, so ``PRE_COMMIT_HOME`` points at a writable copy of ``db.db``.
- The ``type-check`` hook needs a Python that can import mypy and the
  repository's native modules; on a login node it cannot, so it is skipped
  there (and the skip is reported) unless asked otherwise.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..committer import CommitIdentity, audit_commits
from ._git import changed_files, dirty_tracked, git

DEFAULT_MESSAGE = "[None][chore] Apply pre-commit formatting"

#: Classes of change the AST proof accepts in a formatting commit.
FORMATTING_CLASSES: tuple[str, ...] = (
    "unchanged",
    "identical",
    "docstring-whitespace",
    "import-order",
    "import-removed",
    "not-python",
)


class LintError(RuntimeError):
    """The lint pass cannot start (a dirty tree, no pre-commit, ...)."""


# ------------------------------------------------------------------ AST proof


def _dump(tree: ast.AST) -> str:
    return ast.dump(tree, include_attributes=False)


def _normalize_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body[0].value.value = " ".join(body[0].value.value.split())
    return tree


def _explode_import(node: ast.Import | ast.ImportFrom) -> list[tuple[Any, ...]]:
    if isinstance(node, ast.Import):
        return [("import", a.name, a.asname) for a in node.names]
    return [("from", node.module, node.level, a.name, a.asname) for a in node.names]


def _pull_imports(tree: ast.AST) -> list[tuple[Any, ...]]:
    """Remove every import statement from ``tree`` and return them, exploded per name."""
    found: list[tuple[Any, ...]] = []
    for node in ast.walk(tree):
        for attr in ("body", "orelse", "finalbody"):
            stmts = getattr(node, attr, None)
            if not isinstance(stmts, list):
                continue
            kept = []
            for stmt in stmts:
                if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    found.extend(_explode_import(stmt))
                else:
                    kept.append(stmt)
            setattr(node, attr, kept)
    return found


def classify_python_change(before: str, after: str) -> str:
    """How ``after`` differs from ``before``, as far as Python is concerned.

    ``unchanged`` / ``identical`` (same AST) / ``docstring-whitespace`` /
    ``import-order`` / ``import-removed`` are what formatters do;
    ``different`` is anything else, and ``unparsable`` a file one side of
    which is not Python at all.
    """
    if before == after:
        return "unchanged"
    try:
        a, b = ast.parse(before), ast.parse(after)
    except SyntaxError:
        return "unparsable"
    if _dump(a) == _dump(b):
        return "identical"
    a, b = _normalize_docstrings(a), _normalize_docstrings(b)
    if _dump(a) == _dump(b):
        return "docstring-whitespace"
    imports_a, imports_b = _pull_imports(a), _pull_imports(b)
    if _dump(a) != _dump(b):
        return "different"
    if Counter(imports_a) == Counter(imports_b):
        return "import-order"
    if not Counter(imports_b) - Counter(imports_a):
        return "import-removed"
    return "different"


@dataclass
class AstProof:
    """The classification of every file a formatting pass rewrote."""

    entries: list[tuple[str, str]] = field(default_factory=list)

    @property
    def counts(self) -> Counter:
        return Counter(cls for _, cls in self.entries)

    @property
    def unexplained(self) -> list[tuple[str, str]]:
        return [(p, c) for p, c in self.entries if c not in FORMATTING_CLASSES]

    @property
    def ok(self) -> bool:
        return not self.unexplained

    def summary(self) -> str:
        return ", ".join(f"{n} {cls}" for cls, n in sorted(self.counts.items())) or "no files"

    def render(self) -> str:
        lines = [f"AST proof: {self.summary()}", ""]
        lines += [f"- `{path}`: {cls}" for path, cls in self.entries]
        return "\n".join(lines) + "\n"


def prove_formatting(repo: str | Path, paths: Sequence[str], *, rev: str = "HEAD") -> AstProof:
    """Compare each path at ``rev`` with its working-tree content."""
    proof = AstProof()
    for path in paths:
        if not path.endswith((".py", ".pyi")):
            proof.entries.append((path, "not-python"))
            continue
        before = git(repo, "show", f"{rev}:{path}", check=False)
        after_path = Path(repo) / path
        after = after_path.read_text(encoding="utf-8") if after_path.is_file() else ""
        proof.entries.append((path, classify_python_change(before, after)))
    return proof


# ------------------------------------------------------------------ the pass


@dataclass
class LintResult:
    status: str
    files: list[str] = field(default_factory=list)
    log_dir: Path | None = None
    rewritten: list[str] = field(default_factory=list)
    pass2_failures: list[str] = field(default_factory=list)
    proof: AstProof | None = None
    commit: str = ""
    skipped_hooks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in ("nothing", "clean", "committed", "dry-run")

    def render(self) -> str:
        lines = [f"lint: {self.status} ({len(self.files)} files, {len(self.rewritten)} rewritten)"]
        if self.skipped_hooks:
            lines.append(f"  skipped hooks: {', '.join(self.skipped_hooks)}")
        if self.pass2_failures:
            lines.append(f"  pass 2 failing hooks: {', '.join(self.pass2_failures)}")
        if self.proof is not None:
            lines.append(f"  {self.proof.render().splitlines()[0]}")
            for path, cls in self.proof.unexplained:
                lines.append(f"    NOT FORMATTING: {path} ({cls})")
        if self.commit:
            lines.append(f"  committed {self.commit[:10]}")
        lines += [f"  {note}" for note in self.notes]
        if self.log_dir is not None:
            lines.append(f"  logs: {self.log_dir}")
        return "\n".join(lines)


_FAILED_HOOK = re.compile(r"^(?P<name>.+?)\.{3,}.*Failed\s*$")


def failing_hooks(log_text: str) -> list[str]:
    """Hook names pre-commit reported as ``Failed``."""
    return [m.group("name").strip() for m in map(_FAILED_HOOK.match, log_text.splitlines()) if m]


def writable_pre_commit_home(log_dir: Path, *, home: Path | None = None) -> Path:
    """A ``PRE_COMMIT_HOME`` this process can write.

    The usual store when it is writable; otherwise a fresh directory holding a
    copy of the store's database, which still names the hook environments in
    the read-only store by absolute path — so nothing is re-installed.
    """
    store = (home or Path.home()) / ".cache" / "pre-commit"
    probe = store / ".mv2flow-write-probe"
    try:
        probe.touch()
    except OSError:
        pass
    else:
        probe.unlink(missing_ok=True)
        return store
    fallback = log_dir / "pre-commit-home"
    fallback.mkdir(parents=True, exist_ok=True)
    if (store / "db.db").is_file():
        shutil.copy2(store / "db.db", fallback / "db.db")
    return fallback


def _mypy_importable(python: str) -> bool:
    try:
        result = subprocess.run(
            [python, "-c", "import mypy.main"], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def run_lint(
    repo: str | Path,
    *,
    base: str,
    log_dir: str | Path,
    pre_commit: str = "pre-commit",
    dry_run: bool = False,
    message: str = DEFAULT_MESSAGE,
    skip_hooks: Sequence[str] = (),
    auto_skip_type_check: bool = True,
    python: str = "python3",
    allow_semantic_changes: bool = False,
    identity: CommitIdentity | None = None,
    runner: Runner = subprocess.run,
) -> LintResult:
    """Run both passes and, when the second is clean, commit the formatting.

    Refuses to start on a tree with modified tracked files: the formatting
    commit must contain only formatting.
    """
    repo = Path(repo)
    log = Path(log_dir)
    log.mkdir(parents=True, exist_ok=True)
    dirty = dirty_tracked(repo)
    if dirty:
        raise LintError(
            "tracked files are modified; commit or stash them first:\n  " + "\n  ".join(dirty)
        )
    files = changed_files(repo, base)
    (log / "files.txt").write_text("\n".join(files) + ("\n" if files else ""), encoding="utf-8")
    result = LintResult(status="nothing", files=files, log_dir=log)
    if not files:
        return result

    env = dict(os.environ)
    env["PRE_COMMIT_HOME"] = str(writable_pre_commit_home(log))
    skips = list(skip_hooks)
    if auto_skip_type_check and "type-check" not in skips and not _mypy_importable(python):
        skips.append("type-check")
        result.notes.append(
            "type-check skipped: mypy is not importable here; run it where the repository's "
            "Python is (a compute session)"
        )
    if skips:
        env["SKIP"] = ",".join(filter(None, [env.get("SKIP", ""), *skips]))
    result.skipped_hooks = skips

    def one_pass(n: int) -> tuple[int, str]:
        try:
            proc = runner(
                [pre_commit, "run", "--show-diff-on-failure", "--files", *files],
                cwd=str(repo),
                env=env,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise LintError(f"{pre_commit} is not runnable: {exc}") from exc
        text = (proc.stdout or "") + (proc.stderr or "")
        (log / f"pass{n}.log").write_text(text, encoding="utf-8")
        return proc.returncode, text

    one_pass(1)
    rewritten = [line for line in git(repo, "diff", "--name-only").splitlines() if line.strip()]
    result.rewritten = rewritten
    rc2, text2 = one_pass(2)

    def keep_patch_and_restore() -> None:
        (log / "formatting.patch").write_text(git(repo, "diff"), encoding="utf-8")
        current = [line for line in git(repo, "diff", "--name-only").splitlines() if line]
        if current:
            git(repo, "checkout", "--", *current)

    if rc2 != 0:
        result.pass2_failures = failing_hooks(text2) or [f"pre-commit exited {rc2}"]
        keep_patch_and_restore()
        result.status = "pass2-failed"
        return result
    # Pass 2 may itself have rewritten something pass 1 left (rare, but a
    # formatter that is not idempotent is exactly what this guards against).
    rewritten = [line for line in git(repo, "diff", "--name-only").splitlines() if line.strip()]
    result.rewritten = rewritten
    if not rewritten:
        result.status = "clean"
        return result

    proof = prove_formatting(repo, rewritten)
    result.proof = proof
    (log / "ast-proof.md").write_text(proof.render(), encoding="utf-8")
    if dry_run:
        keep_patch_and_restore()
        result.status = "dry-run"
        return result
    if not proof.ok and not allow_semantic_changes:
        keep_patch_and_restore()
        result.status = "semantic-change"
        return result

    body = (
        f"Output of the repository's pre-commit hooks over the {len(files)} files this "
        f"branch changes; {len(rewritten)} rewritten. No hand edits.\n\n"
        f"AST proof: {proof.summary()}."
    )
    commit_env = dict(os.environ)
    if identity is not None:
        commit_env.update(
            GIT_AUTHOR_NAME=identity.name,
            GIT_AUTHOR_EMAIL=identity.email,
            GIT_COMMITTER_NAME=identity.name,
            GIT_COMMITTER_EMAIL=identity.email,
        )
    git(repo, "add", "--", *rewritten)
    git(
        repo,
        "commit",
        "-s",
        "-q",
        "--no-verify",
        "-F",
        "-",
        "--",
        *rewritten,
        env=commit_env,
        input_=f"{message}\n\n{body}\n",
    )
    sha = git(repo, "rev-parse", "HEAD").strip()
    findings = audit_commits(repo, sha)
    if findings:
        git(repo, "reset", "-q", "--soft", "HEAD^")
        result.status = "hygiene-failed"
        result.notes += [f.render() for f in findings]
        return result
    result.commit = sha
    result.status = "committed"
    return result


__all__ = [
    "AstProof",
    "DEFAULT_MESSAGE",
    "FORMATTING_CLASSES",
    "LintError",
    "LintResult",
    "classify_python_change",
    "failing_hooks",
    "prove_formatting",
    "run_lint",
    "writable_pre_commit_home",
]
