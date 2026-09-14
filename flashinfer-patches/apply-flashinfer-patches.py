#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Apply this directory's FlashInfer header patches to the installed package.

FlashInfer compiles its attention kernels from headers that ship inside the
wheel (``flashinfer/data/include``) and caches the result per node, so a fix in
those headers only takes effect after the headers are edited in place and the
JIT cache is dropped.  A ``pip install`` of FlashInfer silently reverts the
edit, which is why the patches live here and are applied as a build step
instead of being carried as a modified copy of the package.

    apply-flashinfer-patches.py            # apply, then drop the JIT cache
    apply-flashinfer-patches.py --revert   # go back to the stock headers
    apply-flashinfer-patches.py --check    # report state, change nothing

The hunks are written against one FlashInfer release, so the patches live in a
directory named for it (``flashinfer-patches/0.6.16/``) and the directory is
chosen from the ``flashinfer-python`` pin in ``requirements.txt``.  A pin with
no directory is an error, not a skip: applying hunks derived from another
release, or silently applying none, both end in a build whose kernels are not
the ones the tree describes.

``--check`` exits 0 when every patch is applied, 1 when none are, and 2 when
the tree is half patched.

Exit code 4 means there is no top-level ``flashinfer`` package for this
interpreter, so there are no headers to patch.  That is the one outcome a build
step may treat as "nothing to do".  Every other non-zero exit says the headers
this build will compile are not the ones it thinks it has: ``patch`` rejected
some hunks and left a file half changed, or FlashInfer is installed but
unimportable, or its include directory is gone.

Whether a patch is already applied is decided by looking for the lines it adds,
not by asking ``patch`` to reverse it: GNU patch answers "Unreversed patch
detected!  Ignoring -R" and then applies the patch *forward*, exit code 0, so a
reverse dry run reports a stock tree as patched and ``--revert`` on a stock tree
would apply it.
"""

import argparse
import importlib.util
import pathlib
import re
import shutil
import subprocess
import sys
import typing

HERE = pathlib.Path(__file__).resolve().parent
REQUIREMENTS = HERE.parent / "requirements.txt"
PACKAGE = "flashinfer-python"

# Not an error: a build that never installs FlashInfer has nothing to patch.
# Kept distinct from 1 so a caller can tell that case from a patch that failed.
EXIT_ABSENT = 4


def absent(reason: str) -> typing.NoReturn:
    print(f"flashinfer is not installed here: {reason}", file=sys.stderr)
    raise SystemExit(EXIT_ABSENT)


def include_dir() -> pathlib.Path:
    """The include directory of the FlashInfer this interpreter imports.

    Exit 4, the one code the build helper treats as "nothing to do", is
    reserved for a single case: no top-level ``flashinfer`` package at all.
    Everything else fails.

    An installed but unimportable FlashInfer, or one whose include directory
    has gone missing, is not the same thing as an absent one: it says the
    headers this build compiles against are not the ones it thinks it has.
    Reporting either as benign let the build skip patching and emit a wheel
    with stock kernels while reporting success, which is the failure this
    script exists to prevent.
    """
    if importlib.util.find_spec("flashinfer") is None:
        absent("no top-level flashinfer package for this interpreter")
    try:
        import flashinfer.jit.env as jit_env
    except ImportError as exc:
        raise SystemExit(
            f"flashinfer is installed but cannot be imported, so its headers "
            f"cannot be patched: {exc}"
        )
    path = pathlib.Path(jit_env.FLASHINFER_INCLUDE_DIR)
    if not path.is_dir():
        raise SystemExit(
            f"flashinfer is installed but its include directory is missing, "
            f"so the kernels this build compiles would be stock: {path}"
        )
    return path


def jit_cache_dirs() -> list[pathlib.Path]:
    """Directories holding compiled kernels, which must go when headers change."""
    import flashinfer.jit.env as jit_env

    out = []
    for attr in ("FLASHINFER_JIT_DIR", "FLASHINFER_GEN_SRC_DIR"):
        path = getattr(jit_env, attr, None)
        if path is not None:
            out.append(pathlib.Path(path))
    return out


def pinned_version(requirements: pathlib.Path = REQUIREMENTS) -> str:
    """The FlashInfer version ``requirements.txt`` pins.

    Read the same way ``scripts/build_wheel.py`` reads its torch pin: find the
    line for the package and hand it to ``packaging.requirements.Requirement``.
    Only an exact ``==`` pin is accepted, because the version has to name a
    directory; a range does not identify one set of hunks.  ``packaging`` is not
    always importable (this script also runs under a bare system interpreter),
    so there is a plain-text fallback for that one line shape.
    """
    try:
        with open(requirements) as fp:
            line = next(
                (ln for ln in fp if ln.strip().startswith(PACKAGE)),
                None,
            )
    except FileNotFoundError:
        raise SystemExit(f"cannot read the FlashInfer pin: {requirements} does not exist")
    if line is None:
        raise SystemExit(f"no {PACKAGE} requirement in {requirements}")

    version = None
    try:
        try:
            from packaging.requirements import Requirement
        except (ImportError, ModuleNotFoundError):
            from pip._vendor.packaging.requirements import Requirement
    except (ImportError, ModuleNotFoundError):
        match = re.fullmatch(rf"{re.escape(PACKAGE)}==([^\s;#]+)", line.strip())
        if match:
            version = match.group(1)
    else:
        spec = list(Requirement(line.strip()).specifier)
        if len(spec) == 1 and spec[0].operator == "==":
            version = spec[0].version

    if version is None:
        raise SystemExit(
            f"the {PACKAGE} requirement in {requirements} is not a single '==' "
            f"pin, so it does not name one set of patches: {line.strip()}"
        )
    return version


def patch_dir(version: str, base: pathlib.Path = HERE) -> pathlib.Path:
    """The patch directory for a pinned version, or an error.

    Fail closed.  Falling back to another release's hunks would compile kernels
    nobody derived, and treating a missing directory as "no patches" would drop
    the mitigations from a build that reports success.
    """
    path = base / version
    if not path.is_dir():
        # Skip dotted and dunder names. Importing this file -- which the unit
        # test does -- leaves a __pycache__ beside the version directories, and
        # listing it reads as a FlashInfer release that does not exist, in the
        # one message a human sees when the build has just failed. A version
        # directory that exists but is empty still belongs in the list: that is
        # a different diagnosis, and patches() reports it.
        have = sorted(
            p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))
        )
        raise SystemExit(
            f"no FlashInfer patches for the pinned version {version}: {path} "
            f"does not exist. Patches exist for {have or 'no version'}; re-derive "
            f"the hunks against {version} (see {base}/README.md) before pinning it."
        )
    return path


def patches(directory: pathlib.Path) -> list[pathlib.Path]:
    found = sorted(directory.glob("[0-9][0-9][0-9][0-9]-*.patch"))
    if not found:
        raise SystemExit(f"no patch files in {directory}")
    return found


def fingerprint(patch: pathlib.Path) -> dict[str, list[str]]:
    """Per target file, the lines the patch adds that the stock text lacks.

    A line that appears on both sides of a hunk is unchanged in substance even
    though difflib emitted it as a delete plus an insert, so it says nothing
    about whether the patch is in the tree.
    """
    added: dict[str, list[str]] = {}
    removed: dict[str, set[str]] = {}
    target = None
    for line in patch.read_text().splitlines():
        if line.startswith("+++ "):
            target = line[4:].split("\t")[0]
            if target.startswith("b/"):
                target = target[2:]
            added.setdefault(target, [])
            removed.setdefault(target, set())
        elif target is None or line.startswith(("--- ", "@@")):
            continue
        elif line.startswith("+"):
            added[target].append(line[1:])
        elif line.startswith("-"):
            removed[target].add(line[1:].strip())
    return {
        name: [line for line in lines if line.strip() and line.strip() not in removed[name]]
        for name, lines in added.items()
    }


def applied(patch: pathlib.Path, root: pathlib.Path) -> bool:
    marks = fingerprint(patch)
    if not any(marks.values()):
        raise SystemExit(f"{patch.name} adds nothing recognisable; cannot judge state")
    for name, lines in marks.items():
        text = (root / name).read_text()
        for line in lines:
            if line not in text:
                return False
    return True


def run_patch(patch: pathlib.Path, root: pathlib.Path, revert: bool) -> int:
    # --fuzz=0: context must match exactly. Offsets are still allowed, so this
    # does not change what applies to the version the hunks were derived
    # against; it stops hunks from another release landing in the wrong place.
    argv = ["patch", "-p1", "--batch", "--no-backup-if-mismatch", "--fuzz=0"]
    if revert:
        argv.append("--reverse")
    argv += ["-i", str(patch)]
    try:
        done = subprocess.run(argv, cwd=root, capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit(
            "the 'patch' command is not installed, so the FlashInfer headers cannot be patched"
        )
    if done.returncode != 0:
        sys.stderr.write(done.stdout + done.stderr)
    return done.returncode


def drop_jit_cache() -> None:
    for path in jit_cache_dirs():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            print(f"dropped JIT cache {path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--revert", action="store_true", help="restore the stock FlashInfer headers")
    mode.add_argument("--check", action="store_true", help="report whether the patches are applied")
    mode.add_argument(
        "--drop-cache", action="store_true", help="only drop this rank's compiled-kernel cache"
    )
    ap.add_argument(
        "--keep-jit-cache",
        action="store_true",
        help="do not drop the compiled-kernel cache afterwards",
    )
    args = ap.parse_args()

    if args.drop_cache:
        # The cache is per rank (FLASHINFER_WORKSPACE_BASE carries the rank), so
        # after patching once run this under the same task count as the job that
        # will use the kernels; otherwise the other ranks replay stale cubins and
        # the patch silently does nothing.
        drop_jit_cache()
        return 0

    root = include_dir()
    version = pinned_version()
    directory = patch_dir(version)
    found = patches(directory)
    print(f"flashinfer headers: {root}")
    print(f"pinned version {version}, patches from {directory}")

    if args.check:
        state = {p.name: applied(p, root) for p in found}
        for name, is_on in state.items():
            print(f"  {'applied' if is_on else 'stock  '} {name}")
        if all(state.values()):
            return 0
        return 1 if not any(state.values()) else 2

    # Revert in reverse order: 0002's context assumes 0001 is in the tree.
    order = list(reversed(found)) if args.revert else found
    changed = False
    for patch in order:
        is_on = applied(patch, root)
        if is_on == args.revert:
            code = run_patch(patch, root, revert=args.revert)
            if code != 0:
                print(f"  FAILED ({code}) {patch.name}", file=sys.stderr)
                print(
                    "  patch fails a hunk at a time, so the headers may now be "
                    "half changed: --check reports the state and --revert "
                    "restores the stock text.",
                    file=sys.stderr,
                )
                return 1
            if applied(patch, root) == args.revert:
                print(
                    f"  patch reported success but the tree did not change: {patch.name}",
                    file=sys.stderr,
                )
                return 1
            changed = True
            print(f"  {'reverted' if args.revert else 'applied '} {patch.name}")
        else:
            print(f"  already {'stock  ' if args.revert else 'applied'} {patch.name}")

    if changed and not args.keep_jit_cache:
        drop_jit_cache()
    elif not changed:
        print("nothing to do; JIT cache left alone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
