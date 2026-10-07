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
"""Update and patch steps of a patched FetchContent dependency.

Both run in the dependency's source directory:

* ``apply`` (PATCH_COMMAND) leaves the tree as the pinned sources plus the
  patch. It accepts a tree that already carries the patch, and replaces a
  previously applied, different version of it. CMake runs it after each update
  step of git dependencies, after each download of URL dependencies, and when
  the patch changes (the patch's hash is part of the command).
* ``update <ref>`` (UPDATE_COMMAND of git dependencies) moves the checkout to
  ``<ref>`` when the pin changed: it undoes the applied patch, checks out the
  pinned commit and updates the submodules if the clone initialized any.

Neither discards changes it did not make: when other local changes are in the
way, they fail and ask the user to resolve them. The applied patch is recorded
in the dependency's git directory, or in the source directory of URL
dependencies, so it can be undone after the patch file changed.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

RECORD_NAME = "trtllm-applied.patch"


class DependencyError(Exception):
    pass


def run(*cmd, check=True):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise DependencyError(f"`{' '.join(cmd)}` failed:\n{p.stdout}{p.stderr}".strip())
    return p


def patch(patch_file, *args):
    return run("patch", "-p1", "--batch", "--quiet", *args, "-i", str(patch_file), check=False)


def is_applied(patch_file):
    return patch(patch_file, "--reverse", "--force", "--dry-run").returncode == 0


def can_apply(patch_file):
    return patch(patch_file, "--forward", "--dry-run").returncode == 0


def git(*args, check=True):
    return run("git", *args, check=check).stdout.strip()


def record_path(require_git=False):
    """The record of the applied patch: in the git directory when the source
    directory is the root of its own git repository, otherwise in the source
    directory, or None if `require_git`. FetchContent sources live inside the
    build tree, which may itself be inside another git checkout."""
    top = git("rev-parse", "--show-toplevel", check=False)
    if not top or not Path(top).samefile(Path.cwd()):
        return None if require_git else Path.cwd() / f".{RECORD_NAME}"
    return Path(git("rev-parse", "--absolute-git-dir")) / RECORD_NAME


def conflict(what):
    return DependencyError(
        f"the source tree in {Path.cwd()} has local changes that conflict with {what}. "
        "Undo them or remove the directory, then reconfigure.")


def undo(patch_file):
    """Undoes `patch_file` if it is applied; fails if it is partly applied."""
    if is_applied(patch_file):
        if patch(patch_file, "--reverse").returncode != 0:
            raise DependencyError(f"failed to undo {patch_file}")
    elif not can_apply(patch_file):
        raise conflict("the applied patch")


def cmd_apply(patch_file):
    record = record_path()
    if record.exists() and record.read_bytes() != patch_file.read_bytes():
        undo(record)
        record.unlink()
    if not is_applied(patch_file):
        if not can_apply(patch_file):
            raise conflict(patch_file.name)
        if patch(patch_file, "--forward").returncode != 0:
            raise DependencyError(f"failed to apply {patch_file}")
    shutil.copyfile(patch_file, record)


def cmd_update(name, ref):
    record = record_path(require_git=True)
    if record is None:
        raise DependencyError(f"{Path.cwd()} is not the root of a git checkout")

    target = git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False)
    if not target:
        depth = ["--depth", "1"] if git("rev-parse", "--is-shallow-repository") == "true" else []
        git("fetch", "--quiet", *depth, "origin", ref)
        target = git("rev-parse", "--verify", "FETCH_HEAD^{commit}")
    if git("rev-parse", "HEAD") == target:
        return

    print(f"-- {name}: checking out {ref}", flush=True)
    if record.exists():
        undo(record)
        record.unlink()
    if git("status", "--porcelain", "--ignore-submodules=dirty"):
        raise DependencyError(f"the source tree in {Path.cwd()} has local changes. "
                              "Undo them or remove the directory, then reconfigure.")
    git("checkout", "--quiet", "--detach", target)
    modules = record.parent / "modules"
    if modules.is_dir() and any(modules.iterdir()):
        git("submodule", "update", "--quiet", "--init", "--recursive")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    modes = parser.add_subparsers(dest="mode", required=True)
    apply = modes.add_parser("apply", help="patch step")
    apply.add_argument("name", help="dependency name, used in messages")
    apply.add_argument("patch_file", type=Path)
    apply.add_argument("--patch-sha256",
                       help="hash of the patch file; ignored, it makes CMake rerun "
                       "the step when the patch changes")
    update = modes.add_parser("update", help="update step of git dependencies")
    update.add_argument("name", help="dependency name, used in messages")
    update.add_argument("ref", help="pinned git ref")
    args = parser.parse_args()
    try:
        if args.mode == "apply":
            cmd_apply(args.patch_file)
        else:
            cmd_update(args.name, args.ref)
    except DependencyError as e:
        print(f"{args.name}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
