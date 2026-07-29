#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
FetchContent cache wrapper — injected by CMake as GIT_EXECUTABLE.

Intercepts ``git clone`` and ``git submodule update --init`` to inject
``--reference <cache>/{shallow,full}/<repo>.git``.  Everything else is
``execv``'d through to the real git.  Cache writes are driven by the
``FetchContent_MakeAvailable`` override in ``3rdparty/CMakeLists.txt``,
not by this wrapper.

End-to-end flow and design rationale live in ``3rdparty/fetch-cache.md``.

Environment (set by CMake):
    _TRTLLM_REAL_GIT            Absolute path to the real git binary
    TRTLLM_FETCHCONTENT_CACHE   Path to the cache directory
"""

import argparse
import os
import subprocess
import sys

REAL_GIT = os.environ.get("_TRTLLM_REAL_GIT", "/usr/bin/git")
CACHE_DIR = os.environ.get("TRTLLM_FETCHCONTENT_CACHE", "")
TAG = "-- [fetch-cache]"


# ---------------------------------------------------------------------------
# strict argparse
# ---------------------------------------------------------------------------

class _ParseError(Exception):
    pass


class _StrictParser(argparse.ArgumentParser):
    """``ArgumentParser`` that raises on unrecognized args instead of
    calling ``sys.exit``."""

    def error(self, message):
        raise _ParseError(message)


def _make_clone_parser() -> _StrictParser:
    """Known cmake FetchContent ``git clone`` args."""
    p = _StrictParser(prog="git clone")
    p.add_argument("--reference")
    p.add_argument("--no-checkout", action="store_true")
    p.add_argument("--depth", type=int)
    p.add_argument("--no-single-branch", action="store_true")
    p.add_argument("--config", action="append", default=[])
    p.add_argument("url")
    p.add_argument("directory", nargs="?")
    return p


def _make_submodule_update_parser() -> _StrictParser:
    """Known cmake FetchContent ``git submodule update`` args."""
    p = _StrictParser(prog="git submodule update")
    p.add_argument("--init", action="store_true")
    p.add_argument("--recursive", action="store_true")
    return p


_clone_parser = _make_clone_parser()
_submod_parser = _make_submodule_update_parser()


# ---------------------------------------------------------------------------
# subcommand detection
# ---------------------------------------------------------------------------

# Git global options that consume the next argument as a value.
# Everything else starting with '-' is treated as a flag (no value).
# This set is small, stable, and only needed to find the subcommand position.
_GIT_GLOBAL_OPTS_WITH_VALUE = frozenset({
    "-c", "-C", "--git-dir", "--work-tree",
    "--exec-path", "--namespace", "--super-prefix",
})


def _find_subcommand(args: list[str]) -> tuple[list[str], str | None, list[str]]:
    """Split ``[global-opts] <subcmd> [subcmd-args]``.

    Returns ``(global_args, subcmd_or_None, rest)``.
    """
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in _GIT_GLOBAL_OPTS_WITH_VALUE:
            i += 2  # skip option + its value
        elif arg.startswith("-"):
            i += 1  # flag or --key=value (self-contained)
        else:
            return args[:i], arg, args[i + 1:]
    return args, None, []


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def passthrough():
    os.execv(REAL_GIT, [REAL_GIT] + sys.argv[1:])


def repo_name(url: str) -> str:
    return os.path.basename(url.rstrip("/")).removesuffix(".git")


def lookup_cache(url: str, shallow: bool) -> str | None:
    """Look up a reference bare for *url*, routing by consumer shape.

    Split-pool rationale is in ``3rdparty/fetch-cache.md`` (I5).  A miss
    returns None; the caller passes through to a plain clone, so
    correctness is unaffected — just no speedup.
    """
    subdir = "shallow" if shallow else "full"
    ref = os.path.join(CACHE_DIR, subdir, f"{repo_name(url)}.git")
    if os.path.isfile(os.path.join(ref, "HEAD")):
        return ref
    return None


def _strict_parse(parser: _StrictParser, args: list[str], context: str):
    """Parse *args* or raise ``RuntimeError`` with an actionable message
    that points the reader at the exact parser to extend."""
    try:
        return parser.parse_args(args)
    except _ParseError as exc:
        builder = ("_make_clone_parser" if context == "git clone"
                   else "_make_submodule_update_parser")
        raise RuntimeError(
            "\n".join([
                "",
                f"fetch_cache_wrapper: unrecognized `{context}` argument(s).",
                "  cmake's calling convention likely changed.  The wrapper",
                "  uses a strict allowlist on purpose so unknown args are",
                "  never silently mis-parsed as a clone URL or flag.",
                "",
                f"  args:  {args}",
                f"  error: {exc}",
                "",
                "How to fix — extend the allowlist in this file:",
                f"  {os.path.abspath(__file__)}",
                f"  Edit `{builder}` and register the new argument(s), e.g.:",
                "    p.add_argument(\"--filter\")                          "
                "# option taking a value",
                "    p.add_argument(\"--progress\", action=\"store_true\")    "
                "# boolean flag",
                "    p.add_argument(\"--config\", action=\"append\", "
                "default=[])  # repeatable",
                "",
                "Temporary workaround (disables the cache for this build):",
                "  TRTLLM_FETCHCONTENT_CACHE= <your build command>",
            ])
        ) from None


# ---------------------------------------------------------------------------
# clone
# ---------------------------------------------------------------------------

def handle_clone(global_args: list[str], clone_args: list[str]):
    ns = _strict_parse(_clone_parser, clone_args, "git clone")

    if ns.reference:
        passthrough()

    # --depth is cmake FetchContent's GIT_SHALLOW TRUE signal.
    shallow = ns.depth is not None
    ref = lookup_cache(ns.url, shallow)
    if not ref:
        passthrough()

    new = global_args + ["clone", "--reference", ref] + clone_args
    print(f"{TAG} Using reference: {ref}", file=sys.stderr)
    os.execv(REAL_GIT, [REAL_GIT] + new)


# ---------------------------------------------------------------------------
# submodule update --init
# ---------------------------------------------------------------------------

def handle_submodule(global_args: list[str], sub_args: list[str]):
    if not sub_args or sub_args[0] != "update":
        passthrough()

    update_args = sub_args[1:]
    ns = _strict_parse(_submod_parser, update_args, "git submodule update")

    if not ns.init:
        passthrough()

    # ── Intercept: per-submodule update with --reference ──

    toplevel = subprocess.run(
        [REAL_GIT, "rev-parse", "--show-toplevel"],
        capture_output=True, text=True,
    )
    if toplevel.returncode != 0:
        passthrough()
    top = toplevel.stdout.strip()

    gitmodules = os.path.join(top, ".gitmodules")
    if not os.path.isfile(gitmodules):
        passthrough()

    # Per-path subprocess calls below pin cwd=top so the worktree is
    # anchored regardless of the caller's CWD.

    # Parse .gitmodules
    r = subprocess.run(
        [REAL_GIT, "config", "--file", gitmodules,
         "--get-regexp", r"submodule\..*\.path"],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        passthrough()

    updated: list[str] = []
    for line in r.stdout.strip().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        key, sub_path = parts
        section = key.rsplit(".path", 1)[0]

        r2 = subprocess.run(
            [REAL_GIT, "config", "--file", gitmodules,
             "--get", f"{section}.url"],
            capture_output=True, text=True,
        )
        if r2.returncode != 0:
            continue
        sub_url = r2.stdout.strip()

        # shallow=False; submodules always route to full/.  See
        # fetch-cache.md "Cache layout".
        ref = lookup_cache(sub_url, shallow=False)
        cmd = [REAL_GIT, "submodule", "update", "--init"]
        if ref:
            cmd += ["--reference", ref]
            print(f"{TAG} submodule {sub_path}: using reference {ref}",
                  file=sys.stderr)
        cmd += ["--", sub_path]

        ret = subprocess.run(cmd, cwd=top).returncode
        # Fallback without --reference if it failed
        if ret != 0 and ref:
            subprocess.run(
                [REAL_GIT, "submodule", "update", "--init", "--", sub_path],
                cwd=top, check=False,
            )
        updated.append(sub_path)

    # Recurse into submodules that have their own .gitmodules
    if ns.recursive:
        for sub_path in updated:
            sub_dir = os.path.join(top, sub_path)
            if os.path.isfile(os.path.join(sub_dir, ".gitmodules")):
                subprocess.run(
                    [sys.executable, __file__,
                     "submodule", "update", "--init", "--recursive"],
                    cwd=sub_dir, check=False,
                )

    sys.exit(0)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    if not CACHE_DIR or not os.path.isdir(CACHE_DIR):
        passthrough()

    global_args, subcmd, rest = _find_subcommand(args)

    if subcmd == "clone":
        handle_clone(global_args, rest)
    elif subcmd == "submodule":
        handle_submodule(global_args, rest)
    else:
        passthrough()


if __name__ == "__main__":
    main()
