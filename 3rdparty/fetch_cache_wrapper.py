#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
FetchContent cache wrapper — injected by CMake as GIT_EXECUTABLE.

Intercepts ``git clone`` and ``git submodule update --init`` to inject
``--reference <cache>/<repo>.git``.  All other git commands pass through.

**Strict parsing**: the args cmake passes are a known, fixed set.  If an
unrecognized arg appears (cmake changed its calling convention), the
wrapper raises ``RuntimeError`` rather than silently misbehaving.

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


def lookup_cache(url: str) -> str | None:
    ref = os.path.join(CACHE_DIR, f"{repo_name(url)}.git")
    if os.path.isfile(os.path.join(ref, "HEAD")):
        return ref
    return None


def _strict_parse(parser: _StrictParser, args: list[str], context: str):
    """Parse *args* or raise ``RuntimeError`` with a clear message."""
    try:
        return parser.parse_args(args)
    except _ParseError as exc:
        raise RuntimeError(
            f"fetch_cache_wrapper: unrecognized {context} args — "
            f"cmake's calling convention may have changed.  "
            f"Please update the wrapper.\n"
            f"  args:  {args}\n"
            f"  error: {exc}"
        ) from None


# ---------------------------------------------------------------------------
# clone
# ---------------------------------------------------------------------------

def handle_clone(global_args: list[str], clone_args: list[str]):
    ns = _strict_parse(_clone_parser, clone_args, "git clone")

    if ns.reference:
        passthrough()

    ref = lookup_cache(ns.url)
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

    # Register submodules (no clone)
    subprocess.run([REAL_GIT, "submodule", "init"], check=False)

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

        ref = lookup_cache(sub_url)
        cmd = [REAL_GIT, "submodule", "update", "--init"]
        if ref:
            cmd += ["--reference", ref]
            print(f"{TAG} submodule {sub_path}: using reference {ref}",
                  file=sys.stderr)
        cmd += ["--", sub_path]

        ret = subprocess.run(cmd).returncode
        # Fallback without --reference if it failed
        if ret != 0 and ref:
            subprocess.run(
                [REAL_GIT, "submodule", "update", "--init", "--", sub_path],
                check=False,
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
