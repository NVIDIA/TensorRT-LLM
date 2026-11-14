# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where package data lives: <repo>/csrc in a source checkout, tp4c1/_csrc in an installed wheel."""

from __future__ import annotations

import functools
import os

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../tp4c1


@functools.lru_cache(maxsize=None)
def repo_root() -> str | None:
    r = os.path.dirname(os.path.dirname(PKG))
    return (
        r
        if os.path.isfile(os.path.join(r, "pyproject.toml"))
        and os.path.isdir(os.path.join(r, "csrc"))
        else None
    )


@functools.lru_cache(maxsize=None)
def csrc_root() -> str:
    inst = os.path.join(PKG, "_csrc")
    if os.path.isdir(inst):
        return inst
    r = repo_root()
    if r is None:
        raise FileNotFoundError(
            "tp4c1: neither tp4c1/_csrc (wheel) nor <repo>/csrc (checkout) found"
        )
    return os.path.join(r, "csrc")


def pkg_root() -> str:
    return PKG


def csrc_dir(family: str, variant: str) -> str:
    return os.path.join(csrc_root(), family, variant)


def data_path(family: str, variant: str, relpath: str) -> str:
    p = os.path.normpath(os.path.join(csrc_dir(family, variant), relpath))
    if not p.startswith(csrc_dir(family, variant) + os.sep):
        raise ValueError(f"data_path {relpath!r} escapes csrc/{family}/{variant}")
    return p


def csrc_abs(relpath: str) -> str:
    p = os.path.normpath(os.path.join(csrc_root(), relpath))
    if not p.startswith(csrc_root() + os.sep):
        raise ValueError(f"{relpath!r} escapes csrc/")
    return p


def pkg_abs(relpath: str) -> str:
    p = os.path.normpath(os.path.join(PKG, relpath))
    if not p.startswith(PKG + os.sep):
        raise ValueError(f"{relpath!r} escapes tp4c1/")
    return p
