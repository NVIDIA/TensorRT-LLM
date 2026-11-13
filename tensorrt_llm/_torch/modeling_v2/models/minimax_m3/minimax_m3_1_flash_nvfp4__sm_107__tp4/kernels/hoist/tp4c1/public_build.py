# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public-build guard. tools/export_public.py writes _public_build.json into an exported tree; with it present, every
module the export removed fails on import with a clear message instead of a bare ModuleNotFoundError. The marker names
removed modules only by sha256, so it discloses nothing. Without the marker (the internal repo) nothing is installed."""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import sys

MARKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_public_build.json")
SCHEMA = "m3llk.public_build/1"
MESSAGE = "is not available in the public build"


class NotInPublicBuild(ImportError):
    """A module, variant or record the public export removed."""


def name_hash(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def load(path: str = MARKER) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        doc = json.load(f)
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r} != {SCHEMA}")
    return doc


def is_public(path: str = MARKER) -> bool:
    return os.path.isfile(path)


def unavailable(what: str) -> str:
    return f"{what} {MESSAGE}"


class _Raise(importlib.abc.Loader):
    def create_module(self, spec):
        raise NotInPublicBuild(unavailable(spec.name), name=spec.name)

    def exec_module(self, module):  # never reached: create_module raises
        raise NotInPublicBuild(unavailable(module.__name__), name=module.__name__)


class RemovedModuleFinder(importlib.abc.MetaPathFinder):
    """Last on sys.meta_path: only consulted for a module no other finder has, and only answers for removed ones."""

    def __init__(self, hashes):
        self.hashes = frozenset(hashes)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] == __package__ and name_hash(fullname) in self.hashes:
            return importlib.machinery.ModuleSpec(fullname, _Raise())
        return None


def install(path: str = MARKER) -> bool:
    """Install the finder when the marker is present; True if the tree is a public build."""
    doc = load(path)
    if doc is None:
        return False
    if not any(isinstance(f, RemovedModuleFinder) for f in sys.meta_path):
        sys.meta_path.append(RemovedModuleFinder(doc.get("removed_modules_sha256", ())))
    return True


__all__ = [
    "MARKER",
    "MESSAGE",
    "NotInPublicBuild",
    "RemovedModuleFinder",
    "install",
    "is_public",
    "name_hash",
    "unavailable",
]
