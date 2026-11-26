# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared validation and resolution of extra test import paths."""

import os

DECLARATION = "__extra_import_path__"


def resolve_entry(entry: str, base_dir: str, project_root: str) -> str:
    """Resolve paths relative to the declaring file, or the project for ``~/``."""
    if entry == "~" or entry.startswith(("~/", "~" + os.sep)):
        return os.path.normpath(os.path.join(project_root, entry[1:].lstrip("/" + os.sep)))
    return os.path.normpath(os.path.join(base_dir, entry))


def declared_paths(value: object, where: str) -> tuple[str, ...]:
    """Validate a list or tuple of filesystem paths without resolving them."""
    if value is None:
        return ()
    if isinstance(value, (str, bytes, os.PathLike)) or not isinstance(value, (list, tuple)):
        raise TypeError(
            f"{where}: {DECLARATION} must be a list of paths, got {type(value).__name__}"
        )
    return tuple(os.fspath(entry) for entry in value)
