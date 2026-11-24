# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from functools import cache
from pathlib import Path

_LIBS_DIR = Path(__file__).resolve().parents[3] / "libs"
_LOAD_LOCK = threading.Lock()


def run_context(params) -> None:
    from tensorrt_llm.bindings.internal import thop

    thop.run_context(params)


class NativeScopeUnavailable(RuntimeError):
    """The requested VisualGen scope was omitted from this build."""


@cache
def _native_manifest(path: Path) -> dict[str, dict[str, str]]:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError as error:
        raise NativeScopeUnavailable(
            f"No VisualGen native manifest at {path}. Build and install the VisualGen extensions."
        ) from error


def load_native_op(scope: str, name: str):
    """Load the catalog or family library for a built scope and return its registered op."""
    import torch

    manifest = _native_manifest(_LIBS_DIR / "visual_gen_native.json")
    if scope not in manifest:
        raise NativeScopeUnavailable(f"VisualGen native scope {scope!r} is absent from this build.")
    entry = manifest[scope]
    lib = (_LIBS_DIR / entry["library"]).resolve()
    with _LOAD_LOCK:
        if str(lib) not in torch.ops.loaded_libraries:
            try:
                torch.ops.load_library(str(lib))
            except (OSError, RuntimeError) as error:
                raise RuntimeError(
                    f"Cannot load {lib} for VisualGen native scope {scope!r}."
                ) from error
    namespace = entry["namespace"]
    try:
        return getattr(getattr(torch.ops, namespace), name)
    except AttributeError as error:
        raise RuntimeError(
            f"{lib} did not register the VisualGen op {namespace}::{name}."
        ) from error
