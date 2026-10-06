# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint helpers for Qwen-Image-2.1 VisualGen candidate entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .config import DEFAULT_HF_MODEL_ID

_RECOGNIZED_MARKERS = (
    "model_index.json",
    "modular_model_index.json",
    "config.json",
    "tokenizer_config.json",
    "scheduler_config.json",
)


@dataclass(frozen=True)
class QwenImage21CheckpointSource:
    """Resolved candidate checkpoint source."""

    source: str
    path_or_id: str
    revision: str | None = None
    local_files_only: bool = False

    @property
    def is_local(self) -> bool:
        return self.source == "local_path"


def checkpoint_has_markers(path: str | Path) -> bool:
    """Return True when ``path`` looks like a Diffusers-style model root."""

    root = Path(path).expanduser()
    if not root.is_dir():
        return False
    return any((root / marker).exists() for marker in _RECOGNIZED_MARKERS)


def resolve_checkpoint_source(
    manifest: Mapping[str, object] | None = None,
    *,
    default_hf_model_id: str = DEFAULT_HF_MODEL_ID,
) -> QwenImage21CheckpointSource:
    """Resolve local/HF checkpoint policy from a frozen manifest/preflight dict."""

    manifest = manifest or {}
    local_path = manifest.get("local_path") or manifest.get("checkpoint_path")
    if isinstance(local_path, str) and local_path and checkpoint_has_markers(local_path):
        return QwenImage21CheckpointSource(
            source="local_path",
            path_or_id=str(Path(local_path).expanduser()),
            revision=manifest.get("revision") if isinstance(manifest.get("revision"), str) else None,
            local_files_only=True,
        )
    hf_model_id = manifest.get("hf_model_id")
    if not isinstance(hf_model_id, str) or not hf_model_id:
        hf_model_id = default_hf_model_id
    revision = manifest.get("revision")
    return QwenImage21CheckpointSource(
        source="huggingface",
        path_or_id=hf_model_id,
        revision=revision if isinstance(revision, str) and revision else None,
        local_files_only=False,
    )
