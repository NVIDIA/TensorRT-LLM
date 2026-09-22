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

from __future__ import annotations

import os
import pathlib
import shutil
from typing import Any

import torch

from tensorrt_llm.logger import logger

GUARDRAIL_HF_REPO = "nvidia/Cosmos-1.0-Guardrail"
GUARDRAIL_REVISION = "cf03c0395fac8c4de386c0bdab12cc4fc8d66362"

# The guardrail repo carries ~17 GB of Cosmos 1.0-era checkpoints that
# CosmosSafetyChecker no longer instantiates; fetch only what it loads.
GUARDRAIL_ALLOW_PATTERNS = ["blocklist/*", "face_blur_filter/*"]


def _materialize_nltk_data(snapshot_dir: str) -> None:
    """Turn the symlinks under blocklist/nltk_data into regular files.

    The HF cache keeps one copy of each file under blobs/ and exposes it in the
    snapshot tree through a symlink. nltk >= 3.10.3 opens its data files with
    O_NOFOLLOW and refuses a symlinked final component, so Blocklist fails on a
    freshly populated cache. Copying the blob over the symlink yields the layout
    hf_hub itself produces with HF_HUB_DISABLE_SYMLINKS=1, limited to the ~25 MB
    nltk reads. hf_hub only checks that a snapshot path exists before skipping
    a download, so the library's own snapshot_download leaves the copies alone.
    """
    root = pathlib.Path(snapshot_dir) / "blocklist" / "nltk_data"
    if not root.is_dir():
        return
    copied = 0
    for path in root.rglob("*"):
        if not path.is_symlink():
            continue
        target = os.path.realpath(path)
        if not os.path.isfile(target):
            continue
        tmp = path.with_name(f".{path.name}.materialize")
        shutil.copyfile(target, tmp)
        os.replace(tmp, path)
        copied += 1
    if copied:
        logger.debug(f"Materialized {copied} nltk_data symlinks under {root}")


def download_guardrail_checkpoint() -> str:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    kwargs = dict(
        repo_id=GUARDRAIL_HF_REPO,
        revision=GUARDRAIL_REVISION,
        allow_patterns=GUARDRAIL_ALLOW_PATTERNS,
    )
    try:
        snapshot = snapshot_download(local_files_only=True, **kwargs)
    except FileNotFoundError:
        snapshot = None

    if snapshot is None:
        logger.warning(f"Guardrail checkpoint not found, downloading from {GUARDRAIL_HF_REPO}")
        try:
            snapshot = snapshot_download(**kwargs)
        except GatedRepoError as e:
            raise ValueError(
                "Cosmos Guardrail checkpoint not found. "
                "Please ensure "
                "a) you have accepted the terms of use (https://huggingface.co/nvidia/Cosmos-1.0-Guardrail) "
                "b) you have set a valid HF_TOKEN environment variable"
            ) from e

    _materialize_nltk_data(snapshot)
    return snapshot


def check_video_safety(video_tensor: torch.Tensor, safety_checker: Any) -> torch.Tensor | None:
    v = video_tensor.detach().cpu()
    was_batched = v.dim() == 5
    if was_batched:
        v = v[0]
    frames_np = v.numpy()
    frames_np = safety_checker.check_video_safety(frames_np)
    if frames_np is None:
        return None

    result = torch.from_numpy(frames_np)
    if was_batched:
        result = result.unsqueeze(0)
    return result.to(video_tensor.device)
