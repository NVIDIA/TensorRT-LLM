# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a single native LTX-2.3 checkpoint before pipeline configuration/loading."""

from pathlib import Path

from tensorrt_llm.llmapi.utils import download_hf_partial

LTX23_MODEL_ID = "Lightricks/LTX-2.3"

_WORKFLOW_CHECKPOINTS = {
    "generation": "ltx-2.3-22b-dev.safetensors",
    "retake": "ltx-2.3-22b-distilled-1.1.safetensors",
}


def resolve_ltx23_checkpoint(model: str, *, workflow: str, revision: str | None = None) -> str:
    """Resolve a local file, directory, or Hub ID to one workflow's checkpoint.

    Explicit safetensors files are preserved. Directories and Hub repositories
    must contain the canonical filename for the requested workflow; no fallback
    to another checkpoint variant is allowed. Hub downloads use the shared
    file-lock-protected utility, including its offline-cache behavior.
    """
    if workflow not in _WORKFLOW_CHECKPOINTS:
        raise ValueError(f"Unsupported LTX-2.3 workflow: {workflow!r}")
    filename = _WORKFLOW_CHECKPOINTS[workflow]
    path = Path(model)
    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError(f"LTX-2.3 checkpoint must be a .safetensors file: {model}")
        return str(path)
    if path.is_dir():
        checkpoint = path / filename
    elif path.suffix == ".safetensors" or path.is_absolute() or model.startswith((".", "~")):
        raise FileNotFoundError(f"LTX-2.3 checkpoint path not found: {model}")
    else:
        checkpoint = (
            Path(download_hf_partial(model, allow_patterns=[filename], revision=revision))
            / filename
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"LTX-2.3 {workflow} checkpoint not found: {checkpoint}")
    return str(checkpoint)
