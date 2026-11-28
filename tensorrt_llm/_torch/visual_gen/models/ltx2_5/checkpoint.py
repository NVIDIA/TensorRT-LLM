# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the supported BF16 distilled transformer without selecting another precision."""

from pathlib import Path

from tensorrt_llm.llmapi.utils import download_hf_partial

BF16_TRANSFORMER = "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"


def resolve_ltx25_checkpoint(model: str, *, revision: str | None = None) -> str:
    """Resolve a split-pack directory, explicit file, or Hub repository to BF16."""
    path = Path(model)
    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError("LTX-2.5 requires a safetensors checkpoint")
        return str(path)
    if path.is_dir():
        checkpoint = path / BF16_TRANSFORMER
    elif path.is_absolute() or path.suffix == ".safetensors" or model.startswith((".", "~")):
        raise FileNotFoundError(model)
    else:
        checkpoint = (
            Path(download_hf_partial(model, allow_patterns=[BF16_TRANSFORMER], revision=revision))
            / BF16_TRANSFORMER
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Required LTX-2.5 BF16 distilled transformer missing: {checkpoint}"
        )
    return str(checkpoint)
