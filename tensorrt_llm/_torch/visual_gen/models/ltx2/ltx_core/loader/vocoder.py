"""One-prefix vocoder checkpoint mapping for ordinary and BWE models."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.sd_ops import SDOps


def _strip_one(key: str, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix("vocoder."): inputs[key]}


def prefix_stage() -> SDOps:
    """Select vocoder keys and strip exactly one wrapper prefix for ordinary/BWE models."""
    return (
        SDOps("vocoder-prefix-once")
        .with_matching(prefix="vocoder.")
        .with_kv_operation(_strip_one, key_prefix="vocoder.")
    )
