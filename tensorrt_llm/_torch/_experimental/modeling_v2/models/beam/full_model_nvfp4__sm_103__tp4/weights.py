# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weights of the Beam NVFP4 full-model target: a direct preload through the native loader.

``BeamNvfp4CheckpointLoader`` is passed as the LLM API's per-instance ``checkpoint_loader``.
The native ``ModelLoader`` constructs the target, materializes it on the rank's device and then
calls ``load_weights(checkpoint_dir, mapping, model=...)``; this loader hands the directory to
the target's ``load_checkpoint``, which streams bounded tiles straight into the target's own
rank-local storage (packed E2M1 codes, E4M3 block scales, FP32 scalars, BF16/FP32 dense
tensors) as one all-rank transaction, and returns no weight dictionary. The native lifecycle
continues as for any preloaded checkpoint (``is_weights_preloaded``). Configuration loading is
the HF loader's; everything else is the HF checkpoint loader's own behavior.

With a DFlash drafter the native loader then reads the drafter checkpoint
(``speculative_model``) through the same loader, without a model: that second load is the HF
loader's ordinary weight dictionary, which the target's ``load_draft_weights`` hands to the
drafter's own loader. It is accepted only after the target transaction completed and only for
a directory other than the target's, so neither load can run the other's path; the target
stays preloaded.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any, Optional

from tensorrt_llm._torch._experimental.modeling_v2.models.beam._attention import fp8_kv_scale_pairs
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader
from tensorrt_llm.mapping import Mapping


def _scale_section(table: dict, section: str, path: str) -> Optional[tuple]:
    entry = table.get(section)
    if entry is None:
        return None
    scales = entry.get("scales") if isinstance(entry, dict) else None
    layers = entry.get("layers") if isinstance(entry, dict) else None
    if not isinstance(scales, dict) or type(layers) is not int or layers < 1:
        raise ValueError(f"{path}: {section!r} must hold 'layers' and a 'scales' map")
    if sorted(scales) != sorted(str(i) for i in range(layers)):
        raise ValueError(
            f"{path}: {section!r} scales must name logical layers 0..{layers - 1}, got "
            f"{sorted(scales)}"
        )
    values = []
    for i in range(layers):
        value = scales[str(i)]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{path}: {section} layer {i} scale {value!r} is not finite and > 0")
        values.append(float(value))
    fp8_kv_scale_pairs(values, f"{path}: {section}")
    return tuple(values)


def read_kv_cache_scales(path: str) -> dict:
    """The FP8 KV-cache scale table in the JSON file ``path``, validated.

    The file maps ``"target"`` and optionally ``"draft"`` to ``{"layers": n, "scales": {"0": s0,
    ..., "n-1": s}}``: one finite positive scale per logical layer whose FP32 value and
    reciprocal are finite and normal, shared by K and V (a value ``x`` is stored as
    ``e4m3(x / s)``, read as ``code * s``). Returns ``{"target":
    (...), "draft": (...) or None, "source": path, "sha256": <file digest>}``.
    """
    with open(path, "rb") as f:
        data = f.read()
    table = json.loads(data)
    if not isinstance(table, dict):
        raise ValueError(f"{path}: a KV-cache scale table is a JSON object")
    target = _scale_section(table, "target", path)
    if target is None:
        raise ValueError(f"{path}: the KV-cache scale table has no 'target' section")
    return {
        "target": target,
        "draft": _scale_section(table, "draft", path),
        "source": os.path.abspath(path),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


class BeamNvfp4CheckpointLoader(HfCheckpointLoader):
    """Preload the Beam NVFP4 full-model target; read a drafter's weights after it.

    ``kv_cache_scales`` (``read_kv_cache_scales``) is the FP8 KV cache's scale table, handed to
    the target's ``load_checkpoint`` on every rank; None for a BF16 KV cache.
    """

    def __init__(self, kv_cache_scales: Optional[dict] = None):
        super().__init__()
        self._preloaded = False
        self._target_dir: Optional[str] = None
        self.kv_cache_scales = kv_cache_scales

    def __reduce__(self):
        # Workers rebuild the loader from its class and the scale table: it carries no other
        # state before a load.
        return (type(self), (self.kv_cache_scales,))

    def load_weights(self, checkpoint_dir: str, mapping: Mapping, **kwargs) -> dict[str, Any]:
        from .modeling import ModelingV2BeamFullModelNvfp4Sm103Tp4

        model = kwargs.get("model")
        if model is None:
            return self._load_draft_weights(checkpoint_dir, mapping, **kwargs)
        if not isinstance(model, ModelingV2BeamFullModelNvfp4Sm103Tp4):
            raise TypeError(
                "BeamNvfp4CheckpointLoader loads only ModelingV2BeamFullModelNvfp4Sm103Tp4 "
                f"(modeling_v2='require'); got {type(model).__name__}"
            )
        self._preloaded = False
        self._target_dir = None
        model.load_checkpoint(checkpoint_dir, kv_cache_scales=self.kv_cache_scales)
        self._target_dir = os.path.realpath(checkpoint_dir)
        self._preloaded = True
        return {}

    def _load_draft_weights(self, checkpoint_dir: str, mapping: Mapping, **kwargs) -> dict:
        """The drafter checkpoint's tensors, read by the HF loader for the drafter's own load."""
        if not self._preloaded:
            raise RuntimeError(
                f"{checkpoint_dir!r}: a weight load without the target model is the drafter's, "
                "which follows the Beam target's own load"
            )
        if os.path.realpath(checkpoint_dir) == self._target_dir:
            raise ValueError(
                f"{checkpoint_dir!r} is the Beam target's checkpoint; a drafter load names the "
                "drafter checkpoint (speculative_model)"
            )
        return super().load_weights(checkpoint_dir, mapping=mapping, **kwargs)

    def is_weights_preloaded(self) -> bool:
        return self._preloaded


__all__ = ["BeamNvfp4CheckpointLoader", "read_kv_cache_scales"]
