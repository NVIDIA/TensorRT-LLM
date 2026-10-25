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

import os
from typing import Any, Optional

from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader
from tensorrt_llm.mapping import Mapping


class BeamNvfp4CheckpointLoader(HfCheckpointLoader):
    """Preload the Beam NVFP4 full-model target; read a drafter's weights after it."""

    def __init__(self):
        super().__init__()
        self._preloaded = False
        self._target_dir: Optional[str] = None

    def __reduce__(self):
        # Workers rebuild the loader from its class: it carries no state before a load.
        return (type(self), ())

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
        model.load_checkpoint(checkpoint_dir)
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


__all__ = ["BeamNvfp4CheckpointLoader"]
