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
"""

from tensorrt_llm._torch.models.checkpoints.beam.checkpoint_loader import BeamNvfp4CheckpointLoader

__all__ = ["BeamNvfp4CheckpointLoader"]
