# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: Qwen3.8-Flash-Next-NVFP4 (hf-fc694b5_orig) / sm_103 / unified.

One target for the three layouts `routing.py` admits at four ranks in one node: tp4
(attention TP4, routed experts split by intermediate), tep4 (attention TP4, experts EP4)
and dep4 (attention DP, experts EP4). The core routes each step to `PrefillTarget` (any
batch with context rows) or `DecodeTarget` (generation only).

The forward is not written yet: `ModelingV2HfFc694b5OrigSm103Unified` raises at
construction, so a deployment routed here fails while the engine is built rather than
serving anything.

The constants below are what `routing.py` admits; its routing test holds the two
files to the same values.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM

# The GPU architecture this target is certified for; a direct instantiation could
# bypass routing.
_SM = (10, 3)

# `moe_config.backend` request -> the routed-expert backend the trunk runs.
_TRUNK_MOE_BACKENDS = {"AUTO": "TRTLLM", "TRTLLM": "TRTLLM"}

# Rows per request of the widest verify step (`max_draft_len + 1`), and of the widest
# one without the GDN cached replay (the per-step recurrent states).
_MAX_VERIFY_WIDTH = 8
_MAX_LEGACY_VERIFY_WIDTH = 4

# Recurrent-state cache dtypes the target serves.
_SSM_STATE_DTYPES = (torch.bfloat16,)


class PrefillTarget:
    """Runs a step that carries context rows. Not written yet."""


class DecodeTarget:
    """Runs a generation-only step. Not written yet."""


class ModelingV2HfFc694b5OrigSm103Unified(DecoderModelForCausalLM):
    """Qwen3.8-Flash-Next-NVFP4 at tp4, tep4 or dep4 on sm_103."""

    def __init__(self, model_config: ModelConfig):
        raise NotImplementedError(
            "ModelingV2HfFc694b5OrigSm103Unified has no forward yet; build this deployment "
            "with modeling_v2='off'"
        )
