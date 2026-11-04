# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the DSpark settings a checkpoint declares outside its config.json."""

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig


def resolve_dspark_checkpoint(spec_cfg: "DSparkDecodingConfig", target_model: str) -> None:
    """Adopt the drafter a Mistral-native target declares in its params.json.

    Runs before ``DSparkDecodingConfig.resolve_draft_config_fields``, which
    reads a drafter's config.json; a native drafter has none, so its trained
    geometry and weight ownership come from here.
    """
    from .mistral.config_loader import resolve_native_dspark_config

    # Re-resolve after model defaults reconstruct a config from public fields.
    spec_cfg._draft_weights_in_target = False
    spec_cfg._draft_model_config = None
    checkpoint = spec_cfg.speculative_model or target_model
    params_path = os.path.join(checkpoint, "params.json")
    if os.path.isfile(params_path):
        with open(params_path, encoding="utf-8") as stream:
            params = json.load(stream)
        resolve_native_dspark_config(params, spec_cfg, checkpoint, target_model)
