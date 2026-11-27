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
    # Re-resolve after model defaults reconstruct a config from public fields.
    spec_cfg._draft_weights_in_target = False
    spec_cfg._draft_model_config = None
    checkpoint = spec_cfg.speculative_model or target_model
    params_path = os.path.join(checkpoint, "params.json")
    if not os.path.isfile(params_path):
        return
    with open(params_path, encoding="utf-8") as stream:
        params = json.load(stream)
    mtp = params.get("mtp") or {}
    dspark = mtp.get("dspark")
    if dspark is None:
        return
    if os.path.realpath(checkpoint) != os.path.realpath(target_model):
        raise ValueError("Native Mistral DSpark must use the target checkpoint")
    if spec_cfg.use_ancestral_draft_sampling:
        raise ValueError(
            "use_ancestral_draft_sampling=True is not supported with the native "
            "Mistral DSpark drafter, whose Markov head is recurrent."
        )
    layers = [int(layer) for layer in mtp["target_activation_layers"].split(",")]
    trained = dict(
        target_layer_ids=layers,
        block_size=int(mtp["depth"]),
        markov_rank=int(dspark["markov_rank"]),
        markov_head_type=dspark["head_type"],
        mask_token_id=int(params["vocab_size"]),
    )
    for field, value in trained.items():
        override = getattr(spec_cfg, field)
        if override is not None and override != value:
            raise ValueError(
                f"Mistral DSpark {field} must match the checkpoint: {value}, got {override}"
            )
    if spec_cfg.max_draft_len != trained["block_size"]:
        raise ValueError(
            "Mistral DSpark max_draft_len must equal the trained block depth "
            f"{trained['block_size']}"
        )
    for field, value in trained.items():
        setattr(spec_cfg, field, value)
    spec_cfg._draft_model_config = mtp
    # Native draft weights are part of the target's manifest, while its paged
    # draft runtime and KV cache remain independent of the target's cache.
    spec_cfg._draft_weights_in_target = True
    spec_cfg.speculative_model = checkpoint
