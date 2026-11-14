# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve DSpark checkpoint settings before model or weight construction."""

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig


def resolve_dspark_checkpoint(spec_cfg: "DSparkDecodingConfig", target_model: str) -> None:
    from tensorrt_llm.llmapi.llm_args import translate_speculators_dspark_config

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

    # An omitted draft path means the draft lives in the target checkpoint.
    if spec_cfg.speculative_model is None:
        spec_cfg.speculative_model = target_model
        if not spec_cfg.draft_is_embedded_in_target:
            raise ValueError(
                "speculative_config.speculative_model is unset, "
                "which means 'load the draft from the target "
                "checkpoint', but the target has no mtp.* draft "
                "weights. Point speculative_model at a standalone "
                "DSpark drafter checkpoint directory."
            )
    # Warm the embedded-vs-standalone probe here, while we are in
    # the main process and the checkpoint path is known local. The
    # flag then travels with the config, so no rank repeats the
    # filesystem read and nothing probes at forward time.
    _ = spec_cfg.draft_is_embedded_in_target
    # Resolve target_layer_ids / mask_token_id / block_size /
    # markov_rank from the draft (or main) model config if not set.
    # Several checkpoint spellings are in the wild for the same knobs
    # and all are accepted here, because a key the reader misses is
    # not an error -- it silently falls back to a default and the
    # drafter degrades (a markov_rank read as 0 skips the Markov
    # head entirely, costing acceptance with no warning):
    #   - top-level ``dspark_*``  (DeepSeek-V4-Pro)
    #   - nested ``dflash_config``  (SpecForge / RadixArk drafters)
    #   - nested ``dspark_config``  (forward compatibility)
    #   - plain top-level keys  (TorchSpec drafters)
    # V4.1 wraps the language model settings in ``text_config``;
    # prefer that scope, retaining the top-level legacy fallback.
    draft_config_path = os.path.join(spec_cfg.speculative_model, "config.json")
    if os.path.exists(draft_config_path):
        with open(draft_config_path) as f:
            draft_cfg = translate_speculators_dspark_config(json.load(f))
        text_cfg = draft_cfg.get("text_config") or {}

        def _dspark_get(key: str, top_level_key: str) -> list[int] | int | None:
            for scope in (text_cfg, draft_cfg):
                for source, name in (
                    (scope.get("dspark_config") or {}, key),
                    (scope.get("dflash_config") or {}, key),
                    (scope, top_level_key),
                    (scope, key),
                ):
                    value = source.get(name)
                    if value is not None:
                        return value
            return None

        # The checkpoint's ``dspark_target_layer_ids`` is
        # authoritative: it fixes both which target hidden states
        # were captured during draft training and the input width of
        # ``main_proj`` (hidden_size * num_capture_layers). If the
        # user leaves ``target_layer_ids`` unset we copy it verbatim
        # (order preserved, since the projection columns are
        # order-dependent). An explicit override that does not match
        # the checkpoint list exactly would either mismatch the
        # projection shape at runtime (different count) or feed the
        # draft hidden states it was not trained on (same count,
        # different layers), so reject it during validation.
        ckpt_layer_ids = _dspark_get("target_layer_ids", "dspark_target_layer_ids")
        if spec_cfg.target_layer_ids is None:
            if ckpt_layer_ids is not None:
                spec_cfg.target_layer_ids = list(ckpt_layer_ids)
        elif ckpt_layer_ids is not None and list(spec_cfg.target_layer_ids) != list(ckpt_layer_ids):
            raise ValueError(
                "DSpark target_layer_ids must match the checkpoint's "
                "dspark_target_layer_ids exactly (the draft "
                "projection weights are trained for that specific "
                f"layer set); got override {spec_cfg.target_layer_ids} "
                f"but the checkpoint specifies {list(ckpt_layer_ids)}. "
                "Leave target_layer_ids unset to use the checkpoint "
                "value."
            )
        if spec_cfg.mask_token_id is None:
            mask_id = _dspark_get("mask_token_id", "dspark_noise_token_id")
            if mask_id is not None:
                spec_cfg.mask_token_id = mask_id
        if spec_cfg.block_size is None:
            block_size = _dspark_get("block_size", "dspark_block_size")
            if block_size is not None:
                spec_cfg.block_size = int(block_size)
        if spec_cfg.markov_rank is None:
            markov_rank = _dspark_get("markov_rank", "dspark_markov_rank")
            if markov_rank is not None:
                spec_cfg.markov_rank = markov_rank
