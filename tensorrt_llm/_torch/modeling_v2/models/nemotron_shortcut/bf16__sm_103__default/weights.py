# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank-local BF16 weight loader for the text decoder."""

import torch

from ..manifest import (
    build_manifest,
    check_checkpoint_keys,
    load_unquantized,
    output_head_entry,
)


@torch.no_grad()
def load(model, weights):
    """Load only the local shards and check both checkpoint and destination coverage."""
    core = model.model
    manifest = build_manifest(core)
    parameters = dict(core.w.items())
    if set(manifest) != set(parameters):
        raise ValueError(f"Parameter manifest mismatch: {sorted(set(manifest) ^ set(parameters))}")

    head = model.lm_head.weight
    parameters["lm_head"] = head
    manifest["lm_head"] = output_head_entry(core, head)

    needed = {key for entries in manifest.values() for key, _, _, _ in entries}
    check_checkpoint_keys(weights, needed, mtp_enabled=core.mtp_enabled)

    for name, entries in manifest.items():
        load_unquantized(parameters[name], entries, weights)
    if head.is_cuda:
        torch.cuda.current_stream(head.device).synchronize()
