# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .checkpoint import load_fasth3_manifest
from .default_pipeline.pipeline import MiniMaxH3Pipeline
from .fasth3.pipeline import FastH3Pipeline, configure_attention


def route(config, checkpoint_dir):
    manifest = load_fasth3_manifest(checkpoint_dir) if checkpoint_dir else {}
    if manifest:
        configure_attention(config, manifest)
        config.primary_pretrained_config._fasth3_manifest = manifest
        return FastH3Pipeline
    return MiniMaxH3Pipeline
