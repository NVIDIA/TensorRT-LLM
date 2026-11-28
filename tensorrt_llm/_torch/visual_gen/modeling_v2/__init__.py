# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def resolve_minimax_h3(config, checkpoint_dir):
    from .models.minimax_h3.pipeline_registry import route

    return route(config, checkpoint_dir)
