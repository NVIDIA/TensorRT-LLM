# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2-NVFP4 on 16 Rubin GPUs (sm107), tp16: model construction and forward."""

from __future__ import annotations


class PrefillTarget:
    """Forwards that carry context requests."""


class DecodeTarget:
    """Generation-only forwards."""


class ModelingV2Glm52Nvfp4Sm107Tp16:
    """GLM-5.2-NVFP4, sm107, tp16."""

    def __init__(self, model_config):
        raise NotImplementedError("the glm_moe_dsa tp16 target is not assembled yet")
