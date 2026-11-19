# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 config: Kimi K3's composite config under K3.1's ``model_type: kimi_k31``.

The text config is K3's ``KimiLinearConfig`` (model_type ``kimi_linear``), so the engine's Kimi KDA hybrid-cache paths
serve K3.1 as K3; K3.1's ``dsa_config`` stays a plain attribute of it.
"""

from tensorrt_llm._torch.configs.kimi_k3 import KimiK3Config


class KimiK31Config(KimiK3Config):
    model_type = "kimi_k31"

    def __init__(self, **kwargs):
        # transformers gives a config subclass without its own __init__ a generated one, which skips
        # KimiK3Config's text_config / vision_config construction.
        super().__init__(**kwargs)
