# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One package per upstream architecture family.

``models/<x>/`` corresponds one-to-one with the built-in zoo's
``modeling_<x>.py``: the same architecture, split into one self-contained
codebase per deployment target instead of one class serving every
checkpoint.

Importing this package imports every family's ``routing`` module, and each one
registers its ``route`` through ``@routes``. A new family adds one line here.
"""

from .deepseek_v3 import routing as deepseek_v3_routing  # noqa: F401
from .glm_moe_dsa import routing as glm_moe_dsa_routing  # noqa: F401
from .gpt_oss import routing as gpt_oss_routing  # noqa: F401
from .kimi_k3_vl import routing as kimi_k3_vl_routing  # noqa: F401
from .minimax_m3 import routing as minimax_m3_routing  # noqa: F401
from .qwen4_exp import routing as qwen4_exp_routing  # noqa: F401
