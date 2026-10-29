# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-contained deployment families, registered by checkpoint architecture.

Each ``models/<family>/`` owns its deployment targets. Independent families
can serve the same upstream architecture on different hardware; the resolver
requires exactly one matching route.

Importing this package imports every family's ``routing`` module, and each one
registers its ``route`` through ``@routes``. A new family adds one line here.
"""

from .deepseek_v3 import routing as deepseek_v3_routing  # noqa: F401
from .glm_moe_dsa import routing as glm_moe_dsa_routing  # noqa: F401
from .glm_moe_dsa_rubin import routing as glm_moe_dsa_rubin_routing  # noqa: F401
from .gpt_oss import routing as gpt_oss_routing  # noqa: F401
from .kimi_k3_vl import routing as kimi_k3_vl_routing  # noqa: F401
from .qwen4_exp import routing as qwen4_exp_routing  # noqa: F401
