# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MegaMoE local backend of the MiniMax-M3.1 DEP targets (``local_impl``) and its host planning (``local_plan``).

``local_plan`` is pure Python; the backend itself is imported on first use (``local_impl``), which registers it with
the MoE resolution.
"""

from . import local_plan

__all__ = ["local_plan"]
