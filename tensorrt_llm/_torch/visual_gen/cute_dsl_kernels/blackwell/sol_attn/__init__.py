# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from https://github.com/NVlabs/Sana (Apache-2.0); see
# THIRD_PARTY_NOTICES.md in this directory for the pin and scope.
"""Sol-Attn."""

from .interface import get_sol_attn_backend, sol_attn

__all__ = ["get_sol_attn_backend", "sol_attn"]
