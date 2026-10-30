# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 (MXFP4) / sm_100 / tp16 attention, routed experts moe_tp 16 x moe_ep 1."""

from ...._native import load_target_ops

native_ops = load_target_ops(__name__)
