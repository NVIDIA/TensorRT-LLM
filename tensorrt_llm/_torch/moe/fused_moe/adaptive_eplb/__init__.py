# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive expert load balancing during decode (``moe_load_balancer`` mode ``adaptive``).

Modules: ``policy`` (time model, layout planner, install decisions, and the route tables, shares and
statistics buffers of the ops) and ``transport`` (expert weights in MNNVL memory and the copies
between ranks).
"""
