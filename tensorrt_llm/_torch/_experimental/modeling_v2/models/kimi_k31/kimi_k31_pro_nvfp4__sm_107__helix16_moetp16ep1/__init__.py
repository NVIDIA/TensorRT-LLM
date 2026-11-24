# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 NVFP4 checkpoint on Rubin (sm_107): the ULL generation deployment.

``ull16``: the generation server, Helix CP16 (attention TP1) with MoE TP16 / EP1, DFlash2 DL7.
Imports no submodule: the family ``route()`` selects the profile (``features.select``) before
any module that reads it is imported.
"""
