# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 NVFP4 checkpoint on Rubin (sm_107): the ULL generation deployment, wide batches.

``ull16``: the generation server, Helix CP16 (attention TP1) with MoE TP16 / EP1, DFlash2 DL7.
An experimental fork of ``kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1`` for batches above
eight requests, selected only by ``TLLM_K31_HELIX16_WIDE`` (``routing.py``); it takes the same
``ull16`` profile. Imports no submodule: the family ``route()`` selects the profile
(``features.select``) before any module that reads it is imported.
"""
