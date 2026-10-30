# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight table and loader: Qwen3.8-Flash-Next-NVFP4 (hf-fc694b5_orig) / sm_103 / unified.

Not written yet. It declares every weight once, cut per rank from the engine's
`Mapping` for the layouts `routing.py` admits, and its loader asserts coverage both
ways: every declared parameter filled, every checkpoint key consumed or named as
unconsumed.
"""
