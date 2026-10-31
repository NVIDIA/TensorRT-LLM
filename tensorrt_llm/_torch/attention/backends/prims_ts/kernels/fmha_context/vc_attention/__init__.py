# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VC-Attention pieces of the context kernel.

``vc_resources`` holds the tile-mean operand ring and the ExpCast helpers,
``vc_scales`` the per-block Q/K scale tables of VC-Attention-QK8.
"""
