# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1-Flash NVFP4 / sm_107 / tp2ctx.

The disaggregated context (prefill) worker: TP2 attention and EP2 experts.
"""

from ...._native import load_target_ops

# This target has no native code of its own (it calls the family's shared ops); the call checks that this build
# carries it.
native_ops = load_target_ops(__name__)
