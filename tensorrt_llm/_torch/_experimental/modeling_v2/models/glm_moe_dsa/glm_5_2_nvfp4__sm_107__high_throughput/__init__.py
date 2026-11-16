# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rubin high-throughput target and its native operator namespace."""

from tensorrt_llm._torch._experimental.modeling_v2._native import load_target_ops

native_ops = load_target_ops(__name__)
