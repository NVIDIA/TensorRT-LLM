# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone fused elementwise/norm/quant operators shared across models.

Each submodule hosts one fused operator as a plain Python callable (not a
registered custom op). Callers own the enablement checks and keep the
unfused op chains as fallbacks for unsupported configurations; see the
Gemma4 decoder (tensorrt_llm/_torch/models/modeling_gemma4.py) for the
reference usage pattern.
"""
