# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attention operations.

This module provides various attention implementations and backends:
- torch_attention: PyTorch reference implementations
- torch_backend_attention: PyTorch-based attention backend
- flashinfer_attention: FlashInfer-based optimized attention
- triton_attention: Triton-based attention implementations
- triton_attention_with_kv_cache: Triton attention with KV cache support
- triton_paged_attention: Triton paged attention (two-stage flash-decode) with HND layout
- triton_paged_attention_one_stage: Triton paged attention (single-stage) with HND layout
- onnx_attention: Placeholder ops for ONNX export of attention mechanisms

All modules are auto-imported by the parent custom_ops/__init__.py via pkgutil.walk_packages,
so no explicit imports are needed here.
"""
