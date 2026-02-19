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

"""RoPE (Rotary Position Embedding) operations.

This module provides various RoPE implementations:
- torch_rope: PyTorch reference implementation
- flashinfer_rope: FlashInfer-based optimized RoPE
- triton_rope: Triton-based RoPE implementation
- triton_rope_kernel: Low-level Triton kernels for RoPE
"""

__all__ = [
    "torch_rope",
    "flashinfer_rope",
    "triton_rope",
    "triton_rope_kernel",
]
