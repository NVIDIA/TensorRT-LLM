# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# Fused per-token AdaLN modulate (CuTe DSL) for WAN's per-token
# (temb.ndim == 4) path and FLUX 3. The ops take raw bf16 temb chunk views
# plus optional [D] fp32 scale_shift_table rows (None for models without a
# table, such as FLUX 3) and fuse the fp32 table+chunk add inline, avoiding
# materialized fp32 [B, S, D] modulator tensors.
#
# Custom ops registered under the ``trtllm::`` torch.library namespace.

from .support import REQUIRED_ALIGNMENT, is_supported, is_supported_device, supports_hidden_size

try:
    from .pertoken_adaln import fused_pertoken_adaln, fused_pertoken_adaln_residual
except (ImportError, OSError):  # CuTe DSL unavailable: callers keep their unfused path
    fused_pertoken_adaln = None
    fused_pertoken_adaln_residual = None

__all__ = [
    "REQUIRED_ALIGNMENT",
    "fused_pertoken_adaln",
    "fused_pertoken_adaln_residual",
    "is_supported",
    "is_supported_device",
    "supports_hidden_size",
]
