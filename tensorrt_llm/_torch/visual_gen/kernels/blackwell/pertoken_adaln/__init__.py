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
# Fused per-token AdaLN modulate (CuTe DSL) for the WAN per-token
# (temb.ndim == 4) path. The ops take raw bf16 temb chunk views plus the
# [D] fp32 scale_shift_table rows and fuse the fp32 table+chunk add inline,
# avoiding materialized fp32 [B, S, D] modulator tensors.
#
# Custom ops registered under the ``trtllm::`` torch.library namespace.

from .pertoken_adaln import fused_pertoken_adaln, fused_pertoken_adaln_residual  # noqa: F401

__all__ = [
    "fused_pertoken_adaln",
    "fused_pertoken_adaln_residual",
]
