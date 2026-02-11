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

"""Normalization operations.

This module provides various normalization implementations:
- rms_norm: RMSNorm implementations (FlashInfer, Triton, reference)
- triton_rms_norm: Low-level Triton RMSNorm kernel
- l2norm: L2 normalization operations
- flashinfer_fused_add_rms_norm: Fused add + RMSNorm operation
"""

__all__ = [
    "rms_norm",
    "triton_rms_norm",
    "l2norm",
    "flashinfer_fused_add_rms_norm",
]
