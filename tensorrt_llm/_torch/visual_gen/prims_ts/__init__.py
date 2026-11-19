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
"""VisualGen glue for the FlashInfer Prims-TS dense GEMMs (``flashinfer.prims_ts.gemm``)."""

from .common import (
    config_for,
    device_supported,
    fp8_gemm_available,
    m_bucket,
    nvfp4_qkv_available,
    register_tactics,
    token_positions,
)
from .fp8 import fp8_gemm_eligible, quantize_per_token

__all__ = [
    "config_for",
    "device_supported",
    "fp8_gemm_available",
    "fp8_gemm_eligible",
    "m_bucket",
    "nvfp4_qkv_available",
    "quantize_per_token",
    "register_tactics",
    "token_positions",
]
