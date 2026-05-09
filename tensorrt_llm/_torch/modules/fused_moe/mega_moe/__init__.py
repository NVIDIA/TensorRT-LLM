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
"""MegaMoE — DeepGEMM ``fp8_fp4_mega_moe`` as a first-class MoE backend.

Targets the W4A8_MXFP4_MXFP8 quant configuration already supported by
``TRTLLMGenFusedMoE``. Shares the ``{expert_id}.w*.weight`` /
``{expert_id}.w*.weight_scale`` loader keys so that identical MXFP4 bytes
fed to both backends produce numerically-aligned outputs.
"""

from .backend import MegaMoEDeepGemmFusedMoE

__all__ = ["MegaMoEDeepGemmFusedMoE"]
