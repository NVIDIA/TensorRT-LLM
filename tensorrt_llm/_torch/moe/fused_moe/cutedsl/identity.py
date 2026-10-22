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
"""Identity segments of the CuteDSL implementations.

Each one publishes ``trtllm.cutedsl.<kernel>.<quant>``. The kernel segment says
how much of the expert MLP one launch covers: ``grouped_gemm`` runs FC1 and FC2
as two grouped GEMMs, ``fused_fc12`` runs both in one persistent kernel, and
``dense_gemm`` packs every expert into one dense matrix for small token counts.
``trtllm.cutedsl.mega_moe.nvfp4``, which also fuses dispatch and combine, lives
in :mod:`..mega_moe`.
"""

from ..impl_identity import MoEImplId, MoEImplQuery

PROVIDER_TRTLLM = "trtllm"
TECHNIQUE_CUTEDSL = "cutedsl"

KERNEL_GROUPED_GEMM = "grouped_gemm"
KERNEL_FUSED_FC12 = "fused_fc12"
KERNEL_DENSE_GEMM = "dense_gemm"

#: Every grouped-GEMM identity, open in ``quant`` only.
GROUPED_GEMM_FAMILY = MoEImplQuery(
    provider=PROVIDER_TRTLLM, technique=TECHNIQUE_CUTEDSL, kernel_name=KERNEL_GROUPED_GEMM
)


def grouped_gemm_identity(quant: str) -> MoEImplId:
    return MoEImplId(PROVIDER_TRTLLM, TECHNIQUE_CUTEDSL, KERNEL_GROUPED_GEMM, quant)
