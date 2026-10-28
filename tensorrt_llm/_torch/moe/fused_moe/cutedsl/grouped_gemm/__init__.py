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
"""CuteDSL grouped GEMM: FC1 and FC2 as two grouped-GEMM launches.

:mod:`.base` is the abstract root, :mod:`.nvfp4` and :mod:`.bf16` are the
leaves, and :mod:`.eligibility` holds the gates their ``can_implement``
composes. Importing this package registers both leaves.
"""

from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...impl_contract import canonical_quant, normalize_quant
from ...impl_identity import MOE_IMPL_REGISTRY
from ..identity import GROUPED_GEMM_FAMILY, grouped_gemm_identity
from .base import CuteDslFusedMoEBase
from .bf16 import TrtllmCutedslGroupedGemmBf16Impl
from .nvfp4 import TrtllmCutedslGroupedGemmNvfp4Impl


def find_cutedsl_grouped_gemm_leaf(quant_algo: QuantAlgo | None) -> type | None:
    """The CuteDSL grouped-GEMM leaf implementing ``quant_algo``, or ``None``."""
    # The same canonicalization resolution does, so a caller naming an alias of
    # a published format reaches the leaf rather than a lookup miss.
    quant = normalize_quant(canonical_quant(quant_algo))
    return MOE_IMPL_REGISTRY.lookup(grouped_gemm_identity(quant))


def cutedsl_grouped_gemm_leaf(quant_algo: QuantAlgo | None) -> type:
    """As :func:`find_cutedsl_grouped_gemm_leaf`, but raising when no leaf serves the format."""
    cls = find_cutedsl_grouped_gemm_leaf(quant_algo)
    if cls is not None:
        return cls
    registered = sorted(
        identity.canonical() for identity, _ in MOE_IMPL_REGISTRY.find(GROUPED_GEMM_FAMILY)
    )
    raise ValueError(
        f"no CuteDSL grouped-GEMM implementation for quant="
        f"{normalize_quant(canonical_quant(quant_algo))}; registered: {registered}"
    )


__all__ = [
    "CuteDslFusedMoEBase",
    "TrtllmCutedslGroupedGemmBf16Impl",
    "TrtllmCutedslGroupedGemmNvfp4Impl",
    "cutedsl_grouped_gemm_leaf",
    "find_cutedsl_grouped_gemm_leaf",
]
