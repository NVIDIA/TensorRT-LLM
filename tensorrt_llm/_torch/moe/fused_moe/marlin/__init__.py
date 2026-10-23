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
"""Everything Marlin: the two registered leaves and the layers under them.

Bottom to top: :mod:`.identity` (the identity strings, the two published
contracts, the descriptor factory), :mod:`.base`
(:class:`.MarlinFusedMoEBase`), :mod:`.eligibility` (the gates each
``can_implement`` composes), and :mod:`.nvfp4` / :mod:`.w4a16_nvfp4`, one
module per leaf.

Importing this package is what registers the leaves, which is why the lookups
are defined here: they are only correct once both leaf modules have loaded.
"""

from tensorrt_llm.models.modeling_utils import QuantAlgo

from ..impl_contract import canonical_quant, normalize_quant
from ..impl_identity import MOE_IMPL_REGISTRY, MoEImplId
from .base import MarlinFusedMoEBase
from .identity import KERNEL_FUSED_MOE, PROVIDER_MARLIN, TECHNIQUE_CUDA
from .nvfp4 import MarlinCudaNvfp4Impl
from .w4a16_nvfp4 import MarlinCudaW4a16Nvfp4Impl


def find_marlin_leaf(quant_algo: QuantAlgo | None) -> type | None:
    """The leaf implementing ``quant_algo``, or ``None`` if none publishes it."""
    # The same canonicalization resolution does, so a caller naming an alias of
    # a published format reaches the leaf rather than a lookup miss.
    quant = normalize_quant(canonical_quant(quant_algo))
    return MOE_IMPL_REGISTRY.lookup(
        MoEImplId(PROVIDER_MARLIN, TECHNIQUE_CUDA, KERNEL_FUSED_MOE, quant)
    )


def marlin_leaf(quant_algo: QuantAlgo | None) -> type:
    """The leaf implementing ``quant_algo``, raising when none publishes it.

    Resolution does not use this -- it walks ``IMPL_PRIORITY`` and asks
    ``can_implement`` -- so this stays a lookup and never becomes a second
    selection path.
    """
    cls = find_marlin_leaf(quant_algo)
    if cls is not None:
        return cls
    registered = sorted(
        identity.canonical()
        for identity in MOE_IMPL_REGISTRY.identities()
        if identity.provider == PROVIDER_MARLIN
    )
    raise ValueError(
        f"no Marlin implementation for quant="
        f"{normalize_quant(canonical_quant(quant_algo))}; registered: {registered}"
    )


# Only what is consumed outside the package.
__all__ = [
    "MarlinCudaNvfp4Impl",
    "MarlinCudaW4a16Nvfp4Impl",
    "MarlinFusedMoEBase",
    "find_marlin_leaf",
    "marlin_leaf",
]
