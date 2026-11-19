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
"""Everything PrimsTS: the registered leaves and the layers under them.

Bottom to top: :mod:`.identity` (the identity strings, the shared contracts,
the descriptor factory), :mod:`.base` (:class:`.PrimsTsFusedMoEBase`),
:mod:`.eligibility` (the gates each ``can_implement`` composes), and
:mod:`.nvfp4` / :mod:`.w4a8_mxfp4_mxfp8` / :mod:`.bf16`, one module per leaf.
The kernels themselves are FlashInfer's, vendored under
``tensorrt_llm/_torch/moe/kernels/prims_ts`` and launched by
``tensorrt_llm/_torch/moe/custom_ops/prims_ts_moe.py``.

Importing this package is what registers the leaves, which is why the lookups
are defined here: they are only correct once every leaf module has loaded.
"""

from tensorrt_llm.models.modeling_utils import QuantAlgo

from ..impl_contract import canonical_quant, normalize_quant
from ..impl_identity import MOE_IMPL_REGISTRY
from .base import PrimsTsFusedMoEBase
from .bf16 import FlashinferPrimsTsBf16Impl
from .identity import PRIMS_TS_FAMILY, prims_ts_identity
from .nvfp4 import FlashinferPrimsTsNvfp4Impl
from .w4a8_mxfp4_mxfp8 import FlashinferPrimsTsW4a8Mxfp4Mxfp8Impl


def find_prims_ts_leaf(quant_algo: QuantAlgo | None) -> type | None:
    """The PrimsTS leaf implementing ``quant_algo``, or ``None``."""
    # The same canonicalization resolution does, so a caller naming an alias of
    # a published format reaches the leaf rather than a lookup miss.
    quant = normalize_quant(canonical_quant(quant_algo))
    return MOE_IMPL_REGISTRY.lookup(prims_ts_identity(quant))


def prims_ts_leaf(quant_algo: QuantAlgo | None) -> type:
    """As :func:`find_prims_ts_leaf`, but raising when no leaf serves the format.

    Resolution does not use this -- it walks ``IMPL_PRIORITY`` and asks
    ``can_implement`` -- so this stays a lookup and never becomes a second
    selection path.
    """
    cls = find_prims_ts_leaf(quant_algo)
    if cls is not None:
        return cls
    registered = sorted(
        identity.canonical() for identity, _ in MOE_IMPL_REGISTRY.find(PRIMS_TS_FAMILY)
    )
    raise ValueError(
        f"no PrimsTS implementation for quant="
        f"{normalize_quant(canonical_quant(quant_algo))}; registered: {registered}"
    )


# Only what is consumed outside the package.
__all__ = [
    "FlashinferPrimsTsBf16Impl",
    "FlashinferPrimsTsNvfp4Impl",
    "FlashinferPrimsTsW4a8Mxfp4Mxfp8Impl",
    "PrimsTsFusedMoEBase",
    "find_prims_ts_leaf",
    "prims_ts_leaf",
]
