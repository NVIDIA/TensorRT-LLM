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
"""The CUTLASS-family leaves, their base and the leaf lookups.

One module per leaf, named for its quantization format. Importing this package
registers the leaves.
"""

from tensorrt_llm.models.modeling_utils import QuantAlgo

from ..impl_contract import canonical_quant, identity_quant_of, normalize_quant
from ..impl_environment import collect_moe_environment
from .base import CutlassFusedMoEBase
from .fp8 import TrtllmCutlassFp8Impl
from .fp8_block_scales_hopper import DeepgemmCudaHopperFp8BlockScalesImpl
from .fp8_block_scales_triton import TrtllmTritonFp8BlockScalesImpl
from .lora import CutlassMoELoraMixin, raise_moe_lora_multichunk_unsupported
from .mxfp8 import TrtllmCutlassMxfp8Impl
from .nvfp4 import TrtllmCutlassNvfp4Impl
from .unquantized import TrtllmCutlassUnquantizedImpl
from .w4a8_awq import TrtllmCutlassW4a8AwqImpl
from .w4a8_mxfp4_fp8 import TrtllmCutlassW4a8Mxfp4Fp8Impl
from .w4a8_mxfp4_mxfp8 import TrtllmCutlassW4a8Mxfp4Mxfp8Impl
from .w4a16 import TrtllmCutlassW4a16Impl
from .w4a16_mxfp4 import TrtllmCutlassW4a16Mxfp4Impl
from .w4a16_nvfp4 import TrtllmCutlassW4a16Nvfp4Impl
from .w8a16 import TrtllmCutlassW8a16Impl

#: Every leaf in this family, in the order ``IMPL_PRIORITY`` ranks them and
#: ``moe_resolution`` falls back to them. The order cannot change which leaf
#: admits a problem: the ``quant`` segments are disjoint, except for the two
#: ``fp8_block_scales`` leaves, whose SM sets are.
CUTLASS_LEAVES: tuple[type[CutlassFusedMoEBase], ...] = (
    DeepgemmCudaHopperFp8BlockScalesImpl,
    TrtllmTritonFp8BlockScalesImpl,
    TrtllmCutlassNvfp4Impl,
    TrtllmCutlassW4a16Nvfp4Impl,
    TrtllmCutlassMxfp8Impl,
    TrtllmCutlassW4a8Mxfp4Fp8Impl,
    TrtllmCutlassW4a8Mxfp4Mxfp8Impl,
    TrtllmCutlassW4a16Mxfp4Impl,
    TrtllmCutlassW4a8AwqImpl,
    TrtllmCutlassW4a16Impl,
    TrtllmCutlassW8a16Impl,
    TrtllmCutlassFp8Impl,
    TrtllmCutlassUnquantizedImpl,
)


def cutlass_leaves_in_resolution_order(
    quant_algo: QuantAlgo | None,
) -> tuple[type[CutlassFusedMoEBase], ...]:
    """Every leaf publishing ``quant_algo``, in ``CUTLASS_LEAVES`` order.

    One entry for most formats and two for ``fp8_block_scales``, so a caller
    asking "would a run serve this?" has to ask each, as resolution does.
    """
    quant = normalize_quant(canonical_quant(quant_algo))
    return tuple(leaf for leaf in CUTLASS_LEAVES if identity_quant_of(leaf) == quant)


def find_cutlass_leaf(
    quant_algo: QuantAlgo | None, *, sm: int | None = None
) -> type[CutlassFusedMoEBase] | None:
    """The leaf implementing ``quant_algo``, or ``None`` if none publishes it.

    A lookup, not an eligibility check, and resolution does not use it. ``sm``
    only picks between the two ``fp8_block_scales`` leaves and defaults to the
    SM resolution sees (``collect_moe_environment``, so an
    ``override_moe_environment`` applies here too).

    Args:
        quant_algo: The layer's quantization; ``None`` means unquantized.
        sm: SM version to pick an ``fp8_block_scales`` leaf for.

    Returns:
        The leaf class, or ``None`` when no leaf publishes the format (or, for
        ``fp8_block_scales``, none runs on ``sm``).
    """
    leaves = cutlass_leaves_in_resolution_order(quant_algo)
    if len(leaves) <= 1:
        return leaves[0] if leaves else None
    if sm is None:
        sm = collect_moe_environment().sm
    return next((leaf for leaf in leaves if leaf.sm_support.accepts(sm)), None)


def cutlass_leaf(
    quant_algo: QuantAlgo | None, *, sm: int | None = None
) -> type[CutlassFusedMoEBase]:
    """As :func:`find_cutlass_leaf`, but raising when no leaf serves the format."""
    leaf = find_cutlass_leaf(quant_algo, sm=sm)
    if leaf is not None:
        return leaf
    where = ""
    if cutlass_leaves_in_resolution_order(quant_algo):
        # The format is published, so the SM is what turned it away.
        where = f" on SM{collect_moe_environment().sm if sm is None else sm}"
    registered = sorted(impl.descriptor.identity.canonical() for impl in CUTLASS_LEAVES)
    raise ValueError(
        f"no CUTLASS implementation for quant="
        f"{normalize_quant(canonical_quant(quant_algo))}{where}; registered: {registered}"
    )


__all__ = [
    "CUTLASS_LEAVES",
    "CutlassFusedMoEBase",
    "CutlassMoELoraMixin",
    "DeepgemmCudaHopperFp8BlockScalesImpl",
    "TrtllmCutlassFp8Impl",
    "TrtllmCutlassMxfp8Impl",
    "TrtllmCutlassNvfp4Impl",
    "TrtllmCutlassUnquantizedImpl",
    "TrtllmCutlassW4a8AwqImpl",
    "TrtllmCutlassW4a8Mxfp4Fp8Impl",
    "TrtllmCutlassW4a8Mxfp4Mxfp8Impl",
    "TrtllmCutlassW4a16Impl",
    "TrtllmCutlassW4a16Mxfp4Impl",
    "TrtllmCutlassW4a16Nvfp4Impl",
    "TrtllmCutlassW8a16Impl",
    "TrtllmTritonFp8BlockScalesImpl",
    "cutlass_leaf",
    "cutlass_leaves_in_resolution_order",
    "find_cutlass_leaf",
    "raise_moe_lora_multichunk_unsupported",
]
