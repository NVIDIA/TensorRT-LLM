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
"""Everything TRTLLM-Gen: the registered leaves and the layers under them.

Self-contained, so the ``..fused_moe_trtllm_gen`` module path above it holds
nothing but the ``TRTLLMGenFusedMoE`` name. Bottom to top:

* :mod:`.identity` -- the provider / technique / kernel strings, the two
  published contracts, and the descriptor factory.
* :mod:`.base` -- :class:`.TrtllmGenFusedMoEBase`, the abstract root every leaf
  shares. Reads the two axes as leaf-declared attributes, never branches on
  ``quant_config`` or the provider string.
* :mod:`.fp4_block_scale` and :mod:`.fp8_block_scale` -- one module per kernel
  ABI, each holding ``run_moe`` plus a class per format that calls it.
  Several formats share the fp4 kernel; the fp8 one stands alone.
* :mod:`.eligibility` -- the checks each ``can_implement`` composes.
* :mod:`.kernel_inputs` -- the ``run_moe`` prologue, as free functions.

One module per leaf, named ``<provider>_<quant>``. Importing this package is
what registers them, which is why ``trtllm_gen_leaf`` is defined at the bottom
of this file: the lookup is only correct once every leaf module has loaded.

Each format has exactly one leaf: every quantized format runs on the native
``trtllm`` provider, and the unquantized bf16 path on FlashInfer.
"""

from tensorrt_llm.models.modeling_utils import QuantAlgo

from ..impl_contract import canonical_quant, check_quant_matches_identity, normalize_quant
from ..impl_identity import MOE_IMPL_REGISTRY, MoEImplId
from .base import TrtllmGenFusedMoEBase
from .eligibility import (
    check_flashinfer_shard_alignment,
    check_no_activation_constants,
    check_no_expert_bias,
    check_trtllm_gen_capabilities,
    check_trtllm_gen_leaf,
)
from .flashinfer_bf16 import FlashinferTrtllmGenBf16Impl
from .fp4_block_scale import (
    TRTLLMGenFp4BlockScaleBase,
    TRTLLMGenNvfp4Base,
    TRTLLMGenW4a8Mxfp4Mxfp8Base,
    TRTLLMGenW4a16Mxfp4Base,
)
from .fp8_block_scale import TRTLLMGenFp8BlockScalesBase
from .identity import (
    KERNEL_FUSED_MOE,
    PROVIDER_FLASHINFER,
    PROVIDER_TRTLLM,
    TECHNIQUE_TRTLLM_GEN,
    TRTLLM_GEN_CAPABILITIES,
    TRTLLM_GEN_INPUT_REQUIREMENT,
    trtllm_gen_descriptor,
)
from .trtllm_fp8_block_scales import TrtllmTrtllmGenFp8BlockScalesImpl
from .trtllm_nvfp4 import TrtllmTrtllmGenNvfp4Impl
from .trtllm_w4a8_mxfp4_fp8 import TrtllmTrtllmGenW4a8Mxfp4Fp8Impl
from .trtllm_w4a8_mxfp4_mxfp8 import TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl
from .trtllm_w4a8_nvfp4_fp8 import TrtllmTrtllmGenW4a8Nvfp4Fp8Impl
from .trtllm_w4a16_mxfp4 import TrtllmTrtllmGenW4a16Mxfp4Impl


def trtllm_gen_leaf(quant_algo: QuantAlgo | None) -> type:
    """The leaf implementing ``quant_algo``.

    For callers that know the format and want the class rather than a verdict.
    Resolution does not use this -- it walks ``IMPL_PRIORITY`` and asks
    ``can_implement`` -- so this stays a lookup and never becomes a second
    selection path.

    Raises when no leaf publishes the format. A caller enumerating formats
    wants ``find_trtllm_gen_leaf``, where absence is an ordinary answer.
    """
    cls = find_trtllm_gen_leaf(quant_algo)
    if cls is not None:
        return cls
    registered = sorted(
        identity.canonical()
        for identity in MOE_IMPL_REGISTRY.identities()
        if identity.technique == TECHNIQUE_TRTLLM_GEN
    )
    raise ValueError(
        f"no TRTLLM-Gen implementation for quant="
        f"{normalize_quant(canonical_quant(quant_algo))}; registered: {registered}"
    )


def find_trtllm_gen_leaf(quant_algo: QuantAlgo | None) -> type | None:
    """The leaf implementing ``quant_algo``, or ``None`` if none publishes it.

    Goes through the registry rather than a table of its own, so a leaf that is
    renamed or unregistered disappears from here too. The providers publish
    disjoint formats, so at most one of them answers.
    """
    # The same two steps resolution takes (``moe_resolution`` canonicalizes,
    # ``MoEImplId`` folds case), so a caller naming NVFP4_AWQ or MIXED_PRECISION
    # here reaches the leaf resolution would pick rather than a lookup miss.
    quant = normalize_quant(canonical_quant(quant_algo))
    for provider in (PROVIDER_TRTLLM, PROVIDER_FLASHINFER):
        cls = MOE_IMPL_REGISTRY.lookup(
            MoEImplId(provider, TECHNIQUE_TRTLLM_GEN, KERNEL_FUSED_MOE, quant)
        )
        if cls is not None:
            return cls
    return None


__all__ = [
    # trtllm provider
    "TrtllmTrtllmGenNvfp4Impl",
    "TrtllmTrtllmGenFp8BlockScalesImpl",
    "TrtllmTrtllmGenW4a16Mxfp4Impl",
    "TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl",
    "TrtllmTrtllmGenW4a8Nvfp4Fp8Impl",
    "TrtllmTrtllmGenW4a8Mxfp4Fp8Impl",
    # flashinfer provider
    "FlashinferTrtllmGenBf16Impl",
    # shared layers
    "TrtllmGenFusedMoEBase",
    "TRTLLMGenFp4BlockScaleBase",
    "TRTLLMGenNvfp4Base",
    "TRTLLMGenW4a16Mxfp4Base",
    "TRTLLMGenW4a8Mxfp4Mxfp8Base",
    "TRTLLMGenFp8BlockScalesBase",
    # identity
    "PROVIDER_TRTLLM",
    "PROVIDER_FLASHINFER",
    "TECHNIQUE_TRTLLM_GEN",
    "KERNEL_FUSED_MOE",
    "TRTLLM_GEN_CAPABILITIES",
    "TRTLLM_GEN_INPUT_REQUIREMENT",
    "trtllm_gen_leaf",
    "find_trtllm_gen_leaf",
    "trtllm_gen_descriptor",
    "check_trtllm_gen_leaf",
    "check_trtllm_gen_capabilities",
    "check_quant_matches_identity",
    "check_flashinfer_shard_alignment",
    "check_no_activation_constants",
    "check_no_expert_bias",
]
