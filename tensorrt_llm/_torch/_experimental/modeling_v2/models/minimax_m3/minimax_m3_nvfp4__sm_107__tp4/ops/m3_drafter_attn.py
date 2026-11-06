# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Eagle3 drafter layer's decode attention on Rubin: ``M3DrafterFullCtxFmha``.

The drafter layer is a plain ``TrtllmAttention`` over its own FP8 paged cache (the full context, every draft pass).
On SM 10.7 its generation phase runs ``thop.attention``'s in-tree trtllm-gen runner, whose split is capped at 8 CTAs
per (request, KV head): at TP4 (1 KV head per rank) 45.4 us per pass at 165k tokens on 212 SMs.

This library keeps the in-tree preprocessing -- ``thop.trtllm_gen_generation_preprocess``: RoPE, the FP8 K/V append
into the paged cache and the FP8 query, as ``FlashInferTrtllmGenFmha`` runs it -- and attends with the full-context
PrimTS split-KV decode of ``ops/m3_fullctx_decode`` at the same pinned recipes as the dense layers.

With ``fused_rope_mode`` set, the preprocess and the query cast run as one launch instead
(``minimax_m3_sm107tp4_drafter_rope_fp8_kv``, the same bytes into the query and the cache), the layer's sub-page
pool and block table taken from the KV cache pool pointers and mapping the preprocess reads.

It serves generation-only steps of the covered (query rows per request, requests) shapes; the target installs it
first in the drafter layer's FMHA list (``install``), so every other step falls through to the stock libraries.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING as TYPE_CHECKING
from typing import Mapping as Mapping
from typing import Optional as Optional
from typing import Tuple as Tuple
from typing import cast

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.attention import (
    drafter as _drafter_shared,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.attention.drafter import (
    DrafterFullCtxBase,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.attention.drafter import (
    drafter_rope_applies as _shared_rope_applies,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.attention.drafter import (
    initialize_drafter_state as _shared_initialize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.attention.drafter import (
    install as _shared_install,
)
from tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen import (
    FlashInferTrtllmGenFmha as FlashInferTrtllmGenFmha,
)
from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase as FmhaPhase
from tensorrt_llm._torch.attention.backends.fmha.phased import FmhaParams as FmhaParams
from tensorrt_llm._torch.attention.backends.fmha.utils import (
    get_attention_chunk_size as get_attention_chunk_size,
)
from tensorrt_llm._torch.attention.backends.fmha.utils import get_bmm1_scale as get_bmm1_scale
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionInputType as AttentionInputType,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.paged_cache import (
    fp8_e4m3_saturate as fp8_e4m3_saturate,
)
from tensorrt_llm._utils import TensorWrapper as TensorWrapper
from tensorrt_llm._utils import convert_to_torch_tensor as convert_to_torch_tensor
from tensorrt_llm.bindings.internal import thop as thop
from tensorrt_llm.functional import PositionEmbeddingType as PositionEmbeddingType
from tensorrt_llm.quantization.mode import QuantMode as QuantMode

from . import m3_fullctx_decode as fullctx

# With PDL, the fused RoPE launch takes the maximum shared-memory carveout so the drafter decode's large-smem CTAs
# co-reside with it (timing only).
ROPE_CARVEOUT = True


def fwd_has_no_mrope(fwd) -> bool:
    return getattr(fwd, "mrope_position_deltas", None) is None


if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention


class M3DrafterFullCtxFmha(DrafterFullCtxBase):
    """Generation phase of the drafter layer: in-tree preprocess, PrimTS split-KV decode."""

    FULLCTX = fullctx
    GEOMETRY_ERROR = "not the TP4 FP8 drafter geometry"

    def _launch_rope(self, *args) -> None:
        torch.ops.trtllm.minimax_m3_sm107tp4_drafter_rope_fp8_kv(*args, ROPE_CARVEOUT)

    def __init__(
        self,
        attn: "TrtllmAttention",
        recipes: Mapping[Tuple[int, int], Mapping],
        max_kv_len: int,
        own_reducer: bool,
        on_taken,
        on_uncovered,
        on_missed,
        fused_rope_mode: Optional[int] = None,
        pdl: bool = False,
        on_rope_taken=None,
    ) -> None:
        super().__init__(attn)
        _shared_initialize(
            self,
            _DRAFTER_BINDINGS,
            recipes,
            max_kv_len,
            own_reducer,
            on_taken,
            on_uncovered,
            on_missed,
            fused_rope_mode,
            pdl,
            on_rope_taken,
            has_one_request_multi=False,
        )

    def _fused_rope_applies(self, params: FmhaParams, attention_input: torch.Tensor) -> bool:
        return _shared_rope_applies(
            self, _DRAFTER_BINDINGS, params, attention_input, has_one_request_multi=False
        )


def install(
    draft_model: torch.nn.Module,
    recipes,
    max_kv_len: int,
    own_reducer: bool,
    pdl: bool,
    on_taken,
    on_uncovered,
    on_missed,
    fused_rope_mode: Optional[int] = None,
    on_rope_taken=None,
) -> int:
    """Prebuild the plans of ``recipes`` and put ``M3DrafterFullCtxFmha`` first in the FMHA list of the TrtllmAttention
    backend of every attention module of ``draft_model``. Call before warmup. Returns the number of attention layers it
    was installed on."""
    return _shared_install(
        draft_model=draft_model,
        recipes=recipes,
        max_kv_len=max_kv_len,
        own_reducer=own_reducer,
        pdl=pdl,
        on_taken=on_taken,
        on_uncovered=on_uncovered,
        on_missed=on_missed,
        fused_rope_mode=fused_rope_mode,
        on_rope_taken=on_rope_taken,
        prebuild=fullctx.prebuild,
        fmha_class=M3DrafterFullCtxFmha,
    )


_DRAFTER_BINDINGS = cast(_drafter_shared.DrafterBindings, sys.modules[__name__])
