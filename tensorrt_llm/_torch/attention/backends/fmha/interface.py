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

from __future__ import annotations

import dataclasses
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import cache
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Optional, final

import torch

from tensorrt_llm._torch.attention.backends.cpp_schema import cpp_metadata
from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )
    from tensorrt_llm.functional import RotaryScalingType


@dataclass(slots=True)
class FmhaParams:
    """Schema for native phased-attention parameters.

    Offset contract, relied on by the native side. It splits by memory space:

    * **Device** per-token and per-sequence tensors are phase-local views. Slice them
      here for the context or generation phase; C++ never re-slices them.
    * **Host** tensors, the KV-cache block offsets and the FP4 scaling factors stay
      whole-batch. C++ indexes them with ``seq_offset`` / ``token_offset``: pointer
      accessors apply the offset themselves, so call sites never pass it; only the
      explicit max-over-range queries take a range.

    Applying an offset to the first group double-counts it; omitting it for the second
    shifts every sequence by the number of context requests.
    """

    fwd: AttentionForwardArgs = None
    local_layer_idx: int = -1
    has_fp8_kv_cache: bool = False
    kv_pool: Optional[torch.Tensor] = None
    use_paged_context_fmha: bool = False
    kv_factor: int = 1
    seq_offset: int = 0
    num_seqs: int = 0
    num_tokens: int = 0
    # First query token of this phase on the axis of the q handed to the
    # library. Phase tensors are already sliced; separate per-token inputs
    # such as sparse block tables use this offset to select the same phase.
    token_offset: int = 0
    num_requests: int = 0

    layer_idx: int = -1
    rotary_embedding_base: float = 10000.0
    rotary_embedding_scale_type: RotaryScalingType = 0
    rotary_embedding_scale: float = 1.0
    rotary_embedding_short_mscale: float = 1.0
    rotary_embedding_long_mscale: float = 1.0
    rotary_embedding_max_positions: int = 1024
    rotary_embedding_original_max_positions: int = 1024
    max_context_length: int = 0
    max_seq_len: int = 0
    max_num_requests: int = 0
    # Total sequence rows (max_num_requests * beam_width). The generation workspace and
    # the multi-block counter are sized per sequence, not per request.
    max_num_sequences: int = 0
    beam_width: int = 1
    use_spec_decoding: bool = False
    is_spec_dec_tree: bool = True
    force_prepare_spec_dec_tree_mask: bool = False
    attention_chunk_size: Optional[int] = None

    workspace: torch.Tensor = None
    output: torch.Tensor = None
    skip_softmax_stat: Optional[torch.Tensor] = cpp_metadata(dtype=torch.uint32)
    qkv_or_q: torch.Tensor = None
    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None

    sequence_length: torch.Tensor = cpp_metadata(dtype=torch.int32)
    host_past_key_value_lengths: torch.Tensor = None
    # CPU totals [context, generation] for callers whose past lengths exclude new tokens.
    host_total_kv_lens: Optional[torch.Tensor] = None
    context_lengths: torch.Tensor = cpp_metadata(dtype=torch.int32)
    host_context_lengths: torch.Tensor = None
    max_context_q_len_override: Optional[int] = None
    kv_cache_block_offsets: Optional[torch.Tensor] = None
    host_kv_cache_pool_pointers: Optional[torch.Tensor] = None
    host_kv_cache_pool_mapping: Optional[torch.Tensor] = None
    cache_indirection: Optional[torch.Tensor] = cpp_metadata(dtype=torch.int32)
    max_attention_window_size: int = 0
    cyclic_attention_window_size: int = 0

    rotary_inv_freq: Optional[torch.Tensor] = cpp_metadata(dtype=torch.float32)
    rotary_cos_sin: Optional[torch.Tensor] = None

    block_ids_per_seq: Optional[torch.Tensor] = cpp_metadata(dtype=torch.int32)

    helix_position_offsets: Optional[torch.Tensor] = cpp_metadata(dtype=torch.int32)
    helix_is_inactive_rank: Optional[torch.Tensor] = cpp_metadata(dtype=torch.bool)

    spec_decoding_generation_lengths: Optional[torch.Tensor] = cpp_metadata(dtype=torch.int32)
    spec_decoding_position_offsets: Optional[torch.Tensor] = cpp_metadata(dtype=torch.int32)
    spec_decoding_packed_mask: Optional[torch.Tensor] = cpp_metadata(dtype=torch.int32)
    spec_decoding_bl_tree_mask_offset: Optional[torch.Tensor] = cpp_metadata(dtype=torch.int64)
    spec_decoding_bl_tree_mask: Optional[torch.Tensor] = cpp_metadata(dtype=torch.uint32)
    spec_bl_tree_first_sparse_mask_offset_kv: Optional[torch.Tensor] = cpp_metadata(
        dtype=torch.int32
    )

    num_sparse_topk: int = 0

    flash_mla_tile_scheduler_metadata: Optional[torch.Tensor] = cpp_metadata(dtype=torch.int32)
    flash_mla_num_splits: Optional[torch.Tensor] = cpp_metadata(dtype=torch.int32)

    trtllm_gen_jit_warmup: bool = False

    is_cross: bool = False

    multi_ctas_kv_counter: Optional[torch.Tensor] = None

    @classmethod
    def _from_arguments(cls, arguments: Mapping[str, object], /, **overrides: object) -> FmhaParams:
        """Build compatibility parameters from same-named legacy arguments."""
        params_fields = dataclasses.fields(cls)
        fields_by_name = {field.name: field for field in params_fields}
        unknown = overrides.keys() - fields_by_name.keys()
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"FmhaParams has no field(s): {names}")
        values = {
            field.name: arguments[field.name] for field in params_fields if field.name in arguments
        }
        values.update(overrides)
        return cls(**values)

    def to_op_params(self) -> Any:
        """Build native parameters from this Python interface."""
        from tensorrt_llm.bindings.internal import thop

        target = thop.FmhaParams()
        build_op_params(target, self)
        return target


@cache
def _native_field_names(source_type: type, native_type: type) -> tuple[str, ...]:
    """Resolve the schema boundary once, independently of per-call field values."""
    return tuple(
        field.name for field in dataclasses.fields(source_type) if hasattr(native_type, field.name)
    )


def build_op_params(target: Any, *sources: object) -> None:
    """Build native parameters from schema dataclasses, with later sources taking priority.

    The generated native bindings define which fields cross the boundary;
    Python-only fields are left alone. A later None suppresses an earlier value
    and leaves the native default intact, including for value-initialized nested structs.
    """
    values = {
        name: getattr(source, name)
        for source in sources
        for name in _native_field_names(type(source), type(target))
    }
    for name, value in values.items():
        if value is None:
            continue
        if dataclasses.is_dataclass(value):
            build_op_params(getattr(target, name), value)
        else:
            setattr(target, name, value)


class FmhaPhase(str, Enum):
    """Attention phase checked by a phased FMHA library."""

    CONTEXT = "context"
    GENERATION = "generation"


class Fmha(ABC):
    """Common runtime contract for TRT-LLM attention FMHA libraries."""

    supports_skip_correction: ClassVar[bool] = False
    supports_block_sparse_inputs: ClassVar[bool] = False
    supports_workspace_reclamation: bool = False
    supports_fp4_mla: ClassVar[bool] = False

    def __init__(self, attn: "TrtllmAttention"):
        self._attn_ref: weakref.ReferenceType["TrtllmAttention"] = weakref.ref(attn)

    @property
    def attn(self) -> "TrtllmAttention":
        attn = self._attn_ref()
        if attn is None:
            raise RuntimeError("The owning TrtllmAttention instance has been garbage collected.")
        return attn

    @classmethod
    @final
    def is_available(cls, attn: "TrtllmAttention") -> bool:
        """Return whether this library can serve the given attention layer.

        Check shared capabilities before the implementation's
        ``_is_available`` hook. Libraries declare their capabilities as class
        attributes and override only the hook for additional static checks.

        Evaluated once per ``FmhaManager`` construction, currently at the end
        of ``TrtllmAttention.update_quant_config()``. Conditions must depend
        only on state finalized before manager construction and invariant for
        its lifetime. Reading state that a model rewrites later, such as a
        remapped ``layer_idx``, silently leaves the library list stale because
        it is not revalidated. Request-varying conditions belong in
        ``is_supported`` instead.
        """
        if attn.skip_correction_threshold > 0.0 and not cls.supports_skip_correction:
            logger.debug(
                f"{cls.__name__} is unavailable: skip-correction is enabled and unsupported."
            )
            return False
        if getattr(attn, "uses_fp4_mla_attention", False) and not cls.supports_fp4_mla:
            return False
        return cls._is_available(attn)

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        """Check implementation-specific static restrictions after capability checks.

        Delegate to ``super()._is_available(attn)`` to reuse a parent hook;
        calling ``is_available`` here would re-enter the shared wrapper.
        """
        return True

    @final
    def is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        """Return whether this library supports the request or requested phase.

        Shared request capability checks belong here, before delegating to
        ``_is_supported``. Libraries override only that hook for their
        request-specific restrictions.

        Forward-varying selection conditions must be represented in
        ``FmhaManager._make_cache_key``. Conditions omitted from that key must
        remain invariant for the attention instance. Size-based conditions
        must also preserve the same result throughout each FMHA cache grid
        cell or add the relevant boundary to the grid's candidate list.
        """
        if (
            forward_args.sparse_runtime_params.block_sparse_inputs is not None
            and not self.supports_block_sparse_inputs
        ):
            logger.debug(f"{type(self).__name__} does not support block-sparse inputs.")
            return False
        return self._is_supported(q, k, v, metadata, forward_args, phase=phase)

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        """Check implementation-specific request restrictions after capability checks.

        Delegate to ``super()._is_supported(...)`` to reuse a parent hook;
        calling ``is_supported`` here would re-enter the shared wrapper.
        """
        return True

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> None:
        raise NotImplementedError
