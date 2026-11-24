# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""K3-owned fused-verify cache payload and replay state."""

from __future__ import annotations

import importlib
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import NamedTuple

import torch

from tensorrt_llm._utils import prefer_pinned

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
    MambaAcceptanceBatch,
    MambaLayerCache,
    MambaRole,
    MambaStateLayout,
    _get_local_mamba_cache_layout,
    _mamba_effective_tp_size,
    _promote_intermediate_states,
    _stack_state_views,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    DEFAULT_BEAM_INDEX,
    BufferConfig,
    DataRole,
    _KVCache,
)

_MAMBA_SSM_SEED_MASK = (1 << 62) - 1
_MAMBA_SSM_UINT64_MASK = (1 << 64) - 1
_MAMBA_SSM_SEED_BASE = 0x6A09E667F3BCC908
_MAMBA_SSM_SEED_MIX_COUNTER = 0x2545F4914F6CDD1D
_MAMBA_SSM_SEED_MIX_SLOT = 0x1B873593CC9E2D51
_MAMBA_SSM_SEED_MIX_RANK = 0x9E3779B97F4A7C15


def _splitmix64(value: int) -> int:
    """Return the SplitMix64 finalizer for ``value``."""
    value = (value + 0x9E3779B97F4A7C15) & _MAMBA_SSM_UINT64_MASK
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _MAMBA_SSM_UINT64_MASK
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & _MAMBA_SSM_UINT64_MASK
    value ^= value >> 31
    return value & _MAMBA_SSM_UINT64_MASK


def compute_deterministic_mamba_seed(counter: int, slot: int, rank_offset: int) -> int:
    """Return a reproducible, nonzero int64 Philox seed."""
    folded = (
        _MAMBA_SSM_SEED_BASE
        + counter * _MAMBA_SSM_SEED_MIX_COUNTER
        + slot * _MAMBA_SSM_SEED_MIX_SLOT
        + rank_offset * _MAMBA_SSM_SEED_MIX_RANK
    ) & _MAMBA_SSM_UINT64_MASK
    seed = _splitmix64(folded) & _MAMBA_SSM_SEED_MASK
    return seed or 1


def allocate_mamba_seed_buffer(
    cache_size: int, rank_offset: int, device: torch.device
) -> torch.Tensor:
    """Allocate the graph-stable per-slot Philox seed buffer."""
    seeds = [compute_deterministic_mamba_seed(0, slot, rank_offset) for slot in range(cache_size)]
    return torch.tensor(seeds, dtype=torch.int64, device=device)


@dataclass(frozen=True, kw_only=True)
class KDAReplayLayerCache(MambaLayerCache):
    """Per-layer KDA replay tensors consumed by fused multi-token verify."""

    prev_num_accepted_tokens: torch.Tensor | None = field(
        default=None,
        metadata={"slot_shared": True},
    )
    kda_conv_q: torch.Tensor | None = None
    kda_conv_k: torch.Tensor | None = None
    kda_conv_v: torch.Tensor | None = None
    kda_k_cache: torch.Tensor | None = None
    kda_g_cache: torch.Tensor | None = None
    kda_v_cache: torch.Tensor | None = None
    kda_beta_cache: torch.Tensor | None = None
    # [slots, 3, num_spec, heads, head_dim]: fp32 vn, beta * k, and decay.
    kda_state_tok: torch.Tensor | None = None

    @property
    def has_kda_replay_caches(self) -> bool:
        return self.kda_conv_q is not None

    def commit_conv_window(self, slot_indices: torch.Tensor, conv_pool: torch.Tensor) -> None:
        """Seed the replay convolution windows from the live pool."""
        from ._kda_kernels import copy_kda_replay_conv_window

        copy_kda_replay_conv_window(
            conv_pool,
            self.kda_conv_q,
            self.kda_conv_k,
            self.kda_conv_v,
            slot_indices,
        )


_KDA_BETA_CACHE_ALIGNMENT_BYTES = 16
# KVCM copies 16-byte payloads; the replay kernel reads only the first int32.
_KDA_REPLAY_COUNT_BYTES = 16
_KDA_REPLAY_COUNT_ROLE = DataRole("prev_num_accepted_tokens")
_KDA_REPLAY_CONV_ROLES = ("kda_conv_q", "kda_conv_k", "kda_conv_v")
_KDA_REPLAY_DRAFT_ROLES = ("kda_k_cache", "kda_g_cache", "kda_v_cache", "kda_beta_cache")
_KDA_DRAFT_RECORDS_ROLE = "kda_state_tok"
# The Kimi K3 (k3_rubin) package's KDA kernels and metadata.
_K3_KDA_PACKAGE = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.kda"
# Seed the replay caches of transferred requests in all layers with one launch
# per buffer kind (the K3 package's seed_kda_replay_layers) instead of a loop of
# small copies per layer. Falls back to the loop when the package cannot be
# imported, the states are not on a GPU, or the layers' views differ in layout.
_KDA_BATCHED_REPLAY_SEED = os.environ.get("TRTLLM_KDA_BATCHED_REPLAY_SEED", "1") == "1"


def _is_disagg_context_server(is_disagg: bool) -> bool:
    """Whether this worker only serves the context phase of disagg requests."""
    return is_disagg and os.getenv("TRTLLM_DISAGG_ROLE") == "context"


def _kda_replay_layout(
    num_spec: int, num_heads: int, head_dim: int, conv_window: int, draft_records: bool = False
) -> dict[str, tuple[torch.dtype, list[int]]]:
    """Per-slot physical shapes of one layer's KDA replay caches.

    Conv caches hold raw bf16 projection inputs (exact in bf16); k / v / beta
    are bf16 and the gate stays fp32 because it feeds the cumulative-gate
    exponents of the replayed chunk. Beta rows keep the 16-byte CuTe
    alignment by padding the head dimension.

    With ``draft_records``, the records of every draft of the last verify
    round (``kda_state_tok``: per draft, the row innovations vn, then beta * k
    and the decay, each fp32 [H, K] with V = K = ``head_dim``) take the place
    of the k / g / v / beta caches.
    """
    dim = num_heads * head_dim
    layout = {
        "kda_conv_q": (torch.bfloat16, [conv_window + num_spec, dim]),
        "kda_conv_k": (torch.bfloat16, [conv_window + num_spec, dim]),
        "kda_conv_v": (torch.bfloat16, [conv_window + num_spec, dim]),
    }
    if draft_records:
        layout[_KDA_DRAFT_RECORDS_ROLE] = (torch.float32, [3, num_spec, num_heads, head_dim])
        return layout
    heads_per_alignment = _KDA_BETA_CACHE_ALIGNMENT_BYTES // torch.bfloat16.itemsize
    aligned_heads = (
        (num_heads + heads_per_alignment - 1) // heads_per_alignment
    ) * heads_per_alignment
    layout.update(
        {
            "kda_k_cache": (torch.bfloat16, [num_spec, dim]),
            "kda_g_cache": (torch.float32, [num_spec, dim]),
            "kda_v_cache": (torch.bfloat16, [num_spec, dim]),
            "kda_beta_cache": (torch.bfloat16, [num_spec, aligned_heads]),
        }
    )
    return layout


def _kda_replay_bytes_per_slot(
    layout: dict[str, tuple[torch.dtype, list[int]]], num_local_layers: int
) -> int:
    """Replay bytes one recurrent-state slot holds across local layers."""
    if not layout or num_local_layers == 0:
        return 0
    layer_bytes = sum(dtype.itemsize * math.prod(shape) for dtype, shape in layout.values())
    return num_local_layers * layer_bytes + _KDA_REPLAY_COUNT_BYTES


def _host_slots_to_device(slots: list[int], device: torch.device) -> torch.Tensor:
    """Copy host slot IDs without synchronizing queued GPU work.

    Each copy uses fresh pinned storage so later host writes cannot race it.
    """
    return torch.tensor(
        slots, dtype=torch.long, pin_memory=device.type == "cuda" and prefer_pinned()
    ).to(device, non_blocking=True)


def _kda_beta_cache_padded_heads(num_heads: int) -> int:
    heads_per_alignment = _KDA_BETA_CACHE_ALIGNMENT_BYTES // torch.bfloat16.itemsize
    return (num_heads + heads_per_alignment - 1) // heads_per_alignment * heads_per_alignment


def _allocate_kda_beta_cache(shape: tuple[int, ...], device: torch.device | None) -> torch.Tensor:
    """Keep logical head count while aligning each physical row for CuTe."""
    return torch.zeros(
        *shape[:-1],
        _kda_beta_cache_padded_heads(shape[-1]),
        dtype=torch.bfloat16,
        device=device,
    )[..., : shape[-1]]


def allocate_kda_replay_fields(
    *,
    num_local_layers: int,
    cache_size: int,
    num_speculative_tokens: int,
    conv_dim: int,
    conv_kernel_size: int,
    num_heads: int,
    device: torch.device,
    draft_records: bool = False,
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    """Allocate the complete KDA fused-verify replay payload."""
    if conv_dim % 3 != 0:
        raise ValueError("KDA replay caches require q/k/v convolution sections of equal width")
    section_dim = conv_dim // 3
    extended_window = conv_kernel_size - 1 + num_speculative_tokens

    def allocate_conv_cache() -> torch.Tensor:
        return torch.zeros(
            num_local_layers,
            cache_size,
            extended_window,
            section_dim,
            dtype=torch.bfloat16,
            device=device,
        ).transpose(-1, -2)

    fields = {
        "prev_num_accepted_tokens": torch.zeros(cache_size, dtype=torch.int32, device=device),
        "kda_conv_q": allocate_conv_cache(),
        "kda_conv_k": allocate_conv_cache(),
        "kda_conv_v": allocate_conv_cache(),
    }
    if draft_records:
        fields["kda_state_tok"] = torch.zeros(
            num_local_layers,
            cache_size,
            3,
            num_speculative_tokens,
            num_heads,
            section_dim // num_heads,
            dtype=torch.float32,
            device=device,
        )
        return fields, [
            value for name, value in fields.items() if name != "prev_num_accepted_tokens"
        ]
    fields.update(
        {
            "kda_k_cache": torch.zeros(
                num_local_layers,
                cache_size,
                num_speculative_tokens,
                section_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            "kda_g_cache": torch.zeros(
                num_local_layers,
                cache_size,
                num_speculative_tokens,
                section_dim,
                dtype=torch.float32,
                device=device,
            ),
            "kda_v_cache": torch.zeros(
                num_local_layers,
                cache_size,
                num_speculative_tokens,
                section_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            "kda_beta_cache": _allocate_kda_beta_cache(
                (num_local_layers, cache_size, num_speculative_tokens, num_heads), device
            ),
        }
    )
    scratch = [
        fields["kda_conv_q"],
        fields["kda_conv_k"],
        fields["kda_conv_v"],
        fields["kda_k_cache"],
        fields["kda_g_cache"],
        fields["kda_v_cache"],
        fields["kda_beta_cache"],
    ]
    return fields, scratch


@dataclass(frozen=True, kw_only=True)
class KDAIntermediateLayerCache(MambaLayerCache):
    """Generic intermediate-state view used during speculative decoding."""

    intermediate_conv_window: torch.Tensor | None = None
    intermediate_ssm: torch.Tensor | None = None


class KDAIntermediateState:
    """Own conventional speculative intermediate states and promotion."""

    def __init__(self) -> None:
        self._ssm_states: tuple[torch.Tensor, ...] = ()
        self._conv_states: tuple[torch.Tensor, ...] = ()
        self._stacked_ssm: torch.Tensor | None = None
        self._stacked_conv: torch.Tensor | None = None
        self.intermediate_ssm: torch.Tensor | None = None
        self.intermediate_conv: torch.Tensor | None = None
        self.intermediate_indices: torch.Tensor | None = None

    def bind(
        self,
        context: MambaStateLayout,
        ssm_states: list[torch.Tensor],
        conv_states: list[torch.Tensor],
    ) -> None:
        """Borrow persistent per-layer views until shutdown; allocate owned scratch."""
        self._ssm_states = tuple(ssm_states)
        self._conv_states = tuple(conv_states)
        self._stacked_ssm = _stack_state_views(ssm_states)
        self._stacked_conv = _stack_state_views(conv_states)
        if not context.mamba_pp_layers:
            return

        device = ssm_states[0].device

        if context.spec_config is not None:
            tokens_per_step = context.spec_config.tokens_per_gen_step
            common_shape = [
                len(context.mamba_pp_layers),
                context.max_batch_size,
                tokens_per_step,
            ]
            self.intermediate_ssm = torch.zeros(
                common_shape + list(context.ssm_state_shape),
                dtype=context.ssm_state_dtype,
                device=device,
            )
            self.intermediate_conv = torch.zeros(
                common_shape + list(context.conv_state_shape),
                dtype=context.conv_state_dtype,
                device=device,
            )
            self.intermediate_indices = torch.arange(
                context.max_batch_size, dtype=torch.int32, device=device
            )

    def _layer_cache_fields(self, layer_offset: int) -> dict[str, torch.Tensor | None]:
        fields: dict[str, torch.Tensor | None] = {}
        if self.intermediate_conv is not None:
            fields["intermediate_conv_window"] = self.intermediate_conv[layer_offset]
        if self.intermediate_ssm is not None:
            fields["intermediate_ssm"] = self.intermediate_ssm[layer_offset]
        return fields

    def reset_slots(self, slots: torch.Tensor, host_slots: list[int]) -> None:
        """Plain intermediate buffers have no request-persistent bookkeeping."""

    def update(self, batch: MambaAcceptanceBatch) -> None:
        if self.intermediate_indices is None:
            return
        _promote_intermediate_states(
            self._ssm_states, self.intermediate_ssm, batch, self._stacked_ssm
        )
        _promote_intermediate_states(
            self._conv_states, self.intermediate_conv, batch, self._stacked_conv
        )

    def shutdown(self) -> None:
        self._ssm_states = ()
        self._conv_states = ()
        self._stacked_ssm = None
        self._stacked_conv = None
        self.intermediate_ssm = None
        self.intermediate_conv = None
        self.intermediate_indices = None

    def make_layer_cache(
        self, layer_offset: int, conv: torch.Tensor, temporal: torch.Tensor
    ) -> MambaLayerCache:
        return KDAIntermediateLayerCache(
            conv=conv, temporal=temporal, **self._layer_cache_fields(layer_offset)
        )


class KDAReplayState:
    """Own all V2 state used by the KDA fused multi-token verifier."""

    @property
    def intermediate_ssm(self) -> torch.Tensor | None:
        return None

    @property
    def intermediate_conv(self) -> torch.Tensor | None:
        return None

    def __init__(
        self,
        num_speculative_tokens: int,
        *,
        stochastic_rounding: bool = False,
        draft_records: bool = False,
    ) -> None:
        self._stochastic_rounding = stochastic_rounding
        self.draft_records = draft_records
        self.pool_layers: list[dict[str, torch.Tensor]] | None = None
        self.num_speculative_tokens = num_speculative_tokens
        self.suspended_requests: set[int] = set()
        self.acceptance_pending: set[int] = set()
        self.unseeded_requests: set[int] = set()
        self._ssm_states: tuple[torch.Tensor, ...] = ()
        self._conv_states: tuple[torch.Tensor, ...] = ()
        self._committed_window = 0
        self.prev_num_accepted_tokens: torch.Tensor | None = None
        self.kda_conv_q: torch.Tensor | None = None
        self.kda_conv_k: torch.Tensor | None = None
        self.kda_conv_v: torch.Tensor | None = None
        self.kda_k_cache: torch.Tensor | None = None
        self.kda_g_cache: torch.Tensor | None = None
        self.kda_v_cache: torch.Tensor | None = None
        self.kda_beta_cache: torch.Tensor | None = None
        self.kda_state_tok: torch.Tensor | None = None
        self.intermediate_indices: torch.Tensor | None = None
        self.rand_seed: torch.Tensor | None = None
        self._seed_request_counter = 0
        self._seed_rank_offset = 0
        self._conv_section_dims: tuple[int, ...] = ()
        # Batched seeding plan of seed_transferred_slots: None until first use,
        # False when the per-layer loop is used.
        self._batched_seed_plan: object | None = None

    def validate(self, context: MambaStateLayout) -> None:
        if context.spec_config is None:
            raise ValueError("KDA replay caches require speculative decoding")
        expected_num_spec = context.spec_config.tokens_per_gen_step - 1
        if self.num_speculative_tokens != expected_num_spec:
            raise ValueError(
                "KDA replay cache width "
                f"({self.num_speculative_tokens}) must match the draft length "
                f"({expected_num_spec})"
            )
        if not context.mamba_pp_layers:
            return
        if context.conv_state_layout != "q_k_v":
            raise ValueError("KDA replay requires conv_state_layout='q_k_v'")
        if len(context.conv_section_dims) != 3 or len(set(context.conv_section_dims)) != 1:
            raise ValueError(
                "KDA replay caches require equal [Q | K | V] convolution-state sections"
            )

    def pool_layout(self, context: MambaStateLayout) -> dict[str, tuple[torch.dtype, list[int]]]:
        if not context.mamba_pp_layers:
            return {}
        if len(context.conv_section_dims) != 3 or len(set(context.conv_section_dims)) != 1:
            raise ValueError(
                "KDA replay caches require equal [Q | K | V] convolution-state sections"
            )
        if self.draft_records and context.ssm_state_shape[1] != context.ssm_state_shape[-1]:
            raise ValueError("KDA draft records require equal value and key head dimensions")
        return _kda_replay_layout(
            self.num_speculative_tokens,
            context.ssm_state_shape[0],
            context.ssm_state_shape[1],
            context.conv_state_shape[1],
            draft_records=self.draft_records,
        )

    def pool_buffer_configs(self, context: MambaStateLayout, layer_id: int) -> list[BufferConfig]:
        layout = self.pool_layout(context)
        buffers = [
            BufferConfig(role=DataRole(name), size=dtype.itemsize * math.prod(shape))
            for name, (dtype, shape) in layout.items()
        ]
        if layout and layer_id == context.mamba_pp_layers[0]:
            buffers.append(BufferConfig(role=_KDA_REPLAY_COUNT_ROLE, size=_KDA_REPLAY_COUNT_BYTES))
        return buffers

    def disagg_ignored_roles(self, context: MambaStateLayout) -> frozenset[DataRole]:
        roles = {DataRole(name) for name in self.pool_layout(context)}
        if roles:
            roles.add(_KDA_REPLAY_COUNT_ROLE)
        return frozenset(roles)

    def _layer_bytes_per_slot(self, section_dim: int, committed_window: int, num_heads: int) -> int:
        extended_window = committed_window + self.num_speculative_tokens
        if self.draft_records:
            return (
                3 * section_dim * extended_window * torch.bfloat16.itemsize
                + 3 * self.num_speculative_tokens * section_dim * torch.float32.itemsize
            )
        bf16_elements = (
            3 * section_dim * extended_window
            + self.num_speculative_tokens * 2 * section_dim
            + self.num_speculative_tokens * _kda_beta_cache_padded_heads(num_heads)
        )
        gate_elements = self.num_speculative_tokens * section_dim
        return bf16_elements * torch.bfloat16.itemsize + gate_elements * torch.float32.itemsize

    def bytes_per_slot(self, context: MambaStateLayout, layer_id: int) -> int:
        if not context.mamba_pp_layers:
            return 0
        shared_bytes = 0
        if layer_id == context.mamba_pp_layers[0]:
            shared_bytes = 4
            if self._stochastic_rounding:
                shared_bytes += 8
        return (
            self._layer_bytes_per_slot(
                context.conv_section_dims[0],
                context.conv_state_shape[1],
                context.ssm_state_shape[0],
            )
            + shared_bytes
        )

    @classmethod
    def estimate_bytes_per_slot(cls, model_config, mapping: Mapping, **kwargs) -> int:
        """Estimate rank-local replay storage using the runtime replay gate and geometry."""
        if _is_disagg_context_server(kwargs.get("is_disagg", False)):
            return 0
        spec_config = kwargs.get("spec_config")
        num_spec = kwargs.get("kda_replay_num_spec")
        if num_spec is None:
            num_spec = get_kda_replay_num_spec(spec_config, manager_supports_replay=True)
        if num_spec is None:
            return 0
        params, local_layers, _ = _get_local_mamba_cache_layout(
            model_config,
            mapping,
            spec_config=spec_config,
            is_draft=kwargs.get("is_draft", False),
            use_separate_draft_kv_cache=kwargs.get("use_separate_draft_kv_cache", False),
        )
        if not local_layers:
            return 0
        tp_size = _mamba_effective_tp_size(mapping)
        quant_config = model_config.quant_config
        stochastic_rounding = kwargs.get(
            "mamba_ssm_stochastic_rounding",
            getattr(quant_config, "mamba_ssm_stochastic_rounding", False)
            and params.mamba_ssm_cache_dtype == torch.float16,
        )
        state = cls(num_spec, draft_records=kwargs.get("kda_draft_records", False))
        return _kda_replay_bytes_per_slot(
            _kda_replay_layout(
                num_spec,
                params.num_heads // tp_size,
                params.head_dim,
                params.conv_kernel - 1,
                draft_records=state.draft_records,
            ),
            local_layers,
        ) + (8 if stochastic_rounding else 0)

    def bind(
        self,
        context: MambaStateLayout,
        ssm_states: list[torch.Tensor],
        conv_states: list[torch.Tensor],
    ) -> None:
        """Borrow persistent views and bind bookkeeping; standalone use allocates replay storage."""
        self._seed_rank_offset = (
            context.mapping.tp_rank * 1_000_003
            + context.mapping.pp_rank * 1_000_033
            + context.mapping.rank * 1_009
        )
        self._conv_section_dims = context.conv_section_dims
        self._ssm_states = tuple(ssm_states)
        self._conv_states = tuple(conv_states)
        self._batched_seed_plan = None
        self._committed_window = context.conv_state_shape[1] if conv_states else 0
        states = ssm_states
        if not states:
            return

        cache_size = context.slot_capacity
        if cache_size is None:
            raise RuntimeError("KDA replay requires the allocated V2 slot capacity")
        device = states[0].device
        if self.pool_layers is None:
            fields, _ = allocate_kda_replay_fields(
                num_local_layers=len(context.mamba_pp_layers),
                cache_size=cache_size,
                num_speculative_tokens=self.num_speculative_tokens,
                conv_dim=context.conv_state_shape[0],
                conv_kernel_size=context.conv_state_shape[1] + 1,
                num_heads=context.ssm_state_shape[0],
                device=device,
                draft_records=self.draft_records,
            )
            for name, value in fields.items():
                setattr(self, name, value)
        self.intermediate_indices = torch.arange(
            context.max_batch_size,
            dtype=torch.int32,
            device=device,
        )
        if self._stochastic_rounding:
            self.rand_seed = allocate_mamba_seed_buffer(
                cache_size,
                self._seed_rank_offset,
                device,
            )
        logger.info(f"Mamba Cache (kda-replay) is allocated for {cache_size} state slots")

    def _layer_cache_fields(self, layer_offset: int) -> dict[str, torch.Tensor | None]:
        if self.pool_layers is not None:
            return {
                **self.pool_layers[layer_offset],
                "prev_num_accepted_tokens": self.prev_num_accepted_tokens,
            }
        return {
            "prev_num_accepted_tokens": self.prev_num_accepted_tokens,
            "kda_conv_q": None if self.kda_conv_q is None else self.kda_conv_q[layer_offset],
            "kda_conv_k": None if self.kda_conv_k is None else self.kda_conv_k[layer_offset],
            "kda_conv_v": None if self.kda_conv_v is None else self.kda_conv_v[layer_offset],
            "kda_k_cache": None if self.kda_k_cache is None else self.kda_k_cache[layer_offset],
            "kda_g_cache": None if self.kda_g_cache is None else self.kda_g_cache[layer_offset],
            "kda_v_cache": None if self.kda_v_cache is None else self.kda_v_cache[layer_offset],
            "kda_beta_cache": (
                None if self.kda_beta_cache is None else self.kda_beta_cache[layer_offset]
            ),
            "kda_state_tok": None
            if self.kda_state_tok is None
            else self.kda_state_tok[layer_offset],
        }

    def make_layer_cache(
        self,
        layer_offset: int,
        conv: torch.Tensor,
        temporal: torch.Tensor,
    ) -> MambaLayerCache:
        return KDAReplayLayerCache(
            conv=conv,
            temporal=temporal,
            **self._layer_cache_fields(layer_offset),
        )

    def reset_slots(self, slots: torch.Tensor, host_slots: list[int]) -> None:
        if self.prev_num_accepted_tokens is None or slots.numel() == 0:
            return
        slots = _host_slots_to_device(host_slots, self.prev_num_accepted_tokens.device)
        self.prev_num_accepted_tokens.index_fill_(0, slots, 0)
        if self.rand_seed is not None:
            self._seed_request_counter += 1
            seeds = [
                compute_deterministic_mamba_seed(
                    self._seed_request_counter,
                    slot,
                    self._seed_rank_offset,
                )
                for slot in host_slots
            ]
            # Fresh pinned staging and a non-blocking upload, as in _host_slots_to_device: a
            # pageable torch.tensor(..., device=) would wait for the decode step in flight.
            seeds = torch.tensor(
                seeds,
                dtype=torch.int64,
                pin_memory=self.rand_seed.is_cuda and prefer_pinned(),
            )
            self.rand_seed.index_copy_(0, slots, seeds.to(self.rand_seed.device, non_blocking=True))

    def relocate_slots(self, old_slots: list[int], new_slots: list[int]) -> None:
        if self.prev_num_accepted_tokens is None:
            return
        if self.pool_layers is not None and self.rand_seed is None:
            return
        moves = [(old, new) for old, new in zip(old_slots, new_slots) if old >= 0 and old != new]
        if not moves:
            return
        device = self.prev_num_accepted_tokens.device
        source_slots = _host_slots_to_device([old for old, _ in moves], device)
        destination_slots = _host_slots_to_device([new for _, new in moves], device)
        if self.pool_layers is None:
            for replay_buffer in self._layer_replay_buffers():
                replay_buffer.index_copy_(
                    1,
                    destination_slots,
                    replay_buffer.index_select(1, source_slots),
                )
            self.prev_num_accepted_tokens.index_copy_(
                0,
                destination_slots,
                self.prev_num_accepted_tokens.index_select(0, source_slots),
            )
        if self.rand_seed is not None:
            self.rand_seed.index_copy_(
                0,
                destination_slots,
                self.rand_seed.index_select(0, source_slots),
            )

    def update(self, batch: MambaAcceptanceBatch) -> None:
        self.record_acceptance(
            batch.destination_state_indices,
            batch.num_accepted_tokens - 1,
            batch.is_dummy_request,
        )

    @torch.inference_mode()
    def seed_transferred_slots(self, state_indices: torch.Tensor) -> None:
        """Seed completed-transfer slots from borrowed persistent conv views."""
        if self.prev_num_accepted_tokens is None or state_indices.numel() == 0:
            return
        committed_window = self._committed_window
        section_offsets = [0]
        for section_dim in self._conv_section_dims:
            section_offsets.append(section_offsets[-1] + section_dim)

        self.prev_num_accepted_tokens.index_fill_(0, state_indices, 0)
        plan = self._get_batched_seed_plan()
        if plan is not None:
            kernels, plan = plan
            kernels.seed_kda_replay_layers(plan, state_indices)
            return
        for layer_offset, conv_state in enumerate(self._conv_states):
            replay = self._layer_cache_fields(layer_offset)
            for name in (*_KDA_REPLAY_CONV_ROLES, *_KDA_REPLAY_DRAFT_ROLES):
                buffer = replay.get(name)
                if buffer is not None:
                    buffer.index_fill_(0, state_indices, 0)
            selected_conv = conv_state.index_select(0, state_indices)
            for section_idx, name in enumerate(_KDA_REPLAY_CONV_ROLES):
                buffer = replay[name]
                section = selected_conv[
                    :, section_offsets[section_idx] : section_offsets[section_idx + 1], :
                ]
                buffer[:, :, :committed_window].index_copy_(
                    0, state_indices, section.to(buffer.dtype)
                )

    def _get_batched_seed_plan(self) -> tuple[object, object] | None:
        """The K3 package's seed kernels and their plan over the bound views, if usable."""
        if self._batched_seed_plan is None:
            self._batched_seed_plan = self._plan_batched_seed() or False
        return self._batched_seed_plan or None

    def _plan_batched_seed(self) -> tuple[object, object] | None:
        conv_states = self._conv_states
        if (
            not _KDA_BATCHED_REPLAY_SEED
            or not conv_states
            or not conv_states[0].is_cuda
            or len(self._conv_section_dims) != 3
            or len(set(self._conv_section_dims)) != 1
        ):
            return None
        try:
            kernels = importlib.import_module(f"{_K3_KDA_PACKAGE}._kda_kernels")
        except ImportError as error:
            logger.warning(
                "TRTLLM_KDA_BATCHED_REPLAY_SEED: the K3 package's seed kernels are "
                f"unavailable ({error}); seeding KDA replay caches layer by layer."
            )
            return None
        layers = [self._layer_cache_fields(offset) for offset in range(len(conv_states))]
        plan = kernels.plan_kda_replay_seed(
            conv_states,
            [tuple(layer[name] for name in _KDA_REPLAY_CONV_ROLES) for layer in layers],
            [
                [layer[name] for layer in layers]
                for name in _KDA_REPLAY_DRAFT_ROLES
                if layers[0].get(name) is not None
            ],
        )
        if plan is None or plan.committed != self._committed_window:
            logger.warning(
                "TRTLLM_KDA_BATCHED_REPLAY_SEED: the KDA layers' replay buffers do not "
                "share one layout; seeding KDA replay caches layer by layer."
            )
            return None
        logger.info(
            f"KDA replay caches of transferred requests are seeded in all {len(conv_states)} "
            "local KDA layers with one launch per buffer kind"
        )
        return kernels, plan

    def record_acceptance(
        self,
        state_indices: torch.Tensor,
        num_accepted_draft_tokens: torch.Tensor,
        is_dummy_request: torch.Tensor | None,
    ) -> None:
        """Record accepted drafts, preserving dummy rows; each input has shape [batch]."""
        if self.prev_num_accepted_tokens is None:
            raise RuntimeError("KDA replay acceptance requires replay bookkeeping")
        slots = state_indices.to(torch.long)
        accepted = num_accepted_draft_tokens.to(torch.int32).clamp(min=0)
        if is_dummy_request is not None:
            current = self.prev_num_accepted_tokens[slots]
            accepted = torch.where(is_dummy_request, current, accepted)
        self.prev_num_accepted_tokens[slots] = accepted

    @staticmethod
    @contextmanager
    def on_cache_stream(stream):
        """Order replay commits and seeding against V2 page migration."""
        current = torch.cuda.current_stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            yield
        current.wait_stream(stream)

    def commit_pending_drafts(self, slot: int) -> None:
        """Fold accepted drafts and the replay conv window into the persistent states."""
        from ._kda_kernels import commit_kda_pending_drafts

        assert self.prev_num_accepted_tokens is not None
        slots = torch.full(
            (1,), slot, dtype=torch.int32, device=self.prev_num_accepted_tokens.device
        )
        num_accepted = self.prev_num_accepted_tokens.index_select(0, slots)
        for layer, (ssm_states, conv_states) in enumerate(zip(self._ssm_states, self._conv_states)):
            replay = self._layer_cache_fields(layer)
            commit_kda_pending_drafts(
                ssm_states,
                conv_states,
                replay["kda_conv_q"],
                replay["kda_conv_k"],
                replay["kda_conv_v"],
                replay.get("kda_k_cache"),
                replay.get("kda_g_cache"),
                replay.get("kda_v_cache"),
                replay.get("kda_beta_cache"),
                slots,
                num_accepted,
                state_tok=replay.get(_KDA_DRAFT_RECORDS_ROLE),
            )
        # zero_() on the element view stays stream-ordered; assigning a Python scalar
        # through the index stages it on the host and waits for the step in flight.
        self.prev_num_accepted_tokens[slot].zero_()

    def _layer_replay_buffers(self) -> tuple[torch.Tensor, ...]:
        buffers = (
            self.kda_conv_q,
            self.kda_conv_k,
            self.kda_conv_v,
            self.kda_k_cache,
            self.kda_g_cache,
            self.kda_v_cache,
            self.kda_beta_cache,
            self.kda_state_tok,
        )
        if any(buffer is None for buffer in buffers[:3]):
            raise RuntimeError("KDA replay buffers are not bound")
        return tuple(buffer for buffer in buffers if buffer is not None)

    def shutdown(self) -> None:
        self.pool_layers = None
        self._ssm_states = ()
        self._conv_states = ()
        self.suspended_requests.clear()
        self.acceptance_pending.clear()
        self.unseeded_requests.clear()
        self.prev_num_accepted_tokens = None
        self.kda_conv_q = None
        self.kda_conv_k = None
        self.kda_conv_v = None
        self.kda_k_cache = None
        self.kda_g_cache = None
        self.kda_v_cache = None
        self.kda_beta_cache = None
        self.kda_state_tok = None
        self.intermediate_indices = None
        self.rand_seed = None
        self._batched_seed_plan = None


def bind_kda_replay_pool(state: KDAReplayState, manager: MambaHybridCacheManagerV2) -> None:
    """Bind model-owned KDA replay roles to views of the manager's native pools."""
    layout = state.pool_layout(manager._state_layout)
    if not layout:
        return
    required = {
        (manager.layer_offsets[layer_id], str(role))
        for layer_id in manager.mamba_pp_layers
        for role in (MambaRole.SSM_STATE, MambaRole.CONV_STATE, *layout)
    }
    required.add((manager.layer_offsets[manager.mamba_pp_layers[0]], str(_KDA_REPLAY_COUNT_ROLE)))
    if not any(
        required.issubset(
            {
                (int(buffer_id.layer_id), str(buffer_id.role))
                for coalesced in variant.coalesced_buffers
                for buffer_id in coalesced.buffer_ids
            }
        )
        for group in manager.impl.pool_group_descs
        for variant in group.slot_desc.variants
    ):
        raise RuntimeError(
            "KDA replay and recurrent-state buffers must share one KVCM "
            "pool group and lifecycle to reuse the same logical slot index"
        )

    state.pool_layers = []
    state._batched_seed_plan = None
    for layer_id in manager.mamba_pp_layers:
        layer = {}
        for name, (dtype, shape) in layout.items():
            view = manager._get_state_buffer(
                manager.layer_offsets[layer_id], DataRole(name), dtype, shape
            )
            if name in _KDA_REPLAY_CONV_ROLES:
                view = view.transpose(-1, -2)
            elif name == "kda_beta_cache":
                view = view[..., : manager.ssm_state_shape[0]]
            if view.shape[0] != manager._state_layout.slot_capacity:
                raise RuntimeError(f"KDA replay buffer {name} has a different slot capacity")
            layer[name] = view
        state.pool_layers.append(layer)
    state.prev_num_accepted_tokens = manager._get_state_buffer(
        manager.layer_offsets[manager.mamba_pp_layers[0]],
        _KDA_REPLAY_COUNT_ROLE,
        torch.int32,
        [_KDA_REPLAY_COUNT_BYTES // torch.int32.itemsize],
    )[..., 0]


def select_kda_replay_state(
    *,
    spec_config: object | None,
    manager_cls: type,
) -> KDAReplayState | None:
    """Apply KDA fused-verify and manager gates at the feature owner."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
        MambaHybridCacheManagerV2,
        MixedMambaHybridCacheManager,
    )

    manager_supports_replay = issubclass(
        manager_cls,
        (MambaHybridCacheManagerV2, MixedMambaHybridCacheManager),
    )
    num_spec = get_kda_replay_num_spec(
        spec_config,
        manager_supports_replay=manager_supports_replay,
    )
    if num_spec is None:
        return None
    return KDAReplayState(num_spec)


def get_kda_replay_num_spec(
    spec_config: object | None, *, manager_supports_replay: bool
) -> int | None:
    """Return the fused-verify replay width when its complete gate passes."""
    if spec_config is None or not manager_supports_replay:
        return None
    from ._kda_kernels import kda_mtp_replay_num_spec

    return kda_mtp_replay_num_spec(spec_config.tokens_per_gen_step - 1)


# Opt-in: prefill a context request whose last recurrent-state snapshot point
# lies inside its final chunk in one step. The KDA layers write the snapshot
# (state and convolution window at the point) into a pending V2 SSM page that
# the context update commits by move. Needs the pending SSM snapshot pages of
# the C++ V2 cache backend, the V2 scheduler and a KDA prefill that stores
# state snapshots (the K3 Rubin package's CuTe DSL prefill); otherwise the
# point keeps splitting the chunk.
_KDA_MIDCHUNK_SNAPSHOT = os.environ.get("TRTLLM_KDA_MIDCHUNK_SNAPSHOT", "0") == "1"
# With mid-chunk snapshots, the other snapshot points of a context request
# (its branch point where it leaves the cached prefix, points requested by
# requests waiting for its prefix, additional offsets) are snapshotted inside
# a context step as well instead of splitting the step there. A step holds
# one such snapshot, so it then ends at the following snapshot point at the
# latest. 0 splits at those points.
_KDA_MIDCHUNK_BRANCH_SNAPSHOT = os.environ.get("TRTLLM_KDA_MIDCHUNK_BRANCH_SNAPSHOT", "1") == "1"
# KDA layers that store mid-chunk snapshots. Their metadata fetches each
# step's plan from the manager (get_kda_midchunk_snapshots).
_MIDCHUNK_SNAPSHOT_KDA_PACKAGE = _K3_KDA_PACKAGE


class _KDAMidchunkSnapshot(NamedTuple):
    """A recurrent-state snapshot taken inside a request's context step."""

    # Prompt position of the snapshot; tokens [0, point) are committed with it.
    point: int
    # Tokens of the step before the snapshot point.
    offset: int
    # Pending SSM page that receives the snapshot. None on a rank without
    # local KDA layers, or when no page was available (snapshot skipped).
    slot: int | None


class KimiK3HybridCacheManager(MambaHybridCacheManagerV2):
    """KDA/MLA cache layout and K3-only replay lifecycle handling.

    With ``TRTLLM_KDA_MIDCHUNK_SNAPSHOT=1`` a context request whose last
    snapshot point lies inside its last context chunk is prefilled in one step
    (see ``_KDA_MIDCHUNK_SNAPSHOT``), and so are its other snapshot points (see
    ``_KDA_MIDCHUNK_BRANCH_SNAPSHOT``).
    """

    kda_context_only = False
    # Step alignment of mid-chunk snapshot points; None when they are off.
    _kda_midchunk_snapshot_alignment: int | None = None

    @override
    def __init__(
        self,
        *args,
        kda_replay_num_spec: int | None = None,
        kda_draft_records: bool = False,
        use_replay_state_update: bool = False,
        **kwargs,
    ) -> None:
        if use_replay_state_update:
            raise ValueError("KDA replay and Mamba2 replay are mutually exclusive")
        self._mamba_ssm_stochastic_rounding = kwargs.pop("mamba_ssm_stochastic_rounding", False)
        self.kda_context_only = _is_disagg_context_server(kwargs.get("is_disagg", False))
        self._requested_num_spec = kda_replay_num_spec
        self._requested_draft_records = kda_draft_records
        self._kda_replay: KDAReplayState | None = None
        self._kda_midchunk_snapshots: dict[int, _KDAMidchunkSnapshot] = {}
        # Requests whose snapshot plan the KDA metadata fetched this step.
        self._kda_midchunk_handed_out: set[int] = set()
        self._kda_midchunk_snapshots_taken_total = 0
        self._kda_midchunk_snapshots_skipped_total = 0
        kwargs.setdefault("conv_state_layout", "q_k_v")
        super().__init__(*args, **kwargs)
        self._kda_midchunk_snapshot_alignment = self._get_kda_midchunk_snapshot_alignment()

    @override
    def _initialize_spec_state(self) -> KDAIntermediateState | KDAReplayState | None:
        num_spec = self._requested_num_spec
        if num_spec is None:
            num_spec = get_kda_replay_num_spec(self.spec_config, manager_supports_replay=True)
        self.kda_context_only = self.kda_context_only and num_spec is not None
        if self.kda_context_only:
            return None
        if num_spec is not None:
            self._kda_replay = KDAReplayState(
                num_spec,
                stochastic_rounding=self._mamba_ssm_stochastic_rounding,
                draft_records=self._requested_draft_records,
            )
            self._kda_replay.validate(self._state_layout)
            return self._kda_replay
        return KDAIntermediateState()

    @property
    def use_kda_replay_update(self) -> bool:
        return self._kda_replay is not None

    @property
    def keeps_kda_draft_records(self) -> bool:
        return self.kda_state_tok is not None

    @property
    def kda_state_tok(self) -> list[torch.Tensor] | None:
        replay = self._kda_replay
        if replay is None or not replay.pool_layers:
            return None
        if _KDA_DRAFT_RECORDS_ROLE not in replay.pool_layers[0]:
            return None
        return [layer[_KDA_DRAFT_RECORDS_ROLE] for layer in replay.pool_layers]

    def _kda_replay_layout(self) -> dict[str, tuple[torch.dtype, list[int]]]:
        if self._kda_replay is None:
            return {}
        return self._kda_replay.pool_layout(self._state_layout)

    @override
    def _get_recurrent_buffer_configs(self, global_layer_id: int) -> list[BufferConfig]:
        buffers = super()._get_recurrent_buffer_configs(global_layer_id)
        if self._kda_replay is not None:
            buffers.extend(
                self._kda_replay.pool_buffer_configs(self._state_layout, global_layer_id)
            )
        return buffers

    @override
    def get_disagg_ignored_roles(self) -> frozenset[DataRole]:
        roles = super().get_disagg_ignored_roles()
        if self._kda_replay is not None:
            roles |= self._kda_replay.disagg_ignored_roles(self._state_layout)
        return roles

    @override
    def _setup_model_state(self) -> None:
        if self._kda_replay is not None:
            bind_kda_replay_pool(self._kda_replay, self)
        super()._setup_model_state()

    @override
    def _build_cache_config(self, config):
        if (
            self._kda_replay is not None
            and self.local_num_mamba_layers > 0
            and self.kv_cache_config.enable_kv_pool_rebalance
        ):
            raise ValueError(
                "KDA replay views are bound once to the SSM pool's slot count, "
                "and a rebalance can grow that pool past them: set "
                "kv_cache_config.enable_kv_pool_rebalance=False."
            )
        return super()._build_cache_config(config)

    def _get_kda_midchunk_snapshot_alignment(self) -> int | None:
        """Step alignment of mid-chunk snapshot points, or None when disabled.

        Depends only on configuration shared by all ranks, so that every rank
        merges the same context chunks.
        """
        if not _KDA_MIDCHUNK_SNAPSHOT or not self.enable_block_reuse or self.is_draft:
            return None
        if not hasattr(_KVCache, "acquire_ssm_snapshot_slot"):
            logger.warning(
                "TRTLLM_KDA_MIDCHUNK_SNAPSHOT=1 is ignored: the KV cache manager V2 "
                "C++ backend has no pending SSM snapshot pages (rebuild TensorRT-LLM "
                "from this source tree)."
            )
            return None
        try:
            kernels = importlib.import_module(f"{_MIDCHUNK_SNAPSHOT_KDA_PACKAGE}._kda_kernels")
            metadata = importlib.import_module(
                f"{_MIDCHUNK_SNAPSHOT_KDA_PACKAGE}.kimi_k3_mamba_metadata"
            )
        except ImportError as error:
            logger.warning(
                "TRTLLM_KDA_MIDCHUNK_SNAPSHOT=1 is ignored: the KDA prefill that "
                f"stores state snapshots is unavailable ({error})."
            )
            return None
        if not kernels.kda_prefill_supports_state_snapshots():
            logger.warning(
                "TRTLLM_KDA_MIDCHUNK_SNAPSHOT=1 is ignored: the KDA prefill kernel "
                "that stores state snapshots is unavailable."
            )
            return None
        logger.info("KDA mid-chunk recurrent-state snapshots are enabled")
        return metadata.KDA_PREFILL_CHUNK_SIZE

    @property
    @override
    def _snapshot_counters_at_info(self) -> bool:
        return (
            self._kda_midchunk_snapshot_alignment is not None or super()._snapshot_counters_at_info
        )

    @override
    def _format_branch_snapshot_counters(self) -> str:
        counters = super()._format_branch_snapshot_counters()
        if self._kda_midchunk_snapshot_alignment is None:
            return counters
        return (
            f"{counters} "
            f"midchunk_snapshots_taken={self._kda_midchunk_snapshots_taken_total} "
            f"midchunk_snapshots_skipped={self._kda_midchunk_snapshots_skipped_total}"
        )

    @override
    def prepare_expect_snapshot_points(self, requests) -> None:
        super().prepare_expect_snapshot_points(requests)
        # Read by the scheduler (_get_forced_context_chunk_size): the last
        # snapshot point no longer splits the context chunk, and a step may
        # pass the next other point.
        alignment = self._kda_midchunk_snapshot_alignment
        for request in requests:
            request.py_midchunk_snapshot_alignment = (
                alignment if not request.is_dummy_request else None
            )
            request.py_midchunk_pass_point = self._midchunk_pass_point(
                request, request.context_current_position
            )

    def _midchunk_pass_point(self, request, current_position: int) -> int | None:
        """The next snapshot point a context step may pass, if any.

        Every point ahead of the cursor but the last one qualifies; the last
        point merges by ``_is_midchunk_snapshot_point`` instead.
        """
        if (
            self._kda_midchunk_snapshot_alignment is None
            or not _KDA_MIDCHUNK_BRANCH_SNAPSHOT
            or request.is_dummy_request
        ):
            return None
        points = sorted(
            point for point in request.expect_snapshot_points if point > current_position
        )
        return points[0] if len(points) > 1 else None

    @override
    def _apply_branch_snapshot_point(self, request, context_current_position=None) -> None:
        super()._apply_branch_snapshot_point(request, context_current_position)
        current_position = (
            request.context_current_position
            if context_current_position is None
            else context_current_position
        )
        request.py_midchunk_pass_point = self._midchunk_pass_point(request, current_position)

    def _plan_kda_midchunk_snapshot(self, request) -> tuple[int, int] | None:
        """``(point, offset)`` of a snapshot inside this context step, if any."""
        alignment = getattr(request, "py_midchunk_snapshot_alignment", None)
        if not isinstance(alignment, int) or self._kda_midchunk_snapshot_alignment is None:
            return None
        start = request.context_current_position
        end = start + request.context_chunk_size
        # The scheduler lets at most one point into a step, but a point
        # requested after the step was scheduled can add another; the
        # deepest one is taken then.
        points = [
            point
            for point in request.expect_snapshot_points
            if start < point < end and (point - start) % alignment == 0
        ]
        if not points:
            return None
        return points[-1], points[-1] - start

    def _prepare_kda_midchunk_snapshots(self, context_requests) -> None:
        self._kda_midchunk_handed_out.clear()
        for request in context_requests:
            request_id = request.py_request_id
            stale = self._kda_midchunk_snapshots.pop(request_id, None)
            kv_cache = self.kv_cache_map.get(request_id)
            if stale is not None and kv_cache is not None:
                kv_cache.release_ssm_snapshot_slot()
            plan = self._plan_kda_midchunk_snapshot(request)
            if plan is None or kv_cache is None:
                continue
            point, offset = plan
            slot = None
            if self.local_num_mamba_layers > 0:
                slot = kv_cache.acquire_ssm_snapshot_slot()
                if slot is None:
                    self._kda_midchunk_snapshots_skipped_total += 1
                    logger.debug(
                        "KDA mid-chunk snapshot skipped for request "
                        f"{request_id} at {point}: no SSM page available"
                    )
            self._kda_midchunk_snapshots[request_id] = _KDAMidchunkSnapshot(point, offset, slot)

    def get_kda_midchunk_snapshots(self, request_ids: list[int]) -> list[tuple[int, int, int]]:
        """``(batch index, slot, offset)`` of this step's snapshots.

        ``request_ids`` are the step's context requests in batch order; the
        offset counts the request's tokens of this step before the snapshot.
        Called by the KDA metadata of a model whose prefill stores the
        snapshots; only snapshots handed out here are committed.
        """
        if not self._kda_midchunk_snapshots:
            return []
        snapshots = []
        for index, request_id in enumerate(request_ids):
            snapshot = self._kda_midchunk_snapshots.get(request_id)
            if snapshot is not None and snapshot.slot is not None:
                snapshots.append((index, snapshot.slot, snapshot.offset))
                self._kda_midchunk_handed_out.add(request_id)
        return snapshots

    def _disable_kda_midchunk_snapshots(self, request_id: int) -> None:
        """Stop planning snapshots that the model does not take.

        The KDA layers of the running model never fetched a planned snapshot,
        so its pending page holds no state. Every rank runs the same model and
        reaches this on the same step.
        """
        logger.warning(
            "TRTLLM_KDA_MIDCHUNK_SNAPSHOT=1 is turned off: the KDA layers did not "
            f"take the mid-chunk snapshot of request {request_id} (the model's KDA "
            "prefill does not store state snapshots)."
        )
        self._kda_midchunk_snapshot_alignment = None

    def _commit_kda_midchunk_snapshot(self, request, snapshot: _KDAMidchunkSnapshot) -> None:
        """Commit the prompt up to the snapshot point with the pending SSM page.

        Runs before the context update of the step, in the order of a step that
        ended at the point: history to the point, then the commit.
        """
        needs_page = self.local_num_mamba_layers > 0
        if needs_page and snapshot.slot is None:
            # Already counted as skipped when no page was available.
            return
        request_id = request.py_request_id
        kv_cache = self.kv_cache_map.get(request_id)
        if kv_cache is None or not kv_cache.is_active:
            # Suspension released the pending page.
            self._kda_midchunk_snapshots_skipped_total += int(needs_page)
            return
        has_page = not needs_page or kv_cache.pending_ssm_snapshot_slot == snapshot.slot
        # Without a fetch by the KDA metadata the pending page holds no state.
        filled = not needs_page or request_id in self._kda_midchunk_handed_out
        self._kda_midchunk_handed_out.discard(request_id)
        # A step that did not run leaves the context position before the point.
        step_ran = request.context_current_position >= snapshot.point
        if (
            not has_page
            or not filled
            or not step_ran
            or kv_cache.num_committed_tokens >= snapshot.point
        ):
            kv_cache.release_ssm_snapshot_slot()
            self._kda_midchunk_snapshots_skipped_total += int(needs_page)
            if has_page and step_ran and not filled:
                self._disable_kda_midchunk_snapshots(request_id)
            return
        if not kv_cache.resize(None, snapshot.point):
            raise ValueError(
                "Failed to resize history length of V2 Mamba cache for "
                f"request {request_id} to {snapshot.point} tokens "
                "at a mid-chunk snapshot"
            )
        tokens = self._augment_tokens_for_block_reuse(
            request.get_tokens(DEFAULT_BEAM_INDEX),
            request,
            start=kv_cache.num_committed_tokens,
            end=snapshot.point,
        )
        kv_cache.commit(tokens, ssm_from_pending=needs_page)
        if needs_page:
            self._kda_midchunk_snapshots_taken_total += 1
            self._count_snapshot_commit(request, snapshot.point)

    @override
    def update_context_resources(self, scheduled_batch) -> None:
        if self._kda_midchunk_snapshots:
            for request in scheduled_batch.context_requests:
                snapshot = self._kda_midchunk_snapshots.pop(request.py_request_id, None)
                if snapshot is not None:
                    self._commit_kda_midchunk_snapshot(request, snapshot)
        super().update_context_resources(scheduled_batch)

    @override
    def suspend_request(self, request) -> None:
        replay = self._kda_replay
        request_id = request.py_request_id
        cache = self.kv_cache_map.get(request_id)
        slot = self._request_id_to_state_index.get(request_id, -1)
        if replay is not None and slot >= 0 and cache is not None and cache.is_active:
            if request_id in replay.acceptance_pending:
                raise RuntimeError(
                    f"Request {request_id} is being suspended before update_resources "
                    "recorded the acceptance of its last verify, so its pending KDA drafts are unknown"
                )
            if request_id in replay.unseeded_requests:
                replay.unseeded_requests.remove(request_id)
            else:
                with replay.on_cache_stream(self._stream):
                    replay.commit_pending_drafts(slot)
            del self._request_id_to_state_index[request_id]
            replay.suspended_requests.add(request_id)
        super().suspend_request(request)

    @override
    def _setup_state_indices(self, requests, num_contexts: int = 0) -> None:
        super()._setup_state_indices(requests, num_contexts)
        replay = self._kda_replay
        if replay is None or self.local_num_mamba_layers == 0:
            return
        replay.suspended_requests.difference_update(
            request.py_request_id for request in requests[:num_contexts]
        )
        slots = []
        for request in requests[num_contexts:]:
            if request.py_request_id in replay.suspended_requests:
                replay.suspended_requests.remove(request.py_request_id)
                slots.append(self._request_id_to_state_index[request.py_request_id])
        if slots:
            with replay.on_cache_stream(self._stream):
                replay.seed_transferred_slots(
                    _host_slots_to_device(slots, replay.prev_num_accepted_tokens.device)
                )

    @override
    def prepare_resources(self, scheduled_batch) -> None:
        super().prepare_resources(scheduled_batch)
        if self._kda_midchunk_snapshot_alignment is not None:
            self._prepare_kda_midchunk_snapshots(scheduled_batch.context_requests)
        replay = self._kda_replay
        if replay is None or self.local_num_mamba_layers == 0:
            return
        replay.unseeded_requests.update(
            request.py_request_id
            for request in scheduled_batch.context_requests
            if request.is_disagg_generation_init_state
        )
        if getattr(self.spec_config, "decoding_type", None) == "NGram":
            replay.acceptance_pending.update(
                request.py_request_id
                for request in scheduled_batch.generation_requests
                if not request.is_dummy
            )

    @override
    def revert_allocate_generation(self, request) -> None:
        if self._kda_replay is not None:
            self._kda_replay.acceptance_pending.discard(request.py_request_id)
        super().revert_allocate_generation(request)

    @override
    def free_resources(self, request, pin_on_release: bool = False) -> None:
        self._kda_midchunk_handed_out.discard(request.py_request_id)
        if self._kda_midchunk_snapshots.pop(request.py_request_id, None) is not None:
            kv_cache = self.kv_cache_map.get(request.py_request_id)
            if kv_cache is not None and kv_cache.is_active:
                kv_cache.release_ssm_snapshot_slot()
        if self._kda_replay is not None:
            self._kda_replay.suspended_requests.discard(request.py_request_id)
            self._kda_replay.acceptance_pending.discard(request.py_request_id)
            self._kda_replay.unseeded_requests.discard(request.py_request_id)
        super().free_resources(request, pin_on_release)

    @override
    def _extra_scratch_bytes_per_slot(self) -> int:
        if self._kda_replay is None or not self._kda_replay._stochastic_rounding:
            return 0
        return torch.int64.itemsize

    @classmethod
    @override
    def _extra_state_bytes_per_rank(cls, model_config, mapping: Mapping, **kwargs) -> int:
        return KDAReplayState.estimate_bytes_per_slot(model_config, mapping, **kwargs)

    @override
    def _on_state_slots_relocated(self, old_slots: list[int], new_slots: list[int]) -> None:
        if self._kda_replay is None:
            return
        self._kda_replay.relocate_slots(old_slots, new_slots)
        fresh_slots = [new for old, new in zip(old_slots, new_slots) if old < 0]
        if fresh_slots:
            slots = _host_slots_to_device(fresh_slots, self.cuda_state_indices.device)
            self._kda_replay.reset_slots(slots, fresh_slots)

    @override
    def update_mamba_states(
        self, attn_metadata, num_accepted_tokens, state_indices=None, accepted_leaf_positions=None
    ) -> None:
        if self.kda_context_only:
            return
        super().update_mamba_states(
            attn_metadata, num_accepted_tokens, state_indices, accepted_leaf_positions
        )

    @override
    def update_resources(
        self, scheduled_batch, attn_metadata=None, kv_cache_dtype_byte_size=None
    ) -> None:
        super().update_resources(scheduled_batch, attn_metadata, kv_cache_dtype_byte_size)
        if (
            self.local_num_mamba_layers
            and self._kda_replay is not None
            and getattr(self.spec_config, "decoding_type", None) == "NGram"
        ):
            self._record_replay_request_acceptance(scheduled_batch)

    def _record_replay_request_acceptance(self, scheduled_batch: object) -> None:
        replay = self._kda_replay
        if replay.prev_num_accepted_tokens is None:
            return
        generation_requests = scheduled_batch.generation_requests
        replay.acceptance_pending.difference_update(
            request.py_request_id for request in generation_requests
        )
        drafted_requests = [
            request
            for request in generation_requests
            if request.py_draft_tokens is not None and len(request.py_draft_tokens) > 0
        ]
        if not drafted_requests:
            return
        if len(drafted_requests) != len(generation_requests):
            raise RuntimeError(
                "Mixed drafted/undrafted generation batch is not supported "
                "for KDA replay bookkeeping"
            )
        state_index_map = self._request_id_to_state_index
        dummy_map = self._request_id_to_is_dummy
        active_requests = [
            request for request in generation_requests if request.py_request_id in state_index_map
        ]
        if not active_requests:
            return
        device = replay.prev_num_accepted_tokens.device
        state_indices = torch.tensor(
            [state_index_map[request.py_request_id] for request in active_requests],
            dtype=torch.int32,
            device=device,
        )
        accepted_drafts = torch.tensor(
            [request.py_num_accepted_draft_tokens for request in active_requests],
            dtype=torch.int32,
            device=device,
        )
        is_dummy_request = torch.tensor(
            [dummy_map.get(request.py_request_id, False) for request in active_requests],
            dtype=torch.bool,
            device=device,
        )
        replay.record_acceptance(state_indices, accepted_drafts, is_dummy_request)

    @override
    def on_state_transfer_complete(self, request_ids: list[int]) -> None:
        replay = self._kda_replay
        if replay is None or replay.prev_num_accepted_tokens is None:
            return
        replay.unseeded_requests.difference_update(request_ids)
        slots = sorted(
            {
                self._request_id_to_state_index[request_id]
                for request_id in request_ids
                if request_id in self._request_id_to_state_index
            }
        )
        if slots:
            replay.seed_transferred_slots(
                _host_slots_to_device(slots, replay.prev_num_accepted_tokens.device)
            )

    def seed_kda_replay_caches_for_disagg_gen(self, request_ids: list[int]) -> None:
        """Compatibility entry point; the executor uses the generic transfer hook."""
        self.on_state_transfer_complete(request_ids)

    @override
    def shutdown(self) -> None:
        self._kda_midchunk_snapshots.clear()
        self._kda_midchunk_handed_out.clear()
        super().shutdown()


def resolve_kimi_ssm_cache_dtype(config) -> torch.dtype:
    """Default KDA state to FP32; BF16 requires an explicit cache configuration.

    BF16 prefill, decode and sequential verify stage state through FP32 and
    round the committed state back to BF16. Fused MTP verify reads and writes
    BF16 state directly, rounding the committed state to nearest.
    Checkpoint-declared state dtypes do not override this default.
    """
    from tensorrt_llm._torch.pyexecutor.config_utils import resolve_ssm_cache_dtype

    declared = resolve_ssm_cache_dtype(config)
    if declared is not None and declared != torch.float32:
        logger.info(
            "Kimi K3: the checkpoint declares "
            f"mamba_ssm_cache_dtype={declared}; keeping the fp32 "
            "recurrent-state pool (kv_cache_config.mamba_ssm_cache_dtype "
            "opts in to bfloat16)."
        )
    return torch.float32


def validate_kimi_state_cache_dtype(
    mamba_ssm_cache_dtype: torch.dtype, mamba_ssm_stochastic_rounding: bool = False
) -> None:
    """Reject cache dtypes and rounding modes unsupported by the KDA kernels."""
    if mamba_ssm_cache_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            "Kimi K3 KDA recurrent-state cache supports float32 (default) or "
            f"bfloat16; got mamba_ssm_cache_dtype={mamba_ssm_cache_dtype}."
        )
    if mamba_ssm_stochastic_rounding and mamba_ssm_cache_dtype != torch.float32:
        raise ValueError(
            "Kimi K3 KDA kernels round the committed recurrent state to "
            "nearest; mamba_ssm_stochastic_rounding is not supported with a "
            f"{mamba_ssm_cache_dtype} state cache."
        )


def get_kimi_cache_params(config, *, spec_config=None, quant_config=None):
    """KDA uses equal Q/K/V sections and defaults to FP32 state alongside MLA."""
    from tensorrt_llm._torch.pyexecutor.config_utils import (
        build_mamba_kv_cache_params,
        get_kimi_linear_layer_masks,
        unwrap_kimi_text_config,
    )

    linear = unwrap_kimi_text_config(config).linear_attn_config
    attention, recurrent = get_kimi_linear_layer_masks(config)
    params = build_mamba_kv_cache_params(
        config,
        state_size=linear["head_dim"],
        conv_kernel=linear["short_conv_kernel_size"],
        num_heads=linear["num_heads"],
        n_groups=linear["num_heads"],
        head_dim=linear["head_dim"],
        mamba_mask=recurrent,
        target_full_attn_mask=attention,
        spec_config=spec_config,
        quant_config=quant_config,
    )
    return params


__all__ = [
    "KDAIntermediateLayerCache",
    "KDAIntermediateState",
    "KDAReplayState",
    "KDAReplayLayerCache",
    "KimiK3HybridCacheManager",
    "allocate_kda_replay_fields",
    "bind_kda_replay_pool",
    "get_kda_replay_num_spec",
    "get_kimi_cache_params",
    "resolve_kimi_ssm_cache_dtype",
    "select_kda_replay_state",
    "validate_kimi_state_cache_dtype",
]
