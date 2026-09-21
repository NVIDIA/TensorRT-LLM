# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""K3-owned fused-verify cache payload and replay state."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import torch

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tensorrt_llm._torch.modules.mamba.cache_manager import (
    allocate_mamba_seed_buffer,
    compute_deterministic_mamba_seed,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
    IntermediateState,
    MambaAcceptanceBatch,
    MambaLayerCache,
    MambaStateLayout,
    ReplayStateUpdateMetadata,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
)
from tensorrt_llm.logger import logger


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
    kda_qkg_cache: torch.Tensor | None = None
    kda_v_cache: torch.Tensor | None = None
    kda_beta_cache: torch.Tensor | None = None

    @property
    def has_kda_replay_caches(self) -> bool:
        return self.kda_qkg_cache is not None

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


def _kda_beta_cache_padded_heads(num_heads: int) -> int:
    heads_per_alignment = _KDA_BETA_CACHE_ALIGNMENT_BYTES // torch.float32.itemsize
    return (num_heads + heads_per_alignment - 1) // heads_per_alignment * heads_per_alignment


def _allocate_kda_beta_cache(shape: tuple[int, ...], device: torch.device | None) -> torch.Tensor:
    """Keep logical head count while aligning each physical row for CuTe."""
    return torch.zeros(
        *shape[:-1],
        _kda_beta_cache_padded_heads(shape[-1]),
        dtype=torch.float32,
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
            dtype=torch.float32,
            device=device,
        ).transpose(-1, -2)

    fields = {
        "prev_num_accepted_tokens": torch.zeros(cache_size, dtype=torch.int32, device=device),
        "kda_conv_q": allocate_conv_cache(),
        "kda_conv_k": allocate_conv_cache(),
        "kda_conv_v": allocate_conv_cache(),
        "kda_qkg_cache": torch.zeros(
            num_local_layers,
            cache_size,
            num_speculative_tokens,
            3,
            section_dim,
            dtype=torch.float32,
            device=device,
        ),
        "kda_v_cache": torch.zeros(
            num_local_layers,
            cache_size,
            num_speculative_tokens,
            section_dim,
            dtype=torch.float32,
            device=device,
        ),
        "kda_beta_cache": _allocate_kda_beta_cache(
            (num_local_layers, cache_size, num_speculative_tokens, num_heads), device
        ),
    }
    scratch = [
        fields["kda_conv_q"],
        fields["kda_conv_k"],
        fields["kda_conv_v"],
        fields["kda_qkg_cache"],
        fields["kda_v_cache"],
        fields["kda_beta_cache"],
    ]
    return fields, scratch


class KDAReplayState:
    """Own all V2 state used by the KDA fused multi-token verifier."""

    @property
    def intermediate_ssm(self) -> torch.Tensor | None:
        return None

    @property
    def intermediate_conv(self) -> torch.Tensor | None:
        return None

    def get_replay_metadata(self) -> ReplayStateUpdateMetadata | None:
        return None

    def __init__(
        self,
        num_speculative_tokens: int,
    ) -> None:
        self.num_speculative_tokens = num_speculative_tokens
        self._conv_states: tuple[torch.Tensor, ...] = ()
        self._committed_window = 0
        self.prev_num_accepted_tokens: torch.Tensor | None = None
        self.kda_conv_q: torch.Tensor | None = None
        self.kda_conv_k: torch.Tensor | None = None
        self.kda_conv_v: torch.Tensor | None = None
        self.kda_qkg_cache: torch.Tensor | None = None
        self.kda_v_cache: torch.Tensor | None = None
        self.kda_beta_cache: torch.Tensor | None = None
        self.intermediate_indices: torch.Tensor | None = None
        self.rand_seed: torch.Tensor | None = None
        self._seed_request_counter = 0
        self._seed_rank_offset = 0
        self._conv_section_dims: tuple[int, ...] = ()

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

    def bytes_per_slot(self, context: MambaStateLayout, layer_id: int) -> int:
        if not context.mamba_pp_layers:
            return 0
        section_dim = context.conv_section_dims[0]
        committed_window = context.conv_state_shape[1]
        extended_window = committed_window + self.num_speculative_tokens
        num_heads = context.ssm_state_shape[0]
        float_elements = (
            3 * section_dim * extended_window
            + self.num_speculative_tokens * 3 * section_dim
            + self.num_speculative_tokens * section_dim
            + self.num_speculative_tokens * _kda_beta_cache_padded_heads(num_heads)
        )
        shared_bytes = 0
        if layer_id == context.mamba_pp_layers[0]:
            shared_bytes = 4
            if context.stochastic_rounding:
                shared_bytes += 8
        return float_elements * 4 + shared_bytes

    def bind(
        self,
        context: MambaStateLayout,
        ssm_states: list[torch.Tensor],
        conv_states: list[torch.Tensor],
    ) -> None:
        """Borrow per-layer conv views [slot, conv_dim, window] and allocate scratch."""
        self._seed_rank_offset = context.seed_rank_offset
        self._conv_section_dims = context.conv_section_dims
        self._conv_states = tuple(conv_states)
        self._committed_window = context.conv_state_shape[1] if conv_states else 0
        states = ssm_states
        if not states:
            return

        cache_size = context.slot_capacity
        if cache_size is None:
            raise RuntimeError("KDA replay requires the allocated V2 slot capacity")
        device = states[0].device
        fields, _ = allocate_kda_replay_fields(
            num_local_layers=len(context.mamba_pp_layers),
            cache_size=cache_size,
            num_speculative_tokens=self.num_speculative_tokens,
            conv_dim=context.conv_state_shape[0],
            conv_kernel_size=context.conv_state_shape[1] + 1,
            num_heads=context.ssm_state_shape[0],
            device=device,
        )
        for name, value in fields.items():
            setattr(self, name, value)
        self.intermediate_indices = torch.arange(
            context.max_batch_size,
            dtype=torch.int32,
            device=device,
        )
        if context.stochastic_rounding:
            self.rand_seed = allocate_mamba_seed_buffer(
                cache_size,
                context.seed_rank_offset,
                device,
            )
        logger.info(f"Mamba Cache (kda-replay) is allocated for {cache_size} state slots")

    def _layer_cache_fields(self, layer_offset: int) -> dict[str, torch.Tensor | None]:
        return {
            "prev_num_accepted_tokens": self.prev_num_accepted_tokens,
            "kda_conv_q": None if self.kda_conv_q is None else self.kda_conv_q[layer_offset],
            "kda_conv_k": None if self.kda_conv_k is None else self.kda_conv_k[layer_offset],
            "kda_conv_v": None if self.kda_conv_v is None else self.kda_conv_v[layer_offset],
            "kda_qkg_cache": (
                None if self.kda_qkg_cache is None else self.kda_qkg_cache[layer_offset]
            ),
            "kda_v_cache": None if self.kda_v_cache is None else self.kda_v_cache[layer_offset],
            "kda_beta_cache": (
                None if self.kda_beta_cache is None else self.kda_beta_cache[layer_offset]
            ),
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
        self.prev_num_accepted_tokens[slots] = 0
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
            self.rand_seed[slots] = torch.tensor(
                seeds,
                dtype=torch.int64,
                device=self.rand_seed.device,
            )

    def relocate_slots(self, old_slots: list[int], new_slots: list[int]) -> None:
        if self.prev_num_accepted_tokens is None:
            return
        moves = [(old, new) for old, new in zip(old_slots, new_slots) if old >= 0 and old != new]
        if not moves:
            return
        device = self.prev_num_accepted_tokens.device
        source_slots = torch.tensor(
            [old for old, _ in moves],
            dtype=torch.long,
            device=device,
        )
        destination_slots = torch.tensor(
            [new for _, new in moves],
            dtype=torch.long,
            device=device,
        )
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

        replay_conv_buffers = (self.kda_conv_q, self.kda_conv_k, self.kda_conv_v)
        for layer_offset, conv_state in enumerate(self._conv_states):
            selected_conv = conv_state.index_select(0, state_indices)
            for section_idx, replay_buffer in enumerate(replay_conv_buffers):
                if replay_buffer is None:
                    raise RuntimeError("KDA replay convolution buffers are not bound")
                replay_layer = replay_buffer[layer_offset]
                replay_layer.index_fill_(0, state_indices, 0)
                section = selected_conv[
                    :,
                    section_offsets[section_idx] : section_offsets[section_idx + 1],
                    :,
                ]
                replay_layer[:, :, :committed_window].index_copy_(
                    0,
                    state_indices,
                    section.to(replay_layer.dtype),
                )
        for replay_buffer in (self.kda_qkg_cache, self.kda_v_cache, self.kda_beta_cache):
            if replay_buffer is None:
                raise RuntimeError("KDA replay history buffers are not bound")
            replay_buffer.index_fill_(1, state_indices, 0)
        self.prev_num_accepted_tokens[state_indices] = 0

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

    def _layer_replay_buffers(self) -> tuple[torch.Tensor, ...]:
        buffers = (
            self.kda_conv_q,
            self.kda_conv_k,
            self.kda_conv_v,
            self.kda_qkg_cache,
            self.kda_v_cache,
            self.kda_beta_cache,
        )
        if any(buffer is None for buffer in buffers):
            raise RuntimeError("KDA replay buffers are not bound")
        return tuple(buffer for buffer in buffers if buffer is not None)

    def shutdown(self) -> None:
        self._conv_states = ()
        self.prev_num_accepted_tokens = None
        self.kda_conv_q = None
        self.kda_conv_k = None
        self.kda_conv_v = None
        self.kda_qkg_cache = None
        self.kda_v_cache = None
        self.kda_beta_cache = None
        self.intermediate_indices = None
        self.rand_seed = None


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
    from ._kda_kernels import is_kda_mtp_verify_available

    if not is_kda_mtp_verify_available():
        return None
    return spec_config.tokens_per_gen_step - 1


class KimiK3HybridCacheManagerV2(MambaHybridCacheManagerV2):
    """KDA/MLA cache layout and K3-only replay lifecycle handling."""

    @override
    def __init__(
        self,
        *args,
        kda_replay_num_spec: int | None = None,
        use_replay_state_update: bool = False,
        **kwargs,
    ) -> None:
        if use_replay_state_update:
            raise ValueError("KDA replay and Mamba2 replay are mutually exclusive")
        self._requested_num_spec = kda_replay_num_spec
        self._kda_replay: KDAReplayState | None = None
        kwargs.setdefault("conv_state_layout", "q_k_v")
        super().__init__(*args, **kwargs)

    @override
    def _initialize_model_state(self) -> IntermediateState | KDAReplayState:
        num_spec = self._requested_num_spec
        if num_spec is None:
            num_spec = get_kda_replay_num_spec(self.spec_config, manager_supports_replay=True)
        if num_spec is not None:
            self._kda_replay = KDAReplayState(num_spec)
            self._kda_replay.validate(self._state_layout)
            return self._kda_replay
        return IntermediateState()

    @property
    def use_kda_replay_update(self) -> bool:
        return self._kda_replay is not None

    @override
    def _extra_scratch_bytes_per_slot(self) -> int:
        if self._kda_replay is None:
            return 0
        return sum(
            self._kda_replay.bytes_per_slot(self._state_layout, layer_id)
            for layer_id in self.mamba_pp_layers
        )

    @override
    def _on_state_slots_relocated(self, old_slots: list[int], new_slots: list[int]) -> None:
        if self._kda_replay is None:
            return
        self._kda_replay.relocate_slots(old_slots, new_slots)
        fresh_slots = [new for old, new in zip(old_slots, new_slots) if old < 0]
        if fresh_slots:
            slots = torch.tensor(
                fresh_slots, dtype=torch.long, device=self.cuda_state_indices.device
            )
            self._kda_replay.reset_slots(slots, fresh_slots)

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
        slots = sorted(
            {
                self._request_id_to_state_index[request_id]
                for request_id in request_ids
                if request_id in self._request_id_to_state_index
            }
        )
        if slots:
            replay.seed_transferred_slots(
                torch.tensor(slots, dtype=torch.long, device=replay.prev_num_accepted_tokens.device)
            )

    def seed_kda_replay_caches_for_disagg_gen(self, request_ids: list[int]) -> None:
        """Compatibility entry point; the executor uses the generic transfer hook."""
        self.on_state_transfer_complete(request_ids)


def resolve_kimi_ssm_cache_dtype(config) -> torch.dtype:
    """Default KDA state to FP32; BF16 requires an explicit cache configuration.

    BF16 prefill, decode and sequential verify stage state through FP32 and
    round the committed state back to BF16. Fused MTP verify requires FP32.
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
    "KDAReplayState",
    "KDAReplayLayerCache",
    "KimiK3HybridCacheManagerV2",
    "allocate_kda_replay_fields",
    "get_kda_replay_num_spec",
    "get_kimi_cache_params",
    "resolve_kimi_ssm_cache_dtype",
    "select_kda_replay_state",
    "validate_kimi_state_cache_dtype",
]
