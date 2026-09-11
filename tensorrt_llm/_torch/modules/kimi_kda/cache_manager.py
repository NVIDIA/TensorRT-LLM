# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KDA-owned fused-verify cache payload and lifecycle strategy."""

from dataclasses import dataclass, field
from typing import Iterable

import torch

from tensorrt_llm._torch.modules.mamba.cache_manager import (
    allocate_mamba_seed_buffer,
    compute_deterministic_mamba_seed,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
    MambaCacheBuildContext,
    MambaLayerCache,
    MambaStateUpdateBatch,
    MambaStateUpdateStrategy,
    SpeculativeMambaLayerCache,
)
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger


@dataclass(frozen=True, kw_only=True)
class KDAReplayLayerCache(SpeculativeMambaLayerCache):
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
        "kda_beta_cache": torch.zeros(
            num_local_layers,
            cache_size,
            num_speculative_tokens,
            num_heads,
            dtype=torch.float32,
            device=device,
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


@torch.inference_mode()
def seed_kda_replay_after_state_transfer(
    cache: KDAReplayLayerCache, slot_indices: torch.Tensor
) -> None:
    """Seed replay state after disaggregated base-state transfer completes."""
    if slot_indices.numel() == 0:
        return
    conv = cache.conv
    section_dim = conv.shape[2] // 3
    committed = conv.shape[3]
    selected = conv.index_select(1, slot_indices)
    for replay, section in (
        (cache.kda_conv_q, selected[:, :, :section_dim]),
        (cache.kda_conv_k, selected[:, :, section_dim : 2 * section_dim]),
        (cache.kda_conv_v, selected[:, :, 2 * section_dim :]),
    ):
        seeded = torch.zeros(
            (replay.shape[0], slot_indices.numel()) + replay.shape[2:],
            dtype=replay.dtype,
            device=replay.device,
        )
        seeded[:, :, :, :committed] = section.to(replay.dtype)
        replay.index_copy_(1, slot_indices, seeded)
    for replay in (cache.kda_qkg_cache, cache.kda_v_cache, cache.kda_beta_cache):
        replay.index_fill_(1, slot_indices, 0)
    cache.prev_num_accepted_tokens[slot_indices] = 0


class KDAReplayStateUpdateStrategy(MambaStateUpdateStrategy):
    """Own all V2 state used by the KDA fused multi-token verifier."""

    def __init__(
        self,
        num_speculative_tokens: int,
        *,
        record_host_acceptance: bool = False,
    ) -> None:
        self.num_speculative_tokens = num_speculative_tokens
        self._record_host_acceptance = record_host_acceptance
        self.prev_num_accepted_tokens: torch.Tensor | None = None
        self.kda_conv_q: torch.Tensor | None = None
        self.kda_conv_k: torch.Tensor | None = None
        self.kda_conv_v: torch.Tensor | None = None
        self.kda_qkg_cache: torch.Tensor | None = None
        self.kda_v_cache: torch.Tensor | None = None
        self.kda_beta_cache: torch.Tensor | None = None
        self.intermediate_indices: torch.Tensor | None = None
        self.rand_seed: torch.Tensor | None = None
        self._dummy_mask: torch.Tensor | None = None
        self._dummy_mask_host: torch.Tensor | None = None
        self._seed_request_counter = 0
        self._seed_rank_offset = 0
        self._conv_section_dims: tuple[int, ...] = ()

    @property
    def key(self) -> str:
        return "kimi_kda_replay"

    @property
    def uses_replay(self) -> bool:
        return True

    @property
    def state_indices_alignment(self) -> int:
        return 16

    def validate(self, context: MambaCacheBuildContext) -> None:
        if context.backend != "v2":
            raise ValueError("KDA replay strategy requires the V2 Mamba cache manager")
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

    def bytes_per_slot(self, context: MambaCacheBuildContext, layer_id: int) -> int:
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
            + self.num_speculative_tokens * num_heads
        )
        shared_bytes = 0
        if layer_id == context.mamba_pp_layers[0]:
            shared_bytes = 4
            if context.stochastic_rounding:
                shared_bytes += 8
        return float_elements * 4 + shared_bytes

    def bind(self, context: MambaCacheBuildContext, manager: object) -> None:
        self._seed_rank_offset = context.seed_rank_offset
        self._conv_section_dims = context.conv_section_dims
        states = getattr(manager, "all_ssm_states")
        if not states:
            return

        cache_size = context.slot_capacity
        if cache_size is None:
            raise RuntimeError("KDA replay strategy requires the allocated V2 slot capacity")
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
        self._dummy_mask = torch.zeros(
            context.state_index_capacity,
            dtype=torch.bool,
            device=device,
        )
        self._dummy_mask_host = torch.zeros(
            context.state_index_capacity,
            dtype=torch.bool,
            pin_memory=prefer_pinned(),
        )
        logger.info(f"Mamba Cache (kda-replay) is allocated for {cache_size} state slots")

    def layer_cache_fields(self, layer_offset: int) -> dict[str, torch.Tensor | None]:
        return {
            "intermediate_conv_window": None,
            "intermediate_ssm": None,
            "mamba_ssm_rand_seed": self.rand_seed,
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
            **self.layer_cache_fields(layer_offset),
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

    @torch.inference_mode()
    def refresh_dummy_request_mask(self, is_dummy: list[bool]) -> None:
        if self._dummy_mask is None or self._dummy_mask_host is None:
            return
        count = len(is_dummy)
        if count > self._dummy_mask_host.shape[0]:
            raise ValueError("Dummy-request batch exceeds the KDA replay mask capacity")
        self._dummy_mask_host.zero_()
        if count:
            self._dummy_mask_host[:count].copy_(torch.tensor(is_dummy, dtype=torch.bool))
        self._dummy_mask.copy_(self._dummy_mask_host, non_blocking=True)

    def dummy_request_mask(self, start: int, end: int) -> torch.Tensor | None:
        if self._dummy_mask is None:
            return None
        return self._dummy_mask[start:end]

    def source_state_indices(self, count: int) -> torch.Tensor:
        if self.intermediate_indices is None:
            raise RuntimeError("KDA replay strategy is not bound")
        return self.intermediate_indices[:count]

    def update(self, batch: MambaStateUpdateBatch, manager: object) -> bool:
        del manager
        self._record_acceptance(
            batch.destination_state_indices,
            batch.num_accepted_tokens - 1,
            batch.is_dummy_request,
        )
        return True

    def on_resources_updated(self, scheduled_batch: object, manager: object) -> None:
        if not self._record_host_acceptance or self.prev_num_accepted_tokens is None:
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
        state_index_map = getattr(manager, "_request_id_to_state_index")
        dummy_map = getattr(manager, "_request_id_to_is_dummy")
        active_requests = [
            request for request in generation_requests if request.py_request_id in state_index_map
        ]
        if not active_requests:
            return
        device = self.prev_num_accepted_tokens.device
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
        self._record_acceptance(state_indices, accepted_drafts, is_dummy_request)

    @torch.inference_mode()
    def on_state_transfer_complete(self, request_ids: list[int], manager: object) -> None:
        if self.prev_num_accepted_tokens is None:
            return
        state_index_map = getattr(manager, "_request_id_to_state_index")
        slots = [
            state_index_map[request_id]
            for request_id in request_ids
            if request_id in state_index_map
        ]
        if not slots:
            return
        device = self.prev_num_accepted_tokens.device
        state_indices = torch.tensor(
            sorted(set(slots)),
            dtype=torch.long,
            device=device,
        )
        committed_window = getattr(manager, "conv_state_shape")[1]
        section_offsets = [0]
        for section_dim in self._conv_section_dims:
            section_offsets.append(section_offsets[-1] + section_dim)

        replay_conv_buffers = (self.kda_conv_q, self.kda_conv_k, self.kda_conv_v)
        for layer_offset, conv_state in enumerate(getattr(manager, "all_conv_states")):
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

    def _record_acceptance(
        self,
        state_indices: torch.Tensor,
        num_accepted_draft_tokens: torch.Tensor,
        is_dummy_request: torch.Tensor | None,
    ) -> None:
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

    def iter_buffers(self) -> Iterable[torch.Tensor]:
        for buffer in (
            self.prev_num_accepted_tokens,
            self.kda_conv_q,
            self.kda_conv_k,
            self.kda_conv_v,
            self.kda_qkg_cache,
            self.kda_v_cache,
            self.kda_beta_cache,
            self.intermediate_indices,
            self.rand_seed,
            self._dummy_mask,
        ):
            if buffer is not None:
                yield buffer

    def shutdown(self, manager: object | None = None) -> None:
        del manager
        self.prev_num_accepted_tokens = None
        self.kda_conv_q = None
        self.kda_conv_k = None
        self.kda_conv_v = None
        self.kda_qkg_cache = None
        self.kda_v_cache = None
        self.kda_beta_cache = None
        self.intermediate_indices = None
        self.rand_seed = None
        self._dummy_mask = None
        self._dummy_mask_host = None


def build_kda_state_update_strategy(
    *,
    spec_config: object | None,
    manager_cls: type,
) -> KDAReplayStateUpdateStrategy | None:
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
    return KDAReplayStateUpdateStrategy(
        num_spec,
        record_host_acceptance=getattr(spec_config, "decoding_type", None) == "NGram",
    )


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


__all__ = [
    "KDAReplayLayerCache",
    "KDAReplayStateUpdateStrategy",
    "allocate_kda_replay_fields",
    "build_kda_state_update_strategy",
    "get_kda_replay_num_spec",
    "seed_kda_replay_after_state_transfer",
]
