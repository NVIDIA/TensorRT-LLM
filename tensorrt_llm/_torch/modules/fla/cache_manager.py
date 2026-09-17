# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated-DeltaNet replay integration for Mamba cache managers."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import torch

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tensorrt_llm._torch.modules.mamba.cache_manager import Mamba2ReplayLayerCache, ReplayHistory
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
    IntermediateState,
    MambaAcceptanceBatch,
    MambaLayerCache,
    MambaStateLayout,
    _mamba_effective_tp_size,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
)
from tensorrt_llm._torch.utils import is_gdn_replay_enabled
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from .cached_replay import (
    CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE,
    commit_gdn_cached_replay_history_layers,
)


@dataclass(frozen=True, kw_only=True)
class GDNReplayLayerCache(Mamba2ReplayLayerCache):
    """GDN cached-replay tensors for one recurrent layer."""


class GDNReplayState(ReplayHistory):
    """Own GDN all-layer cached-replay checkpoint commits."""

    @override
    def __init__(self, tokens_per_gen_step: int) -> None:
        super().__init__(tokens_per_gen_step)
        self._state_descriptors: torch.Tensor | None = None
        self._state_strides: tuple[int, int, int] | None = None

    @override
    def bind(
        self,
        context: MambaStateLayout,
        ssm_states: list[torch.Tensor],
        conv_states: list[torch.Tensor],
    ) -> None:
        super().bind(context, ssm_states, conv_states)
        states = self._ssm_states
        if not states:
            return
        state_shape = list(context.ssm_state_shape)
        expected_inner_strides = (
            state_shape[1] * state_shape[2],
            state_shape[2],
            1,
        )
        reference = states[0]
        for state in states:
            if (
                state.dtype != reference.dtype
                or state.device != reference.device
                or list(state.shape[1:]) != state_shape
                or state.stride()[1:] != expected_inner_strides
            ):
                raise RuntimeError(
                    "GDN cached replay requires V2 SSM layers with matching "
                    "dtype, device, shape, and dense inner dimensions."
                )
        pointers = [state.data_ptr() for state in states]
        slot_stride = reference.stride(0)
        layer_stride_bytes = pointers[1] - pointers[0] if len(pointers) > 1 else 0
        has_affine_layout = (
            layer_stride_bytes % reference.element_size() == 0
            and all(state.stride(0) == slot_stride for state in states)
            and all(
                pointer == pointers[0] + layer * layer_stride_bytes
                for layer, pointer in enumerate(pointers)
            )
        )
        if has_affine_layout:
            self._state_strides = (
                len(pointers),
                layer_stride_bytes // reference.element_size(),
                slot_stride,
            )
        else:
            logger.warning_once(
                "V2 GDN state views are not affine; using indirect replay checkpoint addressing",
                key="gdn_cached_replay_v2_indirect_state_layout",
            )
            self._state_descriptors = torch.tensor(
                [(state.data_ptr(), state.stride(0)) for state in states],
                dtype=torch.int64,
                device=reference.device,
            )
        state_layout = "affine" if self._state_strides is not None else "indirect"
        logger.info_once(
            "Configured GDN cached replay commit mode for V2: small-batch fused, "
            f"large-batch all-layer; state layout: {state_layout}",
            key="gdn_cached_replay_v2_commit_mode_fused",
        )

    @property
    def has_bound_states(self) -> bool:
        return self._state_descriptors is not None or self._state_strides is not None

    @override
    def update(self, batch: MambaAcceptanceBatch) -> None:
        num_decodes = batch.num_generations
        if num_decodes >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE:
            self.commit_all_layers(batch.attention_metadata, num_decodes)
        super().update(batch)

    @override
    def make_layer_cache(
        self,
        layer_offset: int,
        conv: torch.Tensor,
        temporal: torch.Tensor,
    ) -> MambaLayerCache:
        return GDNReplayLayerCache(
            conv=conv,
            temporal=temporal,
            **self._layer_cache_fields(layer_offset),
        )

    def commit_all_layers(self, attention_metadata: object, num_decodes: int) -> None:
        """Commit every local GDN checkpoint using one partitioned launch."""
        states = self._ssm_states
        if (
            not states
            or (self._state_descriptors is None and self._state_strides is None)
            or self.old_x is None
            or self.old_B is None
            or self.old_dt is None
        ):
            raise RuntimeError(
                "GDN cached replay all-layer commit requires V2 replay state buffers."
            )

        mamba_metadata = attention_metadata.mamba_metadata
        if mamba_metadata.replay_num_decodes != num_decodes:
            raise RuntimeError(
                "GDN replay metadata contains "
                f"{mamba_metadata.replay_num_decodes} decode requests, "
                f"but state update received {num_decodes}."
            )
        state_strides = self._state_strides
        commit_gdn_cached_replay_history_layers(
            ssm_states=states[0],
            ssm_state_descriptors=self._state_descriptors,
            ssm_state_num_layers=(state_strides[0] if state_strides is not None else None),
            ssm_state_layer_stride=(state_strides[1] if state_strides is not None else None),
            ssm_state_slot_stride=(state_strides[2] if state_strides is not None else None),
            old_u=self.old_x,
            old_k=self.old_B,
            old_G=self.old_dt,
            replay_work_items=mamba_metadata.replay_work_items[:num_decodes],
            n_writes=mamba_metadata.replay_n_writes,
            history_size=self.replay_history_size,
        )

    @override
    def shutdown(self) -> None:
        super().shutdown()
        self._state_descriptors = None
        self._state_strides = None


def select_gdn_replay_state(
    *,
    spec_config: object | None,
    ssm_cache_dtype: torch.dtype,
    manager_cls: type,
) -> GDNReplayState | None:
    """Apply GDN replay feature gates and return the policy when eligible."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
        CppMambaHybridCacheManager,
        MambaHybridCacheManagerV2,
    )

    use_replay = spec_config is not None and get_sm_version() >= 80
    if spec_config is None:
        logger.info("GDN replay kernel requires speculative decoding; using non-replay path")
    elif spec_config.tokens_per_gen_step > 8:
        logger.info(
            "GDN cached replay supports at most 8 tokens per generation step; using non-replay path"
        )
        use_replay = False
    if spec_config is not None and (
        getattr(spec_config, "eagle_choices", None) is not None
        or getattr(spec_config, "use_dynamic_tree", False)
    ):
        logger.info("GDN replay kernel incompatible with tree attention; using legacy MTP path")
        use_replay = False
    if ssm_cache_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        logger.info(
            "GDN replay kernel does not support quantized SSM cache dtype "
            f"{ssm_cache_dtype}; using legacy MTP path"
        )
        use_replay = False
    if not is_gdn_replay_enabled():
        use_replay = False
    if use_replay and not issubclass(
        manager_cls,
        (CppMambaHybridCacheManager, MambaHybridCacheManagerV2),
    ):
        logger.info(
            "GDN replay requires C++ V1 or V2 Mamba cache manager; "
            f"{manager_cls.__name__} was selected, so the non-replay MTP path will be used"
        )
        use_replay = False
    logger.info(f"GDN replay state update: {'ENABLED' if use_replay else 'DISABLED'}")
    return GDNReplayState(spec_config.tokens_per_gen_step) if use_replay else None


def create_gdn_state(
    manager: MambaHybridCacheManagerV2, use_replay: bool | None
) -> IntermediateState | GDNReplayState:
    """Shared selection for the Qwen3.5 and Qwen4 cache managers."""
    if use_replay is False:
        return IntermediateState()
    if use_replay is True:
        if manager.spec_config is None:
            raise ValueError("GDN replay requires speculative decoding")
        return GDNReplayState(manager.spec_config.tokens_per_gen_step)
    return (
        select_gdn_replay_state(
            spec_config=manager.spec_config,
            ssm_cache_dtype=manager.ssm_state_dtype,
            manager_cls=type(manager),
        )
        or IntermediateState()
    )


def validate_gdn_layout(
    manager: MambaHybridCacheManagerV2, state: IntermediateState | GDNReplayState
) -> None:
    if isinstance(state, GDNReplayState):
        if manager.local_num_mamba_layers and manager._global_n_groups % _mamba_effective_tp_size(
            manager._state_layout.mapping
        ):
            raise ValueError("GDN replay groups must be divisible by the effective TP size")
        state.validate(manager._state_layout)


class Qwen35HybridCacheManagerV2(MambaHybridCacheManagerV2):
    """Qwen3.5/Qwen3Next recurrent state with shared GDN replay."""

    @override
    def __init__(self, *args, use_replay_state_update: bool | None = None, **kwargs) -> None:
        self._requested_replay = use_replay_state_update
        kwargs.setdefault("conv_state_layout", "q_k_v")
        super().__init__(*args, **kwargs)

    @override
    def _initialize_model_state(self) -> IntermediateState | GDNReplayState:
        state = create_gdn_state(self, self._requested_replay)

        validate_gdn_layout(self, state)
        return state

    @property
    def use_gdn_cached_replay_all_layer_commit(self) -> bool:
        state = self._speculative_state
        return isinstance(state, GDNReplayState) and state.has_bound_states


def get_gdn_cache_params(config, *, spec_config=None, quant_config=None):
    """Shared Qwen3.5/Qwen4 GDN geometry; attention and PLE stay model-owned."""
    from tensorrt_llm._torch.pyexecutor.config_utils import (
        build_mamba_kv_cache_params,
        get_qwen3_hybrid_layer_masks,
    )

    attention, recurrent = get_qwen3_hybrid_layer_masks(config)
    return build_mamba_kv_cache_params(
        config,
        state_size=config.linear_key_head_dim,
        conv_kernel=config.linear_conv_kernel_dim,
        num_heads=config.linear_num_value_heads,
        n_groups=config.linear_num_key_heads,
        head_dim=config.linear_value_head_dim,
        mamba_mask=recurrent,
        target_full_attn_mask=attention,
        spec_config=spec_config,
        quant_config=quant_config,
    )


__all__ = [
    "GDNReplayState",
    "GDNReplayLayerCache",
    "Qwen35HybridCacheManagerV2",
    "create_gdn_state",
    "select_gdn_replay_state",
    "validate_gdn_layout",
    "get_gdn_cache_params",
]
