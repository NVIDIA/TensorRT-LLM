# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated-DeltaNet replay integration for Mamba cache managers."""

from dataclasses import dataclass

import torch

from tensorrt_llm._torch.modules.mamba.cache_manager import (
    Mamba2ReplayLayerCache,
    Mamba2ReplayStateUpdateStrategy,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
    MambaCacheBuildContext,
    MambaLayerCache,
    MambaStateUpdateBatch,
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


class GDNReplayStateUpdateStrategy(Mamba2ReplayStateUpdateStrategy):
    """Own GDN all-layer cached-replay checkpoint commits."""

    def __init__(self, tokens_per_gen_step: int) -> None:
        super().__init__(tokens_per_gen_step)
        self._state_descriptors: torch.Tensor | None = None
        self._state_strides: tuple[int, int, int] | None = None

    @property
    def key(self) -> str:
        return "gdn_cached_replay"

    def bind(self, context: MambaCacheBuildContext, manager: object) -> None:
        super().bind(context, manager)
        states = getattr(manager, "all_ssm_states")
        if not states:
            return
        state_shape = getattr(manager, "ssm_state_shape")
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
        setattr(manager, "use_gdn_cached_replay_all_layer_commit", True)
        state_layout = "affine" if self._state_strides is not None else "indirect"
        logger.info_once(
            "Configured GDN cached replay commit mode for V2: small-batch fused, "
            f"large-batch all-layer; state layout: {state_layout}",
            key="gdn_cached_replay_v2_commit_mode_fused",
        )

    def update(self, batch: MambaStateUpdateBatch, manager: object) -> bool:
        num_decodes = batch.num_generations
        if num_decodes >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE:
            self.commit_all_layers(batch.attention_metadata, num_decodes, manager)
        return super().update(batch, manager)

    def make_layer_cache(
        self,
        layer_offset: int,
        conv: torch.Tensor,
        temporal: torch.Tensor,
    ) -> MambaLayerCache:
        return GDNReplayLayerCache(
            conv=conv,
            temporal=temporal,
            **self.layer_cache_fields(layer_offset),
        )

    def commit_all_layers(
        self, attention_metadata: object, num_decodes: int, manager: object
    ) -> None:
        """Commit every local GDN checkpoint using one partitioned launch."""
        states = getattr(manager, "all_ssm_states")
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

    def shutdown(self, manager: object | None = None) -> None:
        super().shutdown(manager)
        self._state_descriptors = None
        self._state_strides = None
        if manager is not None:
            setattr(manager, "use_gdn_cached_replay_all_layer_commit", False)


def build_gdn_state_update_strategy(
    *,
    spec_config: object | None,
    ssm_cache_dtype: torch.dtype,
    manager_cls: type,
) -> GDNReplayStateUpdateStrategy | None:
    """Apply GDN replay feature gates and return the strategy when eligible."""
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
    return GDNReplayStateUpdateStrategy(spec_config.tokens_per_gen_step) if use_replay else None


__all__ = [
    "GDNReplayLayerCache",
    "GDNReplayStateUpdateStrategy",
    "build_gdn_state_update_strategy",
]
