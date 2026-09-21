# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mamba2-owned speculative and replay cache state."""

from __future__ import annotations

import os
import sys

import torch

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
    IntermediateState,
    MambaStateLayout,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.replay import ReplayHistory
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.seeds import (
    _reset_mamba_seed_buffer,
    allocate_mamba_seed_buffer,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping


def mamba_seed_rank_offset(mapping: Mapping) -> int:
    """Return a distinct deterministic seed offset for this parallel rank."""
    return mapping.tp_rank * 1_000_003 + mapping.pp_rank * 1_000_033 + mapping.rank * 1_009


class Mamba2State(IntermediateState):
    """Intermediate promotion plus Mamba2 stochastic-rounding seeds."""

    uses_replay = False

    @override
    def __init__(self) -> None:
        super().__init__()
        self.rand_seed: torch.Tensor | None = None
        self._seed_request_counter = 0
        self._seed_rank_offset = 0

    @override
    def bind(
        self,
        context: MambaStateLayout,
        ssm_states: list[torch.Tensor],
        conv_states: list[torch.Tensor],
    ) -> None:
        super().bind(context, ssm_states, conv_states)
        self._seed_rank_offset = context.seed_rank_offset
        if context.stochastic_rounding and ssm_states:
            states = ssm_states[0]
            self.rand_seed = allocate_mamba_seed_buffer(
                states.shape[0], context.seed_rank_offset, states.device
            )

    @override
    def reset_slots(self, slots: torch.Tensor, host_slots: list[int]) -> None:
        if self.rand_seed is None:
            return
        self._seed_request_counter += 1
        _reset_mamba_seed_buffer(
            self.rand_seed,
            slots,
            host_slots,
            counter=self._seed_request_counter,
            rank_offset=self._seed_rank_offset,
        )

    @override
    def _layer_cache_fields(self, layer_offset: int) -> dict[str, torch.Tensor | None]:
        fields = super()._layer_cache_fields(layer_offset)
        if self.rand_seed is not None:
            fields["mamba_ssm_rand_seed"] = self.rand_seed
        return fields

    @override
    def shutdown(self) -> None:
        super().shutdown()
        self.rand_seed = None


def create_mamba2_state(
    *, spec_config: object | None, use_replay: bool
) -> Mamba2State | ReplayHistory:
    """Build Mamba2 intermediate state or compact replay history."""
    if use_replay:
        if spec_config is None:
            raise ValueError("Mamba replay requires speculative decoding")
        return ReplayHistory(spec_config.tokens_per_gen_step)
    return Mamba2State()


def select_mamba2_state(
    *,
    spec_config: object | None,
    ssm_cache_dtype: torch.dtype,
    stochastic_rounding: bool,
) -> Mamba2State | ReplayHistory:
    """Apply Mamba2 replay gates and return its selected update policy."""
    sm = get_sm_version()
    use_replay = spec_config is not None and sm >= 80
    if spec_config is None:
        logger.info("Replay kernel requires speculative decoding; using non-replay path")
    if spec_config is not None and (getattr(spec_config, "use_dynamic_tree", False)):
        logger.info("Replay kernel incompatible with tree attention; using legacy MTP path")
        use_replay = False
    if stochastic_rounding and ssm_cache_dtype == torch.float16 and (sm < 100 or sm in (120, 121)):
        logger.info(
            "Replay kernel Philox requires 100 <= sm < 120; "
            "using legacy MTP path for stochastic rounding support"
        )
        use_replay = False
    if os.environ.get("TRTLLM_USE_MAMBA_REPLAY", "1") == "0":
        logger.info("Replay kernel is disabled by TRTLLM_USE_MAMBA_REPLAY=0")
        use_replay = False
    else:
        logger.info("Replay kernel is not changed since TRTLLM_USE_MAMBA_REPLAY=1")
    return create_mamba2_state(spec_config=spec_config, use_replay=use_replay)


class NemotronHybridCacheManagerV2(MambaHybridCacheManagerV2):
    """Nemotron/Mamba2 geometry, replay selection and seed ownership."""

    _speculative_state: Mamba2State | ReplayHistory

    @override
    def __init__(self, *args, use_replay_state_update: bool | None = None, **kwargs) -> None:
        self._requested_replay = use_replay_state_update
        kwargs.setdefault("conv_state_layout", "x_b_c")
        super().__init__(*args, **kwargs)

    @override
    def _initialize_model_state(self) -> Mamba2State | ReplayHistory:
        if self._requested_replay is not None:
            state = create_mamba2_state(
                spec_config=self.spec_config, use_replay=self._requested_replay
            )
        else:
            state = select_mamba2_state(
                spec_config=self.spec_config,
                ssm_cache_dtype=self.ssm_state_dtype,
                stochastic_rounding=self._mamba_ssm_stochastic_rounding,
            )

        if isinstance(state, ReplayHistory):
            from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
                _mamba_effective_tp_size,
            )

            if self.local_num_mamba_layers and self._global_n_groups % _mamba_effective_tp_size(
                self._state_layout.mapping
            ):
                raise ValueError("Replay state groups must be divisible by the effective TP size")
            state.validate(self._state_layout)
        return state

    def get_mamba_ssm_rand_seed(self) -> torch.Tensor | None:
        return self._speculative_state.rand_seed


def get_nemotron_cache_params(config, *, spec_config=None, quant_config=None):
    """Derive Mamba2 state geometry from the Nemotron hybrid pattern."""
    from tensorrt_llm._torch.pyexecutor.config_utils import build_mamba_kv_cache_params

    pattern = config.hybrid_override_pattern
    return build_mamba_kv_cache_params(
        config,
        state_size=config.ssm_state_size,
        conv_kernel=config.conv_kernel,
        num_heads=config.mamba_num_heads,
        n_groups=config.n_groups,
        head_dim=config.mamba_head_dim,
        mamba_mask=[layer == "M" for layer in pattern],
        target_full_attn_mask=[layer == "*" for layer in pattern],
        spec_config=spec_config,
        quant_config=quant_config,
    )


__all__ = [
    "Mamba2State",
    "NemotronHybridCacheManagerV2",
    "create_mamba2_state",
    "select_mamba2_state",
    "get_nemotron_cache_params",
    "mamba_seed_rank_offset",
]
