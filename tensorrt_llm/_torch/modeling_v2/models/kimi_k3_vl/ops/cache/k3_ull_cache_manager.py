# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KV cache manager of the Kimi K3 SM107 Helix16 GEN target: FP4 MLA KV + KDA checkpoints.

:class:`K3UllCacheManager` is the runtime's ``KimiK3HybridCacheManager`` (FP4 MLA latent
cache composed into the KV cache manager V2, a standalone drafter's layers appended to it) with
the KDA states kept in checkpoint mode for the MR8 KDA kernels: one ``kda_ckpt`` buffer per
layer and state slot holding a recurrent state for every verify position and an extended conv
page (layout: :mod:`._kda_checkpoints`), read at the slot's accepted-draft count of the last
verify. The live SSM / conv pools stay the persistent and transferred state:

* seeded into checkpoint 0 and conv columns ``[0, W)`` (count reset to 0) when a disaggregated
  transfer completes (``on_state_transfer_complete``) and when a suspended request resumes,
  in all local layers with one gather and one scatter per buffer kind when every kind's
  per-layer views are evenly spaced in the pool (as the V2 manager coalesces them), layer by
  layer otherwise;
* written back from the current checkpoint before a suspension and, with block reuse, before a
  request's state is released;
* between verifies nothing is committed: the speculative tail records the accepted drafts
  (``update_mamba_states`` -> ``KDAReplayState.record_acceptance``, unchanged), and the next
  step's indices select that checkpoint.

The checkpoint buffers and the count share the SSM pool group, so the manager accounts for
them in its pool budget (``_get_recurrent_buffer_configs``) and estimate
(``_extra_state_bytes_per_rank``).

Public API
----------

The target declares :class:`K3UllCacheManager` as ``RuntimeHooks.kv_cache_manager_cls``
(``modeling.py``). The engine builds it in place of the ``KimiK3HybridCacheManager`` it resolves
for this model, after its own checks of that choice (KV cache manager V2, the TRTLLM attention
backend, the Python NIXL transceiver in disaggregated serving), and rejects a declared class that
is not a subclass of the resolved one. The constructor rejects
``kv_cache_config.enable_kv_pool_rebalance`` (the checkpoint views are bound once to the SSM
pool's slot count) and the KDA draft records and stochastic rounding the checkpoints replace.

``K3UllCacheManager`` (``manager`` below; ``layer_idx`` is the global model layer id of a
local KDA layer, as for ``mamba_layer_cache``):

``manager.uses_kda_checkpoints -> bool``
    True once the checkpoint pools are bound.
``manager.kda_checkpoint_views(layer_idx) -> K3UllKdaKernelViews(state_flat, conv_flat,
unit_bytes)``
    The layer's MR8 flat views: ``state_flat`` [rows, H, V, K] in the SSM state dtype,
    ``conv_flat`` [conv_rows, 3 * H * K, W - 1 + num_spec] BF16, both with a row stride of
    ``unit_bytes`` bytes. Fixed addresses for the manager's lifetime.
``manager.kda_kernel_caches(layer_idx) -> list[Tensor]``
    ``[conv_flat, state_flat]``, the order the MR8 KDA providers take as ``kv_cache``.
``manager.checkpoint_indices(state_indices, out_slots=None, out_accepted=None) ->
(slots, accepted)``
    For the requests whose manager slots are ``state_indices`` [n] (``mamba_metadata``'s
    generation state indices), the int32 ``spec_state_indices_tensor`` [n, 1 + num_spec] and
    ``num_accepted_tokens`` [n] (last verify's accepted drafts + 1) of every KDA layer. Device
    operations only (CUDA-graph capturable); ``out_*`` are written in place when given.
``manager.seed_kda_checkpoints(state_indices)``
    Checkpoint 0 and the committed conv window of every local layer from the live pools, count
    reset to 0 (a step that advanced the pools, such as a plain decode of all layers).
``manager.commit_kda_checkpoints(state_indices)``
    The live pools of every local layer from the current checkpoints; the count is kept.

``K3UllKdaState(manager)``: the chain's view of the manager (the ``kda_state`` of the chain):

* ``caches(layer_idx) -> [conv_flat, state_flat]`` (= ``manager.kda_kernel_caches``);
* ``kda_checkpoint_views(layer_idx) -> K3UllKdaKernelViews`` (= the manager's);
* ``indices(state_indices, slots, accepted) -> None``: ``state_indices`` int32/int64 [>= n]
  (``mamba_metadata.generation_state_indices[:n]``), ``slots`` int32 [n, 1 + num_spec] and
  ``accepted`` int32 [n] written in place (= ``manager.checkpoint_indices``);
* ``describe() -> str``.

``manager._speculative_state`` is the :class:`K3UllKdaCheckpointState` (a ``KDAReplayState``):
``prev_num_accepted_tokens`` is the int32 [slots] count of accepted drafts (a strided view of
the count buffer) that ``update_mamba_states`` / a fused accept tail writes as
``num_accepted - 1``; ``checkpoint_indices`` reads it back as ``+ 1``. The manager does not
override ``update_mamba_states``.

``manager.mamba_layer_cache(layer_idx)`` returns a :class:`K3UllKdaLayerCache`
(``has_kda_checkpoints`` True, ``has_kda_replay_caches`` False) with the typed per-slot views
``kda_ckpt`` [slots, 1 + num_spec, H, V, K] and ``kda_vconv`` [slots, conv_dim, W - 1 +
num_spec], ``kda_ckpt_kernel_caches`` (= ``kda_kernel_caches``), ``commit_conv_window`` (seed
one layer after it advanced the pools) and ``load_checkpoints`` (commit one layer before a step
that reads the pools).

The checkpoint state class subclasses the runtime's ``KDAReplayState`` and keeps its
``prev_num_accepted_tokens`` and ``record_acceptance`` semantics. A fused accept tail that
records the acceptance itself must accept a ``KDAReplayState`` subclass.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import NamedTuple

import torch

from tensorrt_llm._torch.modules.kimi_kda.cache_manager import (
    KDAReplayLayerCache,
    KDAReplayState,
    KimiK3HybridCacheManager,
    # The engine's disaggregated-role predicate: the parent manager's __init__ sets
    # kda_context_only with it, so the estimate below and the manager agree by construction.
    _is_disagg_context_server,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
    MambaRole,
    MambaStateLayout,
    ReplayLayout,
    _stack_state_views,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import DataRole

from ..common.k3_ull_runtime import fold_block_offsets_copy
from .k3_ull_kda_checkpoints import (
    KDA_CHECKPOINT_COUNT_BYTES,
    KDA_CHECKPOINT_ROLE,
    KDACheckpointLayout,
    KDACheckpointViews,
    checkpoint_indices,
    checkpoint_position_rows,
    commit_kda_checkpoints,
    estimate_checkpoint_bytes_per_slot,
    kda_checkpoint_layout,
    kda_checkpoint_views,
    seed_kda_checkpoints,
    seed_kda_checkpoints_stacked,
    stack_layer_views,
)

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

# The runtime's per-slot accepted-draft count role (``KDAReplayState.prev_num_accepted_tokens``).
_COUNT_ROLE = "prev_num_accepted_tokens"


def _host_slots_to_device(slots: list[int], device: torch.device) -> torch.Tensor:
    """int64 slot ids on ``device``; a non-blocking copy from fresh pinned memory on CUDA."""
    from tensorrt_llm._utils import prefer_pinned

    return torch.tensor(
        slots, dtype=torch.long, pin_memory=device.type == "cuda" and prefer_pinned()
    ).to(device, non_blocking=True)


def _stack_layers(views: list[torch.Tensor]) -> torch.Tensor | None:
    """``[layers, *shape]`` view of per-layer views at a constant spacing, or None: views of one
    storage (CPU pools) by storage offset, the V2 manager's per-layer CUDA pool views by base
    pointer (the runtime's ``_stack_state_views``)."""
    stacked = stack_layer_views(views)
    if stacked is None:
        stacked = _stack_state_views(views)
    return stacked


class _StackedSeedViews(NamedTuple):
    """What :meth:`K3UllKdaCheckpointState.seed_checkpoints` reads and writes, stacked over the
    local layers."""

    ckpt0: torch.Tensor  # [layers, slots, H, V, K] checkpoint 0
    vconv_window: torch.Tensor  # [layers, slots, conv_dim, W] conv columns [0, W)
    ssm: torch.Tensor  # [layers, slots, H, V, K] live recurrent states
    conv: torch.Tensor  # [layers, slots, conv_dim, W] live conv states


def _local_kda_layers(
    model_config,
    mapping: Mapping,
    *,
    spec_config=None,
    is_draft: bool = False,
    use_separate_draft_kv_cache: bool = False,
):
    """``(params, local KDA layer count)`` of this rank, as the manager partitions the layers."""
    from tensorrt_llm._torch.pyexecutor.config_utils import extract_mamba_kv_cache_params
    from tensorrt_llm._torch.pyexecutor.resource_manager import get_pp_layers

    params = extract_mamba_kv_cache_params(
        model_config.pretrained_config,
        spec_config=spec_config,
        quant_config=model_config.quant_config,
    )
    mamba_mask, attention_mask = params.get_layer_masks(
        is_draft=is_draft, use_separate_draft_kv_cache=use_separate_draft_kv_cache
    )
    combined = [m or a for m, a in zip(mamba_mask, attention_mask)]
    local_layers, _ = get_pp_layers(
        sum(combined), mapping, spec_config=spec_config, layer_mask=combined
    )
    return params, sum(mamba_mask[layer] for layer in local_layers)


def estimate_kda_checkpoint_bytes_per_slot(model_config, mapping: Mapping, **kwargs) -> int:
    """Rank-local checkpoint bytes per state slot (all local KDA layers and the count).

    ``kwargs`` are the manager estimator's: ``spec_config``, ``is_disagg``, ``is_draft``,
    ``use_separate_draft_kv_cache`` and optionally ``kda_replay_num_spec``.
    """
    from tensorrt_llm._torch.pyexecutor.config_utils import mamba_effective_tp_size

    if _is_disagg_context_server(kwargs.get("is_disagg", False)):
        return 0
    spec_config = kwargs.get("spec_config")
    if spec_config is None:
        return 0
    num_spec = kwargs.get("kda_replay_num_spec")
    if num_spec is None:
        num_spec = spec_config.tokens_per_gen_step - 1
    params, local_layers = _local_kda_layers(
        model_config,
        mapping,
        spec_config=spec_config,
        is_draft=kwargs.get("is_draft", False),
        use_separate_draft_kv_cache=kwargs.get("use_separate_draft_kv_cache", False),
    )
    if not local_layers:
        return 0
    layout = kda_checkpoint_layout(
        num_spec,
        params.num_heads // mamba_effective_tp_size(mapping),
        params.head_dim,
        params.conv_kernel - 1,
        params.mamba_ssm_cache_dtype or params.dtype,
        params.dtype,
    )
    return estimate_checkpoint_bytes_per_slot(layout, local_layers)


@dataclass(frozen=True, kw_only=True)
class K3UllKdaLayerCache(KDAReplayLayerCache):
    """One KDA layer's checkpoint views (see the module docstring)."""

    # [slots, 1 + num_spec, H, V, K] recurrent states in the SSM state dtype.
    kda_ckpt: torch.Tensor | None = None
    # [slots, conv_dim, W - 1 + num_spec] BF16 raw convolution inputs, dimension-first.
    kda_vconv: torch.Tensor | None = None
    # ``[conv_flat, state_flat]``: the MR8 kernel views of the same storage.
    kda_ckpt_kernel_caches: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def has_kda_checkpoints(self) -> bool:
        return self.kda_ckpt is not None

    @override
    def commit_conv_window(self, slot_indices: torch.Tensor, conv_pool: torch.Tensor) -> None:
        """Checkpoint 0 and the committed conv window from the live pools of this layer, after a
        step advanced them; the step's acceptance then records no drafts."""
        seed_kda_checkpoints(
            self.kda_ckpt, self.kda_vconv, self.temporal, conv_pool, slot_indices.long()
        )

    def load_checkpoints(self, slot_indices: torch.Tensor) -> None:
        """The live pools of this layer from the slots' current checkpoints, before a step that
        reads them."""
        commit_kda_checkpoints(
            self.kda_ckpt,
            self.kda_vconv,
            self.temporal,
            self.conv,
            slot_indices.long(),
            self.prev_num_accepted_tokens,
        )


class K3UllKdaCheckpointState(KDAReplayState):
    """KDA speculative state in checkpoint mode over the manager's V2 pools.

    KDAReplayRegion supplies the shared recurrent-slot views before :meth:`bind`.
    ``prev_num_accepted_tokens`` (the slot's accepted drafts of the last verify, int32) and
    ``update`` / ``record_acceptance`` / ``reset_slots`` are the base class's.
    """

    def __init__(self, num_speculative_tokens: int) -> None:
        super().__init__(num_speculative_tokens)
        self.checkpoint_layout: KDACheckpointLayout | None = None
        self.checkpoint_views: list[KDACheckpointViews] = []
        self.checkpoint_slot_units = 0
        self.checkpoint_state_units = 0
        self.checkpoint_header_units = 0
        self.checkpoint_unit_bytes = 0
        self._position_rows: torch.Tensor | None = None
        self._stacked_seed: _StackedSeedViews | None = None

    @override
    def validate(self, context: MambaStateLayout) -> None:
        super().validate(context)
        if not context.mamba_pp_layers:
            return
        self.layout(context)

    def layout(self, context: MambaStateLayout) -> KDACheckpointLayout:
        """The per-slot payload of one local layer (raises for unsupported geometry / dtypes)."""
        heads, value_dim, key_dim = context.ssm_state_shape
        if value_dim != key_dim:
            raise ValueError("KDA checkpoints require equal value and key head dimensions")
        if list(context.conv_section_dims) != [heads * key_dim] * 3:
            raise ValueError("KDA checkpoints require [Q | K | V] conv sections of H * K channels")
        return kda_checkpoint_layout(
            self.num_speculative_tokens,
            heads,
            key_dim,
            context.conv_state_shape[1],
            context.ssm_state_dtype,
            context.conv_state_dtype,
        )

    @property
    @override
    def replay_shares_state_slot(self) -> bool:
        return True

    @override
    def replay_layout(self, context: MambaStateLayout, layer_id: int) -> ReplayLayout:
        if not context.mamba_pp_layers:
            return {}
        layout: ReplayLayout = {
            DataRole(KDA_CHECKPOINT_ROLE): (torch.uint8, [self.layout(context).nbytes]),
        }
        if layer_id == context.mamba_pp_layers[0]:
            layout[DataRole(_COUNT_ROLE)] = (
                torch.int32,
                [KDA_CHECKPOINT_COUNT_BYTES // torch.int32.itemsize],
            )
        return layout

    @classmethod
    @override
    def estimate_bytes_per_slot(cls, model_config, mapping: Mapping, **kwargs) -> int:
        return estimate_kda_checkpoint_bytes_per_slot(model_config, mapping, **kwargs)

    def bind_checkpoints(
        self,
        layout: KDACheckpointLayout,
        raw_buffers: list[torch.Tensor],
        count: torch.Tensor,
    ) -> None:
        """Views of the layers' ``kda_ckpt`` buffers (``[slots, payload]`` uint8, in local layer
        order) and the int32 per-slot count ``[slots]``."""
        views = [kda_checkpoint_views(raw, layout) for raw in raw_buffers]
        units = {(v.unit_bytes, v.slot_units, v.state_units, v.header_units) for v in views}
        if len(units) > 1:
            raise RuntimeError("KDA checkpoint buffers of the local layers differ in slot stride")
        if views and any(v.ckpt.shape[0] != count.shape[0] for v in views):
            raise RuntimeError("KDA checkpoint buffers and the count differ in slot capacity")
        self.checkpoint_layout = layout
        self.checkpoint_views = views
        if views:
            (
                self.checkpoint_unit_bytes,
                self.checkpoint_slot_units,
                self.checkpoint_state_units,
                self.checkpoint_header_units,
            ) = units.pop()
        self.pool_layers = [
            {
                "kda_ckpt": v.ckpt,
                "kda_vconv": v.vconv,
                "kda_ckpt_kernel_caches": (v.conv_flat, v.state_flat),
            }
            for v in views
        ]
        self.prev_num_accepted_tokens = count

    @override
    def bind(
        self,
        context: MambaStateLayout,
        ssm_states: list[torch.Tensor],
        conv_states: list[torch.Tensor],
    ) -> None:
        """Borrow the live pools' per-layer views; the checkpoint views are already bound."""
        self._conv_section_dims = context.conv_section_dims
        self._ssm_states = tuple(ssm_states)
        self._conv_states = tuple(conv_states)
        self._committed_window = context.conv_state_shape[1] if conv_states else 0
        if not ssm_states:
            return
        if self._pool_views is None or len(self._pool_views) != len(ssm_states):
            raise RuntimeError("KDA checkpoint views must be bound for every local KDA layer")
        raw = [view[DataRole(KDA_CHECKPOINT_ROLE)] for view in self._pool_views]
        if any(buffer.shape[0] != context.slot_capacity for buffer in raw):
            raise RuntimeError("KDA checkpoint buffers have a different slot capacity")
        count = self._pool_views[0][DataRole(_COUNT_ROLE)][..., 0]
        self.bind_checkpoints(self.layout(context), raw, count)
        device = ssm_states[0].device
        self.intermediate_indices = torch.arange(
            context.max_batch_size, dtype=torch.int32, device=device
        )
        self._position_rows = checkpoint_position_rows(self.checkpoint_views[0], device)
        self._stack_seed_views()
        logger.info(
            f"KDA checkpoints: {len(ssm_states)} layers x {context.slot_capacity} slots x "
            f"{self.checkpoint_layout.nbytes} B ({self.checkpoint_layout.state_dtype} state, "
            f"{self.checkpoint_unit_bytes}-byte index rows)"
        )

    @override
    def _stack_seed_views(self) -> None:
        """Stack what :meth:`seed_checkpoints` touches, so that it runs on all layers with one
        gather and one scatter per buffer kind; None (seeding layer by layer) when a kind's
        per-layer views are not evenly spaced."""
        self._stacked_seed = None
        if not self._ssm_states:
            return
        window = self._conv_states[0].shape[-1]
        stacked = _StackedSeedViews(
            ckpt0=_stack_layers([views.ckpt[:, 0] for views in self.checkpoint_views]),
            vconv_window=_stack_layers(
                [views.vconv[..., :window] for views in self.checkpoint_views]
            ),
            ssm=_stack_layers(list(self._ssm_states)),
            conv=_stack_layers(list(self._conv_states)),
        )
        if any(view is None for view in stacked):
            logger.warning(
                "KDA checkpoints: the local layers' pool views are not evenly spaced; "
                "transferred requests are seeded layer by layer"
            )
            return
        self._stacked_seed = stacked
        logger.info(
            f"KDA checkpoints of transferred requests are seeded in all {len(self._ssm_states)} "
            "local KDA layers with one launch per buffer kind"
        )

    @override
    def make_layer_cache(
        self, layer_offset: int, conv: torch.Tensor, temporal: torch.Tensor
    ) -> K3UllKdaLayerCache:
        return K3UllKdaLayerCache(
            conv=conv, temporal=temporal, **self._layer_cache_fields(layer_offset)
        )

    @torch.inference_mode()
    def seed_checkpoints(self, slots: torch.Tensor) -> None:
        """Checkpoint 0 and the conv window of ``slots`` in every layer from the live pools;
        their counts are reset to 0."""
        if self.prev_num_accepted_tokens is None or slots.numel() == 0:
            return
        slots = slots.long()
        self.prev_num_accepted_tokens.index_fill_(0, slots, 0)
        stacked = self._stacked_seed
        if stacked is not None:
            seed_kda_checkpoints_stacked(
                stacked.ckpt0, stacked.vconv_window, stacked.ssm, stacked.conv, slots
            )
            return
        for layer, (ssm, conv) in enumerate(zip(self._ssm_states, self._conv_states)):
            views = self.pool_layers[layer]
            seed_kda_checkpoints(views["kda_ckpt"], views["kda_vconv"], ssm, conv, slots)

    def commit_checkpoints(self, slots: torch.Tensor) -> None:
        """The live pools of ``slots`` in every layer from their current checkpoints; the counts
        are kept."""
        if self.prev_num_accepted_tokens is None or slots.numel() == 0:
            return
        slots = slots.long()
        for layer, (ssm, conv) in enumerate(zip(self._ssm_states, self._conv_states)):
            views = self.pool_layers[layer]
            commit_kda_checkpoints(
                views["kda_ckpt"],
                views["kda_vconv"],
                ssm,
                conv,
                slots,
                self.prev_num_accepted_tokens,
            )

    @override
    def seed_transferred_slots(
        self, state_indices: torch.Tensor, replay_indices: torch.Tensor | None = None
    ) -> None:
        # The checkpoint and the transferred live state use the same slots.
        self.seed_checkpoints(state_indices)

    def commit_pending_drafts(self, slot: int) -> None:
        if self.prev_num_accepted_tokens is None:
            raise RuntimeError("KDA checkpoints are not bound")
        slots = torch.full(
            (1,), slot, dtype=torch.long, device=self.prev_num_accepted_tokens.device
        )
        self.commit_checkpoints(slots)
        # zero_() on the element view stays stream-ordered.
        self.prev_num_accepted_tokens[slot].zero_()

    def checkpoint_indices(
        self,
        state_indices: torch.Tensor,
        out_slots: torch.Tensor | None = None,
        out_accepted: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """See :meth:`K3UllCacheManager.checkpoint_indices`."""
        if self._position_rows is None or self.prev_num_accepted_tokens is None:
            raise RuntimeError("KDA checkpoint indices need bound checkpoint views")
        return checkpoint_indices(
            state_indices,
            self.prev_num_accepted_tokens,
            self._position_rows,
            self.checkpoint_slot_units,
            out_slots,
            out_accepted,
        )

    @override
    def shutdown(self) -> None:
        super().shutdown()
        self.checkpoint_views = []
        self._position_rows = None
        self._stacked_seed = None


class K3UllKdaKernelViews(NamedTuple):
    """One KDA layer's MR8 flat views (see :meth:`K3UllCacheManager.kda_checkpoint_views`)."""

    state_flat: torch.Tensor
    conv_flat: torch.Tensor
    unit_bytes: int


class K3UllCacheManager(KimiK3HybridCacheManager):
    """FP4 MLA + appended draft layers (the base class) with KDA checkpoints (see the module
    docstring)."""

    @override
    def __init__(self, *args, **kwargs) -> None:
        # kv_cache_config is MambaHybridCacheManagerV2's tenth positional parameter.
        kv_cache_config = kwargs.get("kv_cache_config", args[9] if len(args) > 9 else None)
        if getattr(kv_cache_config, "enable_kv_pool_rebalance", False):
            raise ValueError(
                "KDA checkpoint views are bound once to the SSM pool's slot count: set "
                "kv_cache_config.enable_kv_pool_rebalance=False"
            )
        if kwargs.get("kda_draft_records", False):
            raise ValueError("KDA checkpoints replace the KDA draft records")
        if kwargs.get("mamba_ssm_stochastic_rounding", False):
            raise ValueError("KDA checkpoints do not round the recurrent state stochastically")
        super().__init__(*args, **kwargs)

    @override
    def _initialize_spec_state(self) -> K3UllKdaCheckpointState | None:
        if self.spec_config is None:
            raise ValueError("KDA checkpoints require speculative decoding")
        if self.kda_context_only:
            return None
        num_spec = self._requested_num_spec
        if num_spec is None:
            num_spec = self.spec_config.tokens_per_gen_step - 1
        self._kda_replay = K3UllKdaCheckpointState(num_spec)
        self._kda_replay.validate(self._state_layout)
        return self._kda_replay

    @property
    def uses_kda_checkpoints(self) -> bool:
        replay = self._kda_replay
        return isinstance(replay, K3UllKdaCheckpointState) and bool(replay.checkpoint_views)

    @override
    def _setup_model_state(self) -> None:
        if self._kda_replay is not None and self.local_num_mamba_layers > 0:
            self._check_checkpoint_pool_group()
        super()._setup_model_state()

    def _check_checkpoint_pool_group(self) -> None:
        """The checkpoint, count and live-state buffers must share one pool group, so that they
        share the slot index and its lifecycle."""
        required = {
            (self.layer_offsets[layer], str(role))
            for layer in self.mamba_pp_layers
            for role in (MambaRole.SSM_STATE, MambaRole.CONV_STATE, KDA_CHECKPOINT_ROLE)
        }
        required.add((self.layer_offsets[self.mamba_pp_layers[0]], _COUNT_ROLE))
        if not any(
            required.issubset(
                {
                    (int(buffer_id.layer_id), str(buffer_id.role))
                    for coalesced in variant.coalesced_buffers
                    for buffer_id in coalesced.buffer_ids
                }
            )
            for group in self.impl.pool_group_descs
            for variant in group.slot_desc.variants
        ):
            raise RuntimeError(
                "KDA checkpoint and recurrent-state buffers must share one KVCM pool group"
            )

    @override
    def copy_batch_block_offsets(
        self,
        dst_tensor: torch.Tensor,
        request_ids: list[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
        max_blocks: int | None = None,
    ) -> None:
        """The step's block-offset copy, done by the open input commit's kernel when it takes it
        (:func:`.runtime.fold_block_offsets_copy`); otherwise the base class's copy."""
        if (
            beam_width == 1
            and not self._use_per_layer_page_tables
            and not self._page_table_materializer.uses_device_expansion
        ):
            copy_idx = self.index_mapper.get_copy_index(
                request_ids, num_contexts, beam_width, self.kv_cache_type == CacheTypeCpp.CROSS
            )
            assert copy_idx.shape[0] == num_seqs
            if fold_block_offsets_copy(
                self.host_kv_cache_block_offsets,
                dst_tensor,
                copy_idx,
                self.index_scales,
                self.kv_offset,
                self._stream,
            ):
                return
        super().copy_batch_block_offsets(
            dst_tensor, request_ids, beam_width, num_contexts, num_seqs, max_blocks
        )

    @override
    def free_resources(self, request, pin_on_release: bool = False) -> None:
        self._commit_checkpoints_for_reuse(request)
        super().free_resources(request, pin_on_release)

    def _commit_checkpoints_for_reuse(self, request) -> None:
        """With block reuse a released slot's state is offered to later requests: commit its
        current checkpoint to the live pools before the base release commits the blocks."""
        replay = self._kda_replay
        request_id = request.py_request_id
        if (
            replay is None
            or not self.kv_cache_config.enable_block_reuse
            or self.local_num_mamba_layers == 0
            or request_id in replay.unseeded_requests
        ):
            return
        cache = self.kv_cache_map.get(request_id)
        slot = self._request_id_to_state_index.get(request_id, -1)
        if cache is None or not cache.is_active or slot < 0:
            return
        slots = _host_slots_to_device([slot], replay.prev_num_accepted_tokens.device)
        with replay.on_cache_stream(self._stream):
            replay.commit_checkpoints(slots)

    @classmethod
    @override
    def _extra_state_bytes_per_rank(cls, model_config, mapping: Mapping, **kwargs) -> int:
        return estimate_kda_checkpoint_bytes_per_slot(model_config, mapping, **kwargs)

    @classmethod
    @override
    def _estimate_fixed_region_bytes(cls, model_config, mapping: Mapping, **kwargs) -> int:
        # Checkpoint bytes are included in the recurrent-slot estimate.
        return 0

    def _checkpoint_state(self) -> K3UllKdaCheckpointState:
        replay = self._kda_replay
        if not isinstance(replay, K3UllKdaCheckpointState) or not replay.checkpoint_views:
            raise RuntimeError("this cache manager keeps no KDA checkpoints on this rank")
        return replay

    def kda_checkpoint_views(self, layer_idx: int) -> K3UllKdaKernelViews:
        """The MR8 flat views of KDA layer ``layer_idx`` (global id)."""
        replay = self._checkpoint_state()
        views = replay.checkpoint_views[self.mamba_layer_offsets[layer_idx]]
        return K3UllKdaKernelViews(views.state_flat, views.conv_flat, views.unit_bytes)

    def kda_kernel_caches(self, layer_idx: int) -> list[torch.Tensor]:
        """``[conv_flat, state_flat]`` of KDA layer ``layer_idx`` (global id)."""
        views = self.kda_checkpoint_views(layer_idx)
        return [views.conv_flat, views.state_flat]

    def checkpoint_indices(
        self,
        state_indices: torch.Tensor,
        out_slots: torch.Tensor | None = None,
        out_accepted: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """int32 ``(spec_state_indices [n, 1 + num_spec], num_accepted_tokens [n])`` of the
        requests whose manager slots are ``state_indices`` [n]; device operations only."""
        return self._checkpoint_state().checkpoint_indices(state_indices, out_slots, out_accepted)

    def seed_kda_checkpoints(self, state_indices: torch.Tensor) -> None:
        """Checkpoint 0 and the conv window of every local layer from the live pools; counts
        reset to 0."""
        self._checkpoint_state().seed_checkpoints(state_indices)

    def commit_kda_checkpoints(self, state_indices: torch.Tensor) -> None:
        """The live pools of every local layer from the current checkpoints; counts kept."""
        self._checkpoint_state().commit_checkpoints(state_indices)


class K3UllKdaState:
    """The MR8 chain's KDA state in a :class:`K3UllCacheManager`.

    ``caches(layer)``: ``[conv_flat, state_flat]`` of the layer (fixed addresses, bound before
    capture). ``indices(state_indices, slots, accepted)``: the step's ``spec_state_indices_tensor``
    [n, 1 + num_spec] and ``num_accepted_tokens`` [n] of the requests' manager slots, written in
    place (device ops, CUDA-graph safe). Nothing is committed after the acceptance: the
    speculative tail's recorded count is what the next step's indices read.
    """

    def __init__(self, manager: K3UllCacheManager) -> None:
        if not getattr(manager, "uses_kda_checkpoints", False):
            raise RuntimeError(f"{type(manager).__name__} keeps no KDA checkpoints")
        self.manager = manager

    def caches(self, layer: int) -> list[torch.Tensor]:
        return self.manager.kda_kernel_caches(layer)

    def kda_checkpoint_views(self, layer: int) -> K3UllKdaKernelViews:
        return self.manager.kda_checkpoint_views(layer)

    def indices(
        self, state_indices: torch.Tensor, slots: torch.Tensor, accepted: torch.Tensor
    ) -> None:
        """The first ``slots.shape[0]`` requests of ``state_indices``."""
        self.manager.checkpoint_indices(state_indices[: slots.shape[0]], slots, accepted)

    def describe(self) -> str:
        return f"manager checkpoints ({type(self.manager).__name__})"


__all__ = [
    "K3UllCacheManager",
    "K3UllKdaCheckpointState",
    "K3UllKdaKernelViews",
    "K3UllKdaLayerCache",
    "K3UllKdaState",
    "estimate_kda_checkpoint_bytes_per_slot",
]
