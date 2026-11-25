# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KV cache V2 layout of the NVFP4 MLA latent cache and its BF16 tail ring.

``Fp4MlaCacheLayout`` is the object a V2 manager holds when it stores an MLA
latent cache as NVFP4: the pool, page-table and view code of the layout, and
its state. ``cache_manager.Fp4MlaV2CacheLayoutPolicy`` validates the
configuration and creates it.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional

import torch

from tensorrt_llm._torch.disaggregation.resource.page import MapperKind, RoleLayout
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.resource_manager import DataType
from tensorrt_llm._utils import (
    TensorWrapper,
    binding_to_torch_dtype,
    convert_to_torch_tensor,
    prefer_pinned,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    DataRole,
    KVCacheManagerConfig,
    LayerId,
    PageIndexMode,
    ReusePolicy,
)

from .config import FP4_BLOCK_SIZE
from .layout import get_fp4_mla_v_scale_pool_size

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest


def _persistent_cache_view_mode() -> torch.inference_mode:
    """Context for building cached views of persistent KV-cache storage.

    The views are cached and also written outside inference mode, for
    example by the post-warmup KV-cache scrub, but are often first built
    inside a warmup forward. Building them with inference mode disabled keeps
    them normal tensors.
    """
    return torch.inference_mode(False)


@dataclass(frozen=True)
class Fp4MlaPageTableSpec:
    """Stable page-table conversion contract consumed by FP4 MLA kernels."""

    cache_pool_id: int
    cache_page_index_scale: int
    hp_pool_id: int
    hp_page_index_scale: int
    hp_is_paged: bool = True


class Fp4MlaCacheLayout:
    """FP4 MLA storage layout of one KV cache manager V2.

    It holds what the layout adds to a manager: the configuration, the layer
    bookkeeping of the FP4, FP8-fallback, drafter and HP-ring pools, and the
    views and page tables the FP4 MLA attention code reads.

    A manager that uses the layout keeps it as ``fp4_mla_layout`` and forwards
    its layout hooks to it (``Fp4MlaLayoutHooks``). A manager with the hooks
    and no layout has ``fp4_mla_layout = None``; other managers have no such
    attribute, so shared code reads it through ``fp4_mla_cache_layout``.
    ``Fp4MlaV2CacheLayoutPolicy.configure_manager``
    creates the layout before the manager's base class is constructed, because
    that construction already runs the hooks.
    """

    # Transfer layout of the MLA roles. FP4 and FP8-fallback latent pages are
    # written token-major by the FP4 MLA kernels, and every TP rank holds the
    # same single-head latent state, including its HP ring.
    DISAGG_ROLE_MAPPER_KINDS: dict[DataRole, MapperKind] = {
        Role.KEY: MapperKind.NHD,
        Role.KEY_BLOCK_SCALE: MapperKind.NHD,
        Role.MLA_HP_TAIL: MapperKind.REPLICATED,
    }
    # V scales and packed V derive from K and K scales; the generation worker
    # rebuilds them locally in rebuild_disagg_imported_cache.
    DISAGG_IGNORED_ROLES: frozenset[DataRole] = frozenset((Role.MLA_V_SCALE, Role.MLA_V_PACKED))

    def __init__(self, manager: "KVCacheManagerV2") -> None:
        # Weak: the manager owns its layout, so the pair is not a reference
        # cycle and the layout's device buffers go with the manager.
        self._manager_ref = weakref.ref(manager)
        # Set by configure_manager.
        self.mla_v_scale_head_dim: int = 0
        # None when decode reads V from the KV cache or the worker keeps no packed V.
        self.mla_v_head_dim: Optional[int] = None
        self.fp4_mla_k_residual_dim: int = 0
        # A disaggregated context worker keeps neither V scales nor packed V
        # (see is_fp4_mla_context_only_cache).
        self.fp4_mla_context_only_cache: bool = False
        self._fp4_mla_storage_backend: str = ""
        self._fp4_mla_hp_pool_size: int = 0
        # Global layers of the model that keep an FP8 latent cache.
        self._fp8_mla_global_layer_ids: frozenset[int] = frozenset()
        # Set by apply_cache_config: the manager layers of each kind of pool.
        self._cache_manager_layer_ids: list[LayerId] = []
        self._hp_manager_layer_ids: list[LayerId] = []
        self._fp8_mla_manager_layer_ids: list[LayerId] = []
        # Manager layers of a standalone drafter appended to the lifecycle manager.
        self._draft_manager_layer_ids: tuple[LayerId, ...] = ()
        self._fp4_mla_local_to_compact: dict[int, int] = {}
        self._fp4_mla_compact_to_local: list[int] = []
        self._fp4_mla_view_cache: dict[Any, Any] = {}
        self._fp8_context_scratch: Optional[Any] = None
        self._fp4_mla_q_buffers: Optional[dict[str, torch.Tensor]] = None
        # Set once align_snapshot_points has logged a snapshot move.
        self._snapshot_realignment_logged: bool = False

    @property
    def _manager(self) -> "KVCacheManagerV2":
        """The KV cache manager that owns this layout."""
        manager = self._manager_ref()
        if manager is None:
            raise ReferenceError(
                "The KV cache manager of this FP4 MLA cache layout no longer exists."
            )
        return manager

    @property
    def blocks_in_primary_pool(self) -> int:
        """Pages of the full-history pool the FP4 MLA layers share."""
        return self._role_encoded_page_capacity(self._cache_manager_layer_ids[0], Role.KEY)

    def get_or_create_fp8_context_scratch(self, factory: Callable[[], Any]) -> Any:
        scratch = self._fp8_context_scratch
        if scratch is None:
            scratch = factory()
            self._fp8_context_scratch = scratch
        return scratch

    def get_or_create_fp4_mla_q_buffers(self) -> dict[str, torch.Tensor]:
        buffers = self._fp4_mla_q_buffers
        if buffers is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Cannot create FP4 MLA Q buffers while capturing a CUDA graph. "
                    "Run a warmup forward first."
                )
            buffers = {}
            self._fp4_mla_q_buffers = buffers
        return buffers

    @property
    def fp4_mla_hp_pool_size(self) -> int:
        """Number of BF16 tokens in each layer's rewind-capable HP ring."""
        return self._fp4_mla_hp_pool_size

    def _fp8_mla_local_layer_indices(self) -> list[int]:
        fallback_layers = self._fp8_mla_global_layer_ids
        is_linear_layer = getattr(self._manager, "_is_local_mamba_layer", None)
        return [
            local_layer
            for local_layer in range(self._manager.num_local_layers)
            if self._manager.pp_layers[local_layer] in fallback_layers
            and (not callable(is_linear_layer) or not is_linear_layer(local_layer))
        ]

    def _fp4_mla_local_layer_indices(self) -> list[int]:
        # Draft layers appended by a standalone drafter keep their own storage.
        excluded = set(self._fp8_mla_local_layer_indices()) | set(
            self._manager._appended_draft_layer_specs
        )
        is_linear_layer = getattr(self._manager, "_is_local_mamba_layer", None)
        return [
            local_layer
            for local_layer in range(self._manager.num_local_layers)
            if local_layer not in excluded
            and (not callable(is_linear_layer) or not is_linear_layer(local_layer))
        ]

    def _fp4_mla_compact_layer_idx(self, local_layer: int) -> int:
        try:
            return self._fp4_mla_local_to_compact[local_layer]
        except KeyError as error:
            raise ValueError(
                f"Local layer {local_layer} is not an FP4 MLA attention layer."
            ) from error

    def _fp8_mla_bytes_per_token(self, local_layer_idx: int) -> int:
        return (
            self._manager.num_kv_heads_per_layer[local_layer_idx]
            * self._manager.head_dim_per_layer[local_layer_idx]
            * torch.empty((), dtype=torch.float8_e4m3fn).element_size()
        )

    def buffer_roles_for_layer(
        self, local_layer_idx: int, default: Callable[[int], List[DataRole]]
    ) -> List[DataRole]:
        if local_layer_idx in self._fp8_mla_local_layer_indices():
            return [Role.KEY]
        return default(local_layer_idx)

    def get_buffers(
        self,
        default: Callable[..., Optional[torch.Tensor]],
        layer_idx: int,
        kv_layout: str = "NHD",
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        """Return the layer's primary paged-cache view.

        FP8 fallback layers use a buffer role and physical page geometry that
        differ from the manager-wide NVFP4 layout. Construct their view from
        the role descriptor directly so FMHA dispatch observes the same FP8
        dtype and logical shape as an all-FP8 cache manager. ``kwargs`` are
        the manager's own keyword arguments, for the view it builds.
        """
        local_layer = self._manager.layer_offsets[layer_idx]
        is_linear_layer = getattr(self._manager, "_is_local_mamba_layer", None)
        if callable(is_linear_layer) and is_linear_layer(local_layer):
            return None
        if local_layer not in self._fp8_mla_local_layer_indices():
            return default(layer_idx, kv_layout, **kwargs)
        if kv_layout not in ("NHD", "HND"):
            raise ValueError(f"Unsupported kv_layout: {kv_layout}")

        page_shape = (
            (
                self._manager.kv_factor,
                self._manager.tokens_per_block,
                self._manager.num_kv_heads_per_layer[local_layer],
                self._manager.head_dim_per_layer[local_layer],
            )
            if kv_layout == "NHD"
            else (
                self._manager.kv_factor,
                self._manager.num_kv_heads_per_layer[local_layer],
                self._manager.tokens_per_block,
                self._manager.head_dim_per_layer[local_layer],
            )
        )
        return self._role_view(
            LayerId(local_layer),
            Role.KEY,
            torch.float8_e4m3fn,
            page_shape,
        )

    def get_kv_cache_dtype(self, layer_idx: Optional[int] = None) -> DataType:
        """Storage dtype of a layer's primary cache role; the manager's dtype without ``layer_idx``.

        A drafter layer keeps its own dtype and an FP8 fallback layer stores
        FP8, whatever the manager-wide dtype is.
        """
        if layer_idx is None:
            return self._manager.dtype
        local_layer = self._manager.layer_offsets[layer_idx]
        if local_layer in self._manager._appended_draft_layer_specs:
            return self._manager._appended_draft_layer_specs[local_layer].dtype
        if local_layer in self._fp8_mla_local_layer_indices():
            return DataType.FP8
        return self._manager.dtype

    def _storage_head_dim(self, local_layer_idx: int) -> int:
        return self._manager.head_dim_per_layer[local_layer_idx] + self.fp4_mla_k_residual_dim

    def _v_scale_bytes_per_page(self) -> int:
        if self.fp4_mla_context_only_cache:
            return 0
        return get_fp4_mla_v_scale_pool_size(
            self.mla_v_scale_head_dim, self._manager.tokens_per_block
        )

    def _v_packed_bytes_per_page(self) -> int:
        if self.mla_v_head_dim is None:
            return 0
        return self.mla_v_head_dim * (self._manager.tokens_per_block // 2)

    def _hp_bytes_per_page(self, local_layer_idx: int) -> int:
        return (
            self._fp4_mla_hp_pool_size
            * self._manager.head_dim_per_layer[local_layer_idx]
            * torch.empty((), dtype=torch.bfloat16).element_size()
        )

    def get_layer_bytes_per_token(
        self, local_layer_idx: int, data_role: DataRole, default: Callable[[int, DataRole], int]
    ) -> int:
        if local_layer_idx in self._fp8_mla_local_layer_indices():
            if data_role in (Role.KEY, Role.ALL):
                return self._fp8_mla_bytes_per_token(local_layer_idx)
            raise ValueError(f"Invalid FP8 MLA V2 data role: {data_role}")
        if local_layer_idx not in self._fp4_mla_local_layer_indices():
            return default(local_layer_idx, data_role)
        storage_head_dim = self._storage_head_dim(local_layer_idx)
        role_sizes = {
            Role.KEY: math.ceil(storage_head_dim / 2),
            Role.KEY_BLOCK_SCALE: math.ceil(storage_head_dim / FP4_BLOCK_SIZE),
            Role.MLA_V_SCALE: math.ceil(
                self._v_scale_bytes_per_page() / self._manager.tokens_per_block
            ),
            Role.MLA_V_PACKED: math.ceil(
                self._v_packed_bytes_per_page() / self._manager.tokens_per_block
            ),
            Role.MLA_HP_TAIL: math.ceil(
                self._hp_bytes_per_page(local_layer_idx) / self._manager.tokens_per_block
            ),
        }
        if data_role == Role.ALL:
            roles = [Role.KEY, Role.KEY_BLOCK_SCALE]
            if not self.fp4_mla_context_only_cache:
                roles.append(Role.MLA_V_SCALE)
            if self.mla_v_head_dim is not None:
                roles.append(Role.MLA_V_PACKED)
            return sum(role_sizes[role] for role in roles)
        if data_role not in role_sizes or role_sizes[data_role] <= 0:
            raise ValueError(f"Invalid FP4 MLA V2 data role: {data_role}")
        return role_sizes[data_role]

    def extra_buffers_per_layer(self) -> dict[int, List[BufferConfig]]:
        result = {}
        for local_layer in self._fp4_mla_local_layer_indices():
            buffers = []
            if not self.fp4_mla_context_only_cache:
                buffers.append(
                    BufferConfig(
                        role=Role.MLA_V_SCALE,
                        size=self._v_scale_bytes_per_page(),
                    )
                )
            if self.mla_v_head_dim is not None:
                buffers.append(
                    BufferConfig(
                        role=Role.MLA_V_PACKED,
                        size=self._v_packed_bytes_per_page(),
                    )
                )
            result[local_layer] = buffers
        return result

    def apply_cache_config(
        self, config: KVCacheManagerConfig, *, hp_reuse_policy: ReusePolicy
    ) -> KVCacheManagerConfig:
        """Lay out the FP4 MLA layers of ``config`` and record their manager layer IDs.

        FP8 fallback and drafter layers get a lifecycle of their own, and every
        FP4 MLA layer gets a virtual layer for its HP ring, reused under
        ``hp_reuse_policy``.
        """
        cache_layers = list(config.layers)
        fp8_local_layers = self._fp8_mla_local_layer_indices()
        fp4_local_layers = self._fp4_mla_local_layer_indices()
        if not fp4_local_layers:
            raise ValueError("FP4 MLA V2 layout requires at least one local attention layer.")
        self._fp8_mla_manager_layer_ids = [LayerId(local_layer) for local_layer in fp8_local_layers]
        if fp8_local_layers:
            fp8_lifecycle_window = (
                self._manager.max_seq_len
                + self._manager.num_extra_kv_tokens
                + self._manager._kv_reserve_draft_tokens
                + 1
            )
            for local_layer in fp8_local_layers:
                cache_layers[local_layer] = AttentionLayerConfig(
                    layer_id=LayerId(local_layer),
                    buffers=[
                        BufferConfig(
                            role=Role.KEY,
                            size=self._fp8_mla_bytes_per_token(local_layer)
                            * self._manager.tokens_per_block,
                        )
                    ],
                    # A distinct, non-evicting lifecycle gives FP8 pages their
                    # own attention-op pool without changing the model window.
                    sliding_window_size=fp8_lifecycle_window,
                    num_sink_tokens=None,
                )
        # Draft layers take one more distinct non-evicting lifecycle: the FP4
        # and FP8 tables address pages with their own pools' index scales.
        self._draft_manager_layer_ids = tuple(
            LayerId(local_layer)
            for local_layer in sorted(self._manager._appended_draft_layer_specs)
        )
        for layer_id in self._draft_manager_layer_ids:
            draft_lifecycle_window = (
                self._manager.max_seq_len
                + self._manager._generation_kv_capacity_headroom
                + self._manager._kv_reserve_draft_tokens
                + 2
            )
            draft_layer = cache_layers[int(layer_id)]
            cache_layers[int(layer_id)] = AttentionLayerConfig(
                layer_id=layer_id,
                buffers=list(draft_layer.buffers),
                sliding_window_size=draft_lifecycle_window,
                num_sink_tokens=draft_layer.num_sink_tokens,
                reuse_policy=draft_layer.reuse_policy,
            )
        self._fp4_mla_local_to_compact = {
            local_layer: compact_layer for compact_layer, local_layer in enumerate(fp4_local_layers)
        }
        self._fp4_mla_compact_to_local = fp4_local_layers
        self._cache_manager_layer_ids = [LayerId(i) for i in fp4_local_layers]
        self._hp_manager_layer_ids = []
        for local_layer in fp4_local_layers:
            layer_id = LayerId(len(cache_layers))
            self._hp_manager_layer_ids.append(layer_id)
            cache_layers.append(
                AttentionLayerConfig(
                    layer_id=layer_id,
                    buffers=[
                        BufferConfig(
                            role=Role.MLA_HP_TAIL,
                            size=self._hp_bytes_per_page(local_layer),
                        )
                    ],
                    sliding_window_size=self._fp4_mla_hp_pool_size,
                    num_sink_tokens=None,
                    reuse_policy=hp_reuse_policy,
                )
            )
        return replace(config, layers=cache_layers)

    def prepare_page_table_tensor(self, index_mapper_capacity: int) -> None:
        cache_layer = self._cache_manager_layer_ids[0]
        hp_layer = self._hp_manager_layer_ids[0]
        cache_pool_id = int(self._manager.impl.get_layer_group_id(cache_layer))
        hp_pool_id = int(self._manager.impl.get_layer_group_id(hp_layer))
        if cache_pool_id == hp_pool_id:
            raise RuntimeError("FP4 MLA full-history and HP state must use distinct lifecycles.")
        fp8_pool_id = None
        if self._fp8_mla_manager_layer_ids:
            fp8_layer = self._fp8_mla_manager_layer_ids[0]
            fp8_pool_id = int(self._manager.impl.get_layer_group_id(fp8_layer))
            if fp8_pool_id in (cache_pool_id, hp_pool_id):
                raise RuntimeError("FP4 MLA, FP8 MLA, and HP state must use distinct lifecycles.")
        draft_pool_id = None
        if self._draft_manager_layer_ids:
            draft_pool_id = int(
                self._manager.impl.get_layer_group_id(self._draft_manager_layer_ids[0])
            )

        # The hybrid manager can split a recurrent layer across groups by buffer reuse policy.
        num_pools = len(self._manager.impl.buffer_grouping)
        pool_pointers = [[[0, 0], [0, 0]] for _ in range(num_pools)]
        pool_pointers[cache_pool_id] = [
            [
                int(self._manager.impl.get_mem_pool_base_address(cache_layer, Role.KEY)),
                int(
                    self._manager.impl.get_mem_pool_base_address(cache_layer, Role.KEY_BLOCK_SCALE)
                ),
            ],
            [0, 0],
        ]
        pool_pointers[hp_pool_id] = [
            [int(self._manager.impl.get_mem_pool_base_address(hp_layer, Role.MLA_HP_TAIL)), 0],
            [0, 0],
        ]
        # Single-buffer pools: the FP8 fallback latents and the drafter's.
        single_buffer_pools = [
            (pool_id, layer_ids)
            for pool_id, layer_ids in (
                (fp8_pool_id, self._fp8_mla_manager_layer_ids),
                (draft_pool_id, self._draft_manager_layer_ids),
            )
            if pool_id is not None
        ]
        for pool_id, layer_ids in single_buffer_pools:
            pool_pointers[pool_id] = [
                [int(self._manager.impl.get_mem_pool_base_address(layer_ids[0], Role.KEY)), 0],
                [0, 0],
            ]
        self._manager.kv_cache_pool_pointers = torch.tensor(
            pool_pointers,
            dtype=torch.int64,
            device="cpu",
            pin_memory=prefer_pinned(),
        )

        mapping = [[0, 0] for _ in range(self._manager.num_local_layers)]
        for compact_layer, layer_id in enumerate(self._cache_manager_layer_ids):
            converter = self._manager.impl.get_page_index_converter(layer_id, Role.KEY)
            local_layer = self._fp4_mla_compact_to_local[compact_layer]
            mapping[local_layer] = [cache_pool_id, int(converter.layer_offset)]
        for pool_id, layer_ids in single_buffer_pools:
            for layer_id in layer_ids:
                converter = self._manager.impl.get_page_index_converter(layer_id, Role.KEY)
                mapping[int(layer_id)] = [pool_id, int(converter.layer_offset)]
        self._manager.kv_cache_pool_mapping = torch.tensor(
            mapping,
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )

        self._manager.index_scales = torch.ones(
            num_pools, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
        )
        self._manager.index_scales[cache_pool_id] = int(
            self._manager.impl.get_page_index_converter(cache_layer, Role.KEY).scale
        )
        self._manager.index_scales[hp_pool_id] = int(
            self._manager.impl.get_page_index_converter(hp_layer, Role.MLA_HP_TAIL).scale
        )
        for pool_id, layer_ids in single_buffer_pools:
            self._manager.index_scales[pool_id] = int(
                self._manager.impl.get_page_index_converter(layer_ids[0], Role.KEY).scale
            )
        self._manager.kv_offset = torch.zeros(
            num_pools, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
        )
        self._manager._index_scale_ints = self._manager.index_scales.tolist()
        # FP4 MLA builds one page table per lifecycle pool (each pool holds a
        # single layer kind) and rejects enable_swa_scratch_reuse, so the base
        # per-layer page-table copy never applies.
        self._manager._use_per_layer_page_tables = False
        self._manager.num_attention_op_pools = num_pools
        self._manager.host_kv_cache_block_offsets = torch.zeros(
            num_pools,
            index_mapper_capacity * self._manager.max_beam_width,
            2,
            self._manager.max_blocks_per_seq,
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )

        self._manager._prepare_page_table_materializer(index_mapper_capacity)

    def finalize(self) -> None:
        """Validate the allocated pools and initialize the derived FP4 state. Call after construction."""
        self._validate_fp4_mla_layer_groups()
        # Partial pages leave unused V-scale tiles untouched while the
        # fixed-width PV path can load the complete scale page. Match V1's
        # deterministic initialization before any warmup or graph capture.
        v_scale_pool = self.get_mla_v_scale_pool_base()
        if v_scale_pool is not None:
            stream = self._manager._stream
            with torch.cuda.stream(stream):
                v_scale_pool.zero_()
            stream.synchronize()

    def virtual_layer_pool_roles(
        self, layer_id: int
    ) -> Optional[tuple[DataRole, Optional[DataRole]]]:
        """Pool roles of a virtual layer this layout appended (an HP ring), else ``None``."""
        if LayerId(layer_id) in self._hp_manager_layer_ids:
            return Role.MLA_HP_TAIL, None
        return None

    def align_snapshot_points(self, request: "LlmRequest") -> None:
        """Move a hybrid manager's recurrent-state snapshots onto full cache blocks.

        The FP4 MLA cache quantizes the latent cache in 16-token tiles and
        disables partial-block reuse, so a reused prefix always ends on a
        cache block. A snapshot inside a block would restore the recurrent
        state of a later token than the reused prefix ends at; one below the
        first block can never be reused. Snapshot points also end the
        context chunks, which keeps every non-final chunk tile-aligned.
        """
        block = self._manager.tokens_per_block
        points = request.expect_snapshot_points
        aligned = sorted({point // block * block for point in points if point >= block})
        if aligned != sorted(set(points)) and not self._snapshot_realignment_logged:
            self._snapshot_realignment_logged = True
            logger.info(
                "FP4 MLA moves Mamba snapshot points down to "
                f"{block}-token cache blocks and drops points below the "
                f"first block (logged once; request {request.py_request_id}: "
                f"{sorted(set(points))} -> {aligned})."
            )
        request.expect_snapshot_points = aligned

    def _validate_fp4_mla_layer_groups(self) -> None:
        cache_groups = {
            int(self._manager.impl.get_layer_group_id(layer_id))
            for layer_id in self._cache_manager_layer_ids
        }
        hp_groups = {
            int(self._manager.impl.get_layer_group_id(layer_id))
            for layer_id in self._hp_manager_layer_ids
        }
        if len(cache_groups) != 1 or len(hp_groups) != 1 or cache_groups == hp_groups:
            raise RuntimeError(
                "FP4 MLA V2 requires one full-history and one HP sliding layer group; "
                f"got cache={sorted(cache_groups)}, hp={sorted(hp_groups)}."
            )
        fp8_groups = {
            int(self._manager.impl.get_layer_group_id(layer_id))
            for layer_id in self._fp8_mla_manager_layer_ids
        }
        if self._fp8_mla_manager_layer_ids and (
            len(fp8_groups) != 1 or fp8_groups == cache_groups or fp8_groups == hp_groups
        ):
            raise RuntimeError(
                "FP8 MLA fallback layers require one lifecycle distinct from "
                f"FP4 and HP; got fp8={sorted(fp8_groups)}."
            )
        draft_groups = {
            int(self._manager.impl.get_layer_group_id(layer_id))
            for layer_id in self._draft_manager_layer_ids
        }
        if self._draft_manager_layer_ids and (
            len(draft_groups) != 1 or draft_groups & (cache_groups | hp_groups | fp8_groups)
        ):
            raise RuntimeError(
                "Draft layers require one lifecycle distinct from FP4, FP8 and HP; "
                f"got draft={sorted(draft_groups)}."
            )
        cache_roles = [Role.KEY, Role.KEY_BLOCK_SCALE]
        if not self.fp4_mla_context_only_cache:
            cache_roles.append(Role.MLA_V_SCALE)
        if self.mla_v_head_dim is not None:
            cache_roles.append(Role.MLA_V_PACKED)
        for compact_layer, layer_id in enumerate(self._cache_manager_layer_ids):
            local_layer = self._fp4_mla_compact_to_local[compact_layer]
            converters = [
                self._manager.impl.get_page_index_converter(layer_id, role) for role in cache_roles
            ]
            geometries = {
                (
                    int(converter.scale),
                    int(converter.expansion),
                    int(converter.layer_offset),
                )
                for converter in converters
            }
            if len(geometries) != 1:
                raise RuntimeError(
                    "FP4 MLA V2 canonical and derived cache roles must share "
                    f"one encoded page geometry for local layer {local_layer}; "
                    f"got {sorted(geometries)}."
                )
        num_fp4_layers = len(self._hp_manager_layer_ids)
        for compact_layer, layer_id in enumerate(self._hp_manager_layer_ids):
            local_layer = self._fp4_mla_compact_to_local[compact_layer]
            converter = self._manager.impl.get_page_index_converter(layer_id, Role.MLA_HP_TAIL)
            if (
                int(converter.scale) != num_fp4_layers
                or int(converter.expansion) != 1
                or int(converter.layer_offset) != compact_layer
            ):
                raise RuntimeError(
                    "FP4 MLA V2 HP roles require one coalesced page per local "
                    f"layer; layer {local_layer} has converter={converter}."
                )

    def get_fp4_mla_page_table_spec(self, layer_idx: Optional[int] = None) -> Fp4MlaPageTableSpec:
        local_layer = (
            self._fp4_mla_compact_to_local[0]
            if layer_idx is None
            else self._manager.layer_offsets[layer_idx]
        )
        compact_layer = self._fp4_mla_compact_layer_idx(local_layer)
        cache_layer = self._cache_manager_layer_ids[compact_layer]
        hp_layer = self._hp_manager_layer_ids[compact_layer]
        return Fp4MlaPageTableSpec(
            cache_pool_id=int(self._manager.impl.get_layer_group_id(cache_layer)),
            # copy_batch_block_offsets already applies the V2 converter scale.
            cache_page_index_scale=1,
            hp_pool_id=int(self._manager.impl.get_layer_group_id(hp_layer)),
            hp_page_index_scale=1,
        )

    def _role_encoded_page_capacity(self, layer_id: LayerId, role: DataRole) -> int:
        """Return the tensor extent needed by shared-base encoded page IDs."""
        converter = self._manager.impl.get_page_index_converter(layer_id, role)
        if converter.expansion != 1:
            raise NotImplementedError("FP4 MLA V2 roles require unexpanded page indices.")
        upper = int(self._manager.impl.get_page_index_upper_bound(layer_id, role))
        scale = int(converter.scale)
        offset = int(converter.layer_offset)
        num_slots, remainder = divmod(upper + offset, scale)
        if remainder or num_slots <= 0:
            raise RuntimeError(
                f"FP4 MLA V2 {role} has invalid page-index geometry: "
                f"upper={upper}, scale={scale}, offset={offset}."
            )
        return (num_slots - 1) * scale + 1

    def _role_view(
        self,
        layer_id: LayerId,
        role: DataRole,
        dtype: torch.dtype,
        page_shape: tuple[int, ...],
    ) -> torch.Tensor:
        cache_key = (int(layer_id), role, dtype, page_shape)
        cached = self._fp4_mla_view_cache.get(cache_key)
        if cached is not None:
            return cached
        addr = int(
            self._manager.impl.get_mem_pool_base_address(layer_id, role, PageIndexMode.SHARED)
        )
        page_capacity = self._role_encoded_page_capacity(layer_id, role)
        page_elems = math.prod(page_shape)
        expected_stride = page_elems * torch.empty((), dtype=dtype).element_size()
        actual_stride = int(self._manager.impl.get_page_stride(layer_id, role))
        if expected_stride != actual_stride:
            raise RuntimeError(
                f"FP4 MLA V2 {role} page stride mismatch: {actual_stride} != {expected_stride}."
            )
        inner_strides = []
        stride = 1
        for dim in reversed(page_shape):
            inner_strides.append(stride)
            stride *= dim
        with _persistent_cache_view_mode():
            view = convert_to_torch_tensor(
                TensorWrapper(
                    addr,
                    dtype,
                    (page_capacity, *page_shape),
                    (page_elems, *reversed(inner_strides)),
                )
            )
        self._fp4_mla_view_cache[cache_key] = view
        return view

    def get_fp4_mla_cache_buffers(
        self, layer_idx: int, kv_layout: str = "NHD"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if kv_layout != "NHD":
            raise ValueError("FP4 MLA V2 cache buffers support only NHD layout.")
        # Every attention layer looks these views up once per forward step.
        cache_key = ("cache_buffers", layer_idx)
        cached = self._fp4_mla_view_cache.get(cache_key)
        if cached is not None:
            return cached
        local_layer = self._manager.layer_offsets[layer_idx]
        compact_layer = self._fp4_mla_compact_layer_idx(local_layer)
        manager_layer = self._cache_manager_layer_ids[compact_layer]
        storage_head_dim = self._storage_head_dim(local_layer)
        with _persistent_cache_view_mode():
            kv_cache = self._role_view(
                manager_layer,
                Role.KEY,
                torch.int8,
                (self._manager.kv_factor, self._manager.tokens_per_block, 1, storage_head_dim // 2),
            ).view(torch.uint8)
        sf_cache = self._role_view(
            manager_layer,
            Role.KEY_BLOCK_SCALE,
            torch.uint8,
            (self._manager.tokens_per_block, storage_head_dim // FP4_BLOCK_SIZE),
        )
        self._fp4_mla_view_cache[cache_key] = (kv_cache, sf_cache)
        return kv_cache, sf_cache

    def get_fp4_mla_cache_pool_bases(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return K and K-scale views spanning the pages of every FP4 MLA layer.

        All FP4 layers share one coalesced page geometry, so page ``p`` of the
        ``get_fp4_mla_cache_buffers`` views of compact layer ``c`` is page
        ``p + get_mla_v_scale_page_offset(c)`` of these views.
        """
        cache_key = ("cache_pool_bases",)
        cached = self._fp4_mla_view_cache.get(cache_key)
        if cached is not None:
            return cached
        local_layer = self._fp4_mla_compact_to_local[0]
        cache_layer = self._cache_manager_layer_ids[0]
        storage_head_dim = self._storage_head_dim(local_layer)
        kv_cache = self._role_pool_base_view(
            cache_layer,
            Role.KEY,
            torch.uint8,
            (self._manager.kv_factor, self._manager.tokens_per_block, 1, storage_head_dim // 2),
        )
        sf_cache = self._role_pool_base_view(
            cache_layer,
            Role.KEY_BLOCK_SCALE,
            torch.uint8,
            (self._manager.tokens_per_block, storage_head_dim // FP4_BLOCK_SIZE),
        )
        for compact_layer, local_layer in enumerate(self._fp4_mla_compact_to_local):
            layer_kv, layer_sf = self.get_fp4_mla_cache_buffers(
                self._manager.pp_layers[local_layer]
            )
            page_offset = self.get_mla_v_scale_page_offset(compact_layer)
            expected_kv_ptr = kv_cache.data_ptr() + page_offset * kv_cache.stride(0)
            expected_sf_ptr = sf_cache.data_ptr() + page_offset * sf_cache.stride(0)
            if (layer_kv.data_ptr(), layer_sf.data_ptr()) != (expected_kv_ptr, expected_sf_ptr):
                raise RuntimeError(
                    "FP4 MLA V2 cache views of local layer "
                    f"{local_layer} do not match their pool base and page offset {page_offset}."
                )
        self._fp4_mla_view_cache[cache_key] = (kv_cache, sf_cache)
        return kv_cache, sf_cache

    def _iter_fp4_mla_physical_pool_views(self) -> Iterable[torch.Tensor]:
        """Yield page-major views spanning each physical MLA pool once."""
        local_layer = self._fp4_mla_compact_to_local[0]
        cache_layer = self._cache_manager_layer_ids[0]
        storage_head_dim = self._storage_head_dim(local_layer)
        yield self._role_pool_base_view(
            cache_layer,
            Role.KEY,
            torch.uint8,
            (self._manager.kv_factor, self._manager.tokens_per_block, 1, storage_head_dim // 2),
        )
        yield self._role_pool_base_view(
            cache_layer,
            Role.KEY_BLOCK_SCALE,
            torch.uint8,
            (self._manager.tokens_per_block, storage_head_dim // FP4_BLOCK_SIZE),
        ).view(torch.float8_e4m3fn)
        if self._fp8_mla_manager_layer_ids:
            fp8_layer = self._fp8_mla_manager_layer_ids[0]
            fp8_local_layer = int(fp8_layer)
            yield self._role_pool_base_view(
                fp8_layer,
                Role.KEY,
                torch.float8_e4m3fn,
                (
                    self._manager.kv_factor,
                    self._manager.tokens_per_block,
                    self._manager.num_kv_heads_per_layer[fp8_local_layer],
                    self._manager.head_dim_per_layer[fp8_local_layer],
                ),
            )
        if self._draft_manager_layer_ids:
            # Every appended draft layer lives in this one single-buffer pool.
            draft_layer = self._draft_manager_layer_ids[0]
            draft_spec = self._manager._appended_draft_layer_specs[int(draft_layer)]
            yield self._role_pool_base_view(
                draft_layer,
                Role.KEY,
                binding_to_torch_dtype(draft_spec.dtype),
                (
                    draft_spec.kv_factor,
                    self._manager.tokens_per_block,
                    draft_spec.num_kv_heads,
                    draft_spec.head_dim,
                ),
            )
        if not self.fp4_mla_context_only_cache:
            yield self.get_mla_v_scale_pool_base().view(torch.float8_e4m3fn)
        if self.mla_v_head_dim is not None:
            yield self._role_pool_base_view(
                cache_layer,
                Role.MLA_V_PACKED,
                torch.uint8,
                (self.mla_v_head_dim, self._manager.tokens_per_block // 2),
            )
        yield self._role_pool_base_view(
            self._hp_manager_layer_ids[0],
            Role.MLA_HP_TAIL,
            torch.bfloat16,
            (
                1,
                self._fp4_mla_hp_pool_size * self._manager.head_dim_per_layer[local_layer],
            ),
        )

    def _all_layer_role_view(
        self,
        layer_ids: list[LayerId],
        role: DataRole,
        dtype: torch.dtype,
        page_shape: tuple[int, ...],
    ) -> torch.Tensor:
        cache_key = ("all_layers", tuple(map(int, layer_ids)), role, dtype, page_shape)
        cached = self._fp4_mla_view_cache.get(cache_key)
        if cached is not None:
            return cached
        first_layer = layer_ids[0]
        converter = self._manager.impl.get_page_index_converter(first_layer, role)
        if int(converter.scale) != len(layer_ids):
            raise RuntimeError(
                f"FP4 MLA V2 {role} requires one coalesced buffer per local layer, "
                f"got scale={converter.scale}, layers={len(layer_ids)}."
            )
        page_upper = int(self._manager.impl.get_page_index_upper_bound(first_layer, role))
        physical_page_capacity = page_upper - (len(layer_ids) - 1)
        if physical_page_capacity <= 0:
            raise RuntimeError(f"FP4 MLA V2 {role} pool has no physical pages.")
        page_elems = math.prod(page_shape)
        inner_strides = []
        stride = 1
        for dim in reversed(page_shape):
            inner_strides.append(stride)
            stride *= dim
        addr = int(
            self._manager.impl.get_mem_pool_base_address(first_layer, role, PageIndexMode.SHARED)
        )
        with _persistent_cache_view_mode():
            view = convert_to_torch_tensor(
                TensorWrapper(
                    addr,
                    dtype,
                    (len(layer_ids), physical_page_capacity, *page_shape),
                    (
                        page_elems,
                        page_elems,
                        *reversed(inner_strides),
                    ),
                )
            )
        self._fp4_mla_view_cache[cache_key] = view
        return view

    def _role_pool_base_view(
        self,
        layer_id: LayerId,
        role: DataRole,
        dtype: torch.dtype,
        page_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Return a contiguous view spanning a role's complete coalesced pool."""
        cache_key = ("pool_base", int(layer_id), role, dtype, page_shape)
        cached = self._fp4_mla_view_cache.get(cache_key)
        if cached is not None:
            return cached
        converter = self._manager.impl.get_page_index_converter(layer_id, role)
        if int(converter.layer_offset) != 0 or int(converter.expansion) != 1:
            raise RuntimeError(f"FP4 MLA V2 {role} pool base must use the first unexpanded buffer.")
        addr = int(
            self._manager.impl.get_mem_pool_base_address(layer_id, role, PageIndexMode.SHARED)
        )
        page_upper = int(self._manager.impl.get_page_index_upper_bound(layer_id, role))
        page_elems = math.prod(page_shape)
        expected_stride = page_elems * torch.empty((), dtype=dtype).element_size()
        actual_stride = int(self._manager.impl.get_page_stride(layer_id, role))
        if expected_stride != actual_stride:
            raise RuntimeError(
                f"FP4 MLA V2 {role} page stride mismatch: {actual_stride} != {expected_stride}."
            )
        with _persistent_cache_view_mode():
            pool = convert_to_torch_tensor(TensorWrapper(addr, dtype, (page_upper, *page_shape)))
        self._fp4_mla_view_cache[cache_key] = pool
        return pool

    def get_mla_v_scale_pool(self) -> Optional[torch.Tensor]:
        if self.fp4_mla_context_only_cache:
            return None
        return self._all_layer_role_view(
            self._cache_manager_layer_ids,
            Role.MLA_V_SCALE,
            torch.uint8,
            (self._v_scale_bytes_per_page(),),
        ).view(torch.float8_e4m3fn)

    def get_mla_v_scale_pool_base(self) -> Optional[torch.Tensor]:
        if self.fp4_mla_context_only_cache:
            return None
        return self._role_pool_base_view(
            self._cache_manager_layer_ids[0],
            Role.MLA_V_SCALE,
            torch.uint8,
            (self._v_scale_bytes_per_page(),),
        )

    def get_mla_v_scale_page_offset(self, local_layer: int) -> int:
        if not 0 <= local_layer < len(self._cache_manager_layer_ids):
            raise IndexError(f"Invalid compact FP4 MLA layer {local_layer}.")
        layer_id = self._cache_manager_layer_ids[local_layer]
        return int(
            self._manager.impl.get_page_index_converter(layer_id, Role.MLA_V_SCALE).layer_offset
        )

    def get_mla_v_packed_pool(self, local_layer: int) -> Optional[torch.Tensor]:
        if self.mla_v_head_dim is None:
            return None
        cache_key = ("v_packed_pool", local_layer)
        cached = self._fp4_mla_view_cache.get(cache_key)
        if cached is not None:
            return cached
        if not 0 <= local_layer < len(self._cache_manager_layer_ids):
            raise IndexError(f"Invalid compact FP4 MLA layer {local_layer}.")
        pool = self._role_view(
            self._cache_manager_layer_ids[local_layer],
            Role.MLA_V_PACKED,
            torch.uint8,
            (self.mla_v_head_dim, self._manager.tokens_per_block // 2),
        )
        with _persistent_cache_view_mode():
            pool = pool.reshape(pool.shape[0] * pool.shape[1], pool.shape[2])
        self._fp4_mla_view_cache[cache_key] = pool
        return pool

    def get_mla_v_packed_pool_base(self) -> Optional[torch.Tensor]:
        if self.mla_v_head_dim is None:
            return None
        pool = self._role_pool_base_view(
            self._cache_manager_layer_ids[0],
            Role.MLA_V_PACKED,
            torch.uint8,
            (self.mla_v_head_dim, self._manager.tokens_per_block // 2),
        )
        return pool.reshape(pool.shape[0] * pool.shape[1], pool.shape[2])

    def get_mla_v_packed_page_offset(self, local_layer: int) -> int:
        cache_key = ("v_packed_page_offset", local_layer)
        cached = self._fp4_mla_view_cache.get(cache_key)
        if cached is not None:
            return cached
        if not 0 <= local_layer < len(self._cache_manager_layer_ids):
            raise IndexError(f"Invalid compact FP4 MLA layer {local_layer}.")
        layer_id = self._cache_manager_layer_ids[local_layer]
        offset = int(
            self._manager.impl.get_page_index_converter(layer_id, Role.MLA_V_PACKED).layer_offset
        )
        self._fp4_mla_view_cache[cache_key] = offset
        return offset

    def get_fp4_mla_hp_pool(self) -> torch.Tensor:
        return self._all_layer_role_view(
            self._hp_manager_layer_ids,
            Role.MLA_HP_TAIL,
            torch.bfloat16,
            (
                1,
                self._fp4_mla_hp_pool_size
                * self._manager.head_dim_per_layer[self._fp4_mla_compact_to_local[0]],
            ),
        ).permute(1, 0, 2, 3)

    def _fp4_mla_disagg_role_layouts(self) -> dict[DataRole, RoleLayout]:
        """Declare the HP role as a ring of BF16 latent tokens.

        A token at global position ``p`` sits in ring slot ``p % ring_size``,
        and the ring size depends on the worker's speculative rewind, so a
        context and a generation worker usually size it differently.
        """
        head_dims = {
            int(self._manager.head_dim_per_layer[local_layer])
            for local_layer in self._fp4_mla_compact_to_local
        }
        if len(head_dims) != 1:
            raise ValueError(
                f"FP4 MLA HP ring needs one head dimension across layers, got {sorted(head_dims)}."
            )
        slot_bytes = head_dims.pop() * torch.empty((), dtype=torch.bfloat16).element_size()
        return {Role.MLA_HP_TAIL: RoleLayout(ring_slot_bytes=slot_bytes)}

    def rebuild_disagg_imported_cache(self, requests: Iterable["LlmRequest"]) -> None:
        """Rebuild the derived V sidecars of requests whose prompt KV was imported."""
        from .v_cache import rebuild_fp4_mla_disagg_imported_cache

        requests = list(requests)
        rebuild_fp4_mla_disagg_imported_cache(
            self._manager,
            [request.py_request_id for request in requests],
            [request.prompt_len for request in requests],
        )

    def _fp4_mla_disagg_global_layer_id(self, internal_layer: int) -> Optional[int]:
        """Collision-free transfer ID of an MLA cache or HP-ring layer.

        A model layer's latent cache (FP4 or FP8 fallback) maps to ``2 * L``
        and its HP-ring virtual layer to ``2 * L + 1``, where ``L`` is the
        global model layer; an appended draft layer ``D`` maps to ``2 * D``.
        Returns ``None`` for any other internal layer.
        """
        internal_layer = int(internal_layer)
        if internal_layer in self._manager._appended_draft_layer_specs:
            return 2 * int(self._manager.pp_layers[internal_layer])
        for compact_layer, layer_id in enumerate(self._cache_manager_layer_ids):
            if int(layer_id) == internal_layer:
                local_layer = self._fp4_mla_compact_to_local[compact_layer]
                return 2 * int(self._manager.pp_layers[local_layer])
        if any(int(layer_id) == internal_layer for layer_id in self._fp8_mla_manager_layer_ids):
            return 2 * int(self._manager.pp_layers[internal_layer])
        for compact_layer, layer_id in enumerate(self._hp_manager_layer_ids):
            if int(layer_id) == internal_layer:
                local_layer = self._fp4_mla_compact_to_local[compact_layer]
                return 2 * int(self._manager.pp_layers[local_layer]) + 1
        return None

    def runtime_cache_size_layer_components(self) -> tuple[list[int], list[Optional[int]]]:
        fp4_local_layers = self._fp4_mla_local_layer_indices()
        fp8_local_layers = self._fp8_mla_local_layer_indices()
        sizes = [
            self._manager.get_layer_bytes_per_token(local_layer, Role.ALL)
            for local_layer in fp4_local_layers
        ]
        windows: list[Optional[int]] = [None] * len(fp4_local_layers)
        sizes.extend(
            self._manager.get_layer_bytes_per_token(local_layer, Role.ALL)
            for local_layer in fp8_local_layers
        )
        windows.extend([None] * len(fp8_local_layers))
        sizes.extend(
            self._manager.get_layer_bytes_per_token(local_layer, Role.MLA_HP_TAIL)
            for local_layer in fp4_local_layers
        )
        windows.extend([self._fp4_mla_hp_pool_size] * len(fp4_local_layers))
        draft_layers = self._manager._appended_draft_layer_specs.values()
        sizes.extend(layer.bytes_per_token() for layer in draft_layers)
        windows.extend([None] * len(draft_layers))
        return sizes, windows

    def generation_request_capacity(self) -> int:
        # Pipeline stages retain different in-flight microbatches, and every
        # resident sequence needs one native HP ring page per local layer.
        # GEN slot preemption pauses residents in place, so the paused cap
        # stays resident too.
        return (
            self._manager.max_batch_size * self._manager.mapping.pp_size
            + self._manager.gen_preemption_headroom
        )


__all__ = [
    "Fp4MlaCacheLayout",
    "Fp4MlaPageTableSpec",
]
