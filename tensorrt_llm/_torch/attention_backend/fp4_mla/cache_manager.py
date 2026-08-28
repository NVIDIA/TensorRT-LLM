# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KV Cache Manager V2 specialization for NVFP4 MLA cache state."""

import math
from dataclasses import dataclass, replace
from typing import List, Optional

import torch

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType, KVCacheManager
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor, prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    DataRole,
    LayerId,
    PageIndexMode,
)

from . import (
    _FP4_MLA_CUTEDSL_BACKEND,
    _FP4_MLA_K_RESIDUAL_BACKENDS,
    FP4_BLOCK_SIZE,
    FP4_MLA_K_RESIDUAL_DIM,
    FP4_MLA_TOKENS_PER_BLOCK,
    HP_BLOCK_SIZE,
    _fp4_mla_attention_backend,
    _fp4_mla_cutedsl_fused_v_transpose_enabled,
    get_fp4_mla_v_scale_pool_size,
)


@dataclass(frozen=True)
class Fp4MlaPageTableSpec:
    """Stable page-table conversion contract consumed by FP4 MLA kernels."""

    cache_pool_id: int
    cache_page_index_scale: int
    hp_pool_id: int
    hp_page_index_scale: int
    hp_is_paged: bool = True


class Fp4MlaKVCacheManagerV2(KVCacheManagerV2):
    """V2 manager for canonical FP4 MLA pages and compact HP sequence state.

    K, K block scales, V scales, and optional packed V share one full-history
    lifecycle. Each model layer also has a virtual sliding layer whose page is
    a compact ``16 + max_rewind`` BF16 ring. The HP role therefore follows V2
    allocation, rewind, and prefix lifecycles without storing BF16 values for
    the full sequence.
    """

    def __init__(self, *args, **kwargs) -> None:
        kv_cache_config = args[0] if args else kwargs.get("kv_cache_config")
        dtype = kwargs.get("dtype", DataType.HALF)
        kv_cache_type = args[1] if len(args) > 1 else kwargs.get("kv_cache_type")
        tokens_per_block = kwargs.get("tokens_per_block")
        head_dim = kwargs.get("head_dim")
        if dtype != DataType.NVFP4 or kv_cache_type != CacheTypeCpp.SELFKONLY:
            raise ValueError("Fp4MlaKVCacheManagerV2 requires NVFP4 SELFKONLY cache storage.")
        if kv_cache_config is None or kv_cache_config.dtype not in ("auto", "nvfp4"):
            raise ValueError("Fp4MlaKVCacheManagerV2 requires KvCacheConfig.dtype='nvfp4'.")
        if kv_cache_config.enable_swa_scratch_reuse:
            raise ValueError("FP4 MLA V2 native HP pages do not support enable_swa_scratch_reuse.")

        config_updates = {}
        if kv_cache_config.dtype == "auto":
            config_updates["dtype"] = "nvfp4"
        if kv_cache_config.enable_partial_reuse:
            # FP4 MLA can resume only on a complete 16-token quantization
            # tile. V2 partial reuse may return any token boundary, so keep
            # full-block reuse while disabling only partial-block matches.
            config_updates["enable_partial_reuse"] = False
        if config_updates:
            kv_cache_config = kv_cache_config.model_copy(update=config_updates)
            if args:
                args = (kv_cache_config, *args[1:])
            else:
                kwargs["kv_cache_config"] = kv_cache_config
        if "enable_partial_reuse" in config_updates:
            logger.info(
                "FP4 MLA KV cache manager V2 disables partial block reuse "
                "to preserve 16-token cache-update alignment."
            )
        if tokens_per_block != FP4_MLA_TOKENS_PER_BLOCK:
            raise ValueError(
                "FP4 MLA V2 requires tokens_per_block="
                f"{FP4_MLA_TOKENS_PER_BLOCK}, got {tokens_per_block}."
            )
        if not isinstance(head_dim, int) or head_dim <= FP4_MLA_K_RESIDUAL_DIM:
            raise ValueError(f"FP4 MLA V2 requires a positive scalar MLA head_dim, got {head_dim}.")

        pretrained_config = kwargs.get("pretrained_config")
        self.mla_v_scale_head_dim = int(
            getattr(pretrained_config, "kv_lora_rank", head_dim - FP4_MLA_K_RESIDUAL_DIM)
        )
        if not 0 < self.mla_v_scale_head_dim < head_dim:
            raise ValueError(
                "FP4 MLA V2 requires kv_lora_rank in (0, head_dim), got "
                f"{self.mla_v_scale_head_dim} and {head_dim}."
            )

        self._fp4_mla_storage_backend = _fp4_mla_attention_backend()
        if self._fp4_mla_storage_backend not in _FP4_MLA_K_RESIDUAL_BACKENDS:
            raise ValueError(
                "Fp4MlaKVCacheManagerV2 supports only the triton and cutedsl "
                f"backends, got {self._fp4_mla_storage_backend!r}."
            )
        self.fp4_mla_k_residual_dim = (
            FP4_MLA_K_RESIDUAL_DIM
            if self._fp4_mla_storage_backend in _FP4_MLA_K_RESIDUAL_BACKENDS
            else 0
        )
        self.mla_v_head_dim = (
            self.mla_v_scale_head_dim
            if self._fp4_mla_storage_backend == _FP4_MLA_CUTEDSL_BACKEND
            and not _fp4_mla_cutedsl_fused_v_transpose_enabled()
            else None
        )
        spec_config = kwargs.get("spec_config")
        max_rewind_len = int(spec_config.tokens_per_gen_step - 1) if spec_config else 0
        self._fp4_mla_hp_pool_size = HP_BLOCK_SIZE + max_rewind_len
        self._fp4_mla_view_cache: dict[tuple, torch.Tensor] = {}

        super().__init__(*args, **kwargs)

        self._validate_fp4_mla_layer_groups()
        # Partial pages leave unused V-scale tiles untouched while the
        # fixed-width PV path can load the complete scale page. Match V1's
        # deterministic initialization before any warmup or graph capture.
        with torch.cuda.stream(self._stream):
            self.get_mla_v_scale_pool_base().zero_()
        self._stream.synchronize()

    @property
    def blocks_in_primary_pool(self) -> int:
        return self._role_encoded_page_capacity(self._cache_manager_layer_ids[0], Role.KEY)

    def _storage_head_dim(self, local_layer_idx: int) -> int:
        return self.head_dim_per_layer[local_layer_idx] + self.fp4_mla_k_residual_dim

    def _v_scale_bytes_per_page(self) -> int:
        return get_fp4_mla_v_scale_pool_size(self.mla_v_scale_head_dim, self.tokens_per_block)

    def _v_packed_bytes_per_page(self) -> int:
        if self.mla_v_head_dim is None:
            return 0
        return self.mla_v_head_dim * (self.tokens_per_block // 2)

    def _hp_bytes_per_page(self, local_layer_idx: int) -> int:
        return (
            self._fp4_mla_hp_pool_size
            * self.head_dim_per_layer[local_layer_idx]
            * torch.empty((), dtype=torch.bfloat16).element_size()
        )

    def get_layer_bytes_per_token(self, local_layer_idx: int, data_role: DataRole):
        storage_head_dim = self._storage_head_dim(local_layer_idx)
        role_sizes = {
            Role.KEY: math.ceil(storage_head_dim / 2),
            Role.KEY_BLOCK_SCALE: math.ceil(storage_head_dim / FP4_BLOCK_SIZE),
            Role.MLA_V_SCALE: math.ceil(self._v_scale_bytes_per_page() / self.tokens_per_block),
            Role.MLA_V_PACKED: math.ceil(self._v_packed_bytes_per_page() / self.tokens_per_block),
            Role.MLA_HP_TAIL: math.ceil(
                self._hp_bytes_per_page(local_layer_idx) / self.tokens_per_block
            ),
        }
        if data_role == Role.ALL:
            roles = [Role.KEY, Role.KEY_BLOCK_SCALE, Role.MLA_V_SCALE]
            if self.mla_v_head_dim is not None:
                roles.append(Role.MLA_V_PACKED)
            return sum(role_sizes[role] for role in roles)
        if data_role not in role_sizes or role_sizes[data_role] <= 0:
            raise ValueError(f"Invalid FP4 MLA V2 data role: {data_role}")
        return role_sizes[data_role]

    def _extra_buffers_per_layer(
        self, *, tokens_per_block: int
    ) -> Optional[dict[int, List[BufferConfig]]]:
        result = {}
        for local_layer in range(self.num_local_layers):
            buffers = [
                BufferConfig(
                    role=Role.MLA_V_SCALE,
                    size=self._v_scale_bytes_per_page(),
                )
            ]
            if self.mla_v_head_dim is not None:
                buffers.append(
                    BufferConfig(
                        role=Role.MLA_V_PACKED,
                        size=self._v_packed_bytes_per_page(),
                    )
                )
            result[local_layer] = buffers
        return result

    def _build_cache_config(self, config):
        cache_layers = list(config.layers)
        self._cache_manager_layer_ids = [LayerId(i) for i in range(self.num_local_layers)]
        self._hp_manager_layer_ids = []
        for local_layer in range(self.num_local_layers):
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
                )
            )
        return replace(config, layers=cache_layers)

    def _prepare_page_table_tensor(self, index_mapper_capacity: int) -> None:
        cache_layer = self._cache_manager_layer_ids[0]
        hp_layer = self._hp_manager_layer_ids[0]
        cache_pool_id = int(self.impl.get_layer_group_id(cache_layer))
        hp_pool_id = int(self.impl.get_layer_group_id(hp_layer))
        if cache_pool_id == hp_pool_id:
            raise RuntimeError("FP4 MLA full-history and HP state must use distinct lifecycles.")

        num_pools = len(self.impl.layer_grouping)
        pool_pointers = [[[0, 0], [0, 0]] for _ in range(num_pools)]
        pool_pointers[cache_pool_id] = [
            [
                int(self.impl.get_mem_pool_base_address(cache_layer, Role.KEY)),
                int(self.impl.get_mem_pool_base_address(cache_layer, Role.KEY_BLOCK_SCALE)),
            ],
            [0, 0],
        ]
        pool_pointers[hp_pool_id] = [
            [int(self.impl.get_mem_pool_base_address(hp_layer, Role.MLA_HP_TAIL)), 0],
            [0, 0],
        ]
        self.kv_cache_pool_pointers = torch.tensor(
            pool_pointers,
            dtype=torch.int64,
            device="cpu",
            pin_memory=prefer_pinned(),
        )

        mapping = []
        for layer_id in self._cache_manager_layer_ids:
            converter = self.impl.get_page_index_converter(layer_id, Role.KEY)
            mapping.append([cache_pool_id, int(converter.layer_offset)])
        self.kv_cache_pool_mapping = torch.tensor(
            mapping,
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )

        self.index_scales = torch.ones(
            num_pools, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
        )
        self.index_scales[cache_pool_id] = int(
            self.impl.get_page_index_converter(cache_layer, Role.KEY).scale
        )
        self.index_scales[hp_pool_id] = int(
            self.impl.get_page_index_converter(hp_layer, Role.MLA_HP_TAIL).scale
        )
        self.kv_offset = torch.zeros_like(self.index_scales)
        self._index_scale_ints = self.index_scales.tolist()
        self.num_attention_op_pools = num_pools
        self.host_kv_cache_block_offsets = torch.zeros(
            num_pools,
            index_mapper_capacity * self.max_beam_width,
            2,
            self.max_blocks_per_seq,
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )

    def _validate_fp4_mla_layer_groups(self) -> None:
        cache_groups = {
            int(self.impl.get_layer_group_id(layer_id))
            for layer_id in self._cache_manager_layer_ids
        }
        hp_groups = {
            int(self.impl.get_layer_group_id(layer_id)) for layer_id in self._hp_manager_layer_ids
        }
        if len(cache_groups) != 1 or len(hp_groups) != 1 or cache_groups == hp_groups:
            raise RuntimeError(
                "FP4 MLA V2 requires one full-history and one HP sliding layer group; "
                f"got cache={sorted(cache_groups)}, hp={sorted(hp_groups)}."
            )
        cache_roles = [Role.KEY, Role.KEY_BLOCK_SCALE, Role.MLA_V_SCALE]
        if self.mla_v_head_dim is not None:
            cache_roles.append(Role.MLA_V_PACKED)
        for local_layer, layer_id in enumerate(self._cache_manager_layer_ids):
            converters = [
                self.impl.get_page_index_converter(layer_id, role) for role in cache_roles
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
        for local_layer, layer_id in enumerate(self._hp_manager_layer_ids):
            converter = self.impl.get_page_index_converter(layer_id, Role.MLA_HP_TAIL)
            if (
                int(converter.scale) != self.num_local_layers
                or int(converter.expansion) != 1
                or int(converter.layer_offset) != local_layer
            ):
                raise RuntimeError(
                    "FP4 MLA V2 HP roles require one coalesced page per local "
                    f"layer; layer {local_layer} has converter={converter}."
                )

    def get_fp4_mla_page_table_spec(self, layer_idx: Optional[int] = None) -> Fp4MlaPageTableSpec:
        local_layer = 0 if layer_idx is None else self.layer_offsets[layer_idx]
        cache_layer = self._cache_manager_layer_ids[local_layer]
        hp_layer = self._hp_manager_layer_ids[local_layer]
        return Fp4MlaPageTableSpec(
            cache_pool_id=int(self.impl.get_layer_group_id(cache_layer)),
            # copy_batch_block_offsets already applies the V2 converter scale.
            cache_page_index_scale=1,
            hp_pool_id=int(self.impl.get_layer_group_id(hp_layer)),
            hp_page_index_scale=1,
        )

    def _role_encoded_page_capacity(self, layer_id: LayerId, role: DataRole) -> int:
        """Return the tensor extent needed by shared-base encoded page IDs."""
        converter = self.impl.get_page_index_converter(layer_id, role)
        if converter.expansion != 1:
            raise NotImplementedError("FP4 MLA V2 roles require unexpanded page indices.")
        upper = int(self.impl.get_page_index_upper_bound(layer_id, role))
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
        addr = int(self.impl.get_mem_pool_base_address(layer_id, role, PageIndexMode.SHARED))
        page_capacity = self._role_encoded_page_capacity(layer_id, role)
        page_elems = math.prod(page_shape)
        expected_stride = page_elems * torch.empty((), dtype=dtype).element_size()
        actual_stride = int(self.impl.get_page_stride(layer_id, role))
        if expected_stride != actual_stride:
            raise RuntimeError(
                f"FP4 MLA V2 {role} page stride mismatch: {actual_stride} != {expected_stride}."
            )
        inner_strides = []
        stride = 1
        for dim in reversed(page_shape):
            inner_strides.append(stride)
            stride *= dim
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
        local_layer = self.layer_offsets[layer_idx]
        manager_layer = self._cache_manager_layer_ids[local_layer]
        storage_head_dim = self._storage_head_dim(local_layer)
        kv_cache = self._role_view(
            manager_layer,
            Role.KEY,
            torch.int8,
            (self.kv_factor, self.tokens_per_block, 1, storage_head_dim // 2),
        ).view(torch.uint8)
        sf_cache = self._role_view(
            manager_layer,
            Role.KEY_BLOCK_SCALE,
            torch.uint8,
            (self.tokens_per_block, storage_head_dim // FP4_BLOCK_SIZE),
        )
        return kv_cache, sf_cache

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
        converter = self.impl.get_page_index_converter(first_layer, role)
        if int(converter.scale) != len(layer_ids):
            raise RuntimeError(
                f"FP4 MLA V2 {role} requires one coalesced buffer per local layer, "
                f"got scale={converter.scale}, layers={len(layer_ids)}."
            )
        page_upper = int(self.impl.get_page_index_upper_bound(first_layer, role))
        physical_page_capacity = page_upper - (len(layer_ids) - 1)
        if physical_page_capacity <= 0:
            raise RuntimeError(f"FP4 MLA V2 {role} pool has no physical pages.")
        page_elems = math.prod(page_shape)
        inner_strides = []
        stride = 1
        for dim in reversed(page_shape):
            inner_strides.append(stride)
            stride *= dim
        addr = int(self.impl.get_mem_pool_base_address(first_layer, role, PageIndexMode.SHARED))
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
        converter = self.impl.get_page_index_converter(layer_id, role)
        if int(converter.layer_offset) != 0 or int(converter.expansion) != 1:
            raise RuntimeError(f"FP4 MLA V2 {role} pool base must use the first unexpanded buffer.")
        addr = int(self.impl.get_mem_pool_base_address(layer_id, role, PageIndexMode.SHARED))
        page_upper = int(self.impl.get_page_index_upper_bound(layer_id, role))
        page_elems = math.prod(page_shape)
        expected_stride = page_elems * torch.empty((), dtype=dtype).element_size()
        actual_stride = int(self.impl.get_page_stride(layer_id, role))
        if expected_stride != actual_stride:
            raise RuntimeError(
                f"FP4 MLA V2 {role} page stride mismatch: {actual_stride} != {expected_stride}."
            )
        pool = convert_to_torch_tensor(TensorWrapper(addr, dtype, (page_upper, *page_shape)))
        self._fp4_mla_view_cache[cache_key] = pool
        return pool

    def get_mla_v_scale_pool(self) -> torch.Tensor:
        return self._all_layer_role_view(
            self._cache_manager_layer_ids,
            Role.MLA_V_SCALE,
            torch.uint8,
            (self._v_scale_bytes_per_page(),),
        ).view(torch.float8_e4m3fn)

    def get_mla_v_scale_pool_base(self) -> torch.Tensor:
        return self._role_pool_base_view(
            self._cache_manager_layer_ids[0],
            Role.MLA_V_SCALE,
            torch.uint8,
            (self._v_scale_bytes_per_page(),),
        )

    def get_mla_v_scale_page_offset(self, local_layer: int) -> int:
        layer_id = self._cache_manager_layer_ids[local_layer]
        return int(self.impl.get_page_index_converter(layer_id, Role.MLA_V_SCALE).layer_offset)

    def get_mla_v_packed_pool(self, local_layer: int) -> Optional[torch.Tensor]:
        if self.mla_v_head_dim is None:
            return None
        if not 0 <= local_layer < self.num_local_layers:
            raise IndexError(f"Invalid FP4 MLA local layer {local_layer}.")
        pool = self._role_view(
            self._cache_manager_layer_ids[local_layer],
            Role.MLA_V_PACKED,
            torch.uint8,
            (self.mla_v_head_dim, self.tokens_per_block // 2),
        )
        return pool.reshape(pool.shape[0] * pool.shape[1], pool.shape[2])

    def get_mla_v_packed_pool_base(self) -> Optional[torch.Tensor]:
        if self.mla_v_head_dim is None:
            return None
        pool = self._role_pool_base_view(
            self._cache_manager_layer_ids[0],
            Role.MLA_V_PACKED,
            torch.uint8,
            (self.mla_v_head_dim, self.tokens_per_block // 2),
        )
        return pool.reshape(pool.shape[0] * pool.shape[1], pool.shape[2])

    def get_mla_v_packed_page_offset(self, local_layer: int) -> int:
        layer_id = self._cache_manager_layer_ids[local_layer]
        return int(self.impl.get_page_index_converter(layer_id, Role.MLA_V_PACKED).layer_offset)

    def get_fp4_mla_hp_pool(self) -> torch.Tensor:
        return self._all_layer_role_view(
            self._hp_manager_layer_ids,
            Role.MLA_HP_TAIL,
            torch.bfloat16,
            (1, self._fp4_mla_hp_pool_size * self.head_dim_per_layer[0]),
        ).permute(1, 0, 2, 3)

    def _get_runtime_cache_size_layer_components(self):
        sizes = [
            self.get_layer_bytes_per_token(local_layer, Role.ALL)
            for local_layer in range(self.num_local_layers)
        ]
        windows: list[Optional[int]] = [None] * self.num_local_layers
        sizes.extend(
            self.get_layer_bytes_per_token(local_layer, Role.MLA_HP_TAIL)
            for local_layer in range(self.num_local_layers)
        )
        windows.extend([self._fp4_mla_hp_pool_size] * self.num_local_layers)
        return sizes, windows

    def _get_generation_request_capacity(self) -> int:
        # Pipeline stages retain different in-flight microbatches, and every
        # resident sequence needs one native HP ring page per local layer.
        return self.max_batch_size * self.mapping.pp_size

    def get_cache_bytes_per_token(self) -> int:
        sizes, _ = self._get_runtime_cache_size_layer_components()
        return sum(sizes)

    @staticmethod
    def get_cache_size_per_token(model_config, mapping, num_layers=None, **kwargs):
        config = model_config.pretrained_config
        logical_head_dim = int(config.kv_lora_rank + config.qk_rope_head_dim)
        backend = _fp4_mla_attention_backend()
        if backend not in _FP4_MLA_K_RESIDUAL_BACKENDS:
            raise ValueError(
                "Fp4MlaKVCacheManagerV2 supports only the triton and cutedsl "
                f"backends, got {backend!r}."
            )
        residual = FP4_MLA_K_RESIDUAL_DIM if backend in _FP4_MLA_K_RESIDUAL_BACKENDS else 0
        storage_head_dim = logical_head_dim + residual
        tokens_per_block = int(kwargs["tokens_per_block"])
        v_scale_page = get_fp4_mla_v_scale_pool_size(config.kv_lora_rank, tokens_per_block)
        per_layer = (
            math.ceil(storage_head_dim / 2)
            + math.ceil(storage_head_dim / FP4_BLOCK_SIZE)
            + math.ceil(v_scale_page / tokens_per_block)
        )
        if backend == _FP4_MLA_CUTEDSL_BACKEND and not _fp4_mla_cutedsl_fused_v_transpose_enabled():
            per_layer += config.kv_lora_rank // 2
        local_layers = KVCacheManager._resolve_num_attention_layers(
            model_config, mapping, num_layers
        )
        spec_config = kwargs.get("spec_config")
        rewind = int(spec_config.tokens_per_gen_step - 1) if spec_config else 0
        hp_bytes = (HP_BLOCK_SIZE + rewind) * logical_head_dim * 2 * local_layers
        max_batch_size = int(kwargs.get("max_batch_size") or 0)
        return per_layer * local_layers, hp_bytes * max_batch_size * mapping.pp_size


__all__ = ["Fp4MlaKVCacheManagerV2", "Fp4MlaPageTableSpec"]
