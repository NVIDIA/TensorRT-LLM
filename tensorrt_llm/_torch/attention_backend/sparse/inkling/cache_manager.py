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
"""Inkling's KV cache manager: paged KV plus the short-conv state pool.

The short-conv state is registered with V2 as SSM layers rather than allocated on
the side: ``SsmLayerConfig`` is the framework's container for per-request
fixed-size state, and V2 sizes it with the right rule (one dedicated block per
request, never shared), so the pool's bytes enter V2's quota and come from the
pool allocator at stable device addresses. V2 owns the memory but not the
request-to-slot mapping, so :class:`InklingConvStateCache` still assigns rows, as
Mamba's manager does.

The SSM layers are appended after the attention layers rather than interleaved,
which leaves every attention ``layer_id`` -- and therefore the paged-KV
addressing -- untouched.
"""

from dataclasses import replace
from typing import List

import torch

from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    BatchDesc,
    BufferConfig,
    DataRole,
    KVCacheDesc,
    LayerId,
    PageIndexMode,
    SsmLayerConfig,
)

from .....logger import logger
from ....pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from .conv_state import CONV_ROLES, InklingConvState, InklingConvStateCache


def _resolve_conv_dtype(pretrained_config) -> torch.dtype:
    """The compute dtype the short-conv pool holds.

    Not the manager's ``dtype`` argument: that is the KV cache dtype, which is
    nvfp4/fp8 on quantized releases while this pool holds unquantized pre-conv
    activations. An unresolvable value raises rather than defaulting, so a
    wrong-dtype pool cannot surface far from its cause.
    """
    config = getattr(pretrained_config, "text_config", pretrained_config)
    dtype = getattr(config, "torch_dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        resolved = getattr(torch, dtype, None)
        if isinstance(resolved, torch.dtype):
            return resolved
    raise ValueError(
        f"Inkling short-conv pool needs the model's compute dtype, but "
        f"torch_dtype={dtype!r} on {type(config).__name__} is not a torch dtype"
    )


class InklingHybridCacheManager(KVCacheManagerV2):
    """Paged KV (V2, per-layer geometry) + the short-conv state pool.

    Folding the pool into the manager -- ``CppMambaHybridCacheManager``'s shape
    -- lets it reach the model through ``attn_metadata.kv_cache_manager`` and be
    released by the manager's own ``free_resources``, so conv rows and KV blocks
    cannot drift apart.
    """

    def __init__(self, *args, pretrained_config, mapping, max_batch_size, **kwargs):
        # Resolved before super().__init__() because _build_cache_config runs
        # inside it and the base does not keep pretrained_config. Everything else
        # the conv sizing needs is on self by then.
        self._conv_config = getattr(pretrained_config, "text_config", pretrained_config)
        self._conv_dtype = _resolve_conv_dtype(pretrained_config)
        super().__init__(
            *args,
            pretrained_config=pretrained_config,
            mapping=mapping,
            max_batch_size=max_batch_size,
            **kwargs,
        )
        self._conv_cache = InklingConvStateCache(
            pretrained_config,
            self._conv_tp_size,
            self._num_conv_request_slots,
            torch.device("cuda", torch.cuda.current_device()),
            self._conv_dtype,
            reserve_attention_dp_slot=self._reserve_attention_dp_slot,
            max_draft_len=self.max_draft_len,
            allocate=self._conv_state_buffer,
        )
        logger.info(
            f"Inkling short-conv state pool: {self._conv_cache.num_slots} rows "
            f"({self._num_conv_request_slots} request + "
            f"{self._num_reserved_conv_slots} reserved), "
            f"{self._conv_cache.conv_state_bytes() / (1 << 20):.1f} MiB, "
            "backed by V2 SSM layers"
        )

    # ---- conv geometry, all derived from what the base already resolved -----
    @property
    def _conv_tp_size(self) -> int:
        # The attention TP, not the global one: the k/v convs follow the kv-head
        # split, as V2 does for the paged pool.
        return 1 if self.mapping.enable_attention_dp else self.mapping.tp_size

    @property
    def _reserve_attention_dp_slot(self) -> bool:
        return bool(self.mapping.enable_attention_dp)

    @property
    def _num_conv_request_slots(self) -> int:
        # One row per resident sequence; each pipeline stage holds a microbatch.
        return self.max_batch_size * self.mapping.pp_size

    @property
    def _num_reserved_conv_slots(self) -> int:
        # Asked of the pool rather than re-derived: the two counts must agree or
        # slots_for indexes the V2 buffer out of bounds.
        return InklingConvStateCache.reserved_slot_count(
            reserve_attention_dp_slot=self._reserve_attention_dp_slot
        )

    def _conv_bytes_per_slot(self, global_layer_idx: int) -> List[int]:
        """Bytes one request occupies in each of the layer's four conv states."""
        config = self._conv_config
        kv_dim = (
            config.layer_num_kv_heads(global_layer_idx) * config.layer_head_dim(global_layer_idx)
        ) // self._conv_tp_size
        window = config.sconv_kernel_size - 1
        itemsize = torch.empty((), dtype=self._conv_dtype).element_size()
        return [
            c * window * itemsize for c in (kv_dim, kv_dim, config.hidden_size, config.hidden_size)
        ]

    # ---- V2 configuration -------------------------------------------------
    def _conv_layer_id(self, local_layer_idx: int) -> LayerId:
        """Cache-layer id holding ``local_layer_idx``'s four conv states."""
        return LayerId(self.num_local_layers + local_layer_idx)

    def _build_cache_config(self, config):
        """Append one SSM layer per decoder layer for the short-conv state.

        The attention layers ``_build_base_config`` produced are preserved
        exactly: an Inkling layer needs both KV and conv state, and a
        ``LayerConfig`` is one or the other, so appending is the only way.
        """
        layers = list(config.layers)
        num_attention_layers = len(layers)
        # _conv_layer_id derives SSM ids arithmetically from this count.
        assert num_attention_layers == self.num_local_layers, (
            num_attention_layers,
            self.num_local_layers,
        )
        for local_idx in range(num_attention_layers):
            layers.append(
                SsmLayerConfig(
                    layer_id=LayerId(num_attention_layers + local_idx),
                    buffers=[
                        BufferConfig(role=role, size=nbytes)
                        for role, nbytes in zip(
                            CONV_ROLES, self._conv_bytes_per_slot(self.pp_layers[local_idx])
                        )
                    ],
                )
            )

        num_reserved = self._num_reserved_conv_slots
        # Non-request rows as zero-capacity requests, like Mamba: no attention
        # pages, one state slot each -- exactly their effect on the pool.
        dummies = [KVCacheDesc(capacity=0, history_length=0) for _ in range(num_reserved)]
        constraints = [
            replace(batch, kv_caches=[*batch.kv_caches, *dummies]) for batch in config.constraints
        ]
        # Length-independent floor: a conv row is fixed per resident sequence, so
        # the base's token-scaled constraints do not bound the pool.
        constraints.append(
            BatchDesc(
                [
                    KVCacheDesc(capacity=0, history_length=0)
                    for _ in range(self._num_conv_request_slots + num_reserved)
                ]
            )
        )
        # KVCacheManagerConfig asserts this whenever an SSM layer exists.
        return replace(config, layers=layers, constraints=constraints, commit_min_snapshot=True)

    def _get_pool_roles(self, pool_id: int):
        """Name a role that actually exists in ``pool_id``.

        The base returns ``Role.KEY`` unconditionally, but conv pools hold no KEY.
        The role is resolved rather than named because V2 coalesces the four
        states by size, so which one represents a pool is its packing decision.
        """
        first_layer = int(self.impl.layer_grouping[pool_id][0])
        if first_layer < self.num_local_layers:
            return super()._get_pool_roles(pool_id)
        for role in CONV_ROLES:
            if (pool_id, role) in self._pool_layer_ids_by_role:
                return role, None
        raise RuntimeError(
            f"Inkling conv pool {pool_id} (first layer {first_layer}) holds none "
            f"of {[str(r) for r in CONV_ROLES]}; _build_cache_config and the pool "
            "packing disagree about what landed where."
        )

    def _conv_state_buffer(
        self, local_layer_idx: int, role: DataRole, state_shape: List[int]
    ) -> torch.Tensor:
        """A ``[num_slots, *state_shape]`` view of one conv state's pool memory.

        Mirrors ``MambaHybridCacheManagerV2._get_state_buffer``. The
        ``as_strided`` step converts the raw page-indexed view into slot indexing,
        since V2 packs ``page_index_scale`` pages per logical slot.
        """
        layer_id = self._conv_layer_id(local_layer_idx)
        addr = self.impl.get_mem_pool_base_address(layer_id, role, PageIndexMode.SHARED)
        num_pages = self.impl.get_page_index_upper_bound(layer_id, role)
        raw = convert_to_torch_tensor(
            TensorWrapper(addr, self._conv_dtype, [num_pages] + list(state_shape))
        )
        scale = self.impl.get_page_index_scale(layer_id, role)
        num_slots = (num_pages + scale - 1) // scale
        return raw.as_strided(
            [num_slots] + list(state_shape),
            [raw.stride(0) * scale] + list(raw.stride()[1:]),
        )

    # ---- model-facing -----------------------------------------------------
    @property
    def conv_state_cache(self) -> InklingConvStateCache:
        """The short-conv state pool, for the metadata's per-step publication."""
        return self._conv_cache

    def get_conv_states(self, layer_idx: int) -> InklingConvState:
        """The four short-conv state buffers of ``layer_idx``.

        Named after ``BaseMambaCacheManager.get_conv_states`` but deliberately not
        implementing it: that hook returns one tensor per layer, which cannot
        express four convs at two widths, and Inkling backs no SSM state at all.
        """
        return self._conv_cache.layer_state(layer_idx)

    def get_state_indices(self) -> torch.Tensor:
        """Pool rows of the current batch, in packed batch order."""
        return self._conv_cache.state_indices

    def free_conv_state(self, request_ids) -> None:
        self._conv_cache.free(list(request_ids))

    # ---- KVCacheManagerV2 -----------------------------------------------------
    def free_resources(self, request, *args, **kwargs):
        """Release the conv row together with the request's KV blocks, so a
        leaked row cannot be reused with stale state."""
        rid = getattr(request, "py_request_id", None)
        if rid is not None:
            self.free_conv_state([rid])
        return super().free_resources(request, *args, **kwargs)
