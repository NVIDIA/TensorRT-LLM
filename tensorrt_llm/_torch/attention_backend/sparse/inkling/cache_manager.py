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

The short-conv state is **registered with V2 as SSM layers** rather than
allocated on the side. ``SsmLayerConfig`` is the framework's container for
per-request fixed-size state -- ``MambaHybridCacheManagerV2`` already carries
Mamba's conv/SSM state through it -- and it fits here for the same reason: a
short-conv window is ``[channels, kernel-1]`` per request and does not grow with
the sequence, so it cannot be an attention buffer (there ``BufferConfig.size``
means bytes *per block*, i.e. per ``tokens_per_block`` tokens).

What that buys, in order of importance:

1. **The pool's bytes enter V2's quota.** They used to be a plain ``torch``
   allocation no byte quota knew about, accounted for only because the throwaway
   manager built during KV-cache size estimation also held one, so
   ``configure_kv_cache_capacity``'s peak-memory reading happened to include it.
   That was correct only while the estimation pool and the serving pool were
   exactly the same size -- an invariant nothing enforced and any future
   runtime-dependent sizing would have broken silently.
2. Stable device addresses come from the pool allocator, so a captured CUDA
   graph keeps working without this file having to argue for it.
3. Equal-sized buffers are coalesced into shared pool slots automatically, so
   the k/v pair (kv-head width) and the attn/mlp pair (hidden width) do not each
   need a separate allocation.

What it does **not** buy: slot allocation. V2 owns the memory, not the
request-to-slot mapping; ``MambaHybridCacheManagerV2`` likewise keeps its own
allocator. :class:`InklingConvStateCache` still assigns rows.

The SSM layers are **appended** after the attention layers
``_build_base_config`` already produced, not interleaved with them. Interleaving
is what DeepSeek-V4 does, and it has to, because there one model layer's KV is
itself split across several cache layers; Inkling's KV stays one buffer per
layer, so appending leaves every attention ``layer_id`` where it was and no
mapping table is needed anywhere -- ``get_buffers`` / ``get_batch_cache_indices``
/ the metadata's page tables are all untouched.

There is deliberately no shared conv-state protocol. ``BaseMambaCacheManager``
is the closest existing one, but it mandates SSM state and replay metadata
Inkling cannot back, and its one-tensor-per-layer accessor cannot express
Inkling's four convs per layer at two different widths. If a second short-conv
model appears, widen that hook rather than adding another beside it.
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

    Not the manager's ``dtype`` argument -- that is the KV cache dtype, a C++
    binding type ``torch.zeros`` rejects, and it is ``nvfp4``/``fp8`` on
    quantized releases while the conv pool holds unquantized pre-conv
    activations.

    HuggingFace configs carry ``torch_dtype`` as either a ``torch.dtype`` or its
    name (``"bfloat16"``), so both are accepted. An unresolvable value raises:
    the previous silent fall back to bfloat16 turned an fp16 checkpoint into a
    pool of the wrong dtype, which reaches the conv kernels as a dtype mismatch
    far from its cause.
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


class _InklingConvGeometry:
    """Everything ``_build_cache_config`` needs to size the conv state.

    Split out because it has to be computed *before* ``super().__init__()`` --
    ``_build_cache_config`` runs inside it -- and every input is already an
    argument of the manager's constructor.
    """

    def __init__(self, pretrained_config, mapping, max_batch_size, *, spec_config=None):
        config = getattr(pretrained_config, "text_config", pretrained_config)
        self.config = config
        self.dtype = _resolve_conv_dtype(pretrained_config)
        self.kwin = config.sconv_kernel_size - 1
        # The conv pool's k/v width follows the attention kv-head split, so it
        # takes the attention TP, not the global one -- the same rule
        # KVCacheManagerV2 applies to the paged pool. Dividing by the global
        # tp_size would allocate narrow conv rows for full-width convs.
        self.tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size
        # One row per sequence that can be resident at once. Pipeline stages
        # each hold a microbatch, so the bound is max_batch_size * pp_size --
        # the same count MambaHybridCacheManagerV2 calls
        # ``_max_resident_sequences``.
        self.num_request_slots = max_batch_size * mapping.pp_size
        self.reserve_attention_dp_slot = bool(mapping.enable_attention_dp)
        self.max_draft_len = int(getattr(spec_config, "max_draft_len", 0) or 0)
        # Asked of the pool rather than re-derived: the slot layout is
        # InklingConvStateCache's to define, and the two counts must agree
        # exactly or the V2 buffer is indexed out of bounds by slots_for.
        self.num_reserved_slots = InklingConvStateCache.reserved_slot_count(
            reserve_attention_dp_slot=self.reserve_attention_dp_slot
        )

    def channels(self, global_layer_idx: int) -> List[int]:
        """Per-conv channel counts for one layer, in ``InklingConvState`` order."""
        kv_dim = (
            self.config.layer_num_kv_heads(global_layer_idx)
            * self.config.layer_head_dim(global_layer_idx)
        ) // self.tp_size
        hidden = self.config.hidden_size
        return [kv_dim, kv_dim, hidden, hidden]

    def bytes_per_slot(self, global_layer_idx: int) -> List[int]:
        """Bytes one request occupies in each of the layer's four conv states."""
        itemsize = torch.empty((), dtype=self.dtype).element_size()
        return [c * self.kwin * itemsize for c in self.channels(global_layer_idx)]


class InklingHybridCacheManager(KVCacheManagerV2):
    """Paged KV (V2, per-layer geometry) + the short-conv state pool.

    Folding the pool into the cache manager -- the shape
    ``CppMambaHybridCacheManager`` uses for mamba conv/SSM state -- lets it reach
    the model through the standard ``attn_metadata.kv_cache_manager`` field and
    be released by the manager's own ``free_resources``. The conv rows are then
    freed by the same call that frees the request's KV blocks, so the two views
    cannot drift apart.

    The pool's bytes are declared to V2 through :meth:`_build_cache_config`, so
    they are inside the same quota as the paged KV rather than riding on a
    peak-memory coincidence during capacity estimation. See the module docstring.
    """

    def __init__(self, *args, pretrained_config, mapping, max_batch_size, **kwargs):
        # The three arguments the pool needs are declared, not read back out of
        # ``**kwargs``. KVCacheManagerV2 takes ``mapping`` / ``max_batch_size``
        # keyword-only and absorbs ``pretrained_config`` into ``**kwargs``
        # without storing it, so subscripting kwargs worked only as long as
        # every caller passed all three by keyword: omitting one surfaced as a
        # bare KeyError from inside this constructor rather than as a TypeError
        # naming the parameter.
        #
        # The conv geometry is computed BEFORE super().__init__() because
        # ``_build_cache_config`` runs inside it and has to size the SSM buffers.
        self._conv_geometry = _InklingConvGeometry(
            pretrained_config,
            mapping,
            max_batch_size,
            spec_config=kwargs.get("spec_config"),
        )
        super().__init__(
            *args,
            pretrained_config=pretrained_config,
            mapping=mapping,
            max_batch_size=max_batch_size,
            **kwargs,
        )
        geo = self._conv_geometry
        self._conv_cache = InklingConvStateCache(
            pretrained_config,
            geo.tp_size,
            geo.num_request_slots,
            torch.device("cuda", torch.cuda.current_device()),
            geo.dtype,
            reserve_attention_dp_slot=geo.reserve_attention_dp_slot,
            max_draft_len=geo.max_draft_len,
            allocate=self._conv_state_buffer,
        )
        logger.info(
            f"Inkling short-conv state pool: {self._conv_cache.num_slots} rows "
            f"({geo.num_request_slots} request + {geo.num_reserved_slots} reserved), "
            f"{self._conv_cache.conv_state_bytes() / (1 << 20):.1f} MiB, "
            "backed by V2 SSM layers"
        )

    # ---- V2 configuration -------------------------------------------------
    def _conv_layer_id(self, local_layer_idx: int) -> LayerId:
        """Cache-layer id holding ``local_layer_idx``'s four conv states.

        The SSM layers are appended after the attention ones, so this is plain
        arithmetic and no mapping table is needed. Keeping the attention ids at
        ``0..N-1`` is the point: everything that addresses the paged KV keeps
        working untouched.
        """
        return LayerId(self.num_local_layers + local_layer_idx)

    def _build_cache_config(self, config):
        """Append one SSM layer per decoder layer for the short-conv state.

        The attention layers ``_build_base_config`` produced are preserved
        exactly -- this only extends the list, following
        ``MambaHybridCacheManagerV2``'s "preserve what the base built, then
        declare our own" shape (it replaces its Mamba layers; Inkling's layers
        need *both* KV and conv, and a ``LayerConfig`` is one or the other, so
        appending is the only way to have both).
        """
        geo = self._conv_geometry
        layers = list(config.layers)
        num_attention_layers = len(layers)
        # ``_conv_layer_id`` derives the SSM ids as ``num_local_layers + offset``
        # rather than by lookup, so the base must have emitted exactly one
        # attention layer per local decoder layer. Assert it here instead of
        # letting a mismatch surface as a wrong pool address at first decode.
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
                            CONV_ROLES, geo.bytes_per_slot(self.pp_layers[local_idx])
                        )
                    ],
                )
            )

        # Reserve the non-request rows (CUDA-graph padding sentinels, and the
        # attention-DP idle dummy) the same way Mamba does: as zero-capacity
        # requests. They cost no attention pages but each holds one state slot,
        # which is exactly their effect on the pool.
        dummies = [
            KVCacheDesc(capacity=0, history_length=0) for _ in range(geo.num_reserved_slots)
        ]
        constraints = [
            replace(batch, kv_caches=[*batch.kv_caches, *dummies]) for batch in config.constraints
        ]
        # A floor on the state slots, independent of sequence length. Attention
        # pages scale with tokens so the base's constraints already bound them,
        # but a conv row is fixed-size per resident sequence: without this the
        # pool can be sized for fewer sequences than the scheduler will admit,
        # and the failure is a mid-run "out of rows" rather than a startup error.
        constraints.append(
            BatchDesc(
                [
                    KVCacheDesc(capacity=0, history_length=0)
                    for _ in range(geo.num_request_slots + geo.num_reserved_slots)
                ]
            )
        )
        # KVCacheManagerConfig asserts this whenever any SSM layer is present:
        # their lifecycle needs minimum-snapshot commit semantics. Harmless for
        # Inkling, which refuses block reuse outright, so no commit is ever
        # attempted -- but the runtime config still requires the invariant, and
        # it is a hard assert rather than a default (MambaHybridCacheManagerV2
        # sets it for the same reason).
        return replace(
            config, layers=layers, constraints=constraints, commit_min_snapshot=True
        )

    def _get_pool_roles(self, pool_id: int):
        """Name a role that actually exists in ``pool_id``.

        The base returns ``Role.KEY`` unconditionally, and
        ``_build_pool_mapping_tensors`` then looks up
        ``_pool_layer_ids_by_role[(pool_id, role)]`` -- a dict keyed only by the
        pairs that exist. Appending conv layers creates pools holding no KEY at
        all, so the base's answer is a ``KeyError`` for them.
        ``MambaHybridCacheManagerV2`` overrides this for the same reason.

        Unlike Mamba's version this resolves the role rather than naming one:
        the four conv states come in two widths, V2 coalesces by size, and which
        role represents a given pool is its packing decision, not ours.
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
        ``as_strided`` step is not cosmetic: V2 coalesces same-sized per-layer
        buffers inside one pool slot, so the raw page-indexed view has
        ``page_index_scale`` pages per logical slot. Callers index by slot.
        """
        layer_id = self._conv_layer_id(local_layer_idx)
        addr = self.impl.get_mem_pool_base_address(layer_id, role, PageIndexMode.SHARED)
        num_pages = self.impl.get_page_index_upper_bound(layer_id, role)
        raw = convert_to_torch_tensor(
            TensorWrapper(addr, self._conv_geometry.dtype, [num_pages] + list(state_shape))
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

        Named after ``BaseMambaCacheManager.get_conv_states`` on purpose: this
        is the same question asked of the same kind of manager. It cannot
        *implement* that hook, which returns one tensor per layer and cannot
        express Inkling's four convs at two widths -- widening the shared hook
        is the move if a second short-conv model appears.
        """
        return self._conv_cache.layer_state(layer_idx)

    def get_state_indices(self) -> torch.Tensor:
        """Pool rows of the current batch, in packed batch order."""
        return self._conv_cache.state_indices

    def free_conv_state(self, request_ids) -> None:
        self._conv_cache.free(list(request_ids))

    # ---- KVCacheManagerV2 -----------------------------------------------------
    def free_resources(self, request, *args, **kwargs):
        """Release the conv row with the request's KV blocks.

        This is what lets the model engine's warmup/estimation dummy-batch
        cleanup drop its Inkling-specific branch: it already calls
        ``kv_cache_manager.free_resources(req)`` for every dummy request, and a
        leaked conv row would later be reused, with stale state, by a real
        request whose id collides with a dummy id.
        """
        rid = getattr(request, "py_request_id", None)
        if rid is not None:
            self.free_conv_state([rid])
        return super().free_resources(request, *args, **kwargs)
