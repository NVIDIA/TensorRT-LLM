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

import hashlib
from dataclasses import replace
from typing import List

import torch

from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    DEFAULT_BEAM_INDEX,
    BatchDesc,
    BufferConfig,
    DataRole,
    KVCacheDesc,
    LayerId,
    PageIndexMode,
    SsmLayerConfig,
)

from .....logger import logger
from ....pyexecutor.kv_cache_manager_v2 import BlockReusePolicy, KVCacheManagerV2
from ....pyexecutor.llm_request import LlmRequest
from ....pyexecutor.scheduler import ScheduledRequests
from .conv_state import CONV_ROLE, InklingConvState, InklingConvStateCache


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

    Registering the conv window as V2 SSM layers is what makes block reuse
    servable: V2 attaches it to the block committed at a snapshot ordinal and
    hands it back on a prefix hit.
    """

    def __init__(self, *args, pretrained_config, mapping, max_batch_size, **kwargs):
        # Resolved before super().__init__() because _build_cache_config runs
        # inside it and the base does not keep pretrained_config. Everything else
        # the conv sizing needs is on self by then.
        self._conv_config = getattr(pretrained_config, "text_config", pretrained_config)
        self._conv_dtype = _resolve_conv_dtype(pretrained_config)
        # prepare_expect_snapshot_points needs the interval; the base keeps none.
        self._kv_cache_config = args[0] if args else kwargs.get("kv_cache_config")
        # A property of the model, so warn once per manager, not per request.
        self._warned_multimodal_reuse = False
        super().__init__(
            *args,
            pretrained_config=pretrained_config,
            mapping=mapping,
            max_batch_size=max_batch_size,
            **kwargs,
        )
        # Resolved on first use: `impl` is not guaranteed live at the end of
        # the base constructor. From the first conv layer, as Mamba does.
        self._conv_layer_group_id = None
        self._conv_cache = InklingConvStateCache(
            pretrained_config,
            self._conv_tp_size,
            self._num_conv_request_slots,
            torch.device("cuda", torch.cuda.current_device()),
            self._conv_dtype,
            reserve_attention_dp_slot=self._reserve_attention_dp_slot,
            max_draft_len=self.max_draft_len,
            allocate=self._conv_state_buffer,
            resolve_slot=self._conv_slot_for_request,
        )
        logger.info(
            f"Inkling short-conv state pool: {self._conv_cache.num_slots} rows "
            f"({self._num_conv_request_slots} request + "
            f"{self._num_reserved_conv_slots} reserved), "
            f"{self._conv_cache.conv_state_bytes() / (1 << 20):.1f} MiB, "
            "backed by V2 SSM layers"
        )

    def _conv_slot_for_request(self, request_id: int):
        """V2's recurrent-state slot for ``request_id``, or None if it has none.

        V2 restores a reuse hit into the slot it assigned, so the pool has to
        be indexed its way. None for CUDA-graph padding sentinels, which have no
        cache entry and keep the reserved rows.
        """
        kv_cache = self.kv_cache_map.get(request_id)
        if kv_cache is None:
            return None
        if self._conv_layer_group_id is None:
            self._conv_layer_group_id = self.impl.get_layer_group_id(self._conv_layer_id(0))
        slot = kv_cache.get_ssm_block_base_index(self._conv_layer_group_id)
        return None if slot < 0 else int(slot)

    # ========================= KV cache block reuse =========================
    # Everything to "end KV cache block reuse" runs only under
    # enable_block_reuse.
    def prepare_expect_snapshot_points(self, requests) -> None:
        """Where this batch's requests may snapshot their short-conv window.

        Found by ``py_executor`` through ``hasattr``; the scheduler then ends a
        context chunk only here, which is the half a model cannot do for itself
        -- a snapshot can only be taken where an iteration ends. The interval is
        deliberately coarse: one snapshot costs the whole model's conv window,
        so reuse lands only on multiples of it.
        """
        state_config = getattr(self._kv_cache_config, "mamba_state_config", None)
        interval = getattr(state_config, "periodic_snapshot_interval", 0) or 0
        for request in requests:
            if not self.enable_block_reuse or not interval:
                request.expect_snapshot_points = []
                continue
            request.expect_snapshot_points = list(range(interval, request.prompt_len + 1, interval))

    # The three methods below mirror MambaHybridCacheManagerV2's, copied rather
    # than shared: lifting them into kv_cache_manager_v2.py edits a file every
    # model depends on, and inheriting the Mamba manager drags in an SSM-state
    # interface four short convs at two widths cannot satisfy. Every ``self``
    # here is a KVCacheManagerV2 member. A test compares them as ASTs.
    def _mark_context_position_as_history(self, request: LlmRequest, kv_cache) -> None:
        """Advance history without making later recurrent state reusable."""
        history_length = request.context_current_position
        if history_length <= kv_cache.history_length:
            return
        capacity = max(kv_cache.capacity, history_length)
        if not kv_cache.resize(capacity, history_length=history_length):
            raise ValueError(
                "Failed to resize history length of the Inkling V2 cache for "
                f"request {request.py_request_id} to {history_length} tokens"
            )

    def try_commit_blocks(self, request: LlmRequest, kv_cache=None) -> None:
        """Commit at each declared snapshot point, not only at the end.

        The base commits once as prefill advances, which is enough when
        everything reusable lives in the KV pages. The conv window does not: V2
        attaches it to the block committed at a snapshot ordinal, so a commit
        has to happen there or a later hit has no window to restore.
        """
        should_block_reuse = (
            self.enable_block_reuse and not self.is_draft and not request.is_dummy_request
        )
        if not should_block_reuse:
            return

        if kv_cache is None:
            kv_cache = self.kv_cache_map.get(request.py_request_id)
        if kv_cache is None:
            return

        snapshot_points = request.expect_snapshot_points
        commit_limit = (
            min(max(snapshot_points), request.prompt_len) if snapshot_points else request.prompt_len
        )
        commit_end = min(request.context_current_position, commit_limit)
        if (
            request.context_current_position in request.expect_snapshot_points
            and commit_end > kv_cache.num_committed_tokens
        ):
            tokens = self._augment_tokens_for_block_reuse(
                request.get_tokens(DEFAULT_BEAM_INDEX),
                request,
                start=kv_cache.num_committed_tokens,
                end=commit_end,
            )
            kv_cache.commit(tokens)
        if request.context_current_position >= commit_limit:
            self._mark_context_position_as_history(request, kv_cache)
        if request.context_remaining_length == 0:
            kv_cache.stop_committing()

    def update_context_resources(self, scheduled_batch: ScheduledRequests) -> None:
        for request in scheduled_batch.context_requests:
            kv_cache = self.kv_cache_map.get(request.py_request_id)
            if kv_cache is None or not kv_cache.is_active:
                continue

            should_block_reuse = (
                self.enable_block_reuse and not self.is_draft and not request.is_dummy_request
            )
            is_all_reusable = self.block_reuse_policy == BlockReusePolicy.ALL_REUSABLE
            is_snapshot_boundary = (
                request.context_current_position in request.expect_snapshot_points
            )
            has_pending_snapshot = any(
                point > request.context_current_position for point in request.expect_snapshot_points
            )
            should_resize = not should_block_reuse or (
                not is_all_reusable and not has_pending_snapshot
            )
            should_commit = (
                is_all_reusable or is_snapshot_boundary or request.context_remaining_length == 0
            )

            if should_resize and not kv_cache.resize(None, request.context_current_position):
                raise ValueError(
                    "Failed to resize history length of the Inkling V2 cache "
                    f"for request {request.py_request_id} to "
                    f"{request.context_current_position} tokens at context update"
                )
            if should_commit:
                self.try_commit_blocks(request, kv_cache)
            if request.context_remaining_length == 0:
                if self.conversation_manager is not None:
                    self.conversation_manager.save_drop_plan(request, kv_cache)
                kv_cache.enable_swa_scratch_reuse = False

    def _augment_tokens_for_block_reuse(self, tokens, req, start=0, end=None):
        """Keep multimodal requests out of the shared radix tree.

        The base distinguishes images by rewriting their spans into keys
        carrying a content digest, but only for a request holding
        ``multimodal_hashes`` -- which Inkling produces for none of them, so two
        prompts with DIFFERENT images match as the same prefix: MMMU 81.3% with
        reuse off against 31.3% with it on, same 32 items, no error anywhere.
        Salting position 0 with the request id gives such a request a private
        chain, leaving text requests alone.
        """
        augmented = super()._augment_tokens_for_block_reuse(tokens, req, start, end)
        if not self.enable_block_reuse:
            return augmented
        if getattr(req, "multimodal_hashes", None) is not None:
            return augmented
        if getattr(req, "py_multimodal_data", None) is None:
            return augmented
        # Only the chunk holding position 0 needs it: the tree chains each key
        # through its parent, so a private first block privatises the rest.
        if start != 0 or not len(augmented):
            return augmented
        if not self._warned_multimodal_reuse:
            self._warned_multimodal_reuse = True
            logger.warning(
                "Inkling: KV cache block reuse is disabled for multimodal "
                "requests. Their image/video/audio spans carry no content "
                "digest, so a shared prefix would be matched on placeholder "
                "token ids alone and serve one item's KV for another's. Text "
                "requests are unaffected."
            )
        salt = hashlib.sha256(
            b"inkling-multimodal-no-digest-%d" % int(getattr(req, "request_id", 0))
        ).digest()
        poisoned = list(augmented)
        poisoned[0] = salt
        return poisoned

    def prepare_context(self, req):
        """Observation only: how much prefix was reused, and from whose slot.

        The framework's own ``KVCacheV2SsmSnapshotIterationStats`` would be the
        right source, but a served deployment publishes no
        ``kvCacheIterationStats*`` key at all, so there is nothing to read. A
        reuse number without a gate reads the same whether reuse worked or never
        ran. The slot source rides along because the two failure modes -- no
        reuse, and reuse into a row the kernels do not read -- are
        indistinguishable from accuracy.
        """
        first = bool(getattr(req, "is_first_context_chunk", True))
        ok = super().prepare_context(req)
        if not first:
            return ok
        d = getattr(self, "_reuse_dbg", None)
        if d is None:
            d = self._reuse_dbg = {"n": 0, "hits": 0, "best": 0, "v2_slots": 0}
        d["n"] += 1
        pos = int(getattr(req, "context_current_position", 0) or 0)
        if pos > 0:
            d["hits"] += 1
            d["best"] = max(d["best"], pos)
        if (
            self._conv_slot_for_request(
                getattr(req, "py_request_id", getattr(req, "request_id", -1))
            )
            is not None
        ):
            d["v2_slots"] += 1
        if d["n"] % 64 == 0:
            logger.info(
                f"Inkling prefix reuse: {d['hits']}/{d['n']} requests, "
                f"longest={d['best']} tokens, conv rows from V2: "
                f"{d['v2_slots']}/{d['n']}"
            )
        return ok

    # ======================= end KV cache block reuse =======================

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

    def _conv_section_bytes(self, global_layer_idx: int) -> List[int]:
        """Per-slot bytes of the layer's four conv sections, in pool order."""
        config = self._conv_config
        kv_dim = (
            config.layer_num_kv_heads(global_layer_idx) * config.layer_head_dim(global_layer_idx)
        ) // self._conv_tp_size
        window = config.sconv_kernel_size - 1
        itemsize = torch.empty((), dtype=self._conv_dtype).element_size()
        return [
            c * window * itemsize for c in (kv_dim, kv_dim, config.hidden_size, config.hidden_size)
        ]

    def _conv_bytes_per_slot(self, global_layer_idx: int) -> int:
        """Bytes one request occupies in the layer's conv state.

        One number, not four: the sections share a buffer. ``_conv_section_bytes``
        keeps their widths, which is what a TP-mismatched transfer splits on.
        """
        return sum(self._conv_section_bytes(global_layer_idx))

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
                        BufferConfig(
                            role=CONV_ROLE,
                            size=self._conv_bytes_per_slot(self.pp_layers[local_idx]),
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
        """
        first_layer = int(self.impl.layer_grouping[pool_id][0])
        if first_layer < self.num_local_layers:
            return super()._get_pool_roles(pool_id)
        if (pool_id, CONV_ROLE) in self._pool_layer_ids_by_role:
            return CONV_ROLE, None
        raise RuntimeError(
            f"Inkling conv pool {pool_id} (first layer {first_layer}) does not "
            f"hold {CONV_ROLE}; _build_cache_config and the pool packing "
            "disagree about what landed where."
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
