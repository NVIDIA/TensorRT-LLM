# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 owner-aware packed caches on the shared V2 request lifecycle."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from copy import copy
from dataclasses import replace
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import torch
from cuda.bindings import runtime as cudart

from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._torch.host_staging import copy_host_to_device
from tensorrt_llm._torch.modules.multi_stream_utils import private_stream
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    _RESERVED_REQUEST_IDS,
    GPU_LEVEL,
    KVCacheManagerV2,
    Role,
    _estimate_cache_size_components,
    _fill_kv_pages,
    _get_generation_kv_capacity,
)
from tensorrt_llm._utils import CUASSERT, TensorWrapper, convert_to_torch_tensor, prefer_pinned
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    BAD_PAGE_INDEX,
    AttentionLayerConfig,
    AttentionReusePolicy,
    BufferConfig,
    DataRole,
    LayerId,
    PageIndexMode,
)

from .....configs.deepseek_v41 import (
    bounded_replay_tokens,
    decoder_bounded_replay_enabled,
    encoder_replay_enabled,
)
from .params import CSA2Layout
from .quantization import (
    INDEX_DATA_BYTES,
    INDEX_PAGE_ROWS,
    INDEX_SCALE_BYTES,
    store_index_rows,
    store_layer_rows,
    store_rows,
)


def _same_conversion_inputs(current, previous) -> bool:
    """Whether two ``(rows, caches, page-storage versions, scratch, SWA destination)``
    conversion inputs match.

    Holding the previous caches and destination keeps their identities from being reused.
    """
    if previous is None:
        return False
    rows, caches, versions, scratch, destination = current
    previous_rows, previous_caches, previous_versions, previous_scratch, previous_destination = (
        previous
    )
    if (
        versions != previous_versions
        or (destination is None) != (previous_destination is None)
        or (
            destination is not None
            and (destination.data_ptr(), destination.shape, destination.stride())
            != (
                previous_destination.data_ptr(),
                previous_destination.shape,
                previous_destination.stride(),
            )
        )
        or len(caches) != len(previous_caches)
        or any(cache is not other for cache, other in zip(caches, previous_caches))
        or not np.array_equal(rows, previous_rows)
    ):
        return False
    if scratch is None or previous_scratch is None:
        return scratch is previous_scratch
    return all(np.array_equal(a, b) for a, b in zip(scratch, previous_scratch))


def _validate_context_swa_layer_limit(limit: int | None, num_layers: int) -> None:
    if limit is not None and (type(limit) is not int or not 0 < limit < num_layers):
        raise ValueError("CSA2 context SWA layer limit must be an integer inside the model")


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, SpeculativeConfig
    from tensorrt_llm.mapping import Mapping

    from .....pyexecutor.llm_request import LlmRequest


def _get_replay_capture_window(
    layout: CSA2Layout,
    num_layers: int,
    mapping: Mapping,
    kv_cache_config: KvCacheConfig | None,
    pretrained_config: PretrainedConfig | None,
    spec_config: SpeculativeConfig | None,
) -> int:
    """Required SWA coverage, including embedded DSpark captures."""
    owners = [layer for layer in layout.kv_source_layer_ids if layer < num_layers]
    if not decoder_bounded_replay_enabled() or not owners:
        return 0
    last_owner = max(owners)
    if layout.compress_ratios[last_owner] != 1 or any(
        layer >= last_owner
        for layer in (getattr(pretrained_config, "engram_layer_ids", None) or ())
    ):
        return 0
    if spec_config is None:
        return layout.window_size
    if not (spec_config.spec_dec_mode.is_dspark() and spec_config.draft_is_embedded_in_target):
        return 0
    return max(layout.window_size, pretrained_config.sliding_window)


def _swa_cache_window(
    swa_window: int,
    replay_window: int,
    *,
    is_decoder: bool,
    encoder_replay: bool,
    remote_tail: bool = False,
) -> int:
    """Retain writable replay rows and any remote handoff lookback."""
    if remote_tail and not is_decoder and replay_window > 0:
        # Preserve the handoff lookback reserve and next-query boundary protection.
        return swa_window + replay_window + 1
    if replay_window > 0 and (encoder_replay or is_decoder):
        # The extra row protects the replay start at partial-block endpoints.
        return max(swa_window, replay_window + 1)
    return swa_window


def _cache_size_layer_components(
    layout: CSA2Layout,
    layers: Sequence[int],
    replay_window: int,
    encoder_replay: bool,
    reserve: int,
    swa_layer_limit: int | None,
    *,
    remote_tail: bool = False,
) -> tuple[list[int], list[int | None], list[int]]:
    """Match packed layer geometry and reuse-policy scratch groups for sizing."""
    owners = [layer for layer in layout.kv_source_layer_ids if layer in layers]
    encoder_policy = (
        AttentionReusePolicy.OPTIONAL
        if replay_window and encoder_replay
        else AttentionReusePolicy.REQUIRED
    )
    sizes, windows, scratch_groups = [], [], []
    for layer in layers:
        decoder = bool(owners) and layer >= max(owners)
        if swa_layer_limit is None or layer < swa_layer_limit:
            sizes.append(528)
            windows.append(
                _swa_cache_window(
                    layout.window_size,
                    replay_window,
                    is_decoder=decoder,
                    encoder_replay=encoder_replay,
                    remote_tail=remote_tail,
                )
                + reserve
            )
            scratch_groups.append(
                (
                    AttentionReusePolicy.PRIVATE if replay_window and decoder else encoder_policy
                ).value
            )
        if layer in owners:
            ratio = layout.compress_ratios[layer]
            sizes.append((MAIN_ROW_BYTES + INDEX_ROW_BYTES) // ratio)
            windows.append(None)
            scratch_groups.append(AttentionReusePolicy.REQUIRED.value)
            if ratio == 2:
                sizes.append(4096)
                windows.append(2 + reserve)
                scratch_groups.append(encoder_policy.value)
    return sizes, windows, scratch_groups


class CSA2CacheRole(Enum):
    SWA = "swa"
    GLOBAL = "global"
    INDEX = "index"
    COMPRESSOR_KV = "compressor_kv"
    COMPRESSOR_SCORE = "compressor_score"

    @property
    def role(self) -> DataRole:
        return DataRole(f"csa2_{self.value}")

    @property
    def index_mode(self) -> PageIndexMode:
        return (
            PageIndexMode.SHARED if self in (self.GLOBAL, self.INDEX) else PageIndexMode.PER_LAYER
        )


# Native paged index kernels (DeepGEMM/CuTe FP4 paged MQA logits) read 64-row pages of
# 68-byte records: 64 packed E2M1 bytes followed by four UE8M0 scale bytes.
INDEX_ROW_BYTES = INDEX_DATA_BYTES + INDEX_SCALE_BYTES
MAIN_ROW_BYTES = 288


class _CSA2LayerKind(Enum):
    """Stable virtual-layer identities, separate from the per-buffer roles."""

    SWA = 0
    GLOBAL = 1
    COMPRESSOR_STATE = 2


class CSA2CacheManager(KVCacheManagerV2):
    """Private SWA, owner-only GLOBAL main/index buffers, and ratio-two compressor state.

    The GLOBAL layer group carries two buffers with identical page indices: a
    288-byte main record and a 68-byte index record per compressed position.
    Sharing one page lifecycle keeps copy-on-write, eviction and transfer
    atomic at the allocator page level, while the separate index buffer is
    exactly the ``[pages, 64, 1, 68]`` layout the native paged index kernels
    read in place (as the DeepSeek-V4 indexer cache does), so decode never
    repacks index rows.

    Logical buffer identities include model layer, compression ratio and role.
    SWA belongs to the uncompressed family regardless of the layer's GLOBAL ratio.
    """

    # A list while a deferring context prepare runs: block-offset copies are recorded for it.
    _deferred_block_offsets: list | None = None

    def __init__(
        self,
        kv_cache_config,
        kv_cache_type,
        *,
        num_layers: int,
        tokens_per_block: int,
        mapping,
        pretrained_config=None,
        sparse_attention_config=None,
        layout: CSA2Layout | None = None,
        context_swa_layer_limit: int | None = None,
        bounded_replay_on_generation: bool = False,
        num_kv_heads: int = 1,
        head_dim: int = 512,
        **kwargs,
    ):
        _validate_context_swa_layer_limit(context_swa_layer_limit, num_layers)
        # Set only after the model has actually removed its decoder weights.
        # Keep logical model-layer IDs; only the physical SWA groups are pruned.
        self.context_swa_layer_limit = context_swa_layer_limit
        self.bounded_replay_on_generation = bounded_replay_on_generation
        spec_config = kwargs.get("spec_config")
        if spec_config is not None:
            from tensorrt_llm._torch.speculative.utils import get_num_spec_layers

            if not spec_config.is_linear_tree or getattr(spec_config, "use_dynamic_tree", False):
                raise NotImplementedError(
                    "CSA2 supports contiguous-prefix chain verification, not token trees"
                )
            if get_num_spec_layers(spec_config):
                raise NotImplementedError(
                    "CSA2 virtual draft layers require an explicit model-layer mapping"
                )
        self._page_index_converters = {}
        self._swa_publication_token = None
        self._swa_publication = None
        self._csa2_linear_speculation = spec_config is not None
        self._csa2_request_epoch_counter = 0
        self._csa2_request_epochs = {}
        layer_mask = kwargs.get("layer_mask")
        if layer_mask is not None and not all(layer_mask):
            raise ValueError("CSA2 cache does not support disabled layers in layer_mask")
        if pretrained_config is None:
            model_config = kwargs.get("model_config")
            pretrained_config = getattr(model_config, "pretrained_config", None)
        if layout is None:
            if pretrained_config is None:
                raise ValueError("CSA2 requires pretrained_config or an explicit layout")
            layout = CSA2Layout.from_hf_config(pretrained_config)
        if len(layout.compress_ratios) < num_layers:
            raise ValueError("CSA2 layout must describe every active attention layer")
        if tokens_per_block not in (128, 256):
            raise ValueError("CSA2 cache requires tokens_per_block 128 or 256")
        if kv_cache_type != CacheType.SELFKONLY or num_kv_heads != 1 or head_dim != 512:
            raise ValueError("CSA2 cache requires SELFKONLY with one 512-dimensional KV head")
        if mapping.pp_size != 1 or mapping.cp_size != 1:
            raise ValueError("CSA2 shared cache owners currently require PP=CP=1")
        self.layout = replace(
            layout,
            compress_ratios=layout.compress_ratios[:num_layers],
            kv_source_layer_ids=tuple(i for i in layout.kv_source_layer_ids if i < num_layers),
            index_source_layer_ids=tuple(
                i for i in layout.index_source_layer_ids if i < num_layers
            ),
            candidate_source_layer_id=(
                layout.candidate_source_layer_id
                if layout.candidate_source_layer_id is not None
                and layout.candidate_source_layer_id < num_layers
                else None
            ),
        )
        capture_window = _get_replay_capture_window(
            self.layout, num_layers, mapping, kv_cache_config, pretrained_config, spec_config
        )
        self._decoder_replay_window = bounded_replay_tokens(capture_window) if capture_window else 0
        self._encoder_replay_enabled = encoder_replay_enabled(
            max_num_tokens=kwargs.get("max_num_tokens", 8192),
            replay_window=max(self.layout.window_size, self._decoder_replay_window),
        )
        self._encoder_reuse_policy = (
            AttentionReusePolicy.OPTIONAL
            if self._encoder_replay_enabled
            else AttentionReusePolicy.REQUIRED
        )
        self._encoder_optional_groups: tuple[int, ...] = ()
        self._global_buffers: dict[tuple[int, int, CSA2CacheRole], torch.Tensor] = {}
        self._pool_views = {}
        super().__init__(
            kv_cache_config,
            kv_cache_type,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            mapping=mapping,
            **kwargs,
        )
        self.is_vswa = True
        if self.enable_block_reuse and self._decoder_replay_window:
            owner = max(self.layout.kv_source_layer_ids)
            self._encoder_roles = tuple(
                (layer, ratio, role)
                for layer, ratio, role in self._layer_roles
                if role not in (CSA2CacheRole.GLOBAL, CSA2CacheRole.INDEX)
                and (layer < owner or role != CSA2CacheRole.SWA)
            )
            self._encoder_optional_groups = tuple(
                sorted(
                    {
                        self.impl.get_layer_group_id(self._layer_roles[key])
                        for key in self._encoder_roles
                        if self._encoder_replay_enabled
                    }
                )
            )
            self._log_checkpoint_pools()

    def _log_checkpoint_pools(self):
        """Report physical quota once per group, without counting role views twice."""
        policies = {
            self.impl.get_layer_group_id(layer.layer_id): layer.reuse_policy
            for layer in self.impl.init_config.layers
        }
        for level, tier in enumerate(self.impl.cache_tier_list):
            grouping = self.impl.get_life_cycle_pool_group_indices(level)
            for group, stats in enumerate(self.impl.get_storage_statistics(level)):
                members = [lc for lc, pg in enumerate(grouping) if pg == group]
                optional_pages = sum(
                    policies[lc] == AttentionReusePolicy.OPTIONAL for lc in members
                )
                names = sorted({AttentionReusePolicy(policies[lc]).name for lc in members})
                slot_bytes = sum(
                    stats.slot_sizes if hasattr(stats, "slot_sizes") else stats.slot_size
                )
                logger.info(
                    f"DeepSeek-V4.1 KV pool: tier={tier}, group={group}, policy={names}, "
                    f"slots={stats.total}, slot_bytes={slot_bytes}, bytes={stats.total * slot_bytes}, "
                    f"checkpoint_capacity_upper_bound={stats.total // optional_pages if optional_pages else 0}"
                )

    def _derive_reuse_salt(self, cache_salt: str | None) -> int | None:
        if self._encoder_replay_enabled:
            # Preserve None vs empty-string and compose with the user namespace.
            cache_salt = "deepseek-v41-csa2-encoder:approximate:" + repr(cache_salt)
        return super()._derive_reuse_salt(cache_salt)

    def _context_reuse_tokens(self, req, reuse_limit: int | None = None):
        from .....pyexecutor.ced_replay import requires_full_decoder_prefill

        remote_tail_start = getattr(req, "py_csa2_remote_tail_start", None)
        if remote_tail_start is not None:
            # The destination may reuse the whole transferred prefix. The
            # source must execute its final prefix token to finish context and
            # materialize any Encoder recovery before handing off at S.
            limit = remote_tail_start
            if getattr(req, "py_csa2_remote_tail_mode", None) == "source":
                limit = max(0, limit - 1)
            reuse_limit = limit if reuse_limit is None else min(reuse_limit, limit)
        if req.is_disagg_generation_init_state:
            # The context worker supplies missing working pages, including the
            # private Decoder window. No local prefill/HC replay is needed.
            return super()._context_reuse_tokens(req, reuse_limit)
        if self._decoder_replay_window and requires_full_decoder_prefill(req):
            # Decoder pages are private in this lifecycle. Full-row consumers
            # must run from the beginning instead of resuming only Encoder KV.
            reuse_limit = 0
        if self._decoder_replay_window and not self._encoder_replay_enabled:
            # Required Encoder state resumes without historical recomputation.
            # Leave enough new rows to initialize the private Decoder: there is
            # deliberately no cross-request HC tail snapshot.
            limit = max(0, req.prompt_len - self._decoder_replay_window)
            reuse_limit = limit if reuse_limit is None else min(limit, reuse_limit)
        return super()._context_reuse_tokens(req, reuse_limit)

    def _resume_context_cache(self, req: LlmRequest, cache) -> bool:
        """Select Encoder reuse before the normal context-cache activation."""
        from .....pyexecutor.ced_replay import EncoderCheckpoint, EncoderReplay

        if (
            not self._decoder_replay_window
            or req.is_dummy
            or not req.is_first_context_chunk
            or req.py_ced_replay is not None
            or cache.num_committed_tokens == 0
        ):
            return super()._resume_context_cache(req, cache)

        if req.is_disagg_generation_init_state:
            # Restore each available group independently. The transfer adapter
            # uses the post-resume reuse status to request all missing groups;
            # never issue an EncoderReplay plan on a receive-only request.
            groups = [
                group
                for group in self._encoder_optional_groups
                if cache.reuse_status[group].complete
            ]
            return self._resume_and_restore(req.py_request_id, cache, groups)

        if getattr(req, "py_csa2_remote_tail_mode", None) == "destination":
            # Transfer supplies the Encoder lookback; local reuse status does
            # not describe the working pages received from the context worker.
            return super()._resume_context_cache(req, cache)

        reused = cache.num_committed_tokens
        groups = None
        available = False
        restore = True
        if self._encoder_replay_enabled:
            # TODO: Publish and restore partial Encoder checkpoints at non-block-
            # aligned Global endpoints. Until then, incomplete OPTIONAL coverage falls
            # back to Encoder replay without shortening the Global prefix hit.
            status = cache.reuse_status
            available = any(status[group].complete for group in self._encoder_optional_groups)
            restore = (
                reused <= max(0, req.prompt_len - self._decoder_replay_window)
                and bool(self._encoder_optional_groups)
                and all(status[group].complete for group in self._encoder_optional_groups)
            )
            groups = self._encoder_optional_groups if restore else ()
        if not self._resume_and_restore(req.py_request_id, cache, groups):
            return False

        # Publish only after activation succeeds. Retries and later chunks resume
        # the same working pages without repeating initial OPTIONAL selection.
        req.py_ced_replay = (
            EncoderCheckpoint(req.py_request_id, id(cache), reused)
            if restore
            else EncoderReplay(
                req.py_request_id, id(cache), reused, max(0, reused - self._decoder_replay_window)
            )
        )
        if restore:
            label = (
                "encoder snapshot"
                if self._encoder_replay_enabled
                else "required Encoder checkpoint"
            )
            logger.debug(
                f"DeepSeek-V4.1 {label} restored: P={reused}, "
                f"suffix={req.prompt_len - reused}; encoder replay rows=0"
            )
        elif available and req.prompt_len - reused < self._decoder_replay_window:
            logger.debug(
                f"DeepSeek-V4.1 encoder snapshot bypassed: P={reused}, "
                f"suffix={req.prompt_len - reused}; encoder replay rows={req.py_ced_replay.num_tokens}"
            )
        return True

    def context_replay_tokens(self, req: LlmRequest) -> int | None:
        from .....pyexecutor.ced_replay import EncoderCheckpoint, EncoderReplay

        plan = req.py_ced_replay
        if isinstance(plan, EncoderCheckpoint):
            return 0
        return plan.num_tokens if isinstance(plan, EncoderReplay) else None

    def _swa_retention(self, layer: int) -> int:
        window = _swa_cache_window(
            self.layout.window_size,
            self._decoder_replay_window,
            is_decoder=bool(self.layout.kv_source_layer_ids)
            and layer >= max(self.layout.kv_source_layer_ids),
            encoder_replay=self._encoder_replay_enabled,
            remote_tail=getattr(self, "bounded_replay_on_generation", False),
        )
        return self._window(window)

    def _create_kv_cache(self, request_id, lora_task_id, input_tokens, **kwargs):
        cache = super()._create_kv_cache(request_id, lora_task_id, input_tokens, **kwargs)
        if cache is not None:
            self._csa2_request_epoch_counter += 1
            self._csa2_request_epochs[request_id] = self._csa2_request_epoch_counter
        return cache

    def request_epoch(self, request_id: int) -> int:
        """Distinguish a recycled request ID from its previous cache lifetime."""
        return self._csa2_request_epochs[request_id]

    def request_epochs(self, request_ids) -> tuple[int, ...]:
        """``request_epoch`` of each of ``request_ids``."""
        return tuple(map(self._csa2_request_epochs.__getitem__, request_ids))

    def free_resources(self, request, pin_on_release=False):
        request.py_ced_replay = None
        super().free_resources(request, pin_on_release=pin_on_release)
        self._csa2_request_epochs.pop(request.py_request_id, None)

    def validate_verification(self, lengths: list[int], num_contexts: int, enabled: bool) -> None:
        if not enabled:
            return
        if not self._csa2_linear_speculation:
            raise ValueError("CSA2 verification requires configured chain rewind capacity")
        if max(lengths[num_contexts:], default=0) > self.max_draft_len + 1:
            raise ValueError("CSA2 verification exceeds its reserved rewind window")

    def update_resources(self, scheduled_batch, attn_metadata=None, kv_cache_dtype_byte_size=None):
        # The generic relocation kernel assumes a uniform uncompressed KV pool.
        # Linear acceptance counts need only V2 resize/history updates, whereas
        # explicit token relocation would also require recomputing compressed pairs.
        for request in scheduled_batch.generation_requests:
            if request.py_num_accepted_draft_tokens_indices:
                raise NotImplementedError(
                    "CSA2 accepted-prefix updates do not support token relocation indices"
                )
        return super().update_resources(scheduled_batch, attn_metadata, kv_cache_dtype_byte_size)

    def get_swa_replay_ranges(
        self,
        cached_prefix_lengths: list[int],
        suffix_lengths: list[int] | None = None,
        *,
        decoder: bool = False,
    ) -> tuple[tuple[int, int], ...]:
        """Return required writable query intervals for bounded reconstruction.

        The GLOBAL prefix must already be restored by the caller. Native V2
        matching is unchanged. SWA retention includes the first replay row at
        C-R; the caller prepares every row in the returned range for writes.
        """
        suffix_lengths = (
            [0] * len(cached_prefix_lengths) if suffix_lengths is None else suffix_lengths
        )
        if len(suffix_lengths) != len(cached_prefix_lengths):
            raise ValueError("CSA2 replay prefixes and suffixes must have matching request counts")
        window = self.layout.window_size
        if decoder or self._encoder_replay_enabled:
            window = max(window, self._decoder_replay_window)
        result = []
        for prefix, suffix in zip(cached_prefix_lengths, suffix_lengths):
            if prefix < 0 or suffix < 0 or prefix + suffix > self.max_seq_len:
                raise ValueError("CSA2 replay range exceeds its admitted context")
            if decoder and suffix:
                raise ValueError("Decoder SWA replay requires the complete prompt GLOBAL cache")
            result.append((max(0, prefix - window), prefix + suffix))
        return tuple(result)

    def has_swa_cache(self, layer_idx: int) -> bool:
        """Whether a logical model layer owns a physical SWA allocation."""
        limit = getattr(self, "context_swa_layer_limit", None)
        return limit is None or layer_idx < limit

    def _cache_role_key(
        self, layer_idx: int, role: CSA2CacheRole
    ) -> tuple[int, int, CSA2CacheRole]:
        """Return the model layer, buffer compression family and role."""
        ratio = 1 if role == CSA2CacheRole.SWA else self.layout.compress_ratios[layer_idx]
        return layer_idx, ratio, role

    def _cache_layer_id(self, layer_idx: int, role: CSA2CacheRole) -> LayerId:
        if role == CSA2CacheRole.SWA and not self.has_swa_cache(layer_idx):
            raise ValueError(f"Context decoder layer {layer_idx} has no SWA cache")
        return self._layer_roles[self._cache_role_key(layer_idx, role)]

    def _window(self, base: int) -> int:
        return base + self.max_draft_len + self.reuse_match_backoff

    def _runtime_cache_size_layer_components(self) -> tuple[list[int], list[int | None], list[int]]:
        return _cache_size_layer_components(
            self.layout,
            self.pp_layers,
            self._decoder_replay_window,
            self._encoder_replay_enabled,
            self.max_draft_len + self.reuse_match_backoff,
            getattr(self, "context_swa_layer_limit", None),
            remote_tail=getattr(self, "bounded_replay_on_generation", False),
        )

    def _get_runtime_cache_size_layer_components(self):
        sizes, windows, _ = self._runtime_cache_size_layer_components()
        return sizes, windows

    def _get_runtime_cache_size_scratch_group_keys(self) -> list[int]:
        return self._runtime_cache_size_layer_components()[2]

    def _get_typical_seq_len(self, kv_cache_config):
        typical = kv_cache_config.avg_seq_len or self.max_seq_len
        quota = kv_cache_config.max_gpu_total_bytes
        if not quota:
            return typical
        layer_sizes, windows, scratch_keys = self._runtime_cache_size_layer_components()
        _, per_token, per_request = _estimate_cache_size_components(
            layer_sizes,
            windows,
            self.tokens_per_block,
            scratch=self.enable_swa_scratch_reuse,
            generation_capacity_headroom=self._generation_kv_capacity_headroom,
            scratch_group_keys=scratch_keys,
        )
        if not per_token:  # No GLOBAL pool: nothing scales with the length.
            return typical
        # SWA and compressor state cost a fixed amount per request, GLOBAL a length-proportional one. A split
        # for a full batch of `typical`-length requests that the quota cannot hold caps admission at the
        # window-bounded pools while shorter requests leave GLOBAL idle; split for the length at which a full
        # batch fits instead.
        resumable = quota * kv_cache_config.max_util_for_resume
        fitting = (resumable / self._get_generation_request_capacity() - per_request) / per_token
        return int(min(typical, max(fitting, self.tokens_per_block)))

    def get_layer_bytes_per_token(self, local_layer_idx, data_role):
        # Replaced by the packed declarative layer configuration below.
        return 1

    def _build_cache_config(self, config):
        layers = []
        self._layer_roles: dict[tuple[int, int, CSA2CacheRole], LayerId] = {}
        self._physical_roles: dict[tuple[LayerId, DataRole], tuple[int, int, CSA2CacheRole]] = {}
        # Reuse the V2 extractor's virtual-layer contract. Physical layer IDs
        # include GLOBAL and compressor groups and cannot index pp_layers.
        # Only canonical owners enter this map; consumer aliases below must
        # not change the identity peers use for a shared GLOBAL allocation.
        self._layer_attn_to_layer_id = {}

        def add(
            model_layer: int,
            ratio: int,
            roles: list[CSA2CacheRole],
            sizes: list[int],
            window: int | None,
        ) -> None:
            layer_id = LayerId(len(layers))
            if roles[0] == CSA2CacheRole.SWA:
                kind = _CSA2LayerKind.SWA
            elif roles[0] == CSA2CacheRole.GLOBAL:
                kind = _CSA2LayerKind.GLOBAL
            else:
                kind = _CSA2LayerKind.COMPRESSOR_STATE
            self._layer_attn_to_layer_id[model_layer, kind] = layer_id
            for role in roles:
                self._layer_roles[model_layer, ratio, role] = layer_id
                self._physical_roles[layer_id, role.role] = (model_layer, ratio, role)
            private_config = {}
            if self._decoder_replay_window and window is not None:
                decoder_swa = roles[0] == CSA2CacheRole.SWA and model_layer >= max(
                    self.layout.kv_source_layer_ids
                )
                private_config["reuse_policy"] = (
                    AttentionReusePolicy.PRIVATE if decoder_swa else self._encoder_reuse_policy
                )
            layers.append(
                AttentionLayerConfig(
                    layer_id=layer_id,
                    buffers=[
                        BufferConfig(role=role.role, size=size * self.tokens_per_block)
                        for role, size in zip(roles, sizes)
                    ],
                    sliding_window_size=window,
                    num_sink_tokens=None,
                    **private_config,
                )
            )

        for layer in self.pp_layers:
            if self.has_swa_cache(layer):
                add(
                    layer,
                    1,
                    [CSA2CacheRole.SWA],
                    [528],
                    self._swa_retention(layer),
                )
            if layer in self.layout.kv_source_layer_ids:
                ratio = self.layout.compress_ratios[layer]
                # One page group, two buffers: main rows for attention and
                # native-layout index rows for the paged indexer kernels.
                add(
                    layer,
                    ratio,
                    [CSA2CacheRole.GLOBAL, CSA2CacheRole.INDEX],
                    [MAIN_ROW_BYTES // ratio, INDEX_ROW_BYTES // ratio],
                    None,
                )
                if ratio == 2:
                    add(
                        layer,
                        ratio,
                        [CSA2CacheRole.COMPRESSOR_KV, CSA2CacheRole.COMPRESSOR_SCORE],
                        [2048, 2048],
                        self._window(2),
                    )
        for layer in self.pp_layers:
            owner = self.layout.layer(layer).kv_source
            if owner is not None:
                ratio = self.layout.compress_ratios[layer]
                for role in (CSA2CacheRole.GLOBAL, CSA2CacheRole.INDEX):
                    self._layer_roles[layer, ratio, role] = self._layer_roles[owner, ratio, role]
        scratch = config.swa_scratch_reuse
        if scratch is not None:
            # Linear verification may reject every draft token. Context
            # lookahead (num_extra_kv_tokens) can be one smaller than this.
            scratch = copy(scratch)
            scratch.max_rewind_len = max(scratch.max_rewind_len, self.max_draft_len)
        config = copy(config)
        config.layers = layers
        # The native optional property's setter rejects None. The copied
        # configuration already preserves its disabled/default state.
        if scratch is not None:
            config.swa_scratch_reuse = scratch
        return config

    def get_buffers(self, layer_idx: int, role: CSA2CacheRole = CSA2CacheRole.SWA):
        layer_id = self._cache_layer_id(layer_idx, role)
        addr = self.impl.get_mem_pool_base_address(layer_id, role.role, role.index_mode)
        upper = self.impl.get_page_index_upper_bound(layer_id, role.role)
        converter = self._get_page_index_converter(layer_id, role.role)
        if role.index_mode == PageIndexMode.PER_LAYER and converter.layer_offset is not None:
            upper += converter.layer_offset * converter.expansion
        if role in (CSA2CacheRole.GLOBAL, CSA2CacheRole.INDEX):
            rows, width, dtype = (
                self.tokens_per_block // self.layout.compress_ratios[layer_idx],
                MAIN_ROW_BYTES if role == CSA2CacheRole.GLOBAL else INDEX_ROW_BYTES,
                DataType.UINT8,
            )
        elif role == CSA2CacheRole.SWA:
            rows, width, dtype = self.tokens_per_block, 528, DataType.UINT8
        else:
            rows, width, dtype = self.tokens_per_block, 512, DataType.FLOAT
        # Wrapping a pool costs a tensor construction, so equal views are reused.
        # A view made in inference mode is an inference tensor, which cannot be
        # updated in place outside it.
        key = (addr, dtype, upper, rows, width, torch.is_inference_mode_enabled())
        if key not in self._pool_views:
            self._pool_views[key] = convert_to_torch_tensor(
                TensorWrapper(addr, dtype, (upper, rows, width))
            )
        return self._pool_views[key]

    def has_private_swa_suffix(self, first_layer: int) -> bool:
        """Whether every SWA layer from this boundary uses PRIVATE storage."""
        known = self.__dict__.setdefault("_private_swa_suffix", {})
        if first_layer not in known:
            layers = [
                layer
                for layer in self.pp_layers
                if layer >= first_layer and self.has_swa_cache(layer)
            ]
            # Each init_config access copies the whole configuration.
            config_layers = self.impl.init_config.layers
            known[first_layer] = bool(layers) and all(
                config_layers[self._cache_layer_id(layer, CSA2CacheRole.SWA)].reuse_policy
                == AttentionReusePolicy.PRIVATE
                for layer in layers
            )
        return known[first_layer]

    def get_swa_buffer(self, layer_idx: int) -> torch.Tensor:
        return self.get_buffers(layer_idx, CSA2CacheRole.SWA).view(-1, 528)

    def get_main_buffer(self, layer_idx: int) -> torch.Tensor:
        """Contiguous ``[rows, 288]`` main records of the layer's KV source."""
        owner = self.layout.layer(layer_idx).kv_source
        if owner is None:
            raise ValueError("SWA-only layers have no global cache")
        key = self._cache_role_key(owner, CSA2CacheRole.GLOBAL)
        if key not in self._global_buffers:
            self._global_buffers[key] = self.get_buffers(owner, CSA2CacheRole.GLOBAL).view(
                -1, MAIN_ROW_BYTES
            )
        return self._global_buffers[key]

    def get_index_pages(self, layer_idx: int) -> torch.Tensor:
        """Index records as native ``[pages, 64, 1, 68]`` page-footer pages.

        Each 64-row page stores 64x64 packed data bytes followed by 64x4 scale
        bytes, the layout the paged FP4 MQA-logits kernels read in place. A
        GLOBAL page holds ``tokens_per_block // ratio`` consecutive rows, so it
        is ``index_pages_per_global_page`` consecutive native pages.
        """
        owner = self.layout.layer(layer_idx).kv_source
        if owner is None:
            raise ValueError("SWA-only layers have no global cache")
        key = self._cache_role_key(owner, CSA2CacheRole.INDEX)
        if key not in self._global_buffers:
            pool = self.get_buffers(owner, CSA2CacheRole.INDEX)
            self._global_buffers[key] = pool.view(-1, INDEX_PAGE_ROWS, 1, INDEX_ROW_BYTES)
        return self._global_buffers[key]

    def index_pages_per_global_page(self, layer_idx: int) -> int:
        owner = self.layout.layer(layer_idx).kv_source
        if owner is None:
            raise ValueError("SWA-only layers have no global cache")
        rows = self.tokens_per_block // self.layout.compress_ratios[owner]
        if rows % INDEX_PAGE_ROWS:
            raise ValueError("CSA2 GLOBAL pages must hold whole native index pages")
        return rows // INDEX_PAGE_ROWS

    def write_swa(self, layer_idx, slots, values, transform=None) -> None:
        store_rows(self.get_swa_buffer(layer_idx), slots, values, "swa", transform)

    def write_global(
        self, layer_idx, slots, main, index, main_transform=None, index_transform=None
    ) -> None:
        if self.layout.layer(layer_idx).kv_source != layer_idx:
            raise ValueError("Only the configured KV source may publish global rows")
        store_rows(self.get_main_buffer(layer_idx), slots, main, "main", main_transform)
        store_index_rows(self.get_index_pages(layer_idx), slots, index, index_transform)

    def write_layer_rows(
        self, layer_idx, slots, values, transform=None, *, global_rows=None, index_q=None
    ):
        """One launch for a layer's SWA rows, its global rows and its packed index queries.

        ``global_rows`` is ``(owner, slots, main, index, main_transform,
        index_transform)`` for a Full layer; ``index_q`` returns ``(data, scales)``.
        """
        main = index = None
        if global_rows is not None:
            owner, global_slots, main_values, index_values, main_transform, index_transform = (
                global_rows
            )
            if self.layout.layer(owner).kv_source != owner:
                raise ValueError("Only the configured KV source may publish global rows")
            main = (self.get_main_buffer(owner), global_slots, main_values, main_transform)
            index = (self.get_index_pages(owner), global_slots, index_values, index_transform)
        return store_layer_rows(
            self.get_swa_buffer(layer_idx),
            slots,
            values,
            transform,
            main=main,
            index=index,
            index_q=index_q,
        )

    def _get_page_index_converter(self, layer_id, role):
        # V2 converters are constant for a manager's initialized storage
        # geometry. Base page indices and scratch descriptors remain per-call.
        key = (layer_id, role)
        if key not in self._page_index_converters:
            self._page_index_converters[key] = self.impl.get_page_index_converter(layer_id, role)
        return self._page_index_converters[key]

    @contextmanager
    def _swa_publication_scope(self):
        token = object()
        self._swa_publication = None
        self._swa_publication_token = token
        try:
            yield token
        finally:
            # A nested prepare cannot resurrect an older publication either.
            self._swa_publication = None
            self._swa_publication_token = None

    def _get_swa_publication(self, token, destination, request_ids, num_contexts, page_count):
        publication = self._swa_publication
        if token is None or token is not self._swa_publication_token or publication is None:
            return None
        target, ids, contexts, width, device_layout = publication
        if (
            target is not destination
            or ids != tuple(request_ids)
            or contexts != num_contexts
            or page_count > width
            or device_layout != (target.data_ptr(), target.shape, target.stride())
        ):
            return None
        return target[:, : len(ids), 0, :page_count]

    def get_cache_indices(self, request_id: int, layer_idx: int, role: CSA2CacheRole):
        layer_id = self._cache_layer_id(layer_idx, role)
        pool_id = self.layer_to_pool_mapping_dict[layer_id]
        cache = self.kv_cache_map[request_id]
        converter = self._get_page_index_converter(layer_id, role.role)
        return converter(
            cache.get_base_page_indices(pool_id).tolist(),
            role.index_mode,
            cache.get_scratch_desc(pool_id),
        )

    def compute_batch_page_tables(
        self, request_ids, num_contexts, swa_out=None, page_versions=None
    ) -> None:
        """Convert every per-layer page table of the batch with one kernel.

        Like DeepSeek-V4's sliding block tables, one kernel writes int32
        ``[tables, requests, blocks]`` on the device, including the SWA scratch
        mappings, from a device snapshot of the host base rows of every pool.
        ``batch_page_table`` and ``batch_pages_allocated`` read the result. With
        ``swa_out`` (int32 ``[SWA tables, requests, width]``, unit column stride),
        the leading SWA tables go there instead, and ``batch_page_table`` must not
        read them. A generation-only batch with unchanged conversion inputs and
        ``swa_out`` keeps the previous tables; otherwise only the live columns are
        converted and the rest of ``swa_out`` is BAD_PAGE_INDEX. ``page_versions`` are the
        requests' page-storage versions when the caller has already read them.
        """
        from . import kernel

        count = len(request_ids)
        self._batch_page_tables = self._device_batch_page_tables[:, :count]
        self._batch_scratch = None
        self._batch_rows = np.empty(0, dtype=np.int32)
        if not count:
            return
        rows = self.index_mapper.get_copy_index(list(request_ids), num_contexts, 1)
        self._batch_rows = rows.numpy().copy()
        scratch = self._scratch_descriptors(request_ids)
        if scratch is not None:
            self._batch_scratch = scratch[0]
        caches = [self.kv_cache_map[request_id] for request_id in request_ids]
        inputs = None
        if not num_contexts:
            if page_versions is None:
                page_versions = [cache.page_storage_version for cache in caches]
            inputs = (self._batch_rows, caches, list(page_versions), scratch, swa_out)
            if _same_conversion_inputs(inputs, self._batch_conversion_inputs):
                # The previous conversion recorded this event.
                torch.cuda.current_stream().wait_event(self._base_rows_read[self._base_rows_slot])
                return
        self._batch_conversion_inputs = None
        live = self._live_columns(caches, scratch)
        width = max(live.max(), self._table_live[:count].max())
        if swa_out is None:
            width = max(width, self._swa_table_live[:count].max())
        width = int(width)
        written = 0 if swa_out is None else min(width, swa_out.shape[-1])
        if not width:
            if swa_out is not None:
                swa_out.fill_(BAD_PAGE_INDEX)
            self._batch_conversion_inputs = inputs
            return
        # Snapshot the base rows now: the host rewrites (and releases) them for
        # later batches before this step's device work runs. The copy engine
        # takes the snapshot while the host waits.
        slot = self._base_rows_slot = 1 - self._base_rows_slot
        device_base = self._device_base_rows[slot]
        self._base_rows_stream.wait_event(self._base_rows_read[slot])
        host = self.host_kv_cache_block_offsets
        row_bytes = host.shape[-1] * host.element_size()
        # One copy of the first ``width`` columns of rows 0..max(row) of every pool.
        copy = cudart.cudaMemcpy3DParms()
        copy.srcPtr = cudart.make_cudaPitchedPtr(
            host.data_ptr(), host.stride(1) * host.element_size(), row_bytes, host.shape[1]
        )
        copy.dstPtr = cudart.make_cudaPitchedPtr(
            device_base.data_ptr(), row_bytes, row_bytes, device_base.shape[1]
        )
        copy.extent = cudart.make_cudaExtent(
            width * host.element_size(), int(self._batch_rows.max()) + 1, host.shape[0]
        )
        copy.kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        CUASSERT(cudart.cudaMemcpy3DAsync(copy, self._base_rows_stream.cuda_stream))
        self._base_rows_stream.synchronize()
        device_rows = self._device_batch_rows[:count]
        copy_host_to_device(
            self._host_staging,
            "batch_rows",
            device_rows,
            torch.from_numpy(self._batch_rows),
        )
        if scratch is not None:
            staged = []
            for key, value in zip(("scratch_ranges", "scratch_slots"), scratch):
                buffer = self._device_scratch.get(key)
                if buffer is None or buffer.numel() < value.size:
                    buffer = self._device_scratch[key] = torch.empty(
                        value.size, dtype=torch.int32, device=device_base.device
                    )
                staged.append(buffer[: value.size].view(value.shape))
                copy_host_to_device(self._host_staging, key, staged[-1], torch.from_numpy(value))
            scratch = staged
        scratch_ranges, scratch_slots = scratch or (None, None)
        kernel.convert_page_tables(
            device_base,
            device_rows,
            self._device_batch_page_params,
            self._batch_page_tables[..., :width],
            BAD_PAGE_INDEX,
            scratch_ranges,
            scratch_slots,
            low_out=None if swa_out is None else swa_out[..., :written],
        )
        self._base_rows_read[slot].record()
        self._table_live[:count] = live
        if swa_out is None:
            self._swa_table_live[:count] = live
        else:
            swa_out[..., written:].fill_(BAD_PAGE_INDEX)
        self._batch_conversion_inputs = inputs

    def _live_columns(self, caches, scratch) -> np.ndarray:
        """Per batch row, the page-table columns that can map a page (blocks and scratch)."""
        width = self.host_kv_cache_block_offsets.shape[-1]
        live = np.minimum([cache.num_blocks for cache in caches], width)
        if scratch is not None:
            live = np.maximum(live, np.minimum(scratch[0][..., 1].max(axis=0), width))
        return live

    def _scratch_descriptors(self, request_ids):
        """Host int32 ``[pools, requests, 2]`` scratch ranges and ``[pools, requests, n]``
        slot ids of the batch, or None when no request holds SWA scratch slots."""
        if not self.enable_swa_scratch_reuse:
            return None
        holders = [
            (row, self.kv_cache_map[request_id])
            for row, request_id in enumerate(request_ids)
            if self.kv_cache_map[request_id].has_scratch_slots
        ]
        if not holders:
            return None
        descs = [
            (index, row, desc)
            for row, cache in holders
            for index, pool in enumerate(self._scratch_pools.tolist())
            if (desc := cache.get_scratch_desc(pool))
        ]
        ranges = np.zeros((len(self._scratch_pools), len(request_ids), 2), dtype=np.int32)
        width = max((len(desc.slot_ids) for _, _, desc in descs), default=1)
        slots = np.zeros((len(self._scratch_pools), len(request_ids), width), dtype=np.int32)
        for index, row, desc in descs:
            ranges[index, row] = (desc.range.beg, desc.range.end)
            slots[index, row, : len(desc.slot_ids)] = desc.slot_ids
        return ranges, slots

    def batch_page_table(self, spec, width: int) -> torch.Tensor:
        """Converted int32 ``[requests, width]`` table of ``spec`` for the last computed batch."""
        return self._batch_page_tables[self._batch_page_index[spec], :, :width]

    def batch_page_table_key(self, spec) -> tuple:
        """Conversion of ``spec``'s batch page table: specs with equal keys read the same base
        rows of the same pool with the same scale, offset and scratch geometry, so their converted
        tables are equal for every batch."""
        return tuple(int(v) for v in self._batch_page_params[self._batch_page_index[spec]])

    def batch_pages_allocated(self, specs, rows, columns, width: int) -> np.ndarray:
        """Whether each batch ``(rows, columns)`` page of ``specs`` is mapped, as bool ``[specs, n]``.

        Read on the host from the base page table (a unit converter maps
        exactly the allocated base pages) and the batch's SWA scratch ranges;
        columns beyond ``width`` are unmapped. Only the pool decides, so each
        distinct pool of ``specs`` is read once.
        """
        rows = np.asarray(rows, dtype=np.int64)
        columns = np.asarray(columns, dtype=np.int64)
        base = self.host_kv_cache_block_offsets.numpy()
        limit = min(width, base.shape[-1])
        if not rows.size or limit == 0:
            return np.zeros((len(specs), rows.size), dtype=bool)
        key = tuple(specs)
        groups = self._batch_page_pools.get(key)
        if groups is None:
            groups = self._batch_page_pools[key] = np.unique(
                self._batch_page_params[[self._batch_page_index[spec] for spec in specs], 3],
                return_inverse=True,
            )
        pools, inverse = groups
        pages = base[
            self._scratch_pools[pools, None],
            self._batch_rows[rows][None],
            0,
            np.minimum(columns, limit - 1)[None],
        ]
        allocated = (pages != BAD_PAGE_INDEX) & (columns[None] < limit)
        if self._batch_scratch is not None:
            ranges = self._batch_scratch[pools[:, None], rows[None]]
            allocated |= (columns[None] >= ranges[..., 0]) & (columns[None] < ranges[..., 1])
        return allocated[inverse.reshape(-1)]

    def get_layer_page_index_scale(self, layer_idx: int) -> int:
        return int(
            self.impl.get_page_index_scale(
                self._cache_layer_id(layer_idx, CSA2CacheRole.SWA), CSA2CacheRole.SWA.role
            )
        )

    def get_batch_cache_indices(self, request_ids, layer_idx=None, num_blocks_per_seq=None):
        layer_idx = self.pp_layers[0] if layer_idx is None else layer_idx
        result = []
        for i, request_id in enumerate(request_ids):
            pages = self.get_cache_indices(request_id, layer_idx, CSA2CacheRole.SWA)
            if num_blocks_per_seq is not None:
                pages = pages[: num_blocks_per_seq[i]]
            result.append(pages)
        return result

    def _prepare_page_table_tensor(self, index_mapper_capacity):
        self.num_attention_op_pools = self.num_local_layers
        self.kv_cache_pool_pointers = torch.tensor(
            [
                [
                    self.impl.get_mem_pool_base_address(
                        self._cache_layer_id(layer, CSA2CacheRole.SWA),
                        CSA2CacheRole.SWA.role,
                        PageIndexMode.PER_LAYER,
                    )
                    if self.has_swa_cache(layer)
                    else 0,
                    0,
                ]
                for layer in self.pp_layers
            ],
            dtype=torch.int64,
            pin_memory=prefer_pinned(),
            device="cpu",
        )
        self.kv_cache_pool_mapping = torch.tensor(
            [[i, 0] for i in range(self.num_local_layers)],
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
            device="cpu",
        )
        self.host_kv_cache_block_offsets = torch.full(
            (
                self.num_pools,
                index_mapper_capacity * self.max_beam_width,
                2,
                self.max_blocks_per_seq,
            ),
            BAD_PAGE_INDEX,
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
            device="cpu",
        )
        self._host_staging = {}
        self._init_batch_page_plan()

    def _init_batch_page_plan(self) -> None:
        """Every per-layer table a batch reads, converted together like DeepSeek-V4's
        sliding block tables: local SWA layers first (``pp_layers`` order), then each KV
        owner's GLOBAL and compressor tables. Each params row is (pool, scale, layer
        offset, scratch pool, scratch pages per block)."""
        specs = [
            (layer, CSA2CacheRole.SWA) for layer in self.pp_layers if self.has_swa_cache(layer)
        ]
        for role in (
            CSA2CacheRole.GLOBAL,
            CSA2CacheRole.COMPRESSOR_KV,
            CSA2CacheRole.COMPRESSOR_SCORE,
        ):
            specs += [
                (owner, role)
                for owner in self.layout.kv_source_layer_ids
                if self._cache_role_key(owner, role) in self._layer_roles
            ]
        params = []
        for layer_idx, role in specs:
            layer_id = self._cache_layer_id(layer_idx, role)
            converter = self._get_page_index_converter(layer_id, role.role)
            if converter.expansion != 1:
                raise ValueError("CSA2 page tables require unit page expansion")
            offset = converter.layer_offset if role.index_mode == PageIndexMode.PER_LAYER else 0
            params.append(
                [
                    self.layer_to_pool_mapping_dict[layer_id],
                    converter.scale,
                    offset,
                    0,
                    converter.scratch_pages_per_block,
                ]
            )
        params = np.asarray(params, dtype=np.int32).reshape(-1, 5)
        self._scratch_pools, params[:, 3] = np.unique(params[:, 0], return_inverse=True)
        if -(-self.max_seq_len // self.tokens_per_block) > self.max_blocks_per_seq:
            raise ValueError("CSA2 page tables must cover max_seq_len")
        device = torch.device("cuda", torch.cuda.current_device())
        capacity = self.host_kv_cache_block_offsets.shape[1]
        self._batch_page_index = {spec: i for i, spec in enumerate(specs)}
        self._batch_page_params = params
        self._batch_page_pools = {}
        # Initialize on the stream every consumer runs on: the manager is
        # built on the default stream while forwards run on the non-blocking
        # execution stream, which would not wait for this work.
        with torch.cuda.stream(self._stream):
            self._device_batch_page_params = torch.from_numpy(params).to(device)
            self._device_batch_page_tables = torch.full(
                (len(specs), capacity, self.max_blocks_per_seq),
                BAD_PAGE_INDEX,
                dtype=torch.int32,
                device=device,
            )
            # Two device copies of the host base rows, so a batch's copy never
            # overwrites the rows the previous batch's conversion may still read.
            self._device_base_rows = [
                torch.empty(
                    (self.num_pools, capacity, self.max_blocks_per_seq),
                    dtype=torch.int32,
                    device=device,
                )
                for _ in range(2)
            ]
            self._device_batch_rows = torch.empty(capacity, dtype=torch.int32, device=device)
        self._base_rows_slot = 0
        self._base_rows_read = [torch.cuda.Event(), torch.cuda.Event()]
        self._base_rows_stream = private_stream(device.index, "csa2_copy")
        self._device_scratch = {}
        self._batch_conversion_inputs = None
        # Per table row, the columns that may differ from BAD_PAGE_INDEX: of the
        # tables every conversion writes, and of the SWA tables when they stay here.
        self._table_live = np.zeros(capacity, dtype=np.int64)
        self._swa_table_live = np.zeros(capacity, dtype=np.int64)

    def copy_batch_block_offsets(
        self, dst_tensor, request_ids, beam_width, num_contexts, num_seqs, max_blocks=None
    ):
        if self._deferred_block_offsets is not None:
            self._deferred_block_offsets.append(
                (dst_tensor, request_ids, beam_width, num_contexts, num_seqs, max_blocks)
            )
            return
        # Invalidate before any validation/fill/copy can fail.
        self._swa_publication = None
        if beam_width != 1:
            raise ValueError("CSA2 supports beam width one")
        # Every per-layer table of the batch is converted at once, the SWA
        # tables straight into dst_tensor; prepare_csa2 reuses the rest. The
        # rest of dst_tensor, padding rows up to num_seqs included, is
        # BAD_PAGE_INDEX.
        # Context-only caches keep SWA for a prefix of the layers only.
        swa_layers = sum(self.has_swa_cache(layer) for layer in self.pp_layers)
        count = len(request_ids)
        width = min(dst_tensor.shape[-1], self.max_blocks_per_seq)
        dst_tensor[:, count:].fill_(BAD_PAGE_INDEX)
        dst_tensor[swa_layers:, :count].fill_(BAD_PAGE_INDEX)
        dst_tensor[:swa_layers, :count, 1:].fill_(BAD_PAGE_INDEX)
        dst_tensor[:swa_layers, :count, 0, width:].fill_(BAD_PAGE_INDEX)
        self.compute_batch_page_tables(
            request_ids, num_contexts, swa_out=dst_tensor[:swa_layers, :count, 0, :width]
        )
        if self._swa_publication_token is not None:
            self._swa_publication = (
                dst_tensor,
                tuple(request_ids),
                num_contexts,
                width,
                (dst_tensor.data_ptr(), dst_tensor.shape, dst_tensor.stride()),
            )

    @property
    def blocks_in_primary_pool(self):
        return self.impl.get_page_index_upper_bound(
            self._cache_layer_id(self.pp_layers[0], CSA2CacheRole.SWA), CSA2CacheRole.SWA.role
        )

    def get_num_free_blocks(self):
        reserved = set(self.kv_cache_map) & _RESERVED_REQUEST_IDS
        assert not set(self.kv_cache_map) - reserved, (
            "Capacity query requires an empty cache manager"
        )
        reserved_pages = sum(int(self.kv_cache_map[key].num_blocks) for key in reserved)
        return (
            max(
                self.impl.get_page_index_upper_bound(
                    self._cache_layer_id(layer, CSA2CacheRole.SWA), CSA2CacheRole.SWA.role
                )
                for layer in self.pp_layers
                if self.has_swa_cache(layer)
            )
            - reserved_pages
        )

    def get_cache_bytes_per_token(self):
        return sum(
            (MAIN_ROW_BYTES + INDEX_ROW_BYTES) // self.layout.compress_ratios[owner]
            for owner in self.layout.kv_source_layer_ids
        )

    def get_max_resource_count(self):
        return int(self.impl.get_quota(GPU_LEVEL))

    def get_needed_resource_to_completion(self, request):
        context = request.is_context_init_state
        tokens = request.prompt_len + self.num_extra_kv_tokens
        if not context:
            tokens += request.max_new_tokens
        pages = (max(0, tokens) + self.tokens_per_block - 1) // self.tokens_per_block
        sizes, windows = self._get_runtime_cache_size_layer_components()
        return sum(
            size
            * self.tokens_per_block
            * (
                pages
                if context or window is None
                else min(pages, (window + self.tokens_per_block - 1) // self.tokens_per_block + 1)
            )
            for size, window in zip(sizes, windows)
        )

    def get_disagg_role_mapper_kinds(self):
        return {
            Role.ALL: MapperKind.REPLICATED,
            **{role.role: MapperKind.REPLICATED for role in CSA2CacheRole},
        }

    def get_disagg_pool_view_partition(self, layer_id: LayerId, role: DataRole):
        """Keep remote-tail SWA independently selectable in the transfer layout."""
        if not self.bounded_replay_on_generation or role != CSA2CacheRole.SWA.role:
            return None
        return int(layer_id)

    def remote_tail_excluded_pool_views(self, page_table, split_layer: int):
        """Return decoder/draft SWA views that generation rebuilds locally."""
        excluded = set()
        for group_idx, group in enumerate(page_table.layer_groups):
            for pool_idx, view in enumerate(group.pool_views):
                identities = {
                    identity
                    for entry in view.buffer_entries
                    for (layer_id, role), identity in self._physical_roles.items()
                    if layer_id == LayerId(int(entry["local_layer_id"]))
                    and str(role) in view.pool_role
                }
                if not identities:
                    continue
                transfer = {
                    cache_role != CSA2CacheRole.SWA or model_layer < split_layer
                    for model_layer, _ratio, cache_role in identities
                }
                if len(transfer) != 1:
                    raise ValueError(
                        f"CSA2 pool view ({group_idx}, {pool_idx}) mixes encoder and decoder state"
                    )
                if not transfer.pop():
                    excluded.add((group_idx, pool_idx))
        return excluded

    def _fill_fresh_kv_pages(self, request_id: int) -> None:
        """Fill actual CSA2 buffers without treating model IDs as physical IDs.

        The converted tables include each role's layer offset and scratch
        pages. Track presence by logical page ordinal, so relocating an
        existing page never overwrites its restored contents.
        """
        if self._fresh_page_fill is None:
            return
        cache = self.kv_cache_map.get(request_id)
        if cache is None:
            return
        committed = int(cache.num_committed_tokens or 0)
        protected_blocks = (committed + self.tokens_per_block - 1) // self.tokens_per_block
        state = self._fresh_pages_filled.setdefault(request_id, {})
        filled = 0
        for (physical_layer, data_role), (layer, _ratio, role) in self._physical_roles.items():
            converter = self._get_page_index_converter(physical_layer, data_role)
            pages = np.asarray(self.get_cache_indices(request_id, layer, role), dtype=np.int64)
            key = (physical_layer, data_role)
            previous = state.get(key)
            state[key] = pages.copy()
            protected = protected_blocks * converter.expansion
            fresh = pages != BAD_PAGE_INDEX
            fresh[:protected] = False
            if previous is not None:
                common = min(previous.size, pages.size)
                fresh[:common] &= previous[:common] == BAD_PAGE_INDEX
            if not fresh.any():
                continue
            buffer = self.get_buffers(layer, role)
            indices = np.unique(pages[fresh]).tolist()
            if not _fill_kv_pages(buffer, indices, self._fresh_page_fill):
                raise ValueError(f"CSA2 fresh-page fill does not support {buffer.dtype}")
            filled += len(indices)
        if filled:
            torch.cuda.synchronize()
            if not self._fresh_fill_announced:
                self._fresh_fill_announced = True
                logger.warning(
                    f"CSA2: TRTLLM_KV_FRESH_PAGE_FILL={self._fresh_page_fill} "
                    f"first fill covered {filled} pages for request {request_id}"
                )

    def _iter_guard_candidate_buffers(self) -> Iterator[tuple[int, torch.Tensor]]:
        for layer in self.pp_layers:
            if self.has_swa_cache(layer):
                yield layer, self.get_buffers(layer)

    def _iter_cache_buffers_for_invalid_check(self) -> Iterator[tuple[int, torch.Tensor]]:
        for model_layer, _ratio, role in self._physical_roles.values():
            guard_key = model_layer if role == CSA2CacheRole.SWA else -1
            yield guard_key, self.get_buffers(model_layer, role)

    def check_invalid_values_in_kv_cache(self, fill_with_zero: bool = False) -> bool:
        has_invalid_values = super().check_invalid_values_in_kv_cache(fill_with_zero)
        if fill_with_zero and self._guard_page_by_layer and self._guard_page_value is not None:
            # SWA views overlap, so restore all guards after every pool has been cleared.
            for model_layer, page in self._guard_page_by_layer.items():
                _fill_kv_pages(self.get_buffers(model_layer), [page], self._guard_page_value)
            torch.cuda.synchronize()
        return has_invalid_values

    @classmethod
    def get_cache_size_per_token(
        cls,
        model_config,
        mapping,
        *,
        tokens_per_block,
        max_batch_size=0,
        max_num_tokens=0,
        spec_config=None,
        kv_cache_config=None,
        **kwargs,
    ):
        layout = CSA2Layout.from_hf_config(model_config.pretrained_config)
        if mapping.pp_size != 1 or mapping.cp_size != 1:
            raise ValueError("CSA2 cache sizing requires PP=CP=1")
        from tensorrt_llm._torch.speculative import draft_prompt_lookahead

        reserve = getattr(spec_config, "max_draft_len", 0)
        if kv_cache_config is not None and kv_cache_config.enable_block_reuse:
            reserve += draft_prompt_lookahead(spec_config) or 0
        num_layers = kwargs.get("num_layers") or model_config.get_num_attention_layers()
        limit = getattr(model_config, "extra_attrs", {}).get("csa2_context_swa_layer_limit")
        _validate_context_swa_layer_limit(limit, num_layers)
        capture_window = _get_replay_capture_window(
            layout,
            num_layers,
            mapping,
            kv_cache_config,
            model_config.pretrained_config,
            spec_config,
        )
        replay_window = bounded_replay_tokens(capture_window) if capture_window else 0
        encoder_replay = encoder_replay_enabled(
            max_num_tokens=max_num_tokens or getattr(model_config, "max_num_tokens", None),
            replay_window=max(layout.window_size, replay_window),
        )
        sizes, windows, scratch_groups = _cache_size_layer_components(
            layout,
            range(num_layers),
            replay_window,
            encoder_replay,
            reserve,
            limit,
            remote_tail=bool(
                getattr(model_config, "extra_attrs", {}).get("bounded_replay_on_generation", False)
            ),
        )
        _, headroom = _get_generation_kv_capacity(
            spec_config, is_draft=kwargs.get("is_draft", False)
        )
        context_size, generation_size, generation_fixed = _estimate_cache_size_components(
            sizes,
            windows,
            tokens_per_block,
            scratch=bool(
                kv_cache_config is not None
                and kv_cache_config.enable_swa_scratch_reuse
                and not kwargs.get("is_draft", False)
            ),
            generation_capacity_headroom=headroom,
            scratch_group_keys=scratch_groups,
        )
        return generation_size, (
            generation_fixed * max_batch_size + (context_size - generation_size) * max_num_tokens
        )
