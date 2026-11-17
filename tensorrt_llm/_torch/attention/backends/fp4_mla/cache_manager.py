# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KV Cache Manager V2 specialization for NVFP4 MLA cache state."""

import math
import os
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.disaggregation.resource.page import MapperKind, RoleLayout
from tensorrt_llm._torch.kimi_k3_cache_policy import get_kimi_k3_fp8_kv_layer_ids
from tensorrt_llm._torch.pyexecutor.kv_cache.fp4_mla_cache_mixin import (
    Fp4MlaPageTableSpec,
    Fp4MlaV2CacheLayoutMixin,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType, KVCacheManager
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import DataRole, KVCacheManagerConfig, ReusePolicy

from .config import (
    _FP4_MLA_CUTEDSL_BACKEND,
    _FP4_MLA_K_RESIDUAL_BACKENDS,
    FP4_BLOCK_SIZE,
    FP4_MLA_K_RESIDUAL_DIM,
    FP4_MLA_TOKENS_PER_BLOCK,
    HP_BLOCK_SIZE,
    _fp4_mla_attention_backend,
    _fp4_mla_cutedsl_fused_v_transpose_enabled,
)
from .layout import get_fp4_mla_v_scale_pool_size

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig, KvCacheConfig

# The serving launcher publishes the worker role here, and every MPI rank of
# the worker inherits it.
_DISAGG_ROLE_ENV = "TRTLLM_DISAGG_ROLE"

# Sequence length the KV cache manager V2 core assumes for pool sizing when
# nothing configures one.
_FP4_MLA_DEFAULT_TYPICAL_SEQ_LEN = 2048


def is_fp4_mla_context_only_cache(is_disagg: bool) -> bool:
    """Whether this worker keeps a single-copy FP4 MLA cache.

    A disaggregated context worker never decodes, so it omits the
    generation-only V scales and packed V; the generation worker rebuilds
    them from the transferred K and K scales.
    """
    return bool(is_disagg) and os.getenv(_DISAGG_ROLE_ENV) == "context"


class Fp4MlaV2CacheLayoutPolicy(Fp4MlaV2CacheLayoutMixin):
    """FP4 MLA storage policy for a KV cache manager V2 lifecycle.

    K, K block scales, V scales, and optional packed V share one full-history
    lifecycle. Each model layer also has a virtual sliding layer whose page is
    a compact ``16 + max_rewind`` BF16 ring. The HP role therefore follows V2
    allocation, rewind, prefix, and disaggregated-transfer lifecycles without
    storing BF16 values for the full sequence.

    ``Fp4MlaKVCacheManagerV2`` applies this policy directly for pure MLA
    models. Its block reuse claims only whole ``tokens_per_block`` prefixes,
    so a reused request starts its context on a fresh 16-token quantization
    tile and never reads the HP ring of the shared prefix. Its HP lifecycle is
    therefore request-private: it does not constrain prefix matching, and the
    radix tree retains only the full-history FP4 pages.

    Hybrid linear-attention managers inherit the layout itself
    (``Fp4MlaV2CacheLayoutMixin``) and call ``configure_manager``,
    ``_build_cache_config`` and ``finalize_manager`` of this policy on their
    own instance, next to their SSM lifecycle. Such a manager keeps its own
    disaggregation declarations and merges the policy's attention roles into
    them (``DISAGG_ROLE_MAPPER_KINDS``, ``DISAGG_IGNORED_ROLES``,
    ``_fp4_mla_disagg_global_layer_id``, ``_fp4_mla_disagg_role_layouts``), so
    the recurrent-state roles keep their own transfer mappings.
    """

    uses_fp4_mla_cache_layout: bool = True
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
    # A one-model draft layer at position i reads prompt token i + 1, and the
    # generic reuse-match backoff recomputes that many tokens past a matched
    # block boundary. The resulting context start is not tile aligned, which
    # FP4 MLA cannot append to, so FP4 MLA claims whole blocks instead. The
    # draft KV of the last reused position may then come from a different
    # continuation, which affects draft acceptance only.
    _supports_reuse_match_backoff = False
    # With whole-block claims (_context_reuse_tokens) the page a generation
    # step appends to is never a reused one: the request allocated it, and its
    # earlier rows carry the packed-V sidecar of the request's own steps. A
    # generation step may therefore write packed V directly with block reuse
    # on (cache_update). Hybrid managers do not inherit this policy attribute.
    fp4_mla_reuse_claims_whole_blocks = True

    def __init__(self, *args, **kwargs) -> None:
        kv_cache_config = args[0] if args else kwargs.get("kv_cache_config")
        dtype = kwargs.get("dtype", DataType.HALF)
        kv_cache_type = args[1] if len(args) > 1 else kwargs.get("kv_cache_type")
        kv_cache_config = self.configure_manager(
            self,
            kv_cache_config,
            kv_cache_type,
            dtype=dtype,
            tokens_per_block=kwargs.get("tokens_per_block"),
            head_dim=kwargs.get("head_dim"),
            pretrained_config=kwargs.get("pretrained_config"),
            spec_config=kwargs.get("spec_config"),
            is_disagg=kwargs.get("is_disagg", False),
            mapping=kwargs.get("mapping"),
        )
        if args:
            args = (kv_cache_config, *args[1:])
        else:
            kwargs["kv_cache_config"] = kv_cache_config

        super().__init__(*args, **kwargs)
        self.finalize_manager(self)

    @staticmethod
    def configure_manager(
        manager,
        kv_cache_config,
        kv_cache_type,
        *,
        dtype: DataType,
        tokens_per_block: Optional[int],
        head_dim: Optional[int],
        pretrained_config=None,
        spec_config=None,
        is_disagg: bool = False,
        mapping: Optional[Mapping] = None,
    ):
        """Validate and install FP4 MLA layout state before V2 construction."""
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

        manager.mla_v_scale_head_dim = int(
            getattr(pretrained_config, "kv_lora_rank", head_dim - FP4_MLA_K_RESIDUAL_DIM)
        )
        if not 0 < manager.mla_v_scale_head_dim < head_dim:
            raise ValueError(
                "FP4 MLA V2 requires kv_lora_rank in (0, head_dim), got "
                f"{manager.mla_v_scale_head_dim} and {head_dim}."
            )

        manager._fp8_mla_global_layer_ids = get_kimi_k3_fp8_kv_layer_ids(
            pretrained_config, nvfp4_enabled=True, mapping=mapping
        )
        if manager._fp8_mla_global_layer_ids:
            logger.info(
                "Kimi K3 MLA layers using FP8 KV-cache fallback: "
                f"{sorted(manager._fp8_mla_global_layer_ids)}."
            )
        manager._fp4_mla_storage_backend = _fp4_mla_attention_backend()
        if manager._fp4_mla_storage_backend not in _FP4_MLA_K_RESIDUAL_BACKENDS:
            raise ValueError(
                "Fp4MlaKVCacheManagerV2 supports only the triton and cutedsl "
                f"backends, got {manager._fp4_mla_storage_backend!r}."
            )
        manager.fp4_mla_k_residual_dim = (
            FP4_MLA_K_RESIDUAL_DIM
            if manager._fp4_mla_storage_backend in _FP4_MLA_K_RESIDUAL_BACKENDS
            else 0
        )
        manager.fp4_mla_context_only_cache = is_fp4_mla_context_only_cache(is_disagg)
        if manager.fp4_mla_context_only_cache:
            logger.info(
                "FP4 MLA KV cache on a disaggregated context worker: omitting "
                "the generation-only V scales and packed V."
            )
        manager.mla_v_head_dim = (
            manager.mla_v_scale_head_dim
            if manager._fp4_mla_storage_backend == _FP4_MLA_CUTEDSL_BACKEND
            and not _fp4_mla_cutedsl_fused_v_transpose_enabled()
            and not manager.fp4_mla_context_only_cache
            else None
        )
        max_rewind_len = int(spec_config.tokens_per_gen_step - 1) if spec_config else 0
        manager._fp4_mla_hp_pool_size = HP_BLOCK_SIZE + max_rewind_len
        manager._fp4_mla_view_cache = {}
        manager._attention_cache_layout_policy = Fp4MlaV2CacheLayoutPolicy
        # Turns on the layout hooks the manager inherits from Fp4MlaV2CacheLayoutMixin.
        manager.uses_fp4_mla_cache_layout = True
        return kv_cache_config

    @staticmethod
    def finalize_manager(manager) -> None:
        """Validate allocated pools and initialize derived FP4 state."""
        Fp4MlaV2CacheLayoutMixin._validate_fp4_mla_layer_groups(manager)
        # Partial pages leave unused V-scale tiles untouched while the
        # fixed-width PV path can load the complete scale page. Match V1's
        # deterministic initialization before any warmup or graph capture.
        v_scale_pool = Fp4MlaV2CacheLayoutMixin.get_mla_v_scale_pool_base(manager)
        if v_scale_pool is not None:
            with torch.cuda.stream(manager._stream):
                v_scale_pool.zero_()
            manager._stream.synchronize()

    def _build_cache_config(self, config: KVCacheManagerConfig) -> KVCacheManagerConfig:
        # The pure MLA manager resumes a reused prefix only on a block
        # boundary, where no HP ring content is needed. A hybrid manager
        # composing this policy keeps the HP ring with its recurrent snapshots.
        hp_reuse_policy = (
            ReusePolicy.PRIVATE
            if isinstance(self, Fp4MlaV2CacheLayoutPolicy)
            else ReusePolicy.REQUIRED
        )
        return Fp4MlaV2CacheLayoutMixin._apply_fp4_mla_cache_config(
            self, config, hp_reuse_policy=hp_reuse_policy
        )

    @property
    def blocks_in_primary_pool(self) -> int:
        return self._role_encoded_page_capacity(self._cache_manager_layer_ids[0], Role.KEY)

    def get_disagg_role_mapper_kinds(self) -> dict[DataRole, MapperKind]:
        return {Role.ALL: MapperKind.NHD, **self.DISAGG_ROLE_MAPPER_KINDS}

    def get_disagg_role_layouts(self) -> dict[DataRole, RoleLayout]:
        return {**super().get_disagg_role_layouts(), **self._fp4_mla_disagg_role_layouts()}

    def get_disagg_ignored_roles(self) -> frozenset[DataRole]:
        return self.DISAGG_IGNORED_ROLES

    def get_disagg_global_layer_ids(self, layer_group_id: int) -> list[int]:
        result = []
        for layer_id in self.impl.layer_grouping[layer_group_id]:
            global_layer_id = self._fp4_mla_disagg_global_layer_id(int(layer_id))
            if global_layer_id is None:
                raise ValueError(f"Unknown FP4 MLA V2 internal layer {int(layer_id)}.")
            result.append(global_layer_id)
        return result

    def _get_typical_seq_len(self, kv_cache_config: "KvCacheConfig") -> int:
        """Size the pools from a whole typical batch, also without ``avg_seq_len``.

        The quota is split between lifecycles whose slots differ in size by
        orders of magnitude: the full-history pool of the MLA layers and the
        per-request HP rings. Without a typical length the core derives the
        split from a single 2048-token request and rounds each pool of it up
        to the allocation granularity first, which hands the HP rings a share
        of the quota they never use. A batch of such requests, one per
        resident sequence plus a context chunk, keeps the 2048-token default
        and makes the rounding negligible.
        """
        if kv_cache_config.avg_seq_len is not None:
            return kv_cache_config.avg_seq_len
        return min(self.max_seq_len, _FP4_MLA_DEFAULT_TYPICAL_SEQ_LEN)

    def _context_reuse_tokens(self, req: "LlmRequest", reuse_limit: Optional[int] = None):
        """Restrict prefix matching to whole blocks of the recomputed prompt.

        The radix tree can also hold a request's trailing partial block, and an
        exact match of that block would end the reused prefix inside a 16-token
        quantization tile. Capping the lookup at the last block boundary before
        the final prompt token keeps every claimed prefix block aligned, so the
        first context token always starts a fresh FP4 tile.
        """
        num_tokens = len(self._reuse_token_source(req))
        aligned_limit = max(num_tokens - 1, 0) // self.tokens_per_block * self.tokens_per_block
        if reuse_limit is not None:
            aligned_limit = min(
                aligned_limit, reuse_limit // self.tokens_per_block * self.tokens_per_block
            )
        return KVCacheManagerV2._context_reuse_tokens(self, req, aligned_limit)

    def get_cache_bytes_per_token(self) -> int:
        sizes, _ = self._get_runtime_cache_size_layer_components()
        return sum(sizes)

    @staticmethod
    def get_cache_size_per_token(
        model_config: "ModelConfig",
        mapping: Mapping,
        num_layers: Optional[int] = None,
        *,
        tokens_per_block: int,
        spec_config: Optional["DecodingBaseConfig"] = None,
        max_batch_size: Optional[int] = None,
        **kwargs: object,
    ) -> tuple[int, int]:
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
        context_only = is_fp4_mla_context_only_cache(bool(kwargs.get("is_disagg", False)))
        v_scale_page = (
            0
            if context_only
            else get_fp4_mla_v_scale_pool_size(config.kv_lora_rank, tokens_per_block)
        )
        per_layer = (
            math.ceil(storage_head_dim / 2)
            + math.ceil(storage_head_dim / FP4_BLOCK_SIZE)
            + math.ceil(v_scale_page / tokens_per_block)
        )
        if (
            not context_only
            and backend == _FP4_MLA_CUTEDSL_BACKEND
            and not _fp4_mla_cutedsl_fused_v_transpose_enabled()
        ):
            per_layer += config.kv_lora_rank // 2
        local_layers = KVCacheManager._resolve_num_attention_layers(
            model_config, mapping, num_layers
        )
        fp8_layer_ids = get_kimi_k3_fp8_kv_layer_ids(config, nvfp4_enabled=True, mapping=mapping)
        if fp8_layer_ids:
            local_global_layers = set(mapping.pp_layers(int(config.num_hidden_layers)))
            num_fp8_layers = len(fp8_layer_ids & local_global_layers)
            if num_fp8_layers > local_layers:
                raise ValueError(
                    "Kimi K3 FP8 KV-cache fallback layer count exceeds the local MLA layer count."
                )
        else:
            num_fp8_layers = 0
        num_fp4_layers = local_layers - num_fp8_layers
        rewind = int(spec_config.tokens_per_gen_step - 1) if spec_config else 0
        hp_bytes = (HP_BLOCK_SIZE + rewind) * logical_head_dim * 2 * num_fp4_layers
        max_batch_size = int(max_batch_size or 0)
        cache_bytes = (
            per_layer * num_fp4_layers
            + logical_head_dim
            * torch.empty((), dtype=torch.float8_e4m3fn).element_size()
            * num_fp8_layers
        )
        # Every resident sequence keeps one native HP ring page per local
        # layer across pipeline stages; GEN slot preemption pauses residents
        # in place, so the paused cap stays resident too.
        resident_capacity = max_batch_size * mapping.pp_size + int(
            kwargs.get("gen_preemption_headroom", 0)
        )
        return cache_bytes, hp_bytes * resident_capacity


class Fp4MlaKVCacheManagerV2(Fp4MlaV2CacheLayoutPolicy, KVCacheManagerV2):
    """Compatibility manager applying the FP4 MLA V2 layout policy."""


__all__ = [
    "Fp4MlaKVCacheManagerV2",
    "Fp4MlaPageTableSpec",
    "Fp4MlaV2CacheLayoutMixin",
    "Fp4MlaV2CacheLayoutPolicy",
]
