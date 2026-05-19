# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import copy
import functools
from typing import Callable, Dict, List, Optional, Tuple, Union, final

import torch
import torch.nn as nn
from torch._prims_common import DeviceLikeType

from .._compat import TRTLLM_AVAILABLE, KvCacheConfig

if TRTLLM_AVAILABLE:
    import tensorrt_llm.bindings
    from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import (
        MambaHybridCacheManager,
        MixedMambaHybridCacheManager,
    )
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager, PoolConfiguration
    from tensorrt_llm._utils import torch_dtype_to_binding
    from tensorrt_llm.mapping import Mapping

    CacheTypeCpp = tensorrt_llm.bindings.internal.batch_manager.CacheType
    DataType = tensorrt_llm.bindings.DataType
else:
    # Standalone mode: cache-manager infrastructure not available.
    # CachedSequenceInterface can still be instantiated and used for transforms,
    # but initialize_resources() and other cache-manager methods will raise.
    KVCacheManager = None
    PoolConfiguration = None
    MambaHybridCacheManager = None
    MixedMambaHybridCacheManager = None
    CacheTypeCpp = None
    DataType = None
    Mapping = None
    torch_dtype_to_binding = None

from ..custom_ops.attention_interface import (
    CausalConvResourceHandler,
    KVPagedResourceHandler,
    ResourceHandler,
    ResourceHandlerDict,
    SequenceInfo,
    SpecCausalConvResourceHandler,
    SpecSSMResourceHandler,
    SSMResourceHandler,
    StateResourceHandler,
)
from ..distributed.common import all_gather_object, get_world_size
from ..distributed.common import is_initialized as is_distributed_initialized
from ..utils.cuda_mem_tracker import bytes_to, get_mem_info
from ..utils.logger import ad_logger


def with_pre_callback(method, callback):
    """Wrap method to call callback before the original method."""

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        callback()
        return method(*args, **kwargs)

    return wrapper


@final
class CachedSequenceInterface:
    """An interface responsible for maintaining information about sequences and their caches.

    This class is the single source of truth for sequence and cache configuration. It creates
    SequenceInfo internally, ensuring that tokens_per_block and other fields from KvCacheConfig
    are always consistent.
    """

    def __init__(
        self,
        max_seq_len: int,
        max_batch_size: int,
        max_num_tokens: int,
        device: Optional[DeviceLikeType] = None,
        kv_cache_config: Optional[KvCacheConfig] = None,
        vocab_size_padded: Optional[int] = None,
        spec_config=None,
        requires_uniform_kv_caches: bool = False,
    ) -> None:
        """Initialize the CachedSequenceInterface.

        Args:
            max_seq_len: Maximum sequence length including input and generated tokens.
            max_batch_size: Maximum number of sequences (requests) that can be processed.
            max_num_tokens: Maximum total tokens across all sequences.
            device: Target device for tensors. Defaults to "cuda".
            kv_cache_config: KV cache configuration. If None, uses default KvCacheConfig.
            vocab_size_padded: Padded vocabulary size of the model.
            spec_config: Speculative decoding configuration. Used to set num_extra_kv_tokens,
                max_draft_len, max_total_draft_tokens on KVCacheManager after creation.
            requires_uniform_kv_caches: Whether all KV layers must use a single managed KV
                cache mapping. When True, KV layers incompatible with the managed KV cache
                reference raise during initialization, and managed KV layers must share a
                single page-stride multiplier.
        """
        # TODO (lucaslie): this is somewhat circular/confusing. Here `device` denotes the desired
        # device and not the actual device unlike, e.g., in SequenceInfo. We rely on the attribute
        # here to read the desired device across the inference optimizer pipeline. We should ideally
        # think about a better way to handle this,
        # see https://github.com/NVIDIA/TensorRT-LLM/issues/8371
        self.device = device or "cuda"

        # Initialize kv_cache_config first since SequenceInfo needs tokens_per_block from it
        self._kv_cache_config_original: KvCacheConfig = kv_cache_config or KvCacheConfig()
        self._kv_cache_config_tuned: Optional[KvCacheConfig] = None

        # Create SequenceInfo internally, using tokens_per_block from kv_cache_config
        self.info = SequenceInfo(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            tokens_per_block=self._kv_cache_config_original.tokens_per_block,
            max_num_tokens=max_num_tokens,
            vocab_size_padded=vocab_size_padded,
        )

        self._resource_lookup: ResourceHandlerDict = {}
        self._caches: Dict[str, torch.Tensor] = {}
        # KVCacheManager (or MambaHybridCacheManager) for managed resources
        self._kv_cache_manager: Optional[Union[KVCacheManager, MambaHybridCacheManager]] = None
        # Per-pool sliding window sizes, published by the kvcache transform and
        # consumed by the executor.  Pool index == position in this list, in the
        # same order as the C++ manager's internal pool ordering (i.e. the
        # insertion order of the per-window shape map keys).
        self._kv_group_windows: List[int] = []
        # lookup of unmanaged resources
        self._unmanaged_resources: List[str] = []
        self._spec_config = spec_config
        self._requires_uniform_kv_caches = requires_uniform_kv_caches

        # Propagate spec-dec config into BatchInfo so attention backends can read it
        # via the per-forward batch_info_host tensor without needing the Python config.
        self.info.batch_info.update_max_draft_len(
            spec_config.max_draft_len if spec_config is not None else 0
        )

    @property
    def args(self) -> Tuple[torch.Tensor, ...]:
        """Return all the graph arguments owned by this interface."""
        return tuple(self.named_args.values())

    @property
    def named_args(self) -> Dict[str, torch.Tensor]:
        """Return all the named arguments owned by this interface."""
        return {**self.info.named_args, **self._caches}

    def get_arg(
        self, name: str, truncate: Optional[bool] = None, unflatten: Optional[bool] = None
    ) -> torch.Tensor:
        """Get the argument from the sequence info or caches."""
        if name in self._caches:
            assert not (truncate or unflatten), "truncate or unflatten is not supported for caches"
            return self._caches[name]
        return self.info.get_arg(name, truncate=truncate, unflatten=unflatten)

    def to(self, *args, **kwargs) -> None:
        self.info.to(*args, **kwargs)
        # Only move locally-allocated caches (paged/state caches are managed by cache managers)
        for name, cache in self._caches.items():
            if name in self._unmanaged_resources:
                cache.to(*args, **kwargs)

    def update_kv_cache_config(self, **kwargs) -> None:
        """Update the KVCacheConfig with the given kwargs."""
        for k, v in kwargs.items():
            if k in type(self._kv_cache_config_original).model_fields:
                setattr(self._kv_cache_config_original, k, v)
            else:
                raise ValueError(f"Invalid KVCacheConfig field: {k}")

    def add_resource(self, suffix: str, resource_handler: ResourceHandler) -> str:
        """Add a resource handler to the cache interface.

        Low-level method that adds a single resource. If you have a group of related resources
        that should share the same index, use add_resource_group() instead.
        """
        full_name = f"r{len(self._resource_lookup)}_{suffix}"
        self._resource_lookup[full_name] = resource_handler
        return full_name

    @staticmethod
    def _check_n_groups_constraint(
        ssm: SSMResourceHandler, conv: CausalConvResourceHandler
    ) -> bool:
        """Check if SSM and Conv handlers satisfy the n_groups constraint.

        The MambaCacheManager requires: conv_dim = head_dim * num_heads + 2 * n_groups * d_state
        This method checks if this constraint can be satisfied with integer n_groups >= 0.

        Args:
            ssm: SSM resource handler with num_heads, head_dim, d_state.
            conv: Conv resource handler with conv_dim.

        Returns:
            True if the constraint is satisfied, False otherwise.
        """
        if ssm.d_state == 0:
            # d_state=0 means SSM buffer is empty, any conv_dim works
            return True
        diff = conv.conv_dim - ssm.head_dim * ssm.num_heads
        return diff >= 0 and diff % (2 * ssm.d_state) == 0

    @staticmethod
    def _get_mamba_state_params(
        ssm_ref: Optional[SSMResourceHandler],
        ssm_count: int,
        conv_ref: Optional[CausalConvResourceHandler],
        conv_count: int,
    ) -> Dict[str, Union[int, torch.dtype, None]]:
        """Derive MambaHybridCacheManager parameters from reference state handlers.

        Precondition: If both ssm_ref and conv_ref are provided,
        the n_groups constraint has already been verified to hold.

        Args:
            ssm_ref: Reference SSM handler (defines shape/dtype), or None.
            ssm_count: Number of compatible SSM resources.
            conv_ref: Reference Conv handler (defines shape/dtype), or None.
            conv_count: Number of compatible Conv resources.

        Returns:
            Dictionary of MambaHybridCacheManager constructor parameters.
        """
        # Get SSM parameters (or dummy if not managing SSM)
        if ssm_ref:
            num_heads = ssm_ref.num_heads
            head_dim = ssm_ref.head_dim
            d_state = ssm_ref.d_state
            ssm_dtype = ssm_ref.dtype
        else:
            # Dummy SSM params - d_state=0 means empty tensor (no memory used).
            # MambaCacheManager computes conv_dim = head_dim * num_heads + 2 * n_groups * d_state.
            # When only conv resources exist (e.g. GDN/linear-attention models like Qwen3-Next),
            # we set num_heads = conv_ref.conv_dim so that the formula yields the correct conv_dim
            # (with d_state=0, n_groups=0: conv_dim = head_dim * num_heads = 1 * conv_dim).
            if conv_ref:
                num_heads, head_dim, d_state = conv_ref.conv_dim, 1, 0
            else:
                num_heads, head_dim, d_state = 1, 1, 0
            ssm_dtype = torch.float32

        # Get Conv parameters (or dummy if not managing Conv)
        if conv_ref:
            conv_dim = conv_ref.conv_dim
            d_conv = conv_ref.d_conv
            conv_dtype = conv_ref.dtype
        else:
            # Dummy Conv params - d_conv=1 means state shape (..., 0) (no memory used)
            conv_dim, d_conv = 1, 1
            conv_dtype = torch.float32

        # Determine layer count:
        # - If both buffers used: min() to avoid wasting memory
        # - If only one buffer used: use that buffer's count
        if ssm_count > 0 and conv_count > 0:
            num_layers = min(ssm_count, conv_count)
        else:
            num_layers = max(ssm_count, conv_count)
        assert num_layers > 0, "At least one layer is expected."

        # Derive n_groups from conv_dim constraint (already verified if both are managed)
        # conv_dim = head_dim * num_heads + 2 * n_groups * d_state
        if d_state > 0 and conv_ref:
            n_groups = (conv_dim - head_dim * num_heads) // (2 * d_state)
        else:
            n_groups = 0

        return {
            "mamba_d_state": d_state,
            "mamba_d_conv": d_conv,
            "mamba_num_heads": num_heads,
            "mamba_n_groups": n_groups,
            "mamba_head_dim": head_dim,
            "mamba_num_layers": num_layers,
            "mamba_layer_mask": None,
            "mamba_cache_dtype": conv_dtype,
            "mamba_ssm_cache_dtype": ssm_dtype,
        }

    def _identify_managed_kv_resources(
        self,
    ) -> Tuple[
        ResourceHandlerDict,
        List[PoolConfiguration],
    ]:
        """Identify managed KV resources and the per-pool configurations.

        Each ``KVPagedResourceHandler`` contributes to a pool keyed by its
        effective sliding window (``sliding_window`` if > 0 else
        ``max_seq_len``).  Today, layers sharing a window must share the
        same ``head_dim`` and ``dtype``; the returned list carries one
        ``PoolConfiguration`` per window.  The list shape is deliberately
        flat -- future multi-pool-per-window cases append additional
        entries with the same ``window_size`` rather than nesting shape
        info under a window key.  Insertion order fixes the pool index
        used by the C++ ``mLayerToWindowSize`` routing.

        Returns:
            Tuple of:
                - kv_managed: ResourceHandlerDict -- every KVPagedResourceHandler
                  in registration order; passed to the C++ ctor as the layer
                  list.
                - pool_configurations: List[PoolConfiguration] -- one entry
                  per pool, carrying (window_size, head_dim, dtype).

        Raises:
            RuntimeError: if two layers share an effective window but disagree
                on head_dim or dtype, or if
                ``self._requires_uniform_kv_caches`` is True and more than one
                distinct pool is present.
        """
        kv_managed: ResourceHandlerDict = {}
        pool_by_window: Dict[int, PoolConfiguration] = {}

        max_seq_len = self.info.max_seq_len

        for name, handler in self._resource_lookup.items():
            if not isinstance(handler, KVPagedResourceHandler):
                continue
            # Effective window: full-attention layers (sliding_window == 0) use
            # max_seq_len so the C++ side gets a single concrete window key.
            effective_window = handler.sliding_window if handler.sliding_window > 0 else max_seq_len

            kv_managed[name] = handler

            handler_dtype = torch_dtype_to_binding(handler.dtype)
            existing = pool_by_window.get(effective_window)
            if existing is not None:
                if existing.head_dim != handler.head_dim:
                    raise RuntimeError(
                        f"KV layer {name} has head_dim={handler.head_dim} but window "
                        f"{effective_window} already has head_dim={existing.head_dim}. "
                        "The C++ KVCacheManager keys pools by window, so within a window all "
                        "layers must share head_dim. Place mixed-shape layers in "
                        "distinct windows (e.g. via sliding_window)."
                    )
                if existing.dtype != handler_dtype:
                    raise RuntimeError(
                        f"KV layer {name} has dtype={handler.dtype} but window "
                        f"{effective_window} already has dtype={existing.dtype}. "
                        "The C++ KVCacheManager keys pools by window, so within a window all "
                        "layers must share dtype. Place mixed-dtype layers in distinct "
                        "windows (e.g. via sliding_window)."
                    )
            else:
                pool_by_window[effective_window] = PoolConfiguration(
                    window_size=effective_window,
                    head_dim=handler.head_dim,
                    dtype=handler_dtype,
                )

        pool_configurations: List[PoolConfiguration] = list(pool_by_window.values())

        # If the runtime requires uniform KV caches (e.g. legacy single-pool
        # path), more than one distinct pool is incompatible.
        if len(pool_configurations) > 1 and self._requires_uniform_kv_caches:
            pools_repr = ", ".join(
                f"window={pc.window_size} head_dim={pc.head_dim} dtype={pc.dtype}"
                for pc in pool_configurations
            )
            raise RuntimeError(
                "KV resources are not uniform: "
                f"{pools_repr}. "
                "This configuration requires all KV caches to share a single pool."
            )

        return kv_managed, pool_configurations

    def _identify_managed_state_resources(
        self,
    ) -> Tuple[
        Optional[SSMResourceHandler],
        list,
        list,
        Optional[CausalConvResourceHandler],
        list,
        list,
    ]:
        """Identify SSM and Conv resources compatible with MambaHybridCacheManager.

        Finds reference handlers for SSM and Conv resources, checks the n_groups
        constraint, and collects all compatible resources for each type.

        Returns:
            Tuple of (ssm_ref, ssm_managed, ssm_spec, conv_ref, conv_managed, conv_spec) where:
            - ssm_ref: Reference SSM handler or None
            - ssm_managed: List of (name, handler) tuples for compatible base SSM resources
            - ssm_spec: List of (name, handler) tuples for compatible speculative SSM resources.
              This is only nonempty when speculative decoding is enabled.
            - conv_ref: Reference Conv handler or None (may be None if the n_groups constraint fails)
            - conv_managed: List of (name, handler) tuples for compatible base Conv resources
            - conv_spec: List of (name, handler) tuples for compatible speculative Conv resources.
              This is only nonempty when speculative decoding is enabled.
        """
        ssm_ref: Optional[SSMResourceHandler] = None
        conv_ref: Optional[CausalConvResourceHandler] = None

        # Find the first base (non-spec) handler of each type as reference.
        for handler in self._resource_lookup.values():
            if isinstance(handler, SSMResourceHandler) and ssm_ref is None:
                ssm_ref = handler
            elif isinstance(handler, CausalConvResourceHandler) and conv_ref is None:
                conv_ref = handler
            if ssm_ref and conv_ref:
                break

        # Check n_groups constraint: conv_dim = head_dim * num_heads + 2 * n_groups * d_state
        # If constraint doesn't hold, only manage SSM (more common); Conv goes to local allocation
        if ssm_ref and conv_ref and not self._check_n_groups_constraint(ssm_ref, conv_ref):
            ad_logger.debug(
                "n_groups constraint not satisfied between SSM and Conv handlers. "
                "Conv resources will be allocated locally."
            )
            conv_ref = None  # Don't manage Conv via cache manager

        # Collect resources compatible with the reference handlers for each managed type,
        # using handler equality to match shape and dtype.
        ssm_managed = [
            (name, handler)
            for name, handler in self._resource_lookup.items()
            if isinstance(handler, SSMResourceHandler) and handler == ssm_ref
        ]
        ssm_spec = [
            (name, handler)
            for name, handler in self._resource_lookup.items()
            if isinstance(handler, SpecSSMResourceHandler)
            and handler == SpecSSMResourceHandler.from_base(ssm_ref)
        ]
        conv_managed = [
            (name, handler)
            for name, handler in self._resource_lookup.items()
            if isinstance(handler, CausalConvResourceHandler) and handler == conv_ref
        ]
        conv_spec = [
            (name, handler)
            for name, handler in self._resource_lookup.items()
            if isinstance(handler, SpecCausalConvResourceHandler)
            and handler == SpecCausalConvResourceHandler.from_base(conv_ref)
        ]

        # When speculative decoding is enabled, the backend must supply matching spec buffers.
        # When it is not enabled, spec buffers may still be registered by the backend (e.g.
        # triton_ssm always registers intermediate_ssm_state_cache) but will not be bound.
        if self._spec_config is not None:
            assert len(ssm_spec) == len(ssm_managed), (
                f"Mismatched SSM spec layer count: expected {len(ssm_managed)}, got {len(ssm_spec)}"
            )
            assert len(conv_spec) == len(conv_managed), (
                f"Mismatched Conv spec layer count: expected {len(conv_managed)}, "
                f"got {len(conv_spec)}"
            )

        return ssm_ref, ssm_managed, ssm_spec, conv_ref, conv_managed, conv_spec

    def _prepare_kv_cache_config(
        self,
        max_tokens: Optional[int],
        kv_managed: ResourceHandlerDict,
    ) -> KvCacheConfig:
        """Prepare and configure KvCacheConfig for cache manager creation.

        Handles deep copy, max_tokens synchronization across ranks, block reuse
        settings, copy_on_partial_reuse validation, and free_gpu_memory_fraction
        normalization.

        ``max_attention_window`` is the per-layer window list (one entry per
        managed KV layer, in the order ``kv_managed`` enumerates them).  The C++
        ctor groups these layers internally by window into pools using its
        ``mLayerToWindowSize`` map.  Full-attention layers report
        ``max_seq_len`` (i.e. their effective window).

        Args:
            max_tokens: Maximum tokens to allocate, or None to use config defaults.
            kv_managed: Dict of KV resources that will be managed by KVCacheManager.

        Returns:
            Configured KvCacheConfig ready for cache manager creation.
        """
        # Make a deep copy of the kv_cache_config to avoid modifying the original object
        kv_cache_config = copy.deepcopy(self._kv_cache_config_original)

        # Build per-layer max_attention_window in kv_managed insertion order.
        # The C++ side groups layers by this value into pools.
        if kv_managed:
            kv_cache_config.max_attention_window = [
                handler.sliding_window if handler.sliding_window > 0 else self.info.max_seq_len
                for handler in kv_managed.values()
            ]
        else:
            kv_cache_config.max_attention_window = None

        # Update kv_cache_config based on max_tokens if provided
        if max_tokens is not None:
            # sync max_tokens across ranks
            if is_distributed_initialized():
                max_tokens_gathered = [None] * get_world_size()
                all_gather_object(max_tokens_gathered, max_tokens)
                max_tokens = min(max_tokens_gathered)
            kv_cache_config.free_gpu_memory_fraction = None
            kv_cache_config.max_tokens = min(kv_cache_config.max_tokens or max_tokens, max_tokens)

        # Check if we should disable block reuse
        is_paged = all(handler.is_paged for handler in self._resource_lookup.values())
        if kv_cache_config.enable_block_reuse and not is_paged:
            kv_cache_config.enable_block_reuse = False
            ad_logger.info(f"Setting {kv_cache_config.enable_block_reuse=} for non-paged models.")

        # Check if we can use copy on partial reuse
        num_non_kv_managed_caches = len(self._caches) - len(kv_managed)
        if (
            kv_cache_config.enable_block_reuse
            and kv_cache_config.copy_on_partial_reuse
            and num_non_kv_managed_caches > 0
        ):
            kv_cache_config.copy_on_partial_reuse = False
            ad_logger.info(
                "Disabling copy_on_partial_reuse: requires all resources to be paged and managed by"
                f" KVCacheManager ({num_non_kv_managed_caches=})."
            )

        # Make sure to set free_gpu_memory_fraction to None if set to 0.0
        # NOTE: KVCacheConfig validator enforces that free_gpu_memory_fraction must be between 0.0
        # and 1.0 but we allow 0.0 to be set to disable resizing (corresponding to None in the
        # manager).
        if kv_cache_config.free_gpu_memory_fraction == 0.0:
            kv_cache_config.free_gpu_memory_fraction = None

        return kv_cache_config

    def _build_kv_cache_kwargs(
        self,
        kv_managed: ResourceHandlerDict,
        kv_cache_config: KvCacheConfig,
        pool_configurations: Optional[List[PoolConfiguration]] = None,
    ) -> Dict:
        """Build common kwargs for KVCacheManager or MambaHybridCacheManager.

        With per-pool ``PoolConfiguration`` carried through to the C++ ctor,
        a single manager can host pools with mixed shapes.  We pass:

        - ``head_dim`` / ``dtype`` as scalar defaults (fall-back values for
          managers that run in uniform-shape mode).  We pick the maximum
          head_dim and the dtype of the first pool so the default is a sane
          upper bound on per-pool memory accounting.
        - ``pool_configurations`` as the per-pool config list; the C++ ctor
          uses it to build pools with the correct shapes.

        Args:
            kv_managed: Dict of KV resources to be managed.
            kv_cache_config: Configured KvCacheConfig.
            pool_configurations: One PoolConfiguration per pool, in pool-index
                order.  Empty / None means uniform shape.

        Returns:
            Dict of kwargs suitable for both KVCacheManager and MambaHybridCacheManager.
        """
        pool_configurations = list(pool_configurations) if pool_configurations else []

        # create arguments first that differ whether we have managed kv caches or not
        kv_cache_kwargs: Dict = {}
        if kv_managed:
            # kv_factor is uniform across the managed set: __init__ asserts
            # kv_factor in {1, 2}, and AutoDeploy only mixes kv_factor=2 layers
            # in the same model today.  We take the first handler's value.
            ref_handler = next(iter(kv_managed.values()))
            kv_cache_type = (
                CacheTypeCpp.SELFKONLY if ref_handler.kv_factor == 1 else CacheTypeCpp.SELF
            )
            # Default head_dim: the largest across all pools, so any layer
            # that falls back to the scalar gets a safe upper bound.
            default_head_dim = (
                max(pc.head_dim for pc in pool_configurations)
                if pool_configurations
                else ref_handler.head_dim
            )
            # Default dtype: dtype of the first pool.
            default_dtype = (
                pool_configurations[0].dtype
                if pool_configurations
                else torch_dtype_to_binding(ref_handler.dtype)
            )
            kv_cache_kwargs.update(
                {
                    "kv_cache_type": kv_cache_type,
                    "num_layers": len(kv_managed),
                    "num_kv_heads": [h.num_kv_heads for h in kv_managed.values()],
                    "head_dim": default_head_dim,
                    "dtype": default_dtype,
                    "pool_configurations": pool_configurations,
                }
            )
        else:
            kv_cache_kwargs.update(
                {
                    "kv_cache_type": CacheTypeCpp.SELF,
                    "num_layers": 1,
                    "num_kv_heads": [1],
                    "head_dim": 1,
                    "dtype": DataType.HALF,
                    "pool_configurations": [],
                }
            )
        # remaining arguments are the same for both cases
        kv_cache_kwargs.update(
            {
                "kv_cache_config": kv_cache_config,
                "tokens_per_block": kv_cache_config.tokens_per_block,
                "max_seq_len": self.info.max_seq_len,
                "max_batch_size": self.info.max_batch_size,
                "mapping": Mapping(),
                "layer_mask": [True] * kv_cache_kwargs["num_layers"],
                "spec_config": self._spec_config,
                # NOTE (lucaslie): we can always run with False here since when we are estimating,
                # we are explicitly setting the max_tokens in which case it's okay to use False here
                # since we don't rely on free_gpu_memory_fraction inside the KVCacheManager. This is
                # similar to _torch.pyexecutor._util.KVCacheCreator, which explicitly estimates the
                # max_tokens outside of the KVCacheManager.
                "is_estimating_kv_cache": False,
            }
        )

        return kv_cache_kwargs

    def _create_and_assign_state_views(
        self,
        kv_cache_kwargs: Dict,
        ssm_ref: Optional[SSMResourceHandler],
        ssm_managed: list,
        ssm_spec: list,
        conv_ref: Optional[CausalConvResourceHandler],
        conv_managed: list,
        conv_spec: list,
    ) -> Tuple[MambaHybridCacheManager, int]:
        """Create MambaHybridCacheManager and assign views for state resources.

        Creates the hybrid cache manager with mamba parameters derived from the reference
        handlers, then retrieves and assigns buffer views for all managed SSM and Conv resources,
        as well as speculative resources if they exist.

        Args:
            kv_cache_kwargs: Base kwargs for cache manager (will be extended with mamba params).
            ssm_ref: Reference SSM handler or None.
            ssm_managed: List of base SSM resources.
            ssm_spec: List of speculative SSM resources.
            conv_ref: Reference Conv handler or None.
            conv_managed: List of base Conv resources.
            conv_spec: List of speculative Conv resources.

        Returns:
            Tuple of (manager, num_managed_mamba_layers).
        """
        # Mamba state params can be derived from reference handlers and number of managed (non-speculative) resources.
        mamba_params = self._get_mamba_state_params(
            ssm_ref, len(ssm_managed), conv_ref, len(conv_managed)
        )
        num_managed_mamba_layers = mamba_params["mamba_num_layers"]

        # Create the hybrid cache manager
        manager = MixedMambaHybridCacheManager(
            **mamba_params,
            **kv_cache_kwargs,
        )

        # Retrieve and assign views for Mamba-managed resources (up to num_managed_mamba_layers).
        for layer_idx in range(num_managed_mamba_layers):
            if ssm_managed:
                ssm_name = ssm_managed[layer_idx][0]
                ssm_view = manager.get_ssm_states(layer_idx)
                assert ssm_view.is_contiguous(), f"Non-contiguous state {ssm_name}"
                self._caches[ssm_name] = ssm_view
            if ssm_spec and self._spec_config is not None:
                spec_ssm_name = ssm_spec[layer_idx][0]
                spec_view = manager.get_intermediate_ssm_states(layer_idx)
                if spec_view is None:
                    raise RuntimeError(
                        f"Intermediate SSM state binding returned no view for {spec_ssm_name}. "
                        "Are we using a backend that supports speculative decoding?"
                    )
                assert spec_view.is_contiguous(), f"Non-contiguous state {spec_ssm_name}"
                self._caches[spec_ssm_name] = spec_view
            if conv_managed:
                conv_name = conv_managed[layer_idx][0]
                conv_view = manager.get_conv_states(layer_idx)
                assert conv_view.is_contiguous(), f"Non-contiguous state {conv_name}"
                self._caches[conv_name] = conv_view
            if conv_spec and self._spec_config is not None:
                spec_conv_name = conv_spec[layer_idx][0]
                spec_view = manager.get_intermediate_conv_states(layer_idx)
                if spec_view is None:
                    raise RuntimeError(
                        f"Intermediate conv state binding returned no view for {spec_conv_name}. "
                        "Are we using a backend that supports speculative decoding?"
                    )
                assert spec_view.is_contiguous(), f"Non-contiguous state {spec_conv_name}"
                self._caches[spec_conv_name] = spec_view

        return manager, num_managed_mamba_layers

    def _assign_kv_cache_views(self, kv_managed: Dict[str, KVPagedResourceHandler]) -> int:
        """Retrieve and assign buffer views for managed KV paged resources.

        ``get_buffers`` on ``self._kv_cache_manager`` is per-layer-aware: it
        reads the layer's head_dim from the underlying C++ pool configuration
        when mixed-shape pools are present (Gemma4-style VSWA), and falls back
        to the manager-level scalar otherwise.  We can therefore use a single
        call regardless of single-pool vs. multi-pool deployments.

        Args:
            kv_managed: Dict of KV resources managed by the cache manager.

        Returns:
            block_offset_multiplier derived from the first KV cache view's strides.
        """
        block_offset_multiplier = 0
        for idx, (name, h) in enumerate(kv_managed.items()):
            view = self._kv_cache_manager.get_buffers(idx, kv_layout=h.kv_layout)
            assert view[0].is_contiguous(), f"Non-contiguous kv cache resource for {name}"
            self._caches[name] = view

            # Compute block_offset_multiplier from the first layer's kv_cache strides.
            # This is stride(0)/stride(1) which equals kv_factor for per-layer views
            # or num_layers*kv_factor for interleaved pools.
            # All layers must share this multiplier if we require uniform KV caches.
            layer_block_offset_multiplier = view.stride(0) // view.stride(1)
            if idx == 0:
                block_offset_multiplier = layer_block_offset_multiplier
            elif (
                layer_block_offset_multiplier != block_offset_multiplier
                and self._requires_uniform_kv_caches
            ):
                raise RuntimeError(
                    f"KV cache layer {name} (idx={idx}) has block_offset_multiplier "
                    f"{layer_block_offset_multiplier} != reference {block_offset_multiplier}. "
                    "This configuration requires uniform KV caches, so all KV layers managed by the "
                    "cache manager must share a single block_offset_multiplier."
                )

        return block_offset_multiplier

    def _allocate_unmanaged_resources(self) -> None:
        """Allocate resources not managed by cache managers.

        Resources that haven't been assigned a tensor (still None) are allocated
        locally via their handler's allocate() method.
        """
        self._unmanaged_resources.clear()
        for name, handler in self._resource_lookup.items():
            if self._caches[name] is None:  # Not yet assigned a tensor
                if isinstance(handler, StateResourceHandler):
                    self._caches[name] = self._allocate_unmanaged_state_resource(handler)
                else:
                    self._caches[name] = handler.allocate(self.info)
                self._unmanaged_resources.append(name)

    def _allocate_unmanaged_state_resource(self, handler: StateResourceHandler) -> torch.Tensor:
        """Allocate state resources in the slot domain used by runtime metadata."""
        max_num_state_slots = self.info.max_num_state_slots
        if TRTLLM_AVAILABLE and isinstance(
            self._kv_cache_manager, (MambaHybridCacheManager, MixedMambaHybridCacheManager)
        ):
            # ADEngine passes Mamba cache indices through slot_idx for every stateful
            # op when a Mamba-hybrid cache manager is present. Mirror the padding
            # slots reserved by that manager for CUDA-graph/spec-decoding dummies.
            max_num_state_slots += self.info.batch_info.get_max_draft_len() + 1

        return torch.empty(
            max_num_state_slots,
            *handler.state_shape,
            device=self.info.device,
            dtype=handler.dtype,
        )

    def _has_swa_window(self, pool_configurations: List[PoolConfiguration]) -> bool:
        """Return True if any pool's window is smaller than max_seq_len (i.e. SWA pool exists)."""
        max_seq_len = self.info.max_seq_len
        return any(pc.window_size < max_seq_len for pc in pool_configurations)

    def _compute_total_token_budget(
        self,
        kv_managed: ResourceHandlerDict,
        pool_configurations: List[PoolConfiguration],
        total_max_tokens: Optional[int],
    ) -> Optional[int]:
        """Compute the total max_tokens budget for the unified KVCacheManager.

        With a single C++ manager hosting all pools, ``BaseKVCacheManager::
        calculateMaxNumBlocks`` does the per-pool split internally using each
        pool's ``PoolConfiguration``.  We just need to compute the total token
        budget; the C++ side derives N (max concurrent sequences) from the
        total byte budget divided by the combined per-sequence cost across
        all pools, then gives each pool N × its per-window tokens.

        We retain a Python-side cap to keep the budget feasible during prefill
        warmup before the C++ side has a chance to validate against
        ``calculateMaxNumBlocks``.

        Args:
            kv_managed: All managed KV layers.
            pool_configurations: Per-pool configurations, used here purely to
                detect whether at least one SWA pool exists.
            total_max_tokens: Caller-provided budget, or None to defer to
                ``free_gpu_memory_fraction``.

        Returns:
            Total max_tokens for the manager, or None to defer to the
            ``free_gpu_memory_fraction`` path.
        """
        # If the caller didn't pass a budget and there's no SWA pool, defer to
        # free_gpu_memory_fraction inside the manager — same as the legacy path.
        if total_max_tokens is None and not self._has_swa_window(pool_configurations):
            return None

        if total_max_tokens is None:
            # If the user already pinned max_tokens via KvCacheConfig, keep it:
            # _prepare_kv_cache_config will preserve that value untouched.
            if self._kv_cache_config_original.max_tokens is not None:
                return None
            # SWA present but no total budget — conservative single-sequence estimate.
            # The C++ side will still bound this against available memory.
            return self.info.max_seq_len

        tpb = self.info.tokens_per_block
        # Cap at max_batch_size × max_seq_len (can't need more than one
        # full sequence per slot during prefill).
        capped = min(total_max_tokens, self.info.max_batch_size * self.info.max_seq_len)
        # Floor: at least one block per sequence for warmup feasibility.
        min_tokens = self.info.max_batch_size * tpb
        return max(capped, min_tokens)

    def _create_kv_cache_manager(self, max_tokens: Optional[int] = None) -> Dict:
        """Create a single KVCacheManager that hosts every KV pool.

        For paged resources (KVPagedResourceHandler):
        - All managed layers are passed to one ``KVCacheManager`` instance.
        - One ``PoolConfiguration`` per pool flows through the
          ``pool_configurations`` ctor kwarg; the C++ side routes each layer
          to the correct pool via its ``mLayerToWindowSize`` map.
        - Cross-pool admission, event flush, disagg transfer, and the shared
          radix tree are all handled in C++.

        For state resources (SSMResourceHandler, CausalConvResourceHandler, StateResourceHandler):
        - SSMResourceHandler maps to MambaHybridCacheManager's ssm_states buffer
        - CausalConvResourceHandler maps to MambaHybridCacheManager's conv_states buffer
        - Generic StateResourceHandler and incompatible typed handlers are allocated locally
        - When both SSM and Conv handlers exist, uses min(ssm_count, conv_count) layers

        Args:
            max_tokens: Maximum number of tokens to allocate. If provided, it will use the min value
                between this value and max_tokens in kv_cache_config.

        Returns:
            Dictionary with cache statistics including max_tokens. The max_tokens value may differ
            from the provided ``max_tokens`` arg for two reasons:
                1. the final number of tokens is synced (min) across ranks
                2. rounding for getting a multiple of tokens_per_block
        """
        # 1. Identify managed resources and per-pool configurations
        kv_managed, pool_configurations = self._identify_managed_kv_resources()

        ssm_ref, ssm_managed, ssm_spec, conv_ref, conv_managed, conv_spec = (
            self._identify_managed_state_resources()
        )
        has_state_resources = ssm_managed or conv_managed

        # 2. Compute the total token budget; the C++ side splits it across pools.
        total_max_tokens = self._compute_total_token_budget(
            kv_managed, pool_configurations, max_tokens
        )

        kv_cache_config = self._prepare_kv_cache_config(total_max_tokens, kv_managed)
        kv_cache_kwargs = self._build_kv_cache_kwargs(
            kv_managed,
            kv_cache_config,
            pool_configurations=pool_configurations,
        )

        # 3. Build the unified manager (single instance covers every pool).
        if not kv_managed and not has_state_resources:
            # Pure dummy manager: no KV layers, no state — used for state-only or
            # cache-less models.  We still need a manager so PyExecutor has a
            # handle to call.
            self._kv_cache_manager = KVCacheManager(**kv_cache_kwargs)
        elif has_state_resources:
            kv_cache_kwargs["max_batch_size"] = self.info.max_num_state_slots
            self._kv_cache_manager, _ = self._create_and_assign_state_views(
                kv_cache_kwargs,
                ssm_ref,
                ssm_managed,
                ssm_spec,
                conv_ref,
                conv_managed,
                conv_spec,
            )
        else:
            self._kv_cache_manager = KVCacheManager(**kv_cache_kwargs)

        ad_logger.info(
            f"KV manager: {len(kv_managed)} layers across "
            f"{len(pool_configurations)} pool(s); "
            f"pool_configurations={pool_configurations}, "
            f"max_attention_window={kv_cache_config.max_attention_window}, "
            f"max_tokens={total_max_tokens}"
        )

        # 4. Store tuned config (mirrors the kv_cache_config used at ctor time).
        self._kv_cache_config_tuned = kv_cache_config

        # 5. Refresh per-pool window list from the finalized manager: KVCacheManager
        # may clamp each window to min(window, max_seq_len) during construction, so
        # the transform-time _kv_group_windows can become stale.  Downstream callers
        # (ad_executor.get_cache_indices(window_size=...)) need the post-clamp
        # values to hit the correct pool key in C++.
        if kv_managed and self._kv_group_windows:
            manager_windows = list(dict.fromkeys(self._kv_cache_manager.max_attention_window_vec))
            if manager_windows and manager_windows != self._kv_group_windows:
                ad_logger.info(
                    f"Refreshing kv_group_windows from manager: "
                    f"{self._kv_group_windows} -> {manager_windows}"
                )
                self._kv_group_windows = manager_windows

        # 6. Assign KV views and update cache information.
        block_offset_multiplier = 0
        if kv_managed:
            block_offset_multiplier = self._assign_kv_cache_views(kv_managed)

        num_blocks = getattr(
            self._kv_cache_manager,
            "blocks_in_primary_pool",
            self._kv_cache_manager.get_max_resource_count(),
        )
        self.info.update_cache_information(
            num_blocks=num_blocks,
            block_offset_multiplier=block_offset_multiplier,
        )

        # 7. Allocate remaining unmanaged resources
        self._allocate_unmanaged_resources()

        # 8. Patch shutdown
        self._kv_cache_manager.shutdown = with_pre_callback(
            self._kv_cache_manager.shutdown,
            self._clear_caches,
        )

        # 8. Compute final token count and cache statistics
        max_resource_count = self._kv_cache_manager.get_max_resource_count()
        max_tokens_final = max_resource_count * self._kv_cache_manager.tokens_per_block

        # 9. Collect statistics of different types of resources
        num_state_total = sum(
            1 for h in self._resource_lookup.values() if isinstance(h, StateResourceHandler)
        )
        num_ssm_base_total = sum(
            1 for h in self._resource_lookup.values() if isinstance(h, SSMResourceHandler)
        )
        num_ssm_spec_total = sum(
            1 for h in self._resource_lookup.values() if isinstance(h, SpecSSMResourceHandler)
        )
        num_ssm_total = num_ssm_base_total + num_ssm_spec_total
        num_conv_base_total = sum(
            1 for h in self._resource_lookup.values() if isinstance(h, CausalConvResourceHandler)
        )
        num_conv_spec_total = sum(
            1
            for h in self._resource_lookup.values()
            if isinstance(h, SpecCausalConvResourceHandler)
        )
        num_conv_total = num_conv_base_total + num_conv_spec_total
        num_state_other = num_state_total - num_ssm_total - num_conv_total

        # Count individual cache buffers owned by the cache manager.
        # Spec buffers are only cache-manager-owned when spec decoding is enabled.
        ssm_managed_count = len(ssm_managed) + (
            len(ssm_spec) if self._spec_config is not None else 0
        )
        conv_managed_count = len(conv_managed) + (
            len(conv_spec) if self._spec_config is not None else 0
        )
        total_managed = len(kv_managed) + ssm_managed_count + conv_managed_count

        paged_total = sum(1 for h in self._resource_lookup.values() if h.is_paged)
        kv_total = sum(
            1 for h in self._resource_lookup.values() if isinstance(h, KVPagedResourceHandler)
        )
        paged_other = paged_total - kv_total

        other_total = len(self._caches) - paged_total - num_state_total

        return {
            "total": len(self._caches),
            "total_managed": total_managed,
            "kv_total": kv_total,
            "kv_managed": len(kv_managed),
            "paged_other": paged_other,
            "ssm_total": num_ssm_total,
            "ssm_managed": ssm_managed_count,
            "conv_total": num_conv_total,
            "conv_managed": conv_managed_count,
            "state_other": num_state_other,
            "other": other_total,
            "max_tokens": max_tokens_final,
        }

    def initialize_resources(self) -> int:
        """Initialize resources - paged/state caches via cache managers, others separately.

        Paged resources are managed by KVCacheManager (or the KV portion of MambaHybridCacheManager).
        State resources are managed by the Mamba portion of MambaHybridCacheManager.
        Other resources are allocated locally as a fallback.

        Returns:
            The number of caches initialized.
        """
        assert not self._caches, "Caches already initialized."
        self.info.to(self.device)

        # Make sure self._caches has the same order as self._resource_lookup
        for name in self._resource_lookup.keys():
            self._caches[name] = None  # Will be set by _create_kv_cache_manager

        # Create unified cache manager (handles both paged and state resources)
        if self.needs_resize() or self._requires_token_estimate():
            max_tokens_estimate = self.info.estimate_cache_tokens_per_forward()
        else:
            # if we don't need a resize, we will just use the original settings in kv_cache_config
            # instead of passing in an overwrite here.
            max_tokens_estimate = None
        cache_stats = self._create_kv_cache_manager(max_tokens=max_tokens_estimate)

        # Log cache statistics summary (format: total/managed)
        s = cache_stats
        ad_logger.info(
            f"Cache stats (total/managed): total={s['total']}/{s['total_managed']}, "
            f"kv={s['kv_total']}/{s['kv_managed']}, "
            f"paged_other={s['paged_other']}, "
            f"ssm={s['ssm_total']}/{s['ssm_managed']}, "
            f"conv={s['conv_total']}/{s['conv_managed']}, "
            f"state_other={s['state_other']}, "
            f"other={s['other']}, "
            f"max_tokens={s['max_tokens']}"
        )

        return len(self._caches)

    def _requires_token_estimate(self) -> bool:
        """Check if our kv_cache_config requires max_tokens to be estimated."""
        needs_max_tokens = self._kv_cache_config_original.free_gpu_memory_fraction in [None, 0.0]
        needs_max_tokens |= not any(handler.is_paged for handler in self._resource_lookup.values())
        return needs_max_tokens and self._kv_cache_config_original.max_tokens is None

    def needs_resize(self) -> bool:
        """Check if we need a resize or not."""
        has_paged = any(handler.is_paged for handler in self._resource_lookup.values())
        return has_paged and self._kv_cache_config_original.free_gpu_memory_fraction not in [
            None,
            0.0,
        ]

    def resize_kv_cache_manager(self, mem_exclude: int = 0) -> None:
        """Shutdown existing caches and recreate with optimal capacity for paged resources.

        Args:
            mem_exclude: Extra memory to exclude from the calculation of optimal capacity.
                This is in bytes and typically the memory reserved for the forward pass.

        This implements the two-phase approach: after running a forward pass during estimation
        to allocate intermediate memory, call this method to recreate the cache manager.

        For multi-pool (dual head_dim): the C++
        ``BaseKVCacheManager::calculateMaxNumBlocks`` distributes the budget
        across pools using the per-window head_dim/dtype overrides — the
        Python side just provides the total token budget.
        """
        if not self.needs_resize():
            return

        # Calculate bytes-per-token for ALL paged resources (across all groups)
        paged_bytes_per_token = sum(
            h.bytes_per_token for h in self._resource_lookup.values() if h.is_paged
        )

        # Calculate total bytes for non-paged (non-resizable) resources
        non_paged_bytes_total = sum(
            cache.numel() * cache.element_size()
            for name, cache in self._caches.items()
            if not self._resource_lookup[name].is_paged
        )

        # Shutdown clears ALL cache views (paged and non-paged)
        self._kv_cache_manager.shutdown()

        # Get current free GPU memory after shutdown
        _, free_mem, *_ = get_mem_info(empty_cache=True)

        # Compute available memory for paged caches
        free_gpu_memory_fraction = self._kv_cache_config_original.free_gpu_memory_fraction
        mem_for_paged_optimal = (
            free_mem - non_paged_bytes_total - mem_exclude
        ) * free_gpu_memory_fraction
        max_tokens_optimal = int(mem_for_paged_optimal // paged_bytes_per_token)

        # Recreate the unified manager — the C++ side splits the total token
        # budget across pools internally based on per-window head_dim/dtype.
        cache_stats = self._create_kv_cache_manager(max_tokens=max_tokens_optimal)
        max_tokens_final = cache_stats["max_tokens"]

        # Log resulting memory information
        total_cache_bytes = mem_for_paged_optimal + non_paged_bytes_total
        ad_logger.info(
            f"Resize mem info: free_mem={bytes_to(free_mem, unit='GB'):.2f}GB, "
            f"free_gpu_memory_fraction={free_gpu_memory_fraction}, "
            f"mem_exclude={bytes_to(mem_exclude, unit='GB'):.2f}GB"
        )
        ad_logger.info(
            f"Final cache mem: max_tokens={max_tokens_final}, "
            f"paged={bytes_to(mem_for_paged_optimal, unit='GB'):.2f}GB, "
            f"non_paged={bytes_to(non_paged_bytes_total, unit='GB'):.2f}GB, "
            f"total={bytes_to(total_cache_bytes, unit='GB'):.2f}GB"
        )

    @property
    def kv_group_windows(self) -> List[int]:
        """Per-pool window sizes, in the order the C++ manager exposes them.

        Pool index == position in this list, matching the order of the
        unified manager's ``pool_configurations`` list.
        """
        return self._kv_group_windows

    def set_kv_groups(self, group_windows: List[int]) -> None:
        """Store per-pool window sizes (called by the kvcache transform).

        ``group_windows[i]`` is the effective window of pool ``i``; the order
        must match the unified ``KVCacheManager``'s pool ordering (i.e., the
        insertion order of the per-window keys).
        """
        self._kv_group_windows = list(group_windows)

    @property
    def kv_cache_manager(self) -> Optional[KVCacheManager]:
        """Return the unified KVCacheManager, or None if not initialized."""
        assert self._kv_cache_manager is not None, "KVCacheManager not initialized."
        return self._kv_cache_manager

    @property
    def kv_cache_config_tuned(self) -> KvCacheConfig:
        """Return the KVCacheConfig tuned for the KVCacheManager."""
        assert None not in [self._kv_cache_manager, self._kv_cache_config_tuned], (
            "KVCacheManager not initialized."
        )
        return self._kv_cache_config_tuned

    @property
    def kv_cache_config(self) -> KvCacheConfig:
        """Return the original KVCacheConfig as passed in."""
        return self._kv_cache_config_original

    def _clear_caches(self) -> None:
        """Clear all caches and views before pool release."""
        for k in self._caches:
            self._caches[k] = None
        self._unmanaged_resources.clear()

    def shutdown(self) -> None:
        """Shutdown and release all resources."""
        if self._kv_cache_manager is not None:
            self._kv_cache_manager.shutdown()
        self._clear_caches()


GetInferenceModel = Callable[[CachedSequenceInterface], nn.Module]
