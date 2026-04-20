import copy
import functools
from typing import Callable, Dict, List, Optional, Tuple, Union, final

import torch
import torch.nn as nn
from torch._prims_common import DeviceLikeType

import tensorrt_llm.bindings
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

from ...._utils import torch_dtype_to_binding
from ...pyexecutor.mamba_cache_manager import MambaHybridCacheManager
from ...pyexecutor.resource_manager import KVCacheManager
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

CacheTypeCpp = tensorrt_llm.bindings.internal.batch_manager.CacheType
DataType = tensorrt_llm.bindings.DataType


def with_pre_callback(method, callback):
    """Wrap method to call callback before the original method."""

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        callback()
        return method(*args, **kwargs)

    return wrapper


class MultiPoolKVCacheManager:
    """Wraps one KVCacheManager per KV group behind a unified API.

    A group is one unique ``(head_dim, dtype, kv_factor, kv_layout, sliding_window)``
    tuple, as determined by ``KVPagedResourceHandler.__eq__``.  Each group gets its
    own pool with its own max_attention_window, so ``group_idx`` and ``pool_idx``
    are the same index — callers use ``get_pool(group_idx)`` to reach the backing
    pool.

    Lifecycle methods (prepare/free/shutdown) are delegated to ALL pools.  The
    primary pool (largest window, typically full-attention) provides the C++
    impl for the scheduler and determines overall capacity.  SWA pools have
    fixed size and never constrain scheduling.
    """

    def __init__(self, managers: List[KVCacheManager], primary_idx: int = 0):
        self._managers = managers
        self._primary_idx = primary_idx

    @property
    def impl(self):
        return self._managers[self._primary_idx].impl

    @property
    def tokens_per_block(self):
        return self._managers[self._primary_idx].tokens_per_block

    @property
    def max_blocks_per_seq(self):
        return self._managers[self._primary_idx].max_blocks_per_seq

    @property
    def blocks_in_primary_pool(self):
        return self._managers[self._primary_idx].blocks_in_primary_pool

    def get_num_free_blocks(self):
        return min(m.get_num_free_blocks() for m in self._managers)

    def get_max_resource_count(self):
        return self._managers[self._primary_idx].get_max_resource_count()

    def get_needed_resource_to_completion(self, request):
        return self._managers[self._primary_idx].get_needed_resource_to_completion(request)

    def get_num_kv_blocks(self, num_tokens: int) -> int:
        return self._managers[self._primary_idx].get_num_kv_blocks(num_tokens)

    def prepare_resources(self, scheduled_batch):
        for m in self._managers:
            m.prepare_resources(scheduled_batch)

    def free_resources(self, request, pin_on_release=False):
        for m in self._managers:
            m.free_resources(request, pin_on_release)

    def update_resources(self, scheduled_batch, attn_metadata=None, kv_cache_dtype_byte_size=None):
        for m in self._managers:
            m.update_resources(scheduled_batch, attn_metadata, kv_cache_dtype_byte_size)

    def add_dummy_requests(self, request_ids, **kwargs):
        results = None
        for m in self._managers:
            results = m.add_dummy_requests(request_ids, **kwargs)
        return results

    def shutdown(self):
        for m in self._managers:
            m.shutdown()

    def get_pool(self, group_idx: int) -> KVCacheManager:
        return self._managers[group_idx]

    @property
    def num_pools(self):
        return len(self._managers)

    @property
    def max_concurrent_sequences(self) -> int:
        """Max sequences all pools can serve simultaneously.

        The minimum across pools of (total_blocks / max_blocks_per_seq).
        Use this to cap the scheduler's max_num_requests.
        """
        return min(
            m.get_max_resource_count() // max(m.max_blocks_per_seq, 1) for m in self._managers
        )

    def get_buffers(self, idx: int, kv_layout: str = "NHD"):
        raise NotImplementedError("Use get_pool(group_idx).get_buffers() instead")

    # Passthrough properties accessed by PyExecutor and other consumers
    @property
    def event_buffer_max_size(self):
        return self._managers[self._primary_idx].event_buffer_max_size

    @property
    def enable_block_reuse(self):
        return self._managers[self._primary_idx].enable_block_reuse

    @property
    def enable_partial_reuse(self):
        return self._managers[self._primary_idx].enable_partial_reuse

    @property
    def is_draft(self):
        return self._managers[self._primary_idx].is_draft

    @property
    def kv_cache_pool_pointers(self):
        return self._managers[self._primary_idx].kv_cache_pool_pointers

    @property
    def kv_cache_pool_mapping(self):
        return self._managers[self._primary_idx].kv_cache_pool_mapping

    def get_cache_indices(self, request, **kwargs):
        mgr = self._managers[self._primary_idx]
        # When max_attention_window_vec has N identical entries (one per layer),
        # the underlying get_cache_indices requires an explicit window_size.
        if "window_size" not in kwargs:
            kwargs["window_size"] = max(mgr.max_attention_window_vec)
        return mgr.get_cache_indices(request, **kwargs)

    def store_blocks_for_reuse(self, request, pin_blocks=False):
        for m in self._managers:
            m.store_blocks_for_reuse(request, pin_blocks)


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
        # Per-group sliding window sizes, published by the kvcache transform and
        # consumed by the executor.  group_idx IS pool_idx by construction
        # (KVPagedResourceHandler.__eq__ includes sliding_window, so each unique
        # window becomes a distinct storage pool and metadata set).
        self._kv_group_windows: List[int] = []
        # lookup of unmanaged resources
        self._unmanaged_resources: List[str] = []
        self._spec_config = spec_config

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

    def _identify_managed_kv_groups(
        self,
    ) -> List[Tuple[KVPagedResourceHandler, ResourceHandlerDict]]:
        """Identify KV resource groups for multi-pool KVCacheManager creation.

        Grouping is driven entirely by ``KVPagedResourceHandler.__eq__``, which
        compares ``head_dim``, ``dtype``, ``kv_factor``, ``kv_layout``, and
        ``sliding_window``.  Same-head-dim layers with different sliding windows
        therefore land in separate groups (and thus separate pools).

        Every KVPagedResourceHandler belongs to exactly one group — there are no
        unmanaged KV layers.  Each returned group index is the storage pool
        index used everywhere downstream.

        Returns:
            List of (reference_handler, managed_resources_dict) tuples, one per group.
            Empty list if no KV paged resources exist.
        """
        groups: List[Tuple[KVPagedResourceHandler, ResourceHandlerDict]] = []

        for name, handler in self._resource_lookup.items():
            if not isinstance(handler, KVPagedResourceHandler):
                continue
            # Find matching group or create a new one
            matched = False
            for ref, managed in groups:
                if handler == ref:
                    managed[name] = handler
                    matched = True
                    break
            if not matched:
                groups.append((handler, {name: handler}))

        return groups

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
        kv_ref: Optional[KVPagedResourceHandler] = None,
    ) -> KvCacheConfig:
        """Prepare and configure KvCacheConfig for cache manager creation.

        Handles deep copy, max_tokens synchronization across ranks, block reuse
        settings, copy_on_partial_reuse validation, and free_gpu_memory_fraction
        normalization.

        The max_attention_window vector is uniform within a group and read directly
        from the reference handler's ``sliding_window``: all layers in a group share
        the same window by construction (handler __eq__ includes sliding_window),
        so there is no per-layer scanning.

        Args:
            max_tokens: Maximum tokens to allocate, or None to use config defaults.
            kv_managed: Dict of KV resources that will be managed by KVCacheManager.
            kv_ref: Reference handler for this group.  Its ``sliding_window`` drives
                max_attention_window; None means no KV group (e.g. state-only).

        Returns:
            Configured KvCacheConfig ready for cache manager creation.
        """
        # Make a deep copy of the kv_cache_config to avoid modifying the original object
        kv_cache_config = copy.deepcopy(self._kv_cache_config_original)

        # Set uniform max_attention_window for this group from the handler's
        # sliding_window.  All layers in a group have the same window by
        # construction (__eq__ includes sliding_window).
        if kv_ref is not None and kv_ref.sliding_window > 0:
            window = kv_ref.sliding_window
            kv_cache_config.max_attention_window = [window] * len(kv_managed)
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
        kv_ref: Optional[KVPagedResourceHandler],
        kv_managed: ResourceHandlerDict,
        kv_cache_config: KvCacheConfig,
    ) -> Dict:
        """Build common kwargs for KVCacheManager or MambaHybridCacheManager.

        Args:
            kv_ref: Reference KV handler defining head_dim and dtype, or None.
            kv_managed: Dict of KV resources to be managed.
            kv_cache_config: Configured KvCacheConfig.

        Returns:
            Dict of kwargs suitable for both KVCacheManager and MambaHybridCacheManager.
        """
        # create arguments first that differ whether we have managed kv caches or not
        kv_cache_kwargs = {}
        if kv_managed:
            kv_cache_type = CacheTypeCpp.SELFKONLY if kv_ref.kv_factor == 1 else CacheTypeCpp.SELF
            kv_cache_kwargs.update(
                {
                    "kv_cache_type": kv_cache_type,
                    "num_layers": len(kv_managed),
                    "num_kv_heads": [h.num_kv_heads for h in kv_managed.values()],
                    "head_dim": kv_ref.head_dim,
                    "dtype": torch_dtype_to_binding(kv_ref.dtype),
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
        manager = MambaHybridCacheManager(
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

    def _assign_kv_cache_views(
        self, kv_managed: Dict[str, KVPagedResourceHandler], manager: KVCacheManager
    ) -> int:
        """Retrieve and assign buffer views for managed KV paged resources.

        Args:
            kv_managed: Dict of KV resources managed by the cache manager.
            manager: The KVCacheManager that owns these resources.

        Returns:
            block_offset_multiplier derived from the first KV cache view's strides.
        """
        block_offset_multiplier = 0
        for idx, (name, h) in enumerate(kv_managed.items()):
            view = manager.get_buffers(idx, kv_layout=h.kv_layout)
            assert view[0].is_contiguous(), f"Non-contiguous kv cache resource for {name}"
            self._caches[name] = view

            # Compute block_offset_multiplier from the first layer's kv_cache strides.
            # This is stride(0)/stride(1) which equals kv_factor for per-layer views
            # or num_layers*kv_factor for interleaved pools.
            if idx == 0:
                block_offset_multiplier = view.stride(0) // view.stride(1)

        return block_offset_multiplier

    def _allocate_unmanaged_resources(self) -> None:
        """Allocate resources not managed by cache managers.

        Resources that haven't been assigned a tensor (still None) are allocated
        locally via their handler's allocate() method.
        """
        self._unmanaged_resources.clear()
        for name, handler in self._resource_lookup.items():
            if self._caches[name] is None:  # Not yet assigned a tensor
                self._caches[name] = handler.allocate(self.info)
                self._unmanaged_resources.append(name)

    def _is_swa_group(self, kv_ref: KVPagedResourceHandler) -> bool:
        """Return True if this group uses a sliding window smaller than max_seq_len.

        A group's sliding window comes straight from the reference handler: every
        layer in the group has the same window value (handler __eq__ includes
        sliding_window).
        """
        return kv_ref.sliding_window > 0 and kv_ref.sliding_window < self.info.max_seq_len

    def _compute_group_token_budget(
        self,
        group_idx: int,
        kv_ref: KVPagedResourceHandler,
        kv_managed: ResourceHandlerDict,
        all_groups: List[Tuple[KVPagedResourceHandler, ResourceHandlerDict]],
        total_max_tokens: Optional[int],
    ) -> Optional[int]:
        """Compute the max_tokens budget for a single KV cache group.

        All pools must support the same max number of concurrent sequences N.
        N is derived from the total byte budget divided by the combined per-sequence
        cost across all groups.  Each group then gets N × its per-sequence tokens.

        All groups use max_seq_len for per-sequence cost (not window_size), because
        during prefill each sequence temporarily uses max_seq_len blocks.  SWA savings
        manifest as freed blocks during decode, enabling higher throughput.
        """
        if total_max_tokens is None and not self._is_swa_group(kv_ref):
            return None  # Let free_gpu_memory_fraction handle it

        tpb = self.info.tokens_per_block

        # Compute per-sequence BYTE cost — use max_seq_len for all groups.
        # During prefill, sequences need full max_seq_len blocks regardless of window.
        group_seq_bytes = []
        group_seq_tokens = []
        for _, gm in all_groups:
            bpt = sum(h.bytes_per_token for h in gm.values())
            seq_tokens = self.info.max_seq_len
            group_seq_bytes.append(bpt * seq_tokens)
            group_seq_tokens.append(seq_tokens)
        combined_cost_per_seq = sum(group_seq_bytes)

        if total_max_tokens is None:
            # SWA group, no total budget — conservative 1-sequence estimate
            return self.info.max_seq_len

        # Compute N = max concurrent sequences from total budget.
        # Subtract 2 sequences as safety margin for rounding and allocation overhead.
        total_bpt = sum(sum(h.bytes_per_token for h in m.values()) for _, m in all_groups)
        total_budget_bytes = total_max_tokens * total_bpt
        max_seqs = (
            max(1, int(total_budget_bytes / combined_cost_per_seq) - 2)
            if combined_cost_per_seq > 0
            else 0
        )

        # This group needs max_seqs × its per-sequence tokens
        group_tokens = max_seqs * group_seq_tokens[group_idx]

        # Cap at max_batch_size × max_seq_len (can't need more)
        group_tokens = min(group_tokens, self.info.max_batch_size * self.info.max_seq_len)

        # Floor: at least one block per sequence for warmup feasibility
        min_tokens = self.info.max_batch_size * tpb
        group_tokens = max(group_tokens, min_tokens)

        return group_tokens

    def _get_group_max_window(self, kv_ref: KVPagedResourceHandler) -> int:
        """Return the effective window size for a group.

        Full-attention groups (``sliding_window == 0``) fall back to
        ``max_seq_len``.  SWA groups return their handler's sliding_window.
        """
        return kv_ref.sliding_window if kv_ref.sliding_window > 0 else self.info.max_seq_len

    def _create_kv_cache_manager(self, max_tokens: Optional[int] = None) -> Dict:
        """Create KVCacheManager(s) with standard layout.

        For paged resources (KVPagedResourceHandler):
        - Groups layers by handler equality (head_dim, dtype, kv_factor, kv_layout,
          sliding_window).  A group is a pool is a metadata set: same group_idx
          everywhere.
        - Each group gets its own KVCacheManager pool with a uniform
          max_attention_window derived from the group's sliding_window.
        - If multiple groups exist, wraps them in MultiPoolKVCacheManager.

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
        # 1. Identify managed resource groups (one per unique head_dim/dtype/kv_factor/layout)
        kv_groups = self._identify_managed_kv_groups()
        ssm_ref, ssm_managed, ssm_spec, conv_ref, conv_managed, conv_spec = (
            self._identify_managed_state_resources()
        )
        has_state_resources = ssm_managed or conv_managed

        # Collect ALL managed KV resources for stats (union of all groups)
        kv_managed_all: ResourceHandlerDict = {}
        for _, managed in kv_groups:
            kv_managed_all.update(managed)

        # 2. Create one KVCacheManager per group
        #    SWA groups (window < max_seq_len) get fixed max_tokens.
        #    Full-attention groups get the remaining budget via max_tokens or free_gpu_mem_fraction.
        managers: List[KVCacheManager] = []
        primary_idx = 0  # index of the full-attention (largest-window) group
        max_window_seen = 0

        for group_idx, (kv_ref, kv_managed) in enumerate(kv_groups):
            # Compute this group's token budget
            group_max_tokens = self._compute_group_token_budget(
                group_idx, kv_ref, kv_managed, kv_groups, max_tokens
            )
            group_config = self._prepare_kv_cache_config(group_max_tokens, kv_managed, kv_ref)
            group_kwargs = self._build_kv_cache_kwargs(kv_ref, kv_managed, group_config)

            # NOTE: SWA groups keep max_seq_len from config (NOT window_size).
            # During prefill, sequences temporarily use up to max_seq_len blocks.
            # max_attention_window evicts old blocks during decode, freeing them
            # for new sequences.  The SWA savings are throughput (more concurrent
            # decode sequences), not peak memory reduction.

            if has_state_resources and group_idx == 0:
                group_kwargs["max_batch_size"] = self.info.max_num_state_slots
                mgr, _ = self._create_and_assign_state_views(
                    group_kwargs,
                    ssm_ref,
                    ssm_managed,
                    ssm_spec,
                    conv_ref,
                    conv_managed,
                    conv_spec,
                )
            else:
                mgr = KVCacheManager(**group_kwargs)

            managers.append(mgr)
            is_swa = self._is_swa_group(kv_ref)
            ad_logger.info(
                f"KV pool {group_idx}: {len(kv_managed)} layers, "
                f"head_dim={kv_ref.head_dim}, "
                f"max_attention_window={group_config.max_attention_window}, "
                f"swa={is_swa}, "
                f"max_tokens={group_max_tokens}"
            )

            # Track which group has the largest window (= primary for scheduler)
            group_window = max(group_config.max_attention_window or [self.info.max_seq_len])
            if group_window > max_window_seen:
                max_window_seen = group_window
                primary_idx = group_idx

        # 3. Store manager (wrapper if multi-group, direct if single)
        if len(managers) == 0:
            # No KV resources — create a dummy manager for state-only models
            dummy_config = self._prepare_kv_cache_config(max_tokens, {})
            dummy_kwargs = self._build_kv_cache_kwargs(None, {}, dummy_config)
            if has_state_resources:
                dummy_kwargs["max_batch_size"] = self.info.max_num_state_slots
                self._kv_cache_manager, _ = self._create_and_assign_state_views(
                    dummy_kwargs,
                    ssm_ref,
                    ssm_managed,
                    ssm_spec,
                    conv_ref,
                    conv_managed,
                    conv_spec,
                )
            else:
                self._kv_cache_manager = KVCacheManager(**dummy_kwargs)
        elif len(managers) == 1:
            self._kv_cache_manager = managers[0]
        else:
            self._kv_cache_manager = MultiPoolKVCacheManager(managers, primary_idx=primary_idx)

        # 4. Store tuned config
        self._kv_cache_config_tuned = self._prepare_kv_cache_config(max_tokens, kv_managed_all)

        # 5. Assign KV views per group and update cache information
        block_offset_multiplier = 0
        if managers:
            for group_idx, (_, kv_managed) in enumerate(kv_groups):
                mgr = managers[group_idx]
                bom = self._assign_kv_cache_views(kv_managed, mgr)
                if group_idx == primary_idx:
                    block_offset_multiplier = bom
            primary_mgr = managers[primary_idx]
        else:
            primary_mgr = self._kv_cache_manager

        num_blocks = getattr(
            primary_mgr, "blocks_in_primary_pool", primary_mgr.get_max_resource_count()
        )
        self.info.update_cache_information(
            num_blocks=num_blocks,
            block_offset_multiplier=block_offset_multiplier,
        )

        # 6. Allocate remaining unmanaged resources
        self._allocate_unmanaged_resources()

        # 7. Patch shutdown
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
        total_managed = len(kv_managed_all) + ssm_managed_count + conv_managed_count

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
            "kv_managed": len(kv_managed_all),
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

        For multi-pool (dual head_dim): SWA pools have fixed size (max_batch_size × window).
        Only the full-attention pool benefits from resize.  All pools are shutdown and
        recreated so that the full-attention pool gets the remaining memory after SWA pools
        and forward-pass intermediates are accounted for.
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

        # Recreate all pools.  _compute_group_token_budget handles the split:
        # SWA groups get fixed tokens, full-attention gets the rest.
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
        """Per-group window sizes. group_idx IS pool_idx."""
        return self._kv_group_windows

    def set_kv_groups(self, group_windows: List[int]) -> None:
        """Store per-group window sizes (called by the kvcache transform)."""
        self._kv_group_windows = list(group_windows)

    @property
    def kv_cache_manager(self) -> Optional[Union[KVCacheManager, MultiPoolKVCacheManager]]:
        """Return the KVCacheManager (or multi-pool wrapper), or None if not initialized."""
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
