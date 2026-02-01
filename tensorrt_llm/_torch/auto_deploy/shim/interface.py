import copy
import functools
import math
from typing import Callable, Dict, Optional, Tuple, Union, final

import torch
import torch.nn as nn
from torch._prims_common import DeviceLikeType

import tensorrt_llm.bindings
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

from ...pyexecutor.mamba_cache_manager import MambaHybridCacheManager
from ...pyexecutor.resource_manager import KVCacheManager
from ..custom_ops.attention_interface import (
    PagedResourceHandler,
    ResourceHandler,
    ResourceHandlerDict,
    SequenceInfo,
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
        device: Optional[DeviceLikeType] = None,
        kv_cache_config: Optional[KvCacheConfig] = None,
        max_num_tokens: Optional[int] = None,
        vocab_size_padded: Optional[int] = None,
    ) -> None:
        """Initialize the CachedSequenceInterface.

        Args:
            max_seq_len: Maximum sequence length including input and generated tokens.
            max_batch_size: Maximum number of sequences (requests) that can be processed.
            device: Target device for tensors. Defaults to "cuda".
            kv_cache_config: KV cache configuration. If None, uses default KvCacheConfig.
            max_num_tokens: Maximum total tokens across all sequences. If None, computed from
                max_seq_len and max_batch_size.
            vocab_size_padded: Padded vocabulary size of the model.
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
        # Ordered dicts tracking resource handlers by type
        self._paged_cache_order: ResourceHandlerDict = {}  # Paged resources (kv caches)
        self._state_resource_order: ResourceHandlerDict = {}  # State resources (ssm states)

    @property
    def args(self) -> Tuple[torch.Tensor, ...]:
        """Return all the graph arguments owned by this interface."""
        return tuple(self.named_args.values())

    @property
    def named_args(self) -> Dict[str, torch.Tensor]:
        """Return all the named arguments owned by this interface."""
        return {**self.info.named_args, **self._caches}

    def to(self, *args, **kwargs) -> None:
        self.info.to(*args, **kwargs)
        # Only move locally-allocated caches (paged/state caches are managed by cache managers)
        for name, cache in self._caches.items():
            if name not in self._paged_cache_order and name not in self._state_resource_order:
                cache.to(*args, **kwargs)

    def update_kv_cache_config(self, **kwargs) -> None:
        """Update the KVCacheConfig with the given kwargs."""
        for k, v in kwargs.items():
            if k in type(self._kv_cache_config_original).model_fields:
                setattr(self._kv_cache_config_original, k, v)
            else:
                raise ValueError(f"Invalid KVCacheConfig field: {k}")

    def add_resource(self, name: str, resource_handler: ResourceHandler) -> None:
        """Add a resource handler to the cache interface."""
        self._resource_lookup[name] = resource_handler

    def _create_kv_cache_manager(self, max_tokens: Optional[int] = None) -> int:
        """Create KVCacheManager or MambaHybridCacheManager with multi-layer byte-level params.

        This uses a multi-layer approach with byte-level abstraction:
        - Paged resources: Each resource gets its own layer in KVCacheManager with
          num_kv_heads=bytes_per_token for that resource, head_dim=1.
        - State resources: Each resource gets its own layer in MambaCacheManager with
          head_dim=bytes_per_slot for that resource.

        Each layer's cache is contiguous, avoiding byte-offset slicing within layers.

        When state resources exist, MambaHybridCacheManager is used to manage both.

        Important NOTE on contiguity of managed resources:
        - We only guarantee contiguity for an individual page or an individual state slot.
        - Outside of these individual pages/slots, resources are NOT guaranteed to be contiguous.

        Args:
            max_tokens: Maximum number of tokens to allocate. If provided, it will use the min value
                between this value and max_tokens in kv_cache_config.

        Returns:
            The final number of tokens that can be cached in the KVCacheManager.
            NOTE: this number may differ from the provided ``max_tokens`` arg for two reasons:
                1. the final number of tokens is synced (min) across ranks
                2. rounding for getting a multiple of tokens_per_block
        """
        # Build per-layer num_kv_heads list for paged resources
        # Each paged resource becomes one "layer" with num_kv_heads = bytes_per_token
        num_kv_heads_per_layer = [
            math.prod(h.token_shape) * h.dtype.itemsize for h in self._paged_cache_order.values()
        ]

        # Calculate total bytes per slot for state resources (modeled as single layer)
        cumulative_bytes_per_state = [0]
        for name, handler in self._state_resource_order.items():
            byte_size = math.prod(handler.state_shape) * handler.dtype.itemsize
            cumulative_bytes_per_state.append(cumulative_bytes_per_state[-1] + byte_size)

        # Make a deep copy of the kv_cache_config to avoid modifying the original object
        kv_cache_config = copy.deepcopy(self._kv_cache_config_original)

        # Disable copy_on_partial_reuse
        # TODO: https://github.com/NVIDIA/TensorRT-LLM/issues/10966
        if kv_cache_config.copy_on_partial_reuse:
            kv_cache_config.copy_on_partial_reuse = False
            ad_logger.info("Disabling copy_on_partial_reuse for AutoDeploy backend.")

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
        if kv_cache_config.enable_block_reuse and not self.is_paged():
            kv_cache_config.enable_block_reuse = False
            ad_logger.info(f"Setting {kv_cache_config.enable_block_reuse=} for non-paged models.")

        # Make sure to set free_gpu_memory_fraction to None if set to 0.0
        # NOTE: KVCacheConfig validator enforces that free_gpu_memory_fraction must be between 0.0
        # and 1.0 but we allow 0.0 to be set to disable resizing (corresponding to None in the
        # manager).
        if kv_cache_config.free_gpu_memory_fraction == 0.0:
            kv_cache_config.free_gpu_memory_fraction = None

        # Common KV cache parameters
        kv_cache_kwargs = {
            "kv_cache_config": kv_cache_config,
            "kv_cache_type": CacheTypeCpp.SELFKONLY,  # kv_factor=1, treat K, V separately
            "num_layers": len(self._paged_cache_order),  # correct num layers
            "num_kv_heads": num_kv_heads_per_layer,  # per-layer bytes_per_token
            "head_dim": 1,  # all bytes in num_kv_heads
            "tokens_per_block": kv_cache_config.tokens_per_block,
            "max_seq_len": self.info.max_seq_len,
            "max_batch_size": self.info.max_batch_size,
            "mapping": Mapping(),
            # NOTE (lucaslie): this is the only 1-byte dtype currently supported by the
            # KVCacheManager. Ideally, we would use the typical uint8 dtype for byte-level
            # abstraction, but this is not supported.
            "dtype": DataType.FP8,  # 1-byte dtype for byte-level abstraction
            "layer_mask": None,
            # NOTE (lucaslie): we can always run with False here since when we are estimating, we
            # are explicitly setting the max_tokens in which case it's okay to use False here since
            # we don't rely on free_gpu_memory_fraction inside the KVCacheManager. This is similar
            # to _torch.pyexecutor._util.KVCacheCreator, which explicitly estimates the max_tokens
            # outside of the KVCacheManager.
            "is_estimating_kv_cache": False,
        }

        # update args if we are just doing a dummy cache manager
        if not len(self._paged_cache_order):
            kv_cache_kwargs.update(
                {
                    "num_layers": 1,
                    "num_kv_heads": 1,
                    "head_dim": 1,
                }
            )

        if self._state_resource_order:
            # NOTE: +1 for cuda graph padding
            kv_cache_kwargs["max_batch_size"] = self.info.max_num_state_slots

            self._kv_cache_manager = MambaHybridCacheManager(
                # Mamba params for single-layer byte buffer
                mamba_d_state=1,
                mamba_d_conv=1,  # conv_states will have shape [..., 0] (empty)
                mamba_num_heads=1,
                mamba_n_groups=1,
                mamba_head_dim=cumulative_bytes_per_state[-1],  # Total bytes per slot
                mamba_num_layers=1,  # Single layer
                mamba_layer_mask=None,  # Single enabled layer
                mamba_cache_dtype=torch.uint8,  # Byte-level
                mamba_ssm_cache_dtype=torch.uint8,  # Byte-level
                # KV cache params
                **kv_cache_kwargs,
            )
        else:
            # No state resources - use pure KVCacheManager
            self._kv_cache_manager = KVCacheManager(**kv_cache_kwargs)

        # store the tuned kv_cache_config
        self._kv_cache_config_tuned = kv_cache_config

        # Ensure cache_loc capacity is sufficient for the new KVCacheManager
        blocks_in_primary_pool = self._kv_cache_manager.blocks_in_primary_pool
        tokens_per_block = self._kv_cache_manager.tokens_per_block
        self.info.estimate_cache_loc_capacity(blocks_in_primary_pool)

        # Create paged resource views from per-layer buffers
        for layer_idx, (name, handler) in enumerate(self._paged_cache_order.items()):
            view = self._kv_cache_manager.get_buffers(layer_idx, kv_layout="NHD")
            view = view.view(blocks_in_primary_pool, tokens_per_block, -1).view(handler.dtype)
            view = view.view(blocks_in_primary_pool, tokens_per_block, *handler.token_shape)

            # Sanity check on contiguity of individual pages
            view_one_page = view[0]
            assert view_one_page.is_contiguous(), f"Per-page cache for {name} is not contiguous"

            self._caches[name] = view

        for layer_idx, (name, handler) in enumerate(self._state_resource_order.items()):
            num_states = len(self._kv_cache_manager.state_indices)
            # Get the single-layer ssm_states buffer
            # ssm_states shape: [1, num_states, 1, total_bytes_per_slot, 1]
            ssm_buffer = self._kv_cache_manager.get_ssm_states(0)
            # Flatten to [max_batch, total_bytes_per_slot_for_all_layers]
            ssm_buffer = ssm_buffer.view(num_states, -1)

            offset_start = cumulative_bytes_per_state[layer_idx]
            offset_end = cumulative_bytes_per_state[layer_idx + 1]

            # Slice at byte offset, reinterpret dtype, reshape
            view = ssm_buffer[:, offset_start:offset_end]
            view = view.view(handler.dtype)
            view = view.view(num_states, *handler.state_shape)

            # Sanity check on contiguity of individual state slots
            assert view[0].is_contiguous(), f"Per-slot state for {name} cache is not contiguous"

            self._caches[name] = view

        # Patch shutdown to clear cache views before pool release
        self._kv_cache_manager.shutdown = with_pre_callback(
            self._kv_cache_manager.shutdown,
            self._clear_cache_views,
        )

        max_resource_count = self._kv_cache_manager.get_max_resource_count()
        max_tokens_final = max_resource_count * self._kv_cache_manager.tokens_per_block

        return max_tokens_final

    def initialize_resources(self) -> int:
        """Initialize resources - paged/state caches via cache managers, others separately.

        Paged resources are managed by KVCacheManager (or the KV portion of MambaHybridCacheManager).
        State resources are managed by the Mamba portion of MambaHybridCacheManager.
        Other resources are allocated locally as a fallback.

        Returns:
            The number of caches initialized.
        """
        assert not self._caches and not self._paged_cache_order, "Caches already initialized."
        self.info.to(self.device)

        # Separate resources by type
        for name, handler in self._resource_lookup.items():
            if isinstance(handler, PagedResourceHandler):
                self._paged_cache_order[name] = handler
                self._caches[name] = None  # Will be set by _create_kv_cache_manager
            elif isinstance(handler, StateResourceHandler):
                self._state_resource_order[name] = handler
                self._caches[name] = None  # Will be set by _create_kv_cache_manager
            else:
                # Unknown handler type - allocate locally (fallback)
                self._caches[name] = handler.allocate(self.info)

        # Create unified cache manager (handles both paged and state resources)
        if self.needs_resize() or self._requires_token_estimate():
            max_tokens_estimate = self.info.estimate_cache_tokens_per_forward()
        else:
            # if we don't need a resize, we will just use the original settings in kv_cache_config
            # instead of passing in an overwrite here.
            max_tokens_estimate = None
        self._create_kv_cache_manager(max_tokens=max_tokens_estimate)

        return len(self._caches)

    def is_paged(self) -> bool:
        """Return True if all resources are paged and part of the KVCacheManager."""
        return set(self._paged_cache_order.keys()) == set(self._resource_lookup.keys())

    def _requires_token_estimate(self) -> bool:
        """Check if our kv_cache_config requires."""
        needs_max_tokens = self._kv_cache_config_original.free_gpu_memory_fraction in [None, 0.0]
        needs_max_tokens |= not (self._paged_cache_order)
        return needs_max_tokens and self._kv_cache_config_original.max_tokens is None

    def needs_resize(self) -> bool:
        """Check if we need a resize or not."""
        has_paged = bool(self._paged_cache_order)
        return has_paged and self._kv_cache_config_original.free_gpu_memory_fraction not in [
            None,
            0.0,
        ]

    def resize_kv_cache_manager(self, mem_exclude: int = 0) -> None:
        """Shutdown existing KVCacheManager and create new one with optimal capacity.

        Args:
            mem_exclude: Extra memory to exclude from the calculation of optimal capacity.
                This is in bytes and typically the memory reserved for the forward pass.

        This implements the two-phase approach: after running a forward pass during estimation
        to allocate intermediate memory, call this method to recreate the KVCacheManager.
        The new manager will compute optimal capacity based on current free GPU memory
        via calculate_max_num_blocks.
        """
        if not self.needs_resize():
            return

        # get per-token cache size for resizable resources
        paged_cache_bytes_per_token = self._kv_cache_manager.get_cache_bytes_per_token()

        # get total cache size of state resources that cannot be resized
        # NOTE: this does NOT include resources handled OUTSIDE of the KVCacheManager or
        # MambaHybridCacheManager. Those will persistent and will be accounted for via free_mem even
        # after the initialize kv_cache_manager is shutdown.
        state_cache_bytes_total = sum(
            cache.numel() * cache.element_size()
            for name, cache in self._caches.items()
            if name in self._state_resource_order
        )

        # get unmanaged cache size
        unmanaged_cache_bytes_total = sum(
            cache.numel() * cache.element_size()
            for name, cache in self._caches.items()
            if name not in self._paged_cache_order and name not in self._state_resource_order
        )

        # Shutdown existing KVCacheManager to free memory
        self._kv_cache_manager.shutdown()

        # Get current free GPU memory (roughly includes model weights + non-managed resources)
        _, free_mem, *_ = get_mem_info(empty_cache=True)

        # Compute available memory for the KVCacheManager
        # NOTE: free_mem was obtained AFTER shutdown of initial KVCacheManager - hence it accounts
        # for unmanaged resources but it does NOT account for state resources since those were
        # freed as part of the shutdown.
        free_gpu_memory_fraction = self._kv_cache_config_original.free_gpu_memory_fraction
        mem_for_paged_optimal = (
            free_mem - state_cache_bytes_total - mem_exclude
        ) * free_gpu_memory_fraction
        # Check how many tokens we can fit into the paged cache
        max_tokens_optimal = int(mem_for_paged_optimal // paged_cache_bytes_per_token)

        # Create new KVCacheManager with final capacity
        max_tokens_final = self._create_kv_cache_manager(max_tokens=max_tokens_optimal)

        # Log resulting memory information
        mem_info = [
            f"free_mem={bytes_to(free_mem, unit='GB'):.2f}GB",
            f"free_gpu_memory_fraction={free_gpu_memory_fraction}",
            f"mem_exclude={bytes_to(mem_exclude, unit='GB'):.2f}GB",
            f"mem_exclude_for_state={bytes_to(state_cache_bytes_total, unit='GB'):.2f}GB",
            f"mem_for_paged_optimal={bytes_to(mem_for_paged_optimal, unit='GB'):.2f}GB",
        ]
        total_cache_bytes = (
            mem_for_paged_optimal + state_cache_bytes_total + unmanaged_cache_bytes_total
        )
        mem_cache_info = [
            f"Max Tokens={max_tokens_final}",
            f"Paged={bytes_to(mem_for_paged_optimal, unit='GB'):.2f}GB",
            f"State={bytes_to(state_cache_bytes_total, unit='GB'):.2f}GB",
            f"Unmanaged={bytes_to(unmanaged_cache_bytes_total, unit='GB'):.2f}GB",
            f"Total={bytes_to(total_cache_bytes, unit='GB'):.2f}GB",
        ]
        ad_logger.info(f"Mem info for resize: {' | '.join(mem_info)}")
        ad_logger.info(f"Final Cache Mem: {' | '.join(mem_cache_info)}")

    @property
    def kv_cache_manager(self) -> Optional[KVCacheManager]:
        """Return the KVCacheManager managing paged resources, or None if not initialized."""
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

    def _clear_cache_views(self) -> None:
        """Set paged and state cache views to None before pool release."""
        self._kv_cache_config_tuned = None
        for name in self._paged_cache_order:
            self._caches[name] = None
        for name in self._state_resource_order:
            self._caches[name] = None

    def shutdown(self) -> None:
        """Shutdown and release all resources."""
        if self._kv_cache_manager is not None:
            self._kv_cache_manager.shutdown()
        self._kv_cache_config_tuned = None
        self._caches.clear()


GetInferenceModel = Callable[[CachedSequenceInterface], nn.Module]
