import copy
import functools
import math
from typing import Callable, Dict, Optional, Tuple, Union, final

import torch
import torch.distributed as dist
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
        self.kv_cache_config: KvCacheConfig = kv_cache_config or KvCacheConfig()

        # Create SequenceInfo internally, using tokens_per_block from kv_cache_config
        self.info = SequenceInfo(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            tokens_per_block=self.kv_cache_config.tokens_per_block,
            max_num_tokens=max_num_tokens,
            vocab_size_padded=vocab_size_padded,
        )

        # NOTE: we keep an extra state slot around to simplify cuda graph padding
        # WHY?
        # Requests that just finished won't free their used resources immediately. Specifically, the
        # running order is self.scheduler.schedule_request, self._forward_step() and
        # self._process_previous_batch() in the PyExecutor. Hence, the current forward step will
        # remove finished requests but will not remove mamba_cache immediately and therefore it
        # won't be available in time for padding in the next forward step.
        self.max_num_state_slots = max_batch_size + 1

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
            if k in type(self.kv_cache_config).model_fields:
                setattr(self.kv_cache_config, k, v)
            else:
                raise ValueError(f"Invalid KVCacheConfig field: {k}")

    def add_resource(self, name: str, resource_handler: ResourceHandler) -> None:
        """Add a resource handler to the cache interface."""
        self._resource_lookup[name] = resource_handler

    def _create_kv_cache_manager(self, is_estimating: bool) -> None:
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
            is_estimating: If True, use estimation mode (conservative estimate).

        Returns:
            None
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

        # For estimation mode, we need to set max_tokens if not already set.
        # Use deepcopy to avoid mutating the original config.
        kv_cache_config = self.kv_cache_config
        if is_estimating and kv_cache_config.max_tokens is None:
            kv_cache_config = copy.deepcopy(kv_cache_config)
            # for estimation mode, we assume that max_tokens we need is just a single prefill pass.
            kv_cache_config.max_tokens = self.info.estimate_cache_tokens_per_forward()

        # Force distributed sync when torch distributed is initialized, even though
        # we pass Mapping() with world_size=1. This is because AutoDeploy handles
        # distributed operations externally but still needs all ranks to agree on
        # the number of blocks during calculate_max_num_blocks.
        force_sync = dist.is_initialized()

        # Check if we should disable block reuse
        if kv_cache_config.enable_block_reuse and not self.is_paged():
            kv_cache_config.enable_block_reuse = False
            ad_logger.info(f"Setting {kv_cache_config.enable_block_reuse=} for hybrid models.")

        # Common KV cache parameters
        kv_cache_kwargs = {
            "kv_cache_config": kv_cache_config,
            "kv_cache_type": CacheTypeCpp.SELFKONLY,  # kv_factor=1, treat K and V separately
            "num_layers": len(self._paged_cache_order) or 1,  # correct num layers or dummy layer
            "num_kv_heads": num_kv_heads_per_layer or [1],  # per-layer bytes_per_token or dummy
            "head_dim": 1,  # all bytes are expressed in the num_kv_heads dimension
            "tokens_per_block": kv_cache_config.tokens_per_block,
            "max_seq_len": self.info.max_seq_len,
            "max_batch_size": self.info.max_batch_size,
            "mapping": Mapping(),
            "dtype": DataType.FP8,  # 1-byte dtype for byte-level abstraction
            "layer_mask": None,
            "is_estimating_kv_cache": is_estimating,
            "force_distributed_sync": force_sync,
        }

        if self._state_resource_order:
            # NOTE: +1 for cuda graph padding
            kv_cache_kwargs["max_batch_size"] = self.max_num_state_slots

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
        self._create_kv_cache_manager(is_estimating=True)

        return len(self._caches)

    def is_paged(self) -> bool:
        """Return True if all resources are paged and part of the KVCacheManager."""
        return self._kv_cache_manager is not None and all(
            name in self._paged_cache_order for name in self._resource_lookup.keys()
        )

    def needs_resize(self) -> bool:
        """Check if we need a resize or not."""
        return (
            self._kv_cache_manager is not None
            and self.kv_cache_config.free_gpu_memory_fraction not in [None, 0.0]
        )

    def resize_kv_cache_manager(self) -> None:
        """Shutdown existing KVCacheManager and create new one with optimal capacity.

        This implements the two-phase approach: after running a forward pass during estimation
        to allocate intermediate memory, call this method to recreate the KVCacheManager.
        The new manager will compute optimal capacity based on current free GPU memory
        via calculate_max_num_blocks.
        """
        if not self.needs_resize():
            return

        # Shutdown existing KVCacheManager to free memory
        self._kv_cache_manager.shutdown()

        # Create new KVCacheManager with final capacity (not estimating).
        # KVCacheManager.calculate_max_num_blocks will compute optimal capacity
        # based on current free GPU memory and kv_cache_config settings.
        self._create_kv_cache_manager(is_estimating=False)

    @property
    def kv_cache_manager(self) -> Optional[KVCacheManager]:
        """Return the KVCacheManager managing paged resources, or None if not initialized."""
        return self._kv_cache_manager

    def _clear_cache_views(self) -> None:
        """Set paged and state cache views to None before pool release."""
        for name in self._paged_cache_order:
            self._caches[name] = None
        for name in self._state_resource_order:
            self._caches[name] = None

    def shutdown(self) -> None:
        """Shutdown and release all resources."""
        if self._kv_cache_manager is not None:
            self._kv_cache_manager.shutdown()
        self._caches.clear()


GetInferenceModel = Callable[[CachedSequenceInterface], nn.Module]
