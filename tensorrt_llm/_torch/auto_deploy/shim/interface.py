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
        # lookup of unmanaged resources
        self._unmanaged_resources: List[str] = []

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
            if name in self._unmanaged_resources:
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
            # Dummy SSM params - d_state=0 means empty tensor (no memory used)
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
    ) -> Tuple[Optional[KVPagedResourceHandler], ResourceHandlerDict]:
        """Identify KV resources compatible with the reference handler for KVCacheManager.

        The first KVPagedResourceHandler becomes the reference. All handlers matching
        the reference (via __eq__) are collected for managed allocation.

        Returns:
            Tuple of (reference_handler, managed_resources_dict).
            reference_handler is None if no KV paged resources exist.
        """
        kv_ref: Optional[KVPagedResourceHandler] = None
        kv_managed: ResourceHandlerDict = {}

        for name, handler in self._resource_lookup.items():
            if not isinstance(handler, KVPagedResourceHandler):
                continue
            if kv_ref is None:
                kv_ref = handler
            if handler == kv_ref:
                kv_managed[name] = handler

        return kv_ref, kv_managed

    def _identify_managed_state_resources(
        self,
    ) -> Tuple[Optional[SSMResourceHandler], list, Optional[CausalConvResourceHandler], list]:
        """Identify SSM and Conv resources compatible with MambaHybridCacheManager.

        Finds reference handlers for SSM and Conv resources, checks the n_groups constraint,
        and collects all compatible resources for each type.

        Returns:
            Tuple of (ssm_ref, ssm_managed, conv_ref, conv_managed) where:
            - ssm_ref: Reference SSM handler or None
            - ssm_managed: List of (name, handler) tuples for compatible SSM resources
            - conv_ref: Reference Conv handler or None (may be None if constraint fails)
            - conv_managed: List of (name, handler) tuples for compatible Conv resources
        """
        ssm_ref: Optional[SSMResourceHandler] = None
        conv_ref: Optional[CausalConvResourceHandler] = None

        # Find reference handlers for each state resource type
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

        # Collect compatible resources for each managed type (using __eq__ for comparison)
        ssm_managed = [(n, h) for n, h in self._resource_lookup.items() if ssm_ref == h]
        conv_managed = [(n, h) for n, h in self._resource_lookup.items() if conv_ref == h]

        return ssm_ref, ssm_managed, conv_ref, conv_managed

    def _prepare_kv_cache_config(
        self,
        max_tokens: Optional[int],
        kv_managed: ResourceHandlerDict,
    ) -> KvCacheConfig:
        """Prepare and configure KvCacheConfig for cache manager creation.

        Handles deep copy, max_tokens synchronization across ranks, block reuse settings,
        copy_on_partial_reuse validation, and free_gpu_memory_fraction normalization.

        Args:
            max_tokens: Maximum tokens to allocate, or None to use config defaults.
            kv_managed: Dict of KV resources that will be managed by KVCacheManager.

        Returns:
            Configured KvCacheConfig ready for cache manager creation.
        """
        # Make a deep copy of the kv_cache_config to avoid modifying the original object
        kv_cache_config = copy.deepcopy(self._kv_cache_config_original)

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
                "layer_mask": None,
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
        conv_ref: Optional[CausalConvResourceHandler],
        conv_managed: list,
    ) -> Tuple[MambaHybridCacheManager, int]:
        """Create MambaHybridCacheManager and assign views for state resources.

        Creates the hybrid cache manager with mamba parameters derived from the reference
        handlers, then retrieves and assigns buffer views for all managed SSM and Conv resources.

        Args:
            kv_cache_kwargs: Base kwargs for cache manager (will be extended with mamba params).
            ssm_ref: Reference SSM handler or None.
            ssm_managed: List of (name, handler) tuples for SSM resources.
            conv_ref: Reference Conv handler or None.
            conv_managed: List of (name, handler) tuples for Conv resources.

        Returns:
            Tuple of (manager, num_managed_mamba_layers).
        """
        # Derive Mamba parameters from reference handlers
        mamba_params = self._get_mamba_state_params(
            ssm_ref, len(ssm_managed), conv_ref, len(conv_managed)
        )
        num_managed_mamba_layers = mamba_params["mamba_num_layers"]

        # Create the hybrid cache manager
        manager = MambaHybridCacheManager(
            **mamba_params,
            **kv_cache_kwargs,
        )

        # Retrieve and assign views for Mamba-managed resources (up to num_managed_mamba_layers)
        for layer_idx in range(num_managed_mamba_layers):
            if ssm_managed:
                ssm_view = manager.get_ssm_states(layer_idx)
                assert ssm_view.is_contiguous(), f"Non-contiguous state {ssm_managed[layer_idx][0]}"
                self._caches[ssm_managed[layer_idx][0]] = ssm_view
            if conv_managed:
                conv_view = manager.get_conv_states(layer_idx)
                assert conv_view.is_contiguous(), (
                    f"Non-contiguous state {conv_managed[layer_idx][0]}"
                )
                self._caches[conv_managed[layer_idx][0]] = conv_view

        return manager, num_managed_mamba_layers

    def _assign_kv_cache_views(self, kv_managed: Dict[str, KVPagedResourceHandler]) -> None:
        """Retrieve and assign buffer views for managed KV paged resources.

        Args:
            kv_managed: Dict of KV resources managed by the cache manager.
        """
        for idx, (name, h) in enumerate(kv_managed.items()):
            view = self._kv_cache_manager.get_buffers(idx, kv_layout=h.kv_layout)
            assert view[0].is_contiguous(), f"Non-contiguous kv cache resource for {name}"
            self._caches[name] = view

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

    def _create_kv_cache_manager(self, max_tokens: Optional[int] = None) -> Dict:
        """Create KVCacheManager or MambaHybridCacheManager with standard layout.

        For paged resources (KVPagedResourceHandler):
        - Uses the first KVPagedResourceHandler's head_dim and dtype as reference
        - Compatible resources (matching head_dim and dtype) go into KVCacheManager
        - Incompatible resources are allocated locally via handler.allocate()

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
        # 1. Identify managed resources
        kv_ref, kv_managed = self._identify_managed_kv_resources()
        ssm_ref, ssm_managed, conv_ref, conv_managed = self._identify_managed_state_resources()

        # 2. Prepare configuration
        kv_cache_config = self._prepare_kv_cache_config(max_tokens, kv_managed)
        kv_cache_kwargs = self._build_kv_cache_kwargs(kv_ref, kv_managed, kv_cache_config)

        # 3. Create cache manager (delegate to state helper if state resources exist)
        has_state_resources = ssm_managed or conv_managed
        if has_state_resources:
            # NOTE: +1 for cuda graph padding
            kv_cache_kwargs["max_batch_size"] = self.info.max_num_state_slots
            self._kv_cache_manager, _ = self._create_and_assign_state_views(
                kv_cache_kwargs, ssm_ref, ssm_managed, conv_ref, conv_managed
            )
        else:
            # No typed state resources - use pure KVCacheManager
            self._kv_cache_manager = KVCacheManager(**kv_cache_kwargs)

        # 4. Store tuned config and ensure capacity
        self._kv_cache_config_tuned = kv_cache_config
        self.info.estimate_cache_loc_capacity(self._kv_cache_manager.blocks_in_primary_pool)

        # 5. Assign KV views
        self._assign_kv_cache_views(kv_managed)

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
        num_ssm_total = sum(
            1 for h in self._resource_lookup.values() if isinstance(h, SSMResourceHandler)
        )
        num_conv_total = sum(
            1 for h in self._resource_lookup.values() if isinstance(h, CausalConvResourceHandler)
        )
        num_state_other = num_state_total - num_ssm_total - num_conv_total

        total_managed = len(kv_managed) + len(ssm_managed) + len(conv_managed)

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
            "ssm_managed": len(ssm_managed),
            "conv_total": num_conv_total,
            "conv_managed": len(conv_managed),
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
        The new manager will compute optimal capacity based on current free GPU memory.
        """
        if not self.needs_resize():
            return

        # Calculate bytes-per-token for paged (resizable) resources
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
        # Reserve space for non-paged caches and mem_exclude, then apply free_gpu_memory_fraction
        free_gpu_memory_fraction = self._kv_cache_config_original.free_gpu_memory_fraction
        mem_for_paged_optimal = (
            free_mem - non_paged_bytes_total - mem_exclude
        ) * free_gpu_memory_fraction
        max_tokens_optimal = int(mem_for_paged_optimal // paged_bytes_per_token)

        # Create new cache manager with optimal capacity
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
