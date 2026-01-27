"""
TrtLLM-Gen Attention Backend

This module implements attention computation using flashinfer's trtllm-gen kernels.
It provides a drop-in replacement for thop.attention() with support for trtllm-gen
kernel only (Blackwell architecture: SM100/SM103).

Architecture Overview:
    1. AttentionConfig - Configuration dataclass for attention parameters
    2. TrtllmGenSupportChecker - Validates if configuration is supported
    3. BackendRegistry - Manages available attention backends
    4. FlashInferTrtllmGenBackend - FlashInfer implementation using trtllm-gen
    5. trtllm_gen_attention - Main entry point function

Usage:
    # Check if configuration is supported
    supported, reason = is_supported(num_heads=32, num_kv_heads=8, ...)
    if supported:
        trtllm_gen_attention(q, k, v, output, ...)
    else:
        # Fallback to thop.attention()
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import torch

try:
    import flashinfer

    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False

from tensorrt_llm._torch.attention_backend.interface import AttentionInputType
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.logger import logger

########################################################
# Constants
########################################################

# Alignment for workspace buffers (256 bytes)
WORKSPACE_ALIGNMENT = 256

# Default KV layout for flashinfer
DEFAULT_KV_LAYOUT = "HND"

# Default backend name
DEFAULT_BACKEND = "trtllm-gen"


########################################################
# Configuration Classes
########################################################


@dataclass
class AttentionConfig:
    """
    Configuration for attention computation.

    Encapsulates all parameters needed for attention to enable
    clean parameter passing and validation.
    """

    # Basic attention parameters
    num_heads: int
    num_kv_heads: int
    head_size: int
    layer_idx: int = 0

    # KV Cache parameters
    use_paged_kv_cache: bool = True
    tokens_per_block: int = 64
    max_num_requests: int = 256
    max_context_length: int = 8192
    attention_window_size: int = -1  # -1 means unlimited

    # Data types
    dtype: torch.dtype = torch.float16
    kv_cache_dtype: Optional[torch.dtype] = None
    out_dtype: Optional[torch.dtype] = None

    # RoPE parameters
    position_embedding_type: int = 0
    rotary_embedding_dim: int = 0
    rotary_embedding_base: float = 10000.0
    rotary_embedding_scale_type: int = 0
    rotary_embedding_scales: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    rotary_embedding_max_position_info: List[int] = field(default_factory=lambda: [8192, 8192])

    # Attention mask and features
    mask_type: int = 1  # CAUSAL by default
    q_scaling: float = 1.0
    beam_width: int = 1
    sink_token_length: int = 0

    # Advanced features (not supported by trtllm-gen)
    is_mla_enable: bool = False
    is_fused_qkv: bool = True
    update_kv_cache: bool = True
    cross_attention: bool = False
    is_spec_decoding: bool = False
    has_alibi: bool = False
    is_padded: bool = False
    position_shift_enabled: bool = False

    @property
    def effective_kv_cache_dtype(self) -> torch.dtype:
        """Get effective KV cache dtype, defaulting to input dtype."""
        return self.kv_cache_dtype if self.kv_cache_dtype is not None else self.dtype

    @property
    def effective_out_dtype(self) -> torch.dtype:
        """Get effective output dtype, defaulting to input dtype."""
        return self.out_dtype if self.out_dtype is not None else self.dtype

    @property
    def heads_ratio(self) -> int:
        """Get ratio of query heads to KV heads (for GQA)."""
        return self.num_heads // self.num_kv_heads if self.num_kv_heads > 0 else 1


########################################################
# Support Checker
########################################################


class TrtllmGenSupportChecker:
    """
    Validates if a configuration is supported by trtllm-gen backend.

    Implements all checks from the original C++ AttentionOp to determine
    if trtllm-gen kernel can handle the attention computation.
    """

    # Supported data types
    SUPPORTED_INPUT_DTYPES = {torch.float16, torch.bfloat16, torch.float8_e4m3fn}
    SUPPORTED_KV_CACHE_DTYPES = {torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.uint8}
    SUPPORTED_OUT_DTYPES = {torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.uint8}

    # Unsupported head sizes for context FMHA
    UNSUPPORTED_HEAD_SIZES_CONTEXT = {72, 80}

    # Maximum heads ratio for generation
    MAX_HEADS_RATIO_GENERATION = 16

    # Minimum tokens per block
    MIN_TOKENS_PER_BLOCK = 8

    @classmethod
    def check_hardware(cls) -> Tuple[bool, str]:
        """Check if hardware supports trtllm-gen (Blackwell SM100/SM103)."""
        sm = get_sm_version()
        if sm not in (100, 103):
            return (False, f"trtllm-gen requires SM100 or SM103 (Blackwell). Current: SM{sm}.")
        return True, ""

    @classmethod
    def check_basic_features(cls, config: AttentionConfig) -> Tuple[bool, str]:
        """Check basic feature requirements."""
        if config.is_mla_enable:
            return False, "MLA is not supported by trtllm-gen backend."

        if not config.is_fused_qkv:
            return False, "Only fused QKV is supported by trtllm-gen backend."

        if not config.update_kv_cache:
            return False, "KV cache update must be enabled for trtllm-gen backend."

        if config.cross_attention:
            return False, "Cross attention is not supported by trtllm-gen backend."

        if config.is_spec_decoding:
            return False, "Speculative decoding is not supported by trtllm-gen backend."

        return True, ""

    @classmethod
    def check_dtypes(cls, config: AttentionConfig) -> Tuple[bool, str]:
        """Check if data types are supported."""
        if config.dtype not in cls.SUPPORTED_INPUT_DTYPES:
            return (
                False,
                f"Input dtype {config.dtype} not supported. Supported: FP16, BF16, FP8 (E4M3).",
            )

        if config.kv_cache_dtype is not None:
            if config.kv_cache_dtype not in cls.SUPPORTED_KV_CACHE_DTYPES:
                return (
                    False,
                    f"KV cache dtype {config.kv_cache_dtype} not supported. "
                    f"Supported: FP16, BF16, FP8, FP4.",
                )

        if config.out_dtype is not None:
            if config.out_dtype not in cls.SUPPORTED_OUT_DTYPES:
                return (
                    False,
                    f"Output dtype {config.out_dtype} not supported. "
                    f"Supported: FP16, BF16, FP8, FP4.",
                )

        return True, ""

    @classmethod
    def check_head_config(cls, config: AttentionConfig) -> Tuple[bool, str]:
        """Check head configuration validity."""
        if config.num_heads <= 0:
            return False, "num_heads must be positive."

        if config.num_kv_heads <= 0:
            return False, "num_kv_heads must be positive."

        if config.num_heads % config.num_kv_heads != 0:
            return (
                False,
                f"num_heads ({config.num_heads}) must be divisible by "
                f"num_kv_heads ({config.num_kv_heads}).",
            )

        return True, ""

    @classmethod
    def check_context_phase(cls, config: AttentionConfig) -> Tuple[bool, str]:
        """Check context (prefill) phase specific requirements."""
        if config.head_size in cls.UNSUPPORTED_HEAD_SIZES_CONTEXT:
            return (False, f"[Context] Head size {config.head_size} is not supported.")

        try:
            mask_type_enum = AttentionMaskType(config.mask_type)
            if mask_type_enum == AttentionMaskType.custom_mask:
                return False, "[Context] Custom mask is not supported."
        except ValueError:
            return False, f"[Context] Invalid mask_type: {config.mask_type}."

        if config.has_alibi:
            return False, "[Context] ALiBi is not supported."

        if config.is_padded:
            return False, "[Context] Padded input is not supported."

        return True, ""

    @classmethod
    def check_generation_phase(cls, config: AttentionConfig) -> Tuple[bool, str]:
        """Check generation (decode) phase specific requirements."""
        if config.beam_width != 1:
            return (
                False,
                f"[Generation] Beam search (beam_width={config.beam_width}) "
                "is not supported. Must be 1.",
            )

        if config.position_shift_enabled:
            return False, "[Generation] Position shift is not supported."

        if config.sink_token_length != 0:
            return (
                False,
                f"[Generation] StreamingLLM (sink_token_length="
                f"{config.sink_token_length}) is not supported.",
            )

        if config.tokens_per_block < cls.MIN_TOKENS_PER_BLOCK:
            return (
                False,
                f"[Generation] tokens_per_block ({config.tokens_per_block}) "
                f"must be >= {cls.MIN_TOKENS_PER_BLOCK}.",
            )

        if config.heads_ratio > cls.MAX_HEADS_RATIO_GENERATION:
            return (
                False,
                f"[Generation] num_heads/num_kv_heads ratio ({config.heads_ratio}) "
                f"must be <= {cls.MAX_HEADS_RATIO_GENERATION}.",
            )

        if config.has_alibi:
            return False, "[Generation] ALiBi is not supported."

        return True, ""

    # Supported tokens_per_block values for trtllm-gen kernels
    SUPPORTED_TOKENS_PER_BLOCK = {64}

    @classmethod
    def check_paged_kv_cache(cls, config: AttentionConfig) -> Tuple[bool, str]:
        """Check paged KV cache configuration."""
        if config.use_paged_kv_cache:
            if config.tokens_per_block <= 0:
                return False, "tokens_per_block must be positive."

            # Must be power of 2
            if config.tokens_per_block & (config.tokens_per_block - 1) != 0:
                return (False, f"tokens_per_block ({config.tokens_per_block}) must be power of 2.")

            # Check if tokens_per_block is supported by trtllm-gen kernels
            if config.tokens_per_block not in cls.SUPPORTED_TOKENS_PER_BLOCK:
                return (
                    False,
                    f"tokens_per_block ({config.tokens_per_block}) is not supported by "
                    f"trtllm-gen kernels. Supported values: {sorted(cls.SUPPORTED_TOKENS_PER_BLOCK)}.",
                )

        return True, ""

    @classmethod
    def is_supported(cls, config: AttentionConfig, phase: str = "both") -> Tuple[bool, str]:
        """
        Comprehensive check if configuration is supported.

        Args:
            config: Attention configuration to validate.
            phase: Which phase to check - "context", "generation", or "both".

        Returns:
            Tuple of (is_supported, reason_if_not_supported).
        """
        # Hardware check
        ok, reason = cls.check_hardware()
        if not ok:
            return False, reason

        # Basic features check
        ok, reason = cls.check_basic_features(config)
        if not ok:
            return False, reason

        # Data type check
        ok, reason = cls.check_dtypes(config)
        if not ok:
            return False, reason

        # Head configuration check
        ok, reason = cls.check_head_config(config)
        if not ok:
            return False, reason

        # Phase-specific checks
        if phase in ("context", "both"):
            ok, reason = cls.check_context_phase(config)
            if not ok:
                return False, reason

        if phase in ("generation", "both"):
            ok, reason = cls.check_generation_phase(config)
            if not ok:
                return False, reason

        # Paged KV cache check
        ok, reason = cls.check_paged_kv_cache(config)
        if not ok:
            return False, reason

        return True, ""


########################################################
# Workspace Manager
########################################################


class WorkspaceManager:
    """
    Manages workspace allocation for attention computation.

    Aligned with C++ AttentionOp::getWorkspaceSize*() methods.
    """

    ALIGNMENT = WORKSPACE_ALIGNMENT

    @staticmethod
    def _align_size(size: int) -> int:
        """Align size to boundary."""
        alignment = WorkspaceManager.ALIGNMENT
        return ((size + alignment - 1) // alignment) * alignment

    @staticmethod
    def _get_dtype_size(dtype: torch.dtype) -> int:
        """Get size in bytes for dtype."""
        dtype_sizes = {
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.float32: 4,
            torch.float8_e4m3fn: 1,
            torch.uint8: 1,
            torch.int8: 1,
        }
        return dtype_sizes.get(dtype, 2)

    @classmethod
    def get_context_workspace_size(
        cls,
        dtype: torch.dtype,
        max_num_seq: int,
        max_context_length: int,
        max_num_tokens: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        rotary_embedding_dim: int = 0,
    ) -> int:
        """Calculate workspace size for context (prefill) phase."""
        if max_num_tokens == 0:
            return 0

        dtype_size = cls._get_dtype_size(dtype)
        local_hidden_units_qo = num_heads * head_size

        # Q buffer for paged context FMHA
        q_buf_size = dtype_size * max_num_tokens * local_hidden_units_qo

        # Cumulative sequence lengths
        cu_seqlens_size = 4 * (max_num_seq + 1)  # sizeof(int)

        # Rotary inv freq buffer
        rotary_inv_freq_size = (
            4 * max_num_seq * rotary_embedding_dim // 2 if rotary_embedding_dim > 0 else 0
        )

        # Tokens info: (batch_idx, token_idx_in_seq) per token
        tokens_info_size = 8 * max_num_tokens  # sizeof(int2)

        # FMHA scheduler counter
        fmha_scheduler_counter = 4  # sizeof(uint32_t)

        # BMM scales for FP8
        fmha_bmm1_scale_size = 4 * 2  # sizeof(float) * 2
        fmha_bmm2_scale_size = 4  # sizeof(float)

        # Calculate total with alignment
        workspace_size = 0
        workspace_size += cls._align_size(q_buf_size)
        workspace_size += cls._align_size(cu_seqlens_size) * 3  # q, kv, mask_rows
        workspace_size += cls._align_size(rotary_inv_freq_size)
        workspace_size += cls._align_size(tokens_info_size)
        workspace_size += cls._align_size(fmha_scheduler_counter)
        workspace_size += cls._align_size(fmha_bmm1_scale_size)
        workspace_size += cls._align_size(fmha_bmm2_scale_size)

        return workspace_size

    @classmethod
    def get_generation_workspace_size(
        cls,
        dtype: torch.dtype,
        max_num_seq: int,
        max_attention_window_size: int,
        max_num_tokens: int,
        max_blocks_per_sequence: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        rotary_embedding_dim: int = 0,
        multi_processor_count: int = 132,
    ) -> int:
        """Calculate workspace size for generation (decode) phase."""
        if max_num_tokens == 0:
            return 0

        dtype_size = cls._get_dtype_size(dtype)
        batch_beam = max_num_seq

        # Estimate max sequence length tile
        max_seq_len_tile = max(
            1, (multi_processor_count + batch_beam * num_heads - 1) // (batch_beam * num_heads)
        )
        max_seq_len_tile = max(max_seq_len_tile, 4)

        # Partial output/sum/max buffers for multi-block attention
        partial_out_size = dtype_size * batch_beam * num_heads * head_size * max_seq_len_tile
        partial_sum_size = 4 * batch_beam * num_heads * max_seq_len_tile
        partial_max_size = 4 * batch_beam * num_heads * max_seq_len_tile

        # XQA workspace components
        cu_seqlens_size = 4 * (batch_beam + 1)
        cu_kv_seqlens_size = 4 * (batch_beam + 1)
        rotary_inv_freq_size = (
            4 * batch_beam * rotary_embedding_dim // 2 if rotary_embedding_dim > 0 else 0
        )
        tokens_info_size = 8 * max_num_tokens

        # Scales for trtllm-gen kernels
        bmm1_scale_size = 4 * 2
        bmm2_scale_size = 4

        # Calculate total with alignment
        workspace_size = 0
        workspace_size += cls._align_size(partial_out_size)
        workspace_size += cls._align_size(partial_sum_size)
        workspace_size += cls._align_size(partial_max_size)
        workspace_size += cls._align_size(cu_seqlens_size)
        workspace_size += cls._align_size(cu_kv_seqlens_size)
        workspace_size += cls._align_size(rotary_inv_freq_size)
        workspace_size += cls._align_size(tokens_info_size)
        workspace_size += cls._align_size(bmm1_scale_size)
        workspace_size += cls._align_size(bmm2_scale_size)

        return workspace_size

    @classmethod
    def get_workspace_size(
        cls,
        config: AttentionConfig,
        num_tokens: int,
        num_gen_tokens: int,
        max_blocks_per_sequence: int,
    ) -> int:
        """
        Calculate total workspace size.

        Returns max(context_workspace, generation_workspace).
        """
        context_size = cls.get_context_workspace_size(
            dtype=config.dtype,
            max_num_seq=config.max_num_requests,
            max_context_length=config.max_context_length,
            max_num_tokens=num_tokens,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_size=config.head_size,
            rotary_embedding_dim=config.rotary_embedding_dim,
        )

        generation_size = cls.get_generation_workspace_size(
            dtype=config.dtype,
            max_num_seq=config.max_num_requests,
            max_attention_window_size=config.attention_window_size,
            max_num_tokens=num_gen_tokens,
            max_blocks_per_sequence=max_blocks_per_sequence,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_size=config.head_size,
            rotary_embedding_dim=config.rotary_embedding_dim,
        )

        return max(context_size, generation_size)


########################################################
# Backend Protocol and Registry
########################################################


class AttentionBackendBase(ABC):
    """Abstract base class for attention backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name identifier."""
        pass

    @abstractmethod
    def is_supported(self, config: AttentionConfig) -> Tuple[bool, str]:
        """Check if configuration is supported."""
        pass

    @abstractmethod
    def run_context(
        self,
        query: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        cum_seq_lens_q: torch.Tensor,
        cum_seq_lens_kv: torch.Tensor,
        workspace: torch.Tensor,
        config: AttentionConfig,
        max_q_len: int,
        max_kv_len: int,
        batch_size: int,
        bmm1_scale: float,
        bmm2_scale: float,
        window_left: int = -1,
    ) -> torch.Tensor:
        """Execute context (prefill) phase."""
        pass

    @abstractmethod
    def run_generation(
        self,
        query: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        workspace: torch.Tensor,
        config: AttentionConfig,
        max_kv_len: int,
        bmm1_scale: float,
        bmm2_scale: float,
        window_left: int = -1,
    ) -> torch.Tensor:
        """Execute generation (decode) phase."""
        pass


class BackendRegistry:
    """
    Registry for attention backends.

    Manages available backends and provides automatic selection.
    """

    _backends: Dict[str, Type[AttentionBackendBase]] = {}
    _instances: Dict[str, AttentionBackendBase] = {}
    _default: str = DEFAULT_BACKEND
    # Priority order for backend selection (higher priority first)
    _priority: List[str] = ["trtllm-gen"]

    @classmethod
    def register(cls, name: str, priority: Optional[int] = None):
        """
        Decorator to register a backend class.

        Usage:
            @BackendRegistry.register("my-backend")
            class MyBackend(AttentionBackendBase):
                ...

            @BackendRegistry.register("high-priority-backend", priority=0)
            class HighPriorityBackend(AttentionBackendBase):
                ...

        Args:
            name: Backend name identifier.
            priority: Optional priority index (0 = highest). If None, appends to end.
        """

        def decorator(backend_cls: Type[AttentionBackendBase]):
            cls._backends[name] = backend_cls
            if name not in cls._priority:
                if priority is not None:
                    cls._priority.insert(priority, name)
                else:
                    cls._priority.append(name)
            return backend_cls

        return decorator

    @classmethod
    def get(cls, name: str = "auto") -> AttentionBackendBase:
        """
        Get backend instance by name.

        Args:
            name: Backend name or "auto" for automatic selection.

        Returns:
            Backend instance.
        """
        if name == "auto":
            name = cls._default

        if name not in cls._instances:
            if name not in cls._backends:
                raise ValueError(
                    f"Backend '{name}' not found. Available: {list(cls._backends.keys())}"
                )
            cls._instances[name] = cls._backends[name]()

        return cls._instances[name]

    @classmethod
    def list_backends(cls) -> List[str]:
        """List all registered backend names in priority order."""
        return [name for name in cls._priority if name in cls._backends]

    @classmethod
    def set_default(cls, name: str) -> None:
        """Set the default backend name."""
        if name not in cls._backends:
            raise ValueError(f"Backend '{name}' not found.")
        cls._default = name

    @classmethod
    def set_priority(cls, priority_list: List[str]) -> None:
        """
        Set the priority order for backend selection.

        Args:
            priority_list: List of backend names in priority order (highest first).
        """
        cls._priority = priority_list


class BackendSelector:
    """
    Intelligent backend selector for attention computation.

    Provides methods to select the best backend based on configuration,
    hardware capabilities, and user preferences.

    Usage:
        selector = BackendSelector()

        # Select best backend for config
        backend, reason = selector.select(config)

        # Or with fallback handling
        result = selector.select_with_fallback(config)
        if result.supported:
            backend = result.backend
        else:
            # Use fallback (e.g., thop.attention)
    """

    @dataclass
    class SelectionResult:
        """Result of backend selection."""

        supported: bool
        backend: Optional[AttentionBackendBase]
        backend_name: str
        reason: str
        checked_backends: List[Tuple[str, str]]  # [(name, reason), ...]

    def __init__(
        self,
        preferred_backend: Optional[str] = None,
        fallback_enabled: bool = True,
    ):
        """
        Initialize the backend selector.

        Args:
            preferred_backend: Preferred backend name (None for auto-select).
            fallback_enabled: Whether to try other backends if preferred fails.
        """
        self._preferred = preferred_backend
        self._fallback_enabled = fallback_enabled

    def select(
        self,
        config: AttentionConfig,
        phase: str = "both",
    ) -> Tuple[Optional[AttentionBackendBase], str]:
        """
        Select the best backend for the given configuration.

        Args:
            config: Attention configuration.
            phase: Phase to check ("context", "generation", or "both").

        Returns:
            Tuple of (backend_or_none, reason_string).
        """
        result = self.select_with_details(config, phase)
        return result.backend, result.reason

    def select_with_details(
        self,
        config: AttentionConfig,
        phase: str = "both",
    ) -> "BackendSelector.SelectionResult":
        """
        Select backend with detailed information about the selection process.

        Args:
            config: Attention configuration.
            phase: Phase to check ("context", "generation", or "both").

        Returns:
            SelectionResult with full details.
        """
        checked_backends: List[Tuple[str, str]] = []

        # If preferred backend is specified, try it first
        if self._preferred is not None:
            try:
                backend = BackendRegistry.get(self._preferred)
                supported, reason = backend.is_supported(config)
                checked_backends.append((self._preferred, reason if not supported else ""))

                if supported:
                    return self.SelectionResult(
                        supported=True,
                        backend=backend,
                        backend_name=self._preferred,
                        reason=f"Using preferred backend: {self._preferred}",
                        checked_backends=checked_backends,
                    )
                elif not self._fallback_enabled:
                    return self.SelectionResult(
                        supported=False,
                        backend=None,
                        backend_name="",
                        reason=f"Preferred backend '{self._preferred}' not supported: {reason}",
                        checked_backends=checked_backends,
                    )
            except ValueError as e:
                checked_backends.append((self._preferred, str(e)))
                if not self._fallback_enabled:
                    return self.SelectionResult(
                        supported=False,
                        backend=None,
                        backend_name="",
                        reason=str(e),
                        checked_backends=checked_backends,
                    )

        # Try backends in priority order
        for name in BackendRegistry.list_backends():
            if name == self._preferred:
                continue  # Already tried

            try:
                backend = BackendRegistry.get(name)
                supported, reason = backend.is_supported(config)
                checked_backends.append((name, reason if not supported else ""))

                if supported:
                    return self.SelectionResult(
                        supported=True,
                        backend=backend,
                        backend_name=name,
                        reason=f"Auto-selected backend: {name}",
                        checked_backends=checked_backends,
                    )
            except Exception as e:
                checked_backends.append((name, str(e)))

        # No backend supports this configuration
        reasons = [f"{name}: {reason}" for name, reason in checked_backends if reason]
        return self.SelectionResult(
            supported=False,
            backend=None,
            backend_name="",
            reason="No backend supports this configuration. " + "; ".join(reasons),
            checked_backends=checked_backends,
        )

    @classmethod
    def get_supported_backends(
        cls,
        config: AttentionConfig,
        phase: str = "both",
    ) -> List[Tuple[str, AttentionBackendBase]]:
        """
        Get all backends that support the given configuration.

        Args:
            config: Attention configuration.
            phase: Phase to check.

        Returns:
            List of (name, backend) tuples for supported backends.
        """
        supported = []
        for name in BackendRegistry.list_backends():
            try:
                backend = BackendRegistry.get(name)
                is_supported, _ = backend.is_supported(config)
                if is_supported:
                    supported.append((name, backend))
            except Exception:
                pass
        return supported


def select_backend(
    config: AttentionConfig,
    preferred: Optional[str] = None,
    phase: str = "both",
) -> Tuple[Optional[AttentionBackendBase], bool, str]:
    """
    Convenience function to select the best backend for a configuration.

    This is the recommended way to select a backend before calling
    trtllm_gen_attention or deciding to fallback to thop.attention.

    Args:
        config: Attention configuration.
        preferred: Preferred backend name (None for auto-select).
        phase: Phase to check ("context", "generation", or "both").

    Returns:
        Tuple of (backend_or_none, is_supported, reason).

    Usage:
        config = AttentionConfig(num_heads=32, num_kv_heads=8, head_size=128, ...)
        backend, supported, reason = select_backend(config)

        if supported:
            # Use trtllm_gen_attention with selected backend
            trtllm_gen_attention(...)
        else:
            # Fallback to thop.attention
            logger.info(f"Falling back to thop.attention: {reason}")
            thop.attention(...)
    """
    selector = BackendSelector(preferred_backend=preferred, fallback_enabled=True)
    result = selector.select_with_details(config, phase)
    return result.backend, result.supported, result.reason


def select_backend_from_params(
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    preferred: Optional[str] = None,
    **kwargs,
) -> Tuple[Optional[AttentionBackendBase], bool, str]:
    """
    Select backend from individual parameters (convenience wrapper).

    Args:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of KV attention heads.
        head_size: Size of each attention head.
        dtype: Input data type.
        preferred: Preferred backend name.
        **kwargs: Additional AttentionConfig parameters.

    Returns:
        Tuple of (backend_or_none, is_supported, reason).
    """
    config = AttentionConfig(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        **kwargs,
    )
    return select_backend(config, preferred=preferred)


########################################################
# FlashInfer TrtLLM-Gen Backend Implementation
########################################################


@BackendRegistry.register("trtllm-gen")
class FlashInferTrtllmGenBackend(AttentionBackendBase):
    """
    FlashInfer backend using trtllm-gen kernels.

    Uses flashinfer's trtllm_batch_context_with_kv_cache and
    trtllm_batch_decode_with_kv_cache APIs.

    Requirements:
        - Blackwell architecture (SM100 or SM103)
        - flashinfer package installed
    """

    def __init__(self):
        self._checker = TrtllmGenSupportChecker()
        self._layout = DEFAULT_KV_LAYOUT

    @property
    def name(self) -> str:
        return "trtllm-gen"

    @property
    def layout(self) -> str:
        """KV cache layout (HND or NHD)."""
        return self._layout

    def is_supported(self, config: AttentionConfig) -> Tuple[bool, str]:
        """Check if configuration is supported by this backend."""
        if not FLASHINFER_AVAILABLE:
            return False, "flashinfer package is not installed."
        return self._checker.is_supported(config)

    def _compute_scales(
        self,
        config: AttentionConfig,
        kv_scale_quant_orig: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float]:
        """
        Compute BMM scales for attention.

        Args:
            config: Attention configuration.
            kv_scale_quant_orig: KV cache dequantization scales.

        Returns:
            Tuple of (bmm1_scale, bmm2_scale).
        """
        # Base softmax scale
        if config.q_scaling != 1.0:
            softmax_scale = config.q_scaling / math.sqrt(config.head_size)
        else:
            softmax_scale = 1.0 / math.sqrt(config.head_size)

        bmm1_scale = softmax_scale
        bmm2_scale = 1.0

        # Incorporate KV cache dequantization scales
        # flashinfer accepts torch.Tensor for bmm1_scale and bmm2_scale
        # This avoids GPU sync during CUDA graph capture
        if kv_scale_quant_orig is not None and kv_scale_quant_orig.numel() >= 2:
            # Return tensor scales - flashinfer handles tensor multiplication internally
            k_dequant_scale = kv_scale_quant_orig[0:1].to(torch.float32)
            v_dequant_scale = kv_scale_quant_orig[1:2].to(torch.float32)
            bmm1_scale = softmax_scale * k_dequant_scale
            bmm2_scale = v_dequant_scale

        return bmm1_scale, bmm2_scale

    def run_context(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        cum_seq_lens_q: torch.Tensor,
        cum_seq_lens_kv: torch.Tensor,
        workspace: torch.Tensor,
        config: AttentionConfig,
        max_q_len: int,
        max_kv_len: int,
        batch_size: int,
        bmm1_scale: float,
        bmm2_scale: float,
        window_left: int = -1,
    ) -> torch.Tensor:
        """
        Execute context (prefill) phase using flashinfer.

        Calls flashinfer.prefill.trtllm_batch_context_with_kv_cache.
        """
        return flashinfer.prefill.trtllm_batch_context_with_kv_cache(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            batch_size=batch_size,
            cum_seq_lens_q=cum_seq_lens_q,
            cum_seq_lens_kv=cum_seq_lens_kv,
            window_left=window_left,
            kv_layout=self._layout,
        )

    def run_generation(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        workspace: torch.Tensor,
        config: AttentionConfig,
        max_kv_len: int,
        bmm1_scale: float,
        bmm2_scale: float,
        window_left: int = -1,
    ) -> torch.Tensor:
        """
        Execute generation (decode) phase using flashinfer.

        Calls flashinfer.decode.trtllm_batch_decode_with_kv_cache.
        """
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_kv_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            window_left=window_left,
            kv_layout=self._layout,
        )


########################################################
# Helper Functions
########################################################


def is_sm100_family() -> bool:
    """
    Check if SM version is in SM100 family (Blackwell).

    Returns:
        True if SM is 100 or 103.
    """
    sm = get_sm_version()
    return sm in (100, 103)


def _parse_request_types(
    host_request_types: torch.Tensor,
    num_seqs: int,
) -> Tuple[int, int]:
    """
    Parse request types to count context and generation requests.

    Args:
        host_request_types: Request types tensor (0=context, 1=generation).
        num_seqs: Total number of sequences.

    Returns:
        Tuple of (num_contexts, num_generations).
    """
    request_types = host_request_types.cpu().numpy()
    num_contexts = 0
    for idx in range(num_seqs):
        if request_types[idx] != 0:  # 0 = context
            break
        num_contexts += 1
    return num_contexts, num_seqs - num_contexts


def _get_block_tables(
    kv_cache_block_offsets: torch.Tensor,
    pool_index: int,
    start_idx: int,
    end_idx: int,
) -> torch.Tensor:
    """
    Extract block tables for a range of sequences.

    kv_cache_block_offsets shape: (num_pools, batch_size, 2, max_blocks_per_seq)
    where the "2" dimension is [primary_pool, secondary_pool].

    flashinfer expects block_tables shape: (batch_size, max_blocks_per_seq) with dtype int32.

    Args:
        kv_cache_block_offsets: Full block offsets tensor.
        pool_index: KV cache pool index.
        start_idx: Start sequence index.
        end_idx: End sequence index.

    Returns:
        Block tables tensor for the specified range, shape (num_seqs, max_blocks_per_seq), dtype int32.
    """
    if kv_cache_block_offsets.dim() == 4:
        # Shape: (num_pools, batch_size, 2, max_blocks_per_seq)
        # Extract primary pool (index 0) block offsets
        result = kv_cache_block_offsets[pool_index, start_idx:end_idx, 0, :].contiguous()
    elif kv_cache_block_offsets.dim() == 3:
        # Shape: (batch_size, 2, max_blocks_per_seq)
        result = kv_cache_block_offsets[start_idx:end_idx, 0, :].contiguous()
    else:
        # Shape: (batch_size, max_blocks_per_seq)
        result = kv_cache_block_offsets[start_idx:end_idx].contiguous()

    # flashinfer requires int32 block_tables
    return result.to(torch.int32)


def _create_kv_cache_placeholder(
    num_pages: int,
    num_kv_heads: int,
    page_size: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create placeholder KV cache tensors for flashinfer.

    The actual KV cache is managed externally; flashinfer
    accesses it via block tables.

    Args:
        num_pages: Number of KV cache pages.
        num_kv_heads: Number of KV heads.
        page_size: Tokens per page.
        head_size: Size of each head.
        dtype: Data type.
        device: Device.

    Returns:
        Tuple of (k_cache, v_cache) tensors.
    """
    shape = (num_pages, num_kv_heads, page_size, head_size)
    k_cache = torch.empty(shape, dtype=dtype, device=device)
    v_cache = torch.empty(shape, dtype=dtype, device=device)
    return k_cache, v_cache


########################################################
# Public API - Compatibility Functions
########################################################


def is_supported(
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: Optional[torch.dtype] = None,
    out_dtype: Optional[torch.dtype] = None,
    mask_type: Optional[int] = None,
    has_alibi: bool = False,
    is_padded: bool = False,
    use_paged_kv_cache: bool = True,
    tokens_per_block: int = 64,
    beam_width: int = 1,
    position_shift_enabled: bool = False,
    sink_token_length: int = 0,
    cross_attention: bool = False,
    cyclic_attention_window_size: Optional[int] = None,
    max_attention_window_size: Optional[int] = None,
    is_spec_decoding: bool = False,
    is_mla_enable: bool = False,
    is_fused_qkv: bool = True,
    update_kv_cache: bool = True,
    has_rotary_inv_freq: bool = False,
    has_rotary_cos_sin: bool = False,
    has_kv_scale: bool = False,
    has_cross_kv: bool = False,
    phase: str = "both",
) -> Tuple[bool, str]:
    """
    Check if trtllm-gen backend supports the given configuration.

    This is the compatibility function that wraps TrtllmGenSupportChecker.

    Args:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of KV attention heads.
        head_size: Size of each attention head.
        dtype: Input data type.
        kv_cache_dtype: KV cache data type.
        out_dtype: Output data type.
        mask_type: Attention mask type.
        has_alibi: Whether ALiBi is used.
        is_padded: Whether input is padded.
        use_paged_kv_cache: Whether paged KV cache is used.
        tokens_per_block: Tokens per KV cache block.
        beam_width: Beam search width.
        position_shift_enabled: Whether position shift is enabled.
        sink_token_length: Sink token length for StreamingLLM.
        cross_attention: Whether cross attention is used.
        cyclic_attention_window_size: Cyclic attention window size.
        max_attention_window_size: Max attention window size.
        is_spec_decoding: Whether speculative decoding is enabled.
        is_mla_enable: Whether MLA is enabled.
        is_fused_qkv: Whether QKV is fused.
        update_kv_cache: Whether KV cache update is enabled.
        has_rotary_inv_freq: Whether rotary_inv_freq is provided.
        has_rotary_cos_sin: Whether rotary_cos_sin is provided.
        has_kv_scale: Whether KV scales are provided.
        has_cross_kv: Whether cross KV is provided.
        phase: Phase to check ("context", "generation", or "both").

    Returns:
        Tuple of (is_supported, reason_if_not_supported).
    """
    # Build config from parameters
    config = AttentionConfig(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        out_dtype=out_dtype,
        mask_type=mask_type if mask_type is not None else 1,
        has_alibi=has_alibi,
        is_padded=is_padded,
        use_paged_kv_cache=use_paged_kv_cache,
        tokens_per_block=tokens_per_block,
        beam_width=beam_width,
        position_shift_enabled=position_shift_enabled,
        sink_token_length=sink_token_length,
        cross_attention=cross_attention or has_cross_kv,
        is_spec_decoding=is_spec_decoding,
        is_mla_enable=is_mla_enable,
        is_fused_qkv=is_fused_qkv,
        update_kv_cache=update_kv_cache,
    )

    return TrtllmGenSupportChecker.is_supported(config, phase)


# Compatibility aliases for workspace functions
def get_workspace_size_for_context(
    dtype: torch.dtype,
    max_num_seq: int,
    max_context_length: int,
    max_num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    rotary_embedding_dim: int = 0,
) -> int:
    """Calculate workspace size for context phase. (Compatibility function)"""
    return WorkspaceManager.get_context_workspace_size(
        dtype=dtype,
        max_num_seq=max_num_seq,
        max_context_length=max_context_length,
        max_num_tokens=max_num_tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        rotary_embedding_dim=rotary_embedding_dim,
    )


def get_workspace_size_for_generation(
    dtype: torch.dtype,
    max_num_seq: int,
    max_attention_window_size: int,
    max_num_tokens: int,
    max_blocks_per_sequence: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    rotary_embedding_dim: int = 0,
    multi_processor_count: int = 132,
) -> int:
    """Calculate workspace size for generation phase. (Compatibility function)"""
    return WorkspaceManager.get_generation_workspace_size(
        dtype=dtype,
        max_num_seq=max_num_seq,
        max_attention_window_size=max_attention_window_size,
        max_num_tokens=max_num_tokens,
        max_blocks_per_sequence=max_blocks_per_sequence,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        rotary_embedding_dim=rotary_embedding_dim,
        multi_processor_count=multi_processor_count,
    )


def get_workspace_size(
    dtype: torch.dtype,
    max_num_seq: int,
    max_context_length: int,
    max_attention_window_size: int,
    num_tokens: int,
    num_gen_tokens: int,
    max_blocks_per_sequence: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    rotary_embedding_dim: int = 0,
) -> int:
    """Calculate total workspace size. (Compatibility function)"""
    config = AttentionConfig(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        max_num_requests=max_num_seq,
        max_context_length=max_context_length,
        attention_window_size=max_attention_window_size,
        rotary_embedding_dim=rotary_embedding_dim,
    )
    return WorkspaceManager.get_workspace_size(
        config=config,
        num_tokens=num_tokens,
        num_gen_tokens=num_gen_tokens,
        max_blocks_per_sequence=max_blocks_per_sequence,
    )


########################################################
# Main Entry Point
########################################################


def trtllm_gen_attention(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    output: torch.Tensor,
    output_sf: Optional[torch.Tensor],
    workspace: Optional[torch.Tensor],
    sequence_length: torch.Tensor,
    host_past_key_value_lengths: torch.Tensor,
    host_total_kv_lens: torch.Tensor,
    context_lengths: torch.Tensor,
    host_context_lengths: torch.Tensor,
    host_request_types: torch.Tensor,
    kv_cache_block_offsets: Optional[torch.Tensor],
    host_kv_cache_pool_pointers: Optional[torch.Tensor],
    host_kv_cache_pool_mapping: Optional[torch.Tensor],
    kv_cache: Optional[torch.Tensor],  # Actual KV cache tensor from kv_cache_manager
    cache_indirection: Optional[torch.Tensor],
    kv_scale_orig_quant: Optional[torch.Tensor],
    kv_scale_quant_orig: Optional[torch.Tensor],
    out_scale: Optional[torch.Tensor],
    rotary_inv_freq: Optional[torch.Tensor],
    rotary_cos_sin: Optional[torch.Tensor],
    latent_cache: Optional[torch.Tensor],
    q_pe: Optional[torch.Tensor],
    block_ids_per_seq: Optional[torch.Tensor],
    attention_sinks: Optional[torch.Tensor],
    is_fused_qkv: bool,
    update_kv_cache: bool,
    predicted_tokens_per_seq: int,
    layer_idx: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    tokens_per_block: Optional[int],
    max_num_requests: int,
    max_context_length: int,
    attention_window_size: int,
    sink_token_length: int,
    beam_width: int,
    mask_type: int,
    quant_mode: int,
    q_scaling: float,
    position_embedding_type: int,
    rotary_embedding_dim: int,
    rotary_embedding_base: float,
    rotary_embedding_scale_type: int,
    rotary_embedding_scales: List[float],
    rotary_embedding_max_position_info: List[int],
    use_paged_context_fmha: bool,
    attention_input_type: Optional[int],
    is_mla_enable: bool,
    chunked_prefill_buffer_batch_size: Optional[int],
    q_lora_rank: Optional[int],
    kv_lora_rank: Optional[int],
    qk_nope_head_dim: Optional[int],
    qk_rope_head_dim: Optional[int],
    v_head_dim: Optional[int],
    mrope_rotary_cos_sin: Optional[torch.Tensor],
    mrope_position_deltas: Optional[torch.Tensor],
    mla_tensor_params: List[Optional[torch.Tensor]],
    attention_chunk_size: Optional[int],
    softmax_stats_tensor: Optional[torch.Tensor],
    spec_decoding_bool_params: List[bool],
    spec_decoding_tensor_params: List[Optional[torch.Tensor]],
    sparse_kv_indices: Optional[torch.Tensor],
    sparse_kv_offsets: Optional[torch.Tensor],
    sparse_attn_indices: Optional[torch.Tensor],
    sparse_attn_offsets: Optional[torch.Tensor],
    sparse_attn_indices_block_size: int,
    sparse_mla_topk: Optional[int],
    skip_softmax_threshold_scale_factor_prefill: Optional[float],
    skip_softmax_threshold_scale_factor_decode: Optional[float],
    skip_softmax_stat: Optional[torch.Tensor],
    cu_q_seqlens: Optional[torch.Tensor],
    cu_kv_seqlens: Optional[torch.Tensor],
    fmha_scheduler_counter: Optional[torch.Tensor],
    mla_bmm1_scale: Optional[torch.Tensor],
    mla_bmm2_scale: Optional[torch.Tensor],
    quant_q_buffer: Optional[torch.Tensor],
) -> None:
    """
    TrtLLM-Gen attention using flashinfer backend.

    This function is a drop-in replacement for thop.attention() but only
    supports trtllm-gen kernel (Blackwell architecture).

    It uses flashinfer's batch attention APIs:
    - flashinfer.prefill.trtllm_batch_context_with_kv_cache for context phase
    - flashinfer.decode.trtllm_batch_decode_with_kv_cache for generation phase

    IMPORTANT: Call is_supported() first to check if this backend can handle
    your configuration. If not supported, fallback to thop.attention().

    Args:
        q: Query tensor [num_tokens, hidden_dim].
        k: Key tensor (None if fused QKV).
        v: Value tensor (None if fused QKV).
        output: Output tensor [num_tokens, num_heads * head_size].
        output_sf: Output scale factor for FP4 output.
        workspace: Workspace tensor.
        sequence_length: KV sequence lengths.
        host_past_key_value_lengths: Past KV lengths (host).
        host_total_kv_lens: Total KV lengths.
        context_lengths: Context lengths.
        host_context_lengths: Context lengths (host).
        host_request_types: Request types (0=context, 1=generation).
        kv_cache_block_offsets: Block offsets for paged KV cache.
        host_kv_cache_block_offsets: Block offsets (host).
        host_kv_cache_pool_pointers: KV cache pool pointers.
        host_kv_cache_pool_mapping: KV cache pool mapping.
        ... (other parameters)
    """
    logger.debug(f"trtllm_gen_attention starts at layer {layer_idx}")

    # ========== 1. Build Configuration ==========
    page_size = tokens_per_block if tokens_per_block is not None else 64

    config = AttentionConfig(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        layer_idx=layer_idx,
        dtype=q.dtype,
        tokens_per_block=page_size,
        max_num_requests=max_num_requests,
        max_context_length=max_context_length,
        attention_window_size=attention_window_size,
        mask_type=mask_type,
        q_scaling=q_scaling,
        beam_width=beam_width,
        sink_token_length=sink_token_length,
        position_embedding_type=position_embedding_type,
        rotary_embedding_dim=rotary_embedding_dim,
        rotary_embedding_base=rotary_embedding_base,
        rotary_embedding_scale_type=rotary_embedding_scale_type,
        rotary_embedding_scales=rotary_embedding_scales,
        rotary_embedding_max_position_info=rotary_embedding_max_position_info,
        is_mla_enable=is_mla_enable,
        is_fused_qkv=is_fused_qkv,
        update_kv_cache=update_kv_cache,
    )

    # ========== 2. Get Backend ==========
    backend = BackendRegistry.get(DEFAULT_BACKEND)

    # ========== 3. Parse Request Types ==========
    num_seqs = host_context_lengths.size(0)
    num_tokens = q.size(0)

    attn_input_type = AttentionInputType.mixed
    if attention_input_type is not None:
        attn_input_type = AttentionInputType(attention_input_type)

    num_contexts, num_generations = _parse_request_types(host_request_types, num_seqs)

    # Calculate token counts
    # host_context_lengths is already on CPU, use int() instead of .item()
    host_ctx_lens = host_context_lengths.cpu()
    num_ctx_tokens = int(host_ctx_lens[:num_contexts].sum()) if num_contexts > 0 else 0
    num_gen_tokens = num_tokens - num_ctx_tokens

    # ========== 4. Compute Scales ==========
    bmm1_scale, bmm2_scale = backend._compute_scales(config, kv_scale_quant_orig)

    # ========== 5. Prepare Workspace ==========
    # trtllm-gen backend needs at least 16MB for counter workspace and scratch
    min_workspace_size = 16 * 1024 * 1024  # 16 MB

    # Calculate required workspace size
    max_blocks_per_sequence = 0
    if kv_cache_block_offsets is not None and kv_cache_block_offsets.numel() > 0:
        max_blocks_per_sequence = kv_cache_block_offsets.size(-1)

    required_workspace_size = WorkspaceManager.get_workspace_size(
        config=config,
        num_tokens=num_tokens,
        num_gen_tokens=num_gen_tokens,
        max_blocks_per_sequence=max_blocks_per_sequence,
    )
    required_workspace_size = max(required_workspace_size, min_workspace_size)

    # Check if we need to create/resize workspace
    current_workspace_size = (
        workspace.numel() * workspace.element_size() if workspace is not None else 0
    )

    if current_workspace_size < required_workspace_size:
        workspace = torch.zeros(required_workspace_size, dtype=torch.uint8, device=q.device)

    # ========== 6. Reshape Tensors ==========
    # Input q shape: [num_tokens, (num_heads + 2*num_kv_heads) * head_size] for fused QKV
    # Need: [num_tokens, num_heads, head_size]
    if is_fused_qkv:
        q_tensor = q.view(num_tokens, num_heads + 2 * num_kv_heads, head_size)
        query = q_tensor[:, :num_heads, :].contiguous()
    else:
        query = q.view(num_tokens, num_heads, head_size)

    out_tensor = output.view(num_tokens, num_heads, head_size)

    # Determine window_left for sliding window attention
    window_left = attention_window_size if attention_window_size < max_context_length else -1

    # Check KV cache availability
    # kv_cache is the actual tensor from kv_cache_manager.get_buffers()
    has_kv_cache = (
        kv_cache_block_offsets is not None
        and host_kv_cache_pool_pointers is not None
        and host_kv_cache_pool_mapping is not None
        and kv_cache is not None
    )

    # ========== 7. Context Phase (Prefill) ==========
    if num_contexts > 0 and attn_input_type != AttentionInputType.generation_only:
        logger.debug(
            f"[Layer {layer_idx}] Context phase: {num_contexts} requests, {num_ctx_tokens} tokens"
        )

        ctx_query = query[:num_ctx_tokens]
        ctx_output = out_tensor[:num_ctx_tokens]

        # Build cumulative sequence lengths
        ctx_lens = host_ctx_lens[:num_contexts].to(torch.int32)
        cum_seq_lens_q = torch.zeros(num_contexts + 1, dtype=torch.int32, device=q.device)
        cum_seq_lens_q[1:] = torch.cumsum(ctx_lens.to(q.device), dim=0)

        # KV sequence lengths
        ctx_kv_lens = sequence_length[:num_contexts].to(torch.int32)
        cum_seq_lens_kv = torch.zeros(num_contexts + 1, dtype=torch.int32, device=q.device)
        cum_seq_lens_kv[1:] = torch.cumsum(ctx_kv_lens.to(q.device), dim=0)

        # Use host tensors to avoid device-to-host sync during CUDA graph capture
        # ctx_lens is already on CPU (from host_ctx_lens)
        max_q_len = int(ctx_lens.max())
        # Use max_context_length as upper bound to avoid GPU sync
        max_kv_len = max_context_length

        if has_kv_cache and kv_cache is not None:
            # host_kv_cache_pool_mapping is on CPU, direct indexing is safe
            pool_index = int(host_kv_cache_pool_mapping[layer_idx, 0])
            ctx_block_tables = _get_block_tables(
                kv_cache_block_offsets, pool_index, 0, num_contexts
            )

            # Calculate number of blocks needed per sequence for context
            ctx_kv_lens_device = ctx_kv_lens.to(q.device)

            # Skip block_tables truncation during CUDA graph capture to avoid GPU-to-CPU sync.
            # The clamp operation below ensures safety anyway.
            if not torch.cuda.is_current_stream_capturing():
                num_blocks_per_seq = (ctx_kv_lens_device + page_size - 1) // page_size
                max_num_blocks = int(num_blocks_per_seq.max()) if num_contexts > 0 else 0

                # Truncate block_tables to only include valid blocks
                if max_num_blocks > 0 and max_num_blocks < ctx_block_tables.shape[1]:
                    ctx_block_tables = ctx_block_tables[:, :max_num_blocks].contiguous()

            # Clamp block indices to valid range to prevent illegal memory access
            max_pages = kv_cache.shape[0]
            ctx_block_tables = ctx_block_tables.clamp(0, max_pages - 1)

            # Run context phase
            ctx_result = backend.run_context(
                query=ctx_query,
                kv_cache=kv_cache,
                block_tables=ctx_block_tables,
                seq_lens=ctx_kv_lens_device,
                cum_seq_lens_q=cum_seq_lens_q,
                cum_seq_lens_kv=cum_seq_lens_kv,
                workspace=workspace,
                config=config,
                max_q_len=max_q_len,
                max_kv_len=max_kv_len,
                batch_size=num_contexts,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                window_left=window_left if window_left > 0 else -1,
            )
            ctx_output.copy_(ctx_result)

    # ========== 8. Generation Phase (Decode) ==========
    if num_generations > 0 and attn_input_type != AttentionInputType.context_only:
        logger.debug(
            f"[Layer {layer_idx}] Generation phase: "
            f"{num_generations} requests, {num_gen_tokens} tokens"
        )

        gen_query = query[num_ctx_tokens:]
        gen_output = out_tensor[num_ctx_tokens:]

        # KV sequence lengths for generation
        gen_kv_lens = sequence_length[num_contexts : num_contexts + num_generations].to(torch.int32)
        # Use max_context_length as upper bound to avoid GPU sync during CUDA graph capture
        max_kv_len = max_context_length

        if has_kv_cache and kv_cache is not None:
            # host_kv_cache_pool_mapping is on CPU, direct indexing is safe
            pool_index = int(host_kv_cache_pool_mapping[layer_idx, 0])
            gen_block_tables = _get_block_tables(
                kv_cache_block_offsets,
                pool_index,
                num_contexts,
                num_contexts + num_generations,
            )

            # Calculate number of blocks needed per sequence for generation
            gen_kv_lens_device = gen_kv_lens.to(q.device)

            # Skip block_tables truncation during CUDA graph capture to avoid GPU-to-CPU sync.
            # The clamp operation below ensures safety anyway.
            if not torch.cuda.is_current_stream_capturing():
                num_blocks_per_seq = (gen_kv_lens_device + page_size - 1) // page_size
                max_num_blocks = int(num_blocks_per_seq.max()) if num_generations > 0 else 0

                # Truncate block_tables to only include valid blocks
                if max_num_blocks > 0 and max_num_blocks < gen_block_tables.shape[1]:
                    gen_block_tables = gen_block_tables[:, :max_num_blocks].contiguous()

            # Clamp block indices to valid range to prevent illegal memory access
            max_pages = kv_cache.shape[0]
            gen_block_tables = gen_block_tables.clamp(0, max_pages - 1)

            # Run generation phase
            gen_result = backend.run_generation(
                query=gen_query,
                kv_cache=kv_cache,
                block_tables=gen_block_tables,
                seq_lens=gen_kv_lens_device,
                workspace=workspace,
                config=config,
                max_kv_len=max_kv_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                window_left=window_left if window_left > 0 else -1,
            )
            gen_output.copy_(gen_result.view(num_gen_tokens, num_heads, head_size))

    logger.debug(f"trtllm_gen_attention stops at layer {layer_idx}")
