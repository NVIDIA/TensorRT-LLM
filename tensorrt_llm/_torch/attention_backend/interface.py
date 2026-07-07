import copy
import weakref
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import (TYPE_CHECKING, Any, Dict, Generic, List, Literal, Optional,
                    Protocol, Tuple, Type, TypeVar, Union)

import torch
from typing_extensions import Self

if TYPE_CHECKING:
    from ..speculative.interface import SpecMetadata
    from ..speculative.spec_tree_manager import SpecTreeManager

from tensorrt_llm._utils import get_hf_rope_theta, maybe_pin_memory
from tensorrt_llm.functional import (AttentionMaskType, PositionEmbeddingType,
                                     RopeEmbeddingUtils, RotaryScalingType)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..memory_buffer_utils import Buffers
from ..metadata import KVCacheParams
from ..pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from ..pyexecutor.mamba_cache_manager import BaseMambaCacheManager
from ..pyexecutor.resource_manager import KVCacheManager
from ..pyexecutor.trace_log_utils import log_tensor_size
from ..utils import get_model_extra_attrs
from .sparse.params import SkipSoftmaxKernelParams, SparseMetadataParams

try:
    # Transformers v5
    from transformers.configuration_utils import ALLOWED_ATTENTION_LAYER_TYPES
except ImportError:
    # Transformers v4
    from transformers.configuration_utils import \
        ALLOWED_LAYER_TYPES as ALLOWED_ATTENTION_LAYER_TYPES


@dataclass
class AttentionRuntimeFeatures:
    chunked_prefill: bool = False
    cache_reuse: bool = False
    has_speculative_draft_tokens: bool = False
    # This is the chunk size for MLA chunked prefill, which splits KV cache into chunks.
    chunk_size: int = 0
    # The real chunk size for MLA chunked prefill is this value * chunk_size.
    chunked_prefill_buffer_batch_size: int = 4


# The type of requests in qkv passed to attention
# Please keep sync with AttentionInputType in cpp/tensorrt_llm/thop/attentionOp.cpp
class AttentionInputType(IntEnum):
    mixed = 0  # contains both context and generation
    context_only = 1
    generation_only = 2


@dataclass(kw_only=True)
class AttentionMetadata:
    """
    Metadata for the attention module.
    """

    # The max number of requests in a single batch.
    max_num_requests: int
    # The max number of tokens in all requests in a single batch.
    max_num_tokens: int
    # The max number of sequences in a single batch.
    max_num_sequences: Optional[int] = None
    # The KV cache manager.
    kv_cache_manager: Union[KVCacheManager, KVCacheManagerV2, None] = None
    # Draft KV cache manager for one-model speculative decoding with separate KV cache layouts
    draft_kv_cache_manager: Union[KVCacheManager, KVCacheManagerV2, None] = None
    mapping: Optional[Mapping] = None
    # Sparse settings for metadata allocation/update; dense metadata leaves it None.
    sparse_metadata_params: Optional[SparseMetadataParams] = None
    # Paged KV-cache block layout:
    # NHD: [max_num_pages, 2, page_size, num_kv_heads, head_dim]
    # HND: [max_num_pages, 2, num_kv_heads, page_size, head_dim]
    kv_layout: Literal["NHD", "HND"] = "HND"

    enable_flash_mla: bool = False
    enable_context_mla_with_cached_kv: bool = False
    # Whether CUDA graph is enabled.
    is_cuda_graph: bool = field(default=False, repr=False)

    # The length of each sequence in the batch for query.
    # The shape is (batch_size), and located on CPU memory.
    # For sub metadata of cross attention, it's automatically
    # initialized to seq_lens of parent metadata.
    seq_lens: Optional[torch.Tensor]  # Implemented using property

    # The number of context-phase sequences in the batch.
    num_contexts: int  # Implemented using property

    # The position of each token in each sequence.
    # May be None if positional embedding is applied outside the backend.
    position_ids: Optional[torch.Tensor] = None

    # The number of context-phase sequences in the batch.
    _num_contexts: int = field(init=False, default=0, repr=False)
    # The parameters for the KV cache.
    kv_cache_params: Optional[KVCacheParams] = None

    # The length of each sequence in the batch for key and value.
    # The shape is (batch_size), and located on CPU memory.
    # It defaults to seq_lens if not set.
    # Should only set explicitly for cross attention.
    seq_lens_kv: Optional[torch.Tensor]  # Implemented using property

    # Actual storage for seq_lens and seq_lens_kv
    _seq_lens: Optional[torch.Tensor] = field(init=False,
                                              repr=False,
                                              default=None)
    _seq_lens_kv: Optional[torch.Tensor] = field(init=False,
                                                 repr=False,
                                                 default=None)

    # A copy of seq_lens store on the GPU. Used in the logits
    # processor. Using 2 copies avoids a lot of extraneous
    # copies in flashinfer's prepare() implementation.
    _seq_lens_cuda: Optional[torch.Tensor] = field(init=False,
                                                   repr=False,
                                                   default=None)
    _seq_lens_kv_cuda: Optional[torch.Tensor] = field(init=False,
                                                      repr=False,
                                                      default=None)

    # For self attention, this is the sub metadata for cross attention
    # that works together in one model.
    # For cross attention, this is automatically inited to self,
    # and must not be set explicitly.
    cross: Optional["AttentionMetadata"] = None

    # The request ID of each sequence in the batch.
    # The shape is (batch_size).
    request_ids: Optional[List[int]] = None

    # The prompt length of each sequence in the batch.
    # For context-phase sequence, the value is its token number, which is same with `context_lens`.
    # For generation-phase sequence, the value is the token number of its context phase.
    # The shape is (batch_size) if provided.
    prompt_lens: Optional[List[int]] = None

    # Explicit query/KV sequence boundaries for attention kernels that operate
    # on packed varlen context inputs. These are sequence/segment boundaries,
    # not necessarily request boundaries.
    cu_q_seqlens: Optional[torch.Tensor] = None
    cu_kv_seqlens: Optional[torch.Tensor] = None

    # These fields indicate whether the runtime can use various features.
    # The kernels may or may not have different behaviors when these
    # are enabled.
    runtime_features: AttentionRuntimeFeatures = field(
        default_factory=AttentionRuntimeFeatures)

    # The number of tokens in each rank.
    all_rank_num_tokens: Optional[List[int]] = None

    # These fields are set when changing seq_lens and _num_contexts to avoid computation
    # during execution. If the calculation happens during execution, torch compile treats it
    # as DDS and fails to compile.
    _num_generations: int = field(init=False, default=0, repr=False)
    _num_ctx_tokens: int = field(init=False, default=0, repr=False)
    _num_tokens: int = field(init=False, default=0, repr=False)

    mamba_metadata: Optional[Any] = None
    mamba_chunk_size: int = 128

    # The number of tokens in the padded sequence.
    padded_num_tokens: Optional[int] = None

    # This buffer is currently only used for TrtllmAttentionMetadata.
    cache_indirection: Optional[torch.Tensor] = None
    cuda_graph_buffers: dict[str, list[torch.Tensor]] = None

    _saved_tensors: Dict[str, torch.Tensor] = field(init=False,
                                                    default_factory=dict)
    # The number of heads per kv head.
    num_heads_per_kv: Optional[int] = 1

    multi_item_part_lens: Optional[list[list[int]]] = None
    """Additional token layout information for multi-item scoring.

    Aggregates `TokensPrompt.multi_item_part_lens` for all requests in the batch,
    see `TokensPrompt` for details.
    """

    def __post_init__(self) -> None:
        if self.is_cross:
            assert self.cross is None or self.cross is self, "Cross attention metadata should not have sub metadata"
            self.cross = self
            return

        assert self.cross is None or type(self) is type(
            self.cross
        ), "Top level and cross attention sub metadata type mismatched"

    def on_update_kv_lens(self):
        pass

    def on_update(self):
        if (self._seq_lens is not None
                and self._seq_lens.shape[0] >= self.num_contexts
                and self.num_contexts >= 0):
            self._num_ctx_tokens = self._seq_lens[:self.num_contexts].sum(
            ).item()
            self._num_generations = self._seq_lens.shape[0] - self.num_contexts
        if self._seq_lens_kv is not None:
            self._num_tokens = self._seq_lens_kv.sum().item()
        elif self._seq_lens is not None:
            self._num_tokens = self._seq_lens.sum().item()

    @property
    def seq_lens(self) -> Optional[torch.Tensor]:
        return self._seq_lens

    @seq_lens.setter
    def seq_lens(self, value: Optional[torch.Tensor]):
        # If value not explicitly given, dataclass tries to initialize using class attribute
        value = value if value is not AttentionMetadata.seq_lens else None
        self._seq_lens = value
        self.on_update()

        # The model executor sets seq_lens to None initially.
        if self._seq_lens is not None:
            self._seq_lens = maybe_pin_memory(self._seq_lens)

            if self.is_cuda_graph and self._seq_lens_cuda is not None:
                # Very important: do not reallocate if we are using CUDA graphs.
                # This copy is safe because the batch size is guaranteed to not
                # change in the CUDA graph case. The seqlens can change if we
                # are doing spec decode.
                self._seq_lens_cuda.copy_(self._seq_lens, non_blocking=True)
            else:
                self._seq_lens_cuda = self._seq_lens.cuda(non_blocking=True)

        if self.has_cross_sub_metadata:
            self.cross._seq_lens = self._seq_lens
            self.cross._seq_lens_cuda = self._seq_lens_cuda

    @property
    def num_contexts(self) -> int:
        return self._num_contexts

    @num_contexts.setter
    def num_contexts(self, value: int):
        value = value if value is not AttentionMetadata.num_contexts else 0
        self._num_contexts = value
        self.on_update()

    @property
    def num_generations(self) -> int:
        return self._num_generations

    @num_generations.setter
    def num_generations(self, value: int):
        value = value if value is not AttentionMetadata.num_generations else 0
        self._num_generations = value
        self.on_update()

    @property
    def seq_lens_cuda(self):
        return self._seq_lens_cuda

    @property
    def seq_lens_kv(self) -> Optional[torch.Tensor]:
        return self._seq_lens_kv if self._seq_lens_kv is not None else self._seq_lens

    @seq_lens_kv.setter
    def seq_lens_kv(self, value: Optional[torch.Tensor]):
        value = value if value is not AttentionMetadata.seq_lens_kv else None
        self._seq_lens_kv = value
        self.on_update()
        # The model executor sets seqlens to None initially.
        if self._seq_lens_kv is not None:
            self._seq_lens_kv = maybe_pin_memory(self._seq_lens_kv)
            if self.is_cuda_graph and self._seq_lens_kv_cuda is not None:
                self._seq_lens_kv_cuda.copy_(self._seq_lens_kv,
                                             non_blocking=True)
            else:
                self._seq_lens_kv_cuda = self._seq_lens_kv.cuda(
                    non_blocking=True)

    @property
    def seq_lens_kv_cuda(self):
        return self._seq_lens_kv_cuda if self._seq_lens_kv_cuda is not None else self._seq_lens_cuda

    @property
    def context_lens(self) -> torch.Tensor:
        """
        The length of each context-phase query sequence in the batch.
        The shape is (num_contexts), where num_contexts is the number of context-phase sequences in the batch.
        """
        return self.seq_lens[:self.num_contexts]

    @property
    def num_seqs(self) -> int:
        """
        The number of sequences in the batch.
        """
        return self.seq_lens.shape[0]

    @property
    def is_cross(self) -> bool:
        """
        Is this metadata for cross attention.
        """
        return self.seq_lens is not self.seq_lens_kv

    @property
    def has_cross_sub_metadata(self) -> bool:
        return self.cross is not None and self.cross is not self

    @property
    def num_ctx_tokens(self) -> int:
        return self._num_ctx_tokens

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    def prepare(self):
        """
        Hook to be called before the forward step of the model.
        """
        self._prepare_mamba_metadata()

    def prepare_encoder_only(self) -> None:
        """Hook for encoder-only (no-KV-cache) forward setup.

        Defaults to the full ``prepare()``; backends with a leaner encoder-only
        path override this.
        """
        self.prepare()

    def _prepare_mamba_metadata(self):
        if self.mamba_metadata is False:
            return

        if self.mamba_metadata is None:
            if isinstance(self.kv_cache_manager, BaseMambaCacheManager):
                from ..modules.mamba.mamba2_metadata import Mamba2Metadata
                self.mamba_metadata = Mamba2Metadata(self.max_num_requests,
                                                     self.mamba_chunk_size)
            else:
                self.mamba_metadata = False
                return

        self.mamba_metadata.prepare(self)

    def create_cuda_graph_metadata(self,
                                   max_batch_size: int,
                                   sub_cross_metadata: bool = False,
                                   max_draft_tokens: int = 0,
                                   buffers=None,
                                   encode_only: bool = False) -> Self:
        """
        Creates metadata for CUDA graph execution.
        CUDA graphs require to use pre-allocated buffers for all tensors in fields.
        Please do not re-allocate any tensors stored inside AttentionMetadata
        after the initial warmup run when you're using CUDA graphs.

        When encode_only is True, initialize seq_lens as ones(max_batch_size)
        and leave num_contexts at the caller's discretion.
        """
        if self.is_cuda_graph:
            return self

        cuda_graph_metadata = copy.copy(self)
        cuda_graph_metadata.is_cuda_graph = True
        cuda_graph_metadata.cuda_graph_buffers = buffers
        if self.has_cross_sub_metadata:
            cuda_graph_metadata.cross = cuda_graph_metadata.cross.create_cuda_graph_metadata(
                max_batch_size, True, max_draft_tokens, buffers, encode_only)
        if not sub_cross_metadata:
            # Set to None to force the cuda graph metadata to allocate a tensor
            # with the correct batch size. See seq_lens setter for how this works.
            cuda_graph_metadata._seq_lens_cuda = None
            if encode_only:
                # Encoder: variable seq_lens per batch; the runner in-place
                # updates them via the seq_lens setter each replay.
                cuda_graph_metadata.seq_lens = torch.ones((max_batch_size, ),
                                                          dtype=torch.int)
            else:
                cuda_graph_metadata.seq_lens = torch.ones(
                    (max_batch_size, ),
                    dtype=torch.int) * (1 + max_draft_tokens)
        if self.is_cross:
            cuda_graph_metadata.seq_lens_kv = torch.zeros((max_batch_size, ),
                                                          dtype=torch.int)
        if self.enable_flash_mla:
            if self.kv_cache_manager is not None:
                cuda_graph_metadata.block_ids_per_seq = torch.zeros(
                    [
                        self.kv_cache_manager.max_batch_size,
                        self.kv_cache_manager.max_blocks_per_seq
                    ],
                    dtype=torch.int32,
                    device='cuda',
                )

        if not encode_only:
            # Decoder CUDA graphs are always generation-only (no context
            # requests). Encoder CUDA graphs are the opposite (all context);
            # the caller sets num_contexts = padded_batch_size.
            cuda_graph_metadata.num_contexts = 0
        cuda_graph_metadata.__post_init__()
        return cuda_graph_metadata

    def prepare_for_spec_dec(self, *fields) -> None:
        assert len(self._saved_tensors) == 0
        for f in fields:
            v = getattr(self, f)
            assert isinstance(v, torch.Tensor)
            self._saved_tensors[f] = v
            setattr(self, f, v.clone())

    def restore_from_spec_dec(self) -> None:
        for f, v in self._saved_tensors.items():
            setattr(self, f, v)
        self._saved_tensors.clear()

    def update_spec_dec_param(
            self,
            batch_size,
            is_spec_decoding_enabled,
            is_spec_dec_tree,
            is_spec_dec_dynamic_tree,
            max_draft_len,
            max_total_draft_tokens,
            model_is_wrapped: bool = False,
            spec_metadata: Optional['SpecMetadata'] = None,
            spec_tree_manager: Optional['SpecTreeManager'] = None,
            num_contexts: int = 0):
        """
        Hook to be called when using TRTLLM attention backend in spec-dec mode.

        ``num_contexts`` is the number of context (prefill) requests in the
        mixed batch, occupying the leading rows of slot-storage buffers.
        Backends that consume gen-only slots (e.g. dynamic tree) must skip
        these rows to align with the XQA kernel's expected row layout.
        """

    def update_helix_param(
        self,
        helix_position_offsets: List[int],
        helix_is_inactive_rank: List[bool],
    ) -> None:
        """
        Hook to be called when using helix parallelism.
        """

    def create_cross_metadata(
        self,
        encoder_seq_lens: List[int],
        cross_kv_cache_manager: Union[KVCacheManager, KVCacheManagerV2,
                                      None] = None,
        encoder_num_cached_tokens_per_seq: Optional[List[int]] = None,
    ) -> "AttentionMetadata":
        """Build a sub-metadata instance for cross-attention.

        The returned metadata shares Q-side fields (``seq_lens``,
        ``request_ids``, ``num_contexts``) with ``self`` (the decoder
        self-attention metadata) and overrides the K/V-side with the encoder
        lengths so that ``returned.is_cross is True``.

        This is intended to be called by the runtime / unit tests once the
        encoder lengths and cross-pool KV cache manager are known. The
        returned object is a *new* metadata instance (not stored on
        ``self.cross``); callers can attach it to ``self.cross`` if desired.

        Args:
            encoder_seq_lens: Per-request encoder sequence lengths. On the
                first decoder context step this is the full encoder length;
                on generation steps it should be ``0`` (no new K/V tokens to
                add to the cross pool — the encoder K/V are already cached).
            cross_kv_cache_manager: KV cache manager for the cross pool.
                When ``None``, the returned metadata uses the stateless
                (no-KV-cache) path (suitable for unit tests).
            encoder_num_cached_tokens_per_seq: Per-request count of encoder
                K/V tokens already present in the cross pool. ``None``
                defaults to 0 (context phase, nothing cached yet).

        Returns:
            A new ``AttentionMetadata`` of the same subclass as ``self``,
            with ``seq_lens_kv`` set to ``encoder_seq_lens`` so that
            ``is_cross`` becomes ``True``.
        """
        cross_md = copy.copy(self)
        cross_md._saved_tensors = {}
        if self.is_cuda_graph:
            # Cross-attention has K/V lengths from the encoder, while
            # self-attention has K/V lengths from the decoder. Keep their
            # CUDA graph metadata buffers separate so preparing cross metadata
            # cannot overwrite self-attention sequence lengths.
            cross_md.cuda_graph_buffers = Buffers()
        cross_md._seq_lens_kv_cuda = None
        cross_md.cross = None
        self._update_cross_metadata(
            cross_md,
            encoder_seq_lens,
            cross_kv_cache_manager,
            encoder_num_cached_tokens_per_seq,
            base_kv_cache_params=self.kv_cache_params,
            block_ids_per_seq=None,
        )
        cross_md.__post_init__()
        return cross_md

    def _update_cross_metadata(
        self,
        cross_md: "AttentionMetadata",
        encoder_seq_lens: List[int],
        cross_kv_cache_manager: Union[KVCacheManager, KVCacheManagerV2, None],
        encoder_num_cached_tokens_per_seq: Optional[List[int]],
        *,
        base_kv_cache_params: Optional[KVCacheParams],
        block_ids_per_seq: Optional[List[list]],
    ) -> "AttentionMetadata":
        encoder_seq_lens_tensor = torch.tensor(encoder_seq_lens,
                                               dtype=torch.int)
        cross_md.kv_cache_manager = cross_kv_cache_manager
        cross_md._seq_lens = self.seq_lens
        cross_md._seq_lens_cuda = self.seq_lens_cuda
        cross_md.seq_lens_kv = encoder_seq_lens_tensor

        # Cross-attention keeps decoder-side prompt lengths for the Q-side
        # context metadata. Encoder-side lengths are represented by
        # seq_lens_kv and kv_cache_params.num_cached_tokens_per_seq.
        cross_md.prompt_lens = self.prompt_lens

        if encoder_num_cached_tokens_per_seq is not None:
            cross_md.kv_cache_params = KVCacheParams(
                use_cache=(base_kv_cache_params.use_cache
                           if base_kv_cache_params is not None else
                           (cross_kv_cache_manager is not None)),
                num_cached_tokens_per_seq=list(
                    encoder_num_cached_tokens_per_seq),
                block_ids_per_seq=block_ids_per_seq,
                host_max_attention_window_sizes=(
                    base_kv_cache_params.host_max_attention_window_sizes
                    if base_kv_cache_params is not None else None),
                host_sink_token_length=(
                    base_kv_cache_params.host_sink_token_length
                    if base_kv_cache_params is not None else None),
                num_extra_kv_tokens=(base_kv_cache_params.num_extra_kv_tokens if
                                     base_kv_cache_params is not None else 0),
            )

        cross_md.request_ids = self.request_ids
        cross_md.num_contexts = self.num_contexts
        return cross_md

    def update_cross_metadata(
        self,
        encoder_seq_lens: List[int],
        cross_kv_cache_manager: Union[KVCacheManager, KVCacheManagerV2, None],
        encoder_num_cached_tokens_per_seq: Optional[List[int]] = None,
    ) -> "AttentionMetadata":
        """Refresh an existing CUDA graph cross-attention sub-metadata."""
        if not self.has_cross_sub_metadata:
            raise RuntimeError(
                "CUDA graph cross-attention metadata has not been initialized.")

        cross_md = self.cross
        assert cross_md is not None
        base_kv_cache_params = cross_md.kv_cache_params
        block_ids_per_seq = (base_kv_cache_params.block_ids_per_seq
                             if base_kv_cache_params is not None else None)
        return self._update_cross_metadata(
            cross_md,
            encoder_seq_lens,
            cross_kv_cache_manager,
            encoder_num_cached_tokens_per_seq,
            base_kv_cache_params=base_kv_cache_params,
            block_ids_per_seq=block_ids_per_seq,
        )

    def update_for_spec_dec(self) -> None:
        """
        Hook to be called during forward when using spec-dec one-model mode.
        """

    @staticmethod
    def get_empty(buffers: Buffers,
                  tensor_shape: list[int],
                  dtype: torch.dtype,
                  cache_name: str,
                  capture_graph: bool = False) -> torch.Tensor:
        """
        Finds a compatible, reusable buffer from a cache or creates a new one.

        This function searches for a pre-allocated tensor (buffer) that can be
        reused for an operation involving a tensor with the shape of `tensor_shape`.

        The compatibility rules are: The buffer's total elements must be >= tensor_shape's.

        If a compatible buffer is found, it's returned immediately. Otherwise, a new
        buffer is allocated on the 'cuda' device with the give properties of 'tensor_shape' and 'dtype'.

        Args:
            tensor_shape: The required shape.
            dtype: The required dtype.
            cache_name: The key for the specific list of buffers to search in.
        Returns:
            An existing compatible buffer or a newly created one.
        """
        if buffers is None:
            return torch.zeros(tensor_shape, device='cuda', dtype=dtype)

        return buffers.get_buffer(tensor_shape, dtype, cache_name,
                                  capture_graph)

    @staticmethod
    def get_empty_like(buffers,
                       like_tensor: torch.Tensor,
                       cache_name: str,
                       capture_graph: bool = False) -> torch.Tensor:
        return AttentionMetadata.get_empty(
            buffers,
            like_tensor.shape,
            dtype=like_tensor.dtype,
            cache_name=cache_name,
            capture_graph=capture_graph,
        )


class PositionalEmbedder(Protocol):
    """
    A callable that can apply positional embedding
    """

    def __call__(self, position_ids: torch.Tensor, q: torch.Tensor,
                 k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ...


@dataclass(kw_only=True, unsafe_hash=True)
class RopeParams:
    dim: int = 0
    theta: float = 10000.0
    alpha: float = 1.0
    scale_type: RotaryScalingType = RotaryScalingType.none
    scale: float = 1.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    short_m_scale: float = 1.0
    long_m_scale: float = 1.0
    max_positions: int = 1024
    original_max_positions: int = 1024
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    mscale_all_dim: float = 0.0
    short_factor: Optional[Tuple[float]] = None
    long_factor: Optional[Tuple[float]] = None
    max_seq_len: Optional[int] = None
    duplicate_data: bool = False

    @staticmethod
    def from_config(config) -> "RopeParams":
        rope_params = RopeParams()

        hf_rope_parameters = getattr(config, 'rope_parameters', None)
        if hf_rope_parameters is not None:
            if set(hf_rope_parameters.keys()).issubset(
                    ALLOWED_ATTENTION_LAYER_TYPES):
                # Per-layer-type RoPE config (e.g. Gemma3 in transformers 5.x).
                # Pick "full_attention" as the default; callers override theta
                # for sliding-window layers independently.
                if "full_attention" in hf_rope_parameters:
                    flat = hf_rope_parameters["full_attention"]
                else:
                    fallback_key = next(iter(hf_rope_parameters))
                    logger.warning(
                        f"Per-layer-type rope_parameters has no 'full_attention' entry; "
                        f"falling back to '{fallback_key}'. Available layer types: "
                        f"{list(hf_rope_parameters.keys())}.")
                    flat = hf_rope_parameters[fallback_key]
                config.update(flat)
            else:
                config.update(hf_rope_parameters)

        # get rotary parameters.
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        head_dim = getattr(config, 'head_dim', None)
        if not isinstance(head_dim, int):
            head_dim = hidden_size // num_attention_heads
        rope_scaling = getattr(config, 'rope_scaling', None)
        rope_params.max_positions = config.max_position_embeddings
        rope_params.theta = get_hf_rope_theta(config, 10000.0)
        rope_percentage = (getattr(config, 'rotary_pct', None)
                           or getattr(config, 'partial_rotary_factor', None)
                           or 1.0)
        # rotary embedding dim.
        qk_rope_head_dim = getattr(config, 'qk_rope_head_dim', None)
        rope_params.dim = (getattr(config, 'rotary_dim', None)
                           or getattr(config, 'rotary_emb_base', None)
                           or qk_rope_head_dim
                           or int(head_dim * rope_percentage))
        if qk_rope_head_dim is not None:
            rope_params.duplicate_data = True
        # rotary scaling.
        rope_params.scale_type = RotaryScalingType.none
        rope_params.scale = 1.0
        if rope_scaling is not None:
            rope_params.alpha = rope_scaling.get("alpha", 1.0)
            rotary_scaling_type = rope_scaling.get(
                "type", None) or rope_scaling.get("rope_type")
            rope_params.scale_type = RotaryScalingType.from_string(
                rotary_scaling_type)
            rope_params.scale = rope_scaling.get("factor", 1.0)
            rope_params.low_freq_factor = rope_scaling.get(
                "low_freq_factor", 1.0)
            rope_params.high_freq_factor = rope_scaling.get(
                "high_freq_factor", 4.0)
            rope_params.original_max_positions = getattr(
                config,
                "original_max_position_embeddings", None) or rope_scaling.get(
                    "original_max_position_embeddings", None) or 1024
            rope_params.beta_fast = rope_scaling.get("beta_fast", 32)
            rope_params.beta_slow = rope_scaling.get("beta_slow", 1)
            rope_params.mscale = rope_scaling.get("mscale", 1.0)
            rope_params.mscale_all_dim = rope_scaling.get("mscale_all_dim", 0.0)
            if "short_factor" in rope_scaling:
                rope_params.short_factor = tuple(rope_scaling["short_factor"])
            if "long_factor" in rope_scaling:
                rope_params.long_factor = tuple(rope_scaling["long_factor"])
        # Workaround for DeepSeek V3 Lite since its rope_scaling is null in
        # config.json.
        elif config.model_type == "deepseek_v3":
            rope_params.scale_type = RotaryScalingType.yarn
        # Other metdadata for RoPE.
        rope_params.max_seq_len = getattr(config, 'max_seq_len', None)

        return rope_params

    def create_rope_const_params(self, interleave: bool = True):
        if self.dim == 0:
            return None, None

        RopeConstParams = namedtuple("RopeConstParams", ["inv_freq", "cos_sin"])
        extra_attrs = get_model_extra_attrs()
        if extra_attrs is not None:
            cache = extra_attrs.setdefault("rope_const_params", {})
            rope_const_params = cache.get((self, interleave), None)
            if rope_const_params is not None and rope_const_params.cos_sin(
            ) is not None:
                return (
                    rope_const_params.inv_freq()
                    if rope_const_params.inv_freq is not None else None,
                    rope_const_params.cos_sin(),
                )

        if self.scale_type == RotaryScalingType.yarn:
            rope_inv_freq, rope_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_yarn(
                self.max_positions,
                self.dim,
                self.theta,
                self.scale,
                self.original_max_positions,
                self.beta_fast,
                self.beta_slow,
                self.mscale,
                self.mscale_all_dim,
                self.duplicate_data,
            )
        elif self.scale_type == RotaryScalingType.longrope:
            rope_inv_freq, rope_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_long_rope(
                num_pos=self.max_positions,
                dim=self.dim,
                theta=self.theta,
                original_max_pos=self.original_max_positions,
                short_factor=self.short_factor,
                long_factor=self.long_factor,
                max_seq_len=self.max_seq_len,
                duplicate_data=self.duplicate_data,
            )
        else:
            rope_inv_freq, rope_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
                self.max_positions,
                self.dim,
                self.theta,
                self.scale,
                self.scale_type,
                rope_scaling_config={
                    "factor": self.scale,
                    "alpha": self.alpha,
                    "low_freq_factor": self.low_freq_factor,
                    "high_freq_factor": self.high_freq_factor,
                    "original_max_position_embeddings":
                    self.original_max_positions,
                },
                duplicate_data=self.duplicate_data,
            )
        if rope_inv_freq is not None:
            rope_inv_freq = torch.tensor(
                rope_inv_freq,
                dtype=torch.float32,
                device='cuda',
            )
        if not interleave:
            rope_cos_sin = rope_cos_sin.reshape(
                self.max_positions, -1,
                2)[:, :self.dim // 2, :].transpose(0, 2, 1).reshape(1, -1)
        rope_cos_sin = torch.tensor(
            rope_cos_sin,
            dtype=torch.float32,
            device='cuda',
        )
        if extra_attrs is not None:
            cache[(self, interleave)] = RopeConstParams(
                weakref.ref(rope_inv_freq)
                if rope_inv_freq is not None else None,
                weakref.ref(rope_cos_sin),
            )
        # One-shot log on cache miss (typically 2-4 times per model load).
        log_tensor_size("rope/new_table",
                        rope_cos_sin,
                        max_pos=self.max_positions,
                        dim=self.dim,
                        theta=self.theta,
                        scale_type=self.scale_type,
                        interleave=interleave)
        return rope_inv_freq, rope_cos_sin


@dataclass(kw_only=True, frozen=True)
class PositionalEmbeddingParams:
    type: PositionEmbeddingType
    embedder: Optional[PositionalEmbedder] = None

    # RoPE params
    rope: Optional[RopeParams] = None
    is_neox: bool = True

    # mRoPE params
    mrope_section: Optional[List[int]] = None
    mrope_interleaved: bool = False

    def __post_init__(self) -> None:
        if self.type.is_deferred():
            assert self.embedder is not None, f"{self.type} requires a not-none external embedder"
        else:
            assert self.embedder is None, f"Embedder must be None for {self.type}"

        if self.type.is_rope():
            assert self.rope is not None, f"{self.type} requires a not-none rope"


TMetadata = TypeVar("TMetadata", bound=AttentionMetadata)


class PredefinedAttentionMask(str, Enum):
    """
    Predefined attention mask types

    Attributes:
        CAUSAL: Use causal mask.
        FULL: do not use any mask
    """
    CAUSAL = "causal"
    FULL = "full"


class CustomAttentionMask(str, Enum):
    """
    Custom attention mask types
    """
    CUSTOM = "custom"


AttentionMask = Union[PredefinedAttentionMask, CustomAttentionMask]


@dataclass(kw_only=True, slots=True)
class SparsePrediction:
    """Sparse KV / attention indices predicted by the framework backends.

    RocketKV and DSA produce these from ``sparse_kv_predict`` /
    ``sparse_attn_predict``, telling the attention op which KV tokens to keep
    and which blocks to attend to. Backends that don't predict leave
    ``AttentionForwardArgs.sparse_prediction`` at its default-constructed value
    (all-``None`` / ``0`` fields).
    """
    sparse_kv_indices: Optional[torch.Tensor] = None
    sparse_kv_offsets: Optional[torch.Tensor] = None
    sparse_attn_indices: Optional[torch.Tensor] = None
    sparse_attn_offsets: Optional[torch.Tensor] = None
    sparse_attn_indices_block_size: int = 0
    # DeepSeek-V4 sparse-MLA only: per-token compressed top-k lengths and the
    # base pointer of the compressed KV cache pool (compress_ratio > 1).
    sparse_mla_topk_lens: Optional[torch.Tensor] = None
    compressed_kv_cache_pool_ptr: Optional[int] = None


@dataclass(kw_only=True, slots=True)
class AttentionForwardArgs:
    """Per-forward optional arguments for attention backends."""

    output: Optional[torch.Tensor] = None
    output_sf: Optional[torch.Tensor] = None

    out_scale: Optional[torch.Tensor] = None
    out_scale_sf: Optional[torch.Tensor] = None
    kv_scale_orig_quant: Optional[torch.Tensor] = None
    kv_scale_quant_orig: Optional[torch.Tensor] = None

    attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL
    attention_input_type: AttentionInputType = AttentionInputType.mixed
    attention_window_size: Optional[int] = None
    attention_mask_data: Optional[torch.Tensor] = None
    attention_sinks: Optional[torch.Tensor] = None
    relative_attention_bias: Optional[torch.Tensor] = None
    relative_attention_max_distance: int = 0
    cross_kv: Optional[torch.Tensor] = None

    latent_cache: Optional[torch.Tensor] = None
    q_pe: Optional[torch.Tensor] = None
    mrope_rotary_cos_sin: Optional[torch.Tensor] = None
    mrope_position_deltas: Optional[torch.Tensor] = None

    softmax_stats_tensor: Optional[torch.Tensor] = None
    chunked_prefill_buffer_batch_size: int = 1

    cu_q_seqlens: Optional[torch.Tensor] = None
    cu_kv_seqlens: Optional[torch.Tensor] = None
    fmha_scheduler_counter: Optional[torch.Tensor] = None
    # Testing only: skip the RoPE step of MLA generation (the standalone harness
    # feeds a pre-RoPE'd fused_q). The TRTLLM backend then appends the new latent
    # and inits the trtllm-gen scheduler buffers itself.
    skip_mla_rope_generation: bool = False

    mla_bmm1_scale: Optional[torch.Tensor] = None
    mla_bmm2_scale: Optional[torch.Tensor] = None
    quant_q_buffer: Optional[torch.Tensor] = None
    # Per-tensor FP8 scale (fp32 [1]) for the fused DSv4 FP8-Q-quant path.
    # When non-None alongside `quant_q_buffer`, the C++ op skips
    # `quantizeCopyInputToFp8Kernel`.
    quant_scale_qkv: Optional[torch.Tensor] = None

    sage_attn_num_elts_per_blk_q: int = 0
    sage_attn_num_elts_per_blk_k: int = 0
    sage_attn_num_elts_per_blk_v: int = 0
    sage_attn_qk_int8: bool = False

    topk_indices: Optional[torch.Tensor] = None

    is_fused_qkv: bool = False
    update_kv_cache: bool = True
    # Optional normalized diffusion timestep for timestep-varying sparse attention.
    timestep: Optional[torch.Tensor] = None

    sparse_prediction: SparsePrediction = field(
        default_factory=SparsePrediction)
    skip_softmax_kernel_params: SkipSoftmaxKernelParams = field(
        default_factory=SkipSoftmaxKernelParams)

    @property
    def mask_type(self) -> int:
        """Integer mask type accepted by the C++ attention op
        (``causal`` or ``padding``)."""
        if self.attention_mask == PredefinedAttentionMask.CAUSAL:
            return int(AttentionMaskType.causal)
        if self.attention_mask == PredefinedAttentionMask.FULL:
            return int(AttentionMaskType.padding)
        raise ValueError(
            f"Unexpected attention mask type: {self.attention_mask!r}")


_ATTENTION_FORWARD_ARGS_FIELDS = frozenset(
    AttentionForwardArgs.__dataclass_fields__)


def merge_attention_forward_args(
    forward_args: Optional[AttentionForwardArgs],
    kwargs: Dict[str, Any],
) -> AttentionForwardArgs:
    """Merge legacy attention kwargs into explicit forward arguments."""

    unknown_kwargs = sorted(set(kwargs) - _ATTENTION_FORWARD_ARGS_FIELDS)
    if unknown_kwargs:
        raise ValueError(
            f"Unknown attention forward arguments: {unknown_kwargs}")

    if forward_args is not None:
        if kwargs:
            raise ValueError(
                "Pass attention forward options either through forward_args "
                f"or as legacy kwargs, not both: {sorted(kwargs)}")
        return forward_args

    return AttentionForwardArgs(**kwargs)


class AttentionBackend(Generic[TMetadata]):
    """
    Base class for attention backends.
    """
    Metadata: Type[TMetadata] = AttentionMetadata

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        **kwargs,
    ):
        """
        Initialize the backend.
        Args:
            layer_idx (int): The index of the attention layer in the model.
            num_heads (int): The number of query heads.
            head_dim (int): The size of each attention head (hidden_size // num_heads).
            num_kv_heads (int): The number of kv heads. Defaults to num_heads if None.
            quant_config (QuantConfig): Optional quantization configuration. If None, no quantization is applied.
        """
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or self.num_heads
        self.quant_config = quant_config

    def update_quant_config(self, new_quant_config: Optional[QuantConfig]):
        """
        To support mixed quantization mode, self.quant_config can be modified after __init__ is called.
        Any states or set up related to self.quant_config must be moved to this function, which is called
        after self.quant_config is reset.
        """
        self.quant_config = new_quant_config

    def forward(self,
                q: torch.Tensor,
                k: Optional[torch.Tensor],
                v: Optional[torch.Tensor],
                metadata: TMetadata,
                forward_args: Optional[AttentionForwardArgs] = None,
                **kwargs) -> torch.Tensor:
        """
        Update KV Cache and perform the attention operation.
        Args:
            q (torch.Tensor): Query tensor with shape (num_q_tokens, num_heads * head_dim),
                              or QKV tensor with shape (num_tokens, (num_heads + 2 * num_kv_heads) * head_dim).
            k (Optional[torch.Tensor]): Key tensor with shape (num_new_kv_tokens, num_kv_heads * head_dim),
                                        or KV tensor with shape (num_new_kv_tokens, (2 * num_kv_heads) * head_dim),
                                        or None: if QKV tensor is provided, or there's no new kv token.
            v (Optional[torch.Tensor]): Value tensor with shape (num_new_kv_tokens, num_kv_heads * head_dim),
                                        or None if QKV tensor is provided, or there's no new kv token.
            metadata (AttentionMetadata): Metadata for the attention operation.
            forward_args (AttentionForwardArgs): Per-forward optional attention arguments.
        Returns:
            torch.Tensor with shape (num_q_tokens, num_heads * head_dim)
        """
        raise NotImplementedError

    @classmethod
    def support_fused_rope(cls) -> bool:
        return False

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False

    @classmethod
    def support_mla(cls) -> bool:
        return False

    @classmethod
    def support_multi_item_scoring(cls) -> bool:
        return False

    def create_output(self, q: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        """
        Create the output tensors for the attention operation.
        """
        num_tokens = q.shape[0]
        hidden_size = self.num_heads * self.head_dim
        return [q.new_empty([num_tokens, hidden_size], dtype=q.dtype)]


@dataclass(kw_only=True, unsafe_hash=True)
class MLAParams:
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    qk_nope_head_dim: int = 0
    v_head_dim: int = 0
    rope_append: bool = True
    predicted_tokens_per_seq: int = 1
    chunked_prefill_buffer_batch_size: int = 1
    hidden_size: int = 0
