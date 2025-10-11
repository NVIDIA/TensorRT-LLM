import copy
import weakref
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import (TYPE_CHECKING, Dict, Generic, List, Optional, Protocol,
                    Tuple, Type, TypeVar, Union)

import torch
from typing_extensions import Self

if TYPE_CHECKING:
    from ..speculative.utils import SpecDecodingTensor

from tensorrt_llm.functional import (PositionEmbeddingType, RopeEmbeddingUtils,
                                     RotaryScalingType)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..metadata import KVCacheParams
from ..pyexecutor.resource_manager import KVCacheManager
from ..utils import get_model_extra_attrs


@dataclass
class AttentionRuntimeFeatures:
    chunked_prefill: bool = False
    cache_reuse: bool = False
    has_speculative_draft_tokens: bool = False
    chunk_size: int = 0  # this is the chunk size for MLA chunked prefill, it will split kv cache into chunks to save global memory.
    chunked_prefill_buffer_batch_size: int = 4  # real chunk size for MLA chunked prefill is chunked_prefill_buffer_batch_size * chunk_size.


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
    kv_cache_manager: KVCacheManager
    mapping: Optional[Mapping] = None

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

    # The number of tokens in the padded sequence.
    padded_num_tokens: Optional[int] = None

    # This buffer is currently only used for TrtllmAttentionMetadata.
    cache_indirection: Optional[torch.Tensor] = None
    cuda_graph_buffers: dict[str, list[torch.Tensor]] = None

    _saved_tensors: Dict[str, torch.Tensor] = field(init=False,
                                                    default_factory=dict)
    sparse_attention_config: Optional["SparseAttentionConfig"] = None

    def __post_init__(self) -> None:
        if self.is_cross:
            assert self.cross is None or self.cross is self, "Cross attention metadata should not have sub metadata"
            self.cross = self
            return

        assert self.cross is None or type(self) is type(
            self.cross
        ), "Top level and cross attention sub metadata type mismatched"

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
            self._seq_lens = self._seq_lens.pin_memory()

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
            self._seq_lens_kv = self._seq_lens_kv.pin_memory()
            self._seq_lens_kv_cuda = self._seq_lens_kv.cuda(non_blocking=True)

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

    def create_cuda_graph_metadata(self,
                                   max_batch_size: int,
                                   sub_cross_metadata: bool = False,
                                   max_draft_tokens: int = 0,
                                   buffers=None) -> Self:
        """
        Creates metadata for CUDA graph execution.
        CUDA graphs require to use pre-allocated buffers for all tensors in fields.
        Please do not re-allocate any tensors stored inside AttentionMetadata
        after the initial warmup run when you're using CUDA graphs.
        """
        if self.is_cuda_graph:
            return self

        cuda_graph_metadata = copy.copy(self)
        cuda_graph_metadata.is_cuda_graph = True
        cuda_graph_metadata.cuda_graph_buffers = buffers
        if self.has_cross_sub_metadata:
            cuda_graph_metadata.cross = cuda_graph_metadata.cross.create_cuda_graph_metadata(
                max_batch_size, True)
        if not sub_cross_metadata:
            # Set to None to force the cuda graph metadata to allocate a tensor
            # with the correct batch size. See seq_lens setter for how this works.
            cuda_graph_metadata._seq_lens_cuda = None
            cuda_graph_metadata.seq_lens = torch.ones(
                (max_batch_size, ), dtype=torch.int) * (1 + max_draft_tokens)
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
            is_spec_decoding_enabled,
            is_spec_dec_tree,
            is_spec_dec_dynamic_tree,
            max_draft_tokens,
            spec_decoding_tensor: Optional['SpecDecodingTensor'] = None):
        """
        Hook to be called when using TRTLLM attention backend in spec-dec mode.
        """


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
    duplicate_data: bool = True

    @staticmethod
    def from_config(config) -> "RopeParams":
        rope_params = RopeParams()

        # get rotary parameters.
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        head_dim = getattr(config, 'head_dim', None)
        if not isinstance(head_dim, int):
            head_dim = hidden_size // num_attention_heads
        rope_scaling = getattr(config, 'rope_scaling', None)
        rope_params.max_positions = config.max_position_embeddings
        rope_params.theta = getattr(config, 'rope_theta', 10000.0)
        rope_percentage = (getattr(config, 'rotary_pct', None)
                           or getattr(config, 'partial_rotary_factor', None)
                           or 1.0)
        # rotary embedding dim.
        rope_params.dim = (getattr(config, 'rotary_dim', None)
                           or getattr(config, 'rotary_emb_base', None)
                           or getattr(config, 'qk_rope_head_dim', None)
                           or int(head_dim * rope_percentage))
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
        # Workaround for DeepSeek V3 Lite since its rope_scaling is null in config.json.
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
                })
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
        return rope_inv_freq, rope_cos_sin


@dataclass(kw_only=True, frozen=True)
class PositionalEmbeddingParams:
    type: PositionEmbeddingType
    embedder: Optional[PositionalEmbedder] = None

    # RoPE params
    rope: Optional[RopeParams] = None
    is_neox: bool = True

    # mRoPE params (currently, Qwen2/2.5-VL uses it)
    mrope_section: Optional[List[int]] = None

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
        skip_create_weights_in_init: bool = False,
        sparse_attention_config: Optional["SparseAttentionConfig"] = None,
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
            sparse_attention_config (SparseAttentionConfig): Optional sparse attention configuration. If None, no sparse attention is applied.
        """
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or self.num_heads
        self.quant_config = quant_config
        self.sparse_attention_config = sparse_attention_config

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
                *,
                attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
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
            attention_mask (AttentionMask): Attention mask. See definition of `AttentionMask` for accepted types. Defaults to predefined causal mask.
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
    def support_nvfp4_output(cls) -> bool:
        return False


@dataclass(kw_only=True, unsafe_hash=True)
class MLAParams:
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    qk_nope_head_dim: int = 0
    v_head_dim: int = 0
    predicted_tokens_per_seq: int = 1
    chunked_prefill_buffer_batch_size: int = 1
    hidden_size: int = 0
