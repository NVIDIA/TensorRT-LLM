import copy
import enum
from dataclasses import dataclass, field
from functools import lru_cache
from typing import (Generic, List, Optional, Protocol, Tuple, Type, TypeVar,
                    Union)

import torch
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from typing_extensions import Self

from tensorrt_llm.functional import (PositionEmbeddingType, RopeEmbeddingUtils,
                                     RotaryScalingType)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..metadata import KVCacheParams
from ..pyexecutor.resource_manager import KVCacheManager


@dataclass
class AttentionRuntimeFeatures:
    chunked_prefill: bool = False
    cache_reuse: bool = False


@dataclass(kw_only=True)
class AttentionMetadata:
    """
    Metadata for the attention module.
    """

    # The max number of requests in a single batch.
    max_num_requests: int
    # The max number of tokens in all requests in a single batch.
    max_num_tokens: int
    # The KV cache manager.
    kv_cache_manager: KVCacheManager
    mapping: Optional[Mapping] = None

    # Whether CUDA graph is enabled.
    is_cuda_graph: bool = field(default=False, repr=False)

    # The length of each sequence in the batch for query.
    # The shape is (batch_size), and located on CPU memory.
    # For sub metadata of cross attention, it's automatically
    # initialized to seq_lens of parent metadata.
    seq_lens: Optional[torch.Tensor]  # Implemented using property
    # The position of each token in each sequence.
    # May be None if positional embedding is applied outside the backend.
    position_ids: Optional[torch.Tensor] = None

    # The number of context-phase sequences in the batch.
    num_contexts: int = 0
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

    all_rank_num_tokens: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if self.is_cross:
            assert self.cross is None or self.cross is self, "Cross attention metadata should not have sub metadata"
            self.cross = self
            return

        assert self.cross is None or type(self) is type(
            self.cross
        ), "Top level and cross attention sub metadata type mismatched"

    @property
    def seq_lens(self) -> Optional[torch.Tensor]:
        return self._seq_lens

    @seq_lens.setter
    def seq_lens(self, value: Optional[torch.Tensor]):
        # If value not explicitly given, dataclass tries to initialize using class attribute
        value = value if value is not AttentionMetadata.seq_lens else None
        self._seq_lens = value
        # The model executor sets seq_lens to None initially.
        if self._seq_lens is not None:
            self._seq_lens = self._seq_lens.pin_memory()
            self._seq_lens_cuda = self._seq_lens.cuda(non_blocking=True)
        if self.has_cross_sub_metadata:
            self.cross._seq_lens = self._seq_lens
            self.cross._seq_lens_cuda = self._seq_lens_cuda

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
    def num_generations(self) -> int:
        """
        The number of generation-phase sequences in the batch.
        """
        return self.num_seqs - self.num_contexts

    @property
    def num_seqs(self) -> int:
        """
        The number of sequences in the batch.
        """
        return self.seq_lens.shape[0]

    @property
    def num_ctx_tokens(self) -> int:
        """
        Number of tokens in query sequences in the context phase.
        """
        return int(self.context_lens.sum())

    @property
    def num_tokens(self) -> int:
        """
        Number of key and value tokens in the batch (to be appended into kv cache)
        """
        return int(self.seq_lens_kv.sum())

    @property
    def is_cross(self) -> bool:
        """
        Is this metadata for cross attention.
        """
        return self.seq_lens is not self.seq_lens_kv

    @property
    def has_cross_sub_metadata(self) -> bool:
        return self.cross is not None and self.cross is not self

    def prepare(self):
        """
        Hook to be called before the forward step of the model.
        """

    def create_cuda_graph_metadata(self,
                                   max_batch_size: int,
                                   sub_cross_metadata: bool = False) -> Self:
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
        if self.has_cross_sub_metadata:
            cuda_graph_metadata.cross = cuda_graph_metadata.cross.create_cuda_graph_metadata(
                max_batch_size, True)
        if not sub_cross_metadata:
            cuda_graph_metadata.seq_lens = torch.ones((max_batch_size, ),
                                                      dtype=torch.int)
        if self.is_cross:
            cuda_graph_metadata.seq_lens_kv = torch.zeros((max_batch_size, ),
                                                          dtype=torch.int)
        cuda_graph_metadata.num_contexts = 0
        cuda_graph_metadata.max_num_requests = max_batch_size
        cuda_graph_metadata.max_num_tokens = max_batch_size
        cuda_graph_metadata.__post_init__()
        return cuda_graph_metadata


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

    @staticmethod
    def from_config(config) -> "RopeParams":
        rope_params = RopeParams()

        # get rotary parameters.
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        head_dim = getattr(config, 'head_dim',
                           hidden_size // num_attention_heads)
        rope_scaling = getattr(config, 'rope_scaling', None)
        rope_params.max_positions = config.max_position_embeddings
        rope_params.theta = getattr(config, 'rope_theta', 10000.0)
        rope_percentage = (getattr(config, 'rotary_pct', None)
                           or getattr(config, 'partial_rotary_factor', None)
                           or 1.0)
        # rotary embedding dim.
        rope_params.dim = (getattr(config, 'rotary_dim', None)
                           or getattr(config, 'rotary_emb_base', None)
                           or int(head_dim * rope_percentage))
        # rotary scaling.
        rope_params.scale_type = RotaryScalingType.none
        rope_params.scale = 1.0
        if rope_scaling is not None:
            rotary_scaling_type = rope_scaling.get(
                "type", None) or rope_scaling.get("rope_type")
            rope_params.scale_type = RotaryScalingType.from_string(
                rotary_scaling_type)
            rope_params.scale = rope_scaling.get("factor", 1.0)
            rope_params.low_freq_factor = rope_scaling.get(
                "low_freq_factor", 1.0)
            rope_params.high_freq_factor = rope_scaling.get(
                "high_freq_factor", 4.0)
            rope_params.original_max_positions = rope_scaling.get(
                "original_max_position_embeddings", 1024)
            rope_params.beta_fast = rope_scaling.get("beta_fast", 32)
            rope_params.beta_slow = rope_scaling.get("beta_slow", 1)
            rope_params.mscale = rope_scaling.get("mscale", 1.0)
            rope_params.mscale_all_dim = rope_scaling.get("mscale_all_dim", 0.0)

        return rope_params

    @lru_cache(maxsize=1)
    def create_rope_const_params(self):
        if self.dim == 0:
            return None, None
        assert self.scale_type != RotaryScalingType.longrope, "Long RoPE is not yet supported."
        rope_inv_freq, rope_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
            self.max_positions,
            self.dim,
            self.theta,
            self.scale,
            self.scale_type,
            rope_scaling_config={
                "factor": self.scale,
                "low_freq_factor": self.low_freq_factor,
                "high_freq_factor": self.high_freq_factor,
                "original_max_position_embeddings": self.original_max_positions,
            })
        rope_inv_freq = torch.torch.tensor(
            rope_inv_freq,
            dtype=torch.float32,
            device='cuda',
        )
        rope_cos_sin = torch.torch.tensor(
            rope_cos_sin,
            dtype=torch.float32,
            device='cuda',
        )
        return rope_inv_freq, rope_cos_sin


@dataclass(kw_only=True, frozen=True)
class PositionalEmbeddingParams:
    type: PositionEmbeddingType
    embedder: Optional[PositionalEmbedder] = None

    # RoPE params
    rope: Optional[RopeParams] = None

    def __post_init__(self) -> None:
        if self.type.is_deferred():
            assert self.embedder is not None, f"{self.type} requires a not-none external embedder"
        else:
            assert self.embedder is None, f"Embedder must be None for {self.type}"

        if self.type.is_rope():
            assert self.rope is not None, f"{self.type} requires a not-none rope"


TMetadata = TypeVar("TMetadata", bound=AttentionMetadata)


class PredefinedAttentionMask(str, enum.Enum):
    """
    Predefined attention mask types

    Attributes:
        CAUSAL: Use causal mask.
        FULL: do not use any mask
    """
    CAUSAL = "causal"
    FULL = "full"


# May extend to custom attention mask type
AttentionMask = Union[PredefinedAttentionMask]


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

    @torch.library.custom_op("trtllm::attn_dummy_fwd", mutates_args=())
    @staticmethod
    def dummy_forward(q: torch.Tensor, k: torch.Tensor,
                      v: torch.Tensor) -> torch.Tensor:
        """
        Dummy attention forward function to estimate memory usage.
        Args:
            q (torch.Tensor): Query tensor with shape (1, num_q_tokens, num_heads, head_dim),.
            k (torch.Tensor): Key tensor with shape (1, num_new_kv_tokens, num_kv_heads, head_dim)
            v (torch.Tensor): Value tensor with shape (1, num_new_kv_tokens, num_kv_heads, head_dim)
        Returns:
            torch.Tensor with shape (num_q_tokens, num_heads * head_dim)
        """
        head_dim = q.shape[3]
        assert q.dim() == 4 and q.size()[0] == 1
        assert k.dim() == 4 and k.size()[0] == 1 and k.size()[3] == head_dim
        assert v.dim() == 4 and v.size()[0] == 1 and v.size()[3] == head_dim
        # This is only for memory estimation for now.
        # NOTE: this method is not accurate while it works for most scenario.
        o = _flash_attention_forward(q,
                                     k,
                                     v,
                                     attention_mask=None,
                                     query_length=q.size(1),
                                     is_causal=True)
        return o.reshape(o.size(1), -1)

    @dummy_forward.register_fake
    def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        num_q_tokens = q.size()[1]
        return torch.empty_like(q).reshape(num_q_tokens, -1)


@dataclass(kw_only=True, unsafe_hash=True)
class MLAParams:
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    qk_nope_head_dim: int = 0
    v_head_dim: int = 0
