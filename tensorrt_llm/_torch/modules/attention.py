import math
import weakref
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention_backend import (AttentionForwardArgs, AttentionMetadata,
                                 FlashInferAttentionMetadata,
                                 TrtllmAttentionMetadata)
from ..attention_backend.interface import (AttentionMask, CustomAttentionMask,
                                           PositionalEmbeddingParams,
                                           PredefinedAttentionMask)
from ..attention_backend.utils import create_attention, get_attention_backend
from ..distributed import (AllReduceParams, HelixAllToAllNative, alltoall_helix,
                           cp_allgather, reducescatter)
from ..model_config import ModelConfig
from ..peft.lora.layer import LoraLayer, LoraModuleType
from ..utils import (Fp4QuantizedTensor, get_model_extra_attrs,
                     is_nvfp4_marlin_enabled, is_torch_compiling)
from .linear import Linear, TensorParallelMode, WeightMode, WeightsLoadingConfig
from .multi_stream_utils import maybe_execute_in_parallel
from .rotary_embedding import MRotaryEmbedding, RotaryEmbedding


def extract_extra_attrs(layer_idx: str, attn_type: str):
    assert attn_type in ["mla", "attn"], "Invalid attention type"
    extra_attrs = get_model_extra_attrs()
    assert extra_attrs is not None, "Model extra attrs is not set"

    metadata_ref = extra_attrs.get("attention_metadata", None)
    assert metadata_ref is not None, "Attention metadata is not set"
    metadata = metadata_ref()
    assert isinstance(
        metadata, (FlashInferAttentionMetadata, TrtllmAttentionMetadata)
    ), "Metadata must be a subclass of FlashInferAttentionMetadata or TrtllmAttentionMetadata"

    attn_layers = extra_attrs.get(attn_type + "_layers", None)
    assert attn_layers is not None, "Attention layer is not registered"
    attn_layer_ref = attn_layers.get(layer_idx, None)
    assert attn_layer_ref is not None, f"Cannot find attention layer for layer {layer_idx}"
    attn_layer = attn_layer_ref()

    return metadata, attn_layer


def create_attn_outputs_impl(q: torch.Tensor, attention_mask: str,
                             layer_idx: str) -> List[torch.Tensor]:
    metadata, attn_layer = extract_extra_attrs(layer_idx, "attn")
    assert isinstance(
        attn_layer, Attention
    ), "Attention layer must be a subclass of Attention or an instance of Attention"
    return attn_layer.create_output(q, metadata, attention_mask)


@torch.library.custom_op("trtllm::create_attn_outputs", mutates_args=())
def create_attn_outputs(q: torch.Tensor, attention_mask: str,
                        layer_idx: str) -> List[torch.Tensor]:
    return create_attn_outputs_impl(q, attention_mask, layer_idx)


@create_attn_outputs.register_fake
def _(q, attention_mask, layer_idx):
    return create_attn_outputs_impl(q, attention_mask, layer_idx)


@torch.library.custom_op("trtllm::attn_custom_op_inplace",
                         mutates_args=("output", "output_sf"))
def attn_custom_op_inplace(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    attention_mask: str,
    mrope_rotary_cos_sin: Optional[torch.Tensor],
    mrope_position_deltas: Optional[torch.Tensor],
    attention_window_size: Optional[int],
    attention_mask_data: Optional[torch.Tensor],
    attention_sinks: Optional[torch.Tensor],
    relative_attention_bias: Optional[torch.Tensor],
    relative_attention_max_distance: int,
    layer_idx: str,
    output: torch.Tensor,
    output_sf: Optional[torch.Tensor],
) -> None:
    metadata, attn_layer = extract_extra_attrs(layer_idx, "attn")
    assert isinstance(
        attn_layer, Attention
    ), "Attention layer must be a subclass of Attention or an instance of Attention"
    mask = PredefinedAttentionMask(
        attention_mask
    ) if attention_mask != CustomAttentionMask.CUSTOM else CustomAttentionMask(
        attention_mask)
    # NVFP4 output cannot be supported by torch compile for TRTLLM backend.
    attn_layer._attn_impl(
        q,
        k,
        v,
        metadata,
        mask,
        mrope_rotary_cos_sin,
        mrope_position_deltas,
        attention_window_size,
        attention_mask_data,
        output=output,
        output_sf=output_sf,
        attention_sinks=attention_sinks,
        relative_attention_bias=relative_attention_bias,
        relative_attention_max_distance=relative_attention_max_distance,
    )


def _helix_zero_kv_mask(
        attn_metadata: AttentionMetadata,
        num_tokens: int,
        *,
        seq_start: int = 0,
        num_seqs: Optional[int] = None) -> Optional[torch.Tensor]:
    """Return a per-token bool mask marking tokens with zero local KV on this CP rank.

    These tokens belong to sequences for which this rank owns no KV blocks. Since
    kv_lens_cuda is per-sequence, it is expanded to per-token using the
    seq_lens_cuda query lengths, which stays correct when a sequence spans
    multiple tokens (e.g. speculative decoding). The buffers are static and
    output_size is passed to repeat_interleave, so the mask is CUDA-graph safe.
    Returns None when a length buffer is unavailable.

    Args:
        attn_metadata: Attention metadata holding the length buffers.
        num_tokens: Number of tokens covered by partial_o; used as the static
            output_size of the per-token expansion.
        seq_start: Index of the first sequence covered by partial_o. Defaults to
            0 for generation-only callers; MLA passes num_contexts to select the
            generation slice.
        num_seqs: Number of sequences covered by partial_o. Defaults to all
            sequences after seq_start.
    """
    kv_lens = getattr(attn_metadata, "kv_lens_cuda", None)
    seq_lens = getattr(attn_metadata, "seq_lens_cuda", None)
    if kv_lens is None or seq_lens is None:
        return None
    if num_seqs is None:
        num_seqs = attn_metadata.num_seqs - seq_start
    seq_slice = slice(seq_start, seq_start + num_seqs)
    per_seq_mask = kv_lens[seq_slice] == 0
    return torch.repeat_interleave(per_seq_mask,
                                   seq_lens[seq_slice],
                                   output_size=num_tokens)


def _helix_sanitize_empty_kv(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
    zero_kv_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Force zero-local-KV rows to a no-op contribution for the Helix combine.

    A CP rank that owns no KV blocks for a token attends to zero keys, so the
    attention kernel normalizes by a zero softmax sum and yields NaN. The combine
    weights each rank by sum * exp(max - global_max), so such rows must contribute
    softmax_stats of (-inf, 0) and a zeroed partial_o to act as a no-op. The rows
    are selected by zero_kv_mask, which is robust regardless of what the kernel
    wrote. Passing None disables sanitization.

    Args:
        partial_o: Partial attention output, shape [num_tokens, ...].
        softmax_stats: Per (token, head) (max, sum), shape [num_tokens, num_heads, 2].
        zero_kv_mask: Bool tensor of shape [num_tokens], True where this rank has
            zero local KV.
    """
    if zero_kv_mask is None:
        return partial_o, softmax_stats
    num_tokens = partial_o.shape[0]
    mask = zero_kv_mask[:num_tokens].view(-1, 1)
    # masked_fill overwrites masked rows regardless of their current (possibly NaN)
    # value and is CUDA-graph safe due to static shapes.
    partial_o = partial_o.masked_fill(mask, 0.0)
    sm_max = softmax_stats[..., 0].masked_fill(mask, float("-inf"))
    sm_sum = softmax_stats[..., 1].masked_fill(mask, 0.0)
    softmax_stats = torch.stack([sm_max, sm_sum], dim=-1)
    return partial_o, softmax_stats


def _helix_post_process(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
    mapping: Mapping,
    num_heads_tp_cp: int,
    value_dim: int,
    aux_stream: Optional[torch.cuda.Stream] = None,
    ln_events: Optional[list] = None,
    zero_kv_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Helix CP post-processing: all-to-all exchange and combine partial
    attention outputs across CP ranks.

    This is shared by both MHA (Attention) and MLA modules.  The only
    dimension that differs between the two callers is *value_dim*
    (``head_dim`` for MHA, ``kv_lora_rank`` for MLA).

    zero_kv_mask marks tokens for which this CP rank owns no KV blocks; those rows
    are forced to a no-op contribution before the exchange (see
    _helix_sanitize_empty_kv).

    When *aux_stream* and *ln_events* are provided the two
    ``.contiguous()`` calls in the FIFO-v1 path are overlapped on
    separate CUDA streams for better performance.
    """
    partial_o, softmax_stats = _helix_sanitize_empty_kv(partial_o,
                                                        softmax_stats,
                                                        zero_kv_mask)
    if mapping.cp_config.get("use_nccl_for_alltoall", True):
        # NCCL-based implementation using alltoall_helix.
        chunks = []
        for t in [partial_o, softmax_stats]:
            t = t.transpose(1, 0).contiguous()
            chunks.extend(torch.split(t, t.shape[0] // mapping.cp_size))
        gathered = alltoall_helix(chunks, mapping.cp_group)
        gathered = [t.transpose(1, 2).contiguous() for t in gathered]
        return torch.ops.trtllm.helix_post_process(gathered[0], gathered[1],
                                                   1.0)
    else:
        # FIFO-based implementation using MNNVL workspace.
        helix = HelixAllToAllNative.get(mapping)
        num_tokens = partial_o.shape[0]
        cp_size = mapping.cp_size
        fifo_version = mapping.cp_config.get("fifo_version", 2)

        if fifo_version == 1:

            def reshape_o():
                return partial_o.view(num_tokens, cp_size, num_heads_tp_cp,
                                      value_dim).transpose(1, 2).contiguous()

            def reshape_s():
                return softmax_stats.view(num_tokens, cp_size, num_heads_tp_cp,
                                          2).transpose(1, 2).contiguous()

            if aux_stream is not None and ln_events is not None:
                partial_o, softmax_stats = maybe_execute_in_parallel(
                    reshape_o,
                    reshape_s,
                    ln_events[0],
                    ln_events[1],
                    aux_stream,
                )
            else:
                partial_o = reshape_o()
                softmax_stats = reshape_s()

            partial_o_out, softmax_stats_out = helix.alltoall_native(
                partial_o, softmax_stats)
            return torch.ops.trtllm.helix_post_process_native(
                partial_o_out, softmax_stats_out, 1.0, 2)
        else:
            partial_o = partial_o.view(num_tokens, cp_size,
                                       num_heads_tp_cp * value_dim)
            softmax_stats = softmax_stats.view(num_tokens, cp_size,
                                               num_heads_tp_cp * 2)
            partial_o_out, softmax_stats_out = helix.alltoall_native(
                partial_o, softmax_stats)
            gathered_o = partial_o_out.view(num_tokens, cp_size,
                                            num_heads_tp_cp, value_dim)
            gathered_stats = softmax_stats_out.view(num_tokens, cp_size,
                                                    num_heads_tp_cp, 2)
            return torch.ops.trtllm.helix_post_process_native(
                gathered_o, gathered_stats, 1.0, 1)


def _helix_cp_pad(tensor: torch.Tensor, num_tokens: int,
                  cp_size: int) -> tuple[torch.Tensor, int]:
    """Pad tensor along dim-0 so its length is divisible by cp_size."""
    chunk_size = math.ceil(num_tokens / cp_size)
    padded_size = chunk_size * cp_size
    if num_tokens < padded_size:
        tensor = torch.nn.functional.pad(tensor,
                                         (0, 0, 0, padded_size - num_tokens),
                                         mode="constant",
                                         value=0)
    return tensor, chunk_size


def _helix_cp_allgather_input(hidden_states: torch.Tensor,
                              attn_metadata: AttentionMetadata,
                              mapping: Mapping, layer_idx: int) -> torch.Tensor:
    """AllGather hidden states from CP group for layers after the first.

    The first layer already has the full input from the embedding.
    Subsequent layers need to undo the previous layer's reduce-scatter.
    """
    if (mapping.has_cp_helix() and mapping.enable_attention_dp
            and layer_idx > 0):
        hidden_states = cp_allgather(hidden_states, mapping, dim=0)
        hidden_states = hidden_states[:attn_metadata.num_tokens]
    return hidden_states


def _helix_cp_output_projection(
    o_proj: Linear,
    attn_output: torch.Tensor,
    attn_metadata: AttentionMetadata,
    all_reduce_params: Optional[AllReduceParams],
    mapping: Mapping,
    mapping_o: Mapping,
    layer_idx: int,
    lora_params: Optional[dict] = None,
) -> torch.Tensor:
    """Apply output projection with reduce-scatter when Helix CP+DP is active.

    Reduce-scatter sums partial sums across the CP group and scatters the
    result so each CP rank processes a distinct token chunk through the MLP.
    Falls back to the standard AllReduce path otherwise.
    """
    if mapping.has_cp_helix() and mapping.enable_attention_dp:
        attn_output = o_proj(
            attn_output,
            all_reduce_params=AllReduceParams(enable_allreduce=False),
            lora_params=lora_params,
            layer_idx=layer_idx)

        attn_output, _ = _helix_cp_pad(attn_output, attn_metadata.num_tokens,
                                       mapping.cp_size)
        attn_output = reducescatter(attn_output, mapping_o, dim=0)
    else:
        attn_output = o_proj(attn_output,
                             all_reduce_params=all_reduce_params,
                             lora_params=lora_params,
                             layer_idx=layer_idx)

    return attn_output


def maybe_slice_for_helix_cp(tensor: torch.Tensor,
                             attn_metadata: AttentionMetadata,
                             mapping_with_cp: Optional[Mapping],
                             layer_idx: int) -> torch.Tensor:
    """Slice a tensor to this CP rank's chunk after reduce-scatter.

    For the first decoder layer, the residual comes from the embedding and
    has not been through a prior reduce-scatter.  This function slices it
    so it aligns with the reduce-scattered attention output.  For
    subsequent layers the residual already has the correct size, so this
    is a no-op.

    Call this in the decoder layer on the residual *after* the attention
    forward, so that Attention/MLA forward signatures stay unchanged.
    """
    if (mapping_with_cp is not None and mapping_with_cp.has_cp_helix()
            and mapping_with_cp.enable_attention_dp and layer_idx == 0):
        tensor, chunk_size = _helix_cp_pad(tensor, attn_metadata.num_tokens,
                                           mapping_with_cp.cp_size)
        start = mapping_with_cp.cp_rank * chunk_size
        tensor = tensor[start:start + chunk_size]
    return tensor


def maybe_allgather_for_helix_cp(
        hidden_states: torch.Tensor, attn_metadata: AttentionMetadata,
        mapping_with_cp: Optional[Mapping]) -> torch.Tensor:
    """Restore full token count after the last layer's reduce-scatter.

    With Helix CP + Attention DP, each decoder layer's reduce-scatter
    leaves each CP rank with only its chunk of tokens.  This function
    performs an AllGather across the CP group so that the LM head (and
    final norm) see every token.

    Should be called at the end of the model's ``forward()`` method,
    after the decoder layer loop.
    """
    if (mapping_with_cp is not None and mapping_with_cp.has_cp_helix()
            and mapping_with_cp.enable_attention_dp):
        hidden_states = cp_allgather(hidden_states, mapping_with_cp, dim=0)
        hidden_states = hidden_states[:attn_metadata.num_tokens]
    return hidden_states


class Attention(nn.Module):

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int,
        bias: bool,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        rope_fusion: Optional[bool] = None,
        layer_idx: Optional[int] = None,
        dtype: torch.dtype = None,
        dense_bias: Optional[bool] = None,
        config: Optional[ModelConfig] = None,
        q_scaling: float = 1.0,
        attention_chunk_size: Optional[int] = None,
        disable_deep_gemm: bool = False,
        attn_output_gate: Optional[bool] = None,
        use_custom_cublas_mm: bool = False,
        reduce_output: bool = True,
        mapping_with_cp: Optional[Mapping] = None,
        head_dim: Optional[int] = None,
    ):
        """
        Initialize the Attention module.

        Args:
            hidden_size (int): The size of the hidden dimension.
            num_attention_heads (int): The number of attention heads.
            num_key_value_heads (int): The number of key value heads.
            max_position_embeddings (int): The maximum position embeddings.
            bias (bool): Whether to use bias in the linear layers.
            pos_embd_params (Optional[PositionalEmbeddingParams]): The positional embedding parameters.
            rope_fusion (Optional[bool]): Whether to fuse RoPE into the attention OP and skip applying unfused RoPE. If None, whether to fuse is decided by the capability of the attention backend.
            layer_idx (Optional[int]): The layer index.
            dtype (torch.dtype): The data type.
            dense_bias (Optional[bool]): Whether to use bias in the output projection layer.
            config (Optional[ModelConfig]): The model configuration.
            q_scaling (float): The scaling factor for the qk_scale. The definition is $O = softmax(QK^T * qk_scale) * V, qk_scale = 1 / (sqrt(head_dim) * q_scaling)$. The default value is 1.0.
            attention_chunk_size (Optional[int]): See [Chunked Attention] below.
            disable_deep_gemm (bool): Whether to disable the use of DeepGEMM in Linear layers (currently only matters on SM100 + FP8).
            attn_output_gate (Optional[bool]): Determines whether to use an output gate in the attention Op. If False, the decision is automatically handled by the attention backend based on its capabilities.
            mapping_with_cp (Optional[Mapping]): Override mapping with CP configuration.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx)

        self.register_to_config = False
        # We only register TRTLLM attention layers to config.
        if config is not None:
            if "attn_layers" not in config.extra_attrs:
                config.extra_attrs["attn_layers"] = {}
            suffix = 0
            # Makes sure there is no duplicate attention layer identifier.
            while self.layer_idx_str in config.extra_attrs["attn_layers"]:
                self.layer_idx_str = str(layer_idx) + f"_{suffix}"
                suffix += 1
            config.extra_attrs["attn_layers"][self.layer_idx_str] = weakref.ref(
                self)
            self.register_to_config = True

        config = config or ModelConfig()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        # Prefer an explicit head_dim from the caller; fall back to the
        # pretrained config, then to hidden_size // num_heads. The explicit
        # override is required for sub-modules (e.g. VLM vision encoders)
        # whose head_dim does not match the top-level config's head_dim.
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            self.head_dim = getattr(config.pretrained_config, 'head_dim', None)
            if not isinstance(self.head_dim, int):
                self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.pos_embd_params = pos_embd_params
        self.dense_bias = dense_bias
        self.q_scaling = q_scaling
        self.attn_output_gate = attn_output_gate

        if self.attn_output_gate:
            logger.info_once("using attn output gate!", key="attn_output_gate")

        # [Chunked Attention]
        # Chunked attention is applied to context requests only. Chunked attention will be
        # applied when this field is specified and mMaskType == CAUSAL.
        #
        # In chunked attention, we break context requests into chunks of a specified size. Tokens can only
        # attend to tokens in the same chunk. So, for example, if the chunk size is 3, we might have a mask
        # that looks like this:
        #
        # 1 0 0 0 0 0
        # 1 1 0 0 0 0
        # 1 1 1 0 0 0
        # 0 0 0 1 0 0
        # 0 0 0 1 1 0
        # 0 0 0 1 1 1
        self.attention_chunk_size = attention_chunk_size

        if dense_bias is None:
            self.dense_bias = bias

        # tensor parallel
        if mapping_with_cp is not None:
            logger.warning_once(
                "[Attention::__init__] Overriding mapping with CP detected.",
                key="attention_init_mapping_with_cp")
            self.mapping = mapping_with_cp
        else:
            self.mapping = config.mapping

        tp_size = self.mapping.tp_size
        pp_size = self.mapping.pp_size
        cp_size = self.mapping.cp_size
        dp_size = 1
        if self.mapping.enable_attention_dp:
            dp_size = tp_size
            tp_size = 1

        if self.mapping.cp_size > 1:
            assert self.mapping.has_cp_helix(
            ), f"CP type must be HELIX for Attention, but got {self.mapping.cp_config['cp_type']}."

        mapping = Mapping(
            world_size=dp_size * tp_size * pp_size * cp_size,
            tp_size=tp_size,
            pp_size=pp_size * dp_size,
            cp_size=cp_size,
            cp_config=self.mapping.cp_config,
            rank=self.mapping.rank,
            gpus_per_node=self.mapping.gpus_per_node,
            enable_attention_dp=self.mapping.enable_attention_dp,
        )
        self.tp_size = tp_size
        self.cp_size = cp_size
        self.tp_rank = mapping.tp_rank
        assert self.num_heads % (tp_size * cp_size) == 0
        self.num_heads = self.num_heads // tp_size
        self.num_heads_tp_cp = self.num_heads // cp_size
        self.num_key_value_heads = (self.num_key_value_heads + tp_size -
                                    1) // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        self.use_cute_dsl_blockscaling_mm = config.use_cute_dsl_blockscaling_mm
        self.use_cute_dsl_blockscaling_bmm = config.use_cute_dsl_blockscaling_bmm
        self.use_cute_dsl_bf16_bmm = config.use_cute_dsl_bf16_bmm
        self.use_cute_dsl_bf16_gemm = config.use_cute_dsl_bf16_gemm

        qkv_shard_indices_mapping = {
            "q": (0, self.q_size * (2 if self.attn_output_gate else 1)),
            "k":
            (self.q_size * (2 if self.attn_output_gate else 1), self.kv_size),
            "v":
            (self.q_size * (2 if self.attn_output_gate else 1) + self.kv_size,
             self.kv_size),
        }

        self.qkv_proj = Linear(
            self.hidden_size,
            tp_size * self.q_size * (2 if self.attn_output_gate else 1) +
            2 * tp_size * self.kv_size,
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_QKV_LINEAR),
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization,
            disable_deep_gemm=disable_deep_gemm,
            use_custom_cublas_mm=use_custom_cublas_mm,
            fused_weight_shard_indices_mapping=qkv_shard_indices_mapping,
            use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm)

        self.o_lora = LoraLayer([LoraModuleType.ATTENTION_DENSE],
                                [self.hidden_size])

        # For Helix CP, combine TP and CP for the output projection so each
        # rank's o_proj input is num_heads_tp_cp * head_dim.
        mapping_o = Mapping(
            world_size=dp_size * tp_size * pp_size * cp_size,
            tp_size=tp_size * cp_size,
            pp_size=pp_size * dp_size,
            cp_size=1,
            rank=self.mapping.rank,
            gpus_per_node=self.mapping.gpus_per_node,
            enable_attention_dp=self.mapping.enable_attention_dp,
        )
        self.mapping_o = mapping_o

        self.o_proj = Linear(
            tp_size * self.q_size,
            self.hidden_size,
            bias=self.dense_bias,
            dtype=dtype,
            mapping=mapping_o,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            lora=self.o_lora,
            reduce_output=reduce_output,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization,
            disable_deep_gemm=disable_deep_gemm,
            use_custom_cublas_mm=use_custom_cublas_mm,
            use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
            use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm)

        self.quant_config = config.get_quant_config()
        self.attn_backend = config.attn_backend

        sparse_attn_cfg = config.sparse_attention_config
        sparse_params = (sparse_attn_cfg.to_sparse_params(
            pretrained_config=config.pretrained_config,
            layer_idx=self.layer_idx) if sparse_attn_cfg is not None else None)

        attn_cls = get_attention_backend(self.attn_backend,
                                         sparse_params=sparse_params)

        self.is_marlin_enabled: bool = is_nvfp4_marlin_enabled()

        # These two modules are mutually exclusive - either splitted_qkv_lora or fused_qkv_lora will be used,
        # but never both at the same time. splitted_qkv_lora handles Q,K,V separately while fused_qkv_lora
        # handles them as a single fused operation.
        self.splitted_qkv_lora = LoraLayer([
            LoraModuleType.ATTENTION_Q, LoraModuleType.ATTENTION_K,
            LoraModuleType.ATTENTION_V
        ], [self.q_size, self.kv_size, self.kv_size])
        self.fused_qkv_lora = LoraLayer([LoraModuleType.ATTENTION_QKV],
                                        [self.q_size + 2 * self.kv_size])

        # Whether to fuse RoPE into the attention OP.
        # If true, RoPE will be applied in self.attn.forward.
        # If false, RoPE will be applied in self.apply_rope.
        self.rope_fusion = rope_fusion

        if config.sparse_attention_config is not None:
            # Log sparse attention configuration once
            algo = config.sparse_attention_config.algorithm
            cfg_dump = config.sparse_attention_config.model_dump(
                exclude_none=True)
            logger.info_once(f"Using sparse attention: {algo} {cfg_dump}",
                             key="sparse_attention_config")

            if config.sparse_attention_config.algorithm == "rocket":
                logger.warning_once("disable rope_fusion for RocketKV.",
                                    key="disable_rope_fusion_for_rocketkv")
                self.rope_fusion = False

        if self.rope_fusion and not attn_cls.support_fused_rope():
            logger.warning_once(
                "rope_fusion is true but the attention backend does not support it. Will disable rope_fusion.",
                key="disable_rope_fusion_for_non_supported_backend")
            self.rope_fusion = False
        # If rope_fusion is not specified, enable if the attention backend supports it.
        if self.rope_fusion is None:
            self.rope_fusion = attn_cls.support_fused_rope()

        self.rotary_emb = None
        if not self.rope_fusion and self.pos_embd_params is not None:
            if self.pos_embd_params.type.is_mrope():
                self.rotary_emb = MRotaryEmbedding(
                    self.pos_embd_params.rope,
                    head_dim=self.head_dim,
                    is_neox=self.pos_embd_params.is_neox,
                    mrope_section=self.pos_embd_params.mrope_section,
                    mrope_interleaved=self.pos_embd_params.mrope_interleaved)
            else:
                self.rotary_emb = RotaryEmbedding(
                    self.pos_embd_params.rope,
                    head_dim=self.head_dim,
                    is_neox=self.pos_embd_params.is_neox,
                )

        self.attn = create_attention(
            self.attn_backend,
            self.layer_idx,
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
            pos_embd_params=(self.pos_embd_params if self.rope_fusion or
                             (self.pos_embd_params is not None
                              and not self.pos_embd_params.type.is_rope()) else
                             None),
            quant_config=self.quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            q_scaling=self.q_scaling,
            attention_chunk_size=self.attention_chunk_size,
            sparse_params=sparse_params,
        )

        self.support_fused_qkv = self.attn.support_fused_qkv()

        if not config.skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self):
        # self.attn has no weights but has states that are related to quant_config,
        # which could be modified after __init__
        self.attn.update_quant_config(self.quant_config)

        self.o_proj.create_weights()
        self.has_quant_scale = (self.o_proj.has_fp8_qdq or self.o_proj.has_nvfp4
                                or self.o_proj.has_fp8_block_scales
                                or self.o_proj.has_fp8_rowwise
                                or self.o_proj.has_w4a8_nvfp4_fp8)

    def split_qkv(self, q, k=None, v=None):
        if k is None and v is None:
            q, k, v = q.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q, k, v

    def convert_qkv(self, q, k, v):
        if k is None and v is None and not self.support_fused_qkv:
            q, k, v = self.split_qkv(q)
        elif k is not None and v is not None and self.support_fused_qkv:
            qkv = torch.concat([q, k, v], dim=-1)
            q, k, v = qkv, None, None
        return q, k, v

    def _use_quantize_output(self):
        # If o_proj can't consume, then no need to quantize the output to nvfp4
        if hasattr(self.attn, 'has_nvfp4'
                   ) and self.attn.has_nvfp4 and not self.o_proj.has_nvfp4:
            return False

        # If o_proj does dynamic activation quantization, it computes its own scales
        # at runtime from a BF16 input — attention must NOT pre-quantize output
        if self.o_proj.force_dynamic_quantization:
            return False

        # If no quant is applied, no need to quantize the output
        if self.quant_config is not None and not self.quant_config.layer_quant_mode.has_any_quant(
                exclude_kv_cache=True):
            return False

        has_awq_pre_quant_scale = hasattr(
            self.o_proj,
            'pre_quant_scale') and self.o_proj.pre_quant_scale is not None

        return self.has_quant_scale and not self.attn_output_gate and not has_awq_pre_quant_scale and not self.is_marlin_enabled

    def create_output(self, q: torch.Tensor, attn_metadata: AttentionMetadata,
                      mask_type: str):
        # Attention is treated as mixed request by default.
        return self.attn.create_output(
            q,
            is_quantize_output=self._use_quantize_output(),
            metadata=attn_metadata,
            attention_mask=mask_type,
            is_gen_only=False)

    def _helix_post_process(self, partial_o: torch.Tensor,
                            softmax_stats: torch.Tensor,
                            attn_metadata: AttentionMetadata) -> torch.Tensor:
        """Helix CP post-processing: all-to-all exchange and combine partial
        attention outputs across CP ranks."""
        zero_kv_mask = _helix_zero_kv_mask(attn_metadata, partial_o.shape[0])
        return _helix_post_process(partial_o,
                                   softmax_stats,
                                   self.mapping,
                                   self.num_heads_tp_cp,
                                   self.head_dim,
                                   zero_kv_mask=zero_kv_mask)

    def _attn_impl(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        attention_mask: AttentionMask,
        mrope_rotary_cos_sin: Optional[torch.Tensor],
        mrope_position_deltas: Optional[torch.Tensor],
        attention_window_size: Optional[int],
        attention_mask_data: Optional[torch.Tensor],
        output: Optional[torch.Tensor] = None,
        output_sf: Optional[torch.Tensor] = None,
        attention_sinks: Optional[torch.Tensor] = None,
        relative_attention_bias: Optional[torch.Tensor] = None,
        relative_attention_max_distance: int = 0,
        has_lora: bool = False,
    ):
        num_tokens = attn_metadata.num_tokens

        q = q[:num_tokens, :]
        if k is not None:
            k = k[:num_tokens, :]
        if v is not None:
            v = v[:num_tokens, :]

        # Helix CP generation path: get partial outputs with softmax stats,
        # then exchange and combine across CP ranks.
        # NOTE: The helix post-process combine step works on unquantized
        # (BF16/FP16) partial outputs and softmax stats from each rank.
        # We intentionally skip passing out_scale to FMHA here
        # so it produces BF16 output. After combining, the downstream o_proj
        # linear layer handles quantization (FP8/NVFP4) in its apply() method.
        if self.mapping.has_cp_helix() and attn_metadata.num_contexts == 0:
            assert output is None, (
                "Helix produces BF16 partial outputs which may not match a pre-allocated FP8/NVFP4 buffer for torch.compile inplace output."
            )
            softmax_stats = torch.empty((num_tokens, self.num_heads, 2),
                                        device=q.device,
                                        dtype=torch.float32)
            attn_output = self.attn.forward(
                q,
                k,
                v,
                attn_metadata,
                forward_args=AttentionForwardArgs(
                    attention_mask=attention_mask,
                    mrope_rotary_cos_sin=mrope_rotary_cos_sin,
                    mrope_position_deltas=mrope_position_deltas,
                    attention_window_size=attention_window_size,
                    attention_mask_data=attention_mask_data,
                    softmax_stats_tensor=softmax_stats,
                    attention_sinks=attention_sinks,
                    relative_attention_bias=relative_attention_bias,
                    relative_attention_max_distance=
                    relative_attention_max_distance,
                ))
            if isinstance(attn_output, tuple):
                attn_output = attn_output[0]
            attn_output = self._helix_post_process(attn_output, softmax_stats,
                                                   attn_metadata)
            return attn_output, None

        # Don't set out_scale if o_proj has pre_quant_scale — this prevents
        # FP8/FP4 output and keeps attention output in BF16 for better
        # precision when applying pre_quant_scale. Also don't set out_scale
        # if LoRA is active — LoRA grouped_gemm doesn't support FP8.
        # Pass both scales; the backend selects ``out_scale_sf`` when the
        # kernel writes NVFP4 output (``forward_args.output_sf`` is
        # allocated downstream by ``create_output``) and ``out_scale``
        # otherwise. Deciding here would be premature — ``output_sf`` is
        # not populated yet at this call site.
        out_scale = None
        out_scale_sf = None
        if self._use_quantize_output() and not has_lora:
            out_scale = self.o_proj.inv_input_scale
            out_scale_sf = self.o_proj.input_scale

        kv_scale_orig_quant = None
        kv_scale_quant_orig = None
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_fp4_kv_cache(
        ):
            kv_scale_orig_quant = self.qkv_proj.inv_kv_scales
            kv_scale_quant_orig = self.qkv_proj.kv_scales

        attn_output = self.attn.forward(
            q,
            k,
            v,
            attn_metadata,
            forward_args=AttentionForwardArgs(
                out_scale=out_scale,
                out_scale_sf=out_scale_sf,
                kv_scale_orig_quant=kv_scale_orig_quant,
                kv_scale_quant_orig=kv_scale_quant_orig,
                attention_mask=attention_mask,
                mrope_rotary_cos_sin=mrope_rotary_cos_sin,
                mrope_position_deltas=mrope_position_deltas,
                attention_window_size=attention_window_size,
                attention_mask_data=attention_mask_data,
                output=output[:num_tokens, :] if output is not None else None,
                output_sf=output_sf,
                attention_sinks=attention_sinks,
                relative_attention_bias=relative_attention_bias,
                relative_attention_max_distance=relative_attention_max_distance,
            ))
        if isinstance(attn_output, tuple):
            assert len(
                attn_output
            ) == 2, "attn_output should be a tuple of (output, output_sf)"
            return attn_output[0], attn_output[1]
        return attn_output, None

    def forward_impl(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        attention_mask: AttentionMask,
        attention_window_size: Optional[int],
        attention_mask_data: Optional[torch.Tensor],
        mrope_config: Optional[dict],
        attention_sinks: Optional[torch.Tensor] = None,
        relative_attention_bias: Optional[torch.Tensor] = None,
        relative_attention_max_distance: int = 0,
        has_lora: bool = False,
    ):
        mrope_rotary_cos_sin = None
        mrope_position_deltas = None
        if mrope_config is not None:
            if "mrope_rotary_cos_sin" in mrope_config:
                mrope_rotary_cos_sin = mrope_config["mrope_rotary_cos_sin"]
            if "mrope_position_deltas" in mrope_config:
                mrope_position_deltas = mrope_config["mrope_position_deltas"]

        # Currently only TRTLLM and FLASHINFER are torch compile compatible backends.
        # Only enable custom inplace op when torch compiling.
        use_custom_inplace_op = (self.register_to_config
                                 and (self.attn_backend == "TRTLLM"
                                      or self.attn_backend == "FLASHINFER")
                                 and is_torch_compiling()
                                 and not self.is_marlin_enabled)

        if use_custom_inplace_op:
            outputs = create_attn_outputs(q, attention_mask, self.layer_idx_str)
            assert len(outputs) == 1 or len(outputs) == 2
            output = outputs[0]
            output_sf = outputs[1] if len(outputs) == 2 else None
            attn_custom_op_inplace(
                q,
                k,
                v,
                attention_mask,
                mrope_rotary_cos_sin,
                mrope_position_deltas,
                attention_window_size,
                attention_mask_data,
                attention_sinks,
                relative_attention_bias,
                relative_attention_max_distance,
                self.layer_idx_str,
                output,
                output_sf,
            )
        else:
            output, output_sf = self._attn_impl(
                q,
                k,
                v,
                attn_metadata,
                attention_mask,
                mrope_rotary_cos_sin,
                mrope_position_deltas,
                attention_window_size,
                attention_mask_data,
                attention_sinks=attention_sinks,
                relative_attention_bias=relative_attention_bias,
                relative_attention_max_distance=relative_attention_max_distance,
                has_lora=has_lora,
            )
        if output_sf is not None:
            output = Fp4QuantizedTensor(output, output_sf)

        return output

    def forward(
        self,
        position_ids: Optional[torch.IntTensor],
        hidden_states: Union[torch.Tensor, Fp4QuantizedTensor],
        attn_metadata: AttentionMetadata,
        attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
        mrope_config: Optional[dict] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        attention_window_size: Optional[int] = None,
        attention_mask_data: Optional[torch.Tensor] = None,
        attention_sinks: Optional[torch.Tensor] = None,
        relative_attention_bias: Optional[torch.Tensor] = None,
        relative_attention_max_distance: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for the Attention module.

        Args:
            position_ids (Optional[torch.IntTensor]): The position IDs.
            hidden_states (torch.Tensor): The hidden states.
            attn_metadata (AttentionMetadata): The attention metadata.
            attention_mask (AttentionMask): The attention mask type.
            mrope_config (Optional[dict]): The MROPE configuration.
            all_reduce_params (Optional[AllReduceParams]): The all reduce parameters.
            lora_params (Optional[dict]): The LoRA parameters.
            attention_window_size (Optional[int]): The attention window size.
            attention_mask_data (Optional[torch.Tensor]): The attention mask data.
        Returns:
            torch.Tensor: The output tensor.
        """
        hidden_states = _helix_cp_allgather_input(hidden_states, attn_metadata,
                                                  self.mapping, self.layer_idx)

        qkv = self.qkv_proj(hidden_states)

        if bool(lora_params):
            qkv_lora = self.splitted_qkv_lora(hidden_states, lora_params,
                                              self.layer_idx)
            if qkv_lora is not None:
                qkv = qkv + qkv_lora

            qkv_lora = self.fused_qkv_lora(hidden_states, lora_params,
                                           self.layer_idx)
            if qkv_lora is not None:
                qkv = qkv + qkv_lora

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
            orig_shape = q_gate.shape[:-1]
            # Single line: view -> chunk -> reshape both q and gate
            q, gate = [
                t.reshape(*orig_shape, -1) for t in torch.chunk(
                    q_gate.view(*orig_shape, self.num_heads, -1), 2, dim=-1)
            ]
        else:
            q, k, v = qkv, None, None

        # For dynamic tree spec decoding with Python RoPE, adjust position_ids
        # to use tree offsets (same as C++ kernel: past_seq_len + offset).
        if (not self.rope_fusion
                and getattr(attn_metadata, 'is_spec_dec_dynamic_tree', False)
                and getattr(attn_metadata, 'use_spec_decoding', False)
                and getattr(attn_metadata, 'spec_decoding_position_offsets',
                            None) is not None
                and attn_metadata.spec_decoding_position_offsets.dim() ==
                1  # 1D layout ⇒ dynamic tree
                and position_ids is not None):
            position_ids = self._adjust_position_ids_for_spec_dec(
                position_ids, attn_metadata)

        q, k, v = self.apply_rope(q, k, v, position_ids)
        q, k, v = self.convert_qkv(q, k, v)

        if attention_sinks is not None:
            assert self.attn_backend == "TRTLLM", (
                f"Attention sinks are only supported with attn_backend='TRTLLM'. "
                f"Current backend: {self.attn_backend}.")
        if relative_attention_bias is not None:
            assert self.attn_backend == "TRTLLM", "Relative attention bias is only supported for TRTLLM backend."

        attn_output = self.forward_impl(
            q,
            k,
            v,
            attn_metadata,
            attention_mask,
            attention_window_size,
            attention_mask_data,
            mrope_config=mrope_config,
            attention_sinks=attention_sinks,
            relative_attention_bias=relative_attention_bias,
            relative_attention_max_distance=relative_attention_max_distance,
            has_lora=bool(lora_params),
        )

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        attn_output = _helix_cp_output_projection(self.o_proj, attn_output,
                                                  attn_metadata,
                                                  all_reduce_params,
                                                  self.mapping, self.mapping_o,
                                                  self.layer_idx, lora_params)
        return attn_output

    def apply_rope(self, q: torch.Tensor, k: Optional[torch.Tensor],
                   v: Optional[torch.Tensor], position_ids: torch.Tensor):
        """
        Apply RoPE to the query and key.
        Depending on the implementation, q, k, v could be either fused (q, k, v = concat(q, k, v), None, None) or unfused (none of q, k, v is None).
        Before self.attn.forward, convert_qkv will be called to make sure that the format of (q, k, v) satisfies the requirement of self.attn.
        This method could be overridden in the subclass, in which extra functionalities such as q_norm/k_norm could be added.
        Args:
            q (torch.Tensor): The query tensor.
            k (Optional[torch.Tensor]): The key tensor.
            v (Optional[torch.Tensor]): The value tensor.
            position_ids (torch.Tensor): The position IDs of each token for RoPE.
        Returns:
            tuple: A tuple of (q, k, v).
        """
        # If RoPE is fused into the attention OP, do not apply RoPE here.
        if not self.rope_fusion and position_ids is not None:
            q, k, v = self.split_qkv(q, k, v)
            q, k = self.rotary_emb(position_ids, [q, k])
        return q, k, v

    def _adjust_position_ids_for_spec_dec(self, position_ids, attn_metadata):
        """Replicate C++ kernel's rotary_pos = past_seq_len + offset.

        The dynamic-tree draft loop writes spec_decoding_position_offsets
        consecutively (row stride = gen_len), so a flat reshape to
        (num_gens, gen_len) reads the right rows.
        """
        num_contexts = attn_metadata.num_contexts
        num_gens = attn_metadata.num_seqs - num_contexts
        if num_gens <= 0:
            return position_ids
        gen_len = int(attn_metadata.seq_lens[num_contexts])
        if gen_len <= 0:
            return position_ids
        base_pos = attn_metadata.kv_lens_cuda[num_contexts:num_contexts +
                                              num_gens] - gen_len
        total = num_gens * gen_len
        offsets = attn_metadata.spec_decoding_position_offsets[:total].view(
            num_gens, gen_len)
        start = attn_metadata.num_ctx_tokens
        end = start + total
        adjusted = (base_pos.unsqueeze(1) + offsets).reshape(-1)
        position_ids.view(-1)[start:end] = adjusted
        return position_ids

    def apply_qk_norm(self, q, k):
        raise NotImplementedError(
            f"QK norm is not implemented for {self.__class__.__name__}. "
            "Please override the `apply_qk_norm` method in the subclass.")
