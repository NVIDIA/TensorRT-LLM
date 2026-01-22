import logging
from typing import Optional, Tuple

import flashinfer
import pytest
import torch
import torch.nn as nn
from transformers import DeepseekV3Config
from transformers.cache_utils import Cache

from tensorrt_llm._torch.auto_deploy.models.patches.deepseek import (
    deepseek_v3_attention as patched_attention,
)

"""
References:
- https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py
- https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py#L396-L497
- https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/configs/config_671B.json
- https://github.com/flashinfer-ai/flashinfer/blob/main/tests/test_mla_decode_kernel.py
- https://sourcegraph.com/github.com/flashinfer-ai/flashinfer/-/blob/tests/test_deepseek_mla.py?L503
"""


logger = logging.getLogger(__name__)

device = "cuda:0"
# global_workspace_buffer = None
# global_trtllm_gen_fmha_workspace_buffer = None  # must be zero initialized
# WORKSPACE_BUFFER = 128 * 1024 * 1024


# def get_workspace_buffer():
#     global global_trtllm_gen_fmha_workspace_buffer
#     if global_trtllm_gen_fmha_workspace_buffer is None:
#         global_trtllm_gen_fmha_workspace_buffer = torch.zeros(
#             WORKSPACE_BUFFER, dtype=torch.int8, device=device
#         )
#     return global_trtllm_gen_fmha_workspace_buffer


class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# This is the original DeepseekV3RotaryEmbedding from Hugging Face, with a couple of patches
# for sine and cosine caching. The cache is computed once (at initialization) and reused later.
class DeepseekV3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    # Patch 1: add this method for convenient access to the sine-cosine cache
    # def get_cos_sin_cache(self):
    #     return (
    #         self.cos_cached,
    #         self.sin_cached,
    #     )

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]

        # Patch 2: Prevents the cache from being recomputed and asserts that
        # recomputation is not required.
        # assert self.max_seq_len_cached is not None
        # assert seq_len <= self.max_seq_len_cached
        # if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
        #     self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# This is the original DeepseekV3Attention from Hugging Face, with the original forward method
# and AD's patched forward method (ad_forward)
class DeepseekV3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, config.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(
                config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        assert self.config.rope_scaling is None

    def _init_rope(self):
        assert self.config.rope_scaling is None
        self.rotary_emb = DeepseekV3RotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2).contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, compressed_kv, k_pe

    # This is the Auto Deploy patched version of the forward method
    def ad_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        assert self.q_lora_rank is not None, "q_lora_rank must be set"

        # x * W_DQ (i.e. q down projection)
        q_normed_dn = self.q_a_layernorm(
            self.q_a_proj(hidden_states)
        )  # (bsz, q_len, self.q_lora_rank)

        wq_b = self.q_b_proj.weight  # (self.num_heads * self.q_head_dim, self.q_lora_rank)

        # c_KV = x * W_DKV (i.e. kv down projection)
        compressed_kv_kpe = self.kv_a_proj_with_mqa(hidden_states)  # [bsz, q_len, 512 + 64]
        # Separates the compressed kv into the low-rank part and the positional encoding part
        compressed_kv, k_pe = torch.split(
            compressed_kv_kpe, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )  # compressed_kv ~ [bsz, q_len, 512 ], k_pe ~ [bsz, q_len, 64]
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb.cos_cached, self.rotary_emb.sin_cached

        wkv_b = self.kv_b_proj.weight  # [128 * 256, 512]
        wo_proj = self.o_proj.weight

        # Unpack optional kernel-specific kwargs
        ad_operator = kwargs.get("ad_operator", None)
        seq_len = kwargs.get("seq_len", None)
        input_pos = kwargs.get("input_pos", None)
        cache_loc = kwargs.get("cache_loc", None)
        seq_start = kwargs.get("seq_start", None)
        # position_ids = kwargs.get("position_ids", None)
        ckv_cache = kwargs.get("ckv_cache", None)
        k_pe_cache = kwargs.get("k_pe_cache", None)

        using_cache = ckv_cache is not None and k_pe_cache is not None

        if not using_cache:
            # Perform forward without a KV-cache. These operators are not used by the
            # optimized graph.
            args = (
                q_normed_dn,
                compressed_kv,
                k_pe,
                sin,
                cos,
                wkv_b,
                wq_b,
                None,  # w_uq_ukv placeholder for weight absorption
                wo_proj,
                None,  # w_uv_o placeholder for weight absorption
                # METADATA
                position_ids,
                # CONSTANTS
                self.softmax_scale,
            )
            ops = (
                # This operator is used in the final patched forward method.
                torch.ops.auto_deploy.torch_deepseek_mla_no_cache,
                # This operator is not used. It serves as an example of using the flashinfer ragged kernel.
                torch.ops.auto_deploy.flashinfer_deepseek_mla_no_cache,
            )
            assert ad_operator in ops
            attn_output = ad_operator(*args)
        else:
            # These are the two cached MLA operators. We only care about the flashinfer operator
            # because it is optimized for the MQA mode of MLA (i.e. with weight absorption).
            if ad_operator == torch.ops.auto_deploy.torch_deepseek_mla_with_kv_cache:
                args = (
                    q_normed_dn,
                    compressed_kv,
                    k_pe,
                    sin,
                    cos,
                    wkv_b,
                    wq_b,
                    None,  # w_uq_ukv
                    wo_proj,
                    None,  # w_uv_o
                    # METADATA
                    seq_len,
                    input_pos,
                    cache_loc,
                    seq_start,
                    position_ids,
                    # CACHES
                    ckv_cache,
                    k_pe_cache,
                    # CONSTANTS
                    self.softmax_scale,
                )
            elif ad_operator == torch.ops.auto_deploy.flashinfer_deepseek_mla_with_kv_cache:
                q_indptr = kwargs.get("q_indptr", None)
                kv_page_indptr = kwargs.get("kv_page_indptr", None)
                kv_page_indices = kwargs.get("kv_page_indices", None)
                kv_last_page_len = kwargs.get("kv_last_page_len", None)
                batch_indices = kwargs.get("batch_indices", None)
                positions = kwargs.get("positions", None)
                kv_lens = kwargs.get("kv_lens", None)
                page_size = kwargs.get("page_size", None)
                args = (
                    q_normed_dn,
                    compressed_kv,
                    k_pe,
                    sin,
                    cos,
                    wkv_b,
                    wq_b,
                    None,  # w_uq_uk placeholder for weight absorption
                    wo_proj,
                    None,  # w_uv_o placeholder for weight absorption
                    # METADATA
                    q_indptr,
                    kv_page_indptr,
                    kv_page_indices,
                    kv_last_page_len,
                    batch_indices,
                    positions,
                    kv_lens,
                    page_size,
                    position_ids,
                    # CACHES
                    ckv_cache,
                    k_pe_cache,
                    # CONSTANTS
                    self.softmax_scale,
                )
            else:
                raise ValueError(f"Invalid ad_operator: {ad_operator}")

        attn_output = ad_operator(*args)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


# This is the original unmodified code from Hugging Face
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """

    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _causal_attention_mask(batch_size, seqlen_q, seqlen_k, device, dtype=torch.bfloat16):
    """Utility function to create a causal attention mask."""
    causal_mask = torch.triu(torch.ones(seqlen_q, seqlen_k, device=device), diagonal=1).bool()
    attention_mask = torch.zeros(batch_size, 1, seqlen_q, seqlen_k, dtype=dtype, device=device)
    attention_mask.masked_fill_(causal_mask, float("-inf"))
    return attention_mask


def _create_linear_caches(
    max_batch_size: int, max_seq_len: int, device: str, dtype: torch.dtype, init_randn: bool = False
):
    """
    Create ompressed KV and K positional encoding caches (CKV and KPE) with a [B,S,D] layout.
    Note that there is only one (shared) head.
    """

    def _create_cache(head_dim: int):
        tensor_init = torch.randn if init_randn else torch.zeros
        return tensor_init(
            max_batch_size,
            max_seq_len,
            head_dim,
            device=device,
            dtype=dtype,
        )

    head_dim_ckv = 512  # AKA kv_lora_rank
    qk_rope_head_dim = 64
    ckv_cache = _create_cache(head_dim_ckv)
    kpe_cache = _create_cache(qk_rope_head_dim)
    return ckv_cache, kpe_cache


def _create_paged_caches(
    num_pages: int, page_size: int, device: str, dtype: torch.dtype, init_randn: bool = False
):
    """
    The compressed KV and K positional encoding caches (CKV and KPE) with paged layout.
    Note that there is only one (shared) head.
    """

    def _create_cache(head_dim: int):
        tensor_init = torch.randn if init_randn else torch.zeros
        return tensor_init(
            num_pages,
            page_size,
            head_dim,
            device=device,
            dtype=dtype,
        )

    head_dim_ckv = 512  # AKA kv_lora_rank
    qk_rope_head_dim = 64
    ckv_cache = _create_cache(head_dim_ckv)
    kpe_cache = _create_cache(qk_rope_head_dim)
    return ckv_cache, kpe_cache


def test_debug_flashinfer_prefill_kernel():
    """
    This serves as an example of how to use flashinfer.BatchPrefillWithRaggedKVCacheWrapper.
    """
    torch.manual_seed(42)

    # Test configuration
    num_qo_heads = 64
    num_kv_heads = 16
    head_dim = 128
    softmax_scale = 1.0 / ((128 + 64) ** 0.5)

    bsz = 7
    q_len = 10
    cnt_qo = bsz * q_len  # Total tokens across all batches
    cnt_kv = cnt_qo

    # We're not really testing a ragged input here, so the indptr jumps by q_len for all batches.
    kv_indptr = torch.arange(0, bsz * q_len + 1, q_len).int()
    qo_indptr = torch.arange(0, bsz * q_len + 1, q_len).int()

    # Test inputs
    q = torch.randn(cnt_qo, num_qo_heads, head_dim).to(torch.bfloat16).to("cuda:0")
    k = torch.randn(cnt_kv, num_kv_heads, head_dim).to(torch.bfloat16).to("cuda:0")
    v = torch.randn(cnt_kv, num_kv_heads, head_dim).to(torch.bfloat16).to("cuda:0")

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace_buffer, "NHD")

    prefill_wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=True,
        sm_scale=softmax_scale,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    # FlashInfer output
    o = prefill_wrapper.run(q, k, v)

    # Pytorch reference implementation for ragged GQA
    # The key insight: we have 7 separate sequences of length 10, not one sequence of length 70
    # Each sequence should be processed independently with its own causal mask

    # Expand k and v to match the number of query heads
    num_groups = num_qo_heads // num_kv_heads  # 64 // 16 = 4
    k_expanded = k.repeat_interleave(num_groups, dim=1)  # [70, 16, 128] -> [70, 64, 128]
    v_expanded = v.repeat_interleave(num_groups, dim=1)  # [70, 16, 128] -> [70, 64, 128]

    # Reshape to separate batches: [70, 64, 128] -> [7, 10, 64, 128]
    q_batched = q.view(bsz, q_len, num_qo_heads, head_dim)  # [7, 10, 64, 128]
    k_batched = k_expanded.view(bsz, q_len, num_qo_heads, head_dim)  # [7, 10, 64, 128]
    v_batched = v_expanded.view(bsz, q_len, num_qo_heads, head_dim)  # [7, 10, 64, 128]

    # Process all batches simultaneously using batched operations
    # Transpose all batches at once: [7, 10, 64, 128] -> [7, 64, 10, 128]
    q_batched_t = q_batched.transpose(1, 2)  # [7, 64, 10, 128]
    k_batched_t = k_batched.transpose(1, 2)  # [7, 64, 10, 128]
    v_batched_t = v_batched.transpose(1, 2)  # [7, 64, 10, 128]

    # Batched matmul: [7, 64, 10, 128] @ [7, 64, 128, 10] = [7, 64, 10, 10]
    attn_weights = torch.matmul(q_batched_t, k_batched_t.transpose(-1, -2)) * softmax_scale

    # Create causal mask once and broadcast: [10, 10] -> [1, 1, 10, 10]
    causal_mask = torch.triu(torch.ones(q_len, q_len, device=q.device), diagonal=1).bool()
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 10, 10]

    # Apply mask to all batches and heads at once
    attn_weights.masked_fill_(causal_mask, float("-inf"))

    # Batched softmax
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

    # Batched attention: [7, 64, 10, 10] @ [7, 64, 10, 128] = [7, 64, 10, 128]
    attn_output_batched = torch.matmul(attn_weights, v_batched_t)

    # Transpose back: [7, 64, 10, 128] -> [7, 10, 64, 128]
    attn_output_batched = attn_output_batched.transpose(1, 2)

    # Reshape back to original format: [7, 10, 64, 128] -> [70, 64, 128]
    attn_output = attn_output_batched.reshape(-1, num_qo_heads, head_dim)

    print(f"attn_output.shape: {attn_output.shape}")
    print(f"o.shape: {o.shape}")

    diff = (o - attn_output).abs()
    print(f"test_debug_flashinfer_prefill_kernel max difference: {diff.max()}")

    assert torch.allclose(o, attn_output, atol=2e-2, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("seqlen_q", [1, 6])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("device", [torch.device("cuda:0")])
@torch.inference_mode()
def test_debug_deepseek_patch(dtype, seqlen_q, batch_size, device):
    """Test the patched attention implementation

    Compare the output of the patched attention implementation with the output of
    the Hugging Face implementation.
    """
    config = DeepseekV3Config()
    hf_deepseek_attn = DeepseekV3Attention(config).to(device).to(dtype)
    hidden_states = torch.randn(batch_size, seqlen_q, config.hidden_size, dtype=dtype).to(device)
    attention_mask = _causal_attention_mask(batch_size, seqlen_q, seqlen_q, device, dtype)
    hf_deepseek_ref_output, _, _ = hf_deepseek_attn.forward(hidden_states, attention_mask)
    patched_output, _, _ = patched_attention(hf_deepseek_attn, hidden_states, attention_mask)
    assert patched_output.shape == (batch_size, seqlen_q, config.hidden_size)
    diff = (hf_deepseek_ref_output - patched_output).abs().flatten()
    print(f"test_debug_deepseek_patch max difference: {diff.max()}")
    assert torch.allclose(hf_deepseek_ref_output, patched_output, atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("seqlen_q", [1, 6])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("device", [torch.device("cuda:0")])
@torch.inference_mode()
def test_deepseek_mla_no_cache(dtype, seqlen_q, batch_size, device):
    """Test the no-cache MLA operators.

    These operators are not used by the optimized graph.
    Generate a sequence of length seqlen_q.
    Test both decode (seqlen_q==1) and prefill (seqlen_q>1) configurations but
    the operations tested do not use the KV cache.
    """

    torch.manual_seed(42)
    test_type = "decode" if seqlen_q == 1 else "prefill"

    config = DeepseekV3Config()
    hf_deepseek_attn = DeepseekV3Attention(config).to(device).to(dtype)

    hidden_states = torch.randn(batch_size, seqlen_q, config.hidden_size, dtype=dtype).to(device)

    # Create proper causal mask: 0 for past/current positions, -inf for future positions
    causal_mask = torch.triu(torch.ones(seqlen_q, seqlen_q, device=device), diagonal=1).bool()
    attention_mask = torch.zeros(batch_size, 1, seqlen_q, seqlen_q, dtype=dtype, device=device)
    if seqlen_q > 1:
        attention_mask.masked_fill_(causal_mask, float("-inf"))

    # Compute the reference output using the Hugging Face implementation
    hf_deepseek_ref_output, _, _ = hf_deepseek_attn.forward(hidden_states, attention_mask)
    assert hf_deepseek_ref_output.shape == (batch_size, seqlen_q, config.hidden_size)

    torch_no_cache_output, _, _ = hf_deepseek_attn.ad_forward(
        hidden_states,
        attention_mask,
        attn_impl="no_absorb",
        ad_operator=torch.ops.auto_deploy.torch_deepseek_mla_no_cache,
    )
    assert torch_no_cache_output.shape == (batch_size, seqlen_q, config.hidden_size)
    diff = (hf_deepseek_ref_output - torch_no_cache_output).abs().flatten()
    print(f"=> {test_type} no-cache: hf_ref vs torch_deepseek_mla_no_cache: max(diff)={max(diff)}")
    assert torch.allclose(hf_deepseek_ref_output, torch_no_cache_output, atol=1e-2, rtol=1e-3)

    ad_deepseek_output_no_absorb_kernel, _, _ = hf_deepseek_attn.ad_forward(
        hidden_states,
        attention_mask,
        ad_operator=torch.ops.auto_deploy.flashinfer_deepseek_mla_no_cache,
    )
    assert ad_deepseek_output_no_absorb_kernel.shape == (batch_size, seqlen_q, config.hidden_size)
    diff = (hf_deepseek_ref_output - ad_deepseek_output_no_absorb_kernel).abs()
    print(
        f"=> {test_type} no-cache: hf_ref vs flashinfer_deepseek_mla_no_cache max(diff)={diff.max()}"
    )
    assert torch.allclose(
        hf_deepseek_ref_output, ad_deepseek_output_no_absorb_kernel, atol=1e-2, rtol=1e-3
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("seqlen_q", [8, 1])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("max_batch_size", [32])
@pytest.mark.parametrize("max_seq_len", [64])
@pytest.mark.parametrize("device", [torch.device("cuda:0")])
@torch.inference_mode()
def test_deepseek_mla_prefill_cache(
    dtype, seqlen_q, batch_size, max_batch_size, max_seq_len, device
):
    """Test the cached MLA operators for prefill.

    The cache is initially empty.
    """
    torch.manual_seed(42)
    assert max_batch_size >= batch_size
    assert max_seq_len >= 2 * seqlen_q

    config = DeepseekV3Config()
    hf_deepseek_attn = DeepseekV3Attention(config).to(device).to(dtype)

    hidden_states = torch.randn(batch_size, seqlen_q, config.hidden_size, dtype=dtype).to(device)
    attention_mask = _causal_attention_mask(batch_size, seqlen_q, seqlen_q, device, dtype)

    # Compute the reference output using the Hugging Face implementation
    hf_deepseek_ref_output, ckv_cache, kpe_cache = hf_deepseek_attn.forward(
        hidden_states, attention_mask
    )
    assert hf_deepseek_ref_output.shape == (batch_size, seqlen_q, config.hidden_size)

    # Make the shape of the paged cache (num_pages, page_size, head_dim) match the shape of the
    # linear cache (max_bs, max_seq_len, head_dim)
    page_size = max_seq_len

    # Empty caches
    compressed_kv_normed_cache, k_pe_cache = _create_linear_caches(
        max_batch_size, max_seq_len, device, dtype, init_randn=False
    )
    # Positions in the current sequence are all zero for prefilling.
    input_pos = torch.tensor([0] * batch_size, dtype=torch.int32, device=device)
    # Empty cache (prefill) so all sequences will be in loc 0.
    cache_loc = torch.tensor([0] * batch_size, dtype=torch.int32, device=device)
    seq_len = torch.tensor([seqlen_q] * batch_size, dtype=torch.int32, device=device)
    # Positions within each sequence are ordered 0..(seqlen_q-1)
    position_ids = torch.tensor(
        list(range(0, seqlen_q)) * batch_size, dtype=torch.int32, device=device
    )
    position_ids = position_ids.unsqueeze(0)

    # All sequences will have 1 page of size max_seq_len. This constraint allows us to
    # share a single cache with both linear and as paged access.
    assert page_size == max_seq_len
    pages_per_seq = torch.tensor([1] * batch_size, dtype=torch.int32, device=device)
    input_ids = hidden_states.reshape(-1, config.hidden_size)  # [num_tokens, hidden_size]
    hidden_states = hidden_states.reshape(1, -1, config.hidden_size)

    # Compute batch_info_host: [num_prefill, num_prefill_tokens, num_decode]
    if seqlen_q > 1:
        # Prefill case
        num_prefill = batch_size
        num_prefill_tokens = batch_size * seqlen_q
        num_decode = 0
    else:
        # Decode case
        num_prefill = 0
        num_prefill_tokens = 0
        num_decode = batch_size
    batch_info_host = torch.tensor(
        [num_prefill, num_prefill_tokens, num_decode], dtype=torch.int32, device="cpu"
    )

    # Currently this operator is not interesting because it is not used. Consider deleting it.
    test_torch_cache_op = False
    if test_torch_cache_op:
        seq_len, input_pos, cache_loc, seq_start, position_ids = (
            torch.ops.auto_deploy.torch_attention_prepare_fused_mla_metadata(
                input_ids, position_ids, seq_len, input_pos, cache_loc, pages_per_seq, page_size
            )
        )

        torch_output, _, _ = hf_deepseek_attn.ad_forward(
            hidden_states,
            attention_mask,
            attn_impl="absorb",
            seq_len=seq_len,
            input_pos=input_pos,
            cache_loc=cache_loc,
            seq_start=seq_start,
            position_ids=position_ids,
            ckv_cache=compressed_kv_normed_cache,
            k_pe_cache=k_pe_cache,
            ad_operator=torch.ops.auto_deploy.torch_deepseek_mla_with_kv_cache,
        )

        hf_deepseek_ref_output = hf_deepseek_ref_output.flatten()
        torch_output = torch_output.flatten()
        diff = (hf_deepseek_ref_output - torch_output).abs()
        print(
            f"=> prefill +cache: hf_ref vs torch_deepseek_mla_with_kv_cache max: (diff)={diff.max()}"
        )
        assert torch.allclose(hf_deepseek_ref_output, torch_output, atol=1e-2, rtol=1e-3)

    test_flashinfer_cache_op = True
    if test_flashinfer_cache_op:
        (
            q_indptr,
            kv_page_indptr,
            kv_page_indices,
            kv_last_page_len,
            batch_indices,
            positions,
            kv_len_arr,
            page_size,
            position_ids,
        ) = torch.ops.auto_deploy.flashinfer_mla_prepare_metadata(
            input_ids,
            position_ids,
            batch_info_host,
            seq_len,
            input_pos,
            cache_loc,
            pages_per_seq,
            page_size,
        )

        flashinfer_output, _, _ = hf_deepseek_attn.ad_forward(
            hidden_states,
            attention_mask,
            attn_impl="absorb",
            seq_len=seq_len,
            input_pos=input_pos,
            cache_loc=cache_loc,
            position_ids=position_ids,
            ckv_cache=compressed_kv_normed_cache,
            k_pe_cache=k_pe_cache,
            q_indptr=q_indptr,
            kv_page_indptr=kv_page_indptr,
            kv_page_indices=kv_page_indices,
            kv_last_page_len=kv_last_page_len,
            batch_indices=batch_indices,
            positions=positions,
            page_size=page_size,
            kv_lens=kv_len_arr,
            ad_operator=torch.ops.auto_deploy.flashinfer_deepseek_mla_with_kv_cache,
        )

        hf_deepseek_ref_output = hf_deepseek_ref_output.flatten()
        flashinfer_output = flashinfer_output.flatten()
        diff = (hf_deepseek_ref_output - flashinfer_output).abs()
        print(
            f"=> prefill +cache: hf_ref vs flashinfer_deepseek_mla_with_kv_cache  max: (diff)={diff.max()}"
        )
        assert torch.allclose(hf_deepseek_ref_output, flashinfer_output, atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("seqlen_q", [1])
@pytest.mark.parametrize("n_preloaded_pages", [4])
@pytest.mark.parametrize("max_batch_size", [32])
@pytest.mark.parametrize("max_seq_len", [64])
@pytest.mark.parametrize("device", [torch.device("cuda:0")])
@torch.inference_mode()
def _test_deepseek_mla_decode_cache(
    dtype, seqlen_q, n_preloaded_pages, max_batch_size, max_seq_len, device
):
    """Test the cached MLA operators for decode.

    The cache is initially partially filled.
    """
    torch.manual_seed(42)
    # assert max_batch_size >= n_preloaded_pages
    assert max_seq_len >= 2 * seqlen_q
    assert seqlen_q == 1

    config = DeepseekV3Config()
    hf_deepseek_attn = DeepseekV3Attention(config).to(device).to(dtype)

    len_preloaded_seq = max_seq_len
    hf_hidden_states = torch.randn(
        n_preloaded_pages, len_preloaded_seq, config.hidden_size, dtype=dtype
    ).to(device)
    # last_preloaded_page = n_preloaded_pages - 1
    hf_attention_mask = _causal_attention_mask(
        n_preloaded_pages, len_preloaded_seq, len_preloaded_seq, device, dtype
    )

    # Compute the reference output using the Hugging Face implementation
    hf_deepseek_ref_output, hf_ckv, hf_kpe = hf_deepseek_attn.forward(
        hf_hidden_states, hf_attention_mask
    )
    assert hf_deepseek_ref_output.shape == (n_preloaded_pages, max_seq_len, config.hidden_size)

    # Make the shape of the paged cache (num_pages, page_size, head_dim) match the shape of the
    # linear cache (max_bs, max_seq_len, head_dim)
    page_size = max_seq_len
    # seq_len_in_cache = n_preloaded_pages * (page_size - 1)

    # Create empty linear caches and populate them with the ckv and kpe from the reference output.
    compressed_kv_normed_cache, k_pe_cache = _create_linear_caches(
        max_batch_size, max_seq_len, device, dtype, init_randn=False
    )
    compressed_kv_normed_cache[:n_preloaded_pages, :-1, :] = hf_ckv[:, :-1, :]
    k_pe_cache[:n_preloaded_pages, :-1, :] = hf_kpe[:, :, :-1, :].squeeze()
    # Positions in the current sequence start after the cache for decode.
    input_pos = torch.tensor(
        [len_preloaded_seq - 1] * n_preloaded_pages, dtype=torch.int32, device=device
    )
    # Partially filled cache (prefill) so all sequences start after the preloaded cache.
    cache_loc = torch.tensor(
        [len_preloaded_seq - 1] * n_preloaded_pages, dtype=torch.int32, device=device
    )
    seq_len = torch.tensor([seqlen_q] * n_preloaded_pages, dtype=torch.int32, device=device)

    # All sequences will have 1 page of size max_seq_len. This constraint allows us to
    # share a single cache with both linear and as paged access.
    assert page_size == max_seq_len
    pages_per_seq = torch.tensor([1] * n_preloaded_pages, dtype=torch.int32, device=device)
    fi_hidden_states = hf_hidden_states[:n_preloaded_pages, len_preloaded_seq - 1, :].unsqueeze(
        1
    )  # use the last token of each sequence as the decode input
    input_ids = fi_hidden_states.reshape(-1, config.hidden_size)  # [num_tokens, hidden_size]
    hidden_states = fi_hidden_states  # .reshape(1, -1, config.hidden_size)

    fi_position_ids = torch.tensor(
        list((len_preloaded_seq - 1,)) * n_preloaded_pages, dtype=torch.int32, device=device
    )
    fi_position_ids = fi_position_ids.unsqueeze(0)

    # Compute batch_info_host: [num_prefill, num_prefill_tokens, num_decode]
    # This is decode case (seqlen_q == 1)
    num_prefill = 0
    num_prefill_tokens = 0
    num_decode = n_preloaded_pages
    batch_info_host = torch.tensor(
        [num_prefill, num_prefill_tokens, num_decode], dtype=torch.int32, device="cpu"
    )

    test_torch_cache_op = False
    if test_torch_cache_op:
        seq_len, input_pos, cache_loc, seq_start, position_ids = (
            torch.ops.auto_deploy.torch_attention_prepare_fused_mla_metadata(
                input_ids, fi_position_ids, seq_len, input_pos, cache_loc, pages_per_seq, page_size
            )
        )

        torch_output, _, _ = hf_deepseek_attn.ad_forward(
            hidden_states,
            hf_attention_mask,
            attn_impl="absorb",
            seq_len=seq_len,
            input_pos=input_pos,
            cache_loc=cache_loc,
            seq_start=seq_start,
            position_ids=position_ids,
            ckv_cache=compressed_kv_normed_cache,
            k_pe_cache=k_pe_cache,
            ad_operator=torch.ops.auto_deploy.torch_deepseek_mla_with_kv_cache,
        )

        hf_deepseek_ref_output = hf_deepseek_ref_output.flatten()
        torch_output = torch_output.flatten()
        diff = (hf_deepseek_ref_output - torch_output).abs()
        print(
            f"=> prefill +cache: hf_ref vs torch_deepseek_mla_with_kv_cache max: (diff)={diff.max()}"
        )
        assert torch.allclose(hf_deepseek_ref_output, torch_output, atol=1e-2, rtol=1e-3)
        return

    test_flashinfer_cache_op = True
    if test_flashinfer_cache_op:
        (
            q_indptr,
            kv_page_indptr,
            kv_page_indices,
            kv_last_page_len,
            batch_indices,
            positions,
            kv_len_arr,
            page_size,
            position_ids,
        ) = torch.ops.auto_deploy.flashinfer_mla_prepare_metadata(
            input_ids,
            fi_position_ids,
            batch_info_host,
            seq_len,
            input_pos,
            cache_loc,
            pages_per_seq,
            page_size,
        )

        flashinfer_output, _, _ = hf_deepseek_attn.ad_forward(
            hidden_states,
            hf_attention_mask,
            attn_impl="absorb",
            seq_len=seq_len,
            input_pos=input_pos,
            cache_loc=cache_loc,
            position_ids=position_ids,
            ckv_cache=compressed_kv_normed_cache,
            k_pe_cache=k_pe_cache,
            q_indptr=q_indptr,
            kv_page_indptr=kv_page_indptr,
            kv_page_indices=kv_page_indices,
            kv_last_page_len=kv_last_page_len,
            batch_indices=batch_indices,
            positions=positions,
            page_size=page_size,
            kv_lens=kv_len_arr,
            ad_operator=torch.ops.auto_deploy.flashinfer_deepseek_mla_with_kv_cache,
        )

        hf_deepseek_ref_output = hf_deepseek_ref_output[:, -1, :]
        hf_deepseek_ref_output = hf_deepseek_ref_output.flatten()
        flashinfer_output = flashinfer_output.flatten()
        diff = (hf_deepseek_ref_output - flashinfer_output).abs()
        print(
            f"=> decode +cache: hf_ref vs flashinfer_deepseek_mla_with_kv_cache  max: (diff)={diff.max()}"
        )
        assert torch.allclose(hf_deepseek_ref_output, flashinfer_output, atol=1e-2, rtol=1e-3)
