import math
from typing import Optional

import torch
import torch.nn.functional as F

from tensorrt_llm.models.modeling_utils import QuantConfig

try:
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter
except ImportError:
    AttentionMaskConverter = None

from .interface import (AttentionBackend, AttentionMask, AttentionMetadata,
                        PredefinedAttentionMask)
from .sparse.kernel import triton_index_gather


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch, num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


def generate_causal_mask(batch_size: int, target_length: int,
                         cache_position: torch.Tensor, device: torch.device):
    causal_mask = torch.arange(
        target_length,
        device=device).unsqueeze(0) <= cache_position.unsqueeze(-1)
    causal_mask = causal_mask.expand(batch_size, 1, -1, -1)

    return causal_mask


def generate_sliding_window_mask(batch_size: int, target_length: int,
                                 cache_position: torch.Tensor,
                                 device: torch.device,
                                 attention_window_size: int):
    attention_mask_1 = torch.arange(
        target_length,
        device=device).unsqueeze(0) <= cache_position.unsqueeze(-1)
    attention_mask_2 = torch.arange(target_length, device=device).unsqueeze(
        0) > cache_position.unsqueeze(-1) - attention_window_size
    attention_mask = attention_mask_1 & attention_mask_2
    attention_mask = attention_mask[None,
                                    None, :, :].expand(batch_size, 1, -1, -1)
    return attention_mask


class VanillaAttentionMetadata(AttentionMetadata):

    def prepare(self) -> None:
        # indices of used cache blocks for each sequence
        assert self.request_ids is not None
        self.block_ids_per_seq = self.kv_cache_manager.get_batch_cache_indices(
            self.request_ids) if self.kv_cache_manager is not None else None


class VanillaAttention(AttentionBackend[VanillaAttentionMetadata]):

    Metadata = VanillaAttentionMetadata

    _access_type = {
        1: torch.int8,
        2: torch.int16,
        4: torch.int32,
        8: torch.int64
    }

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        q_scaling: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(layer_idx,
                         num_heads,
                         head_dim,
                         num_kv_heads=num_kv_heads,
                         quant_config=quant_config,
                         **kwargs)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.q_scaling = q_scaling

    def _single_request_sparse_attn_predict(
            self, q: torch.Tensor, k: Optional[torch.Tensor],
            v: Optional[torch.Tensor], kv_cache_tensor: torch.Tensor,
            metadata: AttentionMetadata, past_seen_token: int, sample_idx: int,
            **kwargs) -> tuple[Optional[torch.Tensor], int]:
        raise NotImplementedError

    def _single_request_sparse_kv_predict(
            self, q: Optional[torch.Tensor], k: Optional[torch.Tensor],
            v: Optional[torch.Tensor], metadata: AttentionMetadata,
            past_seen_token: int, sample_idx: int,
            **kwargs) -> tuple[Optional[torch.Tensor], int]:
        raise NotImplementedError

    def _single_request_update_kv_cache(self,
                                        k,
                                        v,
                                        kv_cache_tensor,
                                        past_seen_token,
                                        kv_len,
                                        cache_idx,
                                        sparse_kv_indices=None):
        # select tokens using the sparse kv indices
        if sparse_kv_indices is not None:
            k_selected = triton_index_gather(k, sparse_kv_indices)
            v_selected = triton_index_gather(v, sparse_kv_indices)
        else:
            k_selected, v_selected = k, v

        # get cache position
        seq_len = past_seen_token + kv_len
        cache_position = torch.arange(past_seen_token,
                                      seq_len,
                                      device=kv_cache_tensor.device)

        # get kv cache tensor
        k_out = kv_cache_tensor[cache_idx, 0, :, :, :].unsqueeze(0)
        v_out = kv_cache_tensor[cache_idx, 1, :, :, :].unsqueeze(0)

        # update kv cache
        if k is not None and v is not None:
            access_type = self._access_type[k_selected.dtype.itemsize]
            k_out.view(dtype=access_type).index_copy_(
                1, cache_position, k_selected.view(dtype=access_type))
            v_out.view(dtype=access_type).index_copy_(
                1, cache_position, v_selected.view(dtype=access_type))

        # return past kv and the dense kv tensors for sparse attention
        if sparse_kv_indices is not None:
            k_states = torch.cat([k_out[:, :past_seen_token, :, :], k], dim=1)
            v_states = torch.cat([v_out[:, :past_seen_token, :, :], v], dim=1)
        else:
            k_states, v_states = k_out[:, :seq_len, :, :], v_out[:, :
                                                                 seq_len, :, :]
        return k_states, v_states

    def _single_request_preprocess_inputs(self, q, k, v, kv_dtype):
        bsz = 1
        q_len = q.size(0)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_len = 0
        if k is not None and v is not None:
            kv_len = k.size(0)
            k = k.view(bsz, kv_len, self.num_kv_heads, self.head_dim)
            v = v.view(bsz, kv_len, self.num_kv_heads, self.head_dim)

            if self.quant_config and self.quant_config.layer_quant_mode.has_any_quant(
            ):
                qc = self.quant_config
                if qc.layer_quant_mode.has_fp8_kv_cache():
                    assert kv_dtype == torch.float8_e4m3fn, \
                        f"KV cache should have fp8 dtype, but get {kv_dtype}"
                    k = k.to(torch.float8_e4m3fn)
                    v = v.to(torch.float8_e4m3fn)
            assert k.dtype == v.dtype == kv_dtype, \
                f"KV cache dtype {kv_dtype} does not match k/v dtype {k.dtype}/{v.dtype}"

        return q, k, v, kv_len

    def _single_request_create_attention_mask(self,
                                              attention_mask,
                                              past_seen_token,
                                              kv_len,
                                              q_device,
                                              q_len,
                                              attention_window_size=None):
        """
        Create appropriate attention mask based on the attention type.

        Returns:
            Tuple of (is_causal, attn_mask)
        """
        bsz = 1
        is_causal = False
        attn_mask = None

        # get cache position
        seq_len = past_seen_token + kv_len
        cache_position = torch.arange(past_seen_token, seq_len, device=q_device)

        # create attention mask
        if attention_mask == PredefinedAttentionMask.CAUSAL:
            # Create custom sliding window mask as sdpa doesn't natively support it.
            if attention_window_size is not None:
                attn_mask = generate_sliding_window_mask(
                    bsz, seq_len, cache_position, q_device,
                    attention_window_size)
            elif past_seen_token == 0:
                is_causal = True
            elif q_len != 1:
                # attn_mask: 4-D tensor (batch_size, 1, query_seq_len, seq_len)
                attn_mask = generate_causal_mask(bsz, seq_len, cache_position,
                                                 q_device)
        elif attention_mask == PredefinedAttentionMask.FULL:
            pass
        else:
            raise ValueError("Unexpected attention mask type")

        return attn_mask, is_causal

    def _single_request_attn_forward(self,
                                     q,
                                     key_states,
                                     value_states,
                                     is_causal,
                                     attn_mask,
                                     sparse_indices=None):
        """
        Common attention computation using scaled dot-product attention.
        """
        # select the key and value states using the sparse indices
        if sparse_indices is not None:
            key_states = triton_index_gather(key_states, sparse_indices)
            value_states = triton_index_gather(value_states, sparse_indices)

        # transpose kv
        key_states = key_states.transpose(1, 2).to(q.dtype)
        value_states = value_states.transpose(1, 2).to(q.dtype)

        # repeat kv to support MQA/GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # get qk scale
        qk_scale = None
        if self.q_scaling is not None:
            qk_scale = 1 / (math.sqrt(self.head_dim) * self.q_scaling)

        # attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            key_states,
            value_states,
            is_causal=is_causal,
            attn_mask=attn_mask,
            scale=qk_scale,
        )

        return attn_output

    def _single_request_forward(self,
                                q,
                                k,
                                v,
                                attention_mask: AttentionMask,
                                kv_cache_tensor,
                                past_seen_token,
                                cache_idx,
                                sample_idx,
                                metadata: AttentionMetadata,
                                attention_window_size: Optional[int] = None,
                                **kwargs):
        # preprocess inputs
        q, k, v, kv_len = self._single_request_preprocess_inputs(
            q, k, v, kv_cache_tensor.dtype)

        # predict sparse kv indices
        sparse_kv_indices = None
        if self.sparse_attention_config is not None:
            sparse_kv_indices, kv_len = self._single_request_sparse_kv_predict(
                q, k, v, metadata, past_seen_token, sample_idx)

        # update kv cache
        key_states, value_states = self._single_request_update_kv_cache(
            k, v, kv_cache_tensor, past_seen_token, kv_len, cache_idx,
            sparse_kv_indices)

        # predict sparse attn indices
        sparse_indices = None
        if self.sparse_attention_config is not None:
            sparse_indices, kv_len = self._single_request_sparse_attn_predict(
                q, k, v, kv_cache_tensor, metadata, past_seen_token, sample_idx)

        # create attention mask
        attn_mask, is_causal = self._single_request_create_attention_mask(
            attention_mask, past_seen_token, kv_len, q.device, q.size(2),
            attention_window_size)

        # attention
        attn_output = self._single_request_attn_forward(q, key_states,
                                                        value_states, is_causal,
                                                        attn_mask,
                                                        sparse_indices)

        return attn_output.squeeze(0)

    def no_kv_cache_forward(
            self,
            q: torch.Tensor,
            k: Optional[torch.Tensor],
            v: Optional[torch.Tensor],
            num_heads: int,
            num_kv_heads: int,
            metadata: AttentionMetadata,
            *,
            attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
            position_ids: Optional[torch.Tensor] = None,
            **kwargs) -> torch.Tensor:
        """
        This function is used to perform attention without kv cache.
        Args:
            q (torch.Tensor): Query tensor with shape (seq_len, num_heads * head_dim) or (seq_len, (num_heads + 2 * num_kv_heads) * head_dim),
            k (Optional[torch.Tensor]): Key tensor with shape (seq_len, num_heads * head_dim) or None,
            v (Optional[torch.Tensor]): Value tensor with shape (seq_len, num_heads * head_dim) or None,
        """
        # lazy loading
        from flash_attn.flash_attn_interface import flash_attn_varlen_func
        head_dim = q.shape[-1]
        is_fused_qkv = False
        if (k is None) or (v is None):
            assert (k is None) or (
                v is None), "Both k and v has to be None if any of them is None"
            is_fused_qkv = True

        if is_fused_qkv:
            q_size = int(head_dim * num_heads / (num_heads + 2 * num_kv_heads))
            kv_size = int(head_dim * num_kv_heads /
                          (num_heads + 2 * num_kv_heads))
            q, k, v = q.split([q_size, kv_size, kv_size], dim=-1)
        else:
            q_size = head_dim
        head_dim = int(q_size / num_heads)
        q = q.reshape(-1, num_heads, head_dim).contiguous()
        k = k.reshape(-1, num_kv_heads, head_dim).contiguous()
        v = v.reshape(-1, num_kv_heads, head_dim).contiguous()
        assert q.dim() == 3
        assert k.dim() == 3
        assert v.dim() == 3
        seqlens_in_batch = metadata.seq_lens
        assert seqlens_in_batch is not None, "seq_len can not be None for remove padding inputs attention!"
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = F.pad(
            torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32),
            (1, 0)).to(q.device)

        max_seqlen_q = max_seqlen_k = max_seqlen_in_batch
        cu_seqlens_q = cu_seqlens_k = cu_seqlens

        attn_output_unpad = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=None,
            causal=attention_mask == PredefinedAttentionMask.CAUSAL,
            # window_size=(-1, -1),  # -1 means infinite context window
            alibi_slopes=None,
            deterministic=False,
            return_attn_probs=False,
        )

        return attn_output_unpad.reshape(attn_output_unpad.size(0), -1)

    def forward(self,
                q: torch.Tensor,
                k: Optional[torch.Tensor],
                v: Optional[torch.Tensor],
                metadata: VanillaAttentionMetadata,
                *,
                attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
                attention_window_size: Optional[int] = None,
                **kwargs) -> torch.Tensor:
        if metadata.kv_cache_manager is None:
            # NOTE: WAR for no kv cache attn e.g. BERT,
            # try to separate the kv cache estimation path from no kv cache attn.
            num_heads = self.num_heads
            num_kv_heads = self.num_kv_heads
            return self.no_kv_cache_forward(q=q,
                                            k=k,
                                            v=v,
                                            num_heads=num_heads,
                                            num_kv_heads=num_kv_heads,
                                            metadata=metadata,
                                            attention_mask=attention_mask,
                                            **kwargs)

        past_seen_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq
        cache_indices = [
            block_ids[0] for block_ids in metadata.block_ids_per_seq
        ]
        kv_cache_tensor = metadata.kv_cache_manager.get_buffers(self.layer_idx)

        q_len = q.size(0)

        assert len(cache_indices) == len(past_seen_tokens)
        assert len(cache_indices) == metadata.seq_lens.nelement()

        offset = 0
        offset_kv = 0
        attn_outputs = []
        for sample_idx, (seq_len, seq_len_kv) in enumerate(
                zip(metadata.seq_lens, metadata.seq_lens_kv)):
            single_q = q[offset:offset + seq_len]
            single_k = k[
                offset_kv:offset_kv +
                seq_len_kv] if k is not None and seq_len_kv != 0 else None
            single_v = v[
                offset_kv:offset_kv +
                seq_len_kv] if v is not None and seq_len_kv != 0 else None

            past_seen_token = past_seen_tokens[sample_idx]
            cache_idx = cache_indices[sample_idx]

            attn_output = self._single_request_forward(
                single_q, single_k, single_v, attention_mask, kv_cache_tensor,
                past_seen_token, cache_idx, sample_idx, metadata,
                attention_window_size, **kwargs)

            attn_outputs.append(attn_output)

            offset += seq_len
            offset_kv += seq_len_kv

        attn_output = torch.cat(attn_outputs, dim=1)
        attn_output = attn_output.transpose(0, 1).contiguous()
        attn_output = attn_output.view(q_len, -1)

        return attn_output
