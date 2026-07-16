# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import replace
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from tensorrt_llm.models.modeling_utils import QuantConfig

try:
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter
except ImportError:
    AttentionMaskConverter = None

from .interface import (AttentionBackend, AttentionForwardArgs,
                        AttentionInputType, AttentionMask, AttentionMetadata,
                        PositionalEmbeddingParams, PredefinedAttentionMask,
                        merge_attention_forward_args)
from .sparse.kernel import triton_index_gather
from .sparse.params import SparseParams


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

    def __post_init__(self) -> None:
        super().__post_init__()
        self.kv_layout = "NHD"

    def prepare(self) -> None:
        super().prepare()
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
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        sparse_params: Optional[SparseParams] = None,
        **kwargs,
    ):
        super().__init__(layer_idx,
                         num_heads,
                         head_dim,
                         num_kv_heads=num_kv_heads,
                         quant_config=quant_config,
                         **kwargs)
        self.sparse_params = sparse_params
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.q_scaling = q_scaling
        mla_params = kwargs.get("mla_params", None)
        self.is_mla_enable = mla_params is not None
        if self.is_mla_enable:
            self.kv_lora_rank = mla_params.kv_lora_rank
            self.qk_rope_head_dim = mla_params.qk_rope_head_dim
            self.qk_nope_head_dim = mla_params.qk_nope_head_dim
            self.v_head_dim = mla_params.v_head_dim

        self.sparse_mla_rope_cos_sin = None
        self.sparse_mla_rope_is_neox = True
        if (self.is_mla_enable
                and getattr(self.sparse_params, "algorithm", None) == "dsa"
                and pos_embd_params is not None
                and pos_embd_params.rope is not None):
            self.sparse_mla_rope_cos_sin = pos_embd_params.rope.create_rope_const_params(
                interleave=False)[1].reshape(pos_embd_params.rope.max_positions,
                                             2, -1)
            self.sparse_mla_rope_is_neox = pos_embd_params.is_neox

    @classmethod
    def support_mla(cls) -> bool:
        return True

    def sparse_kv_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: VanillaAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Lower backend-neutral sparse KV selection for Vanilla attention."""
        return None, None

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: VanillaAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Use request-local caller-provided selections in the Vanilla backend."""
        return forward_args.topk_indices, None

    @staticmethod
    def _apply_rotary_embedding(x: torch.Tensor, cos: torch.Tensor,
                                sin: torch.Tensor,
                                is_neox: bool) -> torch.Tensor:
        """Apply RoPE to ``x`` using one cos/sin row per packed token."""
        cos = cos.to(device=x.device, dtype=x.dtype).unsqueeze(1)
        sin = sin.to(device=x.device, dtype=x.dtype).unsqueeze(1)
        rotary_dim = cos.shape[-1] * 2
        x_rotary, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        if is_neox:
            x1, x2 = x_rotary.chunk(2, dim=-1)
        else:
            x1, x2 = x_rotary[..., ::2], x_rotary[..., 1::2]
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin
        if is_neox:
            rotated = torch.cat((out1, out2), dim=-1)
        else:
            rotated = torch.stack((out1, out2), dim=-1).flatten(-2)
        return torch.cat((rotated, x_pass), dim=-1)

    def _prepare_sparse_mla_inputs(
        self,
        fused_q: torch.Tensor,
        latent_cache: torch.Tensor,
        q_pe: Optional[torch.Tensor],
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply sparse MLA RoPE to raw packed query and latent-cache inputs.

        As with the other attention paths, omitting positional-embedding
        parameters means the caller already applied RoPE.
        """
        if self.sparse_mla_rope_cos_sin is None:
            return fused_q, latent_cache

        if positions.numel() == 0:
            return fused_q, latent_cache
        max_position = int(positions.max().item())
        if max_position >= self.sparse_mla_rope_cos_sin.shape[0]:
            raise ValueError(
                f"Sparse MLA position {max_position} exceeds the configured RoPE table "
                f"size {self.sparse_mla_rope_cos_sin.shape[0]}")

        num_tokens = fused_q.shape[0]
        fused_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        query = fused_q.view(num_tokens, self.num_heads, fused_head_dim).clone()
        if q_pe is None:
            raise ValueError(
                "Vanilla sparse MLA requires raw q_pe when RoPE parameters are configured"
            )
        expected_numel = num_tokens * self.num_heads * self.qk_rope_head_dim
        if q_pe.numel() != expected_numel:
            raise ValueError(
                f"Sparse MLA q_pe has {q_pe.numel()} elements, expected {expected_numel}"
            )
        query_rope = q_pe.reshape(num_tokens, self.num_heads,
                                  self.qk_rope_head_dim)

        if latent_cache.shape[1] != fused_head_dim:
            raise ValueError(
                f"Sparse MLA latent cache width must be {fused_head_dim}, got "
                f"{latent_cache.shape[1]}")
        latent_cache = latent_cache.clone()
        key_rope = latent_cache[:, self.kv_lora_rank:].unsqueeze(1)
        cos_sin = self.sparse_mla_rope_cos_sin.index_select(
            0,
            positions.to(device=self.sparse_mla_rope_cos_sin.device,
                         dtype=torch.long))
        cos, sin = cos_sin.unbind(dim=1)
        query[..., -self.qk_rope_head_dim:] = self._apply_rotary_embedding(
            query_rope, cos, sin, self.sparse_mla_rope_is_neox)
        latent_cache[:, self.kv_lora_rank:] = self._apply_rotary_embedding(
            key_rope, cos, sin, self.sparse_mla_rope_is_neox).squeeze(1)
        return query.view(num_tokens, -1), latent_cache

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

        # get qk scale
        qk_scale = None
        if self.q_scaling is not None:
            qk_scale = 1 / (math.sqrt(self.head_dim) * self.q_scaling)

        return torch.nn.functional.scaled_dot_product_attention(
            q,
            key_states,
            value_states,
            is_causal=is_causal,
            attn_mask=attn_mask,
            scale=qk_scale,
            enable_gqa=True,
        )

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
                                attention_window_size: Optional[int] = None):
        # preprocess inputs
        q, k, v, kv_len = self._single_request_preprocess_inputs(
            q, k, v, kv_cache_tensor.dtype)

        # predict sparse kv indices
        sparse_kv_indices = None
        if self.sparse_params is not None:
            sparse_kv_indices, kv_len = self._single_request_sparse_kv_predict(
                q, k, v, metadata, past_seen_token, sample_idx)

        # update kv cache
        key_states, value_states = self._single_request_update_kv_cache(
            k, v, kv_cache_tensor, past_seen_token, kv_len, cache_idx,
            sparse_kv_indices)

        # predict sparse attn indices
        sparse_indices = None
        if self.sparse_params is not None:
            sparse_indices, kv_len = self._single_request_sparse_attn_predict(
                q, k, v, kv_cache_tensor, metadata, past_seen_token, sample_idx)

        # Create attention mask.
        attn_mask, is_causal = self._single_request_create_attention_mask(
            attention_mask, past_seen_token, kv_len, q.device, q.size(2),
            attention_window_size)

        # Run attention.
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
        """Perform attention without kv cache.

        Supports both self-attention (Q and K/V have matching per-request
        lengths) and cross-attention (Q-side lengths from
        ``metadata.seq_lens``, K/V-side lengths from ``metadata.seq_lens_kv``,
        i.e. ``metadata.is_cross is True``).

        Args:
            q: Query tensor, shape ``(seq_len_q, num_heads * head_dim)``
                or ``(seq_len_q, (num_heads + 2*num_kv_heads) * head_dim)``.
            k: Key tensor, shape ``(seq_len_kv, num_kv_heads * head_dim)`` or
                None (fused QKV input).
            v: Value tensor, shape ``(seq_len_kv, num_kv_heads * head_dim)``
                or None (fused QKV input).
        """
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
        seqlens_q = metadata.seq_lens
        assert seqlens_q is not None, "seq_len can not be None for remove padding inputs attention!"
        seqlens_kv = metadata.seq_lens_kv
        # In cross-attention the K/V-side lengths differ from the Q-side
        # lengths and must be tracked separately for cu_seqlens.
        is_cross = metadata.is_cross
        if is_fused_qkv and is_cross:
            raise ValueError(
                "Cross-attention with fused QKV input is not supported: pass "
                "Q, K, V as separate tensors when metadata.is_cross is True.")
        max_seqlen_q = int(seqlens_q.max().item())
        cu_seqlens_q = F.pad(torch.cumsum(seqlens_q, dim=0, dtype=torch.int32),
                             (1, 0)).to(q.device)
        if is_cross:
            assert seqlens_kv is not None, (
                "metadata.seq_lens_kv must be set for cross-attention "
                "(no_kv_cache_forward). Got None.")
            assert seqlens_kv.sum().item() == k.size(0), (
                "K tensor token count does not match metadata.seq_lens_kv: "
                f"k.shape[0]={k.size(0)} vs sum(seq_lens_kv)="
                f"{seqlens_kv.sum().item()}.")
            max_seqlen_k = int(seqlens_kv.max().item())
            cu_seqlens_k = F.pad(
                torch.cumsum(seqlens_kv, dim=0, dtype=torch.int32),
                (1, 0)).to(q.device)
        else:
            max_seqlen_k = max_seqlen_q
            cu_seqlens_k = cu_seqlens_q

        # flash-attn only supports fp16/bf16; fall back to PyTorch SDPA for
        # other dtypes (e.g. float32), mirroring the TRT backend's behaviour
        # of disabling context_fmha for float32.
        if q.dtype not in (torch.float16, torch.bfloat16):
            return self._no_kv_cache_sdpa_fallback(q, k, v, num_heads,
                                                   num_kv_heads, head_dim,
                                                   seqlens_q, cu_seqlens_q,
                                                   max_seqlen_q, attention_mask,
                                                   seqlens_kv, cu_seqlens_k,
                                                   max_seqlen_k, is_cross)

        from flash_attn.flash_attn_interface import flash_attn_varlen_func

        softmax_scale = None
        if self.q_scaling is not None:
            softmax_scale = 1 / (math.sqrt(head_dim) * self.q_scaling)

        attn_output_unpad = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=attention_mask == PredefinedAttentionMask.CAUSAL
            and not is_cross,
            alibi_slopes=None,
            deterministic=False,
            return_attn_probs=False,
        )

        return attn_output_unpad.reshape(attn_output_unpad.size(0), -1)

    def _no_kv_cache_sdpa_fallback(self,
                                   q: torch.Tensor,
                                   k: torch.Tensor,
                                   v: torch.Tensor,
                                   num_heads: int,
                                   num_kv_heads: int,
                                   head_dim: int,
                                   seqlens_q: torch.Tensor,
                                   cu_seqlens_q: torch.Tensor,
                                   max_seqlen_q: int,
                                   attention_mask: AttentionMask,
                                   seqlens_kv: Optional[torch.Tensor] = None,
                                   cu_seqlens_k: Optional[torch.Tensor] = None,
                                   max_seqlen_k: Optional[int] = None,
                                   is_cross: bool = False) -> torch.Tensor:
        """PyTorch SDPA fallback for dtypes not supported by flash-attn.

        When ``seqlens_kv`` / ``cu_seqlens_k`` are provided, K/V are sliced
        independently of Q (cross-attention path).
        """
        del max_seqlen_q, max_seqlen_k  # only seqlens / cu_seqlens are used
        is_causal = (attention_mask == PredefinedAttentionMask.CAUSAL)
        num_requests = seqlens_q.numel()

        if seqlens_kv is None or cu_seqlens_k is None:
            seqlens_kv = seqlens_q
            cu_seqlens_k = cu_seqlens_q

        outputs = []
        for i in range(num_requests):
            start_q = cu_seqlens_q[i].item()
            end_q = cu_seqlens_q[i + 1].item()
            start_k = cu_seqlens_k[i].item()
            end_k = cu_seqlens_k[i + 1].item()
            q_s = q[start_q:end_q].transpose(0, 1).unsqueeze(0)
            k_s = k[start_k:end_k].transpose(0, 1).unsqueeze(0)
            v_s = v[start_k:end_k].transpose(0, 1).unsqueeze(0)

            qk_scale = None
            if self.q_scaling is not None:
                qk_scale = 1 / (math.sqrt(head_dim) * self.q_scaling)

            # SDPA's is_causal flag implies square attention. Cross-attention
            # is never causal: the decoder Q attends to all encoder K/V tokens.
            sdpa_is_causal = (is_causal and not is_cross
                              and (end_q - start_q) == (end_k - start_k))
            out = F.scaled_dot_product_attention(q_s,
                                                 k_s,
                                                 v_s,
                                                 is_causal=sdpa_is_causal,
                                                 scale=qk_scale,
                                                 enable_gqa=True)
            outputs.append(out.squeeze(0).transpose(0, 1))

        result = torch.cat(outputs, dim=0)
        return result.reshape(result.size(0), -1)

    def _mla_forward_generation(self, fused_q: torch.Tensor,
                                metadata: VanillaAttentionMetadata,
                                latent_cache: torch.Tensor) -> torch.Tensor:
        """Absorbed MLA generation: MQA of ``fused_q`` over the latent cache.

        The latent cache stores ``[compressed_kv | k_pe]`` with a single KV head
        (head_dim ``kv_lora_rank + qk_rope_head_dim``). Each query head attends to
        it (MQA); the value is the ``kv_lora_rank`` slice of the same entries, so
        the output head_dim is ``kv_lora_rank``. RoPE is already applied to the
        rope portions of ``fused_q`` / ``latent_cache`` by the MLA module
        (``forward_absorption_generation``) before ``forward`` is called.

        Mirrors :meth:`FlashInferAttention._mla_forward_generation`: the new
        ``latent_cache`` tokens are appended to the paged cache, then attention
        runs over the cached prefix plus the new tokens.
        """
        num_tokens = fused_q.shape[0]
        d_latent = self.kv_lora_rank + self.qk_rope_head_dim
        q = fused_q.view(num_tokens, self.num_heads, d_latent)

        # MLA KV cache: NHD [num_pages, kv_factor=1, page_size, num_kv_heads=1,
        # kv_lora_rank + qk_rope_head_dim]. Vanilla is single-block per sequence.
        from .utils import append_mla_latent_cache
        kv_cache = append_mla_latent_cache(
            metadata.kv_cache_manager,
            self.layer_idx,
            metadata.request_ids,
            metadata.seq_lens.tolist(),
            metadata.kv_cache_params.num_cached_tokens_per_seq,
            latent_cache,
            kv_layout=metadata.kv_layout,
        )
        past = metadata.kv_cache_params.num_cached_tokens_per_seq
        cache_indices = [
            block_ids[0] for block_ids in metadata.block_ids_per_seq
        ]

        # MLA scales by the q/k head_dim (qk_nope + qk_rope), not the latent dim.
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        scale = 1.0 / (math.sqrt(qk_head_dim) * (self.q_scaling or 1.0))

        outputs = []
        offset = 0
        for i, q_len in enumerate(metadata.seq_lens.tolist()):
            past_i = int(past[i])
            ci = cache_indices[i]
            kv_len = past_i + q_len

            # K is the full latent ([compressed_kv | k_pe]); V is the kv_lora
            # slice. One latent head broadcast (MQA) to all query heads.
            latent = kv_cache[ci, 0, :kv_len, 0, :].to(q.dtype)
            k = latent[None, None]  # [1, 1, kv_len, d_latent]
            v = latent[None,
                       None, :, :self.kv_lora_rank]  # [1, 1, kv_len, kv_lora]
            qi = q[offset:offset + q_len].transpose(0,
                                                    1)[None]  # [1, H, q_len, d]

            attn_mask = None
            if q_len > 1:
                # MTP/causal: query j (cached position kv_len - q_len + j) may
                # attend up to and including its own position. SDPA bool mask:
                # True == participate.
                rows = kv_len - q_len + torch.arange(q_len, device=q.device)
                keep = (torch.arange(kv_len, device=q.device)[None, :]
                        <= rows[:, None])
                attn_mask = keep.view(1, 1, q_len, kv_len)

            # enable_gqa broadcasts the single latent head to all query heads.
            out = torch.nn.functional.scaled_dot_product_attention(
                qi,
                k,
                v,
                is_causal=False,
                attn_mask=attn_mask,
                scale=scale,
                enable_gqa=True)  # [1, H, q_len, kv_lora]
            outputs.append(out[0].transpose(0, 1).reshape(
                q_len, self.num_heads * self.kv_lora_rank))
            offset += q_len

        return torch.cat(outputs, dim=0)

    @staticmethod
    def _load_mla_latent_cache(kv_cache: torch.Tensor, block_ids: list[int],
                               kv_len: int, kv_layout: str) -> torch.Tensor:
        """Materialize one request's logical MLA cache from its pages."""
        if kv_len <= 0:
            raise ValueError(f"MLA KV length must be positive, got {kv_len}")
        if kv_layout == "NHD":
            tokens_per_block = kv_cache.shape[2]
        elif kv_layout == "HND":
            tokens_per_block = kv_cache.shape[3]
        else:
            raise ValueError(f"Unsupported KV cache layout: {kv_layout}")

        valid_block_ids = [block_id for block_id in block_ids if block_id != -1]
        num_required_blocks = math.ceil(kv_len / tokens_per_block)
        if len(valid_block_ids) < num_required_blocks:
            raise ValueError(
                f"MLA cache has {len(valid_block_ids)} blocks, but "
                f"{num_required_blocks} are required for {kv_len} tokens")

        chunks = []
        remaining = kv_len
        for block_id in valid_block_ids[:num_required_blocks]:
            num_tokens = min(tokens_per_block, remaining)
            if kv_layout == "NHD":
                chunk = kv_cache[block_id, 0, :num_tokens, 0, :]
            else:
                chunk = kv_cache[block_id, 0, 0, :num_tokens, :]
            chunks.append(chunk)
            remaining -= num_tokens
        return torch.cat(chunks, dim=0)

    def _mla_forward_sparse(
        self,
        fused_q: torch.Tensor,
        metadata: VanillaAttentionMetadata,
        latent_cache: torch.Tensor,
        q_pe: Optional[torch.Tensor],
        topk_indices: torch.Tensor,
        attention_input_type: AttentionInputType,
    ) -> torch.Tensor:
        """Run selected sparse MLA from caller-provided local top-k rows.

        The sparse algorithm owns selection. This golden consumes its
        request-local selections, gathers the selected latent K/V, and performs
        the absorbed MLA attention directly in PyTorch. Selections are block
        indices; this reference implements ``block_size == 1`` (token selection),
        which is what the MLA sparse algorithms (DSA / DeepSeek-V4) use.
        """
        block_size = getattr(self.sparse_params, "indices_block_size", 1)
        if block_size != 1:
            raise NotImplementedError(
                "Vanilla selected MLA supports block_size 1 (token selection); "
                f"got block_size {block_size}")
        if attention_input_type == AttentionInputType.context_only:
            seq_start, seq_end = 0, metadata.num_contexts
        elif attention_input_type == AttentionInputType.generation_only:
            seq_start, seq_end = metadata.num_contexts, metadata.num_seqs
        else:
            raise ValueError(
                "Vanilla DSA requires a context-only or generation-only input")

        seq_lens = metadata.seq_lens.tolist()
        phase_seq_lens = seq_lens[seq_start:seq_end]
        num_phase_tokens = sum(phase_seq_lens)
        fused_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        if fused_q.shape[0] != num_phase_tokens:
            raise ValueError(
                f"DSA query has {fused_q.shape[0]} tokens, but metadata "
                f"describes {num_phase_tokens} tokens for this phase")
        if fused_q.ndim != 2 or fused_q.shape[
                1] != self.num_heads * fused_head_dim:
            raise ValueError(
                "DSA query must have shape "
                f"[{num_phase_tokens}, {self.num_heads * fused_head_dim}]; "
                f"got {tuple(fused_q.shape)}")
        if (latent_cache.ndim != 2
                or latent_cache.shape != (num_phase_tokens, fused_head_dim)):
            raise ValueError("DSA latent cache must have shape "
                             f"[{num_phase_tokens}, {fused_head_dim}]; "
                             f"got {tuple(latent_cache.shape)}")
        if topk_indices.ndim != 2 or topk_indices.shape[0] != num_phase_tokens:
            raise ValueError(
                "DSA top-k indices must have shape [num_phase_tokens, top_k]; "
                f"got {tuple(topk_indices.shape)}")
        if topk_indices.dtype != torch.int32:
            raise ValueError(
                f"DSA top-k indices must have dtype int32, got {topk_indices.dtype}"
            )

        request_ids = metadata.request_ids[seq_start:seq_end]
        past_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq
        phase_past_tokens = past_tokens[seq_start:seq_end]
        valid_mask = topk_indices >= 0
        if torch.any(topk_indices < -1):
            raise ValueError("DSA top-k indices may only use -1 as padding")
        if torch.any(~valid_mask.any(dim=1)):
            raise ValueError(
                "Every DSA query token must select at least one KV token")

        kv_lengths = torch.cat([
            torch.full(
                (q_len, ),
                int(past) + q_len,
                dtype=topk_indices.dtype,
                device=topk_indices.device,
            ) for past, q_len in zip(
                phase_past_tokens, phase_seq_lens, strict=True)
        ])
        if torch.any(valid_mask & (topk_indices >= kv_lengths.unsqueeze(1))):
            raise ValueError(
                "DSA top-k index is out of bounds for its request-local KV length"
            )

        causal_limits = torch.cat([
            torch.arange(
                int(past),
                int(past) + q_len,
                dtype=topk_indices.dtype,
                device=topk_indices.device,
            ) for past, q_len in zip(
                phase_past_tokens, phase_seq_lens, strict=True)
        ])
        if torch.any(valid_mask & (topk_indices > causal_limits.unsqueeze(1))):
            raise ValueError("DSA top-k index selects a future token")
        del valid_mask, kv_lengths, causal_limits

        phase_token_start = sum(seq_lens[:seq_start])
        if metadata.position_ids is not None:
            positions = metadata.position_ids.reshape(
                -1)[phase_token_start:phase_token_start + num_phase_tokens].to(
                    device=fused_q.device, dtype=torch.long)
            if positions.numel() != num_phase_tokens:
                raise ValueError(
                    "DSA metadata does not provide one position ID per phase token"
                )
        else:
            positions = torch.cat([
                torch.arange(int(past),
                             int(past) + q_len,
                             device=fused_q.device,
                             dtype=torch.long) for past, q_len in zip(
                                 phase_past_tokens, phase_seq_lens, strict=True)
            ])
        fused_q, latent_cache = self._prepare_sparse_mla_inputs(
            fused_q, latent_cache, q_pe, positions)

        from .utils import append_mla_latent_cache
        kv_cache = append_mla_latent_cache(
            metadata.kv_cache_manager,
            self.layer_idx,
            request_ids,
            phase_seq_lens,
            phase_past_tokens,
            latent_cache,
            kv_layout=metadata.kv_layout,
        )

        q = fused_q.view(num_phase_tokens, self.num_heads, fused_head_dim)
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        scale = 1.0 / (math.sqrt(qk_head_dim) *
                       (self.q_scaling if self.q_scaling is not None else 1.0))

        outputs = []
        token_offset = 0
        for phase_idx, q_len in enumerate(phase_seq_lens):
            seq_idx = seq_start + phase_idx
            kv_len = int(phase_past_tokens[phase_idx]) + q_len
            latent = self._load_mla_latent_cache(
                kv_cache, metadata.block_ids_per_seq[seq_idx], kv_len,
                metadata.kv_layout).to(q.dtype)

            per_token_outputs = []
            for token_idx in range(q_len):
                row = topk_indices[token_offset + token_idx]
                selected = row[row >= 0].to(device=q.device, dtype=torch.long)
                selected_latent = latent.index_select(0, selected)
                query = q[token_offset + token_idx]
                scores = torch.matmul(query, selected_latent.transpose(
                    0, 1)) * scale
                probabilities = F.softmax(scores, dim=-1,
                                          dtype=torch.float32).to(q.dtype)
                values = selected_latent[:, :self.kv_lora_rank]
                per_token_outputs.append(torch.matmul(probabilities, values))

            outputs.append(
                torch.stack(per_token_outputs).reshape(
                    q_len, self.num_heads * self.kv_lora_rank))
            token_offset += q_len

        return torch.cat(outputs, dim=0)

    def _mla_forward_context(self, q: torch.Tensor, k: torch.Tensor,
                             v: torch.Tensor,
                             metadata: VanillaAttentionMetadata,
                             latent_cache: torch.Tensor) -> torch.Tensor:
        """Up-projected MLA context: causal MHA with asymmetric K/V.

        The module up-projects compressed_kv to full per-head K/V, so this is
        ordinary causal self-attention except the K head_dim (qk_nope + qk_rope)
        differs from the V head_dim (v_head_dim). The latent cache append does
        not affect this prefill output, but later generation reads it from the
        same cache manager, so Vanilla mirrors the production backends and
        appends it here.
        """
        from .utils import append_mla_latent_cache
        append_mla_latent_cache(
            metadata.kv_cache_manager,
            self.layer_idx,
            metadata.request_ids,
            metadata.seq_lens.tolist(),
            metadata.kv_cache_params.num_cached_tokens_per_seq,
            latent_cache,
            kv_layout=metadata.kv_layout,
        )

        qk_head = self.qk_nope_head_dim + self.qk_rope_head_dim
        H = self.num_heads
        q = q.view(-1, H, qk_head)
        k = k.view(-1, self.num_kv_heads, qk_head)
        v = v.view(-1, self.num_kv_heads, self.v_head_dim)
        scale = 1.0 / (math.sqrt(qk_head) * (self.q_scaling or 1.0))

        outputs = []
        offset = 0
        for s in metadata.seq_lens.tolist():
            qi = q[offset:offset + s].transpose(0,
                                                1)[None]  # [1, H, s, qk_head]
            ki = k[offset:offset + s].transpose(0, 1)[None]
            vi = v[offset:offset + s].transpose(0, 1)[None]
            out = torch.nn.functional.scaled_dot_product_attention(
                qi, ki, vi, is_causal=True, scale=scale,
                enable_gqa=True)  # [1, H, s, v_head]
            outputs.append(out[0].transpose(0,
                                            1).reshape(s, H * self.v_head_dim))
            offset += s
        return torch.cat(outputs, dim=0)

    def forward(self,
                q: torch.Tensor,
                k: Optional[torch.Tensor],
                v: Optional[torch.Tensor],
                metadata: VanillaAttentionMetadata,
                forward_args: Optional[AttentionForwardArgs] = None,
                **kwargs) -> torch.Tensor:
        forward_args = merge_attention_forward_args(forward_args, kwargs)

        if metadata.multi_item_part_lens is not None:
            raise ValueError(
                "Vanilla Attention does not support multi-item scoring")

        if self.is_mla_enable:
            if metadata.kv_cache_manager is None:
                raise ValueError("Vanilla MLA requires a KV cache manager.")
            if forward_args.latent_cache is None:
                raise ValueError("Vanilla MLA requires latent_cache.")
            if self.sparse_params is not None:
                sparse_algorithm = self.sparse_params.algorithm
                if sparse_algorithm != "dsa":
                    raise ValueError(
                        "Vanilla selected MLA currently supports only DSA")
                kv_idx, kv_off = self.sparse_kv_predict(q, k, metadata,
                                                        forward_args)
                at_idx, at_off = self.sparse_attn_predict(
                    q, k, metadata, forward_args)
                forward_args.sparse_prediction = replace(
                    forward_args.sparse_prediction,
                    sparse_kv_indices=kv_idx,
                    sparse_kv_offsets=kv_off,
                    sparse_attn_indices=at_idx,
                    sparse_attn_offsets=at_off,
                    sparse_attn_indices_block_size=getattr(
                        self.sparse_params, "indices_block_size"),
                )
            sparse_attn_indices = (
                forward_args.sparse_prediction.sparse_attn_indices)
            if sparse_attn_indices is not None:
                if k is not None or v is not None:
                    raise ValueError(
                        "Vanilla sparse MLA expects absorbed queries and latent cache, "
                        "not explicit K/V tensors")
                return self._mla_forward_sparse(
                    q,
                    metadata,
                    forward_args.latent_cache,
                    forward_args.q_pe,
                    sparse_attn_indices,
                    forward_args.attention_input_type,
                )
            if self.sparse_params is not None:
                raise ValueError(
                    "Vanilla sparse MLA requires sparse attention indices")
            if forward_args.attention_input_type == AttentionInputType.context_only:
                assert k is not None and v is not None
                return self._mla_forward_context(q, k, v, metadata,
                                                 forward_args.latent_cache)
            elif forward_args.attention_input_type == AttentionInputType.generation_only:
                assert k is None and v is None
                return self._mla_forward_generation(q, metadata,
                                                    forward_args.latent_cache)
            else:
                raise ValueError(
                    f"Unsupported attention input type: {forward_args.attention_input_type}"
                )

        if metadata.kv_cache_manager is None:
            # NOTE: WAR for no kv cache attn e.g. BERT,
            # try to separate the kv cache estimation path from no kv cache attn.
            num_heads = self.num_heads
            num_kv_heads = self.num_kv_heads
            return self.no_kv_cache_forward(
                q=q,
                k=k,
                v=v,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                metadata=metadata,
                attention_mask=forward_args.attention_mask)

        past_seen_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq
        cache_indices = [
            block_ids[0] for block_ids in metadata.block_ids_per_seq
        ]
        kv_cache_tensor = metadata.kv_cache_manager.get_buffers(
            self.layer_idx, kv_layout=metadata.kv_layout)

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
                single_q, single_k, single_v, forward_args.attention_mask,
                kv_cache_tensor, past_seen_token, cache_idx, sample_idx,
                metadata, forward_args.attention_window_size)

            attn_outputs.append(attn_output)

            offset += seq_len
            offset_kv += seq_len_kv

        attn_output = torch.cat(attn_outputs, dim=1)
        attn_output = attn_output.transpose(0, 1).contiguous()
        attn_output = attn_output.view(q_len, -1)

        return attn_output


class VanillaIndexer:
    """fp32 reference for the production DSA / DeepSeek-V4 sparse ``Indexer``.

    The production indexer (``sparse/dsa.py``) selects the top-k KV a query may
    attend to. ``VanillaIndexer`` mirrors that selection math in plain fp32 so it
    can serve as the indexer golden, analogous to how ``VanillaAttention`` is the
    attention golden.

    It is a standalone reference, deliberately **not** wired into
    ``VanillaAttention.forward``:

    * The indexer consumes index-space inputs (``qr`` / ``hidden_states`` / an
      index-space K cache) that ``forward(q, k, v, metadata)`` never receives;
      folding it into the forward path would widen the generic backend interface
      with algorithm-specific tensors.
    * A meaningful indexer golden must run against the *production indexer's own
      weights*, so this class **wraps a production ``Indexer`` instance** and
      reads its ``wq_b`` / ``weights_proj`` / ``softmax_scale`` / ... rather than
      owning independent (non-comparable) parameters.

    It owns the parts common to DSA and DeepSeek-V4 -- the index-space query
    projection, the per-head token weights, the logit scoring, and the top-k.
    Algorithm-specific K comes from a compressor / ``wk`` projection, so the
    caller supplies the reference K and this class handles the rest.

    Selection is discrete: top-k over near-tied logits, and an fp32 reference vs
    an fp8/fp4 kernel can pick different borderline tokens. Compare the selected
    index *set*, never exact attention outputs.
    """

    def __init__(self, indexer):
        self.indexer = indexer
        self.n_heads = indexer.n_heads
        self.head_dim = indexer.head_dim
        self.rope_dim = indexer.rope_dim
        self.softmax_scale = indexer.softmax_scale
        self.indexer_k_dtype = getattr(indexer, "indexer_k_dtype", None)

    @property
    def uses_fp4(self) -> bool:
        return self.indexer_k_dtype == "fp4"

    def project_query(
        self,
        qr: torch.Tensor,
        position_ids: torch.Tensor,
        freqs_cis: torch.Tensor,
        rope_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        fp4_prep: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Index-space query: ``wq_b`` GEMM + RoPE on the rope slice.

        ``rope_fn(slice, freqs)`` applies RoPE in place (the caller injects the
        algorithm's rotary helper). ``fp4_prep`` optionally applies the fp4
        indexer quant/dequant. Returns ``[num_tokens, n_heads, head_dim]``.
        """
        num_tokens = qr.shape[0]
        q = F.linear(qr, self.indexer.wq_b.weight)
        q = q.view(num_tokens, self.n_heads, self.head_dim).unsqueeze(0)
        rope_fn(q[..., -self.rope_dim:], freqs_cis[position_ids.long()])
        q = q.squeeze(0)
        if fp4_prep is not None:
            q = fp4_prep(q)
        return q

    def token_weights(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Per-head token weights: ``weights_proj`` GEMM scaled by ``n_heads**-0.5``."""
        weights = F.linear(hidden_states, self.indexer.weights_proj.weight)
        return weights.float() * (self.n_heads**-0.5)

    def scores(self, q_row: torch.Tensor, k: torch.Tensor,
               weights_row: torch.Tensor) -> torch.Tensor:
        """Per-KV index logits for one query token (fp32).

        ``q_row`` is ``[n_heads, head_dim]``, ``k`` is ``[num_kv, head_dim]``,
        ``weights_row`` is ``[n_heads]``. Mirrors the production indexer: per-head
        ReLU(q·k) scaled by ``softmax_scale``, combined with the per-head weights.
        Returns ``[num_kv]`` logits.
        """
        head_scores = torch.einsum("hd,kd->hk", q_row.float(), k.float())
        head_scores = F.relu(head_scores) * self.softmax_scale
        return (head_scores * weights_row.float().unsqueeze(-1)).sum(dim=0)

    def topk_from_scores(self, scores: torch.Tensor,
                         topk_tokens: int) -> torch.Tensor:
        """Top-k KV positions for one query token, ``-1``-padded to ``topk_tokens``."""
        row = torch.full((topk_tokens, ),
                         -1,
                         dtype=torch.int32,
                         device=scores.device)
        if scores.numel() == 0:
            return row
        k = min(topk_tokens, scores.numel())
        row[:k] = torch.topk(scores.float(), k, dim=-1).indices.to(torch.int32)
        return row
