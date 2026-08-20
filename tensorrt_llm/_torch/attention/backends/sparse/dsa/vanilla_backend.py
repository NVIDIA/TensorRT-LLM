# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vanilla correctness backend for DeepSeek Sparse Attention."""

import math
import os
from dataclasses import replace
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
    merge_attention_forward_args,
)
from tensorrt_llm._torch.attention.backends.vanilla import VanillaAttention
from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.runtime.kv_cache_manager_v2._common import BAD_PAGE_INDEX

from ..inline_scale_kv import DIM_NOPE as _INLINE_SCALE_NOPE_DIM
from ..inline_scale_kv import DIM_ROPE as _INLINE_SCALE_ROPE_DIM
from ..inline_scale_kv import QUANT_TILE as _INLINE_SCALE_QUANT_TILE
from ..inline_scale_kv import TOKEN_BYTES as _INLINE_SCALE_TOKEN_BYTES
from .indexer import (
    _compute_slot_mappings,
    _effective_compress_ratio_divisor,
    _select_indexer_compress_ratio,
)
from .metadata import DSAtrtllmAttentionMetadata
from .params import DSAParams


class _TorchRotaryEmbedding(RotaryEmbedding):
    """Pure-torch multi-target RoPE that bypasses fused dispatch."""

    def forward(
        self, position_ids: torch.Tensor, targets: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        position_ids = position_ids.reshape(-1)
        num_tokens = position_ids.numel()
        if num_tokens == 0:
            return targets

        cos_sin = self.rotary_cos_sin[position_ids]
        cos, sin = cos_sin[:, 0, :], cos_sin[:, 1, :]

        def apply_rope(target: torch.Tensor) -> torch.Tensor:
            original_shape = target.shape
            if target.shape[0] != num_tokens:
                raise ValueError(
                    "Packed RoPE targets must have one leading row per position: "
                    f"got target shape {tuple(target.shape)} and {num_tokens} positions"
                )
            target = target.reshape(num_tokens, -1, self.head_dim)
            target = target.transpose(0, 1).unsqueeze(0)
            target = RotaryEmbedding.apply_rotary_pos_emb(
                target,
                cos.to(dtype=target.dtype).unsqueeze(0),
                sin.to(dtype=target.dtype).unsqueeze(0),
                is_neox=self.is_neox,
                inverse=self.inverse,
            )
            return target.squeeze(0).transpose(0, 1).reshape(original_shape)

        return [apply_rope(target) for target in targets]


def _cached_lens(metadata: DSAtrtllmAttentionMetadata) -> list[int]:
    """Return runtime cached lengths used by slot mappings and cache appends."""
    seq_lens = metadata.seq_lens.tolist()
    kv_lens = metadata.kv_lens_cuda[: metadata.num_seqs].tolist()
    return [int(kv_lens[i]) - seq_lens[i] for i in range(metadata.num_seqs)]


class DSAVanillaIndexer(nn.Module):
    """Standalone PyTorch golden for DSA projection, QDQ, scoring, and TopK."""

    # e4m3 and E2M1 maxima; the quantizers scale each block to fill the range.
    _FP8_MAX = 448.0
    _FP4_MAX = 6.0

    def __init__(
        self,
        quant_config: Optional[QuantConfig],
        pos_embd_params: Optional[PositionalEmbeddingParams],
        mla_params: Optional[MLAParams],
        skip_create_weights_in_init: bool,
        sparse_params: DSAParams,
        dtype: Optional[torch.dtype],
        compress_ratio: int = 1,
        layer_idx: int = 0,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ):
        """Mirror :class:`Indexer`'s signature and checkpoint names."""
        super().__init__()
        del aux_stream
        self.hidden_size = mla_params.hidden_size
        self.q_lora_rank = mla_params.q_lora_rank
        self.rope_dim = mla_params.qk_rope_head_dim
        self.n_heads = sparse_params.index_n_heads
        self.head_dim = sparse_params.index_head_dim
        self.index_topk = sparse_params.index_topk
        self.layer_idx = layer_idx
        self.compress_ratio = compress_ratio
        self.use_fp4 = sparse_params.indexer_k_dtype == "fp4"
        self.mtp_index_share = sparse_params.mtp_index_share
        self._indexer_bf16 = os.environ.get("TRTLLM_DSA_INDEXER_BF16", "0") == "1"
        wk_wp_dtype = dtype if self._indexer_bf16 else torch.float32

        self.wq_b = Linear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights_in_init,
        )
        self.wk = Linear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            dtype=wk_wp_dtype,
            quant_config=None,
            skip_create_weights_in_init=skip_create_weights_in_init,
        )
        self.k_norm = LayerNorm(hidden_size=self.head_dim, eps=1e-6)
        self.weights_proj = Linear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            dtype=wk_wp_dtype,
            quant_config=None,
            skip_create_weights_in_init=skip_create_weights_in_init,
        )
        self.rotary_emb = _TorchRotaryEmbedding(
            pos_embd_params.rope,
            head_dim=self.rope_dim,
            is_neox=not sparse_params.indexer_rope_interleave,
        )

        self.softmax_scale = self.head_dim**-0.5
        # Folded into ``weights`` so the scoring loop is a plain weighted sum,
        # matching what the kernels consume.
        self.weight_scale_factor = self.softmax_scale * self.n_heads**-0.5
        self._fused_wk_wp_weight: Optional[torch.Tensor] = None

    def post_load_weights(self) -> None:
        """Cache the projection layout consumed by :meth:`pre_indexer_proj`."""
        self.cache_derived_state()

    def cache_derived_state(self) -> None:
        """Match the production indexer's single PyTorch projection GEMM."""
        self._fused_wk_wp_weight = torch.cat(
            [self.wk.weight.data, self.weights_proj.weight.data], dim=0
        )

    def maybe_join_prev_topk_copy(self) -> None:
        """No aux-stream work to join; the reference never forks one."""

    @staticmethod
    def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
        """Round scales up to UE8M0 without perturbing exact powers of two."""
        bits = x.abs().float().view(torch.int32)
        exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
        return (exp.clamp(1, 254) << 23).view(torch.float32)

    @staticmethod
    def pack_ue8m0_to_int(x: torch.Tensor) -> torch.Tensor:
        """Pack four UE8M0 exponents into one int32, as the FP4 cache stores them."""
        assert x.dtype == torch.float32 and x.size(-1) % 4 == 0
        assert (x.view(torch.int32) & ((1 << 23) - 1) == 0).all()
        return (x.view(torch.int32) >> 23).to(torch.uint8).view(torch.int32)

    @staticmethod
    def unpack_ue8m0_from_int(packed_sf: torch.Tensor) -> torch.Tensor:
        return (packed_sf.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)

    @classmethod
    def quantize_fp8(
        cls, x: torch.Tensor, dims: Tuple[int, ...] = (0,), use_ue8m0: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize to e4m3 with one scale per ``dims`` slice (KV uses dim 0)."""
        excluded = tuple(i for i in range(x.dim()) if i not in set(dims))
        # Keep this floor aligned with fusedCatFp8.cu.  A larger floor changes
        # both the scale and the FP8 codes for otherwise valid tiny rows.
        amax = x.abs().float().amax(dim=excluded, keepdim=True).clamp(1e-12)
        sf = amax / cls._FP8_MAX
        if use_ue8m0:
            sf = cls.ceil_to_ue8m0(sf)
        return (x * (1.0 / sf)).to(torch.float8_e4m3fn), sf.squeeze()

    @classmethod
    def quantize_fp4(
        cls,
        x: torch.Tensor,
        use_ue8m0: bool = True,
        gran_k: int = 128,
        use_packed_ue8m0: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize to packed E2M1 nibbles with one scale per ``gran_k`` block."""
        m, n = x.shape
        assert n % 2 == 0
        assert use_ue8m0 or not use_packed_ue8m0
        padded_n = (n + gran_k - 1) // gran_k * gran_k
        x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
        x_padded[:, :n] = x
        x_view = x_padded.view(m, -1, gran_k)
        # fusedCatFp4.cu uses the same floor before encoding the UE8M0 scale.
        sf = x_view.abs().float().amax(dim=2).clamp_min(1e-12) / cls._FP4_MAX
        if use_ue8m0:
            sf = cls.ceil_to_ue8m0(sf)
        codes = cls._to_e2m1(x_view * (1.0 / sf.unsqueeze(2))).view(m, padded_n)
        pairs = codes.view(m, padded_n // 2, 2)
        packed = (pairs[:, :, 0] & 0x0F) | ((pairs[:, :, 1] & 0x0F) << 4)
        return (
            packed[:, : n // 2].contiguous(),
            cls.pack_ue8m0_to_int(sf) if use_packed_ue8m0 else sf,
        )

    @classmethod
    def dequantize_fp4(
        cls,
        packed: torch.Tensor,
        sf: torch.Tensor,
        gran_k: int = 128,
        use_packed_ue8m0: bool = False,
    ) -> torch.Tensor:
        m, packed_n = packed.shape
        n = packed_n * 2
        if use_packed_ue8m0:
            sf = cls.unpack_ue8m0_from_int(sf)
        codes = torch.zeros((m, n), dtype=torch.int8, device=packed.device)
        codes[:, ::2] = packed & 0x0F
        codes[:, 1::2] = (packed >> 4) & 0x0F
        group = torch.arange(n, device=packed.device) // gran_k
        return cls._from_e2m1(codes) * sf[:, group]

    @staticmethod
    def _uninterleave_block_scales(interleaved: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        """Reverse CUTLASS's 128x4 scale layout with torch indexing."""
        padded_rows = (rows + 127) // 128 * 128
        padded_cols = (cols + 3) // 4 * 4
        row = torch.arange(padded_rows, device=interleaved.device).unsqueeze(1)
        col = torch.arange(padded_cols, device=interleaved.device).unsqueeze(0)
        num_k_tiles = padded_cols // 4
        offsets = (
            (row // 128) * num_k_tiles * 512
            + (col // 4) * 512
            + (row % 32) * 16
            + ((row % 128) // 32) * 4
            + col % 4
        )
        flat = interleaved.contiguous().view(torch.uint8).reshape(-1)
        return flat[offsets][:rows, :cols]

    @staticmethod
    def _qdq_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        scaled = (x.float() / scale.float()).clamp(-448.0, 448.0)
        return scaled.to(torch.float8_e4m3fn).to(x.dtype) * scale.to(x.dtype)

    @classmethod
    def _qdq_fp8_blocks(cls, x: torch.Tensor, block_size: int = 128) -> torch.Tensor:
        rows = x.reshape(-1, x.shape[-1])
        if rows.shape[1] % block_size != 0:
            raise ValueError(
                f"FP8 block quantization requires K divisible by {block_size}, got {rows.shape[1]}"
            )
        blocks = rows.view(rows.shape[0], -1, block_size)
        # fp8_quantize_1x128 uses this floor before encoding the UE8M0 scale.
        amax = blocks.abs().float().amax(dim=-1).clamp_min(1e-10)
        scale = cls.ceil_to_ue8m0(amax / cls._FP8_MAX)
        qdq = cls._qdq_fp8(blocks, scale.unsqueeze(-1))
        return qdq.reshape_as(x)

    @classmethod
    def _round_to_e4m3_rne_satfinite(cls, x: torch.Tensor) -> torch.Tensor:
        """Round to finite E4M3 with the CUDA conversion's RNE semantics."""
        # Positive E4M3FN codes are monotonic. Code 127 is NaN, so stop at
        # code 126 (448) to reproduce ``satfinite`` for out-of-range values.
        codes = torch.arange(127, device=x.device, dtype=torch.int32)
        exponent = codes >> 3
        mantissa = codes & 0x07
        levels = torch.where(
            exponent == 0,
            mantissa.float() * (2.0**-9),
            (1.0 + mantissa.float() / 8.0) * torch.exp2(exponent.float() - 7.0),
        )

        ax = x.abs().float().clamp_max(cls._FP8_MAX)
        upper = torch.searchsorted(levels, ax).clamp_max(levels.numel() - 1)
        lower = (upper - 1).clamp_min(0)
        lower_distance = ax - levels[lower]
        upper_distance = levels[upper] - ax
        take_upper = (upper_distance < lower_distance) | (
            (upper_distance == lower_distance) & ((upper & 1) == 0)
        )
        rounded = levels[torch.where(take_upper, upper, lower)]
        return torch.where(torch.signbit(x), -rounded, rounded)

    @classmethod
    def _qdq_nvfp4(cls, x: torch.Tensor, scale_2: torch.Tensor) -> torch.Tensor:
        block_size = 16
        rows = x.reshape(-1, x.shape[-1])
        if rows.shape[1] % block_size != 0:
            raise ValueError(
                f"NVFP4 quantization requires K divisible by {block_size}, got {rows.shape[1]}"
            )
        blocks = rows.view(rows.shape[0], -1, block_size)
        block_scale = blocks.abs().float().amax(dim=-1) / cls._FP4_MAX
        quantized_scale = block_scale / scale_2.float()
        quantized_scale = cls._round_to_e4m3_rne_satfinite(quantized_scale)
        real_scale = quantized_scale.float() * scale_2.float()
        scaled = torch.where(
            real_scale.unsqueeze(-1) == 0,
            torch.zeros_like(blocks, dtype=torch.float32),
            blocks.float() / real_scale.unsqueeze(-1),
        )
        codes = cls._to_e2m1_rne(scaled)
        qdq = cls._from_e2m1(codes) * real_scale.unsqueeze(-1)
        return qdq.reshape_as(x).to(x.dtype)

    @classmethod
    def _wq_projection_reference(cls, qr: torch.Tensor, linear: Linear) -> torch.Tensor:
        """Apply indexer-Q Linear quantization using only torch tensor operations."""
        weight = linear.weight
        out_features = linear.out_features
        in_features = linear.in_features
        output_dtype = getattr(linear, "dtype", None)
        if output_dtype is None:
            output_dtype = torch.bfloat16 if qr.dtype == torch.float8_e4m3fn else qr.dtype
        expected_shape = (out_features, in_features)
        floating_dtypes = {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }
        if weight.shape == expected_shape and weight.dtype in floating_dtypes:
            return F.linear(qr, weight)

        if weight.shape == expected_shape and weight.dtype == torch.float8_e4m3fn:
            weight_scale = linear.weight_scale
            if weight_scale.ndim <= 1 and weight_scale.numel() == 1:
                dequant_weight = weight.float() * weight_scale.float()
                if qr.dtype == torch.float8_e4m3fn:
                    input_scale = linear.input_scale.float()
                    dequant_input = qr.float() * input_scale
                else:
                    input_scale = linear.input_scale
                    if input_scale is None or linear.force_dynamic_quantization:
                        input_scale = qr.abs().float().amax().clamp_min(1e-12) / cls._FP8_MAX
                    dequant_input = cls._qdq_fp8(qr, input_scale).float()
            elif weight_scale.ndim == 1:
                dequant_weight = weight.float() * weight_scale[:out_features].float().unsqueeze(1)
                if qr.dtype == torch.float8_e4m3fn:
                    dequant_input = qr.float()
                else:
                    input_scale = (
                        qr.abs().float().amax(dim=-1, keepdim=True).clamp_min(1e-12) / cls._FP8_MAX
                    )
                    dequant_input = cls._qdq_fp8(qr, input_scale).float()
            else:
                if weight_scale.dtype == torch.int32:
                    from tensorrt_llm.quantization.utils.fp8_utils import inverse_transform_sf

                    weight_scale = inverse_transform_sf(weight_scale, out_features, in_features)
                expected_scale_shape = (
                    (out_features + 127) // 128,
                    (in_features + 127) // 128,
                )
                if weight_scale.shape != expected_scale_shape:
                    raise ValueError(
                        "Unexpected FP8 block-scale shape for DSA indexer-Q reference: "
                        f"{tuple(weight_scale.shape)} != {expected_scale_shape}"
                    )
                expanded_scale = weight_scale.float().repeat_interleave(128, dim=0)
                expanded_scale = expanded_scale.repeat_interleave(128, dim=1)
                dequant_weight = weight.float() * expanded_scale[:out_features, :in_features]
                if qr.dtype == torch.float8_e4m3fn:
                    dequant_input = qr.float() * linear.input_scale.float()
                    dequant_input = cls._qdq_fp8_blocks(dequant_input).float()
                else:
                    dequant_input = cls._qdq_fp8_blocks(qr).float()
            return F.linear(
                dequant_input.to(output_dtype),
                dequant_weight.to(output_dtype),
            )

        expected_packed_shape = (out_features, in_features // 2)
        if weight.shape == expected_packed_shape and hasattr(linear, "scaling_vector_size"):
            block_size = linear.scaling_vector_size
            if block_size != 16:
                raise NotImplementedError(
                    f"DSA Vanilla indexer-Q supports NVFP4 block size 16, got {block_size}"
                )
            scale_cols = in_features // block_size
            scale_bytes = cls._uninterleave_block_scales(
                linear.weight_scale, out_features, scale_cols
            )
            block_scale = scale_bytes.view(torch.float8_e4m3fn).float()
            block_scale = block_scale * linear.weight_scale_2.float()
            dequant_weight = cls.dequantize_fp4(
                weight.contiguous().view(torch.uint8), block_scale, gran_k=block_size
            )

            dequant_input = qr
            if linear.pre_quant_scale is not None:
                dequant_input = dequant_input * linear.pre_quant_scale
            if getattr(linear.quant_method, "quantizes_nvfp4_activations", False):
                if linear.input_scale is None or linear.force_dynamic_quantization:
                    amax = dequant_input.abs().float().amax().clamp_min(1e-12)
                    input_scale_2 = amax / (cls._FP8_MAX * cls._FP4_MAX)
                else:
                    input_scale_2 = linear.input_scale.float().reciprocal()
                dequant_input = cls._qdq_nvfp4(dequant_input, input_scale_2)
            return F.linear(
                dequant_input.to(output_dtype),
                dequant_weight.to(output_dtype),
            )

        raise NotImplementedError(
            "DSAVanillaIndexer cannot express the configured wq_b weight layout "
            f"with torch operations: shape={tuple(weight.shape)}, dtype={weight.dtype}"
        )

    @staticmethod
    def _to_e2m1(x: torch.Tensor) -> torch.Tensor:
        """Round-to-nearest onto the E2M1 grid {0, .5, 1, 1.5, 2, 3, 4, 6}."""
        ax = x.abs().clamp_max(6.0)
        midpoints = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device, dtype=ax.dtype
        )
        idx = torch.bucketize(ax, midpoints)
        code = idx.to(torch.uint8) | (((x < 0) & (idx != 0)).to(torch.uint8) << 3)
        return code.view(torch.int8)

    @staticmethod
    def _to_e2m1_rne(x: torch.Tensor) -> torch.Tensor:
        """Convert to E2M1 with round-to-nearest-even and finite saturation."""
        ax = x.abs().clamp_max(6.0)
        midpoints = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device, dtype=ax.dtype
        )
        lower = torch.bucketize(ax, midpoints)
        upper = torch.bucketize(ax, midpoints, right=True)
        # At an exact midpoint the two bucket results differ by one. Choose
        # the even E2M1 code, matching ``cvt.rn.satfinite.e2m1x2.f32``.
        idx = torch.where((upper != lower) & ((upper & 1) == 0), upper, lower)
        code = idx.to(torch.uint8) | (((x < 0) & (idx != 0)).to(torch.uint8) << 3)
        return code.view(torch.int8)

    @staticmethod
    def _from_e2m1(code: torch.Tensor) -> torch.Tensor:
        values = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=code.device, dtype=torch.float32
        )
        sign, value_idx = (code & 0x08) != 0, (code & 0x07).to(torch.int32)
        value = values[value_idx]
        return torch.where(sign & (value_idx != 0), -value, value)

    @staticmethod
    def _weighted_relu_scores(
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
        epi_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Compute weighted ``relu(q @ k)`` scores for one Q/K tile."""
        scores = torch.matmul(q.float().transpose(0, 1), k.float().T)
        scores = torch.where(mask.unsqueeze(0), scores, scores.new_zeros(())).relu()
        scores = scores.to(epi_dtype)
        return (weights.to(epi_dtype).T.unsqueeze(-1) * scores).sum(dim=0)

    @classmethod
    def mqa_logits(
        cls,
        q: torch.Tensor,
        kv: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
    ) -> torch.Tensor:
        """Reference DeepGEMM ragged-prefill logits with ``-inf`` padding."""
        seq_len_kv = kv.shape[0]
        positions = torch.arange(seq_len_kv, device=kv.device)
        mask = (positions[None, :] >= cu_seqlen_ks[:, None]) & (
            positions[None, :] < cu_seqlen_ke[:, None]
        )
        logits = cls._weighted_relu_scores(q, kv, weights, mask)
        return logits.masked_fill(~mask, float("-inf"))

    @classmethod
    def paged_mqa_logits(
        cls,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        num_tokens: torch.Tensor,
        num_kv_tokens: torch.Tensor,
        block_tables: torch.Tensor,
        max_model_len: int,
        compress_ratio: int = 1,
    ) -> torch.Tensor:
        """Reference paged-decode logits from a dequantized KV cache."""
        batch_size, next_n = q.shape[0], q.shape[1]
        block_size = kv_cache.shape[1]
        logits = torch.full(
            (batch_size * next_n, max_model_len),
            float("-inf"),
            device=q.device,
            dtype=torch.float32,
        )
        num_tokens_list = num_tokens.tolist()
        num_kv_tokens_list = num_kv_tokens.tolist()

        for i in range(batch_size):
            num_token, num_kv_token = num_tokens_list[i], num_kv_tokens_list[i]
            q_offsets = torch.arange(num_token - next_n, num_token, device=q.device)
            row = slice(i * next_n, (i + 1) * next_n)
            for block_rk in range((num_kv_token + block_size - 1) // block_size):
                block_start = block_rk * block_size
                block_end = min(block_start + block_size, max_model_len)
                block_width = block_end - block_start
                block = kv_cache[block_tables[i][block_rk]][:block_width]
                k_offsets = torch.arange(block_start, block_end, device=q.device)
                causal_mask = k_offsets[None, :] < (q_offsets[:, None] + 1) // compress_ratio
                mask = (k_offsets[None, :] < num_kv_token) & causal_mask
                scores = cls._weighted_relu_scores(
                    q[i], block.view(block_width, -1), weights[row], mask
                )
                logits[row, block_start:block_end] = torch.where(causal_mask, scores, float("-inf"))
        return logits

    @classmethod
    def paged_mqa_logits_quantized(
        cls,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_scales: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        max_model_len: int,
        block_kv: int,
        epi_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Reference paged-decode logits with per-token KV scales."""
        batch, next_n = q.shape[0], q.shape[1]
        logits = torch.full(
            (batch * next_n, max_model_len), float("-inf"), device=q.device, dtype=epi_dtype
        )
        for b in range(batch):
            ctx_len = int(context_lens[b].item())
            q_positions = torch.arange(ctx_len - next_n, ctx_len, device=q.device)
            row = slice(b * next_n, (b + 1) * next_n)
            for blk_idx in range((ctx_len + block_kv - 1) // block_kv):
                phys = int(block_table[b, blk_idx].item())
                block_start = blk_idx * block_kv
                block_end = min(block_start + block_kv, max_model_len)
                block_width = block_end - block_start
                block = kv[phys][:block_width]
                scale = kv_scales[phys, :block_width].to(epi_dtype)
                k_positions = torch.arange(block_start, block_end, device=q.device)
                mask = (k_positions[None, :] < ctx_len) & (
                    k_positions[None, :] <= q_positions[:, None]
                )
                scores = cls._weighted_relu_scores(q[b], block, weights[row], mask, epi_dtype)
                scores = scores * scale.unsqueeze(0)
                logits[row, block_start:block_end] = torch.where(
                    mask, scores, torch.tensor(float("-inf"), device=q.device, dtype=epi_dtype)
                )
        return logits

    @staticmethod
    def select_top_k(logits: torch.Tensor, topk: int) -> torch.Tensor:
        """Top-k per row as int32, ``-1`` in slots with no valid key."""
        num_selected = min(topk, logits.shape[-1])
        values, indices = logits.topk(num_selected, dim=-1)
        indices = indices.to(torch.int32).masked_fill(torch.isneginf(values), -1)
        if num_selected == topk:
            return indices
        padding = indices.new_full((indices.shape[0], topk - num_selected), -1)
        return torch.cat([indices, padding], dim=-1)

    def _gather_keys(
        self, metadata: DSAtrtllmAttentionMetadata, seq_idx: int, kv_len: int
    ) -> torch.Tensor:
        """Gather dequantized indexer keys for one request from paged storage."""
        # Take every geometry input from the cache manager, exactly as the
        # write side does in DSAtrtllmAttentionMetadata.prepare(); reading them
        # from anywhere else lets the two drift apart silently.
        manager = metadata.kv_cache_manager
        head_dim = manager.index_head_dim
        data_bytes_per_token = head_dim // 2 if getattr(manager, "use_fp4", False) else head_dim
        cache = manager.get_indexer_k_cache_buffers(self.layer_idx)
        positions = torch.arange(kv_len, dtype=torch.int64)
        data_idx, scale_idx = _compute_slot_mappings(
            positions,
            metadata.host_indexer_k_cache_block_offsets,
            torch.full((kv_len,), seq_idx, dtype=torch.int64),
            head_dim,
            metadata._tokens_per_block,
            manager.quant_block_size,
            data_bytes_per_token=data_bytes_per_token,
        )

        flat = cache.reshape(-1)
        device = flat.device
        data_offsets = torch.arange(data_bytes_per_token, dtype=torch.int64, device=device)
        scale_offsets = torch.arange(4, dtype=torch.int64, device=device)
        scale = flat[scale_idx.to(device).unsqueeze(1) + scale_offsets]
        k = flat[data_idx.to(device).unsqueeze(1) + data_offsets]
        if self.use_fp4:
            return self.dequantize_fp4(
                k.view(kv_len, data_bytes_per_token),
                scale.view(torch.int32).view(kv_len, 1),
                # The cache uses one 4-byte scale word per token, but that
                # word packs four block-32 UE8M0 exponents.
                gran_k=32,
                use_packed_ue8m0=True,
            )

        k = k.view(torch.float8_e4m3fn).view(kv_len, head_dim).float()
        scale = scale.view(torch.float32).view(kv_len, 1)
        return k * scale

    @staticmethod
    def _copy_dense_topk(
        metadata: DSAtrtllmAttentionMetadata,
        output: torch.Tensor,
        source_start: int,
        target_start: int,
        num_tokens: int,
    ) -> None:
        """Copy metadata's precomputed dense selection for a skipped phase."""
        if metadata.topk_indices_buffer is None:
            raise ValueError("Dense indexer skip requires metadata.topk_indices_buffer")
        output[target_start : target_start + num_tokens].copy_(
            metadata.topk_indices_buffer[source_start : source_start + num_tokens]
        )

    @staticmethod
    def _mtp_last_accepted_rows(
        gen_topk: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        num_contexts: int,
        num_generations: int,
        next_n: int,
    ) -> torch.Tensor:
        """Match :meth:`Indexer._mtp_last_accepted_rows` in plain torch."""
        num_accepted = metadata.mtp_num_accepted
        if num_accepted is None:
            return gen_topk[next_n - 1 :: next_n]
        gen_num_accepted = num_accepted[num_contexts : num_contexts + num_generations]
        base = torch.arange(num_generations, device=gen_topk.device, dtype=torch.long) * next_n
        offset = (gen_num_accepted - 1).clamp(0, next_n - 1)
        return gen_topk[base + offset]

    def sparse_attn_indexer(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        is_generation: Optional[bool] = None,
    ) -> torch.Tensor:
        """Score paged keys and select request-local TopK indices in PyTorch."""
        del k_fp8, k_scale
        compress_ratio = _effective_compress_ratio_divisor(
            _select_indexer_compress_ratio(metadata.compress_ratios)
        )

        if self.use_fp4:
            if q_scale is None:
                raise ValueError("FP4 indexer scoring requires q_scale")
            q_fp8 = self.dequantize_fp4(
                q_fp8.reshape(-1, self.head_dim // 2),
                q_scale.reshape(-1, 1),
                gran_k=32,
                use_packed_ue8m0=True,
            ).view(-1, self.n_heads, self.head_dim)

        num_contexts = metadata.num_contexts
        if is_generation is None:
            seq_start, seq_end = 0, metadata.num_seqs
            cache_name = "indexer_topk_out_buffer"
        elif is_generation:
            seq_start, seq_end = num_contexts, metadata.num_seqs
            cache_name = "indexer_topk_out_buffer_gen"
        else:
            seq_start, seq_end = 0, num_contexts
            cache_name = "indexer_topk_out_buffer_ctx"

        topk_indices_buffer = metadata.get_empty(
            metadata.cuda_graph_buffers,
            (hidden_states.shape[0], self.index_topk),
            cache_name=cache_name,
            dtype=torch.int32,
            capture_graph=metadata.is_cuda_graph,
        )
        # The buffer can be longer than this phase's token count (graph padding);
        # the scoring loop only writes real rows, so mark the rest invalid.
        topk_indices_buffer.fill_(-1)

        seq_lens = metadata.seq_lens.tolist()
        past_lens = _cached_lens(metadata)
        num_ctx_tokens = metadata.num_ctx_tokens
        num_gen_tokens = metadata.num_tokens - num_ctx_tokens
        target_offset = 0 if is_generation is not None else num_ctx_tokens

        if is_generation is not True and metadata.skip_indexer_for_ctx_reqs:
            self._copy_dense_topk(metadata, topk_indices_buffer, 0, 0, num_ctx_tokens)

        reuse_topk = (
            self.mtp_index_share
            and metadata.in_mtp_draft_loop
            and metadata.indexer_skip_topk
            and metadata.shared_topk_indices is not None
        )
        if is_generation is not False and metadata.skip_indexer_for_gen_reqs:
            self._copy_dense_topk(
                metadata,
                topk_indices_buffer,
                num_ctx_tokens,
                target_offset,
                num_gen_tokens,
            )
        elif is_generation is not False and reuse_topk:
            topk_indices_buffer[target_offset : target_offset + num_gen_tokens].copy_(
                metadata.shared_topk_indices[: metadata.num_generations]
            )

        token = 0
        for seq_idx in range(seq_start, seq_end):
            q_len = seq_lens[seq_idx]
            skip_phase = (seq_idx < num_contexts and metadata.skip_indexer_for_ctx_reqs) or (
                seq_idx >= num_contexts and (metadata.skip_indexer_for_gen_reqs or reuse_topk)
            )
            if not skip_phase:
                past = past_lens[seq_idx]
                kv_len = (past + q_len) // compress_ratio
                token_slice = slice(token, token + q_len)
                # [q_len, n_heads, head_dim] x [kv_len, head_dim]. The FP8
                # codes are scored as-is: pre_indexer_proj folded the per-head
                # q scale into ``weights`` together with the attention scale.
                head_logits = torch.einsum(
                    "thd,kd->thk",
                    q_fp8[token_slice].float(),
                    self._gather_keys(metadata, seq_idx, kv_len),
                ).relu()
                logits = torch.einsum("thk,th->tk", head_logits, weights[token_slice].float())

                # Query token i sees compressed keys before
                # floor((past + i + 1) / compress_ratio). TopK remains in the
                # compressed cache's local coordinate system, as in production.
                keys = torch.arange(kv_len, device=logits.device)
                limits = (
                    torch.arange(past, past + q_len, device=logits.device) + 1
                ) // compress_ratio
                logits = logits.masked_fill(keys.unsqueeze(0) >= limits.unsqueeze(1), -float("inf"))
                topk_indices_buffer[token : token + q_len] = self.select_top_k(
                    logits, self.index_topk
                )
            token += q_len

        if self.mtp_index_share and metadata.in_mtp_draft_loop and not reuse_topk:
            rows = None
            if is_generation is not False and metadata.num_generations > 0:
                next_n = num_gen_tokens // metadata.num_generations
                gen_topk = topk_indices_buffer[target_offset : target_offset + num_gen_tokens]
                rows = self._mtp_last_accepted_rows(
                    gen_topk,
                    metadata,
                    num_contexts,
                    metadata.num_generations,
                    next_n,
                )
            if is_generation is not True and num_contexts > 0:
                ctx_last = (
                    torch.cumsum(metadata.seq_lens_cuda[:num_contexts].to(torch.long), dim=0) - 1
                )
                ctx_rows = topk_indices_buffer[ctx_last]
                rows = ctx_rows if rows is None else torch.cat([ctx_rows, rows])
            if rows is not None:
                if metadata.shared_topk_indices is None:
                    metadata.shared_topk_indices = rows.contiguous()
                else:
                    row_start = num_contexts if is_generation else 0
                    metadata.shared_topk_indices[
                        row_start : row_start + rows.shape[0], : rows.shape[1]
                    ].copy_(rows)

        return topk_indices_buffer

    def pre_indexer_proj(
        self, qr: torch.Tensor, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project, apply RoPE, and QDQ without dispatching fused indexer ops."""
        if isinstance(hidden_states, Fp4QuantizedTensor):
            hidden_states = hidden_states.unquantized_hidden_states

        if self._fused_wk_wp_weight is None:
            raise RuntimeError("cache_derived_state() must be called before pre_indexer_proj()")
        if self._indexer_bf16:
            fused_out = F.linear(hidden_states, self._fused_wk_wp_weight)
        else:
            previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                fused_out = F.linear(hidden_states.float(), self._fused_wk_wp_weight)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32
        indexer_k, weights = fused_out.split([self.head_dim, self.n_heads], dim=-1)

        # Calling Linear.forward here could route through the same custom GEMM
        # as the backend under test. Dequantize and fake-quantize explicitly so
        # quantized checkpoints still have an independent torch golden.
        q = self._wq_projection_reference(qr, self.wq_b).view(-1, self.n_heads, self.head_dim)
        k = self.k_norm(indexer_k.to(hidden_states.dtype))
        q_pe, q_nope = q.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        k_pe, k_nope = k.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        q_pe, k_pe = self.rotary_emb(position_ids, [q_pe, k_pe.unsqueeze(1)])
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe[:, 0, :], k_nope], dim=-1)

        if self.use_fp4:
            q_fp8, q_scale = self.quantize_fp4(
                q.reshape(-1, self.head_dim),
                gran_k=32,
                use_packed_ue8m0=True,
            )
            k_fp8, k_scale = self.quantize_fp4(
                k,
                gran_k=32,
                use_packed_ue8m0=True,
            )
            q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim // 2)
            q_scale = q_scale.view(-1, self.n_heads, 1)
            k_scale = k_scale.view(-1, 1)
            # FP4 logits apply q_scale in the kernel epilogue.
            weights = weights.float() * self.weight_scale_factor
        else:
            # Q carries one scale per (token, head); K one per token.
            q_fp8, q_scale = self.quantize_fp8(q, dims=(0, 1), use_ue8m0=True)
            k_fp8, k_scale = self.quantize_fp8(k, dims=(0,), use_ue8m0=True)
            q_scale = q_scale.reshape(-1, self.n_heads, 1)
            k_scale = k_scale.reshape(-1, 1)
            # The FP8 kernels apply no q scale of their own, so fold it into
            # weights together with softmax_scale * n_heads ** -0.5.
            weights = weights.float() * q_scale.squeeze(-1) * self.weight_scale_factor
        return q_fp8, k_fp8, k_scale, weights, q_scale

    def _update_k_cache(
        self, k_fp8: torch.Tensor, k_scale: torch.Tensor, metadata: DSAtrtllmAttentionMetadata
    ) -> None:
        """Scatter new keys through metadata's paged-cache slot mappings."""
        if metadata.kv_cache_manager is None or getattr(metadata, "slot_mapping_fp8", None) is None:
            return
        cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(self.layer_idx)
        flat = cache.reshape(-1)
        device = flat.device
        num_tokens = k_fp8.shape[0]

        data_idx = metadata.slot_mapping_fp8[:num_tokens].to(device)
        data_bytes_per_token = self.head_dim // 2 if self.use_fp4 else self.head_dim
        data_offsets = torch.arange(data_bytes_per_token, dtype=torch.int64, device=device)
        flat[data_idx.unsqueeze(1) + data_offsets] = k_fp8.view(torch.uint8)

        scale_idx = metadata.slot_mapping_scale[:num_tokens].to(device)
        scale_offsets = torch.arange(4, dtype=torch.int64, device=device)
        if self.use_fp4:
            # FP4 already carries four packed UE8M0 exponent bytes per int32
            # scale word. A numeric cast to float32 would replace those bytes
            # with the IEEE representation of the integer value.
            if k_scale.element_size() == 1:
                k_scale = k_scale.view(torch.int32)
            scale_bytes = k_scale.contiguous().view(torch.uint8).view(num_tokens, 4)
        else:
            scale_bytes = k_scale.float().contiguous().view(torch.uint8).view(num_tokens, 4)
        flat[scale_idx.unsqueeze(1) + scale_offsets] = scale_bytes

    def forward_from_projected(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        hidden_states: torch.Tensor,
        indexer_intermediates: list,
        is_generation: Optional[bool] = None,
    ) -> torch.Tensor:
        """Slice whole-batch projections to one phase, then score."""
        if is_generation is None:
            phase_start, phase_end = 0, metadata.num_tokens
        elif is_generation:
            phase_start, phase_end = metadata.num_ctx_tokens, metadata.num_tokens
        else:
            phase_start, phase_end = 0, metadata.num_ctx_tokens

        q_fp8, k_fp8, k_scale, weights, q_scale = indexer_intermediates
        return self.sparse_attn_indexer(
            metadata,
            hidden_states,
            q_fp8[phase_start:phase_end],
            k_fp8,
            k_scale,
            weights[phase_start:phase_end],
            q_scale=q_scale[phase_start:phase_end] if q_scale is not None else None,
            is_generation=is_generation,
        )

    @torch.inference_mode()
    def forward(
        self,
        qr: torch.Tensor,
        hidden_states: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Project, append to the K cache and select, for the whole batch."""
        intermediates = list(self.pre_indexer_proj(qr, hidden_states, position_ids))
        self._update_k_cache(intermediates[1], intermediates[2], metadata)
        return self.forward_from_projected(metadata, hidden_states, intermediates)


class DSAVanillaAttention(VanillaAttention):
    """Standalone PyTorch golden for DSA index selection and sparse MLA."""

    Metadata = DSAtrtllmAttentionMetadata

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        q_scaling: Optional[float] = None,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        mla_params: Optional[MLAParams] = None,
        skip_create_weights_in_init: bool = False,
        attention_chunk_size: Optional[int] = None,
        sparse_params: Optional[DSAParams] = None,
        dtype: Optional[torch.dtype] = None,
        aux_stream: Optional[torch.cuda.Stream] = None,
        **kwargs,
    ):
        sparse_attention_config = kwargs.pop("sparse_attention_config", None)
        self.sparse_attention_config = sparse_attention_config
        if (
            sparse_params is None
            and sparse_attention_config is not None
            and hasattr(sparse_attention_config, "to_sparse_params")
        ):
            sparse_params = sparse_attention_config.to_sparse_params(layer_idx=layer_idx)
        if sparse_params is None:
            raise ValueError("sparse_params is required for DSAVanillaAttention and cannot be None")
        if mla_params is None:
            raise ValueError("DSAVanillaAttention requires MLA parameters")
        self.use_fp8_ds_mla = kwargs.get("kv_cache_dtype", "auto") == "fp8_ds_mla"
        super().__init__(
            layer_idx,
            num_heads,
            head_dim,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            q_scaling=q_scaling,
            sparse_params=sparse_params,
            pos_embd_params=pos_embd_params,
            mla_params=mla_params,
            attention_chunk_size=attention_chunk_size,
            **kwargs,
        )

        # DSA backends own their MLA RoPE (support_fused_rope is True), so the
        # module never rotates externally -- build the table here.
        self.rotary_emb = _TorchRotaryEmbedding(
            pos_embd_params.rope,
            head_dim=self.qk_rope_head_dim,
            is_neox=pos_embd_params.is_neox,
        )

        # Cross-layer indexer sharing mirrors DSATrtllmAttention: only "full"
        # layers own an indexer, shared layers reuse the previous full layer's
        # top-k from metadata.
        self.is_full_indexer_layer = getattr(sparse_params, "is_full_indexer_layer", True)
        if self.is_full_indexer_layer:
            self.indexer = DSAVanillaIndexer(
                quant_config,
                pos_embd_params,
                mla_params,
                skip_create_weights_in_init,
                sparse_params,
                dtype=dtype,
                layer_idx=layer_idx,
                aux_stream=aux_stream,
            )
        else:
            self.indexer = None

    @classmethod
    def support_fused_rope(cls) -> bool:
        """Keep Q/K RoPE ownership inside the DSA backend."""
        return True

    def _token_positions(
        self, metadata: DSAtrtllmAttentionMetadata, seq_start: int, seq_end: int, device
    ) -> torch.Tensor:
        """Absolute RoPE positions for one phase, on the shared cached-length basis."""
        seq_lens = metadata.seq_lens.tolist()
        past_lens = _cached_lens(metadata)
        pieces = []
        for seq_idx in range(seq_start, seq_end):
            past = past_lens[seq_idx]
            pieces.append(
                torch.arange(past, past + seq_lens[seq_idx], dtype=torch.int32, device=device)
            )
        return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.int32, device=device)

    def _apply_mla_rope(
        self,
        fused_q: Optional[torch.Tensor],
        q_pe: Optional[torch.Tensor],
        latent_cache: torch.Tensor,
        positions: torch.Tensor,
        *,
        apply_q: bool = True,
        apply_k: bool = True,
    ) -> None:
        """Rotate q_pe into fused_q's rope slot and latent_cache's k_pe in place."""
        num_tokens = latent_cache.shape[0]
        fused_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        targets = []
        if apply_q:
            if fused_q is None or q_pe is None:
                raise ValueError("Q-side MLA RoPE requires fused_q and q_pe")
            targets.append(q_pe.reshape(num_tokens, self.num_heads * self.qk_rope_head_dim))
        if apply_k:
            targets.append(latent_cache[..., self.kv_lora_rank :])

        rotated = iter(self.rotary_emb(positions, targets))
        if apply_q:
            q_pe_rot = next(rotated)
            fused_q.view(num_tokens, self.num_heads, fused_head_dim)[..., self.kv_lora_rank :] = (
                q_pe_rot.view(num_tokens, self.num_heads, self.qk_rope_head_dim)
            )
        if apply_k:
            latent_cache[..., self.kv_lora_rank :] = next(rotated)

    def mla_rope_generation(
        self,
        fused_q: Optional[torch.Tensor],
        q_pe: Optional[torch.Tensor],
        latent_cache: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        fmha_scheduler_counter: torch.Tensor,
        mla_bmm1_scale: torch.Tensor,
        mla_bmm2_scale: torch.Tensor,
        quant_q_buffer: torch.Tensor,
        out_scale: Optional[torch.Tensor] = None,
        kv_norm_weight: Optional[torch.Tensor] = None,
        kv_norm_eps: float = 1e-6,
        precomputed_cu_seqlens: bool = False,
        precomputed_fmha_scheduler: bool = False,
        kv_only: bool = False,
        kv_done_elsewhere: bool = False,
        quant_scale_qkv: Optional[torch.Tensor] = None,
    ) -> None:
        """Reproduce generation RoPE, cache append, and output-buffer mutations."""
        if kv_only and kv_done_elsewhere:
            raise ValueError("kv_only and kv_done_elsewhere are mutually exclusive")
        if kv_only and kv_norm_weight is None:
            raise ValueError("kv_only requires kv_norm_weight")
        if kv_only and not precomputed_cu_seqlens:
            raise ValueError("kv_only requires precomputed_cu_seqlens")
        if metadata.kv_cache_manager is None:
            raise ValueError("DSA Vanilla generation RoPE requires a KV cache manager")

        # The production generation kernel reads latent_cache through a const
        # pointer.  Rotate (and optionally normalize) a private cache-write
        # tensor so the fake-fused entry point has the same input side effects.
        cache_latent = latent_cache
        if not kv_done_elsewhere:
            cache_latent = latent_cache.clone()
            if kv_norm_weight is not None:
                latent_float = latent_cache.float()
                variance = latent_float.square().mean(dim=-1, keepdim=True)
                cache_latent.copy_(
                    (
                        latent_float * torch.rsqrt(variance + kv_norm_eps) * kv_norm_weight.float()
                    ).to(latent_cache.dtype)
                )

        seq_start, seq_end = metadata.num_contexts, metadata.num_seqs
        seq_lens = metadata.seq_lens.tolist()[seq_start:seq_end]
        past_lens = _cached_lens(metadata)[seq_start:seq_end]
        positions = self._token_positions(metadata, seq_start, seq_end, latent_cache.device)
        self._apply_mla_rope(
            fused_q,
            q_pe,
            cache_latent,
            positions,
            apply_q=not kv_only,
            apply_k=not kv_done_elsewhere,
        )

        if not kv_done_elsewhere:
            self._append_latent_cache(
                metadata,
                metadata.request_ids[seq_start:seq_end],
                seq_lens,
                past_lens,
                cache_latent,
            )

        if not precomputed_cu_seqlens:
            q_lens = torch.tensor(seq_lens, dtype=torch.int32, device=cu_q_seqlens.device)
            kv_lens = torch.tensor(
                [past + length for past, length in zip(past_lens, seq_lens, strict=True)],
                dtype=torch.int32,
                device=cu_kv_seqlens.device,
            )
            cu_q_seqlens[: len(seq_lens) + 1].zero_()
            cu_kv_seqlens[: len(seq_lens) + 1].zero_()
            cu_q_seqlens[1 : len(seq_lens) + 1] = torch.cumsum(q_lens, dim=0) * self.num_heads
            cu_kv_seqlens[1 : len(seq_lens) + 1] = torch.cumsum(kv_lens, dim=0)

        if not precomputed_fmha_scheduler:
            fmha_scheduler_counter.zero_()
            bmm1_scale = 1.0 / (
                math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim) * (self.q_scaling or 1.0)
            )
            if mla_bmm1_scale is not None and mla_bmm1_scale.numel() >= 2:
                mla_bmm1_scale[0] = bmm1_scale
                mla_bmm1_scale[1] = bmm1_scale * math.log2(math.e)
            if mla_bmm2_scale is not None and mla_bmm2_scale.numel() >= 1:
                mla_bmm2_scale[0] = 1.0 if out_scale is None else out_scale.flatten()[0]

        if (
            not kv_only
            and quant_q_buffer is not None
            and quant_q_buffer.numel() > 0
            and fused_q is not None
        ):
            fused_q_view = fused_q.view(
                latent_cache.shape[0],
                self.num_heads,
                self.kv_lora_rank + self.qk_rope_head_dim,
            )
            quant_q_view = quant_q_buffer.view(torch.float8_e4m3fn).view_as(fused_q_view)
            if quant_scale_qkv is None:
                quant_q_view.copy_(fused_q_view.to(torch.float8_e4m3fn))
            else:
                # q_nope was produced by the fused Q projection; only the RoPE
                # suffix remains for mla_rope_generation to quantize.
                scale = quant_scale_qkv.flatten()[0].float()
                quant_q_view[..., self.kv_lora_rank :].copy_(
                    (fused_q_view[..., self.kv_lora_rank :].float() * scale).to(torch.float8_e4m3fn)
                )

    def mla_rope_append_paged_kv_assign_q(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        is_generation: bool = False,
        **kwargs,
    ) -> None:
        """Fake-fused torch reference for the prefill RoPE/append entry point."""
        del kwargs
        if is_generation:
            seq_start, seq_end = metadata.num_contexts, metadata.num_seqs
        else:
            seq_start, seq_end = 0, metadata.num_contexts
        seq_lens = metadata.seq_lens.tolist()[seq_start:seq_end]
        past_lens = _cached_lens(metadata)[seq_start:seq_end]
        num_tokens = sum(seq_lens)
        q_view = q.view(
            num_tokens,
            self.num_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
        )
        q_pe = q_view[..., self.qk_nope_head_dim :]
        q_pe_rot, k_pe_rot = self.rotary_emb(
            self._token_positions(metadata, seq_start, seq_end, q.device),
            [q_pe, latent_cache[..., self.kv_lora_rank :]],
        )
        q_pe.copy_(q_pe_rot)
        latent_cache[..., self.kv_lora_rank :].copy_(k_pe_rot)

        self._append_latent_cache(
            metadata,
            metadata.request_ids[seq_start:seq_end],
            seq_lens,
            past_lens,
            latent_cache,
        )

    def _select_local_topk(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> torch.Tensor:
        """Select request-local KV positions for the torch attention core."""
        del k
        sparse_backend_args = forward_args.sparse_backend_args
        if sparse_backend_args is None:
            raise ValueError("DSA Vanilla attention requires sparse_backend_args")
        if sparse_backend_args.topk_indices is not None:
            return sparse_backend_args.topk_indices

        is_generation = forward_args.attention_input_type == AttentionInputType.generation_only
        phase_start = metadata.num_ctx_tokens if is_generation else 0
        phase_end = metadata.num_tokens if is_generation else metadata.num_ctx_tokens
        shared_topk_indices = metadata.shared_topk_indices
        if self.indexer is None:
            return shared_topk_indices[phase_start:phase_end]

        if not sparse_backend_args.indexer_intermediates:
            raise ValueError(
                "DSA Vanilla attention needs the indexer projections; run "
                "Indexer.pre_indexer_proj (and _update_k_cache) before the forward, "
                "or inject sparse_backend_args.topk_indices."
            )
        topk_indices = self.indexer.forward_from_projected(
            metadata,
            q,
            sparse_backend_args.indexer_intermediates,
            is_generation=is_generation,
        )
        preserve_mtp_topk = metadata.in_mtp_draft_loop and self.indexer.mtp_index_share
        if shared_topk_indices is not None and not preserve_mtp_topk:
            shared_topk_indices[
                phase_start : phase_start + topk_indices.shape[0],
                : topk_indices.shape[1],
            ].copy_(topk_indices)
        return topk_indices

    @staticmethod
    def _local_topk_to_global(
        topk_indices: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        layer_idx: int,
        is_generation: bool,
    ) -> torch.Tensor:
        """Lower request-local positions to the primary-pool coordinate space."""
        if topk_indices.dtype != torch.int32:
            raise ValueError(f"DSA top-k indices must have dtype int32, got {topk_indices.dtype}")
        metadata._ensure_pool_view_cached()
        manager = metadata.kv_cache_manager
        page_index_scale, layer_offset = manager.get_primary_pool_page_index_params(layer_idx)
        if is_generation:
            block_table = metadata._cached_block_table_gen
            req_idx = metadata._cached_req_idx_gen
        else:
            block_table = metadata._cached_block_table_ctx
            req_idx = metadata._cached_req_idx_ctx

        tokens_per_block = metadata._cached_tokens_per_block
        if block_table.shape[1] == 0:
            return torch.full_like(topk_indices, -1)
        safe_indices = topk_indices.clamp_min(0).to(torch.long)
        page_idx = safe_indices // tokens_per_block
        token_in_page = safe_indices % tokens_per_block
        valid = (topk_indices >= 0) & (page_idx < block_table.shape[1])
        page_idx = page_idx.clamp(max=block_table.shape[1] - 1)
        physical_page = block_table[req_idx.to(torch.long).unsqueeze(1), page_idx]
        stride = page_index_scale * tokens_per_block
        global_indices = (
            physical_page.to(torch.long) * stride + layer_offset * tokens_per_block + token_in_page
        )
        return torch.where(valid, global_indices, -1).to(torch.int32)

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run the torch indexer and return TRTLLM-compatible pool indices."""
        local_topk = self._select_local_topk(q, k, metadata, forward_args)
        is_generation = forward_args.attention_input_type == AttentionInputType.generation_only
        local_layer_idx = metadata.kv_cache_manager.layer_offsets[self.layer_idx]
        global_topk = self._local_topk_to_global(
            local_topk, metadata, local_layer_idx, is_generation
        )
        return global_topk, None

    def sparse_kv_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """No-op KV prediction; DSA uses indexer-based selection instead."""
        return None, None

    @staticmethod
    def _pack_inline_scale_latent(
        latent_cache: torch.Tensor, storage_dtype: torch.dtype
    ) -> torch.Tensor:
        """Pack ``fp8_ds_mla`` rows using torch tensor operations only."""
        expected_dim = _INLINE_SCALE_NOPE_DIM + _INLINE_SCALE_ROPE_DIM
        if latent_cache.shape[-1] != expected_dim:
            raise ValueError(
                f"Inline-scale DSA cache expects latent dimension {expected_dim}, "
                f"got {latent_cache.shape[-1]}"
            )
        nope = latent_cache[..., :_INLINE_SCALE_NOPE_DIM].float()
        nope_tiles = nope.view(
            -1,
            _INLINE_SCALE_NOPE_DIM // _INLINE_SCALE_QUANT_TILE,
            _INLINE_SCALE_QUANT_TILE,
        )
        scales = nope_tiles.abs().amax(dim=-1).clamp_min(1e-8) / DSAVanillaIndexer._FP8_MAX
        quantized = (nope_tiles / scales.unsqueeze(-1)).clamp(
            -DSAVanillaIndexer._FP8_MAX, DSAVanillaIndexer._FP8_MAX
        )
        quantized = quantized.to(torch.float8_e4m3fn).view(-1, _INLINE_SCALE_NOPE_DIM)
        rope = latent_cache[..., _INLINE_SCALE_NOPE_DIM:].to(torch.bfloat16).contiguous()
        packed = torch.cat(
            [
                quantized.contiguous().view(torch.uint8),
                scales.contiguous().view(torch.uint8),
                rope.view(torch.uint8),
            ],
            dim=-1,
        )
        if packed.shape[-1] != _INLINE_SCALE_TOKEN_BYTES:
            raise RuntimeError(
                f"Inline-scale DSA row has {packed.shape[-1]} bytes, "
                f"expected {_INLINE_SCALE_TOKEN_BYTES}"
            )
        return packed.view(storage_dtype).reshape(latent_cache.shape[0], -1)

    @staticmethod
    def _unpack_inline_scale_latent(packed_cache: torch.Tensor) -> torch.Tensor:
        """Decode ``fp8_ds_mla`` cache rows using torch tensor operations only."""
        packed = packed_cache.contiguous().view(torch.uint8).reshape(packed_cache.shape[0], -1)
        if packed.shape[-1] != _INLINE_SCALE_TOKEN_BYTES:
            raise ValueError(
                f"Inline-scale DSA row has {packed.shape[-1]} bytes, "
                f"expected {_INLINE_SCALE_TOKEN_BYTES}"
            )
        scale_end = _INLINE_SCALE_NOPE_DIM + 4 * (
            _INLINE_SCALE_NOPE_DIM // _INLINE_SCALE_QUANT_TILE
        )
        nope = packed[:, :_INLINE_SCALE_NOPE_DIM].view(torch.float8_e4m3fn).float()
        scales = packed[:, _INLINE_SCALE_NOPE_DIM:scale_end].contiguous().view(torch.float32)
        nope = nope * scales.repeat_interleave(_INLINE_SCALE_QUANT_TILE, dim=-1)
        rope = packed[:, scale_end:].contiguous().view(torch.bfloat16).float()
        return torch.cat([nope, rope], dim=-1)

    def _append_latent_cache(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        request_ids: list[int],
        seq_lens: list[int],
        past_lens: list[int],
        latent_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Append ordinary or inline-scale latent rows to the paged cache."""
        manager = metadata.kv_cache_manager
        if not getattr(manager, "use_fp8_ds_mla", False):
            from ...utils import append_mla_latent_cache

            return append_mla_latent_cache(
                manager,
                self.layer_idx,
                request_ids,
                seq_lens,
                past_lens,
                latent_cache,
                kv_layout=metadata.kv_layout,
            )

        kv_cache = manager.get_buffers(self.layer_idx, kv_layout=metadata.kv_layout)
        packed = self._pack_inline_scale_latent(latent_cache, kv_cache.dtype)
        blocks_per_seq = manager.get_batch_cache_indices(request_ids, self.layer_idx)
        tokens_per_block = manager.tokens_per_block
        source_offset = 0
        for seq_idx, (q_len, past_len) in enumerate(zip(seq_lens, past_lens, strict=True)):
            written = 0
            blocks = [block for block in blocks_per_seq[seq_idx] if block != BAD_PAGE_INDEX]
            while written < q_len:
                position = past_len + written
                block = blocks[position // tokens_per_block]
                block_offset = position % tokens_per_block
                num_tokens = min(tokens_per_block - block_offset, q_len - written)
                source = packed[source_offset + written : source_offset + written + num_tokens]
                if metadata.kv_layout == "NHD":
                    kv_cache[block, 0, block_offset : block_offset + num_tokens, 0, :].copy_(source)
                elif metadata.kv_layout == "HND":
                    kv_cache[block, 0, 0, block_offset : block_offset + num_tokens, :].copy_(source)
                else:
                    raise ValueError(f"Unsupported KV cache layout: {metadata.kv_layout}")
                written += num_tokens
            source_offset += q_len
        return kv_cache

    @staticmethod
    def _load_latent_cache(
        kv_cache: torch.Tensor,
        block_ids: list[int],
        kv_len: int,
        kv_layout: str,
        *,
        use_fp8_ds_mla: bool = False,
    ) -> torch.Tensor:
        if kv_layout == "NHD":
            tokens_per_block = kv_cache.shape[2]
        elif kv_layout == "HND":
            tokens_per_block = kv_cache.shape[3]
        else:
            raise ValueError(f"Unsupported KV cache layout: {kv_layout}")

        # Drop invalid pages rather than zero-filling in place: this must mirror
        # append_mla_latent_cache, which compacts the same way before indexing
        # (attention/backends/utils.py). VanillaAttention._gather_paged_mla_latent
        # zero-fills instead because it pairs with a different write path.
        valid_block_ids = [block_id for block_id in block_ids if block_id != BAD_PAGE_INDEX]
        num_required_blocks = math.ceil(kv_len / tokens_per_block)
        if len(valid_block_ids) < num_required_blocks:
            raise ValueError(
                f"DSA cache has {len(valid_block_ids)} blocks, but "
                f"{num_required_blocks} are required for {kv_len} tokens"
            )

        chunks = []
        remaining = kv_len
        for block_id in valid_block_ids[:num_required_blocks]:
            num_tokens = min(tokens_per_block, remaining)
            if kv_layout == "NHD":
                chunks.append(kv_cache[block_id, 0, :num_tokens, 0, :])
            else:
                chunks.append(kv_cache[block_id, 0, 0, :num_tokens, :])
            remaining -= num_tokens
        latent_cache = torch.cat(chunks, dim=0)
        if use_fp8_ds_mla:
            return DSAVanillaAttention._unpack_inline_scale_latent(latent_cache)
        return latent_cache

    def _forward_sparse(
        self,
        fused_q: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        latent_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attention_input_type: AttentionInputType,
        append_cache: bool,
    ) -> torch.Tensor:
        if attention_input_type == AttentionInputType.context_only:
            seq_start, seq_end = 0, metadata.num_contexts
        elif attention_input_type == AttentionInputType.generation_only:
            seq_start, seq_end = metadata.num_contexts, metadata.num_seqs
        else:
            raise ValueError("DSA requires a context-only or generation-only input")

        phase_seq_lens = metadata.seq_lens.tolist()[seq_start:seq_end]
        num_phase_tokens = sum(phase_seq_lens)
        fused_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        expected_q_shape = (num_phase_tokens, self.num_heads * fused_head_dim)
        if fused_q.shape != expected_q_shape:
            raise ValueError(
                f"DSA query must have shape {expected_q_shape}, got {tuple(fused_q.shape)}"
            )
        if latent_cache.shape != (num_phase_tokens, fused_head_dim):
            raise ValueError(
                "DSA latent cache must have shape "
                f"[{num_phase_tokens}, {fused_head_dim}], got {tuple(latent_cache.shape)}"
            )
        if topk_indices.ndim != 2 or topk_indices.shape[0] != num_phase_tokens:
            raise ValueError(
                "DSA top-k indices must have shape [num_phase_tokens, top_k], got "
                f"{tuple(topk_indices.shape)}"
            )
        phase_past_tokens = _cached_lens(metadata)[seq_start:seq_end]
        valid_mask = topk_indices >= 0
        if torch.any(topk_indices < -1):
            raise ValueError("DSA top-k indices may only use -1 as padding")
        if torch.any(~valid_mask.any(dim=1)):
            raise ValueError("Every DSA query token must select at least one KV token")

        causal_limits = torch.cat(
            [
                torch.arange(
                    int(past),
                    int(past) + q_len,
                    dtype=topk_indices.dtype,
                    device=topk_indices.device,
                )
                for past, q_len in zip(phase_past_tokens, phase_seq_lens, strict=True)
            ]
        )
        if torch.any(valid_mask & (topk_indices > causal_limits.unsqueeze(1))):
            raise ValueError("DSA top-k index selects a future token")

        if append_cache:
            kv_cache = self._append_latent_cache(
                metadata,
                metadata.request_ids[seq_start:seq_end],
                phase_seq_lens,
                phase_past_tokens,
                latent_cache,
            )
        else:
            kv_cache = metadata.kv_cache_manager.get_buffers(
                self.layer_idx, kv_layout=metadata.kv_layout
            )

        # Always ask the manager: DSA metadata inherits TrtllmAttentionMetadata,
        # whose block_ids_per_seq is a zero-padded CUDA tensor (and only exists
        # under enable_flash_mla), not the list-of-lists this gather wants.
        block_ids_per_seq = metadata.kv_cache_manager.get_batch_cache_indices(
            metadata.request_ids, self.layer_idx
        )
        use_fp8_ds_mla = getattr(metadata.kv_cache_manager, "use_fp8_ds_mla", False)

        q = fused_q.view(num_phase_tokens, self.num_heads, fused_head_dim)
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        scale = 1.0 / (math.sqrt(qk_head_dim) * (self.q_scaling or 1.0))
        outputs = []
        token_offset = 0
        for phase_idx, q_len in enumerate(phase_seq_lens):
            seq_idx = seq_start + phase_idx
            kv_len = int(phase_past_tokens[phase_idx]) + q_len
            latent = self._load_latent_cache(
                kv_cache,
                block_ids_per_seq[seq_idx],
                kv_len,
                metadata.kv_layout,
                use_fp8_ds_mla=use_fp8_ds_mla,
            ).to(q.dtype)
            per_token_outputs = []
            for token_idx in range(q_len):
                row = topk_indices[token_offset + token_idx]
                selected = row[row >= 0].to(device=q.device, dtype=torch.long)
                per_token_outputs.append(
                    self._selected_mla_attention(
                        q[token_offset + token_idx],
                        latent.index_select(0, selected),
                        value_dim=self.kv_lora_rank,
                        scale=scale,
                    )
                )
            outputs.append(
                torch.stack(per_token_outputs).reshape(q_len, self.num_heads * self.kv_lora_rank)
            )
            token_offset += q_len
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        forward_args: Optional[AttentionForwardArgs] = None,
        **kwargs,
    ) -> torch.Tensor:
        forward_args = merge_attention_forward_args(forward_args, kwargs)
        if metadata.multi_item_part_lens is not None:
            raise ValueError("DSA Vanilla attention does not support multi-item scoring")
        if metadata.kv_cache_manager is None:
            raise ValueError("DSA Vanilla attention requires a KV cache manager")
        if forward_args.latent_cache is None:
            raise ValueError("DSA Vanilla attention requires latent_cache")
        if k is not None or v is not None:
            raise ValueError("DSA Vanilla attention expects absorbed queries without K/V")

        # Ordinary generation RoPE already ran via mla_rope_generation. Context
        # RoPE runs in the attention kernel, while fp8_ds_mla generation defers
        # it to the FlashInfer FMHA; reproduce both deferred cases here. The
        # test-only skip flag means the caller supplied already-rotated inputs.
        use_fp8_ds_mla = getattr(metadata.kv_cache_manager, "use_fp8_ds_mla", False)
        apply_deferred_rope = (
            forward_args.attention_input_type == AttentionInputType.context_only
            or (
                forward_args.attention_input_type == AttentionInputType.generation_only
                and use_fp8_ds_mla
            )
        )
        if apply_deferred_rope and not forward_args.skip_mla_rope_generation:
            if forward_args.q_pe is None:
                raise ValueError("DSA Vanilla fused RoPE requires forward_args.q_pe")
            if forward_args.attention_input_type == AttentionInputType.context_only:
                seq_start, seq_end = 0, metadata.num_contexts
            else:
                seq_start, seq_end = metadata.num_contexts, metadata.num_seqs
            self._apply_mla_rope(
                q,
                forward_args.q_pe,
                forward_args.latent_cache,
                self._token_positions(metadata, seq_start, seq_end, q.device),
            )

        # Ordinary generation reached here after mla_rope_generation already
        # appended the rotated latent.  The standalone test path sets the skip
        # flag and supplies pre-rotated tensors, so it still needs this forward
        # to append them. Context and fp8_ds_mla generation also append here.
        cache_already_appended = (
            forward_args.attention_input_type == AttentionInputType.generation_only
            and not use_fp8_ds_mla
            and not forward_args.skip_mla_rope_generation
        )

        local_topk = self._select_local_topk(q, k, metadata, forward_args)
        is_generation = forward_args.attention_input_type == AttentionInputType.generation_only
        local_layer_idx = metadata.kv_cache_manager.layer_offsets[self.layer_idx]
        sparse_attn_indices = self._local_topk_to_global(
            local_topk, metadata, local_layer_idx, is_generation
        )
        forward_args.sparse_runtime_params = replace(
            forward_args.sparse_runtime_params,
            sparse_attn_indices=sparse_attn_indices,
            sparse_attn_offsets=None,
        )
        return self._forward_sparse(
            q,
            metadata,
            forward_args.latent_cache,
            local_topk,
            forward_args.attention_input_type,
            append_cache=not cache_already_appended,
        )


__all__ = ["DSAVanillaAttention", "DSAVanillaIndexer"]
