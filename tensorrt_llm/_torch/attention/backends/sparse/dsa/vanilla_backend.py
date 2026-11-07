# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch golden reference for DeepSeek Sparse Attention (DSA).

DSA attends to a small, per-query subset of each request's KV cache. Every
layer runs two stages, both reproduced here with plain torch operations:

1. Index selection (:class:`DSAVanillaIndexer`). A multi-head indexer projects
   each new token into quantized query/key vectors, appends the keys to a
   paged indexer K cache, scores every query against its request's cached
   keys, and keeps the TopK key positions.
2. Sparse MLA (:class:`DSAVanillaAttention`). Absorbed MLA attention runs over
   only the selected rows of the paged latent KV cache.

The reference never dispatches the fused kernels it validates (FlashInfer
RoPE, DeepGEMM logits, custom GEMMs, CUDA TopK or the sparse MLA kernels), so
a kernel bug cannot be masked by an identical golden. In model runs below
SM100, the DSA attention module computes sparse MLA with FlashMLA itself and
calls this backend only for index selection (``sparse_attn_predict``) and for
RoPE plus the latent-cache append (``mla_rope_append_paged_kv_assign_q``).

Shapes and coordinates use this notation throughout:

- T: packed token count of a batch or phase. Context requests come first,
  then generation requests; a "phase" is one of those two slices.
- H, D: indexer head count and indexer head width; R: RoPE width.
- Request-local position: index into one request's KV sequence, starting at 0
  for its first token. TopK results use these positions and ``-1`` for unused
  slots.
- Primary-pool row: index into the KV pool viewed as
  ``[num_blocks * num_layers * tokens_per_block]`` token rows, the space
  consumed by the TRTLLM sparse attention kernels.
"""

import math
import os
from dataclasses import replace
from typing import Optional

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
from tensorrt_llm._torch.attention.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.modules.linear import Linear, _fp8_block_scales_uses_cute_dsl_sm107
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

from ..inline_scale_kv import DIM_NOPE as _INLINE_SCALE_NOPE_DIM
from ..inline_scale_kv import DIM_ROPE as _INLINE_SCALE_ROPE_DIM
from ..inline_scale_kv import NUM_NOPE_TILES as _INLINE_SCALE_NUM_TILES
from ..inline_scale_kv import QUANT_TILE as _INLINE_SCALE_QUANT_TILE
from ..inline_scale_kv import ROPE_OFFSET as _INLINE_SCALE_ROPE_OFFSET
from ..inline_scale_kv import TOKEN_BYTES as _INLINE_SCALE_TOKEN_BYTES
from .indexer import (
    _compute_slot_mappings,
    _effective_compress_ratio_divisor,
    _select_indexer_compress_ratio,
)
from .metadata import DSAtrtllmAttentionMetadata
from .params import DSAParams


class DSAVanillaEmbedding(RotaryEmbedding):
    """Apply pure-PyTorch RoPE to packed, token-major DSA tensors.

    Targets share token positions but may have different head counts or layouts:
    [T, H, R], [T, H * R], or [T, R] for a single head. Here T is the packed token
    count and R is this object's head_dim. DSA callers pass only the positional
    slices of Q/K, so R is the RoPE width, not the full attention/indexer width.
    The inherited constructor builds the cos/sin table from the model's RoPE
    configuration, including its frequency and scaling settings.

    For each dimension pair (x1, x2), using the cached cosine c and sine s for
    that token's position and that pair's frequency, the forward rotation is::

        y1 = x1 * c - x2 * s
        y2 = x2 * c + x1 * s

    With standard RoPE, c = cos(position * frequency) and
    s = sin(position * frequency). Applying these rotations to Q and K makes
    their dot product depend on relative position. Neox layout pairs the first
    and second halves of the rotated dimensions; interleaved layout pairs
    adjacent dimensions. inverse=True reverses the signs of the sine terms.

    RotaryEmbedding.forward can dispatch a Q/K pair to an in-place FlashInfer
    kernel, and its PyTorch fallback interprets rank-three inputs as
    [batch, sequence, features]. This wrapper explicitly adapts packed layouts
    and calls the static apply_rotary_pos_emb helper instead. That helper uses
    ordinary PyTorch operations without fused dispatch, so sharing its rotation
    math and layout conventions keeps this reference independent of the fused
    kernels without duplicating the equations.
    """

    def forward(
        self, position_ids: torch.Tensor, targets: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Rotate targets with shared positions, preserving each target's shape.

        Args:
            position_ids: Integer positions flattening to [T], one per packed
                token. Positions can restart for each request in the batch.
            targets: Floating-point Q/K slices with T leading rows and one or
                more heads of width self.head_dim per row.

        Returns:
            Rotated tensors in input order, with each input's shape and dtype.
            Inputs are not modified in place.
        """
        position_ids = position_ids.reshape(-1)
        num_tokens = position_ids.numel()
        if num_tokens == 0:
            return targets

        # Look up positions once for all targets; packed row indices need not
        # equal token positions within their respective requests.
        cos_sin = self.rotary_cos_sin[position_ids]
        cos, sin = cos_sin[:, 0, :], cos_sin[:, 1, :]

        def apply_rope(target: torch.Tensor) -> torch.Tensor:
            original_shape = target.shape
            if target.shape[0] != num_tokens:
                raise ValueError(
                    "Packed RoPE targets must have one leading row per position: "
                    f"got target shape {tuple(target.shape)} and {num_tokens} positions"
                )
            # Normalize each layout to [T, H, R], then [1, H, T, R]. The helper
            # broadcasts the position-dependent cos/sin values across heads;
            # the leading singleton is a layout adapter, not a request count.
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
    """Return each request's KV length before this step's tokens are appended.

    ``kv_lens_cuda`` holds every request's total KV length including this
    step's tokens, so subtracting ``seq_lens`` (new tokens per request) leaves
    the cached prefix. That value is the position of the request's first new
    token, shared by RoPE positions, cache-append offsets and causal limits.
    """
    seq_lens = metadata.seq_lens.tolist()
    kv_lens = metadata.kv_lens_cuda[: metadata.num_seqs].tolist()
    return [kv_len - seq_len for kv_len, seq_len in zip(kv_lens, seq_lens, strict=True)]


def _phase_bounds(
    metadata: DSAtrtllmAttentionMetadata, is_generation: Optional[bool]
) -> tuple[int, int, int, int]:
    """Return (seq_start, seq_end, token_start, token_end) in the packed batch.

    Bounds are half-open, with context requests/tokens before generation.
    None selects the whole batch; False/True selects context/generation.
    """
    if is_generation is None:
        return 0, metadata.num_seqs, 0, metadata.num_tokens
    if is_generation:
        return (
            metadata.num_contexts,
            metadata.num_seqs,
            metadata.num_ctx_tokens,
            metadata.num_tokens,
        )
    return 0, metadata.num_contexts, 0, metadata.num_ctx_tokens


def _phase_lens(
    metadata: DSAtrtllmAttentionMetadata, is_generation: Optional[bool]
) -> tuple[int, int, list[int], list[int]]:
    """Return (seq_start, seq_end, seq_lens, past_lens) for the selected requests.

    ``seq_lens`` and ``past_lens`` hold the new and cached token counts of
    requests ``seq_start`` through ``seq_end - 1``, in packed batch order.
    """
    seq_start, seq_end, _, _ = _phase_bounds(metadata, is_generation)
    seq_lens = metadata.seq_lens.tolist()[seq_start:seq_end]
    past_lens = _cached_lens(metadata)[seq_start:seq_end]
    return seq_start, seq_end, seq_lens, past_lens


def _token_positions(
    seq_lens: list[int], past_lens: list[int], device: torch.device
) -> torch.Tensor:
    """Return int32 [sum(seq_lens)] absolute positions of the new tokens.

    Request i contributes ``past_lens[i], ..., past_lens[i] + seq_lens[i] - 1``.
    These positions select RoPE angles and bound causal TopK selections.
    """
    pieces = [
        torch.arange(past, past + seq_len, dtype=torch.int32, device=device)
        for seq_len, past in zip(seq_lens, past_lens, strict=True)
    ]
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.int32, device=device)


def _phase_is_generation(attention_input_type: AttentionInputType) -> bool:
    """Map a DSA phase to ``is_generation``; DSA never runs a mixed batch at once."""
    if attention_input_type == AttentionInputType.context_only:
        return False
    if attention_input_type == AttentionInputType.generation_only:
        return True
    raise ValueError("DSA requires a context-only or generation-only input")


def _byte_rows(starts: torch.Tensor, width: int) -> torch.Tensor:
    """Return int64 [N, width] flat indices of ``width`` bytes from each start offset."""
    return starts.unsqueeze(1) + torch.arange(width, dtype=torch.int64, device=starts.device)


class DSAVanillaIndexer(nn.Module):
    """Standalone PyTorch reference for DSA token selection.

    For T input tokens, H indexer heads and head width D, project MLA's low-rank
    query representation into Q [T, H, D], and hidden states into a shared
    K [T, D] and learned head weights [T, H]. Apply RoPE and quantize Q/K, append
    new K entries to the indexer's paged cache, then score each request's keys.
    Expressed using dequantized Q/K and the unscaled learned weights, scoring is::

        score[t, s] = sum_h(weights[t, h] * relu(dot(Q[t, h], K[s]))) / sqrt(D * H)

    Causal masking and TopK produce int32 request-local key positions, with -1
    for invalid slots. These scores are not softmax probabilities: the attention
    backend uses the selected positions to compute MLA attention from its own
    latent KV cache. This class caches only the keys used for index selection.

    The pipeline maps onto these methods:

    - pre_indexer_proj: project Q, K and head weights, apply RoPE, quantize.
    - _update_k_cache: append the new quantized keys to the paged K cache.
    - forward_from_projected / sparse_attn_indexer: gather each request's keys
      from the cache, score them, mask causally and select TopK.
    - forward: all three steps for the whole batch.

    Quantization helpers reproduce checkpoint and cache formats using torch
    operations; sparse_attn_indexer scores requests with the shared
    weighted_relu_scores helper.
    """

    # e4m3 and E2M1 maxima; the quantizers scale each block to fill the range.
    FP8_MAX = 448.0
    FP4_MAX = 6.0
    # E2M1 magnitudes indexed by the low three code bits, and the midpoints
    # between neighboring magnitudes used for round-to-nearest.
    _E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    _E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    # Bound [H, tile_q, S] float32 intermediates in weighted_relu_scores on
    # multi-thousand-token sequences while keeping each query row's reduction exact.
    _SCORE_QUERY_TILE = 256

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
        """Mirror :class:`Indexer`'s signature and checkpoint names.

        Args:
            quant_config: Checkpoint quantization, applied to ``wq_b`` only;
                ``wk`` and ``weights_proj`` are always unquantized.
            pos_embd_params: Model RoPE configuration for the indexer Q/K.
            mla_params: Supplies hidden_size, q_lora_rank and the RoPE width.
            skip_create_weights_in_init: Defer weight allocation to the loader.
            sparse_params: Indexer head count and width, TopK, K cache dtype
                (FP8 or FP4), RoPE layout and MTP index sharing.
            dtype: Model dtype of ``wq_b``.
            compress_ratio: Accepted for signature parity; scoring reads the
                ratio from ``metadata.compress_ratios``.
            layer_idx: Layer whose indexer K cache this module reads and writes.
            aux_stream: Accepted for signature parity; the reference runs on
                the current stream only.
        """
        super().__init__()
        self.hidden_size = mla_params.hidden_size
        self.q_lora_rank = mla_params.q_lora_rank
        self.rope_dim = mla_params.qk_rope_head_dim
        self.n_heads = sparse_params.index_n_heads
        self.head_dim = sparse_params.index_head_dim
        self.index_topk = sparse_params.index_topk
        self.layer_idx = layer_idx
        self.use_fp4 = sparse_params.indexer_k_dtype == "fp4"
        self.mtp_index_share = sparse_params.mtp_index_share
        # As in the production indexer, TRTLLM_DSA_INDEXER_BF16=1 runs the
        # fused wk/weights_proj projection in the model dtype instead of FP32.
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
        # All H query heads compare against the same D-dimensional key per token.
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
        self.rotary_emb = DSAVanillaEmbedding(
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
        """Combine K/head-weight projections after checkpoint weights are loaded.

        Both consume hidden_states, so concatenating their output rows permits
        one F.linear call. pre_indexer_proj splits its [T, D + H] result back
        into keys and head weights without changing the checkpoint parameters.
        """
        self._fused_wk_wp_weight = torch.cat(
            [self.wk.weight.data, self.weights_proj.weight.data], dim=0
        )

    def maybe_join_prev_topk_copy(self) -> None:
        """No aux-stream work to join; the reference never forks one."""

    @staticmethod
    def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
        """Round scales up to UE8M0 without perturbing exact powers of two."""
        # UE8M0 stores only an exponent. In the float32 bit pattern, bits 23-30
        # hold the biased exponent and bits 0-22 the mantissa. A nonzero
        # mantissa means the value lies between two powers of two, so bump the
        # exponent to round up. Clamping to [1, 254] keeps the result a finite,
        # normal power of two.
        bits = x.abs().float().view(torch.int32)
        exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
        return (exp.clamp(1, 254) << 23).view(torch.float32)

    @staticmethod
    def pack_ue8m0_to_int(x: torch.Tensor) -> torch.Tensor:
        """Pack four UE8M0 exponents into one int32, as the FP4 cache stores them.

        Args:
            x: float32 [..., 4 * N] scales that are exact powers of two.

        Returns:
            int32 [..., N]. Each word holds the biased-exponent bytes of four
            consecutive scales, the first scale in the lowest byte.
        """
        if x.dtype != torch.float32 or x.size(-1) % 4 != 0:
            raise ValueError(
                "UE8M0 packing needs float32 scales with a multiple of 4 per row, got "
                f"{x.dtype} {tuple(x.shape)}"
            )
        if not (x.view(torch.int32) & ((1 << 23) - 1) == 0).all():
            raise ValueError("UE8M0 packing needs exact power-of-two scales")
        return (x.view(torch.int32) >> 23).to(torch.uint8).view(torch.int32)

    @staticmethod
    def unpack_ue8m0_from_int(packed_sf: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`pack_ue8m0_to_int`: int32 [..., N] -> float32 [..., 4 * N]."""
        # Each byte is a biased exponent; shifting it into bits 23-30 with a
        # zero mantissa rebuilds the float32 power of two.
        return (packed_sf.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)

    @classmethod
    def quantize_fp8(
        cls, x: torch.Tensor, dims: tuple[int, ...] = (0,), use_ue8m0: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize to E4M3, retaining ``dims`` when computing per-slice scales.

        All other axes are reduced by abs-max: Q [T, H, D] uses dims=(0, 1)
        for one scale per token/head, while K [T, D] uses dims=(0,) for one
        scale per token. Reconstruction is quantized.float() * scale, with
        the reduced singleton dimensions restored for broadcasting.

        Args:
            x: Tensor to quantize.
            dims: Axes that keep their own scale.
            use_ue8m0: Round scales up to powers of two, as DeepGEMM expects.

        Returns:
            (codes, scales): float8_e4m3fn codes with x's shape, and float32
            scales over ``dims`` with singleton dimensions squeezed.
        """
        excluded = tuple(i for i in range(x.dim()) if i not in set(dims))
        # Keep this floor aligned with fusedCatFp8.cu.  A larger floor changes
        # both the scale and the FP8 codes for otherwise valid tiny rows.
        amax = x.abs().float().amax(dim=excluded, keepdim=True).clamp(1e-12)
        sf = amax / cls.FP8_MAX
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize to packed E2M1 nibbles with one scale per ``gran_k`` block.

        Args:
            x: [M, N] tensor with even N. Columns are zero-padded to a multiple
                of ``gran_k`` before computing block scales.
            use_ue8m0: Round block scales up to powers of two.
            gran_k: Elements that share one scale.
            use_packed_ue8m0: Return scales as int32 words that each pack four
                UE8M0 exponents; requires ``use_ue8m0``.

        Returns:
            (packed, scales): int8 [M, N // 2] bytes holding element 2i in the
            low nibble and element 2i + 1 in the high nibble, and float32
            [M, ceil(N / gran_k)] scales, or int32 [M, ceil(N / gran_k) // 4]
            words when ``use_packed_ue8m0`` is set.
        """
        m, n = x.shape
        if n % 2 != 0:
            raise ValueError(f"E2M1 packing needs an even column count, got {n}")
        if use_packed_ue8m0 and not use_ue8m0:
            raise ValueError("Packed UE8M0 scales require use_ue8m0=True")
        padded_n = (n + gran_k - 1) // gran_k * gran_k
        x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
        x_padded[:, :n] = x
        x_view = x_padded.view(m, -1, gran_k)
        # fusedCatFp4.cu uses the same floor before encoding the UE8M0 scale.
        sf = x_view.abs().float().amax(dim=2).clamp_min(1e-12) / cls.FP4_MAX
        if use_ue8m0:
            sf = cls.ceil_to_ue8m0(sf)
        codes = cls._to_e2m1(x_view * (1.0 / sf.unsqueeze(2))).view(m, padded_n)
        pairs = codes.view(m, padded_n // 2, 2)
        # Consecutive values occupy the low/high four bits of each byte. Scale
        # packing is separate: four UE8M0 exponent bytes can share one int32.
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
        """Decode packed E2M1 nibbles and apply their block scales.

        Args:
            packed: int8 or uint8 [M, N // 2] bytes in :meth:`quantize_fp4`
                order (element 2i in the low nibble).
            sf: float32 [M, N // gran_k] block scales, or int32
                [M, N // gran_k // 4] packed UE8M0 words.
            gran_k: Elements that share one scale.
            use_packed_ue8m0: Unpack ``sf`` with :meth:`unpack_ue8m0_from_int`.

        Returns:
            float32 [M, N] values.
        """
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
        """Reverse CUTLASS's 128x4 scale layout with torch indexing.

        NVFP4 Linear weights keep one E4M3 scale byte per 16-element block in
        the swizzled layout read by CUTLASS block-scaled GEMMs. The logical
        [rows, cols] grid is padded to multiples of 128 rows and 4 columns and
        split into 128x4 tiles of 512 bytes, stored row-tile-major. Inside a
        tile, row r and column c sit at byte ``(r % 32) * 16 + (r // 32) * 4 + c``,
        so each 16-byte line holds four columns of rows r, r + 32, r + 64 and
        r + 96.

        Args:
            interleaved: Swizzled scale storage of any dtype.
            rows: Logical scale rows (weight output features).
            cols: Logical scale columns (input features // 16).

        Returns:
            uint8 [rows, cols] scale bytes in row-major order.
        """
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

    @classmethod
    def _qdq_fp8(cls, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Quantize-dequantize ``x`` through saturating E4M3.

        Args:
            x: Values to round.
            scale: Dequantization scale, broadcastable to ``x``.

        Returns:
            ``e4m3(x / scale) * scale`` in x's dtype, with ``x / scale``
            clamped to the finite E4M3 range first.
        """
        scaled = (x.float() / scale.float()).clamp(-cls.FP8_MAX, cls.FP8_MAX)
        return (scaled.to(torch.float8_e4m3fn).float() * scale.float()).to(x.dtype)

    @classmethod
    def _qdq_fp8_blocks(
        cls,
        x: torch.Tensor,
        block_size: int = 128,
        use_ue8m0: bool = True,
    ) -> torch.Tensor:
        """Quantize-dequantize activations in 1 x ``block_size`` E4M3 blocks.

        Each block of the last dimension gets its own 1x128 scale, rounded up
        to a UE8M0 power of two when the target FP8 block-scaled GEMM path
        uses UE8M0 activation scales or left as float32 otherwise.

        The arithmetic follows ``fp8_quantize_1x128``'s kernel: it forms
        ``quant = 448 / amax``, stores ``dequant = 1 / quant``, and quantizes
        ``x * quant``. UE8M0 results match the kernel exactly. Float32 scales
        can differ by one ulp because the kernel is built with
        ``--use_fast_math``, which occasionally moves an element to the
        adjacent E4M3 code.

        Args:
            x: [..., K] activations; K must be divisible by ``block_size``.
            block_size: Elements per scale.
            use_ue8m0: Whether to round block scales up to powers of two.

        Returns:
            QDQ values with x's shape and dtype.
        """
        rows = x.reshape(-1, x.shape[-1])
        if rows.shape[1] % block_size != 0:
            raise ValueError(
                f"FP8 block quantization requires K divisible by {block_size}, got {rows.shape[1]}"
            )
        blocks = rows.view(rows.shape[0], -1, block_size)
        # fp8_quantize_1x128 uses this floor before encoding the 1x128 scale.
        amax = blocks.abs().float().amax(dim=-1, keepdim=True).clamp_min(1e-10)
        quant_scale = cls.FP8_MAX / amax
        dequant_scale = 1.0 / quant_scale
        if use_ue8m0:
            dequant_scale = cls.ceil_to_ue8m0(dequant_scale)
            quant_scale = 1.0 / dequant_scale
        scaled = (blocks.float() * quant_scale).clamp(-cls.FP8_MAX, cls.FP8_MAX)
        qdq = scaled.to(torch.float8_e4m3fn).float() * dequant_scale
        return qdq.to(x.dtype).reshape_as(x)

    @staticmethod
    def _fp8_block_scales_uses_ue8m0(linear: Linear) -> bool:
        """Return whether ``FP8BlockScalesLinearMethod`` quantizes inputs with UE8M0 scales.

        Matches the dispatch in :meth:`FP8BlockScalesLinearMethod.apply`:

        - Pre-transformed ``int32`` weight scales (DeepGEMM / SM120 layouts)
          always pair with UE8M0 activation scales.
        - On the SM100 family, ``fp8_swap_ab_gemm`` uses UE8M0 scales. When
          ``use_cute_dsl_blockscaling_mm`` or ``disable_deep_gemm`` is set on
          the Linear, only the SM107 CuTe DSL MXFP8 kernel quantizes with
          UE8M0; the other kernels call ``fp8_quantize_1x128(input)``, which
          defaults to ``use_ue8m0=False``.
        - On SM120, ``per_token_quant_and_transform`` defaults to
          ``scale_ue8m0=True``.
        - All other SM versions (including SM90 Hopper) call
          ``fp8_quantize_1x128(input)`` with ``use_ue8m0=False``.
        """
        weight_scale = getattr(linear, "weight_scale", None)
        if weight_scale is not None and weight_scale.dtype == torch.int32:
            return True
        sm_version = get_sm_version()
        if is_sm_100f(sm_version):
            if getattr(linear, "use_cute_dsl_blockscaling_mm", False) or getattr(
                linear, "disable_deep_gemm", False
            ):
                return _fp8_block_scales_uses_cute_dsl_sm107(linear)
            return True
        return sm_version == 120

    @classmethod
    def _round_to_e4m3_rne_satfinite(cls, x: torch.Tensor) -> torch.Tensor:
        """Round to finite E4M3 with the CUDA conversion's RNE semantics."""
        # Positive E4M3FN codes are monotonic. Code 127 is NaN, so stop at
        # code 126 (448) to reproduce ``satfinite`` for out-of-range values.
        # A code is 0eeeemmm: subnormals (e == 0) are m * 2**-9 and normals are
        # (1 + m / 8) * 2**(e - 7).
        codes = torch.arange(127, device=x.device, dtype=torch.int32)
        exponent = codes >> 3
        mantissa = codes & 0x07
        levels = torch.where(
            exponent == 0,
            mantissa.float() * (2.0**-9),
            (1.0 + mantissa.float() / 8.0) * torch.exp2(exponent.float() - 7.0),
        )

        # Pick the nearer of the two neighboring levels; on a tie, keep the
        # level whose code is even (round-to-nearest-even).
        ax = x.abs().float().clamp_max(cls.FP8_MAX)
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
        """Quantize-dequantize activations to NVFP4 with two-level scaling.

        Every 16-element block gets the scale ``amax / 6``, stored as E4M3
        relative to the global ``scale_2``. Values are divided by the
        effective scale ``e4m3(block_scale / scale_2) * scale_2``, rounded to
        E2M1 with round-to-nearest-even and finite saturation, and multiplied
        back. Blocks whose effective scale is zero become zero.

        Args:
            x: [..., K] activations; K must be divisible by 16.
            scale_2: Global dequantization scale (a scalar tensor).

        Returns:
            QDQ values with x's shape and dtype.
        """
        block_size = 16
        rows = x.reshape(-1, x.shape[-1])
        if rows.shape[1] % block_size != 0:
            raise ValueError(
                f"NVFP4 quantization requires K divisible by {block_size}, got {rows.shape[1]}"
            )
        blocks = rows.view(rows.shape[0], -1, block_size)
        block_scale = blocks.abs().float().amax(dim=-1) / cls.FP4_MAX
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
        """Evaluate the checkpoint's Q projection with torch tensor operations.

        This handles quantization of the Linear weights and its input, before
        pre_indexer_proj separately quantizes the projected, rotated Q for
        index scoring. Dequantize weights and apply input quantize-dequantize
        (QDQ) as required by the Linear format, then use F.linear. Input QDQ
        preserves low-precision rounding that a full-precision GEMM would miss.
        Calling Linear.forward could dispatch the same custom GEMM being tested.

        Args:
            qr: [T, q_lora_rank] low-rank query, in the model dtype or already
                quantized to float8_e4m3fn upstream.
            linear: The indexer's ``wq_b``, holding checkpoint weights and scales.

        Returns:
            [T, n_heads * head_dim] projection. Quantized formats return the
            Linear's dtype.
        """
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
            # FP8 checkpoints carry tensor, output-channel, or 128x128 block
            # scales; each branch expands the stored scale for its own layout.
            weight_scale = linear.weight_scale
            if weight_scale.ndim <= 1 and weight_scale.numel() == 1:
                # Per-tensor weight scale. The input uses its static scale, or
                # a dynamic per-tensor scale when none is stored or forced.
                dequant_weight = weight.float() * weight_scale.float()
                if qr.dtype == torch.float8_e4m3fn:
                    input_scale = linear.input_scale.float()
                    dequant_input = qr.float() * input_scale
                else:
                    input_scale = linear.input_scale
                    if input_scale is None or linear.force_dynamic_quantization:
                        input_scale = qr.abs().float().amax().clamp_min(1e-12) / cls.FP8_MAX
                    dequant_input = cls._qdq_fp8(qr, input_scale).float()
            elif weight_scale.ndim == 1:
                # Per-output-channel weight scales with dynamic per-token input scales.
                dequant_weight = weight.float() * weight_scale[:out_features].float().unsqueeze(1)
                if qr.dtype == torch.float8_e4m3fn:
                    dequant_input = qr.float()
                else:
                    input_scale = (
                        qr.abs().float().amax(dim=-1, keepdim=True).clamp_min(1e-12) / cls.FP8_MAX
                    )
                    dequant_input = cls._qdq_fp8(qr, input_scale).float()
            else:
                # 128x128 weight blocks with 1x128 input blocks. int32 scales are
                # UE8M0 exponents packed by FP8BlockScalesLinearMethod on
                # SM100f/SM120; unpack them to the float32 block grid first.
                use_ue8m0 = cls._fp8_block_scales_uses_ue8m0(linear)
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
                    dequant_input = cls._qdq_fp8_blocks(dequant_input, use_ue8m0=use_ue8m0).float()
                else:
                    dequant_input = cls._qdq_fp8_blocks(qr, use_ue8m0=use_ue8m0).float()
            return F.linear(
                dequant_input.to(output_dtype),
                dequant_weight.to(output_dtype),
            )

        expected_packed_shape = (out_features, in_features // 2)
        if weight.shape == expected_packed_shape and hasattr(linear, "scaling_vector_size"):
            # NVFP4 Linear uses block-16 E4M3 scales plus a global scale. This
            # differs from the block-32 UE8M0 format of the indexer's quantized Q/K.
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

            # Optional per-input-channel smoothing applied before quantization.
            dequant_input = qr
            if linear.pre_quant_scale is not None:
                dequant_input = dequant_input * linear.pre_quant_scale
            # Methods that quantize activations also round the input, with a
            # dynamic global scale or the reciprocal of the static input_scale.
            if getattr(linear.quant_method, "quantizes_nvfp4_activations", False):
                if linear.input_scale is None or linear.force_dynamic_quantization:
                    amax = dequant_input.abs().float().amax().clamp_min(1e-12)
                    input_scale_2 = amax / (cls.FP8_MAX * cls.FP4_MAX)
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

    @classmethod
    def _to_e2m1(cls, x: torch.Tensor) -> torch.Tensor:
        """Round to the nearest E2M1 code; exact midpoints take the smaller magnitude.

        Codes hold the sign in bit 3 and an index into ``_E2M1_VALUES`` in
        bits 0-2. Magnitudes saturate at 6, and negative zero encodes as +0.
        """
        ax = x.abs().clamp_max(cls.FP4_MAX)
        midpoints = torch.tensor(cls._E2M1_MIDPOINTS, device=x.device, dtype=ax.dtype)
        idx = torch.bucketize(ax, midpoints)
        code = idx.to(torch.uint8) | (((x < 0) & (idx != 0)).to(torch.uint8) << 3)
        return code.view(torch.int8)

    @classmethod
    def _to_e2m1_rne(cls, x: torch.Tensor) -> torch.Tensor:
        """Convert to E2M1 codes with round-to-nearest-even and finite saturation."""
        ax = x.abs().clamp_max(cls.FP4_MAX)
        midpoints = torch.tensor(cls._E2M1_MIDPOINTS, device=x.device, dtype=ax.dtype)
        lower = torch.bucketize(ax, midpoints)
        upper = torch.bucketize(ax, midpoints, right=True)
        # At an exact midpoint the two bucket results differ by one. Choose
        # the even E2M1 code, matching ``cvt.rn.satfinite.e2m1x2.f32``.
        idx = torch.where((upper != lower) & ((upper & 1) == 0), upper, lower)
        code = idx.to(torch.uint8) | (((x < 0) & (idx != 0)).to(torch.uint8) << 3)
        return code.view(torch.int8)

    @classmethod
    def _from_e2m1(cls, code: torch.Tensor) -> torch.Tensor:
        """Decode E2M1 codes (bit 3 sign, bits 0-2 magnitude index) to float32."""
        values = torch.tensor(cls._E2M1_VALUES, device=code.device, dtype=torch.float32)
        sign, value_idx = (code & 0x08) != 0, (code & 0x07).to(torch.int32)
        value = values[value_idx]
        return torch.where(sign & (value_idx != 0), -value, value)

    @classmethod
    def weighted_relu_scores(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
        epi_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return ``sum_h weights[t, h] * relu(dot(q[t, h], k[s]))`` for one Q/K tile.

        Queries are processed in tiles of at most :attr:`_SCORE_QUERY_TILE`
        rows so multi-thousand-token sequences do not materialize the full
        ``[H, T, S]`` intermediate at once. Each query row is independent, so
        tiling produces identical results.

        Args:
            q: [T, H, D] queries, as FP8 codes or dequantized values.
            k: [S, D] keys shared by all heads.
            weights: [T, H] head weights with any scales already folded in.
            mask: [T, S] bool; False positions contribute zero before the ReLU.
            epi_dtype: dtype of the weighted reduction over heads.

        Returns:
            [T, S] scores in ``epi_dtype``. Masked positions hold 0, not -inf;
            callers apply their own -inf fill before TopK.
        """
        num_queries = q.shape[0]
        num_keys = k.shape[0]
        if num_queries == 0 or num_keys == 0:
            return torch.zeros((num_queries, num_keys), device=q.device, dtype=epi_dtype)
        k_t = k.float().T
        if num_queries <= cls._SCORE_QUERY_TILE:
            scores = torch.matmul(q.float().transpose(0, 1), k_t)
            scores = torch.where(mask.unsqueeze(0), scores, scores.new_zeros(())).relu()
            scores = scores.to(epi_dtype)
            return (weights.to(epi_dtype).T.unsqueeze(-1) * scores).sum(dim=0)

        out = torch.empty((num_queries, num_keys), device=q.device, dtype=epi_dtype)
        zero = k_t.new_zeros(())
        weights_epi = weights.to(epi_dtype)
        for start in range(0, num_queries, cls._SCORE_QUERY_TILE):
            t_slice = slice(start, min(start + cls._SCORE_QUERY_TILE, num_queries))
            scores = torch.matmul(q[t_slice].float().transpose(0, 1), k_t)
            scores = torch.where(mask[t_slice].unsqueeze(0), scores, zero).relu()
            scores = scores.to(epi_dtype)
            out[t_slice] = (weights_epi[t_slice].T.unsqueeze(-1) * scores).sum(dim=0)
        return out

    @staticmethod
    def select_top_k(logits: torch.Tensor, topk: int) -> torch.Tensor:
        """Select the ``topk`` highest-scoring keys of each row.

        Args:
            logits: [T, S] scores with ``-inf`` at masked positions.
            topk: Number of slots per row.

        Returns:
            int32 [T, topk] key indices, with ``-1`` in slots that have no
            valid key.
        """
        num_selected = min(topk, logits.shape[-1])
        values, indices = logits.topk(num_selected, dim=-1)
        # Early causal rows can have fewer valid keys than TopK even when the
        # logits tensor is wide enough, so invalidate selected masked positions.
        indices = indices.to(torch.int32).masked_fill(torch.isneginf(values), -1)
        if num_selected == topk:
            return indices
        padding = indices.new_full((indices.shape[0], topk - num_selected), -1)
        return torch.cat([indices, padding], dim=-1)

    def _gather_keys(
        self, metadata: DSAtrtllmAttentionMetadata, seq_idx: int, kv_len: int
    ) -> torch.Tensor:
        """Return float32 [kv_len, head_dim] keys in request-local position order.

        Translate logical positions through the request's indexer page table,
        gather the data/scale bytes, and dequantize. Physical pages may be
        noncontiguous; the returned rows retain the order used by causal masks
        and TopK indices.

        Each indexer cache block stores ``tokens_per_block`` rows of key data
        followed by ``tokens_per_block`` four-byte scale words: one float32
        scale per FP8 key, or four packed UE8M0 exponents per FP4 key.
        """
        # Use the geometry sources DSAtrtllmAttentionMetadata.prepare() uses
        # for the write-side slot mappings: head width and K dtype from the
        # cache manager, block size and scale granularity from the metadata.
        # Reading them from anywhere else lets the two drift apart silently.
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
            metadata.indexer_quant_block_size,
            data_bytes_per_token=data_bytes_per_token,
        )

        flat = cache.reshape(-1)
        scale = flat[_byte_rows(scale_idx.to(flat.device), 4)]
        k = flat[_byte_rows(data_idx.to(flat.device), data_bytes_per_token)]
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
        """Copy metadata's precomputed dense selection for a skipped phase.

        With indexer skipping enabled, a phase whose requests all have no more
        KV tokens than the TopK budget skips scoring: every causal key would be
        selected anyway. ``prepare_for_skip_indexer`` sets the skip flag and
        ``prepare_dense_topk_indices`` fills ``metadata.topk_indices_buffer``
        with those dense selections, in packed batch row order.
        """
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
        """Return each generation request's TopK row for its last accepted token.

        Every generation request owns ``next_n`` consecutive rows of
        ``gen_topk``. With ``metadata.mtp_num_accepted``, pick row
        ``num_accepted - 1`` (clamped to the request's rows); otherwise pick the
        request's final row. Matches :meth:`Indexer._mtp_last_accepted_rows`.
        """
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
        """Score paged keys and select request-local TopK indices in PyTorch.

        Q and weights contain the selected phase's packed token rows. New K
        entries must already be in the cache so each request can score both its
        cached prefix and its current tokens. Each output row comes from one of
        three sources: metadata's dense selection when the phase skips the
        indexer, the shared MTP selection when a draft step reuses it, or
        scoring the request's cached keys.

        Args:
            metadata: Batch metadata; request fields always describe the full batch.
            hidden_states: Its leading dimension sets the output row count; its
                values are not used.
            q_fp8: [T_phase, H, D] FP8 codes, or [T_phase, H, D // 2] packed FP4.
            k_fp8: Unused; keys are read back from the paged cache.
            k_scale: Unused, as for ``k_fp8``.
            weights: [T_phase, H] float32 head weights with scales folded in.
            q_scale: [T_phase, H, 1] int32 packed UE8M0 words; FP4 only.
            is_generation: None scores the whole batch, False only context
                requests, True only generation requests.

        Returns:
            int32 [hidden_states.shape[0], index_topk] request-local key
            positions, ``-1`` in unused slots. Phase-only calls start at row 0;
            whole-batch calls keep the packed batch row order.
        """
        # Retain the production call signature, but read all keys from the cache
        # rather than treating the current batch's K as the complete history.
        del k_fp8, k_scale
        compress_ratio = _effective_compress_ratio_divisor(
            _select_indexer_compress_ratio(metadata.compress_ratios)
        )

        if self.use_fp4:
            if q_scale is None:
                raise ValueError("FP4 indexer scoring requires q_scale")
            # FP4 scales vary within a head, so reconstruct Q block by block.
            # FP8 has one positive scale per head and folds it into weights.
            q_fp8 = self.dequantize_fp4(
                q_fp8.reshape(-1, self.head_dim // 2),
                q_scale.reshape(-1, 1),
                gran_k=32,
                use_packed_ue8m0=True,
            ).view(-1, self.n_heads, self.head_dim)

        num_contexts = metadata.num_contexts
        seq_start, seq_end, seq_lens, past_lens = _phase_lens(metadata, is_generation)
        if is_generation is None:
            cache_name = "indexer_topk_out_buffer"
        elif is_generation:
            cache_name = "indexer_topk_out_buffer_gen"
        else:
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

        num_ctx_tokens = metadata.num_ctx_tokens
        num_gen_tokens = metadata.num_tokens - num_ctx_tokens
        # A phase-only output starts at row zero; a whole-batch output places
        # generation rows after the packed context rows.
        target_offset = 0 if is_generation is not None else num_ctx_tokens

        # A skipped phase still needs valid indices: use metadata's precomputed
        # dense selection, or the shared selection for an eligible MTP draft.
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
        for seq_idx, q_len, past in zip(
            range(seq_start, seq_end), seq_lens, past_lens, strict=True
        ):
            skip_phase = (seq_idx < num_contexts and metadata.skip_indexer_for_ctx_reqs) or (
                seq_idx >= num_contexts and (metadata.skip_indexer_for_gen_reqs or reuse_topk)
            )
            if not skip_phase:
                kv_len = (past + q_len) // compress_ratio
                token_slice = slice(token, token + q_len)
                # Query token i sees compressed keys before
                # floor((past + i + 1) / compress_ratio). TopK remains in the
                # compressed cache's local coordinate system, as in production.
                keys = torch.arange(kv_len, device=q_fp8.device)
                limits = (
                    torch.arange(past, past + q_len, device=q_fp8.device) + 1
                ) // compress_ratio
                mask = keys.unsqueeze(0) < limits.unsqueeze(1)
                # FP8 Q scales are folded into weights; FP4 Q is dequantized
                # above. _gather_keys reconstructs K scales for both formats.
                logits = self.weighted_relu_scores(
                    q_fp8[token_slice],
                    self._gather_keys(metadata, seq_idx, kv_len),
                    weights[token_slice],
                    mask,
                )
                # The helper zeros masked scores; exclude them from TopK even
                # when valid scores are zero or negative.
                logits = logits.masked_fill(~mask, -float("inf"))
                topk_indices_buffer[token : token + q_len] = self.select_top_k(
                    logits, self.index_topk
                )
            token += q_len

        if self.mtp_index_share and metadata.in_mtp_draft_loop and not reuse_topk:
            # Publish one selection per request for subsequent draft steps:
            # the last accepted generation row or the final context row.
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project and quantize tokens without reading or writing the K cache.

        Args:
            qr: MLA low-rank query input [T, q_lora_rank].
            hidden_states: Layer input [T, hidden_size], or an FP4 wrapper
                retaining that unquantized input for the K/head-weight projection.
            position_ids: Positions flattening to [T] for RoPE.

        Returns:
            (q_fp8, k_fp8, k_scale, weights, q_scale). The fp8 names also cover
            packed FP4 data. Q is [T, H, D] for FP8 or [T, H, D // 2] for FP4;
            K is [T, D] or [T, D // 2], and weights is [T, H]. FP8 scales are
            [T, H, 1] for Q and [T, 1] for K. The FP4 path uses those same scale
            shapes with each int32 packing four block-32 UE8M0 exponents.
        """
        if isinstance(hidden_states, Fp4QuantizedTensor):
            if hidden_states.unquantized_hidden_states is None:
                raise ValueError(
                    "DSA indexer projection needs the unquantized hidden states of an FP4 input"
                )
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
        # Rotate only the positional dimensions; the shared K temporarily gets
        # a singleton head axis to use the same RoPE interface as multi-head Q.
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
            # The reference reconstructs per-block Q scales before scoring;
            # only the common 1 / sqrt(D * H) factor belongs in these weights.
            weights = weights.float() * self.weight_scale_factor
        else:
            # Q carries one scale per (token, head); K one per token.
            q_fp8, q_scale = self.quantize_fp8(q, dims=(0, 1), use_ue8m0=True)
            k_fp8, k_scale = self.quantize_fp8(k, dims=(0,), use_ue8m0=True)
            q_scale = q_scale.reshape(-1, self.n_heads, 1)
            k_scale = k_scale.reshape(-1, 1)
            # For positive s, relu(s * dot(q, k)) = s * relu(dot(q, k)). Fold
            # Q's per-head scale into weights with 1 / sqrt(D * H), allowing
            # scoring to use FP8 Q values directly. _gather_keys applies K scale.
            weights = weights.float() * q_scale.squeeze(-1) * self.weight_scale_factor
        return q_fp8, k_fp8, k_scale, weights, q_scale

    def _update_k_cache(
        self, k_fp8: torch.Tensor, k_scale: torch.Tensor, metadata: DSAtrtllmAttentionMetadata
    ) -> None:
        """Append new quantized indexer keys through metadata's slot mappings.

        Slot mappings hold byte offsets for data and scales in this layer's
        indexer cache. Only current tokens are written; past keys remain for
        later queries. This does not update the separate MLA latent KV cache.
        Like the production indexer, this is a no-op when the metadata has no
        cache manager or slot mappings.

        Args:
            k_fp8: [T, D] FP8 key codes, or [T, D // 2] packed FP4 bytes.
            k_scale: [T, 1] float32 FP8 scales, or [T, 1] int32 words of four
                packed UE8M0 exponents (also accepted as [T, 4] bytes).
            metadata: Batch metadata holding ``slot_mapping_fp8`` and
                ``slot_mapping_scale``.
        """
        if metadata.kv_cache_manager is None or getattr(metadata, "slot_mapping_fp8", None) is None:
            return
        cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(self.layer_idx)
        flat = cache.reshape(-1)
        num_tokens = k_fp8.shape[0]

        data_idx = metadata.slot_mapping_fp8[:num_tokens].to(flat.device)
        data_bytes_per_token = self.head_dim // 2 if self.use_fp4 else self.head_dim
        flat[_byte_rows(data_idx, data_bytes_per_token)] = k_fp8.view(torch.uint8)

        scale_idx = metadata.slot_mapping_scale[:num_tokens].to(flat.device)
        if self.use_fp4:
            # FP4 already carries four packed UE8M0 exponent bytes per int32
            # scale word. A numeric cast to float32 would replace those bytes
            # with the IEEE representation of the integer value.
            if k_scale.element_size() == 1:
                k_scale = k_scale.view(torch.int32)
            scale_bytes = k_scale.contiguous().view(torch.uint8).view(num_tokens, 4)
        else:
            scale_bytes = k_scale.float().contiguous().view(torch.uint8).view(num_tokens, 4)
        flat[_byte_rows(scale_idx, 4)] = scale_bytes

    def forward_from_projected(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        hidden_states: torch.Tensor,
        indexer_intermediates: list[torch.Tensor],
        is_generation: Optional[bool] = None,
    ) -> torch.Tensor:
        """Score an already-projected batch or phase after its K cache update.

        Packed batch order is context tokens followed by generation tokens.
        is_generation=None selects both phases; False/True selects context or
        generation. Slice query-side intermediates to that phase while the
        request metadata and paged K cache continue to describe the full batch.

        Args:
            metadata: Batch metadata.
            hidden_states: Its leading dimension sets the output row count.
            indexer_intermediates: The whole-batch outputs of
                :meth:`pre_indexer_proj`.
            is_generation: Phase selector, as in :meth:`sparse_attn_indexer`.

        Returns:
            int32 request-local TopK positions, as from :meth:`sparse_attn_indexer`.
        """
        _, _, phase_start, phase_end = _phase_bounds(metadata, is_generation)

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
        """Project, append to the K cache and select, for the whole batch.

        Args:
            qr: [T, q_lora_rank] MLA low-rank query.
            hidden_states: [T, hidden_size] layer input.
            metadata: Batch metadata with the indexer slot mappings.
            position_ids: Positions flattening to [T].

        Returns:
            int32 [T, index_topk] request-local TopK positions.
        """
        intermediates = list(self.pre_indexer_proj(qr, hidden_states, position_ids))
        # Scoring reads exclusively from the cache, so append current K first.
        # Per-query causal masks prevent access to later tokens in this append.
        self._update_k_cache(intermediates[1], intermediates[2], metadata)
        return self.forward_from_projected(metadata, hidden_states, intermediates)


class DSAVanillaAttention(VanillaAttention):
    """Standalone PyTorch golden for DSA index selection and sparse MLA.

    This is a drop-in replacement for :class:`DSATrtllmAttention`: it keeps the
    same constructor, ``forward``, RoPE entry points and sparse prediction
    hooks, but evaluates them with torch operations. For each context-only or
    generation-only phase it:

    1. Reproduces the fused kernels' RoPE and latent-cache append, either in an
       entry point the MLA module calls first (``mla_rope_generation``,
       ``mla_rope_append_paged_kv_assign_q``) or deferred to ``forward``.
    2. Selects request-local TopK positions with its own
       :class:`DSAVanillaIndexer`, or reuses the previous full layer's
       selection on shared-indexer layers.
    3. Gathers the selected rows from the paged latent cache and computes
       absorbed MLA attention over them.

    Queries use the absorbed layout [T, num_heads * (kv_lora_rank + R)], and a
    latent row is [kv_lora_rank + R] = [compressed KV | k_pe].
    """

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
        """Build the backend and, on full-indexer layers, its indexer.

        Args:
            layer_idx: Global layer index.
            num_heads: Attention heads on this rank.
            head_dim: Head width forwarded to the base backend.
            num_kv_heads: Forwarded to the base backend.
            quant_config: Checkpoint quantization, also used by the indexer.
            q_scaling: Extra divisor of the softmax scale.
            pos_embd_params: RoPE configuration; required.
            mla_params: MLA dimensions; required.
            skip_create_weights_in_init: Defer indexer weight allocation.
            attention_chunk_size: Forwarded to the base backend.
            sparse_params: DSA parameters. When omitted, they are derived from
                a ``sparse_attention_config`` keyword argument.
            dtype: Model dtype for the indexer weights.
            aux_stream: Accepted for signature parity with DSATrtllmAttention.
            **kwargs: Forwarded to :class:`VanillaAttention`.
        """
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
        if pos_embd_params is None:
            raise ValueError("DSAVanillaAttention requires RoPE parameters")
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
        self.rotary_emb = DSAVanillaEmbedding(
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
        """Rotate the RoPE parts of the absorbed query and latent rows in place.

        Args:
            fused_q: [T, num_heads * (kv_lora_rank + R)] absorbed query; the last
                R columns of each head receive the rotated ``q_pe``.
            q_pe: [T, num_heads, R] or [T, num_heads * R] unrotated query RoPE part.
            latent_cache: [T, kv_lora_rank + R] latent rows; the trailing R
                columns (k_pe) are rotated in place.
            positions: [T] absolute token positions.
            apply_q: Rotate the query side.
            apply_k: Rotate the latent k_pe.
        """
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
        """Torch version of the fused generation RoPE and cache-append kernel.

        Side effects, matching the kernel's outputs (see
        ``TrtllmAttention.mla_rope_generation`` for argument shapes):

        - ``fused_q`` receives the rotated ``q_pe`` in each head's RoPE slot,
          unless ``kv_only``.
        - A rotated (and, with ``kv_norm_weight``, RMS-normalized) copy of
          ``latent_cache`` is appended to the paged cache, unless
          ``kv_done_elsewhere``. ``latent_cache`` itself is left unchanged.
        - ``cu_q_seqlens`` and ``cu_kv_seqlens`` receive prefix sums of the
          query lengths times ``num_heads`` and of the KV lengths, unless
          ``precomputed_cu_seqlens``.
        - ``fmha_scheduler_counter`` is zeroed, ``mla_bmm1_scale`` receives
          ``[softmax_scale, softmax_scale * log2(e)]`` and ``mla_bmm2_scale``
          the output scale, unless ``precomputed_fmha_scheduler``.
        - ``quant_q_buffer`` receives an FP8 copy of ``fused_q``. When
          ``quant_scale_qkv`` is set, q_nope is already quantized there and
          only the RoPE slot is written.
        """
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

        seq_start, seq_end, seq_lens, past_lens = _phase_lens(metadata, is_generation=True)
        positions = _token_positions(seq_lens, past_lens, latent_cache.device)
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
        """Torch version of the fused RoPE, cache-append and Q-assign kernel.

        Rotates the RoPE slice of every head of ``q``
        ([T, num_heads * (qk_nope_head_dim + R)]) and the k_pe slice of
        ``latent_cache`` ([T, kv_lora_rank + R]) in place, then appends the
        phase's latent rows to the paged cache. Below SM100 the DSA module
        calls this before running FlashMLA on the cache.
        """
        del kwargs
        seq_start, seq_end, seq_lens, past_lens = _phase_lens(metadata, is_generation)
        num_tokens = sum(seq_lens)
        q_view = q.view(
            num_tokens,
            self.num_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
        )
        q_pe = q_view[..., self.qk_nope_head_dim :]
        q_pe_rot, k_pe_rot = self.rotary_emb(
            _token_positions(seq_lens, past_lens, q.device),
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
        """Return int32 [T_phase, topk] request-local KV positions for this phase.

        Sources, in priority order:

        1. ``sparse_backend_args.topk_indices`` precomputed by the caller.
        2. On shared-indexer layers, the previous full layer's selection from
           ``metadata.shared_topk_indices``.
        3. This layer's indexer, scoring ``sparse_backend_args.indexer_intermediates``.
           The result is published to ``metadata.shared_topk_indices`` for the
           shared layers that follow, except in an MTP draft loop with index
           sharing, where that buffer holds per-request MTP selections instead.
        """
        del k
        sparse_backend_args = forward_args.sparse_backend_args
        if sparse_backend_args is None:
            raise ValueError("DSA Vanilla attention requires sparse_backend_args")
        if sparse_backend_args.topk_indices is not None:
            return sparse_backend_args.topk_indices

        is_generation = _phase_is_generation(forward_args.attention_input_type)
        _, _, phase_start, phase_end = _phase_bounds(metadata, is_generation)
        shared_topk_indices = metadata.shared_topk_indices
        if self.indexer is None:
            if shared_topk_indices is None:
                raise ValueError(
                    "Shared-indexer DSA layers need metadata.shared_topk_indices from a full layer"
                )
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
        """Convert request-local positions to rows of the primary KV pool.

        Local position p of a token's request lies on page
        ``block_table[req, p // tokens_per_block]``. With the pool viewed as
        [num_blocks * num_layers * tokens_per_block] token rows, its row is
        ``(page * page_index_scale + layer_offset) * tokens_per_block
        + p % tokens_per_block``; the V1 cache manager uses
        ``page_index_scale = num_local_layers``. Padding (-1) and positions past
        the block table map to -1.

        Args:
            topk_indices: int32 [T_phase, topk] request-local positions.
            metadata: Provides the cached per-phase block table and
                token-to-request map.
            layer_idx: Local layer index within the KV cache manager.
            is_generation: Use the generation-phase slices instead of context.

        Returns:
            int32 [T_phase, topk] primary-pool rows.
        """
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

    def _predict_topk(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select once and return both request-local and primary-pool TopK indices."""
        local_topk = self._select_local_topk(q, k, metadata, forward_args)
        is_generation = _phase_is_generation(forward_args.attention_input_type)
        local_layer_idx = metadata.kv_cache_manager.layer_offsets[self.layer_idx]
        global_topk = self._local_topk_to_global(
            local_topk, metadata, local_layer_idx, is_generation
        )
        return local_topk, global_topk

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run the torch indexer and return TRTLLM-compatible pool indices."""
        _, global_topk = self._predict_topk(q, k, metadata, forward_args)
        return global_topk, None

    def sparse_kv_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """No-op KV prediction; DSA uses indexer-based selection instead."""
        return None, None

    @staticmethod
    def _pack_inline_scale_latent(
        latent_cache: torch.Tensor, storage_dtype: torch.dtype
    ) -> torch.Tensor:
        """Encode latent rows into the 656-byte ``fp8_ds_mla`` token layout.

        Per token, bytes [0, 512) hold the compressed KV in E4M3 with one scale
        per 128-element tile, bytes [512, 528) the four float32 tile scales
        (amax / 448), and bytes [528, 656) k_pe as 64 bfloat16 values. Like the
        production packer in ``inline_scale_kv``, values are multiplied by the
        reciprocal scale before the E4M3 cast.

        Args:
            latent_cache: [T, 576] latent rows (512 compressed KV + 64 RoPE).
            storage_dtype: KV cache element dtype; the packed bytes are
                reinterpreted as it, not cast.

        Returns:
            [T, 656 // storage_dtype.itemsize] packed rows in ``storage_dtype``.
        """
        expected_dim = _INLINE_SCALE_NOPE_DIM + _INLINE_SCALE_ROPE_DIM
        if latent_cache.shape[-1] != expected_dim:
            raise ValueError(
                f"Inline-scale DSA cache expects latent dimension {expected_dim}, "
                f"got {latent_cache.shape[-1]}"
            )
        nope = latent_cache[..., :_INLINE_SCALE_NOPE_DIM].float()
        nope_tiles = nope.view(-1, _INLINE_SCALE_NUM_TILES, _INLINE_SCALE_QUANT_TILE)
        scales = nope_tiles.abs().amax(dim=-1).clamp_min(1e-8) / DSAVanillaIndexer.FP8_MAX
        quantized = (nope_tiles * (1.0 / scales.unsqueeze(-1))).clamp(
            -DSAVanillaIndexer.FP8_MAX, DSAVanillaIndexer.FP8_MAX
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
        """Decode rows packed by :meth:`_pack_inline_scale_latent`.

        Args:
            packed_cache: [T, ...] rows spanning 656 bytes per token.

        Returns:
            float32 [T, 576] latent rows.
        """
        packed = packed_cache.contiguous().view(torch.uint8).reshape(packed_cache.shape[0], -1)
        if packed.shape[-1] != _INLINE_SCALE_TOKEN_BYTES:
            raise ValueError(
                f"Inline-scale DSA row has {packed.shape[-1]} bytes, "
                f"expected {_INLINE_SCALE_TOKEN_BYTES}"
            )
        # The tile scales end where the bfloat16 k_pe begins.
        scale_end = _INLINE_SCALE_ROPE_OFFSET
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
        """Append latent rows to the paged cache, packing them first for ``fp8_ds_mla``.

        Args:
            metadata: Batch metadata with the KV cache manager and layout.
            request_ids: Requests whose rows are appended, in packed order.
            seq_lens: New tokens per request.
            past_lens: Cached tokens per request, i.e. each first new position.
            latent_cache: [sum(seq_lens), kv_lora_rank + R] rows to append.

        Returns:
            The layer's KV cache buffer.
        """
        from ...utils import append_mla_latent_cache

        manager = metadata.kv_cache_manager
        if getattr(manager, "use_fp8_ds_mla", False):
            kv_cache = manager.get_buffers(self.layer_idx, kv_layout=metadata.kv_layout)
            # Use the destination dtype so the helper's dtype conversion is a
            # no-op and preserves the packed FP8, scale, and RoPE bytes.
            latent_cache = self._pack_inline_scale_latent(latent_cache, kv_cache.dtype)

        return append_mla_latent_cache(
            manager,
            self.layer_idx,
            request_ids,
            seq_lens,
            past_lens,
            latent_cache,
            kv_layout=metadata.kv_layout,
        )

    @staticmethod
    def _load_latent_cache(
        kv_cache: torch.Tensor,
        block_ids: list[int],
        kv_len: int,
        kv_layout: str,
        *,
        use_fp8_ds_mla: bool = False,
    ) -> torch.Tensor:
        """Read the first ``kv_len`` latent rows of one request from the paged cache.

        Args:
            kv_cache: Layer buffer [num_blocks, 1, tokens_per_block, 1, D] (NHD)
                or [num_blocks, 1, 1, tokens_per_block, D] (HND).
            block_ids: The request's block table; BAD_PAGE_INDEX entries are dropped.
            kv_len: Number of rows to return.
            kv_layout: "NHD" or "HND".
            use_fp8_ds_mla: Decode inline-scale rows to float32.

        Returns:
            [kv_len, D] rows in the cache dtype, or float32 [kv_len, 576] rows
            for ``fp8_ds_mla``.
        """
        if kv_layout == "NHD":
            cache_rows = kv_cache[:, 0, :, 0, :]
        elif kv_layout == "HND":
            cache_rows = kv_cache[:, 0, 0, :, :]
        else:
            raise ValueError(f"Unsupported KV cache layout: {kv_layout}")
        tokens_per_block = cache_rows.shape[1]

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

        page_indices = torch.tensor(
            valid_block_ids[:num_required_blocks], dtype=torch.long, device=kv_cache.device
        )
        latent_cache = cache_rows.index_select(0, page_indices).flatten(0, 1)[:kv_len]
        if use_fp8_ds_mla:
            return DSAVanillaAttention._unpack_inline_scale_latent(latent_cache)
        return latent_cache

    def _forward_sparse(
        self,
        fused_q: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        latent_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        is_generation: bool,
        append_cache: bool,
    ) -> torch.Tensor:
        """Compute absorbed sparse MLA for one phase.

        For query token t, head h and the latent rows S_t that t selected::

            out[t, h] = softmax(q[t, h] @ S_t.T * scale) @ S_t[:, :kv_lora_rank]

        with ``scale = 1 / (sqrt(qk_nope_head_dim + qk_rope_head_dim) * q_scaling)``.

        Args:
            fused_q: [T_phase, num_heads * (kv_lora_rank + R)] absorbed queries
                with RoPE applied.
            metadata: Batch metadata.
            latent_cache: [T_phase, kv_lora_rank + R] rotated latent rows of
                this phase's new tokens.
            topk_indices: int32 [T_phase, topk] request-local positions; ``-1``
                pads unused slots.
            is_generation: Whether this is the generation phase.
            append_cache: Append ``latent_cache`` before reading, unless an
                earlier entry point already appended it.

        Returns:
            [T_phase, num_heads * kv_lora_rank] attention output in fused_q's dtype.
        """
        seq_start, seq_end, phase_seq_lens, phase_past_tokens = _phase_lens(metadata, is_generation)
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
        # Validate the selection: only -1 pads, every query keeps at least one
        # key, and no query selects a position after its own.
        valid_mask = topk_indices >= 0
        if torch.any(topk_indices < -1):
            raise ValueError("DSA top-k indices may only use -1 as padding")
        if torch.any(~valid_mask.any(dim=1)):
            raise ValueError("Every DSA query token must select at least one KV token")
        causal_limits = _token_positions(phase_seq_lens, phase_past_tokens, topk_indices.device)
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
            kv_len = phase_past_tokens[phase_idx] + q_len
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
        """Run DSA sparse MLA for one context-only or generation-only phase.

        Args:
            q: [T_phase, num_heads * (kv_lora_rank + R)] absorbed query. Its
                RoPE slots are written here when RoPE is deferred to this call.
            k: Must be None; MLA reads keys from the latent cache.
            v: Must be None; MLA reads values from the latent cache.
            metadata: Batch metadata with a KV cache manager.
            forward_args: Carries ``latent_cache``, the phase type, ``q_pe`` when
                RoPE is deferred here, and either the indexer projections or
                precomputed TopK in ``sparse_backend_args``.
            **kwargs: Keyword arguments merged into ``forward_args``.

        Returns:
            [T_phase, num_heads * kv_lora_rank] attention output. As a side
            effect, ``forward_args.sparse_runtime_params.sparse_attn_indices``
            receives the primary-pool TopK rows that the TRTLLM kernels consume.
        """
        forward_args = merge_attention_forward_args(forward_args, kwargs)
        if metadata.multi_item_part_lens is not None:
            raise ValueError("DSA Vanilla attention does not support multi-item scoring")
        if metadata.kv_cache_manager is None:
            raise ValueError("DSA Vanilla attention requires a KV cache manager")
        if forward_args.latent_cache is None:
            raise ValueError("DSA Vanilla attention requires latent_cache")
        if k is not None or v is not None:
            raise ValueError("DSA Vanilla attention expects absorbed queries without K/V")
        is_generation = _phase_is_generation(forward_args.attention_input_type)

        # Ordinary generation RoPE already ran via mla_rope_generation. Context
        # RoPE runs in the attention kernel, while fp8_ds_mla generation defers
        # it to the FlashInfer FMHA; reproduce both deferred cases here. The
        # test-only skip flag means the caller supplied already-rotated inputs.
        use_fp8_ds_mla = getattr(metadata.kv_cache_manager, "use_fp8_ds_mla", False)
        apply_deferred_rope = not is_generation or use_fp8_ds_mla
        if apply_deferred_rope and not forward_args.skip_mla_rope_generation:
            if forward_args.q_pe is None:
                raise ValueError("DSA Vanilla fused RoPE requires forward_args.q_pe")
            _, _, seq_lens, past_lens = _phase_lens(metadata, is_generation)
            self._apply_mla_rope(
                q,
                forward_args.q_pe,
                forward_args.latent_cache,
                _token_positions(seq_lens, past_lens, q.device),
            )

        # Ordinary generation reached here after mla_rope_generation already
        # appended the rotated latent.  The standalone test path sets the skip
        # flag and supplies pre-rotated tensors, so it still needs this forward
        # to append them. Context and fp8_ds_mla generation also append here.
        cache_already_appended = (
            is_generation and not use_fp8_ds_mla and not forward_args.skip_mla_rope_generation
        )

        local_topk, sparse_attn_indices = self._predict_topk(q, k, metadata, forward_args)
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
            is_generation,
            append_cache=not cache_already_appended,
        )


__all__ = ["DSAVanillaAttention", "DSAVanillaEmbedding", "DSAVanillaIndexer"]
