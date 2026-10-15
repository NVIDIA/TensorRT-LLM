# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch golden reference for DeepSeek-V4 sparse MLA.

DeepSeek-V4 attends each query token, at absolute position p of its request,
to two key sets kept in separate paged pools:

1. Sliding window (SWA pool): the request's latent rows at positions
   max(0, p - W + 1) through p.
2. Compressed history (COMPRESS pool), chosen by the layer's compression
   ratio r. Entry e summarizes raw tokens [e * r, (e + 1) * r) and is visible
   to the token when e < (p + 1) // r.

   - r = 1: no compressed keys.
   - r = 4: the visible entries this layer's indexer selected (top-k).
   - r = 128: every visible entry.

Keys and values are both the full latent rows, RoPE slice included
(``rope_append=False``), and an optional per-head attention sink adds one extra
logit to every softmax denominator.

:class:`DeepseekV4VanillaAttention` reproduces the backend's share of that
pipeline with plain torch operations: RoPE on the query and the new latent
rows, the SWA append, and attention over both key sets. It never dispatches the
fused kernels it validates, so a kernel bug cannot hide behind an identical
golden. Coverage limits:

- The compressor that fills the COMPRESS pool and the ratio-4 indexer are the
  production modules. The backend builds both exactly as
  DeepseekV4TrtllmAttention does, so the DeepSeek-V4 module and checkpoint
  loading find them, but never calls them; in model runs they run their
  production kernels. Tests that prefill the COMPRESS pool and inject top-k
  exercise only this reference.
- Below SM100 the DeepSeek-V4 module computes attention with FlashMLA and
  never calls ``forward``, so the constructor rejects those GPUs.
- Only token-major floating-point pools (BF16 in production) are read: FP8
  per-tensor and NVFP4 KV caches, the ``fp8_ds_mla`` footer-scale layout and
  KV-cache offload are rejected. The reference synchronizes with the host, so
  it is eager-only.

Shapes and coordinates use this notation throughout:

- T: packed tokens of one phase. Context requests come first, then
  generation requests; a "phase" is one of those two slices.
- H: query heads on this rank. D = kv_lora_rank + R: latent width, where R is
  the RoPE width. W: sliding-window size.
- p: absolute position of a token in its request; cached tokens come first.
- SWA addressing: position p lies in page
  ``sliding_block_tables[layer, SWA, request, p // tokens_per_block]`` at row
  ``p % tokens_per_block``.
- COMPRESS addressing: entry e lies in page
  ``compress_block_tables[r][request, e // tokens_per_page]`` at row
  ``e % tokens_per_page``, with ``tokens_per_page = tokens_per_block // r``.
"""

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    MLAParams,
    PositionalEmbeddingParams,
    merge_attention_forward_args,
)
from tensorrt_llm._torch.attention.backends.vanilla import VanillaAttention
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.models.modeling_utils import QuantConfig

from .._vanilla_utils import (
    VanillaRotaryEmbedding,
    apply_mla_rope,
    phase_is_generation,
    phase_lens,
    token_positions,
)
from ..params import SparseBackendForwardArgs
from .compressor import Compressor
from .indexer import DeepseekV4Indexer
from .metadata import DeepseekV4TrtllmAttentionMetadata
from .params import (
    DEEPSEEK_V4_SPARSE_RATIO,
    DeepseekV4AttentionType,
    DeepSeekV4Params,
    is_compress_layer,
    is_sparse_layer,
)

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig

    from .cache_manager import DeepseekV4CacheManager

# Ratios DeepSeek-V4 metadata supports: window only, indexer-selected entries,
# and every visible entry.
_SUPPORTED_COMPRESS_RATIOS = (1, DEEPSEEK_V4_SPARSE_RATIO, 128)
# RMSNorm epsilon DeepseekV4TrtllmAttention builds its compressor with.
_COMPRESSOR_RMS_NORM_EPS = 1e-6


def _check_kv_cache_quant(quant_config: Optional[QuantConfig]) -> None:
    """Reject the quantized KV caches this reference cannot read."""
    if quant_config is None:
        return
    quant_mode = quant_config.layer_quant_mode
    if quant_mode.has_fp8_kv_cache() or quant_mode.has_fp4_kv_cache():
        raise ValueError(
            "DeepSeek-V4 Vanilla attention supports only BF16/FP16 KV caches; FP8 "
            "per-tensor and NVFP4 pools need dequantization it does not reproduce"
        )


def _paged_addresses(
    block_table: torch.Tensor,
    indices: torch.Tensor,
    cache: torch.Tensor,
    *,
    pool: str,
    seq_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map ascending request-local indices to (page, row) addresses of a pool.

    Index i lies in page ``block_table[i // tokens_per_page]`` at row
    ``i % tokens_per_page``, where ``tokens_per_page = cache.shape[1]``.

    Args:
        block_table: int32 [max_blocks] page table of one request.
        indices: int64 [N] ascending request-local positions or entries.
        cache: [num_pages, tokens_per_page, D] pool the addresses point into.
        pool: Pool name for error messages.
        seq_idx: Batch index of the request, for error messages.

    Returns:
        int64 [N] pages and int64 [N] rows.

    Raises:
        ValueError: If an index lies past the block table or maps to an
            invalid page. The production kernels never read such rows for
            valid metadata, so either means the metadata is inconsistent.
    """
    if indices.numel() == 0:
        return indices, indices
    tokens_per_page = cache.shape[1]
    blocks = indices // tokens_per_page
    if int(blocks[-1]) >= block_table.shape[0]:
        raise ValueError(
            f"DeepSeek-V4 {pool} index {int(indices[-1])} of request {seq_idx} lies "
            f"past its {block_table.shape[0]}-page block table"
        )
    pages = block_table.index_select(0, blocks).long()
    invalid = (pages < 0) | (pages >= cache.shape[0])
    if bool(invalid.any()):
        raise ValueError(
            f"DeepSeek-V4 {pool} index {int(indices[invalid][0])} of request {seq_idx} "
            f"maps to invalid page {int(pages[invalid][0])}"
        )
    return pages, indices % tokens_per_page


@dataclass(frozen=True)
class _RequestRows:
    """Validated cache addresses one request reads or writes in a phase.

    Attributes:
        seq_idx: Batch index of the request.
        past_len: Cached tokens, i.e. the position of the first new token.
        num_new: New tokens in this phase.
        window_start: Lowest position any new token attends,
            ``max(0, past_len - W + 1)``.
        swa_pages: int64 [past_len + num_new - window_start] SWA pages of
            positions ``window_start`` through the last new token.
        swa_rows: int64 rows matching ``swa_pages``.
        compressed_pages: int64 [E] COMPRESS pages of entries 0..E-1, the
            entries visible to the last new token; empty when r = 1.
        compressed_rows: int64 rows matching ``compressed_pages``.
    """

    seq_idx: int
    past_len: int
    num_new: int
    window_start: int
    swa_pages: torch.Tensor
    swa_rows: torch.Tensor
    compressed_pages: torch.Tensor
    compressed_rows: torch.Tensor


@dataclass(frozen=True)
class _PhasePlan:
    """Validated addressing of one phase, computed before any cache write.

    Attributes:
        requests: Per-request addresses, in packed order.
        positions: int32 [T] absolute positions of the phase's tokens.
        swa_cache: [num_pages, tokens_per_block, D] SWA pool of this layer.
        compressed_cache: [num_pages, tokens_per_block // r, D] COMPRESS pool
            of this layer, or None when r = 1.
    """

    requests: list[_RequestRows]
    positions: torch.Tensor
    swa_cache: torch.Tensor
    compressed_cache: Optional[torch.Tensor]

    @property
    def num_tokens(self) -> int:
        return self.positions.numel()


class DeepseekV4VanillaAttention(VanillaAttention):
    """Standalone PyTorch golden for DeepSeek-V4 dual-pool sparse MLA.

    This is a drop-in replacement for :class:`DeepseekV4TrtllmAttention` on
    SM100 and newer: it keeps the same constructor, ``forward`` and
    ``mla_rope_generation`` signatures and owns the same ``indexer`` and
    ``compressor`` modules, and it reports ``support_fused_rope() == True`` so
    the DeepSeek-V4 module leaves RoPE to the backend, exactly as with TRTLLM.
    For each context-only or generation-only phase it:

    1. Rotates the query and the new latent rows and appends the rows to the
       SWA pool (``_rotate_and_append``): inside ``forward`` for context, and
       in ``mla_rope_generation``, which the module calls first, for
       generation.
    2. Gathers each token's sliding window from the SWA pool and its visible
       or selected entries from the COMPRESS pool (``_attend``).
    3. Attends over them with the attention sink (``_attend``).

    Queries use the layout [T, H * D] with each head's RoPE slice last, a
    latent row is [D] = [compressed KV | k_pe], and the output is [T, H * D].
    In both phases the query's RoPE slots are rotated in place, while the
    caller's latent rows stay unrotated: like the kernels, which read them
    through a const pointer, the reference rotates a private copy and appends
    that. The module applies the inverse RoPE to the output's RoPE slice.
    """

    Metadata = DeepseekV4TrtllmAttentionMetadata

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
        sparse_attention_config: Optional["SparseAttentionConfig"] = None,
        sparse_params: Optional[DeepSeekV4Params] = None,
        dtype: Optional[torch.dtype] = None,
        aux_stream: Optional[torch.cuda.Stream] = None,
        **kwargs,
    ) -> None:
        """Build the backend and the production indexer and compressor it owns.

        Args:
            layer_idx: Global layer index.
            num_heads: Query heads on this rank.
            head_dim: Latent width D = kv_lora_rank + R.
            num_kv_heads: Forwarded to the base backend; MLA uses one latent head.
            quant_config: Checkpoint quantization, also passed to the indexer.
                FP8 and NVFP4 KV caches are rejected.
            q_scaling: Extra divisor of the softmax scale.
            pos_embd_params: RoPE configuration; required.
            mla_params: MLA dimensions; required. As in
                DeepseekV4TrtllmAttention, ``v_head_dim`` is replaced by
                ``head_dim`` and ``rope_append`` by False.
            skip_create_weights_in_init: Defer the indexer and compressor weights.
            attention_chunk_size: Forwarded to the base backend.
            sparse_attention_config: User-facing DeepSeek-V4 config. Defaults to
                ``sparse_params``, which carries the fields read here.
            sparse_params: Lowered DeepSeek-V4 parameters; derived from
                ``sparse_attention_config`` when omitted.
            dtype: Model dtype of the indexer and compressor weights.
            aux_stream: Passed to the indexer, as DeepseekV4TrtllmAttention does.
            **kwargs: Forwarded to :class:`VanillaAttention`. ``kv_cache_dtype``
                must not be ``"fp8_ds_mla"``.

        Raises:
            ValueError: For a configuration this reference does not reproduce.
        """
        if sparse_attention_config is None:
            sparse_attention_config = sparse_params
        if sparse_attention_config is None:
            raise ValueError(
                "DeepseekV4VanillaAttention requires sparse_attention_config or sparse_params"
            )
        if sparse_params is None:
            sparse_params = sparse_attention_config.to_sparse_params()
        if mla_params is None:
            raise ValueError("DeepseekV4VanillaAttention requires MLA parameters")
        if pos_embd_params is None:
            raise ValueError("DeepseekV4VanillaAttention requires RoPE parameters")
        if kwargs.get("kv_cache_dtype") == "fp8_ds_mla":
            raise ValueError("DeepseekV4VanillaAttention does not read the fp8_ds_mla KV cache")
        if sparse_attention_config.enable_kv_cache_offload:
            raise ValueError("DeepseekV4VanillaAttention does not support KV-cache offload")
        _check_kv_cache_quant(quant_config)
        sm_version = get_sm_version()
        if sm_version < 100:
            raise ValueError(
                f"DeepseekV4VanillaAttention requires SM100 or newer, got SM{sm_version}: "
                "below SM100 the DeepSeek-V4 module computes attention with FlashMLA "
                "and never calls this backend"
            )
        compress_ratios = sparse_attention_config.compress_ratios
        if layer_idx >= len(compress_ratios):
            raise ValueError(
                f"DeepSeek-V4 layer {layer_idx} has no compression ratio in a "
                f"{len(compress_ratios)}-layer configuration"
            )
        compress_ratio = compress_ratios[layer_idx]
        if compress_ratio not in _SUPPORTED_COMPRESS_RATIOS:
            raise ValueError(
                f"DeepSeek-V4 supports compression ratios {_SUPPORTED_COMPRESS_RATIOS}, "
                f"got {compress_ratio}"
            )

        mla_params = replace(mla_params, v_head_dim=head_dim, rope_append=False)
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
        self.sparse_attention_config = sparse_attention_config
        self.compress_ratio = compress_ratio
        self.window_size = sparse_attention_config.window_size
        # The checkpoint loader installs a per-head float32 [H] sink here.
        self.attn_sink: Optional[torch.Tensor] = None
        self.rotary_emb = VanillaRotaryEmbedding(
            pos_embd_params.rope,
            head_dim=self.qk_rope_head_dim,
            is_neox=pos_embd_params.is_neox,
        )
        # The DeepSeek-V4 module takes these from the backend and calls them
        # itself; checkpoints load self_attn.indexer.* and self_attn.compressor.*.
        self.indexer: Optional[DeepseekV4Indexer] = None
        if is_sparse_layer(compress_ratio):
            self.indexer = DeepseekV4Indexer(
                quant_config,
                pos_embd_params,
                mla_params,
                skip_create_weights_in_init,
                sparse_params,
                dtype,
                compress_ratio,
                layer_idx,
                aux_stream,
            )
        self.compressor: Optional[Compressor] = None
        if is_compress_layer(compress_ratio):
            self.compressor = Compressor(
                mla_params,
                layer_idx,
                compress_ratio,
                _COMPRESSOR_RMS_NORM_EPS,
                skip_create_weights_in_init,
                pos_embd_params,
                kv_cache_dtype="default",
                dtype=dtype,
                rotate_activation=False,
            )

    @classmethod
    def support_fused_rope(cls) -> bool:
        """Keep Q/K RoPE inside the backend, as DeepseekV4TrtllmAttention does."""
        return True

    def update_quant_config(self, new_quant_config: Optional[QuantConfig]) -> None:
        """Adopt ``new_quant_config`` after rejecting a quantized KV cache."""
        _check_kv_cache_quant(new_quant_config)
        super().update_quant_config(new_quant_config)

    def mla_rope_generation(
        self,
        fused_q: Optional[torch.Tensor],
        q_pe: Optional[torch.Tensor],
        latent_cache: torch.Tensor,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        fmha_scheduler_counter: torch.Tensor,
        mla_bmm1_scale: Optional[torch.Tensor],
        mla_bmm2_scale: Optional[torch.Tensor],
        quant_q_buffer: Optional[torch.Tensor],
        out_scale: Optional[torch.Tensor] = None,
        kv_norm_weight: Optional[torch.Tensor] = None,
        kv_norm_eps: float = 1e-6,
        precomputed_cu_seqlens: bool = False,
        precomputed_fmha_scheduler: bool = False,
        kv_only: bool = False,
        kv_done_elsewhere: bool = False,
        quant_scale_qkv: Optional[torch.Tensor] = None,
        fuse_fp4_q_quant: bool = False,
    ) -> None:
        """Torch version of the generation RoPE and SWA-append kernel.

        Rotates ``q_pe`` into each head's RoPE slot of ``fused_q`` and appends a
        rotated copy of ``latent_cache`` to the SWA pool at the generation
        tokens' positions. ``latent_cache`` itself is left unchanged because
        the kernel reads it through a const pointer. The DeepSeek-V4 module
        calls this before ``forward(generation_only)``, which neither rotates
        nor appends again.

        Args:
            fused_q: [T, H * D] generation queries; RoPE slots written in place.
            q_pe: [T, H, R] or [T, H * R] unrotated query RoPE part; may alias
                the RoPE slots of ``fused_q``.
            latent_cache: [T, D] unrotated latent rows of the new tokens.
            metadata: Batch metadata with the DeepSeek-V4 cache manager.
            cu_q_seqlens: TRTLLM FMHA scheduler state; left untouched.
            cu_kv_seqlens: TRTLLM FMHA scheduler state; left untouched.
            fmha_scheduler_counter: TRTLLM FMHA scheduler state; left untouched.
            mla_bmm1_scale: TRTLLM FMHA scheduler state; left untouched.
            mla_bmm2_scale: TRTLLM FMHA scheduler state; left untouched.
            quant_q_buffer: Must be None; FP8 queries belong to the FP8 KV cache.
            out_scale: Must be None; FP8 output is not reproduced.
            kv_norm_weight: Must be None; the fused KV-norm path belongs to the
                FP8 KV cache.
            kv_norm_eps: Unused without ``kv_norm_weight``.
            precomputed_cu_seqlens: Ignored; this backend fills no FMHA state.
            precomputed_fmha_scheduler: Ignored, for the same reason.
            kv_only: Must be False (fused KV-norm path).
            kv_done_elsewhere: Must be False (fused KV-norm path).
            quant_scale_qkv: Must be None (fused FP8 query path).
            fuse_fp4_q_quant: Must be False (fused FP4 query path).

        Raises:
            ValueError: For a request this reference does not reproduce or for
                inconsistent inputs, before any tensor is written.
        """
        del (
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            kv_norm_eps,
            precomputed_cu_seqlens,
            precomputed_fmha_scheduler,
        )
        requested = [
            name
            for name, is_requested in (
                ("quant_q_buffer", quant_q_buffer is not None),
                ("out_scale", out_scale is not None),
                ("kv_norm_weight", kv_norm_weight is not None),
                ("quant_scale_qkv", quant_scale_qkv is not None),
                ("kv_only", kv_only),
                ("kv_done_elsewhere", kv_done_elsewhere),
                ("fuse_fp4_q_quant", fuse_fp4_q_quant),
            )
            if is_requested
        ]
        if requested:
            raise ValueError(
                f"DeepSeek-V4 Vanilla generation RoPE does not support {', '.join(requested)}: "
                "they belong to the fused quantized KV-cache and query paths, which this "
                "backend rejects"
            )
        if fused_q is None or q_pe is None:
            raise ValueError("DeepSeek-V4 Vanilla generation RoPE requires fused_q and q_pe")
        plan = self._phase_plan(metadata, is_generation=True, device=latent_cache.device)
        self._check_query_and_latent(fused_q, latent_cache, plan.num_tokens)
        self._rotate_and_append(fused_q, q_pe, latent_cache, plan)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: DeepseekV4TrtllmAttentionMetadata,
        forward_args: Optional[AttentionForwardArgs] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run DeepSeek-V4 sparse MLA for one context-only or generation-only phase.

        Every input is validated before the first in-place write. Context RoPE
        and the context SWA append happen here; generation expects
        ``mla_rope_generation`` to have done both, as in production.

        Args:
            q: [T, H * D] queries. Context queries arrive unrotated and their
                RoPE slots are rotated here, in place.
            k: Must be None; keys come from the latent pools.
            v: Must be None; values come from the latent pools.
            metadata: Batch metadata with the DeepSeek-V4 cache manager.
            forward_args: Carries the phase type, ``latent_cache`` ([T, D]
                unrotated new rows; context appends a rotated copy), ``q_pe``
                (context RoPE input), ``sparse_backend_args.topk_indices``
                (ratio-4 layers: int32 [T, K] request-local entries padded with
                -1), optional ``attention_sinks`` (float32 [H]; defaults to
                ``attn_sink``) and an optional ``output`` buffer.
            **kwargs: Keyword arguments merged into ``forward_args``.

        Returns:
            [T, H * D] attention output in ``q``'s dtype: ``forward_args.output``
            when given, otherwise a new tensor.

        Raises:
            ValueError: For unsupported arguments or inconsistent inputs.
        """
        forward_args = merge_attention_forward_args(forward_args, kwargs)
        if k is not None or v is not None:
            raise ValueError("DeepSeek-V4 Vanilla attention expects absorbed queries without K/V")
        if metadata.multi_item_part_lens is not None:
            raise ValueError("DeepSeek-V4 Vanilla attention does not support multi-item scoring")
        self._reject_unsupported_forward_args(forward_args)
        if forward_args.latent_cache is None:
            raise ValueError("DeepSeek-V4 Vanilla attention requires latent_cache")
        is_generation = phase_is_generation(forward_args.attention_input_type)
        plan = self._phase_plan(metadata, is_generation, device=q.device)
        self._check_query_and_latent(q, forward_args.latent_cache, plan.num_tokens)
        topk_indices = self._check_topk(forward_args.sparse_backend_args, plan, metadata)
        attention_sink = self._attention_sink(forward_args)
        output = forward_args.output
        expected_shape = (plan.num_tokens, self.num_heads * self.v_head_dim)
        if output is not None and (
            tuple(output.shape) != expected_shape or output.dtype != q.dtype
        ):
            raise ValueError(
                f"DeepSeek-V4 output buffer must be {q.dtype} {list(expected_shape)}, "
                f"got {output.dtype} {list(output.shape)}"
            )

        if not is_generation:
            if forward_args.q_pe is None:
                raise ValueError("DeepSeek-V4 Vanilla context RoPE requires forward_args.q_pe")
            self._rotate_and_append(q, forward_args.q_pe, forward_args.latent_cache, plan)

        result = self._attend(q, plan, topk_indices, attention_sink)
        if output is None:
            return result
        output.copy_(result)
        return output

    @staticmethod
    def _reject_unsupported_forward_args(forward_args: AttentionForwardArgs) -> None:
        """Reject forward arguments that ask for work this reference does not do.

        FP8 output, the fused FP8-query, KV-norm and O-LoRA epilogue paths
        belong to the FP8 KV cache; softmax statistics belong to Helix context
        parallelism, which DeepSeek-V4 rejects; and the test-only
        ``skip_mla_rope_generation`` contradicts backend-owned RoPE.
        """
        requested = [
            name
            for name, is_requested in (
                ("out_scale", forward_args.out_scale is not None),
                ("output_sf", forward_args.output_sf is not None),
                ("quant_q_buffer", forward_args.quant_q_buffer is not None),
                ("quant_scale_qkv", forward_args.quant_scale_qkv is not None),
                ("kv_norm_weight", forward_args.kv_norm_weight is not None),
                ("softmax_stats_tensor", forward_args.softmax_stats_tensor is not None),
                ("enable_dsv4_epilogue_fusion", forward_args.enable_dsv4_epilogue_fusion),
                ("skip_mla_rope_generation", forward_args.skip_mla_rope_generation),
            )
            if is_requested
        ]
        if requested:
            raise ValueError(
                f"DeepSeek-V4 Vanilla attention does not support {', '.join(requested)}"
            )

    def _phase_plan(
        self,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        is_generation: bool,
        device: torch.device,
    ) -> _PhasePlan:
        """Compute and validate every cache address one phase touches.

        Each new token reads its window from the SWA pool, so a request
        touches SWA positions ``max(0, past_len - W + 1)`` through its last new
        position, the new ones included; compressed layers also read every
        entry visible to the last new token.

        Args:
            metadata: Batch metadata with the DeepSeek-V4 cache manager.
            is_generation: Plan the generation phase instead of context.
            device: Device of the phase's tensors.

        Returns:
            The phase plan.

        Raises:
            ValueError: If the metadata does not match this layer, a needed row
                lies outside its request's pages, or a CUDA graph is being
                captured: planning synchronizes with the host.
        """
        if torch.cuda.is_current_stream_capturing():
            raise ValueError(
                "DeepSeek-V4 Vanilla attention synchronizes with the host and cannot be "
                "captured in a CUDA graph"
            )
        manager = metadata.kv_cache_manager
        if manager is None:
            raise ValueError("DeepSeek-V4 Vanilla attention requires a KV cache manager")
        if metadata.window_size != self.window_size:
            raise ValueError(
                f"DeepSeek-V4 metadata window {metadata.window_size} does not match "
                f"layer window {self.window_size}"
            )
        seq_start, _, seq_lens, past_lens = phase_lens(metadata, is_generation)
        swa_cache = self._token_major_pool(manager, DeepseekV4AttentionType.SWA)
        local_layer_idx = manager.layer_offsets[self.layer_idx]
        swa_tables = metadata.sliding_block_tables[
            local_layer_idx, DeepseekV4AttentionType.SWA.value
        ]
        compressed_cache = None
        if is_compress_layer(self.compress_ratio):
            compressed_cache = self._token_major_pool(manager, DeepseekV4AttentionType.COMPRESS)
            compressed_tables = metadata.compress_block_tables[self.compress_ratio]

        requests = []
        for offset, (num_new, past_len) in enumerate(zip(seq_lens, past_lens, strict=True)):
            seq_idx = seq_start + offset
            end = past_len + num_new
            window_start = max(0, past_len - self.window_size + 1)
            swa_pages, swa_rows = _paged_addresses(
                swa_tables[seq_idx],
                torch.arange(window_start, end, dtype=torch.int64, device=swa_tables.device),
                swa_cache,
                pool="SWA",
                seq_idx=seq_idx,
            )
            compressed_pages = compressed_rows = swa_pages.new_empty(0)
            if compressed_cache is not None:
                compressed_pages, compressed_rows = _paged_addresses(
                    compressed_tables[seq_idx],
                    torch.arange(
                        end // self.compress_ratio,
                        dtype=torch.int64,
                        device=compressed_tables.device,
                    ),
                    compressed_cache,
                    pool="COMPRESS",
                    seq_idx=seq_idx,
                )
            requests.append(
                _RequestRows(
                    seq_idx=seq_idx,
                    past_len=past_len,
                    num_new=num_new,
                    window_start=window_start,
                    swa_pages=swa_pages,
                    swa_rows=swa_rows,
                    compressed_pages=compressed_pages,
                    compressed_rows=compressed_rows,
                )
            )
        return _PhasePlan(
            requests=requests,
            positions=token_positions(seq_lens, past_lens, device),
            swa_cache=swa_cache,
            compressed_cache=compressed_cache,
        )

    def _token_major_pool(
        self, manager: "DeepseekV4CacheManager", attention_type: DeepseekV4AttentionType
    ) -> torch.Tensor:
        """Return this layer's [num_pages, tokens_per_page, D] pool of one role.

        Raises:
            ValueError: If the pool is not token-major with 16- or 32-bit
                floating-point rows of width D (FP8 per-tensor, NVFP4 and
                footer-scale pools are byte-packed).
        """
        cache = manager.get_buffers(self.layer_idx, attention_type)
        if (
            cache.ndim != 3
            or cache.shape[-1] != self.head_dim
            or not cache.dtype.is_floating_point
            or cache.element_size() < 2
        ):
            raise ValueError(
                f"DeepSeek-V4 Vanilla attention reads token-major floating-point "
                f"{attention_type.name} pools of width {self.head_dim}, got "
                f"{cache.dtype} {list(cache.shape)}"
            )
        return cache

    def _check_query_and_latent(
        self, q: torch.Tensor, latent_cache: torch.Tensor, num_tokens: int
    ) -> None:
        """Check the [T, H * D] query and [T, D] latent shapes of a phase."""
        expected_q = (num_tokens, self.num_heads * self.head_dim)
        if tuple(q.shape) != expected_q:
            raise ValueError(
                f"DeepSeek-V4 query must have shape {list(expected_q)}, got {list(q.shape)}"
            )
        if tuple(latent_cache.shape) != (num_tokens, self.head_dim):
            raise ValueError(
                f"DeepSeek-V4 latent cache must have shape [{num_tokens}, {self.head_dim}], "
                f"got {list(latent_cache.shape)}"
            )

    def _check_topk(
        self,
        sparse_backend_args: Optional[SparseBackendForwardArgs],
        plan: _PhasePlan,
        metadata: DeepseekV4TrtllmAttentionMetadata,
    ) -> Optional[torch.Tensor]:
        """Return the validated ratio-4 selection, or None on other layers.

        Like the production kernels, only ratio-4 layers read
        ``sparse_backend_args.topk_indices``. Row t lists request-local
        compressed entries of the token at position p; with
        ``visible = (p + 1) // 4``, the sparse kernel reads its first
        ``min(visible, K)`` columns. A row must therefore keep its valid
        entries in those columns, and every entry must be visible.

        Returns:
            int32 [T, K] top-k indices, or None when r != 4.

        Raises:
            ValueError: For a missing, malformed or out-of-contract selection.
        """
        if not is_sparse_layer(self.compress_ratio):
            return None
        topk = None if sparse_backend_args is None else sparse_backend_args.topk_indices
        if topk is None:
            raise ValueError(
                "DeepSeek-V4 ratio-4 layers require sparse_backend_args.topk_indices "
                "from the indexer"
            )
        if topk.ndim != 2 or topk.shape[0] != plan.num_tokens:
            raise ValueError(
                f"DeepSeek-V4 top-k indices must have shape [{plan.num_tokens}, top_k], "
                f"got {list(topk.shape)}"
            )
        if topk.dtype != torch.int32:
            raise ValueError(f"DeepSeek-V4 top-k indices must be int32, got {topk.dtype}")
        width = metadata.max_compressed_indices[self.compress_ratio]
        if topk.shape[1] > width:
            raise ValueError(
                f"DeepSeek-V4 top-k has {topk.shape[1]} columns, but the metadata sizes "
                f"the compressed index region to {width}"
            )
        if bool((topk < -1).any()):
            raise ValueError("DeepSeek-V4 top-k indices may only use -1 as padding")
        valid = topk >= 0
        visible = (plan.positions.to(device=topk.device, dtype=torch.int64) + 1) // (
            self.compress_ratio
        )
        if bool((valid & (topk.long() >= visible.unsqueeze(1))).any()):
            raise ValueError(
                "DeepSeek-V4 top-k index selects an unavailable or future compressed entry"
            )
        columns = torch.arange(topk.shape[1], device=topk.device)
        readable = visible.clamp(max=topk.shape[1]).unsqueeze(1)
        if bool((valid & (columns >= readable)).any()):
            raise ValueError(
                "DeepSeek-V4 top-k rows must list their entries in the first "
                "min((p + 1) // 4, top_k) columns, the ones the sparse kernel reads"
            )
        return topk

    def _attention_sink(self, forward_args: AttentionForwardArgs) -> Optional[torch.Tensor]:
        """Return the float32 [H] attention sink, defaulting to ``attn_sink``.

        DeepseekV4TrtllmAttention passes its ``attn_sink`` to the kernel when
        the caller supplies none; the reference does the same.
        """
        sink = forward_args.attention_sinks
        if sink is None:
            sink = self.attn_sink
        if sink is None:
            return None
        if tuple(sink.shape) != (self.num_heads,):
            raise ValueError(
                f"DeepSeek-V4 attention sink must have shape [{self.num_heads}], "
                f"got {list(sink.shape)}"
            )
        return sink.detach()

    def _rotate_and_append(
        self,
        fused_q: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        plan: _PhasePlan,
    ) -> None:
        """Apply RoPE to a phase and append its latent rows to the SWA pool.

        The query's RoPE slots are rotated in place; ``latent_cache`` is
        rotated in a private copy, which is what gets stored. Every new row is
        stored, as production does, so the window of any later token, including
        tokens kept after a speculative rewind, can be read back from the pool.

        Args:
            fused_q: [T, H * D] queries; RoPE slots written in place.
            q_pe: [T, H, R] or [T, H * R] unrotated query RoPE part.
            latent_cache: [T, D] unrotated latent rows in packed token order.
            plan: The phase plan; its SWA addresses end with the new positions.
        """
        rotated_latent = latent_cache.clone()
        apply_mla_rope(
            self.rotary_emb,
            fused_q,
            q_pe,
            rotated_latent,
            plan.positions,
            num_heads=self.num_heads,
            kv_lora_rank=self.kv_lora_rank,
        )
        pages = torch.cat([r.swa_pages[r.swa_pages.numel() - r.num_new :] for r in plan.requests])
        rows = torch.cat([r.swa_rows[r.swa_rows.numel() - r.num_new :] for r in plan.requests])
        plan.swa_cache[pages, rows] = rotated_latent.to(plan.swa_cache.dtype)

    def _attend(
        self,
        q: torch.Tensor,
        plan: _PhasePlan,
        topk_indices: Optional[torch.Tensor],
        attention_sink: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute dual-pool selected MLA for every token of a phase.

        For the token at position p, with S_p its window rows followed by its
        compressed rows (visible entries for r = 128, the selected ones for
        r = 4, none for r = 1), and per head h::

            logits = q[t, h] @ S_p.T * scale
            out[t, h] = softmax([logits, sink[h]])[:-1] @ S_p

        with ``scale = 1 / (sqrt(qk_nope_head_dim + qk_rope_head_dim) * q_scaling)``,
        the sink column present only when a sink is given. The rows are read
        from the pools after the phase's append.

        Args:
            q: [T, H * D] rotated queries.
            plan: Validated phase plan.
            topk_indices: int32 [T, K] selection on ratio-4 layers, else None.
            attention_sink: float32 [H] sink logits, or None.

        Returns:
            [T, H * D] output in ``q``'s dtype.
        """
        if plan.num_tokens == 0:
            return q.new_empty((0, self.num_heads * self.v_head_dim))
        heads = q.view(plan.num_tokens, self.num_heads, self.head_dim)
        scale = 1.0 / (
            math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim) * (self.q_scaling or 1.0)
        )
        outputs = []
        token = 0
        for request in plan.requests:
            window = plan.swa_cache[request.swa_pages, request.swa_rows].to(q.dtype)
            compressed = None
            if plan.compressed_cache is not None:
                compressed = plan.compressed_cache[
                    request.compressed_pages, request.compressed_rows
                ].to(q.dtype)
            for position in range(request.past_len, request.past_len + request.num_new):
                first = max(0, position - self.window_size + 1) - request.window_start
                keys = [window[first : position - request.window_start + 1]]
                if topk_indices is not None:
                    row = topk_indices[token]
                    keys.append(compressed.index_select(0, row[row >= 0].long()))
                elif compressed is not None:
                    keys.append(compressed[: (position + 1) // self.compress_ratio])
                outputs.append(
                    self._selected_mla_attention(
                        heads[token],
                        torch.cat(keys),
                        value_dim=self.v_head_dim,
                        scale=scale,
                        attention_sink=attention_sink,
                    )
                )
                token += 1
        return torch.stack(outputs).reshape(plan.num_tokens, self.num_heads * self.v_head_dim)


__all__ = ["DeepseekV4VanillaAttention"]
