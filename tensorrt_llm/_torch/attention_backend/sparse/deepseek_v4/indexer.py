# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 indexer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from tensorrt_llm._torch.attention_backend.interface import MLAParams, PositionalEmbeddingParams
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.multi_stream_utils import do_multi_stream
from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding
from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.utils import fp8_utils

from ..dsa.indexer import HAS_FAST_HADAMARD, Indexer, rotate_activation
from .compressor import Compressor, KVCacheDtype, resolve_kv_cache_dtype

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig

    from ..dsa.metadata import DSAtrtllmAttentionMetadata
    from .metadata import DeepseekV4TrtllmAttentionMetadata


class DeepseekV4Indexer(Indexer):
    def __init__(
        self,
        quant_config: Optional[QuantConfig],
        pos_embd_params: Optional[PositionalEmbeddingParams],
        mla_params: Optional[MLAParams],
        skip_create_weights_in_init: bool,
        sparse_attention_config: "SparseAttentionConfig",
        dtype: Optional[torch.dtype],
        compress_ratio: int = 1,
        layer_idx: int = 0,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__(
            quant_config,
            pos_embd_params,
            mla_params,
            skip_create_weights_in_init,
            sparse_attention_config,
            dtype,
            compress_ratio,
            layer_idx,
            aux_stream,
        )
        # Keep the checkpoint's FP8 quantization while deriving the scale view
        # consumed by the fused CuTe DSL indexer-Q projection.
        self.wq_b.use_indexer_q_cutedsl_fusion = True
        # Override base Indexer.weights_proj to bf16 (matches V4 checkpoint).
        self.weights_proj = Linear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            dtype=dtype,
            quant_config=None,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True,
        )
        self.rotary_emb = RotaryEmbedding(
            pos_embd_params.rope,
            head_dim=self.rope_dim,
            is_neox=False,
        )
        rms_norm_eps = 1e-6
        index_head_dim = sparse_attention_config.index_head_dim
        indexer_mla_params = MLAParams(
            hidden_size=mla_params.hidden_size,
            qk_rope_head_dim=mla_params.qk_rope_head_dim,
            qk_nope_head_dim=index_head_dim - mla_params.qk_rope_head_dim,
        )
        # Map the user-facing FP4 knob ("fp8" / "fp4") onto the Compressor's
        # cache layout preset string. The Compressor's preset namespace also
        # covers main-attention layouts (bf16 / fp8_pertensor), which the
        # indexer doesn't use, so the translation lives here at the
        # boundary instead of leaking through the user-facing config.
        self.indexer_k_dtype = sparse_attention_config.indexer_k_dtype
        compressor_preset = "mxfp4" if self.indexer_k_dtype == "fp4" else "fp8_blockwise"
        self.indexer_cache_dtype = resolve_kv_cache_dtype(compressor_preset)
        self.compressor = Compressor(
            indexer_mla_params,
            layer_idx,
            compress_ratio,
            rms_norm_eps,
            skip_create_weights_in_init,
            pos_embd_params,
            dtype=dtype,
            kv_cache_dtype=compressor_preset,
            is_indexer=True,
            rotate_activation=HAS_FAST_HADAMARD,
        )
        self.indexer_start_event = torch.cuda.Event()
        self.weights_proj_event = torch.cuda.Event()
        self.k_cache_update_event = torch.cuda.Event()

    def post_load_weights(self):
        # V4 does not use the V3 fused fp32 wk+weights_proj GEMM, and the
        # base concat would now hit an fp32/bf16 dtype mismatch.
        return

    def _qk_projection_and_rope(self, qr: torch.Tensor, position_ids: torch.Tensor):
        """Project Q and apply RoPE.

        Returns q with layout [num_tokens, n_heads, head_dim] where
        head_dim = nope_dim + rope_dim, RoPE already applied in-place.
        """
        q = self.wq_b(qr)
        q = q.view(-1, self.n_heads, self.head_dim)
        return self._apply_q_rope(q, position_ids)

    def _apply_q_rope(self, q: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Apply RoPE in-place to a projected indexer Q tensor."""
        # Fused in-place RoPE on the rope portion of each head
        nope_dim = self.head_dim - self.rope_dim
        torch.ops.trtllm.mla_rope_inplace(
            q,
            position_ids.view(-1),
            self.rotary_emb.rotary_cos_sin,
            self.n_heads,
            nope_dim,
            self.rope_dim,
            False,
            self.rotary_emb.is_neox,
        )
        return q

    def _project_and_quantize_q(
        self, qr: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project and quantize Q, using the fused MXFP4 path when supported."""
        use_fused_project_mxfp4 = (
            self.indexer_cache_dtype == KVCacheDtype.MXFP4_BLOCKWISE
            and not HAS_FAST_HADAMARD
            and not self.rotary_emb.is_neox
            and self.head_dim == 128
            and self.rope_dim == 64
            and self.wq_b.has_fp8_block_scales
            and hasattr(self.wq_b, "indexer_q_weight_scale_cutedsl")
            and hasattr(
                torch.ops.trtllm,
                "cute_dsl_fp8_indexer_q_gemm_rope_fp4_blackwell",
            )
            and qr.dtype == torch.bfloat16
            and is_sm_100f()
        )
        if use_fused_project_mxfp4:
            q_fp4, q_scale = torch.ops.trtllm.cute_dsl_fp8_indexer_q_gemm_rope_fp4_blackwell(
                qr,
                self.wq_b.weight,
                self.wq_b.indexer_q_weight_scale_cutedsl,
                position_ids.view(-1),
                self.rotary_emb.rotary_cos_sin.view(-1, self.rope_dim),
                self.wq_b.indexer_q_alpha_cutedsl,
                use_tvm_ffi=True,
            )
            return q_fp4.view(-1, self.n_heads, self.head_dim // 2), q_scale.view(
                -1, self.n_heads, 1
            )

        q = self.wq_b(qr).view(-1, self.n_heads, self.head_dim)
        q = self._apply_q_rope(q, position_ids)
        return self._quantize_q(q)

    def _quantize_q(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Rotate + quantize (layout matches compressor K: [nope|pe]). After
        # rotate_activation (Hadamard) the nope/rope split becomes a linear
        # mix; treating the row as a (head_dim - rope_dim, rope_dim) split
        # for fused_cat_fp4 is equivalent to a single per-token FP4 quantize
        # because the kernel only cares about the concatenated row. K goes
        # through the same rotation in compressor_postprocess_scatter so
        # layouts match.
        q = rotate_activation(q)
        q = q.view(-1, self.head_dim)
        if self.indexer_cache_dtype == KVCacheDtype.MXFP4_BLOCKWISE:
            nope_dim = self.head_dim - self.rope_dim
            q_nope, q_pe = q.split([nope_dim, self.rope_dim], dim=-1)
            q_fp8, q_scale = torch.ops.trtllm.fused_cat_fp4(q_nope, q_pe)
            # Two FP4 codes pack into one byte: trailing dim is head_dim // 2.
            q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim // 2)
            q_scale = q_scale.view(-1, self.n_heads, 1)
        else:
            q_fp8, q_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(
                q, use_ue8m0=self.scale_fmt == "ue8m0"
            )
            q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim)
            q_scale = q_scale.view(-1, self.n_heads, 1)
        return q_fp8, q_scale

    def _apply_weight_scale(self, weights: torch.Tensor, q_scale: torch.Tensor) -> torch.Tensor:
        # The DeepGEMM FP4 kernel applies per-block q_scale internally, so
        # weights only carry softmax_scale * n_heads^-0.5. `weights_proj` is
        # bf16 to match the V4 checkpoint, and the FP4 branch's only
        # post-projection multiplier is a Python float (`bf16 * float` stays
        # bf16). DeepGEMM's fp8_fp4_(paged_)mqa_logits asserts
        # `weights.scalar_type() == kFloat`, so cast explicitly here. The FP8
        # branch gets upcast for free via the fp32 `q_scale` multiply in
        # `_weight_scale`, so no cast is needed there.
        if self.indexer_cache_dtype == KVCacheDtype.MXFP4_BLOCKWISE:
            return weights.float() * self.weight_scale_factor
        return self._weight_scale(weights, q_scale)

    def _update_k_cache_if_needed(
        self,
        k_fp8: Optional[torch.Tensor],
        k_scale: Optional[torch.Tensor],
        metadata: DeepseekV4TrtllmAttentionMetadata,
    ) -> None:
        if k_fp8 is None:
            return

        # The MXFP4 compressor's postprocess_scatter performs rotation +
        # quantization + cache scatter in one fused op, so the cache row is
        # already up to date by the time we get here.
        if self.indexer_cache_dtype == KVCacheDtype.MXFP4_BLOCKWISE:
            return

        assert k_scale is not None, "FP8 blockwise indexer cache update requires scale tensor"
        self._update_k_cache(k_fp8, k_scale, metadata)

    def precompute_aux(
        self,
        hidden_states: torch.Tensor,
        metadata: DeepseekV4TrtllmAttentionMetadata,
    ) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """Pre-launch the qr-independent half of the indexer prepare phase.

        Runs weights_proj + internal compressor + k_cache_update on
        ``self.aux_stream`` and records ``weights_proj_event`` /
        ``k_cache_update_event``.  The caller hands the returned tuple back to
        ``forward()`` via the ``pre_aux`` kwarg, which makes the overlapped
        prepare path skip its own aux-stream launch and consume these results
        directly.

        Returns ``None`` when multi-stream mode is off (caller should fall
        back to the normal ``forward()`` call without ``pre_aux``).
        """
        if not (do_multi_stream() and self.aux_stream is not None):
            return None
        self.indexer_start_event.record()
        with torch.cuda.stream(self.aux_stream):
            self.indexer_start_event.wait()
            weights = self.weights_proj(hidden_states)
            self.weights_proj_event.record()
            k_fp8, k_scale = self.compressor(hidden_states, metadata)
            self._update_k_cache_if_needed(k_fp8, k_scale, metadata)
            self.k_cache_update_event.record()
        return (weights, k_fp8, k_scale)

    def _run_overlapped_indexer_prepare(
        self,
        qr: torch.Tensor,
        hidden_states: torch.Tensor,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        position_ids: torch.Tensor,
        pre_aux: Optional[
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
        ] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor
    ]:
        """Prepare indexer inputs by splitting independent work across two streams.

        The current stream owns the Q path.  The auxiliary stream starts from
        the recorded launch point and owns weights projection, compressor, and
        the K-cache update.

        When ``pre_aux`` is provided the aux-stream work has already been
        launched (and ``weights_proj_event`` / ``k_cache_update_event``
        recorded) by ``precompute_aux``; the aux-stream block here is skipped
        and the precomputed tensors are consumed directly.

        Timeline (pre_aux=None):
            current stream:
                record indexer_start_event
                q_proj + RoPE -> quant_q
                wait weights_proj_event -> weight_scale
                wait k_cache_update_event -> return

            aux_stream:
                wait indexer_start_event
                weights_proj -> record weights_proj_event
                compressor -> update_k_cache -> record k_cache_update_event

        Dependency graph:
            q_proj + RoPE -> quant_q -- q_scale --.
                                                   v
            weights_proj --------------------> weight_scale
            compressor -> update_k_cache ----> final wait
        """
        if pre_aux is None:
            self.indexer_start_event.record()

            q_fp8, q_scale = self._project_and_quantize_q(qr, position_ids)

            with torch.cuda.stream(self.aux_stream):
                self.indexer_start_event.wait()

                weights = self.weights_proj(hidden_states)
                self.weights_proj_event.record()

                k_fp8, k_scale = self.compressor(hidden_states, metadata)
                self._update_k_cache_if_needed(k_fp8, k_scale, metadata)
                self.k_cache_update_event.record()
        else:
            weights, k_fp8, k_scale = pre_aux
            # pre_aux tensors were allocated on aux_stream; record on the
            # consuming stream so the caching allocator can't recycle them mid-use.
            cur_stream = torch.cuda.current_stream()
            weights.record_stream(cur_stream)
            if k_fp8 is not None:
                k_fp8.record_stream(cur_stream)
            if k_scale is not None:
                k_scale.record_stream(cur_stream)
            q_fp8, q_scale = self._project_and_quantize_q(qr, position_ids)

        self.weights_proj_event.wait()
        weights = self._apply_weight_scale(weights, q_scale)

        self.k_cache_update_event.wait()
        return q_fp8, q_scale, k_fp8, k_scale, weights

    def _run_serial_indexer_prepare(
        self,
        qr: torch.Tensor,
        hidden_states: torch.Tensor,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        position_ids: torch.Tensor,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor
    ]:
        q_fp8, q_scale = self._project_and_quantize_q(qr, position_ids)

        weights = self.weights_proj(hidden_states)

        weights = self._apply_weight_scale(weights, q_scale)

        k_fp8, k_scale = self.compressor(hidden_states, metadata)
        self._update_k_cache_if_needed(k_fp8, k_scale, metadata)
        return q_fp8, q_scale, k_fp8, k_scale, weights

    def _update_k_cache(
        self,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
    ) -> None:
        # DSV4's indexer compressor already scatters INDEXER_COMPRESS. The
        # shared DSA scatter would duplicate that write.
        return

    def forward(
        self,
        qr: torch.Tensor,
        hidden_states: torch.Tensor,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        position_ids: torch.Tensor,
        pre_aux: Optional[
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
        ] = None,
    ):
        if do_multi_stream() and self.aux_stream is not None:
            q_fp8, q_scale, k_fp8, k_scale, weights = self._run_overlapped_indexer_prepare(
                qr,
                hidden_states,
                metadata,
                position_ids,
                pre_aux=pre_aux,
            )
        else:
            assert pre_aux is None, "pre_aux requires multi-stream mode"
            q_fp8, q_scale, k_fp8, k_scale, weights = self._run_serial_indexer_prepare(
                qr, hidden_states, metadata, position_ids
            )

        # If there are no compressed tokens, return an topk indices buffer with all -1s in the tensor.
        if k_fp8 is None:
            topk_indices = metadata.empty_topk_indices_buffer[: hidden_states.shape[0]]
        else:
            topk_indices = self.sparse_attn_indexer(
                metadata,
                hidden_states,
                q_fp8,
                k_fp8,
                k_scale,
                weights,
                q_scale=q_scale,
            )
        return topk_indices
