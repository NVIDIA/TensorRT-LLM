# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 attention module for integration into the causal encoder/decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.attention.backends.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.attention.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.distributed import AllReduceStrategy
from tensorrt_llm._torch.modules.linear import (
    _FLASHINFER_MXFP8_AUTOTUNE_ACTIVE,
    _FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE,
    Linear,
    MXFP8LinearMethod,
    TensorParallelMode,
    mxfp8_quantize_activation,
)
from tensorrt_llm._torch.modules.multi_stream_utils import do_multi_stream
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.utils import MxFp8QuantizedTensor, is_torch_compiling
from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.math_utils import pad_up
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..deepseek_v4.module import project_sparse_attn_output
from .compressor import CSA2Compressor
from .metadata import CSA2TrtllmMetadata, csa2_side_stream
from .params import (
    CSA2BackendForwardArgs,
    CSA2ForwardState,
    CSA2Layout,
    CSA2Mode,
    CSA2Params,
    RowTransform,
)


@dataclass
class _FP8Query:
    """Decode Q requested as RoPE'd unit-scale E4M3 plus its amax word (``applied`` once made)."""

    positions: torch.Tensor
    quant_q: torch.Tensor
    word: torch.Tensor
    applied: bool = False


class DeepseekV41Attention(nn.Module):
    """Attention component with explicit CED and paged-cache inputs.

    The model supplies packed query positions, cache mappings, and RoPE
    parameters (plain SWA versus compressed YaRN). Decoder global projection
    can consume encoder states independently of the decoder's query/SWA rows.
    Native MXFP8 projections preserve checkpoint FP8 values and block scales.
    BF16 projections remain the default; cache quantization is independent.
    """

    def __init__(
        self,
        layout: CSA2Layout,
        layer_idx: int,
        pos_embd_params: PositionalEmbeddingParams,
        hidden_size: int = 5120,
        num_heads: int = 64,
        head_dim: int = 512,
        rope_head_dim: int = 64,
        q_lora_rank: int = 1280,
        o_lora_rank: int = 1024,
        num_groups: int = 8,
        index_heads: int = 32,
        index_head_dim: int = 128,
        eps: float = 1e-20,
        mapping: Mapping | None = None,
        compute_backend: str = "auto",
        sparse_params: CSA2Params | None = None,
        projection_quantization: Literal["bf16", "mxfp8"] = "bf16",
        kv_cache_dtype: str = "auto",
        aux_stream: torch.cuda.Stream | None = None,
        allreduce_strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
        use_cute_dsl_blockscaling_bmm: bool = False,
        use_cute_dsl_bf16_bmm: bool = False,
    ) -> None:
        super().__init__()
        if num_heads % num_groups or head_dim <= rope_head_dim:
            raise ValueError("Invalid CSA2 head/group geometry")
        mapping = mapping or Mapping()
        if mapping.pp_size != 1 or mapping.cp_size != 1:
            raise ValueError(
                "CSA2 source state requires colocated layers; PP/CP transfer is not implemented"
            )
        if mapping.enable_attention_dp:
            # Match Attention's local projection mapping: every rank holds all
            # heads, so only the TP fields of this local mapping are meaningful
            # (pp_size only satisfies the world-size product). The model/cache
            # keep their original TP group for MoE and independent request state.
            mapping = Mapping(
                world_size=mapping.world_size,
                rank=mapping.rank,
                gpus_per_node=mapping.gpus_per_node,
                tp_size=1,
                pp_size=mapping.world_size,
                enable_attention_dp=True,
            )
        if num_heads % mapping.tp_size or num_groups % mapping.tp_size:
            raise ValueError("CSA2 heads and output groups must be divisible by TP size")
        if projection_quantization not in ("bf16", "mxfp8"):
            raise ValueError("CSA2 projection quantization must be bf16 or mxfp8")
        if sparse_params is not None and sparse_params.layout != layout:
            raise ValueError("CSA2 module and sparse parameters must use the same layout")
        sparse_params = sparse_params or CSA2Params(layout=layout, compute_backend=compute_backend)
        self.projection_quantization = projection_quantization
        self.use_cute_dsl_blockscaling_bmm = use_cute_dsl_blockscaling_bmm
        self.use_cute_dsl_bf16_bmm = use_cute_dsl_bf16_bmm
        self.dtype = torch.bfloat16
        self.aux_stream = aux_stream
        self._prepare_start = torch.cuda.Event() if aux_stream is not None else None
        self._prepare_done = torch.cuda.Event() if aux_stream is not None else None
        self._weights_done = torch.cuda.Event() if aux_stream is not None else None
        self._descriptors_done = torch.cuda.Event() if aux_stream is not None else None
        self._query_done = torch.cuda.Event() if aux_stream is not None else None
        self._touch_done = torch.cuda.Event() if aux_stream is not None else None
        self._touch_sink = None
        self._overlap_query_kv = True
        self._fuse_index_q = sparse_params.fuse_index_q
        if sparse_params.fuse_packed_output_rope and pos_embd_params.is_neox:
            raise ValueError("CSA2 packed output RoPE requires interleaved rotary channels")
        quant_config = None
        if projection_quantization == "mxfp8":
            if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
                (10, 0),
                (10, 3),
                (10, 7),
            ):
                raise ValueError("CSA2 native MXFP8 projections require SM100/SM103/SM107")
            if not hasattr(torch.ops.trtllm, "mxfp8_mxfp8_gemm"):
                raise RuntimeError("CSA2 native MXFP8 GEMM is not compiled")
            quant_config = QuantConfig(quant_algo=QuantAlgo.MXFP8)
        if self._fuse_index_q and (
            projection_quantization != "mxfp8"
            or index_head_dim != 128
            or rope_head_dim != 64
            or pos_embd_params.is_neox
            or q_lora_rank % 128
        ):
            raise ValueError(
                "Fused CSA2 index Q requires native MXFP8, 128-aligned Q rank and interleaved 128D/64D RoPE"
            )
        self.mapping = mapping
        self.num_groups = num_groups
        self.layer = layout.layer(layer_idx)
        self.num_heads_tp = num_heads // mapping.tp_size
        self.qk_head_dim = self.v_head_dim = head_dim
        self.qk_rope_head_dim = rope_head_dim
        self.qk_nope_head_dim = head_dim - rope_head_dim
        self.n_local_groups = num_groups // mapping.tp_size
        self.o_lora_rank = o_lora_rank
        self.eps = eps
        from tensorrt_llm._torch.attention.backends.utils import create_attention

        self.layout = layout
        self.backend = create_attention(
            "TRTLLM",
            layer_idx,
            self.num_heads_tp,
            head_dim,
            num_kv_heads=1,
            sparse_params=sparse_params,
            # The backend derives the staging dtype; the persistent CSA2 cache
            # formats are fixed and never requantized.
            kv_cache_dtype=kv_cache_dtype,
        )
        from .backend import CSA2FlashInfer, CSA2FlashMLA

        self._flash_attention = (
            CSA2FlashMLA(self.backend)
            if self.backend.compute_backend == "flash_mla"
            else CSA2FlashInfer(self.backend)
            if self.backend.compute_backend == "flashinfer"
            else None
        )
        self.index_heads = index_heads
        self.index_head_dim = index_head_dim
        self.wq_a = Linear(
            hidden_size, q_lora_rank, bias=False, dtype=torch.bfloat16, quant_config=quant_config
        )
        self.wq_b = Linear(
            q_lora_rank,
            num_heads * head_dim,
            bias=False,
            dtype=torch.bfloat16,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=quant_config,
        )
        self.wkv = Linear(
            hidden_size, head_dim, bias=False, dtype=torch.bfloat16, quant_config=quant_config
        )
        self.q_norm = RMSNorm(hidden_size=q_lora_rank, eps=eps, dtype=torch.bfloat16)
        self.kv_norm = RMSNorm(hidden_size=head_dim, eps=eps, dtype=torch.bfloat16)
        self.register_buffer("_fused_qkv_weight", None, persistent=False)
        self.register_buffer("_fused_qkv_scale", None, persistent=False)
        self._fuse_qkv_projection = True
        self._fuse_prefill_projection = True
        self._fuse_q_norm_quant = True
        self._fuse_q_rope_projection = True
        # A context-only batch run by the fused context kernel projects Q to unit-scale E4M3 only.
        self._unit_scale_context_query = True
        # The fused context Q projection hands the index-Q projection its MXFP8 latent.
        self._mxfp8_query_latent = True
        # From this many decode tokens the Q-B GEMM writes RoPE'd unit-scale FP8 Q and
        # its amax directly; staging then never reads or quantizes BF16 Q.
        self._fp8_q_projection = True
        self._fp8_q_min_tokens = 256
        self._fp8_q_word = None
        # Steps of at most SMALL_M_ROWS rows run the Q-B and index-Q projections on the small-M
        # warp-MMA kernel, and the grouped O-A projection as a split-K kernel on the BF16
        # attention output that streams the weight across all SMs.
        self._small_m_projection = True
        # The block-128 O-A of the fused context kernel's E4M3 output quantizes O-B's MXFP8
        # input in its epilogue.
        self._fuse_o_a_mxfp8 = True
        self.attn_sink = nn.Parameter(torch.zeros(self.num_heads_tp, dtype=torch.float32))
        # Use V4's grouped O-LoRA projection and inverse-RoPE implementation.
        self.o_a_proj = nn.Parameter(
            torch.empty(
                self.n_local_groups,
                o_lora_rank,
                num_heads * head_dim // num_groups,
                dtype=torch.float8_e4m3fn if quant_config is not None else torch.bfloat16,
            ),
            requires_grad=False,
        )
        self._o_a_fp8_block128 = False
        self.o_a_proj_dequant = None
        self.register_buffer("o_a_proj_scale", None)
        # UE8M0 scales equal to power-of-two 128-block O-A scales, for MXFP8 GEMMs.
        self.register_buffer("_o_a_mxfp8_scale", None, persistent=False)
        self.register_buffer("_fp8_alpha", torch.ones(1, dtype=torch.float32), persistent=False)
        self.o_b_proj = Linear(
            num_groups * o_lora_rank,
            hidden_size,
            bias=False,
            dtype=torch.bfloat16,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=quant_config,
            allreduce_strategy=allreduce_strategy,
        )
        self.rotary_emb = RotaryEmbedding(
            pos_embd_params.rope, head_dim=rope_head_dim, is_neox=pos_embd_params.is_neox
        )
        self.inverse_rotary_emb = RotaryEmbedding(
            pos_embd_params.rope,
            head_dim=rope_head_dim,
            is_neox=pos_embd_params.is_neox,
            inverse=True,
        )
        self.compressor = None
        if self.layer.mode == CSA2Mode.FULL:
            self.compressor = CSA2Compressor(hidden_size, head_dim, self.layer.compress_ratio, eps)
            self.index_wk = nn.Linear(head_dim, index_head_dim, bias=False, dtype=torch.bfloat16)
            self.index_k_norm = RMSNorm(hidden_size=index_head_dim, eps=eps, dtype=torch.bfloat16)
        if self.layer.mode in (CSA2Mode.FULL, CSA2Mode.REINDEX):
            self.index_wq_b = Linear(
                q_lora_rank,
                index_heads * index_head_dim,
                bias=False,
                dtype=torch.bfloat16,
                quant_config=quant_config,
            )
            self.index_weights_proj = nn.Linear(
                hidden_size, index_heads, bias=False, dtype=torch.bfloat16
            )
            self._index_weight_scale = index_head_dim**-0.5 * index_heads**-0.5
            # Candidate consumers' sparse MQA logits read BF16 head weights.
            self._index_weight_dtype = (
                torch.bfloat16 if self.layer.consumes_candidates else torch.float32
            )

    @classmethod
    def from_hf_config(
        cls,
        config: PretrainedConfig,
        layer_idx: int,
        mapping: Mapping | None = None,
        *,
        sparse_params: CSA2Params | None = None,
        projection_quantization: Literal["bf16", "mxfp8"] = "bf16",
        kv_cache_dtype: str = "auto",
        aux_stream: torch.cuda.Stream | None = None,
        allreduce_strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
    ) -> DeepseekV41Attention:
        """Construct an attention component from the published text configuration.

        The CED model owns layer execution and passes its text_config here;
        this does not register the incomplete multimodal model with AutoModel.
        """
        layout = CSA2Layout(
            tuple(config.compress_ratios),
            tuple(config.kv_source_layer_ids),
            tuple(config.index_source_layer_ids),
            config.candidate_source_layer_id,
            config.candidate_topk_blocks,
            config.candidate_block_size,
            config.index_topk,
            config.sliding_window,
        )
        rope = RopeParams.from_config(config)
        if layout.layer(layer_idx).compress_ratio:
            rope.theta = config.compress_rope_theta
            rope.scale_type = RotaryScalingType.yarn
            rope.mscale = 0.0
            rope.mscale_all_dim = 0.0
        else:
            rope.theta = config.rope_theta
            rope.scale_type = RotaryScalingType.none
            rope.scale = 1.0
        positional = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gptj,
            rope=rope,
            is_neox=False,
        )
        return cls(
            layout,
            layer_idx,
            positional,
            config.hidden_size,
            config.num_attention_heads,
            config.head_dim,
            config.qk_rope_head_dim,
            config.q_lora_rank,
            config.o_lora_rank,
            config.o_groups,
            config.index_n_heads,
            config.index_head_dim,
            config.rms_norm_eps,
            mapping=mapping,
            sparse_params=sparse_params,
            projection_quantization=projection_quantization,
            kv_cache_dtype=kv_cache_dtype,
            aux_stream=aux_stream,
            allreduce_strategy=allreduce_strategy,
        )

    def load_hf_weights(self, weights: dict[str, torch.Tensor], prefix: str = "") -> None:
        """Load one HF attention with validated FP8 scales and global TP shapes."""
        from .weights import load_attention_weights

        load_attention_weights(
            self,
            weights,
            prefix,
            self.num_groups,
            self.o_lora_rank,
            self.mapping.tp_size,
            self.mapping.tp_rank,
        )
        self._fused_qkv_weight = self._fused_qkv_scale = None
        if (
            self.projection_quantization == "mxfp8"
            and hasattr(self, "wq_a")
            and hasattr(self, "wkv")
            and self.wq_a.weight.shape[0] % 128 == 0
            and self.wkv.weight.shape[0] % 128 == 0
        ):
            # Each projection ends at a complete 128-row scale tile.
            self._fused_qkv_weight = torch.cat((self.wq_a.weight, self.wkv.weight))
            self._fused_qkv_scale = torch.cat((self.wq_a.weight_scale, self.wkv.weight_scale))
        self._o_a_mxfp8_scale = None
        if self._o_a_fp8_block128 and is_sm_100f():
            from .output_projection import mxfp8_weight_scale

            self._o_a_mxfp8_scale = mxfp8_weight_scale(self.o_a_proj_scale)

    @property
    def takes_mxfp8_input(self) -> bool:
        """Whether decode graphs project the query and SWA rows with one fused MXFP8 GEMM, whose
        quantized input the caller may supply (``forward(hidden_states_mxfp8=...)``)."""
        return (
            self._fused_qkv_weight is not None
            and self._fuse_qkv_projection
            and self.wq_a.quant_method.tune_mxfp8_backends
            and self.wq_b.quant_method.tune_mxfp8_backends
            and self.wkv.quant_method.tune_mxfp8_backends
            and self.wq_a.mxfp8_allowed_backends == self.wkv.mxfp8_allowed_backends
        )

    def _can_fuse_prefill_projection(self, hidden_states: torch.Tensor) -> bool:
        return (
            getattr(self, "_fused_qkv_weight", None) is not None
            and self._fuse_prefill_projection
            and hidden_states.shape[0] > 0
            and hidden_states.dtype == torch.bfloat16
            and not is_torch_compiling()
            and not _FLASHINFER_MXFP8_AUTOTUNE_ACTIVE.get()
            and not _FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.get()
            and all(
                isinstance(linear.quant_method, MXFP8LinearMethod)
                and linear.quant_method.use_cutlass
                and linear.quant_method.backend in ("trtllm", "auto")
                for linear in (self.wq_a, self.wq_b, self.wkv)
            )
        )

    def _project_query_and_swa(
        self,
        hidden_states: torch.Tensor,
        *,
        decode_only: bool,
        query_stream: torch.cuda.Stream | None = None,
        late_stream: torch.cuda.Stream | None = None,
        fp8_query: _FP8Query | None = None,
        prequantized: MxFp8QuantizedTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not decode_only and self._can_fuse_prefill_projection(hidden_states):
            return self._project_prefill_query_and_swa(hidden_states)
        fused = (
            decode_only
            and _FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.get()
            and hidden_states.shape[0] > 0
            and self.takes_mxfp8_input
        )
        if not fused:
            qr = self.q_norm(self.wq_a(hidden_states))
            if late_stream is not None:
                late_stream.wait_stream(torch.cuda.current_stream(hidden_states.device))
                qr.record_stream(late_stream)
            with torch.cuda.stream(late_stream):
                q = self.wq_b(qr)
            return qr, q, None

        from tensorrt_llm._torch.custom_ops.torch_custom_ops import mxfp8_quantize_gemm_autotuned

        tune = _FLASHINFER_MXFP8_AUTOTUNE_ACTIVE.get()
        if prequantized is not None:
            combined = mxfp8_quantize_gemm_autotuned(
                hidden_states.contiguous(),
                self._fused_qkv_weight,
                self._fused_qkv_scale,
                torch.bfloat16,
                self.wq_a.mxfp8_allowed_backends,
                quantized=(prequantized.fp8_tensor, prequantized.scaling_factor),
            )
        elif self._small_m(hidden_states.shape[0]):
            from .projection import mxfp8_quantize_rows, mxfp8_query_gemm

            # A small step's QKV-A GEMM is latency bound: its weights head for L2 while the
            # hidden states are quantized.
            weights = (self._fused_qkv_weight, self._fused_qkv_scale)
            quantized, scales = mxfp8_quantize_rows(hidden_states.contiguous(), prefetch=weights)
            combined = mxfp8_query_gemm(
                quantized,
                scales,
                *weights,
                allowed_backends=self.wq_a.mxfp8_allowed_backends,
            )
        else:
            combined = mxfp8_quantize_gemm_autotuned(
                hidden_states.contiguous(),
                self._fused_qkv_weight,
                self._fused_qkv_scale,
                torch.bfloat16,
                self.wq_a.mxfp8_allowed_backends,
            )
        rank = self.wq_a.weight.shape[0]
        latent, swa = combined[:, :rank], combined[:, rank:]
        if query_stream is not None:
            query_stream.wait_stream(torch.cuda.current_stream(hidden_states.device))
            combined.record_stream(query_stream)
            if fp8_query is not None:
                fp8_query.quant_q.record_stream(query_stream)
        with torch.cuda.stream(query_stream):
            qr, q = self._project_query_latent(
                latent, tune=tune, late_stream=late_stream, fp8_query=fp8_query
            )
        return qr, q, swa

    def _small_m(self, rows: int) -> bool:
        from .projection import SMALL_M_ROWS

        return getattr(self, "_small_m_projection", False) and 0 < rows <= SMALL_M_ROWS

    def _lowm_projects(self, linear: Linear, rows: int) -> bool:
        """Whether a decode-graph small step projects ``rows`` BF16 rows through ``linear`` on
        the small-M warp-MMA kernel, which reads the MXFP8 weights and their swizzled scales."""
        from tensorrt_llm._torch.modules.low_m_gemm import lowm_fp8_mma_fits

        from .projection import LOWM_MAX_WEIGHT_BYTES

        method = linear.quant_method
        return (
            _FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.get()
            and self._small_m(rows)
            and isinstance(method, MXFP8LinearMethod)
            and method.use_cutlass
            and linear.bias is None
            and linear.weight_scale.dim() == 1
            and linear.weight.numel() <= LOWM_MAX_WEIGHT_BYTES
            and lowm_fp8_mma_fits(rows, *linear.weight.shape)
        )

    def _project_prefill_query_and_swa(
        self,
        hidden_states: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        quant_q: torch.Tensor | None = None,
        q_out: torch.Tensor | None = None,
        fp8_only: bool = False,
        latent_mxfp8: bool = False,
    ) -> tuple[
        torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor | None, torch.Tensor
    ]:
        """Return ``(qr, q, swa)``; with ``fp8_only`` Q lands only in ``quant_q`` and ``q`` is None.

        With ``positions``, the Q-B GEMM applies RoPE and writes the BF16 Q into ``q_out``
        (allocated when None) plus its unit-scale E4M3 into ``quant_q`` when given. With
        ``latent_mxfp8`` (and the fused norm), ``qr`` is the normalized latent's MXFP8
        values and scales, as the index-Q projection consumes it, instead of BF16.
        """
        from .projection import native_mxfp8_gemm, rms_norm_mxfp8

        quantized, scales = mxfp8_quantize_activation(hidden_states.contiguous())
        combined = native_mxfp8_gemm(
            self.wq_a,
            quantized,
            scales,
            weight=self._fused_qkv_weight,
            weight_scale=self._fused_qkv_scale,
        )
        rank = self.wq_a.weight.shape[0]
        latent, swa = combined[:, :rank], combined[:, rank:]
        if self._fuse_q_norm_quant and rank % 128 == 0 and rank <= 8192:
            qr, quantized, scales = rms_norm_mxfp8(
                latent,
                self.q_norm.weight,
                self.q_norm.variance_epsilon,
                normalized=not latent_mxfp8,
            )
            if latent_mxfp8:
                qr = quantized, scales
            if positions is not None and fp8_only:
                from .query_projection import fp8_query_gemm

                if not fp8_query_gemm(
                    quantized,
                    scales,
                    self.wq_b.weight,
                    self.wq_b.weight_scale,
                    self.wq_b._mxfp8_alpha,
                    positions.reshape(-1),
                    self.rotary_emb.rotary_cos_sin,
                    quant_q,
                ):
                    raise RuntimeError(
                        "CSA2 FP8 context query projection cannot compile in capture"
                    )
                q = None
            elif positions is not None:
                from .query_projection import query_gemm_rope

                q, _ = query_gemm_rope(
                    quantized,
                    scales,
                    self.wq_b.weight,
                    self.wq_b.weight_scale,
                    self.wq_b._mxfp8_alpha,
                    positions.reshape(-1),
                    self.rotary_emb.rotary_cos_sin,
                    quantize=quant_q is not None,
                    quant_q=quant_q,
                    out=q_out,
                )
            else:
                q = native_mxfp8_gemm(self.wq_b, quantized, scales)
        else:
            qr = self.q_norm(latent)
            q = self.wq_b(qr)
        return qr, q, swa

    def _project_query_latent(
        self,
        latent: torch.Tensor,
        *,
        tune: bool,
        late_stream: torch.cuda.Stream | None = None,
        fp8_query: _FP8Query | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(qr, q)``; ``q`` is None once ``fp8_query`` received the projection."""
        rank = latent.shape[1]
        fused = self._fuse_q_norm_quant and rank % 128 == 0 and rank <= 8192
        # A small step's Q-B runs the small-M kernel on the BF16 rows, which the norm writes alone.
        small_m = fused and fp8_query is None and self._lowm_projects(self.wq_b, latent.shape[0])
        if small_m:
            from .projection import rms_norm_mxfp8

            qr, _, _ = rms_norm_mxfp8(
                latent, self.q_norm.weight, self.q_norm.variance_epsilon, quantize=False
            )
            inputs = (qr,)
        elif fused:
            from .projection import mxfp8_query_gemm, rms_norm_mxfp8

            # A small step's Q-B GEMM is latency bound: its weights head for L2 under the norm.
            prefetch = (
                (self.wq_b.weight, self.wq_b.weight_scale)
                if self._small_m(latent.shape[0]) and fp8_query is None
                else ()
            )
            qr, quantized, scales = rms_norm_mxfp8(
                latent, self.q_norm.weight, self.q_norm.variance_epsilon, prefetch=prefetch
            )
            inputs = (quantized, scales)
        else:
            qr = self.q_norm(latent)
            inputs = (qr,)
        if late_stream is not None:
            late_stream.wait_stream(torch.cuda.current_stream(latent.device))
            for tensor in inputs:
                tensor.record_stream(late_stream)
            if fp8_query is not None:
                fp8_query.quant_q.record_stream(late_stream)
        with torch.cuda.stream(late_stream):
            if fused and fp8_query is not None and not tune:
                from .query_projection import fp8_query_gemm

                fp8_query.applied = fp8_query_gemm(
                    quantized,
                    scales,
                    self.wq_b.weight,
                    self.wq_b.weight_scale,
                    self.wq_b._mxfp8_alpha,
                    fp8_query.positions,
                    self.rotary_emb.rotary_cos_sin,
                    fp8_query.quant_q,
                    fp8_query.word,
                )
            if fp8_query is not None and fp8_query.applied:
                q = None
            elif small_m:
                from tensorrt_llm._torch.modules.low_m_gemm import mxfp8_lowm_linear

                q = mxfp8_lowm_linear(
                    qr, self.wq_b.weight, self.wq_b.weight_scale, scale_swizzled=True
                )
            elif fused:
                q = mxfp8_query_gemm(
                    quantized,
                    scales,
                    self.wq_b.weight,
                    self.wq_b.weight_scale,
                    allowed_backends=self.wq_b.mxfp8_allowed_backends,
                )
            else:
                q = self.wq_b(qr)
        return qr, q

    def _fp8_query_request(
        self, hidden_states: torch.Tensor, positions: torch.Tensor, attn_metadata
    ) -> _FP8Query | None:
        """Ask the decode Q projection for staging's E4M3 Q when the backend consumes it."""
        if not (
            getattr(self, "_fp8_q_projection", False)
            and attn_metadata.num_ctx_tokens == 0
            and hidden_states.shape[0] >= self._fp8_q_min_tokens
            and self.projection_quantization == "mxfp8"
            and self.qk_head_dim == 512
            and self.qk_rope_head_dim == 64
            and not self.rotary_emb.is_neox
            and self.backend.compute_backend == "trtllm"
            and self.backend.native_fp8
            and not self.backend.sparse_params.use_packed_sparse_attention
            and self._flash_attention is None
            and positions.dtype in (torch.int32, torch.int64)
            and positions.is_contiguous()
            and is_sm_100f()
        ):
            return None
        word = self._q_amax_word(hidden_states.device)
        if word is None:
            return None
        quant_q = torch.empty(
            (hidden_states.shape[0], self.num_heads_tp * self.qk_head_dim),
            dtype=torch.float8_e4m3fn,
            device=hidden_states.device,
        )
        return _FP8Query(positions.reshape(-1), quant_q, word)

    def _row_transform(self, positions, norm=None) -> RowTransform:
        """Norm and RoPE to apply while cache rows are quantized (interleaved RoPE only)."""
        rope = None
        if not self.rotary_emb.is_neox:
            table = self.rotary_emb.rotary_cos_sin.view(-1, self.qk_rope_head_dim)
            rope = (positions.reshape(-1).int(), table, self.qk_rope_head_dim)
        return RowTransform(norm=norm, rope=rope)

    def _rope_query(
        self, q: torch.Tensor, positions: torch.Tensor, amax: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, bool]:
        """RoPE of the attention query; small steps take the PDL kernel.

        That kernel also raises ``amax`` (an int32 word), when given, to the FP32 bits of the
        rotated ``amax(|q|)``. Returns the query and whether ``amax`` was raised.
        """
        positions = positions.reshape(-1)
        if (
            self._small_m(q.shape[0])
            and not self.rotary_emb.is_neox
            and self.qk_rope_head_dim == 64
            and q.is_contiguous()
            and positions.is_contiguous()
        ):
            from .projection import gptj_rope_small_m

            gptj_rope_small_m(q, positions, self.rotary_emb.rotary_cos_sin, amax)
            return q, amax is not None
        return self._rope(q, positions, self.num_heads_tp), False

    def _touch_output_weights(self, rows: int) -> bool:
        """Read a small step's O-A and O-B weights into L2 on the current (auxiliary) stream.

        The query stream is idle once Q is rotated, and staging plus attention give the
        reads time to finish before the output projection streams those weights.
        """
        linear = self.o_b_proj
        if not (
            self._small_m(rows)
            and self._o_a_fp8_block128
            and self.o_a_proj.dtype == torch.float8_e4m3fn
            and isinstance(linear.quant_method, MXFP8LinearMethod)
            and linear.quant_method.use_cutlass
            and linear.weight.is_contiguous()
        ):
            return False
        if self._touch_sink is None:
            if torch.cuda.is_current_stream_capturing():
                return False
            self._touch_sink = torch.zeros(64, dtype=torch.int32, device=linear.weight.device)
        from .projection import l2_touch

        l2_touch((self.o_a_proj, linear.weight, linear.weight_scale), self._touch_sink)
        self._touch_done.record()
        return True

    def _q_amax_word(self, device: torch.device) -> torch.Tensor | None:
        """The int32 word Q's producer raises to Q's amax bits; staging consumes and clears it."""
        if self._fp8_q_word is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            self._fp8_q_word = torch.zeros(1, dtype=torch.int32, device=device)
        return self._fp8_q_word

    def _stages_native_fp8_generation(self, attn_metadata) -> bool:
        """Whether this forward's only tile is a generation tile on native FP8 staging."""
        return (
            attn_metadata.num_ctx_tokens == 0
            and self.backend.compute_backend == "trtllm"
            and self.backend.native_fp8
            and not self.backend.sparse_params.use_packed_sparse_attention
            and self._flash_attention is None
        )

    def _rope(self, x: torch.Tensor, positions: torch.Tensor, heads: int) -> torch.Tensor:
        # Incomplete compression groups publish no rows; the native RoPE
        # launcher requires a nonzero token grid.
        if x.shape[0] == 0:
            return x
        torch.ops.trtllm.mla_rope_inplace(
            x,
            positions.reshape(-1),
            self.rotary_emb.rotary_cos_sin,
            heads,
            x.shape[-1] - self.qk_rope_head_dim,
            self.qk_rope_head_dim,
            False,
            self.rotary_emb.is_neox,
        )
        return x

    def _prepare_global(
        self, hidden_states, global_hidden_states, compression, compressed_positions
    ):
        main_kv = index_k = None
        if self.compressor is not None:
            if compressed_positions is None:
                raise ValueError("Full CSA2 mode requires compressed positions")
            global_input = hidden_states if global_hidden_states is None else global_hidden_states
            if global_input.shape[0] == 0:
                if compressed_positions.numel() != 0:
                    raise ValueError("An empty CSA2 global source cannot publish compressed rows")
                # Pure replay reuses existing global/index caches. Do not
                # project cached prefix rows merely to discard their outputs.
                main_kv = global_input.new_empty((0, self.qk_head_dim))
                index_k = global_input.new_empty((0, self.index_head_dim))
            else:
                # An odd ratio-two replay boundary includes its last cached
                # raw token so this same compressor can rebuild partial state.
                latent = self.compressor(global_input, compression)
                # The index-K norm and both RoPEs run inside the cache writes
                # (see ``global_transforms``); the rows stay untransformed here.
                index_k = self.index_wk(latent)
                main_kv = latent
                if self.rotary_emb.is_neox:
                    if index_k.shape[0]:
                        index_k = self.index_k_norm(index_k)
                    index_k = self._rope(index_k.unsqueeze(1), compressed_positions, 1).squeeze(1)
                    main_kv = self._rope(latent.unsqueeze(1), compressed_positions, 1).squeeze(1)
        return main_kv, index_k

    def global_transforms(self, compressed_positions):
        """Transforms applied while ``_prepare_global`` rows are written: main RoPE, index norm + RoPE."""
        if self.compressor is None or self.rotary_emb.is_neox:
            return None, None
        return (
            self._row_transform(compressed_positions),
            self._row_transform(
                compressed_positions,
                norm=(self.index_k_norm.weight, self.index_k_norm.variance_epsilon),
            ),
        )

    def prepare_global_cache(
        self, hidden_states: torch.Tensor, metadata: CSA2TrtllmMetadata
    ) -> None:
        """Publish a CED owner's full encoder rows before narrowing decoder queries."""
        layer_idx = self.layer.layer_idx
        if self.layer.mode != CSA2Mode.FULL or self.compressor is None:
            raise ValueError("Only a Full CSA2 layer can precompute global KV")
        if layer_idx in metadata.csa2_precomputed_kv_layers:
            raise ValueError("CSA2 global KV was already precomputed for this forward")
        source = metadata.select_global_source(layer_idx, hidden_states, None)
        main_kv, index_k = self._prepare_global(
            hidden_states,
            source,
            metadata.get_compression_batch(layer_idx),
            metadata.get_compressed_positions(layer_idx),
        )
        metadata.kv_cache_manager.write_global(
            self.layer.kv_source,
            metadata.csa2_main_write_slots[self.layer.kv_source],
            main_kv,
            index_k,
            *self.global_transforms(metadata.get_compressed_positions(layer_idx)),
        )
        metadata.csa2_precomputed_kv_layers.add(layer_idx)

    def _prepare_auxiliary(
        self,
        hidden_states,
        global_hidden_states,
        compression,
        compressed_positions,
        *,
        skip_global: bool = False,
        skip_weights: bool = False,
    ):
        main_kv, index_k = (
            (None, None)
            if skip_global
            else self._prepare_global(
                hidden_states, global_hidden_states, compression, compressed_positions
            )
        )
        index_weights = None
        if self.layer.mode in (CSA2Mode.FULL, CSA2Mode.REINDEX) and not skip_weights:
            index_weights = self._project_index_weights(hidden_states)
        return index_weights, main_kv, index_k

    def _project_index_weights(self, hidden_states):
        projected = self.index_weights_proj(hidden_states)
        # One launch for ``(projected * scale).to(dtype)``, bit for bit.
        return torch.mul(
            projected,
            self._index_weight_scale,
            out=torch.empty(
                projected.shape, dtype=self._index_weight_dtype, device=projected.device
            ),
        )

    def _project_index_q(self, qr, positions):
        """Index Q of BF16 ``qr``, or of its MXFP8 ``(values, scales)`` pair."""
        if self.layer.mode not in (CSA2Mode.FULL, CSA2Mode.REINDEX):
            return None, None
        if isinstance(qr, tuple):
            from .projection import mxfp8_linear_quantized

            query = mxfp8_linear_quantized(self.index_wq_b, *qr)
            query = query.reshape(-1, self.index_heads, self.index_head_dim)
            return self._rope(query, positions, self.index_heads), None
        if self._fuse_index_q and qr.shape[0] <= 16:
            if qr.shape[0] == 0:
                return (
                    torch.empty((0, self.index_heads, 64), dtype=torch.int8, device=qr.device),
                    torch.empty((0, self.index_heads), dtype=torch.int32, device=qr.device),
                )
            from .kernel import csa2_indexer_q_gemm_rope_fp4

            packed, scale = csa2_indexer_q_gemm_rope_fp4(
                qr,
                self.index_wq_b.weight,
                self.index_wq_b.weight_scale,
                positions.reshape(-1),
                self.rotary_emb.rotary_cos_sin.view(-1, 64),
                self._fp8_alpha,
            )
            return packed.view(-1, self.index_heads, 64), scale.view(-1, self.index_heads)
        if (
            self._lowm_projects(self.index_wq_b, qr.shape[0])
            and not self.rotary_emb.is_neox
            and positions.dtype == torch.int32
        ):
            from tensorrt_llm._torch.modules.low_m_gemm import mxfp8_lowm_linear

            # The kernel rotates in FP32 in its epilogue, before the single BF16 rounding.
            query = mxfp8_lowm_linear(
                qr.contiguous(),
                self.index_wq_b.weight,
                self.index_wq_b.weight_scale,
                rope=(
                    positions,
                    self.rotary_emb.rotary_cos_sin.view(-1, 2, self.qk_rope_head_dim // 2),
                    self.index_head_dim,
                ),
                scale_swizzled=True,
            )
            return query.view(-1, self.index_heads, self.index_head_dim), None
        query = self.index_wq_b(qr).reshape(-1, self.index_heads, self.index_head_dim)
        return self._rope(query, positions, self.index_heads), None

    def _lowm_o_a(self, rows: int) -> bool:
        """Whether a decode-graph small step's O-A runs on the small-M warp-MMA kernel, which
        rotates each head's RoPE channels back as it loads them."""
        from tensorrt_llm._torch.modules.low_m_gemm import default_k_warps, lowm_fp8_mma_fits

        from .projection import LOWM_MAX_WEIGHT_BYTES

        groups, rank, depth = self.o_a_proj.shape
        # The kernel multiplies FP8 weights by their block scales in BF16, which is exact only
        # for the power-of-two scales that also have an MXFP8 form.
        return (
            _FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.get()
            and self._o_a_mxfp8_scale is not None
            and self.o_a_proj.numel() <= LOWM_MAX_WEIGHT_BYTES
            and lowm_fp8_mma_fits(rows, groups * rank, depth)
            and depth % self.qk_head_dim == 0
            and (depth // default_k_warps(depth)) % self.qk_head_dim == 0
        )

    def _project_output_small_m(self, output, positions):
        """O-A with inverse RoPE fused, then O-B, for a small step; None when not applicable."""
        rows = output.shape[0]
        positions = positions.reshape(-1)
        groups, rank, depth = self.o_a_proj.shape
        if not (
            self._small_m(rows)
            and self.o_a_proj.dtype == torch.float8_e4m3fn
            and self.o_a_proj_scale is not None
            and self.o_a_proj_scale.dtype == torch.float32
            and self.o_a_proj_scale.shape == (groups, rank // 128, depth // 128)
            and self.qk_nope_head_dim == 448
            and self.qk_rope_head_dim == 64
            and not self.inverse_rotary_emb.is_neox
            and output.numel() == rows * groups * depth
            and positions.dtype == torch.int32
            and positions.numel() == rows
            and positions.is_contiguous()
        ):
            return None
        from .projection import inverse_rope_block128_small_m, mxfp8_linear_quantized

        # O-B consumes the latent quantized in the O-A reduction, then reduces across TP
        # as its forward would.
        linear = self.o_b_proj
        quantize = (
            isinstance(linear.quant_method, MXFP8LinearMethod)
            and linear.quant_method.use_cutlass
            and linear.bias is None
            and getattr(linear, "lora", None) is None
            and not is_torch_compiling()
            and not (linear.reduce_output and linear.use_fused_gemm_allreduce)
        )
        if output.dtype == torch.bfloat16 and self._lowm_o_a(rows):
            from tensorrt_llm._torch.modules.low_m_gemm import fp8_grouped_lowm, mxfp8_lowm_linear

            # Both kernels load their weights before the PDL wait: O-B's stream in under O-A.
            latent = fp8_grouped_lowm(
                output.reshape(rows, -1).contiguous(),
                self.o_a_proj.view(groups * rank, depth),
                self.o_a_proj_scale.view(groups * rank // 128, depth // 128),
                rank,
                inverse_rope=(
                    positions,
                    self.inverse_rotary_emb.rotary_cos_sin.view(-1, 2, self.qk_rope_head_dim // 2),
                    self.qk_head_dim,
                ),
            )
            if not (quantize and self._lowm_projects(linear, rows)):
                return linear(latent)
            projected = mxfp8_lowm_linear(
                latent, linear.weight, linear.weight_scale, scale_swizzled=True
            )
            return linear.all_reduce(projected) if linear.reduce_output else projected
        o_lora = inverse_rope_block128_small_m(
            output.reshape(rows, -1),
            positions,
            self.inverse_rotary_emb.rotary_cos_sin,
            self.o_a_proj,
            self.o_a_proj_scale,
            name="csa2_o_a",
            quantize_output=quantize,
        )
        if not quantize:
            return linear(o_lora)
        projected = mxfp8_linear_quantized(linear, *o_lora)
        if linear.reduce_output:
            projected = linear.all_reduce(projected)
        return projected

    def _project_output_mxfp8(self, output, positions):
        """O-A as MXFP8 GEMMs with inverse RoPE fused into quantization, then O-B; else None."""
        rows = output.shape[0]
        positions = positions.reshape(-1)
        groups, rank, depth = self.o_a_proj.shape
        if not (
            getattr(self, "_o_a_mxfp8_scale", None) is not None
            and rows > 0
            and self.qk_nope_head_dim == 448
            and self.qk_rope_head_dim == 64
            and not self.inverse_rotary_emb.is_neox
            and output.numel() == rows * groups * depth
            and positions.numel() == rows
        ):
            return None
        from .output_projection import grouped_output_mxfp8

        # Without a TP reduction, O-B consumes the latent quantized in the O-A epilogue.
        linear = self.o_b_proj
        quantize = (
            isinstance(linear.quant_method, MXFP8LinearMethod)
            and linear.quant_method.use_cutlass
            and not linear.reduce_output
            and linear.bias is None
            and getattr(linear, "lora", None) is None
            and not is_torch_compiling()
        )
        o_lora = grouped_output_mxfp8(
            output.reshape(rows, -1),
            positions.contiguous(),
            self.inverse_rotary_emb.rotary_cos_sin,
            self.o_a_proj,
            self._o_a_mxfp8_scale,
            self._fp8_alpha,
            quantize_output=quantize,
        )
        if o_lora is None:
            return None
        if not quantize:
            return linear(o_lora)
        from .projection import mxfp8_linear_quantized

        return mxfp8_linear_quantized(linear, *o_lora)

    def _project_prequantized_output(self, data, scale):
        """O-A of block-128 E4M3 attention output emitting O-B's MXFP8 input, then O-B; else None.

        ``data`` ([groups, M, D] E4M3) and ``scale`` ([groups, D / 128, pad4(M)] FP32) are the
        fused context kernel's inverse-RoPE'd output, as ``project_sparse_attn_output`` reads it.
        """
        linear = self.o_b_proj
        groups, rank, depth = self.o_a_proj.shape
        if not (
            getattr(self, "_fuse_o_a_mxfp8", False)
            and self.o_a_proj.dtype == torch.float8_e4m3fn
            and self.o_a_proj_scale is not None
            and self.o_a_proj_scale.dtype == torch.float32
            and self.o_a_proj_scale.shape == (groups, rank // 128, depth // 128)
            and data.shape[1] > 0
            and isinstance(linear.quant_method, MXFP8LinearMethod)
            and linear.quant_method.use_cutlass
            and linear.bias is None
            and getattr(linear, "lora", None) is None
            and not is_torch_compiling()
            and not (linear.reduce_output and linear.use_fused_gemm_allreduce)
            and is_sm_100f()
        ):
            return None
        from .output_projection import grouped_output_block128_mxfp8
        from .projection import mxfp8_linear_quantized

        o_lora = grouped_output_block128_mxfp8(
            data, scale, self.o_a_proj, self.o_a_proj_scale.contiguous()
        )
        projected = mxfp8_linear_quantized(linear, *o_lora, cute_min_rows=1024)
        if linear.reduce_output:
            projected = linear.all_reduce(projected)
        return projected

    def _project_output(
        self, output, positions, *, inverse_rope_applied: bool = False, prequantized=None
    ):
        if prequantized is not None:
            projected = self._project_prequantized_output(*prequantized)
            if projected is not None:
                return projected
            return project_sparse_attn_output(
                self, [output.flatten(1)], positions, prequantized=prequantized
            )
        if self._o_a_fp8_block128 and inverse_rope_applied:
            raise ValueError("Grouped 128-block FP8 owns its fused inverse-RoPE/quantization")
        if self.projection_quantization == "bf16":
            if not inverse_rope_applied:
                return project_sparse_attn_output(self, [output.flatten(1)], positions)
            grouped = output.reshape(output.shape[0], self.n_local_groups, -1)
            latent = output.new_empty((output.shape[0], self.n_local_groups, self.o_lora_rank))
            torch.ops.trtllm.bmm_out(
                grouped.transpose(0, 1), self.o_a_proj.transpose(1, 2), latent.transpose(0, 1)
            )
            return self.o_b_proj(latent.flatten(1))
        if self._o_a_fp8_block128:
            projected = self._project_output_small_m(output, positions)
            if projected is None:
                projected = self._project_output_mxfp8(output, positions)
            if projected is not None:
                return projected
            # Preserve the existing inverse-RoPE/quantization/BMM path when
            # its 128-block contract fits; the packed reducer does not rotate.
            return project_sparse_attn_output(self, [output.flatten(1)], positions)
        if self.o_a_proj_scale is None:
            raise RuntimeError("Load CSA2 native grouped projection scales before forward")
        if output.shape[0] == 0:
            return output.new_empty((0, self.o_b_proj.out_features))
        if not inverse_rope_applied:
            torch.ops.trtllm.mla_rope_inplace(
                output,
                positions.reshape(-1),
                self.inverse_rotary_emb.rotary_cos_sin,
                self.num_heads_tp,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                True,
                self.inverse_rotary_emb.is_neox,
            )
        grouped = output.reshape(output.shape[0], self.n_local_groups, -1)
        projections = []
        for group in range(self.n_local_groups):
            data, scales = torch.ops.trtllm.mxfp8_quantize(grouped[:, group].contiguous(), True)
            projections.append(
                torch.ops.trtllm.mxfp8_mxfp8_gemm(
                    data,
                    scales,
                    self.o_a_proj[group],
                    self.o_a_proj_scale[group],
                    self._fp8_alpha,
                    self.dtype,
                )
            )
        return self.o_b_proj(torch.cat(projections, dim=-1))

    @property
    def supports_row_split(self) -> bool:
        """Whether ``project_rows`` and ``attend_rows`` reproduce ``forward`` (native path only)."""
        return (
            not self.backend.sparse_params.use_packed_sparse_attention
            and self._flash_attention is None
        )

    def project_rows(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        """Row-wise projections of ``forward``; they read no metadata, so CUDA graphs can hold them.

        Returns ``(q, swa, index_q, index_q_scale, index_weights)`` for ``attend_rows``.
        """
        qr, q, swa = self._project_query_and_swa(hidden_states, decode_only=False)
        q, _ = self._rope_query(q.reshape(-1, self.num_heads_tp, self.qk_head_dim), positions)
        if swa is None:
            swa = self.wkv(hidden_states)
        if self.rotary_emb.is_neox:
            swa = self._rope(self.kv_norm(swa).unsqueeze(1), positions, 1).squeeze(1)
        index_q, index_q_scale = self._project_index_q(qr, positions)
        index_weights = self._prepare_auxiliary(hidden_states, None, None, None, skip_global=True)
        return (
            q,
            swa,
            index_q,
            index_q_scale,
            index_weights[0],
        )

    def attend_rows(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        q: torch.Tensor,
        swa: torch.Tensor,
        index_q: torch.Tensor | None,
        index_q_scale: torch.Tensor | None,
        index_weights: torch.Tensor | None,
        output: torch.Tensor,
        attn_metadata: CSA2TrtllmMetadata | None,
        rows: int,
    ) -> None:
        """Sparse attention of the first ``rows`` projected rows into ``output``; zero the rest.

        The metadata-driven remainder of ``forward``, including a Full layer's own global
        projection when it was not precomputed.
        """
        if rows:
            layer_idx = self.layer.layer_idx
            compression = attn_metadata.get_compression_batch(layer_idx)
            compressed_positions = (
                attn_metadata.get_compressed_positions(layer_idx)
                if self.layer.mode == CSA2Mode.FULL
                else None
            )
            main_kv = index_k = None
            if (
                self.compressor is not None
                and layer_idx not in attn_metadata.csa2_precomputed_kv_layers
            ):
                live = hidden_states[:rows]
                main_kv, index_k = self._prepare_global(
                    live,
                    attn_metadata.select_global_source(layer_idx, live, None),
                    compression,
                    compressed_positions,
                )
            main_transform, index_transform = self.global_transforms(compressed_positions)
            state = CSA2ForwardState(
                metadata=attn_metadata,
                swa_kv=swa[:rows],
                index_q=None if index_q is None else index_q[:rows],
                index_q_scale=None if index_q_scale is None else index_q_scale[:rows],
                index_weights=None if index_weights is None else index_weights[:rows],
                main_kv=main_kv,
                index_k=index_k,
                swa_transform=None
                if self.rotary_emb.is_neox
                else self._row_transform(
                    positions[:rows], norm=(self.kv_norm.weight, self.kv_norm.variance_epsilon)
                ),
                main_transform=main_transform,
                index_transform=index_transform,
            )
            from tensorrt_llm._torch.attention.backends.interface import (
                AttentionForwardArgs,
                AttentionInputType,
            )

            num_ctx = attn_metadata.num_ctx_tokens
            for start, end in ((0, num_ctx), (num_ctx, rows)):
                if start == end:
                    continue
                tile = q[start:end]
                metadata = attn_metadata.get_query_tile_metadata(
                    tile,
                    0 if self.layer.mode == CSA2Mode.SWA else self.layout.index_topk,
                    query_start=start,
                    staging_dtype=self.backend.staging_dtype,
                    layer_idx=layer_idx,
                )
                args = AttentionForwardArgs(
                    attention_input_type=(
                        AttentionInputType.context_only
                        if metadata.num_contexts
                        else AttentionInputType.generation_only
                    ),
                    attention_sinks=self.attn_sink,
                    output=output[start:end].flatten(1),
                    sparse_backend_args=CSA2BackendForwardArgs(state=state, query_start=start),
                )
                self.backend.forward(tile.flatten(1), None, None, metadata, forward_args=args)
        if rows < output.shape[0]:
            output[rows:].zero_()

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: CSA2TrtllmMetadata,
        *,
        global_hidden_states: torch.Tensor | None = None,
        hidden_states_mxfp8: MxFp8QuantizedTensor | None = None,
    ) -> torch.Tensor:
        """Return [tokens, hidden_size]; all packed rows must retain source order.

        ``positions`` and ``compressed_positions`` are contiguous CUDA int32
        tensors. ``compressed_positions`` contains the first token position of each
        completed group. ``global_hidden_states`` is the encoder output when
        decoder global KV is prepared from more rows than its SWA replay.
        ``hidden_states_mxfp8`` is ``hidden_states`` quantized as
        ``torch.ops.trtllm.mxfp8_quantize(hidden_states, True)`` would, for the fused MXFP8
        projection (:attr:`takes_mxfp8_input`).
        """
        precomputed = self.layer.layer_idx in attn_metadata.csa2_precomputed_kv_layers
        compression = attn_metadata.get_compression_batch(self.layer.layer_idx)
        compressed_positions = (
            attn_metadata.get_compressed_positions(self.layer.layer_idx)
            if self.layer.mode == CSA2Mode.FULL
            else None
        )
        global_source = (
            attn_metadata.select_global_source(
                self.layer.layer_idx, hidden_states, global_hidden_states
            )
            if self.compressor is not None and not precomputed
            else None
        )
        overlap = (
            self.aux_stream is not None
            and do_multi_stream()
            and self.layer.mode in (CSA2Mode.FULL, CSA2Mode.REINDEX)
        )
        weights_stream = None
        if overlap:
            if self.aux_stream.device != hidden_states.device:
                raise ValueError("CSA2 auxiliary stream must use the input CUDA device")
            self._prepare_start.record()
            if (
                self.layer.mode == CSA2Mode.FULL
                and global_source is not None
                and attn_metadata.num_ctx_tokens == 0
            ):
                # The index weights and the paged indexer descriptors need neither each other
                # nor the compressor's rows: in forwards without context tokens, derive them on
                # two more side streams next to the compression chain.
                weights_stream = csa2_side_stream(hidden_states.device)
                indexer = getattr(self.backend, "indexer", None)
                descriptor_stream = None
                if indexer is not None:
                    descriptor_stream = csa2_side_stream(hidden_states.device, 1)
                    with torch.cuda.stream(descriptor_stream):
                        self._prepare_start.wait()
                        indexer.prepare_paged_descriptors(attn_metadata, hidden_states.shape[0])
                        self._descriptors_done.record()
                with torch.cuda.stream(weights_stream):
                    self._prepare_start.wait()
                    hidden_states.record_stream(weights_stream)
                    side_weights = self._project_index_weights(hidden_states)
                    if descriptor_stream is not None:
                        self._descriptors_done.wait()
                    self._weights_done.record()
            with torch.cuda.stream(self.aux_stream):
                self._prepare_start.wait()
                for tensor in (hidden_states, global_source, compressed_positions):
                    if tensor is not None:
                        tensor.record_stream(self.aux_stream)
                auxiliary = self._prepare_auxiliary(
                    hidden_states,
                    global_source,
                    compression,
                    compressed_positions,
                    skip_global=precomputed,
                    skip_weights=weights_stream is not None,
                )
                self._prepare_done.record()
            if weights_stream is not None:
                auxiliary = (side_weights, *auxiliary[1:])
        query_overlap = (
            self.aux_stream is not None
            and self._overlap_query_kv
            and do_multi_stream()
            and self.layer.mode in (CSA2Mode.SWA, CSA2Mode.REUSE)
            and attn_metadata.num_ctx_tokens == 0
            and hidden_states.shape[0] > 0
            and _FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.get()
            and self.takes_mxfp8_input
            and self.backend.compute_backend == "trtllm"
            and not self.backend.sparse_params.use_packed_sparse_attention
        )
        # Full/Reindex decode projects the attention query on the auxiliary stream after the
        # indexer inputs; staging waits for it.
        late_query = (
            overlap
            and attn_metadata.num_ctx_tokens == 0
            and hidden_states.shape[0] > 0
            and self.backend.compute_backend == "trtllm"
            and not self.backend.sparse_params.use_packed_sparse_attention
        )
        # Captured context tiles use generation FMHA instead of the shared Q/KV context scale.
        fused_query = (
            hidden_states.shape[0] >= 1024
            and self._fuse_q_rope_projection
            and attn_metadata.num_ctx_tokens >= 1024
            and not attn_metadata.is_cuda_graph
            and self.qk_head_dim == 512
            and self.qk_rope_head_dim == 64
            and not self.rotary_emb.is_neox
            and self.wq_a.weight.shape[0] % 128 == 0
            and self.wq_a.weight.shape[0] <= 8192
            and self._fuse_q_norm_quant
            and self.backend.compute_backend == "trtllm"
            and self.backend.native_fp8
            and not self.backend.sparse_params.use_packed_sparse_attention
            and self._can_fuse_prefill_projection(hidden_states)
            and not torch.cuda.is_current_stream_capturing()
            and is_sm_100f()
        )
        # A direct-decode forward leaves the query RoPE to its attention launch.
        extra_width = 0 if self.layer.mode == CSA2Mode.SWA else self.layout.index_topk
        direct = (
            attn_metadata.num_ctx_tokens == 0
            and self._flash_attention is None
            and not self.rotary_emb.is_neox
            and self.qk_rope_head_dim == 64
            and self.backend.direct_decode_serves(
                hidden_states.shape[0], 0, pad_up(extra_width, 128), hidden_states.device
            )
        )
        quant_q = q_amax = None
        unit_q = touched = False
        if fused_query:
            shape = (hidden_states.shape[0], self.num_heads_tp * self.qk_head_dim)
            quant_q = torch.empty(shape, dtype=torch.float8_e4m3fn, device=hidden_states.device)
            q = torch.empty(shape, dtype=self.dtype, device=hidden_states.device)
            # A context-only batch run by the fused context kernel reads only the E4M3 Q, which
            # then stays at unit scale; the BF16 Q buffer is never written.
            unit_q = (
                getattr(self, "_unit_scale_context_query", False)
                and attn_metadata.num_ctx_tokens == shape[0]
                and self.backend.runs_fused_context(
                    attn_metadata.get_query_tile_metadata(
                        q.view(shape[0], self.num_heads_tp, self.qk_head_dim),
                        0 if self.layer.mode == CSA2Mode.SWA else self.layout.index_topk,
                        query_start=0,
                        staging_dtype=self.backend.staging_dtype,
                        layer_idx=self.layer.layer_idx,
                    )
                )
            )
            # Only the index-Q projection reads the normalized latent, as MXFP8 operands.
            qr, _, swa = self._project_prefill_query_and_swa(
                hidden_states,
                positions=positions,
                quant_q=quant_q,
                q_out=q,
                fp8_only=unit_q,
                latent_mxfp8=getattr(self, "_mxfp8_query_latent", False),
            )
        else:
            fp8_query = self._fp8_query_request(hidden_states, positions, attn_metadata)
            qr, q, swa = self._project_query_and_swa(
                hidden_states,
                decode_only=attn_metadata.num_ctx_tokens == 0,
                query_stream=self.aux_stream if query_overlap else None,
                late_stream=self.aux_stream if late_query else None,
                fp8_query=fp8_query,
                prequantized=hidden_states_mxfp8,
            )
            if fp8_query is not None and fp8_query.applied:
                # Staging reads only the E4M3 Q; this BF16 Q keeps the shapes and is never read.
                quant_q = fp8_query.quant_q
                q_amax = fp8_query.word
                q = torch.empty(quant_q.shape, dtype=self.dtype, device=quant_q.device)
        # CSA2 never applies V4's projected-Q per-head normalization.
        with torch.cuda.stream(self.aux_stream if query_overlap or late_query else None):
            q = q.reshape(-1, self.num_heads_tp, self.qk_head_dim)
            if not fused_query and q_amax is None and not direct:
                word = (
                    self._q_amax_word(q.device)
                    if self._stages_native_fp8_generation(attn_metadata)
                    else None
                )
                q, raised = self._rope_query(q, positions, word)
                if raised:
                    q_amax = word
            if query_overlap:
                self._prepare_done.record()
                touched = self._touch_output_weights(q.shape[0])
            elif late_query:
                self._query_done.record()
        if query_overlap or late_query:
            q.record_stream(torch.cuda.current_stream(hidden_states.device))
        if swa is None:
            swa = self.wkv(hidden_states)
        swa_transform = None
        if self.rotary_emb.is_neox:
            swa = self._rope(self.kv_norm(swa).unsqueeze(1), positions, 1).squeeze(1)
        else:
            swa_transform = self._row_transform(
                positions, norm=(self.kv_norm.weight, self.kv_norm.variance_epsilon)
            )
        main_transform, index_transform = self.global_transforms(compressed_positions)
        index_q, index_q_scale = self._project_index_q(qr, positions)
        if overlap:
            # The cache write needs the compressor's rows (and the index weights unless they
            # were split onto the second side stream, which the indexer joins: ``index_ready``).
            self._prepare_done.wait()
            current = torch.cuda.current_stream(hidden_states.device)
            for tensor in auxiliary:
                if tensor is not None:
                    tensor.record_stream(current)
        else:
            auxiliary = self._prepare_auxiliary(
                hidden_states,
                global_source,
                compression,
                compressed_positions,
                skip_global=precomputed,
            )
        index_weights, main_kv, index_k = auxiliary
        from tensorrt_llm._torch.attention.backends.interface import (
            AttentionForwardArgs,
            AttentionInputType,
        )

        # Only the actual packed path can consume these fields. Keep the
        # established grouped128 inverse-RoPE/FP8 quantizer preferred.
        inverse_rope_applied = (
            self.backend.sparse_params.fuse_packed_output_rope
            and self.backend.sparse_params.use_packed_sparse_attention
            and not self.inverse_rotary_emb.is_neox
            and not self._o_a_fp8_block128
        )
        # A context-only batch run by the fused context kernel gets the grouped128 inverse-RoPE/FP8
        # projection input straight from that kernel (data, scale, positions, cos/sin).
        output_fp8 = None
        if (
            getattr(self.backend, "fused_context", False)
            and self._o_a_fp8_block128
            and self.o_a_proj.dtype == torch.float8_e4m3fn
            and not self.inverse_rotary_emb.is_neox
            and attn_metadata.num_ctx_tokens == q.shape[0]
        ):
            from tensorrt_llm._torch.attention.mla import _is_cute_dsl_fp8_bmm_available

            if _is_cute_dsl_fp8_bmm_available():
                width = self.num_heads_tp // self.n_local_groups * self.qk_head_dim
                output_fp8 = (
                    q.new_empty(
                        (self.n_local_groups, q.shape[0], width), dtype=torch.float8_e4m3fn
                    ),
                    q.new_empty(
                        (self.n_local_groups, width // 128, (q.shape[0] + 3) // 4 * 4),
                        dtype=torch.float32,
                    ),
                    positions.reshape(-1),
                    self.inverse_rotary_emb.rotary_cos_sin,
                )
        state = CSA2ForwardState(
            metadata=attn_metadata,
            swa_kv=swa,
            index_q=index_q,
            index_q_scale=index_q_scale,
            index_weights=index_weights,
            main_kv=main_kv,
            index_k=index_k,
            swa_transform=swa_transform,
            main_transform=main_transform,
            index_transform=index_transform,
            index_ready=self._weights_done if weights_stream is not None else None,
            output_position_ids=positions if inverse_rope_applied else None,
            output_rotary_cos_sin=self.inverse_rotary_emb.rotary_cos_sin
            if inverse_rope_applied
            else None,
            output_fp8=output_fp8,
            query_rope=(positions.reshape(-1), self.rotary_emb.rotary_cos_sin) if direct else None,
        )
        output = torch.empty_like(q, memory_format=torch.contiguous_format)
        # One complete call per real phase; shared KV staging removes the
        # per-query expansion that previously imposed an artificial chunk cap.
        for start, end in (
            (0, attn_metadata.num_ctx_tokens),
            (attn_metadata.num_ctx_tokens, q.shape[0]),
        ):
            if start == end:
                continue
            tile = q[start:end]
            metadata = attn_metadata.get_query_tile_metadata(
                tile,
                extra_width,
                query_start=start,
                staging_dtype=self.backend.staging_dtype,
                layer_idx=self.layer.layer_idx,
            )
            args = AttentionForwardArgs(
                attention_input_type=(
                    AttentionInputType.context_only
                    if metadata.num_contexts
                    else AttentionInputType.generation_only
                ),
                attention_sinks=self.attn_sink,
                output=output[start:end].flatten(1),
                sparse_backend_args=CSA2BackendForwardArgs(
                    state=state,
                    query_start=start,
                    query_ready=(
                        self._prepare_done
                        if query_overlap
                        else self._query_done
                        if late_query
                        else None
                    ),
                    late_query_ready=query_overlap,
                    projected_quant_q=(
                        quant_q[start:end] if quant_q is not None and start == 0 else None
                    ),
                    projected_q_unit_scale=unit_q,
                    projected_q_amax=q_amax,
                ),
            )
            if self.backend.sparse_params.use_packed_sparse_attention:
                self.backend.forward_packed(tile.flatten(1), metadata, args)
            elif self._flash_attention is not None:
                compute = (
                    self._flash_attention.forward_context
                    if metadata.num_contexts
                    else self._flash_attention.forward_generation
                )
                compute(tile.flatten(1), metadata, args)
            else:
                self.backend.forward(tile.flatten(1), None, None, metadata, forward_args=args)
        if state.output_fp8_written:
            return self._project_output(output, positions, prequantized=output_fp8[:2])
        projected = self._project_output(
            output, positions, inverse_rope_applied=inverse_rope_applied
        )
        if touched:
            torch.cuda.current_stream(hidden_states.device).wait_event(self._touch_done)
        return projected
