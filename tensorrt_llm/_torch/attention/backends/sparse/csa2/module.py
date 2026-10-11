# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 attention module for integration into the causal encoder/decoder."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.attention.backends.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.attention.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.distributed import AllReduceStrategy
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.multi_stream_utils import do_multi_stream
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..deepseek_v4.module import project_sparse_attn_output
from .compressor import CSA2Compressor
from .metadata import CSA2TrtllmMetadata
from .params import (
    CSA2BackendForwardArgs,
    CSA2ForwardState,
    CSA2Layout,
    CSA2Mode,
    CSA2Params,
    RowTransform,
)


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
        self._fuse_index_q = sparse_params.fuse_index_q
        if sparse_params.fuse_packed_output_rope and pos_embd_params.is_neox:
            raise ValueError("CSA2 packed output RoPE requires interleaved rotary channels")
        quant_config = None
        if projection_quantization == "mxfp8":
            if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
                (10, 0),
                (10, 3),
            ):
                raise ValueError("CSA2 native MXFP8 projections require SM100/SM103")
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

    def _row_transform(self, positions, norm=None) -> RowTransform:
        """Norm and RoPE to apply while cache rows are quantized (interleaved RoPE only)."""
        rope = None
        if not self.rotary_emb.is_neox:
            table = self.rotary_emb.rotary_cos_sin.view(-1, self.qk_rope_head_dim)
            rope = (positions.reshape(-1).int(), table, self.qk_rope_head_dim)
        return RowTransform(norm=norm, rope=rope)

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
    ):
        main_kv, index_k = (
            (None, None)
            if skip_global
            else self._prepare_global(
                hidden_states, global_hidden_states, compression, compressed_positions
            )
        )
        index_weights = None
        if self.layer.mode in (CSA2Mode.FULL, CSA2Mode.REINDEX):
            projected = self.index_weights_proj(hidden_states)
            # One launch for ``(projected * scale).float()``, bit for bit.
            index_weights = torch.mul(
                projected,
                self._index_weight_scale,
                out=torch.empty(projected.shape, dtype=torch.float32, device=projected.device),
            )
        return index_weights, main_kv, index_k

    def _project_index_q(self, qr, positions):
        if self.layer.mode not in (CSA2Mode.FULL, CSA2Mode.REINDEX):
            return None, None
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
        query = self.index_wq_b(qr).reshape(-1, self.index_heads, self.index_head_dim)
        return self._rope(query, positions, self.index_heads), None

    def _project_output(self, output, positions, *, inverse_rope_applied: bool = False):
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: CSA2TrtllmMetadata,
        *,
        global_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return [tokens, hidden_size]; all packed rows must retain source order.

        ``positions`` and ``compressed_positions`` are contiguous CUDA int32
        tensors. ``compressed_positions`` contains the first token position of each
        completed group. ``global_hidden_states`` is the encoder output when
        decoder global KV is prepared from more rows than its SWA replay.
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
        if overlap:
            if self.aux_stream.device != hidden_states.device:
                raise ValueError("CSA2 auxiliary stream must use the input CUDA device")
            self._prepare_start.record()
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
                )
                self._prepare_done.record()
        qr = self.q_norm(self.wq_a(hidden_states))
        # CSA2 never applies V4's projected-Q per-head normalization.
        q = self._rope(
            self.wq_b(qr).reshape(-1, self.num_heads_tp, self.qk_head_dim),
            positions,
            self.num_heads_tp,
        )
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
            output_position_ids=positions if inverse_rope_applied else None,
            output_rotary_cos_sin=self.inverse_rotary_emb.rotary_cos_sin
            if inverse_rope_applied
            else None,
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
                0 if self.layer.mode == CSA2Mode.SWA else self.layout.index_topk,
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
                sparse_backend_args=CSA2BackendForwardArgs(state=state, query_start=start),
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
        return self._project_output(output, positions, inverse_rope_applied=inverse_rope_applied)
