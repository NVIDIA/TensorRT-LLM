# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 adapters for the V4 decoder and CSA2 attention.

Checkpoint-specific behavior (``<checkpoint>/inference/model.py``):

* Nonzero compression ratios, including unpooled ratio 1, use long-range RoPE.
* mHC uses ``hc_eps`` for Sinkhorn and ``rms_norm_eps`` for mixer statistics.
  Each sublayer consumes the previous sublayer's ``pre`` and forwards its own.
* The final ``pre`` collapses the stream to ``[N, hidden]`` without ``hc_head``.
* Engram uses sharded FP8 tables, keys-first WKV and an FP32 gate, without convolution.
* Dense checkpoint weights use FP8 with block-32 UE8M0 scales and V4.1 key names.

MoE routing, n-gram hashing and collapsed-logit processing reuse the shared code.
"""

import copy
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, cast

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm.disaggregated_params import DisaggScheduleStyle
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention.backends.sparse.csa2.ced import prepare_ced_global_kv
from ..attention.backends.sparse.csa2.decoder_replay import (
    DecoderReplayPlan,
    enter_decoder_replay,
    enter_remote_tail_decoder,
    exit_decoder_replay,
    gather_replayed_rows,
    plan_decoder_replay,
    scatter_replayed_rows,
)
from ..attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
from ..attention.backends.sparse.csa2.module import DeepseekV41Attention as CSA2Attention
from ..attention.backends.sparse.csa2.params import CSA2Mode
from ..configs.deepseek_v41 import (
    DeepseekV41QuantLayout,
    assert_weight_layout,
    decoder_bounded_replay_enabled,
    disagg_context_decoder_skipping_enabled,
    encoder_replay_enabled,
    quant_role_for_weight_key,
)
from ..distributed import AllReduce, AllReduceParams, AllReduceStrategy, allgather
from ..model_config import ModelConfig
from ..modules.engram import Engram, ShardedFp8MultiHeadEmbedding, engram_gate
from ..modules.engram.projection import EngramFp8Projection
from ..modules.mhc.hyper_connection import HCState, mHC
from ..speculative import SpecMetadata
from ..speculative.dspark import DSparkSpecMetadata
from ..utils import AuxStreamType
from .checkpoints.base_weight_loader import ConsumableWeightsDict
from .modeling_deepseekv4 import (
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
    DeepseekV4WeightLoader,
    _deepseek_v4_layer_compress_ratio,
    _deepseek_v4_pos_embd_params,
    _remap_deepseek_v4_checkpoint_keys,
    weight_dequant,
)
from .modeling_utils import register_auto_model

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    from ..pyexecutor.llm_request import LlmRequest


def _engram_history_text_mask(request: "LlmRequest", start: int, end: int) -> list[bool] | None:
    """Mark text positions in a bounded lookback using request image spans."""
    positions = getattr(request, "multimodal_positions", None)
    lengths = getattr(request, "multimodal_lengths", None)
    if positions is None and lengths is None:
        params = getattr(request, "py_disaggregated_params", None)
        positions = getattr(params, "multimodal_positions", None)
        lengths = getattr(params, "multimodal_lengths", None)
    if positions is None or lengths is None:
        return None
    mask = [True] * (end - start)
    for image_start, length in zip(positions, lengths, strict=True):
        for position in range(max(start, image_start), min(end, image_start + length)):
            mask[position - start] = False
    return mask


# ---------------------------------------------------------------------------
# 1. Per-layer RoPE
# ---------------------------------------------------------------------------


def _deepseek_v41_pos_embd_params(
    config: PretrainedConfig,
    model_config: ModelConfig,
    layer_idx: Optional[int],
    predicted_tokens_per_seq: int = 1,
    *,
    long_range: Optional[bool] = None,
):
    """Use long-range RoPE for every nonzero ratio, including unpooled ratio 1."""
    ratio = _deepseek_v4_layer_compress_ratio(config, model_config, layer_idx)
    return _deepseek_v4_pos_embd_params(
        config,
        model_config,
        layer_idx,
        predicted_tokens_per_seq,
        long_range=(ratio != 0) if long_range is None else long_range,
    )


class DeepseekV41Attention(CSA2Attention):
    """Adapt the model's decoder contract to the independent CSA2 component."""

    q_b_norm_enabled = False

    def __init__(self, model_config, layer_idx, aux_stream_dict=None, reduce_output=True):
        config = model_config.pretrained_config
        sparse_config = model_config.sparse_attention_config
        if sparse_config is None:
            raise ValueError("DeepSeek-V4.1 requires CSA2 sparse attention configuration")
        params = sparse_config.to_sparse_params(pretrained_config=config)
        positional = _deepseek_v41_pos_embd_params(config, model_config, layer_idx)
        super().__init__(
            params.layout,
            layer_idx,
            positional,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            rope_head_dim=config.qk_rope_head_dim,
            q_lora_rank=config.q_lora_rank,
            o_lora_rank=config.o_lora_rank,
            num_groups=config.o_groups,
            index_heads=config.index_n_heads,
            index_head_dim=config.index_head_dim,
            eps=config.rms_norm_eps,
            mapping=model_config.mapping,
            sparse_params=params,
            projection_quantization="mxfp8",
            aux_stream=(aux_stream_dict or {}).get(AuxStreamType.MlaCompressor),
            allreduce_strategy=model_config.allreduce_strategy,
            kv_cache_dtype=model_config.extra_attrs.get("kv_cache_dtype", "auto"),
            use_cute_dsl_blockscaling_bmm=model_config.use_cute_dsl_blockscaling_bmm,
            use_cute_dsl_bf16_bmm=model_config.use_cute_dsl_bf16_bmm,
        )
        self.pos_embd_params = positional
        self.sparse_params = params
        self.reduce_output = reduce_output
        self.o_b_proj.reduce_output = reduce_output

    def forward(self, position_ids, hidden_states, attn_metadata, all_reduce_params=None, **kwargs):
        if kwargs.get("lora_params") is not None:
            raise NotImplementedError("DeepSeek-V4.1 CSA2 does not implement attention LoRA")
        # The original sparse O-LoRA helper did not consume all_reduce_params;
        # TP reduction remains owned by the row-parallel output projection.
        return super().forward(
            hidden_states, position_ids.reshape(-1).to(torch.int32), attn_metadata
        )

    def load_weights(self, weights):
        """Load one complete attention subtree with exact key coverage."""
        if len(weights) != 1:
            raise ValueError("CSA2 attention expects one checkpoint weight dictionary")
        renamed = {}
        stems = {
            "q_a_proj": "wq_a",
            "q_b_proj": "wq_b",
            "kv_a_proj_with_mqa": "wkv",
            "q_a_layernorm": "q_norm",
            "kv_a_layernorm": "kv_norm",
            "o_b_proj": "wo_b",
        }
        for key, value in weights[0].items():
            if key == "o_a_proj":
                key = "wo_a.weight"
            elif key.startswith("o_a_proj."):
                key = "wo_a." + key[len("o_a_proj.") :]
            else:
                head, sep, tail = key.partition(".")
                key = stems.get(head, head) + sep + tail
            renamed[key] = value[:] if not isinstance(value, torch.Tensor) else value
        aliases = {
            "o_a_proj": "wo_a.weight",
            "o_b_proj.weight": "wo_b.weight",
            "index_wq_b.weight": "indexer.wq_b.weight",
            "index_weights_proj.weight": "indexer.weights_proj.weight",
            "index_wk.weight": "indexer.wk.weight",
            "index_k_norm.weight": "indexer.k_norm.weight",
        }
        expected = {
            aliases.get(name, name)
            for name, _ in self.named_parameters()
            if not name.endswith("weight_scale")
        }
        scales = {
            key.removesuffix("weight") + "weight_scale_inv"
            for key in expected
            if key.endswith("weight")
            and key in renamed
            and renamed[key].dtype == torch.float8_e4m3fn
        }
        extras = set(renamed) - expected - scales
        if extras:
            raise ValueError(f"CSA2 attention weights have no consumer: {sorted(extras)}")
        self.load_hf_weights(renamed)
        self._checkpoint_loaded_keys = frozenset(weights[0])


# ---------------------------------------------------------------------------
# 5. Engram
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _AttentionDPEngramEmbeddings:
    embeddings: torch.Tensor
    local_start: int
    local_num_tokens: int


class DeepseekV41Engram(Engram):
    """Pinned-CPU FP8 lookup, keys-first WKV and FP32 gating, without convolution.

    WKV splits as ``[hc_mult * dim, dim] -> (keys, value)``. Table scales are
    UE8M0 per 32 values; the gate and residual update cast only at the output.
    The gate folds ``q_weight * k_weight`` into a cached FP32 product.
    """

    def __init__(
        self,
        layer_id: int,
        config,
        vocab_sizes_flat: Optional[List[int]] = None,
        stream: Optional[torch.cuda.Stream] = None,
        mapping: Optional[Mapping] = None,
        fp8_block_size: int = 32,
        allreduce_strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
    ):
        # The base constructor calls _make_multi_head_embedding with these fields.
        self.mapping = mapping
        self.tp_size = 1 if mapping is None else mapping.tp_size
        self.tp_rank = 0 if mapping is None else mapping.tp_rank
        self.enable_attention_dp = (
            mapping is not None and mapping.enable_attention_dp and self.tp_size > 1
        )
        self.fp8_block_size = fp8_block_size

        super().__init__(
            layer_id=layer_id,
            config=config,
            vocab_sizes_flat=vocab_sizes_flat,
            stream=stream,
        )
        self.register_buffer("_gate_norm_product", None, persistent=False)

        # Row shards use the model's all-reduce strategy; head shards use all-gather.
        self.embed_all_reduce = None
        if mapping is not None and self.tp_size > 1 and not self.multi_head_embedding.shard_heads:
            embedding_mapping = mapping
            if self.enable_attention_dp:
                # Table shards still reduce over the full TP group under attention DP.
                embedding_mapping = copy.copy(mapping)
                embedding_mapping.enable_attention_dp = False
            self.embed_all_reduce = AllReduce(
                mapping=embedding_mapping, strategy=allreduce_strategy
            )

    @torch.no_grad()
    def cache_derived_state(self) -> None:
        self._gate_norm_product = (
            self.query_norm_weight.float() * self.key_norm_weight.float()
        ).contiguous()

    def post_load_weights(self) -> None:
        self.cache_derived_state()

    @torch.no_grad()
    def warmup_kernels(self) -> None:
        """Compile both residual-gate variants without projections or collectives."""
        if not self.query_norm_weight.is_cuda:
            return
        hc, dim = self.query_norm_weight.shape
        hidden = self.query_norm_weight.new_zeros((1, hc, dim))
        kv = self.query_norm_weight.new_zeros((1, (hc + 1) * dim))
        # Token count only changes the launch grid, not the compiled signature.
        for product in (None, self._gate_norm_product):
            engram_gate(
                hidden,
                kv,
                self.query_norm_weight,
                self.key_norm_weight,
                self.norm_eps,
                add_residual=True,
                norm_weight_product=product,
            )

    def _apply(self, fn, recurse: bool = True):
        # A module dtype conversion must not narrow the FP32 product.
        self._gate_norm_product = None
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self._gate_norm_product = None
        return super()._load_from_state_dict(*args, **kwargs)

    def _make_multi_head_embedding(self, list_of_N: List[int], D: int) -> nn.Module:
        """Shard the FP8 table without changing hash indices or checkpoint keys.

        Head count and TP size select the cut; table shards stay in pinned CPU memory.
        """
        return ShardedFp8MultiHeadEmbedding(
            list_of_N=list_of_N,
            D=D,
            block_size=self.fp8_block_size,
            dtype=self.config.dtype,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )

    def _make_short_conv(self) -> None:
        return None

    def _make_kv_projection(self, in_features: int, out_features: int) -> nn.Module:
        return EngramFp8Projection(in_features, out_features, self.config.dtype)

    def _combine_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Reassemble per-rank contributions into ``[T, H * D]``.

        Equal contiguous head shards concatenate; zero-masked row shards sum.
        """
        if self.embed_all_reduce is not None:
            return self.embed_all_reduce(
                embeddings, all_reduce_params=AllReduceParams(enable_allreduce=True)
            )
        if self.multi_head_embedding.shard_heads:
            return allgather(embeddings, self.mapping, dim=-1)
        return embeddings

    def precompute(
        self,
        hash_indices: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
        *,
        all_rank_num_tokens: list[int] | None = None,
    ) -> torch.Tensor | _AttentionDPEngramEmbeddings:
        """Launch the local lookup; leave its TP collective on the model stream.

        Returns this rank's flattened contribution, with ``sync_event`` marking
        completion of the lookup and optional cast. ``precompute_kv`` or
        ``forward`` recombines the shards after waiting on that event.
        Waiting here would serialize layer 0 behind both L1 and L14 lookups.
        The collective stays on the model stream because it shares communicator
        ordering and all-reduce workspaces with attention and MoE.

        Attention DP gathers token rows before lookup and retains the local span.
        """
        local_start = 0
        local_num_tokens = hash_indices.shape[0]
        if self.enable_attention_dp:
            if (
                not isinstance(all_rank_num_tokens, (list, tuple))
                or len(all_rank_num_tokens) != self.tp_size
                or any(type(count) is not int or count < 0 for count in all_rank_num_tokens)
                or all_rank_num_tokens[self.tp_rank] != local_num_tokens
            ):
                raise ValueError(
                    "Attention-DP Engram requires all_rank_num_tokens with one nonnegative "
                    "integer per TP rank, matching the local hash-row count."
                )
            local_start = sum(all_rank_num_tokens[: self.tp_rank])
            # Every table shard must look up the same globally ordered token rows.
            if sum(all_rank_num_tokens):
                hash_indices = allgather(
                    hash_indices, self.mapping, dim=0, sizes=all_rank_num_tokens
                )

        embeddings = self._lookup_embeddings(hash_indices, dtype)
        if self.enable_attention_dp:
            return _AttentionDPEngramEmbeddings(embeddings, local_start, local_num_tokens)
        return embeddings

    def _lookup_embeddings(
        self, hash_indices: torch.Tensor, dtype: torch.dtype | None
    ) -> torch.Tensor:
        if self.stream is None:
            embeddings = self.multi_head_embedding(hash_indices)
            embeddings = embeddings.flatten(start_dim=-2)
            return embeddings.to(dtype) if dtype is not None else embeddings

        # Stream waits order work; record_stream also prevents allocator reuse.
        caller_stream = torch.cuda.current_stream()
        self.stream.wait_stream(caller_stream)
        hash_indices.record_stream(self.stream)
        with torch.cuda.stream(self.stream):
            # Leave SM capacity for concurrent model work.
            embeddings = self.multi_head_embedding(hash_indices, background=True)
            embeddings = embeddings.flatten(start_dim=-2)
            if dtype is not None:
                embeddings = embeddings.to(dtype)
            self.sync_event.record()
        embeddings.record_stream(caller_stream)
        return embeddings

    def precompute_kv(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project local lookup results; the caller must wait on ``sync_event`` if streamed."""
        if self.stream is None:
            return self.kv_proj(self._combine_embeddings(embeddings))

        caller_stream = torch.cuda.current_stream()
        caller_stream.wait_event(self.sync_event)
        # TP collectives share communicator ordering with attention and MoE.
        embeddings = self._combine_embeddings(embeddings)
        self.stream.wait_stream(caller_stream)
        embeddings.record_stream(self.stream)
        with torch.cuda.stream(self.stream):
            kv = self.kv_proj(embeddings)
            self.sync_event.record()
        kv.record_stream(caller_stream)
        return kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        embeddings: torch.Tensor | _AttentionDPEngramEmbeddings,
        conv_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        *,
        add_residual: bool = False,
        token_mask: Optional[torch.Tensor] = None,
        precomputed_kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Recombine lookup shards, project WKV, then run the fused FP32 gate.

        ``add_residual=True`` preserves the reference's single cast after the
        FP32 residual addition. The default returns a delta for standalone users.
        A supplied ``precomputed_kv`` must be ready on the caller's stream.
        """
        if use_cache or conv_state is not None:
            raise ValueError(
                "DeepSeek-V4.1's Engram has no convolution, so there is no conv "
                "state to carry across decode steps."
            )
        if precomputed_kv is None:
            if isinstance(embeddings, _AttentionDPEngramEmbeddings):
                if embeddings.local_num_tokens != hidden_states.shape[0]:
                    raise ValueError(
                        "Attention-DP Engram local token count does not match hidden states."
                    )
                # An idle rank still contributes its table shard to active ranks.
                combined = embeddings.embeddings
                if combined.shape[0]:
                    combined = self._combine_embeddings(combined)
                embeddings = combined.narrow(0, embeddings.local_start, embeddings.local_num_tokens)
            else:
                if self.enable_attention_dp:
                    raise ValueError("Attention-DP Engram requires globally prefetched embeddings.")
                embeddings = self._combine_embeddings(embeddings)
            if hidden_states.shape[0] == 0:
                return hidden_states if add_residual else torch.zeros_like(hidden_states)
            kv = self.kv_proj(embeddings)
        else:
            kv = precomputed_kv
        output = engram_gate(
            hidden_states,
            kv,
            self.query_norm_weight,
            self.key_norm_weight,
            self.norm_eps,
            add_residual=add_residual,
            norm_weight_product=self._gate_norm_product,
        )
        if token_mask is not None:
            # Image positions receive no Engram delta.
            output = torch.where(
                token_mask[:, None, None], output, hidden_states if add_residual else 0
            )
        return output


# ---------------------------------------------------------------------------
# 2 + 3. mHC constants and the lagged-`pre` wiring
# ---------------------------------------------------------------------------


class DeepseekV41DecoderLayer(DeepseekV4DecoderLayer):
    """V4 decoder scaffolding with CSA2 attention and lagged mHC coefficients.

    Each sublayer consumes the previous ``pre`` and carries its own in
    ``HCState.pre_mix``. Lagged fused boundaries include RMSNorm and absorb
    deferred post updates when the next layer does not need the residual first.
    """

    attention_cls = DeepseekV41Attention

    def _make_mhc(self) -> mHC:
        """Use hc_eps for sigmoid/Sinkhorn and rms_norm_eps for mixer statistics.

        V4.1 uses 1e-6 and 1e-20 respectively; the mHC module defaults differ.
        """
        config = self.config
        hc_eps = config.hc_eps
        return mHC(
            config.hc_mult,
            config.hidden_size,
            config.hc_sinkhorn_iters,
            dtype=torch.float32,
            eps=hc_eps,
            norm_eps=config.rms_norm_eps,
            sinkhorn_eps=hc_eps,
            post_mult_value=2.0,
        )

    def _make_engram(self, layer_idx: int, engram_config, vocab_sizes_flat, stream):
        return DeepseekV41Engram(
            layer_id=layer_idx,
            config=engram_config,
            vocab_sizes_flat=vocab_sizes_flat,
            stream=stream,
            mapping=self.mapping,
            allreduce_strategy=self.model_config.allreduce_strategy,
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Attention completes TP reduction before mHC; do not reduce the FFN input again.
        self.fusion_config.PRE_MOE_FUSION = False
        self.disable_attn_allreduce = not self.self_attn.reduce_output
        # Override even environment-enabled V4 fusion: it cannot consume lagged pre.
        self.enable_fused_hc = False
        self.defer_post_mapping = False
        # V4.1's own cross-layer deferral: hand hc_ffn.post_mapping to the next layer,
        # whose fused lagged entry (mHC.fused_hc_lagged) absorbs it. Set by
        # DeepseekV41ForCausalLM.post_load_weights once the neighbours are known.
        self._v41_defer_post_mapping = False

    def _decoder_global_input(self, hc_state: HCState) -> torch.Tensor:
        """Use the ordinary fused boundary's collapse/norm and BF16 rounding."""
        if hc_state.is_deferred or hc_state.pre_mix is None:
            raise ValueError("Decoder global production requires resolved encoder states")
        # The fused boundary also computes coefficients. Only its normalized
        # input is needed for Global production; reusing it preserves the same
        # arithmetic as ordinary prefill without restoring the retired kernels.
        _, _, _, layer_input = self.hc_attn.pre_mapping_lagged(
            hc_state.residual,
            hc_state.pre_mix,
            norm_weight=self.input_layernorm.weight,
            norm_eps=self.input_layernorm.variance_epsilon,
        )
        return layer_input

    def forward(
        self,
        position_ids: torch.IntTensor,
        hc_state: HCState,
        attn_metadata,
        spec_metadata: Optional[SpecMetadata] = None,
        input_ids: Optional[torch.IntTensor] = None,
        engram_embeddings=None,
        image_mask: Optional[torch.Tensor] = None,
        precomputed_engram_kv: torch.Tensor | None = None,
        **kwargs,
    ) -> HCState:
        """Run Engram, attention and MoE with lagged mHC coefficients.

        Compute coefficients from the uncollapsed residual, collapse with the
        incoming pre, then apply the sublayer and post update to that same residual.
        Carry the FFN's pre to the next block in the returned HCState.
        """
        residual = hc_state.residual
        pre_mix = hc_state.pre_mix

        if residual.shape[0] == 0:
            # An ADP peer still participates in each MoE collective when other
            # ranks have Decoder queries. Local attention and mHC have no rows.
            if self.mapping.enable_attention_dp and any(attn_metadata.all_rank_num_tokens or ()):
                self.forward_MoE(
                    hidden_states=residual[:, 0, :],
                    attn_metadata=attn_metadata,
                    spec_metadata=spec_metadata,
                    input_ids=input_ids,
                    hidden_states_normalized=True,
                    **({"image_mask": image_mask} if image_mask is not None else {}),
                )
            return HCState.resolved(residual, pre_mix=pre_mix)

        # Engram and DSpark capture need the residual before the fused entry.
        deferred = hc_state.is_deferred
        needs_resolved = (self.engram is not None and engram_embeddings is not None) or (
            spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx)
        )
        if deferred and needs_resolved:
            residual = self.hc_attn.post_mapping(
                x=hc_state.x_prev,
                residual=residual,
                post_layer_mix=hc_state.post_mix,
                comb_res_mix=hc_state.comb_mix,
            )
            deferred = False

        # Engram updates both the mHC coefficient input and post-update residual.
        if self.engram is not None and engram_embeddings is not None:
            residual = self.engram(
                residual,
                engram_embeddings,
                add_residual=True,
                **({"token_mask": ~image_mask} if image_mask is not None else {}),
                precomputed_kv=precomputed_engram_kv,
            )

        capture_this_layer = spec_metadata is not None and spec_metadata.is_layer_capture(
            self.layer_idx
        )
        if capture_this_layer and spec_metadata.spec_dec_mode.is_dspark():
            # Capture the post-Engram entry state, not V4's post-MoE state.
            # DSpark averages the mult axis of the flat [N, mult * hidden] stream.
            capture_rows = getattr(attn_metadata, "csa2_replay_query_rows", None)
            capture_input = residual.reshape(residual.shape[0], -1)
            if capture_rows is None:
                spec_metadata.maybe_capture_hidden_states(self.layer_idx, capture_input, None)
            else:
                cast("DSparkSpecMetadata", spec_metadata).maybe_capture_hidden_states(
                    self.layer_idx, capture_input, None, row_indices=capture_rows
                )

        # --- attention sublayer -------------------------------------------
        # C++ fused boundary (GEMM + bigfuse with the lagged pre-mix and the input RMSNorm
        # folded in); the GEMM backend (FMA / tcgen05 TF32) is the autotuner's per-M choice.
        if deferred:
            # Previous layer's post update + this boundary in one fused pair; `residual`
            # becomes the materialized stream that this layer's post_mapping needs.
            residual, attn_pre, post_mix, comb_mix, layer_input = self.hc_attn.fused_hc_lagged(
                hc_state.x_prev,
                residual,
                hc_state.post_mix,
                hc_state.comb_mix,
                pre_mix,
                norm_weight=self.input_layernorm.weight,
                norm_eps=self.input_layernorm.variance_epsilon,
            )
        else:
            attn_pre, post_mix, comb_mix, layer_input = self.hc_attn.pre_mapping_lagged(
                residual,
                pre_mix,
                norm_weight=self.input_layernorm.weight,
                norm_eps=self.input_layernorm.variance_epsilon,
            )
        x_attn = self.self_attn(
            position_ids=position_ids,
            hidden_states=layer_input,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(enable_allreduce=not self.disable_attn_allreduce),
            **kwargs,
        )
        # Attention's post update + FFN boundary + post_attention RMSNorm in one fused pair.
        residual, ffn_pre, post_mix, comb_mix, layer_input = self.hc_ffn.fused_hc_lagged(
            x_attn,
            residual,
            post_mix,
            comb_mix,
            attn_pre,
            norm_weight=self.post_attention_layernorm.weight,
            norm_eps=self.post_attention_layernorm.variance_epsilon,
        )
        x_ffn = self.forward_MoE(
            hidden_states=layer_input,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            input_ids=input_ids,
            hidden_states_normalized=True,
            **({"image_mask": image_mask} if image_mask is not None else {}),
        )
        if self._v41_defer_post_mapping:
            # Defer hc_ffn.post_mapping into the next layer's fused entry boundary; the
            # next layer materializes it itself if it cannot absorb it (see forward entry).
            return HCState.deferred(residual, post_mix, comb_mix, x_ffn, pre_mix=ffn_pre)
        residual = self.hc_ffn.post_mapping(
            x=x_ffn,
            residual=residual,
            post_layer_mix=post_mix,
            comb_res_mix=comb_mix,
        )

        # `ffn_pre` rides to the next block, which collapses with it.
        return HCState.resolved(residual, pre_mix=ffn_pre)


# ---------------------------------------------------------------------------
# 4. No hc_head: the stream is opened and closed differently
# ---------------------------------------------------------------------------


class DeepseekV41Model(DeepseekV4Model):
    """V4's model body with V4.1's two mHC stream boundaries.

    The shared loop retains Engram lookup, event waits and PP handling. V4.1
    overrides the stream boundaries and schedules eligible Engram projections.
    """

    decoder_layer_cls = DeepseekV41DecoderLayer
    uses_hc_head = False

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args, **kwargs):
        super().__init__(model_config, *args, **kwargs)
        self._decoder_replay_observed = False
        self._decoder_replay_reuse_warning_emitted = False
        self.ced_kv_precompute = False
        self.decoder_replay_split, self.decoder_replay_window = self._resolve_replay_policy(
            model_config
        )
        self.encoder_replay_enabled = encoder_replay_enabled()
        if self.encoder_replay_enabled and (
            self.decoder_replay_split is None or not self.ced_kv_precompute
        ):
            raise ValueError("Encoder recovery requires a supported decoder replay topology")
        mapping = model_config.mapping
        # GB300 measurements cover TP4 head-sharded host tables at T=1.
        self._engram_projection_schedule = (
            {0: 1, 2: 14}
            if torch.cuda.is_available()
            and torch.cuda.get_device_capability() == (10, 3)
            and mapping.tp_size == 4
            and not mapping.has_pp()
            and mapping.cp_size == 1
            and not mapping.enable_attention_dp
            and model_config.pretrained_config.hidden_size == 5120
            else {}
        )
        bounded_replay_on_generation = model_config.extra_attrs.get(
            "bounded_replay_on_generation", False
        )
        if bounded_replay_on_generation and self.decoder_replay_split is None:
            raise ValueError(
                "bounded_replay_on_generation requires supported decoder bounded replay"
            )
        self.disagg_remote_tail_replay = bounded_replay_on_generation
        self.disagg_context_only = disagg_context_decoder_skipping_enabled(
            bounded_replay_on_generation
        )
        if self.disagg_context_only and not self.disagg_remote_tail_replay:
            raise ValueError(
                "Context decoder skipping requires a supported remote-tail replay boundary"
            )
        if self.disagg_remote_tail_replay:
            logger.info(
                "DeepSeek-V4.1 disaggregated remote-tail replay enabled: "
                f"eligible context workers stop before layer {self.decoder_replay_split} "
                f"after the prompt prefix, and generation workers replay the final "
                f"{self.decoder_replay_window} prompt tokens through the full model. "
                "Set bounded_replay_on_generation=false to disable."
            )

    def _precompute_engram_kv(
        self, layer_idx: int, embeddings: dict[int, torch.Tensor]
    ) -> tuple[int, torch.Tensor] | None:
        target = self._engram_projection_schedule.get(layer_idx)
        if target is None or target not in embeddings:
            return None
        local_embeddings = embeddings[target]
        engram = self.layers[target].engram
        if (
            local_embeddings.shape[0] != 1
            or engram.stream is None
            or not engram.multi_head_embedding.shard_heads
            or not isinstance(engram.kv_proj, EngramFp8Projection)
        ):
            return None
        return target, engram.precompute_kv(local_embeddings)

    def forward(self, attn_metadata, *args, **kwargs):
        from ..pyexecutor.ced_replay import uses_private_decoder_cache

        attn_metadata.begin_model_forward()
        spec_metadata = kwargs.get("spec_metadata")
        if isinstance(spec_metadata, DSparkSpecMetadata):
            spec_metadata.context_capture_lens = None
        if self.disagg_context_only:
            mode = getattr(attn_metadata, "csa2_remote_tail_mode", None)
            if mode not in (None, "source"):
                raise ValueError("A context-only model cannot execute a remote-tail destination")
            # Profiling/warmup batches have no request handoff metadata. They
            # must exercise the same encoder-only path as serving requests.
            attn_metadata.csa2_remote_tail_mode = "source"
        requests = kwargs.get("context_requests")
        if (
            requests
            and not self.encoder_replay_enabled
            and not uses_private_decoder_cache(self, attn_metadata.kv_cache_manager)
            and not (
                attn_metadata.mapping is not None and attn_metadata.mapping.enable_attention_dp
            )
        ):
            kwargs.pop("context_requests", None)
            requests = None
        output = super().forward(attn_metadata, *args, **kwargs)
        if isinstance(spec_metadata, DSparkSpecMetadata):
            spec_metadata.context_capture_lens = attn_metadata.csa2_decoder_capture_lens
        if requests and self.encoder_replay_enabled:
            from ..attention.backends.sparse.csa2.ced import complete_encoder_replay

            complete_encoder_replay(requests)
        return output

    def _compute_engram_hashes(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
        attn_metadata: CSA2TrtllmMetadata,
        *,
        token_mask: torch.Tensor | None = None,
        padded_num_tokens: int | None = None,
        refresh: bool = False,
    ) -> dict[int, torch.Tensor]:
        """Keep request history local and exclude graph padding from n-grams."""
        real_num_tokens = attn_metadata.num_tokens
        if padded_num_tokens is None:
            padded_num_tokens = input_ids.numel()
        if not 0 <= real_num_tokens <= input_ids.numel() <= padded_num_tokens:
            raise ValueError("Engram request tokens and padded input rows are inconsistent.")
        compute_hashes = (
            self.engram_hash_provider.refresh_captured_hashes
            if refresh
            else self.engram_hash_provider.compute_hashes
        )
        return compute_hashes(
            input_ids.view(-1)[:real_num_tokens],
            seq_lens=attn_metadata.seq_lens_cuda,
            position_ids=None if position_ids is None else position_ids[..., :real_num_tokens],
            request_ids=attn_metadata.request_ids,
            seq_lens_host=attn_metadata.seq_lens,
            max_seq_len=getattr(attn_metadata.kv_cache_manager, "max_seq_len", None),
            padded_num_tokens=padded_num_tokens,
            **({"token_mask": token_mask[:real_num_tokens]} if token_mask is not None else {}),
        )

    def _precompute_engram(
        self,
        layer_id: int,
        hash_indices: torch.Tensor,
        dtype: torch.dtype,
        attn_metadata: CSA2TrtllmMetadata,
    ) -> torch.Tensor | _AttentionDPEngramEmbeddings:
        return self.layers[layer_id].engram.precompute(
            hash_indices, dtype=dtype, all_rank_num_tokens=attn_metadata.all_rank_num_tokens
        )

    def _resolve_replay_policy(
        self, model_config: ModelConfig[PretrainedConfig]
    ) -> Tuple[Optional[int], int]:
        """``(first replayed layer, window)``, or ``(None, 0)`` if the model cannot replay.

        The final unpooled KV source can publish its full-sequence cache before
        the query pass is narrowed. On the released checkpoint this is layer 20:
        its Q, index selection, attention and MoE then run only on the tail.
        Other topologies retain the conservative split after the last source.

        Retain one SWA window, enlarged only for an embedded DSpark capture
        window. This report-style policy is approximate across stacked SWA
        layers. Global KV/index keys must be complete before queries are narrowed;
        decoder-side Engram, PP/CP and unsupported capture contracts fall back
        to full prefill. Attention DP exchanges the final planned row counts
        before the layer loop and switches every rank at the Decoder boundary.
        """
        if not decoder_bounded_replay_enabled():
            return None, 0

        def refuse(reason: str) -> Tuple[None, int]:
            self.ced_kv_precompute = False
            logger.warning(
                "DeepSeek-V4.1 decoder bounded replay was requested but "
                f"{reason}; running full decoder prefill."
            )
            return None, 0

        sparse_config = model_config.sparse_attention_config
        if sparse_config is None:
            return refuse("no CSA2 layout establishes global cache ownership")
        layout = sparse_config.to_sparse_params(
            pretrained_config=model_config.pretrained_config
        ).layout
        kv_sources = layout.kv_source_layer_ids
        if not kv_sources:
            return refuse("no kv_source_layer_ids establish a cache-independent decoder suffix")
        if any(
            not isinstance(src, int) or src < 0 or src >= self.num_hidden_layers
            for src in kv_sources
        ):
            return refuse("kv_source_layer_ids do not identify valid backbone layers")
        mapping = model_config.mapping
        if mapping.pp_size != 1 or mapping.cp_size != 1:
            return refuse("the replay boundary supports only PP=1 and CP=1")
        spec_config = getattr(model_config, "spec_config", None)
        if mapping.enable_attention_dp and model_config.extra_attrs.get(
            "bounded_replay_on_generation", False
        ):
            if spec_config is not None:
                return refuse("attention-DP remote-tail replay requires speculative decoding off")
            # An idle context rank must stop at the same layer as active ranks,
            # otherwise their MoE collectives diverge. Pruned context workers
            # force source routing even for warmup and dummy batches; generation
            # workers always execute every layer, including remote-tail batches.
            role = os.environ.get("TRTLLM_DISAGG_ROLE")
            if role != "generation" and not disagg_context_decoder_skipping_enabled(
                model_config.extra_attrs.get("bounded_replay_on_generation", False)
            ):
                return refuse(
                    "attention-DP remote-tail replay requires an explicit generation role "
                    "or a context role with decoder weight skipping enabled"
                )
        swa_window = layout.window_size
        if not isinstance(swa_window, int) or isinstance(swa_window, bool) or swa_window <= 0:
            return refuse("the effective SWA window is not a known positive integer")
        last_source = int(max(kv_sources))
        split = last_source + 1
        if last_source < self.num_hidden_layers:
            attention = self.layers[last_source].self_attn
            if (
                attention.compressor is not None
                and attention.layer.mode == CSA2Mode.FULL
                and hasattr(attention, "index_wk")
                and attention.layer.compress_ratio == 1
            ):
                split = last_source
                self.ced_kv_precompute = True
        if split >= self.num_hidden_layers:
            return refuse("there are no decoder layers after the last full KV producer")
        late_engram = [idx for idx in getattr(self, "engram_layer_ids", ()) or () if idx >= split]
        if late_engram:
            return refuse(f"Engram layers {late_engram} need unsupported replay re-indexing")
        # Global KV and index keys must not depend on narrowed query states.
        # Check the constructed modules, not just the source-layer declaration.
        full_sources = {
            src for src in kv_sources if src < split or src == split and self.ced_kv_precompute
        }
        for idx in range(split, self.num_hidden_layers):
            attention = self.layers[idx].self_attn
            precomputed_source = self.ced_kv_precompute and idx == split
            if attention.compressor is not None and not precomputed_source:
                return refuse(f"layer {idx} still produces global KV from replayed states")
            plan = attention.layer
            if plan.compress_ratio != 0 and plan.kv_source not in full_sources:
                return refuse(f"layer {idx} has no full-encoder global KV source")
            if hasattr(attention, "index_wq_b"):
                if plan.kv_source not in full_sources:
                    return refuse(f"layer {idx} has no full-encoder index-key source")
                if hasattr(attention, "index_wk") and not precomputed_source:
                    return refuse(f"layer {idx} still projects index keys from replayed states")

        window = swa_window
        if spec_config is not None:
            if (
                not spec_config.spec_dec_mode.is_dspark()
                or not spec_config.draft_is_embedded_in_target
            ):
                return refuse("this speculative worker has no proven bounded capture contract")
            draft_window = getattr(model_config.pretrained_config, "sliding_window", None)
            captures = spec_config.target_layer_ids
            if (
                not isinstance(draft_window, int)
                or isinstance(draft_window, bool)
                or draft_window <= 0
                or not captures
                or any(
                    not isinstance(c, int) or not 0 <= c < self.num_hidden_layers for c in captures
                )
            ):
                return refuse(
                    "the embedded DSpark window or validated entry-capture layers are unknown"
                )
            window = max(window, draft_window)
        logger.info(
            f"DeepSeek-V4.1 decoder bounded replay enabled: layers {split}-"
            f"{self.num_hidden_layers - 1} replay over a {window}-token window "
            f"(approximate, kv sources {sorted(int(s) for s in kv_sources)}). "
            "Set TRTLLM_V41_DECODER_BOUNDED_REPLAY=0 for full decoder prefill."
        )
        return split, window

    def prepare_adp_inputs(
        self,
        attn_metadata: CSA2TrtllmMetadata,
        *,
        all_token_states_required: bool,
        requests: list["LlmRequest"] | None = None,
    ) -> None:
        """Fix local execution rows before exchanging Encoder/Decoder counts.

        Direct ADP callers use this before the normal input count exchange too;
        full-state requirements must not change after counts are exchanged.
        """
        attn_metadata.decoder_replay_plan = self._make_bounded_replay_plan(
            attn_metadata, all_token_states_required, requests
        )

    def _plan_bounded_replay(
        self,
        attn_metadata: CSA2TrtllmMetadata,
        all_token_states_required: bool,
        requests: list["LlmRequest"] | None = None,
    ) -> tuple[int | None, DecoderReplayPlan | None]:
        from ..pyexecutor.ced_replay import requires_full_decoder_prefill
        from ..utils import get_per_request_prefill_cuda_graph_flag

        if attn_metadata.mapping is not None and attn_metadata.mapping.enable_attention_dp:
            # Padding selection follows the count exchange and is group-uniform.
            # Captured forwards retain the full, padded input on every rank.
            if attn_metadata.is_cuda_graph or get_per_request_prefill_cuda_graph_flag():
                return None, None
            plan = getattr(attn_metadata, "decoder_replay_plan", None)
            if plan is not None:
                if plan.replay_all_rank_num_tokens is None:
                    raise ValueError("Exchange ADP token counts after prepare_adp_inputs().")
                if (
                    plan.saved_all_rank_num_tokens != attn_metadata.all_rank_num_tokens
                    or plan.saved_all_rank_num_tokens[attn_metadata.mapping.tp_rank]
                    != (attn_metadata.padded_num_tokens or attn_metadata.num_tokens)
                ):
                    raise ValueError("ADP input rows changed after exchanging token counts.")
                if (
                    plan.replays_local_tokens
                    and all_token_states_required
                    and not (requests and any(requires_full_decoder_prefill(r) for r in requests))
                ):
                    raise ValueError(
                        "Full token states must be requested in prepare_adp_inputs(), "
                        "before exchanging ADP token counts."
                    )
        else:
            plan = self._make_bounded_replay_plan(
                attn_metadata, all_token_states_required, requests
            )
        if plan is None:
            return None, None
        if plan.replays_local_tokens and not self._decoder_replay_observed:
            self._decoder_replay_observed = True
            logger.info(
                f"DeepSeek-V4.1 decoder bounded replay engaged: "
                f"{plan.num_replay_tokens} of {plan.num_encoder_tokens} rows re-run "
                f"from layer {self.decoder_replay_split}."
            )
        return self.decoder_replay_split, plan

    def _make_bounded_replay_plan(
        self,
        attn_metadata: CSA2TrtllmMetadata,
        all_token_states_required: bool,
        requests: list["LlmRequest"] | None = None,
    ) -> DecoderReplayPlan | None:
        """Replay the decoder half over the last ``n_win`` rows, when that is legal.

        ``all_token_states_required`` is the caller's answer to "does anything
        downstream read a hidden state that is not the last of its sequence" --
        context logits, in practice. The replayed pass produces no such rows, so a
        caller that needs them gets the ordinary single pass.
        """
        from ..pyexecutor.ced_replay import (
            requires_full_decoder_prefill,
            uses_private_decoder_cache,
        )

        if (
            self.decoder_replay_split is None
            or (
                attn_metadata.mapping is not None
                and attn_metadata.mapping.enable_attention_dp
                and self.disagg_remote_tail_replay
            )
            or getattr(attn_metadata, "csa2_remote_tail_mode", None) is not None
            or getattr(attn_metadata.kv_cache_manager, "enable_swa_scratch_reuse", False)
        ):
            return None
        allow_replay = not all_token_states_required or bool(
            requests and any(requires_full_decoder_prefill(r) for r in requests)
        )
        private_decoder = uses_private_decoder_cache(self, attn_metadata.kv_cache_manager)
        if not private_decoder and getattr(
            attn_metadata.kv_cache_manager, "enable_block_reuse", False
        ):
            if not self._decoder_replay_reuse_warning_emitted:
                logger.warning(
                    "DeepSeek-V4.1 decoder bounded replay is disabled because KV block reuse "
                    "is enabled. Replayed decoder SWA is approximate and must not be published "
                    "for prefix caching. Set kv_cache_config.enable_block_reuse=False to use "
                    "bounded replay; running full decoder prefill."
                )
                self._decoder_replay_reuse_warning_emitted = True
            allow_replay = False
        return plan_decoder_replay(
            attn_metadata,
            self.decoder_replay_window,
            requests,
            private_decoder=private_decoder,
            allow_replay=allow_replay,
        )

    def _remote_tail_boundary(self, attn_metadata) -> Optional[int]:
        if not self.disagg_remote_tail_replay:
            return None
        mode = getattr(attn_metadata, "csa2_remote_tail_mode", None)
        return self.decoder_replay_split if mode in ("source", "destination") else None

    def _enter_remote_tail_boundary(
        self, attn_metadata, hc_state: HCState
    ) -> tuple[torch.Tensor | None, HCState]:
        """Materialize boundary GLOBAL and either finish the source or enter replay."""
        hc_state = self._resolve_hc_state(hc_state)
        collapsed = None
        if self.ced_kv_precompute:
            collapsed = prepare_ced_global_kv(
                self.layers[self.decoder_replay_split], hc_state, attn_metadata
            )
        mode = attn_metadata.csa2_remote_tail_mode
        if mode == "source":
            # The boundary owner already collapsed this resolved state before
            # its input norm. Return that same pre-norm value to the output
            # epilogue without a second collapse or any cross-forward storage.
            if collapsed is None:
                collapsed = self._finalize_hc_state(hc_state)
            return collapsed, hc_state
        if mode != "destination":
            raise ValueError(f"unknown CSA2 remote-tail mode {mode!r}")
        enter_remote_tail_decoder(attn_metadata, attn_metadata.csa2_remote_tail_starts)
        return None, hc_state

    def _enter_bounded_replay(
        self,
        plan: DecoderReplayPlan,
        attn_metadata: CSA2TrtllmMetadata,
        position_ids: torch.Tensor | None,
        input_ids: torch.Tensor | None,
        hc_state: HCState,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, HCState]:
        """Resolve deferred mHC state, then compact token rows and rebuild metadata."""
        from ..pyexecutor.ced_replay import uses_private_decoder_cache

        if not plan.updates_local_metadata:
            enter_decoder_replay(attn_metadata, plan)
            return position_ids, input_ids, hc_state
        hc_state = self._resolve_hc_state(hc_state)
        if self.ced_kv_precompute:
            prepare_ced_global_kv(self.layers[self.decoder_replay_split], hc_state, attn_metadata)
        enter_decoder_replay(
            attn_metadata,
            plan,
            self.decoder_replay_split
            if uses_private_decoder_cache(self, attn_metadata.kv_cache_manager)
            else None,
        )
        attn_metadata.csa2_replay_query_rows = plan.rows
        return (
            gather_replayed_rows(position_ids, plan),
            gather_replayed_rows(input_ids, plan),
            HCState.resolved(
                gather_replayed_rows(hc_state.residual, plan),
                pre_mix=gather_replayed_rows(hc_state.pre_mix, plan),
            ),
        )

    def _exit_bounded_replay(self, plan, attn_metadata, hidden_states):
        """Restore the output layout and metadata after successful Decoder execution."""
        hidden_states = scatter_replayed_rows(hidden_states, plan)
        exit_decoder_replay(attn_metadata, plan)
        attn_metadata.csa2_decoder_capture_lens = tuple(
            plan.replay_seq_lens[: attn_metadata.num_contexts].tolist()
        )
        attn_metadata.csa2_precomputed_kv_layers.clear()
        return hidden_states

    def _resolve_hc_state(self, hc_state: HCState) -> HCState:
        """Materialize a deferred post update using any layer's parameter-free mapping."""
        if not hc_state.is_deferred:
            return hc_state
        residual = self.layers[0].hc_attn.post_mapping(
            x=hc_state.x_prev,
            residual=hc_state.residual,
            post_layer_mix=hc_state.post_mix,
            comb_res_mix=hc_state.comb_mix,
        )
        return HCState.resolved(residual, pre_mix=hc_state.pre_mix)

    def _init_hc_state(self, hidden_states: torch.Tensor) -> HCState:
        """Seed the lagged ``pre`` with the reference's one-hot.

        ``make_identity_pre_mix`` (model.py:1159) is ``zeros(..., hc_mult)`` with
        ``[..., 0] = 1.0`` -- a one-hot on stream 0, *not* ones and not
        ``1 / hc_mult``. Since layer 0 receives four identical copies of the
        embedding (``h.unsqueeze(2).repeat(...)``), the resulting collapse returns
        the embedding unscaled; a ones seed would hand layer 0 four times the
        embedding, and a uniform ``1/mult`` seed would be right only by accident
        of the streams being identical -- and would diverge the moment engram
        fires on layer 0.
        """
        pre_mix = hidden_states.new_zeros(
            (hidden_states.shape[0], self.hc_mult, 1), dtype=torch.float32
        )
        pre_mix[:, 0, :] = 1.0
        return HCState.resolved(hidden_states, pre_mix=pre_mix)

    def _finalize_hc_state(self, hc_state: HCState) -> torch.Tensor:
        """Close the stream with the last layer's trailing ``pre``.

        The reference runs ``h = layer.hc_pre(h, pre_mix)`` after the loop
        (model.py:1268) using the last block's ``ffn_pre`` -- one more collapse
        with the same coefficients every other sublayer boundary uses, rather
        than V4's separately-parameterized ``hc_head``. Returns plain
        ``[N, hidden]``; ``DeepseekV4LogitsProcessor`` then applies ``norm`` and
        the head, matching ``head(norm(h))``.
        """
        hc_state = self._resolve_hc_state(hc_state)
        return mHC.collapse(hc_state.residual, hc_state.pre_mix)

    # The descriptor fields a *constructed* layer observably owns. Everything
    # else on ``DeepseekV41LayerDescriptor`` is either bookkeeping only the
    # config knows (which layer sources which cache, ``kind``) or a statement
    # about weights that do not exist until load time (``owns_indexer_wk``,
    # ``compressor_wkv_dtype`` -- an index source that is not a kv source still
    # builds an ``Indexer`` whose ``wk`` gets zero-filled, so the module cannot
    # answer who owns the real ``wk``). Those are emitted but not cross-checked.
    @staticmethod
    def _observe_layer(layer, layer_idx: int) -> Dict[str, Any]:
        """Read the topology a constructed decoder layer actually got.

        CSA2 construction stores the lowered layout, the resolved layer plan,
        and the owned projections. Reading those objects needs no request KV
        cache or runtime metadata.

        The point of going through the *modules* rather than re-reading the
        config is that this is the only thing that fails when the sparse config
        never reaches the attention layer. A config-only dump would agree with
        the reference-derived plan even with ``sparse_attention_config=None``,
        because both sides would be reading the same ``config.json``.
        """
        attn = layer.self_attn
        sparse_params = getattr(attn, "sparse_params", None)
        if sparse_params is None:
            raise ValueError(
                f"layer {layer_idx}: CSA2 sparse parameters are absent; "
                "ModelConfig.from_pretrained must derive CSA2SparseAttentionConfig. The "
                "topology in the checkpoint config is then invisible to the "
                "attention stack."
            )
        layout = sparse_params.layout
        plan = attn.layer
        rope = attn.pos_embd_params
        return {
            "compress_ratio": plan.compress_ratio,
            "has_long_range": plan.compress_ratio != 0,
            "pools_kv": plan.compress_ratio == 2,
            "pool_factor": max(1, plan.compress_ratio),
            "rope_theta": float(rope.rope.theta),
            "yarn_enabled": rope.type is PositionEmbeddingType.yarn,
            "window_size": layout.window_size,
            "is_kv_source": plan.mode == CSA2Mode.FULL,
            "is_index_source": hasattr(attn, "index_wq_b"),
            "owns_compressor": attn.compressor is not None,
            "has_engram": layer.engram is not None,
        }

    def describe_layers(self) -> Dict[str, Any]:
        """Per-layer topology of *this* instance, for ``gates/layer_plan.py --diff``.

        One row per ``compress_ratios`` entry -- 43 for the released checkpoint,
        i.e. 40 decoder layers plus 3 MTP layers -- because that list is what
        both the reference and every ratio-keyed buffer in the backend are sized
        by. Each row is the full descriptor, plus:

        * ``constructed``: whether this target body built the layer. The three
          draft rows are ``False`` here: embedded DSpark constructs and audits
          its stages in a separate draft model, not in this target layer list.
          Their fields in this target-only report are config-derived.

        Every constructed layer is cross-checked against its descriptor row and
        a disagreement raises, so a passing dump means the modules and the
        reference-derived plan agree -- not merely that two readers of the same
        ``config.json`` agree.
        """
        pretrained_config = self.model_config.pretrained_config
        descriptors = pretrained_config.layer_descriptors
        # The inherited V4 path can append MTP layers. V4.1's embedded DSpark
        # owns a separate draft model; this report describes the target only.
        built = {
            idx: layer
            for idx, layer in enumerate(self.layers[: self.num_hidden_layers])
            if not self.disagg_context_only or idx < self.decoder_replay_split
        }

        rows: List[Dict[str, Any]] = []
        verified: set = set()
        problems: List[str] = []
        for descriptor in descriptors:
            row = descriptor.to_dump_dict()
            layer = built.get(descriptor.layer_idx)
            row["constructed"] = layer is not None
            if layer is not None:
                observed = self._observe_layer(layer, descriptor.layer_idx)
                for field, value in observed.items():
                    verified.add(field)
                    if row[field] != value:
                        problems.append(
                            f"layer {descriptor.layer_idx}: {field} "
                            f"descriptor={row[field]!r} module={value!r}"
                        )
            rows.append(row)

        if problems:
            raise ValueError(
                "constructed layers disagree with their config descriptors:\n  "
                + "\n  ".join(problems)
            )

        not_constructed = [d.layer_idx for d in descriptors if d.layer_idx not in built]
        if not_constructed:
            logger.warning(
                f"DeepSeek-V4.1 describe_layers: layers {not_constructed} are "
                "described from the config only -- this instance did not build "
                "them (draft stages are owned separately), so their fields "
                "in this target-only report are unverified."
            )
        return {
            "model": type(self).__name__,
            "variant": "v41",
            "num_hidden_layers": self.num_hidden_layers,
            "num_ratio_entries": len(descriptors),
            "num_constructed": len(built),
            "not_constructed": not_constructed,
            "verified_fields": sorted(verified),
            "layers": rows,
        }


# ---------------------------------------------------------------------------
# 6. Checkpoint layout
# ---------------------------------------------------------------------------

# Engram module renames. The checkpoint names the table `embed` and the fused
# projection `wkv`; TRT-LLM's Engram calls them `multi_head_embedding` and
# `kv_proj`. `q_weight`/`k_weight` are the two `[hc_mult, hidden]` norm weights,
# which the module holds as direct parameters.
_ENGRAM_SUBKEY_RENAME = {
    "embed": "multi_head_embedding",
    "wkv": "kv_proj",
    "q_weight": "query_norm_weight",
    "k_weight": "key_norm_weight",
}

# These weights belong to the multimodal wrapper.
_V41_DROP_PREFIXES = ("vision.", "aligner.")
_V41_DROP_KEYS = ("image_start", "image_end", "image_newline")

# Sub-trees whose `.weight`/`.scale` pair must NOT be dequantized:
#   - routed experts are packed MXFP4 (int8 pairs + a ue8m0 scale per 32 lanes),
#     consumed quantized by the MoE backend;
#   - the engram table is 98 GiB of fp8 and stays fp8, row-sharded
#     (`ShardedFp8MultiHeadEmbedding` dequantizes per lookup).
_V41_NO_DEQUANT = (".ffn.experts.", ".engram.embed.")

_V41_FP8_BLOCK_SIZE = 32

# CSA2 Linear loaders expand block-32 scales without changing FP8 weight bytes.
_V41_NATIVE_FP8_STEM_TAILS = (
    ".attn.wq_a",
    ".attn.wq_b",
    ".attn.indexer.wq_b",
    ".attn.wkv",
    ".attn.wo_b",
)

# The fused grouped output and shared-expert GEMMs still require block-128 scales.
_V41_REQUANT_STEM_TAILS = (
    ".attn.wo_a",
    ".ffn.shared_experts.w1",
    ".ffn.shared_experts.w2",
    ".ffn.shared_experts.w3",
)

_E4M3_MAX = 448.0
# E4M3 subnormal error bound relative to a tile scaled to the format ceiling.
_E4M3_SUBNORMAL_FLOOR_EXP = -9
_V41_REQUANT_BLOCK = 128

# The load census maps flat checkpoint keys (hc_attn_fn) to parameters (hc_attn.fn).
_V41_FLAT_HC_STEMS = ("hc_attn", "hc_ffn", "hc_head")
_V41_FLAT_HC_ATTRS = ("fn", "base", "scale")

# Scale keys whose consumers register a sibling parameter rather than a submodule.
_V41_FLAT_SCALE_PARAMS = {
    "o_a_proj.weight_scale_inv": "o_a_proj_scale",
    "engram.kv_proj.scale": "engram.kv_proj.weight_scale",
}

# Resolve checkpoint components loaded into fused model modules.
_V41_FUSED_MODULES = {
    "gate_proj": "gate_up_proj",
    "up_proj": "gate_up_proj",
    "q_a_proj": "kv_a_proj_with_mqa",
}

# CSA2 owns attention sinks directly; coverage checks their registered leaf names.


def _v41_ignore_reason(
    key: str,
    *,
    keep_mtp_layers: int = 0,
    load_vision_bias: bool = False,
    context_only_split: int | None = None,
    context_only_precompute: bool = False,
) -> Optional[Tuple[str, str]]:
    """``(name pattern, why)`` if this raw checkpoint key is deliberately not loaded.

    One function decides, so the remap and the census cannot disagree about what
    was dropped -- a census that recomputes the drop rules separately from the
    code that applies them audits nothing.

    ``keep_mtp_layers`` is how many MTP layers the *model* built, not how many the
    checkpoint ships: V4's remap routes ``mtp.0.`` to ``model.layers.<n>.`` and has
    no route for the rest, so anything at or past the built count has no consumer
    and is counted as ignored rather than forwarded to be silently dropped.
    """
    if context_only_split is not None:
        reason = (
            "decoder layers and output head",
            "remote-tail context worker; decoder weights are owned by generation workers",
        )
        if key.startswith(("head.", "norm.", "mtp.", "lm_head.", "model.norm.")):
            return reason
        layer_key = key.removeprefix("model.")
        if layer_key.startswith("layers."):
            _, index, rest = layer_key.split(".", 2)
            if int(index) >= context_only_split:
                boundary_key = (
                    context_only_precompute
                    and int(index) == context_only_split
                    and (
                        rest in ("attn_norm.weight", "input_layernorm.weight")
                        or rest.startswith(
                            (
                                "attn.compressor.",
                                "attn.indexer.wk.",
                                "attn.indexer.k_norm.",
                                "self_attn.compressor.",
                                "self_attn.indexer.wk.",
                                "self_attn.indexer.k_norm.",
                            )
                        )
                    )
                )
                if not boundary_key:
                    return reason
    if key.startswith(_V41_DROP_PREFIXES):
        return (
            "vision.* | aligner.*",
            "vision tower and projector; owned by the multimodal wrapper",
        )
    if key in _V41_DROP_KEYS:
        return (
            "image_start | image_end | image_newline",
            "learned image separators; owned by the multimodal wrapper",
        )
    if key.endswith("ffn.gate.bias_vl") and not load_vision_bias:
        # Text-only gates do not allocate the image-selection bias.
        return (
            "layers.<i>.ffn.gate.bias_vl",
            "vision routing disabled; every token uses the text bias",
        )
    if key.startswith("mtp."):
        parts = key.split(".", 2)
        index = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else keep_mtp_layers
        if index >= keep_mtp_layers:
            return (
                "mtp.*" if keep_mtp_layers == 0 else f"mtp.<i>.* for i >= {keep_mtp_layers}",
                "not owned by the target layers; embedded DSpark loads its three "
                "stages in a separate, draft-local checkpoint pass",
            )
    return None


def _v41_tensor_shape(value: Any) -> Sequence[int]:
    """Shape of a loaded tensor or of a lazy safetensors slice."""
    return value.shape if hasattr(value, "shape") else value.get_shape()


def _v41_tensor_dtype(value: Any) -> str:
    """Dtype spelling of a loaded tensor or of a lazy safetensors slice."""
    return str(value.dtype) if hasattr(value, "dtype") else str(value.get_dtype())


def _v41_structural_key(key: str) -> str:
    """A forwarded key under the name the *model* would know it by.

    Three rewrites, all of them things V4's walk does rather than things V4.1
    changed: flat mHC names become structured ones, a block scale spelled as a
    sub-module of a bare ``nn.Parameter`` becomes its flat sibling, and a fused
    sub-module's name becomes the fused module's. Everything else is already a
    model name.

    Relocated leaf parameters (``_V41_RELOCATED_PARAMS``) are deliberately *not*
    rewritten here: their destination is not reachable by name at all, so
    ``_v41_load_coverage`` resolves them as attributes instead.
    """
    if ".self_attn." in key:
        prefix, tail = key.split(".self_attn.", 1)
        exact = {
            "q_a_layernorm.weight": "q_norm.weight",
            "kv_a_layernorm.weight": "kv_norm.weight",
            "indexer.k_norm.weight": "index_k_norm.weight",
        }
        heads = {
            "q_a_proj": "wq_a",
            "q_b_proj": "wq_b",
            "kv_a_proj_with_mqa": "wkv",
            "indexer.wq_b": "index_wq_b",
            "indexer.wk": "index_wk",
            "indexer.weights_proj": "index_weights_proj",
        }
        if tail in exact:
            return prefix + ".self_attn." + exact[tail]
        for old, new in heads.items():
            if tail.startswith(old + "."):
                tail = new + tail[len(old) :]
                break
        if tail.endswith(".weight_scale_inv") and not tail.startswith("o_a_proj."):
            tail = tail.removesuffix("weight_scale_inv") + "weight_scale"
        key = prefix + ".self_attn." + tail
    for attr in _V41_FLAT_HC_ATTRS:
        suffix = f"_{attr}"
        if key.endswith(suffix):
            head = key[: -len(suffix)]
            if head.rsplit(".", 1)[-1] in _V41_FLAT_HC_STEMS:
                return f"{head}.{attr}"
    for suffix, flat in _V41_FLAT_SCALE_PARAMS.items():
        if key.endswith(f".{suffix}"):
            return f"{key[: -len(suffix)]}{flat}"
    parts = key.split(".")
    if len(parts) > 1 and parts[-2] in _V41_FUSED_MODULES:
        parts[-2] = _V41_FUSED_MODULES[parts[-2]]
        return ".".join(parts)
    return key


def _v41_structural_targets(key: str) -> Tuple[str, ...]:
    """Exact leaf destinations of one checkpoint key."""
    return (_v41_structural_key(key),)


@dataclass
class DeepseekV41LoadCensus:
    """Account for every checkpoint tensor: consumed + ignored must equal total."""

    total: int = 0
    folded: int = 0
    forwarded: int = 0
    # Both halves of each requantized dense stem also count as forwarded tensors.
    requantized: int = 0
    checked: Dict[str, Tuple[int, DeepseekV41QuantLayout]] = field(default_factory=dict)
    ignored: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)

    def ignore(self, key: str, reason: Tuple[str, str]) -> None:
        self.ignored.setdefault(reason, []).append(key)

    @property
    def ignored_count(self) -> int:
        return sum(len(names) for names in self.ignored.values())

    def check(self, role: str, layout: DeepseekV41QuantLayout) -> None:
        """Record that one (weight, scale) pair passed ``assert_weight_layout``."""
        seen, _ = self.checked.get(role, (0, layout))
        self.checked[role] = (seen + 1, layout)

    @property
    def checked_pairs(self) -> int:
        return sum(count for count, _ in self.checked.values())

    @property
    def consumed(self) -> int:
        """Folded into a dequantized weight, or forwarded under a model name."""
        return self.folded + self.forwarded

    def render(self, examples: int = 2) -> str:
        closes = self.consumed + self.ignored_count == self.total
        lines = [
            "DeepSeek-V4.1 checkpoint census:",
            f"  raw checkpoint tensors                    {self.total:>7}",
            f"  consumed                                  {self.consumed:>7}",
            f"    .scale folded into its weight           {self.folded:>7}",
            f"    forwarded under a model name            {self.forwarded:>7}",
            f"  deliberately ignored                      {self.ignored_count:>7}",
        ]
        lines.append(f"  dense FP8 stems requantized to 128x128      {self.requantized:>7}")
        for (pattern, reason), names in sorted(self.ignored.items()):
            lines.append(f"    {len(names):>7}  {pattern}")
            lines.append(f"              reason: {reason}")
            lines.append(f"              e.g.    {', '.join(sorted(names)[:examples])}")
        lines.append(
            f"  consumed + ignored == {self.total:<7}             {'OK' if closes else 'MISMATCH'}"
        )
        lines.append(f"  quantized (weight, scale) pairs checked    {self.checked_pairs:>7}")
        # Print the block each role was *held to*, because the number that decides
        # whether the experts were read as MXFP4 or as NVFP4 is the element extent,
        # and on disk it is invisible: MXFP4's 32-element block packs two nibbles
        # per byte and so derives 16 -- exactly NVFP4's block. A census that only
        # says "46412 pairs checked" cannot distinguish the two.
        for role, (count, layout) in sorted(self.checked.items()):
            lines.append(
                f"    {count:>7}  {role}: {layout.weight_dtype} in "
                f"{layout.element_block} element blocks, on disk "
                f"{layout.storage_dtype} {layout.stored_block}"
            )
        return "\n".join(lines)

    def verify(self) -> None:
        if self.consumed + self.ignored_count != self.total:
            raise ValueError(
                "DeepSeek-V4.1 load census does not close: "
                f"{self.consumed} consumed + {self.ignored_count} ignored != "
                f"{self.total} checkpoint tensors. Some tensor left the load path "
                "without a recorded reason.\n" + self.render()
            )


class _V41ForwardedWeights(dict):
    """The remapped weights, with their key namespace pinned at construction.

    The V4.1 subtree-loading pass hands this dict to
    ``ConsumableWeightsDict.take_ownership``, which wraps it and then *deletes*
    from it as modules consume their weights -- so by the time the walk is over
    there is nothing left to audit. Snapshotting the keys here, and recording the
    defaults the loader synthesizes afterwards, is what lets the audit run after
    the walk instead of having to instrument it.
    """

    def __init__(self, mapping: Dict[str, Any], census: DeepseekV41LoadCensus):
        super().__init__(mapping)
        self.census = census
        self.forwarded_keys = frozenset(self)
        self.synthesized_keys: set = set()

    def _record(self, key: str) -> None:
        if key not in self.forwarded_keys:
            self.synthesized_keys.add(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self._record(key)
        super().__setitem__(key, value)

    def update(self, *args, **kwargs) -> None:
        other = dict(*args, **kwargs)
        for key in other:
            self._record(key)
        super().update(other)

    @property
    def all_keys(self) -> set:
        return set(self.forwarded_keys) | self.synthesized_keys


def _v41_load_coverage(
    model: nn.Module, keys, *, skip_modules: Sequence[str] = ()
) -> Tuple[List[str], List[str]]:
    """``(keys no module would load, parameters no key would fill)``.

    Both directions are decided by name, against the modules that were actually
    built. A module exposing ``load_weights`` fuses several checkpoint tensors
    into its own parameters (``Linear`` stacking a TP shard, ``ConfigurableMoE``
    stacking 384 experts into ``w3_w1_weight``), so its subtree is matched by
    prefix; everything else has to name a parameter or buffer exactly. The root
    module is excluded -- it also has ``load_weights``, and accepting its prefix
    would accept every key and audit nothing.

    CSA2 owns its sink directly; the attention container is excluded from
    prefix acceptance so each forwarded attention key must reach a real leaf.
    """

    def included(name: str) -> bool:
        return not any(name == prefix or name.startswith(prefix + ".") for prefix in skip_modules)

    modules = {name: module for name, module in model.named_modules() if included(name)}
    consumers = {
        name
        for name, module in modules.items()
        if name and hasattr(module, "load_weights") and not isinstance(module, CSA2Attention)
    }
    parameters = {name: param for name, param in model.named_parameters() if included(name)}
    targets = set(parameters) | {name for name, _ in model.named_buffers() if included(name)}

    def enclosing(name: str) -> List[str]:
        parts = name.split(".")
        return [
            prefix
            for prefix in (".".join(parts[:i]) for i in range(len(parts) - 1, 0, -1))
            if prefix in consumers
        ]

    unexpected: List[str] = []
    fed: set = set()
    for key in keys:
        missing_target = False
        for structural in _v41_structural_targets(key):
            owners = enclosing(structural)
            fed.update(owners)
            missing_target |= structural not in targets and not owners
        if missing_target:
            unexpected.append(key)

    structural_keys = {target for key in keys for target in _v41_structural_targets(key)}
    unfed = [
        name
        for name in parameters
        if name not in structural_keys and not any(owner in fed for owner in enclosing(name))
    ]
    return sorted(unexpected), sorted(unfed)


def _remap_deepseek_v41_checkpoint_keys(
    weights: Dict,
    num_hidden_layers: int,
    kv_lora_rank: int = 448,
    *,
    keep_mtp_layers: int = 0,
    load_vision_bias: bool = False,
    quantization_config: Optional[Dict[str, Any]] = None,
    context_only_split: int | None = None,
    context_only_precompute: bool = False,
) -> _V41ForwardedWeights:
    """V4.1 checkpoint keys -> model parameter keys, with a census of both.

    A pre-pass over the raw keys followed by V4's remap, rather than a fork of
    it: the attention / FFN / MTP / compressor renames are identical, so the only
    V4.1-specific work is what the pre-pass does.

    1. **Preserve native FP8 pairs.** CSA2 Linear projections keep their
       checkpoint weights and block-32 scales. Only grouped ``wo_a`` and shared
       experts in ``_V41_REQUANT_STEM_TAILS`` need block-128 requantization for
       their current kernels. Routed experts and Engram retain their own policies.

    2. **Rename the engram sub-modules** (``embed``/``wkv``/``q_weight``/``k_weight``).
       V4's ``_rename_layer_subkey`` falls through with ``return rest`` for
       anything it does not recognize, so ``layers.N.engram.*`` reaches
       ``model.layers.N.engram.*`` unchanged once the leaf names are right --
       including ``.scale``, which the sharded table wants under that exact name.

    3. **Separate checkpoint ownership**: the text target excludes the vision
       tower and draft-only MTP heads. Embedded DSpark loads all three stages
       through its own remapper and coverage audit. What is excluded here is
       decided by ``_v41_ignore_reason`` alone, so every dropped tensor carries a
       name pattern and a reason into the returned census.

    4. **Check the quantized layouts while the scales are still here.** Every
       ``<stem>.weight`` with a ``<stem>.scale`` sibling goes through
       ``assert_weight_layout`` before anything is dequantized or dropped. This is
       the only point in the load where both halves of a quantized tensor are in
       hand, and it raises rather than warns: the routed experts are MXFP4 with an
       element block of 32, one scale per 16 *stored* int8 lanes, and reading that
       as NVFP4's block of 16 yields plausible weights and fluent, wrong text.

    Returns a ``_V41ForwardedWeights`` -- a plain dict of model-named tensors that
    also carries the census and remembers its own key namespace, so the loader can
    audit coverage after the module walk has drained it.
    """
    census = DeepseekV41LoadCensus(total=len(weights))

    # One pass to classify every raw key and to check every quantized layout.
    # Classification comes first so that an ignored subtree's `.scale` is never
    # checked against a role expectation that was never meant to cover it.
    dequant_stems = set()
    requant_stems = set()
    for key in weights:
        reason = _v41_ignore_reason(
            key,
            keep_mtp_layers=keep_mtp_layers,
            load_vision_bias=load_vision_bias,
            context_only_split=context_only_split,
            context_only_precompute=context_only_precompute,
        )
        if reason is not None:
            census.ignore(key, reason)
            continue
        if not key.endswith(".scale"):
            continue
        stem = key[: -len(".scale")]
        weight_key = f"{stem}.weight"
        if weight_key not in weights:
            continue
        layout = assert_weight_layout(
            weight_key,
            _v41_tensor_shape(weights[weight_key]),
            _v41_tensor_shape(weights[key]),
            storage_dtype=_v41_tensor_dtype(weights[weight_key]),
            quantization_config=quantization_config,
        )
        census.check(quant_role_for_weight_key(weight_key), layout)
        if any(marker in f".{key}" for marker in _V41_NO_DEQUANT):
            continue
        if stem.endswith((".engram.wkv", *_V41_NATIVE_FP8_STEM_TAILS)):
            continue
        if stem.endswith(_V41_REQUANT_STEM_TAILS):
            requant_stems.add(stem)
        else:
            dequant_stems.add(stem)

    # Convert each weight/scale pair once, releasing it after both keys are emitted.
    requant_pairs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _requant_half(stem: str, want_scale: bool) -> torch.Tensor:
        pair = requant_pairs.pop(stem, None)
        if pair is None:
            pair = _requantize_dense_stem(weights[f"{stem}.weight"], weights[f"{stem}.scale"])
            requant_pairs[stem] = pair
            census.requantized += 1
        return pair[1] if want_scale else pair[0]

    pre: Dict[str, torch.Tensor] = {}
    for key in weights:
        if (
            _v41_ignore_reason(
                key,
                keep_mtp_layers=keep_mtp_layers,
                load_vision_bias=load_vision_bias,
                context_only_split=context_only_split,
                context_only_precompute=context_only_precompute,
            )
            is not None
        ):
            continue

        value = weights[key]
        stem = key.rsplit(".", 1)[0]
        if stem in dequant_stems:
            if key.endswith(".scale"):
                census.folded += 1
                continue
            value = _dequantize_block32(value, weights[f"{stem}.scale"])
        elif stem in requant_stems:
            # Both halves are replaced, so each raw key still maps to exactly one
            # `pre` entry and nothing is folded. The `.scale` is emitted under its
            # checkpoint name and V4's remap renames it to `weight_scale_inv`
            # (`_rename_deepseek_v4_attn_subkey`, `_rename_deepseek_v4_ffn_subkey`)
            # -- which is why the requant allow-list only names stems under `attn.`
            # and `ffn.`, the two subtrees that rename actually reaches.
            value = _requant_half(stem, want_scale=key.endswith(".scale"))

        # The shared remapper views MXFP4 expert storage and fuses compressor
        # tensors. Materialize retained slices only, after ownership filtering.
        if not isinstance(value, torch.Tensor):
            value = value[:]
        pre[_rename_deepseek_v41_engram_key(key)] = value

    if requant_pairs:
        # Every requantized stem has exactly a `.weight` and a `.scale` in the
        # checkpoint (that pairing is what put it in `requant_stems`), so both
        # halves must have been drawn. A leftover means one half was ignored or
        # renamed out from under the other, which would leave a Linear holding a
        # scale for a weight it never received.
        raise ValueError(
            "DeepSeek-V4.1 requantization left one half of "
            f"{len(requant_pairs)} stem(s) unconsumed, e.g. "
            f"{', '.join(sorted(requant_pairs)[:3])}"
        )

    # `_rename_deepseek_v41_engram_key` renames one leaf, so it is injective and
    # `len(pre)` is exactly the number of raw keys that were forwarded.
    census.forwarded = len(pre)
    census.verify()
    logger.info(census.render())

    return _V41ForwardedWeights(
        _remap_deepseek_v4_checkpoint_keys(
            pre, num_hidden_layers=num_hidden_layers, kv_lora_rank=kv_lora_rank
        ),
        census=census,
    )


def _rename_deepseek_v41_engram_key(key: str) -> str:
    """``layers.N.engram.<leaf>...`` -> TRT-LLM's engram sub-module names."""
    marker = ".engram."
    idx = key.find(marker)
    if idx < 0:
        return key
    head = key[: idx + len(marker)]
    rest = key[idx + len(marker) :]
    leaf, sep, tail = rest.partition(".")
    new_leaf = _ENGRAM_SUBKEY_RENAME.get(leaf, leaf)
    return f"{head}{new_leaf}{sep}{tail}" if sep else f"{head}{new_leaf}"


def _dequantize_block32(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp8 e4m3 + ue8m0 block scales -> bf16, on the GPU, back on the host.

    ``weight_dequant`` is a Triton kernel and needs both operands resident, so
    the tensor makes a round trip. Returning to the host rather than leaving the
    result on the device is deliberate: the loader's own TP split copies each
    shard to the GPU anyway, and holding all ~13 GiB of dequantized dense
    weights on the device at once -- on top of the shard being built -- would
    compete with the model's own allocation for no benefit.
    """
    x = weight[:] if not isinstance(weight, torch.Tensor) else weight
    s = scale[:] if not isinstance(scale, torch.Tensor) else scale
    y = weight_dequant(
        x.contiguous().cuda(),
        s.float().contiguous().cuda(),
        block_size=_V41_FP8_BLOCK_SIZE,
    )
    return y.to(torch.bfloat16).cpu()


def _pow2_tile_scale(amax: torch.Tensor) -> torch.Tensor:
    """Per-tile multiplier ``S = 2**ceil(log2(amax / 448))``, one per 128x128 tile.

    Power-of-two rescaling preserves E4M3 values that stay in the normal range
    and avoids overflow. Smaller values can lose precision or underflow to zero.
    """
    ratio = amax / _E4M3_MAX
    # An all-zero tile has amax == 0, and log2(0) -> -inf -> ceil -> -inf ->
    # exp2 -> 0.0: finite at every step, no NaN, but a zero scale would make the
    # reconstruction 0/0. Give it S = 1 instead: any positive scale reproduces zero
    # exactly, and 1.0 keeps the emitted scale tensor readable.
    #
    # Written with `where` rather than boolean-mask assignment because the census
    # tests audit the whole load path on `meta` tensors, and masked indexing needs
    # `nonzero()`, which meta has no data-independent implementation for.
    return torch.where(
        ratio > 0,
        torch.exp2(torch.ceil(torch.log2(ratio))),
        torch.ones_like(ratio),
    )


def _requantize_block128(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """A bf16 dense weight -> ``(fp8_e4m3, fp32 128x128 block scales)``.

    Paired with ``_dequantize_block32``: that reads V4.1's on-disk 32x32 ue8m0
    layout, this writes the 128x128 layout ``Linear``'s ``FP8_BLOCK_SCALES`` path
    already implements. The emitted scale is the **multiplier** -- ``w ~= q * S`` --
    matching what ``linear.py`` stores under the (misleadingly named)
    ``weight_scale_inv``.

    For UE8M0-scaled E4M3 checkpoint values, rescaling is exact in the normal
    range; subnormal rounding or underflow has error at most ``S * 2**-9``.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    block = _V41_REQUANT_BLOCK
    m, n = weight.shape
    tiles_m = (m + block - 1) // block
    tiles_n = (n + block - 1) // block
    w = weight.float()
    pad_m, pad_n = tiles_m * block - m, tiles_n * block - n
    if pad_m or pad_n:
        # Zero padding cannot raise a tile's amax, so the padded tile's scale is
        # the same one the real elements would have produced on their own.
        w = torch.nn.functional.pad(w, (0, pad_n, 0, pad_m))
    amax = w.reshape(tiles_m, block, tiles_n, block).abs().amax(dim=(1, 3))
    scale = _pow2_tile_scale(amax)
    expanded = scale.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    q = (w / expanded)[:m, :n].to(torch.float8_e4m3fn)
    return q.contiguous(), scale.contiguous()


def _requantize_dense_stem(
    weight: torch.Tensor, scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """V4.1's on-disk 32x32 fp8 pair -> the 128x128 fp8 pair ``Linear`` expects.

    Dequantize-then-requantize rather than a direct 32 -> 128 scale merge: the merge
    would still have to re-round every element that its 32-block's scale placed above
    the coarser tile's ceiling, so it is the same arithmetic with a hand-rolled
    kernel in the middle. ``_dequantize_block32`` already runs on the GPU and returns
    to the host; the requantization runs on whatever device it is handed.
    """
    return _requantize_block128(_dequantize_block32(weight, scale))


class DeepseekV41WeightLoader(DeepseekV4WeightLoader):
    """V4's loader, plus the three load-time audits V4.1's checkpoint needs.

    * **Layout** -- every quantized tensor's on-disk block extent is checked
      against its role before anything is dequantized, and a mismatch raises.
      V4.1 ships three different quantized layouts under one
      ``quantization_config`` (routed experts MXFP4 element block 1x32 stored
      1x16, the Engram tables fp8 e4m3 1x32, everything else fp8 e4m3 32x32),
      and ``weight_block_size`` in the config describes only the third.
    * **Census** -- ``consumed + deliberately_ignored`` must equal the number of
      tensors the checkpoint shipped, every ignored group with a name pattern and
      a reason. See ``DeepseekV41LoadCensus``.
    * **Coverage** -- after the module walk, every forwarded key must have had a
      consumer and every parameter a source. V4's walk visits modules and asks
      for their weights, so a key nothing asks for is dropped in silence: a
      rename that misses becomes a zero-initialized projection, not an error.

    All three raise. A warning is worse than nothing at 475 GiB and 96085
    tensors, where the log has long scrolled past by the time the output is wrong.
    """

    def __init__(self, model, is_draft_model: bool = False):
        super().__init__(model, is_draft_model=is_draft_model)
        self.census: Optional[DeepseekV41LoadCensus] = None
        self._forwarded: Optional[_V41ForwardedWeights] = None

    def remap_checkpoint_keys(
        self, weights: Dict, *, num_hidden_layers: int, kv_lora_rank: int = 448
    ) -> _V41ForwardedWeights:
        """Remap checkpoint keys before the V4.1 attention-subtree load.

        Bound rather than static because the census needs two things only the
        loader knows: the checkpoint's ``quantization_config`` (which layout the
        dense role expects) and how many MTP layers the model actually built.
        """
        pretrained = self.model_config.pretrained_config
        text = getattr(pretrained, "text_config", pretrained)
        # What the *model* built, not what the checkpoint ships: V4's remap only
        # routes `mtp.0.`, so one is the ceiling even if more were constructed.
        built = len(getattr(getattr(self.model, "model", None), "layers", ()))
        forwarded = _remap_deepseek_v41_checkpoint_keys(
            weights,
            num_hidden_layers=num_hidden_layers,
            kv_lora_rank=kv_lora_rank,
            keep_mtp_layers=min(1, max(0, built - int(num_hidden_layers))),
            load_vision_bias=bool(getattr(text, "use_vision_bias", False)),
            quantization_config=getattr(text, "quantization_config", None),
            context_only_split=(
                self.model.model.decoder_replay_split
                if getattr(self.model.model, "disagg_context_only", False)
                else None
            ),
            context_only_precompute=getattr(self.model.model, "ced_kv_precompute", False),
        )
        self.census = forwarded.census
        self._forwarded = forwarded
        return forwarded

    def _load_weights_impl(self, weights: Dict, skip_modules: Sequence[str] = ()) -> None:
        """Load CSA2 subtrees locally, then reuse V4's walk for other modules."""
        if any(key == "embed.weight" or key.startswith("layers.") for key in weights):
            remapped = self.remap_checkpoint_keys(
                weights,
                num_hidden_layers=self.config.num_hidden_layers,
                kv_lora_rank=self.config.kv_lora_rank,
            )
            weights = ConsumableWeightsDict.take_ownership(weights, remapped)
        elif getattr(getattr(self.model, "model", None), "disagg_context_only", False):
            # HF-style names already match the module loader. Apply ownership
            # before reading values, including the partially retained boundary.
            census = DeepseekV41LoadCensus(total=len(weights))
            retained = {}
            for key in weights:
                reason = _v41_ignore_reason(
                    key,
                    context_only_split=self.model.model.decoder_replay_split,
                    context_only_precompute=self.model.model.ced_kv_precompute,
                )
                if reason is not None:
                    census.ignore(key, reason)
                else:
                    value = weights[key]
                    retained[key] = value if isinstance(value, torch.Tensor) else value[:]
            census.forwarded = len(retained)
            census.verify()
            self.census = census
            self._forwarded = _V41ForwardedWeights(retained, census)
            weights = ConsumableWeightsDict.take_ownership(weights, self._forwarded)
        attention_subtrees = []
        for name, module in self.model.named_modules():
            if name.startswith("draft_model") or any(skip in name for skip in skip_modules):
                continue
            if not isinstance(module, DeepseekV41Attention):
                continue
            prefix = name + "."
            selected = {
                key[len(prefix) :]: value
                for key, value in weights.items()
                if key.startswith(prefix)
            }
            module.load_weights([selected])
            del selected
            if isinstance(weights, ConsumableWeightsDict):
                weights.mark_consumed(name)
            attention_subtrees.append(name)
        return super()._load_weights_impl(
            weights, skip_modules=[*skip_modules, *attention_subtrees]
        )

    def load_weights(self, weights: Dict, skip_modules: List[str] = []):
        for module in self.model.modules():
            if isinstance(module, DeepseekV41Engram):
                module._gate_norm_product = None
        result = super().load_weights(weights, skip_modules=skip_modules)
        self.assert_load_complete()
        return result

    def assert_load_complete(self) -> None:
        """Raise unless every forwarded key had a consumer and every parameter a source.

        Full-model HF-style loads skip the remap and have no census to audit.
        Context-only HF-style loads audit their filtered parameter ownership.
        """
        if self._forwarded is None:
            return
        unexpected, unfed = _v41_load_coverage(
            self.model, self._forwarded.all_keys, skip_modules=("draft_model",)
        )
        if not unexpected and not unfed:
            logger.info(
                "DeepSeek-V4.1 load coverage: "
                f"{len(self._forwarded.forwarded_keys)} forwarded + "
                f"{len(self._forwarded.synthesized_keys)} synthesized keys all had a "
                "consumer; every model parameter had a source."
            )
            # The census is rendered on the way out as well as on the way to a
            # raise: "every tensor is accounted for" is a claim a reader has to be
            # able to check, and a 475 GiB load is not something anyone reruns to
            # find out where the count went.
            if self.census is not None:
                logger.info("\n" + self.census.render())
            return
        detail = []
        if unexpected:
            detail.append(
                f"{len(unexpected)} checkpoint tensor(s) no module would load, "
                f"e.g. {unexpected[:8]}"
            )
        if unfed:
            detail.append(
                f"{len(unfed)} model parameter(s) no checkpoint tensor would fill, e.g. {unfed[:8]}"
            )
        raise ValueError(
            "DeepSeek-V4.1 weight load is incomplete: "
            + "; ".join(detail)
            + ".\n"
            + (self.census.render() if self.census is not None else "")
        )


def _candidate_prefilter_is_wired(model_config: ModelConfig[PretrainedConfig]) -> bool:
    """Whether this build actually runs both prefilter levels for this config.

    Inspect the lowered ``CSA2Params.layout`` rather than the checkpoint
    config, because that is the ownership and candidate plan CSA2Indexer reads:
    the two can disagree, and the interesting direction is a checkpoint that
    carries the three fields against a ``to_sparse_params`` that does not forward
    them. Reading the config directly would call the feature wired and drop the
    length refusal while the runtime silently skipped level one.
    """
    sparse_config = model_config.sparse_attention_config
    if sparse_config is None:
        return False
    params = sparse_config.to_sparse_params(pretrained_config=model_config.pretrained_config)
    layout = params.layout
    return (
        layout.candidate_source_layer_id is not None
        and layout.candidate_topk_blocks > 0
        and layout.candidate_block_size > 0
    )


def _candidate_inert_up_to_positions(topk_blocks: int, block_size: int, ratio: int = 1) -> int:
    return topk_blocks * block_size * max(1, ratio)


def _candidate_source_ratio(text: PretrainedConfig, candidate_source: int) -> int:
    """The compress ratio of the layer that publishes level one, defaulting to 1.

    Level one counts *compressed* positions, so the length it stays inert up to
    scales with the source layer's pooling factor. The release's source is layer
    20, in the ratio-1 band, where compressed and raw lengths coincide -- so this
    returns 1 there and the bound is unchanged. A config that moved the source
    into the pooled band would otherwise be refused at half the length it is
    actually exact to.
    """
    ratios = getattr(text, "compress_ratios", None)
    if not ratios or not 0 <= candidate_source < len(ratios):
        return 1
    return max(1, int(ratios[candidate_source]))


def _assert_candidate_prefilter_inert(model_config: ModelConfig[PretrainedConfig]) -> None:
    """Refuse a sequence length at which the missing candidate prefilter would matter.

    V4.1 runs a two-level selection: layer ``candidate_source_layer_id`` (20 in the
    release) publishes the top ``candidate_topk_blocks`` blocks of
    ``candidate_block_size`` positions each, and every *later* index source masks
    its own scores with that set before taking its top-k.

    Skipping it is exact only while the mask covers everything a consumer could
    have selected anyway, i.e. while a sequence has no more candidate positions
    than ``candidate_topk_blocks * candidate_block_size`` (16384 in the release).
    Past that the reference scores a strict subset of what we score, so our top-k
    can contain positions the reference excluded -- a silent accuracy loss that
    grows with length and is invisible in any short-context test.

    Hence a refusal at construction rather than a warning at the first long
    request: the check is on the configured ceiling, so a run either cannot start
    or is provably in the inert regime for its whole life. ``max_seq_len`` unset
    means the caller has not committed to a ceiling, and the model's own
    ``max_position_embeddings`` (1 Mi) is far past the threshold, so that is
    refused too rather than assumed short.
    """
    if _candidate_prefilter_is_wired(model_config):
        # Both levels run, so the length ceiling this function exists to enforce
        # no longer binds. Kept as a function rather than deleted: the refusal is
        # still the correct behaviour for any configuration where the wiring is
        # off, and "is it wired" is a property of the build, not of the request.
        return

    sparse_config = model_config.sparse_attention_config
    pretrained = model_config.pretrained_config
    text = getattr(pretrained, "text_config", pretrained)
    candidate_source = getattr(text, "candidate_source_layer_id", None)
    if candidate_source is None or sparse_config is None:
        return

    blocks = int(getattr(text, "candidate_topk_blocks", 0) or 0)
    block_size = int(getattr(text, "candidate_block_size", 0) or 0)
    if blocks <= 0 or block_size <= 0:
        return
    # Not the bare `blocks * block_size` product: the bound is on *compressed*
    # positions, so a pooled candidate source is inert to a proportionally longer
    # raw sequence. Identity at the release's ratio-1 source (2048 * 8 = 16384).
    inert_up_to = _candidate_inert_up_to_positions(
        blocks, block_size, _candidate_source_ratio(text, int(candidate_source))
    )

    max_seq_len = model_config.max_seq_len
    if max_seq_len is not None and int(max_seq_len) <= inert_up_to:
        return
    configured = "unset" if max_seq_len is None else f"{int(max_seq_len)}"
    raise ValueError(
        "DeepSeek-V4.1's two-level candidate prefilter is not wired up for this "
        f"configuration, so this model is only exact up to max_seq_len={inert_up_to} "
        f"(candidate_topk_blocks={blocks} x candidate_block_size={block_size}); "
        f"got max_seq_len={configured}. Above that length, layer "
        f"{int(candidate_source)} would restrict which positions the later index "
        "sources may choose; skipping it silently widens their candidate set. "
        f"Set max_seq_len <= {inert_up_to} explicitly, or check that the sparse "
        "attention config forwards candidate_source_layer_id / "
        "candidate_topk_blocks / candidate_block_size into CSA2Params.layout -- the "
        "Indexer reads them from there, and this refusal only fires when it will "
        "not see them."
    )


class _OmittedContextDecoder(nn.Module):
    """Keep layer indices stable without retaining any decoder parameters."""

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise RuntimeError("Decoder execution is unavailable on a context-only worker")


class _ContextOnlyLogitsProcessor(nn.Module):
    """Supply sampler storage for context handoffs, which carry no output token."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
        attn_metadata: CSA2TrtllmMetadata,
        return_context_logits: bool = False,
    ) -> torch.Tensor:
        if return_context_logits:
            raise ValueError("Context decoder skipping does not support context logits")
        batch_size = attn_metadata.seq_lens_cuda.numel()
        return hidden_states.new_zeros((batch_size, self.vocab_size), dtype=torch.float32)


@register_auto_model("DeepseekV41ForCausalLM")
class DeepseekV41ForCausalLM(DeepseekV4ForCausalLM):
    """DeepSeek-V4.1.

    The target keeps its lagged mHC stream separate from the optional embedded
    DSpark drafter, which loads all three heterogeneous MTP stages separately.

    Remaining limitations:

    * **Other speculative modes.** Ordinary MTP and external draft modes are
      rejected; only the embedded DSpark construction is wired for V4.1.
    * **Vision.** Image encoding is owned by the multimodal wrapper.
    * **Engram WKV.** Native MXFP8 is always enabled; dense attention
      and shared experts always use FP8.
    """

    model_cls = DeepseekV41Model
    weight_loader_cls = DeepseekV41WeightLoader

    def __pp_init__(self) -> None:
        super().__pp_init__()
        self.model_config.extra_attrs.pop("csa2_context_swa_layer_limit", None)
        if not self.model.disagg_context_only:
            return
        # ModelLoader constructs under MetaInitMode and materializes only after
        # this hook, just as for PP pruning. Decoder tensors never reach the GPU
        # or the checkpoint loader on a context-only worker.
        split = self.model.decoder_replay_split
        for index in range(split, len(self.model.layers)):
            layer = self.model.layers[index]
            if index == split and self.model.ced_kv_precompute:
                for name in list(layer._modules):
                    if name not in ("input_layernorm", "self_attn"):
                        delattr(layer, name)
                layer.engram = None
                attention = layer.self_attn
                for name in list(attention._modules):
                    if name not in (
                        "compressor",
                        "index_wk",
                        "index_k_norm",
                        "rotary_emb",
                        "backend",
                    ):
                        delattr(attention, name)
                for name in list(attention._parameters):
                    delattr(attention, name)
                for name in list(attention._buffers):
                    delattr(attention, name)
            else:
                self.model.layers[index] = _OmittedContextDecoder()
        self.model.norm = _OmittedContextDecoder()
        self.lm_head = _OmittedContextDecoder()
        self.epilogue = [self.lm_head]
        self.logits_processor = _ContextOnlyLogitsProcessor(self.config.vocab_size)
        # Publish the validated execution boundary only after pruning. The
        # cache allocator and its estimator consume the same derived state;
        # neither may infer a different boundary from checkpoint layer counts.
        self.model_config.extra_attrs["csa2_context_swa_layer_limit"] = split
        logger.info(
            "DeepSeek-V4.1 context-only weight loading: omitted decoder layers "
            f"{split}-{self.config.num_hidden_layers - 1} and output head; "
            f"retained boundary GLOBAL KV projections={self.model.ced_kv_precompute}."
        )

    def _post_load_decoder_layers(self) -> nn.ModuleList:
        end = (
            self.model.decoder_replay_split
            if self.model.disagg_context_only
            else self.config.num_hidden_layers
        )
        return self.model.layers[:end]

    def post_load_weights(self):
        super().post_load_weights()
        # Cross-layer deferral of hc_ffn.post_mapping into the next layer's fused lagged
        # entry (mHC.fused_hc_lagged): one fewer kernel per layer at decode. Not across an
        # engram entry, the CED boundary (both need a materialized residual), or the final layer;
        # TRTLLM_V41_MHC_DEFER=0 turns it off for A/B.
        defer_ok = os.environ.get("TRTLLM_V41_MHC_DEFER", "1").lower() not in ("0", "false")
        defer_ok = defer_ok and not self.model_config.mapping.has_pp()
        layers = self._post_load_decoder_layers()
        for idx, layer in enumerate(layers):
            next_layer = layers[idx + 1] if idx + 1 < len(layers) else None
            layer._v41_defer_post_mapping = bool(
                defer_ok
                and next_layer is not None
                and getattr(next_layer, "engram", None) is None
                and not (
                    self.model.ced_kv_precompute and idx + 1 == self.model.decoder_replay_split
                )
            )

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        # Both of these are refusals rather than silent degradations: each one
        # would otherwise produce numerically wrong output that looks like a
        # model bug.
        #
        # Pipeline parallelism: the lagged `pre` is a second piece of
        # cross-layer state, carried on `HCState.pre_mix` alongside the residual.
        # `DeepseekV4Model.forward`'s PP branch passes hidden states between
        # ranks and has no channel for it, so a PP split would restart every
        # rank's stream from the identity seed. The release target is TP8/EP8 on
        # one node, so nothing needs it yet.
        if model_config.mapping is not None and model_config.mapping.has_pp():
            raise ValueError(
                "DeepSeek-V4.1 does not support pipeline parallelism: its mHC "
                "stream carries a lagged `pre` across layer boundaries "
                "(HCState.pre_mix) that the PP hidden-state exchange does not "
                "transport. Use tensor/expert parallelism instead "
                f"(got pp_size={model_config.mapping.pp_size})."
            )
        # V4.1's heterogeneous MTP stages are built by modeling_dspark_v41.
        # Other draft builders do not implement its lagged mHC semantics.
        if (
            model_config.spec_config is not None
            and not model_config.spec_config.spec_dec_mode.is_dspark()
        ):
            raise ValueError(
                "DeepSeek-V4.1 supports its embedded DSpark draft stages; "
                "ordinary MTP and other speculative draft modes are not supported "
                f"(got spec_config={type(model_config.spec_config).__name__})."
            )
        if (
            disagg_context_decoder_skipping_enabled(
                getattr(model_config, "extra_attrs", {}).get("bounded_replay_on_generation", False)
            )
            and model_config.spec_config is not None
        ):
            raise ValueError(
                "Context decoder skipping requires speculative decoding to be disabled"
            )
        _assert_candidate_prefilter_inert(model_config)
        super().__init__(model_config)

    def prepare_adp_inputs(
        self, attn_metadata, *, all_token_states_required: bool, requests=None
    ) -> None:
        self.model.prepare_adp_inputs(
            attn_metadata,
            all_token_states_required=all_token_states_required,
            requests=requests,
        )

    def prepare_request_inputs(
        self, scheduled_requests, attn_metadata, promoted_context_request_ids=frozenset()
    ) -> None:
        """Restore Engram's raw n-gram lookback without widening model inputs."""
        provider = getattr(self.model, "engram_hash_provider", None)
        if provider is None or not getattr(self.model, "use_engram", False):
            return
        contexts = [
            request for request in scheduled_requests.context_requests if not request.is_dummy
        ]
        contexts.extend(
            request
            for request in scheduled_requests.generation_requests
            if not request.is_dummy and request.py_request_id in promoted_context_request_ids
        )
        if not contexts:
            return
        lookback = provider.config.max_ngram_size - 1
        prefixes = {}
        from ..pyexecutor.ced_replay import encoder_replay_tokens

        token_masks = {}
        for request in contexts:
            disagg_params = getattr(request, "py_disaggregated_params", None)
            if (
                disagg_params is not None
                and disagg_params.schedule_style == DisaggScheduleStyle.GENERATION_FIRST
            ):
                end = request.prompt_len
                tail_mask = _engram_history_text_mask(request, max(0, end - lookback), end)
                if tail_mask is not None and not all(tail_mask):
                    raise ValueError(
                        "DeepSeek-V4.1 image spans in the Engram prompt lookback require "
                        "context-first disaggregation"
                    )
            start = request.context_current_position - encoder_replay_tokens(request)
            prefix_start = max(0, start - lookback)
            prefixes[request.py_request_id] = (
                start,
                list(request.get_tokens_range(0, prefix_start, start)),
            )
            mask = _engram_history_text_mask(request, prefix_start, start)
            if mask is not None:
                token_masks[request.py_request_id] = mask
        provider.seed_context_history(
            [
                request.py_request_id
                for request in scheduled_requests.all_requests()
                if not request.is_dummy
            ],
            prefixes,
            max_seq_len=attn_metadata.kv_cache_manager.max_seq_len,
            device=self.model.embed_tokens.weight.device,
            token_masks=token_masks or None,
        )

    def prepare_disagg_generation_request(self, request: "LlmRequest") -> None:
        """Restore prompt lookback, retaining image boundaries after KV transfer."""
        provider = self.model.engram_hash_provider
        if provider is None or not self.model.use_engram:
            return
        end = request.prompt_len
        start = max(0, end - provider.config.max_ngram_size + 1)
        mask = _engram_history_text_mask(request, start, end)
        provider.queue_history_seed(
            request.py_request_id,
            start,
            request.get_tokens_range(0, start, end),
            token_mask=mask,
        )

    @classmethod
    def get_model_defaults(cls, llm_args: "TorchLlmArgs") -> dict:
        """Default to cached-context MLA and addressable SWA pages for replay.

        Scratch reuse rotates a small SWA buffer across prefill chunks; replay
        needs paged window positions that remain addressable after metadata is
        rebuilt. Cross-request prefix reuse retains required Encoder checkpoints
        unless Encoder replay permits independent eviction and recomputation.
        Deferred execution keeps prefix reuse disabled by default.

        A replay deployment also needs ``enable_chunked_prefill=True`` to enable
        cached-context MLA while ``kv_cache_config.enable_block_reuse=False``.
        The chunked-prefill capability is needed even when a prompt fits within
        one scheduler chunk: decoder query rows attend to the full cache that
        the encoder pass has already written.
        """
        defaults = super().get_model_defaults(llm_args)
        # CSA2 does not use V4's SM90 packed-cache dtype default.
        defaults["kv_cache_config"].pop("dtype", None)
        if decoder_bounded_replay_enabled():
            defaults["enable_chunked_prefill"] = True
            defaults.setdefault("kv_cache_config", {}).update(
                enable_block_reuse=False, enable_swa_scratch_reuse=False
            )
        return defaults

    def forward(
        self,
        *args,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Tell the model body whether anything downstream reads non-final rows.

        ``return_context_logits`` cannot simply be inspected inside the body:
        ``SpecDecOneEngineForCausalLM.forward`` takes it as a named parameter and
        consumes it itself, so it never appears in the ``**kwargs`` forwarded to
        ``self.model``. Hence a separate key, and hence it is *derived* here rather
        than read from the metadata -- the body must not have to guess.
        """
        all_token_states_required = bool(return_context_logits)
        spec_metadata = kwargs.get("spec_metadata")
        if (
            isinstance(spec_metadata, DSparkSpecMetadata)
            and spec_metadata.gather_ids is not None
            and spec_metadata._requires_all_context_logits is not None
        ):
            # The engine uses return_context_logits for both full context
            # logits and sparse speculative gather. Only its explicit demand
            # can disambiguate them; unknown/direct callers remain conservative.
            all_token_states_required = spec_metadata._requires_all_context_logits
        kwargs["all_token_states_required"] = all_token_states_required
        return super().forward(*args, return_context_logits=return_context_logits, **kwargs)

    def register_cuda_graph_pre_replay_hooks(self, runner) -> None:
        """Refresh local hash rows before the captured DP gather and lookup."""
        if not self.model.use_engram or self.model.engram_hash_provider is None:
            return

        def _refresh(
            key, num_tokens: int, static_tensors: dict[str, Any], current_inputs: dict[str, Any]
        ) -> None:
            if current_inputs.get("input_ids") is None:
                return
            attn_metadata = current_inputs.get("attn_metadata")
            if attn_metadata is None:
                raise ValueError("Engram CUDA graph replay requires attention metadata.")
            position_ids = static_tensors.get("position_ids")
            self.model._compute_engram_hashes(
                static_tensors["input_ids"][:num_tokens],
                None if position_ids is None else position_ids[..., :num_tokens],
                attn_metadata,
                padded_num_tokens=num_tokens,
                refresh=True,
            )

        runner.register_pre_replay_hook(_refresh)
