# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2-NVFP4 on 16 Rubin GPUs (sm107), tp16: model construction and forward.

This target serves the generation side of a disaggregated deployment. Its
context server hands over its MLA KV cache and DSA indexer K cache. The target
owns its modules and weight loader, with checkpoint and cache layouts matching
the context server. Forwards that carry context rows
(engine warm-up, an aggregated deployment) run that same decoder stack in
``PrefillTarget``. Generation-only forwards -- every serving step of the
generation server -- run ``DecodeTarget``. With one-model MTP-Eagle, the draft
loop of a one-request generation step runs on ``DecodeTarget``'s kernels as
well (``MTPDraftWorker``).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple

import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.module import forward_generation_sparse_attn
from tensorrt_llm._torch.attention.backends.sparse.dsa.params import DSABackendForwardArgs
from tensorrt_llm._torch.attention.backends.sparse.params import MTPIndexShareMetadata
from tensorrt_llm._torch.distributed import AllReduceFusionOp, AllReduceParams
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_speculative import _slice_spec_position_ids
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm._torch.modules.swiglu import swiglu
from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.kernel_inputs import (
    extract_routing_params,
    get_data_or_none,
    to_trtllm_gen_act_type,
)
from tensorrt_llm._torch.speculative import MTPEagleWorker, SpecMetadata
from tensorrt_llm._torch.speculative.eagle3 import _reset_mtp_index_share
from tensorrt_llm._torch.speculative.interface import (
    _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
    _FORCE_ACCEPT_RNG_POOL_SIZE,
    _FORCE_ACCEPT_RNG_SLOT_STRIDE,
)
from tensorrt_llm._torch.utils import EventType

from . import weights as _weights
from ._model import Glm52Model
from ._model import setup_aliases as _setup_aliases
from ._speculative import Glm52SpecDecOneEngineForCausalLM
from .ops.attention.mla_kv_half_append import mla_kv_half_append
from .ops.attention.sparse_mla_flash_decode import sparse_mla_flash_decode
from .ops.comm._mnnvl_workspace import (
    LamportExchangeWorkspace,
    hop_supported,
    hop_workspace,
    publish_target,
)
from .ops.comm.mnnvl_allgather import mnnvl_allgather
from .ops.comm.mnnvl_hop import mnnvl_hop
from .ops.comm.mnnvl_moe_finalize_hop import mnnvl_moe_finalize_hop
from .ops.dsa._fp8_paged_mqa_logits import num_sms
from .ops.dsa._hist_topk import (
    MAX_SHARD,
    TOPK,
    get_hist_topk_workspace,
    hist_topk_supported,
    shard_capacity,
)
from .ops.dsa.convert_req_index_to_global import convert_req_index_to_global
from .ops.dsa.fp8_paged_mqa_logits import fp8_paged_mqa_logits
from .ops.dsa.hist_shard_meta import hist_shard_meta
from .ops.dsa.hist_topk import hist_topk
from .ops.dsa.indexer_k_tail import indexer_k_tail
from .ops.dsa.paged_mqa_logits_schedule import paged_mqa_logits_schedule
from .ops.dsa.stash_last_accepted_rows import stash_last_accepted_rows
from .ops.embedding.fused_embedding import fused_embedding
from .ops.gemm.dense_mlp_small_m import INTER_PER_RANK, dense_mlp_small_m
from .ops.gemm.draft_hidden_half import draft_hidden_half
from .ops.gemm.fused_a_gemm import INDEXER_OUT_FEATURES as FUSED_A_INDEXER_OUT_FEATURES
from .ops.gemm.fused_a_gemm import MAX_TOKENS as FUSED_A_MAX_TOKENS
from .ops.gemm.fused_a_gemm import OUT_FEATURES as FUSED_A_OUT_FEATURES
from .ops.gemm.fused_a_gemm import fused_a_gemm
from .ops.gemm.indexer_q_tail import indexer_q_tail
from .ops.gemm.low_m_gemm import low_m_gemm
from .ops.gemm.o_proj_publish import o_proj_publish
from .ops.gemm.q_b_fp8_rope import q_b_fp8_rope
from .ops.gemm.router_gemm import router_gemm
from .ops.moe.lowm_bf16_moe import glm_lowm_bf16_moe
from .ops.moe.routed_experts import (
    FUSED_ROUTER_MAX_TOKENS,
    make_fused_router_workspace,
    routed_experts,
)
from .ops.quantization.fp4_quantize import fp4_quantize
from .ops.spec._draft_exchange import HIDDEN_SIZE as DRAFT_HIDDEN_SIZE
from .ops.spec._draft_exchange import DraftExchangeWorkspace, ehproj_table_supports_tp
from .ops.spec._draft_token_table import build_draft_token_table
from .ops.spec.draft_ehproj_table import glm_draft_ehproj_table
from .ops.spec.draft_glue_rmsnorm_cat import glm_draft_glue_rmsnorm_cat
from .ops.spec.draft_sample_glue import glm_draft_sample_glue
from .ops.spec.gather_last_accepted_rows import glm_gather_last_accepted_rows
from .ops.spec.verify_head_accept import glm_verify_head_accept

# The deployment this target is written for; route() selects it on the same
# facts, and the constructor states them again for a direct construction.
_COMPUTE_CAPABILITY = (10, 7)
_TP = 16
# One Lamport buffer of the verify logits' all-gather: every rank's BF16 vocabulary shard of
# up to 8 rows, with room to spare.
_LOGITS_EXCHANGE_BYTES = 16 << 20


class PrefillTarget:
    """Forwards that carry context rows: the decoder stack the context server runs."""

    def __init__(self, core: "Glm52Nvfp4Sm107Tp16Core") -> None:
        self.core = core

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        return Glm52Model.forward(
            self.core,
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )


@dataclass(frozen=True, eq=False)
class _DraftState:
    """What the MTP draft steps hold for the engine's life (``DecodeTarget.build_draft``)."""

    # The MTP layer (the built-in module at layer num_hidden_layers) and the LM head it shares.
    layer: Any
    lm_head: Any
    # Detached views of the embedding table and the enorm / hnorm weights for the CuTe kernels.
    embed_table: torch.Tensor
    enorm_weight: torch.Tensor
    hnorm_weight: torch.Tensor
    # This rank's columns [lo, hi) of eh_proj's concatenated input (eh_proj is row-parallel),
    # and whether draft step 0's glue kernel launches with PDL.
    window: Tuple[int, int]
    pdl: bool
    # What fused-A prefetches into L2 on the draft steps that run no indexer.
    l2_after_fused_a: List[torch.Tensor]
    # The residual of draft step 0's eh_proj hop: zeros, so its residual output is the sum.
    zero_residual: torch.Tensor
    # glm_draft_ehproj_table's operands: the token half of eh_proj for the tokens of this rank's
    # vocabulary shard, and this rank's rows of eh_proj's hidden half.
    token_table: torch.Tensor
    hidden_rows: torch.Tensor
    # The low-M BF16 MoE's experts and routing constants.
    w3_w1: torch.Tensor
    w2: torch.Tensor
    top_k: int
    routed_scaling_factor: float
    # The post-MoE hop's routing: one slot of weight 1 per row, holding that row (-1 elsewhere).
    expert_weights: torch.Tensor
    expert_rows: torch.Tensor
    # Draft step 0's top-k of the request's last accepted row, and the request's index.
    topk_stash: torch.Tensor
    request_rows: torch.Tensor
    # The exchange workspaces of glm_draft_ehproj_table and glm_draft_sample_glue.
    table_exchange: DraftExchangeWorkspace
    sample_exchange: DraftExchangeWorkspace
    # The side stream that stashes that top-k and maps it to KV pool rows, its fork and ready
    # events, and the events of the hidden half, which runs on the MTP layer's MoE aux stream.
    stream: torch.cuda.Stream
    selection_fork: torch.cuda.Event
    selection_ready: torch.cuda.Event
    half_start: torch.cuda.Event
    half_done: torch.cuda.Event


class DecodeTarget:
    """Generation-only forwards: the serving step of the generation server.

    Routed to only when the batch has no context rows, which the core decides
    per step; nothing in this class reads the phase back.

    A step of up to 8 rows (one request and its 7 draft tokens at c1) runs on
    the target's kernels, layer by layer:

    - The embedding is one lookup in the whole BF16 table every rank holds.
    - Attention: q_a, the compressed KV and its rope part come from one GEMM over
      the stored kv_a_proj_with_mqa weight. q_b, with k_b absorbed and the q_a
      norm folded, writes the RoPE'd FP8 query in its epilogue, while on the aux
      stream one kernel normalizes, ropes and appends the KV half to the FP8
      cache. The flash decode attends over the 2048 selected rows at 4 heads with
      v_b folded in, and o_proj's epilogue publishes this rank's shard into the
      all-reduce workspace.
    - The DSA indexer (on the layers that run one) shares fused-A: its K and
      weight rows are fused-A's last 160 columns. Its Q comes from wq_b (q_a norm
      folded) with an epilogue that ropes, quantizes and scales; its K tail
      normalizes, ropes, quantizes and writes the indexer K cache on a side
      stream. Each rank scores its 1/16 shard of the context's keys, and the ranks
      agree on the exact top-2048 by exchanging histograms. The layers without an
      indexer reuse the last selection, which the flash decode maps to cache rows
      itself and starts gathering before its wait.
    - Every all-reduce is a one-shot MNNVL hop that adds the residual and applies
      the following RMSNorm: post_attention_layernorm after attention, and after
      the MLP or MoE the next layer's input_layernorm (the model's final norm
      after the last layer).
    - The dense layers 0-2 run this rank's 768 intermediate columns as two
      weight-streaming kernels.
    - The MoE layers run the shared expert as two split-K GEMMs around SwiGLU on
      the aux stream, and the routed experts as one trtllm-gen call that runs the
      router GEMV in its routing kernel, joins the NVFP4 input (quantized on its
      own stream) right before FC1, runs FC2 on the K = 128 kernel and leaves the
      finalize step to the post-MoE hop.

    The verify logits come from the LM head on this rank's vocabulary shard
    (split-K GEMM) and one MNNVL all-gather that widens them to FP32.

    A larger batch runs the built-in modules in the same order.

    The MTP draft steps of a one-request step run here too (``draft_first_step``,
    ``draft_next_step``, ``draft_head``, ``draft_ehproj_table``, ``draft_sample_glue``),
    on state that ``build_draft`` makes once:

    - Draft step 0 runs on the verify rows. One kernel writes this rank's window of
      eh_proj's input (embedding gather, enorm, hnorm, concat, column split), eh_proj
      runs on it as a split-K GEMM, and its hop applies input_layernorm with a zero
      residual. The MTP layer's attention is a verify full-indexer layer's; each
      request's top-k of its last accepted row is kept, as KV pool rows of the MTP
      layer, for the later steps (on a side stream). After the post-attention hop each
      request keeps the row of its last accepted token.
    - The later steps take eh_proj's output and its input_layernorm from the previous
      step's table kernel. They run no indexer: they attend to draft step 0's
      selection, or, with a context short enough that the built-in metadata skips the
      indexer, to every position up to their own.
    - Every step's MoE runs on that one row: the router GEMV and the low-M BF16 experts,
      the shared expert on the aux stream, and the hop that adds both and the residual
      and applies shared_head.norm.
    - The draft LM head runs on this rank's vocabulary shard. Before every step but the
      last, the next step's eh_proj hidden half starts first on the aux stream, and
      ``glm_draft_ehproj_table`` exchanges the ranks' argmax and writes the draft
      token, the next step's eh_proj output from a token table built at load, its
      positions and the sequence and KV lengths. The last step's
      ``glm_draft_sample_glue`` writes its token and the next-new-tokens rows.
    """

    def __init__(self, core: "Glm52Nvfp4Sm107Tp16Core") -> None:
        self.core = core
        config = core.model_config.pretrained_config
        self.mapping = core.model_config.mapping
        self.hidden_size = config.hidden_size
        self.index_topk = config.index_topk
        layers = core.layers[: core.num_hidden_layers]
        for layer in layers:
            assert layer.allreduce is not None and layer.next_layer_layernorm is not None
            self._check_attention(layer)
            if isinstance(layer.mlp, GatedMLP):
                assert (
                    layer.mlp_tp_size == _TP
                    and not layer.disable_attn_allreduce
                    and not layer.fusion_config.PRE_MLP_FUSION
                    and layer.fusion_config.POST_MLP_FUSION
                    and layer.post_attention_layernorm.nvfp4_scale is None
                ), f"layer {layer.layer_idx}: unexpected dense-layer configuration"
            else:
                assert (
                    layer.disable_attn_allreduce
                    and layer.fusion_config.PRE_MOE_FUSION
                    and layer.fusion_config.POST_MOE_FUSION
                    and layer.mapping.is_multi_node()
                ), f"layer {layer.layer_idx}: unexpected MoE-layer configuration"
                self._check_shared_expert(layer)
                backend = getattr(layer.mlp.experts, "backend", layer.mlp.experts)
                assert (
                    backend.has_nvfp4
                    and backend.scaling_vector_size == 16
                    and getattr(backend, "fc31_act_scale", None) is None
                    and not backend.bias
                    and backend.w3_w1_weight.shape[-1] * 2 == config.hidden_size
                ), f"layer {layer.layer_idx}: routed experts are not unpadded TRTLLM-Gen NVFP4"

        # The attention operands the kernels read, per layer index (_fold_attention_weights).
        self._q_b_absorbed = {}
        self._fused_a_indexer_weight = {}
        self._wq_b_folded = {}
        self._l2_after_fused_a = {}
        for layer in layers:
            self._fold_attention_weights(layer)
        # The key-sharded top-k exchanges histograms over the TP group's MNNVL workspace
        # (created here, on every rank in the same order).
        self._hist_workspace = (
            get_hist_topk_workspace(self.mapping) if hist_topk_supported(self.mapping) else None
        )
        self._hist_buffers = None
        self._logits_workspace = LamportExchangeWorkspace(self.mapping, _LOGITS_EXCHANGE_BYTES)
        self._k_tail_stream = torch.cuda.Stream()
        self._k_tail_fork = torch.cuda.Event()
        self._k_tail_done = torch.cuda.Event()

        device = core.embed_tokens.weight.device
        # The attention scales the KV-half kernel writes and the flash decode reads.
        self._bmm1_scale = torch.empty(2, dtype=torch.float32, device=device)
        self._bmm2_scale = torch.empty(1, dtype=torch.float32, device=device)
        self._unit_scale = torch.ones(1, dtype=torch.float32, device=device)

        moe_layers = [layer for layer in layers if not isinstance(layer.mlp, GatedMLP)]
        # One exchange workspace per router call site: its launches must run in stream order.
        self._router_workspace = {
            layer.layer_idx: make_fused_router_workspace(device) for layer in moe_layers
        }
        # The routing kernel runs the router GEMV, so the logits it is handed only give the
        # shape; autotuner profiling routes from them, so they hold random values.
        self._router_logits = torch.randn(
            FUSED_ROUTER_MAX_TOKENS, config.n_routed_experts, device=device
        )
        self._quant_stream = torch.cuda.Stream()
        self._quant_fork = torch.cuda.Event()
        self._quant_ready = torch.cuda.Event()
        self._fast_rows = {}
        # Whether the last verify forward put the indexer layers on this rank's key shard; draft
        # step 0 of the same step runs its indexer on that shard.
        self._keys_sharded = False
        self._draft: Optional[_DraftState] = None

    def _check_attention(self, layer) -> None:
        """Assert the layer's MLA weights and KV cache are the layout the kernels take."""
        mla = layer.self_attn
        assert not mla.apply_rotary_emb and not mla.mapping.has_cp_helix()
        assert (
            tuple(mla.kv_a_proj_with_mqa.weight.shape)
            == (
                FUSED_A_OUT_FEATURES,
                self.hidden_size,
            )
            and mla.kv_a_proj_with_mqa.weight.dtype == torch.bfloat16
        ), f"layer {layer.layer_idx}: kv_a_proj_with_mqa is not the BF16 [q_a; kv_a] weight"
        assert (
            getattr(mla.mqa, "has_fp8_kv_cache", False)
            and mla.kv_cache_dtype != "fp8_ds_mla"
            and mla.k_b_proj_trans.dtype == torch.bfloat16
            and mla.v_b_proj.dtype == torch.bfloat16
            and mla.v_b_proj.is_contiguous()
            and mla.q_b_proj.weight.dtype == torch.bfloat16
            and mla.o_proj.weight.dtype == torch.bfloat16
        ), f"layer {layer.layer_idx}: unexpected MLA weights or KV cache layout"
        assert layer.input_layernorm.nvfp4_scale is None, "kv_a_proj is not NVFP4"

    @staticmethod
    def _check_shared_expert(layer) -> None:
        """Assert the MoE layer's shared expert is the plain BF16 SwiGLU MLP the kernels run."""
        shared = layer.mlp.shared_experts
        assert (
            shared is not None
            and layer.mlp.shared_output_scale is None
            and shared._is_plain_swiglu()
            and shared.swiglu_limit is None
            and shared.gate_up_proj.weight.dtype == torch.bfloat16
            and shared.down_proj.weight.dtype == torch.bfloat16
        ), f"layer {layer.layer_idx}: the shared expert is not a plain BF16 SwiGLU MLP"

    def _fold_attention_weights(self, layer) -> None:
        """The layer's attention operands in the forms the kernels read, by layer index."""
        mla = layer.self_attn
        index = layer.layer_idx
        # q_b with k_b absorbed and the q_a norm weight folded: for head h, rows
        # [h * 576, h * 576 + 512) are k_b_trans[h] @ W_q_b[h, :qk_nope] and the next 64 rows
        # W_q_b[h, qk_nope:], every column times gamma_q_a; FP64, rounded to BF16 once.
        qk_head_dim = mla.qk_nope_head_dim + mla.qk_rope_head_dim
        w = mla.q_b_proj.weight.view(mla.num_heads_tp, qk_head_dim, mla.q_lora_rank).double()
        nope = torch.bmm(mla.k_b_proj_trans.double(), w[:, : mla.qk_nope_head_dim])
        absorbed = torch.cat([nope, w[:, mla.qk_nope_head_dim :]], 1)
        absorbed = absorbed * mla.q_a_layernorm.weight.double()
        self._q_b_absorbed[index] = (
            absorbed.to(torch.bfloat16).reshape(-1, mla.q_lora_rank).contiguous()
        )

        # A layer that runs the indexer gets its fused-A weight: kv_a_proj_with_mqa's rows,
        # then the indexer's wk and weights_proj rows (BF16), in one tensor whose first rows
        # the built-in kv_a_proj and the indexer's projections then view, so nothing is stored
        # twice. Its indexer Q comes from wq_b with the q_a norm weight folded into its columns.
        indexer = mla.mqa.indexer
        if indexer is not None:
            assert (
                indexer.wk.weight.dtype == torch.bfloat16
                and indexer.weights_proj.weight.dtype == torch.bfloat16
            ), "the indexer's wk and weights_proj must be BF16 (TRTLLM_DSA_INDEXER_BF16=1)"
            kv_a = mla.kv_a_proj_with_mqa.weight
            wk = indexer.wk.weight
            extended = torch.cat([kv_a, wk, indexer.weights_proj.weight]).contiguous()
            assert extended.shape[0] == FUSED_A_INDEXER_OUT_FEATURES
            mla.kv_a_proj_with_mqa.weight.data = extended[: kv_a.shape[0]]
            indexer.wk.weight.data = extended[kv_a.shape[0] : kv_a.shape[0] + wk.shape[0]]
            indexer.weights_proj.weight.data = extended[kv_a.shape[0] + wk.shape[0] :]
            self._fused_a_indexer_weight[index] = extended
            self._wq_b_folded[index] = (
                indexer.wq_b.weight.double() * mla.q_a_layernorm.weight.double()
            ).to(torch.bfloat16)

        # What fused-A prefetches into L2 while it waits for its input: the weights of the
        # attention GEMMs that follow it.
        prefetch = [self._q_b_absorbed[index], mla.v_b_proj, mla.o_proj.weight]
        if index in self._wq_b_folded:
            prefetch.insert(1, self._wq_b_folded[index])
        self._l2_after_fused_a[index] = [t for t in prefetch if t.is_contiguous()]

    @torch.no_grad()
    def build_draft(self, layer, lm_head, weights: _weights.DraftWeights) -> None:
        """Build what the MTP draft steps hold for the engine's life, once.

        ``layer`` is the MTP layer and ``lm_head`` the vocabulary-parallel LM head it shares;
        ``weights`` holds the checkpoint tensors the token table and the low-M MoE take.
        Collective: every rank creates the two draft exchange workspaces here, in the same
        order. The draft steps after the first reuse draft step 0's selection, as the built-in
        indexer does under MTP index share; without that share, or where the kernels do not
        serve one row, nothing is built and the built-in draft loop runs.
        """
        if self._draft is not None:
            return
        core = self.core
        config = core.model_config.pretrained_config
        mapping = self.mapping
        tp_size, tp_rank = mapping.tp_size, mapping.tp_rank
        hidden = self.hidden_size
        mla = layer.self_attn
        indexer = mla.mqa.indexer
        if (
            indexer is None
            or not indexer.mtp_index_share
            or not ehproj_table_supports_tp(tp_size)
            or not hop_supported(1, hidden, tp_size)
        ):
            return
        self._check_attention(layer)
        self._check_shared_expert(layer)
        columns = 2 * hidden // tp_size
        norms = (layer.enorm, layer.hnorm, layer.input_layernorm, layer.shared_head.norm)
        assert (
            hidden == DRAFT_HIDDEN_SIZE
            and layer.eh_proj.weight.dtype == torch.bfloat16
            and tuple(layer.eh_proj.weight.shape) == (hidden, columns)
            and weights.eh_proj.dtype == torch.bfloat16
            and tuple(weights.eh_proj.shape) == (hidden, 2 * hidden)
            and all(
                norm.weight.dtype == torch.bfloat16 and tuple(norm.weight.shape) == (hidden,)
                for norm in norms
            )
            and layer.enorm.variance_epsilon == layer.hnorm.variance_epsilon
        ), "the MTP layer's eh_proj and norms are not the BF16 GLM-5.2 layout"
        experts = config.n_routed_experts
        inter = config.moe_intermediate_size // tp_size
        gate = layer.mlp.gate
        assert (
            config.n_group == 1
            and gate.weight.dtype == torch.bfloat16
            and tuple(gate.weight.shape) == (experts, hidden)
            and tuple(weights.w3_w1.shape) == (experts, 2 * inter, hidden)
            and tuple(weights.w2.shape) == (experts, hidden, inter)
        ), "the MTP layer's MoE is not the one-group BF16 MoE of the low-M kernels"
        embed_table = core.embed_tokens.weight
        vocab_shard = lm_head.weight.shape[0]
        assert (
            lm_head.weight.dtype == torch.bfloat16
            and vocab_shard * tp_size == lm_head.num_embeddings == embed_table.shape[0]
        ), "the draft LM head is not an unpadded BF16 vocabulary shard"

        # The collective part first, before anything that can fail on one rank only.
        table_exchange = DraftExchangeWorkspace.for_ehproj_table(mapping)
        sample_exchange = DraftExchangeWorkspace.for_sample_glue(mapping)

        self._fold_attention_weights(layer)
        device = embed_table.device
        enorm_weight = layer.enorm.weight.detach()
        token_table, hidden_rows = build_draft_token_table(
            weights.eh_proj,
            embed_table.detach(),
            enorm_weight,
            layer.enorm.variance_epsilon,
            tp_rank,
            tp_size,
            vocab_shard,
        )
        top_k = config.num_experts_per_tok
        expert_weights = torch.zeros((1, top_k), dtype=torch.bfloat16)
        expert_weights[0, 0] = 1
        expert_rows = torch.full((1, top_k), -1, dtype=torch.int32)
        expert_rows[0, 0] = 0
        l2_after_fused_a = [self._q_b_absorbed[layer.layer_idx], mla.v_b_proj, mla.o_proj.weight]
        self._draft = _DraftState(
            layer=layer,
            lm_head=lm_head,
            embed_table=embed_table.detach(),
            enorm_weight=enorm_weight,
            hnorm_weight=layer.hnorm.weight.detach(),
            window=(tp_rank * columns, (tp_rank + 1) * columns),
            pdl=get_env_enable_pdl(),
            l2_after_fused_a=[t for t in l2_after_fused_a if t.is_contiguous()],
            zero_residual=torch.zeros(
                (FUSED_A_MAX_TOKENS, hidden), dtype=torch.bfloat16, device=device
            ),
            token_table=token_table,
            hidden_rows=hidden_rows,
            w3_w1=weights.w3_w1,
            w2=weights.w2,
            top_k=top_k,
            routed_scaling_factor=config.routed_scaling_factor,
            expert_weights=expert_weights.to(device),
            expert_rows=expert_rows.to(device),
            topk_stash=torch.empty((1, self.index_topk), dtype=torch.int32, device=device),
            request_rows=torch.zeros(1, dtype=torch.int32, device=device),
            table_exchange=table_exchange,
            sample_exchange=sample_exchange,
            stream=torch.cuda.Stream(),
            selection_fork=torch.cuda.Event(),
            selection_ready=torch.cuda.Event(),
            half_start=torch.cuda.Event(),
            half_done=torch.cuda.Event(),
        )

    def _fast(self, rows: int) -> bool:
        """Whether every kernel of the step serves this many rows (decided once per shape)."""
        fast = self._fast_rows.get(rows)
        if fast is None:
            fast = (
                rows <= FUSED_A_MAX_TOKENS
                and rows <= FUSED_ROUTER_MAX_TOKENS
                and bool(
                    torch.ops.trtllm.glm_dense_mlp_small_m_supported(
                        rows, self.hidden_size, INTER_PER_RANK
                    )
                )
                and hop_supported(rows, self.hidden_size, self.mapping.tp_size)
            )
            self._fast_rows[rows] = fast
        return fast

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        core = self.core
        layers = core.layers[: core.num_hidden_layers]
        if inputs_embeds is None:
            inputs_embeds = fused_embedding(input_ids, core.embed_tokens.weight)
        residual = inputs_embeds
        hidden_states = layers[0].input_layernorm(inputs_embeds)
        if not self._fast(hidden_states.shape[0]):
            self._keys_sharded = False
            for layer in layers:
                hidden_states, residual = self._builtin_layer(
                    layer, position_ids, hidden_states, residual, attn_metadata
                )
            return hidden_states
        sharded = self._shard_indexer_keys(attn_metadata)
        self._keys_sharded = sharded
        for layer in layers:
            attn_out = self._attention(layer, position_ids, hidden_states, attn_metadata, sharded)
            hidden_states, residual = self._post_attention_hop(layer, attn_out, residual)
            if isinstance(layer.mlp, GatedMLP):
                mlp = layer.mlp
                # This rank's partial sum; the hop adds the 16 ranks'.
                partial = dense_mlp_small_m(
                    hidden_states, mlp.gate_up_proj.weight, mlp.down_proj.weight
                )
                hidden_states, residual = self._hop(layer.next_layer_layernorm, partial, residual)
            else:
                hidden_states, residual = self._moe(layer, hidden_states, residual)
        return hidden_states

    def logits(self, lm_head, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        """FP32 logits over the whole vocabulary of the rows, or None for a batch the kernels
        do not serve: this rank's vocabulary shard from the LM head, then the all-gather."""
        rows = hidden_states.shape[0]
        weight = lm_head.weight
        if not self._fast(rows) or weight.shape[0] * self.mapping.tp_size != lm_head.num_embeddings:
            return None
        shard = low_m_gemm(
            hidden_states.contiguous(), weight, hidden_states.new_empty(rows, weight.shape[0])
        )
        return mnnvl_allgather(
            shard,
            self._logits_workspace.comm_buffer,
            self._logits_workspace.buffer_flags,
            self.mapping.tp_size,
        )

    def drafts(self, rows: int) -> bool:
        """Whether this step's MTP draft steps run here: the draft state exists, this step's
        verify forward put the indexer on this rank's key shard (one request; draft step 0
        reuses it) and the kernels serve draft step 0's ``rows``."""
        return self._draft is not None and self._keys_sharded and self._fast(rows)

    def draft_first_step(
        self,
        attn_metadata,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        batch_indices: torch.Tensor,
        num_accepted: torch.Tensor,
        draft_len: int,
    ):
        """MTP draft step 0 over the verify rows, through the MTP layer.

        ``input_ids`` are the tokens the verify step accepted at every row, ``hidden_states``
        the target's output rows, ``num_accepted`` the accepted count of each request
        (``batch_indices``). Returns ``(hidden, gather_ids, pool_rows)``: the MTP layer's
        output (shared_head.norm applied) at each request's last accepted row; that row's index
        among the verify rows; and the selection the later steps attend to, as KV pool rows,
        or None where they attend to every position (a short context) or no step follows.
        """
        draft = self._draft
        layer = draft.layer
        rows = hidden_states.shape[0]
        col_lo, col_hi = draft.window
        eh_proj_input = glm_draft_glue_rmsnorm_cat(
            draft.embed_table,
            input_ids,
            hidden_states,
            draft.enorm_weight,
            draft.hnorm_weight,
            layer.enorm.variance_epsilon,
            col_lo,
            col_hi,
            draft.pdl,
        )
        partial = low_m_gemm(
            eh_proj_input, layer.eh_proj.weight, eh_proj_input.new_empty((rows, self.hidden_size))
        )
        # A zero residual: the residual output is eh_proj's output, the sum of the partials.
        hidden_states, residual = self._hop(
            layer.input_layernorm, partial, draft.zero_residual[:rows]
        )
        attn_out = self._attention(layer, position_ids, hidden_states, attn_metadata, True)
        pool_rows = None
        if draft_len > 1 and not attn_metadata.skip_indexer_for_gen_reqs:
            pool_rows = self._stash_selection(attn_metadata, num_accepted, draft_len + 1)
        hidden_states, residual = self._post_attention_hop(layer, attn_out, residual)
        gather_ids, hidden_states, residual = glm_gather_last_accepted_rows(
            hidden_states, batch_indices, num_accepted, draft_len + 1, 0, second=residual
        )
        return self._draft_moe(hidden_states, residual), gather_ids, pool_rows

    def draft_next_step(
        self,
        attn_metadata,
        eh_proj_output: Tuple[torch.Tensor, torch.Tensor],
        pool_rows: Optional[torch.Tensor],
        join_selection: bool,
    ) -> torch.Tensor:
        """An MTP draft step after the first, one row per request, through the MTP layer.

        ``eh_proj_output`` is the previous step's ``(input_layernorm(x), x)`` for this step's
        eh_proj output x. The layer runs no indexer. Its rows attend to ``pool_rows``, draft
        step 0's selection (``join_selection``, on the first such step, joins the side stream
        that made it); with a context short enough that the built-in metadata skips the
        indexer, to every position up to their own, the dense selection that metadata lists
        (refreshed for this step by the loop). Returns the MTP layer's output rows.
        """
        draft = self._draft
        layer = draft.layer
        hidden_states, residual = eh_proj_output
        if attn_metadata.skip_indexer_for_gen_reqs:
            positions = attn_metadata.topk_indices_buffer[: hidden_states.shape[0]]
            attn_out = self._attention(
                layer, None, hidden_states, attn_metadata, True, positions=positions
            )
        else:
            if join_selection:
                torch.cuda.current_stream().wait_event(draft.selection_ready)
            attn_out = self._attention(
                layer, None, hidden_states, attn_metadata, True, pool_rows=pool_rows
            )
        hidden_states, residual = self._post_attention_hop(layer, attn_out, residual)
        return self._draft_moe(hidden_states, residual)

    def draft_head(self, hidden_states: torch.Tensor, next_step: bool):
        """The draft LM head on this rank's vocabulary shard: ``(logits, hidden_half)``.

        With ``next_step`` the next step's eh_proj hidden half is enqueued first, on the aux
        stream: this rank's FP32 share of ``W_eh[:, H:] @ hnorm(hidden_states)``, hnorm inside
        the split-K GEMM (``draft_ehproj_table`` joins it); else ``hidden_half`` is None.
        """
        draft = self._draft
        layer = draft.layer
        rows = hidden_states.shape[0]
        hidden_half = None
        if next_step:
            hidden_half = torch.empty(
                (rows, draft.hidden_rows.shape[0]),
                dtype=torch.float32,
                device=hidden_states.device,
            )
            aux_stream = layer.mlp.aux_stream
            draft.half_start.record()
            with torch.cuda.stream(aux_stream):
                aux_stream.wait_event(draft.half_start)
                draft_hidden_half(
                    hidden_states,
                    draft.hidden_rows,
                    hidden_half,
                    draft.hnorm_weight,
                    layer.hnorm.variance_epsilon,
                )
                draft.half_done.record()
        weight = draft.lm_head.weight
        logits = low_m_gemm(hidden_states, weight, hidden_states.new_empty(rows, weight.shape[0]))
        return logits, hidden_half

    def draft_ehproj_table(
        self,
        logits: torch.Tensor,
        hidden_half: torch.Tensor,
        draft_tokens: torch.Tensor,
        step: int,
        positions: torch.Tensor,
        gather_ids: Optional[torch.Tensor],
        seq_lens: torch.Tensor,
        kv_lens: torch.Tensor,
        rewind_accepted: Optional[torch.Tensor],
    ):
        """The end of draft step ``step`` when another follows, in one kernel per rank.

        The ranks exchange their argmax with its token-table row and their share of the hidden
        half (joined here). Every rank writes the draft token into ``draft_tokens[:, step]``,
        the next step's eh_proj output x and ``input_layernorm(x)``, and its positions:
        ``positions`` (at ``gather_ids`` on draft step 0) plus one. On draft step 0
        ``seq_lens`` becomes 1 and ``kv_lens`` drops the rejected drafts
        (``rewind_accepted``); afterwards ``kv_lens`` grows by one. Returns
        ``((normed, x), positions)`` for the next step.
        """
        draft = self._draft
        norm = draft.layer.input_layernorm
        rows = logits.shape[0]
        normed = logits.new_empty((rows, self.hidden_size))
        residual = logits.new_empty((rows, self.hidden_size))
        next_positions = positions.new_empty((rows,))
        first = step == 0
        torch.cuda.current_stream().wait_event(draft.half_done)
        glm_draft_ehproj_table(
            logits,
            draft.table_exchange.buffer,
            draft.table_exchange.flags,
            self.mapping.tp_size,
            draft_tokens,
            step,
            hidden_half,
            draft.token_table,
            norm.weight,
            norm.variance_epsilon,
            normed,
            residual,
            positions,
            next_positions,
            gather_ids,
            seq_lens,
            first,
            kv_lens,
            2 if first else 1,
            rewind_accepted,
            draft_tokens.shape[1],
            None,
            [],
            None,
            None,
            None,
            None,
            None,
        )
        return (normed, residual), next_positions

    def draft_sample_glue(
        self,
        logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        step: int,
        seq_lens: torch.Tensor,
        kv_lens: torch.Tensor,
        rewind_accepted: Optional[torch.Tensor],
        accepted_tokens: torch.Tensor,
        batch_indices: torch.Tensor,
        num_accepted: torch.Tensor,
    ) -> torch.Tensor:
        """The end of the last draft step ``step``, in one kernel per rank.

        The ranks exchange their argmax; every rank writes the draft token into
        ``draft_tokens[:, step]``, the lengths as ``draft_ehproj_table`` does, and returns the
        next-new-tokens rows: each request's last accepted token, then its draft tokens.
        """
        draft = self._draft
        rows = logits.shape[0]
        draft_len = draft_tokens.shape[1]
        next_new_tokens = draft_tokens.new_empty((rows, draft_len + 1))
        first = step == 0
        glm_draft_sample_glue(
            logits,
            draft.sample_exchange.buffer,
            draft.sample_exchange.flags,
            self.mapping.tp_size,
            draft_tokens,
            step,
            # No eh_proj input window and no positions: no step follows.
            None,
            None,
            None,
            None,
            0.0,
            0,
            0,
            None,
            None,
            None,
            None,
            seq_lens,
            first,
            kv_lens,
            2 if first else 1,
            rewind_accepted,
            draft_len,
            None,
            [],
            None,
            None,
            None,
            None,
            None,
            accepted_tokens,
            batch_indices,
            num_accepted,
            next_new_tokens,
        )
        return next_new_tokens

    def _stash_selection(self, attn_metadata, num_accepted: torch.Tensor, next_n: int):
        """The selection the MTP draft steps after the first attend to, made on the side
        stream after draft step 0's attention: each request's top-k of its last accepted row
        (the MTP index share), as KV pool rows of the MTP layer. The first later step joins
        it (``selection_ready``)."""
        draft = self._draft
        manager = attn_metadata.kv_cache_manager
        requests = num_accepted.shape[0]
        tokens_per_block = manager.tokens_per_block
        local_layer = draft.layer.self_attn.mqa.get_local_layer_idx(attn_metadata)
        page_index_scale, layer_offset = manager.get_primary_pool_page_index_params(local_layer)
        main_stream = torch.cuda.current_stream()
        draft.selection_fork.record()
        with torch.cuda.stream(draft.stream):
            draft.stream.wait_event(draft.selection_fork)
            stashed = stash_last_accepted_rows(
                self._hist_buffers["topk"][: requests * next_n],
                num_accepted,
                draft.topk_stash[:requests],
                next_n,
            )
            pool_rows = convert_req_index_to_global(
                draft.request_rows[:requests],
                attn_metadata.block_table[:requests],
                stashed,
                tokens_per_block,
                self.index_topk,
                page_index_scale * tokens_per_block,
                layer_offset,
            )
            draft.selection_ready.record()
        # Allocated on the side stream and read on this one: outside graph capture the memory
        # must outlive this stream's reads (a captured graph keeps it).
        if not torch.cuda.is_current_stream_capturing():
            pool_rows.record_stream(main_stream)
        return pool_rows

    def _draft_moe(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """The MTP layer's BF16 MoE on one row and the post-MoE hop, which adds the routed and
        shared outputs (the routed row as the one slot of weight 1) and the residual and
        applies shared_head.norm: the normalized row."""
        draft = self._draft
        layer = draft.layer
        mlp = layer.mlp
        gate = mlp.gate

        def routed() -> torch.Tensor:
            router_logits = router_gemm(hidden_states, gate.weight)
            return glm_lowm_bf16_moe(
                hidden_states,
                router_logits,
                gate.e_score_correction_bias,
                draft.w3_w1,
                draft.w2,
                draft.top_k,
                draft.routed_scaling_factor,
            )

        routed_output, shared_output = maybe_execute_in_parallel(
            routed,
            lambda: self._shared_expert(mlp, hidden_states),
            mlp.event_dict[EventType.Main],
            mlp.event_dict[EventType.MoeShared],
            mlp.aux_stream,
        )
        norm = layer.shared_head.norm
        comm_buffer, buffer_flags = hop_workspace(
            self.mapping, hidden_states.shape[0], self.hidden_size
        )
        normed, _ = mnnvl_moe_finalize_hop(
            routed_output,
            draft.expert_weights,
            draft.expert_rows,
            shared_output,
            residual,
            norm.weight,
            norm.variance_epsilon,
            comm_buffer,
            buffer_flags,
        )
        return normed

    def _shard_indexer_keys(self, attn_metadata) -> bool:
        """Point this rank's indexer at its 1/16 shard of the request's keys for this step:
        the shard's length and block table, and the logits kernel's schedule over it.

        Applies to one generation request whose shard fits the top-k kernel; returns whether it
        did, and otherwise every indexer layer scores the whole context with the built-in
        indexer."""
        manager = attn_metadata.kv_cache_manager
        if self._hist_workspace is None or attn_metadata.num_seqs != 1:
            return False
        capacity = shard_capacity(manager.max_seq_len, self.mapping.tp_size)
        if capacity > MAX_SHARD:
            return False
        device = self._unit_scale.device
        tokens_per_block = manager.tokens_per_block
        buffers = self._hist_buffers
        # The model outlives the profiling KV manager. Compatible buffers keep
        # their addresses for captured graphs; a larger or different layout needs new storage.
        if (
            buffers is None
            or buffers["capacity"] < capacity
            or buffers["tokens_per_block"] != tokens_per_block
            or buffers["logits"].device != device
        ):
            rows = FUSED_A_MAX_TOKENS
            sms = num_sms(device.index)
            self._hist_buffers = dict(
                capacity=capacity,
                tokens_per_block=tokens_per_block,
                shard_ctx=torch.zeros(1, dtype=torch.int32, device=device),
                shard_block_table=torch.zeros(
                    (1, -(-capacity // tokens_per_block)), dtype=torch.int32, device=device
                ),
                schedule=torch.zeros((sms + 1, 2), dtype=torch.int32, device=device),
                logits=torch.empty(
                    (rows, -(-capacity // 256) * 256), dtype=torch.float32, device=device
                ),
                topk=torch.empty((rows, TOPK), dtype=torch.int32, device=device),
                k_fp8=torch.empty((rows, 128), dtype=torch.float8_e4m3fn, device=device),
                k_scale=torch.empty((rows, 1), dtype=torch.float32, device=device),
                sms=sms,
            )
        buffers = self._hist_buffers
        hist_shard_meta(
            attn_metadata.gen_indexer_kv_lens_cuda_runtime,
            attn_metadata.indexer_k_cache_block_offsets[0],
            buffers["shard_ctx"],
            buffers["shard_block_table"],
            self.mapping.tp_size,
            self.mapping.tp_rank,
            manager.tokens_per_block,
        )
        paged_mqa_logits_schedule(
            buffers["shard_ctx"], manager.tokens_per_block, buffers["sms"], buffers["schedule"]
        )
        return True

    def _attention(
        self,
        layer,
        position_ids,
        hidden_states,
        attn_metadata,
        sharded,
        pool_rows: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ):
        """MLA with the DSA indexer over the generation rows: the attention output ahead of
        o_proj, [rows, heads * v_head_dim].

        On an MTP draft step after the first (``sharded``) the rows attend to a given
        selection and the indexer does not run: ``pool_rows``, KV pool rows, or
        ``positions``, sequence positions the flash decode maps to pool rows itself."""
        mla = layer.self_attn
        mqa = mla.mqa
        indexer = mqa.indexer
        manager = attn_metadata.kv_cache_manager
        num_tokens = attn_metadata.num_tokens
        seq_len_q = num_tokens // attn_metadata.num_seqs
        head_dim = mla.kv_lora_rank + mla.qk_rope_head_dim
        mqa._ensure_rope_table_size(attn_metadata.max_seq_len)
        cos_sin = mqa.rotary_cos_sin
        kv_lens = attn_metadata.kv_lens_cuda_runtime
        local_layer = mqa.get_local_layer_idx(attn_metadata)
        # Asked per step: the executor re-creates the cache manager after its memory profiling.
        kv_pool = manager.get_unique_primary_pool()
        reuses_selection = pool_rows is not None or positions is not None
        own_indexer = sharded and indexer is not None and not reuses_selection
        # A layer that attends to the selection of the indexer layer before it gathers its old KV
        # rows before the flash decode's wait.
        early_gather = sharded and indexer is None and not reuses_selection

        weight = (
            self._fused_a_indexer_weight[layer.layer_idx]
            if own_indexer
            else mla.kv_a_proj_with_mqa.weight
        )
        if reuses_selection:
            l2_prefetch = self._draft.l2_after_fused_a
        elif early_gather:
            # Also the selection the flash decode reads first.
            l2_prefetch = self._l2_after_fused_a[layer.layer_idx] + [
                self._hist_buffers["topk"][:num_tokens]
            ]
        else:
            l2_prefetch = self._l2_after_fused_a[layer.layer_idx]
        kv_a = fused_a_gemm(hidden_states, weight, l2_prefetch)
        kv_a = kv_a[:num_tokens]
        q_a = kv_a[:, : mla.q_lora_rank]
        latent_end = mla.q_lora_rank + head_dim
        quant_q = hidden_states.new_empty(
            (num_tokens, mla.num_heads_tp * head_dim), dtype=torch.uint8
        )

        def q_b(plan="regular"):
            q_b_fp8_rope(
                q_a,
                self._q_b_absorbed[layer.layer_idx],
                quant_q,
                kv_lens,
                cos_sin.view(-1, mla.qk_rope_head_dim, 2),
                self._unit_scale,
                mla.q_a_layernorm.variance_epsilon,
                head_dim,
                mla.qk_rope_head_dim,
                seq_len_q,
                plan,
            )

        def kv_half(trigger_at_entry=False):
            mla_kv_half_append(
                kv_a[:, mla.q_lora_rank : latent_end],
                mla.kv_a_layernorm.weight,
                mla.kv_a_layernorm.variance_epsilon,
                cos_sin,
                kv_lens,
                0,
                seq_len_q,
                attn_metadata.kv_cache_block_offsets,
                manager.kv_cache_pool_pointers,
                manager.kv_cache_pool_mapping,
                local_layer,
                manager.tokens_per_block,
                attn_metadata.max_seq_len,
                self._bmm1_scale,
                self._bmm2_scale,
                mla.softmax_scale,
                trigger_at_entry,
            )

        output = hidden_states.new_empty(
            (hidden_states.shape[0], mla.num_heads_tp * mla.v_head_dim)
        )
        q_heads = quant_q.view(num_tokens, mla.num_heads_tp, head_dim)
        if sharded:
            if own_indexer:

                def select():
                    self._select_rows(
                        indexer, kv_a, latent_end, q_a, position_ids, attn_metadata, num_tokens, mla
                    )

                def kv_half_then_q_b():
                    kv_half()
                    # Queued after the top-k: 18 CTAs leave the SMs to the indexer beside it.
                    q_b("beside_topk")

                # The indexer on this stream; the KV half and q_b beside it on the aux stream.
                maybe_execute_in_parallel(
                    select, kv_half_then_q_b, mla.ln_events[0], mla.ln_events[1], mla.aux_stream
                )
            else:
                # With the early gather, q_b at split-K 4 leaves room for the flash decode, which
                # the KV half lets launch at its entry.
                maybe_execute_in_parallel(
                    lambda: q_b("early_gather" if early_gather else "regular"),
                    lambda: kv_half(early_gather),
                    mla.ln_events[0],
                    mla.ln_events[1],
                    mla.aux_stream,
                )
            if pool_rows is not None:
                sparse_mla_flash_decode(
                    q_heads,
                    kv_pool,
                    pool_rows,
                    kv_lens,
                    self._bmm1_scale,
                    self._bmm2_scale,
                    mla.v_b_proj,
                    output,
                    seq_len_q,
                    self.index_topk,
                )
                return output
            page_index_scale, layer_offset = manager.get_primary_pool_page_index_params(local_layer)
            sparse_mla_flash_decode(
                q_heads,
                kv_pool,
                self._hist_buffers["topk"][:num_tokens] if positions is None else positions,
                kv_lens,
                self._bmm1_scale,
                self._bmm2_scale,
                mla.v_b_proj,
                output,
                seq_len_q,
                self.index_topk,
                attn_metadata.block_table[: attn_metadata.num_seqs],
                manager.tokens_per_block,
                page_index_scale,
                layer_offset,
                early_gather,
            )
            return output

        # The built-in indexer over the whole context, then the selection as cache rows.
        maybe_execute_in_parallel(q_b, kv_half, mla.ln_events[0], mla.ln_events[1], mla.aux_stream)
        indexer_intermediates = None
        if indexer is not None:
            qr = mla.q_a_layernorm(q_a)
            indexer_intermediates = [
                t[:num_tokens]
                for t in indexer.pre_indexer_proj(
                    qr, hidden_states[:num_tokens], position_ids[..., :num_tokens]
                )
            ]
            indexer._update_k_cache(
                indexer_intermediates[1], indexer_intermediates[2], attn_metadata
            )
        indices, _ = mqa.sparse_attn_predict(
            hidden_states[:num_tokens],
            None,
            attn_metadata,
            AttentionForwardArgs(
                attention_input_type=AttentionInputType.generation_only,
                sparse_backend_args=DSABackendForwardArgs(
                    indexer_intermediates=indexer_intermediates
                ),
            ),
        )
        sparse_mla_flash_decode(
            q_heads,
            kv_pool,
            indices,
            kv_lens,
            self._bmm1_scale,
            self._bmm2_scale,
            mla.v_b_proj,
            output,
            seq_len_q,
            self.index_topk,
        )
        if indexer is not None:
            indexer.maybe_join_prev_topk_copy()
        return output

    def _select_rows(self, indexer, kv_a, latent_end, q_a, position_ids, attn_metadata, rows, mla):
        """The indexer of a layer that runs one, on this rank's key shard: writes the exact
        top-2048 cache positions of every row into the shared selection buffer."""
        buffers = self._hist_buffers
        manager = attn_metadata.kv_cache_manager
        rope = indexer.rotary_emb
        cos_sin = rope.rotary_cos_sin.view(rope.max_positions, -1)
        interleave = not rope.is_neox
        positions = position_ids.view(-1)[:rows]
        index_k = kv_a[:, latent_end : latent_end + indexer.head_dim]
        index_weights = kv_a[:, latent_end + indexer.head_dim :]
        k_cache = manager.get_indexer_k_cache_buffers(indexer.layer_idx)

        # The K tail on its own stream, beside the Q tail: k_norm, RoPE, FP8 and the K cache.
        self._k_tail_fork.record()
        with torch.cuda.stream(self._k_tail_stream):
            self._k_tail_stream.wait_event(self._k_tail_fork)
            indexer_k_tail(
                index_k,
                positions,
                cos_sin,
                indexer.k_norm.weight,
                indexer.k_norm.bias,
                indexer.k_norm.variance_epsilon,
                indexer.rope_dim,
                interleave,
                indexer.scale_fmt == "ue8m0",
                buffers["k_fp8"][:rows],
                buffers["k_scale"][:rows],
                k_cache,
                attn_metadata.slot_mapping_fp8[:rows],
                attn_metadata.slot_mapping_scale[:rows],
                rows,
                True,
                True,
                False,
            )
            self._k_tail_done.record()

        q_fp8 = q_a.new_empty((rows, indexer.n_heads, indexer.head_dim), dtype=torch.float8_e4m3fn)
        q_scale = q_a.new_empty((rows, indexer.n_heads, 1), dtype=torch.float32)
        weights = q_a.new_empty((rows, indexer.n_heads), dtype=torch.float32)
        indexer_q_tail(
            q_a,
            self._wq_b_folded[indexer.layer_idx],
            positions,
            cos_sin,
            index_weights,
            q_fp8,
            q_scale,
            weights,
            mla.q_a_layernorm.variance_epsilon,
            indexer.n_heads,
            indexer.head_dim,
            indexer.rope_dim,
            interleave,
            indexer.scale_fmt == "ue8m0",
            indexer.weight_scale_factor,
        )
        torch.cuda.current_stream().wait_event(self._k_tail_done)
        logits = fp8_paged_mqa_logits(
            q_fp8.view(1, rows, indexer.n_heads, indexer.head_dim),
            k_cache,
            weights,
            buffers["shard_ctx"],
            buffers["shard_block_table"],
            buffers["schedule"],
            buffers["logits"][:rows],
            buffers["capacity"],
            True,
        )
        hist_topk(
            logits,
            attn_metadata.gen_indexer_kv_lens_cuda_runtime,
            buffers["topk"][:rows],
            self._hist_workspace.comm_buffer,
            self._hist_workspace.state,
            rows,
            self.mapping.tp_size,
        )

    def _post_attention_hop(self, layer, attn_out: torch.Tensor, residual: torch.Tensor):
        """o_proj published into the all-reduce workspace, then the hop with the residual add
        and post_attention_layernorm: (normed, residual)."""
        rows = attn_out.shape[0]
        lamport, flags = publish_target(self.mapping, rows, self.hidden_size)
        comm_buffer, buffer_flags = hop_workspace(self.mapping, rows, self.hidden_size)
        shard = attn_out.new_empty((rows, self.hidden_size))
        o_proj_publish(
            attn_out,
            layer.self_attn.o_proj.weight,
            shard,
            lamport,
            flags,
            self.mapping.tp_size,
            self.mapping.tp_rank,
        )
        norm = layer.post_attention_layernorm
        return mnnvl_hop(
            shard, residual, norm.weight, norm.variance_epsilon, comm_buffer, buffer_flags, True
        )

    def _hop(self, norm, partial: torch.Tensor, residual: torch.Tensor):
        """The hop of a rank-partial sum with the residual add and `norm`: (normed, residual)."""
        comm_buffer, buffer_flags = hop_workspace(self.mapping, partial.shape[0], self.hidden_size)
        return mnnvl_hop(
            partial, residual, norm.weight, norm.variance_epsilon, comm_buffer, buffer_flags, False
        )

    def _moe(self, layer, hidden_states: torch.Tensor, residual: torch.Tensor):
        """The MoE block and the post-MoE hop that finalizes it: (normed, residual)."""
        mlp = layer.mlp
        backend = getattr(mlp.experts, "backend", mlp.experts)
        rows = hidden_states.shape[0]

        self._quant_fork.record()
        with torch.cuda.stream(self._quant_stream):
            self._quant_stream.wait_event(self._quant_fork)
            x_fp4, x_sf = fp4_quantize(
                hidden_states, backend.fc31_input_scale, backend.scaling_vector_size, False, False
            )
            self._quant_ready.record()
        routing = extract_routing_params(backend.routing_method)

        def routed():
            # Unfinalized: [FC2 rows, expert weights, expanded index -> FC2 row].
            return routed_experts(
                self._router_logits[:rows],
                routing.routing_bias,
                x_fp4,
                x_sf.flatten().view(torch.float8_e4m3fn),
                backend.w3_w1_weight,
                backend.w3_w1_weight_scale.view(torch.float8_e4m3fn),
                None,
                backend.act_alpha,
                backend.act_beta,
                backend.act_clamp,
                backend.w2_weight,
                backend.w2_weight_scale.view(torch.float8_e4m3fn),
                None,
                get_data_or_none(backend, "fc31_scale_c"),
                get_data_or_none(backend, "fc31_alpha"),
                get_data_or_none(backend, "fc2_alpha"),
                backend.num_slots,
                routing.top_k,
                routing.n_group,
                routing.topk_group,
                backend.w3_w1_weight.shape[-2] // 2,
                backend.slot_start,
                backend.expert_size_per_partition,
                routing.routed_scaling_factor,
                backend.routing_method.routing_method_type,
                False,
                to_trtllm_gen_act_type(backend.activation_type),
                None,
                None,
                None,
                backend.max_num_tokens,
                False,
                self._quant_ready.cuda_event,
                hidden_states,
                mlp.gate.weight,
                self._router_workspace[layer.layer_idx],
                True,
            )

        (fc2_output, expert_weights, permuted_rows), shared_output = maybe_execute_in_parallel(
            routed,
            lambda: self._shared_expert(mlp, hidden_states),
            mlp.event_dict[EventType.Main],
            mlp.event_dict[EventType.MoeShared],
            mlp.aux_stream,
        )
        norm = layer.next_layer_layernorm
        comm_buffer, buffer_flags = hop_workspace(self.mapping, rows, self.hidden_size)
        return mnnvl_moe_finalize_hop(
            fc2_output,
            expert_weights,
            permuted_rows,
            shared_output,
            residual,
            norm.weight,
            norm.variance_epsilon,
            comm_buffer,
            buffer_flags,
        )

    def _shared_expert(self, mlp, hidden_states: torch.Tensor) -> torch.Tensor:
        """This rank's partial sum of the shared expert: two split-K GEMMs around SwiGLU."""
        expert = mlp.shared_experts
        rows = hidden_states.shape[0]
        gate_up = expert.gate_up_proj.weight
        h = low_m_gemm(hidden_states, gate_up, hidden_states.new_empty(rows, gate_up.shape[0]))
        h = swiglu(h)
        return low_m_gemm(
            h, expert.down_proj.weight, hidden_states.new_empty(rows, self.hidden_size)
        )

    @staticmethod
    def _builtin_layer(layer, position_ids, hidden_states, residual, attn_metadata):
        """One decoder layer on the built-in modules, for a batch the kernels do not serve."""
        dense = isinstance(layer.mlp, GatedMLP)
        hidden_states = DecodeTarget._attention_builtin(
            layer,
            position_ids,
            hidden_states,
            attn_metadata,
            AllReduceParams(enable_allreduce=dense),
        )
        if dense:
            hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
            hidden_states = layer.mlp(
                hidden_states, final_all_reduce_params=AllReduceParams(enable_allreduce=False)
            )
        else:
            hidden_states, residual = layer.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=layer.post_attention_layernorm.weight,
                    eps=layer.post_attention_layernorm.variance_epsilon,
                    trigger_completion_at_end=False,
                ),
            )
            hidden_states = layer.mlp(
                hidden_states,
                None,
                all_rank_num_tokens=attn_metadata.all_rank_num_tokens,
                final_all_reduce_params=AllReduceParams(enable_allreduce=False),
                do_finalize=True,
            )
        return layer.allreduce(
            hidden_states,
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=residual,
                norm_weight=layer.next_layer_layernorm.weight,
                eps=layer.next_layer_layernorm.variance_epsilon,
                trigger_completion_at_end=dense,
            ),
        )

    @staticmethod
    def _attention_builtin(layer, position_ids, hidden_states, attn_metadata, all_reduce_params):
        """The built-in modules' MLA with the DSA indexer, for a batch the kernels do not serve."""
        mla = layer.self_attn
        num_tokens = attn_metadata.num_tokens
        kv_a = mla.kv_a_proj_with_mqa(hidden_states)
        q, compressed_kv, k_pe = kv_a.split(
            [mla.q_lora_rank, mla.kv_lora_rank, mla.qk_rope_head_dim], -1
        )
        (q, qr), compressed_kv = maybe_execute_in_parallel(
            lambda: mla._q_a_layernorm_maybe_fused(q, return_norm_out=True),
            lambda: mla.kv_a_layernorm(compressed_kv),
            mla.ln_events[0],
            mla.ln_events[1],
            mla.aux_stream,
        )
        latent_cache = torch.concat([compressed_kv, k_pe], dim=-1)
        q = mla.q_b_proj(q)
        indexer = mla.mqa.indexer
        indexer_intermediates = None
        if indexer is not None:
            indexer_intermediates = [
                t[:num_tokens] for t in indexer.pre_indexer_proj(qr, hidden_states, position_ids)
            ]
            indexer._update_k_cache(
                indexer_intermediates[1], indexer_intermediates[2], attn_metadata
            )
        output = hidden_states.new_empty([hidden_states.shape[0], mla.attention_output_hidden_size])
        forward_generation_sparse_attn(
            mla,
            q[:num_tokens],
            compressed_kv[:num_tokens],
            k_pe[:num_tokens],
            attn_metadata,
            output[:num_tokens],
            position_ids=position_ids[..., :num_tokens],
            latent_cache=latent_cache[:num_tokens],
            indexer_intermediates=indexer_intermediates,
        )
        if indexer is not None:
            indexer.maybe_join_prev_topk_copy()
        return mla.o_proj(output, all_reduce_params=all_reduce_params, layer_idx=mla.layer_idx)


class Glm52Nvfp4Sm107Tp16Core(Glm52Model):
    """The decoder stack, with the per-step choice between the two targets."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        super().__init__(model_config)
        self._prefill: Optional[PrefillTarget] = None
        self._decode: Optional[DecodeTarget] = None

    @torch.no_grad()
    def post_load_weights(self) -> None:
        """Build the targets once the weights are final. Reached more than once
        (the shell and the loader's module walk), so it only builds once."""
        if self._decode is not None:
            return
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

    def _select_target(self, attn_metadata: AttentionMetadata) -> "PrefillTarget | DecodeTarget":
        if self._decode is None:
            self.post_load_weights()
        return self._prefill if attn_metadata.num_contexts > 0 else self._decode

    def decode_target(self, attn_metadata: AttentionMetadata) -> Optional[DecodeTarget]:
        """The decode target when this step runs it, else None."""
        target = self._select_target(attn_metadata)
        return target if target is self._decode else None

    def build_draft(self, layer, lm_head, weights: _weights.DraftWeights) -> None:
        """Build the decode target's MTP draft-step state (``DecodeTarget.build_draft``)."""
        if self._decode is None:
            self.post_load_weights()
        self._decode.build_draft(layer, lm_head, weights)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._select_target(attn_metadata).forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )


class MTPDraftWorker(MTPEagleWorker):
    """The one-model MTP-Eagle worker: the draft loop of a one-request generation step runs
    on the decode target's kernels (``DecodeTarget``'s draft steps); every other step runs the
    built-in loop.

    The loop keeps the built-in loop's attention-metadata handling: the MTP index-share flags,
    the switch to one row per request after draft step 0, and the cleanup. The draft-step
    kernels advance the sequence and KV lengths in place, and the draft tokens and the
    next-new-tokens rows come from them.
    """

    def __init__(
        self,
        spec_config,
        model_config: ModelConfig,
        use_separate_draft_kv_cache: bool,
        *,
        mapping,
        decode_target,
    ) -> None:
        super().__init__(spec_config, model_config, use_separate_draft_kv_cache, mapping=mapping)
        # The core's per-step lookup of the decode target (a bound method rather than the
        # core itself, which belongs to the shell and is no submodule of this worker).
        self._decode_target = decode_target
        # The draft tokens of the last draft loop on the decode target and the next-new-tokens
        # rows its last step wrote, for _prepare_next_new_tokens.
        self._drafted_rows: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def prepare_1st_drafter_inputs(
        self,
        input_ids,
        position_ids,
        hidden_states,
        accepted_tokens,
        attn_metadata,
        spec_metadata,
        draft_model,
    ):
        """Without context requests draft step 0's input ids are the accepted-token rows as
        they are, flattened, rather than a copy behind an empty context part."""
        if not self.is_mtp_eagle or attn_metadata.num_contexts > 0:
            return super().prepare_1st_drafter_inputs(
                input_ids,
                position_ids,
                hidden_states,
                accepted_tokens,
                attn_metadata,
                spec_metadata,
                draft_model,
            )
        return {
            "input_ids": accepted_tokens[:, : spec_metadata.runtime_draft_len + 1].flatten(),
            "position_ids": position_ids,
            "hidden_states": hidden_states,
            "attn_metadata": attn_metadata,
            "spec_metadata": spec_metadata,
        }

    def _prepare_next_new_tokens(
        self,
        accepted_tokens,
        next_draft_tokens,
        batch_indices_cuda,
        batch_size,
        num_accepted_tokens,
    ):
        """The rows the last draft-step kernel wrote when these draft tokens are that loop's."""
        drafted, self._drafted_rows = self._drafted_rows, None
        if drafted is not None and drafted[0] is next_draft_tokens:
            return drafted[1]
        return super()._prepare_next_new_tokens(
            accepted_tokens,
            next_draft_tokens,
            batch_indices_cuda,
            batch_size,
            num_accepted_tokens,
        )

    def _sample_and_accept_draft_tokens_base(
        self,
        logits,
        draft_tokens,
        num_contexts,
        batch_size,
        spec_metadata,
    ):
        """Strict acceptance and the forced accepted-token count in one kernel
        (``glm_verify_head_accept``) for a step of generation requests only, with the built-in
        target-token sampling; ragged verification and context requests take the built-in
        chain. The forced count follows ``_apply_force_accepted_tokens``: none during eager
        CUDA-graph warm-up, the integer part plus one, and with a fractional part the same pool
        draw and counter step."""
        if spec_metadata is None or spec_metadata.is_ragged_verify or num_contexts != 0:
            return super()._sample_and_accept_draft_tokens_base(
                logits, draft_tokens, num_contexts, batch_size, spec_metadata
            )
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        tokens_per_request = draft_tokens.shape[-1] + 1
        target_tokens = self._sample_tokens_for_batch(logits, spec_metadata, 0, batch_size)
        mode, base, frac = 0, 0, 0.0
        warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        if self.force_num_accepted_tokens != 0.0 and not warmup:
            int_part = int(self.force_num_accepted_tokens)
            frac = self.force_num_accepted_tokens - int_part
            base = min(int_part + 1, tokens_per_request)
            mode = 2 if frac > 0.0 and base < tokens_per_request else 1
        pool = counter = None
        if mode == 2:
            self._ensure_force_accept_rng_state(logits.device)
            pool, counter = self._force_accept_rng_pool, self._force_accept_rng_counter
        accepted_tokens = torch.empty(
            (batch_size, tokens_per_request), dtype=torch.int, device=logits.device
        )
        num_accepted_tokens = glm_verify_head_accept(
            target_tokens,
            draft_tokens,
            0,
            accepted_tokens,
            pool,
            counter,
            mode,
            base,
            tokens_per_request,
            frac,
            _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
            _FORCE_ACCEPT_RNG_SLOT_STRIDE,
            _FORCE_ACCEPT_RNG_POOL_SIZE,
        )
        return accepted_tokens, num_accepted_tokens

    def _forward_linear_draft_loop(
        self,
        inputs,
        attn_metadata,
        spec_metadata,
        draft_model,
        draft_kv_cache_manager,
        num_contexts,
        batch_size,
        num_accepted_tokens,
        original_all_rank_num_tokens,
        accepted_tokens=None,
        pre_draft_seq_lens=None,
    ):
        self._drafted_rows = None
        decode = self._drafting_target(
            inputs,
            attn_metadata,
            spec_metadata,
            draft_kv_cache_manager,
            num_contexts,
            batch_size,
            num_accepted_tokens,
        )
        if decode is None:
            return super()._forward_linear_draft_loop(
                inputs,
                attn_metadata,
                spec_metadata,
                draft_model,
                draft_kv_cache_manager,
                num_contexts,
                batch_size,
                num_accepted_tokens,
                original_all_rank_num_tokens,
                accepted_tokens,
                pre_draft_seq_lens,
            )
        draft_tokens = self._draft_on_target(
            decode,
            inputs,
            attn_metadata,
            spec_metadata,
            draft_kv_cache_manager,
            batch_size,
            num_accepted_tokens,
        )
        return draft_tokens, None

    def _drafting_target(
        self,
        inputs,
        attn_metadata,
        spec_metadata,
        draft_kv_cache_manager,
        num_contexts,
        batch_size,
        num_accepted_tokens,
    ) -> Optional[DecodeTarget]:
        """The decode target when its kernels draft this step, else None.

        They draft a generation step of one request whose draft layer shares the target's KV
        pool, through the configured draft length with no draft-length schedule (the draft
        steps after the first skip the MTP layer's indexer K write: the next step's draft
        step 0 writes those positions before it reads them), greedily (argmax drafts), with
        nothing that rewrites the drafts (guided decoding, SA, a draft vocabulary map), and
        when the decode target's kernels serve the rows.
        """
        draft_len = spec_metadata.runtime_draft_len
        rows = batch_size * (draft_len + 1)
        input_ids = inputs["input_ids"]
        position_ids = inputs["position_ids"]
        if not (
            self.is_mtp_eagle
            and num_contexts == 0
            and batch_size == 1
            and draft_kv_cache_manager is None
            and draft_len == self.max_draft_len
            and getattr(self.spec_config, "draft_len_schedule", None) is None
            and self.guided_decoder is None
            and self.sa_enhancer is None
            and self._d2t is None
            and not spec_metadata.wants_advanced_draft_sampling
            and not spec_metadata.is_ragged_verify
            and attn_metadata.num_tokens == rows
            and inputs["hidden_states"].shape[0] == rows
            and input_ids.dtype == torch.int32
            and tuple(input_ids.shape) == (rows,)
            and input_ids.is_contiguous()
            and position_ids.dtype in (torch.int32, torch.int64)
            and tuple(position_ids.shape) == (rows,)
            and position_ids.is_contiguous()
            and num_accepted_tokens.dtype == torch.int32
            and num_accepted_tokens.is_contiguous()
            and spec_metadata.batch_indices_cuda.dtype == torch.int32
        ):
            return None
        decode = self._decode_target(attn_metadata)
        if decode is None or not decode.drafts(rows):
            return None
        return decode

    def _draft_on_target(
        self,
        decode: DecodeTarget,
        inputs,
        attn_metadata,
        spec_metadata,
        draft_kv_cache_manager,
        batch_size,
        num_accepted_tokens,
    ) -> torch.Tensor:
        """The draft loop on the decode target's kernels: the draft tokens [batch, draft_len].

        Each step's last kernel writes the next step's sequence and KV lengths in place. The
        steps after the first run no indexer and read nothing else of the DSA decode metadata,
        so the built-in loop's per-step rebuild of it (``update_for_spec_dec``) is left out;
        with a context short enough that the metadata skips the indexer, only the next step's
        dense selection is refreshed from the new lengths. The host side of draft step 0's
        switch to one row per request comes before that.
        """
        draft_len = spec_metadata.runtime_draft_len
        # The accepted tokens' rows: draft step 0's input ids (prepare_1st_drafter_inputs).
        accepted_tokens = inputs["input_ids"].view(batch_size, draft_len + 1)
        batch_indices = spec_metadata.batch_indices_cuda[:batch_size]
        num_accepted = num_accepted_tokens[:batch_size]
        draft_tokens = accepted_tokens.new_empty((batch_size, draft_len))
        positions = inputs["position_ids"]
        eh_proj_output = gather_ids = pool_rows = next_new_tokens = None
        uses_mtp_index_share = isinstance(attn_metadata, MTPIndexShareMetadata)
        with contextlib.ExitStack() as draft_scope:
            if uses_mtp_index_share:
                attn_metadata.set_in_mtp_draft_loop(True)
                attn_metadata.set_mtp_num_accepted(num_accepted_tokens)
                draft_scope.callback(_reset_mtp_index_share, attn_metadata)
            attn_metadata = draft_scope.enter_context(
                self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager)
            )
            # The draft loop's copy of the sequence lengths (prepare_for_spec_dec) and the KV
            # lengths, which the draft-step kernels advance.
            seq_lens = attn_metadata.seq_lens_cuda[:batch_size]
            kv_lens = attn_metadata.kv_lens_cuda[:batch_size]
            for i in range(draft_len):
                if uses_mtp_index_share:
                    attn_metadata.set_skip_topk(i > 0)
                if i == 0:
                    hidden_states, gather_ids, pool_rows = decode.draft_first_step(
                        attn_metadata,
                        inputs["input_ids"],
                        positions,
                        inputs["hidden_states"],
                        batch_indices,
                        num_accepted,
                        draft_len,
                    )
                else:
                    hidden_states = decode.draft_next_step(
                        attn_metadata, eh_proj_output, pool_rows, join_selection=i == 1
                    )
                last = i == draft_len - 1
                logits, hidden_half = decode.draft_head(hidden_states, next_step=not last)
                rewind_accepted = num_accepted if i == 0 else None
                if i == 0:
                    # One row per request from here on, on the host (no context requests to
                    # turn into generation ones); the kernel below fills seq_lens.
                    attn_metadata._seq_lens[:batch_size].fill_(1)
                    attn_metadata.on_update()
                if last:
                    next_new_tokens = decode.draft_sample_glue(
                        logits,
                        draft_tokens,
                        i,
                        seq_lens,
                        kv_lens,
                        rewind_accepted,
                        accepted_tokens,
                        batch_indices,
                        num_accepted,
                    )
                else:
                    eh_proj_output, positions = decode.draft_ehproj_table(
                        logits,
                        hidden_half,
                        draft_tokens,
                        i,
                        positions,
                        gather_ids if i == 0 else None,
                        seq_lens,
                        kv_lens,
                        rewind_accepted,
                    )
                if not last and attn_metadata.skip_indexer_for_gen_reqs:
                    attn_metadata.prepare_dense_topk_indices(
                        attn_metadata.kv_lens_cuda, device=True
                    )
                if i == 0:
                    attn_metadata.use_spec_decoding = False
        self._drafted_rows = (draft_tokens, next_new_tokens)
        return draft_tokens


class ModelingV2Glm52Nvfp4Sm107Tp16(
    Glm52SpecDecOneEngineForCausalLM[Glm52Nvfp4Sm107Tp16Core, PretrainedConfig]
):
    """GLM-5.2-NVFP4, sm107, tp16."""

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config: Any = None) -> Literal["V2"]:
        """The KV cache manager the context server's built-in implementation prefers."""
        return "V2"

    @classmethod
    def get_preferred_transceiver_runtime(
        cls, pretrained_config: Any = None
    ) -> Optional[Literal["CPP", "PYTHON"]]:
        """The transceiver runtime for the masked DSA indexer K-cache transfer."""
        return "PYTHON"

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        mapping = model_config.mapping
        assert torch.cuda.get_device_capability() == _COMPUTE_CAPABILITY, (
            f"this target runs on sm107, not {torch.cuda.get_device_capability()}"
        )
        assert (
            mapping.world_size == _TP
            and mapping.tp_size == _TP
            and mapping.moe_ep_size == 1
            and not mapping.enable_attention_dp
            and mapping.pp_size == 1
        ), f"this target is tp16 without attention DP or expert parallelism, not {mapping}"
        model_config = _weights.normalize_model_config(model_config)
        super().__init__(model=Glm52Nvfp4Sm107Tp16Core(model_config), model_config=model_config)
        if (
            model_config.spec_config is not None
            and model_config.spec_config.spec_dec_mode.is_mtp_one_model()
        ):
            self.num_hidden_layers = self.config.num_hidden_layers
            self.model.layers.extend(self.draft_model.mtp_layers)
        if type(self.spec_worker) is MTPEagleWorker:
            # The target's worker in place of the built-in one, with the same draft KV cache
            # choice and draft model.
            worker = MTPDraftWorker(
                self.spec_config,
                model_config,
                self.use_separate_draft_kv_cache,
                mapping=model_config.mapping,
                decode_target=self.model.decode_target,
            )
            worker.set_draft_model(self.draft_model)
            builtin = self.spec_worker
            self.epilogue[:] = [worker if m is builtin else m for m in self.epilogue]
            self.spec_worker = worker
        # The MTP layer's checkpoint tensors the draft steps derive operands from, held from
        # load_weights to post_load_weights.
        self._draft_weights: Optional[_weights.DraftWeights] = None

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        resource_manager=None,
        **kwargs,
    ) -> torch.Tensor:
        decode = self.model.decode_target(attn_metadata)
        if decode is None or self.spec_worker is None:
            return super().forward(
                attn_metadata=attn_metadata,
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                return_context_logits=return_context_logits,
                spec_metadata=spec_metadata,
                resource_manager=resource_manager,
                **kwargs,
            )
        # A generation step with speculative decoding: the verify logits on the decode
        # target's kernels where they serve the rows, then the spec worker.
        hidden_states = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )
        if attn_metadata.padded_num_tokens is not None:
            num_tokens = attn_metadata.num_tokens
            hidden_states = hidden_states[:num_tokens]
            if input_ids is not None:
                input_ids = input_ids[:num_tokens]
            position_ids = _slice_spec_position_ids(position_ids, num_tokens)
        rows = hidden_states[spec_metadata.gather_ids]
        logits = decode.logits(self.lm_head, rows)
        if logits is None:
            logits = self.logits_processor.forward(rows, self.lm_head, attn_metadata, True)
        return self.spec_worker(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=resource_manager,
        )

    def load_weights(self, weights, *args, **kwargs):
        self._draft_weights = _weights.load_weights(
            self, weights, isinstance(self.spec_worker, MTPDraftWorker)
        )

    def setup_aliases(self) -> None:
        _setup_aliases(self)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()
        draft_weights, self._draft_weights = self._draft_weights, None
        if draft_weights is not None:
            self.model.build_draft(self.draft_model.mtp_layers[0], self.lm_head, draft_weights)
