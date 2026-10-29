# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2-NVFP4 on Rubin (sm107): model construction and forward.

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

import os
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
from tensorrt_llm._torch.speculative import MTPEagleWorker, SpecMetadata, get_num_extra_kv_tokens
from tensorrt_llm._torch.utils import EventType

from . import weights as _weights
from ._full_context import FullContextIndexer
from ._model import Glm52Model
from ._model import setup_aliases as _setup_aliases
from ._mtp_worker import MTPDraftWorker
from ._small_batch_draft import DRAFT_SHAPES, build_small_batch_draft
from ._speculative import Glm52SpecDecOneEngineForCausalLM
from .ops.attention.mla_kv_half_append import mla_kv_half_append
from .ops.attention.sparse_mla_flash_decode import (
    sparse_mla_flash_decode,
    sparse_mla_flash_decode_latent,
)
from .ops.comm._mnnvl_workspace import (
    LamportExchangeWorkspace,
    hop_supported,
    hop_workspace,
    publish_target,
)
from .ops.comm.intra_node_reduce import IpcCollectives
from .ops.comm.mnnvl_allgather import mnnvl_allgather
from .ops.comm.mnnvl_hop import mnnvl_hop
from .ops.comm.mnnvl_moe_finalize_hop import mnnvl_moe_finalize_hop
from .ops.dsa._fp8_paged_mqa_logits import num_sms
from .ops.dsa._full_context_topk import streaming_plan
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
from .ops.gemm._low_m_splitk import (
    bmm_prewait_tactic,
    device_limits,
    one_wave_tactic,
    publish_supported,
    publish_tactic,
    run_splitk_bmm,
    splitk_default_tactic,
)
from .ops.gemm._low_m_splitk_fused import Q_B_PLANS, q_b_plan
from .ops.gemm.dense_mlp_small_m import INTER_PER_RANK, dense_mlp_small_m
from .ops.gemm.draft_hidden_half import draft_hidden_half, hidden_half_tactic
from .ops.gemm.fused_a_gemm import INDEXER_OUT_FEATURES as FUSED_A_INDEXER_OUT_FEATURES
from .ops.gemm.fused_a_gemm import NATIVE_MAX_TOKENS as FUSED_A_MAX_TOKENS
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
from .ops.spec.verify_head_sample import VerifyHeadSampleState

# The deployment this target is written for; route() selects it on the same
# facts, and the constructor states them again for a direct construction.
_COMPUTE_CAPABILITY = (10, 7)
_SUPPORTED_TP = (4, 8, 16)
# One Lamport buffer of the verify logits' all-gather: every rank's BF16 vocabulary shard of
# up to 24 rows, with room to spare.
_LOGITS_EXCHANGE_BYTES = 16 << 20


class PrefillTarget:
    """Forwards that carry context rows: the decoder stack the context server runs."""

    def __init__(self, core: "Glm52Nvfp4Sm107LowLatencyCore") -> None:
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
    """Generation kernels selected from the actual row count and TP mapping.

    TP16 uses the one-request histogram indexer, folded MLA output, MNNVL hops,
    and inline table-based MTP loop. TP8 uses full-context indexing, folded
    8-head MLA, MNNVL hops and a small-batch table-based MTP loop.
    TP4 uses full-context indexing, 16-head
    latent MLA followed by batched v_b, IPC collectives, and a small-batch MTP
    loop. BF16 dense layers use the fused TP16 kernel or owned split-K GEMMs.

    Both paths fuse the router into the NVFP4 expert call, overlap the shared
    expert with routed experts, and gather vocabulary shards from an owned
    low-M LM-head projection. Unsupported batches select the built-in layer
    stack before writing attention caches.
    """

    def __init__(self, core: "Glm52Nvfp4Sm107LowLatencyCore") -> None:
        self.core = core
        config = core.model_config.pretrained_config
        self.mapping = core.model_config.mapping
        self.hidden_size = config.hidden_size
        self.index_topk = config.index_topk
        layers = core.layers[: core.num_hidden_layers]
        self._v_b_tactics = {}
        for layer in layers:
            assert layer.allreduce is not None and layer.next_layer_layernorm is not None
            self._check_attention(layer)
            if isinstance(layer.mlp, GatedMLP):
                assert (
                    layer.mlp_tp_size == self.mapping.tp_size
                    and not layer.disable_attn_allreduce
                    and not layer.fusion_config.PRE_MLP_FUSION
                    and layer.fusion_config.POST_MLP_FUSION
                    and layer.post_attention_layernorm.nvfp4_scale is None
                ), f"layer {layer.layer_idx}: unexpected dense-layer configuration"
                if self.mapping.tp_size == 8:
                    assert layer.mlp.gate_up_proj.weight.shape == (3072, self.hidden_size)
                    assert layer.mlp.down_proj.weight.shape == (self.hidden_size, 1536)
            else:
                assert (
                    layer.disable_attn_allreduce
                    and layer.fusion_config.PRE_MOE_FUSION
                    and layer.fusion_config.POST_MOE_FUSION
                ), f"layer {layer.layer_idx}: unexpected MoE-layer configuration"
                self._check_shared_expert(layer)
                if self.mapping.tp_size == 8:
                    shared = layer.mlp.shared_experts
                    assert shared.gate_up_proj.weight.shape == (512, self.hidden_size)
                    assert shared.down_proj.weight.shape == (self.hidden_size, 256)
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
            get_hist_topk_workspace(self.mapping)
            if self.mapping.tp_size == 16 and hist_topk_supported(self.mapping)
            else None
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
        self._small_batch_draft = None
        self._ipc = None
        self._ipc_reason = ""
        self._full_context = None
        if self.mapping.tp_size == 4:
            workspaces = [
                collective.workspace
                for layer in core.layers
                for collective in (layer.allreduce, layer.moe_allreduce)
                if collective is not None
            ]
            self._ipc, self._ipc_reason = IpcCollectives.bind(
                self.mapping, workspaces, max_rows=24, hidden=self.hidden_size
            )
        if self.mapping.tp_size in (4, 8):
            self._full_context = FullContextIndexer(self, core._decode_context_capacity, device)
        if self.mapping.tp_size == 8:
            self._prepare_tp8_rows(device)

    def _prepare_tp8_rows(self, device: torch.device) -> None:
        """Check the complete TP8 projection/collective route before profiling writes KV."""
        sms, smem = device_limits(device.index)
        for rows in (1, 2, 4, 8):
            # Dense, shared-expert and first-draft eh_proj projections.
            for n, k in ((3072, 6144), (6144, 1536), (512, 6144), (6144, 256)):
                splitk_default_tactic(rows, n, k, sm_count=sms, smem_capacity=smem)
            head = splitk_default_tactic(rows, 19360, 6144, sm_count=sms, smem_capacity=smem)
            one_wave_tactic(head, rows, 19360, 6144, sm_count=sms, smem_capacity=smem)
            if rows <= 2:
                hidden_half_tactic(
                    rows,
                    768,
                    self.hidden_size,
                    in_kernel_norm=rows == 1,
                    sm_count=sms,
                    smem_capacity=smem,
                )
            projected = publish_tactic(
                splitk_default_tactic(rows, 6144, 2048, sm_count=sms, smem_capacity=smem),
                smem_capacity=smem,
            )
            for plan in Q_B_PLANS:
                tactic = q_b_plan(plan, rows, 4608, 2048, device.index)
                if plan == "beside_topk":
                    reserve = (
                        ((4608 + tactic.mma_m - 1) // tactic.mma_m)
                        * ((rows + tactic.mma_n - 1) // tactic.mma_n)
                        * tactic.split_k
                    )
                    streaming_plan(rows, self._full_context.capacity, sms, reserve)
            self._fast_rows[rows] = publish_supported(
                rows, self.hidden_size, projected, smem_capacity=smem
            ) and hop_supported(rows, self.hidden_size, self.mapping.tp_size)
        # Allocate the largest hop before capture; each launch still asks for its current view.
        hop_workspace(self.mapping, 8, self.hidden_size)

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
        if self.mapping.tp_size == 4:
            assert mla.num_heads_tp == 16 and mla.kv_lora_rank == 512 and mla.v_head_dim == 256
            for rows in (1, 2, 3, 6, 12, 18):
                tactic = bmm_prewait_tactic(rows, mla.v_head_dim, mla.kv_lora_rank)
                assert tactic is not None, f"no v_b tactic for {rows} rows"
                self._v_b_tactics[rows] = tactic
        elif self.mapping.tp_size == 8:
            assert (
                mla.num_heads_tp == 8
                and mla.q_lora_rank == 2048
                and mla.kv_lora_rank == 512
                and mla.qk_rope_head_dim == 64
                and mla.v_head_dim == 256
                and mla.v_b_proj.shape == (8, 256, 512)
                and mla.o_proj.weight.shape == (self.hidden_size, 2048)
            ), f"layer {layer.layer_idx}: unexpected TP8 MLA geometry"

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
        if self.mapping.tp_size != 16 or self._draft is not None:
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
        """Decide the complete kernel route before any attention cache writes."""
        if torch.compiler.is_compiling():
            return False
        if self.mapping.tp_size == 8:
            return self._fast_rows.get(rows, False)
        if self.mapping.tp_size == 4:
            return (
                rows in self._v_b_tactics
                and self._ipc is not None
                and self._ipc.supported(rows)
                and self._ipc.supported(rows, finalize=True)
            )
        fast = self._fast_rows.get(rows)
        if fast is None:
            fast = (
                1 <= rows <= 8
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

    def supports_small_batch_draft(self, metadata, first_rows, batch_size, layer) -> bool:
        tp = self.mapping.tp_size
        return (
            tp in (4, 8)
            and batch_size > 0
            and first_rows % batch_size == 0
            and batch_size <= DRAFT_SHAPES.get((tp, first_rows // batch_size - 1), 0)
            and self._full_context is not None
            and self._full_context.supports(metadata, first_rows)
            and self._fast(first_rows)
            and self._fast(batch_size)
            and layer.layer_idx in self._q_b_absorbed
            and layer.layer_idx in self._fused_a_indexer_weight
        )

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
        if not self._fast(hidden_states.shape[0]) or (
            self._full_context is not None
            and not self._full_context.supports(attn_metadata, hidden_states.shape[0])
        ):
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
            quantized = None
            if self._ipc is not None and not isinstance(layer.mlp, GatedMLP):
                hidden_states, residual, packed, scales = self._post_attention_hop(
                    layer, attn_out, residual, premoe=True
                )
                quantized = (packed, scales)
            else:
                hidden_states, residual = self._post_attention_hop(layer, attn_out, residual)
            if isinstance(layer.mlp, GatedMLP):
                mlp = layer.mlp
                # This rank's partial sum; the hop adds the TP ranks'.
                if self.mapping.tp_size == 16:
                    partial = dense_mlp_small_m(
                        hidden_states, mlp.gate_up_proj.weight, mlp.down_proj.weight
                    )
                else:
                    gate_up = low_m_gemm(
                        hidden_states,
                        mlp.gate_up_proj.weight,
                        hidden_states.new_empty(
                            hidden_states.shape[0], mlp.gate_up_proj.weight.shape[0]
                        ),
                    )
                    partial = low_m_gemm(
                        swiglu(gate_up),
                        mlp.down_proj.weight,
                        hidden_states.new_empty(hidden_states.shape[0], self.hidden_size),
                    )
                hidden_states, residual = self._hop(layer.next_layer_layernorm, partial, residual)
            else:
                hidden_states, residual = self._moe(layer, hidden_states, residual, quantized)
        return hidden_states

    def logits(self, lm_head, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        """FP32 logits over the whole vocabulary of the rows, or None for a batch the kernels
        do not serve: this rank's vocabulary shard from the LM head, then the all-gather."""
        rows = hidden_states.shape[0]
        weight = lm_head.weight
        if (
            not 1 <= rows <= 24
            or torch.compiler.is_compiling()
            or weight.shape[0] * self.mapping.tp_size != lm_head.num_embeddings
        ):
            return None
        shard = self._lm_head_shard(lm_head, hidden_states)
        return mnnvl_allgather(
            shard,
            self._logits_workspace.comm_buffer,
            self._logits_workspace.buffer_flags,
            self.mapping.tp_size,
        )

    def _lm_head_shard(self, lm_head, hidden_states):
        weight = lm_head.weight
        return low_m_gemm(
            hidden_states.contiguous(),
            weight,
            hidden_states.new_empty(hidden_states.shape[0], weight.shape[0]),
            one_wave=True,
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
            rows = 8
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
        *,
        draft_state=None,
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
        full_context = self._full_context
        if full_context is not None and full_context.reuse_draft_selection(
            indexer, attn_metadata, draft_state
        ):
            positions = draft_state.topk_stash[: attn_metadata.num_seqs]
        reuses_selection = pool_rows is not None or positions is not None
        own_indexer = (
            (sharded or full_context is not None) and indexer is not None and not reuses_selection
        )
        # A layer that attends to the selection of the indexer layer before it gathers its old KV
        # rows before the flash decode's wait.
        early_gather = (
            (sharded or (full_context is not None and attn_metadata.num_seqs <= 2))
            and indexer is None
            and not reuses_selection
        )
        if full_context is not None and indexer is None:
            positions = attn_metadata.shared_topk_indices[:num_tokens]

        weight = (
            self._fused_a_indexer_weight[layer.layer_idx]
            if own_indexer
            else mla.kv_a_proj_with_mqa.weight
        )
        if reuses_selection:
            l2_prefetch = (self._draft if draft_state is None else draft_state).l2_after_fused_a
        elif early_gather:
            # Also the selection the flash decode reads first.
            l2_prefetch = self._l2_after_fused_a[layer.layer_idx] + [
                self._hist_buffers["topk"][:num_tokens] if sharded else positions
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
        if sharded or full_context is not None:
            if own_indexer:

                def select():
                    nonlocal positions
                    if full_context is not None:
                        positions = full_context.select(
                            layer,
                            kv_a,
                            q_a,
                            position_ids,
                            attn_metadata,
                            draft_state,
                            beside_topk=kv_half_then_q_b,
                        )
                    else:
                        self._select_rows(
                            indexer,
                            kv_a,
                            latent_end,
                            q_a,
                            position_ids,
                            attn_metadata,
                            num_tokens,
                            mla,
                        )

                def kv_half_then_q_b():
                    kv_half()
                    # The q_b launch shares SMs with the reserved TopK grid.
                    q_b("beside_topk")

                if full_context is not None:
                    # The full-context path queues q_b only once logits are ready,
                    # leaving the earlier full-query scoring pass all of the SMs.
                    select()
                else:
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
            if self.mapping.tp_size == 4 and full_context is not None:
                page_index_scale, layer_offset = manager.get_primary_pool_page_index_params(
                    local_layer
                )
                latent = hidden_states.new_empty((num_tokens, mla.num_heads_tp, mla.kv_lora_rank))
                sparse_mla_flash_decode_latent(
                    q_heads,
                    kv_pool,
                    positions,
                    kv_lens,
                    self._bmm1_scale,
                    self._bmm2_scale,
                    latent,
                    seq_len_q,
                    self.index_topk,
                    attn_metadata.block_table[: attn_metadata.num_seqs],
                    manager.tokens_per_block,
                    page_index_scale,
                    layer_offset,
                    early_gather,
                    True,
                )
                tactic = self._v_b_tactics[num_tokens]
                run_splitk_bmm(
                    latent.transpose(0, 1),
                    mla.v_b_proj,
                    output.view(num_tokens, mla.num_heads_tp, mla.v_head_dim).transpose(0, 1),
                    get_env_enable_pdl(),
                    tactic,
                    reserve_full_smem=False,
                )
                return output
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

    def _post_attention_hop(
        self, layer, attn_out: torch.Tensor, residual: torch.Tensor, *, premoe=False
    ):
        """o_proj published into the all-reduce workspace, then the hop with the residual add
        and post_attention_layernorm: (normed, residual)."""
        rows = attn_out.shape[0]
        if self._ipc is not None:
            shard = low_m_gemm(
                attn_out,
                layer.self_attn.o_proj.weight,
                attn_out.new_empty((rows, self.hidden_size)),
            )
            norm = layer.post_attention_layernorm
            if premoe:
                backend = getattr(layer.mlp.experts, "backend", layer.mlp.experts)
                return self._ipc.premoe(shard, residual, norm, backend.fc31_input_scale)
            return self._ipc.plain_reduce(norm, shard, residual)
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
        if self._ipc is not None:
            return self._ipc.plain_reduce(norm, partial, residual)
        comm_buffer, buffer_flags = hop_workspace(self.mapping, partial.shape[0], self.hidden_size)
        return mnnvl_hop(
            partial, residual, norm.weight, norm.variance_epsilon, comm_buffer, buffer_flags, False
        )

    def _moe(self, layer, hidden_states: torch.Tensor, residual: torch.Tensor, quantized=None):
        """The MoE block and the post-MoE hop that finalizes it: (normed, residual)."""
        mlp = layer.mlp
        backend = getattr(mlp.experts, "backend", mlp.experts)
        rows = hidden_states.shape[0]

        if quantized is None:
            self._quant_fork.record()
            with torch.cuda.stream(self._quant_stream):
                self._quant_stream.wait_event(self._quant_fork)
                x_fp4, x_sf = fp4_quantize(
                    hidden_states,
                    backend.fc31_input_scale,
                    backend.scaling_vector_size,
                    False,
                    False,
                )
                self._quant_ready.record()
            quant_ready = self._quant_ready.cuda_event
        else:
            x_fp4, x_sf = quantized
            quant_ready = 0
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
                quant_ready,
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
        if self._ipc is not None:
            return self._ipc.finalize(
                fc2_output, expert_weights, permuted_rows, shared_output, residual, norm
            )
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


class Glm52Nvfp4Sm107LowLatencyCore(Glm52Model):
    """The decoder stack, with the per-step choice between the two targets."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        super().__init__(model_config)
        self._prefill: Optional[PrefillTarget] = None
        self._decode: Optional[DecodeTarget] = None
        self._decode_context_capacity = None

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
        if self.model_config.mapping.tp_size in (4, 8):
            decode = self._decode
            decode._check_attention(layer)
            decode._check_shared_expert(layer)
            decode._fold_attention_weights(layer)
            decode._small_batch_draft = build_small_batch_draft(decode, layer, lm_head, weights)
        else:
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


class ModelingV2Glm52Nvfp4Sm107LowLatency(
    Glm52SpecDecOneEngineForCausalLM[Glm52Nvfp4Sm107LowLatencyCore, PretrainedConfig]
):
    """GLM-5.2-NVFP4 on Rubin, with tensor-parallel attention and experts."""

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
            mapping.tp_size in _SUPPORTED_TP
            and mapping.world_size == mapping.tp_size
            and mapping.moe_tp_size == mapping.tp_size
            and mapping.cp_size == 1
            and mapping.moe_ep_size == 1
            and not mapping.enable_attention_dp
            and mapping.pp_size == 1
        ), (
            f"this target uses TP4/TP8/TP16 without attention DP or expert parallelism, not {mapping}"
        )
        model_config = _weights.normalize_model_config(model_config)
        super().__init__(
            model=Glm52Nvfp4Sm107LowLatencyCore(model_config), model_config=model_config
        )
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
        inferred = self.infer_max_seq_len()
        requested = self.model.model_config.max_seq_len
        allow_long = os.getenv("TLLM_ALLOW_LONG_MAX_MODEL_LEN", "0").lower() in (
            "1",
            "true",
            "yes",
            "y",
        )
        net = (
            inferred if requested is None else requested if allow_long else min(requested, inferred)
        )
        spec = self.model.model_config.spec_config
        extra = (
            0 if spec is None else 2 * spec.max_total_draft_tokens + get_num_extra_kv_tokens(spec)
        )
        self.model._decode_context_capacity = net + extra
        if self.model.model_config.mapping.tp_size in (4, 8) and isinstance(
            self.spec_worker, MTPDraftWorker
        ):
            self.spec_worker._verify_sampler = VerifyHeadSampleState(self.lm_head.weight.device)
        self.model.post_load_weights()
        draft_weights, self._draft_weights = self._draft_weights, None
        if draft_weights is not None:
            self.model.build_draft(self.draft_model.mtp_layers[0], self.lm_head, draft_weights)
