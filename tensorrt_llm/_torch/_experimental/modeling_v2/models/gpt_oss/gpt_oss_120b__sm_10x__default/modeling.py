# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: gpt-oss-120b / sm_10x / default -- one forward for every
parallel layout the routing table admits, one GPU included.

Its weight shapes are computed from the engine's `Mapping` (weights.py), and
the difference between layouts is settled once, in each target's `__init__`,
rather than by branching in the forward:

* **Attention TP** (attention DP off, world > 1). Each rank runs its share of
  the heads over every token. o_proj and the MoE leave partial sums, and both
  residual-add + RMSNorm slots are an allreduce with the norm fused in -- the
  same two allreduces per layer the built-in model runs: `comm/allreduce`
  inside one node, `comm/mnnvl_allreduce` when the TP group spans nodes,
  where the first one's IPC workspace cannot reach. The MoE is split by
  moe_tp x moe_ep inside the same TP group, so no other collective is needed.

* **Attention DP** (attention DP on, world > 1). Each rank runs every head
  over its own requests. Both norm slots are the local
  `flashinfer_fused_add_rmsnorm`, and the MoE block becomes the DeepSeek dep4
  round trip: pad to the group's largest token count, `comm/allgather`, route
  and run this rank's experts over the gathered set, `comm/reducescatter`
  back. With the MoE also split by intermediate (moe_tp > 1) that is the
  built-in model's round trip too; for pure EP it would pick an all-to-all
  when NVLink one-sided is available, where this target still uses
  allgather + reduce-scatter.

* **One rank** (tp1). No collective at all: attention TP is 1, so both norm
  slots are the local `flashinfer_fused_add_rmsnorm`; every head, expert and
  intermediate column is local, o_proj adds its bias, and lm_head is whole.

Weights are target-owned and declared in the sibling weights.py. Each target
holds the per-layer weight views it reads, passes every argument to every op
at the call site, and shares no code with the other target.
"""

import math
from typing import Callable, NamedTuple

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import step_contract_enabled
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.fused_qk_norm_rope import (
    fused_qk_norm_rope,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allgather import allgather
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allreduce import (
    RESIDUAL_RMS_NORM,
    allreduce,
    allreduce_group_args,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_allreduce import (
    mnnvl_allreduce,
    mnnvl_workspace_args,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.reducescatter import reducescatter
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.mxe4m3_mxe2m1_block_scale_moe_runner import (  # noqa: E501
    ACT_TYPE_SWIGLU,
    ROUTING_METHOD_RENORMALIZE,
    glu_vectors,
    mxe4m3_mxe2m1_block_scale_moe_runner,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (  # noqa: E501
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.mxfp8_quantize import (
    mxfp8_quantize,
)
from tensorrt_llm._torch.attention.backends.fmha.phased import get_spec_decoding_position_offsets
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._utils import mpi_disabled

from . import weights as _weights

_CALL_CONSTANTS = dict(
    output_sf=None,
    k=None,
    v=None,
    is_fused_qkv=True,
    update_kv_cache=True,
    predicted_tokens_per_seq=1,
    attention_input_type=0,
    is_mla_enable=False,
    mask_type=1,
    q_scaling=1.0,
    quant_mode=0,
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    # In-kernel RoPE is disabled here; the rope_* values below are inert placeholders, not this model's rope config.
    position_embedding_type=0,
    rotary_inv_freq=None,
    rotary_cos_sin=None,
    rope_dim=0,
    rope_base=10000.0,
    rope_scale_type=0,
    rope_scale=1.0,
    rope_short_m_scale=1.0,
    rope_long_m_scale=1.0,
    rope_max_positions=1024,
    rope_original_max_positions=1024,
    out_scale=None,
    latent_cache=None,
    q_pe=None,
    q_lora_rank=None,
    kv_lora_rank=None,
    qk_nope_head_dim=None,
    qk_rope_head_dim=None,
    v_head_dim=None,
    rope_append=None,
    chunked_prefill_buffer_batch_size=1,
    attention_chunk_size=None,
    softmax_stats_tensor=None,
    sparse_kv_indices=None,
    sparse_kv_offsets=None,
    sparse_attn_indices=None,
    sparse_attn_offsets=None,
    sparse_attn_indices_block_size=0,
    mrope_rotary_cos_sin=None,
    mrope_position_deltas=None,
    helix_position_offsets=None,
    helix_is_inactive_rank=None,
    cross_kv=None,
    relative_attention_bias=None,
    relative_attention_max_distance=0,
    kv_norm_weight=None,
    kv_norm_eps=1e-6,
    skip_correction_threshold=0.0,
)

_FC1_K_ALIGN = _weights.FC1_K_ALIGN

# The MoE runner reads the activation scales as a linear buffer. The swizzled 128x4 order has
# the same byte count whenever num_tokens % 128 == 0, which every decode CUDA-graph batch is, so
# a wrong value here is accepted silently as a wrong answer.
_SF_LINEAR = False

# The MoE runner's certified token columns stop at 8192. A prefill step
# exceeds it when max_num_tokens does, and under attention DP the gathered set
# reaches world * max_num_tokens, so every expert call is chunked; every token
# is independent through routing, quantization and the expert GEMMs, so the
# chunks re-joined are the whole call.
_MOE_MAX_T = 8192


def _in_chunks(experts: Callable, x: torch.Tensor, L: "_LayerWeights") -> torch.Tensor:
    """`experts(x, L)` over at most `_MOE_MAX_T` rows at a time, re-joined in order."""
    parts = [experts(c, L) for c in torch.split(x, _MOE_MAX_T, 0)]
    return parts[0] if len(parts) == 1 else torch.cat(parts, 0)


#: One residual-norm slot: `(x, residual, norm_weight) -> (normed, residual)`.
ResidualNorm = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple]


class _LayerWeights(NamedTuple):
    """One decoder layer's weights at this rank's share, as the views the ops read: `.t()` for
    the GEMM operands, and the *next* layer's input norm (the final norm on the last layer) as
    `norm_next`. `o_bias` is None off attention TP rank 0."""

    qkv: torch.Tensor
    qkv_bias: torch.Tensor
    sinks: torch.Tensor
    o: torch.Tensor
    o_bias: torch.Tensor | None
    norm2: torch.Tensor
    router: torch.Tensor
    router_bias: torch.Tensor
    fc1_w: torch.Tensor
    fc1_s: torch.Tensor
    fc1_b: torch.Tensor
    fc2_w: torch.Tensor
    fc2_s: torch.Tensor
    fc2_b: torch.Tensor
    norm_next: torch.Tensor


def _layer_views(w, num_layers: int, attn_tp_rank: int) -> list[_LayerWeights]:
    # Row-parallel o_proj: each attention TP rank holds a slice of the input
    # columns, so the bias is added on one rank and the allreduce carries it
    # to the rest.
    return [
        _LayerWeights(
            qkv=w[f"l{i}_qkv"].t(),
            qkv_bias=w[f"l{i}_qkv_bias"],
            sinks=w[f"l{i}_sinks"],
            o=w[f"l{i}_o"].t(),
            o_bias=w[f"l{i}_o_bias"] if attn_tp_rank == 0 else None,
            norm2=w[f"l{i}_norm2"],
            router=w[f"l{i}_router"].t(),
            router_bias=w[f"l{i}_router_bias"],
            fc1_w=w[f"l{i}_fc1_w"],
            fc1_s=w[f"l{i}_fc1_s"],
            fc1_b=w[f"l{i}_fc1_b"],
            fc2_w=w[f"l{i}_fc2_w"],
            fc2_s=w[f"l{i}_fc2_s"],
            fc2_b=w[f"l{i}_fc2_b"],
            norm_next=w[f"l{i + 1}_norm1"] if i + 1 < num_layers else w["final_norm"],
        )
        for i in range(num_layers)
    ]


def _residual_norm(core: "GptOssDefaultModelingV2") -> ResidualNorm:
    """The op that fills both residual-norm slots on this layout, chosen once.

    Under attention TP the MoE and o_proj outputs are partial sums, and the
    allreduce that completes them also adds the residual and normalizes;
    otherwise each rank's sum is already whole and the norm is local. Each
    returned function passes every argument of its op at the call; what it
    closes over is fixed for the life of the mapping.
    """
    cfg = core.model_config.pretrained_config
    mapping = core.model_config.mapping
    eps = cfg.rms_norm_eps

    if core.dims.attn_tp == 1:

        def local(x, residual, norm_weight):
            return flashinfer_fused_add_rmsnorm(x, residual, norm_weight, eps)

        return local

    if mapping.is_multi_node():
        # The group spans trays: the IPC workspace cannot reach the other
        # tray, so the multicast one carries the sum, as in the built-in
        # model. Building it is collective over the group.
        mnnvl = mnnvl_workspace_args(
            mapping,
            max_rows=core.model_config.max_num_tokens,
            hidden=cfg.hidden_size,
            dtype=core.dims.dtype,
        )

        def across_trays(x, residual, norm_weight):
            return mnnvl_allreduce(
                x,
                residual,
                norm_weight,
                eps,
                comm_buffer=mnnvl["comm_buffer"],
                buffer_flags=mnnvl["buffer_flags"],
                world=mnnvl["world"],
            )

        return across_trays

    group = allreduce_group_args(mapping)

    def in_tray(x, residual, norm_weight):
        norm_out, residual_out = allreduce(
            x,
            residual,
            norm_weight,
            scale=None,
            bias=None,
            workspace=group["workspace"],
            group=group["group"],
            strategy=group["strategy"],
            op=RESIDUAL_RMS_NORM,
            eps=eps,
            trigger_completion_at_end=False,
            process_group=group["process_group"],
            rank=group["rank"],
        )
        return norm_out, residual_out

    return in_tray


class GptOssDefaultModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = model_config.pretrained_config
        m = model_config.mapping

        # The layout facts routing does not read. Each one changes which ranks
        # hold which rows or heads, so running through would compute a
        # plausible wrong answer rather than fail.
        assert m.moe_cluster_size == 1, "moe_cluster_size is not implemented"
        assert m.enable_attention_dp or m.attn_tp_size == m.tp_size, (
            f"attention TP {m.attn_tp_size} differs from TP {m.tp_size}"
        )
        assert not getattr(m, "enable_lm_head_tp_in_adp", False), (
            "lm_head TP under attention DP is not implemented"
        )
        kv_algo = model_config.quant_config.kv_cache_quant_algo
        assert kv_algo is None, f"unsupported KV algo {kv_algo}; bf16 pool only"

        rope = getattr(cfg, "rope_scaling", None) or getattr(cfg, "rope_parameters", None)
        self.theta = float(rope.get("rope_theta", getattr(cfg, "rope_theta", 0.0)))

        layer_types = list(cfg.layer_types)
        self.sliding = [t == "sliding_attention" for t in layer_types]
        self.window = cfg.sliding_window

        self.dims = _weights.MODEL_WEIGHTS.dims(self)
        self.w = _weights.MODEL_WEIGHTS.declare(self.dims)

        self.rank = m.rank
        self.dp_size = m.world_size if m.enable_attention_dp else 1
        self.dp_group = list(range(m.world_size))
        # The communicator side of the attention-DP allgather and reduce-scatter, routed
        # like `allreduce_group_args`: under MPI the plain symbols, with MPI disabled
        # the `_pg` symbols over the TP process group, which spans the DP ranks. Only
        # the attention-DP MoE block reads it; a single-rank mapping's TP group may be
        # a stand-in with no `boxed()`, so it is not looked up there.
        self.dp_process_group = (
            m.tp_group_pg.boxed() if mpi_disabled() and self.dp_size > 1 else None
        )

        self._prefill: "PrefillTarget | None" = None
        self._decode: "DecodeTarget | None" = None

    def post_load_weights(self) -> None:
        """Construct the two targets, now that the weights are real.

        Must only be called after meta init is over, never from `__init__`, where the
        shell's containers are still meta.

        Named for the shell hook that drives it. Not an override: that hook lives on
        DecoderModelForCausalLM, and a core derives from DecoderModel, which has no
        such method -- so there is no super() to call here.
        """
        cfg = self.model_config.pretrained_config
        rope = getattr(cfg, "rope_scaling", None) or getattr(cfg, "rope_parameters", None)
        rotary_dim = cfg.head_dim
        yarn_factor = float(rope["factor"])
        beta_fast = float(rope["beta_fast"])
        beta_slow = float(rope["beta_slow"])
        orig_max = float(rope["original_max_position_embeddings"])
        truncate = bool(rope.get("truncate", True))

        def correction_dim(rotations: float) -> float:
            return (
                rotary_dim
                * math.log(orig_max / (rotations * 2.0 * math.pi))
                / (2.0 * math.log(self.theta))
            )

        low = correction_dim(beta_fast)
        high = correction_dim(beta_slow)
        if truncate:
            low, high = math.floor(low), math.ceil(high)
        yarn_low = max(low, 0.0)
        yarn_high = min(high, rotary_dim - 1.0)
        yarn_attn_factor = 0.1 * math.log(yarn_factor) + 1.0 if yarn_factor > 1.0 else 1.0

        self._prefill = PrefillTarget(
            self,
            yarn_factor=yarn_factor,
            yarn_low=yarn_low,
            yarn_high=yarn_high,
            yarn_attn_factor=yarn_attn_factor,
        )
        self._decode = DecodeTarget(
            self,
            yarn_factor=yarn_factor,
            yarn_low=yarn_low,
            yarn_high=yarn_high,
            yarn_attn_factor=yarn_attn_factor,
        )

    def _select_target(self, attn_metadata) -> "PrefillTarget | DecodeTarget":
        """Which target runs this step.

        One host-side integer read and nothing else: a decode-only CUDA graph is a
        per-capture constant environment, and a predicate that reached for a second field
        would be reading one this call is not guaranteed to carry. A mixed batch -- context
        and generation rows in one step -- goes to prefill, so prefill is the general case
        and decode the specialization that may assume it has no context rows.
        """
        return self._decode if attn_metadata.num_contexts == 0 else self._prefill

    def _dp_pad_rows(self, md, num_tokens: int) -> int:
        """Rows this rank pads its block with before the MoE round trip: up to
        the largest per-rank token count in the group, so both collectives run
        in their uniform form. Zero without attention DP.

        Under decode CUDA graphs the engine pads every rank to the captured
        batch size, so the counts are equal and the padding is zero rows.
        """
        if self.dp_size == 1:
            return 0
        counts = [int(n) for n in md.all_rank_num_tokens]
        assert len(counts) == self.dp_size, (counts, self.dp_size)
        assert counts[self.rank] == num_tokens, (
            f"rank {self.rank} holds {num_tokens} rows but all_rank_num_tokens "
            f"says {counts[self.rank]} ({counts}); the collectives would gather the wrong split"
        )
        return max(counts) - num_tokens

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Route the step, and nothing else."""
        return self._select_target(attn_metadata).forward(attn_metadata, *args, **kwargs)


class PrefillTarget:
    """The general case: context rows, and possibly generation rows beside them."""

    def __init__(
        self,
        core: GptOssDefaultModelingV2,
        *,
        yarn_factor: float,
        yarn_low: float,
        yarn_high: float,
        yarn_attn_factor: float,
    ) -> None:
        """Take the views and constants this target's forward reads, once, off the real weights.

        The parallel layout is decided here and nowhere else: which op fills
        the two residual-norm slots, whether o_proj adds its bias, and which
        MoE block the forward calls.
        """
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per
        # forward. Each target owns its own flag: the two `step_args` projections differ,
        # so probing one does not prove the other's fields exist.
        self._contract_pending = step_contract_enabled()
        cfg = core.model_config.pretrained_config
        d = core.dims
        w = core.w
        device = w["final_norm"].device
        dtype = w["final_norm"].dtype

        self._yarn_factor = yarn_factor
        self._yarn_low = yarn_low
        self._yarn_high = yarn_high
        self._yarn_attn_factor = yarn_attn_factor

        self._norm0 = w["l0_norm1"]
        self._layers = _layer_views(w, cfg.num_hidden_layers, d.attn_tp_rank)
        # fused_qk_norm_rope requires valid q/k norm weight tensors even when is_qk_norm=False, where
        # the values are unused.
        self._no_qk_norm = torch.zeros(cfg.head_dim, dtype=dtype, device=device)
        self._glu_alpha, self._glu_beta, self._glu_limit = glu_vectors(
            d.local_experts, cfg.swiglu_limit, device
        )

        self._residual_norm = _residual_norm(core)
        self._moe_block = self._moe_attention_dp if core.dp_size > 1 else self._moe_on_every_token

    def _experts(self, x: torch.Tensor, L: _LayerWeights) -> torch.Tensor:
        """Route, quantize and run this rank's experts over `x`."""
        cfg = self.core.model_config.pretrained_config
        d = self.core.dims
        # The router bias rides the GEMM epilogue: the MoE op silently ignores routing_bias
        # on this routing method.
        router_logits = cublas_mm(x, L.router, L.router_bias)
        hidden_fp8, hidden_sf = mxfp8_quantize(x, _SF_LINEAR, _FC1_K_ALIGN)
        return mxe4m3_mxe2m1_block_scale_moe_runner(
            routing_logits=router_logits,
            routing_bias=None,
            hidden_states=hidden_fp8,
            hidden_states_scale=hidden_sf,
            gemm1_weights=L.fc1_w,
            gemm1_weights_scale=L.fc1_s,
            gemm1_bias=L.fc1_b,
            gemm1_alpha=self._glu_alpha,
            gemm1_beta=self._glu_beta,
            gemm1_clamp_limit=self._glu_limit,
            gemm2_weights=L.fc2_w,
            gemm2_weights_scale=L.fc2_s,
            gemm2_bias=L.fc2_b,
            num_experts=cfg.num_local_experts,
            top_k=cfg.num_experts_per_tok,
            n_group=None,
            topk_group=None,
            intermediate_size=d.inter_pad,
            valid_hidden_size=cfg.hidden_size,
            valid_intermediate_size=d.inter_valid,
            local_expert_offset=d.expert_offset,
            local_num_experts=d.local_experts,
            routed_scaling_factor=None,
            routing_method_type=ROUTING_METHOD_RENORMALIZE,
            act_type=ACT_TYPE_SWIGLU,
        )

    def _moe_on_every_token(self, o: torch.Tensor, pad_rows: int, L: _LayerWeights) -> torch.Tensor:
        """Every rank holds every token: one rank, or attention TP, where the
        result is this rank's partial sum for the allreduce to complete."""
        return _in_chunks(self._experts, o, L)

    def _moe_attention_dp(self, o: torch.Tensor, pad_rows: int, L: _LayerWeights) -> torch.Tensor:
        """Each rank holds its own tokens: gather them, run this rank's
        experts over the whole set, and sum back to the rows this rank owns.

        Gathering before the router keeps the routing identical on every rank,
        so each token's top-k ids fall in exactly one expert window per moe_tp
        slice, and the reduce-scatter sums each contribution once."""
        group, pg = self.core.dp_group, self.core.dp_process_group
        num_tokens = o.shape[0]
        o_pad = o if not pad_rows else nn.functional.pad(o, [0, 0, 0, pad_rows])
        o_all = allgather(o_pad, None, group, pg)
        moe_all = _in_chunks(self._experts, o_all, L)
        moe_pad = reducescatter(moe_all, None, group, pg)
        return moe_pad if not pad_rows else moe_pad[:num_tokens]

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast, run when TRTLLM_MODELING_V2_VALIDATE asks for it:
        the metadata fields this target consumes must exist (they are private trtllm
        surface). Calling `step_args` is the check -- it reads every field this target
        consumes, so a rename upstream surfaces here rather than mid-forward. Everything
        checked is fixed at engine construction, so once per target instance is sound,
        and it is off in a served engine."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """Project the prepared metadata onto thop_attention's explicit batch state, once per forward."""
        return dict(
            sequence_length=md.kv_lens_cuda_runtime,
            host_past_key_value_lengths=md.kv_lens_runtime,
            host_total_kv_lens=md.host_total_kv_lens,
            context_lengths=md.prompt_lens_cuda_runtime,
            host_context_lengths=md.prompt_lens_cpu_runtime,
            host_request_types=md.host_request_types_runtime,
            kv_cache_block_offsets=md.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
            workspace=md.effective_workspace,
            tokens_per_block=md.tokens_per_block,
            max_num_requests=md.max_num_requests,
            max_context_length=md.max_context_length,
            max_seq_len=md.max_seq_len,
            num_contexts=md.num_contexts,
            num_ctx_tokens=md.num_ctx_tokens,
            trtllm_gen_jit_warmup=md.trtllm_gen_jit_warmup,
            use_paged_context_fmha=md.use_paged_context_fmha,
            beam_width=md.effective_beam_width,
            cache_indirection=md.cache_indirection,
            block_ids_per_seq=md.block_ids_per_seq,
            max_context_q_len_override=md.max_context_q_len_override,
            is_cross=md.is_cross,
            is_spec_decoding_enabled=md.is_spec_decoding_enabled,
            use_spec_decoding=md.use_spec_decoding,
            is_spec_dec_tree=md.is_spec_dec_tree,
            spec_decoding_generation_lengths=md.spec_decoding_generation_lengths,
            spec_decoding_position_offsets=get_spec_decoding_position_offsets(md),
            spec_decoding_packed_mask=md.spec_decoding_packed_mask,
            spec_decoding_bl_tree_mask_offset=md.spec_decoding_bl_tree_mask_offset,
            spec_decoding_bl_tree_mask=md.spec_decoding_bl_tree_mask,
            spec_decoding_target_max_draft_tokens=md.max_total_draft_tokens,
            spec_bl_tree_first_sparse_mask_offset_kv=md.spec_bl_tree_first_sparse_mask_offset_kv,
            num_sparse_topk=md.num_sparse_topk,
            flash_mla_tile_scheduler_metadata=md.flash_mla_tile_scheduler_metadata,
            flash_mla_num_splits=md.flash_mla_num_splits,
            max_num_sequences=md.max_num_sequences,
            force_prepare_spec_dec_tree_mask=md.force_prepare_spec_dec_tree_mask,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        core = self.core
        cfg = core.model_config.pretrained_config
        d = core.dims
        eps = cfg.rms_norm_eps
        head_dim = cfg.head_dim

        self._check_step_contract(attn_metadata)
        step = self.step_args(attn_metadata)
        # Full-attention layers take the no-window value; sliding layers take the checkpoint's
        # window. Both are host-derived per-capture constants.
        full_window = attn_metadata.max_seq_len
        no_qk_norm = self._no_qk_norm

        pos = torch.reshape(position_ids, [-1])

        if inputs_embeds is None:
            h = nn.functional.embedding(input_ids, core.w["embed"])
        else:
            h = inputs_embeds
        num_tokens = h.shape[0]
        pad_rows = core._dp_pad_rows(attn_metadata, num_tokens)
        attn_out = torch.empty([num_tokens, d.q_width], dtype=h.dtype, device=h.device)

        x = flashinfer_rmsnorm(h, self._norm0, eps)
        residual = h
        for i, L in enumerate(self._layers):
            qkv = cublas_mm(x, L.qkv, L.qkv_bias)
            qkv = fused_qk_norm_rope(
                qkv,
                num_heads_q=d.heads_q,
                num_heads_k=d.heads_kv,
                num_heads_v=d.heads_kv,
                head_dim=head_dim,
                rotary_dim=head_dim,
                eps=eps,
                q_weight=no_qk_norm,
                k_weight=no_qk_norm,
                base=core.theta,
                is_neox=True,
                position_ids=pos,
                factor=self._yarn_factor,
                low=self._yarn_low,
                high=self._yarn_high,
                attention_factor=self._yarn_attn_factor,
                is_qk_norm=False,
            )
            attn_out = thop_attention(
                q=qkv,
                output=attn_out,
                local_layer_idx=i,
                num_heads=d.heads_q,
                num_kv_heads=d.heads_kv,
                head_size=head_dim,
                attention_sinks=L.sinks,
                attention_window_size=core.window if core.sliding[i] else full_window,
                **step,
                **_CALL_CONSTANTS,
            )
            o = cublas_mm(attn_out, L.o, L.o_bias)
            o, residual = self._residual_norm(o, residual, L.norm2)
            moe = self._moe_block(o, pad_rows, L)
            moe, residual = self._residual_norm(moe, residual, L.norm_next)
            x = moe
        return x


class DecodeTarget:
    """The specialization: routed to only when there are no context rows."""

    def __init__(
        self,
        core: GptOssDefaultModelingV2,
        *,
        yarn_factor: float,
        yarn_low: float,
        yarn_high: float,
        yarn_attn_factor: float,
    ) -> None:
        """Take the views and constants this target's forward reads, once, off the real weights.

        The parallel layout is decided here and nowhere else: which op fills
        the two residual-norm slots, whether o_proj adds its bias, and which
        MoE block the forward calls.
        """
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per
        # forward. Each target owns its own flag: the two `step_args` projections differ,
        # so probing one does not prove the other's fields exist.
        self._contract_pending = step_contract_enabled()
        cfg = core.model_config.pretrained_config
        d = core.dims
        w = core.w
        device = w["final_norm"].device
        dtype = w["final_norm"].dtype

        self._yarn_factor = yarn_factor
        self._yarn_low = yarn_low
        self._yarn_high = yarn_high
        self._yarn_attn_factor = yarn_attn_factor

        self._norm0 = w["l0_norm1"]
        self._layers = _layer_views(w, cfg.num_hidden_layers, d.attn_tp_rank)
        # fused_qk_norm_rope requires valid q/k norm weight tensors even when is_qk_norm=False, where
        # the values are unused.
        self._no_qk_norm = torch.zeros(cfg.head_dim, dtype=dtype, device=device)
        self._glu_alpha, self._glu_beta, self._glu_limit = glu_vectors(
            d.local_experts, cfg.swiglu_limit, device
        )

        # A second call with the same mapping reuses the workspace the prefill
        # target's call built, so both targets hold the same one.
        self._residual_norm = _residual_norm(core)
        self._moe_block = self._moe_attention_dp if core.dp_size > 1 else self._moe_on_every_token

    def _experts(self, x: torch.Tensor, L: _LayerWeights) -> torch.Tensor:
        """Route, quantize and run this rank's experts over `x`."""
        cfg = self.core.model_config.pretrained_config
        d = self.core.dims
        # The router bias rides the GEMM epilogue: the MoE op silently ignores routing_bias
        # on this routing method.
        router_logits = cublas_mm(x, L.router, L.router_bias)
        hidden_fp8, hidden_sf = mxfp8_quantize(x, _SF_LINEAR, _FC1_K_ALIGN)
        return mxe4m3_mxe2m1_block_scale_moe_runner(
            routing_logits=router_logits,
            routing_bias=None,
            hidden_states=hidden_fp8,
            hidden_states_scale=hidden_sf,
            gemm1_weights=L.fc1_w,
            gemm1_weights_scale=L.fc1_s,
            gemm1_bias=L.fc1_b,
            gemm1_alpha=self._glu_alpha,
            gemm1_beta=self._glu_beta,
            gemm1_clamp_limit=self._glu_limit,
            gemm2_weights=L.fc2_w,
            gemm2_weights_scale=L.fc2_s,
            gemm2_bias=L.fc2_b,
            num_experts=cfg.num_local_experts,
            top_k=cfg.num_experts_per_tok,
            n_group=None,
            topk_group=None,
            intermediate_size=d.inter_pad,
            valid_hidden_size=cfg.hidden_size,
            valid_intermediate_size=d.inter_valid,
            local_expert_offset=d.expert_offset,
            local_num_experts=d.local_experts,
            routed_scaling_factor=None,
            routing_method_type=ROUTING_METHOD_RENORMALIZE,
            act_type=ACT_TYPE_SWIGLU,
        )

    def _moe_on_every_token(self, o: torch.Tensor, pad_rows: int, L: _LayerWeights) -> torch.Tensor:
        """Every rank holds every token: one rank, or attention TP, where the
        result is this rank's partial sum for the allreduce to complete."""
        return _in_chunks(self._experts, o, L)

    def _moe_attention_dp(self, o: torch.Tensor, pad_rows: int, L: _LayerWeights) -> torch.Tensor:
        """Each rank holds its own tokens: gather them, run this rank's
        experts over the whole set, and sum back to the rows this rank owns.

        Gathering before the router keeps the routing identical on every rank,
        so each token's top-k ids fall in exactly one expert window per moe_tp
        slice, and the reduce-scatter sums each contribution once."""
        group, pg = self.core.dp_group, self.core.dp_process_group
        num_tokens = o.shape[0]
        o_pad = o if not pad_rows else nn.functional.pad(o, [0, 0, 0, pad_rows])
        o_all = allgather(o_pad, None, group, pg)
        moe_all = _in_chunks(self._experts, o_all, L)
        moe_pad = reducescatter(moe_all, None, group, pg)
        return moe_pad if not pad_rows else moe_pad[:num_tokens]

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast, run when TRTLLM_MODELING_V2_VALIDATE asks for it:
        the metadata fields this target consumes must exist (they are private trtllm
        surface). Calling `step_args` is the check -- it reads every field this target
        consumes, so a rename upstream surfaces here rather than mid-forward. Everything
        checked is fixed at engine construction, so once per target instance is sound,
        and it is off in a served engine."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """Project the prepared metadata onto thop_attention's explicit batch state, once per forward."""
        return dict(
            sequence_length=md.kv_lens_cuda_runtime,
            host_past_key_value_lengths=md.kv_lens_runtime,
            host_total_kv_lens=md.host_total_kv_lens,
            context_lengths=md.prompt_lens_cuda_runtime,
            host_context_lengths=md.prompt_lens_cpu_runtime,
            host_request_types=md.host_request_types_runtime,
            kv_cache_block_offsets=md.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
            workspace=md.effective_workspace,
            tokens_per_block=md.tokens_per_block,
            max_num_requests=md.max_num_requests,
            max_context_length=md.max_context_length,
            max_seq_len=md.max_seq_len,
            num_contexts=0,
            num_ctx_tokens=0,
            trtllm_gen_jit_warmup=md.trtllm_gen_jit_warmup,
            use_paged_context_fmha=md.use_paged_context_fmha,
            beam_width=md.effective_beam_width,
            cache_indirection=md.cache_indirection,
            block_ids_per_seq=md.block_ids_per_seq,
            max_context_q_len_override=md.max_context_q_len_override,
            is_cross=md.is_cross,
            is_spec_decoding_enabled=md.is_spec_decoding_enabled,
            use_spec_decoding=md.use_spec_decoding,
            is_spec_dec_tree=md.is_spec_dec_tree,
            spec_decoding_generation_lengths=md.spec_decoding_generation_lengths,
            spec_decoding_position_offsets=get_spec_decoding_position_offsets(md),
            spec_decoding_packed_mask=md.spec_decoding_packed_mask,
            spec_decoding_bl_tree_mask_offset=md.spec_decoding_bl_tree_mask_offset,
            spec_decoding_bl_tree_mask=md.spec_decoding_bl_tree_mask,
            spec_decoding_target_max_draft_tokens=md.max_total_draft_tokens,
            spec_bl_tree_first_sparse_mask_offset_kv=md.spec_bl_tree_first_sparse_mask_offset_kv,
            num_sparse_topk=md.num_sparse_topk,
            flash_mla_tile_scheduler_metadata=md.flash_mla_tile_scheduler_metadata,
            flash_mla_num_splits=md.flash_mla_num_splits,
            max_num_sequences=md.max_num_sequences,
            force_prepare_spec_dec_tree_mask=md.force_prepare_spec_dec_tree_mask,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        core = self.core
        cfg = core.model_config.pretrained_config
        d = core.dims
        eps = cfg.rms_norm_eps
        head_dim = cfg.head_dim

        self._check_step_contract(attn_metadata)
        step = self.step_args(attn_metadata)
        # Full-attention layers take the no-window value; sliding layers take the checkpoint's
        # window. Both are host-derived per-capture constants.
        full_window = attn_metadata.max_seq_len
        no_qk_norm = self._no_qk_norm

        pos = torch.reshape(position_ids, [-1])

        if inputs_embeds is None:
            h = nn.functional.embedding(input_ids, core.w["embed"])
        else:
            h = inputs_embeds
        num_tokens = h.shape[0]
        pad_rows = core._dp_pad_rows(attn_metadata, num_tokens)
        attn_out = torch.empty([num_tokens, d.q_width], dtype=h.dtype, device=h.device)

        x = flashinfer_rmsnorm(h, self._norm0, eps)
        residual = h
        for i, L in enumerate(self._layers):
            qkv = cublas_mm(x, L.qkv, L.qkv_bias)
            qkv = fused_qk_norm_rope(
                qkv,
                num_heads_q=d.heads_q,
                num_heads_k=d.heads_kv,
                num_heads_v=d.heads_kv,
                head_dim=head_dim,
                rotary_dim=head_dim,
                eps=eps,
                q_weight=no_qk_norm,
                k_weight=no_qk_norm,
                base=core.theta,
                is_neox=True,
                position_ids=pos,
                factor=self._yarn_factor,
                low=self._yarn_low,
                high=self._yarn_high,
                attention_factor=self._yarn_attn_factor,
                is_qk_norm=False,
            )
            attn_out = thop_attention(
                q=qkv,
                output=attn_out,
                local_layer_idx=i,
                num_heads=d.heads_q,
                num_kv_heads=d.heads_kv,
                head_size=head_dim,
                attention_sinks=L.sinks,
                attention_window_size=core.window if core.sliding[i] else full_window,
                **step,
                **_CALL_CONSTANTS,
            )
            o = cublas_mm(attn_out, L.o, L.o_bias)
            o, residual = self._residual_norm(o, residual, L.norm2)
            moe = self._moe_block(o, pad_rows, L)
            moe, residual = self._residual_norm(moe, residual, L.norm_next)
            x = moe
        return x


class ModelingV2GptOss120bSm10xDefault(
    DecoderModelForCausalLM[GptOssDefaultModelingV2, PretrainedConfig]
):
    def __init__(self, model_config: ModelConfig):
        cfg = model_config.pretrained_config
        # This checkpoint's config.json declares no dtype, so lm_head would be sized fp32 while every
        # other tensor is bf16, failing two layers away.
        if cfg.torch_dtype is None:
            cfg.torch_dtype = model_config.torch_dtype
        super().__init__(
            GptOssDefaultModelingV2(model_config),
            config=model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )

    def load_weights(self, weights, *args, **kwargs):
        _weights.MODEL_WEIGHTS.load(self, weights)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()
