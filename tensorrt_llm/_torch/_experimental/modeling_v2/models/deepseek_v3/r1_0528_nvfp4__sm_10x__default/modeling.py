# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: deepseek-r1-0528-nvfp4 / sm_10x / default -- one forward for every
parallel layout the routing module admits.

DeepSeek-R1-0528, modelopt NVFP4 export: 61 layers, hidden 7168, 128 query heads, layers 0-2
dense and 3-60 MoE (256 routed experts at top-8 plus a shared expert). Only the MLP path is
NVFP4; attention, router, embedding and lm_head are bf16, and the KV cache is fp8-e4m3. Layer
61 is the checkpoint's bf16 MTP module (`MTPLayer`), live only under a `speculative_config`.

The weight shapes are computed from the engine's `Mapping` (weights.py), and the difference
between layouts is settled once, when the targets are built, rather than by branching in the
forward:

* **Attention TP** (attention DP off, world > 1). Each rank runs its share of the 128 MLA
  heads over every token, and its share of each dense MLP and shared expert. o_proj, the
  MLPs and the routed experts leave partial sums, and both residual-add + RMSNorm slots
  are an allreduce with the norm fused in -- the same two allreduces per layer the
  built-in model runs: `comm/allreduce` inside one node, `comm/mnnvl_allreduce`
  when the TP group spans trays. The routed experts are split by moe_tp x moe_ep inside
  the same TP group, so no other collective is needed.

* **Attention DP** (attention DP on, world > 1). Each rank runs every head and the whole
  shared expert over its own requests. Both norm slots are the local
  `flashinfer_fused_add_rmsnorm`, and the routed experts are a round trip: pad to the
  group's largest token count, `comm/allgather`, route and run this rank's experts over
  the gathered set, `comm/reducescatter` back.

The MTP layer (one-model MTP-Eagle) is one more decoder layer, cut per rank like the others,
and runs through the same two layout-dependent blocks: its two residual-norm slots are the
same op, and its routed experts the same round trip around a bf16 expert call. Its
un-normalized output -- what the worker feeds back as the next draft step's hidden state --
is the residual the second slot returns.
"""

import math
from typing import Callable, NamedTuple, cast

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import step_contract_enabled
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (  # noqa: E501
    flashinfer_silu_and_mul,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.load_paged_kv_cache_for_mla import (  # noqa: E501
    load_paged_kv_cache_for_mla,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.mla_rope_append_paged_kv_assign_q import (  # noqa: E501
    mla_rope_append_paged_kv_assign_q,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.mla_rope_generation import (
    mla_rope_generation,
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
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import bmm_out
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.nvfp4_gemm import nvfp4_gemm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (  # noqa: E501
    fp4_block_scale_moe_runner,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (  # noqa: E501
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.speculative import get_spec_worker, should_use_separate_draft_kv_cache
from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.quantization.mode import QuantAlgo

from . import weights as _weights
from .ops.moe.fused_moe import fused_moe

# sm_100 (B200, GB200) and sm_103 (B300, GB300): the same kernels, certified on both.
_SMS = ((10, 0), (10, 3))

_CACHED_CTX_FIELDS = (
    "enable_context_mla_with_cached_kv",
    "ctx_cached_token_indptr",
    "ctx_kv_indptr",
    "max_ctx_seq_len",
    "max_ctx_kv_len",
)


def _build_step_args(md: TrtllmAttentionMetadata) -> dict:
    """Project the prepared metadata onto the batch state both MLA attention calls of one
    forward share."""
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
        attention_window_size=md.max_seq_len,
        rope_max_positions=md.max_seq_len,
        rope_original_max_positions=md.max_seq_len,
        num_contexts=md.num_contexts,
        num_ctx_tokens=md.num_ctx_tokens,
        trtllm_gen_jit_warmup=md.trtllm_gen_jit_warmup,
        max_num_sequences=md.max_num_sequences,
        force_prepare_spec_dec_tree_mask=md.force_prepare_spec_dec_tree_mask,
    )


_KV_RESIDUAL_DIM = 0

# Both kv scale tensors stay None: None is read as a scale of exactly 1.0, which is what this
# checkpoint's k_scale/v_scale hold and the only value the context flavors are correct at.
# The MLA path reads the rope table and nothing else — rope_scale_type, rope_scale, the two
# m-scales and the two position windows are silently ignored, so the YaRN blend must live in
# the table.
_CALL_INERT = dict(
    output_sf=None,
    out_scale=None,
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    attention_sinks=None,
    update_kv_cache=True,
    beam_width=1,
    mask_type=1,
    use_paged_context_fmha=False,
    is_mla_enable=True,
    rope_append=True,
    position_embedding_type=8,
    rope_scale_type=0,
    rope_scale=1.0,
    rope_short_m_scale=1.0,
    rope_long_m_scale=1.0,
    chunked_prefill_buffer_batch_size=1,
    attention_chunk_size=None,
    softmax_stats_tensor=None,
    cache_indirection=None,
    block_ids_per_seq=None,
    max_context_q_len_override=None,
    is_cross=False,
    cross_kv=None,
    relative_attention_bias=None,
    relative_attention_max_distance=0,
    mrope_rotary_cos_sin=None,
    mrope_position_deltas=None,
    helix_position_offsets=None,
    helix_is_inactive_rank=None,
    is_spec_decoding_enabled=False,
    use_spec_decoding=False,
    is_spec_dec_tree=False,
    spec_decoding_generation_lengths=None,
    spec_decoding_position_offsets=None,
    spec_decoding_packed_mask=None,
    spec_decoding_bl_tree_mask_offset=None,
    spec_decoding_bl_tree_mask=None,
    spec_decoding_target_max_draft_tokens=None,
    spec_bl_tree_first_sparse_mask_offset_kv=None,
    sparse_kv_indices=None,
    sparse_kv_offsets=None,
    sparse_attn_indices=None,
    sparse_attn_offsets=None,
    sparse_attn_indices_block_size=0,
    num_sparse_topk=None,
    flash_mla_tile_scheduler_metadata=None,
    flash_mla_num_splits=None,
    # kv_norm_weight must stay None: a weight here folds kv_a_layernorm into the KV kernel,
    # which this target already applies itself — a silent double normalization.
    kv_norm_weight=None,
    kv_norm_eps=1e-6,
    skip_correction_threshold=0.0,
)

_QUANT_MODE_FP8_KV = 128

_SF_VEC = 16
# The MoE runner reads the activation scales as a linear buffer and the dense GEMM reads the
# 128x4-swizzled one; the two have the same byte count whenever num_tokens is a multiple of 128
# — every decode CUDA-graph batch — and the wrong one is then accepted silently.
_SF_LINEAR = False
_SF_SWIZZLED = True
_ROUTING_METHOD_INERT = 1
_ACT_TYPE_SWIGLU = 0
# A different enum from the trtllm-gen runner's above: 5 is Swiglu here, 0 is not a gated type.
_FUSED_MOE_ACT_SWIGLU = 5
# True = `concat(enorm(e), hnorm(h))`, which contradicts the DeepSeek-V3 report's
# `M_k[RMSNorm(h); RMSNorm(e)]`; a flip is silent, only the draft acceptance rate shows it.
_MTP_EMBED_BLOCK_FIRST = True

# The MoE runner's certified token columns stop at 8192. Under attention DP the gathered set
# reaches world * max_num_tokens, so the expert call is chunked; every token is independent
# through routing, quantization and the expert GEMMs, so the chunks re-joined are the whole
# call.
_MOE_MAX_T = 8192


def _yarn_mscale(factor: float, mscale: float) -> float:
    """DeepSeek's YaRN magnitude scaling `m(x)`."""
    if factor <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(factor) + 1.0


#: A residual-norm slot: `(x, residual, weight) -> (normed, residual)`.
ResidualNorm = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple]


def _residual_norm(core: "DeepseekV3DefaultModelingV2") -> ResidualNorm:
    """The op both residual-norm slots of every layer call, chosen once per engine.

    Under attention TP the o_proj and MLP outputs are partial sums, and the allreduce that
    completes them also adds the residual and normalizes: `comm/allreduce` when the TP
    group fits in one tray, `comm/mnnvl_allreduce` when it spans trays, where the IPC
    workspace cannot reach the other tray (as in the built-in model). Under attention DP
    each rank's sum is already whole and the norm is local.

    Building either communicator is collective over the TP group, which is why this runs
    once, from `post_load_weights`, on every rank in the same order.
    """
    mapping = core.model_config.mapping
    eps = core.model_config.pretrained_config.rms_norm_eps
    d = core.dims

    if d.attn_tp == 1:

        def local(x, residual, weight):
            return flashinfer_fused_add_rmsnorm(x, residual, weight, eps)

        return local

    if mapping.is_multi_node():
        mnnvl = mnnvl_workspace_args(
            mapping, max_rows=core.model_config.max_num_tokens, hidden=d.hidden, dtype=d.dtype
        )

        def across_trays(x, residual, weight):
            return mnnvl_allreduce(x, residual, weight, eps, **mnnvl)

        return across_trays

    group = allreduce_group_args(mapping)

    def in_tray(x, residual, weight):
        return allreduce(
            x,
            residual,
            weight,
            scale=None,
            bias=None,
            op=RESIDUAL_RMS_NORM,
            eps=eps,
            trigger_completion_at_end=False,
            **group,
        )

    return in_tray


class DeepseekV3DefaultModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = model_config.pretrained_config
        m = model_config.mapping

        assert torch.cuda.get_device_capability() in _SMS, (
            "target certified on sm_100 and sm_103, running on "
            f"sm_{''.join(map(str, torch.cuda.get_device_capability()))}"
        )
        # The layouts this forward does not implement. Each one changes which ranks hold
        # which rows or heads, so running through would compute a plausible wrong answer
        # rather than fail.
        assert m.pp_size == 1, "pipeline parallelism is not implemented"
        assert m.cp_size == 1, "context parallelism is not implemented"
        assert m.moe_cluster_size == 1, "moe_cluster_size is not implemented"
        assert m.enable_attention_dp or m.attn_tp_size == m.tp_size, (
            f"attention TP {m.attn_tp_size} differs from TP {m.tp_size}"
        )
        assert not getattr(m, "enable_lm_head_tp_in_adp", False), (
            "lm_head TP under attention DP is not implemented"
        )
        kv_algo = model_config.quant_config.kv_cache_quant_algo
        assert kv_algo == QuantAlgo.FP8, f"unsupported KV algo {kv_algo}; fp8 latent pool only"

        # The first-step derivation runs once and always; the contract probe lives on each
        # target, behind its own flag.
        self._derive_pending = True

        self.rank = m.rank
        self.dp_size = m.world_size if m.enable_attention_dp else 1
        self.dp_group = list(range(m.world_size))
        # The communicator side of the attention-DP allgather and reduce-scatter, routed
        # like `allreduce_group_args`: under MPI the plain symbols, with MPI disabled
        # the `_pg` symbols over the TP process group, which spans the DP ranks.
        self.dp_process_group = m.tp_group_pg.boxed() if mpi_disabled() else None

        self.quant_mode = _QUANT_MODE_FP8_KV

        rope_cfg = getattr(cfg, "rope_scaling", None) or getattr(cfg, "rope_parameters", None)
        theta = rope_cfg.get("rope_theta", getattr(cfg, "rope_theta", None))
        self.theta = float(theta)
        self.rope_factor = float(rope_cfg["factor"])
        self.rope_orig_max = int(rope_cfg["original_max_position_embeddings"])
        self.beta_fast = float(rope_cfg["beta_fast"])
        self.beta_slow = float(rope_cfg["beta_slow"])
        mscale = float(rope_cfg["mscale"])
        mscale_all_dim = float(rope_cfg["mscale_all_dim"])
        # YaRN's magnitude correction splits in two: the amplitude multiplies cos/sin, the
        # temperature folds into q_scaling as 1/m^2. Either in the other's place is silently wrong.
        self.rope_amplitude = _yarn_mscale(self.rope_factor, mscale) / _yarn_mscale(
            self.rope_factor, mscale_all_dim
        )
        temperature = _yarn_mscale(self.rope_factor, mscale_all_dim)
        self.q_scaling = 1.0 / (temperature * temperature)
        self._call = dict(_CALL_INERT, quant_mode=self.quant_mode, q_scaling=self.q_scaling)

        self.dims = _weights.MODEL_WEIGHTS.dims(self)
        self.w = _weights.MODEL_WEIGHTS.declare(self.dims)

        self._rope: dict | None = None
        self._rope_positions = 0
        self._side_stream: torch.cuda.Stream | None = None
        self._residual_norm: ResidualNorm | None = None
        self._mtp: _MtpWeights | None = None
        self._cached_ctx = False

    def _rope_tables(self, device, positions: int) -> dict:
        """The duplicated-layout GPT-J rope table the MLA ops read, plus its inverse-frequency
        vector.

        `positions` is a row count, not a model property: every row depends only on its own
        index, so rebuilding at a larger size changes no existing row."""
        rope_dim = self.model_config.pretrained_config.qk_rope_head_dim
        half = rope_dim // 2
        d = torch.arange(half, dtype=torch.float64)
        freq = self.theta ** (2.0 * d / rope_dim)
        two_pi = 2.0 * math.pi
        log_theta = math.log(self.theta)
        low = max(
            0.0,
            math.floor(
                rope_dim
                * math.log(self.rope_orig_max / (self.beta_fast * two_pi))
                / (2.0 * log_theta)
            ),
        )
        high = min(
            rope_dim - 1.0,
            math.ceil(
                rope_dim
                * math.log(self.rope_orig_max / (self.beta_slow * two_pi))
                / (2.0 * log_theta)
            ),
        )
        ramp = ((d - low) / max(high - low, 0.001)).clamp(0.0, 1.0)
        inv = ramp / (self.rope_factor * freq) + (1.0 - ramp) / freq
        ang = torch.arange(positions, dtype=torch.float64)[:, None] * inv[None, :]
        cos, sin = ang.cos() * self.rope_amplitude, ang.sin() * self.rope_amplitude
        table = torch.empty(positions, rope_dim, 2, dtype=torch.float64)
        table[:, :half, 0] = cos
        table[:, half:, 0] = cos
        table[:, :half, 1] = sin
        table[:, half:, 1] = sin
        return {
            "rotary_cos_sin": table.reshape(1, positions * rope_dim * 2)
            .float()
            .to(device)
            .contiguous(),
            "rotary_inv_freq": inv.float().to(device).contiguous(),
            "rope_dim": rope_dim,
            "rope_base": self.theta,
        }

    def post_load_weights(self) -> None:
        """Build the two targets, now that the weights are real.

        Named for the shell hook that drives it. Not an override: that hook lives on
        DecoderModelForCausalLM, and a core derives from DecoderModel, which has none.

        The side stream is made here, once: a CUDA stream created inside a graph capture is
        not capturable. The residual-norm op is chosen here, once, for both targets and the
        MTP layer: under attention TP choosing it builds a communicator, which is
        collective. The rope tables and the MTP operands stay on the core rather than a
        target, because `MTPLayer` reads them whichever target its call routes to.
        """
        cfg = self.model_config.pretrained_config
        device = self.w["final_norm"].device
        self._rope_positions = cfg.max_position_embeddings
        self._rope = self._rope_tables(device, self._rope_positions)
        self._side_stream = torch.cuda.Stream(device=device)
        self._residual_norm = _residual_norm(self)
        if self.dims.mtp_enabled:
            self._mtp = _mtp_views(self)
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

    def _select_target(self, attn_metadata) -> "PrefillTarget | DecodeTarget":
        """Which target runs this step.

        One host-side integer read and nothing else: a decode-only CUDA graph is a
        per-capture constant environment, and a predicate that reached for a second field
        would be reading one this call is not guaranteed to carry. A mixed batch -- context
        and generation rows in one step -- goes to prefill, so prefill is the general case
        and decode the specialization that may assume it has no context rows.
        """
        return self._decode if attn_metadata.num_contexts == 0 else self._prefill

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        md = attn_metadata
        if self._derive_pending:
            # Two things the first step fixes, both deciding what the model computes
            # rather than whether it is well-formed -- so unconditional, not part of the
            # targets' opt-in contract probe. A rope table shorter than the engine's
            # max_seq_len is read out of bounds with no check, and `rope_max_positions`
            # is one of the arguments the MLA path ignores; `speculative_config` inflates
            # max_seq_len past `max_position_embeddings`, so take the engine's own number.
            if md.max_seq_len > self._rope_positions:
                self._rope_positions = md.max_seq_len
                self._rope = self._rope_tables(self.w["final_norm"].device, self._rope_positions)
            self._cached_ctx = all(hasattr(md, n) for n in _CACHED_CTX_FIELDS) and bool(
                md.enable_context_mla_with_cached_kv
            )
            self._derive_pending = False
        return self._select_target(md).forward(
            attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            lora_params=lora_params,
            **kwargs,
        )

    def _dp_pad_rows(self, md, num_tokens: int) -> int:
        """Rows this rank pads its block with before the MoE round trip: up to the largest
        per-rank token count in the group, so both collectives run in their uniform form.
        Zero without attention DP."""
        if self.dp_size == 1:
            return 0
        counts = [int(n) for n in md.all_rank_num_tokens]
        assert len(counts) == self.dp_size, (counts, self.dp_size)
        assert counts[self.rank] == num_tokens, (
            f"rank {self.rank} holds {num_tokens} rows but all_rank_num_tokens "
            f"says {counts[self.rank]} ({counts}); the collectives would gather the wrong split"
        )
        return max(counts) - num_tokens

    def _mtp_kv_layer(self, md) -> int:
        """The MTP layer's index in the KV pool `md` addresses on this call.

        The checkpoint and the engine call the MTP layer `num_hidden_layers`, a global id;
        the attention ops index the pool `md.kv_cache_manager` lays out, by its local
        offset. The two differ by layout. Under attention DP the engine appends the draft
        layer to the target pool, so offset 61 of 62. Under attention TP it gives the draft
        layer a manager of its own (`_should_create_separate_draft_kv_cache` in
        pyexecutor/_util.py), and the spec worker swaps `md.kv_cache_manager` and the
        block offsets to it for the draft loop, so offset 0 of 1 -- and the global id
        reads past the end of `host_kv_cache_pool_mapping`. The manager's own
        `layer_offsets` resolves both, as the built-in attention does
        (`TrtllmAttention.get_local_layer_idx`). Read per call: the same metadata object
        holds the draft manager only inside the draft loop.
        """
        return int(md.kv_cache_manager.layer_offsets[self.dims.num_layers])

    def _mtp_pad_rows(self, all_rank_num_tokens, rows: int) -> int:
        """The MTP layer's `_dp_pad_rows`, on the count list the spec worker hands it.

        Callers must pass the worker's own basis, never `attn_metadata.all_rank_num_tokens`,
        which from draft step 1 onwards is still the trunk's list: at equal byte counts the
        collectives diverge silently on every rank, with no hang. Zero without attention DP.
        """
        if self.dp_size == 1:
            return 0
        counts = [int(n) for n in all_rank_num_tokens]
        assert len(counts) == self.dp_size, (counts, self.dp_size)
        pad = max(counts) - rows
        assert pad >= 0, f"rank {self.rank} holds {rows} MTP rows, more than any rank in {counts}"
        return pad


class _LayerWeights(NamedTuple):
    """One layer's weights at this rank's share, as the views the ops read.

    GEMM operands are `.t()` views; `k_b` / `v_b_t` are the two halves of this rank's
    `kv_b` in the shapes the absorbed-q and expand BMMs take; `norm_next` is the *next*
    layer's input norm (the final norm on the last layer). The MoE fields are None on the
    dense layers.
    """

    qa: torch.Tensor
    q_norm: torch.Tensor
    qb: torch.Tensor
    kva: torch.Tensor
    kv_norm: torch.Tensor
    kvb: torch.Tensor
    k_b: torch.Tensor
    v_b_t: torch.Tensor
    o: torch.Tensor
    norm2: torch.Tensor
    norm_next: torch.Tensor
    mlp_gu_gscale: torch.Tensor
    mlp_gu_w: torch.Tensor
    mlp_gu_s: torch.Tensor
    mlp_gu_alpha: torch.Tensor
    mlp_dn_gscale: torch.Tensor
    mlp_dn_w: torch.Tensor
    mlp_dn_s: torch.Tensor
    mlp_dn_alpha: torch.Tensor
    router: torch.Tensor | None
    router_bias: torch.Tensor | None
    moe_gscale: torch.Tensor | None
    fc1_w: torch.Tensor | None
    fc1_s: torch.Tensor | None
    fc2_w: torch.Tensor | None
    fc2_s: torch.Tensor | None
    out1_scale: torch.Tensor | None
    out1_gate_scale: torch.Tensor | None
    out2_scale: torch.Tensor | None


def _layer_views(core: DeepseekV3DefaultModelingV2) -> list[_LayerWeights]:
    """Every layer's `_LayerWeights`, read off the realized weights."""
    d = core.dims
    w = core.w
    hn = d.heads * d.nope
    window = slice(d.expert_offset, d.expert_offset + d.local_experts)

    layers = []
    for i in range(d.num_layers):
        kvb = w[f"l{i}_kvb"]
        # The checkpoint stores reciprocals (`input_scale` = 1/g_act, `weight_scale_2` =
        # 1/g_w), so the quantizer's global scale is 1/input_scale and the GEMM alpha is
        # their product — no further reciprocal anywhere. Per-tensor scalars, so the same on
        # every rank whatever slice of the MLP it holds.
        isc1 = w[f"l{i}_mlp_isc1"]
        isc2 = w[f"l{i}_mlp_isc2"]
        moe: dict = dict(
            router=None,
            router_bias=None,
            moe_gscale=None,
            fc1_w=None,
            fc1_s=None,
            fc2_w=None,
            fc2_s=None,
            out1_scale=None,
            out1_gate_scale=None,
            out2_scale=None,
        )
        if i >= d.dense_layers:
            # One quantization of the hidden states feeds every expert on every rank, so the
            # routed FC1 activation scale has to be a single value and the same value on all
            # ranks, or the expert windows and intermediate slices no longer sum to the whole
            # layer; the shared expert's `input_scale` is the max over all routed ones, so it
            # is the safe one.
            e_isc2 = w[f"l{i}_e_isc2"][window]
            gate1 = (isc1[0] * w[f"l{i}_e_ws2_1"][window]).contiguous()
            moe = dict(
                router=w[f"l{i}_router"].t(),
                router_bias=w[f"l{i}_router_bias"],
                moe_gscale=(1.0 / isc1).contiguous(),
                fc1_w=w[f"l{i}_fc1_w"],
                fc1_s=w[f"l{i}_fc1_s"].view(torch.float8_e4m3fn),
                fc2_w=w[f"l{i}_fc2_w"],
                fc2_s=w[f"l{i}_fc2_s"].view(torch.float8_e4m3fn),
                out1_scale=(gate1 / e_isc2).contiguous(),
                out1_gate_scale=gate1,
                out2_scale=(e_isc2 * w[f"l{i}_e_ws2_2"][window]).contiguous(),
            )
        layers.append(
            _LayerWeights(
                qa=w[f"l{i}_qa"].t(),
                q_norm=w[f"l{i}_q_norm"],
                qb=w[f"l{i}_qb"].t(),
                kva=w[f"l{i}_kva"].t(),
                kv_norm=w[f"l{i}_kv_norm"],
                kvb=kvb.t(),
                k_b=kvb[:hn].reshape(d.heads, d.nope, d.kv_lora),
                v_b_t=torch.transpose(kvb[hn:].reshape(d.heads, d.v_dim, d.kv_lora), 1, 2),
                o=w[f"l{i}_o"].t(),
                norm2=w[f"l{i}_norm2"],
                norm_next=w[f"l{i + 1}_norm1"] if i + 1 < d.num_layers else w["final_norm"],
                mlp_gu_gscale=(1.0 / isc1).contiguous(),
                mlp_gu_w=w[f"l{i}_mlp_gu_w"],
                mlp_gu_s=w[f"l{i}_mlp_gu_s"],
                mlp_gu_alpha=(isc1 * w[f"l{i}_mlp_ws2_1"]).contiguous(),
                mlp_dn_gscale=(1.0 / isc2).contiguous(),
                mlp_dn_w=w[f"l{i}_mlp_dn_w"],
                mlp_dn_s=w[f"l{i}_mlp_dn_s"],
                mlp_dn_alpha=(isc2 * w[f"l{i}_mlp_ws2_2"]).contiguous(),
                **moe,
            )
        )
    return layers


def _dense_mlp(x: torch.Tensor, L: _LayerWeights, dt) -> torch.Tensor:
    """One NVFP4 SwiGLU MLP over this rank's tokens, used for both the dense layers' MLP and
    every MoE layer's shared expert. Under attention TP it is this rank's slice of the
    intermediate, and the result a partial sum."""
    xq, xsf = fp4_quantize(x, L.mlp_gu_gscale, _SF_VEC, False, _SF_SWIZZLED)
    gu = nvfp4_gemm(xq, L.mlp_gu_w, xsf, L.mlp_gu_s, L.mlp_gu_alpha, dt)
    act = flashinfer_silu_and_mul(gu)
    aq, asf = fp4_quantize(act, L.mlp_dn_gscale, _SF_VEC, False, _SF_SWIZZLED)
    return nvfp4_gemm(aq, L.mlp_dn_w, asf, L.mlp_dn_s, L.mlp_dn_alpha, dt)


def _experts(core: DeepseekV3DefaultModelingV2, L: _LayerWeights, x: torch.Tensor) -> torch.Tensor:
    """Route, quantize and run this rank's experts -- its EP window at its moe_tp slice of
    the intermediate -- over `x`, at most `_MOE_MAX_T` rows a call. The result is this
    rank's partial sum."""
    cfg = core.model_config.pretrained_config
    d = core.dims
    parts = []
    for chunk in torch.split(x, _MOE_MAX_T, 0):
        logits = cublas_mm(chunk, L.router)
        topk_w, topk_ids = noaux_tc_op(
            logits,
            L.router_bias,
            cfg.n_group,
            cfg.topk_group,
            cfg.num_experts_per_tok,
            float(cfg.routed_scaling_factor),
        )
        xq, xsf = fp4_quantize(chunk, L.moe_gscale, _SF_VEC, False, _SF_LINEAR)
        parts.append(
            fp4_block_scale_moe_runner(
                routing_logits=None,
                routing_bias=None,
                hidden_states=xq,
                hidden_states_scale=xsf.view(torch.float8_e4m3fn),
                gemm1_weights=L.fc1_w,
                gemm1_weights_scale=L.fc1_s,
                gemm1_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_weights=L.fc2_w,
                gemm2_weights_scale=L.fc2_s,
                gemm2_bias=None,
                output1_scale_scalar=L.out1_scale,
                output1_scale_gate_scalar=L.out1_gate_scale,
                output2_scale_scalar=L.out2_scale,
                num_experts=d.num_experts,
                top_k=cfg.num_experts_per_tok,
                n_group=None,
                topk_group=None,
                intermediate_size=d.moe_inter,
                local_expert_offset=d.expert_offset,
                local_num_experts=d.local_experts,
                routed_scaling_factor=None,
                routing_method_type=_ROUTING_METHOD_INERT,
                do_finalize=True,
                act_type=_ACT_TYPE_SWIGLU,
                topk_weights=topk_w,
                topk_ids=topk_ids,
            )[0]
        )
    return parts[0] if len(parts) == 1 else torch.cat(parts, 0)


class _MtpWeights(NamedTuple):
    """The MTP layer's weights at this rank's share, as the views the ops read.

    The attention fields are a trunk layer's, at layer `num_hidden_layers`; the experts
    (`fc1`, `fc2`) and the shared expert (`sh_gu`, `sh_dn`) are bf16, `enorm`, `hnorm`,
    `eh` and `head_norm` are the module's own.
    """

    enorm: torch.Tensor
    hnorm: torch.Tensor
    eh: torch.Tensor
    norm1: torch.Tensor
    qa: torch.Tensor
    q_norm: torch.Tensor
    qb: torch.Tensor
    kva: torch.Tensor
    kv_norm: torch.Tensor
    kvb: torch.Tensor
    k_b: torch.Tensor
    v_b_t: torch.Tensor
    o: torch.Tensor
    norm2: torch.Tensor
    router: torch.Tensor
    router_bias: torch.Tensor
    fc1: torch.Tensor
    fc2: torch.Tensor
    sh_gu: torch.Tensor
    sh_dn: torch.Tensor
    head_norm: torch.Tensor


def _mtp_views(core: DeepseekV3DefaultModelingV2) -> _MtpWeights:
    """The MTP layer's `_MtpWeights`, read off the realized weights exactly as a trunk
    layer's are."""
    d = core.dims
    w = core.w
    j = d.num_layers
    hn = d.heads * d.nope
    kvb = w[f"l{j}_kvb"]
    return _MtpWeights(
        enorm=w[f"l{j}_enorm"],
        hnorm=w[f"l{j}_hnorm"],
        eh=w[f"l{j}_eh"].t(),
        norm1=w[f"l{j}_norm1"],
        qa=w[f"l{j}_qa"].t(),
        q_norm=w[f"l{j}_q_norm"],
        qb=w[f"l{j}_qb"].t(),
        kva=w[f"l{j}_kva"].t(),
        kv_norm=w[f"l{j}_kv_norm"],
        kvb=kvb.t(),
        k_b=kvb[:hn].reshape(d.heads, d.nope, d.kv_lora),
        v_b_t=torch.transpose(kvb[hn:].reshape(d.heads, d.v_dim, d.kv_lora), 1, 2),
        o=w[f"l{j}_o"].t(),
        norm2=w[f"l{j}_norm2"],
        router=w[f"l{j}_router"].t(),
        router_bias=w[f"l{j}_router_bias"],
        fc1=w[f"l{j}_fc1"],
        fc2=w[f"l{j}_fc2"],
        sh_gu=w[f"l{j}_sh_gu"].t(),
        sh_dn=w[f"l{j}_sh_dn"].t(),
        head_norm=w[f"l{j}_head_norm"],
    )


def _mtp_shared(x: torch.Tensor, M: _MtpWeights) -> torch.Tensor:
    """The MTP layer's shared expert: one bf16 SwiGLU MLP over this rank's tokens. Under
    attention TP it is this rank's slice of the intermediate, and the result a partial
    sum."""
    return cublas_mm(flashinfer_silu_and_mul(cublas_mm(x, M.sh_gu)), M.sh_dn)


def _mtp_experts(
    core: DeepseekV3DefaultModelingV2, M: _MtpWeights, x: torch.Tensor
) -> torch.Tensor:
    """Route and run this rank's bf16 MTP experts -- its EP window at its moe_tp slice of
    the intermediate -- over `x`, at most `_MOE_MAX_T` rows a call. The result is this
    rank's partial sum, the trunk's `_experts` at a different dtype.

    `fused_moe` rather than the trtllm-gen runner: the checkpoint excludes
    `model.layers.61*` from NVFP4, and the runner takes NVFP4 operands only. It wants fp32
    `token_final_scales`, which is why the router GEMM emits fp32 here and the trunk's
    emits bf16, and it requires a token's top-k ids to be distinct above 256 tokens -- a
    repeat reads out of bounds in `finalizeMoeRoutingKernel` -- which a top-k over expert
    indices gives structurally. Its `tp_size` / `tp_rank` only decide which rank adds the
    FC2 bias, and there is none: the moe_tp split is in the weights' shapes alone."""
    cfg = core.model_config.pretrained_config
    d = core.dims
    parts = []
    for chunk in torch.split(x, _MOE_MAX_T, 0):
        topk_w, topk_ids = noaux_tc_op(
            cublas_mm(chunk, M.router, None, torch.float32),
            M.router_bias,
            cfg.n_group,
            cfg.topk_group,
            cfg.num_experts_per_tok,
            float(cfg.routed_scaling_factor),
        )
        parts.append(
            fused_moe(
                input=chunk,
                token_selected_experts=topk_ids,
                token_final_scales=topk_w,
                fc1_expert_weights=M.fc1,
                fc1_expert_biases=None,
                fc2_expert_weights=M.fc2,
                fc2_expert_biases=None,
                output_dtype=chunk.dtype,
                quant_scales=[],
                tp_size=d.moe_tp,
                tp_rank=d.moe_tp_rank,
                ep_size=d.moe_ep,
                ep_rank=d.moe_ep_rank,
                activation_type=_FUSED_MOE_ACT_SWIGLU,
            )[0]
        )
    return parts[0] if len(parts) == 1 else torch.cat(parts, 0)


#: The expert call a routed block wraps: `(core, weights, x) -> this rank's partial sum`.
#: `_experts` for a trunk layer, `_mtp_experts` for the MTP layer.
Experts = Callable[[DeepseekV3DefaultModelingV2, tuple, torch.Tensor], torch.Tensor]


def _routed_on_every_token(
    core: DeepseekV3DefaultModelingV2,
    experts: Experts,
    weights: tuple,
    o: torch.Tensor,
    pad_rows: int,
) -> torch.Tensor:
    """Every rank holds every token (attention TP): the result is this rank's partial sum
    for the allreduce in the next residual-norm slot to complete."""
    return experts(core, weights, o)


def _routed_attention_dp(
    core: DeepseekV3DefaultModelingV2,
    experts: Experts,
    weights: tuple,
    o: torch.Tensor,
    pad_rows: int,
) -> torch.Tensor:
    """Each rank holds its own tokens: gather them, run this rank's experts over the whole
    set, and sum back to the rows this rank owns.

    Gathering before the router keeps the routing identical on every rank, so each token's
    top-8 ids fall in exactly one expert window per moe_tp slice, and the reduce-scatter
    sums each contribution once."""
    num_tokens = o.shape[0]
    o_pad = o if not pad_rows else nn.functional.pad(o, [0, 0, 0, pad_rows])
    group, pg = core.dp_group, core.dp_process_group
    routed_pad = reducescatter(
        experts(core, weights, allgather(o_pad, None, group, pg)), None, group, pg
    )
    return routed_pad if not pad_rows else routed_pad[:num_tokens]


#: A routed-expert block: `(core, experts, weights, o, pad_rows) -> routed`.
RoutedBlock = Callable[
    [DeepseekV3DefaultModelingV2, Experts, tuple, torch.Tensor, int], torch.Tensor
]


def _routed_block(core: DeepseekV3DefaultModelingV2) -> RoutedBlock:
    """The routed-expert block this layout runs, chosen once per target. The trunk layers
    and the MTP layer share it: only the expert call inside the round trip differs."""
    return _routed_attention_dp if core.dp_size > 1 else _routed_on_every_token


class PrefillTarget:
    """The general case: context rows, and possibly generation rows beside them."""

    def __init__(self, core: DeepseekV3DefaultModelingV2) -> None:
        """Take the per-layer weight views this target's forward reads, once, off the real
        weights, and the two layout-dependent blocks: the residual-norm op both slots call
        and the routed-expert block. The forward calls them and never asks which layout it
        runs."""
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per
        # forward. Each target owns its own flag: the two `step_args` projections differ,
        # so probing one does not prove the other's fields exist.
        self._contract_pending = step_contract_enabled()
        self._norm0 = core.w["l0_norm1"]
        self._layers = _layer_views(core)
        self._residual_norm = core._residual_norm
        self._routed = _routed_block(core)

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
        """The batch state both MLA calls of one step share."""
        return _build_step_args(md)

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
        self._check_step_contract(attn_metadata)

        cfg = core.model_config.pretrained_config
        d = core.dims
        heads = d.heads
        nope = d.nope
        rope_dim = d.rope_dim
        v_dim = d.v_dim
        kv_lora = d.kv_lora
        q_lora = d.q_lora
        qk_dim = d.qk_dim
        lat_dim = d.lat_dim
        eps = cfg.rms_norm_eps
        quant_mode = core.quant_mode
        q_scaling = core.q_scaling

        md = attn_metadata
        rope = core._rope
        call = core._call
        step = self.step_args(md)
        num_ctx = md.num_contexts
        tc = md.num_ctx_tokens
        # The paged-cache address book, shared by the two MLA preprocessing ops and both
        # attention calls. The latent cache holds no heads, so it is the same on every rank
        # whatever share of the heads the rank runs.
        tokens_per_block = md.tokens_per_block
        block_offsets = md.kv_cache_block_offsets
        pool_ptrs = md.host_kv_cache_pool_pointers
        pool_map = md.host_kv_cache_pool_mapping
        window = md.max_seq_len

        if inputs_embeds is None:
            h = nn.functional.embedding(input_ids, core.w["embed"])
        else:
            h = inputs_embeds
        num_tokens = h.shape[0]
        gen = num_tokens - tc
        dt = h.dtype
        dev = h.device
        # Query rows per generation sequence: the only thing that tells the attention ops the
        # query block is taller, and what aligns the bottom-right within-block causal mask.
        gen_seqs = md.num_seqs - num_ctx
        gen_p = gen // gen_seqs if gen_seqs else 1
        ctx_kv_tokens = int(md.host_total_kv_lens[0])
        pad_rows = core._dp_pad_rows(md, num_tokens)
        cached_ctx = core._cached_ctx

        attn_out = torch.empty([num_tokens, heads * v_dim], dtype=dt, device=dev)
        attn_ctx, attn_gen = torch.split(attn_out, [tc, gen], 0)
        # One set of attention scratch for all layers: every tensor here is fully
        # rewritten by layer i's producer before layer i's consumer reads it, and
        # the whole attention path runs on the main stream, so layer i+1 cannot
        # overwrite what layer i still reads. `k`'s rope half is written only in
        # the cached-ctx flavor; the fresh flavor never reads it.
        tkv = ctx_kv_tokens
        k = torch.empty([tkv, heads, qk_dim], dtype=dt, device=dev)
        k_nope_dst, k_pe_dst = torch.split(k, [nope, rope_dim], -1)
        fused_q = torch.empty([gen, heads, lat_dim], dtype=dt, device=dev)
        fq_nope, _ = torch.split(fused_q, [kv_lora, rope_dim], -1)
        fq_nope_t = torch.transpose(fq_nope, 0, 1)
        lat_out = torch.empty([gen, heads * kv_lora], dtype=dt, device=dev)
        lat_out_t = torch.transpose(torch.reshape(lat_out, [gen, heads, kv_lora]), 0, 1)
        attn_gen_t = torch.transpose(torch.reshape(attn_gen, [gen, heads, v_dim]), 0, 1)
        cu_q = torch.empty([gen + 1], dtype=torch.int32, device=dev)
        cu_kv = torch.empty([gen + 1], dtype=torch.int32, device=dev)
        counter = torch.empty([1], dtype=torch.uint32, device=dev)
        quant_q = torch.empty([gen, heads, lat_dim], dtype=torch.float8_e4m3fn, device=dev)
        bmm1_scale = torch.empty([2], dtype=torch.float32, device=dev)
        bmm2_scale = torch.empty([1], dtype=torch.float32, device=dev)

        x = flashinfer_rmsnorm(h, self._norm0, eps)
        residual = h
        for i, L in enumerate(self._layers):
            q = cublas_mm(flashinfer_rmsnorm(cublas_mm(x, L.qa), L.q_norm, eps), L.qb)
            kva = cublas_mm(x, L.kva)
            ckv_raw, k_pe = torch.split(kva, [kv_lora, rope_dim], -1)
            ckv = flashinfer_rmsnorm(ckv_raw, L.kv_norm, eps)
            latent = torch.cat([ckv, k_pe], -1)
            q_ctx, q_gen = torch.split(q, [tc, gen], 0)
            latent_ctx, latent_gen = torch.split(latent, [tc, gen], 0)

            if cached_ctx:
                # The ctx_* metadata fields exist only in this flavor.
                q_ctx = mla_rope_append_paged_kv_assign_q(
                    q=q_ctx,
                    latent_cache=latent_ctx,
                    num_contexts=num_ctx,
                    cu_ctx_cached_kv_lens=md.ctx_cached_token_indptr,
                    cu_seq_lens=md.ctx_kv_indptr,
                    max_input_uncached_seq_len=int(md.max_ctx_seq_len),
                    cos_sin_cache=rope["rotary_cos_sin"],
                    head_num=heads,
                    nope_size=nope,
                    rope_size=rope_dim,
                    lora_size=kv_lora,
                    kv_cache_block_offsets=block_offsets,
                    host_kv_cache_pool_pointers=pool_ptrs,
                    host_kv_cache_pool_mapping=pool_map,
                    kv_scale_orig_quant=None,
                    residual_dim=_KV_RESIDUAL_DIM,
                    layer_idx=i,
                    tokens_per_block=tokens_per_block,
                    attention_window_size=window,
                    beam_width=1,
                    quant_mode=quant_mode,
                )
                ckv_full, k_pe_full = load_paged_kv_cache_for_mla(
                    out_dtype=dt,
                    num_contexts=num_ctx,
                    num_ctx_kv_tokens=ctx_kv_tokens,
                    max_ctx_kv_len=int(md.max_ctx_kv_len),
                    cu_ctx_kv_lens=md.ctx_kv_indptr,
                    kv_cache_block_offsets=block_offsets,
                    host_kv_cache_pool_pointers=pool_ptrs,
                    host_kv_cache_pool_mapping=pool_map,
                    kv_scale_quant_orig=None,
                    layer_idx=i,
                    kv_lora_rank=kv_lora,
                    qk_rope_head_dim=rope_dim,
                    tokens_per_block=tokens_per_block,
                    attention_window_size=window,
                    beam_width=1,
                    quant_mode=quant_mode,
                )
                latent_arg = None
            else:
                ckv_full, _ = torch.split(ckv, [tc, gen], 0)
                k_pe_full = None
                latent_arg = latent_ctx
            # The context FMHA hard-codes V's row stride as the full packed [k_nope | v]
            # width and reads the column block, so V must stay this split view.
            kv = cublas_mm(ckv_full, L.kvb)
            k_nope, v_view = torch.split(kv, [heads * nope, heads * v_dim], -1)
            k_nope_dst.copy_(torch.reshape(k_nope, [tkv, heads, nope]))
            if k_pe_full is not None:
                k_pe_dst.copy_(
                    torch.reshape(k_pe_full, [tkv, 1, rope_dim]).expand([tkv, heads, rope_dim])
                )
            attn_ctx = thop_attention(
                q=q_ctx,
                k=torch.reshape(k, [tkv, heads * qk_dim]),
                v=v_view,
                output=attn_ctx,
                latent_cache=latent_arg,
                q_pe=None,
                local_layer_idx=i,
                is_fused_qkv=False,
                attention_input_type=1,
                num_heads=heads,
                num_kv_heads=heads,
                head_size=qk_dim,
                q_lora_rank=q_lora,
                kv_lora_rank=kv_lora,
                qk_nope_head_dim=nope,
                qk_rope_head_dim=rope_dim,
                v_head_dim=v_dim,
                cu_q_seqlens=None,
                cu_kv_seqlens=None,
                fmha_scheduler_counter=None,
                # The context call's row count is `num_ctx_tokens` whatever the generation
                # phase carries, and the entry certifies the MLA context flavors at 1 only.
                predicted_tokens_per_seq=1,
                **step,
                **rope,
                **call,
            )

            if gen:
                q3 = torch.reshape(q_gen, [gen, heads, qk_dim])
                q_nope, q_pe = torch.split(q3, [nope, rope_dim], -1)
                # Over an fp8 pool `mla_rope_generation` reads this half to build the quantized
                # query, so this BMM must be issued first on the same stream; overlapping them
                # is a silent race (they only write disjoint halves on a bf16 pool).
                fq_nope_t = bmm_out(a=torch.transpose(q_nope, 0, 1), b=L.k_b, out=fq_nope_t)
                fused_q = mla_rope_generation(
                    fused_q=fused_q,
                    q_pe=q_pe,
                    latent_cache=latent_gen,
                    rotary_cos_sin=rope["rotary_cos_sin"],
                    cu_q_seqlens=cu_q,
                    cu_kv_seqlens=cu_kv,
                    fmha_scheduler_counter=counter,
                    mla_bmm1_scale=bmm1_scale,
                    mla_bmm2_scale=bmm2_scale,
                    quant_q_buffer=quant_q,
                    sequence_length=md.kv_lens_cuda_runtime,
                    host_past_key_value_lengths=md.kv_lens_runtime,
                    host_context_lengths=md.prompt_lens_cpu_runtime,
                    num_contexts=num_ctx,
                    kv_cache_block_offsets=block_offsets,
                    host_kv_cache_pool_pointers=pool_ptrs,
                    host_kv_cache_pool_mapping=pool_map,
                    kv_scale_orig_quant=None,
                    kv_scale_quant_orig=None,
                    kv_cache_scale_orig_quant=None,
                    out_scale=None,
                    block_ids_per_seq=None,
                    helix_tensor_params=[None, None],
                    predicted_tokens_per_seq=gen_p,
                    layer_idx=i,
                    num_heads=heads,
                    num_kv_heads=1,
                    head_size=lat_dim,
                    residual_dim=_KV_RESIDUAL_DIM,
                    tokens_per_block=tokens_per_block,
                    attention_window_size=window,
                    beam_width=1,
                    quant_mode=quant_mode,
                    q_scaling=q_scaling,
                    q_lora_rank=q_lora,
                    kv_lora_rank=kv_lora,
                    qk_nope_head_dim=nope,
                    qk_rope_head_dim=rope_dim,
                    v_head_dim=kv_lora,
                    rope_append=True,
                )
                lat_out = thop_attention(
                    q=torch.reshape(fused_q, [gen, heads * lat_dim]),
                    k=None,
                    v=None,
                    output=lat_out,
                    latent_cache=latent_gen,
                    q_pe=q_pe,
                    local_layer_idx=i,
                    is_fused_qkv=True,
                    attention_input_type=2,
                    num_heads=heads,
                    num_kv_heads=1,
                    head_size=lat_dim,
                    q_lora_rank=q_lora,
                    kv_lora_rank=kv_lora,
                    qk_nope_head_dim=nope,
                    qk_rope_head_dim=rope_dim,
                    v_head_dim=kv_lora,
                    cu_q_seqlens=cu_q,
                    cu_kv_seqlens=cu_kv,
                    fmha_scheduler_counter=counter,
                    mla_bmm1_scale=bmm1_scale,
                    mla_bmm2_scale=bmm2_scale,
                    quant_q_buffer=quant_q,
                    predicted_tokens_per_seq=gen_p,
                    **step,
                    **rope,
                    **call,
                )
                attn_gen_t = bmm_out(a=lat_out_t, b=L.v_b_t, out=attn_gen_t)

            o = cublas_mm(attn_out, L.o)
            o, residual = self._residual_norm(o, residual, L.norm2)
            if L.router is None:
                mlp_out = _dense_mlp(o, L, dt)
            else:
                side = core._side_stream
                main = torch.cuda.current_stream()
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    shared = _dense_mlp(o, L, dt)
                routed = self._routed(core, _experts, L, o, pad_rows)
                main.wait_stream(side)
                mlp_out = torch.add(routed, shared)
            mlp_out, residual = self._residual_norm(mlp_out, residual, L.norm_next)
            x = mlp_out
        return x

    def mtp_forward(
        self,
        embed_tokens: torch.Tensor,
        all_rank_num_tokens,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """The draft step: one more decoder layer over the MTP module's operands,
        `core._mtp`, at this rank's share.

        The same mirror of a trunk layer the MTP module is, behind its `eh_proj` front end,
        through the same two layout-dependent blocks. `shared_head` and its norm stay on
        `MTPLayer` -- the worker calls that entry point directly and it reads no phase.
        """
        core = self.core
        cfg = core.model_config.pretrained_config
        d = core.dims
        heads = d.heads
        nope = d.nope
        rope_dim = d.rope_dim
        v_dim = d.v_dim
        kv_lora = d.kv_lora
        q_lora = d.q_lora
        qk_dim = d.qk_dim
        lat_dim = d.lat_dim
        eps = cfg.rms_norm_eps
        quant_mode = core.quant_mode
        q_scaling = core.q_scaling

        M = core._mtp
        # Read every call, not once: the worker rewrites `attn_metadata` in place between
        # draft step 0 and step 1+, and `core._rope` may have grown on the engine's first step.
        rope = core._rope
        call = core._call
        md = attn_metadata
        layer_idx = core._mtp_kv_layer(md)
        step = self.step_args(md)
        rows = hidden_states.shape[0]
        dt = hidden_states.dtype
        dev = hidden_states.device
        num_ctx = md.num_contexts
        tc = md.num_ctx_tokens
        gen = rows - tc
        gen_seqs = md.num_seqs - num_ctx
        gen_p = gen // gen_seqs if gen_seqs else 1
        tokens_per_block = md.tokens_per_block
        block_offsets = md.kv_cache_block_offsets
        pool_ptrs = md.host_kv_cache_pool_pointers
        pool_map = md.host_kv_cache_pool_mapping
        window = md.max_seq_len
        ctx_kv_tokens = int(md.host_total_kv_lens[0])
        pad_rows = core._mtp_pad_rows(all_rank_num_tokens, rows)

        e = nn.functional.embedding(input_ids, embed_tokens)
        en = flashinfer_rmsnorm(e, M.enorm, eps)
        hn = flashinfer_rmsnorm(hidden_states, M.hnorm, eps)
        halves = [en, hn] if _MTP_EMBED_BLOCK_FIRST else [hn, en]
        x = cublas_mm(torch.cat(halves, -1), M.eh)

        residual = x
        xn = flashinfer_rmsnorm(x, M.norm1, eps)
        attn_out = torch.empty([rows, heads * v_dim], dtype=dt, device=dev)
        attn_ctx, attn_gen = torch.split(attn_out, [tc, gen], 0)
        q = cublas_mm(flashinfer_rmsnorm(cublas_mm(xn, M.qa), M.q_norm, eps), M.qb)
        kva = cublas_mm(xn, M.kva)
        ckv_raw, k_pe = torch.split(kva, [kv_lora, rope_dim], -1)
        ckv = flashinfer_rmsnorm(ckv_raw, M.kv_norm, eps)
        latent = torch.cat([ckv, k_pe], -1)
        q_ctx, q_gen = torch.split(q, [tc, gen], 0)
        latent_ctx, latent_gen = torch.split(latent, [tc, gen], 0)

        if core._cached_ctx:
            # The ctx_* metadata fields exist only in this flavor.
            q_ctx = mla_rope_append_paged_kv_assign_q(
                q=q_ctx,
                latent_cache=latent_ctx,
                num_contexts=num_ctx,
                cu_ctx_cached_kv_lens=md.ctx_cached_token_indptr,
                cu_seq_lens=md.ctx_kv_indptr,
                max_input_uncached_seq_len=int(md.max_ctx_seq_len),
                cos_sin_cache=rope["rotary_cos_sin"],
                head_num=heads,
                nope_size=nope,
                rope_size=rope_dim,
                lora_size=kv_lora,
                kv_cache_block_offsets=block_offsets,
                host_kv_cache_pool_pointers=pool_ptrs,
                host_kv_cache_pool_mapping=pool_map,
                kv_scale_orig_quant=None,
                residual_dim=_KV_RESIDUAL_DIM,
                layer_idx=layer_idx,
                tokens_per_block=tokens_per_block,
                attention_window_size=window,
                beam_width=1,
                quant_mode=quant_mode,
            )
            ckv_full, k_pe_full = load_paged_kv_cache_for_mla(
                out_dtype=dt,
                num_contexts=num_ctx,
                num_ctx_kv_tokens=ctx_kv_tokens,
                max_ctx_kv_len=int(md.max_ctx_kv_len),
                cu_ctx_kv_lens=md.ctx_kv_indptr,
                kv_cache_block_offsets=block_offsets,
                host_kv_cache_pool_pointers=pool_ptrs,
                host_kv_cache_pool_mapping=pool_map,
                kv_scale_quant_orig=None,
                layer_idx=layer_idx,
                kv_lora_rank=kv_lora,
                qk_rope_head_dim=rope_dim,
                tokens_per_block=tokens_per_block,
                attention_window_size=window,
                beam_width=1,
                quant_mode=quant_mode,
            )
            latent_arg = None
        else:
            ckv_full, _ = torch.split(ckv, [tc, gen], 0)
            k_pe_full = None
            latent_arg = latent_ctx
        tkv = ctx_kv_tokens
        # The context FMHA hard-codes V's row stride as the full packed [k_nope | v]
        # width and reads the column block, so V must stay this split view.
        kv = cublas_mm(ckv_full, M.kvb)
        k_nope, v_view = torch.split(kv, [heads * nope, heads * v_dim], -1)
        k = torch.empty([tkv, heads, qk_dim], dtype=dt, device=dev)
        k_nope_dst, k_pe_dst = torch.split(k, [nope, rope_dim], -1)
        k_nope_dst.copy_(torch.reshape(k_nope, [tkv, heads, nope]))
        if k_pe_full is not None:
            k_pe_dst.copy_(
                torch.reshape(k_pe_full, [tkv, 1, rope_dim]).expand([tkv, heads, rope_dim])
            )
        attn_ctx = thop_attention(
            q=q_ctx,
            k=torch.reshape(k, [tkv, heads * qk_dim]),
            v=v_view,
            output=attn_ctx,
            latent_cache=latent_arg,
            q_pe=None,
            local_layer_idx=layer_idx,
            is_fused_qkv=False,
            attention_input_type=1,
            num_heads=heads,
            num_kv_heads=heads,
            head_size=qk_dim,
            q_lora_rank=q_lora,
            kv_lora_rank=kv_lora,
            qk_nope_head_dim=nope,
            qk_rope_head_dim=rope_dim,
            v_head_dim=v_dim,
            cu_q_seqlens=None,
            cu_kv_seqlens=None,
            fmha_scheduler_counter=None,
            predicted_tokens_per_seq=1,
            **step,
            **rope,
            **call,
        )

        if gen:
            q3 = torch.reshape(q_gen, [gen, heads, qk_dim])
            q_nope, q_pe = torch.split(q3, [nope, rope_dim], -1)
            fused_q = torch.empty([gen, heads, lat_dim], dtype=dt, device=dev)
            fq_nope, _ = torch.split(fused_q, [kv_lora, rope_dim], -1)
            fq_nope_t = torch.transpose(fq_nope, 0, 1)
            # Over an fp8 pool `mla_rope_generation` reads this half to build the quantized
            # query, so this BMM must be issued first on the same stream.
            fq_nope_t = bmm_out(a=torch.transpose(q_nope, 0, 1), b=M.k_b, out=fq_nope_t)
            cu_q = torch.empty([gen + 1], dtype=torch.int32, device=dev)
            cu_kv = torch.empty([gen + 1], dtype=torch.int32, device=dev)
            counter = torch.empty([1], dtype=torch.uint32, device=dev)
            quant_q = torch.empty([gen, heads, lat_dim], dtype=torch.float8_e4m3fn, device=dev)
            bmm1_scale = torch.empty([2], dtype=torch.float32, device=dev)
            bmm2_scale = torch.empty([1], dtype=torch.float32, device=dev)
            fused_q = mla_rope_generation(
                fused_q=fused_q,
                q_pe=q_pe,
                latent_cache=latent_gen,
                rotary_cos_sin=rope["rotary_cos_sin"],
                cu_q_seqlens=cu_q,
                cu_kv_seqlens=cu_kv,
                fmha_scheduler_counter=counter,
                mla_bmm1_scale=bmm1_scale,
                mla_bmm2_scale=bmm2_scale,
                quant_q_buffer=quant_q,
                sequence_length=md.kv_lens_cuda_runtime,
                host_past_key_value_lengths=md.kv_lens_runtime,
                host_context_lengths=md.prompt_lens_cpu_runtime,
                num_contexts=num_ctx,
                kv_cache_block_offsets=block_offsets,
                host_kv_cache_pool_pointers=pool_ptrs,
                host_kv_cache_pool_mapping=pool_map,
                kv_scale_orig_quant=None,
                kv_scale_quant_orig=None,
                kv_cache_scale_orig_quant=None,
                out_scale=None,
                block_ids_per_seq=None,
                helix_tensor_params=[None, None],
                predicted_tokens_per_seq=gen_p,
                layer_idx=layer_idx,
                num_heads=heads,
                num_kv_heads=1,
                head_size=lat_dim,
                residual_dim=_KV_RESIDUAL_DIM,
                tokens_per_block=tokens_per_block,
                attention_window_size=window,
                beam_width=1,
                quant_mode=quant_mode,
                q_scaling=q_scaling,
                q_lora_rank=q_lora,
                kv_lora_rank=kv_lora,
                qk_nope_head_dim=nope,
                qk_rope_head_dim=rope_dim,
                v_head_dim=kv_lora,
                rope_append=True,
            )
            lat_out = torch.empty([gen, heads * kv_lora], dtype=dt, device=dev)
            lat_out = thop_attention(
                q=torch.reshape(fused_q, [gen, heads * lat_dim]),
                k=None,
                v=None,
                output=lat_out,
                latent_cache=latent_gen,
                q_pe=q_pe,
                local_layer_idx=layer_idx,
                is_fused_qkv=True,
                attention_input_type=2,
                num_heads=heads,
                num_kv_heads=1,
                head_size=lat_dim,
                q_lora_rank=q_lora,
                kv_lora_rank=kv_lora,
                qk_nope_head_dim=nope,
                qk_rope_head_dim=rope_dim,
                v_head_dim=kv_lora,
                cu_q_seqlens=cu_q,
                cu_kv_seqlens=cu_kv,
                fmha_scheduler_counter=counter,
                mla_bmm1_scale=bmm1_scale,
                mla_bmm2_scale=bmm2_scale,
                quant_q_buffer=quant_q,
                predicted_tokens_per_seq=gen_p,
                **step,
                **rope,
                **call,
            )
            lat_out_t = torch.transpose(torch.reshape(lat_out, [gen, heads, kv_lora]), 0, 1)
            attn_gen_t = torch.transpose(torch.reshape(attn_gen, [gen, heads, v_dim]), 0, 1)
            attn_gen_t = bmm_out(a=lat_out_t, b=M.v_b_t, out=attn_gen_t)

        return _mtp_tail(self, M, attn_out, residual, pad_rows)


class DecodeTarget:
    """The specialization: routed to only when there are no context rows.

    Calls four fewer ops than the prefill target -- the context attention flavor, the
    kv_b projection and the two paged-KV helpers all serve a path this target's routing
    guarantees never runs.
    """

    def __init__(self, core: DeepseekV3DefaultModelingV2) -> None:
        """Take the per-layer weight views this target's forward reads, once, off the real
        weights, and the two layout-dependent blocks: the residual-norm op both slots call
        and the routed-expert block. The forward calls them and never asks which layout it
        runs."""
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per
        # forward. Each target owns its own flag: the two `step_args` projections differ,
        # so probing one does not prove the other's fields exist.
        self._contract_pending = step_contract_enabled()
        self._norm0 = core.w["l0_norm1"]
        self._layers = _layer_views(core)
        self._residual_norm = core._residual_norm
        self._routed = _routed_block(core)

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
        """The batch state both MLA calls of one step share.

        States the two context counts instead of reading them back: this target runs
        only when `num_contexts` is zero, so reading them would ask the metadata to
        confirm what the routing already decided.
        """
        step = _build_step_args(md)
        step["num_contexts"] = 0
        step["num_ctx_tokens"] = 0
        return step

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
        self._check_step_contract(attn_metadata)

        cfg = core.model_config.pretrained_config
        d = core.dims
        heads = d.heads
        nope = d.nope
        rope_dim = d.rope_dim
        v_dim = d.v_dim
        kv_lora = d.kv_lora
        q_lora = d.q_lora
        qk_dim = d.qk_dim
        lat_dim = d.lat_dim
        eps = cfg.rms_norm_eps
        quant_mode = core.quant_mode
        q_scaling = core.q_scaling

        md = attn_metadata
        rope = core._rope
        call = core._call
        step = self.step_args(md)
        tokens_per_block = md.tokens_per_block
        block_offsets = md.kv_cache_block_offsets
        pool_ptrs = md.host_kv_cache_pool_pointers
        pool_map = md.host_kv_cache_pool_mapping
        window = md.max_seq_len

        if inputs_embeds is None:
            h = nn.functional.embedding(input_ids, core.w["embed"])
        else:
            h = inputs_embeds
        num_tokens = h.shape[0]
        gen = num_tokens
        dt = h.dtype
        dev = h.device
        # Query rows per generation sequence: the only thing that tells the attention ops the
        # query block is taller, and what aligns the bottom-right within-block causal mask.
        gen_seqs = md.num_seqs
        gen_p = gen // gen_seqs if gen_seqs else 1
        pad_rows = core._dp_pad_rows(md, num_tokens)

        attn_out = torch.empty([num_tokens, heads * v_dim], dtype=dt, device=dev)
        attn_gen = attn_out
        # One set of attention scratch for all layers: every tensor here is fully
        # rewritten by layer i's mla_rope_generation before layer i's attention reads
        # it, and the whole attention path runs on the main stream, so layer i+1 cannot
        # overwrite what layer i still reads.
        fused_q = torch.empty([gen, heads, lat_dim], dtype=dt, device=dev)
        fq_nope, _ = torch.split(fused_q, [kv_lora, rope_dim], -1)
        fq_nope_t = torch.transpose(fq_nope, 0, 1)
        lat_out = torch.empty([gen, heads * kv_lora], dtype=dt, device=dev)
        lat_out_t = torch.transpose(torch.reshape(lat_out, [gen, heads, kv_lora]), 0, 1)
        attn_gen_t = torch.transpose(torch.reshape(attn_gen, [gen, heads, v_dim]), 0, 1)
        cu_q = torch.empty([gen + 1], dtype=torch.int32, device=dev)
        cu_kv = torch.empty([gen + 1], dtype=torch.int32, device=dev)
        counter = torch.empty([1], dtype=torch.uint32, device=dev)
        quant_q = torch.empty([gen, heads, lat_dim], dtype=torch.float8_e4m3fn, device=dev)
        bmm1_scale = torch.empty([2], dtype=torch.float32, device=dev)
        bmm2_scale = torch.empty([1], dtype=torch.float32, device=dev)

        x = flashinfer_rmsnorm(h, self._norm0, eps)
        residual = h
        for i, L in enumerate(self._layers):
            q = cublas_mm(flashinfer_rmsnorm(cublas_mm(x, L.qa), L.q_norm, eps), L.qb)
            kva = cublas_mm(x, L.kva)
            ckv_raw, k_pe = torch.split(kva, [kv_lora, rope_dim], -1)
            ckv = flashinfer_rmsnorm(ckv_raw, L.kv_norm, eps)
            latent = torch.cat([ckv, k_pe], -1)

            q3 = torch.reshape(q, [gen, heads, qk_dim])
            q_nope, q_pe = torch.split(q3, [nope, rope_dim], -1)
            # Over an fp8 pool `mla_rope_generation` reads this half to build the quantized
            # query, so this BMM must be issued first on the same stream; overlapping them
            # is a silent race (they only write disjoint halves on a bf16 pool).
            fq_nope_t = bmm_out(a=torch.transpose(q_nope, 0, 1), b=L.k_b, out=fq_nope_t)
            fused_q = mla_rope_generation(
                fused_q=fused_q,
                q_pe=q_pe,
                latent_cache=latent,
                rotary_cos_sin=rope["rotary_cos_sin"],
                cu_q_seqlens=cu_q,
                cu_kv_seqlens=cu_kv,
                fmha_scheduler_counter=counter,
                mla_bmm1_scale=bmm1_scale,
                mla_bmm2_scale=bmm2_scale,
                quant_q_buffer=quant_q,
                sequence_length=md.kv_lens_cuda_runtime,
                host_past_key_value_lengths=md.kv_lens_runtime,
                host_context_lengths=md.prompt_lens_cpu_runtime,
                num_contexts=0,
                kv_cache_block_offsets=block_offsets,
                host_kv_cache_pool_pointers=pool_ptrs,
                host_kv_cache_pool_mapping=pool_map,
                kv_scale_orig_quant=None,
                kv_scale_quant_orig=None,
                kv_cache_scale_orig_quant=None,
                out_scale=None,
                block_ids_per_seq=None,
                helix_tensor_params=[None, None],
                predicted_tokens_per_seq=gen_p,
                layer_idx=i,
                num_heads=heads,
                num_kv_heads=1,
                head_size=lat_dim,
                residual_dim=_KV_RESIDUAL_DIM,
                tokens_per_block=tokens_per_block,
                attention_window_size=window,
                beam_width=1,
                quant_mode=quant_mode,
                q_scaling=q_scaling,
                q_lora_rank=q_lora,
                kv_lora_rank=kv_lora,
                qk_nope_head_dim=nope,
                qk_rope_head_dim=rope_dim,
                v_head_dim=kv_lora,
                rope_append=True,
            )
            lat_out = thop_attention(
                q=torch.reshape(fused_q, [gen, heads * lat_dim]),
                k=None,
                v=None,
                output=lat_out,
                latent_cache=latent,
                q_pe=q_pe,
                local_layer_idx=i,
                is_fused_qkv=True,
                attention_input_type=2,
                num_heads=heads,
                num_kv_heads=1,
                head_size=lat_dim,
                q_lora_rank=q_lora,
                kv_lora_rank=kv_lora,
                qk_nope_head_dim=nope,
                qk_rope_head_dim=rope_dim,
                v_head_dim=kv_lora,
                cu_q_seqlens=cu_q,
                cu_kv_seqlens=cu_kv,
                fmha_scheduler_counter=counter,
                mla_bmm1_scale=bmm1_scale,
                mla_bmm2_scale=bmm2_scale,
                quant_q_buffer=quant_q,
                predicted_tokens_per_seq=gen_p,
                **step,
                **rope,
                **call,
            )
            attn_gen_t = bmm_out(a=lat_out_t, b=L.v_b_t, out=attn_gen_t)

            o = cublas_mm(attn_out, L.o)
            o, residual = self._residual_norm(o, residual, L.norm2)
            if L.router is None:
                mlp_out = _dense_mlp(o, L, dt)
            else:
                side = core._side_stream
                main = torch.cuda.current_stream()
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    shared = _dense_mlp(o, L, dt)
                routed = self._routed(core, _experts, L, o, pad_rows)
                main.wait_stream(side)
                mlp_out = torch.add(routed, shared)
            mlp_out, residual = self._residual_norm(mlp_out, residual, L.norm_next)
            x = mlp_out
        return x

    def mtp_forward(
        self,
        embed_tokens: torch.Tensor,
        all_rank_num_tokens,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """The draft step over generation rows only: the prefill target's `mtp_forward`
        with the context path dropped."""
        core = self.core
        cfg = core.model_config.pretrained_config
        d = core.dims
        heads = d.heads
        nope = d.nope
        rope_dim = d.rope_dim
        v_dim = d.v_dim
        kv_lora = d.kv_lora
        q_lora = d.q_lora
        qk_dim = d.qk_dim
        lat_dim = d.lat_dim
        eps = cfg.rms_norm_eps
        quant_mode = core.quant_mode
        q_scaling = core.q_scaling

        M = core._mtp
        # Read every call, not once: the worker rewrites `attn_metadata` in place between
        # draft step 0 and step 1+, and `core._rope` may have grown on the engine's first step.
        rope = core._rope
        call = core._call
        md = attn_metadata
        layer_idx = core._mtp_kv_layer(md)
        step = self.step_args(md)
        rows = hidden_states.shape[0]
        dt = hidden_states.dtype
        dev = hidden_states.device
        gen = rows
        gen_seqs = md.num_seqs
        gen_p = gen // gen_seqs if gen_seqs else 1
        tokens_per_block = md.tokens_per_block
        block_offsets = md.kv_cache_block_offsets
        pool_ptrs = md.host_kv_cache_pool_pointers
        pool_map = md.host_kv_cache_pool_mapping
        window = md.max_seq_len
        pad_rows = core._mtp_pad_rows(all_rank_num_tokens, rows)

        e = nn.functional.embedding(input_ids, embed_tokens)
        en = flashinfer_rmsnorm(e, M.enorm, eps)
        hn = flashinfer_rmsnorm(hidden_states, M.hnorm, eps)
        halves = [en, hn] if _MTP_EMBED_BLOCK_FIRST else [hn, en]
        x = cublas_mm(torch.cat(halves, -1), M.eh)

        residual = x
        xn = flashinfer_rmsnorm(x, M.norm1, eps)
        attn_out = torch.empty([rows, heads * v_dim], dtype=dt, device=dev)
        attn_gen = attn_out
        q = cublas_mm(flashinfer_rmsnorm(cublas_mm(xn, M.qa), M.q_norm, eps), M.qb)
        kva = cublas_mm(xn, M.kva)
        ckv_raw, k_pe = torch.split(kva, [kv_lora, rope_dim], -1)
        ckv = flashinfer_rmsnorm(ckv_raw, M.kv_norm, eps)
        latent = torch.cat([ckv, k_pe], -1)

        q3 = torch.reshape(q, [gen, heads, qk_dim])
        q_nope, q_pe = torch.split(q3, [nope, rope_dim], -1)
        fused_q = torch.empty([gen, heads, lat_dim], dtype=dt, device=dev)
        fq_nope, _ = torch.split(fused_q, [kv_lora, rope_dim], -1)
        fq_nope_t = torch.transpose(fq_nope, 0, 1)
        # Over an fp8 pool `mla_rope_generation` reads this half to build the quantized
        # query, so this BMM must be issued first on the same stream.
        fq_nope_t = bmm_out(a=torch.transpose(q_nope, 0, 1), b=M.k_b, out=fq_nope_t)
        cu_q = torch.empty([gen + 1], dtype=torch.int32, device=dev)
        cu_kv = torch.empty([gen + 1], dtype=torch.int32, device=dev)
        counter = torch.empty([1], dtype=torch.uint32, device=dev)
        quant_q = torch.empty([gen, heads, lat_dim], dtype=torch.float8_e4m3fn, device=dev)
        bmm1_scale = torch.empty([2], dtype=torch.float32, device=dev)
        bmm2_scale = torch.empty([1], dtype=torch.float32, device=dev)
        fused_q = mla_rope_generation(
            fused_q=fused_q,
            q_pe=q_pe,
            latent_cache=latent,
            rotary_cos_sin=rope["rotary_cos_sin"],
            cu_q_seqlens=cu_q,
            cu_kv_seqlens=cu_kv,
            fmha_scheduler_counter=counter,
            mla_bmm1_scale=bmm1_scale,
            mla_bmm2_scale=bmm2_scale,
            quant_q_buffer=quant_q,
            sequence_length=md.kv_lens_cuda_runtime,
            host_past_key_value_lengths=md.kv_lens_runtime,
            host_context_lengths=md.prompt_lens_cpu_runtime,
            num_contexts=0,
            kv_cache_block_offsets=block_offsets,
            host_kv_cache_pool_pointers=pool_ptrs,
            host_kv_cache_pool_mapping=pool_map,
            kv_scale_orig_quant=None,
            kv_scale_quant_orig=None,
            kv_cache_scale_orig_quant=None,
            out_scale=None,
            block_ids_per_seq=None,
            helix_tensor_params=[None, None],
            predicted_tokens_per_seq=gen_p,
            layer_idx=layer_idx,
            num_heads=heads,
            num_kv_heads=1,
            head_size=lat_dim,
            residual_dim=_KV_RESIDUAL_DIM,
            tokens_per_block=tokens_per_block,
            attention_window_size=window,
            beam_width=1,
            quant_mode=quant_mode,
            q_scaling=q_scaling,
            q_lora_rank=q_lora,
            kv_lora_rank=kv_lora,
            qk_nope_head_dim=nope,
            qk_rope_head_dim=rope_dim,
            v_head_dim=kv_lora,
            rope_append=True,
        )
        lat_out = torch.empty([gen, heads * kv_lora], dtype=dt, device=dev)
        lat_out = thop_attention(
            q=torch.reshape(fused_q, [gen, heads * lat_dim]),
            k=None,
            v=None,
            output=lat_out,
            latent_cache=latent,
            q_pe=q_pe,
            local_layer_idx=layer_idx,
            is_fused_qkv=True,
            attention_input_type=2,
            num_heads=heads,
            num_kv_heads=1,
            head_size=lat_dim,
            q_lora_rank=q_lora,
            kv_lora_rank=kv_lora,
            qk_nope_head_dim=nope,
            qk_rope_head_dim=rope_dim,
            v_head_dim=kv_lora,
            cu_q_seqlens=cu_q,
            cu_kv_seqlens=cu_kv,
            fmha_scheduler_counter=counter,
            mla_bmm1_scale=bmm1_scale,
            mla_bmm2_scale=bmm2_scale,
            quant_q_buffer=quant_q,
            predicted_tokens_per_seq=gen_p,
            **step,
            **rope,
            **call,
        )
        lat_out_t = torch.transpose(torch.reshape(lat_out, [gen, heads, kv_lora]), 0, 1)
        attn_gen_t = torch.transpose(torch.reshape(attn_gen, [gen, heads, v_dim]), 0, 1)
        attn_gen_t = bmm_out(a=lat_out_t, b=M.v_b_t, out=attn_gen_t)

        return _mtp_tail(self, M, attn_out, residual, pad_rows)


def _mtp_tail(
    target: "PrefillTarget | DecodeTarget",
    M: _MtpWeights,
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    pad_rows: int,
) -> torch.Tensor:
    """The MTP layer from its attention output on: o_proj, the two residual-norm slots and
    the MLP between them, through the target's layout-dependent blocks.

    The second slot is bound with the module's own `shared_head.norm` and its normalized
    output dropped: what the layer returns is the residual that slot completes -- the
    allreduce of the partial MLP sums under attention TP -- **un-normalized**, because the
    worker feeds it back as the next draft step's hidden state and `shared_head` applies
    the norm itself. The shared expert runs before the routed one on the main stream, as
    the module's only MLP call per draft step."""
    core = target.core
    o = cublas_mm(attn_out, M.o)
    o, residual = target._residual_norm(o, residual, M.norm2)
    shared = _mtp_shared(o, M)
    routed = target._routed(core, _mtp_experts, M, o, pad_rows)
    _, residual = target._residual_norm(torch.add(routed, shared), residual, M.head_norm)
    return residual


class MTPLayer:
    """The checkpoint's multi-token-prediction module, at layer index `num_hidden_layers`,
    replayed once per draft step.

    It differs from a trunk layer in three ways the code alone does not show: an `eh_proj`
    front end that mixes the next token's embedding into `h`, bf16 experts (the checkpoint
    excludes `model.layers.61*` from NVFP4, so they go to `fused_moe`, not the trtllm-gen
    runner), and an output returned **un-normalized** -- `shared_head` applies the module's
    own norm, and the runtime feeds the un-normalized tensor back in as the next step's
    `h`."""

    def __init__(self, core: DeepseekV3DefaultModelingV2, logits_processor) -> None:
        self.core = core
        self.logits_processor = logits_processor

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    def shared_head(
        self, hidden_states, lm_head, attn_metadata, return_context_logits
    ) -> torch.Tensor:
        """The module's own output head: its own RMS norm -- a distinct parameter from the
        trunk's `model.norm` -- then the trunk's `lm_head`, vocab-parallel under attention
        TP and whole under attention DP, as the shell built it."""
        core = self.core
        return self.logits_processor.forward(
            flashinfer_rmsnorm(
                hidden_states,
                core._mtp.head_norm,
                core.model_config.pretrained_config.rms_norm_eps,
            ),
            lm_head,
            attn_metadata,
            return_context_logits,
        )

    def forward(
        self,
        embed_tokens: torch.Tensor,
        all_rank_num_tokens,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """Route the draft step by the phase this call's metadata shows.

        Per call, not per engine forward: the worker rewrites `attn_metadata` in place
        between draft step 0 and step 1+, so one engine step's N draft calls can cross
        from prefill to decode.
        """
        return self.core._select_target(attn_metadata).mtp_forward(
            embed_tokens, all_rank_num_tokens, input_ids, hidden_states, attn_metadata, **kwargs
        )


class DraftModel:
    """The container the spec worker reaches this target's drafter through.

    `embed_tokens` must resolve through the core on every read: this container is built while
    every parameter is still a meta tensor, and the engine materializes the registry by
    replacing the tensor objects, so a reference snapshotted here stays on meta."""

    def __init__(self, core: DeepseekV3DefaultModelingV2, lm_head, logits_processor) -> None:
        self.core = core
        self.mtp_layers = [MTPLayer(core, logits_processor)]
        self.lm_head = lm_head

    @property
    def embed_tokens(self) -> torch.Tensor:
        return self.core.w["embed"]


class ModelingV2DeepseekR10528Nvfp4Sm10xDefault(
    DecoderModelForCausalLM[DeepseekV3DefaultModelingV2, PretrainedConfig]
):
    def __init__(self, model_config: ModelConfig):
        cfg = model_config.pretrained_config
        super().__init__(
            DeepseekV3DefaultModelingV2(model_config),
            config=model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )
        self.spec_config = getattr(model_config, "spec_config", None)
        self.draft_model = None
        self.spec_worker = None
        if self.spec_config is not None:
            self.draft_model = DraftModel(self.model, self.lm_head, self.logits_processor)
            # Under attention TP the engine gives the draft layer a KV manager of its own;
            # the worker swaps it into the metadata for the draft loop only when told to.
            self.spec_worker = get_spec_worker(
                self.spec_config,
                model_config,
                model_config.mapping,
                use_separate_draft_kv_cache=should_use_separate_draft_kv_cache(self.spec_config),
            )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        lora_params: dict | None = None,
        **kwargs,
    ):
        """One engine step.

        `resource_manager` must stay inside `**kwargs` -- naming it here drops it from what the
        base forward receives -- and `position_ids` must be handed on in the engine's `[1, T]`
        shape, since the worker squeezes it itself and flattening it here is silently wrong."""
        if self.spec_worker is None:
            return super().forward(
                attn_metadata,
                cast(torch.IntTensor, input_ids),
                position_ids,
                inputs_embeds,
                return_context_logits,
                spec_metadata,
                lora_params,
                **kwargs,
            )
        hidden = self.model(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            lora_params=lora_params,
            **kwargs,
        )
        logits = self.logits_processor.forward(
            nn.functional.embedding(spec_metadata.gather_ids, hidden),
            self.lm_head,
            attn_metadata,
            True,
        )
        return self.spec_worker(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=kwargs.get("resource_manager"),
        )

    def load_weights(self, weights, *args, **kwargs):
        _weights.MODEL_WEIGHTS.load(self, weights)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()
