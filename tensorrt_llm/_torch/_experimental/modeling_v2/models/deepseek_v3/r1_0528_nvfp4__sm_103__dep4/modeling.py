# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: deepseek-r1-0528-nvfp4 / sm_103 / dep4 — self-contained modeling code.

Flat single-entry forward assembled from catalog entries only; every call
that creates or transforms a tensor is a catalog entry, everything else is
tensor-metadata reads and Python control flow.

DeepSeek-R1-0528, modelopt NVFP4 export: 61 layers, hidden 7168, 128 query
heads, `q_lora_rank` 1536, layers 0-2 dense (intermediate 18432), layers 3-60
MoE with 256 routed experts at top-8 plus one shared expert (intermediate
2048). The quantization is `nvfp4_moe_only`-shaped — every `self_attn*` and
`lm_head` is excluded from NVFP4, so attention, router, embedding and lm_head
are bf16 and only the MLP path is NVFP4 — **and the KV cache is fp8-e4m3**
(`hf_quant_config.json`: `kv_cache_quant_algo: FP8`).

The query path is a LoRA pair (`q_lora_rank: 1536`), routing is group-limited
(`n_group 8`, `topk_group 4`), and the rope table is YaRN-scaled with the
attention temperature carried in `q_scaling` rather than in the table.

**Layer 61 — the checkpoint's bf16 MTP module — is a second forward path this
file also carries, and it exists only when `speculative_config` turns it on.**
Under the target's identity config nothing below MTP is declared, layer 61's
790 keys stay a predicted non-load in the weight manifest, and the shell's
forward is a plain call to the inherited base. `MTPLayer` below is the
module's forward; the checkpoint ships no reference implementation of it and
neither does transformers, so what it computes is stated where it is built.

`dep4` is `tensor_parallel_size: 4` plus `moe_expert_parallel_size: 4` plus
`enable_attention_dp: true`: the requests are split, not the heads. Everything
but the MoE is replicated and runs over a rank's own tokens, so there is no
attention-side collective; the MoE stays expert-parallel, rank `r` holding
experts `[64r, 64r+64)` whole, which costs one `comm/allgather` and one
`comm/reducescatter` per MoE layer.

Weights are target-owned: a flat ParameterDict declared here (HF [out, in]
storage so checkpoint rows copy in unchanged, plus the kernel-ready expert
stacks weights.py builds during the load), with the column-major GEMM views,
the MLA absorption operands, the rope table and every NVFP4 call scalar
derived once after load. The registration shell inherits
DecoderModelForCausalLM for lm_head, packed-batch logits gathering, and the
meta-init/load/post-load hooks.

Everything this forward relies on that is not visible in the call itself is
stated at the call: which scales are read and which are inert, why the two
decode producers are ordered rather than concurrent, why the gather precedes
the router GEMM, why the reduce-scatter is crossed in bf16, and which bounds
are certification boundaries rather than tuning knobs.
"""

import math
from typing import cast

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._core import ModelingV2Core
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import advance_step_generation
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (  # noqa: E501
    flashinfer_silu_and_mul,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.load_paged_kv_cache_for_mla import (  # noqa: E501
    LoadPagedKvCacheForMla,
    load_paged_kv_cache_for_mla,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.mla_rope_append_paged_kv_assign_q import (  # noqa: E501
    MlaRopeAppendPagedKvAssignQ,
    mla_rope_append_paged_kv_assign_q,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.mla_rope_generation import (
    MlaRopeGeneration,
    mla_rope_generation,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    ThopAttention,
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allgather import (
    Allgather,
    allgather,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.reducescatter import (
    Reducescatter,
    reducescatter,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import BmmOut, bmm_out
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import CublasMm, cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.nvfp4_gemm import Nvfp4Gemm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (  # noqa: E501
    Fp4BlockScaleMoeRunner,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fused_moe import fused_moe
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import (
    NoauxTcOp,
    noaux_tc_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (  # noqa: E501
    FlashinferFusedAddRmsnorm,
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    FlashinferRmsnorm,
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    Fp4Quantize,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM, register_auto_model
from tensorrt_llm._torch.speculative import get_spec_worker

from . import weights as _weights

# The cached-prefix context group. The engine only creates these attributes
# when it prepares the metadata for MLA context over reused blocks — under
# trtllm's default kv_cache_config that is on, and with block reuse disabled
# they are absent entirely — so their existence selects the context flavor
# rather than being a hard requirement.
_CACHED_CTX_FIELDS = (
    "enable_context_mla_with_cached_kv",
    "ctx_cached_token_indptr",
    "ctx_kv_indptr",
    "max_ctx_seq_len",
    "max_ctx_kv_len",
)


def _build_step_args(md: TrtllmAttentionMetadata) -> dict:
    """Project the prepared metadata onto the batch state both MLA attention
    calls of one forward share; built once per forward. CUDA-graph classes:
    every tensor here is an engine-owned persistent buffer refreshed in place
    (reference class), and every Python int is an engine-construction or
    per-capture constant (host-derived class — a decode-only graph always
    sees num_contexts == 0). Phase-specific arguments (q/k/v, output, head
    geometry, the scheduler buffers, the fp8 decode buffers) are passed at the
    call sites."""
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
        workspace_=md.effective_workspace,
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


# Feature groups this target holds inert, shared by both MLA calls: the
# contract's MLA columns plus every not-certified group at its listed inert
# value. Causal mask, beam width 1, no sinks, no spec-dec mask / sparse /
# cross / relative-bias / mRoPE / helix / FlashMLA. `quant_mode` and
# `q_scaling` are *not* here — both are derived from this checkpoint's config
# in __init__ and joined on in `self._call`. Neither is
# `predicted_tokens_per_seq`: it is 1 on the context call and the generation
# call's own query-tokens-per-sequence, which is 1 for an ordinary decode step
# and `runtime_draft_len + 1` under MTP, so it is passed per call site.
#
# Both kv scale tensors stay None over the fp8 latent pool: the generation
# call reads neither, and both context flavors are correct only at s = 1.0,
# which None is read as exactly. This checkpoint's k_scale/v_scale are 1.0.
#
# position_embedding_type=8 selects the in-kernel GPT-J rope of the MLA path;
# rope_dim / rope_base / the two tables are per-instance. The seven scalars
# below (rope_scale_type, rope_scale, the two m-scales, and the two position
# windows in the step args) are measured inert on the MLA path — the table's
# content is the only rope input the op reads, and the whole YaRN blend lives
# there. They are held at neutral values rather than at the config's `yarn`
# names, whose trtllm enum coding is not derivable from this target's sources.
# The MLA KV pool stores no residual tail: the op accepts 0 or rope_size
# and rejects non-zero unless the pool is FP4, and this checkpoint's is
# fp8-e4m3. The in-tree caller passes a literal 0 on the same path.
_KV_RESIDUAL_DIM = 0

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
    spec_decoding_position_offsets_for_cpp=None,
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
    # Held at the op's defaults, which are also what the in-tree caller passes
    # on this path. kv_norm_weight is not merely a default: non-None would fold
    # the kv_a_layernorm into the KV kernel and make it read latent_cache
    # RAW, and this target normalizes that itself -- passing the weight would
    # normalize twice. skip_correction is a lossy trtllm-gen MLA option
    # (SM100/SM103, off by default upstream); enabling it is a caller's
    # business, not the identity assembly's.
    kv_norm_weight=None,
    kv_norm_eps=1e-6,
    skip_correction_threshold=0.0,
)

# QuantMode's fp8-KV-cache bit — the value every MLA entry certifies for an
# fp8-e4m3 latent pool. Bits outside the KV-cache group ride along unread
# (`1152` = | FP8_1x128_128x128 and `384` = | FP8_QDQ were measured
# bit-identical to a bare 128 on every MLA flavor and on the gather), so the
# derivation below maps the checkpoint's declared KV algo onto this bit rather
# than reconstructing the engine's full bitmask.
_QUANT_MODE_FP8_KV = 128

# NVFP4 block size: one e4m3 scale per 16 contiguous elements along K.
_SF_VEC = 16
# The MoE runner reads the activation scales as a *linear* (row-major)
# buffer; the dense GEMM reads the 128x4-swizzled one. The two have the same
# byte count whenever num_tokens is a multiple of 128 — every decode
# CUDA-graph batch — and the wrong one is then accepted silently, so both
# layouts are spelled out rather than left to the quantizer's default.
_SF_LINEAR = False
_SF_SWIZZLED = True
# Routing already happened outside (noaux_tc_op), so the runner's routing
# method and group configuration are inert on this pre-routed path — including
# this checkpoint's (n_group 8, topk_group 4), measured bitwise identical to
# None/None at routing_method_type 1.
_ROUTING_METHOD_INERT = 1
# Largest token count the expert call is allowed to see in one invocation.
# `fp4_block_scale_moe_runner` is certified over its whole token column — up
# to 8192 — at this checkpoint's routed geometry (H 7168, I 2048, 256 experts
# top-8) for the full stack and for each of the four 64-wide expert-parallel
# windows; `noaux_tc_op` is certified to num_tokens 8192. Under dep4 the
# gathered token set reaches `4 * max_num_tokens` = 32768, so the call is
# chunked rather than run outside those columns. Routing and the activation
# quantization ride along per chunk — both are per-token, so chunking is a
# no-op for them mathematically.
_MOE_MAX_T = 8192
_ACT_TYPE_SWIGLU = 0
# `fused_moe`'s activation enum is a *different* enum from the trtllm-gen
# runner's above: 5 is Swiglu there, 0 is not a gated type at all.
_FUSED_MOE_ACT_SWIGLU = 5
# The MTP layer's `eh_proj` operand order, at one place so flipping it is a
# one-line experiment. True = `concat(enorm(e), hnorm(h))`, the embedding block
# first — measured on this checkpoint's own weights by two independent
# statistics, which contradicts the
# DeepSeek-V3 report's `M_k[RMSNorm(h); RMSNorm(e)]` notation and agrees with
# the parameter's name. Nothing in the checkpoint's metadata pins it and
# nothing downstream detects a flip: the drafts are simply rejected, so the
# acceptance rate is what confirms it end to end.
_MTP_EMBED_BLOCK_FIRST = True


def _moe_chunk_sizes(total: int) -> list[int]:
    """Split `total` gathered rows into expert-call chunks of at most
    `_MOE_MAX_T` rows. Host arithmetic only — no tensor is created."""
    full, rest = divmod(total, _MOE_MAX_T)
    return [_MOE_MAX_T] * full + ([rest] if rest else [])


def _mtp_dp_rows(all_rank_num_tokens, rank: int, dp_size: int, rows: int) -> int:
    """The uniform row count the MTP layer pads its token block to before its
    MoE round trip: `max` over the group's per-rank counts.

    **The list is a parameter and there is no metadata argument, on purpose.**
    Inside the draft loop `attn_metadata.all_rank_num_tokens` is the *trunk's*
    list from draft step 1 onwards — the worker leaves it in place for the
    whole loop and passes the correct basis in as the `all_rank_num_tokens`
    keyword instead (`spec_metadata.all_rank_num_tokens` at step 0, then
    `spec_metadata.subseq_all_rank_num_tokens`, which is the per-rank
    *sequence* count). Both collective contracts certify that calls pair by
    position and that at equal byte counts a divergence is silent — every rank
    wrong in 98-99% of elements, bitwise reproducibly, with no hang — so a
    helper that *could* reach the metadata is the whole hazard. This one
    cannot. `_dp_rows` on the trunk keeps its own shape; the two are not
    interchangeable."""
    counts = [int(n) for n in all_rank_num_tokens]
    return max(counts)


def _yarn_mscale(factor: float, mscale: float) -> float:
    """DeepSeek's YaRN magnitude scaling `m(x)`: 1.0 at or below factor 1,
    `0.1 * x * ln(factor) + 1` above it."""
    if factor <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(factor) + 1.0


class DeepseekV3ModelingV2(ModelingV2Core):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = model_config.pretrained_config

        mapping = model_config.mapping
        self.rank = mapping.rank
        self.ep_rank = mapping.moe_ep_rank
        self.ep_size = mapping.moe_ep_size
        # The collective group: at pp_size 1 with one expert-parallel split,
        # every rank participates in both halves of the MoE round trip, so
        # the group is the whole world.
        self.dp_group = list(range(mapping.world_size))
        self.dp_size = mapping.world_size

        dt = model_config.torch_dtype
        self.quant_mode = _QUANT_MODE_FP8_KV

        # Locals, not attributes: a core carries no forwarded configuration.
        # Every reader of these -- this method's own weight-shape
        # declarations below, weights.py's load(), and the forward methods --
        # already has, or can cheaply reach, `cfg`
        # (`core.model_config.pretrained_config`) and reads it from there
        # directly. A copy onto `self` would only be a second name for the
        # same value. `local_experts`/`expert_offset` further down and the
        # YaRN quantities above are different: they are *derived* (combine
        # two sources, or have real failure modes worth catching once), so
        # they stay attributes -- see the comments there.
        num_layers = cfg.num_hidden_layers
        hidden = cfg.hidden_size
        vocab = cfg.vocab_size
        # The checkpoint ships `num_nextn_predict_layers` extra decoder layers
        # past `num_hidden_layers` for multi-token prediction — on this one a
        # single bf16 layer 61 with its own 256 experts, embedding, eh_proj,
        # two extra norms and an output head.
        self.mtp_layers = int(getattr(cfg, "num_nextn_predict_layers", 0) or 0)
        # Whether this engine drafts. The checkpoint decides the *mode*
        # (`num_nextn_predict_layers: 1` -> MTP-Eagle one-model, one layer
        # replayed `max_draft_len` times); the caller's
        # `speculative_config` decides whether it runs at all, and the engine
        # has already resolved that onto `model_config.spec_config` by the time
        # the model is built. With it absent — the target's identity config —
        # layer 61 is not part of this model at all: its keys stay a predicted
        # non-load in the weight manifest and nothing below is declared.
        spec_config = getattr(model_config, "spec_config", None)
        self.mtp_enabled = spec_config is not None

        # MLA geometry. num_key_value_heads is 128 in this config but MLA has
        # no separate KV heads: the context call runs Hq == Hkv == heads over
        # head_size nope+rope, the generation call runs Hq == heads against a
        # single latent KV head of width kv_lora+rope. Attention is
        # replicated under DP, so every rank runs the whole head set.
        heads = cfg.num_attention_heads
        nope = cfg.qk_nope_head_dim
        rope_dim = cfg.qk_rope_head_dim
        v_dim = cfg.v_head_dim
        kv_lora = cfg.kv_lora_rank
        q_lora = cfg.q_lora_rank
        qk_dim = nope + rope_dim
        lat_dim = kv_lora + rope_dim
        # The query is a LoRA pair here: q_a_proj -> q_a_layernorm -> q_b_proj.
        # A checkpoint with q_lora_rank null projects directly and needs the
        # single-q_proj path instead (that is the deepseek-v3-lite sibling).

        # RoPE: GPT-J interleaved pairs over the rope slice, YaRN-scaled. The
        # engine hands either the flat pre-migration fields or the
        # transformers-5.x rope_parameters dict; both shapes are resolved here.
        rope_cfg = getattr(cfg, "rope_scaling", None) or getattr(cfg, "rope_parameters", None)
        theta = rope_cfg.get("rope_theta", getattr(cfg, "rope_theta", None))
        self.theta = float(theta)
        self.rope_factor = float(rope_cfg["factor"])
        self.rope_orig_max = int(rope_cfg["original_max_position_embeddings"])
        self.beta_fast = float(rope_cfg["beta_fast"])
        self.beta_slow = float(rope_cfg["beta_slow"])
        mscale = float(rope_cfg["mscale"])
        mscale_all_dim = float(rope_cfg["mscale_all_dim"])
        # The table's amplitude and the softmax temperature are the two halves
        # of YaRN's magnitude correction, and they go to different places: the
        # amplitude multiplies cos/sin (exactly 1.0 whenever the config's two
        # m-scales agree), while the temperature — built from mscale_all_dim,
        # as the reference model does — is folded into the op's q_scaling as
        # 1/m^2. Putting either in the other's place is silently wrong.
        self.rope_amplitude = _yarn_mscale(self.rope_factor, mscale) / _yarn_mscale(
            self.rope_factor, mscale_all_dim
        )
        temperature = _yarn_mscale(self.rope_factor, mscale_all_dim)
        self.q_scaling = 1.0 / (temperature * temperature)
        # Per-call constants: the inert groups above plus the two values this
        # checkpoint derives.
        self._call = dict(_CALL_INERT, quant_mode=self.quant_mode, q_scaling=self.q_scaling)

        # MLP structure: the first `first_k_dense_replace` layers are dense,
        # the rest are routed MoE plus a shared-expert pair fused into one
        # dense linear pair of width n_shared * moe_intermediate_size. Both
        # dense shapes are replicated under attention DP and run over this
        # rank's own tokens.
        dense_layers = cfg.first_k_dense_replace
        num_experts = cfg.n_routed_experts
        moe_inter = cfg.moe_intermediate_size
        shared_inter = cfg.moe_intermediate_size * cfg.n_shared_experts
        dense_inter = cfg.intermediate_size
        # Expert parallelism: the routing space stays global and every rank
        # runs the full top-k over the *gathered* token set, but a rank holds
        # only its own window of experts and the kernel drops every slot
        # outside it. The four windows' outputs sum to the whole layer — that
        # is what the reduce-scatter completes.
        self.local_experts = num_experts // mapping.moe_ep_size
        self.expert_offset = self.local_experts * self.ep_rank

        # Weight declaration. HF [out, in] row-major so checkpoint rows copy
        # in unchanged, except kv_b_proj (row-regrouped at load, see
        # weights.py) and the expert stacks (declared in the MoE runner's
        # kernel-ready shuffled/swizzled layout, at the rank's window).
        # Meta-init intercepts torch.empty here — real CUDA storage arrives
        # when the engine materializes the registry.
        def P(*shape, dtype=dt):
            return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)

        u8, f32 = torch.uint8, torch.float32
        w = nn.ParameterDict()
        for i in range(num_layers):
            w[f"l{i}_norm1"] = P(hidden)
            w[f"l{i}_qa"] = P(q_lora, hidden)
            w[f"l{i}_q_norm"] = P(q_lora)
            w[f"l{i}_qb"] = P(heads * qk_dim, q_lora)
            w[f"l{i}_kva"] = P(lat_dim, hidden)
            w[f"l{i}_kv_norm"] = P(kv_lora)
            w[f"l{i}_kvb"] = P(heads * (nope + v_dim), kv_lora)
            w[f"l{i}_o"] = P(hidden, heads * v_dim)
            # The checkpoint's calibrated fp8 KV-cache scales.
            w[f"l{i}_k_scale"] = P(1, dtype=f32)
            w[f"l{i}_v_scale"] = P(1, dtype=f32)
            w[f"l{i}_norm2"] = P(hidden)
            inter = dense_inter if i < dense_layers else shared_inter
            w[f"l{i}_mlp_gu_w"] = P(2 * inter, hidden // 2, dtype=u8)
            w[f"l{i}_mlp_gu_s"] = P(2 * inter * (hidden // _SF_VEC), dtype=u8)
            w[f"l{i}_mlp_dn_w"] = P(hidden, inter // 2, dtype=u8)
            w[f"l{i}_mlp_dn_s"] = P(hidden * (inter // _SF_VEC), dtype=u8)
            for name in (
                "isc1",
                "isc1_up",
                "ws2_1",
                "ws2_1_up",
                "isc2",
                "ws2_2",
            ):
                w[f"l{i}_mlp_{name}"] = P(1, dtype=f32)
            if i < dense_layers:
                continue
            e, mi = self.local_experts, moe_inter
            w[f"l{i}_router"] = P(num_experts, hidden)
            # fp32 on this checkpoint (the reference model keeps the
            # correction bias in fp32 whatever the rest of the weights are);
            # noaux_tc_op takes bf16 logits against an fp32 bias and returns
            # weights in the *logits* dtype, which is what the MoE runner
            # demands.
            w[f"l{i}_router_bias"] = P(num_experts, dtype=f32)
            w[f"l{i}_fc1_w"] = P(e, 2 * mi, hidden // 2, dtype=u8)
            w[f"l{i}_fc1_s"] = P(e, 2 * mi, hidden // _SF_VEC, dtype=u8)
            w[f"l{i}_fc2_w"] = P(e, hidden, mi // 2, dtype=u8)
            w[f"l{i}_fc2_s"] = P(e, hidden, mi // _SF_VEC, dtype=u8)
            # The per-expert NVFP4 scalars stay whole on every rank: they
            # cost 6 floats per expert, and the window is sliced out in
            # derive_after_load, where the kernel's [local_num_experts]
            # operands are built.
            for name in (
                "isc1",
                "isc1_up",
                "ws2_1",
                "ws2_1_up",
                "isc2",
                "ws2_2",
            ):
                w[f"l{i}_e_{name}"] = P(num_experts, dtype=f32)
        w["final_norm"] = P(hidden)
        w["embed"] = P(vocab, hidden)
        # The MTP module at layer index `num_hidden_layers`, declared only when
        # `speculative_config` turned drafting on. Its attention block is
        # byte-identical in geometry to a trunk layer's; its MLP path is the
        # same structure at a different **dtype** — `hf_quant_config.json`
        # carries `model.layers.61*` as one wildcard entry in its
        # `exclude_modules` list, so every weight here is bf16 while the
        # trunk's MLP is NVFP4. That is the export's choice about where
        # accuracy is worth the bytes, so the stacks are declared bf16 and fed
        # to the unquantized `fused_moe` rather than re-quantized at load to
        # reuse the trunk's expert vocabulary. `embed_tokens` and
        # `shared_head.head` are *not* declared: both are bitwise copies of the
        # trunk's `model.embed_tokens` / `lm_head`, and the draft-model
        # container points at those instead — 1.85 GB per rank saved.
        if self.mtp_enabled:
            e, mi = self.local_experts, moe_inter
            w["mtp_enorm"] = P(hidden)
            w["mtp_hnorm"] = P(hidden)
            w["mtp_eh"] = P(hidden, 2 * hidden)
            w["mtp_norm1"] = P(hidden)
            w["mtp_qa"] = P(q_lora, hidden)
            w["mtp_q_norm"] = P(q_lora)
            w["mtp_qb"] = P(heads * qk_dim, q_lora)
            w["mtp_kva"] = P(lat_dim, hidden)
            w["mtp_kv_norm"] = P(kv_lora)
            w["mtp_kvb"] = P(heads * (nope + v_dim), kv_lora)
            w["mtp_o"] = P(hidden, heads * v_dim)
            w["mtp_k_scale"] = P(1, dtype=f32)
            w["mtp_v_scale"] = P(1, dtype=f32)
            w["mtp_norm2"] = P(hidden)
            w["mtp_router"] = P(num_experts, hidden)
            w["mtp_router_bias"] = P(num_experts, dtype=f32)
            # `fused_moe`'s stacked layout: `[E, 2I, H]` with the **up** rows
            # first and the gate rows last (the opposite half order from the
            # dense gate_up linear below, which flashinfer_silu_and_mul reads
            # gate-first), and `[E, H, I]` for FC2. No interleave, no 32-row
            # block shuffle, no swizzle — those belong to the trtllm-gen
            # block-scale runner the trunk uses, not to this one.
            w["mtp_fc1"] = P(e, 2 * mi, hidden)
            w["mtp_fc2"] = P(e, hidden, mi)
            w["mtp_sh_gu"] = P(2 * shared_inter, hidden)
            w["mtp_sh_dn"] = P(hidden, shared_inter)
            w["mtp_head_norm"] = P(hidden)
        self.w = w

        # Every op this target calls is bound against the real weights in
        # derive_after_load(), where meta is over and the tensors are real --
        # not here. `_mtp` stays a dict placeholder for the MTP module's own
        # bound instances, declared only when `mtp_enabled`.
        self._mtp: dict | None = None
        self._rope: dict | None = None
        self._rope_positions = 0
        self._side_stream: torch.cuda.Stream | None = None
        self._cached_ctx = False

    def _rope_tables(self, device, positions: int) -> dict:
        """The duplicated-layout GPT-J rope table the MLA ops read: per
        position, `rope` (cos, sin) pairs whose second half duplicates the
        first, flattened to `[1, positions * rope * 2]` fp32, plus the
        `[rope/2]` inverse-frequency vector from the same construction.

        `positions` is a row count, not a model property: every row depends
        only on its own index, so a longer table is the same table with more
        rows and rebuilding one at a larger size changes no existing row.

        The inverse frequencies carry this checkpoint's **YaRN** blend: the
        interpolated (factor-divided) frequency for the low-frequency half of
        the spectrum, the original one for the high-frequency half, ramped
        between the two correction dimensions. Everything else about the rope
        configuration is inert on the MLA path — the table's content is the
        only rope input the ops read — so this construction is the whole of
        it, and a table built from the unscaled theta is a silently wrong
        model that diverges with position. Built in fp64 on the host, rounded
        once."""
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

    def derive_after_load(self) -> None:
        """Bind every op this target calls, once, against the real weights.

        Per-layer weight tables go into each catalog entry's `bind_layered`,
        one layer at a time, built straight from `self.w` and keeping the
        `.t()` views the hot-path GEMMs already used (zero-copy). Configuration
        comes from `cfg` throughout, read locally here rather than from a core
        attribute -- see `DeepseekV3ModelingV2.__init__`. The rope table and every
        NVFP4 call scalar are derived the same way they always were, folded
        from the checkpoint's per-tensor `input_scale` / `weight_scale_2`
        pairs, the routed ones sliced to this rank's expert window, which is
        where the kernel's `[local_num_experts]` operands come from.

        Layers `[0, first_k_dense_replace)` have no router/MoE operands and
        are skipped for those three entries -- `bind_layered` taking one layer
        at a time is what makes binding only the layers that have a given
        operand possible, instead of padding the gap with a placeholder.

        Meta is over here, so real tensors may be created."""
        cfg = self.model_config.pretrained_config
        num_layers = cfg.num_hidden_layers
        heads = cfg.num_attention_heads
        nope = cfg.qk_nope_head_dim
        rope_dim = cfg.qk_rope_head_dim
        v_dim = cfg.v_head_dim
        kv_lora = cfg.kv_lora_rank
        q_lora = cfg.q_lora_rank
        qk_dim = nope + rope_dim
        lat_dim = kv_lora + rope_dim
        eps = cfg.rms_norm_eps
        dense_layers = cfg.first_k_dense_replace
        num_experts = cfg.n_routed_experts
        topk = cfg.num_experts_per_tok
        moe_inter = cfg.moe_intermediate_size

        w = self.w
        device = w["final_norm"].device
        self._rope_positions = cfg.max_position_embeddings
        self._rope = self._rope_tables(device, self._rope_positions)
        window = slice(self.expert_offset, self.expert_offset + self.local_experts)

        self._rms0 = FlashinferRmsnorm()
        self._rms0.bind_const(weight=w["l0_norm1"], eps=eps)

        self._qa = CublasMm()
        self._q_lora_norm = FlashinferRmsnorm()
        self._q_lora_norm.bind_const(eps=eps)
        self._qb = CublasMm()
        self._kva = CublasMm()
        self._kv_lora_norm = FlashinferRmsnorm()
        self._kv_lora_norm.bind_const(eps=eps)
        self._kvb = CublasMm()
        self._o = CublasMm()
        self._norm2 = FlashinferFusedAddRmsnorm()
        self._norm2.bind_const(eps=eps)
        self._next_norm = FlashinferFusedAddRmsnorm()
        self._next_norm.bind_const(eps=eps)
        self._absorb = BmmOut()
        self._expand = BmmOut()

        # The two MLA attention flavors: context materializes explicit K/V
        # over the up-projected pool rows; generation works in latent space
        # via absorption. Everything genuinely constant across every layer
        # and both phases -- the `_CALL_INERT` feature groups this checkpoint
        # never uses, plus the two values it derives -- is bound here. `step`
        # and the rope table are rebound every forward instead (see
        # `forward`): the former is request-scoped, and the latter can grow
        # on first use (`_check_step_contract`).
        self._attn_ctx = ThopAttention()
        self._attn_ctx.bind_const(
            num_heads=heads,
            num_kv_heads=heads,
            head_size=qk_dim,
            q_lora_rank=q_lora,
            kv_lora_rank=kv_lora,
            qk_nope_head_dim=nope,
            qk_rope_head_dim=rope_dim,
            v_head_dim=v_dim,
            is_fused_qkv=False,
            attention_input_type=1,
            q_pe=None,
            cu_q_seqlens=None,
            cu_kv_seqlens=None,
            fmha_scheduler_counter=None,
            # The context call's row count is `num_ctx_tokens` whatever the
            # generation phase carries, and the entry certifies the MLA
            # context flavors at 1 only.
            predicted_tokens_per_seq=1,
            **_CALL_INERT,
            quant_mode=self.quant_mode,
            q_scaling=self.q_scaling,
        )
        self._attn_gen = ThopAttention()
        self._attn_gen.bind_const(
            num_heads=heads,
            num_kv_heads=1,
            head_size=lat_dim,
            q_lora_rank=q_lora,
            kv_lora_rank=kv_lora,
            qk_nope_head_dim=nope,
            qk_rope_head_dim=rope_dim,
            v_head_dim=kv_lora,
            is_fused_qkv=True,
            attention_input_type=2,
            k=None,
            v=None,
            **_CALL_INERT,
            quant_mode=self.quant_mode,
            q_scaling=self.q_scaling,
        )

        self._mla_append = MlaRopeAppendPagedKvAssignQ()
        self._mla_append.bind_const(
            head_num=heads,
            nope_size=nope,
            rope_size=rope_dim,
            lora_size=kv_lora,
            kv_scale_orig_quant=None,
            residual_dim=_KV_RESIDUAL_DIM,
            beam_width=1,
            quant_mode=self.quant_mode,
        )
        self._mla_load = LoadPagedKvCacheForMla()
        self._mla_load.bind_const(
            kv_scale_quant_orig=None,
            kv_lora_rank=kv_lora,
            qk_rope_head_dim=rope_dim,
            beam_width=1,
            quant_mode=self.quant_mode,
        )
        self._mla_gen = MlaRopeGeneration()
        self._mla_gen.bind_const(
            kv_scale_orig_quant=None,
            kv_scale_quant_orig=None,
            kv_cache_scale_orig_quant=None,
            out_scale=None,
            block_ids_per_seq=None,
            helix_tensor_params=[None, None],
            num_heads=heads,
            num_kv_heads=1,
            head_size=lat_dim,
            residual_dim=_KV_RESIDUAL_DIM,
            beam_width=1,
            quant_mode=self.quant_mode,
            q_scaling=self.q_scaling,
            q_lora_rank=q_lora,
            kv_lora_rank=kv_lora,
            qk_nope_head_dim=nope,
            qk_rope_head_dim=rope_dim,
            v_head_dim=kv_lora,
            rope_append=True,
        )

        # The only two collectives, each a single instance shared by every
        # layer and (once MTP is on) the draft layer too: the group is fixed
        # at construction and `sizes` is always None -- every rank pads to the
        # group-wide row count first, so the split is always even.
        self._allgather = Allgather()
        self._allgather.bind_const(sizes=None, group=self.dp_group)
        self._reducescatter = Reducescatter()
        self._reducescatter.bind_const(sizes=None, group=self.dp_group)

        # Routing already ran outside (`noaux_tc_op`), so the runner's own
        # routing/group configuration is inert on this pre-routed path.
        self._noaux = NoauxTcOp()
        self._noaux.bind_const(
            n_group=cfg.n_group,
            topk_group=cfg.topk_group,
            topk=topk,
            routed_scaling_factor=float(cfg.routed_scaling_factor),
        )
        self._router = CublasMm()

        self._mlp_gu_quant = Fp4Quantize()
        self._mlp_gu_quant.bind_const(
            sf_vec_size=_SF_VEC, sf_use_ue8m0=False, is_sf_swizzled_layout=_SF_SWIZZLED
        )
        self._mlp_gu_gemm = Nvfp4Gemm()
        self._mlp_dn_quant = Fp4Quantize()
        self._mlp_dn_quant.bind_const(
            sf_vec_size=_SF_VEC, sf_use_ue8m0=False, is_sf_swizzled_layout=_SF_SWIZZLED
        )
        self._mlp_dn_gemm = Nvfp4Gemm()

        # The MoE op reads the activation scales as a *linear* buffer, unlike
        # the dense/shared GEMM's swizzled one -- see `_SF_LINEAR`'s comment.
        self._moe_quant = Fp4Quantize()
        self._moe_quant.bind_const(
            sf_vec_size=_SF_VEC, sf_use_ue8m0=False, is_sf_swizzled_layout=_SF_LINEAR
        )
        self._moe_runner = Fp4BlockScaleMoeRunner()
        self._moe_runner.bind_const(
            # Routing already happened outside; the runner's own routing
            # surface rides along inert, and the router bias does not reach
            # this call at all (it rides noaux_tc_op's bias argument instead).
            routing_logits=None,
            routing_bias=None,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_bias=None,
            num_experts=num_experts,
            top_k=topk,
            n_group=None,
            topk_group=None,
            intermediate_size=moe_inter,
            local_expert_offset=self.expert_offset,
            local_num_experts=self.local_experts,
            routed_scaling_factor=None,
            routing_method_type=_ROUTING_METHOD_INERT,
            do_finalize=True,
            act_type=_ACT_TYPE_SWIGLU,
        )

        hn = heads * nope
        for i in range(num_layers):
            kvb = w[f"l{i}_kvb"]
            self._qa.bind_layered(i, mat_b=w[f"l{i}_qa"].t())
            self._q_lora_norm.bind_layered(i, weight=w[f"l{i}_q_norm"])
            self._qb.bind_layered(i, mat_b=w[f"l{i}_qb"].t())
            self._kva.bind_layered(i, mat_b=w[f"l{i}_kva"].t())
            self._kv_lora_norm.bind_layered(i, weight=w[f"l{i}_kv_norm"])
            self._kvb.bind_layered(i, mat_b=kvb.t())
            self._o.bind_layered(i, mat_b=w[f"l{i}_o"].t())
            self._norm2.bind_layered(i, weight=w[f"l{i}_norm2"])
            next_w = w[f"l{i + 1}_norm1"] if i + 1 < num_layers else w["final_norm"]
            self._next_norm.bind_layered(i, weight=next_w)
            # k_b [H, nope, C] absorbs into q_nope; v_b_t [H, C, v] expands
            # the latent attention output. Both are views of the
            # row-regrouped kv_b_proj.
            self._absorb.bind_layered(i, b=kvb[:hn].reshape(heads, nope, kv_lora))
            self._expand.bind_layered(
                i, b=torch.transpose(kvb[hn:].reshape(heads, v_dim, kv_lora), 1, 2)
            )
            self._attn_ctx.bind_layered(i, local_layer_idx=i)
            self._attn_gen.bind_layered(i, local_layer_idx=i)
            self._mla_append.bind_layered(i, layer_idx=i)
            self._mla_load.bind_layered(i, layer_idx=i)
            self._mla_gen.bind_layered(i, layer_idx=i)

            # The checkpoint stores reciprocals: `input_scale = amax/(448*6)
            # = 1/g_act` and `weight_scale_2 = 1/g_w`, so the quantizer's
            # global scale is `1/input_scale` and the GEMM's alpha is their
            # product — both straight off disk, no further reciprocal.
            isc1 = w[f"l{i}_mlp_isc1"]
            isc2 = w[f"l{i}_mlp_isc2"]
            self._mlp_gu_quant.bind_layered(i, global_scale=(1.0 / isc1).contiguous())
            self._mlp_gu_gemm.bind_layered(
                i,
                weight=w[f"l{i}_mlp_gu_w"],
                weight_scale=w[f"l{i}_mlp_gu_s"],
                alpha=(isc1 * w[f"l{i}_mlp_ws2_1"]).contiguous(),
            )
            self._mlp_dn_quant.bind_layered(i, global_scale=(1.0 / isc2).contiguous())
            self._mlp_dn_gemm.bind_layered(
                i,
                weight=w[f"l{i}_mlp_dn_w"],
                weight_scale=w[f"l{i}_mlp_dn_s"],
                alpha=(isc2 * w[f"l{i}_mlp_ws2_2"]).contiguous(),
            )

            if i < dense_layers:
                continue
            self._router.bind_layered(i, mat_b=w[f"l{i}_router"].t())
            self._noaux.bind_layered(i, bias=w[f"l{i}_router_bias"])
            # One quantization of the gathered hidden states feeds every
            # expert on every rank, so the routed FC1 activation scale must
            # be a single value — and the same value everywhere, or the four
            # windows would not sum to the whole layer. The shared expert
            # sees every token where each routed expert sees only its own
            # subset, and the checkpoint's shared-expert input_scale is
            # exactly the max over *all* routed ones — the conservative
            # choice that cannot saturate an activation block.
            e_isc2 = w[f"l{i}_e_isc2"][window]
            gate1 = (w[f"l{i}_mlp_isc1"][0] * w[f"l{i}_e_ws2_1"][window]).contiguous()
            self._moe_quant.bind_layered(i, global_scale=(1.0 / w[f"l{i}_mlp_isc1"]).contiguous())
            self._moe_runner.bind_layered(
                i,
                gemm1_weights=w[f"l{i}_fc1_w"],
                gemm1_weights_scale=w[f"l{i}_fc1_s"].view(torch.float8_e4m3fn),
                gemm2_weights=w[f"l{i}_fc2_w"],
                gemm2_weights_scale=w[f"l{i}_fc2_s"].view(torch.float8_e4m3fn),
                # output1_scale_scalar, output1_scale_gate_scalar,
                # output2_scale_scalar — the FC1 alpha, that alpha times
                # the per-expert FC2 activation global scale, and the FC2
                # alpha, each [local_num_experts]. Swapping the first two
                # is finite and silent.
                output1_scale_scalar=(gate1 / e_isc2).contiguous(),
                output1_scale_gate_scalar=gate1,
                output2_scale_scalar=(e_isc2 * w[f"l{i}_e_ws2_2"][window]).contiguous(),
            )

        if self.mtp_enabled:
            self._mtp = self._derive_mtp()
        # The side stream the shared-expert / dense-MLP branch runs on. Built
        # here because a stream must exist before the first forward: creating
        # one inside a CUDA-graph capture is not a capturable operation, and
        # the runtime's first eager forwards are already past this point.
        self._side_stream = torch.cuda.Stream(device=device)

    def _derive_mtp(self) -> dict:
        """The MTP layer's operand set, derived exactly as a trunk layer's is:
        column-major GEMM views (`.t()` is zero-copy), and the two MLA absorption
        operands split out of the row-regrouped kv_b_proj. No NVFP4 scalars —
        this module is bf16 throughout, so its expert stacks go to `fused_moe`
        as they are stored."""
        cfg = self.model_config.pretrained_config
        heads = cfg.num_attention_heads
        nope = cfg.qk_nope_head_dim
        v_dim = cfg.v_head_dim
        kv_lora = cfg.kv_lora_rank
        w = self.w
        hn = heads * nope
        kvb = w["mtp_kvb"]
        return {
            "enorm": w["mtp_enorm"],
            "hnorm": w["mtp_hnorm"],
            "eh": w["mtp_eh"].t(),
            "norm1": w["mtp_norm1"],
            "qa": w["mtp_qa"].t(),
            "q_norm": w["mtp_q_norm"],
            "qb": w["mtp_qb"].t(),
            "kva": w["mtp_kva"].t(),
            "kv_norm": w["mtp_kv_norm"],
            "k_b": kvb[:hn].reshape(heads, nope, kv_lora),
            "v_b_t": torch.transpose(kvb[hn:].reshape(heads, v_dim, kv_lora), 1, 2),
            "kvb": kvb.t(),
            "o": w["mtp_o"].t(),
            "norm2": w["mtp_norm2"],
            "router": w["mtp_router"].t(),
            "router_bias": w["mtp_router_bias"],
            "fc1": w["mtp_fc1"],
            "fc2": w["mtp_fc2"],
            "sh_gu": w["mtp_sh_gu"].t(),
            "sh_dn": w["mtp_sh_dn"].t(),
            "head_norm": w["mtp_head_norm"],
        }

    def _probe_step_surface(self, md) -> None:
        _build_step_args(md)

    def _after_contract_check(self, md) -> None:
        """Grow the rope table to cover every position the engine admits, and
        fix the context flavor this engine construction runs. Both are fixed
        at engine construction -- once per model instance is sound, run after
        `_probe_step_surface` has confirmed the metadata surface this reads
        still exists."""
        # The rope table must cover every position the engine admits: a short
        # table is read out of bounds with no check, and `rope_max_positions`,
        # the argument that looks like it bounds this, is one of the inert
        # seven. `max_position_embeddings` is the right size under the identity
        # config, but **a speculative_config inflates the engine's max_seq_len
        # past it** — measured 163840 -> 163848 at `max_draft_len: 3`, which is
        # more than the `max_draft_len - 1` extra KV tokens per sequence the
        # runtime reference documents, so the engine's own number is taken
        # rather than a formula. Growing here is exact: every row of the table
        # depends only on its own position, so the rows the identity config
        # uses are bit-identical either way. This runs on the first forward,
        # before any CUDA-graph capture.
        if md.max_seq_len > self._rope_positions:
            self._rope_positions = md.max_seq_len
            self._rope = self._rope_tables(self.w["final_norm"].device, self._rope_positions)
        # Context flavor, fixed at engine construction: with the cached-KV
        # surface present the target runs append -> gather -> up-project ->
        # explicit-K/V FMHA, which serves reused and fresh sequences alike;
        # without it (block reuse off) no context sequence can carry a
        # prefix, and the fresh-prefill flavor with the in-kernel rope and
        # append is the whole context path.
        self._cached_ctx = all(hasattr(md, n) for n in _CACHED_CTX_FIELDS) and bool(
            md.enable_context_mla_with_cached_kv
        )

    def _dp_rows(self, md, num_tokens: int) -> int:
        """The uniform row count every rank pads its token block to before the
        MoE round trip: `max` over the group's per-rank token counts.

        Attention DP gives each rank a different batch by construction, so the
        split is engine state, not something a rank can derive locally:
        `attn_metadata.all_rank_num_tokens` is where the engine publishes it,
        as a plain list of host ints identical on every rank. Every rank
        therefore computes the same maximum, and both collectives run in their
        uniform form (`sizes=None`).

        **Both collectives are run uniform on purpose, and the ragged form is
        not used at all.** The even split is cheaper for the reduce-scatter at
        these row counts, and `sizes` is a host argument baked into a
        CUDA-graph capture, so only the uniform form is replayable.

        CUDA-graph classification: this returns a host int, but under capture
        the counts are uniform (the engine pads the decode batch to a captured
        size on every rank), so the padding is zero rows and the graph
        contains no padding at all — and no host value reaches either
        collective, whose row counts come from tensor shapes the graph
        fixes."""
        counts = [int(n) for n in md.all_rank_num_tokens]
        return max(counts)

    def _dense_mlp(self, x, layer: int, dt):
        """One NVFP4 SwiGLU MLP over this rank's own tokens: fused gate_up
        GEMM, silu_and_mul, down GEMM. Replicated weights, so the result is
        complete — nothing to reduce. Both quantizations emit the
        128x4-swizzled scale buffer nvfp4_gemm consumes. Runs for every
        layer -- the dense layers' own MLP and every MoE layer's shared
        expert -- which is why `_mlp_gu_quant`/`_mlp_gu_gemm`/
        `_mlp_dn_quant`/`_mlp_dn_gemm` are bound for all of them."""
        xq, xsf = self._mlp_gu_quant(x, layer=layer)
        gu = self._mlp_gu_gemm(act_fp4=xq, act_sf=xsf, output_dtype=dt, layer=layer)
        act = flashinfer_silu_and_mul(gu)
        aq, asf = self._mlp_dn_quant(act, layer=layer)
        return self._mlp_dn_gemm(act_fp4=aq, act_sf=asf, output_dtype=dt, layer=layer)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # "A new engine step has begun" is this forward's knowledge, stated
        # once here before anything below binds per-step state: it is what
        # lets validating() catch a later bind_const call running against
        # last step's metadata. `MTPLayer`, below, is replayed once per draft
        # step but runs inside this same engine forward and does not call
        # this itself -- it must not start a new generation -- so calling it
        # here, exactly once, is what keeps that correct without anyone
        # having to remember it there.
        advance_step_generation()

        # Locals, read straight off cfg: see the comment in __init__ for why
        # the core carries none of this as a forwarded attribute.
        cfg = self.model_config.pretrained_config
        heads = cfg.num_attention_heads
        nope = cfg.qk_nope_head_dim
        rope_dim = cfg.qk_rope_head_dim
        v_dim = cfg.v_head_dim
        kv_lora = cfg.kv_lora_rank
        qk_dim = nope + rope_dim
        lat_dim = kv_lora + rope_dim
        num_layers = cfg.num_hidden_layers
        dense_layers = cfg.first_k_dense_replace

        md = attn_metadata
        self._check_step_contract(md)
        # Read after the contract check: that is where a table too short for
        # the engine's admitted max_seq_len is regrown.
        rope = self._rope

        step = _build_step_args(md)
        self._attn_ctx.bind_const(**step, **rope)
        self._attn_gen.bind_const(**step, **rope)
        # The three MLA preprocessing ops' own per-step surface: the same two
        # values `step` carries for thop_attention under their own parameter
        # names, plus the rope table's single column they read. Rebound every
        # forward rather than once, for the same reason gpt_oss rebinds
        # attention_window_size: cheap, and it cannot be known before the KV
        # cache manager resolves `max_seq_len`.
        self._mla_append.bind_const(
            cos_sin_cache=rope["rotary_cos_sin"],
            tokens_per_block=md.tokens_per_block,
            attention_window_size=md.max_seq_len,
        )
        self._mla_load.bind_const(
            tokens_per_block=md.tokens_per_block, attention_window_size=md.max_seq_len
        )
        self._mla_gen.bind_const(
            rotary_cos_sin=rope["rotary_cos_sin"],
            tokens_per_block=md.tokens_per_block,
            attention_window_size=md.max_seq_len,
        )

        num_ctx = md.num_contexts
        tc = md.num_ctx_tokens
        # The paged-cache address book, shared by the two MLA preprocessing
        # ops and both attention calls. Under attention DP each rank owns a
        # pool holding only its own requests' latent rows.
        block_offsets = md.kv_cache_block_offsets
        pool_ptrs = md.host_kv_cache_pool_pointers
        pool_map = md.host_kv_cache_pool_mapping

        if inputs_embeds is None:
            h = nn.functional.embedding(input_ids, self.w["embed"])
        else:
            h = inputs_embeds
        num_tokens = h.shape[0]
        gen = num_tokens - tc
        dt = h.dtype
        dev = h.device
        # Query tokens per generation sequence, which is the MLA generation
        # call's `predicted_tokens_per_seq`. 1 for an ordinary decode step;
        # under MTP a generation request arrives carrying its whole draft
        # chain, `runtime_draft_len + 1` rows, and that count is the *only*
        # thing expressing the taller query block to the attention ops. Derived
        # from the metadata rather than from a spec object, so it costs nothing
        # when MTP is off and is a per-capture host constant when it is on.
        gen_seqs = md.num_seqs - num_ctx
        gen_p = gen // gen_seqs if gen_seqs else 1
        # The context sequences' cached+new latent-KV row count: what the
        # cache gather returns and what the context FMHA attends over.
        ctx_kv_tokens = int(md.host_total_kv_lens[0]) if tc else 0
        # Every rank pads its token block to the group-wide maximum, so both
        # MoE collectives run in their uniform form; the padded rows are
        # sliced off after the reduce-scatter and never reach the residual
        # stream. Zero rows quantize to an all-zero NVFP4 block (scale byte
        # 0x00, zero data) and route like any other token, so they perturb
        # nothing but wasted expert work.
        dp_rows = self._dp_rows(md, num_tokens)
        pad_rows = dp_rows - num_tokens

        attn_out = torch.empty([num_tokens, heads * v_dim], dtype=dt, device=dev)
        attn_ctx, attn_gen = torch.split(attn_out, [tc, gen], 0)
        x = self._rms0(h)
        residual = h
        for i in range(num_layers):
            # The q-LoRA pair: down-project to q_lora_rank, RMS-norm the
            # latent, up-project to the per-head [nope | rope] rows.
            q = self._qb(self._q_lora_norm(self._qa(x, layer=i), layer=i), layer=i)
            kva = self._kva(x, layer=i)
            ckv_raw, k_pe = torch.split(kva, [kv_lora, rope_dim], -1)
            ckv = self._kv_lora_norm(ckv_raw, layer=i)
            latent = torch.cat([ckv, k_pe], -1)
            q_ctx, q_gen = torch.split(q, [tc, gen], 0)
            latent_ctx, latent_gen = torch.split(latent, [tc, gen], 0)

            if tc:
                if self._cached_ctx:
                    # Reuse-capable context: rotate q_pe/k_pe in place and
                    # append this step's latent rows (quantized to e4m3 on the
                    # way into the pool, at the write-side scale — None = 1.0),
                    # read each sequence's whole [cached + new] latent range
                    # back out (dequantized at the read-side scale — None =
                    # 1.0), up-project it, and attend over explicit K/V. One
                    # call serves a batch mixing reused and fresh sequences —
                    # a fresh one is the cached_s == 0 case. A cached prefix
                    # therefore pays fp8 twice: the gather widens the pool's
                    # e4m3 rows to bf16 and the FMHA below quantizes the
                    # up-projected result straight back.
                    self._mla_append(
                        q=q_ctx,
                        latent_cache=latent_ctx,
                        num_contexts=num_ctx,
                        cu_ctx_cached_kv_lens=md.ctx_cached_token_indptr,
                        cu_seq_lens=md.ctx_kv_indptr,
                        max_input_uncached_seq_len=int(md.max_ctx_seq_len),
                        kv_cache_block_offsets=block_offsets,
                        host_kv_cache_pool_pointers=pool_ptrs,
                        host_kv_cache_pool_mapping=pool_map,
                        layer=i,
                    )
                    ckv_full, k_pe_full = self._mla_load(
                        out_dtype=dt,
                        num_contexts=num_ctx,
                        num_ctx_kv_tokens=ctx_kv_tokens,
                        max_ctx_kv_len=int(md.max_ctx_kv_len),
                        cu_ctx_kv_lens=md.ctx_kv_indptr,
                        kv_cache_block_offsets=block_offsets,
                        host_kv_cache_pool_pointers=pool_ptrs,
                        host_kv_cache_pool_mapping=pool_map,
                        layer=i,
                    )
                    latent_arg = None
                else:
                    # Block reuse off: no context sequence can carry a
                    # prefix, so the fresh-prefill flavor does the rope and
                    # the append inside the attention call and the latent
                    # rows never leave registers.
                    ckv_full, _ = torch.split(ckv, [tc, gen], 0)
                    k_pe_full = None
                    latent_arg = latent_ctx
                tkv = ctx_kv_tokens
                # kv is packed [all heads' k_nope | all heads' v]: the context
                # FMHA hard-codes V's row stride as the full packed width and
                # reads the column block, so V must stay this split view.
                kv = self._kvb(ckv_full, layer=i)
                k_nope, v_view = torch.split(kv, [heads * nope, heads * v_dim], -1)
                k = torch.empty([tkv, heads, qk_dim], dtype=dt, device=dev)
                k_nope_dst, k_pe_dst = torch.split(k, [nope, rope_dim], -1)
                k_nope_dst.copy_(torch.reshape(k_nope, [tkv, heads, nope]))
                if k_pe_full is not None:
                    # k_pe came back from the pool already rotated; every
                    # query head shares it. On the fresh-prefill flavor the
                    # rope slice is left uninitialized instead — that call
                    # overwrites it in place from latent_cache.
                    k_pe_dst.copy_(
                        torch.reshape(k_pe_full, [tkv, 1, rope_dim]).expand([tkv, heads, rope_dim])
                    )
                self._attn_ctx(
                    q=q_ctx,
                    k=torch.reshape(k, [tkv, heads * qk_dim]),
                    v=v_view,
                    output=attn_ctx,
                    latent_cache=latent_arg,
                    layer=i,
                )

            if gen:
                q3 = torch.reshape(q_gen, [gen, heads, qk_dim])
                q_nope, q_pe = torch.split(q3, [nope, rope_dim], -1)
                fused_q = torch.empty([gen, heads, lat_dim], dtype=dt, device=dev)
                fq_nope, _ = torch.split(fused_q, [kv_lora, rope_dim], -1)
                # Absorbed q: (q_nope @ W_k_nope) is what the latent-space
                # dot product needs. Over an fp8 pool the next call **reads**
                # this half to build the quantized query, so this BMM must
                # have finished first — the two are issued in this order on
                # one stream, which is what makes that safe. (On a bf16 pool
                # they write disjoint halves and may overlap; assembling from
                # that reading and switching to an fp8 cache is a silent race.)
                self._absorb(
                    a=torch.transpose(q_nope, 0, 1), out=torch.transpose(fq_nope, 0, 1), layer=i
                )
                cu_q = torch.empty([gen + 1], dtype=torch.int32, device=dev)
                cu_kv = torch.empty([gen + 1], dtype=torch.int32, device=dev)
                counter = torch.empty([1], dtype=torch.uint32, device=dev)
                # The fp8 decode triple: the quantized query the FMHA reads
                # instead of fused_q, and the two folded softmax/output scales
                # it takes instead of either kv scale tensor. All three are
                # written by the call below from q_scaling, the MLA dims and
                # the read-side factor (None = 1.0).
                quant_q = torch.empty([gen, heads, lat_dim], dtype=torch.float8_e4m3fn, device=dev)
                bmm1_scale = torch.empty([2], dtype=torch.float32, device=dev)
                bmm2_scale = torch.empty([1], dtype=torch.float32, device=dev)
                self._mla_gen(
                    fused_q=fused_q,
                    q_pe=q_pe,
                    latent_cache=latent_gen,
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
                    predicted_tokens_per_seq=gen_p,
                    layer=i,
                )
                lat_out = torch.empty([gen, heads * kv_lora], dtype=dt, device=dev)
                self._attn_gen(
                    q=torch.reshape(fused_q, [gen, heads * lat_dim]),
                    output=lat_out,
                    latent_cache=latent_gen,
                    q_pe=q_pe,
                    cu_q_seqlens=cu_q,
                    cu_kv_seqlens=cu_kv,
                    fmha_scheduler_counter=counter,
                    mla_bmm1_scale=bmm1_scale,
                    mla_bmm2_scale=bmm2_scale,
                    quant_q_buffer=quant_q,
                    # The whole of drafting, as far as this op is concerned:
                    # the query block is `gen_p` rows per generation sequence,
                    # token-major, and `gen_p` is what produces the
                    # bottom-right-aligned within-block causal mask (draft row
                    # t sees [0, L_g - gen_p + t] and none of its later
                    # siblings). Certified at 1..4 over this fp8 cell.
                    predicted_tokens_per_seq=gen_p,
                    layer=i,
                )
                self._expand(
                    a=torch.transpose(torch.reshape(lat_out, [gen, heads, kv_lora]), 0, 1),
                    out=torch.transpose(torch.reshape(attn_gen, [gen, heads, v_dim]), 0, 1),
                    layer=i,
                )

            # Replicated o_proj over this rank's own tokens: complete as it
            # stands, so the residual stream is updated with no collective.
            o = self._o(attn_out, layer=i)
            self._norm2(o, residual, layer=i)
            if i < dense_layers:
                mlp_out = self._dense_mlp(o, i, dt)
            else:
                # The shared expert and the routed round trip read the same `o`
                # and meet only at the add below, so they are independent — but
                # on one stream the shared expert's five kernels sit in front of
                # a gather that is 94% exclusive on the device. Forking it onto
                # a side stream lets it run inside that window. Both collective
                # contracts certify a side stream joined to the current one on
                # both ends, which is exactly the shape here; the fork/join pair
                # is also what propagates a CUDA-graph capture into the branch
                # and back, so a decode capture records both streams.
                side = self._side_stream
                main = torch.cuda.current_stream()
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    shared = self._dense_mlp(o, i, dt)
                # The expert-parallel round trip. Gathering *before* the
                # router is what keeps the four windows tiling the routing
                # space exactly once: every rank routes the identical full
                # token set, so a token's top-8 ids agree across ranks and
                # each id falls in exactly one window.
                o_pad = (
                    o
                    if not pad_rows
                    else nn.functional.pad(o, [0, 0, 0, pad_rows], mode="constant", value=0.0)
                )
                o_all = self._allgather(o_pad)
                # The expert call is chunked to `_MOE_MAX_T` rows: the
                # gathered set reaches 4 * max_num_tokens and both the runner's
                # and the routing op's certified token columns stop at 8192.
                # Every token is independent through routing, quantization and
                # the expert GEMMs, so the chunks are the whole call, re-joined.
                parts = []
                for chunk in torch.split(o_all, _moe_chunk_sizes(o_all.shape[0]), 0):
                    # Raw bf16 logits: noaux_tc_op applies the sigmoid
                    # itself, and its weight dtype follows the logits, so
                    # bf16 in means the MoE runner's bf16 topk_weights need
                    # no cast (the fp32 correction bias does not change that).
                    logits = self._router(chunk, layer=i)
                    topk_w, topk_ids = self._noaux(logits, layer=i)
                    # A second quantization of the same hidden states: the
                    # MoE runner reads the linear scale buffer as
                    # float8_e4m3fn, never the swizzled one the shared GEMM
                    # above consumed.
                    xq, xsf = self._moe_quant(chunk, layer=i)
                    parts.append(
                        self._moe_runner(
                            hidden_states=xq,
                            hidden_states_scale=xsf.view(torch.float8_e4m3fn),
                            topk_weights=topk_w,
                            topk_ids=topk_ids,
                            layer=i,
                        )[0]
                    )
                routed_all = parts[0] if len(parts) == 1 else torch.cat(parts, 0)
                # The four windows' partials over the whole token set sum to
                # the layer's routed output; the scatter hands this rank back
                # exactly its own rows. bf16 in: the op sums, and it sums
                # float8 as raw bytes.
                routed_pad = self._reducescatter(routed_all)
                routed = (
                    routed_pad
                    if not pad_rows
                    else torch.split(routed_pad, [num_tokens, pad_rows], 0)[0]
                )
                # Join: the add is the first reader of `shared` on this stream,
                # and `shared` stays referenced until then, so the branch's
                # allocations cannot be recycled underneath it.
                main.wait_stream(side)
                mlp_out = torch.add(routed, shared)
            self._next_norm(mlp_out, residual, layer=i)
            x = mlp_out
        return x


class MTPLayer:
    """The checkpoint's multi-token-prediction module, at layer index
    `num_hidden_layers`, replayed once per draft step.

    Structurally one more decoder layer with a front end bolted on that mixes
    in the embedding of the token being predicted:

        e = embed_tokens(input_ids)                    # the NEXT token
        x = eh_proj(concat(enorm(e), hnorm(h)))        # [T, 2H] -> [T, H]
        x = x + MLA(input_layernorm(x))
        x = x + MoE(post_attention_layernorm(x))
        logits = lm_head(shared_head.norm(x))          # `shared_head`, below

    `h` is the hidden state the runtime hands in: the trunk's own output at
    draft step 0, and this layer's own output at every step after. The layer
    returns `x` **un-normalized** — `shared_head` is where the module's own
    norm is applied — and the runtime slices that with its `gather_ids` for the
    logits and feeds it forward as the next step's `h`.

    A deliberate copy of the trunk's layer body rather than a shared helper:
    the trunk's loop is inlined and tuner-specialized, and three things differ
    here anyway — the front end, the layer index (`num_hidden_layers`, the
    extra pool layer the engine adds under a one-model MTP mode), and the MoE
    dtype. The checkpoint excludes `model.layers.61*` from NVFP4 wholesale, so
    this module is bf16 throughout and its routed experts go to `fused_moe`
    where the trunk's go to `fp4_block_scale_moe_runner`.

    Not an `nn.Module`, and `mtp_layers` is a plain list: every weight this
    layer reads lives in the core's ParameterDict, so registering the layer
    again would put the trunk's parameters on a second path through the shell's
    module tree. The runtime never inspects the container beyond `mtp_layers`,
    `embed_tokens`, `lm_head` and an absent `model.d2t`.

    Two arguments of the documented calling convention are accepted and unused,
    for the same reasons the trunk ignores them: `position_ids` (the MLA ops
    take every position from `sequence_length`, not from this tensor) and
    `spec_metadata` (the one thing the layer needs off it, the DP padding
    basis, arrives as the `all_rank_num_tokens` keyword instead)."""

    def __init__(self, core: DeepseekV3ModelingV2, logits_processor) -> None:
        self.core = core
        self.logits_processor = logits_processor

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    def shared_head(
        self, hidden_states, lm_head, attn_metadata, return_context_logits
    ) -> torch.Tensor:
        """The module's own output head: its own RMS norm — a **distinct**
        parameter from the trunk's `model.norm` — then the trunk's `lm_head`,
        which the checkpoint's `shared_head.head` is a bitwise copy of. The
        projection itself is the inherited shell's `logits_processor`, exactly
        as on the non-speculative path."""
        mw = self.core._mtp
        return self.logits_processor.forward(
            flashinfer_rmsnorm(
                hidden_states,
                mw["head_norm"],
                self.core.model_config.pretrained_config.rms_norm_eps,
            ),
            lm_head,
            attn_metadata,
            return_context_logits,
        )

    def _shared_mlp(self, x, mw):
        """The module's shared expert: one bf16 SwiGLU MLP, replicated, over
        this rank's own tokens — the same structure as the trunk's shared
        expert at the same width, without the NVFP4 quantize/dequantize pair.
        `sh_gu` holds the gate rows first, the half order silu_and_mul reads."""
        return cublas_mm(flashinfer_silu_and_mul(cublas_mm(x, mw["sh_gu"])), mw["sh_dn"])

    def _routed_experts(self, x, mw, all_rank_num_tokens, rows, dt):
        """The module's expert-parallel round trip, the trunk's shape at a
        different dtype: pad to the group-wide row count, gather **before** the
        router so all four ranks route the identical token set, run this rank's
        64-expert window, reduce-scatter the partials back.

        Two things differ from the trunk's. The router GEMM emits **fp32**
        logits: `fused_moe` demands fp32 `token_final_scales` where the
        trtllm-gen runner demands bf16, and `noaux_tc_op`'s weight dtype
        follows its logits, so the dtype is chosen here rather than cast later.
        And the expert call takes the global expert ids directly with
        `ep_size`/`ep_rank` shifting this rank's window, where the trtllm-gen
        runner takes an explicit offset/count pair.

        `fused_moe` requires a token's `topk` ids to be **distinct** above 256
        tokens — a repeat reads out of bounds in `finalizeMoeRoutingKernel`,
        faulting or returning ~200-460 ulp of silent garbage — and this call
        drives `T` to 8192. `noaux_tc_op` returns the indices of the top-k
        largest corrected scores, and a top-k over expert indices cannot repeat
        one, so the precondition holds structurally."""
        core = self.core
        cfg = core.model_config.pretrained_config
        dp_rows = _mtp_dp_rows(all_rank_num_tokens, core.rank, core.dp_size, rows)
        pad_rows = dp_rows - rows
        x_pad = (
            x
            if not pad_rows
            else nn.functional.pad(x, [0, 0, 0, pad_rows], mode="constant", value=0.0)
        )
        x_all = allgather(x_pad, None, core.dp_group)
        parts = []
        for chunk in torch.split(x_all, _moe_chunk_sizes(x_all.shape[0]), 0):
            logits = cublas_mm(chunk, mw["router"], None, torch.float32)
            topk_w, topk_ids = noaux_tc_op(
                logits,
                mw["router_bias"],
                cfg.n_group,
                cfg.topk_group,
                cfg.num_experts_per_tok,
                float(cfg.routed_scaling_factor),
            )
            parts.append(
                fused_moe(
                    chunk,
                    topk_ids,
                    topk_w,
                    mw["fc1"],
                    None,
                    mw["fc2"],
                    None,
                    dt,
                    [],
                    ep_size=core.ep_size,
                    ep_rank=core.ep_rank,
                    activation_type=_FUSED_MOE_ACT_SWIGLU,
                )[0]
            )
        routed_all = parts[0] if len(parts) == 1 else torch.cat(parts, 0)
        routed_pad = reducescatter(routed_all, None, core.dp_group)
        return routed_pad if not pad_rows else torch.split(routed_pad, [rows, pad_rows], 0)[0]

    def forward(
        self,
        embed_tokens: torch.Tensor,
        all_rank_num_tokens,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        core = self.core
        cfg = core.model_config.pretrained_config
        heads = cfg.num_attention_heads
        nope = cfg.qk_nope_head_dim
        rope_dim = cfg.qk_rope_head_dim
        v_dim = cfg.v_head_dim
        kv_lora = cfg.kv_lora_rank
        q_lora = cfg.q_lora_rank
        qk_dim = nope + rope_dim
        lat_dim = kv_lora + rope_dim
        eps = cfg.rms_norm_eps
        num_layers = cfg.num_hidden_layers

        mw, rope = core._mtp, core._rope
        md = attn_metadata
        step = _build_step_args(md)
        rows = hidden_states.shape[0]
        dt = hidden_states.dtype
        dev = hidden_states.device
        num_ctx = md.num_contexts
        tc = md.num_ctx_tokens
        gen = rows - tc
        # **Read the phase from the metadata on every call.** The worker
        # rewrites `attn_metadata` in place between draft step 0 and step 1+ —
        # `_seq_lens` filled with 1, `num_contexts` and `num_ctx_tokens` to 0,
        # one token per generation request — and this layer is invoked N times
        # inside one forward, so a value computed on the first call is wrong on
        # the rest (and, under capture, would be frozen wrong at all N
        # positions). The read is host-side and sync-free: `on_update()`
        # recomputes these from the pinned-host `_seq_lens` and the loop
        # triggers it at exactly that boundary. The quotient is
        # `runtime_draft_len + 1` on step 0 and exactly 1 afterwards.
        gen_seqs = md.num_seqs - num_ctx
        gen_p = gen // gen_seqs if gen_seqs else 1
        tokens_per_block = md.tokens_per_block
        block_offsets = md.kv_cache_block_offsets
        pool_ptrs = md.host_kv_cache_pool_pointers
        pool_map = md.host_kv_cache_pool_mapping
        ctx_kv_tokens = int(md.host_total_kv_lens[0]) if tc else 0
        # The engine raises the KV pool's layer count by
        # `num_nextn_predict_layers` under a one-model MTP mode, so this
        # module's attention addresses layer index `num_hidden_layers` in the
        # same single pool the trunk's 61 layers use. Nothing in the target's
        # config stub or manifest declares that.
        layer_idx = num_layers

        e = nn.functional.embedding(input_ids, embed_tokens)
        en = flashinfer_rmsnorm(e, mw["enorm"], eps)
        hn = flashinfer_rmsnorm(hidden_states, mw["hnorm"], eps)
        halves = [en, hn] if _MTP_EMBED_BLOCK_FIRST else [hn, en]
        x = cublas_mm(torch.cat(halves, -1), mw["eh"])

        residual = x
        xn = flashinfer_rmsnorm(x, mw["norm1"], eps)
        attn_out = torch.empty([rows, heads * v_dim], dtype=dt, device=dev)
        attn_ctx, attn_gen = torch.split(attn_out, [tc, gen], 0)
        q = cublas_mm(
            flashinfer_rmsnorm(cublas_mm(xn, mw["qa"]), mw["q_norm"], eps),
            mw["qb"],
        )
        kva = cublas_mm(xn, mw["kva"])
        ckv_raw, k_pe = torch.split(kva, [kv_lora, rope_dim], -1)
        ckv = flashinfer_rmsnorm(ckv_raw, mw["kv_norm"], eps)
        latent = torch.cat([ckv, k_pe], -1)
        q_ctx, q_gen = torch.split(q, [tc, gen], 0)
        latent_ctx, latent_gen = torch.split(latent, [tc, gen], 0)

        if tc:
            # A context request reaches the draft loop at step 0 only, fed
            # `prompt[1:]` with its first accepted token written at the last
            # position — the same row count and the same per-sequence lengths
            # the trunk saw, so the context flavor the trunk settled at its
            # first forward applies unchanged here.
            if core._cached_ctx:
                mla_rope_append_paged_kv_assign_q(
                    q_ctx,
                    latent_ctx,
                    num_ctx,
                    md.ctx_cached_token_indptr,
                    md.ctx_kv_indptr,
                    int(md.max_ctx_seq_len),
                    rope["rotary_cos_sin"],
                    heads,
                    nope,
                    rope_dim,
                    kv_lora,
                    block_offsets,
                    pool_ptrs,
                    pool_map,
                    None,
                    _KV_RESIDUAL_DIM,
                    layer_idx,
                    tokens_per_block,
                    md.max_seq_len,
                    1,
                    core.quant_mode,
                )
                ckv_full, k_pe_full = load_paged_kv_cache_for_mla(
                    dt,
                    num_ctx,
                    ctx_kv_tokens,
                    int(md.max_ctx_kv_len),
                    md.ctx_kv_indptr,
                    block_offsets,
                    pool_ptrs,
                    pool_map,
                    None,
                    layer_idx,
                    kv_lora,
                    rope_dim,
                    tokens_per_block,
                    md.max_seq_len,
                    1,
                    core.quant_mode,
                )
                latent_arg = None
            else:
                ckv_full, _ = torch.split(ckv, [tc, gen], 0)
                k_pe_full = None
                latent_arg = latent_ctx
            tkv = ctx_kv_tokens
            kv = cublas_mm(ckv_full, mw["kvb"])
            k_nope, v_view = torch.split(kv, [heads * nope, heads * v_dim], -1)
            k = torch.empty([tkv, heads, qk_dim], dtype=dt, device=dev)
            k_nope_dst, k_pe_dst = torch.split(k, [nope, rope_dim], -1)
            k_nope_dst.copy_(torch.reshape(k_nope, [tkv, heads, nope]))
            if k_pe_full is not None:
                k_pe_dst.copy_(
                    torch.reshape(k_pe_full, [tkv, 1, rope_dim]).expand([tkv, heads, rope_dim])
                )
            thop_attention(
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
                **core._call,
            )

        if gen:
            q3 = torch.reshape(q_gen, [gen, heads, qk_dim])
            q_nope, q_pe = torch.split(q3, [nope, rope_dim], -1)
            fused_q = torch.empty([gen, heads, lat_dim], dtype=dt, device=dev)
            fq_nope, _ = torch.split(fused_q, [kv_lora, rope_dim], -1)
            bmm_out(torch.transpose(q_nope, 0, 1), mw["k_b"], torch.transpose(fq_nope, 0, 1))
            cu_q = torch.empty([gen + 1], dtype=torch.int32, device=dev)
            cu_kv = torch.empty([gen + 1], dtype=torch.int32, device=dev)
            counter = torch.empty([1], dtype=torch.uint32, device=dev)
            quant_q = torch.empty([gen, heads, lat_dim], dtype=torch.float8_e4m3fn, device=dev)
            bmm1_scale = torch.empty([2], dtype=torch.float32, device=dev)
            bmm2_scale = torch.empty([1], dtype=torch.float32, device=dev)
            mla_rope_generation(
                fused_q,
                q_pe,
                latent_gen,
                rope["rotary_cos_sin"],
                cu_q,
                cu_kv,
                counter,
                bmm1_scale,
                bmm2_scale,
                quant_q,
                md.kv_lens_cuda_runtime,
                md.kv_lens_runtime,
                md.prompt_lens_cpu_runtime,
                num_ctx,
                block_offsets,
                pool_ptrs,
                pool_map,
                None,
                None,
                None,
                None,
                None,
                [None, None],
                gen_p,
                layer_idx,
                heads,
                1,
                lat_dim,
                _KV_RESIDUAL_DIM,
                tokens_per_block,
                md.max_seq_len,
                1,
                core.quant_mode,
                core.q_scaling,
                q_lora,
                kv_lora,
                nope,
                rope_dim,
                v_dim,
                True,
            )
            lat_out = torch.empty([gen, heads * kv_lora], dtype=dt, device=dev)
            thop_attention(
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
                **core._call,
            )
            bmm_out(
                torch.transpose(torch.reshape(lat_out, [gen, heads, kv_lora]), 0, 1),
                mw["v_b_t"],
                torch.transpose(torch.reshape(attn_gen, [gen, heads, v_dim]), 0, 1),
            )

        o = cublas_mm(attn_out, mw["o"])
        flashinfer_fused_add_rmsnorm(o, residual, mw["norm2"], eps)
        shared = self._shared_mlp(o, mw)
        routed = self._routed_experts(o, mw, all_rank_num_tokens, rows, dt)
        # The layer's output is the residual stream itself, un-normalized:
        # `shared_head` owns the module's norm, and the runtime feeds this
        # tensor straight back in as the next draft step's `h`.
        return torch.add(residual, torch.add(routed, shared))


class DraftModel:
    """The container the spec worker reaches this target's drafter through.

    The runtime never constructs it and inspects exactly four names on it:
    `mtp_layers` (only `[0]` is ever indexed — MTP-Eagle replays one layer),
    `embed_tokens` and `lm_head`, which it hands to the layer and to
    `shared_head`, and `model.d2t`, read with a nested `getattr(..., None)` and
    correctly absent here because draft and target share one vocabulary.

    `embed_tokens` is the **trunk's** embedding weight and `lm_head` the
    trunk's head. The checkpoint's `model.layers.61.embed_tokens.weight` and
    `.shared_head.head.weight` are `torch.equal` to those two, so pointing at
    them is exact and saves 1.85 GB per rank; they stay a predicted non-load in
    the weight manifest even with MTP on.

    `embed_tokens` resolves through the core on every read rather than being
    snapshotted here: this container is built in the shell's `__init__`, where
    every parameter is still a **meta** tensor, and the engine materializes the
    registry by replacing those tensor objects. A reference captured at
    construction stays on meta and fails at the first draft step."""

    def __init__(self, core: DeepseekV3ModelingV2, lm_head, logits_processor) -> None:
        self.core = core
        self.mtp_layers = [MTPLayer(core, logits_processor)]
        self.lm_head = lm_head

    @property
    def embed_tokens(self) -> torch.Tensor:
        return self.core.w["embed"]


@register_auto_model("ModelingV2DeepseekR10528Nvfp4Sm103Dep4")
class ModelingV2DeepseekR10528Nvfp4Sm103Dep4(
    DecoderModelForCausalLM[DeepseekV3ModelingV2, PretrainedConfig]
):
    def __init__(self, model_config: ModelConfig):
        cfg = model_config.pretrained_config
        super().__init__(
            DeepseekV3ModelingV2(model_config),
            config=model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )
        # The speculative branch, built only when `speculative_config` asked for
        # it. `spec_config` is already resolved on the model config by the time
        # the model is built, and the checkpoint — not the target — picked the
        # mode: `num_nextn_predict_layers: 1` gives MTP-Eagle one-model, where
        # `max_draft_len` is a serving knob rather than a checkpoint property.
        #
        # The branch is written out rather than inherited from the in-tree
        # one-engine shell on purpose: that shell builds its drafter through a
        # module-level function with no override point, which dispatches on the
        # config's `model_type` — still the upstream family name, since the
        # stub config patches `architectures` only — and would construct
        # trtllm's own MTP layer instead of this target's.
        self.spec_config = getattr(model_config, "spec_config", None)
        self.draft_model = None
        self.spec_worker = None
        if self.spec_config is not None:
            self.draft_model = DraftModel(self.model, self.lm_head, self.logits_processor)
            self.spec_worker = get_spec_worker(self.spec_config, model_config, model_config.mapping)

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

        Without a spec worker this **delegates to the inherited base forward**
        with every argument it was handed, rather than reimplementing it: the
        release criterion was measured on that base forward, and calling it is
        the only way to keep this path bit-identical. The parameter list
        mirrors the base's exactly for that reason, and `resource_manager`
        deliberately stays inside `**kwargs` — naming it here would drop it
        from what the base receives.

        With one, the shell owns the logits gather: every one-model mode sets
        `without_logits`, so the engine applies no second gather and the worker
        needs the trunk's hidden states **ungathered** beside the gathered
        logits. `position_ids` is handed on in the engine's `[1, T]` shape —
        the worker squeezes it itself, and flattening here would produce a
        silently wrong draft position sequence."""
        if self.spec_worker is None:
            return super().forward(
                attn_metadata,
                # A typing no-op: the base declares `input_ids: torch.IntTensor
                # = None`, a non-Optional annotation with a None default, so
                # the value is forwarded exactly as received — None included.
                cast(torch.IntTensor, input_ids),
                position_ids,
                inputs_embeds,
                return_context_logits,
                spec_metadata,
                lora_params,
                **kwargs,
            )
        # `spec_metadata` is not forwarded into the core: the trunk genuinely
        # does not read it — on MTP_EAGLE_ONE_MODEL the runtime sets
        # `layers_to_capture = ()`, so `is_layer_capture()` is False everywhere
        # and no hidden-state capture hook is owed (that is Eagle3's
        # requirement, not this mode's).
        hidden = self.model(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            lora_params=lora_params,
            **kwargs,
        )
        # `gather_ids` holds one row per context request (its last token) and
        # `runtime_draft_len + 1` per generation request. `embedding` is the
        # catalog's row-lookup entry (torch.nn.functional.embedding), used here
        # for what it is — `hidden[gather_ids]`.
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
        _weights.load(self, weights)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.derive_after_load()
