# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Staircase target: gpt-oss-120b / sm_103 / tp1 — self-contained modeling code.

Flat single-entry forward assembled from catalog entries only; every call
that creates or transforms a tensor is a catalog entry, everything else is
tensor-metadata reads and Python control flow. Attention consumes runtime
state fully explicitly through thop_attention: per-step arguments are
projected from the engine-prepared TrtllmAttentionMetadata once per forward
in _build_step_args and shared by all layers. Two attention arguments are
per-layer here rather than per-step: the fp32 sink logits (one extra softmax
denominator column per query head) and attention_window_size — this
checkpoint alternates sliding_attention (window 128) and full_attention
layers, and the window is a pure mask, so one shared pool and one
block-offset table serve both kinds.

Every layer's MLP is an MXFP4 sparse mixture of experts (128 experts, top-4,
renormalized, clamped GLU) run W4A8: mxfp8_quantize turns the bf16 hidden
states into e4m3 data plus per-32 UE8M0 block scales — widening hidden 2880
to the FC1 K alignment 3072 inside that call — and one
mxe4m3_mxe2m1_block_scale_moe_runner call per layer covers routing, both
grouped GEMMs, the clamped activation, the MXFP8 requantization between them
and the combine. The router bias is folded into the router GEMM's fused bias
because the MoE op silently ignores routing_bias on this routing method. The
expert weights are declared in the kernel-ready padded/shuffled/swizzled
layout — identical for the W4A16 and W4A8 members of this kernel family — and
the manifest loop in weights.py transforms the checkpoint's block/scale
tensors into it at load time.

Weights are target-owned: a flat ParameterDict declared here (HF [out, in]
storage so checkpoint rows copy in unchanged), loaded by the manifest loop
in the sibling weights.py, with column-major GEMM views derived once after
load. The registration shell inherits DecoderModelForCausalLM for lm_head,
packed-batch logits gathering, and the meta-init/load/post-load hooks.

The import-time and first-forward contract checks below fail fast on drift.
"""

import math

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import (
    DecoderModel,
    DecoderModelForCausalLM,
    register_auto_model,
)
from tensorrt_llm._torch.staircase.catalog.attention.fused_qk_norm_rope import fused_qk_norm_rope
from tensorrt_llm._torch.staircase.catalog.attention.thop_attention import thop_attention
from tensorrt_llm._torch.staircase.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch.staircase.catalog.moe.mxe4m3_mxe2m1_block_scale_moe_runner import (  # noqa: E501
    mxe4m3_mxe2m1_block_scale_moe_runner,
)
from tensorrt_llm._torch.staircase.catalog.norm.flashinfer_fused_add_rmsnorm import (  # noqa: E501
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch.staircase.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm
from tensorrt_llm._torch.staircase.catalog.quantization.mxfp8_quantize import mxfp8_quantize
from tensorrt_llm._torch.staircase.catalog.torch.embedding import embedding
from tensorrt_llm._torch.staircase.catalog.torch.empty import empty
from tensorrt_llm._torch.staircase.catalog.torch.reshape import reshape

from . import weights as _weights

# The GPU architecture this target IS. Routing will not send another one here,
# but a direct instantiation could, and the certification is per arch: this
# assert is what the version pin used to be. In-tree the version moves with
# the code, so pinning it is meaningless; the architecture does not.
_SM = (10, 3)


def _check_static_contract() -> None:
    """Import-time fail-fast: op symbol existence."""
    for op in (
        "fused_qk_norm_rope",
        "flashinfer_rmsnorm",
        "flashinfer_fused_add_rmsnorm",
        "cublas_mm",
        "mxfp8_quantize",
        "mxe4m3_mxe2m1_block_scale_moe_runner",
    ):
        assert hasattr(torch.ops.trtllm, op), f"missing op trtllm::{op}"
    from tensorrt_llm.bindings.internal import thop

    assert hasattr(thop, "attention"), "missing pybind thop.attention"


_check_static_contract()

# Metadata fields consumed each step (sourcing mirrors the in-tree
# FallbackFmha for this trtllm version; existence checked at first forward).
_STEP_FIELDS = (
    "kv_lens_cuda_runtime",
    "kv_lens_runtime",
    "host_total_kv_lens",
    "prompt_lens_cuda_runtime",
    "prompt_lens_cpu_runtime",
    "host_request_types_runtime",
    "kv_cache_block_offsets",
    "host_kv_cache_pool_pointers",
    "host_kv_cache_pool_mapping",
    "effective_workspace",
    "tokens_per_block",
    "max_num_requests",
    "max_context_length",
    "max_seq_len",
    "num_contexts",
    "num_ctx_tokens",
    "trtllm_gen_jit_warmup",
    "use_paged_context_fmha",
    "effective_beam_width",
    "cache_indirection",
    "block_ids_per_seq",
    "max_context_q_len_override",
    "is_cross",
    "is_spec_decoding_enabled",
    "use_spec_decoding",
    "is_spec_dec_tree",
    "spec_decoding_generation_lengths",
    "spec_decoding_position_offsets_for_cpp",
    "spec_decoding_packed_mask",
    "spec_decoding_bl_tree_mask_offset",
    "spec_decoding_bl_tree_mask",
    "max_total_draft_tokens",
    "spec_bl_tree_first_sparse_mask_offset_kv",
    "num_sparse_topk",
    "flash_mla_tile_scheduler_metadata",
    "flash_mla_num_splits",
    # Added between 1.3.0rc21 and 1.3.0rc26. Both are engine-prepared
    # per-instance constants (max_num_sequences defaults to
    # max_num_requests; the tree-mask flag is set from
    # is_spec_dec_dynamic_tree), so they project like the rest.
    "max_num_sequences",
    "force_prepare_spec_dec_tree_mask",
)


def _build_step_args(md: TrtllmAttentionMetadata) -> dict:
    """Project the prepared metadata onto thop_attention's explicit batch
    state, once per forward; every runtime-owned value passes through as
    the engine prepared it. CUDA-graph classes: tensors are engine-owned
    persistent buffers refreshed in place (reference class); Python ints
    are per-capture constants (host-derived class — a decode-only graph
    always sees num_contexts == 0). attention_window_size is absent here
    on purpose: it varies per layer and is passed at the call site."""
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
        spec_decoding_position_offsets_for_cpp=md.spec_decoding_position_offsets_for_cpp,
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


# Per-call constants of this target's call shape — the values the in-tree
# path sources from the attention module and forward args: packed-QKV
# causal GQA, RoPE applied outside (fused_qk_norm_rope), bf16 activations
# over a bf16 KV pool, no MLA / mRoPE / cross / relative-bias / sparse
# features. attention_sinks and attention_window_size are per-layer, not
# constants, and are passed at the call site.
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
    # In-kernel RoPE is disabled (position_embedding_type=0): rotation
    # happens outside in fused_qk_norm_rope. The rope_* values below are
    # the contract's inert placeholders, not this model's rope config.
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
    # Added between 1.3.0rc21 and 1.3.0rc26; held at the values the op had
    # before they existed. All three are MLA-only surface that this target
    # does not use -- skip_correction is forced to 0.0 for a non-MLA layer by
    # the engine's own resolver, and kv_norm_* folds an MLA kv_a_layernorm
    # that does not exist here.
    kv_norm_weight=None,
    kv_norm_eps=1e-6,
    skip_correction_threshold=0.0,
)

# The MoE op's clamped gated activation: (up + beta) * gate *
# sigmoid(alpha * gate) after gate.clamp(max=limit), up.clamp(+-limit) —
# exactly this checkpoint's expert activation. alpha/beta are fixed in the
# HF reference; limit is config (swiglu_limit).
_GLU_ALPHA = 1.702
_GLU_BETA = 1.0
# Renormalize routing (top-k first, then fp32 softmax over the selected
# logits) and the only gated-activation kernel in this dtype family.
_ROUTING_METHOD_RENORMALIZE = 1
_ACT_TYPE_SWIGLU = 0
# FC1's K alignment for the trtllm-gen MXFP4 weight family: it sizes the
# declared expert operands and is the alignment mxfp8_quantize pads the
# hidden states up to, so the two always agree.
_FC1_K_ALIGN = 512
# The MoE op reads the activation scales as a linear (row-major) buffer.
# The 128x4 swizzled order has the same byte count whenever num_tokens is a
# multiple of 128 — which every decode CUDA-graph batch of 128 or 256 is —
# and is then accepted silently as a wrong answer, so this is spelled out
# rather than left to the quantizer's default.
_LINEAR_SCALE_LAYOUT = False


def _pad_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


class StaircaseCore(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = model_config.pretrained_config
        assert cfg is not None

        # This target IS this geometry — assert, never adapt.
        assert torch.cuda.get_device_capability() == _SM, (
            f"target certified on sm_{_SM[0]}{_SM[1]}, running on "
            f"sm_{''.join(map(str, torch.cuda.get_device_capability()))}"
        )
        assert model_config.mapping.tp_size == 1, "tp1 target"
        assert model_config.mapping.pp_size == 1, "tp1 target"
        # This checkpoint's config.json declares no dtype at all, so
        # `pretrained_config.torch_dtype` is None. The engine resolves bf16
        # regardless (ModelConfig.torch_dtype defaults it), but the shell
        # sizes lm_head from the *pretrained* value and would materialize it
        # in the torch default fp32 — two layers away from where it is read.
        # The shell normalizes that before super().__init__ (see below), so
        # by the time the core is built both surfaces agree, and both are
        # asserted: the declaration the shell and the KV pool are sized from,
        # and the value quant_mode=0 must agree with.
        dt = model_config.torch_dtype
        assert cfg.torch_dtype == torch.bfloat16, cfg.torch_dtype
        assert dt == torch.bfloat16, f"bf16 target, engine resolved {dt}"
        assert not cfg.tie_word_embeddings, "untied lm_head"
        assert cfg.attention_bias, "q/k/v/o carry bias"
        assert cfg.hidden_act == "silu", "clamped GLU over a silu-shaped gate"

        # bf16 KV pool only: an fp8 pool needs quant_mode=128 plus the two
        # fp32 scale tensors, which this target does not build. The expert
        # quantization (mxfp4 blocks + e8m0 scales) is consumed directly by
        # the MoE op, so quant_config.quant_algo — which the engine reads
        # off quantization_config as W4A8_MXFP4_MXFP8, an mxfp8-activation
        # recipe this target does not implement — is inert here.
        kv_algo = model_config.quant_config.kv_cache_quant_algo
        assert kv_algo is None, f"unsupported KV algo {kv_algo}; bf16 pool only"
        quant = getattr(cfg, "quantization_config", None)
        assert isinstance(quant, dict) and quant.get("quant_method") == "mxfp4", (
            "expert weights must be mxfp4 blocks + e8m0 scales"
        )

        self.num_layers = cfg.num_hidden_layers
        self.hidden = cfg.hidden_size
        self.heads_q = cfg.num_attention_heads
        self.heads_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.eps = cfg.rms_norm_eps

        # RoPE: YaRN over the full head_dim, half-split (neox) pairs. The
        # engine hands this checkpoint the transformers-5.x migrated rope
        # dict (rope_theta lives inside it and cfg.rope_theta is absent);
        # a checkpoint written before that migration keeps the flat field,
        # so both shapes are resolved here and every scalar the ramp
        # depends on is asserted rather than defaulted.
        rope = getattr(cfg, "rope_scaling", None) or getattr(cfg, "rope_parameters", None)
        assert isinstance(rope, dict), "rope parameters must be a dict"
        assert rope.get("rope_type") == "yarn", "YaRN rope scaling"
        assert rope.get("partial_rotary_factor", 1.0) == 1.0, "full-width rotation"
        assert rope.get("attention_factor") is None, "attention_factor is derived"
        assert rope.get("mscale") is None and rope.get("mscale_all_dim") is None, (
            "the derived attention_factor assumes no mscale pair"
        )
        self.theta = float(rope.get("rope_theta", getattr(cfg, "rope_theta", 0.0)))
        assert self.theta > 0.0, "rope theta"
        self.rotary_dim = self.head_dim
        self.yarn_factor = float(rope["factor"])
        beta_fast = float(rope["beta_fast"])
        beta_slow = float(rope["beta_slow"])
        orig_max = float(rope["original_max_position_embeddings"])
        truncate = bool(rope.get("truncate", True))

        def correction_dim(rotations: float) -> float:
            return (
                self.rotary_dim
                * math.log(orig_max / (rotations * 2.0 * math.pi))
                / (2.0 * math.log(self.theta))
            )

        low = correction_dim(beta_fast)
        high = correction_dim(beta_slow)
        if truncate:
            low, high = math.floor(low), math.ceil(high)
        self.yarn_low = max(low, 0.0)
        self.yarn_high = min(high, self.rotary_dim - 1.0)
        self.yarn_attn_factor = (
            0.1 * math.log(self.yarn_factor) + 1.0 if self.yarn_factor > 1.0 else 1.0
        )

        # Sliding window: half the layers mask to the newest `window` keys,
        # the rest are plain causal. The window is a mask only — the op
        # appends at absolute positions — so both kinds share one pool, one
        # layer->pool mapping and one block-offset table.
        layer_types = list(cfg.layer_types)
        assert len(layer_types) == self.num_layers
        assert set(layer_types) <= {"sliding_attention", "full_attention"}, layer_types
        self.sliding = [t == "sliding_attention" for t in layer_types]
        self.window = cfg.sliding_window
        assert any(self.sliding) and self.window > 0, "alternating sliding window"

        # Sparse MoE on every layer; no dense MLP branch exists.
        self.num_experts = cfg.num_local_experts
        self.topk = cfg.num_experts_per_tok
        self.inter = cfg.intermediate_size
        self.glu_limit = float(cfg.swiglu_limit)
        assert 0 < self.topk < self.num_experts, "MoE top-k bound"

        # Kernel bounds the geometry must fit: fused_qk_norm_rope's head_dim
        # set, thop_attention's GQA rule, and the MoE op's padded operand
        # widths (the padded intermediate is what `intermediate_size` means
        # to that call, and hidden_states reach it widened to fc1_k_pad by
        # the quantizer).
        assert self.head_dim in (64, 128, 256), "fused_qk_norm_rope head_dim set"
        assert self.heads_q % self.heads_kv == 0, "thop_attention GQA rule"
        self.inter_pad = _pad_up(self.inter, 128)
        self.fc1_k_pad = _pad_up(self.hidden, _FC1_K_ALIGN)
        self.fc2_rows_pad = _pad_up(self.hidden, 128)
        assert self.hidden % 32 == 0, "valid_hidden_size must be a multiple of 32"
        assert self.inter % 32 == 0, "valid_intermediate_size must be a multiple of 32"

        q_width = self.heads_q * self.head_dim
        kv_width = self.heads_kv * self.head_dim

        # Weight declaration: HF [out, in] row-major storage so checkpoint
        # rows copy in unchanged; GEMM consumes .t() column-major views
        # built after load. The expert tensors are the exception — they are
        # declared in the MoE op's kernel-ready layout (padded, row-permuted
        # weights/scales/biases), which weights.py builds from the
        # checkpoint's block/scale tensors during the load. Meta-init
        # intercepts torch.empty here — real CUDA storage arrives when the
        # engine materializes the registry.
        def P(*shape, dtype=dt):
            return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)

        fc1_rows = 2 * self.inter_pad
        w = nn.ParameterDict()
        for i in range(self.num_layers):
            w[f"l{i}_norm1"] = P(self.hidden)
            w[f"l{i}_qkv"] = P(q_width + 2 * kv_width, self.hidden)
            w[f"l{i}_qkv_bias"] = P(q_width + 2 * kv_width)
            w[f"l{i}_sinks"] = P(self.heads_q, dtype=torch.float32)
            w[f"l{i}_o"] = P(self.hidden, q_width)
            w[f"l{i}_o_bias"] = P(self.hidden)
            w[f"l{i}_norm2"] = P(self.hidden)
            w[f"l{i}_router"] = P(self.num_experts, self.hidden)
            w[f"l{i}_router_bias"] = P(self.num_experts)
            w[f"l{i}_fc1_w"] = P(self.num_experts, fc1_rows, self.fc1_k_pad // 2, dtype=torch.uint8)
            w[f"l{i}_fc1_s"] = P(
                self.num_experts, fc1_rows, self.fc1_k_pad // 32, dtype=torch.uint8
            )
            w[f"l{i}_fc1_b"] = P(self.num_experts, fc1_rows, dtype=torch.float32)
            w[f"l{i}_fc2_w"] = P(
                self.num_experts,
                self.fc2_rows_pad,
                self.inter_pad // 2,
                dtype=torch.uint8,
            )
            w[f"l{i}_fc2_s"] = P(
                self.num_experts,
                self.fc2_rows_pad,
                self.inter_pad // 32,
                dtype=torch.uint8,
            )
            w[f"l{i}_fc2_b"] = P(self.num_experts, self.fc2_rows_pad, dtype=torch.float32)
        w["final_norm"] = P(self.hidden)
        w["embed"] = P(cfg.vocab_size, self.hidden)
        self.w = w

        self._layers: list | None = None
        self._call_tensors: dict | None = None
        self._step_contract_checked = False

    def build_layer_views(self) -> None:
        """Post-load derivation: per-layer tuples of column-major GEMM views,
        norm weights and expert operands (kills hot-path dict lookups; .t()
        is zero-copy), plus the constant fp32 call tensors the MoE and RoPE
        calls require. Meta is over here, so real tensors may be created."""
        w = self.w
        device = w["final_norm"].device
        # Per-expert activation scalars, sized by local_num_experts.
        self._call_tensors = {
            "alpha": torch.full(
                (self.num_experts,), _GLU_ALPHA, dtype=torch.float32, device=device
            ),
            "beta": torch.full((self.num_experts,), _GLU_BETA, dtype=torch.float32, device=device),
            "limit": torch.full(
                (self.num_experts,), self.glu_limit, dtype=torch.float32, device=device
            ),
            # fused_qk_norm_rope requires valid q/k norm weight tensors even
            # with is_qk_norm=False, where their values are unused.
            "no_qk_norm": torch.zeros(self.head_dim, dtype=w["final_norm"].dtype, device=device),
        }
        layers = []
        for i in range(self.num_layers):
            next_norm = w[f"l{i + 1}_norm1"] if i + 1 < self.num_layers else w["final_norm"]
            layers.append(
                (
                    w[f"l{i}_qkv"].t(),
                    w[f"l{i}_qkv_bias"],
                    w[f"l{i}_sinks"],
                    w[f"l{i}_o"].t(),
                    w[f"l{i}_o_bias"],
                    w[f"l{i}_norm2"],
                    w[f"l{i}_router"].t(),
                    w[f"l{i}_router_bias"],
                    w[f"l{i}_fc1_w"],
                    w[f"l{i}_fc1_s"],
                    w[f"l{i}_fc1_b"],
                    w[f"l{i}_fc2_w"],
                    w[f"l{i}_fc2_s"],
                    w[f"l{i}_fc2_b"],
                    next_norm,
                )
            )
        self._layers = layers

    def _check_step_contract(self, md, position_ids) -> None:
        """First-forward fail-fast: the metadata fields this target consumes
        must exist (they are private trtllm surface, pinned by version), and
        the KV pool must be the single shared pool the sliding-window
        surface is certified over. Everything checked is fixed at engine
        construction — once per model instance is sound."""
        missing = [name for name in _STEP_FIELDS if not hasattr(md, name)]
        assert not missing, f"metadata fields missing: {missing}"
        assert position_ids.dtype == torch.int32
        pools = {row[0] for row in md.host_kv_cache_pool_mapping.tolist()}
        assert pools == {0}, (
            f"multi-pool KV addressing is not certified; layer->pool ids {sorted(pools)}"
        )
        self._step_contract_checked = True

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        assert self._layers is not None and self._call_tensors is not None, (
            "load_weights must run before forward"
        )
        assert position_ids is not None
        assert isinstance(attn_metadata, TrtllmAttentionMetadata)
        # Inputs this target does not implement must fail loudly, not be
        # silently dropped (unlike runtime-owned features, which pass through).
        assert lora_params is None, "LoRA is not implemented by this target"
        assert kwargs.get("spec_metadata") is None, (
            "speculative decoding is not implemented by this target"
        )
        if not self._step_contract_checked:
            self._check_step_contract(attn_metadata, position_ids)

        step = _build_step_args(attn_metadata)
        # Full-attention layers take the no-window value; sliding layers take
        # the checkpoint's window. Both are host-derived per-capture constants
        # (max_seq_len is an engine-construction constant).
        full_window = attn_metadata.max_seq_len
        const = self._call_tensors
        no_qk_norm = const["no_qk_norm"]
        pos = reshape(position_ids, [-1])

        if inputs_embeds is None:
            assert input_ids is not None
            h = embedding(input_ids, self.w["embed"])
        else:
            h = inputs_embeds
        num_tokens = h.shape[0]
        dt = h.dtype
        attn_out = empty([num_tokens, self.heads_q * self.head_dim], dt, h.device)

        x = flashinfer_rmsnorm(h, self.w["l0_norm1"], self.eps)
        residual = h
        for i in range(self.num_layers):
            (
                w_qkv,
                b_qkv,
                sinks,
                w_o,
                b_o,
                w_n2,
                w_rt,
                b_rt,
                fc1_w,
                fc1_s,
                fc1_b,
                fc2_w,
                fc2_s,
                fc2_b,
                w_next,
            ) = self._layers[i]
            qkv = cublas_mm(x, w_qkv, b_qkv)
            fused_qk_norm_rope(
                qkv,
                num_heads_q=self.heads_q,
                num_heads_k=self.heads_kv,
                num_heads_v=self.heads_kv,
                head_dim=self.head_dim,
                rotary_dim=self.rotary_dim,
                eps=self.eps,
                q_weight=no_qk_norm,
                k_weight=no_qk_norm,
                base=self.theta,
                is_neox=True,
                position_ids=pos,
                factor=self.yarn_factor,
                low=self.yarn_low,
                high=self.yarn_high,
                attention_factor=self.yarn_attn_factor,
                is_qk_norm=False,
            )
            thop_attention(
                q=qkv,
                output=attn_out,
                local_layer_idx=i,
                num_heads=self.heads_q,
                num_kv_heads=self.heads_kv,
                head_size=self.head_dim,
                attention_sinks=sinks,
                attention_window_size=self.window if self.sliding[i] else full_window,
                **step,
                **_CALL_CONSTANTS,
            )
            o = cublas_mm(attn_out, w_o, b_o)
            flashinfer_fused_add_rmsnorm(o, residual, w_n2, self.eps)
            # The router bias rides the GEMM epilogue: the MoE op silently
            # ignores routing_bias on this routing method.
            router_logits = cublas_mm(o, w_rt, b_rt)
            # The quantizer owns the hidden widening 2880 -> fc1_k_pad: it
            # zero-fills the padded columns and their scale bytes, which
            # multiply zero-valued padded weights.
            hidden_fp8, hidden_sf = mxfp8_quantize(o, _LINEAR_SCALE_LAYOUT, _FC1_K_ALIGN)
            moe = mxe4m3_mxe2m1_block_scale_moe_runner(
                router_logits,
                None,
                hidden_fp8,
                hidden_sf,
                fc1_w,
                fc1_s,
                fc1_b,
                const["alpha"],
                const["beta"],
                const["limit"],
                fc2_w,
                fc2_s,
                fc2_b,
                self.num_experts,
                self.topk,
                None,
                None,
                self.inter_pad,
                self.hidden,
                self.inter,
                0,
                self.num_experts,
                None,
                _ROUTING_METHOD_RENORMALIZE,
                _ACT_TYPE_SWIGLU,
            )
            flashinfer_fused_add_rmsnorm(moe, residual, w_next, self.eps)
            x = moe
        return x


@register_auto_model("StaircaseGptOss120bSm103Tp1")
class StaircaseGptOss120bSm103Tp1(DecoderModelForCausalLM[StaircaseCore, PretrainedConfig]):
    def __init__(self, model_config: ModelConfig):
        cfg = model_config.pretrained_config
        assert cfg is not None
        # DecoderModelForCausalLM sizes lm_head from the *pretrained* dtype,
        # which this checkpoint's config.json does not declare — leaving
        # lm_head.weight fp32 while every other tensor is bf16, and failing
        # two layers away from here. The engine has already resolved the
        # dtype it will run at; adopt it, rather than let the default of a
        # missing field decide. Only fills the gap: a declared dtype wins.
        if cfg.torch_dtype is None:
            cfg.torch_dtype = model_config.torch_dtype
        super().__init__(
            StaircaseCore(model_config),
            config=model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )

    def load_weights(self, weights, *args, **kwargs):
        _weights.load(self, weights)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.build_layer_views()
