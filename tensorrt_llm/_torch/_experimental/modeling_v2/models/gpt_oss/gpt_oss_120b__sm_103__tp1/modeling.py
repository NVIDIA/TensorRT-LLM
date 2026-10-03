# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: gpt-oss-120b / sm_103 / tp1 — self-contained modeling code.

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

from tensorrt_llm._torch._experimental.modeling_v2._router_index import step_contract_enabled
from tensorrt_llm._torch._experimental.modeling_v2._target import Phase, Target, phase_of
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import advance_step_generation
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.fused_qk_norm_rope import (
    FusedQkNormRope,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    ThopAttention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import CublasMm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.mxe4m3_mxe2m1_block_scale_moe_runner import (  # noqa: E501
    Mxe4m3Mxe2m1BlockScaleMoeRunner,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (  # noqa: E501
    FlashinferFusedAddRmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    FlashinferRmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.mxfp8_quantize import (
    Mxfp8Quantize,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import (
    DecoderModel,
    DecoderModelForCausalLM,
    register_auto_model,
)

from . import weights as _weights


def _build_step_args(
    md: TrtllmAttentionMetadata, *, num_contexts: int, num_ctx_tokens: int
) -> dict:
    """Project the prepared metadata onto thop_attention's explicit batch
    state, once per forward; every runtime-owned value passes through as
    the engine prepared it. CUDA-graph classes: tensors are engine-owned
    persistent buffers refreshed in place (reference class); Python ints
    are per-capture constants (host-derived class). attention_window_size is
    absent here on purpose: it is per-layer, not per-step, and is rebound
    every forward with `bind_layered` instead -- see the comment in
    `PrefillTarget.forward` and `DecodeTarget.forward`.

    `num_contexts` and `num_ctx_tokens` are passed rather than read off `md`
    because they are the two values a decode target knows by its routing --
    it states them as 0 instead of reading back what the predicate already
    guaranteed."""
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
        num_contexts=num_contexts,
        num_ctx_tokens=num_ctx_tokens,
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
    # Held at the op's defaults. All three are MLA-only surface that this
    # target does not use -- skip_correction is forced to 0.0 for a non-MLA layer by
    # the engine's own resolver, and kv_norm_* folds an MLA kv_a_layernorm
    # that does not exist here.
    kv_norm_weight=None,
    kv_norm_eps=1e-6,
    skip_correction_threshold=0.0,
)

# FC1's K alignment for the trtllm-gen MXFP4 weight family: it sizes the
# declared expert operands and is the alignment mxfp8_quantize pads the
# hidden states up to, so the two always agree. Kept as a module constant
# (not folded into a call site) because it has two readers: the quantizer's
# bind_const below and `self.fc1_k_pad = _pad_up(self.hidden, _FC1_K_ALIGN)`
# in `ModelingV2Core.__init__`.
_FC1_K_ALIGN = 512


def _pad_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


class ModelingV2Core(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = model_config.pretrained_config
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

        # Locals, not attributes: a core carries no forwarded configuration.
        # Every reader of these five -- this method's own weight-shape
        # declarations below, weights.py's load(), and the target's
        # __init__/forward -- already has, or can cheaply reach, `cfg`
        # (`core.model_config.pretrained_config`) and reads it from there
        # directly. A copy onto `self` would only be a second name for the
        # same value. `inter_pad`, `fc1_k_pad` and `fc2_rows_pad` further
        # down are different: they are *derived* (padded up), not copied,
        # so they stay attributes -- see the comment there. `sliding` and
        # `window`, below, are attributes for a different reason: the
        # forward reads them per layer, every step.
        num_layers = cfg.num_hidden_layers
        hidden = cfg.hidden_size
        heads_q = cfg.num_attention_heads
        heads_kv = cfg.num_key_value_heads
        head_dim = cfg.head_dim

        # RoPE: YaRN over the full head_dim, half-split (neox) pairs. The
        # engine hands this checkpoint the transformers-5.x migrated rope
        # dict (rope_theta lives inside it and cfg.rope_theta is absent);
        # a checkpoint written before that migration keeps the flat field,
        # so both shapes are resolved here and every scalar the ramp
        # depends on is asserted rather than defaulted.
        #
        # Unlike the block above, nothing but `PrefillTarget.__init__` and
        # `DecodeTarget.__init__` (via `build_layer_views`) reads `theta` or
        # the four `yarn_*` once this method returns -- they exist solely to
        # bind `fused_qk_norm_rope`. They stay attributes anyway, rather than
        # becoming a second copy of this derivation inside the target: the
        # ramp has real failure modes
        # (a missing rope key raises KeyError; `math.log` of a non-positive
        # `theta` or `orig_max` raises ValueError), and catching those here,
        # before the weight load, is cheaper than catching them afterward or
        # duplicating ~20 lines of math to catch them in two places. Nothing
        # downstream of `head_dim` has that problem -- `rotary_dim` equals it
        # exactly for this checkpoint -- so the target binds
        # `rotary_dim=cfg.head_dim` directly instead of reading a third name
        # for the same value.
        rope = getattr(cfg, "rope_scaling", None) or getattr(cfg, "rope_parameters", None)
        self.theta = float(rope.get("rope_theta", getattr(cfg, "rope_theta", 0.0)))
        rotary_dim = head_dim
        self.yarn_factor = float(rope["factor"])
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
        self.yarn_low = max(low, 0.0)
        self.yarn_high = min(high, rotary_dim - 1.0)
        self.yarn_attn_factor = (
            0.1 * math.log(self.yarn_factor) + 1.0 if self.yarn_factor > 1.0 else 1.0
        )

        # Sliding window: half the layers mask to the newest `window` keys,
        # the rest are plain causal. The window is a mask only — the op
        # appends at absolute positions — so both kinds share one pool, one
        # layer->pool mapping and one block-offset table.
        layer_types = list(cfg.layer_types)
        self.sliding = [t == "sliding_attention" for t in layer_types]
        self.window = cfg.sliding_window

        # Sparse MoE on every layer; no dense MLP branch exists. `num_experts`
        # and `inter` are locals for the same reason as the five above.
        # `topk` and `swiglu_limit` never even get that far: weights.py
        # doesn't read them either, so (like `eps`, this checkpoint's
        # `rms_norm_eps`) each target reads it straight off `cfg` at its own
        # one call site instead -- see `PrefillTarget.__init__` and
        # `DecodeTarget.__init__`.
        num_experts = cfg.num_local_experts
        inter = cfg.intermediate_size

        # Kernel bounds the geometry must fit: fused_qk_norm_rope's head_dim
        # set, thop_attention's GQA rule, and the MoE op's padded operand
        # widths (the padded intermediate is what `intermediate_size` means
        # to that call, and hidden_states reach it widened to fc1_k_pad by
        # the quantizer). Derived, not copied, so these three stay
        # attributes -- weights.py reaches them through `core.X` and would
        # otherwise have to duplicate this padding math to get there.
        self.inter_pad = _pad_up(inter, 128)
        self.fc1_k_pad = _pad_up(hidden, _FC1_K_ALIGN)
        self.fc2_rows_pad = _pad_up(hidden, 128)

        q_width = heads_q * head_dim
        kv_width = heads_kv * head_dim

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
        for i in range(num_layers):
            w[f"l{i}_norm1"] = P(hidden)
            w[f"l{i}_qkv"] = P(q_width + 2 * kv_width, hidden)
            w[f"l{i}_qkv_bias"] = P(q_width + 2 * kv_width)
            w[f"l{i}_sinks"] = P(heads_q, dtype=torch.float32)
            w[f"l{i}_o"] = P(hidden, q_width)
            w[f"l{i}_o_bias"] = P(hidden)
            w[f"l{i}_norm2"] = P(hidden)
            w[f"l{i}_router"] = P(num_experts, hidden)
            w[f"l{i}_router_bias"] = P(num_experts)
            w[f"l{i}_fc1_w"] = P(num_experts, fc1_rows, self.fc1_k_pad // 2, dtype=torch.uint8)
            w[f"l{i}_fc1_s"] = P(num_experts, fc1_rows, self.fc1_k_pad // 32, dtype=torch.uint8)
            w[f"l{i}_fc1_b"] = P(num_experts, fc1_rows, dtype=torch.float32)
            w[f"l{i}_fc2_w"] = P(
                num_experts,
                self.fc2_rows_pad,
                self.inter_pad // 2,
                dtype=torch.uint8,
            )
            w[f"l{i}_fc2_s"] = P(
                num_experts,
                self.fc2_rows_pad,
                self.inter_pad // 32,
                dtype=torch.uint8,
            )
            w[f"l{i}_fc2_b"] = P(num_experts, self.fc2_rows_pad, dtype=torch.float32)
        w["final_norm"] = P(hidden)
        w["embed"] = P(cfg.vocab_size, hidden)
        self.w = w

        self._targets: dict[Phase, Target] | None = None
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it: read once here
        # rather than per forward, and False for the whole life of a served
        # engine. "pending" rather than "enabled" because the check runs once
        # -- everything it looks at is fixed at engine construction.
        self._contract_pending = step_contract_enabled()

    def build_layer_views(self) -> None:
        """Construct the per-phase targets, now that the weights are real.

        Used to also derive per-layer GEMM views and the MoE/RoPE call
        tensors itself, holding them in two attributes (a per-layer tuple
        and a call-tensor dict) that this class no longer declares. Both are
        deleted outright, not just unused: each target's own `__init__` now
        binds its own per-layer weight tables and op-owned fixtures straight
        into the catalog-entry instances it holds, so there is no longer a
        parallel copy here for a call site to read out of -- and nothing else
        reads `self.w` through this method. `PrefillTarget` and `DecodeTarget`
        each do this independently; the two bindings are identical in
        content, not shared in code.

        Meta is over by the time this runs, so real tensors may be built and
        `.t()`'d; never called from `__init__`, where the shell's containers
        are still meta."""
        self._targets = {
            Phase.PREFILL: PrefillTarget(self),
            Phase.DECODE: DecodeTarget(self),
        }

    def _check_step_contract(self, md, position_ids) -> None:
        """Opt-in first-forward fail-fast, run when TRTLLM_MODELING_V2_VALIDATE
        asks for it: the metadata fields this target consumes must exist (they
        are private trtllm surface). Everything checked is fixed at
        engine construction — once per model instance is sound, and off in a
        served engine, where the only thing this could still do is fail.
        """
        # Calling the projection is the check: it reads every metadata field
        # this target consumes, so a rename or removal upstream surfaces
        # here rather than mid-forward. Deriving it this way is the point --
        # a hand-kept list of the same names drifts silently the first time
        # _build_step_args gains a field and nobody updates the copy.
        try:
            _build_step_args(md, num_contexts=md.num_contexts, num_ctx_tokens=md.num_ctx_tokens)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Route the step, and nothing else.

        The contract check runs here rather than inside a target because it is
        phase-independent: it does not vary by phase, so running it once in
        the dispatcher is equivalent to duplicating it into both
        `PrefillTarget` and `DecodeTarget` and checks nothing they would not.
        """
        if self._contract_pending:
            self._check_step_contract(attn_metadata, kwargs.get("position_ids"))
        return self._targets[phase_of(attn_metadata)].forward(attn_metadata, *args, **kwargs)


class PrefillTarget(Target):
    """The general case: context rows, and possibly generation rows beside them.

    In-flight batching puts both in one step, and that mixed batch routes here
    rather than to decode -- so this target reads both counts off the metadata
    and may not assume either is zero.

    gpt_oss is not MLA, so `TrtllmAttention` accepts `mixed` and the
    context/generation split stays inside the C++ dispatcher: this forward has
    no phase branch to divide. `DecodeTarget`, below, runs the identical body
    -- the two differ only in `step_args` -- as a separate, independent class:
    targets in this tree do not share code with each other, even when sharing
    would be free. A model whose phases run different computations --
    deepseek's MLA, where generation works in latent space and context
    materializes K and V -- overrides `forward` itself instead.
    """

    def __init__(self, core: ModelingV2Core) -> None:
        """Bind every op this target calls, once, against the real weights.

        `Target.__init__` only stores `self.core`; everything below is this
        target's own addition, and is why gpt_oss needed one. Per-layer
        weight tables go into each catalog entry's `bind_layered`, one layer
        at a time, built straight from `core.w` and keeping the `.t()` views
        the hot-path GEMMs already used (zero-copy). Configuration comes
        from `cfg` throughout: `core` carries none of it past what is
        genuinely derived (`inter_pad`, `fc1_k_pad`, `fc2_rows_pad`, `theta`,
        the `yarn_*` quantities) or read per layer by the forward
        (`sliding`, `window`) -- see `ModelingV2Core.__init__` for which is
        which. The one op-owned fixture built here -- `fused_qk_norm_rope`'s
        inert q/k norm weight -- is an input that op needs to satisfy its
        own signature, not state the model computes; the MoE runner's
        per-expert activation vectors are the same kind of fixture, but the
        op builds those itself now, via `bind_glu`.
        """
        super().__init__(core)
        cfg = core.model_config.pretrained_config
        w = core.w
        device = w["final_norm"].device
        dtype = w["final_norm"].dtype
        n = cfg.num_hidden_layers

        self._norm0 = FlashinferRmsnorm()
        self._norm0.bind_const(weight=w["l0_norm1"], eps=cfg.rms_norm_eps)

        self._qkv = CublasMm()
        for i in range(n):
            self._qkv.bind_layered(i, mat_b=w[f"l{i}_qkv"].t(), bias=w[f"l{i}_qkv_bias"])

        # fused_qk_norm_rope requires valid q/k norm weight tensors even with
        # is_qk_norm=False, where their values are unused; built from the
        # scalars the op needs (head_dim, dtype, device) rather than copied
        # off a reference tensor, which would tie this fixture to whichever
        # tensor happened to be handy when it was written. rotary_dim is
        # cfg.head_dim directly -- equal for this checkpoint, and not worth
        # a third name for the same value (see `ModelingV2Core.__init__`).
        no_qk_norm = torch.zeros(cfg.head_dim, dtype=dtype, device=device)
        self._qk_rope = FusedQkNormRope()
        self._qk_rope.bind_const(
            num_heads_q=cfg.num_attention_heads,
            num_heads_k=cfg.num_key_value_heads,
            num_heads_v=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            rotary_dim=cfg.head_dim,
            eps=cfg.rms_norm_eps,
            q_weight=no_qk_norm,
            k_weight=no_qk_norm,
            base=core.theta,
            is_neox=True,
            factor=core.yarn_factor,
            low=core.yarn_low,
            high=core.yarn_high,
            attention_factor=core.yarn_attn_factor,
            is_qk_norm=False,
        )

        # attention_window_size is absent here -- it is not knowable until a
        # forward is underway (its full-attention value is
        # attn_metadata.max_seq_len, the KV cache manager's *resolved* size,
        # not known until after this target's __init__ runs at post-load),
        # so every forward binds it fresh instead -- see the comment in
        # `forward`. local_layer_idx is layered rather than passed at the
        # call site even though it is exactly `layer`'s own value: it is a
        # genuine op argument (the row thop_attention reads out of the pool
        # mapping), not the binding mechanism's own index, and layering it
        # here means the call site states `layer=i` once instead of the same
        # `i` under two names.
        self._attn = ThopAttention()
        for i in range(n):
            self._attn.bind_layered(i, attention_sinks=w[f"l{i}_sinks"], local_layer_idx=i)
        self._attn.bind_const(
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_size=cfg.head_dim,
            **_CALL_CONSTANTS,
        )

        self._o_proj = CublasMm()
        for i in range(n):
            self._o_proj.bind_layered(i, mat_b=w[f"l{i}_o"].t(), bias=w[f"l{i}_o_bias"])

        self._norm2 = FlashinferFusedAddRmsnorm()
        for i in range(n):
            self._norm2.bind_layered(i, weight=w[f"l{i}_norm2"])
        self._norm2.bind_const(eps=cfg.rms_norm_eps)

        self._router = CublasMm()
        for i in range(n):
            self._router.bind_layered(i, mat_b=w[f"l{i}_router"].t(), bias=w[f"l{i}_router_bias"])

        self._quant = Mxfp8Quantize()
        # swizzled_layout=False: the MoE op reads the activation scales as a
        # linear (row-major) buffer. The 128x4 swizzled order has the same
        # byte count whenever num_tokens is a multiple of 128 -- which every
        # decode CUDA-graph batch of 128 or 256 is -- and is then accepted
        # silently as a wrong answer, so this is spelled out rather than
        # left to the quantizer's default.
        self._quant.bind_const(swizzled_layout=False, alignment=_FC1_K_ALIGN)

        self._moe = Mxe4m3Mxe2m1BlockScaleMoeRunner()
        for i in range(n):
            self._moe.bind_layered(
                i,
                gemm1_weights=w[f"l{i}_fc1_w"],
                gemm1_weights_scale=w[f"l{i}_fc1_s"],
                gemm1_bias=w[f"l{i}_fc1_b"],
                gemm2_weights=w[f"l{i}_fc2_w"],
                gemm2_weights_scale=w[f"l{i}_fc2_s"],
                gemm2_bias=w[f"l{i}_fc2_b"],
            )
        # alpha, beta and the clamp limit: the op builds its own per-expert
        # vectors from a count and a scalar -- see `bind_glu`. This target
        # runs tp1, so every expert is local and `cfg.num_local_experts` is
        # the whole count, not a per-layer table: every layer shares the
        # same checkpoint constants (alpha, beta) and config scalar
        # (swiglu_limit).
        self._moe.bind_glu(cfg.num_local_experts, cfg.swiglu_limit, device)
        self._moe.bind_const(
            # The router bias rides the router GEMM's own epilogue instead
            # (see the call site): routing_bias is bound to None here so a
            # reader never has to find a bare positional None at the call and
            # wonder what it is.
            routing_bias=None,
            num_experts=cfg.num_local_experts,
            top_k=cfg.num_experts_per_tok,
            n_group=None,
            topk_group=None,
            intermediate_size=core.inter_pad,
            valid_hidden_size=cfg.hidden_size,
            valid_intermediate_size=cfg.intermediate_size,
            local_expert_offset=0,
            local_num_experts=cfg.num_local_experts,
            routed_scaling_factor=None,
            # Renormalize routing (top-k first, then fp32 softmax over the
            # selected logits) and the only gated-activation kernel in this
            # dtype family.
            routing_method_type=1,
            act_type=0,
        )

        self._norm_next = FlashinferFusedAddRmsnorm()
        for i in range(n):
            next_w = w[f"l{i + 1}_norm1"] if i + 1 < n else w["final_norm"]
            self._norm_next.bind_layered(i, weight=next_w)
        self._norm_next.bind_const(eps=cfg.rms_norm_eps)

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        return _build_step_args(md, num_contexts=md.num_contexts, num_ctx_tokens=md.num_ctx_tokens)

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

        # advance_step_generation must run before any bind_const this forward
        # makes: it is what lets validating() catch a target that forgot to
        # rebind and is running an op against last step's metadata.
        advance_step_generation()
        self._attn.bind_const(**self.step_args(attn_metadata))

        # attention_window_size is per-layer, and rebound every forward
        # rather than once: its full-attention value is
        # attn_metadata.max_seq_len, which reads the KV cache manager's
        # *resolved* max_seq_len, not known until the cache is sized --
        # well after `PrefillTarget.__init__` runs at post-load. The value
        # itself never changes between forwards once the cache exists (no
        # attention backend reassigns `max_seq_len` per step -- see
        # `AttentionMetadata` in interface.py -- and the only writes to it
        # happen in `KVCacheManager.__init__` and engine construction, both
        # long done by the time a forward reaches this target), so this
        # recomputes the same 36-entry table every step. That cost is
        # accepted in exchange for not carrying a one-shot flag and the
        # branch that read it.
        full_window = attn_metadata.max_seq_len
        for i, sliding in enumerate(core.sliding):
            self._attn.bind_layered(
                i, attention_window_size=core.window if sliding else full_window
            )

        pos = torch.reshape(position_ids, [-1])

        if inputs_embeds is None:
            h = nn.functional.embedding(input_ids, core.w["embed"])
        else:
            h = inputs_embeds
        num_tokens = h.shape[0]
        dt = h.dtype
        attn_out = torch.empty(
            [num_tokens, cfg.num_attention_heads * cfg.head_dim], dtype=dt, device=h.device
        )

        x = self._norm0(h)
        residual = h
        for i in range(cfg.num_hidden_layers):
            qkv = self._qkv(x, layer=i)
            qkv = self._qk_rope(qkv, position_ids=pos)
            attn_out = self._attn(q=qkv, output=attn_out, layer=i)
            o = self._o_proj(attn_out, layer=i)
            o, residual = self._norm2(o, residual, layer=i)
            # The router bias rides the GEMM epilogue: the MoE op silently
            # ignores routing_bias on this routing method -- routing_bias
            # itself is bound to None on self._moe (see __init__).
            router_logits = self._router(o, layer=i)
            # The quantizer owns the hidden widening 2880 -> fc1_k_pad: it
            # zero-fills the padded columns and their scale bytes, which
            # multiply zero-valued padded weights.
            hidden_fp8, hidden_sf = self._quant(o)
            moe = self._moe(
                routing_logits=router_logits,
                hidden_states=hidden_fp8,
                hidden_states_scale=hidden_sf,
                layer=i,
            )
            moe, residual = self._norm_next(moe, residual, layer=i)
            x = moe
        return x


class DecodeTarget(Target):
    """The specialization: routed to only when there are no context rows.

    It states the two counts rather than reading them. Not an optimization --
    they are per-capture host constants either way -- but it is what makes the
    invariant checkable: a decode target that reads the phase back has stopped
    being one, and the source gate in test_modeling_v2_claims.py can see that.

    gpt_oss is not MLA, so `TrtllmAttention` accepts `mixed` and the
    context/generation split stays inside the C++ dispatcher: this forward has
    no phase branch to divide. `PrefillTarget`, above, runs the identical body
    -- the two differ only in `step_args` -- as a separate, independent class:
    targets in this tree do not share code with each other, even when sharing
    would be free. A model whose phases run different computations --
    deepseek's MLA, where generation works in latent space and context
    materializes K and V -- overrides `forward` itself instead.
    """

    def __init__(self, core: ModelingV2Core) -> None:
        """Bind every op this target calls, once, against the real weights.

        `Target.__init__` only stores `self.core`; everything below is this
        target's own addition, and is why gpt_oss needed one. Per-layer
        weight tables go into each catalog entry's `bind_layered`, one layer
        at a time, built straight from `core.w` and keeping the `.t()` views
        the hot-path GEMMs already used (zero-copy). Configuration comes
        from `cfg` throughout: `core` carries none of it past what is
        genuinely derived (`inter_pad`, `fc1_k_pad`, `fc2_rows_pad`, `theta`,
        the `yarn_*` quantities) or read per layer by the forward
        (`sliding`, `window`) -- see `ModelingV2Core.__init__` for which is
        which. The one op-owned fixture built here -- `fused_qk_norm_rope`'s
        inert q/k norm weight -- is an input that op needs to satisfy its
        own signature, not state the model computes; the MoE runner's
        per-expert activation vectors are the same kind of fixture, but the
        op builds those itself now, via `bind_glu`.
        """
        super().__init__(core)
        cfg = core.model_config.pretrained_config
        w = core.w
        device = w["final_norm"].device
        dtype = w["final_norm"].dtype
        n = cfg.num_hidden_layers

        self._norm0 = FlashinferRmsnorm()
        self._norm0.bind_const(weight=w["l0_norm1"], eps=cfg.rms_norm_eps)

        self._qkv = CublasMm()
        for i in range(n):
            self._qkv.bind_layered(i, mat_b=w[f"l{i}_qkv"].t(), bias=w[f"l{i}_qkv_bias"])

        # fused_qk_norm_rope requires valid q/k norm weight tensors even with
        # is_qk_norm=False, where their values are unused; built from the
        # scalars the op needs (head_dim, dtype, device) rather than copied
        # off a reference tensor, which would tie this fixture to whichever
        # tensor happened to be handy when it was written. rotary_dim is
        # cfg.head_dim directly -- equal for this checkpoint, and not worth
        # a third name for the same value (see `ModelingV2Core.__init__`).
        no_qk_norm = torch.zeros(cfg.head_dim, dtype=dtype, device=device)
        self._qk_rope = FusedQkNormRope()
        self._qk_rope.bind_const(
            num_heads_q=cfg.num_attention_heads,
            num_heads_k=cfg.num_key_value_heads,
            num_heads_v=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            rotary_dim=cfg.head_dim,
            eps=cfg.rms_norm_eps,
            q_weight=no_qk_norm,
            k_weight=no_qk_norm,
            base=core.theta,
            is_neox=True,
            factor=core.yarn_factor,
            low=core.yarn_low,
            high=core.yarn_high,
            attention_factor=core.yarn_attn_factor,
            is_qk_norm=False,
        )

        # attention_window_size is absent here -- it is not knowable until a
        # forward is underway (its full-attention value is
        # attn_metadata.max_seq_len, the KV cache manager's *resolved* size,
        # not known until after this target's __init__ runs at post-load),
        # so every forward binds it fresh instead -- see the comment in
        # `forward`. local_layer_idx is layered rather than passed at the
        # call site even though it is exactly `layer`'s own value: it is a
        # genuine op argument (the row thop_attention reads out of the pool
        # mapping), not the binding mechanism's own index, and layering it
        # here means the call site states `layer=i` once instead of the same
        # `i` under two names.
        self._attn = ThopAttention()
        for i in range(n):
            self._attn.bind_layered(i, attention_sinks=w[f"l{i}_sinks"], local_layer_idx=i)
        self._attn.bind_const(
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_size=cfg.head_dim,
            **_CALL_CONSTANTS,
        )

        self._o_proj = CublasMm()
        for i in range(n):
            self._o_proj.bind_layered(i, mat_b=w[f"l{i}_o"].t(), bias=w[f"l{i}_o_bias"])

        self._norm2 = FlashinferFusedAddRmsnorm()
        for i in range(n):
            self._norm2.bind_layered(i, weight=w[f"l{i}_norm2"])
        self._norm2.bind_const(eps=cfg.rms_norm_eps)

        self._router = CublasMm()
        for i in range(n):
            self._router.bind_layered(i, mat_b=w[f"l{i}_router"].t(), bias=w[f"l{i}_router_bias"])

        self._quant = Mxfp8Quantize()
        # swizzled_layout=False: the MoE op reads the activation scales as a
        # linear (row-major) buffer. The 128x4 swizzled order has the same
        # byte count whenever num_tokens is a multiple of 128 -- which every
        # decode CUDA-graph batch of 128 or 256 is -- and is then accepted
        # silently as a wrong answer, so this is spelled out rather than
        # left to the quantizer's default.
        self._quant.bind_const(swizzled_layout=False, alignment=_FC1_K_ALIGN)

        self._moe = Mxe4m3Mxe2m1BlockScaleMoeRunner()
        for i in range(n):
            self._moe.bind_layered(
                i,
                gemm1_weights=w[f"l{i}_fc1_w"],
                gemm1_weights_scale=w[f"l{i}_fc1_s"],
                gemm1_bias=w[f"l{i}_fc1_b"],
                gemm2_weights=w[f"l{i}_fc2_w"],
                gemm2_weights_scale=w[f"l{i}_fc2_s"],
                gemm2_bias=w[f"l{i}_fc2_b"],
            )
        # alpha, beta and the clamp limit: the op builds its own per-expert
        # vectors from a count and a scalar -- see `bind_glu`. This target
        # runs tp1, so every expert is local and `cfg.num_local_experts` is
        # the whole count, not a per-layer table: every layer shares the
        # same checkpoint constants (alpha, beta) and config scalar
        # (swiglu_limit).
        self._moe.bind_glu(cfg.num_local_experts, cfg.swiglu_limit, device)
        self._moe.bind_const(
            # The router bias rides the router GEMM's own epilogue instead
            # (see the call site): routing_bias is bound to None here so a
            # reader never has to find a bare positional None at the call and
            # wonder what it is.
            routing_bias=None,
            num_experts=cfg.num_local_experts,
            top_k=cfg.num_experts_per_tok,
            n_group=None,
            topk_group=None,
            intermediate_size=core.inter_pad,
            valid_hidden_size=cfg.hidden_size,
            valid_intermediate_size=cfg.intermediate_size,
            local_expert_offset=0,
            local_num_experts=cfg.num_local_experts,
            routed_scaling_factor=None,
            # Renormalize routing (top-k first, then fp32 softmax over the
            # selected logits) and the only gated-activation kernel in this
            # dtype family.
            routing_method_type=1,
            act_type=0,
        )

        self._norm_next = FlashinferFusedAddRmsnorm()
        for i in range(n):
            next_w = w[f"l{i + 1}_norm1"] if i + 1 < n else w["final_norm"]
            self._norm_next.bind_layered(i, weight=next_w)
        self._norm_next.bind_const(eps=cfg.rms_norm_eps)

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        return _build_step_args(md, num_contexts=0, num_ctx_tokens=0)

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

        # advance_step_generation must run before any bind_const this forward
        # makes: it is what lets validating() catch a target that forgot to
        # rebind and is running an op against last step's metadata.
        advance_step_generation()
        self._attn.bind_const(**self.step_args(attn_metadata))

        # attention_window_size is per-layer, and rebound every forward
        # rather than once: its full-attention value is
        # attn_metadata.max_seq_len, which reads the KV cache manager's
        # *resolved* max_seq_len, not known until the cache is sized --
        # well after `DecodeTarget.__init__` runs at post-load. The value
        # itself never changes between forwards once the cache exists (no
        # attention backend reassigns `max_seq_len` per step -- see
        # `AttentionMetadata` in interface.py -- and the only writes to it
        # happen in `KVCacheManager.__init__` and engine construction, both
        # long done by the time a forward reaches this target), so this
        # recomputes the same 36-entry table every step. That cost is
        # accepted in exchange for not carrying a one-shot flag and the
        # branch that read it.
        full_window = attn_metadata.max_seq_len
        for i, sliding in enumerate(core.sliding):
            self._attn.bind_layered(
                i, attention_window_size=core.window if sliding else full_window
            )

        pos = torch.reshape(position_ids, [-1])

        if inputs_embeds is None:
            h = nn.functional.embedding(input_ids, core.w["embed"])
        else:
            h = inputs_embeds
        num_tokens = h.shape[0]
        dt = h.dtype
        attn_out = torch.empty(
            [num_tokens, cfg.num_attention_heads * cfg.head_dim], dtype=dt, device=h.device
        )

        x = self._norm0(h)
        residual = h
        for i in range(cfg.num_hidden_layers):
            qkv = self._qkv(x, layer=i)
            qkv = self._qk_rope(qkv, position_ids=pos)
            attn_out = self._attn(q=qkv, output=attn_out, layer=i)
            o = self._o_proj(attn_out, layer=i)
            o, residual = self._norm2(o, residual, layer=i)
            # The router bias rides the GEMM epilogue: the MoE op silently
            # ignores routing_bias on this routing method -- routing_bias
            # itself is bound to None on self._moe (see __init__).
            router_logits = self._router(o, layer=i)
            # The quantizer owns the hidden widening 2880 -> fc1_k_pad: it
            # zero-fills the padded columns and their scale bytes, which
            # multiply zero-valued padded weights.
            hidden_fp8, hidden_sf = self._quant(o)
            moe = self._moe(
                routing_logits=router_logits,
                hidden_states=hidden_fp8,
                hidden_states_scale=hidden_sf,
                layer=i,
            )
            moe, residual = self._norm_next(moe, residual, layer=i)
            x = moe
        return x


@register_auto_model("ModelingV2GptOss120bSm103Tp1")
class ModelingV2GptOss120bSm103Tp1(DecoderModelForCausalLM[ModelingV2Core, PretrainedConfig]):
    def __init__(self, model_config: ModelConfig):
        cfg = model_config.pretrained_config
        # DecoderModelForCausalLM sizes lm_head from the *pretrained* dtype,
        # which this checkpoint's config.json does not declare — leaving
        # lm_head.weight fp32 while every other tensor is bf16, and failing
        # two layers away from here. The engine has already resolved the
        # dtype it will run at; adopt it, rather than let the default of a
        # missing field decide. Only fills the gap: a declared dtype wins.
        if cfg.torch_dtype is None:
            cfg.torch_dtype = model_config.torch_dtype
        super().__init__(
            ModelingV2Core(model_config),
            config=model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )

    def load_weights(self, weights, *args, **kwargs):
        _weights.load(self, weights)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.build_layer_views()
