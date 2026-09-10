# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Staircase target: deepseek-r1-0528-nvfp4 / sm_103 / dep4 — self-contained modeling code.

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

Against the `deepseek-v3-lite-nvfp4/dep4` sibling (not in this batch) — same architecture
family, same topology, same quantization shape — five things differ, and each
is re-derived here rather than inherited:

* **the query path is a LoRA pair.** `q_lora_rank: 1536`, so the query is
  `q_b_proj(q_a_layernorm(q_a_proj(x)))` — two GEMMs and a norm — where the
  lite checkpoint's `q_lora_rank: null` projects directly. Nothing downstream
  changes: `q_b_proj` produces the same `[T, H*(nope+rope)]` rows;
* **routing is group-limited.** `n_group 8`, `topk_group 4`: `noaux_tc_op`
  scores 8 contiguous groups of 32 experts by their two best bias-corrected
  scores, keeps the best 4 groups, and takes top-8 inside them. The lite
  sibling runs the ungrouped case. The MoE runner's own `n_group`/`topk_group`
  stay inert — routing is done before the call, on either checkpoint;
* **the rope table is YaRN-scaled** (factor 40 over an original 4096-position
  window, `beta_fast` 32, `beta_slow` 1), and the model's YaRN attention
  temperature `mscale = 0.1*ln(40) + 1` rides in **`q_scaling = 1/mscale^2`**
  rather than in the table: the table's own amplitude is
  `m(mscale)/m(mscale_all_dim)` = exactly 1.0 because this config sets both to
  1.0. The op reads the table's *content* and `q_scaling`; the seven scalar
  rope arguments beside them are inert (measured — see thop_attention.md);
* **the latent KV pool is fp8-e4m3**, which changes what every one of the five
  MLA-family calls a layer issues does — see below;
* **scale.** 128 query heads (attention DP replicates them, so every rank runs
  all 128), 64 routed experts per rank, and a 163840-position rope table.

**Layer 61 — the checkpoint's bf16 MTP module — is a second forward path this
file also carries, and it exists only when a `configs/` variant turns it on.**
Under the target's identity config (`llm_args.yaml`, no `speculative_config`)
nothing below MTP is declared, layer 61's 790 keys stay a predicted non-load in
the weight manifest, and the shell's forward is a plain call to the inherited
base — bit-identical to the assembly the accuracy gate was measured on. With
`configs/mtp{1,2,3}.yaml` the engine resolves a `spec_config` onto the model
config before the model is built, the core declares the module's parameters,
and the shell builds a draft-model container plus the runtime's spec worker.
`MTPLayer` below is the module's forward; `docs/models/multi-token-prediction.md`
is its semantics (the checkpoint ships no reference implementation of it and
neither does transformers), and `docs/references/trtllm-runtime-integration.md`
§13 is the runtime binding. Two consequences the rest of this file carries:
`predicted_tokens_per_seq` is per call site rather than inert, because a
generation request under MTP arrives carrying its whole draft chain; and the
`spec_decoding_*` group stays inert, because on a trtllm-gen arch a
linear-tree draft has
its mask machinery forced off and drafting reaches the attention ops through
`predicted_tokens_per_seq` alone.

**The fp8 latent pool, and the four things it moves.** Nothing validates the
fp8 round trip at any layer: the write scale, the read scale and the two
folded FMHA scales are independent roles with no relation checked anywhere, so
a mistake here is silently mis-scaled output rather than an error.

* **the context append and the cache gather** take the KV scaling factor `s`
  as a write-side `1/s` and a read-side `s`. This checkpoint's 122 per-layer
  `k_scale`/`v_scale` tensors are all exactly 1.0 (loaded and asserted in
  `derive_after_load`), and `s = 1.0` is the **only** correct value on the fp8
  MLA context path — both context flavors quantize q/k/v at 1.0 while applying
  `s^2`/`s` as if they had not, so any other `s` is silently wrong. Both scale
  arguments are therefore passed as `None`, which the ops read as exactly 1.0
  and which is what the engine's own call sites pass;
* **the context FMHA quantizes its own q/k/v to e4m3**, in both flavors. So
  prefill accuracy *is* affected by cache quantization, and a cached-prefix
  context call pays fp8 twice — the gather dequantizes the cached latent rows
  off the pool and the FMHA quantizes the up-projected result straight back;
* **the decode producers are ordered, not concurrent.** `mla_rope_generation`
  does not write `fused_q` here — it **reads** `fused_q[..., :C]` to build
  `quant_q_buffer`, the query the decode FMHA actually consumes. The absorbed-q
  BMM must have finished before it. This forward issues both on the ambient
  stream in that order, which is what makes it safe; the bf16 reading (the two
  producers write disjoint halves and may overlap) is a silent race here;
* **the decode FMHA reads neither kv scale tensor.** Its query comes from
  `quant_q_buffer` and its two scales from `mla_bmm1_scale[1]` and
  `mla_bmm2_scale[0]`, both written by `mla_rope_generation` from `q_scaling`,
  the MLA dims and the read-side factor. The caller owns the dequantization
  entirely.

**The parallel segment.** `dep4` is `tensor_parallel_size: 4` plus
`moe_expert_parallel_size: 4` plus `enable_attention_dp: true`: the requests
are split, not the heads.

* **attention, the q-LoRA pair, both dense MLP shapes, the shared expert, the
  router, the embedding, the norms and the residual stream are replicated**,
  and each rank runs them over its own tokens only — 128 query heads per rank.
  A rank's `o_proj` output is complete, so there is no attention-side
  collective at all;
* **the MoE stays expert-parallel** (`moe_tp_size == 1`): rank `r` holds
  experts `[64r, 64r+64)` whole. Every token must reach every window, so the
  rank's tokens are **gathered** before the router and the four windows'
  partials are **reduce-scattered** back — one `comm/allgather` and one
  `comm/reducescatter` per MoE layer, 116 per forward, none in layers 0-2;
* **the gather goes before the router GEMM, and that placement is
  load-bearing.** Expert parallelism rests on the four windows tiling the
  routing space exactly once, which needs every rank to select the same experts
  for the same token. Ranks hold different tokens here, so the invariant is
  restored by routing on the gathered full token set: identical bytes in,
  replicated deterministic router GEMM and `noaux_tc_op`, identical top-8 out.
  Routing locally and gathering afterwards would break it with no error;
* **the reduce-scatter is crossed in bf16**: the op sums, and it sums
  `float8_e4m3fn` as raw bytes rather than as floats, so a post-quantization
  return trip is silently wrong (the gather is a byte move and would survive
  it — the asymmetry is the trap);
* **`lm_head` is replicated** — the shell builds the whole `[vocab, hidden]` on
  every rank under attention DP, because a rank's logits rows are its own
  tokens' and no other rank computed them.

The expert call is **chunked** to `_MOE_MAX_T` rows: the gathered token set
reaches `4 * max_num_tokens` = 32768 and both `fp4_block_scale_moe_runner` and
`noaux_tc_op` are certified to 8192. That bound is a certification boundary,
not a tuning knob.

Weights are target-owned: a flat ParameterDict declared here (HF [out, in]
storage so checkpoint rows copy in unchanged, plus the kernel-ready expert
stacks weights.py builds during the load), with the column-major GEMM views,
the MLA absorption operands, the rope table and every NVFP4 call scalar
derived once after load. The registration shell inherits
DecoderModelForCausalLM for lm_head, packed-batch logits gathering, and the
meta-init/load/post-load hooks.

The import-time and first-forward contract checks below fail fast on drift.
"""

import math
from typing import cast

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
from tensorrt_llm._torch.speculative import get_spec_worker
from tensorrt_llm._torch.staircase.catalog.activation.flashinfer_silu_and_mul import (  # noqa: E501
    flashinfer_silu_and_mul,
)
from tensorrt_llm._torch.staircase.catalog.attention.load_paged_kv_cache_for_mla import (  # noqa: E501
    load_paged_kv_cache_for_mla,
)
from tensorrt_llm._torch.staircase.catalog.attention.mla_rope_append_paged_kv_assign_q import (  # noqa: E501
    mla_rope_append_paged_kv_assign_q,
)
from tensorrt_llm._torch.staircase.catalog.attention.mla_rope_generation import mla_rope_generation
from tensorrt_llm._torch.staircase.catalog.attention.thop_attention import thop_attention
from tensorrt_llm._torch.staircase.catalog.comm.allgather import allgather
from tensorrt_llm._torch.staircase.catalog.comm.reducescatter import reducescatter
from tensorrt_llm._torch.staircase.catalog.gemm.bmm_out import bmm_out
from tensorrt_llm._torch.staircase.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch.staircase.catalog.gemm.nvfp4_gemm import nvfp4_gemm
from tensorrt_llm._torch.staircase.catalog.moe.fp4_block_scale_moe_runner import (  # noqa: E501
    fp4_block_scale_moe_runner,
)
from tensorrt_llm._torch.staircase.catalog.moe.fused_moe import fused_moe
from tensorrt_llm._torch.staircase.catalog.moe.noaux_tc_op import noaux_tc_op
from tensorrt_llm._torch.staircase.catalog.norm.flashinfer_fused_add_rmsnorm import (  # noqa: E501
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch.staircase.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm
from tensorrt_llm._torch.staircase.catalog.quantization.fp4_quantize import fp4_quantize
from tensorrt_llm._torch.staircase.catalog.torch.add import add
from tensorrt_llm._torch.staircase.catalog.torch.concat import concat
from tensorrt_llm._torch.staircase.catalog.torch.copy_ import copy_
from tensorrt_llm._torch.staircase.catalog.torch.embedding import embedding
from tensorrt_llm._torch.staircase.catalog.torch.empty import empty
from tensorrt_llm._torch.staircase.catalog.torch.expand import expand
from tensorrt_llm._torch.staircase.catalog.torch.pad import pad
from tensorrt_llm._torch.staircase.catalog.torch.reshape import reshape
from tensorrt_llm._torch.staircase.catalog.torch.split import split
from tensorrt_llm._torch.staircase.catalog.torch.transpose import transpose
from tensorrt_llm._torch.staircase.catalog.torch.view_dtype import view_dtype

from . import weights as _weights

# The GPU architecture this target IS. Routing will not send another one here,
# but a direct instantiation could, and the certification is per arch: this
# assert is what the version pin used to be. In-tree the version moves with
# the code, so pinning it is meaningless; the architecture does not.
_SM = (10, 3)


def _check_static_contract() -> None:
    """Import-time fail-fast: op symbol existence. The list is the forward's
    trtllm call set plus `block_scale_interleave`, which the load-time
    expert/scale relayout in weights.py depends on."""
    for op in (
        "cublas_mm",
        "bmm_out",
        "nvfp4_gemm",
        "flashinfer_rmsnorm",
        "flashinfer_fused_add_rmsnorm",
        "flashinfer_silu_and_mul",
        "fp4_quantize",
        "noaux_tc_op",
        "fp4_block_scale_moe_runner",
        "fused_moe",
        "mla_rope_generation",
        "mla_rope_append_paged_kv_assign_q",
        "load_paged_kv_cache_for_mla",
        "allgather",
        "reducescatter",
        "block_scale_interleave",
    ):
        assert hasattr(torch.ops.trtllm, op), f"missing op trtllm::{op}"
    from tensorrt_llm.bindings.internal import thop

    assert hasattr(thop, "attention"), "missing pybind thop.attention"


_check_static_contract()

# Metadata fields consumed each step (sourcing mirrors the in-tree
# FallbackFmha for this trtllm version; existence checked at first forward).
# The tail group is read only by the first-forward contract check: this
# target holds those features at the MLA columns' inert values, and the
# asserts make that honest instead of silently dropping an enabled feature.
_STEP_FIELDS = (
    "all_rank_num_tokens",
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
    "num_seqs",
    "trtllm_gen_jit_warmup",
    "effective_beam_width",
    "cache_indirection",
    "block_ids_per_seq",
    "is_cross",
    "is_spec_decoding_enabled",
    "use_spec_decoding",
    "flash_mla_tile_scheduler_metadata",
    "flash_mla_num_splits",
    # Added between 1.3.0rc21 and 1.3.0rc26. Both are engine-prepared
    # per-instance constants (max_num_sequences defaults to max_num_requests;
    # the tree-mask flag is set from is_spec_dec_dynamic_tree, and this
    # target's MTP is a linear tree), so they project like the rest.
    "max_num_sequences",
    "force_prepare_spec_dec_tree_mask",
)

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
# which None is read as exactly. This checkpoint's k_scale/v_scale are 1.0,
# asserted after load.
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
    # Added between 1.3.0rc21 and 1.3.0rc26; held at the values the op had
    # before they existed, which are also what the in-tree caller passes on
    # this path. kv_norm_weight is not merely a default: non-None would fold
    # the kv_a_layernorm into the KV kernel and make it read latent_cache
    # RAW, and this target normalizes that itself -- passing the weight would
    # normalize twice. skip_correction is a lossy trtllm-gen MLA option
    # (SM100/SM103, off by default upstream); enabling it is a configs/
    # variant's business, not the identity assembly's.
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
# statistics (docs/models/multi-token-prediction.md), which contradicts the
# DeepSeek-V3 report's `M_k[RMSNorm(h); RMSNorm(e)]` notation and agrees with
# the parameter's name. Nothing in the checkpoint's metadata pins it and
# nothing downstream detects a flip: the drafts are simply rejected, so the
# acceptance rate is what confirms it end to end.
_MTP_EMBED_BLOCK_FIRST = True
# thop_attention's certified q_scaling values over an fp8 latent pool are 1.0
# and DeepSeek-R1's YaRN temperature; the config-derived value is checked
# against the latter so a config change lands outside the certified column
# loudly rather than silently.
_CERTIFIED_YARN_Q_SCALING = 0.5336594


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
    assert len(counts) == dp_size, (counts, dp_size)
    assert counts[rank] == rows, (
        f"rank {rank} holds {rows} MTP rows but the worker's "
        f"all_rank_num_tokens says {counts[rank]} ({counts}); the collectives "
        "would gather the wrong split"
    )
    return max(counts)


def _yarn_mscale(factor: float, mscale: float) -> float:
    """DeepSeek's YaRN magnitude scaling `m(x)`: 1.0 at or below factor 1,
    `0.1 * x * ln(factor) + 1` above it."""
    if factor <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(factor) + 1.0


class StaircaseCore(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = model_config.pretrained_config
        assert cfg is not None

        # This target IS this topology and this geometry — assert, never
        # adapt. Read the topology off the mapping the engine built, never
        # off the path segment: the segment names the intent.
        assert torch.cuda.get_device_capability() == _SM, (
            f"target certified on sm_{_SM[0]}{_SM[1]}, running on "
            f"sm_{''.join(map(str, torch.cuda.get_device_capability()))}"
        )
        mapping = model_config.mapping
        assert mapping.world_size == 4, f"dep4 target, world size {mapping.world_size}"
        assert mapping.tp_size == 4, f"dep4 target, tp_size {mapping.tp_size}"
        assert mapping.pp_size == 1, "pipeline parallelism is not implemented"
        assert mapping.moe_ep_size == 4, f"dep4 target, ep {mapping.moe_ep_size}"
        assert mapping.moe_tp_size == 1, (
            "the routed experts are split by expert parallelism only; a "
            f"moe_tp_size of {mapping.moe_tp_size} splits them a second way"
        )
        assert mapping.enable_attention_dp, (
            "attention data parallelism is what the dep4 segment declares; "
            "without it attention would be head-split (that is tep4)"
        )
        self.rank = mapping.rank
        self.ep_rank = mapping.moe_ep_rank
        self.ep_size = mapping.moe_ep_size
        # The collective group: at pp_size 1 with one expert-parallel split,
        # every rank participates in both halves of the MoE round trip, so
        # the group is the whole world.
        self.dp_group = list(range(mapping.world_size))
        self.dp_size = mapping.world_size

        dt = model_config.torch_dtype
        assert dt == torch.bfloat16, f"bf16 target, engine resolved {dt}"
        assert cfg.torch_dtype == torch.bfloat16, cfg.torch_dtype
        assert not cfg.tie_word_embeddings, "untied lm_head"
        assert not cfg.attention_bias, "no q/kv/o bias anywhere"
        assert cfg.hidden_act == "silu", "SwiGLU over a silu gate"
        # The NVFP4 recipe covers the MLP only; the latent KV pool is fp8-e4m3
        # per the checkpoint's own quant config, which is what selects
        # quant_mode on every MLA call. Derived, not hard-coded: a checkpoint
        # declaring no KV quantization is a bf16-pool target and a different
        # assembly.
        kv_algo = model_config.quant_config.kv_cache_quant_algo
        assert kv_algo is not None and str(kv_algo).upper().endswith("FP8"), (
            f"this target is the fp8-e4m3 latent-pool assembly; the checkpoint "
            f"declares kv_cache_quant_algo {kv_algo!r}"
        )
        self.quant_mode = _QUANT_MODE_FP8_KV

        self.num_layers = cfg.num_hidden_layers
        self.hidden = cfg.hidden_size
        self.eps = cfg.rms_norm_eps
        self.vocab = cfg.vocab_size
        # The checkpoint ships `num_nextn_predict_layers` extra decoder layers
        # past `num_hidden_layers` for multi-token prediction — on this one a
        # single bf16 layer 61 with its own 256 experts, embedding, eh_proj,
        # two extra norms and an output head.
        self.mtp_layers = int(getattr(cfg, "num_nextn_predict_layers", 0) or 0)
        # Whether this engine drafts. The checkpoint decides the *mode*
        # (`num_nextn_predict_layers: 1` -> MTP-Eagle one-model, one layer
        # replayed `max_draft_len` times); a `configs/` variant's
        # `speculative_config` decides whether it runs at all, and the engine
        # has already resolved that onto `model_config.spec_config` by the time
        # the model is built. With it absent — the target's identity config —
        # layer 61 is not part of this model at all: its keys stay a predicted
        # non-load in the weight manifest and nothing below is declared.
        spec_config = getattr(model_config, "spec_config", None)
        self.mtp_enabled = spec_config is not None
        if self.mtp_enabled:
            assert self.mtp_layers == 1, (
                "this target implements the one-layer MTP-Eagle module the "
                f"checkpoint ships; num_nextn_predict_layers is {self.mtp_layers}"
            )

        # MLA geometry. num_key_value_heads is 128 in this config but MLA has
        # no separate KV heads: the context call runs Hq == Hkv == heads over
        # head_size nope+rope, the generation call runs Hq == heads against a
        # single latent KV head of width kv_lora+rope. Attention is
        # replicated under DP, so every rank runs the whole head set.
        self.heads = cfg.num_attention_heads
        self.nope = cfg.qk_nope_head_dim
        self.rope = cfg.qk_rope_head_dim
        self.v_dim = cfg.v_head_dim
        self.kv_lora = cfg.kv_lora_rank
        self.q_lora = cfg.q_lora_rank
        self.qk_dim = self.nope + self.rope
        self.lat_dim = self.kv_lora + self.rope
        # The query is a LoRA pair here: q_a_proj -> q_a_layernorm -> q_b_proj.
        # A checkpoint with q_lora_rank null projects directly and needs the
        # single-q_proj path instead (that is the deepseek-v3-lite sibling).
        assert isinstance(self.q_lora, int) and self.q_lora > 0, (
            "this target implements the q-LoRA query path; a checkpoint with "
            f"q_lora_rank {self.q_lora!r} needs the direct q_proj path"
        )
        assert cfg.num_key_value_heads == cfg.num_attention_heads, "MLA: Hkv == Hq"
        # thop_attention aborts the process on a head_size its FMHA kernels
        # do not carry, so both call shapes are pinned here.
        assert self.qk_dim == 192 and self.lat_dim == 576 and self.v_dim == 128, (
            f"certified MLA head dims are 192 (context) / 576 (generation) / "
            f"128 (v), got {self.qk_dim} / {self.lat_dim} / {self.v_dim}"
        )
        # The MLA generation phase compiles a decode kernel per head count, so
        # the count is a certification axis rather than a free shape — and over
        # an fp8-e4m3 latent pool 128 is the *only* certified count.
        assert self.heads == 128, (
            f"thop_attention certifies MLA over an fp8 latent pool at 128 query "
            f"heads only; this topology yields {self.heads}"
        )

        # RoPE: GPT-J interleaved pairs over the rope slice, YaRN-scaled. The
        # engine hands either the flat pre-migration fields or the
        # transformers-5.x rope_parameters dict; both shapes are resolved and
        # every scalar the table depends on is asserted rather than defaulted.
        rope_cfg = getattr(cfg, "rope_scaling", None) or getattr(cfg, "rope_parameters", None)
        assert isinstance(rope_cfg, dict), f"YaRN rope config expected, got {rope_cfg!r}"
        rope_kind = rope_cfg.get("type", rope_cfg.get("rope_type"))
        assert rope_kind == "yarn", f"this target builds a YaRN table, got {rope_kind!r}"
        theta = rope_cfg.get("rope_theta", getattr(cfg, "rope_theta", None))
        assert theta is not None and float(theta) > 0.0, "rope theta"
        self.theta = float(theta)
        self.rope_factor = float(rope_cfg["factor"])
        self.rope_orig_max = int(rope_cfg["original_max_position_embeddings"])
        self.beta_fast = float(rope_cfg["beta_fast"])
        self.beta_slow = float(rope_cfg["beta_slow"])
        mscale = float(rope_cfg["mscale"])
        mscale_all_dim = float(rope_cfg["mscale_all_dim"])
        assert self.rope_factor > 1.0 and self.rope_orig_max > 0, rope_cfg
        assert getattr(cfg, "rope_interleave", True), (
            "the MLA ops apply GPT-J (interleaved-pair) rope in kernel"
        )
        self.max_pos = cfg.max_position_embeddings
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
        assert abs(self.q_scaling - _CERTIFIED_YARN_Q_SCALING) < 1e-6, (
            f"q_scaling {self.q_scaling} is outside thop_attention's certified "
            f"fp8-pool column (1.0 and {_CERTIFIED_YARN_Q_SCALING})"
        )
        # Per-call constants: the inert groups above plus the two values this
        # checkpoint derives.
        self._call = dict(_CALL_INERT, quant_mode=self.quant_mode, q_scaling=self.q_scaling)

        # MLP structure: the first `first_k_dense_replace` layers are dense,
        # the rest are routed MoE plus a shared-expert pair fused into one
        # dense linear pair of width n_shared * moe_intermediate_size. Both
        # dense shapes are replicated under attention DP and run over this
        # rank's own tokens.
        self.dense_layers = cfg.first_k_dense_replace
        assert 0 < self.dense_layers < self.num_layers, "mixed dense/MoE stack"
        assert cfg.moe_layer_freq == 1, "every layer past the dense prefix is MoE"
        self.num_experts = cfg.n_routed_experts
        self.topk = cfg.num_experts_per_tok
        self.moe_inter = cfg.moe_intermediate_size
        self.shared_inter = cfg.moe_intermediate_size * cfg.n_shared_experts
        self.dense_inter = cfg.intermediate_size
        assert 0 < self.topk < self.num_experts, "MoE top-k bound"
        # noaux_tc_op is the whole gate: in-kernel sigmoid, bias correction
        # for selection only, group-limited selection, renormalization and the
        # routed_scaling_factor multiply. A checkpoint with norm_topk_prob
        # false cannot use it.
        assert cfg.topk_method == "noaux_tc", cfg.topk_method
        assert cfg.scoring_func == "sigmoid", cfg.scoring_func
        assert cfg.norm_topk_prob, "noaux_tc_op always renormalizes"
        self.n_group = cfg.n_group
        self.topk_group = cfg.topk_group
        # Group-limited routing, which this checkpoint uses and the lite
        # sibling does not. noaux_tc_op's grouped configuration carries four
        # hard limits of its own; each is checked here because the op reports
        # a violation as one opaque "unsupported configuration".
        assert self.n_group > 1 and 1 <= self.topk_group <= self.n_group, (
            f"grouped routing needs 1 <= topk_group <= n_group, got "
            f"{self.topk_group} / {self.n_group}"
        )
        assert self.num_experts % self.n_group == 0, "experts per routing group"
        assert self.num_experts <= 256 and self.num_experts // self.n_group <= 32, (
            "noaux_tc_op's grouped path takes at most 256 experts and 32 per "
            f"group; this config has {self.num_experts} in {self.n_group} groups"
        )
        assert self.topk <= 8, "noaux_tc_op's grouped path takes at most top-8"
        self.routed_scale = float(cfg.routed_scaling_factor)
        # Expert parallelism: the routing space stays global and every rank
        # runs the full top-k over the *gathered* token set, but a rank holds
        # only its own window of experts and the kernel drops every slot
        # outside it. The four windows' outputs sum to the whole layer — that
        # is what the reduce-scatter completes.
        assert self.num_experts % mapping.moe_ep_size == 0, "experts per rank"
        self.local_experts = self.num_experts // mapping.moe_ep_size
        self.expert_offset = self.local_experts * self.ep_rank
        assert self.local_experts == 64, (
            f"fp4_block_scale_moe_runner certifies the four-way split of 256 "
            f"experts (windows of 64); this topology yields {self.local_experts}"
        )
        # fp4_block_scale_moe_runner's block-scale layout rules.
        assert self.hidden % 256 == 0, "MoE hidden must be a multiple of 256"
        assert self.moe_inter % 64 == 0, "MoE intermediate must be a multiple of 64"
        # nvfp4_gemm needs K and N multiples of 32 on both dense linears
        # (gate_up: K=hidden, N=2*inter; down: K=inter, N=hidden), and
        # flashinfer_silu_and_mul vectorizes the up half from element
        # `inter`, so that offset must be 16-element aligned too. The 128x4
        # scale swizzle adds two more: the gate_up scale matrix has 2*inter
        # rows (a multiple of 128) and the down one inter/16 columns (a
        # multiple of 4) — together, inter must be a multiple of 64.
        assert self.hidden % 32 == 0, "nvfp4_gemm operand width"
        for inter in (self.dense_inter, self.shared_inter):
            assert inter % 64 == 0, (
                "MLP intermediate must be a multiple of 64: nvfp4_gemm "
                "operand width, the silu_and_mul half offset, and the 128x4 "
                "scale swizzle's row/column alignment"
            )

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
        for i in range(self.num_layers):
            w[f"l{i}_norm1"] = P(self.hidden)
            w[f"l{i}_qa"] = P(self.q_lora, self.hidden)
            w[f"l{i}_q_norm"] = P(self.q_lora)
            w[f"l{i}_qb"] = P(self.heads * self.qk_dim, self.q_lora)
            w[f"l{i}_kva"] = P(self.lat_dim, self.hidden)
            w[f"l{i}_kv_norm"] = P(self.kv_lora)
            w[f"l{i}_kvb"] = P(self.heads * (self.nope + self.v_dim), self.kv_lora)
            w[f"l{i}_o"] = P(self.hidden, self.heads * self.v_dim)
            # The checkpoint's calibrated fp8 KV-cache scales. They are loaded
            # rather than skipped so `derive_after_load` can assert the value
            # the whole fp8 MLA path depends on.
            w[f"l{i}_k_scale"] = P(1, dtype=f32)
            w[f"l{i}_v_scale"] = P(1, dtype=f32)
            w[f"l{i}_norm2"] = P(self.hidden)
            inter = self.dense_inter if i < self.dense_layers else self.shared_inter
            w[f"l{i}_mlp_gu_w"] = P(2 * inter, self.hidden // 2, dtype=u8)
            w[f"l{i}_mlp_gu_s"] = P(2 * inter * (self.hidden // _SF_VEC), dtype=u8)
            w[f"l{i}_mlp_dn_w"] = P(self.hidden, inter // 2, dtype=u8)
            w[f"l{i}_mlp_dn_s"] = P(self.hidden * (inter // _SF_VEC), dtype=u8)
            for name in (
                "isc1",
                "isc1_up",
                "ws2_1",
                "ws2_1_up",
                "isc2",
                "ws2_2",
            ):
                w[f"l{i}_mlp_{name}"] = P(1, dtype=f32)
            if i < self.dense_layers:
                continue
            e, mi = self.local_experts, self.moe_inter
            w[f"l{i}_router"] = P(self.num_experts, self.hidden)
            # fp32 on this checkpoint (the reference model keeps the
            # correction bias in fp32 whatever the rest of the weights are);
            # noaux_tc_op takes bf16 logits against an fp32 bias and returns
            # weights in the *logits* dtype, which is what the MoE runner
            # demands.
            w[f"l{i}_router_bias"] = P(self.num_experts, dtype=f32)
            w[f"l{i}_fc1_w"] = P(e, 2 * mi, self.hidden // 2, dtype=u8)
            w[f"l{i}_fc1_s"] = P(e, 2 * mi, self.hidden // _SF_VEC, dtype=u8)
            w[f"l{i}_fc2_w"] = P(e, self.hidden, mi // 2, dtype=u8)
            w[f"l{i}_fc2_s"] = P(e, self.hidden, mi // _SF_VEC, dtype=u8)
            # The per-expert NVFP4 scalars stay whole on every rank: they
            # cost 6 floats per expert, and the shared-expert activation
            # scale is asserted against the max over *all* routed experts,
            # which a window could not see. The window is sliced out in
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
                w[f"l{i}_e_{name}"] = P(self.num_experts, dtype=f32)
        w["final_norm"] = P(self.hidden)
        w["embed"] = P(self.vocab, self.hidden)
        # The MTP module at layer index `num_hidden_layers`, declared only when
        # a configs/ variant turned drafting on. Its attention block is
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
            e, mi = self.local_experts, self.moe_inter
            w["mtp_enorm"] = P(self.hidden)
            w["mtp_hnorm"] = P(self.hidden)
            w["mtp_eh"] = P(self.hidden, 2 * self.hidden)
            w["mtp_norm1"] = P(self.hidden)
            w["mtp_qa"] = P(self.q_lora, self.hidden)
            w["mtp_q_norm"] = P(self.q_lora)
            w["mtp_qb"] = P(self.heads * self.qk_dim, self.q_lora)
            w["mtp_kva"] = P(self.lat_dim, self.hidden)
            w["mtp_kv_norm"] = P(self.kv_lora)
            w["mtp_kvb"] = P(self.heads * (self.nope + self.v_dim), self.kv_lora)
            w["mtp_o"] = P(self.hidden, self.heads * self.v_dim)
            w["mtp_k_scale"] = P(1, dtype=f32)
            w["mtp_v_scale"] = P(1, dtype=f32)
            w["mtp_norm2"] = P(self.hidden)
            w["mtp_router"] = P(self.num_experts, self.hidden)
            w["mtp_router_bias"] = P(self.num_experts, dtype=f32)
            # `fused_moe`'s stacked layout: `[E, 2I, H]` with the **up** rows
            # first and the gate rows last (the opposite half order from the
            # dense gate_up linear below, which flashinfer_silu_and_mul reads
            # gate-first), and `[E, H, I]` for FC2. No interleave, no 32-row
            # block shuffle, no swizzle — those belong to the trtllm-gen
            # block-scale runner the trunk uses, not to this one.
            w["mtp_fc1"] = P(e, 2 * mi, self.hidden)
            w["mtp_fc2"] = P(e, self.hidden, mi)
            w["mtp_sh_gu"] = P(2 * self.shared_inter, self.hidden)
            w["mtp_sh_dn"] = P(self.hidden, self.shared_inter)
            w["mtp_head_norm"] = P(self.hidden)
        self.w = w

        self._attn: list | None = None
        self._mlp: list | None = None
        self._moe: list | None = None
        self._mtp: dict | None = None
        self._next_norm: list | None = None
        self._rope: dict | None = None
        self._rope_positions = 0
        self._side_stream: torch.cuda.Stream | None = None
        self._cached_ctx = False
        self._step_contract_checked = False

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
        half = self.rope // 2
        d = torch.arange(half, dtype=torch.float64)
        freq = self.theta ** (2.0 * d / self.rope)
        two_pi = 2.0 * math.pi
        log_theta = math.log(self.theta)
        low = max(
            0.0,
            math.floor(
                self.rope
                * math.log(self.rope_orig_max / (self.beta_fast * two_pi))
                / (2.0 * log_theta)
            ),
        )
        high = min(
            self.rope - 1.0,
            math.ceil(
                self.rope
                * math.log(self.rope_orig_max / (self.beta_slow * two_pi))
                / (2.0 * log_theta)
            ),
        )
        ramp = ((d - low) / max(high - low, 0.001)).clamp(0.0, 1.0)
        inv = ramp / (self.rope_factor * freq) + (1.0 - ramp) / freq
        ang = torch.arange(positions, dtype=torch.float64)[:, None] * inv[None, :]
        cos, sin = ang.cos() * self.rope_amplitude, ang.sin() * self.rope_amplitude
        table = torch.empty(positions, self.rope, 2, dtype=torch.float64)
        table[:, :half, 0] = cos
        table[:, half:, 0] = cos
        table[:, :half, 1] = sin
        table[:, half:, 1] = sin
        return {
            "rotary_cos_sin": table.reshape(1, positions * self.rope * 2)
            .float()
            .to(device)
            .contiguous(),
            "rotary_inv_freq": inv.float().to(device).contiguous(),
            "rope_dim": self.rope,
            "rope_base": self.theta,
        }

    def derive_after_load(self) -> None:
        """Post-load derivation: column-major GEMM views (`.t()` is
        zero-copy), the two MLA absorption operands split out of the
        row-regrouped kv_b_proj, the rope table, and every NVFP4 call scalar
        folded from the checkpoint's per-tensor `input_scale` /
        `weight_scale_2` pairs — the routed ones sliced to this rank's expert
        window, which is where the kernel's `[local_num_experts]` operands
        come from. Meta is over here, so real tensors may be created."""
        w = self.w
        device = w["final_norm"].device
        self._rope_positions = self.max_pos
        self._rope = self._rope_tables(device, self._rope_positions)
        window = slice(self.expert_offset, self.expert_offset + self.local_experts)

        attn, mlp, moe, nxt = [], [], [], []
        hn = self.heads * self.nope
        for i in range(self.num_layers):
            # The fp8 latent pool is written at 1/s and read at s, and the two
            # roles live in different ops with no relation checked anywhere in
            # the chain. This target passes None for both, which every op reads
            # as exactly 1.0 — correct only for a checkpoint calibrated at 1.0,
            # and additionally the only value the fp8 MLA *context* path is
            # self-consistent at (it quantizes q/k/v at 1.0 while applying
            # s^2/s regardless). So the checkpoint's own scales are checked
            # rather than assumed.
            for role in ("k_scale", "v_scale"):
                s = w[f"l{i}_{role}"]
                assert torch.equal(s, torch.ones_like(s)), (
                    f"layer {i}: {role} is {s.item()}, not 1.0; the fp8 MLA "
                    "context path is only correct at a KV scaling factor of "
                    "1.0, and this assembly passes no scale tensors"
                )
            kvb = w[f"l{i}_kvb"]
            attn.append(
                (
                    w[f"l{i}_qa"].t(),
                    w[f"l{i}_q_norm"],
                    w[f"l{i}_qb"].t(),
                    w[f"l{i}_kva"].t(),
                    w[f"l{i}_kv_norm"],
                    # k_b [H, nope, C] absorbs into q_nope; v_b_t [H, C, v]
                    # expands the latent attention output. Both are views of
                    # the row-regrouped kv_b_proj.
                    kvb[:hn].reshape(self.heads, self.nope, self.kv_lora),
                    transpose(kvb[hn:].reshape(self.heads, self.v_dim, self.kv_lora), 1, 2),
                    kvb.t(),
                    w[f"l{i}_o"].t(),
                    w[f"l{i}_norm2"],
                )
            )
            nxt.append(w[f"l{i + 1}_norm1"] if i + 1 < self.num_layers else w["final_norm"])
            # The fused gate_up GEMM assumes one activation scale and one
            # weight global scale for both halves; the checkpoint stores them
            # per projection, so the equality the fusion rests on is asserted.
            assert torch.equal(w[f"l{i}_mlp_isc1"], w[f"l{i}_mlp_isc1_up"]), (
                f"layer {i}: gate/up input_scale differ; the fused gate_up "
                "GEMM needs one activation scale"
            )
            assert torch.equal(w[f"l{i}_mlp_ws2_1"], w[f"l{i}_mlp_ws2_1_up"]), (
                f"layer {i}: gate/up weight_scale_2 differ; the fused gate_up GEMM needs one alpha"
            )
            # The checkpoint stores reciprocals: `input_scale = amax/(448*6)
            # = 1/g_act` and `weight_scale_2 = 1/g_w`, so the quantizer's
            # global scale is `1/input_scale` and the GEMM's alpha is their
            # product — both straight off disk, no further reciprocal.
            isc1 = w[f"l{i}_mlp_isc1"]
            isc2 = w[f"l{i}_mlp_isc2"]
            mlp.append(
                (
                    w[f"l{i}_mlp_gu_w"],
                    w[f"l{i}_mlp_gu_s"],
                    (isc1 * w[f"l{i}_mlp_ws2_1"]).contiguous(),
                    (1.0 / isc1).contiguous(),
                    w[f"l{i}_mlp_dn_w"],
                    w[f"l{i}_mlp_dn_s"],
                    (isc2 * w[f"l{i}_mlp_ws2_2"]).contiguous(),
                    (1.0 / isc2).contiguous(),
                )
            )
            if i < self.dense_layers:
                moe.append(None)
                continue
            assert torch.equal(w[f"l{i}_e_isc1"], w[f"l{i}_e_isc1_up"]), (
                f"layer {i}: expert gate/up input_scale differ"
            )
            assert torch.equal(w[f"l{i}_e_ws2_1"], w[f"l{i}_e_ws2_1_up"]), (
                f"layer {i}: expert gate/up weight_scale_2 differ"
            )
            # One quantization of the gathered hidden states feeds every
            # expert on every rank, so the routed FC1 activation scale must
            # be a single value — and the same value everywhere, or the four
            # windows would not sum to the whole layer. The shared expert
            # sees every token where each routed expert sees only its own
            # subset, and the checkpoint's shared-expert input_scale is
            # exactly the max over *all* routed ones — the conservative
            # choice that cannot saturate an activation block. Asserted over
            # the full 256, which is why the scalars are loaded whole.
            e_isc1 = w[f"l{i}_e_isc1"]
            assert torch.equal(e_isc1.max(), w[f"l{i}_mlp_isc1"][0]), (
                f"layer {i}: shared-expert input_scale is not the max over "
                "the routed experts; the shared activation quantization "
                "would saturate"
            )
            e_isc2 = w[f"l{i}_e_isc2"][window]
            gate1 = (w[f"l{i}_mlp_isc1"][0] * w[f"l{i}_e_ws2_1"][window]).contiguous()
            moe.append(
                (
                    w[f"l{i}_router"].t(),
                    w[f"l{i}_router_bias"],
                    w[f"l{i}_fc1_w"],
                    view_dtype(w[f"l{i}_fc1_s"], torch.float8_e4m3fn),
                    w[f"l{i}_fc2_w"],
                    view_dtype(w[f"l{i}_fc2_s"], torch.float8_e4m3fn),
                    # output1_scale_scalar, output1_scale_gate_scalar,
                    # output2_scale_scalar — the FC1 alpha, that alpha times
                    # the per-expert FC2 activation global scale, and the FC2
                    # alpha, each [local_num_experts]. Swapping the first two
                    # is finite and silent.
                    (gate1 / e_isc2).contiguous(),
                    gate1,
                    (e_isc2 * w[f"l{i}_e_ws2_2"][window]).contiguous(),
                    # The routed FC1 activation global scale: the same
                    # 1/input_scale the shared-expert GEMM quantizes with,
                    # in the linear scale layout the MoE runner reads.
                    (1.0 / w[f"l{i}_mlp_isc1"]).contiguous(),
                )
            )
        self._attn, self._mlp, self._moe, self._next_norm = attn, mlp, moe, nxt
        if self.mtp_enabled:
            self._mtp = self._derive_mtp()
        # The side stream the shared-expert / dense-MLP branch runs on. Built
        # here because a stream must exist before the first forward: creating
        # one inside a CUDA-graph capture is not a capturable operation, and
        # the runtime's first eager forwards are already past this point.
        self._side_stream = torch.cuda.Stream(device=device)

    def _derive_mtp(self) -> dict:
        """The MTP layer's operand set, derived exactly as a trunk layer's is:
        column-major GEMM views (`.t()` is zero-copy), the two MLA absorption
        operands split out of the row-regrouped kv_b_proj, and the same fp8 KV
        scale check. No NVFP4 scalars — this module is bf16 throughout, so its
        expert stacks go to `fused_moe` as they are stored."""
        w = self.w
        hn = self.heads * self.nope
        for role in ("k_scale", "v_scale"):
            s = w[f"mtp_{role}"]
            assert torch.equal(s, torch.ones_like(s)), (
                f"MTP layer: {role} is {s.item()}, not 1.0; the fp8 MLA "
                "context path is only correct at a KV scaling factor of 1.0, "
                "and this assembly passes no scale tensors"
            )
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
            "k_b": kvb[:hn].reshape(self.heads, self.nope, self.kv_lora),
            "v_b_t": transpose(kvb[hn:].reshape(self.heads, self.v_dim, self.kv_lora), 1, 2),
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

    def _check_step_contract(self, md, position_ids) -> None:
        """First-forward fail-fast: the metadata fields this target consumes
        must exist (private trtllm surface, pinned by version), the paged
        latent pool must be the single pool the MLA entries are certified
        over at the page size their fp8 column covers, the rope table must
        cover every position the engine admits, and every feature this target
        holds inert must actually be off. Everything checked is fixed at
        engine construction — once per model instance is sound."""
        missing = [name for name in _STEP_FIELDS if not hasattr(md, name)]
        assert not missing, f"metadata fields missing: {missing}"
        # position_ids is not consumed: the MLA ops derive each context
        # token's position from its row index within the sequence and each
        # generation token's from sequence_length - 1. Checked anyway so a
        # layout change upstream is loud rather than silent.
        assert position_ids.dtype == torch.int32, position_ids.dtype
        pools = {row[0] for row in md.host_kv_cache_pool_mapping.tolist()}
        assert pools == {0}, (
            f"multi-pool KV addressing is not certified; layer->pool ids {sorted(pools)}"
        )
        # Every MLA entry's fp8-e4m3 column is page 32 only (the bf16 columns
        # also carry 64). 32 is what a default KvCacheConfig produces.
        assert md.tokens_per_block == 32, (
            f"the fp8 latent-pool column of every MLA entry is certified at "
            f"tokens_per_block 32; this engine built {md.tokens_per_block}"
        )
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
        # append is the whole context path. The two cache ops reject any
        # index dtype but int64.
        self._cached_ctx = all(hasattr(md, n) for n in _CACHED_CTX_FIELDS) and bool(
            md.enable_context_mla_with_cached_kv
        )
        if self._cached_ctx:
            for name in ("ctx_cached_token_indptr", "ctx_kv_indptr"):
                t = getattr(md, name)
                assert t.dtype == torch.int64 and t.is_cuda, (name, t.dtype, t.device)
        assert md.effective_beam_width == 1, "beam search is not implemented"
        assert md.cache_indirection is None, "beam search is not implemented"
        assert md.block_ids_per_seq is None, "the FlashMLA layout is not implemented"
        assert md.flash_mla_tile_scheduler_metadata is None, "FlashMLA is not implemented"
        assert md.flash_mla_num_splits is None, "FlashMLA is not implemented"
        assert not md.is_cross, "cross attention is not implemented"
        # The spec-dec **mask** machinery, which is a different thing from
        # speculative decoding being on. Under a linear-tree MTP on a trtllm-gen
        # arch the
        # backend computes `is_spec_decoding_enabled and (not trtllm_gen_arch
        # or is_spec_dec_dynamic_tree)` and gets False, so every
        # `spec_decoding_*` tensor stays None — which is exactly the inert
        # group `_CALL_INERT` holds and every MLA column certifies. Drafting
        # itself reaches the attention ops through `predicted_tokens_per_seq`
        # alone. This assert is the precondition those inert values rest on: it
        # fires on a tree draft, or on a pre-Blackwell arch where the gating
        # does not force the mask off, and either would need the mask surface
        # certified first.
        assert not md.is_spec_decoding_enabled and not md.use_spec_decoding, (
            "the spec-decoding mask surface is live; this target holds the "
            "whole spec_decoding_* group at its inert values, which is only "
            "valid while the Blackwell linear-tree gating keeps it off"
        )
        self._step_contract_checked = True

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
        assert len(counts) == self.dp_size, (counts, self.dp_size)
        assert counts[self.rank] == num_tokens, (
            f"rank {self.rank} holds {num_tokens} rows but the engine's "
            f"all_rank_num_tokens says {counts[self.rank]} ({counts}); the "
            "collectives would gather the wrong split"
        )
        return max(counts)

    def _dense_mlp(self, x, params, dt):
        """One NVFP4 SwiGLU MLP over this rank's own tokens: fused gate_up
        GEMM, silu_and_mul, down GEMM. Replicated weights, so the result is
        complete — nothing to reduce. Both quantizations emit the
        128x4-swizzled scale buffer nvfp4_gemm consumes."""
        gu_w, gu_s, gu_alpha, gu_g, dn_w, dn_s, dn_alpha, dn_g = params
        xq, xsf = fp4_quantize(x, gu_g, _SF_VEC, False, _SF_SWIZZLED)
        gu = nvfp4_gemm(xq, gu_w, xsf, gu_s, gu_alpha, dt)
        act = flashinfer_silu_and_mul(gu)
        aq, asf = fp4_quantize(act, dn_g, _SF_VEC, False, _SF_SWIZZLED)
        return nvfp4_gemm(aq, dn_w, asf, dn_s, dn_alpha, dt)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_w, mlp_w, moe_w = self._attn, self._mlp, self._moe
        next_norm = self._next_norm
        assert attn_w is not None and mlp_w is not None and moe_w is not None, (
            "load_weights must run before forward"
        )
        assert next_norm is not None, "load_weights must run before forward"
        assert position_ids is not None
        assert isinstance(attn_metadata, TrtllmAttentionMetadata)
        # Inputs this target does not implement must fail loudly, not be
        # silently dropped (unlike runtime-owned features, which pass through).
        assert lora_params is None, "LoRA is not implemented by this target"
        # The trunk does not read `spec_metadata` even when the engine drafts:
        # on MTP_EAGLE_ONE_MODEL the runtime sets `layers_to_capture = ()`, so
        # no hidden-state capture hook is owed and the shell keeps the argument
        # to itself. Keeping this assert is a stronger guarantee than deleting
        # it.
        assert kwargs.get("spec_metadata") is None, (
            "spec_metadata reached the trunk; the shell owns the draft loop and must not forward it"
        )
        md = attn_metadata
        if not self._step_contract_checked:
            self._check_step_contract(md, position_ids)
        # Read after the contract check: that is where a table too short for
        # the engine's admitted max_seq_len is regrown.
        rope = self._rope
        assert rope is not None, "load_weights must run before forward"

        step = _build_step_args(md)
        num_ctx = md.num_contexts
        tc = md.num_ctx_tokens
        # The paged-cache address book, shared by the two MLA preprocessing
        # ops and both attention calls. Under attention DP each rank owns a
        # pool holding only its own requests' latent rows.
        tokens_per_block = md.tokens_per_block
        block_offsets = md.kv_cache_block_offsets
        pool_ptrs = md.host_kv_cache_pool_pointers
        pool_map = md.host_kv_cache_pool_mapping
        assert tokens_per_block is not None, "paged KV cache is required"
        assert block_offsets is not None and pool_ptrs is not None, "paged KV cache is required"
        assert pool_map is not None, "paged KV cache is required"

        if inputs_embeds is None:
            assert input_ids is not None
            h = embedding(input_ids, self.w["embed"])
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
        assert gen_p * gen_seqs == gen, (
            f"{gen} generation rows do not divide over {gen_seqs} generation "
            "sequences; a ragged per-sequence draft length cannot be expressed "
            "to the MLA generation call"
        )
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

        attn_out = empty([num_tokens, self.heads * self.v_dim], dt, dev)
        attn_ctx, attn_gen = split(attn_out, [tc, gen], 0)
        x = flashinfer_rmsnorm(h, self.w["l0_norm1"], self.eps)
        residual = h
        for i in range(self.num_layers):
            (
                w_qa,
                w_q_norm,
                w_qb,
                w_kva,
                w_kv_norm,
                k_b,
                v_b_t,
                w_kvb,
                w_o,
                w_n2,
            ) = attn_w[i]
            # The q-LoRA pair: down-project to q_lora_rank, RMS-norm the
            # latent, up-project to the per-head [nope | rope] rows.
            q = cublas_mm(flashinfer_rmsnorm(cublas_mm(x, w_qa), w_q_norm, self.eps), w_qb)
            kva = cublas_mm(x, w_kva)
            ckv_raw, k_pe = split(kva, [self.kv_lora, self.rope], -1)
            ckv = flashinfer_rmsnorm(ckv_raw, w_kv_norm, self.eps)
            latent = concat([ckv, k_pe], -1)
            q_ctx, q_gen = split(q, [tc, gen], 0)
            latent_ctx, latent_gen = split(latent, [tc, gen], 0)

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
                    mla_rope_append_paged_kv_assign_q(
                        q_ctx,
                        latent_ctx,
                        num_ctx,
                        md.ctx_cached_token_indptr,
                        md.ctx_kv_indptr,
                        int(md.max_ctx_seq_len),
                        rope["rotary_cos_sin"],
                        self.heads,
                        self.nope,
                        self.rope,
                        self.kv_lora,
                        block_offsets,
                        pool_ptrs,
                        pool_map,
                        None,
                        _KV_RESIDUAL_DIM,
                        i,
                        tokens_per_block,
                        md.max_seq_len,
                        1,
                        self.quant_mode,
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
                        i,
                        self.kv_lora,
                        self.rope,
                        tokens_per_block,
                        md.max_seq_len,
                        1,
                        self.quant_mode,
                    )
                    latent_arg = None
                else:
                    # Block reuse off: no context sequence can carry a
                    # prefix, so the fresh-prefill flavor does the rope and
                    # the append inside the attention call and the latent
                    # rows never leave registers.
                    assert ctx_kv_tokens == tc, (
                        "a context sequence arrived with a cached KV prefix "
                        "while the cached-KV metadata surface is off"
                    )
                    ckv_full, _ = split(ckv, [tc, gen], 0)
                    k_pe_full = None
                    latent_arg = latent_ctx
                tkv = ctx_kv_tokens
                # kv is packed [all heads' k_nope | all heads' v]: the context
                # FMHA hard-codes V's row stride as the full packed width and
                # reads the column block, so V must stay this split view.
                kv = cublas_mm(ckv_full, w_kvb)
                k_nope, v_view = split(kv, [self.heads * self.nope, self.heads * self.v_dim], -1)
                k = empty([tkv, self.heads, self.qk_dim], dt, dev)
                k_nope_dst, k_pe_dst = split(k, [self.nope, self.rope], -1)
                copy_(k_nope_dst, reshape(k_nope, [tkv, self.heads, self.nope]))
                if k_pe_full is not None:
                    # k_pe came back from the pool already rotated; every
                    # query head shares it. On the fresh-prefill flavor the
                    # rope slice is left uninitialized instead — that call
                    # overwrites it in place from latent_cache.
                    copy_(
                        k_pe_dst,
                        expand(
                            reshape(k_pe_full, [tkv, 1, self.rope]),
                            [tkv, self.heads, self.rope],
                        ),
                    )
                thop_attention(
                    q=q_ctx,
                    k=reshape(k, [tkv, self.heads * self.qk_dim]),
                    v=v_view,
                    output=attn_ctx,
                    latent_cache=latent_arg,
                    q_pe=None,
                    local_layer_idx=i,
                    is_fused_qkv=False,
                    attention_input_type=1,
                    num_heads=self.heads,
                    num_kv_heads=self.heads,
                    head_size=self.qk_dim,
                    q_lora_rank=self.q_lora,
                    kv_lora_rank=self.kv_lora,
                    qk_nope_head_dim=self.nope,
                    qk_rope_head_dim=self.rope,
                    v_head_dim=self.v_dim,
                    cu_q_seqlens=None,
                    cu_kv_seqlens=None,
                    fmha_scheduler_counter=None,
                    # The context call's row count is `num_ctx_tokens`
                    # whatever the generation phase carries, and the entry
                    # certifies the MLA context flavors at 1 only.
                    predicted_tokens_per_seq=1,
                    **step,
                    **rope,
                    **self._call,
                )

            if gen:
                q3 = reshape(q_gen, [gen, self.heads, self.qk_dim])
                q_nope, q_pe = split(q3, [self.nope, self.rope], -1)
                fused_q = empty([gen, self.heads, self.lat_dim], dt, dev)
                fq_nope, _ = split(fused_q, [self.kv_lora, self.rope], -1)
                # Absorbed q: (q_nope @ W_k_nope) is what the latent-space
                # dot product needs. Over an fp8 pool the next call **reads**
                # this half to build the quantized query, so this BMM must
                # have finished first — the two are issued in this order on
                # one stream, which is what makes that safe. (On a bf16 pool
                # they write disjoint halves and may overlap; assembling from
                # that reading and switching to an fp8 cache is a silent race.)
                bmm_out(transpose(q_nope, 0, 1), k_b, transpose(fq_nope, 0, 1))
                cu_q = empty([gen + 1], torch.int32, dev)
                cu_kv = empty([gen + 1], torch.int32, dev)
                counter = empty([1], torch.uint32, dev)
                # The fp8 decode triple: the quantized query the FMHA reads
                # instead of fused_q, and the two folded softmax/output scales
                # it takes instead of either kv scale tensor. All three are
                # written by the call below from q_scaling, the MLA dims and
                # the read-side factor (None = 1.0).
                quant_q = empty([gen, self.heads, self.lat_dim], torch.float8_e4m3fn, dev)
                bmm1_scale = empty([2], torch.float32, dev)
                bmm2_scale = empty([1], torch.float32, dev)
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
                    i,
                    self.heads,
                    1,
                    self.lat_dim,
                    _KV_RESIDUAL_DIM,
                    tokens_per_block,
                    md.max_seq_len,
                    1,
                    self.quant_mode,
                    self.q_scaling,
                    self.q_lora,
                    self.kv_lora,
                    self.nope,
                    self.rope,
                    self.v_dim,
                    True,
                )
                lat_out = empty([gen, self.heads * self.kv_lora], dt, dev)
                thop_attention(
                    q=reshape(fused_q, [gen, self.heads * self.lat_dim]),
                    k=None,
                    v=None,
                    output=lat_out,
                    latent_cache=latent_gen,
                    q_pe=q_pe,
                    local_layer_idx=i,
                    is_fused_qkv=True,
                    attention_input_type=2,
                    num_heads=self.heads,
                    num_kv_heads=1,
                    head_size=self.lat_dim,
                    q_lora_rank=self.q_lora,
                    kv_lora_rank=self.kv_lora,
                    qk_nope_head_dim=self.nope,
                    qk_rope_head_dim=self.rope,
                    v_head_dim=self.kv_lora,
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
                    **step,
                    **rope,
                    **self._call,
                )
                bmm_out(
                    transpose(reshape(lat_out, [gen, self.heads, self.kv_lora]), 0, 1),
                    v_b_t,
                    transpose(reshape(attn_gen, [gen, self.heads, self.v_dim]), 0, 1),
                )

            # Replicated o_proj over this rank's own tokens: complete as it
            # stands, so the residual stream is updated with no collective.
            o = cublas_mm(attn_out, w_o)
            flashinfer_fused_add_rmsnorm(o, residual, w_n2, self.eps)
            if moe_w[i] is None:
                mlp_out = self._dense_mlp(o, mlp_w[i], dt)
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
                assert side is not None, "derive_after_load must run before forward"
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    shared = self._dense_mlp(o, mlp_w[i], dt)
                (
                    w_router,
                    router_bias,
                    fc1_w,
                    fc1_s,
                    fc2_w,
                    fc2_s,
                    o1,
                    o1_gate,
                    o2,
                    routed_g,
                ) = moe_w[i]
                # The expert-parallel round trip. Gathering *before* the
                # router is what keeps the four windows tiling the routing
                # space exactly once: every rank routes the identical full
                # token set, so a token's top-8 ids agree across ranks and
                # each id falls in exactly one window.
                o_pad = o if not pad_rows else pad(o, [0, 0, 0, pad_rows])
                o_all = allgather(o_pad, None, self.dp_group)
                # The expert call is chunked to `_MOE_MAX_T` rows: the
                # gathered set reaches 4 * max_num_tokens and both the runner's
                # and the routing op's certified token columns stop at 8192.
                # Every token is independent through routing, quantization and
                # the expert GEMMs, so the chunks are the whole call, re-joined.
                parts = []
                for chunk in split(o_all, _moe_chunk_sizes(o_all.shape[0]), 0):
                    # Raw bf16 logits: noaux_tc_op applies the sigmoid
                    # itself, and its weight dtype follows the logits, so
                    # bf16 in means the MoE runner's bf16 topk_weights need
                    # no cast (the fp32 correction bias does not change that).
                    logits = cublas_mm(chunk, w_router)
                    topk_w, topk_ids = noaux_tc_op(
                        logits,
                        router_bias,
                        self.n_group,
                        self.topk_group,
                        self.topk,
                        self.routed_scale,
                    )
                    # A second quantization of the same hidden states: the
                    # MoE runner reads the linear scale buffer as
                    # float8_e4m3fn, never the swizzled one the shared GEMM
                    # above consumed.
                    xq, xsf = fp4_quantize(chunk, routed_g, _SF_VEC, False, _SF_LINEAR)
                    parts.append(
                        fp4_block_scale_moe_runner(
                            None,
                            None,
                            xq,
                            view_dtype(xsf, torch.float8_e4m3fn),
                            fc1_w,
                            fc1_s,
                            None,
                            None,
                            None,
                            None,
                            fc2_w,
                            fc2_s,
                            None,
                            o1,
                            o1_gate,
                            o2,
                            self.num_experts,
                            self.topk,
                            None,
                            None,
                            self.moe_inter,
                            self.expert_offset,
                            self.local_experts,
                            None,
                            _ROUTING_METHOD_INERT,
                            True,
                            _ACT_TYPE_SWIGLU,
                            topk_w,
                            topk_ids,
                        )[0]
                    )
                routed_all = parts[0] if len(parts) == 1 else concat(parts, 0)
                # The four windows' partials over the whole token set sum to
                # the layer's routed output; the scatter hands this rank back
                # exactly its own rows. bf16 in: the op sums, and it sums
                # float8 as raw bytes.
                routed_pad = reducescatter(routed_all, None, self.dp_group)
                routed = (
                    routed_pad if not pad_rows else split(routed_pad, [num_tokens, pad_rows], 0)[0]
                )
                # Join: the add is the first reader of `shared` on this stream,
                # and `shared` stays referenced until then, so the branch's
                # allocations cannot be recycled underneath it.
                main.wait_stream(side)
                mlp_out = add(routed, shared)
            flashinfer_fused_add_rmsnorm(mlp_out, residual, next_norm[i], self.eps)
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

    def __init__(self, core: StaircaseCore, logits_processor) -> None:
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
        assert mw is not None, "load_weights must run before the draft loop"
        return self.logits_processor.forward(
            flashinfer_rmsnorm(hidden_states, mw["head_norm"], self.core.eps),
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
        dp_rows = _mtp_dp_rows(all_rank_num_tokens, core.rank, core.dp_size, rows)
        pad_rows = dp_rows - rows
        x_pad = x if not pad_rows else pad(x, [0, 0, 0, pad_rows])
        x_all = allgather(x_pad, None, core.dp_group)
        parts = []
        for chunk in split(x_all, _moe_chunk_sizes(x_all.shape[0]), 0):
            logits = cublas_mm(chunk, mw["router"], None, torch.float32)
            topk_w, topk_ids = noaux_tc_op(
                logits,
                mw["router_bias"],
                core.n_group,
                core.topk_group,
                core.topk,
                core.routed_scale,
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
        routed_all = parts[0] if len(parts) == 1 else concat(parts, 0)
        routed_pad = reducescatter(routed_all, None, core.dp_group)
        return routed_pad if not pad_rows else split(routed_pad, [rows, pad_rows], 0)[0]

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
        mw, rope = core._mtp, core._rope
        assert mw is not None and rope is not None, "load_weights must run before the draft loop"
        assert isinstance(attn_metadata, TrtllmAttentionMetadata)
        assert core._step_contract_checked, (
            "the trunk's first-forward contract check has not run; the shell "
            "calls the core before the worker, so this cannot be reached first"
        )
        md = attn_metadata
        step = _build_step_args(md)
        rows = hidden_states.shape[0]
        assert input_ids.shape[0] == rows, (
            f"the draft step's {input_ids.shape[0]} token ids and "
            f"{rows} hidden-state rows must describe the same tokens"
        )
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
        assert gen_p * gen_seqs == gen, (
            f"{gen} generation rows do not divide over {gen_seqs} generation "
            "sequences; a ragged per-sequence draft length cannot be expressed "
            "to the MLA generation call"
        )
        tokens_per_block = md.tokens_per_block
        block_offsets = md.kv_cache_block_offsets
        pool_ptrs = md.host_kv_cache_pool_pointers
        pool_map = md.host_kv_cache_pool_mapping
        assert tokens_per_block is not None, "paged KV cache is required"
        assert block_offsets is not None and pool_ptrs is not None, "paged KV cache is required"
        assert pool_map is not None, "paged KV cache is required"
        ctx_kv_tokens = int(md.host_total_kv_lens[0]) if tc else 0
        # The engine raises the KV pool's layer count by
        # `num_nextn_predict_layers` under a one-model MTP mode, so this
        # module's attention addresses layer index `num_hidden_layers` in the
        # same single pool the trunk's 61 layers use. Nothing in the target's
        # config stub or manifest declares that.
        layer_idx = core.num_layers

        e = embedding(input_ids, embed_tokens)
        en = flashinfer_rmsnorm(e, mw["enorm"], core.eps)
        hn = flashinfer_rmsnorm(hidden_states, mw["hnorm"], core.eps)
        halves = [en, hn] if _MTP_EMBED_BLOCK_FIRST else [hn, en]
        x = cublas_mm(concat(halves, -1), mw["eh"])

        residual = x
        xn = flashinfer_rmsnorm(x, mw["norm1"], core.eps)
        attn_out = empty([rows, core.heads * core.v_dim], dt, dev)
        attn_ctx, attn_gen = split(attn_out, [tc, gen], 0)
        q = cublas_mm(
            flashinfer_rmsnorm(cublas_mm(xn, mw["qa"]), mw["q_norm"], core.eps),
            mw["qb"],
        )
        kva = cublas_mm(xn, mw["kva"])
        ckv_raw, k_pe = split(kva, [core.kv_lora, core.rope], -1)
        ckv = flashinfer_rmsnorm(ckv_raw, mw["kv_norm"], core.eps)
        latent = concat([ckv, k_pe], -1)
        q_ctx, q_gen = split(q, [tc, gen], 0)
        latent_ctx, latent_gen = split(latent, [tc, gen], 0)

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
                    core.heads,
                    core.nope,
                    core.rope,
                    core.kv_lora,
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
                    core.kv_lora,
                    core.rope,
                    tokens_per_block,
                    md.max_seq_len,
                    1,
                    core.quant_mode,
                )
                latent_arg = None
            else:
                assert ctx_kv_tokens == tc, (
                    "a context sequence arrived with a cached KV prefix while "
                    "the cached-KV metadata surface is off"
                )
                ckv_full, _ = split(ckv, [tc, gen], 0)
                k_pe_full = None
                latent_arg = latent_ctx
            tkv = ctx_kv_tokens
            kv = cublas_mm(ckv_full, mw["kvb"])
            k_nope, v_view = split(kv, [core.heads * core.nope, core.heads * core.v_dim], -1)
            k = empty([tkv, core.heads, core.qk_dim], dt, dev)
            k_nope_dst, k_pe_dst = split(k, [core.nope, core.rope], -1)
            copy_(k_nope_dst, reshape(k_nope, [tkv, core.heads, core.nope]))
            if k_pe_full is not None:
                copy_(
                    k_pe_dst,
                    expand(
                        reshape(k_pe_full, [tkv, 1, core.rope]),
                        [tkv, core.heads, core.rope],
                    ),
                )
            thop_attention(
                q=q_ctx,
                k=reshape(k, [tkv, core.heads * core.qk_dim]),
                v=v_view,
                output=attn_ctx,
                latent_cache=latent_arg,
                q_pe=None,
                local_layer_idx=layer_idx,
                is_fused_qkv=False,
                attention_input_type=1,
                num_heads=core.heads,
                num_kv_heads=core.heads,
                head_size=core.qk_dim,
                q_lora_rank=core.q_lora,
                kv_lora_rank=core.kv_lora,
                qk_nope_head_dim=core.nope,
                qk_rope_head_dim=core.rope,
                v_head_dim=core.v_dim,
                cu_q_seqlens=None,
                cu_kv_seqlens=None,
                fmha_scheduler_counter=None,
                predicted_tokens_per_seq=1,
                **step,
                **rope,
                **core._call,
            )

        if gen:
            q3 = reshape(q_gen, [gen, core.heads, core.qk_dim])
            q_nope, q_pe = split(q3, [core.nope, core.rope], -1)
            fused_q = empty([gen, core.heads, core.lat_dim], dt, dev)
            fq_nope, _ = split(fused_q, [core.kv_lora, core.rope], -1)
            bmm_out(transpose(q_nope, 0, 1), mw["k_b"], transpose(fq_nope, 0, 1))
            cu_q = empty([gen + 1], torch.int32, dev)
            cu_kv = empty([gen + 1], torch.int32, dev)
            counter = empty([1], torch.uint32, dev)
            quant_q = empty([gen, core.heads, core.lat_dim], torch.float8_e4m3fn, dev)
            bmm1_scale = empty([2], torch.float32, dev)
            bmm2_scale = empty([1], torch.float32, dev)
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
                core.heads,
                1,
                core.lat_dim,
                _KV_RESIDUAL_DIM,
                tokens_per_block,
                md.max_seq_len,
                1,
                core.quant_mode,
                core.q_scaling,
                core.q_lora,
                core.kv_lora,
                core.nope,
                core.rope,
                core.v_dim,
                True,
            )
            lat_out = empty([gen, core.heads * core.kv_lora], dt, dev)
            thop_attention(
                q=reshape(fused_q, [gen, core.heads * core.lat_dim]),
                k=None,
                v=None,
                output=lat_out,
                latent_cache=latent_gen,
                q_pe=q_pe,
                local_layer_idx=layer_idx,
                is_fused_qkv=True,
                attention_input_type=2,
                num_heads=core.heads,
                num_kv_heads=1,
                head_size=core.lat_dim,
                q_lora_rank=core.q_lora,
                kv_lora_rank=core.kv_lora,
                qk_nope_head_dim=core.nope,
                qk_rope_head_dim=core.rope,
                v_head_dim=core.kv_lora,
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
                transpose(reshape(lat_out, [gen, core.heads, core.kv_lora]), 0, 1),
                mw["v_b_t"],
                transpose(reshape(attn_gen, [gen, core.heads, core.v_dim]), 0, 1),
            )

        o = cublas_mm(attn_out, mw["o"])
        flashinfer_fused_add_rmsnorm(o, residual, mw["norm2"], core.eps)
        shared = self._shared_mlp(o, mw)
        routed = self._routed_experts(o, mw, all_rank_num_tokens, rows, dt)
        # The layer's output is the residual stream itself, un-normalized:
        # `shared_head` owns the module's norm, and the runtime feeds this
        # tensor straight back in as the next draft step's `h`.
        return add(residual, add(routed, shared))


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

    def __init__(self, core: StaircaseCore, lm_head, logits_processor) -> None:
        self.core = core
        self.mtp_layers = [MTPLayer(core, logits_processor)]
        self.lm_head = lm_head

    @property
    def embed_tokens(self) -> torch.Tensor:
        return self.core.w["embed"]


@register_auto_model("StaircaseDeepseekR10528Nvfp4Sm103Dep4")
class StaircaseDeepseekR10528Nvfp4Sm103Dep4(
    DecoderModelForCausalLM[StaircaseCore, PretrainedConfig]
):
    def __init__(self, model_config: ModelConfig):
        cfg = model_config.pretrained_config
        assert cfg is not None
        super().__init__(
            StaircaseCore(model_config),
            config=model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )
        # The speculative branch, built only when a configs/ variant asked for
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
            mode = self.spec_config.spec_dec_mode
            assert mode.is_mtp_eagle_one_model(), (
                f"this target implements the one-model MTP-Eagle draft loop; "
                f"the engine resolved spec_dec_mode {mode!r}"
            )
            assert self.model.mtp_enabled, "core built without the MTP module"
            assert hasattr(self, "logits_processor"), (
                "the inherited shell no longer exposes logits_processor; the "
                "draft head and the shell's own gather both project through it"
            )
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
            assert spec_metadata is None, (
                "spec_metadata arrived without a spec worker; this engine was "
                "built without a speculative_config"
            )
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
        assert spec_metadata is not None, (
            "the spec worker is built but the engine passed no spec_metadata"
        )
        # `spec_metadata` is not forwarded into the core: the trunk genuinely
        # does not read it — on MTP_EAGLE_ONE_MODEL the runtime sets
        # `layers_to_capture = ()`, so `is_layer_capture()` is False everywhere
        # and no hidden-state capture hook is owed (that is Eagle3's
        # requirement, not this mode's) — and the core's assert that it is
        # absent is a stronger guarantee than deleting the assert would be.
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
            embedding(spec_metadata.gather_ids, hidden),
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
