# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: Qwen3.8-Flash-Next-NVFP4 / sm_103 / tp1 — self-contained modeling code.

The core holds the static structure (weights, derived operands, per-manager pool
views); `PrefillTarget` (any batch with context rows) and `DecodeTarget`
(generation-only) run a step over it, passing every argument to every op at the
call site.

The checkpoint is the composite Qwen4-Exp VLM export. The MTP module is built only
when the engine has a ``speculative_config`` (else ``mtp.*`` stays in
``weights.py``'s non-load set). The vision tower is always built; a text-only
request never reaches it. Every speculation-dependent change is gated on a host
int that is 1 when speculation is off.

Differences from other targets:

* **Hyper-connections.** The inter-block residual is ``4 * 2560 = 10240`` wide.
  There is **no** ``model.language_model.norm.weight``: the stack-final
  ``hyper_connection_mixer`` is the final norm and ``lm_head`` reads its output.
* **QSA sparse attention.** The index plane is maintained on every step so a
  request can cross ``dense_seq_len_threshold`` later without rebuilding its
  prefix; sparse selection replaces the dense core only above it.
* **PLE.** Decoder layer 1 only (``ple_layer_ids: [2]`` is one-based): an n-gram
  side path over a 320,001,536-row fp8 table (47.68 GiB of a 123.5 GiB checkpoint).
* **NVFP4 MoE on every layer** (routed experts only); everything else is bf16.

Integration facts this file depends on:

1. The engine rebuilds the composite config and normalizes its text tower
   (``config_utils.load_pretrained_config``). The shell adopts that text config as
   ``model_config.pretrained_config`` before building the core, because the
   recurrent-state and PLE pool geometry are read from it.
2. ``position_ids`` arrives in the mRoPE layout (``[3, 1, T]``) even for text-only
   batches, because the normalized text config carries ``rope_scaling['type'] ==
   'mrope'``. The main and MTP rope read all three rows; the QSA index plane takes
   coordinate 0 only (see ``_QSA_MROPE_SECTION``).
3. The engine owns the multimodal input processor (resolved off the checkpoint's
   ``architectures[0]``; modeling_v2 has no hook), so it is the built-in
   ``Qwen3VLInputProcessorBase``. This target asserts every ``multimodal_data`` key
   where it is read.

**Host offload.** By default the PLE table lives in host memory the device
addresses (`weights._host_resident_empty`), and the gather reads it unchanged;
``TRTLLM_MODELING_V2_PLE_HOST_OFFLOAD=0`` restores the resident declaration. Output
is identical, so only the device-memory delta sees it. The gather stays
device-driven because row indices are produced on device inside a captured decode
forward, which cannot express a host readback.
"""

import math
import os
from typing import NamedTuple

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import step_contract_enabled
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (  # noqa: E501
    flashinfer_silu_and_mul,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.fused_qk_norm_rope import (
    fused_qk_norm_rope,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (
    fp4_block_scale_moe_runner,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch.attention.backends.fmha.phased import get_spec_decoding_position_offsets
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.speculative import get_spec_worker, should_use_separate_draft_kv_cache
from tensorrt_llm._torch.utils import is_gdn_replay_enabled

from . import weights as _weights
from .ops.activation.fused_sigmoid_mul_inplace import fused_sigmoid_mul_inplace
from .ops.activation.gelu_erf import gelu_erf
from .ops.activation.gelu_tanh import gelu_tanh
from .ops.attention.qsa_expand_block_indices import qsa_expand_block_indices
from .ops.attention.qsa_paged_index_scores import qsa_paged_index_scores
from .ops.attention.qsa_paged_index_store import qsa_paged_index_store
from .ops.attention.qsa_paged_kv_store import qsa_paged_kv_store
from .ops.attention.qsa_paged_sparse_gqa import qsa_paged_sparse_gqa
from .ops.attention.qsa_prefill_compress import qsa_prefill_compress
from .ops.attention.qsa_topk_select import qsa_topk_select
from .ops.attention.vision_rope_2d import vision_rope_2d
from .ops.embedding.ngram_hash import ngram_hash
from .ops.hc.qwen4_exp_gated_residual import qwen4_exp_gated_residual
from .ops.hc.qwen4_exp_post_inject import qwen4_exp_post_inject
from .ops.moe.fp8_block_scale_moe_runner import fp8_block_scale_moe_runner
from .ops.moe.fused_sigmoid_gate_mul_add import fused_sigmoid_gate_mul_add
from .ops.moe.renorm_moe_routing_op import renorm_moe_routing_op
from .ops.norm.flashinfer_gemma_rmsnorm import flashinfer_gemma_rmsnorm
from .ops.norm.layer_norm import layer_norm
from .ops.norm.rms_norm_gated_token_major import rms_norm_gated_token_major
from .ops.ple.gated_ngram_injection import gated_ngram_injection
from .ops.ple.ngram_token_history import ngram_token_history
from .ops.quantization.dequantize_e4m3_per_tensor import dequantize_e4m3_per_tensor
from .ops.quantization.fp8_quantize_1x128 import fp8_quantize_1x128
from .ops.ssm.causal_conv1d_fwd import causal_conv1d_fwd
from .ops.ssm.causal_conv1d_triton_update import causal_conv1d_triton_update
from .ops.ssm.causal_conv1d_update import causal_conv1d_update
from .ops.ssm.chunk_gated_delta_rule import chunk_gated_delta_rule
from .ops.ssm.cu_seqlens_from_seq_lens import cu_seqlens_from_seq_lens
from .ops.ssm.dilated_short_conv_paged import dilated_short_conv_paged
from .ops.ssm.extract_transpose_prefill_slice import extract_transpose_prefill_slice
from .ops.ssm.fused_gdn_gating import fused_gdn_gating
from .ops.ssm.fused_gdn_post_conv import fused_gdn_post_conv
from .ops.ssm.fused_recurrent_gated_delta_rule_update import fused_recurrent_gated_delta_rule_update
from .ops.ssm.fused_sigmoid_gating_delta_rule_update import fused_sigmoid_gating_delta_rule_update
from .ops.ssm.pack_gdn_decode_qkv import pack_gdn_decode_qkv
from .ops.ssm.reset_recurrent_state_rows import reset_recurrent_state_rows
from .ops.vision.patch_embed_conv3d import patch_embed_conv3d
from .ops.vision.pos_embed_interpolate import pos_embed_interpolate

# The GPU architecture this target is certified for; a direct instantiation could
# bypass routing.
_SM = (10, 3)

# QSA backend metadata buffers rebuilt by prepare(). Their absence means the engine
# did not select the QSA metadata and the index plane would never be written.
_QSA_FIELDS = (
    "qsa_block_table",
    "qsa_req_idx_per_token",
    "qsa_logical_positions",
    "qsa_sequence_lengths",
    "qsa_visible_blocks",
    "qsa_topk_indices",
    "qsa_topk_row_starts",
)

# Fields read off `attn_metadata.mamba_metadata`. Absence is silent: a smoke run
# passes with no recurrent state at all.
_MAMBA_FIELDS = (
    "state_indices",
    "has_initial_states",
    "query_start_loc",
    "query_start_loc_long",
)


# Per-call constants of the dense attention call shape: packed-QKV causal GQA, RoPE
# applied outside (fused_qk_norm_rope), bf16 pool. The sparse / skip-softmax group
# stays None/0: QSA calls selection and sparse attention as separate kernels and
# `num_sparse_topk` is 0. `quant_mode` and `predicted_tokens_per_seq` are passed
# beside these.
_CALL_CONSTANTS = dict(
    output_sf=None,
    k=None,
    v=None,
    is_fused_qkv=True,
    update_kv_cache=True,
    attention_input_type=0,
    is_mla_enable=False,
    mask_type=1,
    q_scaling=1.0,
    attention_sinks=None,
    # bf16 pool: no KV scale.
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    # In-kernel RoPE disabled; rope_* values are inert placeholders.
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
    # MLA-only surface, unused here.
    kv_norm_weight=None,
    kv_norm_eps=1e-6,
    skip_correction_threshold=0.0,
)

# ------------------------------------------------------------ vision ------

#: Tower rope base (`vision_config` has no rope field); unrelated to the text 1e7.
_VISION_ROPE_THETA = 10000.0

#: LayerNorm epsilon of `norm1`, `norm2`, `merger.norm`: a literal in the reference.
_VISION_NORM_EPS = 1e-6

#: `mask_type` 0 = padding mask: bidirectional within a batch segment. Causal (1)
#: would make a patch depend on raster order.
_VISION_MASK_PADDING = 0

# Per-call constants of the tower's attention: the encoder-only mode of
# `attention/thop_attention`. Omitting `kv_cache_block_offsets`,
# `host_kv_cache_pool_pointers` and `host_kv_cache_pool_mapping` selects that mode.
# `update_kv_cache=False` is rejected ("KV cache update cannot be disabled now"),
# so it stays True.
_VISION_CALL_CONSTANTS = dict(
    k=None,
    v=None,
    output_sf=None,
    kv_cache_block_offsets=None,
    host_kv_cache_pool_pointers=None,
    host_kv_cache_pool_mapping=None,
    tokens_per_block=None,
    use_paged_context_fmha=False,
    is_fused_qkv=True,
    update_kv_cache=True,
    predicted_tokens_per_seq=1,
    # No pool, so no layer to address.
    local_layer_idx=0,
    max_context_q_len_override=None,
    cache_indirection=None,
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    out_scale=None,
    quant_mode=0,
    beam_width=1,
    mask_type=_VISION_MASK_PADDING,
    q_scaling=1.0,
    # Rotation is applied outside (`ops/attention/vision_rope_2d`); rope_* inert.
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
    attention_input_type=0,
    is_mla_enable=False,
    chunked_prefill_buffer_batch_size=1,
    attention_sinks=None,
    block_ids_per_seq=None,
    latent_cache=None,
    q_pe=None,
    q_lora_rank=None,
    kv_lora_rank=None,
    qk_nope_head_dim=None,
    qk_rope_head_dim=None,
    v_head_dim=None,
    rope_append=None,
    attention_chunk_size=None,
    softmax_stats_tensor=None,
    is_spec_decoding_enabled=False,
    use_spec_decoding=False,
    is_spec_dec_tree=False,
    spec_decoding_generation_lengths=None,
    spec_decoding_position_offsets=None,
    spec_decoding_packed_mask=None,
    spec_decoding_bl_tree_mask_offset=None,
    spec_decoding_bl_tree_mask=None,
    spec_bl_tree_first_sparse_mask_offset_kv=None,
    sparse_kv_indices=None,
    sparse_kv_offsets=None,
    sparse_attn_indices=None,
    sparse_attn_offsets=None,
    sparse_attn_indices_block_size=0,
    mrope_rotary_cos_sin=None,
    mrope_position_deltas=None,
    helix_position_offsets=None,
    helix_is_inactive_rank=None,
    is_cross=False,
    cross_kv=None,
    relative_attention_bias=None,
    relative_attention_max_distance=0,
    kv_norm_weight=None,
    kv_norm_eps=1e-6,
    skip_correction_threshold=0.0,
)

#: Head geometries the encoder-only mode is certified at: MHA, `head_size` 72 or
#: 128, context rows only. `head_size` 72 is context-phase only ("unsupported
#: head_size 72" in generation).
_VISION_CERTIFIED_HEADS = {(16, 16, 72), (16, 16, 128)}

#: Head widths `ops/attention/vision_rope_2d` is certified at.
_VISION_ROPE_HEAD_DIMS = (40, 64, 72, 128)

# NVFP4 block size: one e4m3 scale per 16 contiguous elements along K.
_SF_VEC = _weights.SF_VEC
# The trtllm-gen MoE runner reads activation scales as a *linear* buffer, not the
# 128x4-swizzled one of a dense NVFP4 GEMM. Byte counts match when num_tokens is a
# multiple of 128 (every decode graph batch), so the wrong one passes silently.
_SF_LINEAR = False
# Routing happens outside (renorm_moe_routing_op), so the runner's routing method is
# inert, but it must be 1 (Renormalize): the tier table covers (512 experts, 16
# top-k) only for it, and a tier miss is a logged error with no exception and no
# kernel launch.
_ROUTING_METHOD_RENORM = 1
_ACT_TYPE_SWIGLU = 0

# Skip sentinel the conv ops and paged short conv compare `cache_indices` against
# (the runtime's mamba `PAD_SLOT_ID`). Never a substitution: the scheduler's slot
# vector gives every row a real slot, and `ops/ple/ngram_token_history` reads its
# carry via `index_select`, so a -1 there is a device-side assert that poisons the
# process.
_PAD_SLOT_ID = -1

# Gated-DeltaNet gating form: the only pair `fused_sigmoid_gating_delta_rule_update`
# is certified at; matches `fused_gdn_post_conv`'s default so prefill and decode
# compute the same `g`.
_SOFTPLUS_BETA = 1.0
_SOFTPLUS_THRESHOLD = 20.0

# Gated-DeltaNet Q/K L2-norm epsilon: a literal, not `rms_norm_eps`.
_L2_NORM_EPS = 1e-6

# PLE gate magnitude floor before its signed square root (a model constant).
_PLE_CLAMP_MIN = 1e-6

# --------------------------------------------------------------- host offload
#
# Where the PLE n-gram table lives (module docstring, *Host offload*). No engine
# arg owns this, hence an env var, read once at construction (the table is
# allocated once, during `load_weights`). Default on: 47.68 GiB otherwise comes
# out of the KV pool.
_PLE_HOST_OFFLOAD_ENV = "TRTLLM_MODELING_V2_PLE_HOST_OFFLOAD"

#: Spellings of "off"; anything else, including empty, leaves offload on.
_PLE_HOST_OFFLOAD_OFF = frozenset({"0", "false", "no", "off"})


def _ple_host_offload_enabled() -> bool:
    """Whether the n-gram table is allocated in host memory."""
    return os.environ.get(_PLE_HOST_OFFLOAD_ENV, "1").strip().lower() not in _PLE_HOST_OFFLOAD_OFF


# QSA's fixed constants (backend `constants.py`): one index KV head, three position
# axes, index head exactly twice the rotary width (128 == 2 * 64).
_QSA_INDEX_KV_HEADS = 1
_QSA_POSITION_AXES = 3
_QSA_INDEX_HEAD_TO_ROTARY_RATIO = 2

# The QSA index plane takes coordinate 0 alone, a deliberate departure from the
# checkpoint. `qsa_prefill_compress` (key side, `mrope_section`) and
# `fused_qk_norm_rope` on `index_scratch` (query side, `use_mrope`) must agree:
# both read the three coordinates or neither, else scores mix two position
# encodings. Both-on is certified (`mrope_section` (11, 11, 10)); None is a
# modeling choice. No cost on text; on an image batch the index plane sees a frame
# at one position, changing block selection only above `dense_seq_len_threshold`.
# To flip, set the config's `(s0, s1, s2)` and `_QSA_INDEX_USE_MROPE = True`.
_QSA_MROPE_SECTION = None
_QSA_INDEX_USE_MROPE = False

# ---------------------------------------------------------------- MTP ------

#: The checkpoint's MTP module count (`mtp_num_hidden_layers`), replayed
#: `max_draft_len` times; NOT the draft length.
_MTP_NUM_LAYERS = 1

#: How the `[T, hidden]` embedding branch enters the `[T, hc_count * hidden]` MTP
#: stream; nothing in the checkpoint or config says. True: added into all four
#: streams; False: stream 0 only. A wrong choice yields bit-correct text with every
#: draft rejected; only the acceptance gate can tell.
_MTP_EMBED_BLOCK_ALL_STREAMS = True

#: FP8 block-scale block edge, on both weight-scale axes and the activation row.
_FP8_BLOCK = _weights.FP8_BLOCK

#: False: plain `amax / 448` scale (the runner's recipe). True: power-of-two
#: (UE8M0), a different quantization of the same input.
_MTP_MOE_USE_UE8M0 = False


# ----------------------------------------------- KV cache manager binding ---
#
# The ops are plain functions over raw tensors, and the pool views taken from the
# KV cache manager are `TensorWrapper`s over raw device addresses that do not keep
# memory alive, so a view is only valid for the manager that produced it. `_Bound`
# is everything derived from one manager; the core replaces it as a unit
# (`_bind_cache_manager`) and the hot path pays one identity comparison.
#
# The `check_*` functions hold the shape/dtype contract. They read only
# `.shape` / `.dtype`, so unit tests run them on meta tensors with no engine.
#
# For a later target with pools that are not in `attn_metadata` (mamba/GDN state,
# side caches): the reusable parts are (1) one immutable bound object, replaced
# atomically and keyed on manager identity, (2) pure contract checks, (3) refusing
# a re-bind during CUDA graph capture. Once a second target needs it, the next
# step is a declarative `PoolSpec(name, layers, getter, tail_shape, dtype)` list
# consumed by one shared binder under `modeling_v2/` (not `catalog/`), leaving each
# target only its specs and shape constants. Pure-attention targets that read the
# pool pointers from `attn_metadata` on every call (deepseek_v3) do not need this.


class _PoolGeometry(NamedTuple):
    """The pool shapes this target expects, taken from its config and the metadata."""

    tokens_per_block: int
    heads_kv: int
    head_dim: int
    index_kv_heads: int
    index_head_dim: int
    conv_dim: int
    conv_kernel: int
    num_v_heads: int
    head_v_dim: int
    head_k_dim: int
    stream: int
    ple_state_len: int
    ngram_context_len: int
    dtype: torch.dtype  # the activation dtype


class _Bound(NamedTuple):
    """Everything derived from one KV cache manager."""

    cm: object
    pool_idx: dict  # attention layer -> pool offset (what thop_attention addresses)
    index_cache: dict  # attention layer -> QSA index plane
    position_cache: torch.Tensor  # shared QSA position plane
    kv: dict  # attention layer -> (K, V) views
    conv: dict  # GDN layer -> conv state pool
    ssm: dict  # GDN layer -> recurrent state pool
    ple_state: tuple  # (PLE conv state, PLE n-gram id history)
    # MTP only: the runtime's per-step rollback buffers.
    int_idx: torch.Tensor | None = None
    int_conv: dict | None = None
    int_ssm: dict | None = None


def _require(cm, names, what: str) -> None:
    for name in names:
        assert hasattr(cm, name), (
            f"the KV cache manager {type(cm).__name__} does not expose {name}; this target {what}"
        )


def _check_pool(t, what, lead, tail, dtype=None) -> None:
    """`t` has `lead` leading dims of any size, then exactly `tail`, at `dtype`."""
    assert t is not None, f"{what}: the runtime did not allocate it"
    expected = ["*"] * lead + [str(n) for n in tail]
    assert t.dim() == lead + len(tail) and tuple(t.shape[lead:]) == tuple(tail), (
        f"{what} is {tuple(t.shape)}, expected [{', '.join(expected)}]"
    )
    assert dtype is None or t.dtype == dtype, f"{what} is {t.dtype}, expected {dtype}"


def check_position_plane(g: _PoolGeometry, position) -> None:
    shape = (g.tokens_per_block, _QSA_POSITION_AXES)
    _check_pool(position, "QSA position plane", 1, shape, torch.int32)


def check_attention_pools(g: _PoolGeometry, layer: int, index_cache, pool) -> None:
    """QSA index plane and paged K/V for one attention layer."""
    index_shape = (g.tokens_per_block, g.index_kv_heads, g.index_head_dim)
    _check_pool(index_cache, f"layer {layer}: QSA index plane", 1, index_shape, g.dtype)
    # The sparse path needs adjacent HND K/V role planes: [pages, 2, H, block, D].
    kv_shape = (g.heads_kv, g.tokens_per_block, g.head_dim)
    _check_pool(pool, f"layer {layer}: paged K/V", 2, kv_shape)
    assert pool.shape[1] == 2, f"layer {layer}: paged K/V has {pool.shape[1]} role planes, not 2"


def check_recurrent_pools(g: _PoolGeometry, conv, ssm) -> None:
    _check_pool(conv, "conv state pool", 1, (g.conv_dim, g.conv_kernel - 1), g.dtype)
    # Key dim innermost.
    _check_pool(ssm, "recurrent state pool", 1, (g.num_v_heads, g.head_v_dim, g.head_k_dim))


def check_ple_pools(g: _PoolGeometry, ple_conv, ple_ngram, conv) -> None:
    _check_pool(ple_conv, "PLE conv state", 1, (g.stream, g.ple_state_len), g.dtype)
    # The id-history entry writes int64 into it; an int32 pool raises.
    _check_pool(ple_ngram, "PLE n-gram context pool", 1, (g.ngram_context_len,), torch.int64)
    # The forward addresses all three with one scheduler slot vector.
    assert ple_conv.shape[0] == conv.shape[0] == ple_ngram.shape[0], (
        "the PLE and Gated-DeltaNet pools must share one slot space"
    )


def check_step_buffers(g: _PoolGeometry, int_conv, int_ssm, conv, ssm, int_idx) -> None:
    """MTP verify rollback buffers: `[slots, steps, ...]`, at the pools' dtypes
    (the promotion kernel copies between them without converting)."""
    assert int_conv is not None and int_ssm is not None, (
        "the runtime allocated no per-step conv/recurrent state buffers; a "
        "partially accepted draft cannot be rolled back"
    )
    _check_pool(int_conv, "per-step conv buffer", 2, (g.conv_dim, g.conv_kernel - 1), conv.dtype)
    ssm_shape = (g.num_v_heads, g.head_v_dim, g.head_k_dim)
    _check_pool(int_ssm, "per-step recurrent buffer", 2, ssm_shape, ssm.dtype)
    # `update_mamba_states` reads its source rows from this vector.
    assert int_idx is not None and int_idx.dtype == torch.int32, (
        "the cache manager exposes no int32 intermediate_state_indices"
    )


class Qwen4ExpTp1ModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig, composite_config: PretrainedConfig):
        """`composite_config` is passed in because the text sub-config adopted as
        `model_config.pretrained_config` carries neither the vision half nor the
        media token ids."""
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
        assert not model_config.mapping.enable_attention_dp, "tp1 target"

        # `cfg` is the engine-normalized Qwen4ExpTextConfig (pool geometry is read
        # from it).
        assert getattr(cfg, "model_type", None) == "qwen4_exp_text", (
            f"the core must be built on the normalized text config; got "
            f"model_type {getattr(cfg, 'model_type', None)!r}"
        )

        dt = model_config.torch_dtype
        assert dt == torch.bfloat16, f"bf16 target, engine resolved {dt}"
        assert cfg.torch_dtype == torch.bfloat16, cfg.torch_dtype
        assert not cfg.tie_word_embeddings, "untied lm_head"
        assert not cfg.attention_bias, "no q/k/v/o bias anywhere"
        assert cfg.hidden_act == "silu", "SwiGLU over a silu gate"

        self.num_layers = cfg.num_hidden_layers
        self.hidden = cfg.hidden_size
        self.vocab = cfg.vocab_size
        self.eps = cfg.rms_norm_eps

        # `layer_types` is authoritative; `full_attention_interval` is the runtime
        # helper's fallback, so both must agree.
        layer_types = list(cfg.layer_types)
        assert len(layer_types) == self.num_layers, (len(layer_types), self.num_layers)
        assert set(layer_types) <= {"full_attention", "linear_attention"}, set(layer_types)
        self.is_attn = [t == "full_attention" for t in layer_types]
        interval = cfg.full_attention_interval
        assert self.is_attn == [i % interval == interval - 1 for i in range(self.num_layers)], (
            "layer_types disagrees with full_attention_interval"
        )
        self.attn_layers = [i for i, a in enumerate(self.is_attn) if a]
        self.gdn_layers = [i for i, a in enumerate(self.is_attn) if not a]
        assert self.attn_layers and self.gdn_layers, "this target is the hybrid assembly"

        # The checkpoint picks the speculative mode (`MTP_EAGLE_ONE_MODEL`, one module
        # replayed `max_draft_len` times); the module is built only when the engine
        # asked for speculation.
        spec_config = getattr(model_config, "spec_config", None)
        self.mtp_enabled = spec_config is not None
        declared = getattr(cfg, "mtp_num_hidden_layers", None)
        assert declared in (None, _MTP_NUM_LAYERS), (
            f"this target implements the one-layer MTP module this checkpoint "
            f"ships; the text config declares mtp_num_hidden_layers {declared}"
        )
        # The module shares the trunk's embedding and the shell's lm_head.
        assert not getattr(cfg, "mtp_use_dedicated_embeddings", False), (
            "this checkpoint's MTP module shares the decoder's embedding "
            "(`mtp_use_dedicated_embeddings: false`) and ships no "
            "`mtp.embed_tokens` tensor; a dedicated-embedding module is a "
            "different assembly"
        )
        # The nested `mtp` block, when kept: one `full_attention` layer at the trunk's
        # rope base.
        mtp_block = getattr(cfg, "mtp", None)
        self.mtp_theta = None
        if mtp_block is not None:
            assert isinstance(mtp_block, dict), (
                f"the text config's `mtp` block is {type(mtp_block).__name__}; "
                "this target reads it as the published mapping"
            )
            assert mtp_block.get("num_hidden_layers", _MTP_NUM_LAYERS) == _MTP_NUM_LAYERS, (
                f"mtp.num_hidden_layers is {mtp_block.get('num_hidden_layers')}"
            )
            assert list(mtp_block.get("layer_types", ["full_attention"])) == ["full_attention"], (
                f"the MTP module's layer_types is {mtp_block.get('layer_types')}; this "
                "target implements the full-attention form, which is the only "
                "one whose weights the checkpoint carries"
            )
            # `null` means the trunk's own output.
            assert mtp_block.get("mtp_use_hidden_state_from_layer") is None, (
                f"the MTP module asks for the hidden state of layer "
                f"{mtp_block.get('mtp_use_hidden_state_from_layer')}; this "
                "target hands it the end of the stack, which is what the "
                "runtime passes and all this file can reach"
            )
            if "rope_theta" in mtp_block:
                self.mtp_theta = float(mtp_block["rope_theta"])

        # ---- hyper-connections ------------------------------------------
        self.hc_count = cfg.hc_count
        self.hc_lowrank = cfg.hc_lowrank
        assert self.hc_count >= 2, (
            f"the hyper-connection residual carries hc_count streams; this "
            f"config declares {self.hc_count}, which is not a multi-stream "
            f"residual and is a different assembly"
        )
        self.stream = self.hc_count * self.hidden

        # ---- full attention ---------------------------------------------
        self.heads_q = cfg.num_attention_heads
        self.heads_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        assert self.head_dim in (64, 128, 256), "fused_qk_norm_rope head_dim set"
        assert self.heads_q % self.heads_kv == 0, "thop_attention GQA rule"
        self.q_width = self.heads_q * self.head_dim
        self.kv_width = self.heads_kv * self.head_dim
        self.qkv_width = self.q_width + 2 * self.kv_width
        # The per-head interleaved output gate makes q_proj twice as wide as q. No
        # config flag exists, so it is derived from the weight (asserted in weights.py);
        # the per-head split below would otherwise address the wrong columns.
        assert not hasattr(cfg, "attn_output_gate"), (
            "this config gained an attn_output_gate flag; the gate is derived "
            "from q_proj's width here, and the two must be reconciled"
        )

        # ---- RoPE --------------------------------------------------------
        # Partial, neox half-split pairs, no scaling; `rope_theta` is inside
        # `rope_parameters` (mirrored as `rope_scaling`).
        rope = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None)
        assert isinstance(rope, dict), f"rope parameters must be a dict, got {rope!r}"
        self.theta = float(rope["rope_theta"])
        assert self.theta > 0.0, "rope theta"
        partial = float(rope.get("partial_rotary_factor", cfg.partial_rotary_factor))
        self.rotary_dim = int(self.head_dim * partial)
        assert self.rotary_dim % 2 == 0 and 0 < self.rotary_dim <= self.head_dim, (
            f"rotary_dim {self.rotary_dim} must be even and at most head_dim "
            f"{self.head_dim}; the op enforces only the evenness half"
        )
        # The mRoPE declaration is what makes the engine hand this forward a
        # [3, 1, T] position buffer.
        self.mrope_declared = str(rope.get("type", rope.get("rope_type"))) == "mrope"
        # `fused_qk_norm_rope` implements the interleaved mRoPE layout only (pair `j`
        # reads coordinate 1 when `j % 3 == 1 and j < 3*s1`, 2 when `j % 3 == 2 and
        # j < 3*s2`, else 0); the contiguous variant is a different operator. A text
        # gate cannot see the split.
        assert rope.get("mrope_interleaved") is True, (
            f"this target passes the interleaved mRoPE layout, which is the only "
            f"one attention/fused_qk_norm_rope implements; the config declares "
            f"mrope_interleaved={rope.get('mrope_interleaved')!r}"
        )
        section = list(rope["mrope_section"])
        assert len(section) == _QSA_POSITION_AXES, (
            f"mrope_section must carry one entry per position coordinate; got {section}"
        )
        rotary_pairs = self.rotary_dim // 2
        assert sum(section) == rotary_pairs, (
            f"mrope_section {section} sums to {sum(section)} but rotary_dim "
            f"{self.rotary_dim} makes {rotary_pairs} rotation pairs; the split "
            f"partitions pairs, not dimensions"
        )
        self.mrope_section1 = int(section[1])
        self.mrope_section2 = int(section[2])
        # Row-1 section is load-bearing here; row-2 saturates (10, 11, 12 are
        # bit-identical at 32 pairs). `use_gemma` with `use_mrope` is certified only
        # at the geometry asserted below.
        assert (
            self.heads_q,
            self.heads_kv,
            self.head_dim,
            self.rotary_dim,
            self.theta,
        ) == (24, 2, 256, 64, 1e7), (
            f"the main attention rope runs use_gemma together with use_mrope, "
            f"which attention/fused_qk_norm_rope certifies only at "
            f"(24, 2, 256, 64, 1e7); this config gives "
            f"{(self.heads_q, self.heads_kv, self.head_dim, self.rotary_dim, self.theta)}"
        )
        # No YaRN or other scaling.
        assert "factor" not in rope, (
            f"this target builds a plain (unscaled) rope; rope carries {rope!r}"
        )
        self.max_positions = cfg.max_position_embeddings

        # ---- QSA sparse index --------------------------------------------
        self.index_heads = cfg.indexer_n_heads
        self.index_kv_heads = cfg.indexer_kv_heads
        self.index_head_dim = cfg.indexer_head_dim
        self.token_topk = cfg.indexer_budget
        self.compress_ratio = cfg.indexer_compress_ratio
        assert self.index_kv_heads == _QSA_INDEX_KV_HEADS, (
            f"QSA requires exactly one index KV head; this config declares {self.index_kv_heads}"
        )
        assert self.compress_ratio >= 2 and self.token_topk % self.compress_ratio == 0, (
            f"QSA selection geometry: token_topk {self.token_topk} must divide "
            f"by compress_ratio {self.compress_ratio}, which must be at least 2"
        )
        # Fused index kernels: both powers of two, index head twice the rotary width.
        assert self.index_head_dim == self.rotary_dim * _QSA_INDEX_HEAD_TO_ROTARY_RATIO, (
            f"the fused QSA index kernels require index_head_dim "
            f"({self.index_head_dim}) == 2 * rotary_dim ({self.rotary_dim})"
        )
        assert self.index_head_dim & (self.index_head_dim - 1) == 0, "index_head_dim power of two"
        assert self.compress_ratio & (self.compress_ratio - 1) == 0, "compress_ratio power of two"
        self.block_topk = self.token_topk // self.compress_ratio
        self.expanded_topk = self.token_topk + self.compress_ratio - 1
        # Dense-attention cutoff: QSA resolves max(token_topk, seq_len_threshold);
        # checked against the engine's value below because the CUDA-graph runner
        # splits its short and long decode families on it.
        self.dense_threshold = self.token_topk

        # ---- the engine's sparse configuration, checked at construction -----
        # The index/position side caches, `qsa_*` metadata and sparse kernels all hang
        # off `sparse_attention_config`; checking here fails at cold start rather than
        # at the first forward.
        sparse_cfg = getattr(model_config, "sparse_attention_config", None)
        assert getattr(sparse_cfg, "algorithm", None) == "qsa", (
            f"this target is a QSA sparse-attention assembly and the engine "
            f"resolved sparse_attention_config={sparse_cfg!r}. QSA is this "
            f"architecture's own model default, so an engine that reaches here "
            f"without it dropped the default on the way in rather than being "
            f"asked for dense attention"
        )
        # The engine's own lowering on this core's config: a mismatch with the side
        # cache sizing would otherwise be a silent mis-index.
        qsa_params = sparse_cfg.to_sparse_params(pretrained_config=cfg)
        assert (
            qsa_params.index_n_heads,
            qsa_params.index_kv_heads,
            qsa_params.index_head_dim,
            qsa_params.token_topk,
            qsa_params.compress_ratio,
        ) == (
            self.index_heads,
            self.index_kv_heads,
            self.index_head_dim,
            self.token_topk,
            self.compress_ratio,
        ), (
            f"the engine lowered the QSA index geometry to {qsa_params!r} and "
            f"this target derived heads {self.index_heads}, kv heads "
            f"{self.index_kv_heads}, head_dim {self.index_head_dim}, token_topk "
            f"{self.token_topk}, compress_ratio {self.compress_ratio}; the side "
            f"caches are sized from the engine's copy and the kernels are "
            f"called with this one"
        )
        assert qsa_params.dense_seq_len_threshold == self.dense_threshold, (
            f"the engine's resolved QSA cutoff is "
            f"{qsa_params.dense_seq_len_threshold} and this forward branches at "
            f"{self.dense_threshold}. The runner captures its short decode "
            f"graph family at the engine's number, so a disagreement freezes "
            f"the dense body into the graph that long batches replay — which "
            f"is finite, plausible output. A serving config that raises "
            f"seq_len_threshold above token_topk is the way to get here, and "
            f"it is a configuration this target does not implement"
        )
        self.index_width = (self.index_heads + self.index_kv_heads) * self.index_head_dim
        # Packed [q | k | v] scratch for the fused norm+rope call; the spare KV head
        # is the v slot the op needs but never reads.
        self.index_scratch_width = self.index_width + self.index_head_dim
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

        # ---- gated deltanet ----------------------------------------------
        self.num_k_heads = cfg.linear_num_key_heads
        self.num_v_heads = cfg.linear_num_value_heads
        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        self.conv_kernel = cfg.linear_conv_kernel_dim
        assert self.num_v_heads % self.num_k_heads == 0, "GDN key-head broadcast"
        assert self.head_k_dim == 128 and self.head_v_dim == 128, (
            f"the delta-rule entries certify head dims 128/128; this config "
            f"declares {self.head_k_dim}/{self.head_v_dim}"
        )
        assert self.conv_kernel in (2, 3, 4), "causal_conv1d width domain"
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        # Gate activation of the gated RMS norm (`silu` or `sigmoid`).
        gate_type = getattr(cfg, "output_gate_type", None)
        assert gate_type == "sigmoid", (
            f"gate activation {gate_type!r} is not the sigmoid form this "
            f"target passes to rms_norm_gated_token_major"
        )

        # ---- MoE ----------------------------------------------------------
        self.num_experts = cfg.num_experts
        self.topk = cfg.num_experts_per_tok
        self.moe_inter = cfg.moe_intermediate_size
        self.shared_inter = cfg.shared_expert_intermediate_size
        assert cfg.norm_topk_prob, "renorm_moe_routing_op always renormalizes"
        assert self.num_experts <= 512 and self.topk <= 16, (
            f"renorm_moe_routing_op takes at most 512 experts and top-16; this "
            f"config declares {self.num_experts} / {self.topk}"
        )
        # Every layer is MoE; the dense `intermediate_size` is dead config.
        assert getattr(cfg, "decoder_sparse_step", 1) == 1, "every layer must be MoE"
        assert not getattr(cfg, "mlp_only_layers", []), "every layer must be MoE"
        # trtllm-gen NVFP4 runner operand-width rules and fp4_quantize's block rule.
        assert self.hidden % 256 == 0, "fp4_block_scale_moe_runner hidden alignment"
        assert (2 * self.moe_inter) % 128 == 0, "fp4_block_scale_moe_runner 2I alignment"
        assert (self.hidden // _SF_VEC) % 4 == 0 and (self.moe_inter // _SF_VEC) % 4 == 0, (
            "the 128x4 block-scale swizzle needs both scale widths to be a multiple of 4"
        )
        # flashinfer_silu_and_mul's precondition is on the half width.
        assert (self.shared_inter * 2) % 16 == 0, "flashinfer_silu_and_mul half alignment"

        # ---- PLE -----------------------------------------------------------
        ple_ids = list(getattr(cfg, "ple_layer_ids", None) or [])
        assert len(ple_ids) == 1, (
            f"this target implements the single PLE decoder layer this "
            f"checkpoint ships; the config declares ple_layer_ids={ple_ids}"
        )
        # One-based in the config.
        self.ple_layer = ple_ids[0] - 1
        assert 0 <= self.ple_layer < self.num_layers, self.ple_layer
        assert not self.is_attn[self.ple_layer], (
            f"the PLE lifecycle state rides the recurrent pools, so its layer "
            f"must be a linear-attention layer; layer {self.ple_layer} is not"
        )
        self.ngram_size = cfg.ngram_size
        self.heads_per_ngram = cfg.heads_per_ngram
        self.ngram_heads = self.heads_per_ngram * (self.ngram_size - 1)
        self.ple_embed_dim = cfg.ple_embed_dim
        self.ple_head_dim = self.ple_embed_dim // self.ngram_heads
        assert self.ple_head_dim * self.ngram_heads == self.ple_embed_dim, (
            f"{self.ngram_heads} n-gram heads must tile ple_embed_dim {self.ple_embed_dim}"
        )
        self.ple_conv_kernel = cfg.ple_conv_kernel_size
        # Length of the runtime's paged conv state.
        self.ple_dilation = self.ngram_size
        self.ple_state_len = (self.ple_conv_kernel - 1) * self.ple_dilation
        self.ngram_context_len = self.ngram_size - 1
        # Shard count and padding rule are config; the table height is not, so the
        # table is declared at load time (`weights.py`).
        self.ple_shards = cfg.split_ngram_parts
        self.ngram_vocab_divisor = cfg.make_ngram_vocab_size_divisible_by
        assert self.ple_shards > 0 and self.ngram_vocab_divisor > 0, (
            self.ple_shards,
            self.ngram_vocab_divisor,
        )
        # Fresh-sequence seed for the n-gram id history and the hash's segment token
        # (element 0 when a list, as in the HF reference).
        eos = cfg.eos_token_id
        self.eos_token_id = int(eos[0] if isinstance(eos, (list, tuple)) else eos)

        # ---- the vision tower ------------------------------------------------
        # A 27-block ViT (no KV cache, deepstack or quantization) feeding a 2x2 patch
        # merger. Constants the config omits: `_VISION_ROPE_THETA`,
        # `_VISION_NORM_EPS`, and the merger's erf GELU.
        vcfg = getattr(composite_config, "vision_config", None)
        assert vcfg is not None, (
            "this target implements the composite export's vision tower; the "
            "composite config carries no vision_config"
        )
        assert getattr(vcfg, "model_type", None) == "qwen4_exp_vision", (
            f"vision_config.model_type is {getattr(vcfg, 'model_type', None)!r}; "
            "this target is the Qwen4-Exp vision tower"
        )
        self.v_depth = vcfg.depth
        self.v_hidden = vcfg.hidden_size
        self.v_heads = vcfg.num_heads
        self.v_inter = vcfg.intermediate_size
        self.v_merge = vcfg.spatial_merge_size
        self.v_patch = vcfg.patch_size
        self.v_tpatch = vcfg.temporal_patch_size
        self.v_channels = vcfg.in_channels
        self.v_pos_rows = vcfg.num_position_embeddings
        assert self.v_hidden % self.v_heads == 0, (self.v_hidden, self.v_heads)
        self.v_head_dim = self.v_hidden // self.v_heads
        # vision_rope_2d needs a head divisible by 4 (two axes, one rotation pair
        # each).
        assert self.v_head_dim in _VISION_ROPE_HEAD_DIMS, (
            f"ops/attention/vision_rope_2d is certified at head widths "
            f"{_VISION_ROPE_HEAD_DIMS}; this tower's is {self.v_head_dim}"
        )
        assert (self.v_heads, self.v_heads, self.v_head_dim) in _VISION_CERTIFIED_HEADS, (
            f"the tower's attention is the encoder-only (no KV cache) mode of "
            f"attention/thop_attention, certified at {sorted(_VISION_CERTIFIED_HEADS)}; "
            f"this tower is {(self.v_heads, self.v_heads, self.v_head_dim)}"
        )
        # `hidden_act` names the block MLP's activation only; the merger runs the erf
        # `nn.GELU()`. Applying it to both is wrong by under one bf16 ulp, so there
        # are two entries (`gelu_tanh`, `gelu_erf`).
        assert vcfg.hidden_act == "gelu_pytorch_tanh", (
            f"the vision blocks' MLP activation is {vcfg.hidden_act!r}; this "
            "target calls ops/activation/gelu_tanh for it"
        )
        # No deepstack (`routing.py` refuses it; this is the second lock).
        assert not list(getattr(vcfg, "deepstack_visual_indexes", []) or []), (
            f"deepstack_visual_indexes={list(vcfg.deepstack_visual_indexes)} asks "
            "for per-layer visual feature injection, which this target does not "
            "implement"
        )
        assert vcfg.out_hidden_size == self.hidden, (
            f"the merger projects into the language model's hidden width; "
            f"out_hidden_size {vcfg.out_hidden_size} against hidden {self.hidden}"
        )
        # pos_embed_interpolate asserts a square table; fail before the first image.
        side = int(math.isqrt(self.v_pos_rows))
        assert side * side == self.v_pos_rows, (
            f"num_position_embeddings {self.v_pos_rows} is not a perfect square, "
            "so it is not a square position grid"
        )
        self.v_grid_side = side
        # `spatial_merge_size ** 2` patches become one token at the merger.
        self.v_merge_dim = self.v_hidden * self.v_merge * self.v_merge
        # One flat pixel row per patch, `(in_channels, kt, kh, kw)` channel-slowest.
        self.v_pixel_width = self.v_channels * self.v_tpatch * self.v_patch * self.v_patch
        # Placeholder token ids live on the composite config's top level.
        self.image_token_id = int(composite_config.image_token_id)
        self.video_token_id = int(composite_config.video_token_id)
        assert self.image_token_id < self.vocab and self.video_token_id < self.vocab, (
            f"the media placeholders ({self.image_token_id}, {self.video_token_id}) "
            f"are in-vocabulary against vocab_size {self.vocab}, so the engine "
            "cannot find them with its out-of-vocabulary fallback and this target "
            "must publish them as `multimodal_token_ids`"
        )

        # ---- KV pool dtype --------------------------------------------------
        # An unsupported KV dtype makes the QSA backend silently fall back to dense
        # (`QSA_SPARSE_KV_CACHE_DTYPES`).
        kv_algo = model_config.quant_config.kv_cache_quant_algo
        assert kv_algo is None, (
            f"this target runs the checkpoint's own bf16 KV pool, which is what "
            f"the accuracy reference was measured at and what the QSA sparse "
            f"path requires; the engine resolved kv_cache_quant_algo {kv_algo!r}"
        )
        self.quant_mode = 0
        # The weight table reads the tower's geometry off this.
        self.vision_config = vcfg

        self.mtp_layer_idx = self.num_layers
        if self.mtp_theta is None:
            self.mtp_theta = self.theta
        assert self.mtp_theta > 0.0, self.mtp_theta
        if self.mtp_enabled:
            mode = spec_config.spec_dec_mode
            assert mode.is_mtp_eagle_one_model(), (
                f"this target implements the one-model MTP-Eagle draft loop, "
                f"where one module is replayed; the engine resolved "
                f"spec_dec_mode {mode!r}"
            )
            resolved = getattr(spec_config, "num_nextn_predict_layers", None)
            assert resolved == _MTP_NUM_LAYERS, (
                f"the engine resolved num_nextn_predict_layers={resolved} from "
                f"the checkpoint; this target carries the weights of "
                f"{_MTP_NUM_LAYERS} MTP module"
            )
            # `max_draft_len` becomes `predicted_tokens_per_seq` on every attention
            # call, certified at 1..4, so attention binds.
            max_draft = spec_config.max_draft_len
            assert 1 <= max_draft <= 3, (
                f"max_draft_len={max_draft} needs predicted_tokens_per_seq "
                f"{max_draft + 1}, and attention/thop_attention certifies that "
                f"argument at 1..4. A longer draft needs a kernel-verify "
                f"dispatch, not a bigger number"
            )
            # A linear tree keeps the Blackwell spec-dec mask surface inert.
            assert not getattr(spec_config, "use_dynamic_tree", False), (
                "dynamic-tree drafting is not implemented by this target"
            )
            assert getattr(spec_config, "eagle_choices", None) is None, (
                "static-tree drafting is not implemented by this target"
            )
            assert spec_config.max_total_draft_tokens == max_draft, (
                f"a linear tree mirrors max_total_draft_tokens from "
                f"max_draft_len; the engine resolved "
                f"{spec_config.max_total_draft_tokens} against {max_draft}"
            )

            # The engine picks the replay (default) or legacy verify path at
            # cache-manager construction; this target is closed over the legacy one
            # (`_gdn_verify` + `update_mamba_states`). Checked again against the
            # manager at the first forward.
            assert not is_gdn_replay_enabled(), (
                "the GDN cached-replay verify path is on and this target does "
                "not implement it: its commit kernel "
                "(fused_recurrent_gated_delta_rule_cached_replay_update) has no "
                "catalog entry, and with replay on the runtime allocates no "
                "intermediate SSM buffer for the legacy path to write. Export "
                "TRTLLM_USE_GDN_REPLAY=0 before the ranks start. This is a "
                "requirement of the MTP path, not a tuning knob"
            )
            # The FP8 block-scale runner's own shape rules.
            assert self.hidden % _FP8_BLOCK == 0 and self.moe_inter % _FP8_BLOCK == 0, (
                f"fp8_block_scale_moe_runner needs hidden ({self.hidden}) and "
                f"intermediate ({self.moe_inter}) to be multiples of {_FP8_BLOCK}"
            )
            assert self.topk < self.num_experts and self.topk <= 32, (
                f"the no-groups renormalize routing kernel takes 0 < top_k <= 32 "
                f"and top_k < num_experts; this config declares {self.topk} of "
                f"{self.num_experts}"
            )

        # The dtype the table is declared at in weights.py.
        self.ple_table_dtype = torch.float8_e4m3fn
        # Resolved at construction; read by `weights._declare_ngram_table`.
        self.ple_host_offload = _ple_host_offload_enabled()
        # CPU allocation behind the offloaded table. The `ple_table` parameter is a
        # device-address view that owns none of those pages; dropping this reference
        # frees memory a kernel still points into. None when resident.
        self._ple_table_host = None

        # Every weight but the n-gram table; meta-init intercepts torch.empty.
        self.w = _weights.MODEL_WEIGHTS.declare(_weights.MODEL_WEIGHTS.dims(self))

        # Derived once after load (`post_load_weights`).
        self._moe_scalars: list | None = None
        self._gdn_gating: dict | None = None
        self._cos_sin: torch.Tensor | None = None
        # MTP module operands; None with speculation off.
        self._mtp: _MtpWeights | None = None
        self._prefill: "PrefillTarget | None" = None
        self._decode: "DecodeTarget | None" = None
        self.vision_tower: VisionTower | None = None
        # Everything derived from the KV cache manager (`_bind_cache_manager`);
        # None until the first forward.
        self._bound: _Bound | None = None
        self._binds = 0

    # ---------------------------------------------------------------- load ---

    def _norm_slices(self, key: str) -> tuple:
        """Split a grouped-norm weight into its per-group slices (taken once)."""
        return tuple(torch.split(self.w[key], self.hidden, dim=0))

    def post_load_weights(self) -> None:
        """Derive what the targets read, then build them, now that the weights are real.

        Not an override (DecoderModel has no such hook); named for the shell hook."""
        w = self.w
        device = w["embed"].device
        moe_scalars = []
        for i in range(self.num_layers):
            # The checkpoint stores modelopt reciprocals (`input_scale = 1 / g_act`,
            # `weight_scale_2 = 1 / g_w`), so the quantizer's global scale is
            # `1 / input_scale` and the runner's scalars are products of the two.
            isc1, isc1_up = w[f"l{i}_e_isc1"], w[f"l{i}_e_isc1_up"]
            assert torch.equal(isc1, isc1_up), (
                f"layer {i}: expert gate/up input_scale differ; the fused FC1 "
                "reads one activation quantization for both halves"
            )
            assert torch.equal(w[f"l{i}_e_ws2_1"], w[f"l{i}_e_ws2_1_up"]), (
                f"layer {i}: expert gate/up weight_scale_2 differ; the fused "
                "FC1 carries one alpha per expert for both halves"
            )
            # One activation quantization feeds all 512 experts, so FC1's activation
            # global scale must be one number (the shared expert is bf16, no scale).
            assert torch.equal(isc1.min(), isc1.max()), (
                f"layer {i}: the routed experts do not share one "
                "gate_proj.input_scale, so a single activation quantization "
                "cannot feed them all and this assembly is wrong for this "
                "checkpoint"
            )
            e_isc2 = w[f"l{i}_e_isc2"]
            gate1 = (isc1 * w[f"l{i}_e_ws2_1"]).contiguous()
            moe_scalars.append(
                dict(
                    # [num_experts] each. Swapping the first two is finite and silent.
                    output1_scale_scalar=(gate1 / e_isc2).contiguous(),
                    output1_scale_gate_scalar=gate1,
                    output2_scale_scalar=(e_isc2 * w[f"l{i}_e_ws2_2"]).contiguous(),
                    # Routed FC1 activation global scale (one per layer).
                    global_scale=(1.0 / isc1[:1]).contiguous(),
                )
            )
        self._moe_scalars = moe_scalars
        # Both delta-rule entries index A_log and dt_bias as `ptr + head` and certify
        # fp32; the checkpoint is bf16.
        self._gdn_gating = {
            i: (
                w[f"l{i}_A_log"].float().contiguous(),
                w[f"l{i}_dt_bias"].float().contiguous(),
            )
            for i in self.gdn_layers
        }
        # Reconcile the table height against the checkpoint's id-space tensors: an
        # index past the end is a device-side assert that poisons the process.
        table_rows = w["ple_table"].shape[0]
        used_rows = int(w["ple_offs"][-1]) + int(w["ple_vocab"][-1])
        assert used_rows <= table_rows, (
            f"the n-gram table has {table_rows} rows but its per-head offsets "
            f"and vocabulary sizes address {used_rows}"
        )
        assert table_rows % self.ngram_vocab_divisor == 0, (
            f"the n-gram table's {table_rows} rows are not a multiple of "
            f"make_ngram_vocab_size_divisible_by={self.ngram_vocab_divisor}"
        )
        assert table_rows - used_rows < self.ngram_vocab_divisor, (
            f"the n-gram table carries {table_rows - used_rows} padding rows, "
            f"more than one divisor of {self.ngram_vocab_divisor}: the shards "
            f"do not concatenate to the padded id space this checkpoint "
            f"declares"
        )
        # The index branch's rope table in the QSA kernels' layout:
        # `[max_positions, 2 * rotary_pairs]` fp32, cos block then sin block.
        pairs = self.rotary_dim // 2
        exponent = torch.arange(0, self.rotary_dim, 2, dtype=torch.float64, device=device)
        inv_freq = (1.0 / (self.theta ** (exponent / self.rotary_dim))).to(torch.float32)
        positions = torch.arange(self.max_positions, dtype=torch.float32, device=device)
        angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
        self._cos_sin = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1).contiguous()
        assert self._cos_sin.shape == (self.max_positions, 2 * pairs), self._cos_sin.shape
        self._report_ple_residency()
        if self.mtp_enabled:
            self._mtp = _mtp_views(self)

        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)
        self.vision_tower = VisionTower(self)

    def _report_ple_residency(self) -> None:
        """Check where the n-gram table ended up, and report it once.

        Offload is output-identical, so a stray `.to()` or a driver that declined
        the mapping would pass every smoke test; the pointer's memory kind read
        back from the driver is the signal that cannot be faked. The report is
        printed here, before the KV pool is sized from what is left."""
        table = self.w["ple_table"]
        kind = _weights.memory_kind(table.data_ptr())
        expected = (
            _weights.CUDA_MEMORY_TYPE_HOST
            if self.ple_host_offload
            else _weights.CUDA_MEMORY_TYPE_DEVICE
        )
        assert kind == expected, (
            f"the n-gram table was declared with {_PLE_HOST_OFFLOAD_ENV}="
            f"{'1' if self.ple_host_offload else '0'}, so its pages must be "
            f"{'host' if self.ple_host_offload else 'device'} memory, but the "
            f"driver reports cudaMemoryType {kind} for the pointer the forward "
            f"gathers from. The offload is output-identical, so nothing "
            f"downstream of this would notice"
        )
        mib = 1024.0 * 1024.0
        table_mib = table.numel() * table.element_size() / mib
        free, total = torch.cuda.mem_get_info()
        print(
            f"ple n-gram table: {tuple(table.shape)} {table.dtype}, "
            f"{table_mib:.1f} MiB, "
            f"{'HOST-resident (device-addressable)' if self.ple_host_offload else 'DEVICE-resident'}"
            f" [{_PLE_HOST_OFFLOAD_ENV}="
            f"{os.environ.get(_PLE_HOST_OFFLOAD_ENV, '1 (default)')}]; "
            f"after load: allocated {torch.cuda.memory_allocated() / mib:.1f} MiB, "
            f"reserved {torch.cuda.memory_reserved() / mib:.1f} MiB, "
            f"driver-used {(total - free) / mib:.1f} MiB of {total / mib:.1f} MiB",
            flush=True,
        )

    # ------------------------------------------------------------ fail-fast ---

    def _pool_geometry(self, tokens_per_block: int) -> _PoolGeometry:
        return _PoolGeometry(
            tokens_per_block=tokens_per_block,
            heads_kv=self.heads_kv,
            head_dim=self.head_dim,
            index_kv_heads=self.index_kv_heads,
            index_head_dim=self.index_head_dim,
            conv_dim=self.conv_dim,
            conv_kernel=self.conv_kernel,
            num_v_heads=self.num_v_heads,
            head_v_dim=self.head_v_dim,
            head_k_dim=self.head_k_dim,
            stream=self.stream,
            ple_state_len=self.ple_state_len,
            ngram_context_len=self.ngram_context_len,
            dtype=self.w["embed"].dtype,
        )

    def _bind_cache_manager(self, md, position_ids) -> None:
        """Bind this module to one KV cache manager: fail-fast on the contract, then
        take every pool view from it.

        Called once per manager, not per model: with capacity estimation on, the
        engine builds a throwaway manager, warms up, shuts it down and builds the
        real one on the same module, and a view held across the swap dangles
        silently. The result is built in locals and installed as one `_Bound`, so a
        failed bind leaves the previous binding untouched. The contract checks
        re-run on every bind."""
        # Checked first: some fields below are properties that assert on a
        # manager-less metadata instead of raising AttributeError.
        cm = md.kv_cache_manager
        assert cm is not None, "a paged KV cache is required"
        # A re-bind during capture would bake two managers' pointers into one graph;
        # the engine releases graphs before its own swap, so this is an ordering
        # violation.
        assert not (self._binds and torch.cuda.is_current_stream_capturing()), (
            f"the KV cache manager changed to {type(cm).__name__} during CUDA graph capture"
        )
        self._binds += 1

        missing = [name for name in _QSA_FIELDS if not hasattr(md, name)]
        assert not missing, (
            f"metadata fields missing: {missing}; the engine did not select QSA "
            "metadata for this model"
        )

        # `position_ids` is the int32 [3, 1, T] buffer the engine builds on an mRoPE
        # config even for text-only batches; row 0 is the text position.
        assert position_ids.dtype == torch.int32, position_ids.dtype
        assert self.mrope_declared, "this target takes the mRoPE position layout"
        assert position_ids.dim() == 3 and position_ids.shape[0] == _QSA_POSITION_AXES, (
            f"expected the [{_QSA_POSITION_AXES}, 1, T] mRoPE position buffer, got "
            f"{tuple(position_ids.shape)}"
        )

        # thop_attention addresses the pool by the layer's pool offset, not its
        # ordinal among attention layers; counting them aliases layers with finite,
        # plausible output.
        pool_idx = {i: cm.layer_offsets[i] for i in self.attn_layers}
        pools = {row[0] for row in md.host_kv_cache_pool_mapping.tolist()}
        assert pools <= {0, 1}, (
            f"KV addressing over {len(pools)} pools is not certified "
            f"(thop_attention's cells cover one and two); layer->pool ids {sorted(pools)}"
        )

        # No sliding window: both targets pass `max_seq_len` as the window. The vector
        # is indexed by pool offset and recurrent layers carry a large negative
        # sentinel, so only attention offsets are read.
        windows = cm.max_attention_window_vec
        attn_windows = {windows[off] for off in pool_idx.values()}
        assert attn_windows <= {None, md.max_seq_len}, (
            f"the attention layers' pool windows are {sorted(map(repr, attn_windows))} "
            f"but this target passes max_seq_len {md.max_seq_len} as the window; "
            "a sliding window is not implemented here"
        )

        g = self._pool_geometry(md.tokens_per_block)
        _require(
            cm,
            ("get_index_k_buffer", "get_qsa_position_buffer", "get_buffers"),
            "reads the QSA index plane, the shared position plane and the paged K/V",
        )
        # `qsa_topk_select` checks only that the buffer is at least `index_topk` wide,
        # then addresses it as flat `index_topk`-wide rows ignoring the real stride, so
        # a width mismatch silently mis-writes every row but the first.
        assert md.qsa_topk_indices.shape[1] == self.block_topk, (
            f"the engine sized the QSA selection buffer {md.qsa_topk_indices.shape[1]} "
            f"wide and this target selects {self.block_topk} blocks per row"
        )
        position_cache = cm.get_qsa_position_buffer()
        check_position_plane(g, position_cache)
        index_cache, kv = {}, {}
        for i in self.attn_layers:
            index_cache[i] = cm.get_index_k_buffer(i)
            pool = cm.get_buffers(i, kv_layout="HND")
            check_attention_pools(g, i, index_cache[i], pool)
            kv[i] = (pool[:, 0], pool[:, 1])

        # A missed hybrid predicate would still build and pass a smoke run with 36
        # layers carrying no state.
        mm = md.mamba_metadata
        assert mm is not None and mm is not False, (
            "attn_metadata.mamba_metadata is absent: the runtime allocated no "
            f"recurrent-state pool, so the {len(self.gdn_layers)} linear-attention "
            "layers would run with no state"
        )
        mm_missing = [name for name in _MAMBA_FIELDS if not hasattr(mm, name)]
        assert not mm_missing, f"mamba metadata fields missing: {mm_missing}"
        # `ple_layer_cache` exists only on the V2 hybrid manager; under V1 the PLE
        # state silently does not exist.
        _require(
            cm,
            ("get_conv_states", "get_ssm_states", "ple_layer_cache"),
            "reads the recurrent, conv and PLE pools",
        )
        conv = {i: cm.get_conv_states(i) for i in self.gdn_layers}
        ssm = {i: cm.get_ssm_states(i) for i in self.gdn_layers}
        conv0, ssm0 = conv[self.gdn_layers[0]], ssm[self.gdn_layers[0]]
        check_recurrent_pools(g, conv0, ssm0)

        ple_cache = cm.ple_layer_cache(self.ple_layer)
        assert ple_cache is not None, (
            f"no PLE lifecycle state for decoder layer {self.ple_layer}: the runtime "
            "allocated neither the dilated conv state nor the n-gram id history"
        )
        ple_conv, ple_ngram = ple_cache
        check_ple_pools(g, ple_conv, ple_ngram, conv0)

        assert md.effective_beam_width == 1, "beam search is not implemented"
        assert md.cache_indirection is None, "beam search is not implemented"
        assert not md.is_cross, "cross attention is not implemented"
        # With speculation on, the Blackwell trtllm-gen spec-dec mask machinery is off
        # for a linear tree at sm_103: every `spec_decoding_*` tensor stays None and
        # `predicted_tokens_per_seq` alone produces the within-block mask.
        assert not md.is_spec_decoding_enabled, (
            "the attention metadata reports is_spec_decoding_enabled; on sm_103 a "
            "linear-tree MTP forces that machinery off"
        )
        bound = _Bound(
            cm, pool_idx, index_cache, position_cache, kv, conv, ssm, (ple_conv, ple_ngram)
        )
        if self.mtp_enabled:
            bound = bound._replace(**self._bind_step_buffers(cm, g, conv0, ssm0))
        self._bound = bound

    def _bind_step_buffers(self, cm, g: _PoolGeometry, conv0, ssm0) -> dict:
        """Per-manager fail-fast for the MTP increment: the verify step needs
        runtime-owned per-step buffers that exist only under the legacy GDN path.
        Returns the `_Bound` fields it adds."""
        assert cm.is_speculative(), (
            f"the KV cache manager {type(cm).__name__} reports is_speculative() False, "
            "so it allocated none of the per-step state buffers the verify path writes"
        )
        assert not getattr(cm, "use_replay_state_update", False), (
            "the GDN cached-replay verify path is on. Export TRTLLM_USE_GDN_REPLAY=0 "
            "before the ranks start; see this target's __init__ for why"
        )
        _require(
            cm,
            ("get_intermediate_conv_states", "get_intermediate_ssm_states"),
            "reads both per-step state buffers",
        )
        # `get_intermediate_*` builds a fresh view per call, so bind them here.
        int_conv = {i: cm.get_intermediate_conv_states(i) for i in self.gdn_layers}
        int_ssm = {i: cm.get_intermediate_ssm_states(i) for i in self.gdn_layers}
        int_idx = getattr(cm, "intermediate_state_indices", None)
        g0 = self.gdn_layers[0]
        check_step_buffers(g, int_conv[g0], int_ssm[g0], conv0, ssm0, int_idx)
        return dict(int_idx=int_idx, int_conv=int_conv, int_ssm=int_ssm)

    # -------------------------------------------------------------- forward ---

    def _select_target(self, attn_metadata) -> "PrefillTarget | DecodeTarget":
        """Which target runs this step: decode if `num_contexts == 0`, else prefill.

        A draft loop rewrites `num_contexts` in place between draft steps, so the
        draft step routes through here per call."""
        return self._decode if attn_metadata.num_contexts == 0 else self._prefill

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ):
        """Check what holds for every step, bind the cache manager, then route."""
        assert self._prefill is not None and self._decode is not None, (
            "load_weights must run before forward"
        )
        assert position_ids is not None
        assert isinstance(attn_metadata, TrtllmAttentionMetadata)
        # Unimplemented inputs must fail loudly, not be dropped.
        assert lora_params is None, "LoRA is not implemented by this target"
        # `input_ids` is required even with `inputs_embeds` (upstream passes
        # `input_ids=None` after splicing): the PLE block hashes token ids every step,
        # and text gates would not catch a mis-hash.
        assert input_ids is not None, (
            "this target's PLE layer hashes token ids, so input_ids is required "
            "on every step, including one whose embeddings were spliced upstream"
        )
        # `spec_metadata` is present exactly when drafting. On MTP_EAGLE_ONE_MODEL it
        # sets `layers_to_capture = ()`, so the trunk owes no capture hook.
        spec_metadata = kwargs.get("spec_metadata")
        assert (spec_metadata is not None) == self.mtp_enabled, (
            f"spec_metadata {'arrived' if spec_metadata is not None else 'is absent'} "
            f"but the model was built with mtp_enabled={self.mtp_enabled}; the "
            "draft path and the weights it loads are decided at construction"
        )
        # Keyed on the manager, not a first-forward flag: capacity estimation swaps
        # the manager under the same module (see `_bind_cache_manager`).
        if self._bound is None or attn_metadata.kv_cache_manager is not self._bound.cm:
            self._bind_cache_manager(attn_metadata, position_ids)
        return self._select_target(attn_metadata).forward(
            attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


class _LayerWeights(NamedTuple):
    """One trunk layer's weights as the views the ops read (GEMM operands are `.t()`
    views). Attention fields are None on a Gated-DeltaNet layer and vice versa."""

    ahc_norm: tuple
    ahc_down: torch.Tensor
    ahc_up: torch.Tensor
    ahc_inject: torch.Tensor
    mhc_norm: tuple
    mhc_down: torch.Tensor
    mhc_up: torch.Tensor
    mhc_inject: torch.Tensor
    q: torch.Tensor | None
    k: torch.Tensor | None
    v: torch.Tensor | None
    o: torch.Tensor | None
    q_norm: torch.Tensor | None
    k_norm: torch.Tensor | None
    idx: torch.Tensor | None
    idx_qn: torch.Tensor | None
    idx_kn: torch.Tensor | None
    in_qkv: torch.Tensor | None
    in_z: torch.Tensor | None
    in_a: torch.Tensor | None
    in_b: torch.Tensor | None
    out: torch.Tensor | None
    conv_w: torch.Tensor | None
    a_log: torch.Tensor | None
    dt_bias: torch.Tensor | None
    gnorm: torch.Tensor | None
    router: torch.Tensor
    moe_gscale: torch.Tensor
    fc1_w: torch.Tensor
    fc1_s: torch.Tensor
    fc2_w: torch.Tensor
    fc2_s: torch.Tensor
    out1_scale: torch.Tensor
    out1_gate_scale: torch.Tensor
    out2_scale: torch.Tensor
    sh_gu: torch.Tensor
    sh_dn: torch.Tensor
    sh_gate: torch.Tensor


def _layer_views(core: Qwen4ExpTp1ModelingV2) -> list[_LayerWeights]:
    """Every trunk layer's `_LayerWeights`, read off the realized weights."""
    w = core.w
    layers = []
    for i in range(core.num_layers):
        attn: dict = dict(
            q=None,
            k=None,
            v=None,
            o=None,
            q_norm=None,
            k_norm=None,
            idx=None,
            idx_qn=None,
            idx_kn=None,
        )
        gdn: dict = dict(
            in_qkv=None,
            in_z=None,
            in_a=None,
            in_b=None,
            out=None,
            conv_w=None,
            a_log=None,
            dt_bias=None,
            gnorm=None,
        )
        if core.is_attn[i]:
            attn = dict(
                q=w[f"l{i}_q_w"].t(),
                k=w[f"l{i}_k_w"].t(),
                v=w[f"l{i}_v_w"].t(),
                o=w[f"l{i}_o_w"].t(),
                q_norm=w[f"l{i}_q_norm"],
                k_norm=w[f"l{i}_k_norm"],
                idx=w[f"l{i}_idx_w"].t(),
                idx_qn=w[f"l{i}_idx_qn"],
                idx_kn=w[f"l{i}_idx_kn"],
            )
        else:
            a_log, dt_bias = core._gdn_gating[i]
            gdn = dict(
                in_qkv=w[f"l{i}_in_qkv_w"].t(),
                in_z=w[f"l{i}_in_z_w"].t(),
                in_a=w[f"l{i}_in_a_w"].t(),
                in_b=w[f"l{i}_in_b_w"].t(),
                out=w[f"l{i}_out_w"].t(),
                # [conv_dim, 1, width] -> [conv_dim, width]: a view (unit width stride).
                conv_w=w[f"l{i}_conv_w"].reshape(core.conv_dim, core.conv_kernel),
                a_log=a_log,
                dt_bias=dt_bias,
                gnorm=w[f"l{i}_gnorm"],
            )
        scalars = core._moe_scalars[i]
        layers.append(
            _LayerWeights(
                ahc_norm=core._norm_slices(f"l{i}_ahc_norm"),
                ahc_down=w[f"l{i}_ahc_down"],
                ahc_up=w[f"l{i}_ahc_up"],
                ahc_inject=w[f"l{i}_ahc_inject"],
                mhc_norm=core._norm_slices(f"l{i}_mhc_norm"),
                mhc_down=w[f"l{i}_mhc_down"],
                mhc_up=w[f"l{i}_mhc_up"],
                mhc_inject=w[f"l{i}_mhc_inject"],
                **attn,
                **gdn,
                router=w[f"l{i}_router"].t(),
                moe_gscale=scalars["global_scale"],
                fc1_w=w[f"l{i}_fc1_w"],
                fc1_s=w[f"l{i}_fc1_s"].view(torch.float8_e4m3fn),
                fc2_w=w[f"l{i}_fc2_w"],
                fc2_s=w[f"l{i}_fc2_s"].view(torch.float8_e4m3fn),
                out1_scale=scalars["output1_scale_scalar"],
                out1_gate_scale=scalars["output1_scale_gate_scalar"],
                out2_scale=scalars["output2_scale_scalar"],
                sh_gu=w[f"l{i}_sh_gu"].t(),
                sh_dn=w[f"l{i}_sh_dn"].t(),
                sh_gate=w[f"l{i}_sh_gate"].t(),
            )
        )
    return layers


class _StackWeights(NamedTuple):
    """The weights outside the layer loop: the stack-final mixer (this checkpoint's
    final norm) and the PLE side path."""

    fhc_norm: tuple
    fhc_down: torch.Tensor
    fhc_up: torch.Tensor
    ple_key: torch.Tensor
    ple_val: torch.Tensor
    ple_nk: tuple
    ple_nq: tuple
    ple_nc: tuple
    ple_conv_w: torch.Tensor
    ple_mult: torch.Tensor
    ple_vocab: torch.Tensor
    ple_offs: torch.Tensor
    ple_tscale: torch.Tensor


def _stack_views(core: Qwen4ExpTp1ModelingV2) -> _StackWeights:
    w = core.w
    return _StackWeights(
        fhc_norm=core._norm_slices("fhc_norm"),
        fhc_down=w["fhc_down"],
        fhc_up=w["fhc_up"],
        ple_key=w["ple_key_w"].t(),
        ple_val=w["ple_val_w"].t(),
        ple_nk=core._norm_slices("ple_nk"),
        ple_nq=core._norm_slices("ple_nq"),
        ple_nc=core._norm_slices("ple_nc"),
        ple_conv_w=w["ple_conv_w"],
        ple_mult=w["ple_mult"],
        ple_vocab=w["ple_vocab"],
        ple_offs=w["ple_offs"],
        # `dequantize_e4m3_per_tensor` needs the scale at the input's rank (3), all 1s.
        ple_tscale=w["ple_tscale"].reshape(1, 1, 1),
    )


class _MtpWeights(NamedTuple):
    """The MTP module's weights (keyed at its layer index, `num_hidden_layers`)."""

    pre_e_norm: torch.Tensor
    pre_h_norm: tuple
    fc_e: torch.Tensor
    fc_h: torch.Tensor
    ahc_norm: tuple
    ahc_down: torch.Tensor
    ahc_up: torch.Tensor
    ahc_inject: torch.Tensor
    mhc_norm: tuple
    mhc_down: torch.Tensor
    mhc_up: torch.Tensor
    mhc_inject: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    o: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    router: torch.Tensor
    fc1_w: torch.Tensor
    fc1_s: torch.Tensor
    fc2_w: torch.Tensor
    fc2_s: torch.Tensor
    sh_gu: torch.Tensor
    sh_dn: torch.Tensor
    sh_gate: torch.Tensor
    fhc_norm: tuple
    fhc_down: torch.Tensor
    fhc_up: torch.Tensor


def _mtp_views(core: Qwen4ExpTp1ModelingV2) -> _MtpWeights:
    w = core.w
    p = f"l{core.mtp_layer_idx}_mtp_"
    return _MtpWeights(
        pre_e_norm=w[f"{p}pre_e_norm"],
        pre_h_norm=core._norm_slices(f"{p}pre_h_norm"),
        fc_e=w[f"{p}fc_e"].t(),
        fc_h=w[f"{p}fc_h"].t(),
        ahc_norm=core._norm_slices(f"{p}ahc_norm"),
        ahc_down=w[f"{p}ahc_down"],
        ahc_up=w[f"{p}ahc_up"],
        ahc_inject=w[f"{p}ahc_inject"],
        mhc_norm=core._norm_slices(f"{p}mhc_norm"),
        mhc_down=w[f"{p}mhc_down"],
        mhc_up=w[f"{p}mhc_up"],
        mhc_inject=w[f"{p}mhc_inject"],
        q=w[f"{p}q_w"].t(),
        k=w[f"{p}k_w"].t(),
        v=w[f"{p}v_w"].t(),
        o=w[f"{p}o_w"].t(),
        q_norm=w[f"{p}q_norm"],
        k_norm=w[f"{p}k_norm"],
        router=w[f"{p}router"].t(),
        fc1_w=w[f"{p}fc1_w"],
        fc1_s=w[f"{p}fc1_s"],
        fc2_w=w[f"{p}fc2_w"],
        fc2_s=w[f"{p}fc2_s"],
        sh_gu=w[f"{p}sh_gu"].t(),
        sh_dn=w[f"{p}sh_dn"].t(),
        sh_gate=w[f"{p}sh_gate"].t(),
        fhc_norm=core._norm_slices(f"{p}fhc_norm"),
        fhc_down=w[f"{p}fhc_down"],
        fhc_up=w[f"{p}fhc_up"],
    )


class PrefillTarget:
    """Any batch with context rows (`num_contexts > 0`), possibly with generation rows
    beside them; the recurrent layers' decode-only branches never run here."""

    def __init__(self, core: Qwen4ExpTp1ModelingV2) -> None:
        """Take the per-layer weight views once; MTP operands are read off `core._mtp`."""
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks; each target owns its flag (the
        # two `step_args` projections differ).
        self._contract_pending = step_contract_enabled()
        self._layers = _layer_views(core)
        self._stack = _stack_views(core)

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast (TRTLLM_MODELING_V2_VALIDATE): calling
        `step_args` surfaces an upstream metadata rename here, not mid-forward."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """Project the prepared metadata onto thop_attention's batch state, once per
        forward. Tensors are engine-owned persistent buffers refreshed in place;
        Python ints are per-capture constants."""
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
            # No sliding window: the full admitted length on every attention layer.
            attention_window_size=md.max_seq_len,
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

    # --------------------------------------------------------------- blocks ---

    def _grouped_norm(self, x: torch.Tensor, weights: tuple) -> torch.Tensor:
        """`Qwen4ExpTextRMSNorm(group_size=hidden)`: an independent RMS per `hidden`-wide
        group, then `(1 + w)` over the full width; one gemma-RMS-norm per column
        slice (row stride certified by that entry) plus a concat."""
        core = self.core
        parts = torch.split(x, [core.hidden] * core.hc_count, 1)
        return torch.cat(
            [flashinfer_gemma_rmsnorm(part, w, core.eps) for w, part in zip(weights, parts)], 1
        )

    def _moe_block(self, x: torch.Tensor, L: _LayerWeights, dt: torch.dtype) -> torch.Tensor:
        """NVFP4 routed experts (quantized once in the linear block-scale layout, with
        finished top-k ids and weights so the runner's routing arguments are inert)
        plus a bf16 shared expert folded in by a per-token sigmoid gate."""
        core = self.core
        # The routing op reads logits from data_ptr() ignoring strides; cublas_mm
        # allocates them contiguous.
        logits = cublas_mm(x, L.router)
        topk_ids, topk_w = renorm_moe_routing_op(logits, topk=core.topk, output_dtype=dt)
        xq, xsf = fp4_quantize(x, L.moe_gscale, _SF_VEC, False, _SF_LINEAR)
        routed = fp4_block_scale_moe_runner(
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
            num_experts=core.num_experts,
            top_k=core.topk,
            n_group=None,
            topk_group=None,
            intermediate_size=core.moe_inter,
            local_expert_offset=0,
            local_num_experts=core.num_experts,
            routed_scaling_factor=None,
            routing_method_type=_ROUTING_METHOD_RENORM,
            do_finalize=True,
            act_type=_ACT_TYPE_SWIGLU,
            topk_weights=topk_w,
            topk_ids=topk_ids,
        )[0]
        shared = cublas_mm(flashinfer_silu_and_mul(cublas_mm(x, L.sh_gu)), L.sh_dn)
        gate_logits = cublas_mm(x, L.sh_gate)
        return fused_sigmoid_gate_mul_add(routed, gate_logits, shared)

    def _attention_block(self, x, i, L, qkv, index_scratch, attn_out, qsa, step):
        """One gated GQA layer plus its QSA index branch.

        The q projection is twice as wide as q (q and output gate interleave per
        head, as HF chunks the last dim); only the q half is packed into the QKV
        buffer and the gate half is read later through its own strides. The index
        plane advances on every step, including below the threshold, so a request
        growing past it finds its prefix compressed; above it the main K/V append
        becomes this layer's own call."""
        core = self.core
        (
            req_idx,
            logical,
            seq_lens_q,
            visible_blocks,
            topk_indices,
            topk_row_starts,
            block_table,
            tokens_per_block,
            pos,
            pos3,
            coords,
            sparse,
        ) = qsa
        num_tokens = x.shape[0]

        qg = cublas_mm(x, L.q)
        k = cublas_mm(x, L.k)
        v = cublas_mm(x, L.v)

        # Destination views into the packed QKV buffer.
        qkv_h = torch.reshape(qkv, [num_tokens, core.heads_q + 2 * core.heads_kv, core.head_dim])
        q_dst, k_dst, v_dst = torch.split(qkv_h, [core.heads_q, core.heads_kv, core.heads_kv], 1)
        q_src, gate = torch.split(
            torch.reshape(qg, [num_tokens, core.heads_q, 2 * core.head_dim]),
            [core.head_dim, core.head_dim],
            2,
        )
        q_dst.copy_(q_src)
        k_dst.copy_(torch.reshape(k, [num_tokens, core.heads_kv, core.head_dim]))
        v_dst.copy_(torch.reshape(v, [num_tokens, core.heads_kv, core.head_dim]))
        fused_qk_norm_rope(
            qkv,
            num_heads_q=core.heads_q,
            num_heads_k=core.heads_kv,
            num_heads_v=core.heads_kv,
            head_dim=core.head_dim,
            rotary_dim=core.rotary_dim,
            eps=core.eps,
            q_weight=L.q_norm,
            k_weight=L.k_norm,
            base=core.theta,
            is_neox=True,
            position_ids=pos3,
            is_qk_norm=True,
            # The gemma QK norm, `rmsnorm(x) * (1 + w)`.
            use_gemma=True,
            # Real mRoPE, unconditionally. A wrong wiring is invisible on text (equal
            # rows) and ~500x out on an image batch.
            use_mrope=True,
            mrope_section1=core.mrope_section1,
            mrope_section2=core.mrope_section2,
        )

        # ---- the QSA index branch ------------------------------------------
        # One replicated projection produces [index_heads q | 1 k]. The raw key is
        # stored before any norm/rotation: the compression step applies both itself,
        # so a pre-normalized key is double-normalized (invisible below the
        # sparse threshold).
        qk_index = cublas_mm(x, L.idx)
        index_cache = core._bound.index_cache[i]
        token_k = torch.split(
            torch.reshape(qk_index, [num_tokens, core.index_heads + core.index_kv_heads, -1]),
            [core.index_heads, core.index_kv_heads],
            1,
        )[1]
        qsa_paged_index_store(
            token_k,
            coords,
            index_cache,
            core._bound.position_cache,
            req_idx,
            logical,
            block_table,
            tokens_per_block,
        )
        # Query norm+rotation in the fused call's packed layout; the k slot is
        # computed and discarded, the v slot exists only because the layout has one.
        # Coordinate 0 alone, matching `_QSA_MROPE_SECTION` on the key side.
        scratch_h = torch.reshape(index_scratch, [num_tokens, core.index_scratch_width])
        scratch_h[:, : core.index_width].copy_(qk_index)
        fused_qk_norm_rope(
            index_scratch,
            num_heads_q=core.index_heads,
            num_heads_k=core.index_kv_heads,
            num_heads_v=core.index_kv_heads,
            head_dim=core.index_head_dim,
            rotary_dim=core.rotary_dim,
            eps=core.eps,
            q_weight=L.idx_qn,
            k_weight=L.idx_kn,
            base=core.theta,
            is_neox=True,
            position_ids=pos3 if _QSA_INDEX_USE_MROPE else pos,
            is_qk_norm=True,
            use_gemma=True,
            use_mrope=_QSA_INDEX_USE_MROPE,
            mrope_section1=(_QSA_MROPE_SECTION or (0, 0, 0))[1],
            mrope_section2=(_QSA_MROPE_SECTION or (0, 0, 0))[2],
        )
        q_index = torch.split(
            torch.reshape(
                scratch_h,
                [num_tokens, core.index_heads + 2 * core.index_kv_heads, core.index_head_dim],
            ),
            [core.index_heads, core.index_kv_heads, core.index_kv_heads],
            1,
        )[0]
        qsa_prefill_compress(
            logical_positions=logical,
            request_indices=req_idx,
            block_table=block_table,
            index_cache=index_cache,
            position_cache=core._bound.position_cache,
            k_norm_weight=L.idx_kn,
            cos_sin=core._cos_sin,
            eps=core.eps,
            tokens_per_block=tokens_per_block,
            compress_ratio=core.compress_ratio,
            mrope_section=_QSA_MROPE_SECTION,
        )

        if sparse:
            k_cache, v_cache = core._bound.kv[i]
            # Append the fused-norm+rope'd K/V here: nothing else writes the pool on
            # this path, and selection reads the current token from it.
            qsa_paged_kv_store(
                k_dst, v_dst, k_cache, v_cache, req_idx, logical, block_table, tokens_per_block
            )
            scores = qsa_paged_index_scores(
                q=q_index,
                index_cache=index_cache,
                block_table=block_table,
                query_positions=logical,
                request_indices=req_idx,
                tokens_per_block=tokens_per_block,
                compress_ratio=core.compress_ratio,
                only_visible_blocks=True,
                context_rows=True,
            )
            # `index_topk` is the block budget: the entry's default (2048) is 4x the
            # buffer width and the op ignores the row stride, so it mis-writes silently.
            qsa_topk_select(
                scores, topk_row_starts, visible_blocks, topk_indices, index_topk=core.block_topk
            )
            selected = qsa_expand_block_indices(
                topk_indices,
                logical,
                seq_lens_q,
                compress_ratio=core.compress_ratio,
                token_topk=core.token_topk,
            )
            out = torch.reshape(
                qsa_paged_sparse_gqa(
                    q=q_dst,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=block_table,
                    selected_tokens=selected,
                    request_indices=req_idx,
                    tokens_per_block=tokens_per_block,
                    softmax_scale=core.softmax_scale,
                    # A bound, not a mask: lets the kernel stop scanning a row early
                    # (context-batch branch only).
                    query_positions=logical,
                    compress_ratio=core.compress_ratio,
                ),
                [num_tokens, core.q_width],
            )
        else:
            out = thop_attention(
                q=qkv,
                output=attn_out,
                local_layer_idx=core._bound.pool_idx[i],
                num_heads=core.heads_q,
                num_kv_heads=core.heads_kv,
                head_size=core.head_dim,
                quant_mode=core.quant_mode,
                **step,
                **_CALL_CONSTANTS,
            )
        fused_sigmoid_mul_inplace(out, gate)
        return cublas_mm(out, L.o)

    def _gdn_verify(self, mq_d, a_d, b_d, ssm_states, i, L, si_d, int_idx, n, p) -> torch.Tensor:
        """The recurrent half of a speculative verify step: `n` generation sequences of
        `p = runtime_draft_len + 1` tokens, with per-step conv window and recurrent
        state written to the runtime's rollback buffers, not the pools.

        The pools must not move: which prefix is real is unknown until the sampler
        accepts, and `update_mamba_states` then promotes step `k - 1`. Advancing them
        would leave rejected tokens' state in every recurrent layer, invisible to the
        acceptance rate. `causal_conv1d_triton_update` is the only conv with per-step
        snapshots (its result for a pad-sentinel slot is uninitialized, safe because
        every row is a real sequence); `pack_gdn_decode_qkv` does no Q/K L2 norm,
        hence `use_qk_l2norm_in_kernel=True` below."""
        core = self.core
        # [n*p, C] -> [n, C, p]: the multi-token conv reads channel-major (the
        # transposed view is certified bitwise equal to the contiguous form).
        x3 = torch.transpose(torch.reshape(mq_d, [n, p, core.conv_dim]), 1, 2)
        conv_out = causal_conv1d_triton_update(
            x=x3,
            conv_state=core._bound.conv[i],
            weight=L.conv_w,
            bias=None,
            activation="silu",
            conv_state_indices=si_d,
            pad_slot_id=_PAD_SLOT_ID,
            # Snapshots, not the accepted-token roll.
            num_accepted_tokens=None,
            intermediate_conv_window=core._bound.int_conv[i],
            intermediate_state_indices=int_idx,
            launch_dependent_kernels=False,
        )
        # Back to token-major (a view).
        mq_out = torch.reshape(torch.transpose(conv_out, 1, 2), [n * p, core.conv_dim])
        q, k, v = pack_gdn_decode_qkv(
            mq_out, core.num_k_heads, core.head_k_dim, core.num_v_heads, core.head_v_dim
        )
        g = torch.reshape(
            fused_gdn_gating(
                A_log=L.a_log,
                a=a_d,
                dt_bias=L.dt_bias,
                beta=_SOFTPLUS_BETA,
                threshold=_SOFTPLUS_THRESHOLD,
            ),
            [n, p, core.num_v_heads],
        )
        beta = torch.reshape(torch.sigmoid(b_d), [n, p, core.num_v_heads])
        # Initial states are gathered from the live pool into a batch-scoped copy. The
        # per-step buffer is indexed by batch row (`intermediate_state_indices[:n]`),
        # not slot id: slot ids would write rows nothing promotes, or past the end.
        src = torch.reshape(
            nn.functional.embedding(si_d, torch.reshape(ssm_states, [ssm_states.shape[0], -1])),
            [n, core.num_v_heads, core.head_v_dim, core.head_k_dim],
        )
        out = fused_recurrent_gated_delta_rule_update(
            torch.reshape(q, [n, p, core.num_k_heads, core.head_k_dim]),
            torch.reshape(k, [n, p, core.num_k_heads, core.head_k_dim]),
            torch.reshape(v, [n, p, core.num_v_heads, core.head_v_dim]),
            g,
            beta,
            initial_state_source=src,
            initial_state_indices=int_idx,
            scale=None,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
            disable_state_update=True,
            disable_output_calculation=False,
            intermediate_states_buffer=core._bound.int_ssm[i],
            cache_steps=p,
            output=None,
        )
        return torch.reshape(out, [1, n * p, core.num_v_heads, core.head_v_dim])

    def _gdn_block(self, x, i, L, gdn_state) -> torch.Tensor:
        """One Gated-DeltaNet layer over a batch with a context prefix. Every branch
        is on a host int the engine fixes per capture, never a tensor value."""
        core = self.core
        (
            num_prefill_tokens,
            num_decode_tokens,
            si_p,
            si_all,
            si_d,
            hi_p,
            hi_all,
            qsl_p,
            qsl_all,
            qsl_long,
            is_verify,
            num_decodes,
            gen_p,
            int_idx,
            qsl_long_p,
        ) = gdn_state
        num_tokens = x.shape[0]

        mixed_qkv = cublas_mm(x, L.in_qkv)
        z = cublas_mm(x, L.in_z)
        a = cublas_mm(x, L.in_a)
        b = cublas_mm(x, L.in_b)

        conv_states = core._bound.conv[i]
        ssm_states = core._bound.ssm[i]

        # A context sequence carrying nothing starts from a zeroed slot; only the
        # context prefix is a candidate (a generation row's state must survive).
        reset_recurrent_state_rows(ssm_states, si_p, hi_p, conv_states)
        if num_decode_tokens > 0:
            mq_p, mq_d = torch.split(mixed_qkv, [num_prefill_tokens, num_decode_tokens], 0)
            # Prefill conv wants channel-major, decode token-major.
            mq_p_t = extract_transpose_prefill_slice(mq_p, num_prefill_tokens, 0, core.conv_dim)
            causal_conv1d_fwd(
                x=mq_p_t,
                weight=L.conv_w,
                bias=None,
                conv_states=conv_states,
                query_start_loc=qsl_p,
                cache_indices=si_p,
                has_initial_state=hi_p,
                silu_activation=True,
                pad_slot_id=_PAD_SLOT_ID,
                out=None,
            )
            if is_verify:
                # A context row advances its slot as on the base path; a generation
                # row is a draft chain of unknown accepted length, so it goes to the
                # rollback buffers.
                a_p, a_d = torch.split(a, [num_prefill_tokens, num_decode_tokens], 0)
                b_p, b_d = torch.split(b, [num_prefill_tokens, num_decode_tokens], 0)
                q, k, v, g, beta = fused_gdn_post_conv(
                    prefill=mq_p_t,
                    decode=None,
                    a=a_p,
                    b=b_p,
                    A_log=L.a_log,
                    dt_bias=L.dt_bias,
                    num_k_heads=core.num_k_heads,
                    head_k_dim=core.head_k_dim,
                    num_v_heads=core.num_v_heads,
                    head_v_dim=core.head_v_dim,
                    l2_norm_eps=_L2_NORM_EPS,
                    softplus_threshold=_SOFTPLUS_THRESHOLD,
                )
                # q and k are already L2-normalized by the post-conv call.
                out_p, _ = chunk_gated_delta_rule(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    scale=None,
                    initial_state=ssm_states,
                    initial_state_indices=si_p,
                    inplace_indexed_state_update=True,
                    output_final_state=False,
                    cu_seqlens=qsl_long_p,
                    use_qk_l2norm_in_kernel=False,
                    output=None,
                )
                out_d = self._gdn_verify(
                    mq_d, a_d, b_d, ssm_states, i, L, si_d, int_idx, num_decodes, gen_p
                )
                return self._gdn_epilogue(torch.cat([out_p, out_d], 1), z, L, num_tokens)
            causal_conv1d_update(
                x=mq_d,
                conv_state=conv_states,
                weight=L.conv_w,
                bias=None,
                silu_activation=True,
                cache_seqlens=None,
                conv_state_indices=si_d,
                pad_slot_id=_PAD_SLOT_ID,
            )
            q, k, v, g, beta = fused_gdn_post_conv(
                prefill=mq_p_t,
                decode=mq_d,
                a=a,
                b=b,
                A_log=L.a_log,
                dt_bias=L.dt_bias,
                num_k_heads=core.num_k_heads,
                head_k_dim=core.head_k_dim,
                num_v_heads=core.num_v_heads,
                head_v_dim=core.head_v_dim,
                l2_norm_eps=_L2_NORM_EPS,
                softplus_threshold=_SOFTPLUS_THRESHOLD,
            )
        else:
            mq_t = extract_transpose_prefill_slice(mixed_qkv, num_tokens, 0, core.conv_dim)
            causal_conv1d_fwd(
                x=mq_t,
                weight=L.conv_w,
                bias=None,
                conv_states=conv_states,
                query_start_loc=qsl_all,
                cache_indices=si_all,
                has_initial_state=hi_all,
                silu_activation=True,
                pad_slot_id=_PAD_SLOT_ID,
                out=None,
            )
            q, k, v, g, beta = fused_gdn_post_conv(
                prefill=mq_t,
                decode=None,
                a=a,
                b=b,
                A_log=L.a_log,
                dt_bias=L.dt_bias,
                num_k_heads=core.num_k_heads,
                head_k_dim=core.head_k_dim,
                num_v_heads=core.num_v_heads,
                head_v_dim=core.head_v_dim,
                l2_norm_eps=_L2_NORM_EPS,
                softplus_threshold=_SOFTPLUS_THRESHOLD,
            )
        # q and k are already L2-normalized; the recurrence must not repeat it.
        out, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=None,
            initial_state=ssm_states,
            initial_state_indices=si_all,
            inplace_indexed_state_update=True,
            output_final_state=False,
            cu_seqlens=qsl_long,
            use_qk_l2norm_in_kernel=False,
            output=None,
        )
        return self._gdn_epilogue(out, z, L, num_tokens)

    def _gdn_epilogue(self, out, z, L, num_tokens) -> torch.Tensor:
        """Gated RMS norm and output projection shared by every recurrence."""
        core = self.core
        # Sigmoid gate and a plain (non-gemma) weight (`output_gate_type`).
        o = rms_norm_gated_token_major(
            torch.reshape(out, [-1, core.head_v_dim]),
            torch.reshape(z, [num_tokens, core.num_v_heads, core.head_v_dim]),
            weight=L.gnorm,
            eps=core.eps,
            fp8_scale=None,
            gate_activation="sigmoid",
        )
        return cublas_mm(torch.reshape(o, [num_tokens, core.value_dim]), L.out)

    def _ple_block(self, hc, input_ids, ple_state) -> torch.Tensor:
        """The n-gram side path on one decoder layer; the caller adds the result into
        the hyper-connection stream.

        Per-request stateful: ids become one `ngram_size`-wide window per token
        (predecessors from the paged id history); each window hashes to one row per
        head; rows gather fp8 vectors from one table and dequantize by one per-tensor
        scale; key/value projections and the gate scale one value into each stream;
        a dilated depthwise short conv over a paged state is added back. A sequence
        carrying nothing is seeded with `eos_token_id`, not 0: id 0 is a real token
        the hash cannot distinguish."""
        core = self.core
        S = self._stack
        conv_states, ngram_pool, si_all, carry_all, qsl = ple_state

        window = ngram_token_history(
            input_ids, ngram_pool, si_all, carry_all, qsl, core.eos_token_id, core.ngram_size
        )
        rows = torch.reshape(
            ngram_hash(
                window,
                layer_multipliers=S.ple_mult,
                head_vocab_sizes=S.ple_vocab,
                head_offsets=S.ple_offs,
                eos_token_id=core.eos_token_id,
                num_query_tokens=1,
            ),
            [-1, core.ngram_heads],
        )
        gathered = nn.functional.embedding(rows, core.w["ple_table"])
        ngram = torch.reshape(
            dequantize_e4m3_per_tensor(gathered, S.ple_tscale), [-1, core.ple_embed_dim]
        )

        key_normed = self._grouped_norm(cublas_mm(ngram, S.ple_key), S.ple_nk)
        query_normed = self._grouped_norm(hc, S.ple_nq)
        value = cublas_mm(ngram, S.ple_val)
        gated = gated_ngram_injection(
            key_normed, query_normed, value, core.hc_count, _PLE_CLAMP_MIN
        )
        # The short conv reads channel-major; a bare transpose is the certified form.
        convolved = dilated_short_conv_paged(
            x=torch.transpose(self._grouped_norm(gated, S.ple_nc), 0, 1),
            weight=S.ple_conv_w,
            bias=None,
            conv_states=conv_states,
            query_start_loc=qsl,
            cache_indices=si_all,
            has_initial_state=carry_all,
            dilation=core.ple_dilation,
            silu_activation=True,
            pad_slot_id=_PAD_SLOT_ID,
        )
        return torch.add(gated, torch.transpose(convolved, 0, 1))

    # -------------------------------------------------------------- forward ---

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs,
    ):
        core = self.core
        md = attn_metadata
        self._check_step_contract(md)
        step = self.step_args(md)
        spec_metadata = kwargs.get("spec_metadata")
        mm = md.mamba_metadata

        # Three views of one `[3, 1, T]` buffer: `pos` (row 0, flat `[T]`: QSA index
        # rope), `pos3` (contiguous `[3, T]` for `use_mrope=True`; the engine hands a
        # slice of a larger buffer and the op rejects the row stride, hence the
        # copy), `coords` (`[T, 3]`, stored by the QSA position side cache).
        pos = torch.reshape(torch.split(position_ids, [1, _QSA_POSITION_AXES - 1], 0)[0], [-1])
        coords = torch.transpose(torch.reshape(position_ids, [_QSA_POSITION_AXES, -1]), 0, 1)
        pos3 = torch.empty(
            [_QSA_POSITION_AXES, position_ids.shape[-1]],
            dtype=torch.int32,
            device=position_ids.device,
        )
        pos3.copy_(torch.reshape(position_ids, [_QSA_POSITION_AXES, -1]))

        # Spliced vision rows are already the embedding; a lookup would overwrite them.
        if inputs_embeds is None:
            h = nn.functional.embedding(input_ids, core.w["embed"])
        else:
            assert inputs_embeds.shape[0] == input_ids.shape[0], (
                f"the spliced embedding carries {inputs_embeds.shape[0]} rows for "
                f"{input_ids.shape[0]} token ids"
            )
            assert inputs_embeds.shape[1] == core.hidden, (
                f"the spliced embedding is {inputs_embeds.shape[1]} wide, not {core.hidden}"
            )
            h = inputs_embeds
        num_tokens = h.shape[0]
        dt = h.dtype
        dev = h.device

        # The batch's bookkeeping, built once and shared by every layer.
        batch = md.num_seqs
        num_prefills = md.num_contexts
        num_decodes = batch - num_prefills
        num_prefill_tokens = md.num_ctx_tokens
        num_decode_tokens = num_tokens - num_prefill_tokens
        # Host-int check (graph-safe): an activation buffer longer than the batch
        # would push padding rows through the recurrences into the state pools.
        assert num_tokens == md.num_tokens, (
            f"the engine handed {num_tokens} activation rows for a batch its "
            f"metadata describes as {md.num_tokens} tokens "
            f"(padded_num_tokens={getattr(md, 'padded_num_tokens', None)}). A "
            "prefill/piecewise CUDA-graph backend pads the activation buffer "
            "past the batch and expects the model to slice it back; this "
            "target does not implement that configuration. Decode CUDA graphs "
            "pad with real dummy requests instead, so they are in the metadata "
            "and this assert is silent for them."
        )
        # Rows per generation sequence: 1 without speculation, else
        # `runtime_draft_len + 1`; read from the metadata because the worker rewrites
        # it between draft steps.
        gen_p = num_decode_tokens // num_decodes if num_decodes else 1
        assert gen_p * num_decodes == num_decode_tokens, (
            f"{num_decode_tokens} generation rows do not divide over "
            f"{num_decodes} generation sequences; a ragged per-sequence draft "
            "length can be expressed neither to the attention call nor to the "
            "multi-token recurrence"
        )
        assert core.mtp_enabled or gen_p == 1, (
            f"{num_decode_tokens} generation rows over {num_decodes} generation "
            "sequences with speculation off; the base path implements one query "
            "token per generation sequence, and both the attention call's "
            "within-block mask and the recurrent path's cumulative-length "
            "vector assume it"
        )
        # `gen_p > 1` is load-bearing: verify parks a generation row in the rollback
        # buffers, which `update_mamba_states` promotes only if the worker drafts.
        # With `runtime_draft_len == 0` (a `draft_len_schedule` can do this) it skips
        # drafting before the promotion, leaving every recurrent layer one token
        # stale; that row is the always-accepted golden token, so the ordinary decode
        # path is right.
        if spec_metadata is not None and num_decodes > 0:
            declared_p = spec_metadata.runtime_draft_len + 1
            assert gen_p == declared_p, (
                f"the batch carries {gen_p} query rows per generation sequence "
                f"but the worker's runtime_draft_len says {declared_p}; the "
                "per-step conv snapshots and the recurrent rollback buffer are "
                "sized from the second and written from the first"
            )
            # Per-iteration counterpart of the core's `max_draft_len` bound.
            assert 1 <= gen_p <= 4, (
                f"predicted_tokens_per_seq {gen_p} is outside the 1..4 range "
                f"attention/thop_attention certifies"
            )
        is_verify = spec_metadata is not None and num_decodes > 0 and gen_p > 1
        step["predicted_tokens_per_seq"] = gen_p

        si_all = mm.state_indices[:batch]
        hi_all = mm.has_initial_states[:batch]
        si_p, si_d = torch.split(si_all, [num_prefills, num_decodes], 0)
        hi_p = hi_all[:num_prefills]
        # `query_start_loc` covers the context prefix; `_long` the whole batch.
        qsl_all = mm.query_start_loc
        qsl_p = None if qsl_all is None else qsl_all[: num_prefills + 1]
        qsl_long = mm.query_start_loc_long
        int_idx = None
        qsl_long_p = None
        if is_verify:
            # The runtime's arange over the slot pool, from the cache manager (what
            # `update_mamba_states` reads).
            assert core._bound.int_idx is not None, "the cache-manager bind has not run"
            int_idx = core._bound.int_idx[:num_decodes]
            # Context rows keep an ordinary chunked recurrence over their prefix.
            qsl_long_p = qsl_long[: num_prefills + 1]
        gdn_state = (
            num_prefill_tokens,
            num_decode_tokens,
            si_p,
            si_all,
            si_d,
            hi_p,
            hi_all,
            qsl_p,
            qsl_all,
            qsl_long,
            is_verify,
            num_decodes,
            gen_p,
            int_idx,
            qsl_long_p,
        )

        # PLE carry flags must cover the whole batch, but `has_initial_states` is
        # filled for the context prefix only (generation entries are stale); every
        # generation sequence carries, so pad with True (host-int sized, graph-safe).
        ple_carry = nn.functional.pad(hi_p, [0, num_decodes], mode="constant", value=1.0)
        # Under speculation `Mamba2Metadata.prepare`'s `query_start_loc_long` has one
        # row per generation sequence, so with `gen_p > 1` it is short and both PLE
        # stateful calls are silently mis-segmented; rebuild it from `seq_lens_cuda`,
        # which still holds the true counts during the trunk forward.
        if gen_p == 1:
            ple_qsl = qsl_long
        else:
            ple_qsl = cu_seqlens_from_seq_lens(md.seq_lens_cuda[:batch], torch.int64)
        ple_state = (core._bound.ple_state[0], core._bound.ple_state[1], si_all, ple_carry, ple_qsl)

        # Dense/sparse decision once per forward, from a host-side metadata read (as
        # QSA's backend does): a per-capture constant, matching the engine's separate
        # short/long decode graph families.
        sparse = batch > 0 and int(md.kv_lens_runtime[:batch].max()) > core.dense_threshold

        qsa = (
            md.qsa_req_idx_per_token[:num_tokens],
            md.qsa_logical_positions[:num_tokens],
            md.qsa_sequence_lengths[:num_tokens],
            md.qsa_visible_blocks[:num_tokens],
            md.qsa_topk_indices[:num_tokens],
            md.qsa_topk_row_starts[:num_tokens],
            md.qsa_block_table,
            md.tokens_per_block,
            pos,
            pos3,
            coords,
            sparse,
        )

        # Scratch shared by all attention layers (each fully rewrites them).
        qkv = torch.empty([num_tokens, core.qkv_width], dtype=dt, device=dev)
        index_scratch = torch.empty([num_tokens, core.index_scratch_width], dtype=dt, device=dev)
        attn_out = torch.empty([num_tokens, core.q_width], dtype=dt, device=dev)

        # The embedding repeated across the hc_count residual streams.
        hc_count = core.hc_count
        hc = torch.cat([h] * hc_count, -1)
        for i, L in enumerate(self._layers):
            if i == core.ple_layer:
                hc = torch.add(hc, self._ple_block(hc, input_ids, ple_state))
            mixed, inject = qwen4_exp_gated_residual(
                self._grouped_norm(hc, L.ahc_norm), L.ahc_down, L.ahc_up, L.ahc_inject, hc_count
            )
            if core.is_attn[i]:
                block_out = self._attention_block(
                    mixed, i, L, qkv, index_scratch, attn_out, qsa, step
                )
            else:
                block_out = self._gdn_block(mixed, i, L, gdn_state)
            hc = qwen4_exp_post_inject(hc, block_out, inject)
            mixed, inject = qwen4_exp_gated_residual(
                self._grouped_norm(hc, L.mhc_norm), L.mhc_down, L.mhc_up, L.mhc_inject, hc_count
            )
            hc = qwen4_exp_post_inject(hc, self._moe_block(mixed, L, dt), inject)
        # The stack-final mixer with no injection weight is the final norm.
        S = self._stack
        final, _ = qwen4_exp_gated_residual(
            self._grouped_norm(hc, S.fhc_norm), S.fhc_down, S.fhc_up, None, hc_count
        )
        if spec_metadata is None:
            return final
        # `lm_head` reads the `[T, hidden]` mix; the MTP module reads the
        # `[T, hc_count * hidden]` stream it was collapsed from.
        return final, hc

    # ------------------------------------------------------------- the draft ---

    def _mtp_moe_block(self, x, dt) -> torch.Tensor:
        """The draft layer's MoE: FP8 block-scale routed experts plus a bf16 shared expert.

        The scale buffer comes back transposed `[hidden/128, tokens]` and is passed
        straight through; never reshape it. At `tokens == hidden/128` (20, an
        ordinary decode batch) the shapes coincide, so nothing would catch a
        re-materialization. Pre-routed; `routing_bias` must be `None` (the entry
        refuses one)."""
        core = self.core
        M = core._mtp
        logits = cublas_mm(x, M.router)
        topk_ids, topk_w = renorm_moe_routing_op(logits, topk=core.topk, output_dtype=dt)
        xq, xsf = fp8_quantize_1x128(x, use_ue8m0=_MTP_MOE_USE_UE8M0)
        routed = fp8_block_scale_moe_runner(
            routing_logits=None,
            routing_bias=None,
            hidden_states=xq,
            hidden_states_scale=xsf,
            gemm1_weights=M.fc1_w,
            gemm1_weights_scale=M.fc1_s,
            gemm2_weights=M.fc2_w,
            gemm2_weights_scale=M.fc2_s,
            num_experts=core.num_experts,
            top_k=core.topk,
            num_fused_shared_experts=None,
            n_group=None,
            topk_group=None,
            intermediate_size=core.moe_inter,
            local_expert_offset=0,
            local_num_experts=core.num_experts,
            routed_scaling_factor=None,
            routing_method_type=_ROUTING_METHOD_RENORM,
            topk_weights=topk_w,
            topk_ids=topk_ids,
            act_type=_ACT_TYPE_SWIGLU,
            gemm1_clamp_limit=None,
        )
        shared = cublas_mm(flashinfer_silu_and_mul(cublas_mm(x, M.sh_gu)), M.sh_dn)
        gate_logits = cublas_mm(x, M.sh_gate)
        return fused_sigmoid_gate_mul_add(routed, gate_logits, shared)

    def mtp_forward(
        self,
        embed_tokens: torch.Tensor,
        all_rank_num_tokens,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """One draft step whose metadata still carries context rows.

        `spec_metadata` and `all_rank_num_tokens` are accepted and unused."""
        core = self.core
        assert isinstance(attn_metadata, TrtllmAttentionMetadata)
        assert core._bound is not None, (
            "the trunk's cache-manager bind has not run; the shell calls the "
            "core before the worker, so this cannot be reached first"
        )
        md = attn_metadata
        rows = hidden_states.shape[0]
        assert input_ids.shape[0] == rows, (
            f"the draft step's {input_ids.shape[0]} token ids and {rows} "
            "hidden-state rows must describe the same tokens"
        )
        assert hidden_states.shape[1] == core.stream, (
            f"the MTP module reads the trunk's {core.stream}-wide four-stream "
            f"residual; the runtime handed it {hidden_states.shape[1]} columns"
        )
        dt = hidden_states.dtype
        dev = hidden_states.device

        # Projected on every call: the worker rewrites `attn_metadata` in place
        # between draft step 0 and 1+ (`_seq_lens` to 1, `num_contexts` and
        # `num_ctx_tokens` to 0), so a first-call value is wrong on the rest.
        step = self.step_args(md)
        M = core._mtp
        num_ctx = md.num_ctx_tokens
        gen_rows = rows - num_ctx
        gen_seqs = md.num_seqs - md.num_contexts
        p = gen_rows // gen_seqs if gen_seqs else 1
        assert p * gen_seqs == gen_rows, (
            f"{gen_rows} generation rows do not divide over {gen_seqs} "
            "generation sequences; a ragged per-sequence draft length cannot be "
            "expressed to the attention call"
        )

        # The KV cache written here is not the trunk's: with a separate draft KV
        # manager the worker swaps `attn_metadata.kv_cache_manager` and
        # `kv_cache_block_offsets` for the draft loop. The layer index must be taken
        # from the current manager every call; caching the trunk's addresses another
        # layer's pages with finite, plausible drafts.
        cm = md.kv_cache_manager
        assert cm is not None, "a paged KV cache is required"
        offsets = cm.layer_offsets
        assert core.mtp_layer_idx in offsets, (
            f"the MTP module's attention is global layer index "
            f"{core.mtp_layer_idx}, which the cache manager "
            f"{type(cm).__name__} does not map (it maps {len(offsets)} layers, "
            f"is_draft={getattr(cm, 'is_draft', None)}); the engine allocated "
            "no KV layer for the draft module"
        )
        layer_idx = offsets[core.mtp_layer_idx]
        window = cm.max_attention_window_vec[layer_idx]
        assert window is None or window == md.max_seq_len, (
            f"the MTP layer's pool window is {window!r} but this call passes "
            f"max_seq_len {md.max_seq_len} as the attention window; a sliding "
            "window is not implemented here"
        )

        # Copy into a fresh contiguous `[3, N]`: a sliced view keeps the source row
        # stride and the op rejects it under `use_mrope`.
        assert position_ids.shape[0] == _QSA_POSITION_AXES, (
            f"expected the mRoPE position layout with {_QSA_POSITION_AXES} "
            f"coordinate rows, got {tuple(position_ids.shape)}"
        )
        pos3 = torch.empty(
            [_QSA_POSITION_AXES, position_ids.numel() // _QSA_POSITION_AXES],
            dtype=torch.int32,
            device=position_ids.device,
        )
        pos3.copy_(torch.reshape(position_ids, [_QSA_POSITION_AXES, -1]))

        # ---- the front end --------------------------------------------------
        e = nn.functional.embedding(input_ids, embed_tokens)
        x_e = cublas_mm(flashinfer_gemma_rmsnorm(e, M.pre_e_norm, core.eps), M.fc_e)
        # `fc_hidden` is one `[hidden, hidden]` matrix applied to each of the four
        # streams (the checkpoint's norm statistics say they are distinct streams).
        h_n = self._grouped_norm(hidden_states, M.pre_h_norm)
        x_h = torch.reshape(
            cublas_mm(torch.reshape(h_n, [rows * core.hc_count, core.hidden]), M.fc_h),
            [rows, core.stream],
        )
        if _MTP_EMBED_BLOCK_ALL_STREAMS:
            x = torch.add(x_h, torch.cat([x_e] * core.hc_count, -1))
        else:
            x = torch.add(
                x_h,
                nn.functional.pad(x_e, [0, core.stream - core.hidden], mode="constant", value=0.0),
            )

        # ---- the decoder layer ----------------------------------------------
        mixed, inject = qwen4_exp_gated_residual(
            self._grouped_norm(x, M.ahc_norm), M.ahc_down, M.ahc_up, M.ahc_inject, core.hc_count
        )
        qkv = torch.empty([rows, core.qkv_width], dtype=dt, device=dev)
        attn_out = torch.empty([rows, core.q_width], dtype=dt, device=dev)
        qg = cublas_mm(mixed, M.q)
        k = cublas_mm(mixed, M.k)
        v = cublas_mm(mixed, M.v)
        qkv_h = torch.reshape(qkv, [rows, core.heads_q + 2 * core.heads_kv, core.head_dim])
        q_dst, k_dst, v_dst = torch.split(qkv_h, [core.heads_q, core.heads_kv, core.heads_kv], 1)
        q_src, gate = torch.split(
            torch.reshape(qg, [rows, core.heads_q, 2 * core.head_dim]),
            [core.head_dim, core.head_dim],
            2,
        )
        q_dst.copy_(q_src)
        k_dst.copy_(torch.reshape(k, [rows, core.heads_kv, core.head_dim]))
        v_dst.copy_(torch.reshape(v, [rows, core.heads_kv, core.head_dim]))
        # The trunk rope's certified cell at the module's own base (`mtp.rope_theta`).
        fused_qk_norm_rope(
            qkv,
            num_heads_q=core.heads_q,
            num_heads_k=core.heads_kv,
            num_heads_v=core.heads_kv,
            head_dim=core.head_dim,
            rotary_dim=core.rotary_dim,
            eps=core.eps,
            q_weight=M.q_norm,
            k_weight=M.k_norm,
            base=core.mtp_theta,
            is_neox=True,
            position_ids=pos3,
            is_qk_norm=True,
            use_gemma=True,
            use_mrope=True,
            mrope_section1=core.mrope_section1,
            mrope_section2=core.mrope_section2,
        )
        # Dense attention; see `MTPLayer` for why the module's QSA indexer is unused.
        attn_out = thop_attention(
            q=qkv,
            output=attn_out,
            local_layer_idx=layer_idx,
            predicted_tokens_per_seq=p,
            num_heads=core.heads_q,
            num_kv_heads=core.heads_kv,
            head_size=core.head_dim,
            quant_mode=core.quant_mode,
            **step,
            **_CALL_CONSTANTS,
        )
        fused_sigmoid_mul_inplace(attn_out, gate)
        x = qwen4_exp_post_inject(x, cublas_mm(attn_out, M.o), inject)

        mixed, inject = qwen4_exp_gated_residual(
            self._grouped_norm(x, M.mhc_norm), M.mhc_down, M.mhc_up, M.mhc_inject, core.hc_count
        )
        return qwen4_exp_post_inject(x, self._mtp_moe_block(mixed, dt), inject)


class DecodeTarget:
    """Generation-only batches (no context rows): drops the context-row recurrent
    ops and adds the fused single-token recurrence."""

    def __init__(self, core: Qwen4ExpTp1ModelingV2) -> None:
        """Take the per-layer weight views once; MTP operands are read off `core._mtp`."""
        self.core = core
        # See PrefillTarget.__init__.
        self._contract_pending = step_contract_enabled()
        self._layers = _layer_views(core)
        self._stack = _stack_views(core)

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast; see PrefillTarget._check_step_contract."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """See PrefillTarget.step_args. Context counts are stated, not read: routing
        guarantees them, and a decode-only CUDA graph would freeze whatever it read."""
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
            # No sliding window: the full admitted length on every attention layer.
            attention_window_size=md.max_seq_len,
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

    # --------------------------------------------------------------- blocks ---

    def _grouped_norm(self, x: torch.Tensor, weights: tuple) -> torch.Tensor:
        """See `PrefillTarget._grouped_norm`."""
        core = self.core
        parts = torch.split(x, [core.hidden] * core.hc_count, 1)
        return torch.cat(
            [flashinfer_gemma_rmsnorm(part, w, core.eps) for w, part in zip(weights, parts)], 1
        )

    def _moe_block(self, x: torch.Tensor, L: _LayerWeights, dt: torch.dtype) -> torch.Tensor:
        """See `PrefillTarget._moe_block`."""
        core = self.core
        # The routing op reads logits from data_ptr() ignoring strides.
        logits = cublas_mm(x, L.router)
        topk_ids, topk_w = renorm_moe_routing_op(logits, topk=core.topk, output_dtype=dt)
        xq, xsf = fp4_quantize(x, L.moe_gscale, _SF_VEC, False, _SF_LINEAR)
        routed = fp4_block_scale_moe_runner(
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
            num_experts=core.num_experts,
            top_k=core.topk,
            n_group=None,
            topk_group=None,
            intermediate_size=core.moe_inter,
            local_expert_offset=0,
            local_num_experts=core.num_experts,
            routed_scaling_factor=None,
            routing_method_type=_ROUTING_METHOD_RENORM,
            do_finalize=True,
            act_type=_ACT_TYPE_SWIGLU,
            topk_weights=topk_w,
            topk_ids=topk_ids,
        )[0]
        shared = cublas_mm(flashinfer_silu_and_mul(cublas_mm(x, L.sh_gu)), L.sh_dn)
        gate_logits = cublas_mm(x, L.sh_gate)
        return fused_sigmoid_gate_mul_add(routed, gate_logits, shared)

    def _attention_block(self, x, i, L, qkv, index_scratch, attn_out, qsa, step):
        """One gated GQA layer plus its QSA index branch for a generation-only batch;
        see `PrefillTarget._attention_block`."""
        core = self.core
        (
            req_idx,
            logical,
            seq_lens_q,
            visible_blocks,
            topk_indices,
            topk_row_starts,
            block_table,
            tokens_per_block,
            pos,
            pos3,
            coords,
            sparse,
        ) = qsa
        num_tokens = x.shape[0]

        qg = cublas_mm(x, L.q)
        k = cublas_mm(x, L.k)
        v = cublas_mm(x, L.v)

        # Destination views into the packed QKV buffer.
        qkv_h = torch.reshape(qkv, [num_tokens, core.heads_q + 2 * core.heads_kv, core.head_dim])
        q_dst, k_dst, v_dst = torch.split(qkv_h, [core.heads_q, core.heads_kv, core.heads_kv], 1)
        q_src, gate = torch.split(
            torch.reshape(qg, [num_tokens, core.heads_q, 2 * core.head_dim]),
            [core.head_dim, core.head_dim],
            2,
        )
        q_dst.copy_(q_src)
        k_dst.copy_(torch.reshape(k, [num_tokens, core.heads_kv, core.head_dim]))
        v_dst.copy_(torch.reshape(v, [num_tokens, core.heads_kv, core.head_dim]))
        fused_qk_norm_rope(
            qkv,
            num_heads_q=core.heads_q,
            num_heads_k=core.heads_kv,
            num_heads_v=core.heads_kv,
            head_dim=core.head_dim,
            rotary_dim=core.rotary_dim,
            eps=core.eps,
            q_weight=L.q_norm,
            k_weight=L.k_norm,
            base=core.theta,
            is_neox=True,
            position_ids=pos3,
            is_qk_norm=True,
            # The gemma QK norm, `rmsnorm(x) * (1 + w)`.
            use_gemma=True,
            # Real mRoPE, unconditionally (see PrefillTarget).
            use_mrope=True,
            mrope_section1=core.mrope_section1,
            mrope_section2=core.mrope_section2,
        )

        # ---- the QSA index branch (see PrefillTarget) ------------------------
        # The raw key is stored before any norm/rotation; the compression step
        # applies both itself.
        qk_index = cublas_mm(x, L.idx)
        index_cache = core._bound.index_cache[i]
        token_k = torch.split(
            torch.reshape(qk_index, [num_tokens, core.index_heads + core.index_kv_heads, -1]),
            [core.index_heads, core.index_kv_heads],
            1,
        )[1]
        qsa_paged_index_store(
            token_k,
            coords,
            index_cache,
            core._bound.position_cache,
            req_idx,
            logical,
            block_table,
            tokens_per_block,
        )
        # Query norm+rotation; coordinate 0 alone, matching `_QSA_MROPE_SECTION`.
        scratch_h = torch.reshape(index_scratch, [num_tokens, core.index_scratch_width])
        scratch_h[:, : core.index_width].copy_(qk_index)
        fused_qk_norm_rope(
            index_scratch,
            num_heads_q=core.index_heads,
            num_heads_k=core.index_kv_heads,
            num_heads_v=core.index_kv_heads,
            head_dim=core.index_head_dim,
            rotary_dim=core.rotary_dim,
            eps=core.eps,
            q_weight=L.idx_qn,
            k_weight=L.idx_kn,
            base=core.theta,
            is_neox=True,
            position_ids=pos3 if _QSA_INDEX_USE_MROPE else pos,
            is_qk_norm=True,
            use_gemma=True,
            use_mrope=_QSA_INDEX_USE_MROPE,
            mrope_section1=(_QSA_MROPE_SECTION or (0, 0, 0))[1],
            mrope_section2=(_QSA_MROPE_SECTION or (0, 0, 0))[2],
        )
        q_index = torch.split(
            torch.reshape(
                scratch_h,
                [num_tokens, core.index_heads + 2 * core.index_kv_heads, core.index_head_dim],
            ),
            [core.index_heads, core.index_kv_heads, core.index_kv_heads],
            1,
        )[0]
        qsa_prefill_compress(
            logical_positions=logical,
            request_indices=req_idx,
            block_table=block_table,
            index_cache=index_cache,
            position_cache=core._bound.position_cache,
            k_norm_weight=L.idx_kn,
            cos_sin=core._cos_sin,
            eps=core.eps,
            tokens_per_block=tokens_per_block,
            compress_ratio=core.compress_ratio,
            mrope_section=_QSA_MROPE_SECTION,
        )

        if sparse:
            k_cache, v_cache = core._bound.kv[i]
            # Append the fused-norm+rope'd K/V; the selection reads the current
            # token from the pool.
            qsa_paged_kv_store(
                k_dst, v_dst, k_cache, v_cache, req_idx, logical, block_table, tokens_per_block
            )
            scores = qsa_paged_index_scores(
                q=q_index,
                index_cache=index_cache,
                block_table=block_table,
                query_positions=logical,
                request_indices=req_idx,
                tokens_per_block=tokens_per_block,
                compress_ratio=core.compress_ratio,
                only_visible_blocks=True,
                context_rows=False,
            )
            # `index_topk` is the block budget (see PrefillTarget).
            qsa_topk_select(
                scores, topk_row_starts, visible_blocks, topk_indices, index_topk=core.block_topk
            )
            selected = qsa_expand_block_indices(
                topk_indices,
                logical,
                seq_lens_q,
                compress_ratio=core.compress_ratio,
                token_topk=core.token_topk,
            )
            out = torch.reshape(
                qsa_paged_sparse_gqa(
                    q=q_dst,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=block_table,
                    selected_tokens=selected,
                    request_indices=req_idx,
                    tokens_per_block=tokens_per_block,
                    softmax_scale=core.softmax_scale,
                    # The prefill pair bounds a context row's scan; none here.
                    query_positions=None,
                    compress_ratio=None,
                ),
                [num_tokens, core.q_width],
            )
        else:
            out = thop_attention(
                q=qkv,
                output=attn_out,
                local_layer_idx=core._bound.pool_idx[i],
                num_heads=core.heads_q,
                num_kv_heads=core.heads_kv,
                head_size=core.head_dim,
                quant_mode=core.quant_mode,
                **step,
                **_CALL_CONSTANTS,
            )
        fused_sigmoid_mul_inplace(out, gate)
        return cublas_mm(out, L.o)

    def _gdn_verify(self, mq_d, a_d, b_d, ssm_states, i, L, si_d, int_idx, n, p) -> torch.Tensor:
        """Recurrent half of a speculative verify step; see
        `PrefillTarget._gdn_verify`."""
        core = self.core
        # [n*p, C] -> [n, C, p]: the multi-token conv reads channel-major.
        x3 = torch.transpose(torch.reshape(mq_d, [n, p, core.conv_dim]), 1, 2)
        conv_out = causal_conv1d_triton_update(
            x=x3,
            conv_state=core._bound.conv[i],
            weight=L.conv_w,
            bias=None,
            activation="silu",
            conv_state_indices=si_d,
            pad_slot_id=_PAD_SLOT_ID,
            # Snapshots, not the accepted-token roll.
            num_accepted_tokens=None,
            intermediate_conv_window=core._bound.int_conv[i],
            intermediate_state_indices=int_idx,
            launch_dependent_kernels=False,
        )
        # Back to token-major (a view).
        mq_out = torch.reshape(torch.transpose(conv_out, 1, 2), [n * p, core.conv_dim])
        q, k, v = pack_gdn_decode_qkv(
            mq_out, core.num_k_heads, core.head_k_dim, core.num_v_heads, core.head_v_dim
        )
        g = torch.reshape(
            fused_gdn_gating(
                A_log=L.a_log,
                a=a_d,
                dt_bias=L.dt_bias,
                beta=_SOFTPLUS_BETA,
                threshold=_SOFTPLUS_THRESHOLD,
            ),
            [n, p, core.num_v_heads],
        )
        beta = torch.reshape(torch.sigmoid(b_d), [n, p, core.num_v_heads])
        # Initial states are gathered from the live pool; the per-step buffer is
        # indexed by batch row (`intermediate_state_indices[:n]`), not slot id.
        src = torch.reshape(
            nn.functional.embedding(si_d, torch.reshape(ssm_states, [ssm_states.shape[0], -1])),
            [n, core.num_v_heads, core.head_v_dim, core.head_k_dim],
        )
        out = fused_recurrent_gated_delta_rule_update(
            torch.reshape(q, [n, p, core.num_k_heads, core.head_k_dim]),
            torch.reshape(k, [n, p, core.num_k_heads, core.head_k_dim]),
            torch.reshape(v, [n, p, core.num_v_heads, core.head_v_dim]),
            g,
            beta,
            initial_state_source=src,
            initial_state_indices=int_idx,
            scale=None,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
            disable_state_update=True,
            disable_output_calculation=False,
            intermediate_states_buffer=core._bound.int_ssm[i],
            cache_steps=p,
            output=None,
        )
        return torch.reshape(out, [1, n * p, core.num_v_heads, core.head_v_dim])

    def _gdn_block(self, x, i, L, gdn_state) -> torch.Tensor:
        """One Gated-DeltaNet layer over a generation-only batch: one token per
        sequence, or under speculation the whole draft chain via the verify path."""
        core = self.core
        si_all, qsl_long, is_verify, num_decodes, gen_p, int_idx = gdn_state
        num_tokens = x.shape[0]

        mixed_qkv = cublas_mm(x, L.in_qkv)
        z = cublas_mm(x, L.in_z)
        a = cublas_mm(x, L.in_a)
        b = cublas_mm(x, L.in_b)

        ssm_states = core._bound.ssm[i]
        if is_verify:
            out = self._gdn_verify(
                mixed_qkv, a, b, ssm_states, i, L, si_all, int_idx, num_decodes, gen_p
            )
        else:
            causal_conv1d_update(
                x=mixed_qkv,
                conv_state=core._bound.conv[i],
                weight=L.conv_w,
                bias=None,
                silu_activation=True,
                cache_seqlens=None,
                conv_state_indices=si_all,
                pad_slot_id=_PAD_SLOT_ID,
            )
            # q/k/v stay views over the conv output (native strides, no copy).
            qv, kv, vv = torch.split(mixed_qkv, [core.key_dim, core.key_dim, core.value_dim], 1)
            out = fused_sigmoid_gating_delta_rule_update(
                A_log=L.a_log,
                a=a,
                dt_bias=L.dt_bias,
                softplus_beta=_SOFTPLUS_BETA,
                softplus_threshold=_SOFTPLUS_THRESHOLD,
                q=torch.reshape(qv, [1, num_tokens, core.num_k_heads, core.head_k_dim]),
                k=torch.reshape(kv, [1, num_tokens, core.num_k_heads, core.head_k_dim]),
                v=torch.reshape(vv, [1, num_tokens, core.num_v_heads, core.head_v_dim]),
                b=b,
                initial_state_source=ssm_states,
                initial_state_indices=si_all,
                scale=None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=qsl_long,
                output=None,
            )
        return self._gdn_epilogue(out, z, L, num_tokens)

    def _gdn_epilogue(self, out, z, L, num_tokens) -> torch.Tensor:
        """Gated RMS norm and output projection shared by every recurrence."""
        core = self.core
        # Sigmoid gate, plain (non-gemma) weight.
        o = rms_norm_gated_token_major(
            torch.reshape(out, [-1, core.head_v_dim]),
            torch.reshape(z, [num_tokens, core.num_v_heads, core.head_v_dim]),
            weight=L.gnorm,
            eps=core.eps,
            fp8_scale=None,
            gate_activation="sigmoid",
        )
        return cublas_mm(torch.reshape(o, [num_tokens, core.value_dim]), L.out)

    def _ple_block(self, hc, input_ids, ple_state) -> torch.Tensor:
        """The n-gram side path on one decoder layer; see `PrefillTarget._ple_block`."""
        core = self.core
        S = self._stack
        conv_states, ngram_pool, si_all, carry_all, qsl = ple_state

        window = ngram_token_history(
            input_ids, ngram_pool, si_all, carry_all, qsl, core.eos_token_id, core.ngram_size
        )
        rows = torch.reshape(
            ngram_hash(
                window,
                layer_multipliers=S.ple_mult,
                head_vocab_sizes=S.ple_vocab,
                head_offsets=S.ple_offs,
                eos_token_id=core.eos_token_id,
                num_query_tokens=1,
            ),
            [-1, core.ngram_heads],
        )
        gathered = nn.functional.embedding(rows, core.w["ple_table"])
        ngram = torch.reshape(
            dequantize_e4m3_per_tensor(gathered, S.ple_tscale), [-1, core.ple_embed_dim]
        )

        key_normed = self._grouped_norm(cublas_mm(ngram, S.ple_key), S.ple_nk)
        query_normed = self._grouped_norm(hc, S.ple_nq)
        value = cublas_mm(ngram, S.ple_val)
        gated = gated_ngram_injection(
            key_normed, query_normed, value, core.hc_count, _PLE_CLAMP_MIN
        )
        # The short conv reads channel-major; a bare transpose is the certified form.
        convolved = dilated_short_conv_paged(
            x=torch.transpose(self._grouped_norm(gated, S.ple_nc), 0, 1),
            weight=S.ple_conv_w,
            bias=None,
            conv_states=conv_states,
            query_start_loc=qsl,
            cache_indices=si_all,
            has_initial_state=carry_all,
            dilation=core.ple_dilation,
            silu_activation=True,
            pad_slot_id=_PAD_SLOT_ID,
        )
        return torch.add(gated, torch.transpose(convolved, 0, 1))

    # -------------------------------------------------------------- forward ---

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs,
    ):
        core = self.core
        md = attn_metadata
        self._check_step_contract(md)
        step = self.step_args(md)
        spec_metadata = kwargs.get("spec_metadata")
        mm = md.mamba_metadata

        # Three views of one `[3, 1, T]` buffer (see PrefillTarget.forward): `pos`,
        # contiguous `pos3`, and `coords`.
        pos = torch.reshape(torch.split(position_ids, [1, _QSA_POSITION_AXES - 1], 0)[0], [-1])
        coords = torch.transpose(torch.reshape(position_ids, [_QSA_POSITION_AXES, -1]), 0, 1)
        pos3 = torch.empty(
            [_QSA_POSITION_AXES, position_ids.shape[-1]],
            dtype=torch.int32,
            device=position_ids.device,
        )
        pos3.copy_(torch.reshape(position_ids, [_QSA_POSITION_AXES, -1]))

        # Spliced vision rows are already the embedding; a lookup would overwrite them.
        if inputs_embeds is None:
            h = nn.functional.embedding(input_ids, core.w["embed"])
        else:
            assert inputs_embeds.shape[0] == input_ids.shape[0], (
                f"the spliced embedding carries {inputs_embeds.shape[0]} rows for "
                f"{input_ids.shape[0]} token ids"
            )
            assert inputs_embeds.shape[1] == core.hidden, (
                f"the spliced embedding is {inputs_embeds.shape[1]} wide, not {core.hidden}"
            )
            h = inputs_embeds
        num_tokens = h.shape[0]
        dt = h.dtype
        dev = h.device

        # The batch's bookkeeping, built once; every sequence is a generation sequence.
        batch = md.num_seqs
        # Host-int check (graph-safe); see PrefillTarget.forward.
        assert num_tokens == md.num_tokens, (
            f"the engine handed {num_tokens} activation rows for a batch its "
            f"metadata describes as {md.num_tokens} tokens "
            f"(padded_num_tokens={getattr(md, 'padded_num_tokens', None)}). A "
            "prefill/piecewise CUDA-graph backend pads the activation buffer "
            "past the batch and expects the model to slice it back; this "
            "target does not implement that configuration. Decode CUDA graphs "
            "pad with real dummy requests instead, so they are in the metadata "
            "and this assert is silent for them."
        )
        # Rows per generation sequence: 1 without speculation, else
        # `runtime_draft_len + 1`.
        gen_p = num_tokens // batch if batch else 1
        assert gen_p * batch == num_tokens, (
            f"{num_tokens} generation rows do not divide over {batch} generation "
            "sequences; a ragged per-sequence draft length can be expressed "
            "neither to the attention call nor to the multi-token recurrence"
        )
        assert core.mtp_enabled or gen_p == 1, (
            f"{num_tokens} generation rows over {batch} generation sequences with "
            "speculation off; the base path implements one query token per "
            "generation sequence"
        )
        # `gen_p > 1` is load-bearing for verify (see PrefillTarget.forward).
        if spec_metadata is not None and batch > 0:
            declared_p = spec_metadata.runtime_draft_len + 1
            assert gen_p == declared_p, (
                f"the batch carries {gen_p} query rows per generation sequence "
                f"but the worker's runtime_draft_len says {declared_p}; the "
                "per-step conv snapshots and the recurrent rollback buffer are "
                "sized from the second and written from the first"
            )
            assert 1 <= gen_p <= 4, (
                f"predicted_tokens_per_seq {gen_p} is outside the 1..4 range "
                f"attention/thop_attention certifies"
            )
        is_verify = spec_metadata is not None and batch > 0 and gen_p > 1
        step["predicted_tokens_per_seq"] = gen_p

        si_all = mm.state_indices[:batch]
        qsl_long = mm.query_start_loc_long
        int_idx = None
        if is_verify:
            # The runtime's arange over the slot pool (what `update_mamba_states` reads).
            assert core._bound.int_idx is not None, "the cache-manager bind has not run"
            int_idx = core._bound.int_idx[:batch]
        gdn_state = (si_all, qsl_long, is_verify, batch, gen_p, int_idx)

        # Every generation sequence carries its n-gram history; the engine's
        # `has_initial_states` covers the context prefix only (empty here).
        ple_carry = torch.ones(
            [batch], dtype=mm.has_initial_states.dtype, device=mm.has_initial_states.device
        )
        # PLE segmentation vector: rebuilt from `seq_lens_cuda` when `gen_p > 1` (see
        # PrefillTarget.forward).
        if gen_p == 1:
            ple_qsl = qsl_long
        else:
            ple_qsl = cu_seqlens_from_seq_lens(md.seq_lens_cuda[:batch], torch.int64)
        ple_state = (core._bound.ple_state[0], core._bound.ple_state[1], si_all, ple_carry, ple_qsl)

        # Dense/sparse decision once per forward, from a host-side metadata read: a
        # per-capture constant matching the engine's short/long decode graph families.
        sparse = batch > 0 and int(md.kv_lens_runtime[:batch].max()) > core.dense_threshold

        qsa = (
            md.qsa_req_idx_per_token[:num_tokens],
            md.qsa_logical_positions[:num_tokens],
            md.qsa_sequence_lengths[:num_tokens],
            md.qsa_visible_blocks[:num_tokens],
            md.qsa_topk_indices[:num_tokens],
            md.qsa_topk_row_starts[:num_tokens],
            md.qsa_block_table,
            md.tokens_per_block,
            pos,
            pos3,
            coords,
            sparse,
        )

        # Scratch shared by all attention layers (each fully rewrites them).
        qkv = torch.empty([num_tokens, core.qkv_width], dtype=dt, device=dev)
        index_scratch = torch.empty([num_tokens, core.index_scratch_width], dtype=dt, device=dev)
        attn_out = torch.empty([num_tokens, core.q_width], dtype=dt, device=dev)

        # The embedding repeated across the hc_count residual streams.
        hc_count = core.hc_count
        hc = torch.cat([h] * hc_count, -1)
        for i, L in enumerate(self._layers):
            if i == core.ple_layer:
                hc = torch.add(hc, self._ple_block(hc, input_ids, ple_state))
            mixed, inject = qwen4_exp_gated_residual(
                self._grouped_norm(hc, L.ahc_norm), L.ahc_down, L.ahc_up, L.ahc_inject, hc_count
            )
            if core.is_attn[i]:
                block_out = self._attention_block(
                    mixed, i, L, qkv, index_scratch, attn_out, qsa, step
                )
            else:
                block_out = self._gdn_block(mixed, i, L, gdn_state)
            hc = qwen4_exp_post_inject(hc, block_out, inject)
            mixed, inject = qwen4_exp_gated_residual(
                self._grouped_norm(hc, L.mhc_norm), L.mhc_down, L.mhc_up, L.mhc_inject, hc_count
            )
            hc = qwen4_exp_post_inject(hc, self._moe_block(mixed, L, dt), inject)
        # The stack-final mixer with no injection weight is the final norm.
        S = self._stack
        final, _ = qwen4_exp_gated_residual(
            self._grouped_norm(hc, S.fhc_norm), S.fhc_down, S.fhc_up, None, hc_count
        )
        if spec_metadata is None:
            return final
        # `lm_head` reads the `[T, hidden]` mix; the MTP module reads the
        # `[T, hc_count * hidden]` stream it was collapsed from.
        return final, hc

    # ------------------------------------------------------------- the draft ---

    def _mtp_moe_block(self, x, dt) -> torch.Tensor:
        """FP8 block-scale draft MoE; see `PrefillTarget._mtp_moe_block`."""
        core = self.core
        M = core._mtp
        logits = cublas_mm(x, M.router)
        topk_ids, topk_w = renorm_moe_routing_op(logits, topk=core.topk, output_dtype=dt)
        xq, xsf = fp8_quantize_1x128(x, use_ue8m0=_MTP_MOE_USE_UE8M0)
        routed = fp8_block_scale_moe_runner(
            routing_logits=None,
            routing_bias=None,
            hidden_states=xq,
            hidden_states_scale=xsf,
            gemm1_weights=M.fc1_w,
            gemm1_weights_scale=M.fc1_s,
            gemm2_weights=M.fc2_w,
            gemm2_weights_scale=M.fc2_s,
            num_experts=core.num_experts,
            top_k=core.topk,
            num_fused_shared_experts=None,
            n_group=None,
            topk_group=None,
            intermediate_size=core.moe_inter,
            local_expert_offset=0,
            local_num_experts=core.num_experts,
            routed_scaling_factor=None,
            routing_method_type=_ROUTING_METHOD_RENORM,
            topk_weights=topk_w,
            topk_ids=topk_ids,
            act_type=_ACT_TYPE_SWIGLU,
            gemm1_clamp_limit=None,
        )
        shared = cublas_mm(flashinfer_silu_and_mul(cublas_mm(x, M.sh_gu)), M.sh_dn)
        gate_logits = cublas_mm(x, M.sh_gate)
        return fused_sigmoid_gate_mul_add(routed, gate_logits, shared)

    def mtp_forward(
        self,
        embed_tokens: torch.Tensor,
        all_rank_num_tokens,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """One draft step over generation rows only (every step past the first in an
        engine forward, and the first when the batch had no context).

        `spec_metadata` and `all_rank_num_tokens` are accepted and unused."""
        core = self.core
        assert isinstance(attn_metadata, TrtllmAttentionMetadata)
        assert core._bound is not None, (
            "the trunk's cache-manager bind has not run; the shell calls the "
            "core before the worker, so this cannot be reached first"
        )
        md = attn_metadata
        rows = hidden_states.shape[0]
        assert input_ids.shape[0] == rows, (
            f"the draft step's {input_ids.shape[0]} token ids and {rows} "
            "hidden-state rows must describe the same tokens"
        )
        assert hidden_states.shape[1] == core.stream, (
            f"the MTP module reads the trunk's {core.stream}-wide four-stream "
            f"residual; the runtime handed it {hidden_states.shape[1]} columns"
        )
        dt = hidden_states.dtype
        dev = hidden_states.device

        # Projected on every call: the worker rewrites `attn_metadata` in place
        # between draft steps (see PrefillTarget.mtp_forward).
        step = self.step_args(md)
        M = core._mtp
        gen_seqs = md.num_seqs
        p = rows // gen_seqs if gen_seqs else 1
        assert p * gen_seqs == rows, (
            f"{rows} generation rows do not divide over {gen_seqs} "
            "generation sequences; a ragged per-sequence draft length cannot be "
            "expressed to the attention call"
        )

        # The KV cache written here is the draft manager's, swapped in by the worker
        # per call (see PrefillTarget.mtp_forward); take the layer index from the
        # current manager every time.
        cm = md.kv_cache_manager
        assert cm is not None, "a paged KV cache is required"
        offsets = cm.layer_offsets
        assert core.mtp_layer_idx in offsets, (
            f"the MTP module's attention is global layer index "
            f"{core.mtp_layer_idx}, which the cache manager "
            f"{type(cm).__name__} does not map (it maps {len(offsets)} layers, "
            f"is_draft={getattr(cm, 'is_draft', None)}); the engine allocated "
            "no KV layer for the draft module"
        )
        layer_idx = offsets[core.mtp_layer_idx]
        window = cm.max_attention_window_vec[layer_idx]
        assert window is None or window == md.max_seq_len, (
            f"the MTP layer's pool window is {window!r} but this call passes "
            f"max_seq_len {md.max_seq_len} as the attention window; a sliding "
            "window is not implemented here"
        )

        # Copy into a fresh contiguous `[3, N]`: a sliced view keeps the source row
        # stride and the op rejects it under `use_mrope`.
        assert position_ids.shape[0] == _QSA_POSITION_AXES, (
            f"expected the mRoPE position layout with {_QSA_POSITION_AXES} "
            f"coordinate rows, got {tuple(position_ids.shape)}"
        )
        pos3 = torch.empty(
            [_QSA_POSITION_AXES, position_ids.numel() // _QSA_POSITION_AXES],
            dtype=torch.int32,
            device=position_ids.device,
        )
        pos3.copy_(torch.reshape(position_ids, [_QSA_POSITION_AXES, -1]))

        # ---- the front end --------------------------------------------------
        e = nn.functional.embedding(input_ids, embed_tokens)
        x_e = cublas_mm(flashinfer_gemma_rmsnorm(e, M.pre_e_norm, core.eps), M.fc_e)
        # `fc_hidden` is one `[hidden, hidden]` matrix applied to each of the four
        # streams (the checkpoint's norm statistics say they are distinct streams).
        h_n = self._grouped_norm(hidden_states, M.pre_h_norm)
        x_h = torch.reshape(
            cublas_mm(torch.reshape(h_n, [rows * core.hc_count, core.hidden]), M.fc_h),
            [rows, core.stream],
        )
        if _MTP_EMBED_BLOCK_ALL_STREAMS:
            x = torch.add(x_h, torch.cat([x_e] * core.hc_count, -1))
        else:
            x = torch.add(
                x_h,
                nn.functional.pad(x_e, [0, core.stream - core.hidden], mode="constant", value=0.0),
            )

        # ---- the decoder layer ----------------------------------------------
        mixed, inject = qwen4_exp_gated_residual(
            self._grouped_norm(x, M.ahc_norm), M.ahc_down, M.ahc_up, M.ahc_inject, core.hc_count
        )
        qkv = torch.empty([rows, core.qkv_width], dtype=dt, device=dev)
        attn_out = torch.empty([rows, core.q_width], dtype=dt, device=dev)
        qg = cublas_mm(mixed, M.q)
        k = cublas_mm(mixed, M.k)
        v = cublas_mm(mixed, M.v)
        qkv_h = torch.reshape(qkv, [rows, core.heads_q + 2 * core.heads_kv, core.head_dim])
        q_dst, k_dst, v_dst = torch.split(qkv_h, [core.heads_q, core.heads_kv, core.heads_kv], 1)
        q_src, gate = torch.split(
            torch.reshape(qg, [rows, core.heads_q, 2 * core.head_dim]),
            [core.head_dim, core.head_dim],
            2,
        )
        q_dst.copy_(q_src)
        k_dst.copy_(torch.reshape(k, [rows, core.heads_kv, core.head_dim]))
        v_dst.copy_(torch.reshape(v, [rows, core.heads_kv, core.head_dim]))
        # The trunk rope's certified cell at the module's own base (`mtp.rope_theta`).
        fused_qk_norm_rope(
            qkv,
            num_heads_q=core.heads_q,
            num_heads_k=core.heads_kv,
            num_heads_v=core.heads_kv,
            head_dim=core.head_dim,
            rotary_dim=core.rotary_dim,
            eps=core.eps,
            q_weight=M.q_norm,
            k_weight=M.k_norm,
            base=core.mtp_theta,
            is_neox=True,
            position_ids=pos3,
            is_qk_norm=True,
            use_gemma=True,
            use_mrope=True,
            mrope_section1=core.mrope_section1,
            mrope_section2=core.mrope_section2,
        )
        # Dense attention; see `MTPLayer` for why the module's QSA indexer is unused.
        attn_out = thop_attention(
            q=qkv,
            output=attn_out,
            local_layer_idx=layer_idx,
            predicted_tokens_per_seq=p,
            num_heads=core.heads_q,
            num_kv_heads=core.heads_kv,
            head_size=core.head_dim,
            quant_mode=core.quant_mode,
            **step,
            **_CALL_CONSTANTS,
        )
        fused_sigmoid_mul_inplace(attn_out, gate)
        x = qwen4_exp_post_inject(x, cublas_mm(attn_out, M.o), inject)

        mixed, inject = qwen4_exp_gated_residual(
            self._grouped_norm(x, M.mhc_norm), M.mhc_down, M.mhc_up, M.mhc_inject, core.hc_count
        )
        return qwen4_exp_post_inject(x, self._mtp_moe_block(mixed, dt), inject)


class _VisionBlockWeights(NamedTuple):
    """One vision block's weights (`.t()` GEMM views). LayerNorm has a bias, unlike
    every norm in the language tower."""

    n1_w: torch.Tensor
    n1_b: torch.Tensor
    qkv: torch.Tensor
    qkv_b: torch.Tensor
    proj: torch.Tensor
    proj_b: torch.Tensor
    n2_w: torch.Tensor
    n2_b: torch.Tensor
    fc1: torch.Tensor
    fc1_b: torch.Tensor
    fc2: torch.Tensor
    fc2_b: torch.Tensor


class _VisionWeights(NamedTuple):
    """The tower's weights: patch embedding, position table, blocks and patch merger."""

    patch_w: torch.Tensor
    patch_b: torch.Tensor
    pos: torch.Tensor
    blocks: list
    m_n_w: torch.Tensor
    m_n_b: torch.Tensor
    m_fc1: torch.Tensor
    m_fc1_b: torch.Tensor
    m_fc2: torch.Tensor
    m_fc2_b: torch.Tensor


def _vision_views(core: Qwen4ExpTp1ModelingV2) -> _VisionWeights:
    w = core.w
    return _VisionWeights(
        # Not transposed: the entry takes the checkpoint's `[out, in, kt, kh, kw]`.
        patch_w=w["vis_patch_w"],
        patch_b=w["vis_patch_b"],
        pos=w["vis_pos"],
        blocks=[
            _VisionBlockWeights(
                n1_w=w[f"l{b}_vis_n1_w"],
                n1_b=w[f"l{b}_vis_n1_b"],
                # Row layout (q heads, k, v) is already thop_attention's packed form.
                qkv=w[f"l{b}_vis_qkv_w"].t(),
                qkv_b=w[f"l{b}_vis_qkv_b"],
                proj=w[f"l{b}_vis_proj_w"].t(),
                proj_b=w[f"l{b}_vis_proj_b"],
                n2_w=w[f"l{b}_vis_n2_w"],
                n2_b=w[f"l{b}_vis_n2_b"],
                fc1=w[f"l{b}_vis_fc1_w"].t(),
                fc1_b=w[f"l{b}_vis_fc1_b"],
                fc2=w[f"l{b}_vis_fc2_w"].t(),
                fc2_b=w[f"l{b}_vis_fc2_b"],
            )
            for b in range(core.v_depth)
        ],
        m_n_w=w["vis_m_n_w"],
        m_n_b=w["vis_m_n_b"],
        m_fc1=w["vis_m_fc1_w"].t(),
        m_fc1_b=w["vis_m_fc1_b"],
        m_fc2=w["vis_m_fc2_w"].t(),
        m_fc2_b=w["vis_m_fc2_b"],
    )


class VisionTower:
    """One request's images through the 27-block ViT and its patch merger.

    Not a target: runs only for a context request carrying media, from the shell's
    splice. Attention is bidirectional per frame, expressed to `thop_attention` as
    one context "sequence" per frame; running the rows as one sequence is 315x-582x
    outside tolerance. The position-embedding interpolation and 2-D rope must both
    emit in spatial-merge order (what the merger's 4-to-1 reshape consumes); a
    raster-order emitter gives a correctly shaped, silently wrong tower."""

    def __init__(self, core: Qwen4ExpTp1ModelingV2) -> None:
        self.core = core
        self._w = _vision_views(core)
        # Persistent scratch: `thop_attention` grows `workspace` in place; never
        # the engine's own workspace.
        self._ws: torch.Tensor | None = None

    def __call__(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """`pixel_values` is `[sum(t*h*w), in_channels*kt*kh*kw]`, one flat patch per
        row, channel-slowest. `grid_thw` is one `(frames, height, width)` row per
        image and stays on the host (no synchronization). Returns
        `[sum(t*h*w) / merge**2, out_hidden_size]`, one row per placeholder."""
        core = self.core
        dt, dev = pixel_values.dtype, pixel_values.device
        assert dt == core.w["embed"].dtype, (
            f"the tower runs at the activation dtype {core.w['embed'].dtype}; the "
            f"runtime handed pixel_values as {dt}. The input processor's `dtype` "
            "property is what decides this, and no cast happens here"
        )
        assert pixel_values.dim() == 2 and pixel_values.shape[1] == core.v_pixel_width, (
            f"pixel_values must be [patches, {core.v_pixel_width}] flat patch rows; "
            f"got {tuple(pixel_values.shape)}"
        )
        assert pixel_values.is_cuda, "pixel_values must reach the tower on the device"

        # ---- the frame segmentation, host-side -------------------------------
        # One attention segment per frame: a video's frames do not attend to each other.
        grid = grid_thw.tolist()
        seg_lens: list[int] = []
        for row in grid:
            assert len(row) == 3, f"grid_thw rows are (t, h, w); got {row}"
            frames, height, width = int(row[0]), int(row[1]), int(row[2])
            assert frames >= 1 and height >= 1 and width >= 1, row
            assert height % core.v_merge == 0 and width % core.v_merge == 0, (
                f"grid sides must divide by spatial_merge_size {core.v_merge}; got "
                f"height={height}, width={width}"
            )
            seg_lens.extend([height * width] * frames)
        num_patches = pixel_values.shape[0]
        assert sum(seg_lens) == num_patches, (
            f"grid_thw describes {sum(seg_lens)} patch rows and pixel_values carries {num_patches}"
        )
        num_segments = len(seg_lens)
        max_seg = max(seg_lens)

        # Batch-state tensors for the encoder-only call; they describe the images.
        host_lens = torch.tensor(seg_lens, dtype=torch.int32)
        host_types = torch.zeros([num_segments], dtype=torch.int32)
        host_total = torch.tensor([num_patches, 0], dtype=torch.int32)
        dev_lens = host_lens.to(dev)
        if self._ws is None:
            self._ws = torch.empty([0], dtype=torch.int8, device=dev)

        # ---- patch embedding + absolute position -----------------------------
        V = self._w
        h = torch.add(
            patch_embed_conv3d(pixel_values, V.patch_w, V.patch_b),
            pos_embed_interpolate(V.pos, grid, core.v_merge),
        )
        attn_out = torch.empty([num_patches, core.v_hidden], dtype=dt, device=dev)

        for B in V.blocks:
            qkv = cublas_mm(layer_norm(h, B.n1_w, B.n1_b, _VISION_NORM_EPS), B.qkv, B.qkv_b)
            qkv_h = torch.reshape(qkv, [num_patches, 3 * core.v_heads, core.v_head_dim])
            q_view, k_view, _ = torch.split(qkv_h, [core.v_heads] * 3, 1)
            q_rot, k_rot = vision_rope_2d(
                q_view, k_view, grid_thw, core.v_merge, theta=_VISION_ROPE_THETA
            )
            q_view.copy_(q_rot)
            k_view.copy_(k_rot)
            attn_out = thop_attention(
                q=qkv,
                output=attn_out,
                workspace=self._ws,
                sequence_length=dev_lens,
                host_past_key_value_lengths=host_lens,
                host_total_kv_lens=host_total,
                context_lengths=dev_lens,
                host_context_lengths=host_lens,
                host_request_types=host_types,
                max_num_requests=num_segments,
                max_context_length=max_seg,
                max_seq_len=max_seg,
                attention_window_size=max_seg,
                num_contexts=num_segments,
                num_ctx_tokens=num_patches,
                num_heads=core.v_heads,
                num_kv_heads=core.v_heads,
                head_size=core.v_head_dim,
                **_VISION_CALL_CONSTANTS,
            )
            h = torch.add(h, cublas_mm(attn_out, B.proj, B.proj_b))
            mlp = cublas_mm(layer_norm(h, B.n2_w, B.n2_b, _VISION_NORM_EPS), B.fc1, B.fc1_b)
            # Non-gated MLP with the tanh GELU (not the merger's erf).
            h = torch.add(h, cublas_mm(gelu_tanh(mlp), B.fc2, B.fc2_b))

        # ---- the patch merger ------------------------------------------------
        # The norm runs over `hidden` before the 4-to-1 shuffle; the reshape is legal
        # only because both order-bearing steps emitted in spatial-merge order.
        merged = torch.reshape(
            layer_norm(h, V.m_n_w, V.m_n_b, _VISION_NORM_EPS), [-1, core.v_merge_dim]
        )
        # The erf GELU (`nn.GELU()`); within one bf16 ulp of tanh, hence a separate entry.
        return cublas_mm(gelu_erf(cublas_mm(merged, V.m_fc1, V.m_fc1_b)), V.m_fc2, V.m_fc2_b)


class MTPLayer:
    """The checkpoint's MTP module, replayed once per draft step: one more decoder
    layer (two hyper-connections, gated GQA, 512-expert MoE) behind a two-branch
    front end (`fc_embedding` over the NEXT token's embedding, `fc_hidden` over the
    hidden state per stream). It returns `x` un-collapsed; `shared_head` owns the
    module's mixer. The step runs on the targets (`mtp_forward`) over `core._mtp`.

    The MoE is FP8 128x128 weight blocks (the trunk's is NVFP4); re-quantizing it
    would change the checkpoint's meaning, visible only as lower acceptance.

    Attention is dense, the one deliberate departure from the checkpoint (its QSA
    indexer stays unloaded): with a separate draft KV manager the worker swaps
    `attn_metadata.kv_cache_manager`, `QSAAttentionMetadata` implements no
    `prepare_for_draft_forward`, so `qsa_block_table` would decode the target
    manager's page offsets against the draft pool. Below `dense_seq_len_threshold`
    dense is bit-for-bit the trunk's.

    Not an `nn.Module`: the weights live in the core's `ParameterDict`, and a second
    registration would break the weight load's both-directions coverage assert."""

    def __init__(self, core: Qwen4ExpTp1ModelingV2, logits_processor) -> None:
        self.core = core
        self.logits_processor = logits_processor

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    def shared_head(
        self, hidden_states, lm_head, attn_metadata, return_context_logits
    ) -> torch.Tensor:
        """The module's own head: its stack-final hyper-connection mixer (a full gated
        mixer with its own parameters, not a plain RMSNorm), then the trunk's
        `lm_head` through the shell's `logits_processor`."""
        core = self.core
        M = core._mtp
        parts = torch.split(hidden_states, [core.hidden] * core.hc_count, 1)
        normed = torch.cat(
            [flashinfer_gemma_rmsnorm(part, w, core.eps) for w, part in zip(M.fhc_norm, parts)],
            1,
        )
        final, _ = qwen4_exp_gated_residual(normed, M.fhc_down, M.fhc_up, None, core.hc_count)
        return self.logits_processor.forward(final, lm_head, attn_metadata, return_context_logits)

    def forward(
        self,
        embed_tokens: torch.Tensor,
        all_rank_num_tokens,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """Route the draft step by the phase this call's metadata shows, per call: the
        worker rewrites `attn_metadata` between draft steps, so one engine step's
        draft calls can cross from prefill to decode."""
        return self.core._select_target(attn_metadata).mtp_forward(
            embed_tokens,
            all_rank_num_tokens,
            input_ids,
            position_ids,
            hidden_states,
            attn_metadata,
            **kwargs,
        )


class DraftModel:
    """The container the spec worker reaches the drafter through. The runtime
    inspects only `mtp_layers` (`[0]`: MTP-Eagle replays one layer),
    `embed_tokens`, `lm_head`, and `model.d2t` (correctly absent: shared
    vocabulary).

    `embed_tokens` resolves through the core on every read: this container is
    built while parameters are meta tensors, and materialization replaces the
    tensor objects, so a captured reference stays on meta."""

    def __init__(self, core: Qwen4ExpTp1ModelingV2, lm_head, logits_processor) -> None:
        self.core = core
        self.mtp_layers = [MTPLayer(core, logits_processor)]
        self.lm_head = lm_head

    @property
    def embed_tokens(self) -> torch.Tensor:
        return self.core.w["embed"]


class ModelingV2Qwen38FlashNextNvfp4Sm103Tp1(
    DecoderModelForCausalLM[Qwen4ExpTp1ModelingV2, PretrainedConfig]
):
    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config=None) -> str:
        """Prefer KV cache manager V2: PLE state (`ple_layer_cache`) and QSA's index and
        position planes exist only there. Under V1 the model builds and passes a
        smoke run with the PLE layer stateless. The engine resolves this from the
        built-in class; the declaration serves a direct instantiation."""
        return "V2"

    def __init__(self, model_config: ModelConfig):
        cfg = model_config.pretrained_config
        assert cfg is not None

        # Adopt the engine-normalized text tower (`Qwen4ExpTextConfig`) as the model's
        # config: everything that sizes the pools reads
        # `model_engine.model.model_config.pretrained_config`, and the composite
        # parent mirrors text geometry through a pass that silently skips keys the
        # parent already defaults. The swap is in place because the engine keeps its
        # own reference to this `ModelConfig`.
        text = getattr(cfg, "text_config", None)
        assert text is not None, (
            "this target is the language tower of the composite Qwen4-Exp "
            "export; pretrained_config carries no text_config"
        )
        assert getattr(text, "architectures", None) == ["Qwen4ExpForCausalLM"], (
            f"the text tower must be the normalized Qwen4-Exp text config; got "
            f"architectures {getattr(text, 'architectures', None)!r}"
        )
        assert text.torch_dtype is not None, "the text tower must declare a dtype"
        # Captured before the swap: the core needs the vision half and media token ids.
        composite = cfg
        model_config.pretrained_config = text

        super().__init__(
            Qwen4ExpTp1ModelingV2(model_config, composite),
            config=model_config,
            hidden_size=text.hidden_size,
            vocab_size=text.vocab_size,
        )
        # Media placeholder ids for the executor; built lazily (`__init__` runs under
        # meta init).
        self._mm_token_ids_value = (self.model.image_token_id, self.model.video_token_id)
        self._mm_token_ids_cache: torch.Tensor | None = None
        # `lm_head` is outside the quantization scheme: a dense bf16 `LMHead`; a
        # quantized head would register a different parameter set.
        assert self.lm_head.weight.dtype == torch.bfloat16, (
            f"expected a bf16 lm_head; the shell built {self.lm_head.weight.dtype}, "
            "which means the engine resolved a quantized output head for a "
            "checkpoint that stores it unquantized"
        )

        # The speculative branch, built only when `speculative_config` asked for it.
        # Written out rather than inherited: the in-tree one-engine shell builds its
        # drafter via a module-level function dispatching on `model_type` (still the
        # upstream family name), which would construct trtllm's own MTP layer.
        self.spec_config = getattr(model_config, "spec_config", None)
        self.draft_model = None
        self.spec_worker = None
        # No separate draft checkpoint; the loader reads this, so None is stated.
        self.draft_config = None
        if self.spec_config is not None:
            assert self.model.mtp_enabled, "core built without the MTP module"
            assert hasattr(self, "logits_processor"), (
                "the inherited shell no longer exposes logits_processor; the "
                "draft head and the shell's own gather both project through it"
            )
            self.draft_model = DraftModel(self.model, self.lm_head, self.logits_processor)
            # Must not be left at `get_spec_worker`'s default False: the resource
            # manager decides separately (`_should_create_separate_draft_kv_cache`) and
            # builds the target manager from target layer masks alone, so left False
            # the MTP layer is handed a manager with no `layer_offsets` entry for it.
            self.use_separate_draft_kv_cache = should_use_separate_draft_kv_cache(self.spec_config)
            self.spec_worker = get_spec_worker(
                self.spec_config,
                model_config,
                model_config.mapping,
                use_separate_draft_kv_cache=self.use_separate_draft_kv_cache,
            )
            assert self.spec_worker is not None, (
                f"the engine resolved no spec worker for spec_dec_mode "
                f"{self.spec_config.spec_dec_mode!r}"
            )
            # Caches the draft->target vocab offset map (correctly None here).
            self.spec_worker.set_draft_model(self.draft_model)

    # ----------------------------------------------------------- multimodal ---
    # Names the executor reads off a model class by `getattr`; each is silently
    # optional and its absence costs correctness rather than raising.

    @property
    def multimodal_data_device_paths(self) -> list[str]:
        """Tensors under ``multimodal_data`` the executor moves to CUDA. Absent, it moves
        everything, putting `image_grid_thw` on the device where
        `pos_embed_interpolate` and `vision_rope_2d` would each pay a D2H sync."""
        return [
            "image.pixel_values",
            "mrope_config.mrope_position_ids",
            "mrope_config.mrope_position_deltas",
        ]

    @property
    def multimodal_token_ids(self) -> torch.Tensor:
        """Placeholder ids whose rows the vision embeddings replace. Absent, the
        executor uses `input_ids >= vocab_size`, but this checkpoint's placeholders
        (248056, 248057) are in vocabulary, so nothing is selected and the engine
        generates fluent text while blind. Built on first read (meta init); not a
        parameter or buffer."""
        if self._mm_token_ids_cache is None:
            self._mm_token_ids_cache = torch.tensor(self._mm_token_ids_value, dtype=torch.int32)
        return self._mm_token_ids_cache

    def _multimodal_embeds(self, input_ids, kwargs):
        """Run the vision tower for this batch and splice its rows in; returns the
        `[T, hidden]` embedding buffer, or None when the step carries no media.

        Lives in the shell because the inherited `DecoderModelForCausalLM.forward`
        drops `multimodal_params` from `**kwargs` on the no-speculation path (a
        booting, fluent, blind engine). Host-side bookkeeping: `multimodal_params`
        is a subset of the context requests, so never zip it positionally;
        `multimodal_runtime` (chunked prefill / KV reuse only) gives cached and
        in-chunk row counts; `mm_token_indices` names every placeholder row."""
        mm_params = kwargs.get("multimodal_params") or []
        if not mm_params:
            return None
        core = self.model
        rows = []
        for params in mm_params:
            data = getattr(params, "multimodal_data", None) or {}
            image = data.get("image")
            assert data.get("video") is None, (
                "this target implements the image path only; a video request "
                "reached it. The tower itself is per-frame and would run, but "
                "the temporal half of the position bookkeeping is not derived "
                "here and a wrong one is invisible to every gate"
            )
            if image is None:
                continue
            pixel_values = image.get("pixel_values")
            grid_thw = image.get("image_grid_thw")
            assert pixel_values is not None and grid_thw is not None, (
                f"the multimodal payload carries {sorted(image)}; this target "
                "reads 'pixel_values' and 'image_grid_thw' from it"
            )
            assert not grid_thw.is_cuda, (
                "image_grid_thw must stay on the host; see multimodal_data_device_paths"
            )
            embeds = core.vision_tower(pixel_values, grid_thw)
            runtime = getattr(params, "multimodal_runtime", None)
            if runtime is not None:
                # Chunked prefill or reused prefix: splice only this chunk's slice.
                start = int(runtime.num_cached_mm_tokens)
                count = int(runtime.num_mm_tokens_in_chunk)
                total = embeds.shape[0]
                assert int(runtime.total_embeds_in_request) == total, (
                    f"the executor counted {runtime.total_embeds_in_request} "
                    f"placeholder rows for this request and the tower produced "
                    f"{total}; the two are derived from the same grid and a "
                    "disagreement means the merger's row count is wrong"
                )
                assert 0 <= start <= start + count <= total, (start, count, total)
                if count == 0:
                    continue
                embeds = embeds[start : start + count]
            rows.append(embeds)
        if not rows:
            return None
        vision = rows[0] if len(rows) == 1 else torch.cat(rows, 0)
        mm_token_indices = kwargs.get("mm_token_indices")
        assert mm_token_indices is not None, (
            "the executor shipped multimodal params but no mm_token_indices; "
            "that pairing is what `multimodal_token_ids` on this class buys, and "
            "without it there is no sync-free way to find the placeholder rows"
        )
        assert mm_token_indices.shape[0] == vision.shape[0], (
            f"the batch has {mm_token_indices.shape[0]} placeholder rows and the "
            f"tower produced {vision.shape[0]} embedding rows"
        )
        h = nn.functional.embedding(input_ids, core.w["embed"])
        # Indexed write: placeholder rows form several disjoint, data-dependent runs.
        h.index_copy_(0, mm_token_indices, vision)
        return h

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

        Without a spec worker this delegates to the base forward with every argument
        it was handed (`resource_manager` stays in `**kwargs` deliberately: naming
        it would drop it from what the base receives). With one, the shell owns the
        logits gather (one-model modes set `without_logits`) and the worker needs
        ungathered hidden states; `position_ids` is passed in the engine's layout
        (flattening gives silently wrong draft positions). The vision splice happens
        here because the base forward would not pass `multimodal_params` through."""
        spliced = self._multimodal_embeds(input_ids, kwargs)
        if spliced is not None:
            assert inputs_embeds is None, (
                "the engine supplied inputs_embeds and this step also carries "
                "multimodal params; this target owns the splice and cannot "
                "reconcile two embedding sources"
            )
            inputs_embeds = spliced
        if self.spec_worker is None:
            assert spec_metadata is None, (
                "spec_metadata arrived without a spec worker; this engine was "
                "built without a speculative_config"
            )
            return super().forward(
                attn_metadata,
                input_ids,
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
        # `final` is the `[T, hidden]` mix for `lm_head`, `hc` the stream the MTP reads.
        final, hc = self.model(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            lora_params=lora_params,
            spec_metadata=spec_metadata,
            **kwargs,
        )
        # `gather_ids`: one row per context request, `runtime_draft_len + 1` per
        # generation request.
        logits = self.logits_processor.forward(
            nn.functional.embedding(spec_metadata.gather_ids, final),
            self.lm_head,
            attn_metadata,
            True,
        )
        return self.spec_worker(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hc,
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
