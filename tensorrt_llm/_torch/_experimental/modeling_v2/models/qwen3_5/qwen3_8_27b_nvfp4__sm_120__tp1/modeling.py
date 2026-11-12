# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: Qwen3.8-27B-NVFP4 / sm_120 / tp1 — self-contained modeling code.

A dense Qwen3.5 Gated-DeltaNet hybrid: 64 decoder layers, every fourth a gated
full-attention layer (24 q / 4 kv heads of 256, a sigmoid output gate interleaved with q
per head, interleaved mRoPE over a quarter of each head), the rest Gated-DeltaNet layers
(16 key / 48 value heads of 128, a width-4 causal conv), each followed by a dense SwiGLU
MLP of 17408. The core holds the weights `weights.py` declares and the copies it derives;
`PrefillTarget` (any batch with context rows, run eagerly) and `DecodeTarget`
(generation-only, captured in the engine's decode CUDA graphs) run a step over it,
passing every argument to every op at the call site.

**Every projection is routed by the step's row count** (the CUDA-graph-padded batch at
decode, all tokens of a prefill or mixed step). The FP8 checkpoint projections run on
load-time NVFP4 copies (`weights.derive`): a CUDA-core W4A16 GEMV at the fewest rows, a
W4A16 Marlin GEMM (BF16 split-K reduce up to 16 rows, FP32 above) up to their cap, and a
W4A4 GEMM above it, whose activation the producer quantizes in its own epilogue where it
can (the GDN gated norm, the attention output gate, the post-attention norm). The NVFP4
MLP runs a SwiGLU-epilogue gate_up GEMM from 41 rows. The thresholds are the
`_ROWS` constants below; each one is a measured crossover on this GPU (188 SMs), not a
capacity.

**Engine settings the target serves** (`routing.py` declines others): an fp8-e4m3 paged
KV cache (`kv_cache_config.dtype='fp8'`) and the quantized recurrent-state pool
(`kv_cache_config.mamba_ssm_cache_dtype='int8'`: per slot `[48, 260, 128]` int8 bytes, 128
rows of int16 codes (two byte rows each) plus four rows of fp32 row scales), on the V2 hybrid
cache manager; no speculative decoding; eager prefill steps.

**Attention** runs on the target's own op (`ops/attention/attention_sm120`, `csrc/`): the
engine's KV-cache append, then the target's context FMHA (base-2 softmax, last q-tile KV
split, and on the final layer of a prefill step the leading q-tiles skipped when only each
context's last row is read) and its XQA decode kernel (multi-block count capped at one CTA
wave, shared-prefix (cascade) pass from batch 8 with FP8 prefix math).

**The final layer of a long prefill step** (eager, at least 2048 rows, some context rows)
runs its output gate, o_proj, post-attention norm, MLP and final norm only on the rows the
logits processor reads; every op on those rows routes and tunes as the full step, so the
rows are the ones an unsliced step computes.

Numerics relative to the built-in model: the FP8 projections run NVFP4 copies of their
weights; attention reads an fp8 KV cache; the recurrent state is int16 rows; the context
softmax is base-2 `ex2.approx`; the cascade prefix pass runs FP8 Q/P; the GDN decode
normalizes q/k as `x / (sqrt(sum x^2) + 1e-6)`. The accuracy gate in
`tests/integration/defs/accuracy/modeling_v2/test_qwen3_5.py` certifies the whole.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import step_contract_enabled
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.activation.fused_sigmoid_mul_inplace import (  # noqa: E501
    fused_sigmoid_mul_inplace,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.norm.flashinfer_gemma_rmsnorm import (  # noqa: E501
    flashinfer_gemma_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.norm.rms_norm_gated_token_major import (  # noqa: E501
    rms_norm_gated_token_major,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.ssm.reset_recurrent_state_rows import (  # noqa: E501
    reset_recurrent_state_rows,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata, RopeParams
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel

from . import weights as _weights
from .ops.activation.sigmoid_mul_nvfp4 import sigmoid_mul_nvfp4
from .ops.activation.silu_mul_nvfp4_permuted import silu_mul_nvfp4_permuted
from .ops.attention.attention_sm120 import AttentionSm120Buffers, attention_sm120
from .ops.attention.qkv_gemma_rmsnorm_rope_gate import qkv_gemma_rmsnorm_rope_gate
from .ops.embedding.masked_embedding_gather import masked_embedding_gather
from .ops.gemm.fp8_per_tensor_mm import fp8_per_tensor_mm
from .ops.gemm.marlin_nvfp4_w4a16 import marlin_nvfp4_w4a16
from .ops.gemm.nvfp4_gate_up_swiglu_nvfp4 import nvfp4_gate_up_swiglu_nvfp4
from .ops.gemm.nvfp4_gemm_sm120 import nvfp4_gemm_sm120
from .ops.gemm.nvfp4_w4a16_gemv import nvfp4_w4a16_gemv
from .ops.gemm.nvfp4_w4a16_gemv_conv import nvfp4_w4a16_gemv_conv
from .ops.gemm.small_n_bf16_gemv import small_n_bf16_gemv
from .ops.norm.flashinfer_gemma_fused_add_rmsnorm import flashinfer_gemma_fused_add_rmsnorm
from .ops.norm.gemma_add_rmsnorm_1row import gemma_add_rmsnorm_1row
from .ops.norm.gemma_add_rmsnorm_early_trigger import gemma_add_rmsnorm_early_trigger
from .ops.norm.gemma_add_rmsnorm_nvfp4 import gemma_add_rmsnorm_nvfp4
from .ops.norm.rms_norm_gated_nvfp4 import rms_norm_gated_nvfp4
from .ops.ssm.causal_conv1d_update import causal_conv1d_update
from .ops.ssm.flashinfer_chunk_gated_delta_rule_q8 import flashinfer_chunk_gated_delta_rule_q8
from .ops.ssm.gdn_decode_t1 import gdn_decode_t1
from .ops.ssm.gdn_decode_t1_gated_norm import gdn_decode_t1_gated_norm
from .ops.ssm.gdn_prefill_onepass import gdn_prefill_onepass

_SM = (12, 0)

# ----------------------------------------------------------------- routing ---
# Each `*_ROWS` constant is a dispatch threshold over the step's row count, measured on
# this GPU; none is a buffer size.


class _ProjectionRows(NamedTuple):
    """Up to `gemv` rows: the CUDA-core W4A16 GEMV; up to `marlin` rows: W4A16 Marlin;
    above: the W4A4 copy."""

    gemv: int
    marlin: int


_PROJECTION_ROWS = {
    "qkvz": _ProjectionRows(gemv=3, marlin=32),
    "out": _ProjectionRows(gemv=3, marlin=32),
    "qkv": _ProjectionRows(gemv=2, marlin=64),
    "o": _ProjectionRows(gemv=3, marlin=32),
}
#: Marlin reduces its split-K partials in FP32 (each output rounded once) from here.
_MARLIN_FP32_REDUCE_MIN_ROWS = 17
#: The target Marlin GEMM's launch tables for this checkpoint's projection shapes.
_MARLIN_LAUNCH_HINT = 1
#: A pure-decode in_proj_qkvz GEMV also runs the conv step in its epilogue up to here.
_GDN_CONV_FUSED_MAX_ROWS = 3
#: The decode recurrence with the gated RMS norm folded in, rows [min, max].
_T1_NORM_MIN_ROWS, _T1_NORM_MAX_ROWS = 5, 32
#: in_proj_ba (N = 96) on the small-N GEMV up to here, cuBLAS above.
_BA_GEMV_MAX_ROWS = 16
#: Eager steps of at least this many rows run in_proj_ba on the aux stream beside qkvz.
_BA_FORK_MIN_ROWS = 2048
#: gate_up as the SwiGLU-epilogue GEMM from here; its tile by rows below.
_SWIGLU_EPILOGUE_MIN_ROWS = 41
#: The post-attention norm emits gate_up's NVFP4 input at these rows.
_POST_NORM_NVFP4_ROWS = ((33, 112), (129, None))
#: The one-row-per-CTA gemma add+RMSNorm at these rows.
_ONE_ROW_NORM_ROWS = (16, 128)
#: The next layer's input norm triggers its dependents early up to here.
_EARLY_TRIGGER_MAX_ROWS = 32
#: lm_head on its checkpoint W4A4 (FP32 logits) from here, Marlin W4A16 below.
_LM_HEAD_W4A4_MIN_ROWS = 56
#: The final layer runs everything after attention on the read rows only from here.
_ROW_SLICE_MIN_ROWS = 2048
#: The context FMHA's q-tile height (the skipped-leading-tiles predicate counts them).
_FMHA_Q_TILE = 64
#: The shared-prefix (cascade) XQA decode from this batch size.
_XQA_SHARED_PREFIX_MIN_BATCH = 8

#: Recurrent-state pool slot id the kernels skip.
_PAD_SLOT_ID = -1
#: NVFP4 elements per scale factor.
_SF_VEC = 16

#: q_scaling of the attention call (the softmax scale is 1 / (sqrt(head_size) * q_scaling)).
_Q_SCALING = 1.0


_MAMBA_FIELDS = (
    "state_indices",
    "has_initial_states",
    "has_initial_states_cpu",
    "query_start_loc",
    "query_start_loc_long",
)


def _in_rows(rows: int, ranges) -> bool:
    return any(rows >= lo and (hi is None or rows <= hi) for lo, hi in ranges)


def _swiglu_tile(rows: int) -> int:
    """The SwiGLU-epilogue GEMM's tile: 128x128x256 over the lower half of M up to 64
    rows (2), 128x128x256 up to 447 (1), 256x128x128 from 448 (0)."""
    if rows <= 64:
        return 2
    return 1 if rows < 448 else 0


def _step_args(md: TrtllmAttentionMetadata, num_contexts: int, num_ctx_tokens: int) -> dict:
    """The attention op's batch state: engine-owned persistent buffers refreshed in place,
    and the step's host ints."""
    return dict(
        num_contexts=num_contexts,
        num_ctx_tokens=num_ctx_tokens,
        sequence_length=md.kv_lens_cuda_runtime,
        host_past_key_value_lengths=md.kv_lens_runtime,
        context_lengths=md.prompt_lens_cuda_runtime,
        host_context_lengths=md.prompt_lens_cpu_runtime,
        kv_cache_block_offsets=md.kv_cache_block_offsets,
        host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
        tokens_per_block=md.tokens_per_block,
        # No sliding window: the full admitted length on every attention layer.
        max_seq_len=md.max_seq_len,
    )


class _Bound(NamedTuple):
    """One KV cache manager's views: attention pool offsets by layer and the GDN pools."""

    cm: object
    pool_idx: dict
    conv: dict
    ssm: dict
    attn_buffers: AttentionSm120Buffers


class _Layer(NamedTuple):
    """One decoder layer's weight views and scalars (`w` is `core.w`, `c` `core.copies`)."""

    i: int
    is_attn: bool
    in_norm: torch.Tensor
    post_norm: torch.Tensor
    next_norm: torch.Tensor  # the next layer's input norm, or the final norm
    is_last: bool


# --------------------------------------------------------------- the core ---


class Qwen3827BTp1ModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig, composite_config: PretrainedConfig):
        """`composite_config` is the checkpoint's composite config; the text tower adopted
        as `model_config.pretrained_config` carries no vision half."""
        super().__init__(model_config)
        cfg = model_config.pretrained_config

        # This target IS this geometry — assert, never adapt.
        assert torch.cuda.get_device_capability() == _SM, (
            f"target certified on sm_{_SM[0]}{_SM[1]}, running on "
            f"sm_{''.join(map(str, torch.cuda.get_device_capability()))}"
        )
        mapping = model_config.mapping
        assert mapping.tp_size == 1 and mapping.pp_size == 1, "tp1 target"
        assert not mapping.enable_attention_dp, "tp1 target"
        assert model_config.torch_dtype == torch.bfloat16, model_config.torch_dtype
        assert getattr(model_config, "spec_config", None) is None, (
            "this target implements no speculative decoding"
        )
        assert not cfg.tie_word_embeddings, "untied lm_head"
        assert not cfg.attention_bias, "no q/k/v/o bias"
        assert cfg.hidden_act == "silu", "SwiGLU over a silu gate"
        assert getattr(cfg, "attn_output_gate", True), "the gated-attention assembly"

        self.num_layers = cfg.num_hidden_layers
        self.hidden = cfg.hidden_size
        self.vocab = cfg.vocab_size
        self.eps = cfg.rms_norm_eps

        layer_types = list(cfg.layer_types)
        assert len(layer_types) == self.num_layers
        assert set(layer_types) <= {"full_attention", "linear_attention"}, set(layer_types)
        self.is_attn = [t == "full_attention" for t in layer_types]
        interval = cfg.full_attention_interval
        assert self.is_attn == [i % interval == interval - 1 for i in range(self.num_layers)], (
            "layer_types disagrees with full_attention_interval"
        )
        assert self.is_attn[-1], "the final layer is a full-attention layer"
        self.attn_layers = [i for i, a in enumerate(self.is_attn) if a]
        self.gdn_layers = [i for i, a in enumerate(self.is_attn) if not a]

        # ---- gated attention --------------------------------------------
        self.heads_q = cfg.num_attention_heads
        self.heads_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_width = self.heads_q * self.head_dim
        self.kv_width = self.heads_kv * self.head_dim
        # q_proj interleaves q and the output gate per head.
        self.qkv_rows = 2 * self.q_width + 2 * self.kv_width

        # ---- RoPE: interleaved mRoPE over a quarter of each head ---------
        rope = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None)
        assert isinstance(rope, dict), rope
        self.mrope_section = tuple(rope["mrope_section"])
        assert rope.get("mrope_interleaved", False), "interleaved mRoPE"
        partial = float(rope.get("partial_rotary_factor", cfg.partial_rotary_factor))
        self.rotary_dim = int(self.head_dim * partial)
        assert sum(self.mrope_section) * 2 == self.rotary_dim, (self.mrope_section, self.rotary_dim)
        self._rope_params = _rope_params(cfg)

        # ---- Gated-DeltaNet ---------------------------------------------
        self.num_k_heads = cfg.linear_num_key_heads
        self.num_v_heads = cfg.linear_num_value_heads
        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        assert self.head_k_dim == self.head_v_dim == 128, "the T1 kernels' head width"
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel = cfg.linear_conv_kernel_dim
        assert self.conv_kernel == 4, "the fused conv steps' width"

        self.inter = cfg.intermediate_size

        self.w = _weights.MODEL_WEIGHTS.declare(_weights.MODEL_WEIGHTS.dims(self))
        self.copies: dict | None = None
        self.vision_config = getattr(composite_config, "vision_config", None)

        # in_proj_ba runs beside in_proj_qkvz on its own stream.
        self.aux_stream = torch.cuda.Stream()
        self.events = (torch.cuda.Event(), torch.cuda.Event())

        self._bound: _Bound | None = None
        self._binds = 0
        self._prefill: PrefillTarget | None = None
        self._decode: DecodeTarget | None = None
        self._step_begun = False
        self._cos_sin: torch.Tensor | None = None
        self.kv_scale_orig_quant: torch.Tensor | None = None
        self.kv_scale_quant_orig: torch.Tensor | None = None
        self._derived = False

    def post_load_weights(self) -> None:
        """Derive what the targets read, then build them, now that the weights are real.

        Runs once: the loader calls every module's hook and the shell calls this one too,
        and `derive` frees the FP8 bytes it reads."""
        if self._derived:
            return
        self._derived = True
        _weights.MODEL_WEIGHTS.derive(self)
        device = self.w["embed"].device
        self._cos_sin = (
            self._rope_params.create_rope_const_params(interleave=False)[1]
            .reshape(self._rope_params.max_positions, 2, -1)
            .to(device)
        )
        self.kv_scale_quant_orig = torch.ones(1, dtype=torch.float32, device=device)
        self.kv_scale_orig_quant = 1.0 / self.kv_scale_quant_orig
        layers = []
        for i in range(self.num_layers):
            last = i == self.num_layers - 1
            layers.append(
                _Layer(
                    i=i,
                    is_attn=self.is_attn[i],
                    in_norm=self.w[f"l{i}_in_norm"],
                    post_norm=self.w[f"l{i}_post_norm"],
                    next_norm=self.w["norm"] if last else self.w[f"l{i + 1}_in_norm"],
                    is_last=last,
                )
            )
        self.layers_view = layers
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

    # ------------------------------------------------------------ binding ---

    def _bind_cache_manager(self, md) -> None:
        """Bind to one KV cache manager: check the contract, take every pool view.

        Called once per manager: capacity estimation builds a throwaway manager and then
        the real one on the same module, and a view held across the swap dangles."""
        cm = md.kv_cache_manager
        assert cm is not None, "a paged KV cache is required"
        assert not (self._binds and torch.cuda.is_current_stream_capturing()), (
            "the KV cache manager changed during CUDA graph capture"
        )
        self._binds += 1
        pool_idx = {i: cm.layer_offsets[i] for i in self.attn_layers}
        windows = cm.max_attention_window_vec
        assert {windows[off] for off in pool_idx.values()} <= {None, md.max_seq_len}, (
            "a sliding window is not implemented here"
        )
        mm = md.mamba_metadata
        assert mm is not None and mm is not False, (
            "attn_metadata.mamba_metadata is absent: no recurrent-state pool was allocated"
        )
        missing = [name for name in _MAMBA_FIELDS if not hasattr(mm, name)]
        assert not missing, f"mamba metadata fields missing: {missing}"
        for name in ("get_conv_states", "get_ssm_states", "layer_offsets"):
            assert hasattr(cm, name), f"{type(cm).__name__} has no {name}"
        conv = {i: cm.get_conv_states(i) for i in self.gdn_layers}
        ssm = {i: cm.get_ssm_states(i) for i in self.gdn_layers}
        ssm0 = ssm[self.gdn_layers[0]]
        assert ssm0.dtype == torch.int8 and tuple(ssm0.shape[1:]) == (
            self.num_v_heads,
            2 * self.head_v_dim + 4 * self.head_v_dim // self.head_k_dim,
            self.head_k_dim,
        ), (
            f"the recurrent pool is {ssm0.dtype} {tuple(ssm0.shape)}; this target reads the "
            "int16-row layout in an int8-typed pool (mamba_ssm_cache_dtype='int8')"
        )
        conv0 = conv[self.gdn_layers[0]]
        assert conv0.dtype == torch.bfloat16 and tuple(conv0.shape[1:]) == (
            self.conv_dim,
            self.conv_kernel - 1,
        ), (conv0.dtype, tuple(conv0.shape))
        assert md.effective_beam_width == 1 and md.cache_indirection is None, (
            "beam search is not implemented"
        )
        assert not md.is_cross, "cross attention is not implemented"
        attn_buffers = AttentionSm120Buffers(
            max_tokens=md.max_num_tokens,
            max_seqs=md.max_num_sequences or md.max_num_requests,
            device=self.w["embed"].device,
        )
        self._bound = _Bound(cm, pool_idx, conv, ssm, attn_buffers)

    def _select_target(self, attn_metadata) -> "PrefillTarget | DecodeTarget":
        return self._decode if attn_metadata.num_contexts == 0 else self._prefill

    def begin_step(
        self,
        attn_metadata,
        input_ids,
        position_ids,
        inputs_embeds,
        lora_params,
        return_context_logits,
    ) -> None:
        """Check what holds for every step, bind the cache manager, choose the target and
        let it prepare the step's host-side state."""
        assert self._prefill is not None and self._decode is not None, (
            "load_weights must run before forward"
        )
        assert isinstance(attn_metadata, TrtllmAttentionMetadata)
        assert lora_params is None, "LoRA is not implemented by this target"
        assert position_ids is not None and position_ids.dim() == 3, (
            "the mRoPE position layout [3, 1, T] is required"
        )
        assert (input_ids is None) != (inputs_embeds is None), (
            "exactly one of input_ids and inputs_embeds"
        )
        if self._bound is None or attn_metadata.kv_cache_manager is not self._bound.cm:
            self._bind_cache_manager(attn_metadata)
        self._select_target(attn_metadata).prepare(attn_metadata, return_context_logits)
        self._step_begun = True

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids=None,
        position_ids=None,
        inputs_embeds=None,
        **kwargs,
    ):
        assert self._step_begun, "the shell calls begin_step before every forward"
        self._step_begun = False
        return self._select_target(attn_metadata).forward(
            attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )


def _rope_params(cfg) -> RopeParams:
    """The rope constants the built-in Qwen3.5 attention builds: `RopeParams.from_config`
    on the text config with its scaling marked mRoPE (as the composite model does)."""
    import copy

    text = copy.deepcopy(cfg)
    scaling = dict(getattr(text, "rope_scaling", None) or getattr(text, "rope_parameters", {}))
    scaling["type"] = "mrope"
    text.rope_scaling = scaling
    return RopeParams.from_config(text)


# --------------------------------------------------- shared block pieces ---


def _project(core, tag: str, i: int, x: torch.Tensor, rows: int) -> torch.Tensor:
    """One FP8-checkpoint projection over bf16 `x`, routed by the step's `rows`."""
    route = _PROJECTION_ROWS[tag]
    copies = core.copies
    if rows <= route.gemv:
        g = copies[f"l{i}_{tag}_gv"]
        return nvfp4_w4a16_gemv(x, g.weight, g.weight_sf, g.global_scale)
    if rows <= route.marlin:
        return _marlin(copies[f"l{i}_{tag}_mw"], x, rows)
    a4 = copies[f"l{i}_{tag}_a4"]
    xq, xsf = fp4_quantize(x, a4.act_global, _SF_VEC, False)
    return _w4a4(a4, xq, xsf, rows, torch.bfloat16)


def _marlin(m, x: torch.Tensor, rows: int, launch_hint: int = _MARLIN_LAUNCH_HINT) -> torch.Tensor:
    mode = 2 if rows >= _MARLIN_FP32_REDUCE_MIN_ROWS else 0
    return marlin_nvfp4_w4a16(
        x, m.weight, m.weight_scale, m.weight_global_scale, m.size_n, m.size_k, mode, launch_hint
    )


def _w4a4(a4, xq: torch.Tensor, xsf: torch.Tensor, rows: int, out_dtype) -> torch.Tensor:
    return nvfp4_gemm_sm120(xq, a4.weight, xsf, a4.weight_sf, a4.alpha, out_dtype, rows)


def _qkv(core, i: int, x: torch.Tensor, rows: int) -> torch.Tensor:
    """The attention input projection. Above Marlin's cap it is the W4A4 copy over every
    row of the weight, with the K/V columns (the KV-cache inputs) recomputed in FP8 from
    the checkpoint's own K/V rows."""
    if rows <= _PROJECTION_ROWS["qkv"].marlin:
        return _project(core, "qkv", i, x, rows)
    w = core.w
    out = _project(core, "qkv", i, x, rows)
    kv = fp8_per_tensor_mm(x, w[f"l{i}_qkv_w"], w[f"l{i}_qkv_is"], w[f"l{i}_qkv_ws"])
    out[:, 2 * core.q_width :].copy_(kv)
    return out


def _post_attention_norm(core, L: _Layer, x, residual, rows: int):
    """The post-attention gemma add+RMSNorm: `(h, residual)` where `h` is gate_up's input,
    NVFP4 `(fp4, sf)` where the norm's epilogue quantizes it, else bf16."""
    if _in_rows(rows, _POST_NORM_NVFP4_ROWS):
        fp4, sf, residual = gemma_add_rmsnorm_nvfp4(
            x, residual, L.post_norm, core.eps, core.w[f"l{L.i}_gate_up_is"]
        )
        return (fp4, sf), residual
    if _in_rows(rows, (_ONE_ROW_NORM_ROWS,)):
        return gemma_add_rmsnorm_1row(x, residual, L.post_norm, core.eps)
    return flashinfer_gemma_fused_add_rmsnorm(x, residual, L.post_norm, core.eps)


def _next_norm(core, L: _Layer, x, residual, rows: int):
    """The next layer's input norm (or the final norm after the last layer). The final norm
    routes by the rows it actually normalizes: on a row-sliced final layer, the read rows."""
    if L.is_last:
        rows = x.shape[0]
    elif rows <= _EARLY_TRIGGER_MAX_ROWS:
        return gemma_add_rmsnorm_early_trigger(x, residual, L.next_norm, core.eps)
    lo = _ONE_ROW_NORM_ROWS[0] if L.is_last else _EARLY_TRIGGER_MAX_ROWS + 1
    if lo <= rows <= _ONE_ROW_NORM_ROWS[1]:
        return gemma_add_rmsnorm_1row(x, residual, L.next_norm, core.eps)
    return flashinfer_gemma_fused_add_rmsnorm(x, residual, L.next_norm, core.eps)


def _mlp(core, i: int, h, rows: int) -> torch.Tensor:
    """The dense SwiGLU MLP. `h` is bf16 or NVFP4 `(fp4, sf)` from the norm's epilogue.
    gate_up rows are permuted at load for the SwiGLU-epilogue GEMM, which writes down's
    NVFP4 input; below its threshold (and while the AutoTuner tunes, so gate_up's buckets
    get tuned) gate_up runs alone and the permuted SwiGLU kernel quantizes."""
    w = core.w
    if isinstance(h, tuple):
        xq, xsf = h
    else:
        xq, xsf = fp4_quantize(h, w[f"l{i}_gate_up_is"], _SF_VEC, False)
    if rows >= _SWIGLU_EPILOGUE_MIN_ROWS and not AutoTuner.get().is_tuning_mode:
        dq, dsf = nvfp4_gate_up_swiglu_nvfp4(
            xq,
            w[f"l{i}_gate_up_w"],
            xsf,
            w[f"l{i}_gate_up_sf"],
            w[f"l{i}_gate_up_alpha"],
            w[f"l{i}_down_is"],
            _swiglu_tile(rows),
        )
    else:
        gate_up = nvfp4_gemm_sm120(
            xq,
            w[f"l{i}_gate_up_w"],
            xsf,
            w[f"l{i}_gate_up_sf"],
            w[f"l{i}_gate_up_alpha"],
            torch.bfloat16,
            rows,
        )
        dq, dsf = silu_mul_nvfp4_permuted(gate_up, w[f"l{i}_down_is"])
    return nvfp4_gemm_sm120(
        dq, w[f"l{i}_down_w"], dsf, w[f"l{i}_down_sf"], w[f"l{i}_down_alpha"], torch.bfloat16, rows
    )


def _attention_out(core, L: _Layer, attn_out, gate, rows: int) -> torch.Tensor:
    """The sigmoid output gate and o_proj. Where o_proj takes its W4A4 copy, the gate
    kernel quantizes for it."""
    i = L.i
    if rows > _PROJECTION_ROWS["o"].marlin:
        a4 = core.copies[f"l{i}_o_a4"]
        fp4, sf = sigmoid_mul_nvfp4(attn_out, gate, a4.act_global)
        return _w4a4(a4, fp4, sf, rows, torch.bfloat16)
    fused_sigmoid_mul_inplace(attn_out, gate)
    return _project(core, "o", i, attn_out, rows)


def _gdn_out(core, i: int, o, z, rows: int) -> torch.Tensor:
    """The gated RMS norm (norm before a SiLU gate) and out_proj. Where out_proj takes its
    W4A4 copy, the norm quantizes for it."""
    w = core.w
    o = o.reshape(-1, core.head_v_dim)
    if rows > _PROJECTION_ROWS["out"].marlin:
        a4 = core.copies[f"l{i}_out_a4"]
        fp4, sf = rms_norm_gated_nvfp4(
            o, z, w[f"l{i}_gnorm"], core.eps, a4.act_global, get_env_enable_pdl()
        )
        return _w4a4(a4, fp4, sf, rows, torch.bfloat16)
    y = rms_norm_gated_token_major(o, z, w[f"l{i}_gnorm"], core.eps)
    return _project(core, "out", i, y.view(-1, core.value_dim), rows)


def _ba(core, i: int, x: torch.Tensor, rows: int) -> torch.Tensor:
    """in_proj_ba (`[b ; a]`, N = 96) over bf16 `x`."""
    w = core.w[f"l{i}_ba_w"]
    if rows <= _BA_GEMV_MAX_ROWS:
        return small_n_bf16_gemv(x, w)
    return cublas_mm(x, w.t())


def _split_gdn(core, proj: torch.Tensor, ba: torch.Tensor):
    """q|k|v, z, a and b as column views: `in_proj_qkvz` is `[q | k | v | z]` and
    `in_proj_ba` `[b | a]`."""
    num_tokens = proj.shape[0]
    mixed_qkv = proj[:, : core.conv_dim]
    z = proj[:, core.conv_dim :].view(num_tokens, core.num_v_heads, core.head_v_dim)
    b = ba[:, : core.num_v_heads]
    a = ba[:, core.num_v_heads :]
    return mixed_qkv, z, a, b


def _qkv_views(core, mixed_qkv: torch.Tensor):
    """q, k `[1, T, H, K]` and v `[1, T, HV, V]`: views of the conv output's columns, heads
    packed, the projection's row stride kept."""
    t = mixed_qkv.shape[0]
    q, k, v = torch.split(mixed_qkv, [core.key_dim, core.key_dim, core.value_dim], 1)
    return (
        q.view(1, t, core.num_k_heads, core.head_k_dim),
        k.view(1, t, core.num_k_heads, core.head_k_dim),
        v.view(1, t, core.num_v_heads, core.head_v_dim),
    )


# ---------------------------------------------------------------- prefill ---


class _PrefillStep(NamedTuple):
    """One step's host-side state, built by `PrefillTarget.prepare`."""

    num_tokens: int
    num_prefill_tokens: int
    num_prefills: int
    num_decodes: int
    attn: dict
    si_p: torch.Tensor
    si_d: torch.Tensor
    hi_p: torch.Tensor
    all_initial_states: bool
    qsl_long_p: torch.Tensor
    skip_leading_q_tiles: bool
    row_slice: torch.Tensor | None


class PrefillTarget:
    """Any batch with context rows, possibly with generation rows beside them, run
    eagerly."""

    def __init__(self, core: Qwen3827BTp1ModelingV2) -> None:
        self.core = core
        self._contract_pending = step_contract_enabled()
        self._step: _PrefillStep | None = None

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        return _step_args(md, md.num_contexts, md.num_ctx_tokens)

    def prepare(self, md, return_context_logits: bool) -> None:
        if self._contract_pending:
            try:
                self.step_args(md)
            except AttributeError as exc:
                raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
            self._contract_pending = False
        assert md.padded_num_tokens is None, (
            "a padded prefill step: this target runs prefill steps eagerly and unpadded"
        )
        mm = md.mamba_metadata
        num_tokens = md.num_tokens
        batch = md.num_seqs
        num_prefills = md.num_contexts
        num_decodes = batch - num_prefills
        num_prefill_tokens = md.num_ctx_tokens
        assert num_tokens - num_prefill_tokens == num_decodes, (
            "one query row per generation sequence"
        )
        si_all = mm.state_indices[:batch]
        si_p, si_d = torch.split(si_all, [num_prefills, num_decodes], 0)
        hi_p = mm.has_initial_states[:num_prefills]
        # Every context carrying state (a prefix-cache hit, a later chunk) makes the reset
        # a masked no-op; the host copy says so without a sync.
        all_initial = bool(mm.has_initial_states_cpu[:num_prefills].all())

        ctx_lens = md.seq_lens[:num_prefills].tolist()
        num_q_tiles = (max(ctx_lens) + _FMHA_Q_TILE - 1) // _FMHA_Q_TILE
        # Without context logits only each context's last row of the final layer is read;
        # when every context ends in the same last q-tile, the leading q-tiles of the final
        # layer's context attention are never read.
        skip = (
            not return_context_logits
            and num_q_tiles >= 2
            and min(ctx_lens) > _FMHA_Q_TILE * (num_q_tiles - 1)
        )
        row_slice = None
        if (
            not return_context_logits
            and num_tokens >= _ROW_SLICE_MIN_ROWS
            and batch < num_tokens
            and not AutoTuner.get().is_tuning_mode
        ):
            # The rows the logits processor reads, in its order.
            row_slice = torch.cumsum(md.seq_lens_cuda[:batch], dim=0, dtype=torch.long) - 1
        self._step = _PrefillStep(
            num_tokens=num_tokens,
            num_prefill_tokens=num_prefill_tokens,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            attn=self.step_args(md),
            si_p=si_p,
            si_d=si_d,
            hi_p=hi_p,
            all_initial_states=all_initial,
            qsl_long_p=mm.query_start_loc_long[: num_prefills + 1],
            skip_leading_q_tiles=skip,
            row_slice=row_slice,
        )

    @property
    def rows_selected(self) -> bool:
        return self._step is not None and self._step.row_slice is not None

    def _attention(self, L: _Layer, x, residual, positions):
        core = self.core
        step = self._step
        rows = step.num_tokens
        i = L.i
        w = core.w
        qkv = _qkv(core, i, x, rows)
        qkv, gate = qkv_gemma_rmsnorm_rope_gate(
            qkv,
            w[f"l{i}_q_norm"],
            w[f"l{i}_k_norm"],
            core._cos_sin,
            positions,
            core.eps,
            core.heads_q,
            core.heads_kv,
            core.head_dim,
            core.rotary_dim,
            core.mrope_section,
        )
        attn_out = torch.empty([rows, core.q_width], dtype=x.dtype, device=x.device)
        attention_sm120(
            qkv,
            attn_out,
            layer_idx=core._bound.pool_idx[i],
            kv_scale_orig_quant=core.kv_scale_orig_quant,
            kv_scale_quant_orig=core.kv_scale_quant_orig,
            buffers=core._bound.attn_buffers,
            q_scaling=_Q_SCALING,
            skip_leading_q_tiles=L.is_last and step.skip_leading_q_tiles,
            shared_prefix_min_batch=_XQA_SHARED_PREFIX_MIN_BATCH,
            **step.attn,
        )
        if L.is_last and step.row_slice is not None:
            idx = step.row_slice
            attn_out, gate, residual = attn_out[idx], gate[idx], residual[idx]
        return _attention_out(core, L, attn_out, gate, rows), residual

    def _gdn(self, L: _Layer, x):
        core = self.core
        step = self._step
        rows = step.num_tokens
        i = L.i
        w = core.w
        bound = core._bound
        conv, ssm = bound.conv[i], bound.ssm[i]
        if rows >= _BA_FORK_MIN_ROWS and not torch.cuda.is_current_stream_capturing():
            # in_proj_ba on the aux stream beside the W4A4 in_proj_qkvz, in
            # maybe_execute_in_parallel's order.
            main_event, ba_event = core.events
            main_event.record()
            proj = _project(core, "qkvz", i, x, rows)
            with torch.cuda.stream(core.aux_stream):
                main_event.wait()
                ba = _ba(core, i, x, rows)
                ba_event.record()
            ba_event.wait()
        else:
            proj, ba = maybe_execute_in_parallel(
                lambda: _project(core, "qkvz", i, x, rows),
                lambda: _ba(core, i, x, rows),
                core.events[0],
                core.events[1],
                core.aux_stream,
                disable_on_compile=True,
            )
        mixed_qkv, z, a, b = _split_gdn(core, proj, ba)
        P = step.num_prefill_tokens
        out = torch.empty(
            [1, rows, core.num_v_heads, core.head_v_dim], dtype=x.dtype, device=x.device
        )
        if not step.all_initial_states:
            # A context carrying nothing starts from a zeroed slot.
            reset_recurrent_state_rows(ssm, step.si_p, step.hi_p, conv)
        if step.num_decodes > 0:
            # The generation riders: the conv step, then the single-token recurrence.
            mq_d = mixed_qkv[P:rows]
            causal_conv1d_update(
                mq_d, conv, w[f"l{i}_conv_w"], None, True, None, step.si_d, _PAD_SLOT_ID
            )
            q, k, v = _qkv_views(core, mq_d)
            gdn_decode_t1(
                q,
                k,
                v,
                a[P:rows],
                b[P:rows],
                w[f"l{i}_A_log"],
                w[f"l{i}_dt_bias"],
                ssm,
                step.si_d,
                out[0, P:rows],
            )
        q, k, v, g, beta = gdn_prefill_onepass(
            mixed_qkv[:P],
            a[:P],
            b[:P],
            w[f"l{i}_conv_w"],
            conv,
            step.si_p,
            step.hi_p,
            step.qsl_long_p,
            w[f"l{i}_A_log"],
            w[f"l{i}_dt_bias"],
        )
        flashinfer_chunk_gated_delta_rule_q8(
            q, k, v, g, beta, ssm, step.si_p, step.qsl_long_p, step.num_prefills, out[:, :P]
        )
        return _gdn_out(core, i, out, z, rows)

    def forward(self, attn_metadata, input_ids=None, position_ids=None, inputs_embeds=None):
        core = self.core
        step = self._step
        rows = step.num_tokens
        assert (input_ids if input_ids is not None else inputs_embeds).shape[0] == rows
        positions = position_ids.reshape(3, -1)
        h = (
            inputs_embeds
            if inputs_embeds is not None
            else masked_embedding_gather(input_ids, core.w["embed"])
        )
        residual = h
        x = flashinfer_gemma_rmsnorm(h, core.layers_view[0].in_norm, core.eps)
        for L in core.layers_view:
            if L.is_attn:
                x, residual = self._attention(L, x, residual, positions)
            else:
                x = self._gdn(L, x)
            h, residual = _post_attention_norm(core, L, x, residual, rows)
            x = _mlp(core, L.i, h, rows)
            x, residual = _next_norm(core, L, x, residual, rows)
        return x


# ----------------------------------------------------------------- decode ---


class DecodeTarget:
    """Generation-only batches: one query row per sequence, captured in the engine's
    decode CUDA graphs (every branch is on the padded row count, a per-capture constant)."""

    def __init__(self, core: Qwen3827BTp1ModelingV2) -> None:
        self.core = core
        self._contract_pending = step_contract_enabled()
        self._attn: dict | None = None

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """Context counts are stated, not read: routing guarantees them, and a decode CUDA
        graph would freeze whatever it read."""
        return _step_args(md, num_contexts=0, num_ctx_tokens=0)

    def prepare(self, md, return_context_logits: bool) -> None:
        if self._contract_pending:
            try:
                self.step_args(md)
            except AttributeError as exc:
                raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
            self._contract_pending = False
        self._attn = self.step_args(md)

    rows_selected = False

    def _attention(self, L: _Layer, x, positions, rows: int):
        core = self.core
        i = L.i
        w = core.w
        qkv = _qkv(core, i, x, rows)
        qkv, gate = qkv_gemma_rmsnorm_rope_gate(
            qkv,
            w[f"l{i}_q_norm"],
            w[f"l{i}_k_norm"],
            core._cos_sin,
            positions,
            core.eps,
            core.heads_q,
            core.heads_kv,
            core.head_dim,
            core.rotary_dim,
            core.mrope_section,
        )
        attn_out = torch.empty([rows, core.q_width], dtype=x.dtype, device=x.device)
        attention_sm120(
            qkv,
            attn_out,
            layer_idx=core._bound.pool_idx[i],
            kv_scale_orig_quant=core.kv_scale_orig_quant,
            kv_scale_quant_orig=core.kv_scale_quant_orig,
            buffers=core._bound.attn_buffers,
            q_scaling=_Q_SCALING,
            skip_leading_q_tiles=False,
            shared_prefix_min_batch=_XQA_SHARED_PREFIX_MIN_BATCH,
            **self._attn,
        )
        return _attention_out(core, L, attn_out, gate, rows)

    def _gdn(self, L: _Layer, x, slots, rows: int):
        core = self.core
        i = L.i
        w = core.w
        bound = core._bound
        conv, ssm = bound.conv[i], bound.ssm[i]
        conv_fused = rows <= _GDN_CONV_FUSED_MAX_ROWS
        if conv_fused:
            g = core.copies[f"l{i}_qkvz_gv"]

            def qkvz():
                return nvfp4_w4a16_gemv_conv(
                    x,
                    g.weight,
                    g.weight_sf,
                    g.global_scale,
                    conv,
                    w[f"l{i}_conv_w"],
                    slots,
                    core.conv_dim,
                    _PAD_SLOT_ID,
                )
        else:

            def qkvz():
                return _project(core, "qkvz", i, x, rows)

        proj, ba = maybe_execute_in_parallel(
            qkvz,
            lambda: _ba(core, i, x, rows),
            core.events[0],
            core.events[1],
            core.aux_stream,
            disable_on_compile=True,
        )
        mixed_qkv, z, a, b = _split_gdn(core, proj, ba)
        if not conv_fused:
            causal_conv1d_update(
                mixed_qkv, conv, w[f"l{i}_conv_w"], None, True, None, slots, _PAD_SLOT_ID
            )
        q, k, v = _qkv_views(core, mixed_qkv)
        if _T1_NORM_MIN_ROWS <= rows <= _T1_NORM_MAX_ROWS:
            # The recurrence with the gated norm folded in; its bf16 rows feed out_proj's
            # Marlin copy.
            y = gdn_decode_t1_gated_norm(
                q,
                k,
                v,
                a,
                b,
                w[f"l{i}_A_log"],
                w[f"l{i}_dt_bias"],
                ssm,
                slots,
                z,
                w[f"l{i}_gnorm"],
                core.eps,
            )
            return _project(core, "out", i, y.view(rows, core.value_dim), rows)
        o = gdn_decode_t1(q, k, v, a, b, w[f"l{i}_A_log"], w[f"l{i}_dt_bias"], ssm, slots, None)
        return _gdn_out(core, i, o, z, rows)

    def forward(self, attn_metadata, input_ids=None, position_ids=None, inputs_embeds=None):
        core = self.core
        rows = (input_ids if input_ids is not None else inputs_embeds).shape[0]
        positions = position_ids.reshape(3, -1)
        slots = attn_metadata.mamba_metadata.state_indices[:rows]
        h = (
            inputs_embeds
            if inputs_embeds is not None
            else masked_embedding_gather(input_ids, core.w["embed"])
        )
        residual = h
        x = flashinfer_gemma_rmsnorm(h, core.layers_view[0].in_norm, core.eps)
        for L in core.layers_view:
            if L.is_attn:
                x = self._attention(L, x, positions, rows)
            else:
                x = self._gdn(L, x, slots, rows)
            h, residual = _post_attention_norm(core, L, x, residual, rows)
            x = _mlp(core, L.i, h, rows)
            x, residual = _next_norm(core, L, x, residual, rows)
        return x


# ------------------------------------------------------------------ shell ---


class _LMHead(nn.Module):
    """The checkpoint's NVFP4 lm_head: W4A16 Marlin (bf16 logits) below
    `_LM_HEAD_W4A4_MIN_ROWS` read rows, the checkpoint's own W4A4 (FP32 logits) from it."""

    def __init__(self, core: Qwen3827BTp1ModelingV2) -> None:
        super().__init__()
        self._core = (core,)  # not a submodule: the core is the shell's `model`
        self.vocab_size_padded = core.vocab
        self.num_embeddings = core.vocab

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        core = self._core[0]
        rows = hidden_states.shape[0]
        if AutoTuner.get().is_tuning_mode:
            # Tune the W4A4 head's buckets up to 256 rows once, whatever this call reads.
            a4 = core.copies["lm_a4"]
            probe = torch.randn(
                256, core.hidden, dtype=hidden_states.dtype, device=hidden_states.device
            )
            pq, psf = fp4_quantize(probe, a4.act_global, _SF_VEC, False)
            _w4a4(a4, pq, psf, 256, torch.float32)
        if rows >= _LM_HEAD_W4A4_MIN_ROWS:
            a4 = core.copies["lm_a4"]
            xq, xsf = fp4_quantize(hidden_states, a4.act_global, _SF_VEC, False)
            return _w4a4(a4, xq, xsf, rows, torch.float32)
        # lm_head's shape has no launch-table entry: the stock dispatcher.
        return _marlin(core.copies["lm_mw"], hidden_states, rows, launch_hint=0)


def _carries_media(params) -> bool:
    """A request's multimodal params hold pixels or encoder embeddings, not only the mRoPE
    bookkeeping a text request of a vision-language checkpoint can carry."""
    data = getattr(params, "multimodal_data", None) or {}
    return (
        data.get("image", {}).get("pixel_values") is not None
        or data.get("video", {}).get("pixel_values_videos") is not None
        or data.get("multimodal_embedding") is not None
    )


class ModelingV2Qwen3827BNvfp4Sm120Tp1(
    DecoderModelForCausalLM[Qwen3827BTp1ModelingV2, PretrainedConfig]
):
    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config=None) -> str:
        """The quantized recurrent-state pool exists only on the V2 hybrid manager."""
        return "V2"

    def __init__(self, model_config: ModelConfig):
        cfg = model_config.pretrained_config
        text = getattr(cfg, "text_config", None)
        assert text is not None, (
            "this target is the text tower of the composite Qwen3.5 checkpoint; "
            "pretrained_config carries no text_config"
        )
        composite = cfg
        # Everything that sizes the pools reads `model_config.pretrained_config`; the engine
        # keeps its own reference to this ModelConfig, so the swap is in place.
        model_config.pretrained_config = text
        super().__init__(
            Qwen3827BTp1ModelingV2(model_config, composite),
            config=model_config,
            hidden_size=text.hidden_size,
            vocab_size=text.vocab_size,
        )
        # The checkpoint's NVFP4 head, owned by the target; the base class's head is
        # replaced before materialization so it never takes storage.
        self.lm_head = _LMHead(self.model)
        self.logits_processor = LogitsProcessor()

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
        assert spec_metadata is None, "this target implements no speculative decoding"
        if any(_carries_media(p) for p in kwargs.get("multimodal_params") or ()):
            raise ValueError(
                "the qwen3_5 modeling_v2 target serves text only; image and video requests "
                "need modeling_v2='off'"
            )
        self.model.begin_step(
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds,
            lora_params,
            return_context_logits,
        )
        target = self.model._select_target(attn_metadata)
        hidden = self.model(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        # A row-sliced final layer already returned exactly the rows the processor reads.
        return self.logits_processor.forward(
            hidden, self.lm_head, attn_metadata, return_context_logits or target.rows_selected
        )

    def load_weights(self, weights, *args, **kwargs):
        _weights.MODEL_WEIGHTS.load(self, weights)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()
