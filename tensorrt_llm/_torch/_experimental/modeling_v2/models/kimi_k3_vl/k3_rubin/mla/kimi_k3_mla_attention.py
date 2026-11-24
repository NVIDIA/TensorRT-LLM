# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 specialization of the shared PyTorch-backend MLA module.

The base ``MLA`` class owns context attention, cached/chunked prefill,
absorbed generation, and paged-cache handling. This module only supplies the
K3 projection topology, NoPE identity table, KV-B checkpoint layout, and
gated output projection.

This is the Kimi K3 Rubin package's local copy: main's module merged with the Rubin agentic
branch's generation-path work (side-stream output gate, fused query, fused
Helix tail, MXFP8 layer input). The hooks that branch added to the shared base ``MLA`` are
re-expressed here as overrides of main's ``attention/mla.py``
(``_forward_impl``, ``_attn_forward_gen``, ``forward``). The FP4 MLA engine entry points these
paths use (the KV append without query rows, the query-launch mark and decode PDL switch, the
early step plan, ``DeferredPackedReduce``) live in ``attention/backends/fp4_mla``; FP4-KV layers
install K3's packed decode kernel (``kernels/fp4_mla_mufu16``) as that backend's CuTe DSL kernel
module. The generation-backend selection (``_MLA_TAKES_GEN_BACKEND``) applies only where the base
``MLA`` takes it; main's FMHA registry selects the MLA decode kernel instead.
"""

from __future__ import annotations

import contextlib
import copy
import inspect
from typing import Callable, Optional, Tuple

import torch

from tensorrt_llm._torch.attention.attention import _helix_post_process, _helix_zero_kv_mask
from tensorrt_llm._torch.attention.backends import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionMetadata,
    TrtllmAttention,
)
from tensorrt_llm._torch.attention.backends.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.attention.mla import MLA, _slice_hidden_states_to_num_tokens
from tensorrt_llm._torch.distributed import AllReduceParams
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.multi_stream_utils import (
    do_multi_stream,
    maybe_execute_in_parallel,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.utils import AuxStreamType, Fp4QuantizedTensor
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

from ..features import active as _features
from .ctx_fp8 import KimiK3MlaContextFp8Mixin
from .ctx_split_k import KimiK3MlaContextSplitKMixin
from .helix_tail import OPROJ_PC, K3HelixTail, decode_releases_tail_early, fold_inputs_of

_F = _features()

# TRTLLM_K3_MLA_GATE_SIDE=1: in a generation-only Helix batch (V up-projection before the
# all-to-all) whose o_proj runs the low-M GEMM, the Helix zero-KV mask and the output gate
# sigmoid(g_proj(x)), which only depend on the step metadata and the layer input, run on a
# side stream; o_proj then multiplies the all-to-all's combined output by the gate while
# quantizing it (``k3_fp8w_lowm_gemm_gated``) and reads that output where the combine left
# it. After the combine only o_proj remains. Bit-exact.
_K3_MLA_GATE_SIDE = _F.mla_gate_side
# TRTLLM_K3_MLA_GATE_SIDE_MAX_M=<rows> (with TRTLLM_K3_MLA_GATE_SIDE): take that side-stream
# path up to this many rows also where o_proj runs the mid-M GEMM (above the low-M range; 0: the
# low-M range only). The gate is then the mid-M g_proj GEMM + sigmoid on the side stream, and
# o_proj runs on attn_out * gate: the same values as the default path's
# o_proj(attn_out * sigmoid(g_proj(x))), but the gate leaves the critical path after the decode.
_K3_MLA_GATE_SIDE_MAX_M = _F.mla_gate_side_max_m
# TRTLLM_K3_MID_OPROJ_PC=1 (with TRTLLM_K3_OPROJ_PC and the side-stream gate above the low-M
# range): the fused Helix tail stores o_proj's operand, the MXFP8 form of bf16(out * gate), with the
# scales in the mid-M GEMM's layout (k3_helix_tail ``mx_atom``), and o_proj runs the mid-M GEMM on
# it (``forward_mid_mxfp8``, with the all-reduce push where TRTLLM_K3_MID_AR_PUSH applies): no
# multiply or quantization kernel after the tail. Bitwise o_proj(attn_out * gate).
_K3_MID_OPROJ_PC = False
# TRTLLM_K3_MLA_Q_NORM_PDL=1: the q_a RMSNorm releases the q_b GEMM at launch instead of after
# its store (``q_norm_pdl``; same kernel code, bit-exact), so q_b streams its weights while
# q_a and the norm still run.
_K3_MLA_Q_NORM_PDL = _F.mla_q_norm_pdl
# TRTLLM_K3_HELIX_TAIL_GATE (with TRTLLM_K3_HELIX_TAIL_FUSED and TRTLLM_K3_MLA_GATE_SIDE): where the
# side-stream output gate of a fused-tail layer runs. "early" (default): forked before the decode
# and joined before the tail kernel, so o_proj depends on the tail alone and keeps its
# programmatic early launch; "exchange": forked right before the tail kernel and joined before
# o_proj, the placement of the unfused path.
_K3_HELIX_TAIL_GATE_EARLY = True
# TRTLLM_K3_ULL3_MLA_GATE_AT (with the early gate above, fused-tail batches): when the side-stream
# output gate GEMM (84 CTAs of 98 KB, ~4 us) is forked. It shares no SM with the decode (one
# 227 KB CTA per SM), so wherever it overlaps the decode it delays some of the decode's clusters.
# "decode" (default): right before the decode, joined before the tail (today); "start": at the
# layer start with the zero-KV mask (it only reads the layer input), beside q_a / kv_a;
# "exchange": right after the decode is launched (it then runs beside the tail) and joined before
# o_proj, which then gates its input itself (the gated low-M GEMM; no MXFP8 operand from the tail,
# as TRTLLM_K3_HELIX_TAIL_GATE=exchange). A value "<place>:<T>,<place>" picks the first placement
# up to T tokens (default 16) and the second above, e.g. "exchange:16,start". Scheduling only,
# bit-exact.
_ULL3_GATE_PLACES = ("decode", "start", "exchange")


def _parse_ull3_gate_at(value: str) -> Tuple[str, int, str]:
    small, _, large = value.partition(",")
    small, _, limit = small.partition(":")
    large = large or small
    if small not in _ULL3_GATE_PLACES or large not in _ULL3_GATE_PLACES:
        raise ValueError(
            f"TRTLLM_K3_ULL3_MLA_GATE_AT={value!r}: expected <place>[:<tokens>][,<place>] with "
            f"places {_ULL3_GATE_PLACES}"
        )
    return small, int(limit) if limit else 16, large


_K3_ULL3_MLA_GATE_AT = _parse_ull3_gate_at(_F.ull3_mla_gate_at)


def ull3_gate_place(num_tokens: int) -> str:
    """The TRTLLM_K3_ULL3_MLA_GATE_AT placement of a batch of ``num_tokens`` tokens."""
    small, limit, large = _K3_ULL3_MLA_GATE_AT
    return small if num_tokens <= limit else large


# TRTLLM_K3_MLA_Q_FUSED=1: in a generation-only Helix batch (at most
# TRTLLM_K3_MLA_Q_FUSED_MAX_TOKENS tokens) of an FP4-KV layer with the V up-projection before the
# all-to-all, the query side (q_a RMSNorm, q_b projection, W_UK absorption, RoPE and the FP4
# query quantization of the packed FP4 decode) is one kernel (``k3_mla_q_fused``), and the KV half
# (kv_a RMSNorm and k_pe into the latent row in one ``k3_mla_latent_row`` launch, then the FP4
# generation cache update without its query rows) runs on the attention aux stream beside it;
# the decode joins both. The KV cache bytes are the default path's; the FP4 query differs from it
# only where a q_b output rounds differently to BF16 (another FP32 summation order).
_K3_MLA_Q_FUSED = _F.mla_q_fused
# TRTLLM_K3_MLA_Q_FUSED_NORM (default 1, with TRTLLM_K3_MLA_Q_FUSED): where the q_a RMSNorm is
# FlashInfer's CuTe DSL kernel, the fused query kernel applies it to the raw q_a latent itself with
# the same arithmetic (bitwise the separate norm), so it waits on the q_a projection only.
_K3_MLA_Q_FUSED_NORM = True
# Largest generation batch (tokens) TRTLLM_K3_MLA_Q_FUSED serves (the kernel takes up to 16).
_K3_MLA_Q_FUSED_MAX_TOKENS = 16
# TRTLLM_K3_MLA_Q_FUSED_KV_PRIO=1: that KV half runs on a high-priority stream of its own.
_K3_MLA_Q_FUSED_KV_PRIO = False
# One side stream per device, shared by all MLA layers (their side work never overlaps).
_GATE_SIDE_STREAMS: dict[tuple[int, bool], torch.cuda.Stream] = {}


def _gate_side_stream(device: torch.device, high_priority: bool = False) -> torch.cuda.Stream:
    index = device.index if device.index is not None else torch.cuda.current_device()
    stream = _GATE_SIDE_STREAMS.get((index, high_priority))
    if stream is None:
        priority = torch.cuda.Stream.priority_range()[1] if high_priority else 0
        stream = torch.cuda.Stream(device=index, priority=priority)
        _GATE_SIDE_STREAMS[(index, high_priority)] = stream
    return stream


# The absorbed-generation MLA backend every deployment runs (passed only to an MLA that takes one,
# see _MLA_TAKES_GEN_BACKEND).
_KIMI_K3_MLA_GEN_BACKEND_DEFAULT = "cute-dsl"


def _select_mla_generation_backend(quant_config: Optional[QuantConfig]) -> str:
    """Select K3's absorbed-generation MLA backend.

    FP8 KV cache requires the native CuTe-DSL wrapper, which accepts its
    device scales; TRTLLM-Gen does not support K3's 96 query heads. Other
    KV-cache dtypes keep the existing configured/default backend selection
    unchanged.
    """
    backend = _KIMI_K3_MLA_GEN_BACKEND_DEFAULT
    has_fp8_kv_cache = bool(
        quant_config is not None and quant_config.layer_quant_mode.has_fp8_kv_cache()
    )
    if has_fp8_kv_cache and backend != "cute-dsl":
        logger.info(
            "Kimi K3 MLA: FP8 KV cache requires the CuTe-DSL MLA "
            f"generation backend; overriding '{backend}' -> 'cute-dsl'."
        )
        return "cute-dsl"
    return backend


def _kimi_k3_mla_decode_backend_policy(
    requested_backend: str,
    metadata,
    num_tokens: int,
) -> str:
    """Per-batch MLA decode backend selection for Kimi K3 with non-FP8 KV.

    Installed as ``mla_backend_policy`` on K3's generation attention backend
    only when the KV cache is not FP8 (see :class:`KimiK3MLAAttention`). The
    general attention code applies no such policy on its own.

    CuTe-DSL reuses one staged page table across MLA layers for a
    generation-only, one-token-per-request batch. A mixed context/generation
    batch cannot use that reuse key, so selecting CuTe-DSL would repeat the
    staging copies in every MLA layer and regress time to first token. Keep
    the requested CuTe-DSL fast path for plain decode and use TRTLLM-Gen for
    mixed batches and speculative multi-token verification. For H=96, the
    FlashInfer wrapper declines that effective TRTLLM-Gen backend so FMHA
    selection can continue to the legacy TRTLLM-Gen path.
    """
    is_single_token_generation = num_tokens == metadata.num_generations
    if requested_backend == "cute-dsl" and (
        metadata.num_contexts > 0 or not is_single_token_generation
    ):
        return "trtllm-gen"
    return requested_backend


# Main selects the MLA decode kernel in its FMHA registry (``attention/backends/fmha/``, gated on
# measured shapes) and has neither the base ``MLA``'s ``flashinfer_mla_backend`` argument nor the
# attention backend's ``mla_backend_policy`` hook that the two functions above feed. They are
# applied only where the engine has them; elsewhere ``TLLM_K3_MLA_GEN_BACKEND`` is read and has no
# effect. It only matters for the FP8-KV layers (``TLLM_KIMI_K3_FP8_KV_LAYERS``): FP4-KV layers
# decode through the FP4 MLA backend, and the registry gives an FP8-KV Helix verify group of K3's
# 96 heads to the CuTe DSL MLA decode, which is what ``cute-dsl`` selects.
_MLA_TAKES_GEN_BACKEND = "flashinfer_mla_backend" in inspect.signature(MLA.__init__).parameters


def helix_gen_zero_kv_mask(
    attn_metadata: AttentionMetadata, num_tokens: int
) -> Optional[torch.Tensor]:
    """Per-token mask of the ``num_tokens`` generation tokens this CP rank owns no KV for.

    The Rubin branch's base-MLA helper, on main's ``_helix_zero_kv_mask``; the same mask main's
    ``MLA._attn_forward_gen`` builds inline. It only depends on the step's metadata, so a caller may
    compute it ahead of the attention (``helix_zero_kv_mask_fn``).
    """
    helix_kv_bounds = getattr(attn_metadata, "helix_kv_bounds", None)
    if helix_kv_bounds is not None and getattr(attn_metadata, "_helix_spec_tokens_valid", False):
        # Speculative verify groups: KV ownership is per token (see MLA._attn_forward_gen).
        return helix_kv_bounds[:num_tokens] == 0
    return _helix_zero_kv_mask(
        attn_metadata,
        num_tokens,
        seq_start=attn_metadata.num_contexts,
        num_seqs=attn_metadata.num_generations,
    )


def _fp4_mla_prequantized_q(
    md: AttentionMetadata,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """``(q, q_sf, rows_per_seq)``: the FP4 query buffers the generation cache update recorded on
    ``metadata.fp4_mla_state`` for the decode."""
    state = md.fp4_mla_state
    return (
        state.prequantized_q,
        state.prequantized_q_sf,
        int(state.prequantized_q_rows_per_seq or 0),
    )


# TRTLLM_K3_FP4_MLA_SHARED_DECODE=1: FP4-KV layers keep the FP4 MLA backend's shared packed decode
# kernel instead of K3's (``kernels/fp4_mla_mufu16``). The wave / makespan plans and FP16 split
# partials need K3's kernel and are off with the shared one.
_K3_FP4_MLA_SHARED_DECODE = False


def _install_k3_fp4_mla_decode_kernel() -> None:
    """Make the FP4 MLA backend run K3's packed CuTe DSL decode kernel module, where the CuTe DSL
    backend is available (SM107 with a CuTe DSL that builds the FP4 MLA kernels)."""
    from tensorrt_llm._torch.attention.backends.fp4_mla.config import (
        _cutedsl_backend_available,
        set_fp4_mla_cutedsl_kernel_module,
    )

    if _K3_FP4_MLA_SHARED_DECODE or not _cutedsl_backend_available():
        return
    from ..kernels import fp4_mla_mufu16

    set_fp4_mla_cutedsl_kernel_module(fp4_mla_mufu16)
    logger.info_once(
        "Kimi K3 MLA: the FP4 MLA packed decode runs K3's CuTe DSL kernel module "
        f"({fp4_mla_mufu16.__name__}).",
        key="kimi_k3_fp4_mla_decode_kernel",
    )


def _make_pos_embd_params(
    *,
    qk_rope_head_dim: int,
    max_position_embeddings: int,
) -> PositionalEmbeddingParams:
    """Build a valid rope config so the backend allocates a real cache.

    We use rope_gpt_neox with default theta=10000 and ``duplicate_data
    =True`` (the same convention DeepSeek-V3-style MLA uses when
    ``qk_rope_head_dim`` is present). The resulting ``rotary_cos_sin``
    has the exact shape the C++ MLA rope kernel indexes. Immediately
    after backend construction we overwrite the tensor values with
    ``(cos=1, sin=0)`` — an identity rotation, matching K3's NoPE.
    """
    rope_params = RopeParams(
        dim=qk_rope_head_dim,
        theta=10000.0,
        max_positions=max_position_embeddings,
        original_max_positions=max_position_embeddings,
        duplicate_data=True,
    )
    return PositionalEmbeddingParams(
        type=PositionEmbeddingType.rope_gpt_neox,
        rope=rope_params,
        # Match the working DeepSeek-V3-style MLA reference test
        # (tests/unittest/_torch/attention/test_attention_mla.py) which
        # sets ``is_neox=False``. The MLA fused rope kernel is GPT-J
        # style regardless of this flag, but the C++ FMHA reads this bit
        # elsewhere and stability under identity-cos-sin depends on the
        # standard non-neox layout.
        is_neox=False,
    )


def _write_identity_rope_values(cos_sin: torch.Tensor) -> None:
    """Overwrite a rotary cos/sin table with identity values in place.

    Interleaved (cos, sin) pairs: index [::2] = cos, [1::2] = sin.
    Setting cos=1 and sin=0 per position makes the rotation the
    identity — a mathematical no-op — which preserves K3's NoPE
    semantics without patching the backend.
    """
    flat = cos_sin.reshape(-1)
    with torch.no_grad():
        flat[0::2] = 1.0
        flat[1::2] = 0.0
    # Ensure the identity write reaches CUDA memory before any kernel
    # launched from a different stream can read the table.
    if cos_sin.is_cuda:
        torch.cuda.synchronize(cos_sin.device)


def _install_identity_rope_table(backend: TrtllmAttention) -> None:
    """Install an identity rotary cos/sin table on ``backend``.

    The C++ MLA rope kernels (``mla_rope_generation`` and the context
    preprocess) read this table and apply the rotation; identity values
    make that a copy, preserving K3's NoPE.

    The tensor SHAPE produced by ``create_rope_const_params`` is kept
    intact so the C++ ``float2`` indexing stays valid. Only the values
    are overwritten in place. ``_ensure_rope_table_size`` is replaced
    with an identity-preserving resize: the table may GROW (so the
    fused rope-generation op can never index out of bounds for long
    sequences) but its values are always rewritten to identity right
    after a regeneration, so the real sinusoids never leak in.
    """
    cos_sin = backend.rotary_cos_sin
    if cos_sin is None:
        raise RuntimeError(
            "backend.rotary_cos_sin is None after construction; check "
            "pos_embd_params has a valid RopeParams with dim > 0."
        )
    _write_identity_rope_values(cos_sin)

    orig_resize = backend._ensure_rope_table_size  # bound method

    def _identity_preserving_resize(required_max_positions: int) -> None:
        if required_max_positions <= backend.rope_params.max_positions:
            return
        orig_resize(required_max_positions)
        _write_identity_rope_values(backend.rotary_cos_sin)

    backend._ensure_rope_table_size = _identity_preserving_resize


# ---------------------------------------------------------------------------
# KimiK3MLAAttention.
# ---------------------------------------------------------------------------


class KimiK3MLAAttention(KimiK3MlaContextSplitKMixin, KimiK3MlaContextFp8Mixin, MLA):
    """Kimi K3 MLA implemented as a thin specialization of :class:`MLA`.

    K3 keeps the standard dense MLA attention/cache flow and only changes the
    checkpoint projection topology, positional encoding, KV-B runtime layout,
    and gated output projection.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rms_norm_eps: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        layer_idx: int = 0,
        use_output_gate: bool = True,
        max_position_embeddings: int = 8192,
        model_config: ModelConfig,
        aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream],
        mapping_with_cp: Optional[Mapping] = None,
    ) -> None:
        quant_config = model_config.get_quant_config()
        has_fp8_kv_cache = bool(
            quant_config is not None and quant_config.layer_quant_mode.has_fp8_kv_cache()
        )
        projection_configs = model_config.quant_config_dict or {}
        q_a_config = projection_configs.get("q_a_proj", model_config.quant_config)
        kv_a_config = projection_configs.get("kv_a_proj_with_mqa", model_config.quant_config)
        q_a_algo = q_a_config.quant_algo if q_a_config else None
        kv_a_algo = kv_a_config.quant_algo if kv_a_config else None
        fuse_qkv_a_proj = q_a_algo == kv_a_algo and (
            q_a_algo is None or (q_a_algo == QuantAlgo.FP8_BLOCK_SCALES and q_lora_rank % 128 == 0)
        )
        # Resolve each Linear's recipe before allocating weights, as in the
        # model's layerwise quantization post-init. Shared MLA stays generic.
        deferred_config = copy.copy(model_config)
        deferred_config._frozen = False
        deferred_config.skip_create_weights_in_init = True
        deferred_config._frozen = model_config._frozen
        pos_embd_params = _make_pos_embd_params(
            qk_rope_head_dim=qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
        )
        gen_backend = _select_mla_generation_backend(quant_config)
        gen_backend_kwargs = {}
        if _MLA_TAKES_GEN_BACKEND:
            gen_backend_kwargs["flashinfer_mla_backend"] = gen_backend
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            predicted_tokens_per_seq=1,
            max_position_embeddings=max_position_embeddings,
            bias=False,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=dtype,
            dense_bias=False,
            config=deferred_config,
            aux_stream_dict=aux_stream_dict,
            mapping_with_cp=mapping_with_cp,
            reduce_output=False,
            fuse_qkv_a_proj=fuse_qkv_a_proj,
            rms_norm_eps=rms_norm_eps,
            **gen_backend_kwargs,
        )
        # Keep the base MLA registration enabled so breakable CUDA graphs use
        # the shared custom op. The output gate is a base hook and runs on both
        # the registered and eager paths before the row-parallel o_proj.
        # forward() keeps that: its default path runs the attention through the
        # custom op where registered (see _run_attention); only the
        # generation-only paths of this module call forward_impl() directly.

        self.use_output_gate = use_output_gate

        if use_output_gate:
            # The gate must match o_proj's input sharding (under helix the
            # post-all-to-all 1/cp head chunk); outside helix this equals
            # q_b_proj's head sharding, replicated under attention-DP.
            self.g_proj = Linear(
                hidden_size,
                num_heads * v_head_dim,
                bias=False,
                dtype=dtype,
                mapping=self.o_proj.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=(model_config.quant_config_dict or {}).get(
                    "g_proj", model_config.quant_config
                ),
                skip_create_weights_in_init=True,
                allreduce_strategy=model_config.allreduce_strategy,
                force_dynamic_quantization=model_config.force_dynamic_quantization,
                use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
                use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm,
            )

        for name, module in self.named_children():
            if isinstance(module, Linear):
                module.quant_config = projection_configs.get(name, model_config.quant_config)
        if not model_config.skip_create_weights_in_init:
            self.create_weights()
            for module in self.children():
                if isinstance(module, Linear):
                    module.create_weights()

        # K3 is NoPE. The base MLA backends still require real RoPE tables, so
        # retain their expected shape and replace every rotation with identity.
        assert isinstance(self.mha, TrtllmAttention)
        assert isinstance(self.mqa, TrtllmAttention)
        _install_identity_rope_table(self.mha)
        _install_identity_rope_table(self.mqa)
        if quant_config is not None and quant_config.layer_quant_mode.has_fp4_kv_cache():
            _install_k3_fp4_mla_decode_kernel()
        # Only the absorbed-generation backend (mqa) requests CuTe-DSL, so
        # only it needs K3's per-batch fallback policy. FP8 KV must keep its
        # required CuTe-DSL backend because TRTLLM-Gen does not support K3's
        # 96 query heads, so leave the policy unset and preserve the static
        # backend for every batch composition. Use the config-derived flag:
        # TrtllmAttention has not populated has_fp8_kv_cache yet when weight
        # creation is deferred under MetaInitMode.
        if not has_fp8_kv_cache and hasattr(self.mqa, "mla_backend_policy"):
            self.mqa.mla_backend_policy = _kimi_k3_mla_decode_backend_policy
        self.rotary_emb = None
        self.apply_rotary_emb = False
        # The Rubin branch's base-MLA state, owned here on main (see the module docstring).
        # Helix generation hand-off, set around one forward by the side-gate path: a callable
        # returning the zero-local-KV mask computed ahead of time (called on the current stream
        # where the mask is needed, after the attention kernel), a callable run right before the
        # all-to-all is enqueued (after the V up-projection), and whether the all-to-all's combined
        # output is also left in ``helix_gen_output`` (V up-projection before the all-to-all only).
        self.helix_zero_kv_mask_fn: Optional[Callable[[], Optional[torch.Tensor]]] = None
        self.helix_pre_exchange_fn: Optional[Callable[[], None]] = None
        self.defer_helix_gen_output = False
        self.helix_gen_output: Optional[torch.Tensor] = None
        # TRTLLM_K3_ULL4_MLA_PLAN_EARLY: event of the step metadata the model pre-launched.
        self.step_metadata_ready: Optional[torch.cuda.Event] = None
        # TRTLLM_K3_ULL3_MLA_GATE_AT placement of the side-gate forward in flight.
        self._ull3_gate_place = "decode"
        if not self.is_lite:
            # q_lora + kv_lora + rope rows (2112 for Kimi-K3) leave a partial last 128-row tile,
            # which the FP8 low-M GEMM serves only on request (TRTLLM_K3_FP8W_LOWM_FUSED_A).
            self.kv_a_proj_with_mqa.fp8w_lowm_partial_n_tile = self.fuse_qkv_a_proj
        # TRTLLM_K3_MLA_GATE_SIDE: side-stream fork / join of the zero-KV mask and the gate.
        self._gate_events = tuple(torch.cuda.Event() for _ in range(4))
        # Set around one side-gate forward: joins the side-stream gate where the fused Helix tail
        # (TRTLLM_K3_HELIX_TAIL_FUSED) is enqueued (TRTLLM_K3_HELIX_TAIL_GATE=early) and returns
        # the gate (None if it is not computed yet).
        self.helix_pre_tail_fn: Optional[Callable[[], Optional[torch.Tensor]]] = None
        # TRTLLM_K3_OPROJ_PC: the MXFP8 o_proj operand (xq, xs) the fused tail stored in this
        # forward, consumed by the side-gate forward's o_proj.
        self.helix_tail_mx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        # TRTLLM_K3_MLA_Q_FUSED: fork of the KV half after q_a, join before the decode.
        self._kv_events = (torch.cuda.Event(), torch.cuda.Event())
        # Set by the decoder layer for the next forward only: the MXFP8 form ``(xq, xs)`` of the
        # layer input (``trtllm::k3_attn_res_*_mx``), which the fused q_a / kv_a projection
        # consumes instead of quantizing the BF16 rows itself (see q_a_takes_mxfp8).
        self.next_input_mxfp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        # TRTLLM_K3_MLA_Q_FUSED: (q_b weight scale, its block exponents), built on the first
        # eager forward that takes the fused query.
        self._q_fused_qb_exp: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        # The generation output buffer of the running forward_absorption_generation call, which
        # the fused Helix tail writes into (see _tail_output_buffer).
        self._gen_output_buffer: Optional[torch.Tensor] = None

    def q_a_takes_mxfp8(self, num_rows: int) -> bool:
        """Whether the fused q_a / kv_a projection of ``num_rows`` rows can take their MXFP8 form
        (``next_input_mxfp8``): it runs the low-M kernel on them."""
        takes = getattr(self.kv_a_proj_with_mqa, "takes_mxfp8_input", None)
        return (
            self.fuse_qkv_a_proj
            and not self.is_lite
            and takes is not None
            and (bool(takes(num_rows)) or self.q_a_mx_atom(num_rows))
        )

    def q_a_mx_atom(self, num_rows: int) -> bool:
        """``TRTLLM_K3_MID_MXFP8_IN``: whether the fused q_a / kv_a projection takes the
        MXFP8 rows with the scales in the mid-M GEMM's layout (``takes_mid_mxfp8_input``)."""
        takes_mid = getattr(self.kv_a_proj_with_mqa, "takes_mid_mxfp8_input", None)
        return (
            self.fuse_qkv_a_proj
            and not self.is_lite
            and takes_mid is not None
            and bool(takes_mid(num_rows))
        )

    def _project_qkv_a(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The fused q_a / kv_a projection; on ``next_input_mxfp8`` when the decoder layer
        handed it over (``forward_mxfp8``: bitwise the projection of the BF16 rows)."""
        mxfp8_input, self.next_input_mxfp8 = self.next_input_mxfp8, None
        if mxfp8_input is not None:
            return self.kv_a_proj_with_mqa.forward_mxfp8(hidden_states, *mxfp8_input)
        return self.kv_a_proj_with_mqa(hidden_states)

    def _forward_impl(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache_gen: Optional[torch.Tensor] = None,
    ) -> None:
        """``MLA._forward_impl`` with the fused q_a / kv_a projection on ``next_input_mxfp8``.

        Main's base calls ``kv_a_proj_with_mqa`` directly (the Rubin branch's base went through
        ``_project_qkv_a``), so a forward that was handed the MXFP8 layer input runs this copy of
        the base's fused, non-lite, NoPE path; every other forward runs the base unchanged. Keep it
        in step with ``attention/mla.py`` ``MLA._forward_impl``.
        """
        mxfp8_input = self.next_input_mxfp8
        num_tokens = attn_metadata.num_tokens
        if (
            mxfp8_input is None
            or self.is_lite
            or not self.fuse_qkv_a_proj
            or self.apply_rotary_emb
            or self.llama_4_scaling
            or isinstance(hidden_states, Fp4QuantizedTensor)
            or mxfp8_input[0].shape[0] != min(hidden_states.shape[0], num_tokens)
        ):
            # The BF16 projection gives the same bits (forward_mxfp8 is bitwise it).
            self.next_input_mxfp8 = None
            return super()._forward_impl(
                position_ids,
                hidden_states,
                attn_metadata,
                output,
                latent_cache_gen=latent_cache_gen,
            )
        num_contexts = attn_metadata.num_contexts
        num_generations = attn_metadata.num_generations
        num_ctx_tokens = attn_metadata.num_ctx_tokens

        hidden_states = _slice_hidden_states_to_num_tokens(hidden_states, num_tokens)
        if position_ids is not None:
            position_ids = position_ids[..., :num_tokens]

        q, compressed_kv, k_pe = self._project_qkv_a(hidden_states).split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], -1
        )
        q, compressed_kv = maybe_execute_in_parallel(
            lambda: self._q_a_layernorm_maybe_fused(q),
            lambda: self.kv_a_layernorm(compressed_kv),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        q, latent_cache = maybe_execute_in_parallel(
            lambda: self.q_b_proj(q),
            lambda: torch.concat([compressed_kv, k_pe], dim=-1),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        assert q.shape[0] == num_tokens, (
            f"Expect q.shape[0] to be {num_tokens}, but got {q.shape[0]}"
        )
        assert output is not None, "output must be provided"

        if num_contexts > 0:
            self.forward_context(
                q[:num_ctx_tokens, ...],
                compressed_kv[:num_ctx_tokens, ...],
                k_pe[:num_ctx_tokens, ...],
                position_ids,
                attn_metadata,
                output[:num_ctx_tokens, :],
                latent_cache[:num_ctx_tokens, ...],
            )
        if num_generations > 0:
            if latent_cache_gen is None:
                latent_cache_gen = latent_cache[num_ctx_tokens:, ...]
            self.forward_absorption_generation(
                q[num_ctx_tokens:, ...],
                compressed_kv[num_ctx_tokens:, ...],
                k_pe[num_ctx_tokens:, ...],
                attn_metadata,
                output[num_ctx_tokens:num_tokens, :],
                position_ids=position_ids,
                latent_cache=latent_cache_gen,
            )

    def _apply_output_gate(
        self,
        hidden_states: torch.Tensor,
        attn_output: torch.Tensor,
    ) -> torch.Tensor:
        # Base hook of main's MLA.forward (the path this module leaves to the base, see forward):
        # sigmoid gate on o_proj's input. g_proj matches o_proj's input sharding, so the multiply
        # composes with the helix-CP output projection.
        if self.use_output_gate:
            return attn_output * self.g_proj(hidden_states).sigmoid()
        return attn_output

    def _apply_output_gate_and_o_proj(
        self,
        hidden_states: torch.Tensor,
        attn_out: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
    ) -> torch.Tensor:
        """The output gate and o_proj; o_proj runs as main's ``MLA._project_output``."""
        if self.use_output_gate:
            attn_out = attn_out * self.g_proj(hidden_states).sigmoid()
        return self._project_output([attn_out], position_ids, attn_metadata, all_reduce_params)

    @staticmethod
    def _runs_flashinfer_rmsnorm(norm) -> bool:
        """Whether ``norm(x)`` runs FlashInfer's BF16 RMSNorm (no residual, no fused quant).

        Mirrors main's ``RMSNorm.forward`` dispatch, which (unlike the Rubin branch's) takes the
        FlashInfer kernel whether or not the NVFP4 Marlin path is enabled."""
        return (
            type(norm) is RMSNorm
            and not (norm.use_cuda_tile or norm.use_gemma or norm.is_nvfp4)
            and not norm.return_hp_output
            and IS_FLASHINFER_AVAILABLE
        )

    def _q_a_layernorm_maybe_fused(self, q: torch.Tensor, return_norm_out: bool = False):
        norm = self.q_a_layernorm
        if (
            _K3_MLA_Q_NORM_PDL
            and not return_norm_out
            and self._runs_flashinfer_rmsnorm(norm)
            and get_env_enable_pdl()
            and self._resolve_qa_fused_scale() is None
        ):
            # Where the base path runs FlashInfer's RMSNorm, run the same kernel with its PDL
            # trigger at launch (bit-exact; see ``q_norm_pdl``).
            from .q_norm_pdl import can_use, rmsnorm_release_at_launch

            if can_use(q, norm.weight):
                return rmsnorm_release_at_launch(q, norm.weight, norm.variance_epsilon)
        return super()._q_a_layernorm_maybe_fused(q, return_norm_out)

    def _helix_v_up_enabled(self) -> bool:
        """Whether Helix generation projects V before the CP all-to-all: main builds that weight in
        ``transform_weights`` (``_build_helix_v_up_weight``) for BF16 Helix layers by default;
        the Rubin branch gated it on ``TRTLLM_HELIX_V_UP_BEFORE_A2A``."""
        return (
            self.mapping.has_cp_helix()
            and getattr(self, "_helix_v_up_before_exchange", False)
            and getattr(self, "helix_v_b_proj_full", None) is not None
        )

    def _side_gate_applies(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> bool:
        """Whether this forward takes the ``TRTLLM_K3_MLA_GATE_SIDE`` path."""
        if not (_K3_MLA_GATE_SIDE and self.use_output_gate):
            return False
        if (
            type(hidden_states) is not torch.Tensor
            or hidden_states.dtype != torch.bfloat16
            or hidden_states.dim() != 2
            or hidden_states.shape[0] != attn_metadata.num_tokens
        ):
            return False
        if attn_metadata.num_contexts != 0 or self.sparse_attn_hooks is not None:
            return False
        if not self._helix_v_up_enabled() or not hasattr(self.o_proj, "gated_applies"):
            return False
        from ..kernels.k3_fp8w_lowm_gemm import low_m_max_for

        limit = low_m_max_for(self.o_proj.out_features, self.o_proj.in_features)
        return hidden_states.shape[0] <= max(limit, _K3_MLA_GATE_SIDE_MAX_M)

    def _helix_tail_applies(self, num_tokens: int) -> bool:
        """Whether this generation batch's Helix tail runs as one fused kernel
        (``TRTLLM_K3_HELIX_TAIL_FUSED``, V up-projection before the exchange)."""
        w_uv = getattr(self, "helix_v_b_proj_full", None)
        return (
            isinstance(w_uv, torch.Tensor)
            and w_uv.dtype == torch.bfloat16
            and w_uv.is_contiguous()
            and K3HelixTail.applies(
                self.mapping, num_tokens, self.num_heads_tp, self.kv_lora_rank, self.v_head_dim
            )
        )

    def forward_absorption_generation(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """``MLA.forward_absorption_generation``, with ``output`` exposed to the fused Helix tail
        so the tail writes the combined rows into it and the base has nothing left to copy."""
        self._gen_output_buffer = output
        try:
            return super().forward_absorption_generation(
                q, compressed_kv, k_pe, attn_metadata, output, *args, **kwargs
            )
        finally:
            self._gen_output_buffer = None

    def _tail_output_buffer(self, num_tokens: int) -> Optional[torch.Tensor]:
        """The running ``forward_absorption_generation``'s output as the tail's BF16
        ``[T, H / P * v_head_dim]`` output, or None (no such call, or a buffer the tail cannot
        write as is)."""
        buf = self._gen_output_buffer
        cols = self.num_heads_tp_cp * self.v_head_dim
        if (
            buf is None
            or buf.dtype != torch.bfloat16
            or not buf.is_contiguous()
            or buf.numel() != num_tokens * cols
        ):
            return None
        return buf.view(num_tokens, cols)

    def _attn_forward_gen(
        self,
        attn_backend,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        helix_v_up: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """``MLA._attn_forward_gen``; with ``TRTLLM_K3_HELIX_TAIL_FUSED`` a Helix batch with the V
        up-projection before the exchange runs the decode with its split reduction deferred and
        then one tail kernel (fold, V up-projection, exchange, combine) instead of the reduction,
        the BMM, the all-to-all and the combine. Returns the same ``[T, H / P * v_head_dim]``
        combined output. A Helix batch of the side-gate path (hand-off hooks set) runs main's
        Helix branch with the hooks; every other batch runs main's method unchanged."""
        num_tokens = q.shape[0]
        if helix_v_up and self._helix_tail_applies(num_tokens):
            out = self._attn_forward_gen_fused_tail(
                attn_backend, q, k, v, attn_metadata, num_tokens, **kwargs
            )
        elif self.mapping.has_cp_helix() and (
            self.helix_zero_kv_mask_fn is not None
            or self.helix_pre_exchange_fn is not None
            or self.defer_helix_gen_output
        ):
            out = self._attn_forward_gen_helix_hooked(
                attn_backend, q, k, v, attn_metadata, helix_v_up, **kwargs
            )
        else:
            return super()._attn_forward_gen(
                attn_backend, q, k, v, position_ids, attn_metadata, helix_v_up=helix_v_up, **kwargs
            )
        if helix_v_up and self.defer_helix_gen_output:
            # The side-gate path reads the combined output here. The fused tail wrote it into
            # forward_absorption_generation's output (no copy); the hooked branch did not, and
            # the base copies that one.
            self.helix_gen_output = out
        return out

    def _attn_forward_gen_helix_hooked(
        self,
        attn_backend,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        helix_v_up: bool,
        **kwargs,
    ) -> torch.Tensor:
        """Main's ``MLA._attn_forward_gen`` Helix branch with the Rubin branch's hand-off hooks:
        the zero-KV mask from ``helix_zero_kv_mask_fn`` and ``helix_pre_exchange_fn`` run right
        before the all-to-all. Keep it in step with ``attention/mla.py``."""
        softmax_stats = torch.empty(
            (q.shape[0], self.num_heads_tp, 2), device=q.device, dtype=torch.float32
        )
        kwargs["softmax_stats_tensor"] = softmax_stats
        partial_o = attn_backend.forward(
            q, k, v, attn_metadata, forward_args=AttentionForwardArgs(**kwargs)
        )
        assert partial_o.shape[-1] == self.num_heads_tp * self.kv_lora_rank
        if self.helix_zero_kv_mask_fn is not None:
            zero_kv_mask = self.helix_zero_kv_mask_fn()
        else:
            zero_kv_mask = helix_gen_zero_kv_mask(attn_metadata, partial_o.shape[0])
        value_dim = self.kv_lora_rank
        if helix_v_up:
            partial_o = self._helix_project_v_partials(partial_o)
            value_dim = self.v_head_dim
        if self.helix_pre_exchange_fn is not None:
            self.helix_pre_exchange_fn()
        return _helix_post_process(
            partial_o,
            softmax_stats,
            self.mapping,
            self.num_heads_tp_cp,
            value_dim,
            self.aux_stream,
            self.ln_events,
            zero_kv_mask=zero_kv_mask,
        )

    def _attn_forward_gen_fused_tail(
        self,
        attn_backend,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        num_tokens: int,
        **kwargs,
    ) -> torch.Tensor:
        """``TRTLLM_K3_HELIX_TAIL_FUSED``: the decode with its split reduction deferred, then one
        tail kernel (fold, V up-projection, exchange, combine)."""
        from tensorrt_llm._torch.attention.backends.fp4_mla.fp4_mla_packed_reduce import (
            DeferredPackedReduce,
        )

        softmax_stats = torch.empty(
            (num_tokens, self.num_heads_tp, 2), device=q.device, dtype=torch.float32
        )
        kwargs["softmax_stats_tensor"] = softmax_stats
        # TRTLLM_K3_ULL3_MLA_GATE_AT=exchange: the gate is forked after the decode and joined by
        # o_proj, as with TRTLLM_K3_HELIX_TAIL_GATE=exchange.
        gate_place = getattr(self, "_ull3_gate_place", "decode")
        gate_early = _K3_HELIX_TAIL_GATE_EARLY and gate_place != "exchange"
        if self.helix_pre_exchange_fn is not None and gate_early:
            self.helix_pre_exchange_fn()
        with DeferredPackedReduce(release_early=decode_releases_tail_early(num_tokens)) as deferred:
            partial_o = attn_backend.forward(
                q, k, v, attn_metadata, forward_args=AttentionForwardArgs(**kwargs)
            )
        assert partial_o.shape[-1] == self.num_heads_tp * self.kv_lora_rank
        fold = fold_inputs_of(deferred, partial_o, softmax_stats)
        if self.helix_zero_kv_mask_fn is not None:
            zero_kv_mask = self.helix_zero_kv_mask_fn()
        else:
            zero_kv_mask = helix_gen_zero_kv_mask(attn_metadata, num_tokens)
        if self.helix_pre_exchange_fn is not None and not gate_early:
            self.helix_pre_exchange_fn()
        gate = None
        if self.helix_pre_tail_fn is not None and gate_early:
            gate = self.helix_pre_tail_fn()
        mx_out = None
        mx_atom = False
        out = self._tail_output_buffer(num_tokens)
        mx_applies = getattr(self.o_proj, "gated_mxfp8_applies", None)
        if OPROJ_PC and gate is not None and mx_applies is not None:
            if out is None:
                out = partial_o.new_empty((num_tokens, self.num_heads_tp_cp * self.v_head_dim))
            k = out.shape[1]
            if mx_applies(out, gate):
                xq = torch.empty((num_tokens, k), dtype=torch.float8_e4m3fn, device=q.device)
                xs = torch.empty((num_tokens, k // 32), dtype=torch.uint8, device=q.device)
                mx_out = (gate, xq, xs)
                self.helix_tail_mx = (xq, xs)
            elif _K3_MID_OPROJ_PC and self._mid_oproj_pc_applies(out):
                # TRTLLM_K3_MID_OPROJ_PC: the operand in the mid-M GEMM's scale layout.
                xq = torch.empty((num_tokens, k), dtype=torch.float8_e4m3fn, device=q.device)
                xs = torch.empty(((k // 128) * 512,), dtype=torch.uint8, device=q.device)
                mx_out = (gate, xq, xs)
                mx_atom = True
                self.helix_tail_mx = (xq, xs)
        return K3HelixTail.get(self.mapping)(
            partial_o,
            softmax_stats,
            self.helix_v_b_proj_full,
            zero_kv_mask,
            fold=fold,
            out=out,
            mx_out=mx_out,
            mx_atom=mx_atom,
        )

    def _mid_oproj_pc_applies(self, out: torch.Tensor) -> bool:
        """``TRTLLM_K3_MID_OPROJ_PC``: whether o_proj on ``out``'s rows runs the mid-M GEMM with one
        128-row scale block, so the tail may store its operand in that GEMM's layout."""
        plan = getattr(self.o_proj, "mid_operand_tactic", None)
        return (
            plan is not None
            and out.dim() == 2
            and out.shape[1] % 128 == 0
            and plan(out.shape[0]) is not None
        )

    def _latent_row(self, compressed_kv: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
        """``torch.concat([kv_a_layernorm(compressed_kv), k_pe], -1)``, bit for bit; where the
        norm is FlashInfer's, it writes straight into the concatenated row and ``k_pe`` is
        copied beside it (a tiny copy instead of the batched concat kernel)."""
        norm = self.kv_a_layernorm
        if not (self._runs_flashinfer_rmsnorm(norm) and compressed_kv.dtype == torch.bfloat16):
            return torch.concat([norm(compressed_kv), k_pe], dim=-1)
        from flashinfer.norm import rmsnorm

        lora = compressed_kv.shape[-1]
        latent = torch.empty(
            (compressed_kv.shape[0], lora + k_pe.shape[-1]),
            dtype=compressed_kv.dtype,
            device=compressed_kv.device,
        )
        rmsnorm(
            compressed_kv,
            norm.weight,
            norm.variance_epsilon,
            out=latent[:, :lora],
            enable_pdl=get_env_enable_pdl(),
        )
        latent[:, lora:].copy_(k_pe)
        return latent

    @staticmethod
    def _flashinfer_cute_norm(device: torch.device) -> bool:
        """Whether FlashInfer's ``rmsnorm`` runs its CuTe DSL ``RMSNormKernel`` on ``device``
        (the arithmetic ``k3_mla_latent_row`` reproduces), not its CUDA JIT kernel."""
        try:
            from flashinfer.norm import _use_cuda_norm
        except ImportError:
            return False
        return not _use_cuda_norm(device)

    def _q_fused_w_uk(self) -> Optional[torch.Tensor]:
        """The BF16 absorption weight ``[H, kv_lora_rank, qk_nope_head_dim]`` the default path's
        BMM reads (the module weight, or its dequantized copy)."""
        w_kc = getattr(self, "k_b_proj_trans", None)
        if w_kc is not None and w_kc.dtype != torch.bfloat16:
            w_kc = getattr(self, "k_b_proj_trans_dequant", None)
        return w_kc

    def _q_fused_qb_exponents(self) -> Optional[torch.Tensor]:
        """Block exponents of the q_b weight (``k3_mla_q_fused.block_exponents``), cached per
        weight scale; None if they would have to be built during CUDA graph capture."""
        scale = getattr(self.q_b_proj, "weight_scale", None)
        if not isinstance(scale, torch.Tensor):
            return None
        cached = self._q_fused_qb_exp
        if cached is not None and cached[0] is scale:
            return cached[1]
        if torch.cuda.is_current_stream_capturing():
            return None
        from ..kernels import k3_mla_q_fused

        weight = self.q_b_proj.weight
        try:
            exponents = k3_mla_q_fused.block_exponents(scale, *weight.shape)
        except ValueError:
            return None
        self._q_fused_qb_exp = (scale, exponents)
        return exponents

    def _q_fused_applies(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> bool:
        """Whether this forward takes :meth:`_forward_gen_q_fused` (``TRTLLM_K3_MLA_Q_FUSED``):
        a generation-only Helix batch of an FP4-KV layer whose query fits the fused kernel."""
        if not _K3_MLA_Q_FUSED:
            return False
        num_tokens = attn_metadata.num_tokens
        if (
            type(hidden_states) is not torch.Tensor
            or hidden_states.dim() != 2
            or hidden_states.shape[0] != num_tokens
            or attn_metadata.num_contexts != 0
            or attn_metadata.num_generations <= 0
            or num_tokens % attn_metadata.num_generations != 0
            or self.sparse_attn_hooks is not None
            or self.is_lite
            or not self.fuse_qkv_a_proj
            or self.apply_rotary_emb
            or self.llama_4_scaling
            or not self._helix_v_up_enabled()
            or not isinstance(self.mqa, TrtllmAttention)
            or not self.mqa.uses_fp4_mla_attention
            or getattr(attn_metadata, "helix_position_offsets", None) is None
            or getattr(attn_metadata, "helix_is_inactive_rank", None) is None
            or self.mqa.rotary_cos_sin is None
            or self.mqa.rotary_cos_sin.dtype != torch.float32
            or not self.mqa.rotary_cos_sin.is_contiguous()
        ):
            return False
        from ..kernels import k3_mla_q_fused

        if not (
            self.num_heads_tp == k3_mla_q_fused.NUM_HEADS
            and self.q_lora_rank == k3_mla_q_fused.Q_LORA_RANK
            and self.kv_lora_rank == k3_mla_q_fused.KV_LORA_RANK
            and self.qk_nope_head_dim == k3_mla_q_fused.NOPE_DIM
            and self.qk_rope_head_dim == k3_mla_q_fused.ROPE_DIM
            and num_tokens <= min(k3_mla_q_fused.MAX_TOKENS, _K3_MLA_Q_FUSED_MAX_TOKENS)
        ):
            return False
        weight = getattr(self.q_b_proj, "weight", None)
        w_uk = self._q_fused_w_uk()
        if not isinstance(weight, torch.Tensor) or w_uk is None:
            return False
        if not k3_mla_q_fused.weights_supported(weight, w_uk):
            return False
        return self._q_fused_qb_exponents() is not None

    def _q_fused_latent_row(self, compressed_kv: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
        """``_latent_row`` in one ``k3_mla_latent_row`` launch where the norm is FlashInfer's CuTe
        DSL kernel (whose arithmetic it repeats, bit for bit)."""
        from ..kernels import k3_mla_q_fused

        norm = self.kv_a_layernorm
        if (
            self._runs_flashinfer_rmsnorm(norm)
            and self._flashinfer_cute_norm(compressed_kv.device)
            and k3_mla_q_fused.latent_row_supported(compressed_kv, k_pe, norm.weight)
        ):
            return k3_mla_q_fused.k3_mla_latent_row(
                compressed_kv,
                k_pe,
                norm.weight,
                norm.variance_epsilon,
                use_pdl=get_env_enable_pdl(),
            )
        return self._latent_row(compressed_kv, k_pe)

    def _forward_gen_q_fused(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        """Absorbed Helix generation of an FP4-KV layer with the fused query
        (``TRTLLM_K3_MLA_Q_FUSED``); returns the combined all-to-all output
        (``num_tokens x num_heads_tp_cp * v_head_dim``).

        After the fused q_a / kv_a projection the KV half (kv_a RMSNorm and ``k_pe`` into the
        latent row in one ``k3_mla_latent_row`` launch, then the FP4 generation cache update
        without its query rows, ``TrtllmAttention.fp4_mla_kv_append_generation``) runs on the aux
        stream (with multi-stream on), and the main
        stream runs ``k3_mla_q_fused`` (on the raw q_a rows with the RMSNorm inside, or after the
        separate q_a RMSNorm), which writes the FP4 query rows into the Q buffers the cache update
        recorded on the metadata. The decode joins both. The decode reads only those buffers; the
        BF16 ``fused_q`` it is handed provides shapes.
        """
        from tensorrt_llm._torch.attention.backends.fp4_mla import (
            _get_fp4_mla_q_global_scale,
            _prepare_fp4_mla_q_buffers,
            mark_fp4_mla_query_launch,
            ull2_mla_decode_pdl_enabled,
        )
        from tensorrt_llm._torch.attention.backends.fp4_mla.config import (
            _fp4_mla_packed_rows_per_seq,
        )

        from ..kernels import k3_mla_q_fused

        md = attn_metadata
        num_tokens = md.num_tokens
        heads, lora, rope = self.num_heads_tp, self.kv_lora_rank, self.qk_rope_head_dim
        device = hidden_states.device
        gen_len = num_tokens // md.num_generations
        # The Q buffers the KV half records on the metadata, created (zero-filled, once) on this
        # stream, which writes them.
        _prepare_fp4_mla_q_buffers(
            md, num_tokens, heads, device, rows_per_seq=_fp4_mla_packed_rows_per_seq(heads, gen_len)
        )
        q, compressed_kv, k_pe = self._project_qkv_a(hidden_states).split(
            [self.q_lora_rank, lora, rope], -1
        )
        multi_stream = do_multi_stream() and self.aux_stream is not None
        kv_fork, kv_ready = self._kv_events
        kv_stream = None
        if multi_stream:
            kv_stream = (
                _gate_side_stream(device, high_priority=True)
                if _K3_MLA_Q_FUSED_KV_PRIO
                else self.aux_stream
            )
            kv_fork.record()
        with torch.cuda.stream(kv_stream) if kv_stream is not None else contextlib.nullcontext():
            if kv_stream is not None:
                kv_fork.wait()
            latent_cache = self._q_fused_latent_row(compressed_kv, k_pe)
            self.mqa.fp4_mla_kv_append_generation(latent_cache, md, heads)
            if kv_stream is not None:
                kv_ready.record()
        norm = self.q_a_layernorm
        fuse_norm = (
            _K3_MLA_Q_FUSED_NORM
            and k3_mla_q_fused.NORM_FUSION_AVAILABLE
            and self._runs_flashinfer_rmsnorm(norm)
            and self._flashinfer_cute_norm(device)
            and self._resolve_qa_fused_scale() is None
            and norm.weight.dtype == torch.bfloat16
            and norm.weight.is_contiguous()
            and k3_mla_q_fused.input_supported(q)
        )
        if not fuse_norm:
            q = self._q_a_layernorm_maybe_fused(q)
        assert k3_mla_q_fused.input_supported(q), "the fused query reads BF16 q_a rows"
        prequantized_q, prequantized_q_sf, rows_per_seq = _fp4_mla_prequantized_q(md)
        k3_mla_q_fused.k3_mla_q_fused(
            q,
            self.q_b_proj.weight,
            self._q_fused_qb_exponents(),
            self._q_fused_w_uk(),
            _get_fp4_mla_q_global_scale(md, device),
            self.mqa.rotary_cos_sin,
            md.helix_position_offsets,
            prequantized_q,
            prequantized_q_sf.view(torch.uint8),
            gen_len=gen_len,
            rows_per_seq=rows_per_seq or gen_len * heads,
            use_pdl=get_env_enable_pdl(),
            norm=(norm.weight, norm.variance_epsilon) if fuse_norm else None,
            # TRTLLM_K3_ULL2_MLA_DEC_PDL: the decode (a programmatic dependent reading only the
            # query after its wait) starts its prologue as this kernel's CTAs finish.
            early_trigger=ull2_mla_decode_pdl_enabled(),
        )
        mark_fp4_mla_query_launch(md)
        if kv_stream is not None:
            kv_ready.wait()
        fused_q = torch.empty([num_tokens, heads, lora + rope], dtype=q.dtype, device=device)
        q_pe = fused_q[..., lora:]
        num_seqs = md.num_seqs
        attn_out = self._attn_forward_gen(
            self.mqa,
            fused_q.view([num_tokens, heads * (lora + rope)]),
            None,
            None,
            None,
            md,
            helix_v_up=True,
            attention_input_type=AttentionInputType.generation_only,
            out_scale=self.out_scale,
            latent_cache=latent_cache,
            q_pe=q_pe,
            cu_q_seqlens=torch.empty(num_seqs + 1, dtype=torch.int32, device=device),
            cu_kv_seqlens=torch.empty(num_seqs + 1, dtype=torch.int32, device=device),
            fmha_scheduler_counter=torch.empty(1, dtype=torch.uint32, device=device),
            mla_bmm1_scale=torch.empty(2, dtype=torch.float32, device=device),
            mla_bmm2_scale=torch.empty(1, dtype=torch.float32, device=device),
            quant_q_buffer=torch.empty(
                num_tokens, heads, lora + rope, dtype=torch.uint8, device=device
            ),
        )
        assert attn_out.shape == (num_tokens, self.num_heads_tp_cp * self.v_head_dim)
        return attn_out

    def _forward_side_gate(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        """Generation forward with the zero-KV mask and the output gate on a side stream.

        The mask (one small kernel) is forked once the layer input is written; the gate (one
        GEMM with a sigmoid epilogue) is forked when the all-to-all is enqueued, so it runs on
        the SMs the all-to-all leaves idle instead of beside the q_a / q_b GEMMs. The main
        stream waits for the mask where the all-to-all reads it and for the gate just before
        o_proj, where both are long done. With the fused Helix tail
        (``TRTLLM_K3_HELIX_TAIL_FUSED``, ``TRTLLM_K3_HELIX_TAIL_GATE=early``) the gate is forked
        before the decode and joined before the tail kernel, so o_proj's only dependency is the
        tail.
        """
        num_tokens = hidden_states.shape[0]
        mask_fork, mask_ready, gate_fork, gate_ready = self._gate_events
        multi_stream = do_multi_stream()
        side = _gate_side_stream(hidden_states.device) if multi_stream else None
        gates = []
        mask_joined = [not multi_stream]
        gate_joined = [not multi_stream]

        def gate_of(x):
            forward_sigmoid = getattr(self.g_proj, "forward_sigmoid", None)
            if forward_sigmoid is not None:
                return forward_sigmoid(x)
            return self.g_proj(x).sigmoid()

        if multi_stream:
            mask_fork.record()
            with torch.cuda.stream(side):
                mask_fork.wait()
                zero_kv_mask = helix_gen_zero_kv_mask(attn_metadata, num_tokens)
                mask_ready.record()
        else:
            zero_kv_mask = helix_gen_zero_kv_mask(attn_metadata, num_tokens)

        def zero_kv_mask_fn():
            if not mask_joined[0]:
                mask_ready.wait()
                mask_joined[0] = True
            return zero_kv_mask

        def pre_exchange():
            if gates:
                # Already forked (TRTLLM_K3_ULL3_MLA_GATE_AT=start).
                return
            if multi_stream:
                gate_fork.record()
                with torch.cuda.stream(side):
                    gate_fork.wait()
                    gates.append(gate_of(hidden_states))
                    gate_ready.record()
            else:
                gates.append(gate_of(hidden_states))

        def pre_tail():
            if gates and not gate_joined[0]:
                gate_ready.wait()
                gate_joined[0] = True
            return gates[0] if gates else None

        self._ull3_gate_place = ull3_gate_place(num_tokens) if multi_stream else "decode"
        if self._ull3_gate_place == "start" and _K3_HELIX_TAIL_GATE_EARLY:
            pre_exchange()
        attn_outputs = None
        self.helix_zero_kv_mask_fn = zero_kv_mask_fn
        self.helix_pre_exchange_fn = pre_exchange
        self.helix_pre_tail_fn = pre_tail
        self.helix_tail_mx = None
        self.defer_helix_gen_output = True
        try:
            attn_out = None
            if self._q_fused_applies(hidden_states, attn_metadata):
                attn_out = self._forward_gen_q_fused(hidden_states, attn_metadata)
            else:
                attn_outputs = self._create_outputs(hidden_states, attn_metadata)
                super().forward_impl(None, hidden_states, attn_metadata, attn_output=attn_outputs)
                attn_out = self.helix_gen_output
        finally:
            self._ull3_gate_place = "decode"
            self.helix_zero_kv_mask_fn = None
            self.helix_pre_exchange_fn = None
            self.helix_pre_tail_fn = None
            self.defer_helix_gen_output = False
            self.helix_gen_output = None
        if attn_out is None:
            attn_out = attn_outputs[0]
        if not mask_joined[0]:
            mask_ready.wait()
        if gates:
            gate = gates[0]
            if not gate_joined[0]:
                gate_ready.wait()
        else:
            gate = gate_of(hidden_states)
        mx, self.helix_tail_mx = self.helix_tail_mx, None
        if mx is not None:
            if mx[1].dim() == 1:
                # TRTLLM_K3_MID_OPROJ_PC: the mid-M layout.
                return self.o_proj.forward_mid_mxfp8(attn_out, *mx)
            return self.o_proj.forward_gated_mxfp8(attn_out, gate, *mx)
        if self.o_proj.gated_applies(attn_out, gate):
            return self.o_proj.forward_gated(attn_out, gate)
        return self.o_proj(attn_out * gate)

    def prelaunch_step_metadata(self, attn_metadata: AttentionMetadata) -> bool:
        """``TRTLLM_K3_ULL4_MLA_PLAN_EARLY``: launch, on the current stream, the step metadata the
        first FP4 MLA generation kernel would write (generation lengths fused with the packed
        split plan of this module's query heads, then the device page table), so that the step's
        first FP4 MLA layer finds them current. Returns whether anything was launched; the caller
        sets :attr:`step_metadata_ready` to an event recorded after it."""
        mqa = getattr(self, "mqa", None)
        if not isinstance(mqa, TrtllmAttention) or not mqa.uses_fp4_mla_attention:
            return False
        if attn_metadata.num_contexts != 0:
            return False
        from tensorrt_llm._torch.attention.backends.fp4_mla import (
            prelaunch_fp4_mla_generation_metadata,
        )

        use_helix = (
            getattr(attn_metadata, "helix_position_offsets", None) is not None
            or getattr(attn_metadata, "helix_is_inactive_rank", None) is not None
        )
        return prelaunch_fp4_mla_generation_metadata(
            attn_metadata,
            num_heads=self.num_heads_tp,
            use_helix=use_helix,
            device=self.o_proj.weight.device,
        )

    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
        latent_cache_gen: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``MLA.forward``'s signature (K3 is NoPE: callers pass ``position_ids=None``).

        Helix with attention DP (main's helix_a2a: the input is all-gathered and o_proj's output
        reduce-scattered) and an ``Fp4QuantizedTensor`` input take main's ``MLA.forward``, with
        the output gate as its ``_apply_output_gate`` hook; the Rubin branch has neither. Every
        other forward takes the Rubin branch's paths (:meth:`_forward`).
        """
        ready = self.step_metadata_ready
        if ready is not None:
            # TRTLLM_K3_ULL4_MLA_PLAN_EARLY: the step metadata written on the model's side stream.
            self.step_metadata_ready = None
            torch.cuda.current_stream().wait_event(ready)
        try:
            if isinstance(hidden_states, Fp4QuantizedTensor) or (
                self.mapping.has_cp_helix() and self.mapping.enable_attention_dp
            ):
                # The MXFP8 rows a decoder layer hands over are for its rows, not the gathered ones.
                self.next_input_mxfp8 = None
                return super().forward(
                    position_ids,
                    hidden_states,
                    attn_metadata,
                    all_reduce_params=all_reduce_params,
                    latent_cache_gen=latent_cache_gen,
                )
            return self._forward(
                position_ids, hidden_states, attn_metadata, all_reduce_params, latent_cache_gen
            )
        finally:
            # consumed by the projection; never carried into another forward
            self.next_input_mxfp8 = None

    def _run_attention(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        latent_cache_gen: Optional[torch.Tensor],
    ) -> list:
        """The attention half of main's ``MLA.forward``: the registered custom op (so a breakable
        CUDA graph breaks around it), else ``forward_impl`` eagerly. Returns the output list."""
        if self.register_to_config:
            attn_outputs = [torch.ops.trtllm.create_mla_outputs(hidden_states, self.layer_idx_str)]
            self._forward_custom_op(hidden_states, position_ids, attn_outputs, latent_cache_gen)
        else:
            # _create_outputs() rather than create_output(): the base implementation
            # takes a list so a sparse-attention backend can append its own buffers,
            # and it routes through the sparse hooks when they are installed. The
            # dense path this module uses is element 0.
            attn_outputs = self._create_outputs(hidden_states, attn_metadata)
            self.forward_impl(
                position_ids,
                hidden_states,
                attn_metadata,
                attn_output=attn_outputs,
                latent_cache_gen=latent_cache_gen,
            )
        return attn_outputs

    def _forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
        latent_cache_gen: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._side_gate_applies(hidden_states, attn_metadata):
            return self._forward_side_gate(hidden_states, attn_metadata)
        if self._q_fused_applies(hidden_states, attn_metadata):
            attn_out = self._forward_gen_q_fused(hidden_states, attn_metadata)
            return self._apply_output_gate_and_o_proj(
                hidden_states, attn_out, position_ids, attn_metadata, all_reduce_params
            )
        attn_outputs = self._run_attention(
            position_ids, hidden_states, attn_metadata, latent_cache_gen
        )
        return self._apply_output_gate_and_o_proj(
            hidden_states, attn_outputs[0], position_ids, attn_metadata, all_reduce_params
        )
