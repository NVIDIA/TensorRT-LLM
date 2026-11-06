# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 specialization of the shared PyTorch-backend MLA module.

The base ``MLA`` class owns context attention, cached/chunked prefill,
absorbed generation, and paged-cache handling. This module only supplies the
K3 projection topology, NoPE identity table, KV-B checkpoint layout, and
gated output projection.
"""

from __future__ import annotations

import copy
import os
from typing import Optional

import torch

from ...._utils import get_env_flag
from ....functional import PositionEmbeddingType
from ....logger import logger
from ....mapping import Mapping
from ....models.modeling_utils import QuantConfig
from ....quantization import QuantAlgo
from ...attention_backend import AttentionMetadata, TrtllmAttention
from ...attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ...model_config import ModelConfig
from ...pyexecutor.breakable_cuda_graph import eager_on_graph, is_in_breakable_cuda_graph
from ...utils import AuxStreamType
from ..attention import extract_extra_attrs
from ..linear import Linear, TensorParallelMode
from ..mla import MLA
from ..multi_stream_utils import maybe_execute_in_parallel

_KIMI_K3_MLA_GEN_BACKEND_ENV = "TLLM_K3_MLA_GEN_BACKEND"

# TRTLLM_K3_MLA_GATE_FP8=1: when o_proj takes a prequantized FP8 1x128 input (the K3 FP8
# weight-read projection on Rubin), the output gate emits that operand directly
# (output_gate_fp8.gate_mul_fp8_r128c4: sigmoid, multiply and quantization in one kernel, the
# same bytes as quantizing attn_out * sigmoid(gate)), and o_proj skips its quantize kernel.
_K3_MLA_GATE_FP8 = bool(get_env_flag("TRTLLM_K3_MLA_GATE_FP8", False))
# TRTLLM_K3_MLA_GATE_OVERLAP=1: under CUDA graphs, g_proj (which only reads hidden_states) runs on
# a side stream while the attention runs instead of after it. Same kernels and inputs.
_K3_MLA_GATE_OVERLAP = bool(get_env_flag("TRTLLM_K3_MLA_GATE_OVERLAP", False))
# TRTLLM_K3_MLA_FUSE_GATE_A=1: g_proj runs inside one [g_proj; fused q_a/kv_a] GEMM before the
# attention (built by the model after weight load: modeling_kimi_linear._fuse_mla_gate_and_a_proj,
# which reads the same variable). It and TRTLLM_K3_MLA_GATE_OVERLAP take g_proj off the critical
# path in two incompatible ways, so setting both is an error (_check_gate_flags).
# TRTLLM_K3_MLA_GATE_FP8 only changes how the gate is applied and composes with either.
_K3_MLA_FUSE_GATE_A_ENV = "TRTLLM_K3_MLA_FUSE_GATE_A"


# TRTLLM_K3_MLA_FUSED_QKV=1: a decode batch of at most 32 tokens builds its FP8 attention query and
# its new KV rows in one kernel after the fused [g_proj; q_a/kv_a] GEMM
# (cute_dsl_kernels/rubin/k3_mla_fused_qkv.py: q_a and kv_a RMSNorm, the 1x128 FP8 quantization,
# q_b_proj with the in-cluster K split of its GEMM tactic, the FP8 absorption, the KV append with the
# scale and scheduler buffers); the same bytes as the stock chain of norm, quantize, GEMM, concat,
# absorption and RoPE / KV-append kernels. Other batches (context, larger decode batches, a q_b
# tactic without an in-cluster K split, other KV / attention configurations) take the stock path.
_K3_MLA_FUSED_QKV = bool(get_env_flag("TRTLLM_K3_MLA_FUSED_QKV", False))


def _qb_gemm_k_splits(q_b_proj: torch.nn.Module, num_tokens: int) -> Optional[int]:
    """The K partition of the GEMM tactic ``q_b_proj`` would run for ``num_tokens`` tokens: 1 for
    the unsplit base kernel (and the untuned default), ``cluster_k`` for an in-cluster K split, None
    for any other tactic (mixed clusters, split-K, B-stationary), which the fused kernel does not
    reproduce.

    Makes the same ``AutoTuner.choose_one`` call as ``cute_dsl_mxfp8_gemm_rubin`` with
    ``fine_grained_m`` (so under ``autotune()`` it tunes the GEMM exactly as the stock path would).
    """
    from ...autotuner import AutoTuner
    from ...custom_ops import cute_dsl_custom_ops as cdo

    weight = q_b_proj.weight
    k = weight.shape[1]
    padded_m = (num_tokens + 127) // 128 * 128
    act = torch.empty((num_tokens, k), dtype=torch.float8_e4m3fn, device=weight.device)
    act_scale = torch.empty((padded_m * (k // 32),), dtype=torch.uint8, device=weight.device)
    runner = cdo.CuteDSLKimiK3MXFP8RubinLinear(output_dtype=torch.bfloat16, use_tvm_ffi=True)
    inputs = [
        act,
        weight,
        act_scale,
        q_b_proj.weight_scale,
        cdo._get_mxfp8_gemm_alpha(weight.device),
    ]
    _, tactic = AutoTuner.get().choose_one(
        "trtllm::cute_dsl_mxfp8_gemm_rubin", [runner], runner.__class__.tuning_config, inputs
    )
    if not isinstance(tactic, tuple) or not tactic:
        return 1  # the runner's default: the unsplit base kernel
    if tactic[0] == "base":
        return 1
    if tactic[0] == "streamk" and len(tactic) == 9 and isinstance(tactic[8], int):
        return tactic[8]
    return None


def _check_gate_flags() -> None:
    """Reject ``TRTLLM_K3_MLA_FUSE_GATE_A=1`` together with ``TRTLLM_K3_MLA_GATE_OVERLAP=1``."""
    if _K3_MLA_GATE_OVERLAP and get_env_flag(_K3_MLA_FUSE_GATE_A_ENV, False):
        raise ValueError(
            "TRTLLM_K3_MLA_FUSE_GATE_A=1 and TRTLLM_K3_MLA_GATE_OVERLAP=1 are alternatives: the "
            "first fuses g_proj into the q_a/kv_a GEMM before the attention, the second runs it "
            "on a side stream during the attention. Set at most one of them; "
            "TRTLLM_K3_MLA_GATE_FP8=1 works with either."
        )


def k3_mla_attention_inplace(
    hidden_states: torch.Tensor,
    layer_idx: str,
    output: torch.Tensor,
    fused_a_output: Optional[torch.Tensor] = None,
) -> None:
    """Run the shared MLA attention of one K3 layer eagerly into ``output``.

    The eager break of the breakable prefill CUDA graph: the attention reads the live batch
    metadata and writes the first ``attn_metadata.num_tokens`` rows of ``output``.
    """
    metadata, mla_layer = extract_extra_attrs(layer_idx, "mla")
    mla_layer.forward_impl(
        None,
        hidden_states,
        metadata,
        attn_output=[output],
        fused_a_output=fused_a_output,
    )


maybe_bcg_k3_mla_attention_inplace = eager_on_graph(k3_mla_attention_inplace)


# Deliberate divergence from upstream, which defaults this to "cute-dsl".
#
# On the FlashInfer/CuTe-DSL pair this branch currently ships, the CuTe-DSL MLA
# decode path cannot initialize at all. flashinfer 0.6.15 unpacks the MLA
# scheduler params one MLIR value per field
# (flashinfer/cute_dsl/attention/monolithic/mla_helpers.py), while the installed
# nvidia-cutlass-dsl serializes FastDivmodDivisor as two SSA values (cutlass
# issue #3243). Every rank dies during executor init with
#
#     ValueError: FastDivmodDivisor expects exactly 2 MLIR values
#                 (encoded divisor + scalar divisor SSA), got 1.
#
# which gives no hint that the MLA backend selection is what caused it.
#
# This is a package-pairing problem, not an architecture limit: nothing on that
# path inspects the SM version. So this is a default flip rather than a hard
# block -- set TLLM_K3_MLA_GEN_BACKEND=cute-dsl to opt back in wherever the
# FlashInfer and CuTe-DSL builds do agree. Restore the upstream default once the
# shipped pair is aligned.
_KIMI_K3_MLA_GEN_BACKEND_DEFAULT = "trtllm-gen"


def _select_mla_generation_backend(quant_config: Optional[QuantConfig]) -> str:
    """Select K3's absorbed-generation MLA backend.

    FP8 KV cache requires the native CuTe-DSL wrapper, which accepts its
    device scales; TRTLLM-Gen does not support K3's 96 query heads. Other
    KV-cache dtypes keep the existing configured/default backend selection
    unchanged.

    The non-FP8 default is TRTLLM-Gen on this branch; see
    ``_KIMI_K3_MLA_GEN_BACKEND_DEFAULT`` for why.
    """
    backend = os.environ.get(_KIMI_K3_MLA_GEN_BACKEND_ENV, _KIMI_K3_MLA_GEN_BACKEND_DEFAULT)
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


class KimiK3MLAAttention(MLA):
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
        # Before any allocation: a conflicting gate configuration fails at model build.
        _check_gate_flags()
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
            flashinfer_mla_backend=_select_mla_generation_backend(quant_config),
        )
        # K3 calls forward_impl() directly to insert its output gate before
        # the base row-parallel o_proj. The original executor metadata remains
        # intact, so MLA performs its native mixed context/generation split.
        self.register_to_config = False

        self.use_output_gate = use_output_gate
        # g_proj's side stream (TRTLLM_K3_MLA_GATE_OVERLAP): the MoE latent-down stream, idle
        # during attention and forked from the main stream again by the MoE after this join.
        self._gate_stream = (
            aux_stream_dict.get(AuxStreamType.MoeOutputMemset)
            if use_output_gate and _K3_MLA_GATE_OVERLAP
            else None
        )
        self._gate_fork_event = torch.cuda.Event()
        self._gate_join_event = torch.cuda.Event()

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
        # Only the absorbed-generation backend (mqa) requests CuTe-DSL, so
        # only it needs K3's per-batch fallback policy. FP8 KV must keep its
        # required CuTe-DSL backend because TRTLLM-Gen does not support K3's
        # 96 query heads, so leave the policy unset and preserve the static
        # backend for every batch composition. Use the config-derived flag:
        # TrtllmAttention has not populated has_fp8_kv_cache yet when weight
        # creation is deferred under MetaInitMode.
        if not has_fp8_kv_cache:
            self.mqa.mla_backend_policy = _kimi_k3_mla_decode_backend_policy
        self.rotary_emb = None
        self.apply_rotary_emb = False
        # NoPE: with an FP8 KV cache, build the FP8 decode query in one kernel
        # instead of a BF16 absorption BMM plus its FP8 re-read in the RoPE
        # kernel (opt-in: TRTLLM_K3_MLA_FP8_Q_ABSORB=1).
        self.fp8_nope_q_absorb = os.environ.get("TRTLLM_K3_MLA_FP8_Q_ABSORB", "0") == "1"
        # [g_proj; kv_a_proj_with_mqa] stacked into one projection of the layer input
        # (one activation quant + one GEMM instead of two each). Built after weight load
        # by the model when TRTLLM_K3_MLA_FUSE_GATE_A=1; the two projections then hold row
        # views of its weight.
        self.g_fused_a_proj = None

    def _o_proj_takes_prequantized(self, attn_out: torch.Tensor, gate: torch.Tensor) -> bool:
        """Whether ``o_proj.forward_prequantized`` can take the fused gate's FP8 output."""
        return (
            bool(getattr(self.o_proj, "supports_prequantized_input", False))
            and callable(getattr(self.o_proj, "forward_prequantized", None))
            and attn_out.dim() == 2
            and gate.shape == attn_out.shape
            and attn_out.dtype == torch.bfloat16
            and gate.dtype == torch.bfloat16
            and attn_out.shape[0] > 0
            and attn_out.shape[1] % 128 == 0
            and attn_out.stride(1) == 1
            and gate.stride(1) == 1
        )

    def _apply_output_gate_and_o_proj(
        self,
        hidden_states: torch.Tensor,
        attn_out: torch.Tensor,
        gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``o_proj(attn_out * sigmoid(g_proj(hidden_states)))``; ``gate``: g_proj's output if
        already computed."""
        if not self.use_output_gate:
            return self.o_proj(attn_out)
        if gate is None:
            gate = self.g_proj(hidden_states)
        if _K3_MLA_GATE_FP8 and self._o_proj_takes_prequantized(attn_out, gate):
            from .output_gate_fp8 import gate_mul_fp8_r128c4

            return self.o_proj.forward_prequantized(*gate_mul_fp8_r128c4(attn_out, gate))
        return self.o_proj(attn_out * gate.sigmoid())

    def _fused_qkv_norms_supported(self, device: torch.device) -> bool:
        """Both latent RMSNorms run FlashInfer's CuTe DSL kernel, whose arithmetic the fused kernel
        reproduces (no cuda-tile / Gemma / fused NVFP4 variant, no CUDA-JIT fallback)."""
        cached = getattr(self, "_fused_qkv_norms_ok", None)
        if cached is not None:
            return cached
        from ...flashinfer_utils import IS_FLASHINFER_AVAILABLE
        from ...utils import is_nvfp4_marlin_enabled

        ok = (
            IS_FLASHINFER_AVAILABLE
            and not is_nvfp4_marlin_enabled()
            and self._resolve_qa_fused_scale() is None
        )
        for norm in (self.q_a_layernorm, self.kv_a_layernorm):
            ok = ok and not (
                getattr(norm, "use_cuda_tile", False)
                or getattr(norm, "use_gemma", False)
                or getattr(norm, "return_hp_output", False)
                or (
                    getattr(norm, "is_nvfp4", False)
                    and getattr(norm, "nvfp4_scale", None) is not None
                )
                or norm.weight.dtype != torch.bfloat16
            )
        if ok:
            try:
                from flashinfer.norm import _use_cuda_norm
            except ImportError:
                ok = False
            else:
                ok = not _use_cuda_norm(device)
        self._fused_qkv_norms_ok = ok
        return ok

    def _fused_qkv_plan(
        self, attn_metadata: AttentionMetadata, fused_a_output: Optional[torch.Tensor]
    ) -> Optional[tuple]:
        """``(k_splits, kv_view)`` for ``TRTLLM_K3_MLA_FUSED_QKV`` on this batch (the q_b K partition
        and ``_kv_page_view``), or None to run the stock path (see ``_K3_MLA_FUSED_QKV``)."""
        from ...cute_dsl_kernels.rubin import k3_mla_fused_qkv as fqkv

        num_tokens = attn_metadata.num_tokens
        num_gens = attn_metadata.num_generations
        mqa = self.mqa
        if not (
            fused_a_output is not None
            and attn_metadata.num_contexts == 0
            and 0 < num_gens
            and 0 < num_tokens <= fqkv.MAX_TOKENS
            and num_tokens % num_gens == 0
            and getattr(attn_metadata, "beam_width", 1) == 1
            and self.fp8_nope_q_absorb
            and getattr(mqa, "has_fp8_kv_cache", False)
            and not getattr(mqa, "uses_fp4_mla_attention", False)
            # The CuTe DSL FP8 decode, which reads only quant_q_buffer and the scheduler buffers.
            and getattr(mqa, "flashinfer_mla_backend", None) == "cute-dsl"
            and getattr(mqa, "mla_backend_policy", None) is None
            and getattr(mqa, "rotary_cos_sin", None) is not None
            and mqa.rotary_cos_sin.dtype == torch.float32
            and self.sparse_attn_hooks is None
            and not self.mapping.has_cp_helix()
            and not self.apply_rotary_emb
            and not self.llama_4_scaling
            and self.num_heads_tp == fqkv.NUM_HEADS
            and self.q_lora_rank == fqkv.Q_LORA
            and self.kv_lora_rank == fqkv.KV_LORA
            and self.qk_nope_head_dim == fqkv.NOPE_DIM
            and self.qk_rope_head_dim == fqkv.ROPE_DIM
            and getattr(self, "k_b_proj_trans_dequant", None) is not None
            and self.k_b_proj_trans_dequant.dtype == torch.bfloat16
            and tuple(self.k_b_proj_trans_dequant.shape)
            == (fqkv.NUM_HEADS, fqkv.KV_LORA, fqkv.NOPE_DIM)
            and self.k_b_proj_trans_dequant.is_contiguous()
            and self.k_b_proj_trans_dequant.data_ptr() % 16 == 0
            and getattr(self.q_b_proj, "supports_prequantized_input", False)
            and getattr(self.q_b_proj, "weight_scale", None) is not None
            and self.q_b_proj.weight_scale.dtype == torch.uint8
            and getattr(attn_metadata.kv_cache_manager, "tokens_per_block", None)
            == fqkv.PAGE_TOKENS
            and self._fused_qkv_norms_supported(fused_a_output.device)
        ):
            return None
        kv_view = self._kv_page_view(attn_metadata)
        if kv_view is None:
            return None
        k_splits = _qb_gemm_k_splits(self.q_b_proj, num_tokens)
        if k_splits not in fqkv.SUPPORTED_K_SPLITS:
            return None
        return k_splits, kv_view

    def _kv_page_view(self, attn_metadata: AttentionMetadata) -> Optional[tuple]:
        """``(byte address of page 0, page byte stride, page table, layer slot)`` of this layer's FP8
        KV pool, addressed as the CuTe DSL MLA decode addresses it
        (``CuteDslMlaFmha._run_mla_decode``), or None for a pool layout the fused kernel does not
        write (it needs pages of 64 rows of 576 contiguous bytes)."""
        from ...cute_dsl_kernels.rubin import k3_mla_fused_qkv as fqkv

        kv_pool = attn_metadata.kv_cache_manager.get_buffers(self.mqa.layer_idx)
        if (
            kv_pool.element_size() != 1
            or kv_pool.dim() != 5
            or tuple(kv_pool.shape[1:]) != (1, fqkv.PAGE_TOKENS, 1, fqkv.KV_ROW)
            or kv_pool.stride(4) != 1
            or kv_pool.stride(2) != fqkv.KV_ROW
        ):
            return None
        packed_block = fqkv.PAGE_TOKENS * fqkv.KV_ROW
        block_stride = kv_pool.stride(0)
        layers_in_pool = block_stride // packed_block
        if layers_in_pool > 1 and block_stride == layers_in_pool * packed_block:
            # One pool interleaving several layers: pages of the combined-slot view, this layer's
            # slot added to the page ids (as the CuTe DSL decode folds it into its page table).
            layer_in_pool = kv_pool.storage_offset() // packed_block
            base = kv_pool.data_ptr() - kv_pool.storage_offset() * kv_pool.element_size()
            page_bytes = packed_block
        else:
            layer_in_pool = 0
            base = kv_pool.data_ptr()
            page_bytes = block_stride
        local_layer_idx = self.mqa.get_local_layer_idx(attn_metadata)
        pool_idx = int(attn_metadata.host_kv_cache_pool_mapping[local_layer_idx, 0])
        page_table = attn_metadata.kv_cache_block_offsets[
            pool_idx, attn_metadata.num_contexts :, 0, :
        ]
        if page_table.dtype != torch.int32 or page_table.stride(-1) != 1:
            return None
        return base, page_bytes, page_table, layer_in_pool

    def _forward_fused_qkv(
        self,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        fused_a_output: torch.Tensor,
        plan: tuple,
    ) -> None:
        """Decode attention with the FP8 query and the KV rows from ``k3_mla_fused_qkv``."""
        from ...cute_dsl_kernels.rubin import k3_mla_fused_qkv as fqkv

        num_tokens = attn_metadata.num_tokens
        num_seqs = attn_metadata.num_seqs
        q_len = num_tokens // attn_metadata.num_generations
        fused_a = fused_a_output[:num_tokens]
        device = fused_a.device
        mqa = self.mqa
        mqa._ensure_rope_table_size(attn_metadata.max_seq_len)
        k_splits, (kv_base, kv_page_bytes, page_table, layer_in_pool) = plan
        quant_q_buffer = torch.empty(
            (num_tokens, self.num_heads_tp, self.kv_lora_rank + self.qk_rope_head_dim),
            dtype=torch.uint8,
            device=device,
        )
        cu_q_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)
        cu_kv_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)
        fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=device)
        mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=device)
        mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=device)
        fqkv.run_k3_mla_fused_qkv(
            fused_a,
            self.q_a_layernorm.weight,
            self.kv_a_layernorm.weight,
            self.q_a_layernorm.variance_epsilon,
            self.kv_a_layernorm.variance_epsilon,
            self.q_b_proj.weight,
            self.q_b_proj.weight_scale,
            self.k_b_proj_trans_dequant,
            quant_q_buffer,
            kv_base,
            kv_page_bytes,
            page_table,
            layer_in_pool,
            attn_metadata.kv_lens_cuda_runtime[:num_seqs],
            mqa.rotary_cos_sin,
            mla_bmm1_scale,
            mla_bmm2_scale,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            fqkv.host_bmm1_scale(mqa.q_scaling, self.qk_nope_head_dim + self.qk_rope_head_dim),
            q_len,
            k_splits,
        )
        # The FP8 decode reads quant_q_buffer; the BF16 query only carries its shape.
        fused_q = torch.empty(
            (num_tokens, self.num_heads_tp * (self.kv_lora_rank + self.qk_rope_head_dim)),
            dtype=torch.bfloat16,
            device=device,
        )
        self._attend_absorbed_generation(
            fused_q,
            attn_metadata,
            output[:num_tokens],
            position_ids=None,
            latent_cache=None,
            q_pe=None,
            cu_q_seqlens=cu_q_seqlens,
            cu_kv_seqlens=cu_kv_seqlens,
            fmha_scheduler_counter=fmha_scheduler_counter,
            mla_bmm1_scale=mla_bmm1_scale,
            mla_bmm2_scale=mla_bmm2_scale,
            quant_q_buffer=quant_q_buffer,
        )

    def _forward_impl(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache_gen: Optional[torch.Tensor] = None,
        fused_a_output: Optional[torch.Tensor] = None,
    ) -> None:
        if _K3_MLA_FUSED_QKV and latent_cache_gen is None:
            plan = self._fused_qkv_plan(attn_metadata, fused_a_output)
            if plan is not None:
                self._forward_fused_qkv(attn_metadata, output, fused_a_output, plan)
                return
        super()._forward_impl(
            position_ids,
            hidden_states,
            attn_metadata,
            output,
            latent_cache_gen=latent_cache_gen,
            fused_a_output=fused_a_output,
        )

    def _run_attention(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attn_outputs: list[torch.Tensor],
        fused_a_output: Optional[torch.Tensor] = None,
    ) -> None:
        """Run the shared MLA attention into ``attn_outputs``: an eager break in a breakable graph.

        A prefill CUDA graph batch has ``hidden_states`` padded to the graph's token count, and
        the attention fills only the first ``attn_metadata.num_tokens`` rows of its output. The
        padding rows are zeroed (inside a breakable graph by the graph, ahead of the break), so
        they stay finite for the gate, ``o_proj`` and everything after them.
        """
        if not is_in_breakable_cuda_graph():
            super().forward_impl(
                None,
                hidden_states,
                attn_metadata,
                attn_output=attn_outputs,
                fused_a_output=fused_a_output,
            )
            num_tokens = attn_metadata.num_tokens
            if attn_outputs[0].shape[0] != num_tokens:
                attn_outputs[0][num_tokens:].zero_()
            return
        attn_outputs[0].zero_()
        maybe_bcg_k3_mla_attention_inplace(
            hidden_states, self.layer_idx_str, attn_outputs[0], fused_a_output
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # _create_outputs() rather than create_output(): the base implementation
        # takes a list so a sparse-attention backend can append its own buffers,
        # and it routes through the sparse hooks when they are installed. The
        # dense path this module uses is element 0.
        attn_outputs = self._create_outputs(hidden_states, attn_metadata)
        if self.g_fused_a_proj is not None:
            # TRTLLM_K3_MLA_FUSE_GATE_A: the gate comes out of the fused GEMM, on the full
            # (CUDA-graph padded) input as g_proj ran before: the gate rows match attn_outputs;
            # MLA slices fused_a_output to the real tokens. _gate_stream is None here
            # (_check_gate_flags).
            gate_width = self.g_proj.out_features
            gate, fused_a_output = self.g_fused_a_proj(hidden_states).split(
                [gate_width, self.g_fused_a_proj.out_features - gate_width], -1
            )
            self._run_attention(hidden_states, attn_metadata, attn_outputs, fused_a_output)
            return self._apply_output_gate_and_o_proj(hidden_states, attn_outputs[0], gate)

        def attention() -> torch.Tensor:
            self._run_attention(hidden_states, attn_metadata, attn_outputs)
            return attn_outputs[0]

        # A side-stream fork cannot span the eager break of a breakable CUDA graph: there the gate
        # runs after the attention, on the main stream.
        if self._gate_stream is None or is_in_breakable_cuda_graph():
            return self._apply_output_gate_and_o_proj(hidden_states, attention())
        attn_out, gate = maybe_execute_in_parallel(
            attention,
            lambda: self.g_proj(hidden_states),
            self._gate_fork_event,
            self._gate_join_event,
            self._gate_stream,
            disable_on_compile=True,
        )
        return self._apply_output_gate_and_o_proj(hidden_states, attn_out, gate)
