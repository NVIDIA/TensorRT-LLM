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
from ...utils import AuxStreamType
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


def _check_gate_flags() -> None:
    """Reject ``TRTLLM_K3_MLA_FUSE_GATE_A=1`` together with ``TRTLLM_K3_MLA_GATE_OVERLAP=1``."""
    if _K3_MLA_GATE_OVERLAP and get_env_flag(_K3_MLA_FUSE_GATE_A_ENV, False):
        raise ValueError(
            "TRTLLM_K3_MLA_FUSE_GATE_A=1 and TRTLLM_K3_MLA_GATE_OVERLAP=1 are alternatives: the "
            "first fuses g_proj into the q_a/kv_a GEMM before the attention, the second runs it "
            "on a side stream during the attention. Set at most one of them; "
            "TRTLLM_K3_MLA_GATE_FP8=1 works with either."
        )


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
            super().forward_impl(
                None,
                hidden_states,
                attn_metadata,
                attn_output=attn_outputs,
                fused_a_output=fused_a_output,
            )
            return self._apply_output_gate_and_o_proj(hidden_states, attn_outputs[0], gate)
        forward_impl = super().forward_impl

        def attention() -> torch.Tensor:
            forward_impl(None, hidden_states, attn_metadata, attn_output=attn_outputs)
            return attn_outputs[0]

        if self._gate_stream is None:
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
