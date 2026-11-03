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
from typing import Callable, Optional

import torch

from ....functional import PositionEmbeddingType
from ....mapping import Mapping
from ....quantization import QuantAlgo
from ...attention.backends import AttentionMetadata, TrtllmAttention
from ...attention.backends.interface import PositionalEmbeddingParams, RopeParams
from ...attention.mla import MLA
from ...distributed import AllReduceParams
from ...distributed.ops import push_bf16_projection
from ...flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl
from ...model_config import ModelConfig
from ...utils import AuxStreamType
from ..linear import Linear, TensorParallelMode
from ..low_m_gemm import planned_low_m_linear
from ..low_m_gemm_plans import KIMI_K31_DECODE_PLAN
from ..rms_norm import RMSNorm
from .output_gate import sigmoid_gate_mul
from .vup_gate import VUP_GATE_MAX_ROWS, vup_gate


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
    # Read by attention paths that rotate on the Python side (nvfp4_ds_mla): an identity table is a copy.
    backend.rope_is_identity = True

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

    # TRTLLM_K3_L2PF_V2: set by the model to its prefetch controller's "attention_moe" fork of this
    # layer, called with the token count once the decode core (and the Helix exchange) is issued,
    # before the output gate and o_proj.
    l2pf_attention_hook: Optional[Callable[[int], None]] = None

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
        sparse_config = model_config.sparse_attention_config
        role = ""
        if sparse_config is not None and getattr(sparse_config, "algorithm", None) == "dsa":
            from ...attention.backends.sparse.dsa.params import disagg_role, mark_kimi_k31_dsa

            # Kimi K3.1: the MLA layers carry a DSA indexer (K3's dense MLA does not).
            mark_kimi_k31_dsa()
            role = disagg_role()
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
        )
        # Keep the base MLA registration enabled so breakable CUDA graphs use
        # the shared custom op. The output gate is a base hook and runs on both
        # the registered and eager paths before the row-parallel o_proj.
        # Prefill-graph padding rows leave the core zero, so the gate, o_proj and
        # the MoE router after them only ever see finite rows.
        self.zero_padding_output_rows = True

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
            # modules() also reaches the projections of a sparse indexer
            # owned by the MQA backend.
            for module in self.modules():
                if isinstance(module, Linear):
                    module.create_weights()

        # K3 is NoPE. The base MLA backends still require real RoPE tables, so
        # retain their expected shape and replace every rotation with identity.
        # Sparse MLA (K3.1 DSA) builds no dense MHA backend.
        for backend in (self.mha, self.mqa):
            if backend is not None:
                assert isinstance(backend, TrtllmAttention)
                _install_identity_rope_table(backend)
        self.rotary_emb = None
        self.apply_rotary_emb = False
        # Set by the Kimi K3.1 MLA runtime for one forward (the BF16 o_proj push GEMM): the deferred
        # all-reduce push target a BF16 o_proj may store its output into.
        self.bf16_ar_push = None
        attention_dp = self.mapping.enable_attention_dp
        # Kimi K3.1 packed DSA projections (see _pack_dsa_projections), on a generation server and on an
        # attention-DP context server: the output gate's g_proj rows join the packed hidden-state GEMM
        # [kv_a_proj_with_mqa | wk | weights_proj | g_proj] (same input), so the gate needs no GEMM of its own after
        # the attention, and sigmoid(gate) * attn_output runs as one kernel on the packed output's column slice.
        # The gate values come from another GEMM kernel than g_proj's own (FP32 accumulation in another order), so
        # they can differ from the separate projection in the last BF16 bit.
        self._k31_packed_gate = role == "generation" or (role == "context" and attention_dp)
        # Generation server without attention DP: a generation step of at most this many query rows whose output
        # gate is a slice of the packed projection GEMM runs the V up-projection and the sigmoid gate as one kernel
        # (vup_gate.vup_gate) instead of the BMM and the gate kernel.
        self._k31_vup_gate_max_rows = (
            VUP_GATE_MAX_ROWS if role == "generation" and not attention_dp else 0
        )
        # Kimi K3.1, not on a context server: the q_a_layernorm whose output feeds the packed DSA [q_b | wq_b]
        # projection through the decode plan's TGV GEMM (it loads the weights before its griddepcontrol.wait, the
        # activation after) releases that GEMM at launch (q_norm_pdl, the same bits), so its weight stages fill while
        # the norm runs.
        self._k31_q_norm_release = sparse_config is not None and role != "context"

    def cache_derived_state(self) -> None:
        super().cache_derived_state()
        self._pack_dsa_projections()

    def _pack_dsa_projections(self) -> None:
        """Pack a DSA layer's indexer projections under the MLA projections that share their input.

        Kimi K3.1 only: its MLA layers carry a DSA indexer, K3's dense MLA does not. The hidden-state GEMM becomes
        [kv_a_proj_with_mqa | wk | weights_proj] and the normed-q_a GEMM [q_b_proj | wq_b], one launch per input
        instead of two, with wk / weights_proj in BF16 like the reference implementation. The module weights become
        row views of the packed tensors, so no memory is added. Plain BF16 weights only.
        """
        self.dsa_packed_hidden_weight = None
        self.dsa_packed_q_weight = None
        self.dsa_packed_gemm_plan = None
        # Rows of g_proj at the end of dsa_packed_hidden_weight (_k31_packed_gate), and the
        # (layer input, gate slice) the last packed GEMM produced, which _apply_output_gate consumes.
        self.dsa_packed_gate_rows = 0
        self.dsa_packed_gate_out = None
        indexer = getattr(self.mqa, "indexer", None) if self.mqa is not None else None
        if indexer is None or getattr(indexer, "projection_free", False):
            return
        kv_a, q_b = self.kv_a_proj_with_mqa, getattr(self, "q_b_proj", None)

        def plain_bf16(linear) -> bool:
            return (
                linear is not None
                and linear.weight.dtype == torch.bfloat16
                and linear.bias is None
                and not linear.has_any_quant
                and getattr(linear, "lora", None) is None
            )

        if not plain_bf16(kv_a) or self._resolve_qa_fused_scale() is not None:
            return
        dtype = kv_a.weight.dtype
        g_proj = getattr(self, "g_proj", None) if getattr(self, "use_output_gate", False) else None
        pack_gate = (
            self._k31_packed_gate
            and plain_bf16(g_proj)
            and g_proj.weight.shape[1] == kv_a.weight.shape[1]
        )
        rows = [
            kv_a.weight.data,
            indexer.wk.weight.data.to(dtype),
            indexer.weights_proj.weight.data.to(dtype),
        ]
        if pack_gate:
            rows.append(g_proj.weight.data)
        packed = torch.cat(rows, dim=0)
        n = kv_a.weight.shape[0]
        kv_a.weight.data = packed[:n]
        if pack_gate:
            # g_proj keeps working on its own (paths that bypass the packed GEMM): its weight is a row view.
            g_proj.weight.data = packed[packed.shape[0] - g_proj.weight.shape[0] :]
            self.dsa_packed_gate_rows = g_proj.weight.shape[0]
        self.dsa_packed_hidden_weight = packed
        if os.environ.get("TRTLLM_KIMI_K31_LOW_M_PLAN", "1") != "0":
            self.dsa_packed_gemm_plan = KIMI_K31_DECODE_PLAN
        if (
            plain_bf16(q_b)
            and plain_bf16(indexer.wq_b)
            and q_b.weight.shape[1] == indexer.wq_b.weight.shape[1]
        ):
            packed_q = torch.cat([q_b.weight.data, indexer.wq_b.weight.data], dim=0)
            m = q_b.weight.shape[0]
            q_b.weight.data = packed_q[:m]
            indexer.wq_b.weight.data = packed_q[m:]
            self.dsa_packed_q_weight = packed_q

    def _apply_output_gate(
        self,
        hidden_states: torch.Tensor,
        attn_output: torch.Tensor,
    ) -> torch.Tensor:
        # Sigmoid gate on o_proj's input. g_proj matches o_proj's input
        # sharding, so the multiply composes with the helix-CP output
        # projection.
        if getattr(self, "_output_gate_fused", False):
            # The gate ran with the V up-projection (_absorption_v_up_override).
            self._output_gate_fused = False
            return attn_output
        if self.use_output_gate:
            gate = None
            packed, self.dsa_packed_gate_out = getattr(self, "dsa_packed_gate_out", None), None
            if (
                packed is not None
                and packed[0].data_ptr() == hidden_states.data_ptr()
                and packed[1].shape[0] == hidden_states.shape[0] == attn_output.shape[0]
            ):
                # Packed gate: g_proj(x) is a column slice of this layer's packed GEMM.
                gate = packed[1]
                gated = sigmoid_gate_mul(attn_output, gate, min_tokens=1)
                if gated is not None:
                    return gated
                return attn_output * gate.sigmoid()
            plan = getattr(self, "dsa_packed_gemm_plan", None)
            if plan is not None and hidden_states.dim() == 2:
                gate = planned_low_m_linear(plan, hidden_states, self.g_proj.weight)
            if gate is None:
                gate = self.g_proj(hidden_states)
            gated = sigmoid_gate_mul(attn_output, gate)
            if gated is not None:
                return gated
            return attn_output * gate.sigmoid()
        return attn_output

    def _absorption_v_up_override(
        self, attn_out_latent: torch.Tensor, output: torch.Tensor
    ) -> bool:
        """Up to ``_k31_vup_gate_max_rows`` rows: the V up-projection and the packed output gate
        as one kernel; ``_apply_output_gate`` then passes the gated output through."""
        rows = attn_out_latent.shape[0]
        packed = getattr(self, "dsa_packed_gate_out", None)
        v_dim = self.num_heads_tp_cp * self.v_head_dim
        if (
            not 0 < rows <= self._k31_vup_gate_max_rows
            or not self.use_output_gate
            or packed is None
            or packed[0].shape[0] != rows
            or packed[1].shape != (rows, v_dim)
            or packed[1].stride(1) != 1
            or packed[1].dtype != torch.bfloat16
            or self.v_b_proj.dtype != torch.bfloat16
            or not self.v_b_proj.is_contiguous()
            or attn_out_latent.dtype != torch.bfloat16
            or attn_out_latent.dim() != 2
            or attn_out_latent.stride(1) != 1
            or not output.is_contiguous()
        ):
            return False
        self.dsa_packed_gate_out = None
        vup_gate(attn_out_latent, self.v_b_proj, packed[1], out=output.view(rows, v_dim))
        self._output_gate_fused = True
        return True

    @staticmethod
    def _runs_flashinfer_rmsnorm(norm) -> bool:
        """Whether ``norm(x)`` runs FlashInfer's BF16 RMSNorm (no residual, no fused quant)."""
        return (
            type(norm) is RMSNorm
            and not (norm.use_cuda_tile or norm.use_gemma or norm.is_nvfp4)
            and not norm.return_hp_output
            and IS_FLASHINFER_AVAILABLE
        )

    def _packed_q_b_runs_tgv(self, rows: int) -> bool:
        """Whether the packed DSA [q_b | wq_b] projection of ``rows`` rows is the decode plan's TGV GEMM."""
        packed_q = getattr(self, "dsa_packed_q_weight", None)
        plan = getattr(self, "dsa_packed_gemm_plan", None)
        if packed_q is None or not plan:
            return False
        bounds = plan.get((int(packed_q.shape[0]), int(packed_q.shape[1])))
        choice = next((c for bound, c in bounds if rows <= bound), None) if bounds else None
        return choice is not None and choice[0] == "tgv"

    def _q_a_layernorm_maybe_fused(self, q: torch.Tensor, return_norm_out: bool = False):
        norm = self.q_a_layernorm
        if (
            self._k31_q_norm_release
            and return_norm_out
            and self._runs_flashinfer_rmsnorm(norm)
            and get_env_enable_pdl()
            and self._resolve_qa_fused_scale() is None
            and self._packed_q_b_runs_tgv(q.shape[0])
        ):
            # The DSA projection's norm: its only programmatic dependent is the packed q_b GEMM (the planned TGV
            # GEMM).
            from .q_norm_pdl import can_use, rmsnorm_release_at_launch

            if can_use(q, norm.weight):
                out = rmsnorm_release_at_launch(q, norm.weight, norm.variance_epsilon)
                return out, out
        return super()._q_a_layernorm_maybe_fused(q, return_norm_out)

    def _project_output_impl(
        self,
        attn_output: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams],
    ) -> torch.Tensor:
        target = self.bf16_ar_push
        if (
            target is not None
            and not self.mapping.has_cp_helix()
            and attn_output.dim() == 2
            and push_bf16_projection(attn_output, self.o_proj.weight, target)
        ):
            # BF16 o_proj push GEMM: o_proj stored its rows into the Lamport slots of the
            # all-reduce the runtime defers to A_mid; the runtime returns its LamportPending.
            return target.output
        return super()._project_output_impl(
            attn_output, position_ids, attn_metadata, all_reduce_params
        )
