# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MLA attention layer: its runtime, the generic path on catalog and family operations, and the decode-kernel
path.

Kimi K3's MLA is DeepSeek-V3's latent attention with three differences: no rotary position embedding (the rope ops
run on an identity table, so their "rotation" is a copy), an output gate ``attn * sigmoid(x W_g^T)`` before
``o_proj``, and one fused ``[q_a 1536 | kv_a 512 | k_pe 64]`` projection. The KV cache is a bf16 latent pool with
64-token pages; one TP16 rank runs 6 of the 96 heads.

The generic path (every step the decode kernels do not take: prefill with or without a cached prefix, mixed steps,
generation of more than 8 requests) runs the operations the built-in MLA module runs, in the same order, on the same
kernels:

* projections: ``gemm/cublas_mm`` for the fused ``[q_a | kv_a | k_pe]`` rows, ``torch.nn.functional.linear`` for
  q_b, kv_b, the gate and ``o_proj`` (what the built-in bf16 ``Linear`` runs), the q_a / kv_a RMSNorms on ``norm.py``;
* context: ``attention/thop_attention`` with ``attention_input_type`` 1 over K / V expanded from the latent (fresh
  prefill: kv_b per head with ``gemm/bmm_out`` straight into the strided K buffer); with a cached prefix,
  ``attention/mla_rope_append_paged_kv_assign_q`` and then either the whole prefix
  (``attention/load_paged_kv_cache_for_mla``, chunked prefill off) or the prefix chunk by chunk with the partials
  merged on their softmax statistics (``load_chunked_kv_cache_for_mla``, ``merge_chunked_attention_for_mla``, chunked
  prefill on);
* generation: the k_b absorption (``gemm/bmm_out``) beside ``attention/mla_rope_generation``, the latent attention,
  and the v_b projection (``gemm/bmm_out``).

A pure decode step of at most 8 requests of at most 8 tokens runs the K3 decode kernels instead
(``K3DecodeMLA.forward``). Under a breakable CUDA graph the generic path's attention core runs as one graph break
(``graph/eager_on_graph``), reading the step's metadata from the model's extra attrs.
"""

from __future__ import annotations

import copy
import math
import weakref
from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.load_paged_kv_cache_for_mla import (  # noqa: E501
    load_paged_kv_cache_for_mla,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.mla_rope_append_paged_kv_assign_q import (  # noqa: E501
    mla_rope_append_paged_kv_assign_q,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.mla_rope_generation import (
    mla_rope_generation,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import bmm_out
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.graph.eager_on_graph import (
    eager_on_graph,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl import (
    environment as _environment,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_mla.decode_view import (
    K3MlaGeometry,
    k3_mla_decode_view,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention import (
    cute_dsl_mla_decode as _cute,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention import (
    flashinfer_mla_decode as _flashinfer,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention.k3_mla_attn_vb_out import (
    k3_mla_attn_vb_out,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention.k3_mla_attn_workspace import (
    K3MlaAttnWorkspace,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention.k3_mla_qkv import (
    k3_mla_qkv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention.load_chunked_kv_cache_for_mla import (  # noqa: E501
    load_chunked_kv_cache_for_mla,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention.merge_chunked_attention_for_mla import (  # noqa: E501
    merge_chunked_attention_for_mla,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.comm.k3_allreduce import (
    K3AllReduce,
)
from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import is_in_breakable_cuda_graph
from tensorrt_llm._torch.utils import AuxStreamType, get_model_extra_attrs, maybe_compiled_cat
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

from . import decode_gemv as _decode_gemv
from .norm import RMSNorm, run_in_parallel
from .step import DecodeStep

if TYPE_CHECKING:
    from transformers import PretrainedConfig

# Identity-RoPE table positions. K3 is NoPE (the table holds cos=1/sin=0); the rope ops index it by absolute
# position, so it covers max_position_embeddings and grows to the engine's max_seq_len. KIMI_K3_MLA_MAX_POSITIONS
# overrides the starting size for short-context deployments.
_ROPE_THETA = 10000.0
# PositionEmbeddingType.rope_gpt_neox: what the built-in K3 MLA hands the attention op (the MLA kernels read only the
# rope table).
_POSITION_EMBEDDING_ROPE_GPT_NEOX = 2
_MASK_PADDING, _MASK_CAUSAL = 0, 1
_CONTEXT, _GENERATION = 1, 2
_QUANT_MODE_BF16_KV = 0
# tensorrt_llm.runtime.kv_cache_manager_v2.BAD_PAGE_INDEX: an unassigned page-table entry.
_BAD_PAGE_INDEX = -1

#: The layers a breakable CUDA graph's eager break finds by key (``_core_inplace``), weakly held.
_BREAK_LAYERS: "weakref.WeakValueDictionary[str, K3DecodeMLA]" = weakref.WeakValueDictionary()


def _core_inplace(hidden_states: torch.Tensor, layer_key: str, output: torch.Tensor) -> None:
    """The eager break of a breakable CUDA graph: the layer's attention core on the step's live attention metadata,
    into the graph-owned ``output``. Replays re-run it with the capture-time tensors and key, so the metadata, which
    changes every step, is read from the engine's model attributes, never passed."""
    attrs = get_model_extra_attrs()
    assert attrs is not None and attrs.get("attention_metadata") is not None, (
        "a breakable CUDA graph's MLA break needs the engine's attention metadata in the model attributes"
    )
    layer = _BREAK_LAYERS.get(layer_key)
    assert layer is not None, f"no live Kimi K3 MLA layer {layer_key}"
    layer._attention_core(hidden_states, output, attrs["attention_metadata"]())


class _IdentityRope:
    """The identity rope table (cos=1, sin=0 in the duplicated GPT-J layout ``[positions, rope_dim, 2]`` fp32) every
    K3 MLA layer of a device shares, and its inverse frequencies (theta 10000, which the MLA kernels do not read but
    the attention op is handed). Grown, never shrunk: a larger table holds the smaller one's rows."""

    _tables: Dict[torch.device, "_IdentityRope"] = {}

    def __init__(self, device: torch.device, positions: int, rope_dim: int) -> None:
        self.positions = positions
        self.cos_sin = torch.empty(1, positions * rope_dim * 2, dtype=torch.float32, device=device)
        pairs = self.cos_sin.view(-1, 2)
        pairs[:, 0] = 1.0
        pairs[:, 1] = 0.0
        exponents = torch.arange(0, rope_dim, 2, dtype=torch.float64) / rope_dim
        self.inv_freq = (1.0 / (_ROPE_THETA**exponents).to(torch.float32)).to(device)

    @classmethod
    def get(cls, device: torch.device, positions: int, rope_dim: int) -> "_IdentityRope":
        table = cls._tables.get(device)
        if table is None or table.positions < positions:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"Kimi K3 MLA: the identity rope table needs {positions} positions inside a CUDA graph capture; "
                    "an eager step sizes it first"
                )
            table = cls(device, positions, rope_dim)
            cls._tables[device] = table
        return table


def resolve_attention_quant_config(
    config: ModelConfig | None, layer_idx: int, projection: str
) -> QuantConfig:
    """Resolve a checkpoint projection, including mixed-precision exclusions."""
    if config is None:
        return QuantConfig()
    global_config = config.quant_config or QuantConfig()
    names = [
        f"{prefix}layers.{layer_idx}.self_attn.{projection}"
        for prefix in ("language_model.model.", "model.", "")
    ]
    if any(global_config.is_module_excluded_from_quantization(name) for name in names):
        return QuantConfig(kv_cache_quant_algo=global_config.kv_cache_quant_algo)
    declarations = config.quant_config_dict or {}
    matches = [declarations[name] for name in names if name in declarations]
    if matches:
        selected = matches[0]
        if any(match.quant_algo != selected.quant_algo for match in matches[1:]):
            raise ValueError(f"Conflicting Kimi K3 quantization aliases for {names[0]}")
    elif global_config.quant_algo == QuantAlgo.MIXED_PRECISION:
        selected = QuantConfig()
    else:
        selected = global_config
    if selected.quant_algo not in (None, QuantAlgo.FP8_BLOCK_SCALES):
        raise ValueError(
            f"Kimi K3 attention projection {names[0]} has unsupported checkpoint "
            f"quantization {selected.quant_algo}"
        )
    if selected.quant_algo == QuantAlgo.FP8_BLOCK_SCALES and selected.group_size not in (None, 128):
        raise ValueError(f"Kimi K3 attention requires 128x128 FP8 blocks for {names[0]}")
    result = copy.copy(selected)
    result.kv_cache_quant_algo = global_config.kv_cache_quant_algo
    return result


class KimiMLARuntime(nn.Module):
    """Wraps K3 MLA and applies its external TP output reduction."""

    def __init__(
        self,
        cfg: "PretrainedConfig",
        layer_idx: int,
        model_config: ModelConfig,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        # The mixer owns the projections and the head sharding. Keep only the final output reduction here so the
        # output gate remains between the attention and the row-parallel o_proj.
        mapping = model_config.mapping
        self._o_allreduce = (
            K3AllReduce(mapping, model_config.allreduce_strategy, torch.bfloat16)
            if mapping.tp_size > 1
            else None
        )
        # What the decode kernels take.
        for name in ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "g_proj", "o_proj"):
            algo = resolve_attention_quant_config(model_config, layer_idx, name).quant_algo
            if algo is not None:
                raise ValueError(
                    f"Kimi K3 MLA layer {layer_idx}: {name} is {algo}; this target runs bf16 attention projections "
                    "only (the decode kernels read them)"
                )
        kv_algo = (
            model_config.quant_config.kv_cache_quant_algo
            if model_config.quant_config is not None
            else None
        )
        if kv_algo is not None:
            raise ValueError(
                f"Kimi K3 MLA: a {kv_algo} KV cache; this target runs a bf16 latent cache only"
            )
        if not cfg.mla_use_output_gate:
            raise ValueError("Kimi K3 MLA: this target runs the output-gated MLA only")
        max_positions = _environment.mla_max_positions(cfg.max_position_embeddings)
        self.mixer = K3DecodeMLA(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            q_lora_rank=cfg.q_lora_rank,
            kv_lora_rank=cfg.kv_lora_rank,
            qk_nope_head_dim=cfg.qk_nope_head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            v_head_dim=cfg.v_head_dim,
            rms_norm_eps=cfg.rms_norm_eps,
            dtype=torch.bfloat16,
            layer_idx=layer_idx,
            max_position_embeddings=max_positions,
            mapping=mapping,
            aux_stream=aux_stream_dict.get(AuxStreamType.Attention),
        )

    def will_run_decode_branch(self, attn_metadata: AttentionMetadata, step: DecodeStep) -> bool:
        """Whether the mixer runs ``step`` on the decode kernels (``K3DecodeMLA.will_run_decode_branch``)."""
        return self.mixer.will_run_decode_branch(attn_metadata, step)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        step: DecodeStep,
        reduce_output: bool = True,
        project_output: bool = True,
    ) -> torch.Tensor:
        """``reduce_output=False`` returns ``o_proj``'s TP partial (no all-reduce); ``project_output=False`` the
        mixer's gated attention output before ``o_proj``, only where ``will_run_decode_branch`` holds."""
        out = self.mixer(hidden_states, attn_metadata, step=step, project_output=project_output)
        return self._reduce(out) if project_output and reduce_output else out

    def forward_generic(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        reduce_output: bool = True,
    ) -> torch.Tensor:
        out = self.mixer.forward_generic(hidden_states, attn_metadata)
        return self._reduce(out) if reduce_output else out

    def _reduce(self, out: torch.Tensor) -> torch.Tensor:
        # Head-sharded TP: sum the row-sharded o_proj partials across the head-shard group.
        return self._o_allreduce(out) if self._o_allreduce is not None else out


class _Weight(nn.Module):
    """A projection's weight ``[out_features, in_features]``, under the name the loader and the decode kernels read
    (``<projection>.weight``); no bias."""

    def __init__(self, out_features: int, in_features: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype), requires_grad=False
        )
        self.register_parameter("bias", None)


class K3DecodeMLA(nn.Module):
    """Kimi K3's MLA attention on one rank: the generic path on catalog and family operations, and a decode step's
    attention on the decode kernels.

    A decode step runs ``x [W_a; W_g]^T`` with the gate columns through a sigmoid (the ``mla_ag`` decode GEMV site
    where it takes the rows, else one GEMM), then ``attention/k3_mla_qkv`` (the q_a / kv_a RMSNorms, q_b and the k_b
    absorption into the fused query, the step's latent rows stored into the paged cache) and
    ``attention/k3_mla_attn_vb_out`` (the attention over the paged cache, v_b and the output gate in one launch), then
    ``o_proj`` (the ``o_proj`` site where it takes the rows, else one GEMM). Every other step, and a decode step whose
    cache the kernels do not read (``k3_mla_decode_view`` says why), runs ``forward_generic``.

    Loaded: ``k3_ag_weight`` ``[W_a; W_g]`` ``[q_lora + kv_lora + rope + heads * v, hidden]`` (the checkpoint's q_a,
    kv_a and this rank's g rows, one parameter), ``q_a_layernorm``, ``kv_a_layernorm``, ``q_b_proj``
    ``[heads * (nope + rope), q_lora]``, ``kv_b_proj`` ``[heads * nope | heads * v, kv_lora]`` (all K rows, then all V
    rows) and ``o_proj`` ``[hidden, heads * v]``, this rank's heads. Derived in ``post_load_weights``:
    ``k_b_proj_trans`` ``[heads, kv_lora, nope]`` (the K rows, transposed), ``v_b_proj`` ``[heads, v, kv_lora]`` (a
    view of the V rows), and ``a_proj_weight`` / ``g_proj_weight``, the row views of ``k3_ag_weight`` the generic path
    reads. The attention workspace is the device's ``K3MlaAttnWorkspace``; it and the decode GEMVs' state are set by
    the target in ``post_load_weights``.
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
        rms_norm_eps: float,
        dtype: torch.dtype,
        layer_idx: int,
        max_position_embeddings: int,
        mapping,
        aux_stream: Optional[torch.cuda.Stream],
    ) -> None:
        super().__init__()
        if mapping.cp_size > 1:
            raise ValueError("Kimi K3 MLA: context parallelism is not implemented")
        tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size
        if num_heads % tp_size:
            raise ValueError(f"Kimi K3 MLA: {num_heads} heads do not split over TP {tp_size}")
        self.layer_idx = layer_idx
        self.mapping = mapping
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_heads_tp = num_heads // tp_size
        self.num_heads_tp_cp = self.num_heads_tp
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.dtype = dtype
        self.fuse_qkv_a_proj = True
        self.use_output_gate = True
        self.max_position_embeddings = max_position_embeddings
        # NoPE: no YaRN temperature, so both attention phases scale QK^T by 1 / sqrt(nope + rope).
        self.q_scaling = 1.0
        dims = (kv_lora_rank, qk_rope_head_dim, q_lora_rank, qk_nope_head_dim, v_head_dim)
        if self.num_heads_tp % 6 or dims != (512, 64, 1536, 128, 128):
            raise ValueError(
                "Kimi K3 MLA: the decode kernels take 6k heads, latent 512, rope 64, q_lora 1536, nope 128 and "
                f"v 128 per rank; got {self.num_heads_tp} heads and (latent, rope, q_lora, nope, v) {dims}"
            )
        heads = self.num_heads_tp
        # [W_a; W_g]: the fused q_a / kv_a projection's rows, then the output gate's.
        self._a_rows = q_lora_rank + kv_lora_rank + qk_rope_head_dim
        self.k3_ag_weight = nn.Parameter(
            torch.empty(self._a_rows + heads * v_head_dim, hidden_size, dtype=dtype),
            requires_grad=False,
        )
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=rms_norm_eps, dtype=dtype)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=rms_norm_eps, dtype=dtype)
        self.q_b_proj = _Weight(heads * self.qk_head_dim, q_lora_rank, dtype)
        self.kv_b_proj = _Weight(heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank, dtype)
        self.o_proj = _Weight(hidden_size, heads * v_head_dim, dtype)
        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]
        self.geometry = K3MlaGeometry(
            num_heads=heads,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            q_scaling=self.q_scaling,
            layer_idx=layer_idx,
        )
        # Derived in post_load_weights.
        self.k_b_proj_trans: Optional[torch.Tensor] = None
        self.v_b_proj: Optional[torch.Tensor] = None
        # k3_ag_weight's [q_a; kv_a] and gate rows, views (the generic path's projections).
        self.a_proj_weight: Optional[torch.Tensor] = None
        self.g_proj_weight: Optional[torch.Tensor] = None
        # The decode attention's workspace: one per device, shared by every MLA layer.
        self.k3_workspace: Optional[K3MlaAttnWorkspace] = None
        # The decode GEMVs' state (decode_gemv.py), shared by the target's layers.
        self.decode_gemvs: Optional[_decode_gemv.K3DecodeGemvs] = None
        # The FlashInfer MLA decode's multi-CTA KV counter (mixed steps), made on first use.
        self._fi_counter: Optional[torch.Tensor] = None
        # The key a breakable CUDA graph's eager break finds this layer by (``_core_inplace``).
        self._break_key = f"{layer_idx}:{id(self)}"
        _BREAK_LAYERS[self._break_key] = self

    def post_load_weights(self) -> None:
        """The absorption weights from kv_b_proj and the row views of ``k3_ag_weight`` (once: CUDA graphs captured
        since read them), and the device's identity rope table at the checkpoint's position count. Sizing the table
        here keeps it out of the allocator segments the warm-up's largest steps use: allocated at the first step
        instead, it lands in one of them and pins that segment after the step's temporaries are freed."""
        if self.k_b_proj_trans is not None:
            return
        heads, nope, v, kv = (
            self.num_heads_tp,
            self.qk_nope_head_dim,
            self.v_head_dim,
            self.kv_lora_rank,
        )
        kv_b = self.kv_b_proj.weight
        with torch.no_grad():
            self.k_b_proj_trans = (
                kv_b[: heads * nope].view(heads, nope, kv).transpose(1, 2).contiguous()
            )
        # A view: the V rows are shared with the context path's kv_b projection.
        self.v_b_proj = kv_b[heads * nope :].view(heads, v, kv)
        # Views of the parameter's storage, no copy.
        ag = self.k3_ag_weight.detach()
        self.a_proj_weight, self.g_proj_weight = ag[: self._a_rows], ag[self._a_rows :]
        _IdentityRope.get(ag.device, self.max_position_embeddings, self.qk_rope_head_dim)

    # ------------------------------------------------------------------------------------------------------------
    # The decode branch.
    # ------------------------------------------------------------------------------------------------------------

    def will_run_decode_branch(self, attn_metadata: AttentionMetadata, step: DecodeStep) -> bool:
        """Whether ``forward`` runs ``step`` on the decode kernels (see ``_k3_step_view``). Only there does it return
        the gated attention output before ``o_proj``; the KV writes forbid running the attention twice, so a caller
        that needs it asks first."""
        return self._k3_step_view(attn_metadata, step) is not None

    def _k3_step_view(self, attn_metadata: AttentionMetadata, step: DecodeStep) -> Optional[dict]:
        """The paged-cache view the decode kernels read on ``step``: a decode step, this layer's derived weights
        (``post_load_weights``) and workspace built, outside a breakable CUDA graph, and a cache
        ``k3_mla_decode_view`` takes. Else None."""
        if not (
            step.decode
            and self.k_b_proj_trans is not None
            and self.k3_workspace is not None
            and not is_in_breakable_cuda_graph()
        ):
            return None
        view = k3_mla_decode_view(self.geometry, attn_metadata, step.num_tokens)
        if isinstance(view, str):
            logger.info_once(
                f"Kimi K3 MLA: the generic path for a decode step the decode kernels do not read ({view})",
                key=f"k3_mla_decode_view_{view}",
            )
            return None
        return view

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        step: DecodeStep,
        project_output: bool = True,
    ) -> torch.Tensor:
        """``forward_generic``, except on a decode step whose cache the decode kernels read
        (``will_run_decode_branch``). There only, ``project_output=False`` returns the gated attention output
        ``[M, H * 128]``, the input of ``o_proj``."""
        view = self._k3_step_view(attn_metadata, step)
        if view is None:
            if not project_output:
                raise ValueError("project_output=False needs a step will_run_decode_branch takes")
            return self.forward_generic(hidden_states, attn_metadata)
        x = hidden_states[: step.num_tokens].contiguous()
        rows = self._a_rows
        gemvs = self.decode_gemvs
        ag = None if gemvs is None else gemvs.project("mla_ag", x, self.k3_ag_weight)
        if ag is None:
            ag = F.linear(x, self.k3_ag_weight)
            ag[:, rows:].sigmoid_()
        fused_q = k3_mla_qkv(
            ag,
            self.q_a_layernorm.weight,
            float(self.q_a_layernorm.variance_epsilon),
            self.q_b_proj.weight,
            self.k_b_proj_trans,
            self.kv_a_layernorm.weight,
            float(self.kv_a_layernorm.variance_epsilon),
            view["pool"],
            view["row_stride"],
            view["page_table"],
            view["page_offset"],
            view["seq_len"],
        )
        attn_output = x.new_empty(x.shape[0], self.num_heads_tp * self.v_head_dim)
        k3_mla_attn_vb_out(
            fused_q,
            view["pool"],
            view["row_stride"],
            view["page_table"],
            view["page_offset"],
            view["seq_len"],
            view["softmax_scale"],
            self.v_b_proj,
            attn_output,
            self.k3_workspace,
            gate=ag,
            gate_col0=rows,
        )
        if not project_output:
            return attn_output
        out = None if gemvs is None else gemvs.project("o_proj", attn_output, self.o_proj.weight)
        if out is None:
            out = F.linear(attn_output, self.o_proj.weight)
        return out

    # ------------------------------------------------------------------------------------------------------------
    # The generic path.
    # ------------------------------------------------------------------------------------------------------------

    def forward_generic(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        """``o_proj(attention(x) * sigmoid(x W_g^T))`` over every row of ``hidden_states`` (rows past the step's
        tokens, CUDA-graph padding, carry unspecified values)."""
        attn_output = hidden_states.new_empty(
            hidden_states.shape[0], self.num_heads_tp * self.v_head_dim
        )
        if is_in_breakable_cuda_graph():
            # One graph break: the attention core reads the step's host metadata, which a captured segment would
            # freeze. The break gets tensors only and reads the metadata the engine sets before each replay.
            eager_on_graph(hidden_states, self._break_key, attn_output, inner=_core_inplace)
        else:
            self._attention_core(hidden_states, attn_output, attn_metadata)
        gated = attn_output * F.linear(hidden_states, self.g_proj_weight).sigmoid()
        return F.linear(gated, self.o_proj.weight)

    def _attention_core(
        self, hidden_states: torch.Tensor, output: torch.Tensor, md: AttentionMetadata
    ) -> None:
        """Rows ``[:num_tokens]`` of ``output``: the attention of the step's tokens before the gate (context rows
        first, then generation rows), the step's latent rows written into the paged cache."""
        tc = md.num_ctx_tokens
        num_tokens = md.num_tokens
        x = hidden_states[:num_tokens]
        q_lora, kv_lora, rope_dim = self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim
        rope = _IdentityRope.get(
            x.device,
            max(self.max_position_embeddings, md.max_seq_len, md.kv_cache_manager.max_seq_len),
            rope_dim,
        )

        q, compressed_kv, k_pe = cublas_mm(x, self.a_proj_weight.t()).split(
            [q_lora, kv_lora, rope_dim], -1
        )
        q, compressed_kv = run_in_parallel(
            lambda: self.q_a_layernorm(q),
            lambda: self.kv_a_layernorm(compressed_kv),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        q, latent_cache = run_in_parallel(
            lambda: F.linear(q, self.q_b_proj.weight),
            lambda: torch.concat([compressed_kv, k_pe], dim=-1),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        if md.num_contexts > 0:
            self._context(q[:tc], compressed_kv[:tc], latent_cache[:tc], md, output[:tc], rope)
        if md.num_generations > 0:
            self._generation(q[tc:], latent_cache[tc:], md, output[tc:num_tokens], rope)

    def _local_layer(self, md) -> int:
        """This layer's row of the KV cache manager's pool mapping."""
        return md.kv_cache_manager.layer_offsets[self.layer_idx]

    def _window(self, md) -> int:
        """The attention window the attention op is handed: the KV cache manager's for this layer, else
        max_seq_len."""
        window = md.kv_cache_manager.max_attention_window_vec[self._local_layer(md)]
        return md.max_seq_len if window is None else window

    def _thop(
        self,
        md,
        rope: _IdentityRope,
        *,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        output: torch.Tensor,
        attention_input_type: int,
        mask_type: int,
        latent_cache: Optional[torch.Tensor],
        sequence_length: torch.Tensor,
        host_past_key_value_lengths: torch.Tensor,
        host_total_kv_lens: torch.Tensor,
        softmax_stats_tensor: Optional[torch.Tensor] = None,
        chunked_prefill_buffer_batch_size: int = 1,
        q_pe: Optional[torch.Tensor] = None,
        cu_q_seqlens: Optional[torch.Tensor] = None,
        cu_kv_seqlens: Optional[torch.Tensor] = None,
        fmha_scheduler_counter: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``attention/thop_attention`` with what the built-in MLA's backends hand it: the context backend runs the
        heads as MHA over [nope | rope] = 192, the generation backend as latent MQA over 576 with v = 512."""
        context = attention_input_type == _CONTEXT
        heads = self.num_heads_tp
        spec = bool(md.is_spec_decoding_enabled and md.use_spec_decoding) and not context
        return thop_attention(
            q=q,
            k=k,
            v=v,
            output=output,
            output_sf=None,
            workspace=md.effective_workspace,
            sequence_length=sequence_length,
            host_past_key_value_lengths=host_past_key_value_lengths,
            host_total_kv_lens=host_total_kv_lens,
            context_lengths=md.prompt_lens_cuda_runtime,
            host_context_lengths=md.prompt_lens_cpu_runtime,
            host_request_types=md.host_request_types_runtime,
            max_context_q_len_override=None,
            kv_cache_block_offsets=md.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
            cache_indirection=md.cache_indirection,
            kv_scale_orig_quant=None,
            kv_scale_quant_orig=None,
            out_scale=None,
            rotary_inv_freq=rope.inv_freq,
            rotary_cos_sin=rope.cos_sin,
            latent_cache=latent_cache,
            q_pe=q_pe,
            block_ids_per_seq=None,
            attention_sinks=None,
            is_fused_qkv=False,
            update_kv_cache=True,
            predicted_tokens_per_seq=1,
            local_layer_idx=self._local_layer(md),
            num_heads=heads,
            num_kv_heads=heads if context else 1,
            head_size=self.qk_head_dim if context else self.kv_lora_rank + self.qk_rope_head_dim,
            tokens_per_block=md.tokens_per_block,
            max_num_requests=md.max_num_requests,
            max_context_length=md.max_context_length,
            max_seq_len=md.max_seq_len,
            attention_window_size=self._window(md),
            beam_width=md.beam_width,
            mask_type=mask_type,
            quant_mode=_QUANT_MODE_BF16_KV,
            q_scaling=self.q_scaling,
            position_embedding_type=_POSITION_EMBEDDING_ROPE_GPT_NEOX,
            rope_dim=self.qk_rope_head_dim,
            rope_base=_ROPE_THETA,
            rope_scale_type=0,
            rope_scale=1.0,
            rope_short_m_scale=1.0,
            rope_long_m_scale=1.0,
            rope_max_positions=rope.positions,
            rope_original_max_positions=self.max_position_embeddings,
            use_paged_context_fmha=False,
            attention_input_type=attention_input_type,
            is_mla_enable=True,
            chunked_prefill_buffer_batch_size=chunked_prefill_buffer_batch_size,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim if context else self.kv_lora_rank,
            rope_append=True,
            mrope_rotary_cos_sin=None,
            mrope_position_deltas=None,
            helix_position_offsets=None,
            helix_is_inactive_rank=None,
            attention_chunk_size=None,
            softmax_stats_tensor=softmax_stats_tensor,
            is_spec_decoding_enabled=bool(md.is_spec_decoding_enabled),
            use_spec_decoding=bool(md.use_spec_decoding),
            is_spec_dec_tree=bool(md.is_spec_dec_tree),
            spec_decoding_generation_lengths=md.spec_decoding_generation_lengths if spec else None,
            spec_decoding_position_offsets=_spec_position_offsets(md) if spec else None,
            spec_decoding_packed_mask=md.spec_decoding_packed_mask if spec else None,
            spec_decoding_bl_tree_mask_offset=md.spec_decoding_bl_tree_mask_offset
            if spec
            else None,
            spec_decoding_bl_tree_mask=md.spec_decoding_bl_tree_mask if spec else None,
            spec_bl_tree_first_sparse_mask_offset_kv=(
                md.spec_bl_tree_first_sparse_mask_offset_kv if spec else None
            ),
            sparse_kv_indices=None,
            sparse_kv_offsets=None,
            sparse_attn_indices=None,
            sparse_attn_offsets=None,
            sparse_attn_indices_block_size=1,
            cu_q_seqlens=cu_q_seqlens,
            cu_kv_seqlens=cu_kv_seqlens,
            fmha_scheduler_counter=fmha_scheduler_counter,
            num_contexts=md.num_contexts,
            num_ctx_tokens=md.num_ctx_tokens,
            max_num_sequences=md.max_num_sequences,
            trtllm_gen_jit_warmup=bool(getattr(md, "trtllm_gen_jit_warmup", False)),
        )

    # -- context ------------------------------------------------------------------------------------------------

    def _context(self, q, compressed_kv, latent_cache, md, output, rope: _IdentityRope) -> None:
        """The context requests' attention: a fresh prompt over its own K / V; a cached prefix (block reuse or a
        later chunk) over the whole prefix, read from the cache in one piece or, with chunked prefill, chunk by
        chunk with the partials merged."""
        with_cached_kv = bool(md.enable_context_mla_with_cached_kv)
        cached = with_cached_kv and md.num_ctx_cached_tokens > 0
        if with_cached_kv:
            _warm_up_cat(self.num_heads_tp, q.dtype, q.device)
        if cached and md.runtime_features.chunked_prefill:
            self._context_chunked(q, compressed_kv, latent_cache, md, output, rope)
        elif cached:
            self._context_cached(q, latent_cache, md, output, rope)
        else:
            self._context_fresh(q, compressed_kv, latent_cache, md, output, rope)

    def _context_fresh(
        self, q, compressed_kv, latent_cache, md, output, rope: _IdentityRope
    ) -> None:
        """K / V of the prompt from its latent rows: kv_b per head with the K rows written straight into the
        strided K buffer (its rope half is left for the attention op, which takes k_pe from ``latent_cache``) and
        the V rows into the V half of a [k_nope | v] buffer, whose row stride the context kernel assumes."""
        heads, nope, v_dim, kv = (
            self.num_heads_tp,
            self.qk_nope_head_dim,
            self.v_head_dim,
            self.kv_lora_rank,
        )
        t = compressed_kv.shape[0]
        k = torch.empty_like(q).view(-1, heads, self.qk_head_dim)
        w = self.kv_b_proj.weight
        w_k = w[: heads * nope].view(heads, nope, kv)
        w_v = w[heads * nope :].view(heads, v_dim, kv)
        ckv = compressed_kv.unsqueeze(0).expand(heads, t, kv)
        kv_buf = torch.empty(t, heads * (nope + v_dim), dtype=q.dtype, device=q.device)
        v = kv_buf[:, heads * nope :]
        bmm_out(ckv, w_k.transpose(1, 2), k[..., :nope].transpose(0, 1))
        bmm_out(ckv, w_v.transpose(1, 2), v.unflatten(-1, (heads, v_dim)).transpose(0, 1))
        self._thop(
            md,
            rope,
            q=q,
            k=k.view(-1, heads * self.qk_head_dim),
            v=v,
            output=output,
            attention_input_type=_CONTEXT,
            mask_type=_MASK_CAUSAL,
            latent_cache=latent_cache,
            sequence_length=md.kv_lens_cuda_runtime,
            host_past_key_value_lengths=md.kv_lens_runtime,
            host_total_kv_lens=md.host_total_kv_lens,
        )

    def _append_and_assign_q(self, q, latent_cache, md, rope: _IdentityRope) -> None:
        """This chunk's latent rows into the paged cache, and q's rope half "rotated" in place (copied: NoPE)."""
        manager = md.kv_cache_manager
        mla_rope_append_paged_kv_assign_q(
            q=q,
            latent_cache=latent_cache,
            num_contexts=md.num_contexts,
            cu_ctx_cached_kv_lens=md.ctx_cached_token_indptr,
            cu_seq_lens=md.ctx_kv_indptr,
            max_input_uncached_seq_len=int(md.max_ctx_seq_len),
            cos_sin_cache=rope.cos_sin,
            head_num=self.num_heads_tp,
            nope_size=self.qk_nope_head_dim,
            rope_size=self.qk_rope_head_dim,
            lora_size=self.kv_lora_rank,
            kv_cache_block_offsets=md.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=manager.kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=manager.kv_cache_pool_mapping,
            kv_scale_orig_quant=None,
            residual_dim=0,
            layer_idx=self._local_layer(md),
            tokens_per_block=manager.tokens_per_block,
            attention_window_size=manager.max_seq_len,
            beam_width=1,
            quant_mode=_QUANT_MODE_BF16_KV,
        )

    def _expand_kv(self, compressed_kv: torch.Tensor, k_pe: torch.Tensor):
        """``(K [n, heads * 192], V [n, heads * 128] view)`` of latent rows: kv_b, then k_pe broadcast to every
        head."""
        heads, nope, v_dim = self.num_heads_tp, self.qk_nope_head_dim, self.v_head_dim
        kv = F.linear(compressed_kv, self.kv_b_proj.weight)
        k_nope, v = kv.split([heads * nope, heads * v_dim], -1)
        k = maybe_compiled_cat(
            (
                k_nope.view(-1, heads, nope),
                k_pe.view(-1, 1, self.qk_rope_head_dim).expand(-1, heads, -1),
            ),
            dim=-1,
        )
        return k.view(-1, heads * self.qk_head_dim), v

    def _context_cached(self, q, latent_cache, md, output, rope: _IdentityRope) -> None:
        """Chunked prefill off, a cached prefix (block reuse): the whole prefix and the chunk read back from the
        cache, expanded, and attended in one call."""
        self._append_and_assign_q(q, latent_cache, md, rope)
        manager = md.kv_cache_manager
        full_kv, full_k_pe = load_paged_kv_cache_for_mla(
            out_dtype=q.dtype,
            num_contexts=md.num_contexts,
            num_ctx_kv_tokens=md.num_ctx_cached_tokens + md.num_ctx_tokens,
            max_ctx_kv_len=int(md.max_ctx_kv_len),
            cu_ctx_kv_lens=md.ctx_kv_indptr,
            kv_cache_block_offsets=md.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=manager.kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=manager.kv_cache_pool_mapping,
            kv_scale_quant_orig=None,
            layer_idx=self._local_layer(md),
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            tokens_per_block=manager.tokens_per_block,
            attention_window_size=manager.max_seq_len,
            beam_width=1,
            quant_mode=_QUANT_MODE_BF16_KV,
        )
        k, v = self._expand_kv(full_kv, full_k_pe)
        del full_kv, full_k_pe
        self._thop(
            md,
            rope,
            q=q,
            k=k,
            v=v,
            output=output,
            attention_input_type=_CONTEXT,
            mask_type=_MASK_CAUSAL,
            latent_cache=None,
            sequence_length=md.kv_lens_cuda_runtime,
            host_past_key_value_lengths=md.kv_lens_runtime,
            host_total_kv_lens=md.host_total_kv_lens,
        )

    def _context_chunked(
        self, q, compressed_kv, latent_cache, md, output, rope: _IdentityRope
    ) -> None:
        """Chunked prefill on, a cached prefix: each chunk of the prefix the metadata plans
        (``chunked_loop_num``, at most ``chunk_size * chunked_prefill_buffer_batch_size`` tokens) read from the cache,
        expanded and attended without a mask with its softmax statistics kept, then the step's own tokens attended
        causally; the partials merged in that order on their statistics. Only one chunk's expanded K / V is held at a
        time."""
        self._append_and_assign_q(q, latent_cache, md, rope)
        heads, v_dim = self.num_heads_tp, self.v_head_dim
        tc = md.num_ctx_tokens
        num_contexts = md.num_contexts
        manager = md.kv_cache_manager
        stats = torch.empty((tc, heads, 2), dtype=torch.float, device=q.device)
        temp_stats = torch.empty((tc, heads, 2), dtype=torch.float, device=q.device)
        temp_out = q.new_empty((q.shape[0], heads * v_dim))
        buffer_batch = md.runtime_features.chunked_prefill_buffer_batch_size
        # The attention op reads the context KV total from here; each partial gets its own.
        host_total = md.host_total_kv_lens.clone()

        def attend_and_merge(k, v, mask_type, seq_lens, host_seq_lens, total, merge_op):
            host_total[0] = total
            self._thop(
                md,
                rope,
                q=q,
                k=k,
                v=v,
                output=temp_out,
                attention_input_type=_CONTEXT,
                mask_type=mask_type,
                latent_cache=None,
                sequence_length=seq_lens,
                host_past_key_value_lengths=host_seq_lens,
                host_total_kv_lens=host_total,
                softmax_stats_tensor=temp_stats,
                chunked_prefill_buffer_batch_size=buffer_batch,
            )
            merge_chunked_attention_for_mla(
                merged_attn=output,
                temp_attn=temp_out,
                merged_softmax_stats=stats,
                temp_softmax_stats=temp_stats,
                num_requests=num_contexts,
                cu_q_seq_lens=md.ctx_uncached_token_indptr,
                max_q_seq_len=int(md.max_ctx_seq_len),
                merge_op=merge_op,
                num_heads=heads,
                head_size=v_dim,
            )

        for loop in range(md.chunked_loop_num):
            total = int(md.host_cu_chunked_seq_len[loop, num_contexts])
            if md.max_ctx_cached_token_len == 0:
                chunk_kv = q.new_empty((0, self.kv_lora_rank))
                chunk_k_pe = q.new_empty((0, self.qk_rope_head_dim))
            else:
                chunk_kv, chunk_k_pe = load_chunked_kv_cache_for_mla(
                    out_dtype=q.dtype,
                    num_contexts=num_contexts,
                    num_ctx_cached_tokens=total,
                    cu_ctx_chunked_kv_lens=md.cu_chunked_seq_len[loop],
                    chunked_ld_global_offset=md.chunked_global_offset[loop],
                    kv_cache_block_offsets=md.kv_cache_block_offsets,
                    host_kv_cache_pool_pointers=manager.kv_cache_pool_pointers,
                    host_kv_cache_pool_mapping=manager.kv_cache_pool_mapping,
                    kv_scale_quant_orig=None,
                    layer_idx=self._local_layer(md),
                    lora_size=self.kv_lora_rank,
                    rope_size=self.qk_rope_head_dim,
                    tokens_per_block=manager.tokens_per_block,
                    chunked_max_seq_len=int(md.max_chunk_len_per_loop[loop]),
                    attention_window_size=manager.max_seq_len,
                    beam_width=1,
                    quant_mode=_QUANT_MODE_BF16_KV,
                )
            k, v = self._expand_kv(chunk_kv, chunk_k_pe)
            del chunk_kv, chunk_k_pe
            attend_and_merge(
                k,
                v,
                _MASK_PADDING,
                md.chunked_seq_len[loop],
                md.host_chunked_seq_len[loop],
                total,
                md.merge_op_tensor[loop],
            )
            del k, v

        # The step's own tokens, causally; k_pe as appended (the identity "rotation" of the latent rows).
        k_pe = latent_cache.view(-1, self.kv_lora_rank + self.qk_rope_head_dim)[
            :, self.kv_lora_rank :
        ]
        k, v = self._expand_kv(compressed_kv, k_pe)
        attend_and_merge(
            k,
            v,
            _MASK_CAUSAL,
            md.prompt_lens_cuda_runtime,
            md.prompt_lens_cpu_runtime,
            int(md.prompt_lens_cpu_runtime[:num_contexts].sum()),
            md.merge_op_tensor[md.chunked_loop_num],
        )

    # -- generation ---------------------------------------------------------------------------------------------

    def _generation(self, q, latent_cache, md, output, rope: _IdentityRope) -> None:
        """The generation requests: the k_b absorption into the fused query beside the rope / cache-append op, the
        latent attention, then the v_b projection."""
        heads, nope, kv = self.num_heads_tp, self.qk_nope_head_dim, self.kv_lora_rank
        lat = kv + self.qk_rope_head_dim
        num_tokens = q.shape[0]
        q_nope, q_pe = q.view(-1, heads, self.qk_head_dim).split([nope, self.qk_rope_head_dim], -1)
        num_seqs = md.num_seqs
        cu_q = torch.empty(num_seqs + 1, dtype=torch.int32, device=q.device)
        cu_kv = torch.empty(num_seqs + 1, dtype=torch.int32, device=q.device)
        counter = torch.empty(1, dtype=torch.uint32, device=q.device)
        fused_q = torch.empty(num_tokens, heads, lat, dtype=q.dtype, device=q.device)
        manager = md.kv_cache_manager

        def rope_generation():
            mla_rope_generation(
                fused_q=fused_q,
                q_pe=q_pe,
                latent_cache=latent_cache,
                rotary_cos_sin=rope.cos_sin,
                cu_q_seqlens=cu_q,
                cu_kv_seqlens=cu_kv,
                fmha_scheduler_counter=counter,
                mla_bmm1_scale=None,
                mla_bmm2_scale=None,
                quant_q_buffer=None,
                sequence_length=md.kv_lens_cuda_runtime,
                host_past_key_value_lengths=md.kv_lens_runtime,
                host_context_lengths=md.prompt_lens_cpu_runtime,
                num_contexts=md.num_contexts,
                kv_cache_block_offsets=md.kv_cache_block_offsets,
                host_kv_cache_pool_pointers=manager.kv_cache_pool_pointers,
                host_kv_cache_pool_mapping=manager.kv_cache_pool_mapping,
                kv_scale_orig_quant=None,
                kv_scale_quant_orig=None,
                kv_cache_scale_orig_quant=None,
                out_scale=None,
                block_ids_per_seq=None,
                helix_tensor_params=[None, None],
                # What the built-in module hands it whatever the verify width (the kernel reads the width from the
                # token and request counts).
                predicted_tokens_per_seq=1,
                layer_idx=self._local_layer(md),
                num_heads=heads,
                num_kv_heads=1,
                head_size=lat,
                residual_dim=0,
                tokens_per_block=manager.tokens_per_block,
                attention_window_size=md.max_seq_len,
                beam_width=md.beam_width,
                quant_mode=_QUANT_MODE_BF16_KV,
                q_scaling=self.q_scaling,
                q_lora_rank=self.q_lora_rank,
                kv_lora_rank=kv,
                qk_nope_head_dim=nope,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=kv,
                rope_append=True,
            )

        # Over a bf16 pool the two write disjoint halves of fused_q, so they overlap during CUDA-graph capture, as
        # the built-in module runs them.
        run_in_parallel(
            lambda: bmm_out(
                q_nope.transpose(0, 1),
                self.k_b_proj_trans.transpose(1, 2),
                fused_q[..., :kv].transpose(0, 1),
            ),
            rope_generation,
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        latent_out = self._latent_attention(
            fused_q.view(num_tokens, heads * lat),
            q_pe,
            latent_cache,
            md,
            rope,
            cu_q,
            cu_kv,
            counter,
        )
        bmm_out(
            latent_out.view(-1, heads, kv).transpose(0, 1),
            self.v_b_proj.transpose(1, 2),
            output.view(num_tokens, heads, self.v_head_dim).transpose(0, 1),
        )

    def _latent_attention(
        self, fused_q, q_pe, latent_cache, md, rope: _IdentityRope, cu_q, cu_kv, counter
    ) -> torch.Tensor:
        """The generation requests' latent MQA ``[tokens, heads * kv_lora]`` over the paged cache, on the kernel the
        built-in MLA's attention backend selects for this step: the CuTe DSL decode kernel on a decode-only step
        (``cute_dsl_mla_decode``), FlashInfer's TRTLLM-Gen MLA decode beside context requests
        (``flashinfer_mla_decode``), else the attention op's generation phase."""
        linear = md.beam_width == 1 and not md.is_spec_dec_tree
        if md.num_contexts == 0 and linear and _cute.available():
            return self._latent_attention_cute(fused_q, md)
        if md.num_contexts > 0 and linear and _flashinfer.available():
            total_num_blocks = _total_num_blocks(md.kv_cache_manager, self._local_layer(md))
            if total_num_blocks is not None:
                return self._latent_attention_flashinfer(fused_q, md, total_num_blocks)
        latent_out = fused_q.new_empty(fused_q.shape[0], self.num_heads_tp * self.kv_lora_rank)
        return self._thop(
            md,
            rope,
            q=fused_q,
            k=None,
            v=None,
            output=latent_out,
            attention_input_type=_GENERATION,
            mask_type=_MASK_CAUSAL,
            latent_cache=latent_cache,
            sequence_length=md.kv_lens_cuda_runtime,
            host_past_key_value_lengths=md.kv_lens_runtime,
            host_total_kv_lens=md.host_total_kv_lens,
            q_pe=q_pe,
            cu_q_seqlens=cu_q,
            cu_kv_seqlens=cu_kv,
            fmha_scheduler_counter=counter,
        )

    def _latent_attention_cute(self, fused_q: torch.Tensor, md) -> torch.Tensor:
        """``cute_dsl_mla_decode`` over this layer's pool: R generation requests of S tokens each, the page table
        and KV lengths as the stock CuTe DSL MLA library hands them (16-byte aligned, the per-step split-KV plan
        kept on the metadata and planned once per step), the kernel's workspace at the front of the attention
        workspace."""
        heads, kv = self.num_heads_tp, self.kv_lora_rank
        num_tokens = fused_q.shape[0]
        num_requests = md.num_generations
        seq_len_q = num_tokens // num_requests
        manager = md.kv_cache_manager
        local = self._local_layer(md)
        pool, layer_in_pool, layers_in_pool = _packed_pool(manager.get_buffers(self.layer_idx))
        pool_idx = int(md.host_kv_cache_pool_mapping[local, 0])
        rows = md.kv_cache_block_offsets[pool_idx, :, 0, :][:num_requests].to(torch.int32)
        if layer_in_pool:
            rows = rows + layer_in_pool
        page_table = _aligned(rows).transpose(0, 1)
        if layers_in_pool == 1 and getattr(manager, "_use_per_layer_page_tables", False):
            # Per-layer page indices include the layer offset, but get_buffers() already starts at this layer.
            offset = int(manager.impl.get_page_index_converter(local, "key").layer_offset)
            if offset:
                page_table = torch.where(
                    page_table == _BAD_PAGE_INDEX, page_table, page_table - offset
                )
        cache_seqs = _decode_kv_lengths(md, md.kv_lens_cuda_runtime[:num_requests])
        workspace = md.effective_workspace
        need = _cute.workspace_bytes(heads, seq_len_q, kv, md.max_num_requests)
        if workspace.numel() * workspace.element_size() < need:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Kimi K3 MLA: the attention workspace is smaller than the CuTe DSL decode kernel needs inside "
                    "a CUDA graph capture"
                )
            workspace.resize_((math.ceil(need / workspace.element_size()),))
        workspace = workspace[: math.ceil(need / workspace.element_size())]
        plan = _split_kv_plan(md, cache_seqs, num_requests, seq_len_q, heads)
        latent_out = fused_q.new_empty(num_tokens, heads * kv)
        return _cute.cute_dsl_mla_decode(
            fused_q,
            latent_out,
            pool,
            page_table,
            cache_seqs,
            workspace,
            heads,
            seq_len_q,
            md.tokens_per_block,
            kv,
            self.qk_rope_head_dim,
            1.0 / (math.sqrt(self.qk_head_dim) * self.q_scaling),
            int(md.max_num_requests),
            plan,
        )

    def _latent_attention_flashinfer(
        self, fused_q: torch.Tensor, md, total_num_blocks: int
    ) -> torch.Tensor:
        """``flashinfer_mla_decode`` over this layer's pool for the generation requests of a mixed step: the
        workspace grown to the stock library's requirement, the layer's multi-CTA KV counter, the generation rows of
        the block offsets."""
        heads, kv, rope_dim = self.num_heads_tp, self.kv_lora_rank, self.qk_rope_head_dim
        lat = kv + rope_dim
        num_tokens = fused_q.shape[0]
        num_contexts, num_requests = md.num_contexts, md.num_generations
        seq_len_q = num_tokens // num_requests
        max_num_sequences = md.max_num_sequences or md.max_num_requests
        workspace = md.effective_workspace
        need = _flashinfer.workspace_bytes(
            fused_q.dtype,
            num_tokens,
            md.max_context_length,
            max_num_sequences,
            heads,
            lat,
            rope_dim,
        )
        counter_size = _flashinfer.counter_bytes(
            heads,
            max_num_sequences,
            torch.cuda.get_device_properties(fused_q.device).multi_processor_count,
        )
        grow = workspace.numel() * workspace.element_size() < need
        new_counter = self._fi_counter is None or self._fi_counter.numel() < counter_size
        if (grow or new_counter) and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Kimi K3 MLA: the FlashInfer MLA decode workspace or counter must be sized before CUDA graph capture"
            )
        if grow:
            workspace.resize_((math.ceil(need / workspace.element_size()),))
        if new_counter:
            self._fi_counter = torch.zeros(counter_size, dtype=torch.uint8, device=fused_q.device)
        kv_cache, block_tables = _flashinfer.paged_cache(
            md.host_kv_cache_pool_pointers,
            md.host_kv_cache_pool_mapping,
            md.kv_cache_block_offsets,
            self._local_layer(md),
            md.tokens_per_block,
            lat,
            total_num_blocks,
            num_contexts,
            num_requests,
            fused_q.dtype,
        )
        latent_out = fused_q.new_empty(num_tokens, heads * kv)
        _flashinfer.flashinfer_mla_decode(
            fused_q.view(num_requests, seq_len_q, heads, lat),
            kv_cache,
            workspace,
            block_tables,
            md.kv_lens_cuda_runtime[num_contexts:],
            int(md.kv_lens_runtime[num_contexts : num_contexts + num_requests].max()),
            latent_out.view(num_requests, seq_len_q, heads, kv),
            self.qk_nope_head_dim,
            kv,
            rope_dim,
            _flashinfer.bmm1_scale(self.qk_nope_head_dim, rope_dim, self.q_scaling),
            _flashinfer.enable_pdl(),
            self._fi_counter,
        )
        return latent_out


def _total_num_blocks(manager, local_layer: int) -> Optional[int]:
    """The page-index bound of the layer's pool that FlashInfer's paged cache view spans, as the stock library takes
    it: a cache view's own bound, else a KVCacheManagerV2's page-index upper bound (None for another manager)."""
    view_bound = getattr(manager, "get_attention_op_num_blocks", None)
    if view_bound is not None:
        return int(view_bound(local_layer))
    impl = getattr(manager, "impl", None)
    if impl is not None and hasattr(impl, "get_page_index_upper_bound"):
        return int(impl.get_page_index_upper_bound(local_layer, "key"))
    return None


def _packed_pool(pool: torch.Tensor):
    """``(pool, layer_in_pool, layers_in_pool)``: a layer's view of a layer-interleaved pool as the whole pool with
    the layer's slot as a page offset (the decode kernels and the CuTe DSL kernel address pages by the packed block
    stride), or the pool itself."""
    packed_block = 1
    for size in pool.shape[1:]:
        packed_block *= size
    block_stride = pool.stride(0)
    layers_in_pool = block_stride // packed_block if packed_block else 1
    if layers_in_pool > 1 and block_stride == layers_in_pool * packed_block:
        layer_in_pool = pool.storage_offset() // packed_block
        pool = pool.as_strided(
            (pool.shape[0] * layers_in_pool, *pool.shape[1:]),
            (packed_block, *pool.stride()[1:]),
            0,
        )
        return pool, layer_in_pool, layers_in_pool
    return pool, 0, 1


def _aligned(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor``, or a contiguous copy when its pointer breaks the CuTe DSL runner's 16-byte alignment."""
    if tensor.data_ptr() % 16 == 0:
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def _decode_kv_lengths(md, lengths: torch.Tensor) -> torch.Tensor:
    """This step's int32 KV lengths with the kernel's pointer alignment: a misaligned view is copied into a staging
    buffer kept on the metadata, so every layer of the step passes the same pointer and the split plan is made once
    per step."""
    lengths = lengths.to(torch.int32)
    if lengths.data_ptr() % 16 == 0:
        return lengths
    n = lengths.numel()
    staging = getattr(md, "_cute_dsl_mla_kv_lens_staging", None)
    if staging is None or staging.numel() < n or staging.device != lengths.device:
        if lengths.is_cuda and torch.cuda.is_current_stream_capturing():
            return _aligned(lengths)
        capacity = max(int(getattr(md, "max_num_requests", 0) or 0), n)
        staging = torch.empty(capacity, dtype=torch.int32, device=lengths.device)
        md._cute_dsl_mla_kv_lens_staging = staging
    staged = staging[:n].view(lengths.shape)
    staged.copy_(lengths)
    return staged


def _split_kv_plan(md, cache_seqs, num_requests, seq_len_q, num_heads) -> Optional[torch.Tensor]:
    """This step's per-request split-KV counts, planned by the step's first MLA layer and read by the others: the
    plan buffer and its key live on the metadata (``_cute_dsl_mla_split_plan``, ``_cute_dsl_mla_split_plan_key``,
    which the metadata resets when the lengths change). None (the tactic's uniform split) when it cannot be planned
    here."""
    if not 0 < num_requests <= _cute.mla_split_kv_plan.MAX_REQUESTS:
        return None
    plan = getattr(md, "_cute_dsl_mla_split_plan", None)
    capacity = max(int(getattr(md, "max_num_requests", 0) or 0), num_requests)
    if plan is None or plan.numel() < num_requests or plan.device != cache_seqs.device:
        if torch.cuda.is_current_stream_capturing():
            return None
        plan = torch.empty(capacity, dtype=torch.int32, device=cache_seqs.device)
        md._cute_dsl_mla_split_plan = plan
    rows, waves = _cute.mla_split_kv_plan.geometry(num_heads, seq_len_q)
    key = (num_requests, rows, waves, cache_seqs.data_ptr(), plan.data_ptr())
    if getattr(md, "_cute_dsl_mla_split_plan_key", None) != key:
        _cute.mla_split_kv_plan(cache_seqs, plan, num_requests, rows, waves)
        md._cute_dsl_mla_split_plan_key = key
    return plan[:num_requests]


def _spec_position_offsets(md) -> Optional[torch.Tensor]:
    """The speculative position offsets as the attention op takes them on SM100: ``[max_num_requests, width]``."""
    offsets = md.spec_decoding_position_offsets
    if offsets is not None and offsets.dim() == 1:
        offsets = offsets.view(md.max_num_requests, -1)
    return offsets


_CAT_WARMED: set = set()


def _warm_up_cat(num_heads: int, dtype: torch.dtype, device: torch.device) -> None:
    """Compile ``maybe_compiled_cat`` for the K layouts the cached-prefix paths concatenate, once per head count, as
    the built-in MLA does before a step that may attend a cached prefix: the token axis is marked dynamic so one
    compile serves every token count but 1, which torch.compile specializes and gets its own."""
    key = (num_heads, dtype, device)
    if key in _CAT_WARMED:
        return
    _CAT_WARMED.add(key)
    nope, rope_dim, kv_lora, v_dim = 128, 64, 512, 128

    def warm(num_tokens: int) -> None:
        k_nope = torch.empty(num_tokens, num_heads * (nope + v_dim), dtype=dtype, device=device)[
            :, : num_heads * nope
        ].view(num_tokens, num_heads, nope)
        chunk_k_pe = torch.empty(num_tokens, 1, rope_dim, dtype=dtype, device=device).expand(
            -1, num_heads, -1
        )
        k_pe = torch.empty(num_tokens, 1, kv_lora + rope_dim, dtype=dtype, device=device)[
            :, :, -rope_dim:
        ].expand(-1, num_heads, -1)
        for t in (k_nope, chunk_k_pe, k_pe):
            torch._dynamo.maybe_mark_dynamic(t, 0)
        maybe_compiled_cat((k_nope, chunk_k_pe), dim=-1)
        maybe_compiled_cat((k_nope, k_pe), dim=-1)

    warm(2)
    warm(1)
