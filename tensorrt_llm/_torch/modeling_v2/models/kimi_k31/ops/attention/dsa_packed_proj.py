# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 DSA MLA token-wise projections on a packed hidden-state weight, with the low-M GEMM plan.

Every projection of the layer input ``hidden_states`` -- the fused ``[q_a | kv_a]`` (``kv_a_proj_with_mqa``,
1536 + 512 + 64 rows), the indexer's ``wk`` (128) and ``weights_proj`` (32), and optionally the output gate
``g_proj`` (heads x 128) -- is one GEMM on a packed ``[N, hidden]`` weight instead of three, and each column
block of its output feeds what the separate GEMM's output fed: the q_a / kv_a norms, the rope slot, the index-key
LayerNorm, the index weights, and (``_apply_output_gate``) the sigmoid gate. The normed ``q_a`` feeds ``q_b``
and the indexer's ``wq_b`` as two GEMMs: packing those as well would need both output blocks copied to
contiguous tensors (the MLA query and the FP8 cat-quantize of the index query take one row stride), and the
two copies cost more than the second launch.

``pack_dsa_hidden_projection`` runs once after the checkpoint is loaded (a ``weights.py`` step): it builds the
packed weight and re-points ``kv_a_proj_with_mqa``, ``wk``, ``weights_proj``, the indexer's fused
``[wk | weights_proj]`` weight and ``g_proj`` to row views of it, so the layer keeps working on paths that bypass
the packed GEMM and no weight memory is added. ``forward_dsa_proj_packed`` is ``ops/attention/dsa_graph_ops``'s
``forward_dsa_proj`` on that weight: the same outputs, from the same kernels except the GEMMs (and, with a plan,
the q_a norm, which runs FlashInfer's RMSNorm code with its launch trigger moved to the start: the same bits).

Numerics: the packed GEMM's columns are the separate GEMMs' columns, computed by another kernel (another
summation order); a low-M plan row changes the kernel again. Neither is bitwise the general target's cuBLAS
calls; the rest of the chain is unchanged.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.gemm.low_m_gemm import LowMPlan
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.norm.q_norm_release import (
    can_use as _release_norm_can_use,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_index_key_norm import (  # noqa: E501
    indexer_k_norm,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.common.multi_stream import (
    maybe_execute_in_parallel,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.low_m_linear import k31_low_m_linear
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm.rmsnorm_release import (
    k31_rmsnorm_release,
)

#: Attributes ``pack_dsa_hidden_projection`` sets on the MLA layer.
PACKED_HIDDEN_WEIGHT = "_k31_packed_hidden_weight"
PACKED_GATE_ROWS = "_k31_packed_gate_rows"
PACKED_GATE_OUT = "_k31_packed_gate_out"


def _plain_bf16(linear) -> bool:
    """A 2-D BF16 projection weight (K3.1's projections are bias-free and unquantized)."""
    weight = getattr(linear, "weight", None)
    return weight is not None and weight.dtype == torch.bfloat16 and weight.dim() == 2


def pack_dsa_hidden_projection(attn) -> bool:
    """Pack ``attn``'s hidden-state projections into one weight; returns whether it did.

    ``attn`` is a K3.1 MLA layer with a DSA indexer (``attn.mqa.indexer``) whose ``kv_a_proj_with_mqa``, ``q_b_proj``,
    ``wk``, ``weights_proj`` and ``wq_b`` are plain BF16 weights; its output gate ``g_proj`` is packed too when it is
    one as well. Row order: ``[q_a | kv_a | k_pe | wk | weights_proj (| g_proj)]``. Call after every weight is
    loaded and after the indexer has built its fused ``[wk | weights_proj]`` weight (which this replaces with a
    view).
    """
    indexer = getattr(getattr(attn, "mqa", None), "indexer", None)
    kv_a = getattr(attn, "kv_a_proj_with_mqa", None)
    if indexer is None:
        return False
    if not all(
        _plain_bf16(m)
        for m in (
            kv_a,
            getattr(attn, "q_b_proj", None),
            indexer.wk,
            indexer.weights_proj,
            indexer.wq_b,
        )
    ):
        return False
    g_proj = getattr(attn, "g_proj", None) if getattr(attn, "use_output_gate", False) else None
    pack_gate = _plain_bf16(g_proj) and g_proj.weight.shape[1] == kv_a.weight.shape[1]
    rows = [kv_a.weight.data, indexer.wk.weight.data, indexer.weights_proj.weight.data]
    if pack_gate:
        rows.append(g_proj.weight.data)
    if any(r.shape[1] != kv_a.weight.shape[1] for r in rows):
        return False
    packed = torch.cat(rows, dim=0).contiguous()
    n_kv_a, n_wk, n_wp = (r.shape[0] for r in rows[:3])
    kv_a.weight.data = packed[:n_kv_a]
    indexer.wk.weight.data = packed[n_kv_a : n_kv_a + n_wk]
    indexer.weights_proj.weight.data = packed[n_kv_a + n_wk : n_kv_a + n_wk + n_wp]
    indexer._fused_wk_wp_weight = packed[n_kv_a : n_kv_a + n_wk + n_wp]
    gate_rows = 0
    if pack_gate:
        gate_rows = g_proj.weight.shape[0]
        g_proj.weight.data = packed[packed.shape[0] - gate_rows :]
    setattr(attn, PACKED_HIDDEN_WEIGHT, packed)
    setattr(attn, PACKED_GATE_ROWS, gate_rows)
    setattr(attn, PACKED_GATE_OUT, None)
    return True


def _indexer_from_projections(
    indexer, q: torch.Tensor, indexer_k: torch.Tensor, weights: torch.Tensor
):
    """``K31Indexer._finish_pre_indexer_proj`` with the ``wq_b`` projection ``q`` already computed.

    K3.1's indexer is position-free (no rope): each 128-wide head is cat-quantized from its two halves.
    """
    k = indexer_k_norm(indexer, indexer_k)
    q = q.view(-1, indexer.n_heads, indexer.head_dim)
    half = indexer.head_dim // 2
    q_pe, q_nope = q.split([half, indexer.head_dim - half], dim=-1)
    k_pe, k_nope = k.split([half, indexer.head_dim - half], dim=-1)
    q_out, k_out = maybe_execute_in_parallel(
        lambda: indexer._prep_q_or_k(q_pe, q_nope),
        lambda: indexer._prep_q_or_k(k_pe, k_nope),
        indexer.ln_events[0],
        indexer.ln_events[1],
        indexer.aux_stream,
    )
    q_fp8, q_scale = q_out
    k_fp8, k_scale = k_out
    q_fp8 = q_fp8.view(-1, indexer.n_heads, indexer.head_dim)
    q_scale = q_scale.view(-1, indexer.n_heads, 1)
    weights = indexer._weight_scale(weights, q_scale)
    return q_fp8, k_fp8, k_scale, weights


def forward_dsa_proj_packed(
    self,
    position_ids: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    attn_metadata,
    plan: Optional[LowMPlan],
    pdl: bool,
    defer_index_quant: bool = False,
    defer_q_b: bool = False,
) -> List[torch.Tensor]:
    """``forward_dsa_proj`` on the packed hidden weight (``pack_dsa_hidden_projection`` ran on ``self``).

    Same outputs: ``[q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8, k_scale, weights]``. The
    GEMMs go through ``k31_low_m_linear`` with ``plan`` (``None``: cuBLAS); the gate block, when packed, is
    left in ``self._k31_packed_gate_out`` for ``packed_output_gate``. Like ``forward_dsa_proj`` it neither
    slices by the token count nor reads batch metadata (CUDA-graph safe).

    With ``defer_index_quant`` the indexer's FP8 quantization, weight scaling and the latent concat are
    left to the fused prologue in Op 2 (``dsa_fp8_decode_prologue``, which also writes the index keys into
    the cache): the outputs are ``[q, compressed_kv, kpe_tail, latent_cache, index_q, index_k]`` with
    ``kpe_tail`` the packed output's columns from ``k_pe`` on (``[k_pe | wk | weights_proj (| g_proj)]``:
    one view carries both ``k_pe`` and the BF16 index weights, as custom-op outputs may not alias each
    other), ``latent_cache`` uninitialized, ``index_q`` the BF16 ``wq_b`` output ``[T, heads, 128]`` and
    ``index_k`` the BF16 normed index key ``[T, 128]`` (its LayerNorm runs beside the kv_a norm).

    With ``defer_q_b`` the ``q`` output is the normed ``q_a`` ``[T, q_lora_rank]`` instead of its ``q_b``
    projection: the module runs that GEMM in Op 2 (``DeferredQB.project``, the same call), where it can overlap
    the indexer.
    """
    del position_ids, attn_metadata  # the K3.1 MLA and indexer are position-free here
    assert isinstance(hidden_states, torch.Tensor), "the packed path takes the BF16 layer input"
    indexer = self.mqa.indexer
    gate_rows = getattr(self, PACKED_GATE_ROWS)
    projected = k31_low_m_linear(hidden_states, getattr(self, PACKED_HIDDEN_WEIGHT), plan, pdl)
    sizes = [
        self.q_lora_rank,
        self.kv_lora_rank,
        self.qk_rope_head_dim,
        indexer.head_dim,
        indexer.n_heads,
    ]
    q, compressed_kv, k_pe, indexer_k, index_weights, *gate = projected.split(
        sizes + ([gate_rows] if gate_rows else []), -1
    )
    setattr(self, PACKED_GATE_OUT, (hidden_states, gate[0]) if gate else None)

    def q_side():
        # The q_a norm, then q_b and wq_b on the same stream right behind it: a planned norm releases the GEMMs
        # at launch, so their weights stream while the norm (and the kv_a norm on the side stream) runs.
        if release_norm:
            q_normed = qr = k31_rmsnorm_release(
                q, self.q_a_layernorm.weight, self.q_a_layernorm.variance_epsilon
            )
        else:
            q_normed, qr = self._q_a_layernorm(q)
        return (
            q_normed if defer_q_b else k31_low_m_linear(q_normed, self.q_b_proj.weight, plan, pdl),
            k31_low_m_linear(qr, indexer.wq_b.weight, plan, pdl),
        )

    release_norm = plan is not None and pdl and _release_norm_can_use(q, self.q_a_layernorm.weight)
    if defer_index_quant:
        (q, index_q), (compressed_kv, index_k) = maybe_execute_in_parallel(
            q_side,
            lambda: (self._kv_a_layernorm(compressed_kv), indexer_k_norm(indexer, indexer_k)),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        # The fused prologue in Op 2 writes the latent rows; this keeps the output list's shapes.
        latent_cache = compressed_kv.new_empty(
            (compressed_kv.shape[0], compressed_kv.shape[1] + k_pe.shape[1])
        )
        index_q = index_q.view(-1, indexer.n_heads, indexer.head_dim)
        kpe_tail = projected[:, self.q_lora_rank + self.kv_lora_rank :]
        return [q, compressed_kv, kpe_tail, latent_cache, index_q, index_k]
    (q, index_q), compressed_kv = maybe_execute_in_parallel(
        q_side,
        lambda: self._kv_a_layernorm(compressed_kv),
        self.ln_events[0],
        self.ln_events[1],
        self.aux_stream,
    )
    latent_cache = torch.concat([compressed_kv, k_pe], dim=-1)
    q_fp8, k_fp8, k_scale, weights = _indexer_from_projections(
        indexer, index_q, indexer_k, index_weights
    )
    return [q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8, k_scale, weights]


def packed_output_gate(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
    """The ``g_proj(hidden_states)`` block the last packed GEMM produced for this input, or None.

    Consumed once: the stash is cleared, so a later call on another input recomputes the gate.
    """
    packed = getattr(self, PACKED_GATE_OUT, None)
    setattr(self, PACKED_GATE_OUT, None)
    if packed is None or packed[0].data_ptr() != hidden_states.data_ptr():
        return None
    gate = packed[1]
    return gate if gate.shape[0] == hidden_states.shape[0] else None
