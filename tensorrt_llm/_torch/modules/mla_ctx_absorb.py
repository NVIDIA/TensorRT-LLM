# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Absorbed-MLA attention for a short context chunk over a long cached prefix.

The default context path (``MLA.forward_context_with_chunked_prefill``) re-expands the whole cached
prefix (``kv_b_proj`` over every cached latent, per-head K/V, 3-4 chunk loops with a 24-CTA FMHA each)
for every chunk, so its cost grows with the cached prefix regardless of the new-token count.
For a chunk of N new tokens (N a few hundred to a few thousand) it is cheaper to attend in latent
space, as the decode step does: the FP8 CuTe DSL MLA decode kernel
(``trtllm::cute_dsl_mla_decode_fp8_blackwell``, ``mla_decode_fp8.py``) reads the 576-wide FP8 latent
cache directly and applies the implicit causal mask "query token t of S attends keys
``[0, K - (S - 1) + t)``" (the speculative-decoding mask).

Mapping a context chunk onto that kernel: the N new tokens are split into G groups of S0 query tokens;
group j is one pseudo-request with ``seq_len_q = S0``, its own cached length
``K_j = C - pad + (j + 1) * S0`` (C = cached tokens, ``pad = G * S0 - N`` dummy rows at the front of
group 0) and the same page-table row as the real request. Row p of the padded sequence is real token
``p - pad`` and its mask bound is ``K_j - S0 + 1 + t = C + (p - pad) + 1``, i.e. exactly causal.

Enabled with ``TRTLLM_K3_MLA_CTX_ABSORB=1``; applies to a single-context-request batch with
``cached >= TRTLLM_K3_MLA_CTX_ABSORB_MIN_CACHED`` (default 4096) and
``N <= TRTLLM_K3_MLA_CTX_ABSORB_MAX_TOKENS`` (default 1024); everything else keeps the chunked-prefill
path. ``TRTLLM_K3_MLA_CTX_ABSORB_GROUP`` (default 8) is S0; use ``draft_len + 1`` of the decode so
the already compiled decode kernel variant is reused. ``TRTLLM_K3_MLA_CTX_ABSORB_CHECK=<n>`` runs both
paths for the first ``n`` eligible calls and logs the error of the absorbed path against the default
path and of the default path against itself (nondeterminism floor).
"""

import math
import os
from typing import Optional

import torch

from tensorrt_llm.logger import logger

ENABLED = os.environ.get("TRTLLM_K3_MLA_CTX_ABSORB", "0") == "1"
MAX_TOKENS = int(os.environ.get("TRTLLM_K3_MLA_CTX_ABSORB_MAX_TOKENS", "1024"))
MIN_CACHED = int(os.environ.get("TRTLLM_K3_MLA_CTX_ABSORB_MIN_CACHED", "4096"))
GROUP = int(os.environ.get("TRTLLM_K3_MLA_CTX_ABSORB_GROUP", "8"))
CHECK = int(os.environ.get("TRTLLM_K3_MLA_CTX_ABSORB_CHECK", "0"))
# TRTLLM_K3_MLA_CTX_ABSORB_FAST=1 (default off): host-trimmed launch of the absorbed layer. Same kernels,
# same tactic and same operands as the default absorbed path (bitwise identical output); the per-layer
# page-table / cache-length / query-buffer preparation is done once per step, and the CuTe decode kernel
# is replayed with pre-wrapped arguments instead of the per-call op (runner construction, autotuner lookup,
# workspace layout, 12 tensor adaptations).
FAST = os.environ.get("TRTLLM_K3_MLA_CTX_ABSORB_FAST", "0") == "1"

_state = {"checked": 0, "workspace": None}


def eligible(mla, attn_metadata, q: torch.Tensor, latent_cache: Optional[torch.Tensor]) -> bool:
    """Whether this context call can take the absorbed path."""
    if not ENABLED or latent_cache is None:
        return False
    if attn_metadata.num_contexts != 1 or attn_metadata.num_generations != 0:
        return False
    n = attn_metadata.num_ctx_tokens
    cached = attn_metadata.num_ctx_cached_tokens
    if n <= 0 or n > MAX_TOKENS or cached < MIN_CACHED or cached < GROUP:
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    attn = mla.mha
    if not getattr(attn, "has_fp8_kv_cache", False) or mla.out_scale is not None:
        return False
    if q.dtype != torch.bfloat16 or not hasattr(mla, "k_b_proj_trans"):
        return False
    w_kc = _w_kc_bf16(mla)
    if w_kc is None:
        return False
    return _unit_kv_scale(mla, attn)


def _unit_kv_scale(mla, attn) -> bool:
    """The absorbed call passes unit bmm scales to the decode kernel, which is only right for a
    unit FP8 KV-cache scale. Read the scale once per layer (a host sync, never inside graph
    capture) and cache the verdict."""
    flag = getattr(mla, "_ctx_absorb_unit_kv_scale", None)
    if flag is None:
        scale = getattr(attn, "kv_cache_scaling_factor", None)
        flag = scale is None or (scale.numel() == 1 and float(scale.item()) == 1.0)
        mla._ctx_absorb_unit_kv_scale = flag
        if not flag:
            logger.warning(
                "TRTLLM_K3_MLA_CTX_ABSORB: FP8 KV-cache scale is not 1; "
                "keeping the chunked-prefill path."
            )
    return flag


def _w_kc_bf16(mla) -> Optional[torch.Tensor]:
    w = mla.k_b_proj_trans if mla.k_b_proj_trans.dtype == torch.bfloat16 else mla.k_b_proj_trans_dequant
    if w is None or w.dtype != torch.bfloat16:
        return None
    return w


_BMM_SCALES = {}


def _bmm_scales(device, softmax_scale: float):
    """Device-resident (bmm1, bmm2) scale tensors per (device, softmax scale), built once:
    creating them per call is a blocking host-to-device copy in every absorbed MLA layer."""
    key = (str(device), softmax_scale)
    scales = _BMM_SCALES.get(key)
    if scales is None:
        bmm1 = torch.tensor(
            [softmax_scale, softmax_scale * 1.4426950408889634], dtype=torch.float32, device=device
        )
        bmm2 = torch.ones(1, dtype=torch.float32, device=device)
        scales = _BMM_SCALES[key] = (bmm1, bmm2)
    return scales


def _workspace(num_heads: int, kv_lora_rank: int, max_groups: int, device) -> torch.Tensor:
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    need = CuteDSLNVMlaDecodeBlackwellRunner.get_max_padded_workspace_size(
        num_heads, GROUP, kv_lora_rank, max_groups, cutlass.Float32
    )
    ws = _state["workspace"]
    if ws is None or ws.numel() < need or ws.device != device:
        ws = torch.empty(need, dtype=torch.uint8, device=device)
        _state["workspace"] = ws
    return ws


def forward_absorbed_context(
    mla,
    q: torch.Tensor,
    latent_cache: torch.Tensor,
    attn_metadata,
    output: torch.Tensor,
) -> torch.Tensor:
    """Attention of ``q`` (``[N, H * qk_head_dim]``) over cached + new latents into ``output``
    (``[N, H * v_head_dim]``), same contract as ``forward_context_with_chunked_prefill``."""
    from ..custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner  # noqa: F401
    from .mla_fp8_q_absorb import mla_fp8_q_absorb_nope

    if FAST and _forward_fast(mla, q, latent_cache, attn_metadata, output):
        return output

    attn = mla.mha
    mqa = mla.mqa
    n = attn_metadata.num_ctx_tokens
    cached = int(attn_metadata.num_ctx_cached_tokens)
    heads = mla.num_heads_tp
    lora = mla.kv_lora_rank
    rope = mla.qk_rope_head_dim
    nope = mla.qk_nope_head_dim
    s0 = GROUP
    groups = (n + s0 - 1) // s0
    pad = groups * s0 - n
    rows = groups * s0
    device = q.device
    page_size = attn_metadata.kv_cache_manager.tokens_per_block

    # rope (identity for NoPE K3), append the new latents to the paged cache, assign q_pe into q.
    attn.mla_rope_append_paged_kv_assign_q(q, latent_cache, attn_metadata)

    # FP8 absorbed query [rows, H, lora + rope]; dummy rows at the front stay zero.
    quant_q = torch.zeros(rows, heads, lora + rope, dtype=torch.float8_e4m3fn, device=device)
    mla_fp8_q_absorb_nope(
        q.view(n, heads, nope + rope), _w_kc_bf16(mla), quant_q[pad:], nope, rope
    )

    # Page table: the request's row replicated per group, [max_blocks, groups] like decode.
    pool_mapping = attn_metadata.host_kv_cache_pool_mapping
    local_layer = attn.get_local_layer_idx(attn_metadata)
    pool_idx = int(pool_mapping[local_layer, 0])
    num_pages = (cached + n + page_size - 1) // page_size
    row = attn_metadata.kv_cache_block_offsets[pool_idx, 0, 0, :num_pages].to(torch.int32)

    kv_pool = attn_metadata.kv_cache_manager.get_buffers(attn.layer_idx)
    packed_block = 1
    for size in kv_pool.shape[1:]:
        packed_block *= size
    block_stride = kv_pool.stride(0)
    layers_in_pool = block_stride // packed_block if packed_block else 1
    layer_in_pool = 0
    if layers_in_pool > 1 and block_stride == layers_in_pool * packed_block:
        layer_in_pool = kv_pool.storage_offset() // packed_block
        kv_pool = kv_pool.as_strided(
            (kv_pool.shape[0] * layers_in_pool, *kv_pool.shape[1:]),
            (packed_block, *kv_pool.stride()[1:]),
            0,
        )
        row = row + layer_in_pool
    kv_typed = kv_pool.view(torch.float8_e4m3fn)
    if kv_typed.dim() != 5 or kv_typed.shape[1] != 1 or kv_typed.shape[3] != 1:
        raise RuntimeError(f"unexpected KV cache layout {tuple(kv_typed.shape)}")
    kv_pages = kv_typed[:, 0, :, 0, : lora + rope]
    c_latent = kv_pages[..., :lora].permute(1, 2, 0)
    c_rope = kv_pages[..., lora:].permute(1, 2, 0)
    page_table = row.unsqueeze(0).expand(groups, num_pages).contiguous().transpose(0, 1)

    cache_seqs = (
        torch.arange(groups, dtype=torch.int32, device=device) * s0 + (cached - pad + s0)
    ).contiguous()

    softmax_scale = float(1.0 / (math.sqrt(nope + rope) * mqa.q_scaling))
    bmm1, bmm2 = _bmm_scales(device, softmax_scale)

    max_groups = (MAX_TOKENS + s0 - 1) // s0 + 1
    workspace = _workspace(heads, lora, max_groups, device)

    q4 = quant_q.view(groups, s0, heads, lora + rope)
    q_latent = q4[..., :lora].permute(2, 3, 1, 0)
    q_rope = q4[..., lora:].permute(2, 3, 1, 0)
    out_latent = torch.empty(groups, s0, heads, lora, dtype=torch.bfloat16, device=device)
    o_kernel = out_latent.permute(2, 3, 1, 0)

    torch.ops.trtllm.cute_dsl_mla_decode_fp8_blackwell(
        q_latent,
        q_rope,
        c_latent,
        c_rope,
        page_table,
        cache_seqs,
        o_kernel,
        workspace,
        heads,
        s0,
        page_size,
        bmm1,
        bmm2,
        int(max_groups),
        None,
        None,
        None,
    )

    attn_out_latent = out_latent.view(rows, heads, lora)[pad:]
    attn_output = output.view(n, heads, mla.v_head_dim)
    if mla.v_b_proj.dtype == torch.bfloat16:
        mla._bmm_bf16_out(
            attn_out_latent.transpose(0, 1),
            mla.v_b_proj,
            mla.v_b_proj.transpose(1, 2),
            attn_output.transpose(0, 1),
            mla._ugpu_v_b_proj_shards,
        )
    elif mla.v_b_proj.dtype == torch.float8_e4m3fn:
        from .mla import fp8_block_scaling_bmm_out

        fp8_block_scaling_bmm_out(
            attn_out_latent.contiguous(),
            mla.v_b_proj,
            mla.v_b_proj_scale,
            attn_output.transpose(0, 1),
            mla.v_b_proj_dequant,
            mla.use_cute_dsl_blockscaling_bmm,
        )
    else:
        raise NotImplementedError(f"Missing bmm impl for dtype: {mla.v_b_proj.dtype}.")
    return output


# ---------------------------------------------------------------------------------------------------
# TRTLLM_K3_MLA_CTX_ABSORB_FAST=1: host-trimmed absorbed layer (see the FAST flag above).
#
# The 24 MLA layers of a context step share their page-table row, group count, cache lengths, query
# buffer, workspace, scales and kernel variant; only the KV-pool view (and the layer's offset into the
# page table) differs. The first absorbed layer of a step builds a plan (one real runner call that
# returns the compiled kernel and its arguments instead of launching) and wraps the step-invariant
# arguments once; every layer then does: rope-append, FP8 q-absorb, one in-place page-table add, one
# compiled-kernel call with pre-wrapped arguments, v_b bmm.
# ---------------------------------------------------------------------------------------------------

_fast = {"plan": None, "layers": {}, "cute": None}


class _StepPlan:
    __slots__ = (
        "n", "cached", "groups", "pad", "rows", "pool_idx", "last_layer", "quant_q", "base_pt",
        "pt_buf", "pt_ct", "out_latent", "out_view", "args", "keep", "compiled",
    )


def _cute():
    c = _fast["cute"]
    if c is None:
        import cutlass.cute as cute
        from cuda.bindings import driver as cuda

        c = _fast["cute"] = (cute, cuda)
    return c


def _layer_entry(mla, attn, attn_metadata):
    """Per-layer constants cached across steps: pool index, layer offset in the pool, the latent / rope
    views of the KV pool and their pre-wrapped CuTe tensors. Revalidated against the pool storage."""
    cute, _ = _cute()
    layer = attn.layer_idx
    kv_pool = attn_metadata.kv_cache_manager.get_buffers(layer)
    key = (kv_pool.data_ptr(), tuple(kv_pool.shape), kv_pool.stride(0))
    ent = _fast["layers"].get(layer)
    if ent is not None and ent["key"] == key:
        return ent
    page_size = attn_metadata.kv_cache_manager.tokens_per_block
    lora = mla.kv_lora_rank
    rope = mla.qk_rope_head_dim
    local_layer = attn.get_local_layer_idx(attn_metadata)
    pool_idx = int(attn_metadata.host_kv_cache_pool_mapping[local_layer, 0])
    packed_block = 1
    for size in kv_pool.shape[1:]:
        packed_block *= size
    block_stride = kv_pool.stride(0)
    layers_in_pool = block_stride // packed_block if packed_block else 1
    layer_in_pool = 0
    if layers_in_pool > 1 and block_stride == layers_in_pool * packed_block:
        layer_in_pool = kv_pool.storage_offset() // packed_block
        kv_pool = kv_pool.as_strided(
            (kv_pool.shape[0] * layers_in_pool, *kv_pool.shape[1:]),
            (packed_block, *kv_pool.stride()[1:]),
            0,
        )
    kv_typed = kv_pool.view(torch.float8_e4m3fn)
    if kv_typed.dim() != 5 or kv_typed.shape[1] != 1 or kv_typed.shape[3] != 1:
        raise RuntimeError(f"unexpected KV cache layout {tuple(kv_typed.shape)}")
    kv_pages = kv_typed[:, 0, :, 0, : lora + rope]
    c_latent = kv_pages[..., :lora].permute(1, 2, 0)
    c_rope = kv_pages[..., lora:].permute(1, 2, 0)
    ent = {
        "key": key,
        "pool_idx": pool_idx,
        "layer_in_pool": layer_in_pool,
        "page_size": page_size,
        "c_latent": c_latent,
        "c_rope": c_rope,
        "c_latent_ct": cute.runtime.from_dlpack(c_latent, assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        ),
        "c_rope_ct": cute.runtime.from_dlpack(c_rope, assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        ),
    }
    _fast["layers"][layer] = ent
    return ent


def _build_plan(mla, attn_metadata, ent, n, cached, groups, pad, rows, layer) -> Optional[_StepPlan]:
    """First absorbed layer of a step: the step-invariant tensors, the real runner call (tactic from the
    autotuner exactly as the op does) and the wrapped arguments."""
    from ..custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner
    from ..autotuner import AutoTuner
    import cutlass

    cute, _ = _cute()
    s0 = GROUP
    heads = mla.num_heads_tp
    lora = mla.kv_lora_rank
    rope = mla.qk_rope_head_dim
    device = ent["c_latent"].device
    attn = mla.mha
    page_size = ent["page_size"]
    num_pages = (cached + n + page_size - 1) // page_size

    plan = _StepPlan()
    plan.n, plan.cached, plan.groups, plan.pad, plan.rows = n, cached, groups, pad, rows
    plan.pool_idx = ent["pool_idx"]
    plan.last_layer = layer
    row = attn_metadata.kv_cache_block_offsets[plan.pool_idx, 0, 0, :num_pages].to(torch.int32)
    plan.base_pt = row.unsqueeze(0).expand(groups, num_pages).contiguous()
    plan.pt_buf = torch.empty_like(plan.base_pt)
    pt_view = plan.pt_buf.transpose(0, 1)
    torch.add(plan.base_pt, ent["layer_in_pool"], out=plan.pt_buf)

    plan.quant_q = torch.zeros(rows, heads, lora + rope, dtype=torch.float8_e4m3fn, device=device)
    cache_seqs = (
        torch.arange(groups, dtype=torch.int32, device=device) * s0 + (cached - pad + s0)
    ).contiguous()
    softmax_scale = float(1.0 / (math.sqrt(mla.qk_nope_head_dim + rope) * mla.mqa.q_scaling))
    bmm1, bmm2 = _bmm_scales(device, softmax_scale)
    max_groups = (MAX_TOKENS + s0 - 1) // s0 + 1
    workspace = _workspace(heads, lora, max_groups, device)

    q4 = plan.quant_q.view(groups, s0, heads, lora + rope)
    q_latent = q4[..., :lora].permute(2, 3, 1, 0)
    q_rope = q4[..., lora:].permute(2, 3, 1, 0)
    plan.out_latent = torch.empty(groups, s0, heads, lora, dtype=torch.bfloat16, device=device)
    o_kernel = plan.out_latent.permute(2, 3, 1, 0)

    runner = CuteDSLNVMlaDecodeBlackwellRunner(
        in_dtype=cutlass.Float8E4M3FN,
        num_heads=heads,
        seq_len_q=s0,
        page_size=page_size,
        max_batch_size=int(max_groups),
        emit_softmax_stats=False,
    )
    inputs = [
        q_latent, q_rope, ent["c_latent"], ent["c_rope"], pt_view, cache_seqs, o_kernel,
        workspace, None, None, bmm1, bmm2,
    ]
    _, best_tactic = AutoTuner.get().choose_one(
        "trtllm::cute_dsl_mla_decode_fp8_blackwell", [runner], runner.get_tuning_config(), inputs
    )
    if not (isinstance(best_tactic, tuple) and len(best_tactic) == 4):
        best_tactic = runner.default_tactic(int(q_latent.shape[-1]))
    compiled, args = runner.forward(inputs, tactic=best_tactic, plan_max_split=0, return_launch=True)
    if len(args) != 15 or args[10] is not None or args[11] is not None or o_kernel.dtype != torch.bfloat16:
        return None
    wrap = cute.runtime.from_dlpack
    w = list(args)
    w[0] = wrap(args[0], assumed_align=16).mark_layout_dynamic(leading_dim=1)
    w[1] = wrap(args[1], assumed_align=16).mark_layout_dynamic(leading_dim=1)
    w[5] = (
        wrap(args[5], assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, stride_order=(3, 2, 0, 1), divisibility=8)
    )
    w[6] = wrap(args[6], assumed_align=16).mark_layout_dynamic(leading_dim=0)
    if args[7] is not None:
        w[7] = wrap(args[7], assumed_align=32).mark_layout_dynamic()
    w[9] = wrap(args[9], assumed_align=16).mark_layout_dynamic()
    w[12] = wrap(args[12]).mark_layout_dynamic()
    w[13] = wrap(args[13]).mark_layout_dynamic()
    plan.pt_ct = wrap(args[4], assumed_align=16).mark_layout_dynamic(leading_dim=0)
    w[4] = plan.pt_ct
    plan.args = w
    plan.keep = (args, cache_seqs, q4, q_latent, q_rope, o_kernel, workspace, bmm1, bmm2, pt_view)
    plan.compiled = compiled
    return plan


def _forward_fast(mla, q, latent_cache, attn_metadata, output) -> bool:
    """The absorbed layer on the pre-planned launch. Returns False (nothing done) when the plan cannot
    be built for this layer, so the caller runs the default absorbed path."""
    from .mla_fp8_q_absorb import mla_fp8_q_absorb_nope

    _, cuda = _cute()
    attn = mla.mha
    n = attn_metadata.num_ctx_tokens
    cached = int(attn_metadata.num_ctx_cached_tokens)
    heads = mla.num_heads_tp
    nope = mla.qk_nope_head_dim
    rope = mla.qk_rope_head_dim
    s0 = GROUP
    groups = (n + s0 - 1) // s0
    pad = groups * s0 - n
    rows = groups * s0
    layer = attn.layer_idx
    ent = _layer_entry(mla, attn, attn_metadata)

    if _fast.get("off"):
        return False
    plan = _fast["plan"]
    if (
        plan is None
        or layer <= plan.last_layer
        or plan.n != n
        or plan.cached != cached
        or plan.pool_idx != ent["pool_idx"]
    ):
        # Not a continuation of the current step: new step, one real runner call.
        plan = _fast["plan"] = _build_plan(mla, attn_metadata, ent, n, cached, groups, pad, rows, layer)
        if plan is None:
            _fast["off"] = True
            logger.warning("TRTLLM_K3_MLA_CTX_ABSORB_FAST: unexpected launch layout, using the default path")
            return False
        attn.mla_rope_append_paged_kv_assign_q(q, latent_cache, attn_metadata)
    else:
        plan.last_layer = layer
        attn.mla_rope_append_paged_kv_assign_q(q, latent_cache, attn_metadata)
        torch.add(plan.base_pt, ent["layer_in_pool"], out=plan.pt_buf)
    mla_fp8_q_absorb_nope(
        q.view(n, heads, nope + rope), _w_kc_bf16(mla), plan.quant_q[pad:], nope, rope
    )
    a = list(plan.args)
    a[2] = ent["c_latent_ct"]
    a[3] = ent["c_rope_ct"]
    a[14] = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    plan.compiled(*a)

    attn_out_latent = plan.out_latent.view(rows, heads, mla.kv_lora_rank)[pad:]
    attn_output = output.view(n, heads, mla.v_head_dim)
    if mla.v_b_proj.dtype == torch.bfloat16:
        mla._bmm_bf16_out(
            attn_out_latent.transpose(0, 1),
            mla.v_b_proj,
            mla.v_b_proj.transpose(1, 2),
            attn_output.transpose(0, 1),
            mla._ugpu_v_b_proj_shards,
        )
    elif mla.v_b_proj.dtype == torch.float8_e4m3fn:
        from .mla import fp8_block_scaling_bmm_out

        fp8_block_scaling_bmm_out(
            attn_out_latent.contiguous(),
            mla.v_b_proj,
            mla.v_b_proj_scale,
            attn_output.transpose(0, 1),
            mla.v_b_proj_dequant,
            mla.use_cute_dsl_blockscaling_bmm,
        )
    else:
        raise NotImplementedError(f"Missing bmm impl for dtype: {mla.v_b_proj.dtype}.")
    return True


def _err(a: torch.Tensor, b: torch.Tensor):
    a = a.float()
    b = b.float()
    d = (a - b).abs()
    return (
        float(d.max()),
        float(d.mean()),
        float(d.max() / b.abs().max().clamp_min(1e-12)),
        float(d.sum() / b.abs().sum().clamp_min(1e-12)),
    )


def checked_forward(mla, q, latent_cache, attn_metadata, output, default_fn):
    """Run absorbed + default (twice) on the same input and log the errors, return the default
    output (the check run must not change served numerics). ``default_fn(q, latent_cache,
    attn_metadata, output)`` is the chunked-prefill path."""
    q_ref1 = q.clone()
    q_ref2 = q.clone()
    out_new = torch.empty_like(output)
    out_ref1 = torch.empty_like(output)
    out_ref2 = torch.empty_like(output)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    ev[0].record()
    forward_absorbed_context(mla, q, latent_cache, attn_metadata, out_new)
    ev[1].record()
    default_fn(q_ref1, latent_cache, attn_metadata, out_ref1)
    ev[2].record()
    default_fn(q_ref2, latent_cache, attn_metadata, out_ref2)
    ev[3].record()
    ev[3].synchronize()
    t_new, t_ref = ev[0].elapsed_time(ev[1]), ev[1].elapsed_time(ev[2])
    msg = (
        f"[mla_ctx_absorb check] layer={mla.layer_idx} N={attn_metadata.num_ctx_tokens} "
        f"cached={int(attn_metadata.num_ctx_cached_tokens)} t_absorb_ms={t_new:.3f} t_default_ms={t_ref:.3f} "
        f"new-vs-ref(maxabs,meanabs,maxrel,sumrel)={_err(out_new, out_ref1)} "
        f"ref-vs-ref={_err(out_ref2, out_ref1)} refmaxabs={float(out_ref1.float().abs().max()):.4g}"
    )
    if mla.o_proj is not None and os.environ.get("TRTLLM_K3_MLA_CTX_ABSORB_CHECK_OPROJ", "1") == "1":
        try:
            h_new = mla.o_proj(out_new.clone())
            h_ref1 = mla.o_proj(out_ref1.clone())
            h_ref2 = mla.o_proj(out_ref2.clone())
            msg += (
                f" | post-o_proj new-vs-ref={_err(h_new, h_ref1)} ref-vs-ref={_err(h_ref2, h_ref1)}"
            )
        except Exception as e:  # diagnostic only
            msg += f" | o_proj check failed: {e!r}"
    logger.info(msg)
    output.copy_(out_ref1)
    return output
