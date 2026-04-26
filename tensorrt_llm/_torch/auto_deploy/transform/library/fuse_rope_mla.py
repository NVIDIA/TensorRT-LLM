# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Transform that fuses RoPE into TRT-LLM MLA attention.

Runs at ``post_load_fusion`` **before** ``optimize_rope``, operating on the
pre-cache ``torch_mla`` source ops.  For each ``torch_mla`` node whose
``q_pe`` and ``kpe`` inputs come from a ``torch_rope_with_explicit_cos_sin``
op, this transform:

1. Rewires the MLA op to receive **pre-RoPE** ``q_pe`` and ``kpe``.
2. Constructs a flat ``rotary_cos_sin`` table from the model's cos/sin buffers.
3. Stashes the table in ``node.meta`` for later materialization at ``cache_init``
   by ``TrtllmMLAAttention.prepare_node_for_cache_insertion``.
4. Reverses the NeoX weight de-interleave so projected data arrives in GPTJ
   layout at runtime (matching what ``mla_rope_generation`` expects).
"""

import math
import operator
from typing import Optional, Tuple

import torch
from torch.fx import GraphModule, Node

from ...custom_ops.mla.trtllm_mla import _TRTLLM_MLA_ROPE_INFO_KEY
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

# At post_load_fusion, only the backend-agnostic torch_rope_* IR ops are
# present (optimize_rope has not yet replaced them with flashinfer_rope).
_ROPE_OP_TARGETS = []
for _name in ("torch_rope_with_explicit_cos_sin",):
    try:
        _packet = getattr(torch.ops.auto_deploy, _name)
        _ROPE_OP_TARGETS.append(_packet)
        _ROPE_OP_TARGETS.append(_packet.default)
    except AttributeError:
        pass


def _trace_rope_node(mla_node: Node) -> Optional[Tuple[Node, Node, Node, Node]]:
    """Trace back from a torch_mla node to find the RoPE node.

    Returns (rope_node, q_pe_pre, kpe_pre, rope_node) or None if
    the pattern doesn't match.
    """
    q_pe_node = mla_node.args[1]
    kpe_node = mla_node.args[3]

    q_is_getitem = is_op(q_pe_node, operator.getitem)
    k_is_getitem = is_op(kpe_node, operator.getitem)
    if not (q_is_getitem and k_is_getitem):
        ad_logger.debug(
            f"_trace_rope_node: q_pe is getitem={q_is_getitem} (op={getattr(q_pe_node, 'op', '?')}, "
            f"target={getattr(q_pe_node, 'target', '?')}), "
            f"kpe is getitem={k_is_getitem} (op={getattr(kpe_node, 'op', '?')}, "
            f"target={getattr(kpe_node, 'target', '?')})"
        )
        return None

    rope_from_q, q_idx = q_pe_node.args
    rope_from_k, k_idx = kpe_node.args

    if rope_from_q is not rope_from_k:
        ad_logger.debug(
            f"_trace_rope_node: q_pe and kpe come from different nodes: "
            f"q_pe_src={rope_from_q.name} ({getattr(rope_from_q, 'target', '?')}), "
            f"kpe_src={rope_from_k.name} ({getattr(rope_from_k, 'target', '?')})"
        )
        return None

    rope_node = rope_from_q

    if not any(is_op(rope_node, target) for target in _ROPE_OP_TARGETS):
        ad_logger.debug(
            f"_trace_rope_node: source node {rope_node.name} is not a known RoPE op "
            f"(op={rope_node.op}, target={getattr(rope_node, 'target', '?')}). "
            f"Known targets: {[str(t) for t in _ROPE_OP_TARGETS]}"
        )
        return None

    # torch_rope returns (q_rot, k_rot) at indices (0, 1)
    if not ({q_idx, k_idx} == {0, 1}):
        ad_logger.debug(
            f"_trace_rope_node: unexpected getitem indices q_idx={q_idx}, k_idx={k_idx}"
        )
        return None

    q_pe_pre = rope_node.args[0]
    kpe_pre = rope_node.args[1]

    return rope_node, q_pe_pre, kpe_pre, rope_node


def _build_rotary_cos_sin_from_buffers(
    gm: GraphModule,
    rope_node: Node,
    factory: ModelFactory,
) -> Optional[torch.Tensor]:
    """Construct a flat rotary_cos_sin tensor for mla_rope_generation.

    Handles ``torch_rope_with_explicit_cos_sin`` by tracing cos/sin args
    back to buffers.  Falls back to computing from the HF config.

    Returns a ``[1, max_pos * rope_dim * 2]`` float32 CUDA tensor, or None
    if construction fails.
    """
    if is_op(rope_node, torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin):
        # torch_rope(q, k, cos, sin, unsqueeze_dim)
        cos_node = rope_node.args[2]
        sin_node = rope_node.args[3]
        cos_buf = _trace_to_buffer(gm, cos_node)
        sin_buf = _trace_to_buffer(gm, sin_node)
        if cos_buf is not None and sin_buf is not None:
            # cos_buf/sin_buf are already in NeoX-doubled format
            # [max_pos, head_dim] where head_dim = qk_rope_head_dim.
            # Stack interleaved [cos_0, sin_0, cos_1, sin_1, ...] as expected
            # by mla_rope_generation.
            result = torch.stack([cos_buf.float(), sin_buf.float()], dim=-1)
            return result.reshape(1, -1).contiguous()

    # Fallback: compute from model config
    return _compute_rotary_cos_sin_from_config(factory)


def _get_buffer(gm: GraphModule, target: str) -> Optional[torch.Tensor]:
    """Retrieve a buffer/parameter from the graph module by attribute path."""
    parts = target.split(".")
    obj = gm
    for part in parts:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    if isinstance(obj, torch.Tensor):
        return obj
    return None


def _trace_to_buffer_source(node: Node) -> Optional[Node]:
    """Trace back through unary call_function nodes to find the source get_attr."""
    visited = set()
    current = node
    while id(current) not in visited:
        visited.add(id(current))
        if current.op == "get_attr":
            return current
        if (
            current.op == "call_function"
            and len(current.args) > 0
            and isinstance(current.args[0], Node)
        ):
            current = current.args[0]
        else:
            return None
    return None


def _trace_to_buffer(gm: GraphModule, node: Node) -> Optional[torch.Tensor]:
    """Trace a node back through index/getitem/unary ops to a get_attr buffer.

    Handles common patterns:
    - Direct get_attr (buffer reference)
    - cos_cached[position_ids] (aten.index.Tensor)
    - Chains of unary ops (dtype casts, unsqueeze, etc.) before the buffer
    """
    # Direct buffer reference
    source = _trace_to_buffer_source(node)
    if source is not None:
        buf = _get_buffer(gm, source.target)
        if buf is not None and not buf.is_meta:
            return buf

    # cos_cached[position_ids] appears as aten.index.Tensor — trace the cache source
    if is_op(node, torch.ops.aten.index.Tensor):
        cache_source = _trace_to_buffer_source(node.args[0])
        if cache_source is not None:
            buf = _get_buffer(gm, cache_source.target)
            if buf is not None and not buf.is_meta:
                return buf

    # Walk through the first arg if it's a call_function producing aten.index.Tensor
    if (
        node.op == "call_function"
        and len(node.args) > 0
        and isinstance(node.args[0], Node)
        and is_op(node.args[0], torch.ops.aten.index.Tensor)
    ):
        idx_node = node.args[0]
        cache_source = _trace_to_buffer_source(idx_node.args[0])
        if cache_source is not None:
            buf = _get_buffer(gm, cache_source.target)
            if buf is not None and not buf.is_meta:
                return buf

    return None


def _compute_rotary_cos_sin_from_config(
    factory: ModelFactory,
) -> Optional[torch.Tensor]:
    """Compute rotary_cos_sin from the HuggingFace model config.

    This is the fallback when buffer tracing fails.
    """
    try:
        model_config, _ = factory._get_model_config()
    except Exception:
        ad_logger.debug("Could not get model config for rotary_cos_sin computation")
        return None

    rope_theta = getattr(model_config, "rope_theta", 10000.0)
    qk_rope_head_dim = getattr(model_config, "qk_rope_head_dim", None)
    if qk_rope_head_dim is None:
        return None

    max_position = getattr(model_config, "max_position_embeddings", 8192)
    half_dim = qk_rope_head_dim // 2

    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))

    # Handle YaRN scaling if present
    rope_scaling = getattr(model_config, "rope_scaling", None)
    mscale = 1.0
    if rope_scaling is not None:
        scaling_type = rope_scaling.get("type", "")
        if scaling_type == "yarn":
            factor = rope_scaling.get("factor", 1.0)
            mscale_factor = rope_scaling.get("mscale", 1.0)
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 0.0)

            def _yarn_get_mscale(scale_val, m):
                return 1.0 if scale_val <= 1.0 else 0.1 * m * math.log(scale_val) + 1.0

            # Match the standard TRT-LLM backend formula:
            # _mscale = yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)
            # For DeepSeek (mscale==mscale_all_dim), this cancels to 1.0.
            # The model's softmax_scale already includes mscale^2, so the
            # cos/sin table must NOT duplicate it.
            numerator = _yarn_get_mscale(factor, mscale_factor)
            denominator = _yarn_get_mscale(factor, mscale_all_dim) if mscale_all_dim else 1.0
            mscale = numerator / denominator if denominator != 0 else 1.0
            original_max = rope_scaling.get("original_max_position_embeddings", max_position)
            beta_fast = rope_scaling.get("beta_fast", 32)
            beta_slow = rope_scaling.get("beta_slow", 1)

            def _yarn_find_correction_dim(num_rotations, dim, base, max_pos):
                return (
                    dim * math.log(max_pos / (num_rotations * 2 * math.pi)) / (2 * math.log(base))
                )

            low = math.floor(
                _yarn_find_correction_dim(beta_fast, half_dim * 2, rope_theta, original_max)
            )
            high = math.ceil(
                _yarn_find_correction_dim(beta_slow, half_dim * 2, rope_theta, original_max)
            )
            low = max(low, 0)
            high = min(high, half_dim - 1)

            if low < high:
                smooth = (torch.arange(half_dim, dtype=torch.float32) - low) / (high - low)
                smooth = smooth.clamp(0, 1)
                inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth
            else:
                inv_freq = inv_freq / factor

    t = torch.arange(max_position, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos_vals = emb.cos() * mscale
    sin_vals = emb.sin() * mscale
    result = torch.stack([cos_vals, sin_vals], dim=-1)
    return result.reshape(1, -1).contiguous().cuda()


def _undo_rope_deinterleave(
    gm: GraphModule,
    factory: ModelFactory,
    shared_config: SharedConfig,
) -> int:
    """Reverse the _rope_deinterleave_load_hook permutation on weight tensors.

    AD MLA models de-interleave RoPE-related weights from GPTJ (interleaved)
    to NeoX (split-half) layout at load time.  When fusing RoPE into
    mla_rope_generation (which applies GPTJ rotation), we undo that
    permutation so projected data arrives in GPTJ layout at runtime,
    eliminating the need for a per-step runtime permutation.

    Handles tensor-parallel (TP) column sharding on dim-0.  With TP,
    ``q_b_proj`` is split by heads and ``kv_a_proj_with_mqa`` is evenly
    split so its PE rows may only reside on a subset of ranks.

    Returns the number of weight tensors modified.
    """
    config_error_msg = "Could not get model config; skipping weight re-interleave."
    try:
        model_config, _ = factory._get_model_config()
    except (AttributeError, AssertionError, KeyError, OSError, ValueError) as e:
        raise RuntimeError(config_error_msg) from e

    qk_rope_head_dim = getattr(model_config, "qk_rope_head_dim", None)
    qk_nope_head_dim = getattr(model_config, "qk_nope_head_dim", None)
    kv_lora_rank = getattr(model_config, "kv_lora_rank", None)
    num_heads = getattr(model_config, "num_attention_heads", None)
    if any(v is None for v in (qk_rope_head_dim, qk_nope_head_dim, kv_lora_rank, num_heads)):
        missing_attrs_msg = (
            "Missing MLA config attrs for weight re-interleave; skipping. "
            f"qk_rope_head_dim={qk_rope_head_dim}, qk_nope_head_dim={qk_nope_head_dim}, "
            f"kv_lora_rank={kv_lora_rank}, num_heads={num_heads}"
        )
        raise RuntimeError(missing_attrs_msg)

    d = qk_rope_head_dim
    # The load hook applied: perm = [0, 2, 4, ..., 62, 1, 3, 5, ..., 63]
    # Inverse (argsort) reverses it:  inv = [0, 32, 1, 33, 2, 34, ..., 31, 63]
    perm = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])
    inv_perm = torch.argsort(perm)
    qk_head_dim = qk_nope_head_dim + d
    tp_rank = shared_config.local_rank
    tp_size = shared_config.world_size

    count = 0
    for name, param in gm.named_parameters():
        if name.endswith("q_b_proj.weight"):
            # Derive local head count from actual weight shape; TP column-shards
            # dim-0 so the local shape is [num_heads_local * qk_head_dim, ...].
            num_heads_local = param.data.shape[0] // qk_head_dim
            w = param.data.view(num_heads_local, qk_head_dim, -1)
            w_nope = w[:, :qk_nope_head_dim, :]
            w_rope = w[:, qk_nope_head_dim:, :]
            w_rope = w_rope[:, inv_perm, :]
            param.data = torch.cat([w_nope, w_rope], dim=1).reshape_as(param.data)
            count += 1
        elif name.endswith("kv_a_proj_with_mqa.weight"):
            w = param.data
            full_dim = kv_lora_rank + d
            local_dim = w.shape[0]
            if local_dim == full_dim:
                # Unsharded: apply permutation directly
                w_kv = w[:kv_lora_rank, :]
                w_pe = w[kv_lora_rank:, :]
                w_pe = w_pe[inv_perm, :]
                param.data = torch.cat([w_kv, w_pe], dim=0)
                count += 1
            elif tp_size > 1:
                # TP column-sharded: compute this rank's global row offset and
                # determine how many local rows are KV vs PE.
                base = full_dim // tp_size
                rem = full_dim % tp_size
                global_start = tp_rank * base + min(tp_rank, rem)
                local_kv = min(local_dim, max(0, kv_lora_rank - global_start))
                local_pe = local_dim - local_kv
                if local_pe == d:
                    # All PE rows on this rank: apply full inverse permutation
                    w_kv_part = w[:local_kv, :]
                    w_pe_part = w[local_kv:, :]
                    w_pe_part = w_pe_part[inv_perm, :]
                    param.data = torch.cat([w_kv_part, w_pe_part], dim=0)
                    count += 1
                elif local_pe > 0:
                    ad_logger.warning(
                        f"Partial PE rows ({local_pe}/{d}) on TP rank {tp_rank} "
                        f"for {name}; skipping weight re-interleave for this tensor."
                    )
                # else: no PE rows on this rank, nothing to permute
        elif name.endswith("kv_a_proj_with_mqa.bias"):
            b = param.data
            full_dim = kv_lora_rank + d
            local_dim = b.shape[0]
            if local_dim == full_dim:
                b_kv = b[:kv_lora_rank]
                b_pe = b[kv_lora_rank:]
                b_pe = b_pe[inv_perm]
                param.data = torch.cat([b_kv, b_pe])
                count += 1
            elif tp_size > 1:
                base = full_dim // tp_size
                rem = full_dim % tp_size
                global_start = tp_rank * base + min(tp_rank, rem)
                local_kv = min(local_dim, max(0, kv_lora_rank - global_start))
                local_pe = local_dim - local_kv
                if local_pe == d:
                    b_kv_part = b[:local_kv]
                    b_pe_part = b[local_kv:]
                    b_pe_part = b_pe_part[inv_perm]
                    param.data = torch.cat([b_kv_part, b_pe_part])
                    count += 1
                elif local_pe > 0:
                    ad_logger.warning(
                        f"Partial PE bias ({local_pe}/{d}) on TP rank {tp_rank} "
                        f"for {name}; skipping weight re-interleave for this tensor."
                    )

    return count


@TransformRegistry.register("fuse_rope_into_trtllm_mla")
class FuseRopeIntoTrtllmMLA(BaseTransform):
    """Fuse RoPE into TRT-LLM MLA attention for decode performance.

    Runs at ``post_load_fusion`` before ``optimize_rope``, matching the
    backend-agnostic ``torch_rope_*`` IR ops on ``torch_mla`` source nodes.
    Rewires q_pe/kpe to pre-RoPE inputs and stashes the rotary_cos_sin tensor
    in ``node.meta`` for later materialization at ``cache_init`` by
    ``TrtllmMLAAttention.prepare_node_for_cache_insertion``.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph
        target_op = torch.ops.auto_deploy.torch_mla

        mla_nodes = [n for n in graph.nodes if is_op(n, target_op)]
        if not mla_nodes:
            ad_logger.info("No torch_mla nodes found; skipping.")
            return gm, TransformInfo(skipped=True, detail="no MLA nodes")

        # Try to trace the RoPE pattern from the first MLA node.
        trace_result = _trace_rope_node(mla_nodes[0])
        if trace_result is None:
            ad_logger.debug("Could not trace RoPE node from torch_mla; skipping fusion.")
            return gm, TransformInfo(skipped=True, detail="no rope pattern")

        rope_node, _, _, _ = trace_result

        # Build the rotary_cos_sin tensor from the model's RoPE buffers.
        rotary_cos_sin = _build_rotary_cos_sin_from_buffers(gm, rope_node, factory)
        if rotary_cos_sin is None:
            ad_logger.debug("Could not construct rotary_cos_sin; skipping fusion.")
            return gm, TransformInfo(skipped=True, detail="no rotary_cos_sin")

        replaced = 0
        rewired_mla_nodes = []
        for mla_node in mla_nodes:
            result = _trace_rope_node(mla_node)
            if result is None:
                ad_logger.debug(f"Skipping MLA node {mla_node.name}: no rope pattern")
                continue

            rope_node_i, q_pe_pre, kpe_pre, _ = result

            # Rewire torch_mla to receive pre-RoPE q_pe and kpe.
            # torch_mla(q_nope, q_pe, compressed_kv, kpe, kv_b_proj_weight, ...)
            #   args[1] = q_pe, args[3] = kpe
            old_args = list(mla_node.args)
            old_args[1] = q_pe_pre
            old_args[3] = kpe_pre
            mla_node.args = tuple(old_args)

            rewired_mla_nodes.append(mla_node)
            replaced += 1

        # Stash rope metadata on rewired MLA nodes for cache_init.
        for mla_node in rewired_mla_nodes:
            mla_node.meta[_TRTLLM_MLA_ROPE_INFO_KEY] = {
                "cos_sin_tensor": rotary_cos_sin,
                "is_neox": True,
            }

        graph.eliminate_dead_code()
        gm.recompile()

        # Reverse the NeoX weight de-interleave so projected data arrives in
        # GPTJ layout — matching what mla_rope_generation expects.
        # The is_neox flag is always True when matching torch_rope_with_explicit_cos_sin
        # (which uses the NeoX/split-half rotation style).
        n_fixed = _undo_rope_deinterleave(gm, factory, shared_config)
        ad_logger.info(
            f"Reversed RoPE weight de-interleave on {n_fixed} tensors "
            "(NeoX→GPTJ) for fused decode kernel."
        )

        ad_logger.info(f"Fused RoPE into {replaced} MLA attention node(s).")
        return gm, TransformInfo(
            skipped=False, detail=f"fused {replaced} nodes", num_matches=replaced
        )
