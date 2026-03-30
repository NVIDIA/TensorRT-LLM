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

"""Transform that fuses RoPE into the TRT-LLM MLA cached attention op.

After ``insert_cached_mla_attention`` replaces ``torch_mla`` with
``trtllm_mla_with_cache``, this transform traces back from the cached op's
``q_pe`` and ``kpe`` arguments through ``operator.getitem`` nodes to find the
upstream RoPE op.  It then:

1. Rewires the MLA op to receive **pre-RoPE** ``q_pe`` and ``kpe``.
2. Constructs a flat ``rotary_cos_sin`` table from the model's cos/sin buffers.
3. Replaces ``trtllm_mla_with_cache`` with ``trtllm_mla_fused_rope_with_cache``,
   which calls ``mla_rope_generation`` in the decode path — fusing cache write,
   RoPE, q_pe copy, and scheduler fill into a single kernel.
"""

import math
import operator
from typing import Optional, Tuple

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

# All known RoPE op targets that can feed into trtllm_mla_with_cache.
# Include both OpOverloadPacket and OpOverload (.default) forms because
# is_op() only expands packets→overloads, not the reverse.
_ROPE_OP_TARGETS = []
for _name in ("torch_rope_with_explicit_cos_sin", "flashinfer_rope"):
    try:
        _packet = getattr(torch.ops.auto_deploy, _name)
        _ROPE_OP_TARGETS.append(_packet)
        _ROPE_OP_TARGETS.append(_packet.default)
    except AttributeError:
        pass


def _trace_rope_node(mla_node: Node) -> Optional[Tuple[Node, Node, Node, Node]]:
    """Trace back from an MLA cached attention node to find the RoPE node.

    Returns (rope_node, q_pe_pre, kpe_pre, cos_sin_info_node) or None if
    the pattern doesn't match.  ``cos_sin_info_node`` is the cos_sin_cache
    arg for flashinfer_rope or the cos arg for torch_rope.
    """
    q_pe_node = mla_node.args[1]
    kpe_node = mla_node.args[3]

    q_is_getitem = is_op(q_pe_node, operator.getitem)
    k_is_getitem = is_op(kpe_node, operator.getitem)
    if not (q_is_getitem and k_is_getitem):
        ad_logger.warning(
            f"_trace_rope_node: q_pe is getitem={q_is_getitem} (op={getattr(q_pe_node, 'op', '?')}, "
            f"target={getattr(q_pe_node, 'target', '?')}), "
            f"kpe is getitem={k_is_getitem} (op={getattr(kpe_node, 'op', '?')}, "
            f"target={getattr(kpe_node, 'target', '?')})"
        )
        return None

    rope_from_q, q_idx = q_pe_node.args
    rope_from_k, k_idx = kpe_node.args

    if rope_from_q is not rope_from_k:
        ad_logger.warning(
            f"_trace_rope_node: q_pe and kpe come from different nodes: "
            f"q_pe_src={rope_from_q.name} ({getattr(rope_from_q, 'target', '?')}), "
            f"kpe_src={rope_from_k.name} ({getattr(rope_from_k, 'target', '?')})"
        )
        return None

    rope_node = rope_from_q

    if not any(is_op(rope_node, target) for target in _ROPE_OP_TARGETS):
        ad_logger.warning(
            f"_trace_rope_node: source node {rope_node.name} is not a known RoPE op "
            f"(op={rope_node.op}, target={getattr(rope_node, 'target', '?')}). "
            f"Known targets: {[str(t) for t in _ROPE_OP_TARGETS]}"
        )
        return None

    # torch_rope returns (q_rot, k_rot) at indices (0, 1)
    # flashinfer_rope returns (q_rot, k_rot) at indices (0, 1)
    if not ({q_idx, k_idx} == {0, 1}):
        ad_logger.warning(
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

    Tries to extract cos/sin buffers from the graph module (for
    ``torch_rope_with_explicit_cos_sin``) or the cos_sin_cache buffer
    (for ``flashinfer_rope``).  Falls back to computing from the HF config.

    Returns a ``[1, max_pos * rope_dim * 2]`` float32 CUDA tensor, or None
    if construction fails.
    """
    if is_op(rope_node, torch.ops.auto_deploy.flashinfer_rope):
        # flashinfer_rope(q, k, position_ids, cos_sin_cache, is_neox)
        # cos_sin_cache: [max_pos, head_dim] with [:, :half] = cos, [:, half:] = sin
        cos_sin_cache_node = rope_node.args[3]
        if cos_sin_cache_node.op == "get_attr":
            buf = _get_buffer(gm, cos_sin_cache_node.target)
            if buf is not None:
                half = buf.shape[-1] // 2
                cos_half = buf[:, :half].float()
                sin_half = buf[:, half:].float()
                # Duplicate to match NeoX-doubled format expected by the kernel:
                # kernel indexes with head_dim_idx from 0 to 2*rope_dim-1,
                # where rope_dim = qk_rope_head_dim which equals head_dim
                # (already NeoX-doubled in the model).
                cos_full = torch.cat([cos_half, cos_half], dim=-1)
                sin_full = torch.cat([sin_half, sin_half], dim=-1)
                result = torch.stack([cos_full, sin_full], dim=-1)
                return result.reshape(1, -1).contiguous()

    if is_op(rope_node, torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin):
        # torch_rope(q, k, cos, sin, unsqueeze_dim)
        # cos/sin may be position-indexed slices; trace back to the full buffer.
        cos_node = rope_node.args[2]
        sin_node = rope_node.args[3]
        cos_buf = _trace_to_buffer(gm, cos_node)
        sin_buf = _trace_to_buffer(gm, sin_node)
        if cos_buf is not None and sin_buf is not None:
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


def _trace_to_buffer(gm: GraphModule, node: Node) -> Optional[torch.Tensor]:
    """Trace a node back through index/getitem ops to a get_attr buffer.

    Handles the common pattern: cos = cos_cached[position_ids].
    """
    if node.op == "get_attr":
        return _get_buffer(gm, node.target)

    # cos_cached[position_ids] appears as aten.index.Tensor
    if is_op(node, torch.ops.aten.index.Tensor):
        source = node.args[0]
        if source.op == "get_attr":
            return _get_buffer(gm, source.target)

    # aten.embedding or aten.select
    if len(node.args) > 0 and isinstance(node.args[0], Node) and node.args[0].op == "get_attr":
        return _get_buffer(gm, node.args[0].target)

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
        ad_logger.warning("Could not get model config for rotary_cos_sin computation")
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
            if mscale_factor != 0:
                mscale = (
                    0.1 * mscale_all_dim * math.log(factor) + 1.0
                    if mscale_all_dim
                    else factor**mscale_factor
                )
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
    return result.reshape(1, -1).contiguous()


def _undo_rope_deinterleave(
    gm: GraphModule,
    factory: ModelFactory,
) -> int:
    """Reverse the _rope_deinterleave_load_hook permutation on weight tensors.

    AD MLA models de-interleave RoPE-related weights from GPTJ (interleaved)
    to NeoX (split-half) layout at load time.  When fusing RoPE into
    mla_rope_generation (which applies GPTJ rotation), we undo that
    permutation so projected data arrives in GPTJ layout at runtime,
    eliminating the need for a per-step runtime permutation.

    Returns the number of weight tensors modified.
    """
    try:
        model_config, _ = factory._get_model_config()
    except Exception:
        ad_logger.warning("Could not get model config; skipping weight re-interleave.")
        return 0

    qk_rope_head_dim = getattr(model_config, "qk_rope_head_dim", None)
    qk_nope_head_dim = getattr(model_config, "qk_nope_head_dim", None)
    kv_lora_rank = getattr(model_config, "kv_lora_rank", None)
    num_heads = getattr(model_config, "num_attention_heads", None)
    if any(v is None for v in (qk_rope_head_dim, qk_nope_head_dim, kv_lora_rank, num_heads)):
        ad_logger.warning(
            "Missing MLA config attrs for weight re-interleave; skipping. "
            f"qk_rope_head_dim={qk_rope_head_dim}, qk_nope_head_dim={qk_nope_head_dim}, "
            f"kv_lora_rank={kv_lora_rank}, num_heads={num_heads}"
        )
        return 0

    d = qk_rope_head_dim
    # The load hook applied: perm = [0, 2, 4, ..., 62, 1, 3, 5, ..., 63]
    # Inverse (argsort) reverses it:  inv = [0, 32, 1, 33, 2, 34, ..., 31, 63]
    perm = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])
    inv_perm = torch.argsort(perm)
    qk_head_dim = qk_nope_head_dim + d

    count = 0
    for name, param in gm.named_parameters():
        if name.endswith("q_b_proj.weight"):
            w = param.data.view(num_heads, qk_head_dim, -1)
            w_nope = w[:, :qk_nope_head_dim, :]
            w_rope = w[:, qk_nope_head_dim:, :]
            w_rope = w_rope[:, inv_perm, :]
            param.data = torch.cat([w_nope, w_rope], dim=1).reshape_as(param.data)
            count += 1
        elif name.endswith("kv_a_proj_with_mqa.weight"):
            w = param.data
            w_kv = w[:kv_lora_rank, :]
            w_pe = w[kv_lora_rank:, :]
            w_pe = w_pe[inv_perm, :]
            param.data = torch.cat([w_kv, w_pe], dim=0)
            count += 1
        elif name.endswith("kv_a_proj_with_mqa.bias"):
            b = param.data
            b_kv = b[:kv_lora_rank]
            b_pe = b[kv_lora_rank:]
            b_pe = b_pe[inv_perm]
            param.data = torch.cat([b_kv, b_pe])
            count += 1

    return count


@TransformRegistry.register("fuse_rope_into_trtllm_mla")
class FuseRopeIntoTrtllmMLA(BaseTransform):
    """Fuse RoPE into TRT-LLM MLA cached attention for decode performance.

    Replaces ``trtllm_mla_with_cache`` with ``trtllm_mla_fused_rope_with_cache``
    which receives pre-RoPE q_pe/kpe and a rotary_cos_sin table.  The fused op
    calls ``mla_rope_generation`` in the decode path, combining cache write,
    RoPE application, q_pe copy, and scheduler fill into a single kernel.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph
        target_op = torch.ops.auto_deploy.trtllm_mla_with_cache.default
        fused_op = torch.ops.auto_deploy.trtllm_mla_fused_rope_with_cache.default

        mla_nodes = [n for n in graph.nodes if is_op(n, target_op)]
        if not mla_nodes:
            ad_logger.info("No trtllm_mla_with_cache nodes found; skipping.")
            return gm, TransformInfo(skipped=True, detail="no MLA nodes")

        # Try to trace the RoPE pattern from the first MLA node.
        trace_result = _trace_rope_node(mla_nodes[0])
        if trace_result is None:
            ad_logger.warning(
                "Could not trace RoPE node from trtllm_mla_with_cache; skipping fusion."
            )
            return gm, TransformInfo(skipped=True, detail="no rope pattern")

        rope_node, _, _, _ = trace_result

        # Build the rotary_cos_sin tensor.
        rotary_cos_sin = _build_rotary_cos_sin_from_buffers(gm, rope_node, factory)
        if rotary_cos_sin is None:
            ad_logger.warning("Could not construct rotary_cos_sin; skipping fusion.")
            return gm, TransformInfo(skipped=True, detail="no rotary_cos_sin")

        # Keep rotary_cos_sin on its current device (CUDA after weight_load).
        # It must NOT be moved to CPU because the op runs inside CUDA graph
        # capture where CPU→CUDA transfers are forbidden.
        # If meta, leave as-is (shape-only tracing).

        # Register the rotary_cos_sin as a buffer on the graph module and
        # create a get_attr node for it.
        buf_name = "_ad_rotary_cos_sin"
        gm.register_buffer(buf_name, rotary_cos_sin)

        replaced = 0
        for mla_node in mla_nodes:
            result = _trace_rope_node(mla_node)
            if result is None:
                ad_logger.warning(f"Skipping MLA node {mla_node.name}: no rope pattern")
                continue

            rope_node_i, q_pe_pre, kpe_pre, _ = result

            # Create get_attr node for rotary_cos_sin right before the MLA node.
            with graph.inserting_before(mla_node):
                cos_sin_node = graph.get_attr(buf_name)
                # Copy metadata from the registered buffer.
                cos_sin_node.meta["val"] = rotary_cos_sin.clone()

            # Build new args: same as trtllm_mla_with_cache but with:
            #   - args[1] = pre-RoPE q_pe (was post-RoPE)
            #   - args[3] = pre-RoPE kpe (was post-RoPE)
            #   - args[5] inserted = rotary_cos_sin (new)
            old_args = list(mla_node.args)
            new_args = (
                old_args[0],  # q_nope
                q_pe_pre,  # q_pe (pre-RoPE)
                old_args[2],  # compressed_kv
                kpe_pre,  # kpe (pre-RoPE)
                old_args[4],  # kv_b_proj_weight
                cos_sin_node,  # rotary_cos_sin (NEW)
                *old_args[5:],  # batch_info_host ... kv_lora_rank
            )

            with graph.inserting_after(mla_node):
                new_node = graph.call_function(fused_op, args=tuple(new_args))
                new_node.meta = dict(mla_node.meta)

            mla_node.replace_all_uses_with(new_node)
            graph.erase_node(mla_node)
            replaced += 1

        graph.eliminate_dead_code()
        gm.recompile()

        # If the model de-interleaved RoPE weights to NeoX layout (detected
        # via is_neox=True on the original RoPE op), reverse that permutation
        # so projected data arrives in GPTJ layout — matching what
        # mla_rope_generation expects — without any runtime permutation cost.
        is_neox = True
        if is_op(rope_node, torch.ops.auto_deploy.flashinfer_rope):
            is_neox = rope_node.args[4] if len(rope_node.args) > 4 else True
        if is_neox:
            n_fixed = _undo_rope_deinterleave(gm, factory)
            ad_logger.info(
                f"Reversed RoPE weight de-interleave on {n_fixed} tensors "
                "(NeoX→GPTJ) for fused decode kernel."
            )

        ad_logger.info(f"Fused RoPE into {replaced} MLA attention node(s).")
        return gm, TransformInfo(detail=f"fused {replaced} nodes")
