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

"""TRT-LLM-specific RoPE fusion transform.

This module is split out from ``rope.py`` because it depends on the TRT-LLM
attention backend (``custom_ops.attention.trtllm_attention``), which is only
available when the full TensorRT-LLM runtime is installed. In standalone
auto_deploy mode, this module fails to import and is silently skipped by
``library/__init__.py``; ``rope.py`` itself remains importable so all
backend-agnostic RoPE transforms continue to work.
"""

import operator
from typing import Dict, Optional, Tuple, Type

import torch
from pydantic import Field
from torch.fx import GraphModule, Node

from ...custom_ops.attention.trtllm_attention import _TRTLLM_ROPE_INFO_KEY
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import extract_op_args, is_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)
from .rope import _get_nested_attr, _trace_back_index, _trace_to_buffer_source

# torch_rope IR ops that can be fused into trtllm attention.
# This transform runs BEFORE optimize_rope at post_load_fusion, so only the
# backend-agnostic IR ops are present.  torch_rope_with_qk_interleaving is
# excluded because thop.attention does not support the interleaved rotation style.
_FUSIBLE_ROPE_OPS = {
    torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin,
    torch.ops.auto_deploy.torch_rope_with_complex_freqs,
}


class FuseRopeIntoTrtllmAttentionConfig(TransformConfig):
    """Configuration for ``fuse_rope_into_trtllm_attention``."""

    fuse_qkv_passthrough: bool = Field(
        default=True,
        description=(
            "When the pre-RoPE Q/K/V trace back to a single fused QKV GEMM, "
            "rewire all three to that flat tensor so ``trtllm_mha_with_cache`` "
            "can skip the per-layer split → reshape → cat path."
        ),
    )


@TransformRegistry.register("fuse_rope_into_trtllm_attention")
class FuseRopeIntoTrtllmAttention(BaseTransform):
    """Fuse RoPE into trtllm attention by rewiring Q/K and storing rope metadata.

    Runs at ``post_load_fusion`` **before** ``optimize_rope``, matching the
    backend-agnostic ``torch_rope_*`` IR ops directly with real (non-meta)
    weights.  DCE after this transform removes dead rope nodes; ``optimize_rope``
    then handles remaining ``torch_rope_*`` for non-trtllm backends.

    Stores the thop-format cos_sin tensor in ``attn_node.meta``.
    ``TrtllmAttention.prepare_node_for_cache_insertion`` at ``cache_init``
    materializes it as a graph node.

    Disabled by default; enable in model configs that use ``attn_backend: trtllm``.
    """

    config: FuseRopeIntoTrtllmAttentionConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return FuseRopeIntoTrtllmAttentionConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph
        num_fused = 0

        for attn_node in list(graph.nodes):
            if not is_op(attn_node, torch.ops.auto_deploy.torch_attention):
                continue

            if self._try_fuse_one(gm, attn_node, cm, self.config.fuse_qkv_passthrough):
                num_fused += 1

        info = TransformInfo(
            skipped=num_fused == 0,
            num_matches=num_fused,
            is_clean=num_fused == 0,
            has_valid_shapes=num_fused == 0,
        )
        return gm, info

    @staticmethod
    def _try_fuse_one(
        gm: GraphModule,
        attn_node: Node,
        cm: CachedSequenceInterface,
        fuse_qkv_passthrough: bool,
    ) -> bool:
        """Rewire Q/K to pre-RoPE and store rope metadata for cache insertion."""
        # Step 1: Get Q and K input nodes from the attention node
        q_input, k_input = extract_op_args(attn_node, "query", "key")
        if not isinstance(q_input, Node) or not isinstance(k_input, Node):
            return False

        # Step 2: Trace Q/K through operator.getitem to find a rope op
        rope_node_q, q_idx = _trace_to_rope(q_input)
        rope_node_k, k_idx = _trace_to_rope(k_input)
        if rope_node_q is None or rope_node_k is None:
            return False
        if rope_node_q is not rope_node_k:
            return False
        if sorted([q_idx, k_idx]) != [0, 1]:
            return False

        rope_node = rope_node_q
        pre_rope_q = rope_node.args[q_idx]
        pre_rope_k = rope_node.args[k_idx]
        if not isinstance(pre_rope_q, Node) or not isinstance(pre_rope_k, Node):
            return False

        # Step 3: Extract rope metadata — weights are real at post_load_fusion.
        if is_op(rope_node, torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin):
            rope_info = _extract_explicit_rope_info(gm, rope_node, cm)
        elif is_op(rope_node, torch.ops.auto_deploy.torch_rope_with_complex_freqs):
            rope_info = _extract_complex_rope_info(gm, rope_node)
        else:
            return False

        if rope_info is None:
            return False

        # Step 4: Rewire Q/K to pre-RoPE inputs
        args_list = list(attn_node.args)
        args_list[0] = pre_rope_q
        args_list[1] = pre_rope_k
        attn_node.args = tuple(args_list)

        # Step 5: Store metadata for prepare_node_for_cache_insertion
        attn_node.meta[_TRTLLM_ROPE_INFO_KEY] = rope_info

        # Step 6: If pre-RoPE Q/K/V trace back to a flat fused QKV GEMM output,
        # rewire all three to that flat tensor and store head/dim hints so
        # ``trtllm_mha_with_cache`` can skip the split→reshape→cat path.
        if not fuse_qkv_passthrough:
            return True
        bind_v = attn_node.args[2]
        fused_qkv = _try_trace_to_fused_qkv(pre_rope_q, pre_rope_k, bind_v)
        if fused_qkv is not None:
            fused_qkv_node, num_heads, num_kv_heads, head_dim = fused_qkv
            # Capture the KV dtype now, before V is overwritten and while shape
            # propagation on these nodes is still trustworthy.  Subsequent
            # transforms (e.g. fp8 GEMM rewrites) can drop ``meta['val']`` on
            # the fused-QKV node, so we cannot rely on reading it back later.
            kv_dtype_node = bind_v if bind_v.meta.get("val") is not None else pre_rope_k
            kv_val = kv_dtype_node.meta.get("val")
            args_list = list(attn_node.args)
            args_list[0] = fused_qkv_node
            args_list[1] = fused_qkv_node
            args_list[2] = fused_qkv_node
            attn_node.args = tuple(args_list)
            attn_node.meta["_trtllm_fused_qkv"] = True
            attn_node.meta["_trtllm_num_heads"] = num_heads
            attn_node.meta["_trtllm_num_kv_heads"] = num_kv_heads
            attn_node.meta["_trtllm_head_dim"] = head_dim
            if kv_val is not None:
                attn_node.meta["_trtllm_kv_dtype"] = kv_val.dtype
        return True


def _trace_to_rope(node: Node) -> Tuple[Optional[Node], Optional[int]]:
    """Trace through operator.getitem to find a fusible rope op."""
    if not is_op(node, operator.getitem):
        return None, None
    source = node.args[0]
    item_idx = node.args[1]
    if not isinstance(source, Node) or not isinstance(item_idx, int):
        return None, None
    if is_op(source, _FUSIBLE_ROPE_OPS):
        return source, item_idx
    return None, None


def _extract_explicit_rope_info(
    gm: GraphModule, rope_node: Node, cm: CachedSequenceInterface
) -> Optional[Dict]:
    """Extract rope info from torch_rope_with_explicit_cos_sin. cos=arg[2], sin=arg[3]."""
    cos_node = rope_node.args[2]
    sin_node = rope_node.args[3]

    q_fake = rope_node.args[0].meta.get("val", None)
    head_dim = q_fake.shape[-1] if q_fake is not None else 128
    half_dim = head_dim // 2

    # Unwrap aten.index.Tensor(table, [pos_ids]) to get the full table
    cos_traced = _trace_back_index(cos_node)
    sin_traced = _trace_back_index(sin_node)
    cos_table = cos_traced[0] if cos_traced else cos_node
    sin_table = sin_traced[0] if sin_traced else sin_node

    # Try direct buffer extraction (weights are real at post_load_fusion)
    cos_tensor = _get_real_tensor(gm, cos_table)
    sin_tensor = _get_real_tensor(gm, sin_table)
    if cos_tensor is not None and sin_tensor is not None:
        fi_tensor = torch.cat([cos_tensor[:, :half_dim], sin_tensor[:, :half_dim]], dim=-1).to(
            torch.float32
        )
        return _build_rope_info(fi_tensor, is_neox=True, head_dim=head_dim)

    # Fallback: dynamic cos/sin from inv_freq
    inv_freq = _find_inv_freq_tensor(gm, cos_table, half_dim)
    if inv_freq is not None and cm.info.max_seq_len > 0:
        fi_tensor = _compute_cos_sin_from_inv_freq(inv_freq, cm.info.max_seq_len)
        return _build_rope_info(fi_tensor, is_neox=True, head_dim=head_dim)

    return None


def _extract_complex_rope_info(gm: GraphModule, rope_node: Node) -> Optional[Dict]:
    """Extract rope info from torch_rope_with_complex_freqs. freqs_cis=arg[2]."""
    freqs_tensor = _get_real_tensor(gm, rope_node.args[2])
    if freqs_tensor is None:
        return None

    q_fake = rope_node.args[0].meta.get("val", None)
    head_dim = q_fake.shape[-1] if q_fake is not None else 128

    fi_tensor = torch.cat([freqs_tensor.real.float(), freqs_tensor.imag.float()], dim=-1).to(
        torch.float32
    )
    return _build_rope_info(fi_tensor, is_neox=False, head_dim=head_dim)


def _convert_to_thop_cos_sin(fi_cache: torch.Tensor, rotary_embedding_dim: int) -> torch.Tensor:
    """Convert FlashInfer-format cos_sin_cache to thop.attention format.

    FlashInfer: ``[max_pos, head_dim]`` — ``[cos_0..cos_{d/2-1}, sin_0..sin_{d/2-1}]``
    thop: ``[1, max_pos * dim]`` — interleaved ``[cos_0, sin_0, cos_1, sin_1, ...]``
    """
    half = rotary_embedding_dim // 2
    cos_part = fi_cache[:, :half]
    sin_part = fi_cache[:, half:]
    thop_cache = torch.stack([cos_part, sin_part], dim=-1)
    return thop_cache.reshape(1, -1).float()


def _build_rope_info(fi_cache: torch.Tensor, is_neox: bool, head_dim: int) -> Dict:
    """Build the metadata dict stored in attn_node.meta[_TRTLLM_ROPE_INFO_KEY]."""
    return {
        "cos_sin_tensor": _convert_to_thop_cos_sin(fi_cache, fi_cache.shape[-1]),
        "position_embedding_type": 2 if is_neox else 1,
        "rotary_embedding_dim": head_dim,
    }


def _get_real_tensor(gm: GraphModule, node: Node) -> Optional[torch.Tensor]:
    """Linear trace to buffer source, return real (non-meta) tensor or None."""
    if not isinstance(node, Node):
        return None
    source = _trace_to_buffer_source(node)
    if source is not None and source.op == "get_attr":
        try:
            tensor = _get_nested_attr(gm, source.target)
            if not tensor.is_meta:
                return tensor
        except AttributeError:
            pass
    return None


def _compute_cos_sin_from_inv_freq(inv_freq: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    """Compute FlashInfer-format cos_sin_cache from inv_freq buffer."""
    t = torch.arange(max_seq_len, dtype=inv_freq.dtype, device=inv_freq.device)
    freqs = torch.outer(t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.float32)


def _find_inv_freq_tensor(
    gm: GraphModule, cos_table_node: Node, half_head_dim: int
) -> Optional[torch.Tensor]:
    """Find the inv_freq buffer that feeds into the given cos/sin table node.

    Searches the module's named buffers for a 1-D tensor of size ``half_head_dim``
    whose name contains ``inv_freq``, then verifies it is an ancestor of
    ``cos_table_node`` in the graph (bounded backward walk, max 20 hops).
    """
    # Find candidate inv_freq buffers and their get_attr nodes in the graph
    candidates = {}
    for name, buf in gm.named_buffers():
        if (
            "inv_freq" in name
            and not buf.is_meta
            and buf.dim() == 1
            and buf.shape[0] == half_head_dim
        ):
            candidates[name] = buf

    if not candidates:
        return None

    # If only one candidate, return it directly
    if len(candidates) == 1:
        return next(iter(candidates.values()))

    # Multiple candidates: verify which one is an ancestor of cos_table_node
    # by walking backward from cos_table_node through call_function args.
    visited: set = set()
    frontier = [cos_table_node]
    for _ in range(20):
        next_frontier = []
        for node in frontier:
            if id(node) in visited or not isinstance(node, Node):
                continue
            visited.add(id(node))
            if node.op == "get_attr" and node.target in candidates:
                return candidates[node.target]
            if node.op == "call_function":
                for arg in node.args:
                    if isinstance(arg, Node):
                        next_frontier.append(arg)
        if not next_frontier:
            break
        frontier = next_frontier

    # No verified match — return first candidate as best guess
    return next(iter(candidates.values()))


def _unwrap_contiguous(node: Node) -> Node:
    """Skip past any chain of ``contiguous`` calls, in either ``aten``
    overload form or the Python-level ``Tensor.contiguous`` method form."""
    current = node
    while isinstance(current, Node) and current.op == "call_function":
        is_aten_contig = is_op(current, torch.ops.aten.contiguous.default)
        is_method_contig = getattr(current.target, "__name__", "") == "contiguous"
        if not (is_aten_contig or is_method_contig):
            break
        if not isinstance(current.args[0], Node):
            break
        current = current.args[0]
    return current


def _try_trace_to_fused_qkv(
    pre_rope_q: Node, pre_rope_k: Node, bind_v: Node
) -> Optional[Tuple[Node, int, int, int]]:
    """Trace pre-RoPE Q, K and V back to a single flat fused QKV GEMM output.

    Supports two graph shapes produced by GEMM fusion:

    1. ``fused_qkv → split_with_sizes/split_output → getitem → view``
    2. ``fused_qkv → narrow(-1, offset, size) → view``

    On success, the flat fused QKV tensor can be passed directly to
    ``trtllm_mha_with_cache`` (in fused-QKV mode), skipping per-layer
    split + reshape + cat.

    Returns ``(fused_qkv_node, num_heads, num_kv_heads, head_dim)`` or ``None``.
    """

    def _trace_split(node: Node):
        current = _unwrap_contiguous(node)
        if not (
            is_op(current, torch.ops.aten.view.default)
            or is_op(current, torch.ops.aten.reshape.default)
        ):
            return None
        view_input = current.args[0]
        view_shape = current.args[1] if len(current.args) > 1 else None
        if not isinstance(view_input, Node):
            return None
        if not isinstance(view_shape, (list, tuple)) or len(view_shape) < 4:
            return None
        nh, hd = view_shape[2], view_shape[3]
        if not (isinstance(nh, int) and isinstance(hd, int)):
            return None
        if nh < 0 or hd < 0:
            fake = node.meta.get("val", None)
            if fake is None or len(fake.shape) < 4:
                return None
            nh = int(fake.shape[2]) if nh < 0 else nh
            hd = int(fake.shape[3]) if hd < 0 else hd
        if not is_op(view_input, operator.getitem):
            return None
        getitem_source = view_input.args[0]
        getitem_idx = view_input.args[1]
        if not isinstance(getitem_source, Node) or not isinstance(getitem_idx, int):
            return None
        if getitem_source.op != "call_function":
            return None
        fused_node = getitem_source.args[0]
        if not isinstance(fused_node, Node):
            return None
        return fused_node, getitem_idx, nh * hd, nh, hd

    def _trace_narrow(node: Node):
        current = _unwrap_contiguous(node)
        if not (
            is_op(current, torch.ops.aten.view.default)
            or is_op(current, torch.ops.aten.reshape.default)
        ):
            return None
        view_input = current.args[0]
        view_shape = current.args[1] if len(current.args) > 1 else None
        if not isinstance(view_input, Node):
            return None
        if not isinstance(view_shape, (list, tuple)) or len(view_shape) < 4:
            return None
        hd = view_shape[3]
        if not isinstance(hd, int):
            return None
        narrow_ops = {torch.narrow, torch.Tensor.narrow, torch.ops.aten.narrow.default}
        if view_input.op != "call_function" or view_input.target not in narrow_ops:
            return None
        if len(view_input.args) < 4:
            return None
        narrow_parent = view_input.args[0]
        narrow_dim = view_input.args[1]
        narrow_offset = view_input.args[2]
        narrow_length = view_input.args[3]
        if not isinstance(narrow_parent, Node) or narrow_dim != -1:
            return None
        if not isinstance(narrow_offset, int) or not isinstance(narrow_length, int):
            return None
        nh = narrow_length // hd
        if nh * hd != narrow_length:
            return None
        return narrow_parent, narrow_offset, narrow_length, nh, hd

    q_split = _trace_split(pre_rope_q)
    k_split = _trace_split(pre_rope_k)
    v_split = _trace_split(bind_v)
    if q_split is not None and k_split is not None and v_split is not None:
        q_src, q_idx, _, q_nh, q_hd = q_split
        k_src, k_idx, k_dim, k_nh, k_hd = k_split
        v_src, v_idx, v_dim, _, v_hd = v_split
        if (
            q_src is k_src
            and q_src is v_src
            and (q_idx, k_idx, v_idx) == (0, 1, 2)
            and k_dim == v_dim
            and q_hd == k_hd == v_hd
        ):
            return q_src, q_nh, k_nh, q_hd

    q_narrow = _trace_narrow(pre_rope_q)
    k_narrow = _trace_narrow(pre_rope_k)
    v_narrow = _trace_narrow(bind_v)
    if q_narrow is not None and k_narrow is not None and v_narrow is not None:
        q_parent, q_off, q_len, q_nh, q_hd = q_narrow
        k_parent, k_off, k_len, k_nh, k_hd = k_narrow
        v_parent, v_off, v_len, _, v_hd = v_narrow
        if (
            q_parent is k_parent is v_parent
            and q_off == 0
            and k_off == q_len
            and v_off == q_len + k_len
            and k_len == v_len
            and q_hd == k_hd == v_hd
        ):
            return q_parent, q_nh, k_nh, q_hd

    return None
