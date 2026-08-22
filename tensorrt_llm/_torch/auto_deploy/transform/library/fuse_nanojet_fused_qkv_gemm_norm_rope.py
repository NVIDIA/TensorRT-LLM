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

"""Fuse three FP8 projections, the Q/K RMSNorms and RoPE into one nanojet kernel."""

from collections import Counter
from typing import List, Optional, Tuple, Type

import torch
from torch.fx import GraphModule, Node

from ....nanojet_utils import ensure_tune_configs, nanojet_supports
from ...custom_ops.linear.nanojet_fused_qkv_gemm_norm_rope import register
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.nanojet_graph import (
    Fp8Projection,
    get_attr_tensor,
    match_fp8_projection,
    set_val_meta,
)
from ...utils.node_utils import (
    extract_op_args,
    extract_output_tuple,
    is_op,
    unwrap_input_through_passthrough,
)
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)

_HEAD_MAJOR_UNSQUEEZE_DIM = 2


def _match_norm_over_projection(gm: GraphModule, node: Node):
    """Resolve ``node`` to RMSNorm-over-FP8-projection, or a reason string on rejection."""
    source, _ = unwrap_input_through_passthrough(node)
    if source is None or not is_op(source, torch.ops.auto_deploy.torch_rmsnorm):
        return f"not-rmsnorm({source.target if isinstance(source, Node) else source})"
    eps = extract_op_args(source, "eps")[0]
    weight = extract_op_args(source, "weight")[0]
    if not isinstance(eps, float) or not isinstance(weight, Node):
        return "rmsnorm-args"
    if len(source.users) != 1:
        return f"norm-users={len(source.users)}"
    projection = match_fp8_projection(gm, source.args[0])
    if projection is None:
        return "not-fp8-projection"
    return projection, weight, eps


def _match_rope_table(node: Node) -> Optional[Tuple[Node, Node]]:
    """Resolve a per-token cos/sin to the (table, position index) it was gathered from."""
    source, _ = unwrap_input_through_passthrough(node)
    if source is None or not is_op(source, torch.ops.aten.index.Tensor):
        return None
    if len(source.args) < 2 or not isinstance(source.args[0], Node):
        return None
    indices = source.args[1]
    if not isinstance(indices, (list, tuple)) or len(indices) != 1:
        return None
    return source.args[0], indices[0]


class FuseNanojetFusedQKVGemmNormRopeConfig(TransformConfig):
    """Configuration for the nanojet FP8 QKV + norm + RoPE fusion."""


@TransformRegistry.register("fuse_nanojet_fused_qkv_gemm_norm_rope")
class FuseNanojetFusedQKVGemmNormRope(BaseTransform):
    """Collapse three FP8 projections, two RMSNorms and RoPE into one nanojet node."""

    config: FuseNanojetFusedQKVGemmNormRopeConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return FuseNanojetFusedQKVGemmNormRopeConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        if not self.config.enabled or not register():
            return gm, TransformInfo(skipped=True, num_matches=0)

        ensure_tune_configs(factory)

        # tp_mode says colwise even unsharded, so world size is what decides.
        if shared_config.world_size > 1:
            ad_logger.info("fuse_nanojet_fused_qkv_gemm_norm_rope: skipped, not supported under sharding")
            return gm, TransformInfo(skipped=True, num_matches=0)

        graph = gm.graph
        num_matches = 0
        cache_nodes: dict = {}
        # Only original nodes are looked up, so one snapshot stays valid.
        order = {node: index for index, node in enumerate(graph.nodes)}
        rejected: Counter = Counter()

        for node in list(graph.nodes):
            if not is_op(node, torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin):
                continue
            match = self._try_fuse(gm, node)
            if isinstance(match, str):
                rejected[match] += 1
                continue
            num_matches += self._rewrite(gm, node, match, cache_nodes, order)

        if num_matches:
            graph.eliminate_dead_code()
            gm.recompile()
            ad_logger.info(f"fuse_nanojet_fused_qkv_gemm_norm_rope: {num_matches} FP8 QKV+norm+RoPE fused")
        else:
            found = Counter(
                str(node.target).rsplit(".", 1)[0]
                for node in graph.nodes
                if node.op == "call_function"
                and ("rope" in str(node.target) or "rotary" in str(node.target))
            )

            def summarize(counter):
                return ", ".join(f"{name} x{count}" for name, count in counter.items()) or "none"

            inventory = Counter(
                str(node.target).rsplit(".", 1)[0]
                for node in graph.nodes
                if node.op == "call_function" and str(node.target).startswith("auto_deploy.")
            )
            ad_logger.warning(
                f"fuse_nanojet_fused_qkv_gemm_norm_rope matched nothing; rope ops present: {summarize(found)}"
                f"; rejected by: {summarize(rejected)}"
                f"; auto_deploy ops in graph: {summarize(inventory)}"
            )

        return gm, TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )

    @staticmethod
    def _try_fuse(gm: GraphModule, rope_node: Node):
        """Validate the six-node subgraph, returning its parts or a reason string on rejection.

        Returning why it failed rather than a bare ``None`` is what makes a silent
        zero-match run diagnosable: the caller tallies the reasons into its warning.
        """
        if extract_op_args(rope_node, "unsqueeze_dim")[0] != _HEAD_MAJOR_UNSQUEEZE_DIM:
            return "rope-layout"
        query, key = extract_op_args(rope_node, "q")[0], extract_op_args(rope_node, "k")[0]
        cos, sin = extract_op_args(rope_node, "cos")[0], extract_op_args(rope_node, "sin")[0]
        if not all(isinstance(arg, Node) for arg in (query, key, cos, sin)):
            return "rope-args-not-nodes"

        # Head counts cannot tell Q from K when num_kv_heads == num_heads.
        query_match = _match_norm_over_projection(gm, query)
        key_match = _match_norm_over_projection(gm, key)
        if isinstance(query_match, str) or isinstance(key_match, str):
            reason = query_match if isinstance(query_match, str) else key_match
            return f"norm-over-projection[{reason}]"
        if query_match[2] != key_match[2]:
            return "eps-mismatch"

        query_projection, query_norm_weight, eps = query_match
        key_projection, key_norm_weight, _ = key_match
        if key_projection.node.args[0] is not query_projection.node.args[0]:
            return "activation-mismatch"

        value_projection = FuseNanojetFusedQKVGemmNormRope._find_value_projection(
            gm, query_projection, key_projection
        )
        if value_projection is None:
            return "no-value-projection"

        # n_q/n_kv only describes the stacked weight if K and V match.
        if key_projection.weight.shape[0] != value_projection.weight.shape[0]:
            return "kv-row-mismatch"
        if query_projection.weight.shape[1] != key_projection.weight.shape[1]:
            return "hidden-size-mismatch"

        scales = {p.input_scale for p in (query_projection, key_projection, value_projection)}
        if len(scales) != 1:
            return "input-scale-mismatch"

        cos_table = _match_rope_table(cos)
        sin_table = _match_rope_table(sin)
        if cos_table is None or sin_table is None or cos_table[1] is not sin_table[1]:
            return "rope-table"

        norm_value = query_norm_weight.meta.get("val")
        if norm_value is None:
            norm_value = get_attr_tensor(gm, query_norm_weight)
        if norm_value is None:
            return "norm-weight"
        head_dim = int(norm_value.shape[-1])

        # The rewrite keeps each table's first half, which needs a head_dim-wide table.
        cos_value = cos_table[0].meta.get("val")
        if cos_value is None:
            cos_value = get_attr_tensor(gm, cos_table[0])
        if cos_value is None or int(cos_value.shape[-1]) != head_dim:
            return "rope-table-width"

        if not nanojet_supports(
            "fused_qkv_gemm_norm_rope",
            input_dtype="float8_e4m3fn",
            weight_dtype="float8_e4m3fn",
            head_dim=head_dim,
        ):
            return "nanojet-declined"

        return (
            query_projection,
            key_projection,
            value_projection,
            query_norm_weight,
            key_norm_weight,
            eps,
            cos_table,
            sin_table,
            head_dim,
        )

    @staticmethod
    def _find_value_projection(
        gm: GraphModule, query: Fp8Projection, key: Fp8Projection
    ) -> Optional[Fp8Projection]:
        """The remaining FP8 projection reading the same activations as Q and K."""
        activations = query.node.args[0]
        if not isinstance(activations, Node):
            return None
        candidates: List[Fp8Projection] = []
        for user in activations.users:
            if user is query.node or user is key.node:
                continue
            projection = match_fp8_projection(gm, user)
            if projection is not None:
                candidates.append(projection)
        return candidates[0] if len(candidates) == 1 else None

    def _rewrite(
        self, gm: GraphModule, rope_node: Node, match, cache_nodes: dict, order: dict
    ) -> int:
        (
            query_projection,
            key_projection,
            value_projection,
            query_norm_weight,
            key_norm_weight,
            eps,
            cos_table,
            sin_table,
            head_dim,
        ) = match
        graph = gm.graph
        anchor = max(
            (query_projection.node, key_projection.node, value_projection.node),
            key=lambda candidate: order.get(candidate, 0),
        ).next

        stacked = torch.cat(
            [query_projection.weight, key_projection.weight, value_projection.weight], dim=0
        ).contiguous()
        # node.name, not id(): a reused address would make two layers share a weight.
        weight_name = f"nanojet_qkv_weight_{rope_node.name}"
        gm.register_buffer(weight_name, stacked)

        cache_key = (cos_table[0], sin_table[0])
        if cache_key not in cache_nodes:
            half_dim = head_dim // 2
            table = cos_table[0].meta.get("val")
            with graph.inserting_before(anchor):
                cos_half = graph.call_function(
                    torch.ops.aten.slice.Tensor, args=(cos_table[0], -1, 0, half_dim)
                )
                sin_half = graph.call_function(
                    torch.ops.aten.slice.Tensor, args=(sin_table[0], -1, 0, half_dim)
                )
                stitched = graph.call_function(
                    torch.ops.aten.cat.default, args=([cos_half, sin_half], -1)
                )
                contiguous = graph.call_function(
                    torch.ops.aten.contiguous.default, args=(stitched,)
                )
                if table is not None:
                    half_shape = (*table.shape[:-1], half_dim)
                    set_val_meta(cos_half, table, half_shape)
                    set_val_meta(sin_half, table, half_shape)
                    set_val_meta(stitched, table, (*table.shape[:-1], 2 * half_dim))
                    set_val_meta(contiguous, table, (*table.shape[:-1], 2 * half_dim))
                cache_nodes[cache_key] = contiguous
        cache_node = cache_nodes[cache_key]

        with graph.inserting_before(anchor):
            weight_node = graph.get_attr(weight_name)
            set_val_meta(weight_node, stacked)
            position_key = ("positions", cos_table[1])
            if position_key not in cache_nodes:
                positions = graph.call_function(
                    torch.ops.aten.to.dtype, args=(cos_table[1], torch.int32)
                )
                index_value = cos_table[1].meta.get("val")
                if index_value is not None:
                    positions.meta["val"] = index_value.new_empty(
                        index_value.shape, dtype=torch.int32
                    )
                    positions.meta.pop("tensor_meta", None)
                cache_nodes[position_key] = positions
            fused = graph.call_function(
                torch.ops.auto_deploy.nanojet_fused_qkv_gemm_norm_rope.default,
                args=(
                    query_projection.node.args[0],
                    weight_node,
                    query_norm_weight,
                    key_norm_weight,
                    cache_node,
                    cache_nodes[position_key],
                    query_projection.input_scale_node,
                    eps,
                    query_projection.weight_scale * query_projection.input_scale,
                    key_projection.weight_scale * query_projection.input_scale,
                    value_projection.weight_scale * query_projection.input_scale,
                    int(query_projection.weight.shape[0]),
                    int(key_projection.weight.shape[0]),
                ),
            )
            query_size = int(query_projection.weight.shape[0])
            kv_size = int(key_projection.weight.shape[0])
            # The rotation returns q and k, so its meta is a tuple; either carries the batch
            # and sequence dims and the dtype the fused node keeps.
            rope_value = rope_node.meta.get("val")
            template = rope_value[0] if isinstance(rope_value, (tuple, list)) else rope_value
            leading = tuple(template.shape[:-2]) if template is not None else None

            def slice_out(start: int, stop: int) -> Node:
                node = graph.call_function(
                    torch.ops.aten.slice.Tensor, args=(fused, -1, start, stop)
                )
                if leading is not None:
                    set_val_meta(node, template, (*leading, stop - start))
                return node

            def split_heads(node: Node, width: int) -> Node:
                split = graph.call_function(
                    torch.ops.aten.unflatten.int, args=(node, -1, [width // head_dim, head_dim])
                )
                if leading is not None:
                    set_val_meta(split, template, (*leading, width // head_dim, head_dim))
                return split

            if leading is not None:
                set_val_meta(fused, template, (*leading, query_size + 2 * kv_size))

            query_out = split_heads(slice_out(0, query_size), query_size)
            key_out = split_heads(slice_out(query_size, query_size + kv_size), kv_size)
            value_out = slice_out(query_size + kv_size, query_size + 2 * kv_size)

        # The rotation's schema names its arguments q and k, so result 0 is Q's.
        query_getitem, key_getitem = extract_output_tuple(rope_node, 2)
        for getitem, replacement in ((query_getitem, query_out), (key_getitem, key_out)):
            if getitem is not None:
                getitem.replace_all_uses_with(replacement)
        value_projection.node.replace_all_uses_with(value_out)
        return 1
