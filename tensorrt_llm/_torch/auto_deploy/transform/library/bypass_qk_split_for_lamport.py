# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Transform: bypass split_output for Q/K inputs to lamport_sharded_qk_rmsnorm.

After fuse_gemms runs, the fused Q+K+V output [M, q_dim+k_dim+v_dim] is routed
through a Python `split_output` helper that calls `.contiguous()` on each of
Q, K, V. Q and K feed lamport_sharded_qk_rmsnorm (which now supports strided
inputs), so their contiguous copies are wasted work (2 kernels per layer).

This transform rewrites:

    fused = trtllm_finegrained_fp8_linear_prequant(...)  # [M, 4096]
    (Q_c, K_c, V_c) = split_output(fused)                # 3 contiguous copies
    q_norm, k_norm = lamport_sharded_qk_rmsnorm(Q_c, K_c, ...)
    ... = view(V_c, ...)                                 # needs contiguous

to:

    fused = trtllm_finegrained_fp8_linear_prequant(...)  # [M, 4096]
    Q_view = aten.narrow(fused, -1, 0, q_dim)            # strided view, no kernel
    K_view = aten.narrow(fused, -1, q_dim, k_dim)        # strided view, no kernel
    V_c    = aten.clone(aten.narrow(fused, -1, q_dim+k_dim, v_dim))  # 1 contig kernel
    q_norm, k_norm = lamport_sharded_qk_rmsnorm(Q_view, K_view, ...)

Net: -2 kernels per fused QK-norm call.
"""

import operator
from typing import Tuple, Type

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import eliminate_dead_code
from ...utils.node_utils import is_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


def _split_output_info(node: Node):
    """If node is getitem(split_output, idx), return (split_node, idx, fused_input).

    Returns None otherwise.
    """
    if node.op != "call_function" or node.target is not operator.getitem:
        return None
    split_node, idx = node.args[0], node.args[1]
    if not isinstance(split_node, Node):
        return None
    if split_node.op != "call_function":
        return None
    # The split helper is an inline closure; match by name heuristic.
    target_repr = getattr(split_node.target, "__qualname__", "") or str(split_node.target)
    if "split_output" not in target_repr:
        return None
    fused_input = split_node.args[0]
    if not isinstance(fused_input, Node):
        return None
    return split_node, int(idx), fused_input


@TransformRegistry.register("bypass_qk_split_for_lamport")
class BypassQKSplitForLamport(BaseTransform):
    """Replace split_output copies with narrow views when inputs feed lamport QK norm.

    Works together with the strided-read path in lamport_sharded_qk_rmsnorm:
    Q and K become strided views of the fused GEMM output; V gets a clone
    (still required by the downstream aten.view).
    """

    config: TransformConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return TransformConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph
        cnt = 0

        for node in list(graph.nodes):
            if not is_op(node, torch.ops.auto_deploy.lamport_sharded_qk_rmsnorm):
                continue

            q_node, k_node = node.args[0], node.args[1]
            q_info = _split_output_info(q_node)
            k_info = _split_output_info(k_node)
            if q_info is None or k_info is None:
                continue
            q_split, q_idx, q_fused = q_info
            k_split, k_idx, k_fused = k_info
            # Q and K must come from the same split_output node (same fused GEMM)
            if q_split is not k_split or q_fused is not k_fused:
                continue
            # Expect Q at index 0 and K at index 1 (matches split_output layout)
            if q_idx != 0 or k_idx != 1:
                continue

            # Compute Q and K sizes from their current meta shapes
            q_meta = q_node.meta.get("val")
            k_meta = k_node.meta.get("val")
            if q_meta is None or k_meta is None:
                continue
            try:
                q_dim = int(q_meta.shape[-1])
                k_dim = int(k_meta.shape[-1])
            except Exception:
                continue

            # Insert narrow views for Q and K before the lamport op
            with graph.inserting_before(node):
                q_view = graph.call_function(
                    torch.ops.aten.narrow.default,
                    args=(q_fused, -1, 0, q_dim),
                )
                k_view = graph.call_function(
                    torch.ops.aten.narrow.default,
                    args=(q_fused, -1, q_dim, k_dim),
                )
            # Propagate meta (tensor shape) so downstream shape_prop is happy
            q_view.meta["val"] = q_meta
            k_view.meta["val"] = k_meta

            # Swap the lamport op inputs
            node.replace_input_with(q_node, q_view)
            node.replace_input_with(k_node, k_view)

            # Q and K getitems may now be dead. If split_output's only remaining
            # consumer is the V getitem (idx == 2), replace that with a narrow
            # + clone to keep V contiguous for the downstream aten.view.
            remaining_getitems = [
                u for u in list(q_split.users) if u is not q_node and u is not k_node
            ]
            v_getitem = None
            only_v = True
            for u in remaining_getitems:
                info = _split_output_info(u)
                if info is None or info[1] != 2:
                    only_v = False
                    break
                v_getitem = u

            if only_v and v_getitem is not None:
                v_meta = v_getitem.meta.get("val")
                if v_meta is not None:
                    try:
                        v_dim = int(v_meta.shape[-1])
                    except Exception:
                        v_dim = None
                    if v_dim is not None:
                        with graph.inserting_before(v_getitem):
                            v_narrow = graph.call_function(
                                torch.ops.aten.narrow.default,
                                args=(q_fused, -1, q_dim + k_dim, v_dim),
                            )
                            v_contig = graph.call_function(
                                torch.ops.aten.clone.default,
                                args=(v_narrow,),
                                kwargs={"memory_format": torch.contiguous_format},
                            )
                        v_narrow.meta["val"] = v_meta
                        v_contig.meta["val"] = v_meta
                        v_getitem.replace_all_uses_with(v_contig)
                        graph.erase_node(v_getitem)

            cnt += 1

        if cnt > 0:
            eliminate_dead_code(gm)
        gm.recompile()
        info = TransformInfo(
            skipped=(cnt == 0),
            num_matches=cnt,
            is_clean=(cnt == 0),
            has_valid_shapes=(cnt == 0),
        )
        return gm, info
