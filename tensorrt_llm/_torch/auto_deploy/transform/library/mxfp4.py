from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import GraphModule, Node

from tensorrt_llm._torch.auto_deploy.utils.pattern_matcher import (
    ADPatternMatcherPass,
    register_ad_pattern,
)

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


def _moe_dense_mlp_pattern(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_w: torch.Tensor,
    gate_up_b: torch.Tensor,
    down_w: torch.Tensor,
    down_b: torch.Tensor,
    alpha: float = 1.0,
    limit: float = 10.0,
    minus_limit: float = -10.0,
) -> torch.Tensor:
    batch_size = hidden_states.shape[0]
    hidden_size = hidden_states.shape[2]
    hidden_states = hidden_states.reshape(-1, hidden_size)  # (num_tokens, hidden_size)
    num_experts = routing_weights.shape[1]

    hidden_states = hidden_states.repeat(num_experts, 1)
    hidden_states = hidden_states.view(num_experts, -1, hidden_size)
    gate_up = torch.bmm(hidden_states, gate_up_w) + gate_up_b.unsqueeze(-2)
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=minus_limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    next_states = torch.bmm(((up + 1) * glu), down_w)
    next_states = next_states + down_b.unsqueeze(-2)
    next_states = next_states.view(num_experts, batch_size, -1, hidden_size)
    next_states = (
        next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
    )
    next_states = next_states.sum(dim=0)  # [B, S, H]
    return next_states


def _moe_dense_mlp_repl(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_w: torch.Tensor,
    gate_up_b: torch.Tensor,
    down_w: torch.Tensor,
    down_b: torch.Tensor,
    alpha: float,
    limit: float,
    minus_limit: float,
) -> torch.Tensor:
    return torch.ops.auto_deploy.torch_moe_dense_mlp(
        hidden_states, routing_weights, gate_up_w, gate_up_b, down_w, down_b, alpha, limit
    )


@TransformRegistry.register("match_dense_moe_pattern")
class MatchMOEDenseMLP(BaseTransform):
    def _apply(
        self,
        gm: GraphModule,
        cm,
        factory,
        shared_config,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph
        patterns = ADPatternMatcherPass()

        B, S, H = 2, 4, 8  # batch, seq, hidden
        E, In = 3, 16  # experts, intermediate (I); gate_up has 2I
        T = B * S

        dummy_args = [
            torch.randn(B, S, H, device="meta", dtype=torch.float16),  # hidden_states
            torch.randn(T, E, device="meta", dtype=torch.float16),  # routing_weights
            torch.randn(E, H, 2 * In, device="meta", dtype=torch.float16),  # gate_up_w  [E,H,2I]
            torch.randn(E, 2 * In, device="meta", dtype=torch.float16),  # gate_up_b  [E,2I]
            torch.randn(E, In, H, device="meta", dtype=torch.float16),  # down_w     [E,I,H]
            torch.randn(E, H, device="meta", dtype=torch.float16),  # down_b     [E,H]
            1.07,
            10.1,
            -10.1,
        ]

        op_ignore_types = {
            torch.ops.aten.view.default: (int,),
            torch.ops.aten.reshape.default: (int,),
            torch.ops.aten.repeat.default: (int,),
            torch.ops.aten.slice.Tensor: (int,),
            torch.ops.aten.unsqueeze.default: (int,),
            torch.ops.aten.transpose.int: (int,),
        }

        scalar_workaround = {"alpha": 1.07, "limit": 10.1, "minus_limit": -10.1}

        register_ad_pattern(
            search_fn=_moe_dense_mlp_pattern,
            replace_fn=_moe_dense_mlp_repl,
            patterns=patterns,
            dummy_args=dummy_args,
            op_ignore_types=op_ignore_types,
            scalar_workaround=scalar_workaround,
        )

        num_matches = patterns.apply(graph)
        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=False,
            has_valid_shapes=False,
        )
        return gm, info


def _router_pattern(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    top_k: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden_dim = hidden_states.shape[-1]
    hidden_states = hidden_states.reshape(-1, hidden_dim)  # [T, H]
    router_logits = F.linear(hidden_states, weight, bias)  # (seq_len, num_experts)
    router_top_value, router_indices = torch.topk(router_logits, top_k, dim=-1)  # (seq_len, top_k)
    router_top_value = torch.nn.functional.softmax(
        router_top_value, dim=1, dtype=router_top_value.dtype
    )
    router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
    return router_scores


def _router_repl(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.auto_deploy.torch_moe_router(hidden_states, weight, bias, top_k)


# This is currently not working because the pattern "crosses mutation barrier"
@TransformRegistry.register("match_moe_router_pattern")
class MatchMOERouter(BaseTransform):
    def _apply(
        self,
        gm: GraphModule,
        cm,
        factory,
        shared_config,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph
        patterns = ADPatternMatcherPass()
        print(str(graph))

        B, S, H = 4, 8, 64  # batch, seq, hidden
        E = 16  # num_experts
        K = 3  # top_k

        dummy_args = [
            torch.randn(B, S, H, device="meta", dtype=torch.float16),  # hidden_states
            torch.randn(E, H, device="meta", dtype=torch.float16),  # weight  [E, H]
            torch.randn(E, device="meta", dtype=torch.float16),  # bias    [E]
            3,
        ]

        op_ignore_types = {
            torch.ops.aten.view.default: (int,),
            torch.ops.aten.reshape.default: (int,),
            torch.ops.aten.softmax.int: (torch.dtype,),
        }

        scalar_workaround = {"top_k": K}

        register_ad_pattern(
            search_fn=_router_pattern,
            replace_fn=_router_repl,
            patterns=patterns,
            dummy_args=dummy_args,
            op_ignore_types=op_ignore_types,
            scalar_workaround=scalar_workaround,
        )

        num_matches = patterns.apply(graph)
        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=False,
            has_valid_shapes=False,
        )
        return gm, info


def _get_param(gm: GraphModule, name: str) -> torch.Tensor:
    return gm.get_parameter(name)


def _get_submodule_of_param(gm: GraphModule, param_name: str) -> Tuple[nn.Module, str, str]:
    # Returns (module, module_path, attr_name)
    if "." not in param_name:
        # param on the root
        return gm, "", param_name
    mod_path, _, attr = param_name.rpartition(".")
    return gm.get_submodule(mod_path), mod_path, attr


def _ensure_param(mod: nn.Module, name: str, tensor: torch.Tensor) -> None:
    if not hasattr(mod, name):
        mod.register_parameter(name, nn.Parameter(tensor, requires_grad=False))


def _get_alpha_limit_from_dense(node: Node) -> Tuple[float, float]:
    # torch_moe_dense_mlp(hidden, routing, gu_w, gu_b, dn_w, dn_b, alpha, limit)
    # alpha/limit may be in args or kwargs
    alpha = node.kwargs.get("alpha", None)
    limit = node.kwargs.get("limit", None)
    if alpha is None:
        alpha = float(node.args[6]) if len(node.args) >= 7 else 1.0
    if limit is None:
        limit = float(node.args[7]) if len(node.args) >= 8 else 10.0
    return float(alpha), float(limit)


def _get_topk_from_router(node: Node) -> int:
    # torch_moe_router(hidden, weight, bias, top_k=2)
    if "top_k" in node.kwargs:
        return int(node.kwargs["top_k"])
    return int(node.args[3]) if len(node.args) >= 4 else 2


def _register_mxfp4_expert_params(
    gm: GraphModule,
    gate_up_w_name: str,
    gate_up_b_name: str,
    down_w_name: str,
    down_b_name: str,
) -> Tuple[str, str, str, str]:
    """Create (if missing) the four MXFP4 params under the experts module and return their full names.

    Returns:
      (gu_blocks_name, gu_scales_name, dn_blocks_name, dn_scales_name)
    """
    # Shapes from existing params
    gu_b = _get_param(gm, gate_up_b_name)  # [E, 2I]
    gu_w = _get_param(gm, gate_up_w_name)  # typically [E, H, 2I] or [E, 2I, H]
    dn_b = _get_param(gm, down_b_name)  # [E, H]

    E = int(gu_b.shape[0])
    I2 = int(gu_b.shape[1])  # 2I
    In = I2 // 2

    # infer H from gu_w shape
    assert gu_w.dim() == 3, "gate_up_w must be rank-3"
    if gu_w.shape[1] == I2:
        H = int(gu_w.shape[2])
    elif gu_w.shape[2] == I2:
        H = int(gu_w.shape[1])
    else:
        # Fallback: use down bias last dim
        H = int(dn_b.shape[1])

    # Compute block dims (assume divisible; zero-init anyway)
    H_blk = max(1, H // 32)
    I_blk = max(1, In // 32)

    experts_mod, experts_path, _ = _get_submodule_of_param(gm, gate_up_w_name)

    # New param names under experts module
    gu_blocks_name = "gate_up_proj_blocks"
    gu_scales_name = "gate_up_proj_scales"
    dn_blocks_name = "down_proj_blocks"
    dn_scales_name = "down_proj_scales"

    # Zero-init tensors (uint8 for blocks/scales)
    gu_blocks = torch.zeros((E, 2 * In, H_blk, 16), dtype=torch.uint8)
    gu_scales = torch.zeros((E, 2 * In, H_blk), dtype=torch.uint8)
    dn_blocks = torch.zeros((E, H, I_blk, 16), dtype=torch.uint8)
    dn_scales = torch.zeros((E, H, I_blk), dtype=torch.uint8)

    _ensure_param(experts_mod, gu_blocks_name, gu_blocks)
    _ensure_param(experts_mod, gu_scales_name, gu_scales)
    _ensure_param(experts_mod, dn_blocks_name, dn_blocks)
    _ensure_param(experts_mod, dn_scales_name, dn_scales)

    # Full GM attribute paths for new params
    prefix = (experts_path + ".") if experts_path else ""
    return (
        prefix + gu_blocks_name,
        prefix + gu_scales_name,
        prefix + dn_blocks_name,
        prefix + dn_scales_name,
    )


@TransformRegistry.register("insert_mxfp4_mlp")
class InsertMXFP4MLP(BaseTransform):
    """
    Replace (torch_moe_router -> torch_moe_dense_mlp) with a single auto_deploy::mxfp4_mlp op,
    and register MXFP4 expert params (blocks + scales) on the experts module.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm,
        factory,
        shared_config,
    ) -> Tuple[GraphModule, TransformInfo]:
        num_matches = 0

        for n in list(gm.graph.nodes):
            if not is_op(n, torch.ops.auto_deploy.torch_moe_dense_mlp):
                continue

            # Expect: torch_moe_dense_mlp(hidden, routing, gu_w, gu_b, dn_w, dn_b, alpha, limit)
            if len(n.args) < 6:
                continue

            hidden_node = n.args[0]
            routing_node = n.args[1]
            gate_up_w_node = n.args[2]
            gate_up_b_node = n.args[3]
            down_w_node = n.args[4]
            down_b_node = n.args[5]

            if not isinstance(routing_node, Node) or not is_op(
                routing_node, torch.ops.auto_deploy.torch_moe_router
            ):
                continue

            # Router params: weight, bias, top_k
            router_weight_node = routing_node.args[1]
            router_bias_node = routing_node.args[2]
            top_k = _get_topk_from_router(routing_node)

            # Resolve parameter names so we can find the experts module
            if gate_up_w_node.op != "get_attr" or gate_up_b_node.op != "get_attr":
                continue
            if down_w_node.op != "get_attr" or down_b_node.op != "get_attr":
                continue

            gu_w_name = gate_up_w_node.target
            gu_b_name = gate_up_b_node.target
            dn_w_name = down_w_node.target
            dn_b_name = down_b_node.target

            # Register MXFP4 params on experts
            gu_blocks_name, gu_scales_name, dn_blocks_name, dn_scales_name = (
                _register_mxfp4_expert_params(gm, gu_w_name, gu_b_name, dn_w_name, dn_b_name)
            )

            # Alpha/limit (from dense call)
            alpha, limit = _get_alpha_limit_from_dense(n)

            # Insert the new get_attr nodes for MXFP4 params
            with gm.graph.inserting_before(n):
                gu_blocks_attr = gm.graph.create_node("get_attr", gu_blocks_name)
                gu_scales_attr = gm.graph.create_node("get_attr", gu_scales_name)
                dn_blocks_attr = gm.graph.create_node("get_attr", dn_blocks_name)
                dn_scales_attr = gm.graph.create_node("get_attr", dn_scales_name)

            n.target = torch.ops.auto_deploy.mxfp4_mlp.default
            n.kwargs = {}

            # mxfp4_mlp(
            #   hidden_states,
            #   router_weight, router_bias, top_k,
            #   gate_up_blocks, gate_up_bias, gate_up_scales, alpha, limit,
            #   down_blocks, down_bias, down_scales)
            new_args = (
                hidden_node,
                router_weight_node,
                router_bias_node,
                top_k,
                gu_blocks_attr,
                gate_up_b_node,
                gu_scales_attr,
                float(alpha),
                float(limit),
                dn_blocks_attr,
                down_b_node,
                dn_scales_attr,
            )
            n.args = new_args

            # Remove the now-unneeded router node if nobody else uses it
            if len(routing_node.users) == 0:
                gm.graph.erase_node(routing_node)

            num_matches += 1

        info = TransformInfo(
            skipped=(num_matches == 0),
            num_matches=num_matches,
            is_clean=False,
            has_valid_shapes=True,
        )
        return gm, info


def _slice_expert_dim(gm: GraphModule, tensor_node: Node, lo: int, hi: int) -> Node:
    """Return tensor_node[lo:hi, ...] via aten.slice along dim 0."""
    lo = int(lo)
    hi = int(hi)
    with gm.graph.inserting_after(tensor_node):
        # aten.slice.Tensor(self, dim, start, end, step)
        return gm.graph.call_function(
            torch.ops.aten.slice.Tensor,
            args=(tensor_node, 0, lo, hi, 1),
        )


def _insert_sharded_mxfp4_mlp_ep(
    gm: GraphModule,
    node: Node,
    rank: int,
    world_size: int,
):
    """
    Transform a call to auto_deploy::mxfp4_mlp into:
      - sharded expert parameters along dim 0 (this rank's slice),
      - call to auto_deploy::mxfp4_mlp_ep(..., local_lo, local_hi),
      - followed by torch_dist_all_reduce.

    Expects the original op signature:
      (hidden_states,
       router_weight, router_bias, top_k,
       gate_up_blocks, gate_up_bias, gate_up_scales,
       alpha, limit,
       down_blocks, down_bias, down_scales)
    """

    IDX_GATE_UP_BLOCKS = 4
    IDX_GATE_UP_BIAS = 5
    IDX_GATE_UP_SCALES = 6
    IDX_DOWN_BLOCKS = 9
    IDX_DOWN_BIAS = 10
    IDX_DOWN_SCALES = 11

    gate_up_blocks_node = node.args[IDX_GATE_UP_BLOCKS]
    num_experts = int(gate_up_blocks_node.meta["val"].shape[0])

    # Compute per-rank [lower, upper)
    base = num_experts // world_size
    rem = num_experts % world_size
    if rank < rem:
        local_lo = rank * (base + 1)
        local_hi = local_lo + (base + 1)
    else:
        local_lo = rem * (base + 1) + (rank - rem) * base
        local_hi = local_lo + base

    # Prepare new args with slices for this rank
    args = list(node.args)
    with gm.graph.inserting_before(node):
        args[IDX_GATE_UP_BLOCKS] = _slice_expert_dim(
            gm, args[IDX_GATE_UP_BLOCKS], local_lo, local_hi
        )
        args[IDX_GATE_UP_BIAS] = _slice_expert_dim(gm, args[IDX_GATE_UP_BIAS], local_lo, local_hi)
        args[IDX_GATE_UP_SCALES] = _slice_expert_dim(
            gm, args[IDX_GATE_UP_SCALES], local_lo, local_hi
        )
        args[IDX_DOWN_BLOCKS] = _slice_expert_dim(gm, args[IDX_DOWN_BLOCKS], local_lo, local_hi)
        args[IDX_DOWN_BIAS] = _slice_expert_dim(gm, args[IDX_DOWN_BIAS], local_lo, local_hi)
        args[IDX_DOWN_SCALES] = _slice_expert_dim(gm, args[IDX_DOWN_SCALES], local_lo, local_hi)

    args_ep = tuple(args) + (int(world_size), int(rank))
    node.target = torch.ops.auto_deploy.mxfp4_mlp_ep.default
    node.args = args_ep

    # Add a dist all-reduce after the op (sum partial results across EP ranks)
    with gm.graph.inserting_after(node):
        red = gm.graph.call_function(torch.ops.auto_deploy.torch_dist_all_reduce, args=(node,))
        node.replace_all_uses_with(red)
        # keep dataflow: red(input=node)
        red.replace_input_with(red, node)


@TransformRegistry.register("ep_sharding_mxfp4")
class MXFP4EPSharding(BaseTransform):
    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> GraphModule:
        """
        Walk the graph, rewrite every mxfp4_mlp into the EP form on this rank.
        """
        local_rank, world_size = shared_config.local_rank, shared_config.world_size
        if world_size <= 1:
            info = TransformInfo(
                skipped=True,
                num_matches=0,
                is_clean=True,
                has_valid_shapes=True,
            )
            return gm, info

        num_matches = 0
        for n in list(gm.graph.nodes):
            if is_op(n, torch.ops.auto_deploy.mxfp4_mlp):
                _insert_sharded_mxfp4_mlp_ep(gm, n, local_rank, world_size)
                num_matches += 1
        info = TransformInfo(
            skipped=(num_matches == 0),
            num_matches=num_matches,
            is_clean=False,
            has_valid_shapes=True,
        )
        return gm, info
