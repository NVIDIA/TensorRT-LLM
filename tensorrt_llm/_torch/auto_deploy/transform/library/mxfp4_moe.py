from typing import Tuple

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node

from ...utils.module import get_submodule_of_param
from ...utils.node_utils import is_op
from ...utils.pattern_matcher import ADPatternMatcherPass, register_ad_pattern
from ..interface import BaseTransform, TransformInfo, TransformRegistry


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
            torch.ops.auto_deploy.view.default: (int,),
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
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        return gm, info


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
    gu_b = gm.get_parameter(gate_up_b_name)  # [E, 2I]
    gu_w = gm.get_parameter(gate_up_w_name)  # [E, 2I, H]
    dn_b = gm.get_parameter(down_b_name)  # [E, H]

    E = int(gu_b.shape[0])
    I2 = int(gu_b.shape[1])  # 2I
    In = I2 // 2

    # infer H from gu_w shape
    assert gu_w.dim() == 3, "gate_up_w must be rank-3"
    if gu_w.shape[1] == I2:
        H = int(gu_w.shape[2])
    else:
        # Fallback: use down bias last dim
        H = int(dn_b.shape[1])

    # Compute block dims (assume divisible; zero-init anyway)
    H_blk = max(1, H // 32)
    I_blk = max(1, In // 32)

    experts_mod, experts_path, _ = get_submodule_of_param(gm, gate_up_w_name)

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

    experts_mod.register_parameter(gu_blocks_name, nn.Parameter(gu_blocks, requires_grad=False))
    experts_mod.register_parameter(gu_scales_name, nn.Parameter(gu_scales, requires_grad=False))
    experts_mod.register_parameter(dn_blocks_name, nn.Parameter(dn_blocks, requires_grad=False))
    experts_mod.register_parameter(dn_scales_name, nn.Parameter(dn_scales, requires_grad=False))

    # Free the now-unused bf16 stacked weight params (`gate_up_proj`, `down_proj`).
    # The biases (`gate_up_proj_bias`, `down_proj_bias`) are still consumed by
    # ``triton_mxfp4_moe`` and must remain. For models like GPT-OSS-120B
    # (128 experts × 36 layers × ~33 MB per layer of bf16 placeholder) freeing
    # these saves ~150 GB per rank.
    gu_w_local = gate_up_w_name.split(".")[-1]
    dn_w_local = down_w_name.split(".")[-1]
    for local_name in (gu_w_local, dn_w_local):
        if local_name in experts_mod._parameters:
            del experts_mod._parameters[local_name]

    # Full GM attribute paths for new params
    prefix = (experts_path + ".") if experts_path else ""
    return (
        prefix + gu_blocks_name,
        prefix + gu_scales_name,
        prefix + dn_blocks_name,
        prefix + dn_scales_name,
    )


@TransformRegistry.register("quantize_mxfp4_moe")
class InsertMXFP4MLP(BaseTransform):
    """
    Replace (torch_moe_router -> torch_moe_dense_mlp) with a single auto_deploy::triton_mxfp4_moe op,
    and register MXFP4 expert params (blocks + scales) on the experts module.
    """

    algo_name: str = "mxfp4"

    def _apply(
        self,
        gm: GraphModule,
        cm,
        factory,
        shared_config,
    ) -> Tuple[GraphModule, TransformInfo]:
        qcfg = factory.get_quant_config()
        if not qcfg or qcfg.get("quant_method", "") != self.algo_name:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )
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

            n.target = torch.ops.auto_deploy.triton_mxfp4_moe.default
            n.kwargs = {}

            # triton_mxfp4_moe(
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

            # Erase the old get_attr nodes for gate_up_proj and down_proj.
            # _register_mxfp4_expert_params deleted those attributes from the
            # experts module, so these nodes now reference non-existent attrs.
            # They have no users after the args replacement above, so it is
            # safe to erase them directly.
            for stale_node in (gate_up_w_node, down_w_node):
                if len(stale_node.users) == 0:
                    gm.graph.erase_node(stale_node)

            num_matches += 1

        info = TransformInfo(
            skipped=(num_matches == 0),
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        return gm, info


# ============================================================================
# Step-3: rewrite triton_mxfp4_moe -> trtllm_mxfp4_w4a16_moe_fused (V4)
# ============================================================================
#
# Runs in `post_load_fusion` stage (after weights are loaded). Picks up the
# MXFP4 params that quantize_mxfp4_moe registered, runs the trtllm-gen
# weight prep (pad + shuffle), registers prepared params on the experts
# module, and replaces the triton_mxfp4_moe call with the new op that
# dispatches to torch.ops.trtllm.bf16_mxe2m1_block_scale_moe_runner.
#
# Step-3 scope: supports the non-EP triton_mxfp4_moe path only (tp_size=1).
# triton_mxfp4_moe_ep is left untouched -- TP for the new op arrives in
# step 5 alongside MXFP4TRTLLMGenSharding.


_GPTOSS_GLU_ALPHA: float = 1.702
_GPTOSS_GLU_BETA: float = 1.0
_GPTOSS_GLU_LIMIT: float = 7.0


def _make_swiglu_param(
    num_local_experts: int, value: float, *, dtype=torch.float32
) -> nn.Parameter:
    return nn.Parameter(
        torch.full((num_local_experts,), value, dtype=dtype),
        requires_grad=False,
    )


def _delete_module_attr(module: nn.Module, name: str) -> None:
    """Remove a parameter/buffer/attr from a Module if present."""
    if name in module._parameters:
        del module._parameters[name]
    elif name in module._buffers:
        del module._buffers[name]
    elif hasattr(module, name):
        delattr(module, name)


@TransformRegistry.register("quantize_mxfp4_moe_trtllm_gen")
class QuantizeMXFP4MoETrtllmGen(BaseTransform):
    """Replace ``triton_mxfp4_moe`` with the trtllm-gen ``w4a16_mxfp4`` op.

    Mirrors the ``W4A16MXFP4TRTLLMGenFusedMoEMethod`` path PT uses for
    gpt-oss-120b on B200 by default. Requires that ``quantize_mxfp4_moe``
    has already run (so the MXFP4 ``_blocks``/``_scales``/``_bias`` params
    exist) and that weights have been loaded.

    TP-MoE (V6, Step 5 of MOE_TRTLLM_GEN_PLAN.md): when the runtime
    ``shared_config.dist_config`` reports ``moe_tp_size > 1``, the prep
    helper is invoked with ``tp_size`` / ``tp_rank`` so the per-rank
    ``trtllm_mxfp4_w4a16_moe_fused`` op holds only its ``I/tp`` slice of
    the intermediate dim, and an ``auto_deploy.all_reduce`` placeholder
    is inserted after the call so post-MoE partial outputs sum across
    ranks before the residual add.  EP and ``moe_ep_size > 1`` are
    handled by the legacy ``StackedMoEShardableNode`` on the upstream
    ``triton_mxfp4_moe`` (so the rewrite path here always sees the
    non-EP variant).
    """

    algo_name: str = "mxfp4"

    def _apply(
        self,
        gm: GraphModule,
        cm,
        factory,
        shared_config,
    ) -> Tuple[GraphModule, TransformInfo]:
        qcfg = factory.get_quant_config()
        if not qcfg or qcfg.get("quant_method", "") != self.algo_name:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # Local import: weight-prep helper from step 2.
        from ...custom_ops.fused_moe.mxfp4_weight_prep import prepare_mxfp4_weights_for_trtllm_gen

        # MoE-TP info (default: no TP) — read from runtime DistConfig.
        dc = getattr(shared_config, "dist_config", None)
        moe_tp_size = int(getattr(dc, "moe_tp_size", 1)) if dc is not None else 1
        moe_tp_rank = int(getattr(dc, "moe_tp_rank", 0)) if dc is not None else 0
        allreduce_strategy = (
            str(dc.allreduce_strategy) if dc is not None and moe_tp_size > 1 else "NCCL"
        )

        num_matches = 0

        for n in list(gm.graph.nodes):
            if not is_op(n, torch.ops.auto_deploy.triton_mxfp4_moe):
                continue
            # Step-3 V4 scope: skip the EP variant (covered by step 5).
            if is_op(n, torch.ops.auto_deploy.triton_mxfp4_moe_ep):
                continue

            # triton_mxfp4_moe(
            #   hidden, router_w, router_b, top_k,
            #   gate_up_blocks, gate_up_bias, gate_up_scales,
            #   alpha, limit,
            #   down_blocks, down_bias, down_scales,
            #   layer_type="moe")
            args = n.args
            if len(args) < 12:
                continue
            (
                hidden_node,
                router_w_node,
                router_b_node,
                top_k_arg,
                gu_blocks_node,
                gu_bias_node,
                gu_scales_node,
                _alpha,
                _limit,
                dn_blocks_node,
                dn_bias_node,
                dn_scales_node,
            ) = args[:12]

            # Resolve param names
            for nm, nd in [
                ("gu_blocks", gu_blocks_node),
                ("gu_bias", gu_bias_node),
                ("gu_scales", gu_scales_node),
                ("dn_blocks", dn_blocks_node),
                ("dn_bias", dn_bias_node),
                ("dn_scales", dn_scales_node),
            ]:
                if not isinstance(nd, Node) or nd.op != "get_attr":
                    raise ValueError(f"Expected {nm} arg to be a get_attr node, got {nd!r}")

            # Fetch loaded tensors and run the prep
            gu_blocks_t = gm.get_parameter(gu_blocks_node.target)
            gu_bias_t = gm.get_parameter(gu_bias_node.target)
            gu_scales_t = gm.get_parameter(gu_scales_node.target)
            dn_blocks_t = gm.get_parameter(dn_blocks_node.target)
            dn_bias_t = gm.get_parameter(dn_bias_node.target)
            dn_scales_t = gm.get_parameter(dn_scales_node.target)

            # Infer hidden / intermediate from down: [E, H, I/32, 16] or [E, H, I/2]
            hidden_size = int(dn_blocks_t.shape[1])
            two_i = int(gu_blocks_t.shape[1])
            intermediate_size = two_i // 2

            prep = prepare_mxfp4_weights_for_trtllm_gen(
                gu_blocks_t,
                gu_scales_t,
                gu_bias_t,
                dn_blocks_t,
                dn_scales_t,
                dn_bias_t,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                tp_size=moe_tp_size,
                tp_rank=moe_tp_rank,
            )

            # Locate the experts module that owned the original MXFP4 params,
            # so we can register the new ones in the same place.
            experts_mod, experts_path, _ = get_submodule_of_param(gm, gu_blocks_node.target)
            num_local_experts = int(prep.fc1_weights_mxfp4.shape[0])

            new_param_specs = [
                ("fc1_w_trtllm_gen", prep.fc1_weights_mxfp4),
                ("fc2_w_trtllm_gen", prep.fc2_weights_mxfp4),
                ("fc1_w_scale_trtllm_gen", prep.fc1_weights_scale_ue8m0),
                ("fc2_w_scale_trtllm_gen", prep.fc2_weights_scale_ue8m0),
                ("fc1_bias_trtllm_gen", prep.fc1_bias_f32),
                ("fc2_bias_trtllm_gen", prep.fc2_bias_f32),
            ]
            new_attr_paths = []
            for short, tensor in new_param_specs:
                experts_mod.register_parameter(
                    short, nn.Parameter(tensor.contiguous(), requires_grad=False)
                )
                new_attr_paths.append((experts_path + "." if experts_path else "") + short)

            sa_short, sb_short, sl_short = (
                "swiglu_alpha_trtllm_gen",
                "swiglu_beta_trtllm_gen",
                "swiglu_limit_trtllm_gen",
            )
            experts_mod.register_parameter(
                sa_short, _make_swiglu_param(num_local_experts, _GPTOSS_GLU_ALPHA)
            )
            experts_mod.register_parameter(
                sb_short, _make_swiglu_param(num_local_experts, _GPTOSS_GLU_BETA)
            )
            experts_mod.register_parameter(
                sl_short, _make_swiglu_param(num_local_experts, _GPTOSS_GLU_LIMIT)
            )
            sa_path = (experts_path + "." if experts_path else "") + sa_short
            sb_path = (experts_path + "." if experts_path else "") + sb_short
            sl_path = (experts_path + "." if experts_path else "") + sl_short

            # Build get_attr nodes for the new params.
            with gm.graph.inserting_before(n):
                attr_nodes = [gm.graph.create_node("get_attr", p) for p in new_attr_paths]
                sa_node = gm.graph.create_node("get_attr", sa_path)
                sb_node = gm.graph.create_node("get_attr", sb_path)
                sl_node = gm.graph.create_node("get_attr", sl_path)
            (fc1_w_n, fc2_w_n, fc1_s_n, fc2_s_n, fc1_b_n, fc2_b_n) = attr_nodes

            # Rewrite the op call.
            n.target = torch.ops.auto_deploy.trtllm_mxfp4_w4a16_moe_fused.default
            n.kwargs = {}
            n.args = (
                hidden_node,
                router_w_node,
                router_b_node,
                int(top_k_arg),
                fc1_w_n,
                fc2_w_n,
                fc1_s_n,
                fc2_s_n,
                fc1_b_n,
                fc2_b_n,
                sa_node,
                sb_node,
                sl_node,
                int(prep.valid_hidden_size),
                int(prep.valid_intermediate_size),
                0,  # local_expert_offset
                num_local_experts,
                1,  # routing_method_type = RoutingMethodType.Renormalize
            )

            # MoE-TP: insert an all_reduce after the V4 op so partial
            # ``[..., hidden]`` outputs from each rank sum to the full
            # hidden output before the residual add.  The ``fc2_bias``
            # was already divided by ``tp_size`` inside the prep helper,
            # so the post-AR sum reproduces the unsharded bias.
            if moe_tp_size > 1:
                from .sharding import _get_dist_ops

                _, all_reduce_op = _get_dist_ops("auto")
                with gm.graph.inserting_after(n):
                    red = gm.graph.call_function(
                        all_reduce_op,
                        args=(n, allreduce_strategy),
                    )
                    n.replace_all_uses_with(red)
                    red.replace_input_with(red, n)

            # Free original MXFP4 params + erase their get_attr nodes.
            for old_node in [
                gu_blocks_node,
                gu_bias_node,
                gu_scales_node,
                dn_blocks_node,
                dn_bias_node,
                dn_scales_node,
            ]:
                old_name = old_node.target
                owner_mod, _path, attr_short = get_submodule_of_param(gm, old_name)
                _delete_module_attr(owner_mod, attr_short)
                if len(old_node.users) == 0:
                    gm.graph.erase_node(old_node)

            num_matches += 1

        info = TransformInfo(
            skipped=(num_matches == 0),
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        return gm, info
