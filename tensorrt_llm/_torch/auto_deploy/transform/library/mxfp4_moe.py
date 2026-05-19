# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Literal, Optional, Tuple, Type

import torch
import torch.nn as nn
from pydantic import Field
from torch.fx import GraphModule, Node

from ..._compat import get_sm_version
from ...utils.logger import ad_logger
from ...utils.module import get_submodule_of_param
from ...utils.node_utils import is_op
from ...utils.pattern_matcher import ADPatternMatcherPass, register_ad_pattern
from ..interface import BaseTransform, TransformConfig, TransformInfo, TransformRegistry

# Backend selection for MXFP4 MoE quantization.
# - "triton": use the triton_mxfp4_moe kernel (Ampere/Hopper compatible).
# - "trtllm": use the trtllm-gen MXFP4 MoE kernel (Blackwell SM>=100 only).
# When ``backend`` is left unset (``None``) on the transform config, the
# default is auto-resolved from the current SM: ``trtllm`` on SM>=100,
# ``triton`` otherwise. ``backend="trtllm"`` on SM<100 falls back to
# ``triton`` with a warning (silent fallback, not an error).
MxFP4Backend = Literal["triton", "trtllm"]


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


class InsertMXFP4MLPConfig(TransformConfig):
    """Configuration for ``quantize_mxfp4_moe``."""

    backend: Optional[MxFP4Backend] = Field(
        default=None,
        description=(
            "MXFP4 MoE kernel backend selection. When unset (``None``), the "
            "default is SM-based: ``trtllm`` on SM>=100 (Blackwell), ``triton`` "
            "otherwise. Explicit ``triton`` or ``trtllm`` overrides the default. "
            "``trtllm`` on SM<100 silently falls back to ``triton`` with a warning."
        ),
    )
    trtllm_quant_act: Literal["bf16", "mxfp8"] = Field(
        default="mxfp8",
        description=(
            "Only used when ``backend='trtllm'``. Activation precision for the "
            "trtllm-gen MoE GEMM: ``bf16`` dispatches to "
            "``trtllm_mxfp4_w4a16_moe_fused`` (bf16 input), ``mxfp8`` "
            "pre-quantizes the activation to MXFP8 and dispatches to "
            "``trtllm_mxfp4_w4a8_moe_fused`` (faster cubin family). "
            "Default ``mxfp8`` matches the modeling-side default."
        ),
    )


@TransformRegistry.register("quantize_mxfp4_moe")
class InsertMXFP4MLP(BaseTransform):
    """Quantize MXFP4 MoE: dispatch to triton or trtllm-gen backend.

    Replaces ``(torch_moe_router -> torch_moe_dense_mlp)`` with a single fused
    MoE op. The chosen backend determines the destination op and the parameter
    layout registered on the experts module:

    * ``backend="triton"`` → ``auto_deploy::triton_mxfp4_moe`` with raw HF
      MXFP4 layout (``_blocks`` / ``_scales`` / ``_bias``). Lazy weight
      swizzling happens inside the Triton kernel on first forward.
    * ``backend="trtllm"`` → ``auto_deploy::trtllm_mxfp4_*_moe_fused`` with
      trtllm-gen prepared layout (``fc1_w_trtllm`` / ``fc1_w_scale_trtllm`` /
      ...). Weight preparation (shuffle + interleave) is done on CPU inside
      a state-dict pre-hook registered by this transform, so the raw HF
      tensors are converted before being moved to GPU.
    """

    algo_name: str = "mxfp4"
    config: InsertMXFP4MLPConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return InsertMXFP4MLPConfig

    def _resolve_backend(self) -> MxFP4Backend:
        """Resolve the effective backend from config + runtime SM.

        - ``config.backend is None`` → SM-based default
            * SM>=100 → ``trtllm``
            * SM<100  → ``triton``
        - ``config.backend="trtllm"`` + SM<100 → warn + fallback to ``triton``
        - Otherwise honour the explicit config value.
        """
        requested = self.config.backend
        sm = get_sm_version()
        if requested is None:
            return "trtllm" if sm >= 100 else "triton"
        if requested == "trtllm" and sm < 100:
            ad_logger.warning(
                f"quantize_mxfp4_moe: backend='trtllm' requires SM>=100 (Blackwell), "
                f"but current SM={sm}. Falling back to backend='triton'."
            )
            return "triton"
        return requested

    def _apply(
        self,
        gm: GraphModule,
        cm,
        factory,
        shared_config,
    ) -> Tuple[GraphModule, TransformInfo]:
        """Dispatcher: pick a backend and delegate to the corresponding method.

        The actual graph rewrite + parameter swap lives in
        :meth:`_apply_triton` / :meth:`_apply_trtllm`. This method only:
        1. Skips if quant_method != "mxfp4".
        2. Resolves the backend (``triton`` | ``trtllm``) and dispatches.
        """
        qcfg = factory.get_quant_config()
        if not qcfg or qcfg.get("quant_method", "") != self.algo_name:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        backend = self._resolve_backend()
        ad_logger.info(f"quantize_mxfp4_moe: dispatching to backend={backend!r}")

        if backend == "triton":
            return self._apply_triton(gm, cm, factory, shared_config)
        elif backend == "trtllm":
            return self._apply_trtllm(gm, cm, factory, shared_config)
        else:
            # _resolve_backend should only return "triton" or "trtllm".
            raise ValueError(f"Unexpected backend resolved: {backend!r}")

    def _apply_triton(
        self,
        gm: GraphModule,
        cm,
        factory,
        shared_config,
    ) -> Tuple[GraphModule, TransformInfo]:
        """Triton backend: graph rewrite to ``triton_mxfp4_moe``.

        Replaces ``(torch_moe_router -> torch_moe_dense_mlp)`` with a single
        ``auto_deploy::triton_mxfp4_moe`` op and registers raw HF-layout
        MXFP4 params (``_blocks`` / ``_scales``) on the experts module via
        :func:`_register_mxfp4_expert_params`. The bf16 placeholders
        (``gate_up_proj`` / ``down_proj``) are deleted; biases are kept.

        Weight swizzling for the Triton kernel happens lazily inside the
        kernel on first forward (see ``_prepare_weights_scales_cached`` in
        ``custom_ops/fused_moe/mxfp4_moe.py``) -- no load hook needed
        because the HF state-dict keys already match the registered param
        names (``gate_up_proj_blocks``, ``gate_up_proj_scales``, etc.).
        """
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

    def _apply_trtllm(
        self,
        gm: GraphModule,
        cm,
        factory,
        shared_config,
    ) -> Tuple[GraphModule, TransformInfo]:
        """TRT-LLM-Gen backend: graph rewrite + CPU-side weight prep hook.

        Per MoE node:

        1. Find ``torch_moe_dense_mlp`` + its upstream ``torch_moe_router``.
        2. Look up the experts module that owns the bf16 placeholder params.
        3. Compute prepared-shape MXFP4 params via
           :func:`prepare_mxfp4_weights_for_trtllm` on shape-only zero
           tensors (so ``init_empty_weights`` / meta-device context is
           preserved). Register them on the experts module:
           ``fc1_w_trtllm`` / ``fc1_w_scale_trtllm`` / ``fc1_bias_trtllm`` /
           ``fc2_*`` + SwiGLU constants. Tag the experts module with
           ``_dtype_protected_params`` so ``model.to(dtype)`` doesn't
           corrupt the uint8 / fp32 dtypes.
        4. Delete the bf16 placeholders (``gate_up_proj`` / ``down_proj``).
        5. Rewrite the ``torch_moe_dense_mlp`` node to
           ``trtllm_mxfp4_w4a{8,16}_moe_fused`` (selected by
           ``config.trtllm_quant_act``).
        6. If ``moe_tp_size > 1`` insert an ``auto_deploy.all_reduce`` node
           after the downstream view (matches the modeling-side path).

        Then once for the whole module:

        7. Register a top-level ``load_state_dict`` pre-hook
           (:func:`make_mxfp4_trtllm_load_hook`) that converts raw HF
           MXFP4 state-dict entries into prepared values on CPU before
           they reach ``param.copy_()``.
        """
        import re

        from ...custom_ops.fused_moe.mxfp4_weight_prep import (
            make_mxfp4_trtllm_load_hook,
            make_swiglu_param_tensors,
            prepare_mxfp4_weights_for_trtllm,
        )

        # MoE topology: prefer the build-time ``DistConfig`` set on
        # ``shared_config`` (mirrors the legacy transform path). The
        # ``_resolve_moe_dist_info`` analogue from modeling code lives in
        # mxfp4_weight_prep.py as ``_get_default_dist_info``; here we trust
        # the explicit shared_config first.
        dc = getattr(shared_config, "dist_config", None)
        moe_tp_size = int(getattr(dc, "moe_tp_size", 1)) if dc is not None else 1
        moe_tp_rank = int(getattr(dc, "moe_tp_rank", 0)) if dc is not None else 0
        moe_ep_size = int(getattr(dc, "moe_ep_size", 1)) if dc is not None else 1
        moe_ep_rank = int(getattr(dc, "moe_ep_rank", 0)) if dc is not None else 0
        # Cover MoE-EP as well: any distributed case (tp_size>1) needs the
        # configured strategy. ``moe_tp_size > 1`` alone would miss EP-only.
        _tp_size = int(getattr(dc, "tp_size", 1)) if dc is not None else 1
        allreduce_strategy = (
            str(dc.allreduce_strategy) if dc is not None and _tp_size > 1 else "NCCL"
        )

        # Pre-compute the same dist tuple for the load hook factory so it
        # honours this transform's view of the MoE topology rather than
        # falling back to ``_get_default_dist_info`` at hook-fire time.
        def _hook_dist_info_fn():
            return (moe_tp_size, moe_tp_rank, moe_ep_size, moe_ep_rank)

        quant_act = self.config.trtllm_quant_act
        if quant_act == "mxfp8":
            target_op = torch.ops.auto_deploy.trtllm_mxfp4_w4a8_moe_fused.default
        else:
            target_op = torch.ops.auto_deploy.trtllm_mxfp4_w4a16_moe_fused.default

        # Module-level info needed once for the load hook factory.
        hidden_size_global: Optional[int] = None
        intermediate_size_global: Optional[int] = None
        num_experts_global: Optional[int] = None
        layer_indices: list = []

        layer_re = re.compile(r"\.layers\.(\d+)\.")
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
            if (
                gate_up_w_node.op != "get_attr"
                or gate_up_b_node.op != "get_attr"
                or down_w_node.op != "get_attr"
                or down_b_node.op != "get_attr"
            ):
                continue

            router_weight_node = routing_node.args[1]
            router_bias_node = routing_node.args[2]
            top_k = _get_topk_from_router(routing_node)

            gu_w_name = gate_up_w_node.target
            gu_b_name = gate_up_b_node.target
            dn_w_name = down_w_node.target
            dn_b_name = down_b_node.target

            # Shapes from the bf16 placeholders (meta is fine — only .shape is read).
            # gu_w shape: [E, H, 2I]; dn_w shape: [E, I, H] (we infer I from gu_w).
            gu_w_t = gm.get_parameter(gu_w_name)
            E_full = int(gu_w_t.shape[0])
            H = int(gu_w_t.shape[1])
            two_I = int(gu_w_t.shape[2])
            i_size = two_I // 2

            # Cross-layer consistency check (the load hook is registered once
            # for the whole module, so all layers must share these).
            if hidden_size_global is None:
                hidden_size_global = H
                intermediate_size_global = i_size
                num_experts_global = E_full
            else:
                if (H, i_size, E_full) != (
                    hidden_size_global,
                    intermediate_size_global,
                    num_experts_global,
                ):
                    raise ValueError(
                        f"quantize_mxfp4_moe(backend=trtllm): inconsistent MoE shapes "
                        f"across layers (got H={H}, I={i_size}, E={E_full}; previously "
                        f"H={hidden_size_global}, I={intermediate_size_global}, "
                        f"E={num_experts_global}). All MoE layers must share shape."
                    )

            if E_full % moe_ep_size != 0:
                raise ValueError(
                    f"num_experts ({E_full}) must be divisible by moe_ep_size ({moe_ep_size})"
                )
            e_local = E_full // moe_ep_size

            # Locate the experts module via the gate_up param path.
            experts_mod, experts_path, _ = get_submodule_of_param(gm, gu_w_name)

            # Compute prepared shapes by running the prep helper on shape-only
            # zero tensors (CPU). We only keep ``prep.<>.shape``/``.dtype`` --
            # actual data is filled by the load hook at load time.
            h_blk = max(1, H // 32)
            i_blk = max(1, i_size // 32)
            zero_kw = {"device": "cpu"}
            prep = prepare_mxfp4_weights_for_trtllm(
                torch.zeros((e_local, 2 * i_size, h_blk, 16), dtype=torch.uint8, **zero_kw),
                torch.zeros((e_local, 2 * i_size, h_blk), dtype=torch.uint8, **zero_kw),
                torch.zeros((e_local, 2 * i_size), dtype=torch.bfloat16, **zero_kw),
                torch.zeros((e_local, H, i_blk, 16), dtype=torch.uint8, **zero_kw),
                torch.zeros((e_local, H, i_blk), dtype=torch.uint8, **zero_kw),
                torch.zeros((e_local, H), dtype=torch.bfloat16, **zero_kw),
                hidden_size=H,
                intermediate_size=i_size,
                tp_size=moe_tp_size,
                tp_rank=moe_tp_rank,
            )

            num_local_experts = int(prep.fc1_weights_mxfp4.shape[0])
            local_expert_offset = moe_ep_rank * e_local
            valid_hidden_size = int(prep.valid_hidden_size)
            valid_intermediate_size = int(prep.valid_intermediate_size)

            # Register prepared-shape params (zero-init, meta-aware via
            # ``torch.empty(shape, dtype=...)`` without ``device=``).
            def _empty_like(t):
                return torch.empty(t.shape, dtype=t.dtype)

            prepared_specs = [
                ("fc1_w_trtllm", prep.fc1_weights_mxfp4),
                ("fc1_w_scale_trtllm", prep.fc1_weights_scale_ue8m0),
                ("fc1_bias_trtllm", prep.fc1_bias_f32),
                ("fc2_w_trtllm", prep.fc2_weights_mxfp4),
                ("fc2_w_scale_trtllm", prep.fc2_weights_scale_ue8m0),
                ("fc2_bias_trtllm", prep.fc2_bias_f32),
            ]
            for short, ref in prepared_specs:
                experts_mod.register_parameter(
                    short,
                    nn.Parameter(_empty_like(ref), requires_grad=False),
                )

            a, b, c = make_swiglu_param_tensors(num_local_experts)
            experts_mod.register_parameter(
                "swiglu_alpha_trtllm", nn.Parameter(a, requires_grad=False)
            )
            experts_mod.register_parameter(
                "swiglu_beta_trtllm", nn.Parameter(b, requires_grad=False)
            )
            experts_mod.register_parameter(
                "swiglu_limit_trtllm", nn.Parameter(c, requires_grad=False)
            )

            # Tell ``GptOssExperts._apply`` (and any analogous override) which
            # params must keep their kernel-required dtype across ``.to(dtype)``
            # walks. Generic mechanism: any module that inspects this attribute
            # can opt into dtype protection without hard-coding names.
            experts_mod._dtype_protected_params = tuple(name for name, _ in prepared_specs) + (
                "swiglu_alpha_trtllm",
                "swiglu_beta_trtllm",
                "swiglu_limit_trtllm",
            )

            # Track layer index so the load hook iterates the right range.
            m = layer_re.search(experts_path or "")
            if m:
                layer_indices.append(int(m.group(1)))

            # Build get_attr nodes for the new prepared params.
            prefix_path = (experts_path + ".") if experts_path else ""
            with gm.graph.inserting_before(n):
                fc1_w_attr = gm.graph.create_node("get_attr", prefix_path + "fc1_w_trtllm")
                fc2_w_attr = gm.graph.create_node("get_attr", prefix_path + "fc2_w_trtllm")
                fc1_s_attr = gm.graph.create_node("get_attr", prefix_path + "fc1_w_scale_trtllm")
                fc2_s_attr = gm.graph.create_node("get_attr", prefix_path + "fc2_w_scale_trtllm")
                fc1_b_attr = gm.graph.create_node("get_attr", prefix_path + "fc1_bias_trtllm")
                fc2_b_attr = gm.graph.create_node("get_attr", prefix_path + "fc2_bias_trtllm")
                sa_attr = gm.graph.create_node("get_attr", prefix_path + "swiglu_alpha_trtllm")
                sb_attr = gm.graph.create_node("get_attr", prefix_path + "swiglu_beta_trtllm")
                sl_attr = gm.graph.create_node("get_attr", prefix_path + "swiglu_limit_trtllm")

            # Rewrite the op call. Op target is chosen by ``trtllm_quant_act``.
            #   - "bf16"  -> trtllm_mxfp4_w4a16_moe_fused (bf16 input)
            #   - "mxfp8" -> trtllm_mxfp4_w4a8_moe_fused  (MXFP8 input)
            n.target = target_op
            n.kwargs = {}
            n.args = (
                hidden_node,
                router_weight_node,
                router_bias_node,
                int(top_k),
                fc1_w_attr,
                fc2_w_attr,
                fc1_s_attr,
                fc2_s_attr,
                fc1_b_attr,
                fc2_b_attr,
                sa_attr,
                sb_attr,
                sl_attr,
                valid_hidden_size,
                valid_intermediate_size,
                local_expert_offset,
                num_local_experts,
                1,  # routing_method_type = RoutingMethodType.Renormalize
            )

            # Distributed MoE: insert an all_reduce after the downstream view so
            # the ``MoE -> view -> AR -> add -> norm`` ordering matches
            # ``fuse_allreduce_residual_rmsnorm`` (see legacy transform's
            # rationale for the same placement).
            #
            # Both MoE-TP and MoE-EP need an AR after the local MoE op:
            #   - MoE-TP: each rank computes partial inner-product (summed
            #     by AR to reconstruct the full intermediate-dim contraction).
            #   - MoE-EP: each rank computes outputs only for its local
            #     expert range (zero contribution from other experts);
            #     AR sums per-token outputs across ranks.
            # Use ``tp_size > 1`` (= ``moe_tp_size * moe_ep_size *
            # moe_cluster_size > 1``) so the AR fires for any distributed
            # configuration. Matches taylor's pre-refactor modeling code
            # which emitted an unconditional AR placeholder at this exact
            # spot (commit bad1871004 + 93f78e962c, validated EP=2 GSM8K
            # 88.02%).
            tp_size = (
                int(getattr(dc, "tp_size", 1)) if dc is not None else 1
            )
            if tp_size > 1:
                from .sharding import _get_dist_ops

                _, all_reduce_op = _get_dist_ops("auto")
                view_node = next(
                    (
                        u
                        for u in n.users.keys()
                        if u.op == "call_function" and u.target == torch.ops.aten.view.default
                    ),
                    None,
                )
                anchor = view_node if view_node is not None else n
                with gm.graph.inserting_after(anchor):
                    red = gm.graph.call_function(
                        all_reduce_op,
                        args=(anchor, allreduce_strategy),
                    )
                    anchor.replace_all_uses_with(red)
                    red.replace_input_with(red, anchor)

            # Erase old router node + stale bf16 get_attr nodes if unused.
            if len(routing_node.users) == 0:
                gm.graph.erase_node(routing_node)
            for stale_node in (
                gate_up_w_node,
                gate_up_b_node,
                down_w_node,
                down_b_node,
            ):
                if len(stale_node.users) == 0:
                    gm.graph.erase_node(stale_node)

            # Free bf16 placeholders from the experts module so they don't
            # linger as orphaned attributes (and don't get loaded from HF
            # via the standard load_state_dict path).
            for stale_name in (gu_w_name, gu_b_name, dn_w_name, dn_b_name):
                owner_mod, _path, attr_short = get_submodule_of_param(gm, stale_name)
                _delete_module_attr(owner_mod, attr_short)

            num_matches += 1

        # Register top-level load hook once for the whole module so the
        # raw HF MXFP4 state_dict entries are converted to prepared layout
        # BEFORE ``param.copy_()`` -- avoids the legacy POST_LOAD_FUSION
        # raw/prepared double-allocation cycle.
        if num_matches > 0:
            assert hidden_size_global is not None  # for type checker
            num_layers = (max(layer_indices) + 1) if layer_indices else num_matches
            gm._register_load_state_dict_pre_hook(
                make_mxfp4_trtllm_load_hook(
                    num_layers=num_layers,
                    hidden_size=hidden_size_global,
                    intermediate_size=intermediate_size_global,
                    num_experts=num_experts_global,
                    dist_info_fn=_hook_dist_info_fn,
                )
            )
            ad_logger.info(
                f"quantize_mxfp4_moe (backend=trtllm, quant_act={quant_act}): "
                f"rewrote {num_matches} MoE node(s); registered load hook for "
                f"{num_layers} layer slots."
            )

        info = TransformInfo(
            skipped=(num_matches == 0),
            num_matches=num_matches,
            is_clean=(num_matches == 0),
            has_valid_shapes=(num_matches == 0),
        )
        return gm, info


def _delete_module_attr(module: nn.Module, name: str) -> None:
    """Remove a parameter/buffer/attr from a Module if present."""
    if name in module._parameters:
        del module._parameters[name]
    elif name in module._buffers:
        del module._buffers[name]
    elif hasattr(module, name):
        delattr(module, name)
