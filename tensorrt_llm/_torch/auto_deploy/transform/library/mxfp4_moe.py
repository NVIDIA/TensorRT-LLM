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
        """TRT-LLM-Gen backend: graph rewrite + raw HF param registration.

        Per MoE node:

        1. Find ``torch_moe_dense_mlp`` + its upstream ``torch_moe_router``.
        2. Look up the experts module that owns the bf16 placeholder params.
        3. Delete the bf16 placeholders (``gate_up_proj`` / ``down_proj`` /
           biases).
        4. Register **raw HF MXFP4 params** at the EP-sliced shape
           (``E_local = E_full / moe_ep_size``) on the experts module:
           ``gate_up_proj_{blocks,scales,bias}`` and
           ``down_proj_{blocks,scales,bias}``. Names match HF safetensors so
           the standard ``load_state_dict`` path can populate them (after the
           slim EP-slice hook below trims the leading expert axis when
           ``moe_ep_size > 1``).
        5. Also register the per-expert SwiGLU constants
           (``swiglu_alpha_trtllm`` / beta / limit) — these are not in HF
           safetensors so they are populated with their numeric defaults at
           registration time.
        6. Tag the experts module with ``_dtype_protected_params`` (raw
           uint8 weights, uint8 scales, bf16 biases, fp32 SwiGLU constants
           must all survive ``model.to(dtype)``).
        7. Rewrite the ``torch_moe_dense_mlp`` node to
           ``trtllm_mxfp4_w4a{8,16}_moe_fused`` (selected by
           ``config.trtllm_quant_act``) with args pointing at the **raw**
           params for now. The downstream :class:`FuseMXFP4Moe`
           POST_LOAD_FUSION transform will run
           :func:`prepare_mxfp4_weights_for_trtllm` on the actually-loaded
           GPU tensors, register prepared-shape params, and re-point the op
           args. The op call is therefore not runnable between PATTERN_MATCHER
           and POST_LOAD_FUSION, but no forward pass happens in that window.
        8. If ``tp_size > 1`` insert an ``auto_deploy.all_reduce`` node
           after the downstream view (covers both MoE-TP and MoE-EP).

        Then once for the whole module:

        9. Register a top-level ``load_state_dict`` pre-hook
           (:func:`make_mxfp4_ep_slice_load_hook`) that slices raw HF MXFP4
           tensors on the expert axis when ``moe_ep_size > 1``. The hook
           does **not** run any kernel-layout prep — that runs on GPU in
           :class:`FuseMXFP4Moe` after the weights are loaded.
        """
        import re

        from ...custom_ops.fused_moe.mxfp4_weight_prep import (
            make_mxfp4_sharding_load_hook,
            make_swiglu_param_tensors,
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

            # Per-rank dims after EP+TP slicing. The kernel-layout work
            # (H-axis pad, TMA shuffle, dtype convert) is deferred to
            # ``FuseMXFP4Moe`` at POST_LOAD_FUSION on GPU. EP+TP sharding is
            # done on CPU inside the load hook (see
            # :func:`make_mxfp4_sharding_load_hook`).
            h_blk = max(1, H // 32)

            # TP-aware pre-pad math (mirrors the hook). The hook pads the raw
            # intermediate axis to ``i_padded_tp`` then slices ``per_rank_i``
            # rows; ``per_rank_i`` is guaranteed to be a multiple of 128 by
            # ``_get_weight_alignment``, so it's also the per-rank kernel
            # weight-alignment size that the trtllm-gen runner expects.
            if moe_tp_size > 1:
                from tensorrt_llm._torch.modules.fused_moe.quantization import (
                    _get_weight_alignment,
                )

                _MXFP4_SCALING_VECTOR_SIZE = 32
                _WEIGHT_ALIGNMENT = 128
                alignment_tp = _get_weight_alignment(
                    _WEIGHT_ALIGNMENT, _MXFP4_SCALING_VECTOR_SIZE, moe_tp_size, i_size
                )
                i_padded_tp = (
                    (i_size + alignment_tp - 1) // alignment_tp
                ) * alignment_tp
                per_rank_i = i_padded_tp // moe_tp_size
                slice_start = moe_tp_rank * per_rank_i
                slice_stop = (moe_tp_rank + 1) * per_rank_i
                # ``valid_intermediate_size`` reports the unpadded portion of
                # this rank's slice — used by the kernel to mask OOB MMA in
                # padded regions.
                valid_intermediate_size = max(0, min(i_size, slice_stop) - slice_start)
            else:
                per_rank_i = i_size
                valid_intermediate_size = i_size

            # Local I block-count for down_proj after TP slicing.
            i_blk_local = max(1, per_rank_i // 32)
            two_i_local = 2 * per_rank_i  # gate_up's 2I axis is per-rank too

            num_local_experts = e_local
            local_expert_offset = moe_ep_rank * e_local
            valid_hidden_size = H

            # Register RAW HF MXFP4 params at the EP+TP-sliced shape — names
            # match HF safetensors so the standard load path populates them
            # after the sharding hook does the leading-axis (EP) + intermediate
            # (TP) slice on the state-dict tensors.
            raw_specs = [
                ("gate_up_proj_blocks", (e_local, two_i_local, h_blk, 16), torch.uint8),
                ("gate_up_proj_scales", (e_local, two_i_local, h_blk), torch.uint8),
                ("gate_up_proj_bias", (e_local, two_i_local), torch.bfloat16),
                ("down_proj_blocks", (e_local, H, i_blk_local, 16), torch.uint8),
                ("down_proj_scales", (e_local, H, i_blk_local), torch.uint8),
                ("down_proj_bias", (e_local, H), torch.bfloat16),
            ]
            for name, shape, dtype in raw_specs:
                experts_mod.register_parameter(
                    name,
                    nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False),
                )

            # SwiGLU constants. These are NOT in HF safetensors, so we set
            # them with their numeric defaults here (matches gpt-oss config:
            # alpha=1.702, beta=1.0, limit=7.0). The kernel expects fp32
            # tensors of length ``num_local_experts``.
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

            # Dtype protection: raw uint8 weights, uint8 scales, bf16 biases,
            # and fp32 SwiGLU constants must all survive ``model.to(dtype)``.
            # ``FuseMXFP4Moe`` will update this attribute to the prepared
            # names after running prep at POST_LOAD_FUSION.
            experts_mod._dtype_protected_params = tuple(name for name, _, _ in raw_specs) + (
                "swiglu_alpha_trtllm",
                "swiglu_beta_trtllm",
                "swiglu_limit_trtllm",
            )

            # Track layer index so the load hook iterates the right range.
            m = layer_re.search(experts_path or "")
            if m:
                layer_indices.append(int(m.group(1)))

            # Build get_attr nodes for the RAW params (will be replaced by
            # ``FuseMXFP4Moe`` once GPU-side prep produces the kernel layout).
            prefix_path = (experts_path + ".") if experts_path else ""
            with gm.graph.inserting_before(n):
                gu_blocks_attr = gm.graph.create_node(
                    "get_attr", prefix_path + "gate_up_proj_blocks"
                )
                gu_scales_attr = gm.graph.create_node(
                    "get_attr", prefix_path + "gate_up_proj_scales"
                )
                gu_bias_attr = gm.graph.create_node(
                    "get_attr", prefix_path + "gate_up_proj_bias"
                )
                dn_blocks_attr = gm.graph.create_node(
                    "get_attr", prefix_path + "down_proj_blocks"
                )
                dn_scales_attr = gm.graph.create_node(
                    "get_attr", prefix_path + "down_proj_scales"
                )
                dn_bias_attr = gm.graph.create_node(
                    "get_attr", prefix_path + "down_proj_bias"
                )
                sa_attr = gm.graph.create_node("get_attr", prefix_path + "swiglu_alpha_trtllm")
                sb_attr = gm.graph.create_node("get_attr", prefix_path + "swiglu_beta_trtllm")
                sl_attr = gm.graph.create_node("get_attr", prefix_path + "swiglu_limit_trtllm")

            # Rewrite the op call. Op target is chosen by ``trtllm_quant_act``.
            # The op args point at RAW HF MXFP4 buffers for now — the op is
            # NOT runnable until ``FuseMXFP4Moe`` (POST_LOAD_FUSION) swaps in
            # the prepared layout. That is safe because no forward pass runs
            # between PATTERN_MATCHER and POST_LOAD_FUSION.
            #   - "bf16"  -> trtllm_mxfp4_w4a16_moe_fused (bf16 input)
            #   - "mxfp8" -> trtllm_mxfp4_w4a8_moe_fused  (MXFP8 input)
            n.target = target_op
            n.kwargs = {}
            n.args = (
                hidden_node,
                router_weight_node,
                router_bias_node,
                int(top_k),
                gu_blocks_attr,  # fc1_weights_mxfp4 (raw uint8; FuseMXFP4Moe replaces)
                dn_blocks_attr,  # fc2_weights_mxfp4 (raw uint8)
                gu_scales_attr,  # fc1_weights_scale_ue8m0 (raw uint8)
                dn_scales_attr,  # fc2_weights_scale_ue8m0 (raw uint8)
                gu_bias_attr,    # fc1_bias_f32 (raw bf16; FuseMXFP4Moe converts/pads/shuffles)
                dn_bias_attr,    # fc2_bias_f32 (raw bf16)
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
            # via the standard load_state_dict path). Skip the *bias* names
            # because we re-registered them with the SAME names as new raw
            # HF MXFP4 params (``gate_up_proj_bias`` / ``down_proj_bias``);
            # those are the ones we want to keep, not delete. Only the
            # ``gate_up_proj`` / ``down_proj`` weight tensors (which don't
            # collide with any raw param name) need to be cleaned up here.
            for stale_name in (gu_w_name, dn_w_name):
                owner_mod, _path, attr_short = get_submodule_of_param(gm, stale_name)
                _delete_module_attr(owner_mod, attr_short)

            num_matches += 1

        # Register top-level EP+TP sharding load hook whenever there is any
        # actual sharding on the MoE axes. The hook only does *sharding*
        # (EP leading-axis slice + TP-aware pre-pad / intermediate-axis
        # slice) on the raw HF MXFP4 state-dict entries; the kernel-layout
        # work (H-axis pad, TMA shuffle, dtype convert, bias / tp_size) is
        # deferred to :class:`FuseMXFP4Moe` on GPU at POST_LOAD_FUSION.
        if num_matches > 0 and (moe_ep_size > 1 or moe_tp_size > 1):
            assert num_experts_global is not None  # for type checker
            assert intermediate_size_global is not None
            num_layers = (max(layer_indices) + 1) if layer_indices else num_matches
            gm._register_load_state_dict_pre_hook(
                make_mxfp4_sharding_load_hook(
                    num_layers=num_layers,
                    num_experts=num_experts_global,
                    intermediate_size=intermediate_size_global,
                    moe_ep_size=moe_ep_size,
                    moe_ep_rank=moe_ep_rank,
                    moe_tp_size=moe_tp_size,
                    moe_tp_rank=moe_tp_rank,
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


# ============================================================================
# POST_LOAD_FUSION: GPU-side MXFP4 kernel-layout prep
# ============================================================================


class FuseMXFP4MoeConfig(TransformConfig):
    """Configuration for ``fuse_mxfp4_moe`` (POST_LOAD_FUSION)."""


@TransformRegistry.register("fuse_mxfp4_moe")
class FuseMXFP4Moe(BaseTransform):
    """GPU-side MXFP4 MoE weight prep for the trtllm-gen backend.

    Runs at POST_LOAD_FUSION, after raw HF MXFP4 buffers have been loaded
    onto the experts modules by ``quantize_mxfp4_moe`` (backend=trtllm) +
    the slim EP-slice load hook.

    For each ``trtllm_mxfp4_w4a{8,16}_moe_fused`` node whose first weight
    argument still references a raw ``gate_up_proj_blocks`` buffer:

    1. Read the six raw GPU buffers (gate_up_proj_{blocks,scales,bias} and
       down_proj_{blocks,scales,bias}) from the experts module.
    2. Call :func:`prepare_mxfp4_weights_for_trtllm` on GPU to produce the
       trtllm-gen kernel layout (pad + shuffle + interleave + bf16->fp32 bias).
       Intermediate-axis TP slicing happens inside the prep helper.
    3. Register the six prepared params on the experts module
       (``fc1_w_trtllm``, ``fc1_w_scale_trtllm``, ``fc1_bias_trtllm``,
       ``fc2_w_trtllm``, ``fc2_w_scale_trtllm``, ``fc2_bias_trtllm``).
    4. Update the op call's weight args + insert new ``get_attr`` nodes
       pointing at the prepared params; old raw ``get_attr`` nodes are
       erased by graph cleanup if their use-count drops to zero.
    5. Delete the raw module params and tighten ``_dtype_protected_params``
       to the prepared-name list (so any later ``.to(dtype)`` walk
       protects the kernel-required dtypes).

    Skipping rules:
      - Op target not ``trtllm_mxfp4_w4a{8,16}_moe_fused``: ignore.
      - First weight arg's get_attr target name doesn't end in
        ``gate_up_proj_blocks``: assume already prepped, ignore.
    """

    config: FuseMXFP4MoeConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return FuseMXFP4MoeConfig

    def _apply(
        self,
        gm: GraphModule,
        cm,
        factory,
        shared_config,
    ) -> Tuple[GraphModule, TransformInfo]:
        from ...custom_ops.fused_moe.mxfp4_weight_prep import (
            prepare_mxfp4_weights_for_trtllm,
        )

        # Resolve runtime topology — used for TP slicing inside the prep
        # helper. Mirrors the values read by ``_apply_trtllm`` at
        # PATTERN_MATCHER time so per-rank shapes stay consistent.
        dc = getattr(shared_config, "dist_config", None)
        moe_tp_size = int(getattr(dc, "moe_tp_size", 1)) if dc is not None else 1
        moe_tp_rank = int(getattr(dc, "moe_tp_rank", 0)) if dc is not None else 0

        # Identify candidate ops: both w4a8 and w4a16 share the same first-arg
        # structure (raw ``gate_up_proj_blocks`` get_attr).
        target_ops = (
            torch.ops.auto_deploy.trtllm_mxfp4_w4a8_moe_fused.default,
            torch.ops.auto_deploy.trtllm_mxfp4_w4a16_moe_fused.default,
        )

        num_matches = 0
        for n in list(gm.graph.nodes):
            if n.op != "call_function" or n.target not in target_ops:
                continue
            if len(n.args) < 13:
                continue

            # Arg layout from ``_apply_trtllm`` (see comment block there):
            #   [0]  hidden_node
            #   [1]  router_weight
            #   [2]  router_bias
            #   [3]  top_k
            #   [4]  fc1_weights_mxfp4      <- raw gate_up_proj_blocks
            #   [5]  fc2_weights_mxfp4      <- raw down_proj_blocks
            #   [6]  fc1_weights_scale      <- raw gate_up_proj_scales
            #   [7]  fc2_weights_scale      <- raw down_proj_scales
            #   [8]  fc1_bias               <- raw gate_up_proj_bias (bf16)
            #   [9]  fc2_bias               <- raw down_proj_bias (bf16)
            #   [10] swiglu_alpha
            #   [11] swiglu_beta
            #   [12] swiglu_limit
            #   [13] valid_hidden_size
            #   [14] valid_intermediate_size
            #   [15] local_expert_offset
            #   [16] num_local_experts
            #   [17] routing_method_type
            gu_blocks_node = n.args[4]
            dn_blocks_node = n.args[5]
            gu_scales_node = n.args[6]
            dn_scales_node = n.args[7]
            gu_bias_node = n.args[8]
            dn_bias_node = n.args[9]

            # All six must be ``get_attr`` nodes pointing at raw HF buffers.
            raw_get_attrs = (
                gu_blocks_node,
                dn_blocks_node,
                gu_scales_node,
                dn_scales_node,
                gu_bias_node,
                dn_bias_node,
            )
            if not all(
                isinstance(a, Node) and a.op == "get_attr" for a in raw_get_attrs
            ):
                continue
            if not str(gu_blocks_node.target).endswith("gate_up_proj_blocks"):
                # Already prepped or unexpected layout — skip.
                continue

            # Locate the experts module via the raw param path.
            gu_blocks_name = gu_blocks_node.target
            experts_mod, experts_path, _ = get_submodule_of_param(gm, gu_blocks_name)

            # Read raw GPU tensors and run kernel-layout prep on GPU.
            # The load hook already did EP + TP slicing on CPU, so the
            # tensors here are at the per-rank intermediate size. We pass
            # ``tp_size=1`` to the prep helper to skip its TP-slice path
            # (would slice again otherwise), and divide the bias by the
            # *actual* ``moe_tp_size`` ourselves afterwards.
            gu_blocks = gm.get_parameter(gu_blocks_name).data
            gu_scales = gm.get_parameter(gu_scales_node.target).data
            gu_bias = gm.get_parameter(gu_bias_node.target).data
            dn_blocks = gm.get_parameter(dn_blocks_node.target).data
            dn_scales = gm.get_parameter(dn_scales_node.target).data
            dn_bias = gm.get_parameter(dn_bias_node.target).data

            # Infer per-rank dims from the (already EP+TP-sliced) raw shapes.
            e_local = int(gu_blocks.shape[0])
            two_i_local = int(gu_blocks.shape[1])
            per_rank_i = two_i_local // 2
            H = int(dn_blocks.shape[1])

            prep = prepare_mxfp4_weights_for_trtllm(
                gu_blocks,
                gu_scales,
                gu_bias,
                dn_blocks,
                dn_scales,
                dn_bias,
                hidden_size=H,
                # Pass the per-rank intermediate dim because the helper
                # treats this as the local size (no further slicing).
                intermediate_size=per_rank_i,
                tp_size=1,
                tp_rank=0,
            )

            # Bias-on-rank correction: kernel sums per-rank outputs across
            # all moe_tp ranks via the post-MoE all_reduce, which would add
            # the fc2 bias ``moe_tp_size`` times. Divide once here to make
            # the post-AR sum reproduce the unsharded bias. Matches the prep
            # helper's ``tp_size > 1`` branch (which we skip above).
            fc2_bias = prep.fc2_bias_f32
            if moe_tp_size > 1:
                fc2_bias = fc2_bias / moe_tp_size

            prepared_specs = [
                ("fc1_w_trtllm", prep.fc1_weights_mxfp4),
                ("fc1_w_scale_trtllm", prep.fc1_weights_scale_ue8m0),
                ("fc1_bias_trtllm", prep.fc1_bias_f32),
                ("fc2_w_trtllm", prep.fc2_weights_mxfp4),
                ("fc2_w_scale_trtllm", prep.fc2_weights_scale_ue8m0),
                ("fc2_bias_trtllm", fc2_bias),
            ]
            for short, tensor in prepared_specs:
                experts_mod.register_parameter(
                    short,
                    nn.Parameter(tensor.contiguous(), requires_grad=False),
                )

            # Build prepared get_attr nodes inserted right before the op call.
            prefix_path = (experts_path + ".") if experts_path else ""
            with gm.graph.inserting_before(n):
                fc1_w_attr = gm.graph.create_node("get_attr", prefix_path + "fc1_w_trtllm")
                fc2_w_attr = gm.graph.create_node("get_attr", prefix_path + "fc2_w_trtllm")
                fc1_s_attr = gm.graph.create_node(
                    "get_attr", prefix_path + "fc1_w_scale_trtllm"
                )
                fc2_s_attr = gm.graph.create_node(
                    "get_attr", prefix_path + "fc2_w_scale_trtllm"
                )
                fc1_b_attr = gm.graph.create_node("get_attr", prefix_path + "fc1_bias_trtllm")
                fc2_b_attr = gm.graph.create_node("get_attr", prefix_path + "fc2_bias_trtllm")

            new_args = list(n.args)
            new_args[4] = fc1_w_attr
            new_args[5] = fc2_w_attr
            new_args[6] = fc1_s_attr
            new_args[7] = fc2_s_attr
            new_args[8] = fc1_b_attr
            new_args[9] = fc2_b_attr
            n.args = tuple(new_args)

            # Erase raw get_attr nodes if no other consumer.
            for stale_node in raw_get_attrs:
                if len(stale_node.users) == 0:
                    gm.graph.erase_node(stale_node)

            # Delete raw module params now that prepared replaces them.
            for raw_name in (
                "gate_up_proj_blocks",
                "gate_up_proj_scales",
                "gate_up_proj_bias",
                "down_proj_blocks",
                "down_proj_scales",
                "down_proj_bias",
            ):
                _delete_module_attr(experts_mod, raw_name)

            # Update dtype protection to the prepared-name list.
            experts_mod._dtype_protected_params = tuple(name for name, _ in prepared_specs) + (
                "swiglu_alpha_trtllm",
                "swiglu_beta_trtllm",
                "swiglu_limit_trtllm",
            )

            num_matches += 1

        if num_matches > 0:
            ad_logger.info(
                f"fuse_mxfp4_moe: GPU-prepped {num_matches} MoE node(s)"
            )

        info = TransformInfo(
            skipped=(num_matches == 0),
            num_matches=num_matches,
            is_clean=(num_matches == 0),
            has_valid_shapes=(num_matches == 0),
        )
        return gm, info
