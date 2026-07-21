# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ``fuse_mxfp4_moe`` POST_LOAD_FUSION transform.

Pins the post-quantize / post-load contract of :class:`FuseMXFP4Moe`:

* After running, raw HF MXFP4 buffers on the experts module
  (``gate_up_proj_{blocks,scales,bias}`` / ``down_proj_{blocks,scales,bias}``)
  are deleted and replaced by the six kernel-layout ``*_trtllm`` params
  produced by :func:`prepare_trtllm_gen_moe_mxfp4_weights`.
* The ``trtllm_quant_mxfp4_trtllm_gen_moe_fused`` op's weight/bias arg slots
  are re-pointed at the new prepared get_attr nodes — the op is again
  runnable.
* When ``moe_tp_size > 1``, the prepared fc2 bias is divided by
  ``moe_tp_size`` so that the post-AR sum reproduces the unsharded bias.
"""

from typing import Tuple

import pytest
import torch
import torch.nn as nn

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 (op registration)
import tensorrt_llm._torch.auto_deploy.transform.library.fused_moe_mxfp4  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.prepare_trtllm_gen_moe_mxfp4_weights import (
    prepare_trtllm_gen_moe_mxfp4_weights,
)
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig, TransformRegistry
from tensorrt_llm._torch.auto_deploy.utils.dist_config import DistConfig

# The transform calls ``prepare_trtllm_gen_moe_mxfp4_weights`` which itself
# invokes ``torch.ops.trtllm.shuffle_matrix`` — registered CUDA-only.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fuse_mxfp4_moe runs prepare_trtllm_gen_moe_mxfp4_weights which is CUDA-only",
)


# Small shapes that still respect the MXFP4 block size (32) and the kernel
# weight alignment (128) so prep runs without padding surprises.
E = 4
H = 128
I = 128  # noqa: E741


def _make_raw_mxfp4_tensors(device: str = "cuda") -> Tuple[torch.Tensor, ...]:
    """Build a deterministic raw-HF-layout MXFP4 expert set on ``device``."""
    g = torch.Generator(device="cpu").manual_seed(0)
    gu_blocks = torch.randint(0, 256, (E, 2 * I, H // 32, 16), dtype=torch.uint8, generator=g).to(
        device
    )
    gu_scales = torch.randint(126, 130, (E, 2 * I, H // 32), dtype=torch.uint8, generator=g).to(
        device
    )
    gu_bias = (torch.randn(E, 2 * I, dtype=torch.bfloat16, generator=g) * 0.01).to(device)
    dn_blocks = torch.randint(0, 256, (E, H, I // 32, 16), dtype=torch.uint8, generator=g).to(
        device
    )
    dn_scales = torch.randint(126, 130, (E, H, I // 32), dtype=torch.uint8, generator=g).to(device)
    dn_bias = (torch.randn(E, H, dtype=torch.bfloat16, generator=g) * 0.01).to(device)
    return gu_blocks, gu_scales, gu_bias, dn_blocks, dn_scales, dn_bias


def _build_pre_fuse_gm(raw_tensors: Tuple[torch.Tensor, ...]) -> torch.fx.GraphModule:
    """Build a tiny GM in the exact pre-``FuseMXFP4Moe`` shape ``QuantizeMXFP4MOE`` leaves.

    Shape mirrors ``_apply_trtllm``'s output:
      * root has an ``experts`` submodule with the six raw HF MXFP4 params
        and the three SwiGLU constant params.
      * Graph: ``(hidden, router_w, router_b) -> trtllm_quant_mxfp4_trtllm_gen_moe_fused``
        whose weight/bias args are get_attrs pointing at the raw experts params.
    """
    gu_blocks, gu_scales, gu_bias, dn_blocks, dn_scales, dn_bias = raw_tensors

    root = nn.Module()
    root.experts = nn.Module()
    raw_specs = [
        ("gate_up_proj_blocks", gu_blocks),
        ("gate_up_proj_scales", gu_scales),
        ("gate_up_proj_bias", gu_bias),
        ("down_proj_blocks", dn_blocks),
        ("down_proj_scales", dn_scales),
        ("down_proj_bias", dn_bias),
    ]
    for name, t in raw_specs:
        root.experts.register_parameter(name, nn.Parameter(t.clone(), requires_grad=False))

    # SwiGLU constants (gpt-oss defaults). Must live on the experts module
    # because their get_attr nodes are inserted with that path.
    a = torch.full((E,), 1.702, dtype=torch.float32, device=gu_blocks.device)
    b = torch.full((E,), 1.0, dtype=torch.float32, device=gu_blocks.device)
    c = torch.full((E,), 7.0, dtype=torch.float32, device=gu_blocks.device)
    root.experts.register_parameter("swiglu_alpha_trtllm", nn.Parameter(a, requires_grad=False))
    root.experts.register_parameter("swiglu_beta_trtllm", nn.Parameter(b, requires_grad=False))
    root.experts.register_parameter("swiglu_limit_trtllm", nn.Parameter(c, requires_grad=False))

    graph = torch.fx.Graph()
    hidden = graph.placeholder("hidden")
    router_w = graph.placeholder("router_w")
    router_b = graph.placeholder("router_b")

    gu_blocks_n = graph.get_attr("experts.gate_up_proj_blocks")
    dn_blocks_n = graph.get_attr("experts.down_proj_blocks")
    gu_scales_n = graph.get_attr("experts.gate_up_proj_scales")
    dn_scales_n = graph.get_attr("experts.down_proj_scales")
    gu_bias_n = graph.get_attr("experts.gate_up_proj_bias")
    dn_bias_n = graph.get_attr("experts.down_proj_bias")
    sa_n = graph.get_attr("experts.swiglu_alpha_trtllm")
    sb_n = graph.get_attr("experts.swiglu_beta_trtllm")
    sl_n = graph.get_attr("experts.swiglu_limit_trtllm")

    moe = graph.call_function(
        torch.ops.auto_deploy.trtllm_quant_mxfp4_trtllm_gen_moe_fused.default,
        args=(
            hidden,
            router_w,
            router_b,
            2,  # top_k
            gu_blocks_n,
            dn_blocks_n,
            gu_scales_n,
            dn_scales_n,
            gu_bias_n,
            dn_bias_n,
            sa_n,
            sb_n,
            sl_n,
            H,  # valid_hidden_size
            I,  # valid_intermediate_size
            "mxfp8",  # act_dtype
            0,  # local_expert_offset
            E,  # num_local_experts
            1,  # routing_method_type = Renormalize
        ),
    )
    graph.output(moe)

    return torch.fx.GraphModule(root, graph)


def _run_fuse(gm: torch.fx.GraphModule, dist_config: DistConfig):
    """Apply just ``FuseMXFP4Moe`` with the given ``dist_config``."""
    shared_config = SharedConfig(
        local_rank=dist_config.rank,
        world_size=dist_config.world_size,
        dist_config=dist_config,
    )
    config_cls = TransformRegistry.get_config_class("fuse_mxfp4_moe")
    config = config_cls(stage="post_load_fusion")
    transform = TransformRegistry.get("fuse_mxfp4_moe")(config)
    return transform._apply(gm, cm=None, factory=None, shared_config=shared_config)


def _moe_node(gm: torch.fx.GraphModule) -> torch.fx.Node:
    target_op = torch.ops.auto_deploy.trtllm_quant_mxfp4_trtllm_gen_moe_fused.default
    nodes = [n for n in gm.graph.nodes if n.op == "call_function" and n.target is target_op]
    assert len(nodes) == 1, f"expected exactly one MoE op node, found {len(nodes)}"
    return nodes[0]


# ---------------------------------------------------------------------------
# TP=1 — single-rank: raw → prepared swap, no bias /= moe_tp_size
# ---------------------------------------------------------------------------


def test_fuse_mxfp4_moe_tp1_raw_to_prepared_swap():
    """Single-rank: every raw HF buffer becomes a prepared ``*_trtllm`` buffer; arg slots re-pointed."""
    device = "cuda"
    raw = _make_raw_mxfp4_tensors(device=device)
    gm = _build_pre_fuse_gm(raw)

    dc = DistConfig(world_size=1, rank=0, tp_size=1, moe_tp_size=1, moe_ep_size=1)
    _, info = _run_fuse(gm, dc)

    # TransformInfo: exactly one MoE node was prepped, not idempotent-skip.
    assert info.skipped is False
    assert info.num_matches == 1

    # Raw HF params are gone.
    raw_names = (
        "gate_up_proj_blocks",
        "gate_up_proj_scales",
        "gate_up_proj_bias",
        "down_proj_blocks",
        "down_proj_scales",
        "down_proj_bias",
    )
    for name in raw_names:
        assert not hasattr(gm.experts, name) or getattr(gm.experts, name, None) is None, (
            f"raw param {name!r} should have been removed"
        )

    # Prepared params are registered (six kinds).
    prepared_names = (
        "fc1_w_trtllm",
        "fc1_w_scale_trtllm",
        "fc1_bias_trtllm",
        "fc2_w_trtllm",
        "fc2_w_scale_trtllm",
        "fc2_bias_trtllm",
    )
    for name in prepared_names:
        assert hasattr(gm.experts, name), f"prepared param {name!r} missing"

    # Op args 4..9 (fc1_w, fc2_w, fc1_s, fc2_s, fc1_b, fc2_b) point at prepared get_attrs.
    n = _moe_node(gm)
    ARG_FC1_W, ARG_FC2_W, ARG_FC1_S, ARG_FC2_S, ARG_FC1_B, ARG_FC2_B = 4, 5, 6, 7, 8, 9
    expected_targets = {
        ARG_FC1_W: "experts.fc1_w_trtllm",
        ARG_FC2_W: "experts.fc2_w_trtllm",
        ARG_FC1_S: "experts.fc1_w_scale_trtllm",
        ARG_FC2_S: "experts.fc2_w_scale_trtllm",
        ARG_FC1_B: "experts.fc1_bias_trtllm",
        ARG_FC2_B: "experts.fc2_bias_trtllm",
    }
    for slot, want in expected_targets.items():
        arg = n.args[slot]
        assert isinstance(arg, torch.fx.Node) and arg.op == "get_attr", (
            f"arg slot {slot} is not a get_attr Node (got {arg!r})"
        )
        assert arg.target == want, f"arg slot {slot} target = {arg.target!r}, want {want!r}"

    # TP=1: fc2_bias matches the raw prep output exactly (no /= moe_tp_size division).
    prep = prepare_trtllm_gen_moe_mxfp4_weights(
        *raw, hidden_size=H, intermediate_size=I, tp_size=1, tp_rank=0
    )
    torch.testing.assert_close(gm.experts.fc2_bias_trtllm.data, prep.fc2_bias_f32, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# TP=2 — fc2 bias must be divided by moe_tp_size so post-AR sum reproduces the unsharded bias
# ---------------------------------------------------------------------------


def test_fuse_mxfp4_moe_tp2_divides_fc2_bias_by_moe_tp_size():
    """``moe_tp_size > 1`` divides only the prepared ``fc2_bias`` by ``moe_tp_size``.

    Other prepared tensors (fc1/fc2 weights, fc1/fc2 scales, fc1 bias) must
    match the TP=1 prep output 1:1 — the transform leaves them alone in the
    scratch path; only ``fc2_bias`` is scaled.
    """
    device = "cuda"
    raw = _make_raw_mxfp4_tensors(device=device)
    gm = _build_pre_fuse_gm(raw)

    moe_tp_size = 2
    dc = DistConfig(
        world_size=2,
        rank=0,
        tp_size=moe_tp_size,
        moe_tp_size=moe_tp_size,
        moe_ep_size=1,
    )
    _, info = _run_fuse(gm, dc)
    assert info.num_matches == 1

    # Golden: run prep on the SAME raw tensors at tp=1 (the transform path with
    # scratch skips the helper's tp_size > 1 branch and does the division itself).
    prep = prepare_trtllm_gen_moe_mxfp4_weights(
        *raw, hidden_size=H, intermediate_size=I, tp_size=1, tp_rank=0
    )

    # fc2_bias was divided by moe_tp_size; everything else matches 1:1.
    torch.testing.assert_close(
        gm.experts.fc2_bias_trtllm.data, prep.fc2_bias_f32 / moe_tp_size, atol=0, rtol=0
    )
    torch.testing.assert_close(gm.experts.fc1_bias_trtllm.data, prep.fc1_bias_f32, atol=0, rtol=0)
    assert torch.equal(gm.experts.fc1_w_trtllm.data, prep.fc1_weights_mxfp4)
    assert torch.equal(gm.experts.fc2_w_trtllm.data, prep.fc2_weights_mxfp4)
    assert torch.equal(gm.experts.fc1_w_scale_trtllm.data, prep.fc1_weights_scale_ue8m0)
    assert torch.equal(gm.experts.fc2_w_scale_trtllm.data, prep.fc2_weights_scale_ue8m0)


# ---------------------------------------------------------------------------
# Idempotency: re-running on an already-prepped graph is a no-op
# ---------------------------------------------------------------------------


def test_fuse_mxfp4_moe_idempotent_on_already_prepped_graph():
    """Re-running ``FuseMXFP4Moe`` on its own output skips (no double-prep)."""
    device = "cuda"
    raw = _make_raw_mxfp4_tensors(device=device)
    gm = _build_pre_fuse_gm(raw)

    dc = DistConfig(world_size=1, rank=0, tp_size=1, moe_tp_size=1, moe_ep_size=1)
    _, info1 = _run_fuse(gm, dc)
    assert info1.num_matches == 1

    _, info2 = _run_fuse(gm, dc)
    assert info2.skipped is True, "second run should skip — no raw HF buffers left to prep"
    assert info2.num_matches == 0
