import operator

import torch
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy.custom_ops.normalization.flashinfer_fused_add_rms_norm import (  # noqa
    flashinfer_fused_add_rms_norm,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.normalization.rms_norm import *  # noqa
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.interface import TransformConfig
from tensorrt_llm._torch.auto_deploy.transform.library.fused_add_rms_norm import FuseAddRMSNorm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class AddCastNormModel(torch.nn.Module):
    """Pattern 1: add + cast(to.dtype) + rms_norm."""

    def __init__(self, hidden_size=128, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(hidden_size, device="cuda", dtype=torch.bfloat16)
        )
        self.eps = eps

    def forward(self, x, residual):
        added = x + residual
        cast = added.to(torch.bfloat16)
        norm = torch.ops.auto_deploy.flashinfer_rms_norm(cast, self.weight, self.eps)
        return norm, added


class AddNormModel(torch.nn.Module):
    """Pattern 2: add + rms_norm (no intermediate cast)."""

    def __init__(self, hidden_size=128, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(hidden_size, device="cuda", dtype=torch.bfloat16)
        )
        self.eps = eps

    def forward(self, x, residual):
        added = x + residual
        norm = torch.ops.auto_deploy.flashinfer_rms_norm(added, self.weight, self.eps)
        return norm, added


class MultiUserModel(torch.nn.Module):
    """Both add and rms_norm outputs have multiple users (DeepSeek V3 MoE pattern).

    add_result  has 2 users: rms_norm + next residual add
    norm_result has 2 users: linear1  + linear2
    """

    def __init__(self, hidden_size=128, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(hidden_size, device="cuda", dtype=torch.bfloat16)
        )
        self.linear1 = torch.nn.Linear(
            hidden_size, hidden_size, bias=False, device="cuda", dtype=torch.bfloat16
        )
        self.linear2 = torch.nn.Linear(
            hidden_size, hidden_size, bias=False, device="cuda", dtype=torch.bfloat16
        )
        self.eps = eps

    def forward(self, residual, attn_output, moe_output):
        # add with 2 users (norm + next_add)
        add_result = residual + attn_output
        # rms_norm with 2 users (linear1, linear2)
        norm_result = torch.ops.auto_deploy.flashinfer_rms_norm(add_result, self.weight, self.eps)
        out1 = self.linear1(norm_result)
        out2 = self.linear2(norm_result)
        combined = out1 + out2
        # add_result also feeds into next residual add
        next_residual = add_result + moe_output
        return combined, next_residual


class ChainedModel(torch.nn.Module):
    """Two consecutive add+norm pairs sharing residual (like transformer layers).

    Layer 1: add1 = embed + attn_out,  norm1 = rms_norm(add1)   -- add1 has 2 users
    Layer 2: add2 = add1  + mlp_out,   norm2 = rms_norm(add2)   -- add2 has 2 users
    """

    def __init__(self, hidden_size=128, eps=1e-5):
        super().__init__()
        self.weight1 = torch.nn.Parameter(
            torch.ones(hidden_size, device="cuda", dtype=torch.bfloat16)
        )
        self.weight2 = torch.nn.Parameter(
            torch.ones(hidden_size, device="cuda", dtype=torch.bfloat16)
        )
        self.linear = torch.nn.Linear(
            hidden_size, hidden_size, bias=False, device="cuda", dtype=torch.bfloat16
        )
        self.eps = eps

    def forward(self, embed, attn_out, mlp_out):
        add1 = embed + attn_out
        norm1 = torch.ops.auto_deploy.flashinfer_rms_norm(add1, self.weight1, self.eps)
        branch1 = self.linear(norm1)

        add2 = add1 + mlp_out
        norm2 = torch.ops.auto_deploy.flashinfer_rms_norm(add2, self.weight2, self.eps)
        branch2 = self.linear(norm2)

        return branch1 + branch2, add2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_fused_ops(gm):
    """Count flashinfer_fused_add_rms_norm wrapper calls in the graph."""
    return sum(
        1
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target is flashinfer_fused_add_rms_norm
    )


def _count_rms_norm_ops(gm):
    """Count flashinfer_rms_norm calls in the graph."""
    return sum(1 for n in gm.graph.nodes if is_op(n, torch.ops.auto_deploy.flashinfer_rms_norm))


def _count_add_ops(gm):
    """Count aten.add.Tensor calls in the graph."""
    return sum(1 for n in gm.graph.nodes if is_op(n, torch.ops.aten.add.Tensor))


def _export_model(model, *inputs, dynamic_dim0=True):
    """Export a model to a GraphModule, optionally with a dynamic batch dimension."""
    if dynamic_dim0:
        dyn = Dim.DYNAMIC
        ds = tuple({0: dyn} for _ in inputs)
    else:
        ds = None
    return torch_export_to_gm(model, args=inputs, dynamic_shapes=ds, clone=True)


def _apply_transform(gm):
    """Apply fuse_add_rms_norm via InferenceOptimizer (integration-style)."""
    return InferenceOptimizer(
        None,
        {"fuse_add_rms_norm": {"stage": "post_load_fusion"}},
    )(None, gm)


def _apply_transform_direct(gm):
    """Apply the transform directly (unit-test style)."""
    config = TransformConfig(stage="post_load_fusion")
    transform = FuseAddRMSNorm(config=config)
    gm, info = transform._apply(gm, None, None, None)
    return gm, info


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fuse_add_cast_rms_norm():
    """Original test: add + cast(bf16) + rms_norm → fused op."""
    model = AddCastNormModel()
    bsz, seq_len, hidden = 2, 8, 128
    x = torch.randn(bsz, seq_len, hidden, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)

    gm = _export_model(model, x, residual)
    gm_t = _apply_transform(gm)

    # Structure check
    assert _count_fused_ops(gm_t) >= 1, "fused op not found in graph"
    assert _count_rms_norm_ops(gm_t) == 0, "unfused rms_norm still in graph"

    # Numerical check
    y_fused = gm_t(x.clone(), residual.clone())
    y_ref = model(x.clone(), residual.clone())
    torch.testing.assert_close(y_fused[0], y_ref[0], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(y_fused[1], y_ref[1], atol=1e-2, rtol=1e-2)


def test_fuse_add_rms_norm_no_cast():
    """Pattern 2: add + rms_norm (no cast) → fused op."""
    model = AddNormModel()
    bsz, seq_len, hidden = 2, 8, 128
    x = torch.randn(bsz, seq_len, hidden, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)

    gm = _export_model(model, x, residual)
    gm_t = _apply_transform(gm)

    # Structure check
    assert _count_fused_ops(gm_t) >= 1, "fused op not found in graph"
    assert _count_rms_norm_ops(gm_t) == 0, "unfused rms_norm still in graph"

    # Numerical check
    y_fused = gm_t(x.clone(), residual.clone())
    y_ref = model(x.clone(), residual.clone())
    torch.testing.assert_close(y_fused[0], y_ref[0], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(y_fused[1], y_ref[1], atol=1e-2, rtol=1e-2)


def test_fuse_add_rms_norm_multi_user():
    """Multi-user: both add (2 users) and rms_norm (2 users) → fused op.

    This is the key pattern from the DeepSeek V3 / GLM4-MoE graph that failed
    with the old inductor-based pattern matcher due to num_users constraints.
    """
    model = MultiUserModel()
    bsz, seq_len, hidden = 2, 8, 128
    residual = torch.randn(bsz, seq_len, hidden, device="cuda", dtype=torch.bfloat16)
    attn_out = torch.randn_like(residual)
    moe_out = torch.randn_like(residual)

    gm = _export_model(model, residual, attn_out, moe_out)

    # Before: 1 add+norm fusible pair, add has 2 users, norm has 2 users
    assert _count_rms_norm_ops(gm) == 1

    gm_t, info = _apply_transform_direct(gm)

    # Structure check
    assert info.num_matches == 1, f"Expected 1 match, got {info.num_matches}"
    assert _count_fused_ops(gm_t) == 1, "fused op not found in graph"
    assert _count_rms_norm_ops(gm_t) == 0, "unfused rms_norm still in graph"

    # Verify getitem nodes for both outputs
    getitems = [
        n
        for n in gm_t.graph.nodes
        if n.op == "call_function"
        and n.target is operator.getitem
        and isinstance(n.args[0], torch.fx.Node)
        and n.args[0].target is flashinfer_fused_add_rms_norm
    ]
    assert len(getitems) == 2, f"Expected 2 getitem nodes, got {len(getitems)}"

    # Numerical check
    y_fused = gm_t(residual.clone(), attn_out.clone(), moe_out.clone())
    y_ref = model(residual.clone(), attn_out.clone(), moe_out.clone())
    torch.testing.assert_close(y_fused[0], y_ref[0], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(y_fused[1], y_ref[1], atol=1e-2, rtol=1e-2)


def test_fuse_add_rms_norm_chained():
    """Chained: two consecutive add+norm pairs across transformer layers."""
    model = ChainedModel()
    bsz, seq_len, hidden = 2, 8, 128
    embed = torch.randn(bsz, seq_len, hidden, device="cuda", dtype=torch.bfloat16)
    attn_out = torch.randn_like(embed)
    mlp_out = torch.randn_like(embed)

    gm = _export_model(model, embed, attn_out, mlp_out)

    # Before: 2 add+norm fusible pairs
    assert _count_rms_norm_ops(gm) == 2

    gm_t, info = _apply_transform_direct(gm)

    # Structure check
    assert info.num_matches == 2, f"Expected 2 matches, got {info.num_matches}"
    assert _count_fused_ops(gm_t) == 2, "Expected 2 fused ops"
    assert _count_rms_norm_ops(gm_t) == 0, "unfused rms_norm still in graph"

    # Verify second fused op receives add_out from first fused op (residual chain)
    fused_nodes = [
        n
        for n in gm_t.graph.nodes
        if n.op == "call_function" and n.target is flashinfer_fused_add_rms_norm
    ]
    assert len(fused_nodes) == 2
    # The second fused op's residual arg should be a getitem from the first fused op
    second_residual_arg = fused_nodes[1].args[1]
    assert (
        second_residual_arg.op == "call_function"
        and second_residual_arg.target is operator.getitem
        and second_residual_arg.args[0] is fused_nodes[0]
    ), "Second fused op's residual should come from first fused op's add_out"

    # Numerical check
    y_fused = gm_t(embed.clone(), attn_out.clone(), mlp_out.clone())
    y_ref = model(embed.clone(), attn_out.clone(), mlp_out.clone())
    torch.testing.assert_close(y_fused[0], y_ref[0], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(y_fused[1], y_ref[1], atol=1e-2, rtol=1e-2)
