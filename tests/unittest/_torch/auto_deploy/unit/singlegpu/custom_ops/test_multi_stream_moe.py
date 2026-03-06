"""Tests for multi-stream MoE shared-expert transform across model architectures.

Verifies that ``_execute_shared_expert_in_aux_stream`` correctly identifies the
shared-expert branch and moves it to the auxiliary CUDA stream for the MoE
patterns used in DeepSeek V3, GLM4 MoE Lite, Mixtral, and Nemotron-H (with
and without latent projections).

Architecture patterns tested:

  **DeepSeek V3 / GLM4 MoE Lite** — Gated-MLP shared expert
    (gate_proj + up_proj → SiLU gate → down_proj).
    Routed MoE dispatched first; shared expert on ``identity``.
    Merge: ``moe_out + shared_out``.

  **Mixtral** — Pure routed MoE, *no* shared expert.
    The transform must produce zero matches (no-op).

  **Nemotron-H (no latent)** — Simple-MLP shared expert
    (up_proj → ReLU² → down_proj).
    Shared expert dispatched first; routed MoE second.
    Merge: ``shared_out + routed_out``.

  **Nemotron-H (with latent projection)** — Same shared expert as above,
    but the routed path wraps the MoE op with
    ``fc1_latent_proj → MoE → fc2_latent_proj``.
    Tests that the BFS from MoE to the merge ``add`` traverses the extra
    projection nodes correctly.

Each architecture is tested for:
  1. Pattern matching — correct number of replacements.
  2. Graph structure — ``begin_aux``, ``end_aux``, ``wait_aux`` nodes present.
  3. Numerical correctness — output matches eager reference within tolerance.
  4. CUDA graph compatibility — capture + replay produces correct output.
  5. Multi-layer stacking — multiple MoE layers handled independently.
"""

import torch
import torch.nn as nn

from tensorrt_llm._torch.auto_deploy.transform.library.multi_stream_moe import (
    _execute_shared_expert_in_aux_stream,
)
from tensorrt_llm._torch.auto_deploy.utils.multi_stream_utils import (
    begin_aux_stream_passthrough,
    cuda_stream_manager,
    end_aux_stream_passthrough,
    wait_aux_stream_passthrough,
)

# ---------------------------------------------------------------------------
# Mock fused-MoE custom op (distinct name to avoid conflicts with other tests)
# ---------------------------------------------------------------------------


@torch.library.custom_op("auto_deploy::mock_fused_moe_moe_test", mutates_args=())
def mock_fused_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_weight: torch.Tensor,
) -> torch.Tensor:
    """Mock fused MoE: a simple linear transform standing in for the real kernel."""
    return torch.ops.aten.linear(x, expert_weight)


@mock_fused_moe.register_fake
def _mock_fused_moe_fake(x, selected_experts, routing_weights, expert_weight):
    return torch.ops.aten.linear(x, expert_weight)


_MOE_OPS = [torch.ops.auto_deploy.mock_fused_moe_moe_test]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_gm(model, example_input):
    """Export *model* to an FX ``GraphModule``."""
    return torch.export.export(model, (example_input,)).module()


def _stream_targets(gm):
    """Return the set of ``call_function`` targets present in *gm*."""
    return {n.target for n in gm.graph.nodes if n.op == "call_function"}


def _assert_stream_nodes_present(gm):
    """Assert that the three stream-management passthrough nodes are in the graph."""
    targets = _stream_targets(gm)
    assert begin_aux_stream_passthrough in targets, "begin_aux_stream_passthrough not in graph"
    assert end_aux_stream_passthrough in targets, "end_aux_stream_passthrough not in graph"
    assert wait_aux_stream_passthrough in targets, "wait_aux_stream_passthrough not in graph"


def _assert_numerical_correctness(gm, model, test_x, *, atol=1e-5):
    """Assert that *gm* and *model* produce the same output on *test_x*."""
    ref = model(test_x)
    out = gm(test_x)
    assert torch.allclose(out, ref, atol=atol), (
        f"Output mismatch: max diff = {(out - ref).abs().max().item()}"
    )


def _assert_cuda_graph_correctness(gm, model, test_x, *, atol=1e-5):
    """Assert correctness under CUDA graph capture + replay."""
    ref = model(test_x)
    out_shape = ref.shape

    static_x = torch.randn_like(test_x)
    static_out = torch.empty(out_shape, device="cuda", dtype=ref.dtype)

    # Warm-up (required before capture).
    for _ in range(3):
        static_out.copy_(gm(static_x))

    cuda_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph):
        static_out.copy_(gm(static_x))

    static_x.copy_(test_x)
    cuda_graph.replay()

    assert torch.allclose(static_out, ref, atol=atol), (
        f"CUDA graph output mismatch: max diff = {(static_out - ref).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Mock modules — shared expert variants
# ---------------------------------------------------------------------------


class _GatedMLP(nn.Module):
    """DeepSeek / GLM4 shared expert: ``down_proj(silu(gate_proj(x)) * up_proj(x))``."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _SimpleMLP(nn.Module):
    """Nemotron-H shared expert: ``down_proj(relu(up_proj(x)) ** 2)``."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.relu(self.up_proj(x)) ** 2)


# ---------------------------------------------------------------------------
# Mock MoE layer modules — one per architecture pattern
# ---------------------------------------------------------------------------


class MockDeepSeekGLM4MoELayer(nn.Module):
    """DeepSeek V3 / GLM4 MoE Lite pattern.

    Graph topology::

        hidden_states ─┬─ gate ─ topk ─────────────────────────┐
                        ├─ shared_experts (gated MLP) ─ shared_out  │
                        └─ mock_fused_moe ──────────── moe_out ─┘
                                                        moe_out + shared_out → layernorm → out

    Routed MoE is dispatched *before* the shared expert in graph order.
    The ``add`` has the routed output on the left.
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int, num_experts: int = 8):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.shared_experts = _GatedMLP(hidden_dim, intermediate_dim)
        self.expert_weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        logits = self.gate(hidden_states)
        routing_weights, selected_experts = torch.topk(logits, k=2, dim=-1)

        # Routed path first (matches DeepSeek / GLM4 dispatch order).
        moe_out = torch.ops.auto_deploy.mock_fused_moe_moe_test(
            hidden_states, selected_experts, routing_weights, self.expert_weight
        )
        # Shared expert on original input.
        shared_out = self.shared_experts(identity)

        return self.layernorm(moe_out + shared_out)


class MockMixtralMoELayer(nn.Module):
    """Mixtral pattern — pure routed MoE, **no** shared expert.

    The transform must return 0 replacements for this topology.
    """

    def __init__(self, hidden_dim: int, num_experts: int = 8):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.expert_weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.gate(hidden_states)
        routing_weights, selected_experts = torch.topk(logits, k=2, dim=-1)
        moe_out = torch.ops.auto_deploy.mock_fused_moe_moe_test(
            hidden_states, selected_experts, routing_weights, self.expert_weight
        )
        return self.layernorm(moe_out)


class MockNemotronHMoELayer(nn.Module):
    """Nemotron-H pattern *without* latent projections.

    Graph topology::

        hidden_states ─┬─ gate ─ topk ─────────────────────────┐
                        ├─ shared_experts (simple MLP) ─ shared_out │
                        └─ mock_fused_moe ──────────── moe_out ─┘
                                                    shared_out + moe_out → layernorm → out

    Shared expert is dispatched *before* the MoE in graph order.
    The ``add`` has the shared output on the left.
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int, num_experts: int = 8):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.shared_experts = _SimpleMLP(hidden_dim, intermediate_dim)
        self.expert_weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        logits = self.gate(hidden_states)
        routing_weights, selected_experts = torch.topk(logits, k=2, dim=-1)

        # Shared expert dispatched first (matches Nemotron-H dispatch order).
        shared_out = self.shared_experts(residuals)
        # Routed path.
        moe_out = torch.ops.auto_deploy.mock_fused_moe_moe_test(
            hidden_states, selected_experts, routing_weights, self.expert_weight
        )

        return self.layernorm(shared_out + moe_out)


class MockNemotronHLatentMoELayer(nn.Module):
    """Nemotron-H pattern *with* latent projections.

    Graph topology::

        hidden_states ─┬─ gate ─ topk ──────────────────────────────────────┐
                        ├─ shared_experts (simple MLP) ────────── shared_out   │
                        └─ fc1_latent ─ mock_fused_moe ─ fc2_latent ─ routed_out ┘
                                                          shared_out + routed_out → ln → out

    The latent projections add nodes between the MoE op and the merge ``add``,
    testing that the forward BFS from MoE correctly traverses extra projection
    nodes.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        latent_dim: int,
        num_experts: int = 8,
    ):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.shared_experts = _SimpleMLP(hidden_dim, intermediate_dim)
        self.fc1_latent_proj = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.fc2_latent_proj = nn.Linear(latent_dim, hidden_dim, bias=False)
        # Expert weight operates in latent space.
        self.expert_weight = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        logits = self.gate(hidden_states)
        routing_weights, selected_experts = torch.topk(logits, k=2, dim=-1)

        # Shared expert dispatched first.
        shared_out = self.shared_experts(residuals)

        # Latent projection → MoE → back-projection.
        x_latent = self.fc1_latent_proj(hidden_states)
        moe_out = torch.ops.auto_deploy.mock_fused_moe_moe_test(
            x_latent, selected_experts, routing_weights, self.expert_weight
        )
        routed_out = self.fc2_latent_proj(moe_out)

        return self.layernorm(shared_out + routed_out)


# ===================================================================
# Tests — DeepSeek V3 / GLM4 MoE Lite (gated-MLP shared expert)
# ===================================================================


def test_deepseek_glm4_pattern_and_correctness():
    """Single-layer: pattern match + graph structure + numerical correctness."""
    hidden_dim, intermediate_dim = 128, 256
    cuda_stream_manager.add_device(torch.cuda.current_device())

    model = MockDeepSeekGLM4MoELayer(hidden_dim, intermediate_dim).eval().to("cuda")
    example = torch.randn(4, hidden_dim, device="cuda")
    gm = _build_gm(model, example)

    gm, num = _execute_shared_expert_in_aux_stream(gm, _MOE_OPS)

    assert num == 1, f"Expected 1 replacement, got {num}"
    _assert_stream_nodes_present(gm)
    _assert_numerical_correctness(gm, model, torch.randn(4, hidden_dim, device="cuda"))


def test_deepseek_glm4_cuda_graph():
    """CUDA graph capture + replay for DeepSeek / GLM4 pattern."""
    hidden_dim, intermediate_dim = 128, 256
    cuda_stream_manager.add_device(torch.cuda.current_device())

    model = MockDeepSeekGLM4MoELayer(hidden_dim, intermediate_dim).eval().to("cuda")
    example = torch.randn(4, hidden_dim, device="cuda")
    gm = _build_gm(model, example)
    gm, num = _execute_shared_expert_in_aux_stream(gm, _MOE_OPS)
    assert num == 1

    _assert_cuda_graph_correctness(gm, model, torch.randn(4, hidden_dim, device="cuda"))


def test_deepseek_glm4_multi_layer():
    """Two stacked DeepSeek/GLM4 MoE layers — both should be transformed."""
    hidden_dim, intermediate_dim = 128, 256
    cuda_stream_manager.add_device(torch.cuda.current_device())

    model = (
        nn.Sequential(
            MockDeepSeekGLM4MoELayer(hidden_dim, intermediate_dim),
            MockDeepSeekGLM4MoELayer(hidden_dim, intermediate_dim),
        )
        .eval()
        .to("cuda")
    )
    example = torch.randn(4, hidden_dim, device="cuda")
    gm = _build_gm(model, example)

    gm, num = _execute_shared_expert_in_aux_stream(gm, _MOE_OPS)

    assert num == 2, f"Expected 2 replacements, got {num}"
    _assert_numerical_correctness(gm, model, torch.randn(4, hidden_dim, device="cuda"))


# ===================================================================
# Tests — Mixtral (no shared expert → no-op)
# ===================================================================


def test_mixtral_no_shared_expert_no_match():
    """Mixtral has no shared expert; the transform must produce zero matches."""
    hidden_dim = 128
    cuda_stream_manager.add_device(torch.cuda.current_device())

    model = MockMixtralMoELayer(hidden_dim).eval().to("cuda")
    example = torch.randn(4, hidden_dim, device="cuda")
    gm = _build_gm(model, example)

    gm, num = _execute_shared_expert_in_aux_stream(gm, _MOE_OPS)

    assert num == 0, f"Expected 0 replacements for Mixtral (no shared expert), got {num}"

    # Graph should NOT contain any stream-management nodes.
    targets = _stream_targets(gm)
    assert begin_aux_stream_passthrough not in targets
    assert end_aux_stream_passthrough not in targets
    assert wait_aux_stream_passthrough not in targets

    # Numerical correctness should still hold (graph unchanged).
    _assert_numerical_correctness(gm, model, torch.randn(4, hidden_dim, device="cuda"))


# ===================================================================
# Tests — Nemotron-H without latent projections
# ===================================================================


def test_nemotron_h_pattern_and_correctness():
    """Single-layer Nemotron-H (no latent): pattern + graph + correctness."""
    hidden_dim, intermediate_dim = 128, 256
    cuda_stream_manager.add_device(torch.cuda.current_device())

    model = MockNemotronHMoELayer(hidden_dim, intermediate_dim).eval().to("cuda")
    example = torch.randn(4, hidden_dim, device="cuda")
    gm = _build_gm(model, example)

    gm, num = _execute_shared_expert_in_aux_stream(gm, _MOE_OPS)

    assert num == 1, f"Expected 1 replacement, got {num}"
    _assert_stream_nodes_present(gm)
    _assert_numerical_correctness(gm, model, torch.randn(4, hidden_dim, device="cuda"))


def test_nemotron_h_cuda_graph():
    """CUDA graph capture + replay for Nemotron-H (no latent) pattern."""
    hidden_dim, intermediate_dim = 128, 256
    cuda_stream_manager.add_device(torch.cuda.current_device())

    model = MockNemotronHMoELayer(hidden_dim, intermediate_dim).eval().to("cuda")
    example = torch.randn(4, hidden_dim, device="cuda")
    gm = _build_gm(model, example)
    gm, num = _execute_shared_expert_in_aux_stream(gm, _MOE_OPS)
    assert num == 1

    _assert_cuda_graph_correctness(gm, model, torch.randn(4, hidden_dim, device="cuda"))


def test_nemotron_h_multi_layer():
    """Two stacked Nemotron-H (no latent) layers — both should be transformed."""
    hidden_dim, intermediate_dim = 128, 256
    cuda_stream_manager.add_device(torch.cuda.current_device())

    model = (
        nn.Sequential(
            MockNemotronHMoELayer(hidden_dim, intermediate_dim),
            MockNemotronHMoELayer(hidden_dim, intermediate_dim),
        )
        .eval()
        .to("cuda")
    )
    example = torch.randn(4, hidden_dim, device="cuda")
    gm = _build_gm(model, example)

    gm, num = _execute_shared_expert_in_aux_stream(gm, _MOE_OPS)

    assert num == 2, f"Expected 2 replacements, got {num}"
    _assert_numerical_correctness(gm, model, torch.randn(4, hidden_dim, device="cuda"))


# ===================================================================
# Tests — Nemotron-H with latent projections
# ===================================================================


def test_nemotron_h_latent_pattern_and_correctness():
    """Single-layer Nemotron-H (with latent): pattern + graph + correctness."""
    hidden_dim, intermediate_dim, latent_dim = 128, 256, 64
    cuda_stream_manager.add_device(torch.cuda.current_device())

    model = MockNemotronHLatentMoELayer(hidden_dim, intermediate_dim, latent_dim).eval().to("cuda")
    example = torch.randn(4, hidden_dim, device="cuda")
    gm = _build_gm(model, example)

    gm, num = _execute_shared_expert_in_aux_stream(gm, _MOE_OPS)

    assert num == 1, f"Expected 1 replacement, got {num}"
    _assert_stream_nodes_present(gm)
    _assert_numerical_correctness(gm, model, torch.randn(4, hidden_dim, device="cuda"))


def test_nemotron_h_latent_cuda_graph():
    """CUDA graph capture + replay for Nemotron-H (with latent) pattern."""
    hidden_dim, intermediate_dim, latent_dim = 128, 256, 64
    cuda_stream_manager.add_device(torch.cuda.current_device())

    model = MockNemotronHLatentMoELayer(hidden_dim, intermediate_dim, latent_dim).eval().to("cuda")
    example = torch.randn(4, hidden_dim, device="cuda")
    gm = _build_gm(model, example)
    gm, num = _execute_shared_expert_in_aux_stream(gm, _MOE_OPS)
    assert num == 1

    _assert_cuda_graph_correctness(gm, model, torch.randn(4, hidden_dim, device="cuda"))


def test_nemotron_h_latent_multi_layer():
    """Two stacked Nemotron-H (with latent) layers — both should be transformed."""
    hidden_dim, intermediate_dim, latent_dim = 128, 256, 64
    cuda_stream_manager.add_device(torch.cuda.current_device())

    model = (
        nn.Sequential(
            MockNemotronHLatentMoELayer(hidden_dim, intermediate_dim, latent_dim),
            MockNemotronHLatentMoELayer(hidden_dim, intermediate_dim, latent_dim),
        )
        .eval()
        .to("cuda")
    )
    example = torch.randn(4, hidden_dim, device="cuda")
    gm = _build_gm(model, example)

    gm, num = _execute_shared_expert_in_aux_stream(gm, _MOE_OPS)

    assert num == 2, f"Expected 2 replacements, got {num}"
    _assert_numerical_correctness(gm, model, torch.randn(4, hidden_dim, device="cuda"))
