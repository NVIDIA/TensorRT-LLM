"""
Module-level test for GptOssGate (router) module.

Compares the raw logits produced by:
  - HuggingFace GptOssTopKRouter: F.linear(hidden_states, weight, bias)
  - TRT-LLM GptOssGate: torch.ops.trtllm.cublas_mm(hidden_states, weight.t(), bias=bias)

Both should produce identical logits given the same weights and inputs.
"""
import sys
import os
import pytest
import torch
import torch.nn.functional as F

# Ensure the repo root is on the path so we can import tensorrt_llm
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

CHECKPOINT_DIR = "/scratch.trt_llm_data/llm-models/gpt_oss/gpt-oss-20b"
SHARD_FILE = os.path.join(CHECKPOINT_DIR,
                           "model-00000-of-00002.safetensors")

# ---------------------------------------------------------------------------
# HuggingFace reference router (logits-only, no routing)
# ---------------------------------------------------------------------------

class HFGptOssTopKRouter(torch.nn.Module):
    """Minimal reproduction of HF GptOssTopKRouter -- logits only."""

    def __init__(self, num_experts: int, hidden_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.weight = torch.nn.Parameter(
            torch.empty(num_experts, hidden_dim))
        self.bias = torch.nn.Parameter(torch.empty(num_experts))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        return F.linear(hidden_states, self.weight, self.bias)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def router_weights():
    """Load layer-0 router weight and bias from the HF checkpoint."""
    from safetensors.torch import load_file
    tensors = load_file(SHARD_FILE)
    weight = tensors["model.layers.0.mlp.router.weight"]   # [32, 2880]
    bias = tensors["model.layers.0.mlp.router.bias"]       # [32]
    return weight, bias


@pytest.fixture(scope="module")
def hf_router(router_weights):
    """Create HF router on CUDA with real weights."""
    weight, bias = router_weights
    num_experts, hidden_dim = weight.shape
    router = HFGptOssTopKRouter(num_experts, hidden_dim)
    router.weight.data.copy_(weight)
    router.bias.data.copy_(bias)
    return router.to("cuda", dtype=torch.bfloat16)


@pytest.fixture(scope="module")
def trt_gate(router_weights):
    """Create TRT-LLM GptOssGate on CUDA with real weights."""
    from tensorrt_llm._torch.models.modeling_gpt_oss import GptOssGate
    from tensorrt_llm._torch.modules.fused_moe import CutlassFusedMoE
    weight, bias = router_weights
    num_experts, hidden_dim = weight.shape

    gate = GptOssGate(
        hidden_size=hidden_dim,
        num_experts=num_experts,
        top_k=4,
        dtype=torch.bfloat16,
        apply_routing=False,
        moe_backend_cls=CutlassFusedMoE,
    )
    gate.load_weights(weights=[{"weight": weight, "bias": bias}])
    return gate.to("cuda")


@pytest.fixture(scope="module")
def random_input():
    """Random bfloat16 input of shape [32, 2880]."""
    torch.manual_seed(42)
    return torch.randn(32, 2880, dtype=torch.bfloat16, device="cuda")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGptOssGateLogits:
    """Compare raw logits between HF router and TRT-LLM gate."""

    def test_weight_shapes(self, hf_router, trt_gate):
        """Verify both modules have the expected weight shapes."""
        assert hf_router.weight.shape == (32, 2880), (
            f"HF weight shape mismatch: {hf_router.weight.shape}")
        assert trt_gate.weight.shape == (32, 2880), (
            f"TRT weight shape mismatch: {trt_gate.weight.shape}")
        assert hf_router.bias.shape == (32,), (
            f"HF bias shape mismatch: {hf_router.bias.shape}")
        assert trt_gate.bias.shape == (32,), (
            f"TRT bias shape mismatch: {trt_gate.bias.shape}")

    def test_weights_match(self, hf_router, trt_gate):
        """Verify the loaded weights are identical."""
        assert torch.equal(hf_router.weight.data, trt_gate.weight.data), (
            "Weight data mismatch between HF and TRT-LLM modules")
        assert torch.equal(hf_router.bias.data, trt_gate.bias.data), (
            "Bias data mismatch between HF and TRT-LLM modules")

    def test_logits_close(self, hf_router, trt_gate, random_input):
        """Core test: both produce the same logits for the same input."""
        with torch.no_grad():
            hf_logits = hf_router(random_input)       # F.linear
            trt_logits = trt_gate(random_input)        # cublas_mm

        if not torch.allclose(hf_logits, trt_logits, rtol=1e-3, atol=1e-3):
            diff = (hf_logits - trt_logits).abs()
            raise AssertionError(
                f"Logits mismatch!\n"
                f"  hf_logits  shape={hf_logits.shape} dtype={hf_logits.dtype}\n"
                f"  trt_logits shape={trt_logits.shape} dtype={trt_logits.dtype}\n"
                f"  max_abs_diff={diff.max().item():.6e}\n"
                f"  mean_abs_diff={diff.mean().item():.6e}\n"
                f"  hf  sample={hf_logits[0, :5].tolist()}\n"
                f"  trt sample={trt_logits[0, :5].tolist()}"
            )

    def test_logits_dtype(self, trt_gate, random_input):
        """TRT-LLM gate should output bfloat16 (matching its out_dtype)."""
        with torch.no_grad():
            logits = trt_gate(random_input)
        assert logits.dtype == torch.bfloat16, (
            f"Expected bfloat16 output, got {logits.dtype}")

    def test_routing_method_property(self, trt_gate):
        """routing_method should return RenormalizeMoeRoutingMethod(top_k=4)."""
        from tensorrt_llm._torch.modules.fused_moe import (
            RenormalizeMoeRoutingMethod)
        rm = trt_gate.routing_method
        assert isinstance(rm, RenormalizeMoeRoutingMethod), (
            f"Expected RenormalizeMoeRoutingMethod, got {type(rm)}")
        assert rm.top_k == 4, f"Expected top_k=4, got {rm.top_k}"


class TestGptOssGateLoadWeights:
    """Test load_weights edge cases."""

    def test_load_weight_only(self, router_weights):
        """load_weights with allow_partial_loading=True, bias=None."""
        from tensorrt_llm._torch.models.modeling_gpt_oss import GptOssGate
        from tensorrt_llm._torch.modules.fused_moe import CutlassFusedMoE
        weight, bias = router_weights
        num_experts, hidden_dim = weight.shape

        gate = GptOssGate(
            hidden_size=hidden_dim,
            num_experts=num_experts,
            top_k=4,
            dtype=torch.bfloat16,
            moe_backend_cls=CutlassFusedMoE,
        )
        # Only weight, no bias -- should work with allow_partial_loading
        gate.load_weights(
            weights=[{"weight": weight}],
            allow_partial_loading=True,
        )
        assert torch.equal(gate.weight.data, weight)

    def test_load_missing_weight_raises(self, router_weights):
        """load_weights without weight and allow_partial_loading=False
        should raise AssertionError."""
        from tensorrt_llm._torch.models.modeling_gpt_oss import GptOssGate
        from tensorrt_llm._torch.modules.fused_moe import CutlassFusedMoE
        weight, bias = router_weights
        num_experts, hidden_dim = weight.shape

        gate = GptOssGate(
            hidden_size=hidden_dim,
            num_experts=num_experts,
            top_k=4,
            dtype=torch.bfloat16,
            moe_backend_cls=CutlassFusedMoE,
        )
        with pytest.raises(AssertionError):
            gate.load_weights(weights=[{"bias": bias}])
