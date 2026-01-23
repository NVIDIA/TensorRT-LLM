import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _graph_test_helpers import run_test_transformed_gm
from _model_test_utils import MoEOpModel
from _torch_test_utils import fp4_compatible, fp8_compatible, trtllm_ops_available

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale
from tensorrt_llm._torch.utils import ActivationType


class BlockSparseTop2MLP(nn.Module):
    def __init__(self, ffn_dim, hidden_dim):
        super().__init__()
        self.ffn_dim = ffn_dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = F.silu

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class BlockSparseTop2MLPFP8(nn.Module):
    def __init__(self, ffn_dim, hidden_dim, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.ffn_dim = ffn_dim
        self.hidden_dim = hidden_dim
        # Input scale fixed to 1.0
        self.register_buffer("inp_scale", torch.tensor(1.0, dtype=torch.float, device=device))
        # FP8 weight scale factor depends on dtype
        wt_factor = 448 if dtype == torch.bfloat16 else 432

        w1_fp32 = torch.randn(ffn_dim, hidden_dim, device=device)
        w3_fp32 = torch.randn(ffn_dim, hidden_dim, device=device)
        w2_fp32 = torch.randn(hidden_dim, ffn_dim, device=device)
        w1_scale = (w1_fp32.abs().max() / wt_factor).float().to(device)
        w3_scale = (w3_fp32.abs().max() / wt_factor).float().to(device)
        w2_scale = (w2_fp32.abs().max() / wt_factor).float().to(device)

        self.register_buffer("w1_scale", w1_scale)
        self.register_buffer("w3_scale", w3_scale)
        self.register_buffer("w2_scale", w2_scale)

        w1_fp8 = (w1_fp32 / w1_scale).to(torch.float8_e4m3fn)
        w3_fp8 = (w3_fp32 / w3_scale).to(torch.float8_e4m3fn)
        w2_fp8 = (w2_fp32 / w2_scale).to(torch.float8_e4m3fn)
        self.register_parameter("w1_fp8", nn.Parameter(w1_fp8))
        self.register_parameter("w3_fp8", nn.Parameter(w3_fp8))
        self.register_parameter("w2_fp8", nn.Parameter(w2_fp8))
        self.act_fn = F.silu

    def forward(self, hidden_states: torch.Tensor):
        x = hidden_states
        w1_out = torch.ops.auto_deploy.torch_quant_fp8_linear(
            x,
            self.w1_fp8,
            bias=None,
            input_scale=self.inp_scale,
            weight_scale=self.w1_scale,
        )
        w3_out = torch.ops.auto_deploy.torch_quant_fp8_linear(
            x,
            self.w3_fp8,
            bias=None,
            input_scale=self.inp_scale,
            weight_scale=self.w3_scale,
        )
        fused = self.act_fn(w1_out) * w3_out
        out = torch.ops.auto_deploy.torch_quant_fp8_linear(
            fused,
            self.w2_fp8,
            bias=None,
            input_scale=self.inp_scale,
            weight_scale=self.w2_scale,
        )
        return out


class BlockSparseTop2MLPFP4(nn.Module):
    def __init__(self, ffn_dim, hidden_dim, input_sample, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.ffn_dim = ffn_dim
        self.hidden_dim = hidden_dim

        # Prepare full-precision weights
        w1_fp32 = torch.randn(ffn_dim, hidden_dim, device=device, dtype=dtype) * 0.01
        w3_fp32 = torch.randn(ffn_dim, hidden_dim, device=device, dtype=dtype) * 0.01
        w2_fp32 = torch.randn(hidden_dim, ffn_dim, device=device, dtype=dtype) * 0.01

        # Compute input scale
        inp_scale = fp4_global_scale(input_sample)

        # Compute per-weight-layer scales (global scale, no per-vector partition here)
        scale_1 = fp4_global_scale(w1_fp32)
        scale_2 = fp4_global_scale(w2_fp32)
        scale_3 = fp4_global_scale(w3_fp32)

        # Quantize weights using fake quant op
        w1_fp4, w1_weight_scale = torch.ops.trtllm.fp4_quantize(w1_fp32, scale_1, 16, False)
        w2_fp4, w2_weight_scale = torch.ops.trtllm.fp4_quantize(w2_fp32, scale_2, 16, False)
        w3_fp4, w3_weight_scale = torch.ops.trtllm.fp4_quantize(w3_fp32, scale_3, 16, False)

        # Compute alpha = 1 / (input_scale * weight_scale)
        alpha_1 = 1.0 / (inp_scale * scale_1)
        alpha_2 = 1.0 / (inp_scale * scale_2)
        alpha_3 = 1.0 / (inp_scale * scale_3)

        # Register all quantized tensors and metadata
        self.register_parameter("w1_fp4", nn.Parameter(w1_fp4, requires_grad=False))
        self.register_parameter("w2_fp4", nn.Parameter(w2_fp4, requires_grad=False))
        self.register_parameter("w3_fp4", nn.Parameter(w3_fp4, requires_grad=False))

        self.register_buffer("input_scale", inp_scale)
        self.register_buffer("w1_weight_scale", w1_weight_scale)
        self.register_buffer("w2_weight_scale", w2_weight_scale)
        self.register_buffer("w3_weight_scale", w3_weight_scale)

        self.register_buffer("w1_alpha", alpha_1)
        self.register_buffer("w2_alpha", alpha_2)
        self.register_buffer("w3_alpha", alpha_3)

        self.act_fn = F.silu

    def forward(self, hidden_states):
        x = hidden_states
        w1_out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
            x,
            self.w1_fp4,
            bias=None,
            input_scale=self.input_scale,
            weight_scale=self.w1_weight_scale,
            alpha=self.w1_alpha,
        )
        w3_out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
            x,
            self.w3_fp4,
            bias=None,
            input_scale=self.input_scale,
            weight_scale=self.w3_weight_scale,
            alpha=self.w3_alpha,
        )
        fused = self.act_fn(w1_out) * w3_out
        out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
            fused,
            self.w2_fp4,
            bias=None,
            input_scale=self.input_scale,
            weight_scale=self.w2_weight_scale,
            alpha=self.w2_alpha,
        )
        return out


def make_mlp_block(
    quant_type: str,
    ffn_dim: int,
    hidden_dim: int,
    input_sample: None,
    dtype=torch.bfloat16,
    device="cuda",
):
    if quant_type == "FP8":
        return BlockSparseTop2MLPFP8(ffn_dim, hidden_dim, dtype=dtype, device=device)
    elif quant_type == "NVFP4":
        return BlockSparseTop2MLPFP4(ffn_dim, hidden_dim, input_sample, dtype=dtype, device=device)
    else:
        return BlockSparseTop2MLP(ffn_dim, hidden_dim)


class BlockSparseMoE(nn.Module):
    def __init__(
        self,
        hidden_size=64,
        num_experts=3,
        intermediate_size=32,
        quant_type="",
        input_sample=None,
        dtype=torch.bfloat16,
        device="cuda",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = 2
        self.gate = nn.Linear(hidden_size, num_experts, bias=False).to(device=device, dtype=dtype)
        self.experts = nn.ModuleList(
            [
                make_mlp_block(
                    quant_type, intermediate_size, hidden_size, input_sample, dtype, device
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


class MoEPatternModel(nn.Module):
    def __init__(self, quant_type: str = ""):
        super().__init__()
        self.embedding = nn.Embedding(1000, 64)
        input_ids = self.get_input(device="cpu")  # or pass as constructor arg
        input_sample = self.embedding(input_ids)
        self.block_sparse_moe = BlockSparseMoE(
            hidden_size=64,
            num_experts=3,
            intermediate_size=32,
            quant_type=quant_type,
            input_sample=input_sample,
        )

    def forward(self, x):
        embedded = F.embedding(x, self.embedding.weight)
        residual = embedded
        hidden_states = self.block_sparse_moe(embedded)
        hidden_states = residual + hidden_states
        return hidden_states

    def get_input(self, device):
        torch.manual_seed(2345)
        return torch.randint(0, 1000, (2, 2), device=device)


@pytest.mark.parametrize(
    "quant_type,expected_op,atol,rtol",
    [
        pytest.param("", torch.ops.auto_deploy.torch_moe, 1e-3, 1e-3, id="simple"),
        pytest.param(
            "FP8",
            torch.ops.auto_deploy.torch_quant_fp8_moe,
            0.05,
            0.01,
            marks=pytest.mark.skipif(not fp8_compatible(), reason="Requires FP8 support"),
            id="fp8",
        ),
        pytest.param(
            "NVFP4",
            torch.ops.auto_deploy.torch_quant_nvfp4_moe,
            0.05,
            0.01,
            marks=[
                pytest.mark.skipif(
                    not fp4_compatible() or not trtllm_ops_available(),
                    reason="Requires FP4 + TRTLLM support",
                ),
                pytest.mark.skip("https://nvbugs/5410946"),
            ],
            id="fp4",
        ),
    ],
)
def test_moe_matching(quant_type, expected_op, atol, rtol):
    with torch.inference_mode():
        device = "cuda"
        torch.manual_seed(2345)
        model = MoEPatternModel(quant_type=quant_type).to(device=device)

        if quant_type == "":
            model = model.to(dtype=torch.bfloat16)
        else:
            model.embedding = model.embedding.to(dtype=torch.bfloat16)
            model.block_sparse_moe.gate = model.block_sparse_moe.gate.to(dtype=torch.bfloat16)

        x = model.get_input(device=device)
        gm = torch_export_to_gm(model, args=(x,), clone=True)
        gm_transformed = InferenceOptimizer(
            None,
            {
                "match_moe_pattern": {
                    "stage": "pattern_matcher",
                },
                "match_fp8_moe_pattern": {
                    "stage": "pattern_matcher",
                },
                "match_nvfp4_moe_pattern": {
                    "stage": "pattern_matcher",
                },
            },
        )(None, gm)

        run_test_transformed_gm(
            model,
            x,
            gm_transformed,
            lambda gm: any(is_op(n, expected_op) for n in gm.graph.nodes),
            lambda num: num,
            atol=atol,
            rtol=rtol,
            test_load_hook=True,
            strict_loading=True,
        )


def test_moe_fusion():
    device = "cuda"
    model = MoEOpModel().to(device=device, dtype=torch.bfloat16)
    x = model.get_input(device=device, dtype=torch.bfloat16)
    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        None,
        {
            "fuse_moe": {
                "stage": "post_load_fusion",
            },
        },
    )(None, gm)

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        lambda gm: any(
            is_op(
                n, {torch.ops.auto_deploy.torch_moe_fused, torch.ops.auto_deploy.trtllm_moe_fused}
            )
            for n in gm.graph.nodes
        ),
        lambda num_p_og: num_p_og,
        atol=0.2,
        rtol=0.5,
        test_load_hook=False,  # state_dict changed after loading hook
        strict_loading=True,
    )

    # expert weights are fused and stacked in fusion
    num_param_nodes = len(list(model.named_parameters()))
    num_param_nodes_fused = len(list(gm_transformed.named_parameters()))
    assert (
        num_param_nodes_fused < num_param_nodes
    ), f"""number of parameter nodes after fusion {num_param_nodes_fused} <
        number of parameter nodes before fusion {num_param_nodes}"""


def test_fuse_moe_cleanup():
    # Ensure deterministic allocations and a clean slate
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.empty_cache()

    device = "cuda"
    dtype = torch.bfloat16

    # Build model and export to GraphModule (pre-fusion)
    model = MoEOpModel().to(device=device, dtype=dtype)
    x = model.get_input(device=device, dtype=dtype)
    gm = torch_export_to_gm(model, args=(x,), clone=True)

    # Count parameters and measure memory before fusion
    num_param_nodes_before = len(list(gm.named_parameters()))
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated()

    # Apply MoE fusion which should stack weights and clean up unstacked params
    # We need to ensure the cleanup is done as part of the transformation to avoid OOM during the transformation itself.
    gm_transformed = InferenceOptimizer(
        None,
        {
            "fuse_moe": {
                "stage": "post_load_fusion",
                "run_graph_cleanup": False,  # verify cleanup is done as part of the transformation
                "run_shape_prop": False,  # shape_prop can also trigger cleanup
            },
        },
    )(None, gm)

    # Ensure that parameter count decreased after fusion (unstacked params cleaned)
    num_param_nodes_after = len(list(gm_transformed.named_parameters()))
    assert num_param_nodes_after < num_param_nodes_before, (
        f"Expected fewer parameters after fusion: before={num_param_nodes_before}, after={num_param_nodes_after}"
    )

    # Memory should not increase after fusion/cleanup
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    mem_after = torch.cuda.memory_allocated()
    assert mem_after <= mem_before, (
        f"CUDA memory increased after fusion: before={mem_before} after={mem_after}"
    )


class MoEOpModelNVFP4(nn.Module):
    """MoE model using torch_quant_nvfp4_moe op for testing fusion to trtllm_quant_nvfp4_moe_fused.

    This model creates weights with 3D block scales that are compatible with
    the trtllm fused MoE kernel.
    """

    def __init__(
        self,
        hidden_size=512,  # Already aligned to all requirements (16, 128, etc.)
        intermediate_size=256,  # Already aligned - no padding needed
        num_experts=3,
        top_k=2,
        dtype=torch.bfloat16,
        is_gated_mlp=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.dtype = dtype
        self.is_gated_mlp = is_gated_mlp

        # Constants for NVFP4 layout
        NVFP4_BLOCK_SIZE = 16
        NVFP4_PACK_FACTOR = 2
        FLOAT8_E4M3_MAX = 448.0
        FLOAT4_E2M1_MAX = 6.0

        self.gate = nn.Linear(hidden_size, num_experts, dtype=dtype)

        # Create sample input for scale computation
        sample_input = torch.randn(2, hidden_size, dtype=dtype, device="cuda") * 0.01
        inp_scale = fp4_global_scale(sample_input)

        # Per-expert quantized weights and scales
        self.w1_weight = nn.ParameterList()
        self.w2_weight = nn.ParameterList()
        self.w3_weight = nn.ParameterList() if is_gated_mlp else None

        for i in range(num_experts):
            w1_fp32 = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype) * 0.01
            w2_fp32 = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=dtype) * 0.01

            # Compute global scales
            w1_amax = torch.abs(w1_fp32).max().to(torch.float32)
            w2_amax = torch.abs(w2_fp32).max().to(torch.float32)

            scale_1 = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
            scale_2 = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax

            # Quantize weights (non-swizzled layout)
            w1_fp4, w1_bs = torch.ops.trtllm.fp4_quantize(w1_fp32, scale_1, NVFP4_BLOCK_SIZE, False)
            w2_fp4, w2_bs = torch.ops.trtllm.fp4_quantize(w2_fp32, scale_2, NVFP4_BLOCK_SIZE, False)

            # fp4_quantize pads block scales but not weights - infer padded dims from block scale size
            _, w1_k_packed = w1_fp4.shape
            _, w2_k_packed = w2_fp4.shape
            w1_k_padded = w1_k_packed * NVFP4_PACK_FACTOR  # Convert from uint8 to FP4 element count
            w2_k_padded = w2_k_packed * NVFP4_PACK_FACTOR

            # Calculate padded N dimension from block scale tensor size
            w1_n_padded = w1_bs.numel() // (w1_k_padded // NVFP4_BLOCK_SIZE)
            w2_n_padded = w2_bs.numel() // (w2_k_padded // NVFP4_BLOCK_SIZE)

            # Reshape block scales to 3D format [N_padded, K/block]
            w1_bs_3d = w1_bs.view(w1_n_padded, w1_k_padded // NVFP4_BLOCK_SIZE)
            w2_bs_3d = w2_bs.view(w2_n_padded, w2_k_padded // NVFP4_BLOCK_SIZE)

            self.w1_weight.append(nn.Parameter(w1_fp4, requires_grad=False))
            self.w2_weight.append(nn.Parameter(w2_fp4, requires_grad=False))

            self.register_buffer(f"w1_input_scale_{i}", inp_scale)
            self.register_buffer(f"w2_input_scale_{i}", inp_scale)
            self.register_buffer(f"w1_weight_scale_{i}", w1_bs_3d.contiguous())
            self.register_buffer(f"w2_weight_scale_{i}", w2_bs_3d.contiguous())
            self.register_buffer(f"w1_alpha_{i}", 1.0 / (inp_scale * scale_1))
            self.register_buffer(f"w2_alpha_{i}", 1.0 / (inp_scale * scale_2))

            if is_gated_mlp:
                w3_fp32 = (
                    torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype) * 0.01
                )
                w3_amax = torch.abs(w3_fp32).max().to(torch.float32)
                scale_3 = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w3_amax
                w3_fp4, w3_bs = torch.ops.trtllm.fp4_quantize(
                    w3_fp32, scale_3, NVFP4_BLOCK_SIZE, False
                )

                # Infer padded dimensions for w3
                _, w3_k_packed = w3_fp4.shape
                w3_k_padded = w3_k_packed * NVFP4_PACK_FACTOR
                w3_n_padded = w3_bs.numel() // (w3_k_padded // NVFP4_BLOCK_SIZE)
                w3_bs_3d = w3_bs.view(w3_n_padded, w3_k_padded // NVFP4_BLOCK_SIZE)

                self.w3_weight.append(nn.Parameter(w3_fp4, requires_grad=False))
                self.register_buffer(f"w3_input_scale_{i}", inp_scale)
                self.register_buffer(f"w3_weight_scale_{i}", w3_bs_3d.contiguous())
                self.register_buffer(f"w3_alpha_{i}", 1.0 / (inp_scale * scale_3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)

        w1_list = list(self.w1_weight)
        w2_list = list(self.w2_weight)
        w3_list = list(self.w3_weight) if self.is_gated_mlp else []

        w1_input_scale = [getattr(self, f"w1_input_scale_{i}") for i in range(self.num_experts)]
        w2_input_scale = [getattr(self, f"w2_input_scale_{i}") for i in range(self.num_experts)]
        w3_input_scale = (
            [getattr(self, f"w3_input_scale_{i}") for i in range(self.num_experts)]
            if self.is_gated_mlp
            else []
        )
        w1_weight_scale = [getattr(self, f"w1_weight_scale_{i}") for i in range(self.num_experts)]
        w2_weight_scale = [getattr(self, f"w2_weight_scale_{i}") for i in range(self.num_experts)]
        w3_weight_scale = (
            [getattr(self, f"w3_weight_scale_{i}") for i in range(self.num_experts)]
            if self.is_gated_mlp
            else []
        )
        w1_alpha = [getattr(self, f"w1_alpha_{i}") for i in range(self.num_experts)]
        w2_alpha = [getattr(self, f"w2_alpha_{i}") for i in range(self.num_experts)]
        w3_alpha = (
            [getattr(self, f"w3_alpha_{i}") for i in range(self.num_experts)]
            if self.is_gated_mlp
            else []
        )

        out = torch.ops.auto_deploy.torch_quant_nvfp4_moe(
            x,
            selected_experts,
            routing_weights,
            w1_list,
            w2_list,
            w3_list,
            w1_input_scale,
            w2_input_scale,
            w3_input_scale,
            w1_weight_scale,
            w2_weight_scale,
            w3_weight_scale,
            w1_alpha,
            w2_alpha,
            w3_alpha,
            is_gated_mlp=self.is_gated_mlp,
            act_fn=ActivationType.Silu if self.is_gated_mlp else ActivationType.Relu2,
        )
        return out

    def get_input(self, device, dtype=torch.bfloat16):
        return torch.randn(2, self.hidden_size, device=device, dtype=dtype) * 0.01


@pytest.mark.skipif(
    not fp4_compatible() or not trtllm_ops_available(),
    reason="Requires FP4 + TRTLLM support",
)
@pytest.mark.parametrize("is_gated_mlp", [True, False], ids=["gated_mlp", "mlp"])
@pytest.mark.parametrize(
    "hidden_size,intermediate_size",
    [
        (512, 256),  # Standard aligned dimensions
        (1024, 512),  # Larger aligned dimensions
        (768, 384),  # Common transformer dimensions (divisible by 16)
        (512, 128),  # Smaller intermediate
        (256, 256),  # Equal dimensions
    ],
    ids=["512x256", "1024x512", "768x384", "512x128", "256x256"],
)
def test_nvfp4_moe_fusion(is_gated_mlp, hidden_size, intermediate_size):
    """Test that torch_quant_nvfp4_moe fuses to trtllm_quant_nvfp4_moe_fused.

    Note: This test uses swizzled block scales that are compatible with the fused trtllm kernel.
    The non-fused op (torch_quant_nvfp4_moe) uses a different internal path that expects
    non-swizzled scales, so we don't compare outputs between non-fused and fused.
    Instead, we verify the fusion transformation works correctly and produces valid output.

    Tests both gated MLP (with w3) and non-gated MLP (without w3) variants with various
    hidden_size and intermediate_size configurations.
    """
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    model = MoEOpModelNVFP4(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        is_gated_mlp=is_gated_mlp,
    ).to(device=device)
    x = model.get_input(device=device, dtype=dtype)

    # Export to GraphModule
    gm = torch_export_to_gm(model, args=(x,), clone=True)

    # Verify non-fused op is present before fusion
    has_nonfused = any(
        is_op(n, torch.ops.auto_deploy.torch_quant_nvfp4_moe) for n in gm.graph.nodes
    )
    assert has_nonfused, "Expected torch_quant_nvfp4_moe op before fusion"

    # Apply NVFP4 MoE fusion
    gm_transformed = InferenceOptimizer(
        None,
        {
            "fuse_nvfp4_moe": {
                "stage": "post_load_fusion",
            },
        },
    )(None, gm)

    # Verify fused op is present after fusion
    has_fused = any(
        is_op(n, torch.ops.auto_deploy.trtllm_quant_nvfp4_moe_fused)
        for n in gm_transformed.graph.nodes
    )
    assert has_fused, "Expected trtllm_quant_nvfp4_moe_fused op after fusion"

    # Run fused graph to verify it produces valid output (not NaN/Inf)
    with torch.inference_mode():
        fused_output = gm_transformed(x)

    assert not torch.isnan(fused_output).any(), "Fused output contains NaN"
    assert not torch.isinf(fused_output).any(), "Fused output contains Inf"
