import pytest
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from _graph_test_helpers import run_test_transformed_gm
from _model_test_utils import MoEOpModel
from _torch_test_utils import fp4_compatible, fp8_compatible, trtllm_ops_available
from utils.util import skip_pre_hopper

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
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


class FP8MoEModuleForInputScaleTest(nn.Module):
    """Module wrapping torch_quant_fp8_moe for testing FP8 MoE input scale handling."""

    def __init__(
        self,
        num_experts,
        w1_weight,
        w2_weight,
        w1_input_scale,
        w2_input_scale,
        w1_weight_scale,
        w2_weight_scale,
        is_gated_mlp,
        act_fn,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.is_gated_mlp = is_gated_mlp
        self.act_fn = act_fn

        for i in range(num_experts):
            self.register_buffer(f"w1_{i}", w1_weight[i])
            self.register_buffer(f"w2_{i}", w2_weight[i])
            self.register_buffer(f"w1_iscale_{i}", w1_input_scale[i])
            self.register_buffer(f"w2_iscale_{i}", w2_input_scale[i])
            self.register_buffer(f"w1_wscale_{i}", w1_weight_scale[i])
            self.register_buffer(f"w2_wscale_{i}", w2_weight_scale[i])

    def forward(self, x, selected_experts, routing_weights):
        return torch.ops.auto_deploy.torch_quant_fp8_moe(
            x,
            selected_experts,
            routing_weights,
            [getattr(self, f"w1_{i}") for i in range(self.num_experts)],
            [getattr(self, f"w2_{i}") for i in range(self.num_experts)],
            [],  # w3 is empty for non-gated MLP
            [getattr(self, f"w1_iscale_{i}") for i in range(self.num_experts)],
            [getattr(self, f"w2_iscale_{i}") for i in range(self.num_experts)],
            [],  # w3 input scale is empty for non-gated MLP
            [getattr(self, f"w1_wscale_{i}") for i in range(self.num_experts)],
            [getattr(self, f"w2_wscale_{i}") for i in range(self.num_experts)],
            [],  # w3 weight scale is empty for non-gated MLP
            is_gated_mlp=self.is_gated_mlp,
            act_fn=self.act_fn,
        )


@skip_pre_hopper
@pytest.mark.parametrize("backend", ["trtllm", "triton"])
@pytest.mark.parametrize("allow_different_input_scales", [False, True])
@pytest.mark.parametrize("scales_identical", [True, False])
@pytest.mark.skipif(
    not fp8_compatible() or not trtllm_ops_available(),
    reason="Requires fp8 and trtllm support",
)
def test_fp8_moe_different_input_scales(backend, allow_different_input_scales, scales_identical):
    """
    Test FP8 MoE behavior with different/identical input scales via InferenceOptimizer.

    Tests the allow_different_input_scales config option for both trtllm and triton backends:
    - When scales_identical=True: should always work
    - When scales_identical=False and allow_different_input_scales=False: should fail with assertion
    - When scales_identical=False and allow_different_input_scales=True: should work (uses max)
    """
    from tensorrt_llm._torch.auto_deploy.transform.library.fused_moe import _stack_fp8_moe_weights

    torch.manual_seed(0)

    batch_size, num_experts, top_k = 4, 2, 2
    hidden_size, intermediate_size = 128, 128
    # Use non-gated MLP (Relu2) because triton backend only supports non-gated MLP
    is_gated_mlp = False
    act_fn = ActivationType.Relu2

    # Generate test data
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device="cuda") * 0.5

    # Simple routing: distribute tokens across experts
    selected_experts = torch.zeros((batch_size, top_k), dtype=torch.int64, device="cuda")
    for i in range(batch_size):
        selected_experts[i, 0] = i % num_experts
        selected_experts[i, 1] = (i + 1) % num_experts
    routing_weights = torch.ones((batch_size, top_k), device="cuda", dtype=torch.float32) / top_k

    # Create per-expert weights and scales (non-gated MLP, no w3)
    w1_weight, w2_weight = [], []
    w1_input_scale, w2_input_scale = [], []
    w1_weight_scale, w2_weight_scale = [], []

    for expert_id in range(num_experts):
        # Random FP8 weights
        w1_fp8 = torch.randn(intermediate_size, hidden_size, device="cuda").to(torch.float8_e4m3fn)
        w2_fp8 = torch.randn(hidden_size, intermediate_size, device="cuda").to(torch.float8_e4m3fn)
        w1_weight.append(w1_fp8)
        w2_weight.append(w2_fp8)

        # Random weight scales (shape [1])
        w1_weight_scale.append(torch.tensor([0.1], dtype=torch.float32, device="cuda"))
        w2_weight_scale.append(torch.tensor([0.1], dtype=torch.float32, device="cuda"))

        # Input scales: either identical or different per expert (shape [1])
        if scales_identical:
            inp_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        else:
            # Different input scales per expert - big variance to test max() behavior
            inp_scale = torch.tensor([0.5 + 0.5 * expert_id], dtype=torch.float32, device="cuda")

        w1_input_scale.append(inp_scale)
        w2_input_scale.append(inp_scale)

    # Create a module with the FP8 MoE op
    module = FP8MoEModuleForInputScaleTest(
        num_experts,
        w1_weight,
        w2_weight,
        w1_input_scale,
        w2_input_scale,
        w1_weight_scale,
        w2_weight_scale,
        is_gated_mlp,
        act_fn,
    ).cuda()
    gm = fx.symbolic_trace(module)

    # Compute reference output from original graph before transformation
    with torch.inference_mode():
        ref_output = gm(x, selected_experts, routing_weights)

    # Expected behavior:
    # - scales_identical=True: always works
    # - scales_identical=False, allow_different_input_scales=False: assertion error
    # - scales_identical=False, allow_different_input_scales=True: works with max()

    if not scales_identical and not allow_different_input_scales:
        # Should fail with assertion error
        with pytest.raises(AssertionError, match="input scales should have the same value"):
            _stack_fp8_moe_weights(
                gm, backend=backend, allow_different_input_scales=allow_different_input_scales
            )
    else:
        # Should succeed
        num_transformed = _stack_fp8_moe_weights(
            gm, backend=backend, allow_different_input_scales=allow_different_input_scales
        )
        gm.recompile()

        assert num_transformed == 1, f"Expected 1 transform, got {num_transformed}"

        # Verify that max() is used when scales differ
        if not scales_identical:
            expected_max_w1_input_scale = torch.stack(w1_input_scale).max()
            # Attribute name differs between backends
            if backend == "trtllm":
                actual_w1_input = getattr(gm, "quant_moe_fc1_act_scale_0")
            else:  # triton
                actual_w1_input = getattr(gm, "quant_moe_w1_input_scale_0").squeeze()

            assert torch.allclose(actual_w1_input, expected_max_w1_input_scale), (
                f"w1 input scale max mismatch. Got {actual_w1_input}, expected {expected_max_w1_input_scale}"
            )

        # Run the transformed graph and compare to reference output
        with torch.inference_mode():
            output = gm(x, selected_experts, routing_weights)
            assert output.shape == ref_output.shape, (
                f"Output shape mismatch: {output.shape} vs {ref_output.shape}"
            )

            assert torch.allclose(output, ref_output, rtol=0.05, atol=0.05), (
                f"Output mismatch. rtol=0.05, atol=0.05. Max diff: {(output - ref_output).abs().max()}"
            )


class NVFP4MoEModuleForInputScaleTest(nn.Module):
    """Module wrapping torch_quant_nvfp4_moe for testing NVFP4 MoE input scale handling."""

    def __init__(
        self,
        num_experts,
        w1_weight,
        w2_weight,
        w3_weight,
        w1_input_scale,
        w2_input_scale,
        w3_input_scale,
        w1_weight_scale,
        w2_weight_scale,
        w3_weight_scale,
        w1_alpha,
        w2_alpha,
        w3_alpha,
        is_gated_mlp,
        act_fn,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.is_gated_mlp = is_gated_mlp
        self.act_fn = act_fn

        for i in range(num_experts):
            self.register_buffer(f"w1_{i}", w1_weight[i])
            self.register_buffer(f"w2_{i}", w2_weight[i])
            self.register_buffer(f"w1_iscale_{i}", w1_input_scale[i])
            self.register_buffer(f"w2_iscale_{i}", w2_input_scale[i])
            self.register_buffer(f"w1_wscale_{i}", w1_weight_scale[i])
            self.register_buffer(f"w2_wscale_{i}", w2_weight_scale[i])
            self.register_buffer(f"w1_alpha_{i}", w1_alpha[i])
            self.register_buffer(f"w2_alpha_{i}", w2_alpha[i])
            if is_gated_mlp:
                self.register_buffer(f"w3_{i}", w3_weight[i])
                self.register_buffer(f"w3_iscale_{i}", w3_input_scale[i])
                self.register_buffer(f"w3_wscale_{i}", w3_weight_scale[i])
                self.register_buffer(f"w3_alpha_{i}", w3_alpha[i])

    def forward(self, x, selected_experts, routing_weights):
        return torch.ops.auto_deploy.torch_quant_nvfp4_moe(
            x,
            selected_experts,
            routing_weights,
            [getattr(self, f"w1_{i}") for i in range(self.num_experts)],
            [getattr(self, f"w2_{i}") for i in range(self.num_experts)],
            [getattr(self, f"w3_{i}") for i in range(self.num_experts)]
            if self.is_gated_mlp
            else [],
            [getattr(self, f"w1_iscale_{i}") for i in range(self.num_experts)],
            [getattr(self, f"w2_iscale_{i}") for i in range(self.num_experts)],
            [getattr(self, f"w3_iscale_{i}") for i in range(self.num_experts)]
            if self.is_gated_mlp
            else [],
            [getattr(self, f"w1_wscale_{i}") for i in range(self.num_experts)],
            [getattr(self, f"w2_wscale_{i}") for i in range(self.num_experts)],
            [getattr(self, f"w3_wscale_{i}") for i in range(self.num_experts)]
            if self.is_gated_mlp
            else [],
            [getattr(self, f"w1_alpha_{i}") for i in range(self.num_experts)],
            [getattr(self, f"w2_alpha_{i}") for i in range(self.num_experts)],
            [getattr(self, f"w3_alpha_{i}") for i in range(self.num_experts)]
            if self.is_gated_mlp
            else [],
            is_gated_mlp=self.is_gated_mlp,
            act_fn=self.act_fn,
        )


@skip_pre_hopper
@pytest.mark.parametrize("allow_different_input_scales", [False, True])
@pytest.mark.parametrize("scales_identical", [True, False])
@pytest.mark.parametrize("is_gated_mlp", [False, True])
@pytest.mark.skipif(
    not trtllm_ops_available(),
    reason="Requires trtllm ops",
)
def test_nvfp4_moe_different_input_scales(
    allow_different_input_scales, scales_identical, is_gated_mlp
):
    """
    Test NVFP4 MoE behavior with different/identical input scales via _stack_nvfp4_moe_weights.

    Tests the allow_different_input_scales config option for both gated and non-gated MLP:
    - When scales_identical=True: should always work
    - When scales_identical=False and allow_different_input_scales=False: should fail with assertion
    - When scales_identical=False and allow_different_input_scales=True: should work (uses min)

    Note: NVFP4 uses min() (not max() like FP8) because scales are in kernel format (2688/amax):
    smaller scale = larger amax = larger dynamic range.

    This test uses mock tensors to test the transform logic without running the actual NVFP4 kernel.
    """
    from tensorrt_llm._torch.auto_deploy.transform.library.fused_moe import _stack_nvfp4_moe_weights

    torch.manual_seed(0)

    num_experts = 2
    hidden_size, intermediate_size = 128, 128
    act_fn = ActivationType.Silu if is_gated_mlp else ActivationType.Relu2

    # NVFP4 constants
    FP4_GLOBAL_SCALE_MAX = 448 * 6  # 2688
    NVFP4_BLOCK_SIZE = 16

    # Create per-expert mock weights and scales
    # We use mock tensors with correct shapes to test the transform logic
    # without needing actual FP4 quantization (which requires SM>=100)
    w1_weight, w2_weight, w3_weight = [], [], []
    w1_input_scale, w2_input_scale, w3_input_scale = [], [], []
    w1_weight_scale, w2_weight_scale, w3_weight_scale = [], [], []
    w1_alpha, w2_alpha, w3_alpha = [], [], []

    for expert_id in range(num_experts):
        # Mock FP4 weights (uint8 packed, half the size in last dim)
        w1_fp4 = torch.randint(
            0, 255, (intermediate_size, hidden_size // 2), dtype=torch.uint8, device="cuda"
        )
        w2_fp4 = torch.randint(
            0, 255, (hidden_size, intermediate_size // 2), dtype=torch.uint8, device="cuda"
        )

        # Mock block scales (2D): shape (m, n // NVFP4_BLOCK_SIZE)
        # With 128x128 dims, no padding needed (already multiples of 128 and 8)
        w1_block_scale = torch.randn(
            intermediate_size, hidden_size // NVFP4_BLOCK_SIZE, dtype=torch.float32, device="cuda"
        ).to(torch.float8_e4m3fn)
        w2_block_scale = torch.randn(
            hidden_size, intermediate_size // NVFP4_BLOCK_SIZE, dtype=torch.float32, device="cuda"
        ).to(torch.float8_e4m3fn)

        w1_weight.append(w1_fp4)
        w2_weight.append(w2_fp4)
        w1_weight_scale.append(w1_block_scale)
        w2_weight_scale.append(w2_block_scale)

        # Input scales: either identical or different per expert
        # For NVFP4, scale = FP4_GLOBAL_SCALE_MAX / amax
        if scales_identical:
            # Same amax for all experts -> same input scale
            inp_scale = torch.tensor(FP4_GLOBAL_SCALE_MAX / 1.0, dtype=torch.float32, device="cuda")
        else:
            # Different amax per expert -> different input scales
            # Expert 0: amax=1.0, scale=2688/1.0=2688
            # Expert 1: amax=2.0, scale=2688/2.0=1344
            amax = 1.0 + expert_id
            inp_scale = torch.tensor(
                FP4_GLOBAL_SCALE_MAX / amax, dtype=torch.float32, device="cuda"
            )

        w1_input_scale.append(inp_scale)
        w2_input_scale.append(inp_scale)

        # Mock weight_scale_2 (global scale for this expert's weights)
        w1_scale_2 = torch.tensor(100.0, dtype=torch.float32, device="cuda")
        w2_scale_2 = torch.tensor(100.0, dtype=torch.float32, device="cuda")

        # Alpha = 1 / (input_scale * weight_scale_2)
        w1_alpha.append((1.0 / (inp_scale * w1_scale_2)).to(torch.float32))
        w2_alpha.append((1.0 / (inp_scale * w2_scale_2)).to(torch.float32))

        # For gated MLP, create w3 weights/scales/alpha (same shape as w1)
        if is_gated_mlp:
            w3_fp4 = torch.randint(
                0, 255, (intermediate_size, hidden_size // 2), dtype=torch.uint8, device="cuda"
            )
            w3_block_scale = torch.randn(
                intermediate_size,
                hidden_size // NVFP4_BLOCK_SIZE,
                dtype=torch.float32,
                device="cuda",
            ).to(torch.float8_e4m3fn)
            w3_weight.append(w3_fp4)
            w3_weight_scale.append(w3_block_scale)
            # w3 uses the same input scale as w1 (they process the same input)
            w3_input_scale.append(inp_scale)
            w3_scale_2 = torch.tensor(100.0, dtype=torch.float32, device="cuda")
            w3_alpha.append((1.0 / (inp_scale * w3_scale_2)).to(torch.float32))

    # Create a module with the NVFP4 MoE op
    module = NVFP4MoEModuleForInputScaleTest(
        num_experts,
        w1_weight,
        w2_weight,
        w3_weight,
        w1_input_scale,
        w2_input_scale,
        w3_input_scale,
        w1_weight_scale,
        w2_weight_scale,
        w3_weight_scale,
        w1_alpha,
        w2_alpha,
        w3_alpha,
        is_gated_mlp,
        act_fn,
    ).cuda()
    gm = fx.symbolic_trace(module)

    # Expected behavior:
    # - scales_identical=True: always works
    # - scales_identical=False, allow_different_input_scales=False: assertion error
    # - scales_identical=False, allow_different_input_scales=True: works with min()

    if not scales_identical and not allow_different_input_scales:
        # Should fail with assertion error
        with pytest.raises(AssertionError, match="FC1 input scales differ"):
            _stack_nvfp4_moe_weights(gm, allow_different_input_scales=allow_different_input_scales)
    else:
        # Should succeed
        num_transformed = _stack_nvfp4_moe_weights(
            gm, allow_different_input_scales=allow_different_input_scales
        )
        gm.recompile()

        assert num_transformed == 1, f"Expected 1 transform, got {num_transformed}"

        # Verify that min() is used when scales differ
        if not scales_identical:
            # For gated MLP, global scale = min(w1_scales.min(), w3_scales.min())
            # For non-gated MLP, global scale = w1_scales.min()
            if is_gated_mlp:
                expected_min_input_scale = torch.minimum(
                    torch.stack(w1_input_scale).min(),
                    torch.stack(w3_input_scale).min(),
                )
            else:
                expected_min_input_scale = torch.stack(w1_input_scale).min()

            actual_input_scale = getattr(gm, "nvfp4_moe_w3_w1_input_scale_stacked_0")

            assert torch.allclose(actual_input_scale, expected_min_input_scale), (
                f"FC1 input scale min mismatch. Got {actual_input_scale}, expected {expected_min_input_scale}"
            )

            # Verify alpha was recomputed correctly
            # new_alpha = old_alpha * per_expert_input_scale / global_input_scale
            expected_alpha = (
                torch.stack(w1_alpha) * torch.stack(w1_input_scale) / expected_min_input_scale
            )
            actual_alpha = getattr(gm, "nvfp4_moe_w1_alpha_stacked_0")
            assert torch.allclose(actual_alpha, expected_alpha, rtol=1e-5, atol=1e-5), (
                f"Alpha recomputation mismatch. Got {actual_alpha}, expected {expected_alpha}"
            )
