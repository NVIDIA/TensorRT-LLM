# load_moe.py
import os

import torch
from torch.utils.cpp_extension import load

from tensorrt_llm._utils import nvtx_range

# Recommended so NVCC generates code for your GPUs. Example: Ada/Hopper.
# You can also set this in your shell env.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6;8.9;9.0")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(THIS_DIR, "_moe_build")
try:
    os.makedirs(BUILD_DIR, exist_ok=True)
except PermissionError:
    import tempfile

    BUILD_DIR = os.path.join(tempfile.gettempdir(), "_moe_build")
    os.makedirs(BUILD_DIR, exist_ok=True)

moe_ext = load(
    name="moe_cuda_jit",
    sources=[
        os.path.join(THIS_DIR, "moe_binding.cpp"),
        os.path.join(THIS_DIR, "moe_op.cu"),
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        # (Optional) show register usage and launch config while debugging:
        # "-Xptxas=-v",
        # (Optional) target a specific SM if needed:
        # "-gencode=arch=compute_90,code=sm_90",
    ],
    extra_ldflags=[
        "-lcublasLt",
        "-lcublas",
    ],
    verbose=True,
    with_cuda=True,
    build_directory=BUILD_DIR,
    is_python_module=True,  # exposes a Python-importable module object
)


@torch.library.custom_op("auto_deploy::cuda_moe", mutates_args=())
def cuda_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    mlp_style: str = "mlp",
    act_fn: str = "relu2",
) -> torch.Tensor:
    """
    CUDA-accelerated MoE forward pass.

    Args:
        x: Input tensor [B, H]
        selected_experts: Expert indices [B, S] (will be converted to int32 if needed)
        routing_weights: Router weights [B, S]
        w1_weight: Stacked w1 weights [num_experts, intermediate_dim, hidden_dim]
        w2_weight: Stacked w2 weights [num_experts, hidden_dim, intermediate_dim]
        mlp_style: MLP style (currently only "mlp" supported)
        act_fn: Activation function (currently only "relu2" supported)

    Returns:
        Output tensor [B, H]
    """
    # Ensure selected_experts is int32 for CUDA kernel compatibility
    if selected_experts.dtype != torch.int32:
        selected_experts = selected_experts.to(torch.int32)

    # Convert routing_weights to float32 to match Triton implementation
    routing_weights = routing_weights.to(torch.float32)

    with nvtx_range("cuda_moe"):
        output = moe_ext.moe_forward(x, selected_experts, routing_weights, w1_weight, w2_weight)
    return output


@cuda_moe.register_fake
def cuda_moe_fake(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    mlp_style: str = "mlp",
    act_fn: str = "relu2",
) -> torch.Tensor:
    return torch.empty_like(x)


# # Quick smoke test
# def _smoke():
#     device = "cuda"
#     dtype = torch.bfloat16
#     B, H, I, N, S = 64, 512, 384, 8, 2

#     x = torch.randn(B, H, device=device, dtype=dtype)
#     sel = torch.randint(0, N, (B, S), device=device, dtype=torch.int32)
#     rw  = torch.rand(B, S, device=device, dtype=dtype)
#     rw  = rw / rw.sum(dim=1, keepdim=True)

#     w1 = [torch.randn(I, H, device=device, dtype=dtype) for _ in range(N)]
#     w2 = [torch.randn(H, I, device=device, dtype=dtype) for _ in range(N)]

#     y = moe_ext.moe_forward(x, sel, rw, w1, w2)
#     print("Output:", y.shape, y.dtype, y.device)

# if __name__ == "__main__":
#     _smoke()
