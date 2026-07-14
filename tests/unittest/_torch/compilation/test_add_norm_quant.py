import pytest
import torch

from tensorrt_llm._torch.compilation.backend import Backend
from tensorrt_llm._torch.custom_ops import flashinfer_rmsnorm


@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("enable_inductor", [False, True])
def test_add_norm_quant_fusion(dtype, enable_inductor):
    backend = Backend(enable_inductor)
    SEQ_LEN = 16
    HIDDEN_SIZE = 1024
    eps = 1e-6

    torch.manual_seed(42)
    x = torch.randn(SEQ_LEN, HIDDEN_SIZE, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    norm_weight = torch.randn((HIDDEN_SIZE,), dtype=dtype, device="cuda")
    scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    # Compiled function that is being tested
    @torch.compile(backend=backend)
    def func(
        x: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        scale: torch.Tensor,
        eps: float,
    ):
        inter_out = x + residual
        normed = flashinfer_rmsnorm(inter_out, norm_weight, eps)
        fp8_out, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(normed, scale)
        return fp8_out, inter_out

    # Reference unfused path
    def ref_func(
        x: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        scale: torch.Tensor,
        eps: float,
    ):
        inter_out = x + residual
        normed = flashinfer_rmsnorm(inter_out, norm_weight, eps)
        fp8_out, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(normed, scale)
        return fp8_out, inter_out

    ref_fp8, ref_inter = ref_func(x.clone(), residual.clone(), norm_weight, scale, eps)
    actual_fp8, actual_inter = func(x.clone(), residual.clone(), norm_weight, scale, eps)

    add_norm_quant_fusion_pass_id = 0
    assert backend.match_count[add_norm_quant_fusion_pass_id] == 1, "Pattern Matching Failed"
    torch.testing.assert_close(
        actual_inter,
        ref_inter,
        rtol=0.05,
        atol=0.15,
    )
    # fused kernel applies norm and quant in a single pass with different rounding than
    # the two-step reference, yielding up to 1 fp8 ULP difference (~0.5 in float)
    torch.testing.assert_close(
        actual_fp8.to(dtype),
        ref_fp8.to(dtype),
        rtol=0.125,
        atol=0.5,
    )


if __name__ == "__main__":
    test_add_norm_quant_fusion(torch.bfloat16, True)
