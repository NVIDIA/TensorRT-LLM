import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.models.patches.nemotron_h import _rms_norm_ref


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA for the custom Triton kernel"
)
@pytest.mark.parametrize("B,T,H,group", [(2, 3, 4096, 512), (1, 1, 4096, 4096), (4, 2, 2048, 256)])
@pytest.mark.parametrize("use_gate", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_custom_op_matches_ref(B, T, H, group, use_gate, dtype):
    torch.manual_seed(0)
    device = torch.device("cuda")

    x = torch.randn(B, T, H, dtype=dtype, device=device)
    z = torch.randn_like(x) if use_gate else None
    w = torch.ones(H, dtype=dtype, device=device)

    y_ref = _rms_norm_ref(
        x, w, bias=None, z=z, eps=1e-5, group_size=group, norm_before_gate=False, upcast=True
    )

    # Custom op (currently returns fp32). Cast it back to x.dtype for apples-to-apples with ref.
    y_op_fp32 = torch.ops.auto_deploy.torch_rmsnorm_gated(x, w, z, 1e-5, group, False)
    y_op = y_op_fp32.to(x.dtype)

    assert y_ref.dtype == x.dtype and y_op.dtype == x.dtype
    torch.testing.assert_close(y_op, y_ref, rtol=1e-3, atol=1e-3)
