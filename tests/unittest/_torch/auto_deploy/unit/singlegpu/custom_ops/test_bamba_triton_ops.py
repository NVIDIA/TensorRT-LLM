import pytest
import torch

# Import the reference torch custom ops from the existing patch module
from tensorrt_llm._torch.auto_deploy.models.patches.bamba import _ssm_transform as ssm_transform_ref
from tensorrt_llm._torch.auto_deploy.models.patches.bamba import (
    _ssm_transform_cached as ssm_transform_cached_ref,
)

# Ensure Triton-backed ops are registered
from tensorrt_llm._torch.custom_ops import auto_deploy_bamba_triton_ops  # noqa: F401


def make_inputs(device, dtype, *, B=2, T=8, H=8, P=64, G=1, N=128, chunk=4):
    inter = H * P
    hs = torch.randn(B, T, inter, device=device, dtype=dtype)
    dt = torch.randn(B, T, H, device=device, dtype=dtype)
    A = -torch.rand(H, device=device, dtype=torch.float32) - 1.0
    Bmat = torch.randn(B, T, G * N, device=device, dtype=dtype)
    Cmat = torch.randn(B, T, G * N, device=device, dtype=dtype)
    D = torch.randn(H, device=device, dtype=torch.float32)
    dt_bias = torch.rand(H, device=device, dtype=torch.float32) - 4.0
    return hs, A, Bmat, Cmat, D, dt, dt_bias, N, [0.0, float("inf")], B, T, P, H, G, chunk


@pytest.mark.parametrize("dtype", [torch.float32])
def test_ssm_transform_triton_vs_ref(dtype):
    device = "cuda"
    hs, A, Bmat, Cmat, D, dt, dt_bias, N, tlim, B, T, P, H, G, chunk = make_inputs(device, dtype)

    # reference torch custom op
    y_ref, state_ref = ssm_transform_ref(
        hs, A, Bmat, Cmat, D, dt, dt_bias, N, tlim, B, T, P, H, G, chunk
    )

    # triton custom op
    y_tri, state_tri = torch.ops.auto_deploy.ssm_transform_triton(
        hs, A, Bmat, Cmat, D, dt, dt_bias, N, tlim, B, T, P, H, G, chunk
    )

    # Combined Triton kernel vs naive reference can differ slightly; use looser tol
    rtol = {torch.float16: 1e-2, torch.float32: 1e-2, torch.bfloat16: 1e-1}[dtype]
    atol = {torch.float16: 2e-2, torch.float32: 3e-2, torch.bfloat16: 1e-1}[dtype]
    torch.testing.assert_close(y_tri, y_ref, rtol=rtol, atol=atol)
    # state_ref in ref path is final state at chunk boundaries (shape [B,H,P,N])
    torch.testing.assert_close(state_tri, state_ref, rtol=1e-4, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32])
def test_ssm_transform_cached_triton_vs_ref(dtype):
    device = "cuda"
    B, H, P, N = 2, 8, 64, 128
    G = 1
    inter = H * P
    hs = torch.randn(B, inter, device=device, dtype=dtype)
    dt = torch.randn(B, 1, H, device=device, dtype=dtype)
    A = -torch.rand(H, device=device, dtype=torch.float32) - 1.0
    Bmat = torch.randn(B, G * N, device=device, dtype=dtype)
    Cmat = torch.randn(B, G * N, device=device, dtype=dtype)
    D = torch.randn(H, device=device, dtype=torch.float32)
    dt_bias = torch.rand(H, device=device, dtype=torch.float32) - 4.0
    state0 = torch.randn(B, H, P, N, device=device, dtype=dtype)
    tlim = [0.0, float("inf")]

    # reference op
    y_ref, state_ref = ssm_transform_cached_ref(
        hs,
        A,
        Bmat,
        Cmat,
        D,
        dt,
        dt_bias,
        N,
        tlim,
        B,
        1,
        P,
        H,
        G,
        1,
        state0.clone(),
    )

    # triton op
    y_tri, state_tri = torch.ops.auto_deploy.ssm_transform_cached_triton(
        hs,
        A,
        Bmat,
        Cmat,
        D,
        dt,
        dt_bias,
        N,
        tlim,
        B,
        1,
        P,
        H,
        G,
        1,
        state0.clone(),
    )

    rtol = {torch.float16: 1e-2, torch.float32: 1e-2, torch.bfloat16: 1e-1}[dtype]
    atol = {torch.float16: 2e-2, torch.float32: 3e-2, torch.bfloat16: 1e-1}[dtype]
    torch.testing.assert_close(y_tri, y_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(state_tri, state_ref, rtol=rtol, atol=atol)
