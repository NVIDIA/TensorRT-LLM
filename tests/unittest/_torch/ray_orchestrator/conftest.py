import os
import sys
import types

import pytest

from tensorrt_llm._utils import mpi_disabled


def pytest_configure(config):
    if config.getoption("--run-ray"):
        os.environ["TLLM_DISABLE_MPI"] = "1"
        os.environ["TLLM_RAY_FORCE_LOCAL_CLUSTER"] = "1"
        os.environ["RAY_raylet_start_wait_time_s"] = "120"


run_ray_flag = "--run-ray" in sys.argv
if run_ray_flag:
    os.environ["TLLM_DISABLE_MPI"] = "1"
    os.environ["TLLM_RAY_FORCE_LOCAL_CLUSTER"] = "1"
    os.environ["RAY_raylet_start_wait_time_s"] = "120"


def _install_finegrained_fp8_stub():
    # transformers >= 5.5 lazy-loads the FP8 triton kernel from HF Hub
    # (kernels-community/finegrained-fp8). Jenkins workers don't reach HF Hub,
    # so `lazy_load_kernel` returns None and FP8Linear.forward crashes inside
    # `_load_triton_kernel`. Install pure-PyTorch fallbacks for the four
    # entry points it expects, before any HF FP8 model is loaded.
    try:
        import torch
        import transformers.integrations.finegrained_fp8 as fg_fp8
        import transformers.integrations.hub_kernels as hub_kernels
    except ImportError:
        return

    _FP8_DTYPE = torch.float8_e4m3fn
    _FP8_MAX = torch.finfo(_FP8_DTYPE).max
    _FP8_MIN = torch.finfo(_FP8_DTYPE).min

    def fp8_act_quant(x, block_size=128):
        orig_shape = x.shape
        K = orig_shape[-1]
        assert K % block_size == 0, (
            f"fp8_act_quant: last dim {K} not divisible by block_size {block_size}"
        )
        x_grouped = x.reshape(-1, K // block_size, block_size).to(torch.float32)
        s = (x_grouped.abs().amax(dim=-1) / 448.0).clamp(min=1e-12)
        qx = (x_grouped / s.unsqueeze(-1)).clamp(_FP8_MIN,
                                                 _FP8_MAX).to(_FP8_DTYPE)
        qx = qx.reshape(orig_shape)
        s = s.reshape(*orig_shape[:-1], K // block_size)
        return qx, s

    def w8a8_fp8_matmul(A,
                        B,
                        As,
                        Bs,
                        block_size=None,
                        output_dtype=torch.float32):
        orig_A_shape = A.shape
        K = orig_A_shape[-1]
        N = B.shape[0]
        A_2d = A.reshape(-1, K)
        M = A_2d.shape[0]

        Af = A_2d.to(torch.float32)
        Bf = B.to(torch.float32)

        if block_size is not None and len(block_size) == 2 and not (
                block_size[0] >= N and block_size[1] >= K):
            block_n, block_k = block_size
            As_flat = As.reshape(M, K // block_k)
            Af = Af * As_flat.repeat_interleave(block_k, dim=-1)
            Bs_exp = Bs.repeat_interleave(block_n,
                                          dim=0).repeat_interleave(block_k,
                                                                   dim=-1)
            Bf = Bf * Bs_exp[:N, :K]
        else:
            As_t = As if torch.is_tensor(As) else torch.tensor(float(As))
            Bs_t = Bs if torch.is_tensor(Bs) else torch.tensor(float(Bs))
            if As_t.ndim == 0:
                Af = Af * float(As_t)
            elif As_t.numel() == M:
                Af = Af * As_t.reshape(M, 1).to(torch.float32)
            else:
                Af = Af * As_t.to(torch.float32)
            if Bs_t.numel() == 1:
                Bf = Bf * float(Bs_t.flatten()[0])
            else:
                Bf = Bf * Bs_t.reshape(N, -1).to(torch.float32)

        C = Af @ Bf.t()
        return C.reshape(*orig_A_shape[:-1], N).to(output_dtype)

    def w8a8_fp8_matmul_batched(input,
                                weights,
                                weight_scale_inv,
                                block_size=None,
                                expert_ids=None,
                                output_dtype=None):
        output_dtype = output_dtype or input.dtype
        S, K = input.shape
        E, N, _ = weights.shape
        input_f = input.to(torch.float32)
        selected_w = weights[expert_ids].to(torch.float32)  # (S, N, K)
        selected_ws = weight_scale_inv[expert_ids]
        if block_size is not None:
            block_n, block_k = block_size
            ws_exp = selected_ws.repeat_interleave(
                block_n, dim=-2).repeat_interleave(block_k, dim=-1)
            selected_w = selected_w * ws_exp[:, :N, :K]
        else:
            if selected_ws.numel() == S:
                selected_w = selected_w * selected_ws.reshape(S, 1, 1).to(
                    torch.float32)
            elif selected_ws.ndim == 3:
                selected_w = selected_w * selected_ws.to(torch.float32)
            else:
                selected_w = selected_w * float(selected_ws.flatten()[0])
        out = torch.bmm(input_f.unsqueeze(1),
                        selected_w.transpose(-1, -2)).squeeze(1)
        return out.to(output_dtype)

    def w8a8_fp8_matmul_grouped(input,
                                weights,
                                weight_scale_inv,
                                tokens_per_expert=None,
                                block_size=None,
                                offsets=None,
                                output_dtype=None):
        output_dtype = output_dtype or input.dtype
        S, K = input.shape
        E, N, _ = weights.shape
        out = torch.empty(S, N, dtype=output_dtype, device=input.device)
        if offsets is None:
            offsets = torch.cumsum(tokens_per_expert, dim=0)
        start = 0
        for e in range(E):
            end = int(offsets[e].item())
            if end <= start:
                start = end
                continue
            sub_in = input[start:end].to(torch.float32)
            w = weights[e].to(torch.float32)
            ws = weight_scale_inv[e]
            if block_size is not None:
                block_n, block_k = block_size
                w = w * ws.repeat_interleave(block_n, dim=0).repeat_interleave(
                    block_k, dim=-1)[:N, :K]
            else:
                w = w * (float(ws) if ws.numel() == 1 else ws.to(torch.float32))
            out[start:end] = (sub_in @ w.t()).to(output_dtype)
            start = end
        return out

    stub = types.SimpleNamespace(
        w8a8_fp8_matmul=w8a8_fp8_matmul,
        fp8_act_quant=fp8_act_quant,
        w8a8_fp8_matmul_batched=w8a8_fp8_matmul_batched,
        w8a8_fp8_matmul_grouped=w8a8_fp8_matmul_grouped,
    )
    hub_kernels._KERNEL_MODULE_MAPPING["finegrained-fp8"] = stub

    def _patched_load_triton_kernel():
        fg_fp8.triton_fp8_matmul = w8a8_fp8_matmul
        fg_fp8.triton_fp8_act_quant = fp8_act_quant
        fg_fp8.triton_batched_fp8_matmul = w8a8_fp8_matmul_batched
        fg_fp8.triton_grouped_fp8_matmul = w8a8_fp8_matmul_grouped
        fg_fp8._triton_available = True

    # The deepgemm kernel is also lazy-loaded from HF Hub and would raise
    # AttributeError (not ImportError) when the hub is unreachable, escaping
    # the caller's try/except in `w8a8_fp8_matmul`. Force the triton fallback
    # by raising ImportError up-front.
    def _patched_load_deepgemm_kernel():
        fg_fp8._deepgemm_available = False
        raise ImportError("deepgemm disabled by ray_orchestrator conftest stub")

    fg_fp8._load_triton_kernel = _patched_load_triton_kernel
    fg_fp8._load_deepgemm_kernel = _patched_load_deepgemm_kernel
    _patched_load_triton_kernel()


_install_finegrained_fp8_stub()


def pytest_collection_modifyitems(config, items):
    """Skip ray_orchestrator tests when MPI is not disabled.

    Uses hook instead of module-level pytest.skip() which is incompatible
    with conftest loading in pytest 8+.
    """
    if not mpi_disabled():
        skip_ray = pytest.mark.skip(
            reason=
            "Ray tests are only tested in Ray CI stage or with --run-ray flag")
        for item in items:
            if "ray_orchestrator" in item.nodeid:
                item.add_marker(skip_ray)
