# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical regression for adjacent mHC boundaries in a production-style graph."""

import pytest
import torch


@pytest.fixture
def deep_gemm():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Mega mHC requires SM100, SM103 or SM107")
    from tensorrt_llm import deep_gemm as module

    assert callable(module.mega_mhc)
    old_sms = module.get_num_sms()
    try:
        yield module
    finally:
        module.set_num_sms(old_sms)
        module.use_deterministic_algorithms(False)


def _inputs(rows: int, hidden: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(1701 + rows + hidden)

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, dtype=dtype, device="cuda", generator=generator)

    norm = (randn(hidden).abs() + 0.5).bfloat16()
    norm[:32] = 0
    norm[32:64] = 2.0**-20
    norm[64:96] = 3e-6
    return dict(
        x=randn(rows, hidden, dtype=torch.bfloat16),
        residual=randn(rows, 4, hidden, dtype=torch.bfloat16),
        post_mix=randn(rows, 4, 1).sigmoid() * 2,
        comb_res_mix=randn(rows, 4, 4).softmax(-1),
        shifted_prev_mix=randn(rows, 4, 1).sigmoid() + 1e-6,
        fn=randn(24, 4 * hidden) / hidden**0.5,
        mix_scales=torch.tensor([0.5, 0.8, 0.4], device="cuda"),
        mix_bases=randn(24) * 0.1,
        rmsnorm_weight=norm,
    )


def _outputs(rows: int, hidden: int, shifted: bool, mode: str) -> dict[str, torch.Tensor]:
    out = dict(
        new_residual=torch.empty(rows, 4, hidden, device="cuda", dtype=torch.bfloat16),
        new_post_mix=torch.empty(rows, 4, 1, device="cuda"),
        new_comb_res_mix=torch.empty(rows, 4, 4, device="cuda"),
    )
    if shifted:
        out["new_prev_mix"] = torch.empty(rows, 4, 1, device="cuda")
    if mode != "fp8_only":
        out["y_bf16"] = torch.empty(rows, hidden, device="cuda", dtype=torch.bfloat16)
    if mode != "bf16":
        out["y_fp8"] = torch.empty(rows, hidden, device="cuda", dtype=torch.float8_e4m3fn)
        # Column-major packed UE8M0 words, with TMA-aligned strides.
        out["y_gemm_sf"] = torch.empty(
            hidden // 128, (rows + 3) // 4 * 4, device="cuda", dtype=torch.int32
        )[:, :rows].T
    return out


def _launch(deep_gemm, inputs, outputs, shifted):
    args = dict(inputs)
    if not shifted:
        args["shifted_prev_mix"] = None
        args["new_prev_mix"] = None
    deep_gemm.mega_mhc(
        **args,
        **outputs,
        hc_mult=4,
        hc_norm_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_post_scale=2.0,
        sinkhorn_eps=1e-6,
        num_sinkhorn_iters=20,
        rmsnorm_eps=1e-6,
        rmsnorm_scale=1.0,
    )


def _reference(inputs, shifted):
    residual = inputs["residual"].double()
    updated = (
        inputs["post_mix"].double() * inputs["x"].double()[:, None, :]
        + torch.einsum("tji,tjh->tih", inputs["comb_res_mix"].double(), residual)
    ).bfloat16()
    flat = updated.double().flatten(1)
    mixes = (flat @ inputs["fn"].double().T) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + 1e-6
    )
    scales, bases = inputs["mix_scales"].double(), inputs["mix_bases"].double()
    pre = torch.sigmoid(mixes[:, :4] * scales[0] + bases[:4]) + 1e-6
    post = 2.0 * torch.sigmoid(mixes[:, 4:8] * scales[1] + bases[4:8])
    comb = (mixes[:, 8:] * scales[2] + bases[8:]).reshape(-1, 4, 4).softmax(-1) + 1e-6
    comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    collapse_pre = inputs["shifted_prev_mix"].double() if shifted else pre[:, :, None]
    collapsed = (updated.double() * collapse_pre).sum(1).bfloat16().double()
    normalized = (
        collapsed
        * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + 1e-6)
        * inputs["rmsnorm_weight"].double()
    ).bfloat16()
    return updated, pre, post, comb, normalized


def _check_reference(expected, outputs, shifted):
    residual, pre, post, comb, normalized = expected
    torch.testing.assert_close(
        outputs["new_residual"].float(), residual.float(), rtol=8e-3, atol=1e-5
    )
    torch.testing.assert_close(
        outputs["new_post_mix"].double().flatten(1), post, rtol=1e-3, atol=1e-4
    )
    torch.testing.assert_close(outputs["new_comb_res_mix"].double(), comb, rtol=1e-3, atol=1e-4)
    if shifted:
        torch.testing.assert_close(
            outputs["new_prev_mix"].double().flatten(1), pre, rtol=1e-3, atol=1e-4
        )
    if "y_bf16" in outputs:
        torch.testing.assert_close(
            outputs["y_bf16"].float(), normalized.float(), rtol=2e-2, atol=2e-2
        )


# num_sms deliberately bounds task waves independently of a GPU's total SM count.
_CASES = [pytest.param(4097, 5120, 10, False, id="many-task-waves")]


@pytest.mark.parametrize("rows,hidden,num_sms,deterministic", _CASES)
@pytest.mark.parametrize("shifted", [True], ids=["shifted"])
@pytest.mark.parametrize("mode", ["bf16"])
@torch.inference_mode()
def test_mega_mhc_pipeline_changed_input_graph(
    deep_gemm, rows, hidden, num_sms, deterministic, shifted, mode
):
    deep_gemm.set_num_sms(num_sms)
    deep_gemm.use_deterministic_algorithms(deterministic)
    inputs = _inputs(rows, hidden)
    captured = _outputs(rows, hidden, shifted, mode)
    first_captured = _outputs(rows, hidden, shifted, "bf16")
    first_eager = _outputs(rows, hidden, shifted, "bf16")
    eager = _outputs(rows, hidden, shifted, mode)
    reference_outputs = _outputs(rows, hidden, shifted, "bf16")
    second_fn = (
        torch.randn(
            inputs["fn"].shape,
            device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(9173 + rows + hidden),
        )
        / hidden**0.5
    )

    def second_inputs(first):
        args = dict(inputs)
        args.update(
            x=first["y_bf16"],
            residual=first["new_residual"],
            post_mix=first["new_post_mix"],
            comb_res_mix=first["new_comb_res_mix"],
            fn=second_fn,
        )
        if shifted:
            args["shifted_prev_mix"] = first["new_prev_mix"]
        return args

    def paired(first, second):
        _launch(deep_gemm, inputs, first, shifted)
        _launch(deep_gemm, second_inputs(first), second, shifted)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        paired(first_captured, captured)
        with torch.cuda.graph(graph, stream=stream):
            paired(first_captured, captured)
    torch.cuda.current_stream().wait_stream(stream)
    previous_residual = captured["new_residual"].clone()
    generator = torch.Generator(device="cuda").manual_seed(731)
    for phase in range(3):
        inputs["x"].normal_(generator=generator).mul_(0.25 + phase * 0.1)
        inputs["residual"].normal_(generator=generator).mul_(0.5 + phase * 0.1)
        graph.replay()
        torch.cuda.synchronize()
        assert not torch.equal(captured["new_residual"], previous_residual)
        paired(first_eager, eager)
        _launch(deep_gemm, second_inputs(first_eager), reference_outputs, shifted)
        torch.cuda.synchronize()
        for name in captured:
            assert torch.equal(
                captured[name].contiguous().view(torch.uint8),
                eager[name].contiguous().view(torch.uint8),
            ), name
        for name in first_captured:
            assert torch.equal(
                first_captured[name].contiguous().view(torch.uint8),
                first_eager[name].contiguous().view(torch.uint8),
            ), name
        _check_reference(_reference(inputs, shifted), first_captured, shifted)
        expected_math = _reference(second_inputs(first_captured), shifted)
        _check_reference(expected_math, captured, shifted)
        _check_reference(expected_math, reference_outputs, shifted)
        if mode != "fp8_only":
            assert torch.equal(captured["y_bf16"], reference_outputs["y_bf16"])
        previous_residual.copy_(captured["new_residual"])
        for _ in range(8):
            graph.replay()
        torch.cuda.synchronize()
        for name in captured:
            assert torch.equal(
                captured[name].contiguous().view(torch.uint8),
                eager[name].contiguous().view(torch.uint8),
            ), name
