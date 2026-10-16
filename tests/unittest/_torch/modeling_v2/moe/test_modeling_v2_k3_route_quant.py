# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the moe/k3_route_quant catalog entry: trtllm::k3_route_quant on one GPU (sm_100).

The reference is the stock op it replaces, trtllm::kimi_k3_noaux_tc_mxfp8_quant: the four outputs (top-16 ids,
routing weights, MXFP8 codes, UE8M0 scales) must be its bits. Checks, in file order:
- single calls at M 1-8, 16, 33 and 64 with random logits, with and without the early trigger: the stock op's bits,
  run-to-run identical, and each M's rows the bits of the same rows of the 64-token call;
- edge cases at M 1, 3 and 8 (40 tied selection keys, equal logits, huge logits, zero / large / denormal latent
  rows): the stock op's bits;
- a captured sequence of calls at three M replayed with rewritten inputs, eager calls of other M between replays:
  every call the bits of the same call made alone;
- M 0 and 65, a non-contiguous or mistyped input and a mis-sized bias raise ValueError, and the next call is correct;
- the first call of a build under CUDA-graph capture raises RuntimeError (the kernel compiles on its first call).
The op keeps no state between calls: its only process state is the compile cache.
"""

import contextlib
import functools

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.k3_route_quant import k3_route_quant

E, K, H = 896, 16, 3584
RSF = 2.827
DEV = "cuda"
M_ALL = (1, 2, 3, 4, 5, 6, 7, 8, 16, 33, 64)
EDGE_CASES = ("40_tied_keys", "all_equal_logits", "huge_logits", "zero_large_denormal_rows")
REPLAYS = 4


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="k3_route_quant needs sm_100")


@pytest.fixture(autouse=True)
def _inference_mode():
    """Run every check as the model runs the op, under inference mode."""
    with torch.inference_mode():
        yield


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Same shape, dtype and bits."""
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _same_outputs(got, want) -> bool:
    return len(got) == len(want) and all(_same(a, b) for a, b in zip(got, want))


@functools.lru_cache(maxsize=None)
def _bias() -> torch.Tensor:
    gen = torch.Generator(device=DEV).manual_seed(20260929)
    return (torch.randn(E, generator=gen, device=DEV) * 0.1).float()


def _draw(seed: int, m: int):
    """Router logits fp32 [m, 896] (std 2.5) and the latent bf16 [m, 3584] from a seed."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    logits = (torch.randn(m, E, generator=gen, device=DEV) * 2.5).float()
    return logits, (torch.randn(m, H, generator=gen, device=DEV) * 0.7).bfloat16()


def _stock(logits, latent):
    return torch.ops.trtllm.kimi_k3_noaux_tc_mxfp8_quant(logits, _bias(), latent, RSF)


def _edge_case(case: str):
    """8 tokens of an edge case: router logits, bias and latent."""
    gen = torch.Generator(device=DEV).manual_seed(20260930)
    m = 8
    bias = _bias().clone()
    latent = (torch.randn(m, H, generator=gen, device=DEV) * 0.7).bfloat16()
    logits = torch.randn(m, E, generator=gen, device=DEV) * 2.5
    if case == "40_tied_keys":
        # 40 equal keys compete for the top 16: ties go to the lower expert id.
        logits[:, 100:140] = 3.0
        bias[100:140] = 0.25
    elif case == "all_equal_logits":
        logits = torch.full((m, E), 0.3, device=DEV)
        bias = torch.zeros(E, device=DEV)
    elif case == "huge_logits":
        logits = torch.randn(m, E, generator=gen, device=DEV) * 40.0
    elif case == "zero_large_denormal_rows":
        latent[0] = 0
        latent[1, :64] = 0
        latent[2] = latent[2] * 3e4
        latent[3, ::7] = torch.tensor(1e-39).bfloat16()
        latent[4, 5] = torch.tensor(-3e38).bfloat16()
    else:
        raise ValueError(case)
    return logits.float().contiguous(), bias, latent


@contextlib.contextmanager
def _cold_compile_cache():
    """The op's compile cache emptied for the duration (and restored after), so that the next call is a first call."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_route_quant import op

    saved = dict(op._compiled)
    op._compiled.clear()
    try:
        yield
    finally:
        op._compiled.clear()
        op._compiled.update(saved)


def _raised_under_capture(fn) -> str:
    """Run ``fn`` under CUDA-graph capture; the message of the RuntimeError it raised, or '' if it raised none."""
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    resting = torch.cuda.current_stream()
    stream.wait_stream(resting)
    message = ""
    try:
        with torch.cuda.graph(graph, stream=stream):
            try:
                fn()
            except RuntimeError as exc:
                message = str(exc)
    finally:
        torch.cuda.set_stream(resting)
    del graph
    return message


@pytest.mark.parametrize("m", M_ALL)
def test_single_call(m):
    """The stock op's four outputs bit for bit, with and without the early trigger.

    Shapes and dtypes as the contract states; run-to-run identical; the rows of each M the bits of the same rows of
    the 64-token call.
    """
    logits64, latent64 = _draw(1, 64)
    logits, latent = logits64[:m].contiguous(), latent64[:m].contiguous()
    want = _stock(logits, latent)
    full = k3_route_quant(logits64, _bias(), latent64, RSF)
    for early in (False, True):
        got = k3_route_quant(logits, _bias(), latent, RSF, early_trigger=early)
        ids, weights, quantized, scales = got
        assert ids.dtype == torch.int32 and tuple(ids.shape) == (m, K)
        assert weights.dtype == torch.bfloat16 and tuple(weights.shape) == (m, K)
        assert quantized.dtype == torch.float8_e4m3fn and tuple(quantized.shape) == (m, H)
        assert scales.dtype == torch.uint8 and tuple(scales.shape) == (m, H // 32)
        assert all(t.is_contiguous() and t.device == logits.device for t in got)
        same = [_same(a, b) for a, b in zip(got, want)]
        rerun = _same_outputs(
            k3_route_quant(logits, _bias(), latent, RSF, early_trigger=early), got
        )
        rows = all(_same(a, b[:m]) for a, b in zip(got, full))
        print(
            f"OPCHECK op=k3_route_quant M={m} early_trigger={early} "
            f"stock_bits(ids,w,q,sf)={same} det={rerun} rows_as_m64={rows}"
        )
        assert all(same) and rerun and rows


@pytest.mark.parametrize("case", EDGE_CASES)
def test_edge_case(case):
    """Ties, saturation and zero / large / denormal latent rows at M 1, 3 and 8: the stock op's bits."""
    logits8, bias, latent8 = _edge_case(case)
    for m in (1, 3, 8):
        logits, latent = logits8[:m].contiguous(), latent8[:m].contiguous()
        want = torch.ops.trtllm.kimi_k3_noaux_tc_mxfp8_quant(logits, bias, latent, RSF)
        for early in (False, True):
            got = k3_route_quant(logits, bias, latent, RSF, early_trigger=early)
            same = [_same(a, b) for a, b in zip(got, want)]
            print(f"OPCHECK op=k3_route_quant case={case} M={m} early_trigger={early} same={same}")
            assert all(same), f"{case} M {m} early_trigger {early}: {same}"


def test_graph_capture_and_replay():
    """A captured sequence (M 8 with the early trigger, M 3 without, M 64 with) replayed with rewritten inputs.

    Between replays an eager call of another M (5, 1, 40, 2). Every replayed and eager call returns the bits of the
    same call made alone.
    """
    plan = ((8, True), (3, False), (64, True))
    static = [_draw(100 + i, m) for i, (m, _) in enumerate(plan)]

    def step():
        return [
            k3_route_quant(lg, _bias(), x, RSF, early_trigger=early)
            for (lg, x), (_, early) in zip(static, plan)
        ]

    step()  # every build compiled eagerly
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outs = step()
    for rep in range(REPLAYS):
        inputs = [_draw(200 + 10 * rep + i, m) for i, (m, _) in enumerate(plan)]
        alone = [
            k3_route_quant(lg, _bias(), x, RSF, early_trigger=early)
            for (lg, x), (_, early) in zip(inputs, plan)
        ]
        eager_logits, eager_latent = _draw(300 + rep, (5, 1, 40, 2)[rep])
        eager_alone = k3_route_quant(eager_logits, _bias(), eager_latent, RSF)
        torch.cuda.synchronize()
        for (lg, x), (new_lg, new_x) in zip(static, inputs):
            lg.copy_(new_lg)
            x.copy_(new_x)
        graph.replay()
        torch.cuda.synchronize()
        bad = [i for i, (got, want) in enumerate(zip(outs, alone)) if not _same_outputs(got, want)]
        assert not bad, f"replay {rep}: calls {bad} differ from the same calls alone"
        eager = k3_route_quant(eager_logits, _bias(), eager_latent, RSF)
        assert _same_outputs(eager, eager_alone), f"eager call after replay {rep}"
    del graph


def test_unsupported_calls_refused():
    """M 0 and 65, a strided or mistyped input and a mis-sized bias raise ValueError; the next call is correct."""
    logits, latent = _draw(400, 4)
    want = k3_route_quant(logits, _bias(), latent, RSF)
    for m in (0, 65):
        bad_logits = torch.zeros(m, E, device=DEV)
        bad_latent = torch.zeros(m, H, dtype=torch.bfloat16, device=DEV)
        with pytest.raises(ValueError):
            k3_route_quant(bad_logits, _bias(), bad_latent, RSF)
    wide = torch.zeros(4, 2 * E, device=DEV)
    wide[:, :E] = logits
    with pytest.raises(ValueError):
        k3_route_quant(wide[:, :E], _bias(), latent, RSF)  # a strided view of the logits
    with pytest.raises(ValueError):
        k3_route_quant(logits.bfloat16(), _bias(), latent, RSF)  # bf16 logits
    with pytest.raises(ValueError):
        k3_route_quant(logits, _bias(), latent.half(), RSF)  # fp16 latent
    with pytest.raises(ValueError):
        k3_route_quant(logits, _bias()[:-1].contiguous(), latent, RSF)  # 895 biases
    with pytest.raises(ValueError):
        k3_route_quant(logits, _bias(), latent[:3].contiguous(), RSF)  # M differs
    assert _same_outputs(k3_route_quant(logits, _bias(), latent, RSF), want)


def test_first_call_of_a_build_refuses_capture():
    """The kernel compiles on the first call of each build (early trigger, PDL): under CUDA-graph capture that call
    raises RuntimeError instead of compiling into the capture, for both early-trigger builds."""
    logits, latent = _draw(500, 2)
    want = k3_route_quant(logits, _bias(), latent, RSF)
    for early in (False, True):
        with _cold_compile_cache():
            message = _raised_under_capture(
                lambda early=early: k3_route_quant(
                    logits, _bias(), latent, RSF, early_trigger=early
                )
            )
        assert "outside CUDA-graph capture" in message, f"early_trigger {early}: {message!r}"
    assert _same_outputs(k3_route_quant(logits, _bias(), latent, RSF), want)
