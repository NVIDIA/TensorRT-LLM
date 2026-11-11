# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the FP8 unified target's MoE and dense entries and the paths no cell reaches."""

import zlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_post_inject import (  # noqa: E501
    qwen4_exp_post_inject,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.moe.renorm_moe_routing_op import (  # noqa: E501
    renorm_moe_routing_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.quantization.fp8_quantize_1x128 import (  # noqa: E501
    fp8_quantize_1x128,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.embedding.embed_rows__decode import (  # noqa: E501
    embed_rows,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.gemm.low_m_bf16_mm__decode import (  # noqa: E501
    low_m_bf16_mm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.gemm.low_m_bf16_mm__decode import (  # noqa: E501
    plan as low_m_plan,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.moe import (  # noqa: E501
    fp8_block_scale_moe_runner as _runner_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.moe import (  # noqa: E501
    moe_prefill_grouped__prefill as _grouped,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.moe import (  # noqa: E501
    moe_tpb_decode__decode as _tpb,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.moe.fp8_block_scale_moe_runner import (  # noqa: E501
    BLOCK,
    WIDE_DECODE_MIN_ROWS,
    WIDE_DECODE_SMALL_TACTICS,
    WIDE_DECODE_TACTIC,
    fp8_block_scale_moe_front,
    fp8_block_scale_moe_runner,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.moe.moe_front_router_quant__decode import (  # noqa: E501
    moe_front_router_quant,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.moe.mtp_moe_combine_inject__decode import (  # noqa: E501
    mtp_moe_combine_inject,
)
from tensorrt_llm._torch.moe.fused_shared_expert import fused_sigmoid_gate_mul_add

__extra_import_path__ = ["../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
)

assert torch.cuda.is_available(), "the MoE and dense entries require a CUDA device"

DEV = "cuda"
_BF16_ULP = 2.0**-8


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


# ------------------------------------------------------- fp8_block_scale_moe_runner ---


def _moe_compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Gate in bf16 ulp of each row's max: 16 element-wise, 0.75 RMS. Kernel and reference differ
    only by accumulation order and FC1 sums at e4m3 rounding boundaries (measured 6.65 / 0.306)."""
    assert out.dtype == ref.dtype == torch.bfloat16, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    o, r = out.float(), ref.float()
    rowmax = r.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    d = (o - r) / (rowmax * 2.0**-8)
    elt, rms = d.abs().max().item(), d.pow(2).mean().sqrt().item()
    assert elt <= 16.0, f"{elt:.2f} ulp element-wise > 16"
    assert rms <= 0.75, f"{rms:.3f} ulp RMS > 0.75"


CKPT = dict(hidden=2560, inter=640, num_experts=512, top_k=10)

#: The arguments a step writes: what a captured graph reads afresh on each replay.
STEP_INPUTS = ("hidden_states", "hidden_states_scale", "topk_weights", "topk_ids")

_WEIGHTS: dict = {}


def _weights(hidden: int, inter: int, experts: int, seed: int):
    """`(w1, w1s, w2, w2s)` drawn directly in e4m3 / fp32, so every value is exact. Cached: the
    checkpoint stack is 2.5 GB. Drawn 32 experts at a time to cap the fp32 transient."""
    key = (hidden, inter, experts, seed)
    if key not in _WEIGHTS:
        g = torch.Generator(device=DEV).manual_seed(seed)
        w1 = torch.empty(experts, 2 * inter, hidden, dtype=torch.float8_e4m3fn, device=DEV)
        w2 = torch.empty(experts, hidden, inter, dtype=torch.float8_e4m3fn, device=DEV)
        for lo in range(0, experts, 32):
            hi = min(lo + 32, experts)
            w1[lo:hi] = torch.randn(hi - lo, 2 * inter, hidden, device=DEV, generator=g)
            w2[lo:hi] = torch.randn(hi - lo, hidden, inter, device=DEV, generator=g)
        w1s = 0.01 + 0.02 * torch.rand(
            experts, 2 * inter // BLOCK, hidden // BLOCK, device=DEV, generator=g
        )
        w2s = 0.01 + 0.02 * torch.rand(
            experts, hidden // BLOCK, inter // BLOCK, device=DEV, generator=g
        )
        _WEIGHTS[key] = (w1, w1s, w2, w2s)
    return _WEIGHTS[key]


def _stack_window(offset: int, width: int):
    """`(w1, w1s, w2, w2s)` of experts `[offset, offset + width)` of the checkpoint-shape stack."""
    win = slice(offset, offset + width)
    return tuple(t[win] for t in _weights(2560, 640, 512, seed=2560 + 640 + 512))


def _kwargs(hs, hs_scale, weights, offset, local, num_experts=512, top_k=10, inter=640, **routing):
    w1, w1s, w2, w2s = weights
    # fmt: off
    base = dict(
        routing_logits=None, routing_bias=None, hidden_states=hs, hidden_states_scale=hs_scale,
        gemm1_weights=w1, gemm1_weights_scale=w1s, gemm2_weights=w2, gemm2_weights_scale=w2s,
        num_experts=num_experts, top_k=top_k, num_fused_shared_experts=None, n_group=None,
        topk_group=None, intermediate_size=inter, local_expert_offset=offset,
        local_num_experts=local, routed_scaling_factor=None, routing_method_type=1,
        topk_weights=None, topk_ids=None, gemm1_clamp_limit=None,
    )
    # fmt: on
    return dict(base, **routing)


def _runner_kwargs(xq, xsf, offset, width, **routing) -> dict:
    return _kwargs(xq, xsf, _stack_window(offset, width), offset, width, **routing)


def _args(
    tokens, hidden, inter, num_experts, top_k, local_expert_offset=0, local_num_experts=None, seed=0
):
    """Pre-routed op arguments over a window of one stack; one seed's windows split one layer."""
    local = num_experts if local_num_experts is None else local_num_experts
    window = slice(local_expert_offset, local_expert_offset + local)
    stack = _weights(hidden, inter, num_experts, seed=hidden + inter + num_experts)
    g = torch.Generator(device=DEV).manual_seed(seed)
    hs = (2.0 * torch.randn(tokens, hidden, device=DEV, generator=g)).to(torch.float8_e4m3fn)
    # [H/128, T]: the transposed layout fp8_quantize_1x128 returns here.
    hs_scale = 0.5 + torch.rand(hidden // BLOCK, tokens, device=DEV, generator=g)
    logits = torch.randn(tokens, num_experts, device=DEV, generator=g)
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    return _kwargs(
        hs,
        hs_scale,
        tuple(t[window] for t in stack),
        local_expert_offset,
        local,
        num_experts,
        top_k,
        inter,
        topk_weights=weights.to(torch.bfloat16).contiguous(),
        topk_ids=ids.to(torch.int32).contiguous(),
    )


def _window_args(cell, salt: int = 0) -> tuple[dict, torch.Tensor]:
    """The cell's arguments and in-window slots, redrawn while the routing misses the window."""
    geometry = {k: v for k, v in cell.spec.items() if k != "capture"}
    for attempt in range(8):
        args = _args(**geometry, seed=_seed(cell.why) + salt + attempt)
        hit = _in_window(args)
        if hit.any():
            return args, hit
    raise AssertionError(f"no slot routes into the window: {cell.why!r} would certify nothing")


def _in_window(args) -> torch.Tensor:
    """`[T, top_k]`: which slots the call's window answers for."""
    local = args["topk_ids"].long() - args["local_expert_offset"]
    return (local >= 0) & (local < args["local_num_experts"])


def _moe_check(out: torch.Tensor, args: dict, hit: torch.Tensor) -> None:
    assert out.shape == args["hidden_states"].shape and out.dtype == torch.bfloat16
    miss = ~hit.any(-1)
    assert torch.equal(out[miss], torch.zeros_like(out[miss]))
    _moe_compare(out, fp8_block_scale_moe_runner.reference(**args))


def _warm_on_side_stream(fn) -> None:
    """One eager call on a side stream before a capture, as `torch.cuda.graph` asks."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)


def _check_moe_runner(cell):
    args, hit = _window_args(cell)
    out = fp8_block_scale_moe_runner(**args)
    _moe_check(out, args, hit)
    if not cell.spec.get("capture"):
        return
    static = {k: (v.clone() if k in STEP_INPUTS else v) for k, v in args.items()}
    _warm_on_side_stream(lambda: fp8_block_scale_moe_runner(**static))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = fp8_block_scale_moe_runner(**static)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(replayed, out), "the replay differs from the eager call it captured"
    second, hit2 = _window_args(cell, salt=1 << 20)
    assert not torch.equal(second["topk_ids"], args["topk_ids"]), "the second step reroutes"
    for name in STEP_INPUTS:
        static[name].copy_(second[name])
    graph.replay()
    torch.cuda.synchronize()
    step2 = replayed.clone()
    assert torch.equal(step2, fp8_block_scale_moe_runner(**second)), "second replay vs eager"
    _moe_check(step2, second, hit2)


def _windows_sum(width, tokens):
    """A member's windows summed in fp32 pass the 512-expert gate; without any one they fail."""
    seed = 900 + tokens + width
    offsets = range(0, CKPT["num_experts"], width)
    whole = _args(tokens, **CKPT, seed=seed)
    parts = [
        fp8_block_scale_moe_runner(
            **_args(tokens, **CKPT, local_expert_offset=o, local_num_experts=width, seed=seed)
        )
        for o in offsets
    ]
    ref = fp8_block_scale_moe_runner.reference(**whole)
    _moe_compare(torch.stack([p.float() for p in parts]).sum(0).to(torch.bfloat16), ref)
    ids = whole["topk_ids"].long()
    for i, o in enumerate(offsets):
        if not ((ids >= o) & (ids < o + width)).any():
            continue
        short = torch.stack([p.float() for j, p in enumerate(parts) if j != i]).sum(0)
        with pytest.raises(AssertionError):
            _moe_compare(short.to(torch.bfloat16), ref)


def _pad_window(offset: int, width: int):
    return M._weights.ExpertWindow(*_stack_window(offset, width), offset=offset, count=width)


def _padded_rows(bucket: int, live: int, seed: int) -> torch.Tensor:
    """`[bucket, 2560]` bf16: `live` random rows, then the zero rows of a bucket's padding."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.zeros(bucket, 2560, dtype=torch.bfloat16, device=DEV)
    x[:live] = torch.randn(live, 2560, device=DEV, generator=g).to(torch.bfloat16)
    return x


def _padding_rows(offset, width):
    """A prefill graph's padding rows (`PrefillTarget._pad_mask`) route to expert -1 and come out
    zero, real rows bitwise unchanged, eager and replayed; unmasked they tie onto the lowest ids."""
    core = SimpleNamespace(topk=10, num_experts=512, moe_inter=640)
    g = torch.Generator(device=DEV).manual_seed(77)
    router = (0.02 * torch.randn(512, 2560, device=DEV, generator=g)).to(torch.bfloat16).t()
    win = _pad_window(offset, width)
    bucket, live = 2048, 1500
    x = _padded_rows(bucket, live, seed=5)
    pad_ids, _ = renorm_moe_routing_op(
        cublas_mm(x[live:], router), topk=10, output_dtype=torch.bfloat16
    )
    assert int(pad_ids.max()) < 10, "padding rows no longer tie onto the lowest expert ids"

    plain = M._routed_experts(core, x, router, win)
    mask = torch.unsqueeze(torch.arange(bucket, device=DEV) >= live, 1)
    masked = M._routed_experts(core, x, router, win, pad_mask=mask)
    assert torch.equal(masked[:live], plain[:live]), "the mask changed a real row"
    assert not masked[live:].any() and not plain[live:].any(), "a padding row is not zero"

    # As the prefill graph captures it: the mask from a device row count rewritten each step.
    live_dev = torch.full([1], live, dtype=torch.int32, device=DEV)
    xs = x.clone()

    def body():
        rows = torch.arange(bucket, dtype=torch.int32, device=DEV)
        return M._routed_experts(
            core, xs, router, win, pad_mask=torch.unsqueeze(rows >= live_dev, 1)
        )

    _warm_on_side_stream(body)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = body()
    for n, seed in ((live, 5), (700, 6)):
        xs.copy_(_padded_rows(bucket, n, seed))
        live_dev.fill_(n)
        graph.replay()
        torch.cuda.synchronize()
        eager = M._routed_experts(core, xs, router, win)
        assert torch.equal(out[:n], eager[:n]), f"live {n}: the replay changed a real row"
        assert not out[n:].any(), f"live {n}: a padding row is not zero"


@pytest.mark.parametrize(
    "kind, width, offset",
    [("sum", 128, 0), ("sum", 256, 0), ("pad", 512, 0), ("pad", 256, 0), ("pad", 256, 256)],
    ids=lambda v: str(v),
)
def test_moe_windows(kind, width, offset):
    """A member's MoE windows sum to the stack; a prefill graph's padding rows route to none."""
    if kind == "sum":
        _windows_sum(width, tokens=width)
    else:
        _padding_rows(offset, width)


# ---------------------------------------------------------------- low_m_bf16_mm__decode ---


def _low_m_compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Accumulation order only, as `cublas_mm`: 4 bf16 ulp element-wise, 2 ulp relative RMS."""
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0, ulp=ULP[torch.bfloat16])


def _low_m_build(m, n, k, seed):
    """`mat_b` is a `[N, K]` weight's `.t()`, as a target stores its projections."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    mat_a = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(n, k, generator=g, device="cuda") * k**-0.5).to(torch.bfloat16)
    return mat_a, weight.t()


def _check_low_m(cell):
    s = cell.spec
    assert low_m_plan(s["m"], s["n"], s["k"]).kind == s["kind"], low_m_plan(s["m"], s["n"], s["k"])
    mat_a, mat_b = _low_m_build(s["m"], s["n"], s["k"], seed=_seed(cell.why) % 2**31)
    out = low_m_bf16_mm(mat_a, mat_b, s["pdl"])
    assert out.shape == (s["m"], s["n"]) and out.dtype == torch.bfloat16 and out.is_contiguous()
    _low_m_compare(out, low_m_bf16_mm.reference(mat_a, mat_b, s["pdl"]))


def test_the_low_m_dependent_chain_under_graph_replay() -> None:
    """Decode replays these in CUDA graphs with PDL, each link reading the one before it, so a
    kernel that read its activation before `griddepcontrol.wait` would see the previous replay's
    values. One chain per branch at the target's rows: warp at M=4, direct at M=2."""
    chains = []
    for m, branches in ((4, ("warp", "warp", "warp")), (2, ("direct", "direct", "warp"))):
        o = _low_m_build(m, 2560, 1536, seed=m)[1]
        gdn_in = _low_m_build(m, 4128, 2560, seed=10 + m)[1]
        attn_in = _low_m_build(m, 4352, 2560, seed=20 + m)[1]
        got = tuple(low_m_plan(m, w.shape[1], w.shape[0]).kind for w in (o, gdn_in, attn_in))
        assert got == branches, got
        x = torch.zeros(m, 1536, dtype=torch.bfloat16, device="cuda")
        chains.append((x, o, gdn_in, attn_in))

    def run():
        outs = []
        for x, o, gdn_in, attn_in in chains:
            h = low_m_bf16_mm(x, o, True)
            outs.append((h, low_m_bf16_mm(h, gdn_in, True), low_m_bf16_mm(h, attn_in, True)))
        return outs

    _warm_on_side_stream(lambda: (run(), run()))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = run()
    for trial in range(3):
        g = torch.Generator(device="cuda").manual_seed(100 + trial)
        for x, *_ in chains:
            x.copy_(torch.randn(x.shape, generator=g, device="cuda").to(x.dtype))
        graph.replay()
        torch.cuda.synchronize()
        for (x, o, gdn_in, attn_in), (h, g_out, a_out) in zip(chains, outs):
            _low_m_compare(h, low_m_bf16_mm.reference(x, o))
            _low_m_compare(g_out, low_m_bf16_mm.reference(h, gdn_in))
            _low_m_compare(a_out, low_m_bf16_mm.reference(h, attn_in))


# ------------------------------------------------------- moe_front_router_quant__decode ---


def _front_operands(rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 MoE rows whose 1x128 blocks span many octaves (one zero block), and a router."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(rows, 2560, device=DEV, generator=g)
    octaves = torch.randint(-24, 12, (rows, 20, 1), device=DEV, generator=g).float()
    x = (x.reshape(rows, 20, 128) * torch.exp2(octaves)).reshape(rows, 2560)
    x[0, :128] = 0.0
    router = 0.02 * torch.randn(512, 2560, device=DEV, generator=g)
    return x.to(torch.bfloat16).contiguous(), router.to(torch.bfloat16).contiguous()


def _front_check(x, router, logits, xq, xsf, label: str) -> None:
    """Logits within one bf16 ulp of each row's largest |logit| of the exact product; codes and
    scales bitwise `fp8_quantize_1x128`'s, the scales within 4 fp32 ulp of the exact quotient."""
    _, _, ref_scales = moe_front_router_quant.reference(x, router)
    exact = x.double() @ router.double().t()
    rowmax = exact.abs().amax(dim=-1, keepdim=True)
    err = ((logits.double() - exact) / rowmax).abs().max().item()
    assert err <= _BF16_ULP, f"{label}: logits {err / _BF16_ULP:.3f} bf16 ulp off"
    want_q, want_s = fp8_quantize_1x128(x, use_ue8m0=False)
    assert torch.equal(xq.view(torch.uint8), want_q.view(torch.uint8)), f"{label}: e4m3 codes"
    assert torch.equal(xsf, want_s), f"{label}: activation scales"
    ratio = (xsf.double() / ref_scales.double() - 1.0).abs().max().item()
    assert ratio <= 4 * 2.0**-23, f"{label}: scales {ratio:.3g} from the exact quotient"


def _check_front(cell):
    rows = cell.spec["rows"]
    x, router = _front_operands(rows, _seed(cell.why))
    moe_front_router_quant.is_valid(x, router)
    logits, xq, xsf = moe_front_router_quant(x, router)
    _front_check(x, router, logits, xq, xsf, cell.why[:44])
    if not cell.spec["capture"]:
        return
    static = x.clone()
    _warm_on_side_stream(lambda: moe_front_router_quant(static, router))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = moe_front_router_quant(static, router)
    graph.replay()
    torch.cuda.synchronize()
    for got, want in zip(outs, (logits, xq, xsf)):
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
    x2, _ = _front_operands(rows, _seed(cell.why) + 1)
    static.copy_(x2)
    graph.replay()
    torch.cuda.synchronize()
    for got, want in zip(outs, moe_front_router_quant(x2, router)):
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8)), "second replay"
    _front_check(x2, router, *[o.clone() for o in outs], f"{cell.why[:44]} (second replay)")


def _coarse(logits: torch.Tensor) -> torch.Tensor:
    """Three values a row: exact ties decide most of the ranking."""
    return (logits.float().sign() * (logits.float().abs() > 0.5)).to(torch.bfloat16)


def _runner_routes_as_renorm(rows, width, offset, ties):
    """`routing_logits=` is bitwise the runner pre-routed by `renorm_moe_routing_op`."""
    x, router = _front_operands(rows, 7 * rows + width + offset)
    logits, xq, xsf = moe_front_router_quant(x, router)
    if ties:
        logits = _coarse(logits)
    ids, weights = renorm_moe_routing_op(logits, topk=10, output_dtype=torch.bfloat16)
    pre_routed = fp8_block_scale_moe_runner(
        **_runner_kwargs(xq, xsf, offset, width, topk_weights=weights, topk_ids=ids)
    )
    routed = fp8_block_scale_moe_runner(
        **_runner_kwargs(xq, xsf, offset, width, routing_logits=logits)
    )
    torch.cuda.synchronize()
    assert torch.equal(routed, pre_routed), (
        f"{(routed != pre_routed).sum().item()} of {routed.numel()} elements differ"
    )


def _front_is_the_logits_routed_runner(rows, width, offset, ties):
    """`fp8_block_scale_moe_front` and its folded quantization are bitwise the logits-routed
    call and `fp8_quantize_1x128`; a capture replays on new rows (counters reset each launch)."""

    def operands(seed):
        x, router = _front_operands(rows, seed)
        return x, cublas_mm(x, router.t())

    def want(x, logits):
        xq, xsf = fp8_quantize_1x128(x, use_ue8m0=False)
        return fp8_block_scale_moe_runner(
            **_runner_kwargs(xq, xsf, offset, width, routing_logits=logits)
        )

    x, logits = operands(13 * rows + width + offset)
    if ties:
        logits = _coarse(logits)
    xq, xsf = fp8_quantize_1x128(x, use_ue8m0=False)
    weights = _stack_window(offset, width)
    geometry = (512, 10, 640, offset, width, 1)
    call = fp8_block_scale_moe_front(logits, xq, xsf, *weights, *geometry)
    assert call is not None
    routed = call(None)
    fq, fsf = torch.empty_like(xq), torch.empty_like(xsf)
    folded = fp8_block_scale_moe_front(logits, fq, fsf, *weights, *geometry)(x)
    ref = want(x, logits)
    torch.cuda.synchronize()
    assert torch.equal(routed, ref), f"{(routed != ref).sum().item()} of {ref.numel()} differ"
    assert torch.equal(folded, ref), f"{(folded != ref).sum().item()} of {ref.numel()} differ"
    assert torch.equal(fq.view(torch.uint8), xq.view(torch.uint8)) and torch.equal(fsf, xsf)

    static_x, static_logits = x.clone(), logits.clone()
    gq, gsf = torch.empty_like(xq), torch.empty_like(xsf)
    fp8_block_scale_moe_front(static_logits, gq, gsf, *weights, *geometry)(static_x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fp8_block_scale_moe_front(static_logits, gq, gsf, *weights, *geometry)(static_x)
    for seed in (1, 2):
        x2, logits2 = operands(1000 * seed + rows)
        static_x.copy_(x2)
        static_logits.copy_(logits2)
        graph.replay()
        ref2 = want(x2, logits2)
        torch.cuda.synchronize()
        assert torch.equal(out, ref2), f"replay {seed}: {(out != ref2).sum().item()} differ"


def _routed_experts_past_the_front_end(rows, offset, ties, monkeypatch):
    """Past `_MOE_FRONT_MAX_ROWS` decode routes in the runner front, bitwise the pre-routed call."""
    x, router = _front_operands(rows, 13 * rows + offset)
    if ties:
        # A router whose logits take three values a row: ties everywhere.
        router = (router.float().sign() * (router.float().abs() > 0.03) * 0.001).to(torch.bfloat16)
    window = _pad_window(offset, 256)
    core = SimpleNamespace(num_experts=512, topk=10, moe_inter=640, dp_size=1, moe_max_rows=16384)
    calls = []
    real = M.renorm_moe_routing_op

    def spy(*args, **kwargs):
        calls.append(args[0].shape[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(M, "renorm_moe_routing_op", spy)
    new = M._routed_experts(core, x, router.t(), window, mm=M._decode_projection)
    assert calls == [], "the logits-routed call launched the renorm routing"
    ids, w = real(M._decode_projection(x, router.t()), topk=10, output_dtype=x.dtype)
    xq, xsf = fp8_quantize_1x128(x, use_ue8m0=False)
    old = M._window_runner(core, window, xq, xsf, ids, w)
    torch.cuda.synchronize()
    assert torch.equal(new, old), f"{(new != old).sum().item()} of {new.numel()} elements differ"


def _wide_tactic_under_skewed_routing(rows, offset):
    """A logits-routed 256-expert window takes its pinned decode tactic for the row count, which
    under skewed routing like a real decode batch's is bitwise the fallback and in the gate."""
    x, router = _front_operands(rows, 17 * rows + offset)
    logits = (x.float() @ router.float().t()).to(torch.bfloat16)
    xq, xsf = fp8_quantize_1x128(x, use_ue8m0=False)
    hot = offset + 7
    logits[:, hot] = logits.abs().amax() + 4.0
    logits[: rows // 3, offset + 100] = logits.abs().amax() + 2.0
    inputs = [logits, None, xq, xsf, *_stack_window(offset, 256), None, None]
    geometry = (512, 10, 640, offset, 256, 1, 0, None, 8192, False)
    _runner_mod._DISPATCH.clear()
    runner, tactic = _runner_mod._dispatch(geometry, inputs)
    _runner_mod._DISPATCH.clear()
    fallback = [int(v) for v in runner.get_fallback_tactic(2560, rows)]
    if rows >= WIDE_DECODE_MIN_ROWS:
        want = WIDE_DECODE_TACTIC
    else:
        want = [t for start, t in WIDE_DECODE_SMALL_TACTICS if rows >= start][-1]
    assert list(tactic) == list(want), (tactic, want)
    out = runner.forward(inputs, tactic=list(tactic))
    ref = runner.forward(inputs, tactic=fallback)
    ids, weights = renorm_moe_routing_op(logits, topk=10, output_dtype=torch.bfloat16)
    assert bool((ids == hot).any(dim=-1).all()), "the hot expert is not in every row's top-10"
    exact = fp8_block_scale_moe_runner.reference(
        **_runner_kwargs(xq, xsf, offset, 256, topk_weights=weights, topk_ids=ids)
    )
    torch.cuda.synchronize()
    assert torch.equal(out, ref), "the chosen tactic changes the output"
    _moe_compare(out, exact)


_FRONT_WINDOWS = ((128, 384), (256, 256), (512, 0))
#: `(path, rows, window width, window offset, coarse logits)`.
# fmt: off
_FRONT_CASES = (
    [("runner", r, w, o, t) for r in (1, 16) for w, o in _FRONT_WINDOWS for t in (False, True)]
    + [("front", r, w, o, t) for r in (1, 129, 576) for w, o in _FRONT_WINDOWS for t in (False, True)]
    + [("routed_experts", r, 256, o, t) for r in (96, 576) for o in (0, 256) for t in (False, True)]
    + [("wide_tactic", r, 256, o, False) for r in (3, 96, 167, 168, 576) for o in (0, 256)]
)
# fmt: on


@pytest.mark.parametrize("path, rows, width, offset, ties", _FRONT_CASES, ids=lambda v: str(v))
def test_decode_moe_front_routes_as_renorm(path, rows, width, offset, ties, monkeypatch):
    """The decode MoE front end routes as `renorm_moe_routing_op` does, along each `path`."""
    if path == "runner":
        _runner_routes_as_renorm(rows, width, offset, ties)
    elif path == "front":
        _front_is_the_logits_routed_runner(rows, width, offset, ties)
    elif path == "routed_experts":
        _routed_experts_past_the_front_end(rows, offset, ties, monkeypatch)
    else:
        _wide_tactic_under_skewed_routing(rows, offset)


# ------------------------------------------------ contiguous grouped prefill experts ---


def _logits_args(tokens: int, offset: int, local: int, seed: int):
    """bf16 router logits and the runner's arguments pre-routed from them over a stack window."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(tokens, 2560, device=DEV, generator=g).to(torch.bfloat16)
    logits = torch.randn(tokens, 512, device=DEV, generator=g).to(torch.bfloat16)
    xq, xsf = fp8_quantize_1x128(x, use_ue8m0=False)
    ids, w = renorm_moe_routing_op(logits, topk=10, output_dtype=torch.bfloat16)
    routing = dict(topk_weights=w.contiguous(), topk_ids=ids.contiguous())
    return logits, _runner_kwargs(xq, xsf, offset, local, **routing)


def _reroute(args: dict, logits: torch.Tensor) -> None:
    ids, w = renorm_moe_routing_op(logits, topk=10, output_dtype=torch.bfloat16)
    args["topk_ids"], args["topk_weights"] = ids.contiguous(), w.contiguous()


def _grouped_call(args, **kw):
    names = ("hidden_states", "hidden_states_scale", "topk_ids", "topk_weights", "gemm1_weights")
    names += ("gemm1_weights_scale", "gemm2_weights", "gemm2_weights_scale", "local_expert_offset")
    return _grouped.moe_prefill_grouped(*(args[n] for n in names), **kw)


def _grouped_matches_the_reference(tokens, offset, local):
    """The contiguous grouped path is in the gate, zero outside the window, deterministic despite
    atomic slot ranks, and bitwise the same with either FC1 epilogue."""
    _, args = _logits_args(tokens, offset, local, seed=_seed(f"grouped{tokens}-{offset}"))
    out = _grouped_call(args)
    _moe_check(out, args, _in_window(args))
    assert torch.equal(out, _grouped_call(args))
    epi = _grouped.EPI_QUANT
    try:
        _grouped.EPI_QUANT = not epi
        assert torch.equal(out, _grouped_call(args))
    finally:
        _grouped.EPI_QUANT = epi


def _grouped_skewed_and_empty():
    """All rows on ten experts (warmup dummy tokens) pass the gate; an empty window gives zeros."""
    logits, args = _logits_args(2048, 0, 256, seed=_seed("grouped-skew"))
    hot = torch.tensor([3, 4, 40, 41, 42, 100, 101, 180, 250, 255], device=DEV)
    logits = logits.clone()
    logits[:, hot] += 30.0
    _reroute(args, logits)
    out = _grouped_call(args)
    _moe_check(out, args, _in_window(args))
    assert torch.equal(out, _grouped_call(args))
    args["local_expert_offset"] = 256
    assert torch.equal(_grouped_call(args), torch.zeros_like(out))


def _grouped_fin_first(first):
    """`fin_first` rows are final when `after_first` gets the output, and the whole output is
    bitwise the one-launch finalize's."""
    logits, args = _logits_args(3000, 128, 256, seed=_seed("grouped-finfirst"))
    logits = logits.clone()
    logits[:, 140:150] += 4.0
    _reroute(args, logits)
    ref = _grouped_call(args)
    r0, r1 = (0, 1408) if first == "head" else (1500, ref.shape[0])
    seen = []

    def after_first(out):
        torch.cuda.synchronize()
        seen.append(out[r0:r1].clone())

    got = _grouped_call(args, fin_first=(r0, r1), after_first=after_first)
    torch.cuda.synchronize()
    assert len(seen) == 1 and torch.equal(seen[0], ref[r0:r1])
    assert torch.equal(got, ref)


def _grouped_graph_replay():
    """Replays over two steps' inputs are bitwise their eager calls (tiles rebuilt on device)."""
    _, args = _logits_args(2048, 256, 256, seed=_seed("groupedcap"))
    _, args2 = _logits_args(2048, 256, 256, seed=_seed("groupedcap2"))
    eager, eager2 = _grouped_call(args), _grouped_call(args2)
    static = {k: (v.clone() if k in STEP_INPUTS else v) for k, v in args.items()}
    _warm_on_side_stream(lambda: _grouped_call(static))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _grouped_call(static)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
    for k in STEP_INPUTS:
        static[k].copy_(args2[k])
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager2)


@pytest.mark.parametrize(
    "case",
    ["2048-0-256", "4096-256-256", "3000-128-128", "skewed", "fin_first-head", "fin_first-tail",
     "graph"],
)  # fmt: skip
def test_prefill_grouped(case):
    """The contiguous grouped prefill experts against the runner's reference, case by case."""
    if case == "skewed":
        _grouped_skewed_and_empty()
    elif case.startswith("fin_first-"):
        _grouped_fin_first(case.split("-")[1])
    elif case == "graph":
        _grouped_graph_replay()
    else:
        _grouped_matches_the_reference(*map(int, case.split("-")))


# ------------------------------------------------------------------ embed_rows__decode ---


def _check_embed_rows(cell):
    """Bitwise the reference over int32 and int64 ids (-1 clamped); a captured launch replays."""
    rows, copies, clamp = cell.spec["rows"], cell.spec["copies"], cell.spec["clamp"]
    g = torch.Generator(device=DEV).manual_seed(_seed(cell.why))
    table = torch.randn(1000, 2560, device=DEV, generator=g).to(torch.bfloat16)
    for dtype in (torch.int32, torch.int64):
        ids = torch.randint(0, 1000, (rows,), device=DEV, generator=g).to(dtype)
        if clamp:
            ids[-1] = -1
        got = embed_rows(ids, table, copies, clamp)
        assert torch.equal(
            got.view(torch.int16), embed_rows.reference(ids, table, copies, clamp).view(torch.int16)
        )
    static = ids.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = embed_rows(static, table, copies, clamp)
    static.copy_(torch.randint(0, 1000, (rows,), device=DEV, generator=g).to(static.dtype))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, embed_rows.reference(static, table, copies, clamp))


# ------------------------------------------------------- mtp_moe_combine_inject__decode ---


def _check_mtp_combine(cell):
    """Bitwise the unfused ops, within bf16 rounding of the fp32 reference; a capture replays."""
    rows, hc, hidden = cell.spec["rows"], cell.spec["hc_count"], cell.spec["hidden"]
    g = torch.Generator(device=DEV).manual_seed(_seed(cell.why))

    def operands():
        buf = torch.randn(rows, 2 * hc * hidden, device=DEV, generator=g).to(torch.bfloat16)
        x = buf[:, : hc * hidden]
        routed = (0.5 * torch.randn(rows, hidden, device=DEV, generator=g)).to(torch.bfloat16)
        shared = (0.5 * torch.randn(rows, hidden, device=DEV, generator=g)).to(torch.bfloat16)
        gug = (2.0 * torch.randn(rows, 1288, device=DEV, generator=g)).to(torch.bfloat16)
        gate = gug[:, 1280:1281] if cell.spec["gate"] == "strided" else gug[:, 1280].contiguous()
        inject = (2.0 * torch.sigmoid(torch.randn(rows, hc, device=DEV, generator=g))).to(
            torch.bfloat16
        )
        return x, routed, gate, shared, inject

    def composed(x, routed, gate, shared, inject):
        return qwen4_exp_post_inject(
            x, fused_sigmoid_gate_mul_add(routed.clone(), gate, shared), inject
        )

    ops = operands()
    got = mtp_moe_combine_inject(*ops)
    assert torch.equal(got.view(torch.int16), composed(*ops).view(torch.int16))
    torch.testing.assert_close(
        got.float(), mtp_moe_combine_inject.reference(*ops).float(), atol=0.07, rtol=0.02
    )
    static = [t.clone() for t in ops]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = mtp_moe_combine_inject(*static)
    new = operands()
    for s, n in zip(static, new):
        s.copy_(n)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, composed(*new))


# ------------------------------------------------------------ moe_tpb_decode__decode ---


_TPB_BLOCKS: dict = {}


def _tpb_dims(rank: int) -> SimpleNamespace:
    """The loader's dims of tep4 rank `rank` with the block-major copy on."""
    return SimpleNamespace(
        moe_tpb=True, tp_rank=rank, num_experts=512, moe_inter=640, shared_inter=640,
        hidden=2560, dtype=torch.bfloat16,
    )  # fmt: skip


def _loaded(d, fake, names) -> list:
    """The loader's parameters `names` of trunk layer 0, filled from `fake` as `load` fills them."""
    entries = {e.name: e for e in M._weights.MODEL_WEIGHTS.WEIGHTS}
    out = []
    for name in names:
        e = entries[name]
        param = torch.empty(e.shape(d), dtype=e.dtype or d.dtype, device=DEV)
        sources = [(*s, e.transform) for s in e.sources(d, 0, "")]
        M._weights.MODEL_WEIGHTS.fill(param, sources, fake, d)
        out.append(param)
    return out


def _tpb_blocks(rank: int):
    """Tep4 rank `rank`'s block-major routed experts as the loader lays them out from a
    checkpoint holding the stack `_weights` draws (rows `[up ; gate]`). Cached per rank."""
    if rank not in _TPB_BLOCKS:
        w1, w1s, w2, w2s = _weights(2560, 640, 512, seed=2560 + 640 + 512)
        p = f"{M._weights._trunk_mlp(0)}.experts"
        fake = {}
        for e in range(512):
            fake.update({
                f"{p}.{e}.up_proj.weight": w1[e, :640],
                f"{p}.{e}.gate_proj.weight": w1[e, 640:],
                f"{p}.{e}.down_proj.weight": w2[e],
                f"{p}.{e}.up_proj.weight_scale_inv": w1s[e, :5],
                f"{p}.{e}.gate_proj.weight_scale_inv": w1s[e, 5:],
                f"{p}.{e}.down_proj.weight_scale_inv": w2s[e],
            })  # fmt: skip
        names = ("tpb_w1", "tpb_s1", "tpb_w2", "tpb_s2")
        _TPB_BLOCKS[rank] = _tpb.BlockExperts(*_loaded(_tpb_dims(rank), fake, names), rank)
    return _TPB_BLOCKS[rank]


def _tpb_shared_blocks(gate, up, dn, rank: int):
    """Tep4 rank `rank`'s shared-expert slice `(gu, dn)` as the loader lays it out."""
    p = f"{M._weights._trunk_mlp(0)}.shared_expert"
    fake = {f"{p}.gate_proj.weight": gate, f"{p}.up_proj.weight": up, f"{p}.down_proj.weight": dn}
    return tuple(_loaded(_tpb_dims(rank), fake, ("tpb_sh_gu", "tpb_sh_dn")))


def _tpb_call(fn, logits, args, blocks, **kw):
    return fn(logits, args["hidden_states"], args["hidden_states_scale"], blocks, **kw)


def _tpb_shared(rows: int, seed: int):
    """A MoE site's bf16 rows and a bf16 shared expert: gate / up rows `[640, 2560]`, the
    gate-logit row `[2560]` and the down projection `[2560, 640]`."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(rows, 2560, device=DEV, generator=g)
    x = (x / x.abs().amax(-1, keepdim=True)).to(torch.bfloat16).contiguous()
    gate = (0.02 * torch.randn(640, 2560, device=DEV, generator=g)).to(torch.bfloat16)
    up = (0.02 * torch.randn(640, 2560, device=DEV, generator=g)).to(torch.bfloat16)
    gl = (0.02 * torch.randn(2560, device=DEV, generator=g)).to(torch.bfloat16).contiguous()
    dn = (0.02 * torch.randn(2560, 640, device=DEV, generator=g)).to(torch.bfloat16)
    return x, gate, up, gl, dn


def _unfused_shared(x, gate, up, gl, dn):
    """`(sigmoid(gate logit), shared)` of the unfused bf16 path, in fp32."""
    with true_fp32_matmul():
        gugate = (x.float() @ torch.cat([gate, up]).float().t()).to(torch.bfloat16).float()
        act = (
            (torch.nn.functional.silu(gugate[:, :640]) * gugate[:, 640:]).to(torch.bfloat16).float()
        )
        shared = (act @ dn.float().t()).to(torch.bfloat16).float()
        logit = (x.float() @ gl.float()).to(torch.bfloat16).float().unsqueeze(1)
    return torch.sigmoid(logit), shared


def _check_tpb(cell):
    """Block-major decode experts in the gate of `reference`, deterministic, and replayable."""
    rows, rank = cell.spec["rows"], cell.spec["rank"]
    logits, args = _logits_args(rows, 0, 512, seed=_seed(cell.why))
    blocks = _tpb_blocks(rank)
    out = _tpb_call(_tpb.moe_tpb_decode, logits, args, blocks)
    _moe_compare(out, _tpb_call(_tpb.moe_tpb_decode.reference, logits, args, blocks))
    assert torch.equal(out, _tpb_call(_tpb.moe_tpb_decode, logits, args, blocks)), "a re-run"
    if not cell.spec["capture"]:
        return
    logits2, args2 = _logits_args(rows, 0, 512, seed=_seed(cell.why) + 1)
    eager2 = _tpb_call(_tpb.moe_tpb_decode, logits2, args2, blocks)
    slog = logits.clone()
    sx = args["hidden_states"].clone()
    ssf = args["hidden_states_scale"].clone()
    _warm_on_side_stream(lambda: _tpb.moe_tpb_decode(slog, sx, ssf, blocks))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _tpb.moe_tpb_decode(slog, sx, ssf, blocks)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, out)
    slog.copy_(logits2)
    sx.copy_(args2["hidden_states"])
    ssf.copy_(args2["hidden_states_scale"])
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, eager2)


def _tpb_ranks_sum(rows):
    """The four round-robin ranks' partials summed in fp32 sit in the MoE gate of the 512-expert
    runner reference and of the trtllm-gen runner's four 128-expert windows summed alike."""
    logits, args = _logits_args(rows, 0, 512, seed=_seed(f"tpb-sum-{rows}"))
    parts = [_tpb_call(_tpb.moe_tpb_decode, logits, args, _tpb_blocks(r)) for r in range(4)]
    total = sum(p.float() for p in parts).to(torch.bfloat16)
    with true_fp32_matmul():
        ref = fp8_block_scale_moe_runner.reference(**args)
    _moe_compare(total, ref)
    xq, xsf = args["hidden_states"], args["hidden_states_scale"]
    routing = dict(topk_weights=args["topk_weights"], topk_ids=args["topk_ids"])
    windows = [
        fp8_block_scale_moe_runner(**_runner_kwargs(xq, xsf, 128 * r, 128, **routing))
        for r in range(4)
    ]
    _moe_compare(total, sum(p.float() for p in windows).to(torch.bfloat16))


def _tpb_shared_slice(rows):
    """With the shared fold each rank's partial sits in its reference's gate, and the four summed
    sit in the layer's: the routed reference + sigmoid(gate logit) x the unfused shared expert."""
    logits, args = _logits_args(rows, 0, 512, seed=_seed(f"tpb-shared-{rows}"))
    x, gate, up, gl, dn = _tpb_shared(rows, _seed(f"tpb-shared-w-{rows}"))
    total = torch.zeros(rows, 2560, device=DEV)
    for r in range(4):
        gu_r, dn_r = _tpb_shared_blocks(gate, up, dn, r)
        sh = _tpb.SharedBlocks(x, gu_r, gl, dn_r)
        blocks = _tpb_blocks(r)
        out = _tpb_call(_tpb.moe_tpb_decode, logits, args, blocks, shared=sh)
        with true_fp32_matmul():
            ref = _tpb_call(_tpb.moe_tpb_decode.reference, logits, args, blocks, shared=sh)
        _moe_compare(out, ref)
        assert torch.equal(out, _tpb_call(_tpb.moe_tpb_decode, logits, args, blocks, shared=sh))
        total += out.float()
    with true_fp32_matmul():
        routed = fp8_block_scale_moe_runner.reference(**args).float()
    scale, shared = _unfused_shared(x, gate, up, gl, dn)
    _moe_compare(total.to(torch.bfloat16), (routed + scale * shared).to(torch.bfloat16))


def _tpb_blocks_dealt():
    """The four ranks' 640 blocks are every (expert, block) of the 512 x 5 exactly once, each rank
    holds a block of every expert, and `weights.tpb_blocks` agrees with the op's `block_ids`."""
    seen = set()
    for r in range(4):
        d = SimpleNamespace(moe_tpb=True, tp_rank=r, num_experts=512, moe_inter=640)
        blocks = M._weights.tpb_blocks(d)
        e, b = _tpb.block_ids(r)
        assert blocks == list(zip(e, b))
        assert len(blocks) == 640 and {x for x, _ in blocks} == set(range(512))
        assert all((x + y) % 4 == r for x, y in blocks)
        seen |= set(blocks)
    assert seen == {(x, y) for x in range(512) for y in range(5)}


def _tpb_window(rank: int, shared) -> "M._weights.ExpertWindow":
    """Tep4 rank `rank`'s 128-expert window with its block-major copy and shared-expert slice."""
    bl = _tpb_blocks(rank)
    gate, up, gl, dn = shared
    sh_gu, sh_dn = _tpb_shared_blocks(gate, up, dn, rank)
    bw = M._weights.BlockWindow(bl.w1, bl.s1, bl.w2, bl.s2, rank, sh_gu, gl, sh_dn)
    return _pad_window(128 * rank, 128)._replace(blocks=bw)


def _moe_block_folds_shared(rows, monkeypatch):
    """A deferred tep4 decode step of `_MOE_TPB_SHARED_MIN_ROWS` (5) to 8 rows folds each rank's
    shared-expert slice into its block launch, the four partials summing into the gate of the
    unfused `routed + sigmoid(gate) * shared`. A 3-row step keeps the side-stream shared expert."""
    x, router = _front_operands(rows, 57 + rows)
    x = (x.float() / x.float().abs().amax(dim=-1, keepdim=True)).to(torch.bfloat16).contiguous()
    _, gate, up, gl, dn = _tpb_shared(rows, 91 + rows)
    gug = torch.zeros(1288, 2560, dtype=torch.bfloat16, device=DEV)
    gug[:640], gug[640:1280], gug[1280] = gate, up, gl
    folded_calls = []
    real = M.moe_tpb_decode

    def spy(*args, **kwargs):
        folded_calls.append(
            len(args) > 4 and args[4] is not None or kwargs.get("shared") is not None
        )
        return real(*args, **kwargs)

    monkeypatch.setattr(M, "moe_tpb_decode", spy)

    def block(rank):
        core = SimpleNamespace(
            num_experts=512, topk=10, moe_inter=640, shared_inter=640, dp_size=1,
            tp_sum=lambda t: t, _side_stream=torch.cuda.Stream(),
            _side_fork=torch.cuda.Event(), _side_join=torch.cuda.Event(),
        )  # fmt: skip
        L = SimpleNamespace(
            router=router.t(),
            experts=_tpb_window(rank, (gate, up, gl, dn)),
            sh_gug=gug.t(),
            sh_dn=dn.t(),
        )
        return M._moe_block(core, x, L, None, mm=M._decode_projection, defer=True)

    out = [block(r) for r in range(4)]
    torch.cuda.synchronize()
    fold = rows >= M._MOE_TPB_SHARED_MIN_ROWS
    assert folded_calls == [fold] * 4, folded_calls
    # The folded partials already carry sigmoid(gate) * shared; the unfolded block returns it apart.
    got = sum(p.float() for p, _, _ in out)
    if fold:
        assert all(g is None and s is None for _, g, s in out), "a folded block returned a shared"
    else:
        _, g0, s0 = out[0]
        got = got + torch.sigmoid(g0.float()) * s0.float()
    scale, shared = _unfused_shared(x, gate, up, gl, dn)
    logits = moe_front_router_quant(x, router)[0]
    ids, wts = renorm_moe_routing_op(logits, topk=10, output_dtype=torch.bfloat16)
    xq, xsf = fp8_quantize_1x128(x, use_ue8m0=False)
    args = _runner_kwargs(xq, xsf, 0, 512, topk_ids=ids, topk_weights=wts)
    with true_fp32_matmul():
        ref_routed = fp8_block_scale_moe_runner.reference(**args).float()
    _moe_compare(got.to(torch.bfloat16), (ref_routed + scale * shared).to(torch.bfloat16))


_TPB_CASES = (
    [("ranks_sum", rows) for rows in (1, 3, 6, 8)]
    + [("shared_slice", rows) for rows in (3, 6, 8)]
    + [("moe_block", rows) for rows in (3, 5, 8)]
    + [("blocks_dealt", 0)]
)


@pytest.mark.parametrize("case, rows", _TPB_CASES, ids=lambda v: str(v))
def test_tpb_decode_moe(case, rows, monkeypatch):
    """The TPB decode experts: rank sums, the shared-expert fold, and the block deal."""
    if case == "ranks_sum":
        _tpb_ranks_sum(rows)
    elif case == "shared_slice":
        _tpb_shared_slice(rows)
    elif case == "moe_block":
        _moe_block_folds_shared(rows, monkeypatch)
    else:
        _tpb_blocks_dealt()


# ------------------------------------------------------------------ certified cells ---

#: Each entry and its checker, which builds a cell's operands and compares entry and `reference`.
_CELL_CHECKS = (
    (fp8_block_scale_moe_runner, _check_moe_runner),
    (low_m_bf16_mm, _check_low_m),
    (moe_front_router_quant, _check_front),
    (embed_rows, _check_embed_rows),
    (mtp_moe_combine_inject, _check_mtp_combine),
    (_tpb.moe_tpb_decode, _check_tpb),
)
CASES = [(entry, check, cell) for entry, check in _CELL_CHECKS for cell in entry.CELLS]


@pytest.mark.parametrize(
    "entry, check, cell", CASES, ids=[f"{type(e).__name__}-{c.why[:40]}" for e, _, c in CASES]
)
def test_certified_cells(entry, check, cell, certifying):  # noqa: F811 — fixture by name
    certifying(entry)
    check(cell)
