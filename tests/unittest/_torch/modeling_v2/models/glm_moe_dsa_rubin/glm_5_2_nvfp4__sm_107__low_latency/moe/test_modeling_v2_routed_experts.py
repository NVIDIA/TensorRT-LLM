# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 tp16 routed_experts entry."""

import zlib

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.moe.routed_experts import (  # noqa: E501
    ACT_TYPE_SWIGLU,
    FUSED_ROUTER_WORKSPACE_WORDS,
    ROUTING_METHOD_DEEPSEEK_V3,
    make_fused_router_workspace,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.moe.routed_experts import (  # noqa: E501
    routed_experts as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — needs the path declared above
    assert_within_ulp,
    certifying,
    validating,
)

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the GLM-5.2 tp16 target's entries are certified on sm_107"
)

DEV = "cuda"
HIDDEN, INTER, EXPERTS, TOP_K = 6144, 128, 256, 8
ROUTED_SCALE = 2.5
SV = 16  # NVFP4 block size: one e4m3 scale per 16 elements along K
E2M1_MAX, E4M3_MAX = 6.0, 448.0
# Expert-selection scores (sigmoid + bias) of the 8th and 9th expert closer than
# this leave the reference's choice to the sigmoid's last bit; a case compared
# against the reference is drawn clear of it.
SELECTION_MARGIN = 1e-6
# Long enough (tens of ms) that a kernel not waiting for the side stream reads
# the stale input long before the side stream writes the real one.
SLEEP_CYCLES = 1 << 27
_BUFFER_WORDS = (FUSED_ROUTER_WORKSPACE_WORDS - 4) // 2
_EPOCH_WORD = 2 * _BUFFER_WORDS


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """16 ulp per element against the row's own scale, 2 ulp relative RMS.

    The band of the catalog `fp4_block_scale_moe_runner` entry, which compares
    the same kernels against the same reference over bit-identical NVFP4
    operands: they differ only in accumulation order and in whether a marginal
    FC1 value lands on the same e2m1 code at the FC2-input requantization. Its
    measured worst over every geometry it covers is 9.97 ulp element-wise and
    0.88 ulp RMS, 4.68 / 0.70 for DeepSeek-R1's 256-expert stack; a reference
    that skips the requantization lands 23 ulp RMS out. GLM's FC2 sums over 128
    terms where R1's sums over 2048, so accumulation order moves it less.
    Routing adds one source those comparisons did not have: a routing weight
    the kernel rounds to bf16 one ulp away from the reference's moves one
    slot's share of a row by 2^-8, under 1 ulp of the row. A different expert
    set is hundreds of ulp out.
    """
    assert_within_ulp(out, ref, element_ulp=16.0, rms_ulp=2.0)


# ── operands ──────────────────────────────────────────────────────────────


def _rand_nvfp4(e: int, n: int, k: int, gen: torch.Generator):
    """Random NVFP4 expert stack: `(packed [E, N, K/2] uint8, codes [E, N, K]
    uint8, scales [E, N, K/16] float8_e4m3fn)`. Codes and power-of-two scales
    are drawn directly, so every weight's fp32 value is exact."""
    codes = torch.randint(0, 16, (e, n, k), dtype=torch.uint8, device=DEV, generator=gen)
    exps = torch.randint(-2, 1, (e, n, k // SV), device=DEV, generator=gen, dtype=torch.int32)
    scales = torch.exp2(exps.float()).to(torch.float8_e4m3fn)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()
    return packed, codes, scales


def _blk32_perm(m: int) -> torch.Tensor:
    """Gather index of the 32-row block shuffle: within each block of 32 rows,
    source row `4u + v` lands at destination row `8v + u`."""
    j = torch.arange(32)
    idx = torch.empty(32, dtype=torch.long)
    idx[(j % 4) * 8 + j // 4] = j
    return (idx.repeat(m // 32) + torch.arange(m // 32).repeat_interleave(32) * 32).to(DEV)


def _interleave_perm(m: int) -> torch.Tensor:
    """Gather index interleaving `[up | gate]` rows into up0, gate0, up1, ..."""
    p = torch.empty(m, dtype=torch.long)
    p[0::2] = torch.arange(0, m // 2)
    p[1::2] = torch.arange(m // 2, m)
    return p.to(DEV)


def _swizzle(s: torch.Tensor) -> torch.Tensor:
    """Per-expert 128x4 block-scale swizzle of `[E, M, C]` uint8 scales."""
    e, m, c = s.shape
    return s.reshape(e, m // 128, 4, 32, c // 4, 4).permute(0, 1, 4, 3, 2, 5).reshape(e, m, c)


def _prepare(packed: torch.Tensor, scales: torch.Tensor, perm: torch.Tensor):
    """Shuffle the rows of a stack and its scales, then swizzle the scales."""
    w = torch.index_select(packed, 1, perm).contiguous()
    s = torch.index_select(scales.view(torch.uint8), 1, perm)
    return w, _swizzle(s).contiguous().view(torch.float8_e4m3fn)


_STACK = {}


def _stack(inter=INTER):
    """GLM-5.2's expert stack at MoE TP16 -- 256 experts, H 6144, I 128 --
    built once: the kernel operands and the reference's `(codes, scales)`."""
    if inter not in _STACK:
        gen = torch.Generator(device=DEV).manual_seed(5216)
        up_p, up_c, up_s = _rand_nvfp4(EXPERTS, inter, HIDDEN, gen)
        gt_p, gt_c, gt_s = _rand_nvfp4(EXPERTS, inter, HIDDEN, gen)
        dn_p, dn_c, dn_s = _rand_nvfp4(EXPERTS, HIDDEN, inter, gen)
        fc1_perm = _interleave_perm(2 * inter)[_blk32_perm(2 * inter)]
        w1, s1 = _prepare(torch.cat([up_p, gt_p], dim=1), torch.cat([up_s, gt_s], dim=1), fc1_perm)
        w2, s2 = _prepare(dn_p, dn_s, _blk32_perm(HIDDEN))
        _STACK[inter] = dict(
            gemm1_weights=w1,
            gemm1_weights_scale=s1,
            gemm2_weights=w2,
            gemm2_weights_scale=s2,
            up=(up_c, up_s),
            gate=(gt_c, gt_s),
            down=(dn_c, dn_s),
        )
    return _STACK[inter]


def _router(gen: torch.Generator):
    """Router weight and bias. The weight sits on a 2^-8 grid within +-1/32 so
    that, against a router input on a quarter grid, every partial sum of a
    router GEMV is exact in fp32: any summation order -- the kernel's, torch's
    -- gives the same logits."""
    weight = (
        torch.randint(-8, 9, (EXPERTS, HIDDEN), device=DEV, generator=gen).float() * 2.0**-8
    ).to(torch.bfloat16)
    bias = (torch.randn(EXPERTS, device=DEV, generator=gen) * 0.05).to(torch.bfloat16)
    return weight, bias


def _router_input(tokens: int, gen: torch.Generator) -> torch.Tensor:
    """bf16 router input on a quarter grid in [-1, 1] (see `_router`)."""
    ints = torch.randint(-4, 5, (tokens, HIDDEN), device=DEV, generator=gen)
    return (ints.float() * 0.25).to(torch.bfloat16)


def _input_global_scale(x: torch.Tensor) -> torch.Tensor:
    """The FC1-input global scale a checkpoint's `input_scale` encodes."""
    return (E4M3_MAX * E2M1_MAX / x.float().abs().amax()).reshape(1).to(torch.float32)


def _quantize(x: torch.Tensor, g1: torch.Tensor):
    """The experts' NVFP4 input, as the target quantizes it: linear scales."""
    xq, xsf = torch.ops.trtllm.fp4_quantize(x, g1, SV, False, False)
    return xq, xsf.view(torch.float8_e4m3fn)


def _scalars(xq, xsf, chosen, g1: torch.Tensor, inter=INTER):
    """The three per-expert scalars for weights with a global scale of 1, and
    `g2` calibrated on the true FC1 activation of the chosen experts."""
    stack = _stack(inter)
    x = _dequant_act(xq, xsf) / g1
    amax = torch.zeros((), device=DEV)
    with true_fp32_matmul():
        for e in chosen.unique().tolist():
            tok = (chosen == e).any(dim=1).nonzero(as_tuple=True)[0]
            up = x[tok] @ _dequant(*_slice(stack["up"], e)).t()
            gate = x[tok] @ _dequant(*_slice(stack["gate"], e)).t()
            amax = torch.maximum(amax, (up * gate * torch.sigmoid(gate)).abs().max())
    g2 = E4M3_MAX * E2M1_MAX / amax
    a1 = torch.full((EXPERTS,), 1.0, device=DEV) / g1
    return a1 * g2, a1, torch.full((EXPERTS,), 1.0, device=DEV) / g2


_E2M1 = None


def _dequant(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    global _E2M1
    if _E2M1 is None:
        _E2M1 = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
            dtype=torch.float32,
            device=DEV,
        )
    return _E2M1[codes.long()] * scales.float().repeat_interleave(SV, dim=1)


def _slice(operand, e: int):
    codes, scales = operand
    return codes[e], scales[e]


def _dequant_act(xq: torch.Tensor, xsf: torch.Tensor) -> torch.Tensor:
    codes = torch.stack([xq & 0x0F, xq >> 4], dim=-1).reshape(xq.shape[0], -1)
    return _dequant(codes, xsf.view(torch.float8_e4m3fn).reshape(xq.shape[0], -1))


def _selection(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """`(logits, top-8 by score + bias, smallest 8th-to-9th score gap)`."""
    with true_fp32_matmul():
        logits = x.float() @ weight.float().t()
    sel = 0.5 * torch.tanh(0.5 * logits) + 0.5 + bias.float()
    top = torch.topk(sel, TOP_K + 1, dim=-1)
    margin = (top.values[:, TOP_K - 1] - top.values[:, TOP_K]).min().item()
    return logits, top.indices[:, :TOP_K], margin


def _case(tokens: int, seed: int, clear_margin: bool = False, inter=INTER) -> dict:
    """One call's inputs. `clear_margin` redraws (deterministically, from the
    next seeds) a router input whose expert choice sits within
    `SELECTION_MARGIN` of a tie, which only a comparison against the
    reference needs."""
    for attempt in range(16):
        gen = torch.Generator(device=DEV).manual_seed(seed + attempt)
        weight, bias = _router(gen)
        x = _router_input(tokens, gen)
        logits, chosen, margin = _selection(x, weight, bias)
        if not clear_margin or margin > SELECTION_MARGIN:
            break
    else:
        raise AssertionError(f"no draw from seed {seed} clears the selection margin")
    g1 = _input_global_scale(x)
    xq, xsf = _quantize(x, g1)
    s1, sg, s2 = _scalars(xq, xsf, chosen, g1, inter)
    return dict(
        x=x,
        g1=g1,
        xq=xq,
        xsf=xsf,
        logits=logits,
        router_weight=weight,
        routing_bias=bias,
        output1_scale_scalar=s1,
        output1_scale_gate_scalar=sg,
        output2_scale_scalar=s2,
        **({"intermediate": inter} if inter != INTER else {}),
    )


# ── the call ──────────────────────────────────────────────────────────────


def _call_args(
    case: dict,
    *,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    fused_router: bool,
    workspace=None,
    event=None,
    specialized_gemm2: bool,
    do_finalize: bool = False,
    routing_logits=None,
    output=None,
):
    """The target's call, every argument explicit."""
    inter = case.get("intermediate", INTER)
    stack = _stack(inter)
    tokens = case["x"].shape[0]
    if routing_logits is None:
        # With the fused router only the shape is read; filled so that an
        # autotuning run would profile on plausible routing.
        routing_logits = (
            torch.randn(tokens, EXPERTS, device=DEV) if fused_router else case["logits"]
        )
    return dict(
        routing_logits=routing_logits,
        routing_bias=case["routing_bias"],
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=stack["gemm1_weights"],
        gemm1_weights_scale=stack["gemm1_weights_scale"],
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=stack["gemm2_weights"],
        gemm2_weights_scale=stack["gemm2_weights_scale"],
        gemm2_bias=None,
        output1_scale_scalar=case["output1_scale_scalar"],
        output1_scale_gate_scalar=case["output1_scale_gate_scalar"],
        output2_scale_scalar=case["output2_scale_scalar"],
        num_experts=EXPERTS,
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        intermediate_size=inter,
        local_expert_offset=0,
        local_num_experts=EXPERTS,
        routed_scaling_factor=ROUTED_SCALE,
        routing_method_type=ROUTING_METHOD_DEEPSEEK_V3,
        do_finalize=do_finalize,
        act_type=ACT_TYPE_SWIGLU,
        topk_weights=None,
        topk_ids=None,
        output=output,
        tune_max_num_tokens=8192,
        use_dp=False,
        hidden_states_ready_event=None if event is None else event.cuda_event,
        router_hidden_states=case["x"] if fused_router else None,
        router_weight=case["router_weight"] if fused_router else None,
        router_workspace=workspace if fused_router else None,
        prefer_specialized_gemm2=specialized_gemm2,
    )


def _call(case: dict, **kwargs):
    return op(**_call_args(case, **kwargs))


def _quantize_aside(case: dict, side: torch.cuda.Stream, delay: bool, out=None):
    """The target's input path: quantize on `side`, forked after the current
    stream's work, and return `(xq, xsf, event)` with `event` recorded there.
    `out` holds buffers to write instead of fresh tensors; `delay` sleeps on
    `side` first."""
    forked = torch.cuda.Event()
    ready = torch.cuda.Event()
    forked.record()
    with torch.cuda.stream(side):
        side.wait_event(forked)
        if delay:
            torch.cuda._sleep(SLEEP_CYCLES)
        xq, xsf = _quantize(case["x"], case["g1"])
        if out is not None:
            out[0].copy_(xq)
            out[1].copy_(xsf)
            xq, xsf = out
        ready.record(side)
    return xq, xsf, ready


def _valid_rows(outs):
    """The unfinalized outputs as `(rows each slot reads, weights, map)`."""
    rows, weights, perm = outs
    return rows[perm.long().reshape(-1)], weights, perm


def _finalize(outs) -> torch.Tensor:
    """`sum_k weights[t, k] * rows[map[t, k]]` in fp32, rounded once: what the
    post-MoE hop computes."""
    rows, weights, perm = outs
    gathered = rows.float()[perm.long()]
    return (gathered * weights.float().unsqueeze(-1)).sum(dim=1).to(torch.bfloat16)


def _assert_same(a, b, what: str) -> None:
    if len(a) == len(b) == 1:
        assert torch.equal(a[0], b[0]), f"{what}: finalized outputs differ"
        return
    for x, y, name in zip(_valid_rows(a), _valid_rows(b), ("rows", "weights", "map")):
        assert torch.equal(x, y), f"{what}: {name} differ"


def _kernels(fn) -> list[str]:
    """Names of the kernels `fn` launches, in the order they start."""
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    launched = [
        e
        for e in prof.events()
        if str(e.device_type) == "DeviceType.CUDA" and e.device_time_total > 0
    ]
    return [e.name for e in sorted(launched, key=lambda e: e.time_range.start)]


def _small_k_covers(tokens: int, inter: int = 128) -> bool:
    """Owned FC2 gate: K128 allows 15 units/64 tiles, K256 allows 29 units/128 tiles."""
    tiles = min(tokens * TOP_K, EXPERTS)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    units = tiles * HIDDEN // 128
    return tiles <= (64 if inter == 128 else 128) and -(-units // sms) <= (
        15 if inter == 128 else 29
    )


# ── tests ─────────────────────────────────────────────────────────────────


_MUTATION_CASES = [
    pytest.param(False, False, False, id="no_mutable_inputs"),
    pytest.param(True, False, False, id="workspace"),
    pytest.param(False, True, False, id="finalized"),
    pytest.param(False, True, True, id="caller_output"),
    pytest.param(True, True, True, id="workspace_and_caller_output"),
]


def _mutation_case_args(fused_router, do_finalize, caller_output):
    case = _case(2, seed=4800)
    return _call_args(
        case,
        hidden_states=case["xq"],
        hidden_states_scale=case["xsf"],
        fused_router=fused_router,
        workspace=make_fused_router_workspace(DEV) if fused_router else None,
        specialized_gemm2=True,
        do_finalize=do_finalize,
        routing_logits=case["logits"],
        output=torch.zeros(2, HIDDEN, device=DEV, dtype=torch.bfloat16) if caller_output else None,
    )


@pytest.mark.parametrize("fused_router,do_finalize,caller_output", _MUTATION_CASES)
def test_custom_op_mutation_schema_and_fake(
    fused_router, do_finalize, caller_output, monkeypatch
) -> None:
    """Optional mutations and all return modes agree with the registered schema and fake."""
    args = _mutation_case_args(fused_router, do_finalize, caller_output)
    # SchemaCheckMode compares every input with allclose, whose float8 CUDA
    # arithmetic is unsupported. Cast only its comparison operands; the op
    # must receive the original float8 scales to satisfy its dtype contract.
    allclose = torch.allclose

    def allclose_with_float8(actual, expected, *comparison_args, **comparison_kwargs):
        if actual.dtype == expected.dtype == torch.float8_e4m3fn:
            actual, expected = actual.float(), expected.float()
        return allclose(actual, expected, *comparison_args, **comparison_kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "allclose", allclose_with_float8)
        torch.library.opcheck(
            torch.ops.trtllm.glm_routed_experts.default,
            (),
            args,
            test_utils=("test_schema",),
        )
    torch.library.opcheck(
        torch.ops.trtllm.glm_routed_experts.default,
        (),
        args,
        test_utils=("test_faketensor",),
    )

    workspace = args["router_workspace"]
    epoch = None if workspace is None else workspace[_EPOCH_WORD].item()
    actual = op(**args)
    assert isinstance(actual, list)
    assert len(actual) == (1 if do_finalize else 3)
    if caller_output:
        assert actual[0].shape == (0,)
        # The public result remains an alias-free sentinel; the supplied buffer
        # contains the same values as a call that allocates its finalized output.
        expected = op(**{**args, "output": None})[0]
        torch.testing.assert_close(args["output"], expected, rtol=0, atol=0)
    elif do_finalize:
        assert actual[0].shape == (2, HIDDEN)
    else:
        assert actual[0].shape[1] == HIDDEN
        assert actual[1].dtype == torch.bfloat16
    if workspace is not None:
        assert workspace[_EPOCH_WORD].item() == epoch + 1 + int(caller_output)


@pytest.mark.parametrize("fused_router,do_finalize,caller_output", _MUTATION_CASES)
def test_custom_op_functionalization(fused_router, do_finalize, caller_output) -> None:
    """AOT keeps both mutations and subsequent reads, including repeated calls."""
    eager_args = _mutation_case_args(fused_router, do_finalize, caller_output)
    compiled_args = dict(eager_args)
    for name in ("router_workspace", "output"):
        if eager_args[name] is not None:
            compiled_args[name] = eager_args[name].clone()

    def run(**args):
        result = op(**args)
        if not do_finalize:
            # Padding rows are uninitialized; only mapped rows have defined values.
            values = _valid_rows(result)
        elif caller_output:
            values = (args["output"].clone(), result[0])
        else:
            values = tuple(result)
        if fused_router:
            values = (*values, args["router_workspace"][_EPOCH_WORD].clone())
        return values

    # The unfinalized row capacity depends on an opaque autotuned tile.
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=not do_finalize):
        compiled = torch.compile(run, backend="aot_eager", fullgraph=True)
        for launch in (1, 2):
            expected = run(**eager_args)
            actual = compiled(**compiled_args)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if caller_output:
                torch.testing.assert_close(
                    compiled_args["output"], eager_args["output"], rtol=0, atol=0
                )
            if fused_router:
                assert compiled_args["router_workspace"][_EPOCH_WORD].item() == launch


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    """Each cell as the target makes the call, against the reference.

    The late-join cells quantize on a side stream behind a sleep, so the
    result also proves FC1 waited for the input.
    """
    certifying(op)
    spec = cell.spec
    geometry = (spec["hidden"], spec["intermediate"], spec["num_experts"], spec["top_k"])
    assert geometry in (
        (HIDDEN, INTER, EXPERTS, TOP_K),
        (HIDDEN, 256, EXPERTS, TOP_K),
        (HIDDEN, 512, EXPERTS, TOP_K),
    ), geometry
    assert spec["routed_scaling_factor"] == ROUTED_SCALE and spec["local_expert_offset"] == 0
    case = _case(
        spec["tokens"],
        seed=zlib.crc32(cell.why.encode()) % 2**31,
        clear_margin=True,
        inter=spec["intermediate"],
    )
    workspace = make_fused_router_workspace(DEV) if spec["fused_router"] else None
    if spec["late_join"]:
        side = torch.cuda.Stream()
        xq, xsf, ready = _quantize_aside(case, side, delay=True)
    else:
        xq, xsf, ready = case["xq"], case["xsf"], None
    outs = _call(
        case,
        hidden_states=xq,
        hidden_states_scale=xsf,
        fused_router=spec["fused_router"],
        workspace=workspace,
        event=ready,
        specialized_gemm2=spec["specialized_gemm2"],
    )
    torch.cuda.synchronize()

    stack = _stack(spec["intermediate"])
    ref, _, ref_weights = op.reference(
        routing_logits=None if spec["fused_router"] else case["logits"],
        routing_bias=case["routing_bias"],
        hidden_states=case["xq"],
        hidden_states_scale=case["xsf"],
        up=stack["up"],
        gate=stack["gate"],
        down=stack["down"],
        output1_scale_scalar=case["output1_scale_scalar"],
        output1_scale_gate_scalar=case["output1_scale_gate_scalar"],
        output2_scale_scalar=case["output2_scale_scalar"],
        top_k=TOP_K,
        local_expert_offset=0,
        routed_scaling_factor=ROUTED_SCALE,
        router_hidden_states=case["x"] if spec["fused_router"] else None,
        router_weight=case["router_weight"] if spec["fused_router"] else None,
    )
    # Each token's weights, as a set: the kernel stores them in its top-k order.
    torch.testing.assert_close(
        outs[1].float().sort(dim=-1).values,
        ref_weights.float().sort(dim=-1).values,
        rtol=2.0**-7,
        atol=0.0,
    )
    _compare(_finalize(outs), ref)


@pytest.mark.parametrize("tokens", range(1, 9))
def test_fused_router_matches_the_router_gemm(tokens) -> None:
    """The fused router computes the logits of `dsv3_router_gemm_op` bit for
    bit and routes them as the routing kernel does: every output is bitwise
    the one the op gives when routed from those logits. Over every row count
    the kernel has an instance for, on an off-grid router input where
    summation order shows."""
    gen = torch.Generator(device=DEV).manual_seed(4000 + tokens)
    case = _case(tokens, seed=4100 + tokens)
    x = torch.randn(tokens, HIDDEN, device=DEV, generator=gen).to(torch.bfloat16)
    weight = (torch.randn(EXPERTS, HIDDEN, device=DEV, generator=gen) * 0.02).to(torch.bfloat16)
    case.update(x=x, router_weight=weight)
    logits = torch.ops.trtllm.dsv3_router_gemm_op(x, weight.t(), None, torch.float32)
    workspace = make_fused_router_workspace(DEV)
    common = dict(
        hidden_states=case["xq"], hidden_states_scale=case["xsf"], specialized_gemm2=False
    )
    fused = _call(case, fused_router=True, workspace=workspace, **common)
    routed = _call(case, fused_router=False, routing_logits=logits, **common)
    _assert_same(fused, routed, f"{tokens} rows")


@pytest.mark.parametrize("tokens,inter", ((1, 128), (8, 128), (4, 256), (8, 256)))
@pytest.mark.parametrize("do_finalize", (False, True))
def test_specialized_gemm2_matches_the_cubin(tokens, inter, do_finalize) -> None:
    """Owned FC2 matches generic FC2, including in-place caller-output finalization."""
    case = _case(tokens, seed=4200 + tokens, inter=inter)
    common = dict(
        hidden_states=case["xq"],
        hidden_states_scale=case["xsf"],
        fused_router=False,
        do_finalize=do_finalize,
    )
    output = torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=DEV) if do_finalize else None
    specialized = _call(case, specialized_gemm2=True, output=output, **common)
    cubin = _call(case, specialized_gemm2=False, **common)
    if output is None:
        _assert_same(specialized, cubin, f"{tokens} rows, K{inter}")
    else:
        # The mutated caller output cannot alias a custom-op return.
        assert len(specialized) == 1 and specialized[0].numel() == 0
        _assert_same([output], cubin, f"{tokens} rows, K{inter}, caller output")
        allocated = _call(case, specialized_gemm2=True, **common)
        _assert_same(allocated, cubin, f"{tokens} rows, K{inter}, allocated output")
    marker = f"gemm2K{inter}Kernel"
    names = _kernels(lambda: _call(case, specialized_gemm2=True, **common))
    assert any(marker in name for name in names) == _small_k_covers(tokens, inter), names
    names = _kernels(lambda: _call(case, specialized_gemm2=False, **common))
    assert not any("gemm2K128Kernel" in name or "gemm2K256Kernel" in name for name in names), names


def test_c1_call_launches_three_kernels() -> None:
    """The c1 call is the fused router, FC1 and FC2: no router GEMV, no
    separate routing kernel."""
    case = _case(8, seed=4300)
    workspace = make_fused_router_workspace(DEV)
    shape_only = torch.randn(8, EXPERTS, device=DEV)
    names = _kernels(
        lambda: _call(
            case,
            hidden_states=case["xq"],
            hidden_states_scale=case["xsf"],
            fused_router=True,
            workspace=workspace,
            specialized_gemm2=True,
            routing_logits=shape_only,
        )
    )
    assert len(names) == 3, names
    assert "routingFusedRouterKernel" in names[0], names
    assert ("gemm2K128Kernel" in names[2]) == _small_k_covers(8), names


def test_late_join_waits_for_the_input() -> None:
    """FC1 reads the input only after the event: the buffers handed to the op
    hold a different activation's quantization until the side stream, held
    back by a sleep, overwrites them. Routing launches without waiting."""
    case = _case(8, seed=4400)
    stale = _case(8, seed=4401)
    workspace = make_fused_router_workspace(DEV)
    expected = _call(
        case,
        hidden_states=case["xq"],
        hidden_states_scale=case["xsf"],
        fused_router=True,
        workspace=workspace,
        specialized_gemm2=True,
    )
    buffers = (stale["xq"].clone(), stale["xsf"].clone())
    side = torch.cuda.Stream()
    xq, xsf, ready = _quantize_aside(case, side, delay=True, out=buffers)
    late = _call(
        case,
        hidden_states=xq,
        hidden_states_scale=xsf,
        fused_router=True,
        workspace=workspace,
        event=ready,
        specialized_gemm2=True,
    )
    torch.cuda.synchronize()
    _assert_same(late, expected, "late join")


@pytest.mark.parametrize("tokens,inter", [(8, 128), (12, 128), (18, 128), (4, 256), (8, 256)])
def test_fused_call_replays_in_a_cuda_graph(tokens, inter) -> None:
    """The c1 step captured as the target runs it -- input quantized on a side
    stream, joined inside the op -- replays the eager result on new inputs, and
    every launch leaves the router workspace ready for the next one."""
    cases = [_case(tokens, seed=4500 + i, inter=inter) for i in range(3)]
    static = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in cases[0].items()}
    workspace = make_fused_router_workspace(DEV)
    side = torch.cuda.Stream()

    def step():
        xq, xsf, ready = _quantize_aside(static, side, delay=False)
        return _call(
            static,
            hidden_states=xq,
            hidden_states_scale=xsf,
            fused_router=True,
            workspace=workspace,
            event=ready,
            specialized_gemm2=tokens <= 8,
            routing_logits=static["logits"],
        )

    step()  # load the kernels before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()

    launches = 1
    for case in cases:
        for key, value in case.items():
            if isinstance(value, torch.Tensor):
                static[key].copy_(value)
        graph.replay()
        launches += 1
        torch.cuda.synchronize()
        eager = _call(
            case,
            hidden_states=case["xq"],
            hidden_states_scale=case["xsf"],
            fused_router=True,
            workspace=workspace,
            specialized_gemm2=tokens <= 8,
        )
        launches += 1
        torch.cuda.synchronize()
        _assert_same(captured, eager, "replay")
    words = workspace.view(torch.int32)
    epoch = int(words[_EPOCH_WORD].item())
    assert epoch == launches, (epoch, launches)
    current = words[(epoch & 1) * _BUFFER_WORDS : ((epoch & 1) + 1) * _BUFFER_WORDS]
    assert bool((current == -(2**31)).all()), "the next launch's exchange buffer is not idle"


def test_fused_router_refuses_more_than_24_rows() -> None:
    """Past 24 rows the fused router is a loud error; the caller routes from
    the router GEMM's logits instead (the 32-row cell)."""
    case = _case(25, seed=4600)
    with pytest.raises(RuntimeError, match="fused router"):
        _call(
            case,
            hidden_states=case["xq"],
            hidden_states_scale=case["xsf"],
            fused_router=True,
            workspace=make_fused_router_workspace(DEV),
            specialized_gemm2=True,
        )


# ── guards ────────────────────────────────────────────────────────────────


def test_guard_refuses_a_strided_router_input() -> None:
    case = _case(2, seed=4700)
    case["x"] = torch.cat([case["x"], case["x"]], dim=1)[:, ::2]
    with pytest.raises(AssertionError, match="router_hidden_states must be contiguous"):
        with validating(op):
            _call(
                case,
                hidden_states=case["xq"],
                hidden_states_scale=case["xsf"],
                fused_router=True,
                workspace=make_fused_router_workspace(DEV),
                specialized_gemm2=True,
            )


def test_guard_refuses_a_zero_filled_workspace() -> None:
    """A zero-filled workspace reads as published zero sums: routing on zero
    logits, with no error."""
    case = _case(2, seed=4701)
    workspace = torch.zeros(FUSED_ROUTER_WORKSPACE_WORDS, dtype=torch.int32, device=DEV)
    with pytest.raises(AssertionError, match="between-launch state"):
        with validating(op):
            _call(
                case,
                hidden_states=case["xq"],
                hidden_states_scale=case["xsf"],
                fused_router=True,
                workspace=workspace,
                specialized_gemm2=True,
            )


def test_guard_refuses_an_unrecorded_event() -> None:
    """An event never recorded has no cudaEvent_t yet: FC1 would not wait."""
    case = _case(2, seed=4702)
    with pytest.raises(AssertionError, match="null cudaEvent_t"):
        with validating(op):
            _call(
                case,
                hidden_states=case["xq"],
                hidden_states_scale=case["xsf"],
                fused_router=True,
                workspace=make_fused_router_workspace(DEV),
                event=torch.cuda.Event(),
                specialized_gemm2=True,
            )


@pytest.mark.parametrize("tokens", [9, 12, 16, 17, 18, 24])
def test_wide_fused_router_matches_separate_routing(tokens) -> None:
    """Exact-grid router operands isolate routing/tile semantics from GEMM rounding."""
    case = _case(tokens, seed=6100 + tokens, clear_margin=True)
    common = dict(
        hidden_states=case["xq"], hidden_states_scale=case["xsf"], specialized_gemm2=False
    )
    fused = _call(case, fused_router=True, workspace=make_fused_router_workspace(DEV), **common)
    separate = _call(case, fused_router=False, **common)
    torch.cuda.synchronize()
    # Physical tile order may differ: compare each token's finalized value and routing weights.
    _compare(_finalize(fused), _finalize(separate))
    assert torch.equal(fused[1].sort(dim=-1).values, separate[1].sort(dim=-1).values)


def test_fused_router_workspace_alternates_small_and_wide_calls() -> None:
    """Both kernels rotate/reset the same16388-word workspace across changing row widths."""
    workspace = make_fused_router_workspace(DEV)
    for launch, tokens in enumerate((8, 12, 1, 18, 6, 24, 8), start=1):
        case = _case(tokens, seed=6200 + tokens, clear_margin=True)
        common = dict(
            hidden_states=case["xq"], hidden_states_scale=case["xsf"], specialized_gemm2=False
        )
        fused = _call(case, fused_router=True, workspace=workspace, **common)
        separate = _call(case, fused_router=False, **common)
        torch.cuda.synchronize()
        _compare(_finalize(fused), _finalize(separate))
        epoch = int(workspace[_EPOCH_WORD].item())
        assert epoch == launch
        current = workspace[(epoch & 1) * _BUFFER_WORDS : ((epoch & 1) + 1) * _BUFFER_WORDS]
        assert bool((current == -(2**31)).all())
