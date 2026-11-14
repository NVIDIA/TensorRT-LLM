# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode-sized CuTe DSL FC2 (decode_moe CUTE_FC2) after the FC1-only trtllm-gen runner vs the runner's own FC2
cubin: bit-identical un-finalized outputs (gemm2 rows, expert weights, expanded -> permuted index)."""

from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
from tensorrt_llm._utils import is_blackwell_datacenter

pytestmark = pytest.mark.skipif(not is_blackwell_datacenter(), reason="needs SM 10.0 / 10.3")

_TILE = 8


class _Shape(NamedTuple):
    """An MoE layer as one rank runs it: ``local_picks`` of a token's ``top_k`` experts are among the rank's
    ``local`` experts (ids 0 .. local - 1), the others are another rank's (ids >= local, ignored)."""

    num_experts: int
    local: int
    hidden: int
    inter: int
    top_k: int
    local_picks: int


_ALL_LOCAL = _Shape(384, 384, 5120, 576, 6, 6)
# An expert-parallel rank of a 512-expert top-10 layer (128 local experts), with all or some picks local.
_EP_LOCAL = _Shape(128, 128, 4096, 512, 10, 10)
_EP_REMOTE = _Shape(512, 128, 4096, 512, 10, 3)


def _runner():
    return torch.classes.trtllm.FP4BlockScaleMoERunner(0)


def _inputs(m, shape, kind):
    torch.manual_seed(11 + m)
    dev = "cuda"
    e, hidden, inter, top_k = shape.local, shape.hidden, shape.inter, shape.top_k
    x = torch.randn(m, hidden, device=dev, dtype=torch.bfloat16)
    gs = (448.0 * 6.0 / x.float().abs().amax()).reshape(1).float()
    xq, xsf = torch.ops.trtllm.fp4_quantize(x, gs, 16, False, False)
    ids = []
    for _ in range(m):
        loc = torch.randperm(e, device=dev)[: shape.local_picks]
        rest = e + torch.randperm(shape.num_experts - e, device=dev)[: top_k - shape.local_picks]
        ids.append(torch.cat([loc, rest])[torch.randperm(top_k, device=dev)])
    ids = torch.stack(ids).int()
    w = torch.softmax(torch.randn(m, top_k, device=dev), -1).to(torch.bfloat16)

    def u8(*s):
        b = torch.randint(0, 256, s, dtype=torch.uint8, device=dev)
        return b & 0x88 if kind == "signed_zeros" else b

    def sf(lo, hi, *s):
        return torch.randint(lo, hi, s, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)

    lo, hi = (0x00, 0x7F) if kind == "full_sf" else (0x28, 0x41)
    w13, w13s = u8(e, 2 * inter, hidden // 2), sf(0x28, 0x41, e, 2 * inter, hidden // 16)
    w2, w2s = u8(e, hidden, inter // 2), sf(lo, hi, e, hidden, inter // 16)
    one = torch.full((e,), 1e-4, device=dev)
    alpha2 = torch.rand(e, device=dev) * 1e-3 + 1e-4
    clamp = torch.full((e,), 1e5, device=dev)
    return (
        ids,
        w,
        (xq, xsf.flatten().view(torch.float8_e4m3fn), w13, w13s, clamp, w2, w2s, one, alpha2),
    )


def _common(shape, ins, ids, w, tactic):
    xq, xsf, w13, w13s, clamp, w2, w2s, one, alpha2 = ins
    return (
        None,
        None,
        xq,
        xsf,
        w13,
        w13s,
        None,
        None,
        None,
        clamp,
        w2,
        w2s,
        None,
        one,
        one,
        alpha2,
        shape.num_experts,
        shape.top_k,
        None,
        None,
        shape.inter,
        0,
        shape.local,
        None,
        1,
        False,
        tactic,
        w,
        ids,
        None,
    )


def _assert_same_expert_rows(got, ref):
    """Same expert row per (token, k), each gathered through its own run's permutation (the routing kernel's
    permuted order may differ between runs). The returned expert weights are the runner's scratch when topk_weights
    are passed in, so they are not compared."""
    pg, pr = got[2].flatten(), ref[2].flatten()
    assert torch.equal(pg >= 0, pr >= 0)
    valid = pr >= 0
    assert int(valid.sum()) > 0
    assert torch.equal(
        got[0][pg[valid].long()].view(torch.int16), ref[0][pr[valid].long()].view(torch.int16)
    )


def _bitwise_tactics(runner, m, shape):
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_moe import cute_fc2_reproduces

    cfg = runner.get_valid_configs(shape.top_k, shape.hidden, shape.inter, shape.local, m)
    tactics = [
        t
        for t in cfg
        if t[0] == _TILE and cute_fc2_reproduces(runner.get_gemm2_kernel_options(t[0], t[1]))
    ]
    assert tactics, "no tile-8 tactic whose FC2 cubin the CuTe FC2 reproduces"
    return tactics


def test_fc2_cubin_selection_is_structural():
    """The CuTe FC2 takes the tile-8 FC2 cubins with M128 N8 K64 MMAs on one CTA per tile, whatever their tile K,
    stages, unrolling or scheduler, and no split-K or other MMA shape or dtype; decided from the cubin's options as
    the C++ runner reports them, not from its name."""
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_moe import cute_fc2_reproduces

    def options(**over):
        base = dict(
            dtypeA="E2m1",
            dtypeB="E2m1",
            dtypeC="Bfloat16",
            dtypeAcc="Fp32",
            tileM="128",
            tileN="8",
            tileK="256",
            mmaM="128",
            mmaN="8",
            mmaK="64",
            clusterDimX="1",
            clusterDimY="1",
            clusterDimZ="1",
            numSlicesForSplitK="1",
        )
        return {**base, **over}

    assert cute_fc2_reproduces(options())
    assert cute_fc2_reproduces(options(tileK="512"))
    assert not cute_fc2_reproduces(options(clusterDimZ="2", numSlicesForSplitK="2"))
    assert not cute_fc2_reproduces(options(numSlicesForSplitK="2"))
    assert not cute_fc2_reproduces(options(mmaK="128"))
    assert not cute_fc2_reproduces(options(tileN="16", mmaN="16"))
    assert not cute_fc2_reproduces(options(dtypeA="E4m3"))
    assert not cute_fc2_reproduces({})


def test_fc2_cubin_options_match_the_cubin_names():
    """Every FC2 cubin the runner offers reports the options its name encodes (tile, MMA shape, cluster, split-K,
    dtypes), so the options-based selection takes exactly the tile-8 M128 N8 K64 single-CTA cubins."""
    import re

    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_moe import cute_fc2_reproduces
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    old_name = re.compile(
        r"^bmm_(?P<out>[A-Za-z0-9]+)_(?P<ab>[A-Za-z0-9]+)_(?P<acc>[A-Za-z0-9]+)_Ab\d+_Bb\d+"
        r"_t(?P<tile>\d+x\d+)x\d+(?:u2)?_s\d+_et\d+x\d+_m(?P<mma>\d+x\d+x\d+)_c(?P<cluster>\d+x\d+x\d+)"
        r"_r[MN]_(?P<split>splitK\d+_)?TN_transOut_"
    )
    runner = torch.classes.trtllm.FP4BlockScaleMoERunner(int(ActType_TrtllmGen.SwiGlu))
    picked = 0
    # (local experts, intermediate): this file's EP shape and Qwen3.5-397B's TEP8 / TEP16 experts (hidden 4096, top-10).
    for local, inter in ((_EP_LOCAL.local, _EP_LOCAL.inter), (64, 1024), (32, 1024)):
        for m in (1, 8, 32, 128):
            configs = runner.get_valid_configs(_EP_LOCAL.top_k, _EP_LOCAL.hidden, inter, local, m)
            for tile, config in configs:
                opts = runner.get_gemm2_kernel_options(tile, config)
                name = runner.get_gemm2_kernel_name(tile, config)
                assert f"_t{opts['tileM']}x{opts['tileN']}x{opts['tileK']}" in name, (name, opts)
                assert f"_m{opts['mmaM']}x{opts['mmaN']}x{opts['mmaK']}_" in name, (name, opts)
                cluster = f"_c{opts['clusterDimX']}x{opts['clusterDimY']}x{opts['clusterDimZ']}_"
                assert cluster in name, (name, opts)
                split = re.search(r"_splitK(\d+)_", name)
                assert opts["numSlicesForSplitK"] == (split.group(1) if split else "1"), (
                    name,
                    opts,
                )
                assert name.startswith(
                    f"bmm_{opts['dtypeC']}_{opts['dtypeA']}{opts['dtypeB']}_{opts['dtypeAcc']}_"
                )
                picked += cute_fc2_reproduces(opts)
                # The name-parsing selection this replaced (d7b87c20fc) picks the same cubins.
                m_old = old_name.match(name)
                by_name = (
                    m_old is not None
                    and (m_old["out"], m_old["ab"], m_old["acc"])
                    == ("Bfloat16", "E2m1E2m1", "Fp32")
                    and m_old["tile"] == "128x8"
                    and m_old["mma"] == "128x8x64"
                    and m_old["cluster"] == "1x1x1"
                    and m_old["split"] is None
                )
                assert cute_fc2_reproduces(opts) == by_name, (name, opts)
    assert picked


@pytest.mark.parametrize("kind", ["random", "signed_zeros", "full_sf"])
@pytest.mark.parametrize(
    "m,shape",
    [(m, _ALL_LOCAL) for m in (1, 6, 8, 16)]
    + [(m, _ALL_LOCAL._replace(num_experts=192, local=192)) for m in (1, 6, 8, 16)]
    + [(m, s) for s in (_EP_LOCAL, _EP_REMOTE) for m in (56, 77, 112, 126)],
)
def test_fc1_only_then_cute_fc2_is_bitwise(m, shape, kind):
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_moe import fc1_only_args, run_cute_fc2

    runner = _runner()
    ids, w, ins = _inputs(m, shape, kind)
    fc2_args = SimpleNamespace(
        gemm2_weights=ins[5], gemm2_weights_scale=ins[6], output2_scale_scalar=ins[8]
    )
    for tactic in _bitwise_tactics(runner, m, shape):
        common = _common(shape, ins, ids, w, list(tactic))
        ref = runner.run_moe(*common)
        got = run_cute_fc2(runner.run_moe_fc1(*fc1_only_args(common)), fc2_args, None)
        torch.cuda.synchronize()
        _assert_same_expert_rows(got, ref)


def test_fc1_only_requires_unfinalized():
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_moe import fc1_only_args

    runner = _runner()
    ids, w, ins = _inputs(6, _ALL_LOCAL, "random")
    tactic = _bitwise_tactics(runner, 6, _ALL_LOCAL)[0]
    common = list(_common(_ALL_LOCAL, ins, ids, w, tactic))
    assert len(fc1_only_args(common)) == len(common) - 3
    common[25] = True  # do_finalize
    with pytest.raises(ValueError):
        fc1_only_args(common)


@pytest.mark.parametrize("decode_tuning", [True, False])
def test_runner_forward_dispatch(decode_tuning, monkeypatch):
    """FP4BlockScaleMoERunner.forward with a tile-8 tactic: the CuTe FC2 where the model declares its decode kernels,
    the cubin otherwise; same bits either way."""
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen import decode_moe

    runner = _runner()
    m, shape = 6, _ALL_LOCAL
    ids, w, (xq, xsf, w13, w13s, clamp, w2, w2s, one, alpha2) = _inputs(m, shape, "random")
    tactic = list(_bitwise_tactics(runner, m, shape)[0])
    args = (
        shape.num_experts,
        shape.top_k,
        None,
        None,
        shape.inter,
        0,
        shape.local,
        None,
        1,
        False,
        0,
    )
    inputs = [
        None,
        None,
        xq,
        xsf,
        w13,
        w13s,
        None,
        None,
        None,
        clamp,
        w2,
        w2s,
        None,
        one,
        one,
        alpha2,
        w,
        ids,
    ]
    ref = FP4BlockScaleMoERunner(*args).forward(inputs, tactic=tactic)
    called = []
    orig = decode_moe.run_cute_fc2
    monkeypatch.setattr(decode_moe, "run_cute_fc2", lambda *a: called.append(1) or orig(*a))
    got = FP4BlockScaleMoERunner(*args, decode_tuning=decode_tuning).forward(inputs, tactic=tactic)
    torch.cuda.synchronize()
    assert bool(called) == decode_tuning
    _assert_same_expert_rows(got, ref)


@pytest.mark.parametrize("prerouted", [False, True])
def test_fc1_only_args_drop_exactly_the_fc2_arguments(prerouted):
    """The FC1-only entries take run_moe's / run_moe_prerouted's arguments without gemm2_bias, do_finalize and
    output; the pre-routed tables and tile that follow output stay in order."""
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_moe import (
        DO_FINALIZE_ARG,
        GEMM2_BIAS_ARG,
        OUTPUT_ARG,
        fc1_only_args,
    )

    # run_moe: 30 arguments; run_moe_prerouted adds 6 routing tables and the routing tile.
    run_args = [f"arg{i}" for i in range(30 + (7 if prerouted else 0))]
    run_args[GEMM2_BIAS_ARG], run_args[DO_FINALIZE_ARG], run_args[OUTPUT_ARG] = None, False, None
    dropped = {GEMM2_BIAS_ARG, DO_FINALIZE_ARG, OUTPUT_ARG}
    assert fc1_only_args(run_args) == tuple(a for i, a in enumerate(run_args) if i not in dropped)
    for i, value in ((GEMM2_BIAS_ARG, "bias"), (DO_FINALIZE_ARG, True), (OUTPUT_ARG, "out")):
        bad = list(run_args)
        bad[i] = value
        with pytest.raises(ValueError):
            fc1_only_args(bad)


def test_fp4_moe_fake_refuses_the_lamport_output():
    """The ops' fake fails closed with a Lamport buffer: whether the rows come back in it depends on the kernel the
    host picks per call, which a trace cannot know."""
    import inspect

    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import _fp4_block_scale_moe_fake

    params = inspect.signature(_fp4_block_scale_moe_fake).parameters
    kwargs = {name: None for name, p in params.items() if p.default is inspect.Parameter.empty}
    kwargs.update(
        hidden_states=torch.empty((4, 64), dtype=torch.uint8, device="meta"),
        num_experts=8,
        top_k=2,
        do_finalize=False,
        lamport_fc2_out=torch.empty((16, 128), dtype=torch.bfloat16, device="meta"),
    )
    with pytest.raises(NotImplementedError, match="eager-only"):
        _fp4_block_scale_moe_fake(**kwargs)
    kwargs["lamport_fc2_out"] = None
    assert (
        len(_fp4_block_scale_moe_fake(**kwargs)) == 3
    )  # rows, top-k weights, expanded -> permuted


def test_both_fp4_ops_trace_through_the_one_fake():
    """The plain and the decode FP4 MoE ops register the one fake, which takes each op's own arguments (the decode
    op's state arguments included) and shapes the finalized output from the hidden states."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    from tensorrt_llm._torch.custom_ops import trtllm_gen_custom_ops as G

    decode_only = {"decode_spare_sms": 8, "decode_early_release_rows": 4, "decode_tuning": True}
    for op, extra in (
        (torch.ops.trtllm.fp4_block_scale_moe_runner, {}),
        (torch.ops.trtllm.fp4_block_scale_moe_decode_runner, decode_only),
    ):
        with FakeTensorMode():
            kwargs = {}
            for arg in op.default._schema.arguments:
                text = str(arg.type)
                if text == "Tensor":
                    kwargs[arg.name] = torch.empty((1,), dtype=torch.uint8, device="cuda")
                elif text in ("int", "bool"):
                    kwargs[arg.name] = 1 if text == "int" else False
                elif arg.has_default_value():
                    continue
                else:
                    kwargs[arg.name] = None
            kwargs.update(
                hidden_states=torch.empty((4, 64), dtype=torch.uint8, device="cuda"),
                do_finalize=True,
                **extra,
            )
            (out,) = op(**kwargs)
        assert out.shape == (4, 128) and out.dtype == torch.bfloat16, op
    assert G.fp4_block_scale_moe_runner._abstract_fn is G._fp4_block_scale_moe_fake
    assert G.fp4_block_scale_moe_decode_runner._abstract_fn is G._fp4_block_scale_moe_fake
