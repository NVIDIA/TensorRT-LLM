# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""sm_103 correctness tests of the routed_moe routed NVFP4 MoE against the native region.

Real GLM-5.2 NVFP4 weights (layer ``GLM_MOE_DSA_ROUTED_MOE_LAYER``, default 10, of TP8 rank
``GLM_MOE_DSA_ROUTED_MOE_RANK``, default 0) through the stock TRT-LLM-Gen leaf are the oracle.
Error budget (frozen before any performance tuning):
expert IDs and routing weights bit-exact; finalized output
``nrms(routed_moe - native) <= 1.5 * nrms(emul - native) + 2e-3`` where ``emul`` is
the FP64-accumulated emulation, and no row may exceed 4x the per-tensor
budget.

Native expert IDs and their top-k order are compared through the native raw
tuple: for every ``(t, k)`` the expert scale equals native's bit for bit and
the routed_moe FC2 row is the row of the same expert as native's row at
``expanded_idx_to_permuted_idx[t, k]`` (see ``routed_moe_fixtures.classify_vs_native``
for the exact / reorder / budget classes of the floating-point results).
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import routed_moe_fixtures as fx  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)

DEBUG_ROUTING = 1
DEBUG_QUANT = 2
DEBUG_FC1 = 4


@pytest.fixture(scope="module")
def layer():
    torch.manual_seed(0)
    return fx.build_native_layer(
        layer=int(os.environ.get("GLM_MOE_DSA_ROUTED_MOE_LAYER", "10")),
        tp_rank=int(os.environ.get("GLM_MOE_DSA_ROUTED_MOE_RANK", "0")),
    )


@pytest.fixture(scope="module")
def runner(layer):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe.routed_moe import (  # noqa: E501
        RoutedMoe,
    )

    return RoutedMoe.from_native(layer.module, layer.bias)


@pytest.fixture(scope="module")
def decoder(layer):
    return fx.ExpertDecoder(layer, layer.hidden_proxy(fx.prompt_token_ids()[:1])[0])


def _inputs(layer, rows: int, kind: str, seed: int = 0):
    ids = fx.prompt_token_ids()
    if kind == "proxy":
        start = (seed * 7) % (len(ids) - rows)
        x = layer.hidden_proxy(ids[start : start + rows])
        return x, layer.router_logits(x)
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(rows, fx.HIDDEN, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    if kind == "random":
        return x, layer.router_logits(x)
    if kind.startswith("distinct"):
        d = int(kind[len("distinct") :])
        sets = fx.expert_sets_with_distinct(rows, d, seed=seed)
        return x, fx.logits_for_expert_sets(sets, layer.bias, seed=seed)
    raise ValueError(kind)


def _dbg_buffer():
    return torch.zeros(2 * 1024 * 1024, dtype=torch.int32, device="cuda")


@pytest.mark.parametrize("rows", [1, 2, 6, 8])
@pytest.mark.parametrize("kind", ["proxy", "random"])
def test_routing_matches_native_bitwise(layer, runner, decoder, rows, kind):
    x, logits = _inputs(layer, rows, kind, seed=rows)
    dbg = _dbg_buffer()
    runner(x, logits, debug=DEBUG_ROUTING, debug_buffer=dbg)
    torch.cuda.synchronize()
    ids_ref, w_ref = fx.routing_reference(logits, layer.bias)
    ids = dbg[:64].view(8, 8)[:rows].cpu()
    w = dbg[64:128].view(torch.float32).view(8, 8)[:rows].cpu()
    assert torch.equal(ids, ids_ref.cpu()), (ids, ids_ref)
    assert torch.equal(w, w_ref.float().cpu())
    num_slots = int(dbg[256].item())
    expect = sorted(set(ids_ref.flatten().tolist()))
    assert num_slots == len(expect)
    assert dbg[128 : 128 + num_slots].cpu().tolist() == expect
    # native unfinalized expert weights, same top-k order
    outs = layer.native_forward(x, logits, do_finalize=False)
    assert torch.equal(outs[1].float().cpu(), w_ref.float().cpu())
    # native expert IDs and order, through the raw rows
    _assert_raw_matches_native(
        runner(x, logits, do_finalize=False), outs, lambda: decoder.compare(runner, logits)
    )
    # and decoded exactly through the probe fingerprints
    dec = decoder.compare(runner, logits)
    assert dec["ids_equal"] and dec["scale_exact"], dec
    assert torch.equal(dec["native_ids"].cpu().int(), ids_ref.cpu()), (dec["native_ids"], ids_ref)


def _assert_raw_matches_native(raw_routed_moe, raw_native, id_check=None):
    """Every (t, k): same expert scale bits and same expert as native."""
    rep = fx.raw_report(raw_routed_moe, raw_native)
    assert rep["scale_exact"], (raw_routed_moe[1], raw_native[1])
    if not rep["same_expert"]:
        assert id_check is not None, rep
        dec = id_check()
        assert dec["ids_equal"] and dec["scale_exact"] and dec["margin"] > 4, (rep, dec)
    return rep


@pytest.mark.parametrize("rows", [1, 6, 8])
def test_activation_quant_matches_native_bitwise(layer, runner, rows):
    x, logits = _inputs(layer, rows, "proxy", seed=3)
    dbg = _dbg_buffer()
    runner(x, logits, debug=DEBUG_QUANT, debug_buffer=dbg)
    torch.cuda.synchronize()
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe.kernel import (  # noqa: E501
        SFB_BYTES,
        SFB_CHUNK_STRIDE,
        SFB_STAGE_STRIDE,
    )

    raw = dbg.view(torch.uint8)[4096 : 4096 + 24 * 1024 + SFB_BYTES]
    sb = raw[: 24 * 1024].view(24, 1024)
    sfb_flat = raw[24 * 1024 :]
    xq, xsf = layer.module.quantize_input(x)
    xq = xq.view(torch.uint8).view(rows, fx.HIDDEN // 2)
    xsf = xsf.view(torch.uint8).view(rows, fx.HIDDEN // 16)
    kb = torch.arange(24, device="cuda")[:, None, None]
    n = torch.arange(rows, device="cuda")[None, :, None]
    b = torch.arange(128, device="cuda")[None, None, :]
    phys = n * 128 + (((b // 16) ^ n) * 16) + (b % 16)
    got = torch.gather(sb[:, None, :].expand(24, rows, 1024), 2, phys.expand(24, rows, 128))
    want = xq.view(rows, 24, 128).permute(1, 0, 2)
    assert torch.equal(got, want)
    f = torch.arange(16, device="cuda")[None, None, :]
    sf_phys = kb * SFB_STAGE_STRIDE + (f // 4) * SFB_CHUNK_STRIDE + n * 16 + f % 4
    got_sf = sfb_flat[sf_phys.expand(24, rows, 16)]
    want_sf = xsf.view(rows, 24, 16).permute(1, 0, 2)
    assert torch.equal(got_sf, want_sf)


@pytest.mark.parametrize("rows,kind", [(1, "proxy"), (6, "proxy"), (8, "distinct40")])
def test_fc1_accumulators_match_exact(layer, runner, rows, kind):
    x, logits = _inputs(layer, rows, kind, seed=5)
    dbg = _dbg_buffer()
    runner(x, logits, debug=DEBUG_ROUTING | DEBUG_FC1, debug_buffer=dbg)
    torch.cuda.synchronize()
    num_slots = int(dbg[256].item())
    slots = dbg[128 : 128 + num_slots].tolist()
    acc = dbg.view(torch.float32)[65536 : 65536 + num_slots * 4 * 128 * 8].view(
        num_slots, 4 * 128, 8
    )
    m = layer.module
    xq, xsf = m.quantize_input(x)
    xd = fx.unpack_fp4(xq.view(torch.uint8)) * fx.e4m3_bytes_to_float(
        xsf.view(torch.uint8).reshape(rows, fx.HIDDEN // 16)
    ).repeat_interleave(16, dim=1)
    worst = 0.0
    for s, e in enumerate(slots):
        w13, _ = fx.dequant_expert(m, e)
        ref = w13 @ xd.T  # [512, rows]
        got = acc[s, :, :rows].double()
        scale = ref.abs().max().item() + 1e-30
        worst = max(worst, ((got - ref).abs().max().item()) / scale)
    assert worst < 1e-5, worst


def _check_vs_native(layer, out, x, logits, label):
    native = layer.native_forward(x, logits)
    emul = fx.emulate_moe(layer, x, logits)
    e_native = fx.error_stats(out, native)
    e_emul = fx.error_stats(emul, native)
    e_self = fx.error_stats(out, emul)
    print(
        f"[routed_moe] {label}: routed_moe-vs-native {e_native} emul-vs-native {e_emul} routed_moe-vs-emul {e_self}"
    )
    budget = 1.5 * e_emul["nrms"] + 2e-3
    assert e_native["nrms"] <= budget, (e_native, e_emul)
    row_ref = native.float().pow(2).mean(-1).sqrt()
    row_err = (out.float() - native.float()).pow(2).mean(-1).sqrt()
    assert torch.all(row_err <= 4 * budget * row_ref + 1e-6), (row_err, row_ref)
    return e_native


@pytest.mark.parametrize("rows", [1, 2, 4, 6, 8])
@pytest.mark.parametrize("kind", ["proxy", "random", "distinct8"])
def test_finalized_matches_native(layer, runner, rows, kind):
    x, logits = _inputs(layer, rows, kind, seed=11 + rows)
    out = runner(x, logits)
    torch.cuda.synchronize()
    _check_vs_native(layer, out, x, logits, f"rows={rows} {kind}")


def test_raw_tuple_consumable_by_stock_finalizer(layer, runner):
    rows = 6
    x, logits = _inputs(layer, rows, "proxy", seed=2)
    fc2, scale, idx = runner(x, logits, do_finalize=False)
    out_final = runner(x, logits)
    torch.cuda.synchronize()
    assert fc2.shape == (rows * 8, fx.HIDDEN) and fc2.dtype == torch.bfloat16
    assert scale.shape == (rows, 8) and scale.dtype == torch.bfloat16
    assert idx.shape == (rows, 8) and idx.dtype == torch.int32
    _, w_ref = fx.routing_reference(logits, layer.bias)
    assert torch.equal(scale, w_ref)
    # stock finalize semantics: sum_k scale[t,k] * fc2[idx[t,k]] in FP32
    fin = torch.zeros(rows, fx.HIDDEN, device="cuda", dtype=torch.float32)
    for k in range(8):
        fin += scale[:, k : k + 1].float() * fc2[idx[:, k].long()].float()
    fin = fin.to(torch.bfloat16)
    _check_vs_native(layer, fin, x, logits, "raw tuple")
    e = fx.error_stats(fin, out_final)
    assert e["nrms"] < 2e-3, e


@pytest.mark.parametrize("offset", [1, 4, 7])
def test_misaligned_activation_is_copied_not_faulted(layer, runner, offset):
    """A contiguous activation view whose base is not 16-byte aligned (the bulk
    copy's requirement) gives the same raw and finalized outputs as the aligned
    tensor: the runner copies it instead of faulting or changing path."""
    rows = 6
    x, logits = _inputs(layer, rows, "proxy", seed=3)
    store = torch.empty(rows * fx.HIDDEN + offset, device="cuda", dtype=torch.bfloat16)
    x_mis = store[offset:].view(rows, fx.HIDDEN)
    x_mis.copy_(x)
    assert x_mis.is_contiguous() and x_mis.data_ptr() % 16 != 0
    if runner.schedule(rows, True) != "xc" or runner.schedule(rows, False) != "xc":
        pytest.skip("bit-exact repeat calls are a property of the expert-cluster schedule")
    # Outputs may live in the runner's workspace: copy each before the next call.
    ref = runner(x, logits).clone()
    got = runner(x_mis, logits).clone()
    raw_ref = [t.clone() for t in runner(x, logits, do_finalize=False)]
    raw_got = [t.clone() for t in runner(x_mis, logits, do_finalize=False)]
    torch.cuda.synchronize()
    assert torch.equal(got, ref)
    for a, b in zip(raw_got, raw_ref):
        assert torch.equal(a, b)


def test_cuda_graph_replay(layer, runner):
    rows = 6
    x0, l0 = _inputs(layer, rows, "proxy", seed=1)
    x_static = x0.clone()
    l_static = l0.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner(x_static, l_static)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out_static = runner(x_static, l_static)
    for it in range(12):
        kind = ["proxy", "random", "distinct8", "distinct48"][it % 4]
        x, logits = _inputs(layer, rows, kind, seed=100 + it)
        x_static.copy_(x)
        l_static.copy_(logits)
        graph.replay()
        torch.cuda.synchronize()
        eager = runner(x, logits)
        torch.cuda.synchronize()
        e = fx.error_stats(out_static, eager)
        assert e["nrms"] < 1e-6, (it, kind, e)
        if it < 4:
            _check_vs_native(layer, out_static.clone(), x, logits, f"graph replay {it} {kind}")


def _real_inputs(layer, rows: int, count: int, offset: int = 0):
    """``count`` windows of the captured hidden states of this layer and their
    router logits (GPU)."""
    win = [x for x, _ in fx.real_windows(fx.load_real_capture(layer.layer), rows)]
    assert win and win[0] is not None, f"no captured hidden states for layer {layer.layer}"
    step = max(1, len(win) // count)
    xs = [win[(offset + i * step) % len(win)].cuda().contiguous() for i in range(count)]
    return [(x, layer.router_logits(x)) for x in xs]


def _expert_ids(logits, bias):
    ids, _ = fx.routing_reference(logits, bias)
    return set(ids.flatten().tolist())


def _paddings(layer, x, logits, pads: int, other, seed: int):
    """Padded-row contents ``{name: (x_pad, logits_pad)}`` for ``pads`` rows."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rand = (torch.randn(pads, fx.HIDDEN, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    out = {
        "zeros": torch.zeros(pads, fx.HIDDEN, dtype=torch.bfloat16, device="cuda"),
        "scaled_copy": (x[:1].float() * 3.0).to(torch.bfloat16).repeat(pads, 1),
        "real": other[:pads].contiguous(),
        "random": rand,
    }
    out = {k: (v, layer.router_logits(v)) for k, v in out.items()}
    # padded rows whose top-8 sets avoid every live expert: the call's
    # distinct-expert count grows by 8 per padded row (two rounds when it
    # exceeds the cluster count)
    used = _expert_ids(logits, layer.bias)
    free = [
        e
        for e in torch.randperm(
            fx.NUM_EXPERTS, generator=torch.Generator().manual_seed(seed)
        ).tolist()
        if e not in used
    ]
    sets = [free[8 * i : 8 * i + 8] for i in range(pads)]
    out["new_experts"] = (rand, fx.logits_for_expert_sets(sets, layer.bias, seed=seed))
    return out


@pytest.mark.parametrize("live", [1, 2, 4, 6])
def test_padded_rows_do_not_change_live_rows(layer, runner, live):
    """Padding invariance of the default (expert-cluster) schedule, bit for bit.

    Real hidden-state windows of ``live`` rows are padded to 8 rows with
    zeros, a scaled copy of a live row, rows of another real window, random
    rows, and rows routed to experts no live row uses (more distinct experts,
    up to a two-round call). The live rows' raw FC2 rows, expert scales and
    finalized output must equal the unpadded call's bit for bit.
    """
    if runner.schedule(8, True) != "xc" or runner.schedule(8, False) != "xc":
        pytest.skip("padding invariance is a property of the expert-cluster schedule")
    pads = 8 - live
    clusters = runner._xc_ctas(8, True) // 4
    others = _real_inputs(layer, 8, 3, offset=5)
    seen = {}
    for i, (x, logits) in enumerate(_real_inputs(layer, live, 3, offset=live)):
        fin = runner(x, logits).clone()
        raw = [t.clone() for t in runner(x, logits, do_finalize=False)]
        torch.cuda.synchronize()
        for name, (xp, lp) in _paddings(layer, x, logits, pads, others[i][0], 70 + i).items():
            xa = torch.cat([x, xp]).contiguous()
            la = torch.cat([logits, lp.float()]).contiguous()
            fin_p = runner(xa, la)
            raw_p = runner(xa, la, do_finalize=False)
            torch.cuda.synchronize()
            d = len(_expert_ids(la, layer.bias))
            seen.setdefault(name, []).append(d)
            tag = f"live={live} window {i} padding {name} D={d}"
            assert torch.equal(fin_p[:live].view(torch.int16), fin.view(torch.int16)), tag
            rows_p = raw_p[0][: live * 8].view(torch.int16)
            assert torch.equal(rows_p, raw[0].view(torch.int16)), tag
            assert torch.equal(raw_p[1][:live], raw[1]), tag
            assert torch.isfinite(fin_p).all(), tag
    two_round = [d for ds in seen.values() for d in ds if d > clusters]
    print(
        f"[routed_moe] padding invariance live={live}: bit-exact for paddings {seen} "
        f"(distinct experts per call; {len(two_round)} two-round calls, {clusters} clusters)"
    )
    assert two_round, "no padded call reached the two-round schedule"


REORDER_REL = 2.0**-20


def test_fp32_accumulator_output_reorder_class(layer, runner):
    """routed_moe_into_hop "fp32" output: the kernel adds the eight weighted rows of a token
    with atomic FP32 adds. Every run, with or without padded rows, lies within
    the FP32 reorder bound of the exact sum of the same products (the raw
    rows' ``scale * row``, each exact in FP32); the run-to-run spread is
    counted, not assumed."""
    rows = 6
    if runner.schedule(8, False, True) != "xc":
        pytest.skip("FP32 accumulator output needs the expert-cluster schedule")
    others = _real_inputs(layer, 8, 3, offset=9)
    distinct_runs = []
    for i, (x, logits) in enumerate(_real_inputs(layer, rows, 3, offset=2)):
        fc2, scale, idx = runner(x, logits, do_finalize=False)
        terms = scale.double()[:, :, None] * fc2[idx.long()].double()
        exact = terms.sum(1)
        bound = REORDER_REL * terms.abs().sum(1)
        runs = []
        for _ in range(8):
            acc = torch.zeros(rows, fx.HIDDEN, dtype=torch.float32, device="cuda")
            runner(x, logits, red_out=acc)
            runs.append(acc)
        xp = torch.cat([x, others[i][0][:2]]).contiguous()
        lp = torch.cat([logits, layer.router_logits(others[i][0][:2]).float()]).contiguous()
        acc8 = torch.zeros(8, fx.HIDDEN, dtype=torch.float32, device="cuda")
        runner(xp, lp, red_out=acc8)
        torch.cuda.synchronize()
        for j, acc in enumerate(runs + [acc8[:rows]]):
            assert ((acc.double() - exact).abs() <= bound).all(), f"window {i} run {j}"
        distinct_runs.append(len({r.cpu().numpy().tobytes() for r in runs}))
    print(
        f"[routed_moe] fp32 accumulator: within the FP32 reorder bound in every run; distinct "
        f"results over 8 runs per window {distinct_runs}"
    )


def test_workspace_created_once_in_device_memory(layer, runner):
    """The per-device workspace is device memory even when first requested
    inside the served loader's ``MetaInitMode``, is never replaced, and a
    request for more CTA slots than it holds raises."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe import (  # noqa: E501
        routed_moe as rm,
    )
    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode

    dev = runner.device
    key = dev.index if dev.index is not None else torch.cuda.current_device()
    saved = rm._WORKSPACES.pop(key, None)
    try:
        with MetaInitMode():
            assert torch.empty(1, device="cuda").is_meta, "MetaInitMode is not active"
            fresh = rm.RoutedMoe.from_native(layer.module, layer.bias)
            ws = rm._workspace(fresh.device, fresh.num_ctas)
        for name, t in ws.tensors().items():
            assert not t.is_meta and t.is_cuda and t.device.index == key, name
            assert t.data_ptr() != 0, name
        assert ws.num_ctas >= rm._sm_count(dev)
        assert rm._workspace(dev, fresh.num_ctas) is ws
        with pytest.raises(RuntimeError):
            rm._workspace(dev, ws.num_ctas + 4)
        assert rm._WORKSPACES[key] is ws
        print(f"[routed_moe] workspace cuda:{key}: {ws.num_ctas} CTA slots, {ws.nbytes()} bytes")
    finally:
        if saved is not None:
            rm._WORKSPACES[key] = saved


def test_eager_calls_from_two_streams_are_ordered(layer, runner):
    """A call issued on another stream than the previous call waits for the
    previous stream's submitted work before it uses the shared workspace."""
    rows = 6
    x, logits = _inputs(layer, rows, "proxy", seed=4)
    x2, l2 = _inputs(layer, rows, "distinct40", seed=6)
    ref = runner(x, logits).clone()
    ref2 = runner(x2, l2).clone()
    torch.cuda.synchronize()
    a, b = torch.cuda.Stream(), torch.cuda.Stream()
    a.wait_stream(torch.cuda.current_stream())
    b.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(a):
        torch.cuda._sleep(100_000_000)  # tens of milliseconds of device time
        out_a = runner(x, logits)
    with torch.cuda.stream(b):
        out_b = runner(x2, l2)
        done_b = torch.cuda.Event()
        done_b.record(b)
    done_b.synchronize()
    assert a.query(), "the call on the second stream did not wait for the first stream"
    torch.cuda.synchronize()
    assert torch.equal(out_a.view(torch.int16), ref.view(torch.int16))
    assert torch.equal(out_b.view(torch.int16), ref2.view(torch.int16))


def test_fallback_is_decided_before_device_work(layer):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe.routed_moe import (  # noqa: E501
        _WORKSPACES,
        routed_output,
    )

    x9 = torch.randn(9, fx.HIDDEN, device="cuda", dtype=torch.bfloat16)
    l9 = torch.randn(9, 256, device="cuda")
    snapshot = {k: v.ws.clone() for k, v in _WORKSPACES.items()}
    assert routed_output(layer.module, layer.bias, x9, l9, True) is None
    x1 = torch.randn(1, fx.HIDDEN, device="cuda", dtype=torch.bfloat16)
    assert (
        routed_output(
            layer.module,
            layer.bias,
            x1,
            torch.randn(1, 256, device="cuda").to(torch.bfloat16),
            True,
        )
        is None
    )
    assert (
        routed_output(
            layer.module, layer.bias, x1.float(), torch.randn(1, 256, device="cuda"), True
        )
        is None
    )
    # FP32 router bias (non-TRTLLM gate), a bias tensor that is not the routing
    # method's, and modules without registered expert parameters stay native
    l1 = torch.randn(1, 256, device="cuda")
    assert routed_output(layer.module, layer.bias.float(), x1, l1, True) is None
    assert routed_output(layer.module, layer.bias.clone(), x1, l1, True) is None
    assert routed_output(torch.nn.Linear(4, 4), layer.bias, x1, l1, True) is None

    class _DelegatingOnly(torch.nn.Module):
        """Exposes the expert tensors as properties but owns none of them."""

        w3_w1_weight = property(lambda self: layer.module.w3_w1_weight)
        w2_weight = property(lambda self: layer.module.w2_weight)

    assert routed_output(_DelegatingOnly(), layer.bias, x1, l1, True) is None
    torch.cuda.synchronize()
    for k, v in _WORKSPACES.items():
        assert torch.equal(v.ws, snapshot[k])


def test_dispatch_registration():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    fn = dispatch.get_impl("routed_moe", "routed_output")
    assert callable(fn)
    assert callable(dispatch.get_impl("routed_moe", "eligibility"))


# ---------------------------------------------------------------------------
# Near-tie routing: FP32 logits built so that the 8th/9th selection keys tie
# exactly, or differ by one key ulp, including experts with equal BF16 bias.
# ---------------------------------------------------------------------------
TIE_PATTERNS = (
    "same_logit",  # equal bias, identical logits: exact key tie
    "same_key",  # equal bias, different logits, exact key tie
    "plus_ulp",  # equal bias, key(b) = key(a) + 1 ulp
    "minus_ulp",  # equal bias, key(b) = key(a) - 1 ulp
    "cross_bias_lo",  # bias(a) < bias(b), exact key tie
    "cross_bias_hi",  # bias(a) > bias(b), exact key tie
    "three_way",  # three equal keys for the last two slots
    "inner_tie",  # two winners with equal keys (top-k order only)
)
_WIN_LOGIT = 6.0
_CONT_LOGIT = 0.3
_BOOST = 0.02


def _tie_row(bias, pattern: str, gen: torch.Generator):
    """One row of logits for ``pattern`` plus its resolution variants.

    Returns ``(row, variants)``: ``variants`` are logit rows in which each
    possible resolution of the tie wins clearly (by ``_BOOST`` in logit); the
    native outcome of ``row`` must equal exactly one of them.
    """
    b = bias.float().cpu()
    vals = sorted(set(b.tolist()))
    groups = [[e for e in range(fx.NUM_EXPERTS) if b[e].item() == v] for v in vals]
    big = [g for g in groups if len(g) >= 3]
    perm = lambda n: torch.randperm(n, generator=gen).tolist()  # noqa: E731
    row = -12.0 + 0.5 * torch.rand(fx.NUM_EXPERTS, generator=gen)
    if pattern in ("cross_bias_lo", "cross_bias_hi"):
        # cont[0] is the lower expert index; its bias is the lower one for
        # "lo" and the higher one for "hi"
        lo_g = groups[0] + (groups[1] if len(groups) > 1 else [])
        hi_g = groups[-1] + (groups[-2] if len(groups) > 2 else [])
        want_lo_first = pattern == "cross_bias_lo"
        cands = [(x, y) for x in lo_g for y in hi_g if (x < y) == want_lo_first]
        assert cands, pattern
        x, y = cands[perm(len(cands))[0]]
        cont = [x, y] if want_lo_first else [y, x]
    else:
        g = big[perm(len(big))[0]]
        pick = [g[i] for i in perm(len(g))[:3]]
        cont = sorted(pick[:3] if pattern == "three_way" else pick[:2])
    n_win = {"three_way": 6, "inner_tie": 6}.get(pattern, 7)
    others = [e for e in perm(fx.NUM_EXPERTS) if e not in cont]
    winners = others[:n_win]
    for i, e in enumerate(winners):
        row[e] = _WIN_LOGIT + 0.01 * i
    bdev = bias.float().cuda()
    a = cont[0]
    if pattern == "inner_tie":
        # two contenders become winners with identical keys; no boundary tie
        for e in cont[:2]:
            row[e] = _WIN_LOGIT + 0.5
    else:
        for e in cont:
            row[e] = _CONT_LOGIT
    key_a = fx._key_of(torch.tensor(row[a].item(), device="cuda"), bdev[a]).item()
    c = cont[1]
    if pattern == "same_key" or pattern.startswith("cross_bias"):
        hit = fx.logit_for_key(key_a, bdev[c], 0, avoid=row[a].item())
        assert hit is not None, pattern
        row[c] = hit
    elif pattern in ("plus_ulp", "minus_ulp"):
        hit = fx.logit_for_key(key_a, bdev[c], 1 if pattern == "plus_ulp" else -1)
        assert hit is not None, pattern
        row[c] = hit
    variants = []
    if pattern == "three_way":
        for pair in ((0, 1), (0, 2), (1, 2)):
            v = row.clone()
            for i in pair:
                v[cont[i]] += _BOOST
            variants.append(v)
    else:
        for e in cont[:2]:
            v = row.clone()
            v[e] += _BOOST
            variants.append(v)
    return row, variants


def _tie_batch(bias, patterns, seed):
    gen = torch.Generator().manual_seed(seed)
    rows, variants = [], []
    for p in patterns:
        r, v = _tie_row(bias, p, gen)
        rows.append(r)
        variants.append(v)
    logits = torch.stack(rows).cuda().contiguous()
    nvar = max(len(v) for v in variants)
    var_batches = [
        torch.stack([v[min(i, len(v) - 1)] for v in variants]).cuda().contiguous()
        for i in range(nvar)
    ]
    return logits, var_batches, [len(v) for v in variants]


def _check_ties(layer, runner, decoder, x, logits, var_batches, nvars, label):
    native_raw = layer.native_forward(x, logits, do_finalize=False)
    native_fin = layer.native_forward(x, logits)
    routed_moe_raw = runner(x, logits, do_finalize=False)
    routed_moe_fin = runner(x, logits)
    torch.cuda.synchronize()
    tie_rows = fx.raw_rows_by_topk(native_raw)
    var_rows = [
        fx.raw_rows_by_topk(layer.native_forward(x, v, do_finalize=False)) for v in var_batches
    ]
    resolution = []
    for t in range(x.shape[0]):
        match = [i for i in range(nvars[t]) if torch.equal(tie_rows[t], var_rows[i][t])]
        distinct = len({tuple(var_rows[i][t].flatten()[::97].tolist()) for i in range(nvars[t])})
        assert distinct == nvars[t], f"{label} row {t}: variants not distinguishable"
        assert len(match) == 1, f"{label} row {t}: native outcome matches variants {match}"
        resolution.append(match[0])
    cls, rep = fx.classify_vs_native(
        routed_moe_fin,
        routed_moe_raw,
        native_fin,
        native_raw,
        lambda: fx.emulate_moe(layer, x, logits),
        label,
        lambda: decoder.compare(runner, logits),
    )
    # the decoded native experts of every tie row, independent of the rows
    dec = decoder.compare(runner, logits)
    assert dec["ids_equal"] and dec["scale_exact"], (label, dec)
    gaps = fx.boundary_gap(logits, layer.bias)
    print(
        f"[routed_moe] ties {label}: resolution={resolution} class={cls} "
        f"rows_exact={rep['rows_exact']}/{rep['rows']} decode_margin={dec['margin']:.1f} "
        f"boundary_gap={gaps.tolist()}"
    )
    return resolution


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_boundary_ties_match_native(layer, runner, decoder, seed):
    """All eight tie patterns in one R=8 call; routed_moe equals native bit for bit
    and the native outcome is identified as one clear resolution.
    """
    logits, var_batches, nvars = _tie_batch(layer.bias, TIE_PATTERNS, seed)
    x, _ = _inputs(layer, len(TIE_PATTERNS), "proxy", seed=20 + seed)
    res = _check_ties(layer, runner, decoder, x, logits, var_batches, nvars, f"R=8 seed={seed}")
    # identical logits and bias give identical keys in any arithmetic; they
    # resolve to the lower expert index (variant 0), as the native 64-bit key
    # (score bits, 65535 - index) orders them
    for p, r in zip(TIE_PATTERNS, res):
        if p in ("same_logit", "three_way", "inner_tie"):
            assert r == 0, (p, r)


@pytest.mark.parametrize("pattern", TIE_PATTERNS)
def test_boundary_tie_single_row(layer, runner, decoder, pattern):
    logits, var_batches, nvars = _tie_batch(layer.bias, [pattern], 7)
    x, _ = _inputs(layer, 1, "proxy", seed=31)
    _check_ties(layer, runner, decoder, x, logits, var_batches, nvars, f"R=1 {pattern}")


def test_near_tie_sweep_matches_native(layer, runner, decoder):
    """Randomized near-boundary rows: the 8th/9th keys at -2..+2 key ulps
    (random contender biases and logits); routed_moe equals native bit for bit.
    """
    gen = torch.Generator().manual_seed(77)
    bdev = layer.bias.float().cuda()
    batches = 48
    exact_ties = 0
    classes = {}
    for i in range(batches):
        rows = []
        for t in range(8):
            row = -12.0 + 0.5 * torch.rand(fx.NUM_EXPERTS, generator=gen)
            order = torch.randperm(fx.NUM_EXPERTS, generator=gen).tolist()
            for k, e in enumerate(order[:7]):
                row[e] = _WIN_LOGIT + 0.01 * k
            a, c = order[7], order[8]
            row[a] = 0.1 + 0.6 * torch.rand(1, generator=gen).item()
            key_a = fx._key_of(torch.tensor(row[a].item(), device="cuda"), bdev[a]).item()
            off = int(torch.randint(-2, 3, (1,), generator=gen).item())
            hit = fx.logit_for_key(key_a, bdev[c], off)
            row[c] = hit if hit is not None else row[a]
            rows.append(row)
        logits = torch.stack(rows).cuda().contiguous()
        exact_ties += int((fx.boundary_gap(logits, layer.bias) == 0).sum().item())
        x, _ = _inputs(layer, 8, "random", seed=200 + i)
        cls, _ = fx.check_call(
            layer, runner, x, logits, label=f"near-tie batch {i}", decoder=decoder
        )
        dec = decoder.compare(runner, logits)
        assert dec["ids_equal"] and dec["scale_exact"], (i, dec)
        classes[cls] = classes.get(cls, 0) + 1
    print(
        f"[routed_moe] near-tie sweep: {batches * 8} rows, {exact_ties} exact boundary key ties, "
        f"batch classes {classes}"
    )
