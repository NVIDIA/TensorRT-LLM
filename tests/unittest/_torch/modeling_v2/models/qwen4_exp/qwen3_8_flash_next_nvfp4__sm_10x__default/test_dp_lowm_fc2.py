# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep8's in-house FC2 at the lean gathered row counts (`DP_MOE_LOWM_FC2_ENV`, default off).

* The switch: off unset or at 0, `_DP_MOE_LOWM_FC2_ROWS` at 1, an integer sets the upper bound
  (within the lean rows); ignored with a warning away from a dep8 rank's window.
* The pin: `_DP_MOE_LOWM_FC2_TACTIC` where `moe_lowm_fc2` reproduces its FC2 cubin, with the
  listed W64 tactic as the build check's reference where the runner does not list it; none where
  the FC2 cubin is not one `moe_lowm_fc2` reproduces.
* The run: an un-finalized call at a lever row count runs the runner's FC1-only entry and
  `moe_lowm_fc2` into its FC2 rows; a finalized call, another row count or `lowm=False` runs the
  runner's own pair.
* On SM103 (GPU): over 48 layers of random NVFP4 window stacks at dep8 caps 2 / 3 / 4 / 6 / 8
  (64..256 gathered rows), the route front (lean form, PDL, release after its wait) followed by
  the lever's FC1 + `moe_lowm_fc2` writes the same FC2 row for every (row, pick) as today's pin on
  the cubins, bit for bit, eagerly and replayed in one CUDA graph.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

W64 = SimpleNamespace(topk=10, hidden=2560, expert_inter=640, local_experts=64, dp_size=8)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "value, expected",
    [
        ("", None),
        ("0", None),
        ("off", None),
        ("1", (64, 256)),
        ("128", (64, 128)),
        ("256", (64, 256)),
    ],
)
def test_rows(monkeypatch, value, expected):
    monkeypatch.setenv(M.DP_MOE_LOWM_FC2_ENV, value)
    assert M._dp_lowm_fc2_rows(W64) == expected


@pytest.mark.cpu_only
@pytest.mark.parametrize("value", ["2", "63", "257", "x"])
def test_rows_rejects_another_value(monkeypatch, value):
    monkeypatch.setenv(M.DP_MOE_LOWM_FC2_ENV, value)
    with pytest.raises(ValueError):
        M._dp_lowm_fc2_rows(W64)


@pytest.mark.cpu_only
@pytest.mark.parametrize("dp_size, local", [(4, 128), (1, 128), (8, 128)])
def test_rows_ignored_away_from_dep8(monkeypatch, dp_size, local):
    monkeypatch.setenv(M.DP_MOE_LOWM_FC2_ENV, "1")
    warned = []
    monkeypatch.setattr(M.logger, "warning", lambda msg: warned.append(msg))
    core = SimpleNamespace(dp_size=dp_size, local_experts=local)
    assert M._dp_lowm_fc2_rows(core) is None
    assert len(warned) == 1 and M.DP_MOE_LOWM_FC2_ENV in warned[0]


class _Runner:
    def __init__(self, valid, tile_n="8", tile_k="512"):
        self.valid, self.tile_n, self.tile_k = valid, tile_n, tile_k

    def get_valid_configs(self, top_k, hidden, inter, num_experts, rows):
        assert (top_k, hidden, inter, num_experts) == (10, 2560, 640, 64)
        return [list(t) for t in self.valid]

    def get_gemm2_kernel_options(self, tile, config):
        from tensorrt_llm._torch.moe.fused_moe.trtllm_gen import decode_moe

        options = dict(decode_moe._CUTE_FC2_OPTIONS)
        options.update(tileN=self.tile_n, mmaN=self.tile_n, tileK=self.tile_k)
        return options


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "rows, valid, expected",
    [
        (64, [(8, 53), (16, 32)], ((8, 53), None, 512)),  # listed: no reference
        (96, [(8, 53), (16, 32)], ((8, 53), None, 512)),
        (128, [(16, 32), (32, 32)], ((8, 53), (16, 32), 512)),  # W64's pin, listed
        (192, [(32, 32), (64, 5)], ((8, 53), (32, 32), 512)),
        (224, [(32, 20)], ((8, 53), (32, 20), 512)),  # W64's (32, 32) not listed: its reference
        (256, [], None),  # nothing listed to hold the pin to
    ],
)
def test_pin(monkeypatch, rows, valid, expected):
    pytest.importorskip("cutlass")
    monkeypatch.setattr(M, "_dp_hicap", lambda core=None: frozenset())
    assert M._dp_lowm_fc2_pin(_Runner(valid), W64, rows) == expected


@pytest.mark.cpu_only
def test_pin_declines_a_cubin_it_does_not_reproduce(monkeypatch):
    pytest.importorskip("cutlass")
    monkeypatch.setattr(M, "_dp_hicap", lambda core=None: frozenset())
    assert M._dp_lowm_fc2_pin(_Runner([(8, 53)], tile_n="16"), W64, 64) is None
    assert M._dp_lowm_fc2_pin(_Runner([(8, 53)], tile_k="128"), W64, 64) is None


class _RecRunner:
    def __init__(self, rows, hidden):
        self.calls, self.rows, self.hidden = [], rows, hidden

    def run_moe_prerouted(self, *args):
        self.calls.append(("pair", len(args), args[25]))  # do_finalize
        return [torch.full((self.rows, self.hidden), 1.0)]

    def run_moe_prerouted_fc1(self, *args):
        self.calls.append(("fc1", len(args), None))
        z = torch.zeros(1)
        return [z, z, torch.zeros(self.rows, self.hidden), z, z, z, z, torch.zeros(1, 1)]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "finalize, lowm, rows, expected",
    [
        (False, True, 64, "fc1"),
        (True, True, 64, "pair"),
        (False, False, 64, "pair"),
        (False, True, 512, "pair"),  # not a lever row count
    ],
)
def test_run_dispatch(monkeypatch, finalize, lowm, rows, expected):
    pytest.importorskip("cutlass")
    import tensorrt_llm._torch.kernels.blackwell.moe_lowm_fc2 as lowm_mod

    fc2 = []
    monkeypatch.setattr(
        lowm_mod, "moe_lowm_fc2", lambda *args, cubin_k_tile=None: fc2.append(cubin_k_tile)
    )
    runner = _RecRunner(rows, 8)
    st = M._RouteFront(runner)
    st.tactics = {rows: (8, 53)}
    st.views = {rows: [[torch.zeros(1, dtype=torch.int32)] * 6]}
    st.lowm = {64: 512}
    L = SimpleNamespace()
    for name in ("fc1_w", "fc1_s", "fc2_w", "fc2_s", "out1_scale", "out1_gate_scale", "out2_scale"):
        setattr(L, name, torch.zeros(1))
    core = SimpleNamespace(expert_offset=0, local_experts=64, topk=10, expert_inter=640)
    routed = (
        torch.zeros(rows, 4),
        torch.zeros(rows, 1),
        torch.zeros(rows, 10),
        torch.zeros(rows, 10),
    )
    out = M._route_front_run(core, st, None, L, 0, None, None, finalize, routed, lowm=lowm)[0]
    kind, nargs, fin = runner.calls[0]
    assert kind == expected and len(runner.calls) == 1
    if kind == "fc1":
        # the pair's arguments without FC2 bias, finalize and output
        assert nargs == 37 - 3 and fc2 == [512] and float(out.sum()) == 0.0
    else:
        assert nargs == 37 and fin is finalize and fc2 == []


# ---------------------------------------------------------------------------------------------
# GPU: bitwise over 48 layers at dep8 caps 2 / 3 / 4 / 6 / 8.

_SM103 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3)
H, INTER, E, K, LOCAL, NL = 2560, 640, 512, 10, 64, 48


def _stacks(gen, dev):
    def sf(*shape):
        return torch.randint(0x08, 0x70, shape, dtype=torch.uint8, device=dev, generator=gen).view(
            torch.float8_e4m3fn
        )

    x_scale, h_scale = 448.0 * 6.0 / 8.0, 448.0 * 6.0 / 24.0
    layers = []
    for _ in range(NL):
        a1 = (
            1.5
            / (x_scale * 2.93 * 1.3 * H**0.5)
            * (0.8 + 0.4 * torch.rand(LOCAL, device=dev, generator=gen))
        )
        a2 = (0.8 + 0.4 * torch.rand(LOCAL, device=dev, generator=gen)) / (
            h_scale * 2.93 * 1.3 * INTER**0.5
        )
        layers.append(
            SimpleNamespace(
                moe_gscale=torch.tensor([x_scale], device=dev),
                fc1_w=torch.randint(
                    0, 256, (LOCAL, 2 * INTER, H // 2), dtype=torch.uint8, device=dev, generator=gen
                ),
                fc1_s=sf(LOCAL, 2 * INTER, H // 16),
                fc2_w=torch.randint(
                    0, 256, (LOCAL, H, INTER // 2), dtype=torch.uint8, device=dev, generator=gen
                ),
                fc2_s=sf(LOCAL, H, INTER // 16),
                out1_scale=(h_scale * a1).float().contiguous(),
                out1_gate_scale=a1.float().contiguous(),
                out2_scale=a2.float().contiguous(),
                # a skewed router: a few hot experts a window, as under real routing
                bias=2.0 * torch.randn(E, device=dev, generator=gen),
            )
        )
    return layers


def _front(rows, tactic, lowm_k, dev):
    st = M._RouteFront(M._moe_runner())
    tile = int(tactic[0])
    assert M._route_front_compile(rows, H, H, LOCAL, tile, 0)
    n = sum(M.table_sizes(rows, K, LOCAL, tile))
    st.tactics[rows] = tactic
    st.flats[rows] = [(torch.arange(n, device=dev) % rows).to(torch.int32) for _ in range(NL)]
    st.views[rows] = [M.table_views(f, rows, K, LOCAL, tile) for f in st.flats[rows]]
    if lowm_k is not None:
        st.lowm[rows] = lowm_k
    return st


@pytest.mark.skipif(not _SM103, reason="the route front and moe_lowm_fc2 are certified on SM103")
@pytest.mark.parametrize("cap", [2, 3, 4, 6, 8])
def test_bitwise_every_layer(monkeypatch, cap):
    pytest.importorskip("cutlass")
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "1")
    monkeypatch.setenv("TRTLLM_QWEN4_EXP_DP_ROUTE_FRONT_LEAN_WAIT", "1")
    monkeypatch.setattr(M, "_dp_hicap", lambda core=None: frozenset())
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(20261128 + cap)
    rows = 8 * cap * 4
    layers = _stacks(gen, dev)
    runner = M._moe_runner()
    stock_tactic, _ = M._route_front_w64_tactic(runner, W64, rows)
    pin = M._dp_lowm_fc2_pin(runner, W64, rows)
    assert pin is not None, f"no lever pin at {rows} rows"
    lever_tactic, _, tile_k = pin
    stock = _front(rows, stock_tactic, None, dev)
    lever = _front(rows, lever_tactic, tile_k, dev)
    cores = [
        SimpleNamespace(
            expert_offset=LOCAL * (i % (E // LOCAL)),
            local_experts=LOCAL,
            topk=K,
            expert_inter=INTER,
        )
        for i in range(NL)
    ]
    xs = [torch.randn(rows, H, device=dev, generator=gen).to(torch.bfloat16) for _ in range(NL)]
    logits = [
        (torch.randn(rows, E, device=dev, generator=gen) + L.bias).to(torch.bfloat16).contiguous()
        for L in layers
    ]

    def step(st):
        outs = []
        for i, L in enumerate(layers):
            out, _, _, ids, wts = M._route_front_run(
                cores[i], st, xs[i], L, i, logits[i], finalize=False
            )
            outs.append((out, ids, wts))
        return outs

    def rows_of(st, outs):
        """Per layer: each (row, pick)'s FC2 row (-1 picks zeroed), the experts and weights."""
        got = []
        for i, (out, ids, wts) in enumerate(outs):
            e2p = st.views[rows][i][0].long()
            live = e2p >= 0
            fc2 = torch.zeros(rows * K, H, dtype=torch.bfloat16, device=dev)
            fc2[live] = out[e2p[live]]
            got.append((fc2.view(torch.int16), live, ids.clone(), wts.view(torch.int16).clone()))
        return got

    def same(a, b, what):
        for i, (x, y) in enumerate(zip(a, b)):
            for j, name in enumerate(("FC2 rows", "picks in the window", "experts", "weights")):
                assert torch.equal(x[j], y[j]), f"{what}: layer {i} {name} differ at {rows} rows"

    ref = rows_of(stock, step(stock))
    eager = rows_of(lever, step(lever))
    torch.cuda.synchronize()
    same(eager, ref, "eager lever vs stock")
    hits = sum(int(r[1].sum()) for r in ref)
    assert hits > 0
    assert lever.lowm == {rows: tile_k}

    # The same 48 layers replayed in one graph (route front -> FC1 -> moe_lowm_fc2 chained by PDL).
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step(lever)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step(lever)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        same(rows_of(lever, captured), ref, "graph lever vs stock")
