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
"""Launch configuration of the cluster routing tail by token count (``route_tail_bands``:
``"helix8"`` on K3.1's helix8 generation server, ``""`` elsewhere).

* no bands (``""``, every other server): :func:`tail_config` is :func:`default_config` at every
  token count, so :func:`run_k3_route_tail` launches with the default configuration;
* bands: the first band with ``m <= max_m`` gives ``(num_ctas, num_warps, warps_per_token)``,
  token counts above the last band keep :func:`default_config`; invalid bands raise;
* every configuration routes bit-identically (top-k ids, weights and the ``moe_sort`` tables, also
  under ``route_approx`` and in a replayed CUDA graph).
"""

import pytest
import torch
from _rubin_route_tail_inputs import router_logits

TOP_K = 16
TILE = 128
SENTINEL = -7

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="needs an SM107 GPU",
)


def _module():
    import tensorrt_llm  # noqa: F401  (registers the reference ops)
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (
        rubin_route_tail,
    )

    return rubin_route_tail


def test_parse_valid():
    rt = _module()
    assert rt._parse_route_tail_cfg("") == ()
    assert rt._parse_route_tail_cfg(" , ") == ()
    assert rt._parse_route_tail_cfg("64:8x16x2") == ((64, 8, 16, 2),)
    assert rt._parse_route_tail_cfg("32:1x8x2, 64:2x16x1,128:4X16X4,256:8x32x1") == (
        (32, 1, 8, 2),
        (64, 2, 16, 1),
        (128, 4, 16, 4),
        (256, 8, 32, 1),
    )
    assert rt._parse_route_tail_cfg("8:1x7x7") == ((8, 1, 7, 7),)


@pytest.mark.parametrize(
    "spec",
    [
        "64",  # no configuration
        "64:8x16",  # two fields
        "a:8x16x2",  # not an int
        "64:8x16x2,32:8x8x2",  # not ascending
        "64:8x16x2,64:8x8x2",  # repeated max_m
        "0:8x8x1",  # max_m < 1
        "257:8x8x1",  # above the cluster tail
        "64:1x8x1",  # 64 tokens on one CTA (> 32 per CTA)
        "256:4x16x1",  # 64 per CTA
        "64:9x8x1",  # more than 8 CTAs
        "64:8x8x3",  # team of 3 warps
        "64:8x6x4",  # team does not divide the warps
        "64:8x33x1",  # more than 32 warps
        "64:8x32x2",  # 16 multi-warp teams
    ],
)
def test_parse_invalid(spec):
    rt = _module()
    with pytest.raises(ValueError, match="routing tail bands"):
        rt._parse_route_tail_cfg(spec)


def test_no_bands_is_default_config():
    """No bands: tail_config == default_config at every token count."""
    rt = _module()
    for m in range(1, rt.MAX_M + 1):
        assert rt.tail_config(m) == rt.tail_config(m, "") == rt.default_config(m), m


def test_bands_follow_the_parameter():
    """No bands unless ``route_tail_bands`` selects the helix8 ones (``HELIX8_ROUTE_TAIL_CFG``,
    K3.1's helix8 generation server); any other value raises."""
    rt = _module()
    assert rt._route_tail_bands() == rt._route_tail_bands("") == ()
    assert rt._route_tail_bands("helix8") == rt._parse_route_tail_cfg(rt.HELIX8_ROUTE_TAIL_CFG)
    with pytest.raises(ValueError, match="route_tail_bands"):
        rt._route_tail_bands("dep16")
    with pytest.raises(ValueError, match="route_tail_bands"):
        rt.tail_config(8, "dep16")


def test_the_helix8_bands():
    """The helix8 bands: the first band with ``m <= max_m``, above the last one
    :func:`default_config`."""
    rt = _module()
    bands = rt._route_tail_bands("helix8")
    for m in range(1, rt.MAX_M + 2):
        band = next(((c, t, w) for max_m, c, w, t in bands if m <= max_m), None)
        expect = rt.default_config(m) if band is None else band
        assert rt.tail_config(m, "helix8") == expect, m


# Bit-identical routing for every configuration.


def _outputs(rt, m, num_local):
    """Sentinel-filled outputs (``moe_sort`` writes only valid entries: compare whole buffers)."""
    max_tiles, max_permuted = rt.sort_buffer_sizes(m, num_local, TILE)
    i32 = dict(dtype=torch.int32, device="cuda")
    return (
        torch.full((m, TOP_K), SENTINEL, **i32),
        torch.full((m, TOP_K), float(SENTINEL), dtype=torch.float32, device="cuda"),
        torch.full((max_tiles,), SENTINEL, **i32),
        torch.full((max_tiles,), SENTINEL, **i32),
        torch.full((m, TOP_K), SENTINEL, **i32),
        torch.full((max_permuted,), SENTINEL, **i32),
        torch.full((1,), SENTINEL, **i32),
        torch.full((1,), SENTINEL, **i32),
    )


def _route(rt, logits, m, bias, offset, num_local, cfg=None, **params):
    """``params``: ``route_approx`` / ``route_tail_bands`` of :func:`run_k3_route_tail`."""
    outs = _outputs(rt, m, num_local)
    kw = {} if cfg is None else dict(num_ctas=cfg[0], warps_per_token=cfg[1], num_warps=cfg[2])
    kw.update(params)
    rt.run_k3_route_tail(logits, m, bias, 2.5, outs[0], outs[1], outs[2:], offset, num_local, **kw)
    return outs


def _assert_same(a, b, what):
    for i, (x, y) in enumerate(zip(a, b)):
        xv = x.view(torch.int32) if x.dtype == torch.float32 else x
        yv = y.view(torch.int32) if y.dtype == torch.float32 else y
        assert torch.equal(xv, yv), f"{what}: output {i} differs"


# (num_ctas, warps_per_token, num_warps) configurations compared besides the helix8 bands.
CANDIDATES = [
    (1, 2, 8),
    (1, 1, 16),
    (2, 1, 8),
    (2, 2, 16),
    (4, 1, 8),
    (4, 2, 16),
    (4, 4, 16),
    (8, 1, 4),
    (8, 2, 16),
    (8, 4, 16),
    (8, 4, 32),
]


@pytest.mark.parametrize("m", [8, 32, 48, 64, 96, 128, 144, 192])
@pytest.mark.parametrize("approx", [False, True])
@pytest.mark.parametrize("mode", ["realistic", "ties"])
def test_configs_route_bit_identical(m, approx, mode):
    rt = _module()
    logits, bias = router_logits(m, seed=100 + m, mode=mode)
    offset, num_local = 3 * 112, 112  # EP8 rank 3 (helix8 / TEP8)
    ref = _route(rt, logits, m, bias, offset, num_local, route_approx=approx)
    vals, ids = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, TOP_K, 2.5)
    assert torch.equal(ref[0], ids.to(torch.int32))
    assert torch.equal(ref[1].view(torch.int32), vals.to(torch.float32).view(torch.int32))
    bands = [(c, t, w) for _, c, w, t in rt._route_tail_bands("helix8")]
    for cfg in CANDIDATES + bands:
        if (m + cfg[0] - 1) // cfg[0] > 32:
            continue
        got = _route(rt, logits, m, bias, offset, num_local, cfg, route_approx=approx)
        _assert_same(got, ref, f"m={m} cfg={cfg}")


#: Token counts at and around the helix8 band edges.
BAND_EDGE_MS = [
    1, 7, 8, 31, 32, 33, 40, 47, 48, 49, 63, 64, 65, 80, 96, 97, 127, 128, 129, 144, 160, 192,
    255, 256,
]  # fmt: skip


@pytest.mark.parametrize("m", BAND_EDGE_MS)
@pytest.mark.parametrize("approx", [False, True])
@pytest.mark.parametrize("mode", ["realistic", "ties", "rand"])
def test_the_helix8_bands_route_bit_identical(monkeypatch, m, approx, mode):
    """The helix8 bands, through the launcher's own configuration choice (tail_config), route
    every token count at and around their band edges exactly like no bands: all eight outputs
    (top-k ids, weights and the six moe_sort tables, sentinel-filled buffers). The launcher
    compiles the band's configuration with the bands and the default one without them."""
    rt = _module()
    seen = []
    orig = rt._compiled_tail

    def spy(num_ctas, num_warps, warps_per_token, *rest):
        seen.append((num_ctas, warps_per_token, num_warps))
        return orig(num_ctas, num_warps, warps_per_token, *rest)

    monkeypatch.setattr(rt, "_compiled_tail", spy)
    logits, bias = router_logits(m, seed=300 + m, mode=mode)
    offset, num_local = 3 * 112, 112
    ref = _route(rt, logits, m, bias, offset, num_local, route_approx=approx)
    assert seen[-1] == rt.default_config(m)
    got = _route(
        rt, logits, m, bias, offset, num_local, route_approx=approx, route_tail_bands="helix8"
    )
    assert seen[-1] == rt.tail_config(m, "helix8")
    _assert_same(got, ref, f"m={m} helix8 band {rt.tail_config(m, 'helix8')}")


def test_override_cuda_graph_replay():
    """An overridden launch (4 CTAs x 16 warps, two-warp teams) captured in a CUDA graph
    replays on new logits like eager."""
    rt = _module()
    cfg = dict(num_ctas=4, warps_per_token=2, num_warps=16)
    m = 64
    logits, bias = router_logits(m, seed=5, mode="realistic")
    static = logits.clone()
    outs = _outputs(rt, m, 112)
    rt.run_k3_route_tail(static, m, bias, 2.5, outs[0], outs[1], outs[2:], 336, 112, **cfg)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rt.run_k3_route_tail(static, m, bias, 2.5, outs[0], outs[1], outs[2:], 336, 112, **cfg)
    for seed in (11, 12):
        new, _ = router_logits(m, seed=seed, mode="realistic")
        static.copy_(new)
        graph.replay()
        torch.cuda.synchronize()
        ref = _route(rt, new, m, bias, 336, 112)
        torch.cuda.synchronize()
        assert torch.equal(outs[0], ref[0]) and torch.equal(outs[1], ref[1])
        n = int(ref[7].item())
        assert int(outs[7].item()) == n and int(outs[6].item()) == int(ref[6].item())
        assert torch.equal(outs[2][:n], ref[2][:n]) and torch.equal(outs[3][:n], ref[3][:n])
        assert torch.equal(outs[4], ref[4])


# Top-k-only launches (no sort), (num_ctas, warps_per_token, num_warps): clusters up to 8 CTAs, plain
# grids above (dep16's router_tf32_splitk routing runs 96 CTAs of 4-warp teams).
TOPK_GRIDS = [(8, 1, 16), (8, 2, 24), (24, 4, 16), (32, 4, 12), (48, 4, 8), (96, 4, 4), (24, 7, 28)]


def _fold(parts, m):
    """The split-K partials' sum over rows [0, m) in split order, FP32."""
    acc = parts[0, :m].clone()
    for part in parts[1:]:
        acc = acc + part[:m]
    return acc


@pytest.mark.parametrize("m", [1, 7, 48, 96, 160])
@pytest.mark.parametrize("splits", [1, 2, 8])
@pytest.mark.parametrize("mode", ["realistic", "ties"])
def test_topk_grids_route_split_partials_bit_identical(m, splits, mode):
    """Without the sort, every launch above (plain grids of up to MAX_TOPK_CTAS CTAs included) on
    ``splits`` partials ``[S, R, 896]`` (R > m) routes like noaux_tc_op on their sum in split
    order, bit for bit, and its ``logits_out`` is that sum. A lane whose later partials fit in
    _SPLIT_PREFETCH_REGS loads them all before the first add; the adds keep split order."""
    rt = _module()
    logits, bias = router_logits(m + 8, seed=500 + m, mode=mode)
    gen = torch.Generator(device="cuda").manual_seed(600 + m)
    noise = torch.randn((splits - 1, m + 8, logits.shape[1]), device="cuda", generator=gen) * 0.5
    parts = torch.cat([(logits - noise.sum(0))[None], noise]) if splits > 1 else logits
    summed = _fold(parts, m) if splits > 1 else logits[:m]
    vals, ids = torch.ops.trtllm.noaux_tc_op(summed.contiguous(), bias, 1, 1, TOP_K, 2.5)
    assert rt.MAX_TOPK_CTAS == 96
    for ctas, team, warps in TOPK_GRIDS:
        if (m + ctas - 1) // ctas > 32:
            continue
        got_ids = torch.full((m, TOP_K), SENTINEL, dtype=torch.int32, device="cuda")
        got_w = torch.full((m, TOP_K), float(SENTINEL), device="cuda")
        out = torch.full((m, logits.shape[1]), float("nan"), device="cuda")
        rt.run_k3_route_tail(
            parts, m, bias, 2.5, got_ids, got_w, None, 0, logits.shape[1], logits_out=out,
            num_ctas=ctas, warps_per_token=team, num_warps=warps, route_approx=True,
        )  # fmt: skip
        torch.cuda.synchronize()
        what = f"m={m} splits={splits} cfg={(ctas, team, warps)}"
        assert torch.equal(got_ids, ids.to(torch.int32)), what
        assert torch.equal(got_w.view(torch.int32), vals.float().view(torch.int32)), what
        assert torch.equal(out.view(torch.int32), summed.contiguous().view(torch.int32)), what


def test_a_sorting_launch_stays_within_a_cluster():
    """The sort broadcasts over distributed shared memory: it takes at most MAX_CTAS CTAs."""
    rt = _module()
    with pytest.raises(ValueError):
        rt.K3RouteTailKernel(9, 4, 4, 1, True, True)
    rt.K3RouteTailKernel(96, 4, 4, 8, False, True)
    with pytest.raises(ValueError):
        rt.K3RouteTailKernel(97, 4, 4, 1, False, True)
