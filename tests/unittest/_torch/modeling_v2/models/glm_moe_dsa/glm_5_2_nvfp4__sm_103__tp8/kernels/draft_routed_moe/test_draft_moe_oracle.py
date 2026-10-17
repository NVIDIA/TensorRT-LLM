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
"""draft_routed_moe against the native ``trtllm_bf16_moe`` on the real MTP-layer rank shard (single GPU).

Every comparison is bit for bit on the BF16 outputs. On a mismatch the stage
diagnostics compare the routing weights with the native ``expert_weights`` and
draft_routed_moe's per-(token, slot) FC2 rows (``debug_fc2``) with the native unfinalized
``gemm2_output`` through ``expanded_idx_to_permuted_idx``.

Environment: ``GLM_MOE_DSA_DRAFT_ROUTED_MOE_JIT_ROOT`` (JIT build directory when the image has no
op), ``GLM_MOE_DSA_DRAFT_ROUTED_MOE_TP_RANK`` (rank shard, default 0),
``GLM_MOE_DSA_DRAFT_ROUTED_MOE_TRIALS`` (calls per row count).
"""

import contextlib
import json
import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import draft_moe_fixtures as fx  # noqa: E402
import draft_moe_jit  # noqa: E402

TRIALS = int(os.environ.get("GLM_MOE_DSA_DRAFT_ROUTED_MOE_TRIALS", "24"))

# Keys of a launch configuration (``_launch``): FC1 and FC2 "ctas,warps" and the flag bits, each
# "<value>" or "<one row>+<more rows>".
FC1, FC2, FLAGS = "fc1", "fc2", "flags"

# Kernel families: the default selectors (TMA-fed kernels: 64 FC1 CTAs of 4 MMA warps at R = 1, 152 CTAs with two
# chains per warp at R > 1 and the triples path above 38 distinct experts with its early trigger, FC1 loads L2
# evict-first at R > 1, the 3-warp FC2 on 128 CTAs, FC2 weight prefetch at R = 1), the
# cp.async kernels, the TMA-fed kernels on other grids (flag 64) and the TMA-fed kernels with two chains per FC1 warp,
# two experts per FC2 warp step, FC1 L2 prefetch and the FC2 weight prefetch (flags 8, 16, 128, 2; the FC1 ring of
# 32 KB stages is 6 deep at R <= 2, 4 deep at R = 3..7 and 3 deep at R = 8), and the FC1 triples path on a 120-CTA
# grid (flag 4096: R > 1 calls whose 128-row tiles exceed the grid but whose 64-row sub-tiles fit three per CTA; the
# others run the regular path, e.g. R = 8 with about 50 distinct experts; flag 8192: the early trigger on that path
# only), and the default R > 1 kernels with the activation streamed through the FC1 ring (flag 65536). All must be
# bit-exact.
FAMILIES = {
    "default": {},
    "cpasync": {
        FC1: "128,2",
        FC2: "96,4",
        FLAGS: "2",
    },
    "tma": {
        FC1: "128,2+152,4",
        FC2: "96,4",
        FLAGS: "96",
    },
    "tmapairs": {
        FC1: "64,2+152,4",
        FC2: "96,4",
        FLAGS: "250",
    },
    "triples": {
        FC1: "64,4+120,4",
        FC2: "128,3",
        FLAGS: "66+12360",
    },
    "streamx": {
        FC1: "64,4+152,4",
        FC2: "128,3",
        FLAGS: "66+77896",
    },
}
DELAYED_FLAGS = {
    "default": ("66+45128", "70+76"),
    "cpasync": ("2", "6"),
    "tma": ("96", "100"),
    "tmapairs": ("250", "254"),
    "triples": ("66+12360", "70+4172"),
    "streamx": ("66+77896", "70+77900"),
}


def _per_rows(raw, parse):
    """``(one row, more rows)`` of a "<value>" or "<one row>+<more rows>" entry."""
    parts = raw.split("+")
    assert len(parts) <= 2, raw
    return parse(parts[0]), parse(parts[-1])


def _ctas_warps(raw):
    ctas, warps = (int(v) for v in raw.split(","))
    return ctas, warps


@contextlib.contextmanager
def _launch(config):
    """``config`` set over draft_routed_moe's launch defaults for the duration of the block."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.draft_routed_moe import (  # noqa: E501
        routed_moe as rm,
    )

    saved = rm.DEFAULT_FC1, rm.DEFAULT_FC2, rm.DEFAULT_FLAGS
    config = config or {}
    try:
        if FC1 in config:
            rm.DEFAULT_FC1 = _per_rows(config[FC1], _ctas_warps)
        if FC2 in config:
            rm.DEFAULT_FC2 = _per_rows(config[FC2], _ctas_warps)
        if FLAGS in config:
            rm.DEFAULT_FLAGS = _per_rows(config[FLAGS], int)
        yield
    finally:
        rm.DEFAULT_FC1, rm.DEFAULT_FC2, rm.DEFAULT_FLAGS = saved


@pytest.fixture(scope="module")
def layer():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("needs an SM100-family GPU")
    origin = draft_moe_jit.ensure_op(
        os.environ.get("GLM_MOE_DSA_DRAFT_ROUTED_MOE_JIT_ROOT", "/tmp/draft_routed_moe_jit")
    )
    print(f"GLM_MOE_DSA_DRAFT_ROUTED_MOE_OP_ORIGIN {origin}", flush=True)
    return fx.build_native(int(os.environ.get("GLM_MOE_DSA_DRAFT_ROUTED_MOE_TP_RANK", "0")))


def _draft_moe(layer, x, logits, debug=False, config=None):
    """draft_routed_moe's output, its debug FC2 rows and whether the TMA-fed kernels ran."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.draft_routed_moe import (  # noqa: E501
        routed_moe as rm,
    )

    with _launch(config):
        dbg = (
            torch.full(
                (x.shape[0], fx.TOP_K, fx.HIDDEN), float("nan"), dtype=torch.bfloat16, device="cuda"
            )
            if debug
            else None
        )
        tma = bool(rm.launch_config(x.shape[0])[2] & rm.FLAG_TMA)
        out = rm.run(layer.module, layer.bias, x, logits, debug_fc2=dbg)
        return out, dbg, tma


def _bits(t):
    return t.contiguous().view(torch.int16)


_IDENTITY = {}


def _identity_layer(layer):
    """The same rank's native leaf with identity-like down projections (built on the first mismatch only)."""
    if layer.tp_rank not in _IDENTITY:
        _IDENTITY[layer.tp_rank] = fx.build_native(layer.tp_rank, identity_fc2=True)
    return _IDENTITY[layer.tp_rank]


def _h_image_order(rows: int = 8, inter: int = 256) -> torch.Tensor:
    """Index map of an intermediate entry stored as its shared-memory image (TMA family): plain [tok, j] -> image."""
    tok = torch.arange(rows)[:, None]
    j = torch.arange(inter)[None, :]
    u = j // 8
    return tok * inter + (((u & ~7) | ((u & 7) ^ (tok & 7))) * 8) + (j & 7)


def _diagnose(layer, x, logits, out_draft_moe, dbg, tma=False):
    """Stage-level comparison of one mismatching call (returns a dict of counts).

    Reads draft_routed_moe's routing record and intermediate scratch, which still hold this call's
    values.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.draft_routed_moe import (  # noqa: E501
        routed_moe as rm,
    )

    scratch = rm._scratch(x.device, create=False)
    route = scratch.route.cpu()
    h_draft_moe = scratch.h.view(64, 8 * (fx.INTER // fx.TP_SIZE)).clone()
    if tma:
        h_draft_moe = h_draft_moe[:, _h_image_order().flatten().to(h_draft_moe.device)]
    h_draft_moe = h_draft_moe.view(64, 8, fx.INTER // fx.TP_SIZE)
    gemm2, weights, idx = layer.native(x, logits, do_finalize=False)[:3]
    rows = x.shape[0]
    idx = idx.reshape(-1).long()
    w_draft_moe = route[129 : 129 + rows * fx.TOP_K].view(torch.float32).reshape(rows, fx.TOP_K)
    slot_d = route[65 : 65 + rows * fx.TOP_K].reshape(rows, fx.TOP_K).long()
    w_nat = weights.float().cpu().reshape(rows, fx.TOP_K)
    fc2_nat = gemm2[idx].reshape(rows, fx.TOP_K, fx.HIDDEN)
    fc2_diff = (_bits(fc2_nat) != _bits(dbg)).sum(-1)
    # Native FC1 + SwiGLU of every (token, slot): gemm2 output of the identity leaf, columns 0..255.
    ident = _identity_layer(layer)
    g_id, _, idx_id = ident.native(x, logits, do_finalize=False)[:3]
    h_nat = g_id[idx_id.reshape(-1).long()].reshape(rows, fx.TOP_K, fx.HIDDEN)[
        ..., : fx.INTER // fx.TP_SIZE
    ]
    h_mine = torch.stack(
        [torch.stack([h_draft_moe[slot_d[t, k], t] for k in range(fx.TOP_K)]) for t in range(rows)]
    )
    h_diff = (h_nat.float() != h_mine.float()).sum(-1)
    return {
        "weights_differ": int((w_draft_moe != w_nat).sum()),
        "fc1_h_differ_per_slot": h_diff.tolist(),
        "fc2_rows_differ_per_slot": fc2_diff.tolist(),
        "out_elems_differ": int((_bits(out_draft_moe) != _bits(layer.native(x, logits))).sum()),
    }


def _check(layer, x, logits, label, config=None):
    native = layer.native(x, logits)
    out, dbg, tma = _draft_moe(layer, x, logits, debug=True, config=config)
    torch.cuda.synchronize()
    differ = int((_bits(out) != _bits(native)).sum())
    if differ:
        info = _diagnose(layer, x, logits, out, dbg, tma)
        print(f"GLM_MOE_DSA_DRAFT_ROUTED_MOE_MISMATCH {label} " + json.dumps(info), flush=True)
    return differ


@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
def test_bit_exact_gaussian(layer, rows, family):
    bad = 0
    for trial in range(TRIALS):
        x = fx.hidden_rows(layer, rows, seed=1000 * rows + trial)
        bad += (
            _check(
                layer,
                x,
                layer.logits(x),
                f"gauss {family} R={rows} trial={trial}",
                config=FAMILIES[family],
            )
            > 0
        )
    print(
        f"GLM_MOE_DSA_DRAFT_ROUTED_MOE_ORACLE gaussian {family} R={rows} calls={TRIALS} mismatching_calls={bad}",
        flush=True,
    )
    assert bad == 0


@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("rows", [1, 6])
def test_bit_exact_embeddings(layer, rows, family):
    """Consecutive real token embeddings (normalized) as activation rows."""
    ids = list(range(1000, 1000 + 64 * rows))
    bad = 0
    for trial in range(8):
        x = fx.embedding_rows(layer, ids[trial * rows : (trial + 1) * rows])
        bad += (
            _check(
                layer,
                x,
                layer.logits(x),
                f"emb {family} R={rows} trial={trial}",
                config=FAMILIES[family],
            )
            > 0
        )
    print(
        f"GLM_MOE_DSA_DRAFT_ROUTED_MOE_ORACLE embeddings {family} R={rows} calls=8 mismatching_calls={bad}",
        flush=True,
    )
    assert bad == 0


@pytest.mark.parametrize("family", list(FAMILIES))
def test_routing_ties(layer, family):
    """Equal selection keys.

    Equal logits for every expert (the bias decides, ties to the lower index), and pairs of experts with equal
    bias and equal logits.
    """
    bad = 0
    x = fx.hidden_rows(layer, 6, seed=7)
    flat = torch.zeros(6, fx.NUM_EXPERTS, device="cuda")
    flat[1] += 3.0
    flat[2] -= 3.0
    bias = layer.bias.float()
    # rows 3..5: logits that make sigmoid + bias equal for groups of experts with equal bias
    for r in range(3, 6):
        flat[r] = torch.where(bias == bias[r * 7], 1.0, -2.0)
    bad += _check(layer, x, flat.contiguous(), "ties", config=FAMILIES[family]) > 0
    big = torch.randn(6, fx.NUM_EXPERTS, device="cuda") * 40.0
    bad += _check(layer, x, big.contiguous(), "saturated logits", config=FAMILIES[family]) > 0
    print(
        f"GLM_MOE_DSA_DRAFT_ROUTED_MOE_ORACLE ties {family} calls=2 mismatching_calls={bad}",
        flush=True,
    )
    assert bad == 0


@pytest.mark.parametrize(
    "config",
    [
        {
            FC1: "148,1",
            FC2: "148,1",
            FLAGS: "0",
        },
        {
            FC1: "64,4",
            FC2: "48,2",
            FLAGS: "6",
        },
        {
            FC1: "1,1",
            FC2: "1,1",
            FLAGS: "2",
        },
        {
            FC1: "128,2",
            FC2: "96,4",
            FLAGS: "26",
        },
        {
            FC1: "37,4",
            FC2: "5,2",
            FLAGS: "24",
        },
        {
            FC1: "1,1+148,4",
            FC2: "148,1+5,2",
            FLAGS: "0+30",
        },
        {
            FC1: "148,2+64,4",
            FC2: "96,4",
            FLAGS: "34+58",
        },
        {
            FC1: "128,4",
            FC2: "37,1",
            FLAGS: "62",
        },
        # Delayed producer of the FC2 grid: one FC1 warp (milliseconds) with the early trigger, so the FC2 CTAs
        # are resident long before the routing record and the intermediate exist.
        {
            FC1: "1,1",
            FC2: "148,4",
            FLAGS: "6",
        },
        # TMA-fed kernels: one CTA running every tile (persistent loop, ring wrap-around across tiles), deep rings
        # with the early trigger, other grid shapes, and a one-CTA FC1 producer ahead of a full FC2 grid.
        {
            FC1: "1,1",
            FC2: "1,1",
            FLAGS: "64",
        },
        {
            FC1: "37,4",
            FC2: "5,2",
            FLAGS: "100",
        },
        {
            FC1: "152,2",
            FC2: "152,4",
            FLAGS: "96",
        },
        {
            FC1: "64,4+256,1",
            FC2: "192,2+96,4",
            FLAGS: "64+96",
        },
        {
            FC1: "1,1",
            FC2: "152,4",
            FLAGS: "68",
        },
        # TMA-fed with two chains per FC1 warp: one CTA of one warp, an odd grid with the early trigger, L2 prefetch
        # and the early FC2 prefetch, grids with more CTAs than tiles, and one 4-warp CTA ahead of a full FC2 grid.
        {
            FC1: "1,1",
            FC2: "1,1",
            FLAGS: "72",
        },
        {
            FC1: "37,4",
            FC2: "5,2",
            FLAGS: "460",
        },
        {
            FC1: "152,2+100,4",
            FC2: "152,4",
            FLAGS: "226+98",
        },
        {
            FC1: "1,4",
            FC2: "152,4",
            FLAGS: "236",
        },
        # TMA-fed FC2 with 3 MMA warps (48-row groups): the full grid of 128 groups and an odd persistent grid.
        {
            FC1: "128,2+152,4",
            FC2: "128,3",
            FLAGS: "66+68",
        },
        {
            FC1: "64,4",
            FC2: "37,3",
            FLAGS: "64",
        },
        # 3-warp FC2 with expert pairs, after FC1 tile pairs at R > 1: the full grid and an odd persistent grid.
        {
            FC1: "64,4+152,4",
            FC2: "128,3",
            FLAGS: "82+90",
        },
        {
            FC1: "32,4+136,4",
            FC2: "37,3",
            FLAGS: "74+88",
        },
        # FC1 triples (flag 4096): the default grid with tile pairs, without tile pairs (64-row tiles), with the FC2
        # weight prefetch after the last copy and right after routing, and a grid too small for one round of
        # triples (regular path).
        {
            FC1: "64,4+152,4",
            FC2: "128,3",
            FLAGS: "66+4172",
        },
        {
            FC1: "152,4",
            FC2: "128,3",
            FLAGS: "4164",
        },
        {
            FC1: "112,4",
            FC2: "96,4",
            FLAGS: "4170+4426",
        },
        {
            FC1: "20,4",
            FC2: "37,3",
            FLAGS: "4168",
        },
    ],
)
def test_launch_configs(layer, config):
    bad = 0
    for rows in (1, 3, 6, 8):
        x = fx.hidden_rows(layer, rows, seed=99 + rows)
        bad += _check(layer, x, layer.logits(x), f"config {config} R={rows}", config=config) > 0
    assert bad == 0


def _distinct(layer, logits):
    """Distinct experts of a call (float32 restatement of the selection; only used to pick test cases)."""
    keys = 0.5 * torch.tanh(0.5 * logits.float()) + 0.5 + layer.bias.float()[None, :]
    return len(set(keys.topk(fx.TOP_K, dim=-1).indices.flatten().tolist()))


@pytest.mark.parametrize("flags", ["24", "58", "16", "80", "90"])
def test_pairs_odd_distinct(layer, flags):
    """Expert pairs in FC2 / tile pairs in FC1 with an odd number of distinct experts.

    An unpaired expert per item makes the pair parity alternate across a warp's items; every warp has several
    items.
    """
    config = {
        FC1: "37,2",
        FC2: "48,1",
        FLAGS: flags,
    }
    bad, odd, seed = 0, 0, 0
    while odd < 8 and seed < 400:
        rows = (2, 3, 5, 7)[seed % 4]
        x = fx.hidden_rows(layer, rows, seed=7000 + seed)
        lg = layer.logits(x)
        seed += 1
        if _distinct(layer, lg) % 2 == 0:
            continue
        odd += 1
        bad += (
            _check(layer, x, lg, f"odd D flags={flags} R={rows} seed={seed - 1}", config=config) > 0
        )
    print(
        f"GLM_MOE_DSA_DRAFT_ROUTED_MOE_ORACLE pairs flags={flags} odd-D calls={odd} mismatching_calls={bad}",
        flush=True,
    )
    assert odd == 8 and bad == 0


# Grids and flags of the triples test: the default grid with tile pairs (one round of triples up to 57 distinct
# experts; early trigger always or on the triples path only), 64-row tiles (8 per expert), and a 120-CTA grid
# (triples from 31 to 45 distinct experts).
TRIPLES_CONFIGS = (
    ("152,4", "128,3", "4172"),
    ("152,4", "128,3", "12360"),
    ("152,4", "128,3", "4160"),
    ("120,4", "96,4", "4168"),
    ("152,4", "128,3", "45128"),
    ("152,4", "128,3", "77896"),
    ("152,4", "128,3", "110664"),
)


@pytest.mark.parametrize("config", TRIPLES_CONFIGS, ids=lambda c: "/".join(c))
def test_triples_distinct(layer, config):
    """R = 5..8 with chosen distinct-expert counts around the FC1 path boundaries.

    The boundaries are 38/39 and 57/58 distinct experts on the 152-CTA grid with tile pairs and 30/31 and 45/46 on
    120 CTAs; every call is bit-exact whichever FC1 path it takes.
    """
    fc1, fc2, flags = config
    pool = fx.RowPool.gaussian(layer)
    cases = [
        (5, 39),
        (5, 40),
        (6, 30),
        (6, 31),
        (6, 38),
        (6, 39),
        (6, 44),
        (6, 45),
        (6, 46),
        (6, 48),
    ]
    cases += [(7, 41), (7, 50), (7, 56), (8, 57), (8, 58), (8, 64)]
    bad, reached = 0, 0
    for i, (rows, target) in enumerate(cases):
        x, d = pool.take(rows, target, seed=3000 + i)
        reached += d == target
        bad += (
            _check(
                layer,
                x,
                layer.logits(x),
                f"triples {config} R={rows} D={d}",
                config={
                    FC1: fc1,
                    FC2: fc2,
                    FLAGS: flags,
                },
            )
            > 0
        )
    print(
        f"GLM_MOE_DSA_DRAFT_ROUTED_MOE_ORACLE triples {config} calls={len(cases)} at target D={reached} "
        f"mismatching_calls={bad}",
        flush=True,
    )
    assert bad == 0 and reached >= len(cases) - 2


def _route_weight_bits(layer, x, logits):
    """(draft_routed_moe routing record weights, native expert_weights) of one call as int32 bit
    patterns of their FP32 values."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.draft_routed_moe import (  # noqa: E501
        routed_moe as rm,
    )

    rows = x.shape[0]
    out = rm.run(layer.module, layer.bias, x, logits)
    scratch = rm._scratch(x.device, create=False)
    fast_w = scratch.route[129 : 129 + rows * fx.TOP_K].clone()
    nat_w = layer.native(x, logits, do_finalize=False)[1].reshape(-1).float().view(torch.int32)
    return fast_w, nat_w, out


def test_route_record(layer):
    """draft_routed_moe's routing weights (its routing record) against the native ``expert_weights``, bit for bit.

    102,400 rows (R = 8; logits of several scales, every eighth call rounded so that keys tie), plus the outputs of
    every 64th call.
    """
    calls, rows = 12800, 8
    x = fx.hidden_rows(layer, rows, seed=4)
    g = torch.Generator(device="cuda").manual_seed(12)
    differ = torch.zeros((), dtype=torch.int64, device="cuda")
    out_bad = 0
    for i in range(calls):
        scale = (0.5, 2.0, 6.0, 30.0)[i % 4]
        lg = torch.randn(rows, fx.NUM_EXPERTS, device="cuda", generator=g) * scale
        if i % 8 == 7:
            lg = torch.round(lg * 4.0) / 4.0
        fast_w, nat_w, out = _route_weight_bits(layer, x, lg.contiguous())
        differ += (fast_w != nat_w).sum()
        if i % 64 == 0:
            out_bad += int((_bits(out) != _bits(layer.native(x, lg.contiguous()))).sum()) > 0
    differ = int(differ)
    print(
        f"GLM_MOE_DSA_DRAFT_ROUTED_MOE_ORACLE route record rows={calls * rows} differing weights={differ} "
        f"mismatching outputs={out_bad}/{calls // 64}",
        flush=True,
    )
    assert differ == 0 and out_bad == 0


def test_route_record_nonfinite(layer):
    """Logits with NaN, +Inf and -Inf entries.

    Routing weights and outputs against native, bit for bit (NaN payloads included).
    """
    rows = 8
    x = fx.hidden_rows(layer, rows, seed=4)
    g = torch.Generator(device="cuda").manual_seed(13)
    w_bad, out_bad, calls = 0, 0, 64
    for i in range(calls):
        lg = torch.randn(rows, fx.NUM_EXPERTS, device="cuda", generator=g) * 2.0
        cols = torch.randint(0, fx.NUM_EXPERTS, (3,), device="cuda", generator=g)
        lg[i % rows, cols[0]] = float("inf")
        lg[(i + 1) % rows, cols[1]] = float("-inf")
        if i % 2:
            lg[(i + 2) % rows, cols[2]] = float("nan")
        fast_w, nat_w, out = _route_weight_bits(layer, x, lg.contiguous())
        nat_out = layer.native(x, lg.contiguous())
        w_differ = (fast_w != nat_w).view(rows, fx.TOP_K)
        o_differ = (_bits(out) != _bits(nat_out)).sum(-1)
        if (bool(w_differ.any()) or bool(o_differ.any())) and w_bad + out_bad < 4:
            from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.draft_routed_moe import (  # noqa: E501
                routed_moe as rm,
            )

            route = rm._scratch(x.device, create=False).route.cpu()
            distinct = route[1:65]
            ids = distinct[route[65 : 65 + rows * fx.TOP_K].long()].view(rows, fx.TOP_K)
            zw, nw = fast_w.view(rows, -1).tolist(), nat_w.view(rows, -1).tolist()
            zo, no = _bits(out), _bits(nat_out)
            for r in range(rows):
                if bool(w_differ[r].any()) or int(o_differ[r]) > 0:
                    print(
                        f"GLM_MOE_DSA_DRAFT_ROUTED_MOE_NONFINITE call={i} row={r} "
                        f"nan_in_row={bool(lg[r].isnan().any())} "
                        f"draft_moe_ids={ids[r].tolist()} fast_w={[hex(v & 0xFFFFFFFF) for v in zw[r]]} "
                        f"native_w={[hex(v & 0xFFFFFFFF) for v in nw[r]]} out_differ={int(o_differ[r])} "
                        f"draft_moe_out_nan={int(out[r].isnan().sum())} native_out_nan={int(nat_out[r].isnan().sum())} "
                        f"draft_moe_out0={hex(int(zo[r, 0]) & 0xFFFF)} native_out0={hex(int(no[r, 0]) & 0xFFFF)}",
                        flush=True,
                    )
        w_bad += bool(w_differ.any())
        out_bad += bool(o_differ.any())
    print(
        f"GLM_MOE_DSA_DRAFT_ROUTED_MOE_ORACLE route record non-finite calls={calls} mismatching "
        f"weights={w_bad} outputs={out_bad}",
        flush=True,
    )
    assert w_bad == 0 and out_bad == 0


@pytest.mark.parametrize("family", list(FAMILIES))
def test_delayed_producer(layer, family):
    """Inputs written late by their producer.

    A copy writes the inputs after a long device-side sleep on the same stream, eagerly and inside a CUDA graph
    (PDL edges as captured), with the default and the early-trigger configurations.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.draft_routed_moe import (  # noqa: E501
        routed_moe as rm,
    )

    bad = 0
    flag_sets = DELAYED_FLAGS[family]
    for flags in flag_sets:
        with _launch({**FAMILIES[family], FLAGS: flags}):
            for rows in (1, 6):
                x0 = fx.hidden_rows(layer, rows, seed=11 + rows)
                lg0 = layer.logits(x0)
                ref = layer.native(x0, lg0)
                xs = torch.zeros_like(x0)
                ls = torch.zeros_like(lg0)
                torch.cuda.synchronize()
                torch.cuda._sleep(20_000_000)
                xs.copy_(x0)
                ls.copy_(lg0)
                out = rm.run(layer.module, layer.bias, xs, ls)
                torch.cuda.synchronize()
                bad += int((_bits(out) != _bits(ref)).sum()) > 0
                xs.zero_()
                ls.zero_()
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    with torch.cuda.graph(graph, stream=stream):
                        torch.cuda._sleep(20_000_000)
                        xs.copy_(x0)
                        ls.copy_(lg0)
                        out_g = rm.run(layer.module, layer.bias, xs, ls)
                for _ in range(3):
                    xs.zero_()
                    ls.zero_()
                    torch.cuda.synchronize()
                    graph.replay()
                    torch.cuda.synchronize()
                    bad += int((_bits(out_g) != _bits(ref)).sum()) > 0
    print(
        f"GLM_MOE_DSA_DRAFT_ROUTED_MOE_ORACLE delayed producer {family} calls=16 mismatching={bad}",
        flush=True,
    )
    assert bad == 0


@pytest.mark.parametrize("family", list(FAMILIES))
def test_graph_replay(layer, family):
    """Captured once, replayed with new inputs copied into the static buffers."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.draft_routed_moe import (  # noqa: E501
        routed_moe as rm,
    )

    with _launch(FAMILIES[family]):
        _graph_replay(layer, rm, family)


def _graph_replay(layer, rm, family):
    for rows in (1, 6):
        xs = torch.empty(rows, fx.HIDDEN, dtype=torch.bfloat16, device="cuda")
        ls = torch.empty(rows, fx.NUM_EXPERTS, dtype=torch.float32, device="cuda")
        x0 = fx.hidden_rows(layer, rows, seed=5)
        xs.copy_(x0)
        ls.copy_(layer.logits(x0))
        rm.run(layer.module, layer.bias, xs, ls)  # scratch exists before capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph, stream=stream):
                out = rm.run(layer.module, layer.bias, xs, ls)
        bad = 0
        for trial in range(6):
            x = fx.hidden_rows(layer, rows, seed=500 + trial)
            xs.copy_(x)
            ls.copy_(layer.logits(x))
            graph.replay()
            torch.cuda.synchronize()
            bad += int((_bits(out) != _bits(layer.native(xs, ls))).sum()) > 0
        print(
            f"GLM_MOE_DSA_DRAFT_ROUTED_MOE_ORACLE graph {family} R={rows} replays=6 mismatching={bad}",
            flush=True,
        )
        assert bad == 0


def test_eligibility(layer):
    """Every unsupported call declines before any device work."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.draft_routed_moe import (  # noqa: E501
        routed_moe as rm,
    )

    x = fx.hidden_rows(layer, 6, seed=3)
    lg = layer.logits(x)
    assert rm.eligibility(layer.module, layer.bias, x, lg, True) == (True, "")
    cases = {
        "unfinalized": (layer.module, layer.bias, x, lg, False),
        "rows 9": (
            layer.module,
            layer.bias,
            fx.hidden_rows(layer, 9, seed=1),
            layer.logits(fx.hidden_rows(layer, 9, seed=1)),
            True,
        ),
        "fp16 activation": (layer.module, layer.bias, x.half(), lg, True),
        "bf16 logits": (layer.module, layer.bias, x, lg.bfloat16(), True),
        "strided logits": (layer.module, layer.bias, x, torch.cat([lg, lg], 1)[:, ::2], True),
        "other bias": (layer.module, layer.bias.clone(), x, lg, True),
        "fp32 bias": (layer.module, layer.bias.float(), x, lg, True),
        "not a module": (object(), layer.bias, x, lg, True),
    }
    for name, args in cases.items():
        ok, reason = rm.eligibility(*args)
        print(f"GLM_MOE_DSA_DRAFT_ROUTED_MOE_DECLINE {name}: {reason}", flush=True)
        assert not ok and reason, name
    assert rm.routed_output(layer.module, layer.bias, x, lg, False) is None
