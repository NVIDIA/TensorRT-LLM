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
"""Collective checks of routed_moe_into_hop: routed_moe's finalize kernel -> post_moe_hop routed-input hop.

Used by ``test_routed_into_hop_mpi.py``. Every function here is collective over the two-host TP8
group (call it in the same order on every rank).

A per-rank *producer* decides what a rank hands to the post_moe_hop for one call:

* ``routed_moe_into_hop``: routed_moe writes into the hop's routed-input buffer ->
  ``hop.hop_routed`` (the cluster kernel's routed-input variant of the hop's mode,
  ``TRTLLM_STAIRCASE_GLM_LL_ROUTED_MOE_INTO_HOP_ACCUMULATION``, default "fp32":
  "row" = finalized by ``RoutedMoeFinalizeKernel``, "identity" = identity-layout raw
  rows, "fp32" = FP32 accumulator added by the routed kernel);
* ``routed_moe``: routed_moe raw tuple -> ``hop(...)`` (default variant);
* ``native``: the native TRT-LLM-Gen raw tuple -> ``hop(...)``.

The hop is constructed, and its routed input enabled, inside the served
model loader's ``MetaInitMode`` (``served_construction``), as
the target's decoder-layer adoption does in the server.

Ranks may pick different producers for the same call (a rank whose routed_moe
declines routed_moe_into_hop keeps the raw tuple); the collective order is one hop per call
on every rank either way. The oracle of a call is the same per-rank choice
with ``routed_moe_into_hop`` replaced by ``routed_moe``, i.e. every rank on the default (raw-tuple)
variant. In modes "row" and "identity" routed_moe_into_hop's local partial must be
bit-identical to the raw variant's, so the reduced outputs must be
bit-identical too. In mode "fp32" the top-k sum order differs from the native
finalize: the published partial must lie within the FP32 reorder bound of the
raw variant's partial (``REORDER_REL * sum_k |w y| + ulp_bf16``), and the
outputs must equal the native-arithmetic emulation of the published partials.
"""

import os
import random
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "routed_moe"))
sys.path.insert(
    0, os.path.join(os.path.dirname(HERE), "..", "..", "shared", "kernels", "post_moe_hop")
)
import post_moe_hop_mpi_utils as U  # noqa: E402
import routed_moe_chain_bench as CB  # noqa: E402
import routed_moe_fixtures as fx  # noqa: E402

FEATURES = {
    "routed_moe_into_hop": "post_moe_hop,routed_moe,routed_moe_into_hop",
    "routed_moe": "routed_moe",
    "native": "",
}
NORM_MAX_ULP = 2
# eight exact FP32 products summed in another order: |delta| <= 7 * 2^-24 * sum|w y|
REORDER_REL = 2.0**-20


_WINDOWS = {}


def windows(layer: int, rows: int, count: int, offset: int = 0):
    """``count`` real hidden-state windows of the captured fixture (spread over it)."""
    key = (layer, rows)
    if key not in _WINDOWS:
        _WINDOWS[key] = [x for x, _ in fx.real_windows(fx.load_real_capture(layer), rows)]
    win = _WINDOWS[key]
    step = max(1, len(win) // count)
    return [win[(offset + i * step) % len(win)].cuda() for i in range(count)]


def served_construction():
    """The context in which the served model loader constructs the model.

    ``PyTorchModelLoader`` builds the model inside ``MetaInitMode``: ``empty``
    and ``empty_like`` produce meta tensors, and afterwards only module
    parameters and buffers are materialized. The target's decoder-layer adoption
    builds the post_moe_hop and enables its routed input inside that context, so
    the harness does the same.
    """
    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode

    return MetaInitMode()


_ROUTED_BUFFERS = {
    "row": ("_routed_buffer",),
    "identity": ("_routed_rows", "_routed_scales"),
    "fp32": ("_routed_acc",),
}


def routed_buffers(hop):
    """``{attribute: tensor or None}`` of the routed-input buffers of the hop's mode."""
    return {name: getattr(hop, name, None) for name in _ROUTED_BUFFERS[hop.routed_mode]}


def buffer_report(hop):
    """Where each routed-input buffer of the hop's mode lives (for logs)."""
    return {
        name: None if t is None else f"{t.device} meta={t.is_meta} ptr={t.data_ptr():#x}"
        for name, t in routed_buffers(hop).items()
    }


# MoE layers of GLM-5.2 that build a post_moe_hop (layers 3..77)
MOE_LAYERS = 75


def buffer_bytes(hop) -> int:
    """Device bytes of the routed-input buffers of the hop's mode."""
    return sum(t.numel() * t.element_size() for t in routed_buffers(hop).values() if t is not None)


def buffer_problems(hop):
    """Why the routed-input buffers are not usable device memory ([] when they are)."""
    device = hop.workspace.state.device
    problems = []
    for name, t in routed_buffers(hop).items():
        if t is None:
            problems.append(f"{name} missing")
        elif t.is_meta:
            problems.append(f"{name} is a meta tensor (data_ptr {t.data_ptr():#x})")
        elif t.device != device:
            problems.append(f"{name} on {t.device}, the hop's workspace on {device}")
        elif t.data_ptr() == 0 or t.data_ptr() % 16:
            problems.append(f"{name} data_ptr {t.data_ptr():#x}")
    return problems


def build_hop(env, producer, mode=None):
    """The post_moe_hop with routed input, constructed as the served model does.

    Inside ``served_construction()``, with the layer's MoE module as the
    declared raw-tuple producer, followed by the collective
    ``enable_routed_input(mode)``. Returns ``(hop, enabled)``.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHop,
    )

    with served_construction():
        hop = PostMoeHop(env.mapping, fx.HIDDEN, fx.TOP_K, torch.bfloat16, raw_producer=producer)
        enabled = hop.engaged and hop.enable_routed_input(mode)
    return hop, enabled


def build_stack(env, layers, check_buffers=True):
    """Served MoE layers (real weights), the post_moe_hop with routed input on.

    With ``check_buffers`` the routed-input buffers must be device memory on
    every rank: the served construction must not leave them on the meta device.
    """
    CB.set_features("")
    moes, gammas = CB.build_layers(env, layers)
    hop, enabled = build_hop(env, next(iter(moes.values())).experts)
    assert hop.engaged, hop.reason
    assert enabled
    if check_buffers:
        problems = buffer_problems(hop)
        assert U.all_ranks(not problems), f"rank {env.rank}: {problems}"
    for moe in moes.values():
        moe._routed_into_hop = hop
    return moes, gammas, hop


def produce(moe, x, producer: str):
    """The MoE half of one call (a decode step); returns ``(kind, payload)`` without the hop."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    CB.set_features(FEATURES[producer])
    with dispatch.step(dispatch.Phase.DECODE):
        out = moe.forward(x, do_finalize=False)
    if len(out) == 2:
        assert producer == "routed_moe_into_hop", f"{producer}: unexpected [shared, routed] result"
        return "routed", out
    assert producer != "routed_moe_into_hop", (
        "routed_moe_into_hop did not engage (raw tuple returned)"
    )
    return "raw", out


def hop_call(hop, kind, payload, res, gamma, with_shared: bool):
    """The post_moe_hop half of one call (collective)."""
    if kind == "routed":
        shared, routed = payload
        return hop.hop_routed(routed, shared if with_shared else None, res, gamma, fx.RMS_EPS)
    shared, fc2, scale, perm = payload
    return hop(fc2, scale, perm, shared if with_shared else None, res, gamma, fx.RMS_EPS)


def local_partial(kind, payload, with_shared: bool, hidden: int = fx.HIDDEN, hop=None):
    """The rank partial the hop publishes: ``sanitize(bf16(routed + shared))``."""
    if kind == "routed":
        shared, routed = payload
        if routed.dtype == torch.float32:
            routed = routed.to(torch.bfloat16)
        elif routed.shape[0] != shared.shape[0]:
            # identity-layout raw rows; the routing weights are in the hop
            rows = shared.shape[0]
            ident = torch.arange(rows * fx.TOP_K, dtype=torch.int32, device=routed.device)
            routed = U.reference_local_partial(
                routed, hop._routed_scales[:rows], ident.view(rows, fx.TOP_K), None, hidden
            )
        local = routed if not with_shared else shared + routed
    else:
        shared, fc2, scale, perm = payload
        local = U.reference_local_partial(fc2, scale, perm, shared if with_shared else None, hidden)
    return U.sanitize_bf16(local)


def call(moe, hop, x, res, gamma, producer: str, with_shared: bool):
    """One call; returns ``(outputs, kind, payload)``.

    In mode "fp32" the returned payload holds a copy of the accumulator taken
    before the hop read and cleared the hop's own buffer.
    """
    kind, payload = produce(moe, x, producer)
    seen = payload
    if kind == "routed" and payload[1].dtype == torch.float32:
        seen = [payload[0], payload[1].clone()]
    out = hop_call(hop, kind, payload, res, gamma, with_shared)
    return out, kind, seen


def exact_mode(hop) -> bool:
    """Whether routed_moe_into_hop's partial must equal the raw variant's bit for bit."""
    return getattr(hop, "routed_mode", "row") != "fp32"


def reorder_ok(local, ref_payload, with_shared: bool):
    """FP32-reorder bound of a routed_moe_into_hop "fp32" partial against the raw variant's."""
    shared, fc2, scale, perm = ref_payload
    ref = local_partial("raw", ref_payload, with_shared)
    routed_ref = local_partial("raw", ref_payload, False)
    idx = perm.to(torch.long).clamp(min=0)
    rows = fc2[idx.view(-1)].view(perm.shape[0], perm.shape[1], -1).float()
    mag = (scale.float()[:, :, None] * rows).abs().sum(1)
    # one BF16 rounding of the routed sum, one of the shared add (|x| 2^-7 >= ulp(x))
    ulp = (ref.float().abs() + routed_ref.float().abs()).clamp(min=2.0**-126) * 2.0**-7
    bound = REORDER_REL * mag + ulp
    diff = (local.float() - ref.float()).abs()
    return bool((diff <= bound).all()), int((diff > 0).sum())


def oracle_producer(producer: str) -> str:
    return "routed_moe" if producer == "routed_moe_into_hop" else producer


def check_emulation(env, hop, kind, payload, with_shared, res, gamma, out, res_out):
    """Published slots == gathered expected partials; outputs == native emulation.

    Collective (gathers every rank's expected partial). Must directly follow
    the checked call (reads the workspace's last buffer).
    """
    rows = res.shape[0]
    mine = local_partial(kind, payload, with_shared, hop=hop)
    expected = U.gather_bf16(mine).permute(1, 0, 2).cuda()
    published = U.published_partials(hop, rows)
    slots_exact = torch.equal(published.view(torch.int16), expected.view(torch.int16))
    emu_norm, emu_res = U.emulate_post_reduction(expected, res, gamma, fx.RMS_EPS)
    res_exact = torch.equal(res_out.view(torch.int16), emu_res.view(torch.int16))
    norm_ulp = U.compare_stats(out, emu_norm)["max_ulp"]
    return slots_exact and res_exact and norm_ulp <= NORM_MAX_ULP, {
        "slots_exact": slots_exact,
        "res_exact": res_exact,
        "norm_max_ulp": norm_ulp,
    }


def same_or_close(hop, got, ref) -> bool:
    """Bit-equal in the exact modes, close in mode "fp32".

    In mode "fp32" the routed_moe_into_hop ranks' top-k sums are reordered, so the reduced
    outputs differ by the BF16 rounding of the reordered partials (bounded
    exactly per call by ``run_mixed``; here a small bound on residual and norm).
    """
    if exact_mode(hop):
        return identical(got, ref)
    ok = True
    for a, b in zip(got, ref):
        d = (a.float() - b.float()).abs()
        ok &= bool((d <= 2.0**-6 * b.float().abs() + 2.0**-10).all())
    return ok


def identical(a, b) -> bool:
    return torch.equal(a[0].view(torch.int16), b[0].view(torch.int16)) and torch.equal(
        a[1].view(torch.int16), b[1].view(torch.int16)
    )


def warm(env, moes, rows_list, producers=("routed_moe_into_hop", "routed_moe", "native")):
    """Compile every producer variant on every rank before any collective call.

    A rank that compiles (seconds) while its peers already poll inside the post_moe_hop
    kernel would make them hit the spin limit; this runs only the MoE half.
    """
    for moe in moes.values():
        for rows in rows_list:
            x = torch.zeros(rows, fx.HIDDEN, dtype=torch.bfloat16, device="cuda")
            for producer in producers:
                produce(moe, x, producer)
    CB.set_features("")
    torch.cuda.synchronize()
    env.comm.barrier()


def mixed_pattern(env, pattern: str):
    """Per-rank producer of a mixed-variant call."""
    if pattern == "odd-routed_moe":
        return "routed_moe" if env.rank % 2 else "routed_moe_into_hop"
    if pattern == "rank0-routed_moe":
        return "routed_moe" if env.rank == 0 else "routed_moe_into_hop"
    if pattern == "host1-routed_moe":
        return "routed_moe" if env.rank >= env.world // 2 else "routed_moe_into_hop"
    if pattern == "odd-native":
        return "native" if env.rank % 2 else "routed_moe_into_hop"
    if pattern == "all-routed_moe_into_hop":
        return "routed_moe_into_hop"
    raise ValueError(pattern)


def run_mixed(env, moes, gammas, hop, pattern, with_shared, rows_list, per_rows=3, seed=0):
    """Mixed-variant calls vs their all-raw-variant oracle; returns (ok, n, failures)."""
    producer = mixed_pattern(env, pattern)
    g = torch.Generator(device="cuda").manual_seed(900 + seed)
    failures, n = [], 0
    for layer in moes:
        for rows in rows_list:
            for i, x in enumerate(windows(layer, rows, per_rows, offset=seed)):
                res = (torch.randn(rows, fx.HIDDEN, device="cuda", generator=g) * 4.0).bfloat16()
                got, kind, payload = call(
                    moes[layer], hop, x, res, gammas[layer], producer, with_shared
                )
                torch.cuda.synchronize()
                emu_ok, stats = check_emulation(
                    env, hop, kind, payload, with_shared, res, gammas[layer], *got
                )
                mine = local_partial(kind, payload, with_shared, hop=hop)
                ref, ref_kind, ref_payload = call(
                    moes[layer], hop, x, res, gammas[layer], oracle_producer(producer), with_shared
                )
                torch.cuda.synchronize()
                if kind == "routed" and not exact_mode(hop):
                    same, n_diff = reorder_ok(mine, ref_payload, with_shared)
                    stats["reorder_elems"] = n_diff
                else:
                    same = identical(got, ref)
                if not (same and emu_ok):
                    failures.append((layer, rows, i, stats, same))
                n += 1
    CB.set_features("")
    return not failures, n, failures


def run_stress(
    env,
    moes,
    gammas,
    hop,
    batches: int,
    batch: int,
    max_delay_us: float,
    rows_list=(1, 6, 8),
    seed: int = 0,
):
    """Back-to-back batches with random per-rank producers and device delays.

    No host synchronization separates the calls of a batch; every call is
    checked bit for bit against its all-raw-variant oracle afterwards. The row
    count, layer and window of a call are drawn from a generator seeded
    identically on every rank (collective shapes must agree); the producer,
    the shared-output choice and the delay are drawn per rank. Kernel-internal
    delays come from ``routed_moe.DELAY_INJECTION`` if the caller set it (and compiled them
    with ``warm``).
    """
    common = random.Random(31 + seed)
    local = random.Random(1000 + 17 * seed + env.rank)
    cycles_per_us = 2000.0  # approximate sm_103 SM clock; only the magnitude matters
    layers = list(moes)
    gen = torch.Generator(device="cuda").manual_seed(4000 + seed)
    failures, calls, delayed, producers_seen = [], 0, 0, set()
    for b in range(batches):
        plan = []
        for j in range(batch):
            layer = layers[common.randrange(len(layers))]
            rows = rows_list[common.randrange(len(rows_list))]
            x = windows(layer, rows, 1, offset=common.randrange(1 << 16))[0]
            res = (torch.randn(rows, fx.HIDDEN, device="cuda", generator=gen) * 4.0).bfloat16()
            producer = ("routed_moe_into_hop", "routed_moe_into_hop", "routed_moe", "native")[
                local.randrange(4)
            ]
            with_shared = local.random() < 0.75
            delay = local.uniform(0.0, max_delay_us) if local.random() < 0.3 else 0.0
            plan.append((layer, x, res, producer, with_shared, delay))
            producers_seen.add(producer)
        torch.cuda.synchronize()
        env.comm.barrier()
        outs = []
        for layer, x, res, producer, with_shared, delay in plan:
            if delay:
                torch.cuda._sleep(int(delay * cycles_per_us))
                delayed += 1
            got, _, _ = call(moes[layer], hop, x, res, gammas[layer], producer, with_shared)
            outs.append(got)
        torch.cuda.synchronize()
        for j, ((layer, x, res, producer, with_shared, _), got) in enumerate(zip(plan, outs)):
            ref, _, _ = call(
                moes[layer], hop, x, res, gammas[layer], oracle_producer(producer), with_shared
            )
            torch.cuda.synchronize()
            if not same_or_close(hop, got, ref):
                failures.append((b, j, layer, x.shape[0], producer, with_shared))
        calls += len(plan)
    CB.set_features("")
    bits = hop.error_bits()
    return not failures and bits == 0, {
        "calls": calls,
        "delayed_on_rank": delayed,
        "producers_on_rank": sorted(producers_seen),
        "error_bits": bits,
        "failures": failures[:6],
    }


def run_graph(env, moes, gammas, hop, pattern: str, replays: int = 4, rows: int = 6):
    """A multi-stream CUDA graph (as served) of four mixed-producer calls.

    Per-rank producers come from ``pattern``, the shared output alternates;
    the graph is replayed on fresh real windows and every replay is checked
    bit for bit against the eager all-raw-variant oracle. Returns
    ``(ok, replays_checked)``.
    """
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    layers = list(moes)
    producer = mixed_pattern(env, pattern)
    plan = [(layers[i % len(layers)], i % 2 == 0) for i in range(4)]
    wins = {layer: windows(layer, rows, replays + 1, offset=11) for layer in layers}
    xs = [wins[layer][0].clone() for layer, _ in plan]
    g = torch.Generator(device="cuda").manual_seed(77)
    ress = [
        (torch.randn(rows, fx.HIDDEN, device="cuda", generator=g) * 4.0).bfloat16() for _ in plan
    ]

    def body():
        return [
            call(moes[layer], hop, x, r, gammas[layer], producer, sh)[0]
            for (layer, sh), x, r in zip(plan, xs, ress)
        ]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            body()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    env.comm.barrier()
    graph = torch.cuda.CUDAGraph()
    with with_multi_stream(True):
        with torch.cuda.graph(graph, stream=stream):
            outs = body()
    torch.cuda.synchronize()
    ok = True
    for rep in range(1, replays + 1):
        for (layer, _), x in zip(plan, xs):
            x.copy_(wins[layer][rep])
        torch.cuda.synchronize()
        env.comm.barrier()
        graph.replay()
        torch.cuda.synchronize()
        for (layer, sh), x, r, got in zip(plan, xs, ress, outs):
            ref, _, _ = call(moes[layer], hop, x, r, gammas[layer], oracle_producer(producer), sh)
            torch.cuda.synchronize()
            ok &= same_or_close(hop, got, ref)
    CB.set_features("")
    del graph
    return ok and hop.error_bits() == 0, replays
