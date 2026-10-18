# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the split-KV decode attention (``split_kv``) on real weights in the TP8 decode shape.

Run like ``test_modeling_v2_musespark_open_decode_forward.py`` (eight MPI ranks on two
NVLink-connected four-GPU GB300 nodes, the checkpoint in ``MUSESPARK_REFERENCE_CKPT``, the
serving configuration's switches), whose model build, KV cache, decode runs and comparisons
this test reuses with its own arms: the same six checkpoint layers (model layers 1 and 5
are full-attention NoPE layers), the FP8 ``KVCacheManagerV2`` with 32-token pages,
one request prefilled past the window, then decode steps of M = 1..8 rows and the
fallback widths 9 and 16, eagerly and replayed from a CUDA graph per arm.

Arms: the built-in model; ``spill_free`` = ``preattn_prep,prep_append,split_kv`` (the split-KV
attention over the bit-identical prep alone); ``a0`` = a target set without ``split_kv``;
``a`` = ``a0`` + ``split_kv``. ``a0`` is the tier-3 target set (``TIER3_TARGET``) +
``prep_append,prep_append_prefetch_trim``, or with ``MODELING_V2_SPLIT_KV_BASE=tier4`` the
tier-4 target set (``TIER4_TARGET``: also the sliding-window attention block and the direct
publish, whose o_proj hand-off then follows ``combine_norm_gate``).

The split-KV attention is not bit-identical (``split_kv_attention``), so it is gated by what
it must keep:

* every decode forward of 1..8 rows runs ``full_attention_split_kv`` and ``combine_norm_gate`` on
  both full-attention layers (on the capture in graph mode), and the early-release norm-gate
  on two layers fewer; at 9 and 16 rows neither runs and ``a`` gives ``a0``'s bits, the
  ``spill_free`` arm the built-in model's;
* error vs fp64, per (full-attention layer, context) cell: every eager ``full_attention_split_kv``
  call, merged by ``combine_only``, and the built-in FMHA run on the same call are measured
  against an fp64 attention of the call's own BF16 query over the FP8 cache it reads; over
  all calls of a cell (``_cell_failures``) the worst element of ``full_attention_split_kv``'s error is
  at most the built-in's worst element, and its total mean-abs error (weighted by the
  calls' element counts) at most the built-in's. The cells: each width's decode steps
  (~2.1k keys; both split-KV arms, every step, all ranks), asserted by ``_compare``; with the
  profile, 100k and ~450k keys (all ranks), asserted by the profile test. A single call
  whose worst element exceeds the built-in's on that call is recorded (``_RESULTS``,
  ``calls_over_builtin_max``), not failed;
* the prefill and the cache it wrote equal the built-in model's; ``a``'s KV cache bytes of
  model layers 0 and 1 (whose K/V come from layer 0's output) and layer 0's outputs equal
  ``a0``'s: the split-KV attention writes no cache byte;
* no amplification: ``a``'s relative RMS distance from the built-in model is at most 1.25x
  the sum of ``a0``'s and the ``spill_free`` arm's; every output finite;
* ``a`` is deterministic: a second run gives the same bits, and in graph mode so does a run
  with one rank's replay started ~2 ms late on its GPU;
* every captured decode graph of a split-KV arm logs its census: 2 of 2 full-attention layers
  at 1..8 rows, nothing at 9 and 16, never a warning.

``test_profile_the_full_layers_at_long_context`` (``MODELING_V2_SPLIT_KV_PROFILE=1``) replays one
8-row decode graph of ``a0`` and ``a`` over a random FP8 cache of 100k and ~450k tokens and
writes every rank's replay span, kernel durations and (``a``, from the eager warm-up) the
fp64 errors of both full layers' ``full_attention_split_kv`` calls (``MODELING_V2_COMPOSE_OUT``); it gates
that the step runs, its output is finite and each (layer, context) cell meets the fp64 rule
above.

``MODELING_V2_COMPOSE_OUT=<dir>`` also writes rank 0's per-arm, per-width result table there.
"""

import contextlib
import importlib.util
import json
import math
import os
import statistics

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.full_attention_prep_append import (  # noqa: E501
    attention as full_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.full_attention_split_kv import (  # noqa: E501
    attention as split_kv_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention import (  # noqa: E501
    combine_norm_gate as cng,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention import (  # noqa: E501
    full_attention_split_kv as fa4,
)


def _decode_forward():
    """The decode-forward composition test module, loaded from beside this file as a library."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "..", "test_modeling_v2_musespark_open_decode_forward.py")
    spec = importlib.util.spec_from_file_location("_decode_forward_for_split_kv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = _decode_forward()


def _fixture_body(fixture):
    """The generator function behind a pytest fixture, across pytest's wrappings of it."""
    get = getattr(fixture, "_get_wrapped_function", None)
    if get is not None:
        return get()
    wrapped = getattr(fixture, "__pytest_wrapped__", None)
    if wrapped is not None:
        return wrapped.obj
    return getattr(fixture, "__wrapped__", fixture)


#: The target set under the split-KV attention (``MODELING_V2_SPLIT_KV_BASE``): tier 3 or tier 4.
BASE = os.environ.get("MODELING_V2_SPLIT_KV_BASE", "tier3")
A0 = {"tier3": C.TIER3_TARGET + ",prep_append,prep_append_prefetch_trim", "tier4": C.TIER4_TARGET}[
    BASE
]
#: (arm, the arm's ids (``_compose_plans.plan_of_ids``), the arm whose bits it must equal): no arm
#: here has one (the split-KV attention is not bit-identical); the gates are in ``_compare``.
ARMS = (
    ("builtin", None, None),
    ("spill_free", "preattn_prep,prep_append,split_kv", None),
    ("a0", A0, None),
    ("a", A0 + ",split_kv", None),
)
SPLIT_KV_ARMS = ("spill_free", "a")
#: Model layers 1 and 5 (checkpoint 1 and 61) are the full-attention layers.
FULL_LAYERS = 2
#: Widest decode step the prep append folds (and the split-KV attention then serves).
FOLD_MAX_ROWS = 8
#: Entry point -> (feature ids that run it, call sites per decode forward, widest width).
ENTRIES = dict(C.ENTRIES)
ENTRIES["full_attention_split_kv"] = ({"split_kv"}, FULL_LAYERS, FOLD_MAX_ROWS)
ENTRIES["combine_norm_gate"] = ({"split_kv"}, FULL_LAYERS, FOLD_MAX_ROWS)
#: No-amplification margin (as the decode-forward test's).
MARGIN = 1.25
#: GPU cycles one rank spins before each replay in the delayed-rank run (~2 ms at 2 GHz).
DELAY_CYCLES = 4_000_000

_INT_EXPECTED_COUNTS = C._expected_counts
#: Every eager full_attention_split_kv call's errors (``_probing``), cleared per comparison.
_PROBES: list = []
#: The eager decode steps' (layer, width) cells checked so far, every rank's calls, for the
#: pooled per-layer record (``test_write_the_result_table``).
_SHORT_CELLS: dict = {}


def _cells(probes, context) -> dict:
    """``probes`` (every rank's) grouped by (full-attention layer, ``context``)."""
    cells = {}
    for probe in probes:
        cells.setdefault((probe["layer"], context), []).append(probe)
    return cells


def _cell_summary(probes) -> dict:
    """One (layer, context) cell: worst elements and element-weighted mean-abs errors of
    ``full_attention_split_kv`` and the built-in FMHA over the cell's calls, and the single calls whose
    worst element exceeds the built-in's on that call (reported, not a failure)."""
    weights = [p["elements"] for p in probes]
    total = sum(weights)
    v4_max = max(p["v4_max"] for p in probes)
    builtin_max = max(p["builtin_max"] for p in probes)
    v4_mean = sum(p["v4_mean"] * w for p, w in zip(probes, weights)) / total
    builtin_mean = sum(p["builtin_mean"] * w for p, w in zip(probes, weights)) / total
    over = [p for p in probes if p["v4_max"] > p["builtin_max"]]
    tiny = 1e-30
    return {
        "calls": len(probes),
        "contexts": [min(p["ctx"] for p in probes), max(p["ctx"] for p in probes)],
        "v4_max": v4_max,
        "builtin_max": builtin_max,
        "ratio_max": v4_max / max(builtin_max, tiny),
        "v4_mean": v4_mean,
        "builtin_mean": builtin_mean,
        "ratio_mean": v4_mean / max(builtin_mean, tiny),
        "finite": all(p["finite"] for p in probes),
        "calls_over_builtin_max": len(over),
        "worst_single_call_ratio_max": max(
            p["v4_max"] / max(p["builtin_max"], tiny) for p in probes
        ),
    }


def _cell_failures(cells) -> list:
    """The cells that break the fp64 rule: over the cell's calls ``full_attention_split_kv``'s worst
    element above the built-in FMHA's worst element, or its total mean-abs error above the
    built-in's; or a non-finite output."""
    failed = []
    for key in sorted(cells, key=str):
        s = _cell_summary(cells[key])
        if not s["finite"] or s["v4_max"] > s["builtin_max"] or s["v4_mean"] > s["builtin_mean"]:
            failed.append(f"layer {key[0]} at {key[1]}: {s}")
    return failed


def _expected_counts(selected: frozenset, rows: int) -> dict:
    # The composition test's own counts include the prep append's and its prefetch trim's
    # (folded prep, prefetch) and the split-KV attention's (combine_norm_gate takes the two
    # full-attention layers' norm-gate).
    return _INT_EXPECTED_COUNTS(selected, rows)


def _gather_kv(pool, table, ctx, scale):
    keys = torch.arange(ctx, device=pool.device)
    raw = pool.view(torch.uint8)

    def gather(kv):
        pages = table[kv, keys // fa4.PAGE].long()
        return raw[pages, :, keys % fa4.PAGE].view(torch.float8_e4m3fn).double() / scale

    return gather(0), gather(1)


def _fp64_attention(q, pool, table, ctx, scale):
    """fp64 attention of the BF16 query (never quantized) over the FP8 K/V the cache holds;
    row t sees keys 0 .. ctx - rows + t."""
    rows = q.shape[0]
    heads, kv_heads, dim = fa4.Q_HEADS, fa4.KV_HEADS, fa4.HEAD_DIM
    k, v = _gather_kv(pool, table, ctx, scale)
    q = q.double().view(rows, heads, dim)
    out = torch.empty(rows, heads, dim, dtype=torch.float64, device=q.device)
    group = heads // kv_heads
    for t in range(rows):
        end = ctx - rows + t + 1
        for h in range(kv_heads):
            s = q[t, h * group : (h + 1) * group] @ k[:end, h].T / math.sqrt(dim)
            out[t, h * group : (h + 1) * group] = torch.softmax(s, -1) @ v[:end, h]
    return out.reshape(rows, heads * dim)


def _e4m3_p_bound_used(merged, own, qkv, pool, table, ctx, scale):
    """How much of the kernel's certified E4M3-P bound (the op's GPU test, ``_e4m3_p_bound``)
    the merged partials use against the op's own reference: P rounded to E4M3 moves a
    probability p >= 2^-10 by at most 2^-4 p and a smaller one by at most min(p, 2^-14), plus
    2^-10 p for exp2.approx and the score rounding, and 2^-10 of the weighted |V| for the FP16
    partials and the merge; one BF16 ulp of slack."""
    rows = qkv.shape[0]
    heads, kv_heads, dim = fa4.Q_HEADS, fa4.KV_HEADS, fa4.HEAD_DIM
    k, v = _gather_kv(pool, table, ctx, scale)
    q = qkv[:, : heads * dim].float()
    c = q * 1048577.0
    hi = c - (c - q)

    def e4m3(x):
        return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()

    q = (e4m3(hi) + e4m3(q - hi)).double().view(rows, heads, dim)
    bound = torch.empty(rows, heads, dim, dtype=torch.float64, device=q.device)
    group = heads // kv_heads
    for t in range(rows):
        end = ctx - rows + t + 1
        for h in range(kv_heads):
            s = q[t, h * group : (h + 1) * group] @ k[:end, h].T / math.sqrt(dim)
            p = torch.exp(s - s.amax(-1, keepdim=True))
            dp = torch.where(p >= 2.0**-10, 2.0**-4 * p, p.clamp(max=2.0**-14)) + 2.0**-10 * p
            vv = v[:end, h].abs()
            den = p.sum(-1, keepdim=True)
            bound[t, h * group : (h + 1) * group] = (dp @ vv + 2.0**-10 * (p @ vv)) / den
    bound = bound.reshape(rows, -1)
    top = torch.maximum(merged.double().abs(), own.double().abs())
    _, exponent = torch.frexp(top)
    ulp = torch.ldexp(torch.ones_like(top), exponent - 8)
    return float(((merged.double() - own.double()).abs() / (bound + ulp)).max())


@contextlib.contextmanager
def _probing():
    """Every eager ``full_attention_split_kv`` launch also runs the real FMHA on the same call into a
    scratch output and measures both against ``_fp64_attention`` (into ``_PROBES``)."""
    import flashinfer.decode as real_decode

    launch = split_kv_attention.SplitKvCall.launch

    def probed(self, kwargs):
        launch(self, kwargs)
        if torch.cuda.is_current_stream_capturing():
            return
        rows = self.folded.rows
        part_o, part_lse = self.pending[:2]
        merged = cng.combine_only(part_o, part_lse, rows)
        builtin = torch.empty_like(kwargs["out"])
        real_decode.trtllm_batch_decode_with_kv_cache(**dict(kwargs, out=builtin))
        ctx = int(kwargs["seq_lens"].reshape(-1)[0])
        ref = _fp64_attention(
            self.folded.qkv_out[:rows, : fa4.Q_HEADS * fa4.HEAD_DIM],
            kwargs["kv_cache"][0],
            kwargs["block_tables"][0],
            ctx,
            float(self.kv_scale.reshape(-1)[0]),
        )
        o = (merged.reshape(rows, -1).double() - ref).abs()
        b = (builtin.reshape(rows, -1).double() - ref).abs()
        q = self.folded.qkv_out[:rows, : fa4.Q_HEADS * fa4.HEAD_DIM].float()
        # The op's own reference (its Q_hi + Q_lo, E4M3 cache, exact P): how far the merged
        # partials are from what the kernel is written to compute.
        own = fa4.full_attention_split_kv.reference(
            self.folded.qkv_out[:rows],
            kwargs["kv_cache"][0],
            kwargs["block_tables"][0],
            kwargs["seq_lens"],
            self.kv_scale,
            self.splits,
        )
        _PROBES.append(
            {
                "layer": getattr(self.folded, "layer_idx", None),
                "rows": rows,
                "ctx": ctx,
                "elements": int(o.numel()),
                "v4_max": float(o.max()),
                "v4_mean": float(o.mean()),
                "builtin_max": float(b.max()),
                "builtin_mean": float(b.mean()),
                "v4_vs_own_reference_max": float(
                    (merged.reshape(rows, -1).double() - own.double()).abs().max()
                ),
                "own_reference_vs_fp64_max": float((own.double() - ref).abs().max()),
                "e4m3_p_bound_used": _e4m3_p_bound_used(
                    merged.reshape(rows, -1),
                    own,
                    self.folded.qkv_out[:rows],
                    kwargs["kv_cache"][0],
                    kwargs["block_tables"][0],
                    ctx,
                    float(self.kv_scale.reshape(-1)[0]),
                ),
                "q_absmax": float(q.abs().max()),
                "q_rms": float(q.pow(2).mean().sqrt()),
                "q_above_448": int((q.abs() > 448.0).sum()),
                "finite": bool(torch.isfinite(merged).all()),
            }
        )

    split_kv_attention.SplitKvCall.launch = probed
    try:
        yield
    finally:
        split_kv_attention.SplitKvCall.launch = launch


@contextlib.contextmanager
def _delayed_rank(e, rank: int):
    """``rank`` spins ``DELAY_CYCLES`` on its stream before every graph replay."""
    replay = torch.cuda.CUDAGraph.replay

    def late(self):
        if e.rank == rank:
            torch.cuda._sleep(DELAY_CYCLES)
        return replay(self)

    torch.cuda.CUDAGraph.replay = late
    try:
        yield
    finally:
        torch.cuda.CUDAGraph.replay = replay


@pytest.fixture(scope="module")
def env():
    """The decode-forward test's environment with this file's arms; the folded prep's and the
    split-KV attention's entry points counted."""
    saved = (C.ARMS, C.ENTRIES, C.COMBINED, C.DETERMINISM, C._expected_counts, C._metadata)
    C.ARMS, C.ENTRIES, C.COMBINED, C.DETERMINISM = ARMS, ENTRIES, {}, ()
    C._expected_counts = _expected_counts
    original_metadata = C._metadata

    def metadata(*args, **kwargs):
        # As the prep-append test: give the eager steps' attention workspace its size up front so
        # the first full-attention layer of an eager step folds too.
        md = original_metadata(*args, **kwargs)
        for workspace in (md.workspace, md.cuda_graph_workspace):
            if workspace is not None and workspace.numel() * workspace.element_size() < (1 << 30):
                workspace.resize_((1 << 30) // workspace.element_size())
        return md

    C._metadata = metadata
    patched = [
        (full_attention, "preattn_prep_append", "preattn_prep_append"),
        (fa4, "full_attention_split_kv", "full_attention_split_kv"),
        (cng, "combine_norm_gate", "combine_norm_gate"),
    ]
    if BASE == "tier4" and os.environ.get("MODELING_V2_COMPOSE_TIER") not in ("swa_block", "tier4"):
        # The tier-4 set runs swa_block on the sliding-window layers; the decode-forward test
        # counts that entry point only under its own swa_block / tier4 sets.
        from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.swa_block import (  # noqa: E501
            attention as swa_attention_block,
        )

        patched.append((swa_attention_block, "swa_block_attention", "swa_block_attention"))
    originals = [(module, attr, getattr(module, attr)) for module, attr, _ in patched]
    for module, attr, name in patched:
        setattr(module, attr, C._Counted(name, getattr(module, attr)))
    outputs = full_attention.FoldedPreprocess.outputs

    def counted_outputs(self, args):
        C._COUNTS["preprocess_elided"] = C._COUNTS.get("preprocess_elided", 0) + 1
        return outputs(self, args)

    full_attention.FoldedPreprocess.outputs = counted_outputs
    try:
        with _probing():
            yield from _fixture_body(C.env)()
    finally:
        full_attention.FoldedPreprocess.outputs = outputs
        for module, attr, original in originals:
            setattr(module, attr, original)
        (C.ARMS, C.ENTRIES, C.COMBINED, C.DETERMINISM, C._expected_counts, C._metadata) = saved


def _full_attentions(model):
    return [layer.self_attn for layer in model.layers if not layer.self_attn.is_sliding]


def test_the_arms_have_the_expected_layout(env):
    """The split-KV mixin sits right in front of the prep-append class on the full-attention
    layers of the split-KV arms, and nowhere else."""
    wrong = []
    for arm, selection, _ in ARMS:
        split_kv_on = "split_kv" in C._selection(selection)
        for index, layer in enumerate(env.models[arm].layers):
            mro = type(layer.self_attn).__mro__
            has = split_kv_attention.FullAttnSplitKvAttention in mro
            want = split_kv_on and not layer.self_attn.is_sliding
            if has != want or (
                has
                and (
                    mro[1] is not split_kv_attention.FullAttnSplitKvAttention
                    or not issubclass(mro[2], full_attention.FullAppendAttention)
                )
            ):
                wrong.append(f"{arm} layer {index}: {[c.__name__ for c in mro[:4]]}")
    assert not wrong, wrong


def _compare(e, rows: int, mode: str) -> None:
    """Every arm at ``rows`` against the gates of the module docstring (collective)."""
    graph = mode == "graph"
    runs, report = {}, {}
    _PROBES.clear()
    for arm, selection, _ in ARMS:
        runs[arm], counts = C._run(e, e.models[arm], rows, graph)
        C._RESULTS[(mode, arm, rows)] = {"counts": counts}
        e.comm.Barrier()
    probes = list(_PROBES)
    again, _ = C._run(e, e.models["a"], rows, graph)
    report["a (second run)"] = C._differences(again, runs["a"])
    if graph:
        late = rows % C.WORLD
        with _delayed_rank(e, late):
            delayed, _ = C._run(e, e.models["a"], rows, True)
        report[f"a (rank {late} late)"] = C._differences(delayed, runs["a"])
    served = rows <= FOLD_MAX_ROWS
    builtin_prefill = {k: v for k, v in runs["builtin"].items() if k.startswith("prefill")}
    for arm, selection, _ in ARMS[1:]:
        result = C._RESULTS[(mode, arm, rows)]
        found = C._differences(
            {k: v for k, v in runs[arm].items() if k.startswith("prefill")}, builtin_prefill
        )
        result["distance"] = C._distance(runs[arm], runs["builtin"])
        if not result["distance"]["finite"]:
            found.append("an output is not finite")
        expected = _expected_counts(C._selection(selection), rows)
        counts = result["counts"]
        least = C.STEPS if mode == "eager" else 1
        for entry, sites in expected.items():
            if counts.get(entry, 0) != sites * least:
                found.append(f"{entry} ran {counts.get(entry, 0)} times, expected {sites * least}")
        if counts.get("prefetch_pending") or counts.get("prefetch_dropped"):
            found.append(
                f"dense_gemm_prefetch forks pending {counts.get('prefetch_pending')} "
                f"dropped {counts.get('prefetch_dropped')}"
            )
        report[arm] = found
    result = C._RESULTS[(mode, "a", rows)]
    result["distance_vs_a0"] = C._distance(runs["a"], runs["a0"])
    if not served:
        # The prep append does not fold these steps, so the split-KV attention never acts: a0's
        # bits, and the built-in's for the spill_free arm.
        report["a"] += C._differences(runs["a"], runs["a0"])
        report["spill_free"] += C._differences(runs["spill_free"], runs["builtin"])
    else:
        same = [
            name
            for name in runs["a0"]
            if name.startswith("cache/layer0.")
            or name.startswith("cache/layer1.")
            or "/layer0." in name
        ]
        report["a"] += C._differences(
            {name: runs["a"][name] for name in same}, {name: runs["a0"][name] for name in same}
        )
        whole = result["distance"]["rel_rms"]
        parts = (
            C._RESULTS[(mode, "a0", rows)]["distance"]["rel_rms"]
            + C._RESULTS[(mode, "spill_free", rows)]["distance"]["rel_rms"]
        )
        result["singles_sum"] = parts
        if whole > MARGIN * parts:
            report["a"].append(f"rel RMS {whole:.3e} > {MARGIN} x (a0 + spill_free) {parts:.3e}")
    if not graph:
        want = len(SPLIT_KV_ARMS) * FULL_LAYERS * C.STEPS if served else 0
        found = [] if len(probes) == want else [f"{len(probes)} probed calls, expected {want}"]
        for probe in probes:
            probe["rank"] = e.rank
            probe["ratio_max"] = probe["v4_max"] / max(probe["builtin_max"], 1e-30)
            probe["ratio_mean"] = probe["v4_mean"] / max(probe["builtin_mean"], 1e-30)
        # The fp64 rule per (layer, context) cell: this width's decode steps on every rank.
        every_probe = [p for rank_probes in e.comm.allgather(probes) for p in rank_probes]
        cells = _cells(every_probe, f"eager decode, {rows} rows")
        _SHORT_CELLS.update(cells)
        found += _cell_failures(cells)
        C._RESULTS[(mode, "probes", rows)] = {
            "probes": probes,
            "cells": {f"layer {k[0]}": _cell_summary(v) for k, v in cells.items()},
        }
        report["fp64 cells"] = found
    everyone = e.comm.allgather({arm: found for arm, found in report.items() if found})
    failed = {
        f"rank {rank} {arm}": found[:6]
        for rank, bad in enumerate(everyone)
        for arm, found in bad.items()
    }
    assert not failed, f"{mode} at {rows} rows: {failed}"


@pytest.mark.parametrize("rows", C.ROWS)
def test_split_kv_in_eager_decode(env, rows):
    _compare(env, rows, "eager")


@pytest.mark.parametrize("rows", C.ROWS)
def test_split_kv_in_cuda_graphs(env, rows):
    _compare(env, rows, "graph")


def test_every_captured_graph_reports_its_partials(env):
    """The census of each split-KV arm: every captured decode graph logged one line, 2 of 2
    full-attention layers at 1..8 rows, a silent record at 9 and 16, never a warning."""
    captured = {(arm, rows) for (mode, arm, rows) in C._RESULTS if mode == "graph"}
    if not captured:
        pytest.skip("no CUDA-graph comparison ran in this session")
    problems = []
    for arm in SPLIT_KV_ARMS:
        widths = sorted(rows for a, rows in captured if a == arm)
        if not widths:
            continue
        census = _full_attentions(env.models[arm])[0]._split_kv_census
        reports = list(census.reports)
        served = [r for r in reports if r[1] == r[2] == FULL_LAYERS and r[3] == "info"]
        silent = [r for r in reports if r[1] == 0 and r[3] == "silent"]
        other = [r[4] for r in reports if r not in served and r not in silent]
        want_served = sum(1 for rows in widths if rows <= FOLD_MAX_ROWS)
        want_silent = len(widths) - want_served
        if other:
            problems.append(f"{arm}: {other[:2]}")
        if len(served) < want_served or len(silent) < want_silent:
            problems.append(
                f"{arm}: {len(served)} served / {len(silent)} silent captures of {len(reports)}, "
                f"want at least {want_served} / {want_silent}"
            )
    everyone = env.comm.allgather(problems)
    assert not any(everyone), everyone


def _fill_cache(kv, metadata, length: int, seed: int) -> None:
    """Random FP8 K/V in every page of positions 0 .. length - 1 of every layer."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
    from tensorrt_llm.bindings.internal import thop
    from tensorrt_llm.quantization.mode import QuantMode

    for layer in range(len(C.CKPT_LAYERS)):
        attention_layer = kv.layer_offsets[layer]
        total = kv.impl.get_page_index_upper_bound(attention_layer, Role.KEY)
        pool, tables, _ = thop.build_trtllm_gen_kv_cache_metadata(
            metadata.host_kv_cache_pool_pointers,
            metadata.host_kv_cache_pool_mapping,
            metadata.kv_cache_block_offsets,
            attention_layer,
            kv.num_kv_heads_per_layer[attention_layer],
            C.TOKENS_PER_BLOCK,
            kv.head_dim,
            2,
            int(total),
            int(QuantMode.FP8_KV_CACHE),
            0,
            1,
            torch.bfloat16,
        )
        pages = tables[0, :, : -(-length // C.TOKENS_PER_BLOCK)].long().flatten()
        generator = torch.Generator(device="cuda").manual_seed(seed + layer)
        for chunk in pages.split(4096):
            values = torch.randn(
                (chunk.numel(),) + tuple(pool.shape[1:]), generator=generator, device="cuda"
            )
            pool.view(torch.uint8)[chunk] = values.to(torch.float8_e4m3fn).view(torch.uint8)


@torch.inference_mode()
def _profile_long(e, model, ctx: int, rows: int = 8, replays: int = 5) -> dict:
    """One ``rows``-row decode graph over a random cache of ``ctx`` tokens: the replay span
    (CUDA events) and every kernel's duration in ``replays`` profiled replays."""
    kv = C._kv_cache(e, ctx)
    try:
        kv.add_dummy_requests([0], [ctx])
        metadata = C._metadata(e, kv, rows, ctx - rows, ctx - rows, 0)
        _fill_cache(kv, metadata, ctx, seed=ctx)
        metadata = metadata.create_cuda_graph_metadata(1, max_draft_tokens=rows - 1)
        metadata.prepare()
        position_ids = torch.arange(ctx - rows, ctx, dtype=torch.int, device="cuda")[None]
        embeds = e.table[:rows].clone()

        def forward():
            hidden = model(attn_metadata=metadata, position_ids=position_ids, inputs_embeds=embeds)
            e.copies.join()
            return hidden

        _PROBES.clear()
        forward()  # eager: compiles, and probes every full_attention_split_kv call against fp64
        torch.cuda.synchronize()
        probes = list(_PROBES)
        e.comm.Barrier()
        e.sink.clear()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = forward()
        torch.cuda.synchronize()
        e.comm.Barrier()
        spans = []
        for _ in range(20):
            e.comm.Barrier()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            spans.append(start.elapsed_time(end) * 1000.0)
        activities = [torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities) as prof:
            for _ in range(replays):
                e.comm.Barrier()
                graph.replay()
                torch.cuda.synchronize()
        kernels = [
            (event.name, event.time_range.start, event.time_range.elapsed_us())
            for event in prof.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        ]
        finite = bool(torch.isfinite(out).all())
        del graph
        return {
            "ctx": ctx,
            "rows": rows,
            "span_us_median": statistics.median(spans[5:]),
            "finite": finite,
            "probes": probes,
            "kernels": kernels,
        }
    finally:
        e.sink.clear()
        kv.shutdown()


def _summary(kernels, replays: int) -> dict:
    """Per kernel name: calls per replay and median duration (µs)."""
    by_name = {}
    for name, _, duration in kernels:
        by_name.setdefault(name, []).append(duration)
    return {
        name: {"per_replay": len(d) / replays, "median_us": statistics.median(d)}
        for name, d in sorted(by_name.items(), key=lambda item: -sum(item[1]))
    }


@pytest.mark.skipif(
    os.environ.get("MODELING_V2_SPLIT_KV_PROFILE") != "1", reason="MODELING_V2_SPLIT_KV_PROFILE=1"
)
@pytest.mark.parametrize("ctx", (100_000, 450_000))
def test_profile_the_full_layers_at_long_context(env, ctx):
    """a0 and a, one 8-row decode graph each over a random cache of ``ctx`` tokens, every
    rank: replay span and kernel table (written to ``MODELING_V2_COMPOSE_OUT``)."""
    out = {}
    for arm in ("a0", "a"):
        result = _profile_long(env, env.models[arm], ctx)
        result["summary"] = _summary(result.pop("kernels"), 5)
        out[arm] = result
        env.comm.Barrier()
    everyone = env.comm.allgather(out)
    # The fp64 rule per (layer, context) cell over every rank's eager calls of arm a.
    cells = _cells([p for r in everyone for p in r["a"]["probes"]], f"{ctx} keys")
    table = {f"layer {k[0]}": _cell_summary(v) for k, v in cells.items()}
    out_dir = os.environ.get("MODELING_V2_COMPOSE_OUT")
    if out_dir and env.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"profile_split_kv_{ctx}.json"), "w") as f:
            json.dump(everyone, f, indent=1, default=str)
        with open(os.path.join(out_dir, f"cells_split_kv_{ctx}.json"), "w") as f:
            json.dump(table, f, indent=1, sort_keys=True)
    for rank, result in enumerate(everyone):
        print(
            f"\nrank {rank} ctx {ctx}: a0 {result['a0']['span_us_median']:.1f} us, "
            f"a {result['a']['span_us_median']:.1f} us"
        )
    for key, summary in sorted(table.items()):
        print(f"\nctx {ctx} {key}: {summary}")
    assert all(r[arm]["finite"] for r in everyone for arm in ("a0", "a"))
    assert len(cells) == FULL_LAYERS, sorted(cells, key=str)
    failed = _cell_failures(cells)
    assert not failed, failed


def test_write_the_result_table(env):
    assert C._RESULTS, "the composition tests recorded nothing"
    out_dir = os.environ.get("MODELING_V2_COMPOSE_OUT")
    if out_dir and env.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        table = {f"{m}:{a}@{r}": v for (m, a, r), v in sorted(C._RESULTS.items())}
        # Recorded, not gated: each layer's eager decode cells pooled over every width.
        pooled = {}
        for (layer, _), probes in _SHORT_CELLS.items():
            pooled.setdefault(layer, []).extend(probes)
        table["eager:pooled_cells"] = {
            f"layer {layer}": _cell_summary(probes) for layer, probes in sorted(pooled.items())
        }
        table["base"] = {"MODELING_V2_SPLIT_KV_BASE": BASE, "a0": A0}
        with open(os.path.join(out_dir, "compose_split_kv.json"), "w") as f:
            json.dump(table, f, indent=1, sort_keys=True, default=str)
    torch.cuda.synchronize()
