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
"""Modularized attention perf tests (§12.7 case set).

Two independent detection paths (notes §11.5):
  - DISCRETE tests  -> zero-threshold ``==`` asserts, pre-merge gate (marked ``discrete``)
  - CONTINUOUS test -> gpu_time vs baseline, post-merge detector (marked ``continuous``)

Bootstrap protocol (notes §10.4): on an arch with no golden, the discrete test
SKIPS and prints the observed value (so you bless it after confirming K-run zero
variance); the continuous test SKIPS and records the observed baseline. Every run
appends a row to ``attention_perf_log.csv`` for the experiment report.

Run discrete only (fast, no clock lock):
    pytest tests/microbenchmarks/attention_perf -m discrete -v
Run continuous (needs a quiet/locked GPU):
    pytest tests/microbenchmarks/attention_perf -m continuous -v
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from pathlib import Path

import pytest
import torch
from attention_perf_harness import AttnCase, collect_signals, device_name

_HERE = Path(__file__).parent
_GOLDEN_PATH = _HERE / "golden_attention.json"
# Log dir is overridable: under docker the source tree is NFS root_squash (the
# container's root cannot write here), so point ATTN_PERF_LOG_DIR at a writable
# bind mount to persist the experiment report; otherwise logging is best-effort.
_LOG_PATH = Path(os.environ.get("ATTN_PERF_LOG_DIR", _HERE)) / "attention_perf_log.csv"

_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# Continuous threshold = per-case, variance-based (notes §3.2, revised).
#   threshold_ms = baseline_ms * (1 + max(k_sigma * cv, rel_floor))
# where (baseline_ms, cv) are blessed from a CROSS-SESSION bootstrap (N independent
# containers, each a fresh process/CUDA-context — not one container repeated).
# A flat % is wrong: the 2.79 ms FMHA kernel has cross-session CV~0.03% (a 5% gate
# is ~170x its noise -> blind to real <5% regressions) while the 0.086 ms XQA
# kernel has CV~0.36% (short-kernel noise). Per-case k*cv adapts the gate to each
# kernel's measured noise. rel_floor is the safety floor: the bootstrap so far is
# same-time-window, so it does NOT yet capture multi-hour thermal / cross-build
# drift; the 1% floor absorbs that. Tighten the floor only after a spaced-out
# (multi-hour) bootstrap confirms the kernel stays stable thermally.
_DEFAULT_K_SIGMA = 4.0
_DEFAULT_REL_FLOOR = 0.01
_LEGACY_CONT_THRESHOLD = 1.05  # fallback for scalar (pre-variance) golden entries

# Continuous gpu_time flutters run-to-run (cross-session: fresh CUDA context,
# clock/thermal state, allocator layout). A single run's in-call median can't
# remove that — the whole run's median itself shifts session to session. So the
# continuous path repeats the measurement across R FRESH subprocesses (each an
# independent session) and gates on the median-of-medians, which is robust to a
# lone hot/cold run. R>1 also yields the TRUE cross-run CV — exactly the cv the
# variance gate wants (and the value to bless into the golden). Default R=3 is
# the smallest odd count that gives a real median (drops one outlier run) at 3x
# cost; nightly can raise ATTN_PERF_REPEATS (e.g. 5) for a tighter cv, and R=1
# restores the single-run in-call behavior for a quick local check.
_CONT_REPEATS = max(1, int(os.environ.get("ATTN_PERF_REPEATS", "3")))

# --------------------------------------------------------------------------- #
# golden store — JSON keyed by case_id -> gpu(device name) -> value. CSV log for raw runs.
# Flat files on purpose: this is the research phase, not production monitoring
# (PLATFORM §0). Graduate verified cases to TRT_perf DB / OpenSearch later.
# --------------------------------------------------------------------------- #


def _load_golden() -> dict:
    if _GOLDEN_PATH.exists():
        return json.loads(_GOLDEN_PATH.read_text())
    return {}


def _gpu_key() -> str:
    """GPU identity used to key golden: the DEVICE NAME, not sm_arch.

    One arch spans multiple GPUs (sm90=H100/H200/H20, sm120=RTX-PRO-6000/5090)
    whose perf differs, so a baseline must be pinned to the concrete GPU.
    Resolves the generic "NVIDIA Graphics Device" (e.g. B300) via PCI id. Single
    source of truth for the CSV gpu_name column and the DB `gpu` column.
    """
    return device_name()


def _golden_for(case_id: str, gpu: str):
    return _load_golden().get(case_id, {}).get(gpu)


def _log_run(row: dict) -> None:
    """Append one run to the CSV experiment log.

    Best-effort: a write failure (e.g. read-only NFS mount under docker) must
    never fail the test itself.
    """
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        new = not _LOG_PATH.exists()
        with _LOG_PATH.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)
    except OSError as e:
        print(f"[warn] perf log not written to {_LOG_PATH}: {e}")


def _record(sig, golden, verdict: str, **extra) -> None:
    row = {
        "case_id": sig.case_id,
        # Human-readable pinned-input shape (batch/seq/heads/dtype), so the DB
        # row is self-describing without decoding case_id. Not a golden key.
        "shape": sig.shape,
        "arch": sig.arch,
        # Real device name (e.g. "NVIDIA H100 80GB HBM3") so the DB stores the
        # concrete GPU per run, not just the sm_arch. Auto-correct on any node.
        # Same resolver as the golden key (_gpu_key): keeps CSV gpu_name == the
        # golden/DB key even for B300, which the driver mis-reports as generic.
        "gpu_name": _gpu_key(),
        # Physical node hostname — perf can differ across nodes of the same GPU.
        "node": os.uname().nodename,
        "phase": sig.phase,
        "use_paged_context_fmha": sig.use_paged_context_fmha,
        "launch_count": sig.launch_count,
        "nvrtc_compile_count": sig.nvrtc_compile_count,
        "cubin_present": sig.cubin_present,
        "gpu_time_median_ms": sig.gpu_time_median_ms,
        "gpu_time_p99_ms": sig.gpu_time_p99_ms,
        "observed_cv": None,  # filled by continuous path (this run's in-call CV)
        "threshold_ms": None,  # filled by continuous path (variance-based gate)
        "golden": golden,
        "verdict": verdict,
    }
    row.update(extra)
    _log_run(row)


# --------------------------------------------------------------------------- #
# case definitions (§12.7)
# --------------------------------------------------------------------------- #

_CTX = AttnCase(
    case_id="attn_ctx_dispatch",
    phase="context",
    seq_len=32,
    num_cached_tokens=0,
    chunked_prefill=False,
)
_CTX_TIME = AttnCase(
    case_id="attn_ctx_fmha_time", phase="context", seq_len=2048, num_cached_tokens=0
)
_DEC = AttnCase(
    case_id="attn_decode_dispatch", phase="generation", seq_len=1, num_cached_tokens=1024
)
_DEC_TIME = AttnCase(
    case_id="attn_decode_xqa_time", phase="generation", seq_len=1, num_cached_tokens=1024
)


def _bootstrap_or_skip(case_id, gpu, observed):
    golden = _golden_for(case_id, gpu)
    if golden is None:
        pytest.skip(
            f"[bootstrap] no golden for {case_id} on {gpu!r}. observed={observed!r}. "
            f"Confirm K-run zero variance, then add to {_GOLDEN_PATH.name}."
        )
    return golden


def _observed_cv(raw_times_ms, median) -> float:
    """In-call coefficient of variation from the 50 timed iters of one run.

    Logged each run so the experiment report tracks measurement noise over time;
    the GATE uses the blessed golden CV, not this (a single run's CV is itself
    noisy). Population stdev (we have the whole sample, not an estimate of it).
    """
    if not raw_times_ms or len(raw_times_ms) < 2 or not median:
        return 0.0
    return statistics.pstdev(raw_times_ms) / median


def _continuous_gate(golden_entry):
    """Resolve a continuous golden entry to (baseline_ms, margin, threshold_ms).

    Variance-based entry (dict): {baseline_ms, cv, n, [k_sigma], [rel_floor]} ->
        threshold = baseline * (1 + max(k_sigma*cv, rel_floor)).
    Legacy scalar entry (float, pre-variance bootstrap): falls back to the flat
        baseline * 1.05 gate so old goldens keep working until re-blessed.
    """
    if isinstance(golden_entry, dict):
        baseline = float(golden_entry["baseline_ms"])
        cv = float(golden_entry.get("cv", 0.0))
        k = float(golden_entry.get("k_sigma", _DEFAULT_K_SIGMA))
        floor = float(golden_entry.get("rel_floor", _DEFAULT_REL_FLOOR))
        margin = max(k * cv, floor)
    else:  # legacy scalar baseline
        baseline = float(golden_entry)
        margin = _LEGACY_CONT_THRESHOLD - 1.0
    return baseline, margin, baseline * (1.0 + margin)


def _bootstrap_continuous_msg(case_id, gpu, median, cv, n):
    """Build a skip message handing the user a paste-ready golden entry.

    Prefer a K-repeat bootstrap (cross-run CV) over this single-run CV.
    """
    suggested = {"baseline_ms": round(median, 4), "cv": round(cv, 4), "n": n}
    pytest.skip(
        f"[bootstrap] no golden for {case_id} on {gpu!r}. "
        f"observed median={median:.4f}ms, in-call cv={cv:.4f}. "
        f"Run K repeats, then add the cross-run entry to {_GOLDEN_PATH.name}: "
        f'"{gpu}": {json.dumps(suggested)}'
    )


# --------------------------------------------------------------------------- #
# DISCRETE path — pre-merge gate, zero threshold (notes §11.5 path 1)
# --------------------------------------------------------------------------- #


@_GPU
@pytest.mark.discrete
def test_attn_ctx_dispatch():
    """Context FMHA takes the expected path: use_paged_context_fmha is a config-deterministic discrete signal."""
    sig = collect_signals(_CTX, warmup=10, iters=10, lock_clock=False)
    observed = sig.use_paged_context_fmha
    golden = _bootstrap_or_skip(_CTX.case_id, _gpu_key(), observed)
    _record(sig, golden, verdict="checked")
    assert observed == golden, (
        f"Cat3 dispatch regression: use_paged_context_fmha "
        f"observed={observed} golden={golden} on {sig.arch}"
    )

    # no-JIT assert turns on once the dev-exposed counter exists (HOOK)
    if sig.nvrtc_compile_count is not None:
        assert sig.nvrtc_compile_count == 0, (
            f"Cat2 JIT recompile: nvrtc_compile_count={sig.nvrtc_compile_count}"
        )


@_GPU
@pytest.mark.discrete
def test_attn_decode_dispatch():
    """Decode-path discrete structure: launch_count (when CUPTI is available)."""
    sig = collect_signals(_DEC, warmup=10, iters=10, lock_clock=False)
    if sig.launch_count is None:
        pytest.skip(
            "launch_count unobserved (cupti package missing) — install cupti to enable this assert"
        )
    golden = _bootstrap_or_skip(_DEC.case_id, _gpu_key(), sig.launch_count)
    _record(sig, golden, verdict="checked")
    assert sig.launch_count == golden, (
        f"Cat2/4 decode-path regression: launch_count "
        f"observed={sig.launch_count} golden={golden} on {sig.arch}"
    )


# --------------------------------------------------------------------------- #
# CONTINUOUS path — post-merge detector, runs INDEPENDENTLY (notes §11.5 path 2)
# This is the blind-spot path: catches kernel-internal slowdown that leaves the
# discrete structure unchanged. Does NOT wait for a discrete failure.
# --------------------------------------------------------------------------- #


def _run_continuous(case, what: str, repeats: int = None):
    """Shared continuous-path body: measure gpu_time, gate vs threshold.

    Repeats the measurement across ``repeats`` fresh subprocesses (default
    ``_CONT_REPEATS``) and gates on the median-of-medians — stable against a lone
    hot/cold run (see _CONT_REPEATS). With repeats>1 the reported cv is the TRUE
    cross-run cv; with repeats==1 it is the single-run in-call cv (unchanged
    behavior). Gates against a variance-based threshold. Bootstrap-skips
    (printing a paste-ready entry) when no golden exists for this (case, arch).
    """
    repeats = _CONT_REPEATS if repeats is None else max(1, repeats)
    runs = [collect_signals(case, warmup=10, iters=50, lock_clock=True) for _ in range(repeats)]
    per_run = [r.gpu_time_median_ms for r in runs]
    observed = statistics.median(per_run)
    # Representative sig for the CSV row = the run whose median is closest to the
    # aggregate (so raw_times/p99 recorded belong to a real, central run).
    sig = min(runs, key=lambda r: abs((r.gpu_time_median_ms or 0.0) - observed))

    if len(per_run) > 1:
        # True cross-session noise across the R independent runs.
        cv = statistics.pstdev(per_run) / observed if observed else 0.0
        n = len(per_run)
    else:
        cv = _observed_cv(sig.raw_times_ms, observed)
        n = len(sig.raw_times_ms)

    golden_entry = _golden_for(case.case_id, _gpu_key())
    if golden_entry is None:
        _record(sig, None, verdict="bootstrap", observed_cv=round(cv, 4))
        _bootstrap_continuous_msg(case.case_id, _gpu_key(), observed, cv, n)

    baseline, margin, threshold = _continuous_gate(golden_entry)
    verdict = "pass" if observed <= threshold else "REGRESSION"
    _record(
        sig, baseline, verdict=verdict, observed_cv=round(cv, 4), threshold_ms=round(threshold, 4)
    )
    assert observed <= threshold, (
        f"{what} gpu_time regression on {sig.arch}: observed={observed:.4f}ms > "
        f"threshold={threshold:.4f}ms (baseline={baseline:.4f}ms × {1 + margin:.4f}; "
        f"this run cv={cv:.4f})"
    )


@_GPU
@pytest.mark.continuous
def test_attn_decode_xqa_time():
    """Decode XQA internal slowdown.

    Caught even when every discrete signal is green (the green-discrete +
    red-continuous blind spot).
    """
    _run_continuous(_DEC_TIME, "decode XQA")


@_GPU
@pytest.mark.continuous
def test_attn_ctx_fmha_time():
    """Context FMHA gpu_time -- backstops the context discrete-layer blind spot."""
    _run_continuous(_CTX_TIME, "context FMHA")


# --------------------------------------------------------------------------- #
# Larger, model-representative GQA cases (§ design: cover real attention shapes
# from the QA perf-core list, not a toy config). The attention kernel depends
# only on the HEAD SHAPE (num_heads, num_kv_heads, head_dim) + dtype + seq_len +
# batch — so models that share a shape share a case:
#   q8b  = Qwen3-8B                         -> 32 / 8 / 128
#   l70b = Qwen3-32B / Llama-3.3-70B /
#          Llama-3.3-Nemotron-Super-49B     -> 64 / 8 / 128
# seq points map from perf ISL/OSL: prefill seq_len=2048 (long prompt), decode =
# 1 new token over num_cached_tokens=1024 at batch 256 (throughput concurrency).
# MLA (DeepSeek) and SWA+sink (gpt_oss) are separate attention families needing
# harness extensions — added in follow-ups.
# --------------------------------------------------------------------------- #

# (label, num_heads, num_kv_heads, head_dim)
_GQA_SHAPES = [("q8b", 32, 8, 128), ("l70b", 64, 8, 128)]

# Separate case_ids for the discrete (scalar golden) vs continuous (dict golden)
# variants of the same physical shape — mirrors the toy attn_ctx_dispatch /
# attn_ctx_fmha_time split so golden[case_id] is unambiguous.
_CTX_KW = dict(phase="context", batch_size=1, seq_len=2048, num_cached_tokens=0)
_DEC_KW = dict(phase="generation", batch_size=256, seq_len=1, num_cached_tokens=1024)


def _gqa(label, nh, nkv, hd, suffix, kw):
    return AttnCase(
        case_id=f"attn_{label}_{suffix}", num_heads=nh, num_kv_heads=nkv, head_dim=hd, **kw
    )


_GQA_CTX_DISP = [_gqa(*s, "ctx_dispatch", _CTX_KW) for s in _GQA_SHAPES]
_GQA_CTX_TIME = [_gqa(*s, "ctx_fmha_time", _CTX_KW) for s in _GQA_SHAPES]
_GQA_DEC_DISP = [_gqa(*s, "decode_dispatch", _DEC_KW) for s in _GQA_SHAPES]
_GQA_DEC_TIME = [_gqa(*s, "decode_xqa_time", _DEC_KW) for s in _GQA_SHAPES]
_ID = lambda c: c.case_id  # noqa: E731


@_GPU
@pytest.mark.discrete
@pytest.mark.parametrize("case", _GQA_CTX_DISP, ids=_ID)
def test_gqa_ctx_dispatch(case):
    """Context dispatch flag for a model-representative GQA prefill shape."""
    sig = collect_signals(case, warmup=10, iters=10, lock_clock=False)
    observed = sig.use_paged_context_fmha
    golden = _bootstrap_or_skip(case.case_id, _gpu_key(), observed)
    _record(sig, golden, verdict="checked")
    assert observed == golden, (
        f"GQA ctx dispatch regression ({case.case_id}): use_paged_context_fmha "
        f"observed={observed} golden={golden} on {sig.arch}"
    )


@_GPU
@pytest.mark.discrete
@pytest.mark.parametrize("case", _GQA_DEC_DISP, ids=_ID)
def test_gqa_decode_dispatch(case):
    """Decode launch_count for a model-representative GQA decode shape."""
    sig = collect_signals(case, warmup=10, iters=10, lock_clock=False)
    if sig.launch_count is None:
        pytest.skip("launch_count unobserved (cupti package missing)")
    golden = _bootstrap_or_skip(case.case_id, _gpu_key(), sig.launch_count)
    _record(sig, golden, verdict="checked")
    assert sig.launch_count == golden, (
        f"GQA decode dispatch regression ({case.case_id}): launch_count "
        f"observed={sig.launch_count} golden={golden} on {sig.arch}"
    )


@_GPU
@pytest.mark.continuous
@pytest.mark.parametrize("case", _GQA_CTX_TIME, ids=_ID)
def test_gqa_ctx_time(case):
    """Context FMHA gpu_time for a model-representative GQA prefill shape."""
    _run_continuous(case, f"GQA ctx FMHA ({case.case_id})")


@_GPU
@pytest.mark.continuous
@pytest.mark.parametrize("case", _GQA_DEC_TIME, ids=_ID)
def test_gqa_decode_time(case):
    """Decode XQA gpu_time for a model-representative GQA decode shape."""
    _run_continuous(case, f"GQA decode XQA ({case.case_id})")


# --------------------------------------------------------------------------- #
# MLA (DeepSeek-V3/R1 latent attention) — decode. The real +48%/+91% DeepSeek-R1
# regressions live on this path (MLA decode), NOT the GQA path above. is_mla
# routes the harness to the latent-KV two-call forward (mla_rope_generation ->
# forward). DeepSeek dims: 128 heads, q_lora 1536, kv_lora 512, qk_nope 128,
# qk_rope 64, v 128 (harness defaults). use_paged_context_fmha is forced False
# for MLA (not a tripwire) -> the discrete signal here is launch_count.
# --------------------------------------------------------------------------- #

_MLA_DEC_KW = dict(
    phase="generation",
    batch_size=128,
    seq_len=1,
    num_cached_tokens=2048,
    is_mla=True,
    num_heads=128,
)
_MLA_DEC_DISP = AttnCase(case_id="attn_mla_decode_dispatch", **_MLA_DEC_KW)
_MLA_DEC_TIME = AttnCase(case_id="attn_mla_decode_time", **_MLA_DEC_KW)


@_GPU
@pytest.mark.discrete
def test_mla_decode_dispatch():
    """Decode launch_count for DeepSeek-style MLA (latent attention)."""
    sig = collect_signals(_MLA_DEC_DISP, warmup=10, iters=10, lock_clock=False)
    if sig.launch_count is None:
        pytest.skip("launch_count unobserved (cupti package missing)")
    golden = _bootstrap_or_skip(_MLA_DEC_DISP.case_id, _gpu_key(), sig.launch_count)
    _record(sig, golden, verdict="checked")
    assert sig.launch_count == golden, (
        f"MLA decode dispatch regression: launch_count "
        f"observed={sig.launch_count} golden={golden} on {sig.arch}"
    )


@_GPU
@pytest.mark.continuous
def test_mla_decode_time():
    """Decode gpu_time for DeepSeek-style MLA (latent attention)."""
    _run_continuous(_MLA_DEC_TIME, "MLA decode (latent attention)")


# MLA with FP8 KV cache — the attention-side quant of the "fp4" DeepSeek models
# (fp4 = weight quant, never touches the attention kernel; the MLA decode's quant
# knob is the fp8 KV cache). Exercises the fp8 MLA decode kernel path, the closest
# attention analog to the deepseek-r1-fp4 e2e regression.
_MLA_FP8_KW = dict(_MLA_DEC_KW, kv_cache_fp8=True)
_MLA_FP8_DISP = AttnCase(case_id="attn_mla_fp8_decode_dispatch", **_MLA_FP8_KW)
_MLA_FP8_TIME = AttnCase(case_id="attn_mla_fp8_decode_time", **_MLA_FP8_KW)


@_GPU
@pytest.mark.discrete
def test_mla_fp8_decode_dispatch():
    """Decode launch_count for MLA with fp8 KV cache (fp4-model attention path)."""
    sig = collect_signals(_MLA_FP8_DISP, warmup=10, iters=10, lock_clock=False)
    if sig.launch_count is None:
        pytest.skip("launch_count unobserved (cupti package missing)")
    golden = _bootstrap_or_skip(_MLA_FP8_DISP.case_id, _gpu_key(), sig.launch_count)
    _record(sig, golden, verdict="checked")
    assert sig.launch_count == golden, (
        f"MLA fp8 decode dispatch regression: launch_count "
        f"observed={sig.launch_count} golden={golden} on {sig.arch}"
    )


@_GPU
@pytest.mark.continuous
def test_mla_fp8_decode_time():
    """Decode gpu_time for MLA with fp8 KV cache (fp4-model attention path)."""
    _run_continuous(_MLA_FP8_TIME, "MLA fp8 decode (latent attention)")


# --------------------------------------------------------------------------- #
# DSA (DeepSeek-V3.2 sparse attention) — CONTEXT/prefill. The real +202%
# deepseek-v32-fp4 regression (rc15->rc16, NVBug 6229365) lives on the sparse
# attention path. is_dsa routes the harness to an MLA module whose backend is
# DSATrtllmAttention (MLA + a sparse Indexer that top-k-selects KV); the forward
# is two-stage (indexer -> forward_context_dsa). seq_len (4096) > index_topk
# (2048) so the indexer selects a genuine sparse subset rather than degenerating
# to dense MHA. DeepSeek-V3.2 dims: 128 heads, q_lora 1536, kv_lora 512,
# qk_nope 128, qk_rope 64, v 128, hidden 7168; indexer 64 heads / dim 128.
#
# ARCH: the indexer's DeepGEMM MQA-logits kernels are built for sm90/sm100/sm103
# ONLY (no sm120 path, no CPU fallback) -> these cases skip on sm120 / RTX PRO
# 6000 and any arch without DeepGEMM. They must be blessed + validated on the
# cluster (H100/B200/B300). NOT yet run anywhere -> no goldens -> bootstrap-skip.
# --------------------------------------------------------------------------- #

try:
    from tensorrt_llm._utils import get_sm_version

    _DSA_ARCH_OK = torch.cuda.is_available() and get_sm_version() in (90, 100, 103)
except (ImportError, RuntimeError):
    # ImportError: _utils/get_sm_version unavailable; RuntimeError: CUDA/driver
    # query failed. Either way DSA cases can't run here -> skip, don't crash.
    _DSA_ARCH_OK = False
_DSA = pytest.mark.skipif(
    not _DSA_ARCH_OK, reason="DSA indexer needs DeepGEMM MQA-logits kernels (sm90/sm100/sm103 only)"
)

_DSA_CTX_KW = dict(
    phase="context",
    batch_size=1,
    seq_len=4096,
    num_cached_tokens=0,
    is_dsa=True,
    num_heads=128,
    hidden_size=7168,
    max_position_embeddings=8192,
    index_n_heads=64,
    index_head_dim=128,
    index_topk=2048,
)
_DSA_CTX_DISP = AttnCase(case_id="attn_dsa_ctx_dispatch", **_DSA_CTX_KW)
_DSA_CTX_TIME = AttnCase(case_id="attn_dsa_ctx_fmha_time", **_DSA_CTX_KW)


@_GPU
@_DSA
@pytest.mark.discrete
def test_dsa_ctx_dispatch():
    """Context launch_count for DeepSeek-V3.2 sparse attention (indexer + sparse MLA)."""
    sig = collect_signals(_DSA_CTX_DISP, warmup=10, iters=10, lock_clock=False)
    if sig.launch_count is None:
        pytest.skip("launch_count unobserved (cupti package missing)")
    golden = _bootstrap_or_skip(_DSA_CTX_DISP.case_id, _gpu_key(), sig.launch_count)
    _record(sig, golden, verdict="checked")
    assert sig.launch_count == golden, (
        f"DSA context dispatch regression: launch_count "
        f"observed={sig.launch_count} golden={golden} on {sig.arch}"
    )


@_GPU
@_DSA
@pytest.mark.continuous
def test_dsa_ctx_time():
    """Context gpu_time for DeepSeek-V3.2 sparse attention (indexer + sparse MLA)."""
    _run_continuous(_DSA_CTX_TIME, "DSA context (sparse attention)")
