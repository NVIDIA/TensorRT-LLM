# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The drafter context window's write against the whole write, at the released widths.

With a drafter context window the last stage projects (``fc`` + ``hidden_norm``) only the
carried rows' captures (each request's kept rows and the plan's lead rows), a contiguous
``[carried, 5 * hidden]`` tensor, and runs the fused drafter K/V GEMM over each request's
carried rows; the whole write projects the step's rows of the capture buffer
(``captured_hidden_states[:rows, :width]``) and runs the K/V GEMM over each request's whole
chunk. The GEMMs therefore run at another M on another operand, and the drafter K/V rows a
generation server reads are the whole write's only if every kept row comes out bit for bit
the same. The plan's lead rows make each GEMM run over the whole write's rows or over at least
``window.MIN_GEMM_ROWS`` rows (below that cuBLAS picks split-K kernels for these shapes).

These tests run both writes on random weights of the released drafter's context-write widths
(captures of five 8192-wide target layers, an 8192-wide drafter of five layers with eight
128-wide KV heads, BF16) in a child process with the context server's GEMM environment
(``TRTLLM_LOW_M_GEMM_BACKEND=auto``, PDL on), on SM 10.7:

* ``test_kept_rows_equal_the_whole_write_at_every_kept_count``: the ops of both writes
  (``project_target_hidden``, then ``precompute_context_kv`` or the one-launch store's
  ``context_k`` + RoPE) for every kept count of a sweep, against whole chunks of the prefill
  capture sizes up to 8192 (and a few other sizes), with the whole input read from a capture
  buffer of the served width and from a wider one (a row stride past the read width), plus
  steps of several requests;
* ``test_window_pages_equal_the_whole_write``: the context-server worker's window write
  (``_store_window_context``, one-launch store and built-in calls) against the built-in whole
  write (``DFlashWorker._store_prefill_context``) on its paged arena, requests whose window
  straddles two chunks and a step padded with the rows of a request that keeps none included;
  every kept row of the drafter K/V pages is compared, the rows a generation server's window
  reads among them, and every row below the kept ones must stay unwritten;
* ``test_rows_at_every_gemm_size_from_the_floor_equal_the_whole_gemm``: the projection and the
  drafter K/V (both stores' ops) of the last M rows of a full step, for every M from
  ``window.MIN_GEMM_ROWS`` to the step's rows, against the same rows of the full step's. It
  also records, without judging them, the sizes below the floor whose rows are bitwise.

Every row that is not bit for bit the whole write's fails the test. The failure lists, per kept
count, the largest absolute and ULP difference of the projection, K and V, and where the first
difference appears. No tolerance applies: a generation server cannot tell a window-written row
from a whole-written one.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_PP = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_107__pp"
)
DTYPE = torch.bfloat16
SEED = 1234

# The released drafter's context write: five captured target layers of width 8192 into an
# 8192-wide drafter of five layers with eight KV heads of width 128.
CAPTURE_LAYERS = (1, 16, 30, 44, 59)
TARGET_HIDDEN = 8192
DRAFT_HIDDEN = 8192
DRAFT_LAYERS = 5
DRAFT_HEADS = 64
KV_HEADS = 8
HEAD_DIM = 128
EPS = 1e-6
CAPTURE_WIDTH = TARGET_HIDDEN * len(CAPTURE_LAYERS)

# The window a generation server's drafter reads, the reuse rewind and the KV block size of the
# served context server: a prompt keeps the drafter context of its last KEEP positions.
WINDOW = 2048
REWIND = 512
TOKENS_PER_BLOCK = 32
KEEP = WINDOW + REWIND + TOKENS_PER_BLOCK

MAX_TOKENS = 8192
MAX_POSITIONS = 16384
# The context server's prefill capture sizes, and a few step sizes between them.
CAPTURE_SIZES = (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560,
    2816, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192,
)  # fmt: skip
WHOLE_ROWS = CAPTURE_SIZES + (KEEP, 3000, 5003, 8191)
KEPT_ROWS = (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 300, 384, 511, 512, 513, 700, 1024, 1500, 2048, KEEP,
)  # fmt: skip
# Steps of several first chunks: (chunk rows, kept rows) per request. A request that keeps no
# row may lend the step its last rows (the projection's lead rows).
MIXED_STEPS = (
    ((5000, 2000, 1192), (KEEP, 37, 1192)),
    ((4096, 4096), (1, KEEP)),
    ((KEEP, KEEP, KEEP, 416), (KEEP, 1, 2048, 416)),
    ((8, 8, 8), (1, 8, 3)),
    ((300, 7892), (300, 0)),
    ((100, 50, 7000), (100, 0, 0)),
    ((40, 4000, 4000), (1, 0, 3)),
)
# Sizes below the floor recorded (not judged) by the size sweep.
BELOW_FLOOR_FIRST = 257
# The context server's GEMM environment, set for the child process only.
CHILD_ENV = {"TRTLLM_LOW_M_GEMM_BACKEND": "auto", "TRTLLM_ENABLE_PDL": "1"}


def _modules():
    from importlib import import_module

    return (
        import_module(f"{_PP}.kernels.dflash_context.worker"),
        import_module(f"{_PP}.kernels.dflash_context.window"),
        import_module(f"{_PP}.kernels.dflash_context.store"),
    )


def _require_sm107():
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the served context server's GEMMs run on SM 10.7")


# --------------------------------------------------------------------------
# The drafter at the released context-write widths
# --------------------------------------------------------------------------
def _tiny_drafter():
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    path = Path(__file__).parents[4] / "speculative" / "hw_agnostic" / "test_dflash2_semantics.py"
    spec = importlib.util.spec_from_file_location("_dflash2_semantics", path)
    sem = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sem)
    config = ModelConfig(pretrained_config=sem._tiny_config(True), attn_backend="TRTLLM")
    drafter = DFlashForCausalLM(config, dflash_attention_backend="TRTLLM").to("cuda")
    drafter.load_weights(dict(sem._tiny_weights(True)))
    return drafter


def _rope_cache(positions: int, head_dim: int, theta: float = 1e6) -> torch.Tensor:
    """A flashinfer cos/sin cache ``[positions, head_dim]`` fp32: the cosines, then the sines."""
    inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    angles = torch.arange(positions, dtype=torch.float64)[:, None] * inv_freq[None, :]
    return torch.cat([angles.cos(), angles.sin()], dim=-1).float().cuda().contiguous()


@torch.no_grad()
def _released_width_drafter():
    """The speculative tests' tiny DFlash 2 drafter with random context-write weights of the
    released drafter's widths (``fc``, ``hidden_norm``, the fused K/V weight, the stacked
    k_norm) and a RoPE table past every position the tests write."""
    drafter = _tiny_drafter()
    gen = torch.Generator(device="cuda").manual_seed(SEED)

    def randn(*shape, scale=1.0, shift=0.0):
        return (torch.randn(*shape, device="cuda", generator=gen) * scale + shift).to(DTYPE)

    drafter.fc = nn.Linear(CAPTURE_WIDTH, DRAFT_HIDDEN, bias=False, device="cuda", dtype=DTYPE)
    drafter.fc.weight.copy_(randn(DRAFT_HIDDEN, CAPTURE_WIDTH, scale=CAPTURE_WIDTH**-0.5))
    drafter.hidden_norm = nn.RMSNorm(DRAFT_HIDDEN, eps=EPS, device="cuda", dtype=DTYPE)
    drafter.hidden_norm.weight.copy_(randn(DRAFT_HIDDEN, scale=0.1, shift=1.0))
    kv_rows = DRAFT_LAYERS * 2 * KV_HEADS * HEAD_DIM
    drafter._fused_kv_weight = randn(kv_rows, DRAFT_HIDDEN, scale=DRAFT_HIDDEN**-0.5)
    drafter._fused_kv_bias = None
    drafter._k_norm_stacked = randn(DRAFT_LAYERS, HEAD_DIM, scale=0.1, shift=1.0)
    drafter._k_norm_eps = EPS
    drafter._input_ln_eps = None
    drafter._num_attn_layers = DRAFT_LAYERS
    drafter._num_heads = DRAFT_HEADS
    drafter._num_kv_heads = KV_HEADS
    drafter._head_dim = HEAD_DIM
    drafter._cos_sin_cache_fp32 = _rope_cache(MAX_POSITIONS, HEAD_DIM)
    drafter._rope_dummy_q = None
    drafter._is_neox = True
    drafter.config.max_position_embeddings = MAX_POSITIONS
    return drafter


def _gemm_environment(drafter) -> dict:
    """The low-M GEMM dispatch as the model loader leaves it under this process's env."""
    from tensorrt_llm._torch.modules import low_m_gemm

    if low_m_gemm.LOW_M_GEMM_ACTIVE:
        low_m_gemm.prepare_low_m_gemm(drafter)
    return {
        "low_m_gemm_active": bool(low_m_gemm.LOW_M_GEMM_ACTIVE),
        "fc_class": type(drafter.fc).__name__,
        "fc_low_m_dispatcher": hasattr(drafter.fc, "_low_m_gemm_dispatcher"),
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "blas": torch.backends.cuda.preferred_blas_library().name,
        "bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
    }


# --------------------------------------------------------------------------
# Bitwise comparison, with the size of any difference
# --------------------------------------------------------------------------
def _ordered(x: torch.Tensor) -> torch.Tensor:
    """BF16 bit patterns as int32 in value order (adjacent values one apart, both zeros 0)."""
    bits = x.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    magnitude = bits & 0x7FFF
    return torch.where(bits >= 0x8000, -magnitude, magnitude)


def _diff(got: torch.Tensor, ref: torch.Tensor):
    """``(rows that differ in any bit, max abs difference, max ULP difference)`` of two BF16
    tensors of one shape whose first dimension is the row."""
    assert got.shape == ref.shape and got.dtype == ref.dtype == DTYPE, (got.shape, ref.shape)
    rows = got.shape[0]
    if rows == 0:
        return torch.zeros(0, dtype=torch.bool, device=got.device), 0.0, 0
    differs = got.contiguous().view(torch.int16) != ref.contiguous().view(torch.int16)
    bad_rows = differs.reshape(rows, -1).any(dim=1)
    if not bool(bad_rows.any()):
        return bad_rows, 0.0, 0
    delta = (got.float() - ref.float()).abs()
    max_abs = float(torch.nan_to_num(delta, nan=float("inf")).max())
    max_ulp = int((_ordered(got) - _ordered(ref)).abs().max())
    return bad_rows, max_abs, max_ulp


def _compare(label: dict, parts: dict, window_rows: torch.Tensor) -> dict:
    """One report row: each part's ``(got, ref)`` compared; ``window_rows`` marks the kept rows
    a generation server's window reads."""
    row = dict(label)
    bitwise = True
    for name, (got, ref) in parts.items():
        bad_rows, max_abs, max_ulp = _diff(got, ref)
        rows_bad = int(bad_rows.sum())
        row[f"{name}_rows_differ"] = rows_bad
        row[f"{name}_window_rows_differ"] = int((bad_rows & window_rows).sum())
        row[f"{name}_max_abs"] = max_abs
        row[f"{name}_max_ulp"] = max_ulp
        bitwise = bitwise and rows_bad == 0
    row["bitwise"] = bitwise
    return row


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


# --------------------------------------------------------------------------
# The ops of the two writes
# --------------------------------------------------------------------------
def _kept_kv(drafter, store_mod, projected, first: int, store: str, lead: int = 0):
    """The drafter K/V of the last ``rows - lead`` of ``projected`` rows, which sit at positions
    ``first..``, computed over all of them as the window write does: by the built-in calls
    (``precompute_context_kv``, then the stored rows) or as the one-launch store computes them
    (``store.run_k`` + RoPE of the stored rows)."""
    rows = projected.shape[0] - lead
    chunk = projected.detach().to(DTYPE)
    if store == "built-in":
        positions = torch.arange(first - lead, first + rows, dtype=torch.long, device="cuda")
        k, v = drafter.precompute_context_kv(chunk, positions)
        return k[lead:], v[lead:]
    run = store_mod.Run(offset=0, rows=rows, table_row=0, first=first, lead=lead)
    k, kv_flat = store_mod.run_k(drafter, chunk, run, DTYPE)
    positions = store_mod.rope_positions([run], DRAFT_LAYERS).to("cuda")
    store_mod.rope_inplace(drafter, k, positions)
    v = kv_flat.view(rows, DRAFT_LAYERS, 2, KV_HEADS, HEAD_DIM)[:, :, 1]
    return k, v


def _sweep_cases():
    """``(kind, chunk rows per request, kept rows per request)`` of every step of the sweep."""
    for whole in WHOLE_ROWS:
        for kept in KEPT_ROWS:
            if kept <= whole:
                yield "single", (whole,), (kept,)
    for lengths, kept in MIXED_STEPS:
        yield "mixed", lengths, kept


@torch.no_grad()
def _sweep_ops(report_path: str) -> None:
    """Child: the kept rows of every sweep step against the whole write's, as JSON rows."""
    _, window_mod, store_mod = _modules()
    drafter = _released_width_drafter()
    environment = _gemm_environment(drafter)
    gen = torch.Generator(device="cuda").manual_seed(SEED + 1)
    served = torch.randn(MAX_TOKENS, CAPTURE_WIDTH, device="cuda", generator=gen).to(DTYPE)
    wider = torch.full((MAX_TOKENS, CAPTURE_WIDTH + 128), float("nan"), device="cuda", dtype=DTYPE)
    wider[:, :CAPTURE_WIDTH] = served
    # The whole write reads ``captured_hidden_states[:rows, :width]``: a served buffer is
    # exactly ``width`` wide; a wider one gives the same values at a longer row stride.
    buffers = {"capture-buffer": served, "wider-buffer": wider}
    rows = []

    def whole_write(buffer, lengths):
        """The whole write of a step of first chunks: the projection, each request's K/V."""
        total = sum(lengths)
        projected = drafter.project_target_hidden(buffer[:total, :CAPTURE_WIDTH])
        kvs, offset = [], 0
        for length in lengths:
            positions = torch.arange(length, dtype=torch.long, device="cuda")
            chunk = projected[offset : offset + length].to(DTYPE)
            kvs.append(drafter.precompute_context_kv(chunk, positions))
            offset += length
        return projected, kvs

    floor = window_mod.MIN_GEMM_ROWS
    for kind, lengths, kept in _sweep_cases():
        # Each request is a first chunk whose prompt keeps its last ``kept`` chunk rows: the
        # chunk ends ``KEEP - kept`` positions before the prompt's end.
        prompts = [length + KEEP - count for length, count in zip(lengths, kept)]
        plan = window_mod.plan_window(lengths, [0] * len(lengths), [p - KEEP for p in prompts])
        assert plan.kept == tuple(kept), (plan.kept, kept)
        # Every GEMM of the window write runs over the whole write's rows or the floor's.
        assert plan.count == sum(lengths) or plan.count >= floor, plan
        for length, count, carried in zip(lengths, plan.kept, plan.carried):
            assert not count or carried == length or carried >= floor, plan
        captures = served[: sum(lengths)].index_select(0, plan.index("cuda"))
        assert captures.is_contiguous()
        projected_kept = drafter.project_target_hidden(captures)
        for form, buffer in buffers.items():
            projected, kvs = whole_write(buffer, lengths)
            for store in ("built-in", "one-launch"):
                begin = 0
                for i, (length, skip, lead, offset, prompt) in enumerate(
                    zip(lengths, plan.skips, plan.leads, plan.offsets(), prompts)
                ):
                    count = length - skip
                    if count == 0:
                        # A request that keeps no row stores nothing (its lead rows, if any,
                        # serve the projection only).
                        begin += length
                        continue
                    k_ref, v_ref = kvs[i]
                    k_got, v_got = _kept_kv(
                        drafter,
                        store_mod,
                        projected_kept[offset : offset + lead + count],
                        skip,
                        store,
                        lead,
                    )
                    positions = torch.arange(skip, length, device="cuda")
                    label = dict(
                        kind=kind,
                        form=form,
                        store=store,
                        step_rows=list(lengths),
                        step_kept=list(plan.kept),
                        step_leads=list(plan.leads),
                        request=i,
                        chunk_rows=length,
                        kept=count,
                        lead=lead,
                        fc_m_whole=sum(lengths),
                        fc_m_carried=plan.count,
                        kv_m_whole=length,
                        kv_m_carried=lead + count,
                    )
                    parts = {
                        "projection": (
                            projected_kept[offset + lead : offset + lead + count],
                            projected[begin + skip : begin + length],
                        ),
                        "k": (_flat(k_got), _flat(k_ref[skip:])),
                        "v": (_flat(v_got.contiguous()), _flat(v_ref[skip:].contiguous())),
                    }
                    rows.append(_compare(label, parts, positions >= prompt - WINDOW))
                    begin += length
    report = {"environment": environment, "kernels": _gemm_kernels(drafter), "rows": rows}
    Path(report_path).write_text(json.dumps(report))


@torch.no_grad()
def _sweep_sizes(report_path: str) -> None:
    """Child: for every M from ``BELOW_FLOOR_FIRST`` to a full step, the last M
    rows of a full step projected and run through the drafter K/V (both stores' ops) on their
    own, against the same rows of the full step's; one JSON row per M."""
    _, window_mod, store_mod = _modules()
    drafter = _released_width_drafter()
    environment = _gemm_environment(drafter)
    floor = window_mod.MIN_GEMM_ROWS
    gen = torch.Generator(device="cuda").manual_seed(SEED + 3)
    served = torch.randn(MAX_TOKENS, CAPTURE_WIDTH, device="cuda", generator=gen).to(DTYPE)
    projected = drafter.project_target_hidden(served)
    refs = {
        store: _kept_kv(drafter, store_mod, projected, 0, store)
        for store in ("built-in", "one-launch")
    }
    rows = []
    for m in range(BELOW_FLOOR_FIRST, MAX_TOKENS + 1):
        begin = MAX_TOKENS - m
        # A fresh operand, as the window write's carried captures and projection are.
        got = drafter.project_target_hidden(served[begin:].clone())
        parts = {"projection": (got, projected[begin:])}
        for store, (k_ref, v_ref) in refs.items():
            k_got, v_got = _kept_kv(drafter, store_mod, projected[begin:].clone(), begin, store)
            parts[f"k_{store}"] = (_flat(k_got), _flat(k_ref[begin:]))
            parts[f"v_{store}"] = (_flat(v_got.contiguous()), _flat(v_ref[begin:].contiguous()))
        label = dict(kind="size", kept=m, judged=m >= floor, floor=floor)
        rows.append(_compare(label, parts, torch.zeros(m, dtype=torch.bool, device="cuda")))
    Path(report_path).write_text(json.dumps({"environment": environment, "rows": rows}))


#: Profiler event names that are not GEMM kernels.
_NOT_KERNELS = ("aten::", "cuda", "Memcpy", "Memset", "Activity Buffer", "ProfilerStep")


def _gemm_kernels(drafter) -> dict:
    """The kernels the projection and the K/V GEMM launch at each M of the sweep (diagnostic:
    a profiler that does not run leaves a note instead)."""
    from torch.profiler import ProfilerActivity, profile

    _, window_mod, _ = _modules()
    sizes = set(WHOLE_ROWS) | set(KEPT_ROWS) | {320, 384, 448, 576, 640}
    for lengths, kept in MIXED_STEPS:
        prompts = [length + KEEP - count for length, count in zip(lengths, kept)]
        plan = window_mod.plan_window(lengths, [0] * len(lengths), [p - KEEP for p in prompts])
        sizes |= {sum(lengths), sum(kept), plan.count, *plan.carried} - {0}
    out = {}
    try:
        for m in sorted(sizes):
            hidden = torch.ones(m, CAPTURE_WIDTH, device="cuda", dtype=DTYPE)
            projected = torch.ones(m, DRAFT_HIDDEN, device="cuda", dtype=DTYPE)
            names = {}
            for name, operand, weight in (
                ("fc", hidden, drafter.fc.weight),
                ("kv", projected, drafter._fused_kv_weight),
            ):
                torch.nn.functional.linear(operand, weight)
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    torch.nn.functional.linear(operand, weight)
                    torch.cuda.synchronize()
                names[name] = sorted(
                    {e.key for e in prof.key_averages() if not e.key.startswith(_NOT_KERNELS)}
                )
            out[str(m)] = names
    except RuntimeError as error:
        out["unavailable"] = str(error)
    return out


# --------------------------------------------------------------------------
# The worker's window write against its whole write, on the paged arena
# --------------------------------------------------------------------------
#: (name, steps): each step lists (request id, chunk start, chunk rows, prompt rows).
WORKER_CASES = (
    ("final-chunk", [[(7, 0, 8192, 8192)]]),
    ("one-row-kept", [[(7, 0, 8192, 8192 + KEEP - 1)]]),
    ("two-chunks", [[(7, 0, 8192, 10000)], [(7, 8192, 1808, 10000)]]),
    ("mixed-step", [[(7, 0, 5000, 5000), (8, 0, 2000, 2000 + KEEP - 37), (9, 0, 1192, 1192)]]),
    ("short-chunk", [[(7, 0, 2304, 2304 + KEEP - 2048)]]),
    # The window starts 292 rows before the first chunk's end: 220 lead rows ahead of them.
    ("straddle-short-keep", [[(7, 0, 8192, 8192 + 2300)], [(7, 8192, 2300, 8192 + 2300)]]),
    # A whole short prompt beside a chunk that keeps nothing: the step's projection runs over
    # the floor's rows, the other request's last rows among them, which stay unwritten.
    ("padded-step", [[(7, 0, 300, 300), (8, 0, 7892, 7892 + KEEP + 100)]]),
)


def _metadata():
    from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata
    from tensorrt_llm.llmapi import DFlashDecodingConfig

    config = DFlashDecodingConfig(max_draft_len=7)
    return DFlashSpecMetadata(
        max_draft_len=7,
        max_total_draft_tokens=7,
        spec_dec_mode=config.spec_dec_mode,
        max_num_requests=4,
        layers_to_capture=list(CAPTURE_LAYERS),
        hidden_size=TARGET_HIDDEN,
        max_num_tokens=MAX_TOKENS,
        dtype=DTYPE,
    )


def _attn(lengths):
    seq_lens = torch.tensor(lengths, dtype=torch.int32)
    return types.SimpleNamespace(
        num_seqs=len(lengths),
        num_contexts=len(lengths),
        num_ctx_tokens=sum(lengths),
        _seq_lens=seq_lens,
        seq_lens=seq_lens,
        kv_cache_manager=None,
        max_seq_len=MAX_POSITIONS,
    )


def _worker(worker_mod, drafter, meta, lengths):
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker
    from tensorrt_llm.llmapi import DFlashDecodingConfig
    from tensorrt_llm.mapping import Mapping

    worker = DFlashWorker(
        DFlashDecodingConfig(max_draft_len=7, attention_backend="TRTLLM"), Mapping()
    )
    worker_mod.install(worker)
    worker._lazy_init_ctx_buffers(drafter, meta, _attn(lengths), None)
    assert worker._ctx_paged
    for pool in worker._ctx_kv_buf:
        pool.fill_(float("nan"))
    return worker


def _page_rows(worker, request, positions, layer, kv):
    slot = worker._req_to_slot[request]
    page = worker._ctx_page_size
    pages = worker._ctx_page_table[slot, positions // page].long()
    return worker._ctx_kv_buf[layer][pages, kv, :, positions % page, :]


def _step_rows(hidden, step):
    return torch.cat([hidden[req][start : start + length] for req, start, length, _ in step])


@torch.no_grad()
def _worker_cases(report_path: str) -> None:
    """Child: each worker case's kept page rows against the built-in whole write's."""
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    worker_mod, window_mod, store_mod = _modules()
    drafter = _released_width_drafter()
    environment = _gemm_environment(drafter)
    meta = _metadata()
    rows = []
    for name, steps in WORKER_CASES:
        prompts = {req: prompt for step in steps for req, _, _, prompt in step}
        written = {req: 0 for req in prompts}
        for step in steps:
            for req, _, length, _ in step:
                written[req] += length
        gen = torch.Generator(device="cuda").manual_seed(SEED + 2)
        hidden = {
            req: torch.randn(written[req], CAPTURE_WIDTH, device="cuda", generator=gen).to(DTYPE)
            for req in sorted(prompts)
        }
        first_lengths = [length for _, _, length, _ in steps[0]]
        whole = _worker(worker_mod, drafter, meta, first_lengths)
        for step in steps:
            lengths = [length for _, _, length, _ in step]
            positions = [start + j for _, start, length, _ in step for j in range(length)]
            meta.captured_hidden_states[: sum(lengths)].copy_(_step_rows(hidden, step))
            meta.request_ids = [req for req, _, _, _ in step]
            meta.host_position_ids = torch.tensor(positions, dtype=torch.int32)
            DFlashWorker._store_prefill_context(
                whole,
                drafter,
                meta,
                _attn(lengths),
                torch.tensor(positions, dtype=torch.long, device="cuda"),
                sum(lengths),
            )
        for store in ("one-launch", "built-in"):
            kept = _worker(worker_mod, drafter, meta, first_lengths)
            if store == "one-launch":
                reason = store_mod.refusal(kept, drafter)
                assert reason is None, reason
            fc_m = []
            for step in steps:
                lengths = [length for _, _, length, _ in step]
                starts = [start for _, start, _, _ in step]
                keep_from = [max(0, prompt - KEEP) for _, _, _, prompt in step]
                plan = window_mod.plan_window(lengths, starts, keep_from)
                captures = None
                if plan.count:
                    captures = _step_rows(hidden, step).index_select(0, plan.index("cuda"))
                fc_m.append([sum(lengths), plan.count])
                floor = window_mod.MIN_GEMM_ROWS
                assert plan.count in (0, sum(lengths)) or plan.count >= floor, plan
                kept._store_window_context(
                    drafter,
                    types.SimpleNamespace(request_ids=[req for req, _, _, _ in step]),
                    _attn(lengths),
                    plan,
                    captures,
                    one_launch=store == "one-launch",
                )
            torch.cuda.synchronize()
            for req, prompt in sorted(prompts.items()):
                end = written[req]
                first = max(0, prompt - KEEP)
                assert kept._req_ctx_pos[req] == whole._req_ctx_pos[req] == end
                assert kept._ctx_len_host[kept._req_to_slot[req]] == end
                assert whole._ctx_len_host[whole._req_to_slot[req]] == end
                if first:
                    below = torch.arange(0, min(first, end), device="cuda")
                    for layer in range(DRAFT_LAYERS):
                        for kv in (0, 1):
                            dropped = _page_rows(kept, req, below, layer, kv)
                            assert torch.isnan(dropped.float()).all(), (name, req, layer, kv)
                if first >= end:
                    # The request keeps none of the rows written so far: every one of them was
                    # checked unwritten above, and there is no kept row to compare.
                    continue
                at = torch.arange(first, end, device="cuda")
                parts = {}
                for layer in range(DRAFT_LAYERS):
                    for kv, half in ((0, "k"), (1, "v")):
                        ref = _flat(_page_rows(whole, req, at, layer, kv))
                        assert torch.isfinite(ref.float()).all(), (name, req, layer, half)
                        parts[f"{half}{layer}"] = (_flat(_page_rows(kept, req, at, layer, kv)), ref)
                label = dict(
                    kind="worker",
                    case=name,
                    store=store,
                    request=req,
                    prompt=prompt,
                    kept=end - first,
                    fc_m_whole_and_carried=fc_m,
                )
                rows.append(_compare(label, parts, at >= prompt - WINDOW))
            del kept
        del whole
        torch.cuda.empty_cache()
    Path(report_path).write_text(json.dumps({"environment": environment, "rows": rows}))


# --------------------------------------------------------------------------
# The tests: a child with the context server's GEMM environment, then its rows judged
# --------------------------------------------------------------------------
def _run_child(mode: str, report: Path) -> dict:
    env = dict(os.environ)
    env.update(CHILD_ENV)
    result = subprocess.run(
        [sys.executable, __file__, mode, str(report)],
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-6000:]
    data = json.loads(report.read_text())
    environment = data["environment"]
    assert environment["low_m_gemm_active"], environment
    assert tuple(environment["capability"]) == (10, 7), environment
    return data


def _by_kept_count(rows) -> str:
    """Per kept count: cases compared, cases not bitwise, and the largest differences."""
    table = {}
    for row in rows:
        entry = table.setdefault(row["kept"], {"cases": 0, "failed": 0, "worst": {}})
        entry["cases"] += 1
        entry["failed"] += not row["bitwise"]
        for key, value in row.items():
            if key.endswith(("_max_abs", "_max_ulp")):
                entry["worst"][key] = max(entry["worst"].get(key, 0), value)
    lines = ["kept | cases | not bitwise | largest nonzero differences"]
    for kept in sorted(table):
        entry = table[kept]
        worst = {key: value for key, value in entry["worst"].items() if value}
        lines.append(f"{kept} | {entry['cases']} | {entry['failed']} | {worst}")
    return "\n".join(lines)


def _first_difference(row) -> str:
    """The first part (projection, then K, then V, then each page) that differs."""
    for key, value in row.items():
        if key.endswith("_rows_differ") and not key.endswith("_window_rows_differ") and value:
            return key[: -len("_rows_differ")]
    return ""


def _judge(data: dict, report: Path) -> None:
    rows = data["rows"]
    assert rows
    summary = _by_kept_count(rows)
    print(f"environment: {data['environment']}")
    print(summary)
    print(f"full report: {report}")
    failed = [row for row in rows if not row["bitwise"]]
    if failed:
        examples = "\n".join(
            f"{ {k: v for k, v in row.items() if v or not k.endswith('_differ')} } "
            f"-> first difference in {_first_difference(row)}"
            for row in failed[:12]
        )
        pytest.fail(
            f"{len(failed)} of {len(rows)} kept-row cases are not bit for bit the whole write's "
            f"(no tolerance applies).\n{summary}\nFirst failing cases:\n{examples}\n"
            f"Full report: {report}",
            pytrace=False,
        )


def test_kept_rows_equal_the_whole_write_at_every_kept_count(tmp_path):
    _require_sm107()
    report = tmp_path / "window_ops.json"
    data = _run_child("ops", report)
    rows = data["rows"]
    # Every kept count ran, against every whole size that holds it, in both input forms and
    # with both stores.
    assert {row["kept"] for row in rows if row["kind"] == "single"} == set(KEPT_ROWS)
    singles = sum(1 for whole in WHOLE_ROWS for kept in KEPT_ROWS if kept <= whole)
    assert sum(row["kind"] == "single" for row in rows) == singles * 2 * 2
    _judge(data, report)


def _spans(values) -> str:
    """Sorted integers as ``a-b`` spans."""
    out, start, prev = [], None, None
    for value in sorted(values):
        if start is None:
            start = prev = value
        elif value == prev + 1:
            prev = value
        else:
            out.append(f"{start}-{prev}" if prev != start else f"{start}")
            start = prev = value
    if start is not None:
        out.append(f"{start}-{prev}" if prev != start else f"{start}")
    return ", ".join(out)


def test_rows_at_every_gemm_size_from_the_floor_equal_the_whole_gemm(tmp_path):
    _require_sm107()
    report = tmp_path / "window_sizes.json"
    data = _run_child("sizes", report)
    rows = data["rows"]
    judged = [row for row in rows if row["judged"]]
    floor = rows[0]["floor"]
    assert [row["kept"] for row in judged] == list(range(floor, MAX_TOKENS + 1))
    below = [row["kept"] for row in rows if not row["judged"] and row["bitwise"]]
    print(f"environment: {data['environment']}")
    print(f"sizes below the floor {floor} whose rows are bitwise (not judged): {_spans(below)}")
    failed = [row for row in judged if not row["bitwise"]]
    if failed:
        examples = "\n".join(
            f"{row} -> first difference in {_first_difference(row)}" for row in failed[:8]
        )
        pytest.fail(
            f"{len(failed)} of {len(judged)} sizes from the floor on are not bit for bit the "
            f"full step's rows (no tolerance applies): {_spans(row['kept'] for row in failed)}."
            f"\n{examples}\nFull report: {report}",
            pytrace=False,
        )


def test_window_pages_equal_the_whole_write(tmp_path):
    from tensorrt_llm._torch.speculative import dflash_attention

    _require_sm107()
    if (reason := dflash_attention.dflash_trtllm_gen_unavailability_reason()) is not None:
        pytest.skip(reason)
    report = tmp_path / "window_pages.json"
    data = _run_child("worker", report)
    rows = data["rows"]
    assert {row["case"] for row in rows} == {name for name, _ in WORKER_CASES}
    assert {row["store"] for row in rows} == {"one-launch", "built-in"}
    _judge(data, report)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "ops":
        _sweep_ops(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "worker":
        _worker_cases(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "sizes":
        _sweep_sizes(sys.argv[2])
    else:
        raise SystemExit(f"usage: {sys.argv[0]} ops|worker|sizes REPORT_JSON")
