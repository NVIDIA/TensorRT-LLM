#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""B2 CUDA-graph decode localizer -- production full-model TP=4 per-layer replay.

Motivation (iter77). The enabled (cuda_graph=on) served path emits stuck-token
garbage ('!!!!!' -- token 0 repeated) while the baseline (cuda_graph=off) path is
correct (served GSM8K 0.91/0.93). crit6 showed prefill logits are IDENTICAL
cg-off-vs-cg-on (pos0 10/10) while the FIRST decode step already diverges (pos1
7/10). Every isolated / reduced-model / TP=1 localizer is graph-clean, so B2 lives
in the full-model TP=4 PRODUCTION stack (TP collectives and/or the production
CUDAGraphRunner). TP=1/TP=2 cannot hold this ~403GB checkpoint (iter76), so B2 can
only be observed at TP=4.

This driver reproduces B2 IN-PROCESS (no server needed) and drives the model-side
capture-safe per-layer fingerprint (env INKLING_FP -> InklingModel._ink_fp): a
persistent GPU buffer written by a device->device copy_ recorded INTO the decode
graph (so it survives capture/replay, unlike the .cpu() dump_sink). It runs a
single fixed prompt (batch=1, the exact B2 smoke condition) free-running for
INKLING_FP_STEPS tokens; the model dumps, per rank per decode step, the residual
after every decoder layer + the final norm.

Because prefill is identical cg-off-vs-cg-on, DECODE STEP 0 receives the SAME input
token in both configs, so its per-layer fingerprints are directly comparable:
inkling_fp_analyze.py loads the cg0 and cg1 dumps and reports (a) the first layer
where cg0 and cg1 diverge on the same rank (B2's origin layer) and (b) cross-rank
residual consistency within each config (the all-reduced residual MUST be identical
across TP ranks; divergence there pins a TP-collective-under-graph bug).

Run: trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_fp_localize_test.py
Env: INKLING_CHECKPOINT, INKLING_SGLANG_REF (crit6 capture for input_ids),
     INKLING_TP(=4), INKLING_CUDA_GRAPH(0/1), INKLING_OVERLAP, INKLING_FP(dump base),
     INKLING_FP_STEPS(default 8), INKLING_MOE_BACKEND(default TRTLLM).
"""

import json
import os
import sys

CKPT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/"
    "users/kleinc/hf_data/Inkling-NVFP4-full")
REF = os.environ.get(
    "INKLING_SGLANG_REF",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/"
    "codes/agent-flow/workspace/inkling-bringup/results/sglang_ref_logit_replay.json")

CUDA_GRAPH = os.environ.get("INKLING_CUDA_GRAPH", "0") == "1"
OVERLAP = os.environ.get("INKLING_OVERLAP", "1" if CUDA_GRAPH else "0") == "1"
TP = int(os.environ.get("INKLING_TP", "4"))
STEPS = int(os.environ.get("INKLING_FP_STEPS", "8"))
MOE_BACKEND = os.environ.get("INKLING_MOE_BACKEND", "TRTLLM")
# feedback #3/#4 determinism prerequisite: the per-step/per-layer localization is
# only valid under batch=1 + autotuner-off + TP all-reduce autotune off. The B2
# op/no-probe repro was run under the NON-deterministic default (max_batch_size=8,
# autotuner ON), so B2 reproduced only run-to-run (the Heisenbug / NO_LEAD /
# collapse=True-then-False across identical-config jobs 5562031 vs 5563553).
# INKLING_FP_DETERMINISTIC=1 enforces the mandated config so cg1/ov1 B2 either
# reproduces EVERY run (deterministic -> localizable) or NEVER (it WAS the
# batch>1 cross-row-MoE / autotuner non-determinism under cuda-graph). Byte-
# unchanged when unset (max_batch_size=8, enable_autotuner=True == the defaults).
# TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1 must be exported by the launcher (before MPI
# worker spawn), so the driver only reports it here.
DETERMINISTIC = os.environ.get("INKLING_FP_DETERMINISTIC") == "1"
# Lane-1 (feedback #19 compute-sanitizer) footprint lever: initcheck/memcheck add
# device shadow-memory overhead on top of the ~120GB/rank NVFP4 weights, so the
# sanitizer arm can lower the KV pool (default 0.75, byte-unchanged for every other
# run) to leave headroom and avoid an OOM that would mask the uninitialized-read
# report. Only the sanitizer sbatch sets this; production/other arms never do.
KV_FRACTION = float(os.environ.get("INKLING_KV_FRACTION", "0.75"))


def _max_seq_config():
    """(max_seq_len, max_num_tokens) for the LLM build -- the feedback #19 Lane-1
    (compute-sanitizer) WALL-TIME lever. initcheck is ~10-50x slower, and job
    5564727 spent >50min in the runtime's max_seq_len context warmup alone (17min
    at the default 2048) and never reached the decode where B2's uninitialized read
    fires. The B2 repro prompt is only ~5 tokens and B2 fires at kv_pos=5 (decode
    step 0, KV page 0, page_size=32), so a much smaller max_seq_len/max_num_tokens
    leaves the decode path -- and therefore the B2 read -- byte-identical while
    slashing the warmup. Read at call time so ``--det-selftest`` can exercise it.
    Defaults 2048/2048 keep every other arm/run byte-unchanged (the accepted
    baseline stages are not touched)."""
    return (int(os.environ.get("INKLING_FP_MAX_SEQ", "2048")),
            int(os.environ.get("INKLING_FP_MAX_NUM_TOKENS", "2048")))


def _det_llm_config(deterministic: bool):
    """(max_batch_size, enable_autotuner) under the feedback #3/#4 determinism
    gate: deterministic -> batch=1 + autotuner OFF; else the batch=8 + autotuner-ON
    default (byte-unchanged production path). Single source of truth so the LLM
    build and the ``--det-selftest`` cannot drift."""
    return (1, False) if deterministic else (8, True)


def _resolved_llm_config(deterministic: bool):
    """(max_batch_size, enable_autotuner) after applying the optional
    ``INKLING_FP_BS`` / ``INKLING_FP_AUTOTUNE`` overrides on top of the determinism
    gate. This lets the feedback #19 compute-sanitizer (Lane 1) arm select
    ``max_batch_size=8`` -- the proven-collapse config whose CUDA-graph padding/
    bucket capture is part of the B2 surface -- while turning the autotuner OFF so
    initcheck does not have to instrument the whole autotuning storm at build time.
    Byte-unchanged when neither override is set (production/other arms unaffected),
    so the accepted baseline stages are not touched."""
    bs, at = _det_llm_config(deterministic)
    bs_env = os.environ.get("INKLING_FP_BS")
    if bs_env:
        bs = int(bs_env)
    at_env = os.environ.get("INKLING_FP_AUTOTUNE")
    if at_env is not None and at_env != "":
        at = at_env == "1"
    return bs, at


def _max_consec_repeat(ids):
    best = cur = 0
    prev = None
    for x in ids:
        cur = cur + 1 if x == prev else 1
        prev = x
        best = max(best, cur)
    return best


def _stat_selftest() -> int:
    """Focused unit test for the iter-79 ALLOCATION-FREE ``_ink_fp_stat`` op
    fingerprint (feedback #17). Validates value-correctness of the scratch path
    and that it AGREES with the no-scratch fallback on the finiteness verdict and
    max_abs, on GPU when available (the real device path) else CPU. This is the
    fail-fast in-container guard for the new code path -- the CUDA-graph
    transparency itself is proven end-to-end by the G1 arm reproducing B2."""
    import torch

    from tensorrt_llm._torch.models.modeling_inkling import (  # noqa: PLC0415
        _INK_FP_STAT_W, _ink_fp_stat)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== INKLING_FP_STAT_SELFTEST dev={dev} stat_w={_INK_FP_STAT_W} ===",
          flush=True)
    scratch = torch.zeros(1 << 17, dtype=torch.float32, device=dev)

    def _run(t, use_scratch):
        slot = torch.zeros(_INK_FP_STAT_W, dtype=torch.float32, device=dev)
        _ink_fp_stat(slot, t, scratch if use_scratch else None)
        return slot.cpu()

    fails = []

    def _check(name, cond):
        print(f"    stat_case {name}: {'ok' if cond else 'FAIL'}", flush=True)
        if not cond:
            fails.append(name)

    # (a) finite bf16 vector: flag 0, max_abs 3.0, numel 3; scratch == fallback.
    t = torch.tensor([1.0, -3.0, 2.0], dtype=torch.bfloat16, device=dev)
    s, f = _run(t, True), _run(t, False)
    _check("finite_flag0", s[0].item() == 0.0 and f[0].item() == 0.0)
    _check("finite_maxabs3", abs(s[1].item() - 3.0) < 1e-2)
    _check("finite_numel3", s[3].item() == 3.0)
    _check("finite_scratch_eq_fallback",
           (s[0].item() > 0) == (f[0].item() > 0)
           and abs(s[1].item() - f[1].item()) < 1e-2)

    # (b) nan present: nonfinite flag > 0 on BOTH paths.
    t = torch.tensor([1.0, float("nan"), 2.0], dtype=torch.float32, device=dev)
    s, f = _run(t, True), _run(t, False)
    _check("nan_flag_scratch", s[0].item() > 0)
    _check("nan_flag_fallback", f[0].item() > 0)

    # (c) inf present: nonfinite flag > 0 (max_abs nan_to_num'd, so not checked).
    t = torch.tensor([1.0, float("inf"), -2.0], dtype=torch.float32, device=dev)
    s = _run(t, True)
    _check("inf_flag_scratch", s[0].item() > 0)

    # (d) NON-CONTIGUOUS bf16 slice (the q/k/v fused-slice case that made the old
    # reshape(-1) allocate): scratch copy_ must gather it correctly, flag 0.
    base = torch.randn(4, 8, dtype=torch.bfloat16, device=dev)
    t = base[:, ::2]
    assert not t.is_contiguous(), "test setup: expected non-contiguous slice"
    s = _run(t, True)
    _check("noncontig_numel16", s[3].item() == float(t.numel()))
    _check("noncontig_flag0", s[0].item() == 0.0)
    _check("noncontig_maxabs",
           abs(s[1].item() - t.abs().max().float().item()) < 5e-2)

    if fails:
        print(f"INKLING_FP_STAT_SELFTEST FAIL cases={fails}", flush=True)
        return 1
    print("INKLING_FP_STAT_SELFTEST PASS (alloc-free scratch path: finiteness "
          "flag + max_abs + numel correct, agrees with fallback, handles "
          "non-contiguous bf16)", flush=True)
    return 0


def _b2fix_selftest() -> int:
    """Focused unit test for the feedback #18 NO-PROBE B2 candidate toggles
    (``INKLING_B2_FIX``): the env gate parses correctly and is zero-cost when
    unset (production byte-unchanged), and the eager ``full_meta`` / ``persist_out``
    buffer edits behave. CPU-runnable (no full model); the fail-fast guard for the
    new no-probe code path before the expensive TP=4 free-run arms."""
    import torch

    from tensorrt_llm._torch.models.modeling_inkling import (  # noqa: PLC0415
        _B2_FIX_NAMES, InklingDecodeMeta, _b2_fix_active)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== INKLING_B2FIX_SELFTEST dev={dev} names={_B2_FIX_NAMES} ===",
          flush=True)
    fails = []

    def _check(name, cond):
        print(f"    b2fix_case {name}: {'ok' if cond else 'FAIL'}", flush=True)
        if not cond:
            fails.append(name)

    class _Owner:  # minimal stand-in for the InklingAttention head geometry
        local_num_heads = 3
        head_dim = 8

        class qkv_proj:

            class weight:
                dtype = torch.bfloat16

    saved = os.environ.get("INKLING_B2_FIX")
    try:
        # (a) gate parsing: unset -> all inactive (production byte-unchanged).
        os.environ.pop("INKLING_B2_FIX", None)
        _check("gate_unset_all_inactive",
               all(not _b2_fix_active(n) for n in _B2_FIX_NAMES))
        # (b) single value selects exactly one.
        os.environ["INKLING_B2_FIX"] = "persist_out"
        _check(
            "gate_single",
            _b2_fix_active("persist_out")
            and not _b2_fix_active("zero_kvpool"))
        # (c) comma list selects a subset; no substring false-positive.
        os.environ["INKLING_B2_FIX"] = "zero_kvpool,sync_meta"
        _check(
            "gate_list",
            _b2_fix_active("zero_kvpool") and _b2_fix_active("sync_meta")
            and not _b2_fix_active("full_meta"))
        os.environ["INKLING_B2_FIX"] = "full_metaX"
        _check("gate_no_substring", not _b2_fix_active("full_meta"))

        # (d) full_meta: whole cap wiped (seq_lens->1, page_table->0), no out_buf.
        os.environ["INKLING_B2_FIX"] = "full_meta"
        m = InklingDecodeMeta(0)
        m.cap, m.max_pages = 4, 2
        m.seq_lens = torch.arange(1, 5, dtype=torch.int32, device=dev)
        m.page_table = torch.arange(8, dtype=torch.int32, device=dev).view(4,
                                                                           2) + 1
        m._b2_fix_eager(dev)
        _check("full_meta_seqlens_1", bool((m.seq_lens == 1).all().item()))
        _check("full_meta_pt_0", bool((m.page_table == 0).all().item()))
        _check("full_meta_no_outbuf", m.out_buf is None)

        # (e) persist_out: stable zeroed [cap, nh, hd] buffer sized from owner.
        os.environ["INKLING_B2_FIX"] = "persist_out"
        m2 = InklingDecodeMeta(0)
        m2.cap = 5
        m2._owner = _Owner
        m2._b2_fix_eager(dev)
        ob = m2.out_buf
        _check("persist_out_shape",
               ob is not None and tuple(ob.shape) == (5, 3, 8))
        _check("persist_out_zero",
               ob is not None and bool((ob == 0).all().item()))
        _check("persist_out_dtype",
               ob is not None and ob.dtype == torch.bfloat16)
        m2._b2_fix_eager(dev)  # cap unchanged -> reused, not reallocated
        _check("persist_out_stable", m2.out_buf is ob)

        # (f) toggle OFF -> _b2_fix_eager is a no-op (no buffer touched).
        os.environ.pop("INKLING_B2_FIX", None)
        m3 = InklingDecodeMeta(0)
        m3.cap, m3.max_pages = 2, 1
        m3.seq_lens = torch.full((2, ), 9, dtype=torch.int32, device=dev)
        m3.page_table = torch.full((2, 1), 7, dtype=torch.int32, device=dev)
        m3._owner = _Owner
        m3._b2_fix_eager(dev)
        _check("off_noop_seqlens", bool((m3.seq_lens == 9).all().item()))
        _check("off_noop_no_outbuf", m3.out_buf is None)

        # (g) pad_scatter: drive the real refresh() with a minimal mock
        # attn_metadata + KV manager. A CUDA-graph padding/dummy gen row
        # (num_cached==0 -> seq_len==1) is redirected to the reserved scratch page
        # (num_pages-1); the real row (num_cached>=1 -> seq_len>=2) keeps its page.
        class _KvParams:
            num_cached_tokens_per_seq = [3, 0]  # real: 3 cached; dummy: 0

        class _Mgr:
            max_blocks_per_seq = 4

            def get_buffers(self, layer, kv_layout="HND"):
                return torch.zeros(10, 2, 1, 1, 1, device=dev)  # num_pages=10

            def get_batch_cache_indices(self, ids, layer):
                return [[5], []]  # row0 real -> page 5; row1 dummy -> no page

        class _AttnMeta:
            request_ids = [100, 999]  # 1 real + 1 CUDA-graph dummy
            num_contexts = 0
            is_cuda_graph = True
            kv_cache_manager = _Mgr()
            kv_cache_params = _KvParams()

        os.environ["INKLING_B2_FIX"] = "pad_scatter"
        m4 = InklingDecodeMeta(0)
        ok4 = m4.refresh(_AttnMeta(), dev)
        _check("pad_scatter_ready", ok4 is True)
        _check("pad_scatter_num_pages", m4.num_pages == 10)
        _check("pad_scatter_real_kept",
               ok4 and int(m4.page_table[0, 0].item()) == 5)
        _check("pad_scatter_dummy_scratch",
               ok4 and int(m4.page_table[1, 0].item()) == 9)
        _check("pad_scatter_real_seqlen",
               ok4 and int(m4.seq_lens[0].item()) == 4)

        # (h) pad_scatter OFF -> dummy row keeps the production default (page 0)
        # and num_pages is never resolved (byte-identical to production).
        os.environ.pop("INKLING_B2_FIX", None)
        m5 = InklingDecodeMeta(0)
        ok5 = m5.refresh(_AttnMeta(), dev)
        _check("pad_scatter_off_dummy_page0",
               ok5 and int(m5.page_table[1, 0].item()) == 0)
        _check("pad_scatter_off_num_pages_untouched", m5.num_pages is None)

        # (i) CONFOUND FIX (iter-90): with pad_scatter ON, an EAGER
        # (is_cuda_graph=False) num_gen==1 step must NOT redirect its sole row even
        # though sl==1 (num_cached==0). iter-88 keyed on sl==1 alone and so also
        # scattered the real warmup row to scratch, invalidating its NO_LEAD. The
        # corrected detector gates on ``is_graph and num_gen > 1`` -> a padded
        # CUDA-graph batch only; an eager/num_gen==1 row keeps its real page and
        # num_pages is never resolved.
        class _KvParamsEager:
            num_cached_tokens_per_seq = [0]  # sole row, sl==1

        class _MgrEager:
            max_blocks_per_seq = 4

            def get_buffers(self, layer, kv_layout="HND"):
                return torch.zeros(10, 2, 1, 1, 1, device=dev)

            def get_batch_cache_indices(self, ids, layer):
                return [[7]]  # sole row -> page 7

        class _AttnMetaEager:
            request_ids = [100]  # num_gen == 1, no padding
            num_contexts = 0
            is_cuda_graph = False  # eager warmup, not a padded graph batch
            kv_cache_manager = _MgrEager()
            kv_cache_params = _KvParamsEager()

        os.environ["INKLING_B2_FIX"] = "pad_scatter"
        m6 = InklingDecodeMeta(0)
        ok6 = m6.refresh(_AttnMetaEager(), dev)
        _check("pad_scatter_eager_no_redirect",
               ok6 and int(m6.page_table[0, 0].item()) == 7)
        _check("pad_scatter_eager_num_pages_untouched", m6.num_pages is None)
    finally:
        if saved is None:
            os.environ.pop("INKLING_B2_FIX", None)
        else:
            os.environ["INKLING_B2_FIX"] = saved

    if fails:
        print(f"INKLING_B2FIX_SELFTEST FAIL cases={fails}", flush=True)
        return 1
    print(
        "INKLING_B2FIX_SELFTEST PASS (gate parse + full_meta memset + "
        "persist_out stable zeroed buffer + pad_scatter dummy->scratch-page "
        "redirect + real-page-avoiding scratch + eager/num_gen==1 confound-fix "
        "no-redirect + toggle-off no-op)",
        flush=True)
    return 0


def _det_selftest() -> int:
    """Validate the feedback #3/#4 determinism gate mapping (CPU-only, no GPU):
    the exact (max_batch_size, enable_autotuner) the LLM build consumes."""
    fails = []
    if _det_llm_config(True) != (1, False):
        fails.append(f"deterministic->{_det_llm_config(True)} != (1, False)")
    if _det_llm_config(False) != (8, True):
        fails.append(f"nondet->{_det_llm_config(False)} != (8, True)")
    # DETERMINISTIC must reflect the env exactly (strict "1", not truthiness).
    saved = os.environ.get("INKLING_FP_DETERMINISTIC")
    try:
        for val, want in (("1", True), ("0", False), ("", False), (None, False)):
            if val is None:
                os.environ.pop("INKLING_FP_DETERMINISTIC", None)
            else:
                os.environ["INKLING_FP_DETERMINISTIC"] = val
            got = os.environ.get("INKLING_FP_DETERMINISTIC") == "1"
            if got != want:
                fails.append(f"env[{val!r}]->{got} != {want}")
    finally:
        if saved is None:
            os.environ.pop("INKLING_FP_DETERMINISTIC", None)
        else:
            os.environ["INKLING_FP_DETERMINISTIC"] = saved

    # feedback #19 Lane-1 sanitizer overrides: _resolved_llm_config folds
    # INKLING_FP_BS / INKLING_FP_AUTOTUNE on top of the determinism gate, and is
    # byte-unchanged when neither is set. Validate the exact (bs, at) the sanitizer
    # arm will consume so the batch=8 + autotuner-off knob cannot silently drift.
    sbs = os.environ.get("INKLING_FP_BS")
    sat = os.environ.get("INKLING_FP_AUTOTUNE")
    try:
        os.environ.pop("INKLING_FP_BS", None)
        os.environ.pop("INKLING_FP_AUTOTUNE", None)
        # (a) no override -> identical to the determinism gate (production default).
        if _resolved_llm_config(True) != (1, False):
            fails.append(f"resolved_det_noovr->{_resolved_llm_config(True)} != (1, False)")
        if _resolved_llm_config(False) != (8, True):
            fails.append(f"resolved_nondet_noovr->{_resolved_llm_config(False)} != (8, True)")
        # (b) batch=8 + autotuner-off (the sanitizer arm) regardless of det gate.
        os.environ["INKLING_FP_BS"] = "8"
        os.environ["INKLING_FP_AUTOTUNE"] = "0"
        if _resolved_llm_config(True) != (8, False):
            fails.append(f"resolved_ovr_bs8_at0->{_resolved_llm_config(True)} != (8, False)")
        # (c) autotuner override alone flips the bool but keeps the gate batch.
        os.environ.pop("INKLING_FP_BS", None)
        os.environ["INKLING_FP_AUTOTUNE"] = "1"
        if _resolved_llm_config(True) != (1, True):
            fails.append(f"resolved_ovr_at1->{_resolved_llm_config(True)} != (1, True)")
    finally:
        for k, v in (("INKLING_FP_BS", sbs), ("INKLING_FP_AUTOTUNE", sat)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # feedback #19 Lane-1 wall-time lever: _max_seq_config folds INKLING_FP_MAX_SEQ
    # / INKLING_FP_MAX_NUM_TOKENS, defaulting to the byte-unchanged 2048/2048.
    smseq = os.environ.get("INKLING_FP_MAX_SEQ")
    smntok = os.environ.get("INKLING_FP_MAX_NUM_TOKENS")
    try:
        os.environ.pop("INKLING_FP_MAX_SEQ", None)
        os.environ.pop("INKLING_FP_MAX_NUM_TOKENS", None)
        if _max_seq_config() != (2048, 2048):
            fails.append(f"maxseq_default->{_max_seq_config()} != (2048, 2048)")
        os.environ["INKLING_FP_MAX_SEQ"] = "512"
        os.environ["INKLING_FP_MAX_NUM_TOKENS"] = "512"
        if _max_seq_config() != (512, 512):
            fails.append(f"maxseq_override->{_max_seq_config()} != (512, 512)")
    finally:
        for k, v in (("INKLING_FP_MAX_SEQ", smseq),
                     ("INKLING_FP_MAX_NUM_TOKENS", smntok)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    if fails:
        print(f"INKLING_FP_DET_SELFTEST FAIL cases={fails}", flush=True)
        return 1
    print("INKLING_FP_DET_SELFTEST PASS (det->bs1/autotune-off; "
          "nondet->bs8/autotune-on; strict '1' env parse; "
          "resolved overrides bs8/at0 + at-only; "
          "max_seq default 2048/override 512)", flush=True)
    return 0


def main() -> int:
    import torch  # noqa: F401
    from transformers import AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import \
        InklingForConditionalGeneration  # noqa: F401  (registers auto-model)
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig

    assert torch.cuda.is_available(), "B2 localizer needs CUDA GPUs"
    # INKLING_FP (the per-layer/op fingerprint dump base) is OPTIONAL. When it is
    # UNSET the model allocates NO fingerprint buffers and records NO device->device
    # copy_ probes into the decode CUDA graph -- a PURE production forward. This is
    # the decisive control for the B2 instrumentation-artifact probe: the residual
    # `_ink_fp` copy_ recorded into the graph is itself a candidate cause of the
    # cg1 collapse, so the no-instrumentation arm reports whether the REAL model
    # output collapses with zero graph perturbation.
    fp_base = os.environ.get("INKLING_FP")

    with open(REF) as f:
        refdoc = json.load(f)
    ref = refdoc["prompts"] if isinstance(refdoc, dict) else refdoc
    ref = [r for r in ref if r.get("input_ids")]
    assert ref, "no ref prompt with input_ids"
    # batch=1: the exact B2 smoke condition (1 served chat request collapsed).
    prompt_ids = list(ref[0]["input_ids"])
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    print(f"[fp] tp={TP} cuda_graph={CUDA_GRAPH} overlap={OVERLAP} moe={MOE_BACKEND} "
          f"steps={STEPS} prompt_len={len(prompt_ids)} "
          f"fp={fp_base or '<none:no-instrumentation>'}", flush=True)
    # Greppable determinism-config provenance (feedback #3/#4): the harness asserts
    # this to prove the mandated deterministic config is actually live in the run.
    # _resolved_llm_config folds in the feedback #19 sanitizer overrides (batch=8 +
    # autotuner-off) on top of the determinism gate; kv_fraction is the Lane-1
    # footprint lever. All default to the byte-unchanged production values.
    _dbs, _dat = _resolved_llm_config(DETERMINISTIC)
    _mseq, _mntok = _max_seq_config()
    print(f"INKLING_FP_DET_CONFIG deterministic={DETERMINISTIC} "
          f"max_batch_size={_dbs} enable_autotuner={_dat} "
          f"kv_fraction={KV_FRACTION} "
          f"max_seq_len={_mseq} max_num_tokens={_mntok} "
          f"allreduce_autotune_disabled={os.environ.get('TLLM_DISABLE_ALLREDUCE_AUTOTUNE', '0')}",
          flush=True)

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=KV_FRACTION,
                                    dtype="auto", enable_block_reuse=False)
    llm = LLM(
        CKPT,
        tensor_parallel_size=TP,
        trust_remote_code=True,
        attn_backend="TRTLLM",
        moe_config=MoeConfig(backend=MOE_BACKEND),
        kv_cache_config=kv_cache_config,
        cuda_graph_config=CudaGraphConfig() if CUDA_GRAPH else None,
        disable_overlap_scheduler=not OVERLAP,
        max_seq_len=_mseq,
        max_batch_size=_dbs,
        enable_autotuner=_dat,
        max_num_tokens=_mntok,
    )
    hard_path = "CudaGraphConfig()" if CUDA_GRAPH else "eager(no-graph)"
    print(f"[fp] cuda_graph_hard_path={hard_path}", flush=True)

    try:
        # Greedy decode. The model-side hook dumps the per-layer decode
        # fingerprint (and, when INKLING_FP_OPS is set, the intra-layer op-level
        # fingerprint) per rank per step to ${INKLING_FP}[.ops].rank{r}.step{s}.
        # INKLING_FP_FORCE_LEN forces EXACTLY STEPS tokens (ignore_eos +
        # min_tokens) so the cg0 (eager, ground truth) and cg1 (graph) arms are
        # step-aligned for the op-level positional bisection (feedback #17); a
        # natural greedy decode can emit EOS at different steps in the two arms.
        # This forcing does NOT create the collapse -- the eager arm forced the
        # same way stays coherent while the graph arm collapses to token 0 ('!'),
        # the nan->argmax-0 B2 signature.
        force_len = os.environ.get("INKLING_FP_FORCE_LEN") == "1"
        sp = SamplingParams(max_tokens=STEPS, temperature=0.0)
        if force_len:
            sp = SamplingParams(max_tokens=STEPS, min_tokens=STEPS,
                                temperature=0.0, ignore_eos=True)
        out = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids)], sp)[0]
        trt_ids = [int(x) for x in out.outputs[0].token_ids]
        rep = _max_consec_repeat(trt_ids)
        uni = len(set(trt_ids))
        try:
            txt = tok.decode([i for i in trt_ids if i >= 0])
        except Exception:  # noqa: BLE001
            txt = "<decode-err>"
        try:
            on_vocab = all(0 <= i < tok.vocab_size for i in trt_ids)
        except Exception:  # noqa: BLE001
            on_vocab = None
        collapse = (rep >= 8) or (uni < 3)
        print(f"[fp] FREE-RUN out_ids={trt_ids}", flush=True)
        print(f"[fp] FREE-RUN n_out={len(trt_ids)} max_repeat={rep} unique={uni} "
              f"{'COLLAPSE' if collapse else 'ok'} text={txt[:80]!r}", flush=True)
        # Per-config metadata for the analyzer's step->token alignment and the
        # capture-vs-replay call (feedback #17 STEP 3): the FIRST decode step whose
        # emitted token diverges cg0-vs-cg1 is the real replay-corruption step; a
        # fingerprint that is non-finite BEFORE that step is a capture/warmup
        # artifact, not the emitted-decode path.
        if fp_base:
            try:
                with open(f"{fp_base}.meta.json", "w") as _mf:
                    json.dump({
                        "out_ids": trt_ids,
                        "prompt_len": len(prompt_ids),
                        "n_out": len(trt_ids),
                        "cuda_graph": CUDA_GRAPH,
                        "overlap": OVERLAP,
                        "moe_backend": MOE_BACKEND,
                        "collapse": collapse,
                        "max_repeat": rep,
                        "unique": uni,
                        "force_len": force_len,
                    }, _mf)
            except Exception:  # noqa: BLE001
                import traceback
                traceback.print_exc()
        print(f"INKLING_FP_RUN_DONE cuda_graph={CUDA_GRAPH} overlap={OVERLAP} "
              f"collapse={collapse} max_repeat={rep} unique={uni} "
              f"n_out={len(trt_ids)} prompt_len={len(prompt_ids)} "
              f"on_vocab={on_vocab} cuda_graph_hard_path={hard_path}", flush=True)
    finally:
        llm.shutdown()
    return 0


if __name__ == "__main__":
    try:
        if "--stat-selftest" in sys.argv:
            sys.exit(_stat_selftest())
        if "--b2fix-selftest" in sys.argv:
            sys.exit(_b2fix_selftest())
        if "--det-selftest" in sys.argv:
            sys.exit(_det_selftest())
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        sys.exit(1)
