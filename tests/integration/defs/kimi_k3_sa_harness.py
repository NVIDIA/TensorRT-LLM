# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 SA spec-dec / sanity test harness via the LLM API.

Runs greedy prompts through the standard TRT-LLM PyTorch backend and
asserts the outputs contain expected content, optionally with SA
(suffix-automaton) speculative decoding and parity checking against a
baseline run. Two modes:

* full model (default): needs 16 GPUs (4x GB300 trays), asserts output
  quality (coherent completions, correct GSM8K-style answer).
* truncated (``KIMI_K3_NUM_LAYERS=<N>``, e.g. 4 on a single 4-GPU tray):
  the harness builds a temporary truncated copy of the checkpoint config
  (weight shards are symlinked; extra layers are ignored at load time) and
  only asserts the pipeline runs e2e (load -> prefill -> decode ->
  shutdown); output text is NOT checked (a 4/93-layer model produces
  gibberish by construction).

Driven by tests/integration/defs/test_kimi_k3_specdec.py; can also be run
standalone:

    python tests/integration/defs/kimi_k3_sa_harness.py

Environment:
  KIMI_K3_CKPT                 model dir (required)
  KIMI_K3_TP                   tensor_parallel_size == EP width (default 16;
                               896 % TP must be 0)
  KIMI_K3_ADP                  1 (default) = DEP deployment (attention-DP +
                               MoE EP dispatch/combine, EP width == TP);
                               0 = plain EP (replicated attention +
                               allreduce) — both modes are test-coverable
  KIMI_K3_NUM_LAYERS           truncate to first N layers via a temporary
                               checkpoint copy (skips output-quality
                               assertions)
  KIMI_K3_MOE_BACKEND          moe_config.backend passed to the LLM
                               (default AUTO). Currently a no-op for
                               Kimi K3: KimiK3MoERuntime pins the routed
                               MoE backend to TRTLLM regardless; parity
                               holds because the baseline and spec runs
                               share the same backend
  KIMI_K3_SPEC_MODE            'off' (default) or 'sa' (suffix
                               automaton, one-engine)
  KIMI_K3_SPEC_DRAFT_LEN       SA max_draft_len (default 2)
  KIMI_K3_SA_NGRAM_SIZE        SA matching mode (default -1 = longest
                               match; >=1 = fixed ngram size)
  KIMI_K3_SPEC_PARITY          0 (off) | 1/text (bit-identical outputs;
                               full model) | logits (tolerance logprob
                               comparison along shared prefix + near-tie
                               check at divergence; truncated models)
  KIMI_K3_OUTPUT_JSON          dump this run's outputs (text/token_ids/
                               logprobs) to a JSON file
  KIMI_K3_BASELINE_JSON        load the parity baseline from a prior run's
                               JSON instead of a second in-process model
                               (required for full-model parity: the model
                               does not fit twice in one process)
  KIMI_K3_SPEC_LP_TOL          logits-parity drift tolerance (default 1.0;
                               catastrophic-corruption net on truncated
                               noise models — see _compare_logits_parity)
  KIMI_K3_SPEC_TIE_TOL         logits-parity near-tie gap bound (default 0.3)
Exit code 0 = PASS, 1 = FAIL.
"""

import json
import os
import sys
import tempfile
import time

# tensorrt_llm imports live inside _build_llm/_generate so the comparison
# machinery (_compare_logits_parity, _parity_prompts, _chosen_logprob) stays
# importable from client-side tools (kimi_k3_disagg_parity.py) that run
# without a GPU-side tensorrt_llm install.

# Keeps the truncated-checkpoint temp dir alive for the process lifetime.
_TRUNCATED_CKPT_DIR = None


def _truncated_checkpoint(ckpt: str, num_layers: int) -> str:
    """Build a temporary checkpoint dir truncated to the first N layers.

    The doctored ``config.json`` sets ``num_hidden_layers`` and filters the
    1-indexed ``linear_attn_config`` layer schedules consistently; every
    other file (tokenizer, index, weight shards) is symlinked. Extra
    checkpoint layers are ignored at load time, so the full shard set can
    stay in place.

    The copy lives in node-local tmp, so truncated runs are single-node
    only (remote ranks could not resolve the doctored config); the 4-GPU
    integration test this serves always fits one node.
    """
    global _TRUNCATED_CKPT_DIR
    _TRUNCATED_CKPT_DIR = tempfile.TemporaryDirectory(prefix="kimi-k3-truncated-")
    out = _TRUNCATED_CKPT_DIR.name
    with open(os.path.join(ckpt, "config.json")) as f:
        config = json.load(f)
    text = config.get("text_config", config)
    assert 0 < num_layers <= text["num_hidden_layers"]
    text["num_hidden_layers"] = num_layers
    lin = dict(text["linear_attn_config"])
    lin["kda_layers"] = [n for n in lin["kda_layers"] if n <= num_layers]
    lin["full_attn_layers"] = [n for n in lin["full_attn_layers"] if n <= num_layers]
    text["linear_attn_config"] = lin
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f)
    for entry in os.scandir(ckpt):
        if entry.name != "config.json":
            os.symlink(entry.path, os.path.join(out, entry.name))
    return out


PROMPTS_AND_CHECKS = [
    ("The capital of France is", "Paris"),
    ("1 + 1 = 2, 2 + 2 = 4, 4 + 4 =", "8"),
    ("Water is made of hydrogen and", "oxygen"),
    (
        "Question: Natalia sold clips to 48 of her friends in April, and "
        "then she sold half as many clips in May. How many clips did "
        "Natalia sell altogether in April and May?\nAnswer:",
        "#### 72",
    ),
]


def _graphs_enabled() -> bool:
    return os.environ.get("KIMI_K3_CUDA_GRAPHS", "1") == "1"


def _build_llm(ckpt: str, tp: int, spec_mode: str, adp: bool):
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig

    speculative_config = None
    if spec_mode == "sa":
        from tensorrt_llm.llmapi import SADecodingConfig

        speculative_config = SADecodingConfig(
            max_draft_len=int(os.environ.get("KIMI_K3_SPEC_DRAFT_LEN", "2")),
            # -1 = longest match via the suffix automaton (the SA
            # differentiator); >=1 pins a fixed ngram size.
            max_matching_ngram_size=int(os.environ.get("KIMI_K3_SA_NGRAM_SIZE", "-1")),
            # Cross-request pattern reuse (SA's differentiator on
            # homogeneous workloads); pool must be >= max_batch_size.
            enable_global_pool=os.environ.get("KIMI_K3_SA_GLOBAL_POOL", "0") == "1",
        )
    elif spec_mode != "off":
        raise ValueError(f"KIMI_K3_SPEC_MODE={spec_mode!r} (expected 'off' or 'sa')")
    return LLM(
        model=ckpt,
        tensor_parallel_size=tp,
        enable_attention_dp=adp,
        moe_expert_parallel_size=tp if adp else None,
        trust_remote_code=True,  # tiktoken tokenizer ships with the ckpt
        # Currently a no-op for Kimi K3: KimiK3MoERuntime pins the routed
        # MoE backend to TRTLLM regardless of moe_config.backend. Kept as
        # a passthrough for checkpoints that honor it; parity is
        # unaffected because the baseline and spec runs share the backend.
        moe_config=MoeConfig(backend=os.environ.get("KIMI_K3_MOE_BACKEND", "AUTO")),
        max_batch_size=int(os.environ.get("KIMI_K3_MAX_BATCH_SIZE", "8")),
        max_seq_len=int(os.environ.get("KIMI_K3_MAX_SEQ_LEN", "4096")),
        max_num_tokens=int(os.environ.get("KIMI_K3_MAX_NUM_TOKENS", "4096")),
        enable_chunked_prefill=False,
        # Non-spec standalone runs use the upstream defaults (overlap +
        # CUDA graphs). Spec-dec runs keep the validated conservative
        # settings (SA is graph-safe/overlap-capable in principle, but the
        # K3 verify/promote path is only certified eager without overlap).
        # Parity runs force BOTH instances into the conservative regime via
        # KIMI_K3_CUDA_GRAPHS=0 (set in main) — a baseline compared against
        # an eager spec run must not execute under a different regime.
        # KIMI_K3_SPEC_CUDA_GRAPHS=1 opts a spec-dec run into CUDA graphs
        # (EXPERIMENTAL: SA is graph-safe by design — draft padding keeps
        # shapes static — but the K3 verify/promote path is not yet
        # certified under capture). Overlap stays off for spec runs.
        disable_overlap_scheduler=(speculative_config is not None or not _graphs_enabled()),
        cuda_graph_config=(
            CudaGraphConfig(enable_padding=True, max_batch_size=8)
            if _graphs_enabled()
            and (
                speculative_config is None or os.environ.get("KIMI_K3_SPEC_CUDA_GRAPHS", "0") == "1"
            )
            else None
        ),
        speculative_config=speculative_config,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            free_gpu_memory_fraction=float(os.environ.get("KIMI_K3_FREE_GPU_FRACTION", "0.25")),
            # tokens_per_block=64 keeps the MLA (576, 512) generation path
            # on the flashinfer trtllm-gen kernel (32 falls back to a C++
            # path requiring num_heads % 64 == 0; K3 has 96 query heads).
            tokens_per_block=64,
        ),
    )


def _parity_prompts(extra: int):
    """Deterministic extra prompt pool for spec-dec parity statistics.

    More prompts = more independent trajectories: shared-prefix drift is
    sampled widely and divergence events (near-tie vs not) become
    meaningful in aggregate — a state bug corrupts trajectories
    systematically, rounding flips are scattered. Templates mix factual
    stubs, arithmetic, and repetitive structure (the latter gives the SA
    real acceptances, exercising the state-promotion path). Model loading
    dominates runtime, so extra prompts are nearly free.
    """
    subjects = [
        "France",
        "Japan",
        "Brazil",
        "Canada",
        "Egypt",
        "Kenya",
        "Norway",
        "Peru",
        "Thailand",
        "Greece",
        "Chile",
        "Poland",
    ]
    templates = [
        "The capital of {s} is",
        "Q: Name three facts about {s}.\nA: 1.",
        "{s}, {s}, {s}. The word repeated above is",
        "Count by twos: 2, 4, 6, 8, 10,",
    ]
    return [
        (
            templates[i % len(templates)].format(s=subjects[(i // len(templates)) % len(subjects)]),
            None,
        )
        for i in range(extra)
    ]


def _generate(llm, prompts, max_tokens: int, want_logprobs: bool = False):
    from tensorrt_llm import SamplingParams

    want_stats = os.environ.get("KIMI_K3_SPEC_STATS", "0") == "1"
    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        logprobs=5 if want_logprobs else None,
        return_perf_metrics=want_stats,
    )
    t0 = time.monotonic()
    outputs = llm.generate(prompts, sampling)
    wall = time.monotonic() - t0
    print(
        f"[sanity] generate wall: {wall:.1f}s for {len(prompts)} prompts x {max_tokens} max_tokens"
    )
    if want_stats:
        _print_spec_stats(outputs)
    return [out.outputs[0] for out in outputs]


def _print_spec_stats(outputs):
    """Aggregate speculative-decoding stats (KIMI_K3_SPEC_STATS=1).

    tokens/step (avg_decoded_tokens_per_iter) is the headline spec-dec
    win; acceptance rate = accepted draft tokens / drafted tokens. On a
    non-spec baseline run both simply report the trivial values.
    """
    tokens_per_iter = [
        out.avg_decoded_tokens_per_iter
        for out in outputs
        if getattr(out, "avg_decoded_tokens_per_iter", None) is not None
    ]
    accepted = drafted = 0
    for out in outputs:
        pm = getattr(out.outputs[0], "request_perf_metrics", None)
        sd = getattr(pm, "speculative_decoding", None) if pm else None
        if sd is not None:
            accepted += sd.total_accepted_draft_tokens
            drafted += sd.total_draft_tokens
    if tokens_per_iter:
        mean_tpi = sum(tokens_per_iter) / len(tokens_per_iter)
        print(
            f"[sanity] spec stats: tokens/step mean {mean_tpi:.3f} "
            f"(min {min(tokens_per_iter):.3f}, "
            f"max {max(tokens_per_iter):.3f}, n={len(tokens_per_iter)})"
        )
    if drafted > 0:
        print(f"[sanity] spec stats: acceptance {accepted}/{drafted} = {accepted / drafted:.1%}")


def _chosen_logprob(logprob_dict, token_id):
    entry = logprob_dict.get(token_id)
    if entry is None:
        return None
    return getattr(entry, "logprob", entry)


def _dump_completions(path, completions, want_logprobs):
    import json

    payload = []
    for comp in completions:
        entry = {"text": comp.text, "token_ids": list(comp.token_ids)}
        if want_logprobs and comp.logprobs is not None:
            entry["logprobs"] = [
                {str(tid): float(getattr(lp, "logprob", lp)) for tid, lp in pos.items()}
                for pos in comp.logprobs
            ]
        payload.append(entry)
    with open(path, "w") as f:
        json.dump(payload, f)
    print(f"[sanity] outputs dumped to {path}")


def _load_completions(path):
    import json
    from types import SimpleNamespace

    with open(path) as f:
        payload = json.load(f)
    loaded = []
    for entry in payload:
        logprobs = None
        if "logprobs" in entry:
            logprobs = [{int(t): lp for t, lp in pos.items()} for pos in entry["logprobs"]]
        loaded.append(
            SimpleNamespace(text=entry["text"], token_ids=entry["token_ids"], logprobs=logprobs)
        )
    return loaded


def _compare_logits_parity(base, spec, prompt, failures, tol=None, tie_tol=None):
    """Tolerance-based spec-dec parity along the shared output prefix.

    Logits at position i depend only on tokens 0..i-1, so positions up to
    and INCLUDING the first divergence were computed from identical
    histories and must agree to rounding tolerance. At the divergence
    itself the flip must be a near-tie (top-2 logprob gap < tie_tol) —
    that positively identifies benign reduction-order rounding, whereas a
    confident token flipping or a drifting prefix indicates a real
    verification/state bug. Positions after the divergence are
    uncomparable (legitimately different histories) and ignored.

    Tolerances: KIMI_K3_SPEC_LP_TOL (default 1.0) bounds shared-prefix
    drift; KIMI_K3_SPEC_TIE_TOL (default 0.3) bounds the divergence gap.
    On truncated NOISE models the reduction-order wobble is amplified by
    the recurrent state layer-over-layer AND position-over-position
    (~0.2 observed at position 1-5, ~0.4 at position 11, with
    kernel-exact verify math), so the drift tolerance is calibrated as a
    catastrophic-corruption net: real state bugs (wrong slot / wrong
    step / missed promotion) produce O(1-10) logprob errors. Precision
    correctness is owned by the exact KDA verify unit test; full-model
    strict text parity is the e2e proof.
    """
    if tol is None:
        tol = float(os.environ.get("KIMI_K3_SPEC_LP_TOL", "1.0"))
    if tie_tol is None:
        tie_tol = float(os.environ.get("KIMI_K3_SPEC_TIE_TOL", "0.3"))
    b_ids, s_ids = list(base.token_ids), list(spec.token_ids)
    b_lp, s_lp = base.logprobs, spec.logprobs
    if not b_lp or len(b_lp) != len(b_ids):
        failures.append(
            f"baseline for {prompt!r} has "
            f"{len(b_lp) if b_lp else 0} logprob entries for "
            f"{len(b_ids)} tokens; logits parity requires aligned "
            "baseline logprobs (dump the baseline with "
            "KIMI_K3_DUMP_LOGPROBS=1)"
        )
        return "drift"
    shared = 0
    while shared < min(len(b_ids), len(s_ids)) and b_ids[shared] == s_ids[shared]:
        shared += 1
    # One-engine spec samplers (SA/MTP/Eagle3) do not emit per-token
    # logprobs (SpecSampler stores tokens only); the spec run's
    # logprobs list is then shorter than its token_ids and NOT
    # position-aligned. Fall back to one-sided certification: the
    # divergence near-tie classification below (baseline-side logprobs)
    # still applies; shared-prefix drift is only checkable when the spec
    # sampler returned aligned per-token logprobs (host-drafter modes).
    s_lp_aligned = bool(s_lp) and len(s_lp) == len(s_ids)
    if not s_lp_aligned:
        print(
            f"[sanity] NOTE: spec run returned "
            f"{len(s_lp) if s_lp else 0} logprob entries for "
            f"{len(s_ids)} tokens (one-engine sampler); one-sided "
            "parity — drift check skipped, near-tie classification "
            "(baseline-side) active"
        )
    for i in range(shared if s_lp_aligned else 0):
        lb = _chosen_logprob(b_lp[i], b_ids[i])
        ls = _chosen_logprob(s_lp[i], s_ids[i])
        if lb is None or ls is None:
            continue
        if abs(lb - ls) > tol:
            failures.append(
                f"logit drift for {prompt!r} at shared position {i}: "
                f"baseline lp={lb:.4f} vs specdec lp={ls:.4f} "
                f"(|diff| > {tol})"
            )
            return "drift"
    if shared < min(len(b_ids), len(s_ids)):
        # First divergence: expected to be a near-tie in the baseline
        # distribution. The drift check above is the hard regression
        # signal (prompt-agnostic, stable); a single non-tie flip is only
        # a warning (borderline gaps occur legitimately on noise models),
        # but the caller hard-fails when non-ties dominate in aggregate —
        # a real state bug corrupts trajectories systematically.
        top = sorted((getattr(v, "logprob", v) for v in b_lp[shared].values()), reverse=True)
        gap = top[0] - top[1] if len(top) > 1 else float("inf")
        if gap > tie_tol:
            message = (
                f"non-tie divergence for {prompt!r} at position {shared}: "
                f"baseline top-2 logprob gap {gap:.4f} > {tie_tol} "
                f"(confident token flipped — investigate if drift also seen)"
            )
            if os.environ.get("KIMI_K3_SPEC_TIE_STRICT", "0") == "1":
                failures.append(message)
            else:
                print(f"[sanity] WARNING: {message}")
            return "non_tie"
        print(
            f"[sanity] {prompt!r}: benign divergence at position "
            f"{shared} (top-2 gap {gap:.4f}, shared prefix verified)"
        )
        return "benign"
    print(f"[sanity] {prompt!r}: full output identical ({shared} tokens)")
    return "identical"


def main() -> int:
    ckpt = os.environ["KIMI_K3_CKPT"]
    tp = int(os.environ.get("KIMI_K3_TP", "16"))
    num_layers = os.environ.get("KIMI_K3_NUM_LAYERS")
    truncated = num_layers is not None
    if truncated:
        ckpt = _truncated_checkpoint(ckpt, int(num_layers))
    max_tokens = int(os.environ.get("KIMI_K3_MAX_TOKENS", "64"))
    spec_mode = os.environ.get("KIMI_K3_SPEC_MODE", "off")
    # Spec-dec parity modes (baseline greedy runs first, then spec):
    #   1 / text : outputs must be BIT-IDENTICAL (use on the full model,
    #              where confident logits absorb kernel-rounding noise)
    #   logits   : tolerance-based logprob comparison along the shared
    #              prefix; divergences must be near-ties (use on truncated
    #              models, where noise logits flip argmax on rounding)
    spec_parity = os.environ.get("KIMI_K3_SPEC_PARITY", "0")
    if spec_parity == "1":
        spec_parity = "text"
    if spec_parity not in ("0", "text", "logits"):
        raise ValueError(
            f"KIMI_K3_SPEC_PARITY={spec_parity!r} (expected '0', '1'/'text', or 'logits')"
        )

    # DEP deployment (attention data-parallel + MoE expert-parallel
    # dispatch/combine; EP width == tp) is the default. KIMI_K3_ADP=0
    # selects the plain EP mode (replicated attention + latent allreduce).
    adp = os.environ.get("KIMI_K3_ADP", "1") == "1"
    print(
        f"[sanity] ckpt={ckpt} tp(EP)={tp} adp={adp} truncated={truncated} "
        f"moe_backend={os.environ.get('KIMI_K3_MOE_BACKEND', 'AUTO')} "
        f"spec_mode={spec_mode} spec_parity={spec_parity}"
    )

    want_logprobs = spec_parity == "logits" or os.environ.get("KIMI_K3_DUMP_LOGPROBS", "0") == "1"
    # Cross-process parity: a prior baseline run dumps its outputs via
    # KIMI_K3_OUTPUT_JSON; this run loads them via KIMI_K3_BASELINE_JSON
    # instead of loading a second in-process model (the full model does
    # not fit twice — shutdown does not release everything).
    baseline_json = os.environ.get("KIMI_K3_BASELINE_JSON")
    output_json = os.environ.get("KIMI_K3_OUTPUT_JSON")

    # Extra parity-only prompts (default 0). More trajectories make the
    # aggregate divergence statistics meaningful; quality checks below
    # apply only to the base PROMPTS_AND_CHECKS.
    # Honored whenever explicitly requested (default 0 = no change):
    # baseline-dump runs (KIMI_K3_OUTPUT_JSON, parity off) need the same
    # extra prompts as the parity run that will consume the dump.
    extra = int(os.environ.get("KIMI_K3_SPEC_NUM_PROMPTS", "0"))
    prompt_set = PROMPTS_AND_CHECKS + _parity_prompts(extra)
    prompt_texts = [p for p, _ in prompt_set]

    baseline = None
    if spec_parity != "0":
        # Regime-match the parity baseline to the (eager) spec run.
        os.environ.setdefault("KIMI_K3_CUDA_GRAPHS", "0")
        # A cross-process baseline permits no-spec regression checks
        # (e.g. verifying a model-class change is behavior-neutral by
        # comparing two spec-off runs across code states).
        assert spec_mode != "off" or baseline_json, (
            "KIMI_K3_SPEC_PARITY requires a spec mode, or a cross-process "
            "baseline via KIMI_K3_BASELINE_JSON for no-spec regression "
            "checks"
        )
        if baseline_json:
            baseline = _load_completions(baseline_json)
            if len(baseline) != len(prompt_set):
                raise ValueError(
                    f"baseline {baseline_json} holds {len(baseline)} "
                    f"completions but this run uses {len(prompt_set)} "
                    "prompts; set KIMI_K3_SPEC_NUM_PROMPTS to the value "
                    "used for the baseline run"
                )
        else:
            llm = _build_llm(ckpt, tp, "off", adp)
            baseline = _generate(llm, prompt_texts, max_tokens, want_logprobs)
            llm.shutdown()

    llm = _build_llm(ckpt, tp, spec_mode, adp)
    completions = _generate(llm, prompt_texts, max_tokens, want_logprobs)
    llm.shutdown()
    if output_json:
        _dump_completions(output_json, completions, want_logprobs)

    failures = []
    for comp, (prompt, expected) in zip(completions, prompt_set):
        text = comp.text
        print("=" * 80)
        print(f"PROMPT:   {prompt!r}")
        print(f"OUTPUT:   {text!r}")
        if not truncated and expected is not None and expected not in text:
            failures.append(f"expected {expected!r} in output of {prompt!r}")
    if spec_parity == "text":
        for base, spec, (prompt, _) in zip(baseline, completions, prompt_set):
            if base.text != spec.text:
                failures.append(
                    f"spec-dec parity mismatch for {prompt!r}:\n"
                    f"    baseline: {base.text!r}\n"
                    f"    specdec:  {spec.text!r}"
                )
    elif spec_parity == "logits":
        outcomes = [
            _compare_logits_parity(base, spec, prompt, failures)
            for base, spec, (prompt, _) in zip(baseline, completions, prompt_set)
        ]
        divergences = [o for o in outcomes if o in ("benign", "non_tie")]
        non_ties = outcomes.count("non_tie")
        print(
            f"[sanity] logits-parity summary: {len(outcomes)} prompts, "
            f"{outcomes.count('identical')} identical, "
            f"{len(divergences)} divergences ({non_ties} non-tie), "
            f"{outcomes.count('drift')} drift"
        )
        # Aggregate systemic check: scattered non-tie flips are rounding;
        # a real state bug makes them dominate.
        if non_ties >= 2 and non_ties > 0.25 * max(len(divergences), 1):
            failures.append(
                f"non-tie divergences dominate: {non_ties}/"
                f"{len(divergences)} divergences exceeded the tie bound "
                "(systemic — suspect state bug)"
            )

    if failures:
        print("[sanity] FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    suffix = " (pipeline only, truncated model)" if truncated else ""
    if spec_parity != "0":
        suffix += f" (spec-dec {spec_parity} parity verified)"
    print("[sanity] PASS" + suffix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
