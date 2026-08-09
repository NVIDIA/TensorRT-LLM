# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 two-endpoint parity harness (aggregated vs disaggregated).

Compares a REFERENCE deployment against a CANDIDATE deployment through
their OpenAI-compatible ``/v1/completions`` endpoints and produces a
parity report:

* per-prompt first-token agreement (in disagg the first token is produced
  by the ctx server and handed off — a mismatch points at the
  KV/KDA-state transfer or the first-token handoff),
* longest-common-prefix (LCP) length of the greedy token streams,
* logprob drift along the shared prefix + near-tie classification at the
  first divergence (reuses ``_compare_logits_parity`` from
  ``kimi_k3_sa_harness.py`` — same tolerances, same aggregate
  "non-ties must not dominate" rule),
* optional GSM8K accuracy diff (runs ``lm_eval --model local-completions``
  against both endpoints, the same flow as
  ``examples/disaggregated/slurm/benchmark/submit.py``).

The harness only needs HTTP access to the two endpoints — no GPUs, no
tensorrt_llm runtime — so it runs from a login node against servers
launched elsewhere.

DRY-RUN RECIPE (two aggregated servers; debugs the harness before disagg
transfer works — expect near-perfect parity):

    # server A (reference) and server B (candidate): identical aggregated
    # deployments of the same checkpoint on two ports, e.g.
    trtllm-serve $KIMI_K3_CKPT --backend pytorch --port 8000 \
        --config examples/kimi_k3/eval_extra_llm_options.yaml &
    trtllm-serve $KIMI_K3_CKPT --backend pytorch --port 8001 \
        --config examples/kimi_k3/eval_extra_llm_options.yaml &

    python tests/integration/defs/kimi_k3_disagg_parity.py \
        --reference http://localhost:8000 --candidate http://localhost:8001 \
        --extra-prompts 16 --report-json parity_dryrun.json

REAL RECIPE (aggregated vs disagg proxy):

    # reference: aggregated DEP16 deployment (as in
    # examples/kimi_k3/run_gsm8k_kimi_k3.sbatch, but served)
    trtllm-serve $KIMI_K3_CKPT --backend pytorch --port 8000 ...

    # candidate: ctx + gen workers behind the disagg proxy
    trtllm-serve disaggregated -c disagg_config.yaml   # port 9000

    python tests/integration/defs/kimi_k3_disagg_parity.py \
        --reference http://ref-host:8000 --candidate http://proxy-host:9000 \
        --extra-prompts 48 --gsm8k --gsm8k-limit 200 \
        --report-json parity_disagg.json

Notes:
* Requests are sent sequentially (batch of 1) to both endpoints so batch
  composition cannot perturb the comparison.
* Per-token logprobs are requested via the completions ``logprobs`` field
  (PyTorch backend supports top-k dicts). Endpoints that reject
  ``detokenize=false`` or ``logprobs`` degrade gracefully: the harness
  falls back to decoded-token-string comparison, then to text-only, and
  reports which capability level each endpoint provided. One-engine
  spec-dec samplers (SA) do not emit per-token logprobs — run the parity
  probe with spec dec off first, or accept token-level-only
  parity for SA-on runs.
* GSM8K mode shells out to ``lm_eval`` (must be installed, e.g.
  ``pip install lm-eval[api]``) and diffs the exact_match metrics between
  the two endpoints (default tolerance 0.02).

Self-test (no servers, canned responses — exercises the comparison and
classification logic):

    python tests/integration/defs/kimi_k3_disagg_parity.py --self-test

Exit code 0 = PASS, 1 = FAIL (any parity failure, endpoint error, or
GSM8K delta above tolerance).
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kimi_k3_sa_harness import (  # noqa: E402
    PROMPTS_AND_CHECKS,
    _compare_logits_parity,
    _parity_prompts,
)

# Request-payload variants, tried in order per endpoint until one is
# accepted (capability is cached per endpoint afterwards):
#   ids+logprobs : token_ids (detokenize=false) + per-token logprobs
#   str+logprobs : decoded token strings + per-token logprobs
#   text-only    : plain text completion (weakest comparison)
_VARIANTS = ("ids+logprobs", "str+logprobs", "text-only")


def _http_json(url, payload=None, timeout=300):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _served_model(base_url, timeout):
    """Model id from /v1/models, or None if the endpoint does not expose it.

    The `trtllm-serve disaggregated` proxy only implements the completions
    routes (no /v1/models -> HTTP 404), so a missing model listing must not
    be fatal: the caller falls back to the other endpoint's model name or
    to --model. (Dry runs against two aggregated servers do not hit this:
    both expose /v1/models.)
    """
    try:
        models = _http_json(f"{base_url}/v1/models", timeout=timeout)
        return models["data"][0]["id"]
    except urllib.error.HTTPError as e:
        print(f"[parity] NOTE: {base_url}/v1/models unavailable (HTTP {e.code})")
        return None
    # URLError/OSError: connection-level failures. KeyError/IndexError/
    # TypeError: malformed payload shapes ([], {"data": None},
    # {"data": [None]}, ...). ValueError covers json.JSONDecodeError from
    # non-JSON response bodies.
    except (urllib.error.URLError, OSError, KeyError, IndexError, TypeError, ValueError) as e:
        print(f"[parity] NOTE: {base_url}/v1/models unavailable ({e})")
        return None


class Endpoint:
    """One OpenAI-compatible completions endpoint with capability fallback."""

    def __init__(self, base_url, model, max_tokens, logprobs_k, timeout):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.logprobs_k = logprobs_k
        self.timeout = timeout
        self.variant = None  # discovered on first successful request

    def _payload(self, prompt, variant):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        if variant != "text-only" and self.logprobs_k > 0:
            payload["logprobs"] = self.logprobs_k
        if variant == "ids+logprobs":
            payload["detokenize"] = False
        return payload

    def complete(self, prompt):
        """Return a normalized completion record for one prompt."""
        variants = [self.variant] if self.variant else list(_VARIANTS)
        if self.logprobs_k <= 0:
            variants = ["text-only"]
        last_error = None
        for variant in variants:
            try:
                rsp = _http_json(
                    f"{self.base_url}/v1/completions",
                    self._payload(prompt, variant),
                    timeout=self.timeout,
                )
            except urllib.error.HTTPError as e:
                # 4xx = capability rejection -> try the next variant;
                # anything else is a real endpoint failure.
                if 400 <= e.code < 500 and variant != variants[-1]:
                    last_error = e
                    continue
                raise
            if self.variant is None:
                self.variant = variant
                if variant != _VARIANTS[0]:
                    print(
                        f"[parity] NOTE: {self.base_url} degraded to "
                        f"'{variant}' requests ({last_error})"
                    )
            return _normalize_choice(rsp["choices"][0])
        raise last_error


def _normalize_choice(choice):
    """Normalize a CompletionResponseChoice dict for comparison.

    Returns a namespace with:
      text            str ('' when detokenize=false)
      ids             per-position token identities: ints (token_ids),
                      else decoded token strings, else None
      token_logprobs  chosen-token logprob per position, or None
      top_logprobs    per-position {token_str: logprob} dict list, or None
    """
    lp = choice.get("logprobs") or {}
    tokens = lp.get("tokens") or None
    token_logprobs = lp.get("token_logprobs") or None
    top_logprobs = lp.get("top_logprobs") or None
    ids = choice.get("token_ids") or tokens
    return SimpleNamespace(
        text=choice.get("text", ""),
        ids=ids,
        tokens=tokens,
        token_logprobs=token_logprobs,
        top_logprobs=top_logprobs,
    )


def _position_logprob_dict(chosen_key, chosen_lp, top_map):
    """Build a per-position {token: logprob} dict for _compare_logits_parity.

    ``top_map`` (decoded-token-string keyed, from CompletionLogProbs)
    includes the chosen token itself; when re-keying the chosen entry
    under ``chosen_key`` (a token id), drop exactly one entry carrying
    the chosen logprob so the top-2 near-tie gap is not computed against
    a duplicate of the winner. Alternate entries get synthetic keys —
    only the chosen key is ever looked up.
    """
    d = {}
    if top_map:
        remaining = list(top_map.items())
        for i, (_, v) in enumerate(remaining):
            if v == chosen_lp:
                del remaining[i]
                break
        for j, (_, v) in enumerate(remaining):
            d[("alt", j)] = v
    d[chosen_key] = chosen_lp
    return d


def _to_parity_namespace(comp):
    """Adapt a normalized completion to _compare_logits_parity's shape.

    Returns a namespace with token_ids + per-position logprob dicts, or
    None when the endpoint returned no aligned per-token logprobs.
    """
    if comp.ids is None or comp.token_logprobs is None or len(comp.token_logprobs) != len(comp.ids):
        return None
    logprobs = []
    for i, (tid, lp) in enumerate(zip(comp.ids, comp.token_logprobs)):
        top = comp.top_logprobs[i] if comp.top_logprobs and i < len(comp.top_logprobs) else None
        logprobs.append(_position_logprob_dict(tid, lp, top))
    return SimpleNamespace(token_ids=list(comp.ids), logprobs=logprobs)


def _lcp(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def compare_pair(ref, cand, prompt, failures, lp_tol, tie_tol):
    """Compare one prompt's completions; returns the per-prompt record.

    Classification (reference logprobs available):
      identical / benign (near-tie flip) / non_tie / drift — exactly the
      kimi_k3_sa_harness semantics. Without logprobs, only token-level
      facts are recorded (outcome 'divergence_unclassified').
    """
    ref_ids = list(ref.ids) if ref.ids is not None else None
    cand_ids = list(cand.ids) if cand.ids is not None else None
    record = {"prompt": prompt}
    if ref_ids is None or cand_ids is None:
        # Text-only endpoints: character-level LCP is the best available.
        match = ref.text == cand.text
        lcp = _lcp(ref.text, cand.text)
        record.update(
            comparison="text",
            text_match=match,
            char_lcp=lcp,
            outcome="identical" if match else "divergence_unclassified",
        )
        if not match:
            print(
                f"[parity] {prompt!r}: text mismatch at char {lcp} "
                "(token/logprob detail unavailable)"
            )
        return record

    lcp = _lcp(ref_ids, cand_ids)
    full = lcp == len(ref_ids) == len(cand_ids)
    record.update(
        comparison="tokens",
        first_token_match=lcp > 0,
        lcp=lcp,
        ref_len=len(ref_ids),
        cand_len=len(cand_ids),
        full_match=full,
    )

    ref_ns = _to_parity_namespace(ref)
    cand_ns = _to_parity_namespace(cand)
    if ref_ns is not None and cand_ns is not None:
        outcome = _compare_logits_parity(
            ref_ns, cand_ns, prompt, failures, tol=lp_tol, tie_tol=tie_tol
        )
        # Mean |delta| of chosen-token logprobs over the shared prefix —
        # a drift trend indicator below the hard tolerance.
        deltas = [abs(ref.token_logprobs[i] - cand.token_logprobs[i]) for i in range(lcp)]
        if deltas:
            record["mean_abs_lp_delta"] = sum(deltas) / len(deltas)
            record["max_abs_lp_delta"] = max(deltas)
    elif full:
        outcome = "identical"
    else:
        outcome = "divergence_unclassified"
        print(
            f"[parity] {prompt!r}: token divergence at position {lcp} "
            "(logprobs unavailable — cannot classify near-tie vs real)"
        )
    record["outcome"] = outcome

    if lcp == 0:
        # In disagg the first token comes from the ctx server via the
        # handoff; a confident first-token flip is the KV/KDA-transfer
        # bug signature. Only a certified near-tie is excusable.
        if outcome == "benign":
            print(
                f"[parity] WARNING: first-token near-tie flip for "
                f"{prompt!r} (benign, but watch the aggregate)"
            )
        else:
            failures.append(
                f"first-token mismatch for {prompt!r} "
                f"(outcome={outcome}; disagg first-token handoff or "
                "KV/state transfer suspect)"
            )
    return record


def _aggregate(outcomes, failures):
    """Apply the kimi_k3_sa_harness systemic check.

    Scattered non-tie flips are rounding; a real transfer/state bug makes
    them dominate.
    """
    divergences = [o for o in outcomes if o in ("benign", "non_tie")]
    non_ties = outcomes.count("non_tie")
    unclassified = outcomes.count("divergence_unclassified")
    print(
        f"[parity] summary: {len(outcomes)} prompts, "
        f"{outcomes.count('identical')} identical, "
        f"{len(divergences)} classified divergences ({non_ties} non-tie), "
        f"{unclassified} unclassified, {outcomes.count('drift')} drift"
    )
    if non_ties >= 2 and non_ties > 0.25 * max(len(divergences), 1):
        failures.append(
            f"non-tie divergences dominate: {non_ties}/{len(divergences)} "
            "divergences exceeded the tie bound (systemic — suspect "
            "transfer/state bug)"
        )


def run_gsm8k(base_url, model, out_dir, limit, concurrency, timeout, trust_remote_code=False):
    """Run lm_eval GSM8K against one endpoint; returns {metric: value}.

    Mirrors the accuracy flow of examples/disaggregated/slurm/benchmark/
    submit.py (local-completions against /v1/completions). ``model`` must
    be tokenizer-resolvable for lm_eval (a local checkpoint dir or HF repo
    id) — the bare served-model name from /v1/models usually is not; pass
    --tokenizer in that case.
    """
    os.makedirs(out_dir, exist_ok=True)
    model_args = (
        f"model={model},base_url={base_url}/v1/completions,"
        f"num_concurrent={concurrency},max_retries=3,"
        f"tokenized_requests=false,timeout={timeout},"
        "max_gen_toks=256,max_length=4096"
    )
    if trust_remote_code:
        # e.g. Kimi K3's tiktoken-based tokenizer ships as checkpoint code
        model_args += ",trust_remote_code=true"
    cmd = [
        "lm_eval",
        "--model",
        "local-completions",
        "--tasks",
        "gsm8k",
        "--model_args",
        model_args,
        "--log_samples",
        "--output_path",
        out_dir,
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    print(f"[parity] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    results = sorted(
        glob.glob(os.path.join(out_dir, "**", "results*.json"), recursive=True),
        key=os.path.getmtime,
    )
    if not results:
        raise RuntimeError(f"lm_eval produced no results json in {out_dir}")
    with open(results[-1]) as f:
        gsm8k = json.load(f)["results"]["gsm8k"]
    return {k: v for k, v in gsm8k.items() if k.startswith("exact_match") and "stderr" not in k}


def _diff_gsm8k(ref_scores, cand_scores, tol, failures):
    report = {}
    for metric, ref_val in sorted(ref_scores.items()):
        cand_val = cand_scores.get(metric)
        delta = None if cand_val is None else cand_val - ref_val
        report[metric] = {"reference": ref_val, "candidate": cand_val, "delta": delta}
        print(
            f"[parity] gsm8k {metric}: reference={ref_val:.4f} "
            f"candidate={cand_val:.4f} delta={delta:+.4f}"
            if cand_val is not None
            else f"[parity] gsm8k {metric}: candidate missing"
        )
        if delta is None or abs(delta) > tol:
            failures.append(
                f"gsm8k {metric} delta {delta} exceeds tolerance {tol} "
                f"(reference {ref_val} vs candidate {cand_val})"
            )
    return report


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--reference",
        help="base URL of the reference deployment (e.g. aggregated: http://host:8000)",
    )
    parser.add_argument(
        "--candidate",
        help="base URL of the candidate deployment (e.g. disagg proxy: http://host:9000)",
    )
    parser.add_argument(
        "--model", default=None, help="served model name (default: query /v1/models)"
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="tokenizer path/repo for lm_eval GSM8K (a local checkpoint "
        "dir; default: the --model value, which must then be "
        "tokenizer-resolvable)",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--extra-prompts",
        type=int,
        default=16,
        help="extra deterministic parity prompts on top of "
        "the builtin set (kimi_k3_sa_harness pool)",
    )
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="newline-separated prompt file replacing the builtin prompt set",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=5,
        help="top-k logprobs to request per token (0 disables logprob-level parity)",
    )
    parser.add_argument(
        "--lp-tol",
        type=float,
        default=float(os.environ.get("KIMI_K3_SPEC_LP_TOL", "1.0")),
        help="shared-prefix logprob drift tolerance",
    )
    parser.add_argument(
        "--tie-tol",
        type=float,
        default=float(os.environ.get("KIMI_K3_SPEC_TIE_TOL", "0.3")),
        help="near-tie top-2 logprob gap bound at divergence",
    )
    parser.add_argument(
        "--tie-strict", action="store_true", help="fail (not warn) on any single non-tie divergence"
    )
    parser.add_argument("--timeout", type=int, default=600, help="per-request timeout in seconds")
    parser.add_argument(
        "--gsm8k",
        action="store_true",
        help="also run lm_eval GSM8K against both endpoints and diff scores",
    )
    parser.add_argument(
        "--gsm8k-only", action="store_true", help="skip the token/logprob probe; GSM8K diff only"
    )
    parser.add_argument(
        "--gsm8k-limit", type=int, default=None, help="lm_eval --limit (default: full GSM8K)"
    )
    parser.add_argument("--gsm8k-concurrency", type=int, default=8)
    parser.add_argument(
        "--gsm8k-trust-remote-code",
        action="store_true",
        help="pass trust_remote_code=true to lm_eval (checkpoints whose "
        "tokenizer ships as remote code, e.g. Kimi K3)",
    )
    parser.add_argument(
        "--gsm8k-tol", type=float, default=0.02, help="max tolerated |accuracy delta|"
    )
    parser.add_argument(
        "--work-dir", default=None, help="directory for lm_eval outputs (default: temp dir)"
    )
    parser.add_argument(
        "--report-json", default=None, help="write the full parity report to this path"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the built-in comparison-logic self-test (no servers needed)",
    )
    args = parser.parse_args(argv)
    if not args.self_test and not (args.reference and args.candidate):
        parser.error("--reference and --candidate are required (unless --self-test)")
    return args


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.self_test:
        return _self_test()
    if args.tie_strict:
        os.environ["KIMI_K3_SPEC_TIE_STRICT"] = "1"

    failures = []
    report = {"reference": args.reference, "candidate": args.candidate}

    if args.model:
        model = args.model
    else:
        model = _served_model(args.reference, args.timeout)
        cand_model = _served_model(args.candidate, args.timeout)
        # Either side may lack /v1/models (e.g. the disagg proxy); fall
        # back to the side that reports one.
        model = model if model is not None else cand_model
        if model is None:
            print("[parity] ERROR: neither endpoint exposes /v1/models; pass --model explicitly")
            return 1
        if cand_model is not None and cand_model != model:
            print(
                f"[parity] NOTE: endpoints serve different model names "
                f"({model!r} vs {cand_model!r}); sending {model!r} to both, "
                "pass --model to override"
            )
    report["model"] = model

    if not args.gsm8k_only:
        if args.prompts_file:
            with open(args.prompts_file) as f:
                prompts = [line.rstrip("\n") for line in f if line.strip()]
        else:
            prompts = [p for p, _ in PROMPTS_AND_CHECKS] + [
                p for p, _ in _parity_prompts(args.extra_prompts)
            ]
        print(
            f"[parity] probing {len(prompts)} prompts x "
            f"{args.max_tokens} max_tokens, logprobs={args.logprobs}"
        )

        endpoints = {}
        for role, url in (("reference", args.reference), ("candidate", args.candidate)):
            endpoints[role] = Endpoint(url, model, args.max_tokens, args.logprobs, args.timeout)

        records, outcomes = [], []
        for prompt in prompts:
            try:
                ref = endpoints["reference"].complete(prompt)
                cand = endpoints["candidate"].complete(prompt)
            except (urllib.error.URLError, OSError, KeyError) as e:
                failures.append(f"endpoint error for {prompt!r}: {e}")
                records.append({"prompt": prompt, "error": str(e)})
                continue
            record = compare_pair(ref, cand, prompt, failures, args.lp_tol, args.tie_tol)
            records.append(record)
            outcomes.append(record["outcome"])
        _aggregate(outcomes, failures)
        report["prompts"] = records
        report["capability"] = {role: ep.variant for role, ep in endpoints.items()}

    if args.gsm8k or args.gsm8k_only:
        work_dir = args.work_dir or tempfile.mkdtemp(prefix="k3-parity-")
        scores = {}
        for role, url in (("reference", args.reference), ("candidate", args.candidate)):
            scores[role] = run_gsm8k(
                url,
                args.tokenizer or model,
                os.path.join(work_dir, f"gsm8k_{role}"),
                args.gsm8k_limit,
                args.gsm8k_concurrency,
                args.timeout,
                args.gsm8k_trust_remote_code,
            )
        report["gsm8k"] = _diff_gsm8k(
            scores["reference"], scores["candidate"], args.gsm8k_tol, failures
        )

    if args.report_json:
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[parity] report written to {args.report_json}")

    if failures:
        print("[parity] FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[parity] PASS")
    return 0


# ---------------------------------------------------------------------------
# Self-test: canned completions through the real comparison pipeline.
# ---------------------------------------------------------------------------


def _fake(ids, lps, top=None, text=""):
    return SimpleNamespace(text=text, ids=ids, tokens=None, token_logprobs=lps, top_logprobs=top)


def _self_test() -> int:
    lp_tol, tie_tol = 1.0, 0.3
    checks = []

    def check(name, cond):
        checks.append((name, cond))
        print(f"[self-test] {'PASS' if cond else 'FAIL'}: {name}")

    top_confident = [{"a": -0.05, "b": -3.5}] * 4
    top_tie = [{"a": -0.60, "b": -0.72}] * 4

    # 1. identical streams + logprobs -> identical, no failures.
    f = []
    r = compare_pair(
        _fake([1, 2, 3], [-0.1, -0.2, -0.3], top_confident),
        _fake([1, 2, 3], [-0.1, -0.2, -0.3], top_confident),
        "identical",
        f,
        lp_tol,
        tie_tol,
    )
    check(
        "identical outcome",
        r["outcome"] == "identical" and not f and r["full_match"] and r["lcp"] == 3,
    )

    # 2. near-tie divergence mid-stream -> benign, no failures.
    f = []
    r = compare_pair(
        _fake([1, 2, 3], [-0.1, -0.2, -0.6], top_tie),
        _fake([1, 2, 4], [-0.1, -0.2, -0.7], top_tie),
        "near-tie",
        f,
        lp_tol,
        tie_tol,
    )
    check(
        "near-tie -> benign",
        r["outcome"] == "benign" and not f and r["lcp"] == 2 and r["first_token_match"],
    )

    # 3. logprob drift on the shared prefix -> drift + failure.
    f = []
    r = compare_pair(
        _fake([1, 2, 3], [-0.1, -0.2, -0.3], top_confident),
        _fake([1, 2, 3], [-0.1, -5.2, -0.3], top_confident),
        "drift",
        f,
        lp_tol,
        tie_tol,
    )
    check("drift fails", r["outcome"] == "drift" and len(f) == 1)

    # 4. confident (non-tie) divergence mid-stream -> warning only.
    f = []
    r = compare_pair(
        _fake([1, 2, 3], [-0.1, -0.2, -0.05], top_confident),
        _fake([1, 2, 4], [-0.1, -0.2, -3.5], top_confident),
        "non-tie",
        f,
        lp_tol,
        tie_tol,
    )
    check("single non-tie warns", r["outcome"] == "non_tie" and not f)

    # 5. confident first-token mismatch -> hard failure.
    f = []
    r = compare_pair(
        _fake([1, 2], [-0.05, -0.2], top_confident),
        _fake([9, 2], [-3.5, -0.2], top_confident),
        "first-token",
        f,
        lp_tol,
        tie_tol,
    )
    check("first-token mismatch fails", r["lcp"] == 0 and len(f) >= 1)

    # 6. no logprobs available -> unclassified divergence, first-token
    #    mismatch still fails (cannot be excused without logprobs).
    f = []
    r = compare_pair(_fake([5, 6], None), _fake([7, 6], None), "no-logprobs", f, lp_tol, tie_tol)
    check("logprob-less mismatch fails", r["outcome"] == "divergence_unclassified" and len(f) == 1)

    # 7. text-only endpoints -> character comparison.
    f = []
    r = compare_pair(
        _fake(None, None, text="hello world"),
        _fake(None, None, text="hello there"),
        "text-only",
        f,
        lp_tol,
        tie_tol,
    )
    check(
        "text-only unclassified",
        r["comparison"] == "text" and r["outcome"] == "divergence_unclassified",
    )

    # 8. aggregate rule: dominating non-ties -> systemic failure.
    f = []
    _aggregate(["non_tie", "non_tie", "benign"], f)
    check("dominating non-ties fail", len(f) == 1)
    f = []
    _aggregate(["benign"] * 8 + ["non_tie", "identical"], f)
    check("scattered non-tie passes", not f)

    # 9. chosen-token duplicate is dropped from the near-tie gap.
    d = _position_logprob_dict(42, -0.05, {"a": -0.05, "b": -3.5})
    top = sorted(d.values(), reverse=True)
    check(
        "top-2 gap excludes chosen duplicate", len(d) == 2 and abs((top[0] - top[1]) - 3.45) < 1e-9
    )

    # 10. gsm8k diff: within-tolerance passes, above-tolerance fails.
    f = []
    _diff_gsm8k({"exact_match,strict-match": 0.90}, {"exact_match,strict-match": 0.89}, 0.02, f)
    ok_within = not f
    f = []
    _diff_gsm8k({"exact_match,strict-match": 0.90}, {"exact_match,strict-match": 0.80}, 0.02, f)
    check("gsm8k tolerance", ok_within and len(f) == 1)

    # 11. _served_model returns None for malformed /v1/models payloads,
    #     invalid JSON, and connection failures; extracts the id otherwise.
    global _http_json
    orig_http_json = _http_json
    bad_responses = [
        [],
        {"data": None},
        {"data": [None]},
        {"data": [{}]},
        json.JSONDecodeError("Expecting value", "not-json", 0),
        urllib.error.URLError("connection refused"),
    ]
    try:
        results = []
        for rsp in bad_responses:

            def _canned(url, payload=None, timeout=0, _rsp=rsp):
                if isinstance(_rsp, Exception):
                    raise _rsp
                return _rsp

            _http_json = _canned
            results.append(_served_model("http://stub", timeout=1))

        def _good(url, payload=None, timeout=0):
            return {"data": [{"id": "the-model"}]}

        _http_json = _good
        served = _served_model("http://stub", timeout=1)
    finally:
        _http_json = orig_http_json
    check("_served_model malformed payloads -> None", all(r is None for r in results))
    check("_served_model well-formed payload", served == "the-model")

    failed = [name for name, cond in checks if not cond]
    if failed:
        print(f"[self-test] FAIL ({len(failed)}/{len(checks)}): {failed}")
        return 1
    print(f"[self-test] PASS ({len(checks)} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
