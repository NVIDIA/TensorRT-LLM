# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Kimi K3 DSpark acceptance / speedup readiness harness.

Drives generation over a GSM8K prompt subset with the DSpark (DFlash)
drafter and reports the weights-drop-day figures of merit:

- AL: mean accepted tokens per target verify step (bonus token included),
  per request (from RequestOutput.avg_decoded_tokens_per_iter) and
  aggregate (from the runtime accept-site histogram),
- per-position acceptance-rate curve (position 1..max_draft_len),
- accepted-draft-count histogram,
- TPOT / decode-throughput numbers for the no-spec A/B (run once with
  --spec-off, once with --drafter; see run_dspark_acceptance.sbatch which
  fires both in one submission),
- confidence-vs-acceptance calibration table, for calibrating the DSpark
  confidence_threshold and Sequential Temperature Scaling. Rows are only
  collected when the confidence provider
  (``dspark_confidence``) is in the tree and the drafter ships a
  confidence head; otherwise "confidence_calibration" is null. For
  UNBIASED calibration leave --confidence-threshold unset (0): trimmed
  positions are forced rejects.

The per-position / calibration counters come from the opt-in recorder in
tensorrt_llm/_torch/speculative/accept_stats.py, enabled here by setting
TLLM_DFLASH_ACCEPT_STATS_DIR before the LLM is built (attention-DP ranks
hold disjoint request sets, so all per-rank files are merged). The
recorder syncs a few scalars per step: keep it OFF (--no-accept-stats)
for the TPOT-reference legs of an A/B.

The recorder accumulates every eager DFlash step, including the warmup
batch, and can't be reset across the launcher process boundary; the
harness instead snapshots the post-warmup counts and subtracts them from
the final totals so the reported figures cover only the timed run. That
subtraction needs the warmup steps flushed to disk before timing, so the
stats leg forces TLLM_DFLASH_ACCEPT_STATS_FLUSH_EVERY=1 (export it before
trtllm-llmapi-launch for TP>1, as run_dspark_acceptance.sbatch does).

Example (inside the container, see run_dspark_acceptance.sbatch). The
recorder dir MUST be exported before trtllm-llmapi-launch: it pre-spawns the
MPI worker ranks (where DFlashWorker lives), so setting os.environ inside this
script never reaches them and the accept-stats come back empty for TP>1:
    export TLLM_DFLASH_ACCEPT_STATS_DIR=/tmp/dspark-stats
    trtllm-llmapi-launch python3 examples/kimi_k3/measure_dspark_acceptance.py \
        --model /path/to/kimi-k3 --drafter /path/to/dspark-drafter \
        --tp-size 16 --num-prompts 64 --max-tokens 256 \
        --stats-dir /tmp/dspark-stats --output-json results_spec.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorrt_llm import LLM


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the Kimi K3 target checkpoint")
    parser.add_argument(
        "--drafter",
        default=None,
        help="Path to the DSpark/DFlash drafter checkpoint. "
        "Omit (or pass --spec-off) for the no-spec reference.",
    )
    parser.add_argument(
        "--spec-off", action="store_true", help="Force speculation off (no-spec TPOT reference)."
    )
    parser.add_argument("--tp-size", type=int, default=16)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--max-num-tokens", type=int, default=4096)
    parser.add_argument("--kv-frac", type=float, default=0.20)
    parser.add_argument(
        "--max-draft-len",
        type=int,
        default=7,
        help="K = drafter block_size - 1 (K3 dspark: 8-1=7).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Enable confidence-scheduled verification (leave unset for calibration runs).",
    )
    parser.add_argument(
        "--confidence-policy", default="first_below", choices=["first_below", "cumulative"]
    )
    parser.add_argument(
        "--num-prompts", type=int, default=64, help="GSM8K test-split prompts to run."
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Optional JSONL with a 'prompt' field per line; overrides the GSM8K subset.",
    )
    parser.add_argument(
        "--stats-dir",
        default=None,
        help="Directory for the runtime accept-stats JSONs (default: alongside --output-json).",
    )
    parser.add_argument(
        "--no-accept-stats",
        action="store_true",
        help="Do not enable the runtime recorder (use for "
        "TPOT-reference legs; per-position stats are lost).",
    )
    parser.add_argument("--output-json", default="dspark_acceptance.json")
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_file:
        prompts = []
        with open(args.prompt_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(json.loads(line)["prompt"])
        return prompts[: args.num_prompts]
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [
        f"Question: {row['question']}\nAnswer:"
        for row in ds.select(range(min(args.num_prompts, len(ds))))
    ]


def build_llm(args: argparse.Namespace) -> tuple[LLM, bool]:
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import KvCacheConfig

    spec_on = args.drafter is not None and not args.spec_off
    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        enable_attention_dp=True,
        moe_expert_parallel_size=args.tp_size,
        trust_remote_code=True,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        max_num_tokens=args.max_num_tokens,
        # Mirror eval_extra_llm_options_dflash.yaml: eager (K3 CUDA-graphs
        # verify/accept regime not certified; the recorder also skips
        # graph batches), no overlap scheduler, no chunked prefill.
        cuda_graph_config=None,
        disable_overlap_scheduler=True,
        enable_chunked_prefill=False,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            free_gpu_memory_fraction=args.kv_frac,
            tokens_per_block=64,
        ),
    )
    if spec_on:
        from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig

        spec_kwargs = dict(
            max_draft_len=args.max_draft_len,
            speculative_model=args.drafter,
        )
        if args.confidence_threshold is not None:
            # Shipped by the DSpark confidence-scheduled verification MR;
            # feature-detect so this harness also runs on trees without it
            # (where only threshold-off calibration-free measurement works).
            if "confidence_threshold" not in DFlashDecodingConfig.model_fields:
                raise SystemExit(
                    "--confidence-threshold requires the DSpark "
                    "confidence-scheduled verification support in the tree "
                    "(DFlashDecodingConfig has no confidence_threshold "
                    "field). Leave it unset for AL/AR measurement."
                )
            spec_kwargs["confidence_threshold"] = args.confidence_threshold
            spec_kwargs["confidence_policy"] = args.confidence_policy
        llm_kwargs["speculative_config"] = DFlashDecodingConfig(**spec_kwargs)
    return LLM(**llm_kwargs), spec_on


def _remove_warmup(merged: dict, warmup: dict) -> None:
    """Subtract warmup-phase counts from the merged accept-stats, in place.

    ``merged`` and ``warmup`` are both outputs of ``merge_snapshots``; after
    this the reported histogram, AL/AR, and calibration reflect only the timed
    generation. Only the fields the harness reports (the accepted-draft
    histogram and the confidence-calibration counts) are adjusted. Counts are
    monotonic, so the differences are non-negative.
    """
    hist = merged["accepted_draft_hist"]
    for i, n in enumerate(warmup["accepted_draft_hist"]):
        hist[i] -= n
    merged["num_steps"] -= warmup["num_steps"]
    mcc, wcc = merged["confidence_calibration"], warmup["confidence_calibration"]
    for field in ("attempts", "accepted"):
        for k, row in enumerate(wcc[field]):
            for b, n in enumerate(row):
                mcc[field][k][b] -= n


def main() -> None:
    args = parse_arguments()

    spec_on = args.drafter is not None and not args.spec_off
    stats_dir = None
    if spec_on and not args.no_accept_stats:
        stats_dir = args.stats_dir or (
            os.path.splitext(os.path.abspath(args.output_json))[0] + ".stats"
        )
        # Must be set before the LLM (and its worker processes) is built.
        os.environ["TLLM_DFLASH_ACCEPT_STATS_DIR"] = stats_dir
        # Flush after every verify step so the short warmup batch (a handful
        # of steps, well under the default flush period) is on disk before the
        # timed run starts — the harness reads it back as a baseline and
        # subtracts it below. This env, like the stats dir, only reaches
        # pre-spawned MPI workers when exported before trtllm-llmapi-launch
        # (run_dspark_acceptance.sbatch does this for the TP>1 dspark leg);
        # the in-script set covers the single-process case. The per-step flush
        # writes a tiny JSON and only runs on the stats leg, which is a
        # measurement run, not a TPOT reference (--no-accept-stats).
        os.environ.setdefault("TLLM_DFLASH_ACCEPT_STATS_FLUSH_EVERY", "1")

    prompts = load_prompts(args)
    llm, spec_on = build_llm(args)

    from tensorrt_llm import SamplingParams

    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    result = {
        "mode": "dspark" if spec_on else "spec_off",
        "model": args.model,
        "drafter": args.drafter if spec_on else None,
        "confidence_threshold": args.confidence_threshold if spec_on else None,
        "confidence_policy": args.confidence_policy if spec_on else None,
        "num_prompts": len(prompts),
        "max_tokens": args.max_tokens,
        "max_batch_size": args.max_batch_size,
    }

    try:
        # Warmup (excluded from timing): one short batch to pay JIT and
        # allocator costs so the timed section measures steady state.
        llm.generate(prompts[: min(2, len(prompts))], SamplingParams(max_tokens=8, temperature=0.0))

        # The accept-stats recorder lives in the worker ranks and accumulates
        # every eager DFlash step, warmup included — so the warmup batch would
        # otherwise bias the aggregate AL/AR, histogram, and calibration. We
        # can't reset the recorder across the launcher process boundary, so
        # snapshot its post-warmup counts here and subtract them from the final
        # totals below (see _remove_warmup). Relies on the per-step flush set
        # above so warmup is already on disk.
        warmup_stats = None
        if stats_dir and os.path.isdir(stats_dir):
            from tensorrt_llm._torch.speculative.accept_stats import (
                load_rank_snapshots,
                merge_snapshots,
            )

            base_snaps = load_rank_snapshots(stats_dir)
            if base_snaps:
                warmup_stats = merge_snapshots(base_snaps)
            else:
                print(
                    "WARNING: could not snapshot warmup accept-stats "
                    "(no rank files yet); aggregate AL/AR will include the "
                    "warmup batch. Export TLLM_DFLASH_ACCEPT_STATS_FLUSH_EVERY=1 "
                    "before trtllm-llmapi-launch for multi-rank runs."
                )

        t0 = time.monotonic()
        outputs = llm.generate(prompts, sampling_params)
        wall_s = time.monotonic() - t0

        total_out_tokens = 0
        per_request_al = []
        for out in outputs:
            n = len(out.outputs[0].token_ids)
            total_out_tokens += n
            al = getattr(out, "avg_decoded_tokens_per_iter", None)
            if al is not None:
                per_request_al.append(float(al))

        result.update(
            wall_s=wall_s,
            total_output_tokens=total_out_tokens,
            output_tokens_per_s=total_out_tokens / wall_s if wall_s else None,
            # Batched-decode proxy for TPOT: seconds per generated token.
            # A/B speedup = spec_off.tpot_proxy_ms / dspark.tpot_proxy_ms
            # at identical prompt set / batch size / max_tokens.
            tpot_proxy_ms=(wall_s / total_out_tokens * 1e3) if total_out_tokens else None,
            per_request_al={
                "mean": (sum(per_request_al) / len(per_request_al)) if per_request_al else None,
                "min": min(per_request_al) if per_request_al else None,
                "max": max(per_request_al) if per_request_al else None,
                "n": len(per_request_al),
            },
        )
    finally:
        llm.shutdown()

    if stats_dir:
        from tensorrt_llm._torch.speculative.accept_stats import (
            calibration_table,
            load_rank_snapshots,
            merge_snapshots,
            summarize_hist,
        )

        snaps = load_rank_snapshots(stats_dir) if os.path.isdir(stats_dir) else []
        if not snaps:
            # Env changes made here don't reach MPI worker ranks that
            # trtllm-llmapi-launch pre-spawned; multi-rank runs must export
            # TLLM_DFLASH_ACCEPT_STATS_DIR in the launching shell (the
            # sbatch runner does this for the dspark leg).
            print(
                f"WARNING: accept-stats requested but no "
                f"dflash_accept_stats_rank*.json found in {stats_dir}; "
                f"per-position AR / calibration unavailable. Export "
                f"TLLM_DFLASH_ACCEPT_STATS_DIR before trtllm-llmapi-launch "
                f"for multi-rank runs."
            )
        else:
            merged = merge_snapshots(snaps)
            if warmup_stats is not None:
                _remove_warmup(merged, warmup_stats)
            summary = summarize_hist(merged["accepted_draft_hist"])
            cc = merged["confidence_calibration"]
            has_calib = any(any(row) for row in cc["attempts"])
            result["accept_stats"] = {
                "stats_dir": stats_dir,
                "num_rank_files": len(snaps),
                "accepted_draft_hist": merged["accepted_draft_hist"],
                **summary,
                "confidence_calibration": calibration_table(cc["attempts"], cc["accepted"])
                if has_calib
                else None,
            }

    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 72)
    print(
        f"mode={result['mode']}  prompts={result['num_prompts']}  "
        f"wall={result.get('wall_s', 0):.1f}s  "
        f"out_tokens={result.get('total_output_tokens')}  "
        f"tok/s={result.get('output_tokens_per_s') or 0:.1f}  "
        f"tpot_proxy={result.get('tpot_proxy_ms') or 0:.2f} ms"
    )
    if result.get("per_request_al", {}).get("mean") is not None:
        al = result["per_request_al"]
        print(
            f"AL (per-request avg_decoded_tokens_per_iter): "
            f"mean={al['mean']:.3f} min={al['min']:.3f} max={al['max']:.3f}"
        )
    stats = result.get("accept_stats")
    if stats:
        print(
            f"AL (accept-site, {stats['num_steps']} steps): "
            f"{stats['al']:.3f}   hist={stats['accepted_draft_hist']}"
        )
        curve = "  ".join(f"p{k + 1}={v:.3f}" for k, v in enumerate(stats["ar_per_position"]))
        print(f"AR per position: {curve}")
        if stats["confidence_calibration"]:
            print(
                "confidence calibration: per-position ECE = "
                + "  ".join(
                    f"p{k + 1}={p['ece']:.3f}" if p["ece"] is not None else f"p{k + 1}=n/a"
                    for k, p in enumerate(stats["confidence_calibration"]["per_position"])
                )
            )
    print(f"results written to {os.path.abspath(args.output_json)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
