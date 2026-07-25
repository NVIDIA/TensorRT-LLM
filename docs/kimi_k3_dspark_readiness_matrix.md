# Kimi K3 DSpark weights-drop-day experiment matrix

Tooling: `examples/kimi_k3/measure_dspark_acceptance.py` +
`examples/kimi_k3/run_dspark_acceptance.sbatch` (spec-off / dspark-on A/B in
one submission) + the opt-in accept-site recorder
(`tensorrt_llm/_torch/speculative/accept_stats.py`,
`TLLM_DFLASH_ACCEPT_STATS_DIR`). Dry-run validated against the dummy
checkpoint (`dummy-dspark0724`, random weights): AL ~= 1.0, AR ~= 0 at every
position — plumbing only.

When Hao Guo's team delivers the trained drafter, fire these in order.
Deployment matches the certified SA regime: DEP16 (attention-DP + MoE EP16),
eager (`cuda_graph_config: null` — the K3 CUDA-graphs verify/accept regime is
uncertified, and the recorder skips graph batches), `max_batch_size` 8,
GSM8K prompts, greedy (`temperature=0`).

## R1 — Acceptance + speedup A/B (eager, the headline run)

```
export REPO=<worktree>; export EXTRA_MOUNTS="$MAIN:$MAIN:rw"
sbatch --export=ALL --output=<lustre>/dspark-accept-%j.log \
    examples/kimi_k3/run_dspark_acceptance.sbatch \
    --model <k3-weights> --drafter <trained-drafter> --image <sqsh> \
    --outdir <lustre>/dspark-r1 --num-prompts 128 --max-tokens 256
```

Produces: AL (per-request + accept-site aggregate), per-position AR curve,
acceptance histogram, spec-off vs dspark TPOT proxy, E2E speedup, and —
when the DSpark confidence MR (!75) plus its provider hookup are in the
tree, the drafter ships a confidence head, and no threshold is set —
unbiased confidence-calibration counts. Without them,
`confidence_calibration` is null and the rest of R1 is unaffected.

Accept: AL >= 2.5 and AR(position 1) >= 0.6 (K2.7-Code-DFlash precedent),
E2E speedup > 1.3x at batch 8. Reject/escalate: AL < 1.5 (suspect
checkpoint/convention mismatch — check shift_label and target_layer_ids
against the drafter config before blaming training).

## R2 — SWA-convention A/B (tali's open question)

Drafter layers window `(w-1, w-1)` (as-shipped `config.json`, `use_swa:
true, sliding_window: 1024`) vs full-attention drafter layers
(`config.json.bak_fullattn` twin on the dummy; ask training which convention
the real checkpoint was trained with and build the twin the same way).
Run R1's dspark leg twice (`--skip-baseline` for the second), one per
drafter-config variant, same prompts.

Accept: conventions differ by < 0.05 AL → keep as-shipped config. If the
full-attn variant wins by more, the SWA plumbing disagrees with training —
file against the drafter-forward semantics (!74), do not ship windowed.

## R3 — GSM8K accuracy gate (unchanged eval path)

```
sbatch --export=ALL examples/kimi_k3/run_gsm8k_kimi_k3.sbatch \
    --model <k3-weights> --image <sqsh> --dflash <trained-drafter>
```

Accept: score within noise of the no-spec baseline (>= 96.4 reference;
greedy spec decode must not change outputs materially). Reject: any drop
> 1 point → acceptance bug, not a perf problem; stop before perf tuning.

## R4 — Confidence-threshold sweep (after R1 calibration; needs !75)

From R1's `confidence_calibration` table pick 3 candidate thresholds
(empirical-acceptance bins bracketing ~0.5–0.8), then per candidate:

```
sbatch ... run_dspark_acceptance.sbatch --skip-baseline \
    --confidence-threshold <t> [--confidence-policy cumulative] ...
```

Accept: some t where TPOT improves over the R1 dspark leg with AL drop
< 0.2. Check per-position ECE first: if ECE >> 0.1 the head needs !75's
Sequential Temperature Scaling before the threshold is meaningful —
calibrate STS from R1's (confidence, accepted) counts, fold the fitted
temperature into the checkpoint (or rerun R1 to re-collect), then sweep.

## R5 (stretch) — batch scaling

R1's A/B at `--max-batch-size` 1 and 8 (and 16 if memory allows) to bound
where drafting stops paying. Informs the serving default.

## Notes

- Calibration runs MUST leave `--confidence-threshold` unset: with the
  trim active, trimmed positions are forced rejects and bias the pairs.
- TPOT-reference legs run with the recorder off (the sbatch does this);
  the recorder host-syncs a few scalars per verify step.
- Recorder files are per rank; attention-DP ranks hold disjoint requests,
  so the harness merges all rank files (`merge_snapshots`).
- Queue reality: GB300 batch-partition jobs can pend for hours; submit R1
  and R3 together, R2/R4 after R1 lands.
