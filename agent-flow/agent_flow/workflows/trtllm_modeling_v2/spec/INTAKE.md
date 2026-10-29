<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Intake: from a request to `task.yaml`

For the agent that receives the user's request and launches the workflow. Intake decides what
every later hour costs: a spec that runs gates eager, gates members a model does not fit or
multiplies variants will be slow however good the roles are, and `notes` are the only thing every
role reads about the model. Intake writes files and reads the tree; it starts no GPU job.

## 1. Read before writing

- The checkpoint: `config.json` (including any `text_config`, `vision_config`, sparse or
  quantization blocks), the safetensors index, the tokenizer and chat template.
- The reference implementation the user names, enough to write the architecture brief below, and
  whether it can load this checkpoint's weights as stored (its quantization in particular).
- `tensorrt_llm/_torch/_experimental/modeling_v2/README.md` and this workflow's README.
- How other engines (vLLM, SGLang) serve this model, where they do: the kernels, the parallel
  layouts and the deployment they use inform the members, features and notes below.

## 2. Start from a derived spec

```bash
trtllm-modeling-v2 intake derive --checkpoint <ckpt> --repo <trtllm checkout> \
    --goal "<members> <features> kv=bf16/fp8/nvfp4 isl=<n> osl=<n> concurrency=<a>/<b>/<c>" \
    --out <dir>
```

It writes `task.derived.yaml`, `derive.report.md` (every value the goal did not state and where
it came from) and `intake.report.md` (the lint below, with prior art). With a `site.yaml` pass
`--site` instead of `--repo`. Edit the derived spec; copy `task.example.yaml` only for what the
goal words cannot say.

- A goal that names features gets those: a drafter or vision tower the checkpoint carries and the
  request does not ask for is left out (`derive.report.md` says so). Name it to build it.
- `kv=` lists the KV dtypes in one word: the first is every gate's, each other one a variant
  (`gates: [accuracy]`) judged against the unit's own score.
- Derive asks rather than guesses (and stops until the spec answers): `model_key` when
  `config.json` names no model. A tuning goal that names no concurrency tunes at 1/32/128.

## 3. Map the request

| the request says | the spec | why |
|---|---|---|
| members, e.g. "TP4, TEP4, ADP4" | `members: [tp4, tep4, dep4]` (ADP = attention DP + EP = `dep`) | |
| "X needs full validation" | X takes every unit and **one variant per feature axis** at X: the variant's `members: [X]`, `gates: [accuracy]` | a cross product of axes (KV dtype x chunking x reuse ...) multiplies the contract; add a combination only when the request names it |
| "Y only aggregate / text-only" | Y in `members.default` for core alone; its variant and feature cells in `coverage.accept` | |
| "<feature> smoke" | the feature in `features`, `profile: {<feature>: smoke}`, its members narrowed | |
| a KV cache dtype list | base `kv_cache_config.dtype: auto`; one variant per other dtype (`llm_api_options.kv_cache_config.dtype`, `gates: [accuracy]`) | |
| chunked prefill, block reuse | `runtime.chunked_prefill: true`, `kv_cache_reuse: true` in **every** gate; one `chunked` variant (`gates: [accuracy]`) with a `max_num_tokens` below the benchmark's prompts (e.g. 512 for 8-shot GSM8K) so they are really split; `no_reuse` only if asked | features that are on everywhere are proven together, which is how they serve |
| disagg | `accuracy.disagg.test`: the node id of the target's disaggregated-serving accuracy test, written beside the tree's own (the tests that serve through `trtllm-serve disaggregated`), with the environment it reads in `accuracy.disagg.env` (`LLM_MODELS_ROOT` ...); it runs once, at the member the request validates fully, on **one node**: its context and generation workers share that node's devices (two engines of two on a 4-GPU node; `accuracy.disagg.devices` to say otherwise, never more than a node) | boot and accuracy run one aggregated engine and never cross the hop; the tree's launcher (`launch_disaggregated_llm`) starts every worker on one host; the lint's `disagg-placement` says whether two engines fit the node (`environment.gpu_memory_gib` lets it size them) |
| any other feature (MTP, speculative, offload ...) | name it in `--goal`; `derive.report.md` and `task.example.yaml` say what its unit owes | the same one-variant-per-axis rule applies |
| workload isl / osl / concurrency | `max_seq_len = isl + osl`; `max_batch_size` >= the per-rank concurrency (attention-DP ranks split it); `max_num_tokens` = isl (or 8192); `perf.workload` | |
| "perf tune" | `perf.enabled: true`; `perf.workload.concurrency` 1/32/128 unless the goal names it; a `dep` member only the high point x its attention-DP ranks | it starts once the onboarding contract holds and measures one baseline (no stock reference, no repeats); a goal is `ratio_to_baseline` or `pct_of_sol` |

**The smallest contract that covers the request.** Every check loads the model (minutes on a
large checkpoint), and every change to the target runs the stale checks again, so the contract's
size is paid once per fix, not once per run. Prove each requested feature **once**, at the member
the request validates fully, and accept the rest in `coverage.accept`, where the delivery report
states it:

- A storage or scheduling variant (a KV dtype, `chunked`, `no_reuse`) takes `gates: [accuracy]`.
  The accuracy check loads the same engine and fails unless each rank reports the variant's
  setting (its KV dtype, its chunking); a variant that boots, boots at **every** member of its
  unit.
- Members the request scopes to text-only or aggregate take core alone, not the variants.
- A feature unit runs at one member (`members: {<feature>: [X]}`), and every variant stays on
  `unit: core`. Each feature then has a check that exercises it (disagg its test, multimodal
  its boot), and each axis a variant; a variant on a feature unit (a KV dtype under MTP) is a
  combination, added only when the request names it. A variant of disagg owes nothing.
- Leave the boot boundaries off (section 5) and the profile at `standard` unless the request asks.

The lint prints the checks per unit and variant; past ~30 deserves a second look.

**The disagg test.** The run writes it; the flow deploys nothing of its own. Name the node id it
will have, say in `notes` which task writes it (the lint's `disagg-test-pending` notes it until
it exists), and give it what it reads in `accuracy.disagg.env`. A test that skips (too few
devices, a missing model root) fails the gate: set `accuracy.disagg.devices` when it needs more
than two engines at its member's width.

## 4. Runtime

`cuda_graph: true`, `overlap_scheduler: true`, `chunked_prefill: true`, `kv_cache_reuse: true`,
`--kv_cache_fraction` around 0.7. Every accuracy gate runs a full benchmark with CUDA graphs: eager
is 10-50x slower, each fix-and-recheck pays it, and it scores a path the target does not serve.
The lint blocks a unit or variant whose accuracy runs eager, and the same holds for a benchmark
you start by hand.

## 5. Accuracy and boot

- **Protocol.** Score the benchmark (GSM8K by default) as few-shot completion (no
  `--apply_chat_template`) unless the model only works chat-formatted. A chat-template protocol on
  a model that reasons first needs `--max_output_length` >= 1024 and its stop strings, or it scores
  truncation.
- **Anchors.** Each unit's anchor is a row of `reference_yaml` under `accuracy.model_key`, which
  the spec states (`org/model`; the checkpoint path's last two components are no model's name).
  Without a written or measured score, record a **low placeholder** (e.g. 10 on GSM8K, in
  percent): it catches an engine that answers nothing and fails no working one. Commit the row to
  the run's branch: each check measures a snapshot of a commit and reads its anchor table there,
  and preflight refuses an uncommitted one. Replace the row (committed) when a measured anchor is
  known, the user's, the reference implementation's on the target hardware, or one the onboarding
  measures; every accuracy check then runs again.
- **Variant anchors.** A configuration variant that names no row of its own (`discriminators`,
  `reference_yaml`, `model_key`, `benchmark`) is judged against its unit's own score at the same
  member, on the same target (`anchor: relative`): a KV dtype against `auto`, the chunked variant
  against the unchunked engine. Its floor allows for the error of both scores. A variant that
  names a row of its own is judged against that row (its `discriminators` are layered over the
  unit's).
- **Boot prompts.** One checked factual prompt and one checked arithmetic word problem (a
  benchmark-style question with its numeric answer as the keyword): cheap semantic signal at every
  boot.
- **Boot boundaries** (`accuracy.envelope`). Off unless the request asks for them (prompts at
  `max_seq_len`, loaded batches). When on, one boot per unit runs them (`members` picks it), and a
  probe whose legs part on a near-tie (`tie_margin`) passes with a note. Give `probes` with one
  right continuation, e.g.
  `["1, 2, 3, 4, 5, 6, 7,", "Monday, Tuesday, Wednesday,"]`: open-ended prompts have near-tied
  tokens that flip with batch shape (MoE routing, low-precision GEMM tactics) far more often.

## 6. Environment

- `environment.python` exists before launch: the editable install the gates measure, or an
  interpreter the image already has. Every gate's command runs under it.
- `environment.substrate_checkpoint` is required when the stock modeling path does not register the
  checkpoint's architecture (the lint checks): a small checkpoint the stock path supports, on this
  cluster. Preflight's substrate gate boots it instead; skipping the gate also drops its MPI and
  NVRTC checks.
- A `session_spool` environment's spools exist once their sessions run. Until then lint with
  `--sessions-planned`; start the sessions before launch.

## 7. Notes: the run's brief

`notes` reach every role verbatim, in every turn. Keep them a brief (under ~6000 characters) of
what the roles need while they work:

- **Reference** — where it is, what it supports, how it loads this checkpoint's weights.
- **Architecture** — layer pattern, attention kinds with head dims, sparse/indexer configuration,
  MoE (experts, top-k, clamps, router), norm and residual dtypes, towers.
- **Prior art in this tree worth surveying** — the top leads of each kind in `intake.report.md`
  (attention backends, custom ops, other targets' `ops/`, catalog entries), one line each named by
  its `<dir>/<file>.py`: what it offers, or why it does not fit.
- **Engine constraints** — from `intake.report.md` and the design they imply.
- **Scope decisions** — what is gated where, what is checked outside the flow, and why.
- **References outside the tree** — where you know them: another engine's implementation of
  this model (a checkout path, a package, a URL) and the kernels it uses.
- **Embargo** — none, unless the request asks for one (an evaluation that must not see an existing
  implementation of this model): then those exact paths in `source_embargo`. Everything else, in
  the tree and outside it, is prior art; never forbid kernels, backends, other models' targets,
  kernel libraries or other engines' implementations.

Nothing about the intake itself: what intake may or may not do, launch blockers, evidence,
provenance and questions for the user belong in your own summary beside the spec. Notes are read
as standing orders, so "no build is allowed" written for the intake stops the run's own builds.

## 8. Lint, then launch

```bash
trtllm-modeling-v2 intake lint --task task.yaml --report intake.report.md [--sessions-planned]
```

Fix every `block` and every `warn`, or keep one on purpose with a line in `notes`:
`intake-accept: <rule>[, <rule>] -- <why>`. A `block` is what preflight or the first boot would stop
on (a requested cell no check covers, a KV-cache dtype the engine does not load), or an accuracy
check that runs eager, and cannot be accepted. The report also prints the contract size; a contract past ~30 checks
deserves a second look. Then `--dry-run`, then launch. Preflight journals the same lint, so the run's record shows
what was accepted. Once it runs, steer it only through `feedback` (the README's "While it runs"):
cancelling the run's own jobs proves nothing and costs them again.
