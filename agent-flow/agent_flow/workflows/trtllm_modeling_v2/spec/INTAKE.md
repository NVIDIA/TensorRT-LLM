<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Intake: from a request to `task.yaml`

For the agent that receives the user's request and launches the workflow. Intake decides what
every later hour costs: a spec that runs gates eager, gates disaggregation across two nodes or
multiplies variants will be slow however good the roles are, and `notes` are the only thing every
role reads about the model. Intake writes files and reads the tree; it starts no GPU job.

## 1. Read before writing

- The checkpoint: `config.json` (including any `text_config`, `vision_config`, sparse or
  quantization blocks), the safetensors index, the tokenizer and chat template.
- The reference implementation the user names, enough to write the architecture brief below, and
  whether it can load this checkpoint's weights as stored (its quantization in particular).
- `tensorrt_llm/_torch/_experimental/modeling_v2/README.md` and this workflow's README.

## 2. Start from a derived spec

```bash
trtllm-modeling-v2 intake derive --checkpoint <ckpt> --repo <trtllm checkout> \
    --goal "<members> <features> isl=<n> osl=<n> concurrency=<a>/<b>/<c>" --out <dir>
```

It writes `task.derived.yaml`, `derive.report.md` (every value the goal did not state and where
it came from) and `intake.report.md` (the lint below, with prior art). With a `site.yaml` pass
`--site` instead of `--repo`. Edit the derived spec; copy `task.example.yaml` only for what the
goal words cannot say.

## 3. Map the request

| the request says | the spec | why |
|---|---|---|
| members, e.g. "TP4, TEP4, ADP4" | `members: [tp4, tep4, dep4]` (ADP = attention DP + EP = `dep`) | |
| "X needs full validation" | X takes every unit and **one variant per feature axis** at X: the variant's `members: [X]`, `gates: [accuracy]` | a cross product of axes (KV dtype x chunking x reuse ...) multiplies the contract; add a combination only when the request names it |
| "Y only aggregate / text-only" | Y in `members.default` for core alone; its variant and feature cells in `coverage.accept` | |
| "<feature> smoke" | the feature in `features`, `profile: {<feature>: smoke}`, its members narrowed | |
| a KV cache dtype list | base `kv_cache_config.dtype: auto`; one variant per other dtype (`llm_api_options.kv_cache_config.dtype`, `gates: [accuracy]`) | |
| chunked prefill, block reuse | `runtime.chunked_prefill: true`, `kv_cache_reuse: true` in **every** gate; one `chunked` variant (`gates: [accuracy]`) with a `max_num_tokens` below the benchmark's prompts (e.g. 512 for 8-shot GSM8K) so they are really split; `no_reuse` only if asked | features that are on everywhere are proven together, which is how they serve |
| disagg (or any feature whose witness runs two engines) | gated at the widest member whose legs fit **one** node (`dep2` on a 4-GPU node) | a wider member needs a cross-node allocation the sessions do not manage: check it outside the flow and list it in `coverage.accept` |
| any other feature (MTP, speculative, offload ...) | name it in `--goal`; `derive.report.md` and `task.example.yaml` say what its unit and witness owe | the same one-variant-per-axis rule applies |
| workload isl / osl / concurrency | `max_seq_len = isl + osl`; `max_batch_size` >= the per-rank concurrency (attention-DP ranks split it); `max_num_tokens` = isl (or 8192); `perf.workload` | |
| "perf tune" | `perf.enabled: true`, concurrencies per member | |

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
  `unit: core`. Each feature then has a check that exercises it (disagg its witness, multimodal
  its boot), and each axis a variant; a variant on a feature unit (a KV dtype under disagg) is a
  combination, added only when the request names it.
- Leave the boot boundaries off (section 5) and the profile at `standard` unless the request asks.

The lint prints the checks per unit and variant; past ~30 deserves a second look.

**Witness launchers.** A witness with a contrast leg (`accuracy.witness.<feature>.contrast_command`)
runs a launcher the run itself writes. Point it at the path that launcher will have, say in
`notes` which task writes it, and set every variable it reads in `contrast_env` (that leg) or
`legs_env` (both legs). A stub that needs a variable the spec never sets fails the witness forever.

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
- **Anchors.** A placeholder is allowed only together with a measurement started now: the reference
  implementation on the target hardware, under the gates' exact protocol, per KV dtype. If the
  reference cannot load the checkpoint as stored, say which weights it measures instead. Until the
  measurement lands, set the placeholder **a few points under the score you expect**: the floor is
  anchor - 2 x SE (about anchor - 1.2 for 1319 questions at 95%), so a placeholder at the expected
  score fails a working target, and one far below it (50 on GSM8K) passes a broken one. The lint
  prints the floor each anchor implies.
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
- **Embargo** — exact paths in `source_embargo`. Do not forbid in-tree kernels, attention backends
  or other models' targets: reusing them is the cheapest rung of the sourcing ladder.

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
what was accepted.
