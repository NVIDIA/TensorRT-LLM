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
- For native code, `cpp/tensorrt_llm/modeling_v2/README.md`: family libraries, build declarations,
  namespaces, registration, fake implementations, loading errors and rebuild/install identity.

These checkout-local guides define Modeling V2. The flow is an optional development workflow,
not an additional architecture contract. Naming and Python layout are conventions: use the
derived scaffold as a starting point, keep model-specific code at the narrowest scope, allow
family-shared code, and never import a sibling target. Separate prefill/decode classes, weight
tables and `step_args` helpers are choices, not acceptance requirements. Lifecycle methods may
be inherited or delegated. Follow the checkout's actual source checks and CI selectors: a guide
may describe a convention before the corresponding implementation change is merged. Every new
family needs routing registration, unit/accuracy tests reachable from CI lists and appropriate
CODEOWNERS entries; owners review shared changes separately from isolated model changes.
- How other engines (vLLM, SGLang) serve this model, where they do: the kernels, the parallel
  layouts and the deployment they use inform the members, features and notes below.

## 2. Start from a derived spec

```bash
trtllm-modeling-v2 intake derive --checkpoint <ckpt> --repo <trtllm checkout> \
    --goal "<members> <features> kv=bf16/fp8/nvfp4 isl=<n> osl=<n> concurrency=<a>/<b>/<c>" \
    --out <dir>
```

It writes `task.derived.yaml` and `derive.report.md` (every value the goal did not state and where
it came from), and prints the spec's warnings (section 9). With a `site.yaml` pass
`--site` instead of `--repo`. Edit the derived spec; copy `task.example.yaml` only for what the
goal words cannot say.

- A goal that names features gets those: a drafter or vision tower the checkpoint carries and the
  request does not ask for is left out (`derive.report.md` says so). Name it to build it.
- `kv=` lists the KV dtypes in one word: the first is every gate's, each other one a variant
  (`gates: [accuracy]`) judged against the unit's own score.
- Derive asks rather than guesses (and stops until the spec answers): `model_key` when
  `config.json` names no model.

## 3. Map the request

| the request says | the spec | why |
|---|---|---|
| members, e.g. "TP4, TEP4, ADP4" | `members: [tp4, tep4, dep4]` (ADP = attention DP + EP = `dep`) | |
| "X needs full validation" | X takes every unit and **one variant per feature axis** at X: the variant's `members: [X]`, `gates: [accuracy]` | a cross product of axes (KV dtype x chunking x reuse ...) multiplies the contract; add a combination only when the request names it |
| "Y only aggregate / text-only" | Y in `members.default` for core alone; its variant and feature cells in `coverage.accept` | |
| "<feature> smoke" | the feature in `features`, `profile: {<feature>: smoke}`, its members narrowed | |
| a KV cache dtype list | base `kv_cache_config.dtype: auto`; one variant per other dtype (`llm_api_options.kv_cache_config.dtype`, `gates: [accuracy]`) | |
| chunked prefill, block reuse | off in core (`runtime.chunked_prefill: false`, `kv_cache_reuse: false`); one `chunked` variant (`gates: [accuracy]`) with a `max_num_tokens` below the benchmark's prompts (e.g. 512 for 8-shot GSM8K) so they are really split, one `reuse` variant with `kv_cache_config.enable_block_reuse: true`, each only when asked (`chunked_prefill`, `prefix_reuse` in `--goal`) | the core configuration is CUDA graphs and the overlap scheduler at TP/TEP/ADP, the KV and SSM caches in the checkpoint's own dtype; every other feature is one variant on core, never combined (the lint blocks a cross product unless the notes accept the one the request names) |
| disagg | `accuracy.disagg.test`: the node id of the target's disaggregated-serving accuracy test, written beside the tree's own (the tests that serve through `trtllm-serve disaggregated`, configured, set up and launched as they are), with the environment it reads in `accuracy.disagg.env` (`LLM_MODELS_ROOT` ...); it runs once, at the member the request validates fully. Derive sizes it from the checkpoint: **one node** when a context and a generation engine both fit (`accuracy.disagg.devices`, two engines of two on a 4-GPU node), else `accuracy.disagg.nodes` with one engine per node, as the tree's multi-node disaggregated tests place them | boot and accuracy run one aggregated engine and never cross the hop; a multi-node test runs one pytest process per node of one gates allocation (`MV2FLOW_JOB_NODE_RANK/HOSTS/NODES`), so the gates session must hold that many nodes |
| any other feature (MTP, speculative, offload ...) | name it in `--goal`; `derive.report.md` and `task.example.yaml` say what its unit owes | the same one-variant-per-axis rule applies |
| workload isl / osl / concurrency | `max_seq_len = isl + osl`; `max_batch_size` >= the concurrency; both are **global totals**: under attention DP each rank serves its share (`max_batch_size / ranks`, rounded up); `max_num_tokens` = isl (or 8192) | |

**The smallest contract that covers the request.** Every check loads the model (minutes on a
large checkpoint). The workers run their own unit's checks while they build, and the certification
runs the whole contract once on the final commit, and again after each repair. Prove each
requested feature **once**, at the member the request validates fully, and accept the rest in
`coverage.accept`, where the delivery report states it:

- A storage or scheduling variant (a KV dtype, `chunked`, `reuse`) takes `gates: [accuracy]`.
  The accuracy check loads the same engine and fails unless each rank reports the variant's
  setting (its KV dtype, its chunking); a variant that boots, boots at **every** member of its
  unit.
- Members the request scopes to text-only or aggregate take core alone, not the variants.
- A feature unit runs at one member (`members: {<feature>: [X]}`), and every variant stays on
  `unit: core`. Each feature then has a check that exercises it (disagg its test, multimodal
  its boot), and each axis a variant; a variant on a feature unit (a KV dtype under MTP) is a
  combination, added only when the request names it. A variant of disagg owes nothing.
- Leave the profile at `standard` unless the request asks.

**The disagg test.** The run writes it; the flow deploys nothing of its own. Name the node id it
will have, say in `notes` which task writes it, and give it what it reads in `accuracy.disagg.env`. A test that skips (too few
devices, a missing model root) fails the gate: set `accuracy.disagg.devices` when it needs more
than two engines at its member's width.

## 4. Runtime

`cuda_graph: true`, `overlap_scheduler: true`, `chunked_prefill: false`, `kv_cache_reuse: false`
(the core configuration; chunking and reuse are variants), `--kv_cache_fraction` around 0.7. Every accuracy gate runs a full benchmark with CUDA graphs: eager
is 10-50x slower, each fix-and-recheck pays it, and it scores a path the target does not serve.
The spec's warnings name a deployment whose accuracy runs eager (section 9); a
benchmark you start by hand runs captured too.

## 5. Accuracy and boot

- **Protocol.** Score the benchmark (GSM8K by default) as few-shot completion (no
  `--apply_chat_template`) unless the model only works chat-formatted. A chat-template protocol on
  a model that reasons first needs `--max_output_length` >= 1024 and its stop strings, or it scores
  truncation.
- **Benchmark and score.** Which benchmark, how much of it and how its score is read are the
  user's to choose. A subset: `benchmark_args: ["--num_samples=200"]` (the floor's standard
  error uses that count). Another benchmark: `benchmark`, with an anchor row for it. A benchmark
  whose output is neither lm-eval's `accuracy:` line nor MMLU's weighted average needs
  `score_pattern` (a regular expression whose first group is the score, on the anchor's scale)
  and `samples`; without one, its gate quotes the lines that look like a score and fails.
- **What fails a check.** Output that is wrong (nothing generated, a boot keyword missing) or a
  benchmark under its floor. The golden rule is the benchmark score close to its reference.
  Kernel entries are judged by their output distribution against their reference
  (`trtllm-modeling-v2-entry`).
- **Anchors.** Each unit's anchor is a row of `reference_yaml` under `accuracy.model_key`, which
  the spec states (`org/model`; the checkpoint path's last two components are no model's name):
  the user's score, else the reference implementation's (HuggingFace `transformers`, where it
  runs the checkpoint natively), else an earlier run's. **Get a real anchor before launch.** The
  certification holds each benchmark to the row as written: no score the run measures replaces
  it. A low placeholder (e.g. 10 on GSM8K, in percent) only catches an engine that answers
  nothing, so a run delivered on it certifies no accuracy. Commit the row to
  the checkout: the certification reads the anchor table from the commit it measures and refuses
  a checkout with uncommitted changes.
- **Datasets.** Staged from preflight on, beside the first turns: the spec's
  `accuracy.dataset_path`, else the copy under `$LLM_MODELS_ROOT/datasets/` (where the tree's
  `accuracy_core.py` reads it), else a download into the workspace. A gate waits for one being
  staged and runs as before without one that could not be; `datasets/status.json` says which.
  The benchmark itself is the tree's (`trtllm-eval` runs the task `accuracy_core.py` defines: for
  GSM8K, 1319 questions, 4096 input and 256 output tokens, seed 0).
- **Variant anchors.** A configuration variant that names no row of its own (`discriminators`,
  `reference_yaml`, `model_key`, `benchmark`) is judged against its unit's own score at the same
  member, on the same target (`anchor: relative`): a KV dtype against `auto`, the chunked variant
  against the unchunked engine. Its floor allows for the error of both scores. A variant that
  names a row of its own is judged against that row (its `discriminators` are layered over the
  unit's).
- **Boot prompts.** One checked factual prompt and one checked arithmetic word problem (a
  benchmark-style question with its numeric answer as the keyword): cheap semantic signal at every
  boot.

## 6. Environment

- `environment.python` exists before launch: the editable install the gates measure, or an
  interpreter the image already has. Every gate's command runs under it.
- `environment.kind` is `local`: every command runs on the orchestrator's host. On a cluster,
  launch the workflow inside the allocation that holds the GPUs.

## 7. Agents: yours

The roles run on your harness, model and reasoning effort. Write all three into the spec:

```yaml
agents:
  defaults: {backend: <claude-code | codex>, model: <your model id>, reasoning_effort: <your effort>}
```

- The spec is the only source: nothing is inherited from your session or the environment, and a
  role with no stated model stops the run before any worker starts.
- Take them from your own session: the exact model id you run as and your effort. When you cannot
  tell one, ask the user; never write another.
- `intake derive --site` copies the site's `agents` block into the spec: replace it with yours
  unless the user asked for it.
- The roles sign in on the orchestrator's host, so your model must be served there (one behind a
  proxy only your workstation reaches is not). When it is not, stop and ask the user what to use;
  never change the harness or the model yourself.
- `agents.roles.<role>` only for a role the user asks to run differently.
- The agent probe (preflight, so `--dry-run` too) runs every pinned agent on the orchestrator's
  host and prints each one's backend, model and effort: check them against yours before launch.

## 8. Notes: the run's brief

`notes` reach every role verbatim, in every turn. Keep them a brief (under ~6000 characters) of
what the roles need while they work:

- **Reference** — where it is, what it supports, how it loads this checkpoint's weights.
- **Architecture** — layer pattern, attention kinds with head dims, sparse/indexer configuration,
  MoE (experts, top-k, clamps, router), norm and residual dtypes, towers.
- **Prior art in this tree worth surveying** — the leads you checked in the tree that hold
  (attention backends, custom ops, other targets' `ops/`, catalog entries), one line each named by
  its `<dir>/<file>.py`: what it offers.
- **Engine constraints** — what the engine's kernels refuse for this geometry, and the design they imply.
- **Scope decisions** — what is gated where, what is checked outside the flow, and why.
- **References outside the tree** — where you know them: another engine's implementation of
  this model (a checkout path, a package, a URL) and the kernels it uses.
- **Reuse** — existing implementations, tests, kernels and other engines are prior art to read
  before writing. Keep copied code's license and origin, derive values from this checkpoint,
  and validate the implementation against its reference.

Nothing about the intake itself: what intake may or may not do, launch blockers, evidence,
provenance and questions for the user belong in your own summary beside the spec. Notes are read
as standing orders, so "no build is allowed" written for the intake stops the run's own builds.

## 9. Warnings, then launch

`intake derive` prints them, and preflight journals them again (`--dry-run` runs preflight, touches
no GPU and exits 0 when it passes). Each says what to do; none stops a run:

- **first deployment too narrow**: the checkpoint's weights take more than 75% of the memory of
  the devices the first deployment has (`environment.gpu_memory_gib` says how much one holds).
- **no accuracy anchor**: the first deployment names no benchmark, or the tree has no reference
  row for `accuracy.model_key` and the benchmark declares no `anchor`.
- **accuracy without CUDA graphs**: a scored deployment's options set `cuda_graph_config: null`.

Then launch. Once it runs, steer it only through `feedback` (the README's "While it
runs").
