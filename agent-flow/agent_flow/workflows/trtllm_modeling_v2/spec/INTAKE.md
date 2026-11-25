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
- `tensorrt_llm/_torch/modeling_v2/README.md` and this workflow's README.
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
it came from). With a `site.yaml` pass
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
| a member wider than one node, e.g. "DEP16" on nodes of 4 GPUs | `members: [dep16]`, `environment.gpus_per_node: 4`, a gates session of `nodes: 4` started with its lead, and `environment.step_args` (README, "anything wider than one node"); a width that fills whole nodes | the member is **one engine across nodes**: each of its checks is a job of whole nodes that the session's lead runs once |
| "X needs full validation" | X takes every unit and **one variant per feature axis** at X: the variant's `members: [X]`, `gates: [accuracy]` | a cross product of axes (KV dtype x chunking x reuse ...) multiplies the contract; add a combination only when the request names it |
| "Y only aggregate / text-only" | Y in `members.default` for core alone | |
| "<feature> smoke" | the feature in `features`, `profile: {<feature>: smoke}`, its members narrowed | |
| a KV cache dtype list | base `kv_cache_config.dtype: auto`; one variant per other dtype (`llm_api_options.kv_cache_config.dtype`, `gates: [accuracy]`) | |
| chunked prefill, block reuse | off in core (`runtime.chunked_prefill: false`, `kv_cache_reuse: false`); one `chunked` variant (`gates: [accuracy]`) with a `max_num_tokens` below the benchmark's prompts (e.g. 512 for 8-shot GSM8K) so they are really split, one `reuse` variant with `runtime: {kv_cache_reuse: true}` (the runtime switch decides: `kv_cache_config.enable_block_reuse` alone is overwritten by it), each only when asked (`chunked_prefill`, `prefix_reuse` in `--goal`) | the core configuration is CUDA graphs and the overlap scheduler at TP/TEP/ADP, the KV and SSM caches in the checkpoint's own dtype; every other feature is one variant on core, never combined unless the request names the combination |
| disagg, the deployment known ("context 2 x DEP4, generation 3 x TEP4") | the three files the tree's own disaggregated examples use: `accuracy.disagg.config` (the proxy's, for `trtllm-serve disaggregated -c`: `context_servers.num_instances`, `generation_servers.num_instances`), `.context` and `.generation` (the LLM-API options a server of that side takes with `--config`: its own layout, limits, transceiver); the servers' environment in `accuracy.disagg.env` (`UCX_TLS` ...) | **the flow serves it**: one `trtllm-serve` per instance behind `trtllm-serve disaggregated`, the disagg unit's benchmark (the tree's own `trtllm-eval` command of it, with the unit's `benchmark_args`) scored through the proxy and held to that unit's anchor. A server gets its side's file as it is (the gate adds `modeling_v2: require`), nothing of the unit's options. The flow reads only the instances and a server's ranks (`tensor_parallel_size` x `pipeline_parallel_size`), one device a rank. It is one job on one node when its ranks fit; else a job across a gates session's nodes, as a wide member's is |
| disagg, proven by a test of its own | `accuracy.disagg.test`: the node id of the target's disaggregated-serving accuracy test, written beside the tree's own (the tests that serve through `trtllm-serve disaggregated`, configured, set up and launched as they are), with the environment it reads in `accuracy.disagg.env` (`LLM_MODELS_ROOT` ...); it runs once, at the member the request validates fully. Derive sizes it from the checkpoint: **one node** when a context and a generation engine both fit (`accuracy.disagg.devices`, two engines of two on a 4-GPU node), else `accuracy.disagg.nodes` with one engine per node, as the tree's multi-node disaggregated tests place them | boot and accuracy run one aggregated engine and never cross the hop; a multi-node test runs one pytest process per node of one gates allocation (`MV2FLOW_JOB_NODE_RANK/HOSTS/NODES`), so the gates session must hold that many nodes |
| any other feature (MTP, speculative, offload ...) | name it in `--goal`; `derive.report.md` and `task.example.yaml` say what its unit owes | the same one-variant-per-axis rule applies |
| workload isl / osl / concurrency | `max_seq_len = isl + osl`; `max_batch_size` >= the concurrency; both are **global totals**: under attention DP each rank serves its share (`max_batch_size / ranks`, rounded up); `max_num_tokens` = isl (or 8192) | |

**The smallest contract that covers the request.** Every check loads the model (minutes on a
large checkpoint). A check that passed stands, so each is paid once, and again only after it
failed or when a landing's worker names it. Prove each requested feature **once**, at the member
the request validates fully. One check of any kind covers a feature; the delivery report lists
the checks behind each (`coverage.yaml`):

- A storage or scheduling variant (a KV dtype, `chunked`, `reuse`) takes `gates: [accuracy]`.
  That gate judges the variant's score against its unit's. No gate reads the dtype, the
  chunking or the reuse the engine resolved, and such a variant owes no boot. A variant whose
  gates include `boot` boots at the members of its unit where its benchmark does not run.
- Members the request scopes to text-only or aggregate take core alone, not the variants.
- A feature unit runs at one member (`members: {<feature>: [X]}`), and every variant stays on
  `unit: core`. Each feature then has a check that exercises it (disagg its test, multimodal
  its boot), and each axis a variant; a variant on a feature unit (a KV dtype under MTP) is a
  combination, added only when the request names it. A variant of disagg owes nothing.
- Leave the profile at `standard` unless the request asks.

`--dry-run` journals the contract's checks per unit and variant; past ~30 deserves a second look.

**Disagg: the deployment or a test.** State the deployment whenever the request says how it is
served: no agent then writes serving or test code, and the check is the same for every model.
It needs an anchor row for the disagg unit and its benchmark's data where the gates run (`accuracy.dataset_path`
or the offline caches).

A `test` is the other way: the run writes it; the flow deploys nothing of its own. Name the node
id it will have, say in `notes` which task writes it, and give it what it reads in `accuracy.disagg.env`. A test that skips (too few
devices, a missing model root) fails the gate: set `accuracy.disagg.devices` when it needs more
than two engines at its member's width.

## 4. Runtime

`cuda_graph: true`, `overlap_scheduler: true`, `chunked_prefill: false`, `kv_cache_reuse: false`
(the core configuration; chunking and reuse are variants), `--kv_cache_fraction` around 0.7. Every accuracy gate runs a full benchmark with CUDA graphs: eager
is 10-50x slower and scores a path the target does not serve.
Run a unit or variant's accuracy eager only on purpose (a short subset, a capture still being
built), and say so in `notes`; a benchmark you start by hand runs captured too.

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
- **What fails a check.** Nothing generated, or a benchmark under its floor: the golden rule
  is the benchmark score close to its reference. Kernel entries are judged by their output
  distribution against their reference (`trtllm-modeling-v2-entry`).
- **Drafting floors.** A unit's `accuracy.acceptance` entry holds its own drafter; a later
  unit that runs the same drafter (`features: [mtp, multimodal]`) is held to the entry of the
  speculator it builds on unless it has its own, then to `default`.
- **Anchors.** Each unit's anchor is a row of `reference_yaml` under `accuracy.model_key`, which
  the spec states (`org/model`; the checkpoint path's last two components are no model's name):
  the user's score, else the reference implementation's (HuggingFace `transformers`, where it
  runs the checkpoint natively), else an earlier run's. Without any, record a **low
  placeholder** (e.g. 10 on GSM8K, in percent): it lets a working target integrate and catches an
  engine that answers nothing. The first benchmark that passes on the placeholder at a member
  keeps its score as the unit's anchor there (`anchors.json`), and every later benchmark of the
  unit at that member is held to it, as a variant is to its unit's score. Commit the row to the
  run's branch: each check measures a snapshot of a commit and reads its anchor table there, and
  preflight refuses an uncommitted one.
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
- **Boot prompts.** Two or three short prompts. Boot and serve require an answer, not a
  word in it: an answer cut at `max_tokens` passes, and whether it is right is the
  benchmark's to judge.

## 6. Environment

- `environment.python` exists before launch: the editable install the gates measure, or an
  interpreter the image already has. Every gate's command runs under it.
- A `session_spool` environment's spools exist once their sessions run: start the sessions
  before launch.
- Gate jobs go to `environment.spool` and every `gates` or `any` session; a `work` session gets
  only what a worker submits to it.
- `environment.gpus_per_node` (or a session's) is what tells a member or a deployment wider
  than one node from one that fits: without it everything is launched on one node. What spans
  nodes needs a gates session of that many nodes (`nodes: N`) started with its lead,
  `environment.step_args`, and a `TMPDIR` that is a directory of each node: README, "anything
  wider than one node". Preflight refuses a spec whose gates sessions hold too few nodes.

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
- **Prior art in this tree worth surveying** — the leads you know (attention backends, custom
  ops, other targets' `ops/`, catalog entries), one line each named by its `<dir>/<file>.py`: what
  it offers, or why it does not fit. The run writes its own list, with the engine's limits for
  this checkpoint, into `prior_art.md` when it starts.
- **Engine constraints** — those you know, and the design they imply.
- **Scope decisions** — what is gated where, what is checked outside the flow, and why.
- **References outside the tree** — where you know them: another engine's implementation of
  this model (a checkout path, a package, a URL) and the kernels it uses.
- **Reuse** — existing implementations, tests, kernels and other engines are prior art to read
  before writing. Keep copied code's license and origin, derive values from this checkpoint,
  and validate the implementation against its reference.

Nothing about the intake itself: what intake may or may not do, launch blockers, evidence,
provenance and questions for the user belong in your own summary beside the spec. Notes are read
as standing orders, so "no build is allowed" written for the intake stops the run's own builds.

## 9. Dry run, then launch

`trtllm-modeling-v2 --task task.yaml --workspace W --dry-run` resolves the spec, probes the host
and the agents, and journals the contract and the checks behind each feature, without an agent
turn. Then launch. Once it runs, steer it only through `feedback` (the README's "While it runs"):
cancelling the run's own jobs proves nothing and costs them again.
