<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Intake: from a request to `task.yaml`

For the agent that receives the user's request and launches the workflow. Intake decides what
every later hour costs: a spec that scores without CUDA graphs, lists a deployment the model
does not fit or multiplies deployments will be slow however good the roles are, and `notes` are the only thing every
role reads about the model. Intake writes files and reads the tree; it starts no GPU job.

## 1. Read before writing

- The checkpoint: `config.json` (including any `text_config`, `vision_config`, sparse or
  quantization blocks), the safetensors index, the tokenizer and chat template.
- The reference implementation the user names, enough to write the architecture brief below, and
  whether it can load this checkpoint's weights as stored (its quantization in particular).
- The ModelingV2 developer guide of the checkout (the Markdown file titled `# Modeling V2`:
  `git grep -l "^# Modeling V2$" -- '*.md'`) and this workflow's [README](../README.md): the
  keys a spec must state are its one table.
- For native code, the ModelingV2 native-code guide (titled `# Modeling V2 native code`): family
  libraries, build declarations, namespaces, registration, fake implementations and loading
  errors. A run rebuilds no native code: a rebuild is yours, before the run or between two.

These checkout-local guides define Modeling V2. The flow is an optional development workflow,
not an additional architecture contract. Naming and Python layout are conventions: use the
derived scaffold as a starting point, keep model-specific code at the narrowest scope, allow
family-shared code, and never import a sibling target. Separate prefill/decode classes, weight
tables and `step_args` helpers are choices, not acceptance requirements. Lifecycle methods may
be inherited or delegated. Follow the checkout's actual source checks and CI selectors: a guide
may describe a convention before the corresponding implementation change is merged. Every new
family needs routing registration, unit/accuracy tests reachable from CI lists and appropriate
CODEOWNERS entries; owners review shared changes separately from isolated model changes. (A run
writes its target's accuracy test and that test's entry in a CI list itself, once the first
deployment is onboarded: no survey row and no spec line asks for them.)
- How other engines (vLLM, SGLang) serve this model, where they do: the kernels, the parallel
  layouts and the deployment they use inform the deployments and notes below.

A spec states deployments and nothing else about what runs: code makes the tasks from one
survey turn, and there is no tuning stage. How a run works is [DESIGN.md](../DESIGN.md).

## 2. Start from a derived spec

```bash
trtllm-modeling-v2 intake derive --checkpoint <ckpt> --repo <trtllm checkout> \
    --goal "<layouts> <features> kv=bf16/fp8/nvfp4 isl=<n> osl=<n> concurrency=<a>/<b>/<c>" \
    --out <dir>
```

It writes `task.derived.yaml`, each deployment's native options (`engines/<name>.yaml`),
`derive.report.md` (every value the goal did not state and where it came from) and
`intake.report.md` (the advice below, with the engine's own model of the architecture
where it registers one), and prints the advice. With a `site.yaml`
pass `--site` instead of `--repo`. Review the derived spec and its options files, edit them,
then save the spec as `task.yaml` beside `engines/` (its options paths are relative to it):
the commands below take that name. Copy
`task.example.yaml` only for what the goal words cannot say.

- Each layout the goal names (`tp1`, `tep4`, `dep4`) is one deployment of the core configuration
  at that layout: CUDA graphs and the overlap scheduler on, the KV cache in the checkpoint's own
  format, block reuse and chunked prefill off. The first is the first deployment: it defines
  `onboarded`.
- A goal gets the features it names and no other: a drafter or vision tower the checkpoint
  carries is left out unless the goal names it, and `derive.report.md` says the word that adds
  it. A first run is one deployment.
- `kv=` lists the KV dtypes in one word: the first is every deployment's, each other one a
  deployment of its own judged against the first one's score.
- Derive asks rather than guesses (and stops until the spec answers): `model_key` when
  `config.json` names no model.

## 3. Map the request to deployments

A deployment is one configuration the checkpoint is served in: a name and the options
`trtllm-serve --config` takes, read as they are. `deployments` is the only way a spec says what
runs. No deployment says how it is launched: its world size is `tensor_parallel_size x
pipeline_parallel_size x context_parallel_size`, as the engine enforces it (expert parallelism
shares those ranks), and the pool places it.

| the request says | the spec | why |
|---|---|---|
| layouts, e.g. "TP4, TEP4, ADP4" | one deployment each: `tp4`, `tep4`, `dep4` (ADP = attention DP + EP: `enable_attention_dp: true`) | each is checked in its own session, at its own width |
| a KV cache dtype list | the first deployment keeps the checkpoint's own (`dtype` unsaid); one more deployment per other dtype (`kv_cache_config: {dtype: fp8}`, `relative_to: <first>`) | where no reference row fits, it is judged against the first deployment's own score |
| chunked prefill, block reuse | off in the first deployment; one `chunked` deployment (`enable_chunked_prefill: true` and a `max_num_tokens` below the benchmark's prompts, e.g. 512 for 8-shot GSM8K, so they are really split) and one `reuse` deployment (`kv_cache_config.enable_block_reuse: true`), each `relative_to` the first, each only when asked | each feature on its own, never a cross product; the session of each proves the feature it turns on (`probes`) |
| a drafter (MTP, EAGLE3, ...) | a deployment whose options state `speculative_config` | its session reads the accepted tokens per step from the answers |
| a vision or audio tower | a deployment with `inputs: [text, image]` | its session also sends an image |
| disaggregation | a deployment with `ctx` and `gen`, each with its own options (they may differ in every parallel size) and `instances`; both state `cache_transceiver_config: {backend: DEFAULT}`; `env` states the transports the site's network needs (`UCX_TLS`) | one session starts one server per role instance and the proxy, proves that generation received the context's state, and scores through the hop |
| "<feature> smoke" | that deployment's `benchmarks: []` | its session's own checks, and no score |
| a native test that must also pass | the deployment's `tests`: pytest node ids of the tree with `::` (`<file>::<Class>::<test>`; a bare file is refused), run in one process on one node: a test that makes an MPI pool of its own is left out and its check reads `not verified` | what a session does not replace: a state-transfer test across unequal layouts, a feature's own test |
| workload isl / osl / concurrency | `max_seq_len = isl + osl`; `max_batch_size` >= the concurrency; under attention DP each rank serves its share (`max_batch_size / ranks`, rounded up); `max_num_tokens` = isl (or 8192) | the options are the engine's own: the flow derives nothing from them |

**The smallest spec that covers the request.** Every session loads the model (minutes on a large
checkpoint), and the final pass measures again every session whose target changed since it was
measured, so each deployment is paid again for every candidate. Prove each requested feature **once**, on one deployment, and leave out what
the request does not ask for: a configuration the run does not owe is a deployment the spec does
not list.

**A first milestone first, when a load is long.** Count what one candidate costs: the loads
`intake lint` lists (one session for each deployment, and one more for each benchmark it says
is `trtllm-eval in its own load`) times the minutes one load of this checkpoint takes. The
final pass measures again every session whose target changed since it was measured, and a
repair after it is a new candidate, so that product is paid once more for each. When it comes
to hours, or one load alone takes tens of minutes, onboard in two runs, so that one milestone
is reached before anything else is asked for:

1. The first run lists the first deployment alone. It ends when that deployment is
   `onboarded` and delivered. The ref it pinned is `refs/mv2/<run>/onboarded` (`<run>` is
   the run's slug); the closing report and `<workspace>/onboarded.json` name it.
2. The second run starts on the checkout as the first one delivered it, states `onboarded:
   <that ref>` at the top of its spec and lists the other deployments. Its survey writes only
   the rows they add, no deployment waits for a first one, and nothing this run changes makes
   the first deployment's session be measured again: it is not a deployment of this run.
   (So what this run changes is not proven on the first deployment: list it here too when
   that proof is wanted.) The run pins no `onboarded` of its own, and its closing report
   says `not pinned`: the milestone is the first run's. A deployment that would have been
   judged `relative_to` the first one states, on each of its benchmarks, the score the first
   run measured as its `anchor`, with that run's report as the `source`; its floor is then
   the declared anchor's own.

## 4. One session per deployment

Each deployment is checked in one session: one job holding its devices, its servers loaded once,
every check a client of them, in this order, each with its own line.

| check | when | what it shows |
|---|---|---|
| `devices` | always, before any load | each server has devices of its own; two servers of one node share none |
| `attest` | always | every rank of every role instance ran the options it was given (a value `auto` or `DEFAULT` leaves to the engine is shown as chosen), from the tree measured, on the devices given. The Modeling V2 requirement is `modeling_v2: require` unless the deployment's options state another value (`off` for a control on the built-in path, `auto`); a stated value is used as stated in the options, the launch and the attestation, and the `attest` line and the session's line say so |
| `boot`, `api` | always | the boot prompts come back completed, with their keywords; a chat completion and a streamed completion answer |
| `image` | `inputs: [text, image]` | a chat completion with an image |
| `chunk` | `probes` (or the options turn chunked prefill on) | a prompt past `max_num_tokens` is served |
| `hop` | `ctx` and `gen` | generation received the context's state for each request, as the measured tree lets a client see it; `not verified`, with the reason, where it offers no way |
| `accuracy[<benchmark>]` | each benchmark | the score through the endpoint against its floor |
| `acceptance` | the options state a drafter | accepted tokens per step, read from the answers |
| `reuse` | `probes` (or the options turn block reuse on) | a prompt sent again resumes from the cache. It changes the engine's state, so it runs after everything that scores |

- A benchmark the endpoint evaluator does not carry (it carries `gsm8k`), or one marked `through:
  eval`, is `trtllm-eval` in a load of its own after the session, and its line says so. Through
  `ctx` and `gen` there is no such load: it would not cross the hop, so the benchmark is blocked.
  The endpoint as the default judge has been compared with `trtllm-eval` on one small model only;
  the parity run on a mid-size model is still owed, so mark a benchmark `through: eval` where its
  score must be `trtllm-eval`'s own.
- A check that lacks what it is judged by (an anchor, a dataset, the score it is relative to) is
  **blocked** before its load, with the reason, and everything else goes on.
- Run one by hand, exactly as the run does: `trtllm-modeling-v2 gate deployment --task task.yaml
  --deployment <name>`. While a run of that workspace (the task file's directory) is going, its
  pool serves the check; with none going, the command asks for one holder of the job's nodes
  itself, for the job's wall with the pool's lead and not the spec's whole `time`, counts it
  against `environment.slurm.max_node_hours`, and gives it back when the check is done (on one
  host that states no spool, it runs in the command's own process). A kernel row's own test:
  `gate kernel --row <row>`; any other row's: `gate cpu --row <row> --test <file>`.

## 5. Accuracy and boot

- **Protocol.** Score the benchmark (GSM8K by default) as few-shot completion (no
  `--apply_chat_template`) unless the model only works chat-formatted. A chat-template protocol on
  a model that reasons first needs `--max_output_length` >= 1024 and its stop strings, or it scores
  truncation. A benchmark's protocol is its own `args`.
- **Benchmark and score.** Which benchmark, how much of it and how its score is read are the
  user's to choose. A subset: `args: ["--num_samples=200"]` (the floor's standard error uses that
  count). A benchmark whose output is neither lm-eval's `accuracy:` line nor MMLU's weighted
  average needs `score_pattern` (a regular expression whose first group is the score, on the
  anchor's scale) and `samples`; without one, its line quotes the lines that look like a score
  and fails.
- **What fails a check.** Output that is wrong (nothing generated, a boot keyword missing) or a
  benchmark under its floor.
- **Anchors.** A benchmark's anchor is, in this order: the `anchor` it declares (with its
  `source`), the tree's reference row for `accuracy.model_key` (`org/model`) with exactly the
  deployment's tags (what its options imply: a quantized KV cache, a drafter; and the
  benchmark's `discriminators`: the checkpoint's weight format), the score of the deployment
  `relative_to` names, or nothing. They are frozen in `<workspace>/anchors.yaml` at the first
  start, so no commit of the run moves what its checks are held to. A declared anchor is
  delivered with the target: the task the flow makes for the delivery's files adds its row to
  the tree's reference file (the model's key, the score, the count and the source as a
  comment), because the accuracy test for CI reads its anchor there.
- **No anchor yet.** Leave the benchmark out and onboard on the session's own checks, or declare
  a low placeholder (e.g. `anchor: 10` on GSM8K, with a `source` that says so): it lets a working
  target integrate and catches an engine that answers nothing. The advice names a placeholder,
  and nothing replaces it by itself.
- **Changing a requirement.** No anchor is beyond change. One that is mistaken, was a
  placeholder, or was frozen as nothing is changed by an explicit command, which records what
  changed, from what to what and why (the run journals every change at each start):
  `trtllm-modeling-v2 resolve --workspace <ws> change-requirement <task|run> --deployment <d>
  --benchmark <b> --anchor <v> --reason "<why>"` (the closing report lists them).
- **Datasets.** Staged from preflight on, beside the first turns: the benchmark's `dataset_path`,
  else the copy under `$LLM_MODELS_ROOT/datasets/`, else a download into the workspace. A session
  waits for one being staged; `datasets/status.json` says which. Accuracy runs offline. Both
  judges (the session's endpoint evaluator and `trtllm-eval`) build the tree's own evaluator,
  which reads the path it is given (the benchmark's `dataset_path`, else in a run the staged
  one) and, given none (a check by hand stages nothing), the task's dataset by its id in the
  HuggingFace caches under `HF_HOME` on the node; the tree's native tests read
  `$LLM_MODELS_ROOT/datasets/`. A site has both. The session's job asks the evaluator's loader on
  its node before it starts a server: a benchmark whose data it cannot load is blocked there,
  with where it looked, and the other checks go on.
- **Boot prompts.** One checked factual prompt and one checked arithmetic word problem (a
  benchmark-style question with its numeric answer as the keyword): cheap semantic signal in
  every session.
- **A drafter's acceptance** (`accuracy.acceptance`). The workload, and per drafting deployment
  what its reading is held to: `min_absolute`, or `measured_reference` (the floor is 0.95 of it)
  with its `floor_source`. No other engine is loaded for it.

## 6. Environment

- `environment.python` exists before launch: the editable install the checks measure, or an
  interpreter the image already has. Every check's command runs under it.
- The environment is never chosen silently: one the spec does not name is detected (whether
  the host the command runs on has `sbatch`) and written into the spec the run reads. `intake
  derive --repo` writes `kind: local` on any host. On Slurm, remove that line and state
  `environment.slurm`: `account`, `partition`, `time` and `gpus_per_node` are required, each
  missing one prints how to find its value, and the values are the site's (its site file, not
  this document). The flow serves no watcher that somebody else starts: a spec that states
  `environment.sessions` or `kind: session_spool` is refused with that line.
- A deployment wider than one node needs `environment.slurm`; on one host it is refused
  before a load, with that reason. A session's wall is the deployment's `timeout`.
- No stock model is loaded to prove the host: a workspace's first start runs the pool's
  allocation probe for the widest shape the spec needs, and a pass is kept for that shape.
  What the pool does with the block is
  [DESIGN.md](../DESIGN.md), "Compute".

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
- **Code of this tree worth a survey's time** — the engine's own model of the architecture,
  if `intake.report.md` names one, and what else you know fits: one line each, named by what
  to search for (a class, a function, an op), never by a path: the tree moves.
- **Engine constraints** — from `intake.report.md` and the design they imply.
- **Scope decisions** — which deployment proves what, what is checked outside the flow, and why.
- **References outside the tree** — where you know them: another engine's implementation of
  this model (a checkout on this host, a package, a URL) and the kernels it uses.
- **Reuse** — existing implementations, tests, kernels and other engines are prior art to read
  before writing. Keep copied code's license and origin, derive values from this checkpoint,
  and validate the implementation against its reference.

Nothing about the intake itself: what intake may or may not do, launch blockers, evidence,
provenance and questions for the user belong in your own summary beside the spec. Notes are read
as standing orders, so "no build is allowed" written for the intake stops the run's own builds.

## 9. Read the advice, then launch

```bash
trtllm-modeling-v2 intake lint --task task.yaml --report intake.report.md
trtllm-modeling-v2 --task task.yaml --workspace <ws> --dry-run
```

Both print what the spec is likely to cost or miss: a first deployment too narrow for the
weights, a benchmark with nothing to hold it to or with an anchor that reads as a placeholder,
where the checkpoint's `config.json` cannot meet a deployment as
stated (heads or experts that do not split, a drafter with no draft layers, an image with no
tower, block reuse on a hybrid checkpoint with no snapshot policy), a KV dtype the engine does
not name, notes with no brief, the engine's limits for this geometry and what the tree already
has for it. It is **advice**: nothing here stops a run. Only a spec that cannot be read as one
does (a schema error, an options file that does not resolve), and a check that lacks what it is
judged by is blocked before its own load while everything else goes on. The run's journal keeps
the advice. Once it runs, [OPERATING.md](../OPERATING.md) is the protocol for watching it.

**A dry run decides nothing.** What it leaves in the workspace (the spec's copy, the
contract) each real start writes again from the spec as it is then, and it freezes no anchor:
repeat it as often as the spec changes.

**A dry run proves the host it is typed on.** Its preflight and its agent probe run there.
Where a site's script runs the command inside a job, that is another node: a dry run on the
login node says nothing about that node's agents, proxy or `PATH`, and a dry run submitted as
a job is not built. The first proof is the run's own preflight and agent probe, when it starts
there.
