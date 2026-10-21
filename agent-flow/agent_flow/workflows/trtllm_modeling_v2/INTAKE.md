<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Intake: from a request to a confirmed task

Status: the request schema, the fast path, the feasibility checks,
confirmation and compilation are implemented (`spec/intake/`); the
natural-language request parser and its interactive loop (section 7), the
memory and compatibility-matrix checks, budgets and `nice_to_have` are not.
Step 2 (the pipeline described in [README.md](README.md)) reads only the
compiled spec.

## The fast path

```bash
trtllm-modeling-v2 --checkpoint /models/newmodel --site site.yaml --workspace W \
    [--goal "mtp disagg@tp1/tep4 tp1 tep4 dep4 isl=8192 osl=1024 tune"] [--confirm]
```

derives the request in seconds (`spec/intake/derive.py`), writes it to
`W/task.intake.yaml` beside `W/intake.report.md` (what was read, what was
defaulted, what is still open), compiles it to `W/task.resolved.yaml` and
starts the run. There is no separate intake step: the request's checks run in
phase 0, first, beside the survey and the substrate gate, together with the
file-reading checks of the compiled spec (each unit's and each variant's
accuracy anchor, the speculation each launch path is asked for, the capture
lists), and a block stops the run there with every block listed.

| value | from |
|---|---|
| architecture, MoE or dense, weight and KV-cache quantization | `config.json`, `hf_quant_config.json` |
| speculative decoding (`mtp` or `dspark`) and its depth; multimodal | the checkpoint's draft layers and towers: built unless the goal says `-<feature>` (`text_only` is `-multimodal`) |
| other features (`disagg`, `kv_quant`, ...) | the goal only; `<feature>@tp1/tep4` narrows one to members |
| `device.arch`, `gpus_per_node`, `nodes` | `arch=` in the goal, else `site.yaml device` |
| members | the goal's `tp1 tep4 ...`, else `site.yaml defaults.members`, else `tp1` |
| workload | the goal's `isl= osl= max_seq_len= max_batch_size= concurrency=1/16`, else `defaults.workload`, else isl 4096, osl 1024, max batch 64 (fitted to `max_position_embeddings`) |
| core switches | `cuda_graph`, `overlap_scheduler`, `chunked_prefill`, `prefix_reuse` required unless the goal says `-<switch>`; a hybrid checkpoint gets a 2048-token state snapshot interval for block reuse (`snapshot=`) |
| accuracy anchor | `benchmark= model_key= reference=`, else `defaults.accuracy`, else gsm8k keyed by the config's `_name_or_path` |
| boot prompts | `defaults.boot`, else a checked `The capital of France is` -> `Paris` and one open prompt |
| witness declarations | `defaults.witness.<feature>`, and a replay window from `sliding_window`; what is still missing is an open question |
| environment, sessions, agents, commit policy, perf spools | `site.yaml`, as for any intake task |

`site.yaml` takes two optional blocks for this: `device: {arch, gpus_per_node,
nodes}` and `defaults: {members, workload, accuracy, boot, witness}`.

Confirmation is recorded, not awaited. `--confirm [BY]` on the command, or
`trtllm-modeling-v2 intake confirm W/task.intake.yaml --by NAME` at any time,
writes `W/task.intake.yaml.confirmed`; the request itself is never rewritten.
Until the run is past preflight the request may be edited and the same command
re-run. After that, `W/task.intake.yaml.started` holds the digest it started
from, and an edit is refused until it is confirmed; a request already
confirmed is refused once it changes, as a hand-written one is. Re-running the
command resumes the request on disk; another `--goal` or `--checkpoint` is
refused, and `--clean` derives it again.

## 1. Problem

A run today starts from `task.example.yaml` (810 lines) validated by
`task_schema.py` (2,400 lines), plus a natural-language prompt. Three things
go wrong with that:

- **The user reviews the wrong thing.** About a dozen fields express what the
  user wants. The rest — `environment` (~200 lines), accuracy floors, witness
  workloads, agent backends, commit policy — say how to run it. Nobody can
  check a requirement buried in that.
- **Requirements are inferred by agents, mid-run.** Whether a checkpoint's
  `mtp.*` weights are MTP heads or a DSpark drafter, whether `max_seq_len`
  has to be set explicitly, which benchmark protocol a reasoning model needs:
  these are taught to agents through prompts and skills. When an agent gets
  one wrong, the run spends hours before a gate notices.
- **Infeasible requests are discovered late.** Preflight
  (`check_checkpoint`, `check_speculation_support`, `check_accuracy_anchors`,
  …) runs after the user has committed to a run, and it only checks what the
  spec already says. It cannot say "this model does not fit TP4 on this GPU"
  or "this checkpoint has no MTP weights".

## 2. Goal

Add a **Step 1, intake**. It turns a natural-language request into a short
`task.yaml` that the user reads, corrects and confirms before any GPU or agent
time is spent. After that, Step 2 runs as it does today.

```text
request (natural language)
   │
   ▼
Step 1  intake ──► task.yaml (requirements; the user confirms this)
   │               intake.report.md (what was checked, what was inferred)
   │  + site.yaml (per cluster, reused)
   ▼
compile ──► task.resolved.yaml (today's schema, unchanged)
   │
   ▼
Step 2  trtllm-modeling-v2 (preflight → plan → smith → assemble → deliver → [tune])
```

Non-goals:

- changing Step 2's stages, gates or schema;
- letting intake write modeling code or launch GPU jobs. Intake may read
  checkpoints, the repository and the site description, nothing more.

## 3. Three files

| file | written by | reviewed by user | holds |
|---|---|---|---|
| `task.yaml` | intake, edited by the user | **yes**, target 50–80 lines | requirements: model, device, parallelism, workload, features, accuracy, perf, constraints, notes |
| `site.yaml` | once per cluster | no | `environment` (`local` or `session_spool`, with its sessions), `trtllm_repo_path`, perf spools and `env_script`, `agents` defaults |
| `task.resolved.yaml` | `compile`, never edited | no (audit only) | today's full schema, the only file Step 2 reads |

Splitting `site.yaml` out removes the largest block from what the user reads,
and it is the one block that does not change between tasks on the same
cluster.

## 4. `task.yaml`

### 4.1 Example

```yaml
version: 1
goals: [onboard, tune]       # any of: onboard | add_features | tune; see 4.2

model:
  checkpoint: /models/Qwen3-30B-A3B-NVFP4
  architecture: Qwen3MoeForCausalLM          # inferred: config.json
  quantization: {weights: nvfp4, kv_cache: bf16}   # inferred: hf_quant_config.json
  kind: moe                                  # inferred: num_experts=128
  in_tree: tensorrt_llm/_torch/models/modeling_qwen3_moe.py   # inferred
  reference_impl: {engine: vllm, path: /src/vllm}   # also the accuracy oracle
  base: null                 # required when goals has no `onboard`: the existing target and branch

device:
  arch: sm_100               # B200
  gpus_per_node: 8
  nodes: 1

parallelism:                 # one entry = one config; several = `unified`
  - {kind: tep, world: 4}
  - {kind: dep, world: 4}

workload:
  isl: 8192
  osl: 1024
  max_seq_len: 9216          # inferred: isl + osl, never the checkpoint's 262144
  max_batch_size: 64
  concurrency: [1, 16, 64]

features:                    # section 5
  cuda_graph: required
  overlap_scheduler: required
  chunked_prefill: {status: required, params: {chunk_tokens: 8192}}
  prefix_reuse: required
  kv_cache: {status: nice_to_have, params: {dtype: fp8}}
  long_context: out_of_scope
  speculative: {status: required, params: {algo: mtp, draft_len: 3}}   # inferred: num_nextn_predict_layers=1
  host_offload: {status: nice_to_have, params: {tensors: [per_layer_embed]}, members: [dep4]}
  disaggregated: out_of_scope
  multimodal: out_of_scope     # becomes scope: text_only

accuracy:
  benchmark: gsm8k
  protocol: {chat_template: true, reasoning: off, shots: 0}
  reference: {source: reference_yaml}        # or: {source: measure, engine: vllm}

perf:                        # present exactly when goals contains `tune`; section 5.8
  metric: output_tps_per_gpu
  baseline: stock            # stock | vllm | sglang | <branch>
  goal: best_effort          # or: {ratio_to_baseline: 1.1} and/or {pct_of_sol: 0.7}
  configs:
    - name: low_latency
      parallel: tep4
      features: {cuda_graph: on, speculative: on}
      workload: {isl: 8192, osl: 1024, concurrency: [1, 4, 16]}
      constraint: {tpot_ms_max: 30}

constraints:
  embargo: [tensorrt_llm/_torch/models/modeling_qwen3_moe.py, /src/vllm]
  base_ref: origin/main
  budget: {gpu_hours: 200, wall_days: 3}

deliverable: {branch: auto, pr: false, tests: [unit, accuracy_reference]}

notes: |
  Follow modeling_v2 naming. Reuse the NVFP4 MoE kernels in
  tensorrt_llm/_torch/modules/fused_moe; do not add new CUDA.

provenance:                  # maintained by intake; see 4.3
  model.architecture: inferred
  workload.max_seq_len: inferred
  features.speculative.algo: inferred
  perf.constraint: user
open_questions: []           # must be empty to confirm
# confirmed: {by, at, sha256} is written to task.yaml.confirmed, beside this file
# derived: {checkpoint, site, goal, at}, when the fast path wrote this file
```

### 4.2 Fields

**`goals`** is a set, not one mode. A request often asks for more than one
thing ("onboard it, then tune it"), so intake records every goal it heard and
the run executes them in a fixed order: build first, then tune.

| goal | what it means | requires |
|---|---|---|
| `onboard` | build a new target for a checkpoint with none | `model`, `features`, `accuracy` |
| `add_features` | add features to an existing target | `model.base`, the new features with `status: required` |
| `tune` | improve the performance of a target | the `perf` block |

Legal combinations and how Step 2 runs them:

| `goals` | Step 2 |
|---|---|
| `[onboard]` | `trtllm-modeling-v2 --task` |
| `[onboard, tune]` | the same run with `perf.enabled: true`: tune is chained after delivery, on the delivered tree |
| `[add_features]` | `--task` starting from `model.base` (a gap today, section 8) |
| `[add_features, tune]` | as above, then tune on the delivered tree |
| `[tune]` | `trtllm-modeling-v2 tune` on `model.base` |

Rules the checks enforce:

- `onboard` and `add_features` are exclusive: one creates the target the other
  presupposes. A request that reads as both ("onboard X with MTP") is
  `onboard` with MTP in its features.
- Without `onboard`, `model.base` is required and must exist.
- With `tune` after a build goal, the tune stage measures the tree the build
  delivered, so its accuracy anchor is that build's accepted score unless
  `perf` names another.
- An empty set is an open question, not a default.

**`features`** lists capabilities with a status, parameters and the parallel
configs they must work on; section 5 defines them.

**`perf`** gains a metric, a goal and per-configuration constraints
(section 5.8). Today the tune stage keeps whatever beats noise and has no
definition of done.

**`notes`** is free text for code style, reference implementations,
reference kernels and infrastructure. It is passed to every role verbatim.
It may not change requirements; anything that would belongs in a field.

### 4.3 Provenance

Every value records where it came from: `user` (stated), `inferred` (read from
the checkpoint, the repository or the site) or `default`. The review summary
lists `inferred` and `default` values first: those are the ones the user did
not say and has to check.

`open_questions` holds what intake could not decide. Confirmation is refused
while it is non-empty.

## 5. Features

### 5.1 Capabilities and configurations

A feature appears in two places, and the two must not be confused:

- **Capability** (`features:`): the target must support it, and a build goal
  must prove that with a gate. "The target supports MTP."
- **Configuration** (`perf.configs[*].features`, and the switches the gates run
  with): a concrete serving setup that turns capabilities on or off. "Measure
  TEP4 with MTP on, CUDA graph on, at concurrency 1–16."

A configuration may only turn on what is a capability: one the build goal
delivers with `status: required`, or one `model.base` already has. Turning on
anything else is a `block` (section 6).

### 5.2 Entry shape

Every feature has the same shape, so intake, the checks and compile treat
them uniformly:

```yaml
features:
  <name>:
    status: required          # required | nice_to_have | out_of_scope
    params: {...}             # per feature, section 5.3
    members: [tep4]           # optional: the parallel configs it must work on
    accept: {...}             # optional: overrides the default acceptance rule
```

The short form `<name>: required` means `{status: required}` with default
params.

**Status**

| status | build goal | delivery | prompts |
|---|---|---|---|
| `required` | built and gated | blocked if its gate fails | listed as work |
| `nice_to_have` | built on idle resources within its budget share, never ahead of `required` work | not blocked; on failure the attempt moves to a side branch (section 5.7) | listed as work, marked optional |
| `out_of_scope` | not built | not checked | stated as out of scope, so an absent path is not read as a defect |

A feature the request did not mention:

- the checkpoint carries it (MTP weights, a vision tower, …) → an open
  question with a recommended answer. Intake never silently builds or drops a
  capability the weights have;
- the checkpoint does not carry it → `out_of_scope`, provenance `default`.

**`members`** defaults to every parallel config the feature applies to
(section 5.6). Narrowing it is how a user says "host offload only on DEP8". It compiles
to today's per-unit `members` mapping.

### 5.3 Catalog

One row per feature: its parameters, what intake reads from the checkpoint,
and how Step 2 proves it works. Step 2's existing feature names are in
brackets.

**Core.** These are runtime behaviours every target has. They need no
increment of their own, but the gates must run with them on, or the target is
never tested in the configuration it will serve.

| feature | params | inferred from | proof |
|---|---|---|---|
| `cuda_graph` | `batch_sizes` (default: powers of two up to `workload.max_batch_size`), `padding` | — | every gate runs with it; the captured sizes are the serving envelope the reviewer checks against |
| `overlap_scheduler` | — | — | every gate runs with it |
| `chunked_prefill` | `chunk_tokens` (`max_num_tokens`) | — | every gate runs with it. **Gap**: nothing ensures a gate prompt is longer than `chunk_tokens`, so the chunked path may never run; intake should add such a prompt |
| `kv_cache` | `dtype` (`auto` \| `fp8` \| `nvfp4`), `tokens_per_block`, `memory_fraction`, `host_cache_gb` | `hf_quant_config.kv_cache_quant_algo` | accuracy gate with the matching `discriminators` row |
| `prefix_reuse` | `workload` (a shared-prefix set) | sliding-window layers (`layer_types`, `sliding_window`) | **gap**: a hit-rate witness, plus outputs equal with reuse on and off |
| `long_context` | `length`, `benchmark` (`ruler` \| `longbench` \| `needle`), `rope_scaling` | `max_position_embeddings`, `rope_scaling` | **gap**: a long-context accuracy gate at `length` |

`kv_cache.dtype` is the engine's KV storage type. It is not `[kv_quant]`: that
feature moves the target to the checkpoint's own native KV formats (FP4
compressed KV, MXFP4 indexer keys, …) and is listed below.

**Speculative decoding.** At most one algorithm per configuration.

| feature | params | inferred from | proof |
|---|---|---|---|
| `speculative` | `algo` (`mtp` \| `eagle3` \| `dflash` \| `dspark`), `draft_len`, `drafter` (path, eagle3/dflash), `tree` (`linear` \| `dynamic`) | `num_nextn_predict_layers`, `mtp.*` tensors, `dspark_*` config keys, a drafter checkpoint next to the target | acceptance gate (acceptance length vs `=off`, or an absolute floor for a new architecture) |

`algo` is inferred, not taken from the request's wording. A request for "MTP"
on a checkpoint whose `mtp.*` stages are a DSpark drafter becomes `dspark`,
with the reason shown to the user.

**Model-structure features.** These exist only for some architectures.
Intake offers them when the checkpoint has the structure.

| feature | params | inferred from | proof |
|---|---|---|---|
| `kv_quant` [`kv_quant`] | `formats` per KV pool, `bytes_per_token` | checkpoint KV scales and formats | bytes-per-token witness |
| `decoder_replay` / `encoder_replay` [same] | `window` | SWA layers | work witness (accuracy alone passes an empty implementation) |
| `host_offload` [`host_offload`] | `tensors`, predicted `memory_delta` | a tensor too large to keep resident | residency witness |

**Serving topology**

| feature | params | inferred from | proof |
|---|---|---|---|
| `disaggregated` [`disagg`] | `ctx: {parallel, count}`, `gen: {parallel, count}`, `transfer` (`nixl` \| `ucx`) | per-sequence state beyond attention KV (mamba/linear-attention state) | equivalence witness: the same outputs as one engine |

**Multimodality**

| feature | params | inferred from | proof |
|---|---|---|---|
| `multimodal` [`multimodal`] | `modalities` (`image` \| `video` \| `audio`), `max_items_per_request`, `max_resolution` | `vision_config`, `audio_config`, `preprocessor_config.json` | MMMU-Pro (or the modality's benchmark), or the logit reference made with image prompts (`onboard/logit_reference.py --image`): one of them judges the vision path, or preflight stops on the uncovered cells; plus an image boot prompt |

`multimodal: out_of_scope` on a multimodal checkpoint compiles to
`scope: text_only`.

**Serving-API features.** Guided decoding, logprobs, logits post-processors,
reasoning and tool-call parsers, LoRA. These are runtime features. They need
no modeling increment, but they can still break on a new target.

| feature | params | proof |
|---|---|---|
| `guided_decoding` | `backend` | **gap**: a serve smoke test with a schema |
| `logprobs` | `top_k` | **gap**: a serve smoke test |
| `reasoning_parser`, `tool_parser` | parser name | **gap**: a serve smoke test |
| `lora` | `max_rank`, `adapters` | **gap**: needs a modeling increment (LoRA layers), no skill exists |

**Custom features.** Anything else the user asks for:

```yaml
features:
  custom:
    - name: fused_qk_norm_rope
      description: fuse QK-norm and RoPE into one kernel in attention
      status: nice_to_have
      accept: {kind: perf, metric: tpot_ms, max_regression: 0, min_gain: 0.02}
```

A custom feature without an `accept` rule is an open question. "Done" has to
be checkable, otherwise an agent decides it. Allowed `accept.kind` values:
`accuracy` (a benchmark and floor), `perf` (a metric and gain), `test`
(a pytest node id that must pass), `review` (a human signs off at delivery).
Today custom features compile to `notes` plus a gap.

### 5.4 From the request to features

The words in a request do not map one-to-one onto features. Intake normalizes
them and records the mapping in provenance, so the user can see each
translation:

| the request says | becomes | watch out for |
|---|---|---|
| "MTP", "投机解码", "speculative" | `speculative.algo` from the checkpoint | `mtp.*` may be a DSpark drafter; no algo → ask if several are possible |
| "Eagle", "DFlash" | `speculative.algo` + `drafter` path | the drafter checkpoint must exist |
| "FP8 KV", "KV 量化" | `kv_cache.dtype: fp8` | not `kv_quant`, unless the checkpoint ships native KV formats |
| "PD 分离", "disagg" | `disaggregated` | ctx/gen parallelism are separate questions |
| "EP", "attention DP", "TEP/DEP" | `parallelism`, not a feature | — |
| "EPLB", "DWDP" | not a feature: no target exercises them, so there is nothing to build from | reported as out of scope |
| "128K 长上下文" | `long_context.length: 131072` | compare against `max_position_embeddings` and RoPE scaling |
| "前缀缓存", "prefix caching" | `prefix_reuse` | SWA models need `encoder_replay` for reuse to help |
| "VLM", "多模态", "图片输入" | `multimodal.modalities` from the config | "text only" → `out_of_scope` |
| "CUDA graph", "overlap" | core switches | — |
| "像 vLLM 那样实现" | `notes` + `model.reference_impl` | the reference may also be embargoed |

### 5.5 Compatibility

Rules, with their source. Each is a check in section 6. Breaking a rule
between two `required` features is a `block`. If either side is
`nice_to_have`, it is a `warn` and the `nice_to_have` side is dropped.

| rule | source |
|---|---|
| `speculative.algo: mtp` and `dspark` never both | `task_schema.MTP_NAMESPACE_FEATURES` |
| speculative × pipeline or helix parallelism: No | feature-combination-matrix |
| logits post-processor × disaggregated, × speculative: No | feature-combination-matrix |
| `multimodal: required` needs a vision/audio config in the checkpoint | definition |
| `long_context.length` ≤ `max_position_embeddings` × RoPE scaling factor | config |
| `prefix_reuse` on an SWA model without `encoder_replay`: reuse limited to the window | `trtllm-modeling-v2-bounded-replay` skill (warn) |
| an "Untested" cell in the matrix | warn; the user decides |
| architecture-specific limits (e.g. the CSA2 table at the top of the matrix) | applied when `model.architecture` matches |

The matrix is read from the document at intake time, not copied into code, so
the rules follow the doc. A small parser turns its table into `{(a, b): Yes |
No | Untested | Known issues}`.

### 5.6 Feature × parallel config

Not every feature applies to every member. Intake builds this table, shows it
in the report and compiles it to per-unit `members`:

```text
                 tp4    tep4   dep4
core             ✓      ✓      ✓
speculative      ✓      ✓      ✓
host_offload     –      –      ✓      (user narrowed)
disaggregated    ctx: tep4 ×1, gen: dep4 ×1
```

A `required` feature with no applicable member is a `block`, for example
`disaggregated` naming a member the run does not have.

### 5.7 Ordering and `nice_to_have`

Intake does not fix a build order. Step 2's planner decomposes and orders the
work freely ([docs/design.md, section 6](docs/design.md#6-the-task-graph-and-the-planner)).
Intake hands it only the hard dependencies that follow from the features
themselves:

- every feature needs a booting core on at least one member;
- a feature's witness compares against the target without it, so that
  target must exist on the same tree;
- `disaggregated`'s equivalence witness compares against the single-engine
  target with every other delivered feature on.

A `nice_to_have` feature:

- **runs only on idle resources**, within a budget share (default 20% of the
  run's budget). The orchestrator never gives it a device a ready `required`
  task could use;
- **is abandoned** when its share is spent, when one of its checks fails N
  times with no new hypothesis from the planner, or as soon as its change
  turns a passing `required` check red;
- **is moved to a side branch when abandoned.** The delivered tree does not
  contain it. Its commits go to `<branch>-<feature>-wip`, with the failing
  check's evidence, and the delivery report names the branch;
- **is reported as one of**: delivered; not delivered (reason, evidence, side
  branch); not started (budget).

**Gap**: Step 2 has no `nice_to_have` today. Every listed feature is required.

### 5.8 Perf configurations

`perf.configs` names the serving setups to tune. Each one picks a parallel
config, the features it turns on and a workload:

```yaml
perf:
  metric: output_tps_per_gpu
  baseline: stock
  configs:
    - name: low_latency
      parallel: tep4
      features: {cuda_graph: on, speculative: on}
      workload: {isl: 8192, osl: 1024, concurrency: [1, 4, 16]}
      constraint: {tpot_ms_max: 30}
    - name: max_throughput
      parallel: dep4
      features: {cuda_graph: on, speculative: off}
      workload: {isl: 8192, osl: 1024, concurrency: [128, 256]}
```

Checks:

- every `on` is a capability (section 5.1);
- the parallel config is a member;
- the workload fits in memory with these features (spec decoding adds draft
  KV; disaggregated splits memory per role);
- a multimodal workload names a dataset with media.

Each config compiles to one `perf.configs` entry: its workload, its
constraint, and its feature switches as `llm_api_options` over the serving
config derived from the workload. `perf.goal` compiles to
`perf.goal.ratio_to_reference` and/or `perf.goal.pct_of_sol` (a fraction of
the speed-of-light ceiling; README "Tuning"). **Gap**: the tune stage has no
disaggregated or multimodal measurement path, and measures every config at
the run's parallel configuration. Configs that need another are recorded
and warned.

## 6. Feasibility checks

Each check returns `pass`, `warn` or `block`, a one-line reason and a
suggested fix. A `block` turns into an open question.

| area | check | example `block` |
|---|---|---|
| checkpoint | `config.json` parses; architecture known; quantization readable | not a HF checkpoint |
| weights present | MTP layers (`num_nextn_predict_layers`), `dspark_*` keys, EAGLE3/DFlash drafter, vision tower | "MTP requested; checkpoint has no `mtp.*` weights" |
| MTP vs DSpark | `mtp.*` namespace read as one or the other, never both | "the `mtp.*` stages are a DSpark drafter; use `dspark`" |
| memory | weights / (tp·ep) + KV for `max_batch_size × max_seq_len` + activations ≤ HBM × fraction | "TP2 needs 212 GB/GPU; B200 has 180 GB — use TEP4 or FP8 KV" |
| divisibility | heads, kv heads % tp; experts % ep; world ≤ gpus_per_node × nodes | "8 KV heads cannot be split 16 ways" |
| quant × arch | NVFP4 needs sm_100+; KV dtype has a kernel on this arch | "NVFP4 on sm_90" |
| length | isl + osl and `long_context.length` ≤ `max_position_embeddings` and RoPE scaling | "131072 requested; config caps at 32768 without YaRN" |
| features | the section 5.5 rules; section 5.6 applicability; every unmentioned capability the checkpoint carries; every custom feature has `accept` | `mtp` and `dspark` both required; speculative with pipeline parallelism |
| perf configs | section 5.8: only capabilities turned on, member exists, fits in memory | `speculative: on` in a config, but speculative is `out_of_scope` |
| goals | `onboard` with a modeling_v2 target already present → ask `add_features`; `onboard` with an in-tree stock implementation → ask about embargo; no `onboard` and no `model.base`; `onboard` + `add_features` together | base target missing |
| accuracy | a reference row matches the discriminators, or a reference engine is available; protocol matches a reasoning model | "no gsm8k row for nvfp4 + mtp; measure with vllm?" |
| site | the cluster has this arch and enough GPUs; the container and python resolve | "site has sm_90 only" |
| budget | rough GPU-hours for the requested units and members vs `budget` | warn only |

**These checks share code with `preflight.py`.** Where a check exists there
(`check_checkpoint`, `check_speculation_support`, `check_accuracy_anchors`,
`check_multirank_substrate`), intake calls it; new checks go next to it so that
Step 2's preflight runs them too. Two copies of these rules would drift.

## 7. Interaction loop

```text
1. parse the request          → draft task.yaml, provenance = user
2. inspect checkpoint/repo/site → fill inferred fields
3. run feasibility checks     → blocks become open_questions
4. show intake.report.md:       requirements table, check table, questions
5. user answers / edits       → back to 2
6. open_questions empty and the user confirms
                              → write confirmed {by, at, sha256}
```

Rules:

- Ask only blocking questions, at most a handful per round, each with a
  recommended answer.
- Confirmation is refused while an open question or a block stands.
- Step 2 refuses a hand-written `task.yaml` that is unconfirmed, and any
  request whose content no longer matches its confirmation
  (`<task>.confirmed`, or a `confirmed` block in the file). A derived request
  runs unconfirmed (the fast path, above).
- If Step 2 finds a requirement it cannot meet (a feature that does not work
  on this checkpoint), it stops and hands back to intake. Agents may not
  downgrade a requirement on their own.
- A natural-language prompt given to Step 2 is an implementation hint only.

## 8. Compile: `task.yaml` + `site.yaml` → today's schema

| `task.yaml` | today's key |
|---|---|
| `goals` contains `tune` after a build goal | `perf.enabled: true` (tune chained after delivery) |
| `goals: [tune]` | the `tune` subcommand on `model.base` |
| `model.checkpoint` | `checkpoint_path` |
| `model.architecture` / segment override | `target.family`, `target.checkpoint_segment` |
| `device.arch` | `arch` |
| `parallelism` (one entry) | `parallel: <kind><world>` |
| `parallelism` (several) | `parallel: unified`, `members: [...]` |
| `workload.max_seq_len`, `max_batch_size` | `accuracy.max_seq_len`, `accuracy.max_batch_size`, `runtime.extra_flags` |
| `features.{cuda_graph, overlap_scheduler, chunked_prefill}` | `runtime.*`; `chunk_tokens`, `batch_sizes` → `runtime.extra_flags` and `accuracy.llm_api_options` |
| `features.prefix_reuse` | `runtime.kv_cache_reuse` |
| `features.kv_cache` | `accuracy.llm_api_options.kv_cache_config`, `accuracy.discriminators` |
| `features.speculative` | `features: [mtp \| eagle3 \| dflash \| dspark]`; the unit's (and every later unit's) `speculative_config` (`draft_len`, `drafter`) in its options file and `spec_dec_algo` in its discriminators |
| `features.{kv_quant, decoder_replay, encoder_replay, host_offload}` | `features: [<same name>]` + `accuracy.witness` where the feature has one |
| `features.disaggregated` | `features: [disagg]` + `accuracy.witness` |
| `features.multimodal` required / out of scope | `features: [multimodal]` / `scope: text_only` |
| `features.*.members` | per-unit `members` (`{default: [...], <unit>: [...]}`) |
| `model.quantization.weights`, `features.kv_cache.params.dtype` | `accuracy.discriminators` (`quant_algo`, `kv_cache_quant_algo`), unless `accuracy.discriminators` is stated |
| every unit | `accuracy.llm_api_options.<unit>`: an options file under `llm_api/` beside the compiled spec, with the runtime switches, `kv_cache_config` and `cuda_graph_config` the quickstart gates also take |
| `accuracy.boot`, `accuracy.witness` | the same keys, as they are |
| `perf.configs`, `perf.goal` | `perf.configs` (workload, constraint, `llm_api_options` with the config's `parallel` as `tensor_parallel_size`/`moe_expert_parallel_size`/`enable_attention_dp` and `speculative: on` as the run's `speculative_config`), `perf.goal`; without configs, one per member at the run's workload, with the draft |
| `accuracy.benchmark`, reference | `accuracy.benchmark`, `reference_yaml`, `discriminators`, `model_key` |
| `accuracy.protocol` | `accuracy.benchmark_args`, `accuracy.llm_api_options` |
| `perf` | `perf.enabled`, `perf.workload`, `perf.reference`, `perf.identity_configs` |
| `constraints.embargo` | `source_embargo`, `perf.source_embargo`, `perf.sister_engines` |
| `constraints.base_ref` | `commit_policy.base_ref` |
| `notes` | `notes`, appended verbatim to every role prompt |
| `constraints.budget` | `perf.campaign` limits (partial) |
| `site.yaml` | `environment`, `trtllm_repo_path`, `agents`, `perf.spools`, `perf.dev_spools`, `perf.env_script` |

Fields with **no consumer in Step 2 today**. Compile records them and warns,
by name (as it does for any feature parameter it does not compile,
`accuracy.reference`, `accuracy.protocol` keys other than `chat_template`,
`perf.metric`, a `perf.baseline` other than `stock`, `deliverable`,
`model.in_tree` and `model.reference_impl`); each needs a Step 2 change before
it is enforced:

- `features.long_context` — needs a long-context gate;
- `features.prefix_reuse` as a verified feature — needs a hit-rate witness;
- `chunked_prefill` — no gate prompt is guaranteed to exceed `chunk_tokens`;
- `status: nice_to_have` — Step 2 treats every feature as required;
- serving-API features and `custom` features — no gate;
- `perf.configs` that are disaggregated, multimodal, or at another parallel
  configuration — aggregate text-only measurement at the run's;
- `perf.metric` — the keep rule judges both headline metrics;
- `constraints.budget.gpu_hours` / `wall_days` — no run-wide budget exists;
- `goals: [add_features, ...]` with `model.base` — Step 2 has no "start from an
  existing target" entry; today a feature increment is built in the same run as
  core. `[onboard, tune]` and `[tune]` are already supported.

## 9. What this removes from prompts

Facts that are now decided in Step 1 and handed to agents rather than taught
to them:

- which speculator the checkpoint actually carries (MTP vs DSpark);
- `max_seq_len` / `max_batch_size` and why they must be explicit;
- the accuracy protocol for a reasoning model;
- which features are out of scope and why that is not a defect;
- the workload shapes that the serving envelope is judged against.

Once intake ships, the corresponding passages in `agents/prompts/` and `agents/skills/`
can be cut to a statement of the confirmed value. That list is to be built by
reading the prompts against section 8, as a follow-up.

## 10. Interface

```bash
trtllm-modeling-v2 --checkpoint C --site site.yaml --workspace W [--goal G] [--confirm [BY]]
                                                 # the fast path: derive, check in phase 0, run
trtllm-modeling-v2 intake derive --checkpoint C --site site.yaml [--goal G] -o task.yaml
                                                 # derive only; writes intake.report.md beside it
trtllm-modeling-v2 intake check task.yaml        # the feasibility checks; exit 1 on a block
trtllm-modeling-v2 intake confirm task.yaml --by NAME   # writes task.yaml.confirmed
trtllm-modeling-v2 intake compile task.yaml --site site.yaml -o task.resolved.yaml
trtllm-modeling-v2 --task task.yaml --site site.yaml --workspace W   # compiles, verifies, runs
```

`--task task.resolved.yaml` keeps working, so existing specs and runs are
unaffected.

Parsing a natural-language request is the only step that needs a model, and
it is not built: the goal line of the fast path stands in for it. Everything
else (inspection, checks, compile, confirmation) is deterministic code with
unit tests.

## 11. Implementation

Done: the `task.yaml` and `site.yaml` schemas; the fast path (`derive.py`);
compile with the section 8 table; the section 6 checks for the checkpoint,
weights present, MTP vs DSpark, divisibility and device capacity, quant x
arch, length, feature members, a hybrid checkpoint's block reuse, perf configs
and goals, run again in phase 0 with the compiled spec's anchor, speculation
and capture-list checks; confirmation beside the request.

Not done: the request parser and its interactive loop; the memory and
compatibility-matrix checks; the site and budget checks; Step 2 consumers for
the section 8 gaps, one at a time, each in its own change.

## 12. Open questions

- Is `task.yaml` per target, or may one request expand into several targets
  (one checkpoint, two archs)?
- Does `add_features` build on a delivered branch, or on a target already
  merged into `tensorrt_llm/_torch/_experimental/modeling_v2/`?
- Who owns `site.yaml`: one shared file per cluster in the repo, or a user
  file?
- Is the perf goal a pass/fail condition for delivery, or a stopping rule for
  the tune campaign only?
- With `[onboard, tune]`, does a perf goal the build cannot reach (the delivered
  target is far below baseline) stop the run before tune, or is tune expected
  to close any gap?
