<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Intake: from a request to a confirmed task

Intake (Step 1) turns what a user wants into a short `task.yaml` that the user
reads, corrects and confirms before any GPU or agent time is spent, and
compiles it with the cluster's `site.yaml` into the resolved spec the run
(Step 2) reads. Intake reads checkpoints, the repository and the site
description; it writes no modeling code and launches no GPU job.

Built (`spec/intake/`): the request and site schemas, the fast path, the
feasibility checks section 6 marks as built, confirmation and compilation. Not
built: a natural-language request parser (the fast path's goal line stands in
for it), the memory, compatibility-matrix, site and budget checks,
`nice_to_have` scheduling, and Step 2 consumers for the fields section 8
lists.

## 1. Three files

| file | written by | reviewed by the user | holds |
|---|---|---|---|
| `task.yaml` | intake, edited by the user | **yes** | requirements: model, device, parallelism, workload, features, accuracy, perf, constraints, notes |
| `site.yaml` | once per cluster | no | how the cluster runs it (section 3) |
| `task.resolved.yaml` | `compile`, never edited | no (audit only) | the full spec ([task.example.yaml](task.example.yaml)), the only file Step 2 reads |

## 2. The fast path

```bash
trtllm-modeling-v2 --checkpoint /models/newmodel --site site.yaml --workspace W \
    [--goal "mtp disagg@tp1/tep4 tp1 tep4 dep4 isl=8192 osl=1024 tune"] [--confirm]
```

derives the request in seconds (`spec/intake/derive.py`), writes it to
`W/task.intake.yaml` beside `W/intake.report.md` (what was read, what was
defaulted, what is still open), compiles it to `W/task.resolved.yaml` and
starts the run. The request's checks run in phase 0, first, together with the
file-reading checks of the compiled spec (each unit's and each variant's
accuracy anchor, the speculation each launch path is asked for, the capture
lists; `preflight.check_intake_request`), and a block stops the run with every
block listed.

| value | from |
|---|---|
| architecture, MoE or dense, weight and KV-cache quantization | `config.json`, `hf_quant_config.json` |
| speculative decoding (`mtp` or `dspark`) and its depth; multimodal | the checkpoint's draft layers and towers: built unless the goal says `-<feature>` or `no-<feature>` (`text_only` is `-multimodal`) |
| other features (`disagg`, `kv_quant`, `eagle3`, ...) | the goal only; `<feature>@tp1/tep4` gates one on those members only |
| `device.arch`, `gpus_per_node`, `nodes` | `arch=` in the goal, else `site.yaml` `device` |
| members | the goal's `tp1 tep4 ...`, else `site.yaml` `defaults.members`, else `tp1` |
| workload | the goal's `isl= osl= max_seq_len= max_batch_size= concurrency=1/16`, else `defaults.workload`, else isl 4096, osl 1024, max batch 64 (fitted to `max_position_embeddings`) |
| core switches | `cuda_graph`, `overlap_scheduler`, `chunked_prefill`, `prefix_reuse` required unless the goal says `-<switch>`; `chunk=<tokens>` sets the prefill chunk, and a hybrid checkpoint gets a 2048-token state snapshot interval for block reuse (`snapshot=`) |
| speculation and KV-cache parameters | `draft_len=`, `drafter=<dir>`, `kv=fp8`, `memory=0.6` |
| accuracy anchor | `benchmark= model_key= reference=<yaml>`, else `defaults.accuracy`, else gsm8k keyed by the config's `_name_or_path` |
| boot prompts | `defaults.boot`, else a checked `The capital of France is` -> `Paris` and one open prompt |
| witness declarations | `defaults.witness.<feature>`, and a replay window from `sliding_window`; what is still missing (a `disagg` witness's `contrast_command` and `contrast_env`, say) is an open question |
| `tune` in the goal | the `tune` goal |
| environment, sessions, agents, commit policy, perf spools | `site.yaml` |

Confirmation is recorded, not awaited (section 7). Re-running the command
resumes the request on disk; another `--goal` or `--checkpoint` is refused, and
`--clean` derives it again.

## 3. `site.yaml`

Written once per cluster: the block that does not change between tasks there,
kept out of what the user reads. It holds `trtllm_repo_path`, `environment`
(`local` or `session_spool`, with its sessions), `agents`, `perf` (`spools`,
`dev_spools`, `env_script`) and `commit_policy`, which compile copies into the
resolved spec unchanged; and, for the fast path, `device: {arch,
gpus_per_node, nodes}` and `defaults: {members, workload, accuracy, boot,
witness, profile}`.

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

profile: standard            # bringup | standard | certified: how much the contract owes

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

**`goals`** is a set: a request often asks for more than one thing ("onboard
it, then tune it"), so every goal heard is recorded.

| `goals` | Step 2 |
|---|---|
| `[onboard]` | `trtllm-modeling-v2 --task` |
| `[onboard, tune]` | the same run with `perf.enabled: true`: a `tune` task in the planned run, which starts once the core's inner checks pass |
| `[add_features]`, `[add_features, tune]` | not supported: Step 2 has no entry from an existing target, and compile warns that the run rebuilds core with the new features |
| `[tune]` | `trtllm-modeling-v2 tune` on `model.base` |

`onboard` and `add_features` are exclusive (a request that reads as both,
"onboard X with MTP", is `onboard` with MTP in its features); without
`onboard`, `model.base` is required; `tune` needs a workload or
`perf.configs`; an empty set is an open question, not a default.

**`features`** lists capabilities with a status, parameters and the parallel
configs they must work on (section 5). **`perf`** names the metric, the goal
and per-configuration constraints (section 5.8). **`notes`** is free text for
code style, reference implementations and infrastructure, passed to every role
verbatim; it may not change requirements, and anything that would belongs in
a field.

### 4.3 Provenance

Every value records where it came from: `user` (stated), `inferred` (read from
the checkpoint, the repository or the site) or `default`. Review the
`inferred` and `default` values first: those are the ones the user did not say.
`open_questions` holds what intake could not decide; confirmation is refused
while it is non-empty.

## 5. Features

### 5.1 Capabilities and configurations

A feature appears in two places:

- **Capability** (`features:`): the target must support it, and a build goal
  must prove that with a gate. "The target supports MTP."
- **Configuration** (`perf.configs[*].features`, and the switches the gates run
  with): a concrete serving setup that turns capabilities on or off. "Measure
  TEP4 with MTP on, CUDA graph on, at concurrency 1-16."

A configuration may only turn on a capability: one the build goal delivers
with `status: required`, or one `model.base` already has.

### 5.2 Entry shape

```yaml
features:
  <name>:
    status: required          # required | nice_to_have | out_of_scope
    params: {...}             # per feature, section 5.3
    members: [tep4]           # optional: the parallel configs it must work on
    accept: {...}             # optional: overrides the default acceptance rule
```

The short form `<name>: required` means `{status: required}` with default
params. An `out_of_scope` feature is not built or checked
(`multimodal: out_of_scope` compiles to `scope: text_only`, which every prompt
states). A feature the request did not mention is a block when the checkpoint
carries it (draft layers, a vision or audio tower), and `out_of_scope`
otherwise: intake never silently builds or drops a capability the weights
have. `members` defaults to every parallel
config the feature applies to, and compiles to the run's per-unit `members`.

### 5.3 Catalog

One row per feature: its parameters, what intake reads from the checkpoint,
and how Step 2 proves it works. Step 2's feature names are in brackets.

**Core.** Runtime behaviours every target has: no increment of their own, but
every gate runs with them on.

| feature | params | inferred from | proof |
|---|---|---|---|
| `cuda_graph` | `batch_sizes` (default: powers of two up to `workload.max_batch_size`), `padding` | — | every gate runs with it; the captured sizes are the serving envelope |
| `overlap_scheduler` | — | — | every gate runs with it |
| `chunked_prefill` | `chunk_tokens` (`max_num_tokens`) | — | every gate runs with it; a gate proves the chunked path ran only where its prompt is longer than `chunk_tokens` (a chunked variant's boot, the logit reference's long prompt) |
| `kv_cache` | `dtype` (`auto` \| `fp8` \| `nvfp4`), `memory_fraction` | `hf_quant_config.kv_cache_quant_algo` | accuracy gate with the matching `discriminators` row |
| `prefix_reuse` | `snapshot_interval` (a hybrid checkpoint's state snapshots) | sliding-window or recurrent layers | the reuse probe (block reuse attested at every gate that asks for it) |
| `long_context` | `length`, `benchmark` | `max_position_embeddings`, `rope_scaling` | **gap**: a long-context accuracy gate |

`kv_cache.dtype` is the engine's KV storage type. It is not `[kv_quant]`,
which moves the target to the checkpoint's own native KV formats.

**Speculative decoding.** At most one algorithm per configuration.

| feature | params | inferred from | proof |
|---|---|---|---|
| `speculative` | `algo` (`mtp` \| `eagle3` \| `dflash` \| `dspark`), `draft_len`, `drafter` (path, eagle3/dflash) | `num_nextn_predict_layers`, `mtp.*` tensors, `dspark_*` config keys | acceptance gate (acceptance length vs `=off`, or an absolute floor for a new architecture) |

`algo` is inferred, not taken from the request's wording: a request for "MTP"
on a checkpoint whose `mtp.*` stages are a DSpark drafter becomes `dspark`.

**Model-structure features**, offered when the checkpoint has the structure:

| feature | params | inferred from | proof |
|---|---|---|---|
| `kv_quant` [`kv_quant`] | `formats` per KV pool, `bytes_per_token` | checkpoint KV scales and formats | bytes-per-token witness |
| `decoder_replay` / `encoder_replay` [same] | `window` | SWA layers | work witness |
| `host_offload` [`host_offload`] | `tensors`, predicted memory delta | a tensor too large to keep resident | residency witness |
| `disaggregated` [`disagg`] | `ctx: {parallel, count}`, `gen: {parallel, count}`, `transfer` | per-sequence state beyond attention KV | equivalence witness: the same outputs as one engine |
| `multimodal` [`multimodal`] | `modalities` (`image` \| `video` \| `audio`) | `vision_config`, `audio_config`, `preprocessor_config.json` | a multimodal benchmark, or the logit reference made with image prompts; plus an image boot prompt |

`multimodal: out_of_scope` on a multimodal checkpoint compiles to
`scope: text_only`.

**Serving-API features** (`guided_decoding`, `logprobs`, `reasoning_parser`,
`tool_parser`, `lora`) and **custom features** (`features.custom`, each with
a `name`, `description`, `status` and an `accept` rule) are recorded with no
gate: compile warns, and a custom feature reaches the agents as a note.

Parameters compile only where section 8 says so; any other is recorded and
warned.

### 5.4 From a request's words to features

The words in a request do not map one-to-one onto features:

| the request says | becomes | watch out for |
|---|---|---|
| "MTP", "投机解码", "speculative" | `speculative.algo` from the checkpoint | `mtp.*` may be a DSpark drafter |
| "Eagle", "DFlash" | `speculative.algo` + `drafter` path | the drafter checkpoint must exist |
| "FP8 KV", "KV 量化" | `kv_cache.dtype: fp8` | not `kv_quant`, unless the checkpoint ships native KV formats |
| "PD 分离", "disagg" | `disaggregated` | ctx/gen parallelism are separate questions |
| "EP", "attention DP", "TEP/DEP" | `parallelism`, not a feature | — |
| "EPLB", "DWDP" | not a feature: no target exercises them | reported as out of scope |
| "128K 长上下文" | `long_context.length: 131072` | compare against `max_position_embeddings` and RoPE scaling |
| "前缀缓存", "prefix caching" | `prefix_reuse` | SWA models need `encoder_replay` for reuse to help |
| "VLM", "多模态", "图片输入" | `multimodal.modalities` from the config | "text only" -> `out_of_scope` |
| "CUDA graph", "overlap" | core switches | — |
| "像 vLLM 那样实现" | `notes` + `model.reference_impl` | the reference may also be embargoed |

### 5.5 Compatibility

Rules a request must respect. Only the first is checked (the resolved spec's
schema refuses it); the others are for the user to apply until a check reads
TensorRT-LLM's feature-combination matrix.

| rule | source |
|---|---|
| `speculative.algo: mtp` and `dspark` never both | `task_schema.MTP_NAMESPACE_FEATURES` |
| speculative × pipeline or helix parallelism: No | feature-combination-matrix |
| logits post-processor × disaggregated, × speculative: No | feature-combination-matrix |
| `multimodal: required` needs a vision/audio config in the checkpoint | definition (checked: section 6) |
| `long_context.length` ≤ `max_position_embeddings` × RoPE scaling factor | config (checked: section 6) |
| `prefix_reuse` on an SWA model without `encoder_replay`: reuse limited to the window | the `encoder_replay` feature |

### 5.6 Feature × parallel config

Not every feature applies to every member. A feature's `members` compile to
the run's per-unit `members`, and a feature narrowed to a member the run does
not have is a block.

### 5.7 Ordering and `nice_to_have`

Intake fixes no build order: Step 2's planner orders the work
([docs/design.md §6](docs/design.md#6-the-task-graph-and-the-planner)). A
`nice_to_have` feature compiles as required, with a warning: Step 2 has no
optional units, no budget share for one, and no side branch for an abandoned
one.

### 5.8 Perf configurations

`perf.configs` names the serving setups to tune. Each picks a parallel config,
the features it turns on and a workload:

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

A config may turn on only capabilities (section 5.1), and its parallel config
must be a member; both are checked. Each config compiles to one
`perf.configs` entry (section 8). The tune campaign measures aggregate,
text-only serving at the run's parallel configurations, so a disaggregated or
multimodal config, or one at another parallel configuration, is recorded and
warned.

## 6. Feasibility checks

Each check returns `pass`, `warn` or `block`, a one-line reason and a
suggested fix (`spec/intake/checks.py`, `intake check`). A `block` is an open
question: the task cannot be confirmed while one stands.

| area | check | example `block` | built |
|---|---|---|---|
| checkpoint | `config.json` parses; architecture known | not a HF checkpoint | yes |
| weights present | MTP layers (`num_nextn_predict_layers`), `dspark_*` keys, an EAGLE3/DFlash drafter, a vision tower; every capability the checkpoint carries is mentioned | "the checkpoint carries draft layers the request does not mention" | yes |
| MTP vs DSpark | `mtp.*` read as one or the other, never both | "the `mtp.*` stages are a DSpark drafter; use `dspark`" | yes |
| divisibility | heads and KV heads % tp; experts % ep; world ≤ gpus_per_node × nodes | "8 KV heads cannot be split 16 ways" | yes |
| quant × arch | NVFP4 needs sm_100+ | "NVFP4 on sm_90" | yes |
| length | isl + osl and `long_context.length` ≤ `max_position_embeddings` with RoPE scaling | "131072 requested; config caps at 32768 without YaRN" | yes |
| feature members | a narrowed feature names members the run has | "members tp8 are not among the run's tep4, dep4" | yes |
| block reuse | a hybrid checkpoint that asks for block reuse has a state snapshot interval | — | yes |
| perf configs | only capabilities turned on, parallel config is a member | `speculative: on` in a config, but speculative is `out_of_scope` | yes |
| goals | `onboard` and `add_features` together; no `onboard` and no `model.base`; `tune` without a workload | base target missing | yes |
| accuracy | a reference row matches the discriminators | "no gsm8k row for nvfp4 + mtp" | in phase 0, on the compiled spec |
| memory | weights / (tp·ep) + KV for `max_batch_size × max_seq_len` + activations ≤ HBM × fraction | "TP2 needs 212 GB/GPU; B200 has 180 GB" | no |
| site, budget | the cluster has this arch and enough GPUs; rough GPU-hours against `budget` | "site has sm_90 only" | no |

## 7. Confirmation

`trtllm-modeling-v2 intake confirm <request> --by NAME` (or `--confirm [BY]`
on the fast path) writes `{by, at, sha256}` to `<request>.confirmed` beside the
request, which is never rewritten; a `confirmed` block inside the request is
read too (`spec/intake/freeze.py`). Confirmation is refused while an open
question or a block stands.

- Step 2 refuses a hand-written request that is unconfirmed, and any request
  whose content no longer matches its confirmation.
- A derived request runs unconfirmed. Until the run is past preflight it may
  be edited and the same command re-run; after that,
  `<request>.started` holds the digest it started from, and an edit is refused
  until it is confirmed.
- A requirement Step 2 cannot meet ends the run `escalated`; agents may not
  downgrade a requirement, and changing the request is the user's step.
- A natural-language prompt given to Step 2 is an implementation hint only.

## 8. Compile: `task.yaml` + `site.yaml` → the resolved spec

| `task.yaml` | resolved spec |
|---|---|
| `goals` contains `tune` beside a build goal | `perf.enabled: true` (a `tune` task in the planned run) |
| `goals: [tune]` | the `tune` subcommand on `model.base` |
| `model.checkpoint` | `checkpoint_path` |
| `model.architecture` / segment override | `target.family`, `target.checkpoint_segment` |
| `device.arch` | `arch` |
| `parallelism` (one entry) | `parallel: <kind><world>` |
| `parallelism` (several) | `parallel: unified`, `members: [...]` |
| `workload.max_seq_len`, `max_batch_size` | `accuracy.max_seq_len`, `accuracy.max_batch_size`, `runtime.extra_flags` |
| `features.{cuda_graph, overlap_scheduler, chunked_prefill}` | `runtime.*`; `chunk_tokens` -> `--max_num_tokens` and the options' `max_num_tokens` |
| `features.prefix_reuse` | `runtime.kv_cache_reuse`; `snapshot_interval` -> the options' `kv_cache_config.mamba_state_config` |
| `features.kv_cache` | `dtype` -> `--kv_cache_dtype` and the options' `kv_cache_config`; `memory_fraction` -> `--kv_cache_fraction` and `free_gpu_memory_fraction`; the `discriminators` |
| `features.speculative` | `features: [mtp \| eagle3 \| dflash \| dspark]`; the unit's (and every later unit's) `speculative_config` (`draft_len`, `drafter`) in its options file and `spec_dec_algo` in its discriminators |
| `features.{kv_quant, decoder_replay, encoder_replay, host_offload}` | `features: [<same name>]` + `accuracy.witness` where the feature has one |
| `features.disaggregated` | `features: [disagg]` + `accuracy.witness` |
| `features.multimodal` required / out of scope | `features: [multimodal]` / `scope: text_only` |
| `features.*.members` | per-unit `members` (`{default: [...], <unit>: [...]}`) |
| `model.quantization.weights`, `features.kv_cache.params.dtype` | `accuracy.discriminators` (`quant_algo`, `kv_cache_quant_algo`), unless `accuracy.discriminators` is stated |
| every unit | `accuracy.llm_api_options.<unit>`: an options file under `llm_api/` beside the compiled spec, with the runtime switches, `kv_cache_config` and `cuda_graph_config` the quickstart gates also take |
| `accuracy.boot`, `accuracy.witness` | the same keys, as they are |
| `accuracy.benchmark`, reference | `accuracy.benchmark`, `reference_yaml`, `discriminators`, `model_key` |
| `accuracy.protocol.chat_template` | `accuracy.benchmark_args: [--apply_chat_template]` |
| `perf.configs`, `perf.goal` | `perf.configs` (workload, constraint, `llm_api_options` with the config's `parallel` as `tensor_parallel_size`/`moe_expert_parallel_size`/`enable_attention_dp` and `speculative: on` as the run's `speculative_config`), `perf.goal` (`ratio_to_reference` and/or `pct_of_sol`); without configs, one per member at the run's workload, with the draft |
| `perf` | `perf.enabled`, `perf.reference` (`stock`, else `none`) |
| `constraints.embargo` | `source_embargo` |
| `constraints.base_ref` | `commit_policy.base_ref` |
| `constraints.budget` | `budget`: `gpu_hours`, and `wall_days` as `wall_hours` |
| `notes` | `notes`, appended verbatim to every role prompt |
| `profile` | `profile` |
| `site.yaml` | `environment`, `trtllm_repo_path`, `agents`, `perf.spools`, `perf.dev_spools`, `perf.env_script`, `commit_policy` |

Compile validates its output with Step 2's own schema and prints a `warning:`
naming, by name, each requirement no consumer in Step 2 holds yet:

- `long_context`, the serving-API features and custom features: no gate;
- feature parameters section 8 does not compile, and `accept` rules;
- `status: nice_to_have`: built as required;
- `accuracy.reference` and `accuracy.protocol` keys other than
  `chat_template`;
- `perf.metric` (the keep rule judges both headline metrics), and a
  `perf.baseline` other than `stock`;
- `deliverable`, `model.in_tree` and `model.reference_impl` (say how the
  agents use them in `notes`);
- `goals: [add_features, ...]`: the run rebuilds core with the new features.

## 9. Interface

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

`--task task.resolved.yaml` runs a resolved spec as it is.
