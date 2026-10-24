<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# trtllm-modeling-v2: worked examples

One run followed from the request to the closing verdict, then a feature unit, a
configuration variant, a tune campaign, and the output of the driver, a reload
and a resume. Every file and line of output below has the shape the package's
own functions give it; scores, times, hashes and throughputs are illustrative,
and paths are placeholders (`/path/to/...`; the workspace is `/path/to/ws`).
[user-guide.md](user-guide.md) explains each command. One model's values for the
generic features (pool formats, replay windows, drafter constants) are in
[examples/](../examples/).

## 1. End to end: DeepSeek-R1-0528 NVFP4 on B200, with MTP, then tuned

The request: onboard the public NVFP4 checkpoint of DeepSeek-R1-0528 as a
modeling_v2 target on sm_100, unified over `tep4` and `dep4`, with its MTP head,
then tune it. TensorRT-LLM already has a target for this checkpoint at sm_103
and dep4; this run builds a new one, `r1_0528_nvfp4__sm_100__unified`, and
embargoes the stock implementation so the run derives its forward.

### 1.1 The request: `task.yaml` and `site.yaml`

The user-facing `task.yaml` ([INTAKE.md §4](../INTAKE.md#4-taskyaml)) states
requirements only:

```yaml
version: 1
goals: [onboard, tune]

model:
  checkpoint: /path/to/DeepSeek-R1-0528-FP4
  architecture: DeepseekV3ForCausalLM
  quantization: {weights: nvfp4, kv_cache: bf16}
  kind: moe
  in_tree: tensorrt_llm/_torch/models/modeling_deepseekv3.py
  target: {checkpoint_segment: r1_0528_nvfp4}

device: {arch: sm_100, gpus_per_node: 8, nodes: 1}

parallelism:
  - {kind: tep, world: 4}
  - {kind: dep, world: 4}

workload: {isl: 1024, osl: 1024, max_batch_size: 64, concurrency: [1, 16, 64]}

features:
  cuda_graph: required
  overlap_scheduler: required
  prefix_reuse: required
  chunked_prefill: out_of_scope
  speculative: {status: required, params: {algo: mtp, draft_len: 3}}

accuracy:
  benchmark: gsm8k
  protocol: {chat_template: true}
  model_key: deepseek-ai/DeepSeek-R1

perf:
  baseline: stock
  goal: {ratio_to_baseline: 1.1}

constraints:
  embargo: [tensorrt_llm/_torch/models/modeling_deepseekv3.py]
  base_ref: origin/main

notes: |
  Reuse the NVFP4 MoE and MLA entries the catalog already has; add an op
  under the target's own ops/ only where no catalog entry fits.

provenance:
  model.architecture: inferred
  features.speculative.algo: inferred
open_questions: []
```

`site.yaml` is written once per cluster and holds what does not change between
tasks: the checkout, the launch environment and its compute sessions, the perf
sessions.

```text
trtllm_repo_path: /path/to/TensorRT-LLM
environment:
  kind: session_spool
  python: /path/to/TensorRT-LLM/.venv/bin/python
  spool: /path/to/spool/gates-1
  partition: batch
  account: your-account
  container_image: /path/to/images/trtllm.sqsh
  container_name: trtllm-modeling-v2
  mounts: [/path/to:/path/to]
  gpus_per_node: 8
  env: {TRTLLM_MODELING_V2: require, TMPDIR: /path/to/scratch/tmp, CLAUDE_CODE_TMPDIR: /path/to/scratch/tmp}
  sessions:
    - {name: gates-1, spool: /path/to/spool/gates-1, role: gates, time_limit: "24:00:00", handover_lead: "01:00:00"}
    - {name: work-1, spool: /path/to/spool/work-1, role: work, time_limit: "24:00:00", handover_lead: "01:00:00"}
  session_wait: 1h
perf:
  spools: [/path/to/spool/perf-1, /path/to/spool/perf-2]
  dev_spools: [/path/to/spool/dev-1]
  env_script: /path/to/perf_env.sh
```

### 1.2 `intake check`, `confirm`, `compile`

`intake check` runs the feasibility checks against the checkpoint's
`config.json` and exits 1 on any block. A first draft that did not mention
speculation is refused, because the checkpoint carries a draft layer
(`num_nextn_predict_layers: 1`):

```console
$ trtllm-modeling-v2 intake check draft.yaml
[pass] checkpoint: architecture DeepseekV3ForCausalLM
[block] speculative: the checkpoint carries draft layers the request does not mention -- say whether speculative decoding is required or out_of_scope
```

With `speculative` stated, the checks pass (the divisibility of 128 heads and
256 experts over 4 ranks, and the length against `max_position_embeddings`
with YaRN, say nothing when they hold):

```console
$ trtllm-modeling-v2 intake check task.yaml
[pass] checkpoint: architecture DeepseekV3ForCausalLM
[pass] speculative: mtp weights present
$ trtllm-modeling-v2 intake confirm task.yaml --by "Jane Doe"
confirmed by Jane Doe at 2026-10-19T07:58:59Z (aeeb3b8ebe68)
$ trtllm-modeling-v2 intake compile task.yaml --site site.yaml -o task.resolved.yaml
wrote task.resolved.yaml
```

`confirm` writes `{by, at, sha256}` to `task.yaml.confirmed` beside the request,
which it never rewrites; a run refuses a `task.yaml` that is unconfirmed or whose
content no longer matches the digest. `compile` validates its output with the
run's own schema.

### 1.3 The resolved spec, and what a run needs beyond it

The compiled spec, abbreviated (the `environment` block is `site.yaml`'s):

```text
checkpoint_path: /path/to/DeepSeek-R1-0528-FP4
trtllm_repo_path: /path/to/TensorRT-LLM
arch: sm_100
parallel: unified
members: [tep4, dep4]
features: [mtp]
runtime: {cuda_graph: true, overlap_scheduler: true, kv_cache_reuse: true, chunked_prefill: false,
          extra_flags: [--max_seq_len=2048, --max_batch_size=64]}
accuracy: {benchmark: gsm8k, reference_yaml: tests/integration/defs/accuracy/references/gsm8k.yaml,
           model_key: deepseek-ai/DeepSeek-R1, benchmark_args: [--apply_chat_template],
           max_seq_len: 2048, max_batch_size: 64}
target: {checkpoint_segment: r1_0528_nvfp4}
source_embargo: [tensorrt_llm/_torch/models/modeling_deepseekv3.py]
perf: {enabled: true, reference: stock, workload: {isl: 1024, osl: 1024, concurrency: [1, 16, 64]},
       goal: {ratio_to_reference: 1.1}, spools: [...], dev_spools: [...], env_script: /path/to/perf_env.sh}
```

Intake does not carry everything a run of this checkpoint needs. The anchor
file holds only tagged rows for `deepseek-ai/DeepSeek-R1`, and the compiled spec
names no `discriminators`, so `trtllm-modeling-v2 --task task.yaml --site
site.yaml --dry-run` stops in preflight, before any agent:

```text
✘ preflight: accuracy anchor for unit `core` does not resolve, so the gate that ENDS this unit cannot run.

  unit:           core
  benchmark:      gsm8k
  reference_yaml: tests/integration/defs/accuracy/references/gsm8k.yaml
  model_key:      deepseek-ai/DeepSeek-R1
  discriminators: {}

  /path/to/TensorRT-LLM/tests/integration/defs/accuracy/references/gsm8k.yaml:deepseek-ai/DeepSeek-R1 has no entry matching {}. Available discriminator sets: [{'quant_algo': 'NVFP4'}, {'quant_algo': 'NVFP4', 'spec_dec_algo': 'MTP'}, {'quant_algo': 'NVFP4', 'kv_cache_quant_algo': 'FP8'}, ...]
```

The run therefore takes the compiled spec as a starting point and passes the
completed file with `--task` alone (a spec without `goals` and `version` is
used as it is). The additions: the row each unit is anchored to, each unit's
LLM-API options file (the `mtp` unit's carries the speculator, where the boot
and acceptance gates also read the draft depth), the acceptance workload, the
envelope boot, the serve gate turned off (the spec's `max_seq_len` turns it on
by default), and a KV-cache fraction. Relative option paths resolve against the
spec's own directory.

```yaml
accuracy:
  discriminators:
    core: {quant_algo: NVFP4}
    mtp: {quant_algo: NVFP4, spec_dec_algo: MTP}
  llm_api_options:
    core: scripts/core_llm_api.yaml
    mtp: scripts/mtp_llm_api.yaml
  acceptance:
    num_prompts: 32
    max_tokens: 256
    regression: {test: t, alpha: 0.05, min_effect: 0.1}
  envelope: {enabled: true}
  serve: {enabled: false}
```

```text
# scripts/mtp_llm_api.yaml (core_llm_api.yaml is the same without speculative_config)
kv_cache_config: {free_gpu_memory_fraction: 0.8, enable_block_reuse: true}
cuda_graph_config: {enable_padding: true, max_batch_size: 64}
enable_chunked_prefill: false
speculative_config: {decoding_type: MTP, max_draft_len: 3}
```

and `--kv_cache_fraction=0.8` appended to `runtime.extra_flags`. The perf block
needs completing too; section 4 shows it.

### 1.4 `--dry-run`

```console
$ trtllm-modeling-v2 --task task.full.yaml --workspace /path/to/ws --dry-run
─────────────────── preflight ───────────────────
✔ preflight
slug:        deepseek-r1-0528-fp4-sm_100-unified
branch:      trtllm-modeling-v2/deepseek-r1-0528-fp4-sm_100-unified
target:      tensorrt_llm/_torch/_experimental/modeling_v2/models/deepseek_v3/r1_0528_nvfp4__sm_100__unified
world size:  4
devices:     (allocated by Slurm)
launch:      session_spool
prefix:      CLAUDE_CODE_TMPDIR=/path/to/scratch/tmp TMPDIR=/path/to/scratch/tmp TRTLLM_MODELING_V2=require
interpreter: /path/to/TensorRT-LLM/.venv/bin/python
notes:
  - unified segment: world size 4, the widest of the member set (tep4, dep4)
  - members: tep4, dep4 -> 2 x (boot + accuracy) per unit, 4 accuracy gate(s) over 2 unit(s): core, mtp
  - accuracy anchor [core]: gsm8k deepseek-ai/DeepSeek-R1 {'quant_algo': 'NVFP4'} = 95.42 (tests/integration/defs/accuracy/references/gsm8k.yaml)
  - accuracy anchor [mtp]: gsm8k deepseek-ai/DeepSeek-R1 {'quant_algo': 'NVFP4', 'spec_dec_algo': 'MTP'} = 95.42 (tests/integration/defs/accuracy/references/gsm8k.yaml)
  - speculation: MTP supported by the gates' launch paths
  - cuda graphs: core, mtp: 11 batches [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64], padded, at boot, acceptance, envelope and accuracy alike
  - session_spool: devices are the scheduler's to allocate, so preflight claims none and asserts no architecture locally. `arch: sm_100` is taken on trust here and is what the receipts will be keyed by.
✔ agent probe 1 agent(s) ran Bash in the workspace and read the checkout's HEAD: claude-code <model> (default effort): surveyor, kernel_writer, assembler, tuner, planner
✔ survey 6 structural demand(s), 3 unrecognized key(s); 0/8 entries certified on sm_100, 41 unwrapped op(s), 12 FlashInfer callable(s) the engine does not call
  unrecognized structural keys: first_k_dense_replace, qk_nope_head_dim, qk_rope_head_dim
→ --dry-run: preflight passed; stopping before the first agent turn.
```

The bill is in the second note: four accuracy gates of about an hour each. The
substrate gate does not run on a dry run; on the real run it starts beside the
first turns.

### 1.5 The contract

Preflight compiles `contract.yaml`, every check the run owes, and the journal
renders it:

| unit | gate | member | tier | required |
|---|---|---|---|---|
| core | boot | tep4 | inner | yes |
| core | boot | dep4 | inner | yes |
| core | accuracy[gsm8k] | tep4 | inner | yes |
| core | accuracy[gsm8k] | dep4 | inner | yes |
| mtp | boot | tep4 | inner | yes |
| mtp | boot | dep4 | inner | yes |
| mtp | accuracy[gsm8k] | tep4 | inner | yes |
| mtp | accuracy[gsm8k] | dep4 | inner | yes |
| mtp | acceptance | tep4 | proof | yes |
| mtp | acceptance | dep4 | proof | yes |
| mtp | envelope | tep4 | delivery | yes |
| mtp | envelope | dep4 | delivery | yes |

Boot and accuracy checks also carry an `equivalence` key in `contract.yaml`:
two checks of different units with the same key share a pass (`SHARED` in the
standing).

### 1.6 The task graph

The run starts with the survey (revision 1, the orchestrator's) beside the
planner's first turn. When the survey's `vocabulary.yaml` lands with six queue
rows, the orchestrator adds one kernel task per row (revision 3). Three of them
change code other models run, so it also adds a regression check that follows
them (revision 4), and tells the planner both:

```text
the orchestrator added regression, a regression check with no agent turn: once k1-thop-attention,
k2-nvfp4-gemm, k3-allgather land(s), it runs the modeling_v2 package tests the changes to shared
code can affect; a failure comes back to you
task survey wrote 6 vocabulary queue row(s); the orchestrator added a kernel task for 6 of them,
which start now: k1-thop-attention, k2-nvfp4-gemm, k3-allgather, k4-fp4-block-scale-moe-runner,
k5-noaux-tc-op, k6-mla-rope-generation; k1-thop-attention, k2-nvfp4-gemm, k3-allgather change
code other models run: a regression check follows them
```

The planner's next graph wires `core` to the rows it calls. It leaves the
regression check out, as it is the orchestrator's: the revision keeps it, and
the loop adds `routing`, which changes the family's routing, to its
dependencies. The graph the planner wrote (`graph.proposed.yaml`, revision 5),
with `T` standing for the target directory
`tensorrt_llm/_torch/_experimental/modeling_v2/models/deepseek_v3/r1_0528_nvfp4__sm_100__unified`
and `TT` for its test directory
`tests/unittest/_torch/modeling_v2/models/deepseek_v3/r1_0528_nvfp4__sm_100__unified`
(the file spells both out):

```text
tasks:
  - {id: survey, kind: survey, ref: T, resources: {gpus: 1, est_minutes: 120},
     goal: "map the checkpoint's computation steps onto the catalog: write plan.md and vocabulary.yaml"}
  - {id: routing, kind: assemble, ref: core, resources: {gpus: 0, est_minutes: 60},
     goal: "the target's package and its entry in the deepseek_v3 family routing; finished when the router
            selects the target for this checkpoint at sm_100 under TRTLLM_MODELING_V2=require",
     touches: [tensorrt_llm/_torch/_experimental/modeling_v2/models/deepseek_v3/routing.py, T/__init__.py]}
  - {id: k1-thop-attention, kind: kernel, ref: "1", resources: {gpus: 1, est_minutes: 60},
     goal: "verify `attention/thop_attention` as vocabulary queue row 1 says, with its GPU test driving the cells it certifies",
     touches: [tensorrt_llm/_torch/_experimental/modeling_v2/catalog/attention/thop_attention.py,
               tests/unittest/_torch/modeling_v2/attention/test_modeling_v2_thop_attention.py]}
  # k2-nvfp4-gemm (row 2) and k3-allgather (row 3, gpus: 4) verify catalog entries the same way;
  # k4-fp4-block-scale-moe-runner, k5-noaux-tc-op and k6-mla-rope-generation onboard the target's own
  # ops (est_minutes: 120), e.g. touches: [T/ops/moe/noaux_tc_op.py, TT/moe/test_modeling_v2_noaux_tc_op.py]
  - {id: core, kind: assemble, ref: core, resources: {gpus: 4, est_minutes: 480},
     goal: "the target's forward and weight loading at tep4; finished when it boots under modeling_v2=require
            and its gsm8k score clears the NVFP4 anchor's floor",
     depends_on: [routing, k1-thop-attention, k2-nvfp4-gemm, k4-fp4-block-scale-moe-runner, k5-noaux-tc-op,
                  k6-mla-rope-generation],
     touches: [T/modeling.py, T/weights.py], satisfies: ["core:boot@tep4", "core:accuracy[gsm8k]@tep4"]}
  - {id: core-dp, kind: assemble, ref: core, depends_on: [core, k3-allgather], touches: [T/modeling.py],
     goal: "the attention-DP branch of the core (dep4), with the gathers of k3; the tep4 checks keep passing",
     resources: {gpus: 4, est_minutes: 240}, satisfies: ["core:boot@dep4", "core:accuracy[gsm8k]@dep4"]}
  - {id: mtp, kind: feature, ref: mtp, depends_on: [core], touches: [T/modeling.py, T/weights.py],
     goal: "the MTP head at tep4, one layer replayed to depth 3 (MTP-Eagle); finished when acceptance
            against the stock path holds", resources: {gpus: 4, est_minutes: 360},
     satisfies: ["mtp:boot@tep4", "mtp:accuracy[gsm8k]@tep4", "mtp:acceptance@tep4"]}
  - {id: mtp-dp, kind: feature, ref: mtp, depends_on: [mtp, core-dp], touches: [T/modeling.py],
     goal: "the MTP head under attention DP (dep4)", resources: {gpus: 4, est_minutes: 240},
     satisfies: ["mtp:boot@dep4", "mtp:accuracy[gsm8k]@dep4", "mtp:acceptance@dep4"]}
  - {id: envelope, kind: check, depends_on: [mtp-dp], resources: {gpus: 0, est_minutes: 120},
     goal: "the envelope boot of the finished target at both members",
     satisfies: ["mtp:envelope@tep4", "mtp:envelope@dep4"]}
  - {id: tune, kind: tune, depends_on: [mtp-dp], touches: [], resources: {gpus: 0, est_minutes: 1440},
     goal: "the tune campaign with MTP on at both perf configs, on the target as it is when it starts,
            until perf.goal or the campaign's budget"}
```

`graph.yaml` holds the orchestrator's regression check beside them:

```text
  - {id: regression, kind: check, regression: true, resources: {gpus: 0, est_minutes: 60},
     depends_on: [k1-thop-attention, k2-nvfp4-gemm, k3-allgather, routing],
     goal: "run the modeling_v2 package tests the landed changes to shared code can affect
            (source-level gates, the catalog's and the other targets' GPU tests), against the results
            at the base"}
```

What follows from it:

- The envelope checks are claimed by the `check` task `envelope`, which runs
  them with no agent turn and never depends on `tune`, so correctness is
  delivered beside the campaign.
- The regression check starts once the four changes it follows have landed.
- `tune` has `gpus: 0` because every job of the campaign runs on the perf and
  dev sessions. It depends on `mtp-dp` because its perf configs measure with
  MTP on.
- `core-dp` and `mtp` both name `T/modeling.py`, so they never run at the same
  time; the kernel tasks, `routing` and the survey start together.
- Each task that changes the tree works in `worktrees/<id>` on the branch
  `mv2/deepseek-r1-0528-fp4-sm_100-unified/<id>`.

### 1.7 A receipt and a task envelope

When `k4-fp4-block-scale-moe-runner` lands, its receipt
`tasks/k4-fp4-block-scale-moe-runner.receipt.yaml` is read off the kernel writer's
decision file and dispatch report:

```yaml
task: k4-fp4-block-scale-moe-runner
kind: kernel
attempt: 1
decision: LANDED
summary: 'Onboarded ops/moe/fp4_block_scale_moe_runner at sm_100 from the tier-1 op torch.ops.trtllm.fp4_block_scale_moe_runner:
  3 cells (decode 1-64 tokens per rank, prefill up to 8192, both at EP4), each driven by its GPU test
  against the bf16 reference.'
commits:
- '8c41d2e7a0 [None][feat] modeling_v2: ops/moe/fp4_block_scale_moe_runner for the R1 NVFP4 target at
  sm_100'
- '1f0b93c4de [None][test] modeling_v2: GPU test of fp4_block_scale_moe_runner''s three cells'
entry: tensorrt_llm/_torch/_experimental/modeling_v2/models/deepseek_v3/r1_0528_nvfp4__sm_100__unified/ops/moe/fp4_block_scale_moe_runner.py
tests: 18 passed in 274.9s on sm_100 (1 GPU, spool work-1)
decision_file: /path/to/ws/decisions/core-kernel_writer-4.yaml
result_file: /path/to/ws/reports/09-kernel_writer-onboard-ops-moe-fp4-block-scale-moe-runner.md
notes:
- 'For the planner: the cells cover up to 256 tokens per rank (64 requests x 4 tokens under MTP depth
  3); a larger max_batch_size needs this row verified again.'
written_at: '2026-10-19T12:40:02+00:00'
```

When `mtp` starts, its envelope `tasks/mtp.json` is the task as the graph holds
it plus the receipt of each task it depends on:

```json
{
 "id": "mtp", "kind": "feature",
 "goal": "the MTP head at tep4, one layer replayed to depth 3 (MTP-Eagle); finished when acceptance against the stock path holds",
 "depends_on": ["core"],
 "touches": ["tensorrt_llm/.../r1_0528_nvfp4__sm_100__unified/modeling.py", "tensorrt_llm/.../r1_0528_nvfp4__sm_100__unified/weights.py"],
 "satisfies": ["mtp:boot@tep4", "mtp:accuracy[gsm8k]@tep4", "mtp:acceptance@tep4"],
 "priority": "required", "status": "pending", "attempts": 1, "ref": "mtp",
 "exclusive": false, "regression": false, "resources": {"gpus": 4, "est_minutes": 360},
 "receipts": [
  {"task": "core", "kind": "assemble", "attempt": 1, "decision": "UNIT_READY",
   "summary": "Assembled the core forward and weight loading at tep4 from the catalog and the target's own ops: 61 layers, MLA through thop_attention, the first 3 dense, then the NVFP4 MoE through ops/moe. ...",
   "commits": ["3d9e0c51b7 [None][feat] modeling_v2: deepseek_v3 R1 NVFP4 target at sm_100, core forward",
               "a72f4e18c0 [None][feat] modeling_v2: weight table of the R1 NVFP4 target at sm_100"],
   "decision_file": "/path/to/ws/decisions/core-assembler-1.yaml",
   "result_file": "/path/to/ws/reports/14-assembler-core.md",
   "notes": ["For the planner: dep4 needs the gathers k3 verifies; nothing in the forward assumes tep."],
   "written_at": "2026-10-20T09:41:07+00:00"}
 ]
}
```

### 1.8 The journal

A task's start is its worker's `dispatch` line; its end is its `integration`
line and the `gate` lines of the checks it claims. Excerpts of `journal.md`, in
order:

```text
- `2026-10-19T08:00:12+00:00` **run opened** — slug=deepseek-r1-0528-fp4-sm_100-unified
- `2026-10-19T08:00:41+00:00` **agent probe passed** — run because this is a new run; claude-code <model> (default effort): surveyor, kernel_writer, assembler, tuner, planner; `/path/to/ws/gates/agent-probe/pass.json`
- `2026-10-19T08:00:43+00:00` **phase 0** — the survey starts beside the planner's first turn
- `2026-10-19T08:00:44+00:00` **dispatch 01 · surveyor** — survey
- `2026-10-19T08:00:44+00:00` **dispatch 02 · planner** — planner
- `2026-10-19T08:09:57+00:00` **substrate gate passed** — run because this is a new run; `/path/to/ws/gates/substrate/pass.json`
- `2026-10-19T10:41:07+00:00` **integration** — k4-fp4-block-scale-moe-runner: worktree made sparse without a config: examples, tensorrt_llm, tests/integration, tests/test_common, tests/unittest/_torch, tests/unittest/utils, triton_kernels
- `2026-10-19T10:41:09+00:00` **dispatch 09 · kernel_writer** — onboard ops/moe/fp4_block_scale_moe_runner
- `2026-10-19T12:39:52+00:00` **integration** — k4-fp4-block-scale-moe-runner: integrated mv2/deepseek-r1-0528-fp4-sm_100-unified/k4-fp4-block-scale-moe-runner (2 commit(s))
- `2026-10-19T13:12:44+00:00` **planned** — [regression] regression starts on the changes to shared code landed since the last check: routing, k1-thop-attention, k2-nvfp4-gemm, k3-allgather
- `2026-10-19T13:51:20+00:00` **regression** — regression: CLEAN at 7c2a9e41d0 against 1e2d3c4b5a: 46 test(s) ran, 212 result(s) reused
- `2026-10-20T03:12:40+00:00` **dispatch 14 · assembler** — core
- `2026-10-20T09:40:58+00:00` **integration** — core: integrated mv2/deepseek-r1-0528-fp4-sm_100-unified/core (2 commit(s))
- `2026-10-20T09:48:13+00:00` **gate** — [PASS] boot@tep4: engine cold-started and produced 1 keyword check(s) in the generated texts for 2 prompt(s); workload: accuracy.boot [engine 10/10 fields as declared; reuse probe resumed at 64 of 96; min free 30.80 of 179.06 GiB] (412s) [tree v1.3.0rc6-231-g3d9e0c51b7]
- `2026-10-20T10:45:21+00:00` **gate** — [PASS] accuracy[gsm8k]@tep4: measured 95.2237 vs floor 94.2688 [reference 95.4200 - 2.00 x SE 0.58 (n=1319) = 94.2688] (margin +0.9549) [engine 1/1 fields as declared; reuse 1318/1319 requests, 843520 tokens (resumed at 640 x1318); min free 28.53 of 179.06 GiB] in 3388s [tree v1.3.0rc6-231-g3d9e0c51b7]
```

The `[engine ...]` segment is the attestation: every field of the engine's
configuration the harness read back agreed with the declaration. A field that
differs fails the gate, naming it:

```text
[FAIL] boot@tep4: the engine did not run the declared configuration -- kv_cache_dtype: declared fp8, engine auto (read from llm.args). [...] [engine 10/11 fields as declared, 1 DIFFER; reuse probe resumed at 64 of 96; min free 30.88 of 179.06 GiB] (401s) [tree v1.3.0rc6-262-g5b0e9d21c4]
```

Later, the delivery tier and the correctness milestone. `mtp-dp` has landed
(commit `5b0e9d21c4`), its checks and the `envelope` task's checks pass, and
every required check holds on one tree:

```text
- `2026-10-20T14:21:40+00:00` **gate** — [PASS] envelope@dep4: served 66 requests at max_batch_size 64, max_seq_len 2048, max_num_tokens 8192 (unchunked), cuda graph batches [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64], 2 of 2032 prompt tokens; 2 probe(s) agree with batch 1 on their first 4 tokens [engine 10/10 fields as declared; reuse probe resumed at 64 of 96; min free 12.58 of 179.06 GiB] (1306s) [tree v1.3.0rc6-262-g5b0e9d21c4]
- `2026-10-20T14:37:11+00:00` **CORRECTNESS DELIVERED 5b0e9d21c4a8f3e7d61b2c09a4e5f6d7c8b9a0e1** — 30.6 h after the run started (2026-10-19T08:00:12+00:00); all 12 required checks of the contract hold on it, the delivery tier's included; branch `r1-nvfp4-sm100-correct`
```

Full support holds, so the tune campaign (section 4) is told to wrap up. A gate
line names `<gate>@<member>`, not the unit; the unit of each verdict is in
`contract_status.jsonl`. Had a later re-check failed, the journal would say
`correctness stays delivered at 5b0e9d21c4: on the target as it is now mtp:acceptance@dep4 failed`.

### 1.9 The contract standing and the closing verdict

The planner's request opens with the standing. After `core`'s two tep4 checks:

```text
2 passed, 0 failed, 10 missing
- MISSING core:boot@dep4
- MISSING core:accuracy[gsm8k]@dep4
- MISSING mtp:boot@tep4
[...]
- MISSING mtp:envelope@dep4
```

Had a task landed after the milestone, the standing would say
`the target has moved since, and on it 0 required checks failed and 12 are stale or missing`
until the contract held again. The run's last journal sections:

```text
## 2026-10-21T01:56:02+00:00 — delivery checks

[ok] target:modeling.py: tensorrt_llm/_torch/_experimental/modeling_v2/models/deepseek_v3/r1_0528_nvfp4__sm_100__unified/modeling.py
[ok] target:prefill-decode: PrefillTarget and DecodeTarget define step_args and forward
[ok] catalog:attention/thop_attention.py: ThopAttention: 4 cell(s) on sm_100
[ok] ops:ops/moe/fp4_block_scale_moe_runner.py: Fp4BlockScaleMoeRunner: 3 cell(s) on sm_100
[...]
[ok] routing:target: routing.py references models.deepseek_v3.r1_0528_nvfp4__sm_100__unified
[ok] accuracy:reference: gsm8k.yaml carries an entry for deepseek-ai/DeepSeek-R1
[ok] tests:test-db: test_modeling_v2_deepseek_v3 is listed in l0_dgx_b200.yml
[ok] git:commit-hygiene: every commit since 1e2d3c4b5a under tensorrt_llm/, tests/, cpp/, examples/ is signed off by its author, with no co-author, symlink or placeholder

## 2026-10-21T01:56:04+00:00 — contract

12 passed, 0 failed, 0 missing

Correctness delivered at 5b0e9d21c4 (`r1-nvfp4-sm100-correct`), 30.6 h after the run started; the target is still there.

## 2026-10-21T02:03:17+00:00 — run closed

- verdict: **tuned**
- stage reached: `planned`
- human stops: **0**
- cells that hold: core@dep4, core@tep4, mtp@dep4, mtp@tep4; the most at once on `5b0e9d21c4` (branch `r1-nvfp4-sm100-best`)
- metrics: `/path/to/ws/metrics.json`

### Gates

[every gate summary line this process fired, in order]
```

`tuned` is `delivered` with the tune task finished.

## 2. Feature units: the MTP acceptance gate and the disagg equivalence witness

Boot and accuracy cannot see either feature: rejection sampling keeps a wrong
draft path's output correct, and a disaggregated pair that recomputes lost
state still answers fluently. Each unit therefore owes a third check.

**MTP.** The unit is declared by `features: [mtp]`, its options file carrying
`speculative_config` (1.3), and its anchor row
(`{quant_algo: NVFP4, spec_dec_algo: MTP}`). Every unit from `mtp` on owes
`acceptance` at every member. DeepSeek-R1 has a stock implementation, so the
gate runs the same 32 prompts under `TRTLLM_MODELING_V2=require` and `=off`
and compares them prompt by prompt; it fails below a ratio of 0.95 when the
paired test finds the drop significant, at or below 1.0, or on a significant
drop of at least `min_effect` against the previous tree's verdict:

```text
[PASS] acceptance@dep4: measured 2.5859 vs reference 2.5851 - tol 0.00 (margin +0.0009) -- acceptance_length 2.5859 vs stock 2.5851 (ratio 1.000, floor 0.95); paired over 32 prompt(s): mean 2.586 vs 2.589 (diff -0.002, SE 0.010); t -0.25 p=0.401, Wilcoxon W+=247 p=0.380; judged by t at alpha 0.05 with minimum effect 0.1; up/down/equal 13/19/0; no earlier verdict for this member and workload; this one is the baseline [require: per prompt [0] 2.77 (263 tok/95 it), [1] 2.68 (204 tok/76 it), [...], [31] 2.42 (213 tok/88 it); prompt mean 2.59, iteration-weighted 2.59 over 2837 generation iterations; off: per prompt [...]] [engine require: 10/10 fields as declared; reuse probe resumed at 64 of 96; min free 30.80 of 179.06 GiB | off: 10/10 fields as declared; reuse probe resumed at 64 of 96; min free 30.80 of 179.06 GiB] in 1874s [tree v1.3.0rc6-262-g5b0e9d21c4]
```

For an architecture with no stock path, the `=off` leg cannot load; declare
`accuracy.acceptance.<unit>.min_absolute` with its `floor_source`, or
`measured_reference`, and the verdict records the weaker claim.

**Disagg.** Added after `mtp`, the unit needs its witness, which the contract
owes only with witnesses on (`accuracy.witness.enabled: true`, or the
`certified` profile). Leg B is a launcher of the run's own (a context engine, a
generation engine, a proxy, a client), run as one process on twice the member's
devices; the block must name it in `contrast_command`, beside its
`contrast_env`, and a word of `contrast_command` naming a file beside the spec is
resolved against the spec's directory. `max_batch_size: 1` gives both legs one
schedule, and `prompt_tokens` adds prose prompts at the lengths a hop has edges
at; keep the longest under `max_seq_len`.

```yaml
features: [mtp, disagg]
accuracy:
  discriminators:
    disagg: {quant_algo: NVFP4, spec_dec_algo: MTP}
  witness:
    enabled: true
    disagg:
      contrast_env: {TRTLLM_DISAGG: "1"}
      contrast_command: /path/to/run/scripts/disagg_contrast.py
      prompt_tokens: [131, 1500]
      max_tokens: 64
      max_batch_size: 1
      timeout: 2700
```

(`llm_api_options.disagg` names the `mtp` options file.) The contract gains the
`disagg` unit's boot and accuracy, `equivalence` and `acceptance` at both
members (speculation stays on), and the envelope moves to `disagg`, the last
unit. At `dep4` leg B pins each hop to one attention-DP rank pair
(`MV2FLOW_WITNESS_RANK_PAIRS=0:0,1:1,2:2,3:3,0:1,1:2,2:3,3:0,0:2,1:3,2:0,3:1,0:3,1:0,2:1,3:2`),
the four prompts repeat until every pair is crossed, and before a pass leg B
runs the longest prompt once more with `MV2FLOW_WITNESS_ABLATE=state`, which
must change the continuation:

```text
[PASS] equivalence@dep4: 16 continuation(s) token-identical across the hop (compared as token ids; both legs at max_batch_size=1); every attention-DP rank pair crossed (16 pairs over 16 hops); with the received state destroyed (KV pages of request 3) continuation #3 (1500 words) first differed at token 1, so the witness sees a hop that loses state [engine aggregated: 10/10 fields as declared; reuse 12/16 requests, 7296 tokens (resumed at 1728 x3, 128 x3, 32 x6); min free 29.40 of 179.06 GiB] (2213s) [tree v1.3.0rc6-249-g1c7d3e9b40]
```

A continuation that stays identical under the ablation fails as
`INSENSITIVE, not a pass`; a hop without its `[i] Witness ranks:` line fails as
unpinned. The other witnesses (residency, kv_quant, replay) are declared the
same way ([design.md §5.2](design.md#52-gates-and-witnesses)).

## 3. A configuration variant: KV cache in FP8 on the core

A variant is a configuration a built unit must also serve. It builds nothing,
adds checks named by the variant, and runs them on the unit's target with an
overlay:

```yaml
variants:
  kv_fp8:
    unit: core
    members: [tep4]
    gates: [boot, accuracy]
    runtime: {extra_flags: [--kv_cache_dtype=fp8]}
    llm_api_options: {kv_cache_config: {dtype: fp8}}
    accuracy: {discriminators: {quant_algo: NVFP4, kv_cache_quant_algo: FP8}}
    note: KV cache stored as FP8
```

The contract gains two proof-tier checks, `kv_fp8:boot@tep4` and
`kv_fp8:accuracy[gsm8k]@tep4`, and the dry run's capture-list note covers
`kv_fp8 (core)`. The variant's boot runs core's flags plus `--kv_cache_dtype=fp8`,
so it declares eleven fields; its accuracy gate reads core's options file
merged with the overlay (written under `<workspace>/variants/`) and the row
`{quant_algo: NVFP4, kv_cache_quant_algo: FP8}`:

```text
[PASS] boot@tep4: engine cold-started and produced 1 keyword check(s) in the generated texts for 2 prompt(s); workload: accuracy.boot [engine 11/11 fields as declared; reuse probe resumed at 64 of 96; min free 30.88 of 179.06 GiB] (405s) [tree v1.3.0rc6-262-g5b0e9d21c4]
```

The planner claims the variant's checks with a `feature` task whose `ref` is
`kv_fp8`, after the unit it configures. A variant that passes on core's target
needs that task only to claim its checks; one that fails needs the target
changed while core's own checks keep passing.

## 4. Tuning

### 4.1 The perf block

The tune stage measures each perf config at the serving config derived from
its workload, with the config's `llm_api_options` over it. The derived config
carries no parallelism and no speculator, so a target that serves at four
ranks with MTP says so in each config:

```yaml
perf:
  enabled: true
  reference: stock
  spools: [/path/to/spool/perf-1, /path/to/spool/perf-2]
  dev_spools: [/path/to/spool/dev-1]
  env_script: /path/to/perf_env.sh
  noise_repeats: 2
  configs:
    - name: tep4
      workload: {isl: 1024, osl: 1024, concurrency: [1, 16]}
      llm_api_options:
        tensor_parallel_size: 4
        moe_expert_parallel_size: 4
        speculative_config: {decoding_type: MTP, max_draft_len: 3}
      constraint: {tpot_ms_max: 20}
    - name: dep4
      workload: {isl: 1024, osl: 1024, concurrency: [64]}
      llm_api_options:
        tensor_parallel_size: 4
        moe_expert_parallel_size: 4
        enable_attention_dp: true
        speculative_config: {decoding_type: MTP, max_draft_len: 3}
  goal: {ratio_to_reference: 1.1}
  sol: {enabled: true, gpu: b200}
  campaign:
    max_parallel_tuners: 4
    max_iterations: 12
    max_hours: 24
    accuracy: {unit: mtp, members: [tep4, dep4]}
```

The stage writes `tune/configs/serving-tep4.yaml`: `modeling_v2: require`,
`max_batch_size: 16`, `max_seq_len: 2051` (ISL + OSL + the draft length),
`max_num_tokens: 8192`, `enable_chunked_prefill: false`, CUDA graphs padded
over every batch from 1 to 16, then the three keys above. A site is one config
at one concurrency (`dep4.con64`); the headlines are tok/s/user at the lowest
concurrency and output tok/s/GPU at the highest.

### 4.2 Ground truth

A perf smoke serves each world size once, then the baseline, its repeats and
the stock reference (`modeling_v2: "off"`) are measured at once across the perf
nodes. The journal's `tune` lines:

```text
[tune] the perf smoke passed at world sizes [4]
[tune] noise floor: same-host spread up to 0.90% (dep4.con64.tok_s_user) over 2 runs; across hosts up to 2.70% (dep4.con64.output_tok_s_gpu), not a floor: a keep is judged on its own host
[tune] keep gate: the contract's gsm8k gate at mtp@tep4,dep4, anchor 95.4200 (reference 95.4200 - 2.00 x SE 0.58 (n=1319) = 94.2688), then the target's suite (5 paths)
```

With `noise_factor` 3 and `min_gain` 0.01, a WIN needs at least
`max(1%, 3 x spread)` on some headline and no loss past `max(1%, 3 x spread)`
on any: 1.8% at `tep4.con1`, 2.4% at `tep4.con16`, 2.7% at `dep4.con64`. Its
first token at `tep4.con1`, the one config here that runs concurrency 1, may
come at most 10% later than the control's (`perf.campaign.ttft_tolerance`).

### 4.3 A tuner request

One roadmap item becomes one candidate: a worktree, a branch, and a request
(`tune/candidates/<slug>/request.md`), which the tuner turn's prompt names
beside the `tune` tag's section. Its head and its last sections:

```text
# Tuner request: absorb-bmm-fp8

Issued 2026-10-21 01:07 UTC by the tune stage.

## Paths

| name | value |
|---|---|
| repo (main checkout, read-only for you) | `/path/to/TensorRT-LLM` |
| **your worktree** (edit here only) | `/path/to/ws/tune/worktrees/absorb-bmm-fp8`, branch `tune/3f9a1c/absorb-bmm-fp8` off `5b0e9d21c4` |
| target dir (inside the worktree) | `tensorrt_llm/_torch/_experimental/modeling_v2/models/deepseek_v3/r1_0528_nvfp4__sm_100__unified` |
| dev spools (1-GPU jobs) | `/path/to/spool/dev-1` |
| **decision file to write** | `/path/to/ws/tune/candidates/absorb-bmm-fp8/decision.yaml` |
[...]

## The hypothesis to pursue

The absorbed MLA projections (q_nope x W_UK, attn x W_UV) run as two bf16 BMMs per layer in decode; the profile puts them at 9% of a tep4 con1 step. An FP8 BMM with per-head scales halves their bytes.

## Deliverables

An FP8 entry under ops/gemm/ with its GPU test against the bf16 BMM, the forward switched to it, a micro-benchmark of both at the decode shapes of every perf config, decision.yaml.

## What the measurement reads from your decision

[...]
- `changed_members: [<perf config>, ...]`: the configs whose engine runs your change;
- `path_marker: "<text>"`: a line your change prints wherever its path is taken (once per rank,
  when the engine is built or the path first runs), and nowhere else.
```

The tuner answers `CANDIDATE`; Python measures it, stacked on the best tree, on
an idle node beside two controls of that node, and judges it.

### 4.4 A keep

`rope-append-fused` wins at every site. The keep gate (the contract's accuracy
gate at `mtp@tep4,dep4` on the tree a keep would make, plus the target's suite)
passes, and the candidate is cherry-picked onto the best branch:

```text
[campaign] rope-append-fused: WIN tep4.con1.tok_s_user +4.1% (>= 1.8%); tep4.con16.output_tok_s_gpu +3.6% (>= 2.4%); tep4.con16.tok_s_user +3.5% (>= 2.4%); dep4.con64.tok_s_user +3.2% (>= 2.7%); dep4.con64.output_tok_s_gpu +3.3% (>= 2.7%)
[campaign] KEEP rope-append-fused -> tune/best e4b1c0d9f2
```

Its ledger entry in `tune/perf_ledger.jsonl` (one JSON object per line, shown
wrapped):

```json
{"ts": "2026-10-20T17:31:09+0000", "kind": "candidate_kept", "slug": "rope-append-fused",
 "commits": ["e4b1c0d9f2a7"], "best": "e4b1c0d9f2a7",
 "deltas": {"tep4.con1.tok_s_user": 0.041, "tep4.con16.output_tok_s_gpu": 0.036, "tep4.con16.tok_s_user": 0.035,
            "dep4.con64.tok_s_user": 0.032, "dep4.con64.output_tok_s_gpu": 0.033},
 "gate": {"verdict": "PASS", "score": 95.07, "anchor": 95.42, "floor": 94.2688,
          "floor_rule": "reference 95.4200 - 2.00 x SE 0.58 (n=1319) = 94.2688",
          "scores": {"tep4": 95.3, "dep4": 95.07}, "unit": "mtp", "members": ["tep4", "dep4"], "benchmark": "gsm8k",
          "tests": "212 passed in 1903.2s", "ungated": false, "tree": "/path/to/ws/tune/worktrees/...", "commit": "e4b1c0d9f2a7...",
          "reason": "gsm8k at mtp: tep4 95.30, dep4 95.07 >= floor 94.27 (anchor 95.42); tests: 212 passed in 1903.2s",
          "parts": {"dep4": "PASS", "tep4": "PASS", "tests": "PASS"}},
 "ungated": false}
```

### 4.5 A LOSS split into a member-gated variant

`absorb-bmm-fp8` wins at both `tep4` sites and loses at `dep4.con64`:

```json
{"ts": "2026-10-20T19:02:44+0000", "kind": "candidate_e2e", "slug": "absorb-bmm-fp8",
 "candidate": "c-absorb-bmm-fp8", "control": "t-ctl-perf2-1+t-ctl-perf2-2",
 "deltas": {"tep4.con1.tok_s_user": 0.052, "tep4.con16.output_tok_s_gpu": 0.034, "tep4.con16.tok_s_user": 0.031,
            "dep4.con64.tok_s_user": -0.061, "dep4.con64.output_tok_s_gpu": -0.058},
 "verdict": "LOSS",
 "reason": "dep4.con64.tok_s_user -6.1% (<= -2.7%); dep4.con64.output_tok_s_gpu -5.8% (<= -2.7%); tep4.con1.tok_s_user +5.2% (>= 1.8%); tep4.con16.output_tok_s_gpu +3.4% (>= 2.4%); tep4.con16.tok_s_user +3.1% (>= 2.4%)",
 "losses": ["dep4.con64.tok_s_user", "dep4.con64.output_tok_s_gpu"], "broken": [], "inert": [], "inert_losses": [],
 "host": "perf-node-2", "controls": ["t-ctl-perf2-1", "t-ctl-perf2-2"], "samples": ["c-absorb-bmm-fp8"],
 "inert_why": "", "remeasured": "", "noise": {"spread": {"...": 0.009}, "samples": 2}}
```

The losses are more than twice their floor, so the LOSS stands without a
re-measure. Every losing site has attention DP on and no winning site does,
so the LOSS is a split:

```json
{"ts": "2026-10-20T19:02:44+0000", "kind": "candidate_split", "slug": "absorb-bmm-fp8", "shape": "split",
 "mode": "attention DP on", "condition": "mapping.enable_attention_dp", "terms": [["attention_dp", "==", true]],
 "members": ["dep4"], "losses": ["dep4.con64.tok_s_user", "dep4.con64.output_tok_s_gpu"],
 "wins": ["tep4.con1.tok_s_user", "tep4.con16.output_tok_s_gpu", "tep4.con16.tok_s_user"], "broken": [],
 "lost": ["dep4.con64"], "won": ["tep4.con1", "tep4.con16"], "region": ["dep4.con64"], "why": ""}
```

```text
[campaign] absorb-bmm-fp8: split: loses only at dep4.con64 (attention DP on), wins at tep4.con1, tep4.con16; a member-gated variant goes to a tuner
[campaign] absorb-bmm-fp8: member-gated variant absorb-bmm-fp8-gated asked for: the old path where mapping.enable_attention_dp
```

The variant's request is a candidate request whose hypothesis is the split:

```text
Kind: SPLIT, a member-gated variant of `absorb-bmm-fp8` (its decision: `/path/to/ws/tune/candidates/absorb-bmm-fp8/decision.yaml`).

`absorb-bmm-fp8` lost only where **attention DP on**: `dep4.con64.tok_s_user` -6.1%, `dep4.con64.output_tok_s_gpu` -5.8%.
It won elsewhere: `tep4.con1.tok_s_user` +5.2%, `tep4.con16.output_tok_s_gpu` +3.4%, `tep4.con16.tok_s_user` +3.1%. Members that lost: `dep4`; every site
(member.concurrency) the mode covers: `dep4.con64`.

Gate the same change on that mode. Where `mapping.enable_attention_dp`, the target runs exactly the path it
ran before `absorb-bmm-fp8`; everywhere else, `absorb-bmm-fp8`'s path, unchanged. [...]
```

The variant is measured, judged and gated like any candidate, and wins only if
it loses nowhere and holds the candidate's wins:

```text
[campaign] absorb-bmm-fp8-gated: WIN tep4.con1.tok_s_user +4.9% (>= 1.8%); tep4.con16.output_tok_s_gpu +3.3% (>= 2.4%); tep4.con16.tok_s_user +3.0% (>= 2.4%)
[campaign] absorb-bmm-fp8-gated: member-gated variant of absorb-bmm-fp8: WIN, neutral where attention DP on, every win of the candidate held
[campaign] KEEP absorb-bmm-fp8-gated -> tune/best 0c19ab7e44
```

The ledger records `split_variant` (`mode`, `condition`) and `variant_verdict`
(`win: true` and the same reason).

### 4.6 The summary

When the roadmaps are spent, the campaign ends, measures `t-final` and writes
`tune/summary.md`:

```text
[tune] the campaign ends (roadmaps): best 0c19ab7e44, kept ['rope-append-fused', 'absorb-bmm-fp8-gated']
[tune] what the delivered target does not carry: /path/to/ws/UNMERGED_OPTIMIZATIONS.md
```

```text
# Tune stage summary

### tep4

| label | modeling_v2 | commit | con lo tok/s/user | con lo TPOT ms | con lo TTFT ms | con hi tok/s/GPU | con hi TPOT ms | con hi TTFT ms | vs baseline |
|---|---|---|---|---|---|---|---|---|---|
| t-baseline | require | 5b0e9d21c4 | 212.40 | 4.71 | 61 | 487.2 | 8.21 | 140 | con1 +0.0%, con16 +0.0% |
| t-reference | off | 5b0e9d21c4 | 205.18 | 4.87 | 63 | 470.6 | 8.49 | 143 | con1 -3.4%, con16 -3.4% |
| t-final | require | 0c19ab7e44 | 232.37 | 4.30 | 59 | 521.8 | 7.68 | 136 | con1 +9.4%, con16 +7.1% |

### dep4

| label | modeling_v2 | commit | con lo tok/s/user | con lo TPOT ms | con lo TTFT ms | con hi tok/s/GPU | con hi TPOT ms | con hi TTFT ms | vs baseline |
|---|---|---|---|---|---|---|---|---|---|
| t-baseline | require | 5b0e9d21c4 | 58.30 | 17.10 | 410 | 932.6 | 17.10 | 410 | con64 +0.0%, con64 +0.0% |
| t-reference | off | 5b0e9d21c4 | 56.32 | 17.70 | 418 | 900.9 | 17.70 | 418 | con64 -3.4%, con64 -3.4% |
| t-final | require | 0c19ab7e44 | 60.34 | 16.52 | 402 | 964.3 | 16.52 | 402 | con64 +3.5%, con64 +3.4% |

Modeling kept: ['rope-append-fused', 'absorb-bmm-fp8-gated']

Goal: ratio_to_reference 1.1, not reached
[...]
```

The task ends `TUNED`, and the planner hears
``tuned from 5b0e9d21c4: best branch `tune/3f9a1c/best` at 0c19ab7e44, kept [...], goal not reached; summary `/path/to/ws/tune/summary.md` ``.
Nothing in the run merges that branch: `UNMERGED_OPTIMIZATIONS.md` lists the
two kept changes, each with its hypothesis, commits, measured deltas and keep
gate, for a human to merge.

## 5. The driver, a reload and a resume

**`driver wait` and `driver status`** under `--agent-transport external`, with
one turn handed out and two waiting:

```text
NEW 007-kernel_writer  kernel_writer: onboard ops/moe/noaux_tc_op
orchestrator: running (pid 41237 on <host>, heartbeat 9s ago)
in flight: 3/8, and 1 planner turn(s) beside the cap
005-surveyor     done                   surveyor   survey
006-planner          CLAIMED 4m             planner        planner
007-kernel_writer            WAITING 0m             kernel_writer          onboard ops/moe/noaux_tc_op
session gates-1  gates        heartbeat 4s running=1 pending=0
session work-1   work         heartbeat 6s running=2 pending=0
```

```bash
trtllm-modeling-v2 driver --workspace /path/to/ws prompt 007-kernel_writer --claim --by driver-a   # hand this to a subagent
trtllm-modeling-v2 driver --workspace /path/to/ws answer 007-kernel_writer --from result.md
```

**A reload** of new flow code:

```console
$ trtllm-modeling-v2 driver --workspace /path/to/ws reload
reload 3f1c9a0e requested: the orchestrator drains, then re-executes (waits past 1800s); `wait` returns RELOADED when its loop runs again
```

The journal then says
`reload — pid 41237 re-executed at 14:02:11 after draining 7m (waited for check mtp:boot@dep4 7m)`
and, from the new process, `restart (reload) — pid 41237 on <host> reloaded into this process; nothing was cut short`.

**A resume after a crash**, the same command run again: the resumed process
stands on a substrate gate that passed within 12 h in the same environment,
journals what the stop cost, re-runs the checks that were cut short and sets the
tasks that were running back to pending (their worktrees keep their work):

```text
- `2026-10-20T16:05:40+00:00` **run resumed** — slug=deepseek-r1-0528-fp4-sm_100-unified
- `2026-10-20T16:05:41+00:00` **substrate gate not run** — passed 8.0 h ago in this environment; `/path/to/ws/gates/substrate/pass.json`
- `2026-10-20T16:05:52+00:00` **restart (hard-stop)** — pid 41237 on <host>, started 2026-10-19 08:00:09, stopped without a reload; cut short: check mtp:accuracy[gsm8k]@dep4, turn assembler mtp-dp; check jobs it left queued or running, whose checks run again: mv2-chk-0412 (running in /path/to/spool/gates-1); tasks set back to pending: mtp-dp
```
