"""Shared prose blocks for modeling-bringup prompt extensions.

These blocks are adapted from ``trtllm-modeling-agent``'s prompts so the
agent_team agents share a consistent TensorRT-LLM bring-up vocabulary and policy.
Each ``*_extra.py`` imports the subset relevant to its role and composes
``SYSTEM_PROMPT_EXTENSION``.
"""

DOMAIN_PRIMING = """\
## TensorRT-LLM model bring-up frame

- Treat HuggingFace (HF) as the semantic source of truth and vLLM as an
  implementation reference. Verify whether cited TensorRT-LLM classes/functions are
  real definitions, wrappers, import aliases, or bindings before relying on them.
- For attention work, read
  `tensorrt_llm/_torch/modules/ATTENTION_DEVELOPER_GUIDE.md` and reason across
  module, backend, runtime contract, and KV-cache semantics.
- For MoE work, read
  `tensorrt_llm/_torch/modules/fused_moe/MOE_DEVELOPER_GUIDE.md` and reason
  across routing, expert parallelism, quantization, and fused-kernel contracts.
- Use `KVCacheManagerV2` for new-model bring-up. The TRTLLM and FlashInfer
  attention backends are both valid targets when the plan tests the selected
  backend with `KVCacheManagerV2`.
- Attention scope includes model code, attention modules/backends, runtime
  contracts, ModelConfig/cpp conversion, KV-cache allocation/shape logic, and
  focused tests. If the selected backend or runtime/cache path needs changes,
  plan and implement those changes inside the attention task; do not skip them
  just because the current backend/runtime support is missing.
- Do not read, cite, or use `auto_deploy/` as a technical source for modeling
  bring-up; it may only be mentioned as an excluded path. Do not edit
  `auto_deploy/` or `tests/.../auto_deploy/` paths.
- Runtime-adjacent edits are allowed when they are part of the current task.
- Pass-critical unit and focused parity tests must include CUDA/GPU execution.
  Do not treat skipped or CPU-only tests as pass evidence; if CUDA/GPU
  execution is unavailable, keep iterating or fail rather than declare success.
"""

REFERENCE_TEST_POLICY = """\
## Reference-test policy

- If the target model behavior is newer than TensorRT-LLM's pinned `transformers`
  version, copy the minimal HF/vLLM semantics needed by the test into local
  test helpers or golden functions.
- Do not use local `transformers` shims, monkeypatches, or environment-specific
  installed `transformers` imports as pass evidence.
"""

BUILD_VALIDATION_POLICY = """\
## Build-validation policy

- Python-only changes do not require a TensorRT-LLM rebuild before validation
  unless the local repo's own build rules say otherwise.
- C++/CUDA/header changes require a TensorRT-LLM rebuild before treating tests as
  pass evidence.
- CMake/`CMakeLists.txt` changes require a clean rebuild before treating tests
  as pass evidence.
- After a native rebuild, ensure the rebuilt package/wheel is the one used by
  the validation commands. Tests run against a stale extension or wheel are
  not pass evidence.
"""

VALIDATION_EVIDENCE_LABELS = """\
## Modeling-bringup evidence labels

Most snake_case names in this prompt bundle are evidence labels for
`acceptance-criteria.md`, status summaries, and final reports. In
`agent-flow` they are **not** built-in functions, pytest fixtures, scripts, or
automatic controller checks. Do not assume a test exists just because one of
these labels is named; create or identify the concrete command, test file,
script, and artifact that proves the labeled outcome in the target TensorRT-LLM
workspace. If an external harness already has a typed evidence kind with the
same spelling, reuse that spelling in reports.

`reference_tier` and `validation_tier` are the two schema metadata fields in
this group, not evidence labels. Their accepted values are closed sets:
`reference_tier` must be one of `static`, `minimal_golden`, `reduced_source`,
or `real_source`; `validation_tier` must be one of `static`, `unit`,
`integration`, or `real_runtime`.

- `source_activation_replay`: HF/source-model activations from the real
  checkpoint are captured at representative layer boundaries and replayed
  through the implemented TensorRT-LLM path for the same contract.
- `source_logit_replay`: short real prompts run through the HF/source model and
  the implemented TensorRT-LLM path; final logits and greedy-argmax token are
  compared under deterministic decoding.
- `generation_parity`: HF/source and TensorRT-LLM generate token by token for
  fixed prompts under deterministic decoding, with per-step token/logit
  comparison.
- `real_runtime`: evidence that the selected TensorRT-LLM runtime/backend path
  actually dispatched at representative or checkpoint-scale dimensions.
  `real_runtime` is also the `validation_tier` enum value for this tier. The
  enum value and evidence label refer to the same runtime-backed concept, but a
  criterion must still name the concrete command/artifact and selected runtime
  path; writing only `validation_tier=real_runtime` is not enough to prove a
  required `real_runtime` evidence item.
- `accuracy_canary`: a small deterministic slice of each configured accuracy
  benchmark that runs before the long benchmark to catch catastrophic
  regressions cheaply.
- `cuda_graph_hard_path`: proof that the enabled CUDA-graph configuration
  actually ran under graph capture/replay rather than silently falling back to a
  non-graph path.
- `reference_tier` and `validation_tier`: acceptance-criteria metadata fields
  describing reference quality and execution shape; they are typed fields with
  the accepted values listed above, and they are not test names.
"""

CONTAINER_BOOTSTRAP = """\
## Slurm container bootstrap for TensorRT-LLM bring-up

This task's `task.yaml` contains a `slurm-environment` section. Read
these fields from it and use them verbatim:

1. `slurm_partition` — the Slurm partition to submit GPU jobs to.
2. `docker_image` — the container image, typically an enroot/pyxis
   `.sqsh` image.
3. Top-level `trtllm_repo_path` — the TensorRT-LLM repo path on the
   Slurm host/login node. Treat this exact absolute path as the intended
   in-container path too: the Slurm job must bind-mount the host
   TensorRT-LLM checkout to the same path inside the container.

Do not invent a different partition, image, or repo path, and do not
silently fall back to a local non-Slurm run when `task.yaml` contains
`slurm-environment`. If any value is unusable at runtime, flag it as a
blocker for the human in the loop instead of guessing.

Every fresh container session that will run TensorRT-LLM code (tests,
builds, benchmarks, eval) must run inside the container after changing
directory to the mounted TensorRT-LLM checkout path from `task.yaml`:

```bash
cd <trtllm_repo_path>
./scripts/build_wheel.py --trt_root /usr/local/tensorrt --benchmarks --skip_building_wheel -f --nvtx --use_ccache
pip install -e .[devel]
curl -fsSL https://claude.ai/install.sh | bash
pip install claude-agent-sdk
```

- `--skip_building_wheel -f --use_ccache` performs an in-place native
  build whose object cache survives across container sessions; the
  bootstrap is still required every fresh container because the
  editable `pip install -e .[devel]` and the `claude-agent-sdk` install
  live inside the container's site-packages.
- The bootstrap is the SLURM/container equivalent of the
  build-validation policy's rebuild step. After bootstrap, re-run
  `./scripts/build_wheel.py ...` only when C++/CUDA/header/CMake files
  change (per the build-validation policy); Python-only edits are
  picked up automatically by the editable install.
- Wrap the bootstrap inside the same SLURM container invocation
  (typically `srun --partition=<slurm_partition>
  --container-image=<docker_image>
  --container-mounts=<trtllm_repo_path>:<trtllm_repo_path>,...` — using
  the partition, image, and same-path TensorRT-LLM bind mount from
  `task.yaml`) that runs your TensorRT-LLM command so the test executes
  in the same container session that was just bootstrapped. A bare
  TensorRT-LLM command on the login node, one inside a container that
  did not mount the source checkout, one that mounted it at a different
  path, or one that skipped the bootstrap is not valid pass evidence.
- The bootstrap's TRT root (`/usr/local/tensorrt`) and
  `claude-agent-sdk` install are fixed project conventions for
  modeling bring-up; do not parameterize or swap them based on
  per-task assumptions.
"""

DESIGN_REVIEW_POLICY = """\
## Design-review policy

- Build source semantics from atomic source-side contracts, not convenience
  bundles. A useful contract has one primary concern, one primary owner
  boundary, and one primary proof path. Split behavior when those differ
  across module, backend, runtime, or KV-cache ownership.
- Use concern labels such as `schedule_or_mask`, `tensor_geometry`,
  `pre_attention_transform`, `positional_encoding`, `projection_topology`,
  `kv_cache_contract`, and `output_semantics` as decomposition hints when
  splitting a contract — not as plan schema obligations.
- If variants or checkpoint configs can affect a contract, inspect the real
  variant inventory. Do not make family-wide claims unless grounded in that
  inventory; otherwise name the supported subset or say `Unknown`.
- When the model exposes a mismatch with TensorRT-LLM model, backend, runtime,
  cache, ModelConfig, or binding contracts, identify the mismatch, enumerate
  plausible directions, reject directions that make logical model semantics
  diverge from runtime/cache/backend contracts, and choose the direction that
  preserves a coherent end-to-end TensorRT-LLM abstraction.
- Prefer directions that keep existing TensorRT-LLM contracts composable with
  future large-feature combinations such as speculative decoding,
  disaggregated serving, chunked prefill, and scheduler/cache variants. Do not
  implement those features unless the task requires them; preserve the
  interfaces they need.
- **Less is exponentially more.** Do not add new schemas, runtime concepts,
  adapters, flags, or helper layers unless existing abstractions cannot
  express the source semantics. Prefer reusing, narrowing, or extending
  existing code over introducing another entity. When two directions are
  equally correct and composable, prefer the one with fewer files, fewer code
  paths, and fewer lines of code.
- Do not accept local encoding tricks, padding tricks, fake config values, or
  model-specific workarounds merely because focused tests or benchmarks pass.
  Benchmarks validate behavior; they do not justify an architecturally
  rejected direction.
- Treat validation as `validated`, `partially validated`, or `not validated`.
  `validated` requires an independent reference, hard-path coverage, and a
  mutation check or equivalent negative control when implementation changed.
  Passing tests prove agreement with their reference, not source-model parity.
- If an active or likely-active contract is `Unknown` and could change owner
  boundary, backend choice, runtime/cache contract, architecture direction,
  or proof path, resolve it before planning or treat it as blocking pass.
- **Layer separation:** Plan, implement, and evaluate are separate roles.
  Plan owns the architecture search space; implement executes inside that
  space (and may deviate from a specific *prescription* with documented
  evidence when the criteria still hold); evaluate judges fit. An
  architecture-level change the worker had to invent on its own — outside
  any direction `plan.md` enumerated — exceeds the layer boundary. The
  workflow has no programmatic re-plan stage, so flag that case as a hard
  blocker in the iteration summary for the human in the loop; do not
  silently absorb it as a routine deviation.
- Bring-up is **parity-first**; performance optimization is out of scope.
  Kernel-level additions default to a native torch op or an OpenAI Triton
  kernel. Do **not** touch `cpp/` (C++/CUDA/header/CMake): native rebuilds on
  every iteration collapse feedback density. Performance-tuned C++ kernels
  belong to a separate post-bring-up phase. The only acceptable exception is
  when Triton/torch ops fundamentally cannot express the required semantics;
  document this in the plan's architecture decision with the specific
  construct that fails. "The existing template uses C++" or "for consistency"
  is not a sufficient reason.
"""

ATTENTION_SCOPE = """\
## Attention-stage scope

- Focus on the target model's attention semantics: Q/K/V projection geometry,
  Q/K/V norms, positional encoding, masks/windows, KV-cache layout, attention
  backend dispatch, and the ModelConfig/runtime contracts needed by those
  paths.
- Non-attention modules needed for construction or parity tests may be minimal
  HF-style Python/Torch reference scaffolding. Keep them simple and local.
- Do not require production non-attention parity to pass attention. Full MLP,
  MoE/router, logits, and end-to-end model correctness belong to the
  full-model stage.
- Do not optimize or add fused MoE/CUTLASS/C++ kernels, global MPI/import
  behavior, distributed runtime behavior, or unrelated availability shims in
  the attention stage. Leave production MoE and full-model wiring to the
  full-model stage unless the issue directly changes attention correctness.
"""

ATTENTION_VALIDATION_POLICY = """\
## Attention-validation policy

- `real_runtime` evidence proves selected backend dispatch, but attention pass
  evidence also needs source-observable activation replay when source attention
  behavior can silently drift.
- Attention test plans must include pass-critical evidence labeled
  **source_activation_replay**: hook the HF source model with the real
  checkpoint to capture hidden states entering representative attention layers
  and run the same layer through the selected TensorRT-LLM attention backend
  with `KVCacheManagerV2`.
- Attention test plans must also include pass-critical evidence labeled
  **source_logit_replay** between focused layer replay and long accuracy
  benchmarks. Use short real prompts, the real checkpoint/source model, and the
  implemented TensorRT-LLM path to compare final logits and the greedy-argmax
  token. Decode deterministically (`temperature=0`, `top_k=1`, no sampling);
  compare HF and TensorRT-LLM logits via max-abs / cosine and require the
  greedy-argmax token to match.
- Attention test plans must also include pass-critical evidence labeled
  **generation_parity**: generate >=32 tokens per prompt for at least 5 fixed
  prompts with both the HF source model and the implemented TensorRT-LLM path
  under deterministic greedy decoding (`temperature=0`, `top_k=1`, no
  sampling), compare per-step logits, and require the greedy-argmax token to
  match at every step. This is the cheapest signal that catches KV-cache,
  attention-mask, and sampling regressions; single-step source_logit_replay
  does not. Cost target: a few minutes per run.
- Cover each active attention variant that changes behavior when multiple
  coexist (mask/window patterns, attention-scope variants). Include prefill
  and decode/cache-reuse coverage when the model supports generation.
- Compare against HF attention outputs, not only a local SDPA golden or random
  hidden-state reference. Report `max_abs`, `mean_abs`, cosine similarity, and
  the prompt/layer/config used.
- Include negative controls for wrong RoPE or position handling, wrong V or
  K=V materialization, wrong score scale, wrong mask/window behavior, and
  fake KV geometry when those contracts exist.
- Pass-critical attention CUDA/runtime unit or smoke tests must cover both
  `cuda_graph=false, overlap_scheduler=false` and
  `cuda_graph=true, overlap_scheduler=true` for the selected backend with
  `KVCacheManagerV2`. The enabled test must exercise the CUDA graph hard
  path (e.g. `CudaGraphConfig()`).
- A `cuda_graph=true` claim must mean every required kernel actually ran
  under graph capture/replay. Silent fallback to a non-graph path for any
  operator invalidates the hard-path evidence: it masks capture/replay bugs
  and degrades long-generation accuracy. When a kernel is known not to
  support the CUDA graph hard path, an explicit fallback is an
  architecture-level decision that belongs in `plan.md`. If the plan
  doesn't already cover it, surface the conflict as a hard blocker in the
  iteration summary — the workflow has no programmatic re-plan stage, so
  the human in the loop has to act on the summary out-of-band — rather
  than silently swallowing it inside the worker.
- Because CUDA graph is a required production path, every pass-critical
  attention validation test that executes model code must cover the CUDA graph
  matrix: one baseline run with `cuda_graph=false` and one enabled run with
  `cuda_graph=true` plus hard-path evidence. Static checks and native rebuild
  checks may be supplemental but cannot satisfy this requirement.
- Random hidden states, reduced configs, static checks, or local goldens are
  useful supporting evidence, but they cannot be the only pass-critical
  attention evidence.
"""

FULL_MODEL_SCOPE = """\
## Full-model-stage scope

- Complete the target model's full wiring after attention is in place,
  including embeddings, norms, MLP/MoE/router behavior, weight loading,
  logits, and end-to-end parity tests.
- Full-model test plans must include pass-critical evidence labeled
  **source_logit_replay** before relying on benchmark accuracy: short real
  prompts, real checkpoint, deterministic greedy decoding,
  greedy-argmax-token equality with HF, and reported logit max-abs/cosine.
- Full-model test plans must also include pass-critical evidence labeled
  **generation_parity**: HF and TensorRT-LLM each generate >=32 tokens for at
  least 5 fixed prompts using the real checkpoint under deterministic greedy
  decoding, with per-step token-equality assertions. This catches KV-cache,
  attention-mask, and sampling regressions that single-step replay misses, and
  runs in a few minutes — much cheaper than rerunning the full benchmark. When
  the full benchmark scores far below the gate while local replays pass, this
  is usually where the divergence appears.
- Both `source_logit_replay` and `generation_parity` must each cover the CUDA
  graph matrix: one baseline `cuda_graph=false` run and one enabled
  `cuda_graph=true` run with hard-path evidence (e.g. `CudaGraphConfig()`).
- MoE wiring and routing/runtime changes belong here when required for
  full-model correctness. Kernel additions follow the design-review
  Python-first rule (native torch op or OpenAI Triton).
"""

MOE_VALIDATION_POLICY = """\
## MoE-validation policy (when the target has active MoE)

- Do not rely only on small random-expert tests. Full-model MoE pass evidence
  needs source-observable activation replay when router or expert behavior
  can silently drift.
- MoE test plans must include pass-critical evidence labeled
  **source_activation_replay** for the router-plus-expert path: hook the HF
  source model with the real checkpoint to capture inputs entering
  representative MoE layer(s) and run the same path through the selected
  TensorRT-LLM MoE backend.
- Compare router logits, selected experts, source-defined routing weights
  after scaling/normalization (including `per_expert_scale` when present),
  expert outputs, and post-MoE residual or layer output when the source has
  one.
- Evidence must name the selected MoE backend (e.g. `CUTLASS`, `VANILLA`,
  `TRTLLMGen`), the activation implementation (e.g. exact GELU, tanh GELU,
  SwiGLU), the op path (e.g. `torch.ops.trtllm.fused_moe`, TRTLLMGen, or
  Python fallback), and whether native rebuild was required and used. A
  generic "MoE parity passed" statement is not enough.
- Include negative controls for wrong activation, wrong routing or scaling,
  wrong expert selection, wrong packed-weight layout, and wrong parallel
  sharding when those contracts exist.
- If the current machine exposes only one GPU, multi-GPU TP/EP/NCCL sharding
  tests are deferred environment coverage rather than pass-critical local
  evidence — but single-GPU CUDA backend tests, source replay, LLM API
  smoke, and configured accuracy gates remain required.
"""

ACCURACY_GATE_FRAMEWORK = """\
## Accuracy-gate framework

- Accuracy benchmarks and score thresholds come from the user spec
  (`task.yaml` / `acceptance-criteria.md`), not from the harness. Only the
  validation framework is fixed.
- For each configured benchmark, both the attention canary and the full-model
  full benchmark must cover the required `cuda_graph` and
  `overlap_scheduler` configurations:
  * baseline: `cuda_graph=false`, `overlap_scheduler=false`
  * enabled: `cuda_graph=true`, `overlap_scheduler=true`
- Run configured accuracy gates only after required focused CUDA/runtime,
  integration, and parity tests pass. If a required focused test is failing,
  skipped, or unexecuted, fix that first instead of using accuracy evidence
  as a substitute.
- Run baseline first. If the baseline score is below the configured
  threshold, keep fixing baseline before running the enabled configuration.
- Before running a long accuracy benchmark, run a short LLM API smoke for the
  same baseline and enabled configurations. The enabled smoke must exercise
  the CUDA graph hard path and should fail early if CUDA graph capture/replay
  or overlap scheduling is not actually wired.
- Between LLM API smoke and the long benchmark, run evidence labeled
  `accuracy_canary`: a short deterministic subset of each configured benchmark
  for the same baseline and enabled configurations. It is a cheap
  catastrophic-regression gate, not a substitute for the full benchmark in the
  full-model stage.
- The enabled configuration must exercise the CUDA graph hard path, not
  merely report `cuda_graph=true`. Use the repo's CUDA graph config path
  (e.g. `CudaGraphConfig()`) together with enabled overlap scheduling
  (e.g. `disable_overlap_scheduler=False` when that is the local API).
- Use `tests/integration/defs/accuracy/test_llm_api_pytorch.py` as the LLM
  API reference pattern when applicable.
- When reporting evidence, name the benchmark, score (when applicable),
  `cuda_graph`, `overlap_scheduler`, and `cuda_graph_hard_path` for enabled
  runs.

### Accuracy debugging methodology

When accuracy is lower than expected, stop treating the full dataset as
the main debugger. Use this smaller-loop method before broad rewrites or
another full benchmark run:

- Export output results from the reference path and the TensorRT-LLM path
  with prompt id/text, generated tokens, score labels, decoding config,
  `cuda_graph`, `overlap_scheduler`, and logit/layer artifacts when
  available.
- Find the `wrong` results, then trace them back to the small set of
  `bad prompts` that reproduce the failure under deterministic decoding.
- Debug and verify only that bad-prompt set first. Use per-layer comparison,
  source activation/logit replay, token-by-token generation comparison, and
  smaller reproductions that isolate the failing module, backend, runtime,
  KV-cache, or sampling contract.
- Keep iterating on those prompts until they are correct under the same
  decoding and hard-path configuration that failed.
- After the bad-prompt set is fixed, follow the same sequence as the initial
  gate: rerun the LLM API smoke, then the accuracy canary, then the full
  configured dataset for both baseline and enabled configurations.
"""

STATUS_DONE_TODO_RUBRIC = """\
## status.md `Done / TODO` section

In addition to the rolling-state sections (current status, execution
path, what was tried, pointers for the next step), `status.md` must
carry a `## Done / TODO` section that both the Coder and the Reviewer
keep current when they call `update_status`. It is the cheap signal
the next agent uses to pick up where this iteration left off and the
human uses to read out progress at a glance.

### Format (use these exact headings)

```markdown
## Done / TODO

### Done
- <acceptance-criteria item or plan sub-task>: closed in iteration <n> — <evidence>

### TODO
- <acceptance-criteria item or plan sub-task>: <blocker, planned next step>
```

### Rules

- Every `- [ ]` line in `acceptance-criteria.md`, plus any high-risk
  sub-task `plan.md` calls out but the criteria do not break out,
  appears in exactly one of `Done` or `TODO` every turn. No item
  silently disappears between iterations.
- An item moves to `Done` only with **executed evidence** observed
  this turn or a prior turn: a command that ran, a test that passed,
  a file that landed. "Code written but not run" stays in `TODO`.
- Each `Done` row cites the iteration that closed it and a concrete
  evidence pointer (command, log path, test name, file path).
- Each `TODO` row names the immediate blocker and the planned next
  move. "Source replay failing because RoPE base mismatch; next:
  rerun parity test with corrected base" is acceptable; "TBD" or
  bare item names without a next step are not.
- `status.md` is overwritten every turn. The Coder produces the first
  draft each iteration by carrying the prior status forward; the
  Reviewer adjusts the section post-review to reflect what was
  actually verified (e.g. demoting an item the Coder marked `Done` if
  the rerun failed, or promoting a `TODO` to `Done` if independent
  verification passed).
"""

TEST_COMMAND_CACHE = """\
## `test_command.md` — verified test command cache (Slurm-only)

**Slurm-only mechanism.** `test_command.md` exists **only** when
`task.yaml` contains `slurm-environment` (partition + container image +
repo path, per the container-bootstrap section). On local (non-Slurm)
hosts, **do not create, read, or maintain
`test_command.md`** — skip every rule in this section, run commands
directly, and rely on `progress.yaml` for run history. The cache pays
for itself only when each invocation carries a long `srun` / `sbatch`
wrapper that is expensive to reconstruct from memory; on a local box
the commands are short enough that caching is iteration noise.

The rest of this section assumes a Slurm environment. If `task.yaml`
does not contain `slurm-environment`, stop reading here.

`test_command.md` is a workspace-scoped cache of **verified test
commands** (test runs, benchmarks, evaluations) shared by the Coder,
Reviewer, and QA. Each record has a one-line **purpose** and the exact
**bash command** that has been confirmed to run successfully from the
Slurm login node. The file starts empty and is built up as the team
verifies commands. Reads and writes are equally welcome from any of
the three agents. Use `Edit`/`Write` directly — there is no dedicated
MCP tool for this file. Do not store secrets.

**Always read `test_command.md` first** before writing or running any
test/benchmark/eval command. If the entry you need is already there,
run it as-is. If it is missing or the file is empty, draft the command
(see the delegation rule below for TensorRT-LLM commands).

**On every command outcome, update the cache:**
- **Success** — append a new entry (or refine an existing one) with
  the exact command you ran plus a one-line purpose.
- **Failure** — diagnose and fix the command, rerun until it succeeds,
  then **overwrite** the matching entry with the corrected command. If
  the failing command came from `test_command.md`, the corrected
  version replaces it in place; never leave a known-broken command in
  the cache.

QA may also read and edit the cache, but the cache itself is **not a
spec**; it does not change the verdict QA owes against `task.yaml` and
the acceptance criteria, and editing it is not a substitute for
verifying acceptance criteria at runtime.

The team caches verified `trtllm-bench`, `trtllm-eval`, `srun` /
`sbatch` submissions, and related TensorRT-LLM bring-up commands here
so the next iteration reuses what already works instead of
re-drafting. The rules below tighten the protocol so the cache stays
small and scannable across many iterations.

- **Keep only currently-passing commands; delete everything else.** A
  cache entry exists only if (a) the command itself just succeeded with
  `rc=0` in the current iteration AND (b) it still satisfies the
  current acceptance criteria. When either condition stops holding,
  **delete the entire entry** — do not leave it with inline
  "superseded by ..." prose, do not move it to an `## Archive` section,
  do not keep a chain of historical attempts. Failed-attempt root
  causes, supersede chains, and per-iteration validation diaries belong
  in `progress.yaml` and git history, never in this file.
- **Per-entry template — three header lines, then the command.** Every
  entry MUST follow this exact shape, and no Reviewer/QA prose may be
  appended to it across iterations:

      ## <Short purpose, one line>
      criteria: <acceptance-criteria ids this command verifies, comma-separated>
      verified: <job-id or run-id, exit status, elapsed>
      outputs:  <key output files + headline metrics, one line>

      ```bash
      <single self-contained command>
      ```

  If a field would not fit on one line, shorten it rather than wrap.
  Do not append Reviewer iteration notes ("Reviewer iteration 13 re-ran
  ..."); rerun evidence goes in `progress.yaml`, and if the rerun
  changes the verified job/exit/outputs, overwrite the three header
  lines in place.
- **No top-of-file narrative.** Do not prepend supersede summaries,
  iteration histories, or "why iter-N was superseded by iter-M" prose
  to `test_command.md`. The file should open directly on the first
  entry.
- **Prefer specialist help for fresh command drafting.** When the task
  involves the TensorRT-LLM toolchain (any `trtllm-*` binary,
  slurm-based benchmark/eval, or related test infrastructure) and you
  need a fresh command, use any available test-command specialist or
  equivalent workflow support rather than constructing the command from
  memory. Hand over the relevant context (model, parallelism, dataset
  paths, ISL/OSL, Slurm partition, container image, etc.) and use the
  verified command it returns.
- **Every cached command must be a self-contained `srun` (or `sbatch`)
  invocation runnable directly from the login node.** It must include
  the partition / account / node-count / GPU / time / container / mount
  flags needed to launch on its own; it must not assume the agent is
  already inside a compute-node shell or an interactive allocation.
  Verify a command by running it from the login node before caching
  it. Bare `python …` / `trtllm-bench …` / `trtllm-eval …` lines
  without an `srun` wrapper must not be cached — if a command has no
  `srun` wrapper, the task is not a Slurm task and `test_command.md`
  should not exist in the first place.
- Do not store secrets or absolute paths under `/tmp` that will not
  survive the workspace.
"""

SOURCE_BOUNDARY = """\
## Source boundary

- Use only these local source roots:
  * The TensorRT-LLM repo root specified by the user spec
  * The current workspace
  * HF/vLLM reference files or directories explicitly named in the user spec
- Do not read or cite local files outside those roots.
- Do not read, cite, or use `auto_deploy/` as a technical source; it may only
  be mentioned as an excluded path.
- Do not use external agent prompts, skills, memories, or policy documents as
  technical sources unless the user explicitly names them in the spec.
- Evidence must cite HF/vLLM, TensorRT-LLM, current workspace artifacts, or the
  user spec — not external skill or agent documents.
"""
