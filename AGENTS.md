# AGENTS.md

TensorRT-LLM: open-source library for optimized LLM inference on NVIDIA GPUs.
Python and C++ codebase with PyTorch and AutoDeploy execution paths.

> If a `CLAUDE.local.md` file exists alongside this file, read and respect it — it contains developer-specific overrides that supplement this shared guidance.

## Rules (Read First)

**CRITICAL (YOU MUST):**
- Read and follow `CODING_GUIDELINES.md` for ALL code changes (C++ and Python)
- NVIDIA copyright header on ALL new files (update year on modified files)
- `git commit -s` (DCO sign-off required). Never attribute AI tools in sign-off line. Always rely on `git` to do the sign off instead of directly adding sign off in commit message.
- Do not add co-authors to the git commit message unless explicitly instructed to do so by the user.
- `pre-commit` hooks run on commit — if files are modified by hooks, re-stage and commit again
- LLM args or nested-config changes must run `python3 scripts/generate_llm_args_golden_manifest.py` and commit
  `tensorrt_llm/usage/llm_args_golden_manifest.json`; new fields require telemetry/privacy CODEOWNER approval
- When adding or renaming a public model architecture, update
  `tensorrt_llm/usage/architecture_allowlist.py` with its exact Hugging Face architecture name;
  never add private or customer-specific names
- PR title format: `[JIRA/NVBUG/None][type] description` (e.g., `[TRTLLM-5516][perf] optimize cuda graph padding`)
- Set `LLM_MODELS_ROOT` env var when running tests that need model weights

## Common Commands

| Task | Command |
|------|---------|
| Unit tests | `pytest tests/unittest/` |
| Specific test | `pytest tests/unittest/llmapi/test_llm_args.py` |
| Pattern match | `pytest tests/unittest -k "test_llm_args"` |
| Integration tests | `LLM_MODELS_ROOT=/path/to/models pytest tests/integration/defs/...` |
| Serve model | `trtllm-serve <hf_model> --port 8000` |
| Serve with config | `trtllm-serve <hf_model> --config config.yaml` |
| Benchmark | `trtllm-bench --model <hf_model> throughput --dataset <path>` |
| Find CI stage for test | `python scripts/test_to_stage_mapping.py --tests "test_name"` |

### Installation & Build

Building TensorRT-LLM requires Docker and may involve compiling C++ components.
See the [Installation Guide](docs/source/installation/installation-guide.md) for pre-built release containers and pip install,
[build from source](docs/source/installation/build-from-source.md) for development builds,
and [Container Images](docs/source/installation/containers.md) for information about the container images.

### Reference Configs

`examples/configs/database/` contains pareto-optimized serving configurations
across multiple models, GPUs, ISL/OSL combinations, and concurrency levels.
Use these as starting points for deployment and benchmarking rather than hand-tuning parameters.
See [deployment guides](docs/source/deployment-guide/) for model-specific walkthroughs.

## Architecture

See [architecture diagram](.github/tava_architecture_diagram.md) for the full Mermaid diagram.

### Backends

| Backend | Status | Entry Point | Key Path |
|---------|--------|-------------|----------|
| **PyTorch** | Default | `TorchLlmArgs` | `_torch/pyexecutor/` → `PyExecutor` → PyTorch Engine |
| **AutoDeploy** | Beta | `_torch/auto_deploy/` shim | `_torch/auto_deploy/shim/ad_executor.py` → adapts `PyExecutor` → graph transforms + torch.export |

### Shared C++ Core (via Nanobind)

Both backends share these C++ components:
- **Scheduling pipeline**: Scheduler → BatchManager (in-flight batching) → KV Cache Manager
- **Decoding pipeline**: Decoder (token generation orchestration) → Sampling

### Request Flow
```text
HuggingFace Model → LLM API → Executor (PyTorch/AutoDeploy)
    → Scheduler → Model Forward → Decoder → Sampling → Generated Tokens
```

### Serving
- `trtllm-serve`: OpenAI-compatible REST + gRPC server, supports all backends
- **Disaggregated serving**: separates prefill (context) and decode (generation) across GPUs
  - KV cache exchange via NIXL (default), UCX, or MPI

## Key Files

| File | Role |
|------|------|
| `tensorrt_llm/llmapi/llm.py` | Main API entry point |
| `tensorrt_llm/llmapi/llm_args.py` | Complete configuration schema (Pydantic) |
| `tensorrt_llm/llmapi/llm_utils.py` | Model loading, model-specific default overrides |
| `tensorrt_llm/models/modeling_utils.py` | Base classes for all models (`PretrainedConfig`, `PretrainedModel`) |
| `tensorrt_llm/executor/executor.py` | Execution abstraction (`GenerationExecutor`) |
| `tensorrt_llm/models/automodel.py` | Auto-discovery and model registry |
| `tensorrt_llm/_torch/models/` | PyTorch backend model implementations (distinct from the top-level `models/` package) |
| `tensorrt_llm/_torch/attention/ATTENTION_DEVELOPER_GUIDE.md` | Attention, MLA, backend families, sparse backends, metadata contracts, and KV-cache behavior - **read before modifying anything under `tensorrt_llm/_torch/attention/`** |
| `tensorrt_llm/_torch/moe/fused_moe/MOE_DEVELOPER_GUIDE.md` | MoE architecture, backends, communication, development patterns — **read before modifying MoE code** |
| `CODING_GUIDELINES.md` | C++ and Python coding standards (referenced throughout, must read before contributing) |

## Design Patterns

| Pattern | Key Points |
|---------|------------|
| **Config hierarchy** | `BaseLlmArgs` → `TorchLlmArgs`, model-specific defaults override generics, Pydantic validation |
| **Model architecture** | Each model: `Config` (inherits `PretrainedConfig`) + `ForCausalLM` (inherits `PretrainedModel`) |
| **Model defaults** | Architecture-specific overrides in `llm_utils.py` (attention kernels, quant, spec decoding, cache) |
| **Attention backends** | `TorchLlmArgs.attn_backend` selects kernel: `TRTLLM` (default), `FlashInfer`, `FlashAttention` |
| **Distributed execution** | Tensor/pipeline parallelism via `Mapping` class, multiple backends (MPI, Ray, RPC) |
| **Auto-discovery** | Models self-register via `automodel.py`, resolved by HF config `architectures` field |

## VisualGen

VisualGen is a vertical alongside LLM for Diffusion-Transformer (DiT)-based image/video generation
(text-to-image, text-to-video, image-to-video). It is **not** an LLM backend — it has
its own engine, args, params, and outputs — but shares ops and kernels with the
PyTorch backend where it makes sense (attention, quantization, parallelism).

Key entry points:
- Public Python API: `from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams`.
- Serving CLI: `trtllm-serve <HF id> --visual_gen_args <YAML path>`.

Key files:
- `tensorrt_llm/_torch/visual_gen/ENGINEERING_CRITERIA.md`: **Engineering criteria for any change under `tensorrt_llm/visual_gen/` or `tensorrt_llm/_torch/visual_gen/`** — API discipline, feature/test/lossy-vs-lossless requirements, examples & docs rules. Read before modifying anything in those trees.
- `tensorrt_llm/visual_gen/`: VisualGen public Python API. **User-facing surface — before modifying anything here, pause and confirm with the user that a public API change is actually intended; do not infer it from the surrounding task.**
- `tensorrt_llm/_torch/visual_gen/`: VisualGen internal implementation. All non-user-facing code belongs here.

## Anti-Patterns / Gotchas

- **Pre-commit modifies files in-place** — if hooks fail, files are already modified. Re-stage (`git add`) and commit again.
- **Protected APIs exist** — changes to LLM API signatures will fail `tests/unittest/api_stability` tests. Get code owner review.
- **Integration tests need GPUs + models** — always set `LLM_MODELS_ROOT` and ensure GPU access. Unit tests don't.
- **Copyright year** — update to current year when modifying existing files; add full header to new files.
- **Avoid broad exception handling** — catch specific exceptions, not bare `except:` (see `CODING_GUIDELINES.md`).
- **One concern per PR** — avoid scope creep. If a PR touches unrelated areas, split it.
- **User-facing configuration classes** - when editing or defining any user-facing configuration classes (particularly `BaseLlmArgs` or any class used in its fields), you **MUST** follow the Pydantic guidelines in `CODING_GUIDELINES.md`.

## Development Workflow

1. Set up build environment (see [installation docs](docs/source/installation/))
2. Make changes following `CODING_GUIDELINES.md`
3. Test locally with `pytest`

## Branching policy and PRs

- The main repository (`upstream`) is located at https://github.com/NVIDIA/TensorRT-LLM/
- Branches should always be pushed to the user-specified fork (usually `origin`)
- If pushing fails to due pre-push pre-commits hooks getting updated, just re-push immediately
- PRs should be opened on the main repository
   - Target `main` unless fixing a release branch bug
   - See `CONTRIBUTING.md` for full PR policies

### GitHub CLI authentication (`GH_CONFIG_DIR`)

The `gh` CLI uses `~/.config/gh` by default for authentication. Different GitHub hosts or forks may require a different config directory. **Before running any `gh` command** (e.g., `gh pr create`, `gh api`, `gh pr comment`):

1. Check if the user has specified a custom `GH_CONFIG_DIR` (e.g., in `CLAUDE.local.md` or environment). If so, use it.
2. If not explicitly set, default to `~/.config/gh`; do not ask for confirmation.
3. Prefix all `gh` commands with the resolved config dir: `GH_CONFIG_DIR=<path> gh ...`
4. If the command fails due to missing authentication or the wrong GitHub host/account, report the failure and ask for the correct `GH_CONFIG_DIR`.

## CI / Testing

See [CI overview](docs/source/developer-guide/ci-overview.md) for full details.

| Layer | Location | Notes |
|-------|----------|-------|
| Unit tests | `tests/unittest/` | Run in pre-merge CI; some tests require GPU |
| API stability | `tests/unittest/api_stability/` | Protects committed API signatures |
| Integration tests | `tests/integration/defs/` | Requires GPU + `LLM_MODELS_ROOT` |
| Test lists | `tests/integration/test_lists/test-db/` | Per-GPU YAML files (`l0_a10.yml`, `l0_h100.yml`, etc.) |
| Test waives | `tests/integration/test_lists/waives.txt` | Skip known-failing tests with NVBug links |
| Performance | See [benchmarking guide](docs/source/developer-guide/perf-benchmarking.md) | `trtllm-bench` and `trtllm-serve` benchmarks |

### Triggering CI

CI is triggered by posting comments on the PR. Basic commands:
- `/bot run` — trigger the standard CI pipeline
- `/bot run --disable-fail-fast` — run all stages even if earlier ones fail (only add when explicitly needed)

For a full list of up-to-date bot commands, post `/bot help` as a PR comment and check the bot's reply.

### Advisory semantic conflict review

The `CodeRabbit Semantic Conflict Review` workflow performs best-effort semantic
compatibility analysis for open, non-draft PRs targeting `main` or `release/**`.
No opt-in label is needed. It compares both branches from their merge base and
follows affected callers, contracts, configuration, and tests across files.

- PR creation, reopening, updates, and becoming ready evaluate the threshold;
  they do not automatically spend an AI call. An hourly scan also evaluates it.
- The first analysis needs new target commits and either 24 hours since the
  merge-base commit or at least 30 target commits beyond that base. After a
  completed PASS/FAIL analysis, count from its target SHA and completion time.
  PR updates invalidate the old verdict but do not bypass these thresholds.
- An authorized `ci: full pre-merge approved` label or enabling auto-merge
  bypasses the threshold. Approval-label authors are checked against the existing
  `trt-llm-ci-approvers` team using its existing token. All pre-merge requests
  share a one-hour cooldown, including when the SHA pair changes. A signal
  during cooldown is skipped; ordinary scans and the post-merge audit remain.
- Exact revision pairs, including requests still awaiting a reply, are deduplicated.
  With no completed analysis, new-pair routine requests wait 24 hours to avoid
  repeated service-failure retries. A missing reply for the same pair requires
  a manual retry. Workflow dispatch accepts one PR number and bypasses the
  thresholds, cooldown and deduplication for that PR only.
- Merge events request a post-merge audit regardless of thresholds or cooldown.
  The hourly scan recovers merges from the preceding 24 hours. The audit pins
  the actual merge commit and historical target, including for release PRs;
  later target updates do not invalidate it. Squash-only rules or the two-parent
  merge must establish the historical target; ambiguous rebase history is rejected.
  A pre-merge result/request can be reused only when its head/target pair and
  GitHub's recorded test-merge tree match the actual merged tree. Otherwise the
  audit includes the actual merged code in a new analysis request.

This dedicated custom check is `off` in `.coderabbit.yaml` during ordinary
reviews. The workflow explicitly requests `evaluate custom pre-merge check`
with warning mode and the configured instructions so regular reviews cannot
bypass the spending policy. CodeRabbit Custom Pre-Merge Checks access is needed.
Its native Post-Merge Actions only support the default branch; this workflow
uses an explicit command on the merged PR instead. Acceptance of Actions-bot
commands, especially on merged/release PRs, requires deployment validation.
An absent/rejected AI reply remains without a verdict, never a semantic pass.

The `Semantic conflict with target branch` Check starts neutral. Stale results
become neutral when the PR event or hourly scan observes a version change.
The verifier checks the bot identity, most recent trusted request, exact revision
record, and GitHub merge base. PASS becomes success; FAIL makes the Check and
publishing job red; Inconclusive remains neutral. A successful request job only
means orchestration succeeded. Checks on the actual merge SHA use the distinct
`Semantic conflict audit (post-merge)` name, with a receipt linking the analysis
on the original PR. Evidence includes code locations and regression scenarios.
The instructions first discover cross-branch interactions, then verify their
contracts, including test replacements and the production paths they exercise.
Explanations precede the machine record and must cite immutable source links
with full SHAs and line numbers from both head and target. A PASS/FAIL without
those citations becomes Inconclusive, including in the preview; an older PASS
cannot substitute for that incomplete reply. Citation presence does not prove
the AI's reasoning or the cited code is correct.

CodeRabbit can make mistakes, including false positives. Keep these checks and
workflows non-required: their failures then do not block merging. No required
waiting gate is added, and auto-merge does not wait for this analysis. Audit does
not revert code or modify branches. Repository rules remain unchanged.
The thresholds and cooldown limit frequency, not total calls per PR.

Changes to this automation run the separate read-only `CodeRabbit Semantic
Review Preview` workflow, including fork drafts. `Automation tests and result
lookup (not AI approval)` runs Node tests and reads actual CodeRabbit replies;
`AI verdict (advisory; skipped = unavailable)` runs only for a verified current
PASS/FAIL and is otherwise gray/skipped. `precommit-check.yml` is unchanged.

Both workflows use the same verifier. Privileged jobs load only trusted default
branch scripts, never PR code. The preview has read-only permissions. Before
merge, request `@coderabbitai evaluate custom pre-merge check` with the name
`Semantic conflict with target branch`, `--mode warning`, and `--instructions`
containing the configured instructions and fixed head/target/merge-base SHAs.
After the reply arrives, rerun the **tests and result lookup** job; rerunning only
the AI job reuses old outputs. Preview tests do not establish production trigger,
permission, command-acceptance or post-merge behavior.

### Trouble Shooting

- Use `TLLM_LOG_LEVEL_BY_MODULE` to enable per-module log filtering (e.g., `"debug:_torch,runtime;info:serve"`); see [Module-Level Logging](docs/source/developer-guide/overview.md#module-level-logging) for details.

## Key Documentation

| Topic | Path |
|-------|------|
| Coding guidelines | `CODING_GUIDELINES.md` |
| Architecture overview | `docs/source/developer-guide/overview.md` |
| PyTorch backend | `docs/source/torch/arch_overview.md` |
| Adding a new model | `docs/source/torch/adding_new_model.md` |
| AutoDeploy | `docs/source/features/auto_deploy/auto-deploy.md` |
| Disaggregated serving | `docs/source/features/disagg-serving.md` |
| Speculative decoding | `docs/source/features/speculative-decoding.md` |
| Quantization | `docs/source/features/quantization.md` |
| Parallelism strategies | `docs/source/features/parallel-strategy.md` |
| KV cache | `docs/source/features/kvcache.md` |
| API change guidelines | `docs/source/developer-guide/api-change.md` |
| Feature compatibility matrix | `docs/source/features/feature-combination-matrix.md` |
| Supported models | `docs/source/models/supported-models.md` |
| Deployment guides | `docs/source/deployment-guide/` |
| Examples & customization | `docs/source/examples/` |
| Performance analysis | `docs/source/developer-guide/perf-analysis.md` |
