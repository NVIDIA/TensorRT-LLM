# AGENTS.md

TensorRT-LLM: open-source library for optimized LLM inference on NVIDIA GPUs.
Python and C++ codebase supporting TensorRT engine-based and PyTorch-based execution paths.

> If a `CLAUDE.local.md` file exists alongside this file, read and respect it — it contains developer-specific overrides that supplement this shared guidance.

## Rules (Read First)

**CRITICAL (YOU MUST):**
- Read and follow `CODING_GUIDELINES.md` for ALL code changes (C++ and Python)
- NVIDIA copyright header on ALL new files (update year on modified files)
- `git commit -s` (DCO sign-off required). Never attribute AI tools in sign-off line. Always rely on `git` to do the sign off instead of directly adding sign off in commit message.
- Do not add co-authors to the git commit message unless explicitly instructed to do so by the user.
- `pre-commit` hooks run on commit — if files are modified by hooks, re-stage and commit again
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
See [build from source](docs/source/installation/build-from-source-linux.md) for full instructions,
or [pip install](docs/source/installation/linux.md) for pre-built wheels.
For container images, see [NGC containers](docs/source/installation/containers.md).

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
| **TensorRT** | Legacy | `TrtLlmArgs` | `builder.py` → `trtllm.Executor` → TensorRT Engine |

### Shared C++ Core (via Nanobind)

Both PyTorch and TensorRT backends share these C++ components:
- **Scheduling pipeline**: Scheduler → BatchManager (in-flight batching) → KV Cache Manager
- **Decoding pipeline**: Decoder (token generation orchestration) → Sampling

### Request Flow
```text
HuggingFace Model → LLM API → Executor (PyTorch/AutoDeploy/TensorRT)
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
| `tensorrt_llm/_torch/models/` | PyTorch backend model implementations (distinct from `models/` used by TensorRT backend) |
| `tensorrt_llm/_torch/modules/ATTENTION_DEVELOPER_GUIDE.md` | Attention, MLA, backend families, sparse backends, metadata contracts, and KV-cache behavior - **read before modifying `tensorrt_llm/_torch/modules/attention.py` or `_torch/attention_backend/`** |
| `tensorrt_llm/_torch/modules/fused_moe/MOE_DEVELOPER_GUIDE.md` | MoE architecture, backends, communication, development patterns — **read before modifying MoE code** |
| `CODING_GUIDELINES.md` | C++ and Python coding standards (referenced throughout, must read before contributing) |

## Design Patterns

| Pattern | Key Points |
|---------|------------|
| **Config hierarchy** | `BaseLlmArgs` → `TrtLlmArgs` / `TorchLlmArgs`, model-specific defaults override generics, Pydantic validation |
| **Model architecture** | Each model: `Config` (inherits `PretrainedConfig`) + `ForCausalLM` (inherits `PretrainedModel`) |
| **Model defaults** | Architecture-specific overrides in `llm_utils.py` (attention kernels, quant, spec decoding, cache) |
| **Attention backends** | `TorchLlmArgs.attn_backend` selects kernel: `TRTLLM` (default), `FlashInfer`, `FlashAttention` |
| **Distributed execution** | Tensor/pipeline parallelism via `Mapping` class, multiple backends (MPI, Ray, RPC) |
| **Auto-discovery** | Models self-register via `automodel.py`, resolved by HF config `architectures` field |

## Anti-Patterns / Gotchas

- **Pre-commit modifies files in-place** — if hooks fail, files are already modified. Re-stage (`git add`) and commit again.
- **Protected APIs exist** — changes to LLM API signatures will fail `tests/unittest/api_stability` tests. Get code owner review.
- **Integration tests need GPUs + models** — always set `LLM_MODELS_ROOT` and ensure GPU access. Unit tests don't.
- **Copyright year** — update to current year when modifying existing files; add full header to new files.
- **Avoid broad exception handling** — catch specific exceptions, not bare `except:` (see `CODING_GUIDELINES.md`).
- **One concern per PR** — avoid scope creep. If a PR touches unrelated areas, split it.
- **User-facing configuration classes** - when editing or defining any user-facing configuration classes (particularly `BaseLlmArgs` or any class used in its fields), you **MUST** follow the Pydantic guidelines in `CODING_GUIDELINES.md`.
- **TensorRT backend is legacy** — `TrtLlmArgs` / `backend="tensorrt"` and all exclusive tooling (`trtllm-build`, `trtllm-refit`, `convert_checkpoint.py`, `ModelRunner*`) are legacy. Bug fixes OK; new features target PyTorch or AutoDeploy.

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
- `/bot run --extra-stage "DGX_B200-4_GPUs-AutoDeploy-1, DGX_H100-4_GPUs-AutoDeploy-1"` — include AutoDeploy CI stages (use for AutoDeploy-related PRs)

For a full list of up-to-date bot commands, post `/bot help` as a PR comment and check the bot's reply.

### Retrieving CI Test Failures from a PR

See the CI failure retrieval skill (`.claude/skills/ci-failure-retrieval/SKILL.md`) for step-by-step scripts to query Jenkins test results via the API.

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
