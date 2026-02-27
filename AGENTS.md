# AGENTS.md

TensorRT-LLM: open-source library for optimized LLM inference on NVIDIA GPUs.
Python and C++ codebase supporting TensorRT engine-based and PyTorch-based execution paths.

> If a `CLAUDE.local.md` file exists alongside this file, read and respect it — it contains developer-specific overrides that supplement this shared guidance.

## Rules (Read First)

**CRITICAL (YOU MUST):**
- Read and follow `CODING_GUIDELINES.md` for ALL code changes (C++ and Python)
- NVIDIA copyright header on ALL new files (update year on modified files)
- `git commit -s` (DCO sign-off required). Never attribute AI tools in sign-off line
- `pre-commit` hooks run on commit — if files are modified by hooks, re-stage and commit again
- PR title format: `[JIRA/NVBUG/None][type] description` (e.g., `[TRTLLM-5516][perf] optimize cuda graph padding`)
- Python imports: `from package.subpackage import module` (never `from module import Class`)
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

`examples/configs/database/` contains 170+ pareto-optimized serving configurations
across multiple models, GPUs, ISL/OSL combinations, and concurrency levels.
Use these as starting points for deployment and benchmarking rather than hand-tuning parameters.
See [deployment guides](docs/source/deployment-guide/) for model-specific walkthroughs.

## Architecture

See [architecture diagram](.github/tava_architecture_diagram.md) for the full Mermaid diagram.

### Backends

| Backend | Status | Entry Point | Key Path |
|---------|--------|-------------|----------|
| **PyTorch** | Default | `LLM(backend="pytorch")` | `_torch/pyexecutor/` → `PyExecutor` → PyTorch Engine |
| **AutoDeploy** | Beta | `LLM(backend="_autodeploy")` | `_torch/auto_deploy/` → `ADExecutor` → graph transforms + torch.export |
| **TensorRT** | Legacy | `LLM(backend="tensorrt")` | `builder.py` → `trtllm.Executor` → TensorRT Engine |

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

## Design Patterns

| Pattern | Key Points |
|---------|------------|
| **Config hierarchy** | `LlmArgs` → `TrtLlmArgs` / `TorchLlmArgs`, model-specific defaults override generics, Pydantic validation |
| **Model architecture** | Each model: `Config` (inherits `PretrainedConfig`) + `ForCausalLM` (inherits `PretrainedModel`) |
| **Model defaults** | Architecture-specific overrides in `llm_utils.py` (attention kernels, quant, spec decoding, cache) |
| **Distributed execution** | Tensor/pipeline parallelism via `Mapping` class, multiple backends (MPI, Ray, RPC) |
| **Auto-discovery** | Models self-register via `automodel.py`, resolved by HF config `architectures` field |

## Anti-Patterns / Gotchas

- **Pre-commit modifies files in-place** — if hooks fail, files are already modified. Re-stage (`git add`) and commit again.
- **Protected APIs exist** — changes to LLM API signatures will fail `tests/api_stability` tests. Get code owner review.
- **Integration tests need GPUs + models** — always set `LLM_MODELS_ROOT` and ensure GPU access. Unit tests don't.
- **Copyright year** — update to current year when modifying existing files; add full header to new files.
- **Avoid broad exception handling** — catch specific exceptions, not bare `except:` (see `CODING_GUIDELINES.md`).
- **Python import style is enforced** — `from package.subpackage import module`, never `from module import Class`. Pre-commit will not catch this.
- **One concern per PR** — avoid scope creep. If a PR touches unrelated areas, split it.
- **User-facing configuration classes** - when editing or defining any user-facing configuration classes (particularly `LlmArgs` or any class used in its fields), you **MUST** follow the Pydantic guidelines in `CODING_GUIDELINES.md`.

## Development Workflow

1. Set up build environment (see [installation docs](docs/source/installation/))
2. Make changes following `CODING_GUIDELINES.md`
3. Test locally with `pytest`
4. Submit PR:
   - PR title format: `[JIRA/NVBUG/None][type] description` (e.g., `[TRTLLM-5516][perf] optimize cuda graph padding`)
   - Sign commits with DCO (`git commit -s`)
   - Target `main` unless fixing a release branch bug
   - See `CONTRIBUTING.md` for full PR policies

## CI / Testing

See [CI overview](docs/source/developer-guide/ci-overview.md) for full details.

| Layer | Location | Notes |
|-------|----------|-------|
| Unit tests | `tests/unittest/` | Run in pre-merge CI; some tests require GPU |
| API stability | `tests/api_stability/` | Protects committed API signatures |
| Integration tests | `tests/integration/defs/` | Requires GPU + `LLM_MODELS_ROOT` |
| Test lists | `tests/integration/test_lists/test-db/` | Per-GPU YAML files (`l0_a10.yml`, `l0_h100.yml`, etc.) |
| Test waives | `tests/integration/test_lists/waives.txt` | Skip known-failing tests with NVBug links |
| Performance | See [benchmarking guide](docs/source/developer-guide/perf-benchmarking.md) | `trtllm-bench` and `trtllm-serve` benchmarks |

## Key Documentation

| Topic | Path |
|-------|------|
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
