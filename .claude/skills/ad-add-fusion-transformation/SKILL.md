---
name: ad-add-fusion-transformation
description: >
  Claude Code skill (trtllm-agent-toolkit): implement or extend TensorRT-LLM AutoDeploy fusion
  transforms under transform/library/ in a TensorRT-LLM checkout. Prefer existing kernels and custom
  ops; use Triton only when no viable existing-kernel path exists. Use ad-graph-dump for
  AD_DUMP_GRAPHS_DIR workflows. Covers TRT-LLM paths, registry, default.yaml registration, graph
  validation, tests, and a review checklist — without prescribing profiling tools or throughput
  targets.
license: Apache-2.0
tags:
  - tensorrt-llm
  - autodeploy
  - fusion
  - graph-transform
  - optimization
metadata:
  author: NVIDIA Corporation
---

# Autodeploy: Add Fusion Transformation Pass

## Where this skill applies

This file lives in the **trtllm-agent-toolkit** plugin. Paths such as `tensorrt_llm/...`, `examples/auto_deploy/...`, and `tests/...` are relative to a **TensorRT-LLM source checkout** on the user’s machine — not the plugin tree.

After installing the plugin (see the toolkit `README.md`), skills use the `trtllm-agent-toolkit:` prefix (for example `trtllm-agent-toolkit:ad-add-fusion-transformation`).

## Related skills in this plugin

| Skill | Use it for |
|-------|------------|
| **ad-graph-dump** | Enabling `AD_DUMP_GRAPHS_DIR`, dump file layout, and how to read SSA graph output. |
| **trtllm-codebase-exploration** | Mapping existing transforms, custom ops, and search patterns before writing a pass. |
| **trtllm-code-contribution** | TensorRT-LLM pre-commit, tests, DCO sign-off, and PR expectations. |
| **triton-kernel-writing** | Implementing a **Triton** op only after existing-kernel lookup fails. |
| **triton-tileir-optimization** | Tuning **existing** Triton kernels for the TileIR backend when that path applies. |
| **cuda-kernel-writing** | Raw CUDA extension work if the viable path is a PyTorch C++ extension (not Triton). |
| **cute-kernel-writing** / **cudeepy-kernel-writing** | CuTe DSL/LIR or CuDeepy-generated kernels when that is the chosen integration path. |

Use this skill when you already know **which subgraph or pattern** you are targeting (from graph dumps, logs, or code reading). For dump capture and file semantics, follow **ad-graph-dump** first.

## When to use this skill

- Adding, extending, or reviewing a fusion under AutoDeploy transforms in a TensorRT-LLM tree.

### Workflow (concise)

1. Confirm the pattern in **current** graph dumps (see **ad-graph-dump**).
2. Search for an existing kernel or custom-op path before new Triton or CUDA.
3. Implement the smallest change that proves correctness and matching; add tests.
4. Re-run dumps and tests; if outputs drift, separate matching issues from metadata loss from numeric differences.

## Finding fusion candidates (lightweight)

Do this before writing a new pass so you work on real graph structure.

### Inputs

- Graph dump directory from a run with `AD_DUMP_GRAPHS_DIR` set (see **ad-graph-dump**).
- Model id and active AutoDeploy config (registry YAML, `default.yaml` overlays).
- TensorRT-LLM source tree for kernel and transform lookup.

### Outputs

- Ordered list of candidates with: graph evidence, existing-kernel lookup (`found` / `not_found`), recommendation (`use_existing_kernel`, `needs_triton_fallback`, `defer`), and trade-offs (complexity, correctness risk).

### Discovery workflow

1. Parse dumps for repeated unfused patterns (element-wise chains, norm chains, epilogues, attention-adjacent ops).
2. Search the tree for equivalent transforms or custom ops; record file/symbol evidence.
3. If nothing fits, mark Triton or other kernel work as a deliberate fallback.
4. Prefer candidates with clear recurrence, existing support, and lower numerical risk.

### Per-candidate template

```text
Candidate: <short-name>
Affected graph pattern: <pattern>
Existing kernel lookup: <found|not_found>
Evidence: <path/symbol>
Recommendation: <use_existing_kernel|needs_triton_fallback|defer>
Strengths / weaknesses / risks:
- ...
```

### Guardrails

- Do not skip existing-kernel lookup.
- Do not default to Triton when a viable existing op already exists.
- If uncertain, `defer` and narrow the question with one more dump or test.

---

## Inputs (implementation)

- Chosen candidate or concrete subgraph.
- Active model and config files.
- Fresh graph dumps when available.
- Current baseline: match counts from logs, unit test status, any accuracy notes you already maintain.

## Outputs (implementation)

- Pass design or patch: registered transform, `default.yaml` entry, optional model-registry YAML.
- Path decision: `existing_kernel_path` vs `triton_fallback_path` (or other kernel stack).
- Validation notes: graph evidence, `[SUMMARY] matches=...` before/after from AutoDeploy logs, test results.

## Implementation workflow

1. Align the pass with **observed** graph structure from dumps — not assumed op names from docs alone.
2. Search `transform/library/`, `custom_ops/`, `torch.ops.auto_deploy.*`, and related tests for reuse.
3. Integrate an existing op when possible; otherwise delegate kernel work to the appropriate skill (**triton-kernel-writing**, **cuda-kernel-writing**, etc.).
4. Keep one logical change per patch; extend tests in the same change.
5. Re-read dumps after the change; if match counts collapse, suspect pattern availability or metadata propagation.

## Where fusion passes live

- Transforms: `tensorrt_llm/_torch/auto_deploy/transform/library/`
- Registry / base behavior: `tensorrt_llm/_torch/auto_deploy/transform/interface.py`
- Default transform list: `tensorrt_llm/_torch/auto_deploy/config/default.yaml`
- Dump helper: `tensorrt_llm/_torch/auto_deploy/utils/graph_writer.py`
- Graph utilities: `tensorrt_llm/_torch/auto_deploy/utils/node_utils.py`, `tensorrt_llm/_torch/auto_deploy/utils/_graph.py`
- Custom ops: `tensorrt_llm/_torch/auto_deploy/custom_ops/`

Tests (typical):

- `tests/unittest/auto_deploy/singlegpu/transformations/library/`
- `tests/integration/defs/accuracy/test_llm_api_autodeploy.py` (when behavior or numerics may change)

## How to add a transform

### Implement the pass

Create or update a module under `transform/library/` and register the class:

```python
@TransformRegistry.register("my_transform_key")
class MyTransform(BaseTransform):
    @classmethod
    def get_config_class(cls):
        return MyTransformConfig
```

Use a dedicated config class only when the pass needs parameters beyond the base transform config.

### Register in `default.yaml`

Add a key under `transforms:` in `tensorrt_llm/_torch/auto_deploy/config/default.yaml`. **Copy the field set from the closest existing transform** in the same section of the file (required keys depend on the transform config class and on how peers are declared). New experimental passes should stay **`enabled: false`** until covered by tests and dumps.

### Enable for a specific model

For targeted rollout, adjust registry YAMLs under `examples/auto_deploy/model_registry/configs/` rather than turning on unproven passes globally.

## Implementation rules

- Prefer existing AutoDeploy / TRT-LLM ops and `torch.ops.auto_deploy` entries.
- Prefer stable, backend-neutral graph contracts; avoid hiding real dataflow in `node.meta` when an edge should carry it.
- Use metadata for observable tensor facts (shape, dtype) and preserve it across rewrites when replacements should remain traceable.
- **One hypothesis per patch** — do not mix unrelated fusions.

## Existing kernel first, Triton second

Before Triton:

1. Search `transform/library/` and `custom_ops/`.
2. Search `torch.ops.auto_deploy.*` and TRT-LLM custom op definitions.
3. Read tests for similar integrations.

Use **triton-kernel-writing** only when no suitable op exists and you accept owning kernel + integration work.

## Validation order

1. Graph dumps — pattern present, rewrite visible (see **ad-graph-dump**).
2. Unit tests for the transform.
3. Integration or accuracy checks when numerics or end-to-end behavior may change.

## Match counts

AutoDeploy logs `[SUMMARY] matches=<n>` (or `skipped` / `disabled`) per transform. Compare before and after your change; a large drop usually indicates pattern or metadata issues, not “slow runs.”

## Testing expectations

Follow **trtllm-code-contribution** for repo conventions. Cover:

- Happy-path micrograph or exported-graph rewrites.
- Failure modes that must **not** fuse (multiple consumers, mixed consumers).
- Metadata preservation when an upstream pass feeds your pattern.

Primary unittest location for library transforms:

- `tests/unittest/auto_deploy/singlegpu/transformations/library/`

## Review checklist

- Target structure appears in current dumps.
- Transform registered and listed in `default.yaml` consistently with peer entries.
- Model-registry toggles are intentional.
- Non-zero `matches` where expected, or `skipped` is explained.
- Before/after dump snippets or diffs saved for the review thread.
- Tests cover both success and intentional non-match cases.
- If outputs change, classify match loss vs metadata loss vs acceptable numeric drift.

## Guardrails

- Do not bundle unrelated passes in one change.
- If dumps contradict expectations, document what you observed before chasing unrelated hypotheses.

## Iteration note (template)

```text
Candidate: <name>
Path: <existing_kernel_path|triton_fallback_path|other>
Rationale:
- ...
Graph validation: <pass|fail — what files / ops>
Summary logs: <matches before / after>
Tests: <what ran>
Open risks:
- ...
```
