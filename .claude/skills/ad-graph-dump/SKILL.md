---
name: ad-graph-dump
description: >
  Enable and interpret TensorRT-LLM AutoDeploy FX graph text dumps via AD_DUMP_GRAPHS_DIR.
  Use when you need before/after graphs per transform, to locate subgraphs, or to confirm a
  rewrite ran. Paths and behavior are grounded in tensorrt_llm/_torch/auto_deploy (GraphWriter,
  BaseTransform). Complements ad-add-fusion-transformation.
license: Apache-2.0
tags:
  - tensorrt-llm
  - autodeploy
  - graph-dump
  - debugging
metadata:
  author: NVIDIA Corporation
---

# AutoDeploy: Graph dumps (`AD_DUMP_GRAPHS_DIR`)

## Where this skill applies

This file is part of **trtllm-agent-toolkit**. Commands and paths such as `examples/auto_deploy/` and `tensorrt_llm/` are relative to a **TensorRT-LLM source checkout**, not the plugin repository.

## When to use this skill

- You need to see how the FX graph changes **after each registered transform** runs.
- You are verifying that a subgraph exists, that a fusion matched, or that metadata / wrappers (`getitem`, `view`, `reshape`) appeared or disappeared between dumps.
- You are pairing log output with on-disk graph files while debugging AutoDeploy.

## Related skills in this plugin

| Skill | Use it for |
|-------|------------|
| **ad-add-fusion-transformation** | Implementing or reviewing fusion passes once you know what the graphs show. |
| **trtllm-codebase-exploration** | Searching the TRT-LLM tree for transforms, custom ops, and patterns. |
| **trtllm-code-contribution** | Tests and contribution hygiene after you change TRT-LLM. |

## Environment variable

Set:

```bash
export AD_DUMP_GRAPHS_DIR=/path/to/output/dir
```

Implementation: `GraphWriter.DUMP_GRAPHS_ENV == "AD_DUMP_GRAPHS_DIR"` in `tensorrt_llm/_torch/auto_deploy/utils/graph_writer.py`.

If unset, no graph files are written.

## When dumps are produced

After **each** transform application, `BaseTransform` calls `graph_writer.dump_graph(mod, t_name, self.config.stage.value)` from `tensorrt_llm/_torch/auto_deploy/transform/interface.py` (immediately after `_visualize_graph`). So the dump reflects the module **after** that transform has run.

## Rank / process behavior

From `GraphWriter.dump_graph`:

- Dumps run only when `AD_DUMP_GRAPHS_DIR` is set.
- If `ADLogger.rank` is set and is not `0`, dumping is skipped (non–rank-0 processes do not write files).

## Directory lifecycle

On the **first** dump on rank 0, `GraphWriter` **removes** the target directory if it already exists, then recreates it. Do not point `AD_DUMP_GRAPHS_DIR` at a directory that must be preserved without copying it first.

## File naming and ordering

Files are named:

```text
{NNN}_{<stage.value>}_{<transform_key>}.txt
```

- `NNN` is a monotonically increasing three-digit counter (001, 002, …) in run order across all dumps in that process.
- The middle segment is each transform’s `config.stage` value (same enum/string used in `default.yaml` under each transform’s `stage:` field).
- The last segment is the transform’s registry key (`transform_name` passed into `dump_graph`).

So lexicographic sort by filename matches **pipeline order** for that run.

## File contents

Each file is text and starts with headers similar to:

```text
# Transform: <transform_key>
# Stage: <stage.value>
# GraphModules found: <count>
```

Then, for every `torch.fx.GraphModule` found under `mod.named_modules()` (including the root), the writer emits a section title and an SSA-style listing with shape/dtype metadata via `dump_ssa_with_meta()` in the same module.

Use this to compare operator chains, consumers, and `node.meta` shape/dtype hints across consecutive files.

## Example: capture dumps from a registry build

From the **root of the TensorRT-LLM clone** (adjust the script and flags to your workflow):

```bash
AD_DUMP_GRAPHS_DIR=/tmp/ad-graphs \
  python examples/auto_deploy/build_and_run_ad.py --model <hf-model-id> --use-registry
```

Pick any AutoDeploy entrypoint you already use; the requirement is only that the code path runs the transform pipeline with `AD_DUMP_GRAPHS_DIR` set in the environment.

## Logs vs dump files

While a transform runs, logging is patched so messages can be prefixed with `[stage=<stage.value>, transform=<transform_key>]` (see `with_transform_logging` in `transform/interface.py`). Transform summaries log `[SUMMARY]` with `matches=<n>` or `skipped` / `disabled` (`_log_transform_summary`). Use those lines together with the numbered dump files to tie **match counts** to **graph shape** before and after a specific transform.

## Pitfalls

- **Stale directory**: Because the dump dir is deleted on first use, a second run in the same shell without changing `AD_DUMP_GRAPHS_DIR` overwrites prior output.
- **No GraphModules**: If the module has no `GraphModule` children, `dump_graph` returns without creating a new file for that step (see early return in `graph_writer.py`).
- **Distributed**: Only rank 0 writes; other ranks skip silently.

## Source references

- `tensorrt_llm/_torch/auto_deploy/utils/graph_writer.py` — env var, filenames, SSA dump.
- `tensorrt_llm/_torch/auto_deploy/transform/interface.py` — call site after each transform; log prefix decorator.
