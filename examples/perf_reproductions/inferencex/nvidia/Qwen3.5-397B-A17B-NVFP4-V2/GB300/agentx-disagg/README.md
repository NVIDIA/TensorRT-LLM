<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# InferenceX: Qwen3.5-397B-A17B NVFP4 on GB300 NVL72, AgentX Disaggregated Serving

This entry reproduces the TensorRT-LLM result for the InferenceX config
`qwen3.5-fp4-gb300-dynamo-trt-agentic-disagg` (AgentX agentic-coding traces,
disaggregated prefill/decode with Dynamo), submitted in
[SemiAnalysisAI/InferenceX#4356](https://github.com/SemiAnalysisAI/InferenceX/pull/4356).

**Code status: feature branch.** The result was produced from a TensorRT-LLM
feature branch, not from `main`. The pinned NVIDIA container below already
contains that code, so you do not need to build TensorRT-LLM. The
[upstream status](#upstream-status) section lists which features have reached
`main`.

## What Was Used

| Component | Version |
|---|---|
| TensorRT-LLM commit | [`4d09932ca7240f6fbb4d0dea811691446cc8cf92`](https://github.com/NVIDIA/TensorRT-LLM/commit/4d09932ca7240f6fbb4d0dea811691446cc8cf92) on [`feat/1.3.0rc24-iter-perf-stats-interval`](https://github.com/NVIDIA/TensorRT-LLM/tree/feat/1.3.0rc24-iter-perf-stats-interval) |
| Container | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0a0.dev202610210406` |
| Container digest | `sha256:16b3237826b7969a3a736ceae930f7d63fe44f3c44c8e6c59a9d4b01dd66c199` (amd64 and arm64 digests in [`source.yaml`](source.yaml)) |
| `ai-dynamo`, `ai-dynamo-runtime` | `1.6.0.dev20261010` |
| Checkpoint | [`nvidia/Qwen3.5-397B-A17B-NVFP4-V2`](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4-V2) |
| InferenceX recipe | `disagg-variants.yaml` at commit `bb54127914f6c7c039a5a89c5fd8590441e4fc90` (see [`source.yaml`](source.yaml)) |

To see the code, check out the commit:

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout 4d09932ca7240f6fbb4d0dea811691446cc8cf92
```

## Points

Each point has a directory under [`configs/`](configs/) with the engine
arguments (`prefill.yaml`, `decode.yaml`) and worker environment
(`prefill.env`, `decode.env`) of each role. They are generated from the
InferenceX recipe by
[`export_srt_recipe.py`](../../../../_tools/export_srt_recipe.py);
[`configs/points.yaml`](configs/points.yaml) summarizes the topology. Nodes are
four-GPU GB300 nodes.

| Point (configs/ directory) | Prefill | Decode | Decode batch | Concurrency | Nodes |
|---|---|---|---:|---|---:|
| `disagg-gb300-1p2d-tep4-tep8-c128-b16-mtp-kvoffload` | 1× TEP4 | 2× TEP8 (MoE TP2/EP4) | 16 | 128 | 5 |
| `disagg-gb300-2p1d-tep2-tep8-c192-b64-mtp-kvoffload` | 2× TEP2 | 1× TEP8 (MoE TP2/EP4) | 64 | 192 | 3 |
| `disagg-gb300-2p2d-tep2-tep8-c176-b32-mtp-kvoffload` | 2× TEP2 | 2× TEP8 (MoE TP2/EP4) | 32 | 176 | 5 |
| `disagg-gb300-2p2d-tep4-tep8-c264-b32-mtp-kvoffload` | 2× TEP4 | 2× TEP8 (MoE TP2/EP4) | 32 | 264 | 6 |
| `disagg-gb300-3p1d-dep4-dep16-b12-mtp-kvoffload` | 3× DEP4 | 1× DEP16 | 12 | 600, 700 | 7 |
| `disagg-gb300-3p2d-dep4-dep4-c704-b32-mtp-kvoffload` | 3× DEP4 | 2× DEP4 | 32 | 704 | 5 |

The Qwen3.5 optimizations of this branch are selected with one variable per
role: `TRTLLM_Q35_PROFILE=gen` on decode workers and `TRTLLM_Q35_PROFILE=ctx` on
prefill workers (already in the `.env` files). The profile applies only to
Qwen3.5 on SM103 (GB300); a `TRTLLM_Q35_*` variable set explicitly in the
environment keeps its value.

## Run It

### Option A: the InferenceX harness (exact reproduction)

Check out InferenceX at the recipe commit in [`source.yaml`](source.yaml) and
run the config key `qwen3.5-fp4-gb300-dynamo-trt-agentic-disagg` with its
multi-node srt-slurm recipes, following InferenceX's
`inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes/RECIPES.md`. This uses
the same launcher, benchmark client, trace dataset, and node placement as the
submission.

The InferenceX performance jobs fix MTP acceptance: the launcher sets
`TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS` from the benchmark's golden
acceptance length. Use the harness to reproduce the published numbers; do not
set that variable in production or accuracy runs.

### Option B: launch the workers yourself

Follow the container, Dynamo, frontend, and multi-node instructions of the
[Qwen3.5 GB300 NVL72 agentic serving guide](../../../../../../../docs/source/deployment-guide/deployment-guide-for-qwen3.8-qwen3.5-on-trtllm.md),
with these differences:

- Use the container and Dynamo versions from [What Was Used](#what-was-used).
- For each role, `source configs/<point>/<role>.env` instead of the guide's
  environment file, and pass `configs/<point>/<role>.yaml` as
  `--extra-engine-args`.
- Start the number of workers per role, ranks per worker, and nodes shown in
  [`configs/points.yaml`](configs/points.yaml).

## Check Your Run

Inside the container, after installing the Dynamo wheels, confirm the TensorRT-LLM and Dynamo versions:

```bash
python3 examples/perf_reproductions/_common/check_image.py \
    examples/perf_reproductions/inferencex/nvidia/Qwen3.5-397B-A17B-NVFP4-V2/GB300/agentx-disagg/source.yaml
```

After the workers start, confirm that every worker applied its profile. Each
rank logs one line; decode applies 89 switches, TEP prefill 10, and DEP prefill
(attention DP) none:

```bash
python3 examples/perf_reproductions/_common/check_worker_logs.py \
    examples/perf_reproductions/inferencex/nvidia/Qwen3.5-397B-A17B-NVFP4-V2/GB300/agentx-disagg/source.yaml \
    decode='/path/to/logs/decode*.log' prefill_tep='/path/to/logs/prefill*.log'
```

Use `prefill_dep` instead of `prefill_tep` for the DEP4 prefill points. Then
run the readiness and metrics checks from the deployment guide.

## Results

This entry does not repeat the published numbers. See them on the InferenceX
website: the [Qwen3.5-397B-A17B model page](https://inferencex.semianalysis.com/model/qwen-3-5)
and the [AgentX results](https://inferencex.semianalysis.com/inference/agentic)
(config `qwen3.5-fp4-gb300-dynamo-trt-agentic-disagg`). Throughput and latency
depend on the network fabric, host memory, and the benchmark client placement
used by the harness.

## Upstream Status

| Feature | Status on `main` |
|---|---|
| `TRTLLM_Q35_PROFILE` role profiles | Branch only |
| Fused MoE front, dynamic expert assignment, low-M FP8 GEMMs | Branch only |
| Fused FP8 decode attention, context-parallel and segmented decode attention | Branch only |
| Fused GEMM + all-reduce + RMSNorm, low-latency all-reduce and exchange kernels | Branch only |
| Fused GDN kernels, in-forward and branch-point Mamba snapshots | Branch only |
| MTP context draft CUDA graphs, steady-state decode preparation, fused accept, sharded sampling | Branch only |
| Breakable prefill CUDA graphs | Equivalent on `main` (`prefill_cuda_graph_backend`, #16609) |
| Sampled iteration perf stats (`iter_perf_stats_interval`) | On `main` (#19717) |
| Shortest-first context scheduling, disaggregated first-step streaming | Branch only |

[`source.yaml`](source.yaml) holds the authoritative list.

## Known Differences

- The branch selects breakable prefill CUDA graphs with `TRTLLM_PREFILL_BCG=1`
  (in the TEP prefill `.env` files); `main` uses the
  `prefill_cuda_graph_backend` configuration field instead.
- The worker environments also set branch-specific variables outside the
  Qwen3.5 profile, such as `TRTLLM_CTX_SCHED_SRPT` and
  `TRTLLM_DISAGG_GEN_EMIT_FIRST_STEP`; they have no effect on `main`.
