<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# InferenceX: Qwen3.5-397B-A17B NVFP4 on GB300 NVL72, AgentX Disaggregated Serving

This entry reproduces the TensorRT-LLM result for the InferenceX config
`qwen3.5-fp4-gb300-dynamo-trt-agentic-disagg` (AgentX agentic-coding traces,
disaggregated prefill/decode with Dynamo), submitted in
[SemiAnalysisAI/InferenceX#4747](https://github.com/SemiAnalysisAI/InferenceX/pull/4747).

**Code status: on `main`.** The result was produced from a TensorRT-LLM `main`
commit, not from a feature branch: the optimizations of the earlier submission
([SemiAnalysisAI/InferenceX#4356](https://github.com/SemiAnalysisAI/InferenceX/pull/4356))
landed via [#20815](https://github.com/NVIDIA/TensorRT-LLM/pull/20815), and the
runs are configured with regular engine arguments. The pinned NVIDIA container
below already contains that code, so you do not need to build TensorRT-LLM.

## What Was Used

| Component | Version |
|---|---|
| TensorRT-LLM commit | [`0fd58c90ede974e01ecc53627ad5a9bc76846703`](https://github.com/NVIDIA/TensorRT-LLM/commit/0fd58c90ede974e01ecc53627ad5a9bc76846703) on [`main`](https://github.com/NVIDIA/TensorRT-LLM/tree/main) |
| Container | `nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc3.dev202611020007` |
| Container digest | `sha256:f656ce79a88bfdbcd591fe0a5919fcb20c714b59fd73c8b9adb4101659762c7c` (amd64 and arm64 digests in [`source.yaml`](source.yaml)) |
| `ai-dynamo`, `ai-dynamo-runtime` | `1.6.0.dev20261010` |
| Checkpoint | [`nvidia/Qwen3.5-397B-A17B-NVFP4-V2`](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4-V2) |
| InferenceX recipe | `disagg-variants.yaml` at commit `a9fc88e5b0a7ce82990c672d8e3bfd18cbf8e0a6` (see [`source.yaml`](source.yaml)) |

To see the code, check out the commit:

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout 0fd58c90ede974e01ecc53627ad5a9bc76846703
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
| `disagg-gb300-1p1d-tep4-tep16-c8-b1-mnt7-mtp6-kvoffload` | 1× TEP4 | 1× TEP16 (MoE TP2/EP8) | 1 | 8 | 5 |
| `disagg-gb300-1p1d-tep4-tep16-c16-b4-mnt28-mtp6-kvoffload` | 1× TEP4 | 1× TEP16 (MoE TP2/EP8) | 4 | 16 | 5 |
| `disagg-gb300-1p1d-tep4-tep8-c28-b6-mnt42-mtp6-kvoffload` | 1× TEP4 | 1× TEP8 (MoE TP2/EP4) | 6 | 28 | 3 |
| `disagg-gb300-1p2d-tep4-tep8-c128-b16-mnt112-mtp6-kvoffload` | 1× TEP4 | 2× TEP8 (MoE TP2/EP4) | 16 | 128 | 5 |
| `disagg-gb300-2p1d-tep2-tep8-c224-b64-mnt448-mtp6-kvoffload` | 2× TEP2 | 1× TEP8 (MoE TP2/EP4) | 64 | 224 | 3 |
| `disagg-gb300-2p2d-tep2-tep8-c176-b32-mnt224-mtp6-kvoffload` | 2× TEP2 | 2× TEP8 (MoE TP2/EP4) | 32 | 176 | 5 |
| `disagg-gb300-2p2d-tep4-tep8-c264-b32-mnt224-mtp6-kvoffload` | 2× TEP4 | 2× TEP8 (MoE TP2/EP4) | 32 | 264 | 6 |
| `disagg-gb300-3p1d-tep2-tep8-b128-mnt896-mtp6-kvoffload` | 3× TEP2 | 1× TEP8 (MoE TP2/EP4) | 128 | 352, 392 | 4 |
| `disagg-gb300-3p2d-dep4-dep4-c720-b36-mnt252-mtp6-kvoffload` | 3× DEP4 | 2× DEP4 | 36 | 720 | 5 |

The point names encode the topology and the tuned decode budget: `<n>p<m>d` is
the number of prefill and decode workers, `tep4`/`tep16`/`dep4` the prefill and
decode parallel sizes, `c` the benchmark concurrency, `b` the decode
`max_batch_size`, and `mnt` the decode `max_num_tokens` — the draft budget
`max_batch_size` batches × (`max_draft_len` 6 + 1) tokens, e.g. `mnt896` for
`b128`. `mtp6` marks decode MTP with draft length 6, and `kvoffload` the native
KV-cache host offload of `use_kv_cache_manager_v2`: each rank on both roles
pins 128 GiB of host memory as the offload tier (`kv_cache_config.host_cache_size`).

All role tuning is regular engine arguments; no environment switch selects it:
TEP prefill workers run breakable prefill CUDA graphs (`prefill_cuda_graph_backend:
breakable`, captured up to 16384 tokens), decode workers run MTP draft length 6
with the per-point CUDA-graph batch lists, and the largest point runs attention
data parallelism (`enable_attention_dp: true`) on both roles, with the DEP
prefill on piecewise prefill CUDA graphs captured up to 8192 tokens. Settings
the recipe clears from its shared base are simply absent from the exported
files, so every `configs/<point>/<role>.yaml` is a standalone engine argument
file.

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

After the workers start, every role rank logs its native KV-offload host tier;
decode ranks log the CUDA-graph capture of the batch list from the exported
`decode.yaml`, TEP prefill ranks log the breakable prefill-graph capture, and
the DEP4 prefill ranks log the piecewise path (the deprecation warning
confirms `enable_piecewise_cuda_graph` engaged):

```bash
python3 examples/perf_reproductions/_common/check_worker_logs.py \
    examples/perf_reproductions/inferencex/nvidia/Qwen3.5-397B-A17B-NVFP4-V2/GB300/agentx-disagg/source.yaml \
    all_roles='/path/to/logs/*.log' decode='/path/to/logs/decode*.log' prefill_tep='/path/to/logs/prefill*.log'
```

Use `prefill_dep` instead of `prefill_tep` for the DEP4 prefill point. Each
rank logs each line once. Then run the readiness and metrics checks from the
deployment guide.

## Results

This entry does not repeat the published numbers. See them on the InferenceX
website: the [Qwen3.5-397B-A17B model page](https://inferencex.semianalysis.com/model/qwen-3-5)
and the [AgentX results](https://inferencex.semianalysis.com/inference/agentic)
(config `qwen3.5-fp4-gb300-dynamo-trt-agentic-disagg`). Throughput and latency
depend on the network fabric, host memory, and the benchmark client placement
used by the harness.

## Upstream Status

Everything this entry runs is on `main`; the [features](https://github.com/NVIDIA/TensorRT-LLM/pull/20815)
of the earlier feature-branch submission merged in #20815.

| Feature | Status on `main` |
|---|---|
| Fused MoE front, dynamic expert assignment, low-M FP8 GEMMs | On `main` (#20815) |
| Fused FP8 decode attention, context-parallel and segmented decode attention | On `main` (#20815) |
| Fused GEMM + all-reduce + RMSNorm, low-latency all-reduce and exchange kernels | On `main` (#20815) |
| Fused GDN kernels, in-forward and branch-point Mamba snapshots | On `main` (#20815) |
| MTP context draft CUDA graphs, steady-state decode preparation, fused accept, sharded sampling | On `main` (#20815) |
| Breakable prefill CUDA graphs (`prefill_cuda_graph_backend`) | On `main` (#16609, #20815) |
| Sampled iteration perf stats (`iter_perf_stats_interval`) | On `main` (#19717) |
| Shortest-first context scheduling, disaggregated first-step streaming | On `main` (#20815) |

[`source.yaml`](source.yaml) holds the authoritative list.

## Known Differences

- Unlike the earlier submission, no environment variable selects the Qwen3.5
  optimizations: the `.env` files carry only the transport, allocator, garbage
  collection, and logging settings of the recipe, which apply to any model.
- The largest point clears a few settings relative to the recipe's shared
  base — DEP4 prefill drops `allreduce_strategy` and MTP drafting; DEP4 decode
  drops the GEN transfer-window budget (`max_tokens_in_buffer`) and falls back
  to the default MoE-TP. The exported files omit cleared settings, as noted in
  [Points](#points).
