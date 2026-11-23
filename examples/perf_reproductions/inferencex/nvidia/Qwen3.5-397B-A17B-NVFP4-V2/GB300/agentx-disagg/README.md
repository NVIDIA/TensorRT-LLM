<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# InferenceX: Qwen3.5-397B-A17B NVFP4 on GB300 NVL72, AgentX Disaggregated Serving

This entry reproduces the TensorRT-LLM result for the InferenceX config
`qwen3.5-fp4-gb300-dynamo-trt-agentic-disagg` (AgentX agentic-coding traces,
disaggregated prefill/decode with Dynamo), submitted in
[SemiAnalysisAI/InferenceX#5147](https://github.com/SemiAnalysisAI/InferenceX/pull/5147).
That submission replaced the earlier
[SemiAnalysisAI/InferenceX#4747](https://github.com/SemiAnalysisAI/InferenceX/pull/4747)
points; this entry tracks the current one.

**Code status: on `main`.** The result was produced from a TensorRT-LLM `main`
nightly, not from a feature branch: the optimizations of the earlier submissions
(#20815 for round 1, #21997 for round 2, including the remote KV cache tier)
are on `main`, and the runs are configured with regular engine arguments. The pinned
NVIDIA container below already contains that code, so you do not need to build
TensorRT-LLM.

## What Was Used

| Component | Version |
|---|---|
| TensorRT-LLM commit | [`66af44a786db24a4a53704ec12d07f4bb992559a`](https://github.com/NVIDIA/TensorRT-LLM/commit/66af44a786db24a4a53704ec12d07f4bb992559a) on [`main`](https://github.com/NVIDIA/TensorRT-LLM/tree/main) |
| Container | `nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc4.dev202611170009` |
| Container digest | `sha256:80378a7a27ed5eb597ba81e12ec72a6e124909b8970d04d08fcf8a9622094032` (amd64 and arm64 digests in [`source.yaml`](source.yaml)) |
| `ai-dynamo`, `ai-dynamo-runtime` | `1.8.0.dev20261119` |
| Checkpoint | [`nvidia/Qwen3.5-397B-A17B-NVFP4-V2`](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4-V2) |
| InferenceX recipe | `disagg-variants.yaml` at commit `3735ed9e27233597c1bb62508a5e3ceb2717bd99` (see [`source.yaml`](source.yaml)) |

To see the code, check out the commit:

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout 66af44a786db24a4a53704ec12d07f4bb992559a
```

## Points

Each point has a directory under [`configs/`](configs/) with the engine
arguments (`prefill.yaml`, `decode.yaml`) and worker environment
(`prefill.env`, `decode.env`) of each role. They are generated from the
InferenceX recipe by
[`export_srt_recipe.py`](../../../../_tools/export_srt_recipe.py);
[`configs/points.yaml`](configs/points.yaml) summarizes the topology. Nodes are
four-GPU GB300 nodes; the frontend shares the first decode node and the AgentX
client the last, so node counts are prefill plus decode nodes.

All points run TEP2 prefill (MTP draft length 1) and MTP6 decode. Ten points
use the TensorRT-LLM remote KV cache (`kv_cache_config.remote_cache_config`,
[#21997](https://github.com/NVIDIA/TensorRT-LLM/pull/21997)): each decode
rank donates 160 GiB of host memory to the pool and each prefill rank claims up
to 200 GiB on top of its own 168 GiB host cache, while decode keeps no host
cache of its own. The other three keep a plain 128 GiB host cache per rank on
both roles; their names end in `kvoffload`.

| Point (configs/ directory) | Prefill | Decode | Decode batch | Concurrency | KV | Nodes |
|---|---|---|---:|---:|---|---:|
| `disagg-gb300-1p1d-tep2-tep8-c240-b16-mnt896-mtp6` | 1× TEP2 | 1× TEP8 (MoE TP2/EP4) | 16 | 240 | remote | 3 |
| `disagg-gb300-1p2d-tep2-tep8-c48-b4-mnt896-mtp6` | 1× TEP2 | 2× TEP8 (MoE TP2/EP4) | 4 | 48 | remote | 5 |
| `disagg-gb300-1p2d-tep2-tep8-c200-b5-mnt896-mtp6` | 1× TEP2 | 2× TEP8 (MoE TP2/EP4) | 5 | 200 | remote | 5 |
| `disagg-gb300-2p1d-tep2-tep8-c192-b64-mnt896-mtp6` | 2× TEP2 | 1× TEP8 (MoE TP2/EP4) | 64 | 192 | remote | 3 |
| `disagg-gb300-2p1d-tep2-tep8-c352-b64-mnt896-mtp6` | 2× TEP2 | 1× TEP8 (MoE TP2/EP4) | 64 | 352 | remote | 3 |
| `disagg-gb300-2p1d-tep2-tep8-c416-b64-mnt896-mtp6` | 2× TEP2 | 1× TEP8 (MoE TP2/EP4) | 64 | 416 | remote | 3 |
| `disagg-gb300-2p3d-tep2-tep8-c384-b8-mnt896-mtp6` | 2× TEP2 | 3× TEP8 (MoE TP2/EP4) | 8 | 384 | remote | 7 |
| `disagg-gb300-3p2d-tep2-tep8-c456-b64-mnt896-mtp6` | 3× TEP2 | 2× TEP8 (MoE TP2/EP4) | 64 | 456 | remote | 6 |
| `disagg-gb300-3p2d-tep2-tep8-c512-b32-mnt896-mtp6` | 3× TEP2 | 2× TEP8 (MoE TP2/EP4) | 32 | 512 | remote | 6 |
| `disagg-gb300-4p1d-tep2-dep8-c840-b48-mnt336-mtp6` | 4× TEP2 | 1× DEP8 (CUTEDSL) | 48 | 840 | remote (NIXL) | 4 |
| `disagg-gb300-1p1d-tep2-tep16-c32-b1-mnt7-mtp6-kvoffload` | 1× TEP2 | 1× TEP16 (MoE TP2/EP8) | 1 | 32 | host, 128 GiB/rank | 5 |
| `disagg-gb300-1p2d-tep2-tep8-c96-b2-mnt896-mtp6-kvoffload` | 1× TEP2 | 2× TEP8 (MoE TP2/EP4) | 2 | 96 | host, 128 GiB/rank | 5 |
| `disagg-gb300-1p4d-tep2-tep8-c96-b1-mnt896-mtp6-kvoffload` | 1× TEP2 | 4× TEP8 (MoE TP2/EP4) | 1 | 96 | host, 128 GiB/rank | 9 |

On the remote points, the point names drop the `kvoffload` suffix: `mnt896` is the
decode `max_num_tokens` — `max_batch_size` batches × (`max_draft_len` 6 + 1)
tokens — and `mtp6` marks decode MTP with draft length 6.

Remote-cache rendezvous happens under `/logs/rt_rdv` (a directory shared by every
rank of the point). Nine remote points use the fabric transport (NVLink,
compact host memory); the 4P1D DEP8 point uses the NIXL transport
(host-to-host RDMA over InfiniBand) with 176 GiB donated per decode rank and
agent bounce buffers on both roles, because on that cluster host-memory fabric
handles exported from some nodes cannot be imported, and 4P1D's donations leave
too little spare capacity to lose one decode node's donors.

All role tuning is regular engine arguments; no environment switch selects it:
TEP prefill workers run breakable prefill CUDA graphs (`prefill_cuda_graph_backend:
breakable`) with shortest-uncached-first context scheduling, and decode workers
run MTP draft length 6 with per-point CUDA-graph batch lists. Request admission
and routing sit with the frontend and Dynamo (`DYN_ROUTER_*` variables in the
exported `.env` files): on the nine points that admit decode requests by
priority (`waiting_queue_policy: priority`), decode workers advertise the
Dynamo KV router with the `output_length_prediction` request classifier, which
forwards each request's predicted reply length as its TensorRT-LLM priority —
replacing `trtllm-serve`'s `output_length_prediction`; 2P1D conc 192, 3P2D
conc 456/512, and 4P1D DEP8 conc 840 keep FCFS admission and route decode
least-loaded. Settings the recipe clears from its shared base are simply absent
from the exported files, so every `configs/<point>/<role>.yaml` is a
standalone engine argument file.

## Run It

### Option A: the InferenceX harness (exact reproduction)

Check out InferenceX at the recipe commit in [`source.yaml`](source.yaml) and
run the config key `qwen3.5-fp4-gb300-dynamo-trt-agentic-disagg` with its
multi-node srt-slurm recipes, following InferenceX's
`inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes/RECIPES.md`. This uses
the same launcher, benchmark client, trace dataset, and node placement as the
submission. The launcher stages the recipe directory's `configs/`
(including `dynamo-router-policy-output-length.yaml` that defines the
`output_length_prediction` classifier) into the job, mounted at `/configs`. The
Dynamo gateway is installed from the `1.8.0.dev20261119` wheels via the
recipe's `dynamo.source`.

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
- On the nine priority-admission points, give the frontend and the decode
  workers the `output_length_prediction` router policy (a two-line yaml; see
  Option A): pass it as the frontend's `--router-policy-config` before the
  decode workers advertise the classifier.

## Check Your Run

Inside the container, after installing the Dynamo wheels, confirm the TensorRT-LLM
and Dynamo versions:

```bash
python3 examples/perf_reproductions/_common/check_image.py \
    examples/perf_reproductions/inferencex/nvidia/Qwen3.5-397B-A17B-NVFP4-V2/GB300/agentx-disagg/source.yaml
```

After the workers start, every rank logs the KV-cache tier layout it built:
remote points log decode donations and prefill claims of the remote tier, host
points log the 128 GiB host tier, decode ranks log the CUDA-graph capture of
the batch list from the exported `decode.yaml`, and TEP prefill ranks log the
breakable prefill-graph capture:

```bash
python3 examples/perf_reproductions/_common/check_worker_logs.py \
    examples/perf_reproductions/inferencex/nvidia/Qwen3.5-397B-A17B-NVFP4-V2/GB300/agentx-disagg/source.yaml \
    prefill_tep='/path/to/logs/prefill*.log' decode='/path/to/logs/decode*.log' \
    prefill_remote='/path/to/logs/prefill*.log' decode_remote='/path/to/logs/decode*.log' \
    all_roles_host='/path/to/logs/*.log'
```

Use the remote patterns on the ten remote-cache points (the NIXL point logs
`nixl` instead of `fabric` plus a 176 GiB donation) and the
`all_roles_host` pattern on the three `kvoffload` points, where it confirms
the 128 GiB tier. Each rank logs each line once. Then run the readiness and
metrics checks from the deployment guide.

## Results

This entry does not repeat the published numbers. See them on the InferenceX
website: the [Qwen3.5-397B-A17B model page](https://inferencex.semianalysis.com/model/qwen-3-5)
and the [AgentX results](https://inferencex.semianalysis.com/inference/agentic)
(config `qwen3.5-fp4-gb300-dynamo-trt-agentic-disagg`). Throughput and latency
depend on the network fabric, host memory, and the benchmark client placement
used by the harness.

## Upstream Status

Everything this entry runs is on `main`; the round 1 optimizations of the
first feature-branch submission merged in
[#20815](https://github.com/NVIDIA/TensorRT-LLM/pull/20815) and the round 2
optimizations — including the KV cache manager v2 remote tier and GEN
admission by output-length prediction — merged in
[#21997](https://github.com/NVIDIA/TensorRT-LLM/pull/21997).

| Feature | Status on `main` |
|---|---|
| KV cache manager v2 remote tier (`kv_cache_config.remote_cache_config`, fabric and NIXL transports) | On `main` (#21997) |
| GEN admission by online output-length prediction; pooled GEN dispatch; ADP home-wait routing | On `main` (#21997) |
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

- No environment variable selects the Qwen3.5 optimizations; the `.env` files
  carry transport, allocator, garbage-collection, UCX, and Dynamo router
  settings of the recipe, which apply to any model. The UCX variables
  (`UCX_IB_DIRECT_NIC=n`, `UCX_CUDA_COPY_DMABUF=no`) work around a driver
  limitation on these nodes: PCIe-mapped GPU dma-bufs are rejected because the
  GPU and NIC sit in different IOMMU groups, so NIXL registers GPU memory
  through `nvidia_peermem` instead.
- The FCFS points (2P1D conc 192/352/416, 3P2D conc 456/512, 4P1D DEP8) clear
  the recipe's priority-admission settings — `waiting_queue_policy`,
  `waiting_queue_max_wait_s`, and `defer_generation_only_release` on decode
  (352/416 keep the last one), plus `DYN_ROUTER_POLICY_CONFIG` and the
  frontend's `router-policy-config`. The exported files omit cleared settings,
  as noted in [Points](#points).
- The 4P1D DEP8 point clears the GEN transfer-window budget
  (`max_tokens_in_buffer`) and falls back to default MoE-TP; its decode keeps
  no host cache and donates 176 GiB per rank to the NIXL remote tier.
