<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4.1-Flash AgentX configuration data points

This preview provides configurations for three selected AgentX data points for DeepSeek-V4.1-Flash
on GB300 NVL72, with a Slurm launcher and the public AgentX client. The branch contains the
performance implementation from
[TensorRT-LLM #21438](https://github.com/NVIDIA/TensorRT-LLM/pull/21438) at
`e6930b1c65f2c756656ac5a525e28b0d3261a4b9`, together with these configurations.

## Package contents

- `points/`: three operating points, each with context, generation and orchestrator YAMLs plus
  `point.env` for topology, client sessions and worker environment settings.
- `run.sh` and `slurm/`: job submission, optional node-local checkpoint staging, worker placement,
  runtime checks, per-run address/auth rendering, health checks and result collection.
- `client/`: installation of the pinned public AgentX client, dataset-cache preparation and
  smoke/full benchmark execution.
- `cluster.env.example`: local partition/account, image, model/storage paths and fabric settings.
  `.gitignore` excludes the local site file and run artifacts.

## Build the runtime

On an **aarch64 build host with Docker/BuildKit**, build the release image from this checkout.
The repository's Dockerfile supplies the toolchain and builds both Python and native code for
GB300 (SM103). Run from the repository root:

```bash
git lfs pull
git submodule update --init --recursive
make -C docker release_build CUDA_ARCHS=103 \
    IMAGE_WITH_TAG=tensorrt-llm-dsv41:local \
    BUILD_WHEEL_OPTS='--no-venv --use_ccache -j 32'

# Export the local Docker image to shared storage for Slurm/pyxis.
# Set SHARED to the same path as in cluster.env; enroot must be on this build host.
SHARED=/shared/dsv41-agentx
mkdir -p "$SHARED"
enroot import -o "$SHARED/tensorrt-llm-dsv41.sqsh" dockerd://tensorrt-llm-dsv41:local
```

This builds a local image; it does not push to a registry. As an alternative to the squashfs
export, distribute the built image through your own registry and set `IMAGE` to that tag.
See the [source-build guide](../../../../docs/source/installation/build-from-source.md) for
build prerequisites and options. BuildKit storage must have enough room for the development
image, CUDA build and wheel. Each job checks the installed wheel's source-commit metadata and
imports the native package on every node before loading the model. The wheel may come from this checkout's HEAD, or its
direct parent when the only changes since that parent are inside this config directory.
An image built from the parent implementation can therefore serve the same code with these configs.
The image must contain this branch's compiled native code as well as its Python package.

## Configure and run

Requirements:

- GB300 NVL72 nodes with 4 GPUs each, Slurm and pyxis/enroot. Keep all job nodes in one NVL72 domain.
- Shared storage mounted at the same path on every node. Put the model, run output and caches
  under paths covered by `CONTAINER_MOUNTS`. The launcher copies its runtime scripts into each run.
- Hugging Face checkpoint `nvidia/DeepSeek-V4.1-Flash-NVFP4` at revision
  `3431dde3247c13b5957f682b1e3c6fcae2566079`, including its DSpark draft model (about 530 GB).
- Access to GitHub, PyPI and Hugging Face on the client node, or a prepared client venv and dataset
  cache. The client is pinned to `SemiAnalysisAI/agentx-harness@754356e9a39acc6cc6afb242d123bb57c3fb6f75`.

From this directory:

```bash
cp cluster.env.example cluster.env
# Edit partition, account, shared/model paths and container mounts for your system.
# IMAGE defaults to the squashfs exported above.
source cluster.env
hf download nvidia/DeepSeek-V4.1-Flash-NVFP4 \
    --revision 3431dde3247c13b5957f682b1e3c6fcae2566079 --local-dir "$MODEL_PATH"

./run.sh points/max-throughput_5xDEP2+1xDEP8_c7000 --dry-run
./run.sh points/max-throughput_5xDEP2+1xDEP8_c7000 --smoke
./run.sh points/mid_5xDEP2+1xDEP16_c5100 --smoke
./run.sh points/low-latency_1xDEP2+4xTEP8_c84 --smoke
```

`--dry-run` prints the launch commands and renders configs without submitting a job. `--smoke`
runs a 300-second functional check with 512 sessions on the DEP points and 84 on the low-latency
point. Set `CONCURRENCY` to override that smoke workload. Omit `--smoke` to use the point's full
concurrency and the 3,600-second measurement window.
The shortened smoke uses `--unsafe-override` because the AgentX scenario normally requires at
least 900 seconds. It is not a performance measurement.

| Point | Context servers | Generation servers | Worker GPUs | Nodes including client | Sessions |
| --- | --- | --- | ---: | ---: | ---: |
| Maximum throughput | 5 × DEP2 | 1 × DEP8 | 18 | 6 | 7,000 |
| Intermediate | 5 × DEP2 | 1 × DEP16 | 26 | 8 | 5,100 |
| Low latency | 1 × DEP2 | 4 × TEP8 | 34 | 9 | 84 |

DEP uses data-parallel attention and expert-parallel MoE. TEP8 uses tensor-parallel attention
across 8 GPUs and MoE split 4-way by tensor and 2-way by expert. Node counts assume 4 GPUs per node.
The intermediate point targets P90 decode interactivity of at least 150 tokens/s/user and P90
TTFT of at most 10 seconds; this checkout's performance against that target is unmeasured.

## Reference performance

The [public performance chart in #21438](https://github.com/NVIDIA/TensorRT-LLM/pull/21438)
shows these approximate GB300 NVL72 AgentX results for the original bring-up implementation:

| Point | P90 decode interactivity, tokens/s/user | Total token throughput, tokens/s/worker GPU |
| --- | ---: | ---: |
| Maximum throughput, c7000 | ~85 | ~1.54M |
| Intermediate, c5100 | ~225 | ~1.17M |
| High decode interactivity, c84 | ~1,100 | ~15K |

Interactivity is `1000 / P90 inter-token latency (ms)`; evaluate TTFT separately. Total token
throughput counts input tokens, including cached prefixes, plus output tokens, divided by worker
GPUs. The c84 point targets decode speed; the intermediate point targets both decode interactivity
and TTFT.

These historical values are planning references for the full point settings, not measurements of
this checkout. The PR also reports 30-minute port-versus-bring-up comparisons with small changes
for c7000 and c84; its reported intermediate case is c7600, a different configuration from c5100.
That comparison does not establish a performance tolerance for these configs on another system.
Match the checkpoint, client/scenario, full session count and 3,600-second window, forced DSpark
acceptance, placement and fabric settings when comparing. A `--smoke` run checks function only.

## Placement and configuration

- Context servers use GPU order `0,2` and `3,1`, with each rank's CPUs bound to its GPU's NUMA node.
  Host allocations use NUMA nodes 0 and 1. The DEP points have a dedicated client node and place
  the orchestrator on the last generation node, bound to NUMA node 1. The low-latency point places
  the client on the context node's NUMA node 1 and the orchestrator on the first generation node.
- `point.env` carries topology, client concurrency and worker environment settings. The role YAMLs
  carry serving parameters; `slurm/launch.sbatch` fills model paths, server addresses and cache paths.
- DEP subagent routing uses `X-Dynamo-Parent-Session-ID`. Each run creates a fresh random
  `internal_request_auth_key`; the checked-in configs contain only `<AUTH_KEY>`. Rendered configs
  have owner-only permissions. No site credentials are needed in this directory.
- Forced DSpark acceptance defaults to 1.9 accepted draft tokens per step for 3 draft tokens and
  2.51 for 5, matching the AgentX golden acceptance lengths minus 1. The generated text does not
  represent the model's real acceptance. Set `FORCE_ACCEPTED_TOKENS=` in `point.env` for real acceptance;
  validate accuracy separately.
- Set `FABRIC_ENV` to your host MPI transport and devices. Empty, the worker's UCX settings use TCP
  between nodes. No partition, account, RDMA device selection or node list is specific to a site.
- The client installs into node-local temporary storage unless `CLIENT_VENV` names a prepared venv.
  Use `client/prepare.sh` inside the same image to fill `CLIENT_HF_HOME` once. Set `CLIENT_OFFLINE=1`
  only after preparing both the venv and dataset cache. Long runs at high concurrency need enough
  client memory; no additional client memory patch is included.

## Storage and results

Set `STAGE_DIR` to node-local NVMe when the shared checkpoint storage is slow. `slurm/stage.sh`
copies the checkpoint onto each server node and reuses complete copies across jobs. Keep it empty
when `MODEL_PATH` is already on fast storage. Account for checkpoint staging and cold compilation
when setting job and health-check timeouts.

Every job writes `$WORK_DIR/<date>-<point>[-smoke]/`, with rendered configs, a manifest, server/client
logs, aiperf results and `slurm-<job>.out`. The final line is `RESULT PASS|FAIL ...`. A smoke passes
when all servers and the orchestrator become healthy, the client exits successfully, the result
file exists and the request error rate is at most 10%.

This code-and-config combination has not yet been built or run on GPUs as a package. Validate it
with an image built from this checkout before using it for performance claims. Site overlays,
environment dumps, existing auth values and raw benchmark logs are outside the portable package.
