<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# On-demand startup benchmark

Compare checkpoint I/O policies on the **same exclusive node, installed runtime,
checkpoint and serving configuration**. Each trial starts a fresh server, waits
for `/health`, records `/server_info` and exits. No generation or load-generator
requests are sent. These tests are excluded from automatic CI stage selection.

## Measurement contract

- Primary end-to-end metric: monotonic time immediately before `trtllm-serve`
  process creation until the first successful `/health` response (100 ms polling).
  This includes model initialization, loading, and normal runtime warmup,
  autotuning and CUDA-graph capture performed before server readiness.
- Model-loading metrics: existing rank-0 total loading, checkpoint preparation,
  weight population, finalization and post-load processing timers; checkpoint
  pipeline is preparation + population + finalization. Do not interpret
  preparation or population alone as storage-I/O time.
- This is **not launch-to-first-token**: there is no inference request. Allocation,
  container import, checkpoint download, cache preparation and teardown are
  outside the timed interval. It does not separately time all LLM initialization,
  autotuning or graph-capture phases. Raw startup metadata and server logs are
  retained for additional phase instrumentation without changing the harness.
- Compare confirmed effective policies, not merely requested `auto`. Missing,
  conflicting, inactive or fallback policy evidence invalidates a loader
  comparison, while retaining the observed startup and failure artifacts.
  Policy logs are server-reported evidence, **not an all-rank telemetry census**.

## Cold-cache scope and safety

Every trial evicts all discovered main/draft/auxiliary checkpoint data files with
`POSIX_FADV_DONTNEED`, then requires **zero resident client pages** using Linux
`mincore`, without reading payloads. Complete numbered/indexed shard coverage and
file identities are checked. Supply unindexed auxiliary directories explicitly
in `checkpoint_dirs`; external draft models must also be listed there.

For NFS checkpoints, node-local FS-Cache must be disabled. Before eviction,
the runner matches each open file's `/proc/self/fdinfo` mount ID to
`/proc/self/mountinfo`, including container bind mounts. It rejects `fsc` or
`fsc=<tag>` in the effective superblock options; Linux reports these whenever
NFS FS-Cache is enabled (absence means the default `nofsc`). Missing, ambiguous
or malformed mount information fails closed. Per-file mount/source/options and
`nfs_fscache` status are saved in `result.json` and the CSV cache evidence. Mount
information is checked again after eviction, including an approved reset helper.
No remount or cache-service shutdown is performed.

Non-NFS checkpoints remain supported: their NFS check is `not_applicable`,
and only RAM page-cache coldness is verified. Other local-disk caching layers,
including caches underneath stacked filesystems, are **not** certified cold.
Keep checkpoint sources matched between policies and qualify the storage setup;
do not interpret `cache.verified` as proof that every storage layer is cold.

**NFS-server caches are allowed; they are not cleared or verified.** This is a
client-cold comparison, not a storage-backend-cold benchmark. Storage-appliance
and device caches are also outside scope. Run paired trials close together and
avoid concurrent storage-heavy jobs.

Eviction/verification runs before **every policy and repetition**, outside the
timed interval. During startup, the harness does not evict checkpoint pages or
disable normal buffered I/O: native caching and rank-striped read-ahead remain
unchanged. Exclusive checkpoint use and stable mounts are required so another
process cannot repopulate pages or change storage settings between verification
and launch.

Execution requires an exclusive whole-node allocation and explicit acknowledgment.
Never use `--exclusive-node` as a substitute for actually reserving the node.
On systems where residency is unavailable, permission-masked, or pages cannot be
evicted, the trial fails closed. Container root does not by itself prove that
residency verification is available. Qualify this on the target NFS mount first.
An optional `--cache-reset-command '["/approved/helper", "argument"]'` replaces
the eviction step, **not verification**. No shell, implicit sudo, global
`drop_caches`, or storage-admin action is performed automatically.

Linux child-subreaper/PID-fd teardown must prove all trial descendants have exited
before proceeding. Failed teardown aborts the campaign before another eviction.
Before and after each cache reset, the runner reaps exited children and allows
up to one second for transient children to exit. Remaining live or unverifiable
children block launch; `result.json` records process-check stages, reaped PIDs/exit
codes and remaining PIDs, parent PIDs, states, command names and start times.
If children exit during cache verification, eviction/verification is repeated
once after quiescence; repeated child activity fails closed. This does not weaken
the cold-cache checks or include cache preparation in startup timings.
Use Python 3.10+ on modern Linux for execution; planning/collection need no GPUs.

## Profiles

| Profile | Checkpoint client cache | Mutable runtime caches |
| --- | --- | --- |
| `application_cold` (default) | Evicted and verified before every trial | Fresh directories per trial |
| `loader_isolation` | Evicted and verified before every trial | Independent copies of an immutable prepared seed |

Controlled runtime directories include CUDA, Triton, TorchInductor, Torch
extensions, FlashInfer workspace, DeepGEMM JIT, Hugging Face modules and the
TRT-LLM autotuner cache. Offline Hugging Face mode prevents checkpoint downloads.
Image-provided libraries, precompiled kernels and `FLASHINFER_CUBIN_DIR` artifacts
remain part of the fixed runtime image. This is not a claim that every possible
kernel/backend cache is reset or that the entire OS is cold.

For `loader_isolation`, supply `--runtime-cache-seed /shared/cache-seed`. Prepare
it using the same model, hardware, runtime and serving envelope; run a preparation
trial with `--keep-runtime-cache`, then copy its `runtime_cache/` directory to an
immutable seed. Seed layout follows
`CACHE_PATHS` in `runner.py`, including `autotuner.json`. Clones prevent the first
policy from changing the second policy's starting state. Metadata fingerprints
detect seed changes; they are not cryptographic content attestations. Cache hits
must still be qualified: some compiled artifacts depend on absolute paths.

## Matrix: six core cases, two optional

| Case | Size group | Topology |
| --- | --- | --- |
| Qwen3.8 dense 27B NVFP4 | Small | TP1 |
| Same exact Qwen checkpoint | Small | TP4 |
| DeepSeek V4 Pro mixed FP4/FP8 | Large | TP8/EP8, MTP1 |
| Qwen3.8 MAX 2.4T NVFP4 | Large | TP8/EP1, MTP3 |
| GLM5.3 FP8 | Large | TP8/EP8 |
| Kimi K2.5 NVFP4 | Large | TP8/EP1 |
| Kimi K2.5 NVFP4 (optional) | Large | TP8/EP8 |
| DeepSeek R1-0528 FP8 (optional) | Large | TP8/EP1, MTP3 |

All are **pending target-node qualification**, not assertions of successful B300
fit. `matrix.yaml` cites source recipes and flags proposed envelopes. Set
`QWEN38_27B_MODEL` and `GLM53_MODEL` to pre-staged, pinned local snapshots; no
verified internal paths were available for those exact versions. Other paths
are relative to `LLM_MODELS_ROOT`. Do not substitute a different model/version
silently. Initially single-node TP/PP/EP only; no disaggregated or multi-node runs.

Default variants are `native` and explicit `rank_striped_read_ahead`. Add named
variants in a copy of the matrix as new policies land; each variant may change
only `checkpoint_io_policy`. The selected runtime must implement that policy.
Default personal campaigns use 3 repetitions (36 launches for 6 cases × 2
policies). Adjacent repetitions reverse policy order. Three repetitions are an
initial descriptive experiment, not a significance guarantee.

## Personal SLURM execution

Use a shared checkout and a **prebuilt image containing the runtime under test**.
The mounted checkout supplies QA scripts only: there is no build, editable
install or `PYTHONPATH` override. Record runner commit and installed runtime
version separately; runner commit is not proof of the installed product commit.
The installed wheel's `Source Commit` is recorded when available. A hash of the
QA source files identifies the runner even when mounted Git-worktree metadata
is unavailable. CI installs its newly built wheel into the test image; compare
that product identity as well as the base image.
Prefer an immutable image digest and preserve its build provenance.

```bash
export QWEN38_27B_MODEL=/shared/models/Qwen3.8-27B-NVFP4
export GLM53_MODEL=/shared/models/GLM-5.3

python jenkins/scripts/startup_benchmark/submit.py \
  --matrix jenkins/scripts/startup_benchmark/matrix.yaml \
  --repo /shared/TensorRT-LLM \
  --models-root /shared/models \
  --output /shared/results/startup-plan \
  --image "$RUNTIME_IMAGE" --partition "$PARTITION" \
  --cases qwen38_27b_nvfp4_tp1 --repeats 1
```

This is a **dry-run**: inspect `submission.json` and generated batch scripts.
The checked-in matrix allows one hour per startup (also the default when
`timeout_seconds` is omitted). With two policies and `--repeats 1`, the calculated
per-case allocation is **2h13m** without a reset helper, or **2h23m** with one.
These are scheduling budgets, not expected runtimes. The CLI default remains
three repeats (**6h19m** without a helper), which requires a longer-limit QoS or
partition. Check site limits before submitting; use `--repeats 1` for a pilot.
For slower models, increase `timeout_seconds` in a custom matrix and request a
sufficient allocation; `--time` alone does not change the startup deadline.
To submit, repeat with `--submit` and a **new empty** output directory. Supply
`--account`, `--constraint`, `--time` and extra `--mount` entries as required by
the cluster. Checkpoints selected by explicit environment paths must be covered
by mounts; preserve those variables in the submission environment. The default
walltime includes selected variants/repeats, startup/helper deadlines and setup
headroom; increase `--time` for slow cache cloning or residency verification.
One exclusive allocation runs all variants for one case, sequentially. Jobs are
chained with `afterany` to avoid contention between this campaign's cases.

On an already exclusive node inside the target image, invoke `runner.py run`
directly with the same matrix/case/profile/output options, `--runtime-image`
and `--exclusive-node`. Do not run one driver per rank.

After jobs finish:

```bash
python jenkins/scripts/startup_benchmark/runner.py collect \
  --output /shared/results/startup-run
```

Artifacts: `summary.json`, `results.csv`, `report.md`, per-trial `result.json`,
`server_info.json`, `server.log`, and effective YAML. Generated runtime caches
are removed after verified teardown by default to avoid large CI artifacts;
`--keep-runtime-cache` retains them for seed preparation.
Reports retain failed, invalid and missing planned trials; only exact-matched,
cache-verified, confirmed-policy pairs contribute to comparisons. Report
per-case median timings and paired reductions; no pooled fleet speedup is
calculated. Checkpoint fingerprints use file identities plus config/index
contents, not a payload hash that would warm the checkpoint before measurement.

## Explicit CI execution

After the PR/stage is available and the existing multi-GPU approval gate permits
it, request one shard, for example:

```text
/bot run --stage-list "DGX_B300-8_GPUs-PyTorch-Startup-OnDemand-1"
```

Eight shards partition the eight named pytest cases in
`l0_b300_startup_ondemand.yml`; use the generated test selection to identify the
case for a shard, rather than assuming numeric order. Each runs both policies
once under `application_cold`. These are opt-in and excluded from regular and
post-merge selections. Request shards sequentially for cleaner storage
conditions; requesting all at once can create storage contention. Qualification
paths and privileges must be available in the CI environment; stage registration
alone does not establish them.

The QA wrapper also accepts `TRTLLM_STARTUP_BENCHMARK_PROFILE`, `_VARIANTS`,
`_REPEATS` and `_RUNTIME_CACHE_SEED` when invoked directly. CI's default timeout
covers one paired repetition; use the personal CLI for larger campaigns.
CPU-only safety/collector/submission tests are covered by the existing
`unittest/tools` CPU test-list entry. No new regularly scheduled GPU work is added.
