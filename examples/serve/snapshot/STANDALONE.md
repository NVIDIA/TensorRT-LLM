<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Capture and Restore

This is an opt-in, same-host experiment, not a production-ready restore service.
See [FEASIBILITY.md](FEASIBILITY.md) for the boundary and limits.
No Dynamo, Kubernetes, GMS or memory-release cycle is required.

Use a dedicated Linux x86_64 GPU test host with permission to run CRIU. Keep the
same host boot, PID/mount/network namespaces, filesystem, GPU assignment and
tool binaries for capture and restore. Do not use a shared production server.
The adapter refuses occupied source PIDs and incompatible artifacts. It does
not package container filesystems, remap GPUs or move templates between nodes.

Build the matched `cuda-checkpoint-helper` from Snapshot
[`bc9d2161d2a7`](https://github.com/ai-dynamo/snapshot/tree/bc9d2161d2a7f203551e5b7e237defd90b1e9aa1)
and provide it with CRIU in `SNAPSHOT_BIN`. `criu check` must succeed in the
actual execution environment. This adapter is experimental TRTLLM code, not an
existing standalone `snapshotctl` command. Model/build identity in `profile.json`
is operator-declared, not an attestation.

## Run the Prototype

Create `profile.json` with these nonempty fields: `model_revision`,
`weights_digest`, `quantization`, `layout`, `image_digest`, `trtllm_revision`,
`graph_buckets`, and `topology`. For the initial single-GPU profile use
`{"nodes":1,"tp":1,"pp":1,"cp":1}`. Pin the model revision and immutable runtime
image. Start with Qwen3-0.6B and the supplied `config.yaml`.

Use the same profile, prompts, model and serving options for the cold probe and
the captured candidate. `requests.json` is the deterministic request list
described in [README.md](README.md). Keep `--served_model_name model` consistent
with its model field.

```bash
export LLM_MODELS_ROOT=/path/to/models
export MODEL="$LLM_MODELS_ROOT/Qwen3-0.6B"
export SNAPSHOT_BIN=/absolute/path/to/matched/tools
export UCX_TLS=tcp,self
export TLLM_NCCL_SYMMETRIC_ZERO_COPY=0

python scripts/snapshot_probe.py --mode cold --run-dir /work/cold \
  --profile profile.json --requests requests.json -- \
  trtllm-serve "$MODEL" --config examples/serve/snapshot/config.yaml \
  --served_model_name model --host 127.0.0.1 --port 0 --report_addr '{address}'

python scripts/snapshot_startup.py capture --artifact /work/template \
  --snapshot-bin "$SNAPSHOT_BIN" --profile profile.json -- \
  trtllm-serve "$MODEL" --config examples/serve/snapshot/config.yaml \
  --served_model_name model --host 127.0.0.1 --port 0 --report_addr '{address}'

python scripts/snapshot_startup.py restore --artifact /work/template \
  --snapshot-bin "$SNAPSHOT_BIN" --profile profile.json \
  --requests requests.json --baseline /work/cold/report.json
```

Capture stops user admission before it can begin, waits for warmed rank
evidence, checkpoints GPU state, then dumps and terminates the whole process
tree. It publishes the manifest only after successful capture and source death.
On restore, the coordinator verifies hashes and compatibility, restores CPU and
CUDA state, then supplies a fresh session. The native HTTP gate returns 503 to
ordinary clients until private generation matches the cold baseline.

Each trial retains host logs and a JSON report. Restore exits after verification
and terminates its candidate by default. Add `--serve` to keep the validated
loopback server running; interrupt the coordinator to stop it. Run restore again
only after the previous candidate has fully exited. Artifacts and control tokens
must remain private: they contain model data and can contain process secrets.
Do not restore untrusted or manually edited artifacts.
If CRIU fails before returning a restored root PID, its partial process tree
cannot be safely identified by this prototype. Keep the test environment isolated,
inspect the CRIU log and remove the dedicated test container before retrying.
Do not reuse resources merely because the coordinator exited.

The template boundary precedes `PyExecutor.start_worker()`, so capture does not
accept live requests or KV transfers. This is not a live-server drain API.
Private generation exercises the resumed executor and CUDA graphs. No new model
constructor or warmup is invoked by the restore command. Source inspection and
graph identity checks do not substitute for GPU execution and timing evidence.

## Phase 2: Fixed MPI Cohort

For a single-node TP profile, change `topology.tp` and pass matching `--tp_size`
to **both** cold and capture commands. Place the matched `cuinterpose-launch`,
`cuinterpose-coordinator`, `libcuinterpose.so` and `libcuinterpose_core.so` beside
the CUDA helper. The capture command automatically launches through the
interposer when TP exceeds one. Use that same interposer for the cold baseline
as well, by prefixing its `trtllm-serve` command with:

```text
/absolute/tools/cuinterpose-launch --library /absolute/tools/libcuinterpose.so --
```

The coordinator inspects and prepares the complete CUDA sharing cohort before
locking devices. After CPU and CUDA restore, it restores shared mappings before
releasing TRTLLM ranks. Every rank checks weight/buffer addresses, graph handles,
KV pool addresses and capacity metadata. V2 additionally checks reserved cache
owners. These checks preserve the captured clean metadata; they do not rebuild
reference counts or reset a live scheduler. MPI ranks must agree on the template
and fresh session before all-rank runtime-ready evidence is accepted.

The validation lease expires if the coordinator disappears. With `--serve`,
the coordinator renews a five-second admission lease. Expiry blocks new HTTP
requests; it does not cancel already running requests or free their resources.
Administrative state remains in the private control directory outside the
captured process tree. An external orchestrator can implement this file protocol;
this PR does not implement a Dynamo/Kubernetes adapter or second membership service.

### Qualification Limits

Single-GPU and single-node TP are prototype execution paths, **not qualified
deployment profiles** until their real GPU capture/restore tests pass. Multi-node,
P/D workers, remote KV connectors, PP/CP/attention-DP, GMS and live-request capture
are rejected. Whole-cohort same-host TCP/MPI restore is not peer rediscovery or
transport reconstruction for a survivor or P/D partner. These remain Phase 2
delivery gaps, together with portable artifacts and exact-profile performance
qualification. Live-KV migration and interrupted-transfer resumption are deferred.

Repeat a cold trial and several restores with the same configuration. The
restore timing values are cumulative from adapter entry and include artifact
hash verification. Report the preflight, CPU restore, CUDA restore, memory-ready,
runtime-ready, first completion, validated responses and serving-ready boundaries.
Do not compare a restore that excludes integrity I/O against a complete cold
startup. Archive success/attempt counts and logs; compute p50/p95 only from real
successful runs. Retain failed attempts separately.
