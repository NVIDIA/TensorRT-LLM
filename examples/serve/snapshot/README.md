<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Snapshot Startup Prototype

For the native capture/restore workflow, see [STANDALONE.md](STANDALONE.md).
The sections below describe its Phase 0 measurement tool.

`scripts/snapshot_probe.py` is a Phase 0 measurement tool, **not a production
restore controller**. It launches an isolated native `trtllm-serve` candidate,
checks HTTP health, sends deterministic completion requests, and compares a
restore candidate against a cold baseline. No Dynamo, Kubernetes or GMS client
is required by the tool. An external Snapshot host adapter is still required to
produce and launch a restored candidate.

Snapshot restores execution state. TensorRT-LLM must establish serving validity.
A successful generation probe does not establish capture safety, valid KV
ownership, participating-rank agreement or permission to accept user traffic.
The report keeps all those qualification gates `UNTESTED`, even on success.

## Run A Cold Baseline

Use a Linux GPU environment with the selected TensorRT-LLM build and model.
Create `profile.json` with the exact model/weight revision, quantization/layout,
build/container and driver versions, GPU/rank topology, graph buckets and
transport. Its contents are operator declarations, not measured attestation.
Use the same file for cold and restored attempts; the tool compares its digest.

Create `requests.json` with a JSON array such as:

```json
[
  {
    "model": "probe-model",
    "prompt": "The capital city of Germany is",
    "max_tokens": 8,
    "temperature": 0,
    "stream": false
  }
]
```

Include several fresh prompts and lengths within the selected graph profile.
Use non-sensitive prompts. Reports and logs can contain generated text or paths.
The tool creates a new mode-0700 directory for every attempt. It rejects existing
directories instead of deleting their contents.

```bash
python scripts/snapshot_probe.py \
  --mode cold --profile profile.json --requests requests.json \
  --run-dir /tmp/startup-cold-01 --timeout 600 -- \
  trtllm-serve /path/to/model --served_model_name probe-model \
  --host 127.0.0.1 --port 0 --report_addr '{address}' --config config.yaml
```

Use one frontend. The existing native `--report_addr` option publishes the
kernel-selected port atomically. `{address}` must be one complete argv element.
The probe does not use a shell, follow redirects or honor HTTP proxy settings.
It accepts numeric loopback addresses only. Keep the GPU workers and the HTTP
candidate isolated from production discovery and traffic.

## Probe A Restored Candidate

Supply a separately reviewed foreground host adapter that restores the complete
candidate process tree. Pass its address-report argument as `{address}`. It must
publish the **restored candidate's** loopback address, remain alive throughout
the probe, and keep local descendants in its process group. The tool terminates
that group after every attempt. Remote ranks, daemonized processes and external
resources need adapter-owned cleanup; the probe does not manage them.

Run the same command structure with `--mode restore` and
`--baseline /tmp/startup-cold-01/report.json`, replacing the native cold-launch
argv with that adapter's argv. There is no bundled restore command yet. Merely
passing a cold launcher under `--mode restore` can pass the generation probe;
it cannot qualify Snapshot. Do not use that result as evidence of fast restore.

The adapter must prove source termination, a complete compatible artifact and
isolated restore. It must not fall back silently to model initialization. Record
its evidence separately. Do not create `ready-for-snapshot` based on `/health`
or `/memory_status`. Neither endpoint proves that all requests and transfers
have retired. This tool does not emit capture or activation signals.

## Read The Evidence

`report.json` records the command, profile/request digests, observed completions,
failure reason, and launch-to-HTTP-health/first-completion times. Exit zero means
only that the generation probe passed. Restore comparisons require exact text,
finish reason and completion-token count equality. A cold result proves output
availability, not model accuracy. Establish the baseline's accuracy separately.
Nondeterministic profiles can fail exact comparison and need investigation, not
an automatic waiver. Missing, malformed and failed outputs are failures.

Repeat each exact profile with fresh run directories. Report successful/total
attempts alongside p50/p95 timings; retain every failure. First-completion time
includes health polling and completion validation, so it is not raw engine
restore time. Capture cost, artifact I/O, CPU/CUDA restore, graph reuse and HBM
usage need separate instrumentation. No speedup is claimed by this prototype.

## Next Qualification Gates

1. Agree on the non-Kubernetes host interface and full MPI restore boundary.
2. Prove capture quiescence across all requests and transfers. Preserve storage
   until the owning layer proves that future accesses cannot occur.
3. Validate memory-ready, then KV/scheduler/communication runtime-ready, then
   serving-ready across the configured cohort. Reuse existing lifecycle and
   membership authorities. Do not equate `/memory_status` with serving-ready.
4. Terminate the source and verify weight/graph reuse plus fresh generation.
   Qualify MPI single-GPU, single-node multi-GPU and multi-node separately.
5. Add admission controls, compatibility enforcement and failure injection before
   offering a production restore path. Keep GMS, live-request preservation and
   interrupted-transfer resumption out of this initial scope.

References: [Snapshot TRTLLM guide](https://github.com/ai-dynamo/snapshot/blob/bc9d2161d2a7f203551e5b7e237defd90b1e9aa1/docs/guides/tensorrt-llm.md),
[workload contract](https://github.com/ai-dynamo/snapshot/blob/bc9d2161d2a7f203551e5b7e237defd90b1e9aa1/docs/reference/workload-contract.md),
[memory-control proposal](https://github.com/NVIDIA/TensorRT-LLM/pull/19016).
