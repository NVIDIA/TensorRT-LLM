<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Standalone resident checkpoint coordinator (experimental)

This example runs on the CPU, outside both the worker snapshot and the MPI job.
It drives the authenticated engine controls from [PR #21598](https://github.com/NVIDIA/TensorRT-LLM/pull/21598).
It does not change normal `trtllm-serve` startup. It does not require Dynamo, Kubernetes,
GMS or KV sleep. The first target is one host and one GPU: dense Qwen3, MPI TP1,
warmed graphs and resident GPU-only KV V2 without prefix reuse or external KV transfer.

**This is the coordinator, not a qualified Snapshot deployment.** You also need a
compatible engine and an independently authorized site adapter. No production host
adapter or privileged service is installed by this example. The experimental driver
from [PR #21481](https://github.com/NVIDIA/TensorRT-LLM/pull/21481) is a candidate
backend, not a drop-in adapter. Do not use its `PASS` alone to admit traffic.

## Run one capture and restore

1. Provision a private state directory on a local filesystem with hard-link and
   directory-fsync support. Keep it outside the captured container and MPI failure
   domain. Only the coordinator user may write it. Protect the artifact store separately.
2. Choose a new template ID (32 lowercase hex characters). At source launch, use it as
   the Snapshot container's ownership label. Start the compatible engine with
   `--enable_checkpoint_mode`, warm it and generate the deterministic reference for
   this exact model/profile. Save `prompt_token_ids` and `expected_token_ids` in
   `probe.json`; see the [engine's reference generation instructions](https://github.com/chienchunhung/TensorRT-LLM/blob/cfab58992fd154781ad980f058359528fc635d8a/docs/source/features/checkpoint-lifecycle.md#operations).
3. Supply the source control credential through `TRTLLM_RUNTIME_CONTROL_API_KEY`.
   Invoke the coordinator, not the worker, with the full immutable source container ID:

   ```sh
   python examples/serve/checkpoint/coordinator.py \
     --state "$PRIVATE_STATE" --host-adapter "$SITE_ADAPTER" --template "$TEMPLATE" \
     capture --source "$SOURCE_CONTAINER_ID" --probe probe.json
   ```

4. Supply the independently authorized destination credential through the same
   environment variable. Restore from the published manifest:

   ```sh
   python examples/serve/checkpoint/coordinator.py \
     --state "$PRIVATE_STATE" --host-adapter "$SITE_ADAPTER" --template "$TEMPLATE" restore
   ```

Capture returns the template ID only after source termination and manifest publication.
Restore returns the destination container ID only after engine activation **and**
host publication. A new restore uses a fresh attempt, not the source session.
The coordinator never requests source continuation, relocation or cold fallback.

## Site adapter boundary

The following is a proposed **local adapter contract**, not a Snapshot API. The
absolute executable is trusted deployment configuration, not workload input.
The coordinator sends one JSON request on stdin and reads one final JSON object
on stdout. A nonzero exit, a timeout, a mismatched receipt, or a progress marker
is not success. The adapter must supervise host jobs independently of the CLI's
lifetime. The engine key is not forwarded to the executable.

```text
request: {version: 1, action, attempt, template, deadline_monotonic, input: {...}}
receipt: {version: 1, action, attempt, template, status: "succeeded", result: {...}}
```

Both processes use the same host monotonic clock. Honor the deadline **on the host**,
including before committing an operation. No evidence or deadlines survive reboot
as authorization. Serialize operations for each owned resource. Never accept a
client-selected PID, arbitrary runtime socket, or a captured credential as proof of ownership.

| Action | Input | Required effect and result |
| --- | --- | --- |
| `bind_source` | `source` | Verify the immutable source ID and ownership label. Bind a private control channel to that container. Return `source, control_port`. This read-only binding must expire at its deadline without stopping the source. |
| `capture` | `source, identity` | Only after the engine's current capture-safe proof, create the ready marker in this source's fresh private control mount. Run capture, verify artifact integrity and source termination, then commit the artifact. Return `source, identity, artifact_digest, source_terminated: true, snapshot`. |
| `restore` | `source, identity, artifact_digest` | Recheck the stored bytes against the digest. Create a distinct, single-use inert target with fresh control paths. Check host/image/device compatibility, run restore, and independently bind the private control channel and current controller authority to the observed destination. Keep all public ingress closed. Return `source, target, identity, artifact_digest, control_port, isolated: true, authority_established: true, host_result_id: attempt, snapshot`. |
| `publish` | `target` | Verify the same attempt still owns that target. Open ingress only after engine activation and successful `/ready` on that destination. Keep normal health checks active. Durably record the outcome; return `target, published: true`. |
| `discard` | None | Contain or clean up only the resources owned by this attempt, including late host work. Return `contained: true` only after confirming containment. It does not mean resume the source or remove a published template. |

`identity` binds the source engine session, template, profile and cohort. The
adapter must enforce that binding, not echo it without checking. Keep destination
ingress isolated even if validation or activation completes and the coordinator
dies. A destination cannot use credentials, replay state or listener identity
copied from the source as evidence of fresh authority.

`snapshot` is the final driver result (or its unchanged required fields) from
the pinned experiment. Match its revision, operation, attempt, artifact, container
and status. Capture also requires `source_terminated`; restore requires
`memory_ready` and a positive `restored_pid`. These fields do **not** replace
integrity checks, compatibility checks, engine validation or serving authority.

## Failure and qualification

The private state directory records intent before mutation. A rejected source
preflight does not stop the worker. Once prepare may have been dispatched, an
uncertain result requires containment. Even a visible manifest is unusable while
`pending-capture.json` exists: capture publication is not confirmed. On pre-publication restore failure the
coordinator asks the host to contain the attempt; it never opens ingress as a
timeout fallback. If publication might have occurred, do not blindly discard or
retry. Use the recorded attempt and target to reconcile the host's durable outcome.
An unresolved `active-restore.json` blocks another restore for this template.
Remove this interlock only after an operator verifies ownership, routing and
containment. Never recover by deleting records from an active coordinator.

CPU tests cover control ordering and negative receipts; they do not execute CRIU
or CUDA restore. Qualification requires an approved host and a source-terminated
restore that serves correct fresh tokens, replays warmed graphs and does not
reload or transform weights or recapture graphs. Test stale artifacts, deadlines,
coordinator loss and repeated cycles. Qualify TP2 separately before extending
the single-rank coordinator gate.
