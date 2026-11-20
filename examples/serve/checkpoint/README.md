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

**Experimental: this profile has not yet passed a real process/GPU restore.**
Use the bundled Docker adapter on an authorized Linux GPU host. No user-written
adapter is required. Ordinary `trtllm-serve` does not install or start any host
service. The first deployment is for a trusted local operator, not a multi-tenant
service; Docker daemon access is host-administrator authority.

Snapshot [requires PageBroker](https://github.com/ai-dynamo/snapshot/pull/251),
including without GPU CustomStorage. PageBroker and the workload can share the
same node; PageBroker must stay outside the workload snapshot. A platform can
provision the service once. Deploying this example does not grant permission to
run privileged helpers or change an existing cluster policy. The earlier
[experiment](https://github.com/NVIDIA/TensorRT-LLM/pull/21481) stopped before capture
when the cluster rejected PageBroker creation. A successful upstream Kubernetes
test is a useful comparison, not qualification of this standalone deployment.
No Snapshot-team workflow integration is needed for this experiment. An allocation
is useful only if its host owner permits this setup or provisions it in advance.

## One-time setup on the GPU host

This prototype targets **one allocated physical GPU on a Linux x86-64 host**
with local rootful Docker, Docker-managed containerd, Buildx and the NVIDIA
container runtime. It does not target a remote Docker context, a Kubernetes
workload pod or an unapproved shared node. Source and destination use the same
16-CPU and 128-GiB workload limits. The host-side Python adapter needs PyYAML.
Use a compatible workload image
containing the engine controls and the local model and kernel assets. The
setup step needs access to the pinned build inputs (GitHub and the image
registries); restore must not depend on downloading model weights.

An authorized host operator runs setup, not the inference worker:

```sh
export WARP_ROOT=/var/lib/trtllm-warp
# WARP_ROOT must be an existing, operator-owned, private (0700) directory.
python examples/serve/checkpoint/deployment.py setup --root "$WARP_ROOT" \
  --gpu "$GPU_UUID" --allow-privileged-host-driver
python examples/serve/checkpoint/deployment.py health \
  --config "$WARP_ROOT/deployment.json"
```

Setup builds the bundled helper against one pinned Snapshot revision and starts
or reuses this deployment's matching PageBroker. It does not adopt arbitrary
containers or grant access to Docker. The health command checks the local
deployment and queries the broker's capabilities; it does **not** prove a CUDA
restore. If the site disallows this deployment, stop before preparing a source.

## Run one capture and restore

1. Complete the one-time host setup first. Set `WARP_ROOT` to its private deployment
   directory. Provision a separate private state directory on a local filesystem with hard-link and
   directory-fsync support. Keep it outside the captured container and MPI failure
   domain. Only the coordinator user may write it. Protect the artifact store separately.
   Treat the artifact as sensitive: a resident process image can contain model
   weights and the source's control key. Do not share it or use a production key
   for this experiment.
2. Prepare a trusted local image containing the S3 engine, model and tokenizer at
   `/model`, populated runtime/kernel caches, and this [resident TP1 configuration](resident-tp1.yaml)
   at `/etc/trtllm/resident-tp1.yaml`. Pin the image by digest. Choose a new
   template ID and supply the engine key in the operator environment. The launcher
   labels the source, selects the allocated GPU, mounts the interpose library and
   starts the worker without an external listener.

   ```sh
   export TEMPLATE=$(python3 -c 'import secrets; print(secrets.token_hex(16))')
   SOURCE_CONTAINER_ID=$(python3 examples/serve/checkpoint/source_launcher.py \
     --deployment "$WARP_ROOT/deployment.json" --template "$TEMPLATE" \
     --image "$WORKLOAD_IMAGE" --probe-out probe.json)
   ```

   `--probe-out` sends a greedy request to the ready source and creates a new
   file with `prompt_token_ids` and `expected_token_ids` for this model. See the
   [engine's reference generation instructions](https://github.com/chienchunhung/TensorRT-LLM/blob/cfab58992fd154781ad980f058359528fc635d8a/docs/source/features/checkpoint-lifecycle.md#operations).
3. Supply the source control credential through `TRTLLM_RUNTIME_CONTROL_API_KEY`.
   Invoke the coordinator, not the worker, with the full immutable source container ID:

   ```sh
   python examples/serve/checkpoint/coordinator.py \
     --deployment "$WARP_ROOT/deployment.json" --state "$PRIVATE_STATE" --template "$TEMPLATE" \
     capture --source "$SOURCE_CONTAINER_ID" --probe probe.json
   ```

4. Restore from the published manifest. Keep the engine credential available to the
   coordinator. The fixed profile does not rotate the captured engine key: it proves
   destination ownership separately using the local host and a private control channel.

   ```sh
   TARGET=$(python examples/serve/checkpoint/coordinator.py \
     --deployment "$WARP_ROOT/deployment.json" --state "$PRIVATE_STATE" --template "$TEMPLATE" restore)
   ```

5. Send a **new** request to the published destination. Use the served model name,
   a different prompt from the validation probe, and compare the returned token IDs
   with a cold baseline for the same model and sampling settings:

   ```sh
   python examples/serve/checkpoint/coordinator.py \
     --deployment "$WARP_ROOT/deployment.json" --state "$PRIVATE_STATE" --template "$TEMPLATE" \
     generate --target "$TARGET" --model resident-tp1 --prompt-token-ids 17 42 --max-tokens 4
   ```

Capture returns the template ID only after source termination and manifest publication.
Leave that container **stopped, not removed**, for the template's lifetime: this
backend checks its immutable ID, image and exited state again before each restore.
Do not restart it or use automatic container cleanup.
Restore returns the destination container ID only after engine activation **and**
host publication. A new restore uses a fresh attempt, not the source session.
The coordinator never requests source continuation, relocation or cold fallback.

## Host adapter boundary

The following is a **local adapter contract**, not a supported upstream Snapshot API.
An operator can override the bundled backend with `--host-adapter /absolute/path`
when qualifying another deployment. Never accept the executable from a workload.
The coordinator sends one JSON request on stdin and reads one final JSON object
on stdout. A nonzero exit, a timeout, a mismatched receipt, or a progress marker
is not success. The adapter must supervise host jobs independently of the CLI's
lifetime. The engine key is not forwarded to the executable. The bundled backend
also binds the control port to an immutable `control_container`; it never treats
an unverified host loopback port as the restored engine.

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
| `publish` | `target` | Verify identity, engine activation and `/ready` on this destination. Record publication before making the target available to consumers. Return `target, published: true`. |
| `discard` | None | Contain or clean up only the resources owned by this attempt, including late host work. Return `contained: true` only after confirming containment. It does not mean resume the source or remove a published template. |

`identity` binds the source engine session, template, profile and cohort. The
adapter must enforce that binding, not echo it without checking. Keep destination
ingress isolated even if validation or activation completes and the coordinator
dies. A destination cannot use credentials, replay state or listener identity
copied from the source as evidence of fresh authority.

The bundled backend supports **operator-local inference only**. It never publishes
a TCP port. For capture and publication it receives fresh signed status headers
(not the key), and checks the live engine state from inside the owned container.
For generation it resolves the local publication record and rechecks the target
and its readiness. External ingress, routing, identity rotation, cross-node restore
and multi-tenant access are outside this profile.

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
or CUDA restore. On a permitted host, collect these observations before calling
the profile qualified:

1. Stop the captured source and verify that its task cannot serve requests.
2. On the distinct restored target, verify that the control listener belongs to
   the restored process tree. Before validation and activation, both `/ready`
   and ordinary generation must reject traffic, even if memory restore finished.
3. After activation and publication, send a new prompt and compare its tokens
   with the cold baseline. Collect logs/traces showing replay of the warmed graphs
   without weight loading, transformation or graph recapture.
4. Repeat the cycle. Record time to first successful generation against cold startup,
   and exercise stale artifacts, timeouts and coordinator loss. Qualify TP2 separately.
