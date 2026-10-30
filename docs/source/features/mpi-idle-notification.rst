.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Idle request notification for MPI tensor parallelism
==================================================

Idle request notification is an **opt-in** host-CPU wait policy for non-pipelined
MPI tensor/context parallelism. When the leader's request queue is empty, followers
wait on an authenticated notification instead of remaining inside the
request-count MPI collective. New work resumes the existing count and payload
broadcasts. Sampling and the model computation are unchanged.

This feature is disabled by default. It does not promise an application
throughput gain or transparent recovery after a rank fails.

Configuration
-------------

Set these variables before creating the executor, and propagate them to every
participating MPI rank through the launcher's normal environment mechanism:

``TLLM_MPI_IDLE_NOTIFICATION``
    ``0`` or unset disables notification. ``1`` enables the supported path.
    Other values are rejected. The selection must agree on every MPI rank;
    inconsistent selections reject startup before the notification transport
    is created. This is not a live toggle for an already running executor.

``TLLM_MPI_IDLE_NOTIFICATION_ADDRESSES``
    A JSON object mapping short hostnames to locally assigned, peer-reachable
    IPv4 addresses. Keys use the hostname before its first dot. Every rank
    needs the entry for its own host; using one complete shared map avoids
    inconsistent launch configuration. This variable is only required when
    notification is enabled.

For example, create a launcher configuration file using the following
**placeholder schema**, replacing all placeholders with the current deployment
information rather than copying addresses from another machine:

.. code-block:: json

    {
      "<short-hostname-1>": "<locally-assigned-peer-reachable-IPv4-1>",
      "<short-hostname-2>": "<locally-assigned-peer-reachable-IPv4-2>"
    }

Then configure the existing MPI launch environment:

.. code-block:: bash

    export TLLM_MPI_IDLE_NOTIFICATION_ADDRESSES="$(cat /path/to/interface-map.json)"
    export TLLM_MPI_IDLE_NOTIFICATION=1
    # Launch the application with these variables propagated to all MPI ranks.

Do not put credentials or authentication keys in this map. The existing
``ZeroMqQueue`` creates authentication material and exchanges endpoints through
MPI bootstrap. It selects an available port; no fixed port is required in the
map. The configured address must bind locally, and must not be loopback,
unspecified or multicast. Successful local binding alone does not establish
cross-host reachability or firewall admission.

To opt out on the next executor creation, set the same selection on every rank:

.. code-block:: bash

    export TLLM_MPI_IDLE_NOTIFICATION=0
    unset TLLM_MPI_IDLE_NOTIFICATION_ADDRESSES

Supported modes
---------------

This expanded candidate supports a non-pipelined MPI TP/CP group with at least
two ranks, TP times CP equal to the MPI world, PP1 and ``MPI_THREAD_MULTIPLE``.
Optional KV-pool rebalance retains its existing pre-fetch executor-loop check;
it does not add an independent idle wake source or periodic notification timer.
This expansion is source-reviewed preparation, not distributed validation.
The built-in NGram drafter and its built-in NGram pool are supported: draft
preparation executes inline over scheduled generation requests. Standalone
Suffix Automaton (SA) is supported when the initialized selected model exposes
the built-in SA worker and its metadata refers to the executor's built-in SA
pool. SA executes inside model forward; its pending copies/masks are flushed
on the next forward, and its kernels use the existing current CUDA stream.
Retained completed-request pool contents are cached state, not a background
completion job. Both private and global SA pools retain their original policy.

SA with an additional drafter/draft engine, custom worker/pool, unreviewed
auxiliary state handlers, or missing/mismatched initialized metadata is not
admitted. Custom NGram subclasses/pools and other speculative implementations
remain implementation limits of this candidate, not fundamental
incompatibilities with notification.

The existing batching wait is also supported. On an idle fetch, the whole
original queue operation is deferred until followers are parked, so it still
starts exactly one batch deadline after the initial request/drain. A request
already queued when an idle executor resumes may take the idle notification
route; active traffic keeps its previous route. No additional batching knob,
deadline or queue polling loop is introduced.
It rejects explicit enablement with the following modes rather than silently
disabling model features:

* attention-DP, Ray or TorchDist;
* pipeline parallelism;
* connector or disaggregated KV transfer;
* resource-governor activity;
* speculative decoding other than the admitted built-in NGram/standalone SA paths;
* executor sleep/wakeup;
* Gen-admission timing/clock synchronization.

Synchronous multimodal item scheduling is retained only when the executor and
model engine agree on its enabled state and scheduler ownership. Unsupported
configurations keep their existing behavior when notification is disabled.

Lifecycle and limits
--------------------

Startup validates the selected mode and interface on all ranks and duplicates
the metadata communicator before executor workers start. Each worker creates,
uses and closes its own notification sockets. The ready path keeps its existing
count/payload behavior; active single-round broadcasting is not routed through
notification. No Python polling thread or periodic wake timer is added.

Cancellation and shutdown continue through the existing request queue. The
notification protocol must deliver their normal request-count/payload path;
it is not a separate request or cancellation transport. It retains the existing
launcher and rank-crash containment rather than recovering a failed MPI world.
Normal-loop communicator disposal also runs if a local profiler or queue close
raises during teardown; a genuinely failed loop does not wait in collective
cleanup before launcher containment.

The initial opt-in policy reflects the explicit interface configuration,
restricted progress modes and pending current-port distributed validation.
Supported-mode automatic enablement with opt-out can be reconsidered after
interface/capability discovery, progress ownership, shutdown/failure and wake
latency are covered. Unconditional global enablement is not implied.
