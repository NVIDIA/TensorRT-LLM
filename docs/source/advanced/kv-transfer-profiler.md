<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

# Correlated KV transfer profiler events

Select `kv` in `TLLM_PROFILE_CUSTOM_REGIONS` before starting a process, and
start an external PyTorch profiler or execution-trace observer. KV profiling
does not start a profiler or change runtime profiling endpoints. See the
[shared metadata contract](profiler-metadata.md) for process-start selection,
supported literal types, `TRTLLM1` framing, and observational error behavior.

The current KV transceiver factory wraps its selected Python/NIXL or C++
runtime only when KV profiling is selected. Configuration validation, backend
resolution, Mamba restrictions, and runtime selection still happen first.
When KV is disabled, the factory returns the original transceiver unchanged.

| Event | Meaning |
| --- | --- |
| `trtllm::kv_transfer_submit` | CPU scope around the first asynchronous send or receive admission. |
| `trtllm::kv_transfer_sync` | CPU scope around a blocking receive call. |
| `trtllm::kv_transfer_complete` | Correlated observation of a completed transfer. |
| `trtllm::kv_transfer_failed` | Correlated observation of failure or authoritative admission rejection. |
| `trtllm::kv_transfer_cancelled` | Correlated observation of safe cancellation or successful transceiver shutdown. |

Submit scopes measure CPU admission, not asynchronous fabric duration. The
zero-work terminal markers indicate when a caller observes retirement; the
gap from submit to terminal can include scheduling and polling delays. It is
not a GPU-kernel interval or precise network-transfer timer. A blocking
receive scope includes waiting, followed by its observed terminal marker.

## Identity and metadata

Each locally tracked transfer carries `request_id`, `transfer_id`, `role`
(`context` or `generation`), `direction` (`send` or `receive`), resolved
`backend`, instance `rank`, and `prompt_tokens`. `aliases` records separate
`request`, `transfer`, and `status` numeric domains: identical numbers in
different domains do not imply the same observation.

`transfer_id` is the disaggregated request ID when present, otherwise the
peer context request ID, otherwise the local request ID. Zero is valid.
The status ID follows the existing owner: Python uses its disaggregated
local handle; C++ uses the native request ID. A peer context ID is not a
replacement local status ID. Correlate terminal and submit payloads using
the identity fields together with role, direction, rank and event order;
request reuse begins a new local observation.

Metadata is encoded lazily only for an active scope, then cached for that
observation's terminal. Prompt length reflects the first active metadata
extraction, not necessarily factory construction or submission when the
profiler was inactive. Enabled but inactive profiling constructs no payload
or encoded string, although it retains lightweight correlation bookkeeping.
Changing the environment does not reconfigure a running process.

## Ownership, cancellation and limits

The adapter delegates all status calls exactly once with their original
arguments, even with no tracked requests. It adds no polling, waits,
collectives, cancellation attempts, or resource-release decisions. It
preserves named status results, backend exceptions, capability properties,
reuse, diagnostics, and shutdown behavior.

Python sends expose precise opt-in terminal outcomes at their existing safe
retirement boundary, including cancellations omitted from ordinary status
lists or folded into bridge errors. This bounded table contains no request
references and is removed as observations retire. A retired boolean alone
does not identify an outcome. Python receive cancellation comes from the
named `cancelled_requests` list. C++ receives report through request-state
mutation with empty status lists; timeout-related native failures reported
as `DISAGG_TRANS_ERROR` are recorded as failed, not guessed cancellations.

`cancel_request(False)` is retryable and emits no terminal. A later True
means the existing owner permits safe release, and emits cancellation once.
Duplicate submissions still reach the backend but do not duplicate an
existing observation. Terminal records are removed independently by
direction; stale status IDs cannot re-emit a retired marker.

Admission exceptions are not automatically safe-release evidence. Python's
authoritative ownership predicate keeps partially admitted observations
pending until retirement. Native admission can be unknown; after an
exception the adapter holds only a weak request observation. Its expiry
removes profiling bookkeeping without emitting a terminal or releasing
native resources. Non-weak-referenceable unknown requests lose observation
immediately. A distinct request reusing the same direction/status ID also
discards the older uncertain observation: numeric outcomes cannot distinguish
those generations. This does not release resources or emit a terminal marker.
These cases may therefore have a submit marker without a
terminal. The original exception always remains the caller's outcome.
Refused or raising shutdown retains pending observations for retry;
successful shutdown emits cancellation and clears them.

Profiling costs include bookkeeping, pending-record scans, callbacks, and
active encoding. These CPU markers do not promise per-replay Python events
under CUDA graphs, native transport coverage, or cross-process clock
synchronization. Actual Python/C++ transfer and multi-rank validation are
separate from CPU contract tests.
