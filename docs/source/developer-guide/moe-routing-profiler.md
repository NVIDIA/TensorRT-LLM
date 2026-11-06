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

# MoE routing profiler

Set `TLLM_PROFILE_CUSTOM_REGIONS=moe` before starting the process (or include
`moe` in a comma-separated selection). This is opt-in and does not start a
profiler. Use an active PyTorch profiler on the emitting thread, with an
execution-trace observer and shape recording to retain the metadata string input.
An observer alone is insufficient; Chrome exports omit the metadata fields.
Selection, literal decoding, and observational error handling follow the
[shared profiler metadata contract](profiler-metadata.md).

Current `ConfigurableMoE` schedulers emit `trtllm::moe_routing` user scopes before
EPLB remapping. These scopes inspect shapes only, without reading tensor values
or copying data to CPU:

| Field | Meaning |
| --- | --- |
| `moe.layer` | Layer index, or `None` if unavailable |
| `moe.num_tokens` | Rows of this materialized local chunk |
| `moe.num_experts` | Logical expert count, not EPLB slots |
| `moe.experts_per_token` | Top-k column count |

The external scheduler observes ordinary separated routing, materialized fused
route-plus-quantization outputs, and caller-provided `PreroutedRouting` tensors.
The fused-communication scheduler observes real rows after attention-DP padding
stripping, including its empty `[0, top_k]` chunks. Events describe physical chunk
work: DP padding, substituted empty chunks, and repeated execution can add
observations. Routing, weights, quantization, communication, and EPLB are unchanged.

## Expert IDs from Router Replay

To include expert-ID values, also enable the existing engine option
`enable_return_routed_experts`. Profiling does not enable Router Replay (R3)
automatically. The per-request `return_routed_experts` option controls returned
outputs; it is not needed for profiler metadata once engine capture is enabled.

R3 already copies pre-EPLB logical expert IDs to CPU after each forward. For
copies staged and processed while MoE profiling is active, the profiler emits
separate `trtllm::moe_routing_ids` scopes from the raw completed CPU buffer:

| Field | Meaning |
| --- | --- |
| `moe.layer` | Original model layer index |
| `moe.step`, `moe.rank` | Originating executor iteration and engine rank |
| `moe.num_tokens` | R3 forward-layout row count, including any dummy rows |
| `moe.experts_per_token` | Top-k column count |
| `moe.selected_expert_ids` | Logical integer IDs in original top-k order |

Each layer records all rows and top-k IDs from the completed R3 buffer, without
sampling or truncation. Routing weights are not recorded. ID records cover a
full forward layout rather than individual
scheduler chunks; they do not reconstruct cached prefix rows or fabricated
empty-chunk work. Shape coverage of `PreroutedRouting` does not add R3 capture
of caller-provided IDs; ID scopes report only the existing R3 buffer. They
describe logical experts, not physical EPLB slots or expert placement on ranks.

## Cost and coverage limits

The profiler adds no device buffer, copy stream, device-to-host copy, or wait.
CPU list conversion and literal encoding scale with the full row count, top-k,
and number of layers when IDs are emitted. Disabled or inactive MoE profiling
installs no R3 annotation callback and skips tensor-value serialization; the
helper/activity checks still execute.
R3 keeps its existing costs, including forced drains for prefill/request
completion and ring reuse. Enabling R3 solely for profiling therefore has a cost.

ID scopes are best-effort completion-time observations, not forward timings.
Their timestamps mark readback processing; use `moe.step`, `moe.rank`, and
`moe.layer` for source attribution. A source step may precede the current trace
window. Copies that complete after profiling stops are omitted; profiling adds
no final flush. Without R3, only shape scopes are available.

R3's existing restrictions apply, including pipeline parallelism, speculative
decoding/MTP, and backend paths without capturable separated routing. Profiling
neither broadens those paths nor recomputes routing to obtain IDs. The R3 buffer
must be warmed before CUDA graph capture under its existing contract.

Shape observations are skipped during CUDA graph capture and `torch.compile`
tracing, and Python hooks do not run on graph replay. R3 remains graph-capable;
its completed readbacks can produce ID scopes without matching shape scopes on
replay. The Vanilla/TRITON legacy implementations and model-private routing such
as Step3p7's Python expert fallback remain outside this ConfigurableMoE boundary. Unsupported
backend selection emits a warning.
