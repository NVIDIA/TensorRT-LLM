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
profiler. Use an active PyTorch profiler with an execution-trace observer and
shape recording to retain the metadata string input. Selection, literal
decoding, and observational error handling follow the
[shared profiler metadata contract](profiler-metadata.md).

Current `ConfigurableMoE` schedulers emit `trtllm::moe_routing` user scopes only
when selected and profiling is active. Each scope's `TRTLLM1` envelope contains
these fields:

| Field | Meaning |
| --- | --- |
| `moe.layer` | Layer index, or `None` if unavailable |
| `moe.num_tokens` | Rows of this materialized local chunk |
| `moe.num_experts` | Logical expert count, not EPLB slots |
| `moe.experts_per_token` | Top-k column count |
| `moe.selected_expert_ids` | Integer row lists in original top-k order |
| `moe.routing_weights` | Original floating weight row lists, or `None` if unavailable |

Observations occur before EPLB maps logical expert IDs to slots and before
input-weight folding or backend-required scale dtype conversion. The external
scheduler supports ordinary separated routing and materialized fused
route-plus-quantization outputs without re-running routing or quantization.
The fused-communication scheduler observes real rows after attention-DP
padding stripping, including its existing empty `[0, top_k]` chunk tensors.
Unavailable weights in the standalone helper are explicit `None`, never
fabricated ones; current scheduler requirements for materialized weights are
unchanged.

Events describe physical chunk work, not unique requests or global token counts.
External DP padding, substituted empty chunks, and repeated layer execution can
produce additional observations. Existing RouterReplay row offsets and duplicate
suppression are independent of these events. Profiling does not change scheduler,
backend, communication, quantization, or EPLB selection and adds no collectives.

## Cost and coverage limits

Active eager observations detach, copy expert IDs and available weights to CPU,
convert them to lists, and serialize a literal payload. CUDA copies synchronize
and can perturb chunk overlap. Payload size grows with rows times top-k; this
is not a zero-overhead path. Disabled or selected-but-inactive calls do not read
tensor values, copy to CPU, query CUDA capture state, or construct metadata.
Active CPU-only metadata does not initialize CUDA.

During genuine current-stream CUDA graph capture the new profiler observation
is skipped before CPU serialization. The existing graph-capable RouterReplay
device-buffer write remains active; its persistent buffer must still be warmed
before capture under its existing contract. Graph replay executes no new Python
hook, so replay-only profiling cannot produce fresh per-replay routing events.

Fully backend-internal fused routing that exposes no selected IDs or weights
is unavailable. No routing is recomputed, separated routing forced, backend
substituted, or weights fabricated to obtain a trace. Empty observations contain
empty ID/weight lists and retain the top-k count. Unsupported/non-finite payloads
may lose the observation under the shared codec's error policy without changing
inference.

CPU helper and scripted-scheduler tests are protected by the existing MoE CPU
CI parent list. Genuine CUDA graph/RouterReplay tests and eager overhead timing
are local/QA checks; they require a supported GPU/runtime and do not establish
multi-rank or full MoE kernel coverage on their own.
