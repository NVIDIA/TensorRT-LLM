<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# BART decoder launch-path optimization opportunities

This note records the remaining host-side bottlenecks observed while profiling
the BART PyTorch encoder-decoder path, along with possible ways to reduce GPU
launch starvation. The proposals are diagnostic follow-up work, not verified
performance improvements, except where a completed experiment is marked
verified.

## Profile context

The analysis uses the Nsight Systems report:

```text
/tmp/bart-encoder-piecewise-profile/piecewise-direct.nsys-rep
```

The run used continuous admission with a maximum scheduled decoder batch of 32
requests. Generation step 208 contained 31 generation requests and no context
requests. It therefore already used the padded batch-32 decoder CUDA graph
bucket. Raising concurrency can amortize host overhead, but does not address
the remaining launch cost at a fixed concurrency.

A blank region between NVTX ranges means that the CPU work is not covered by an
NVTX annotation. It does not by itself mean that the CPU thread is idle. In
this trace, the gaps before `_update_requests` and `_fetch_new_requests` mostly
overlap forward kernels from the current batch. The interval after
`prepare_resources`, however, substantially overlaps a genuinely idle GPU.

## Representative launch-path breakdown

For generation step 208, the host path from the end of `prepare_resources` to
the end of `[Executor] _forward_step` decomposed as follows:

| Host region | Profiled duration |
| --- | ---: |
| Before `[Executor] _forward_step` | 126.9 us |
| `[Executor]` entry to `_prepare_inputs` | 76.2 us |
| `_prepare_inputs` | 707.4 us |
| `_prepare_inputs` end to `cudaGraphLaunch` | 102.9 us |
| `cudaGraphLaunch` API call | 306.9 us (The experiment confirms the ~300 µs cudaGraphLaunch duration is Nsight tracing overhead.) |
| Remaining host unwind | 36.0 us |

These are durations under CUDA and NVTX tracing, not unprofiled latency
measurements. In particular, the approximately 280--310 us steady-state
`cudaGraphLaunch` duration is unusually high for a reused graph and may include
substantial profiler overhead. It must be confirmed with an unprofiled host
timer around graph replay, or a less intrusive graph-level trace, before
treating it as an implementation bottleneck. The experiment below provides
that confirmation.

## Promising approaches

### Reduce encoder-decoder input preparation

Cache stable generation metadata for an unchanged batch, including request IDs,
sequence slots, prompt lengths, cross-KV lengths and pointers, and the decoder
graph key. Update only sampled tokens, positions, and KV-length deltas on each
step.

Fine-grained debug-only NVTX ranges were added to the native encoder-decoder
fast path. Set `TLLM_NVTX_DEBUG=1` to enable them. The profiles are:

```text
/tmp/bart-encoder-fast-input-profile/native-fast-full-c32-r256.nsys-rep
/tmp/bart-encoder-fast-input-profile/native-fast-fine-c32-r256.nsys-rep
```

The first report has coarse ranges with less annotation overhead. Across its
893 generation-only fast-path calls, the representative P50 breakdown was:

| Fast-path region | P50 |
| --- | ---: |
| Complete encoder-decoder fast path | 513.8 us |
| Input-ID staging | 161.3 us |
| Attention-metadata preparation | 145.9 us |
| Cross-attention preparation | 37.9 us |
| Host-buffer retirement | 35.8 us |
| Position-ID staging | 30.5 us |
| Sequence-length staging | 15.8 us |
| Native request collation | 15.0 us |

The second report subdivides input-ID and attention-metadata preparation. In
generation step 208, which contained nine generation requests, input-ID
staging took 198.1 us under tracing:

| Input-ID subregion | Profiled duration | CUDA API work |
| --- | ---: | --- |
| Copy previous batch indices | 33.2 us | One asynchronous copy |
| Gather sampled tokens | 76.9 us | Two kernel launches |
| Copy gathered tokens | 28.2 us | One asynchronous copy |
| Fill graph-padding tokens | 23.3 us | One kernel launch |

The complete input-ID region issued three kernel launches and two asynchronous
copies.

#### Sampled-token staging consolidation (Verified: clean end to end +1.0%)

The sampled-token experiment replaced advanced indexing followed by a copy
with `torch.index_select(..., out=input_ids_cuda_slice)`. It also retains the
device sequence-slot indices while the ordered generation request IDs remain
unchanged. Any non-encoder-decoder preparation path invalidates that cache.

The optimized profiles are:

```text
/tmp/bart-encoder-fast-input-profile/sampled-token-direct-c32-r256.nsys-rep
/tmp/bart-encoder-fast-input-profile/sampled-token-direct-cache-c32-r256.nsys-rep
```

| Fine-grained P50 | Before | Direct gather | Direct gather plus index reuse |
| --- | ---: | ---: | ---: |
| Complete encoder-decoder fast path | 612.0 us | 560.8 us | 519.1 us |
| Input-ID staging | 194.9 us | 142.8 us | 109.3 us |
| Sampled-token gather/copy | 103.5 us | 58.3 us | 58.9 us |
| Previous-index copy count | 907 | 907 | 202 |

Index reuse eliminated 705 of 907 previous-index copies. For generation step
208, the sampled-token portion fell from 105.0 to 48.7 us and changed from two
kernel launches plus one asynchronous copy to one kernel launch. A standalone
unprofiled CUDA microbenchmark reduced this operation from 29.1 to 10.6 us at
batch 9 and from 29.6 to 14.3 us at batch 32.

Because CUDA API tracing inflates the profiled ranges, a diagnostic experiment
temporarily placed `time.perf_counter_ns()` timers around the generation-only
`model_engine.forward()` call and input-ID staging. The timers did not
synchronize the GPU, did not emit per-step output, skipped the first 256
generation steps, and collected 5,785 samples per run. Both NVTX environment
switches were disabled. The fine-grained `nvtx_range_debug` context managers
were still present as null context managers, however, so this established the
direction of the host-path change but was not a completely
instrumentation-free measurement.

| Unprofiled host region | Before | Optimized | Reduction |
| --- | ---: | ---: | ---: |
| Input-ID staging, mean | 142.652 us | 94.670 us | 33.6% |
| Input-ID staging, P50 | 131.785 us | 81.866 us | 37.9% |
| Complete generation forward launch, mean | 657.765 us | 615.086 us | 6.5% |
| Complete generation forward launch, P50 | 600.186 us | 550.230 us | 8.3% |

A fully clean end-to-end experiment then physically removed all added
fine-grained ranges from `model_engine.py` and `trtllm.py`, removed the host
timers, and ran without Nsight. It alternated optimized and pre-change code for
four concurrency-32, 2,048-request runs per version:

| Order | Version | Mean latency | Makespan | Requests/s | Output tokens/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Optimized | 213.794 ms | 13.822200 s | 148.167 | 9953.842 |
| 2 | Before | 217.299 ms | 14.046938 s | 145.797 | 9794.590 |
| 3 | Optimized | 215.025 ms | 13.905088 s | 147.284 | 9894.507 |
| 4 | Before | 217.570 ms | 14.039977 s | 145.869 | 9800.087 |
| 5 | Optimized | 216.035 ms | 13.956041 s | 146.746 | 9855.947 |
| 6 | Before | 215.996 ms | 13.968241 s | 146.618 | 9849.773 |
| 7 | Optimized | 214.392 ms | 13.863271 s | 147.728 | 9924.353 |
| 8 | Before | 217.528 ms | 14.061549 s | 145.645 | 9784.413 |

| Four-run average | Before | Optimized | Change |
| --- | ---: | ---: | ---: |
| Mean latency | 217.098 ms | 214.812 ms | -1.05% |
| P50 latency | 202.168 ms | 200.133 ms | -1.01% |
| P90 latency | 345.073 ms | 340.892 ms | -1.21% |
| Makespan | 14.029176 s | 13.886650 s | -1.02% |
| Requests/s | 145.982 | 147.481 | +1.03% |
| Output tokens/s | 9807.216 | 9907.162 | +1.02% |

Two of the eight runs differed slightly in natural-EOS placement, once on
each version; output-token throughput gives the same approximately 1.0% result
after accounting for that small workload variation. The clean comparison
therefore confirms a modest end-to-end gain, while the diagnostic host timers
and Nsight profiles explain where it originates.

Attention-metadata preparation took 188.7 us in the same step:

| Attention-metadata subregion | Profiled duration | CUDA API work |
| --- | ---: | --- |
| Stage prompt and KV lengths | 40.7 us | Two asynchronous copies |
| Update host metadata | 23.6 us | Host-only tensor updates |
| Copy KV block offsets | 75.1 us | One asynchronous copy plus event query/record |
| Bind runtime views | 10.0 us | Host-only view binding |

KV block-offset staging is the largest individual metadata subregion. Reusing
its pinned staging storage and avoiding the copy when the request-to-block
mapping is unchanged should be measured after token staging is consolidated.
This optimization must preserve the existing completion-event lifetime rules
for overlapped scheduling.

Nested NVTX annotations and CUDA API tracing materially inflate all absolute
durations in the fine-grained report. The ranges establish relative
attribution; they are not unprofiled latency measurements. In particular,
native request collation is already small, while mixed admission makes
cross-attention preparation dominant only on the relatively infrequent
batch-change iterations.

The representative `_prepare_inputs` range issued 28 CUDA API calls:

- Six asynchronous copies.
- Three small preparation kernels.
- Three event queries.
- Four event records.

The remaining calls were stream-state queries and kernel-name lookups recorded
by the profiler. Packing the small metadata transfers into one pinned buffer,
fusing the preparation kernels, or capturing both into the decoder graph would
reduce launch-critical work.

#### Metadata, retirement, and position staging follow-up

Three safe changes were prototyped together and separately:

- Bound KV block-offset staging to the columns required by the batch's maximum
  KV length.
- Re-record the completion event associated with an available host-buffer set
  instead of constructing another event.
- For decoder CUDA-graph replay, copy position IDs directly from their pinned
  host buffer into the graph's static position tensor and skip the otherwise
  redundant device-to-device copy in `CUDAGraphRunner.replay()`.

Reusing the completion event and bounding the block-offset copy were neutral
in two 2,048-request runs:

| Two-run average | Before | Event + block bound | Change |
| --- | ---: | ---: | ---: |
| Mean latency | 214.292 ms | 214.212 ms | -0.04% |
| Makespan | 13.855197 s | 13.849266 s | -0.04% |
| Requests/s | 147.815 | 147.879 | +0.04% |

The direct position-ID path by itself was also within run-to-run noise:

| Two-run average | Before | Direct position staging | Change |
| --- | ---: | ---: | ---: |
| Mean latency | 208.198 ms | 207.623 ms | -0.28% |
| Makespan | 13.458528 s | 13.421465 s | -0.28% |
| Requests/s | 152.172 | 152.597 | +0.28% |

One pair favored the position change by 0.80%, while the next favored the
baseline by 0.24%. A longer 8,192-request B--O--B run then compared all three
safe changes without Nsight or debug NVTX instrumentation:

| Order | Version | Mean latency | Makespan | Requests/s | Output tokens/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Before | 225.061 ms | 57.744981 s | 141.865 | 10517.278 |
| 2 | Optimized | 227.479 ms | 58.368348 s | 140.350 | 10404.954 |
| 3 | Before | 231.405 ms | 59.377079 s | 137.966 | 10228.189 |
| Baseline average | Before | 228.233 ms | 58.561030 s | 139.916 | 10372.734 |

The optimized run was 0.31% faster than the average of its surrounding
baselines, but those baselines differed by 2.75%. All three runs generated the
same 607,320 output tokens with the same output hash. The measured change is
therefore not distinguishable from environmental drift, and none of these
three code changes was retained.

An additional attempt cached prompt lengths and a pinned KV block-offset
snapshot, skipping H2D staging when their host contents appeared unchanged.
That is not safe under the current overlapped metadata ownership contract: the
2,048-request output changed from 137,584 to 150,955 tokens and produced a
different hash. The prototype was removed. Future block-table reuse must use
an explicit cache-manager generation/ownership contract rather than infer
device-buffer validity from equal host contents.

### Graph replay timing without tracing (Verified: 10 us launch, 47 us runner)

An environment-injected timer measured graph replay during the
`timed_generate` range of the same concurrency-32, 256-request workload. The
benchmark ran as plain Python without Nsight Systems or CUDA API tracing. The
timer was injected into the MPI executor worker and used
`time.perf_counter_ns()` immediately around both:

- `torch.cuda.CUDAGraph.replay()`, which is the direct counterpart of the
  profiled `cudaGraphLaunch` API call.
- `CUDAGraphRunner.replay()`, which additionally includes graph lookup and
  copies of input IDs and position IDs into the graph's static tensors.

The native extension available to this worktree did not contain
`prepare_encoder_decoder_inputs`, so the runs forced the Python collation
fallback. This matches the earlier eager-versus-piecewise profile setup and
does not change the graph replay implementation being timed.

| Run and scope | Calls | Mean | P50 | P99 |
| --- | ---: | ---: | ---: | ---: |
| Run 1, CUDA graph replay | 767 | 10.493 us | 10.293 us | 16.677 us |
| Run 2, CUDA graph replay | 767 | 10.331 us | 9.881 us | 20.569 us |
| Run 3, CUDA graph replay | 767 | 10.542 us | 10.209 us | 18.724 us |
| Run 3, complete decoder graph runner | 767 | 47.394 us | 46.718 us | 61.594 us |

The median cost of an empty timer pair was 66--69 ns. All three runs produced
the same output hash and natural-EOS count. Their end-to-end mean latencies
were 217.267, 218.838, and 224.212 ms, respectively; these absolute values
describe the fallback diagnostic rather than the optimized native collation
path.

The untraced replay is about 27--30 times shorter than the 280--310 us
steady-state `cudaGraphLaunch` calls in the Nsight trace. Graph launch itself
therefore is not the approximately 300 us bottleneck implied by the traced
timeline. Even the full graph runner is below 50 us at the median, and only
about 10 us of that is the native graph replay. Further work should prioritize
input preparation and the host path leading into the runner rather than graph
executable reuse or upload.

### Reuse the input-copy completion event

The model engine constructs and records a new completion event after every
encoder-decoder input preparation. Re-recording the event owned by an
available host-buffer set preserved correctness, but the experiment above
showed no measurable end-to-end benefit.

### Remove redundant pre-forward Python work

The overlap loop sorts generation requests every iteration even though the
ordering correction is only needed for disaggregated generation. The
aggregated BART path can skip that list allocation and key-function traversal.

A qualified generation-only fast path can also skip:

- Context-token summation.
- Context-logit checks.
- Encoder-output attachment.
- Cache-indirection lookup when it is not used.

Using an NVTX context directly instead of constructing a decorated nested
function on every iteration removes another small fixed cost.

### Reduce fixed stream-handoff overhead

The stream waits around forward implicitly create, record, wait on, and destroy
CUDA events. Reusing preallocated handoff events, or keeping forward and its
dependent sampling work on one stream when KV transfer is inactive, can remove
this fixed overhead. Any change must preserve the ordering required by KV
onboard/offload and asynchronous sampling.

### Prepare batch-static metadata earlier

Split decoder input preparation into:

- A batch-static portion: request ordering, block tables, cross-attention
  metadata, and graph selection.
- A token-dependent portion: sampled tokens, positions, and KV-length
  increments.

The batch-static portion can run while the preceding GPU step is executing.
Once its sampled token is available, the launch-critical path should contain
only a compact device update and graph replay.

### Continue decode while encoder futures are pending (Verified: neutral end to end)

Encoder-init requests and existing generation requests are independent. The
executor submits encoder forward to one persistent host worker immediately
before decoder forward, keeps the encoder request IDs in the scheduler's
in-flight set, and retains the returned futures in FIFO order. At the start of
each later iteration, the executor polls the oldest future with `done()` and
queries its CUDA completion event. Neither operation waits. Only after both
report completion does the main executor thread publish the encoder output,
transition the request from `ENCODER_INIT` to `CONTEXT_INIT`, and remove its ID
from the in-flight set. This gives request-state mutation and error handling
back to the main executor thread while preventing both duplicate encoder
submission and premature decoder-context admission.

This removes the prior same-iteration `future.result()` barrier. In
`/tmp/bart-encoder-pending-futures-c32-r256.nsys-rep`, the encoder range from
40.744076 to 40.768364 seconds overlaps host execution of generation-only
decoder steps 211 through 218. Other steady-state encoder ranges similarly
overlap four to nine generation-only decoder forwards. The executor therefore
continues fetch, schedule, input preparation, forward, and sampling work while
the encoder future is pending.

GPU concurrency remains limited. In the 2.371-second timed window, encoder
stream 21 had 66.106 ms of kernel-busy time, decoder stream 17 had 41.064 ms,
and sampler stream 7 had 14.210 ms. Encoder kernels overlapped decoder kernels
for 0.595 ms and sampler kernels for 0.075 ms, about 1.0% of encoder busy time.
The host pipeline is now independent, but the large BART kernels still consume
most available device resources.

The compatible temporary native mixed-batch binding used by the earlier
measurements was cleaned before this experiment. The following A/B runs
therefore forced the generic Python input path for both versions; they isolate
the pending-future scheduling change but are not directly comparable to the
native-fast-path throughput above.

| Pair | Version | Mean latency | Makespan | Requests/s |
| ---: | --- | ---: | ---: | ---: |
| 1 | Blocking encoder | 240.779 ms | 15.563809 s | 131.587 |
| 1 | Pending futures | 243.423 ms | 15.714822 s | 130.323 |
| 2 | Blocking encoder | 242.596 ms | 15.690760 s | 130.523 |
| 2 | Pending futures | 240.687 ms | 15.532974 s | 131.849 |
| Average | Blocking encoder | 241.688 ms | 15.627285 s | 131.055 |
| Average | Pending futures | 242.055 ms | 15.623898 s | 131.086 |

Average throughput changed by +0.02% and mean latency by +0.15%, both within
run-to-run noise. The blocking baseline produced 137,584 tokens and hash
`93761bbaed28ad0f` in both runs. Pending-future execution changed batch
composition and produced 137,594 and 137,545 tokens; the natural-EOS and
length-stop request counts remained identical. A 64-request token-level check
found only request 27 diverged, beginning at generated token 94, with the same
128-token output length. This is consistent with a close greedy decision
changing under different BF16 batch numerics rather than request/output
misassociation.

The change removes a real software barrier but does not improve this workload's
end-to-end performance. A material gain still requires coarser encoder replay
(for example, a whole-encoder CUDA graph) or kernels that leave complementary
GPU resources available.

#### Prioritize decoder kernels over encoder kernels (Rejected)

An experiment created the encoder-decoder execution stream with CUDA priority
`-1` while retaining priority `0` for the encoder stream. Decoder-only models
kept priority `0`. CUDA stream priority favors pending work on the
higher-priority stream when the GPU scheduler can choose new work, but it
cannot preempt an encoder kernel that is already running.

The clean benchmark alternated default and high decoder priority at concurrency
32 for 2,048 requests. Both versions used the generic encoder-decoder input
path because the loaded native extension predates
`prepare_encoder_decoder_inputs`. No Nsight or CUDA API tracing was active.

| Order | Decoder priority | Mean latency | P50 latency | P90 latency | P99 latency | Makespan | Requests/s | Output tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 242.986 ms | 227.216 ms | 374.525 ms | 461.013 ms | 15.685588 s | 130.566 | 8764.224 |
| 2 | -1 | 244.408 ms | 228.988 ms | 380.483 ms | 462.689 ms | 15.791230 s | 129.692 | 8711.038 |
| 3 | 0 | 244.749 ms | 229.705 ms | 381.593 ms | 456.817 ms | 15.825666 s | 129.410 | 8689.302 |
| 4 | -1 | 245.734 ms | 231.712 ms | 379.946 ms | 460.800 ms | 15.923400 s | 128.616 | 8644.699 |

| Two-run average | Priority 0 | Priority -1 | Change |
| --- | ---: | ---: | ---: |
| Mean latency | 243.868 ms | 245.071 ms | +0.49% |
| P50 latency | 228.461 ms | 230.350 ms | +0.83% |
| P90 latency | 378.059 ms | 380.215 ms | +0.57% |
| P99 latency | 458.915 ms | 461.745 ms | +0.62% |
| Makespan | 15.755627 s | 15.857315 s | +0.65% |
| Requests/s | 129.988 | 129.154 | -0.64% |
| Output tokens/s | 8726.763 | 8677.869 | -0.56% |

Output lengths varied by at most 0.13% between runs; token-normalized
throughput therefore gives the same conclusion as request throughput. The
benchmark records final-response latency rather than per-token inter-token
latency, so it does not exclude a small latency redistribution from existing
generation requests toward replacement encoder requests. It does show that
the proposed priority does not improve end-to-end request latency or
throughput for this workload. The stream-priority change was not retained.

#### Re-evaluate encoder batch waiting (Current settings retained)

The pending-future scheduler was adjusted at concurrency 32 because encoder
launch and completion no longer block the decoder host loop. A 256-request
screen covered iteration deadlines 24, 32, 40, 48, and 64 and token-threshold
ratios 0.08, 0.12, 0.1708984375, and 0.25. The existing setting is 48
iterations and ratio 0.1708984375, which corresponds to 11,200 tokens or
approximately one full batch of 32 average-length encoder inputs.

The short screen selected 64 iterations and ratio 0.08, but an alternating
2,048-request validation rejected it: it averaged 129.14 requests/s versus
130.71 requests/s for the existing configuration. The less aggressive
64-iteration, 0.12-ratio candidate was positive in two 2,048-request pairs,
improving mean latency and requests/s by 1.50% and output-token throughput by
1.61%. A longer confirmation reduced that result to noise and exposed worse
median and tail latency:

| Order | Iterations / ratio | Mean latency | P50 latency | P90 latency | P99 latency | Makespan | Requests/s | Output tokens/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 48 / 0.1708984375 | 256.200 ms | 245.680 ms | 379.094 ms | 452.209 ms | 32.940426 s | 124.346 | 8883.522 |
| 2 | 64 / 0.12 | 254.405 ms | 247.532 ms | 373.355 ms | 463.342 ms | 32.761208 s | 125.026 | 8936.117 |
| 3 | 48 / 0.1708984375 | 254.777 ms | 243.711 ms | 373.563 ms | 452.560 ms | 32.788736 s | 124.921 | 8928.371 |
| 4 | 64 / 0.12 | 255.028 ms | 247.642 ms | 374.130 ms | 465.251 ms | 32.799612 s | 124.880 | 8925.166 |

| Two-run average | 48 / 0.1708984375 | 64 / 0.12 | Change |
| --- | ---: | ---: | ---: |
| Mean latency | 255.489 ms | 254.717 ms | -0.30% |
| P50 latency | 244.696 ms | 247.587 ms | +1.18% |
| P90 latency | 376.329 ms | 373.743 ms | -0.69% |
| P99 latency | 452.385 ms | 464.297 ms | +2.63% |
| Makespan | 32.864581 s | 32.780410 s | -0.26% |
| Requests/s | 124.634 | 124.953 | +0.26% |
| Output tokens/s | 8905.947 | 8930.642 | +0.28% |

Both versions used the generic encoder-decoder input path because the loaded
native extension predates `prepare_encoder_decoder_inputs`; neither used
Nsight or CUDA API tracing. The 0.26--0.28% throughput change is within the
observed run-to-run variation, while the P50 and P99 regressions are larger.
The existing 48-iteration, 0.1708984375-ratio configuration was therefore
retained.

#### Drain the decoder before admitting another encoder wave (Rejected)

A stricter scheduling experiment stopped admitting encoder requests whenever
the scheduler had any decoder-context or generation requests. New requests
therefore accumulated until the entire active decoder wave completed, at which
point the scheduler released the waiting encoder requests together. This
formed larger encoder batches, but also made every replacement request wait
for the longest output in the preceding decoder wave.

The clean concurrency-32 comparison used 1,024 requests, the same dataset and
request order, no Nsight or CUDA API tracing, and the generic encoder-decoder
input path because the loaded native extension predates
`prepare_encoder_decoder_inputs`.

| Policy | Mean latency | P50 latency | P90 latency | P99 latency | Makespan | Requests/s | Output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current bounded encoder accumulation | 225.910 ms | 199.376 ms | 387.849 ms | 472.617 ms | 7.383563 s | 138.686 | 8276.492 |
| Drain decoder completely | 306.773 ms | 285.259 ms | 425.774 ms | 526.375 ms | 10.012829 s | 102.269 | 6109.162 |
| Change | +35.8% | +43.1% | +9.8% | +11.4% | +35.6% | -26.3% | -26.2% |

The larger encoder waves do not compensate for the head-of-line admission
barrier. In particular, short decoder requests cannot be replaced until the
longest request in the current wave finishes, leaving capacity unused during
the decoder tail. The strict policy was removed; bounded encoder accumulation
continues to admit replacement work while generation proceeds.

#### Preallocate stable encoder output and prepare decoder context early

The stronger follow-up publishes stable encoder-output views before submitting
the encoder worker. `ModelEngine` allocates one exactly sized output tensor for
the encoder batch, each request receives a view into that tensor, and the
worker copies the final hidden states into those stable addresses. A CUDA event
is also attached before submission. A separate recorded flag prevents the main
thread from querying or waiting on the event until the worker has enqueued it.
This removes the lifetime hazard that otherwise prevents decoder-context
preparation from starting before encoder completion.

The scheduler can now capacity-plan those requests in `CONTEXT_INIT` while the
encoder worker is active. The executor removes requests whose encoder event is
not ready from the current decoder batch, retains its generation requests, and
prepares the deferred requests' persistent KV, cross-KV, and sequence-slot
resources in that generation-only iteration. The requests remain in the
executor's in-flight set so they cannot be scheduled twice. Once the recorded
event queries ready, the executor releases them to a later mixed decoder batch.
That mixed batch skips resource allocation already completed by the lookahead
pass and waits once on the shared encoder event before consuming the stable
output views.

This is host/preparation overlap and launch-bubble removal. It does not attempt
to execute decoder-context kernels before the encoder event is satisfied, and
it retains generation-only decoder iterations while the encoder kernels run.
The optimization is enabled only when every active non-null resource manager is
one of the KV-cache, cross-KV-cache, or sequence-slot managers and a cross-KV
manager is present. Other configurations use the non-blocking encoder future
path without early resource preparation.

In
`/tmp/bart-encoder-stable-lookahead-early-c32-r256.nsys-rep`, there are 916
decoder forwards and 928 `prepare_resources` ranges. The 12 additional calls
occur immediately before generation-only forwards while encoder work is still
active. For example, decoder-context preparation runs from 66.385924 to
66.386937 seconds while the encoder runs from 66.381330 to 66.412223 seconds.
The following generation-only forward starts at 66.387195 seconds. When the
encoder event becomes ready, mixed forward 342 performs only a 98 us resource
preparation before starting at 66.414147 seconds.

Across steady-state encoder admissions, excluding the two startup groups, the
median timings changed as follows:

| Admission interval | Pending-future baseline | Stable-output lookahead | Change |
| --- | ---: | ---: | ---: |
| Encoder worker return to mixed-forward start | 2.665 ms | 2.405 ms | -9.8% |
| Last encoder GPU operation to first decoder GPU operation | 3.081 ms | 2.900 ms | -5.9% |
| Last encoder GPU operation to first decoder GEMM | 5.400 ms | 5.295 ms | -1.9% |

One representative pair shows the intended effect more clearly, although the
mixed batch shapes are not identical. The pending-future trace's 13-context,
15-generation batch starts 2.622 ms after its encoder worker returns and its
first decoder input operation begins 3.027 ms after the last encoder GPU
operation. The lookahead trace's 15-context, 12-generation batch reduces those
intervals to 1.924 and 2.253 ms, respectively.

Clean unprofiled measurements used concurrency 32, 2,048 requests, and forced
the generic encoder-decoder input path in both temporary overlays because the
available compatible native binding predates the current collation interface.
The repeated results remain noisy:

| Version | Runs | Median mean latency | Median makespan | Median requests/s |
| --- | ---: | ---: | ---: | ---: |
| Pending-future baseline | 5 | 245.028 ms | 15.871215 s | 129.039 |
| Stable-output lookahead | 4 | 242.236 ms | 15.658510 s | 130.794 |

The medians correspond to -1.14% mean latency, -1.34% makespan, and +1.36%
throughput. Individual optimized runs ranged from 127.942 to 136.162 requests/s,
so this is a small directional gain rather than a statistically decisive
end-to-end improvement. The trace provides the stronger causal result: the
resource work moves earlier and the steady-state admission bubble shrinks, but
most of the path to the first decoder GEMM remains elsewhere in input
preparation and model launch.

#### Run encoder and decoder context together on the side stream

A stronger experiment moves the dependent decoder context-only forward into
the encoder worker. The worker executes the encoder and then its decoder
context batch on stream 21, while the main executor continues launching
generation-only CUDA graphs on stream 17.

The implementation uses a second `PyTorchModelEngine` that shares the BART
model but owns separate decoder input buffers, attention metadata, and graph
runner state. CUDA graphs are disabled for the context engine. KV, cross-KV,
and sequence-slot resources are reserved on the main executor thread because
their managers mutate shared allocation state; decoder input preparation and
model launch remain in the worker. Only one worker context batch can be active,
so its engine and stable encoder-output storage cannot be reused early.

Two ordering details are required for correctness:

- The main sampler has reusable device storage and already has one outstanding
  generation sample under overlap scheduling. Completed context logits are
  therefore sampled only after the prior main-lane sample has been retired.
- A worker context batch is not the main executor's previous batch.
  `py_batch_idx` is cleared after its sampled token reaches the host request,
  causing the first main-lane generation step to use the explicit host-token
  admission path.

The two engines have distinct destination buffers, but their asynchronous
metadata copies are staged from shared resource-manager state. Reusing that
state before an H2D copy completed corrupted subsequent generation tokens.
Each lane now waits only for its input-copy event after enqueueing the complete
forward. This does not wait for model kernels: host preparation can proceed on
both threads, and the already-enqueued model work remains concurrent on the two
CUDA streams.

The implementation is restricted to single-rank BART and mBART with overlap
scheduling, KV-cache manager V1, no attention DP, drafting, guided decoding,
KV-cache transfer, connector, or early first-token response. Other
configurations retain the stable-output lookahead path.

The profile is:

```text
/tmp/bart-encoder-context-worker-c32-r256.nsys-rep
```

During its 2.651-second `timed_generate` interval, stream 21 executes 96.831 ms
of encoder plus context kernels. Stream 17 executes 973.694 ms of decoder CUDA
graphs. Their kernels overlap for 20.641 ms, or 21.3% of stream-21 kernel time.
Steady worker `_run_encoder_context_step` host ranges also overlap seven to ten
main generation `_forward_step` ranges each. The requested host and GPU
concurrency is therefore present, although most stream-21 work still cannot
co-reside with the large decoder kernels.

Clean unprofiled measurements alternated the stable-output lookahead baseline
and this worker-context version at concurrency 32 with 2,048 requests:

| Order | Version | Mean latency | P50 latency | Makespan | Requests/s | Output tokens/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Stable-output lookahead | 245.145 ms | 232.250 ms | 15.862079 s | 129.113 | 8671.499 |
| 2 | Worker encoder + context | 264.948 ms | 251.086 ms | 17.171612 s | 119.267 | 8091.727 |
| 3 | Stable-output lookahead | 240.438 ms | 225.812 ms | 15.510191 s | 132.042 | 8869.717 |
| 4 | Worker encoder + context | 263.153 ms | 251.669 ms | 17.003768 s | 120.444 | 8199.771 |

| Two-run average | Stable-output lookahead | Worker encoder + context | Change |
| --- | ---: | ---: | ---: |
| Mean latency | 242.792 ms | 264.051 ms | +8.76% |
| P50 latency | 229.031 ms | 251.378 ms | +9.76% |
| Makespan | 15.686135 s | 17.087690 s | +8.94% |
| Requests/s | 130.578 | 119.856 | -8.21% |
| Output tokens/s | 8770.608 | 8145.749 | -7.12% |

Separating context from the main mixed batch changes BF16 batch numerics and
therefore a small number of greedy decisions. The worker runs generated 1.18%
more output tokens on average; output-token throughput still regressed by
7.12%, so output length does not explain the result. The approximately
20.6 ms of hidden stream-21 work is smaller than the cost of separate eager
context launches, sampling/finalization, input-staging fences, and GPU resource
contention. This design proves that the work can overlap, but it is not an
end-to-end performance improvement for the measured workload.

## Suggested experiment order

1. Add fine-grained NVTX ranges inside the encoder-decoder input fast path.
   (Completed.)
2. Gather sampled tokens directly into the persistent input-ID buffer and
   avoid restaging stable previous-batch indices. (Completed; approximately
   1.0% throughput improvement across four fully clean interleaved runs.)
3. Reuse KV block-offset staging storage and skip unchanged block-table copies.
   (Attempted; unsafe without an explicit device-buffer ownership contract.)
4. Reuse or ping-pong `_prepare_inputs_event` and measure the host-time change.
   (Completed; no measurable end-to-end gain.)
5. Measure broader stable encoder-decoder metadata reuse. (Position staging and
   bounded block-table staging completed; no verified end-to-end gain.)

These experiments determine whether the next large gain is stable metadata
reuse or consolidation of device-side input updates.
