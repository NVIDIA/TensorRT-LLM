<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# BART PyTorch plan to approach legacy TensorRT performance

## Objective

Close the remaining BART continuous-admission performance gap between the
optimized PyTorch path and the legacy TensorRT path without changing output
semantics or regressing latency tails.

In the matched concurrency-32, 1,024-request benchmark, optimized PyTorch
reached 172.6 requests/s and 10,305 output tokens/s, while legacy TensorRT
reached 200.9 requests/s and 11,289 output tokens/s. PyTorch therefore reached
85.9% of TensorRT request throughput and 91.3% of its output-token throughput.

Recent measurements around the pending-encoder-future experiments used the
generic encoder-decoder input fallback because the loaded native extension
predated `prepare_encoder_decoder_inputs`. Those absolute results must not
replace the native-fast-path baseline above. Rebuild or load a compatible
extension before evaluating progress against legacy TensorRT.

## Profile-derived hypothesis

The remaining gap is not primarily caused by a lack of concurrent encoder and
decoder GPU execution.

The concurrency-32, 256-request profiles contain:

| Backend | Encoder passes | Decoder passes |
| --- | ---: | ---: |
| Legacy TensorRT | 180 | 681 |
| PyTorch pending-future design | 17 | 932 |

The backends can generate somewhat different output lengths because BF16 batch
composition affects close greedy decisions, so the pass counts are not a pure
scheduling comparison. Nevertheless, they show a major structural difference:
legacy TensorRT can run cheap replacement encoder microbatches and replenish
the decoder promptly, while PyTorch must coalesce expensive eager encoder
launches into large waves. The resulting PyTorch decoder batches spend more
iterations partially occupied.

The legacy profile also showed no encoder/decoder, encoder/sampler, or
decoder/sampler kernel overlap. Its performance does not depend on concurrent
execution across those streams. Earlier closed-batch measurements found about
217.9 ms of PyTorch GPU work versus 226.4 ms of legacy TensorRT GPU work.
Consequently, the primary target is cheap and timely decoder replenishment plus
lower per-iteration orchestration, not maximum GPU kernel overlap.

## Design 1: occupancy-aware encoder replenishment

### Motivation

The current encoder admission policy waits until either:

- Waiting encoder tokens reach
  `batch_wait_max_tokens_ratio * max_num_tokens`; or
- `batch_wait_timeout_iters` expires.

At concurrency 32, the configured token threshold is approximately 32
average-length encoder inputs. This gives efficient encoder execution, but it
can leave the decoder substantially underfilled while replacements wait.

The rejected full-drain policy moved in the wrong direction: it waited until
all decoder work completed before releasing another encoder wave. That added a
head-of-line barrier, reduced request throughput by 26.3%, and increased mean
latency by 35.8%.

### Proposed policy

Release waiting encoder requests when any of these conditions is true:

1. The waiting encoder request count reaches a modest microbatch target.
2. The active generation count falls below a decoder low watermark.
3. The oldest encoder request reaches a maximum iteration deadline.
4. There are no active decoder requests.

Continue accumulating replacements while the decoder remains sufficiently
full. This preserves encoder efficiency during the steady state but refills
decoder capacity before reaching a long, underfilled tail.

Initial screening matrix:

| Parameter | Values |
| --- | --- |
| Encoder microbatch target | 4, 8, 12 |
| Decoder low watermark at concurrency 32 | 20, 24, 28 |
| Maximum wait | 16, 24, 48 iterations |

Record, in addition to end-to-end performance:

- Encoder-pass count and batch-size distribution.
- Decoder-pass count.
- Generation batch-size distribution and average occupancy.
- Total encoder, decoder, and sampler GPU busy time.
- Delay from request submission to encoder launch.
- Delay from encoder completion to mixed decoder-context launch.

### Measured result: rejected before small-encoder optimization

The policy was implemented with the existing pending-future encoder path and
screened at concurrency 32. The proposed default released eight waiting
encoder requests or released earlier when total decoder occupancy reached 24.
Two less aggressive settings were also tested.

All runs used the same generic encoder-decoder input path because the installed
native extension predates `prepare_encoder_decoder_inputs`. No Nsight, debug
NVTX, or host timers were enabled.

| Requests | Policy | Mean latency | P50 latency | P90 latency | P99 latency | Requests/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 256 | Existing bounded accumulation | 224.531 ms | 189.549 ms | 385.358 ms | 464.158 ms | 134.155 |
| 256 | Target 8 / low watermark 24 | 320.268 ms | 273.850 ms | 608.803 ms | 689.613 ms | 94.240 |
| 256 | Target 12 / low watermark 20 | 285.727 ms | 240.157 ms | 513.444 ms | 582.509 ms | 105.972 |
| 256 | Target 16 / low watermark 16 | 268.005 ms | 237.374 ms | 459.735 ms | 549.034 ms | 113.317 |

The documented 1,024-request comparison confirmed the short screen:

| Policy | Mean latency | P50 latency | P90 latency | P99 latency | Makespan | Requests/s | Output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing bounded accumulation | 218.841 ms | 195.114 ms | 375.102 ms | 443.033 ms | 7.121912 s | 143.782 | 8568.205 |
| Target 8 / low watermark 24 | 333.685 ms | 279.505 ms | 639.218 ms | 723.822 ms | 10.822124 s | 94.621 | 5651.201 |
| Change | +52.5% | +43.3% | +70.4% | +63.4% | +52.0% | -34.2% | -34.0% |

The optimized run produced 61,158 output tokens versus 61,022 for the
baseline, a 0.22% difference that does not explain the regression. Earlier
replenishment increases eager encoder launch frequency and GPU disruption more
than fuller decoder batches save. The implementation was removed.

Occupancy-aware replenishment should be reconsidered only after design 2 makes
small encoder batches materially cheaper. Its experiment then becomes a useful
way to trade the new microbatch cost against decoder occupancy.

### Cost-aware follow-up

If a fixed low-watermark policy is directionally positive, replace the static
threshold with a measured cost decision. Maintain an exponentially weighted
estimate of encoder cost by microbatch size and decoder-step cost by graph
bucket. Dispatch a waiting encoder microbatch when its estimated cost is lower
than the decoder work expected to be saved by replenishing the open lanes.

The initial policy should remain simple until the profile counters prove that
decoder occupancy predicts end-to-end performance.

## Design 2: make small encoder batches cheap

Occupancy-aware admission can succeed only if the additional encoder launches
are inexpensive enough. The desired behavior is not necessarily to reproduce
all 180 legacy encoder passes, but to permit more frequent batches of roughly
1--8 requests without returning to the previously measured batch-one eager
encoder collapse.

### Whole-encoder CUDA graphs for microbatches

Prioritize whole-encoder graph capture for batch sizes 1, 2, 4, and 8. Use
exact-shape graph keys initially:

```text
(batch size, packed token count, maximum sequence length, exact cu_seqlens layout)
```

Exact keys avoid the dummy sequence padding and numerical changes observed in
the piecewise graph experiment. Capture graphs lazily and use an LRU limit to
bound static-buffer and graph-executable memory.

The previous bounded whole-encoder graph prototype improved mean latency by
3.8%, 3.8%, and 6.4% at concurrencies 32, 64, and 128. Its relative benefit
may be larger for the small encoder batches most affected by eager Python and
CUDA launch overhead.

### Supporting work

- Pack encoder token and position inputs in native code.
- Reuse stable pinned and device input/output storage for each graph key.
- Preserve the fused residual/layer-normalization and GELU implementations.
- Keep variable-length attention numerically identical; do not pad real
  sequences merely to reduce the number of graph keys.
- Benchmark graph lookup, input staging, replay, and output publication
  separately from capture.

Once small encoder execution is faster, rerun the occupancy-aware policy
matrix and progressively lower its microbatch target.

### Measured result: graph replay helps, but small-batch execution still loses

An experimental implementation added independent whole-encoder graphs without
replacing the decoder CUDA-graph configuration. It uses exact keys containing
the full sequence-length layout, lazy thread-local capture on the encoder
worker, a 64-entry LRU, and graph-resident pinned/device staging. Real
sequences and token counts are not padded. Runtime capture must be
thread-local: CUDA's default global capture mode otherwise rejects unrelated
sampler synchronization on the main executor thread.

The first attribution used the same admission schedule on both sides. At
batch one, exact graphs reduced mean latency by 8.0% relative to eager
execution, but the schedule was still far slower than the existing
large-batch policy because batch-one encoder kernels lose too much GPU
efficiency. At batch eight, where this cyclic workload repeatedly reuses two
exact eight-request layouts, graph replay improved the 256-request screen only
slightly:

| Requests | Encoder schedule | Encoder execution | Mean latency | Requests/s |
| ---: | --- | --- | ---: | ---: |
| 256 | Target 1 / no low watermark | Eager | 509.427 ms | 61.176 |
| 256 | Target 1 / no low watermark | Exact graph | 468.626 ms | 66.505 |
| 256 | Target 8 / no low watermark | Eager | 243.296 ms | 119.891 |
| 256 | Target 8 / no low watermark | Exact graph | 241.235 ms | 122.983 |

The matched concurrency-32, 1,024-request comparison was negative. Enabling
graphs alone while retaining the existing bounded-accumulation policy did not
help because steady-state encoder batches were normally larger than eight;
the remaining small tail batches paid capture cost without enough replay.
Forcing the best screened target-eight schedule increased encoder frequency
enough to outweigh its slightly cheaper launches.

| Policy | Mean latency | P50 latency | P90 latency | P99 latency | Makespan | Requests/s | Output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing bounded accumulation | 215.505 ms | 189.828 ms | 368.375 ms | 454.794 ms | 7.046579 s | 145.319 | 8679.106 |
| Exact graphs, existing admission | 219.949 ms | 192.038 ms | 382.436 ms | 459.412 ms | 7.210709 s | 142.011 | 8494.866 |
| Exact graphs, target 8 / no low watermark | 240.486 ms | 199.376 ms | 456.963 ms | 544.848 ms | 7.884365 s | 129.877 | 7758.392 |
| Target-8 change versus baseline | +11.6% | +5.0% | +24.0% | +19.8% | +11.9% | -10.6% | -10.6% |

The three runs produced 61,158, 61,254, and 61,170 output tokens,
respectively, so output-count variation does not explain the latency result.
As in the design-1 screen, these measurements used the generic
encoder-decoder input path because the loaded native extension predates
`prepare_encoder_decoder_inputs`.

Exact whole-encoder graphs therefore reduce launch overhead for a fixed small
batch, but not enough to make frequent PyTorch encoder replenishment
competitive. Do not enable this policy by default. A future attempt needs a
material reduction in small-batch kernel time or a graph strategy that
preserves larger encoder batches while removing their host launch overhead.

### Measured result: `(batch, total tokens, max bucket)` keys

The cyclic benchmark produces 59 unique `(B,T,S)` triples for
`B in {1,2,4,8}` when maximum sequence length is bucketed in 64-token
increments. The implementation accepts those triples as an allowlist, captures
each graph on its first real batch, and retains up to 64 graphs in an LRU.
Encoder graph metadata uses a separate buffer arena from the concurrently
executing decoder graphs, and shared host staging is retired before the next
replay updates it.

Individual sequence lengths are runtime graph inputs, not part of the key.
TRTLLM attention rebuilds `cu_seqlens` and padding offsets on the GPU from the
current device sequence-length buffer. `B`, `T`, and the maximum-length bucket
keep tensor extents, workspace requirements, and FMHA launch dimensions stable.

The earlier cross-layout hang came from inconsistent capture warmup metadata,
not a requirement for exact layouts. Graph metadata initialized its device
sequence lengths to ones, preparation updated only the stable host buffer, and
the H2D copy occurred for the first time inside graph capture. Warmup therefore
ran with real packed-token counts on the host but all-one sequence lengths on
the GPU. Capture now stages input IDs, position IDs, and sequence lengths to
their device buffers before warmup. The graph still captures the H2D copies for
subsequent replays. Exact-layout rejection is removed, so all layouts sharing a
`(B,T,S)` key reuse the same graph.

Matched concurrency-32, 1,024-request results:

| Backend / encoder mode | Mean latency | P50 latency | P90 latency | P99 latency | Makespan | Requests/s | Output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch eager, target 8 | 245.536 ms | 204.663 ms | 467.781 ms | 521.870 ms | 8.058071 s | 127.078 | 7573.525 |
| PyTorch `(B,T,S)` graph with exact-layout guard | 234.166 ms | 194.920 ms | 453.589 ms | 511.825 ms | 7.685696 s | 133.235 | 7947.751 |
| PyTorch `(B,T,S)` graph with cross-layout replay | 233.765 ms | 192.882 ms | 447.072 ms | 508.841 ms | 7.682285 s | 133.294 | 7945.943 |
| PyTorch exact batch-8 admission at decoder occupancy 24 | 225.955 ms | 187.235 ms | 433.918 ms | 492.370 ms | 7.373542 s | 138.875 | 8274.721 |
| PyTorch exact batch 8, serialized encoder/decoder forward | 225.123 ms | 183.728 ms | 435.884 ms | 485.110 ms | 7.342011 s | 139.471 | 8317.203 |
| PyTorch exact batch 8, overlapped encoder plus mixed decoder graphs | 159.100 ms | 130.335 ms | 306.821 ms | 341.215 ms | 5.229246 s | 195.822 | 11673.959 |
| Legacy TensorRT | 163.359 ms | 127.874 ms | 344.281 ms | 425.151 ms | 5.363881 s | 190.907 | 10783.982 |

Cross-layout replay did not materially change end-to-end performance versus
the exact-layout guard: mean latency fell by 0.2%, request throughput rose by
0.04%, and makespan fell by 0.04%, all within run-to-run variation. It removes
an invalid hidden key and increases graph coverage, but graph eligibility and
scheduling still dominate the aggregate result. The four PyTorch runs
generated 61,028, 61,084, 61,043, and 61,014 output tokens, respectively;
legacy generated 57,844, so request throughput and latency are the cleaner
cross-backend comparisons.

The corrected Nsight profile contains 121 encoder batches in the timed
generation window. Eighty-eight batches executed a CUDA graph: 11 captured a
new key and immediately launched it, while 77 were cache-hit replays.
Thirty-three batches used eager fallback.

The eager fallbacks were caused by scheduler overfill rather than missing
graphs for a supported batch size. Once at least eight encoder requests were
available, admission returned the entire scheduler result, producing batches
of 9 through 12 in steady state plus startup and tail batches of 31 and 5.
Exact batch-8 admission now returns only the first eight requests once decoder
occupancy reaches 24 or lower and leaves excess requests eligible for the next
iteration. A deadline releases a smaller supported power-of-two tail.

The clean exact-batch-8 run reduced mean latency by 3.3% and increased request
throughput by 4.2% versus cross-layout replay with uncapped admission. Its
profile contained exactly 128 batch-8 encoder steps for 1,024 requests. All
128 were cache-hit graph replays; there were no timed captures and no eager
fallbacks. Encoder dispatch occurred alongside 24 generation requests in 58
steps. Other steady-state dispatches occurred after the decoder completed
multiple requests between scheduling passes and crossed directly below 24.

Serializing encoder and decoder forward launches produced no measurable
end-to-end change. The executor launched the encoder on its dedicated stream,
blocked on its completion event, and only then entered decoder forward.
Compared with asynchronous exact-batch-8 admission, mean latency and makespan
were 0.4% lower, request throughput was 0.4% higher, P90 was 0.5% higher, and
the other percentiles moved by less than 2%. This mixed movement is within
single-run variation and indicates that encoder/decoder GPU concurrency was
not contributing material throughput in this workload.

The final design restores asynchronous encoder/decoder overlap and captures
whole-model CUDA graphs for mixed decoder batches. Its graph key contains the
padded decoder batch size, exact decoder context-query extents, and packed
encoder-hidden-state row count. Cross-attention sequence layouts remain
runtime metadata rather than graph keys. Encoder hidden states are copied into
one graph-stable buffer before replay, so a replacement encoder output never
leaves a captured pointer referring to request-owned storage.

Mixed graphs are never captured on a live request. Startup first warms all
reachable batch-8 and paired-batch-16 replenishment shapes to finish sizing
the shared attention workspace, then captures the same 61 mixed shapes on a
second pass. An unseen shape falls back to eager execution. This avoids both
repeating live KV-cache writes during capture and invalidating older graph
pointers through a late workspace resize.

Against the preceding best overlapped exact-batch-8 result, mixed decoder
graphs reduced mean latency by 29.6%, P50 by 30.4%, P90 by 29.3%, P99 by
30.7%, and makespan by 29.1%. Request throughput and output-token throughput
increased by 41.0% and 41.1%, respectively. It also slightly exceeded legacy
TensorRT in this run: mean latency was 2.6% lower, makespan was 2.5% lower,
and request throughput was 2.6% higher. Output-token throughput is not a clean
cross-backend comparison because legacy produced fewer output tokens.

The final Nsight timed window contained 126 mixed decoder steps, and all 126
launched a decoder CUDA graph. All 2,205 generation-only decoder steps and all
128 encoder steps also replayed graphs. The single context-only decoder step
remained eager, and there were no CUDA graph captures in the timed window.
Typical mixed `_forward_step` ranges fell from roughly 18--20 ms in the eager
profile to roughly 1.9--2.1 ms with replay.

The final PyTorch run produced 61,046 output tokens versus 61,014 in the
preceding best run, a 0.05% difference. BF16 continuous admission can change
close greedy decisions when faster completion changes batch composition, so
the output hash is not expected to remain fixed across scheduling changes.

Raw logs:

- `/tmp/bart-bts-graphs-eager-target8-pytorch-c32-r1024.log`
- `/tmp/bart-bts-graphs-safe-allow59-pytorch-c32-r1024.log`
- `/tmp/bart-bts-graphs-legacy-c32-r1024.log`
- `/tmp/bart-bts-graphs-cross-layout-fixed-allow59-c32-r1024.nsys-rep`
- `/tmp/bart-bts-graphs-cross-layout-fixed-allow59-c32-r1024.sqlite`
- `/tmp/bart-bts-graphs-exact8-low24-pytorch-c32-r1024.log`
- `/tmp/bart-bts-graphs-exact8-low24-c32-r1024.nsys-rep`
- `/tmp/bart-bts-graphs-exact8-low24-c32-r1024.sqlite`
- `/tmp/bart-bts-graphs-exact8-low24-serialized-pytorch-c32-r1024.log`
- `/tmp/bart-bts-graphs-exact8-low24-serialized-c32-r1024.nsys-rep`
- `/tmp/bart-mixed-decoder-graphs-final-overlap-c32-r1024.log`
- `/tmp/bart-mixed-decoder-graphs-final-overlap-c32-r1024.nsys-rep`
- `/tmp/bart-mixed-decoder-graphs-final-overlap-c32-r1024.sqlite`

## Design 3: a decoder supergraph

The current decoder CUDA graph captures model execution, but input staging,
attention-metadata updates, sampling, request updates, stream handoffs, and
completion processing remain separate. Legacy TensorRT hides most of this work
behind one engine enqueue.

For the qualified single-rank, single-beam, greedy BART path, maintain a
persistent device-side lane table containing:

- Request and sequence-slot identifiers.
- Current token and position.
- Self-KV block descriptors and lengths.
- Cross-KV descriptors and encoder-output pointers.
- Active, EOS, and length-complete state.

Capture one decoder supergraph containing:

```text
sampled-token gather
→ position and KV-length update
→ decoder model
→ greedy argmax
→ next-token scatter
→ EOS and length-completion update
```

The next decoder iteration should consume the sampled token directly from
device storage. The CPU should receive a compact completion record and update
only lane admissions, removals, and externally visible request state. Batch
membership changes should be expressed as deltas to persistent lane metadata
rather than a complete rebuild of every active request.

Use double-buffered launch packets so the batch-static portion of the next
iteration can be prepared while the current graph runs. Once sampled tokens
are available, the launch-critical path should consist of a compact device
update and one graph replay.

Qualification must initially exclude streaming, beam search, speculative
decoding, guided decoding, LoRA, cache reuse, attention data parallelism,
disaggregated transfer, and other features that require the general path.

## Design 4: two-token decoder graph unrolling

After the decoder supergraph is correct and beneficial, capture two dependent
greedy decoder steps in one graph replay:

```text
decoder step N
→ greedy token N
→ device state update
→ decoder step N+1
→ greedy token N+1
```

A device finish mask must prevent a token after EOS or the length limit from
being appended. A lane that completes after the first step may still perform
dummy model computation during the second step.

Start with two steps only. Larger unrolling would save more host launches but
would also delay new-request admission and waste more computation on completed
lanes. This experiment also requires sufficient KV capacity to be reserved
before replay.

## GPU work ordering

Do not treat encoder/decoder GPU overlap as a goal by itself. The legacy
profile serialized its encoder, decoder, and sampler kernels, and the PyTorch
pending-future design achieved only about 1% encoder/decoder kernel overlap.

For occupancy-aware microbatches, compare:

1. Encoder execution ordered after the current decoder step and before a later
   decoder step.
2. The existing independent encoder stream.

Prefer deterministic serialized ordering if concurrent streams introduce
resource contention. The encoder host worker can still prepare and submit work
asynchronously even if CUDA events serialize its GPU execution.

Keep decoder context requests in the main mixed decoder batch. Moving encoder
plus decoder-context execution to a side worker produced 20.6 ms of real GPU
overlap but regressed request throughput by 8.2% because separate eager context
launches, sampling, staging fences, and GPU contention cost more than the
hidden work saved.

## Experiment order

1. Rebuild or load the compatible native extension and reproduce the optimized
   PyTorch and legacy TensorRT baselines with the same workload.
2. Add lightweight counters for encoder-pass sizes, decoder occupancy, and
   admission delays.
3. Screen the occupancy-aware encoder policy without changing encoder kernels.
4. Implement exact-shape whole-encoder CUDA graphs for small microbatches.
5. Rerun the occupancy policy matrix and select the best cost/occupancy point.
6. Prototype the qualified decoder supergraph.
7. If a meaningful host bubble remains, test two-token graph unrolling.
8. Confirm every promising result with alternating, fully unprofiled long
   runs, followed by Nsight profiling for causal attribution.

## Validation criteria

Use the checked-in continuous-admission workload and report both request and
executed-token throughput because natural EOS can differ between batch
compositions.

A change should be retained only if:

- It improves both request throughput and output-token throughput outside
  observed run-to-run noise.
- Mean, P50, P90, and P99 latency do not show an unacceptable redistribution.
- Natural-EOS and length-stop behavior remains valid.
- Repeated runs show no request/output association errors.
- General configurations fall back without semantic changes.
- The native fast path, rather than the generic compatibility fallback, is
  active in the comparison.

## Designs not to revisit without new evidence

- Draining every decoder request before admitting another encoder wave.
- Moving decoder-context execution onto the encoder worker.
- Decoder-versus-encoder CUDA stream-priority changes.
- Additional tuning of only the existing fixed token and iteration thresholds.
- Maximizing encoder/decoder kernel concurrency as an independent objective.
- Retaining finished decoder rows as permanent graph-padding lanes.

The most useful immediate experiment is occupancy-aware replenishment. It
directly tests whether the legacy path's frequent replacement encoding and
lower decoder-pass count explain the remaining gap. If it improves decoder
occupancy but loses the gain to encoder cost, that result becomes a precise
performance requirement for the small-batch whole-encoder graph work.
