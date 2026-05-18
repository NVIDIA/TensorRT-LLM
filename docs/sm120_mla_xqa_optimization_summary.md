# SM120 MLA XQA Optimization Summary

This note summarizes the RTX PRO 6000 Blackwell Server Edition long-context decode work so far: what was slow, what we changed, why each change helped, and how to think about it in the broader LLM performance picture.

Current branch:

```text
zhangcl/TensorRT-LLM
codex/sm120-mla-separate-reduce-clean
implementation head summarized here: 3517f836b0c1bb687f97fad9f3ba24da626deed6
```

## One-Page Summary

The original TRT-LLM `fmha_v2` decode path was extremely slow on 8x RTX PRO 6000 BSE at 100K+ context because it did not expose enough parallel work to the GPU during single-token decode. The observed kernel had very low SM, DRAM, and L2 utilization, so it was not limited by raw compute or memory bandwidth. It was mostly waiting: too little parallelism, too much serial work along the long KV sequence.

The main fix was to route MLA decode to an XQA flash-decode-style kernel that splits the KV sequence across many CTAs, writes partial attention results, then combines them with an online-softmax reduce. That changed attention from a mostly serialized long-K loop into a much more parallel workload.

Measured end-to-end result on Kimi-K2.5-NVFP4 at 100K context:

| Variant | TPOT | Output Throughput | Notes |
|---|---:|---:|---|
| Original `fmha_v2` fallback | 469.19 ms/token | 2.13 tok/s | Long-context decode fallback |
| SM120 MLA XQA, 128-head padded | 58.46 ms/token | 17.10 tok/s | First correct fast path |
| SM120 MLA XQA, 64-head specialization | 55.90 ms/token | 17.89 tok/s | Validated, about 4.5 percent faster than 128-head padded |
| SM120 MLA XQA, 32-head specialization | pending | pending | Compiles locally; needs RTX6K runtime validation |

The big win is not from tiny code cleanup. It is from changing the shape of the attention problem so the GPU can keep many SMs busy while decoding one token at very long context.

## The Original Problem

The slow kernel was:

```text
fmha_v2_flash_attention_e4m3_fp32_64_64_S_q_paged_kv_576x512_output_bf16_sm120_kernel_nl_tiled
```

The important parts are:

- `S_q`: single-query decode. There is only one query token, so the usual query-position parallelism is gone.
- `paged_kv`: the kernel reads from paged KV cache.
- `576x512`: this matches the MLA layout: 512 latent KV dimensions plus 64 RoPE dimensions, with 512 output/value dimensions.
- `nl_tiled`: an internal tiled loop variant, which appears to process long-K work with limited external parallelism.

The measured symptom was striking:

| Metric | Value |
|---|---:|
| SM throughput | 0.35 percent of peak |
| DRAM throughput | 0.43 percent of peak |
| L2 throughput | 0.31 percent of peak |
| L1 throughput | 29 percent of peak |
| One long-context fmha call | about 7.6 ms |
| All layers per decode token | about 450 ms in attention alone |

This means the kernel was not saturating compute, HBM, or L2. It was latency/parallelism limited. The GPU had plenty of theoretical capability, but the kernel did not present enough independent work to use it.

## Why Decode Attention Is Hard At Long Context

A decode attention step computes:

```text
scores = Q x K_cache^T
probs  = softmax(scores)
out    = probs x V_cache
```

During prefill, there are many query tokens. That gives the GPU a large Q dimension to parallelize over.

During decode, there is usually one new query token per sequence. For batch=1 and concurrency=1, the natural Q parallelism collapses. The only large dimension left is the KV sequence length, which is about 100K tokens here.

A naive or insufficiently parallel decode kernel can end up doing:

```text
one query -> scan 100K K/V tokens -> maintain running softmax state -> write one output
```

The online softmax creates a serial-looking dependency:

```text
running_max, running_sum, running_output
```

Those values must be updated as the kernel walks over K tiles. FlashAttention handles this efficiently inside a tile stream, but if there are too few CTAs or warps, the long K loop still becomes a wall-time bottleneck.

## What MLA Changes

Kimi-K2.5 inherits DeepSeek-style MLA, or Multi-head Latent Attention.

In vanilla attention, the KV cache stores per-token K and V vectors for many heads. The cache grows linearly with sequence length and can be very large.

In MLA, the model stores a compressed latent KV representation. For this model shape:

```text
stored per token per rank ~= kv_lora 512 + qk_rope 64 = 576 FP8 elements
```

The key idea is that the model learns low-rank latent factors that can reconstruct or project the needed K/V information for attention. Instead of storing full per-head K/V cache, it stores a compact latent state plus positional RoPE information.

RoPE means Rotary Positional Embedding. It injects token position into Q/K by rotating pairs of hidden dimensions with position-dependent sine/cosine factors. In this model, the 64 RoPE dimensions are kept separately because positional information should not be compressed away in the same way as the latent content vector.

MLA reduces KV-cache traffic a lot, but only if the kernel actually uses the MLA representation directly. A generic FMHA path can leave a lot of MLA's advantage on the floor.

## The Flash-Decode/XQA Fix

The important algorithmic change is split-KV decode:

```text
KV sequence split into many subsequences
        |
        v
many CTAs compute partial attention results independently
        |
        v
separate reduce combines partials with online-softmax math
        |
        v
final attention output
```

Each partial contains:

```text
partial_output
row_max
row_sum
```

The final reduce cannot just average outputs. Softmax is global over the full KV sequence. To combine partials correctly, the reduce must rescale each partial by its row max:

```text
global_max = max(row_max_i)
global_sum = sum_i(row_sum_i * exp(row_max_i - global_max))
output = sum_i(partial_output_i * row_sum_i * exp(row_max_i - global_max)) / global_sum
```

That is the same online-softmax principle used by FlashAttention, but applied across independent KV splits.

Why this helps:

- It creates many independent CTAs from the long K dimension.
- It lets more SMs work on one user's one-token decode.
- It turns the long serial scan into parallel partial scans plus a smaller reduce.
- It makes long-context decode much more bandwidth/throughput shaped instead of latency shaped.

## Porting The Kernel To SM120

The XQA MLA kernel existed in a form closer to B200/B300 style assumptions. RTX PRO 6000 BSE is SM120/SM120a and differs from datacenter Blackwell in important ways.

Key porting and enablement work:

1. Runtime cluster launch

   The original kernel used compile-time cluster attributes that NVCC rejected for plain SM120. We changed it to use runtime `cudaLaunchKernelEx`/driver launch attributes. RTX PRO 6000 BSE reports cluster launch support at runtime, so this path is valid.

2. Barrier host-parse fixes

   NVCC's host-side parse was instantiating CGA barrier templates with `__CUDA_ARCH__` undefined. We adjusted barrier code so CGA and relaxed CTA arrives parse cleanly during host compilation.

3. SM120a build target

   Plain `sm_120` rejected features such as `ldmatrix.b8`, `m16n16`, and `setmaxnreg`. Building for `sm_120a` exposed the required features.

4. CMake architecture override

   The standalone XQA CMake hard-coded older architectures. We made the default conditional so `-DCMAKE_CUDA_ARCHITECTURES=120a-real` actually takes effect.

5. Runtime selector wiring

   The kernel could pass standalone tests but still not run in Kimi until the JIT XQA selector recognized the MLA shape and routed Kimi's attention call to the SM120 MLA XQA path.

6. Autotune blockers

   Kimi-K2.5-NVFP4 hit unrelated SM120 NVFP4 GEMM/MoE tactic issues before attention ran. We added tactic skip/fallback handling so the server could reach decode and exercise the attention kernel.

## Correctness Debugging

The first end-to-end attempts exposed several bugs. The debug flow was important because attention bugs often show up downstream as sampler NaNs, not at the actual broken kernel.

Useful debug gates added:

- `XQA_MLA_DEBUG_SYNC`
- `XQA_MLA_NAN_CHECK`
- `XQA_MLA_REDUCE_REF_CHECK`
- stage-by-stage checks: Q input, main output, partials, reduce output, final unpadded output
- host CPU double-precision reference reduce
- fast standalone regression: `RefCheck.mlaShortContextManySubSeq`

Major bug found:

```text
CGA-X scratch buffer was undersized.
```

The buffer size was effectively based on `8704` bytes where the actual required chunk was `9216` bytes. The CGA-X buffer overlapped `partialResults`, corrupting partial outputs. This produced enormous finite values around `8e37`, then later downstream NaNs.

After fixing the scratch size:

| Signal | Before | After |
|---|---:|---:|
| max_abs_partial | about 8.4e37 | O(1), for example 0.02 to 1.7 depending run |
| NaN assertions | many | 0 |
| reduce vs CPU reference | matched wrong corrupted values | matched correct values within BF16 noise |

The lesson: the reduce kernel was not the root cause. It was faithfully reducing corrupted partials from the main kernel due to workspace overlap.

## Optimization 1: Restore Fast Separate Reduce

During debugging, we temporarily used a conservative double-precision normalized reduce. It was useful for correctness, but too slow.

Once the scratch corruption was fixed, we restored the faster single-precision online reduce.

Result:

```text
fmha_v2 fallback:        469.19 ms/token
SM120 MLA XQA fast path:  58.46 ms/token
speedup:                  about 8x
```

Why it helped:

- The kernel now used MLA compressed KV directly.
- It split the long K dimension across CTAs.
- It reduced partials correctly without expensive debug math.
- It changed the dominant attention path from serial/latency-bound to much more parallel.

## Optimization 2: 128-Head Padded To 64-Head Physical Kernel

Kimi's runtime shape on TP=8 exposes 8 real query heads per local MLA group in this path. The first working kernel used a 128-head specialization, so the wrapper padded:

```text
8 real heads -> 128 kernel heads
```

That means 120 fake heads were carried through Q padding, output padding, scratch sizing, and parts of the kernel work.

We added a 64-head physical specialization:

```text
8 real heads -> 64 kernel heads
```

The first 64-head attempt failed because it accidentally made each warp own 16 head rows. The kernel's row-max/row-sum machinery assumes a warp owns 32 rows. That caused short-context Inf outputs. The fix was:

```text
64-head physical kernel:
4 math warps
32 rows per warp
2 chunks x 32 rows in partial results
```

Validated result:

| Variant | TPOT | Throughput |
|---|---:|---:|
| 128-head padded | 58.51 ms/token | 17.09 tok/s |
| 64-head physical | 55.90 ms/token | 17.89 tok/s |

Why the gain was only about 4.5 percent:

- Q/output padding was cut in half, but the full decode step is dominated by long-context KV reads and other model work.
- The full model still has MoE, dense GEMMs, layernorms, collectives, sampling, and framework overhead.
- Attention was improved dramatically already; this head-padding change optimizes a smaller part of the remaining time.

Still, this is a clean win with no correctness cost.

## Optimization 3: 64-Head Physical To 32-Head Physical Kernel

The next safe step is:

```text
8 real heads -> 32 kernel heads
```

This is not true 8-head yet, but it is the smallest physical head tile that preserves the current warp-level invariant:

```text
one warp owns 32 rows
```

Changes made:

- Allow MLA compile for `HEAD_GRP_SIZE >= 32`.
- Runtime head group `<= 32` selects a 32-head kernel.
- JIT allows 32/64/128 MLA kernels.
- Producer side uses 2 math warps for 32-head.
- Consumer side uses 4 column splits and 1 row split, so it still covers a 32-row tile.
- Output swizzle helper was generalized so small head tiles can use a subset of swizzle rows safely.
- Fast regression test now checks only 8 real heads while using the selected physical kernel.

Local status:

| Build | Status |
|---|---|
| `HEAD_GRP_SIZE=32` standalone XQA | compiles for `sm_120a`, no spills |
| `HEAD_GRP_SIZE=64` standalone XQA | still compiles |
| `HEAD_GRP_SIZE=128` standalone XQA | still compiles |
| RTX6K runtime validation | pending |

Expected log if selected in Kimi:

```text
Padding SM120 MLA XQA head group from 8 to 32 for the selected kernel
```

## Why Not Jump Directly To 8 Heads?

A true 8-head kernel sounds natural because Kimi's runtime group is 8. But the current kernel has a deep design assumption:

```text
warp-level row-stat operations are built around 32 rows
```

If we simply compile `HEAD_GRP_SIZE=8`, we either fail static assertions or risk corrupt row-max/row-sum handling. The first 64-head failure was already a warning sign: reducing below 32 rows per warp without redesign caused Inf outputs.

A true 8-head kernel needs a new mapping for the unused 24 rows of a warp tile. Possible directions:

1. Pack multiple query tokens into the 32-row warp tile.

   Useful for speculative decoding or multi-token decode, but less useful for strict one-token decode.

2. Pack multiple requests or beams into the 32-row warp tile.

   Useful at higher concurrency, but the benchmark is batch=1/concurrency=1.

3. Pack multiple KV splits/subsequences into the row dimension.

   Potentially useful for long-context decode, but this changes the meaning of row-wise softmax state and needs careful redesign.

4. Write a different scalar/vectorized path for 8 heads.

   This might reduce dead head work, but could lose tensor-core efficiency. Since this workload is heavily long-K memory/parallelism shaped, the win may not justify a very complex rewrite unless profiling shows head-padding compute is a meaningful bottleneck.

The practical path is:

```text
128 -> 64 -> 32 -> profile -> decide whether true 8 is worth a new schedule
```

## Big Picture: How This Fits LLM Performance

LLM serving performance has two very different phases.

### Prefill

Prefill processes many prompt tokens at once. It has a large Q dimension and usually gets good GPU parallelism. It is often compute-heavy and GEMM-heavy.

### Decode

Decode generates one or a few tokens at a time. For each new token, every layer must run:

```text
norms -> attention over full KV history -> MLP/MoE -> residuals
```

For long context, attention cost grows with sequence length. For batch=1/concurrency=1, decode is especially hard because there is little natural batch or query-token parallelism.

In this case, the original attention kernel dominated TPOT:

```text
about 7.6 ms per attention call x about 60/61 layers = most of the 469 ms/token fallback TPOT
```

After XQA MLA, attention is much faster, so the bottleneck starts spreading into the rest of the model:

- MoE expert GEMMs
- NVFP4 dense GEMMs
- memory movement
- launch overhead
- CUDA graph behavior
- NCCL/collectives
- sampling and framework overhead

This is why an enormous attention-kernel improvement turns into an 8x end-to-end speedup rather than an unbounded speedup.

## How To Read The Performance Wins

The progression is:

```text
469 ms/token -> 58 ms/token:
    algorithmic/dispatch win
    use MLA-aware split-KV flash decode instead of fallback fmha_v2

58 ms/token -> 55.9 ms/token:
    shape/padding win
    reduce fake head work from 128 physical heads to 64

future 55.9 ms/token -> maybe lower:
    more shape/padding wins
    reduce fake head work from 64 physical heads to 32, then maybe true 8
```

The first win is large because it changes parallelism. The later wins are smaller because they optimize overhead within an already much better path.

## Current Test Strategy

Fast gate before full Kimi:

```bash
./build-xqa-mla-sm120-hgrp32-exp/unitTests \
  --gtest_filter=RefCheck.mlaShortContextManySubSeq \
  --gtest_brief=1
```

Broader standalone checks:

```bash
./build-xqa-mla-sm120-hgrp32-exp/unitTests \
  --gtest_filter=RefCheck.mla:RefCheck.mlaSeparateReduce \
  --gtest_brief=1
```

End-to-end Kimi debug check:

```bash
XQA_MLA_SEPARATE_REDUCE=1
XQA_MLA_NAN_CHECK=1
XQA_MLA_REDUCE_REF_CHECK=1
XQA_MLA_DEBUG_SYNC=1
TLLM_LOG_LEVEL=TRACE
TRTLLM_MOE_SKIP_TACTICS=6,7
```

Things to confirm:

- runtime selects 32-head physical kernel
- short-context `seq_len0=1` is finite
- long-context `seq_len0=103999` passes CPU reference reduce
- no NaN checks fail
- clean no-debug TPOT improves or at least does not regress versus 55.90 ms/token

## Mental Model To Keep

The core optimization story is:

```text
Use the model's compressed MLA cache format.
Split the long KV sequence across CTAs.
Combine partials with correct online-softmax math.
Then reduce fake padded head work only after the main algorithm is correct.
```

In LLM performance terms:

- The first job is to expose parallelism.
- The second job is to reduce memory traffic.
- The third job is to remove dead work.
- Correctness instrumentation is not optional, because attention corruption often appears several layers later as NaN logits or sampler failures.

The work so far moved RTX PRO 6000 BSE long-context Kimi decode from a fallback path that barely used the GPU to a model-aware attention path that is over 8x faster end to end, with smaller head-specialization improvements now being layered on top.
