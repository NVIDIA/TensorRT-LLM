# Selective prefill TopK

This internal DKG example follows FlashInfer's selective prefill module layout.
K1/K2/K3/K4 remain in `prefill/`; `interface.py` and `adapter/` provide the planned
API. Corresponding modules retain matching names for maintenance, with
DKG-specific imports, tests and integration.

## API

```python
import torch
from blackwell.kernel.throwaway_topk.interface import FP8MQATopKWrapper

op = FP8MQATopKWrapper(
    backend="cutedsl", num_heads=64, head_dim=128, top_k=2048,
    max_context_length=131072, max_chunk_size=8192, max_batch_size=1,
    device="cuda", qkv_dtype="fp8", logits_dtype=torch.bfloat16,
    weights_dtype=torch.float32, math_mode="relu_fast",
)
op.plan(qo_indptr, kv_indptr)
op.prepare_pool(k, k_scale=k_scale)
indices = op.run(q, weights, k, k_scale=k_scale)
```

`plan` takes CUDA INT32 prefixes in **original token units, including Q**.
For a single request, `qo_indptr=[0,Q]` and `kv_indptr=[0,kv_len]`.
Physical K has `kv_len // compression_ratio` live rows; inputs are already
compressed. The adapter computes each query's causal bounds. It never
compresses or gathers paged KV. Outputs are request-local INT32 indices with
`-1` padding; ties may select different valid indices.

Construct and prepare pool bindings outside CUDA Graph capture. Keep the
wrapper, inputs and captured graphs alive, and serialize calls on one wrapper.
Live prefixes and payloads can change within the prepared contract. Arbitrary
capacity/layout changes require a new binding or capture. Inspect `error` and
`repair_error` outside capture.

The raw `api.ThrowawayTopK` constructor is available for kernel tests;
new integration should use the planned interface above. DKG omits FlashInfer's
package-specific API logging and trace decorators. It does not require an
installed FlashInfer package to run selective prefill.

## Supported model configurations

| Model | Heads | TopK | Compression ratio |
| --- | ---: | ---: | ---: |
| DS-V3.2 | 64 | 2048 | 1 |
| Qwen3.8-Flash-Next | 4 | 512 | 4 |
| GLM-5.3-Flash | 32 | 512 | 4 |
| DS-V4-Flash | 64 | 512 | 4 |
| DS-V4-Pro | 64 | 1024 | 4 |
| GLM-5.3 | 32 | 2048 | 1 |

Policy selection uses heads, head dimension, TopK and compression ratio. An
optional `model_config` must agree with those values. The fixed-weight Qwen
contract is selected only by an explicit `model_config="Qwen3.8-Flash-Next"`.
Inputs support token-scaled FP8, MXFP8 and packed MXFP4; logits support
FP16/BF16/FP32 and weights support FP32/BF16. `relu_strict` and `relu_fast`
accumulate heads in FP32. With `math_mode=None`,
FP8/MXFP4 default to `relu_fast` and MXFP8 to `relu_strict`; MXFP8 rejects
`relu_fast`. Explicit `native_fp16`/`native_bf16` use matching head arithmetic and logits dtype;
changing storage dtype alone does not select these modes.

The head dimension is 128. Query capacity aligns to 64 rows for H4, eight for
H32 and four for H64. Block-scaled K storage aligns to 256 rows. The public
wrapper supports single-request prefill on SM100/SM103, not decode or H8.
Explicit Qwen configuration requires weights equal to `1/sqrt(128)` rounded to
the weights dtype and unit K scales for token-scaled FP8. Live numerical
validation reports error flags for invalid inputs.

## Input and arithmetic contracts

K stores the contiguous, precompressed keys for the single planned request.
`num_keys` bounds physical storage and defaults to
`floor(max_context_length / compression_ratio)`; live prefixes specify original
token counts. `max_chunk_size` bounds all supplied Q rows, including padding.

| Property | Contract |
| --- | --- |
| Token-scaled FP8 | K has shape `[N,D]`; K scales are FP32 `[N]` |
| MXFP8 | K is E4M3FN `[N,D]`; UE8M0 scales are UINT8 `[N,4]`; storage is padded to 256 rows |
| MXFP4 | K is packed UINT8 `[N,D/2]`; UE8M0 scales are UINT8 `[N,4]`; storage is padded to 256 rows |
| Weights | FP32 or BF16 matching `weights_dtype`; heads are contiguous, with padded row strides allowed |
| Policy | Inferred from shape; an explicit Qwen name enables its numerical constraints |
| Output | Shared INT32 `[Q,top_k]` storage, request-local indices, `-1` padding; tied indices need not be stable |

QK dots accumulate in FP32. `relu_strict` and `relu_fast` also use FP32 head
accumulation. Native modes round ReLU results and weights to FP16/BF16, use four
native FMA chains, and reduce as `(c0+c2)+(c1+c3)`. They require matching logits
dtype; input or logits storage dtype alone does not select native arithmetic.
K1, K2 and K4 use the same score rule.

For token-scaled FP8 with `native_fp16`, QK is divided by 512 and weights are
multiplied by 512. This input-independent rebalance bounds the E4M3FN/D128 dot
product before FP16 conversion. Other native modes use no rebalance. FP8 applies
its FP32 key scale after head reduction; MXFP4 applies block scales within the
dot product. Native rounding, underflow and overflow differ from FP32 reduction.

## Planning and graph lifetime

`plan` accepts CUDA INT32 `[2]` query/KV prefixes for one request. `next_n`
optionally asserts an exact live query count. `host_length_hints=(query_prefix,
kv_bounds)` accepts contiguous CPU INT32/INT64 tensors of shapes `[2]` and `[1]`;
these bound device lengths. Omit host hints when capturing `plan`.

Construct the wrapper and eagerly run or prepare each K/scale binding before
capture. The caller owns its CUDA graphs; the wrapper caches prepared launches,
not graphs. Calls on one wrapper must be serialized. Keep inputs and bindings
alive until their graphs are discarded; clear pools only after discarding graphs.

Retired JIT modules drain at `plan`, `prepare_pool` and `clear_pools`, except
while the calling stream is capturing. Do not call these methods while another
thread performs global capture. This prevents module unloading during capture.

Metadata and scoring run in order on one stream. Status resets at `plan`.
Read flags outside capture; validation does not synchronize to raise exceptions.

| Status | Meaning |
| --- | --- |
| `error` bit 1 | Invalid staged extents; suppresses dependent reads |
| `error` bit 2 | Invalid prefixes, query-count assertion or host-bound violation |
| `error` bit 4 | Capacity exceeded; may coexist with bit 2 |
| `repair_error` bit 16 | Fast-mode or explicit Qwen numerical contract violation |

Invalid rows produce `-1` output. Fast mode requires finite weights that are zero
or have magnitude at least `2**-125`; MXFP4 scale bytes must be at most 159.
Supplied Q storage may exceed the live prefix; inactive output rows remain `-1`.

## Layout and algorithm

| Location | Responsibility |
| --- | --- |
| `interface.py` | FlashInfer-compatible selective public wrapper |
| `adapter/` | Planning, live input validation, pool binding and metadata kernels |
| `api.py` | Raw input contract and prefill construction |
| `prefill/` | The shared four-stage algorithm and model policies |
| `packed.py`, `ordering.py`, `pdl.py` | Local arithmetic, TMA ordering and PDL helpers |

K1 samples scores to choose a filtering threshold. K2 computes scores and
retains candidates instead of materializing dense logits. K3 selects exact
TopK from those candidates and queues failed rows. K4 performs exact recovery
and reuses candidate storage. Model-dependent templates select sampling,
candidate layouts and selection paths. PDL remains enabled by default.

Imports do not replace shared CuTe pipeline or arithmetic methods. Kernel
modules use package-local helpers, matching FlashInfer. The synchronization
and numerical logic is copied with the kernels, not reimplemented in DKG.

## Maintaining the two implementations

Paths below are relative to `flashinfer/` in FI and `throwaway_topk/` in DKG.

| FI module | DKG module |
| --- | --- |
| `experimental/attn_scores/kernels/selective_topk/prefill/` | `prefill/` with matching filenames |
| `experimental/attn_scores/kernels/selective_topk/*.py` | Root operator and helper modules with matching filenames |
| `attn_scores/mqa_topk.py::FP8MQATopKWrapper` | `interface.py::FP8MQATopKWrapper` |
| `experimental/attn_scores/selective_prefill.py` | `adapter/selective_prefill.py` |
| `experimental/attn_scores/_topk_{capacity,runtime,host_hints}.py` | Corresponding files in `adapter/` |
| `experimental/attn_scores/_capture_status.py` | `adapter/_capture_status.py` |
| `experimental/attn_scores/indexer_config.py`, `_selective_math_policy.py` | Corresponding files in `adapter/` |
| `experimental/attn_scores/kernels/*_metadata.py` | `adapter/kernels/` with matching filenames |
| `experimental/attn_scores/__init__.py::_continuous_plan_type` | `adapter/__init__.py::_continuous_plan_type` |

When porting a change, compare the corresponding modules and carry over the
algorithm, launch parameters and buffer contracts together. Keep DKG imports and
internal markers. FI package logging, tracing and paged Full/GVR support are not
part of this package. Exact file equality is not a maintenance requirement.

Run the affected tests in `test_throwaway_topk.py` after a change. Kernel, launch
or buffer changes also require paired performance measurements.

## Validation and benchmarking

`test_throwaway_topk.py` under `cutlass_ir/compiler/test/python/examples/sm_100f`
exercises the planned API with Q4096/8192, total KV256K/1M, six models, FP8/MXFP4
and FP16/BF16/FP32 logits. The same file covers metadata validation, default math
modes and query-count/host-bound checks during graph replay.

For performance comparisons, use the same inputs, model policy, arithmetic mode
and CUDA Graph timing for both implementations. KV lengths **include Q**.
Flush L2 outside the timed region and balance arm order across inputs. Validate
exact score multisets before and after timing, allowing tied indices to differ.
Keep compilation and input loading outside timing. Experiment drivers, input
captures and performance reports are maintained separately from this package.
