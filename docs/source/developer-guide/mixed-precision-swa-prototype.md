# Mixed-precision KV with native SWA storage

This prototype adds an original-precision pool alongside full KV history.
The extra pool uses existing SWA storage: sink tokens keep First N, and the
window keeps Last N. These settings control which KV values are retained,
not which tokens the model attends to.

## What stays unchanged

- Both pools are `REQUIRED`. Native prefix matching checks full history and
  the high-precision sink/window together and returns the usable prefix.
- Missing high-precision pages shorten the match through the existing SWA
  rules. There is no separate tail reconstruction or low-precision fallback.
- `block_reuse_config.policy`, pool ratios, offload, eviction and window
  release use their existing implementations. High pages are not pinned to GPU.
- When cold-page compression is enabled, history uses the selected codec;
  the extra pool uses the native byte-for-byte copy path. The quantizer,
  including its RoPE option, needs no changes.
- No C++ or native binding changes are required.

## Writing and reading

The Python `MixedPrecisionKVCacheManager` subclass creates the extra pool.
It accepts `first_n`, `last_n` and `high_dtype` as internal constructor
arguments; this draft does not add public serving options. First/Last counts
must be block aligned. SWA's window includes the next write position, so
Last N previously written tokens uses a window of `N + 1`.

Before Attention, the manager-side view converts history to the computation
dtype and selects the existing high-precision pages wherever SWA has retained
them. It exposes one ordinary paged cache to the backend. Attention remains
full-context; the extra SWA pool does not add an SWA Attention mask.

The backend writes new post-RoPE KV into that view using its existing kernels.
Only new tokens are then written to both pools, directly from the same values.
High-precision values are not obtained by dequantizing the low-precision copy.
Previously shared KV is not overwritten by this writeback.

One layer-sized workspace is reused between layers. Its maximum capacity and
page-conversion temporaries are deducted from the GPU pool budget. This adds
data movement and temporary memory; it is a functional prototype, not a
performance claim.

## Scope and validation

The adapters use real TRTLLM and FlashInfer Attention, not a VANILLA fallback.
The current reader supports eager dense K/V with FP8, BF16 or FP16 hot storage.
NVFP4 cold pages are restored by the native codec before the reader runs.
Packed active-NVFP4, MLA/sparse cache layouts, CUDA graphs, speculative decoding,
context parallelism and disaggregated serving need separate integration; this
draft does not claim those combinations.

Focused local GPU tests cover both backend families, two simultaneous requests,
prefill and decode writes, original-precision values, bounded workspace reuse,
and native common-prefix matching with `all_reusable` and `per_request`.
Test definitions are not a claim that a run passed; validation logs are reported
with the branch delivery.
