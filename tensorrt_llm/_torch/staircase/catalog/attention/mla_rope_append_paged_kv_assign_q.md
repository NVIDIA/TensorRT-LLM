---
receipts:
  sm_100: {status: passed, trtllm: 1.3.0rc21}
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 13}
---

# mla_rope_append_paged_kv_assign_q

**Wraps** `torch.ops.trtllm.mla_rope_append_paged_kv_assign_q` (one call).

## Semantics

The context-phase (prefill) preprocessing step of MLA attention when a
cached KV prefix exists (KV reuse or chunked prefill), over one prepared
batch. Two configurations are certified, `beam_width=1` and DeepSeek MLA
geometry (`lora_size=512`, `rope_size=64`, `nope_size=128`) in both:

- **matching-dtype latent pool** (`quant_mode=0`,
  `kv_scale_orig_quant=None`): bf16 or fp16 activations with a paged latent
  KV cache of the same dtype. The appended row is a plain dtype-preserving
  write.
- **fp8-e4m3 latent pool** (`quant_mode=128`): bf16 activations with a
  one-byte-per-element e4m3 pool. Only the **appended pool row** is
  quantized — `q` and `latent_cache` are still rotated in place as bf16.

The rest of this section describes the matching-dtype pool; the fp8
subsection below states exactly which of its three effects changes (only
the third) and which do not.

The batch has `num_contexts` context-phase sequences first; trailing
generation sequences are ignored entirely. Rows of `q` and `latent_cache`
are the context sequences' **new (uncached) tokens**, concatenated in batch
order. For context sequence `s`, with

```
cached_s = cu_ctx_cached_kv_lens[s+1] - cu_ctx_cached_kv_lens[s]   # prefix already in cache
kv_s     = cu_seq_lens[s+1]           - cu_seq_lens[s]             # cached + new
new_s    = kv_s - cached_s                                         # rows this op consumes
```

one call computes, for each new token `i in [0, new_s)` (tensor row `t`,
absolute position `p = cached_s + i`), with `R = rope_size`,
`C = lora_size`, `N = nope_size`, `H = head_num`:

```
# GPT-J (interleaved-pair) RoPE, fp32 math, one rounding to the I/O dtype.
# (cos_d, sin_d) = cos_sin_cache row p, pair d, d in [0, R/2)
rope(x)[2d]   = x[2d] * cos_d - x[2d+1] * sin_d
rope(x)[2d+1] = x[2d] * sin_d + x[2d+1] * cos_d

# 1. q RoPE in place (per head h): the rope tail of each head is rotated,
#    reading its own pre-rotation contents ("assign q").
q[t, h*(N+R)+N : (h+1)*(N+R)] = rope(q[t, h*(N+R)+N : (h+1)*(N+R)])
# 2. k_pe RoPE in place on the latent row:
latent_cache[t, C:] = rope(latent_cache[t, C:])
# 3. latent-cache append, at page/slot of position p in seq s's blocks:
cache_row(s, p) = concat(latent_cache[t, :C],   # bitwise copy
                         rope(k_pe))            # same rotated values as 2.
```

Fusion boundary: q RoPE, k_pe RoPE, and the paged-cache append happen
inside the call. The caller still owns the projections that produced `q`
and `latent_cache`, reading the full `[cached + new]` latent KV back out of
the pool, the `kv_b_proj` up-projection, and the context FMHA itself.
`q`'s nope regions, `latent_cache`'s compressed region, and every
already-cached pool row (positions `< cached_s`) are bitwise untouched.

### fp8-e4m3 latent pool (`quant_mode=128`)

`quant_mode=128` is `QuantMode`'s fp8-KV-cache bit. It switches the
**latent pool's** element type to fp8-e4m3, one byte per element, while
`q` and `latent_cache` stay bf16. Certified at the DeepSeek-R1-0528 cell —
`head_num = 128`, `tokens_per_block = 32`, `lora/nope/rope = 512/128/64`,
`beam_width = 1`, single-layer pool — with the KV scaling factor omitted
(the production call) and explicit 1.0 / 1/1.5 / 0.5 / 0.25 / 2.0 swept
beside it, over fresh-prefill (`cached_s = 0`) and cached-prefix
(`cached_s > 0`) context sequences in the same call.

One fp32 scalar steers it:

```
w = kv_scale_orig_quant[0]   # 1.0 when that tensor is None
e4m3(x) = round-to-nearest cast of an fp32 x to float8_e4m3fn,
          saturating at +-448 (see Notes)
```

Effects 1 and 2 above are **unchanged** — `q`'s rope tails and
`latent_cache`'s k_pe are rotated in place and written back as bf16,
un-quantized. Effect 3 becomes, with `bf16(.)` the rounding to the
activation dtype that the in-place write already performs:

```
# 3. latent-cache append, quantized, at page/slot of position p in seq s:
cache_row(s, p) = e4m3(concat(latent_cache[t, :C],
                              bf16(rope(k_pe))) * w)
```

`w` is a **per-tensor static scale**, not a block or dynamic one: every
appended row is the same multiply-and-round, bit-exact against that mirror
at each scale tested. The multiply happens in fp32 *after* the rounding to
bf16, so `e4m3(bf16(x) * w)` and `e4m3(bf16(x * w))` are different tensors
for a non-power-of-two `w` and the kernel computes the first (measured, see
Notes).

**`q` is not quantized here, and neither is `latent_cache`.** Both come
back as ordinary bf16 rotations, matching the same reference the
`quant_mode=0` path matches, and neither is an e4m3 round trip (6.2% of
their elements survive an e4m3 round trip unchanged — the same fraction as
the untouched nope slice of the same tensor, and far from the 100% a
quantized tensor would show). The FMHA that consumes `q` afterwards is a
separate call — `trtllm::attention` with
`attention_input_type = context_only` and `latent_cache=None` — and on the
fp8 MLA context path that op was measured to quantize its own `q`/`k`/`v`
to e4m3 internally. That is that op's fact, established separately and on
its fresh-prefill flavor rather than the cached-KV flavor this op feeds;
what matters here is that nothing this op writes into `q` is pre-quantized,
so the pair is not a double quantization.

The `w` on the pool row is likewise **not** applied to `q`: two runs over
identical inputs at `w = 1.0` and `w = 0.25` returned bitwise-identical `q`
and `latent_cache` while the appended rows moved (tested).

`w` is a write-side factor only. Whatever later reads the pool back has to
supply its own read-side factor `1/w`; this op takes no such argument and
checks no relation. The generation-phase sibling
`trtllm::mla_rope_generation`, which appends into the same pool for decode
steps, takes both (`kv_scale_orig_quant` and `kv_scale_quant_orig`) and
uses them independently; its own append was separately measured to be the
same `e4m3(row * w)` static rule, so a pool written by both phases is
uniformly scaled as long as the caller passes the same `w` to each.

## Signature

```python
def mla_rope_append_paged_kv_assign_q(
    q: torch.Tensor,
    latent_cache: torch.Tensor,
    num_contexts: int,
    cu_ctx_cached_kv_lens: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    max_input_uncached_seq_len: int,
    cos_sin_cache: torch.Tensor,
    head_num: int,
    nope_size: int,
    rope_size: int,
    lora_size: int,
    kv_cache_block_offsets: torch.Tensor,
    host_kv_cache_pool_pointers: torch.Tensor,
    host_kv_cache_pool_mapping: torch.Tensor,
    kv_scale_orig_quant: Optional[torch.Tensor],
    layer_idx: int,
    tokens_per_block: int,
    attention_window_size: int,
    beam_width: int,
    quant_mode: int,
) -> None
```

With `nc` = `num_contexts`, `S` = total sequences in the batch (contexts
first), `T` = `sum(new_s)` over context sequences, and `C`/`R`/`N`/`H` as
above:

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `q` | `[T, H*(N+R)]`; per head only the trailing `R` written, in the activation dtype on both paths (never quantized) | bf16 / fp16 (bf16 only on the fp8 path) | 2-D enforced (other ranks rejected), contiguous | CUDA |
| `latent_cache` | `[T, C+R]` = per-token `[compressed_kv \| k_pe]`; only `[..., C:]` written, in the activation dtype on both paths | same as `q` | 2-D enforced, contiguous | CUDA |
| `num_contexts` | leading context-phase sequence count | Python int | — | — |
| `cu_ctx_cached_kv_lens` | `[>= nc+1]`; `[0, cumsum(cached_s)]`; only the first `nc+1` entries read | int64 | contiguous | CUDA |
| `cu_seq_lens` | `[>= nc+1]`; `[0, cumsum(kv_s)]`; only the first `nc+1` entries read | int64 | contiguous | CUDA |
| `max_input_uncached_seq_len` | `max(new_s)` over context seqs (grid bound; production passes the exact max) | Python int | — | — |
| `cos_sin_cache` | `[1, max_pos * R * 2]`: per position `R` fp32 `(cos, sin)` pairs, second `R/2` a duplicate of the first (duplicated/`duplicate_data=True` layout; only pairs `[0, R/2)` are read per position) | fp32 | contiguous | CUDA |
| `head_num` | query heads `H`; 16 (matching-dtype pool) and 128 (fp8 pool) certified | Python int | — | — |
| `nope_size` | `N` (128 certified) | Python int | — | — |
| `rope_size` | `R` (64 certified) | Python int | — | — |
| `lora_size` | `C` (512 certified) | Python int | — | — |
| `kv_cache_block_offsets` | `[1, >= S, 2, max_blocks_per_seq]`; raw block ids per seq (K and V rows identical, kv_factor=1 pool); contexts first | int32 | contiguous | CUDA |
| `host_kv_cache_pool_pointers` | `[num_pools, 2]`: (primary ptr, secondary ptr=0) | int64 | contiguous | CPU |
| `host_kv_cache_pool_mapping` | `[num_layers, 2]`: (pool index, layer-within-pool) row per layer | int32 | contiguous | CPU |
| `kv_scale_orig_quant` | `quant_mode=0`: `None`. `quant_mode=128`: `[1]` holding the **write-side** factor `w` — everything this call quantizes is multiplied by it. `None` = 1.0, which is what the engine's own MLA call site passes. Only element `[0]` is read (a longer tensor is accepted) | fp32 (fp16 rejected, see *Notes*) | contiguous | CUDA |
| `residual_dim` | — | `int` | **rc26 addition.** Must be `0` or `rope_size`; the op rejects non-zero unless the KV pool is FP4. `0` on every path this entry certifies (bf16 and fp8-e4m3 pools), which is also what the in-tree caller passes | — |
| `layer_idx` | row into `host_kv_cache_pool_mapping` | Python int | — | — |
| `tokens_per_block` | pool page size; 32 and 64 certified on the matching-dtype pool, 32 on the fp8 pool (32 is what a default `KvCacheConfig` produces) | Python int | — | — |
| `attention_window_size` | `>= max(kv_s)`; production passes the manager's `max_seq_len` (smaller values imply cyclic-cache addressing, not certified) | Python int | — | — |
| `beam_width` | `1` | Python int | — | — |
| `quant_mode` | `0` (matching-dtype pool) and `128` = `QuantMode.FP8_KV_CACHE` (fp8-e4m3 pool) certified. A **non-fp8 quantized** KV cache is rejected — see *Notes*. `128 \| 256` (an fp8-weights bit alongside) was accepted rather than rejected; its output was not compared | Python int | — | — |
| returns | — (mutates `q`, `latent_cache`, and the paged pool) | — | — | — |

## Metadata consumed

No thread-local or registered-layer state: all inputs are explicit
arguments. The length/addressing tensors are exactly what a
`TrtllmAttentionMetadata` prepared with
`enable_context_mla_with_cached_kv=True` and its `KVCacheManager` expose:
`ctx_cached_token_indptr` → `cu_ctx_cached_kv_lens`, `ctx_kv_indptr` →
`cu_seq_lens`, `max_ctx_seq_len` → `max_input_uncached_seq_len`,
`metadata.kv_cache_block_offsets`, `manager.kv_cache_pool_pointers`,
`manager.kv_cache_pool_mapping`, `manager.tokens_per_block`,
`manager.max_seq_len` → `attention_window_size`.

## Preconditions

- The paged pool addressed by `host_kv_cache_pool_pointers` +
  `host_kv_cache_pool_mapping[layer_idx]` is an MLA latent cache:
  kv_factor 1, one kv head, row width `C + R`, page size
  `tokens_per_block`. Per page the layout is `[tokens_per_block, C + R]`
  and the page slab is `tokens_per_block * (C + R)` **elements**, whose
  byte width the op takes from `quant_mode` and the activation dtype, not
  from anything the cache manager tells it: the activation dtype's width at
  `quant_mode=0` (2 bytes for the certified bf16/fp16), 1 byte (e4m3) at
  `quant_mode=128`. The pool's real element type must match — a
  `KVCacheManager` built with `DataType.BF16` / `DataType.HALF` and with
  `DataType.FP8` respectively is what was certified.
- Every context sequence has block capacity for `kv_s` tokens **before**
  the call (block offsets in `kv_cache_block_offsets` cover
  `ceil(kv_s / tokens_per_block)` pages).
- `q` and `latent_cache` are 2-D, contiguous, of the same dtype, with
  exactly `T` rows: the context sequences' new tokens in batch order
  (production slices `q[:num_ctx_tokens]`; generation-token rows must not
  be included).
- `q`'s rope tails and `latent_cache`'s k_pe hold **pre-rotation** values;
  the call rotates them in place. A table with cos=1/sin=0 makes the RoPE
  identity (pre-rotated inputs pass through).
- `cu_ctx_cached_kv_lens` / `cu_seq_lens` are int64 (the kernel rejects
  other index dtypes), on device, start at 0, and satisfy
  `cached_s <= kv_s` per sequence.
- `max_input_uncached_seq_len >= new_s` for every context sequence
  (certified with the exact max), and `kv_s <= attention_window_size`,
  `kv_s - 1 < max_pos` of the table.
- `cos_sin_cache` uses the duplicated `(cos, sin)`-pair layout above
  (`RopeParams(..., duplicate_data=True).create_rope_const_params()`
  produces it; MLA models set `duplicate_data=True`).
- **Matching-dtype pool only:** `quant_mode=0` and
  `kv_scale_orig_quant=None`.
- **fp8 pool only** (`quant_mode=128`):
  - `q` and `latent_cache` are bf16 (fp16 activations over an fp8 pool were
    not run).
  - `kv_scale_orig_quant` is a `[1]` fp32 CUDA tensor or `None` (= 1.0).
    `None` is the production configuration and is what the engine's own MLA
    call site passes. There is no read-side factor here: a consumer that
    dequantizes the pool needs `1/w` from somewhere else, and nothing in
    this call checks it.
  - Input magnitudes must survive e4m3: `|value * w|` below `2**-10`
    (~9.8e-4) rounds to zero, and anything above 448 saturates to `±448`
    (measured — see *Notes*). The certified runs are unit-normal
    activations at `w <= 2`.

## Notes

- Schema under-annotation: the registered schema marks **nothing** mutable —
  it declares `(Tensor q, Tensor latent_cache, ...) -> ()`, with no
  `Tensor(a!)` anywhere — yet the call rotates `q` and `latent_cache` in
  place and appends to the paged pool. Do not rely on the schema's alias
  info (e.g. under torch.compile functionalization).
- Precision of the in-place rotations (bf16 and fp16, sm_100, **both**
  paths — the fp8 path rotates `q` and `latent_cache` exactly as the
  matching-dtype path does): the `compressed_kv` copy is bitwise. Both RoPE
  outputs match an fp32 reference rounded once to the I/O dtype and are
  bitwise on all but 1.4e-5 to 1.7e-5 of elements: the kernel and a torch
  fp32 reference evaluate `x*cos ∓ y*sin` in different orders, so where the
  correctly-rounded fp32 result lands near a rounding midpoint of the I/O
  dtype the two round to adjacent values. Measured in bf16 at `H = 16` on a
  ~400-token context call: 7 such elements in 411 648 rotated `q` values at
  page size 32 and 8 at page size 64, all within one bf16 ulp (0 in 25 728
  rotated `k_pe` values either way) — it is arithmetic, not addressing. At
  `H = 128` on the same batch shape the tail reaches further: 46 elements in
  3 293 184 differ, one of them by 4 bf16 ulps, but the **absolute**
  deviation stays ≤ 0.0078 because the elements that drift are the ones
  where `x*cos - y*sin` nearly cancels. Default `assert_close` tolerances
  absorb all of it with margin, which is what the test gates on; a caller
  needing bit-exactness against a torch reference will not get it.
- Precision of the fp8 append (sm_100): **everything came back bit-exact**
  against `e4m3(row * w)`, the `compressed_kv` half and the roped `k_pe`
  half alike, at `w` ∈ {1.0 (both as `None` and as an explicit tensor),
  1/1.5, 0.5, 0.25, 2.0}. e4m3's 3-bit mantissa absorbs the evaluation-order
  difference the bf16 rotation pays an ulp for. The test still gates the
  roped half at one e4m3 ulp (`2**-3` relative) plus a bit-exact-fraction
  floor rather than bitwise, because the tie it covers is arithmetic;
  neither half has been approached.
- **The double rounding is real and a mirror must reproduce it.** The
  quantized `k_pe` is `e4m3(bf16(rope(k_pe)) * w)` — and that bf16 value is
  the one the call also writes back into `latent_cache` (both came back
  bit-exact against the same torch reference), so a caller already holding
  the post-call `latent_cache` can mirror the pool from it directly.
  Quantizing the unrounded fp32 RoPE result instead disagrees with the
  kernel on 2.4-3.3% of the appended `k_pe` bytes (measured at every scale),
  which is enough to fail a bit-exact gate and small enough to pass a sloppy
  one. Applying the scale before the bf16 rounding —
  `e4m3(bf16(rope * w))` — disagrees on 1.05% of bytes at `w = 1/1.5` and is
  identical at the power-of-two scales, so only a non-power-of-two scale
  separates the two orders.
- A `quant_mode` asking for a **non-fp8 quantized** KV cache is rejected:
  `quant_mode = 64` (`QuantMode.INT8_KV_CACHE`) raises
  `RuntimeError: [TensorRT-LLM][ERROR] Assertion failed: Only FP8 KV cache
  is supported for now (../tensorrt_llm/thop/mlaPreprocessOp.cpp:316)`.
  Tested. `quant_mode = 8192` (`NVFP4_KV_CACHE`) also fails, but earlier and
  for a different reason — `Expected hostKvCachePoolPointers.dim() == 3`,
  the nvfp4 pool-pointer layout — so it is not the same guard.
- e4m3 range behaviour at the boundaries (scratch probe, `w = 1.0`): the
  kernel **saturates**, writing `+448` for an input of 1000 or `+inf` and
  `-448` for -1000, and flushes 1e-5 to `+0`. `torch.Tensor.to(float8_e4m3fn)`
  does *not* saturate — it yields the NaN byte `0x7F` for 1000 and for
  `inf` — so a torch mirror and the kernel agree only while every
  `|value * w|` stays ≤ 448. Every certified run does.
- `kv_scale_orig_quant` validation is thin: a `[2]` fp32 CUDA tensor is
  accepted and only `[0]` is used, and a **CPU** fp32 tensor is accepted
  too — in one scratch probe on this machine it even produced the same bytes
  as the equivalent CUDA tensor, so device placement is not checked and a
  wrong-device scale will not announce itself. A `float16` tensor is
  rejected loudly (`RuntimeError: expected scalar type Float but found
  Half`).
- **Not idempotent**: the q and k_pe rotations read their own pre-call
  contents in place, so calling twice on the same buffers composes two
  rotations (unlike ops that read a separate source). Re-preparing inputs
  is required before any retry.
- `cached_s = 0` (fresh prefill, nothing cached) is valid and certified on
  both paths; positions then start at 0. `cached_s > 0` — a context call
  over a prefix already in the pool, which is what block reuse makes the
  engine do — is certified on both paths too, and the prefix rows come back
  bitwise unchanged.
- Pool writes are exactly bounded on both paths: one `C+R`-element row per
  new context token, at the `(page, slot)` its absolute position addresses,
  and nothing else in the pool moves — checked by a whole-pool byte snapshot
  on every case, including a 402-row single call spanning four sequences and
  a mixed batch whose trailing generation sequences own their own pages.
  That is also what pins the page-slab geometry described under
  *Preconditions*.
- Page size (sm_100): certified at `tokens_per_block` 32 and 64 on the
  matching-dtype pool, 32 on the fp8 pool. 32 is what
  a `KvCacheConfig` at its defaults hands the op; 64 is what a tuned MLA
  target opts into. At 32 the cached prefixes were chosen to put the first
  new token at every alignment a 32-slot page has — 0 (fresh prefill),
  32 and 96 (slot 0 of a fresh page, after one and three full pages),
  1 / 33 / 100 (mid-page), 31 (a page's last slot, so the run crosses on its
  very first token), 512 (sixteen full pages) and 511 (the last slot of page
  15, which the single new token closes) — with new-token runs of 1 to 300
  tokens, the longest walking ten pages in one call, in bf16 (including
  through pool-mapping row 1 of a two-layer pool) and in fp16, plus a mixed
  batch whose trailing generation sequences own their own pages. Every effect
  held at the same gates as at page 64: RoPE within the one-ulp band above,
  `compressed_kv` copy bitwise, cached prefix and the untouched slices
  bitwise. The fp8 cases run page 32 only, over the same alignment set —
  first new token at 0, 17, 31, 32, 96, 100 and 511, runs of 1 to 300 tokens,
  and a 402-row four-sequence call. Other page sizes are untested rather than
  known-bad.
- Both `q.dim() == 2` and `latent_cache.dim() == 2` are enforced with
  runtime errors. A comment in the runtime describes `latent_cache` as
  `[tokens, 1, C+R]`, but the op itself rejects 3-D — callers flatten to
  rows first.
- Only the DeepSeek-V3 geometry (`C=512`, `R=64`, `N=128`) was exercised;
  other sizes are unverified.
- Sibling ops exist for adjacent MLA roles:
  `trtllm::load_paged_kv_cache_for_mla` (gathers the full `[cached + new]`
  latent KV this op just completed, and takes its own read-side
  `kv_scale_quant_orig`), `trtllm::load_chunked_kv_cache_for_mla`
  and `trtllm::merge_chunked_attention_for_mla` (chunked-prefill variants),
  `trtllm::mla_rope_generation` (the generation-phase counterpart), and the
  context FMHA that consumes the up-projected K/V.
- Not certified on either path: `beam_width > 1`,
  `attention_window_size` below the KV length (cyclic-cache addressing), and
  MTP / speculative-decoding batches. Not certified on the fp8 path
  specifically:
  `tokens_per_block = 64`, head counts other than 128, fp16 activations,
  multi-layer latent pools, `|value * w| > 448`, and a YaRN-scaled rope
  table (the fp8 cases run the unscaled theta-10000 table; the table's
  *content* is not an axis this op's gate exercises).


## rc26 change to the accepted KV-cache formats

The op accepted only an fp8-e4m3 latent pool in 1.3.0rc21 and now also
accepts NVFP4; its rejection message changed accordingly from
`Only FP8 KV cache is supported for now` to `Only FP8 and NVFP4 KV
caches are supported for now`. An int8 pool (`quant_mode=64`) is still
rejected. **NVFP4 is accepted by the op but not certified here** — no
cell in this entry's test drives it, so it stays outside the envelope.
