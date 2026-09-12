---
receipts:
  sm_100: {status: passed, trtllm: 1.3.0rc21}
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 18}
---

# load_paged_kv_cache_for_mla

**Wraps** `torch.ops.trtllm.load_paged_kv_cache_for_mla` (one call).

## Semantics

The cache-read step of MLA context-phase attention with KV reuse: after a
context batch's full latent KV (cached prefix + this step's new tokens) is
resident in the paged MLA cache, one call gathers it back into dense
tensors so the caller can up-project and run context FMHA over the full
`[past + new]` range.

For each context sequence `s` in `[0, num_contexts)`, with total latent-KV
length `L_s = cu_ctx_kv_lens[s+1] - cu_ctx_kv_lens[s]`, the call reads the
sequence's paged latent rows at positions `[0, L_s)` — each row is the
per-token `[compressed_kv | k_pe]` of width `C + R`
(`C = kv_lora_rank`, `R = qk_rope_head_dim`) — and writes them,
sequences concatenated in batch order, into two freshly allocated
contiguous tensors:

```
T = num_ctx_kv_tokens = sum(L_s)
compressed_kv[t, :] = cache_row(t)[:C]    # [T, C]
k_pe[t, :]          = cache_row(t)[C:]    # [T, R]
```

The paged pool is never modified, and sequences at batch index
`>= num_contexts` (generation sequences) contribute nothing to the outputs.
Two pool configurations are certified, selected by `quant_mode`:

**High-precision pool (`quant_mode = 0`).** The cache element type is
`out_dtype` and the gather is a bitwise copy.

**fp8-e4m3 pool (`quant_mode` with the fp8-KV bit, `128`).** The cache
holds one `__nv_fp8_e4m3` byte per element and the gather dequantizes.
With `r = kv_scale_quant_orig[0]` (`1.0` when that tensor is omitted):

```
compressed_kv[t, :] = out_dtype( float(cache_row(t)[:C]) * r )
k_pe[t, :]          = out_dtype( float(cache_row(t)[C:]) * r )
```

One fp32 multiply per element, by the exact fp32 number in
`kv_scale_quant_orig[0]`, rounded once to `out_dtype` — not a multiply in
`out_dtype`, which differs on 24.7% of the elements when `r` is not exactly
representable there (measured, `r = 1/1.5`, bf16). The byte→float step is
exact for all 256 e4m3 byte values, so at `r = 1` the gather is a lossless
widening. There is no per-block scale anywhere: the fp8 latent pool stores
nothing but e4m3 elements, and `r` is a single static number the caller
supplies.

Fusion boundary: only this cache read happens inside the call. The caller
still owns writing the latent rows in the first place (the cached prefix
from earlier steps plus this step's new tokens — a sibling
context-preprocessing op appends the latter, quantizing them on the way in
under its own **write-side** scale), the `kv_b_proj` up-projection of
`compressed_kv`, and the subsequent context attention. **This call applies
`kv_scale_quant_orig` unconditionally and checks nothing about how the pool
came to hold what it holds** — see Preconditions.

## Signature

```python
def load_paged_kv_cache_for_mla(
    out_dtype: torch.dtype,
    num_contexts: int,
    num_ctx_kv_tokens: int,
    max_ctx_kv_len: int,
    cu_ctx_kv_lens: torch.Tensor,
    kv_cache_block_offsets: torch.Tensor,
    host_kv_cache_pool_pointers: torch.Tensor,
    host_kv_cache_pool_mapping: torch.Tensor,
    kv_scale_quant_orig: Optional[torch.Tensor],
    layer_idx: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    tokens_per_block: int,
    attention_window_size: int,
    beam_width: int,
    quant_mode: int,
) -> Tuple[torch.Tensor, torch.Tensor]
```

With `S` = total sequences in the batch (contexts first), `C` =
`kv_lora_rank`, `R` = `qk_rope_head_dim`, `T` = `num_ctx_kv_tokens`:

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `out_dtype` | — | the **output** element type, not the cache's: `torch.bfloat16` / `torch.float16` / `torch.float32`. On the fp8 path all three are certified over the same e4m3 pool; on the high-precision path it also fixes the cache's element width (see Preconditions) and bf16 / fp16 are certified there | — | — |
| `num_contexts` | leading context-phase sequence count | Python int | — | — |
| `num_ctx_kv_tokens` | `T = sum(L_s)` over context seqs, cached **plus** new tokens (sizes the outputs) | Python int | — | — |
| `max_ctx_kv_len` | `max(L_s)` over context seqs | Python int | — | — |
| `cu_ctx_kv_lens` | `[>= num_contexts+1]`; `[0, cumsum(L_s)]`; only the first `num_contexts+1` entries are read | int64 | contiguous | CUDA |
| `kv_cache_block_offsets` | `[1, >= S, 2, max_blocks_per_seq]`; raw block ids per seq (K and V rows identical, kv_factor=1 pool); contexts first | int32 | contiguous | CUDA |
| `host_kv_cache_pool_pointers` | `[num_pools, 2]`: (primary ptr, secondary ptr=0) | int64 | contiguous | CPU |
| `host_kv_cache_pool_mapping` | `[num_layers, 2]`: (pool index, layer-within-pool) row per layer | int32 | contiguous | CPU |
| `kv_scale_quant_orig` | the **read-side** dequantization scale; element `[0]` is the only one read. `None` behaves exactly as `1.0` and is what the engine's own call site passes. Read only when `quant_mode` has the fp8-KV bit — silently ignored otherwise | fp32 (others raise) | any 1-element-or-larger shape; a 0-dim scalar also works | CUDA (a CPU tensor is also accepted — see Notes) |
| `layer_idx` | row into `host_kv_cache_pool_mapping` | Python int | — | — |
| `kv_lora_rank` | `C` (512 certified) | Python int | — | — |
| `qk_rope_head_dim` | `R` (64 certified) | Python int | — | — |
| `tokens_per_block` | pool page size; 32 and 64 certified on both paths (32 is what a default `KvCacheConfig` produces) | Python int | — | — |
| `attention_window_size` | `>= max(L_s)`; production passes the manager's `max_seq_len` (smaller values imply cyclic-cache addressing, not certified) | Python int | — | — |
| `beam_width` | `1` | Python int | — | — |
| `quant_mode` | `0` = high-precision (bf16/fp16) KV cache; `128` (`QuantMode.FP8_KV_CACHE`) = e4m3 KV cache. Both certified. Other quantization bits set **alongside** `128` are not read — `1152` (`FP8_KV_CACHE \| FP8_1x128_128x128`, the combination trtllm's auto-deploy MLA path builds for an fp8 latent cache) and `384` (`\| FP8_QDQ`) return bit-identical results. `64` and `8192` are rejected (see Notes) | Python int | — | — |
| returns `compressed_kv` | `[T, C]`, freshly allocated | `out_dtype` | contiguous | CUDA |
| returns `k_pe` | `[T, R]`, freshly allocated | `out_dtype` | contiguous | CUDA |

## Metadata consumed

No thread-local or registered-layer state: all inputs are explicit
arguments. The addressing tensors are exactly what a
`TrtllmAttentionMetadata` prepared with
`enable_context_mla_with_cached_kv=True` and its `KVCacheManager` expose:
`ctx_kv_indptr` → `cu_ctx_kv_lens`,
`num_ctx_cached_tokens + num_ctx_tokens` → `num_ctx_kv_tokens`,
`max_ctx_kv_len` → `max_ctx_kv_len`, `metadata.kv_cache_block_offsets`,
`manager.kv_cache_pool_pointers`, `manager.kv_cache_pool_mapping`,
`manager.tokens_per_block`, `manager.max_seq_len` →
`attention_window_size`. Nothing about `kv_scale_quant_orig` comes from
metadata; it is the caller's number.

## Preconditions

- The paged pool addressed by `host_kv_cache_pool_pointers` +
  `host_kv_cache_pool_mapping[layer_idx]` is an MLA latent cache:
  kv_factor 1, one kv head, row width `C + R`, page size
  `tokens_per_block`. Per page the layout is `[tokens_per_block, C + R]`
  (page base = pool base + `block_id * tokens_per_block * (C + R)`
  elements).
- **`quant_mode` alone fixes the pool's element width**, because the pool
  arrives as a raw pointer and nothing else describes it: one byte per
  element when the fp8-KV bit (`128`) is set, `sizeof(out_dtype)` when it
  is not. So the caller owes a matching pool — e4m3 under `quant_mode=128`,
  `out_dtype` under `quant_mode=0`. A mismatch in either direction is
  **silent** (see Notes).
- `out_dtype` is one of fp16 / fp32 / bf16; anything else raises
  `RuntimeError: out_dtype only support float16, float32, bfloat16` (tested
  with e4m3, int8 and fp64). The output tensors are allocated before that
  check, but no kernel runs. Note that under `quant_mode=128` `out_dtype`
  is necessarily **not** the cache dtype: the cache is e4m3 and e4m3 is not
  an accepted `out_dtype`.
- Every position `[0, L_s)` of every context sequence has already been
  written to the pool (cached prefix and this step's new tokens alike).
  The op reads raw pool memory, so slots never written simply propagate
  whatever they currently hold.
- `kv_cache_block_offsets` covers `ceil(L_s / tokens_per_block)` pages for
  every context sequence.
- `cu_ctx_kv_lens` is int64 (the kernel rejects other index dtypes), on
  device, starts at 0, and is consistent with `num_ctx_kv_tokens`
  (`cu_ctx_kv_lens[num_contexts] == num_ctx_kv_tokens`) and
  `max_ctx_kv_len` (`>=` every `L_s`).
- `num_contexts > 0`, `num_ctx_kv_tokens > 0` and `max_ctx_kv_len > 0`: all
  three are checked host-side and raise `RuntimeError` before any device
  work. Production cannot reach a violation — the MLA context path runs only
  under `num_contexts > 0`, and the cached-KV branch additionally requires
  `num_ctx_cached_tokens > 0` — but a caller driving the op directly owns
  them.
- `L_s <= attention_window_size` for every context sequence.
- `beam_width == 1`.

**fp8 pool only (`quant_mode` fp8-KV bit set):**

- `kv_scale_quant_orig` is an fp32 tensor whose element `[0]` is the
  dequantization scale, or `None` for `1.0`. A `float16` or `float64`
  tensor raises `RuntimeError: expected scalar type Float but found
  Half`/`Double`; nothing else about it is validated (see Notes).
- **`kv_scale_quant_orig` must be the reciprocal of the scale the pool was
  written at, and nothing anywhere checks that.** The write-side scale is
  an argument of a different op (`kv_scale_orig_quant` on the sibling that
  appends the new context rows) and this call never sees it. A pool written
  at `w` and read at `r` yields exactly `float(e4m3(row * w)) * r`, so any
  `r != 1/w` is a silently mis-scaled prefill with no error at any layer —
  measured at `(w, r)` = (2, 2), (4, 1), (1, 3), each landing exactly
  `w * r` away from the original rows. Production for
  DeepSeek-R1-0528-FP4 is `w = r = 1.0`: all 124 per-layer `k_scale` /
  `v_scale` tensors in that checkpoint are 0-dim fp32 scalars equal to 1.0
  (read from its safetensors shards), and the engine's own call sites
  hard-code `None` on both the write and the read side.
- `out_dtype` must have the range to hold `max|e4m3 element| * r`. The
  product is **not** clamped to `out_dtype`: with an e4m3 `±448` in the
  pool and `r = 256`, fp16 output is `±inf` while bf16 and fp32 hold
  `±114688` (measured). A NaN byte in the pool (`0x7f` / `0xff`) becomes a
  NaN output.
- The pool must actually hold e4m3 bytes at `C + R` bytes per row. The op
  reads them as raw bytes; no header, no per-block scale, no metadata.

A caller violating none of these gets a correct result.

## Notes

- Schema argument names disagree with certified usage: the schema calls the
  third and fourth arguments `num_ctx_cached_tokens` and
  `max_ctx_cached_kv_len`, but the production call path passes cached
  **plus** new token totals (`num_ctx_cached_tokens + num_ctx_tokens`,
  `max_ctx_kv_len`). The totals are what size the outputs and bound the
  read range; the wrapper parameter names reflect that.
- Precision, high-precision pool (sm_100): bf16→bf16 and fp16→fp16 gathers
  are bitwise (`rtol=0, atol=0` holds), including rows crossing page
  boundaries and pool-mapping rows other than 0 (`layer_idx=1` of a
  two-layer pool certified).
- Precision, fp8 pool (sm_100): the gather is **bit-exact** against
  `(float(e4m3 byte) * r)` rounded once to `out_dtype`, over
  `r ∈ {omitted, 1.0, 1/1.5, 0.5, 0.25, 2.0}` × `out_dtype ∈ {bf16, fp16,
  fp32}`, and separately over pool contents written at `w ∈ {0.25, 0.5,
  1.0, 1.5, 2.0, 4.0}`. Both output halves are covered by the same gate, so
  the `C`/`R` split of the 1-byte row is pinned at byte `C`.
- fp8 domain (sm_100): all 256 e4m3 byte values — `±448`, `±0`, the
  subnormals down to `2^-9`, and both NaN bytes — come back **exactly** at
  `r = 1.0` in fp16, bf16 and fp32. The byte→float step neither saturates
  nor flushes to zero; it is a plain conversion, and only the multiply that
  follows can leave `out_dtype`'s range (see Preconditions). The *write*
  side of the round trip behaves differently — the sibling append op's own
  contract records that it clamps to `±448` where
  `torch.Tensor.to(float8_e4m3fn)` yields the NaN byte — so a torch mirror
  of the whole round trip is only valid while every `|value * w| <= 448`.
  Every case certified here stays inside that.
- Page size (sm_100): certified at `tokens_per_block` 32 and 64 on both
  paths. 32 is what a `KvCacheConfig` at its defaults hands the op; 64 is
  what a tuned MLA target opts into. Both were run through the same bitwise
  gate on real `KVCacheManager` state, over gathered ranges that end on a
  page boundary and mid-page alike — high-precision pool, at 32: 37 / 96
  (three exact pages) / 400 (twelve pages plus sixteen rows) / 512 (sixteen
  exact pages) in bf16 through pool-mapping row 1 of a two-layer pool, a
  mixed batch with trailing generation sequences (128 = four exact pages,
  and 63), and 64 / 608 / 7 in fp16; the deepest sequence walked nineteen
  offsets-row entries. fp8 pool, at 32: gathered ranges of 37 / 96 / 400 /
  512 (1045 rows in one call) plus a mixed batch (128 and 63); at 64:
  1024 / 1024 / 135 (2183 rows in one call). The gather stayed bitwise in
  every case, so
  `page = t // tokens_per_block`, `slot = t % tokens_per_block` holds at
  both sizes and both element widths. Other page sizes are untested rather
  than known-bad.
- The outputs are new allocations sized `[T, C]`/`[T, R]`; a `T == 0` batch
  is **rejected** by the op rather than merely unexercised (see
  Preconditions), and production never reaches it anyway — the path is
  guarded behind `num_ctx_cached_tokens > 0`.
- The pool is not written on either path: a whole-pool byte snapshot around
  a 1045-row fp8 gather shows zero changed bytes, and a sentinel row placed
  one slot past each context sequence's range never appears in the output.
  Two identical calls return bitwise-identical tensors.
- A cache element-type or element-width mismatch (sm_100) is **silent in
  every form measured**. The two mismatches an fp8 pool makes reachable —
  reading it two bytes wide, and reading a two-byte pool one byte wide —
  are gated in this entry's test; the other observations below are from
  scratch probes.
  - *Same width, wrong type* (bf16 pool read as fp16, or fp16 pool read as
    bf16, `quant_mode=0` both times): addressing is unchanged and the cache
    bits are copied out verbatim, so the output is bit-identical to the
    correct gather and numerically wrong — max abs diff 2.0 one way and 512
    the other, nothing raised, nothing out of bounds.
  - *Read wider than the pool* (fp32 `out_dtype` over a 2-byte pool, or
    `quant_mode=0` over an e4m3 pool): the stride doubles, so token `t` of
    block `b` is read from pool row
    `2 * (b * tokens_per_block + t % tokens_per_block)` and consumes two
    rows. Verified bit for bit both ways — 565 gathered rows in the
    fp32-over-bf16 case, of which 310 came back holding a sentinel written
    nowhere near that sequence, and the exact doubled row indices in the
    e4m3 case (block 0 token 1 → row 2, block 1 token 0 → row 64). Past
    half the pool it leaves the allocation entirely: at block 1584 of a
    2048-block pool compute-sanitizer reported invalid 16-byte `__global__`
    reads 77.9 MB beyond the nearest allocation, while the call still
    returned without raising.
  - *Read narrower than the pool* (`quant_mode=128` over a bf16 pool): the
    1-byte slab for block `b` starts at byte
    `b * tokens_per_block * (C + R)`, i.e. inside bf16 block `b // 2`, and
    each token consumes `C + R` bytes. Verified bit for bit at blocks 0 and
    1. This direction stays inside the allocation — the read walks half the
    bytes the pool holds — but every value is garbage, including NaNs
    wherever a bf16 byte happens to be `0x7f`/`0xff` (107 of 36864 elements
    in one scratch probe).
- `kv_scale_quant_orig` is read **only** when `quant_mode` has the fp8-KV
  bit. With `quant_mode = 0` a scale of 4.0 produces byte-identical output
  to `None`: a silently ignored argument, and the shape a caller who set up
  the scale but forgot the `quant_mode` bit lands in.
- `kv_scale_quant_orig` validation is otherwise thin. Only the dtype is
  checked, and loudly. A `[2]` fp32 CUDA tensor is accepted and only `[0]`
  is used, and a 0-dim fp32 scalar tensor is accepted — both tested. A
  **CPU** fp32 tensor is accepted too: the pointer is handed to the kernel
  unchecked, and in a scratch probe on this machine it produced results
  bit-identical to the equivalent CUDA tensor at two distinct values. That
  works because this device reports
  `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 1` (driver 595.58.03), so
  it is a property of the machine rather than of the op — not something to
  rely on, and left out of the test for that reason. Device placement is
  not validated either way, so a wrong-device scale will not announce
  itself.
- Rejected `quant_mode` values, both tested:
  `64` (`QuantMode.INT8_KV_CACHE`) raises
  `RuntimeError: [TensorRT-LLM][ERROR] Assertion failed: Only FP8 KV cache
  is supported for now (../tensorrt_llm/thop/mlaPreprocessOp.cpp:134)`;
  `8192` (`NVFP4_KV_CACHE`) fails earlier and for a different reason —
  `Expected hostKvCachePoolPointers.dim() == 3`, the nvfp4 pool-pointer
  layout, checked before the quant_mode branch — so it is not the same
  guard. What the guard rejects is a *KV-cache* quantization bit other than
  fp8; bits outside that group ride along unread, which is why `1152`
  (`FP8_KV_CACHE | FP8_1x128_128x128`, the combination trtllm's
  auto-deploy MLA path builds for an fp8 latent cache) and `384`
  (`| FP8_QDQ`) are bit-identical to plain `128` (tested).
- The engine's own MLA backend passes `None` for `kv_scale_quant_orig` at
  this call site while still passing its real `quant_mode`, so an fp8 KV
  cache is dequantized at 1.0 there regardless of any calibrated per-layer
  scale. Harmless for a checkpoint whose scales are 1.0; the `r != 1`
  behaviour certified here is reachable only from a direct op caller.
- Sibling ops exist for adjacent MLA context-phase roles:
  `trtllm::mla_rope_append_paged_kv_assign_q` (RoPE + append of the new
  context tokens that this op then reads back, and the owner of the
  write-side `kv_scale_orig_quant`),
  `trtllm::load_chunked_kv_cache_for_mla` (slice-wise variant for chunked
  prefill, with its own `kv_scale_quant_orig`),
  `trtllm::merge_chunked_attention_for_mla`, and the context FMHA that
  consumes the up-projected K/V.
