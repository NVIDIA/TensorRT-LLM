---
receipts:
  sm_100: {status: passed, trtllm: 1.3.0rc21}
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 22}
---

# mla_rope_generation

**Wraps** `torch.ops.trtllm.mla_rope_generation` (one call).

## Semantics

The generation-phase (decode) preprocessing step of MLA absorbed attention,
over one prepared batch. Two configurations are certified, and they differ
in what the call writes — read the one you are on:

- **bf16 latent pool** (`quant_mode=0`, every fp8/scale tensor `None`): the
  roped q lands in `fused_q`'s tail and the appended cache row is bf16.
- **fp8-e4m3 latent pool** (`quant_mode=128`): `fused_q` is **not written at
  all**; the roped q lands quantized in `quant_q_buffer`, the appended cache
  row lands quantized, and the two decode-FMHA scale buffers are filled.

Both are certified with bf16 activations (`fused_q`, `q_pe`,
`latent_cache`), `rope_append=True`, `beam_width=1`,
`block_ids_per_seq=None`, `out_scale=None` and no helix parallelism, at
`predicted_tokens_per_seq` (`P` below) 1, 2, 3 and 4.

The batch has `num_contexts` context-phase sequences first, then `G`
generation-phase sequences. The op processes **only the generation
sequences**. Each of them contributes `P` new query tokens — `P = 1` is an
ordinary decode step, `P > 1` the MTP shape, where `P = max_draft_len + 1`.
`fused_q`/`q_pe`/`latent_cache`/`quant_q_buffer` therefore have `G * P`
rows, **token-major within a sequence**: row `n` is generation sequence
`g = n // P`'s `t = n % P`-th new token. Per-sequence tensors
(`sequence_length`, `host_past_key_value_lengths`, `host_context_lengths`,
`kv_cache_block_offsets`) cover the whole batch in order and stay one entry
per sequence; the op indexes them at `num_contexts + g`.

**Every row gets its own rope angle.** With
`L_g = sequence_length[num_contexts + g]` the total KV length of generation
sequence `g` *including all `P` of its new tokens*, row `(g, t)` sits at
0-based absolute position

```
pos(g, t) = L_g - P + t          # t = 0 .. P-1, consecutive, ending at L_g - 1
```

so the `P` tokens of one sequence occupy `P` consecutive positions and each
is rotated at its own. At `P = 1` this is the familiar `L_g - 1`. The
position comes from the **device** `sequence_length` tensor: running the
same prepared step with `host_past_key_value_lengths` perturbed to
`sequence_length - 1` left every position, every pool slot and
`cu_kv_seqlens` unchanged (observed on the fp8 path at `P = 3`).

**On the bf16 path**, one call then computes, with `R = qk_rope_head_dim`
and `C = kv_lora_rank` (the fp8 path replaces effects 1 and 2 — see below):

```
# GPT-J (interleaved-pair) RoPE, fp32 math, one rounding to bf16.
# (cos_d, sin_d) = rotary_cos_sin row pos(g,t), pair d, d in [0, R/2)
rope(x, p)[2d]   = x[2d] * cos_d - x[2d+1] * sin_d
rope(x, p)[2d+1] = x[2d] * sin_d + x[2d+1] * cos_d

# for every row n = g*P + t:
# 1. q RoPE into the fused-q tail (per head h):
fused_q[n, h, C:] = rope(q_pe[n, h, :], pos(g,t))
# 2. latent-cache append, at page/slot of position pos(g,t) in seq g's blocks:
cache_row(g, pos(g,t)) = concat(latent_cache[n, :C],                  # bitwise copy
                                rope(latent_cache[n, C:], pos(g,t)))  # k_pe rotated
# 3. decode-FMHA scheduler buffers (trtllm-gen MQA layout):
cu_q_seqlens[0:G+1]  = arange(G+1) * num_heads * P
cu_kv_seqlens[0:G+1] = [0, cumsum(L_g over generation seqs)]
fmha_scheduler_counter[0] = 0
```

Fusion boundary: q RoPE, k_pe RoPE, paged-cache append and scheduler-buffer
fill happen inside the call. The caller still owns the up/down projections,
the absorbed-q BMM that fills `fused_q[..., :C]`, and the subsequent
latent-space attention. `q_pe` and `latent_cache` are inputs only: despite
the mutable schema annotation on `q_pe` neither is modified (observed).

On the bf16 path `fused_q[..., :C]` is bitwise untouched and never read, so
the absorbed-q BMM and this call may run concurrently. **On the fp8 path
they may not** — see below.

### fp8-e4m3 latent pool (`quant_mode=128`)

`quant_mode=128` is `QuantMode`'s fp8-KV-cache bit. It switches the
**latent pool's** element type to fp8-e4m3, one byte per element, while
`fused_q`, `q_pe` and `latent_cache` all stay bf16. Certified at the
DeepSeek-R1-0528 cell — `H = 128`, `tokens_per_block = 32`,
`C/R/nope/v = 512/64/128/128`, `q_lora_rank = 1536`, single-layer pool — at
`P` 1, 2, 3 and 4, with the KV scaling factor omitted (the production call)
and explicit 1.0 / 1.5 / 2.0 swept beside it, at `q_scaling` 1.0 and
DeepSeek-R1's YaRN value `1/mscale^2 = 0.5336594`.

Two independent fp32 scalars steer it, each with its own role and neither
derived from the other:

```
w = kv_scale_orig_quant[0]   # 1.0 when that tensor is None
r = kv_scale_quant_orig[0]   # 1.0 when that tensor is None
e4m3(x) = round-to-nearest cast of an fp32 x to float8_e4m3fn
```

One call then computes, with `rope` as above and `bf16(.)` the rounding to
the activation dtype that happens before quantization, for every row
`n = g*P + t` at `p = pos(g,t)`:

```
# 1. q RoPE, quantized. fused_q is READ, never written:
quant_q_buffer[n, h, :C] = e4m3(fused_q[n, h, :C] * w)
quant_q_buffer[n, h, C:] = e4m3(bf16(rope(q_pe[n, h, :], p)) * w)
# 2. latent-cache append, quantized, at page/slot of position p:
cache_row(g, p) = e4m3(concat(latent_cache[n, :C],
                              bf16(rope(latent_cache[n, C:], p))) * w)
# 3. decode-FMHA scales (a per-batch scalar pair — not per sequence, and
#    P does not enter them):
x = r*r / (q_scaling * sqrt(qk_nope_head_dim + qk_rope_head_dim))
mla_bmm1_scale = [x, x * log2(e)]
mla_bmm2_scale = [r]
# 4. scheduler buffers: exactly as on the bf16 path
```

`w` is a **per-tensor static scale**, not a block or dynamic one: every
appended row and every quantized q element is the same multiply-and-round,
bit-exact against that mirror at each scale tested.

Fusion boundary on this path: the same RoPE / append / scheduler fill, plus
the q quantization and the two scale derivations. The caller still owns the
absorbed-q BMM — and now must have **finished** it before this call, since
`fused_q[..., :C]` is read here to build `quant_q_buffer`'s head. (The
engine's own MLA module enforces exactly that by dropping the auxiliary
stream when the KV cache is fp8.)

**Why `r` appears in both scales.** The fp8 MLA generation FMHA — the
sibling op `trtllm::attention` called with
`attention_input_type = generation_only`, `is_mla_enable=True` — takes its
query from `quant_q_buffer` rather than from the bf16 fused q, takes its
BMM1 scale from `mla_bmm1_scale[1]` (the log2-domain copy; `[0]` is inert
there) and its BMM2 scale from `mla_bmm2_scale[0]`, and reads neither KV
scale tensor. Dequantization is therefore the producer's job: `r*r` undoes
the `w` on the query and on `K`, and `r` undoes the `w` on `V = K[:, :C]`,
leaving the plain softmax scale `1 / (q_scaling * sqrt(nope + R))`. That
cancellation is only correct when `r = 1/w`; this op does not check it, and
a non-reciprocal pair produces a silently mis-scaled decode (pinned here by
running `w = 0.25` against `r = 3.0` and observing each drive its own half).
The consuming call's behaviour above is that op's fact, established
separately — this entry's test gates the values written, not their use.

## Signature

```python
def mla_rope_generation(
    fused_q: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    rotary_cos_sin: Optional[torch.Tensor],
    cu_q_seqlens: torch.Tensor,
    cu_kv_seqlens: torch.Tensor,
    fmha_scheduler_counter: torch.Tensor,
    mla_bmm1_scale: Optional[torch.Tensor],
    mla_bmm2_scale: Optional[torch.Tensor],
    quant_q_buffer: Optional[torch.Tensor],
    sequence_length: torch.Tensor,
    host_past_key_value_lengths: torch.Tensor,
    host_context_lengths: torch.Tensor,
    num_contexts: int,
    kv_cache_block_offsets: Optional[torch.Tensor],
    host_kv_cache_pool_pointers: Optional[torch.Tensor],
    host_kv_cache_pool_mapping: Optional[torch.Tensor],
    kv_scale_orig_quant: Optional[torch.Tensor],
    kv_scale_quant_orig: Optional[torch.Tensor],
    out_scale: Optional[torch.Tensor],
    block_ids_per_seq: Optional[torch.Tensor],
    helix_tensor_params: List[Optional[torch.Tensor]],
    predicted_tokens_per_seq: int,
    layer_idx: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    tokens_per_block: int,
    attention_window_size: int,
    beam_width: int,
    quant_mode: int,
    q_scaling: float,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    rope_append: bool,
) -> None
```

With `G` = generation sequences, `P` = `predicted_tokens_per_seq`,
`S` = total sequences (`num_contexts + G`), `H` = `num_heads`,
`C` = `kv_lora_rank`, `R` = `qk_rope_head_dim`, `D = C + R`
(= `head_size`). The four per-token tensors have `G * P` rows, ordered
token-major within a sequence (row `n = g*P + t`); everything else is one
entry per sequence and does **not** scale with `P`:

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `fused_q` | `[G*P, H, D]`; bf16 pool: only `[..., C:]` written, `[..., :C]` untouched and unread. fp8 pool: **entirely read-only**, and `[..., :C]` must already hold the absorbed q | bf16 | contiguous | CUDA |
| `q_pe` | `[G*P, H, R]`; read only | bf16 | last dim contiguous; strided head-dim views (e.g. a slice of packed `[G*P, H, nope+R]` q) work | CUDA |
| `latent_cache` | `[G*P, D]` = per-token `[compressed_kv \| k_pe]`; read only | bf16 | contiguous | CUDA |
| `rotary_cos_sin` | `[1, max_pos * R * 2]`: per position `R` fp32 `(cos, sin)` pairs, second `R/2` a duplicate of the first (duplicated/`duplicate_data=True` layout; only pairs `[0, R/2)` are read per position) | fp32 | contiguous | CUDA |
| `cu_q_seqlens` | `[>= G+1]`; exactly `G+1` entries written whatever `P` is (entries past `G+1` are left alone — observed) | int32 | contiguous | CUDA |
| `cu_kv_seqlens` | `[>= G+1]`; same, `G+1` entries | int32 | contiguous | CUDA |
| `fmha_scheduler_counter` | `[1]`; written (zeroed) | uint32 | — | CUDA |
| `mla_bmm1_scale` | bf16 pool: `None`. fp8 pool: `[2]`, written with `[x, x*log2(e)]`; may be `None`, and is then simply not written | fp32 | contiguous | CUDA |
| `mla_bmm2_scale` | bf16 pool: `None`. fp8 pool: `[1]`, written with `[kv_scale_quant_orig]`; may be `None`, and is then simply not written | fp32 | contiguous | CUDA |
| `quant_q_buffer` | bf16 pool: `None`. fp8 pool: `[G*P, H, D]`, **required** — one e4m3 byte per element, fully overwritten | uint8 (a `float8_e4m3fn` view of the same bytes behaves identically) | contiguous | CUDA |
| `sequence_length` | `[S]`; per-seq total KV length incl. **all `P`** of this step's tokens. This is the tensor the `P` positions and `cu_kv_seqlens` are derived from | int32 | contiguous | CUDA |
| `host_past_key_value_lengths` | `[S]`; pass the same values as `sequence_length` (total KV length, **not** past-only, despite the name). Perturbing it to `sequence_length - 1` changed no observable effect at the certified fp8 cell, so it is `sequence_length` that steers this op; larger divergences were not measured | int32 | contiguous | CPU |
| `host_context_lengths` | `[S]`; per-seq prompt length | int32 | contiguous | CPU |
| `num_contexts` | leading context-phase sequence count | Python int | — | — |
| `kv_cache_block_offsets` | `[1, >= S, 2, max_blocks_per_seq]`; raw block ids per seq (K and V rows identical, kv_factor=1 pool) | int32 | contiguous | CUDA |
| `host_kv_cache_pool_pointers` | `[num_pools, 2]`: (primary ptr, secondary ptr=0) | int64 | contiguous | CPU |
| `host_kv_cache_pool_mapping` | `[num_layers, 2]`: (pool index, layer-within-pool) row per layer | int32 | contiguous | CPU |
| `kv_scale_orig_quant` | `[1]`, the **write-side** factor `w`: everything quantized by this call is multiplied by it. `None` = 1.0 (what the engine's own MLA call site passes). Ignored on the bf16 path | fp32 | contiguous | CUDA |
| `kv_scale_quant_orig` | `[1]`, the **read-side** factor `r`: folded into `mla_bmm1_scale` (as `r*r`) and copied into `mla_bmm2_scale`. `None` = 1.0. Not used for anything else, and not derived from `kv_scale_orig_quant`. Ignored on the bf16 path | fp32 | contiguous | CUDA |
| `out_scale` | `None`; fp8-output scale otherwise (not certified) | — | — | — |
| `block_ids_per_seq` | `None`; flash-MLA layout otherwise (not certified) | — | — | — |
| `helix_tensor_params` | `[None, None]`; helix position offsets / inactive-rank mask otherwise (not certified) | Python list | — | — |
| `predicted_tokens_per_seq` | `P`, the query tokens each generation sequence carries. `1` (ordinary decode) and `2`, `3`, `4` (the MTP path, `max_draft_len + 1`) certified. It is a single scalar for the whole batch — a ragged per-sequence draft length cannot be expressed here. Values above 4 untested | Python int | — | — |
| `layer_idx` | row into `host_kv_cache_pool_mapping` | Python int | — | — |
| `num_heads` | query heads `H`; 16 and 128 certified | Python int | — | — |
| `num_kv_heads` | `1` (MLA latent cache) | Python int | — | — |
| `head_size` | `D = kv_lora_rank + qk_rope_head_dim` | Python int | — | — |
| `tokens_per_block` | pool page size; 32 and 64 certified on the bf16 path, 32 on the fp8 path (32 is what a default `KvCacheConfig` produces) | Python int | — | — |
| `attention_window_size` | `>= max total KV length`; smaller values imply cyclic-cache addressing (not certified) | Python int | — | — |
| `beam_width` | `1` | Python int | — | — |
| `quant_mode` | `0` (bf16 latent pool) and `128` (fp8-e4m3 latent pool) certified | Python int | — | — |
| `q_scaling` | softmax-scale divisor baked into `mla_bmm1_scale`; live on the fp8 path (1.0 and 0.5336594 certified), inert on the bf16 path, which writes no scale at all (bitwise-identical outputs at 1.0 and 7.5, observed) | Python float | — | — |
| `q_lora_rank` | unused by both certified paths (`0` and `1536` both fine) | Python int | — | — |
| `kv_lora_rank` | `C` | Python int | — | — |
| `qk_nope_head_dim`, `qk_rope_head_dim` | model MLA dims (`R` must match `q_pe`/table; their sum is the `sqrt` in `mla_bmm1_scale`) | Python int | — | — |
| `v_head_dim` | model v head dim; not read on either certified path | Python int | — | — |
| `rope_append` | `True` certified (rotate k_pe + append to cache) | Python bool | — | — |
| returns | — (mutates `cu_q_seqlens`, `cu_kv_seqlens`, `fmha_scheduler_counter`, the paged pool, plus `fused_q`'s tail on the bf16 path or `quant_q_buffer`/`mla_bmm1_scale`/`mla_bmm2_scale` on the fp8 path) | — | — | — |

## Metadata consumed

No thread-local or registered-layer state: all inputs are explicit
arguments. The length/addressing tensors are exactly what a prepared
`TrtllmAttentionMetadata` and its `KVCacheManager` expose:
`kv_lens_cuda_runtime` → `sequence_length`, `kv_lens_runtime` →
`host_past_key_value_lengths`, `prompt_lens_cpu_runtime` →
`host_context_lengths`, `metadata.kv_cache_block_offsets`,
`manager.kv_cache_pool_pointers`, `manager.kv_cache_pool_mapping`.

## Preconditions

- The paged pool addressed by `host_kv_cache_pool_pointers` +
  `host_kv_cache_pool_mapping[layer_idx]` is an MLA latent cache:
  kv_factor 1, one kv head, row width `D`, page size `tokens_per_block`.
  Per page the layout is `[tokens_per_block, D]` and the page slab is
  `tokens_per_block * D` **elements**, whose byte width the op derives from
  `quant_mode` alone: 2 bytes (bf16) at `quant_mode=0`, 1 byte (e4m3) at
  `quant_mode=128`. The pool's real element type must match — a
  `KVCacheManager` built with `DataType.BF16` and `DataType.FP8`
  respectively is what was certified.
- Every generation sequence has block capacity for `L_g` tokens **before**
  the call (block offsets in `kv_cache_block_offsets` cover
  `ceil(L_g / tokens_per_block)` pages); with a `KVCacheManager`, call
  `impl.add_token(request_id)` **`P` times** per sequence per step, then
  prepare the metadata. The `P` rows of a sequence may straddle a page
  boundary; the op follows the block-offset row across it.
- `sequence_length` / `host_past_key_value_lengths` hold `L_g` (past + all
  `P` new tokens), and `L_g <= attention_window_size`,
  `L_g - 1 < max_pos` of the table, `L_g >= P`.
- Batch order: context sequences first; `fused_q`/`q_pe`/`latent_cache`
  contain generation tokens only, `G * P` rows in generation-sequence order
  with a sequence's `P` tokens contiguous and in position order.
- The `G * P` cache rows written by one call must address **distinct
  physical slots**. This holds automatically for a `KVCacheManager`-allocated
  batch (each row goes to its own position, distinct sequences hold distinct
  pages), and it is the reason `P > 1` needs no extra care from the caller. If
  a caller-built block-offset table aliases them onto one slot, the writes
  race and leave torn cache rows, nondeterministically and with no error —
  reproduced here on purpose as a control, at `P = 2` with 8 sequences
  aliased onto one page: 6 of 6 identical armed calls left different pool
  images, while the un-aliased geometry repeated bitwise in the same
  process.
- `rotary_cos_sin` uses the duplicated `(cos, sin)`-pair layout above
  (`RopeParams(..., duplicate_data=True).create_rope_const_params()`
  produces it; MLA models set `duplicate_data=True`). A table with
  cos=1/sin=0 makes both RoPEs identity (pre-rotated inputs pass through).
- `cu_q_seqlens`/`cu_kv_seqlens`/`fmha_scheduler_counter` are
  caller-allocated with the dtypes above; contents need not be initialized.
- **bf16 path only:** `quant_mode=0` and every optional fp8/scale tensor
  `None`.
- **fp8 path only** (`quant_mode=128`):
  - `quant_q_buffer` must be allocated `[G*P, H, D]` at one byte per element.
    It is **not** presence-checked: passing `None` is not rejected, the
    kernel launches anyway and the run dies with `CUDA error: an illegal
    memory access was encountered` (observed in a scratch probe at this
    configuration; the CUDA context is lost, the process aborts, and the KV
    manager's own `release_pools` fails on the way out). Its previous
    contents are irrelevant — the call overwrites every byte.
  - `fused_q[..., :C]` must already hold the absorbed q **when the call is
    issued**: this path reads it. Overlapping the absorbed-q BMM with this
    call on a second stream is a race here, unlike on the bf16 path.
  - `mla_bmm1_scale` `[2]` fp32 and `mla_bmm2_scale` `[1]` fp32 may be
    `None` — accepted, and the buffer is then simply not written (observed;
    the rest of the outputs are unaffected). A decode consuming those scales
    still needs them, so in practice both are allocated.
  - `kv_scale_orig_quant` and `kv_scale_quant_orig` are `[1]` fp32 CUDA
    tensors or `None` (= 1.0). They are used independently; pass reciprocals
    (`kv_scale_orig_quant = 1/s`, `kv_scale_quant_orig = s`) unless a
    deliberately asymmetric round trip is what you want. `None` for both is
    the production configuration and is what the engine's own MLA call site
    passes.
  - Input magnitudes must survive e4m3: `|value * w|` below `2**-10`
    (~9.8e-4) rounds to zero, and 448 is the largest finite e4m3 value (what
    the kernel does above it was not measured). The certified runs are
    unit-normal activations at `w <= 1`.

## Notes

- Schema annotation is wrong in both directions: it marks `fused_q` and
  `q_pe` mutable (`Tensor(a!)`), but `q_pe` is never modified — and on the
  fp8 path neither is `fused_q` — while `cu_q_seqlens`, `cu_kv_seqlens`,
  `fmha_scheduler_counter`, `quant_q_buffer`, `mla_bmm1_scale` and
  `mla_bmm2_scale` — all filled by the call — carry no annotation at all. Do
  not rely on the schema's alias info (e.g. under torch.compile
  functionalization).
- Precision, bf16 pool (sm_100): the `compressed_kv` copy is bitwise. Both
  RoPE outputs match an fp32 reference rounded once to bf16 to within **one
  bf16 ulp**, and are bitwise on all but ~1.5e-5 of elements: the kernel and
  a torch fp32 reference evaluate `x*cos ∓ y*sin` in different orders, so
  where the correctly-rounded fp32 result lands exactly on a bf16 rounding
  midpoint the two round to adjacent values. Measured on a 64-sequence
  decode step: 1 such element in 65 536 rotated `q_pe` values and 0 in 4096
  rotated `k_pe` values, with identical counts at page sizes 32 and 64 — it
  is arithmetic, not addressing. Default `assert_close` tolerances absorb it
  with margin, which is what the test gates on.
- Precision, fp8 pool (sm_100): **everything came back bit-exact**, both the
  pure copies (`quant_q_buffer`'s absorbed-q head, the appended
  `compressed_kv` half) and the roped halves. e4m3's 3-bit mantissa absorbs
  the evaluation-order difference the bf16 pool pays one ulp for, but the
  double rounding is real and a mirror must reproduce it: the kernel rounds
  the fp32 RoPE result to **bf16 first** and quantizes that. Quantizing the
  fp32 RoPE result directly disagrees with the kernel on ~3% of the roped q
  elements (and up to 9% of the much smaller per-row `k_pe` samples). The
  test still gates the roped halves at one e4m3 ulp
  (`2**-3` relative) plus a bit-exact-fraction floor rather than bitwise,
  because the tie it covers is arithmetic; neither half has been approached.
- The fp8 scales are exact enough to gate tightly: `mla_bmm2_scale` is a
  bitwise copy of `kv_scale_quant_orig`, not a computation — a scratch probe
  passing 0.7 (whose fp32 value is 0.699999988079071) got that exact fp32
  number back, and the test gates the copy bitwise at 1.0 / 1.5 / 2.0 / 3.0.
  `mla_bmm1_scale` matches a double-precision reference rounded once to fp32
  to within 7.2e-8 relative (0.6 fp32 ulp) across the swept grid, exactly on
  several cases.
- Pool writes are exactly bounded: `P` `D`-byte rows per generation
  sequence, at the slots its `P` positions address, and nothing else in the
  pool moves — checked by a whole-pool byte snapshot on every fp8 case, at
  `P` 1 through 4, 64-sequence batches included.
- Re-running the same prepared step overwrites the same `P` cache slots
  (positions are derived from `sequence_length`), so a repeated call is
  idempotent, not double-appending — and on the fp8 path the second run
  reproduces `quant_q_buffer` and the pool bitwise. This holds at `P > 1`
  too: the rows do not advance by `P` on the second call.
- The scheduler buffers feed the trtllm-gen decode MQA: `cu_q_seqlens` is
  in units of q rows (`H` rows per generation *token*, so `H * P` per
  generation sequence), `cu_kv_seqlens` in tokens (the cumulative
  `sequence_length` over the generation sequences, which already includes
  all `P` new tokens). They are filled identically on both paths, and both
  fills were observed at `P` 1, 2, 3 and 4 — `cu_q_seqlens[i] = i * H * P`
  exactly, and no entry past index `G` is written.
- Page size (sm_100): certified at `tokens_per_block` 32 and 64 on the bf16
  path, 32 on the fp8 path. 32 is what a `KvCacheConfig` at its defaults
  hands the op; 64 is what a tuned MLA target opts into. At 32 the appended
  rows were placed at every alignment a 32-slot page has — position 31 (a
  page's last slot), 32 and 96 (slot 0 of a fresh page after one and three
  full pages), 64 (after two), and mid-page — including 64-sequence batches
  run for two consecutive steps in which sequences cross into a new page
  between the steps and others open a fresh page on the first, and mixed
  batches whose leading context sequence is skipped. At `P > 1` the same
  sweep covers a sequence whose `P` rows **straddle** the boundary, at every
  split from 1 row in the old page to `P-1` (`P` = 2, 3 and 4, fp8 and bf16
  pools) — a shape unreachable at `P = 1`, where one call writes one slot
  per sequence. Every effect held at the same gates as at page 64. Other
  page sizes are untested rather than known-bad.
- `q_pe` in production is a strided slice of the packed q tensor
  (`[G*P, H, nope+R]` split); the kernel handles that stride — certified for
  both contiguous and packed-slice layouts, on both paths, at `P` 1-4.
- The `rotary_cos_sin` *content* is not a certified axis: what the op is
  gated on is that it applies the table's `(cos, sin)` pairs, and both
  certified configurations ran the unscaled theta-10000 table. A YaRN-scaled
  table of the same layout has not been run through this op here.
- Sibling ops exist for adjacent MLA roles:
  `trtllm::mla_rope_append_paged_kv_assign_q` (context-phase RoPE + cache
  append), `trtllm::attention` / `trtllm::mla_custom_op_inplace` (the
  attention core that consumes `fused_q` or `quant_q_buffer` and these
  scheduler buffers), and `trtllm::merge_chunked_attention_for_mla`.
- `predicted_tokens_per_seq` is the *only* way a multi-token generation
  query reaches this op: its signature carries no spec-decoding mask, tree
  or draft-token argument of any kind, so there is nothing else for an MTP
  caller to set here. (What the resulting query block is then allowed to
  attend to is the attention op's fact, not this one's.)
- Not certified on either path: `predicted_tokens_per_seq` above 4,
  `block_ids_per_seq` (the flash-MLA layout), helix, `out_scale` (fp8
  attention output), `attention_window_size` below the KV length, and
  multi-layer pools. Not certified on the fp8 path specifically:
  `tokens_per_block = 64`, head counts other than 128, and activation dtypes
  other than bf16.


## rc26 additions

Parameters that did not exist in 1.3.0rc21. Every value this entry certifies
reproduces the op's pre-rc26 behaviour, and matches what the in-tree caller
passes on the same path.

| Parameter | Certified value | Why |
|---|---|---|
| `kv_cache_scale_orig_quant` | `None` | With no value the op uses `kv_scale_orig_quant` for the cache scale (`dsv3RopeOp.cpp`), which is exactly what it did before this parameter existed. |
| `residual_dim` | `0` | Must be `0` or `rope_size`; non-zero requires an FP4 KV pool, and this entry certifies bf16 and fp8-e4m3 pools. |
| `kv_norm_weight` / `kv_norm_eps` | `None` / `1e-6` | A non-`None` weight folds the `kv_a_layernorm` into this kernel, which then reads `latent_cache` **raw**. A caller that already normalized would normalize twice. Only DeepSeek-V4's sparse module folds it upstream. |
| `precomputed_cu_seqlens` | `False` | The kernel fills the cu-seqlens buffers itself; `True` only when the Q half is skipped. |
| `precomputed_fmha_scheduler` | `False` | The scheduler counter and the two bmm scales come from this call, not from a sparse index kernel. |
| `kv_only` / `kv_done_elsewhere` | `False` / `False` | Both halves run in this one call. |
| `quant_scale_qkv` | `None` | `q_nope` in `quant_q_buffer` is not pre-quantized. |
