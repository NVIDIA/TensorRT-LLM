---
receipts:
  sm_100: {status: passed, trtllm: 1.3.0rc21}
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 96}
---

# thop_attention

**Wraps** `tensorrt_llm.bindings.internal.thop.attention` (one call).

This is a pybind binding, not a `torch.ops` op; its inclusion in the catalog
is an approved policy exception to the `torch.ops.trtllm.*` entry shape.

## Semantics

The fused attention core of one decoder layer over one prepared batch, with
**fully explicit state**: unlike the registered-layer attention entry
points, this binding reads no Python-side registry, no thread-local
metadata, and no layer objects. Every piece of batch state, cache
addressing, and layer config is an argument.

The certified surface is bf16 activations over a caller-owned paged pool —
bf16 (`quant_mode=0`) or fp8-e4m3 (`quant_mode=128`, see *Standard
configuration over an fp8-e4m3 KV pool* and *MLA over an fp8-e4m3 latent
pool*; the two configurations quantize differently and are certified
separately) — with no other quantization and no spec-dec / sparse / cross /
helix / beam features (their arguments held at the inert values listed in
*Signature*), in two configurations:

- **Standard** (`is_mla_enable=False`): packed-QKV GQA/MHA self-attention,
  paged KV-cache append + masked FMHA, mixed batches in one call, over a
  pool holding one layer or several layers side by side (see *Paged KV
  cache addressing*), on either context execution path
  (`use_paged_context_fmha` `False` or `True` — the latter is what lets a
  context call carry a cached prefix, see *Paged-context FMHA*), optionally
  with per-query-head **attention sinks** (bf16 pool, causal mask — see
  *Attention sinks*) and/or a **sliding window** (`attention_window_size`
  smaller than the sequence — see *Sliding window*).
- **MLA** (`is_mla_enable=True`): DeepSeek-style multi-head latent
  attention over a paged **latent** cache (kv_factor 1: one latent row of
  width `C + R` per token, `C = kv_lora_rank`, `R = qk_rope_head_dim`),
  bf16 or fp8-e4m3.
  A batch is served by **one call per phase** — `attention_input_type=1`
  (context_only) over the context rows, then `=2` (generation_only) over
  the generation rows — sharing the same full-batch state tensors. Mixed
  `attention_input_type=0` is rejected for MLA. The context phase has two
  certified flavors selected by `latent_cache`: fresh prefill
  (`latent_cache` given: in-kernel RoPE + latent append) and no-append
  (`latent_cache=None`: pure FMHA over explicit K/V — the cached-KV and
  chunked-prefill context flows, including `softmax_stats_tensor` output).
  Both flavors, the generation call and the mixed-batch pairing of two of
  them are certified over **both** pool element types; the chunked
  partial-pass pattern of the no-append flavor is bf16-only. The generation
  call is additionally certified at `predicted_tokens_per_seq` 1, 2, 3 and 4
  — one speculative-decoding step, where a generation sequence contributes a
  `P`-token draft chain instead of a single token.
  Mixed `attention_input_type=0` is **not** rejected — nothing in the
  binding checks it (the "MLA cannot be mixed" `ValueError` lives one layer
  up, in the Python attention backend, which this op bypasses). On a
  single-phase batch it is bitwise identical to that phase's own value; on a
  genuinely mixed batch it runs **both** phase paths off one set of scalars,
  and `head_size`, the `q`/`output` row widths and
  `cu_q_seqlens`/`cu_kv_seqlens` can serve only one of them — so at most one
  row group comes back right, the other is silently wrong or never written,
  and the generation write overruns a context-sized `output`. Pass the
  phase's own value.

### Standard configuration

One call, for a batch of `ns` sequences — context-phase sequences first
(`host_request_types[i] == 0`, `i < num_contexts`), generation-phase after
(`== 1`). Rows of `q` are: all context tokens sequence by sequence
(`num_ctx_tokens` rows total), then one row per generation sequence
(`predicted_tokens_per_seq == 1`). For each sequence `s` with new-token
count `l_s` and total KV length `kv_s = sequence_length[s]`:

```
# 1. slice K and V out of the packed q rows and append them to the paged
#    cache at token positions kv_s - l_s .. kv_s - 1
#    (a generation sequence appends one token at position kv_s - 1; a context
#     sequence appends l_s of them, l_s < kv_s only on the paged-context path
#     — see *Paged-context FMHA*)
# 2. for each new query token i (absolute position p = kv_s - l_s + i),
#    each query head h, over kv head h // (Hq/Hkv):
keys      = [0, p]                   # mask_type == 1 (causal)
keys      = [p - W + 1, p]           # mask_type == 1, W = attention_window_size
                                     #   < p + 1 (see *Sliding window*)
keys      = [0, kv_s - 1]            # mask_type == 0 (padding: no mask)
logits    = q[i, h] @ K[keys].T / (q_scaling * sqrt(D))
out[i, h] = softmax(logits) @ V[keys]                          # fp32 accum
# with attention_sinks given (see *Attention sinks*):
out[i, h] = softmax(concat(logits, sink[h]))[:-1] @ V[keys]
```

Results are written into `output[:num_tokens]` in place; the op returns
nothing. Generation sequences always attend over all `kv_s` cached+current
tokens (a decode step is `l_s = 1`), or over the newest `W` of them when a
sliding window is active.

Fusion boundary: KV-cache append, masking, softmax, and GQA attention
happen inside the call (plus RoPE when `position_embedding_type` selects
it — not certified for this configuration). The caller owns the QKV
projection, any q/k norm, external RoPE, and the output projection.

#### Paged-context FMHA

`use_paged_context_fmha` selects the **context** execution path; it does
not change the generation path. Both values are certified.

- `False` — the context FMHA reads K and V out of the **packed `q` rows**
  and never touches the pool, so a context sequence can only attend to its
  own new tokens: `kv_s` must equal `l_s` (the sequence starts empty).
- `True` — the context FMHA reads K and V out of the **paged pool**, after
  this call's own append has written the new tokens into it. `kv_s > l_s`
  is then legal: the sequence's first `kv_s - l_s` tokens are a cached
  prefix that earlier calls wrote, query row `i` sits at absolute position
  `p = kv_s - l_s + i`, and the causal mask is bottom-right aligned exactly
  as *Standard configuration* states. This is what KV-cache reuse and
  chunked prefill need, and an engine turns it on for every call of the
  batch as soon as either is enabled (trtllm's defaults enable reuse).

Two consequences pinned by test:

- **With nothing cached the flag is observationally inert.** A fresh
  prefill and a decode step run from identical state at both values come
  back **bitwise identical** in `output` and in pool content (certified on
  32/8/128, 32/4/128 and 64/8/64). The K/V source does move even then —
  aliasing every page of a fresh-prefill sequence onto one garbage page
  leaves the `False` output bitwise unchanged but moves the `True` output
  31x outside the tolerance band — it is the append running first that
  leaves the pool holding exactly the packed rows.
- **With a cached prefix the flag is load-bearing and its absence is
  silent.** The same cached-prefix context call at `False` returns without
  raising, 51x outside the band around the correct answer. Nothing in the
  op checks the combination.

The append is unchanged by the flag: the new tokens land bit-exactly at
their absolute positions and nothing outside the token ranges is written,
so a repeated identical call is idempotent on both paths.

Certified surface for `True`: the standard bf16-pool configuration,
`mask_type=1` (causal), `is_fused_qkv=True`, `q_scaling=1.0`,
`tokens_per_block` 32, geometries 32/8/128, 32/4/128 and 64/8/64, on a
single-layer pool and on a real 4-layer `KVCacheManager` pool, with and
without `attention_sinks` and a sliding window. `True` with `mask_type=0`
(padding), with the fp8-e4m3 pool, or with MLA is not certified (MLA
context takes separate q/k/v and its own cached-KV flavor instead).

#### Head geometry

Head **counts are a free axis**, not an enumerated list: `num_heads` (`Hq`)
and `num_kv_heads` (`Hkv`) are plain runtime parameters of the same
kernels, and every pair satisfying `Hq % Hkv == 0` is served — MHA
(`Hq == Hkv`), MQA (`Hkv == 1`), and any GQA ratio, power of two or not.
Certified on the bf16 pool: `Hq`/`Hkv` of 32/8 and 32/4 (the two shipped
targets), 64/8, 16/1, 28/4, 12/3, 8/2, 4/4 — ratios 1, 4, 7, 8 and 16, with
non-power-of-2 head counts (28, 12, 3) among them. All of them run the same
kernels at the same accuracy (see *Notes*); nothing distinguishes the
"round" geometries.

`head_size` (`D`) is the **enumerated** axis: it selects the FMHA kernel,
and only the shipped head dims exist. Certified: 64, 128, 256 (all on the
bf16 pool; 128 also on the fp8-e4m3 pool). A scratch probe additionally saw
80 work (its generation kernel is JIT-generated), while 32, 96 and 192 are
rejected — hard, see *Preconditions*. Treat any `D` outside {64, 128, 256}
as untested rather than merely unsupported. Certified `Hq`/`Hkv`/`D`
triples on the bf16 pool: 32/8/128, 32/4/128, 64/8/64, 16/1/128, 28/4/128,
12/3/128, 8/2/128, 8/2/256, 4/4/64.

`Hq % Hkv != 0` is the one geometry error the op does not report. A
context-only call at 6/4/128 returned normally, having computed only the
first `(Hq // Hkv) * Hkv = 4` head columns of `output` and left the other
two all-zero; the same geometry does raise once a generation-phase call
selects a decode kernel (`numQHeads should be multiple of numKVHeads`, from
the XQA dispatcher), and `Hq = 7`, `Hkv = 2` aborts the process outright
inside kernel selection. Because the context path fails silently, the
wrapper asserts `num_heads % num_kv_heads == 0` before the call — one of the
entry's two guard asserts (the other guards `attention_sinks`), holding for
every certified configuration including MLA (8/8, 16/16, 32/32 and 128/128
context, 8/1, 16/1, 32/1 and 128/1 generation).

#### Attention sinks

`attention_sinks` is an optional fp32 `[Hq]` CUDA tensor of per-query-head
**sink logits**. When given, each head's softmax gains one extra logit
column that is dropped from the weighted sum: the sink lands in the
denominator only, so the attention weights of every row sum to less than 1
and the output shrinks accordingly. For query row `i`, head `h`, over the
masked logit range of *Standard configuration*:

```
out[i, h]  = softmax(concat(logits, sink[h]))[:-1] @ V[:n_keys]
           = mass[i, h] * (softmax(logits) @ V[:n_keys])
mass[i, h] = sigmoid(logsumexp(logits) - sink[h])       # < 1, per (row, head)
```

Facts pinned by test, not by inspection:

- The sink is a logit in the **scaled**-score domain — it is compared
  against `q @ K.T / (q_scaling * sqrt(D))`, *after* the softmax scale, not
  against the raw dot products. (Certified at `head_size = 64`,
  `q_scaling = 1.0`, so a scale of 1/8; the rival "sink joins the unscaled
  row" hypothesis sits 12.8x to 41x outside the tolerance band on every
  certified case.)
- It is honoured in **both** phases, which take different FMHA kernels:
  the context kernel, the short-history decode kernel, and the
  multi-CTA-KV decode kernel that folds partial softmax states across CTAs
  (`...MultiCtasKvCga...`, selected at 600 and 2000 cached tokens). A
  single `attention_input_type=0` call mixing a context and a generation
  sequence applies it to both row groups.
- Indexing is by **query** head: sink `h` belongs to `output`'s head column
  `h`, not to the kv head `h // (Hq/Hkv)`.
- The KV-cache append is unaffected — the appended pages stay bit-exact and
  nothing outside the token ranges is written.
- `attention_sinks=None` and a sink that cannot contribute (`-100.0` or
  `-inf` per head) produce **bitwise identical** output in both phases: the
  argument's only effect is the one added denominator term.

Certified surface for sinks: the standard configuration over a bf16 pool
(`quant_mode=0`), `mask_type=1` (causal), `is_fused_qkv=True`,
`q_scaling=1.0`, geometry 64/8/64 — the gpt-oss-120b tp1 layer shape —
both without a window (`attention_window_size = max_seq_len`) and with the
sliding window of *Sliding window* below, and on both context execution
paths including a context call over a cached prefix (see *Paged-context
FMHA*). Sinks combined with `mask_type=0`, with the fp8-e4m3 pool, with
`q_scaling != 1.0`, or with MLA are not certified.

#### Sliding window

`attention_window_size` (`W`) is a **per-call token count**. When a
sequence's KV length exceeds it, query row at absolute position `p` attends

```
keys [max(0, p - W + 1), p]          # exactly min(p + 1, W) keys
```

— `W` keys, never `W + 1`; a sequence still shorter than `W` is plain
causal. `W >= every sequence's total KV` is the no-window behavior. The
window applies identically to context rows and generation rows, and with
`attention_sinks` the sink logit joins the denominator of that **windowed**
logit set (each key of a uniform-logit row of `n = min(p + 1, W)` keys
weighs `1 / (n + exp(sink[h]))`).

Facts pinned by test, not by inspection:

- The boundary is exact. A one-hot-V probe (all-zero K, so every unmasked
  logit is 0 and the softmax is uniform) reads the per-key weights straight
  out of `output`: keys below `p - W + 1` come back **bitwise zero**, keys
  from `p - W + 1` up to `p` all carry the same `1 / (n + exp(sink[h]))`
  with `n = min(p + 1, W)`. Certified at `W = 128` in both phases and at
  `W = 33` and `100` — the window is a token count, not a page count, and
  may be smaller than `tokens_per_block`.
- **The window is a mask only: cache addressing is unchanged.** The op
  appends every new token at its **absolute** position — page
  `t // tokens_per_block` of the sequence's `kv_cache_block_offsets` row,
  slot `t % tokens_per_block` — and wraps nothing, neither at `W` nor at
  `max_seq_len`. Whatever cyclic reuse of pool memory happens is the
  caller's page mapping, not the op's (see *Paged KV cache addressing under
  a sliding window*).
- A single context prefill longer than `W` is legal in one call: each row
  gets its own window. At `use_paged_context_fmha=False` the context FMHA
  reads the packed `q` rows, never the pool — aliasing every page of the
  sequence onto one garbage page left the context output bitwise identical
  (the append still runs, so the pool itself is then wrong). At `True` the
  same aliasing moves the output far off: that path reads the pool (see
  *Paged-context FMHA*).
- `sequence_length` must stay the **global** cached+new count: it is both
  the origin the window is measured back from and the append position.
  Capping it to `W` moves the write to slot `W - 1` and truncates the
  attended range — silently, no error.
- `attention_window_size` is per call, so layers with different windows
  share one pool, one layer→pool mapping and one block-offset table inside
  the same batch (certified on a real 4-layer `KVCacheManager` pool with
  windows alternating 128 / full — the gpt-oss layer alternation). The
  shared pool must then still be sized for the full-attention layers; a
  smaller pool per window group needs multi-pool addressing, which is not
  certified.

Certified surface for the window: the standard bf16-pool configuration,
`mask_type=1`, `is_fused_qkv=True`, `q_scaling=1.0`, geometry 64/8/64,
`tokens_per_block` 32, `W` of 128 / 100 / 33, with and without
`attention_sinks`, over prefill (including prefills 1.5-2.3x the window),
decode, mixed batches, histories up to 2000 cached tokens, and 250 decode
steps over a wrapping page ring; at `W = 128` also over a context call
whose cached prefix (200 tokens) runs well past the window, on the
paged-context path. `W` with `mask_type=0`, with the fp8-e4m3 pool, or with
MLA is not certified.

### Standard configuration over an fp8-e4m3 KV pool

`quant_mode=128` (the FP8-KV-cache bit alone — a bf16 checkpoint serving
with a quantized cache) switches the pool element type to fp8-e4m3 while
`q` and `output` stay bf16. Both fp32 `[1]` CUDA scale tensors are then
live: `kv_scale_orig_quant` holding `1/s` (applied on write) and
`kv_scale_quant_orig` holding `s` (applied on read), `s` = the KV-cache
scaling factor — `s = 1.0` for uncalibrated bf16-checkpoint serving, the
production default. Passing `None` for either is silently equivalent to
`1.0` rather than an error (see *Preconditions*), so at any other `s` the
tensors are the caller's only defence. The call's three roles were pinned
against per-role references:

- **Append** (both phases): each new K/V token row lands in the pool as
  `e4m3(k * kv_scale_orig_quant)` — fp32 multiply, round-to-nearest cast.
  Bit-exact against that mirror for power-of-2 scales (1.0 and 2.0
  certified); at `s = 1.5` ~0.6% of elements differ from the mirror by
  exactly one e4m3 ulp (the kernel's intermediate precision differs
  slightly from a pure fp32 multiply), never by more. Pages outside the
  written ranges stay bitwise untouched.
- **Context FMHA** reads the **bf16 packed `q` rows, not the pool**:
  context output matches the bf16-KV fp32 reference at the same
  tolerances as the bf16-pool surface (an fp8-round-trip reference is
  ~1.4e-1 off). Prefill accuracy is unaffected by cache quantization;
  only what later reads the cache sees e4m3.
- **Generation FMHA** attends over dequantized pool rows
  `k_hat = v_hat = pool_row * kv_scale_quant_orig`, with `q` additionally
  put through the same e4m3 round-trip in-kernel (quantize by
  `kv_scale_orig_quant`, dequantize by `kv_scale_quant_orig` — pinned by
  a non-power-of-2 scale sweep: only that scale collapses the error).
  Decode output matched an fp32 reference over `(q_hat, K_hat, V_hat)`
  within `2^-4 * (1 + |ref|)` — observed max abs err 3.1e-2 on
  magnitude-~1 outputs, at most 44% of that allowance; the residual is
  the kernel's internal e4m3 handling of the softmax probabilities (the
  decode kernel computes every MMA operand in e4m3).

Certified fp8-pool surface: GQA 32/8/128, `tokens_per_block` 32, causal
mask; on a single-layer pool: context-only / generation-only / mixed
batches, scales 1.0, 1.5, 2.0, decode histories up to 200 tokens with
page-boundary crossings; on a 4-layer shared pool (real `DataType.FP8`
manager state, `s = 1.0`): per-layer context-only prefill and
generation-only page-crossing decode with bit-exact per-layer appends and
bitwise sibling-layer isolation (see the multi-layer note in *Notes*; the
non-1.0 scale sweep and mixed batches were not re-run multi-layer). The
bf16-pool-only extras (padding mask, the paged-context path, and every head
geometry other than 32/8/128) were not re-run under fp8. MLA over an fp8
pool is certified separately, and behaves differently in every role but the
append — see *MLA over an fp8-e4m3 latent pool*. Nothing in this section
transfers to it: in particular "context FMHA reads the bf16 `q` rows, not
the pool" is a fact about **this** configuration only.

### MLA configuration

Both phases scale `QK^T` by
`1 / (q_scaling * sqrt(qk_nope_head_dim + qk_rope_head_dim))` — the
generation phase too, despite its `head_size` being `C + R`. `q_scaling` is
a live argument, not a fixed 1.0: it was swept over four values in both
phases and moves only that scale (see the `q_scaling` row of *Geometry and
masking scalars*, and *Notes*).

#### MLA head count

`H` = `num_heads` is a plain runtime parameter of the context call, which
runs as MHA (`num_kv_heads = H`, `head_size = nope + R`); the generation
call runs as latent MQA (`num_kv_heads = 1`, `head_size = C + R`) and there
`H` does reach kernel selection (see the JIT paragraph below). Certified:
**`H` = 8, 16, 32 and 128** — 32 is the deepseek-v3-lite tp1 layer
shape, 16 a TP-slice-like count, 8 the tep4 slice of that same checkpoint
(32 attention heads split over 4 tensor-parallel ranks), and 128 the
DeepSeek-R1-0528 layer shape, which an attention-DP rank runs whole because
DP partitions requests rather than heads — all at the same
`C`/`R`/`nope`/`v_head_dim` (512/64/128/128). The first three were run
through the
identical four-case set at `tokens_per_block = 64` — fresh prefill including
a page-crossing sequence, cached-KV no-append context, generation decode over
a page boundary, and a mixed batch served by two phase calls — against the
same fp32 references and the same tolerances, and they land in the same
accuracy envelope: every `H = 8` case sits between 36.8% and 45.9% of the
tolerance allowance, at or below the `H = 32` run of the same case in all
eight comparisons and below the `H = 16` run in all but one (the page-64
mixed batch, 38.5% against 37.9%) — per-case fractions in *Notes*. `H = 8`
runs that whole four-case set at
`tokens_per_block = 32` as well, on the page-32 geometries of *MLA page
size*, so it is certified at both page sizes across all four cases — the
coverage `H = 32` has. Those four cases all pass `q_lora_rank = 1536`, but
`q_lora_rank = 0` at `H = 8` no longer rests on the `H = 32` measurement:
the inertness sweep of *MLA q-LoRA rank* is driven at `H = 8` as well, which
puts it on the `...VarSeqQ8...` decode kernel this head count compiles
instead of on a kernel a tep4 target never runs.

**`H = 128`** runs that same four-case set at `tokens_per_block = 32` only,
on the page-32 geometries of *MLA page size*, and runs it **twice**: once on
the unscaled rope table with `q_scaling = 1.0`, so the head count is the only
thing moving against the `H = 8` and `H = 32` runs of the same cases, and
once at the complete DeepSeek-R1-0528 cell — the checkpoint's YaRN-scaled
rope table plus `q_scaling = 1/mscale² ≈ 0.53366` (see *In-kernel RoPE
arguments* and the `q_scaling` row of *Geometry and masking scalars*), which
is the combination a DeepSeek-R1 layer actually passes. It lands in the same
accuracy envelope as the smaller counts: 37.2-63.3% of the tolerance
allowance on the baseline set, 37.2-60.7% at the R1 cell — per-case fractions
in *Notes*. `H = 128` is **not** certified at `tokens_per_block = 64`, and
none of the four cases was re-run at `q_lora_rank = 0` (they all pass 1536,
the R1 value); the rank's inertness rests on the `H = 32` and `H = 8` sweeps.

`H = 128` is also the **only** head count certified over an fp8-e4m3 latent
pool, where the same four cases run again — baseline cell and R1 cell alike —
against fp8 references at the wider fp8 tolerance (see *MLA over an fp8-e4m3
latent pool*).

Everything scales with `H` exactly as the shape tables below state: `q`/`k`
row width `H * (nope+R)`, context `output` width `H * v_head_dim`, the `v`
split view's row stride `H * (nope + v_head_dim)` elements, generation `q`
width `H * (C+R)`, generation `output` width `H * C`, `q_pe` `[G*P, H, R]`,
and `cu_q_seqlens = arange(G+1) * H * P` (`P` = `predicted_tokens_per_seq`,
1 unless a speculative-decoding step). The paged latent pool is untouched by
the head count (its row width is `C + R`).

The **context** phase JIT-compiles nothing at any certified head count. The
**generation** phase compiles a decode kernel whose name carries a q-tile
that the head count moves at the low end only: `H = 8` takes
`...PagedKvDenseP{32,64}VarSeqQ8Kv128StaticSwapsAbForGen` where `H` = 16, 32
**and 128** all take `...VarSeqQ16...`. So `H = 8` decode is a genuinely
different compiled kernel, while 16, 32 and 128 are separate compile-cache
entries behind one kernel name — each still pays its own ~5.5 s compile (see
*Notes*). Head counts other than 8, 16, 32 and 128 are untested rather than
known-bad, and a new one may pay a decode JIT compile of its own.

Two parts of the MLA surface stay certified at `H = 16` only, because no
other head count was run through them: the
`softmax_stats_tensor` output, and the chunked partial-pass pattern of the
no-append flavor (padding mask, per-pass stats, zero-KV sequences,
downstream merge). Both are independent of the head count in the shape
tables — the stats tensor is `[>= Tc, >= H, 2]` — but neither has a run at
any other head count behind it.

#### MLA page size

`tokens_per_block` is certified at **32 and 64** for MLA. 32 is what a
`KvCacheConfig` at its defaults produces, so it is the page size an MLA
target hits before anyone tunes it; 64 is the tuned value. Like the head
count this is a kernel-selection axis for the generation phase only: it
JIT-compiles a decode kernel whose name carries the page size —
`...PagedKvDenseP32VarSeqQ16Kv128StaticSwapsAbForGen` at 32 versus
`...P64...` at 64 (see *Notes*) — while the context phase triggers no JIT
compile at either size.

Certified at page 32: all three MLA call flavors at `H = 32` (the
deepseek-v3-lite tp1 layer shape), at `H = 8` (its tep4 slice) and at
`H = 128` (the DeepSeek-R1-0528 layer shape, on the baseline rope/scale and
again at the full R1 cell), including
the mixed-batch pairing of two of them at all three counts, plus the
generation flavor also at `H = 16` because that is the one flavor where the
page size changes the compiled kernel and the decode compile cache is keyed
by head count. The page-32 geometry is chosen
for 32-slot pages rather than reused from the page-64 cases — a 64-token
prefill fills a 64-slot page exactly but spans two 32-slot pages, so the
crossing arithmetic differs:

- **fresh prefill**: sequences of 96 (three exact pages — a 64-slot page
  never ends there) and 33 (a one-token spill into a second page), append
  checked page by page;
- **generation decode**: a 64-token history filling two pages exactly, so
  the first decode token opens page 2, alongside a 31-token history whose
  first decode token takes page 0's last slot and whose second opens page
  1 — a decode read range crossing a boundary mid-case, two steps, all
  four head counts;
- **mixed batch**: a 64-token history decoded next to a fresh 33-token
  context sequence, the two phase calls sharing one page-32 offsets table;
- **cached-KV no append**: prefixes of 96 / 31 / 0 reaching KV lengths of
  128 (four exact pages), 40 and 25, with the pages reserved as production
  reserves them and the pool verified bitwise untouched.

Page 32 is also the only size certified over an fp8-e4m3 latent pool, where
all four geometries above run again at `H = 128` (see *MLA over an fp8-e4m3
latent pool*).

Accuracy at page 32 lands on the page-64 magnitudes (per-case fractions in
*Notes*, including the one page-32 case that sets the file's headroom
ceiling — a flavor that never touches the pool, so its number tracks its KV
geometry rather than the page size), the paged latent append holds to the
same gate, and the generation call still writes nothing. Page sizes other
than 32 and 64 are untested for MLA rather than known-bad; the standard
configuration is certified at 32 only.

#### MLA q-LoRA rank

`q_lora_rank` names the rank of the checkpoint's `q_a_proj`: DeepSeek-V3
down-projects the query through a rank-1536 LoRA pair, while a checkpoint
whose config carries `"q_lora_rank": null` (deepseek-v3-lite) has no q-LoRA
at all and projects the query directly. Certified: **1536 and 0**.

The argument is **inert**: no observable of any certified MLA call depends
on it. That matches where the query projection sits — entirely outside this
call, which receives finished `q` rows (absorbed or not) whether or not a
q-LoRA produced them. Measured on sm_100 by sweeping
`1536 → 0 → 1536 → 4096` (the last larger than `C`, so arithmetic that
touched the value would move something) over identical inputs, in all three
MLA call flavors at `tokens_per_block = 32`, on the page-32 geometries of
*MLA page size*, at **two** of the certified head counts — `H = 32` (the
deepseek-v3-lite tp1 layer shape) and `H = 8` (its tep4 slice), two
independent sweeps with their own seeds; `H = 128` and `H = 16` were not
swept. Both swept counts are measured rather than one inferred from the other
because the generation phase compiles a different decode kernel at each
(`...P32VarSeqQ16...` at `H = 32`, `...P32VarSeqQ8...` at `H = 8` — see
*MLA head count*), so an `H = 32` sweep alone is evidence about a kernel a
tep4 target never runs. Not swept: the mixed batch (a pairing of two of
those flavors), `H = 16`, `tokens_per_block = 64` (both sweeps are
page-32), and the no-append flavor's chunked partial-pass pattern with
`softmax_stats_tensor` — the sweep takes its one-shot cached-KV pattern.
Every observable came back **bitwise identical** at both head counts: the
`output` rows, the whole paged latent pool, the in-place-roped `q`/`k` of
the fresh-prefill flavor, and the byte count the op resized `workspace_`
to. The repeated `1536` is the run-to-run determinism control that makes
those comparisons mean something. Kernel selection does not see it either:
a process running only the two sweeps compiled exactly one decode kernel
per head count — one `...VarSeqQ16...` serving the four `H = 32` rank
values and one `...VarSeqQ8...` serving the four `H = 8` ones — where a
real selection axis (the head count) compiles once per value; see *Notes*.

`None` is **not** an accepted value on the MLA path even though the
signature admits it: the C++ unwraps the optional unconditionally and the
call raises `RuntimeError: bad optional access`. Pass the checkpoint's rank,
or `0` when it has none.

**Context phase — fresh prefill** (`attention_input_type=1`, separate
`q`/`k`/`v`, `is_fused_qkv=False`, `latent_cache` given): certified context
sequences of this flavor have no cached prefix, so token `i` of a sequence
sits at absolute position `i`. Per context sequence of length `L`, with
per-token latent row `latent_cache[t] = [ckv_t | k_pe_t]`:

```
# GPT-J (interleaved-pair) RoPE at position i, fp32 math, from the
# duplicated-layout rotary_cos_sin table:
#   rope(x)[2d] = x[2d]*cos_d - x[2d+1]*sin_d
#   rope(x)[2d+1] = x[2d]*sin_d + x[2d+1]*cos_d
q'[i, h] = [ q[i, h, :nope] | rope_i(q[i, h, nope:]) ]   # written back into q
k'[i, h] = [ k[i, h, :nope] | rope_i(k_pe_i) ]           # k_pe broadcast to all
                                                         # heads, written back into k
cache_row(seq, i) = [ ckv_i | rope_i(k_pe_i) ]           # paged latent append
out[i, h] = softmax_causal(q'[i, h] @ K'.T * scale) @ V  # headDim nope+R,
                                                         # headDimV v_head_dim
```

`output[:num_ctx_tokens]` is written. The rope slice of the incoming `k`
(`k[..., nope:]`) is ignored (the kernel fills it from `latent_cache`);
the nope slices of `q`/`k` are left bitwise intact, and `latent_cache` is
read-only. Fusion boundary (context): q_pe/k_pe RoPE, latent-cache append,
and causal FMHA happen inside; the caller owns the projections that build
`q`, `k_nope`, `v`, and `latent_cache`, and the output projection.

**Context phase — no append** (`attention_input_type=1`, separate
`q`/`k`/`v`, `is_fused_qkv=False`, `latent_cache=None`): the in-kernel
RoPE and cache append are skipped entirely — the call is a pure masked
FMHA over caller-supplied K/V, used for context over a cached KV prefix
and for chunked-prefill partial passes. Per context sequence `s` with
`n = host_context_lengths[s]` q rows and `m = sequence_length[s]` K/V rows
(this call's KV range; `m` is independent of `n`):

```
out[i, h] = softmax(q[i, h] @ K[:j_i].T * scale) @ V[:j_i]   # fp32 accum
  j_i = (m - n) + i + 1    # mask_type == 1: causal, bottom-right aligned
                           # (query i sits at absolute position m - n + i)
  j_i = m                  # mask_type == 0: padding (whole KV range)
```

and, when `softmax_stats_tensor` is given, per (q row `t`, head `h`) over
the same masked logit range (natural-log domain, fp32):

```
softmax_stats[t, h] = (max_j logit_j, sum_j exp(logit_j - max))
```

Nothing is rotated, appended, or mutated except `output` (and the stats
tensor): `q` is consumed as-is (production pre-rotates its rope tails
upstream — a sibling op, `trtllm::mla_rope_append_paged_kv_assign_q`,
exists for that step and the latent append), `k` carries rope(k_pe) in its
per-head tails, and `q`, `k`, `v`, and the paged pool were verified
bitwise untouched by the call. This flavor is certified on pure-context
batches (`ns == num_contexts`). Two usage patterns are certified:

- **One-shot cached-KV context** (causal): K/V cover each sequence's full
  `[cached + new]` range (`sequence_length = cached + new >` q length),
  one call computes the final context output.
- **Chunked partial passes**: per chunk loop, K/V cover one chunk slice of
  the cached range per sequence — `sequence_length[s]` = that sequence's
  chunk length, `0` allowed (the sequence sits the loop out; its output
  and stats rows are then **undefined** and must be skipped downstream) —
  with `mask_type=0` and stats emitted; then one final causal pass over
  the new tokens only (`sequence_length == context_lengths`), stats
  emitted. Each pass's output/stats pair is folded into the running
  full-range result by a sibling merge op
  (`trtllm::merge_chunked_attention_for_mla`) under the production
  copy/merge/skip plan — verified end to end: the merged output and stats
  equal single-pass attention over the full `[cached + new]` range.

**Generation phase** (`attention_input_type=2`, `q` = absorbed "fused q",
`is_fused_qkv=True`, `k`/`v` `None`): a pure latent-MQA **read** of the
paged cache. With `P = predicted_tokens_per_seq`, each generation sequence
contributes `P` query rows — one for an ordinary decode step, `P > 1` for a
speculative-decoding step whose `P` rows are that sequence's draft chain.
The rows are **token-major within a sequence**: row `n` is sequence
`g = n // P`'s `t = n % P`-th token, and it sits at absolute position
`L_g - P + t`, where `L_g = sequence_length[num_contexts + g]` is the total
KV length **including all `P` of this step's tokens**. Per row:

```
K = cache rows [0, L_g - P + t]                        # [L_g-P+t+1, C+R]
V = K[:, :C]
out[n, h] = softmax(q[n, h] @ K.T * scale) @ V         # [H, C] per token
```

`output[:G*P]` is written (`G` = generation-sequence count). At `P = 1` the
key range is the whole cache, which is the ordinary decode rule. At `P > 1`
the **within-block mask is causal and bottom-right aligned against the
sequence's own KV length**: draft row `t` sees the cache up to and including
its own position and **not** rows `t+1 .. P-1` of its own block, whose tokens
do not exist yet.

That is measured rather than inferred, and by a readout rather than a
tolerance: with each cached latent row's `ckv` half set to a one-hot vector,
the decode's `V` is that one-hot and the output row becomes the attention
weight vector itself, so the attended key set can be read off column by
column. At `P` 1-4, `H = 128`, page 32, over both pool element types, every
row weighted exactly `[0, L_g - P + t]` and every other column came back
**bitwise zero** — including the 1 to 3 future-sibling columns a full
within-block mask would have weighted at 0.023-0.039 each. `mask_type` does
not change this (see its row in *Geometry and masking scalars*), and on
sm_100 no mask tensor is involved at all: for a linear-tree draft the Python
backend forces `is_spec_decoding_enabled` off there, so
`spec_decoding_packed_mask` and its siblings are `None` and
`predicted_tokens_per_seq` is the only thing producing the mask.

No RoPE is applied and nothing is written to the cache: the generation-phase
preprocessing — the RoPE of `q_pe` at each row's own position, the latent
append of all `P` rows at positions `L_g - P .. L_g - 1`, and the
decode-scheduler-buffer fill — happens **before** this call (a sibling op,
`trtllm::mla_rope_generation`, exists for it).

Where the roped `q_pe` has to **land** is pool-dependent, and the line above
describes only the bf16 case. Over a **bf16** latent pool this call reads its
query from `q`, so the preprocessing must leave the roped `q_pe` in fused q's
tail (`q[..., C:]`) — that is the layout certified here. Over an
**fp8-e4m3** pool this call does not read `q` at all: the query arrives
already quantized in `quant_q_buffer` (see *MLA over an fp8-e4m3 latent
pool*), so whether a producer ever materializes a roped tail in `q` is
outside what this op observes. What it requires on that path is only that
`quant_q_buffer` hold the right bytes.

The `latent_cache` and `q_pe` arguments must be non-None (presence-checked)
but their data is **not consumed** — verified by passing garbage in both
while output, pool, and `q` stayed correct/untouched, at `P` 1 through 4; they
are not mutated either. The caller owns the absorbed-q BMM that fills
`q[..., :C]` and the `v_b_proj` BMM applied to the `[G*P, H*C]` output
afterwards.

#### MLA over an fp8-e4m3 latent pool

`quant_mode=128` switches the **latent pool's** element type to fp8-e4m3
(one byte per element; the pool is `[num_blocks, tokens_per_block, C+R]`
e4m3) while `q`, `k`, `v`, `latent_cache`, `q_pe` and `output` all stay
bf16. Certified at the DeepSeek-R1-0528 layer geometry — `H = 128`,
`tokens_per_block = 32`, `C/R/nope/v = 512/64/128/128`,
`q_lora_rank = 1536`, causal mask, single-layer pool, one call per phase —
for **all three MLA call flavors** (fresh-prefill context, no-append context,
generation) and for the **mixed batch** that pairs two of them, at KV-cache
scaling factor `s = 1.0` (the production value: a
`kv_cache_quant_algo: FP8` checkpoint whose per-layer `k_scale`/`v_scale`
are 1.0). `s` enters as the same
two fp32 `[1]` CUDA tensors the standard configuration uses —
`kv_scale_orig_quant` = `1/s`, `kv_scale_quant_orig` = `s` — and passing
`None` for both is again silently the `s = 1.0` path (bitwise identical
output and pool, verified). `s` = 1.5 and 2.0 are swept beside 1.0 on the
fresh-prefill, generation and no-append flavors; the mixed batch runs
`s = 1.0` only.

Two rope/scale cells are covered, and not uniformly. The **fresh-prefill**
and **generation** flavors run both: the unscaled theta-10000 table at
`q_scaling = 1.0`, and the complete DeepSeek-R1-0528 cell — that checkpoint's
YaRN-scaled `rotary_cos_sin` table together with
`q_scaling = 1/mscale² ≈ 0.53366` (see *In-kernel RoPE arguments*). The
**no-append** flavor, the **mixed batch** and the `quant_mode`-bit checks run
the R1 cell only. Both axes are certified over the bf16 pool as axes of their
own; the fp8 runs pin them **in combination with the pool**, which is what a
DeepSeek-R1 layer actually passes and what separate certifications do not
cover. Both reach the
fp8 path and both are gated: an fp8 R1-cell prefill sits 12.7x outside a
reference built at `q_scaling = 1.0` and 3.7x outside one built from the
unscaled table (the matching reference uses 0.40 of the same band); its
decode sits 4.2x outside the `q_scaling = 1.0` reference and the no-append
flavor 10.8x. The rope table is
pinned far harder by the append than by either output gate: the appended
`k_pe` half is bit-exact against a table-aware mirror while a mirror built
from the unscaled table differs in 2336 of the 6144 e4m3 bytes of a 96-token
request (38.0%, against a gate that allows 6).

**Only the append resembles the standard configuration's fp8 surface.** The
two read paths are different code and different arguments; read them here,
not there.

- **Append** (context call, both halves of the row): each new latent row
  lands as `e4m3(row * kv_scale_orig_quant)` — fp32 multiply, round-to-
  nearest cast. Bit-exact against that mirror at every scale run (1.0, 1.5,
  2.0), for the bitwise-copied `ckv` half **and** the in-kernel-roped `k_pe`
  half alike: e4m3's 3-bit mantissa absorbs the fp32 evaluation-order
  difference that costs the bf16 pool its ~8-in-500 000 one-ulp elements.
  Pages outside the written rows stay bitwise zero, including when a
  request's pages are scattered and out of order.
- **Context FMHA runs in fp8, not bf16 — in *both* context flavors.** The op
  quantizes `q`, `k` and `v` to e4m3 itself and computes the masked FMHA on
  those operands, so **prefill accuracy is affected by cache quantization** —
  not because the context reads the pool back (the `s != 1.0` runs match a
  model built from the caller's own bf16 operands quantized at 1.0, which is
  not what the pool rows hold), but because the op quantizes its inputs.
  This is not inherited from one flavor to the other: it was measured
  separately on the fresh-prefill flavor (`latent_cache` given) and on the
  no-append flavor (`latent_cache=None`), which appends nothing and whose K/V
  arrive **already dequantized** from an fp8 pool — it quantizes them again
  anyway.
  Pinned bitwise on each: with the softmax
  collapsed onto one key, the output row comes back as `e4m3(V_row)`
  exactly (max abs deviation 0, 99.98% of the raw bytes equal — the rest
  signed zeros), where a bf16 path would return `V_row`, 60-63 bf16 ulps
  away. On
  realistic inputs each flavor matches an fp32 reference over e4m3-rounded
  q/k/v within `2^-3 + 2^-4 * |ref|` (fresh prefill 40-49% of it, no-append
  41%); the fresh-prefill run sits 26x (baseline cell) to 49.8x (R1 cell)
  outside the bf16 band around the un-quantized reference — the mixed batch's
  context call 58.1x — while the no-append run sits 2.2x
  outside the *fp8* band around it.
  The quantization uses **scale 1.0** — `kv_scale_orig_quant` is *not*
  applied to `q`/`k`/`v` — while the FMHA nevertheless applies the
  read-side dequantization factors that scale implies: `s^2` on the softmax
  scale and `s` on the output. The two only cancel at `s = 1.0`, so
  **both context flavors are correct at `s = 1.0` alone**. At `s` = 1.5 and
  2.0 the output is what the `s = 1.0` math would give with those two factors
  applied (measured within 0.58x / 0.84x of the same band on the fresh flavor
  and 0.86x on the no-append one), i.e. 21x-48x
  away from the correct result, silently; the no-append peaked readout returns
  exactly `s * e4m3(V_row)`, bit for bit, which is where the output factor is
  read off directly. There is no caller-side knob that
  fixes it: `quant_scale_qkv`, the argument that looks like it would set the
  quantization scale, is inert here — a scratch probe passing `[1/s]` in it
  at `s = 2.0` returned bitwise-identical output.
- **The no-append flavor still touches no pool.** Under `quant_mode=128` as
  under `0`, that call reads and writes nothing but `output`: `q`, `k`, `v`
  and the whole e4m3 pool came back bitwise intact from a call whose pages
  held real cached content. Its operands in the certified run were built the
  way a target builds them — the cached prefix written into the pool as
  `e4m3(row * kv_scale_orig_quant)`, read back as
  `bf16(float(cache_byte) * kv_scale_quant_orig)` and pushed through a
  `kv_b_proj`-shaped matmul, with only the new tokens fresh — so the flavor
  is certified on operands that have been through the pool, not just on
  operands of the right shape.
- **Generation FMHA takes its query from `quant_q_buffer`, and its two
  scales from `mla_bmm1_scale` / `mla_bmm2_scale`.** `q` is not read at all
  on this path (replacing it with garbage left the output bitwise
  identical), and neither kv scale tensor is read either (dropping both is
  bitwise identical at `s = 2.0`, where a read-side dequant would be
  glaring). Mechanically the kernel computes, over the **raw** e4m3 numbers
  in `quant_q_buffer` and in the pool:

  ```
  out[g, h] = softmax2(qq[g, h] @ Kraw.T * mla_bmm1_scale[1])
              @ Kraw[:, :C] * mla_bmm2_scale[0]
  # softmax2 = exp2-domain softmax, so mla_bmm1_scale[1] is the natural-log
  # scale times log2(e); mla_bmm1_scale[0] is never read (zeroing it changes
  # nothing, zeroing [1] flattens the softmax)
  ```

  The caller therefore owns the dequantization. Filling the three buffers
  like this —

  ```
  quant_q_buffer = e4m3(fused_q * kv_scale_orig_quant)        # [G*P, H, C+R]
  x              = 1 / (q_scaling * sqrt(nope + R)) * s**2
  mla_bmm1_scale = [x, x * log2(e)]                             # fp32 [2]
  mla_bmm2_scale = [s]                                          # fp32 [1]
  ```

  — makes the call compute exactly latent MQA over the e4m3 round trip of
  both operands (`s^2` undoing the `1/s` on the query and on `K`, `s` the
  one on `V = K[:, :C]`), at the plain softmax scale:

  ```
  q_hat = e4m3(q * 1/s) * s        K_hat = pool_row * s        V_hat = K_hat[:, :C]
  out[n, h] = softmax(q_hat[n, h] @ K_hat.T / (q_scaling*sqrt(nope+R))) @ V_hat
  ```

  with `K_hat` cut to row `n`'s own key range when `predicted_tokens_per_seq`
  is above 1 (see *Semantics*), and the two bmm scales unchanged by it — they
  are per-batch scalars, so an fp8 MTP decode needs no per-draft-row
  rescaling. This whole recipe was re-run at `P` = 1, 2, 3 and 4.

  and that holds at **every** scale certified — 1.0, 1.5 and 2.0 all matched
  this reference within 18% of the fp8 band, the non-power-of-two scale
  included. Leaving `s` out of the two bmm
  scales — the mistake the standard configuration's "dequantized on read"
  wording invites — lands 2.9x (`s = 1.5`) and 3.6x (`s = 2.0`) outside it.
  All three buffers are presence-checked: `None` raises
  `Assertion failed: quant_q_buf is nullptr. (attentionOp.cpp:1050)`,
  `bmm1_scale is nullptr. (:1051)`, `bmm2_scale is nullptr. (:1052)`.
  Their *contents* are unchecked, and `quant_q_buffer` is read through a raw
  pointer (an identical `uint8` view of the same bytes behaves identically).
  The generation call still writes nothing: the pool comes back bitwise
  unchanged.

  That block is a recipe for **this** op's caller — the certified runs passed
  exactly those bytes, built in plain torch — and not a description of how a
  producer arrives at them. In particular `quant_q_buffer` is written in terms
  of the finished bf16 fused q this call would have read on the bf16 path;
  a producer may instead assemble it from the absorbed `q[..., :C]` plus a
  freshly roped `q_pe`, and never materialize a roped tail in `q` at all
  (which is what the sibling preprocessing op does here). Both routes have to
  land on the same bytes; only the bytes are this op's business.

**Quantization bits outside the KV-cache group ride along unread.** A target
derives `quant_mode` from its checkpoint's quant config and will not get a
bare `128`: an nvfp4 checkpoint served with an fp8 KV cache produces `1152`
(`FP8_KV_CACHE | FP8_1x128_128x128`), and fp8-QDQ weights add `FP8_QDQ` for
`384`. Both were run against a bare `128` on all three MLA flavors at the R1
cell and are **bit-identical** — output and pool alike. Nothing validates the
extra bits in either direction; this is a measurement, not a guarantee about
bits that were not tried (`64` = INT8 KV and `8192` = NVFP4 KV were not tried
on this op).

The generation flavor is additionally certified over this pool at
`predicted_tokens_per_seq` = 1, 2, 3 and 4 — one speculative-decoding step per
call — including the mixed batch that pairs it with a context call. See the
`predicted_tokens_per_seq` and `mask_type` rows of *Geometry and masking
scalars* and the generation block of *Semantics*; the context flavors run at
1 only.

Not certified over the fp8 latent pool: the no-append flavor's **chunked
partial-pass pattern** (`mask_type=0`, `softmax_stats_tensor`, zero-KV
sequences — only its one-shot cached-KV pattern is certified),
`tokens_per_block = 64`, head counts other than 128, multi-layer latent
pools, `predicted_tokens_per_seq` above 4, and `s != 1.0` in either context
flavor (measured-wrong, see above, rather than untested).

## Signature

```python
def thop_attention(
    q, k, v, output, output_sf, workspace_,
    sequence_length, host_past_key_value_lengths, host_total_kv_lens,
    context_lengths, host_context_lengths, host_request_types,
    max_context_q_len_override,
    kv_cache_block_offsets, host_kv_cache_pool_pointers,
    host_kv_cache_pool_mapping, cache_indirection,
    kv_scale_orig_quant, kv_scale_quant_orig, out_scale,
    rotary_inv_freq, rotary_cos_sin, latent_cache, q_pe,
    block_ids_per_seq, attention_sinks,
    is_fused_qkv, update_kv_cache, predicted_tokens_per_seq,
    local_layer_idx, num_heads, num_kv_heads, head_size, tokens_per_block,
    max_num_requests, max_context_length, max_seq_len,
    attention_window_size, beam_width, mask_type, quant_mode, q_scaling,
    position_embedding_type, rope_dim, rope_base, rope_scale_type,
    rope_scale, rope_short_m_scale, rope_long_m_scale, rope_max_positions,
    rope_original_max_positions, use_paged_context_fmha,
    attention_input_type, is_mla_enable, chunked_prefill_buffer_batch_size,
    q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
    v_head_dim, rope_append, mrope_rotary_cos_sin, mrope_position_deltas,
    helix_position_offsets, helix_is_inactive_rank, attention_chunk_size,
    softmax_stats_tensor, is_spec_decoding_enabled, use_spec_decoding,
    is_spec_dec_tree, spec_decoding_generation_lengths,
    spec_decoding_position_offsets_for_cpp, spec_decoding_packed_mask,
    spec_decoding_bl_tree_mask_offset, spec_decoding_bl_tree_mask,
    spec_bl_tree_first_sparse_mask_offset_kv,
    sparse_kv_indices, sparse_kv_offsets, sparse_attn_indices,
    sparse_attn_offsets, sparse_attn_indices_block_size,
    # keyword tail, defaults mirror the binding:
    num_sparse_topk=None, sparse_attn_kv_lens=None,
    skip_softmax_threshold_scale_factor_prefill=None,
    skip_softmax_threshold_scale_factor_decode=None, skip_softmax_stat=None,
    cu_q_seqlens=None, cu_kv_seqlens=None, fmha_scheduler_counter=None,
    mla_bmm1_scale=None, mla_bmm2_scale=None, quant_q_buffer=None,
    flash_mla_tile_scheduler_metadata=None, flash_mla_num_splits=None,
    sage_attn_num_elts_per_blk_q=0, sage_attn_num_elts_per_blk_k=0,
    sage_attn_num_elts_per_blk_v=0, sage_attn_qk_int8=False,
    num_contexts=0, num_ctx_tokens=0, trtllm_gen_jit_warmup=False,
    aux_kv_cache_pool_ptr=None, is_cross=False, cross_kv=None,
    relative_attention_bias=None, relative_attention_max_distance=0,
    spec_decoding_target_max_draft_tokens=None, quant_scale_qkv=None,
    dsv4_inv_rope_cos_sin_cache=None, enable_dsv4_epilogue_fusion=False,
) -> None
```

Exact per-argument types are in the wrapper; the binding is keyword-callable
and the wrapper forwards every argument by keyword.

With `Hq` = `num_heads`, `Hkv` = `num_kv_heads`, `D` = `head_size`,
`ns` = batch size, `T` = total new tokens in the batch
(`num_ctx_tokens + (ns - num_contexts) * predicted_tokens_per_seq`). `T` is
the standard configuration's `q`/`output` row count; an **MLA** batch is
served by one call per phase, so its context call has `num_ctx_tokens` rows
and its generation call `(ns - num_contexts) * predicted_tokens_per_seq`.
Certified MLA dims (DeepSeek-V3 / R1 geometry): `C = kv_lora_rank = 512`,
`R = qk_rope_head_dim = 64`, `nope = qk_nope_head_dim = 128`,
`v_head_dim = 128`, at `H` = 8, 16, 32 **or** 128 query heads — a
plain runtime parameter for the context call as in the standard
configuration, and a decode-kernel selection axis for the generation call
(see *MLA head count*); every MLA shape below is written in terms of `H`.

### Data tensors — standard configuration

| Argument | Shape / value | Dtype | Device |
|---|---|---|---|
| `q` | `[T, (Hq + 2*Hkv) * D]`, packed per-token `q \| k \| v`, contiguous (both pool dtypes: activations stay bf16) | bf16 | CUDA |
| `k`, `v` | `None` — K/V ride inside packed `q` (`is_fused_qkv=True`; separate-KV mode not certified for this configuration) | — | — |
| `output` | `[T, Hq * D]`, contiguous; rows `[:T]` overwritten in place (both pool dtypes) | bf16 (same as `q`) | CUDA |
| `kv_scale_orig_quant` | `quant_mode=0`: `None`; `quant_mode=128`: `[1]` holding `1/s` — **required for any `s != 1.0`** (`None` is silently read as `1.0`, never rejected) | fp32 | CUDA |
| `kv_scale_quant_orig` | `quant_mode=0`: `None`; `quant_mode=128`: `[1]` holding `s` — **required for any `s != 1.0`** (same silent `1.0` fallback) | fp32 | CUDA |
| `attention_sinks` | `None` (no sink), or exactly `Hq` values, **contiguous** — one sink logit per query head (see *Attention sinks*; certified on the bf16 pool with `mask_type=1`). Shape beyond the element count is ignored (`[Hq]`, `[1, Hq]`, `[Hq, 1]` all behave identically) | fp32 | CUDA |
| `latent_cache`, `q_pe` | `None` | — | — |
| `cu_q_seqlens`, `cu_kv_seqlens`, `fmha_scheduler_counter` | `None` (the op fills internal ones when absent) | — | — |

### Data tensors — MLA context call, fresh prefill (`latent_cache` given)

`Tc` = `num_ctx_tokens` (this call's `q` rows), `Hq == Hkv == H`,
`head_size = nope + R` (192 certified).

| Argument | Shape / value | Dtype | Device |
|---|---|---|---|
| `q` | `[Tc, H * (nope+R)]`, contiguous; rope slice of each head **overwritten in place** with rope(q_pe) | bf16 | CUDA |
| `k` | `[Tc, H * (nope+R)]`, contiguous; incoming rope slice ignored (may be uninitialized) and **overwritten in place** with rope(k_pe); nope slice consumed as K | bf16 | CUDA |
| `v` | `[Tc, H * v_head_dim]` row view whose **row stride is `H * (nope + v_head_dim)` elements** — the `[.., H*nope:]` split of a packed `[Tc, H*(nope+v)]` kv-projection buffer (certified); the context FMHA hard-codes this stride, so a fully contiguous `[Tc, H*v_head_dim]` tensor is misread (see *Notes*) | bf16 | CUDA |
| `output` | `[Tc, H * v_head_dim]`, contiguous; rows `[:Tc]` overwritten | bf16 | CUDA |
| `latent_cache` | `[Tc, C + R]` per-token `[ckv \| k_pe]`, contiguous; read-only | bf16 | CUDA |
| `q_pe` | `None` | — | — |
| `kv_scale_orig_quant`, `kv_scale_quant_orig` | `quant_mode=0`: `None`; `quant_mode=128`: `[1]` holding `1/s` and `s`. Only the append consumes `1/s`; the context FMHA applies `s^2`/`s` to a computation it quantized at 1.0, so **`s = 1.0` (or `None`) is the only correct value here** — see *MLA over an fp8-e4m3 latent pool* | fp32 | CUDA |
| `mla_bmm1_scale`, `mla_bmm2_scale`, `quant_q_buffer` | `None` (the context call neither checks nor reads them at either `quant_mode`) | — | — |
| `softmax_stats_tensor` | `None` (stats certified only for the no-append flavor) | — | — |
| `cu_q_seqlens`, `cu_kv_seqlens`, `fmha_scheduler_counter` | `None` (the op fills internal ones when absent) | — | — |

### Data tensors — MLA context call, no append (`latent_cache=None`)

`Tc` = `num_ctx_tokens` (this call's `q` rows); `Tkv` = sum of
`sequence_length[s]` over the context sequences = total K/V rows, the
per-sequence KV ranges concatenated in batch order.

| Argument | Shape / value | Dtype | Device |
|---|---|---|---|
| `q` | `[Tc, H * (nope+R)]`, contiguous; consumed as-is (rope tails prepared upstream) and **not mutated** | bf16 | CUDA |
| `k` | `[Tkv, H * (nope+R)]`, contiguous, read-only; per-head tails hold rope(k_pe) broadcast to all heads (caller-built) | bf16 | CUDA |
| `v` | `[Tkv, H * v_head_dim]` row view, read-only; **row stride `H * (nope + v_head_dim)` elements required** (same hard-coded stride as the fresh flavor — certified as the split of packed `[Tkv, H*(nope+v)]` buffers; contiguous rows are misread) | bf16 | CUDA |
| `output` | `[Tc, H * v_head_dim]`, contiguous; rows `[:Tc]` overwritten (rows of zero-KV sequences: undefined) | bf16 | CUDA |
| `latent_cache` | `None` — selects this flavor (skips in-kernel RoPE + append; the paged pool is neither read nor written, at either `quant_mode`) | — | — |
| `q_pe` | `None` | — | — |
| `kv_scale_orig_quant`, `kv_scale_quant_orig` | `quant_mode=0`: `None`; `quant_mode=128`: `[1]` holding `1/s` and `s`. This flavor appends nothing, so `1/s` reaches nothing at all, while the FMHA still applies `s^2`/`s` to a computation it quantized at 1.0 — **`s = 1.0` (or `None`) is the only correct value here**, exactly as for the fresh-prefill flavor. See *MLA over an fp8-e4m3 latent pool* | fp32 | CUDA |
| `softmax_stats_tensor` | `None`, or `[>= Tc, >= H, 2]` contiguous, rows `[:Tc]` overwritten per (token, head) with `(max, sum)` as in *Semantics* (rows of zero-KV sequences: undefined). Presence-checked by the op: fp32, dim 0 `>=` the call's q rows, dim 1 `>= num_heads`, dim 2 `== 2`. Certified on the bf16 pool only | fp32 | CUDA |
| `mla_bmm1_scale`, `mla_bmm2_scale`, `quant_q_buffer` | `None` (this flavor neither checks nor reads them at either `quant_mode`) | — | — |
| `cu_q_seqlens`, `cu_kv_seqlens`, `fmha_scheduler_counter` | `None` (the op fills internal ones when absent) | — | — |

### Data tensors — MLA generation call

`G` = generation-sequence count, `P` = `predicted_tokens_per_seq` (query rows
per generation sequence: 1 for an ordinary decode step, `P > 1` for a
speculative-decoding step — 1, 2, 3 and 4 certified), `head_size = C + R`
(576 certified), `Hkv = 1`. Every per-row tensor is `G*P` rows tall and
**token-major within a sequence**: row `n` is sequence `n // P`'s
`n % P`-th token.

| Argument | Shape / value | Dtype | Device |
|---|---|---|---|
| `q` | `[G*P, H * (C+R)]` fused q (absorbed q_nope in `[..., :C]`, rope(q_pe) in `[..., C:]`, both prepared before the call — at `P > 1` each row must have been roped at **its own** absolute position `L_g - P + n % P`; this call applies no rope and cannot tell), contiguous; not mutated. Under `quant_mode=128` it is **not read either** — the query comes from `quant_q_buffer` below — but it is still the tensor whose shape and dtype the call is configured around | bf16 | CUDA |
| `k`, `v` | `None` (`is_fused_qkv=True`) | — | — |
| `output` | `[G*P, H * C]`, contiguous; rows `[:G*P]` overwritten (per-head latent output — `v_b_proj` still to apply). A taller buffer's rows past `G*P` are left bitwise untouched | bf16 | CUDA |
| `latent_cache` | non-None required; **data not consumed and not mutated** (this step's rows must already be in the pool). Certified passing `[G*P, C + R]`, the production shape; the row count is not checked (`[G, …]` and `[1, …]` were also accepted, bitwise identically) | bf16 | CUDA |
| `q_pe` | non-None required; **data not consumed and not mutated**. Certified passing `[G*P, H, R]`, same unchecked-row-count caveat | bf16 | CUDA |
| `cu_q_seqlens` | `[G+1]` **required** (presence only): `arange(G+1) * H * P` (q-row prefix sums — this is what the sibling preprocessing op writes at every `P`) | int32 | CUDA |
| `cu_kv_seqlens` | `[G+1]`, `[0, cumsum(L_g)]` over generation sequences — **neither checked nor read on this path** | int32 | CUDA |
| `fmha_scheduler_counter` | `[1]` **required** (presence only), zeroed before the call | uint32 | CUDA |
| `quant_q_buffer` | `quant_mode=0`: `None`; `quant_mode=128`: `[G*P, H, C+R]` **required and read as the query** — `e4m3(fused_q * kv_scale_orig_quant)`, contiguous; `q` itself is then unread. Read through a raw pointer, so an equal-bytes `uint8` view behaves identically — and so a buffer shorter than `G*P` rows is read past its end rather than rejected | e4m3 | CUDA |
| `mla_bmm1_scale` | `quant_mode=0`: `None`; `quant_mode=128`: `[2]` **required**, `[x, x*log2(e)]` with `x = s^2 / (q_scaling * sqrt(nope+R))`. Element `[1]` is the one read. Per batch, **not** per `P` — the same pair serves every draft row | fp32 | CUDA |
| `mla_bmm2_scale` | `quant_mode=0`: `None`; `quant_mode=128`: `[1]` **required**, `[s]` — multiplies the output | fp32 | CUDA |
| `kv_scale_orig_quant`, `kv_scale_quant_orig` | **not read on this path at either `quant_mode`** (bitwise-identical output with both `None` at `s = 2.0`); pass them anyway if the same tensors serve the context call | fp32 | CUDA |

`None` fails for two of the three — `cu_q_seqlens` (`seqQOffset is nullptr`,
`attentionOp.cpp:1045`) and `fmha_scheduler_counter` (`fmha_tile_counter is
nullptr`, `:1047`) — and both are presence-checked only, their contents
bitwise inert here. `cu_kv_seqlens` is accepted as `None`, and `None`,
all-zeros, non-monotonic and 1000x-inflated values all returned bitwise
identical output at `G = 4` and `G = 16` with pairwise-distinct `L_g`: the
paged decode kernel takes each sequence's KV length from `sequence_length`
(a one-token change there does move the output, from 0.016 to 0.499 max abs
error). Fill all three as the sibling generation-preprocessing op does —
that is the production shape and the certified one.

**The scheduler buffers stay inert at `P > 1`** — the taller query block is
exactly the change that could have made them start mattering, so all of it
was re-measured at `P = 3`, `G = 2`, over the fp8 R1 cell. `cu_q_seqlens`
zeroed, left in the `P = 1` form `i*H`, given `i*P` without the head factor,
inflated 1000x, and made non-monotonic all returned **bitwise-identical**
output; so did `cu_kv_seqlens` zeroed, inflated and non-monotonic. Pass the
production values anyway: they are what the sibling op writes, and nothing
here promises the next version reads them the same way.

### Shared data tensors

| Argument | Shape / value | Dtype | Device |
|---|---|---|---|
| `output_sf` | `None` (NVFP4-output path not certified) | — | — |
| `workspace_` | persistent scratch tensor, any length (0 ok); the op grows it **in place** via `resize_()` when too small — 33 MB to 109 MB across the certified shapes. The MLA context requirement scales with tokens x heads and still sets the peak: 113 924 096 B for an `H = 128` context call over 545 tokens, against 52 579 072 B for the same call over 129 tokens and 100 024 320 B for the standard-configuration head-geometry sweep. The MLA **generation** call is well below that and, measured, does not grow with `predicted_tokens_per_seq`: 39 100 416 B at `H = 128`, page 32 for every `(G, P, L)` tried — `G` 2 and 4, `P` 1 and 4, `L` 50 and 500 — so a `P`-times-taller query block moves nothing here. Pass a plain resizable tensor, not a view; reuse it across calls to avoid re-allocation | int8 | CUDA |

### Batch state (the "prepared metadata" of this op — all int32)

| Argument | Shape | Content | Device |
|---|---|---|---|
| `sequence_length` | `[ns]` | per-sequence KV length **of this call**: total `cached + new` everywhere except an MLA chunked partial pass, where a context row holds that sequence's chunk length instead (0 = sits this loop out). For a generation row at `predicted_tokens_per_seq = P` the `new` part is all `P` of this step's tokens, and this tensor — not its host twin — is what the MLA generation call takes its per-sequence key range from (shifting it by one token moved every draft row's attended set by one, read out directly) | CUDA |
| `host_past_key_value_lengths` | `[ns]` | same values as `sequence_length`. On the MLA generation path the per-sequence values are otherwise inert, but the vector must not be **all** zero — see *Preconditions* | CPU |
| `host_total_kv_lens` | `[2]` | `[0]` = sum of `sequence_length` over context sequences (= the K/V row count for a no-append MLA context call), `[1]` = sum over generation sequences | CPU |
| `context_lengths` | `[ns]` | context sequences: this call's new-token (q-row) count — equals the prompt length only when nothing is cached; generation sequences: the original prompt length. On a context row it is load-bearing and unchecked (see *Preconditions*); on a generation row it is inert under a sliding window (0, 1 and the prompt length gave **bitwise identical** output) and inert on the MLA generation path at `predicted_tokens_per_seq` above 1 as well (0 and the full KV length both bitwise identical at `P = 3`) | CUDA |
| `host_context_lengths` | `[ns]` | same values as `context_lengths` | CPU |
| `host_request_types` | `[ns]` | 0 = context, 1 = generation; all 0s must precede all 1s | CPU |
| `num_contexts` (int) | — | number of context sequences. Binding default 0 is only correct for generation-only batches — always pass the true count | — |
| `num_ctx_tokens` (int) | — | sum of new-token counts over context sequences; same caveat | — |
| `max_context_q_len_override` | — | `None` (encoder CUDA-graph override; not certified) | — |

CPU tensors may be pageable or pinned (pinning is a perf optimization).
For a mixed MLA batch, the two per-phase calls share these full-batch
(`ns`-sized) tensors and the same `num_contexts`/`num_ctx_tokens`; only
`q`/`output` and the phase-specific arguments differ between the calls.
The generation call indexes sequences starting at `num_contexts`.

### Paged KV cache addressing

The cache is a caller-owned pool addressed in **slab** units: one slab =
one layer's K (or V) block, `tokens_per_block * Hkv * D` elements
(standard) or `tokens_per_block * (C + R)` (MLA). A pool page holds the
slabs of every layer sharing the pool, layers side by side. Certified:
one pool, holding 1 layer (both configurations) or 4 layers (standard
configuration, bf16 and fp8-e4m3 pools both; multi-layer MLA pools not
certified). The certified multi-layer state — pool, mapping, block
offsets — was produced by a real 4-layer `KVCacheManager` (one per pool
dtype) and consumed unmodified, so the shapes below are exactly what a
multi-layer manager emits.

Standard configuration (kv_factor 2, HND), `L` layers in the pool: pool
tensor `[num_blocks, L, 2, Hkv, tokens_per_block, D]` CUDA contiguous —
element type bf16 (`quant_mode=0`) or fp8-e4m3 (`quant_mode=128`) —
page `b` holds layer `l`'s K slab at
`[b, l, 0]` and its V slab at `[b, l, 1]` (single layer is the `L = 1`
case, `[num_blocks, 2, Hkv, tokens_per_block, D]`). The op sizes slabs
from `quant_mode`, never from the tensor (it sees only a raw pointer): a
pool whose dtype disagrees with `quant_mode` is silently corrupted.

MLA (kv_factor 1, single layer): pool tensor
`[num_blocks, tokens_per_block, C + R]` CUDA contiguous — element type bf16
(`quant_mode=0`) or fp8-e4m3 (`quant_mode=128`) — one latent row per token,
no separate K/V slabs. The slab width is `tokens_per_block * (C + R)`
**elements** regardless of the per-call `num_kv_heads`/`head_size` (context
passes `H`/192, generation `1`/576; both address the same pool), so the slab
is that many bytes under `quant_mode=128` and twice that under `quant_mode=0`
— the op derives the element width from `quant_mode` alone. Certified at
both element types with the requests' pages scattered out of order, every
page outside the reserved sets left bitwise zero. How many *bytes per token*
a KV-cache manager reserves for such a pool is the manager's business, not
this op's: the op is handed a base pointer plus slab offsets and addresses
exactly the geometry above.

| Argument | Shape / value | Dtype | Device |
|---|---|---|---|
| `kv_cache_block_offsets` | `[1, max_num_requests, 2, max_blocks_per_seq]`; row `[0, s, 0, j]` = K-slab offset of sequence `s`'s `j`-th page, `[0, s, 1, j]` = V-slab offset. Offsets count slabs and are **layer-agnostic** — every layer of the pool shares one offsets row per sequence; layer selection comes from the pool mapping, not the offsets. Standard `L`-layer pool → page `p` has K `p * 2L`, V `p * 2L + 1` (single layer: K `2p`, V `2p + 1`); MLA pool → raw block id `p` in **both** rows. Only the first `ceil(kv_s / tokens_per_block)` entries per row are read | int32 | CUDA |
| `host_kv_cache_pool_pointers` | `[1, 2]`: `[0, 0]` = pool base address (`pool.data_ptr()`), `[0, 1]` = secondary-pool address (0 = none) | int64 | CPU |
| `host_kv_cache_pool_mapping` | `[num_layers, 2]`, one row per layer: row `local_layer_idx` = (pool index, layer index within pool). The row's **layer column drives the pool-base shift** — the call addresses slabs starting `layer_in_pool * kv_factor` slabs from the pool base (see *Notes*). Certified: the identity rows a real manager produces — `[[0, 0] .. [0, 3]]` for the 4-layer standard pool; `[1, 2]` zeros for single-layer pools | int32 | CPU |
| `local_layer_idx` (int) | row of `host_kv_cache_pool_mapping` for this layer (production: the layer's index among the rank's local layers); 0-3 certified (standard), 0 (MLA) | — | — |
| `tokens_per_block` (int) | pool page size; must be a power of two; 32 certified (standard), 32 and 64 certified (MLA — see *MLA page size*) | — | — |
| `update_kv_cache` (bool) | `True` (also for the MLA generation and no-append context calls, which nevertheless write nothing) | — | — |
| `cache_indirection` | `None` (beam search only) | — | — |
| `block_ids_per_seq` | `None` (FlashMLA path only) | — | — |

A sequence's page set is shared by every layer of the pool: the per-layer
calls of one batch pass identical offsets and pool pointers and differ
only in `local_layer_idx`. Multiple pools (`num_pools > 1`: extra
offsets/pointer rows, mapping rows with pool index > 0) are not certified.

### Paged KV cache addressing under a sliding window

`attention_window_size` changes **nothing** about addressing: token `t` of a
sequence always lives at entry `t // tokens_per_block` of that sequence's
offsets row, slot `t % tokens_per_block`, and the append writes there. Every
memory saving is the caller's page mapping. What the op requires:

- The offsets row is indexed by **absolute** page index, so it must have at
  least `ceil(total_kv / tokens_per_block)` entries — that count keeps
  growing with the sequence, window or no window, and bounds
  `kv_cache_block_offsets.shape[3]`.
- Only entries holding at least one **in-window** token need to point at
  valid live pages. An entry whose tokens are all older than
  `total_kv - W` is never read: pointing it at a page filled with garbage
  left a decode's output **bitwise identical** (checked entry by entry over
  a 7-page sequence with `W = 128`; the first entry holding an in-window
  token does change the output). Aged-out pages may therefore be recycled,
  and stale ids may stay in the row — as long as they remain in-pool slab
  offsets (out-of-pool ids were not tried and must not be).
- The same holds for a **context** call on the paged-context path, with the
  in-window range measured from that call's rows: query row `i` of a
  context sequence reads keys `[max(0, p - W + 1), p]` at
  `p = kv_s - l_s + i`, so the call's read span is
  `[max(0, kv_s - l_s + 1 - W), kv_s - 1]`. Measured entry by entry over an
  8-page sequence (`W = 128`, 200 cached tokens, 40 new): every entry
  holding an in-window **cached** token changed the output (318x outside
  the band with a decoy there), while entries below the window and the one
  entry holding only this call's own new tokens left it **bitwise
  identical** — that last one because the append lands wherever the offsets
  point and the FMHA reads it straight back. The ring bound that follows
  from that span (not separately certified) is
  `P * tokens_per_block >= W + l_s - 1` for a cached-prefix context call,
  on top of the decode bound below.
- A **bounded ring** is the resulting production shape: give the sequence
  `P` physical pages and map absolute page index `j` to `ring[j % P]`. The
  op's absolute-position append then lands token `t` at physical slot
  `t mod (P * tokens_per_block)`, so the pool holds a moving window of the
  sequence and genuinely wraps, overwriting exactly the slot of the token
  that has just aged out. Requires
  `P * tokens_per_block >= attention_window_size`; certified at
  `P * tokens_per_block` of 128 (= `W`) and 192 over 250 wrapping decode
  steps. One page short (96 < 128) is **silently wrong** — no exception,
  output far off the reference.
- The same call's new tokens must map to **distinct** physical slots:
  `P * tokens_per_block >= that call's new-token count`. A prefill longer
  than the ring makes several tokens of one append target one slot; those
  writes race (observed: two identical runs left different pool bytes,
  torn rows). So a sliding-window sequence still needs
  `ceil(prompt_len / tokens_per_block)` distinct pages for its prefill call
  — the window shrinks the steady-state decode footprint, not the prefill's.
- The append remains surgical: one decode step changes exactly one
  `(K, V)` row pair in the whole pool, the evicted slot's, and touches
  nothing else. No compaction, zeroing, or relocation ever happens.

### Geometry and masking scalars

| Argument | Standard (certified) | MLA context | MLA generation |
|---|---|---|---|
| `is_fused_qkv` | `True` (packed QKV in `q`) | `False` | `True` |
| `attention_input_type` | 0 = mixed (either subset may be empty; 1/2 not certified here) | 1 = context_only | 2 = generation_only |
| `num_heads`, `num_kv_heads`, `head_size` | must match packed `q` and pool. Head counts are a free axis subject to `num_heads % num_kv_heads == 0`; `head_size` is enumerated by the available FMHA kernels (see *Head geometry*). Certified on the bf16 pool: 32/8/128, 32/4/128, 64/8/64, 16/1/128, 28/4/128, 12/3/128, 8/2/128, 8/2/256, 4/4/64; on the fp8-e4m3 pool: 32/8/128 | `H`/`H`/192 (`H`/`H`/`nope+R`); 8/8/192, 16/16/192, 32/32/192 and 128/128/192 certified on the bf16 latent pool, 128/128/192 only on the fp8-e4m3 one | `H`/1/576 (`H`/1/`C+R`); 8/1/576, 16/1/576, 32/1/576 and 128/1/576 certified on the bf16 latent pool, 128/1/576 only on the fp8-e4m3 one |
| `mask_type` | 1 = causal, 0 = padding — both certified on the bf16 pool (padding on context-only batches); fp8 pool certified with 1 only | 1 = causal (bottom-right aligned when KV > q); 0 = padding certified for the no-append chunked partial passes, on the bf16 latent pool only | **Inert — this path takes its mask from `predicted_tokens_per_seq` alone.** Pass 1. At `P = 1` a decode row attends to all `L_g` cached tokens; at `P > 1` draft row `t` attends to `[0, L_g - P + t]`, bottom-right-aligned causal within the block. `mask_type = 0` returns **bitwise-identical** output at `P` = 2 and 4 (fp8, `H = 128`, page 32) — the plausible guess that padding drops the within-block mask and lets a draft row see its later siblings is wrong |
| `q_lora_rank` | `None` | the checkpoint's `q_a_proj` rank — `1536` and `0` (no q-LoRA) certified and **inert**: no MLA path reads it (see *MLA q-LoRA rank*). `None` raises here, unlike in the standard configuration | same value as context |
| `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim` | all `None` | 512 / 128 / 64 certified | same values as context |
| `v_head_dim` | `None` | 128 (context V/output head width) | 512 (= `C`, latent output width) |
| `rope_append` | `None` | `True` | `True` (`False` widens the output to `C+R` per source — not certified) |
| `predicted_tokens_per_seq` | 1 | 1 | **1, 2, 3 and 4 certified** (`P`): the number of query rows each generation sequence contributes. 1 is an ordinary decode step; `P > 1` is one speculative-decoding step verifying a `P`-token draft chain, and `P` is then the *only* thing that produces the within-block mask (see `mask_type`). One scalar for the whole batch — a ragged per-sequence draft length cannot be expressed here. Certified over the fp8-e4m3 latent pool at the complete DeepSeek-R1-0528 cell and over a bf16 latent pool, both at `H = 128`, page 32; above 4, and at other head counts or page sizes, untested. `P` also reaches decode-kernel selection as the kernel's `maxSeqLenQ` — see *Notes* |
| `q_scaling` | softmax scale `1 / (q_scaling * sqrt(head_size))`; 1.0 certified | free positive scalar; scale uses `sqrt(nope+R)`. Certified 1.0, 0.53366 (`= 1/mscale²`, DeepSeek-R1's YaRN temperature — see *In-kernel RoPE arguments*), 2.0 and 0.25 on the bf16 latent pool at `H = 128`; 1.0 and 0.53366 on the fp8-e4m3 one, in both context flavors | same values; the scale still uses `sqrt(nope+R)`, **not** `sqrt(head_size)`. Over an fp8 pool it reaches the kernel only through the caller's `mla_bmm1_scale` (see *MLA over an fp8-e4m3 latent pool*) |
| `quant_mode` | 0 = bf16 pool; 128 = fp8-e4m3 pool (see *Standard configuration over an fp8-e4m3 KV pool*; other bits not certified) | 0, or 128 for an fp8-e4m3 latent pool. `1152` (`\| FP8_1x128_128x128`) and `384` (`\| FP8_QDQ`) are bit-identical to `128` here — only the KV-cache bit is read (see *MLA over an fp8-e4m3 latent pool*) | same value as context |
| `beam_width` | 1 | 1 | 1 |
| `max_num_requests` | `>= ns`; equals `kv_cache_block_offsets.shape[1]` | same | same |
| `max_context_length` | host launch/workspace bound `>=` max per-sequence q-row count of the call (certified `= max_seq_len`) | same | same |
| `max_seq_len` | `>=` max `sequence_length`; per-sequence page capacity bound. Does not select what a query attends to: with a window active, decodes at `max_seq_len` of `W`, `total + 1` and `4 * (total + 1)` returned **bitwise identical** output (it does steer kernel selection — see *Notes*) | same | same |
| `attention_window_size` | sliding-window token count `W`; `>= max_seq_len` = no window. Smaller = each query attends only the newest `W` keys (see *Sliding window*); 128 / 100 / 33 certified on the bf16 pool with `mask_type=1`. Per-call scalar: layers sharing a pool may differ in it | same (no window certified) | same (no window certified) |
| `use_paged_context_fmha` | context execution path; both values certified. `False` = context FMHA over the packed `q` rows, so every context sequence must start empty (`sequence_length == host_context_lengths`). `True` = context FMHA over the paged pool, which is what a context call over a cached prefix requires (`sequence_length > host_context_lengths`) — see *Paged-context FMHA*. With nothing cached the two are bitwise identical | `False` (MLA context always takes separate q/k/v) | `False` |
| `chunked_prefill_buffer_batch_size` | 1 | 1 | 1 |
| `attention_chunk_size` | `None` | `None` | `None` |
| `is_cross` / `cross_kv` | `False` / `None` | same | same |
| `trtllm_gen_jit_warmup` | `False` | `False` | `False` |

### In-kernel RoPE arguments

Standard configuration: RoPE disabled — `position_embedding_type=0`
(learned-absolute; no rotation applied), `rotary_inv_freq=None`,
`rotary_cos_sin=None`, `rope_dim=0`, `rope_base=10000.0`,
`rope_scale_type=0`, `rope_scale=1.0`, `rope_short_m_scale=1.0`,
`rope_long_m_scale=1.0`, `rope_max_positions=1024`,
`rope_original_max_positions=1024`, `mrope_rotary_cos_sin=None`,
`mrope_position_deltas=None`. Non-zero `rope_dim` with a rope-type
`position_embedding_type` applies RoPE to q/k inside the call — not
certified for this configuration.

MLA (all calls receive the same values; only the fresh-prefill context
call rotates anything): `position_embedding_type=8` (yarn, as MLA models
set), `rope_dim=R`, and `rotary_cos_sin` = fp32
`[1, max_positions * R * 2]` duplicated-layout table — per position, `R`
`(cos, sin)` pairs at flat offsets `p*2R + 2d` and `p*2R + 2d + 1`, whose
second `R/2` pairs duplicate the first; only pairs `[0, R/2)` are read per
position.

**The table's content is the only rope input the op reads.** Two contents
are certified, both at `H = 128`, page 32, on the fresh-prefill context call
(the only MLA flavor that rotates anything):

- the **unscaled** table, `angle(p, d) = p / theta^(2d/R)` at
  `theta = 10000` — what a config without rope scaling produces, and what
  every MLA case at `H` = 8, 16 and 32 above uses;
- the **YaRN-scaled** table a DeepSeek-R1-0528 config produces: `theta`
  10000, `factor` 40, `original_max_position_embeddings` 4096,
  `beta_fast` 32, `beta_slow` 1, `mscale` 1.0, `mscale_all_dim` 1.0.

The YaRN content, written out so a caller can build it without reaching into
TensorRT-LLM (fp32 throughout; `d` indexes the half-dimension `[0, R/2)` and
`j` the table's `R` pairs per position, `[0, R)`):

```
freq(d)     = theta ** (2d / R)
low         = max(0,     floor(R * ln(orig_max_pos / (beta_fast * 2*pi))
                               / (2 * ln(theta))))          # 10 for R1
high        = min(R - 1, ceil (R * ln(orig_max_pos / (beta_slow * 2*pi))
                               / (2 * ln(theta))))          # 23 for R1
ramp(d)     = clamp((d - low) / max(high - low, 0.001), 0, 1)
inv_freq(d) = ramp(d) / (factor * freq(d)) + (1 - ramp(d)) / freq(d)
m(x)        = 1.0 if factor <= 1 else 0.1 * x * ln(factor) + 1.0
amplitude   = m(mscale) / m(mscale_all_dim)                  # 1.0 for R1
angle(p, j) = p * inv_freq(j mod (R/2))       # the duplication lives here
table[p*2R + 2j]     = cos(angle(p, j)) * amplitude
table[p*2R + 2j + 1] = sin(angle(p, j)) * amplitude
```

Setting `factor = 1` collapses this to the unscaled table (verified: the two
constructions agree to 3.8e-6 max abs, fp32 evaluation-order noise). Two
consequences
worth naming: YaRN rescales only the **low-frequency** half of the spectrum
(`ramp` is 0 below `low`), so the two tables agree closely at small positions
and diverge with distance; and the table's `amplitude` is **1.0 whenever the
config's two mscales are equal**, which is R1's case — the model's YaRN
attention temperature `mscale = 0.1*ln(40) + 1 = 1.36885` is folded into the
softmax scale through `q_scaling = 1/mscale² ≈ 0.53366` instead, never into
the table.

**Everything else about the rope configuration is inert on the MLA path** —
measured, not inferred. With the table held fixed, the seven scalars
`rope_base`, `rope_scale_type`, `rope_scale`, `rope_short_m_scale`,
`rope_long_m_scale`, `rope_max_positions`, `rope_original_max_positions` and
the `rotary_inv_freq` tensor moved **no** observable of a fresh-prefill
context call: output rows, the whole paged latent pool, the in-place-roped
`q`/`k`, all bitwise identical across R1's production scalar set, the
unscaled set above, a deliberately out-of-range set (`rope_base` 500000,
`rope_scale_type` 3, `rope_scale` 7.5, m-scales 3.0/9.0, position windows 77
and 13 — both shorter than the sequences in flight), and `rotary_inv_freq`
zeroed or passed as `None`. Swapping the table in the same comparison does
move all of them, which is what makes the bitwise result mean something.
So: pass whatever scalars your config carries, and get the table right.

Two obligations the table brings, neither of them checked:

- It must hold **one row per absolute position the call ropes**. Nothing
  bounds-checks it and `rope_max_positions` is not consulted: a 64-row table
  under a 96-token prefill returned without raising and was wrong from output
  row 64 on (max abs diff 3.9 against magnitude-4 outputs) — the kernel reads
  past the end of the tensor. Certified with 1024 rows over sequences
  reaching position 511.
- Row content does not depend on the row count, so a table truncated to
  `max_seq_len` rows is **bit-identical** to the leading rows of the full
  163840-row table an engine builds for R1 (verified element for element).
  Truncating is therefore safe; short-changing the positions in flight is
  not.

`rotary_inv_freq` = the `[R/2]` fp32 tensor from the same construction. It
is accepted, unread on this path (above), and still worth passing as
production does.

### Feature groups not certified — pass these inert values

| Group | Arguments → inert value |
|---|---|
| Quantized output / quantized QKV input | `out_scale=None`, `quant_scale_qkv=None` (fp8/fp4-quantized attention *output* and the fused DSv4 QKV-quantization path are not certified — on the MLA fp8 path `quant_scale_qkv` was additionally observed to be inert, a `[1]` fp32 `1/s` there leaving the context output bitwise identical. The fp8-e4m3 **KV pool** is certified in both configurations — see those sections, whose two `kv_scale_*` tensors then carry the scale, `None` being read as `1.0` rather than rejected. On the bf16 pool both stay `None`, and so does `quant_q_buffer`.) |
| Folded KV RMSNorm (MLA) | `kv_norm_weight=None`, `kv_norm_eps=1e-6`. Non-`None` weight folds the `kv_a_layernorm` into the KV kernel, which then reads `latent_cache` **raw** — a caller that already normalized would normalize twice. Only DeepSeek-V4's sparse module uses it upstream |
| Skip-correction (MLA) | `skip_correction_threshold=0.0`. A lossy trtllm-gen MLA option, SM100/SM103 only, off by default upstream (`enable_mla_skip_correction`). The engine forces it to 0.0 on a non-MLA layer regardless |
| Spec-dec tree mask | `force_prepare_spec_dec_tree_mask` — engine-set, `True` only for a dynamic tree; a linear-tree draft leaves it `False` |
| Sequence-count sizing | `max_num_sequences` — engine-set, defaults to `max_num_requests` |
| MLA DSv4 / FlashMLA / sparse-MLA sub-features | `flash_mla_tile_scheduler_metadata=None`, `flash_mla_num_splits=None`, `dsv4_inv_rope_cos_sin_cache=None`, `enable_dsv4_epilogue_fusion=False`, `aux_kv_cache_pool_ptr=None`, `sparse_attn_kv_lens=None` (`mla_bmm1_scale` / `mla_bmm2_scale` / `quant_q_buffer` are **certified** for the MLA generation call over an fp8-e4m3 latent pool, where all three are required — see that section; `None` everywhere else) |
| Speculative decoding — the **mask/offset tensor** group | `is_spec_decoding_enabled=False`, `use_spec_decoding=False`, `is_spec_dec_tree=False`, all `spec_decoding_*=None`, `spec_bl_tree_first_sparse_mask_offset_kv=None`, `spec_decoding_target_max_draft_tokens=None`. This does **not** mean speculative decoding is unavailable: `predicted_tokens_per_seq > 1` is certified for the MLA generation call with exactly these inert values, and that is also what the engine passes on sm_100 for a linear-tree draft, where `is_spec_decoding_enabled` is forced off (the Python backend computes it as `is_spec_decoding_enabled and (not trtllm_gen_arch or is_spec_dec_dynamic_tree)`), leaving the mask, position-offset and generation-length tensors all `None`. The within-block mask then comes from `predicted_tokens_per_seq` alone — see its row above. A **tree** draft, which is what these tensors describe, is not certified |
| Sparse / skip-softmax | `sparse_kv_indices=None`, `sparse_kv_offsets=None`, `sparse_attn_indices=None`, `sparse_attn_offsets=None`, `sparse_attn_indices_block_size=0`, `num_sparse_topk=None`, `skip_softmax_threshold_scale_factor_prefill=None`, `skip_softmax_threshold_scale_factor_decode=None`, `skip_softmax_stat=None` (`softmax_stats_tensor` is certified for the no-append MLA context call — see that table; `None` everywhere else) |
| SageAttention | `sage_attn_num_elts_per_blk_q=0`, `sage_attn_num_elts_per_blk_k=0`, `sage_attn_num_elts_per_blk_v=0`, `sage_attn_qk_int8=False` |
| Helix CP | `helix_position_offsets=None`, `helix_is_inactive_rank=None` |
| Relative bias (T5) | `relative_attention_bias=None`, `relative_attention_max_distance=0` |

`attention_sinks` is **certified** for the standard bf16-pool causal
configuration — see *Attention sinks*; pass `None` everywhere else
(MLA calls, the fp8-e4m3 pool, `mask_type=0`).

## Metadata consumed

None — the op reads no thread-local model attrs, no registered layers, and
no global Python state. Every input above is an explicit argument. (The
batch-state tensor block plays the role that prepared attention metadata
plays for the registered-layer entry points.)

## Preconditions

- Batch order: context sequences before generation sequences; `q` rows in
  that order. `num_contexts` / `num_ctx_tokens` passed explicitly and
  consistent with `host_request_types`.
- `num_heads % num_kv_heads == 0` (the wrapper asserts it: a context-only
  call at a non-multiple geometry returns a silently wrong `output`).
- `head_size` is one of the kernel's head dims. On sm_100 an unsupported
  `head_size` **aborts the process**: the kernel-selection failure is
  thrown where it cannot unwind (`std::terminate`), so it is neither
  catchable nor loggable by the caller — observed at 32 (`Invalid TilePvM
  as MMA only supports 64 or 128`), 96 and 192 (`Unsupported HeadDim for
  BMM2-N`). Same for a `num_heads` that kernel selection cannot group into
  CTAs (observed at `Hq = 7`, `Hkv = 2`: `Internal error numHeadsQ=7,
  numHeadsPerCta=3, numCtasForAllHeads=2`). Verify a new `head_size` in a
  throwaway process before wiring it into a target.
- Standard-configuration context sequences must start empty
  (`sequence_length == host_context_lengths ==` new-token count) **unless**
  `use_paged_context_fmha=True`; a cached prefix at `False` is silently
  wrong (see *Paged-context FMHA*). The MLA fresh-prefill flavor requires
  empty starts unconditionally; MLA context over cached tokens is certified
  only via the no-append flavor (`latent_cache=None`, explicit full-range
  or chunked K/V), and `use_paged_context_fmha` stays `False` for MLA.
- On the standard paged-context path (`use_paged_context_fmha=True`), a
  context row's `context_lengths` / `host_context_lengths` is **this
  call's** new-token (q-row) count, not the sequence's prompt or KV length.
  Nothing checks it: passing the full KV length landed 101x outside the
  tolerance band and `0` landed 53x outside, both without raising, and both
  also corrupt the pool. `sequence_length` stays the global cached+new
  count, as everywhere else.
- The paged-context path additionally requires every page holding a key the
  context call reads to be valid at call time, and those pages to be
  distinct from one another — the whole `[0, sequence_length)` range
  without a window, the narrower range given in *Paged KV cache addressing
  under a sliding window* with one. The packed path (`False`) reads no page
  at all during a context call, so this obligation is new and nothing
  checks it: aliasing a sequence's pages onto a single page returns a
  result 31x outside the tolerance band without raising.
- No-append MLA context calls additionally require: `k`/`v` rows are
  exactly the per-sequence KV ranges described by `sequence_length`,
  concatenated in batch order, with `host_total_kv_lens[0]` equal to that
  row count; `v`'s row stride is `H * (nope + v_head_dim)` elements; under
  `mask_type=1` every context sequence satisfies `sequence_length >=
  context_lengths` (the causal mask offset is their difference); under
  `mask_type=0` any KV length including 0 is allowed, but the output and
  stats rows of zero-KV sequences are undefined and must never be read
  (production merge plans skip them). `q` arrives fully prepared — the
  call applies no RoPE even though the rope-table arguments are passed.
  Under `quant_mode=128` all of the above is unchanged, and the pool stays
  untouched (it is not read to build `k`/`v` — assembling those from the
  cached latent rows is entirely the caller's step); what changes is that the
  op quantizes the `k`/`v` it is handed to e4m3 before the FMHA, so operands
  the caller has already dequantized off an fp8 pool go through e4m3 a second
  time.
- A caller-provided `softmax_stats_tensor` (no-append MLA context only)
  is fp32 `[>= Tc, >= H, 2]`, contiguous, on the same device (the op
  rejects other dtypes/sizes); certified at the exact size `[Tc, H, 2]`.
- For every generation sequence, the cached tokens at positions
  `[0, kv_s - 1)` were actually written by prior calls addressing the same
  pool pages, and `sequence_length` equals the true cached count + 1.
- Per-sequence pages cover `ceil(total_kv / tokens_per_block)` entries in
  its `kv_cache_block_offsets` row, where `total_kv` is the sequence's
  full `[cached + new]` KV length (also for chunked partial passes, whose
  `sequence_length` entries are smaller); distinct sequences use distinct
  pages; all offsets lie inside the pool. `total_kv <= max_seq_len` for
  every sequence.
- With a sliding window the entry count above is unchanged (the row is
  indexed by absolute page index), but only the entries holding in-window
  tokens must be valid, and the pages behind them must be distinct over
  each call's new-token range plus the whole window — see *Paged KV cache
  addressing under a sliding window* for the ring rule
  (`P * tokens_per_block >= max(attention_window_size, this call's
  new tokens)`), which is not checked anywhere and fails silently.
- `attention_sinks`, when given, is fp32 (the op raises `Expected
  attention_sinks to have float dtype` for anything else) and holds exactly
  `num_heads` values in contiguous memory on the current CUDA device. Only
  the dtype is checked by the op: the buffer is read as `num_heads` raw
  fp32 values from `data_ptr()`, so every other violation is **silent** —
  a shorter tensor is read past its end, a longer one has its tail ignored,
  a strided view is read as its underlying memory (a stride-2 view holding
  the right values returned results ~2.1 off), and an empty tensor
  disables the sink entirely. The wrapper therefore asserts contiguity and
  `numel() == num_heads`; nothing else guards this argument. A pageable
  **CPU** fp32 tensor was also accepted and gave the correct result (the
  values do reach the kernel), but CUDA is the certified device.
- All tensors contiguous unless noted (MLA context `v` is the strided
  split view described in *Signature*), on the devices listed above; every
  CUDA tensor on the current device; dtypes exactly as listed (host length
  tensors are int32 — int64 is not certified).
- `quant_mode` and the pool element type must agree (0 ↔ bf16,
  128 ↔ fp8-e4m3): the op sizes and interprets slabs from `quant_mode`
  alone and cannot detect a mismatched pool allocation.
- With `quant_mode=128`, `kv_scale_orig_quant` and `kv_scale_quant_orig`
  are fp32 `[1]` CUDA tensors holding `1/s` and `s` for one scaling factor
  `s` (mutual consistency is the caller's job). **`None` in either slot or
  in both is not rejected — it is silently read as `1.0`**, bitwise
  identical to passing 1.0 tensors in both the pool bytes and the output,
  in both configurations, in every phase, and on both the single-layer and
  the 4-layer manager pool.
  A forgotten scale is therefore a wrong-number bug, not a crash: inert at
  the production `s = 1.0`, and 1.19 off on magnitude-1.5 decode outputs
  over a standard-configuration pool written at `s = 2.0`.
  Certified `s`: 1.0 (the bf16-checkpoint production value), 1.5, 2.0.
- On the **MLA** path under `quant_mode=128` the same two tensors reach far
  less of the computation, and `s != 1.0` is not survivable in the context
  phase (full measurements in *MLA over an fp8-e4m3 latent pool*):
  - the fresh-prefill append consumes `kv_scale_orig_quant` and is bit-exact
    at any `s`;
  - **both** context flavors quantize `q`/`k`/`v` at 1.0 but apply `s^2` and
    `s` as if they had not, so a context call at `s != 1.0` returns a silently
    wrong result — measured on the fresh-prefill flavor (21x / 48x outside the
    band at 1.5 / 2.0) and on the no-append flavor (48x at 2.0), which appends
    nothing and so has no use for `kv_scale_orig_quant` at all. Pass
    `s = 1.0`, or `None`;
  - the generation call reads **neither** tensor. Its query comes from
    `quant_q_buffer` and its scales from `mla_bmm1_scale`/`mla_bmm2_scale`,
    all three required (`None` raises `quant_q_buf is nullptr.` /
    `bmm1_scale is nullptr.` / `bmm2_scale is nullptr.`, from
    `attentionOp.cpp:1050-1052`), and the caller must fold `s` into them as
    shown in that section. Nothing checks their contents.
- Bits of `quant_mode` outside the KV-cache group are **not** a reason to
  strip the value a checkpoint's quant config produces: `1152` and `384` were
  measured bit-identical to a bare `128` on every MLA flavor (see *MLA over an
  fp8-e4m3 latent pool*). Bits that were not tried are unknown, not inert.
- MLA batches take one call per phase; both calls of a mixed batch use the
  same full-batch state tensors. `attention_input_type=0` is **accepted, not
  rejected**: inert (bitwise identical) on a single-phase batch, silently
  unsatisfiable on a mixed one — see *Semantics*.
- Every MLA call needs an int `q_lora_rank` — `0` for a checkpoint without
  a q-LoRA — even though the value is never read: `None` raises
  `RuntimeError: bad optional access` (see *MLA q-LoRA rank*).
- MLA generation additionally requires, before the call: every generation
  sequence's full latent history `[0, L_g)` resident in the pool — the
  context call's append covers the prefill rows; this step's `P` rows at
  positions `L_g - P .. L_g - 1` come from the caller's generation
  preprocessing — and the caller-allocated `cu_q_seqlens` / `cu_kv_seqlens` /
  `fmha_scheduler_counter` filled as listed in *Signature*. Over an
  fp8-e4m3 pool the same preprocessing must also have filled
  `quant_q_buffer` with the e4m3 query and both `mla_bmm*_scale` buffers
  with the scales above; a decode row appended by the caller must be
  quantized the same way the op's own append quantizes
  (`e4m3(row * kv_scale_orig_quant)`), since the kernel applies no
  per-row scale of its own.
- With `predicted_tokens_per_seq = P` above 1, MLA generation additionally
  requires `L_g >= P` for every generation sequence — draft row 0 attends to
  `[0, L_g - P]`, so a shorter `L_g` leaves it no keys. `L_g == P` is legal
  and certified (row 0 then attends to exactly one key, its own). `q` /
  `output` / `quant_q_buffer` must be `G*P` rows tall and token-major within
  a sequence; `sequence_length[num_contexts + g]` must count **all `P`** of
  this step's tokens. Nothing checks the row count — `q` and `quant_q_buffer`
  are addressed from `P` and the sequence index, so a `G`-row buffer is read
  past its end rather than rejected.
- `host_past_key_value_lengths` must be non-zero for at least one sequence.
  Its per-sequence values are otherwise inert on the MLA generation path
  (`sequence_length - 1`, `- P` and all-ones are each bitwise identical to
  the true values at `P = 3`), but an **all-zero** vector makes the call
  return without writing a single element of `output` — a sentinel-filled
  buffer comes back untouched, at `P` = 1 and above alike, with no error.
  Passing the same values as `sequence_length`, as everywhere else in this
  contract, satisfies it.
- The MLA fresh-prefill context call clobbers the rope slice of each
  `q`/`k` head in place: treat both tensors as consumed (their nope slices
  stay intact, but do not rely on pre-call rope-slice contents
  afterwards). The no-append context call mutates neither.
- `rotary_cos_sin` must hold a row for every absolute position the
  fresh-prefill context call ropes, in the duplicated layout and with the
  content its rope configuration implies (see *In-kernel RoPE arguments*).
  Nothing checks either: a short table is read past its end and comes back
  silently wrong from the first over-long position on, and the scalar rope
  arguments beside it — including `rope_max_positions` — are not consulted at
  all, so they cannot substitute for getting the table right.
- First generation-phase call per process JIT-compiles its FMHA kernel via
  NVRTC (~5-10 s; one kernel per configuration — standard GQA and MLA
  decode compile separately): the C++ locates the shipped kernel headers
  (`tensorrt_llm/include/trtllm_gen_kernels/fmha/`) by running
  `pip show tensorrt_llm` in a subprocess, so a `pip` that resolves the
  installed package must be on PATH (this repo pins `pip` as a project
  dependency for exactly this reason); otherwise the call raises
  `NVRTC_ERROR_COMPILATION` ("could not open source file cuda.h").
  Context-phase kernels are precompiled into `libtensorrt_llm.so` and need
  no JIT.
- The op is synchronous-schema but asynchronous: it launches on the current
  CUDA stream and returns; synchronize before reading `output` on the host.

## Notes

- One Python-level invocation launches several device kernels (QKV/KV-cache
  preprocessing, FMHA, postprocessing) inside the single C++ op — the same
  attention core the TRTLLM backend entry points route into, certified here
  as a single catalog call.
- On sm_100 the op selects trtllm-gen FMHA kernels. Generation kernels are
  JIT-compiled per process (in-memory kernel cache; the log warns "Possible
  JIT Cache Missing ... generateAndCompileKernel took ~5-10 s" on first
  use of each kernel variant — the MLA decode variant is
  `...HQk576HV512...PagedKv...`; the fp8-KV GQA decode variant is
  `fmhaSm100aKernel_QkvE4m3OBfloat16H128PagedKvDense...` — its name pins
  that q, K, and V are all e4m3 in the decode MMAs). Re-runs within the
  process are fast. Context kernels — including the padding-mask and
  softmax-stats variants of the no-append flavor, and the bf16 context
  kernel that keeps serving the context phase under `quant_mode=128` — are
  precompiled (no JIT observed).
- The generation-kernel variant is keyed by `head_size` and the q-tile size
  the head group maps to, rather than by the head counts directly — several
  distinct geometries share one name: the bf16
  variants observed are `...H128PagedKvDenseP32VarSeqQ8Kv128Static...` for
  `Hq/Hkv` of 32/8, 32/4 and 28/4, `...Q16...` for 16/1, and `...H256...` /
  `...H80...` for those head sizes. So a first call at a new head geometry
  costs at most one extra ~6 s JIT compile, and often none. The mapping is
  not constant in the head count either, though: MLA generation at
  `H` = 16, 32 **and 128** all report
  `fmhaSm100aKernel_QkvBfloat16OBfloat16HQk576HV512HVPerCta128PagedKvDenseP64VarSeqQ16Kv128StaticSwapsAbForGen`
  (the `P32` sibling at page 32), while `H = 8` reports
  `...VarSeqQ8Kv128StaticSwapsAbForGen` — same kernel
  family, different q-tile, hence a different compiled kernel. The q-tile
  therefore stops growing well below the head count: 16x the heads of the
  `H = 8` case still map to `Q16`. Where head
  counts do share a name the **compile cache is still keyed more finely**: a
  process that runs 16, 32 and 128 pays the ~5.5 s `...Q16...` compile three
  times, once per head count, while any number of calls at one head count —
  across batch sizes and histories — compile once. The **page size is part of
  the name** too: the same MLA generation call at `tokens_per_block = 32`
  compiles
  `fmhaSm100aKernel_QkvBfloat16OBfloat16HQk576HV512HVPerCta128PagedKvDenseP32VarSeqQ16Kv128StaticSwapsAbForGen`
  (~5.4-5.9 s, observed once per head count that maps to it), a genuinely
  different kernel from the `...P64...` one — so page 32 and page 64 MLA
  decode are separate compiled variants, both certified. Full-test-file counts on sm_100 measured
  from the run behind the current receipt, all at
  `predicted_tokens_per_seq = 1`: `...P64VarSeqQ16...` twice (the
  page-64 16/32 pair), `...P32VarSeqQ16...` **three times** (16/32/128),
  `...P64VarSeqQ8...` once and
  `...P32VarSeqQ8...` once (`H = 8`, each serving both that page size's decode
  and mixed-batch cases, the page-32 one additionally serving the `H = 8`
  q-LoRA-rank sweep's decode calls) — all seven ~5.4-5.7 s, a wall-clock
  figure that moves ~0.15 s run to run. One `H = 128` compile serves all ten
  of the file's `H = 128` decode calls at `P = 1` — 3 from the baseline decode
  and mixed-batch cases, 3 from their R1-cell twins, 4 from the `q_scaling`
  sweep. The MTP section adds two more bf16 compiles on top, one per `P` it
  runs there (`...HVPerCta128...P32VarSeqQ16...` at `P = 2` and
  `...HVPerCta256...P32VarSeqQ16...` at `P = 4`), for nine bf16 MLA decode
  compiles in the file. The counts are the stable part: identical across
  full-file runs.
  Every JIT compile logged across the file is a `...ForGen` decode variant —
  no MLA context call at any certified head count, page size or pool dtype
  triggered one. `quant_mode` is part of the decode kernel name too: MLA
  generation over an fp8-e4m3 latent pool compiles
  `fmhaSm100aKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta128PagedKvDenseP32VarSeqQ16Kv128StaticSwapsAbForGen`
  — the `QkvE4m3` sibling of the bf16 `...P32VarSeqQ16...` name, so q, K and V
  are all e4m3 in the decode MMAs — once for every fp8 MLA decode call at
  `predicted_tokens_per_seq = 1`: the baseline-cell decode, the R1-cell
  decode, the mixed-batch decode and the `quant_mode`-bits decodes all share
  it. So on the fp8 path neither `q_scaling`, nor the rope table, nor the extra
  `quant_mode` bits reach decode kernel selection, and no fp8 MLA **context**
  call — either flavor — triggers a JIT compile at all.
- **`predicted_tokens_per_seq` does reach decode kernel selection**: it becomes
  the kernel's `maxSeqLenQ`, which appears in the JIT log line and moves the
  `HVPerCta` split in the name. Observed at `H = 128`, page 32, both pool
  dtypes: `P` = 1 and 2 take `...HV512HVPerCta128PagedKvDenseP32VarSeqQ16...`
  and `P` = 3 and 4 take the `...HVPerCta256...` sibling, with `P` = 1 and
  `P` = 2 paying **separate** ~5.4-5.7 s compiles despite sharing a name — the
  same finer-than-the-name cache keying the head count shows. The file's fp8
  MLA decode compiles are therefore three, at `maxSeqLenQ` 1, 2 and 3, with
  `P = 4` reusing the `maxSeqLenQ = 3` entry. Batch size can move it as well:
  a probe at `G = 4`, `P = 4` compiled a further variant whose name carries no
  `HVPerCta` segment at all. So a target sweeping `max_draft_len` should
  expect one decode JIT compile per `P` it runs, not one for the model.
- First call logs "Attention workspace size is not enough" and resizes
  `workspace_` in place — expected when starting from an empty tensor. The
  size tracks the call's tokens x heads: ~33 MB standard, ~39 MB MLA and
  ~35 MB no-append MLA context at the 8/16/32-head shapes, but ~50 MB for an
  `H = 128` MLA context call over 129 tokens and ~109 MB over 545, which is
  the file's peak (the standard-configuration head-geometry sweep reaches
  ~95 MB). `quant_mode=128` raises the MLA context requirement — the fp8
  path stages quantized q/k/v in the workspace: 61 033 728 B for the same
  129-token `H = 128` call that needs 52 579 072 B in bf16, and 142 612 480 B
  over 512 tokens (measured outside the shipped cases, which stay below the
  file's bf16 peak). The **no-append** flavor pays the same surcharge — it
  quantizes the K/V it is handed, so it stages them too: the shipped R1-cell
  case (66 q rows over 193 K/V rows at `H = 128`) asks for 52 816 640 B under
  `quant_mode=128` against 43 288 832 B at the same shape in bf16. An engine
  at production `max_num_tokens` will be well above all of
  these — budget from the shape, not from these figures.
- The paged-cache append is bit-exact wherever it is a copy: after a
  standard call, page `p` of a
  sequence holds exactly the bf16 K/V slices packed into `q`; after an MLA
  fresh-prefill context call it holds `[ckv | rope(k_pe)]` with `ckv`
  bitwise-copied (verified for both configurations, including page-boundary
  crossings, and for MLA at page sizes 32 and 64). The MLA RoPE half is
  bit-identical to an fp32 reference rounded once to bf16 **except at
  fp32→bf16 rounding ties**: kernel and torch reference evaluate
  `x*cos ∓ y*sin` in different fp32 orders, and where the correctly-rounded
  fp32 result lands exactly on a bf16 midpoint the two round to adjacent
  bf16 values. Measured on sm_100 over 60 prefill runs spanning both page
  sizes (495 360 roped elements): 8 such elements — 1.6e-5 of the total,
  never more than one bf16 ulp, and the same count at page 32 as at page 64,
  so it is arithmetic rather than addressing. The entry's test gates the
  RoPE half at one bf16 ulp (`rtol = 2^-7`, `atol = 0`) plus a hard "at most
  one element in a thousand differs at all" floor, and keeps the `ckv` half
  strictly bitwise; that gate leaves the real failure modes far outside it —
  measured on one 96-token page-32 request, an append shifted one slot lands
  at 6.1e5 ulp with 99.9% of elements differing, the pool read with page
  size 64 instead of 32 at 2.6e5 ulp / 33%, RoPE applied at position `i+1`
  at 5.1e5 ulp / 69%, `k_pe` left un-roped at 4.4e5 ulp / 93%. The
  standard-configuration fp8-pool
  append is bit-exact against
  `e4m3(k * kv_scale_orig_quant)` (fp32 multiply, RN cast) for power-of-2
  scales; at `s=1.5` ~0.6% of elements sit one e4m3 ulp off that mirror
  (never more) — see the fp8 section. The **MLA** fp8 append is bit-exact
  against the same mirror at every scale run (1.0, 1.5, 2.0) and on both
  halves of the latent row, roped `k_pe` included: the e4m3 grid is coarse
  enough to swallow the rounding tie that costs the bf16 latent append its
  one-ulp elements. Its gate is one e4m3 ulp plus the same
  one-in-a-thousand inexact floor, and no run has approached it — across the
  appended-row checks the fp8 MLA cases make (three scales, both rope tables,
  all four call shapes), not one byte differs from the mirror, in either half.
  Two things make that bit-exactness load-bearing rather than merely tidy.
  First, it is the sharpest place the **rope table** shows up: a mirror roped
  by the unscaled theta-10000 table instead of R1's YaRN one differs in 2336
  of the 6144 e4m3 bytes of a 96-token request (38.0%) against a gate that
  allows 6 — ~389x — so the same comparison that passes on the certified
  content demonstrably fails on the wrong one. Second, the comparison can see
  a **racing** append: replaying this entry's own documented arming sequence
  (a call whose new tokens do not map to distinct physical slots — here a
  96-token prefill with all three of its absolute pages aliased onto one
  physical page) made an identical seeded call leave different pool bytes on
  every repeat — five of five in a scratch probe, 11 177 to 12 513 bytes
  apart — while the certified geometry reproduced bitwise. The shipped
  control asserts exactly that (four armed repeats must not all agree, three
  certified ones must), and it fired in five of five fresh processes. Both
  controls run in the test file, in the same processes as the certified
  cases.
- Multi-layer pool addressing (standard configuration, sm_100): certified
  against state produced by a real 4-layer `KVCacheManager` (bf16,
  GQA 8/2/128, `tokens_per_block` 32) — identity mapping rows,
  layer-agnostic offsets (K `8p`, V `8p + 1`), prefill plus page-crossing
  decode at `local_layer_idx` 0-3. Each layer's append landed bit-exactly
  in its own slab group with sibling layers bitwise untouched, and decode
  read back each layer's own history. A doctored-mapping call
  (`local_layer_idx=1`, row 1 rewritten to layer-in-pool 3) landed its
  append in layer 3's slabs while layer 1 stayed bitwise intact: the
  pool-base shift is `mapping[local_layer_idx][1] * kv_factor` slabs —
  the row's layer column, `local_layer_idx` only selecting the row. The
  same 4-layer prefill + page-crossing-decode certification was re-run
  over an fp8-e4m3 pool from a `DataType.FP8` manager (`quant_mode=128`,
  GQA 32/8/128, `tokens_per_block` 32, `s = 1.0` — identical mapping and
  offset layout): every layer's context and generation append landed
  bit-exactly against the `e4m3(k * kv_scale_orig_quant)` mirror in its
  own slab group, sibling layers stayed bitwise untouched by every call,
  and each layer's decode read back its own history — pinning that the
  layer-base shift is computed in e4m3-sized slabs under `quant_mode=128`.
  The doctored-mapping probe was not repeated under fp8.
- Accuracy (bf16, sm_100) vs fp32 references, compared with `atol=5e-3,
  rtol=1.6e-2` (not the default bf16 `atol=1e-5`): standard max abs err
  1.6e-2 on magnitude-~1 outputs (4-layer shared pool the same: 1.6e-2
  prefill, 7.8e-3 decode); MLA fresh-prefill context max abs err
  1.6e-2; MLA generation max abs err 7.8e-3; no-append MLA context max abs
  err 1.6e-2 (one-shot cached-KV), 7.8e-3 (chunked partial passes) rising to
  1.6e-2 on the final causal pass, 1.6e-2 (chunk passes merged vs a
  single-pass full-range reference). The largest fraction of the combined
  `atol + rtol*|ref|` allowance any case in the test file uses is 66% — the
  maximum sits on the `q_scaling` sweep's `q_scaling = 1.0` context run at
  `H = 128` (66.1%), just ahead of the paged-context 4-layer shared-pool case
  (65.8%), the page-32 cached-KV no-append MLA context case at `H = 32`
  (63.9%), the page-32 `H = 128` fresh prefill (63.3%), the fp8-pool
  multi-layer context (61.5%) and the YaRN-table case (61.3%);
  this bullet's standard GQA/MHA cases sit at 36-52% and its MLA
  single-pass cases at 36-64% (the sink, sliding-window and paged-context
  bullets below carry their own numbers; the latent append's one-ulp gate
  is a separate, tighter one, and one case does approach it — the page-32
  `H = 32` mixed batch reaches 84% of the one-ulp gate on its roped
  elements). Every figure in this bullet and this ceiling
  were re-measured across the whole file in the state the current receipt
  certifies, by an instrumented sibling run of the same seeded cases.
  Head geometry does not move these numbers: the two shipped-target
  geometries (32/8/128, 32/4/128) and the MQA / odd-ratio / odd-count /
  `head_size` 256 sweep all land at max abs err 1.6e-2 and at most 53% of
  the allowance, the same envelope the 8/2/128 and 4/4/64 cases sit in.
  A scratch probe of wider ratios (64/8/128, 64/1/128, 128/1/128) stayed at
  the same 1.6e-2 max abs err but pushed decode to ~75% of the allowance —
  headroom narrows as the head group grows, so re-measure before adopting a
  ratio above 16. The 64/8/64 geometry without sinks sits in the same
  envelope: max abs err 1.6e-2, at most 50% of the allowance.
- MLA head count does not move the numbers either (bf16, sm_100, same fp32
  references and `atol=5e-3, rtol=1.6e-2`). At `H = 32`, page 64:
  fresh-prefill
  context max abs err 1.6e-2 (53% of the allowance), cached-KV no-append
  context 7.8e-3 (50%), generation decode 7.8e-3 (37%), mixed batch 1.6e-2
  context (51%) / 7.8e-3 decode (36%) — the same magnitudes the `H = 16`
  runs of those four cases produce in the same process (1.6e-2 at 49%,
  1.6e-2 at 40%, 7.8e-3 at 39%, 1.6e-2 at 38% / 7.8e-3 at 36%), so doubling
  the head count costs no headroom.
  Halving it below 16 costs none either, and the `H = 8` cell has the
  mildest worst case of the three (45.9% against 48.5% at `H = 16` and 63.9%
  at `H = 32`): at page 64, fresh-prefill context 1.6e-2 (46%),
  cached-KV no-append context 1.6e-2 (40%), generation decode 7.8e-3 (37%
  then 36%) over two steps, mixed batch 7.8e-3 context (38%) / 7.8e-3 decode
  (36%); at page 32, fresh-prefill 7.8e-3 (39%), cached-KV no-append 7.8e-3
  (40%), decode 7.8e-3 (37% then 34%), mixed batch 7.8e-3 context (37%) /
  7.8e-3 decode (36%). Case by case, every `H = 8` figure is at or below the
  `H = 32` one for the same case; against `H = 16` the only one above is the
  page-64 mixed batch's context row, 38.5% where `H = 16` reads 37.9%.
  Multiplying it by 4 to `H = 128` costs no headroom either, at page 32:
  on the baseline rope/scale, fresh-prefill context 1.6e-2 (63%), generation
  decode 7.8e-3 (40%) then 1.6e-2 (40%) over two steps, mixed batch 1.6e-2
  context (52%) / 7.8e-3 decode (37%), cached-KV no-append context 1.6e-2
  (48%); at the full R1 cell (YaRN table, `q_scaling ≈ 0.53366`) the same
  four cases read 1.6e-2 (61%), 1.6e-2 (49% then 43%), 1.6e-2 context (58%) /
  7.8e-3 decode (37%), and 1.6e-2 (54%). Same max abs errors as every smaller
  count; the fractions sit inside the 34-64% band the others span, and the
  head count is not what orders them (the `H = 128` fresh prefill's 63% is
  the highest of its set, where at `H = 8` and `H = 32` the no-append case
  is).
  The paged latent append held to its gate (see the append bullet above) at
  all four head counts, including page-boundary crossings, and every count
  leaves `latent_cache`, `q_pe`, `q`/`k`/`v` and the pool untouched exactly
  where the per-flavor tables say they do. At `H = 8` the appended rows came
  back bitwise exact on both halves in all six append-checking cases (no
  fp32→bf16 rounding tie among their roped elements), so the one-ulp gate was
  not approached there; the same held at `H = 128`, whose 7 append-checking
  cases make 16 appended-row checks between them — every one bitwise exact on
  both halves, on both table contents.
- MLA page size does not move the numbers either (bf16, sm_100, same fp32
  references and `atol=5e-3, rtol=1.6e-2`). At `tokens_per_block = 32`,
  `H = 32`: fresh-prefill context max abs err 1.6e-2 (50% of the allowance),
  generation decode 1.6e-2 (42%) then 7.8e-3 (37%) over two steps, mixed
  batch 1.6e-2 context (42%) / 7.8e-3 decode (37%), cached-KV no-append
  context 1.6e-2 (64%); the page-32 decode case at `H = 16` sits at 7.8e-3 /
  37% on both steps, and the four page-32 cases at `H = 8` (numbers in the
  head-count bullet above) span 34-40%. Those are the page-64
  magnitudes and, the no-append case aside, the page-64 band. The no-append flavor never reads or writes
  the pool, so its 64% tracks that case's KV geometry (prefixes 96/31/0
  reaching 128/40/25 keys) rather than the page size — it was the file's
  ceiling until the `H = 128` `q_scaling` sweep edged past it at 66%.
  The paged latent append held to the same gate at page 32, the generation
  call still wrote nothing, and the no-append call still left `q`/`k`/`v`
  and the pool bitwise intact.
- MLA q-LoRA rank moves nothing whatsoever, at either shipped head count
  (bf16, sm_100, `tokens_per_block = 32`; `H = 32` is the deepseek-v3-lite
  tp1 cell, `H = 8` its tep4 slice; same fp32 references and `atol=5e-3,
  rtol=1.6e-2`). At `q_lora_rank = 0`, `H = 32`: fresh-prefill context max
  abs err 1.6e-2 (47% of the allowance), generation decode 7.8e-3 (36% then
  37%) over two steps, cached-KV no-append context 1.6e-2 (44%). At
  `q_lora_rank = 0`, `H = 8`: fresh-prefill context 1.6e-2 (44%),
  generation decode 7.8e-3 (37% then 36%), cached-KV no-append context
  1.6e-2 (37%). All six sit between 35.9% and 46.7% of the allowance — the
  band the page-32 cases above sit in — and the two head counts interleave
  there rather than separating (the `H = 8` decode's first step, 36.5%, is
  the one figure above its `H = 32` counterpart, 35.9%). Each fresh-prefill
  case's appended latent rows came back bitwise exact on both halves (no
  fp32→bf16 rounding tie among either run's 8256 roped elements, so the
  one-ulp gate was not even approached), and the no-append calls again left
  `q`/`k` and the pool bitwise intact. Since the four-value rank sweep is
  bitwise identical throughout (see *MLA q-LoRA rank*), these are element
  for element the numbers the `q_lora_rank = 1536` runs of the same six
  cases produce. The rank reaches neither kernel selection nor the compile
  cache at either count: a process running only the two four-value sweeps
  generated
  `...HQk576HV512HVPerCta128PagedKvDenseP32VarSeqQ16Kv128StaticSwapsAbForGen`
  exactly **once** (the `H = 32` sweep, 8 decode calls) and
  `...HQk576HV512HVPerCta128PagedKvDenseP32VarSeqQ8Kv128StaticSwapsAbForGen`
  exactly **once** (the `H = 8` sweep, 8 decode calls); and in the entry's
  full test run, measured with and without the `H = 8` sweep present, the
  counts are identical either way — the first variant twice, once each for
  the two head counts that share it (`H = 16` and `H = 32`), the second once
  (`H = 8`) — so those 8 extra decode calls add no compile at all, against
  one compile per value on a real selection axis.
- MLA `q_scaling` is a live axis of both phases, and only of the softmax
  scale (bf16, sm_100, `H = 128`, `tokens_per_block = 32`, YaRN table). Swept
  over 1.0, 0.53366, 2.0 and 0.25 on identical inputs, each run compared
  against fp32 references built at all four scales: the matching reference is
  the only one within tolerance, at 30-66% of the allowance, while the full
  4x4 cross matrix spans 20.1x-178x the allowance in both phases — the
  tightest cell is 1.0 against 2.0 in the context phase (20.1x), and a
  reference at 1.0 for a run at any other value (the "argument silently
  ignored" hypothesis) spans 20.1x-70.5x. The axis is finer than the sweep
  resolves: 0.5 against 0.53366, a 6.3% change of scale, separates by only
  2.95x-3.61x, so the shipped values are kept apart rather than the gate
  loosened. Across the sweep the paged latent pool comes back **bitwise
  identical** — `q_scaling` reaches the softmax scale and nothing else, not
  the RoPE and not the append. It does not reach kernel selection either: the
  sweep's four decode calls compile nothing beyond the one `H = 128` decode
  kernel the file already pays for.
- The MLA rope table is read for its content only (bf16, sm_100, `H = 128`,
  `tokens_per_block = 32`). A fresh-prefill context call driven by the
  DeepSeek-R1-0528 YaRN table reproduces a table-aware fp32 reference at 61%
  of the allowance and sits 27x outside a reference built from the unscaled
  theta-10000 table; the converse run sits 27x outside the YaRN reference.
  That separation is position-dependent — 8.6x over a 96-token sequence,
  16.9x at 256, 27.0x at 512, 31.7x at 960 — because YaRN rescales only the
  low-frequency half of the spectrum, so a short probe would not have
  resolved the two tables at all. Meanwhile the seven scalar rope arguments
  and `rotary_inv_freq` are bitwise inert beside the table (see *In-kernel
  RoPE arguments*), and a plain-torch rebuild of the YaRN formula matches
  TensorRT-LLM's own table to 1.9e-6 max abs on cos/sin values in [-1, 1] —
  a few fp32 ulp from a different evaluation order of the same blend, where
  formula slips land 4-5 orders of magnitude away (dropping the
  interpolation, or moving `beta_fast` 32 → 64, both differ by 2.0; taking
  the amplitude as `mscale` rather than the `mscale`/`mscale_all_dim` ratio
  differs by 3.7e-1).
- Accuracy with attention sinks (bf16, sm_100, 64/8/64 causal) vs fp32
  sink-aware references at the same `atol=5e-3, rtol=1.6e-2`: max abs err
  1.6e-2, at most 50% of the allowance, across context prefill, two decode
  steps, a mixed batch, decode over 600- and 2000-token histories, and the
  constant-V probe that reads the softmax row mass directly. Every case
  also had to sit **outside** the tolerance band of two rival references —
  "sink silently ignored" and "sink pre-scaled" — and the tightest of those
  separations was 12.8x the allowance (decode over 2000 cached tokens,
  where a small sink is genuinely negligible against 2000 keys, so the test
  draws the sink near `log(kv_len)` to keep the mechanism observable).
- Accuracy with the sliding window (bf16, sm_100, 64/8/64 causal, `W` 128 /
  100 / 33) vs fp32 windowed references, sinks on and off, at the same
  `atol=5e-3, rtol=1.6e-2`: max abs err 3.9e-3 (at most 30% of the
  allowance) across prefill past the window, decode, a mixed batch, a
  2000-token history, a 300-token prefill, and 250 wrapping decode steps;
  the per-layer alternating-window case sits at 1.6e-2 / 49%, its
  full-attention layers being the wider ones. Each case is additionally
  gated **outside** the tolerance band of a "window ignored" (full causal)
  reference on top of the two sink rivals. The exact per-key weights read
  out by the one-hot probe match `1 / (n + exp(sink[h]))` to 3.6e-3
  relative — 59% of the 6e-3 (~1.5 bf16 ulp) gate — while out-of-window
  keys are bitwise zero, which is what pins the boundary at `W` rather than
  `W + 1` keys.
- Accuracy on the paged-context path (bf16, sm_100, `use_paged_context_fmha
  = True`) vs the same fp32 references at the same `atol=5e-3, rtol=1.6e-2`:
  max abs err 1.6e-2 across every case — fresh prefill and decode at the
  three shipped geometries (53% of the allowance), cached prefixes at
  32/8/128 and 32/4/128 including page-boundary crossings and a mixed
  cached/fresh/generation batch (54%), the six-prefix page-grid batch
  (59%), two chunked-prefill loops (42%), and the 4-layer shared-pool
  cached-context cycle (66%, the file's ceiling). The gpt-oss cell sits
  lower: 7.8e-3 / 40% over sinks with and without the 128-token window, and
  1.6e-2 / 49% on the 200-token windowed prefill that sets up the read-set
  probe; each sink case is gated outside the sink-ignored (11.9x-56x),
  sink-pre-scaled (11.3x-53x) and — with the window — window-ignored
  (32x-43x) rivals. The one-hot probe on a cached-prefix context row reads
  the per-key weights back to 2.2e-3 relative, 36% of the same 6e-3 gate,
  with out-of-window keys bitwise zero. Wrong batch states land 31x-318x
  outside the band
  (aliased pages 31x, `use_paged_context_fmha=False` over a cached prefix
  51x, `context_lengths` = full KV length 101x or 0 53x, a decoy in place
  of an in-window cached page 248x-318x); the test gates them at 20x.
- Sliding-window kernel variants (sm_100): the FMHA family is chosen from
  the batch's KV length against the window, **not** from `max_seq_len`. A
  call moves from `...PagedKvDense...` / `...PackedQkvCausal...` to
  `...SlidingOrChunkedCausal...` exactly when the batch's longest KV run
  *exceeds* `attention_window_size` — at window 128, KV 127 and 128 take the
  dense kernel and KV 129 the sliding one. `max_seq_len` does not enter the
  choice: at window 128 and KV 41 the values 128, 256 and 1024 all give
  `...PagedKvDense...`, and at KV 200 both 128 and 256 give
  `...SlidingOrChunkedCausal...`; the dispatcher's `maxSeqLenKv` is the
  batch's real KV length in every case (41, 128, 129, 200), never clamped to
  the window. One windowed sequence therefore pays two decode JIT compiles
  over its lifetime: `...PagedKvDense...` while its history still fits the
  window, `...SlidingOrChunkedCausal...` from the first step past it. Decode
  JIT-compiles
  `fmhaSm100aKernel_QkvBfloat16OBfloat16H64PagedKvSlidingOrChunkedCausalP32VarSeqQ8Kv128StaticSwapsAbForGen`
  once (~6 s) and `...SlidingOrChunkedCausalP32MultiCtasKvCga...` for long
  histories — selected on the *total* cached length, so a 128-token window
  over 2001 cached tokens still takes it; the context variants are
  precompiled (no JIT observed). Where both families can compute the same
  row (a short sequence alone, then batched behind one past the window) they
  agree bit for bit.
- Paged-context kernels are precompiled too: across a full run of this
  entry's test file every JIT compile logged is a `...ForGen` decode
  variant, none a context one, so switching `use_paged_context_fmha` on
  costs no compile. It does not steer the decode kernel either — a decode
  step at both flag values is bitwise identical.
- Attention-sink quirks (sm_100): the sink pointer is read as `num_heads`
  fp32 values with no size, stride, or device check (see *Preconditions*).
  Enabling sinks costs no extra JIT compile: at 64/8/64 the decode kernel
  `fmhaSm100aKernel_QkvBfloat16OBfloat16H64PagedKvDenseP32VarSeqQ8Kv128StaticSwapsAbForGen`
  was generated once and then served both the sink and the no-sink calls in
  the same process — adding or removing the argument triggered no further
  generation. Long histories select the
  `...H64PagedKvDenseP32MultiCtasKvCga...` variant, which folds partial
  softmax states across CTAs and applies the sink in that reduction. A
  `-inf` or `-100.0` sink reproduces the no-sink output bit for bit — the
  sink adds exactly one term to the softmax denominator and changes nothing
  else in the code path.
- Accuracy (fp8-e4m3 pool, standard configuration, sm_100): context matches
  the bf16 numbers above
  (same kernel, same inputs). Decode vs the fp32 quantization-aware
  reference of the fp8 section: max abs err 3.1e-2 across a 10-seed sweep,
  at most 44% of the `2^-4 * (1 + |ref|)` allowance; a 200-token-history
  decode sat at 9.3e-3. A reference that skips the read-side dequant
  diverges to ~4.4e-1 — the gate separates the failure mode by ~7x. The
  4-layer shared pool sits at the same levels per layer: context max abs
  err 1.6e-2 (61% of the bf16-tolerance allowance — the highest fraction
  observed on any certified case), decode max abs err 3.5e-2 (46% of the
  fp8 allowance).
- Accuracy (fp8-e4m3 **latent** pool, MLA, sm_100, `H = 128`, page 32,
  `s = 1.0`) vs fp32 references over e4m3-rounded operands, compared with
  `atol = 2^-3, rtol = 2^-4`: context prefill (96 + 33 tokens) max abs err
  7.0e-2, 49% of the allowance; generation decode 2.1e-2 / 2.3e-2 over two
  steps, 17-18%; the `s` = 1.5 and 2.0 decodes 2.1e-2, 16%. The floor is 2^-3
  rather than the standard configuration's 2^-4 because the residual is the
  kernel's e4m3 handling of the softmax probabilities, whose error scales
  with the `|V| ~ 1` rows rather than with the output element: at a 2^-4
  floor the context case runs 0.72-1.08 of the allowance over six seeds —
  one seed *over* — because a `129 x 128 x 128` context output samples that
  noise 16x more often than a two-row decode does. Rival separations from
  the same runs: the bf16-KV reference (cache quantization ignored) 26x
  outside the **bf16** band and 1.3-1.6x outside this one; the `s = 1.0`
  math against `s` = 1.5 / 2.0 context runs 21x / 48x; a zeroed
  `mla_bmm1_scale[1]` 4.2x; a decode whose bmm scales omit the kv scale
  2.9x at `s = 1.5` and 3.6x at `s = 2.0`. What the loosened gate cannot resolve is the query's
  own e4m3 rounding in decode (0.23x, against the correct model's 0.17x),
  so no claim rests on it; the "is the context math fp8" question is settled
  bitwise instead by the peaked-softmax V readout described in that
  section.
- Accuracy over the fp8 latent pool at the **complete DeepSeek-R1-0528 cell**
  (YaRN table + `q_scaling = 1/mscale²`, same `atol = 2^-3, rtol = 2^-4`):
  fresh-prefill context (96 + 33) 40.1% of the allowance, generation decode
  19.8% / 21.4% over two steps, mixed batch 36.4% (context) and 14.7%
  (decode), no-append context (prefixes 96/31/0 reaching 128/40/25) 41.4%,
  and the no-append realistic case of the operand/scale test 34.6% — the same
  14-49% envelope the baseline-cell fp8 cases sit in, so neither the rope
  table nor the softmax scale costs headroom. Rival separations measured on
  these runs: a reference at `q_scaling = 1.0` sits 12.7x outside the band in
  fresh prefill, 10.8x in the no-append flavor and 4.2x in decode; an
  unscaled-rope-table reference 3.7x in fresh prefill at 96 tokens (10.5x at
  512, since YaRN rescales only the low-frequency half of the spectrum and the
  two tables diverge with distance); the bf16-KV reference 49.8x outside the
  **bf16** band in fresh prefill, 58.1x in the mixed batch, and 2.2x outside
  the fp8 band in the no-append flavor. The rope table is pinned far harder
  by the append than by any of these: see the append bullet above.
- Softmax stats (no-append MLA context, sm_100): the emitted `(max, sum)`
  match an fp32 reference over the same bf16 inputs to 2e-6 abs (max stat)
  / 1.2e-6 rel (sum stat) — natural-log domain, scaled logits. Feeding each
  pass's output/stats into `trtllm::merge_chunked_attention_for_mla` under
  the production copy/merge/skip plan reproduced single-pass full-range
  attention (output within the tolerances above, merged stats at the same
  1e-6-level agreement), certifying the stats as directly consumable by
  that sibling op.
- V-stride hard-coding: for bf16 separate-QKV context MLA, the trtllm-gen
  kernel computes V's row stride as `num_kv_heads * (head_size - 64 +
  v_head_dim)` elements — the packed kv-projection width — rather than
  reading it from the tensor (per source; the binding's `v_stride_in_bytes`
  plumbing is unused in this version). A scratch probe passing a fully
  contiguous `[Tkv, H*v_head_dim]` V returned wrong results (max abs err
  ~4) and such a call reads out of bounds past the buffer. Always pass the
  strided split view, in both context flavors.
- `chunked_prefill_buffer_batch_size` is consumed only for fp8-context-MLA
  workspace sizing per source; inert in the certified bf16 configurations
  (certified at 1, including the chunked partial passes). Every fp8 MLA
  context case — both flavors — passes 1 as well and none was swept over it,
  so the argument is certified at that one value on the path where source says
  it is live. The chunked partial-pass *pattern* of the no-append flavor
  (`mask_type=0` + `softmax_stats_tensor`) is not certified over an fp8 pool;
  its one-shot cached-KV pattern is.
- Re-running the same prepared batch overwrites the same cache slots
  (append position derives from `sequence_length` minus new-token count),
  so a repeated call is idempotent, not double-appending. The MLA
  generation and no-append MLA context calls write nothing at all (pool
  bitwise unchanged).
- MLA generation head geometry: trtllm-gen decode kernels exist for
  `(C+R, C)` of (576, 512) and (320, 256) per source; only (576, 512) is
  certified, at 8, 16, 32 and 128 query heads (the compiled kernel reports
  `HVPerCta128` at all four — the head count moves the q-tile, not the
  per-CTA head-value width, and it stops moving it above 16).
- Sibling ops exist for adjacent roles:
  `torch.ops.trtllm.attn_custom_op_inplace` and
  `torch.ops.trtllm.mla_custom_op_inplace` (registered-layer wrappers over
  the same attention core, state via model extra attrs),
  `torch.ops.trtllm.create_attn_outputs` / `create_mla_outputs`
  (output-buffer allocation), `torch.ops.trtllm.mla_rope_generation`
  (MLA generation preprocessing: q_pe RoPE + latent append + the scheduler
  buffers this op consumes),
  `torch.ops.trtllm.mla_rope_append_paged_kv_assign_q`,
  `load_paged_kv_cache_for_mla`, `load_chunked_kv_cache_for_mla` and
  `merge_chunked_attention_for_mla` (the MLA cached-KV / chunked-prefill
  context flow), and the split-phase trtllm-gen bindings
  `thop.trtllm_gen_context_preprocess` / `thop.trtllm_gen_generation_preprocess`
  / `thop.trtllm_gen_context_postprocess`.
