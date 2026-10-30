# Latent KV cache (MLA)

What a target assembled on Multi-head Latent Attention had to establish by
observation, because no contract or reference stated it. Written from the
deepseek-v3-lite-nvfp4/sm_100/tp1 onboard (DeepSeek-V3-Lite: 30 layers,
32 q heads, `kv_lora_rank` 512, `qk_nope_head_dim` 128,
`qk_rope_head_dim` 64, `v_head_dim` 128, attention entirely bf16). The
mechanism — one compressed latent per token in place of per-head K and V
— recurs across the DeepSeek family and its derivatives; this file is
about the mechanism, not that checkpoint.

**Attention TP replicates the latent pool; it does not shard it.**
Measured on the first multi-rank MLA target (`tep4`, world size 4): every
rank allocated 125.76 GiB for 3,907,296 tokens = **34,560 B/token** =
`num_layers * (kv_lora_rank + qk_rope_head_dim) * 2` = `30 * 576 * 2` —
bit-for-bit the world-size-1 figure, **with no rank factor anywhere**.

The structural reason, which is why this generalizes rather than being a
fact about one engine: the latent is *one compressed row per token that
every head reads*. Splitting heads across ranks therefore splits nothing
in the cache. `kv_a_proj_with_mqa` and `kv_a_layernorm` stay replicated,
each rank computes the identical latent row, and each appends it to its
own full-width pool.

So **attention TP buys zero KV-cache memory on MLA** — the opposite of
GQA, where the pool divides by `tp_size`. Budget a `tepN` MLA target for
the full single-rank pool on every rank.

**Attention DP does not narrow the pool either — it multiplies what the
pool holds.** An earlier revision of this file predicted that a `dep`
segment would be "the topology that actually partitions the latent, by
partitioning the *requests* instead of the heads". The first attention-DP
MLA target (`dep4`, world size 4) measured it and **the prediction is half
wrong**: every rank still allocates a full-width pool at **34,560 B/token
— bit-for-bit the tp1 and tep4 figure, with no rank factor anywhere**
(146.5 GiB / 4,552,704 tokens per rank; 149.18 GiB / 4,634,880 under the
gate's config).

What actually changes is *whose* tokens are in it. Under TP the ranks hold
**identical** latent rows, so three of the four copies are pure waste;
under DP they hold **disjoint requests**, so the same per-rank bytes become
job-wide capacity of `world_size x` the single-rank pool — 18,210,816
tokens against `tep4`'s 3,907,296. The win is real and it is 4x, but it is
a *duplication* win, not a *sharding* win, and the distinction matters when
sizing: **per-rank pool arithmetic is identical across tp1, tepN and depN,
so a rank that fits at world size 1 fits at any of them, and one that does
not fit is not rescued by adding ranks.**

Restated as the rule to size by: on MLA, no parallel topology divides the
per-rank latent pool. Choose `dep` for the aggregate capacity and for the
per-rank *compute* split, never to make the pool fit.

Everything else below still comes from world-size-1 runs, and the sizing
facts hold per rank unchanged.

**This file does not repeat the vocabulary.**
`.claude/skills/trtllm-modeling-v2-kernel-onboard/references/decompositions/mla-custom-op.md`
owns that: which entry
plays which role, the assembly per batch mode (§4), and the
assembler-side obligations certification established (§4.5). Read it
first. What follows is only what a target discovers *beyond* it, on the
way to a green gate.

One item there can now be struck: §5 recorded that the end-to-end parity
run of an assembled MLA pipeline was deliberately left to the first MLA
target. It was run — against a pure-torch reference built from the HF
`DeepseekV3Attention` semantics, **0.93–1.02 bf16 ulp** across 1-seq
prefill, 2-seq prefill, mixed-batch context and mixed-batch decode,
identical at `tokens_per_block` 32 and 64.

## Three things the contracts describe in prose and a target must measure

**The `rotary_cos_sin` byte layout.** The contracts name a TensorRT-LLM
producer for it (`RopeParams(..., duplicate_data=True)`) — a symbol a
target is not allowed to read. The prose admits two readings, and both
are buildable. Only one is right: **interleaved `(cos, sin)` pairs at
flat offset `p * 2R + 2d`**, where `R` is `qk_rope_head_dim` and `d`
indexes the half-dimension. The "cos block, then sin block" reading is
**36x worse** at the same geometry. Build both and measure; do not
reason it out.

**Which `v` the context FMHA wants.** The contract says "the
`[.., H*nope:]` split of a packed `[Tc, H*(nope+v)]` kv-projection
buffer". That is ambiguous against the packing an HF `kv_b_proj`
actually produces, which is **per-head** `[k_nope | v]` interleaved head
by head. The op settles it by rejecting a 3-D `v`
(`Expected v->dim() == 2`), and parity confirms it. The consequence is a
load-time obligation no contract states:

> **A DeepSeek-style `kv_b_proj` must be row-regrouped at load** into
> `[all heads' k_nope ; all heads' v]`. Left in checkpoint order it is
> silently wrong.

This composes with the constraint §4.5 already records — V must stay the
*split view* of the kv_b output, because the kernel hard-codes its row
stride. Regroup at load, then split; never materialize a contiguous V.

**`create_mla_outputs` is not usable from a target.** Its
contract requires a live TensorRT-LLM `MLA` module registered in
thread-local extra attrs, which a target does not own and cannot
construct. The decomposition doc lists it under "output allocation",
which reads as available; it is not. Allocate with a plain `torch.empty`. The
same is not true of its sibling `create_attn_outputs` for non-MLA layers.

## The engine builds the latent pool for you — and turns on a path you must handle

**No target-side declaration is needed**, and in tree there is nothing to
declare it in: the checkpoint is read exactly as published, and its own MLA
config fields are sufficient. The engine allocated a **kv_factor-1** pool
sized exactly
`num_layers * (kv_lora_rank + qk_rope_head_dim) * 2 B` per token
(30 × 576 × 2 = 34,560 B/token) and set
`enable_context_mla_with_cached_kv=True` **itself**.
`.claude/skills/trtllm-modeling-v2-core/references/runtime-interface.md` §6
lists only standard-pool fields and does not cover this.

**Block reuse is on by default, and it does hand an MLA model cached
context prefixes.** This is the trap. Observed directly on a plain
boot-gate workload: `num_ctx_cached_tokens=155`, `kv_lens_runtime=[162]`,
`prompt_lens_cpu=[7]`. A first build that asserted "no cached prefix"
had the assert fire on its third probe request. **Implementing only the
fresh-prefill context flavor is not sufficient under defaults.**

**The cached-KV metadata surface is conditional, not merely False.** With
reuse on, the step object carries `enable_context_mla_with_cached_kv`,
`ctx_cached_token_indptr`, `ctx_kv_indptr` (both int64 CUDA),
`ctx_uncached_token_indptr`, `max_ctx_seq_len`, `max_ctx_kv_len` and
`num_ctx_cached_tokens`. With `enable_block_reuse: false` **those
attributes do not exist at all** — so a startup contract check that
asserts their presence turns a legal config into a hard failure. Key the
flavor on presence, and keep both branches.

Two things that make this cheaper than it sounds:

- **The cached path needs no extra glue.** `kv_lens_runtime` and
  `prompt_lens_cpu_runtime` already satisfy the no-append call's
  `sequence_length >= context_lengths`, and `host_total_kv_lens[0]` is
  exactly the gather's row count. No cumsum, no host bookkeeping.
- **One path serves both.** append → gather → up-project → explicit-K/V
  is correct at `cached_s == 0` too. That is what lets a single batch mix
  reused and fresh sequences without a second code path.

## Sizing facts that surprise

**The attention workspace is an order of magnitude above the certified
figure.** `thop_attention`'s contract records **33 MB to ~109 MB** across
its certified shapes — the peak being an `H = 128` MLA context call over
545 tokens (113,924,096 B) — while an MLA engine at `max_num_tokens=8192`
resized it to **531,628,032 B (507 MiB)**. Budget for it before blaming a
KV-pool setting for missing memory. The contract's figure tracks
tokens x heads, so it grows with the head count a target actually runs;
read the current number there rather than a remembered one.

**An fp8 latent pool adds a workspace surcharge, because the quantized
operands are staged there.** Measured on the no-append context flavor at
`H = 128`, identical shape (66 q rows over 193 K/V rows): **52,816,640 B
under fp8 against 43,288,832 B under bf16**. An engine sized from bf16 MLA
figures under-budgets on an fp8 KV cache.

**First-boot JIT is a visible cost, once.** Eight `HQk576HV512` decode
kernels NVRTC-compile at ~5.6–7.2 s each — ~47 s of the first engine
build, with no on-disk cache. Context kernels are precompiled. Exclude
first calls from any timing.

**`q_lora_rank` may legitimately be absent.** A checkpoint with
`q_lora_rank: null` uses a direct `q_proj` and no `q_a_proj` /
`q_a_layernorm` / `q_b_proj`. Passing `q_lora_rank=0` to the ops is
accepted and inert, and the contracts now enumerate it —
`thop_attention.md`'s geometry table lists `1536` and `0` for MLA, and
`mla_rope_generation.md` carries its own `q_lora_rank` row.

Inert is measured, not assumed, and it had to be measured **twice**: the
sweep (`1536 → 0 → 1536 → 4096` over identical inputs, bitwise equal on
every observable) runs at `H = 32` and again at `H = 8`. One run would not
have covered the other, because the MLA generation phase compiles a
*different* decode kernel per head count at the low end —
`...VarSeqQ8Kv128...` at 8 heads where 16 and 32 both take
`...VarSeqQ16...`. An argument shown inert on one compiled kernel is not
thereby inert on another. A tensor-parallel MLA target slices heads, so
it is the head count the sweep did **not** run at that it will execute.

## What the structure dictates, and what it makes irrelevant

Established by measurement during the perf campaign; these are properties
of the mechanism, so the next MLA target can skip the sweeps.

**KV capacity does not bind, and it is not close.** The latent cache is
`num_layers * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes` per token
— one shared latent, no head multiplicity, no separate V — which here is
`30 * 576 * 2 = 34,560 B/token`. The same geometry as per-head MHA
(`2 * 30 * 32 * 128 * 2`) would be 491,520, so the latent form is
**14.2x cheaper per token**. A 30-layer model on one B200
got a 142.25 GiB pool = **4,419,680 tokens**, against a 256-request
ceiling at ISL+OSL 2048 — two orders of magnitude of slack.
`kv_cache_config.free_gpu_memory_fraction` and the capacity scheduler
policy are inert. Do not spend sweeps on them. (The same conclusion was
reached for a different reason on a sparse-MoE target — see
`expert-weight-packing.md` — so it now holds for two unrelated
structures.)

**`tokens_per_block` matters, but not for the reason the kernel list
suggests.** Raising it 32 → 64 was worth +22% at concurrency 256. The
MLA FMHA kernel does have page-size-specific variants and was 27.4% of
GPU time, so the obvious story is a better kernel. That story is wrong.
Tying two knobs isolated it: `tokens_per_block: 64` and
`max_seq_len: 2048` reach the *same* `max_blocks_per_seq` by different
routes and measured identically (+24.8% vs +25.7%). The win is
**host-side KV block bookkeeping**, which scales with
`max_blocks_per_seq = max_seq_len / tokens_per_block` — not the kernel
variant. Prefer the workload-independent knob.

**Decode is host-bound, not roofline-bound.** An nsys capture over 100
pure-decode iterations put GPU idle at **0.401** against a 0.15
generation threshold, with the forward fully graph-launched
(`cudaLaunchKernel` only 0.9% of the window). The exposed cost was KV
block bookkeeping — `prepare_resources` 3306 µs/step,
`_prepare_inputs` 2443 µs/step. A decode step that reads ~16.9 GB in
11.37 ms is running at ~1.5 TB/s against a ~7.9 TB/s B200 roofline. On
this mechanism, look at the host before the kernels.

## An fp8-e4m3 latent pool

Established by the DeepSeek-R1-0528 NVFP4 dep4 target, the first on a
quantized latent cache (`hf_quant_config.json`'s `kv_cache_quant_algo:
FP8`), and by the five catalog certifications it needed. Everything above
describes the bf16 pool; this section is only what changes.

**The pool is exactly half as wide, and a target declares nothing.** The
per-token width becomes `num_layers * (kv_lora_rank + qk_rope_head_dim) *
1` — measured **35,136 B/token** at 61 layers (`61 * 576`), against
`* 2` in bf16. Every op derives the element width from **`quant_mode`
alone**, because the pool arrives as a raw pointer and nothing else
describes it. That is also the failure mode: a `quant_mode` disagreeing
with the pool's real dtype is silently addressed at the wrong stride, in
either direction.

**Derive `quant_mode` from the checkpoint's quant config; do not hand-code
the fp8-KV bit.** Bits outside the KV-cache group ride along unread —
`1152` (`FP8_KV_CACHE | FP8_1x128_128x128`) and `384` (`| FP8_QDQ`) are
bit-identical to a bare `128` on every MLA flavor and on the gather.

**The workspace grows, and by more than the certification figure
suggests.** An engine at `H = 128` / `max_num_tokens = 8192` sized the
attention workspace to **2,126,512,128 B (1.98 GiB) per rank** — about 4x
the 531,628,032 B the bf16 MLA engine above reached. At the op level the
same asymmetry is visible at ~10x smaller scale (52,816,640 B against
43,288,832 B for an identical no-append shape), because the quantized
operands are staged in the workspace. Size from the fp8 figure.

### The two phases divide the quantization labour *oppositely*

This is the part nothing in either op's signature says, and it is the
easiest thing to get backwards.

| | context | decode |
|---|---|---|
| who quantizes the query | **the FMHA itself**, in-kernel, at scale 1.0 | **the preprocessing op**, into `quant_q_buffer` |
| what the preprocessing op writes | plain **bf16** `q` (an ordinary rotation) | `quant_q_buffer` + `mla_bmm1_scale` + `mla_bmm2_scale` |
| are the kv scale tensors read by the FMHA | — | **no, neither** — the dequantization must be folded into the two bmm scales by whoever produces them |

So `q` is quantized exactly once on the context path, and the decode path
needs `mla_bmm1_scale = [x, x*log2(e)]` with
`x = r²/(q_scaling * sqrt(nope + R))` and `mla_bmm2_scale = [r]`. The FMHA
reads `mla_bmm1_scale[**1**]` — element `[0]` is inert.

**On an fp8 pool the decode producers are ordered, not concurrent.** The
preprocessing op does **not** write `fused_q` there; it *reads*
`fused_q[..., :C]` to build the quantized query. The absorbed-q BMM must
therefore have finished before it.
`.claude/skills/trtllm-modeling-v2-kernel-onboard/references/decompositions/mla-custom-op.md`
§4.2 carries both cases — following the bf16 "disjoint halves may overlap"
reading on an fp8 pool is a silent race whose symptom is a quantized query
built from uninitialized memory.

### Three things that will look like defects and are not

* **A cached prefix pays fp8 twice.** The engine dequantizes cached latent
  rows off the pool, projects them through `kv_b_proj`, and the context
  FMHA quantizes the result straight back to e4m3. No argument turns it
  off. A cached-prefix context call therefore carries strictly more
  quantization error than a fresh prefill of the same tokens — account for
  that before attributing an accuracy gap to block reuse.
* **Nothing validates the round trip at any layer.** The write scale, the
  read scale and the folded bmm scales are independent roles and no op
  checks any relation between them. A pool written at `w` and read at
  `r != 1/w` is silently mis-scaled, never an error.
* **The scale must be 1.0, and upstream cannot reach anything else
  anyway.** The fp8 MLA *context* path is internally inconsistent at
  `s != 1.0` in both its flavors — it quantizes at 1.0 but applies `s²`
  and `s` as if it had quantized at `1/s`. Every engine call site
  hard-codes the factor to 1.0, so this is unreachable from stock trtllm
  and reachable only from a direct op caller, which a target is. Check the
  checkpoint: DeepSeek-R1-0528-FP4's 124 `k_scale`/`v_scale` tensors are
  all 0-dim fp32 scalars equal to exactly 1.0.

**JIT cost.** First boot compiles **8 fp8 MLA decode kernels per rank** at
~6-7 s each — the `QkvE4m3` siblings of the bf16 names (`QkvE4m3O
Bfloat16HQk576HV512...VarSeqQ16Kv128...ForGen` plus `MultiCtasKvCga` /
`HVPerCta256`). No MLA *context* call compiles anything, either pool.
`H = 128` takes the same `VarSeqQ16` q-tile as 16 and 32 — the tile stops
growing at the low end — but still pays its own compile.

**Page size.** Only **32** is certified over an fp8 pool, across all four
MLA entries; their bf16 columns also carry 64. That matters because page
64 is a real perf lever elsewhere (+22% at con=256 on a tp1 sibling), so
on an fp8 target it is a certification request rather than a knob.
