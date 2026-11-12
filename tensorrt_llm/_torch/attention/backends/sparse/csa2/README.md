<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Compressed Sparse Attention 2 (CSA2)

This package implements DeepSeek-V4.1 CSA2 attention, indexing, cache ownership
and runtime metadata. The V4.1 model adapter integrates these components with
causal encoder/decoder (CED) replay and embedded DSpark. Model registration and
orchestration live in the model and executor modules.

## Backend and hardware routing

`CSA2TrtllmAttention` owns the shared cache/indexer preparation. The module
invokes its composed compute helper on the applicable hardware; the native
provider inherits `TrtllmAttention.forward`. There is no CSA2 FMHA registration
or controller, and enabling the dense PrimTS FMHA library does not select CSA2.

| Architecture | Default module compute path | Computation |
| --- | --- | --- |
| SM100/103/107 | `CSA2PrimsTS` | PrimTS sparse MLA reading packed FP8 SWA / NVFP4 MAIN |
| SM100 family, native provider | `CSA2TrtllmAttention.forward` | Native trtllm-gen dynamic sparse MLA |
| SM100 family, native provider, generation tiles with 64 query heads | `CSA2TrtllmAttention.forward_fused_decode` | Fused sparse decode kernel |
| SM103, native provider, E4M3-staged generation tiles of ratio-2 layers with 64 query heads and at least 1024 rows per rank | `CSA2TrtllmAttention._forward_gen_fmha` | `gen_fmha` sparse MLA core |
| SM120/121, checkpoints without candidate consumers | `CSA2FlashInfer` | FlashInfer BF16 FA2 |
| SM90, checkpoints without candidate consumers | `CSA2FlashMLA` | FlashMLA sparse BF16 |

A checkpoint with a `candidate_source_layer_id` (the released one has candidate
consumers at layers 24/28/32/36) needs an SM100-family GPU, a DeepGEMM build with
the sparse MQA-logits kernels, 32 index heads of 128 dimensions and a
`candidate_block_size` that is a multiple of 8; model construction rejects any
other combination with an error naming the missing requirement.

`CSA2Params.compute_backend` accepts `auto`, `prims_ts`, `trtllm`, `flash_mla`
and `flashinfer`. An explicit internal provider takes precedence over
`TRTLLM_CSA2_COMPUTE_BACKEND`; otherwise that environment variable overrides
hardware defaults. Resolve the choice when constructing the model. For example,
set `TRTLLM_CSA2_COMPUTE_BACKEND=trtllm` for native A/B testing or rollback.
`auto` plus packed attention or explicit `use_fp8_staging=True` keeps the native
provider on SM100-family GPUs. Forced PrimTS rejects these native-only options
and GPUs other than SM100/103/107 rather than silently falling back.

The PrimTS default favors `kv_cache_config.dtype="fp8"`. The other KV dtypes
(`auto`, `bfloat16`, `fp8_ds_mla`) store SWA as the 528-byte scaled records, and
for those the native fused decode kernel serves generation tiles of 64 query heads
about twice as fast as PrimTS on SM100, SM103 and SM107 (standalone, 192 to 2,048
query rows). Such deployments can select it with `TRTLLM_CSA2_COMPUTE_BACKEND=trtllm`.

`CSA2PrimsTS` passes persistent packed caches directly to the managed PrimTS
sparse MLA kernel through its `kv_producer` extension. The CSA2 producer reads
SWA `uint8[N,512]` for explicit `kv_cache_config.dtype="fp8"` (unit-scale E4M3),
or `uint8[N,528]` for the legacy format (E4M3 plus group-32 UE8M0 scales), and
MAIN `uint8[N,288]` (packed E2M1 plus group-16 E4M3 scales), including padded
outer strides. The format is explicit; padding never determines its encoding.
Aligned payload chunks use vector loads; misaligned rows use bounded byte loads.
The producer folds persistent group scales into kernel-local E4M3 K/V operands;
it allocates no requantized KV bank, and captured context phases fill no decoded
bank either. Each context query selects up to `index_topk` MAIN rows
from its request's GLOBAL rows, so decoding selections converts the same records
many times. An eager context phase instead reads its MAIN rows from a
unit-scale E4M3 `uint8[rows,512]` bank laid out by the native shared tables'
main offsets, which `_prims_ts_bank.py` decodes with the producer's own
conversion. A phase whose selections cover at least its requests' MAIN rows
decodes all of them. One that selects fewer, such as a bounded-replay tile of a
long context, decodes only the rows it keeps: the compact metadata marks each
kept bank row in a retained, zeroed `uint8[rows]` vector, and the decode
converts each marked row once and clears its mark. The compact metadata
addresses bank rows instead of packed records, with the same validity, and the
kernel copies those rows as it copies ordinary FP8 SWA rows. Each decode warp
owns eight bank rows, whose pool records its lanes resolve together, and reads
four rows at a time before converting them. The bank is decoded on the model
stream after the layer's MAIN publication; its storage and marks are retained,
grow by powers of two up to 64 MiB of bank and are part of the PrimTS workspace
reservation. Larger phases keep decoding selections in the kernel.
Generation tiles with FP8 SWA whose gather schedule applies decode their
selections into a selection bank instead, one unit-scale E4M3 `uint8[512]` row per
compacted selection. Query rows of one request may select the same MAIN row: the
compaction then also claims an epoch-tagged INT64 owner slot per kept pool row
(never cleared, since each call claims with the next epoch), and
`dedup_main_records` decodes each selected row once, into the bank row of its
smallest selection, which all of that row's selections read. A layer whose
selections and MAIN rows are final before its projections (a Reuse layer of a
generation-only forward) compacts and decodes its selection bank on a side stream
beside its projections and KV write; other layers do both on the model stream
after their MAIN publication.
External BF16 Q is converted at unit scale into preallocated E4M3 Q scratch,
except in eager context phases of at least 1024 tokens on the fused MXFP8 prefill
Q projection: its Q-B GEMM applies RoPE and rounds the BF16 Q RN/satfinite to E4M3
at unit scale, and PrimTS reads that Q directly. A context-only batch then writes
no BF16 Q at all. The
native E4M3 QK/PV path uses FP8 MMA with FP32 accumulation, E4M3 softmax P and
BF16 output.
The packed profile uses single-pipe M64/BK128 Keep-AB with Q2/KV8 stages and
masked head tiles for 1–128 query heads. Native PrimTS scheduling, joint softmax,
sinks, split-KV reduction and epilogue are reused; ordinary PrimTS providers
and managed vendor source are unchanged. Grids of at most one resident wave use
the native nonpersistent launch with KV and V splits; larger grids, whose tiles
have one split and one V owner, use the same family's static persistent
scheduler, which overlaps each tile's prologue and epilogue with its CTA's next
tile and also stays within CUDA's 65,535-row grid-Z limit. As in the native
Keep-AB policy, gather issuers share one cached copy of each tile's sparse
coordinates and correction warps rescale paired rows.

The producer is needed only for rows that must be decoded. With FP8 SWA and, for
MAIN, a decoded bank, every selected row is already its unit-scale E4M3 operand,
and on persistent grids whose CTAs each own more than four KV tiles
`_prims_ts_packed.py` instead launches the native
PrimTS Keep-AB kernel with its default TMA producer: each `gather4` copy brings
four selected rows of a D128 slice into the swizzled operand stage, and the
kernel runs the same profile. Its results are bitwise identical to the
producer's. Gather4 reads the pools as dense `uint8[N,512]` rows and fills rows
beyond a pool with zeros instead of masking them, so it requires the
`valid_prefix` plan promise that every entry before a live length is a valid
row of its pool; CSA2's compact metadata provides it, and the kernel then
derives masks from the lengths. Pools with other row strides, plans without the
promise, and smaller grids, where the producer's copies are faster, keep the
producer. Q and K/V use unit
tensor scales, as flashinfer's DeepSeek V4.1 PrimTS decode does: after the
query-ready dependency, BF16 Q is rounded RN/satfinite to E4M3, and the producer
decodes each selected SWA or MAIN row with its persistent group scales into
RN/satfinite E4M3 operands, so magnitudes beyond 448 saturate. The softmax and
output scales are the host FP32 launch arguments `1 / (q_scaling * sqrt(512))` and
one, which a captured CUDA Graph replays unchanged; sinks are not rescaled and the
native P normalization already cancels its factor of 448. The helper validates the
softmax scale before cache publication or indexing. FP8 rounding is not equivalent
to the unquantized reference and requires workload-specific accuracy validation.

Packed query metadata retains geometry without native staging pools. Its GPU
metadata kernel compacts SWA and MAIN selections independently and resolves
logical MAIN rows through the live owner page table, request IDs and visibility.
Physical row zero is valid; `-1` denotes padding. Duplicates remain distinct
attention entries, and source/pool bounds are checked before packed loads.
Compression, indexing and CED publication use the same ownership and lifecycle
for both SWA formats. Persistent scales, sinks and the configured softmax scale
are each included once;
the normal BF16 output projection owns inverse RoPE.

FP8 context tiles that stage the shared E4M3 bank with 64 query heads per rank
run `torch.ops.trtllm.csa2_sparse_context` (`cpp/tensorrt_llm/kernels/csa2SparseContext`)
instead of the trtllm-gen context FMHA, which serves every other tile. Staging
is unchanged; the kernel reads FP8 Q, or quantizes BF16 Q with the staging scale
on load to the bytes the context FP8 quantizer writes, and follows the trtllm-gen
FP8 softmax (P rounded to E4M3 after scaling by 448, FP32 row sums, sink in the
denominator).

Generation steps of at most 16 query rows per rank can instead attend the selected
rows straight from the persistent caches (`direct_decode.py`): one CuTe DSL launch
decodes the SWA rows (528-byte scaled or 512-byte unit-scale E4M3 records) and main
rows exactly into BF16 shared memory and attends them with the BF16 query and the
sink, so neither the FP8 staging nor the native FMHA runs, and PrimTS skips its
selection compaction and split-KV reduction. The backend decides per tile in
`direct_decode_applies` (generation-only forwards without context tokens,
SM100/SM103/SM107, a geometry that fits one CTA's shared memory): native FP8 staging
takes it with 528-byte scaled SWA records and PrimTS with either SWA format; every
layer of such a forward takes it. Larger native batches keep the format-aware staging
path unless the fused decode below takes them, native payload-only FP8 SWA always
stages, and larger PrimTS batches run PrimTS.

Other generation tiles with 64 query heads per rank over 528-byte scaled SWA
records, except the `gen_fmha` tiles below, do not stage either:
`forward_fused_decode` runs `torch.ops.trtllm.csa2_sparse_decode`
(`cpp/tensorrt_llm/kernels/csa2SparseDecode`), which gathers each query's
selected SWA and main rows from the packed pools, maps logical main positions
through the owner's page table, dequantizes the rows exactly to BF16 and
attends with the sink in FP32. Its split-token scratch and arrival counters
are per device, so fused decode calls on one GPU run in stream order. In a
generation forward of more than 16 rows whose o_a block scales are powers of
two, its epilogue writes the grouped MXFP8 o_a projection's input instead of
BF16 (inverse RoPE and MXFP8 quantization as
`projection.inverse_rope_mxfp8` does them).

On SM103, a generation tile of a compression-ratio-2 layer with at least 1024
query rows per rank (`_GEN_FMHA_MIN_ROWS`), 64 query heads, E4M3 staging and at
most 640 staged rows per query (128 SWA rows plus up to 512 main rows) is staged
instead and attended by the `gen_fmha` core (`gen_fmha.py`, the CuTe DSL kernel
in `attention/kernels/blackwell/sparse_mla/`). Its independent-query staging
allocates both pools as one bank (`CSA2TrtllmMetadata.gen_fmha_bank`) and
publishes each query's indices as rows of that bank; the core, launched with PDL
after the staging, reads the same FP8 query, BMM scales, sink and staged E4M3
rows as the native generation kernel but uses its own softmax (lazy rescale,
E4M3 P). With eight local o_a groups it writes the grouped MXFP8 o_a input
instead of BF16, as the fused decode does. The module asks `gen_fmha_serves`
before projecting and the backend decides per tile in `gen_fmha_applies`, from
what a captured graph fixes; ratio-1 and SWA-only layers and smaller tiles keep
the paths above.

Decode work of the indexer that does not depend on the layer's own projections runs on
CSA2 side streams: a Full layer of a forward without context tokens projects its index
weights on one and refreshes the paged indexer descriptors on another while the auxiliary
stream runs the compressor (the indexer joins them through `CSA2ForwardState.index_ready`),
and the candidate source publishes its candidates and builds its consumers' sparse logits
schedule on the first, next to its own TopK (consumers wait for both). Every kernel and
input stays the same, so results are bit-identical.

`backend.py` contains the single sparse backend and both plain Flash helper
classes; `prims_ts.py` contains the PrimTS helper, `_prims_ts_packed.py` its host
plan/launch adapter, `_prims_ts_producer.py` its packed KV producer,
`_prims_ts_bank.py` its context MAIN bank decoder, and
`_prims_ts_quantization.py` its unit-scale E4M3 Q staging.
The separate opt-in custom packed kernel uses the backend's explicit
`forward_packed` method, not the PrimTS helper. The module issues one attention
call per real phase (all context queries, then all generation queries) and every
path invokes the same sparse preparation once per phase. Staging/indexer GPU
kernels are collected in `kernel.py`; PrimTS metadata preparation lives in
`_prims_ts_metadata.py`.

Persistent SWA storage follows the original `kv_cache_config.dtype`, independently
of the compute backend and its staging precision:

| KV-cache dtype | D512 persistent SWA row |
|---|---|
| `fp8` | 512 E4M3 payload bytes, unit persistent scale, no scale footer |
| `fp8_ds_mla` | Legacy MXFP8: 512 E4M3 payload bytes + 16 UE8M0 scale bytes, group size 32 |
| Omitted / `auto`, or `bfloat16` | Legacy 528-byte format |

The format is fixed for a cache manager's lifetime and must agree with its
attention backend before publication. `QuantAlgo.FP8` alone does not select
ordinary FP8 storage. MAIN/NVFP4 and INDEX layouts do not depend on this choice;
projection quantization is separate as well. Transfers require matching SWA
layouts and do not convert between formats.

Native trtllm-gen and Flash helpers decode selected SWA/main rows into bounded
staging pools. Their staging policy is independent of persistent SWA storage.
Native staging is BF16 unless FP8 is asked for:
with `fp8` or `fp8_ds_mla` (or an FP8 KV-cache algorithm in a directly
constructed backend's quantization config under `auto`), native trtllm-gen
(or, for its tiles, the `gen_fmha` core) consumes E4M3 staging.
`CSA2Params.use_fp8_staging` overrides that choice --
left unset it follows the native path's own support, `False` pins BF16, and
`True` demands E4M3 and rejects a configuration that cannot serve it rather
than falling back. The E4M3 per-tensor KV scale is range-aware: every forward
reduces the selected rows' persistent group-scale bounds, or the unit bound for
ordinary FP8 SWA, to one ceiling on what they can decode to, then picks the
smallest power of two mapping that ceiling onto 448. A power of two costs no relative precision because E4M3's
relative error is the same in every binade, and a unit scale used to clamp
every channel above 448 (observed on image tokens at a chunked-prefill
boundary). Generation buys that range from Q, which carries the reciprocal
scale so BMM1's dequant product stays exactly one. The in-tree split-KV
reduction no longer needs that product to be one -- `fmhaReduction.cu` reads
the device `bmm1_scale` the main kernel reads -- but the identical correction
factors also live inside the closed cubins serving the other two
`MultiCtasKvMode` values, so the reciprocal stays until those are fixed.
Context cannot pay the same way, because `attentionOp.cpp` points
`dequant_scale_q` and `dequant_scale_kv` at one tensor: Q is divided by the
same scale as KV instead, which cannot saturate it and only costs resolution
underneath, so context takes range up to Q's own amax and the three binades
that de-saturate the main pool. Split-KV plays no part in either bound: every
MLA multi-CTA-KV cubin is a generation kernel. FP8 staging is only available on
trtllm-gen without custom packed attention. FlashMLA and FlashInfer keep BF16
staging even when an FP8 cache dtype is requested. PrimTS accepts BF16 queries,
quantizes Q to E4M3 internally, and reads persistent packed records without KV staging. None of these helpers
forwards an FP8 KV algorithm to the inherited quantization state. Any other KV-cache
quantization algorithm (for example NVFP4) is rejected because it has
no CSA2 meaning. Native trtllm-gen does not consume the persistent CSA2 FP4
format directly, and FlashInfer likewise avoids requantizing into V4's
different FP8 footer cache format. The library adapters return BF16;
unsupported custom masks and scaled/quantized output contracts are rejected.

Eager native context preserves real request query groups. Its virtual staged
KV coordinates prevent the native causal mask from clipping already selected
rows; actual source causality remains encoded in selection and visibility.
When the selected rows of a context phase would exceed the union of its
requests' visible SWA/GLOBAL rows, that phase decodes each logical row once
into a shared row bank and attention indexes the bank directly instead of
expanding rows per query. Fixed-shape native context graphs, and native
generation tiles that neither the direct nor the fused decode takes, use
independent-query staging with fused compaction/dequantization kernels (one bank
for the `gen_fmha` tiles).

`CSA2SparseAttentionConfig(algorithm="csa2")` selects the cache manager through
the existing sparse registry. Geometry comes from the checkpoint text
configuration. `ModelConfig` recognizes `deepseek_v41`/`deepseek_v41_text` for
this attention configuration. The V4.1 model adapter selects this configuration
and delegates attention to the independent CSA2 module.

## Projections, compression and indexing

`DeepseekV41Attention` owns projections, normalization, RoPE, compression and
grouped output projection. It consumes prepared `CSA2TrtllmMetadata`, retains
CSA2's omission of projected-Q head normalization, and derives index K from
the compressed main latent before main-KV RoPE. Ratio-two compression reuses
the native V4 non-overlap compressor with FP32 value/gate state and zero APE
and, like V4, projects values and gates with the checkpoint's fused `wkv_gate`
weight. Ratio one has an uncompressed global cache and no compression gate.
Normalization uses the shared `RMSNorm` module; index head weights are FP32,
except that candidate consumers project BF16 weights, the dtype their sparse
MQA logits read.

BF16 projection execution is the default. Optional
`projection_quantization="mxfp8"` uses native MXFP8 linear/grouped output
operations on SM100/SM103, with checkpoint scale validation and support for
the admitted 32/128-block weight layouts. Eligible small-query MXFP8 index-Q
projection can fuse GEMM, interleaved RoPE and CSA2 nearest-even FP4 conversion
with `fuse_index_q`; larger queries retain the unfused native projection path.
An optional auxiliary stream overlaps compression/index preparation with Q
work when the existing multi-stream policy is enabled.

`CSA2Indexer` subclasses DSA `Indexer` in projection-free mode and reuses its
MQA kernels and `TopK` module. CSA2-specific phase orchestration, candidate
loading and logical mapping remain local. Indexing runs once for the complete
layer batch before attention tiles consume the results. Full prefill gathers
cached and new keys once per request chunk; bounded query tiling limits
logits workspace. TP query splitting uses the shared partition/allgather
primitives and synchronizes candidate-source outputs as well as selections.

SM100 decode uses native paged FP4 MQA for supported index-head geometry,
including padding smaller head counts into native specializations. The index
buffer is read in place: metadata derives a native block table from the owner's
GLOBAL page table once per owner and forward, unallocated pages read page zero
and are masked by one validity byte per page, and decode TopK selects directly
over the paged logits. Candidate consumers score only their published candidate
blocks with DeepGEMM sparse MQA logits: straight from the index pages for decode
with the exact radix TopK, which selects over the kernel's BF16 logits as they
are, and over each context request's gathered prefix for prefill; they require
SM100 and 32 index heads of 128D rows (see the hardware note above). Decode
batches of more than 64 rows on SM100/SM103 select the same positions by ranking
groups of 8 columns first
(`kernels/blackwell/paged_mqa_logits/sparse_logits_topk_dsl.py`): every
selected logit lies in the top_k groups with the best smallest key, so only their
8 * top_k logits are ranked. Reuse layers consume logical selections from their
index source and resolve current physical pages without recomputing indexer logits.

CuTe DSL kernels in `kernel.py` handle the decode selection glue: one derives
the native block table, visibility and page validity of an owner from its
page table, one masks unbacked pages in the paged logits, one validates and
sorts the selected rows in place, one maps candidate consumers' Top-K columns to
logical positions in place, and the candidate source publishes its block
hierarchy (block maxima over backed positions, pinned newest block, Top-K blocks
expanded to ascending positions and DeepGEMM sparse block ids). The TopK module
sits between the mask and the finalize launch, so exact, self-sampling and
temporal GVR TopK all select over the same masked scores; a temporal prior that
lands on a masked column is dropped before the GVR kernel seeds its threshold.

On SM100/SM103, split-row decode batches without temporal GVR hints skip that
chain from more than 64 rows on the candidate source (of 8-position candidate
blocks) and from 256 rows on other Full layers: one `block_select.py` launch
selects from the scorer's maxima of every 32 visible positions and, on the
candidate source, also publishes the candidates. It ranks 32-position blocks,
then 8-position blocks, then positions, ties lowest index first at every level,
so it selects the masked TopK's positions up to the choice among exact ties at
the k-th logit (and the publication's blocks up to ties at the k-th block).

TopK is exact by default: eligible prefill of at least 8192 score columns uses
the single-pass self-sampling engine, narrower prefill and decode the radix
TopK. `use_self_sampling_topk` alone selects that prefill engine, independently
of `enable_heuristic_topk`, which selects GVR decode. `use_cute_dsl_topk`
selects the CuTe DSL radix kernels over the CUDA radix TopK on SM100-family
GPUs: the exact TopK, the candidate source's cluster block TopK for up to 64
rows, and the fused decode selection of up to 16 rows (`decode_select.py`),
which also stands in for the self-sampling GVR decode and breaks ties at the
k-th logit by the lowest column. Short sequences whose keys all fit can
enumerate visible positions without logits or TopK. Internal `CSA2Params` also
exposes eligible CuTe DSL exact TopK, self-sampling/temporal GVR decode, and CuTe
paged MQA/emission options. Temporal
priors use request allocation epochs and logical positions, seed first decode
from accepted prefill, and follow request identity across reordering. Rewind
and multi-query verification invalidate unsafe hints. Emission state resets
before replay when row ownership changes and on target/draft transitions. Candidate-restricted Reindex does not consume temporal hints from
an incompatible selection domain.

## Cache lifecycle and speculation

`CSA2CacheManager` specializes `KVCacheManagerV2` and reuses allocation,
commit, prefix reuse, copy-on-write, scratch, release and tier storage. Every
attention layer has private SWA storage; only KV-source layers own global
main/index storage, and only ratio-two owners allocate FP32 compressor state.
Packed pool views and the fused decode's captured TMA descriptors keep the pool
sizes of construction, so the manager rejects
`kv_cache_config.enable_kv_pool_rebalance`. Without a `pool_ratio`, the
SWA/GLOBAL split is sized for the GPU quota the pools receive (after the
`max_tokens` cap and the cross-rank minimum).

Each global position owns a 288-byte main record and a 68-byte index record,
stored as two buffers of one GLOBAL page group. Main uses NVFP4 E2M1 values
with E4M3 scales per 16 channels; index uses E2M1 values with UE8M0 scales per
32 channels. Both buffers share identical physical page numbering and one page
lifecycle, so copy-on-write and transfer preserve both payloads and scales
together. The index buffer uses the native page-footer layout of the paged FP4
MQA-logits kernels (`[pages, 64, 1, 68]`: 64x64 packed data bytes then 64x4
scale bytes per 64-row page), so decode reads it in place through a block table
derived from the GLOBAL page table, as the DeepSeek-V4 indexer reads its own
index cache. The published layout uses 890 global bytes per original
token, excluding SWA, compressor state and compute workspace. SWA uses 512-byte
unit-scale E4M3 rows for explicit `fp8`, or legacy 528-byte MXFP8 rows for
`fp8_ds_mla`, `auto` and `bfloat16`. All formats include their RoPE channels.
These persistent formats are part of CSA2's cache contract and do not require
a generic NVFP4 KV-cache flag. Projection precision and the selected attention
compute path do not change the persistent main-cache encoding.

Long prefill uses scratch-aware page mappings. Partial groups retain raw FP32
values and scores. CUDA BF16 cache publication on SM100 uses fused exact-byte
CuTe DSL quantize/scatter kernels (`kernel.py`), including the split
data/scale packing of index queries; other paths use the encoder and masked
scatter. Invalid slots, including padding slot -1, never write a cache row.
Native staged attention uses CuTe DSL gather/dequantization on SM100
to read selected packed rows directly into the BF16 or FP8 staging pools,
including strided main/index views; the staging kernels resolve logical main
selections through the owner's live page table. Invalid read slots produce
zero rows. The packing and
unpacking helpers in `quantization.py` remain the reference fallback for
unsupported gather layouts or hardware.

Attention-local contiguous chain verification supports accepted-prefix rewind
through V2 resource updates. SWA/state windows and scratch rewind capacity
cover the configured draft tail; continuation recomputes incomplete groups
and cannot expose rejected compressed rows. Target/draft metadata hooks
support identical explicit layer layouts and independent cache-dependent
state. Non-linear trees, beams, explicit token relocation indices and implicit
virtual draft-layer mappings remain rejected. These local mechanisms do not
establish complete model-level MTP drafting or generation. Embedded DSpark uses
the separate model and worker integration.

## GLOBAL-prefix SWA reconstruction

After an authoritative cached GLOBAL prefix of length C, the caller can use
`get_swa_replay_ranges()` to plan writable query intervals and
`set_swa_bounded_replay()` to prepare their reconstruction. Encoder replay
starts at `max(0, C - window_size)` and may include an uncached suffix; model-level
Encoder recovery uses the larger Decoder capture window when required by DSpark. Each
query's SWA is restricted to the replay segment. This reconstruction is
approximate across layers; its reference is truncated replay, not a complete
historical forward.

Cached main/index records remain read-only. Only uncached source rows produce
new GLOBAL entries. At an odd ratio-two boundary, the last cached raw token
is included to reconstruct partial compressor state before continuation.
Pure replay with no required source rows skips global projection. Decoder
replay reconstructs only the final window and requires prompt GLOBAL entries
to be already prepared.

Replay preparation is one-shot. The caller supplies the declared query inputs
and provisions every writable row in the returned interval; an ordinary
retained SWA window may omit its first reconstruction token. Captured replay
can refresh same-geometry positions and page mappings, while changes to replay
mode or source geometry require fresh metadata and recapture.

Model-level Encoder recovery uses the OPTIONAL reuse groups described below.
The cache manager selects reusable groups during normal request preparation;
missing Encoder state is recovered through the model layer loop. There is no
separate native reconstruction acknowledgement or per-layer completion event.

## Model decoder-suffix replay

The CED prepass applies the boundary layer's collapse, normalization and
GLOBAL/index projections to the complete encoder states before narrowing
decoder queries. The query pass preserves those cache entries and still
executes its attention and FFN. Replay is enabled by default with a shared
Encoder/Decoder range of 140 tokens (`DEFAULT_BOUNDED_REPLAY_TOKENS`), or one
token more than the required SWA window (including the embedded DSpark capture
window) when that window is wider, without a physical-ring or page-alignment
constraint. `TRTLLM_V41_BOUNDED_REPLAY_TOKENS` overrides the range; a value that
does not exceed the required window is an error. The released checkpoint's SWA
window remains 128.
This policy is approximate across stacked
SWA layers and requires accuracy validation. Set
`TRTLLM_V41_DECODER_BOUNDED_REPLAY=0` for full decoder prefill. Model defaults
enable chunked prefill and SWA scratch reuse, and disable prefix block reuse;
explicit user settings take precedence. Scratch remains allocated for the full
chunk, including Decoder layers whose queries are skipped. Compatible layers
within a lifecycle share scratch; different reuse policies keep separate scratch
allocations. Retained replay-window KV stays in ordinary pages and survives the
Decoder metadata rebuild. A normal page reserved for DSpark also remains in use
if its block enters a later chunk's scratch range; only blocks without normal
pages use scratch. Full-context logits and additional-output requests run
every context row and can use scratch in both model halves.

Ordinary text requests execute the intersection of each context chunk with
the final prompt window `[max(0, prompt_end - window), prompt_end)`. Earlier
chunks publish complete Encoder GLOBAL/index KV without Decoder queries. A
window spanning several chunks continues through normal Decoder SWA pages;
its read floor stays at the same absolute position across those chunks.
Whole-prompt endpoints are frozen during input preparation, before overlap
scheduling can advance the requests. Direct calls without endpoints retain
the last `min(chunk_length, window)` rows of each chunk. Warmup and dummy
context forwards use the same preparation path as serving.
PRIVATE Decoder storage is determined by the cache manager, not by the presence
of executor requests; requests provide prompt endpoints, output requirements
and recovery floors. Warmup passes its dummy requests through the normal input path.
The planner always supplies SWA floors from the retained query interval and
recovery state, independently of PRIVATE storage. The Decoder boundary consumes
these floors directly without inferring them again.
Ordinary and speculative requests share the standard model layer loop and
decoder replay enter/exit hooks. A forward-local plan maps decoder token rows
back to the encoder batch without removing requests. Empty local Decoder
inputs skip attention and mHC, but still participate in every MoE collective
when another attention-DP rank has Decoder rows. When all ranks have zero
Decoder rows, all skip Decoder computation after GLOBAL publication.
Successful replay exit restores both the output row layout and metadata before
logits selection or drafting. Forward failures propagate without model-side
metadata rollback; a failed engine execution is not retryable.
Outputs use the ordinary scatter and logits selection contract. Embedded DSpark also supports Encoder
recovery and OPTIONAL Encoder pages. Its rolling draft window consumes only the
actual Decoder suffix of each context chunk, including replayed historical rows.
The target publishes per-context valid capture lengths after replay exit.
DSpark skips unwritten captures and advances its rolling window's valid length
only by captured rows; absolute prompt positions still advance normally.
All generation verification rows retain their original batch offsets.
A wider embedded draft window widens the default range for both Decoder replay
and Encoder recovery.
Decoder KV carries progress between chunks through normal cache suspension
and resume; no HC residual/pre-mix tail is retained across forwards. Only the
final context chunk's logits are used for sampling. Requests asking for context logits
retain and return every context row, including intermediate chunks. Decoder rows are a subset of the physical
encoder inputs, so the normal input-token budget covers both phases.
Chunk positions come from packed attention metadata, including with overlap
scheduling. Request phases and sequence slots follow the normal executor
lifecycle; CED keeps no separate progress state or per-chunk CUDA event.
Forwards run in order on the execution stream, and KVCache resume supplies
the page readiness dependencies for transfers.

Attention DP prepares Decoder row selection during input preparation, including
the caller's full-state demand and per-request output requirements. The existing
input count exchange carries both Encoder counts (including recovery rows) and
Decoder counts, alongside speculative counts when enabled. Model forward consumes
that same plan without another collective. Decoder counts retain each eligible
context's intersection with its final window and every generation row, including
zero-row peers. If the subsequent group-wide padding
decision selects a prefill graph, every rank retains the complete padded input
and uses its input counts throughout. Padded direct callers also retain all rows.
At the Decoder layer boundary, all ranks adopt the Decoder counts for MoE,
including ranks with only generation or dummy requests. Local row selection
does not skip EP collectives needed by another rank. After forward, the input
counts and output layout are restored; embedded DSpark remains supported and
its draft counts stay separate. Identity passes avoid activation gather/scatter.
Encoder snapshot selection remains local to each ADP rank.
Direct ADP callers opt into replay by calling `prepare_adp_inputs()` before the
normal input count exchange, with the same full-state requirement as forward.
Without preparation they retain all input rows. Changing a prepared compact
pass into a full-state request after the exchange is an error before layer
execution; it cannot silently switch to incompatible communication counts.
Disaggregated remote-tail execution keeps its input counts across all layers:
the replay rows are selected before model execution. Its existing ADP role
requirements and restriction against speculative decoding still apply.

Encoder bounded replay is enabled by default when Decoder bounded replay is
enabled and the model resolved a replay split with CED precompute. Prefix reuse
uses independently evictable OPTIONAL Encoder groups;
`TRTLLM_V41_ENCODER_REPLAY=0` requires cached Encoder state instead. A topology
without that split keeps required Encoder checkpoints, and an explicit
`TRTLLM_V41_ENCODER_REPLAY=1` for it is an error at model construction.
When `max_num_tokens` is no larger than the recovery window (including an
embedded DSpark capture window), Encoder replay falls back to required Encoder
checkpoints. The model publishes its decision as `csa2_encoder_replay` in
`ModelConfig.extra_attrs`; the cache pools and memory sizing follow it, and a
cache manager without a Decoder replay window never enables Encoder replay.
Decoder bounded replay remains enabled.

Encoder groups keep REQUIRED/OPTIONAL policy even when prefix reuse is disabled;
the global reuse switch disables lookup and publication without changing scratch
grouping. Decoder SWA remains PRIVATE.

Missing Encoder state can be recovered with bounded replay. A complete
Encoder checkpoint can be used even for a one-token chunk if its prefix ends
before or at the final decoder window. A hit inside that window still needs
Encoder replay to produce the missing decoder inputs.

Global partial-prefix matching follows `enable_partial_reuse`, independently
of Decoder replay. Partial Encoder checkpoint reuse remains future work:
OPTIONAL coverage at a non-block-aligned Global endpoint is currently incomplete
and triggers Encoder replay while retaining the Global hit.

Encoder recovery starts at `max(0, P - R)`, using the same configured `R` as
Decoder replay. Physical SWA retention is at least `R + 1` for Decoder layers
and, when Encoder replay is enabled, Encoder layers, before speculative and
reuse-backoff reserves. The extra row protects the first replay row at
partial-block endpoints. Checkpoint coverage and pool sizing use the same
enlarged window; the attention window itself remains unchanged. First resume
uses the ordinary fresh-page allocation for missing OPTIONAL and PRIVATE groups,
which covers historical replay rows without an additional native cache interface.
The executor declares the pending Encoder replay before its first metadata
preparation, including the active Global claim, source interval and SWA read
floor. Historical query rows are excluded from GLOBAL/compressor source rows
from the outset. The prepared floor participates in normal address refresh,
including device-side speculative acceptance corrections under overlap.
Encoder forward consumes these prepared views without another metadata rebuild
or KV-length save/restore; only successful execution consumes the replay plan.
Compressor state retains its ordinary short tail.

Unsupported topology and speculative workers other than embedded DSpark retain full prefill.
An Engram layer after the final KV owner retains the ordinary cache lifecycle rather than allocating private
decoder replay pages. The executor passes ordinary context requests to the model
only when the cache uses that lifecycle. Physical row ranges come from attention
metadata; Encoder recovery comes from the request's resume result. There is no
separate CED request plan, warmup flag, or decode-only flag. Engram history uses
the ordinary input-preparation hook with the Encoder recovery start position.
CUDA graph defaults are inherited from the normal model configuration.
Engines with prefill graphs enabled retain per-chunk replay even on eager
chunks, since a later graph may consume that chunk's Decoder history. Ordinary
generation CUDA graphs do not disable final-window replay.

Disaggregated execution uses the same CED lifecycle. Ordinary context workers
complete the final Decoder window before handoff; generation workers resume available Encoder groups
independently and receive missing state without scheduling local Encoder replay.
The V2 transfer adapter reports each native lifecycle group's admitted reuse
endpoint. It skips only restored pages: a Global hit does not skip missing
OPTIONAL Encoder pages or PRIVATE Decoder pages. Generation admission does not
back off the prefix to leave a local Decoder window, since the context worker
supplies that window. CPU-resident reusable pages follow normal cache resume
before the transceiver obtains GPU page addresses.

Requests for context logits, additional outputs, or multimodal inputs retain
their complete decoder query range. With private decoder pages, these requests
start at zero rather than claiming an Encoder-only cached prefix. Context logits
use the ordinary executor output layout, including mixed batches. Additional output names still follow the model's
existing output contract; this does not add arbitrary hidden-state exports.

## Metadata, graphs and workspace

`CSA2TrtllmMetadata.prepare()` resolves real request IDs, cached lengths and V2
page converters. Model-driven metadata leaves an eager all-context batch's prepare
and block-offset copy to its first CSA2 layer, and an eager context batch prepares
the KV owner tables at its first layer with a KV owner. Layer-specific SWA/visibility, owner page tables/write slots,
compression inputs, candidates and routing results live directly on metadata.
`global_slot_tile()` resolves pages without a persistent token-by-context map.
As in DeepSeek-V4, the manager converts every per-layer page table of a batch
(SWA, GLOBAL and compressor state) in one CuTe DSL launch from a prepare-time
snapshot of the batch's base page rows; the generic block-offset upload copies
the SWA tables from it and prepare reuses the rest, checking allocations against
the base pages. The forward resolves all layers' SWA slots and visibility in one
launch at its first layer. Decoder boundaries raise read-only SWA floors: rows
below them still write their SWA. Live KV-length updates refresh positions and
every KV owner's GLOBAL write slots and compressed groups in one CuTe DSL launch; host-resident metadata uses the equivalent PyTorch
expressions. A steady generation step keeps the previous geometry and re-reads
the page tables only when a request crosses a page.

`set_source_batch()` permits encoder source lengths to differ from decoder
query lengths. Compressor output capacity follows source rows; incomplete
outputs are zero-filled with position zero and write slot -1. Supplied source
hidden states must match that prepared source batch. The model CED helper uses
a cache-only prepass and per-forward metadata markers to preserve full GLOBAL
ownership while query publications and DSpark captures follow their row maps.

Device-side accepted-length updates refresh positions, GLOBAL slots and
ratio-two output groups without replacing captured buffers; SWA slots and
visibility follow the refreshed positions at the next forward. Generation-row
slots beyond the request's allocated capacity (the overlap scheduler's
reservation) stay `-1`. The model resets routing and CED markers at each
complete forward, including repeated graph warmup calls.

Set outer metadata's `is_cuda_graph` before warmup, and warm the standard
forward before capture. Host prepare refreshes persistent device metadata
before replay. Calls and replays sharing metadata must remain serialized.
PrimTS plans live in each query tile's `fmha_plan_caches["csa2_prims_ts"]` and
must be warmed before capture; a cold captured profile is rejected. The row
count selects compile-time split-KV and reduction-topology kernel variants, so
the first plan of each MAIN format (packed records or the context bank) in an
arena compiles every variant up to the arena's query capacity; later row counts
only plan. All profiles
of a selected width share one fixed-address metadata/scratch arena on that
serialized stream. The owning tables exist before graph cloning and remain
shared across graph metadata and target/draft cache-manager swaps. Each forward
refreshes separate compact SWA/MAIN physical-row indices and lengths, device
scales and sinks. Replay reads live page mappings and packed cache payloads;
plans do not retain the raw pools from the layer that first warmed them. Do not run layers,
phases or replays sharing that arena concurrently on separate streams.
Eager buffers replace obsolete geometry; native indexer staging uses one
configured maximum graph arena per device/topk across graph batch sizes and
serialized owners. Replays refresh packed pages, visibility and scheduling.

`workspace_reservation_bytes()` reports one retained native index arena from
explicit serving capacities, including each owner's schedule. The backend
declares eager power-of-two page capacity and the optional shared graph arena
through `runtime_workspace_fixed_bytes()`. Before KV-pool sizing the executor
conservatively reserves the full fixed capacity alongside the profiled peak.
A decode arena may first appear after peak prefill, so its final retained bytes
do not establish how much arena storage coexisted with that earlier peak.
`get_workspace_bytes()` also deduplicates metadata/frame/prior storage separately
from manager pools. The index-prefill gather consumes the full compressed
prefix even with chunked prefill; it declares 69 bytes per gathered key for
each query piece of a request on the native path (512 on the decoded fallback).
The executor's attended-KV admission cap bounds that gather independently of the
fixed graph arena. The fixed reserve also covers one prefill indexer group's FP32
logits and candidate block maxima, bounded by the MQA logits element budget.
For PrimTS, the fixed reserve uses the packed Keep-AB selector's workspace bound
across admitted query counts, including split-KV topology changes. Each selected
width owns one shared arena for native reduction scratch, FP32 LSE, E4M3 Q,
bounded range partials, live Q/KV scale metadata and effective native scales,
and a dummy MAIN record wide enough for either MAIN format, so packed-record and
bank plans share the arena. Separate compact source indices/lengths and two
base FP32 scale words are also included, including SWA-only layouts, as is the
bounded per-device context MAIN bank for widths with MAIN rows. Plans own
only views of that arena, with no per-profile defaults or decoded KV storage. All retained provider tensor
storage is exposed to `get_workspace_bytes()`; manager-owned packed cache pools
are not counted as plan workspace. Projection,
other provider/native scratch and the optional packed kernel's split-KV
workspace still require representative peak-memory profiling at the actual
serving geometry.

## Optional packed attention and remaining scope

`use_packed_sparse_attention` enables an eligible SM100 BF16-Q kernel that
reads packed SWA/main rows directly, bypassing BF16 selected-KV staging. It
uses split-KV partial workspace. `fuse_packed_output_rope` additionally fuses
inverse interleaved RoPE into the output reduction. Both are off by default;
select them with workload-specific numerical validation and profiling.

PP/CP and disabled-layer masks are unsupported. Disaggregation tests cover
owner-role mappings and transferred cache bytes; they do not establish
whole-model disaggregated serving. The model integrates CED and embedded
DSpark, including full-encoder cache production and target-row capture mapping.
Full-checkpoint and GPU validation results are tracked separately from these
implementation capabilities. Whole-model accuracy and performance require
their own measurements.
Component/runtime tests live in `tests/unittest/_torch/attention/sparse/csa2/`.

Numerical definitions follow the official
[reference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py)
and [quantization kernels](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py).
