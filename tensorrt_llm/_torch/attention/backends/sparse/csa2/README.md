<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Compressed Sparse Attention 2 (CSA2)

This package implements DeepSeek-V4.1 CSA2 attention, indexing, cache ownership
and runtime metadata. The V4.1 model adapter integrates these components with
causal encoder/decoder (CED) replay and embedded DSpark. Model registration and
orchestration live in the model and executor modules.

## Backend and hardware routing

`CSA2TrtllmAttention` owns the shared cache/indexer preparation. On the default
SM100 path it inherits `TrtllmAttention.forward`, including output allocation
and native FMHA selection. The module directly invokes its composed Flash
helper on other hardware; there is no CSA2 FMHA registration or controller.

| Architecture | Module compute path | Computation |
| --- | --- | --- |
| SM100 family | `CSA2TrtllmAttention.forward` | Native trtllm-gen dynamic sparse MLA |
| SM120/121 | `CSA2FlashInfer` | FlashInfer BF16 FA2 |
| SM90 | `CSA2FlashMLA` | FlashMLA sparse BF16 |

FP8 context tiles that stage the shared E4M3 bank with 64 query heads per rank
run `torch.ops.trtllm.csa2_sparse_context` (`cpp/tensorrt_llm/kernels/csa2SparseContext`)
instead of the trtllm-gen context FMHA, which serves every other tile. Staging
is unchanged; the kernel reads FP8 Q, or quantizes BF16 Q with the staging scale
on load to the bytes the context FP8 quantizer writes, and follows the trtllm-gen
FP8 softmax (P rounded to E4M3 after scaling by 448, FP32 row sums, sink in the
denominator).

Generation steps of at most 16 query rows per rank instead attend the
selected rows straight from the persistent caches (`direct_decode.py`): one
CuTe DSL launch decodes the SWA and main rows exactly into BF16 shared memory
and attends them with the BF16 query and the sink, so neither the FP8 staging
nor the native FMHA runs. The backend decides per tile in
`direct_decode_applies` (generation-only forwards without context tokens,
SM100/SM103, a geometry that fits one CTA's shared memory); every layer of such
a forward takes it, and larger batches keep the staging path.

Decode work of the indexer that does not depend on the layer's own projections runs on
CSA2 side streams: a Full layer of a forward without context tokens projects its index
weights on one and refreshes the paged indexer descriptors on another while the auxiliary
stream runs the compressor (the indexer joins them through `CSA2ForwardState.index_ready`),
and the candidate source publishes its candidates and builds its consumers' sparse logits
schedule on the first, next to its own TopK (consumers wait for both). Every kernel and
input stays the same, so results are bit-identical.

`backend.py` contains the single sparse backend and both plain Flash helper
classes. The opt-in packed path uses the backend's explicit `forward_packed`
method. The module issues one attention call per real phase (all context
queries, then all generation queries) and every path invokes the same sparse
preparation once per phase; CSA2-specific GPU kernels are collected in
`kernel.py`.

The default compute path decodes selected SWA/main rows into bounded staging
pools. `kv_cache_config.dtype` selects only this staging dtype; the persistent
cache formats never change with it. Staging is BF16 unless FP8 is asked for:
with `fp8` or `fp8_ds_mla` (or an FP8 KV-cache algorithm in a directly
constructed backend's quantization config under `auto`), native trtllm-gen
consumes E4M3 staging. `CSA2Params.use_fp8_staging` overrides that choice --
left unset it follows the native path's own support, `False` pins BF16, and
`True` demands E4M3 and rejects a configuration that cannot serve it rather
than falling back. The E4M3 per-tensor KV scale is range-aware: every forward
reduces the selected rows' persistent group-scale bytes to one exact ceiling on
what they can decode to, then picks the smallest power of two mapping that
ceiling onto 448. A power of two costs no relative precision because E4M3's
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
trtllm-gen without packed attention; the FlashMLA and FlashInfer adapters never
forward an FP8 KV algorithm to the inherited quantization state. Any other
KV-cache quantization algorithm (for example NVFP4) is rejected because it has
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
expanding rows per query. Generation and fixed-shape context graphs use
independent-query staging with fused compaction/dequantization kernels.

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
SM100 and 32 index heads of 128D rows. Reuse layers
consume logical selections from their index source and resolve current physical
pages without recomputing indexer logits.

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

TopK is exact by default: eligible prefill of at least 8192 score columns uses
the single-pass self-sampling engine, narrower prefill and decode the radix
TopK. Short sequences whose keys all fit can
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
token, excluding SWA, compressor state and compute workspace. SWA uses 528-byte
E4M3/power-of-two rows. All formats include their RoPE channels.
These persistent formats are part of CSA2's cache contract and do not require
a generic NVFP4 KV-cache flag. Projection precision and the selected attention
compute path do not change the persistent main-cache encoding.

Long prefill uses scratch-aware page mappings. Partial groups retain raw FP32
values and scores. CUDA BF16 cache publication on SM100 uses fused exact-byte
CuTe DSL quantize/scatter kernels (`kernel.py`), including the split
data/scale packing of index queries; other paths use the encoder and masked
scatter. Invalid slots, including padding slot -1, never write a cache row.
The default staged attention path uses CuTe DSL gather/dequantization on SM100
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
Encoder/Decoder range of 140 tokens, controlled by
`TRTLLM_V41_BOUNDED_REPLAY_TOKENS`. This must exceed the required SWA window
(including the embedded DSpark capture window), without a physical-ring or
page-alignment constraint. The released checkpoint's SWA window remains 128.
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
A wider embedded draft window requires setting `TRTLLM_V41_BOUNDED_REPLAY_TOKENS`
above that window for both Decoder replay and Encoder recovery.
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
enabled. Prefix reuse uses independently evictable OPTIONAL Encoder groups;
`TRTLLM_V41_ENCODER_REPLAY=0` requires cached Encoder state instead.
When `max_num_tokens` is no larger than the recovery window (including an
embedded DSpark capture window), Encoder replay falls back to required Encoder
checkpoints. The model, cache pools and memory sizing use this same decision;
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
Projection/provider/native scratch and the optional packed kernel's split-KV
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
