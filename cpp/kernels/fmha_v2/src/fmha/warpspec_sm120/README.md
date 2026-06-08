# halfspec — TMA-load + sync-MMA warp-specialized FMHA for sm_120 / sm_121

> Codename **halfspec** = half of the Hopper warp-specialization recipe.
> TMA-driven async loads survive the port to consumer Blackwell; async MMA does
> not (sm_120 / sm_121 have no `wgmma.async` equivalent). The compute warps
> therefore stay on `mma.sync`, while a dedicated producer warp drives the
> loads with TMA.

This directory implements a warp-specialized context FMHA for the sm_120
family (sm_120 / sm_121). It targets BF16, causal mask, `head_dim ==
head_dim_v` in `{128, 256}`, and the PACKED_QKV layout. The kernel carries the
per-warp skip-softmax optimization into the warp-specialized design.

## Files

| File | Role |
|------|------|
| `kernel_traits.h` | `Kernel_traits_halfspec_sm120`: wraps `fmha::Kernel_traits_v2` for the LDGSTS-friendly `Smem_tile_*` types, then layers on the producer/consumer warp roles, the granular smem buffers, the circular-buffer barriers, and the V re-tile (see below). |
| `dma_sync_mma.h` | Producer (`DMA::run`). Issues `cp.async.bulk.tensor.3d.shared::cta.global.tile` for Q / K / V into the granular buffers. `DMA::Host::init_params` builds the three `CUtensorMap` descriptors with the driver-API `cuTensorMapEncodeTiled`. |
| `compute_sync_mma.h` | Consumer (`Compute::run`). The kv-loop body — BMM1 (`fmha::gemm`) + softmax + causal mask + per-warp skip-softmax vote + BMM2 + epilogue — reading the granular `Smem_tile_q/k/v` per ring slot. |

The translation unit and the in-engine dispatch bridges live in
`cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/halfspec_sm120/fused_multihead_flash_attention_ws_sm120.cu`,
and the entry kernel in
`cpp/kernels/fmha_v2/src/fused_multihead_flash_attention_kernel_ws_sm120.h`.

## How the runner reaches this kernel

The kernel is opt-in. The PyTorch attention op exposes a `use_halfspec_fmha`
flag (plumbed `attentionOp` → `MHARunnerParams` → `Launch_params`); when it is
set and the config matches (sm_120 / sm_121, BF16 in/out, causal, `head_dim ==
head_dim_v` in `{128, 256}`, PACKED_QKV), `FusedMultiHeadAttentionXMMAKernelV2::run`
dispatches to the `run_halfspec_*` bridges instead of the cubin path. The flag
is a no-op on every other architecture and shape.

The translation unit is compiled only into the `_context_attention_kernels_120`
CMake target (sm_120 family). The all-architecture dispatch TU references the
bridge symbols under `TLLM_ENABLE_HALFSPEC_SM120`, which CMake defines only when
sm_120 is built, so builds that exclude sm_120 neither reference nor link the
(then-absent) symbols.

## Design rationale

### Why TMA loads, not "just split the warps"

In the non-warp-specialized tiled kernel, the Q / K / V loads are *multi-thread*
LDGSTS operations: each of the 128 threads issues several `LDGSTS` instructions
to cover `(tile rows × D bytes)`. There is no way to "have warp 0 do the load"
without rewriting the gmem/smem tile load helpers — the partition is baked into
them. TMA fixes exactly this: a single descriptor + a single
`cp.async.bulk.tensor` from one thread issues an entire tile load, and the
consumers wait on an `mbarrier`. So the producer warp uses TMA, not LDGSTS.

### TMA descriptor format

Blackwell's TMA engine requires the driver-API `cuTensorMapEncodeTiled`
(128-byte `CUtensorMap`) descriptor — the same form the shipping
trtllmGenKernels FMHA uses. The fmha_v2 hand-rolled 64-byte `fmha::cudaTmaDesc`
(Hopper-era bit layout) is rejected and faults at `UTMALDG`. The descriptors
are built host-side in `DMA::Host::init_params` and passed to the kernel as
`__grid_constant__` params.

### Why the LDGSTS smem tiles can be filled by TMA

The make-or-break question for reusing the existing consumer `Smem_tile_*` is
whether their LDGSTS XOR swizzle equals a TMA hardware swizzle mode. It does:
the Q and K granular tiles use `BYTES_PER_ROW = 128`, `BYTES_PER_STS = 16`,
`ROWS_PER_XOR_PATTERN = 8`, i.e. a physical 16-byte chunk index of
`(col / 8) ^ (row % 8)` — byte-identical to the TMA 128B hardware swizzle. So a
chunked 128B-swizzle TMA load fills `Smem_tile_q/k` directly and the consumer's
`ldmatrix` reads correct data.

### V is re-tiled to 64-wide DV chunks

The natural `Smem_tile_v` packs the full `DV` (256) into the lead dim, giving
512-byte smem rows that no TMA swizzle mode can reproduce (`cuTensorMapEncodeTiled`
caps the leading box dim at the 128-byte swizzle width; a 512-byte leading dim
only encodes with `SWIZZLE_NONE`, which is plain row-major and does not match
the consumer's XOR-swizzled read). Instead, V is tiled into `BMM2_DV_CHUNK = 64`
wide groups so the V smem tile has `LEAD_DIM = 64` → 128-byte rows — the same
proven layout as K — and the existing `N == 64` `ldsmt` read path applies
unchanged. The producer streams `DV / 64` dv-chunks per kv-tile; the consumer
BMM2 contracts per dv-chunk into the corresponding `acc_o` sub-range.

### `setmaxnreg` is unavailable here

`setmaxnreg.{dec,inc}` is a Hopper / datacenter-Blackwell instruction
(sm_90 / 100 / 103); ptxas hard-errors on sm_120 / sm_121. The producer/consumer
register-budget split therefore does not exist on this hardware and is guarded
off (no-op on sm_120 / sm_121).

## What the port wins, and what it does not

Wins on sm_120 / sm_121:

- **Fewer load instructions** — one `cp.async.bulk.tensor` per tile replaces
  the many per-thread `LDGSTS` of the tiled kernel.
- **Per-buffer-slot waits** (`mbarrier`) instead of CTA-wide `__syncthreads()`
  between load and compute: a consumer warp unblocks as soon as its tile lands.

Does not win:

- **MMA / softmax overlap** — there is no `wgmma.async` on sm_120, so a consumer
  warp's `mma.sync` blocks its issuing thread until result registers commit. The
  Hopper warpspec hides BMM1/BMM2 MMA latency behind softmax/`frag_p` work; that
  is not achievable with sync MMA only.
- **Register-budget split** — `setmaxnreg` is unavailable (see above).

## Relationship to a CuTe-DSL kernel

CUTLASS 4.x has Blackwell sm_120 FMHA examples implementing the TMA-load +
sync-MMA pattern in CuTe DSL. A longer-term direction is to route the sm_120 /
sm_121 dispatch into a CuTe-DSL kernel. This fmha_v2 implementation maps the
relationship between the existing fmha_v2 infrastructure and that design and is
self-contained: the dispatch is gated, and the directory plus the entry-kernel
header are isolated (no other code includes them).
