---
receipts:
  sm_100: {status: passed, tests: 5}
---

# k3_head_gemv

**Wraps** `torch.ops.trtllm.k3_head_gemv` (one call).

## Semantics

The product of a decode step's activation rows with a large weight (an lm_head vocabulary shard), for at most 8 rows,
in bf16:

```
y[m, n] = bf16(acc[m, n])          acc[m, n] = sum over k of x[m, k] * weight[n, k], accumulated in fp32
                                   m < M <= 8, n < N, k < K
```

The weight is cut into 128-row tiles and 128-column k-tiles. The products are accumulated in fp32 by tcgen05 MMAs
(M 128 x N 8 x K 16; the weight is the 128-row operand, the activation the 8 token columns) into TMEM accumulators,
and each sum is rounded once, to nearest, to bf16. Under the stream-K schedule (certified):

- The `T = (N / 128) x (K / 128)` (tile, k-tile) items, tile-major, are cut into `G = min(SMs, T)` equal contiguous
  ranges, one persistent CTA per range: CTA `c` owns items `[c T / G, (c + 1) T / G)`.
- A tile whose items all lie in one range is accumulated by that CTA in k order and stored.
- A tile split over several CTAs has one piece per CTA, in k order. Each piece accumulates its k-tiles in k order in
  fp32. Pieces 1, 2, ... store their fp32 partials in the workspace and raise their flag words; the CTA of piece 0
  (the finalizer) waits for those flags and adds the partials after its own accumulator in k order,
  `((acc_0 + acc_1) + acc_2) + ...`, before the one bf16 rounding.

The partition depends only on `(N, K, G)`, so every sum has a fixed order and the result is bit-identical from run to
run on one GPU (certified). A GPU with another SM count partitions the items differently, so its sums may round
differently. The kernel always computes 8 token columns (rows of `x` past `M` arrive as zeros from the TMA), so the
result of an `M`-row call is bit-identical to the same rows of an 8-row call (certified). It is not bit-identical to
cuBLAS (`F.linear`), whose accumulation order differs.

The `dynamic` schedule (a workspace created with `schedule="dynamic"`; accepted, not certified) cuts each tile's K
into chunks of `chunk_tiles` k-tiles; one persistent CTA per SM takes one (tile, chunk) unit and claims the rest from
a counter, and the last unit of a tile to finish adds the tile's unit partials in chunk order before the one bf16
rounding. Its order is fixed too.

Fusion boundary. Inside: the product and its bf16 rounding. Outside: any bias or logit processing (soft-capping,
temperature, fp32 conversion) and the gather of the vocabulary shards across ranks. `x` and `weight` are read only;
the call writes the result and the workspace's buffers (*State*), nothing else.

## Signature

```python
def k3_head_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    workspace: K3HeadGemvWorkspace,
    keep_tiles: int = 0,
    ring: int = 6,
    prefetch: int = 16,
) -> torch.Tensor
```

The wrapper passes `workspace.partials`, `workspace.flags` and `workspace.claim` as the op's state buffers and
`workspace.chunk_tiles` and `workspace.schedule` as its schedule, so a call always runs the schedule its workspace was
created for. `K3HeadGemvWorkspace` is importable from this entry's wrapper module.

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `x` | `[M, K]`, `1 <= M <= 8` | bf16 | contiguous, 16-byte aligned | CUDA |
| `weight` | `[N, K]`, `N % 128 == 0`, `K % 128 == 0` | bf16 | contiguous, 16-byte aligned | CUDA, `x`'s device |
| `workspace` | a `K3HeadGemvWorkspace` for `(N, K)` on `x`'s device (*State*) | — | — | — |
| `keep_tiles` | scalar | Python int | — | — |
| `ring` | scalar | Python int, 1-6 | — | — |
| `prefetch` | scalar | Python int, `>= 0` | — | — |
| returns | `[M, N]` | bf16 | contiguous, the first `M` rows of a new `[8, N]` buffer | `x`'s device |

`keep_tiles`: weight tiles below it load at normal L2 priority, the rest with an EVICT_FIRST hint (*Notes*). `ring`:
the shared-memory pipeline stages. `prefetch` (stream-K): the k-tiles past the ring that each CTA prefetches into L2
before its grid-dependency wait.

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `x` | `[M, 7168]`, `M` = 1, 2, ..., 8 | bf16 | contiguous | CUDA |
| `weight` | `[10240, 7168]`, two different weights | bf16 | contiguous | CUDA |
| `workspace` | `K3HeadGemvWorkspace.create(10240, 7168, device)` (stream-K) | — | — | — |
| `keep_tiles` | 0 (every cell); 40 and 80 at `M` 1 and 8 | int | — | — |
| `ring`, `prefetch` | 6 and 16 (the defaults) | int | — | — |
| returns | `[M, 10240]` | bf16 | contiguous | CUDA |

`[10240, 7168]` is Kimi K3's TP16 per-rank LM-head vocabulary shard (163840 / 16 rows). The `dynamic` schedule is
accepted but not certified. Every cell: the error is at most 8e-3 of `max |ref|` against an fp64 torch product; a
rerun is bit-identical; the rows are bit-identical to the same rows of the 8-row call; the workspace's words are all
zero after the call; `keep_tiles` 40 and 80 are bit-identical to 0. The call sequences and refusals certified on top
of the cells are listed under *State* and *Preconditions*.

## State

**Object.** `K3HeadGemvWorkspace` (`tensorrt_llm/_torch/cute_dsl_kernels/k3_head_gemv/op.py`, re-exported by this
entry's wrapper module), owned by the caller: a frozen dataclass of the weight shape (`n_out`, `k_in`), `schedule`,
`chunk_tiles` (0 for stream-K) and three tensors.

**Contents and size.** With `tiles = N / 128`, `G = min(SMs, tiles x K / 128)` and `P` the most CTAs any tile is
split over (`k3_head_gemv_kernel.streamk_max_pieces(N, K, G)`), a stream-K workspace holds:

- `partials`, fp32 `[tiles x P x 128 x 8]`: slot `(t, p)` holds the fp32 partial `[128 rows][8 tokens]` of piece
  `p > 0` of tile `t` (slot `p = 0` is not used);
- `flags`, int32 `[tiles x P]`: word `(t, p)` is raised to 1 by piece `p > 0` of tile `t` once its partial is stored,
  and lowered to 0 by the tile's finalizer once it has read the partial;
- `claim`, int32 `[1]`: not used by stream-K (the dynamic schedule's unit counter).

At the certified shape on a GPU with 148 or 152 SMs: 80 tiles, 56 k-tiles, `P` = 3, so 240 flag words (960 B) and
245,760 partial floats (960 KiB). The sizes do not depend on `M`: one workspace serves every `M`. A dynamic workspace
holds the fp32 partial of every (tile, chunk) unit, one unit count per tile in `flags` and the unit counter in `claim`.

**Who creates it, and when.** The target, once per weight shape and schedule, in `post_load_weights`, with
`K3HeadGemvWorkspace.create(n_out, k_in, device, schedule="streamk", chunk_tiles=0)`. It is single-GPU, not
collective, and eager: it allocates `partials` uninitialized and zeroes `flags` and `claim` on the current stream (the
first call must be ordered after that), and it raises `RuntimeError` under CUDA-graph capture (certified). An unknown
schedule raises `ValueError` (certified). The sizes follow the SM count of `device`, so the workspace is created for
the device its calls run on. No environment variable is read.

**Which ops may share one object.** Only `k3_head_gemv` calls of the workspace's weight shape and schedule on its
device: any `M`, any weight of that shape, any `keep_tiles` (and any `ring` or `prefetch`, which do not change the
layout). At TP16 the target's LM head and DSpark's draft-logits head both have the `[10240, 7168]` shape. Certified:
`M` = 8, 8, 2, 7, 8, 1, 1, 8, 3, 8 back to back on one workspace, and two weights alternating on one workspace, every
call bit-identical to the same call on a fresh workspace. Two workspaces are independent: calls alternating between
two, each with its own weight, are all correct (certified). The op refuses, with `ValueError` before launching, a
workspace whose buffers do not fit the call: another dtype or device, another number of `flags` words, too few
`partials` (certified with a workspace created for `[5120, 7168]`). It checks nothing else, so the sharing rule is the
caller's.

**Call-order invariant.** A launch on a workspace must start after the previous launch on it has completed: they use
the same flag words and partial slots. The calls that share a workspace are therefore ordered on one stream, which
holds under PDL too: a launch reads and writes the workspace only after its grid-dependency wait
(`griddepcontrol.wait`), that is, after the work before it on the stream has completed. Eager calls from a second
stream must not share the workspace unless the caller orders them (a synchronization or an event between, e.g., a
load-time call on one stream and the decode steps on another). Captured calls replay in capture order on the
replaying stream, so a graph holding calls on a workspace may be replayed on the stream that issues the eager calls
on it, between those calls. Certified: a graph of `M` 1, 8 and 3 calls on one workspace, replayed three times with `x`
rewritten in place and an eager `M` 8 call on the same workspace between replays, every result bit-identical to the
same call on a fresh workspace. Not exercised: two launches running at once on one workspace raise and lower each
other's words, so a finalizer can read the other launch's partial (a silently wrong result) or wait for a word the
other launch has already lowered (a hang).

**What a later launch reads.** Every word of `flags`, which must be 0 when the launch starts: a finalizer waits until
the word of each later piece of its tile is non-zero and then reads that piece's slot of `partials`. `partials` are
written before they are read within one launch, so their contents between launches do not matter. A dynamic launch
also reads the per-tile unit counts and the unit counter, which must be 0 as well.

**How it is re-armed.** By the launch itself. Each finalizer lowers its tile's words after reading the partials (in the
dynamic schedule the finalizing unit resets its tile's count, and the last claim rolls the counter back to 0), so a
launch that completes leaves every word at 0 for the next one, whatever its `M` (certified: all words zero after every
sequence above). The layout depends only on `(N, K, schedule, SMs)`; no call depends on an earlier call's `M`. A
launch that never completes can leave words raised; the workspace must then be recreated (`create()` zeroes them).

## Metadata consumed

Besides `workspace`, which is an explicit argument:

- The current CUDA stream: the kernel is launched there.
- A per-process compile cache: stream-K kernels keyed by `(N, K, ring, G, P, prefetch, PDL)`, dynamic ones by
  `(N, K, chunk_tiles, ring, G, PDL)`. `M` and `keep_tiles` are runtime arguments. The first call for a key compiles
  the kernel (seconds) and must be made eagerly: under CUDA-graph capture it raises `RuntimeError` ("run once per
  shape outside CUDA-graph capture first"). The cache is result-neutral.
- The device's SM count, read on every call: it sets `G`, hence the partition and the workspace sizes.
- `TRTLLM_ENABLE_PDL` (default `"1"`), read on every call: whether the kernel is launched with PDL. It is part of the
  cache key and changes scheduling, not results. The test runs with the default.

## Preconditions

- `x` and `weight` bf16, 2-D and contiguous; `x.shape[1] == weight.shape[1]`; `1 <= M <= 8`; `x` on a CUDA device;
  `N % 128 == 0` and `K % 128 == 0`.
- Stream-K: `1 <= ring <= 6`. Dynamic: the workspace's `chunk_tiles` divides `K / 128`.
- A call outside these raises `ValueError` ("k3_head_gemv: unsupported call ...") before launching anything
  (certified: `M` = 0 and 9, `N` = 200).
- `workspace` created for this weight shape and schedule on `x`'s device: one whose buffers do not fit the call
  raises `ValueError` ("... is not one for weight ...") before launching anything (certified, see *State*).
- `create()` runs outside CUDA-graph capture (else `RuntimeError`), and every key is compiled by an eager call before
  a capture uses it (*Metadata consumed*).
- Not checked by the op: `weight` on `x`'s device; 16-byte-aligned data pointers (the op declares that alignment to
  the kernel; row slices `x[a:b]` of these shapes are aligned); `prefetch >= 0`; an SM 10.x GPU (the kernel uses
  tcgen05); the CuTe DSL (`cutlass`) and `cuda-python` (`cuda.bindings`), which the call and `create()` import, so
  without them they raise `ImportError`.
- Under PDL each CTA loads weight tiles (its first `ring` k-tiles, and `prefetch` more into L2) before it waits for
  the kernel it follows on the stream, so `weight` must not be written by that kernel; a head weight is constant
  during inference. `x` is read, and the workspace and `y` written, after the wait.

## Notes

- PDL: with `TRTLLM_ENABLE_PDL` = `"1"` the kernel is launched with programmatic dependent launch. Before
  `griddepcontrol.wait` a stream-K CTA only loads weight tiles; it releases its dependents once it has issued all of
  its loads (a dynamic CTA, after its last claim). A dependent must itself wait for this grid before reading `y`, as
  every PDL kernel waits for its predecessor before reading its output.
- L2: weight tiles `>= keep_tiles` are loaded with an EVICT_FIRST hint, so the streamed shard does not evict the rest
  of L2; tiles `< keep_tiles` load at normal priority, so a later reader of the same weight can find them in L2.
  Cache policy only: `keep_tiles` 0, 40 and 80 are bit-identical (certified).
- Grid: `G` persistent CTAs of 256 threads (dynamic: one per SM). Stream-K launches at most one CTA per (tile,
  k-tile) item, so every CTA's range is non-empty and raises the flag its finalizer waits for, also for a weight with
  fewer items than the GPU has SMs. Such weights are not certified here.
- The result is a view of the first `M` rows of a new `[8, N]` buffer: the kernel writes all 8 token rows, the rows
  past `M` from the zero-filled activation rows.
