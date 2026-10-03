---
receipts: {}
---

# k3_mla_attn_vb_out

**Wraps** `torch.ops.trtllm.k3_mla_attn_vb_out` (one call). The same contract covers two sibling wrappers, one call
each: `k3_mla_attn_out` (`torch.ops.trtllm.k3_mla_attn_out`, the attention output itself) and `k3_mla_attn`
(`torch.ops.trtllm.k3_mla_attn`, the same into a new tensor, page offset 0). All three take a caller-owned
`K3MlaAttnWorkspace` (*State*).

## Semantics

Kimi K3's MLA decode attention for `R <= 8` requests of `T <= 8` tokens over the paged latent cache, with v_b and
the output gate in the same launch. `q` is `fused_q` (from `attention/k3_mla_qkv`), request-major: rows
`i T .. i T + T - 1` are request `i`'s tokens. Request `i`'s cache rows `kv_i[k]`, `k < L_i = seq_len[i]`, are row
`(page_table[i][k // 64] + page_offset) * 64 + k % 64` of `pool` (512 latent columns, then 64 rope columns). Per
request `i`, token `t` and head `h`:

```
o[t, h]   = softmax_k( softmax_scale * q[t, h] . kv_i[k] ) @ kv_i[k, :512]     k <= L_i - T + t (bottom-right causal)
y[t, h]   = bf16( bf16(o[t, h]) @ w_vb[h]^T )                                    # v_b, 128 per head
out[t, 128 h : 128 h + 128] = y[t, h]                     # without gate
                            = bf16(y[t, h] * s[t, h])     # with gate: s = gate[t, gate_col0 + 128 h .. + 128]
```

`gate` holds the output gate's sigmoid values (the op multiplies, it does not apply the sigmoid); the gated output is
bit-identical to `bf16(plain * s)` (certified). `k3_mla_attn_out` writes `o` itself (`[M, heads * 512]`, bf16) into
`out`; `k3_mla_attn` returns it in a new tensor and is bit-identical to `k3_mla_attn_out` at page offset 0
(certified).

How it computes: one cluster of 16 CTAs per (request, group of 6 heads), split-KV over 128-row tiles; fp32 softmax;
each CTA's partial `O / l` (a convex combination of V rows) is kept in fp16, and the 16 partials are merged in fp32
in a fixed order. More than `CLUSTER_WAVE` = 7 clusters (`R x heads / 6`) would not co-reside on GB200; when all
their CTAs fit on the SMs at once (one per SM: at most 9 clusters on 148 SMs) the launch takes the **no-cluster
mode**, which exchanges the merge statistics through the workspace instead of cluster shared memory. The merge sums
the same values in the same order in both modes, so the outputs do not depend on the mode. Request `i`'s rows are
computed as the one-request call on its own rows, pages and length would compute them (the op test,
`test_k3_mla_attn.py`).

Accuracy (certified bound): every request's output within `1e-2` (max relative, per request) of a float64
reference: the attention in float64 with the causal mask above, rounded to bf16; v_b in float64; the gate applied
to the bf16-rounded product.

Fusion boundary. Inside: the attention over the cache, v_b, the output gate. Outside: `fused_q` and the cache append
(`attention/k3_mla_qkv`, which must store the step's rows before this call reads them), and the output projection
that consumes `out`.

## Signature

```python
def k3_mla_attn_vb_out(
    q: torch.Tensor,
    pool: torch.Tensor,
    row_stride: int,
    page_table: torch.Tensor,
    page_offset: int,
    seq_len: torch.Tensor,
    softmax_scale: float,
    w_vb: torch.Tensor,
    out: torch.Tensor,
    workspace: K3MlaAttnWorkspace,
    gate: Optional[torch.Tensor] = None,
    gate_col0: int = 0,
) -> None

def k3_mla_attn_out(q, pool, row_stride, page_table, page_offset, seq_len, softmax_scale, out,
                    workspace: K3MlaAttnWorkspace) -> None

def k3_mla_attn(q, pool, row_stride, page_table, seq_len, softmax_scale,
                workspace: K3MlaAttnWorkspace) -> torch.Tensor
```

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `q` | `[M, heads * 576]`, `M = R T`, `R <= 8`, `T <= 8`, `heads` a multiple of 6 (6 at TP16, 24 at TP4) | bf16 | contiguous | CUDA |
| `pool` | the layer's paged latent pool, flat | bf16 | contiguous | CUDA |
| `row_stride` | elements per cache row, `>= 576`, a multiple of 8 | Python int | — | — |
| `page_table` | `[R, W]` with unit column stride and any row stride `>= W` (rows of `kv_cache_block_offsets`), or `[W]` when `R = 1` | int32 | see shape | CUDA |
| `page_offset` | the layer's slot in a layer-interleaved pool | Python int | — | — |
| `seq_len` | `[R]`, `L_i >= T`: each request's length including its `T` new rows | int32 | contiguous | CUDA |
| `softmax_scale` | `1 / (sqrt(qk_nope + qk_rope) * q_scaling)` (the view's) | Python float | — | — |
| `w_vb` | `[heads, 128, 512]` (`v_b_proj`) | bf16 | contiguous | CUDA |
| `out` | `[M, heads * 128]` (`k3_mla_attn_out`: `[M, heads * 512]`) | bf16 | contiguous | CUDA |
| `workspace` | a `K3MlaAttnWorkspace` of `q`'s device for `heads / 6` head groups | — | — | — |
| `gate` | `None`, or `[M, C]` with unit column stride and `gate_col0 + heads * 128 <= C` | bf16 | rows of any stride | CUDA |
| `gate_col0` | `>= 0` | Python int | — | — |
| returns | `None` (`k3_mla_attn`: `[M, heads * 512]`, newly allocated) | bf16 | contiguous | = `q.device` |

Certified at `(R, T, heads)` = `(3, 8, 6)` (3 clusters), `(8, 1, 6)` and `(2, 8, 24)` (8 clusters: the no-cluster
mode), with 64 to 4100 cached rows per request before a step (pages crossed, several tiles per CTA), over a real
`KVCacheManager` (below).

## State

**Object.** `K3MlaAttnWorkspace` (`catalog/attention/k3_mla_attn_workspace.py`), one per device and head-group
count (`heads / 6`: 1 at TP16, 4 at TP4), owned by the caller. The pool is the KV cache manager's (as
`attention/k3_mla_qkv`); this call only reads it.

**Contents and size.** One fp16 buffer, `attn_workspace_elems(groups)` elements: the per-CTA partials of 8 requests
x `groups` x 16 CTAs (64 KiB each: 8 MiB per head group), then the no-cluster mode's fp32 `(m, l)` exchange (48 KiB
per group) and two int32 arrival counters per (request, head group), each on its own 128-byte line (2 KiB per
group): 8,439,808 bytes per head group, 33,759,232 at TP4. The size does not depend on `R` or `T`: every call fits.

**Who creates it, and when.** The target, in `post_load_weights`, with `K3MlaAttnWorkspace.create(device, groups)`:

- eager: it allocates, so it refuses to run under CUDA-graph capture (`RuntimeError`, certified);
- it zeroes the counters and leaves the partials as allocated (a call reads only words it wrote, below), then
  synchronizes the device, so the workspace is ready on any stream.

**Which ops may share one object.** All three forms, for every MLA layer of the device with the same head-group
count: one workspace serves a model's layers. Two workspaces are independent (certified: the layers' calls
alternating between two workspaces over three no-cluster steps; every output bit-identical to the same call on a new
workspace, each workspace's counters counting only its own launches).

**Call-order invariant.** Calls on one workspace run one at a time. Every access a call makes to the workspace
(partials, exchange, counters) follows its grid-dependency wait, so a call may directly follow another on the same
stream; the earlier call must have completed by the time the later one passes its wait. That holds for calls on one
stream when every kernel between them waits on its predecessor or is launched without PDL, as the kernels of a decode
step do. Calls that overlap on one workspace (two streams, or a graph replay beside an eager call) overwrite each
other's partials and counters; they are outside the contract (not measured).

**What a later launch reads.** Only the counters, and only in the no-cluster mode: a launch reads each of its
(request, head group) counters after its grid-dependency wait and before it arrives, then waits for 16 arrivals past
`count & ~15`. The partials and the exchange are written and read within one call: the op test refills them with NaN
before every call and the outputs keep their bits (`test_k3_mla_attn.py::test_attn_workspace_poison`). Before its
grid-dependency wait a launch reads only the page table, the lengths and the cache pages before the one holding row
`L_i - T` (earlier steps wrote them); `q` and the pages from that one on after it.

**How it is re-armed.** Never. The counters only grow: each no-cluster launch adds 16 to both counters of each of its
requests' head groups and leaves the others; cluster-mode launches leave them all (certified: counters equal to 16 x
the launches after a sequence of eager calls and graph replays). They are compared by signed difference, so they run
through the int32 wrap (op test: counters started at `2^31 - 16` and at `-16` give the bits of zeroed ones). A
launch must start with its counters at a multiple of 16, which every complete launch leaves; `create()` starts them
at 0.

**Why the test drives call sequences.** One workspace serves every layer and step, eagerly and inside captured
graphs, and its counters carry over from launch to launch. The test runs 3 layers x 2 decode steps on one shared
workspace eagerly, then one step's 3 calls captured as a CUDA graph on the same workspace and replayed for 2 more
steps with rewritten `q` and the metadata prepared for each step. Every output is
compared bit for bit with the same call on a new workspace and within `1e-2` with the reference; in the no-cluster
mode the shared workspace's counters are checked at the end.

**What a wrong workspace does.** Measured (the test's negative control): a one-head-group call given a workspace
laid out for four head groups, an fp32 tensor, or a buffer 8 elements short raises `ValueError` naming the expected
workspace, before any launch; neither `out` nor the workspace changes.

## Metadata consumed

Through `k3_mla_decode_view` (see `attention/k3_mla_qkv`): the page table and lengths are views of the prepared
metadata's `kv_cache_block_offsets` and `kv_lens_cuda_runtime`, `page_offset` the layer's slot, `row_stride` the
pool's token stride, `softmax_scale` from the attention module's `qk_nope_head_dim`, `qk_rope_head_dim` and
`q_scaling`.

## Preconditions

- `q`, the page table and the lengths describe `R <= 8` requests of the same `T <= 8` tokens, `heads` a multiple of 6,
  `row_stride >= 576` and a multiple of 8; `w_vb`, `out` and `gate` as above. Anything else raises `ValueError`
  before any launch (the op test, `test_attn_rejects`), as does a workspace of another layout, dtype or device
  (certified, above).
- `seq_len[i] >= T`: the step's `T` new rows are in the cache (stored by `k3_mla_qkv` first).
- The first call of each configuration compiles the kernel and must run outside CUDA-graph capture (it raises
  `RuntimeError` inside one).
- The no-cluster mode needs its 16 x `R x groups` CTAs co-resident (one per SM); the op takes it only when they fit,
  and otherwise launches clusters in more than one wave.
- SM 100 / SM 103 only (CuTe DSL `tcgen05` kernel).

## Notes

- The op-level test is `tests/unittest/_torch/cute_dsl_kernels/kimi_k3/test_k3_mla_attn.py`: every split of up to 8
  tokens and the R x 8 verify steps against the float64 reference and the stock CuTe DSL MLA decode
  (`trtllm::cute_dsl_mla_decode_fp16_blackwell`), the TP4 head count, interleaved pools of rows of 640, the
  workspace poison and counter-wrap checks above, refusals, and (with `K3_BASE_TRTLLM`) batch-1 identity against
  the unmodified single-request kernel.
- `k3_mla_attn` takes no page offset: it fits a pool with one layer per block, or layer 0 of an interleaved one.
