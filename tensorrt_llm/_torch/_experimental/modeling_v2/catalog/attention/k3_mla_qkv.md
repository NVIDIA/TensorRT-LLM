---
receipts:
  sm_100: {status: passed, tests: 8}
---

# k3_mla_qkv

**Wraps** `torch.ops.trtllm.k3_mla_qkv` (one call). The same contract covers two sibling wrappers, one call each:
`k3_mla_q` (`torch.ops.trtllm.k3_mla_q`, the query path alone) and `k3_mla_qkv_out`
(`torch.ops.trtllm.k3_mla_qkv_out`, the cache rows stored densely instead of into the pool).

## Semantics

Kimi K3's MLA decode query path for `M <= 64` tokens in one launch, plus the step's latent KV rows stored into the
paged latent cache. Per token `t` and head `h`, from the fused projection's rows `ag` (fp32 statistics, one bf16
rounding wherever the unfused model rounds):

```
q_n[t]        = bf16(rmsnorm(ag[t, :1536]) * w_qa)                      # q_a_layernorm
q[t, h]       = bf16(q_n[t] @ w_qb[192 h : 192 h + 192]^T)                # q_b_proj: 128 nope | 64 pe
fused_q[t, h] = [ bf16(q[t, h, :128] @ w_kb[h]^T) | q[t, h, 128:] ]       # k_b absorption (512) | q_pe (64)
row[t]        = [ bf16(rmsnorm(ag[t, 1536:2048]) * w_kv) | ag[t, 2048:2112] ]   # kv_a_layernorm (512) | rope (64)
```

Kimi K3 is NoPE: `q_pe` and the rope columns are copied, not rotated. `fused_q` is returned in a new tensor
`[M, heads * 576]`; `row[t]` is stored into `pool`. The `M` tokens are `R` requests of `T = M / R` tokens each,
request-major: token `t = i T + u` is token `u` of request `i`, at position `p = seq_len[i] - T + u`, stored as row
`(page_table[i][p // 64] + page_offset) * 64 + p % 64` of `row_stride` elements (64-bit element index). A token
with `p < 0` is not stored. Nothing else in `pool` is written (certified: every layer's whole pool compared after
every step).

- `k3_mla_q` returns the same `fused_q` bits and stores nothing (certified).
- `k3_mla_qkv_out` returns the same `fused_q` bits and stores `row[t]` densely into `kv_out[t]` (`[M, 576]`), the
  same bits `k3_mla_qkv` stores into the pool (certified).
- `fused_q` rows are computed in chunks of 8, 16 or 32 tokens by call size; every 8-token chunk of rows is
  bit-identical to the call on those 8 rows alone (the op test, `test_k3_mla_q.py::test_q`).

Accuracy (certified bounds): `fused_q` within `2e-2` (max relative, per half) of a float64 reference that keeps the
model's bf16 roundings; the latent columns of `row` within `1e-2` of an fp32 reference (`(x * rrms) * w`, one bf16
rounding); the rope columns bit-exact.

Fusion boundary. Inside: both RMSNorms, the q_b GEMM, the k_b absorption, the cache append. Outside: the fused
projection that produced `ag` (`[q_a 1536 | kv_a latent 512 | rope 64 | gate heads * 128]`), the attention that reads
`fused_q` and the cache (`attention/k3_mla_attn_vb_out`), and the output projection.

## Signature

```python
def k3_mla_qkv(
    ag: torch.Tensor,
    w_qa: torch.Tensor,
    eps: float,
    w_qb: torch.Tensor,
    w_kb: torch.Tensor,
    w_kv: torch.Tensor,
    kv_eps: float,
    pool: torch.Tensor,
    row_stride: int,
    page_table: torch.Tensor,
    page_offset: int,
    seq_len: torch.Tensor,
    trigger_early: bool = True,
) -> torch.Tensor

def k3_mla_q(ag, w_qa, eps, w_qb, w_kb, trigger_early=True) -> torch.Tensor

def k3_mla_qkv_out(ag, w_qa, eps, w_qb, w_kb, w_kv, kv_eps, kv_out, trigger_early=True) -> torch.Tensor
```

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `ag` | `[M, C]`, `M` = `R T` up to 64; `C >= 2112`, a multiple of 8 (Kimi K3: `C = 2112 + heads * 128`) | bf16 | contiguous | CUDA |
| `w_qa`, `w_kv` | `[1536]`, `[512]` | bf16 | contiguous | CUDA |
| `eps`, `kv_eps` | scalar | Python float | — | — |
| `w_qb` | `[heads * 192, 1536]`, `heads` 6 (TP16) or 24 (TP4); the op takes 2 to 24 | bf16 | contiguous | CUDA |
| `w_kb` | `[heads, 512, 128]` (`k_b_proj_trans`) | bf16 | contiguous | CUDA |
| `pool` | the layer's paged latent pool, flat; at least 64 rows | bf16 | contiguous | CUDA |
| `row_stride` | elements per cache row, `>= 576`, a multiple of 8 (576 for the manager's MLA pool) | Python int | — | — |
| `page_table` | `[R, W]` int32 with unit column stride and any row stride `>= W` (rows of `kv_cache_block_offsets`), or `[W]` when `R = 1` | int32 | see shape | CUDA |
| `page_offset` | the layer's slot in a layer-interleaved pool, added to every page-table entry | Python int | — | — |
| `seq_len` | `[R]`, each request's length including its `T` new tokens | int32 | contiguous | CUDA |
| `trigger_early` | default `True`: dependents may launch early (they wait for the whole grid before reading) | Python bool | — | — |
| `kv_out` (`k3_mla_qkv_out`) | `[M, 576]` | bf16 | contiguous | CUDA |
| returns | `fused_q` `[M, heads * 576]`, newly allocated | bf16 | contiguous | = `ag.device` |

Certified at `R` up to 8 and `T` 1 and 8 (no speculation; a DSpark verify step), 6 and 24 heads, over a real
`KVCacheManager` (below).

## State

**Object.** None of its own (kind P: it writes the caller's cache). `pool` is the KV cache manager's MLA latent
pool: `KVCacheManager.get_buffers(layer)` of a `SELFKONLY` manager (one latent head of 576, `kv_factor` 1, 64 tokens
per block), owned and sized by the manager. The addressing (`pool`, `row_stride`, `page_table`, `page_offset`,
`seq_len`) is the attention metadata's view of one generation step: `k3_mla_decode_view(attn, metadata, M)`
(`attention/backends/fmha/cute_dsl_mla.py`), computed per layer after `metadata.prepare()`.

**What a call writes.** One 576-column row per token, at that token's position in its request (positions below 0
skipped); nothing else (certified, above). The manager keeps every layer in one pool, interleaved by block: a layer's
rows sit in its slot of each block, and `page_offset` names the slot.

**Which calls may share one object.** Every layer's call writes the same pool, kept apart by `page_offset` (each
layer's slot) and by position (each step's tokens). Two managers, such as two models' caches, are independent
(certified: calls alternating between two managers step by step and layer by layer; each pool holds exactly its own
rows).

**Call order.** Calls of one step write disjoint rows, so their order does not matter; a step's rows must be stored
before the attention of the same layer reads them, which it does after its grid-dependency wait, so the attention
may directly follow this call.

**What a launch reads before its grid-dependency wait.** The page-table rows and the lengths: host-filled metadata
buffers (`kv_cache_block_offsets`, `kv_lens_cuda_runtime`, written by `TrtllmAttentionMetadata.prepare()` before the
step runs or its graph replays). They must not be produced by the kernel directly before this one in the stream.
`ag` is read after the wait.

**Re-arm.** Nothing to re-arm.

**Why the test drives call sequences.** A misplaced row is silent: it lands in another layer's slot or another
position, and only a later read sees it. The test therefore keeps each layer's expected image from the manager's own
block ids (`get_batch_cache_indices`), not from the op's addressing, and compares the whole pool of every layer after
every step: 3 layers x 2 decode steps eagerly, then one step captured as a CUDA graph and replayed for 2 more steps
with rewritten inputs and the metadata prepared for each step.

**What a wrong slot does.** Measured (the test's negative control): layer 1's call given layer 0's `page_offset`.
Nothing raises; layer 0's rows at the step's positions now hold layer 1's values, and layer 1's slot is not written.
The op trusts `page_offset`.

## Metadata consumed

Through `k3_mla_decode_view`: request `i`'s pages are `kv_cache_block_offsets[pool, num_contexts + i, 0, :]` (a
strided view, no copy), its length `kv_lens_cuda_runtime[num_contexts + i]`, `page_offset` the layer's index in the
pool, `row_stride` the pool's token stride. The view applies to generation-only steps (`num_contexts == 0`) of `R
<= 8` requests of the same `T <= 8` tokens, beam width 1, no tree speculation, 64-token pages, a bf16 pool, and
`heads` a multiple of 6 with `kv_lora_rank` 512 and `qk_rope_head_dim` 64; otherwise it returns a reason string and
the target takes its generic path.

## Preconditions

- `M <= 64`, and `page_table` / `seq_len` describe `R` requests of `T = M / R` tokens (`M % R == 0`; a `[W]` table
  only when `R = 1`); `heads * 6 <= 148` (at most 24 heads) and, with the cache half, `heads >= 2`. Out of contract
  calls raise `ValueError` before any launch, the pool unchanged (certified).
- `seq_len[i] >= T` for a request whose tokens should all be stored; a smaller length stores only the positions
  `>= 0` (the op test, `test_qkv_short_length`).
- The first call of each configuration compiles the kernel and must run outside CUDA-graph capture (it raises
  `RuntimeError` inside one).
- SM 100 / SM 103 only (CuTe DSL `tcgen05` kernel).

## Notes

- The op-level test is `tests/unittest/_torch/cute_dsl_kernels/kimi_k3/test_k3_mla_q.py`: `fused_q` at `M` 1 to 64
  against the reference and the unfused chain (the model's RMSNorm, the q_b GEMM, the k_b bmm), the cache rows of R
  x T steps in a sentinel-filled pool, an interleaved pool of rows of 640, the short-length case, refusals, and (with
  `K3_BASE_TRTLLM`) batch-1 identity against the unmodified single-request kernel.
- The weights are TMA'd before the grid-dependency wait (EVICT_FIRST); the q_a and kv_a rows after it.
