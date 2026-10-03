---
receipts:
  sm_100: {status: passed, tests: 36}
---

# k3_drafter_attn

**Wraps** `torch.ops.trtllm.k3_drafter_attn` (one call), a CuTe DSL kernel
(`tensorrt_llm/_torch/cute_dsl_kernels/k3_drafter/`).

## Semantics

The block attention of Kimi K3's DSpark drafter. `R = ctx_len.numel()` requests each bring a draft block of
`T = M / R` tokens; the `M` rows of `qkv` are the requests' blocks in order. Per request `r`, row `i` of its block
and query head `h` (groups of `num_heads / num_kv_heads` = 6 query heads per KV head `g(h)`), head dim 64:

```
K_r, V_r = [rows 0 .. ctx_len[r] - 1 of request r's cache pages] ++ [the block's own k, v rows of qkv]
out[r T + i, h] = softmax(q[r T + i, h] . K_r[g(h)]^T / 8) V_r[g(h)]
```

Within a block every row attends to every row (non-causal). The block's own `k` / `v` are read from `qkv`, not
from the cache, and are not written into it. `out` is written; `qkv`, `cache`, `page_table` and `ctx_len` are read
only (the cache certified bit for bit). Each request's rows depend only on its own context and block (the op's own
test checks this).

Fusion boundary. Inside: the paged gather of each request's context `K` / `V`, the block's own `K` / `V`, the
softmax over both, and the product with `V`. Outside: storing the block's `K` / `V` in the cache, if the caller keeps
them, and the q / k norm and RoPE (`attention/k3_drafter_attn_qknorm` fuses those in).

## Signature

```python
def k3_drafter_attn(
    qkv: torch.Tensor,
    cache: torch.Tensor,
    page_table: torch.Tensor,
    ctx_len: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    out: torch.Tensor,
) -> None
```

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `qkv` | `[M = R T, (num_heads + 2 num_kv_heads) * 64]`: q heads, then k heads, then v heads; `R` <= 8, `T` <= 8 | bf16 | contiguous | CUDA |
| `cache` | `[pages, 2, num_kv_heads, 64, 64]` (HND pages of 64 rows, `K` then `V`) | bf16 | each page dense; the page stride may exceed a page (a multiple of 8 elements) | CUDA |
| `page_table` | `[>= R, width]`, row `r` = request `r`'s pages, dense rows at any row stride (e.g. a view of a block table); or `[width]` when `R` = 1 | int32 | — | CUDA |
| `ctx_len` | `[R]`, cached rows per request, 0 allowed | int32 | — | CUDA |
| `num_heads`, `num_kv_heads` | 6 / 1 (Kimi K3 at TP16) and 24 / 4 (TP4) certified; `num_heads` = 6 `num_kv_heads` | Python int | — | — |
| `out` | `[M, num_heads * 64]` | bf16 | contiguous | CUDA |

Certified splits `R x T`: 1x1, 1x8, 2x4, 4x2, 8x1, 3x1, 2x8, 8x8, and 1x7 to 8x7 (DSpark's block at
`max_draft_len` 7), with context lengths 0 to 2041 (blocks crossing a page, 64- and 128-row boundaries, 2041 rows
spanning the kernel's 16-tile round), at both head layouts.

## Metadata consumed

None. The op compiles on its first call for each (`num_heads`, `num_kv_heads`, page stride, `R` > 1, PDL) and keeps
the result in a process-wide cache; that first call must happen outside CUDA-graph capture (it raises
`RuntimeError` there). The lengths, page tables and `qkv` are read on the device, so a captured call replays with
them rewritten in place (certified: 4 replays with new lengths, rotated page-table rows and new `qkv`).

## Preconditions

- SM100 (tcgen05, TMA, clusters).
- The shapes and dtypes above; a call outside them raises `ValueError` before any launch (certified: 12 query heads
  per KV head, one page-table row for two requests, page-table columns not dense, int64 lengths, an fp32 output,
  `T` = 16).
- Page-table entries below `ceil(ctx_len[r] / 64)` name pages holding request `r`'s rows; later entries and the
  rows of the last page past `ctx_len[r]` are not read (certified with those rows NaN: the output has no NaN).

## Notes

- Certified on GB200 (sm_100) against an fp64 torch reference that gathers the pages, within 1e-2 of the largest
  output magnitude (the kernel rounds the probabilities and the output to bf16). Reruns are bit-identical.
- The op's own test (`tests/unittest/_torch/cute_dsl_kernels/kimi_k3/test_k3_drafter_attn.py`) also compares every
  split against the DFlash TRTLLM path it replaces (`append_paged_kv_cache` + the trtllm-gen context kernel,
  non-causal) and checks request isolation.
