# Q8KV4 paged sparse prefill

Prefill attention for the MiniMax sparse-attention (MSA) NVFP4 KV cache on SM100, SM103 and
SM107: E4M3 queries, E2M1 keys and values with E4M3 block scales, TopK-selected pages, chunked
prefill with bottom-right causal alignment. `fmha_sm100_plan` / `fmha_sm100` route NVFP4 sparse
prefill here automatically for E4M3 queries; this page documents that routing and the package's
own API.

The kernel is KV-stationary: one CTA takes one (KV head, page) pair and the queries whose TopK
lists select that page, dequantizes the page to E4M3 once, and writes one split (normalized output
and log-sum-exp) per query and selected page; the CuTe-DSL combine then merges each query's
splits. The k2q CSR, the work schedule and the combine are those of `fmha_sm100`'s CuTe-DSL
sparse prefill.

## Requirements

- CUDA Toolkit 13.4 or newer and an SM100, SM103 or SM107 GPU. The dequantization uses the
  QMUL4 instruction where the toolchain accepts it for the target (SM100 and SM103; the JIT
  probes `nvcc` once per toolchain and architecture) and an exact packed-FP16 path otherwise
  (SM107, which has no QMUL4). Both paths produce the same E4M3 values.
  `FMHA_SM100_PREFILL_BF16X_DISABLE_QMUL4=1` compiles the FP16 path on any GPU.
- CUTLASS headers: the repository submodule (4.8) or any newer release through `CUTLASS_ROOT`;
  SM107 needs 4.8 or newer.
- The CuTe-DSL sparse stack (`nvidia-cutlass-dsl`, `quack-kernels`) for the CSR builder and the
  combine.
- The kernel is JIT-compiled on first use into `MINFER_FMHA_CACHE_DIR` (or
  `TORCH_EXTENSIONS_DIR`, or `~/.cache/minfer/fmha_sm100`), one extension per block-scale shift.
  `FMHA_SM100_PREFILL_BF16X_ARCH` selects the target for offline builds without a device
  (`python -m tensorrt_llm._torch.attention.backends.fmha.msa_kernels.prefill_bf16x.build --arch 103a`).

## Through `fmha_sm100`

```python
plan = fmha_sm100_plan(qo_lens, kv_lens, num_q_heads, num_kv_heads=num_kv_heads,
                       page_size=128, kv_block_num=topk,
                       prefill_backend="auto",     # or "q8kv4" / "cute_dsl"
                       block_scale_shift=3)        # TransformerEngine block-scale convention
out, _ = fmha_sm100(q_e4m3, k_cache, v_cache, plan, kv_indices=kv_indices,
                    kv_block_indexes=kv_block_indexes, q_scale=q_dequant_scale,
                    k_scale=k_global, v_scale=v_global)
```

`k_cache` / `v_cache` are the K and V slot views `cache[:, 0::2]` / `cache[:, 1::2]` of a
uint8 `[pages, 2 * Hkv, 128, 72]` NVFP4 cache of per-head K/V slots; `k_scale` /
`v_scale` are one-element fp32 CUDA tensors. A sparse prefill plan (batches whose longest query
chunk exceeds 32 tokens, or `sparse_kernel_mode="prefill"`) is marked for this kernel when the
batch fits it: page size 128, 16 Q heads per KV head, 4, 8, 16 or 32 blocks, causal, no max-score
output, an SM100/SM103/SM107 device and a CUDA 13.4+ toolkit. `fmha_sm100` then runs uint8
caches on it when Q is E4M3 (contiguous) and the global scales are one-element fp32 tensors;
`q_scale` folds into the softmax scale, `o_scale` into the output, and `qo_offset` /
`q_offset_override` move the causal alignment. BF16 Q, dense caches and batches that do not fit
keep the CuTe-DSL NVFP4 kernel;
`prefill_backend="q8kv4"` raises instead of falling back, `"cute_dsl"` never plans this kernel,
and `kv_dtype="fp8"` skips it for FP8 caches. Mixed batches (`split_prefill_decode`) route their
prefill part here and their decode part to the Q8KV4 decode kernel.

## Direct API

```python
from tensorrt_llm._torch.attention.backends.fmha.msa_kernels.prefill_bf16x import BatchPrefillWithPagedKVCacheWrapper, interleave_v_scales

wrapper = BatchPrefillWithPagedKVCacheWrapper()
wrapper.plan(topk_indices, cu_seqlens_q, cu_seqlens_k, kv_indices, kv_indptr=kv_indptr,
             total_k=total_k, total_rows=total_rows, max_seqlen_q=max_q, max_seqlen_k=max_k,
             block_scale_shift=0)
out, lse = wrapper.run(q, (k_data, v_data), kv_cache_sf=(k_scale, v_scale),
                       kv_global_scale=(k_global, v_global), return_lse=True)
```

`plan` builds the k2q CSR and schedule from one layer's TopK lists outside CUDA Graph capture and
reserves the split workspace; `run` may be captured and replayed. `run_prefill` runs the
forward and the combine over a CSR and schedule the caller built with `build_k2q_csr(...,
return_schedule=True)` (that is what `fmha_sm100` does per layer), so a caller that already
builds them for the CuTe-DSL kernel swaps only the attention call:

```python
from tensorrt_llm._torch.attention.backends.fmha.msa_kernels.prefill_bf16x import run_prefill

run_prefill(q_e4m3, (k_data, v_data), (k_sf, v_sf), kv_indices=block_table, kv_indptr=None,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, k2q_row_ptr=k2q_row_ptr,
            schedule=schedule, sm_scale=softmax_scale * q_scale, topk=topk, seqused_k=seq_lens,
            k_global_scale=k_global, v_global_scale=v_global, block_scale_shift=3, out=out)
```

### Split workspace

The forward writes one partial output and LSE per (query, head, selected page) and the combine
merges them. `PrefillWorkspace` owns those buffers: a wrapper creates its own, wrappers that run
in order on one stream may share one, and `run_prefill` takes one through `workspace=` (without
it the partials are fresh allocations per call).

```python
from tensorrt_llm._torch.attention.backends.fmha.msa_kernels.prefill_bf16x import PrefillWorkspace

workspace = PrefillWorkspace(device)
workspace = PrefillWorkspace(device, partial_bytes=workspace.required_bytes(topk, max_total_q, Hq))
run_prefill(..., workspace=workspace)
```

On a GPU with several memory locality domains (Rubin, Blackwell) the workspace stripes the
partial buffer over the domains and the combine reads every tile from the domain of its own SM,
which raises its bandwidth where the inter-domain fabric is the limit; the kernels and results
are the same as with interleaved memory. Reserve the largest batch up front with
`partial_bytes`: a batch that outgrows a striped buffer synchronizes the device before replacing
it. `FMHA_SM100_PREFILL_BF16X_LOCALITY_DOMAINS=0` keeps the buffer interleaved (for
comparisons). One workspace must not be used from two streams at once.

## Data contract

- `q`: `[total_q, Hq, 128]` E4M3, contiguous; request `b`'s query chunk occupies
  `cu_seqlens_q[b]:cu_seqlens_q[b + 1]` and is the last part of its KV sequence.
- `k_data` / `v_data`: `[pages, Hkv, 128, 64]` uint8, two E2M1 values per byte; token rows
  contiguous, page and head strides free (multiples of 16 bytes), so packed pages are read in
  place.
- `k_scale` / `v_scale`: `[pages, Hkv, 128, 8]` E4M3 (or uint8) block scales, one per 16 values,
  same stride rules. K blocks are linear (`token * 8 + group`); V blocks are in token-quad order
  (`(token // 4) * 32 + group * 4 + token % 4`, vLLM's layout); `interleave_v_scales` converts a
  linear tensor.
- Global scales: `value = code x block_scale x global_scale`; the kernel reads the one-element
  fp32 tensors on the device, so captured graphs follow updated tensors. Omitted means 1.0.
- `block_scale_shift`: 0 when `code x block_scale` already fits E4M3; 3 when block scales use the
  full E4M3 range next to a global scale. The kernel divides the block scales by `2 ** shift`
  before the product and folds the factor back exactly; block scales below `2 ** (shift - 6)`
  round to E4M3 subnormals.
- Pages: a `[B, max_pages]` int32 table whose rows are contiguous but may sit at any row stride
  (vLLM's block table can be passed as is), or a flat int32 list `kv_indices` with `kv_indptr`
  (`[B + 1]`) giving each request's first entry. `cu_seqlens_k` gives the KV lengths; an optional
  `seqused_k` (`[B]`) replaces them for masking and causal alignment.
- `topk_indices`: `[Hkv, total_q, topk]` int32 logical page ids relative to the request (the
  wrapper's layout; `fmha_sm100` takes `[total_q, Hkv or Hq, topk]`), valid entries first, then
  `-1`. Each valid entry is one split: the query's own page is masked to the tokens up to the
  query, earlier pages are read in full, later pages and pages past the request contribute
  nothing. A query without a visible page gets zeros and an LSE of `-inf`.
- `out`: `[total_q, Hq, 128]` bf16; `lse`: `[total_q, Hq]` fp32, natural log of the true logits.

Numerics: QK and PV run on the FP8 tensor core; probabilities are quantized as
`E4M3(p * 448)` per split and the normalizer sums them unquantized; splits are stored in bf16.
Runs are bitwise deterministic for identical inputs. Supported: head dim 128, page size 128,
16 Q heads per KV head, causal only.

## Tests and benchmark

```bash
python -m pytest tests/q8kv4_prefill -q -m "not full"   # smoke set
python -m pytest tests/q8kv4_prefill -q                  # full matrices
python benchmarks/bench_q8kv4_prefill.py --suite full
```

The tests generate their inputs and carry an fp32 reference that mirrors the kernel's roundings;
the benchmark compares this kernel with the CuTe-DSL NVFP4 kernel (E4M3 and BF16 Q) through the
same `fmha_sm100` calls and needs an exclusive GPU.
