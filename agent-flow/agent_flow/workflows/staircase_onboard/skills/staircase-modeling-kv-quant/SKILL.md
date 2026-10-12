---
name: staircase-modeling-kv-quant
description: Load when switching a target's KV pools from bf16 containers of the reference's fake-quantized values to the checkpoint's native storage formats (DeepSeek-V4.1's FP4 main KV, MXFP4 indexer K, FP8 SWA KV). Output-identical by construction when attention dequantizes exactly, so boot and accuracy cannot see it; it needs a bytes-per-token witness against a declared figure.
---

# staircase-modeling-kv-quant

`--features kv_quant`. Core stores the **same values** the reference computes —
the reference fake-quantizes KV to its storage formats and attends over the
dequantized result — but in bf16 pools, because the only sm_103 sparse
attention kernels that serve DeepSeek-V4.1 are exact only over bf16 KV. This
unit switches the pools to the formats the checkpoint was trained for. It is
what a request for "FP4 KV cache", "FP8 attention" or "890 bytes of KV per
token" actually asks for.

## The rule: store the reference's value set, never a different one

Real storage of the value set the reference already fake-quantizes to is
**lossless relative to the reference** — the model was trained (QAT) with
exactly these values. Storage of any *other* value set is a lossy
substitution, however close its bit budget, and must be named as one in the
plan, in the spec (`exact: false`), and in the report.

Substitutions that look like this feature and are not:

| format | why it is a different value set |
|---|---|
| `kv_cache_config.dtype: fp8` as a per-tensor FP8 cache | one scalar scale, not UE8M0 per 32 channels |
| `fp8_ds_mla` — DeepSeek-V4's 584 B layout | 448 FP8 + 64 bf16 RoPE dims with block-64 scales; V4.1 quantizes all 512 dims, RoPE included, after RoPE (`tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/footer_scale_kv.py` is the V4 layout) |
| TRT's `nvfp4` compressed pool as DeepSeek-V4 uses it | NVFP4 with a second-level global scale and a 64-dim residual; V4.1 has no global scale (it is 1) and all 512 dims in FP4 |

The DeepSeek-V4 machinery in
`tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/` — the
`KVCacheDtype` presets in `compressor.py`, `DeepseekV4CacheManager` and its
`get_cache_bytes_per_token` in `cache_manager.py` — is the closest thing in
tree, and it encodes the **V4** layouts. Read it to learn the interface; do
not assume its byte formulas describe V4.1.

## DeepSeek-V4.1's formats

From the tech report (§2.4.4) and the reference's fake-quantization:

| pool | who owns one | format | bytes per entry |
|---|---|---|---|
| SWA KV | every layer (40), the window only | FP8 E4M3 per 32 channels with a UE8M0 scale, over all 512 dims including RoPE | 512 + 16 = **528** |
| main (compressed global) KV | the KV-source layers only: 2, 8, 14 (compress ratio 2) and 20 (ratio 1) | FP4 E2M1 with an E4M3 scale per 16, **no global scale** | 256 + 32 = **288** |
| indexer K | the same KV-source layers | MXFP4: E2M1 with a UE8M0 scale per 32, 128 dims | 64 + 4 = **68** |

**Only layers 2, 8, 14 and 20 own compressed and indexer pools.** Every other
layer reads a source layer's pool through `kv_source_layer_ids`. A pool
allocated per compressed layer (38 of them) comes to 10324 B/token, more than
ten times the figure below, and is the sign the sharing was never wired.

**Bytes per token** is the persistent, growing part — global KV only:

```
3 x (288 + 68) / 2  +  (288 + 68)  =  534 + 356  =  890 B/token
   ratio-2 layers       ratio-1 layer
```

The tech report's headline figure. The same formula gives 3200 for bf16
containers of these values (core) and 1490 with FP8 main KV. The SWA window
is per request, not per token — 40 layers x 128 x 528 B, about 2.7 MB — and so
is the compressor's carry-over state; neither belongs in the figure.

## Attention over native storage

The kernels that serve V4.1 on sm_103 — trtllm-gen's H512 DynamicTokenSparse
path through FlashInfer's `trtllm_batch_decode_sparse_mla_dsv4`, and the
catalog's `thop_attention` — are exact only over bf16 KV. The native formats
are consumed by SGLang's FlashMLA fork, which TRT does not vendor, or by a
**dequantize-and-gather to bf16** step ahead of the kernel. So:

- native storage + exact dequantization + bf16 Q is **output-identical** to
  core. Declare `exact: true` (the default) and the witness demands token
  identity with bf16 storage;
- an FP8-query kernel is not: FP8 Q is an approximation beyond the reference.
  Declare `exact: false`, say so in the plan, and accuracy becomes the only
  output check. The verdict records that it is the weaker claim.

## Gates

boot → accuracy → **kv_quant witness**, all under `TRTLLM_MODELING_V2=require`.

**The counter.** When `TRTLLM_MODELING_V2_WITNESS=1` is set, the target prints,
once per rank after its KV pools are allocated:

```
MODELING_V2_WITNESS kv_bytes_per_token=890
```

Computed from the pools **as allocated** — the tensors' dtypes and shapes,
summed over pools, a ratio-r pool contributing its entry size over r — for one
token of a request this rank serves. Not typed in: a constant print passes the
native leg and is caught by the contrast leg, which must report more. SWA and
compressor state are per request and excluded.

**The verdict**, against the spec's declared figure:

- native storage reports `expected_bytes_per_token` within `bytes_tolerance`
  (default 2%) — enough for row padding, not for a layout change: padding the
  4-byte indexer scale to 16 moves 890 to 920;
- with `exact: true`, the leg under `contrast_env` (bf16 storage) reports MORE
  bytes per token — otherwise it selected nothing and the next check compares
  native storage with itself — and the greedy continuations are
  **token-identical**. A difference means the stored values are not the
  reference's: a layout, scale or rounding from some other format.

```yaml
features: [kv_quant]
accuracy:
  witness:
    kv_quant:
      contrast_env: {TRTLLM_DSV41_KV_STORAGE: bf16}   # the leg WITH bf16 storage
      expected_bytes_per_token: 890
      exact: true
      # bytes_tolerance: 0.02
      max_batch_size: 1   # both legs one request at a time
```

Compare at batch 1 -- `max_batch_size: 1` in the witness block runs both legs
there: SGLang reports that this model's output is not bitwise stable across
batch composition, and the identity check must measure storage, not batching.

## What the gates still do not tell you

Whether the dequantize step is fast enough, and how much KV capacity the unit
bought. Both are perf and capacity measurements; the witness says only that
the pools hold the checkpoint's formats and that the output did not move.

## Report additions

- each pool: owner layers, format, bytes per entry, and the allocated dtype
- measured bytes per token against 890, and the bf16 leg's figure
- `exact`, and if false, which attention path is lossy and why it was chosen
- how attention reads the native values (dequantize-gather, or which kernel)
- confirmation that only the KV-source layers allocate compressed and indexer
  pools
