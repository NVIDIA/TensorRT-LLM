---
name: trtllm-modeling-v2-kv-quant
description: Load when switching a target's KV pools from bf16 containers of the reference's fake-quantized values to the checkpoint's native storage formats (for example FP4 compressed KV, MXFP4 indexer keys, FP8 sliding-window KV). Output-identical by construction when attention dequantizes exactly, so boot and accuracy cannot see it; it needs a bytes-per-token witness against a declared figure.
---

# trtllm-modeling-v2-kv-quant

`--features kv_quant`. Core stores the **same values** the reference computes —
the reference fake-quantizes KV to its storage formats and attends over the
dequantized result — but in bf16 pools, typically because the attention
kernels that serve the model are exact only over bf16 KV. This unit switches
the pools to the formats the checkpoint was trained for. It is what a request
for "FP4 KV cache", "FP8 attention" or "N bytes of KV per token" actually asks
for.

## The rule: store the reference's value set, never a different one

Real storage of the value set the reference already fake-quantizes to is
**lossless relative to the reference** — the model was trained (QAT) with
exactly these values. Storage of any *other* value set is a lossy
substitution, however close its bit budget, and must be named as one in the
plan, in the spec (`exact: false`), and in the report.

Substitutions that look like this feature and are not:

| format | why it is a different value set |
|---|---|
| `kv_cache_config.dtype: fp8` as a per-tensor FP8 cache | one scalar scale, not a block scale per group of channels |
| an in-tree preset with the same bit budget | its scale type, scale granularity, quantized dims (is RoPE included, and before or after rotation?) or second-level global scale differ from the reference's; compare each, never the bit width alone |

The sparse-attention cache machinery in
`tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/` — the
`KVCacheDtype` presets in `compressor.py`, `DeepseekV4CacheManager` and its
`get_cache_bytes_per_token` in `cache_manager.py` — is the closest thing in
tree to a multi-pool quantized KV cache. Read it to learn the interface; do
not assume its byte formulas describe the checkpoint in hand.

## Deriving the checkpoint's formats

From the checkpoint's tech report and the reference's fake-quantization, write
down one row per KV pool before touching the target:

| pool | who owns one | format | bytes per entry |
|---|---|---|---|
| e.g. sliding-window KV | which layers, and whether only the window is held | element type, scale type and group size, which dims | payload + scales |
| e.g. compressed global KV | which layers **own** a pool, and at what compress ratio | | |
| e.g. indexer keys | | | |

**Find out which layers own a pool and which read another layer's.** A model
whose layers share pools through a source-layer mapping (a config key such as
`kv_source_layer_ids`) allocates far fewer pools than it has layers. A pool
allocated per layer multiplies the figure, and is the sign the sharing was
never wired.

**Bytes per token** is the persistent, growing part, global KV only: for each
pool-owning layer, its entry size divided by its compress ratio, summed. The
same formula over bf16 containers of the same values gives core's figure,
which the contrast leg must exceed. A sliding window is per request, not per
token, and so is any compressor carry-over state; neither belongs in the
figure.

## Attention over native storage

The attention kernels that serve the model may be exact only over bf16 KV.
The native formats are then consumed by a kernel that reads them directly,
where one exists, or by a **dequantize-and-gather to bf16** step ahead of the
kernel. So:

- native storage + exact dequantization + bf16 Q is **output-identical** to
  core. Declare `exact: true` (the default) and the witness demands token
  identity with bf16 storage;
- an FP8-query kernel is not: FP8 Q is an approximation beyond the reference.
  Declare `exact: false`, say so in the plan, and accuracy becomes the only
  output check. The verdict records that it is the weaker claim.

## Gates

boot → accuracy → **kv_quant witness**, all under `modeling_v2=require`.

**The counter.** When `TRTLLM_MODELING_V2_WITNESS=1` is set, the target prints,
once per rank after its KV pools are allocated:

```
MODELING_V2_WITNESS kv_bytes_per_token=656
```

Computed from the pools **as allocated** — the tensors' dtypes and shapes,
summed over pools, a ratio-r pool contributing its entry size over r — for one
token of a request this rank serves. Not typed in: a constant print passes the
native leg and is caught by the contrast leg, which must report more. SWA and
compressor state are per request and excluded.

**The verdict**, against the spec's declared figure:

- native storage reports `expected_bytes_per_token` within `bytes_tolerance`
  (default 2%) — enough for row padding, not for a layout change: padding a
  4-byte scale to 16 bytes on one pool already moves the figure by a few
  percent;
- with `exact: true`, the leg under `contrast_env` (bf16 storage) reports MORE
  bytes per token — otherwise it selected nothing and the next check compares
  native storage with itself — and the greedy continuations are
  **token-identical**. A difference means the stored values are not the
  reference's: a layout, scale or rounding from some other format. Both legs
  print their token ids under `TRTLLM_WITNESS_FULL_OUTPUT=1`
  (`[i] Witness token ids: [...]` from the tree's
  `examples/llm-api/quickstart_advanced.py`); text that agrees without them
  is not a pass.

```yaml
features: [kv_quant]
accuracy:
  witness:
    kv_quant:
      contrast_env: {MYMODEL_KV_STORAGE: bf16}   # the leg WITH bf16 storage
      expected_bytes_per_token: 656              # derived above, before the run
      exact: true
      # bytes_tolerance: 0.02
      max_batch_size: 1   # both legs one request at a time
```

Compare at batch 1 -- `max_batch_size: 1` in the witness block runs both legs
there: a model's output is often not bitwise stable across batch composition,
and the identity check must measure storage, not batching.

## What the gates still do not tell you

Whether the dequantize step is fast enough, and how much KV capacity the unit
bought. Both are perf and capacity measurements; the witness says only that
the pools hold the checkpoint's formats and that the output did not move.

## Report additions

- each pool: owner layers, format, bytes per entry, and the allocated dtype
- measured bytes per token against the declared figure, and the bf16 leg's figure
- `exact`, and if false, which attention path is lossy and why it was chosen
- how attention reads the native values (dequantize-gather, or which kernel)
- confirmation that only the pool-owning layers allocate pools
