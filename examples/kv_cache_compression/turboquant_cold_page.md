<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TurboQuant Cold-Page KV-Cache Compression (Prototype)

TurboQuant cold-page compression stores supported Attention KV buffers with
[TurboQuant](https://arxiv.org/abs/2504.19874) at 2 to 4 bits per value while
their Pages reside in a Host or Disk cache tier. Like
[NVFP4 cold-page compression](nvfp4_cold_page.md), it is a storage-boundary
optimization: the active GPU cache and Attention kernels continue to use the
model's normal runtime KV type.

For an overview of all available compression methods, see
[KV Cache Compression](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kv-cache-compression.md).

## Motivation

A cold tier that holds more reusable prefixes avoids recomputing them. NVFP4
cold Pages need 4.5 bits per value; TurboQuant reaches 2 to 4 bits per value,
plus a few bytes of scale per vector, so the same Host or Disk quota holds up
to about twice as many Pages and each offload or onboard moves fewer bytes. The
benefit is largest for workloads with substantial prefix reuse and real
cold-tier pressure, and it comes with a larger, setting-dependent quantization
error that must be validated for the workload.

## Feature Behavior

TurboQuant is data-oblivious: it needs no calibration and no checkpoint
metadata. For every quantized K or V vector, one per token and KV head, the
encoder rotates the vector with a random orthogonal matrix, replaces each
rotated coordinate with the index of the nearest value of a Lloyd-Max codebook,
and stores one BF16 scale for the vector. The decoder looks up the codebook
values, applies the scale, and rotates back.

| `turboquant_bits` | Algorithm | Use it for |
| --- | --- | --- |
| 2, 3, or 4 | The paper's TurboQuant-mse (Algorithm 1): the smallest reconstruction error for the bit width | K and V reconstruction whose error stays the same when a Page is re-encoded |
| 2.5 or 3.5 | Two Algorithm 1 instances per vector: in each group of vectors that share a page, layer, buffer, and KV head, half of the channels use one bit more than the other half; channels with more than four times the median channel energy are always in that half | The default (3.5 bits), where the paper reports quality neutrality for the KV cache; bit widths between the integer settings, with outlier channels kept more precise |

The fractional settings choose the high-bit half from the channel energies in
the Page being encoded: the outlier channels, whose energy exceeds four times
the median channel energy, and then the lowest-numbered other channels until
half of the channels are chosen. The split adapts to every Page without
calibration. The rotation matrices are fixed by `turboquant_seed` and generated
identically in every process.

The paper's TurboQuant-prod (Algorithm 2), which spends one bit on a QJL sign
of the residual so that inner products with decoded keys are unbiased, is not
offered. KVCacheManagerV2 re-encodes a Page each time it offloads the Page
again after an onboard, and the QJL error of `prod` compounds over these round
trips, whereas the error of TurboQuant-mse stays the same or settles after a few
round trips (see [Reconstruction Quality](#reconstruction-quality)).

Each cold Page is one compact storage blob. For conventional MHA, MQA, and GQA,
the K and V buffers are encoded; for key-only MLA, the latent Attention key is
encoded, including widths that are not a power of two, such as 576 values.
Values that are not quantized, such as RoPE when `skip_rope_quantization` is
`true`, are copied at the hot-cache precision into the same blob, and auxiliary
roles in the same Attention lifecycle, such as a DSA index key, are appended
losslessly. Non-Attention lifecycles, including GDN, SSM, and Conv state, use
the default lossless cold-page codec.

Disk migration uses pinned Host staging, exactly as for NVFP4: the Host and
Disk tiers share one TurboQuant representation.

| `turboquant_bits` | Bits per stored value, head dimension 128[^bits] | BF16 Page compression |
| --- | --- | --- |
| `2`, `2.5`, `3`, `3.5`, `4` | 2.125, 2.78, 3.125, 3.78, 4.125 | 7.5x, 5.8x, 5.1x, 4.2x, 3.9x |
| NVFP4, for comparison | 4.5 | 3.6x |

[^bits]: Including per-vector scales and, for 2.5 and 3.5
    bits, the channel masks at the default 32 tokens per Page. The cold-page
    size formula is in the
    [feature documentation](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kv-cache-compression.md#turboquant-cold-pages).

This is different from an active quantized KV cache. Do not set
`kv_cache_config.dtype: nvfp4` for this feature: TurboQuant decodes into an
FP16, BF16, or FP8 active KV cache, and TensorRT-LLM rejects the configuration
when the active KV cache is NVFP4. Cold-page compression does not change Page
identity, token count, block reuse, scheduling, or the Attention-visible GPU
layout.

## Supported Cache Types

| Cache type | Cold-page behavior |
| --- | --- |
| MHA, MQA, and GQA Attention KV | Supported; K and V are encoded with TurboQuant |
| Key-only MLA Attention KV | Supported; the latent Attention key is encoded with TurboQuant |
| GDN, SSM, and Conv state | Skipped by quantization and preserved losslessly |
| DSA and other auxiliary buffers | Skipped by quantization and preserved losslessly |
| DeepSeek-V4 CSA cache | Supported; each 512-value row is encoded with TurboQuant (only its 448 NoPE values with `skip_rope_quantization`), and the indexer cache is preserved losslessly |
| DeepSeek-V4 SWA, HCA, and compressor state | Preserved losslessly |

The current implementation requires the PyTorch backend, the native C++
KVCacheManagerV2, and a GPU with compute capability 8.0 or newer (8.9 or newer
for an FP8 active KV cache); its kernels are written in Triton. Hot Attention KV
can use FP16, BF16, or FP8. KV-cache
block reuse remains supported because Page and token identity are unchanged.
One-model MTP-EAGLE and EAGLE3 are supported: target and draft KVCMs encode
their cold Pages independently, and draft cold Pages always quantize whole
vectors. HELIX context parallelism is not currently supported.

Set `kv_cache_config.use_kv_cache_manager_v2: true` explicitly. A nonzero Host
or Disk cache is also required for Pages to cross a compression boundary.

### Tested Models

- Qwen3.8-Flash-Next (GQA with partial RoPE, QSA sparse attention, GDN layers)
- GLM-5.3-Flash (kpool sparse MLA latent, KDA layers)

This is a tested-model list, not an exhaustive support list. Other models that
use the supported cache types above are expected to work, subject to their
checkpoint, parallelism, Attention-backend, and hardware requirements.

## Qwen3-8B on One GPU

The following example uses a regular BF16 checkpoint. Attention KV stays BF16
on the GPU and uses 3-bit TurboQuant only while it resides in the 8 GiB Host
tier.

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (
    ColdPageQuantizationCompressionConfig,
    KvCacheConfig,
)

with LLM(
    model="Qwen/Qwen3-8B",
    backend="pytorch",
    max_seq_len=4096,
    max_batch_size=4,
    kv_cache_config=KvCacheConfig(
        use_kv_cache_manager_v2=True,
        dtype="auto",
        enable_block_reuse=False,
        host_cache_size=8 << 30,
    ),
    kv_cache_compression_config=ColdPageQuantizationCompressionConfig(
        quant="turboquant",
        turboquant_bits=3,
    ),
) as llm:
    outputs = llm.generate(
        ["Explain why prefix caching helps agentic workloads."],
        SamplingParams(max_tokens=128, temperature=0.0),
    )
    print(outputs[0].outputs[0].text)
```

The equivalent `trtllm-serve` configuration is:

```yaml
backend: pytorch

kv_cache_config:
  use_kv_cache_manager_v2: true
  dtype: auto
  enable_block_reuse: false
  host_cache_size: 8589934592  # 8 GiB

kv_cache_compression_config:
  algorithm: quantization_for_cold_page
  quant: turboquant
  turboquant_bits: 3     # 2, 2.5, 3, 3.5, or 4
```

```bash
trtllm-serve Qwen/Qwen3-8B --config qwen3-cold-turboquant.yaml
```

The same configuration is available from the LLM API example script:

```bash
python examples/llm-api/llm_kv_cache_compression.py \
  --compression-method turboquant-cold-page \
  --model Qwen/Qwen3-8B \
  --turboquant-bits 3
```

This smoke test disables block reuse and covers basic inference and
configuration; a short request can remain entirely on the GPU. To exercise the
cold tier, use the workload in the next section.

## Verify Activation

Configuration alone does not prove that a Page crossed the compression
boundary. The following script keeps the GPU KV cache small, enables block
reuse, pushes a set of long prefixes out to the Host tier, and then asks new
questions about the first prefixes so that their Pages are onboarded and
decoded again. It prints the KV-cache iteration statistics that show the
movement.

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (
    ColdPageQuantizationCompressionConfig,
    KvCacheConfig,
)

documents = [
    f"Document {index}: " + " ".join(f"fact-{index}-{word}" for word in range(400))
    for index in range(8)
]
sampling = SamplingParams(max_tokens=16, temperature=0.0)


def collect(llm):
    """Drain the iteration statistics recorded since the previous call and total them."""
    totals = {"offload": 0, "onboard": 0, "cached_tokens": {}, "cold_slot_sizes": set()}
    records = []
    while batch := llm.get_stats(timeout=2):
        records += batch
    for stats in records:
        for pool in stats.get("kvCacheIterationStatsByPoolGroup", {}).values():
            totals["offload"] += pool["iterOffloadBytes"]
            totals["onboard"] += pool["iterOnboardBytes"]
        tiers = stats.get("kvCacheLevelTiers", [])
        for tier, tokens in zip(tiers, stats.get("iterCachedTokensByLevel", [])):
            totals["cached_tokens"][tier] = totals["cached_tokens"].get(tier, 0) + tokens
        for pool in stats.get("kvCacheIterationStatsByColdPoolGroup", {}).values():
            totals["cold_slot_sizes"].update(pool["slotSize"])
    return totals


with LLM(
    model="Qwen/Qwen3-8B",
    backend="pytorch",
    max_seq_len=4096,
    max_batch_size=1,
    enable_iter_perf_stats=True,
    kv_cache_config=KvCacheConfig(
        use_kv_cache_manager_v2=True,
        enable_block_reuse=True,
        max_tokens=8192,  # a GPU tier that the documents overflow
        host_cache_size=8 << 30,
    ),
    kv_cache_compression_config=ColdPageQuantizationCompressionConfig(
        quant="turboquant",
        turboquant_bits=4,
    ),
) as llm:
    for document in documents:  # fill the GPU tier and offload the oldest prefixes
        llm.generate([document + "\nSummarize the document."], sampling)
    fill = collect(llm)
    for document in documents[:2]:  # reuse prefixes that now live in the Host tier
        llm.generate([document + "\nWhich fact comes first?"], sampling)
    reuse = collect(llm)  # only the two reuse requests

    print("fill: offload bytes:", fill["offload"])
    print("reuse: onboard bytes:", reuse["onboard"])
    print("reuse: reused prompt tokens by tier:", reuse["cached_tokens"])
    print("cold slot sizes:", sorted(fill["cold_slot_sizes"] | reuse["cold_slot_sizes"]))
```

Look for these signals:

* Offload bytes (`iterOffloadBytes`) of the fill phase become nonzero while the
  later documents push the earlier prefixes out of the GPU tier, and onboard
  bytes (`iterOnboardBytes`) of the reuse phase become nonzero when the last two
  requests reuse them.
* The reused prompt tokens of the reuse phase, which holds only the last two
  requests, are attributed to the `host` tier (`iterCachedTokensByLevel`,
  labeled by `kvCacheLevelTiers`).
* The cold slot size (`slotSize` of `kvCacheIterationStatsByColdPoolGroup`) is
  the TurboQuant cold-page size rather than the hot Page size: about 1.2 MiB
  instead of 4.5 MiB for Qwen3-8B (36 layers, 8 KV heads, head dimension 128)
  at 4 bits. This shows that the cold tier holds TurboQuant Pages, not lossless
  copies.

For a scripted version of this check on a multi-GPU model, see
`examples/kv_cache_compression/cold_tier_accuracy.py`: it prints the offload and onboard bytes,
the per-tier reuse, and the cold and hot page sizes, and marks a run invalid when no prefix
block was reused from a cold tier.

With `trtllm-serve`, run enough concurrent or reusable long-context requests to
create GPU KV pressure, then inspect the server's `/metrics` endpoint. Both of
the following counters should become nonzero when the workload offloads and
later reuses cold Pages:

* `trtllm_kv_cache_offload_bytes_total`
* `trtllm_kv_cache_onboard_bytes_total`

The counters cover all KVCM migrations, including lifecycles that use the
lossless fallback. In disaggregated serving, each context or generation worker
owns its own local KVCM and cold-tier quota, so put `kv_cache_config` and
`kv_cache_compression_config` on every worker that uses a local Host or Disk
tier, and check each worker.

## Disk Tier

Add a Disk tier behind the Host tier with a positive `disk_cache_size` and an
existing `disk_cache_path` directory. TensorRT-LLM writes the same TurboQuant
cold Pages to Disk through pinned Host staging.

```yaml
kv_cache_config:
  use_kv_cache_manager_v2: true
  host_cache_size: 8589934592    # 8 GiB
  disk_cache_size: 68719476736   # 64 GiB
  disk_cache_path: /path/to/kv-cache-dir

kv_cache_compression_config:
  algorithm: quantization_for_cold_page
  quant: turboquant
  turboquant_bits: 4
```

## Skipping RoPE Quantization

`skip_rope_quantization` has the same meaning and model support as for NVFP4
cold pages: by default all values are quantized, and with
`skip_rope_quantization: true` the position-encoded (RoPE) part of each K vector
is copied at its hot-cache precision while the rest is encoded with TurboQuant.
RoPE preservation is available for DeepSeek-V4, GLM-5 (`glm_moe_dsa`), and the
Qwen3.5 series; other models log a warning and quantize whole vectors, and
draft-model cold Pages always quantize whole vectors.

```yaml
kv_cache_compression_config:
  algorithm: quantization_for_cold_page
  quant: turboquant
  turboquant_bits: 4
  skip_rope_quantization: true
```

DeepSeek-V4 compressed (CSA) rows follow the option too: by default TurboQuant
encodes all 512 values of a row as one instance, and with
`skip_rope_quantization: true` it copies the 64 RoPE values and encodes the 448
NoPE values. There is no TurboQuant counterpart of the NVFP4 2FP4 RoPE format,
and `nvfp4_residual_dim` is rejected with `quant: turboquant`. DeepSeek-V4 requires
`tokens_per_block` of 128 or 256, and cold-page compression does not cover the
FP8 `fp8_ds_mla` footer-scale KV layout.

## Accuracy and Performance

TurboQuant is lossy, and its error depends on the bit width. Measure
the accuracy effect on your model and workload before deployment, and compare
it with NVFP4 cold pages at the same cold-tier quota.

### Reconstruction Quality

`tests/microbenchmarks/kv_cache_compression/turboquant_quality.py` measures the normalized MSE
`||x - x_hat||^2 / ||x||^2` of BF16 K/V vectors with head dimension 128 on structured inputs
("outlier channels" has two channels 10-25x larger than the rest, like real K):

| Bits | Paper (d = 128) | Gaussian | Outlier channels | Sparse | One-hot | All-equal | Norms 1e-6..1e6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.117 | 0.115 | 0.110 | 0.116 | 0.115 | 0.127 | 0.115 |
| 2.5 | between | 0.073 | 0.045 | 0.041 | 0.032 | 0.055 | 0.073 |
| 3 | 0.034 | 0.034 | 0.032 | 0.034 | 0.034 | 0.035 | 0.033 |
| 3.5 | between | 0.021 | 0.012 | 0.011 | 0.009 | 0.017 | 0.021 |
| 4 | 0.0094 | 0.0092 | 0.0086 | 0.0092 | 0.0092 | 0.0094 | 0.0091 |
| NVFP4 | | 0.0091 | 0.0084 | 0.0014 | 0 | 0 | 0.51 |

The per-vector least-squares scale makes TurboQuant slightly better than the paper's bound.

KVCacheManagerV2 re-encodes a Page each time it offloads the Page again after an onboard. The
same tool repeats the round trip on the outlier-channel input, each time re-encoding the decoded
Page:

| Bits | 1 round trip | 4 round trips | 16 round trips |
| --- | --- | --- | --- |
| 4 / 3 / 2 | 0.0086 / 0.0314 / 0.1092 | 0.0086 / 0.0314 / 0.1092 | 0.0086 / 0.0314 / 0.1092 |
| 3.5 / 2.5 | 0.0123 / 0.0448 | 0.0137 / 0.0497 | 0.0137 / 0.0497 |
| NVFP4 | 0.0084 | 0.0084 | 0.0084 |

The cold bytes settle after a few round trips and then repeat with period 1 or 2: most vectors
reproduce their codes, and a few alternate between two codes (on GB300 every setting tested
settled within 17 round trips; the unit tests check 32). At integer bit widths the error stays
where one round trip leaves it (within 1.5% with an FP8 KV cache, whose decoded values are rounded
again). Fractional bits also choose the outlier channels from the decoded data again, so channels
move between the two groups before a Page settles: the error grows by 10-20% on these 128-value
heads, and by more on MLA latents (below). The paper's TurboQuant-prod is not offered because its
error compounds instead; a prototype measured 0.55, 16.9, and 2.3e6 after 1, 4, and 16 round trips
at 2 bits, and 0.051, 0.31, and 7.0 at 4 bits.

With `--vectors FILE --rope 64`, the tool runs the same round trips on real vectors. On 6,144 MLA
latents of GLM-5.3 (BF16, 512 NoPE + 64 RoPE values; 48 layers x 128 tokens read from the KV cache
of an end-to-end run), the normalized MSE of the whole latent is:

| Bits | 1 / 4 / 16 round trips | 1 / 4 / 16 round trips with `skip_rope_quantization` |
| --- | --- | --- |
| 2 | 0.1148 / 0.1148 / 0.1148 | 0.0213 / 0.0213 / 0.0213 |
| 2.5 | 0.0427 / 0.0701 / 0.0736 | 0.0136 / 0.0148 / 0.0148 |
| 3 | 0.0328 / 0.0328 / 0.0328 | 0.0062 / 0.0062 / 0.0062 |
| 3.5 | 0.0120 / 0.0172 / 0.0178 | 0.0039 / 0.0045 / 0.0045 |
| 4 | 0.0089 / 0.0089 / 0.0089 | 0.0017 / 0.0017 / 0.0017 |

The 64 RoPE values carry most of a latent's energy (median norm 13.5, against 4.2 for the 512 NoPE
values), so keeping them lossless lowers the error of the whole latent. On whole latents the
error of fractional bit widths grows by 50-70% over repeated round trips before it settles; at the
default 3.5 bits it stays below the error of 3 bits, and 4 bits keeps the error of one round trip
where Pages migrate repeatedly.

### Encode and Decode Time

`tests/microbenchmarks/kv_cache_compression/bench_cold_page_codec.py` times one encode and one
decode call (256 Pages per call unless noted, cold Pages in pinned host memory, median of 20)
for TurboQuant and NVFP4 on the same GPU, and the speed of light (SOL) of the same transfer. On
one GB300, with the pinned memory on the GPU's own CPU socket (the GPU reads it at 211 GB/s and
writes it at 193 GB/s) and on the other socket (126 GB/s and 122 GB/s):

| Case | Geometry and TurboQuant setting | Cold Page KiB, TurboQuant / NVFP4 | Own socket: encode / decode us (vs NVFP4) | Other socket: encode / decode us (vs NVFP4) |
| --- | --- | --- | --- | --- |
| `glm5-bf16-mse4` | GLM-5 MLA latent, 78 x {K} x 1 x 32 x 576, BF16, 4 bits | 717 / 790 | 1078 / 1200 (0.98x / 1.20x) | 1591 / 1566 (0.92x / 0.94x) |
| `glm5-bf16-mse3` | same, 3 bits | 541 / 790 | 847 / 1248 (0.77x / 1.25x) | 1227 / 1541 (0.71x / 0.93x) |
| `glm5-bf16-mse3.5` | same, 3.5 bits (the default) | 630 / 790 | 3596 / 3122 (3.28x / 3.13x) | 3619 / 3611 (2.11x / 2.17x) |
| `glm5-skiprope-bf16-mse4` | same, 4 bits, `skip_rope_quantization` (512 values quantized) | 941 / 1014 | 1782 / 1403 (1.27x / 1.10x) | 2045 / 2037 (0.93x / 0.96x) |
| `mla-fp8-mse4` | DeepSeek-V3.2 / Kimi-K3 MLA latent, 61 x {K} x 1 x 32 x 576, FP8, 4 bits | 560 / 618 | 796 / 920 (0.94x / 1.18x) | 1240 / 1230 (0.93x / 0.94x) |
| `mla-fp8-mse3.5` | same, 3.5 bits | 493 / 618 | 3020 / 2506 (3.55x / 3.20x) | 3105 / 2850 (2.32x / 2.19x) |
| `dsv4-csa-bf16-mse4` | DeepSeek-V4 CSA, 18 x {compressed, indexer} x 1 x 32 x {512, 34}, BF16, 4 bits | 183 / 220 | 449 / 380 (0.63x / 1.19x) | 482 / 450 (0.47x / 0.83x) |
| `dsv4-csa-bf16-mse3.5` | same, 3.5 bits | 168 / 220 | 727 / 486 (0.99x / 1.53x) | 759 / 580 (0.73x / 1.07x) |
| `dsv4-csa-skiprope-bf16-mse4` | same, `skip_rope_quantization` (both formats keep RoPE) | 237 / 252 | 364 / 392 (1.03x / 1.20x) | 548 / 537 (1.00x / 0.99x) |
| `llama8b-bf16-mse4` | Llama-3.1-8B GQA, 32 x {K, V} x 8 x 32 x 128, BF16, 4 bits | 1056 / 1152 | 1458 / 1334 (0.92x / 0.92x) | 2292 / 2247 (0.92x / 0.93x) |
| `llama8b-bf16-mse3.5` | same, 3.5 bits | 968 / 1152 | 2914 / 1392 (1.83x / 0.96x) | 2914 / 2125 (1.17x / 0.88x) |
| `gqa-hd64-bf16-mse4` | GQA, 36 x {K, V} x 8 x 32 x 64, BF16, 4 bits | 612 / 648 | 849 / 784 (0.94x / 0.96x) | 1333 / 1322 (0.94x / 0.96x) |
| `glm5-bf16-mse4-8pages` | as `glm5-bf16-mse4`, 8 Pages per call | 717 / 790 | 64 / 84 (1.25x / 1.75x) | 74 / 102 (1.04x / 1.50x) |

Conventional K/V Pages are bound by the host link in both formats (96-100% of SOL), so
TurboQuant takes about the NVFP4 time scaled by the cold-page size. MLA latents and DeepSeek-V4
CSA rows need more decode work per byte, because every instance (256 to 512 values) is rotated
back on tensor cores: on the faster link their decode takes 1.1-1.25x the NVFP4 time, while
encode stays at or below it; on the slower link both are link bound and TurboQuant is faster.
With `skip_rope_quantization` the GLM-5 latent is a single 512-value instance, whose encode takes
1.27x the NVFP4 time; its decode accumulates the output in two halves from codes read once.
DeepSeek-V4 CSA rows are one 512-value instance by default; NVFP4's default format quantizes
their RoPE as two FP4 components, which makes its encode slow, so `dsv4-csa-skiprope-bf16-mse4`
compares like with like. Small calls are bound by fixed costs, such as the one decode launch per
instance of a 576-value latent.

Fractional bit widths, including the default 3.5 bits, take longer: every KV head of a Page first
votes on its channel split and is then coded on its own, 32 rows at a time, whereas integer bit
widths code larger tiles across heads and Pages. On the faster link MLA latents take 3.3-3.6x the
NVFP4 time to encode and about 3.2x to decode, Llama-3.1-8B K/V Pages 1.8x to encode and about the
NVFP4 time to decode, and DeepSeek-V4 CSA rows about the NVFP4 time to encode and 1.5x to decode;
2.5 bits takes about as long. `--bits 2,2.5,3,3.5,4` runs any case at every bit width. Where the
migration time matters more than a 9-12% smaller cold Page, 4 bits is faster and more accurate.
The launch shapes of rows wider than 256 values were measured on GB300 (SM103) and apply to
SM100-family GPUs; other GPUs use generic shapes. Run the benchmark on your system before relying
on these numbers.

### End-to-End Smoke Test

`examples/kv_cache_compression/cold_tier_accuracy.py` checks that a model still answers through
decoded cold Pages; it is a smoke test, not accuracy evidence. It prefills synthetic prefixes of
name-and-code facts, pushes their Pages to the Host tier with unrelated prompts, and then asks
easy exact-match questions whose prefixes must be onboarded and decoded again. A run counts only
if the iteration statistics of its question phase show Attention Pages reused from the cold tier
(Disk with `--disk`) and bytes onboarded; they do not show which questions those Pages served,
and each Page goes to the cold tier and back once. With TP4/EP4 on four GB300 GPUs, greedy
decoding, two 6144-token prefixes, and 400 questions:

| Model | none | NVFP4 | TurboQuant 4 | TurboQuant 3.5 | TurboQuant 3 | TurboQuant 2.5 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-Flash-Next-FP8 | 400/400 | 400/400 | 400/400 | 400/400 | 400/400 | 395/400 |
| GLM-5.3-Flash | not run | not run | 400/400 | not run | 397/400 | not run |

## Enablement Checklist

1. Run the PyTorch backend on a GPU with compute capability 8.0 or newer, or
   8.9 or newer with an FP8 active KV cache.
2. Select the native C++ KVCacheManagerV2 with
   `use_kv_cache_manager_v2: true`.
3. Keep `kv_cache_config.dtype` at the model's intended runtime KV type:
   `auto`, FP16/BF16, or FP8. An active NVFP4 KV cache is rejected.
4. Provision a nonzero Host cache and, optionally, a Disk cache. A positive
   `disk_cache_size` also requires `disk_cache_path` to name an existing
   directory.
5. Set `kv_cache_compression_config.algorithm` to
   `quantization_for_cold_page`, `quant` to `turboquant`, and choose
   `turboquant_bits`.
6. Apply both cache configurations to every worker that owns a local cold tier.
7. Verify nonzero offload and onboard bytes, and a cold slot size smaller than
   the hot Page, under a workload that creates real KV pressure.

`KvCacheConfig` and KVCacheManagerV2 own capacity, Page/Slot allocation,
cache-level routing, migration batching, completion events, and mapping
publication. `ColdPageQuantizationCompressionConfig` and its codec own only the
cold representation and its encode/decode policy.

For the underlying KVCM lifecycle and codec ABI, see
[KVCacheManagerV2 Cold-Page Codec Design](https://nvidia.github.io/TensorRT-LLM/developer-guide/kv-cache-cold-page-codec.html).
