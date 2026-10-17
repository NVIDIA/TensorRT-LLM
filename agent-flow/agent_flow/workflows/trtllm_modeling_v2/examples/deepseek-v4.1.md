<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Worked example: DeepSeek-V4.1-Flash NVFP4

What one checkpoint required of the workflow's generic features, kept here as
a worked example. The skills describe each feature without naming a model;
this file shows one model's values filled in.

**This file is for operators writing a `task.yaml`.** The run does not
materialize `examples/` into its workspace, so no role is handed it. A run
that onboards this checkpoint, or a sibling of it, to measure the workflow
should embargo it (`source_embargo: [package:examples/deepseek-v4.1.md]`),
because the facts below are what the run is meant to derive.

Sources: the DeepSeek-V4.1-Flash tech report, the checkpoint's own reference
code, and SGLang's implementation (branch `origin/dsv4.1`).

## `dspark` (trtllm-modeling-v2-speculator)

- The tech report (§2.4.3): "We omit the MTP module during backbone
  pre-training and use DSpark for speculative decoding." The drafter ships
  under `mtp.0..2` with `num_nextn_predict_layers: 3`, which is what makes
  `mtp` the easy wrong choice. The `dspark_*` keys sit under `text_config`.
- Block size 5, `dspark_noise_token_id` 128799. Each of the three stages is a
  full block: sliding-window attention plus a 128-expert top-3 MoE, with
  hyper-connections.
- Each stage's ring has window 128 and is filled from the *input* of target
  layers 37, 38 and 39 (`dspark_target_layer_ids`), averaged over the
  hyper-connection copies. vLLM's DeepSeek-V4 path captures layer outputs,
  which is a different checkpoint's contract.
- The Markov head is a rank-256 bigram bias.
- Acceptance ceiling 6 (`block_size + 1`). vLLM's DeepSeek-V4 DSpark recipe
  uses 1.2 as its smoke floor.

```yaml
features: [dspark]
accuracy:
  llm_api_options:
    dspark: scripts/dspark_llm_api.yaml   # decoding_type: DSpark, max_draft_len: 5
  acceptance:
    dspark:
      min_absolute: 1.2
```

## `kv_quant` (trtllm-modeling-v2-kv-quant)

The reference fake-quantizes KV to these formats (tech report §2.4.4):

| pool | who owns one | format | bytes per entry |
|---|---|---|---|
| SWA KV | every layer (40), the window only | FP8 E4M3 per 32 channels with a UE8M0 scale, over all 512 dims including RoPE | 512 + 16 = **528** |
| main (compressed global) KV | the KV-source layers only: 2, 8, 14 (compress ratio 2) and 20 (ratio 1) | FP4 E2M1 with an E4M3 scale per 16, **no global scale** | 256 + 32 = **288** |
| indexer K | the same KV-source layers | MXFP4: E2M1 with a UE8M0 scale per 32, 128 dims | 64 + 4 = **68** |

Only layers 2, 8, 14 and 20 own compressed and indexer pools; the others read
a source layer's pool through `kv_source_layer_ids`. A pool allocated per
compressed layer (38 of them) comes to 10324 B/token, which is the sign the
sharing was never wired.

```
3 x (288 + 68) / 2  +  (288 + 68)  =  534 + 356  =  890 B/token
   ratio-2 layers       ratio-1 layer
```

The same formula gives 3200 B/token for bf16 containers of these values (the
core unit) and 1490 B/token with FP8 main KV. Padding the 4-byte indexer scale
to 16 moves 890 to 920. The SWA window is per request: 40 x 128 x 528 B,
about 2.7 MB.

How this differs from the DeepSeek-V4 layouts in
`tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/`: V4's
`fp8_ds_mla` is 448 FP8 + 64 bf16 RoPE dims with block-64 scales, where V4.1
quantizes all 512 dims, RoPE included, after RoPE. V4's `nvfp4` pool has a
second-level global scale and a 64-dim residual, where V4.1 has no global
scale and all 512 dims in FP4.

The sm_103 kernels that serve this model (trtllm-gen's H512
DynamicTokenSparse path through FlashInfer's
`trtllm_batch_decode_sparse_mla_dsv4`, and the catalog's `thop_attention`) are
exact only over bf16 KV, which is why core stores bf16 containers. SGLang
reports that this model's output is not bitwise stable across batch
composition, so compare at batch 1.

```yaml
features: [kv_quant]
accuracy:
  witness:
    kv_quant:
      contrast_env: {TRTLLM_DSV41_KV_STORAGE: bf16}   # the leg WITH bf16 storage
      expected_bytes_per_token: 890
      exact: true
      max_batch_size: 1
```

## `decoder_replay` / `encoder_replay` (trtllm-modeling-v2-bounded-replay)

The model is a causal encoder-decoder: layers 0-19 are the "encoder", layers
20-39 the "decoder", and decoder global KV is projected from the encoder's
final hidden state. Sliding window 128. The last entry of
`kv_source_layer_ids` is 20; the tech report starts the decoder there, while
SGLang computes `late_layer_start = max(kv_source_layer_ids) + 1` and runs
layer 20 over every prompt token, because its compressor produces the
decoder's global KV.

The tech report (§3.2.2, "SWA Bounded Replay"):

> "SWA Bounded Replay instead replays only the most recent n_win tokens and
> truncates SWA to the replay segment, accepting approximate states: for a
> replay starting at position r, a query at position i attends to SWA keys in
> [max(r, i−n_win+1), i]."

> "**Encoder SWA Bounded Replay** makes prefix caching depend only on global
> KV, allowing SWA KV to be removed from the persistent KV cache. When the
> encoder SWA KV is missing, we replay the last n_win tokens of the cached
> prefix and process them together with the uncached suffix. The replayed
> tokens regenerate only SWA KV, reusing the cached global KV without
> recomputation or overwriting, while the uncached suffix generates both
> global KV and SWA KV. By design, the replayed prefix state is approximate…"

> "**Decoder SWA Bounded Replay** bounds the decoder forward pass to n_win
> tokens, nearly halving total prefill computation. Under CED, decoder global
> KV is projected from the final encoder hidden states. The only obstacle to
> ending prefill at the encoder is decoder SWA KV … at every prefill, we
> replay the last n_win tokens of the prompt, feed their encoder outputs
> through the decoder layers under the same SWA truncation, and use the
> resulting decoder SWA KV only for decoding, not for prefix caching. … not
> mathematically equivalent … we additionally simulate the same replay during
> post-training for train-aware adaptation."

SGLang's reference mechanics are in
`python/sglang/srt/model_executor/encoder_swa_replay.py`. Its argument checks
(`python/sglang/srt/arg_groups/deepseek_v4_hook.py`) refuse `encoder_replay`
with prefill CUDA graphs, attention DP, PD disaggregation and LoRA, and
`decoder_replay` with prefill CUDA graphs and attention DP. Compress ratio 2
on layers 2, 8 and 14 makes the reuse boundary even.

```yaml
features: [decoder_replay, encoder_replay]
accuracy:
  witness:
    decoder_replay:
      contrast_env: {TRTLLM_DSV41_DECODER_SWA_BOUNDED_REPLAY: "0"}
      window: 128
      prompt_tokens: 4096
    encoder_replay:
      contrast_env: {TRTLLM_DSV41_ENCODER_SWA_BOUNDED_REPLAY: "0"}
      window: 128
      prefix_tokens: 4096
      reuse_alignment: 2
```

## `host_offload` (trtllm-modeling-v2-host-offload)

The Engram table: two modules of 384M rows x 256 FP8 plus E8M0 scales,
**202.76 GB**, larger than the rest of the checkpoint. Per-rank copies on a
four-GPU node would be 811 GB of host memory; the shared single-copy layout
holds it once. SGLang ships that layout behind
`SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=shared` (`python/sglang/srt/layers/engram.py`).
Row-sharded at TP4, the residency witness's `expected_mib` is
202.76 GB / 4 = **48,340 MiB** per rank.

## `disagg`

Both bounded replays are refused under disaggregation, so the disagg
equivalence witness pins them off on both legs:

```yaml
accuracy:
  witness:
    disagg:
      contrast_env: {TRTLLM_DISAGG: "1"}
      contrast_command: scripts/disagg_leg_b.py
      max_batch_size: 1
      legs_env:
        TRTLLM_DSV41_ENCODER_SWA_BOUNDED_REPLAY: "0"
        TRTLLM_DSV41_DECODER_SWA_BOUNDED_REPLAY: "0"
```

## The smith queue

The run queued 33 dispatches, six of them needed only by a feature unit: the
Engram host-gather ports, the ViT attention verify, a DSpark-only routing
surface and the MXFP4 drafter MoE. Untagged, any one of them failing would
have closed the run before core started, which is why queue rows carry
`unit:`.
