<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GLM-5.3-Flash FP8 on 4x B200: deployment

This target (`glm_5_3_flash_fp8__sm_10x__tp4`) serves the GLM-5.3-Flash FP8 checkpoint on one node of four
B200 (sm_100) GPUs with tensor parallelism 4 and MTP-3 speculative decoding. Routing does not select it on
sm_103 (B300 / GB300), where it has not run. Compared with a standard
GLM-5.3-Flash deployment, it needs `modeling_v2: require` in the server config and a larger `--max_num_tokens`; its
CUDA kernels are in the glm5_next family library the build compiles.

## 1. Build

Build or install TensorRT LLM as usual, for a CUDA architecture list that includes `100` and with multi-device
support (the default): the build then compiles the target's kernels into `libth_modeling_v2_glm5_next.so`.

## 2. Hardware and checkpoint

- One node, 4x B200, NVLink.
- The GLM-5.3-Flash FP8 checkpoint (128 x 128 block-FP8 experts and MLA projections).

## 3. Server config

`config.yaml`:

```yaml
modeling_v2: require          # select this ModelingV2 target; startup fails if the setup does not match it
speculative_config:
  decoding_type: MTP
  max_draft_len: 3            # one-model MTP-3 on the checkpoint's MTP layer
kv_cache_config:
  dtype: fp8
  free_gpu_memory_fraction: 0.5
  enable_block_reuse: false
cuda_graph_config:
  enable_padding: true
  max_batch_size: 64
enable_chunked_prefill: true
enable_attention_dp: false
disable_mm_encoder: true
moe_config:
  backend: AUTO
```

`free_gpu_memory_fraction` can be raised for more KV-cache capacity.

## 4. Launch

```bash
trtllm-serve /path/to/GLM-5.3-Flash \
  --backend pytorch --tp_size 4 --ep_size 1 \
  --max_batch_size 64 --max_num_tokens 32768 --max_seq_len 131072 \
  --trust_remote_code --config config.yaml
```

What differs from a standard deployment:

- `--max_num_tokens 32768`: long prompts prefill in larger chunks. The measurements below use it; the default
  token budget has not been measured on this target.
- `--max_seq_len 131072` is the model's context length. Prompts of up to 13,312 tokens have been measured on this
  target (the operations' certified cells stop there); longer ones are untested.
- Without the glm5_next family library (a build whose CUDA architectures include no `100`, or without multi-device
  support) the target is not selected, and `modeling_v2: require` fails at startup.
- Keep the all-reduce strategy at its default (AUTO, or ONESHOT): the target runs the one-shot all-reduce, whose
  workspace its fused kernels share, and routing declines other strategies.

## 5. Check

- The server log shows `modeling_v2: ModelingV2Glm53FlashFp8Sm10xTp4 built`.

## 6. Measured

One node of 4x B200, the config and launch above, against the built-in GLM-5.3-Flash model (`modeling_v2: off`)
with the same settings on the same node. Throughput: 13,312-token prompts, 33 output tokens (ignore_eos,
temperature 0), the median of two runs of 32 requests at C1 and 64 or 128 at C8 / C32 / C64.

| | built-in | this target |
| --- | --- | --- |
| C1 TTFT p50 | 271 ms | 205 ms |
| C1 output tok/s/user p50 | 397 | 492 |
| C8 output tok/s | 126 | 175 |
| C32 output tok/s | 135 | 193 |
| C64 output tok/s | 135 | 198 |
| KV-cache memory per rank (`free_gpu_memory_fraction` 0.5) | 45.2 GiB | 44.0 GiB |
| GSM8K, accuracy harness, no speculation | 93.2% | 92.5% |
| GSM8K, accuracy harness, MTP-3 | 92.3% | 93.1% |
| MTP-3 acceptance length (accuracy harness) | 3.382 | 3.380 |
| GSM8K, 400 questions through the server, MTP-3 | 98.5% | 99.0% |

The accuracy-harness rows are `test_accuracy.py` (5-shot, no chat template, the CI protocol); the last row asks
the server above zero-shot through its OpenAI chat endpoint (greedy, chat template), a different protocol.
