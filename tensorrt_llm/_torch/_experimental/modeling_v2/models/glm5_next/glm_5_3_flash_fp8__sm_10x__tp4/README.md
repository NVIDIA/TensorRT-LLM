# GLM-5.3-Flash FP8 on 4x B200: deployment

This target (`glm_5_3_flash_fp8__sm_10x__tp4`) serves the GLM-5.3-Flash FP8 checkpoint on one node of four
Blackwell (sm_100 / sm_103) GPUs with tensor parallelism 4 and MTP-3 speculative decoding. Compared with a standard
GLM-5.3-Flash deployment, it needs `modeling_v2: require` in the server config and a larger `--max_num_tokens`; its
CUDA kernels are in the glm5_next family library the build compiles.

## 1. Build

Build or install TensorRT LLM as usual, for a CUDA architecture list that includes an SM 10.x one (`100` for
B200 / GB200, `103` for B300 / GB300): the build then compiles the target's kernels into
`libth_modeling_v2_glm5_next.so`.

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

- `--max_num_tokens 32768`: long prompts prefill in large chunks (about +5% output throughput at 8 concurrent
  requests with 13K-token prompts).
- Without the glm5_next family library (a build whose CUDA architectures include no SM 10.x one) the target is not
  selected, and `modeling_v2: require` fails at startup.
- Keep the default all-reduce strategy: the target selects the one-shot all-reduce, whose workspace its fused
  kernels share.

## 5. Check

- The server log shows `modeling_v2: ModelingV2Glm53FlashFp8Sm10xTp4 built`.
- GSM8K (400 questions): 98-99%.
