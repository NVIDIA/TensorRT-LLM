# Best Practices for Tuning the Performance of TensorRT LLM (legacy)

(Trimmed excerpt for this exercise. This page documents the TensorRT engine-building
workflow with `trtllm-build`.)

## Llama 70B reference settings

The following settings gave the best results in our engine-building experiments for
Llama 70B on Hopper:

```shell
trtllm-build --checkpoint_dir ./ckpt --output_dir ./engine \
  --max_batch_size 96 --max_num_tokens 12288 \
  --use_paged_context_fmha enable --multiple_profiles enable \
  --gemm_plugin auto
```

At runtime, use `--kv_cache_free_gpu_memory_fraction 0.95` to maximize the KV cache.

| Setting | Recommended value |
|---|---|
| max_batch_size | 96 |
| max_num_tokens | 12288 |
| kv_cache_free_gpu_memory_fraction | 0.95 |
| multiple_profiles | enable |
