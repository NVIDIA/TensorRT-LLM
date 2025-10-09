## AutoDeploy Custom Operators

All AutoDeploy custom operators follow the following naming convention:

`torch.ops.auto_deploy.<kernel_backend>_<op_category>_<op_name>`

The table below lists the operators ordered by their backend.

### Available Custom Operators

| Operator Name | Description |
|--------------|-------------|
| `torch.ops.auto_deploy.flashinfer_attention_mha_with_cache` | FlashInfer attention with KV cache support |
| `torch.ops.auto_deploy.flashinfer_rope` | FlashInfer RoPE (Rotary Position Embedding) implementation |
| `torch.ops.auto_deploy.torch_attention_deepseek_fused_mla` | DeepSeek fused MLA (Multi-head Linear Attention) |
| `torch.ops.auto_deploy.torch_attention_deepseek_mla` | DeepSeek MLA implementation |
| `torch.ops.auto_deploy.torch_attention` | Grouped SDPA implementation with `bsnd` and `bnsd` layout supported |
| `torch.ops.auto_deploy.torch_attention_repeat_kv` | KV repetition for attention |
| `torch.ops.auto_deploy.torch_attention_sdpa` | Standard SDPA implementation |
| `torch.ops.auto_deploy.torch_dist_all_gather` | Distributed all-gather operation |
| `torch.ops.auto_deploy.torch_dist_all_reduce` | Distributed all-reduce operation |
| `torch.ops.auto_deploy.torch_linear_simple` | Simple linear layer implementation |
| `torch.ops.auto_deploy.torch_moe` | Mixture of Experts implementation |
| `torch.ops.auto_deploy.torch_moe_fused` | Fused Mixture of Experts implementation |
| `torch.ops.auto_deploy.torch_quant_fn` | Generic quantization function that scales, rounds, and clamps input values |
| `torch.ops.auto_deploy.torch_quant_fused_fp8_linear_all_reduce` | Fused FP8 linear layer followed by all-reduce operation |
| `torch.ops.auto_deploy.torch_quant_nvfp4_linear` | FP4 quantized linear layer |
| `torch.ops.auto_deploy.torch_quant_fp8_linear` | FP8 quantized linear layer |
| `torch.ops.auto_deploy.torch_rope_with_complex_freqs` | RoPE with complex frequencies |
| `torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin` | RoPE with explicit cosine/sine |
| `torch.ops.auto_deploy.torch_rope_with_qk_interleaving` | RoPE with QK interleaving |
| `torch.ops.auto_deploy.triton_attention_fused_flattened_mha_with_cache` | Triton fused flattened MHA with cache |
| `torch.ops.auto_deploy.triton_attention_fused_flattened_mha_with_cache_rope_fusion` | Triton fused flattened MHA with cache and RoPE fusion |
| `torch.ops.auto_deploy.triton_attention_fused_mha_with_cache` | Triton fused MHA with cache |
| `torch.ops.auto_deploy.triton_attention_fused_mha_with_paged_cache` | Triton fused MHA with paged cache |
| `torch.ops.auto_deploy.triton_attention_flattened_mha_with_cache` | Triton flattened MHA with cache |
| `torch.ops.auto_deploy.triton_attention_fused_flattened_mla_with_cache` | Triton fused flattened Multi-head Latent Attention with cache support |
| `torch.ops.auto_deploy.triton_rope_on_flattened_inputs` | Triton RoPE on flattened inputs |
| `torch.ops.auto_deploy.triton_rope_with_input_pos` | Triton RoPE with input positions |
| `torch.ops.auto_deploy.trtllm_moe_fused` | TensorRT LLM fused MoE implementation |
| `torch.ops.auto_deploy.trtllm_dist_fused_linear_all_reduce` | TensorRT LLM fused linear layer followed by all-reduce operation |
