## AutoDeploy Custom Operators

All AutoDeploy custom operators follow the following naming convention:

`torch.ops.auto_deploy.<kernel_backend>_<op_category>_<op_name>`

The table below lists the operators grouped by category.

### Available Custom Operators

#### Attention

| Operator Name | Description |
|--------------|-------------|
| `torch.ops.auto_deploy.torch_attention` | Grouped SDPA implementation with `bsnd` and `bnsd` layout supported |
| `torch.ops.auto_deploy.torch_attention_sdpa` | Standard scaled dot-product attention (SDPA) implementation |
| `torch.ops.auto_deploy.torch_attention_repeat_kv` | KV repetition for grouped-query attention |
| `torch.ops.auto_deploy.torch_cached_attention_with_cache` | PyTorch backend attention with KV cache management |
| `torch.ops.auto_deploy.flashinfer_attention_mha_with_cache` | FlashInfer multi-head attention with KV cache support |
| `torch.ops.auto_deploy.flashinfer_attention_prepare_metadata` | FlashInfer attention metadata preparation |
| `torch.ops.auto_deploy.triton_attention_flattened_mha_with_cache` | Triton flattened MHA with cache |
| `torch.ops.auto_deploy.torch_onnx_attention_plugin` | Fused attention with RoPE placeholder for ONNX export |
| `torch.ops.auto_deploy.torch_onnx_gather_nd` | N-dimensional gather operation for ONNX export |

#### MLA (Multi-head Latent Attention)

| Operator Name | Description |
|--------------|-------------|
| `torch.ops.auto_deploy.torch_mla` | Multi-head Latent Attention (MLA) implementation |
| `torch.ops.auto_deploy.torch_cached_mla_with_cache` | PyTorch backend cached MLA with KV cache |
| `torch.ops.auto_deploy.flashinfer_mla_with_cache` | FlashInfer MLA with cache |
| `torch.ops.auto_deploy.flashinfer_mla_prepare_metadata` | FlashInfer MLA metadata preparation |

#### RoPE (Rotary Position Embedding)

| Operator Name | Description |
|--------------|-------------|
| `torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin` | RoPE with explicit cosine/sine |
| `torch.ops.auto_deploy.torch_rope_with_complex_freqs` | RoPE with complex frequencies |
| `torch.ops.auto_deploy.torch_rope_with_qk_interleaving` | RoPE with QK interleaving |
| `torch.ops.auto_deploy.triton_rope_with_input_pos` | Triton RoPE with input positions |
| `torch.ops.auto_deploy.triton_rope_with_qk_interleaving` | Triton RoPE with QK interleaving (DeepSeek-style) |
| `torch.ops.auto_deploy.trtllm_moe_fused` | TensorRT-LLM fused MoE implementation |
| `torch.ops.auto_deploy.trtllm_dist_fused_linear_all_reduce` | TensorRT-LLM fused linear layer followed by all-reduce operation |
