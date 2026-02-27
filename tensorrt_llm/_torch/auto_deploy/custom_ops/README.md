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
| `torch.ops.auto_deploy.triton_rope_on_flattened_inputs` | Triton RoPE on flattened inputs |
| `torch.ops.auto_deploy.triton_rope_on_interleaved_qk_inputs` | Triton fused RoPE on interleaved QK inputs (position lookup + de-interleave + RoPE) |
| `torch.ops.auto_deploy.flashinfer_rope` | FlashInfer RoPE implementation |

#### Linear

| Operator Name | Description |
|--------------|-------------|
| `torch.ops.auto_deploy.torch_linear_simple` | Simple linear layer wrapper (avoids view ops in export graph) |
| `torch.ops.auto_deploy.torch_moe_router` | MoE router: linear projection + top-k + softmax + scatter |

#### MoE (Mixture of Experts)

| Operator Name | Description |
|--------------|-------------|
| `torch.ops.auto_deploy.torch_moe` | Mixture of Experts implementation (PyTorch backend) |
| `torch.ops.auto_deploy.torch_moe_fused` | Fused Mixture of Experts implementation (PyTorch backend) |
| `torch.ops.auto_deploy.torch_moe_dense_mlp` | Dense MLP implementation for MoE (PyTorch backend) |
| `torch.ops.auto_deploy.torch_quant_fp8_moe` | FP8 quantized MoE (PyTorch backend) |
| `torch.ops.auto_deploy.torch_quant_nvfp4_moe` | NVFP4 quantized MoE (PyTorch backend) |
| `torch.ops.auto_deploy.triton_moe_fused` | Fused MoE (Triton backend) |
| `torch.ops.auto_deploy.triton_quant_fp8_moe` | FP8 quantized MoE (Triton backend) |
| `torch.ops.auto_deploy.triton_mxfp4_moe` | MXFP4 MoE with triton-kernels matmul_ogs |
| `torch.ops.auto_deploy.triton_mxfp4_moe_ep` | MXFP4 MoE with Expert Parallelism (triton-kernels) |
| `torch.ops.auto_deploy.trtllm_moe_fused` | Fused MoE (TRT-LLM backend) |
| `torch.ops.auto_deploy.trtllm_quant_fp8_moe_fused` | FP8 quantized fused MoE (TRT-LLM backend) |
| `torch.ops.auto_deploy.trtllm_quant_nvfp4_moe_fused` | NVFP4 quantized fused MoE (TRT-LLM backend) |

#### Quantization

| Operator Name | Description |
|--------------|-------------|
| `torch.ops.auto_deploy.torch_quant_fn` | Generic quantization function that scales, rounds, and clamps input values |
| `torch.ops.auto_deploy.torch_quant_fp8_linear` | FP8 quantized linear layer (PyTorch backend) |
| `torch.ops.auto_deploy.torch_quant_nvfp4_linear` | NVFP4 quantized linear layer (PyTorch backend) |
| `torch.ops.auto_deploy.torch_quant_fp8_bmm` | FP8 quantized batch matrix multiply (PyTorch backend) |
| `torch.ops.auto_deploy.trtllm_quant_fp8_linear` | FP8 quantized linear layer (TRT-LLM backend) |
| `torch.ops.auto_deploy.torch_fake_quant_fp8_linear` | Fake FP8 quantized linear (for calibration/simulation) |
| `torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear` | Fake NVFP4 quantized linear (for calibration/simulation) |
| `torch.ops.auto_deploy.torch_fake_quant_int4_linear` | Fake INT4 quantized linear (for calibration/simulation) |
| `torch.ops.auto_deploy.torch_fake_quant_int4_gptq_linear` | Fake INT4 GPTQ quantized linear (for calibration/simulation) |

#### Normalization

| Operator Name | Description |
|--------------|-------------|
| `torch.ops.auto_deploy.torch_rmsnorm` | RMSNorm (PyTorch backend) |
| `torch.ops.auto_deploy.torch_rmsnorm_gated` | Gated RMSNorm with optional SiLU gating (PyTorch backend) |
| `torch.ops.auto_deploy.triton_rms_norm` | RMSNorm (Triton backend) |
| `torch.ops.auto_deploy.triton_rmsnorm_gated` | Gated RMSNorm with optional SiLU gating (Triton backend) |
| `torch.ops.auto_deploy.flashinfer_rms_norm` | RMSNorm (FlashInfer backend) |
| `torch.ops.auto_deploy.flashinfer_fused_add_rms_norm_inplace` | Fused residual add + RMSNorm in-place (FlashInfer backend) |
| `torch.ops.auto_deploy.sharded_rmsnorm` | RMSNorm for tensor-parallel sharded activations (uses all-reduce) |
| `torch.ops.auto_deploy.torch_l2norm` | L2 normalization (PyTorch backend) |
| `torch.ops.auto_deploy.fla_l2norm` | L2 normalization (FLA Triton kernel backend) |

#### Mamba (SSM + Causal Conv)

| Operator Name | Description |
|--------------|-------------|
| `torch.ops.auto_deploy.torch_ssm` | State Space Model (SSM) computation (PyTorch backend) |
| `torch.ops.auto_deploy.torch_cached_ssm` | Cached SSM with state management (PyTorch backend) |
| `torch.ops.auto_deploy.triton_cached_ssm` | Cached SSM with state management (Triton backend) |
| `torch.ops.auto_deploy.flashinfer_cached_ssm` | Cached SSM with state management (FlashInfer backend) |
| `torch.ops.auto_deploy.mamba_ssm_prepare_metadata` | Mamba SSM metadata preparation (chunk indices, offsets, seq_idx) |
| `torch.ops.auto_deploy.torch_causal_conv1d` | Causal 1D convolution (PyTorch backend) |
| `torch.ops.auto_deploy.torch_cached_causal_conv1d` | Cached causal 1D convolution (PyTorch backend) |
| `torch.ops.auto_deploy.triton_cached_causal_conv1d` | Cached causal 1D convolution (Triton backend) |
| `torch.ops.auto_deploy.cuda_cached_causal_conv1d` | Cached causal 1D convolution (CUDA backend) |

#### FLA (Flash Linear Attention)

| Operator Name | Description |
|--------------|-------------|
| `torch.ops.auto_deploy.fla_delta_rule` | FLA chunked delta rule computation |
| `torch.ops.auto_deploy.fla_cached_delta_rule` | FLA cached delta rule with state management |

#### Distributed

| Operator Name | Description |
|--------------|-------------|
| `torch.ops.auto_deploy.torch_dist_all_gather` | All-gather (PyTorch backend, demollm mode) |
| `torch.ops.auto_deploy.torch_dist_all_reduce` | All-reduce (PyTorch backend, demollm mode) |
| `torch.ops.auto_deploy.trtllm_dist_all_gather` | All-gather (TRT-LLM backend, MPI mode) |
| `torch.ops.auto_deploy.trtllm_dist_all_reduce` | All-reduce (TRT-LLM backend, MPI mode) |
| `torch.ops.dist.trtllm_fused_allreduce_residual_rmsnorm` | Fused all-reduce + residual add + RMSNorm (TRT-LLM backend, MPI mode) |

#### Utilities

| Operator Name | Description |
|--------------|-------------|
| `torch.ops.auto_deploy.triton_utils_fused_gather_scatter` | Triton fused gather + scatter for overlap scheduling input_ids reordering |
| `torch.ops.auto_deploy.gather_logits_before_lm_head` | Gather hidden states using logits indices before LM head |
