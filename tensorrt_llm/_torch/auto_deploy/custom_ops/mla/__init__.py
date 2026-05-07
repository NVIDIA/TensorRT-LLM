"""MLA (Multi-head Latent Attention) custom ops.

This module provides various MLA implementations and backends:
- torch_mla: PyTorch reference MLA implementation
- torch_backend_mla: PyTorch-based MLA backend
- flashinfer_mla: FlashInfer-based optimized MLA
- flashinfer_trtllm_mla: FlashInfer TRTLLM-gen MLA (Blackwell Path 2)
- triton_mla: Triton-based MLA implementation
- trtllm_mla: TRT-LLM thop.attention-based MLA (requires tensorrt_llm)
"""
