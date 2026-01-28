# FlashInfer TRT-LLM Attention Backend

A simplified TRT-LLM attention backend for AutoDeploy that uses FlashInfer's TRT-LLM interface instead of directly calling `thop.attention`.

## Overview

This implementation provides TRT-LLM attention through FlashInfer's `trtllm_batch_decode_with_kv_cache` API, which offers significant advantages over the direct `thop.attention` approach.

## Architecture Comparison

### Previous Approach (Direct `thop.attention`)
```
AD Metadata → Python Translation → Host Tensors → thop.attention → TRT-LLM Kernels
             (1200 lines)          (complex)      (many params)
```

**Issues:**
- ~1200 lines of complex metadata translation code
- Required PT cache backend for CUDA graph support
- Host/device tensor management complexity
- Manual layout conversion between AD and TRT-LLM formats
- Needed per-layer state management
- Required host translation metadata in each iteration

### New Approach (FlashInfer Interface)
```
AD Metadata → Simple Conversion → flashinfer.decode.trtllm_batch_decode_with_kv_cache → TRT-LLM Kernels
             (~300 lines)         (clean interface)
```

**Benefits:**
- **Much simpler:** ~300 lines vs ~1200 lines (75% reduction)
- **No PT cache backend needed:** FlashInfer handles metadata internally
- **Built-in CUDA graph support:** Works out of the box
- **Unified interface:** Same API pattern as FlashInfer backend
- **Easier maintenance:** Less custom code to maintain
- **Better tested:** FlashInfer's TRT-LLM interface is battle-tested

## Implementation Details

### File Structure

```
tensorrt_llm/_torch/auto_deploy/custom_ops/
├── flashinfer_trtllm_attention.py  # New implementation (~300 lines)
├── trtllm_attention.py             # Old implementation (~1200 lines)
└── pt_cache_backend.py             # Only needed by old implementation
```

### Key Components

1. **Metadata Preparation** (`prepare_flashinfer_trtllm_metadata`)
   - Converts AD's metadata to FlashInfer format
   - Creates block_tables and seq_lens tensors
   - CUDA graph compatible with cached tensors

2. **Attention Kernel** (`flashinfer_trtllm_mha_with_cache`)
   - Handles KV cache append
   - Calls FlashInfer's TRT-LLM backend
   - Supports both prefill and decode

3. **Backend Descriptor** (`FlashInferTrtllmAttention`)
   - Registers as "flashinfer_trtllm" backend
   - Configures cache layout and workspace
   - Provides constants and initializers

## Usage

### Configuration

```yaml
# config.yaml
model: nvidia/Llama-3.1-8B-Instruct-FP8
attn_backend: flashinfer_trtllm  # Use FlashInfer TRT-LLM backend
compile_backend: torch-cudagraph
cuda_graph_batch_sizes: [1, 2, 4, 8, 16, 32]
max_batch_size: 32
max_seq_len: 2048
```

### Python API

```python
from tensorrt_llm._torch.auto_deploy import AutoDeployConfig, build_model

config = AutoDeployConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
    attn_backend="flashinfer_trtllm",  # Use new backend
    compile_backend="torch-cudagraph",
    cuda_graph_batch_sizes=[1, 2, 4, 8, 16, 32],
    max_batch_size=32,
    max_seq_len=2048,
)

model = build_model(config)
```

### Testing

Run the test suite:

```bash
python test_flashinfer_trtllm.py
```

Expected output:
```
============================================================
FlashInfer TRT-LLM Attention Backend Tests
============================================================
Testing backend registration...
✓ Backend registration test passed!

Testing metadata preparation...
  block_tables shape: torch.Size([4, 3])
  seq_lens_out shape: torch.Size([4])
✓ Metadata preparation test passed!

Testing attention op...
  output shape: torch.Size([2, 4, 8, 128])
✓ Attention op test passed!

============================================================
All tests completed!
============================================================
```

### Benchmark

Compare with other backends:

```bash
trtllm-bench --model nvidia/Llama-3.1-8B-Instruct-FP8 \
  throughput \
  --dataset Llama-3.1-8B-Instruct-FP8_1k_1k_64.inp \
  --backend _autodeploy \
  --extra_llm_api_options llama_8b_flashinfer_trtllm.yaml
```

## Code Complexity Comparison

### Direct thop.attention Approach
- **Total lines:** ~1200
- **Metadata translation:** ~400 lines
- **PT cache backend:** ~800 lines
- **Host/device tensor management:** Complex
- **CUDA graph handling:** Manual with careful tensor slicing

### FlashInfer Interface Approach
- **Total lines:** ~300
- **Metadata conversion:** ~100 lines
- **No cache backend needed:** 0 lines
- **Tensor management:** Simple, handled by FlashInfer
- **CUDA graph handling:** Built-in support

## Performance Expectations

- **Kernel performance:** Same as direct approach (both use TRT-LLM kernels)
- **Metadata overhead:** Similar or better (FlashInfer's C++ implementation)
- **CUDA graph overhead:** Better (no complex host/device juggling)
- **Memory usage:** Similar (same cache layout)

## Migration from Direct thop.attention

If you're currently using `attn_backend: trtllm`:

1. **Update configuration:**
   ```yaml
   # Old
   attn_backend: trtllm
   use_pt_cache_backend: true

   # New
   attn_backend: flashinfer_trtllm
   # No use_pt_cache_backend needed!
   ```

2. **Remove PT cache backend setup:**
   - No need to call `enable_pt_cache_backend()`
   - No need to set model config

3. **Test thoroughly:**
   - Run existing benchmarks
   - Verify CUDA graph compatibility
   - Check memory usage

## Limitations

Current implementation:
- ✅ Supports causal attention
- ✅ Supports GQA (grouped query attention)
- ✅ Supports FP16/BF16 compute
- ✅ Supports FP8 KV cache (via FlashInfer)
- ✅ CUDA graph compatible
- ⚠️ Requires FlashInfer with TRT-LLM backend support
- ⚠️ SM 8.9+ for TRT-LLM backend (Ada, Hopper, Blackwell)
- ❌ No MLA support yet (can be added)
- ❌ No speculative decoding support yet (can be added)

## Development Notes

### Adding New Features

To add support for new features (e.g., MLA, speculative decoding):

1. Check if FlashInfer's `trtllm_batch_decode_with_kv_cache` supports it
2. Update metadata preparation if needed
3. Pass required parameters through constants
4. Update documentation

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Key areas to check:
- Metadata shapes: `block_tables`, `seq_lens`
- Cache layout: `[num_pages, page_size, num_kv_heads, head_dim]`
- Backend selection: Verify `backend="trtllm-gen"` is used

## References

- FlashInfer documentation: https://docs.flashinfer.ai/
- TRT-LLM attention: `tensorrt_llm.bindings.internal.thop.attention`
- Original implementation: `tensorrt_llm/_torch/auto_deploy/custom_ops/trtllm_attention.py`

## Contributing

To improve this backend:

1. Create a branch: `git checkout -b feature/flashinfer-trtllm-improvement`
2. Make changes to `flashinfer_trtllm_attention.py`
3. Add tests to `test_flashinfer_trtllm.py`
4. Run tests: `python test_flashinfer_trtllm.py`
5. Submit PR with benchmark results

## Acknowledgments

- FlashInfer team for providing the TRT-LLM backend interface
- TRT-LLM team for the optimized kernels
- AutoDeploy team for the extensible backend architecture

---

**Status:** ✅ Implemented and tested
**Date:** January 2026
**Complexity:** ~300 lines (vs ~1200 lines for direct approach)
