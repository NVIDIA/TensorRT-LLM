## Summary

Refactor the sparse attention framework to cleanly separate sparse-attention-specific logic from the MLA attention module, establishing a unified interface pattern for all sparse attention methods (DSA, RocketKV, SkipSoftmax).

### Key changes:

- **Split monolithic `dsa.py`/`rocket.py` into per-algorithm subdirectories** (`dsa/`, `rocket/`, `skip_softmax/`) with separate files for backend, metadata, indexer, kernels, and FlashMLA forward
- **Move all DSA-specific dispatch logic from `attention.py` into `dsa/backend.py`**: `pre_attn_process()`, `forward_sparse_context()`, `forward_sparse_generation()`, `sparse_attn_inplace()`
- **Add `forward_generation()` to MLA** (parallel to existing `forward_context()`), with sparse dispatch via `hasattr` checks on the backend
- **Unify sparse parameter preparation with three-interface design** in `trtllm.py`:
  - `sparse_params()` — non-index parameters (block_size, topk, skip_softmax thresholds)
  - `sparse_kv_predict()` — KV block indices/offsets
  - `sparse_attn_predict()` — attention block indices/offsets
  - `_prepare_sparse_params()` orchestrates all three — NOT overridden by subclasses
- **Add `SparseParams` dataclass** (`sparse/params.py`) containing all sparse parameters for `wrapper.plan()`
- **Create `SkipSoftmaxTrtllmAttention`** backend that only overrides `sparse_params()`, unifying SkipSoftmax with other sparse methods (eliminates `isinstance` checks in `trtllm.py`)
- **Restore piecewise CUDA graph support** for DSA via `sparse_attn_inplace()` interface and custom ops in `dsa/custom_ops.py`
- **Move FlashMLA (SM < 100) sparse forward** to `dsa/flash_mla.py`
- **Remove backward-compat wrappers** (`sparse/utils.py`, `sparse/kernel.py`)
- **C++ cleanup**: document `SparseAttentionParams` struct, clean up sparse flags in `AttentionOp`

### Architecture after refactor:

```
MLA.forward_impl()
  → mqa.pre_attn_process()          # indexer projections + k-cache scatter
  → forward_context(**intermediates) # dispatches to backend
    → mqa.forward_sparse_context()  # SM>=100: absorption, SM<100: FlashMLA
  → forward_generation(**intermediates)
    → mqa.forward_sparse_generation()

MLA.forward() [register_to_config path]
  → mqa.sparse_attn_inplace()       # piecewise CUDA graph via custom ops

TrtllmAttention.forward()
  → _prepare_sparse_params()        # orchestrates three interfaces:
    → sparse_params()               # each backend overrides what it needs
    → sparse_kv_predict()
    → sparse_attn_predict()
  → wrapper.plan(**sparse_params)
```

## Test plan

- [x] `tests/unittest/_torch/attention/` — 236 passed, 0 failed, 216 skipped
- [ ] Integration tests with DSA model (DeepSeek-V3)
- [ ] Verify piecewise CUDA graph capture works end-to-end
