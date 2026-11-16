# Test Cases Failure Fix Recommendations


| Error | Root cause pattern | Recommended fix |
|-------|--------------------|-----------------|
| `ImportError` on a class | Class not exported | Add missing export to `__init__.py` |
| `AssertionError` on tensor values | Wrong computation in `forward()` | Compare `forward()` output against HuggingFace reference using `torch.allclose` |
| Shape mismatch | Wrong dimension in `__init__` or `forward()` | Verify dimensions against model config fields |
| `RuntimeError: CUDA error` | Device placement issue | Check tensor `.device` before ops; verify CUDA availability |
| `KeyError` on weight name | Wrong weight name mapping | Fix HF → TRT-LLM name translation in `load_weights()` |
| `shape mismatch in QKV fusion` | Wrong concat order | Ensure Q, K, V concatenated as `[Q, K, V]` along dim 0 |
| `AttributeError` on config field | Missing config field | Add field with appropriate default to the model config class |
| `CUDA out of memory` | Batch/sequence too large | Reduce test batch size or sequence length; call `torch.cuda.empty_cache()` before test |
| `TimeoutError` / test hangs | Deadlock or infinite loop | Inspect distributed setup; reduce test size |
