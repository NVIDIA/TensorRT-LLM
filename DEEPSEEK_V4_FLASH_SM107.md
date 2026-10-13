# Running DeepSeek-V4-Flash on Rubin (SM107)

Build notes for DeepSeek-V4-Flash on Rubin (SM107) with the TensorRT-LLM PyTorch backend.

## 1. Build (from source, in the container)

```bash
git config --global --add safe.directory '*'   # FetchContent clones are host-owned

./scripts/build_wheel.py --trt_root /usr/local/tensorrt --benchmarks \
  --use_ccache -a "107-real" -f --nvtx --nvrtc_dynamic_linking
```
- `--nvrtc_dynamic_linking` links against the dynamic NVRTC libraries.
- **cutlass SM107 patch** is applied **automatically** by FetchContent: `3rdparty/fetch_content.json`
  gives the cutlass dependency `"patch_file": "patches/cutlass_sm107.patch"`, and the loader
  (`3rdparty/CMakeLists.txt`) runs it (`patch -p1 --forward --batch`, idempotent) on the cloned
  cutlass v4.4.2 — same mechanism `xgrammar`/`deep_ep` use. No manual step; survives clean rebuilds.
  The patch (`3rdparty/patches/cutlass_sm107.patch`) maps `__CUDA_ARCH__==1070` / family-1070 to the
  SM100 (Blackwell) family so the `sm_107f` cubin enables SM100 MMA/TMA/tcgen05.

## 2. Run

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=<repo> \
  python examples/llm-api/quickstart_advanced.py \
  --model_dir <DeepSeek-V4-Flash> \
  --tp_size 4 --trust_remote_code \
  --tokens_per_block 128 \
  --moe_backend CUTLASS \
  --disable_autotuner
```

## 3. Flags

| flag / change | reason |
|---|---|
| `--tokens_per_block 128` | `DeepseekV4CacheManager` accepts 128 or 256. |
| `--moe_backend CUTLASS` | Use the CUTLASS MoE backend for this configuration. |
| `--disable_autotuner` | Skip autotuner warmup for this configuration. |
| `3rdparty/patches/cutlass_sm107.patch` (auto-applied via `fetch_content.json`) | enables SM100-family MMA/TMA/tcgen05 for sm107 at build; see above. |
