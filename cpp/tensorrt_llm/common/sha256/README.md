# SHA-256 (vendored from Bitcoin Core, modified)

This directory contains the SHA-256 implementation from the
[Bitcoin Core](https://github.com/bitcoin/bitcoin) project, **reduced and
adapted by NVIDIA** to the minimal single-block hasher needed by TensorRT-LLM.
It provides a portable scalar SHA-256 with runtime dispatch to a
hardware-accelerated transform (x86 SHA-NI or ARMv8 crypto extensions), exposed
through the `CSHA256` class.

TensorRT-LLM uses it in the C++ KVCacheManagerV2 (`blockRadixTree`) to hash
token sequences into block keys, byte-identically to the Python backend's
`hashlib.sha256` block keys.

## Provenance

- **Upstream project:** Bitcoin Core — https://github.com/bitcoin/bitcoin
- **Source path upstream:** `src/`, `src/crypto/`, and `src/compat`
- **Upstream commit:** `70d9ec7f3d452789d04dce81dc02db0b3b778bb5` (branch `master`)

## Contents

| File | Notes |
|------|-------|
| `sha256.h` | `CSHA256` API; `SHA256D64` declaration removed |
| `sha256.cpp` | Scalar core + `CSHA256` + slim runtime dispatch |
| `sha256_x86_shani.cpp` | x86 SHA-NI transform (guard changed) |
| `sha256_arm_shani.cpp` | ARMv8 crypto transform (guard changed) |
| `attributes.h` | `ALWAYS_INLINE` macro used by the x86 transform |
| `sha256_endian.h` | Big-endian helpers replacing upstream `common.h` |

## NVIDIA modifications

The files are reduced to the single-block path TensorRT-LLM needs.

Changes vs. upstream:

- **Removed** the public `SHA256D64` double-hash API (and its scalar
  `TransformD64`) and the standalone SSE4 / SSE4.1 / AVX2 multi-block transform
  files (unused — TensorRT-LLM only calls single-block `CSHA256`). The two
  SHA-NI files retain their upstream 2-way `Transform_2way` helpers; these are
  now unreferenced but were left in place to keep the transforms byte-close to
  upstream.
- **Removed** the upstream support-header chain (`crypto/common.h`,
  `compat/endian.h`, `compat/byteswap.h`, `compat/cpuid.h`), which pulled in
  C++20 (`<bit>`, `<concepts>`). Endian helpers are now the small,
  NVIDIA-authored `sha256_endian.h`; CPU detection uses a `CPUID` leaf-7 check
  via `__get_cpuid_count` (x86) / `getauxval(AT_HWCAP)` (aarch64). The vendored
  sources now build as plain **C++17**.
- **Flattened** the directory (no `crypto/` / `compat/` subtree) and switched
  the HW-transform build guards from the upstream `ENABLE_*` macros to target
  architecture macros. Build wiring (per-file `-msha` / `-march=armv8-a+crypto`)
  lives in `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/CMakeLists.txt`.

