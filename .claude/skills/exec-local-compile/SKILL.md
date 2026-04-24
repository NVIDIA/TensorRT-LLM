---
name: exec-local-compile
description: Compile TensorRT-LLM on a compute node inside a Docker container. Use this when already on a compute node with GPUs visible.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Compile TensorRT-LLM (Local / Compute Node)

Compile TensorRT-LLM from source on a compute node inside a Docker container.

## When to Use

| Scenario | Use This Skill? |
|----------|----------------|
| On a compute node with GPUs visible (`nvidia-smi` works) | Yes |
| On a SLURM login node (no GPUs) | No — use `exec-slurm-compile` instead |

## Prerequisites

- You are inside a Docker/enroot container on a compute node
- `nvidia-smi` succeeds (GPUs visible)
- `/usr/local/tensorrt` exists (TensorRT installation in the container)

## Instructions

### Step 1: Verify Environment

Run `nvidia-smi` to confirm you are on a compute node with GPU access.

### Step 2: Locate the Codebase

`cd` to the TensorRT-LLM repository. If the path is not provided by the user, ask for it.

### Step 3: (Optional) Checkout Branch

If the user specifies a branch (e.g., "compile ToT"), checkout and pull:
```bash
git checkout main && git pull
```

### Step 4: Build

Run the build command (**incremental by default** — omit `-c`/`--clean` unless explicitly requested or the incremental build fails):

```bash
./scripts/build_wheel.py --trt_root /usr/local/tensorrt --benchmarks -ccache -a "<arch>" -f --nvtx
```

Replace `<arch>` with the target GPU architecture (see Architecture Reference below). If not specified by the user, auto-detect from `nvidia-smi`.

### Step 5: Install

```bash
pip install -e .[devel]
```

### Step 6: Verify

```bash
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
```

## Build Flags

| Flag | Description |
|------|-------------|
| `--trt_root /usr/local/tensorrt` | TensorRT installation path (standard in NVIDIA containers) |
| `--benchmarks` | Build the C++ benchmarks |
| `-a "<arch>"` | Target GPU architecture(s) |
| `--nvtx` | Enable NVTX markers for profiling |
| `-ccache` | Use ccache for faster recompilation |
| `-f` / `--fast_build` | Skip some kernels for faster dev compilation. **Always use for dev builds.** |
| `-c` / `--clean` | Clean build directory before building. Only when needed (see below). |
| `--skip_building_wheel` | Build in-place without creating a wheel file |
| `--no-venv` | Skip virtual environment creation |

## Architecture Reference

| Value | GPU Family |
|-------|-----------|
| `"100-real"` | Blackwell (B200, GB200) |
| `"90-real"` | Hopper (H100, H200) |
| `"89-real"` | Ada Lovelace (L40S) |
| `"80-real"` | Ampere (A100) |
| `"90;100-real"` | Multiple architectures |

## Incremental vs. Clean Builds

**Default to incremental builds** — CMake only recompiles changed files, saving significant time.

Use a **clean build** (`-c`) only when:
- The user explicitly requests a clean/fresh build
- An incremental build fails with linker errors, stale object files, or CMake cache issues
- Major branch changes (e.g., rebasing across many commits) that may invalidate the build cache
- Build system files changed (`CMakeLists.txt`, `*.cmake`)
