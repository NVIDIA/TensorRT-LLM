(build-from-source-linux)=
(build-from-source)=

# Build from Source

Building from source is mostly intended for developers who wish to modify, customize, and contribute to TensorRT LLM. If you only need to run TensorRT LLM, use the [Installation Guide](installation-guide) instead.

## Prerequisites

Use [Docker](https://www.docker.com) to build and run TensorRT LLM. Instructions to install an environment to run Docker containers for the NVIDIA platform can be found [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

TensorRT LLM uses git-lfs, which needs to be installed in advance:

```bash
apt-get update && apt-get -y install git git-lfs
git lfs install
```

## Step 1: Clone the Repository

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull
```

## Step 2: Pull the Development Container

Pull the pre-built TensorRT LLM `devel` container from NGC. Replace `x.y.z` with the desired version. Browse the [available tags on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/devel/tags) to find the latest release.

```bash
docker pull nvcr.io/nvidia/tensorrt-llm/devel:x.y.z
```

## Step 3: Start the Container

From the repository root, start a development container with the source tree mounted into it.

```bash
docker run --rm -it \
        --ipc=host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        --gpus=all \
        --volume <path_to_tensorrt_llm_on_host>:<path_to_tensorrt_llm_in_container> \
        --workdir <path_to_tensorrt_llm_in_container> \
        nvcr.io/nvidia/tensorrt-llm/devel:x.y.z
```

```{admonition} Note on Docker flags
:class: dropdown note
- `--ipc=host` is required to avoid `Bus error (core dumped)` when running TensorRT LLM inside the container.
- `--ulimit memlock=-1` allows unlimited locked memory, which is needed for GPU workloads.
- `--ulimit stack=67108864` sets the stack size to 64 MB to prevent stack overflows in deeply nested C++/CUDA code paths.
```

## Step 4: Build TensorRT LLM

Once inside the container, build TensorRT LLM from source using `scripts/build_wheel.py`. Run `python3 ./scripts/build_wheel.py --help` for the full list of options.

### Typical development build

Build the C++ code, skip wheel packaging, and use symlinks so that changes are reflected immediately. Then install in editable mode for Python development.

```bash
python3 scripts/build_wheel.py --use_ccache -a "90-real" --skip_building_wheel --linking_install_binary
pip install -e .
```

Key flags used above:

| Flag | Purpose |
|------|---------|
| `--use_ccache` | Use ccache for faster incremental rebuilds |
| `-a "90-real"` | Build only for a specific GPU architecture (e.g. Hopper). Reduces compile time significantly. See {ref}`support-matrix-hardware` for values. |
| `--skip_building_wheel` | Skip `.whl` packaging -- only needed for distribution, not development |
| `--linking_install_binary` | Symlink built libraries instead of copying them |
| `pip install -e .` | Editable install so Python changes take effect without reinstalling |

### Other common options

| Flag | Purpose |
|------|---------|
| `--clean` | Clean the build directory before building |
| `--build_type RelWithDebInfo` | Build with debug info (default: `Release`) |
| `-j <N>` | Number of parallel compile jobs (default: number of available CPUs) |
| `--fast_build` | Skip compiling some kernels to speed up compilation -- for development only |
| `--cpp_only` | Build only the C++ runtime library, without Python bindings |

### Python-only build (no C++ compilation)

If you only need to modify Python code, you can skip C++ compilation entirely by reusing precompiled binaries:

```bash
TRTLLM_USE_PRECOMPILED=1 pip install -e .
```

This downloads a precompiled wheel matching the version in `tensorrt_llm/version.py` and extracts its compiled libraries into your working directory. Override the version with `TRTLLM_USE_PRECOMPILED=x.y.z` or specify a custom URL/path with `TRTLLM_PRECOMPILED_LOCATION`.
