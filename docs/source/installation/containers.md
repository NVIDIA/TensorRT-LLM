(build-containers)=

# Container Images

TensorRT LLM uses a [multi-stage Dockerfile](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docker/Dockerfile.multi) that produces three image types:

| Stage | Purpose | NGC Image |
|-------|---------|-----------|
| **`devel`** | Development environment with all build dependencies pre-installed. No TensorRT LLM source or wheel included. Mount your source checkout and build inside. | [`devel`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/devel) |
| **`wheel`** | Intermediate build stage. Extends `devel`, copies the source tree, and compiles the TensorRT LLM wheel. Not published as a standalone image. | -- |
| **`release`** | Runtime image. Extends `devel`, installs the pre-built wheel from the `wheel` stage. Ready to use with no further compilation. | [`release`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release) |

## Pre-built Images on NGC

The `devel` and `release` images are published to NGC and can be pulled directly:

```bash
# Pull the development image (for building from source)
docker pull nvcr.io/nvidia/tensorrt-llm/devel:x.y.z

# Pull the release image (ready to run)
docker pull nvcr.io/nvidia/tensorrt-llm/release:x.y.z
```

Replace `x.y.z` with the desired version. Browse the available tags for [`devel`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/devel/tags) and [`release`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags) on NGC.

{{container_tag_admonition}}

## Building Images Locally

All local image builds require the TensorRT LLM source tree and approximately 63 GB of free disk space. Clone the repository first if you have not already:

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull
```

### Build the `devel` Image

**On systems with GNU `make`**

Create a Docker image for development. The image will be tagged locally with `tensorrt_llm/devel:latest`.

```bash
make -C docker build
```

Run the container:

```bash
make -C docker run
```

If you prefer to work with your own user account in that container, instead of `root`, add the `LOCAL_USER=1` option.

```bash
make -C docker run LOCAL_USER=1
```

**On systems without GNU `make`**

```bash
docker build --pull \
            --target devel \
            --file docker/Dockerfile.multi \
            --tag tensorrt_llm/devel:latest \
            .
```

```bash
docker run --rm -it \
        --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all \
        --volume ${PWD}:/code/tensorrt_llm \
        --workdir /code/tensorrt_llm \
        tensorrt_llm/devel:latest
```

Note: please make sure to set `--ipc=host` as a docker run argument to avoid `Bus error (core dumped)`.

Once inside the container, follow the steps in [Building from Source](build-from-source) (starting from Step 4) to build TensorRT LLM.

### Build the `release` Image (One Step)

This builds TensorRT LLM from source and installs it into a single ready-to-run container image.

```bash
make -C docker release_build
```

You can add the `CUDA_ARCHS="<list of architectures in CMake format>"` optional argument to specify which architectures should be supported by TensorRT LLM. It restricts the supported GPU architectures but helps reduce compilation time:

```bash
# Restrict the compilation to Ada and Hopper architectures.
make -C docker release_build CUDA_ARCHS="89-real;90-real"
```

After the image is built, the Docker container can be run.

```bash
make -C docker release_run
```

The `make` command supports the `LOCAL_USER=1` argument to switch to the local user account instead of `root` inside the container. The examples of TensorRT LLM are installed in the `/app/tensorrt_llm/examples` directory.

## Using Enroot (Slurm Clusters)

If you wish to use enroot instead of Docker, you can build a sqsh file that has the identical environment as the development image `tensorrt_llm/devel:latest`.

1. Allocate a compute node:
    ```bash
    salloc --nodes=1
    ```

2. Create a sqsh file with essential TensorRT LLM dependencies installed:
    ```bash
    # Using default sqsh filename (enroot/tensorrt_llm.devel.sqsh)
    make -C enroot build_sqsh

    # Or specify a custom path (optional)
    make -C enroot build_sqsh SQSH_PATH=/path/to/dev_trtllm_image.sqsh
    ```

3. Once this squash file is ready, you can follow the steps under [Building from Source](build-from-source) by launching an enroot sandbox:
    ```bash
    export SQSH_PATH=/path/to/dev_trtllm_image.sqsh

    # Start a pseudo terminal for interactive session
    make -C enroot run_sqsh

    # Or, you could run commands directly
    make -C enroot run_sqsh RUN_CMD="python3 scripts/build_wheel.py"
    ```

## Advanced Topics

For more information on building and running various TensorRT LLM container images,
check <https://github.com/NVIDIA/TensorRT-LLM/tree/main/docker>.
