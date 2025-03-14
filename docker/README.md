# The Docker Build System

## Multi-stage Builds with Docker

TensorRT-LLM can be compiled in Docker using a multi-stage build implemented in [`Dockerfile.multi`](Dockerfile.multi).
The following build stages are defined:

* `devel`: this image provides all dependencies for building TensorRT-LLM.
* `wheel`: this image contains the source code and the compiled binary distribution.
* `release`: this image has the binaries installed and contains TensorRT-LLM examples in `/app/tensorrt_llm`.

## Building Docker Images with GNU `make`

The GNU [`Makefile`](Makefile) in the `docker` directory provides targets for building, pushing, and running each stage
of the Docker build. The corresponding target names are composed of two components, namely, `<stage>` and `<action>`
separated by `_`. The following actions are available:

* `<stage>_build`: builds the docker image for the stage.
* `<stage>_push`: pushes the docker image for the stage to a docker registry (implies `<stage>_build`).
* `<stage>_run`: runs the docker image for the stage in a new container.

For example, the `release` stage is built and pushed from the top-level directory of TensorRT-LLM as follows:

```bash
make -C docker release_push
```

Note that pushing the image to a docker registry is optional. After building an image, run it in a new container with
```bash
make -C docker release_run
```

### Building and Running Options

The full image name and tag can be controlled by supplying `IMAGE_WITH_TAG` to `make`:

```bash
make -C docker devel_push IMAGE_WITH_TAG="urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:dev"
```

Containers can be started with the local user instead of `root` by appending `LOCAL_USER=1` to the run target:

```bash
make -C docker devel_run LOCAL_USER=1
```

Specific CUDA architectures supported by the `wheel` can be specified WITH `CUDA_ARCHS`:

```bash
make -C docker release_build CUDA_ARCHS="80-real;90-real"
```

For more build options, see the variables defined in [`Makefile`](Makefile).

### Jenkins Integration

[`Makefile`](Makefile) has special targets for building, pushing and running the Docker build image used on Jenkins.
The full image name and tag is defined in [`L0_MergeRequest.groovy`](../jenkins/L0_MergeRequest.groovy). The `make`
system will parse this name as the value of `LLM_DOCKER_IMAGE`. To build and push a new Docker image for Jenkins,
define a new image name and tag in [`L0_MergeRequest.groovy`](../jenkins/L0_MergeRequest.groovy) and run

```bash
make -C docker jenkins_push
```

Start a new container using the same image as Jenkins using your local user account with

```bash
make -C docker jenkins_run LOCAL_USER=1
```

One may also build a release image based on the Jenkins development image:

```bash
make -C docker trtllm_build CUDA_ARCHS="80-real;90-real"
```

These images can be pushed to
the [internal artifact repository](https://urm.nvidia.com/artifactory/sw-tensorrt-docker/tensorrt-llm-staging/release/):

```bash
make -C docker trtllm_push
```

Generally, only images built for all CUDA architectures should be pushed to the artifact repository. These images can
be deployed in docker in the usual way:

```bash
make -C docker trtllm_run LOCAL_USER=1 DOCKER_PULL=1
```

The argument `DOCKER_PULL=1` instructs `make` to pull the latest version of the image before deploying it in the container.
By default, images are tagged by their `git` branch name and may be frequently updated.

### Binary Compatible Environment

Currently, `BatchManager` is released as a closed source binary library. In order to make it deployable in a wider
scope, the compilation environment needs to be constructed in the following way.

The compilation environment for x86_64 architecture

```bash
make -C docker centos7_push
```
