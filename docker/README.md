# The Docker Build System

## Multi-stage Builds with Docker

TensorRT LLM can be compiled in Docker using a multi-stage build implemented in [`Dockerfile.multi`](Dockerfile.multi).
The following build stages are defined:

* `devel`: this image provides all dependencies for building TensorRT-LLM.
* `wheel`: this image contains the source code and the compiled binary distribution.
* `release`: this image has the binaries installed and contains TensorRT LLM examples in `/app/tensorrt_llm`.

## Building Docker Images with GNU `make`

The GNU [`Makefile`](Makefile) in the `docker` directory provides targets for building, pushing, and running each stage
of the Docker build. The corresponding target names are composed of two components, namely, `<stage>` and `<action>`
separated by `_`. The following actions are available:

* `<stage>_build`: builds the docker image for the stage.
* `<stage>_push`: pushes the docker image for the stage to a docker registry (implies `<stage>_build`).
* `<stage>_run`: runs the docker image for the stage in a new container.

For example, the `release` stage is built and pushed from the top-level directory of TensorRT LLM as follows:

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

Extra docker volumes can be mounted in addition to the code repository by appending `EXTRA_VOLUMES=` to the run target:
```bash
make -C docker devel_run LOCAL_USER=1 EXTRA_VOLUMES="-v /pathA:/pathA -v /pathB:/pathB"
```

Specific CUDA architectures supported by the `wheel` can be specified with `CUDA_ARCHS`:

```bash
make -C docker release_build CUDA_ARCHS="80-real;90-real"
```

The `run` action maps the locally checked out source code into the `/code/tensorrt_llm` directory within the container.

The `DOCKER_RUN_ARGS` option can be used to pass additional options to Docker,
e.g., in order to mount additional volumes into the container.

For more build options, see the variables defined in [`Makefile`](Makefile).

### NGC Integration

When building from source, one can conveniently download a docker image for development from
the [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/) and start it like so:

```bash
make -C docker ngc-devel_run LOCAL_USER=1 DOCKER_PULL=1
```

As before, specifying `LOCAL_USER=1` will run the container with the local user's identity. Specifying `DOCKER_PULL=1`
is optional, but it will pull the latest image from the NGC Catalog.

We also provide an image with pre-installed binaries for release. This can be used like so:

```bash
make -C docker ngc-release_run LOCAL_USER=1 DOCKER_PULL=1
```

If you want to deploy a specific version of TensorRT-LLM, you can specify the version with
`IMAGE_TAG=<version_tag>` (cf. [release history on GitHub](https://github.com/NVIDIA/TensorRT-LLM/releases) and [tags in NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags)). The application examples and benchmarks are installed
in `/app/tensorrt_llm`.

See the description of the `<stage>_run` make target in
[Building and Running Options](#building-and-running-options) for additional information and
running options.

If you cannot access the NGC container images, you can instead locally build and use
equivalent containers as [described above](#building-docker-images-with-gnu-make).

### Jenkins Integration

[`Makefile`](Makefile) has special targets for building, pushing and running the Docker build image used on Jenkins.
The full image names and tags are defined in [`current_image_tags.properties`](../jenkins/current_image_tags.properties). The `make`
system will parse the names/tags from this file.

#### Running

Start a new container using the same image as Jenkins using your local user account with

```bash
make -C docker jenkins_run LOCAL_USER=1
```

If you do not have access to the [internal artifact repository](https://urm.nvidia.com/artifactory/sw-tensorrt-docker/tensorrt-llm/), you can instead either use the [NGC Develop
image](#ngc-integration) or [build an image locally](#building-docker-images-with-gnu-make).

#### Release images based on Jenkins image

One may also build a release image based on the Jenkins development image:

```bash
make -C docker trtllm_build CUDA_ARCHS="80-real;90-real"
```

Note that the above requires access to the Jenkins development image from the
[internal artifact repository](https://urm.nvidia.com/artifactory/sw-tensorrt-docker/tensorrt-llm/).

The resulting images can be pushed to
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
By default, the release images built in the above manner are tagged by their `git` branch name and may be frequently updated.

#### Building CI images

To build and push a new Docker image for Jenkins, define new image names and tags in [`current_image_tags.properties`](../jenkins/current_image_tags.properties) and run

```bash
# Commands assume an amd64 host
make -C docker jenkins_build
#
docker buildx create --name multi-builder
make -C docker jenkins-aarch64_build \
    DOCKER_BUILD_ARGS="--platform arm64 --builder=multi-builder"
#
# check jenkins/BuildDockerImage.groovy for current Python versions
make -C docker jenkins-rockylinux8_build PYTHON_VERSION=3.12.3
make -C docker jenkins-rockylinux8_build PYTHON_VERSION=3.10.12
```

The resulting images then need to be pushed:

```bash
sh -c '. jenkins/current_image_tags.properties && echo $LLM_DOCKER_IMAGE $LLM_SBSA_DOCKER_IMAGE $LLM_ROCKYLINUX8_PY310_DOCKER_IMAGE $LLM_ROCKYLINUX8_PY312_DOCKER_IMAGE' | tr ' ' '\n' | xargs -I{} docker push {}
```

Alternatively, it is possible to trigger the image build by opening a new pull request and commenting

```text
/bot run --stage-list "Build-Docker-Images"
```

The resulting images can then be re-tagged using `scripts/rename_docker_images.py`
and the new tags included in [`current_image_tags.properties`](../jenkins/current_image_tags.properties).

### Docker rootless

Some aspects require special treatment when using [Docker rootless mode](https://docs.docker.com/engine/security/rootless/). The `docker/Makefile` contains heuristics to detect Docker rootless mode. When assuming
Docker rootless mode, the `%_run` targets in `docker/Makefile` will output
a corresponding message. The heuristics can be overridden by specifying
`IS_ROOTLESS=0` or `IS_ROOTLESS=1`, respectively.

Since Docker rootless mode remaps the UID/GID and the remapped UIDs and GIDs
 (typically configured in `/etc/subuid` and `/etc/subgid`) generally do not coincide
with the local UID/GID, both IDs need to be translated using a tool like `bindfs` in order to be able to smoothly share a local working directory with any containers
started with `LOCAL_USER=1`. In this case, the `SOURCE_DIR` and `HOME_DIR` Makefile variables need to be set to the locations of the translated versions of the TensorRT LLM working copy and the user home directory, respectively.
