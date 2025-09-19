# Using Dev Containers

The TensorRT LLM repository contains a [Dev Containers](https://containers.dev/)
configuration in `.devcontainer`. These files are intended for
use with [Visual Studio Code](https://code.visualstudio.com/).

Due to the various container options supported by TensorRT LLM (see
[](/installation/build-from-source-linux.md) and
<https://github.com/NVIDIA/TensorRT-LLM/tree/main/docker>), the Dev
Container configuration also offers some degree of customization.

Generally, the `initializeCommand` in `devcontainer.json` will run
`make_env.py` to generate an
[`.env` file for `docker-compose`](https://docs.docker.com/compose/how-tos/environment-variables/variable-interpolation/#env-file-syntax).
Most importantly, the `docker-compose.yml` uses `${DEV_CONTAINER_IMAGE}`
as base image.
The generated `.devcontainer/.env` is not tracked by Git and combines
data from the following sources:

* `jenkins/current_image_tags.properties` which contains the image tags
  currently used by CI.

* `.devcontainer/devcontainer.env` which contains common configuration
  settings and is tracked by Git.

* `.devcontainer/devcontainer.env.user` (optional) which is ignored by
  Git and can be edited to customize the Dev Container behavior.

The source files are processed using `sh`, in the order in which they
are listed above. Thus, features like command substitution are supported.

The following sections provide more detail on particular Dev Container
configuration parameters which can be customized.

```{note}
After editing any of the configuration files, it may be necessary
to execute the "Dev Containers: Reopen Folder in SSH" (if applicable) and
"Dev Containers: Rebuild and Reopen in Container" Visual Studio Code
commands.
```

## Container image selection

By default, `make_env.py` will attempt to auto-select a suitable container
image as follows:

1. Reuse the development container image used by CI. This requires access
   to the NVIDIA internal artifact repository.

1. Use the most recent
   [NGC Development container image](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/devel)
   associated with a Git tag which is reachable from the currently checked
   out commit.

1. Build a development image locally.

Set `DEV_CONTAINER_IMAGE=<some_uri>` to bypass the aforementioned discovery
mechanism if desired.

By setting `LOCAL_BUILD=0`, the local image build can be disabled. In this
case, execution fails if no suitable pre-built image is found.

Setting `LOCAL_BUILD=1` forces building of a local image, even if a pre-built
image is available.

## Volume Mounts

[Docker volume mounts](https://docs.docker.com/engine/storage/volumes/#use-a-volume-with-docker-compose) can be customized by editing
`docker-compose.yml`, which allows using any variables defined in `.env`.

By default, the Dev Container configuration mounts the VS Code workspace into
`/workspaces/tensorrt_llm` and `~/.cache/huggingface` into `/huggingface`.
The source paths can be overridden by setting `SOURCE_DIR` and `HOME_DIR`
in `.devcontainer/devcontainer.env.user`, respectively. This is of
particular relevance when using
[Docker Rootless Mode](https://docs.docker.com/engine/security/rootless/),
which requires configuring UID/GID translation using a tool like `bindfs`.
The Dev Container scripts contain heuristics to detect Docker Rootless
Mode and will issue an error if these variables are not set.
An analogous logic is applied to `HF_HOME`.


## Overriding Docker Compose configuration

When starting the container, `.devcontainer/docker-compose.yml`
is [merged](https://docs.docker.com/compose/how-tos/multiple-compose-files/merge/) with
`.devcontainer/docker-compose.override.yml`. The latter file is not
tracked by Git and will be created by `make_env.py` if it does not exist.

This mechanism can be used, e.g., to add custom volume mounts:

```{literalinclude} /../../.devcontainer/docker-compose.override-example.yml
```

It is possible to conditionally mount volumes by combining, e.g.,
[this method] (https://stackoverflow.com/a/61954812) and shell command
substitution in `.devcontainer/devcontainer.env.user`.

If no `.devcontainer/docker-compose.override.yml` file is found, the Dev Container
initialization script will create one with the contents listed above.
