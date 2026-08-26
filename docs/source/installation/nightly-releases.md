<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

(nightly-releases)=

# Install a Nightly Release

TensorRT LLM nightly releases provide early access to development builds.

```{warning}
Nightly releases are intended for evaluation and development. Use a clean environment. You can pin an exact version
when reproducibility is required.
```

## Install the Wheel

Complete the {ref}`Linux pip installation prerequisites <install-prerequisites>`, then use a clean virtual environment:

```bash
python3 -m venv .venv-trtllm-nightly
source .venv-trtllm-nightly/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install --pre tensorrt_llm \
    --extra-index-url https://pypi.nvidia.com/trtllm_nightly/
```

Verify the installed version and GPU access:

```bash
python3 -c 'import tensorrt_llm, torch; print(tensorrt_llm.__version__); print(torch.cuda.get_device_name())'
```

View the source commit recorded in the wheel metadata:

```bash
python3 -m pip show --verbose tensorrt_llm | grep "Source Commit"
```

To pin a version, list the available builds and install one explicitly:

```bash
python3 -m pip index versions tensorrt_llm --pre \
    --index-url https://pypi.nvidia.com/trtllm_nightly/
python3 -m pip install --pre "tensorrt_llm==<version-from-the-list>" \
    --extra-index-url https://pypi.nvidia.com/trtllm_nightly/
```

If `pip` cannot find a compatible build, check the available versions, Python version, and machine architecture, or
use an available release container tag.

## Use a Container

### Release Image

Browse the available [NGC release image
tags](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags), then pull and run the
matching version:

```bash
export TRTLLM_NIGHTLY_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:<tag-from-ngc>"
docker pull "${TRTLLM_NIGHTLY_IMAGE}"
docker run --rm -it --ipc=host --gpus all \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    "${TRTLLM_NIGHTLY_IMAGE}"
```

Verify the package version in the container:

```bash
docker run --rm "${TRTLLM_NIGHTLY_IMAGE}" python3 -c \
    'import tensorrt_llm; print(tensorrt_llm.__version__)'
```

View the source commit recorded in the release image:

```bash
docker run --rm "${TRTLLM_NIGHTLY_IMAGE}" printenv TRT_LLM_GIT_COMMIT
```

### Development Image

Use a [development image tag](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/devel/tags)
when you need build tools and development dependencies:

```bash
export TRTLLM_NIGHTLY_DEVEL_IMAGE="nvcr.io/nvidia/tensorrt-llm/devel:<tag-from-ngc>"
docker pull "${TRTLLM_NIGHTLY_DEVEL_IMAGE}"
docker run --rm -it --ipc=host --gpus all \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    "${TRTLLM_NIGHTLY_DEVEL_IMAGE}"
```

See {ref}`Container Images <build-containers>` for more information about the image types.
