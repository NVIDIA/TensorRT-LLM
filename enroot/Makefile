BASE_IMAGE         ?= $(shell grep '^ARG BASE_IMAGE=' ../docker/Dockerfile.multi | grep -o '=.*' | tr -d '="')
BASE_TAG           ?= $(shell grep '^ARG BASE_TAG=' ../docker/Dockerfile.multi | grep -o '=.*' | tr -d '="')
SQSH_PATH          ?= tensorrt_llm.devel.sqsh
SOURCE_DIR        ?= $(shell readlink -f ..)
RUN_CMD            ?= --pty bash

PYTHON_VERSION ?= 3.12.3
TORCH_INSTALL_TYPE ?= skip
GITHUB_MIRROR ?=
CUDA_VERSION ?=
CUDNN_VERSION ?=
NCCL_VERSION ?=
CUBLAS_VERSION ?=
TRT_VERSION ?=

ifndef MAKEFILE_PYXIS_INCLUDED
MAKEFILE_PYXIS_INCLUDED := 1


build_sqsh:
	@echo "Building trtllm sqsh image."
	@echo "Base image: $(BASE_IMAGE):$(BASE_TAG)"
	@echo "Location: $(SQSH_PATH)"

	srun \
		--container-image "$(BASE_IMAGE):$(BASE_TAG)" \
		--container-save "$(SQSH_PATH)" \
		--container-mounts "$(SOURCE_DIR):/code/tensorrt_llm" --container-workdir /code/tensorrt_llm/docker/common \
		--container-mount-home --container-remap-root \
		$(if $(CUDA_VERSION), --container-env CUDA_VER="$(CUDA_VERSION)") \
		$(if $(CUDNN_VERSION), --container-env CUDNN_VER="$(CUDNN_VERSION)") \
		$(if $(NCCL_VERSION), --container-env NCCL_VER="$(NCCL_VERSION)") \
		$(if $(CUBLAS_VERSION), --container-env CUBLAS_VER="$(CUBLAS_VERSION)") \
		$(if $(TRT_VERSION), --container-env TRT_VER="$(TRT_VERSION)") \
		$(if $(PYTHON_VERSION), --container-env PYTHON_VERSION="$(PYTHON_VERSION)") \
		$(if $(GITHUB_MIRROR), --container-env GITHUB_MIRROR="$(GITHUB_MIRROR)") \
		$(if $(TORCH_INSTALL_TYPE), --container-env TORCH_INSTALL_TYPE="$(TORCH_INSTALL_TYPE)") \
		./install.sh --all

endif

run_sqsh:
	@echo "Running srun job step with:"
	@echo "    sqsh image: $(SQSH_PATH)"
	@echo "    run command: $(RUN_CMD)"

	srun \
		--container-image "$(SQSH_PATH)" \
		--container-mounts "$(SOURCE_DIR):/code/tensorrt_llm" --container-workdir /code/tensorrt_llm \
		--container-mount-home --container-remap-root \
		--container-env PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.99999 \
		$(RUN_CMD)
