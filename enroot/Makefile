ifndef MAKEFILE_PYXIS_INCLUDED
MAKEFILE_PYXIS_INCLUDED := 1

BASE_IMAGE         ?= $(shell grep '^ARG BASE_IMAGE=' ../docker/Dockerfile.multi | grep -o '=.*' | tr -d '="')
BASE_TAG           ?= $(shell grep '^ARG BASE_TAG=' ../docker/Dockerfile.multi | grep -o '=.*' | tr -d '="')
SQSH_PATH          ?= tensorrt_llm.devel.sqsh
SOURCE_DIR         ?= $(shell readlink -f ..)
CODE_DIR           ?= /code/tensorrt_llm
RUN_CMD            ?= --pty bash

PYTHON_VERSION ?= 3.12.3
TORCH_INSTALL_TYPE ?= skip
GITHUB_MIRROR ?=
CUDA_VERSION ?=
CUDNN_VERSION ?=
NCCL_VERSION ?=
CUBLAS_VERSION ?=
TRT_VERSION ?=

build_sqsh:
	@echo "Building trtllm sqsh image."
	@echo "Base image: $(BASE_IMAGE):$(BASE_TAG)"
	@echo "Location: $(SQSH_PATH)"

	srun \
		--container-image "$(BASE_IMAGE):$(BASE_TAG)" \
		--container-save "$(SQSH_PATH)" \
		--container-mounts "$(SOURCE_DIR):$(CODE_DIR)" --container-workdir $(CODE_DIR)/docker/common \
		--container-mount-home --container-remap-root \
		--export PYTHON_VERSION=$(PYTHON_VERSION),GITHUB_MIRROR=$(GITHUB_MIRROR),TORCH_INSTALL_TYPE=$(TORCH_INSTALL_TYPE),CUDA_VER=$(CUDA_VERSION),CUDNN_VER=$(CUDNN_VERSION),NCCL_VER=$(NCCL_VERSION),CUBLAS_VER=$(CUBLAS_VERSION),TRT_VER=$(TRT_VERSION) \
		./install.sh --all

run_sqsh:
	@echo "Running srun job step with:"
	@echo "    sqsh image: $(SQSH_PATH)"
	@echo "    run command: $(RUN_CMD)"

	srun \
		--container-image "$(SQSH_PATH)" \
		--container-mounts "$(SOURCE_DIR):$(CODE_DIR)" --container-workdir $(CODE_DIR) \
		--container-mount-home --container-remap-root \
		--export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.99999 \
		$(RUN_CMD)

endif
