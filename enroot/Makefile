BASE_IMAGE         ?= $(shell grep '^ARG BASE_IMAGE=' ../docker/Dockerfile.multi | grep -o '=.*' | tr -d '="')
BASE_TAG           ?= $(shell grep '^ARG BASE_TAG=' ../docker/Dockerfile.multi | grep -o '=.*' | tr -d '="')
SQSH_PATH          ?= tensorrt_llm.devel.sqsh
SOURCE_DIR        ?= $(shell readlink -f ..)

ifndef MAKEFILE_PYXIS_INCLUDED
MAKEFILE_PYXIS_INCLUDED := 1


build_sqsh:
	@echo "Building trtllm sqsh image. Base image: $(BASE_IMAGE):$(BASE_TAG). Location: $(SQSH_PATH)"

	srun \
		--container-image $(BASE_IMAGE):$(BASE_TAG) \
		--container-save $(SQSH_PATH) \
		--container-mounts $(SOURCE_DIR):/code/tensorrt_llm --container-workdir /code/tensorrt_llm/docker/common \
		--container-mount-home --container-remap-root \
		--container-env TRT_VER \
		--container-env CUDA_VER \
		--container-env CUDNN_VER \
		--container-env NCCL_VER \
		--container-env CUBLAS_VER \
		./install.sh --all

endif