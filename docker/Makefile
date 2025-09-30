# Default base image for the docker build as defined in Dockerfile.multi
BASE_IMAGE         ?= $(shell grep '^ARG BASE_IMAGE=' Dockerfile.multi | grep -o '=.*' | tr -d '="')
BASE_TAG           ?= $(shell grep '^ARG BASE_TAG=' Dockerfile.multi | grep -o '=.*' | tr -d '="')
TRITON_IMAGE       ?= $(shell grep '^ARG TRITON_IMAGE=' Dockerfile.multi | grep -o '=.*' | tr -d '="')
TRITON_BASE_TAG    ?= $(shell grep '^ARG TRITON_BASE_TAG=' Dockerfile.multi | grep -o '=.*' | tr -d '="')
# Name of the new image
IMAGE_NAME         ?= tensorrt_llm
IMAGE_TAG          ?= latest

# Used to share .cache when LOCAL_USER=1. Possibility of override is
# helpful, e.g., for use with Docker rootless mode.
HOME_DIR           ?= $(HOME)

# Local user information
USER_ID            ?= $(shell id --user)
USER_NAME          ?= $(shell id --user --name)
GROUP_ID           ?= $(shell id --group)
GROUP_NAME         ?= $(shell id --group --name)

# Try to detect Docker rootless mode
IS_ROOTLESS        ?= $(shell ./detect_rootless.sh)

# Set this to 1 to add the current user to the docker image and run the container with the user
LOCAL_USER         ?= 0
ifeq ($(LOCAL_USER),1)
IMAGE_TAG_SUFFIX   ?= -$(USER_NAME)
endif

# Set this to 1 to use the image from Jenkins as the image for the `devel` stage in the build phase
JENKINS_DEVEL      ?= 0

# Default stage of the docker multi-stage build
STAGE              ?=
# Set this to define a custom image name and tag
IMAGE_WITH_TAG     ?= $(IMAGE_NAME)$(if $(STAGE),/$(STAGE)):$(IMAGE_TAG)
PUSH_TO_STAGING    ?= 1
DOCKER_BUILD_OPTS  ?= --pull --load
DOCKER_BUILD_ARGS  ?=
DOCKER_PROGRESS    ?= auto
PLATFORM           ?= $(shell uname -m | grep -q 'aarch64' && echo "arm64" || echo "amd64")
CUDA_ARCHS         ?= $(if $(filter arm64,$(PLATFORM)),'90-real;100-real;120-real',)
BUILD_WHEEL_OPTS   ?=
BUILD_WHEEL_ARGS   ?= $(shell grep '^ARG BUILD_WHEEL_ARGS=' Dockerfile.multi | grep -o '=.*' | tr -d '="')$(if $(CUDA_ARCHS), --cuda_architectures $(CUDA_ARCHS))$(if $(BUILD_WHEEL_OPTS), $(BUILD_WHEEL_OPTS))
BUILD_WHEEL_SCRIPT ?=
TORCH_INSTALL_TYPE ?= skip
CUDA_VERSION       ?=
CUDNN_VERSION      ?=
NCCL_VERSION       ?=
CUBLAS_VERSION     ?=
TRT_VERSION        ?=
GIT_COMMIT         ?= $(shell git rev-parse HEAD)
TRT_LLM_VERSION    ?= $(shell grep '^__version__' ../tensorrt_llm/version.py | grep -o '=.*' | tr -d '= "')
GITHUB_MIRROR      ?=
PYTHON_VERSION     ?=
NGC_STAGING_REPO   ?= nvcr.io/nvstaging/tensorrt-llm
NGC_REPO           ?= nvcr.io/nvidia/tensorrt-llm
NGC_USE_STAGING    ?= 0
NGC_AUTO_REPO      ?= $(if $(filter 1,$(NGC_USE_STAGING)),$(NGC_STAGING_REPO),$(NGC_REPO))

define add_local_user
	docker build \
		--progress $(DOCKER_PROGRESS) \
		--build-arg BASE_IMAGE_WITH_TAG=$(1) \
		--build-arg USER_ID=$(USER_ID) \
		--build-arg USER_NAME=$(USER_NAME) \
		--build-arg GROUP_ID=$(GROUP_ID) \
		--build-arg GROUP_NAME=$(GROUP_NAME) \
		--file Dockerfile.user \
		--tag $(1)$(IMAGE_TAG_SUFFIX) \
		..
endef

# Rewrite `/tensorrt-llm:` in image tag with `/tensorrt-llm-staging:` to avoid directly overwriting
define rewrite_tag
$(shell echo $(IMAGE_WITH_TAG) | sed "s/\/tensorrt-llm:/\/tensorrt-llm-staging:/g")
endef

%_build: DEVEL_IMAGE = $(if $(findstring 1,$(JENKINS_DEVEL)),$(shell . ../jenkins/current_image_tags.properties && echo $$LLM_DOCKER_IMAGE))
%_build:
	@echo "Building docker image: $(IMAGE_WITH_TAG)"
	docker buildx build $(DOCKER_BUILD_OPTS) $(DOCKER_BUILD_ARGS) \
		--progress $(DOCKER_PROGRESS) \
		$(if $(BASE_IMAGE), --build-arg BASE_IMAGE=$(BASE_IMAGE)) \
		$(if $(BASE_TAG), --build-arg BASE_TAG=$(BASE_TAG)) \
		$(if $(TRITON_IMAGE), --build-arg TRITON_IMAGE=$(TRITON_IMAGE)) \
		$(if $(TRITON_BASE_TAG), --build-arg TRITON_BASE_TAG=$(TRITON_BASE_TAG)) \
		$(if $(BUILD_WHEEL_ARGS), --build-arg BUILD_WHEEL_ARGS="$(BUILD_WHEEL_ARGS)") \
		$(if $(BUILD_WHEEL_SCRIPT), --build-arg BUILD_WHEEL_SCRIPT="$(BUILD_WHEEL_SCRIPT)") \
		$(if $(TORCH_INSTALL_TYPE), --build-arg TORCH_INSTALL_TYPE="$(TORCH_INSTALL_TYPE)") \
		$(if $(CUDA_VERSION), --build-arg CUDA_VER="$(CUDA_VERSION)") \
		$(if $(CUDNN_VERSION), --build-arg CUDNN_VER="$(CUDNN_VERSION)") \
		$(if $(NCCL_VERSION), --build-arg NCCL_VER="$(NCCL_VERSION)") \
		$(if $(CUBLAS_VERSION), --build-arg CUBLAS_VER="$(CUBLAS_VERSION)") \
		$(if $(TRT_VERSION), --build-arg TRT_VER="$(TRT_VERSION)") \
		$(if $(TRT_LLM_VERSION), --build-arg TRT_LLM_VER="$(TRT_LLM_VERSION)") \
		$(if $(DEVEL_IMAGE), --build-arg DEVEL_IMAGE="$(DEVEL_IMAGE)") \
		$(if $(GIT_COMMIT), --build-arg GIT_COMMIT="$(GIT_COMMIT)") \
		$(if $(GITHUB_MIRROR), --build-arg GITHUB_MIRROR="$(GITHUB_MIRROR)") \
		$(if $(PYTHON_VERSION), --build-arg PYTHON_VERSION="$(PYTHON_VERSION)") \
		$(if $(STAGE), --target $(STAGE)) \
		--file Dockerfile.multi \
		--tag $(IMAGE_WITH_TAG) \
		..

%_user:
	$(call add_local_user,$(IMAGE_WITH_TAG))

%_push: %_build
	@if [ $(PUSH_TO_STAGING) = 0 ]; then \
		echo "Pushing docker image: $(IMAGE_WITH_TAG)"; \
		docker push $(IMAGE_WITH_TAG)$(IMAGE_TAG_SUFFIX); \
	fi
	@if [ $(PUSH_TO_STAGING) = 1 ]; then \
		echo "Rewriting docker tag: $(IMAGE_WITH_TAG) to $(call rewrite_tag)"; \
		docker tag $(IMAGE_WITH_TAG)$(IMAGE_TAG_SUFFIX) $(call rewrite_tag)$(IMAGE_TAG_SUFFIX); \
		echo "Pushing docker image: $(call rewrite_tag)"; \
		docker push $(call rewrite_tag)$(IMAGE_TAG_SUFFIX); \
	fi

%_pull:
	@echo "Pulling docker image: $(IMAGE_WITH_TAG)"
	docker pull $(IMAGE_WITH_TAG)

DOCKER_RUN_OPTS ?= --rm -it --ipc=host --ulimit stack=67108864 $(if $(filter 0,$(IS_ROOTLESS)),--ulimit memlock=-1)
DOCKER_RUN_ARGS   ?=
# Check if NVIDIA_VISIBLE_DEVICES is set and not empty
NVIDIA_VISIBLE_DEVICES_VAL = $(shell echo $$NVIDIA_VISIBLE_DEVICES)
ifeq ($(NVIDIA_VISIBLE_DEVICES_VAL),)
  # If empty or not set, use all GPUs
  GPU_OPTS ?= --gpus=all
else
  # If set, use the specified devices
  GPU_OPTS ?= --gpus='"device=$(NVIDIA_VISIBLE_DEVICES_VAL)"'
endif
SOURCE_DIR        ?= $(shell readlink -f ..)
CODE_DIR          ?= /code/tensorrt_llm
EXTRA_VOLUMES     ?=
CCACHE_DIR        ?= $(CODE_DIR)/cpp/.ccache
CONAN_DIR         ?= $(CODE_DIR)/cpp/.conan
USER_CACHE_DIR    ?= $(shell readlink -f "${HOME_DIR}/.cache")
RUN_CMD           ?=
CONTAINER_NAME    ?= tensorrt_llm
WORK_DIR          ?= $(CODE_DIR)
DOCKER_PULL       ?= 0

%_run:
ifeq ($(IS_ROOTLESS),1)
	@echo "Assuming Docker rootless mode."
endif
ifeq ($(DOCKER_PULL),1)
	@$(MAKE) --no-print-directory $*_pull
endif
ifeq ($(LOCAL_USER),1)
	$(call add_local_user,$(IMAGE_WITH_TAG))
endif
	docker run $(DOCKER_RUN_OPTS) $(DOCKER_RUN_ARGS) \
    		$(GPU_OPTS) \
    		--volume $(SOURCE_DIR):$(CODE_DIR) \
    		$(EXTRA_VOLUMES) \
    		$(if $(and $(filter 1,$(LOCAL_USER)),$(shell [ -w "$(USER_CACHE_DIR)" ] && echo 1)),--volume $(USER_CACHE_DIR):/home/$(USER_NAME)/.cache:rw) \
    		--env "CCACHE_DIR=$(CCACHE_DIR)" \
    		--env "CCACHE_BASEDIR=$(CODE_DIR)" \
    		--env "CONAN_HOME=$(CONAN_DIR)" \
    		--workdir $(WORK_DIR) \
    		--hostname $(shell hostname)-$* \
    		--name $(CONTAINER_NAME)-$*-$(USER_NAME) \
    		--tmpfs /tmp:exec \
    		$(IMAGE_WITH_TAG)$(IMAGE_TAG_SUFFIX) $(RUN_CMD)

devel_%: STAGE = devel
tritondevel_%: STAGE = tritondevel
tritonrelease_%: STAGE = tritonrelease
tritonrelease_%: DEVEL_IMAGE = tritondevel
tritonrelease_run: WORK_DIR = /app/tensorrt_llm

wheel_%: STAGE = wheel
wheel_run: WORK_DIR = /src/tensorrt_llm

release_%: STAGE = release
release_run: WORK_DIR = /app/tensorrt_llm

# For x86_64
jenkins_%: IMAGE_WITH_TAG = $(shell . ../jenkins/current_image_tags.properties && echo $$LLM_DOCKER_IMAGE)
jenkins_%: STAGE = tritondevel

# For aarch64
jenkins-aarch64_%: IMAGE_WITH_TAG = $(shell . ../jenkins/current_image_tags.properties && echo $$LLM_SBSA_DOCKER_IMAGE)
jenkins-aarch64_%: STAGE = tritondevel

# For x86_64
jenkins-rockylinux8_%: PYTHON_VERSION_TAG_ID = $(if $(findstring 3.12,${PYTHON_VERSION}),PY312,$(if $(findstring 3.10,${PYTHON_VERSION}),PY310,$(error Unknown PYTHON_VERSION specified)))
jenkins-rockylinux8_%: IMAGE_WITH_TAG = $(shell . ../jenkins/current_image_tags.properties && echo $$LLM_ROCKYLINUX8_${PYTHON_VERSION_TAG_ID}_DOCKER_IMAGE)
jenkins-rockylinux8_%: STAGE = tritondevel
jenkins-rockylinux8_%: BASE_IMAGE = nvcr.io/nvidia/cuda
jenkins-rockylinux8_%: BASE_TAG = 13.0.0-devel-rockylinux8

rockylinux8_%: STAGE = tritondevel
rockylinux8_%: BASE_IMAGE = nvcr.io/nvidia/cuda
rockylinux8_%: BASE_TAG = 13.0.0-devel-rockylinux8

# For x86_64 and aarch64
ubuntu22_%: STAGE = tritondevel
ubuntu22_%: BASE_IMAGE = nvcr.io/nvidia/cuda
ubuntu22_%: BASE_TAG = 13.0.0-devel-ubuntu22.04

trtllm_%: STAGE = release
trtllm_%: PUSH_TO_STAGING := 0
trtllm_%: DEVEL_IMAGE = $(shell \
    if [ "$(PLATFORM)" = "amd64" ]; then \
        . ../jenkins/current_image_tags.properties && echo $$LLM_DOCKER_IMAGE; \
    elif [ "$(PLATFORM)" = "arm64" ]; then \
        . ../jenkins/current_image_tags.properties && echo $$LLM_SBSA_DOCKER_IMAGE; \
    fi)
trtllm_%: IMAGE_NAME = $(shell . ../jenkins/current_image_tags.properties && echo $$IMAGE_NAME)
trtllm_%: IMAGE_TAG = $(shell git rev-parse --abbrev-ref HEAD | tr '/' '_')-$(PLATFORM)
trtllm_run: WORK_DIR = /app/tensorrt_llm

# This requires a docker installation with multi-platform support
ngc-devel_%: STAGE = devel
ngc-devel_%: DOCKER_BUILD_OPTS = --pull --builder=multi-builder --platform linux/arm64,linux/amd64
ngc-devel_%: IMAGE_NAME = $(NGC_STAGING_REPO)
ngc-devel_%: IMAGE_TAG = $(TRT_LLM_VERSION)

ngc-devel_push: DOCKER_BUILD_ARGS = --push
ngc-devel_push: ngc-devel_build ;

ngc-devel_run:  IMAGE_NAME = $(NGC_AUTO_REPO)
ngc-devel_pull: IMAGE_NAME = $(NGC_AUTO_REPO)

ngc-release_%: STAGE = release
ngc-release_%: DOCKER_BUILD_OPTS = --pull --load --platform linux/$(PLATFORM)
ngc-release_%: DEVEL_IMAGE = $(NGC_STAGING_REPO)/devel:$(TRT_LLM_VERSION)
ngc-release_%: IMAGE_NAME = $(NGC_STAGING_REPO)
ngc-release_%: IMAGE_TAG = $(TRT_LLM_VERSION)-$(PLATFORM)

ngc-release_run:  WORK_DIR = /app/tensorrt_llm
ngc-release_run:  IMAGE_NAME = $(NGC_AUTO_REPO)
ngc-release_run:  IMAGE_TAG = $(TRT_LLM_VERSION)
ngc-release_pull: IMAGE_NAME = $(NGC_AUTO_REPO)
ngc-release_pull: IMAGE_TAG = $(TRT_LLM_VERSION)

ngc-manifest_%: STAGE = release
ngc-manifest_%: IMAGE_NAME = $(NGC_STAGING_REPO)
ngc-manifest_%: IMAGE_TAG = $(TRT_LLM_VERSION)

ngc-manifest_create:
	docker pull $(IMAGE_WITH_TAG)-amd64
	docker pull $(IMAGE_WITH_TAG)-arm64
	docker manifest create $(IMAGE_WITH_TAG) \
  		--amend $(IMAGE_WITH_TAG)-amd64 \
  		--amend $(IMAGE_WITH_TAG)-arm64

ngc-manifest_push: ngc-manifest_create
	docker manifest push $(IMAGE_WITH_TAG)

build: devel_build ;

push: devel_push ;

run: devel_run ;

.PHONY: build push run ngc-manifest_create ngc-manifest_push
