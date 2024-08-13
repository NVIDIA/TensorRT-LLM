# Default base image for the docker build as defined in Dockerfile.multi
BASE_IMAGE         ?= $(shell grep 'ARG BASE_IMAGE=' Dockerfile.multi | grep -o '=.*' | tr -d '="')
BASE_TAG           ?= $(shell grep 'ARG BASE_TAG=' Dockerfile.multi | grep -o '=.*' | tr -d '="')
# Name of the new image
IMAGE_NAME         ?= tensorrt_llm
IMAGE_TAG          ?= latest

# Local user information
USER_ID            ?= $(shell id --user)
USER_NAME          ?= $(shell id --user --name)
GROUP_ID           ?= $(shell id --group)
GROUP_NAME         ?= $(shell id --group --name)

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
DOCKER_BUILD_OPTS  ?= --pull
DOCKER_BUILD_ARGS  ?=
DOCKER_PROGRESS    ?= auto
CUDA_ARCHS         ?=
BUILD_WHEEL_OPTS   ?=
BUILD_WHEEL_ARGS   ?= $(shell grep 'ARG BUILD_WHEEL_ARGS=' Dockerfile.multi | grep -o '=.*' | tr -d '="')$(if $(CUDA_ARCHS), --cuda_architectures $(CUDA_ARCHS))$(if $(BUILD_WHEEL_OPTS), $(BUILD_WHEEL_OPTS))
TORCH_INSTALL_TYPE ?= skip
CUDA_VERSION       ?=
CUDNN_VERSION      ?=
NCCL_VERSION       ?=
CUBLAS_VERSION     ?=
TRT_VERSION        ?=
GIT_COMMIT         ?= $(shell git rev-parse HEAD)
TRT_LLM_VERSION    ?= $(shell grep '^__version__' ../tensorrt_llm/version.py | grep -o '=.*' | tr -d '= "')

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

%_build: DEVEL_IMAGE = $(if $(findstring 1,$(JENKINS_DEVEL)),$(shell grep 'LLM_DOCKER_IMAGE = ' ../jenkins/L0_MergeRequest.groovy | grep -o '".*"' | tr -d '"'))
%_build:
	@echo "Building docker image: $(IMAGE_WITH_TAG)"
	DOCKER_BUILDKIT=1 docker build $(DOCKER_BUILD_OPTS) $(DOCKER_BUILD_ARGS) \
		--progress $(DOCKER_PROGRESS) \
		$(if $(BASE_IMAGE), --build-arg BASE_IMAGE=$(BASE_IMAGE)) \
		$(if $(BASE_TAG), --build-arg BASE_TAG=$(BASE_TAG)) \
		$(if $(BUILD_WHEEL_ARGS), --build-arg BUILD_WHEEL_ARGS="$(BUILD_WHEEL_ARGS)") \
		$(if $(TORCH_INSTALL_TYPE), --build-arg TORCH_INSTALL_TYPE="$(TORCH_INSTALL_TYPE)") \
		$(if $(CUDA_VERSION), --build-arg CUDA_VER="$(CUDA_VERSION)") \
		$(if $(CUDNN_VERSION), --build-arg CUDNN_VER="$(CUDNN_VERSION)") \
		$(if $(NCCL_VERSION), --build-arg NCCL_VER="$(NCCL_VERSION)") \
		$(if $(CUBLAS_VERSION), --build-arg CUBLAS_VER="$(CUBLAS_VERSION)") \
		$(if $(TRT_VERSION), --build-arg TRT_VER="$(TRT_VERSION)") \
		$(if $(TRT_LLM_VERSION), --build-arg TRT_LLM_VER="$(TRT_LLM_VERSION)") \
		$(if $(DEVEL_IMAGE), --build-arg DEVEL_IMAGE="$(DEVEL_IMAGE)") \
		$(if $(GIT_COMMIT), --build-arg GIT_COMMIT="$(GIT_COMMIT)") \
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

DOCKER_RUN_OPTS   ?= --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
DOCKER_RUN_ARGS   ?=
GPU_OPTS          ?= --gpus=all
SOURCE_DIR        ?= $(shell readlink -f ..)
CODE_DIR          ?= /code/tensorrt_llm
CCACHE_DIR        ?= ${CODE_DIR}/cpp/.ccache
RUN_CMD           ?=
CONTAINER_NAME    ?= tensorrt_llm
WORK_DIR          ?= $(CODE_DIR)
DOCKER_PULL       ?= 0

%_run:
ifeq ($(DOCKER_PULL),1)
	@$(MAKE) --no-print-directory $*_pull
endif
ifeq ($(LOCAL_USER),1)
	$(call add_local_user,$(IMAGE_WITH_TAG))
endif
	docker run $(DOCKER_RUN_OPTS) $(DOCKER_RUN_ARGS) \
    		$(GPU_OPTS) \
    		--volume $(SOURCE_DIR):$(CODE_DIR) \
    		--env "CCACHE_DIR=${CCACHE_DIR}" \
    		--env "CCACHE_BASEDIR=${CODE_DIR}" \
    		--workdir $(WORK_DIR) \
    		--hostname $(shell hostname)-$* \
    		--name $(CONTAINER_NAME)-$*-$(USER_NAME) \
			--tmpfs /tmp:exec \
    		$(IMAGE_WITH_TAG)$(IMAGE_TAG_SUFFIX) $(RUN_CMD)

devel_%: STAGE = devel

wheel_%: STAGE = wheel
wheel_run: WORK_DIR = /src/tensorrt_llm

release_%: STAGE = release
release_run: WORK_DIR = /app/tensorrt_llm

# For x86_64
jenkins_%: IMAGE_WITH_TAG = $(shell grep 'LLM_DOCKER_IMAGE = ' ../jenkins/L0_MergeRequest.groovy | grep -o '".*"' | tr -d '"')
jenkins_%: STAGE = devel

# For aarch64
jenkins-aarch64_%: IMAGE_WITH_TAG = $(shell grep 'LLM_SBSA_DOCKER_IMAGE = ' ../jenkins/GH200Builder.groovy | grep -o '".*"' | tr -d '"')
jenkins-aarch64_%: STAGE = devel

# For x86_64
centos7_%: IMAGE_WITH_TAG = $(shell grep 'LLM_CENTOS7_DOCKER_IMAGE = ' ../jenkins/L0_MergeRequest.groovy | grep -o '".*"' | tr -d '"')
centos7_%: STAGE = devel
centos7_%: BASE_IMAGE = nvidia/cuda
# Latest NGC centos7 CUDA image tag is 12.4.0. It also looks like no further update.
# We need to reinstall the new CUDA toolkit when using this base image.
centos7_%: BASE_TAG = 12.4.0-devel-centos7

# For x86_64 and aarch64
ubuntu22_%: STAGE = devel
ubuntu22_%: BASE_IMAGE = nvidia/cuda
ubuntu22_%: BASE_TAG = 12.5.1-devel-ubuntu22.04

trtllm_%: STAGE = release
trtllm_%: PUSH_TO_STAGING := 0
trtllm_%: DEVEL_IMAGE = $(shell grep 'LLM_DOCKER_IMAGE = ' ../jenkins/L0_MergeRequest.groovy | grep -o '".*"' | tr -d '"')
trtllm_%: IMAGE_NAME = $(shell grep 'IMAGE_NAME = ' ../jenkins/BuildDockerImage.groovy | grep -o '".*"' | tr -d '"')
trtllm_%: IMAGE_TAG = $(shell git rev-parse --abbrev-ref HEAD | tr '/' '_')
trtllm_run: WORK_DIR = /app/tensorrt_llm

build: devel_build ;

push: devel_push ;

run: devel_run ;

.PHONY: build push run
