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

# Default stage of the docker multi-stage build
STAGE              ?=
# Set this to define a custom image name and tag
IMAGE_WITH_TAG     ?= $(IMAGE_NAME)$(if $(STAGE),/$(STAGE)):$(IMAGE_TAG)
DOCKER_BUILD_OPTS  ?= --pull
DOCKER_BUILD_ARGS  ?=
DOCKER_PROGRESS    ?= auto
CUDA_ARCHS         ?=
BUILD_WHEEL_ARGS   ?= $(shell grep 'ARG BUILD_WHEEL_ARGS=' Dockerfile.multi | grep -o '=.*' | tr -d '="')$(if $(CUDA_ARCHS), --cuda_architectures $(CUDA_ARCHS))
TORCH_INSTALL_TYPE ?= skip

define add_local_user
	docker build \
		--progress $(DOCKER_BUILD_OPTS) $(DOCKER_BUILD_ARGS) \
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

%_build:
	@echo "Building docker image: $(IMAGE_WITH_TAG)"
	DOCKER_BUILDKIT=1 docker build $(DOCKER_BUILD_OPTS) $(DOCKER_BUILD_ARGS) \
		--progress $(DOCKER_PROGRESS) \
		$(if $(BASE_IMAGE), --build-arg BASE_IMAGE=$(BASE_IMAGE)) \
		$(if $(BASE_TAG), --build-arg BASE_TAG=$(BASE_TAG)) \
		$(if $(BUILD_WHEEL_ARGS), --build-arg BUILD_WHEEL_ARGS="$(BUILD_WHEEL_ARGS)") \
		$(if $(TORCH_INSTALL_TYPE), --build-arg TORCH_INSTALL_TYPE="$(TORCH_INSTALL_TYPE)") \
		$(if $(STAGE), --target $(STAGE)) \
		--file Dockerfile.multi \
		--tag $(IMAGE_WITH_TAG) \
		..

%_user:
	$(call add_local_user,$(IMAGE_WITH_TAG))

%_push: %_build
	@echo "Pushing docker image: $(IMAGE_WITH_TAG)"
	docker push $(IMAGE_WITH_TAG)$(IMAGE_TAG_SUFFIX)

DOCKER_RUN_OPTS   ?= --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
DOCKER_RUN_ARGS   ?=
GPU_OPTS          ?= --gpus=all
SOURCE_DIR        ?= $(shell readlink -f ..)
CODE_DIR          ?= /code/tensorrt_llm
RUN_CMD           ?=
CONTAINER_NAME    ?= tensorrt_llm

%_run:
ifeq ($(LOCAL_USER),1)
	$(call add_local_user,$(IMAGE_WITH_TAG))
endif
	docker run $(DOCKER_RUN_OPTS) $(DOCKER_RUN_ARGS) \
    		$(GPU_OPTS) \
    		--volume $(SOURCE_DIR):$(CODE_DIR) \
    		--workdir $(CODE_DIR) \
    		--hostname $(shell hostname)-$* \
    		--name $(CONTAINER_NAME)-$*-$(USER_NAME) \
			--tmpfs /tmp:exec \
    		$(IMAGE_WITH_TAG)$(IMAGE_TAG_SUFFIX) $(RUN_CMD)

devel_%: STAGE = devel

wheel_%: STAGE = wheel

release_%: STAGE = release

jenkins_%: IMAGE_WITH_TAG = $(shell grep 'LLM_DOCKER_IMAGE = ' ../jenkins/L0_MergeRequest.groovy | grep -o '".*"' | tr -d '"')
jenkins_%: STAGE = devel

centos7_%: IMAGE_WITH_TAG = $(shell grep 'LLM_CENTOS7_DOCKER_IMAGE = ' ../jenkins/L0_MergeRequest.groovy | grep -o '".*"' | tr -d '"')
centos7_%: STAGE = devel
centos7_%: TORCH_INSTALL_TYPE = src_cxx11_abi
centos7_%: BASE_IMAGE = nvidia/cuda
centos7_%: BASE_TAG = 12.2.0-devel-centos7

ubuntu20_%: IMAGE_WITH_TAG = $(shell grep 'LLM_UBUNTU20_DOCKER_IMAGE = ' ../jenkins/L0_MergeRequest.groovy | grep -o '".*"' | tr -d '"')
ubuntu20_%: STAGE = devel
ubuntu20_%: TORCH_INSTALL_TYPE = src_cxx11_abi
ubuntu20_%: BASE_IMAGE = nvidia/cuda
ubuntu20_%: BASE_TAG = 12.2.0-devel-ubuntu20.04

old-cuda_%: IMAGE_WITH_TAG = $(shell grep 'LLM_OLD_CUDA_DOCKER_IMAGE = ' ../jenkins/L0_MergeRequest.groovy | grep -o '".*"' | tr -d '"')
old-cuda_%: BASE_TAG = 23.07-py3
old-cuda_%: STAGE = devel

build: devel_build ;

push: devel_push ;

run: devel_run ;

.PHONY: build push run
