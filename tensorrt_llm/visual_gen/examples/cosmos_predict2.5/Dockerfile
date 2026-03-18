# syntax=docker/dockerfile:1.4

# Combined environment for cosmos-predict2.5, cosmos-transfer2.5, and visual_gen.
# Uses the NVIDIA TensorRT-LLM release container as the base, which already ships
# TensorRT-LLM, PyTorch, and CUDA.

# Build:
#   docker build -f examples/cosmos_predict2.5/Dockerfile -t cosmos-env .
#
# Run:
#   docker run --rm -it --runtime=nvidia --ipc=host \
#     -v /your/workspace:/workspace \
#     -v /your/hf-cache:/root/.cache/huggingface \
#     -e HF_TOKEN="$HF_TOKEN" \
#     cosmos-env
#
# LFS objects (sample assets) are not pulled during the build to keep the image
# lean.  Pull them at runtime if needed:
#   git -C /opt/cosmos-predict2.5 lfs pull
#   git -C /opt/cosmos-transfer2.5 lfs pull

ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc4

FROM ${BASE_IMAGE}

# Install packages
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ffmpeg \
        git \
        git-lfs \
        libx11-dev \
        tree \
        wget

# Ensure uv is available (the nvcr pytorch container includes it, but pin the
# same version used by the upstream stable Dockerfiles for reproducibility)
COPY --from=ghcr.io/astral-sh/uv:0.8.12 /uv /uvx /usr/local/bin/

# Install just: https://just.systems/man/en/pre-built-binaries.html
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin --tag 1.42.4

RUN git lfs install

# Clone cosmos-predict2.5.
# GIT_LFS_SKIP_SMUDGE=1 avoids downloading LFS objects at clone time.
ARG COSMOS_PREDICT_REF=07563da955076ac5fac18617f7d5751e92cf0a04
RUN GIT_LFS_SKIP_SMUDGE=1 git clone \
        https://github.com/nvidia-cosmos/cosmos-predict2.5.git \
        /opt/cosmos-predict2.5 && \
    git -C /opt/cosmos-predict2.5 checkout ${COSMOS_PREDICT_REF}

# Clone cosmos-transfer2.5.
ARG COSMOS_TRANSFER_REF=3f38bef7c5baaa2cffc24af08137cfbda7482b82
RUN GIT_LFS_SKIP_SMUDGE=1 git clone \
        https://github.com/nvidia-cosmos/cosmos-transfer2.5.git \
        /opt/cosmos-transfer2.5 && \
    git -C /opt/cosmos-transfer2.5 checkout ${COSMOS_TRANSFER_REF}

# Install cosmos-predict2.5 dependencies with full dep resolution so all
# transitive deps are pulled in automatically. Use a constraints file to pin
# transformers==4.56.0: cosmos requires >=4.51.3 which pip would resolve to
# >=4.57, breaking tensorrt_llm 1.2.0rc4.
COPY examples/cosmos_predict2.5/requirements.txt /tmp/vg-requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    echo 'transformers==4.56.0' > /tmp/constraints.txt && \
    cd /opt/cosmos-predict2.5 && pip install --no-deps \
        -r docker/nightly-requirements.txt \
        -r /tmp/vg-requirements.txt \
        -c /tmp/constraints.txt

RUN pip install --no-deps /opt/cosmos-transfer2.5/packages/* && \
    pip install --no-deps -e /opt/cosmos-transfer2.5

# Install cosmos-predict2.5 local packages and the project itself last so that
# shared packages (cosmos-cuda, cosmos-oss) use the pinned predict2.5 versions.
RUN pip install --no-deps /opt/cosmos-predict2.5/packages/* && \
    pip install --no-deps -e /opt/cosmos-predict2.5

# ── visual_gen ────────────────────────────────────────────────────────────────
# Sparse-clone only the tensorrt_llm/visual_gen subdirectory from the
# feat/visual_gen branch of TensorRT-LLM.
ARG VISUAL_GEN_REF=feat/visual_gen
RUN git clone --no-checkout --depth 1 --branch ${VISUAL_GEN_REF} \
        https://github.com/NVIDIA/TensorRT-LLM.git /tmp/trt-llm && \
    git -C /tmp/trt-llm sparse-checkout set tensorrt_llm/visual_gen && \
    git -C /tmp/trt-llm checkout && \
    mv /tmp/trt-llm/tensorrt_llm/visual_gen /opt/visual_gen && \
    rm -rf /tmp/trt-llm

# Install visual_gen in editable mode.
# uv is used here because pip 25.x's setup.py develop spawns /usr/bin/python
# internally, which lacks the container's packages; uv avoids this issue.
RUN uv pip install --system --break-system-packages --no-build-isolation \
        --prerelease=allow --no-deps \
        -e /opt/visual_gen

# ── SageAttention ─────────────────────────────────────────────────────────────
RUN pip install --no-deps --no-build-isolation sageattention

# All project roots on PYTHONPATH so their modules are importable.
ENV PYTHONPATH="/opt/visual_gen:/opt/cosmos-transfer2.5:/opt/cosmos-predict2.5"

# HuggingFace checkpoint cache — mount a host directory here for persistence:
#   docker run ... -v /your/hf-cache:/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /workspace

CMD ["/bin/bash"]
