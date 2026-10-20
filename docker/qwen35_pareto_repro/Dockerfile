# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Qwen3.5-397B-A17B NVFP4 GB300 reproduction image: the release image this branch started from + this branch's
# tensorrt_llm/ changes + the four compiled files rebuilt from this branch. Build context = a payload made by
# make_payload.sh:
#   docker/qwen35_pareto_repro/make_payload.sh /tmp/q35_payload
#   docker build -f docker/qwen35_pareto_repro/Dockerfile -t <tag> /tmp/q35_payload
# (aarch64 / GB300 only: the compiled file names in make_payload.sh are the aarch64 ones.)
ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.3.0a0.dev202610151219
FROM ${BASE_IMAGE}
COPY . /tmp/q35_payload
RUN bash /tmp/q35_payload/apply_payload.sh /tmp/q35_payload && rm -rf /tmp/q35_payload
