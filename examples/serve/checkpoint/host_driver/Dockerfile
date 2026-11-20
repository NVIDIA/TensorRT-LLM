# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The named Snapshot context is pinned by deployment.py.
FROM golang:1.27.3@sha256:6928d94d1142af58816fed5a62338d605a266ccd53a0deaab6b1096f19059261 AS build
COPY --from=snapshot /api /src/api
COPY --from=snapshot /agent /src/agent
COPY main.go main_test.go deadline.go deadline_test.go /src/agent/cmd/trtllm-host/
COPY health/main.go /src/agent/cmd/trtllm-broker-health/main.go
WORKDIR /src/agent
RUN go test ./cmd/trtllm-host
RUN CGO_ENABLED=0 go build -trimpath -o /trtllm-snapshot-host ./cmd/trtllm-host
RUN CGO_ENABLED=0 go build -trimpath -o /trtllm-broker-health ./cmd/trtllm-broker-health

FROM ghcr.io/ai-dynamo/snapshot/agent@sha256:50a3058893c543dc0e48bfac89dc74a07eb26ff91ad3634c5fef25f37ab9f8c9
COPY --from=build /trtllm-snapshot-host /usr/local/bin/trtllm-snapshot-host
COPY --from=build /trtllm-broker-health /usr/local/bin/trtllm-broker-health
LABEL com.nvidia.trtllm.snapshot.revision="4b1300a82c6676df020c201fa884cc602251e12e"
LABEL com.nvidia.trtllm.snapshot.agent.digest="sha256:50a3058893c543dc0e48bfac89dc74a07eb26ff91ad3634c5fef25f37ab9f8c9"
ENTRYPOINT ["/usr/local/bin/trtllm-snapshot-host"]
