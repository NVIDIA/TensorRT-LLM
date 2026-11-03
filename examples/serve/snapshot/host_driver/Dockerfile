# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Supply a clean pinned Snapshot checkout as the named `snapshot` build context.
FROM golang:1.27.3@sha256:6928d94d1142af58816fed5a62338d605a266ccd53a0deaab6b1096f19059261 AS build
COPY --from=snapshot /api /src/api
COPY --from=snapshot /agent /src/agent
COPY *.go /src/agent/cmd/trtllm-host/
WORKDIR /src/agent
RUN go test ./cmd/trtllm-host
RUN CGO_ENABLED=0 go build -trimpath -o /trtllm-snapshot-host ./cmd/trtllm-host

FROM ghcr.io/ai-dynamo/snapshot/agent@sha256:50a3058893c543dc0e48bfac89dc74a07eb26ff91ad3634c5fef25f37ab9f8c9
COPY --from=build /trtllm-snapshot-host /usr/local/bin/trtllm-snapshot-host
LABEL com.nvidia.trtllm.snapshot.revision="4b1300a82c6676df020c201fa884cc602251e12e"
LABEL com.nvidia.trtllm.snapshot.agent.digest="sha256:50a3058893c543dc0e48bfac89dc74a07eb26ff91ad3634c5fef25f37ab9f8c9"
ENTRYPOINT ["/usr/local/bin/trtllm-snapshot-host"]
