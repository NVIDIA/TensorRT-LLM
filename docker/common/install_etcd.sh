#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euxo pipefail

# Source revision for etcd v3.7.2. Keep the archive checksum in sync.
ETCD_COMMIT=68c065e562994b89e333e77b039ad066f933c586
ETCD_SHA256=3904e91af8ecadf5b863a0133c60f0fbe6f5bda3f513e26c94f3592608de6f6e
GO_VERSION=1.26.10
X_NET_VERSION=v0.60.0
OTEL_VERSION=v1.45.0
ETCD_INSTALL_DIR=${ETCD_INSTALL_DIR:-/usr/local/bin}
GO_DOWNLOAD_BASE=https://dl.google.com/go
ETCD_SOURCE_URL="https://codeload.github.com/etcd-io/etcd/tar.gz/${ETCD_COMMIT}"

if [[ -n "${GITHUB_MIRROR:-}" ]]; then
    GO_DOWNLOAD_BASE=https://artifactory.nvidia.com/artifactory/hw-hmc-dl-google-remote/go
    export GOPROXY=https://artifactory.nvidia.com/artifactory/proxy-golang-org-remote
    export GOSUMDB="sum.golang.org https://artifactory.pdx.nvidia.com/artifactory/proxygolangorg-go-virtual/sumdb/sum.golang.org"
    ETCD_SOURCE_URL="https://artifactory.pdx.nvidia.com/artifactory/github-releases-remote/etcd-io/etcd/archive/${ETCD_COMMIT}.tar.gz"
fi

case "$(uname -m)" in
    x86_64|amd64)
        ARCH=amd64
        GO_SHA256=ccfe7ae822d2c2f41f4fe7c956199967a4dac22d0ee0d2b2bfdb440aab3b20c5
        ;;
    aarch64|arm64)
        ARCH=arm64
        GO_SHA256=3f4227ce7b206a513e794f1393adb473a07b46e94d18bd15d1c684bbb7319797
        ;;
    *)
        echo "Unsupported etcd build architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

BUILD_DIR=$(mktemp -d)
trap 'rm -rf "${BUILD_DIR}"' EXIT

# Keep scratch files and Go's configuration inside the directory removed on exit.
export TMPDIR="${BUILD_DIR}/tmp"
export GOTMPDIR="${TMPDIR}"
export XDG_CONFIG_HOME="${BUILD_DIR}/config"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}"

curl -fLsS --retry 3 "${GO_DOWNLOAD_BASE}/go${GO_VERSION}.linux-${ARCH}.tar.gz" \
    -o "${BUILD_DIR}/go.tar.gz"
printf '%s  %s\n' "${GO_SHA256}" "${BUILD_DIR}/go.tar.gz" | sha256sum -c -
tar -xzf "${BUILD_DIR}/go.tar.gz" -C "${BUILD_DIR}"

curl -fLsS --retry 3 "${ETCD_SOURCE_URL}" -o "${BUILD_DIR}/etcd.tar.gz"
printf '%s  %s\n' "${ETCD_SHA256}" "${BUILD_DIR}/etcd.tar.gz" | sha256sum -c -
mkdir "${BUILD_DIR}/etcd"
tar -xzf "${BUILD_DIR}/etcd.tar.gz" -C "${BUILD_DIR}/etcd" --strip-components=1

# Isolate the build from installed Go versions, user settings, and module caches.
export GOROOT="${BUILD_DIR}/go"
export PATH="${GOROOT}/bin:${PATH}"
export GOPATH="${BUILD_DIR}/gopath"
export GOMODCACHE="${GOPATH}/pkg/mod"
export GOCACHE="${BUILD_DIR}/go-cache"
export GOENV=off GOTOOLCHAIN=local GOFLAGS=-modcacherw
export GOWORK="${BUILD_DIR}/etcd/go.work"
export CGO_ENABLED=0 GOOS=linux GOARCH="${ARCH}"
go telemetry off

# Workspace module selection applies these dependency versions to all three binaries.
(
    cd "${BUILD_DIR}/etcd/server"
    go get "golang.org/x/net@${X_NET_VERSION}" \
        "go.opentelemetry.io/otel@${OTEL_VERSION}" \
        "go.opentelemetry.io/otel/sdk@${OTEL_VERSION}" \
        "go.opentelemetry.io/otel/exporters/otlp/otlptrace@${OTEL_VERSION}" \
        "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc@${OTEL_VERSION}"
)

cd "${BUILD_DIR}/etcd"
for binary in etcd etcdctl etcdutl; do
    module=${binary}
    if [[ "${binary}" == etcd ]]; then
        module=server
    fi
    go build -trimpath \
        -ldflags "-X go.etcd.io/etcd/api/v3/version.GitSHA=${ETCD_COMMIT}-patched" \
        -o "${BUILD_DIR}/bin/${binary}" "./${module}"
    go version -m "${BUILD_DIR}/bin/${binary}"
done

install -d "${ETCD_INSTALL_DIR}"
install -m 0755 "${BUILD_DIR}/bin/etcd" "${BUILD_DIR}/bin/etcdctl" "${BUILD_DIR}/bin/etcdutl" \
    "${ETCD_INSTALL_DIR}/"
