#!/bin/sh
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -eu

DIND_SYS_CLASS_NET="${DIND_SYS_CLASS_NET:-/sys/class/net}"
DIND_DOCKER_READY_TIMEOUT_SECONDS="${DIND_DOCKER_READY_TIMEOUT_SECONDS:-60}"
DIND_DOCKER_INFO_TIMEOUT_SECONDS="${DIND_DOCKER_INFO_TIMEOUT_SECONDS:-5}"

fail()
{
    echo "$*" >&2
    exit 1
}

detect_network_interface()
{
    ip -4 route show default |
        awk '{ for (i = 1; i <= NF; i++) if ($i == "dev") { print $(i + 1); exit } }'
}

interface_mtu()
{
    mtu_path="${DIND_SYS_CLASS_NET}/$1/mtu"

    if [ ! -r "${mtu_path}" ]; then
        echo "Unable to read MTU for interface '$1'" >&2
        return 1
    fi

    mtu="$(cat "${mtu_path}")"
    case "${mtu}" in
        ''|*[!0-9]*)
            echo "Invalid MTU '${mtu}' for interface '$1'" >&2
            return 1
            ;;
    esac

    printf '%s\n' "${mtu}"
}

load_pod_network()
{
    pod_interface="$(detect_network_interface)"
    [ -n "${pod_interface}" ] || fail "Unable to detect the pod network interface"
    pod_mtu="$(interface_mtu "${pod_interface}")"
}

start_docker()
{
    load_pod_network
    dind_entrypoint="$(command -v dockerd-entrypoint.sh || true)"
    [ -n "${dind_entrypoint}" ] || fail "Unable to find dockerd-entrypoint.sh"

    echo "Starting Docker with MTU ${pod_mtu} from interface ${pod_interface}"
    exec "${dind_entrypoint}" --mtu="${pod_mtu}" "$@"
}

wait_for_docker()
{
    if ! timeout "${DIND_DOCKER_READY_TIMEOUT_SECONDS}s" sh -c '
        until timeout "${1}s" docker info >/dev/null 2>&1; do
            sleep 1
        done
    ' dind-mtu-wait "${DIND_DOCKER_INFO_TIMEOUT_SECONDS}"; then
        fail "Docker daemon did not become ready within ${DIND_DOCKER_READY_TIMEOUT_SECONDS} seconds"
    fi
}

validate_docker_mtu()
{
    wait_for_docker
    load_pod_network
    docker_mtu="$(interface_mtu docker0)"

    if [ "${pod_mtu}" != "${docker_mtu}" ]; then
        fail "DIND MTU mismatch: ${pod_interface}=${pod_mtu}, docker0=${docker_mtu}"
    fi

    echo "Verified DIND network MTU: ${pod_interface}=${pod_mtu}, docker0=${docker_mtu}"
}

usage()
{
    echo "Usage: dind-mtu {start|validate}"
}

case "${1:-}" in
    start)
        shift
        start_docker "$@"
        ;;
    validate)
        validate_docker_mtu
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
