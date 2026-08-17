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

readonly DIND_SYS_CLASS_NET="${DIND_SYS_CLASS_NET:-/sys/class/net}"
readonly DIND_DOCKER_READY_TIMEOUT_SECONDS="${DIND_DOCKER_READY_TIMEOUT_SECONDS:-60}"
readonly DIND_DOCKER_INFO_TIMEOUT_SECONDS="${DIND_DOCKER_INFO_TIMEOUT_SECONDS:-5}"

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

read_interface_mtu()
{
    interface_name="$1"
    mtu_path="${DIND_SYS_CLASS_NET}/${interface_name}/mtu"

    if [ ! -r "${mtu_path}" ]; then
        echo "Unable to read MTU for interface '${interface_name}'" >&2
        return 1
    fi

    interface_mtu="$(cat "${mtu_path}")"
    case "${interface_mtu}" in
        ''|*[!0-9]*)
            echo "Invalid MTU '${interface_mtu}' for interface '${interface_name}'" >&2
            return 1
            ;;
    esac

    printf '%s\n' "${interface_mtu}"
}

require_positive_integer()
{
    value="$1"
    value_name="$2"
    case "${value}" in
        ''|0|*[!0-9]*)
            fail "${value_name} must be a positive integer, got '${value}'"
            ;;
    esac
}

start_docker()
{
    network_interface="$(detect_network_interface)"
    if [ -z "${network_interface}" ]; then
        fail "Unable to detect the pod network interface"
    fi

    pod_mtu="$(read_interface_mtu "${network_interface}")"
    dind_entrypoint="$(command -v dockerd-entrypoint.sh || true)"
    if [ -z "${dind_entrypoint}" ]; then
        fail "Unable to find dockerd-entrypoint.sh"
    fi

    echo "Starting Docker with MTU ${pod_mtu} from interface ${network_interface}"
    exec "${dind_entrypoint}" --mtu="${pod_mtu}" "$@"
}

wait_for_docker()
{
    require_positive_integer "${DIND_DOCKER_READY_TIMEOUT_SECONDS}" \
        "DIND_DOCKER_READY_TIMEOUT_SECONDS"
    require_positive_integer "${DIND_DOCKER_INFO_TIMEOUT_SECONDS}" \
        "DIND_DOCKER_INFO_TIMEOUT_SECONDS"

    if ! command -v timeout >/dev/null 2>&1; then
        fail "Unable to find timeout for bounded Docker readiness probes"
    fi

    if ! timeout "${DIND_DOCKER_READY_TIMEOUT_SECONDS}s" sh -c '
        probe_timeout_seconds="$1"
        until timeout "${probe_timeout_seconds}s" docker info >/dev/null 2>&1; do
            sleep 1
        done
    ' dind-mtu-wait "${DIND_DOCKER_INFO_TIMEOUT_SECONDS}"; then
        fail "Docker daemon did not become ready within ${DIND_DOCKER_READY_TIMEOUT_SECONDS} seconds"
    fi
}

validate_docker_mtu()
{
    wait_for_docker

    network_interface="$(detect_network_interface)"
    if [ -z "${network_interface}" ]; then
        fail "Unable to detect the pod network interface"
    fi

    pod_mtu="$(read_interface_mtu "${network_interface}")"
    docker_mtu="$(read_interface_mtu docker0)"

    if [ "${pod_mtu}" != "${docker_mtu}" ]; then
        fail "DIND MTU mismatch: ${network_interface}=${pod_mtu}, docker0=${docker_mtu}"
    fi

    echo "Verified DIND network MTU: ${network_interface}=${pod_mtu}, docker0=${docker_mtu}"
}

usage()
{
    echo "Usage: dind-mtu {start|validate}"
}

mode="${1:-}"
if [ "$#" -gt 0 ]; then
    shift
fi

case "${mode}" in
    start)
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
