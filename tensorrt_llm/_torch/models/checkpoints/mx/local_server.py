# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Local Docker-backed ModelExpress server bootstrap."""

import json
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Optional

from tensorrt_llm.logger import logger

DEFAULT_MX_LOCAL_SERVER_HOST = "127.0.0.1"
DEFAULT_MX_LOCAL_SERVER_IMAGE = "nvcr.io/nvidia/ai-dynamo/modelexpress-server:0.5.0"
DEFAULT_MX_LOCAL_SERVER_PORT = 8001
DEFAULT_MX_LOCAL_SERVER_STARTUP_TIMEOUT_S = 30
DEFAULT_MX_LOCAL_REDIS_IMAGE = "redis:8-alpine"

_DOCKER_COMMAND_TIMEOUT_S = 600
_MX_SERVER_CONTAINER_PORT = 8001
_REDIS_CONTAINER_PORT = 6379
_TRTLLM_MX_LABEL = "com.nvidia.trtllm.mx-local-server=true"
_TRTLLM_MX_ROLE_LABEL = "com.nvidia.trtllm.mx-local-server.role"
_TRTLLM_MX_PORT_LABEL = "com.nvidia.trtllm.mx-local-server.port"


@dataclass(frozen=True)
class _LocalMXServerSpec:
    host: str
    port: int
    server_image: str
    redis_image: str
    startup_timeout_s: int

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def network_name(self) -> str:
        return f"trtllm-mx-{self.port}"

    @property
    def redis_name(self) -> str:
        return f"{self.network_name}-redis"

    @property
    def server_name(self) -> str:
        return f"{self.network_name}-server"


def ensure_local_mx_server(
    *,
    host: str = DEFAULT_MX_LOCAL_SERVER_HOST,
    port: int = DEFAULT_MX_LOCAL_SERVER_PORT,
    server_image: str = DEFAULT_MX_LOCAL_SERVER_IMAGE,
    redis_image: str = DEFAULT_MX_LOCAL_REDIS_IMAGE,
    startup_timeout_s: int = DEFAULT_MX_LOCAL_SERVER_STARTUP_TIMEOUT_S,
) -> str:
    """Ensure a local MX server and Redis are running.

    Args:
        host: Host interface used for the MX server URL.
        port: Host TCP port to map to the MX server container port.
        server_image: Docker image for the ModelExpress server.
        redis_image: Docker image for Redis.
        startup_timeout_s: Seconds to wait for the MX server port.

    Returns:
        The local MX server URL.

    Raises:
        RuntimeError: If Docker is unavailable or the local server cannot be
            started.
    """
    spec = _LocalMXServerSpec(
        host=host,
        port=port,
        server_image=server_image,
        redis_image=redis_image,
        startup_timeout_s=startup_timeout_s,
    )

    if _port_open(spec.host, spec.port):
        if _server_container_matches(spec):
            logger.info("Using local MX server already listening at %s", spec.url)
            return spec.url
        raise RuntimeError(
            f"Port {spec.host}:{spec.port} is already in use, but the "
            f"expected TRT-LLM-managed MX server container "
            f"'{spec.server_name}' is not running with the requested local "
            "server configuration. Stop the process using that port, choose "
            "another mx_config.local_server.port, or configure the existing MX "
            "server explicitly with mx_config.server_url or MODEL_EXPRESS_URL."
        )

    _verify_docker_available()
    _ensure_network(spec.network_name)
    _ensure_redis_container(spec)
    _ensure_mx_server_container(spec)
    _wait_for_server(spec)
    logger.info("Started local MX server at %s", spec.url)
    return spec.url


def _verify_docker_available() -> None:
    _docker(["version", "--format", "{{.Server.Version}}"], timeout_s=30)


def _ensure_network(network_name: str) -> None:
    result = _docker(["network", "inspect", network_name], check=False)
    if result.returncode == 0:
        return
    result = _docker(["network", "create", network_name], check=False)
    if result.returncode == 0 or _is_already_exists(result):
        return
    _raise_docker_error(["docker", "network", "create", network_name], result)


def _ensure_redis_container(spec: _LocalMXServerSpec) -> None:
    running = _container_running(spec.redis_name)
    if running is True:
        _validate_existing_container(
            spec.redis_name,
            expected_image=spec.redis_image,
            expected_env=[],
            expected_network=spec.network_name,
        )
        return
    if running is False:
        _validate_existing_container(
            spec.redis_name,
            expected_image=spec.redis_image,
            expected_env=[],
            expected_network=spec.network_name,
        )
        _start_container(spec.redis_name)
        return

    cmd = [
        "run",
        "-d",
        "--name",
        spec.redis_name,
        "--label",
        _TRTLLM_MX_LABEL,
        "--label",
        f"{_TRTLLM_MX_ROLE_LABEL}=redis",
        "--label",
        f"{_TRTLLM_MX_PORT_LABEL}={spec.port}",
        "--network",
        spec.network_name,
        spec.redis_image,
    ]
    result = _docker(cmd, check=False)
    if result.returncode == 0:
        return
    if _is_already_exists(result):
        _validate_existing_container(
            spec.redis_name,
            expected_image=spec.redis_image,
            expected_env=[],
            expected_network=spec.network_name,
        )
        _start_container(spec.redis_name)
        return
    _raise_docker_error(["docker", *cmd], result)


def _ensure_mx_server_container(spec: _LocalMXServerSpec) -> None:
    running = _container_running(spec.server_name)
    if running is True:
        _validate_existing_container(
            spec.server_name,
            expected_image=spec.server_image,
            expected_env=_server_env(spec),
            expected_network=spec.network_name,
            expected_port=(spec.host, spec.port),
        )
        return
    if running is False:
        _validate_existing_container(
            spec.server_name,
            expected_image=spec.server_image,
            expected_env=_server_env(spec),
            expected_network=spec.network_name,
            expected_port=(spec.host, spec.port),
        )
        _start_container(spec.server_name)
        return

    cmd = [
        "run",
        "-d",
        "--name",
        spec.server_name,
        "--label",
        _TRTLLM_MX_LABEL,
        "--label",
        f"{_TRTLLM_MX_ROLE_LABEL}=server",
        "--label",
        f"{_TRTLLM_MX_PORT_LABEL}={spec.port}",
        "--network",
        spec.network_name,
        "-p",
        f"{spec.host}:{spec.port}:{_MX_SERVER_CONTAINER_PORT}",
    ]
    for env in _server_env(spec):
        cmd.extend(["-e", env])
    cmd.append(spec.server_image)
    result = _docker(cmd, check=False)
    if result.returncode == 0:
        return
    if _is_already_exists(result):
        _validate_existing_container(
            spec.server_name,
            expected_image=spec.server_image,
            expected_env=_server_env(spec),
            expected_network=spec.network_name,
            expected_port=(spec.host, spec.port),
        )
        _start_container(spec.server_name)
        return
    _raise_docker_error(["docker", *cmd], result)


def _start_container(container_name: str) -> None:
    result = _docker(["start", container_name], check=False)
    if result.returncode == 0 or "already running" in _result_details(result).lower():
        return
    if _container_running(container_name) is True:
        return
    _raise_docker_error(["docker", "start", container_name], result)


def _container_running(container_name: str) -> bool | None:
    result = _docker(
        ["container", "inspect", "--format", "{{.State.Running}}", container_name],
        check=False,
        timeout_s=30,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip().lower() == "true"


def _server_container_matches(spec: _LocalMXServerSpec) -> bool:
    if _container_running(spec.server_name) is not True:
        return False
    _validate_existing_container(
        spec.server_name,
        expected_image=spec.server_image,
        expected_env=_server_env(spec),
        expected_network=spec.network_name,
        expected_port=(spec.host, spec.port),
    )
    return True


def _server_env(spec: _LocalMXServerSpec) -> list[str]:
    return [
        f"MODEL_EXPRESS_SERVER_PORT={_MX_SERVER_CONTAINER_PORT}",
        "MODEL_EXPRESS_LOG_LEVEL=info",
        "MX_METADATA_BACKEND=redis",
        f"REDIS_URL=redis://{spec.redis_name}:{_REDIS_CONTAINER_PORT}",
    ]


def _validate_existing_container(
    container_name: str,
    *,
    expected_image: str,
    expected_env: list[str],
    expected_network: Optional[str] = None,
    expected_port: Optional[tuple[str, int]] = None,
) -> None:
    container = _inspect_container(container_name)
    if container is None:
        raise RuntimeError(
            f"Docker container '{container_name}' disappeared while starting "
            "the local MX server. Retry the launch."
        )

    config = container.get("Config") or {}
    actual_image = config.get("Image")
    actual_env = set(config.get("Env") or [])

    mismatches = []
    if actual_image != expected_image:
        mismatches.append(f"image is {actual_image!r}, expected {expected_image!r}")
    missing_env = [env for env in expected_env if env not in actual_env]
    if missing_env:
        mismatches.append(f"missing env {missing_env!r}")
    if expected_network is not None:
        if not _container_has_network(container, expected_network):
            mismatches.append(f"not attached to Docker network {expected_network!r}")
    if expected_port is not None:
        expected_host, expected_host_port = expected_port
        if not _container_publishes_port(container, expected_host, expected_host_port):
            mismatches.append(f"does not publish {expected_host}:{expected_host_port}")

    if mismatches:
        raise RuntimeError(
            f"Existing Docker container '{container_name}' does not match "
            f"the requested local MX server configuration: "
            f"{'; '.join(mismatches)}. Remove the container or choose another "
            "mx_config.local_server.port."
        )


def _inspect_container(container_name: str) -> Optional[dict[str, Any]]:
    result = _docker(["container", "inspect", container_name], check=False, timeout_s=30)
    if result.returncode != 0:
        return None
    try:
        containers = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Docker returned invalid inspect data for '{container_name}'.") from e
    if not containers:
        return None
    return containers[0]


def _container_has_network(container: dict[str, Any], expected_network: str) -> bool:
    networks = (container.get("NetworkSettings") or {}).get("Networks") or {}
    if networks:
        return expected_network in networks

    host_config = container.get("HostConfig") or {}
    return host_config.get("NetworkMode") == expected_network


def _container_publishes_port(
    container: dict[str, Any],
    expected_host: str,
    expected_host_port: int,
) -> bool:
    port_key = f"{_MX_SERVER_CONTAINER_PORT}/tcp"
    expected_host_port = str(expected_host_port)
    runtime_bindings = _network_port_bindings(container, port_key)
    if runtime_bindings:
        return _has_port_binding(runtime_bindings, expected_host, expected_host_port)

    host_bindings = _host_port_bindings(container, port_key)
    return _has_port_binding(host_bindings, expected_host, expected_host_port)


def _has_port_binding(
    bindings: list[dict[str, str]],
    expected_host: str,
    expected_host_port: str,
) -> bool:
    return any(
        binding.get("HostIp") == expected_host and binding.get("HostPort") == expected_host_port
        for binding in bindings
    )


def _network_port_bindings(container: dict[str, Any], port_key: str) -> list[dict[str, str]]:
    network_ports = (container.get("NetworkSettings") or {}).get("Ports") or {}
    return network_ports.get(port_key) or []


def _host_port_bindings(container: dict[str, Any], port_key: str) -> list[dict[str, str]]:
    host_bindings = (container.get("HostConfig") or {}).get("PortBindings") or {}
    return host_bindings.get(port_key) or []


def _wait_for_server(spec: _LocalMXServerSpec) -> None:
    deadline = time.monotonic() + spec.startup_timeout_s
    while time.monotonic() < deadline:
        if _port_open(spec.host, spec.port):
            return
        time.sleep(0.5)

    logs = _container_logs(spec.server_name)
    raise RuntimeError(
        f"Timed out after {spec.startup_timeout_s}s waiting for local MX "
        f"server at {spec.url}. Last container logs:\n{logs}"
    )


def _port_open(host: str, port: int, timeout_s: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _container_logs(container_name: str) -> str:
    result = _docker(["logs", "--tail", "50", container_name], check=False, timeout_s=30)
    logs = (result.stdout or "") + (result.stderr or "")
    return logs.strip() or "<no logs available>"


def _result_details(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stderr or result.stdout or "").strip()


def _is_already_exists(result: subprocess.CompletedProcess[str]) -> bool:
    details = _result_details(result).lower()
    return "already exists" in details or "already in use" in details


def _raise_docker_error(cmd: list[str], result: subprocess.CompletedProcess[str]) -> None:
    details = _result_details(result)
    raise RuntimeError(f"Docker command failed ({result.returncode}): {' '.join(cmd)}\n{details}")


def _docker(
    args: list[str],
    *,
    check: bool = True,
    timeout_s: int = _DOCKER_COMMAND_TIMEOUT_S,
) -> subprocess.CompletedProcess[str]:
    cmd = ["docker", *args]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "Docker CLI is required to start a local MX server. Install "
            "Docker or set mx_config.local_server.enabled=false and provide "
            "mx_config.server_url for an externally managed MX server."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Docker command timed out after {timeout_s}s: {' '.join(cmd)}") from e

    if check and result.returncode != 0:
        _raise_docker_error(cmd, result)
    return result
