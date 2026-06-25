# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the local ModelExpress server bootstrap helper."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.models.checkpoints.mx import local_server


def _completed(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def _server_env(port=8123):
    return [
        "MODEL_EXPRESS_SERVER_PORT=8001",
        "MODEL_EXPRESS_LOG_LEVEL=info",
        "MX_METADATA_BACKEND=redis",
        f"REDIS_URL=redis://trtllm-mx-{port}-redis:6379",
    ]


def _container_config(image, *, env=None, port=None):
    ports = {}
    port_bindings = {}
    if port is not None:
        port_binding = {
            "HostIp": "127.0.0.1",
            "HostPort": str(port),
        }
        ports["8001/tcp"] = [port_binding]
        port_bindings["8001/tcp"] = [port_binding]
    return {
        "Config": {
            "Image": image,
            "Env": env or [],
        },
        "HostConfig": {
            "NetworkMode": "trtllm-mx-8123",
            "PortBindings": port_bindings,
        },
        "NetworkSettings": {
            "Networks": {
                "trtllm-mx-8123": {},
            },
            "Ports": ports,
        },
    }


def _stopped_container_config(image, *, env=None, port=None):
    container = _container_config(image, env=env, port=port)
    container["NetworkSettings"] = {
        "Networks": {},
        "Ports": {},
    }
    return container


def test_existing_managed_local_server_is_reused(monkeypatch):
    monkeypatch.setattr(local_server, "_port_open", lambda host, port: True)
    monkeypatch.setattr(local_server, "_container_running", lambda name: True)
    monkeypatch.setattr(
        local_server,
        "_inspect_container",
        lambda name: _container_config("example/mx:dev", env=_server_env(), port=8123),
    )

    def _fail_if_docker_called(*_args, **_kwargs):
        raise AssertionError("Docker should not be called for an open MX port")

    monkeypatch.setattr(local_server, "_docker", _fail_if_docker_called)

    assert (
        local_server.ensure_local_mx_server(
            port=8123,
            server_image="example/mx:dev",
            redis_image="redis:7-alpine",
        )
        == "http://127.0.0.1:8123"
    )


def test_open_port_without_managed_server_is_rejected(monkeypatch):
    monkeypatch.setattr(local_server, "_port_open", lambda host, port: True)
    monkeypatch.setattr(local_server, "_container_running", lambda name: None)

    with pytest.raises(RuntimeError, match="already in use"):
        local_server.ensure_local_mx_server(port=8123)


def test_starts_redis_and_mx_server_when_no_local_port(monkeypatch):
    port_checks = iter([False, True])
    docker_calls = []

    monkeypatch.setattr(local_server, "_port_open", lambda host, port: next(port_checks))
    monkeypatch.setattr(local_server, "_container_running", lambda name: None)

    def _fake_docker(args, *, check=True, timeout_s=600):
        docker_calls.append(args)
        if args[:2] == ["network", "inspect"]:
            return _completed(returncode=1)
        return _completed()

    monkeypatch.setattr(local_server, "_docker", _fake_docker)

    url = local_server.ensure_local_mx_server(
        port=8123,
        server_image="example/mx:dev",
        redis_image="redis:7-alpine",
        startup_timeout_s=1,
    )

    assert url == "http://127.0.0.1:8123"
    assert ["network", "create", "trtllm-mx-8123"] in docker_calls

    redis_run = next(
        call
        for call in docker_calls
        if call[:2] == ["run", "-d"] and "trtllm-mx-8123-redis" in call
    )
    assert "redis:7-alpine" in redis_run

    server_run = next(
        call
        for call in docker_calls
        if call[:2] == ["run", "-d"] and "trtllm-mx-8123-server" in call
    )
    assert "example/mx:dev" in server_run
    assert "127.0.0.1:8123:8001" in server_run
    assert "MX_METADATA_BACKEND=redis" in server_run
    assert "REDIS_URL=redis://trtllm-mx-8123-redis:6379" in server_run


def test_network_create_race_is_treated_as_success(monkeypatch):
    docker_calls = []

    def _fake_docker(args, *, check=True, timeout_s=600):
        docker_calls.append(args)
        if args[:2] == ["network", "inspect"]:
            return _completed(returncode=1)
        if args[:2] == ["network", "create"]:
            return _completed(returncode=1, stderr="network already exists")
        return _completed()

    monkeypatch.setattr(local_server, "_docker", _fake_docker)

    local_server._ensure_network("trtllm-mx-8123")

    assert ["network", "create", "trtllm-mx-8123"] in docker_calls


def test_container_run_race_starts_existing_containers(monkeypatch):
    docker_calls = []
    spec = local_server._LocalMXServerSpec(
        host="127.0.0.1",
        port=8123,
        server_image="example/mx:dev",
        redis_image="redis:7-alpine",
        startup_timeout_s=1,
    )

    monkeypatch.setattr(local_server, "_container_running", lambda name: None)
    monkeypatch.setattr(
        local_server,
        "_inspect_container",
        lambda name: _container_config(
            "redis:7-alpine" if name.endswith("-redis") else "example/mx:dev",
            env=[] if name.endswith("-redis") else _server_env(),
            port=None if name.endswith("-redis") else 8123,
        ),
    )

    def _fake_docker(args, *, check=True, timeout_s=600):
        docker_calls.append(args)
        if args[:2] == ["run", "-d"]:
            return _completed(returncode=1, stderr="container name already in use")
        return _completed()

    monkeypatch.setattr(local_server, "_docker", _fake_docker)

    local_server._ensure_redis_container(spec)
    local_server._ensure_mx_server_container(spec)

    assert ["start", "trtllm-mx-8123-redis"] in docker_calls
    assert ["start", "trtllm-mx-8123-server"] in docker_calls


def test_stopped_server_container_uses_host_config_for_validation(monkeypatch):
    docker_calls = []
    spec = local_server._LocalMXServerSpec(
        host="127.0.0.1",
        port=8123,
        server_image="example/mx:dev",
        redis_image="redis:7-alpine",
        startup_timeout_s=1,
    )

    monkeypatch.setattr(local_server, "_container_running", lambda name: False)
    monkeypatch.setattr(
        local_server,
        "_inspect_container",
        lambda name: _stopped_container_config("example/mx:dev", env=_server_env(), port=8123),
    )
    monkeypatch.setattr(
        local_server, "_docker", lambda args, **kwargs: docker_calls.append(args) or _completed()
    )

    local_server._ensure_mx_server_container(spec)

    assert ["start", "trtllm-mx-8123-server"] in docker_calls


def test_runtime_network_mismatch_is_rejected(monkeypatch):
    spec = local_server._LocalMXServerSpec(
        host="127.0.0.1",
        port=8123,
        server_image="example/mx:dev",
        redis_image="redis:7-alpine",
        startup_timeout_s=1,
    )
    container = _container_config("example/mx:dev", env=_server_env(), port=8123)
    container["NetworkSettings"]["Networks"] = {"wrong-network": {}}

    monkeypatch.setattr(local_server, "_container_running", lambda name: True)
    monkeypatch.setattr(local_server, "_inspect_container", lambda name: container)

    with pytest.raises(RuntimeError, match="not attached"):
        local_server._ensure_mx_server_container(spec)


def test_existing_server_container_config_mismatch_is_rejected(monkeypatch):
    spec = local_server._LocalMXServerSpec(
        host="127.0.0.1",
        port=8123,
        server_image="example/mx:dev",
        redis_image="redis:7-alpine",
        startup_timeout_s=1,
    )

    monkeypatch.setattr(local_server, "_container_running", lambda name: True)
    monkeypatch.setattr(
        local_server,
        "_inspect_container",
        lambda name: _container_config("example/mx:old", env=_server_env(), port=8123),
    )

    with pytest.raises(RuntimeError, match="does not match"):
        local_server._ensure_mx_server_container(spec)


def test_timeout_reports_server_logs(monkeypatch):
    monkeypatch.setattr(local_server, "_port_open", lambda host, port: False)
    monkeypatch.setattr(local_server, "_container_running", lambda name: None)
    monkeypatch.setattr(local_server, "_docker", lambda *args, **kwargs: _completed())
    monkeypatch.setattr(local_server, "_container_logs", lambda name: "server log line")

    with pytest.raises(RuntimeError, match="server log line"):
        local_server.ensure_local_mx_server(port=8123, startup_timeout_s=1)
