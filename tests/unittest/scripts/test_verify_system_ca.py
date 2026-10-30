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

"""CPU checks for the release image's default trust-store verification."""

import hashlib
import importlib.util
import ssl
import subprocess
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.cpu_only

_PATH = Path(__file__).resolve().parents[3] / "docker/common/verify_system_ca.py"
_SPEC = importlib.util.spec_from_file_location("verify_system_ca", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
verifier = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verifier)


@pytest.fixture(scope="module")
def certificates(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("system-ca")
    paths = []
    for name in ("distrusted", "custom"):
        certificate = root / f"{name}.crt"
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-keyout",
                str(root / f"{name}.key"),
                "-out",
                str(certificate),
                "-days",
                "1",
                "-subj",
                f"/CN=test-{name}",
                "-addext",
                "basicConstraints=critical,CA:TRUE",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        paths.append(certificate)
    return paths[0], paths[1]


def context(certificate: Path) -> ssl.SSLContext:
    result = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    result.load_verify_locations(cafile=str(certificate))
    return result


@pytest.fixture
def trust_store(
    certificates: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Path, Path, Path]]:
    distrusted, custom = certificates
    fingerprint = hashlib.sha256(ssl.PEM_cert_to_DER_cert(distrusted.read_text())).hexdigest()
    monkeypatch.setattr(verifier, "_DISTRUSTED_ROOTS", frozenset({fingerprint}))
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path))
    default = context(custom)
    default.load_verify_locations(capath=str(tmp_path))
    with patch.object(verifier.ssl, "create_default_context", return_value=default):
        yield distrusted, custom, tmp_path


def hashed_name(certificate: Path) -> str:
    value = subprocess.check_output(
        ["openssl", "x509", "-in", str(certificate), "-noout", "-hash"], text=True
    ).strip()
    return f"{value}.0"


@pytest.mark.parametrize("location", ["default store", "hashed directory"])
def test_rejects_distrusted_root(trust_store: tuple[Path, Path, Path], location: str) -> None:
    distrusted, _, directory = trust_store
    if location == "default store":
        with patch.object(verifier.ssl, "create_default_context", return_value=context(distrusted)):
            with pytest.raises(RuntimeError, match="default CA store"):
                verifier.verify_system_ca()
    else:
        # OpenSSL loads hashed-directory roots lazily, so get_ca_certs() misses them.
        (directory / hashed_name(distrusted)).symlink_to(distrusted)
        with pytest.raises(RuntimeError, match="Distrusted CA"):
            verifier.verify_system_ca()
