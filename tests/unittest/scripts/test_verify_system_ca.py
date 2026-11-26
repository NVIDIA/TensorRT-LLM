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

"""CPU check for the release image's default trust-store verification."""

import hashlib
import importlib.util
import ssl
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_PATH = Path(__file__).resolve().parents[3] / "docker/common/verify_system_ca.py"
_SPEC = importlib.util.spec_from_file_location("verify_system_ca", _PATH)
verifier = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verifier)


def openssl(*args: str) -> str:
    return subprocess.run(["openssl", *args], check=True, capture_output=True, text=True).stdout


@pytest.mark.parametrize("location", ["default store", "hashed directory"])
def test_rejects_distrusted_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, location: str):
    cert, store = tmp_path / "ca.crt", tmp_path / "certs"
    store.mkdir()
    openssl("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(tmp_path / "ca.key"),
            "-out", str(cert), "-days", "1", "-subj", "/CN=test",
            "-addext", "basicConstraints=critical,CA:TRUE")  # fmt: skip
    der = ssl.PEM_cert_to_DER_cert(cert.read_text())
    monkeypatch.setattr(verifier, "_DISTRUSTED_ROOTS", {hashlib.sha256(der).hexdigest()})
    monkeypatch.setenv("SSL_CERT_DIR", str(store))
    default = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    if location == "default store":
        default.load_verify_locations(cafile=str(cert))
    else:
        # OpenSSL loads hashed-directory roots lazily, so get_ca_certs() misses them.
        name = openssl("x509", "-in", str(cert), "-noout", "-hash").strip()
        (store / f"{name}.0").symlink_to(cert)
    monkeypatch.setattr(verifier.ssl, "create_default_context", lambda: default)
    with pytest.raises(RuntimeError, match="Distrusted CA"):
        verifier.verify_system_ca()
