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

"""Verify that the default Python/OpenSSL stores exclude revoked CA roots."""

import hashlib
import os
import re
import ssl
from pathlib import Path

# DER fingerprint of emSign ECC Root CA - G3.
_REVOKED_ROOTS = frozenset(
    {
        "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b",
    }
)
_HASHED_CERTIFICATE = re.compile(r"[0-9a-f]{8}\.[0-9]+")


def _verify_context(context: ssl.SSLContext, label: str) -> None:
    for certificate in context.get_ca_certs(binary_form=True):
        fingerprint = hashlib.sha256(certificate).hexdigest()
        if fingerprint in _REVOKED_ROOTS:
            raise RuntimeError(f"Revoked CA in {label}: {fingerprint}")


def verify_system_ca() -> None:
    """Check loaded roots and the hashed directories OpenSSL loads lazily."""
    _verify_context(ssl.create_default_context(), "the default CA store")
    paths = ssl.get_default_verify_paths()
    directories = os.environ.get(paths.openssl_capath_env, paths.openssl_capath) or ""
    for directory in sorted({Path(item) for item in directories.split(os.pathsep) if item}):
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not _HASHED_CERTIFICATE.fullmatch(path.name) or not path.is_file():
                continue
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.load_verify_locations(cafile=str(path))
            _verify_context(context, str(path))


if __name__ == "__main__":
    verify_system_ca()
