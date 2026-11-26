# Copyright (c) 2026, NVIDIA CORPORATION.
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
"""TLS configuration shared by the coordinator listener and fleet clients."""

import ipaddress
import os
import ssl
from dataclasses import dataclass
from typing import Optional, Union
from urllib.parse import urlsplit

from tensorrt_llm.serve.cluster_storage import is_loopback_host


@dataclass(frozen=True)
class CoordinatorTLS:
    ca: str
    cert: str
    key: str

    @classmethod
    def from_env(cls) -> Optional["CoordinatorTLS"]:
        names = (
            "TRTLLM_DISAGG_COORDINATOR_TLS_CA",
            "TRTLLM_DISAGG_COORDINATOR_TLS_CERT",
            "TRTLLM_DISAGG_COORDINATOR_TLS_KEY",
        )
        ca, cert, key = (os.environ.get(name) for name in names)
        if ca and cert and key:
            return cls(ca, cert, key)
        if ca or cert or key:
            raise ValueError("Set all three coordinator TLS variables: " + ", ".join(names))
        return None

    def server_options(self) -> dict:
        return dict(
            ssl_ca_certs=self.ca,
            ssl_certfile=self.cert,
            ssl_keyfile=self.key,
            ssl_cert_reqs=ssl.CERT_REQUIRED,
        )

    def client_context(self) -> ssl.SSLContext:
        context = ssl.create_default_context(cafile=self.ca)
        context.load_cert_chain(self.cert, self.key)
        return context


def is_wildcard_host(host: str) -> bool:
    if not host:
        return True
    try:
        return ipaddress.ip_address(host).is_unspecified
    except ValueError:
        return False


def coordinator_client_ssl(url: str) -> Union[bool, ssl.SSLContext]:
    endpoint = urlsplit(url)
    if endpoint.scheme == "http" and is_loopback_host(endpoint.hostname):
        return True
    if endpoint.scheme == "https":
        tls = CoordinatorTLS.from_env()
        if tls is not None:
            return tls.client_context()
    raise ValueError("A remote coordinator requires HTTPS and coordinator TLS credentials")
