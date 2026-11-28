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
"""The signing of the rendezvous store's values."""

import os
import re

import pytest
from _torch.distributed.store_group_test_utils import InMemoryStore
from torch.distributed import TCPStore

from tensorrt_llm._torch.distributed.membership import (
    SHARED_SECRET_ENV,
    SignedStore,
    UnsignedValueError,
)
from tensorrt_llm._torch.distributed.membership.signed_store import shared_secret_digest

pytestmark = pytest.mark.cpu_only

_SHORT_SECRET = f"shared secret of at least 32 bytes (64 hex digits) in {SHARED_SECRET_ENV}"


def test_store_client_requires_set_secret(monkeypatch):
    """Refuse a store client and a secret digest without a shared secret, naming the variable."""
    monkeypatch.delenv(SHARED_SECRET_ENV, raising=False)
    with pytest.raises(ValueError, match=re.escape(_SHORT_SECRET)):
        SignedStore(InMemoryStore())
    with pytest.raises(ValueError, match=re.escape(_SHORT_SECRET)):
        shared_secret_digest(b"label")


@pytest.mark.parametrize("secret", ["ab" * 31, "zz" * 32], ids=["short", "not_hex"])
def test_store_client_requires_ranks_secret(secret):
    """Refuse a store client, as of the node supervisor or the authority, with a weak secret."""
    with pytest.raises(ValueError, match=re.escape(_SHORT_SECRET)):
        SignedStore(InMemoryStore(), secret)


def _libuv_store():
    """Host a TCPStore on torch's default server backend, whose values are limited to 8 MiB."""
    return TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False, use_libuv=True)


def test_value_over_store_server_limit():
    """Set, get and delete values over the 8 MiB that torch's default TCPStore server accepts."""
    server = _libuv_store()
    keys = server.num_keys()
    signed = SignedStore(server, "ab" * 32)
    large = os.urandom(9 << 20)
    signed.set("large", large)
    signed.set("small", b"small")
    assert signed.get("large") == large
    assert signed.multi_get(["small", "large"]) == [b"small", large]
    assert signed.delete_key("large") and signed.delete_key("small")
    assert server.num_keys() == keys


def test_forged_chunks_rejected():
    """Reject a large value whose chunks another process set under the deployment's signed key."""
    server = _libuv_store()
    signed = SignedStore(server, "ab" * 32)
    signed.set("large", os.urandom(9 << 20))
    deployment_key = server.get("large")
    SignedStore(server, "cd" * 32).set("large", os.urandom(9 << 20))
    server.set("large", deployment_key)
    with pytest.raises(UnsignedValueError, match="do not match its signed digest"):
        signed.get("large")
