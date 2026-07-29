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

import pytest

from tensorrt_llm._torch.disaggregation.native.peer_allowlist import (
    ALLOWED_PEER_HOSTS_ENV,
    PeerEndpointAllowlist,
    PeerEndpointNotAllowedError,
    check_peer_endpoint,
    get_peer_endpoint_allowlist,
    reset_peer_endpoint_allowlist,
)

ATTACKER = "tcp://10.13.37.99:31337"
PEER = "tcp://10.0.3.12:29010"


@pytest.fixture(autouse=True)
def _reset_allowlist():
    reset_peer_endpoint_allowlist()
    yield
    reset_peer_endpoint_allowlist()


def test_unconfigured_allowlist_allows_everything():
    allowlist = PeerEndpointAllowlist([])

    assert not allowlist.enabled
    assert allowlist.allows(ATTACKER)
    allowlist.check(ATTACKER)


def test_configured_allowlist_rejects_unknown_host():
    allowlist = PeerEndpointAllowlist(["10.0.3.12"])

    assert allowlist.enabled
    assert allowlist.allows(PEER)
    assert not allowlist.allows(ATTACKER)
    with pytest.raises(PeerEndpointNotAllowedError, match="Refusing to connect"):
        allowlist.check(ATTACKER)


def test_port_is_not_part_of_the_match():
    allowlist = PeerEndpointAllowlist(["10.0.3.12"])

    assert allowlist.allows("tcp://10.0.3.12:1")
    assert allowlist.allows("tcp://10.0.3.12:65535")


def test_cidr_entry_matches_contained_addresses():
    allowlist = PeerEndpointAllowlist(["10.0.3.0/24"])

    assert allowlist.allows("tcp://10.0.3.1:29010")
    assert allowlist.allows("tcp://10.0.3.254:29010")
    assert not allowlist.allows("tcp://10.0.4.1:29010")


def test_loopback_and_cloud_metadata_are_rejected():
    allowlist = PeerEndpointAllowlist(["10.0.3.0/24"])

    for endpoint in (
        "tcp://127.0.0.1:22",
        "tcp://127.0.0.1:8080",
        "tcp://169.254.169.254:80",
        "tcp://172.17.0.1:2375",
    ):
        assert not allowlist.allows(endpoint), endpoint


def test_ipv6_entry_and_endpoint():
    allowlist = PeerEndpointAllowlist(["fd00::/8"])

    assert allowlist.allows("tcp://[fd00::1]:29010")
    assert not allowlist.allows("tcp://[fe80::1]:29010")


def test_hostname_entry_matches_literal_host():
    allowlist = PeerEndpointAllowlist(["ctx-worker-0"])

    assert allowlist.allows("tcp://ctx-worker-0:29010")
    assert allowlist.allows("tcp://CTX-Worker-0:29010")
    assert not allowlist.allows("tcp://ctx-worker-1:29010")


def test_localhost_entry_resolves_to_its_addresses():
    allowlist = PeerEndpointAllowlist(["localhost"])

    assert allowlist.allows("tcp://127.0.0.1:29010")


def test_non_tcp_transports_are_rejected():
    allowlist = PeerEndpointAllowlist(["10.0.3.12"])

    for endpoint in (
        "ipc:///var/run/docker.sock",
        "inproc://kv",
        "pgm://10.13.37.99:9999",
        "udp://10.13.37.99:9999",
    ):
        assert not allowlist.allows(endpoint), endpoint


def test_malformed_and_empty_endpoints_are_rejected_when_enabled():
    allowlist = PeerEndpointAllowlist(["10.0.3.12"])

    for endpoint in (None, "", "tcp://", "tcp://:29010", "not-an-endpoint"):
        assert not allowlist.allows(endpoint), endpoint


def test_blank_entries_do_not_enable_the_allowlist():
    assert not PeerEndpointAllowlist(["", "  ", "\t"]).enabled


def test_unparsable_entry_raises_instead_of_failing_open():
    for entries in (["10.0.3.12/0"], ["not a host!"], ["10.0.3.12", "bogus!!"]):
        with pytest.raises(ValueError, match=ALLOWED_PEER_HOSTS_ENV):
            PeerEndpointAllowlist(entries)


def test_mistyped_prefix_is_not_widened_to_everything():
    with pytest.raises(ValueError):
        PeerEndpointAllowlist(["10.0.3.12/0"])


def test_trailing_dot_is_normalized_on_both_sides():
    assert PeerEndpointAllowlist(["10.0.3.12."]).allows("tcp://10.0.3.12:29010")
    assert PeerEndpointAllowlist(["10.0.3.12"]).allows("tcp://10.0.3.12.:29010")
    assert PeerEndpointAllowlist(["ctx-worker-0"]).allows("tcp://ctx-worker-0.:29010")


def test_underscore_hostname_is_accepted():
    allowlist = PeerEndpointAllowlist(["kv_svc.ns"])

    assert allowlist.enabled
    assert allowlist.allows("tcp://kv_svc.ns:29010")


def test_ipv4_mapped_ipv6_is_unwrapped_before_matching():
    allowlist = PeerEndpointAllowlist(["10.0.3.0/24"])

    assert allowlist.allows("tcp://[::ffff:10.0.3.12]:29010")
    assert not allowlist.allows("tcp://[::ffff:127.0.0.1]:22")


def test_userinfo_cannot_spoof_an_allowed_host():
    allowlist = PeerEndpointAllowlist(["10.0.3.12"])

    assert not allowlist.allows("tcp://10.0.3.12@10.13.37.99:31337")
    assert not allowlist.allows("tcp://user:10.0.3.12@10.13.37.99:31337")


def test_obfuscated_ipv4_forms_do_not_match():
    allowlist = PeerEndpointAllowlist(["127.0.0.0/8"])

    for endpoint in ("tcp://0177.0.0.1:22", "tcp://2130706433:22", "tcp://0x7f.0.0.1:22"):
        assert not allowlist.allows(endpoint), endpoint


def test_uppercase_scheme_is_rejected():
    assert not PeerEndpointAllowlist(["10.0.3.12"]).allows("TCP://10.13.37.99:31337")


def test_zmq_source_destination_syntax_cannot_smuggle_a_host():
    # libzmq dials the half after ";"; a source-only parse would allow these.
    allowlist = PeerEndpointAllowlist(["10.0.3.0/24", "127.0.0.0/8"])

    for endpoint in (
        "tcp://10.0.3.12:0;10.13.37.99:31337",
        "tcp://127.0.0.1:0;10.13.37.99:31337",
        "tcp://127.0.0.1:0;evil.example.com:80",
        "tcp://0.0.0.0:0;10.13.37.99:31337",
        "tcp://lo:0;10.13.37.99:31337",
        "tcp://[::1]:0;10.13.37.99:31337",
    ):
        assert not allowlist.allows(endpoint), endpoint


def test_only_a_bare_tcp_host_port_endpoint_is_accepted():
    allowlist = PeerEndpointAllowlist(["10.0.3.12"])

    assert allowlist.allows("tcp://10.0.3.12:29010")
    assert allowlist.allows("tcp://10.0.3.12:*")
    for endpoint in (
        "tcp://10.0.3.12:29010/path",
        "tcp://10.0.3.12:29010?x=1",
        "tcp://10.0.3.12:29010#frag",
        "tcp://10.0.3.12:29010 ",
        "tcp://10.0.3.12:29010\n",
        "tcp://10.0.3.12%00.evil.com:9999",
        "tcp://10.0.3.12:999999",
        "tcp://10.0.3.12",
    ):
        assert not allowlist.allows(endpoint), repr(endpoint)


def test_bad_configuration_is_cached_and_not_reparsed(monkeypatch):
    monkeypatch.setenv(ALLOWED_PEER_HOSTS_ENV, "10.0.3.12, bogus!!")

    with pytest.raises(ValueError, match=ALLOWED_PEER_HOSTS_ENV) as first:
        get_peer_endpoint_allowlist()
    with pytest.raises(ValueError) as second:
        check_peer_endpoint(PEER)

    assert first.value is second.value


def test_env_var_drives_the_process_wide_allowlist(monkeypatch):
    monkeypatch.setenv(ALLOWED_PEER_HOSTS_ENV, "10.0.3.12, 10.0.4.0/24")

    assert get_peer_endpoint_allowlist().enabled
    check_peer_endpoint(PEER)
    check_peer_endpoint("tcp://10.0.4.7:29010")
    with pytest.raises(PeerEndpointNotAllowedError):
        check_peer_endpoint(ATTACKER)


def test_check_peer_endpoint_is_a_no_op_without_env_var(monkeypatch):
    monkeypatch.delenv(ALLOWED_PEER_HOSTS_ENV, raising=False)

    assert not get_peer_endpoint_allowlist().enabled
    check_peer_endpoint(ATTACKER)


def test_rejection_is_a_value_error_subclass():
    assert issubclass(PeerEndpointNotAllowedError, ValueError)


try:
    from tensorrt_llm._torch.disaggregation.native import transfer
except ImportError:
    transfer = None

requires_transfer = pytest.mark.skipif(
    transfer is None, reason="requires built tensorrt_llm bindings"
)


class _StubMessenger:
    def __init__(self, mode, endpoint=None):
        self.mode = mode
        self.endpoint = endpoint

    def send(self, _):
        raise AssertionError("peer should not be contacted in this test")


@pytest.fixture
def stub_messenger(monkeypatch):
    monkeypatch.setattr(transfer, "ZMQMessenger", _StubMessenger)


@pytest.fixture
def allowlisted_peer(monkeypatch):
    monkeypatch.setenv(ALLOWED_PEER_HOSTS_ENV, "10.0.3.12")


def _bare(cls):
    return object.__new__(cls)


@requires_transfer
def test_receiver_dealer_refuses_endpoint_outside_allowlist(allowlisted_peer):
    receiver = _bare(transfer.Receiver)
    receiver._dealers = {}

    with pytest.raises(PeerEndpointNotAllowedError):
        receiver._get_or_connect_dealer(ATTACKER)
    assert receiver._dealers == {}


@requires_transfer
def test_receiver_dealer_connects_to_allowlisted_peer(allowlisted_peer, stub_messenger):
    receiver = _bare(transfer.Receiver)
    receiver._dealers = {}

    assert receiver._get_or_connect_dealer(PEER).endpoint == PEER


@requires_transfer
def test_sender_dealer_refuses_endpoint_outside_allowlist(allowlisted_peer):
    sender = _bare(transfer.Sender)
    sender._dealers = {}

    with pytest.raises(PeerEndpointNotAllowedError):
        sender._get_or_connect_dealer(ATTACKER)
    assert sender._dealers == {}


@requires_transfer
def test_sender_thread_dealer_refuses_endpoint_outside_allowlist(allowlisted_peer):
    import threading

    sender = _bare(transfer.Sender)
    sender._thread_local = threading.local()

    with pytest.raises(PeerEndpointNotAllowedError):
        sender._get_or_connect_thread_dealer(ATTACKER)


@requires_transfer
def test_sender_info_refuses_request_supplied_endpoint(allowlisted_peer):
    from tensorrt_llm.disaggregated_params import DisaggregatedParams

    receiver = _bare(transfer.Receiver)
    receiver._sender_ep_instance_map = {}

    with pytest.raises(PeerEndpointNotAllowedError):
        receiver._get_sender_info(DisaggregatedParams(ctx_info_endpoint=ATTACKER))


@requires_transfer
def test_hooks_are_inert_without_allowlist(monkeypatch, stub_messenger):
    monkeypatch.delenv(ALLOWED_PEER_HOSTS_ENV, raising=False)
    receiver = _bare(transfer.Receiver)
    receiver._dealers = {}

    assert receiver._get_or_connect_dealer(ATTACKER).endpoint == ATTACKER
