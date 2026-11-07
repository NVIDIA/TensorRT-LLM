# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""etcd credentials and TLS options of the disagg cluster configuration."""

import secrets

import pytest

from tensorrt_llm.llmapi.disagg_utils import DisaggClusterConfig, extract_disagg_cluster_config

pytestmark = pytest.mark.cpu_only

ETCD_USER = secrets.token_hex(4)
ETCD_SECRET = secrets.token_hex(16)
RECORD_KEY = secrets.token_hex(16)
ETCD_URI = "etcd://etcd-host:2379"
OPTIONS = {
    "user": ETCD_USER,
    "password": ETCD_SECRET,
    "ca_cert": "/ca.pem",
    "cert_key": "/key.pem",
    "cert_cert": "/cert.pem",
    "record_auth_key": RECORD_KEY,
}


def test_cluster_config_etcd_options():
    assert DisaggClusterConfig(ETCD_URI).etcd_client_options() == {}
    cfg = DisaggClusterConfig(ETCD_URI, **OPTIONS)
    assert cfg.etcd_client_options() == OPTIONS
    assert ETCD_SECRET not in repr(cfg) and RECORD_KEY not in repr(cfg)


def test_cluster_config_rejects_credentials_without_ca_cert():
    with pytest.raises(ValueError, match="require ca_cert") as excinfo:
        DisaggClusterConfig(ETCD_URI, user=ETCD_USER, password=ETCD_SECRET)
    assert ETCD_SECRET not in str(excinfo.value)


@pytest.mark.parametrize("options", [{"ca_cert": "/ca.pem"}, {"record_auth_key": RECORD_KEY}])
def test_cluster_config_etcd_options_require_an_etcd_uri(options):
    with pytest.raises(ValueError, match="require an etcd:// cluster_uri"):
        DisaggClusterConfig("http://host:8000", **options)


def test_extract_cluster_config_reads_the_options():
    # A worker takes the uri from --disagg_cluster_uri and the credentials from its config file.
    cfg = extract_disagg_cluster_config({"cluster_name": "c", **OPTIONS}, cluster_uri=ETCD_URI)
    assert cfg.cluster_uri == ETCD_URI and cfg.cluster_name == "c"
    assert cfg.etcd_client_options() == OPTIONS


def test_extract_cluster_config_validates_the_options_it_sets():
    with pytest.raises(ValueError, match="require ca_cert"):
        extract_disagg_cluster_config(
            {"cluster_uri": ETCD_URI, "user": ETCD_USER, "password": ETCD_SECRET}
        )
