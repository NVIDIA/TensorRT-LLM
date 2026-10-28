# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import hmac
import json
from abc import ABC, abstractmethod
from typing import Any, Optional

from tensorrt_llm.llmapi.disagg_utils import MetadataServerConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.cluster_storage import etcd3

RECORD_SIGNATURE_FIELD = "signature"
_SIGNATURE_PREFIX = "sha256="


class RemoteDictionary(ABC):

    @abstractmethod
    def get(self, key: str) -> str:
        pass

    @abstractmethod
    def put(self, key: str, value: str):
        pass

    @abstractmethod
    def remove(self, key: str):
        pass

    @abstractmethod
    def keys(self) -> list[str]:
        pass


class EtcdDictionary(RemoteDictionary):

    def __init__(self,
                 host: str,
                 port: int,
                 user: Optional[str] = None,
                 password: Optional[str] = None,
                 ca_cert: Optional[str] = None,
                 cert_key: Optional[str] = None,
                 cert_cert: Optional[str] = None):
        self._client = etcd3.client(host,
                                    port,
                                    ca_cert=ca_cert,
                                    cert_key=cert_key,
                                    cert_cert=cert_cert,
                                    user=user,
                                    password=password)

    def get(self, key: str) -> str:
        return self._client.get(key)

    def put(self, key: str, value: str):
        self._client.put(key, value)

    def remove(self, key: str):
        self._client.delete(key)

    def keys(self) -> list[str]:
        # TODO: Confirm the final save key format
        # This implementation assumes that key is in the
        # format of "trtllm/executor_name/key"
        unique_keys = set()
        for _, metadata in self._client.get_all():
            key = metadata.key.decode('utf-8')
            sub_keys = key.split('/')
            if len(sub_keys) >= 2:
                top_prefix = "/".join(sub_keys[:2])
                unique_keys.add(top_prefix)
        return list(unique_keys)


def _sign_record(record_auth_key: str, key: str, value: dict) -> str:
    unsigned = {k: v for k, v in value.items() if k != RECORD_SIGNATURE_FIELD}
    payload = json.dumps({
        "key": key,
        "value": unsigned
    },
                         sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    digest = hmac.new(record_auth_key.encode("utf-8"), payload,
                      hashlib.sha256).hexdigest()
    return f"{_SIGNATURE_PREFIX}{digest}"


class JsonDictionary:
    """JSON view of a RemoteDictionary.

    With ``record_auth_key`` set, every dict value written is signed with an
    HMAC bound to its key, and ``get`` returns None for a value whose signature
    is missing or does not match.
    """

    def __init__(self,
                 dict: RemoteDictionary,
                 record_auth_key: Optional[str] = None):
        self._dict = dict
        self._record_auth_key = record_auth_key

    def get(self, key: str) -> Any:
        bytes_data, _ = self._dict.get(key)
        if bytes_data is None:
            return None
        value = json.loads(bytes_data.decode('utf-8'))
        if self._record_auth_key is None:
            return value
        if not isinstance(value, dict):
            logger.warning(f"Ignoring unsigned metadata record {key}")
            return None
        provided = value.get(RECORD_SIGNATURE_FIELD)
        expected = _sign_record(self._record_auth_key, key, value)
        if not isinstance(provided, str) or not hmac.compare_digest(
                provided.encode("utf-8"), expected.encode("utf-8")):
            logger.warning(
                f"Ignoring metadata record {key} with an invalid signature")
            return None
        return {k: v for k, v in value.items() if k != RECORD_SIGNATURE_FIELD}

    def put(self, key: str, value: Any):
        if self._record_auth_key is not None:
            if not isinstance(value, dict):
                raise ValueError("Signed metadata records must be dicts")
            value = dict(value)
            value[RECORD_SIGNATURE_FIELD] = _sign_record(
                self._record_auth_key, key, value)
        self._dict.put(key, json.dumps(value))

    def remove(self, key: str):
        self._dict.remove(key)

    def keys(self) -> list[str]:
        return self._dict.keys()


def create_metadata_server(
    metadata_server_cfg: Optional[MetadataServerConfig]
) -> Optional[JsonDictionary]:

    if metadata_server_cfg is None:
        return None

    if metadata_server_cfg.server_type == 'etcd':
        dict = EtcdDictionary(host=metadata_server_cfg.hostname,
                              port=metadata_server_cfg.port,
                              user=metadata_server_cfg.user,
                              password=metadata_server_cfg.password,
                              ca_cert=metadata_server_cfg.ca_cert,
                              cert_key=metadata_server_cfg.cert_key,
                              cert_cert=metadata_server_cfg.cert_cert)
    else:
        raise ValueError(
            f"Unsupported metadata server type: {metadata_server_cfg.server_type}"
        )

    if metadata_server_cfg.record_auth_key is None:
        logger.warning(
            "Metadata server record_auth_key is not set; server records are "
            "not signed or verified.")
    return JsonDictionary(dict, metadata_server_cfg.record_auth_key)
