import json
from abc import ABC, abstractmethod

import etcd3


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

    def __init__(self, host: str, port: int):
        self._client = etcd3.client(host, port)

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


class JsonDictionary:

    def __init__(self, dict: RemoteDictionary):
        self._dict = dict

    def get(self, key: str) -> str:
        return json.loads(self._dict.get(key))

    def put(self, key: str, value: str):
        self._dict.put(key, json.dumps(value))

    def remove(self, key: str):
        self._dict.remove(key)

    def keys(self) -> list[str]:
        return self._dict.keys()
