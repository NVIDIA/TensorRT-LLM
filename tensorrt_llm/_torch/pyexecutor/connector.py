from typing import Optional

from tensorrt_llm.bindings.internal.batch_manager import \
    KvCacheConnector as KvCacheConnectorCpp
from tensorrt_llm.bindings.internal.batch_manager import KvCacheConnectorRole


class KvCacheConnector(KvCacheConnectorCpp):

    def __init__(self, role: KvCacheConnectorRole):
        super().__init__(role)
        self.connector_metadata = None

    def bind_connector_metadata(self, metadata: object):
        self.connector_metadata = metadata

    def _get_connector_metadata(self) -> object:
        return self.connector_metadata

    def build_connector_metadata(self) -> Optional[object]:
        return None
