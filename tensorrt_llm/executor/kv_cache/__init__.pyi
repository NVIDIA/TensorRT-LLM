

class KVCacheCreatedData:
    @property
    def num_blocks_per_cache_level(self) -> list[int]: ...

class UniqueToken:
    @property
    def token_id(self) -> int: ...

    @property
    def token_extra_id(self) -> int: ...

class KVCacheStoredBlockData:
    @property
    def block_hash(self) -> int: ...

    @property
    def tokens(self) -> list[UniqueToken]: ...

    @property
    def lora_id(self) -> int | None: ...

    @property
    def cache_level(self) -> int: ...

    @property
    def priority(self) -> int: ...

class KVCacheStoredData:
    @property
    def parent_hash(self) -> int | None: ...

    @property
    def blocks(self) -> list[KVCacheStoredBlockData]: ...

class KVCacheRemovedData:
    @property
    def block_hashes(self) -> list[int]: ...

class KVCacheEventDiffInt:
    @property
    def old_value(self) -> int: ...

    @property
    def new_value(self) -> int: ...

class KVCacheUpdatedData:
    @property
    def block_hash(self) -> int: ...

    @property
    def cache_level(self) -> KVCacheEventDiffInt | None: ...

    @property
    def priority(self) -> KVCacheEventDiffInt | None: ...

class KVCacheEvent:
    @property
    def event_id(self) -> int: ...

    @property
    def data(self) -> KVCacheCreatedData | KVCacheStoredData | KVCacheRemovedData | KVCacheUpdatedData: ...

    @property
    def window_size(self) -> int: ...

    @property
    def attention_dp_rank(self) -> int | None: ...

class KVCacheEventManager:
    def get_latest_events(self, timeout_ms: float | None = None) -> List: ...
