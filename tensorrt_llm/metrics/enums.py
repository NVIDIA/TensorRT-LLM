from enum import Enum


class MetricNames(Enum):
    TTFT = "ttft"
    TPOT = "tpot"
    E2E = "e2e"
    REQUEST_QUEUE_TIME = "request_queue_time"
    ARRIVAL_TIMESTAMP = 'arrival_timestamp'


class RequestEventTiming(Enum):
    ARRIVAL_TIME = "arrival_time"
    FIRST_TOKEN_TIME = "first_token_time"  # nosec: B105
    FIRST_SCHEDULED_TIME = "first_scheduled_time"
    LAST_TOKEN_TIME = "last_token_time"  # nosec: B105
    KV_CACHE_TRANSFER_START = "kv_cache_transfer_start"
    KV_CACHE_TRANSFER_END = "kv_cache_transfer_end"
    KV_CACHE_SIZE = "kv_cache_size"
