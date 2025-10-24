from enum import Enum


class MetricNames(Enum):
    TTFT = "ttft"
    TPOT = "tpot"
    E2E = "e2e"
    REQUEST_QUEUE_TIME = "request_queue_time"
    REQUEST_METADATA_TIME = "request_metadata_time"
    KV_CACHE_KERNEL_TIME = "kv_cache_kernel_time"
    KV_CACHE_TRANSFER_TIME = "kv_cache_transfer_time"


class RequestEventTiming(Enum):
    ARRIVAL_TIME = "arrival_time"
    FIRST_TOKEN_TIME = "first_token_time"  # nosec: B105
    FIRST_SCHEDULED_TIME = "first_scheduled_time"
    LAST_TOKEN_TIME = "last_token_time"  # nosec: B105
    REQUEST_INFO_TIME = "request_info_time"
    KV_CACHE_KERNEL_END_TIME = "kv_cache_kernel_end_time"
    KV_CACHE_TRANSFER_START_TIME = "kv_cache_transfer_start_time"
    KV_CACHE_TRANSFER_END_TIME = "kv_cache_transfer_end_time"
