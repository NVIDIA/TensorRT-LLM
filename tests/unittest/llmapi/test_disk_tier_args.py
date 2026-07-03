# tests/unittest/llmapi/test_disk_tier_args.py
"""Disk-tier plumbing unit tests.

pydantic->pybind passthrough, retention pickle, and the serve-layer TTL helper.
"""

import datetime
import pickle

from tensorrt_llm.bindings import executor as tllm


def test_kv_cache_config_disk_fields():
    cfg = tllm.KvCacheConfig()
    assert cfg.disk_cache_retained_only is False  # default preserves spill-all
    cfg.disk_cache_size = 1 << 30
    cfg.disk_cache_path = "/tmp/kvdisk-test"
    cfg.disk_cache_retained_only = True
    assert cfg.disk_cache_size == 1 << 30
    assert cfg.disk_cache_path == "/tmp/kvdisk-test"
    assert cfg.disk_cache_retained_only is True


def test_kv_cache_config_disk_fields_pickle():
    cfg = tllm.KvCacheConfig()
    cfg.disk_cache_size = 42
    cfg.disk_cache_path = "/x"
    cfg.disk_cache_retained_only = True
    back = pickle.loads(pickle.dumps(cfg))
    assert back.disk_cache_size == 42
    assert back.disk_cache_path == "/x"
    assert back.disk_cache_retained_only is True


def test_retention_config_disk_retention_ms_roundtrip():
    cfg = tllm.KvCacheRetentionConfig([])
    assert cfg.disk_retention_ms is None  # unset = never enters a retained-only tier
    cfg.disk_retention_ms = datetime.timedelta(seconds=3600)
    assert cfg.disk_retention_ms == datetime.timedelta(seconds=3600)
    back = pickle.loads(pickle.dumps(cfg))
    assert back.disk_retention_ms == datetime.timedelta(seconds=3600)


def test_llm_args_kv_cache_config_disk_fields():
    import os

    os.makedirs("/tmp/kvdisk-test", exist_ok=True)  # pydantic validator requires an existing dir
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig as PydanticKvCacheConfig

    py_cfg = PydanticKvCacheConfig(
        disk_cache_size=1 << 30, disk_cache_path="/tmp/kvdisk-test", disk_cache_retained_only=True
    )
    assert py_cfg.disk_cache_retained_only is True
    cpp_cfg = py_cfg._to_pybind()
    assert cpp_cfg.disk_cache_size == 1 << 30
    assert cpp_cfg.disk_cache_path == "/tmp/kvdisk-test"
    assert cpp_cfg.disk_cache_retained_only is True


def test_serve_disk_retention_helper():
    from tensorrt_llm.serve.openai_server import _disk_retention_config

    class Req:
        kv_cache_ttl_seconds = None

    assert _disk_retention_config(Req()) is None

    Req.kv_cache_ttl_seconds = 0
    assert _disk_retention_config(Req()) is None  # ttl=0 behaves non-retained

    Req.kv_cache_ttl_seconds = 3600
    cfg = _disk_retention_config(Req())
    assert cfg is not None
    assert cfg.disk_retention_ms == datetime.timedelta(seconds=3600)

    class Bare:  # object without the field at all (non-serve request types)
        pass

    assert _disk_retention_config(Bare()) is None
